//
// unit_test.h - shared declarations for the standalone IQK unit_test.
//
// bench:: — speed harness: times every quant of the IQK sub-backend (full
// registry auto-discovered from ggml_type_size/blck_size, so new quants are
// covered without code changes) across the inference hot path (convert,
// mul_mat, mul_mat_moe, fused, repack), plus an editable flat-file DB with
// before/after delta and CSV export for tracking commits / compile paths.
//
// Metric contract (see bench.cpp): MElem/s is always over WEIGHT elements
// (n*nrc_x) and is the exact cross-commit comparator; GFLOPS uses a documented
// nominal flop model per op and must never be compared across op kinds. No
// MIPS: integer/float instruction mix varies per kernel, MElem/s is exact.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace bench {

// Adaptive timer shared with the precision suites: time-based per-stage
// warmup (warmup_ms, caches/frequency/AVX license) plus a ~1s global IQ4_XS
// convert burst before stage 1 (skipped by no_spin), then geometrically
// growing reps until target_ms total or cap. Returns ms/call, reps via
// reps_out. Cap raised to 1M: sub-us stages (e.g. 16x1 direct GEMM at
// ~350 ns) otherwise stop at ~35 ms total, far below target, with higher
// variance. Defaults (100/50) are the shortest windows that held A/A
// agreement; lengthen via Options for noisier boxes, never shorten blind.
double timed_ms(const std::function<void()> & fn, int & reps_out,
                double target_ms = 100.0, int cap = 1000000, double warmup_floor_ms = 50.0);

// Compile-time SIMD path label (AVX2 / VNNI256 / VNNI-INT8 / IFMA / FANCY).
const char * simd_name();

struct Options {
    std::vector<std::string> filters; // --bench-filter substrings (op or quant); empty = all
    int fixed_reps = 0;               // --bench-reps N: exact reps instead of adaptive
    int threads = 1;                  // --bench-threads N: threads for mul/moe/fused entries
    int pin_cpu = -1;                 // --bench-pin N: pin bench thread to CPU N; -1 = no pin
    uint64_t pin_mask = 0;            // sweep-computed affinity mask (see --bench-pcores)
    bool use_mask = false;            // true: apply pin_mask instead of pin_cpu
    std::vector<int> pcores;          // --bench-pcores topology list (also rebuilt for pingpong children)
    double target_ms = 100.0;         // --bench-target-ms: adaptive window per stage
    double warmup_ms = 50.0;          // --bench-warmup-ms: per-stage warmup floor
    bool no_spin = false;             // --bench-no-spin: skip the ~1s global warmup burst
    std::string db_path = "iqk_bench.db";
    std::string save_label;           // --bench-save LABEL: append run to DB
};

struct Result {
    std::string op;      // convert | mul | moe | fused | repack
    std::string quant;   // ggml type name, e.g. iq4_xs_r8
    std::string path;    // convert | direct | - (n/a)
    int n = 0, nrc_x = 0, nrc_y = 0; // nrc_y = -1 when n/a
    std::string btype;   // activation type used, or -
    int reps = 0;
    double ms = 0.0, melem_s = 0.0, gflops = 0.0;
    std::string note;
};

// Bench the full registry (honoring filters). Prints [BENCH] lines with exact
// [i/N] progress (see bench plan line), returns collected rows; the caller
// appends to the DB when save_label is set.
int run(const Options & opt, std::vector<Result> & rows_out);

// Dual-branch pingpong: each validated family runs here (self, in-process)
// then in the other branch (child process = exe path argument, never a
// hardcoded name), back-to-back with cooldowns. Live [PING] pair table per
// family, DB labels label_a/label_b, official print_delta at the end.
// Scalar threads only (combine with --bench-t8421 by running it per branch).
int run_pingpong(const Options & opt, const std::string & exe_b,
                 const std::string & label_a, const std::string & label_b);

// Flat-file DB: '|' separated, '#' comments, editable in any text editor.
// Fields (15): date|label|simd|op|quant|path|n|nrc_x|nrc_y|btype|reps|ms|melem_s|gflops|note
int db_append(const std::string & db_path, const std::string & label,
              const std::vector<Result> & rows);
// Print before/after table for labels A/B with % deltas (ms based).
int print_delta(const std::string & db_path, const std::string & a, const std::string & b);
// Export label (empty = all labels) to CSV with header row.
int export_csv(const std::string & db_path, const std::string & csv_path,
               const std::string & label);

} // namespace bench
