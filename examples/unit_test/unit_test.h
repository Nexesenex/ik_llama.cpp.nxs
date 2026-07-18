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

#include <functional>
#include <string>
#include <vector>

namespace bench {

// Adaptive timer shared with the precision suites: warmup (caches, frequency
// ramp) then geometrically growing reps until target_ms total or cap.
// Returns ms/call, reps via reps_out.
double timed_ms(const std::function<void()> & fn, int & reps_out,
                double target_ms = 200.0, int cap = 100000);

// Compile-time SIMD path label (AVX2 / VNNI256 / VNNI-INT8 / IFMA / FANCY).
const char * simd_name();

struct Options {
    std::vector<std::string> filters; // --bench-filter substrings (op or quant); empty = all
    int fixed_reps = 0;               // --bench-reps N: exact reps instead of adaptive
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

// Bench the full registry (honoring filters). Prints [BENCH] lines, returns
// collected rows; the caller appends to the DB when save_label is set.
int run(const Options & opt, std::vector<Result> & rows_out);

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
