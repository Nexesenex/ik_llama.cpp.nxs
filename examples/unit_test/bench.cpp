//
// bench.cpp - IQK speed harness (see unit_test.h for the metric contract).
//
// Coverage: full quant registry auto-discovered from ggml_type_size /
// ggml_blck_size (new quants appear with zero bench code). Stages:
// convert (iqk_convert_repack dispatcher), mul / moe / fused (public IQK
// entries, B-type auto-probed), repack (families with harness wrappers).
// Unsupported combos print [SKIP], never abort: every probed entry returns
// bool. quantize/dequantize rows are NOT benched here: most
// quantize_row_* are empty link stubs in this harness, so timing them would
// print fiction (growth path: link real ggml-quants.c, then wire per family).

#include "unit_test.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "iqk/iqk_common.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_mul_mat.h"
#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#ifdef TEST_DISPATCH
extern "C" bool iqk_convert_repack_for_test(int typeA, int n, const void * vx, size_t bx,
                                            void * vy, size_t stride_y, int nrc_x);
extern "C" int iqk_dequant_type(int type, int Ny);
#endif

// Test-only repack hooks (real code in iqk_quantize.cpp).
extern "C" void iqk_test_repack_q5_0(int nrows, int n_per_row, const block_q5_0 * x, block_q5_0_r4 * y);
extern "C" void iqk_test_repack_q4_0(int nrows, int n_per_row, const block_q4_0 * x, block_iq4_nl_r8 * y);
extern "C" void iqk_test_repack_mxfp4(int nrows, int n_per_row, const block_mxfp4 * x, block_mxfp4_r8 * y);
extern "C" void iqk_test_repack_iq4_nl(int nrows, int n_per_row, const block_iq4_nl * x, block_iq4_nl_r4 * y);
extern "C" void iqk_test_repack_q6_0(int nrows, int n_per_row, const block_q6_0 * x, block_q6_0_r4 * y);
extern "C" void iqk_test_repack_q8_0(int nrows, int n_per_row, const block_q8_0 * x, block_q8_0_r8 * y);
extern "C" void iqk_test_repack_iq4_xs(int nrows, int n_per_row, const block_iq4_xs * x, block_iq4_xs_r8 * y);

namespace bench {

const char * simd_name() {
    // Composite build-config label: base ISA plus every extra ISA flag the
    // build carries, so two builds differing ONLY by one of these flags never
    // share a DB simd label (silent miscomparison across compile paths).
    // Caveat, verified in-tree: no IQK kernel line consumes HAVE_VNNIINT8 /
    // HAVE_NECONVERT / HAVE_IFMA256 yet (the dpbssd aliases in iqk_config.h
    // are unused; only ggml-quants.c keys on __AVXVNNIINT8__ and that TU is
    // not linked here) — so VNNI256-vs-VNNI-INT8 deltas on IQK stages today
    // measure identical machine code plus general codegen effects of the
    // -mavxvnniint8 flag. XADD/CMPCCXADD is a barrier/threadpool property,
    // invisible to this single-thread bench by design (see --bench-pin note
    // for the future multithreaded axis).
    static char buf[64];
    const char * base =
#if defined(HAVE_FANCY_SIMD)
        "FANCY512";
#elif defined(HAVE_VNNIINT8) || defined(__AVXVNNIINT8__)
        "VNNI-INT8";
#elif defined(HAVE_VNNI256)
        "VNNI256";
#else
        "FALLBACK";
#endif
    std::string s = base;
#ifdef HAVE_IFMA256
    s += "+ifma";
#endif
#ifdef HAVE_NECONVERT
    s += "+necv";
#endif
    snprintf(buf, sizeof(buf), "%s", s.c_str());
    return buf;
}

// Per-stage warmup by TIME, not call count: 3 calls cannot lift core turbo
// or the AVX license on modern cores (measured A/A drift without this:
// mean -23% run to run, up to -57% on sub-us stages). Spin until min_ms
// elapses; min_calls keeps coverage for slow callables. Safety cap guards
// pathologically fast callables.
static void warmup_ms(const std::function<void()> & fn, double min_ms = 50.0, int min_calls = 3) {
    int calls = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        fn();
        if (++calls >= min_calls &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= min_ms) break;
        if (calls >= 2000000) break;
    }
}

double timed_ms(const std::function<void()> & fn, int & reps_out,
                double target_ms, int cap, double warmup_floor_ms) {
    warmup_ms(fn, warmup_floor_ms);
    int reps = 1;
    double ms = 0.0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        if (ms * reps >= target_ms || reps >= cap) break;
        reps *= 10;
    }
    reps_out = reps;
    return ms;
}

// Deterministic filler: stable bytes across binaries/commits (same policy as
// the per-suite reseeds in the precision path). Content is irrelevant to
// speed (fixed-work kernels), validity of layout is what matters.
static uint64_t xs_state = 0;
static void xs_seed(uint64_t s) { xs_state = s ? s : 0x9E3779B97F4A7C15ull; }
static uint64_t xs_next() {
    xs_state ^= xs_state << 13; xs_state ^= xs_state >> 7; xs_state ^= xs_state << 17;
    return xs_state;
}
static void xs_fill(void * p, size_t n) {
    uint8_t * b = (uint8_t *)p;
    for (size_t i = 0; i < n; ++i) b[i] = (uint8_t)(xs_next() >> (8 * (i % 8)));
}

// Valid activations: quantize deterministic floats with the real row
// quantizers (same +/-1 distribution as the precision suites). Random bytes
// interpreted as Q8 blocks are NaN/Inf/denormal poison for the FMA tails and
// unrepresentative of production data. Weights (A) keep xs_fill: generic
// per-quant validity for all 70 registry types is infeasible, kernels are
// fixed-work so content cancels across deterministic-seed comparisons, and
// any residual shows up in the A/A gate. Q8_0 has no real row quantizer
// linked here: keeps xs_fill, and the probe order (Q8_K, Q8_K32 first) means
// it only fires when nothing else prepares.
static void fill_B_valid(int btype, uint8_t * B, size_t rb, int nrc_y, int n, uint64_t seed) {
    if (btype != GGML_TYPE_Q8_K && btype != GGML_TYPE_Q8_K32) {
        xs_seed(seed);
        xs_fill(B, (size_t)nrc_y * rb);
        return;
    }
    xs_seed(seed);
    std::vector<float> f((size_t)n);
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int k = 0; k < n; ++k) f[(size_t)k] = 0.05f * (float)((int)(xs_next() % 41) - 20);
        if (btype == GGML_TYPE_Q8_K) iqk_quantize_row_q8_K(f.data(), B + (size_t)iy * rb, n);
        else quantize_row_q8_K32(f.data(), B + (size_t)iy * rb, n);
    }
}

static std::string date_now() {
    char buf[32] = "?";
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M", std::localtime(&t));
    return std::string(buf);
}

// Affinity apply: single mask for the calling thread. Workers call it at
// thread entry (Windows does not propagate thread affinity to children).
// Topology comes from --bench-pcores (never hardcoded: hybrid layouts vary).
// quiet suppresses the print: worker entry runs per rep, and per-rep lines
// would flood the log (the assignment is printed once by run() instead).
static void apply_mask(uint64_t bits, bool valid, bool quiet = false) {
    if (!valid) return;
    // Per-rep worker entry must stay silent on success (log flood), but a
    // failing mask must surface once (else silently unpinned forever).
    static bool warned_once = false;
    if (!bits) {
        if (!warned_once) { printf("  [WARN] empty affinity mask, continuing unpinned\n"); warned_once = true; }
        return;
    }
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)bits)) {
        if (!warned_once) { printf("  [WARN] affinity mask 0x%llx failed, continuing unpinned\n",
               (unsigned long long)bits); warned_once = true; }
    } else if (!quiet) {
        printf("  [BENCH] affinity mask 0x%llx\n", (unsigned long long)bits);
    }
#elif defined(__linux__)
    cpu_set_t set; CPU_ZERO(&set);
    for (int c = 0; c < 64; ++c) if (bits & ((uint64_t)1 << c)) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        if (!warned_once) { printf("  [WARN] affinity mask 0x%llx failed, continuing unpinned\n",
               (unsigned long long)bits); warned_once = true; }
    } else if (!quiet) {
        printf("  [BENCH] affinity mask 0x%llx\n", (unsigned long long)bits);
    }
#else
    (void)bits;
    if (!warned_once) { printf("  [WARN] affinity unsupported on this platform, continuing unpinned\n"); warned_once = true; }
#endif
}

static bool match_filter(const Options & o, const std::string & op, const std::string & q) {
    if (o.filters.empty()) return true;
    for (size_t i = 0; i < o.filters.size(); ++i) {
        if (op.find(o.filters[i]) != std::string::npos) return true;
        if (q.find(o.filters[i]) != std::string::npos) return true;
    }
    return false;
}

// Progress counters, defined ahead of emit() which stamps them. See the
// plan machinery below for the exactness contract.
static int g_plan_total = 0;
static int g_plan_done = 0;

static void emit(Result r, double flop_per_elem, long long weight_elems,
                 std::vector<Result> & out) {
    // Sub-resolution clocks yield ms==0 -> guard against inf rates.
    r.melem_s = r.ms > 0.0 ? (weight_elems / 1e6) / (r.ms / 1e3) : 0.0;
    r.gflops = r.melem_s * flop_per_elem / 1e3;
    printf("  [BENCH] [%d/%d] %-7s %-14s %-7s n=%-5d x=%-5d y=%-3d B=%-7s reps=%-6d %9.4g ms/call %9.4g Melem/s %9.4g GFLOPS%s\n",
           ++g_plan_done, g_plan_total,
           r.op.c_str(), r.quant.c_str(), r.path.c_str(), r.n, r.nrc_x, r.nrc_y,
           r.btype.c_str(), r.reps, r.ms, r.melem_s, r.gflops,
           r.note.empty() ? "" : ("  (" + r.note + ")").c_str());
    out.push_back(r);
}

// Multithreaded invocation for entries with native ith/nth splits
// (mul/moe/fused): T threads each take their slice, joined per call, so the
// timed window covers real threaded throughput incl. work-split/barriers.
// Locking: with an ordered core list, thread t pins to cores[t % n] —
// exclusive 1:1 while T <= n — so workers can neither leave the set nor
// migrate inside it (the E-core lesson). Without a list, falls back to the
// shared mask (--bench-pin alone piles every thread on that one core) or
// unpinned. Convert/repack have no entry split: they stay single-threaded.
// Workers pin quietly (per-rep prints would flood); the assignment prints
// once in run(). Threads spawn per rep: overhead dominates sub-us stages,
// so judge scaling on production shapes.
static void run_mt(int T, const std::vector<int> & cores, uint64_t mask, bool have_mask,
                   const std::function<void(int,int)> & core) {
    if (T <= 1) { core(0, 1); return; }
    std::vector<std::thread> th;
    for (int t = 1; t < T; ++t) {
        th.emplace_back([&, t]() {
            if (!cores.empty()) apply_mask((uint64_t)1 << (unsigned)cores[(size_t)t % cores.size()], true, true);
            else apply_mask(mask, have_mask, true);
            core(t, T);
        });
    }
    if (!cores.empty()) apply_mask((uint64_t)1 << (unsigned)cores[0], true, true);
    else apply_mask(mask, have_mask, true);
    core(0, T);
    for (auto & h : th) h.join();
}

static double run_timed(const Options & o, const std::function<void()> & fn, int & reps) {
    if (o.fixed_reps > 0) {
        warmup_ms(fn, o.warmup_ms);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < o.fixed_reps; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        reps = o.fixed_reps;
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    }
    return timed_ms(fn, reps, o.target_ms, 1000000, o.warmup_ms);
}

// Plain (non-quantized) types: no IQK compute to measure.
static bool plain_type(enum ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_I8 ||
           t == GGML_TYPE_I16 || t == GGML_TYPE_I32 || t == GGML_TYPE_I64 ||
           t == GGML_TYPE_F64 || t == GGML_TYPE_BF16;
}

// Activation-type candidates for the GEMM entries, first win. MulMat::prepare
// rejects mismatched (weight,B) combos by returning false, so probing is safe.
static const int kBTypes[] = { GGML_TYPE_Q8_K, GGML_TYPE_Q8_K32, GGML_TYPE_Q8_0 };

// Progress contract (counters live ahead of emit() above): every registry
// iteration below prints exactly one [BENCH]/[SKIP] line (probe failures
// print SKIP at all shapes, never vanish silently), so the precomputed plan
// total is exact. Tags stay parser-stable: "[BENCH]" / "[SKIP]" substrings
// preserved, counter appended as [i/N].

// Repack families: single source shared by run() and planned_lines(), so the
// plan count cannot drift from the executed set. Add a row here and the
// family is covered, counted, and timed (needs its hook in run() below).
struct RepackFam { const char * tag; enum ggml_type st, dt; int n, x; };
static const RepackFam kRepacks[] = {
    { "q5_0", GGML_TYPE_Q5_0, GGML_TYPE_Q5_0_R4, 1024, 64 },
    // q4_0 repacks into the iq4_nl_r8 layout (stub sizes Q4_0_R8 so).
    { "q4_0", GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_R8, 1024, 64 },
    { "mxfp4", GGML_TYPE_MXFP4, GGML_TYPE_MXFP4_R8, 1024, 64 },
    { "iq4_nl", GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL_R4, 1024, 64 },
    { "q6_0", GGML_TYPE_Q6_0, GGML_TYPE_Q6_0_R4, 1024, 64 },
    { "q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0_R8, 1024, 64 },
    { "iq4_xs", GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS_R8, 1024, 64 },
};

// Deterministic plan-line count mirroring run() iteration-for-iteration:
// registry n/a SKIP, convert xi 0..1, 3 ops x 3 shapes honoring the blck
// guard, repack rows honoring the filter. Probe outcomes (converter/B-combo)
// print SKIP lines, so they need no entry here. NOTE: ggml_row_size()==0
// stays a silent continue in run() (unreachable: the registry only admits
// sized types); keep this function in sync when adding loops.
static int planned_lines(const Options & o) {
    int total = 0;
    for (int ti = 0; ti < GGML_TYPE_COUNT; ++ti) {
        enum ggml_type t = (enum ggml_type)ti;
        if (ggml_type_size(t) == 0 || ggml_blck_size(t) == 0) continue;
        if (plain_type(t)) continue;
        const char * nm = ggml_type_name(t);
        std::string qname = (nm && nm[0] && strcmp(nm, "?")) ? nm : ("t" + std::to_string(ti));
        int64_t bs = ggml_blck_size(t);
        int n = 1024 % bs == 0 ? 1024 : (2048 % bs == 0 ? 2048 : 0);
        if (!n) { ++total; continue; }
#ifdef TEST_DISPATCH
        if (match_filter(o, "convert", qname)) total += 2;
#else
        (void)0;
#endif
        for (int stage = 0; stage < 3; ++stage) {
            const char * op = stage == 0 ? "mul" : (stage == 1 ? "moe" : "fused");
            if (!match_filter(o, op, qname)) continue;
            for (int si = 0; si < 3; ++si) {
                int nn = (si == 2) ? 2048 : n;
                if (nn % ggml_blck_size(t) != 0) continue;
                ++total;
            }
        }
    }
    for (size_t i = 0; i < sizeof(kRepacks) / sizeof(kRepacks[0]); ++i) {
        std::string tag = std::string("repack-") + kRepacks[i].tag;
        if (!match_filter(o, tag, ggml_type_name(kRepacks[i].st))) continue;
        ++total;
    }
    return total;
}

int run(const Options & opt, std::vector<Result> & rows_out) {
    int T = opt.threads < 1 ? 1 : opt.threads;
    const char * omp = std::getenv("OMP_NUM_THREADS");
    printf("=== IQK speed bench (simd=%s, threads=%d, OMP_NUM_THREADS=%s) ===\n",
           simd_name(), T, omp ? omp : "<unset>");
    printf("MElem/s is over WEIGHT elements (n*nrc_x) for every op; GFLOPS uses the\n");
    printf("nominal model printed per op (convert/repack 8/elem, gemm 2*n*x*y, fused\n");
    printf("4*n*x*y silu excluded). Compare MElem/s across commits, never GFLOPS\n");
    printf("across op kinds. MIPS omitted: int/float mix varies per kernel.\n");
    printf("global warmup: ~1s IQ4_XS convert burst before stage 1 (turbo/AVX settle)%s.\n",
           opt.no_spin ? ", SKIPPED by --bench-no-spin" : "");
    if (T > 1) printf("note: convert/repack stay single-threaded (no entry split support).\n");
    // Core lock set, ordered: --bench-pcores list first (validated here, not
    // just at mask build: run_mt indexes it per thread), else the single
    // --bench-pin cpu, else empty (unpinned). Thread t locks to list[t % n).
    std::vector<int> lock_cores;
    for (int c : opt.pcores) {
        if (c < 0 || c >= 64) { printf("  [WARN] lock core %d out of range, dropped\n", c); continue; }
        lock_cores.push_back(c);
    }
    if (lock_cores.empty() && opt.pin_cpu >= 0) {
        if (opt.pin_cpu >= 64) printf("  [WARN] bench-pin %d out of range, continuing unpinned\n", opt.pin_cpu);
        else lock_cores.push_back(opt.pin_cpu);
    }
    // One affinity for the whole run: single pin becomes a one-bit mask so
    // both paths share apply_mask. Workers re-apply at thread entry.
    // Scalar runs lock the first list core exclusively (not the whole mask).
    uint64_t amask = 0; bool have_mask = false;
    if (T == 1 && !lock_cores.empty()) { amask = (uint64_t)1 << (unsigned)lock_cores[0]; have_mask = true; }
    else if (opt.use_mask) { amask = opt.pin_mask; have_mask = true; }
    else if (!lock_cores.empty()) {
        for (int c : lock_cores) amask |= (uint64_t)1 << (unsigned)c;
        have_mask = true;
    }
    apply_mask(amask, have_mask);
    if (T > 1 && !lock_cores.empty()) {
        std::string asg;
        for (int t = 0; t < T; ++t) {
            if (t) asg += " ";
            asg += "T" + std::to_string(t) + "->cpu" + std::to_string(lock_cores[(size_t)t % lock_cores.size()]);
        }
        printf("  [BENCH] thread lock: %s\n", asg.c_str());
        if (T > (int)lock_cores.size())
            printf("  [WARN] threads=%d exceed %d locked cores; sharing round-robin\n", T, (int)lock_cores.size());
    }
    g_plan_total = planned_lines(opt);
    g_plan_done = 0;
    printf("=== bench plan: %d lines ([BENCH]/[SKIP], exact) ===\n", g_plan_total);

    // Global warmup: representative IQK compute burst (~1s) so core turbo and
    // the AVX license settle before stage 1. Without this, run N+1 measures a
    // warmer core than run N (measured A/A drift on identical code: mean -23%).
    // IQ4_XS convert is always available (native converter, no FANCY gate) and
    // exercises both the integer VNNI and float-scale tails.
    if (!opt.no_spin) {
#ifdef TEST_DISPATCH
    {
        const int wn = 1024, wx = 64;
        size_t wrow = ggml_row_size(GGML_TYPE_IQ4_XS, wn);
        if (wrow) {
            std::vector<uint8_t> wsrc((size_t)wx * wrow + 64), wdst((size_t)wx * wrow * 16 + 64);
            xs_fill(wsrc.data(), wsrc.size());
            auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                iqk_convert_repack_for_test((int)GGML_TYPE_IQ4_XS, wn, wsrc.data(), wrow, wdst.data(), 0, wx);
                if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= 1000.0) break;
            }
        }
    }
#else
    (void)0;
#endif
    }

    // Registry: every sized non-plain type. n per type honors blck_size
    // (R8/R4 families need ne % blck == 0; stub asserts otherwise).
    struct Entry { enum ggml_type t; std::string name; int n; };
    std::vector<Entry> reg;
    for (int ti = 0; ti < GGML_TYPE_COUNT; ++ti) {
        enum ggml_type t = (enum ggml_type)ti;
        if (ggml_type_size(t) == 0 || ggml_blck_size(t) == 0) continue;
        if (plain_type(t)) continue;
        const char * nm = ggml_type_name(t);
        std::string label = (nm && nm[0] && strcmp(nm, "?")) ? nm : ("t" + std::to_string(ti));
        int64_t bs = ggml_blck_size(t);
        int n = 1024 % bs == 0 ? 1024 : (2048 % bs == 0 ? 2048 : 0);
        if (!n) { printf("  [SKIP] [%d/%d] bench n/a for %s\n", ++g_plan_done, g_plan_total, label.c_str()); continue; }
        Entry e; e.t = t; e.name = label; e.n = n;
        reg.push_back(e);
    }

    const int kXs[] = { 16, 64 };   // TG-like, PP-like (max num_rows is 16)
    int n_stages = 0, n_skip = 0;

    for (size_t ri = 0; ri < reg.size(); ++ri) {
        enum ggml_type t = reg[ri].t;
        const std::string & qname = reg[ri].name;
        int n = reg[ri].n;
        size_t rowA = ggml_row_size(t, n);
        if (!rowA) continue;

        // Section header: groups one quant's stages so long bench runs stream
        // readably (one header + its stage lines, then the next quant).
        if (match_filter(opt, "convert", qname) || match_filter(opt, "mul", qname) ||
            match_filter(opt, "moe", qname) || match_filter(opt, "fused", qname))
            printf("--- bench %-14s n=%-5d ---\n", qname.c_str(), n);

        // ---- convert (dispatcher; bool-probed) ----
#ifdef TEST_DISPATCH
        if (match_filter(opt, "convert", qname)) {
            for (int xi = 0; xi < 2; ++xi) {
                int nrc_x = kXs[xi];
                xs_seed((uint64_t)(t + 1) * 100003ull + (uint64_t)nrc_x);
                std::vector<uint8_t> src((size_t)nrc_x * rowA + 64);
                xs_fill(src.data(), src.size());
                std::vector<uint8_t> dst((size_t)nrc_x * rowA * 16 + 64);
                bool ok = iqk_convert_repack_for_test((int)t, n, src.data(), rowA, dst.data(), 0, nrc_x);
                if (!ok) {
                    printf("  [SKIP] [%d/%d] convert %-14s x=%-5d (no converter)\n",
                           ++g_plan_done, g_plan_total, qname.c_str(), nrc_x);
                    ++n_skip;
                    continue;
                }
                int reps = 0;
                auto fn = [&]() {
                    iqk_convert_repack_for_test((int)t, n, src.data(), rowA, dst.data(), 0, nrc_x);
                };
                Result r; r.op = "convert"; r.quant = qname; r.path = "-";
                r.n = n; r.nrc_x = nrc_x; r.nrc_y = -1; r.btype = "-";
                r.ms = run_timed(opt, fn, reps); r.reps = reps;
                emit(r, 8.0, (long long)n * nrc_x, rows_out); ++n_stages;
            }
        }
#else
        (void)0;
#endif

        // ---- mul / moe / fused (public entries, B-type auto-probed) ----
        for (int stage = 0; stage < 3; ++stage) {
            const char * op = stage == 0 ? "mul" : (stage == 1 ? "moe" : "fused");
            if (!match_filter(opt, op, qname)) continue;
            struct PShape { int x, y; };
            PShape shapes[3] = { {16, 1}, {64, 8}, {1792, 64} };
            for (int si = 0; si < 3; ++si) {
                int nrc_x = shapes[si].x, nrc_y = shapes[si].y;
                int nn = (si == 2) ? 2048 : n;
                if (nn % ggml_blck_size(t) != 0) continue;
                size_t rA = ggml_row_size(t, nn);
                xs_seed((uint64_t)(t + 1) * 100003ull + (uint64_t)nrc_x * 17 + (uint64_t)nrc_y);
                std::vector<uint8_t> A((size_t)nrc_x * rA + 64);
                xs_fill(A.data(), A.size());
                std::vector<uint8_t> A2;
                if (stage == 2) { A2.assign(A.size(), 0); xs_fill(A2.data(), A2.size()); }
                std::vector<float> C((size_t)nrc_x * nrc_y + 16, -1.f);
                std::vector<mmid_row_mapping> map((size_t)nrc_y);
                for (int iy = 0; iy < nrc_y; ++iy) { map[(size_t)iy].i1 = iy; map[(size_t)iy].i2 = 0; }

                // Probe B candidates once each; first true wins.
                int btype = -1;
                size_t rB = 0;
                std::vector<uint8_t> B;
                for (size_t bi = 0; bi < sizeof(kBTypes) / sizeof(kBTypes[0]); ++bi) {
                    // Guard first: stub ggml_row_size asserts ne % blck == 0.
                    int64_t bbs = ggml_blck_size((enum ggml_type)kBTypes[bi]);
                    if (!bbs || nn % bbs != 0) continue;
                    size_t rb = ggml_row_size((enum ggml_type)kBTypes[bi], nn);
                    if (!rb) continue;
                    B.assign((size_t)nrc_y * rb + 64, 0);
                    fill_B_valid(kBTypes[bi], B.data(), rb, nrc_y, nn,
                                 (uint64_t)(t + 1) * 100003ull + (uint64_t)nrc_x * 17 + (uint64_t)nrc_y * 131 + bi);
                    bool ok = false;
                    if (stage == 0) ok = iqk_mul_mat(nrc_x, nrc_y, nn, (int)t, A.data(), (long)rA,
                                                    kBTypes[bi], B.data(), (long)rb, C.data(), nrc_x, 0, 1);
                    else if (stage == 1) ok = iqk_mul_mat_moe(nrc_x, nrc_y, nn, nrc_y, (int)t, A.data(), (long)rA,
                                                             kBTypes[bi], B.data(), (long)rb, C.data(),
                                                             (long)nrc_x * (long)sizeof(float), 0, map.data(), 0, 1);
                    else ok = iqk_moe_fused_up_gate(nrc_x, nrc_y, nn, nrc_y, (int)GGML_UNARY_OP_SILU,
                                                   (int)t, A.data(), A2.data(), (long)rA,
                                                   kBTypes[bi], B.data(), (long)rb, nullptr, nullptr, C.data(),
                                                   (long)nrc_x * (long)sizeof(float), 0, map.data(), 0.f, 0, 1);
                    if (ok) { btype = kBTypes[bi]; rB = rb; break; }
                }
                if (btype < 0) {
                    printf("  [SKIP] [%d/%d] %-7s %-14s n=%-5d x=%-5d y=%-3d (no B-type combo)\n",
                           ++g_plan_done, g_plan_total, op, qname.c_str(), nn, nrc_x, nrc_y);
                    ++n_skip;
                    continue;
                }
                std::function<void(int,int)> core;
                if (stage == 0) core = [&](int ith, int nth) {
                    iqk_mul_mat(nrc_x, nrc_y, nn, (int)t, A.data(), (long)rA,
                                btype, B.data(), (long)rB, C.data(), nrc_x, ith, nth);
                };
                else if (stage == 1) core = [&](int ith, int nth) {
                    iqk_mul_mat_moe(nrc_x, nrc_y, nn, nrc_y, (int)t, A.data(), (long)rA,
                                    btype, B.data(), (long)rB, C.data(),
                                    (long)nrc_x * (long)sizeof(float), 0, map.data(), ith, nth);
                };
                else core = [&](int ith, int nth) {
                    iqk_moe_fused_up_gate(nrc_x, nrc_y, nn, nrc_y, (int)GGML_UNARY_OP_SILU,
                                          (int)t, A.data(), A2.data(), (long)rA,
                                          btype, B.data(), (long)rB, nullptr, nullptr, C.data(),
                                          (long)nrc_x * (long)sizeof(float), 0, map.data(), 0.f, ith, nth);
                };
                std::function<void()> fn = [&]() { run_mt(T, lock_cores, amask, have_mask, core); };
                std::string path = "-";
#ifdef TEST_DISPATCH
                path = iqk_dequant_type((int)t, nrc_y) != (int)t ? "convert" : "direct";
#endif
                int reps = 0;
                Result r; r.op = op; r.quant = qname; r.path = path;
                r.n = nn; r.nrc_x = nrc_x; r.nrc_y = nrc_y;
                r.btype = ggml_type_name((enum ggml_type)btype);
                if (T > 1) r.note = "t" + std::to_string(T) + (!lock_cores.empty() ? ",lock" : "");
                r.ms = run_timed(opt, fn, reps); r.reps = reps;
                // Per-elem model over WEIGHT elements: gemm does nrc_y MACs
                // per weight (2 flop), fused does up+gate (4 flop, silu
                // excluded and noted in the header).
                emit(r, stage == 2 ? 4.0 * nrc_y : 2.0 * nrc_y, (long long)nn * nrc_x, rows_out); ++n_stages;
            }
        }
    }

    // ---- repack (families with harness wrappers only; rest honestly absent) ----
    // Table lives in kRepacks (shared with planned_lines()): add families there.
    {
        bool did_repack = false;
        const RepackFam * rps = kRepacks;
        const size_t n_rps = sizeof(kRepacks) / sizeof(kRepacks[0]);
        for (size_t i = 0; i < n_rps; ++i) {
            std::string tag = std::string("repack-") + rps[i].tag;
            if (!match_filter(opt, tag, ggml_type_name(rps[i].st))) continue;
            did_repack = true;
            size_t bs = ggml_row_size(rps[i].st, rps[i].n);
            size_t bd = ggml_row_size(rps[i].dt, rps[i].n);
            if (!bs || !bd) { printf("  [SKIP] [%d/%d] %-14s (no sizes)\n", ++g_plan_done, g_plan_total, tag.c_str()); ++n_skip; continue; }
            xs_seed(9000 + i);
            std::vector<uint8_t> src((size_t)rps[i].x * bs + 64), dst((size_t)rps[i].x * bd + 64);
            xs_fill(src.data(), src.size());
            // Per-family call through the matching hook (signatures differ by block type).
            std::function<void()> fn;
            if (rps[i].st == GGML_TYPE_Q5_0) fn = [&]() {
                iqk_test_repack_q5_0(rps[i].x, rps[i].n, (const block_q5_0 *)src.data(), (block_q5_0_r4 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_Q4_0) fn = [&]() {
                iqk_test_repack_q4_0(rps[i].x, rps[i].n, (const block_q4_0 *)src.data(), (block_iq4_nl_r8 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_MXFP4) fn = [&]() {
                iqk_test_repack_mxfp4(rps[i].x, rps[i].n, (const block_mxfp4 *)src.data(), (block_mxfp4_r8 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_IQ4_NL) fn = [&]() {
                iqk_test_repack_iq4_nl(rps[i].x, rps[i].n, (const block_iq4_nl *)src.data(), (block_iq4_nl_r4 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_Q8_0) fn = [&]() {
                iqk_test_repack_q8_0(rps[i].x, rps[i].n, (const block_q8_0 *)src.data(), (block_q8_0_r8 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_IQ4_XS) fn = [&]() {
                iqk_test_repack_iq4_xs(rps[i].x, rps[i].n, (const block_iq4_xs *)src.data(), (block_iq4_xs_r8 *)dst.data());
            };
            else if (rps[i].st == GGML_TYPE_Q6_0) fn = [&]() {
                iqk_test_repack_q6_0(rps[i].x, rps[i].n, (const block_q6_0 *)src.data(), (block_q6_0_r4 *)dst.data());
            };
            else { printf("  [SKIP] [%d/%d] %-14s (no wrapper)\n", ++g_plan_done, g_plan_total, tag.c_str()); ++n_skip; continue; }
            int reps = 0;
            Result r; r.op = tag; r.quant = ggml_type_name(rps[i].st); r.path = "-";
            r.n = rps[i].n; r.nrc_x = rps[i].x; r.nrc_y = -1; r.btype = "-";
            r.note = std::string("-> ") + ggml_type_name(rps[i].dt);
            r.ms = run_timed(opt, fn, reps); r.reps = reps;
            emit(r, 8.0, (long long)rps[i].n * rps[i].x, rows_out); ++n_stages;
        }
        if (did_repack) {
            printf("  [INFO] repack: 7 families wired (no harness wrapper for the rest;\n");
            printf("  [INFO] repack: quantize/dequantize rows not benched (mostly link stubs).\n");
        }
    }

    printf("=== bench done: %d stages, %d skips ===\n", n_stages, n_skip);
    return 0;
}

// ---- editable flat-file DB ----
static std::string db_key(const std::string & op, const std::string & quant, const std::string & path,
                          int n, int x, int y, const std::string & btype) {
    char k[256];
    snprintf(k, sizeof(k), "%s|%s|%s|%d|%d|%d|%s", op.c_str(), quant.c_str(), path.c_str(), n, x, y, btype.c_str());
    return std::string(k);
}

int db_append(const std::string & db_path, const std::string & label,
              const std::vector<Result> & rows) {
    FILE * f = fopen(db_path.c_str(), "a");
    if (!f) { fprintf(stderr, "bench: cannot append %s\n", db_path.c_str()); return 1; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "# iqk bench db: '|' separated, edit freely. Keep field order.\n");
        fprintf(f, "# date|label|simd|op|quant|path|n|nrc_x|nrc_y|btype|reps|ms|melem_s|gflops|note\n");
    }
    std::string dt = date_now(), sm = simd_name();
    for (size_t i = 0; i < rows.size(); ++i) {
        const Result & r = rows[i];
        std::string note = r.note;
        for (size_t p = note.find('|'); p != std::string::npos; p = note.find('|')) note[p] = '/';
        fprintf(f, "%s|%s|%s|%s|%s|%s|%d|%d|%d|%s|%d|%.6g|%.6g|%.6g|%s\n",
                dt.c_str(), label.c_str(), sm.c_str(), r.op.c_str(), r.quant.c_str(),
                r.path.c_str(), r.n, r.nrc_x, r.nrc_y, r.btype.c_str(),
                r.reps, r.ms, r.melem_s, r.gflops, note.c_str());
    }
    fclose(f);
    printf("  [BENCH] saved %d rows as '%s' -> %s\n", (int)rows.size(), label.c_str(), db_path.c_str());
    return 0;
}

struct DbRow {
    std::string date, label, simd, op, quant, path, btype, note;
    int n, x, y, reps;
    double ms, melem, gflops;
};

static bool db_load(const std::string & db_path, std::vector<DbRow> & out, std::string & err) {
    FILE * f = fopen(db_path.c_str(), "r");
    if (!f) { err = "cannot open " + db_path; return false; }
    char line[4096];
    int lineno = 0, bad = 0, first_bad = 0;
    while (fgets(line, sizeof(line), f)) {
        ++lineno;
        if (!line[0] || line[0] == '#') continue;
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L) continue;
        // Split first 14 '|' fields; remainder is note (may contain '|').
        std::string f14[14];
        const char * p = line;
        int fi = 0;
        for (; fi < 14; ++fi) {
            const char * q = strchr(p, '|');
            if (!q) break;
            f14[fi].assign(p, q);
            p = q + 1;
        }
        if (fi < 14) { if (!bad) first_bad = lineno; ++bad; continue; }
        DbRow r;
        r.note = p;
        r.date = f14[0];
        r.label = f14[1]; r.simd = f14[2]; r.op = f14[3]; r.quant = f14[4];
        r.path = f14[5]; r.btype = f14[9];
        r.n = atoi(f14[6].c_str()); r.x = atoi(f14[7].c_str()); r.y = atoi(f14[8].c_str());
        r.reps = atoi(f14[10].c_str());
        r.ms = atof(f14[11].c_str()); r.melem = atof(f14[12].c_str()); r.gflops = atof(f14[13].c_str());
        (void)0;
        out.push_back(r);
    }
    fclose(f);
    if (bad) fprintf(stderr, "bench: %d malformed lines skipped in %s (first at line %d)\n",
                     bad, db_path.c_str(), first_bad);
    return true;
}

// First simd recorded under a label (empty DB/label -> "?").
static std::string simd_of(const std::vector<DbRow> & rows, const std::string & label) {
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].label == label) return rows[i].simd;
    }
    return "?";
}

int print_delta(const std::string & db_path, const std::string & a, const std::string & b) {
    std::vector<DbRow> rows;
    std::string err;
    if (!db_load(db_path, rows, err)) { fprintf(stderr, "bench: %s\n", err.c_str()); return 1; }
    // Index B rows by key (last wins on hand-edited dupes).
    struct KMap { std::string key; size_t idx; };
    std::vector<KMap> idx;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].label != b) continue;
        std::string k = db_key(rows[i].op, rows[i].quant, rows[i].path, rows[i].n, rows[i].x, rows[i].y, rows[i].btype);
        bool found = false;
        for (size_t j = 0; j < idx.size(); ++j) {
            if (idx[j].key == k) { idx[j].idx = i; found = true; break; }
        }
        if (!found) { KMap m; m.key = k; m.idx = i; idx.push_back(m); }
    }
    printf("=== bench delta: '%s' (%s) -> '%s' (%s) (ms based, negative = faster) ===\n",
           a.c_str(), simd_of(rows, a).c_str(), b.c_str(), simd_of(rows, b).c_str());
    printf("  # columns: op quant | n nrc_x nrc_y | A-ms -> B-ms (pct) | A-Melem -> B-Melem; pct<0 means B faster\n");
    int n_match = 0, n_faster = 0, n_slower = 0, n_only_a = 0;
    double sum_pct = 0.0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].label != a) continue;
        std::string k = db_key(rows[i].op, rows[i].quant, rows[i].path, rows[i].n, rows[i].x, rows[i].y, rows[i].btype);
        size_t hit = (size_t)-1;
        for (size_t j = 0; j < idx.size(); ++j) {
            if (idx[j].key == k) { hit = idx[j].idx; break; }
        }
        if (hit == (size_t)-1) { ++n_only_a; continue; }
        const DbRow & A = rows[i], & B = rows[hit];
        double pct = A.ms > 0 ? 100.0 * (B.ms - A.ms) / A.ms : 0.0;
        sum_pct += pct;
        // 2% significance band: pre-warmup-fix A/A runs jittered +/-1..8% on
        // small stages, so 0.5% flagged pure noise. Re-derive from an A/A run
        // if the harness or machine changes.
        if (pct < -2.0) ++n_faster; else if (pct > 2.0) ++n_slower;
        printf("  [DELTA] %-7s %-14s n=%-5d x=%-5d y=%-3d %8.4g -> %8.4g ms (%+7.2f%%)  Melem %8.4g -> %8.4g\n",
               A.op.c_str(), A.quant.c_str(), A.n, A.x, A.y, A.ms, B.ms, pct, A.melem, B.melem);
        ++n_match;
    }
    // B-only rows (added coverage).
    int n_only_b = 0;
    for (size_t j = 0; j < idx.size(); ++j) {
        bool seen = false;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].label != a) continue;
            std::string k = db_key(rows[i].op, rows[i].quant, rows[i].path, rows[i].n, rows[i].x, rows[i].y, rows[i].btype);
            if (k == idx[j].key) { seen = true; break; }
        }
        if (!seen) ++n_only_b;
    }
    printf("=== delta: %d matched (%d faster, %d slower, mean %+.2f%%), %d only-in-%s, %d only-in-%s ===\n",
           n_match, n_faster, n_slower, n_match ? sum_pct / n_match : 0.0,
           n_only_a, a.c_str(), n_only_b, b.c_str());
    return 0;
}

// ---- dual-branch pingpong (see run_pingpong in unit_test.h) ----
static const int kCooldownSec = 5;

static void sleep_sec(int s) {
    std::this_thread::sleep_for(std::chrono::seconds(s));
}

// Capture a child harness's stdout lines. Same-generation binary required
// (older ones print no [BENCH] lines; warned, skipped, never fatal).
static bool run_child_capture(const std::string & cmd, std::vector<std::string> & lines) {
#ifdef _WIN32
    FILE * p = _popen(cmd.c_str(), "r");
#else
    FILE * p = popen(cmd.c_str(), "r");
#endif
    if (!p) return false;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) {
        size_t L = strlen(buf);
        while (L && (buf[L - 1] == '\n' || buf[L - 1] == '\r')) buf[--L] = 0;
        lines.emplace_back(buf);
    }
#ifdef _WIN32
    int rc = _pclose(p);
#else
    int rc = pclose(p);
#endif
    return rc == 0;
}

// Parse one "  [BENCH] ..." child stdout line into a Result. Field layout
// mirrors emit() above; the trailing "(note)" is optional. Accepts both the
// current [i/N]-counter layout and the legacy counter-less layout (older
// binaries), so pingpong stays compatible across harness generations.
static bool parse_bench_line(const std::string & line, Result & r) {
    if (line.find("[BENCH]") == std::string::npos) return false;
    char op[16] = "", q[32] = "", path[16] = "", bt[16] = "";
    int n = 0, x = 0, y = 0, reps = 0, ci = 0, cn = 0;
    double ms = 0, me = 0, gf = 0;
    int k = sscanf(line.c_str(), " [BENCH] [%d/%d] %15s %31s %15s n=%d x=%d y=%d B=%15s reps=%d %lg ms/call %lg Melem/s %lg GFLOPS",
                   &ci, &cn, op, q, path, &n, &x, &y, bt, &reps, &ms, &me, &gf);
    if (k < 14) {
        k = sscanf(line.c_str(), " [BENCH] %15s %31s %15s n=%d x=%d y=%d B=%15s reps=%d %lg ms/call %lg Melem/s %lg GFLOPS",
                   op, q, path, &n, &x, &y, bt, &reps, &ms, &me, &gf);
        if (k < 12) return false;
    }
    r.op = op; r.quant = q; r.path = path;
    r.n = n; r.nrc_x = x; r.nrc_y = y; r.btype = bt;
    r.reps = reps; r.ms = ms; r.melem_s = me; r.gflops = gf;
    const char * np = strstr(line.c_str(), "  (");
    if (np) {
        std::string note = np + 3;
        if (!note.empty() && note.back() == ')') note.pop_back();
        r.note = note;
    } else {
        r.note.clear();
    }
    return true;
}

// Validated family chunks (same partition as the manual alternating protocol:
// full coverage, one chunk per tool call, streamable). Fixed table on purpose:
// ad-hoc filters would silently unbalance the A/B comparison.
struct PingFam { const char * name; const char * f[10]; int nf; };
static const PingFam kPingFams[] = {
    { "iq4xs", { "iq4_xs" }, 1 },
    { "kq",    { "q4_K", "q4_k", "q5_K", "q5_k", "q6_K", "q6_k" }, 6 },
    { "q8leg", { "q8", "q4_0", "q5_0", "q6_0", "q4_1", "q5_1", "iq4_nl", "mxfp4", "q8_1" }, 9 },
    { "small", { "iq2", "iq3", "q2_K", "q2_k", "q3_K", "q3_k", "iq1", "q1_0", "pq2", "ptq1" }, 10 },
};

int run_pingpong(const Options & opt, const std::string & exe_b,
                 const std::string & label_a, const std::string & label_b) {
    if (!opt.filters.empty())
        printf("  [WARN] pingpong ignores %d extra --bench-filter (fixed families)\n", (int)opt.filters.size());
    printf("=== pingpong: self [%s] vs '%s' [%s] (5 s cooldown between runs) ===\n",
           label_a.c_str(), exe_b.c_str(), label_b.c_str());
    int rc = 0;
    for (size_t fi = 0; fi < sizeof(kPingFams) / sizeof(kPingFams[0]) && rc == 0; ++fi) {
        const PingFam & fam = kPingFams[fi];
        printf("--- pingpong family %s ---\n", fam.name);
        // A side: self, in-process.
        Options oA = opt;
        oA.filters.clear();
        for (int k = 0; k < fam.nf; ++k) oA.filters.emplace_back(fam.f[k]);
        std::vector<Result> rowsA;
        rc = run(oA, rowsA);
        if (rc == 0) rc = db_append(opt.db_path, label_a, rowsA);
        sleep_sec(kCooldownSec);
        if (rc != 0) break;
        // B side: child harness with the same measurement flags (never
        // --bench-save: the driver appends parsed rows itself).
        std::string cmd = "\"" + exe_b + "\" --bench";
        for (int k = 0; k < fam.nf; ++k) { cmd += " --bench-filter \""; cmd += fam.f[k]; cmd += "\""; }
        cmd += " --bench-threads " + std::to_string(oA.threads < 1 ? 1 : oA.threads);
        { char tb[128]; snprintf(tb, sizeof(tb), " --bench-target-ms %.4g --bench-warmup-ms %.4g", oA.target_ms, oA.warmup_ms); cmd += tb; }
        if (oA.no_spin) cmd += " --bench-no-spin";
        if (oA.fixed_reps > 0) cmd += " --bench-reps " + std::to_string(oA.fixed_reps);
        if (!oA.pcores.empty()) {
            cmd += " --bench-pcores ";
            for (size_t k = 0; k < oA.pcores.size(); ++k) { if (k) cmd += ","; cmd += std::to_string(oA.pcores[k]); }
        } else if (oA.pin_cpu >= 0) {
            cmd += " --bench-pin " + std::to_string(oA.pin_cpu);
        }
        cmd += " --bench-db \"" + opt.db_path + "\"";
        std::vector<std::string> bout;
        if (!run_child_capture(cmd, bout)) {
            fprintf(stderr, "pingpong: child run failed for family %s, continuing\n", fam.name);
            sleep_sec(kCooldownSec);
            continue;
        }
        std::vector<Result> rowsB;
        int nbad = 0;
        for (size_t i = 0; i < bout.size(); ++i) {
            if (bout[i].find("[BENCH]") == std::string::npos) continue;
            Result r;
            if (parse_bench_line(bout[i], r)) rowsB.push_back(r);
            else ++nbad;
        }
        if (nbad) printf("  [WARN] pingpong: %d unparsable BENCH lines from child (harness mismatch?)\n", nbad);
        if (rowsB.empty()) {
            fprintf(stderr, "pingpong: no BENCH rows from child for family %s (same-harness binary required)\n", fam.name);
            sleep_sec(kCooldownSec);
            continue;
        }
        if (db_append(opt.db_path, label_b, rowsB)) { rc = 1; break; }
        // Live pair table per family (same key rule as print_delta).
        for (size_t i = 0; i < rowsA.size(); ++i) {
            std::string k = db_key(rowsA[i].op, rowsA[i].quant, rowsA[i].path, rowsA[i].n, rowsA[i].nrc_x, rowsA[i].nrc_y, rowsA[i].btype);
            for (size_t j = 0; j < rowsB.size(); ++j) {
                const Result & B = rowsB[j];
                if (k != db_key(B.op, B.quant, B.path, B.n, B.nrc_x, B.nrc_y, B.btype)) continue;
                const Result & A = rowsA[i];
                double pct = A.ms > 0 ? 100.0 * (B.ms - A.ms) / A.ms : 0.0;
                printf("  [PING %s>%s] %-7s %-14s n=%-5d x=%-5d y=%-3d %8.4g -> %8.4g ms (%+7.2f%%)\n",
                       label_a.c_str(), label_b.c_str(),
                       A.op.c_str(), A.quant.c_str(), A.n, A.nrc_x, A.nrc_y, A.ms, B.ms, pct);
                break;
            }
        }
        sleep_sec(kCooldownSec);
    }
    if (rc == 0) rc = print_delta(opt.db_path, label_a, label_b);
    return rc;
}

int export_csv(const std::string & db_path, const std::string & csv_path,
               const std::string & label) {
    std::vector<DbRow> rows;
    std::string err;
    if (!db_load(db_path, rows, err)) { fprintf(stderr, "bench: %s\n", err.c_str()); return 1; }
    FILE * f = fopen(csv_path.c_str(), "w");
    if (!f) { fprintf(stderr, "bench: cannot write %s\n", csv_path.c_str()); return 1; }
    fprintf(f, "label,date,simd,op,quant,path,n,nrc_x,nrc_y,btype,reps,ms,melem_s,gflops,note\n");
    auto csvq = [](const std::string & s, char * out, size_t cap) {
        bool q = s.find_first_of(",\"\n") != std::string::npos;
        size_t o = 0;
        if (q && o < cap) out[o++] = '"';
        for (size_t i = 0; i < s.size() && o + 2 < cap; ++i) {
            if (s[i] == '"' && o + 2 < cap) { out[o++] = '"'; out[o++] = '"'; }
            else out[o++] = s[i];
        }
        if (q && o < cap) out[o++] = '"';
        out[o < cap ? o : cap - 1] = 0;
    };
    int n_out = 0;
    char cell[512];
    for (size_t i = 0; i < rows.size(); ++i) {
        const DbRow & r = rows[i];
        if (!label.empty() && r.label != label) continue;
        csvq(r.label, cell, sizeof(cell)); std::string c0 = cell;
        csvq(r.date, cell, sizeof(cell)); std::string cD = cell;
        csvq(r.simd, cell, sizeof(cell)); std::string c1 = cell;
        csvq(r.op, cell, sizeof(cell)); std::string c2 = cell;
        csvq(r.quant, cell, sizeof(cell)); std::string c3 = cell;
        csvq(r.path, cell, sizeof(cell)); std::string c4 = cell;
        csvq(r.btype, cell, sizeof(cell)); std::string c5 = cell;
        csvq(r.note, cell, sizeof(cell)); std::string c6 = cell;
        fprintf(f, "%s,%s,%s,%s,%s,%s,%d,%d,%d,%s,%d,%.6g,%.6g,%.6g,%s\n",
                c0.c_str(), cD.c_str(), c1.c_str(), c2.c_str(), c3.c_str(), c4.c_str(),
                r.n, r.x, r.y, c5.c_str(), r.reps, r.ms, r.melem, r.gflops, c6.c_str());
        ++n_out;
    }
    fclose(f);
    printf("  [BENCH] exported %d rows -> %s\n", n_out, csv_path.c_str());
    return 0;
}

} // namespace bench
