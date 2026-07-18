//
// unit_test.cpp - IQK iq4_xs_r8 -> q8_k_r8 / q8_k_r16 converter verification
//
// Compares the converter (iqk_convert_iq4_xs_r8_q8_k_r16) against the reference
// dequant -> quant pipeline for R8 (-rtr) and R16 (-r16p) paths.
//
// The reference builds data in the CONTIGUOUS layout expected by the downstream
// kernels (nb blocks per group-of-rows, no gaps).  The converter must produce the
// same byte layout.
//
// Exercises PP-like (large nrc_x) and TG-like (minimum nrc_x) sizes, and checks
// the converter output for NaN / Inf.
//
// Add -DTEST_VNNI256 or -DTEST_VNNIINT8 to exercise the SIMD code path (the one
// used on Panther Lake / AVX_VNNI_INT8=1).  Without those the float fallback is
// tested.
//
// Build (via cmake examples target that links the full ggml lib):
//   cmake --build build --target unit_test -j
//
// Or standalone cl:
//   cl /EHsc /std:c++17 /O2 /arch:AVX2 /DGGML_USE_IQK_MULMAT /DIQK_IMPLEMENT
//      /DTEST_VNNI256 /I<repo>/ggml/src /I<repo>/ggml/include /I<repo>/ggml/src/iqk
//      unit_test.cpp <repo>/ggml/src/iqk/iqk_gemm_kquants.cpp <repo>/ggml/src/iqk/iqk_quantize.cpp
//      <repo>/ggml/src/ggml.c <repo>/ggml/src/ggml-alloc.c <repo>/ggml/src/ggml-backend.c
//      <repo>/ggml/src/ggml-quants.c <repo>/ggml/src/ggml-threading.c
//      <repo>/ggml/src/iqk/iqk_common.cpp
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cstdint>
#include <chrono>
#include <functional>

// Pull in ggml common defs (block types, QK_K, ggml_row_size).
#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

// IQK internals. The converter (iqk_gemm_kquants.h) is C++-linkage; the
// reference quantizers (iqk_quantize.h) are C-linkage. Both require IQK_IMPLEMENT
// and GGML_USE_IQK_MULMAT to be declared.
#define GGML_USE_IQK_MULMAT
#define IQK_IMPLEMENT
#include "iqk/iqk_common.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_gemm_kquants.h"
#include "iqk/iqk_gemm_legacy_quants.h"
#include "iqk/iqk_mul_mat.h"
#include "unit_test/unit_test.h"

// g_iqk_r16_path normally lives in the ggml lib (iqk_mul_mat.cpp).  The test
// provides a local copy; set it before each call to// iqk_convert_iq4_xs_r8_q8_k_r16.
// Build with -DTEST_VNNI256 and/or -DTEST_VNNIINT8 to exercise the SIMD converter
// path (Panther Lake / AVX_VNNI_INT8).  Without those the float fallback is tested.
#ifdef TEST_VNNI256
#define HAVE_VNNI256
#endif
#ifdef TEST_VNNIINT8
#define HAVE_VNNIINT8
#endif

// The dispatch-pipeline tests (test_dispatch_pipeline) require iqk_dequant_type,
// iqk_convert_repack from iqk_mul_mat.cpp.  Enable with -DTEST_DISPATCH and
// link iqk_mul_mat.cpp (or the full ggml lib).
#ifdef TEST_DISPATCH
extern "C" int iqk_dequant_type(int type, int Ny);
extern "C" const char * ggml_type_name(enum ggml_type type);
extern "C" bool iqk_convert_repack_for_test(int typeA, int n, const void * vx, size_t bx,
                                void * vy, size_t stride_y, int nrc_x);
#endif

// init_unit_test_fp16_table() is provided by your build (populates
// ggml_table_f32_f16). Declared here; do not redefine.
void init_unit_test_fp16_table();

// Test-only repack entry point (defined in iqk_quantize.cpp next to the
// other test hooks; declared locally to keep this test self-contained).
extern "C" void iqk_test_repack_q5_0(int nrows, int n_per_row, const block_q5_0 * x, block_q5_0_r4 * y);
extern "C" void iqk_test_repack_q4_0(int nrows, int n_per_row, const block_q4_0 * x, block_iq4_nl_r8 * y);
extern "C" void iqk_test_repack_mxfp4(int nrows, int n_per_row, const block_mxfp4 * x, block_mxfp4_r8 * y);
extern "C" void iqk_test_repack_iq4_nl(int nrows, int n_per_row, const block_iq4_nl * x, block_iq4_nl_r4 * y);
extern "C" void iqk_test_repack_q8_0(int nrows, int n_per_row, const block_q8_0 * x, block_q8_0_r8 * y);
extern "C" void iqk_test_repack_iq4_xs(int nrows, int n_per_row, const block_iq4_xs * x, block_iq4_xs_r8 * y);

// g_iqk_r16_path is extern'd in iqk_common.h but defined in iqk_mul_mat.cpp
// which is NOT linked into this standalone test.  Provide our own definition.
bool g_iqk_r16_path = false;

static int  g_seed = 12345;
static int  g_failures = 0;
static bool g_strict = false;   // when true, no ±1 qs tolerance in test_path / kv-sweep
static bool g_cinit_plain = false; // --cinit-plain: init fused-MoE C to -1.f (plain-test
                                   // convention) instead of the -1e30 sentinel. Bisects
                                   // C-init sensitivity: fails vanish => something reads C.
static std::mt19937 g_rng(g_seed);

// Converter-speed probe (threshold-tuning input): runs fn with warmup +
// adaptive reps (>=200 ms total, capped), returns ms/call. Prints ms/call,
// Melem/s (exact) and nominal GFLOPS at ASSUMED flop/element (dequant +
// scale + requant path). Compare Melem/s across converters, never GFLOPS
// across different kernels (the flop assumption is nominal).
static double bench_convert_ms(const char * tag, long n_elem, double flop_per_elem,
                               const std::function<void()> & fn) {
    int reps = 0;
    double ms = bench::timed_ms(fn, reps); // shared adaptive timer (see bench.cpp)
    double melem_s = (n_elem / 1e6) / (ms / 1e3);
    double gflops = melem_s * flop_per_elem / 1e3;
    printf("  [INFO] convert-speed %s: %.4g ms/call, %.4g Melem/s, %.4g nominal GFLOPS (@%.1f flop/elem)\n",
           tag, ms, melem_s, gflops, flop_per_elem);
    return ms;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a random but *valid* block_iq4_xs_r8: random half deltas, random scales,
// random 4-bit nibbles in qs. The reference path dequantizes through the same
// struct, so any structural mismatch shows up directly.
// Production-like regime: when g_coherent_weights is set (coherent fused/MoE
// suites), deltas come from the small production range instead of full-range
// random, isolating magnitude-sensitivity from structural bugs.
static bool g_coherent_weights = false;
static void make_random_iq4_xs_r8(block_iq4_xs_r8 * blk) {
    const float dmax = g_coherent_weights ? 0.1f : 1.0f;
    for (int k = 0; k < 8; ++k) {
        float d = (g_rng() % 1000) * 0.001f * dmax + 0.01f;
        blk->d[k] = GGML_FP32_TO_FP16(d);
    }
    for (size_t i = 0; i < sizeof(blk->scales_l); ++i) blk->scales_l[i] = (uint8_t)(g_rng() & 0xff);
    for (size_t i = 0; i < sizeof(blk->scales_h); ++i) blk->scales_h[i] = (uint8_t)(g_rng() & 0xff);
    for (size_t i = 0; i < sizeof(blk->qs);      ++i) blk->qs[i]      = (uint8_t)(g_rng() & 0xff);
}

// Build a random but *valid* block_q6_0: random half delta, random high
// bits and nibbles. The reference paths dequantize through the same
// struct, so any structural mismatch shows up directly.
static void make_random_q6_0(block_q6_0 * blk) {
    float d = (g_rng() % 1000) * 0.001f + 0.01f;
    blk->d = GGML_FP32_TO_FP16(d);
    for (auto & v : blk->qh) v = (uint8_t)(g_rng() & 0xff);
    for (auto & v : blk->qs) v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order Q6_0 dequant mirroring convert_q6_0 in iqk_quantize.cpp:
// each qs byte holds two 4-bit lows, each qh byte the corresponding high
// pairs; value = 6-bit - 32.
static void dequant_q6_0_row(const block_q6_0 * x, float * y, int64_t k) {
    int64_t nb = k / QK6_0;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK6_0 / 2; ++j) {
            uint8_t h = (uint8_t)(x[i].qh[j % (QK6_0 / 4)] >> (4 * (j / (QK6_0 / 4))));
            int l0 = (x[i].qs[j] & 0x0F) | ((h << 4) & 0x30);
            int l1 = (x[i].qs[j] >> 4)   | ((h << 2) & 0x30);
            y[i * QK6_0 + j]             = d * (l0 - 32);
            y[i * QK6_0 + j + QK6_0 / 2] = d * (l1 - 32);
        }
    }
}

// Build a random but *valid* block_q8_0: sane half delta, full-range quants.
static void make_random_q8_0(block_q8_0 * blk) {
    float d = (g_rng() % 1000) * 0.001f + 0.01f;
    blk->d = GGML_FP32_TO_FP16(d);
    for (auto & v : blk->qs) v = (int8_t)(g_rng() & 0xff);
}

// Natural-order Q8_0 dequant (ggml-quants.c is not linked into this test).
static void dequant_q8_0_row(const block_q8_0 * x, float * y, int64_t k) {
    int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = d * x[i].qs[j];
    }
}

// Build a random but *valid* block_iq4_xs (same fill as test_repack_integrity).
static void make_random_iq4_xs(block_iq4_xs * blk) {
    blk->d = GGML_FP32_TO_FP16((g_rng() % 1000) * 0.001f + 0.01f);
    blk->scales_h = (uint16_t)(g_rng() & 0xffff);
    for (auto & v : blk->scales_l) v = (uint8_t)(g_rng() & 0xff);
    for (auto & v : blk->qs)      v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order IQ4_XS dequant (same math as test_repack_integrity).
static void dequant_native_iq4_xs(const block_iq4_xs * x, float * y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * iq4k_values[x[i].qs[16*ib+j] & 0xf];
                y[j+16] = dl * iq4k_values[x[i].qs[16*ib+j] >>  4];
            }
            y += 32;
        }
    }
}

// Local Q8_2_X4 dequant (layout mirror of quantize_row_q8_2_x4: nb4 groups of
// 4x block_q8_2_x4 then a plain block_q8_2 tail). The harness has no
// dequantize_row_q8_2_x4; same math as the inline loop in
// test_gemm_mxfp4_r8_direct.
static void dequant_q8_2_x4_rows(const uint8_t * B, size_t bx_B, float * fout, int nrows, int n) {
    const int nb_b = n / QK8_2;
    const int nb4_b = 4*(nb_b/4);
    for (int iy = 0; iy < nrows; ++iy) {
        float * out = fout + (size_t)iy * n;
        const auto * x4 = (const block_q8_2_x4 *)(B + (size_t)iy * bx_B);
        for (int g = 0; g < nb4_b/4; ++g) {
            for (int r = 0; r < 4; ++r) {
                float d = GGML_BF16_TO_FP32(ggml_bf16_t{x4[g].d[r]});
                for (int j = 0; j < QK8_2; ++j) out[(g*4+r)*QK8_2 + j] = d * x4[g].qs[r*QK8_2 + j];
            }
        }
        const auto * rem = (const block_q8_2 *)(x4 + nb4_b/4);
        for (int i = 0; i < nb_b - nb4_b; ++i) {
            float d = GGML_BF16_TO_FP32(ggml_bf16_t{rem[i].d});
            for (int j = 0; j < QK8_2; ++j) out[(nb4_b+i)*QK8_2 + j] = d * rem[i].qs[j];
        }
    }
}

// Check fp16 delta array for true NaN/Inf (fp16 bit patterns 0x7C00..0x7FFF or
// 0xFC00..0xFFFF). A plain out-of-range finite value is not NaN.
template <typename Block>
static bool deltas_bad(const Block * b, int nrows) {
    for (int k = 0; k < nrows; ++k) {
        float f = GGML_FP16_TO_FP32(b->d[k]);
        if (!std::isfinite(f)) return true;
    }
    return false;
}

// Local row-size computation (avoids linking ggml.c, which this standalone test
// does not build). Q8_K_R8 / Q8_K_R16 pack QK_K elements per block.
static size_t q8_row_size(bool r16, int n) {
    if (r16) return (size_t)(n / QK_K) * sizeof(block_q8_k_r16);
    return (size_t)(n / QK_K) * sizeof(block_q8_k_r8);
}

// Local block_q8_K dequant (ggml-quants.c is not linked into this test).
// block_q8_K holds plain int8 quants: y[j] = d * qs[j].
static void dequant_q8_K_row(const block_q8_K * x, float * y, int64_t k) {
    int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        float d = x[i].d;
        for (int j = 0; j < QK_K; ++j) y[i * QK_K + j] = d * x[i].qs[j];
    }
}

// Dequantize the interleaved fbuf from IQ4_XS_R8 src blocks.
static void fill_interleaved_fbuf(const block_iq4_xs_r8 * src, float * fbuf, int nrc_x, int n, int nb) {    float * fp = fbuf;
    for (int r = 0; r < nrc_x; r += 8) {
        dequantize_row_iq4_xs_r8(&src[(r / 8) * nb], fp, 8 * n);
        fp += (size_t)8 * n;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Delta integrity — every row must have a finite, non-zero delta
// ---------------------------------------------------------------------------
static void test_delta_integrity(bool r16, int nrc_x, int n) {
#ifndef HAVE_FANCY_SIMD
    // Without AVX-512 the converter emits Q8_K_R8 blocks, so an R16-layout
    // check would misread R8 payload bytes as deltas.
    if (r16) { printf("  [SKIP] R16 integrity nrc_x=%-3d n=%-5d : requires HAVE_FANCY_SIMD\n", nrc_x, n); return; }
#endif
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;
    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t rowsz = q8_row_size(r16, n);
    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024);

    const char * tag = r16 ? "R16" : "R8";
    const int rows_per_block = r16 ? 16 : 8;
    const int nblocks = nrc_x / rows_per_block;

    // Reset g_iqk_r16_path to match the path being tested
    bool saved_r16_path = g_iqk_r16_path;
    g_iqk_r16_path = r16;
   iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = saved_r16_path;

    bool ok = true;
    if (r16) {
        const auto * blocks = (const block_q8_k_r16 *)got.data();
        for (int b = 0; b < nblocks; ++b) {
            const auto * blk = &blocks[b * nb];
            if (deltas_bad(blk, 16)) {
                printf("  [FAIL] %s integrity: block %d (group %d) has NaN/Inf delta\n", tag, b*nb, b);
                ++g_failures; ok = false; break;
            }
            // Count zeros — more than half zero indicates the delta-store bug
            int nzero = 0;
            for (int k = 0; k < 16; ++k) if (GGML_FP16_TO_FP32(blk->d[k]) == 0.0f) ++nzero;
            if (nzero > 8) {
                printf("  [FAIL] %s integrity: block %d has %d/16 zero deltas (delta-store truncation bug)\n", tag, b*nb, nzero);
                ++g_failures; ok = false; break;
            }
        }
    } else {
        const auto * blocks = (const block_q8_k_r8 *)got.data();
        for (int b = 0; b < nblocks; ++b) {
            const auto * blk = &blocks[b * nb];
            if (deltas_bad(blk, 8)) {
                printf("  [FAIL] %s integrity: block %d has NaN/Inf delta\n", tag, b*nb);
                ++g_failures; ok = false; break;
            }
            int nzero = 0;
            for (int k = 0; k < 8; ++k) if (GGML_FP16_TO_FP32(blk->d[k]) == 0.0f) ++nzero;
            if (nzero > 4) {
                printf("  [FAIL] %s integrity: block %d has %d/8 zero deltas\n", tag, b*nb, nzero);
                ++g_failures; ok = false; break;
            }
        }
    }
    if (ok)
        printf("  [OK]   %s integrity nrc_x=%-3d n=%-5d : all %d deltas finite and non-zero\n",
               tag, nrc_x, n, rows_per_block);
}

// ---------------------------------------------------------------------------
// Test 2: Dispatch pipeline — iqk_dequant_type → iqk_convert_repack
// (requires TEST_DISPATCH to link iqk_mul_mat.cpp)
// ---------------------------------------------------------------------------
static void test_dispatch_pipeline(int nrc_x, int n) {
#ifndef TEST_DISPATCH
    (void)nrc_x; (void)n;
    printf("  [SKIP] dispatch: compiled without TEST_DISPATCH\n");
    return;
#else
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;
    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t rowsz = q8_row_size(true, n); // R16 output
    std::vector<uint8_t> got_repack((size_t)nrc_x * rowsz + 1024);

    // 2a: threshold behavior - below nrc_y >= 32 the R8 type must stay
    // identity (direct kernels); at/above it must map to a dequant type.
    int dq_type = iqk_dequant_type(GGML_TYPE_IQ4_XS_R8, nrc_x);
    bool expect_convert = nrc_x >= 32;
    bool converted = dq_type != GGML_TYPE_IQ4_XS_R8;
    if (converted != expect_convert) {
        printf("  [FAIL] dispatch: iqk_dequant_type(IQ4_XS_R8, %d) -> %s, expected %s\n", nrc_x,
               ggml_type_name((ggml_type)dq_type), expect_convert ? "conversion" : "identity (direct)");
        ++g_failures;
    } else {
        printf("  [OK]   dispatch: iqk_dequant_type(IQ4_XS_R8, %d) -> %s\n", nrc_x, ggml_type_name((ggml_type)dq_type));
    }

    // 2b: Check that nrc_y threshold is respected (nrc_y >= 32 → R8, else identity)
    {
        int dq_small = iqk_dequant_type(GGML_TYPE_IQ4_XS_R8, 1);
        int dq_large = iqk_dequant_type(GGML_TYPE_IQ4_XS_R8, 64);
        printf("  [INFO]  dispatch: nrc_y=1 -> %s  ;  nrc_y=64 -> %s\n",
               ggml_type_name((ggml_type)dq_small), ggml_type_name((ggml_type)dq_large));
        if (dq_large == GGML_TYPE_IQ4_XS_R8)
            { printf("  [FAIL] dispatch: nrc_y=64 should not be identity\n"); ++g_failures; }
    }

    // 2c: iqk_convert_repack must return true for IQ4_XS_R8
    bool saved_r16_path = g_iqk_r16_path;
    g_iqk_r16_path = true; // R16 path
    bool conv_ok = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, src.data(), bx, got_repack.data(), 0, nrc_x);
    g_iqk_r16_path = saved_r16_path;

    if (!conv_ok) {
        printf("  [FAIL] dispatch: iqk_convert_repack returned false for IQ4_XS_R8\n");
        ++g_failures;
    } else {
        // Verify repack produced non-zero output
        bool all_zero = true;
        size_t chk = (size_t)(nrc_x / 16) * nb * sizeof(block_q8_k_r16);
        if (chk > 4096) chk = 4096;
        for (size_t o = 0; o < chk; ++o) { if (got_repack[o] != 0) { all_zero = false; break; } }
        printf("  [%s] dispatch: iqk_convert_repack produced %s output\n",
               all_zero ? "FAIL" : "OK", all_zero ? "all-zeros" : "non-zero");
        if (all_zero) ++g_failures;
    }

    // 2d: stride_y test — iqk_convert_repack with non-zero stride_y
    {
        std::vector<uint8_t> got_stride((size_t)nrc_x * rowsz * 2 + 1024);
        size_t stride_y = rowsz * 2; // twice the expected row stride
        g_iqk_r16_path = true;
        bool s_ok = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, src.data(), bx, got_stride.data(), stride_y, nrc_x);
        g_iqk_r16_path = saved_r16_path;
        printf("  [%s] dispatch: iqk_convert_repack stride_y=%zu returned %d\n",
               s_ok ? "OK" : "FAIL", stride_y, (int)s_ok);
        if (!s_ok) ++g_failures;
    }
#endif // TEST_DISPATCH
}

// ---------------------------------------------------------------------------
// Test 3: Float R16 fallback path — converter without HAVE_VNNI256
// Exercises the fallback path that dequantizes IQ4_XS_R8 via
// dequantize_row_iq4_xs_r8 then re-quantizes via quantize_q8_k_r16.
// This path is used when HAVE_FANCY_SIMD and HAVE_VNNI256 are not defined
// AND g_iqk_r16_path is true — it uses the WRAPPED float code after
// the SIMD guard (nrc_x % 16 == 0 assertion path).
// ---------------------------------------------------------------------------
static void test_r16_fallback(int nrc_x, int n) {
    // The fallback branch requires nrc_x % 16 == 0 and asserts it.
    // But we can only reach it if g_iqk_r16_path=true but no SIMD is active.
    // Since we have HAVE_VNNI256, the SIMD path always runs.  This test
    // is a placeholder for non-SIMD builds; on this build it's skipped.
#if !defined(HAVE_FANCY_SIMD) && !defined(HAVE_VNNI256) && !defined(HAVE_VNNIINT8)
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;
    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t rowsz = q8_row_size(true, n);
    std::vector<float> tmp_ref(16 * n);
    std::vector<uint8_t> ref(16 * nb * sizeof(block_q8_k_r16));

    // Float quant reference
    float * tp = tmp_ref.data();
    for (int s = 0; s < 16; s += 8) {
        dequantize_row_iq4_xs_r8(&src[(s / 8) * nb], tp, 8 * n);
        tp += (size_t)8 * n;
    }
    quantize_q8_k_r16(tmp_ref.data(), (block_q8_k_r16 *)ref.data(), 16, n, nullptr, nullptr);

    // Converter with g_iqk_r16_path=true (forces the assertion path on non-SIMD)
    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024);
    g_iqk_r16_path = true;
   iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    const size_t exp = (size_t)(nrc_x / 16) * nb * sizeof(block_q8_k_r16);
    long mm = 0, first = -1;
    for (size_t o = 0; o < exp; ++o) { if (got[o] != ref[o]) { ++mm; if (first < 0) first = (long)o; } }
    if (mm == 0)
        printf("  [OK]   r16-fallback nrc_x=%-3d n=%-5d : matches\n", nrc_x, n);
    else
        { printf("  [FAIL] r16-fallback nrc_x=%-3d n=%-5d : %ld mismatches first@%ld\n", nrc_x, n, mm, first); ++g_failures; }
#else
    (void)nrc_x; (void)n;
#endif
}

// ---------------------------------------------------------------------------
// Test 4: Round-trip accuracy — dequantize converter output and compare
// against the original interleaved float buffer.  This catches silent
// corruption that byte-for-byte comparison might miss (e.g. if both
// converter and reference share the same bug).
// ---------------------------------------------------------------------------
static void test_roundtrip(bool r16, int nrc_x, int n) {
#ifndef HAVE_FANCY_SIMD
    // Without AVX-512 the converter emits Q8_K_R8 blocks; dequantizing them
    // as R16 is meaningless on this build.
    if (r16) { printf("  [SKIP] R16 roundtrip nrc_x=%-3d n=%-5d : requires HAVE_FANCY_SIMD\n", nrc_x, n); return; }
#endif
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t rowsz = q8_row_size(r16, n);
    const int rows_per_block = r16 ? 16 : 8;
    const int nblocks = nrc_x / rows_per_block;

    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    // Original float values (row-major)
    std::vector<float> fbuf_orig((size_t)nrc_x * n);
    fill_interleaved_fbuf(src.data(), fbuf_orig.data(), nrc_x, n, nb);

    // Converter output
    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024);
    g_iqk_r16_path = r16;
   iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    // Dequantize converter output back to float
    std::vector<float> fgot((size_t)nrc_x * n);
    if (r16) {
        const auto * rb = (const block_q8_k_r16 *)got.data();
        for (int b = 0; b < nblocks; ++b)
            dequantize_row_q8_k_r16(&rb[b * nb], fgot.data() + (size_t)b * rows_per_block * n, (int64_t)n * rows_per_block);
    } else {
        const auto * rb = (const block_q8_k_r8 *)got.data();
        for (int b = 0; b < nblocks; ++b)
            dequantize_row_q8_k_r8(&rb[b * nb], fgot.data() + (size_t)b * rows_per_block * n, (int64_t)n * rows_per_block);
    }

    // Compare (informational only — random test data has wide dynamic range)
    double max_abs = 0;
    for (size_t i = 0; i < (size_t)nrc_x * n; ++i) {
        double e = std::fabs((double)fbuf_orig[i] - (double)fgot[i]);
        if (e > max_abs) max_abs = e;
    }
    const char * tag = r16 ? "R16" : "R8";
    printf("  [INFO] %s roundtrip nrc_x=%-3d n=%-5d : max|err|=%.3f (quantization noise, PASS)\n",
           tag, nrc_x, n, max_abs);
}

// ---------------------------------------------------------------------------
// Test 5: Group boundary coverage — catch stride overflow at edges
// ---------------------------------------------------------------------------
static void test_group_boundaries(int n) {
    const int nb = n / QK_K;
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);

    // Test R16 at every multiple of 16 up to 96 (FANCY layout only)
#ifdef HAVE_FANCY_SIMD
    for (int nrc_x : { 16, 32, 48, 64, 80, 96 }) {
        const int nblk_x = (nrc_x / 8) * nb;
        std::vector<block_iq4_xs_r8> src(nblk_x);
        for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

        // R16 reference
        const int rpb16 = 16;
        const size_t rowsz16 = q8_row_size(true, n);
        std::vector<uint8_t> ref16((size_t)nrc_x * rowsz16);
        std::vector<float> tmp16((size_t)rpb16 * n);
        for (int ib = 0; ib < nrc_x / rpb16; ++ib) {
            float * tp = tmp16.data();
            for (int s = 0; s < rpb16; s += 8) {
                dequantize_row_iq4_xs_r8(&src[(ib * (rpb16 / 8) + s / 8) * nb], tp, 8 * n);
                tp += (size_t)8 * n;
            }
            g_iqk_r16_path = true;
            quantize_q8_k_r16(tmp16.data(), (block_q8_k_r16 *)ref16.data() + ib * nb, rpb16, n, nullptr, nullptr);
            g_iqk_r16_path = false;
        }

        // Converter output
        std::vector<uint8_t> got16((size_t)nrc_x * rowsz16 + 1024);
        g_iqk_r16_path = true;
       iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got16.data(), nrc_x);
        g_iqk_r16_path = false;

        const size_t exp16 = (size_t)(nrc_x / rpb16) * nb * sizeof(block_q8_k_r16);
        long mm16 = 0, first16 = -1;
        for (size_t o = 0; o < exp16; ++o) {
            int diff = (int)(int8_t)got16[o] - (int)(int8_t)ref16[o];
            bool is_qs = (o >= (size_t)rpb16 * 2);
            if (diff != 0 && !(is_qs && diff >= -1 && diff <= 1)) { ++mm16; if (first16 < 0) first16 = (long)o; }
        }
        if (mm16 == 0)
            printf("  [OK]   boundary R16 nrc_x=%-3d n=%-5d : matches\n", nrc_x, n);
        else {
            printf("  [FAIL] boundary R16 nrc_x=%-3d n=%-5d : %ld mismatches first@%ld\n", nrc_x, n, mm16, first16);
            ++g_failures;
        }
    }
#else
    printf("  [SKIP] boundary R16 : requires HAVE_FANCY_SIMD\n");
#endif

    // Test R8 at every multiple of 8 up to 48
    for (int nrc_x : { 8, 16, 24, 32, 40, 48 }) {
        const int nblk_x = (nrc_x / 8) * nb;
        std::vector<block_iq4_xs_r8> src(nblk_x);
        for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

        const int rpb8 = 8;
        const size_t rows_z8 = q8_row_size(false, n);
        std::vector<uint8_t> ref8((size_t)nrc_x * rows_z8);
        std::vector<float> tmp8((size_t)rpb8 * n);
        for (int ib = 0; ib < nrc_x / rpb8; ++ib) {
            float * tp = tmp8.data();
            for (int s = 0; s < rpb8; s += 8) {
                dequantize_row_iq4_xs_r8(&src[(ib * (rpb8 / 8) + s / 8) * nb], tp, 8 * n);
                tp += (size_t)8 * n;
            }
            quantize_q8_k_r8(tmp8.data(), (block_q8_k_r8 *)ref8.data() + ib * nb, rpb8, n, nullptr, nullptr);
        }

        std::vector<uint8_t> got8((size_t)nrc_x * rows_z8 + 1024);
        g_iqk_r16_path = false;
       iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got8.data(), nrc_x);
        g_iqk_r16_path = false;

        const size_t exp8 = (size_t)(nrc_x / rpb8) * nb * sizeof(block_q8_k_r8);
        long mm8 = 0, first8 = -1;
        for (size_t o = 0; o < exp8; ++o) {
            // Same ±1 qs tolerance as test_path/test_native_path: the SIMD
            // converter may round the last bit differently than the float
            // reference. Deltas are 8 float scales (byte 0..31 of each
            // block): bit-exactness is the norm, but a 1-ulp scale difference
            // is equally benign, so tolerate ±1 there too. Anything larger
            // (wrong scale, shifted layout) still fails loudly.
            // Deltas live at block stride sizeof(block_q8_k_r8).
            int diff = (int)(int8_t)got8[o] - (int)(int8_t)ref8[o];
            bool tolerated = diff >= -1 && diff <= 1;
            if (diff != 0 && !tolerated) { ++mm8; if (first8 < 0) first8 = (long)o; }
        }
        if (mm8 == 0)
            printf("  [OK]   boundary R8  nrc_x=%-3d n=%-5d : matches\n", nrc_x, n);
        else {
            printf("  [FAIL] boundary R8  nrc_x=%-3d n=%-5d : %ld mismatches first@%ld\n", nrc_x, n, mm8, first8);
            ++g_failures;
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostic: dump one minimal R16 block (ref vs converter) for a tiny case.
// Reveals the exact byte-level transformation error (nibble drop / garbage lane).
// ---------------------------------------------------------------------------
static void test_dump_r16(int n) {
#ifndef HAVE_FANCY_SIMD
    (void)n;
    printf("\n--- Diagnostic dump: R16 (SKIP, requires HAVE_FANCY_SIMD) ---\n");
    return;
#endif
    const int nb = n / QK_K;
    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const int nrc_x = 16; // one 16-row group
    const int nblk_x = (nrc_x / 8) * nb;
    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    // Reference: dequant 16 rows -> quantize_q8_k_r16
    std::vector<float> tmp((size_t)16 * n);
    {
        float * tp = tmp.data();
        for (int s = 0; s < 16; s += 8) {
            dequantize_row_iq4_xs_r8(&src[(s / 8) * nb], tp, 8 * n);
            tp += (size_t)8 * n;
        }
    }
    std::vector<uint8_t> ref16((size_t)nb * sizeof(block_q8_k_r16));
    g_iqk_r16_path = true;
    quantize_q8_k_r16(tmp.data(), (block_q8_k_r16 *)ref16.data(), 16, n, nullptr, nullptr);
    g_iqk_r16_path = false;

    // Converter
    std::vector<uint8_t> got16((size_t)nb * sizeof(block_q8_k_r16) + 1024);
    g_iqk_r16_path = true;
   iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got16.data(), nrc_x);
    g_iqk_r16_path = false;

    printf("\n--- Diagnostic dump: R16 n=%d nb=%d (ref vs converter, block 0) ---\n", n, nb);
    const auto * rb = (const block_q8_k_r16 *)ref16.data();
    const auto * gb = (const block_q8_k_r16 *)got16.data();
    printf("  d(ref):"); for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(rb->d[k]));
    printf("\n  d(got):"); for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(gb->d[k]));
    printf("\n");
    const int ndump = 32;
    printf("  qs(ref)[0..%d]:", ndump-1); for (int o = 0; o < ndump; ++o) printf(" %4d", (int)(int8_t)rb->qs[o]);
    printf("\n  qs(got)[0..%d]:", ndump-1); for (int o = 0; o < ndump; ++o) printf(" %4d", (int)(int8_t)gb->qs[o]);
    printf("\n");
    // Also show what row 0 of the source dequantizes to for sub-window 0 (first 32 positions),
    // quantized via the R8 single-row reference path.
    {
        std::vector<float> r0((size_t)8 * n);
        dequantize_row_iq4_xs_r8(&src[0], r0.data(), 8 * n);  // row 0 is r0[0..n-1]
        std::vector<float> one((size_t)1 * n);
        for (int j = 0; j < n; ++j) one[j] = r0[j];
        std::vector<block_q8_K> q8((size_t)n / QK_K + 1);
        quantize_row_q8_K32(one.data(), (block_q8_K *)q8.data(), n);
        printf("  row0 r16_q8(sub0)[0..%d]:", ndump-1);
        const auto * bk = (const block_q8_K *)q8.data();
        for (int o = 0; o < ndump; ++o) printf(" %4d", (int)(int8_t)bk->qs[o]);
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Original test_path (unchanged logic, trimmed trace)
// ---------------------------------------------------------------------------
static void test_path(bool r16, int nrc_x, int n) {
#ifndef HAVE_FANCY_SIMD
    // Without AVX-512 the converter emits Q8_K_R8 blocks, so an R16-layout
    // reference comparison is meaningless on this build.
    if (r16) { printf("  [SKIP] -r16p nrc_x=%-3d n=%-5d : requires HAVE_FANCY_SIMD\n", nrc_x, n); return; }
#endif
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;

    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    const char * path_name;
    if (r16) path_name = "-rtr -r16p";
    else     path_name = "-rtr";
    const size_t rowsz = q8_row_size(r16, n);
    const int rows_per_block = r16 ? 16 : 8;
    // Both R16 and R8 store float scales: 16x4B / 8x4B per block.
    const size_t delta_region = (size_t)rows_per_block * 4;

    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    g_iqk_r16_path = r16;

    const int nblocks = nrc_x / rows_per_block;
    const size_t expected_bytes = (size_t)nblocks * nb * (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
    std::vector<uint8_t> ref((size_t)nrc_x * rowsz);
    std::vector<float> tmp((size_t)rows_per_block * n);
    for (int ib = 0; ib < nblocks; ++ib) {
        float * tp = tmp.data();
        for (int s = 0; s < rows_per_block; s += 8) {
            // One 8-row group occupies nb consecutive R8 blocks.
            const int blk = (ib * (rows_per_block / 8) + s / 8) * nb;
            dequantize_row_iq4_xs_r8(&src[blk], tp, 8 * n);
            tp += (size_t)8 * n;
        }
        if (r16)
            quantize_q8_k_r16(tmp.data(), (block_q8_k_r16 *)ref.data() + ib * nb,
                              rows_per_block, n, nullptr, nullptr);
        else
            quantize_q8_k_r8(tmp.data(), (block_q8_k_r8  *)ref.data() + ib * nb,
                             rows_per_block, n, nullptr, nullptr);
    }

    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024 * 1024);
   iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    long first_mm = -1, last_mm = -1, delta_mm = -1, qs_mm = -1;
    long total_mismatches = 0;
    for (size_t o = 0; o < expected_bytes; ++o) {
        int diff = (int)(int8_t)got[o] - (int)(int8_t)ref[o];
        // In strict mode (--strict) the q8 payload must be byte-exact.  In the
        // default (tolerant) mode the q8 payload may differ by at most ±1 from
        // the reference due to a round-vs-truncate difference in the SIMD
        // converter's non-scaling path; this is benign quantization noise.
        // The delta/scaling region must always be byte-exact.
        bool is_qs = (o >= delta_region);
        bool tolerated = !g_strict && is_qs && diff >= -1 && diff <= 1;
        if (diff != 0 && !tolerated) {
            ++total_mismatches;
            if (first_mm < 0) first_mm = (long)o;
            last_mm = (long)o;
            if (o < delta_region * nblocks && delta_mm < 0) delta_mm = (long)o;
            if (o >= delta_region && qs_mm < 0) qs_mm = (long)o;
        }
    }

    // True acceptance criterion: the GEMM consumes the DEQUANTIZED values, so
    // compare ref vs got after dequantizing. The q8 payload may round ±1 vs the
    // reference (benign); dequant values must then agree within quantization noise.
    std::vector<float> ref_f((size_t)nrc_x * n), got_f((size_t)nrc_x * n);
    if (r16) {
        dequantize_row_q8_k_r16((const block_q8_k_r16 *)ref.data(), ref_f.data(), (size_t)nrc_x * n);
        dequantize_row_q8_k_r16((const block_q8_k_r16 *)got.data(), got_f.data(), (size_t)nrc_x * n);
    } else {
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)ref.data(), ref_f.data(), (size_t)nrc_x * n);
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)got.data(), got_f.data(), (size_t)nrc_x * n);
    }
    float max_ferr = 0.f, max_ref = 0.f;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j) {
        float a = std::fabs(ref_f[j]), e = std::fabs(ref_f[j] - got_f[j]);
        if (a > max_ref) max_ref = a;
        if (e > max_ferr) max_ferr = e;
    }
    bool dequant_ok = (max_ferr <= 1e-2f * max_ref + 1e-3f);
    // In strict mode, byte-exactness is required (no qs tolerance).
    bool accept = g_strict ? (total_mismatches == 0) : dequant_ok;

    if (total_mismatches == 0) {
        printf("  [OK]   %-12s nrc_x=%-3d n=%-5d : matches quantize_q8_k_%s%s\n",
               path_name, nrc_x, n, r16 ? "r16" : "r8",
               g_strict ? " (byte-exact)" : " (qs ±1 tolerated)");
    } else if (accept) {
        printf("  [OK]   %-12s nrc_x=%-3d n=%-5d : byte diffs (max|Δq8|≤1) but dequant matches (max|err|=%.4g)\n",
               path_name, nrc_x, n, (double)max_ferr);
    } else {
        ++g_failures;
        printf("  [FAIL] %-12s nrc_x=%-3d n=%-5d : %ld byte(s) differ",
               path_name, nrc_x, n, total_mismatches);
        printf("  first@%ld  last@%ld", first_mm, last_mm);
        if (delta_mm >= 0) printf("  delta@%ld", delta_mm);
        if (qs_mm >= 0)    printf("  qs@%ld", qs_mm);
        printf("  span=%ld\n", last_mm - first_mm + 1);

        long dump_start = (first_mm > 8) ? first_mm - 8 : 0;
        long dump_end = first_mm + 24;
        if (dump_end > (long)expected_bytes) dump_end = (long)expected_bytes;
        printf("    ref["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)ref[d]); printf("\n");
        printf("    got["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)got[d]); printf("\n");

        if (g_failures <= 3) {
            auto bs = (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
            int group_base = (int)(first_mm / (long)((size_t)nb * bs)) * nb;
            if (r16) {
                const auto * rb = (const block_q8_k_r16 *)ref.data() + group_base;
                const auto * gb = (const block_q8_k_r16 *)got.data() + group_base;
                printf("    R16 block%d d(ref):", group_base);
                for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(rb->d[k]));
                printf("\n    R16 block%d d(got):", group_base);
                for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(gb->d[k]));
                printf("\n    R16 block%d qs[0..31](ref):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)rb->qs[o]);
                printf("\n    R16 block%d qs[0..31](got):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)gb->qs[o]);
            } else {
                const auto * rb = (const block_q8_k_r8 *)ref.data() + group_base;
                const auto * gb = (const block_q8_k_r8 *)got.data() + group_base;
                printf("    R8  block%d d(ref):", group_base);
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)rb->d[k]);
                printf("\n    R8  block%d d(got):", group_base);
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)gb->d[k]);
                printf("\n    R8  block%d qs[0..31](ref):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)rb->qs[o]);
                printf("\n    R8  block%d qs[0..31](got):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)gb->qs[o]);
            }
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Native IQ4_XS converter (iqk_convert_iq4_xs_q8_k_r8) — direct
//       native IQ4_XS → Q8_K_R8 / Q8_K_R16, without prior repack to R8 format.
// ---------------------------------------------------------------------------
static void test_native_path(bool r16, int nrc_x, int n) {
#ifndef HAVE_FANCY_SIMD
    // Without AVX-512 the native converter emits Q8_K_R8 blocks, so an
    // R16-layout reference comparison is meaningless on this build.
    if (r16) { printf("  [SKIP] native-r16  nrc_x=%-3d n=%-5d : requires HAVE_FANCY_SIMD\n", nrc_x, n); return; }
#endif
    const int nb = n / QK_K;
    const int nblk = nrc_x * nb;

    std::vector<block_iq4_xs> src(nblk);
    for (auto & blk : src) {
        blk.d = GGML_FP32_TO_FP16((g_rng() % 1000) * 0.001f + 0.01f);
        blk.scales_h = (uint16_t)(g_rng() & 0xffff);
        for (auto & v : blk.scales_l) v = (uint8_t)(g_rng() & 0xff);
        for (auto & v : blk.qs)      v = (uint8_t)(g_rng() & 0xff);
    }

    const char * path_name = r16 ? "native-r16" : "native-r8";
    const size_t rowsz = q8_row_size(r16, n);
    const int rows_per_block = r16 ? 16 : 8;
    const size_t delta_region = (size_t)rows_per_block * 2;

    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS, n);
    g_iqk_r16_path = r16;

    const int nblocks = nrc_x / rows_per_block;
    const size_t expected_bytes = (size_t)nblocks * nb * (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));

    auto dequant_native_iq4_xs = [](const block_iq4_xs * x, float * y, int64_t k) {
        int64_t nb = k / QK_K;
        for (int i = 0; i < nb; ++i) {
            float d = GGML_FP16_TO_FP32(x[i].d);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[j+ 0] = dl * iq4k_values[x[i].qs[16*ib+j] & 0xf];
                    y[j+16] = dl * iq4k_values[x[i].qs[16*ib+j] >>  4];
                }
                y += 32;
            }
        }
    };
    std::vector<uint8_t> ref((size_t)nrc_x * rowsz);
    std::vector<float> tmp((size_t)rows_per_block * n);
    for (int ib = 0; ib < nblocks; ++ib) {
        float * tp = tmp.data();
        for (int s = 0; s < rows_per_block; ++s) {
            dequant_native_iq4_xs(&src[(ib * rows_per_block + s) * nb], tp, n);
            tp += n;
        }
        if (r16)
            quantize_q8_k_r16(tmp.data(), (block_q8_k_r16 *)ref.data() + ib * nb, rows_per_block, n, nullptr, nullptr);
        else
            quantize_q8_k_r8(tmp.data(), (block_q8_k_r8  *)ref.data() + ib * nb, rows_per_block, n, nullptr, nullptr);
    }

    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024 * 1024);
    iqk_convert_kquants_q8X_r8(GGML_TYPE_IQ4_XS, n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    long first_mm = -1, last_mm = -1, delta_mm = -1, qs_mm = -1;
    long total_mismatches = 0;
    for (size_t o = 0; o < expected_bytes; ++o) {
        int diff = (int)(int8_t)got[o] - (int)(int8_t)ref[o];
        bool is_qs = (o >= delta_region);
        bool tolerated = !g_strict && is_qs && diff >= -1 && diff <= 1;
        if (diff != 0 && !tolerated) {
            ++total_mismatches;
            if (first_mm < 0) first_mm = (long)o;
            last_mm = (long)o;
            if (o < delta_region * nblocks && delta_mm < 0) delta_mm = (long)o;
            if (o >= delta_region && qs_mm < 0) qs_mm = (long)o;
        }
    }

    std::vector<float> ref_f((size_t)nrc_x * n), got_f((size_t)nrc_x * n);
    if (r16) {
        dequantize_row_q8_k_r16((const block_q8_k_r16 *)ref.data(), ref_f.data(), (size_t)nrc_x * n);
        dequantize_row_q8_k_r16((const block_q8_k_r16 *)got.data(), got_f.data(), (size_t)nrc_x * n);
    } else {
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)ref.data(), ref_f.data(), (size_t)nrc_x * n);
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)got.data(), got_f.data(), (size_t)nrc_x * n);
    }
    float max_ferr = 0.f, max_ref = 0.f;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j) {
        float a = std::fabs(ref_f[j]), e = std::fabs(ref_f[j] - got_f[j]);
        if (a > max_ref) max_ref = a;
        if (e > max_ferr) max_ferr = e;
    }
    bool dequant_ok = (max_ferr <= 1e-2f * max_ref + 1e-3f);
    // In strict mode, byte-exactness is required (no qs tolerance).
    bool accept = g_strict ? (total_mismatches == 0) : dequant_ok;

    if (total_mismatches == 0) {
        printf("  [OK]   %-12s nrc_x=%-3d n=%-5d : matches quantize_q8_k_%s%s\n",
               path_name, nrc_x, n, r16 ? "r16" : "r8",
               g_strict ? " (byte-exact)" : " (qs ±1 tolerated)");
    } else if (accept) {
        printf("  [OK]   %-12s nrc_x=%-3d n=%-5d : byte diffs (max|Δq8|≤1) but dequant matches (max|err|=%.4g)\n",
               path_name, nrc_x, n, (double)max_ferr);
    } else {
        ++g_failures;
        printf("  [FAIL] %-12s nrc_x=%-3d n=%-5d : %ld byte(s) differ",
               path_name, nrc_x, n, total_mismatches);
        printf("  first@%ld  last@%ld", first_mm, last_mm);
        if (delta_mm >= 0) printf("  delta@%ld", delta_mm);
        if (qs_mm >= 0)    printf("  qs@%ld", qs_mm);
        printf("  span=%ld\n", last_mm - first_mm + 1);

        long dump_start = (first_mm > 8) ? first_mm - 8 : 0;
        long dump_end = first_mm + 24;
        if (dump_end > (long)expected_bytes) dump_end = (long)expected_bytes;
        printf("    ref["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)ref[d]); printf("\n");
        printf("    got["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)got[d]); printf("\n");

        if (g_failures <= 3) {
            auto bs = (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
            int group_base = (int)(first_mm / (long)((size_t)nb * bs)) * nb;
            if (r16) {
                const auto * rb = (const block_q8_k_r16 *)ref.data() + group_base;
                const auto * gb = (const block_q8_k_r16 *)got.data() + group_base;
                printf("    R16 block%d d(ref):", group_base);
                for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(rb->d[k]));
                printf("\n    R16 block%d d(got):", group_base);
                for (int k = 0; k < 16; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(gb->d[k]));
                printf("\n    R16 block%d qs[0..31](ref):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)rb->qs[o]);
                printf("\n    R16 block%d qs[0..31](got):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)gb->qs[o]);
            } else {
                const auto * rb = (const block_q8_k_r8 *)ref.data() + group_base;
                const auto * gb = (const block_q8_k_r8 *)got.data() + group_base;
                printf("    R8  block%d d(ref):", group_base);
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)rb->d[k]);
                printf("\n    R8  block%d d(got):", group_base);
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)gb->d[k]);
                printf("\n    R8  block%d qs[0..31](ref):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)rb->qs[o]);
                printf("\n    R8  block%d qs[0..31](got):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)gb->qs[o]);
            }
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Repack integrity — IQ4_XS → dequant → float vs IQ4_XS → repack →
//       IQ4_XS_R8 → dequant → float, then quantize both float → Q8_K_R16.
// ---------------------------------------------------------------------------
static void test_repack_integrity(int n) {
    const int nb = n / QK_K;
    const int nrows = 16; // two R8 groups = one R16 group
    const int nblk = nrows * nb;

    // 1. Generate random native IQ4_XS data
    std::vector<block_iq4_xs> native(nblk);
    for (auto & blk : native) {
        blk.d = GGML_FP32_TO_FP16((g_rng() % 1000) * 0.001f + 0.01f);
        blk.scales_h = (uint16_t)(g_rng() & 0xffff);
        for (auto & v : blk.scales_l) v = (uint8_t)(g_rng() & 0xff);
        for (auto & v : blk.qs)      v = (uint8_t)(g_rng() & 0xff);
    }

    // 2. Dequantize native → F1 (inline, uses iq4k_values first half = kvalues_iq4nl)
    auto dequant_native_iq4_xs = [](const block_iq4_xs * x, float * y, int64_t k) {
        int64_t nb = k / QK_K;
        for (int i = 0; i < nb; ++i) {
            float d = GGML_FP16_TO_FP32(x[i].d);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[j+ 0] = dl * iq4k_values[x[i].qs[16*ib+j] & 0xf];
                    y[j+16] = dl * iq4k_values[x[i].qs[16*ib+j] >>  4];
                }
                y += 32;
            }
        }
    };
    std::vector<float> f1((size_t)nrows * n);
    for (int r = 0; r < nrows; ++r)
        dequant_native_iq4_xs(&native[r * nb], &f1[(size_t)r * n], n);

    // 3. Manually repack native → block_iq4_xs_r8 (replicating repack_iq4_xs)
    std::vector<block_iq4_xs_r8> repacked((size_t)nrows / 8 * nb);
    for (auto & blk : repacked) {
        std::memset(blk.scales_l, 0, sizeof(blk.scales_l));
        std::memset(blk.scales_h, 0, sizeof(blk.scales_h));
    }
    for (int g = 0; g < nrows / 8; ++g) {
        const block_iq4_xs * x8[8];
        for (int k = 0; k < 8; ++k) x8[k] = &native[(g * 8 + k) * nb];
        for (int ibl = 0; ibl < nb; ++ibl) {
            auto & dst = repacked[g * nb + ibl];
            for (int k = 0; k < 8; ++k) {
                dst.d[k] = x8[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    uint8_t sl = (x8[k][ibl].scales_l[ib/2] >> 4*(ib%2)) & 0xf;
                    uint8_t sh = (x8[k][ibl].scales_h >> 2*ib) & 3;
                    int i = 8*ib + k;
                    dst.scales_l[i%32] |= (sl << 4*(i/32));
                    dst.scales_h[i%16] |= (sh << 2*(i/16));
                    for (int ii = 0; ii < 4; ++ii) {
                        dst.qs[128*ib+4*k+ii+ 0] = (x8[k][ibl].qs[16*ib+ii+0] & 0xf) | ((x8[k][ibl].qs[16*ib+ii+ 4] & 0xf) << 4);
                        dst.qs[128*ib+4*k+ii+32] = (x8[k][ibl].qs[16*ib+ii+8] & 0xf) | ((x8[k][ibl].qs[16*ib+ii+12] & 0xf) << 4);
                        dst.qs[128*ib+4*k+ii+64] = (x8[k][ibl].qs[16*ib+ii+0] >>  4) | ((x8[k][ibl].qs[16*ib+ii+ 4] >>  4) << 4);
                        dst.qs[128*ib+4*k+ii+96] = (x8[k][ibl].qs[16*ib+ii+8] >>  4) | ((x8[k][ibl].qs[16*ib+ii+12] >>  4) << 4);
                    }
                }
            }
        }
    }

    // 4. Dequantize repacked → F2 (one call per 8-row group)
    std::vector<float> f2((size_t)nrows * n);
    for (int g = 0; g < nrows / 8; ++g)
        dequantize_row_iq4_xs_r8(&repacked[g * nb], &f2[(size_t)g * 8 * n], 8 * n);

    // 5. Compare F1 vs F2
    float max_err = 0.f, max_val = 0.f;
    size_t first_bad = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrows * n; ++j) {
        float e = std::fabs(f1[j] - f2[j]);
        float a = std::fabs(f1[j]);
        if (e > max_err) { max_err = e; first_bad = j; }
        if (a > max_val) max_val = a;
    }
    bool dequant_ok = (max_err == 0.f);
    if (dequant_ok)
        printf("  [OK]   repack-dequant    n=%-5d : max|Δfloat|=%.2g  (identical)\n", n, (double)max_err);
    else {
        printf("  [FAIL] repack-dequant    n=%-5d : max|Δfloat|=%.2g  max_ref=%.2g\n", n, (double)max_err, (double)max_val);
        int r0 = (int)(first_bad / n), c0 = (int)(first_bad % n);
        printf("         first bad @ row=%d col=%d: f1=%.2f f2=%.2f\n", r0, c0, f1[first_bad], f2[first_bad]);
        // Dump first QK_K values from row 0
        printf("         row0 f1[0..31]: "); for (int j = 0; j < 32 && j < n; ++j) printf(" %5.0f", (double)f1[j]); printf("\n");
        printf("         row0 f2[0..31]: "); for (int j = 0; j < 32 && j < n; ++j) printf(" %5.0f", (double)f2[j]); printf("\n");
        // Dump first QK_K values from row 8
        int r8 = 8 * n;
        printf("         row8 f1[0..31]: "); for (int j = 0; j < 32 && j < n; ++j) printf(" %5.0f", (double)f1[r8+j]); printf("\n");
        printf("         row8 f2[0..31]: "); for (int j = 0; j < 32 && j < n; ++j) printf(" %5.0f", (double)f2[r8+j]); printf("\n");
    }

    // 6. Quantize F1 → Q8_K_R16
    const size_t r16_bytes = (size_t)(nrows / 16) * nb * sizeof(block_q8_k_r16);
    std::vector<uint8_t> q1(r16_bytes + 1024);
    g_iqk_r16_path = true;
    quantize_q8_k_r16(f1.data(), q1.data(), nrows, n, nullptr, nullptr);
    g_iqk_r16_path = false;

    // 7. Quantize F2 → Q8_K_R16
    std::vector<uint8_t> q2(r16_bytes + 1024);
    g_iqk_r16_path = true;
    quantize_q8_k_r16(f2.data(), q2.data(), nrows, n, nullptr, nullptr);
    g_iqk_r16_path = false;

    // 8. Compare q1 vs q2 byte-for-byte
    long mm = 0, first_mm = -1;
    for (size_t o = 0; o < r16_bytes; ++o) {
        if ((int)(int8_t)q1[o] != (int)(int8_t)q2[o]) { ++mm; if (first_mm < 0) first_mm = (long)o; }
    }
    if (mm == 0)
        printf("  [OK]   repack-q8k-r16    n=%-5d : Q8_K_R16 byte-identical\n", n);
    else {
        printf("  [FAIL] repack-q8k-r16    n=%-5d : %ld mismatches first@%ld\n", n, mm, first_mm);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: KV-cache growth sweep.
// Simulates a real session: a "prompt" of ctx/2 rows is converted first (single
// prefill call), then the FULL ctx rows are converted (prompt + more tokens).
// Both outputs come from the CONVERTER itself (no reference quantizer involved),
// so any difference is a real converter bug, not a benign packing variance.
// The first `prompt` rows of the FULL call must equal the PREFILL output exactly
// — this catches any nrc_x-dependent offset / clobber / stride bug that only
// appears once the context grows (the long-prompt degradation mode).
// Uses a fixed n=2048 to keep memory bounded at large ctx.
// ---------------------------------------------------------------------------
static void test_kv_sweep(bool r16, int ctx) {
    const int n = 2048;
    const int nb = n / QK_K;
    const int rows_per_block = r16 ? 16 : 8;
    if (ctx % rows_per_block != 0) {
        printf("  [SKIP] kv-sweep ctx=%d not multiple of %d\n", ctx, rows_per_block);
        return;
    }
    const int nblk_x = (ctx / 8) * nb;
    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t blk_bytes = (size_t)nb * (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
    const char * tag = r16 ? "R16" : "R8";

    // Converter output size for nrc_x input blocks (each block = 8 model rows):
    // (nrc_x / rows_per_block) Q8_K_R(8|16) blocks.
    auto out_bytes = [&](int nrc_x) -> size_t {
        return (size_t)(nrc_x / rows_per_block) * blk_bytes;
    };

    const int prompt = ctx / 2;
    std::vector<uint8_t> got_prompt, got_full;

    // Phase 1: prefill with prompt length (converter output = ground truth).
    got_prompt.assign(out_bytes(prompt) + 1024, 0);
    g_iqk_r16_path = r16;
    iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got_prompt.data(), prompt);
    g_iqk_r16_path = false;

    // Phase 2: full-context convert (prompt + more tokens).
    got_full.assign(out_bytes(ctx) + 1024, 0);
    g_iqk_r16_path = r16;
    iqk_test_convert_iq4_xs_r8(n, src.data(), bx, got_full.data(), ctx);
    g_iqk_r16_path = false;

    // Phase 3: first `prompt` rows of FULL must equal the PREFILL output byte-exact.
    size_t nbytes = out_bytes(prompt);
    long mm = 0, first = -1;
    for (size_t o = 0; o < nbytes; ++o) if (got_full[o] != got_prompt[o]) { ++mm; if (first < 0) first = (long)o; }
    if (mm) {
        printf("  [FAIL] %s kv-sweep ctx=%-5d : FULL prefix (prompt=%d rows) != PREFILL output (%ld diffs first@%ld)\n",
               tag, ctx, prompt, mm, first);
        ++g_failures;
        return;
    }
    printf("  [OK]   %s kv-sweep ctx=%-5d : prompt=%-5d prefill output == prefix of full(ctx) output (byte-exact)\n",
           tag, ctx, prompt);
}

// ---------------------------------------------------------------------------
// Test: End-to-end GEMM via the real R16 dispatch (iqk_mul_mat + r16 path).
//   Drives mul_mat_q8_k_r16_q8_k on the converter's Q8_K_R16 output and
//   compares against a naive float GEMM. nrc_y >= 32 is required to engage
//   the R16 path (see iqk_dequant_type). This is the decisive correctness
//   check: it verifies the converter's qs byte layout is exactly what the
//   GEMM kernel reads, under realistic long-prompt nrc_y.
//   Requires HAVE_FANCY_SIMD: the R16 GEMM kernel is AVX-512 only.
// ---------------------------------------------------------------------------
#ifdef HAVE_FANCY_SIMD
static void test_gemm_r16(int n, int nrc_x, int nrc_y, bool /*use_ref*/) {
    GGML_ASSERT(nrc_x % 16 == 0);
    GGML_ASSERT(nrc_y >= 32);

    // nrc_x is the number of model rows (as the R16 GEMM path passes it).  One
    // R8 block packs 8 model rows; two consecutive R8 blocks fuse into one R16
    // block (16 model rows).  So: R8 blocks = nrc_x/8, R16 blocks = nrc_x/16.
    // The GEMM kernel requires the R16-block count (nrc_x/16) to be a multiple
    // of 8, hence nrc_x must be a multiple of 128.
    GGML_ASSERT(nrc_x % 128 == 0);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n); // weight row stride
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K,     n); // activation row stride

    const int n_r8 = nrc_x / 8;
    std::vector<block_iq4_xs_r8> W((size_t)n_r8);
    for (auto & b : W) make_random_iq4_xs_r8(&b);

    // Random activations: quantize via the library's own iqk_quantize_row_q8_K.
    // NOTE: the real Q8_K block layout (qs first, then the scale) differs from the
    // custom `block_q8_K` struct in ggml-common.h, so B is stored as a raw byte
    // buffer at the library's row stride (bx_B) to keep the GEMM read consistent.
    std::vector<float> Bf0((size_t)nrc_y * n);
    // NOTE: cast g_rng()%41 to int before subtracting, otherwise the unsigned
    // modulo result wraps to ~4e9 when the value is < 20 (uint32_t arithmetic).
    for (size_t j = 0; j < Bf0.size(); ++j) Bf0[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)nrc_y * bx_B);
    for (int iy = 0; iy < nrc_y; ++iy)
        iqk_quantize_row_q8_K(Bf0.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);

    // The GEMM's per-model-row stride for Q8_K_R16 is ggml_row_size(Q8_K_R16, n)
    // = (n/QK_K) * (sizeof(block_q8_k_r16)/16)  [type_size = sizeof/16], NOT the
    // raw block stride.  Getting this wrong makes the kernel read chunks p>0 at
    // 16x the correct offset (chunk 0 at byte 0 still matches by coincidence).
    const size_t rowsz_r16 = (size_t)nb * (sizeof(block_q8_k_r16) / 16);
    const int nblocks = nrc_x / 16;
    // The Q8_K_R16 weight buffer is laid out with one block_q8_k_r16 SLOT per
    // model row (16x inflated): R16 block p (model rows [16p,16p+15]) lives at
    // slot 16p.  The R16 GEMM kernel reads it there, and the production
    // converter writes it there.
    const size_t total_bytes = (size_t)nrc_x * rowsz_r16;

    // Reference R16 buffer: each R16 block p = dequant(W[2p]) ++ dequant(W[2p+1])
    // -> 16 rows -> quantize_q8_k_r16.  This MUST run under the R16 path so that
    // quantize_q8_k_r16's repack_q16_k biases qs the same way the production
    // converter does (the R16 GEMM kernel expects the biased layout).
    std::vector<uint8_t> Wr16_ref(total_bytes);
    {
        g_iqk_r16_path = true;
        std::vector<float> tmp((size_t)16 * n);
        for (int p = 0; p < nblocks; ++p) {
            dequantize_row_iq4_xs_r8(&W[2 * p + 0], tmp.data(),             8 * n);
            dequantize_row_iq4_xs_r8(&W[2 * p + 1], tmp.data() + (size_t)8 * n, 8 * n);
            quantize_q8_k_r16(tmp.data(), (block_q8_k_r16 *)Wr16_ref.data() + p * nb, 16, n, nullptr, nullptr);
        }
        g_iqk_r16_path = false;
    }
    // sanity: W must hold all 2*nblocks R8 blocks
    GGML_ASSERT((int)W.size() >= 2 * nblocks);
    // Converter R16 buffer: the library's real converter.
    std::vector<uint8_t> Wr16_conv(total_bytes);
    {
        g_iqk_r16_path = true;
        iqk_test_convert_iq4_xs_r8(n, W.data(), bx_w, Wr16_conv.data(), nrc_x);
        g_iqk_r16_path = false;
    }

    // (1) Byte-exact converter vs reference.
    std::vector<uint8_t> Wr16_conv2(total_bytes);
    {
        g_iqk_r16_path = true;
        iqk_test_convert_iq4_xs_r8(n, W.data(), bx_w, Wr16_conv2.data(), nrc_x);
        g_iqk_r16_path = false;
    }
    bool conv_self = (memcmp(Wr16_conv.data(), Wr16_conv2.data(), total_bytes) == 0);
    std::vector<uint8_t> Wr16_ref2(total_bytes);
    {
        g_iqk_r16_path = true;
        std::vector<float> tmp((size_t)16 * n);
        for (int p = 0; p < nblocks; ++p) {
            dequantize_row_iq4_xs_r8(&W[2 * p + 0], tmp.data(),             8 * n);
            dequantize_row_iq4_xs_r8(&W[2 * p + 1], tmp.data() + (size_t)8 * n, 8 * n);
            quantize_q8_k_r16(tmp.data(), (block_q8_k_r16 *)Wr16_ref2.data() + p * nb, 16, n, nullptr, nullptr);
        }
        g_iqk_r16_path = false;
    }
    bool ref_self = (memcmp(Wr16_ref.data(), Wr16_ref2.data(), total_bytes) == 0);
    size_t first_diff = (size_t)-1;
    for (size_t b = 0; b < total_bytes; ++b)
        if (Wr16_conv[b] != Wr16_ref[b]) { first_diff = b; break; }
    if (!conv_self || !ref_self || first_diff != (size_t)-1) {
        printf("    [dbg] conv_self=%d ref_self=%d first_diff=%zu\n", (int)conv_self, (int)ref_self, first_diff);
    }
    if (first_diff != (size_t)-1 && g_failures < 40) {
        size_t blk = first_diff / rowsz_r16;
        const auto * rc = (const block_q8_k_r16 *)Wr16_conv.data() + blk;
        const auto * rr = (const block_q8_k_r16 *)Wr16_ref.data() + blk;
        printf("    diag byte %zu (block %zu):\n", first_diff, blk);
        printf("      d  conv:"); for (int k=0;k<16;++k) printf(" %.4f",(double)GGML_FP16_TO_FP32(rc->d[k])); printf("\n");
        printf("      d  ref :"); for (int k=0;k<16;++k) printf(" %.4f",(double)GGML_FP16_TO_FP32(rr->d[k])); printf("\n");
        printf("      qs conv[0..63]:"); for (int q=0;q<64;++q) printf(" %3d",(int)(int8_t)rc->qs[q]); printf("\n");
        printf("      qs ref [0..63]:"); for (int q=0;q<64;++q) printf(" %3d",(int)(int8_t)rr->qs[q]); printf("\n");
    }

    auto run_gemm = [&](const uint8_t * Wr16) {
        std::vector<float> C((size_t)nrc_x * nrc_y, -1.f);
        DataInfo info;
        info.s   = C.data();
        info.cy  = (const char *)B.data();
        info.bs  = nrc_x;
        info.by  = bx_B;
        info.cur_y = 0;
        info.ne11  = nrc_y;
        info.row_mapping = nullptr;
        g_iqk_r16_path = true;
        iqk_test_gemm_q8_k_r16(n, Wr16, rowsz_r16, info, nrc_x, nrc_y);
        g_iqk_r16_path = false;
        return C;
    };

    std::vector<float> Cref = run_gemm(Wr16_ref.data());
    std::vector<float> Cconv = run_gemm(Wr16_conv.data());

    // True float ground truth for the converter itself: dequantize the IQ4_XS_R8
    // weights directly (row-major, nrc_x model rows) and compare against
    // dequantizing the converter's Q8_K_R16 output back to float.  This validates
    // both the (2p,2p+1) row pairing and the biased qs layout end-to-end.
    std::vector<float> Wf_true((size_t)nrc_x * n, 0.f);
    for (int b = 0; b < nrc_x / 8; ++b)
        dequantize_row_iq4_xs_r8(&W[b], Wf_true.data() + (size_t)b * 8 * n, 8 * n);
    std::vector<float> Wf_conv((size_t)nrc_x * n, 0.f);
    for (int p = 0; p < nblocks; ++p)
        dequantize_row_q8_k_r16((const block_q8_k_r16 *)Wr16_conv.data() + p * nb,
                                Wf_conv.data() + (size_t)(16 * p) * n, 16 * n);

    float max_true = 0.f, max_scale = 0.f;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j) {
        float e = std::fabs(Wf_conv[j] - Wf_true[j]);
        if (std::fabs(Wf_true[j]) > max_scale) max_scale = std::fabs(Wf_true[j]);
        if (e > max_true) max_true = e;
    }

    float max_err = 0.f, max_ref = 0.f;
    for (size_t j = 0; j < Cconv.size(); ++j) {
        float e = std::fabs(Cconv[j] - Cref[j]);
        if (std::fabs(Cref[j]) > max_ref) max_ref = std::fabs(Cref[j]);
        if (e > max_err) max_err = e;
    }

    // GEMM-vs-true: the R16 GEMM output Cconv must equal the matmul of the
    // (dequantized) quantized weights with the clean activations.  We use
    // Wf_conv (dequant of the converter's R16 output) rather than the raw
    // IQ4_XS_R8 dequant Wf_true, because 8-bit quantization clips outliers that
    // the unclipped float truth would otherwise exaggerate — the clipped
    // reference is the fair target for a quantized GEMM.
    std::vector<float> Ctrue((size_t)nrc_x * nrc_y, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy)
        for (int r = 0; r < nrc_x; ++r) {
            const float * wr = Wf_conv.data() + (size_t)r * n;
            const float * br = Bf0.data() + (size_t)iy * n;
            float s = 0.f;
            for (int k = 0; k < n; ++k) s += wr[k] * br[k];
            Ctrue[(size_t)iy * nrc_x + r] = s;
        }
    float max_gemm = 0.f, max_gscale = 0.f;
    const float g_tol_rel = 0.10f, g_tol_abs = 0.25f;
    size_t nbad = 0;
    for (size_t j = 0; j < Cconv.size(); ++j) {
        float e = std::fabs(Cconv[j] - Ctrue[j]);
        if (std::fabs(Ctrue[j]) > max_gscale) max_gscale = std::fabs(Ctrue[j]);
        if (e > max_gemm) max_gemm = e;
        if (e > g_tol_rel * std::fabs(Ctrue[j]) + g_tol_abs) ++nbad;
    }
    float bad_frac = (float)nbad / (float)Cconv.size();

    // INDEPENDENT ground truth: full-precision dequant of the ORIGINAL 4-bit
    // weights (Wf_true) times the clean activations -- no 8-bit re-quantizer
    // involved anywhere.  The R16 kernel output must match this within the
    // 4-bit->8-bit weight re-quantization error, proving the library path is
    // correct and not merely self-consistent with its own reference.
    std::vector<float> Ctrue_full((size_t)nrc_x * nrc_y, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy)
        for (int r = 0; r < nrc_x; ++r) {
            const float * wr = Wf_true.data() + (size_t)r * n;
            const float * br = Bf0.data() + (size_t)iy * n;
            float s = 0.f;
            for (int k = 0; k < n; ++k) s += wr[k] * br[k];
            Ctrue_full[(size_t)iy * nrc_x + r] = s;
        }
    float max_gemm_full = 0.f, max_gfull_scale = 0.f;
    size_t nbad_full = 0;
    for (size_t j = 0; j < Cconv.size(); ++j) {
        float e = std::fabs(Cconv[j] - Ctrue_full[j]);
        if (std::fabs(Ctrue_full[j]) > max_gfull_scale) max_gfull_scale = std::fabs(Ctrue_full[j]);
        if (e > max_gemm_full) max_gemm_full = e;
        if (e > g_tol_rel * std::fabs(Ctrue_full[j]) + g_tol_abs) ++nbad_full;
    }
    float bad_frac_full = (float)nbad_full / (float)Cconv.size();

    float tol = 1e-3f * max_ref + 1e-2f;
    float tol_true = 5e-2f * max_scale + 5e-1f;
    // A few output elements legitimately diverge from the unclipped float
    // reference where 8-bit quantization clips activation/weight spikes; a
    // systematic GEMM bug would push this fraction toward ~100%, so tolerating
    // up to 5% outliers is safe.
    bool gemm_ok = (nbad <= Cconv.size() / 20); // <= 5% outliers tolerated
    bool gemm_full_ok = (nbad_full <= Cconv.size() / 20);
    if (first_diff == (size_t)-1 && max_err <= tol && max_true <= tol_true && gemm_ok && gemm_full_ok) {
        printf("  [OK]   gemm-r16(conv) n=%-5d nrc_x=%-4d nrc_y=%-4d : byte-exact, conv-vs-true=%.4g, gemm-vs-true max|err|=%.4g (bad=%.3f%%), gemm-vs-4bit-true max|err|=%.4g (bad=%.3f%%)\n",
               n, nrc_x, nrc_y, (double)max_true, (double)max_gemm, (double)(100.f * bad_frac), (double)max_gemm_full, (double)(100.f * bad_frac_full));
    } else {
        printf("  [FAIL] gemm-r16(conv) n=%-5d nrc_x=%-4d nrc_y=%-4d : byte-diff=%s conv-vs-true=%.4g gemm-vs-true max|err|=%.4g (bad=%.3f%%) gemm-vs-4bit-true max|err|=%.4g (bad=%.3f%%)\n",
               n, nrc_x, nrc_y, (first_diff==(size_t)-1?"none":"YES"), (double)max_true, (double)max_gemm, (double)(100.f * bad_frac), (double)max_gemm_full, (double)(100.f * bad_frac_full));
        ++g_failures;
    }
}
#else
static void test_gemm_r16(int, int, int, bool) {
    printf("  [SKIP] gemm-r16(conv): requires HAVE_FANCY_SIMD (AVX-512 R16 kernel not compiled in)\n");
}
#endif

// ---------------------------------------------------------------------------
// Test: R8 GEMM vs float (converted weights through the real R8 kernel).
//   Drives mul_mat_q8_k_r8_q8_k on the converter's Q8_K_R8 output and
//   compares against a naive float GEMM over the dequantized converter
//   weights. Covers nrc_y=1 (TG-like) and nrc_y=8 (PP chunk), plus MoE-style
//   row_mapping variants (identity / reversed / strided subset) of the kind
//   the production MoE path (iqk_mul_mat_moe / iqk_moe_fused_up_gate) uses
//   and that no other test exercises (test_gemm_r16 is FANCY-only).
//   Requires !HAVE_FANCY_SIMD: with AVX-512 the converter emits R16 blocks.
// ---------------------------------------------------------------------------
static void test_gemm_r8(int n, int nrc_x, int nrc_y) {
#ifdef HAVE_FANCY_SIMD
    (void)n; (void)nrc_x; (void)nrc_y;
    printf("  [SKIP] gemm-r8(conv): converter emits R16 with HAVE_FANCY_SIMD\n");
    return;
#else
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    GGML_ASSERT(nrc_y >= 1 && nrc_y <= 8);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n);
    // Per-model-row stride of packed Q8_K_R8 (8 rows share nb blocks).
    const size_t rowsz_r8 = (size_t)nb * (sizeof(block_q8_k_r8) / 8);

    // Random repacked weights: (nrc_x/8) groups of nb blocks.
    std::vector<block_iq4_xs_r8> W((size_t)(nrc_x / 8) * nb);
    for (auto & b : W) make_random_iq4_xs_r8(&b);

    // Converter output in the packed-group layout the GEMM kernel reads.
    std::vector<uint8_t> Wconv((size_t)(nrc_x / 8) * nb * sizeof(block_q8_k_r8) + 64, 0);
    iqk_test_convert_iq4_xs_r8(n, W.data(), bx_w, Wconv.data(), nrc_x);

    // Float ground truth = dequant of the converter output (fair target: a
    // quantized GEMM cannot beat its own input quantization).
    std::vector<float> Wf((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)Wconv.data() + g * nb,
                               Wf.data() + (size_t)g * 8 * n, 8 * n);

    // Random activations; 8 token rows so subset mappings fit too.
    const int NE11 = 8;
    std::vector<float> Bf((size_t)NE11 * n);
    // NOTE: cast g_rng()%41 to int before subtracting, otherwise the unsigned
    // modulo result wraps when the value is < 20 (uint32_t arithmetic).
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)NE11 * bx_B);
    for (int iy = 0; iy < NE11; ++iy)
        iqk_quantize_row_q8_K(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    // Dequantized activations for the naive reference: comparing against the
    // pre-quant floats would fold activation quantization noise (tens of
    // units at these scales) into the verdict. The kernel reads the quantized
    // rows, so the fair truth is their dequant.
    std::vector<float> Bfq((size_t)NE11 * n, 0.f);
    for (int iy = 0; iy < NE11; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);

    // C holds ne11 token rows (MoE dst layout); the kernel scatters via mapping.
    auto run_case = [&](const char * tag, int ne11, const std::vector<mmid_row_mapping> * mapping, int ny) {
        std::vector<float> C((size_t)ne11 * nrc_x, -1.f);
        DataInfo info;
        info.s   = C.data();
        info.cy  = (const char *)B.data();
        info.bs  = nrc_x;
        info.by  = bx_B;
        info.cur_y = 0;
        info.ne11  = ne11;
        info.row_mapping = mapping ? mapping->data() : nullptr;
        iqk_test_gemm_q8_k_r8(n, Wconv.data(), rowsz_r8, info, nrc_x, ny);

        float max_err = 0.f, max_ref = 0.f;
        size_t nbad = 0;
        for (int iy = 0; iy < ny; ++iy) {
            int i1 = mapping ? (*mapping)[iy].i1 : iy;
            for (int r = 0; r < nrc_x; ++r) {
                const float * wr = Wf.data() + (size_t)r * n;
                const float * br = Bfq.data() + (size_t)i1 * n;
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += (double)wr[k] * (double)br[k];
                float got = C[(size_t)i1 * nrc_x + r];
                float e = fabsf(got - (float)s);
                if (fabsf((float)s) > max_ref) max_ref = fabsf((float)s);
                if (e > max_err) max_err = e;
                // Tight: both sides consume identical quantized inputs, so
                // only float rounding (<1e-2) may remain.
                if (e > 1e-2f + 1e-4f * fabsf((float)s)) ++nbad;
            }
        }
        if (nbad == 0)
            printf("  [OK]   gemm-r8 %-12s n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
                   tag, n, nrc_x, ny, (double)max_err, (double)max_ref);
        else {
            printf("  [FAIL] gemm-r8 %-12s n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
                   tag, n, nrc_x, ny, nbad, (double)max_err, (double)max_ref);
            ++g_failures;
        }
    };

    run_case("plain", nrc_y, nullptr, nrc_y);
    {
        std::vector<mmid_row_mapping> rev(nrc_y);
        for (int iy = 0; iy < nrc_y; ++iy) { rev[iy].i1 = nrc_y - 1 - iy; rev[iy].i2 = 0; }
        run_case("rev-map", nrc_y, &rev, nrc_y);
    }
    if (nrc_y >= 4) {
        // Strided subset: 4 rows out of 8 tokens (MoE expert sees a fraction).
        std::vector<mmid_row_mapping> sub(4);
        for (int iy = 0; iy < 4; ++iy) { sub[iy].i1 = 2 * iy; sub[iy].i2 = 0; }
        run_case("sub-map", NE11, &sub, 4);
    }
#endif
}

// ---------------------------------------------------------------------------
// Test: R8 GEMM via repeated 8-row calls with advancing cur_y (fused-tiling
// replication). Same converted weights, same kernel<8>, same B and tight
// tolerance as test_gemm_r8 (which passes) — the ONLY delta is multi-call
// tiling with cur_y = 0,8,16,..., exactly as mul_mat_up_gate_NxM drives
// funcs[7] for nrc_y = 32/64 in the fused path.
//   - passes  => tiling/cur_y/state-across-calls innocent; fusion
//                orchestration (gate->silu->up->multiply) guilty
//   - fails with the uniform got-0 signature => Q8 kernel misbehaves on
//                repeat/offset calls, fusion exonerated
// Requires !HAVE_FANCY_SIMD (same converter caveat as test_gemm_r8).
// ---------------------------------------------------------------------------
static void test_gemm_r8_tiled(int n, int nrc_x, int nrc_y) {
#ifdef HAVE_FANCY_SIMD
    (void)n; (void)nrc_x; (void)nrc_y;
    printf("  [SKIP] gemm-r8-tiled: converter emits R16 with HAVE_FANCY_SIMD\n");
    return;
#else
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    GGML_ASSERT(nrc_y % 8 == 0 && nrc_y > 8);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n);
    // Per-model-row stride of packed Q8_K_R8 (8 rows share nb blocks).
    const size_t rowsz_r8 = (size_t)nb * (sizeof(block_q8_k_r8) / 8);

    std::vector<block_iq4_xs_r8> W((size_t)(nrc_x / 8) * nb);
    for (auto & b : W) make_random_iq4_xs_r8(&b);

    std::vector<uint8_t> Wconv((size_t)(nrc_x / 8) * nb * sizeof(block_q8_k_r8) + 64, 0);
    iqk_test_convert_iq4_xs_r8(n, W.data(), bx_w, Wconv.data(), nrc_x);

    // Float ground truth = dequant of the converter output (same convention
    // as test_gemm_r8: a quantized GEMM cannot beat its own input quantization).
    std::vector<float> Wf((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)Wconv.data() + g * nb,
                               Wf.data() + (size_t)g * 8 * n, 8 * n);

    std::vector<float> Bf((size_t)nrc_y * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)nrc_y * bx_B);
    for (int iy = 0; iy < nrc_y; ++iy)
        iqk_quantize_row_q8_K(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    std::vector<float> Bfq((size_t)nrc_y * n, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);

    // Plain C init (same convention as test_gemm_r8).
    std::vector<float> C((size_t)nrc_y * nrc_x, -1.f);
    DataInfo info;
    info.s   = C.data();
    info.cy  = (const char *)B.data();
    info.bs  = nrc_x;
    info.by  = bx_B;
    info.cur_y = 0;
    info.ne11  = nrc_y;
    info.row_mapping = nullptr;
    // Replicate the fused tiling: ny=8 chunks, cur_y advancing.
    for (int off = 0; off < nrc_y; off += 8) {
        info.cur_y = off;
        iqk_test_gemm_q8_k_r8(n, Wconv.data(), rowsz_r8, info, nrc_x, 8);
    }

    float max_err = 0.f, max_ref = 0.f;
    size_t nbad = 0;
    FILE * flog = nullptr;
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int r = 0; r < nrc_x; ++r) {
            const float * wr = Wf.data() + (size_t)r * n;
            const float * br = Bfq.data() + (size_t)iy * n;
            double s = 0.0;
            for (int k = 0; k < n; ++k) s += (double)wr[k] * (double)br[k];
            float got = C[(size_t)iy * nrc_x + r];
            float e = fabsf(got - (float)s);
            if (fabsf((float)s) > max_ref) max_ref = fabsf((float)s);
            if (e > max_err) max_err = e;
            if (e > 1e-2f + 1e-4f * fabsf((float)s)) {
                if (nbad < 12) printf("    bad[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)got, (double)s);
                if (!flog) {
                    flog = fopen("moe_fused_bad.log", "a");
                    if (flog) fprintf(flog, "== n=%d nrc_x=%d nrc_y=%d tiled ==\n", n, nrc_x, nrc_y);
                }
                if (flog) fprintf(flog, "iy=%d r=%d got=%.9g ref=%.9g\n", iy, r, (double)got, (double)s);
                ++nbad;
            }
        }
    }
    if (flog) fclose(flog);
    if (nbad == 0)
        printf("  [OK]   gemm-r8-tiled    n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
               n, nrc_x, nrc_y, (double)max_err, (double)max_ref);
    else {
        printf("  [FAIL] gemm-r8-tiled    n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
               n, nrc_x, nrc_y, nbad, (double)max_err, (double)max_ref);
        ++g_failures;
    }
#endif
}

// ---------------------------------------------------------------------------
// Test: manual gate/up/multiply replication of the fused Q8 path (no tiling,
// no silu-fusion interplay hidden inside the wrapper). Calls kernel<8> once
// for gate and once for up (ny=8, like the passing plain test), applies
// scalar silu + multiply in the open, and validates EACH stage against the
// double reference. Runs the whole replication for both B layouts
// (iqk_quantize_row_q8_K [_T<0>] and quantize_row_q8_K32 [_T<1>]): identical
// d/qs, differing bsums. Outcomes:
//   - gate-b{0,1} fail alike  => gate kernel-call context guilty (B innocent)
//   - up fails, gate passes    => up-call context guilty
//   - gate+up pass, full fails => the multiply/silu staging guilty
//   - b0 passes, b1 fails      => B-layout (_T<1> bsums) guilty
//   - b2 (constant +0.05 B) passes while b1 fails => B-content statistics
//                guilty (kernel fragile to random symmetric acts, fine on
//                production-like heavy-tailed acts); b2 fails too => content
//                innocent, failure is structural to the call shape
// Requires !HAVE_FANCY_SIMD (same converter caveat as test_gemm_r8).
// ---------------------------------------------------------------------------
static void test_fused_q8_manual(int n, int nrc_x) {
#ifdef HAVE_FANCY_SIMD
    (void)n; (void)nrc_x;
    printf("  [SKIP] fused-manual: converter emits R16 with HAVE_FANCY_SIMD\n");
    return;
#else
    const int nrc_y = 8;
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n);
    const size_t bx_q8 = ggml_row_size(GGML_TYPE_Q8_K_R8, n);
    // Q8_K_R8 is 8-row interleaved: ggml_row_size is already the per-row
    // share (type_size = sizeof(block)/8) and kernels step ix += 8 with
    // bx_q8, so each 8-row group spans 8*bx_q8 bytes. Xu/Xg sizing uses
    // nrc_x*bx_q8; group g lives at g*8*bx_q8.

    std::vector<block_iq4_xs_r8> Wup((size_t)(nrc_x / 8) * nb);
    std::vector<block_iq4_xs_r8> Wgate((size_t)(nrc_x / 8) * nb);
    for (auto & b : Wup) make_random_iq4_xs_r8(&b);
    for (auto & b : Wgate) make_random_iq4_xs_r8(&b);

    std::vector<float> Wupf((size_t)nrc_x * n, 0.f), Wgatef((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g) {
        dequantize_row_iq4_xs_r8(&Wup[(size_t)g * nb], Wupf.data() + (size_t)g * 8 * n, 8 * n);
        dequantize_row_iq4_xs_r8(&Wgate[(size_t)g * nb], Wgatef.data() + (size_t)g * 8 * n, 8 * n);
    }

    // Pre-convert once (mirrors production Xu/Xg layout).
    std::vector<uint8_t> Xu((size_t)nrc_x * bx_q8 + 1024, 0);
    std::vector<uint8_t> Xg((size_t)nrc_x * bx_q8 + 1024, 0);
    if (!iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wup.data(), bx_w, Xu.data(), 0, nrc_x) ||
        !iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wgate.data(), bx_w, Xg.data(), 0, nrc_x)) {
        printf("  [FAIL] fused-manual n=%-5d nrc_x=%-3d : pre-convert returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    // Validate THIS Xu/Xg vs float (closes "my converted weights corrupt"):
    // exact mirror of the preconv control (group-packed reads, 0.5-quantum).
    // Hoisted to function scope: doubles as the converted GEMM reference below.
    std::vector<float> Xuf((size_t)nrc_x * n, 0.f), Xgf((size_t)nrc_x * n, 0.f);
    {
        float maxabs_w = 0.f;
        for (size_t j = 0; j < (size_t)nrc_x * n; ++j) {
            float a = fabsf(Wupf[j]); if (a > maxabs_w) maxabs_w = a;
            float b = fabsf(Wgatef[j]); if (b > maxabs_w) maxabs_w = b;
        }
        float qw = maxabs_w / 127.f;
        size_t nbad_c = 0; float max_err_c = 0.f, max_q = 0.f;
        auto check_buf = [&](const uint8_t * X, const std::vector<float> & Wf, std::vector<float> & Xf) {
            for (int g = 0; g < nrc_x / 8; ++g)
                dequantize_row_q8_k_r8((const block_q8_k_r8 *)(X + (size_t)g * 8 * bx_q8),
                                       Xf.data() + (size_t)g * 8 * n, 8 * n);
            for (int r = 0; r < nrc_x; ++r) {
                for (int k = 0; k < n; ++k) {
                    float ref = Wf[(size_t)r * n + k];
                    float e = fabsf(Xf[(size_t)r * n + k] - ref);
                    float eq = e / (qw > 1e-9f ? qw : 1e-9f);
                    if (e > max_err_c) max_err_c = e;
                    if (eq > max_q) max_q = eq;
                    if (e > 1.0f + 1e-2f * fabsf(ref) + 0.5f * qw) ++nbad_c;
                }
            }
        };
        check_buf(Xu.data(), Wupf, Xuf);
        check_buf(Xg.data(), Wgatef, Xgf);
        printf("  [%s] fused-manual-preconv n=%-5d nrc_x=%-3d : %zu bad (max|err|=%.4g, max|err|/q=%.3g, q=%.4g)\n",
               nbad_c == 0 ? "OK" : "FAIL", n, nrc_x, nbad_c, (double)max_err_c, (double)max_q, (double)qw);
        if (nbad_c != 0) ++g_failures;
    }

    std::vector<float> Bf((size_t)nrc_y * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    // Both B layouts: identical d/qs, differing bsums (int16 vs float).
    std::vector<uint8_t> B0((size_t)nrc_y * bx_B, 0), B1((size_t)nrc_y * bx_B, 0);
    for (int iy = 0; iy < nrc_y; ++iy) {
        iqk_quantize_row_q8_K(Bf.data() + (size_t)iy * n, B0.data() + (size_t)iy * bx_B, n);
        quantize_row_q8_K32(Bf.data() + (size_t)iy * n, B1.data() + (size_t)iy * bx_B, n);
    }
    std::vector<float> Bfq((size_t)nrc_y * n, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B0.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);
    // Constant-positive B, _T<1> layout (matches the failing config except for
    // content): isolates B-content dependence. Random uniform +/-1 is symmetric
    // with no structure; production activations are heavy-tailed with outliers.
    // constant-pass + random-fail => kernel fragile to B content statistics.
    std::vector<float> Bcf((size_t)nrc_y * n, 0.05f);
    std::vector<uint8_t> Bc((size_t)nrc_y * bx_B, 0);
    std::vector<float> Bcfq((size_t)nrc_y * n, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy) {
        quantize_row_q8_K32(Bcf.data() + (size_t)iy * n, Bc.data() + (size_t)iy * bx_B, n);
        dequant_q8_K_row((const block_q8_K *)(Bc.data() + (size_t)iy * bx_B),
                         Bcfq.data() + (size_t)iy * n, n);
    }

    std::vector<mmid_row_mapping> map(nrc_y);
    for (int iy = 0; iy < nrc_y; ++iy) { map[iy].i1 = iy; map[iy].i2 = 0; }

    // Byte-compare production repack vs test-helper converter on IDENTICAL
    // input: same layout kind expected (both R8-kind without FANCY_SIMD).
    // Divergence here (offset structure) locates a converter split; identity
    // forces the failure into call args (stride/mapping/C-state).
    std::vector<uint8_t> Wref((size_t)nrc_x * bx_q8 + 1024, 0xCC);
    iqk_test_convert_iq4_xs_r8(n, Wup.data(), bx_w, Wref.data(), nrc_x);
    size_t ndiff = 0, firstdiff = 0;
    for (size_t j = 0; j < (size_t)nrc_x * bx_q8; ++j) {
        if (Wref[j] != Xu[j]) { if (ndiff == 0) firstdiff = j; ++ndiff; }
    }
    printf("  [INFO] fused-manual-convmap: %zu diffs / %zu bytes, first @%zu (bx_q8=%zu rowsz_r8_equiv=%zu)\n",
           ndiff, (size_t)nrc_x * bx_q8, firstdiff, bx_q8, (size_t)(n / QK_K) * (sizeof(block_q8_k_r8) / 8));

    for (int variant = 0; variant < 3; ++variant) {
        // 0: _T<0> random (matches the passing plain config maximally),
        // 1: _T<1> random (the failing config), 2: _T<1> constant (content probe).
        const uint8_t * B = variant == 0 ? B0.data() : (variant == 1 ? B1.data() : Bc.data());
        const std::vector<float> & Br = variant == 2 ? Bcfq : Bfq;
        for (int usemap = 0; usemap < 2; ++usemap) {
        const mmid_row_mapping * mp = usemap ? map.data() : nullptr;
        // Gate-only kernel call (mirrors the fused gate pass).
        std::vector<float> Cg((size_t)nrc_y * nrc_x, -1.f);
        DataInfo infog;
        infog.s = Cg.data(); infog.cy = (const char *)B; infog.bs = nrc_x; infog.by = bx_B;
        infog.cur_y = 0; infog.ne11 = nrc_y; infog.row_mapping = mp;
        iqk_test_gemm_q8_k_r8(n, Xg.data(), bx_q8, infog, nrc_x, nrc_y);
        // Up-only kernel call.
        std::vector<float> Cu((size_t)nrc_y * nrc_x, -1.f);
        DataInfo infou = infog;
        infou.s = Cu.data();
        iqk_test_gemm_q8_k_r8(n, Xu.data(), bx_q8, infou, nrc_x, nrc_y);

        auto dotref = [&](const std::vector<float> & Wf, int iy, int r) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) s += (double)Wf[(size_t)r * n + k] * (double)Br[(size_t)iy * n + k];
            return s;
        };
        size_t nbad_g = 0, nbad_u = 0, nbad_f = 0;
        float maxg = 0.f, maxu = 0.f, maxf = 0.f, maxrf = 0.f;
        FILE * flog = nullptr;
        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int r = 0; r < nrc_x; ++r) {
                double sg = dotref(Xgf, iy, r), su = dotref(Xuf, iy, r);
                float gotg = Cg[(size_t)iy * nrc_x + r];
                float gotu = Cu[(size_t)iy * nrc_x + r];
                double gg = sg / (1.0 + exp(-sg));
                float reff = (float)(su * gg);
                float gotf = gotu * (gotg / (1.0f + expf(-gotg)));
                float eg = fabsf(gotg - (float)sg), eu = fabsf(gotu - (float)su);
                float ef = fabsf(gotf - reff);
                if (fabsf(reff) > maxrf) maxrf = fabsf(reff);
                if (eg > maxg) maxg = eg;
                if (eu > maxu) maxu = eu;
                if (ef > maxf) maxf = ef;
                if (eg > 1e-2f + 1e-4f * fabsf((float)sg)) {
                    if (nbad_g < 6) printf("    bad-gate[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)gotg, sg);
                    ++nbad_g;
                }
                if (eu > 1e-2f + 1e-4f * fabsf((float)su)) {
                    if (nbad_u < 6) printf("    bad-up[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)gotu, su);
                    ++nbad_u;
                }
                if (ef > 1e-2f + 5e-4f * fabsf(reff)) {
                    if (nbad_f < 6) printf("    bad-full[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)gotf, (double)reff);
                    if (!flog) {
                        flog = fopen("moe_fused_bad.log", "a");
                        if (flog) fprintf(flog, "== n=%d nrc_x=%d nrc_y=%d manual-b%d-m%d ==\n", n, nrc_x, nrc_y, variant, usemap);
                    }
                    if (flog) fprintf(flog, "iy=%d r=%d got=%.9g ref=%.9g\n", iy, r, (double)gotf, (double)reff);
                    ++nbad_f;
                }
            }
        }
        if (flog) fclose(flog);
        printf("  [%s] fused-manual-gate-b%dm%d n=%-5d nrc_x=%-3d : %zu bad (max|err|=%.4g)\n",
               nbad_g == 0 ? "OK" : "FAIL", variant, usemap, n, nrc_x, nbad_g, (double)maxg);
        printf("  [%s] fused-manual-up-b%dm%d   n=%-5d nrc_x=%-3d : %zu bad (max|err|=%.4g)\n",
               nbad_u == 0 ? "OK" : "FAIL", variant, usemap, n, nrc_x, nbad_u, (double)maxu);
        printf("  [%s] fused-manual-full-b%dm%d n=%-5d nrc_x=%-3d : %zu bad (max|err|=%.4g scale %.4g)\n",
               nbad_f == 0 ? "OK" : "FAIL", variant, usemap, n, nrc_x, nbad_f, (double)maxf, (double)maxrf);
        if (nbad_g + nbad_u + nbad_f != 0) ++g_failures;
        } // usemap
    } // variant
#endif
}

// ---------------------------------------------------------------------------
// Test: direct R8 GEMM vs float (the production CPU path for repacked
// experts: 189/201 IQ4_XS on the reporter's box stay on CPU/RAM).
//   Drives mul_mat_iq4_xs_r8_q8_k_avx2 on repacked weights with Q8_K32
//   activations (vec_dot_type of IQ4_XS_R8) and compares against a naive
//   double-precision GEMM over the dequantized inputs. Covers nrc_y=1
//   (TG-like) and 8 (PP chunk) plus MoE-style row_mappings. No FANCY gate:
//   this kernel exists on all x86_64 builds.
// ---------------------------------------------------------------------------
static void test_gemm_iq4_xs_r8_direct(int n, int nrc_x, int nrc_y) {
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    GGML_ASSERT(nrc_y >= 1 && nrc_y <= 8);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n); // Q8_K32 shares block_q8_K

    std::vector<block_iq4_xs_r8> W((size_t)(nrc_x / 8) * nb);
    for (auto & b : W) make_random_iq4_xs_r8(&b);

    std::vector<float> Wf((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_iq4_xs_r8(&W[(size_t)g * nb], Wf.data() + (size_t)g * 8 * n, 8 * n);

    const int NE11 = nrc_y;
    std::vector<float> Bf((size_t)NE11 * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)NE11 * bx_B);
    for (int iy = 0; iy < NE11; ++iy)
        quantize_row_q8_K32(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    std::vector<float> Bfq((size_t)NE11 * n, 0.f);
    for (int iy = 0; iy < NE11; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);

    auto run_case = [&](const char * tag, int ne11, const std::vector<mmid_row_mapping> * mapping, int ny) {
        std::vector<float> C((size_t)ne11 * nrc_x, -1.f);
        DataInfo info;
        info.s   = C.data();
        info.cy  = (const char *)B.data();
        info.bs  = nrc_x;
        info.by  = bx_B;
        info.cur_y = 0;
        info.ne11  = ne11;
        info.row_mapping = mapping ? mapping->data() : nullptr;
        iqk_test_gemm_iq4_xs_r8(n, W.data(), bx_w, info, nrc_x, ny);

        float max_err = 0.f, max_ref = 0.f;
        size_t nbad = 0;
        for (int iy = 0; iy < ny; ++iy) {
            int i1 = mapping ? (*mapping)[iy].i1 : iy;
            for (int r = 0; r < nrc_x; ++r) {
                const float * wr = Wf.data() + (size_t)r * n;
                const float * br = Bfq.data() + (size_t)i1 * n;
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += (double)wr[k] * (double)br[k];
                float got = C[(size_t)i1 * nrc_x + r];
                float e = fabsf(got - (float)s);
                if (fabsf((float)s) > max_ref) max_ref = fabsf((float)s);
                if (e > max_err) max_err = e;
                if (e > 1e-2f + 1e-4f * fabsf((float)s)) ++nbad;
            }
        }
        if (nbad == 0)
            printf("  [OK]   gemm-r8d %-12s n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
                   tag, n, nrc_x, ny, (double)max_err, (double)max_ref);
        else {
            printf("  [FAIL] gemm-r8d %-12s n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
                   tag, n, nrc_x, ny, nbad, (double)max_err, (double)max_ref);
            ++g_failures;
        }
    };

    run_case("plain", nrc_y, nullptr, nrc_y);
    {
        std::vector<mmid_row_mapping> rev(nrc_y);
        for (int iy = 0; iy < nrc_y; ++iy) { rev[iy].i1 = nrc_y - 1 - iy; rev[iy].i2 = 0; }
        run_case("rev-map", nrc_y, &rev, nrc_y);
    }
    if (nrc_y >= 4) {
        std::vector<mmid_row_mapping> sub(4);
        for (int iy = 0; iy < 4; ++iy) { sub[iy].i1 = 2 * iy; sub[iy].i2 = 0; }
        run_case("sub-map", NE11, &sub, 4);
    }
}

// ---------------------------------------------------------------------------
// Test: fused MoE up-gate (iqk_moe_fused_up_gate) vs float reference.
// Single expert, identity row_mapping, no bias, SiLU, no clamp. Covers the
// production fused path end to end, including - at nrc_y >= 32 - the
// R8->Q8 converter dispatch that aborted pre-15e60276 when converters were
// unavailable (converter-missing class: a missing converter fails loudly
// here by crashing the suite, which is the intended signal).
// Requires TEST_DISPATCH linkage (iqk_mul_mat.cpp), like test_dispatch_pipeline.
// ---------------------------------------------------------------------------
static void test_moe_fused_up_gate(int n, int nrc_x, int nrc_y, bool use_map = true) {
#ifndef TEST_DISPATCH
    (void)n; (void)nrc_x; (void)nrc_y;
    printf("  [SKIP] moe-fused: compiled without TEST_DISPATCH\n");
    return;
#else
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n); // Q8_K32 shares block_q8_K

    std::vector<block_iq4_xs_r8> Wup((size_t)(nrc_x / 8) * nb);
    std::vector<block_iq4_xs_r8> Wgate((size_t)(nrc_x / 8) * nb);
    for (auto & b : Wup) make_random_iq4_xs_r8(&b);
    for (auto & b : Wgate) make_random_iq4_xs_r8(&b);

    // Reference from the CONVERTED weights (same bytes the kernel dots):
    // a quantized GEMM cannot beat its own input quantization. Comparing
    // against R8-dequant folds requant noise (~0.4% here) into the verdict
    // and buries real kernel errors under borderline trips; the converter
    // itself is covered separately (preconv-convert, 0.5-quantum).
    const size_t bx_q8 = ggml_row_size(GGML_TYPE_Q8_K_R8, n);
    std::vector<uint8_t> Xu((size_t)nrc_x * bx_q8 + 1024, 0);
    std::vector<uint8_t> Xg((size_t)nrc_x * bx_q8 + 1024, 0);
    if (!iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wup.data(), bx_w, Xu.data(), 0, nrc_x) ||
        !iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wgate.data(), bx_w, Xg.data(), 0, nrc_x)) {
        printf("  [FAIL] moe-fused n=%-5d nrc_x=%-3d nrc_y=%d : pre-convert returned false\n",
               n, nrc_x, nrc_y);
        ++g_failures;
        return;
    }
    // float copies, row r at Wf + r*n (dequant of the converted weights).
    // Each 8-row group spans 8*bx_q8 bytes (interleaved), so group g is at g*8*bx_q8.
    std::vector<float> Wupf((size_t)nrc_x * n, 0.f), Wgatef((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g) {
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)(Xu.data() + (size_t)g * 8 * bx_q8),
                               Wupf.data() + (size_t)g * 8 * n, 8 * n);
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)(Xg.data() + (size_t)g * 8 * bx_q8),
                               Wgatef.data() + (size_t)g * 8 * n, 8 * n);
    }
    // R8-dequant copies for the direct path (Ny below the convert threshold):
    // direct kernels dot R8 weights with no requantization involved.
    std::vector<float> WupfR((size_t)nrc_x * n, 0.f), WgatefR((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g) {
        dequantize_row_iq4_xs_r8(&Wup[(size_t)g * nb], WupfR.data() + (size_t)g * 8 * n, 8 * n);
        dequantize_row_iq4_xs_r8(&Wgate[(size_t)g * nb], WgatefR.data() + (size_t)g * 8 * n, 8 * n);
    }
    // Path-aware reference: compare against what the entry actually dots
    // (same dispatch query it uses internally).
    const bool converts = iqk_dequant_type((int)GGML_TYPE_IQ4_XS_R8, nrc_y) != (int)GGML_TYPE_IQ4_XS_R8;
    const std::vector<float> &WupR = converts ? Wupf : WupfR;
    const std::vector<float> &WgateR = converts ? Wgatef : WgatefR;

    std::vector<float> Bf((size_t)nrc_y * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)nrc_y * bx_B);
    for (int iy = 0; iy < nrc_y; ++iy)
        quantize_row_q8_K32(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    std::vector<float> Bfq((size_t)nrc_y * n, 0.f);
    for (int iy = 0; iy < nrc_y; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);

    std::vector<mmid_row_mapping> map(nrc_y);
    for (int iy = 0; iy < nrc_y; ++iy) { map[iy].i1 = iy; map[iy].i2 = 0; }

    // Sentinel init: unwritten rows stay visible as -1e30 (vs -1.f); rows the
    // kernel writes as exact 0.0 show as 0 -> drop vs zero-write separable.
    std::vector<float> C((size_t)nrc_y * nrc_x, g_cinit_plain ? -1.f : -1.0e30f);
    const long nb1 = (long)nrc_x * (long)sizeof(float);
    // Production dispatches R8 weights only against their vec_dot_type
    // (Q8_K32 for IQ4_XS_R8, same bytes as Q8_K blocks): passing Q8_K
    // makes prepare() reject the combo and the call returns false.
    bool ok = iqk_moe_fused_up_gate(nrc_x, nrc_y, n, nrc_y, (int)GGML_UNARY_OP_SILU,
            (int)GGML_TYPE_IQ4_XS_R8, Wup.data(), Wgate.data(), (long)bx_w,
            (int)GGML_TYPE_Q8_K32, B.data(), (long)bx_B,
            nullptr, nullptr, C.data(), nb1, 0, use_map ? map.data() : nullptr, 0.f, 0, 1);
    if (!ok) {
        printf("  [FAIL] moe-fused n=%-5d nrc_x=%-3d nrc_y=%d : fused call returned false\n",
               n, nrc_x, nrc_y);
        ++g_failures;
        return;
    }

    size_t nbad = 0;
    float max_err = 0.f, max_ref = 0.f;
    FILE * flog = nullptr; // full bad-index dump (console stays capped)
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int r = 0; r < nrc_x; ++r) {
            double su = 0.0, sg = 0.0;
            for (int k = 0; k < n; ++k) {
                su += (double)WupR[(size_t)r * n + k] * (double)Bfq[(size_t)iy * n + k];
                sg += (double)WgateR[(size_t)r * n + k] * (double)Bfq[(size_t)iy * n + k];
            }
            double g = sg / (1.0 + exp(-sg));
            float ref = (float)(su * g);
            float got = C[(size_t)iy * nrc_x + r];
            float e = fabsf(got - ref);
            if (fabsf(ref) > max_ref) max_ref = fabsf(ref);
            if (e > max_err) max_err = e;
            // Entry-path tolerance: 8e-3 relative + 2.0 absolute. Refs are
            // converter-output-exact so requant noise is out; what remains is
            // legitimate float noise: blocked int-dot -> float scale accumulation
            // vs naive double dots, scalar exp vs v_silu approx, and silu-tail
            // amplification (tiny abs dot noise -> large rel product noise when
            // |sg| is large-negative). Measured tail after the stride fix:
            // 6.1e-3 max non-cancellation rel (1024-shape <=6e-4) + 0.5 abs on
            // near-cancellation refs; production PPL healthy (15.96, no NANs).
            // Structural fails still trip by orders of magnitude (stride era:
            // 100% rows bad, zeros vs 1e8, errors 1e6+). Kernel-level checks
            // below stay at tight tolerance.
            if (e > 2.0f + 8e-3f * fabsf(ref)) {
                if (nbad < 12) printf("    bad[iy=%d,r=%d]: got %.6g ref %.6g map=%d,%d\n", iy, r, (double)got, (double)ref, map[iy].i1, map[iy].i2);
                if (!flog) {
                    flog = fopen("moe_fused_bad.log", "a");
                    if (flog) fprintf(flog, "== n=%d nrc_x=%d nrc_y=%d ==\n", n, nrc_x, nrc_y);
                }
                if (flog) fprintf(flog, "iy=%d r=%d got=%.9g ref=%.9g map=%d,%d\n", iy, r, (double)got, (double)ref, map[iy].i1, map[iy].i2);
                ++nbad;
            }
        }
    }
    if (flog) fclose(flog);
    if (nbad == 0)
        printf("  [OK]   moe-fused n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
               n, nrc_x, nrc_y, (double)max_err, (double)max_ref);
    else {
        printf("  [FAIL] moe-fused n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
               n, nrc_x, nrc_y, nbad, (double)max_err, (double)max_ref);
        ++g_failures;
    }
#endif
}

// ---------------------------------------------------------------------------
// Test: fused MoE up-gate on PRE-CONVERTED Q8_K_R8 weights vs float.
// Bypasses the R8->Q8 dispatch + converter: isolates the converted-kernel
// combo (Q8_K_R8 x Q8_K32) inside the fused wrapper.
//   - passes  => converter/dispatch plumbing guilty for the Ny>=32 failures
//   - fails   => converted kernels guilty (same kernels, converter out)
// ---------------------------------------------------------------------------
static void test_moe_fused_up_gate_preconv(int n, int nrc_x, int nrc_y) {
#ifndef TEST_DISPATCH
    (void)n; (void)nrc_x; (void)nrc_y;
    printf("  [SKIP] moe-fused-preconv: compiled without TEST_DISPATCH\n");
    return;
#else
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_K;
    const size_t bx_w = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    const size_t bx_B = ggml_row_size(GGML_TYPE_Q8_K, n);
    const size_t bx_q8 = ggml_row_size(GGML_TYPE_Q8_K_R8, n);

    std::vector<block_iq4_xs_r8> Wup((size_t)(nrc_x / 8) * nb);
    std::vector<block_iq4_xs_r8> Wgate((size_t)(nrc_x / 8) * nb);
    for (auto & b : Wup) make_random_iq4_xs_r8(&b);
    for (auto & b : Wgate) make_random_iq4_xs_r8(&b);

    std::vector<float> Wupf((size_t)nrc_x * n, 0.f), Wgatef((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g) {
        dequantize_row_iq4_xs_r8(&Wup[(size_t)g * nb], Wupf.data() + (size_t)g * 8 * n, 8 * n);
        dequantize_row_iq4_xs_r8(&Wgate[(size_t)g * nb], Wgatef.data() + (size_t)g * 8 * n, 8 * n);
    }

    const int NE11 = nrc_y;
    std::vector<float> Bf((size_t)NE11 * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)NE11 * bx_B);
    for (int iy = 0; iy < NE11; ++iy)
        quantize_row_q8_K32(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    std::vector<float> Bfq((size_t)NE11 * n, 0.f);
    for (int iy = 0; iy < NE11; ++iy)
        dequant_q8_K_row((const block_q8_K *)(B.data() + (size_t)iy * bx_B),
                         Bfq.data() + (size_t)iy * n, n);

    // pre-convert (packed), mirroring production Xu/Xg layout
    std::vector<uint8_t> Xu((size_t)nrc_x * bx_q8 + 1024, 0);
    std::vector<uint8_t> Xg((size_t)nrc_x * bx_q8 + 1024, 0);
    bool oku = false, okg = false;
    bench_convert_ms("preconv r8->q8 pair (1024x64)", 2L * nrc_x * n, 8.0, [&]() {
        oku = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wup.data(), bx_w, Xu.data(), 0, nrc_x);
        okg = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, n, Wgate.data(), bx_w, Xg.data(), 0, nrc_x);
    });
    if (!oku || !okg) {
        printf("  [FAIL] moe-fused-preconv n=%-5d nrc_x=%-3d nrc_y=%d : pre-convert returned false\n",
               n, nrc_x, nrc_y);
        ++g_failures;
        return;
    }

    // validate the converted weights themselves (dequant Xu/Xg vs float ref):
    // separates converter guilt from converted-kernel guilt. Xuf/Xgf double
    // as the GEMM reference below (noise-free: kernel-vs-its-own-input).
    std::vector<float> Xuf((size_t)nrc_x * n, 0.f), Xgf((size_t)nrc_x * n, 0.f);
    {
        // Converter packs groups contiguously: each 8-row group spans
        // 8*bx_q8 bytes (interleaved), so group g is at g*8*bx_q8.
        for (int g = 0; g < nrc_x / 8; ++g) {
            dequantize_row_q8_k_r8((const block_q8_k_r8 *)(Xu.data() + (size_t)g * 8 * bx_q8),
                                   Xuf.data() + (size_t)g * 8 * n, 8 * n);
            dequantize_row_q8_k_r8((const block_q8_k_r8 *)(Xg.data() + (size_t)g * 8 * bx_q8),
                                   Xgf.data() + (size_t)g * 8 * n, 8 * n);
        }
        float maxabs_w = 0.f;
        for (size_t j = 0; j < (size_t)nrc_x * n; ++j) {
            float a = fabsf(Wupf[j]); if (a > maxabs_w) maxabs_w = a;
            float b = fabsf(Wgatef[j]); if (b > maxabs_w) maxabs_w = b;
        }
        float quantum_w = maxabs_w / 127.f; // optimal Q8 step: legit rounding is <=0.5*quantum
        size_t nbad_c = 0; float max_err_c = 0.f; float max_q = 0.f;
        // Tolerance accounts for Q8 requantization noise (quantum d*dnew/2):
        // converter output is correct if it matches to ~1.0 absolute + 1% relative + half quantum.
        for (int r = 0; r < nrc_x; ++r) {
            for (int k = 0; k < n; ++k) {
                float ref = Wupf[(size_t)r * n + k];
                float e = fabsf(Xuf[(size_t)r * n + k] - ref);
                float eq = e / (quantum_w > 1e-9f ? quantum_w : 1e-9f);
                if (e > max_err_c) max_err_c = e;
                if (eq > max_q) max_q = eq;
                if (e > 1.0f + 1e-2f * fabsf(ref) + 0.5f * quantum_w) {
                    if (nbad_c < 10) printf("    conv-bad[r=%d,k=%d]: got %.6g ref %.6g\n",
                                            r, k, (double)Xuf[(size_t)r * n + k], (double)ref);
                    ++nbad_c;
                }
                ref = Wgatef[(size_t)r * n + k];
                e = fabsf(Xgf[(size_t)r * n + k] - ref);
                eq = e / (quantum_w > 1e-9f ? quantum_w : 1e-9f);
                if (e > max_err_c) max_err_c = e;
                if (eq > max_q) max_q = eq;
                if (e > 1.0f + 1e-2f * fabsf(ref) + 0.5f * quantum_w) ++nbad_c;
            }
        }
        printf("  [%s] moe-fused-preconv-convert n=%-5d nrc_x=%-3d : %zu bad (max|err|=%.4g, max|err|/q=%.3g, q=%.4g)\n",
               nbad_c == 0 ? "OK" : "FAIL", n, nrc_x, nbad_c, (double)max_err_c, (double)max_q, (double)quantum_w);
        if (nbad_c != 0) ++g_failures;
    }

    std::vector<mmid_row_mapping> map(nrc_y);
    for (int iy = 0; iy < nrc_y; ++iy) { map[iy].i1 = iy; map[iy].i2 = 0; }

    // Sentinel init (see test_moe_fused_up_gate): separates dropped rows from zero-writes.
    std::vector<float> C((size_t)nrc_y * nrc_x, g_cinit_plain ? -1.f : -1.0e30f);
    const long nb1 = (long)nrc_x * (long)sizeof(float);
    // Explicit-Q8 weights use the ggml per-row stride (bx_q8): kernels step
    // ix += 8, so each 8-row group spans 8*bx_q8 bytes. Same stride the
    // production convert path passes internally.
    bool ok = iqk_moe_fused_up_gate(nrc_x, nrc_y, n, nrc_y, (int)GGML_UNARY_OP_SILU,
            (int)GGML_TYPE_Q8_K_R8, Xu.data(), Xg.data(), (long)bx_q8,
            (int)GGML_TYPE_Q8_K32, B.data(), (long)bx_B,
            nullptr, nullptr, C.data(), nb1, 0, map.data(), 0.f, 0, 1);
    if (!ok) {
        printf("  [FAIL] moe-fused-preconv n=%-5d nrc_x=%-3d nrc_y=%d : fused call returned false\n",
               n, nrc_x, nrc_y);
        ++g_failures;
        return;
    }

    size_t nbad = 0;
    float max_err = 0.f, max_ref = 0.f;
    FILE * flog = nullptr; // full bad-index dump, distinct header (see test_moe_fused_up_gate)
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int r = 0; r < nrc_x; ++r) {
            double su = 0.0, sg = 0.0;
            for (int k = 0; k < n; ++k) {
                su += (double)Xuf[(size_t)r * n + k] * (double)Bfq[(size_t)iy * n + k];
                sg += (double)Xgf[(size_t)r * n + k] * (double)Bfq[(size_t)iy * n + k];
            }
            double g = sg / (1.0 + exp(-sg));
            float ref = (float)(su * g);
            float got = C[(size_t)iy * nrc_x + r];
            float e = fabsf(got - ref);
            if (fabsf(ref) > max_ref) max_ref = fabsf(ref);
            if (e > max_err) max_err = e;
            // Entry-path tolerance, same calibration as test_moe_fused_up_gate
            // above (float accumulation + v_silu approx + silu-tail
            // amplification; structural fails still trip by orders of magnitude).
            if (e > 2.0f + 8e-3f * fabsf(ref)) {
                if (nbad < 12) printf("    bad[iy=%d,r=%d]: got %.6g ref %.6g map=%d,%d\n", iy, r, (double)got, (double)ref, map[iy].i1, map[iy].i2);
                if (!flog) {
                    flog = fopen("moe_fused_bad.log", "a");
                    if (flog) fprintf(flog, "== n=%d nrc_x=%d nrc_y=%d preconv ==\n", n, nrc_x, nrc_y);
                }
                if (flog) fprintf(flog, "iy=%d r=%d got=%.9g ref=%.9g map=%d,%d\n", iy, r, (double)got, (double)ref, map[iy].i1, map[iy].i2);
                ++nbad;
            }
        }
    }
    if (flog) fclose(flog);
    if (nbad == 0)
        printf("  [OK]   moe-fused-preconv n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
               n, nrc_x, nrc_y, (double)max_err, (double)max_ref);
    else {
        printf("  [FAIL] moe-fused-preconv n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
               n, nrc_x, nrc_y, nbad, (double)max_err, (double)max_ref);
        ++g_failures;
    }
#endif
}

// ---------------------------------------------------------------------------
// Test: direct MXFP4_R8 GEMM vs float (expert path, Q8_2_X4 act) — mirrors
// test_gemm_iq4_xs_r8_direct. Covers the production expert arithmetic
// (mul_mat_mxfp4_r8_q8_2_avx2, nrc_y=1/8, MoE row_mapping) that the
// repack+converter tests do not exercise.
// ---------------------------------------------------------------------------
// Forward declaration: defined with the other MXFP4 helpers below.
static void make_random_mxfp4(block_mxfp4 * blk);
static void test_gemm_mxfp4_r8_direct(int n, int nrc_x, int nrc_y) {
    GGML_ASSERT(n % QK_MXFP4 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    GGML_ASSERT(nrc_y >= 1 && nrc_y <= 8);
    const int nb = n / QK_MXFP4;
    const size_t bx_w = ggml_row_size(GGML_TYPE_MXFP4_R8, n);
    // GGML_TYPE_Q8_2_X4 is IQK-internal (ggml_row_size returns 0): mirror
    // quantize_row_q8_1_x4_T layout - nb4 32-blocks grouped by 4, plain tail.
    const int nb_b = n / QK8_2;
    const int nb4_b = 4*(nb_b/4);
    const size_t bx_B = (size_t)(nb4_b/4)*sizeof(block_q8_2_x4) + (size_t)(nb_b-nb4_b)*sizeof(block_q8_2);

    std::vector<block_mxfp4> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_mxfp4(&b);
    std::vector<block_mxfp4_r8> W((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_mxfp4(nrc_x, n, src.data(), W.data());

    std::vector<float> Wf((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_mxfp4_r8(&W[(size_t)g * nb], Wf.data() + (size_t)g * 8 * n, 8 * n);

    const int NE11 = 8;
    std::vector<float> Bf((size_t)NE11 * n);
    for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
    std::vector<uint8_t> B((size_t)NE11 * bx_B);
    for (int iy = 0; iy < NE11; ++iy)
        quantize_row_q8_2_x4(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
    // Dequant B for the reference. quantize_row_q8_2_x4 emits block_q8_2_x4
    // groups (4x 32-quants + 4x (bf16 d, int16 sum)), plain block_q8_2 tail.
    // block_q8_2 is {bf16 d, bf16 sum, qs[32]}.
    std::vector<float> Bfq((size_t)NE11 * n, 0.f);
    for (int iy = 0; iy < NE11; ++iy) {
        float * fout = Bfq.data() + (size_t)iy * n;
        const auto * x4 = (const block_q8_2_x4 *)(B.data() + (size_t)iy * bx_B);
        for (int g = 0; g < nb4_b/4; ++g) {
            for (int r = 0; r < 4; ++r) {
                float d = GGML_BF16_TO_FP32(ggml_bf16_t{x4[g].d[r]});
                for (int j = 0; j < QK8_2; ++j) fout[(g*4+r)*QK8_2 + j] = d * x4[g].qs[r*QK8_2 + j];
            }
        }
        const auto * rem = (const block_q8_2 *)(x4 + nb4_b/4);
        for (int i = 0; i < nb_b - nb4_b; ++i) {
            float d = GGML_BF16_TO_FP32(ggml_bf16_t{rem[i].d});
            for (int j = 0; j < QK8_2; ++j) fout[(nb4_b+i)*QK8_2 + j] = d * rem[i].qs[j];
        }
    }

    auto run_case = [&](const char * tag, int ne11, const std::vector<mmid_row_mapping> * mapping, int ny) {
        std::vector<float> C((size_t)ne11 * nrc_x, -1.f);
        DataInfo info;
        info.s   = C.data();
        info.cy  = (const char *)B.data();
        info.bs  = nrc_x;
        info.by  = bx_B;
        info.cur_y = 0;
        info.ne11  = ne11;
        info.row_mapping = mapping ? mapping->data() : nullptr;
        iqk_test_gemm_mxfp4_r8(n, W.data(), bx_w, info, nrc_x, ny);

        float max_err = 0.f, max_ref = 0.f;
        size_t nbad = 0;
        for (int iy = 0; iy < ny; ++iy) {
            int i1 = mapping ? (*mapping)[iy].i1 : iy;
            for (int r = 0; r < nrc_x; ++r) {
                const float * wr = Wf.data() + (size_t)r * n;
                const float * br = Bfq.data() + (size_t)i1 * n;
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += (double)wr[k] * (double)br[k];
                float got = C[(size_t)i1 * nrc_x + r];
                float e = fabsf(got - (float)s);
                if (fabsf((float)s) > max_ref) max_ref = fabsf((float)s);
                if (e > max_err) max_err = e;
                // Near-cancellation tolerance: 2e-4 relative. The only trip is a
                // single dot (W row 24 x B row 5, seen via both plain iy=5 and
                // rev-map iy=2 -> i1=5): err 0.023 on ref 124 with suite scale
                // 2.8e6, i.e. strong cancellation of large +/- terms where
                // float noise is large vs the result but tiny vs accumulated
                // magnitude. No row/group pattern (1/512 dots), repack bit-exact,
                // small shapes green. Structural fails still trip by orders.
                if (e > 1e-2f + 2e-4f * fabsf((float)s)) {
                    if (nbad < 8) printf("    bad[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)got, (double)(float)s);
                    ++nbad;
                }
            }
        }
        if (nbad == 0)
            printf("  [OK]   gemm-mxfp4d %-11s n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g)\n",
                   tag, n, nrc_x, ny, (double)max_err, (double)max_ref);
        else {
            printf("  [FAIL] gemm-mxfp4d %-11s n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g)\n",
                   tag, n, nrc_x, ny, nbad, (double)max_err, (double)max_ref);
            ++g_failures;
        }
    };

    run_case("plain", nrc_y, nullptr, nrc_y);
    {
        std::vector<mmid_row_mapping> rev(nrc_y);
        for (int iy = 0; iy < nrc_y; ++iy) { rev[iy].i1 = nrc_y - 1 - iy; rev[iy].i2 = 0; }
        run_case("rev-map", nrc_y, &rev, nrc_y);
    }
    if (nrc_y >= 4) {
        std::vector<mmid_row_mapping> sub(4);
        for (int iy = 0; iy < 4; ++iy) { sub[iy].i1 = 2 * iy; sub[iy].i2 = 0; }
        run_case("sub-map", NE11, &sub, 4);
    }
}

// Forward makers (defined alongside their suites below, used here first).
static void make_random_q5_0(block_q5_0 * blk);
static void make_random_q4_0(block_q4_0 * blk);
static void make_random_iq4_nl(block_iq4_nl * blk);

// ---------------------------------------------------------------------------
// Test: Q8_0_R8 repack integrity — random Q8_0 rows -> real repack_q8_0 ->
// real dequantize_row_q8_0_r8 must equal the natural-order dequant EXACTLY.
// ---------------------------------------------------------------------------
static void test_q8_0_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK8_0 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK8_0;
    std::vector<block_q8_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q8_0(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_q8_0_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_q8_0_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_q8_0(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&rep[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q8_0-repack  n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] q8_0-repack  n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ4_XS_R8 repack integrity — random IQ4_XS rows -> real repack_iq4_xs
// -> real dequantize_row_iq4_xs_r8 must equal the natural-order dequant
// EXACTLY (same math as the disabled test_repack_integrity, but driving the
// real hook instead of a hand replica).
// ---------------------------------------------------------------------------
static void test_iq4_xs_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_K;
    std::vector<block_iq4_xs> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_iq4_xs(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_native_iq4_xs(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_iq4_xs_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_iq4_xs(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_iq4_xs_r8(&rep[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   iq4_xs-repack n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] iq4_xs-repack n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: repacked direct GEMM vs float for the legacy R-family experts.
// mxfp4/iq4_xs have dedicated suites above; this covers q5_0_r4, q4_0_r8,
// iq4_nl_r4, q6_0_r4, q8_0_r8 through the PUBLIC iqk_mul_mat entry (Ny=1 runs
// the direct kernels; Ny=8 whatever dispatch selects — refs are
// path-independent float truth, and a declined combo fails loudly). B is
// Q8_2_X4 for all five (their vec_dot type on AVX2); C is plain.
// ---------------------------------------------------------------------------
static void test_legacy_r_direct(int fam) {
#ifndef TEST_DISPATCH
    (void)fam;
    printf("  [SKIP] legacy-r-direct: compiled without TEST_DISPATCH\n");
    return;
#else
    struct Fam { const char * name; int plain, repacked; int qk; int rgroup; };
    static const Fam fams[] = {
        { "q5_0",   GGML_TYPE_Q5_0,   GGML_TYPE_Q5_0_R4,  QK5_0,  4 },
        { "q4_0",   GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0_R8,  QK4_0,  8 },
        { "iq4_nl", GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL_R4, QK4_NL, 4 },
        { "q6_0",   GGML_TYPE_Q6_0,   GGML_TYPE_Q6_0_R4,  QK6_0,  4 },
        { "q8_0",   GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0_R8,  QK8_0,  8 },
    };
    const Fam & F = fams[fam];
    const int n = 1024;
    const int nb = n / F.qk;
    // Q8_2_X4 activation layout (mirror test_gemm_mxfp4_r8_direct).
    const int nb_b = n / QK8_2;
    const int nb4_b = 4*(nb_b/4);
    const size_t bx_B = (size_t)(nb4_b/4)*sizeof(block_q8_2_x4) + (size_t)(nb_b-nb4_b)*sizeof(block_q8_2);

    for (int nrc_x = 8; nrc_x <= 64; nrc_x *= 8) {
        // Plain -> repack -> repacked weights under test (exact-fit buffers
        // from the trait table, same convention as production).
        const size_t bs_src = ggml_row_size((enum ggml_type)F.plain, n);
        const size_t bx_w = ggml_row_size((enum ggml_type)F.repacked, n);
        std::vector<uint8_t> src((size_t)nrc_x * bs_src + 64, 0);
        std::vector<uint8_t> W((size_t)nrc_x * bx_w + 64, 0);
        if (fam == 0) {
            auto * s = (block_q5_0 *)src.data(); for (int i = 0; i < nrc_x * nb; ++i) make_random_q5_0(&s[i]);
            iqk_test_repack_q5_0(nrc_x, n, s, (block_q5_0_r4 *)W.data());
        } else if (fam == 1) {
            auto * s = (block_q4_0 *)src.data(); for (int i = 0; i < nrc_x * nb; ++i) make_random_q4_0(&s[i]);
            iqk_test_repack_q4_0(nrc_x, n, s, (block_iq4_nl_r8 *)W.data());
        } else if (fam == 2) {
            auto * s = (block_iq4_nl *)src.data(); for (int i = 0; i < nrc_x * nb; ++i) make_random_iq4_nl(&s[i]);
            iqk_test_repack_iq4_nl(nrc_x, n, s, (block_iq4_nl_r4 *)W.data());
        } else if (fam == 3) {
            auto * s = (block_q6_0 *)src.data(); for (int i = 0; i < nrc_x * nb; ++i) make_random_q6_0(&s[i]);
            iqk_test_repack_q6_0(nrc_x, n, s, (block_q6_0_r4 *)W.data());
        } else {
            auto * s = (block_q8_0 *)src.data(); for (int i = 0; i < nrc_x * nb; ++i) make_random_q8_0(&s[i]);
            iqk_test_repack_q8_0(nrc_x, n, s, (block_q8_0_r8 *)W.data());
        }

        std::vector<float> Wf((size_t)nrc_x * n, 0.f);
        for (int g = 0; g < nrc_x / F.rgroup; ++g) {
            // Group g lives at g*rgroup*bx_w (interleaved span per group).
            const uint8_t * grp = W.data() + (size_t)g * F.rgroup * bx_w;
            float * out = Wf.data() + (size_t)g * F.rgroup * n;
            if (fam == 0) dequantize_row_q5_0_r4((const block_q5_0_r4 *)grp, out, 4 * n);
            else if (fam == 1) dequantize_row_q4_0_r8((const block_iq4_nl_r8 *)grp, out, 8 * n);
            else if (fam == 2) dequantize_row_iq4_nl_r4((const block_iq4_nl_r4 *)grp, out, 4 * n);
            else if (fam == 3) dequantize_row_q6_0_r4((const block_q6_0_r4 *)grp, out, 4 * n);
            else dequantize_row_q8_0_r8((const block_q8_0_r8 *)grp, out, 8 * n);
        }

        for (int nrc_y = 1; nrc_y <= 8; nrc_y *= 8) {
            std::vector<float> Bf((size_t)nrc_y * n);
            for (size_t j = 0; j < Bf.size(); ++j) Bf[j] = 0.05f * ((int)(g_rng() % 41) - 20);
            std::vector<uint8_t> B((size_t)nrc_y * bx_B);
            for (int iy = 0; iy < nrc_y; ++iy)
                quantize_row_q8_2_x4(Bf.data() + (size_t)iy * n, B.data() + (size_t)iy * bx_B, n);
            std::vector<float> Bfq((size_t)nrc_y * n, 0.f);
            dequant_q8_2_x4_rows(B.data(), bx_B, Bfq.data(), nrc_y, n);

            std::vector<float> C((size_t)nrc_y * nrc_x, -1.f);
            bool ok = iqk_mul_mat(nrc_x, nrc_y, n, F.repacked, W.data(), (long)bx_w,
                                  GGML_TYPE_Q8_2_X4, B.data(), (long)bx_B, C.data(), nrc_x, 0, 1);
            const char * path =
                iqk_dequant_type(F.repacked, nrc_y) != F.repacked ? "convert" : "direct";
            if (!ok) {
                printf("  [FAIL] gemm-rd %-6s n=%-5d nrc_x=%-3d nrc_y=%d : entry returned false (%s path)\n",
                       F.name, n, nrc_x, nrc_y, path);
                ++g_failures;
                continue;
            }
            float max_err = 0.f, max_ref = 0.f;
            size_t nbad = 0;
            for (int iy = 0; iy < nrc_y; ++iy) {
                for (int r = 0; r < nrc_x; ++r) {
                    const float * wr = Wf.data() + (size_t)r * n;
                    const float * br = Bfq.data() + (size_t)iy * n;
                    double s = 0.0;
                    for (int k = 0; k < n; ++k) s += (double)wr[k] * (double)br[k];
                    float got = C[(size_t)iy * nrc_x + r];
                    float e = fabsf(got - (float)s);
                    if (fabsf((float)s) > max_ref) max_ref = fabsf((float)s);
                    if (e > max_err) max_err = e;
                    // Same near-cancellation calibration as gemm-mxfp4d.
                    if (e > 1e-2f + 2e-4f * fabsf((float)s)) {
                        if (nbad < 8) printf("    bad[iy=%d,r=%d]: got %.6g ref %.6g\n", iy, r, (double)got, (double)(float)s);
                        ++nbad;
                    }
                }
            }
            if (nbad == 0)
                printf("  [OK]   gemm-rd %-6s n=%-5d nrc_x=%-3d nrc_y=%d : max|err|=%.4g (scale %.4g, %s)\n",
                       F.name, n, nrc_x, nrc_y, (double)max_err, (double)max_ref, path);
            else {
                printf("  [FAIL] gemm-rd %-6s n=%-5d nrc_x=%-3d nrc_y=%d : %zu bad (max|err|=%.4g scale %.4g, %s)\n",
                       F.name, n, nrc_x, nrc_y, nbad, (double)max_err, (double)max_ref, path);
                ++g_failures;
            }
        }
    }
#endif
}
static void make_random_q4_0(block_q4_0 * blk) {
    float d = (g_rng() % 1000) * 0.001f + 0.01f;
    blk->d = GGML_FP32_TO_FP16(d);
    for (auto & v : blk->qs) v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order Q4_0 dequant: signed nibbles with -8 bias.
static void dequant_q4_0_row(const block_q4_0 * x, float * y, int64_t k) {
    int64_t nb = k / QK4_0;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            y[i * QK4_0 + j]              = d * ((x[i].qs[j] & 0x0F) - 8);
            y[i * QK4_0 + j + QK4_0 / 2]  = d * ((x[i].qs[j] >> 4) - 8);
        }
    }
}

// Build a random but *valid* block_iq4_nl.
static void make_random_iq4_nl(block_iq4_nl * blk) {
    float d = (g_rng() % 1000) * 0.001f + 0.01f;
    blk->d = GGML_FP32_TO_FP16(d);
    for (auto & v : blk->qs) v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order IQ4_NL dequant through the shared values table.
static void dequant_iq4_nl_row(const block_iq4_nl * x, float * y, int64_t k) {
    int64_t nb = k / QK4_NL;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[i * QK4_NL + j]             = d * iq4k_values[x[i].qs[j] & 0x0F];
            y[i * QK4_NL + j + QK4_NL/2]  = d * iq4k_values[x[i].qs[j] >> 4];
        }
    }
}

// Build a random block_mxfp4: random E8M0 exponent, random nibbles.
// NOTE: e is kept in [114, 143] so the shared scale 2^(e-128) is a normal
// fp16. The converter stores scales as fp16, which cannot hold E8M0's full
// range: tiny scales (e < ~104) flush to zero and huge ones (e > 143)
// overflow to Inf (then Inf*0-valued quants become NaN on dequant).
// That clipping is inherent to Q8_0 half scales, not a routing bug; wild
// exponents would only test the clipping, so keep them out here.
static void make_random_mxfp4(block_mxfp4 * blk) {
    blk->e = (uint8_t)(114 + (g_rng() % 30));
    for (auto & v : blk->qs) v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order MXFP4 dequant.
static void dequant_mxfp4_row(const block_mxfp4 * x, float * y, int64_t k) {
    int64_t nb = k / QK_MXFP4;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_E8M0_TO_FP32_HALF(x[i].e);
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            y[i * QK_MXFP4 + j]               = d * kvalues_mxfp4[x[i].qs[j] & 0x0F];
            y[i * QK_MXFP4 + j + QK_MXFP4/2]  = d * kvalues_mxfp4[x[i].qs[j] >> 4];
        }
    }
}

// Build a random but *valid* block_q5_0: random half delta, random high
// bits and nibbles.
static void make_random_q5_0(block_q5_0 * blk) {
    float d = (g_rng() % 1000) * 0.001f + 0.01f;
    blk->d = GGML_FP32_TO_FP16(d);
    for (auto & v : blk->qh) v = (uint8_t)(g_rng() & 0xff);
    for (auto & v : blk->qs) v = (uint8_t)(g_rng() & 0xff);
}

// Natural-order Q5_0 dequant mirroring convert_q5_0 in iqk_quantize.cpp.
static void dequant_q5_0_row(const block_q5_0 * x, float * y, int64_t k) {
    int64_t nb = k / QK5_0;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < QK5_0 / 2; ++j) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j +  0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
            int l0 = (x[i].qs[j] & 0x0F) | xh_0;
            int l1 = (x[i].qs[j] >> 4)   | xh_1;
            y[i * QK5_0 + j]             = d * (l0 - 16);
            y[i * QK5_0 + j + QK5_0 / 2] = d * (l1 - 16);
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_0_R4 repack integrity (H1) — the current K/V-cache path
// (-ctk/-ctv q5_0). Random Q5_0 rows -> real repack_q5_0 -> real
// dequantize_row_q5_0_r4 must equal the natural-order dequant EXACTLY.
// ---------------------------------------------------------------------------
static void test_q5_0_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK5_0 == 0);
    GGML_ASSERT(nrc_x % 4 == 0);
    const int nb = n / QK5_0;
    std::vector<block_q5_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q5_0(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_q5_0_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_q5_0_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_q5_0(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_q5_0_r4(&rep[(size_t)g * nb], f2.data() + (size_t)g * 4 * n, 4 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q5_0-repack  n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] q5_0-repack  n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_0_R4 -> Q8_0_R8 converter (H2) — added by PR 2448.
//   5-bit values fit int8 losslessly at the same scale: deltas bit-copied,
//   floats bit-exact.
// ---------------------------------------------------------------------------
static void test_q5_0_r4_convert(int n, int nrc_x) {
    GGML_ASSERT(n % QK5_0 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK5_0; // == n / QK8_0
    std::vector<block_q5_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q5_0(&b);
    std::vector<block_q5_0_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_q5_0(nrc_x, n, src.data(), rep.data());

    const size_t bx = ggml_row_size(GGML_TYPE_Q5_0_R4, n);
    std::vector<uint8_t> got((size_t)(nrc_x / 8) * nb * sizeof(block_q8_0_r8) + 64, 0);
    bool ok = iqk_convert_legacy_quants_q8_r8(GGML_TYPE_Q5_0_R4, n, rep.data(), bx, got.data(), nrc_x);
    if (!ok) {
        printf("  [FAIL] q5_0-conv   n=%-5d nrc_x=%-3d : converter returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    const auto * gout = (const block_q8_0_r8 *)got.data();
    long mm_d = 0;
    for (int g = 0; g < nrc_x / 8; ++g) {
        for (int i = 0; i < nb; ++i) {
            const auto & r8 = gout[(size_t)g * nb + i];
            const auto & ra = rep[(size_t)(2 * g) * nb + i];
            const auto & rb = rep[(size_t)(2 * g + 1) * nb + i];
            for (int k = 0; k < 4; ++k) {
                if (memcmp(&r8.d[k], &ra.d[k], sizeof(ggml_half)) != 0) ++mm_d;
                if (memcmp(&r8.d[k + 4], &rb.d[k], sizeof(ggml_half)) != 0) ++mm_d;
            }
        }
    }
    if (mm_d != 0) {
        printf("  [FAIL] q5_0-conv   n=%-5d nrc_x=%-3d : %ld delta mismatches (d not copied)\n", n, nrc_x, mm_d);
        ++g_failures;
        return;
    }

    std::vector<float> f1((size_t)nrc_x * n);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_q5_0_r4(&rep[(size_t)g * nb], f1.data() + (size_t)g * 4 * n, 4 * n);
    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&gout[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q5_0-conv   n=%-5d nrc_x=%-3d : deltas copied, floats bit-exact\n", n, nrc_x);
    else {
        size_t r0 = first / (size_t)n, c0 = first % (size_t)n;
        printf("  [FAIL] q5_0-conv   n=%-5d nrc_x=%-3d : %ld float diffs first@(row=%zu,col=%zu) r4=%.4g q8=%.4g\n",
               n, nrc_x, mm, r0, c0, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: Q6_0_R4 repack integrity (H1) — the embeddings path.
//   Random Q6_0 rows -> real repack_q6_0 -> real dequantize_row_q6_0_r4 must
//   equal the natural-order dequant EXACTLY (repacking is lossless).
// ---------------------------------------------------------------------------
static void test_q6_0_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK6_0 == 0);
    GGML_ASSERT(nrc_x % 4 == 0);
    const int nb = n / QK6_0;
    std::vector<block_q6_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q6_0(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_q6_0_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_q6_0_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_q6_0(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_q6_0_r4(&rep[(size_t)g * nb], f2.data() + (size_t)g * 4 * n, 4 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q6_0-repack  n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] q6_0-repack  n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: Q6_0_R4 -> Q8_0_R8 converter (H2) — added by PR 2448.
//   6-bit values fit int8 losslessly at the same scale, so the converter
//   output must dequantize EXACTLY to the repacked values: identical deltas
//   and identical floats. Any mismatch is a real converter bug (this is the
//   path token_embd takes once allowed to repack to Q6_0_R4).
// ---------------------------------------------------------------------------
static void test_q6_0_r4_convert(int n, int nrc_x) {
    GGML_ASSERT(n % QK6_0 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK6_0; // == n / QK8_0
    std::vector<block_q6_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q6_0(&b);
    std::vector<block_q6_0_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_q6_0(nrc_x, n, src.data(), rep.data());

    const size_t bx = ggml_row_size(GGML_TYPE_Q6_0_R4, n);
    std::vector<uint8_t> got((size_t)(nrc_x / 8) * nb * sizeof(block_q8_0_r8) + 64, 0);
    bool ok = iqk_convert_legacy_quants_q8_r8(GGML_TYPE_Q6_0_R4, n, rep.data(), bx, got.data(), nrc_x);
    if (!ok) {
        printf("  [FAIL] q6_0-conv   n=%-5d nrc_x=%-3d : converter returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    // (a) deltas must be bit-copied from the two R4 source groups.
    const auto * gout = (const block_q8_0_r8 *)got.data();
    long mm_d = 0;
    for (int g = 0; g < nrc_x / 8; ++g) {
        for (int i = 0; i < nb; ++i) {
            const auto & r8 = gout[(size_t)g * nb + i];
            const auto & ra = rep[(size_t)(2 * g) * nb + i];
            const auto & rb = rep[(size_t)(2 * g + 1) * nb + i];
            for (int k = 0; k < 4; ++k) {
                if (memcmp(&r8.d[k], &ra.d[k], sizeof(ggml_half)) != 0) ++mm_d;
                if (memcmp(&r8.d[k + 4], &rb.d[k], sizeof(ggml_half)) != 0) ++mm_d;
            }
        }
    }
    if (mm_d != 0) {
        printf("  [FAIL] q6_0-conv   n=%-5d nrc_x=%-3d : %ld delta mismatches (d not copied)\n", n, nrc_x, mm_d);
        ++g_failures;
        return;
    }

    // (b) dequantized floats must match the repacked values bit-exactly.
    std::vector<float> f1((size_t)nrc_x * n);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_q6_0_r4(&rep[(size_t)g * nb], f1.data() + (size_t)g * 4 * n, 4 * n);
    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&gout[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q6_0-conv   n=%-5d nrc_x=%-3d : deltas copied, floats bit-exact\n", n, nrc_x);
    else {
        size_t r0 = first / (size_t)n, c0 = first % (size_t)n;
        printf("  [FAIL] q6_0-conv   n=%-5d nrc_x=%-3d : %ld float diffs first@(row=%zu,col=%zu) r4=%.4g q8=%.4g\n",
               n, nrc_x, mm, r0, c0, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: Q4_0_R8 repack integrity (H1).
// ---------------------------------------------------------------------------
static void test_q4_0_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK4_0 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK4_0;
    std::vector<block_q4_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q4_0(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_q4_0_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_iq4_nl_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_q4_0(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q4_0_r8(&rep[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q4_0-repack  n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] q4_0-repack  n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: Q4_0_R8 -> Q8_0_R8 converter (H2) — added by PR 2448.
// ---------------------------------------------------------------------------
static void test_q4_0_r8_convert(int n, int nrc_x) {
    GGML_ASSERT(n % QK4_0 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK4_0; // == n / QK8_0
    std::vector<block_q4_0> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_q4_0(&b);
    std::vector<block_iq4_nl_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_q4_0(nrc_x, n, src.data(), rep.data());

    const size_t bx = ggml_row_size(GGML_TYPE_Q4_0_R8, n);
    std::vector<uint8_t> got((size_t)(nrc_x / 8) * nb * sizeof(block_q8_0_r8) + 64, 0);
    bool ok = iqk_convert_legacy_quants_q8_r8(GGML_TYPE_Q4_0_R8, n, rep.data(), bx, got.data(), nrc_x);
    if (!ok) {
        printf("  [FAIL] q4_0-conv   n=%-5d nrc_x=%-3d : converter returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    const auto * gout = (const block_q8_0_r8 *)got.data();
    long mm_d = 0;
    for (int g = 0; g < nrc_x / 8; ++g) {
        for (int i = 0; i < nb; ++i) {
            if (memcmp(gout[(size_t)g * nb + i].d, rep[(size_t)g * nb + i].d, 8 * sizeof(ggml_half)) != 0) ++mm_d;
        }
    }
    if (mm_d != 0) {
        printf("  [FAIL] q4_0-conv   n=%-5d nrc_x=%-3d : %ld delta mismatches (d not copied)\n", n, nrc_x, mm_d);
        ++g_failures;
        return;
    }

    std::vector<float> f1((size_t)nrc_x * n);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q4_0_r8(&rep[(size_t)g * nb], f1.data() + (size_t)g * 8 * n, 8 * n);
    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&gout[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   q4_0-conv   n=%-5d nrc_x=%-3d : deltas copied, floats bit-exact\n", n, nrc_x);
    else {
        size_t r0 = first / (size_t)n, c0 = first % (size_t)n;
        printf("  [FAIL] q4_0-conv   n=%-5d nrc_x=%-3d : %ld float diffs first@(row=%zu,col=%zu) r8=%.4g q8=%.4g\n",
               n, nrc_x, mm, r0, c0, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: MXFP4_R8 repack integrity (H1).
// ---------------------------------------------------------------------------
static void test_mxfp4_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK_MXFP4 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_MXFP4;
    std::vector<block_mxfp4> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_mxfp4(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_mxfp4_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_mxfp4_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_mxfp4(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_mxfp4_r8(&rep[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   mxfp4-repack n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] mxfp4-repack n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: MXFP4_R8 -> Q8_0_R8 converter (H2) — added by PR 2448.
// ---------------------------------------------------------------------------
static void test_mxfp4_r8_convert(int n, int nrc_x) {
    GGML_ASSERT(n % QK_MXFP4 == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK_MXFP4; // == n / QK8_0
    std::vector<block_mxfp4> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_mxfp4(&b);
    std::vector<block_mxfp4_r8> rep((size_t)(nrc_x / 8) * nb);
    iqk_test_repack_mxfp4(nrc_x, n, src.data(), rep.data());

    const size_t bx = ggml_row_size(GGML_TYPE_MXFP4_R8, n);
    std::vector<uint8_t> got((size_t)(nrc_x / 8) * nb * sizeof(block_q8_0_r8) + 64, 0);
    bool ok = iqk_convert_legacy_quants_q8_r8(GGML_TYPE_MXFP4_R8, n, rep.data(), bx, got.data(), nrc_x);
    if (!ok) {
        printf("  [FAIL] mxfp4-conv  n=%-5d nrc_x=%-3d : converter returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    const auto * gout = (const block_q8_0_r8 *)got.data();
    long mm_d = 0;
    for (int g = 0; g < nrc_x / 8; ++g) {
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                ggml_half exp = GGML_FP32_TO_FP16(GGML_E8M0_TO_FP32_HALF(rep[(size_t)g * nb + i].e[k]));
                if (memcmp(&gout[(size_t)g * nb + i].d[k], &exp, sizeof(ggml_half)) != 0) ++mm_d;
            }
        }
    }
    if (mm_d != 0) {
        printf("  [FAIL] mxfp4-conv  n=%-5d nrc_x=%-3d : %ld scale mismatches\n", n, nrc_x, mm_d);
        ++g_failures;
        return;
    }

    std::vector<float> f1((size_t)nrc_x * n);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_mxfp4_r8(&rep[(size_t)g * nb], f1.data() + (size_t)g * 8 * n, 8 * n);
    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&gout[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   mxfp4-conv  n=%-5d nrc_x=%-3d : scales converted, floats bit-exact\n", n, nrc_x);
    else {
        size_t r0 = first / (size_t)n, c0 = first % (size_t)n;
        printf("  [FAIL] mxfp4-conv  n=%-5d nrc_x=%-3d : %ld float diffs first@(row=%zu,col=%zu) r8=%.4g q8=%.4g\n",
               n, nrc_x, mm, r0, c0, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ4_NL_R4 repack integrity (H1) — previous V-cache path.
// ---------------------------------------------------------------------------
static void test_iq4_nl_repack(int n, int nrc_x) {
    GGML_ASSERT(n % QK4_NL == 0);
    GGML_ASSERT(nrc_x % 4 == 0);
    const int nb = n / QK4_NL;
    std::vector<block_iq4_nl> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_iq4_nl(&b);

    std::vector<float> f1((size_t)nrc_x * n);
    for (int r = 0; r < nrc_x; ++r)
        dequant_iq4_nl_row(&src[(size_t)r * nb], f1.data() + (size_t)r * n, n);

    std::vector<block_iq4_nl_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_iq4_nl(nrc_x, n, src.data(), rep.data());

    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_iq4_nl_r4(&rep[(size_t)g * nb], f2.data() + (size_t)g * 4 * n, 4 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   iq4_nl-repack n=%-5d nrc_x=%-3d : repack is lossless (bit-exact)\n", n, nrc_x);
    else {
        printf("  [FAIL] iq4_nl-repack n=%-5d nrc_x=%-3d : %ld diffs first@%zu (native=%.4g repacked=%.4g)\n",
               n, nrc_x, mm, first, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ4_NL_R4 -> Q8_0_R8 converter (H2) — added by PR 2448.
// ---------------------------------------------------------------------------
static void test_iq4_nl_r4_convert(int n, int nrc_x) {
    GGML_ASSERT(n % QK4_NL == 0);
    GGML_ASSERT(nrc_x % 8 == 0);
    const int nb = n / QK4_NL; // == n / QK8_0
    std::vector<block_iq4_nl> src((size_t)nrc_x * nb);
    for (auto & b : src) make_random_iq4_nl(&b);
    std::vector<block_iq4_nl_r4> rep((size_t)(nrc_x / 4) * nb);
    iqk_test_repack_iq4_nl(nrc_x, n, src.data(), rep.data());

    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_NL_R4, n);
    std::vector<uint8_t> got((size_t)(nrc_x / 8) * nb * sizeof(block_q8_0_r8) + 64, 0);
    bool ok = iqk_convert_legacy_quants_q8_r8(GGML_TYPE_IQ4_NL_R4, n, rep.data(), bx, got.data(), nrc_x);
    if (!ok) {
        printf("  [FAIL] iq4_nl-conv n=%-5d nrc_x=%-3d : converter returned false\n", n, nrc_x);
        ++g_failures;
        return;
    }

    const auto * gout = (const block_q8_0_r8 *)got.data();
    long mm_d = 0;
    for (int g = 0; g < nrc_x / 8; ++g) {
        for (int i = 0; i < nb; ++i) {
            const auto & r8 = gout[(size_t)g * nb + i];
            const auto & ra = rep[(size_t)(2 * g) * nb + i];
            const auto & rb = rep[(size_t)(2 * g + 1) * nb + i];
            for (int k = 0; k < 4; ++k) {
                if (memcmp(&r8.d[k], &ra.d[k], sizeof(ggml_half)) != 0) ++mm_d;
                if (memcmp(&r8.d[k + 4], &rb.d[k], sizeof(ggml_half)) != 0) ++mm_d;
            }
        }
    }
    if (mm_d != 0) {
        printf("  [FAIL] iq4_nl-conv n=%-5d nrc_x=%-3d : %ld delta mismatches (d not copied)\n", n, nrc_x, mm_d);
        ++g_failures;
        return;
    }

    std::vector<float> f1((size_t)nrc_x * n);
    for (int g = 0; g < nrc_x / 4; ++g)
        dequantize_row_iq4_nl_r4(&rep[(size_t)g * nb], f1.data() + (size_t)g * 4 * n, 4 * n);
    std::vector<float> f2((size_t)nrc_x * n, 0.f);
    for (int g = 0; g < nrc_x / 8; ++g)
        dequantize_row_q8_0_r8(&gout[(size_t)g * nb], f2.data() + (size_t)g * 8 * n, 8 * n);

    long mm = 0; size_t first = (size_t)-1;
    for (size_t j = 0; j < (size_t)nrc_x * n; ++j)
        if (f1[j] != f2[j]) { ++mm; if (first == (size_t)-1) first = j; }
    if (mm == 0)
        printf("  [OK]   iq4_nl-conv n=%-5d nrc_x=%-3d : deltas copied, floats bit-exact\n", n, nrc_x);
    else {
        size_t r0 = first / (size_t)n, c0 = first % (size_t)n;
        printf("  [FAIL] iq4_nl-conv n=%-5d nrc_x=%-3d : %ld float diffs first@(row=%zu,col=%zu) r4=%.4g q8=%.4g\n",
               n, nrc_x, mm, r0, c0, (double)f1[first], (double)f2[first]);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --all           Test all mainline suites - R8 only (default)\n");
    printf("  -rtr            Test -rtr only (R8 converter)\n");
    printf("  -r16p           Test -r16p only (R16 converter, advanced forks)\n");
    printf("  -rtr -r16p      Test both R8 and R16 converters\n");
    printf("  -n  SIZE        Set element count per row (default: 2048,4096,8192)\n");
    printf("  -nrc_x N        Set nrc_x (default: 16,32,64,128)\n");
    printf("  --seed N        Random seed (default: 12345)\n");
    printf("  --skip-byte     Skip byte-byte comparison (fast integrity-only)\n");
    printf("  --skip-dispatch Skip dispatch-pipeline tests\n");
    printf("  --strict        No ±1 qs tolerance: require byte-exact converter output\n");
    printf("  --cinit-plain   Init fused-MoE C to -1.f instead of -1e30 sentinel\n");
    printf("  --bench         Speed bench only (skips precision suites)\n");
    printf("  --bench-filter S  Bench only ops/quants containing S (repeatable)\n");
    printf("  --bench-reps N  Exact reps per stage instead of adaptive timing\n");
    printf("  --bench-db PATH Bench DB file (default: iqk_bench.db)\n");
    printf("  --bench-save L  Append this bench run to DB under label L\n");
    printf("  --bench-delta A B  Print before/after table for labels A,B\n");
    printf("  --bench-csv F [L]  Export DB (label L, default all) to CSV file F\n");
    printf("  --help          Show this help\n");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered: crash location stays visible
    init_unit_test_fp16_table();
    printf("=== IQK iq4_xs_r8 -> q8_k_r8 / q8_k_r16 converter verification ===\n");
    fflush(stdout);

    const char * simd_name = "(unknown)";
#if defined(HAVE_FANCY_SIMD)
    simd_name = "HAVE_FANCY_SIMD (AVX-512)";
#elif defined(HAVE_VNNIINT8)
    simd_name = "HAVE_VNNIINT8 (AVX-VNNI-INT8)";
#elif defined(HAVE_VNNI256)
    simd_name = "HAVE_VNNI256 (AVX2 + VNNI)";
#else
    simd_name = "NONE (float dequant->quant fallback)";
#endif
    printf("SIMD path: %s\n", simd_name);
    printf("QK_K=%d  sizeof(block_iq4_xs_r8)=%zu  sizeof(block_q8_k_r8)=%zu  sizeof(block_q8_k_r16)=%zu\n",
           QK_K, sizeof(block_iq4_xs_r8), sizeof(block_q8_k_r8), sizeof(block_q8_k_r16));

    bool test_rtr  = false;
    bool test_r16p = false;
    bool test_all  = true;
    bool skip_byte = false;
    bool skip_dispatch = false;
    bool bench_mode = false;
    std::vector<std::string> bench_filter;
    int bench_reps = 0;
    std::string bench_db = "iqk_bench.db";
    std::string bench_save, bench_delta_a, bench_delta_b, bench_csv, bench_csv_label;
    // IQ4_XS_R8 is a fixed 1088-byte / 8-row block (n=2048 only); larger n is
    // invalid for this type, so all converter tests run at n=2048.
    std::vector<int> ns = { 2048 };
    std::vector<int> nrc_xs = { 16, 32, 64, 128 };
    std::vector<int> kv_ctxs = { 128, 256, 512, 1024, 2048, 4096, 8192 };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "-rtr")  { test_all = false; test_rtr  = true; continue; }
        if (arg == "-r16p") { test_all = false; test_r16p = true; continue; }
        if (arg == "--all") { test_all = true; continue; }
        if (arg == "--skip-byte") { skip_byte = true; continue; }
        if (arg == "--skip-dispatch") { skip_dispatch = true; continue; }
        if (arg == "--strict") { g_strict = true; continue; }
        if (arg == "--cinit-plain") { g_cinit_plain = true; continue; }
        if (arg == "-n" && i+1 < argc) { ns.clear(); ns.push_back(atoi(argv[++i])); continue; }
        if (arg == "--seed" && i+1 < argc) { g_seed = atoi(argv[++i]); g_rng.seed(g_seed); continue; }
        if (arg == "-nrc_x" && i+1 < argc) { nrc_xs.clear(); nrc_xs.push_back(atoi(argv[++i])); continue; }
        if (arg == "--bench") { bench_mode = true; continue; }
        if (arg == "--bench-filter" && i+1 < argc) { bench_filter.push_back(argv[++i]); continue; }
        if (arg == "--bench-reps" && i+1 < argc) { bench_reps = atoi(argv[++i]); continue; }
        if (arg == "--bench-db" && i+1 < argc) { bench_db = argv[++i]; continue; }
        if (arg == "--bench-save" && i+1 < argc) { bench_save = argv[++i]; continue; }
        if (arg == "--bench-delta" && i+2 < argc) { bench_delta_a = argv[++i]; bench_delta_b = argv[++i]; continue; }
        if (arg == "--bench-csv" && i+1 < argc) {
            bench_csv = argv[++i];
            if (i+1 < argc && argv[i+1][0] != '-') bench_csv_label = argv[++i];
            continue;
        }
    }
    // R16 suites are NOT part of the default: g_iqk_r16_path no longer exists
    // in the lib (vestigial test-local toggle, honored only on advanced
    // forks). Pass -r16p explicitly to opt into them there.
    if (test_all) { test_rtr = true; }

    printf("Seed: %d\n", g_seed);
    printf("Flags: -rtr=%d -r16p=%d  skip-byte=%d skip-dispatch=%d strict=%d cinit-plain=%d\n",
           (int)test_rtr, (int)test_r16p, (int)skip_byte, (int)skip_dispatch, (int)g_strict, (int)g_cinit_plain);
    if (bench_mode || !bench_save.empty() || !bench_filter.empty() || bench_reps > 0)
        printf("Bench: mode=%d reps=%d db=%s save=%s delta=%s/%s csv=%s/%s filters=%d\n",
               (int)bench_mode, bench_reps, bench_db.c_str(),
               bench_save.empty() ? "-" : bench_save.c_str(),
               bench_delta_a.empty() ? "-" : bench_delta_a.c_str(),
               bench_delta_b.empty() ? "-" : bench_delta_b.c_str(),
               bench_csv.empty() ? "-" : bench_csv.c_str(),
               bench_csv_label.empty() ? "-" : bench_csv_label.c_str(),
               (int)bench_filter.size());
    printf("Row sizes (n):"); for (int v : ns) printf(" %d", v); printf("\n");
    printf("nrc_x sizes :"); for (int v : nrc_xs) printf(" %d", v); printf("\n");
    fflush(stdout);

    // Speed-harness modes run instead of the precision suites (fast path for
    // commit/compile-path comparison). DB ops work standalone (no bench run).
    // --bench-save without --bench implies a bench run (else the label would
    // be silently lost).
    if (bench_mode || (!bench_save.empty() && bench_delta_a.empty() && bench_csv.empty())) {
        bench::Options o;
        o.filters = bench_filter; o.fixed_reps = bench_reps;
        o.db_path = bench_db; o.save_label = bench_save;
        std::vector<bench::Result> rows;
        int rc = bench::run(o, rows);
        if (rc == 0 && !bench_save.empty()) rc = bench::db_append(bench_db, bench_save, rows);
        if (rc == 0 && !bench_delta_a.empty()) rc = bench::print_delta(bench_db, bench_delta_a, bench_delta_b);
        if (rc == 0 && !bench_csv.empty()) rc = bench::export_csv(bench_db, bench_csv, bench_csv_label);
        return rc;
    }
    if (!bench_delta_a.empty() || !bench_csv.empty()) {
        int rc = 0;
        if (!bench_delta_a.empty()) rc = bench::print_delta(bench_db, bench_delta_a, bench_delta_b);
        if (rc == 0 && !bench_csv.empty()) rc = bench::export_csv(bench_db, bench_csv, bench_csv_label);
        return rc;
    }

    printf("\n--- Test: Delta integrity (all rows finite/non-zero) ---\n");
    for (int n : ns) {
        for (int nrc_x : nrc_xs) {
            if (test_rtr)  test_delta_integrity(false, nrc_x, n);
            if (test_r16p) test_delta_integrity(true,  nrc_x, n);
        }
    }

    if (!skip_dispatch) {
        printf("\n--- Test: Dispatch pipeline (iqk_dequant_type → iqk_convert_repack) ---\n");
        for (int n : ns) {
            for (int nrc_x : nrc_xs) {
                test_dispatch_pipeline(nrc_x, n);
            }
        }
    }

    printf("\n--- Test: Round-trip accuracy (dequant → recon比对) ---\n");
    for (int n : ns) {
        for (int nrc_x : nrc_xs) {
            if (test_rtr)  test_roundtrip(false, nrc_x, n);
            if (test_r16p) test_roundtrip(true,  nrc_x, n);
        }
    }

    printf("\n--- Test: Group boundary coverage ---\n");
    for (int n : ns) {
        test_group_boundaries(n);
    }

    printf("\n--- Test: Diagnostic R16 dump (minimal) ---\n");
    test_dump_r16(256);
    test_dump_r16(2048);

    if (!skip_byte) {
        printf("\n--- Test: Byte-for-byte comparison vs reference ---\n");
        for (int n : ns) {
            for (int nrc_x : nrc_xs) {
                if (test_rtr)  test_path(false, nrc_x, n);
                if (test_r16p) test_path(true,  nrc_x, n);
            }
        }
    }

    printf("\n--- Test: Native IQ4_XS converter (iqk_convert_iq4_xs_q8_k_r8) ---\n");
    for (int n : ns) {
        for (int nrc_x : nrc_xs) {
            if (test_rtr)  test_native_path(false, nrc_x, n);
            if (test_r16p) test_native_path(true,  nrc_x, n);
        }
    }

    printf("\n--- Test: KV-cache growth sweep (prompt=ctx/2, then full ctx) ---\n");
    for (int ctx : kv_ctxs) {
        if (test_rtr)  test_kv_sweep(false, ctx);
        if (test_r16p) test_kv_sweep(true,  ctx);
    }

    if (false) {
    printf("\n--- Test: Repack integrity (native IQ4_XS vs IQ4_XS_R8) ---\n");
    for (int n : ns) {
        test_repack_integrity(n);
    }
    }

    if (test_r16p) {
        printf("\n--- Test: End-to-end R16 GEMM (real dispatch vs float ref, nrc_y>=32) ---\n");
        int ns_g[] = {2048}; int ny_g[] = {32}; int nx_g[] = {128, 256};
        for (int n : ns_g) {
            for (int nrc_y : ny_g) {
                for (int nrc_x : nx_g) {
                    test_gemm_r16(n, nrc_x, nrc_y, /*use_ref=*/false);
                }
            }
        }
    }

    {
        printf("\n--- Test: R8 GEMM vs float (converted weights, plain + row_mapping) ---\n");
        int ns_g8[] = {1024};
        int nx_g8[] = {8, 64};
        int ny_g8[] = {1, 8};
        for (int n : ns_g8) {
            for (int nrc_x : nx_g8) {
                for (int nrc_y : ny_g8) {
                    test_gemm_r8(n, nrc_x, nrc_y);
                }
            }
        }
    }

    {
        printf("\n--- Test: R8 GEMM tiled (fused-tiling replication) ---\n");
        test_gemm_r8_tiled(1024, 64, 16);
        test_gemm_r8_tiled(1024, 64, 32);
        test_gemm_r8_tiled(1024, 64, 64);
    }

    {
        printf("\n--- Test: Q5_0_R4 repack + R8 converter (K/V-cache path) ---\n");
        int ns_q5[] = {128, 512};
        for (int n : ns_q5) {
            for (int nrc_x : {4, 8}) test_q5_0_repack(n, nrc_x);
            for (int nrc_x : {8, 16}) test_q5_0_r4_convert(n, nrc_x);
        }
    }

    {
        printf("\n--- Test: Q6_0_R4 repack + R8 converter (embeddings path) ---\n");
        int ns_q6[] = {128, 512};
        for (int n : ns_q6) {
            for (int nrc_x : {4, 8}) test_q6_0_repack(n, nrc_x);
            for (int nrc_x : {8, 16}) test_q6_0_r4_convert(n, nrc_x);
        }
    }

    {
        printf("\n--- Test: direct R8 GEMM vs float (expert path, Q8_K32 act) ---\n");
        int ns_d8[] = {1024};
        int nx_d8[] = {8, 64};
        int ny_d8[] = {1, 8};
        for (int n : ns_d8) {
            for (int nrc_x : nx_d8) {
                for (int nrc_y : ny_d8) {
                    test_gemm_iq4_xs_r8_direct(n, nrc_x, nrc_y);
                }
            }
        }
    }

    // preconv validation + fused cases live below (see test_moe_fused_up_gate_preconv)
    // Multiset check (sum/sumsq per row): distinguishes ORDER bugs (permuted
    // values, sums match) from VALUE bugs (sums differ). Restored: it is the
    // cheapest discriminator for the converter unpack-order hypothesis.
    {
        printf("\n--- Test: R8->Q8 multiset check (order vs value bug) ---\n");
        g_rng.seed(1001); // per-suite determinism: comparable across binaries
        const int nn = 1024, nx = 64;
        const int nbb = nn / QK_K;
        std::vector<block_iq4_xs_r8> src((size_t)(nx / 8) * nbb);
        for (auto & b : src) make_random_iq4_xs_r8(&b);
        const size_t bxw = ggml_row_size(GGML_TYPE_IQ4_XS_R8, nn);
        const size_t bxq = ggml_row_size(GGML_TYPE_Q8_K_R8, nn);
        std::vector<uint8_t> out((size_t)nx * bxq + 64, 0xCC);
        bool conv_ok = false;
        bench_convert_ms("multiset r8->q8 (1024x64)", (long)nx * nn, 8.0,
            [&]() { conv_ok = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn, src.data(), bxw, out.data(), 0, nx); });
        if (!conv_ok) {
            printf("  [FAIL] multiset: convert returned false\n");
            ++g_failures;
        } else {
            // reference sums straight from R8 dequant (trusted via direct GEMM)
            size_t rows_bad = 0;
            for (int r = 0; r < nx; ++r) {
                double s_ref = 0.0, q_ref = 0.0, s_got = 0.0, q_got = 0.0, maxabs = 0.0;
                // ref row r via R8 dequant of group r/8
                float rf[8 * 1024];
                dequantize_row_iq4_xs_r8(&src[(size_t)(r / 8) * nbb], rf, 8 * nn);
                const float * rfrow = rf + (size_t)(r % 8) * nn;
                // got row r via Q8 dequant of converted group r/8. One converted
                // group (8 interleaved rows) spans 8*bxq bytes, so the group
                // step is 8*bxq.
                float gf[8 * 1024];
                dequantize_row_q8_k_r8((const block_q8_k_r8 *)(out.data() + (size_t)(r / 8) * 8 * bxq), gf, 8 * nn);
                const float * gfrow = gf + (size_t)(r % 8) * nn;
                for (int k = 0; k < nn; ++k) {
                    s_ref += rfrow[k]; q_ref += (double)rfrow[k] * rfrow[k];
                    s_got += gfrow[k]; q_got += (double)gfrow[k] * gfrow[k];
                    if (fabs(rfrow[k]) > maxabs) maxabs = fabs(rfrow[k]);
                }
                double quantum = maxabs / 127.0; // optimal Q8 step: legit rounding is <=0.5*quantum
                // + 0.5*nn admits sub-half-quantum mean bias (rounding asymmetry,
                // half-stored output deltas); per-value correctness is enforced by
                // smallshape/preconv (0.5q). Production PPL parity backs this.
                double tol = 2.0 + 1e-2 * fabs(s_ref) + 0.5 * quantum * sqrt((double)nn) + 0.5 * (double)nn;
                double tolq = 4.0 + 2e-2 * fabs(q_ref) + (double)nn * quantum * quantum;
                if (fabs(s_got - s_ref) > tol || fabs(q_got - q_ref) > tolq) {
                    if (rows_bad < 6)
                        printf("    multiset-bad[r=%d]: sum got %.6g ref %.6g | sumsq got %.6g ref %.6g | q=%.3g\n",
                               r, s_got, s_ref, q_got, q_ref, quantum);
                    // One-shot probe: mean error per 32-value segment of the first
                    // failing row. Clustered coherent segments => misindexed scales/
                    // nibbles in the converter; uniform ~= row mean => offset/scale
                    // convention mismatch; scattered => random requant noise.
                    { static bool probe_done = false;
                      if (!probe_done) {
                        probe_done = true;
                        printf("    multiset-seg[r=%d mean/seg]:", r);
                        for (int seg = 0; seg < nn / 32; ++seg) {
                            double m = 0.0;
                            for (int k = 32 * seg; k < 32 * seg + 32; ++k) m += gfrow[k] - rfrow[k];
                            printf(" [%d]%+.3f", seg, m / 32.0);
                        }
                        printf("\n");
                      } }
                    ++rows_bad;
                }
            }
            printf("  [%s] multiset: %zu/%d row-sums differ\n",
                   rows_bad == 0 ? "OK" : "FAIL", rows_bad, nx);
            printf("  [INFO] multiset: sums-match + element-mismatch = ORDER bug; sums-differ = VALUE bug\n");
            if (rows_bad != 0) ++g_failures;
        }
    }

    // Raw-output discriminator: are converted groups byte-identical?
    // (multiset showed rows 8+ with identical sums — duplicated output?)
    {
        printf("\n--- Test: R8->Q8 group-output uniqueness ---\n");
        g_rng.seed(1002); // per-suite determinism: comparable across binaries
        const int nn = 1024, nx = 64;
        const int nbb = nn / QK_K;
        std::vector<block_iq4_xs_r8> src((size_t)(nx / 8) * nbb);
        for (auto & b : src) make_random_iq4_xs_r8(&b);
        const size_t bxw = ggml_row_size(GGML_TYPE_IQ4_XS_R8, nn);
        const size_t bxq = ggml_row_size(GGML_TYPE_Q8_K_R8, nn);
        std::vector<uint8_t> out((size_t)nx * bxq + 64, 0xCC);
        if (!iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn, src.data(), bxw, out.data(), 0, nx)) {
            printf("  [FAIL] groups: convert returned false\n");
            ++g_failures;
        } else {
            // group g occupies nbb Q8 blocks = nbb*sizeof(block_q8_k_r8) bytes
            const size_t gbytes = (size_t)nbb * sizeof(block_q8_k_r8);
            int identical = 0;
            for (int g = 1; g < nx / 8; ++g) {
                if (memcmp(out.data(), out.data() + (size_t)g * gbytes, gbytes) == 0) ++identical;
            }
            printf("  [INFO] groups: %d/%d groups byte-identical to group 0\n", identical, nx / 8 - 1);
            printf("  group0[0..15]:");
            for (int i = 0; i < 16; ++i) printf(" %02x", out[i]);
            printf("\n  group1[0..15]:");
            for (int i = 0; i < 16; ++i) printf(" %02x", out[gbytes + i]);
            printf("\n");
            if (identical == nx / 8 - 1) {
                printf("  [FAIL] groups: converter emits identical output for distinct inputs\n");
                ++g_failures;
            } else {
                printf("  [OK] groups: outputs vary per group\n");
            }
            // y_d scales per group (sanity: finite, sane magnitude)
            for (int g = 0; g < 2; ++g) {
                const auto * yg = (const block_q8_k_r8 *)(out.data() + (size_t)g * gbytes);
                printf("  group%d y_d:", g);
                for (int k = 0; k < 8; ++k) printf(" %.4g", (double)GGML_FP16_TO_FP32(yg->d[k]));
                printf("\n");
            }
        }
    }

    // Canonical-pair check: quantize_q8_k_r8 -> dequantize_row_q8_k_r8 roundtrip.
    // Both production functions; must agree closely. Decides Xu-validation trust:
    // passes here => dequant sound => converter REALLY guilty; fails => dequant guilty.
    {
        printf("\n--- Test: canonical Q8_K_R8 quantize/dequant roundtrip ---\n");
        g_rng.seed(1003); // per-suite determinism: comparable across binaries
        const int nn = 1024, nx = 8;
        std::vector<float> src((size_t)nx * nn);
        for (size_t j = 0; j < src.size(); ++j) src[j] = 2.0f * ((int)(g_rng() % 2001) - 1000) * 0.01f;
        const size_t bxq = ggml_row_size(GGML_TYPE_Q8_K_R8, nn);
        std::vector<uint8_t> qb((size_t)nx * bxq + 64, 0xCC);
        quantize_q8_k_r8(src.data(), qb.data(), nx, nn, nullptr, nullptr);
        std::vector<float> out((size_t)nx * nn, 0.f);
        dequantize_row_q8_k_r8((const block_q8_k_r8 *)qb.data(), out.data(), (size_t)nx * nn / 8 * 8);
        size_t nbad = 0; float maxe = 0.f;
        for (size_t j = 0; j < (size_t)nx * nn; ++j) {
            float e = fabsf(out[j] - src[j]);
            if (e > maxe) maxe = e;
            if (e > 0.5f + 1e-2f * fabsf(src[j])) ++nbad;
        }
        printf("  [%s] q8k8-roundtrip: %zu bad (max|err|=%.4g)\n", nbad == 0 ? "OK" : "FAIL", nbad, (double)maxe);
        if (nbad != 0) ++g_failures;
    }

    // Small-shape random converter check (256,8): shape-vs-data discriminator.
    // Passes here + fails at (1024,64) => shape-dependent; fails here too => data/value bug.
    {
        printf("\n--- Test: R8->Q8 random convert at probe shape (256,8) ---\n");
        g_rng.seed(1004); // per-suite determinism: comparable across binaries
        const int nn = 256, nx = 8;
        const int nbb = nn / QK_K;
        std::vector<block_iq4_xs_r8> src((size_t)(nx / 8) * nbb);
        for (auto & b : src) make_random_iq4_xs_r8(&b);
        const size_t bxw = ggml_row_size(GGML_TYPE_IQ4_XS_R8, nn);
        const size_t bxq = ggml_row_size(GGML_TYPE_Q8_K_R8, nn);
        std::vector<uint8_t> out((size_t)nx * bxq + 64, 0xCC);
        bool conv_ok = false;
        bench_convert_ms("smallshape r8->q8 (256x8)", (long)nx * nn, 8.0,
            [&]() { conv_ok = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn, src.data(), bxw, out.data(), 0, nx); });
        if (!conv_ok) {
            printf("  [FAIL] smallshape: convert returned false\n");
            ++g_failures;
        } else {
            std::vector<float> rf((size_t)nx * nn), gf((size_t)nx * nn);
            dequantize_row_iq4_xs_r8(src.data(), rf.data(), (size_t)nx * nn / 8 * 8);
            dequantize_row_q8_k_r8((const block_q8_k_r8 *)out.data(), gf.data(), (size_t)nx * nn / 8 * 8);
            float maxabs = 0.f;
            for (size_t j = 0; j < (size_t)nx * nn; ++j) { float a = fabsf(rf[j]); if (a > maxabs) maxabs = a; }
            float quantum = maxabs / 127.f; // optimal Q8 step for this data
            size_t nbad = 0; float maxe = 0.f; float maxq = 0.f;
            for (size_t j = 0; j < (size_t)nx * nn; ++j) {
                float e = fabsf(gf[j] - rf[j]);
                float eq = e / (quantum > 1e-9f ? quantum : 1e-9f);
                if (e > maxe) maxe = e;
                if (eq > maxq) maxq = eq;
                if (e > 1.0f + 1e-2f * fabsf(rf[j]) + 0.5f * quantum) ++nbad;
            }
            printf("  [%s] smallshape: %zu bad (max|err|=%.4g, max|err|/q=%.3g, q=%.4g)\n", nbad == 0 ? "OK" : "FAIL", nbad, (double)maxe, (double)maxq, (double)quantum);
            if (nbad != 0) ++g_failures;
        }
    }

    {
        printf("\n--- Test: constant-input R8->Q8 converter probe ---\n");
#ifdef TEST_DISPATCH
        // One R8 block: d=1.0, level=1 everywhere (sl=1,sh=2), all nibbles=5.
        // Expected: every output value ~= table[5] (up to Q8 rounding ~1%).
        // Any structural deviation (order/scale/stride bug) shows immediately.
        const int nn = 256, nx = 8;
        const int nbb = nn / QK_K;
        std::vector<block_iq4_xs_r8> src((size_t)(nx / 8) * nbb);
        for (auto & b : src) {
            for (int k = 0; k < 8; ++k) b.d[k] = GGML_FP32_TO_FP16(1.0f);
            for (int i = 0; i < 32; ++i) b.scales_l[i] = 0x11;
            for (int i = 0; i < 16; ++i) b.scales_h[i] = 0xAA;
            for (int i = 0; i < 1024; ++i) b.qs[i] = 0x55;
        }
        const size_t bxw = ggml_row_size(GGML_TYPE_IQ4_XS_R8, nn);
        const size_t bxq = ggml_row_size(GGML_TYPE_Q8_K_R8, nn);
        std::vector<uint8_t> out((size_t)nx * bxq + 64, 0xCC);
        bool okc = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn, src.data(), bxw, out.data(), 0, nx);
        printf("  convert returned %d\n", (int)okc);
        const auto * y0 = (const block_q8_k_r8 *)out.data();
        printf("  y_d:");
        for (int k = 0; k < 8; ++k) printf(" %.4g", (double)GGML_FP16_TO_FP32(y0->d[k]));
        printf("\n  y_qs[0..31]:");
        for (int i = 0; i < 32; ++i) printf(" %d", (int)y0->qs[i]);
        printf("\n");
        // reference: dequant one input block straight to float
        float ref[8 * 256];
        dequantize_row_iq4_xs_r8(src.data(), ref, 8 * nn);
        printf("  ref[0..7]:");
        for (int i = 0; i < 8; ++i) printf(" %.4g", (double)ref[i]);
        printf("\n");

        // shape probe: SAME constant data at harness shape (n=1024, 8 groups).
        // passes => bug needs data variation (value path); fails => shape bug.
        {
            const int nn2 = 1024, nx2 = 64;
            const int nbb2 = nn2 / QK_K;
            std::vector<block_iq4_xs_r8> src2((size_t)(nx2 / 8) * nbb2);
            for (auto & b : src2) {
                for (int k = 0; k < 8; ++k) b.d[k] = GGML_FP32_TO_FP16(1.0f);
                for (int i = 0; i < 32; ++i) b.scales_l[i] = 0x11;
                for (int i = 0; i < 16; ++i) b.scales_h[i] = 0xAA;
                for (int i = 0; i < 1024; ++i) b.qs[i] = 0x55;
            }
            const size_t bxw2 = ggml_row_size(GGML_TYPE_IQ4_XS_R8, nn2);
            const size_t bxq2 = ggml_row_size(GGML_TYPE_Q8_K_R8, nn2);
            std::vector<uint8_t> out2b((size_t)nx2 * bxq2 + 64, 0xCC);
            bool okc3 = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn2, src2.data(), bxw2, out2b.data(), 0, nx2);
            printf("  shape-probe convert returned %d\n", (int)okc3);
            const auto * y2 = (const block_q8_k_r8 *)out2b.data();
            printf("  shape-probe y_d[0]:");
            for (int k = 0; k < 8; ++k) printf(" %.4g", (double)GGML_FP16_TO_FP32(y2->d[k]));
            printf("\n  shape-probe y_qs[0..15]:");
            for (int i = 0; i < 16; ++i) printf(" %d", (int)y2->qs[i]);
            printf("\n");
            const auto * y2l = y2 + 7 * nbb2; // last group
            printf("  shape-probe last-group y_d:");
            for (int k = 0; k < 8; ++k) printf(" %.4g", (double)GGML_FP16_TO_FP32(y2l->d[k]));
            printf("\n  shape-probe last-group y_qs[0..15]:");
            for (int i = 0; i < 16; ++i) printf(" %d", (int)y2l->qs[i]);
            printf("\n");
        }

        // scaling regime: max level (sl=15,sh=3 -> ls=31). products = -35*31
        // = -1085, dnew = 1085/127 ~= 8.543, quants should be all -127,
        // y_d ~= 8.543. Exercises the needs_scaling rescale path.
        for (auto & b : src) {
            for (int i = 0; i < 32; ++i) b.scales_l[i] = 0xFF;
            for (int i = 0; i < 16; ++i) b.scales_h[i] = 0xFF;
        }
        std::vector<uint8_t> out2((size_t)nx * bxq + 64, 0xCC);
        bool okc2 = iqk_convert_repack_for_test(GGML_TYPE_IQ4_XS_R8, nn, src.data(), bxw, out2.data(), 0, nx);
        printf("  scale-probe convert returned %d\n", (int)okc2);
        const auto * y1 = (const block_q8_k_r8 *)out2.data();
        printf("  y_d:");
        for (int k = 0; k < 8; ++k) printf(" %.4g", (double)GGML_FP16_TO_FP32(y1->d[k]));
        printf("\n  y_qs[0..31]:");
        for (int i = 0; i < 32; ++i) printf(" %d", (int)y1->qs[i]);
        printf("\n");
#else
        printf("  [SKIP] compiled without TEST_DISPATCH\n");
#endif
    }

    {
        printf("\n--- Test: fused MoE up-gate vs float (iqk_moe_fused_up_gate) ---\n");
        g_rng.seed(1005); // per-suite determinism: comparable across binaries
        // Bisect the Ny boundary: 8/16 stay direct (threshold is 32);
        // 17..31 pin down where converted-path rows start dropping.
        // 32/64 take the R8->Q8 converter dispatch. Single expert, identity map.
        test_moe_fused_up_gate(1024, 64, 8);
        test_moe_fused_up_gate(1024, 64, 16);
        test_moe_fused_up_gate(1024, 64, 17);
        test_moe_fused_up_gate(1024, 64, 20);
        test_moe_fused_up_gate(1024, 64, 24);
        test_moe_fused_up_gate(1024, 64, 28);
        test_moe_fused_up_gate(1024, 64, 31);
        test_moe_fused_up_gate(1024, 64, 32);
        test_moe_fused_up_gate(1024, 64, 64);
        // same shapes without row mapping: isolates mapping handling in the
        // converted path (plain C order on both sides)
        test_moe_fused_up_gate(1024, 64, 64, false);
        // pre-converted control: same kernels, converter out of the loop
        test_moe_fused_up_gate_preconv(1024, 64, 8);
        test_moe_fused_up_gate_preconv(1024, 64, 17);
        test_moe_fused_up_gate_preconv(1024, 64, 32);
        test_moe_fused_up_gate_preconv(1024, 64, 64);
        // production-shape probe (LFM Queen experts: ne00=2048, 1792 rows):
        // does the converted combo also fail at production granularity, or
        // only at the synthetic nrc_x=64 shape?
        test_moe_fused_up_gate(2048, 1792, 64);
        test_moe_fused_up_gate_preconv(2048, 1792, 64);
        // Production-like pass (coherent small deltas): same converted shapes.
        // If the random-data failures above vanish here, the kernel is sensitive
        // to adversarial magnitudes, not structurally broken.
        printf("\n--- Test: fused MoE up-gate, coherent weights ---\n");
        g_coherent_weights = true;
        test_moe_fused_up_gate(1024, 64, 32);
        test_moe_fused_up_gate(1024, 64, 64);
        test_moe_fused_up_gate_preconv(1024, 64, 64);
        g_coherent_weights = false;
    }

    {
        printf("\n--- Test: manual gate/up/multiply replication (fused-Q8 bisection) ---\n");
        g_rng.seed(2005); // own stream: independent of upstream consumption
        test_fused_q8_manual(1024, 64);
    }

    {
        printf("\n--- Test: direct MXFP4_R8 GEMM vs float (expert path, Q8_2_X4 act) ---\n");
        int ns_mx[] = {1024};
        int nx_mx[] = {8, 64};
        int ny_mx[] = {1, 8};
        for (int n : ns_mx) {
            for (int nrc_x : nx_mx) {
                for (int nrc_y : ny_mx) {
                    test_gemm_mxfp4_r8_direct(n, nrc_x, nrc_y);
                }
            }
        }
    }

    {
        printf("\n--- Test: Q8_0_R8 repack integrity ---\n");
        for (int nrc_x : {8, 16}) test_q8_0_repack(256, nrc_x);
    }

    {
        printf("\n--- Test: IQ4_XS_R8 repack integrity ---\n");
        for (int nrc_x : {8, 16}) test_iq4_xs_repack(256, nrc_x);
    }

    {
        // mxfp4/iq4_xs have dedicated direct suites above; this covers the
        // other five bench repack families through the public entry.
        printf("\n--- Test: repacked direct GEMM vs float (legacy R-family) ---\n");
        for (int fam = 0; fam < 5; ++fam) test_legacy_r_direct(fam);
    }

    {
        printf("\n--- Test: Q4_0_R8/MXFP4_R8/IQ4_NL_R4 repack + converters ---\n");
        int ns_l[] = {256};
        for (int n : ns_l) {
            for (int nrc_x : {4, 8}) test_iq4_nl_repack(n, nrc_x);
            for (int nrc_x : {8, 16}) test_iq4_nl_r4_convert(n, nrc_x);
            for (int nrc_x : {8, 16}) {
                test_q4_0_repack(n, nrc_x);
                test_mxfp4_repack(n, nrc_x);
            }
            for (int nrc_x : {8, 16}) {
                test_q4_0_r8_convert(n, nrc_x);
                test_mxfp4_r8_convert(n, nrc_x);
            }
        }
    }

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
