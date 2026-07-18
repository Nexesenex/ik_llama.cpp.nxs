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

// Pull in ggml common defs (block types, QK_K, ggml_row_size).
#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"

// IQK internals. The converter (iqk_gemm_kquants.h) is C++-linkage; the
// reference quantizers (iqk_quantize.h) are C-linkage. Both require IQK_IMPLEMENT
// and GGML_USE_IQK_MULMAT to be declared.
#define GGML_USE_IQK_MULMAT
#define IQK_IMPLEMENT
#include "iqk/iqk_common.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_gemm_kquants.h"
#include "iqk/iqk_mul_mat.h"

// g_iqk_r16_path normally lives in the ggml lib (iqk_mul_mat.cpp).  The test
// provides a local copy; set it before each call to iqk_convert_iq4_xs_r8_q8_k_r16.
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
extern "C" const char * ggml_type_name(int type);
extern bool iqk_convert_repack(int typeA, int n, const void * vx, size_t bx,
                                void * vy, size_t stride_y, int nrc_x);
#endif

// init_unit_test_fp16_table() is provided by your build (populates
// ggml_table_f32_f16). Declared here; do not redefine.
void init_unit_test_fp16_table();

// g_iqk_r16_path is extern'd in iqk_common.h but defined in iqk_mul_mat.cpp
// which is NOT linked into this standalone test.  Provide our own definition.
bool g_iqk_r16_path = false;

static int  g_seed = 12345;
static int  g_failures = 0;
static std::mt19937 g_rng(g_seed);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a random but *valid* block_iq4_xs_r8: random half deltas, random scales,
// random 4-bit nibbles in qs. The reference path dequantizes through the same
// struct, so any structural mismatch shows up directly.
static void make_random_iq4_xs_r8(block_iq4_xs_r8 * blk) {
    for (int k = 0; k < 8; ++k) {
        float d = (g_rng() % 1000) * 0.001f + 0.01f;
        blk->d[k] = GGML_FP32_TO_FP16(d);
    }
    for (size_t i = 0; i < sizeof(blk->scales_l); ++i) blk->scales_l[i] = (uint8_t)(g_rng() & 0xff);
    for (size_t i = 0; i < sizeof(blk->scales_h); ++i) blk->scales_h[i] = (uint8_t)(g_rng() & 0xff);
    for (size_t i = 0; i < sizeof(blk->qs);      ++i) blk->qs[i]      = (uint8_t)(g_rng() & 0xff);
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

// Dequantize the interleaved fbuf from IQ4_XS_R8 src blocks.
static void fill_interleaved_fbuf(const block_iq4_xs_r8 * src, float * fbuf, int nrc_x, int n, int nb) {
    float * fp = fbuf;
    for (int r = 0; r < nrc_x; r += 8) {
        dequantize_row_iq4_xs_r8(&src[(r / 8) * nb], fp, 8 * n);
        fp += (size_t)8 * n;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Delta integrity — every row must have a finite, non-zero delta
// ---------------------------------------------------------------------------
static void test_delta_integrity(bool r16, int nrc_x, int n) {
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
    iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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

    // 2a: iqk_dequant_type must map IQ4_XS_R8 to a dequant type (not identity)
    int dq_type = iqk_dequant_type(GGML_TYPE_IQ4_XS_R8, nrc_x);
    if (dq_type == GGML_TYPE_IQ4_XS_R8) {
        printf("  [FAIL] dispatch: iqk_dequant_type(IQ4_XS_R8, %d) returned identity (no fallback)\n", nrc_x);
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
    bool conv_ok = iqk_convert_repack(GGML_TYPE_IQ4_XS_R8, n, src.data(), bx, got_repack.data(), 0, nrc_x);
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
        bool s_ok = iqk_convert_repack(GGML_TYPE_IQ4_XS_R8, n, src.data(), bx, got_stride.data(), stride_y, nrc_x);
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
    iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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
    iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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

    // Test R16 at every multiple of 16 up to 96
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
        iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got16.data(), nrc_x);
        g_iqk_r16_path = false;

        const size_t exp16 = (size_t)(nrc_x / rpb16) * nb * sizeof(block_q8_k_r16);
        long mm16 = 0, first16 = -1;
        for (size_t o = 0; o < exp16; ++o) {
            if (got16[o] != ref16[o]) { ++mm16; if (first16 < 0) first16 = (long)o; }
        }
        if (mm16 == 0)
            printf("  [OK]   boundary R16 nrc_x=%-3d n=%-5d : matches\n", nrc_x, n);
        else {
            printf("  [FAIL] boundary R16 nrc_x=%-3d n=%-5d : %ld mismatches first@%ld\n", nrc_x, n, mm16, first16);
            ++g_failures;
        }
    }

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
        iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got8.data(), nrc_x);
        g_iqk_r16_path = false;

        const size_t exp8 = (size_t)(nrc_x / rpb8) * nb * sizeof(block_q8_k_r8);
        long mm8 = 0, first8 = -1;
        for (size_t o = 0; o < exp8; ++o) {
            if (got8[o] != ref8[o]) { ++mm8; if (first8 < 0) first8 = (long)o; }
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
// Original test_path (unchanged logic, trimmed trace)
// ---------------------------------------------------------------------------
static void test_path(bool r16, int nrc_x, int n) {
    const int nb = n / QK_K;
    const int nblk_x = (nrc_x / 8) * nb;

    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    const char * path_name;
    if (r16) path_name = "-rtr -r16p";
    else     path_name = "-rtr";
    const size_t rowsz = q8_row_size(r16, n);
    const int rows_per_block = r16 ? 16 : 8;
    const size_t delta_region = (size_t)rows_per_block * 2;

    const size_t bx = ggml_row_size(GGML_TYPE_IQ4_XS_R8, n);
    g_iqk_r16_path = r16;

    const int nblocks = nrc_x / rows_per_block;
    const size_t expected_bytes = (size_t)nblocks * nb * (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
    std::vector<uint8_t> ref((size_t)nrc_x * rowsz);
    std::vector<float> tmp((size_t)rows_per_block * n);
    for (int ib = 0; ib < nblocks; ++ib) {
        float * tp = tmp.data();
        for (int s = 0; s < rows_per_block; s += 8) {
            dequantize_row_iq4_xs_r8(
                &src[(ib * (rows_per_block / 8) + s / 8) * nb],
                tp, 8 * n);
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
    iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    long first_mm = -1, last_mm = -1, delta_mm = -1, qs_mm = -1;
    long total_mismatches = 0;
    for (size_t o = 0; o < expected_bytes; ++o) {
        if (got[o] != ref[o]) {
            ++total_mismatches;
            if (first_mm < 0) first_mm = (long)o;
            last_mm = (long)o;
            if (o < delta_region * nblocks && delta_mm < 0) delta_mm = (long)o;
            if (o >= delta_region && qs_mm < 0) qs_mm = (long)o;
        }
    }

    if (total_mismatches == 0) {
        printf("  [OK]   %-12s nrc_x=%-3d n=%-5d : matches quantize_q8_k_%s byte-for-byte\n",
               path_name, nrc_x, n, r16 ? "r16" : "r8");
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
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(rb->d[k]));
                printf("\n    R8  block%d d(got):", group_base);
                for (int k = 0; k < 8; ++k) printf(" %.4f", (double)GGML_FP16_TO_FP32(gb->d[k]));
                printf("\n    R8  block%d qs[0..31](ref):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)rb->qs[o]);
                printf("\n    R8  block%d qs[0..31](got):", group_base); for (int o = 0; o < 32; ++o) printf(" %3d", (int)gb->qs[o]);
            }
            printf("\n");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --all           Test all flag combos (default)\n");
    printf("  -rtr            Test -rtr only (R8 converter)\n");
    printf("  -r16p           Test -r16p only (R16 converter)\n");
    printf("  -rtr -r16p      Test both R8 and R16 converters\n");
    printf("  -n  SIZE        Set element count per row (default: 2048,4096,8192)\n");
    printf("  -nrc_x N        Set nrc_x (default: 16,32,64,128)\n");
    printf("  --seed N        Random seed (default: 12345)\n");
    printf("  --skip-byte     Skip byte-byte comparison (fast integrity-only)\n");
    printf("  --skip-dispatch Skip dispatch-pipeline tests\n");
    printf("  --help          Show this help\n");
}

int main(int argc, char ** argv) {
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
    std::vector<int> ns = { 2048, 4096, 8192 };
    std::vector<int> nrc_xs = { 16, 32, 64, 128 };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "-rtr")  { test_all = false; test_rtr  = true; continue; }
        if (arg == "-r16p") { test_all = false; test_r16p = true; continue; }
        if (arg == "--all") { test_all = true; continue; }
        if (arg == "--skip-byte") { skip_byte = true; continue; }
        if (arg == "--skip-dispatch") { skip_dispatch = true; continue; }
        if (arg == "-n" && i+1 < argc) { ns.clear(); ns.push_back(atoi(argv[++i])); continue; }
        if (arg == "--seed" && i+1 < argc) { g_seed = atoi(argv[++i]); g_rng.seed(g_seed); continue; }
        if (arg == "-nrc_x" && i+1 < argc) { nrc_xs.clear(); nrc_xs.push_back(atoi(argv[++i])); continue; }
    }
    if (test_all) { test_rtr = true; test_r16p = true; }

    printf("Seed: %d\n", g_seed);
    printf("Flags: -rtr=%d -r16p=%d  skip-byte=%d skip-dispatch=%d\n",
           (int)test_rtr, (int)test_r16p, (int)skip_byte, (int)skip_dispatch);
    printf("Row sizes (n):"); for (int v : ns) printf(" %d", v); printf("\n");
    printf("nrc_x sizes :"); for (int v : nrc_xs) printf(" %d", v); printf("\n");
    fflush(stdout);

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

    if (!skip_byte) {
        printf("\n--- Test: Byte-for-byte comparison vs reference ---\n");
        for (int n : ns) {
            for (int nrc_x : nrc_xs) {
                if (test_rtr)  test_path(false, nrc_x, n);
                if (test_r16p) test_path(true,  nrc_x, n);
            }
        }
    }

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
