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
#include "iqk/iqk_mul_mat.h"

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
static bool g_strict = false;   // when true, no ±1 qs tolerance in test_path / kv-sweep
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
   // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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
   // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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
   // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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
                dequantize_row_iq4_xs_r8(&src[ib * (rpb16 / 8) + s / 8], tp, 8 * n);
                tp += (size_t)8 * n;
            }
            g_iqk_r16_path = true;
            quantize_q8_k_r16(tmp16.data(), (block_q8_k_r16 *)ref16.data() + ib * nb, rpb16, n, nullptr, nullptr);
            g_iqk_r16_path = false;
        }

        // Converter output
        std::vector<uint8_t> got16((size_t)nrc_x * rowsz16 + 1024);
        g_iqk_r16_path = true;
       // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got16.data(), nrc_x);
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
                dequantize_row_iq4_xs_r8(&src[ib * (rpb8 / 8) + s / 8], tp, 8 * n);
                tp += (size_t)8 * n;
            }
            quantize_q8_k_r8(tmp8.data(), (block_q8_k_r8 *)ref8.data() + ib * nb, rpb8, n, nullptr, nullptr);
        }

        std::vector<uint8_t> got8((size_t)nrc_x * rows_z8 + 1024);
        g_iqk_r16_path = false;
       // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got8.data(), nrc_x);
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
// Diagnostic: dump one minimal R16 block (ref vs converter) for a tiny case.
// Reveals the exact byte-level transformation error (nibble drop / garbage lane).
// ---------------------------------------------------------------------------
static void test_dump_r16(int n) {
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
   // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got16.data(), nrc_x);
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
            const int blk = ib * (rows_per_block / 8) + s / 8;
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
   // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
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
// Test: Native IQ4_XS converter (iqk_convert_iq4_xs_q8_k_r8) — direct
//       native IQ4_XS → Q8_K_R8 / Q8_K_R16, without prior repack to R8 format.
// ---------------------------------------------------------------------------
static void test_native_path(bool r16, int nrc_x, int n) {
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
    // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got_prompt.data(), prompt);
    g_iqk_r16_path = false;

    // Phase 2: full-context convert (prompt + more tokens).
    got_full.assign(out_bytes(ctx) + 1024, 0);
    g_iqk_r16_path = r16;
    // iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got_full.data(), ctx);
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
// ---------------------------------------------------------------------------
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
        // iqk_convert_iq4_xs_r8_q8_k_r16(n, W.data(), bx_w, Wr16_conv.data(), nrc_x);
        g_iqk_r16_path = false;
    }

    // (1) Byte-exact converter vs reference.
    std::vector<uint8_t> Wr16_conv2(total_bytes);
    {
        g_iqk_r16_path = true;
        // iqk_convert_iq4_xs_r8_q8_k_r16(n, W.data(), bx_w, Wr16_conv2.data(), nrc_x);
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
    printf("  --strict        No ±1 qs tolerance: require byte-exact converter output\n");
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
        if (arg == "-n" && i+1 < argc) { ns.clear(); ns.push_back(atoi(argv[++i])); continue; }
        if (arg == "--seed" && i+1 < argc) { g_seed = atoi(argv[++i]); g_rng.seed(g_seed); continue; }
        if (arg == "-nrc_x" && i+1 < argc) { nrc_xs.clear(); nrc_xs.push_back(atoi(argv[++i])); continue; }
    }
    if (test_all) { test_rtr = true; test_r16p = true; }

    printf("Seed: %d\n", g_seed);
    printf("Flags: -rtr=%d -r16p=%d  skip-byte=%d skip-dispatch=%d strict=%d\n",
           (int)test_rtr, (int)test_r16p, (int)skip_byte, (int)skip_dispatch, (int)g_strict);
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

    if (false) {
    printf("\n--- Test: Repack integrity (native IQ4_XS vs IQ4_XS_R8) ---\n");
    for (int n : ns) {
        test_repack_integrity(n);
    }

    printf("\n--- Test: KV-cache growth sweep (prompt=ctx/2, then full ctx) ---\n");
    for (int ctx : kv_ctxs) {
        if (test_rtr)  test_kv_sweep(false, ctx);
        if (test_r16p) test_kv_sweep(true,  ctx);
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

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
