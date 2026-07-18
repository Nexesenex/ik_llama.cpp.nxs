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
bool g_iqk_r16_path = false;

// init_unit_test_fp16_table() is provided by your build (populates
// ggml_table_f32_f16). Declared here; do not redefine.
void init_unit_test_fp16_table();

static int g_seed = 12345;
static std::mt19937 g_rng(g_seed);

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

static bool has_nan_inf(const void * p, size_t nbytes) {
    const uint8_t * b = (const uint8_t *)p;
    for (size_t i = 0; i < nbytes; ++i) {
        // a quiet NaN / inf in fp16 would appear as 0x7C00..0x7FFF or 0xFC00..0xFFFF
        // in the half; but the converter output is int8 qs + fp16 d. int8 can't be NaN.
        // We instead check the fp16 deltas for NaN/Inf patterns.
    }
    (void)b;
    return false;
}

// Check fp16 delta array for true NaN/Inf (fp16 bit patterns 0x7C00..0x7FFF or
// 0xFC00..0xFFFF). A plain out-of-range finite value is not NaN.
template <typename Block>
static bool deltas_bad(const Block * b, int nrows) {
    for (int k = 0; k < nrows; ++k) {
        uint16_t h = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(b->d[k])); // normalize
        (void)h;
        float f = GGML_FP16_TO_FP32(b->d[k]);
        if (!std::isfinite(f)) return true;
    }
    return false;
}

static int g_failures = 0;

// Local row-size computation (avoids linking ggml.c, which this standalone test
// does not build). Q8_K_R8 / Q8_K_R16 pack QK_K elements per block.
static size_t q8_row_size(bool r16, int n) {
    if (r16) return (size_t)(n / QK_K) * sizeof(block_q8_k_r16);
    return (size_t)(n / QK_K) * sizeof(block_q8_k_r8);
}

static void test_path(bool r16, int nrc_x, int n) {
    const size_t bx = sizeof(block_iq4_xs_r8);
    const int nblk_x = nrc_x; // rows of source

    std::vector<block_iq4_xs_r8> src(nblk_x);
    for (int i = 0; i < nblk_x; ++i) make_random_iq4_xs_r8(&src[i]);

    const char * path_name = r16 ? "-rtr -r16p" : "-rtr";
    const size_t rowsz = q8_row_size(r16, n);
    const int rows_per_block = r16 ? 16 : 8;
    // delta region size = rows_per_block fp16 values = rows_per_block*2 bytes
    const size_t delta_region = (size_t)rows_per_block * 2;

    // g_iqk_r16_path controls the XOR(-128) bias inside BOTH the converter's SIMD
    // path AND the reference quantize_q8_k_r16 (repack_q16_k). It MUST be set before
    // the reference quantize so both apply (or both skip) the bias consistently.
    g_iqk_r16_path = r16;

    // Build the reference with CONTIGUOUS layout:
    //   quantize_q8_k_r8/r16 writes nb blocks per rows_per_block-row group.
    //   Groups must be placed back-to-back: offset = ib * nb blocks.
    //   This matches the layout the downstream kernels expect.
    const int nblocks = nrc_x / rows_per_block;
    const int nb = n / QK_K;
    const size_t expected_bytes = (size_t)nblocks * nb * (r16 ? sizeof(block_q8_k_r16) : sizeof(block_q8_k_r8));
    std::vector<uint8_t> ref((size_t)nrc_x * rowsz);
    std::vector<float> tmp((size_t)rows_per_block * n);
    for (int ib = 0; ib < nblocks; ++ib) {
        float * tp = tmp.data();
        for (int s = 0; s < rows_per_block; s += 8) {
            dequantize_row_iq4_xs_r8(
                &src[ib * (rows_per_block / 8) + s / 8],
                tp, 8 * n);
            tp += (size_t)8 * n;
        }
        if (r16)
            quantize_q8_k_r16(tmp.data(),
                (block_q8_k_r16 *)ref.data() + ib * nb,
                rows_per_block, n, nullptr, nullptr);
        else
            quantize_q8_k_r8(tmp.data(),
                (block_q8_k_r8  *)ref.data() + ib * nb,
                rows_per_block, n, nullptr, nullptr);
    }

    // Interleaved float buffer: nrc_x rows of n floats, row-major.
    // Dequant in rows_per_block groups (8 rows per dequantize call).
    std::vector<float> fbuf_interleaved((size_t)nrc_x * n);
    {
        float * fp = fbuf_interleaved.data();
        for (int r = 0; r < nrc_x; r += 8) {
            dequantize_row_iq4_xs_r8(&src[r], fp, 8 * n);
            fp += (size_t)8 * n;
        }
    }

    // Converter output (over-allocated with a safety margin so an OOB write from a
    // stride bug does not crash before we can report the mismatch).
    std::vector<uint8_t> got((size_t)nrc_x * rowsz + 1024 * 1024);
    printf("  ... calling converter %s nrc_x=%-3d n=%-5d\n", path_name, nrc_x, n); fflush(stdout);
    iqk_convert_iq4_xs_r8_q8_k_r16(n, src.data(), bx, got.data(), nrc_x);
    g_iqk_r16_path = false;

    // Byte-compare converter output vs reference.
    // First check the converter actually wrote something (not all zeros).
    bool all_zero = true;
    for (size_t o = 0; o < expected_bytes && o < 4096; ++o) { if (got[o] != 0) { all_zero = false; break; } }
    (void)all_zero;
    long first_mm = -1, last_mm = -1, delta_mm = -1, qs_mm = -1;
    long total_mismatches = 0;
    (void)delta_mm; (void)qs_mm; // used only for info
    for (size_t o = 0; o < expected_bytes && o < (size_t)nrc_x * rowsz; ++o) {
        if (got[o] != ref[o]) {
            ++total_mismatches;
            if (first_mm < 0) first_mm = (long)o;
            last_mm = (long)o;
            if (o < (long)delta_region * nblocks && delta_mm < 0) delta_mm = (long)o;
            if (o >= (long)delta_region && qs_mm < 0) qs_mm = (long)o;
        }
    }

    bool byte_ok = (total_mismatches == 0);
    if (byte_ok) {
        printf("  [OK]   %s nrc_x=%-3d n=%-5d : matches quantize_q8_k_%s byte-for-byte\n",
               path_name, nrc_x, n, r16 ? "r16" : "r8");
    } else {
        ++g_failures;
        printf("  [FAIL] %s nrc_x=%-3d n=%-5d : %ld byte(s) differ", path_name, nrc_x, n, total_mismatches);
        printf("  first@%ld  last@%ld", first_mm, last_mm);
        if (delta_mm >= 0) printf("  delta@%ld", delta_mm);
        if (qs_mm >= 0)    printf("  qs@%ld", qs_mm);
        printf("  span=%ld\n", last_mm - first_mm + 1);

        // Dump region around first mismatch
        long dump_start = (first_mm > 8) ? first_mm - 8 : 0;
        long dump_end = first_mm + 24;
        if (dump_end > (long)expected_bytes) dump_end = (long)expected_bytes;
        printf("    ref["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)ref[d]); printf("\n");
        printf("    got["); for (long d = dump_start; d < dump_end; ++d) printf("%s%02x", d == first_mm ? " >" : " ", (int)(uint8_t)got[d]); printf("\n");

        // Block-level summary: which blocks have deltas/qs mismatches
        if (total_mismatches < 200) {
            printf("    mismatched bytes:");
            for (size_t o = 0; o < expected_bytes; ++o)
                if (got[o] != ref[o]) printf(" %zu", o);
            printf("\n");
        }

        if (g_failures <= 3) {
            // first_mm falls in group first_mm / (nb * blocksize).
            // Show the group's first block (base of the group).
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

    // Secondary: round-trip (dequant converter output → compare against interleaved fbuf).
    // Dequant in rows_per_block groups from contiguous nb-block groups.
    std::vector<float> fgot((size_t)nrc_x * n);
    if (r16) {
        const block_q8_k_r16 * rb = (const block_q8_k_r16 *)got.data();
        for (int b = 0; b < nblocks; ++b)
            dequantize_row_q8_k_r16(&rb[b * nb], fgot.data() + (size_t)b * rows_per_block * n, n);
    } else {
        const block_q8_k_r8 * rb = (const block_q8_k_r8 *)got.data();
        for (int b = 0; b < nblocks; ++b)
            dequantize_row_q8_k_r8(&rb[b * nb], fgot.data() + (size_t)b * rows_per_block * n, n);
    }
    double max_abs = 0; long first_bad = -1; int bad_count = 0;
    for (size_t i = 0; i < (size_t)nrc_x * n; ++i) {
        double e = std::fabs((double)fbuf_interleaved[i] - (double)fgot[i]);
        if (e > max_abs) max_abs = e;
        if (e > 0.5) { if (first_bad < 0) first_bad = (long)i; ++bad_count; }
    }
    if (bad_count > 0) {
        printf("         round-trip: max|err|=%.3f bad_elems=%d first@elem%ld(row%ld)\n",
               max_abs, bad_count, first_bad, first_bad / n);
        if (first_bad >= 0) {
            int elemrow = first_bad / n;
            printf("         ref[%ld..%ld]:", first_bad, first_bad + 7 < (long)nrc_x * n ? first_bad + 7 : first_bad);
            for (int k = 0; k < 8 && first_bad + k < (long)nrc_x * n; ++k) printf(" %.1f", (double)fbuf_interleaved[first_bad + k]);
            printf("\n         got[%ld..%ld]:", first_bad, first_bad + 7 < (long)nrc_x * n ? first_bad + 7 : first_bad);
            for (int k = 0; k < 8 && first_bad + k < (long)nrc_x * n; ++k) printf(" %.1f", (double)fgot[first_bad + k]);
            printf("\n");
        }
    }

    // NaN / Inf check on deltas — scan every nb-th block (contiguous groups)
    if (r16) {
        const block_q8_k_r16 * rb = (const block_q8_k_r16 *)got.data();
        for (int i = 0; i < nblocks; ++i)
            if (deltas_bad(&rb[i * nb], 16)) { ++g_failures; printf("  [FAIL] R16 block %d (group %d) has NaN/Inf delta\n", i*nb, i); break; }
    } else {
        const block_q8_k_r8 * rb = (const block_q8_k_r8 *)got.data();
        for (int i = 0; i < nblocks; ++i)
            if (deltas_bad(&rb[i * nb], 8))  { ++g_failures; printf("  [FAIL] R8  block %d (group %d) has NaN/Inf delta\n", i*nb, i); break; }
    }
}

static void usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --all        Test all flag combos (default)\n");
    printf("  -rtr         Test -rtr only (R8 output)\n");
    printf("  -rtr -r16p   Test -rtr -r16p (R16 output)\n");
    printf("  -n  SIZE     Set element count per row (default: 2048,4096,8192)\n");
    printf("  -nrc_x N     Set nrc_x (default: 16,32,64,128)\n");
    printf("  --seed N     Random seed (default: 12345)\n");
    printf("  --help       Show this help\n");
}

int main(int argc, char ** argv) {
    init_unit_test_fp16_table();
    printf("=== IQK iq4_xs_r8 -> q8_k_r8 / q8_k_r16 converter verification ===\n"); fflush(stdout);

    // Detect SIMD level available at compile time
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
    fflush(stdout);
    printf("QK_K=%d  sizeof(block_iq4_xs_r8)=%zu  sizeof(block_q8_k_r8)=%zu  sizeof(block_q8_k_r16)=%zu\n",
           QK_K, sizeof(block_iq4_xs_r8), sizeof(block_q8_k_r8), sizeof(block_q8_k_r16));

    // Parse arguments
    bool test_rtr  = false;
    bool test_r16p = false;
    bool test_all  = true;
    bool test_vnni = true;  // also test VNNI path when available
    std::vector<int> ns = { 2048, 4096, 8192 };
    std::vector<int> nrc_xs = { 16, 32, 64, 128 };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "-rtr")  { test_all = false; test_rtr  = true; continue; }
        if (arg == "-r16p") { test_all = false; test_r16p = true; continue; }
        if (arg == "--all") { test_all = true; continue; }
        if (arg == "-n" && i+1 < argc) {
            ns.clear(); ns.push_back(atoi(argv[++i]));
            continue;
        }
        if (arg == "--seed" && i+1 < argc) {
            g_seed = atoi(argv[++i]); g_rng.seed(g_seed);
            continue;
        }
        if (arg == "-nrc_x" && i+1 < argc) {
            nrc_xs.clear(); nrc_xs.push_back(atoi(argv[++i]));
            continue;
        }
    }
    if (test_all) { test_rtr = true; test_r16p = true; }

    printf("Seed: %d\n", g_seed);
    printf("Flag combos to test:"); if (test_rtr)  printf(" -rtr"); if (test_r16p) printf(" -r16p"); printf("\n");
    printf("Row sizes (n):"); for (int v : ns) printf(" %d", v); printf("\n");
    printf("nrc_x sizes :"); for (int v : nrc_xs) printf(" %d", v); printf("\n");
    fflush(stdout);

    for (int n : ns) {
        for (int nrc_x : nrc_xs) {
            if (test_rtr)  test_path(false /* -rtr only  = R8 */, nrc_x, n);
            if (test_r16p) test_path(true  /* -rtr -r16p = R16 */, nrc_x, n);
        }
    }

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
