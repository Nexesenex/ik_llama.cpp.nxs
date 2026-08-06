//
// unit_test_cuda.cpp - byte-for-byte CUDA block-quant verification (Q8_0, Q4_0)
//
// For each type, three producers of GGUF quantized bytes are compared on
// identical input:
//
//   1. GPU : ggml_cuda_quantize_q8_0 / ggml_cuda_quantize_q4_0
//            (ggml/src/ggml-cuda/quantize_gguf.cu)
//   2. CPU : ggml_quantize_chunk       (the fork's real llama-quantize path;
//            for Q4_0 without imatrix/symmetric this is the vanilla ref)
//   3. REF : local copy of the quantize_row_*_ref implementations
//            (ggml/src/ggml-quants.c:943 for q8_0, :673 for q4_0) with the fp16
//            step done by __float2half_rn, no ggml internals
//
// The triggering scenario is llama-quantize --cuda-quantize bf16 -> q8_0
// producing garbage perplexity (no NaN). gpu vs cpu answers "is the kernel
// byte-exact?"; cpu vs ref answers "is the fork's CPU path still the vanilla
// reference?"; gpu vs ref ties the two together with an independent baseline.
//
// Layout note: the 32-value quant blocks tile the flat row-major F32 buffer
// contiguously (n_per_row % 32 == 0), so rows need no bookkeeping and per-slice
// calls can be concatenated. test_slices reproduces do_quantize's ne[2]
// slicing exactly: the CUDA branch and the CPU branch both quantize one expert
// slice at a time into consecutive slots, and both must equal a single
// whole-tensor call.
//
// Edge cases (docs/cuda-quantize.md §6): all-zero, single outlier, ±max,
// exact .5 rounding ties, denormals, huge/tiny magnitudes, mixed signs.
// A >1<<20-block tensor exercises the chunk loop in the CUDA host wrapper.
//
// Build (must run with GGML_CUDA on):
//   cmake --build build --target unit_test_cuda -j
// Run:
//   unit_test_cuda [--seed N] [--device N] [--all-devices] [--big] [--huge] [--quick]
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static int  g_seed  = 12345;
static int  g_failures = 0;
static bool g_quick = false;
static std::mt19937 g_rng(g_seed);

// ---------------------------------------------------------------------------
// Per-type plumbing
// ---------------------------------------------------------------------------

struct quant_spec {
    const char * name;
    ggml_type    type;
    int64_t      qk;
    size_t       blk_size;
    size_t (*cuda_quantize)(const float *, void *, int64_t, int64_t);
    void (*ref)(void *, const float *, int64_t, int64_t);
};

// Local copy of quantize_row_q8_0_ref (ggml/src/ggml-quants.c:943). The fp16
// conversion uses __float2half_rn (nearest-even), what GGML_FP32_TO_FP16 maps
// to and what the CUDA kernel stores via __half_as_ushort.
static void ref_quantize_q8_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK8_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK8_0;
        block_q8_0 *  yb = (block_q8_0 *)dst + ib;

        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }

        const float d  = amax/127.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));
        for (int j = 0; j < QK8_0; ++j) {
            yb->qs[j] = (int8_t)roundf(xb[j]*id);
        }
    }
}

// Local copy of quantize_row_q4_0_ref (ggml/src/ggml-quants.c:673).
static void ref_quantize_q4_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK4_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK4_0;
        block_q4_0 *  yb = (block_q4_0 *)dst + ib;

        // signed value with max |x|; first occurrence wins |x| ties
        float amax = 0.0f;
        float max  = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = xb[j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max/-8.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));

        // MIN(15, (int8_t)(x*id + 8.5f)): truncation toward zero, then clamp.
        // |x*id| <= 8 so the truncation is always in [0, 16] -> nibble [0, 15].
        for (int j = 0; j < QK4_0/2; ++j) {
            const uint8_t xi0 = (uint8_t)std::min(15, (int)(int8_t)(xb[j]*id + 8.5f));
            const uint8_t xi1 = (uint8_t)std::min(15, (int)(int8_t)(xb[j + QK4_0/2]*id + 8.5f));
            yb->qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

static void fill_random_uniform(float * dst, int64_t n) {
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (int64_t i = 0; i < n; ++i) dst[i] = dist(g_rng);
}

// Weight-like: 90% of 32-value blocks tight around zero (N(0,0.05)), 10% wide
// (N(0,1)) - covers near-amax-scale diversity without denormal-range noise.
static void fill_random_weight_like(float * dst, int64_t n) {
    std::normal_distribution<float> tight(0.0f, 0.05f);
    std::normal_distribution<float> wide(0.0f, 1.0f);
    for (int64_t ib = 0; ib < n/QK8_0; ++ib) {
        const bool w = (g_rng() % 10) == 0;
        for (int j = 0; j < QK8_0; ++j) {
            dst[ib*QK8_0 + j] = w ? wide(g_rng) : tight(g_rng);
        }
    }
}

// Crafted edge-case blocks, one pattern per block (QK8_0 == QK4_0 == 32).
static void fill_edge_cases(float * dst, int64_t n) {
    const int64_t nb = n/QK8_0;
    const int64_t pat = 8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float * xb = dst + ib*QK8_0;
        switch (ib % pat) {
            case 0: // all zeros (d == 0 -> id == 0 path)
                for (int j = 0; j < QK8_0; ++j) xb[j] = 0.0f;
                break;
            case 1: // single outlier, rest tiny
                for (int j = 0; j < QK8_0; ++j) xb[j] = 0.001f;
                xb[g_rng() % QK8_0] = 1.0e6f;
                break;
            case 2: // values at exactly ±amax
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1000.0f : -1000.0f;
                break;
            case 3: // exact .5 rounding ties (amax == 127 -> d == 1 -> id == 1)
                for (int j = 0; j < QK8_0; ++j) {
                    const float v = (float)(j % 5) + 0.5f; // 0.5 1.5 2.5 3.5 4.5
                    xb[j] = (j & 1) ? -v : v;
                }
                xb[QK8_0-1] = 127.0f;
                break;
            case 4: // denormals / subnormals
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1.0e-38f : 1.0e-30f;
                break;
            case 5: // huge magnitudes
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1.0e30f : 1.0e38f;
                break;
            case 6: // mixed signs, small (amax from a negative value)
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? -0.4f : 0.1f;
                break;
            default: // moderate varied magnitudes
                for (int j = 0; j < QK8_0; ++j) xb[j] = (float)((g_rng() % 2001) - 1000)/8.0f;
                break;
        }
    }
}

static void dump_block(const char * who, const uint8_t * blk, size_t blk_size) {
    printf("    %s: ", who);
    for (size_t j = 0; j < blk_size; ++j) printf("%02x", blk[j]);
    printf("\n");
}

// Report the first differing quant block plus the diff count.
static int64_t compare_buffers(const char * tag, const uint8_t * a, const uint8_t * b, size_t n, size_t blk_size) {
    int64_t ndiff = 0;
    int64_t first_blk = -1;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++ndiff;
            if (first_blk < 0) first_blk = (int64_t)i / blk_size;
        }
    }
    if (first_blk >= 0) {
        const uint8_t * ra = a + first_blk*blk_size;
        const uint8_t * rb = b + first_blk*blk_size;
        printf("  [FAIL] %s: %lld/%zu bytes differ; first differing quant block %lld\n",
               tag, (long long)ndiff, n, (long long)first_blk);
        dump_block("ref", ra, blk_size);
        dump_block("gpu", rb, blk_size);
    }
    return ndiff;
}

// ---------------------------------------------------------------------------
// Byte-exact comparison of the three producers
// ---------------------------------------------------------------------------

static void test_one(const char * tag, int64_t nrows, int64_t n_per_row,
        void (*fill)(float *, int64_t), int device, const quant_spec & spec) {
    if (n_per_row % spec.qk != 0 || nrows <= 0) return;

    const int64_t nelements = nrows*n_per_row;
    std::vector<float> src(nelements);
    fill(src.data(), nelements);

    const size_t row_size = ggml_row_size(spec.type, n_per_row);
    const size_t out_size = nrows*row_size;

    std::vector<uint8_t> out_cpu(out_size);
    std::vector<uint8_t> out_gpu(out_size);
    std::vector<uint8_t> out_ref(out_size);

    // CPU: real llama-quantize path
    const size_t nb_cpu = ggml_quantize_chunk(spec.type, src.data(), out_cpu.data(),
            0, nrows, n_per_row, nullptr, nullptr);

    // REF: local vanilla copy
    spec.ref(out_ref.data(), src.data(), nrows, n_per_row);
    const size_t nb_ref = out_size;

    // GPU: real CUDA path (always runs on device 0 in this POC)
    cudaSetDevice(device);
    const size_t nb_gpu = spec.cuda_quantize(src.data(), out_gpu.data(), nrows, n_per_row);

    if (nb_cpu != nb_ref || nb_gpu != nb_ref) {
        printf("  [FAIL] %s: size mismatch cpu=%zu gpu=%zu ref=%zu\n", tag, nb_cpu, nb_gpu, nb_ref);
        ++g_failures;
        return;
    }

    const int64_t d_gpu_cpu = compare_buffers(tag, out_cpu.data(), out_gpu.data(), out_size, spec.blk_size);
    const int64_t d_cpu_ref = compare_buffers(tag, out_ref.data(), out_cpu.data(), out_size, spec.blk_size);

    if (d_gpu_cpu == 0 && d_cpu_ref == 0) {
        printf("  [OK]   %s nrows=%-6lld n_per_row=%-5lld : gpu==cpu==ref\n",
               tag, (long long)nrows, (long long)n_per_row);
    } else {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// do_quantize slice reproduction (multi-slice tensors, ne[2] > 1)
// ---------------------------------------------------------------------------

// Reproduces do_quantize (src/llama-quantize.cpp) for a tensor with ne[2]
// expert slices: the CUDA branch and the threaded CPU branch both quantize each
// slice into consecutive slots. Also checks a single whole-tensor call gives
// the same bytes (the quantizer is contiguous, so it must).
static void test_slices(int device, int64_t ne0, int64_t ne1, int64_t ne2, const quant_spec & spec) {
    char tag[96];
    snprintf(tag, sizeof(tag), "slices ne0=%lld ne1=%lld ne2=%lld",
             (long long)ne0, (long long)ne1, (long long)ne2);

    std::vector<float> src(ne0*ne1*ne2);
    fill_random_uniform(src.data(), src.size());

    const size_t row_size    = ggml_row_size(spec.type, ne0);
    const size_t matrix_size = row_size*ne1;

    std::vector<uint8_t> cpu(ne2*matrix_size);
    std::vector<uint8_t> cpu_whole(ne2*matrix_size);
    std::vector<uint8_t> gpu(ne2*matrix_size);

    // CPU per-slice (do_quantize threaded path)
    for (int64_t i02 = 0; i02 < ne2; ++i02) {
        ggml_quantize_chunk(spec.type, src.data() + i02*ne0*ne1, cpu.data() + i02*matrix_size,
                0, ne1, ne0, nullptr, nullptr);
    }

    // CPU whole-tensor single call (do_quantize single-thread path)
    ggml_quantize_chunk(spec.type, src.data(), cpu_whole.data(), 0, ne1*ne2, ne0, nullptr, nullptr);

    // GPU per-slice (do_quantize CUDA branch)
    cudaSetDevice(device);
    size_t nb_gpu_total = 0;
    for (int64_t i02 = 0; i02 < ne2; ++i02) {
        nb_gpu_total += spec.cuda_quantize(src.data() + i02*ne0*ne1, gpu.data() + i02*matrix_size, ne1, ne0);
    }

    if (nb_gpu_total != cpu.size()) {
        printf("  [FAIL] %s: gpu bytes %zu != cpu bytes %zu\n", tag, nb_gpu_total, cpu.size());
        ++g_failures;
        return;
    }

    const int64_t d_whole   = compare_buffers(tag, cpu.data(), cpu_whole.data(), cpu.size(), spec.blk_size);
    const int64_t d_gpu_cpu = compare_buffers(tag, cpu.data(), gpu.data(), cpu.size(), spec.blk_size);

    if (d_whole == 0 && d_gpu_cpu == 0) {
        printf("  [OK]   %s : gpu-slices==cpu-slices==cpu-whole\n", tag);
    } else {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Device diagnostics
// ---------------------------------------------------------------------------

static int print_devices(void) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        printf("  [FAIL] no CUDA device available\n");
        ++g_failures;
        return 0;
    }
    for (int i = 0; i < nd; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printf("  [INFO] device %d: %s (sm_%d%d, %d MiB)\n",
                   i, prop.name, prop.major, prop.minor, (int)(prop.totalGlobalMem >> 20));
        }
    }
    return nd;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int device = -1; // default: first enumerated device
    bool all_devices = false;
    bool big = false;
    bool huge = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--seed"  && i+1 < argc) g_seed = atoi(argv[++i]);
        else if (arg == "--device" && i+1 < argc) device = atoi(argv[++i]);
        else if (arg == "--all-devices") all_devices = true;
        else if (arg == "--big")   big = true;
        else if (arg == "--huge")  huge = true;
        else if (arg == "--quick") g_quick = true;
        else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }
    g_rng.seed(g_seed);
    printf("=== unit_test_cuda ===\n");
    printf("seed %d%s\n", g_seed, g_quick ? ", quick mode" : "");

    const int nd = print_devices();
    if (nd == 0) return 1;

    const int devices[2] = { device >= 0 ? device : 0, 0 };
    const int ntest_dev  = all_devices ? nd : 1;

    // quantize always runs on device 0 in this POC; report which one is real.
    printf("  [INFO] ggml_cuda_quantize_* runs on device 0 (POC); testing device %d\n", devices[0]);

    const quant_spec specs[] = {
        { "q8_0", GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), ggml_cuda_quantize_q8_0, ref_quantize_q8_0 },
        { "q4_0", GGML_TYPE_Q4_0, QK4_0, sizeof(block_q4_0), ggml_cuda_quantize_q4_0, ref_quantize_q4_0 },
    };
    const size_t nspec = sizeof(specs)/sizeof(specs[0]);

    static const int64_t ns_npr[]   = { 32, 64, 256, 512, 2048, 4096 };
    static const int64_t ns_nrows[] = { 1, 17, 128, 1000, 16384 };
    const int64_t npr_cnt = g_quick ? 3 : (int64_t)(sizeof(ns_npr)/sizeof(ns_npr[0]));
    const int64_t nrow_cnt = g_quick ? 2 : (int64_t)(sizeof(ns_nrows)/sizeof(ns_nrows[0]));

    // cap nelements so the largest default case stays ~256 MiB of f32
    const int64_t cap = g_quick ? (1<<24) : (1<<26);

    for (size_t s = 0; s < nspec; ++s) {
        const quant_spec & spec = specs[s];

        printf("\n=== type %s ===\n", spec.name);

        printf("\n--- Test: gpu vs cpu vs ref, random + weight-like + edge fills ---\n");
        for (int di = 0; di < ntest_dev; ++di) {
            const int dev = all_devices ? di : devices[0];
            for (int64_t k = 0; k < npr_cnt; ++k) {
                for (int64_t r = 0; r < nrow_cnt; ++r) {
                    const int64_t n_per_row = ns_npr[k];
                    const int64_t nrows     = std::min(ns_nrows[r], cap/n_per_row);
                    test_one("random-uniform", nrows, n_per_row, fill_random_uniform, dev, spec);
                    test_one("weight-like",   nrows, n_per_row, fill_random_weight_like, dev, spec);
                    test_one("edge-cases",    std::min<int64_t>(nrows, 1024), n_per_row, fill_edge_cases, dev, spec);
                }
            }
        }

        printf("\n--- Test: chunk-loop boundary (>1<<20 quant blocks) ---\n");
        test_one("chunk-boundary", 16385, 2048, fill_random_uniform, devices[0], spec);

        printf("\n--- Test: do_quantize ne[2] slice reproduction ---\n");
        test_slices(devices[0], 32,   17,   4, spec);
        test_slices(devices[0], 512,  128,  4, spec);
        test_slices(devices[0], 2048, 128,  4, spec);
        test_slices(devices[0], 2048, 128, 17, spec);

        if (big) {
            printf("\n--- Test: big tensor (token-embd scale) ---\n");
            test_one("big", 65536, 2048, fill_random_uniform, devices[0], spec);
        }
        if (huge) {
            printf("\n--- Test: huge tensor (Llama-3.2-1B token_embd 128256x2048) ---\n");
            test_one("huge-token_embd", 128256, 2048, fill_random_uniform, devices[0], spec);
        }
    }

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
