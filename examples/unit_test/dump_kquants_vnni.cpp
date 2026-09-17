// Differential battery for the VNNI guard flips (8d3ab8d5f40 follow-up).
//
// Builds twice: WITH VNNI (normal) and WITHOUT (fallback). Both binaries run
// the same fixed battery (fixed seed) of direct GEMM kernels over IDENTICAL
// random bytes and dump raw outputs. The two dumps must be bit-identical:
// the flipped branches are pure integer dot-product rewrites (dpbusd vs
// maddubs+madd) with identical float tails, so any differing byte is a real
// behavioral change, not quantization noise (no quantizers involved).
//
// Usage: dump_kquants_vnni <out.bin>   (same for the novnni twin)

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#define GGML_USE_IQK_MULMAT
#define IQK_IMPLEMENT
#include "iqk/iqk_common.h"
#include "iqk/iqk_gemm_kquants.h"

// init_unit_test_fp16_table() is provided by fp16_table.cpp (populates
// ggml_table_f32_f16). Declared here; do not redefine.
void init_unit_test_fp16_table();

namespace {

std::mt19937 g_rng(12345);

void fill_random(void * ptr, size_t nbytes) {
    auto * p = (uint8_t *)ptr;
    for (size_t i = 0; i < nbytes; ++i) p[i] = (uint8_t)(g_rng() & 0xff);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("Usage: %s <out.bin>\n", argv[0]);
        return 2;
    }
    init_unit_test_fp16_table();

#if defined(HAVE_VNNI256)
    printf("variant: VNNI\n");
#else
    printf("variant: FALLBACK (no VNNI)\n");
#endif
#if defined(HAVE_FANCY_SIMD)
    printf("FANCY on (unexpected for this battery)\n");
#endif

    const int n = 512;
    const int nb = n / QK_K; // 2
    const int nrc_x = 32;
    const int nys[] = {1, 2, 8};

    // (typeA needs nb blocks/row of its plain block; R4 groups share rows)
    struct fam { const char * tag; ggml_type ta; ggml_type tb; size_t plain_sz; size_t b_row; };
    const fam fams[] = {
        {"q4_k",    GGML_TYPE_Q4_K,   GGML_TYPE_Q8_2_X4, sizeof(block_q4_K),   2 * (size_t)nb * sizeof(block_q8_2_x4)},
        {"q5_k",    GGML_TYPE_Q5_K,   GGML_TYPE_Q8_2_X4, sizeof(block_q5_K),   2 * (size_t)nb * sizeof(block_q8_2_x4)},
        {"q6_k",    GGML_TYPE_Q6_K,   GGML_TYPE_Q8_2_X4, sizeof(block_q6_K),   2 * (size_t)nb * sizeof(block_q8_2_x4)},
        {"q2_k_r4", GGML_TYPE_Q2_K_R4, GGML_TYPE_Q8_K,   sizeof(block_q2_K),   (size_t)nb * sizeof(block_q8_K)},
        {"q6_k_r4", GGML_TYPE_Q6_K_R4, GGML_TYPE_Q8_K,   sizeof(block_q6_K),   (size_t)nb * sizeof(block_q8_K)},
    };

    FILE * f = fopen(argv[1], "wb");
    if (!f) { printf("cannot open %s\n", argv[1]); return 2; }

    int ncases = 0;
    for (size_t fi = 0; fi < sizeof(fams) / sizeof(fams[0]); ++fi) {
        const size_t bx = (size_t)nb * fams[fi].plain_sz;
        std::vector<uint8_t> A((size_t)nrc_x * bx);
        fill_random(A.data(), A.size());
        std::vector<uint8_t> B(8 * fams[fi].b_row);
        fill_random(B.data(), B.size());

        for (int nrc_y : nys) {
            std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
            mul_mat_t func16 = nullptr;
            if (!iqk_set_kernels_kquants(n, (int)fams[fi].ta, (int)fams[fi].tb, funcs, func16)) {
                printf("  [SKIP] %-8s nrc_y=%d : no kernels\n", fams[fi].tag, nrc_y);
                continue;
            }
            if (!funcs[nrc_y - 1]) {
                printf("  [SKIP] %-8s nrc_y=%d : null func\n", fams[fi].tag, nrc_y);
                continue;
            }
            std::vector<float> C((size_t)nrc_y * nrc_x, -1.f);
            DataInfo info;
            info.s   = C.data();
            info.cy  = (const char *)B.data();
            info.bs  = nrc_x;
            info.by  = fams[fi].b_row;
            info.cur_y = 0;
            info.ne11  = nrc_y;
            info.row_mapping = nullptr;
            funcs[nrc_y - 1](n, A.data(), bx, info, nrc_x);

            if (fwrite(C.data(), sizeof(float), C.size(), f) != C.size()) {
                printf("write failed\n");
                fclose(f);
                return 2;
            }
            printf("  dumped %-8s nrc_y=%-2d floats=%zu\n", fams[fi].tag, nrc_y, C.size());
            ++ncases;
        }
    }

    fclose(f);
    printf("wrote %d cases\n", ncases);
    return 0;
}
