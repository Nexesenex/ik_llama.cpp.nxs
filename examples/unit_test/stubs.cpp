// Stub / minimal definitions for symbols referenced by the inlined iqk
// quantizer sources but not needed by this unit test. The quantize/dequantize
// entry points are never called by the test, so empty bodies suffice. The
// ggml row-size helpers are implemented minimally for the types the test
// actually exercises.

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>

#define GGML_COMMON_DECL_C
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "iqk/iqk_quantize.h"

extern "C" {

// ---- quantize / dequantize entry points referenced but unused by the test ----
void   quantize_iq2_s (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq2_xs (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq2_xxs(const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq3_xxs(const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq4_nl (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq4_xs (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q2_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q3_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q4_0  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q4_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q5_0  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q5_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q6_0  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q6_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q8_0  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq3_s (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}

// iq1s / iq1m reference helpers (unused by the test)
void iq1s_process_1block(const uint8_t *, float *, const float *, int, int, const uint8_t *) {}
void iq1m_process_1block(const uint8_t *, float *, const float *, int, int, const uint8_t *) {}

// Prism ternary reference quantizers (PQ2_0 / PTQ1_0).
// Mirrors ggml-quants.c (origin b25a013d029 Prism ternary #2550).
// The standalone unit_test inlines ggml/src/iqk/iqk_quantize.cpp, whose
// quantize_row_pq2_0_r8 / quantize_row_ptq1_0_r8 call these refs. They are
// not exercised by the test, but must link. Keep in sync with ggml-quants.c.
void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_PQ2_0;

    assert(k % qk == 0);

    const int nb = (int)(k / qk);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float a = fabsf(x[i*qk + j]);
            if (a > amax) amax = a;
        }
        const float d  = amax;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        memset(y[i].qs, 0, sizeof(y[i].qs));

        for (int j = 0; j < qk; ++j) {
            const float w = x[i*qk + j];
            int q = (int)roundf(w * id) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            y[i].qs[j / 4] |= ((uint8_t)q << ((j % 4) * 2));
        }
    }
}

void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k) {
    static const size_t ptq1_0_stages[2] = {16, 8};

    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; j++) {
            amax = MAX(amax, fabsf(x[j]));
        }

        const float d  = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        size_t j = 0;
        for (size_t s = 0; s < 2; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        int xi = (int)lroundf(x[m + n*c] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += (uint8_t)xi;
                    }
                    // ceiling division (243 == pow(3, 5))
                    q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
                    y[i].qs[j + m] = q;
                }
                x += 5*c;
            }
        }
        // 4 elements per byte
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = (int)lroundf(x[h + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += (uint8_t)xi;
            }
            q *= 3; // shift the first value to the most significant trit
            q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
            y[i].qh[h] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

bool ggml_is_contiguous(const struct ggml_tensor * tensor) { (void)tensor; return true; }
int64_t ggml_nrows(const struct ggml_tensor * tensor) { (void)tensor; return 0; }

float ggml_bf16_to_fp32(ggml_bf16_t b) {
    uint16_t u16;
    memcpy(&u16, &b, sizeof(u16));
    uint32_t u = (uint32_t)u16 << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ---- dispatch helpers ----
// NOTE: the real iqk_mul_mat is linked from iqk_mul_mat.cpp (see CMakeLists),
// so there is intentionally NO stub for it here (a stub would be a duplicate
// symbol). Only genuinely unlinked ggml entry points are stubbed below.

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d\n", file, line);
    abort();
}

// ---- quant trait table: SINGLE source of truth for the standalone harness ----
// Mirrors the ggml.c type_traits columns (type_size / blck_size /
// row_meta_size) plus the ggml.c type_name strings. Previously these lived in
// four parallel switches that drifted apart (the Q8_K_R8 span-vs-share bug
// hid there); now adding a quant = one table row, and type_size / blck_size /
// row_meta / type_name below cannot diverge.
// Startup-initialized like the fudge table: MSVC rejects C99
// array-designator initializers. For the 19 legacy rows the effective
// ggml_row_size is unchanged (their span-type/N*blck convention equals the
// ggml.c per-row convention byte-for-byte, cf. the static_asserts in
// ggml-common.h); added rows follow ggml.c verbatim.
struct stub_qtraits { size_t tsize; int64_t blck; size_t meta; const char * name; };
static stub_qtraits stub_traits[GGML_TYPE_COUNT];
struct QTraitsInit {
    QTraitsInit() {
        stub_traits[GGML_TYPE_I8]            = { sizeof(int8_t),  1,           0, "i8" };
        stub_traits[GGML_TYPE_I16]           = { sizeof(int16_t), 1,           0, "i16" };
        stub_traits[GGML_TYPE_I32]           = { sizeof(int32_t), 1,           0, "i32" };
        stub_traits[GGML_TYPE_I64]           = { sizeof(int64_t), 1,           0, "i64" };
        stub_traits[GGML_TYPE_F64]           = { sizeof(double),  1,           0, "f64" };
        stub_traits[GGML_TYPE_F32]           = { sizeof(float),   1,           0, "f32" };
        stub_traits[GGML_TYPE_F16]           = { sizeof(ggml_fp16_t), 1,        0, "f16" };
        stub_traits[GGML_TYPE_BF16]          = { sizeof(ggml_bf16_t), 1,        0, "bf16" };
        stub_traits[GGML_TYPE_BF16_R16]      = { sizeof(ggml_bf16_t), 1,        0, "bf16_r16" };
        stub_traits[GGML_TYPE_I2_S]          = { 1,                   1,        0, "i2_s" };
        stub_traits[GGML_TYPE_Q4_0]          = { sizeof(block_q4_0),  QK4_0,    0, "q4_0" };
        stub_traits[GGML_TYPE_Q4_1]          = { sizeof(block_q4_1),  QK4_1,    0, "q4_1" };
        stub_traits[GGML_TYPE_Q5_0]          = { sizeof(block_q5_0),  QK5_0,    0, "q5_0" };
        stub_traits[GGML_TYPE_Q5_1]          = { sizeof(block_q5_1),  QK5_1,    0, "q5_1" };
        stub_traits[GGML_TYPE_Q6_0]          = { sizeof(block_q6_0),  QK6_0,    0, "q6_0" };
        stub_traits[GGML_TYPE_Q8_0]          = { sizeof(block_q8_0),  QK8_0,    0, "q8_0" };
        stub_traits[GGML_TYPE_Q8_1]          = { sizeof(block_q8_1),  QK8_1,    0, "q8_1" };
        stub_traits[GGML_TYPE_Q8_0_X4]       = { sizeof(block_q8_0),  QK8_0,    0, "q8_0_x4" };
        stub_traits[GGML_TYPE_Q8_1_X4]       = { sizeof(block_q8_1),  QK8_1,    0, "q8_1_x4" };
        stub_traits[GGML_TYPE_Q8_2_X4]       = { sizeof(block_q8_2),  QK8_2,    0, "q8_2_x4" };
        stub_traits[GGML_TYPE_Q4_0_4_4]      = { sizeof(block_q4_0),  QK4_0,    0, "q4_0_4x4" };
        stub_traits[GGML_TYPE_Q4_0_4_8]      = { sizeof(block_q4_0),  QK4_0,    0, "q4_0_4x8" };
        stub_traits[GGML_TYPE_Q4_0_8_8]      = { sizeof(block_q4_0),  QK4_0,    0, "q4_0_8x8" };
        stub_traits[GGML_TYPE_Q4_0_R8]       = { sizeof(block_iq4_nl_r8), 8 * QK4_0, 0, "q4_0_r8" };
        stub_traits[GGML_TYPE_Q5_0_R4]       = { sizeof(block_q5_0_r4), 4 * QK5_0, 0, "q5_0_r4" };
        stub_traits[GGML_TYPE_Q6_0_R4]       = { sizeof(block_q6_0_r4), 4 * QK6_0, 0, "q6_0_r4" };
        stub_traits[GGML_TYPE_Q8_0_R8]       = { sizeof(block_q8_0_r8), 8 * QK8_0, 0, "q8_0_r8" };
        stub_traits[GGML_TYPE_IQ4_NL]        = { sizeof(block_iq4_nl), QK4_NL,  0, "iq4_nl" };
        stub_traits[GGML_TYPE_IQ4_NL_R4]     = { sizeof(block_iq4_nl_r4), 4 * QK4_NL, 0, "iq4_nl_r4" };
        stub_traits[GGML_TYPE_MXFP4]         = { sizeof(block_mxfp4), QK_MXFP4, 0, "mxfp4" };
        stub_traits[GGML_TYPE_MXFP4_R8]      = { sizeof(block_mxfp4_r8), 8 * QK_MXFP4, 0, "mxfp4_r8" };
        stub_traits[GGML_TYPE_IQ4_XS]        = { sizeof(block_iq4_xs), QK_K,    0, "iq4_xs" };
        stub_traits[GGML_TYPE_IQ4_XS_R8]     = { sizeof(block_iq4_xs_r8), 8 * QK_K, 0, "iq4_xs_r8" };
        stub_traits[GGML_TYPE_Q8_K]          = { 2*sizeof(float) + QK_K + (QK_K/16)*sizeof(int16_t), QK_K, 0, "q8_K" };
        stub_traits[GGML_TYPE_Q8_K_R8]       = { sizeof(block_q8_k_r8)/8, QK_K, 0, "q8_k_r8" };
        stub_traits[GGML_TYPE_Q8_K_R16]      = { sizeof(block_q8_k_r16)/16, QK_K, 0, "q8_k_r16" };
        stub_traits[GGML_TYPE_Q2_K]          = { sizeof(block_q2_K),  QK_K,     0, "q2_K" };
        stub_traits[GGML_TYPE_Q2_K_R4]       = { sizeof(block_q2_K),  QK_K,     0, "q2_k_r4" };
        stub_traits[GGML_TYPE_Q3_K]          = { sizeof(block_q3_K),  QK_K,     0, "q3_K" };
        stub_traits[GGML_TYPE_Q3_K_R4]       = { sizeof(block_q3_K),  QK_K,     0, "q3_k_r4" };
        stub_traits[GGML_TYPE_Q4_K]          = { sizeof(block_q4_K),  QK_K,     0, "q4_K" };
        stub_traits[GGML_TYPE_Q4_K_R4]       = { sizeof(block_q4_K),  QK_K,     0, "q4_k_r4" };
        stub_traits[GGML_TYPE_Q5_K]          = { sizeof(block_q5_K),  QK_K,     0, "q5_K" };
        stub_traits[GGML_TYPE_Q5_K_R4]       = { sizeof(block_q5_K),  QK_K,     0, "q5_k_r4" };
        stub_traits[GGML_TYPE_Q6_K]          = { sizeof(block_q6_K),  QK_K,     0, "q6_K" };
        stub_traits[GGML_TYPE_Q6_K_R4]       = { sizeof(block_q6_K),  QK_K,     0, "q6_k_r4" };
        stub_traits[GGML_TYPE_IQ2_XXS]       = { sizeof(block_iq2_xxs), QK_K,   0, "iq2_xxs" };
        stub_traits[GGML_TYPE_IQ2_XXS_R4]    = { sizeof(block_iq2_xxs), QK_K,   0, "iq2_xxs_r4" };
        stub_traits[GGML_TYPE_IQ2_XS]        = { sizeof(block_iq2_xs), QK_K,    0, "iq2_xs" };
        stub_traits[GGML_TYPE_IQ2_XS_R4]     = { sizeof(block_iq2_xs), QK_K,    0, "iq2_xs_r4" };
        stub_traits[GGML_TYPE_IQ3_XXS]       = { sizeof(block_iq3_xxs), QK_K,   0, "iq3_xxs" };
        stub_traits[GGML_TYPE_IQ3_XXS_R4]    = { sizeof(block_iq3_xxs), QK_K,   0, "iq3_xxs_r4" };
        stub_traits[GGML_TYPE_IQ3_S]         = { sizeof(block_iq3_s), QK_K,     0, "iq3_s" };
        stub_traits[GGML_TYPE_IQ3_S_R4]      = { sizeof(block_iq3_s), QK_K,     0, "iq3_s_r4" };
        stub_traits[GGML_TYPE_IQ2_S]         = { sizeof(block_iq2_s), QK_K,     0, "iq2_s" };
        stub_traits[GGML_TYPE_IQ2_S_R4]      = { sizeof(block_iq2_s), QK_K,     0, "iq2_s_r4" };
        stub_traits[GGML_TYPE_IQ1_S]         = { sizeof(block_iq1_s), QK_K,     0, "iq1_s" };
        stub_traits[GGML_TYPE_IQ1_S_R4]      = { sizeof(block_iq1_s_r4)/4, 32,  2, "iq1_s_r4" };
        stub_traits[GGML_TYPE_IQ1_M]         = { sizeof(block_iq1_m), QK_K,     0, "iq1_m" };
        stub_traits[GGML_TYPE_IQ1_M_R4]      = { sizeof(block_iq1_m_r4)/4, 32,  2, "iq1_m_r4" };
        stub_traits[GGML_TYPE_IQ1_BN]        = { sizeof(block_iq1_bn), QK_IQ1BN, 2, "iq1_bn" };
        stub_traits[GGML_TYPE_IQ2_BN]        = { sizeof(block_iq2_bn), QK_IQ1BN, 4, "iq2_bn" };
        stub_traits[GGML_TYPE_IQ2_BN_R4]     = { sizeof(block_iq2_bn), QK_IQ1BN, 4, "iq2_bn_r4" };
        stub_traits[GGML_TYPE_IQ4_KS]        = { sizeof(block_iq4_ks), QK_K,    4, "iq4_ks" };
        stub_traits[GGML_TYPE_IQ4_KS_R4]     = { sizeof(block_iq4_ks), QK_K,    4, "iq4_ks_r4" };
        stub_traits[GGML_TYPE_IQ4_KS_R16]    = { sizeof(block_iq4_ks_r16)/16, QK8_0, 4, "iq4_ks_r16" };
        stub_traits[GGML_TYPE_IQ3_KS_R16]    = { 13,                  QK8_0,   4, "iq3ks_r16" };
        stub_traits[GGML_TYPE_IQ5_KS_R4]     = { sizeof(block_iq5_ks), QK_K,   4, "iq5_ks_r4" };
        stub_traits[GGML_TYPE_IQ4_KSS]       = { sizeof(block_iq4_kss), QK_K,  4, "iq4_kss" };
        stub_traits[GGML_TYPE_IQ5_KS]        = { sizeof(block_iq5_ks), QK_K,   4, "iq5_ks" };
        stub_traits[GGML_TYPE_Q8_K64]        = { sizeof(block_q8_K64), 64,     0, "q8_K64" };
        stub_traits[GGML_TYPE_Q8_K128]       = { sizeof(block_q8_K128), 128,   0, "q8_K128" };
        stub_traits[GGML_TYPE_Q8_KV]         = { 32,                  32,      8, "q8_KV" };
        stub_traits[GGML_TYPE_Q8_KV_R8]      = { 32,                  32,      4, "q8_KV_r8" };
        stub_traits[GGML_TYPE_Q8_K16]        = { 64,                  64,      20, "q8_K16" };
        stub_traits[GGML_TYPE_Q8_K32]        = { sizeof(block_q8_K),  QK_K,    0, "q8_K32" };
        stub_traits[GGML_TYPE_Q8_KR8]        = { sizeof(block_q8_K),  QK_K,    0, "q8_KR8" };
        stub_traits[GGML_TYPE_Q1_0_G128]     = { sizeof(block_q1_0_g128), QK1_0_G128, 0, "q1_0_g128" };
        stub_traits[GGML_TYPE_Q1_0_G128_R8]  = { sizeof(block_q1_0_g128_r8)/QK1_0_G128_R8_ROWS, QK1_0_G128, 0, "q1_0_g128_r8" };
        stub_traits[GGML_TYPE_PQ2_0]         = { sizeof(block_pq2_0), QK_PQ2_0, 0, "pq2_0" };
        stub_traits[GGML_TYPE_PQ2_0_R8]      = { sizeof(block_pq2_0_r8)/QK_PQ2_0_R8_ROWS, QK_PQ2_0, 0, "pq2_0_r8" };
        stub_traits[GGML_TYPE_PTQ1_0]        = { sizeof(block_ptq1_0), QK_PTQ1_0, 0, "ptq1_0" };
        stub_traits[GGML_TYPE_PTQ1_0_R8]     = { sizeof(block_ptq1_0_r8)/QK_PTQ1_0_R8_ROWS, QK_PTQ1_0, 0, "ptq1_0_r8" };
        stub_traits[GGML_TYPE_IQ2_K]         = { sizeof(block_iq2_k), QK_K,     0, "iq2_k" };
        stub_traits[GGML_TYPE_IQ2_K_R4]      = { sizeof(block_iq2_k), QK_K,     0, "iq2_k_r4" };
        stub_traits[GGML_TYPE_IQ2_KS]        = { sizeof(block_iq2_ks), QK_K,    2, "iq2_ks" };
        stub_traits[GGML_TYPE_IQ1_KT]        = { sizeof(block_iq1_kt), QK_K,    4, "iq1_kt" };
        stub_traits[GGML_TYPE_IQ2_KT]        = { sizeof(block_iq2_kt), QK_K,    4, "iq2_kt" };
        stub_traits[GGML_TYPE_IQ3_KT]        = { sizeof(block_iq3_kt), QK_K,    4, "iq3_kt" };
        stub_traits[GGML_TYPE_IQ4_KT]        = { sizeof(block_iq4_kt), QK_K,    4, "iq4_kt" };
        stub_traits[GGML_TYPE_IQ3_K]         = { sizeof(block_iq3_k), QK_K,     0, "iq3_k" };
        stub_traits[GGML_TYPE_IQ3_KS]        = { sizeof(block_iq3_ks), QK_K,    2, "iq3_ks" };
        stub_traits[GGML_TYPE_IQ2_KL]        = { sizeof(block_iq2_kl), QK_K,    2, "iq2_kl" };
        stub_traits[GGML_TYPE_IQ4_K]         = { sizeof(block_iq4_k), QK_K,     0, "iq4_k" };
        stub_traits[GGML_TYPE_IQ4_K_R4]      = { sizeof(block_iq4_k), QK_K,     0, "iq4_k_r4" };
        stub_traits[GGML_TYPE_IQ3_K_R4]      = { sizeof(block_iq3_k), QK_K,     0, "iq3_k_r4" };
        stub_traits[GGML_TYPE_IQ5_K]         = { sizeof(block_iq5_k), QK_K,     0, "iq5_k" };
        stub_traits[GGML_TYPE_IQ5_K_R4]      = { sizeof(block_iq5_k), QK_K,     0, "iq5_k_r4" };
        stub_traits[GGML_TYPE_IQ6_K]         = { sizeof(block_iq6_k), QK_K,     0, "iq6_k" };
    }
};
static QTraitsInit g_qtraits_init;
static const stub_qtraits * qtraits_of(enum ggml_type t) {
    if ((int)t < 0 || t >= GGML_TYPE_COUNT) return nullptr;
    return &stub_traits[(int)t];
}

// ---- ggml_type_name: table lookup (unlisted/heredoc types print "?") ----
const char * ggml_type_name(enum ggml_type type) {
    const stub_qtraits * q = qtraits_of(type);
    return q && q->name ? q->name : "?";
}

// ---- ggml_internal_get_type_traits: link shim, never executed by the test ----
// Only reachable from FA/indexer paths (unused by every suite here). Aborts
// loudly if that ever changes, instead of returning garbage.
ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type) {
    (void)type;
    fprintf(stderr, "ggml_internal_get_type_traits: unavailable in standalone unit_test build\n");
    abort();
    ggml_type_traits_t dummy;
    memset(&dummy, 0, sizeof(dummy));
    return dummy;
}

size_t ggml_type_size(enum ggml_type t) { const stub_qtraits * q = qtraits_of(t); return q ? q->tsize : 0; }
int64_t ggml_blck_size(enum ggml_type t) { const stub_qtraits * q = qtraits_of(t); return q ? q->blck : 0; }
static size_t row_meta(enum ggml_type t) { const stub_qtraits * q = qtraits_of(t); return q ? q->meta : 0; }

static bool is_kt_tail(enum ggml_type t) {
    return t == GGML_TYPE_IQ2_KT || t == GGML_TYPE_IQ3_KT || t == GGML_TYPE_IQ4_KT;
}
size_t ggml_row_size(enum ggml_type t, int64_t ne) {
    // Mirrors ggml.c ggml_row_size(), including the IQ3_KS_R16 band and
    // KT-tail formulas. Previously only the plain meta-free types were
    // covered, so any new bench over KS/KT/KV quants would size buffers wrong.
    if (t == GGML_TYPE_IQ3_KS_R16) {
        assert(ne % 32 == 0);
        return GGML_PAD(64 + (ne/32)*sizeof(block_iq3_ks_r16), 16)/16;
    }
    if (is_kt_tail(t)) {
        assert(ne % 32 == 0);
        const int nt = (int)((ne % QK_K)/32);
        const size_t tail = t == GGML_TYPE_IQ2_KT ? nt > 0 ? 4 + 8*(size_t)nt : 0
                          : t == GGML_TYPE_IQ3_KT ? (size_t)((nt + 1)/2 + 12*nt) : 16*(size_t)nt;
        return GGML_PAD(row_meta(t) + ggml_type_size(t)*(size_t)(ne/QK_K) + tail, 4);
    }
    int64_t bs = ggml_blck_size(t);
    if (bs == 0) return 0;
    assert(ne % bs == 0);
    return row_meta(t) + (size_t)(ggml_type_size(t) * ne / bs);
}
// Mirrors ggml.c: number of blocks per row (32 for KT-tail types).
int64_t ggml_row_blck_size(enum ggml_type t) {
    if (t == GGML_TYPE_IQ2_KT || t == GGML_TYPE_IQ3_KT || t == GGML_TYPE_IQ4_KT) return 32;
    return ggml_blck_size(t);
}

// ---- quantization fudge factors (referenced by iqk_quantize.cpp) ----
// Mirrors the table in ggml.c. Note: types not explicitly listed default to
// 0.0f (C static-zero initialization), exactly as in the real ggml build.
static float fudge_factors[GGML_TYPE_COUNT];
namespace {
// MSVC cl.exe rejects C99 array-designator initializers ([i] = v), so fill
// this mirror of the ggml.c fudge table at startup instead. Values must stay
// in sync with ggml.c (see the notes there); types not listed here default
// to 0.0f, exactly as in the real ggml build.
struct FudgeTableInit {
    FudgeTableInit() {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) fudge_factors[i] = 0.0f;
        fudge_factors[GGML_TYPE_I8]         = 1.0f;
        fudge_factors[GGML_TYPE_I16]        = 1.0f;
        fudge_factors[GGML_TYPE_I32]        = 1.0f;
        fudge_factors[GGML_TYPE_I64]        = 1.0f;
        fudge_factors[GGML_TYPE_F64]        = 1.0f;
        fudge_factors[GGML_TYPE_F32]        = 1.0f;
        fudge_factors[GGML_TYPE_F16]        = 1.0f;
        fudge_factors[GGML_TYPE_Q4_0]       = 1.0f;
        fudge_factors[GGML_TYPE_Q4_1]       = 1.0f;
        fudge_factors[4]                    = 1.0f;
        fudge_factors[5]                    = 1.0f;
        fudge_factors[GGML_TYPE_Q5_0]       = 1.0f;
        fudge_factors[GGML_TYPE_Q5_1]       = 1.0f;
        fudge_factors[GGML_TYPE_Q6_0]       = 1.0f;
        fudge_factors[GGML_TYPE_Q8_0]       = 1.0f;
        fudge_factors[GGML_TYPE_Q8_1]       = 1.0f;
        fudge_factors[GGML_TYPE_Q8_0_X4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_1_X4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_2_X4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q2_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q2_K_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q3_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q3_K_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q4_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q4_K_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q5_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q5_K_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q6_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q6_K_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K_R8]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K_R16]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_XXS]    = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_XXS_R4] = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_XS]     = 1.05f;
        fudge_factors[GGML_TYPE_IQ2_XS_R4]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ3_XXS]    = 1.0125f;
        fudge_factors[GGML_TYPE_IQ3_XXS_R4] = 1.0f;
        fudge_factors[GGML_TYPE_IQ3_S]      = 1.033f;
        fudge_factors[GGML_TYPE_IQ3_S_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_S]      = 0.9875f;
        fudge_factors[GGML_TYPE_IQ2_S_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ1_S]      = 1.125f;
        fudge_factors[GGML_TYPE_IQ1_S_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ1_M]      = 1.085f;
        fudge_factors[GGML_TYPE_IQ1_M_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ1_BN]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_BN]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_BN_R4]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_NL]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_XS]     = 1.0f;
        fudge_factors[GGML_TYPE_MXFP4]      = 1.0f;
        fudge_factors[GGML_TYPE_MXFP4_R8]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_KS]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_KS_R4]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_KS_R16] = 1.0f;
        fudge_factors[GGML_TYPE_IQ5_KS_R4]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_KSS]    = 1.01f;
        fudge_factors[GGML_TYPE_IQ5_KS]     = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K]       = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K64]     = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K128]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_KV]      = 1.0f;
        fudge_factors[GGML_TYPE_Q8_KV_R8]   = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K16]     = 1.0f;
        fudge_factors[GGML_TYPE_Q8_K32]     = 1.0f;
        fudge_factors[GGML_TYPE_Q8_KR8]     = 1.0f;
        fudge_factors[GGML_TYPE_BF16]       = 1.0f;
        fudge_factors[GGML_TYPE_BF16_R16]   = 1.0f;
        fudge_factors[GGML_TYPE_Q4_0_4_4]   = 1.0f;
        fudge_factors[GGML_TYPE_Q4_0_4_8]   = 1.0f;
        fudge_factors[GGML_TYPE_Q4_0_8_8]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_K]      = 1.03f;
        fudge_factors[GGML_TYPE_IQ2_K_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_KS]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ1_KT]     = 1.07f;
        fudge_factors[GGML_TYPE_IQ2_KT]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ3_KT]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_KT]     = 1.0f;
        fudge_factors[GGML_TYPE_Q1_0_G128]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ3_K]      = 1.01f;
        fudge_factors[GGML_TYPE_IQ3_KS]     = 1.0f;
        fudge_factors[GGML_TYPE_IQ2_KL]     = 1.025f;
        fudge_factors[GGML_TYPE_IQ4_K]      = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_K_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ3_K_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ5_K]      = 1.0f;
        fudge_factors[GGML_TYPE_IQ5_K_R4]   = 1.0f;
        fudge_factors[GGML_TYPE_IQ6_K]      = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_NL_R4]  = 1.0f;
        fudge_factors[GGML_TYPE_IQ4_XS_R8]  = 1.0f;
        fudge_factors[GGML_TYPE_Q4_0_R8]    = 1.0f;
        fudge_factors[GGML_TYPE_Q8_0_R8]    = 1.0f;
        fudge_factors[GGML_TYPE_Q5_0_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_Q6_0_R4]    = 1.0f;
        fudge_factors[GGML_TYPE_I2_S]       = 1.0f;
    }
};
static FudgeTableInit g_fudge_table_init;
}

float ggml_get_quantize_fudge_factor(enum ggml_type type) { return fudge_factors[type]; }
void  ggml_set_quantize_fudge_factor(enum ggml_type type, float fudge) { fudge_factors[type] = fudge; }

} // extern "C"
