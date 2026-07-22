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

#define GGML_COMMON_DECL_C
#include "ggml.h"
#include "ggml-common.h"
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
void   quantize_q6_1  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q6_K  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_q8_0  (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}
void   quantize_iq3_s (const float *, void *, int64_t, int64_t, const float *, const struct quantize_user_data *) {}

// iq1s / iq1m reference helpers (unused by the test)
void iq1s_process_1block(const uint8_t *, float *, const float *, int, int, const uint8_t *) {}
void iq1m_process_1block(const uint8_t *, float *, const float *, int, int, const uint8_t *) {}

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

// ---- dispatch helpers not exercised by the test ----
bool iqk_mul_mat(int, int, int, int, const void *, size_t, int, const void *, size_t, float *, size_t, int, int) {
    return false;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d\n", file, line);
    abort();
}

// ---- minimal ggml row-size helpers (only the types the test touches) ----
static size_t type_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:         return sizeof(float);
        case GGML_TYPE_Q8_0:        return 34;   // 32 qs + 2 d (legacy); unused here
        case GGML_TYPE_Q8_K:        return 2*sizeof(float) + QK_K + (QK_K/16)*sizeof(int16_t); // block_q8_K
        case GGML_TYPE_Q8_K_R8:     return sizeof(block_q8_k_r8);
        case GGML_TYPE_Q8_K_R16:    return sizeof(block_q8_k_r16);
        case GGML_TYPE_IQ4_XS:      return sizeof(block_iq4_xs);
        case GGML_TYPE_IQ4_XS_R8:   return sizeof(block_iq4_xs_r8);
        default:                    return 0;
    }
}
static int64_t blck_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:         return 1;
        case GGML_TYPE_Q8_K:        return QK_K;
        case GGML_TYPE_Q8_K_R8:     return QK_K;
        case GGML_TYPE_Q8_K_R16:    return QK_K;
        case GGML_TYPE_IQ4_XS:      return QK_K;
        case GGML_TYPE_IQ4_XS_R8:   return 8 * QK_K;
        default:                    return 0;
    }
}
size_t ggml_type_size(enum ggml_type t) { return type_size(t); }
int64_t ggml_blck_size(enum ggml_type t) { return blck_size(t); }
size_t ggml_row_size(enum ggml_type t, int64_t ne) {
    int64_t bs = blck_size(t);
    if (bs == 0) return 0;
    return (size_t)(type_size(t) * ne / bs);
}

} // extern "C"
