// Provides storage for the FP16<->FP32 conversion table used by the inlined
// iqk quantizer/dequantizer code. The table is normally defined in ggml.c and
// initialized by ggml_init(); we define it here and initialize it explicitly so
// the unit test does not need to link the ggml DLL's (unexported) copy.
#include "ggml-impl.h"

float ggml_table_f32_f16[1 << 16];

void init_unit_test_fp16_table() {
    for (int i = 0; i < (1 << 16); ++i) {
        union { uint16_t u16; ggml_fp16_t fp16; } u = { (uint16_t)i };
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }
}
