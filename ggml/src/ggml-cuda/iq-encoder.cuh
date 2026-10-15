#pragma once

#include "qleg-encoder.cuh"

size_t ggml_cuda_quantize_iq(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix);
