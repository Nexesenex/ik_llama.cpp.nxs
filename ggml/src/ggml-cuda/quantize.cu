//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "quantize.cuh"
#include <cstdint>

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix0 < kx ? x[ix1*kx + ix0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded, const uint64_t stride) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix0 < kx ? x[ix1*stride + ix0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
    const float * __restrict__ x, void * __restrict__ vy, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t ix0 = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*4;

    if (ix0 >= kx0_padded) {
        return;
    }

    const float4 * x4 = (const float4 *) x;

    const int64_t ix1 = kx1*blockIdx.z + blockIdx.y;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.y*gridDim.x*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (ix0 / (4*QK8_1))*kx1 + blockIdx.y;                   // block index in channel
    const int64_t iqs = ix0 % (4*QK8_1);                                            // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = ix0 < kx0 ? x4[(ix1*kx0 + ix0)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int mask = vals_per_scale/8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Exchange calculate sum across vals_per_sum/4 threads.
#pragma unroll
        for (int mask = vals_per_sum/8; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask, WARP_SIZE);
        }
    }

    const float d = amax/127.f;
    const float d_inv = d > 0 ? 1/d : 0.f;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1_id(
    const float * __restrict__ x, void * __restrict__ vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t ix0 = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*4;

    if (ix0 >= kx0_padded) {
        return;
    }

    const float4 * x4 = (const float4 *) x;

    const mmid_row_mapping * mapping = (const mmid_row_mapping *)row_mapping;
    const int64_t ii = mapping[blockIdx.y].i2;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.y*gridDim.x*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (ix0 / (4*QK8_1))*kx1 + blockIdx.y;                   // block index in channel
    const int64_t iqs = ix0 % (4*QK8_1);                                            // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = ix0 < kx0 ? x4[(ii*kx0 + ix0)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int mask = vals_per_scale/8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Exchange calculate sum across vals_per_sum/4 threads.
#pragma unroll
        for (int mask = vals_per_sum/8; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask, WARP_SIZE);
        }
    }

    const float d = amax/127.f;
    const float d_inv = d > 0 ? 1/d : 0.f;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1*channels, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx0_padded);

    GGML_UNUSED(type_x);
}

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % (4*QK8_1) == 0);

    const int64_t block_num_x = (kx0_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(block_num_x, kx1, channels);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_q8_1_id_cuda(
    const float * x, void * vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % (4*QK8_1) == 0);

    const int64_t block_num_x = (kx0_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(block_num_x, kx1, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_id<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, row_mapping, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_id<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, row_mapping, kx0, kx1, kx0_padded);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_id<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, vy, row_mapping, kx0, kx1, kx0_padded);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}


void quantize_tensor_q8_1_cuda(const struct ggml_tensor * src, void * vy, const enum ggml_type type, cudaStream_t stream) {
    GGML_ASSERT(src->ne[1] == 1 && src->ne[3] == 1);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    const int64_t src_padded_col_size = GGML_PAD(src->ne[0], MATRIX_ROW_PADDING);
    GGML_ASSERT(src_padded_col_size % QK8_1 == 0);
    if (src->ne[2] == 1 || ggml_is_contiguous(src)) {
        quantize_row_q8_1_cuda((const float *)src->data, vy, src->ne[0], 1, 1, src_padded_col_size, type, stream);
        return;
    }
    const int64_t block_num_x = (src_padded_col_size + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, src->ne[2]*src->ne[3], 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const uint64_t stride = src->nb[2]/sizeof(float);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>((const float *)src->data, vy, src->ne[0], src_padded_col_size, stride);
}
