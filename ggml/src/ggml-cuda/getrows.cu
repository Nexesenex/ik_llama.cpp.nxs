//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "getrows.cuh"
#include "dequantize.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 = (blockIdx.y*blockDim.x + threadIdx.x)*2;
    const int i10 =  blockIdx.x*blockDim.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    if (i01 >= 0 && i01 < ne01) {
        dequantize_kernel(src0_row, ib, iqs, v);
    } else {
        v.x = v.y = 0;
    }

    dst_row[iybs + iqs + 0]        = v.x;
    dst_row[iybs + iqs + y_offset] = v.y;
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 =  blockIdx.y*blockDim.x + threadIdx.x;
    const int i10 =  blockIdx.x*blockDim.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = i01 >= 0 && i01 < ne01 ? dst_t(src0_row[i00]) : dst_t(0);
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    GGML_ASSERT(ne11*ne12 < 65536);
    const dim3 block_nums(ne10, block_num_x, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template<typename src0_t>
static void get_rows_cuda_float(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                const src0_t * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    GGML_ASSERT(ne11*ne12 < 65536);
    const dim3 block_nums(ne10, block_num_x, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template <typename data_t>
static __global__ void k_get_rows_dim0(int n,
        size_t nb01, size_t nb02, size_t nb03, size_t nb11, size_t nb12, size_t nb13, size_t nb1, size_t nb2, size_t nb3,
        const data_t * __restrict__ src, const int * __restrict__ idx, data_t * __restrict__ dst) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    src += nb01*i1 + nb02*i2 + nb03*i3;
    idx += nb11*i1 + nb12*i2 + nb13*i3;
    dst +=  nb1*i1 +  nb2*i2 +  nb3*i3;

    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[idx[j]];
}

template <typename data_t>
static __global__ void k_get_rows_dim1(int n,
        size_t nb01, size_t nb02, size_t nb03, size_t nb11, size_t nb12, size_t nb1, size_t nb2, size_t nb3,
        const data_t * __restrict__ src, const int * __restrict__ idx, data_t * __restrict__ dst) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    idx += nb11*i2 + nb12*i3;
    src += nb01*idx[i1] + nb02*i2 + nb03*i3;
    dst +=  nb1*i1 +  nb2*i2 +  nb3*i3;

    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

// Helper for k-quant scale extraction (matches convert.cu implementation)
static inline __device__ void get_scale_min_k4_gr(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Gather + dequantize Q2_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q2_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];

    const int64_t tid = threadIdx.x;
    const int64_t n = tid/32;
    const int64_t l = tid - 32*n;
    const int64_t is = 8*n + l/16;

    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block*QK_K + 128*n;

    if (row_idx < 0 || row_idx >= ne01) {
        y[l+ 0] = 0;
        y[l+32] = 0;
        y[l+64] = 0;
        y[l+96] = 0;
        return;
    }

    const block_q2_K * x = (const block_q2_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    const uint8_t q = x->qs[32*n + l];

    float dall = __low2half(x->dm);
    float dmin = __high2half(x->dm);
    y[l+ 0] = dall * (x->scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x->scales[is+0] >> 4);
    y[l+32] = dall * (x->scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x->scales[is+2] >> 4);
    y[l+64] = dall * (x->scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x->scales[is+4] >> 4);
    y[l+96] = dall * (x->scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x->scales[is+6] >> 4);
}

// Gather + dequantize Q3_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q3_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block*QK_K + 128*n + 32*j;

    if (row_idx < 0 || row_idx >= ne01) {
        for (int l = l0; l < l0+4; ++l) {
            y[l] = 0;
        }
        return;
    }

    const block_q3_K * x = (const block_q3_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                          (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    float d_all = x->d;
    float dl = d_all * (us - 32);

    const uint8_t * q = x->qs + 32*n;
    const uint8_t * hm = x->hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

// Gather + dequantize Q4_K rows. 32 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q4_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 8;
    const int64_t ir  = tid % 8;
    const int64_t is  = 2 * il;

    y += 64 * il + 4 * ir;

    if (row_idx < 0 || row_idx >= ne01) {
        for (int l = 0; l < 4; ++l) {
            y[l +  0] = 0;
            y[l + 32] = 0;
        }
        return;
    }

    const block_q4_K * x = (const block_q4_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * q = x->qs + 32 * il + 4 * ir;

    uint8_t sc, m;
    get_scale_min_k4_gr(is + 0, x->scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4_gr(is + 1, x->scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < 4; ++l) {
        y[l +  0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >>  4) - m2;
    }
}

// Gather + dequantize Q5_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q5_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 16;
    const int64_t ir  = tid % 16;
    const int64_t is  = 2 * il;

    y += 64 * il + 2 * ir;

    if (row_idx < 0 || row_idx >= ne01) {
        y[ 0] = 0;
        y[ 1] = 0;
        y[32] = 0;
        y[33] = 0;
        return;
    }

    const block_q5_K * x = (const block_q5_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * ql = x->qs + 32 * il + 2 * ir;
    const uint8_t * qh = x->qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4_gr(is + 0, x->scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4_gr(is + 1, x->scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

// Gather + dequantize Q6_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q6_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid / 32;
    const int64_t il  = tid % 32;
    const int64_t is  = 8 * ip + il / 16;

    y += 128 * ip + il;

    if (row_idx < 0 || row_idx >= ne01) {
        y[ 0] = 0;
        y[32] = 0;
        y[64] = 0;
        y[96] = 0;
        return;
    }

    const block_q6_K * x = (const block_q6_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    const float d = x->d;
    const uint8_t * ql = x->ql + 64 * ip + il;
    const uint8_t   qh = x->qh[32 * ip + il];
    const int8_t  * sc = x->scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

static void get_rows_q2_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q2_K<float><<<block_nums, 64, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q3_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q3_K<float><<<block_nums, 64, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

// Gather + dequantize Q8_K rows. 256 threads, blockIdx.x = block-in-row, blockIdx.y = token, blockIdx.z = batch.
template<typename dst_t>
static __global__ void k_get_rows_q8_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t ne01, int64_t ne12,
        int64_t s1, int64_t s2, int64_t s3,
        int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t s10, int64_t s11, int64_t s12) {
    const int64_t i_block = blockIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t i11 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;

    const int32_t row_idx = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * y = dst + i10*s1 + i11*s2 + i12*s3 + i_block*QK_K;

    const int64_t tid = threadIdx.x;

    if (row_idx < 0 || row_idx >= ne01) {
        y[tid] = 0;
        return;
    }

    const block_q8_K * x = (const block_q8_K *)((const char *)src0 + row_idx*nb01 + i11*nb02 + i12*nb03) + i_block;

    y[tid] = x->d * x->qs[tid];
}

static void get_rows_q4_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q4_K<float><<<block_nums, 32, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q5_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q5_K<float><<<block_nums, 64, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q6_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q6_K<float><<<block_nums, 64, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q8_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT(ne11*ne12 < 65536);
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const int64_t s2_dst = nb2 / ggml_element_size(dst);
    const int64_t s3_dst = nb3 / ggml_element_size(dst);
    const int64_t s10 = nb10 / ggml_element_size(src1);
    const int64_t s11 = nb11 / ggml_element_size(src1);
    const int64_t s12 = nb12 / ggml_element_size(src1);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q8_K<float><<<block_nums, QK_K, 0, stream>>>(
        src0_d, src1_d, dst_d, ne01, ne12,
        s1_dst, s2_dst, s3_dst, nb01, nb02, nb03, s10, s11, s12);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    if (dst->type != GGML_TYPE_F32 || dst->op_params[0] == 1) {
        constexpr int k_block_size = 512;
        GGML_ASSERT(src0->type == dst->type);
        if (dst->op_params[0] == 1) {
            //printf("%s(%s, %s) - dim0\n", __func__, src0->name, ggml_type_name(src0->type));
            GGML_ASSERT(src0->ne[1] == src1->ne[1] && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
            GGML_ASSERT(src0->ne[0] >= src1->ne[0]);
            GGML_ASSERT(dst->ne[0] == src1->ne[0]);
            auto type_size = ggml_type_size(dst->type);
            GGML_ASSERT(type_size == 2 || type_size == 4);
            dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
            if (type_size == 2) {
                k_get_rows_dim0<<<grid, k_block_size, 0, ctx.stream()>>>(dst->ne[0],
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size, src1->nb[3]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size, dst->nb[3]/type_size,
                        (const uint16_t *)src0->data, (const int *)src1->data, (uint16_t *)dst->data);
            } else {
                k_get_rows_dim0<<<grid, k_block_size, 0, ctx.stream()>>>(dst->ne[0],
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size, src1->nb[3]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size,  dst->nb[3]/type_size,
                        (const uint32_t *)src0->data, (const int *)src1->data, (uint32_t *)dst->data);
            }
        } else {
            //printf("%s(%s, %s) - dim1\n", __func__, src0->name, ggml_type_name(src0->type));
            GGML_ASSERT(src0->ne[2] == src1->ne[1]);
            GGML_ASSERT(src0->ne[3] == src1->ne[2]);
            GGML_ASSERT(src0->ne[3] == 1);
            GGML_ASSERT(dst->ne[0] == src0->ne[0] && dst->ne[1] == src1->ne[0] && dst->ne[2] == src1->ne[1] && dst->ne[2]);

            auto row_size = ggml_row_size(dst->type, dst->ne[0]);
            GGML_ASSERT(row_size % 2 == 0);
            auto type_size = row_size % 4 == 0 ? 4 : 2;
            dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
            if (type_size == 2) {
                k_get_rows_dim1<<<grid, k_block_size, 0, ctx.stream()>>>(row_size/type_size,
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size, dst->nb[3]/type_size,
                        (const uint16_t *)src0->data, (const int *)src1->data, (uint16_t *)dst->data);
            } else {
                k_get_rows_dim1<<<grid, k_block_size, 0, ctx.stream()>>>(row_size/type_size,
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size,  dst->nb[3]/type_size,
                        (const uint32_t *)src0->data, (const int *)src1->data, (uint32_t *)dst->data);
            }
        }
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32));

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float(src0, src1, dst, (const half *)src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            get_rows_cuda_float(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_q2_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_q3_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_q4_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_q5_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_q6_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_K:
            get_rows_q8_K_cuda(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            break;
    }
}
