//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "binbcast.cuh"

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t>
static __global__ void k_bin_bcast(const src0_t *         src0,
                                   const src1_t *         src1,
                                   dst_t *                dst,
                                   const int              ne0,
                                   const int              ne1,
                                   const int              ne2,
                                   const uint3            ne3,
                                   const uint3            ne10,
                                   const uint3            ne11,
                                   const uint3            ne12,
                                   const uint3            ne13,
                                   /*int s0, */ const int s1,
                                   const int              s2,
                                   const int              s3,
                                   /*int s00,*/ const int s01,
                                   const int              s02,
                                   const int              s03,
                                   /*int s10,*/ const int s11,
                                   const int              s12,
                                   const int              s13) {
    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3.z) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const uint32_t i10 = fastmodulo(i0, ne10);

        float result = src0_row ? (float) src0_row[i0] : 0.0f;
        dst_row[i0] = (dst_t) bin_op(result, (float) src1_row[i10]);
    }
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t>
static __global__ void k_bin_bcast_unravel(const src0_t *         src0,
                                           const src1_t *         src1,
                                           dst_t *                dst,
                                           const uint3            ne0,
                                           const uint3            ne1,
                                           const uint3            ne2,
                                           const uint32_t         ne3,
                                           const uint3            prod_012,
                                           const uint3            prod_01,
                                           const uint3            ne10,
                                           const uint3            ne11,
                                           const uint3            ne12,
                                           const uint3            ne13,
                                           /*int s0, */ const int s1,
                                           const int              s2,
                                           const int              s3,
                                           /*int s00,*/ const int s01,
                                           const int              s02,
                                           const int              s03,
                                           /*int s10,*/ const int s11,
                                           const int              s12,
                                           const int              s13) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    const int i11 = fastmodulo(i1, ne11);
    const int i12 = fastmodulo(i2, ne12);
    const int i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = fastmodulo(i0, ne10);

    float result = src0_row ? (float) src0_row[i0] : 0.0f;
    dst_row[i0] = (dst_t) bin_op(result, (float) src1_row[i10]);
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};

        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};

        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
            //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
            //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
            //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s00 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(std::min<unsigned int>(ne2*ne3, block_size / block_dims.x / block_dims.y), 64U);

            dim3 block_nums(
                (hne0 + block_dims.x - 1) / block_dims.x,
                (ne1 + block_dims.y - 1) / block_dims.y,
                (ne2*ne3 + block_dims.z - 1) / block_dims.z
            );

            if (block_nums.z > 65535) {
                // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                int         block_num  = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
                const uint3 prod_012    = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
                const uint3 prod_01     = init_fastdiv_values((uint32_t) (ne0 * ne1));
                const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
                const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
                const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

                k_bin_bcast_unravel<bin_op><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3, prod_012, prod_01,
                    init_fastdiv_values((uint32_t) ne10), init_fastdiv_values((uint32_t) ne11),
                    init_fastdiv_values((uint32_t) ne12), init_fastdiv_values((uint32_t) ne13),
                    /* s0, */ s1, s2, s3,
                    /* s00,*/ s01, s02, s03,
                    /* s10,*/ s11, s12, s13);
            } else {
                const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);
                k_bin_bcast<bin_op><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv,
                    init_fastdiv_values((uint32_t) ne10), init_fastdiv_values((uint32_t) ne11),
                    init_fastdiv_values((uint32_t) ne12), init_fastdiv_values((uint32_t) ne13),
                    /* s0, */ s1, s2, s3,
                    /* s00,*/ s01, s02, s03,
                    /* s10,*/ s11, s12, s13);
            }
        }
    }
};

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    //GGML_ASSERT(src1->type == GGML_TYPE_F32);

    if (src1->type == GGML_TYPE_F32) {
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
        } else {
            fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
                    ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
            GGML_ABORT("fatal error");
        }
    }
    else if (src1->type == GGML_TYPE_F16) {
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const float *)src0_dd, (const half *)src1_dd, (float *)dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (half *) dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (float *)dst_dd, stream);
        } else {
            fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
                    ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
            GGML_ABORT("fatal error");
        }
    }
    else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == dst->src[0]->type);
    if (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) {
        ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
        return;
    }
    auto src = dst->src[0];
    auto bs = ggml_blck_size(src->type);
    auto ts = ggml_type_size(src->type);
    if (src->nb[0] != ts || ts*(src->ne[0]/bs) % 2 != 0) {
        fprintf(stderr, "%s: unsupported case type = %s, nb[0] = %zu, type_size = %zu\n", __func__, ggml_type_name(src->type), src->nb[0], ts);
        GGML_ABORT("fatal error");
    }
    auto aux_src = *src;
    aux_src.type = GGML_TYPE_F16;
    aux_src.ne[0] = ts*(src->ne[0]/bs)/2;
    aux_src.nb[0] = 2;
    auto aux_dst = *dst;
    aux_dst.type = GGML_TYPE_F16;
    aux_dst.ne[0] = ts*(dst->ne[0]/bs)/2;
    aux_dst.nb[0] = 2;
    aux_dst.src[0] = &aux_src;
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(&aux_dst, &aux_src, &aux_dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

static __global__ void k_fast_add(int64_t ne0, int64_t nelem, const float * x, const float * y, float * z, const uint3 ne0_fd) {
    int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    z[i] = x[i] + y[fastmodulo((uint32_t)i, ne0_fd)];
}

template <typename src1_t, typename src2_t, typename dst_t>
static __global__ void k_fast_add_2(int64_t ne0, int64_t nelem, const src1_t * x, const src2_t * y, dst_t * z) {
    int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    z[i] = (dst_t)((float)x[i] + (float)y[i]);
}

template <int block_size, typename data_t>
static __global__ void k_add_same(int64_t nelem, const data_t * x, const data_t * y, data_t * z) {
    int64_t i = block_size*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    if constexpr (std::is_same_v<data_t, nv_bfloat16>) {
#if __CUDA_ARCH__ >= CC_AMPERE
        z[i] = x[i] + y[i];
#else
        z[i] = __float2bfloat16((float)x[i] + (float)y[i]);
#endif
    } else {
        z[i] = x[i] + y[i];
    }
}

template <int block_size>
static __global__ void k_add_same_q8_0(int nelem, const block_q8_0 * x, const block_q8_0 * y, block_q8_0 * z) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    int ib = i / QK8_0;
    int iq = i % QK8_0;
    float sum = (float)x[ib].d * x[ib].qs[iq] + (float)y[ib].d * y[ib].qs[iq];
    float asum = fabsf(sum);
    float max = warp_reduce_max(asum);
    float d = max / 127;
    float id = d > 0 ? 1/d : 0;
    z[ib].qs[iq] = roundf(sum * id);
    if (threadIdx.x % WARP_SIZE == 0) {
        z[ib].d = (half)d;
    }
}

template <int block_size>
static __global__ void k_add_same_q8_0(int nelem, const block_q8_0 * x, const float * y, block_q8_0 * z) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    int ib = i / QK8_0;
    int iq = i % QK8_0;
    float sum = (float)x[ib].d * x[ib].qs[iq] + y[i];
    float asum = fabsf(sum);
    float max = warp_reduce_max(asum);
    float d = max / 127;
    float id = d > 0 ? 1/d : 0;
    z[ib].qs[iq] = roundf(sum * id);
    if (threadIdx.x % WARP_SIZE == 0) {
        z[ib].d = (half)d;
    }
}

void ggml_op_add_same_type(ggml_backend_cuda_context & ctx, enum ggml_type type, size_t nelem,
        const void * x, const void * y, void * z) {
    constexpr int kBlockSize = 256;
    int nblocks = (nelem + kBlockSize - 1)/kBlockSize;
    if (type == GGML_TYPE_F32) {
        k_add_same<kBlockSize><<<nblocks, kBlockSize, 0, ctx.stream()>>>(nelem,
                (const float *)x, (const float *)y, (float *)z);
    } else if (type == GGML_TYPE_F16) {
        k_add_same<kBlockSize><<<nblocks, kBlockSize, 0, ctx.stream()>>>(nelem,
                (const half *)x, (const half *)y, (half *)z);
    } else if (type == GGML_TYPE_BF16) {
        k_add_same<kBlockSize><<<nblocks, kBlockSize, 0, ctx.stream()>>>(nelem,
                (const nv_bfloat16 *)x, (const nv_bfloat16 *)y, (nv_bfloat16 *)z);
    } else if (type == GGML_TYPE_Q8_0) {
        k_add_same_q8_0<kBlockSize><<<nblocks, kBlockSize, 0, ctx.stream()>>>(nelem,
                (const block_q8_0 *)x, (const block_q8_0 *)y, (block_q8_0 *)z);
    } else {
        GGML_ABORT("Unsupported add operation");
    }
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[0]->type == dst->src[1]->type && dst->src[0]->type == dst->type &&
        ggml_is_contiguous(dst->src[0]) && ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst) &&
        ggml_are_same_shape(dst->src[0], dst->src[1])) {
        //printf("%s(%s, %s): using fast same\n", __func__, dst->name, ggml_type_name(dst->type));
        ggml_op_add_same_type(ctx, dst->type, ggml_nelements(dst), dst->src[0]->data, dst->src[1]->data, dst->data);
        return;
    }
    if (ggml_nrows(dst->src[1]) == 1 && dst->src[0]->ne[0] == dst->src[1]->ne[0] &&
        dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(dst, dst->src[0]) && ggml_is_contiguous(dst)) {
        constexpr int kBlockSize = 256;
        auto nelem = ggml_nelements(dst);
        int nblocks = (nelem + kBlockSize - 1)/kBlockSize;
        const uint3 ne0_fd = init_fastdiv_values((uint32_t)dst->ne[0]);
        k_fast_add<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                (const float *)dst->src[0]->data, (const float *)dst->src[1]->data, (float *)dst->data, ne0_fd);
        return;
    }
    if (ggml_is_contiguous(dst->src[0]) && ggml_are_same_shape(dst->src[0], dst->src[1]) && ggml_is_contiguous(dst)) {
        constexpr int kBlockSize = 256;
        auto nelem = ggml_nelements(dst);
        int nblocks = (nelem + kBlockSize - 1)/kBlockSize;
        if (dst->type == GGML_TYPE_F16) {
            if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[1]->type == GGML_TYPE_F16) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const half *)dst->src[0]->data, (const half *)dst->src[1]->data, (half *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const half *)dst->src[0]->data, (const float *)dst->src[1]->data, (half *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const float *)dst->src[1]->data, (half *)dst->data);
            } else {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const half *)dst->src[1]->data, (half *)dst->data);
            }
        } else if (dst->type == GGML_TYPE_BF16) {
            if (dst->src[0]->type == GGML_TYPE_BF16 && dst->src[1]->type == GGML_TYPE_BF16) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const nv_bfloat16 *)dst->src[0]->data, (const nv_bfloat16 *)dst->src[1]->data, (nv_bfloat16 *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_BF16 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const nv_bfloat16 *)dst->src[0]->data, (const float *)dst->src[1]->data, (nv_bfloat16 *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const float *)dst->src[1]->data, (nv_bfloat16 *)dst->data);
            } else {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const nv_bfloat16 *)dst->src[1]->data, (nv_bfloat16 *)dst->data);
            }
        } else if (dst->type == GGML_TYPE_Q8_0) {
            GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q8_0 && dst->src[1]->type == GGML_TYPE_F32);
            k_add_same_q8_0<kBlockSize><<<nblocks, kBlockSize, 0, ctx.stream()>>>(nelem,
                        (const block_q8_0 *)dst->src[0]->data, (const float *)dst->src[1]->data, (block_q8_0 *)dst->data);
        } else {
            if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[1]->type == GGML_TYPE_F16) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const half *)dst->src[0]->data, (const half *)dst->src[1]->data, (float *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const half *)dst->src[0]->data, (const float *)dst->src[1]->data, (float *)dst->data);
            }
            else if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_F32) {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const float *)dst->src[1]->data, (float *)dst->data);
            } else {
                k_fast_add_2<<<nblocks, kBlockSize, 0, ctx.stream()>>>(dst->ne[0], nelem,
                        (const float *)dst->src[0]->data, (const half *)dst->src[1]->data, (float *)dst->data);
            }
        }
        return;
    }
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

static __global__ void scale_f32_l(const float * x, float * dst, const void * data, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float * scale = (const float *)data;
    dst[i] = scale[0] * x[i];
}

static void scale_f32_cuda_l(const float * x, float * dst, const void * data, const int k, cudaStream_t stream) {
    constexpr int CUDA_SCALE_BLOCK_SIZE = 512; //256;
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32_l<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, data, k);
}

static void ggml_cuda_op_scale_tensor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32);

    scale_f32_cuda_l(src0_d, dst_d, dst->src[1]->data, ggml_nelements(src0), stream);
}

static __global__ void k_mul_fast(int ne0, int nelem, const float * x, const float * y, float * z, const uint3 ne0_fd) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    const int i1 = fastdiv((uint32_t)i, ne0_fd);
    z[i] = x[i] * y[i1];
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_nelements(dst->src[1]) == 1 && dst->src[1]->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
        ggml_cuda_op_scale_tensor(ctx, dst);
        return;
    }
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[0] == 1 && src0->ne[1] == src1->ne[1] && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]) {
        constexpr int kBlockSize = 256;
        int nelem = ggml_nelements(src0);
        int nblock = (nelem + kBlockSize - 1)/kBlockSize;
        const uint3 ne0_fd = init_fastdiv_values((uint32_t)src0->ne[0]);
        k_mul_fast<<<nblock, kBlockSize, 0, ctx.stream()>>>(src0->ne[0], nelem, (const float *)src0->data, (const float *)src1->data, (float *)dst->data, ne0_fd);
        return;
    }
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(src0, src1, dst, src0->data, src1->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}
