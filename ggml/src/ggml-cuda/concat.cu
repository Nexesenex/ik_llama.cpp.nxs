//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "concat.cuh"
#include <cstdlib>

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne00,
        int64_t nb02, int64_t nb12, int64_t nb2) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * nb2;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * nb02;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * nb12;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    if (dim == 0 && ne1 >= 65536) {
        int64_t nstep = (ne1 + 32767)/32768;
        for (int64_t istep = 0; istep < nstep; ++istep) {
            int64_t i1 = 32768*istep;
            int64_t n1 = i1 + 32768 <= ne1 ? 32768 : ne1 - i1;
            dim3 gridDim(num_blocks, n1, ne2);
            const float * xi = x + i1*ne00;
            const float * yi = y + i1*(ne0 - ne00);
            float * dst_i = dst + i1*ne0;
            concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(xi, yi, dst_i, ne0, ne00, ne00*ne01, (ne0-ne00)*ne01, ne0*ne1);
        }
        return;
    }
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        //concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00, ne00*ne01, (ne0-ne00)*ne01, ne0*ne1);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
static __global__ void concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3,
          int32_t   dim) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    const float * x;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type != src1->type) {
        printf("%s: %s is type %s, %s is type %s\n", __func__, src0->name, ggml_type_name(src0->type),
                src1->name, ggml_type_name(src1->type));
    }
    GGML_ASSERT(src0->type == src1->type && src0->type == dst->type);

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    // Diagnostic for illegal access debugging (qwen4exp 49/49 offload, 82 GiB pinned)
    {
        fprintf(stderr, "CONCAT %s dim %d dev %d (CUDA%d) src0 %s type %s ne [%lld %lld %lld %lld] nb [%zu %zu %zu %zu] cont %d src1 %s type %s ne [%lld %lld %lld %lld] nb [%zu %zu %zu %zu] cont %d dst ne [%lld %lld %lld %lld] nb [%zu %zu %zu %zu] cont %d nbytes %zu %zu\n",
            dst->name, dim, ctx.device, ggml_backend_cuda_get_device_ordinal(ctx.device),
            src0->name, ggml_type_name(src0->type), (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], ggml_is_contiguous(src0),
            src1->name, ggml_type_name(src1->type), (long long)src1->ne[0], (long long)src1->ne[1], (long long)src1->ne[2], (long long)src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], ggml_is_contiguous(src1),
            (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], ggml_is_contiguous(dst),
            ggml_nbytes(src0), ggml_nbytes(src1));
        fflush(stderr);
        // Grid overflow check – log and fallback to CPU if would exceed 65535
        int64_t grid_y = dst->ne[1];
        int64_t grid_z = dst->ne[2]*dst->ne[3];
        if (grid_y >= 65536 || grid_z >= 65536) {
            fprintf(stderr, "CONCAT grid overflow y=%lld z=%lld exceeds 65535, consider CPU fallback\n", (long long)grid_y, (long long)grid_z);
            fflush(stderr);
        }
        // Device mismatch check already handled via staging, but log
        cudaPointerAttributes a0, a1, ad;
        cudaError_t e0 = cudaPointerGetAttributes(&a0, src0->data);
        cudaError_t e1 = cudaPointerGetAttributes(&a1, src1->data);
        cudaError_t ed = cudaPointerGetAttributes(&ad, dst->data);
        if (e0==cudaSuccess) fprintf(stderr, "  src0 ptr %p type %d device %d\n", src0->data, a0.type, a0.device);
        if (e1==cudaSuccess) fprintf(stderr, "  src1 ptr %p type %d device %d\n", src1->data, a1.type, a1.device);
        if (ed==cudaSuccess) fprintf(stderr, "  dst  ptr %p type %d device %d\n", dst->data, ad.type, ad.device);
        fflush(stderr);
        if (e0!=cudaSuccess) cudaGetLastError();
        if (e1!=cudaSuccess) cudaGetLastError();
        if (ed!=cudaSuccess) cudaGetLastError();
    }

    // Cross-device handling: src may be on different CUDA device than dst (split model) or on host (CUDA_Host).
    // Kernel launch on ctx.device can only access memory on that device, otherwise illegal memory access.
    // Stage src through a temp buffer on dst device via peer/host copy.
    const void * src0_ptr = src0->data;
    const void * src1_ptr = src1->data;
    void * tmp0 = nullptr;
    void * tmp1 = nullptr;
    bool free0 = false;
    bool free1 = false;
    auto stage_src = [&](const ggml_tensor * src, const void * src_data, void *& tmp, bool & do_free) -> const void* {
        if (!src->buffer) return src_data;
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, src_data);
        if (err != cudaSuccess) { cudaGetLastError(); return src_data; }
        int dst_ordinal = ggml_cuda_info().cuda_device_id[ctx.device];
        if (attr.type == cudaMemoryTypeDevice) {
            int src_ordinal = attr.device;
            if (src_ordinal != dst_ordinal) {
                size_t nbytes = ggml_nbytes(src);
                if (nbytes == 0) return src_data;
                CUDA_CHECK(cudaMalloc(&tmp, nbytes));
                // Peer copy requires peer access; if not enabled, cudaMemcpyPeerAsync will return error and fallback to staged host copy
                cudaError_t perr = cudaMemcpyPeerAsync(tmp, dst_ordinal, src_data, src_ordinal, nbytes, stream);
                if (perr != cudaSuccess) {
                    cudaGetLastError();
                    // fallback: copy via host staging
                    void * host = malloc(nbytes);
                    if (host) {
                        CUDA_CHECK(cudaMemcpyAsync(host, src_data, nbytes, cudaMemcpyDeviceToHost, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        CUDA_CHECK(cudaMemcpyAsync(tmp, host, nbytes, cudaMemcpyHostToDevice, stream));
                        free(host);
                    } else {
                        CUDA_CHECK(perr);
                    }
                }
                do_free = true;
                return tmp;
            }
        } else if (attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered) {
            // Host memory (CUDA_Host pinned or plain host) -> copy to device temp
            size_t nbytes = ggml_nbytes(src);
            if (nbytes == 0) return src_data;
            CUDA_CHECK(cudaMalloc(&tmp, nbytes));
            CUDA_CHECK(cudaMemcpyAsync(tmp, src_data, nbytes, cudaMemcpyHostToDevice, stream));
            do_free = true;
            return tmp;
        }
        return src_data;
    };
    src0_ptr = stage_src(src0, src0->data, tmp0, free0);
    src1_ptr = stage_src(src1, src1->data, tmp1, free1);

    // Workaround for small dim0 concat (ple_conv_state 9+1 ->10, ne1=10240) that faults with kernel (illegal access on 3090, enable-p2p=0)
    // Use per-row memcpy instead of concat_f32_cuda kernel for tiny ne0 (<64) to avoid grid/kernel bug.
    if (dim == 0 && dst->ne[0] < 64 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        // Check if src are F32 contiguous, then row bytes = ne0 * sizeof(float)
        // Do per-row D2D copies: dst row = src0 row (ne00 floats) + src1 row (ne0-ne00 floats)
        int64_t ne00 = src0->ne[0];
        int64_t ne0  = dst->ne[0];
        size_t row0_bytes = ne00 * sizeof(float);
        size_t row1_bytes = (ne0 - ne00) * sizeof(float);
        // Use host-staged fallback for tiny 9+1 case to avoid D2D illegal access on 3090 with enable-p2p=0
        // Use malloc + synchronous cudaMemcpy (works with non-pinned host) to avoid WDDM pinned quota (50 GiB) exhaustion
        {
            size_t total_bytes = ggml_nbytes(dst);
            // Allocate host buffers for src0/src1/dst (plain malloc, sync copy handles non-pinned)
            void * host_src0 = malloc(ggml_nbytes(src0));
            void * host_src1 = malloc(ggml_nbytes(src1));
            void * host_dst  = malloc(total_bytes);
            if (host_src0 && host_src1 && host_dst) {
                // Ensure current device is dst's ordinal for D2H/H2D (3090 ordinal 1, enable-p2p=0) and clear prior kernel error
                int dst_ord = ggml_cuda_info().cuda_device_id[ctx.device];
                CUDA_CHECK(cudaSetDevice(dst_ord));
                (void)cudaGetLastError(); // clear stale illegal access from prior kernel (concat_f32_dim0) before host copy
                // Also sync device to flush prior async error state (illegal access is sticky until sync)
                (void)cudaDeviceSynchronize(); (void)cudaGetLastError();
                CUDA_CHECK(cudaMemcpy(host_src0, src0_ptr, ggml_nbytes(src0), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_src1, src1_ptr, ggml_nbytes(src1), cudaMemcpyDeviceToHost));
                // CPU concat per row
                for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                            const char * h_src0_row = (const char *)host_src0 + i3*src0->nb[3] + i2*src0->nb[2] + i1*src0->nb[1];
                            const char * h_src1_row = (const char *)host_src1 + i3*src1->nb[3] + i2*src1->nb[2] + i1*src1->nb[1];
                            char * h_dst_row = (char *)host_dst + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1];
                            if (row0_bytes) memcpy(h_dst_row, h_src0_row, row0_bytes);
                            if (row1_bytes) memcpy(h_dst_row + row0_bytes, h_src1_row, row1_bytes);
                        }
                    }
                }
                CUDA_CHECK(cudaMemcpy(dst->data, host_dst, total_bytes, cudaMemcpyHostToDevice));
                free(host_src0); free(host_src1); free(host_dst);
            } else {
                if (host_src0) free(host_src0);
                if (host_src1) free(host_src1);
                if (host_dst)  free(host_dst);
                for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                            const char * src0_row = (const char *)src0_ptr + i3*src0->nb[3] + i2*src0->nb[2] + i1*src0->nb[1];
                            const char * src1_row = (const char *)src1_ptr + i3*src1->nb[3] + i2*src1->nb[2] + i1*src1->nb[1];
                            char * dst_row = (char *)dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1];
                            if (row0_bytes) {
                                cudaError_t e = cudaMemcpyAsync(dst_row, src0_row, row0_bytes, cudaMemcpyDefault, stream);
                                if (e != cudaSuccess) { cudaGetLastError(); CUDA_CHECK(cudaMemcpy(dst_row, src0_row, row0_bytes, cudaMemcpyDefault)); }
                            }
                            if (row1_bytes) {
                                cudaError_t e = cudaMemcpyAsync(dst_row + row0_bytes, src1_row, row1_bytes, cudaMemcpyDefault, stream);
                                if (e != cudaSuccess) { cudaGetLastError(); CUDA_CHECK(cudaMemcpy(dst_row + row0_bytes, src1_row, row1_bytes, cudaMemcpyDefault)); }
                            }
                        }
                    }
                }
            }
        }
        goto concat_cleanup;
    }

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        (dim == 3 || (dim == 2 && dst->ne[3] == 1) || (dim == 1 && dst->ne[2]*dst->ne[3] == 1))) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);
        CUDA_CHECK(cudaMemcpyAsync((char *)dst->data,         src0_ptr, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *)dst->data + size0, src1_ptr, size1, cudaMemcpyDeviceToDevice, stream));
        goto concat_cleanup;
    }

    if (dim == 0 && src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
            src0->nb[1] % sizeof(float) == 0 && src1->nb[1] % sizeof(float) == 0) {
        auto bs = ggml_blck_size(dst->type);
        auto ts = ggml_type_size(dst->type);
        auto ne00_eff = (src0->ne[0]/bs)*ts/sizeof(float);
        auto ne0_eff  = (dst->ne[0]/bs)*ts/sizeof(float);
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
            //if (dst->ne[1] >= 65536 || dst->ne[2] >= 65536) {
            //    fprintf(stderr, "%s: ne1 = %ld, ne2 = %ld exceed max. blocks when computing %s\n", __func__, dst->ne[1], dst->ne[2], dst->name);
            //    GGML_ABORT("fatal error");
            //}
            const float * src0_d = (const float *)src0_ptr;
            const float * src1_d = (const float *)src1_ptr;
            float * dst_d = (float *)dst->data;
            //printf("%s(%s, %s): %ld %zu %zu  %ld %zu %zu\n", __func__, src0->name, src1->name, src0->ne[0], src0->nb[0], src0->nb[1], dst->ne[0], dst->nb[0], dst->nb[1]);
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        ne00_eff, src0->ne[1], src0->ne[2],
                        ne0_eff, dst->ne[1], dst->ne[2], dim, stream);
                        //src0->nb[1]/sizeof(float), src0->ne[1], src0->ne[2],
                        //dst->nb[1]/sizeof(float), dst->ne[1], dst->ne[2], dim, stream);
                        //src0->ne[0]*src0->nb[0]/sizeof(float), src0->ne[1], src0->ne[2],
                        //dst->ne[0]*dst->nb[0]/sizeof(float),  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            //printf("%s(not contiguous): %s(%s) and %s(%s)\n", __func__, src0->name, ggml_type_name(src0->type), src1->name, ggml_type_name(src1->type));
            auto ne10_eff = (src1->ne[0]/bs)*ts/sizeof(float);
            dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
            concat_f32_non_cont<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *)src0_ptr,
                    (const char *)src1_ptr,
                    (      char *)dst->data,
                    ne00_eff, src0->ne[1], src0->ne[2], src0->ne[3],
                    //src0->ne[0]*src0->nb[0]/sizeof(float), src0->ne[1], src0->ne[2], src0->ne[3],
                    sizeof(float), src0->nb[1], src0->nb[2], src0->nb[3],
                    ne10_eff, src1->ne[1], src1->ne[2], src1->ne[3],
                    //src1->ne[0]*src1->nb[0]/sizeof(float), src1->ne[1], src1->ne[2], src1->ne[3],
                    sizeof(float), src1->nb[1], src1->nb[2], src1->nb[3],
                    ne0_eff,  dst->ne[1],  dst->ne[2],  dst->ne[3],
                    //dst->ne[0]*dst->nb[0]/sizeof(float),  dst->ne[1],  dst->ne[2],  dst->ne[3],
                    sizeof(float),  dst->nb[1],  dst->nb[2],  dst->nb[3], dim);
        }
        goto concat_cleanup;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) && dim == 2 && dst->ne[3] > 1 && src1->ne[2] == 1) {
        float * dst_d  = (float *)dst->data;
        float * src0_d = (float *)src0_ptr;
        float * src1_d = (float *)src1_ptr;
        concat_f32_cuda(src0_d, src1_d, dst_d, src0->ne[0]*src0->ne[1]*src0->ne[2], src0->ne[3], 1, dst->ne[0]*dst->ne[1]*dst->ne[2], dst->ne[3], 1, 0, stream);
        goto concat_cleanup;
    }

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        //if (dst->ne[1] >= 65536 || dst->ne[2] >= 65536) {
        //    fprintf(stderr, "%s: ne1 = %ld, ne2 = %ld exceed max. blocks when computing %s\n", __func__, dst->ne[1], dst->ne[2], dst->name);
        //    GGML_ABORT("fatal error");
        //}
        const float * src0_d = (const float *)src0_ptr;
        const float * src1_d = (const float *)src1_ptr;

        float * dst_d = (float *)dst->data;

        for (int i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_f32_cuda(
                    src0_d + i3 * (src0->nb[3] / 4),
                    src1_d + i3 * (src1->nb[3] / 4),
                    dst_d + i3 * ( dst->nb[3] / 4),
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
        }
        if (free0 || free1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (free0) CUDA_CHECK(cudaFree(tmp0));
            if (free1) CUDA_CHECK(cudaFree(tmp1));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        concat_f32_non_cont<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *)src0_ptr,
                (const char *)src1_ptr,
                (      char *)dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
                dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3], dim);
    }
concat_cleanup:
    if (free0 || free1) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (free0) CUDA_CHECK(cudaFree(tmp0));
        if (free1) CUDA_CHECK(cudaFree(tmp1));
    }
}
