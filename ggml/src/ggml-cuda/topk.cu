#include <algorithm>
#include <cstdint>

#include "topk.cuh"
#include "common.cuh"

// Compute top-K indices and values of each row of an F32 matrix.
// K is stored in dst->op_params[0] (0 = use ne[1] of dst for K).
// Output tensor is F32 with shape [2, K, nrows]:
//   plane 0 = values, plane 1 = indices (as float).
// One block per row. K-iteration approach: each iteration finds the
// max among elements not yet selected, using shared memory to
// broadcast the set of already-found indices.
static __global__ void topk_f32(
        const float * __restrict__ x,
        float     * __restrict__ dst,
        const int64_t ncols,
        const int K,
        const int64_t nrows) {

    const int64_t row    = blockIdx.x;
    const float * rowx   = x + row * ncols;
    const int     n_warps = blockDim.x / WARP_SIZE;
    const int     lane_id = threadIdx.x % WARP_SIZE;
    const int     warp_id = threadIdx.x / WARP_SIZE;

    extern __shared__ int selected_shared[]; // size K

    for (int k = 0; k < K; ++k) {
        float maxval = -FLT_MAX;
        int   argmax = -1;

        // Scan this thread's columns, skip already-found indices
        for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            bool skip = false;
            for (int s = 0; s < k; ++s) {
                if (selected_shared[s] == col) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;

            const float val = rowx[col];
            if (val > maxval) {
                maxval = val;
                argmax = col;
            }
        }

        // Warp shuffle reduction
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float v = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   c = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            if (v > maxval) {
                maxval = v;
                argmax = c;
            }
        }

        // Multi-warp: cross-warp reduction via shared memory
        if (n_warps > 1) {
            constexpr int max_warps = 1024 / WARP_SIZE;
            __shared__ float s_maxval[max_warps];
            __shared__ int   s_argmax[max_warps];
            if (lane_id == 0) {
                s_maxval[warp_id] = maxval;
                s_argmax[warp_id] = argmax;
            }
            __syncthreads();

            if (warp_id == 0) {
                if (lane_id < n_warps) {
                    maxval = s_maxval[lane_id];
                    argmax = s_argmax[lane_id];
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const float v = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                    const int   c = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                    if (v > maxval) {
                        maxval = v;
                        argmax = c;
                    }
                }
            }
        }

        if (warp_id == 0 && lane_id == 0) {
            // Plane 0: values, Plane 1: indices
            dst[k * nrows + row] = maxval;
            dst[(K + k) * nrows + row] = (float)argmax;
            selected_shared[k] = argmax;
        }
        __syncthreads();
    }
}

void ggml_cuda_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int K = dst->op_params[0] > 0 ? dst->op_params[0] : (int)dst->ne[1];

    GGML_ASSERT(K <= 256 && "topk K must be <= 256");
    GGML_ASSERT(K <= (int64_t)ne00);
    GGML_ASSERT(dst->ne[0] == 2);
    GGML_ASSERT(dst->ne[1] == K);
    GGML_ASSERT(dst->ne[2] == nrows);

    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);
    const size_t shared_mem = (size_t)K * sizeof(int);

    topk_f32<<<blocks_num, blocks_dim, shared_mem, stream>>>(src0_d, dst_d, ne00, K, nrows);
}
