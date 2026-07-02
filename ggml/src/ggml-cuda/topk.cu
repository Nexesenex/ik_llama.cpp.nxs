#include <algorithm>
#include <cstdint>

#include "topk.cuh"
#include "common.cuh"

// Compute top-K indices and values of each row of an F32 matrix.
// K is stored in dst->op_params[0] (0 = use ne[1] of dst for K).
// Output tensor is F32 with shape [2, K, nrows]:
//   plane 0 = values, plane 1 = indices (as float).
// One block per row. Uses a shared-memory bitmap to track already-selected
// columns, giving O(1) skip checks instead of O(k) linear scan.
//
// Shared memory layout:
//   [bitmap: (ncols+31)/32 ints] [s_maxval: max_warps floats] [s_argmax: max_warps ints]
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
    const int     bitmap_ints = ((int)ncols + 31) / 32;

    extern __shared__ char smem[];
    int   * bitmap   = (int   *)smem;
    float * s_maxval = (float *)(smem + bitmap_ints * sizeof(int));
    int   * s_argmax = (int   *)(smem + bitmap_ints * sizeof(int) + 32 * sizeof(float));

    // Initialize bitmap to 0
    for (int i = threadIdx.x; i < bitmap_ints; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();

    for (int k = 0; k < K; ++k) {
        float maxval = -FLT_MAX;
        int   argmax = -1;

        // Scan this thread's columns, using the bitmap for O(1) skip
        for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            if (bitmap[col >> 5] & (1u << (col & 31))) continue;
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
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float v = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                    int   c = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
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
            bitmap[argmax >> 5] |= (1u << (argmax & 31));
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

    GGML_ASSERT(K <= 4096 && "topk K must be <= 4096");
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

    const int bitmap_ints = ((int)ne00 + 31) / 32;
    const size_t shared_mem = (size_t)bitmap_ints * sizeof(int)   // bitmap
                            + 32 * sizeof(float)                   // s_maxval (max_warps = 32)
                            + 32 * sizeof(int);                    // s_argmax

    topk_f32<<<blocks_num, blocks_dim, shared_mem, stream>>>(src0_d, dst_d, ne00, K, nrows);
}
