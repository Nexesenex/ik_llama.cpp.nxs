//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// Bit-exact CUDA quantization of legacy (non-OLS) block quants for GGUF.
//
// Q8_0 target: quantize_row_q8_0_ref (ggml/src/ggml-quants.c):
//   amax = max(|x_j|) over the 32-value block
//   d    = amax/127            (__fdiv_rn: exact under -use_fast_math)
//   id   = d ? 1/d : 0         (__fdiv_rn: exact under -use_fast_math)
//   q_j  = roundf(x_j*id)      // round half away from zero
//   d    = FP16(d)             // round to nearest even
//
// Q4_0 target: quantize_row_q4_0_ref (ggml/src/ggml-quants.c):
//   max  = signed value with max |x_j| (first occurrence wins |x| ties)
//   d    = max/-8              (__fdiv_rn: exact under -use_fast_math)
//   id   = d ? 1/d : 0         (__fdiv_rn: exact under -use_fast_math)
//   q_j  = MIN(15, (int8_t)(x_j*id + 8.5f))  // truncation toward zero
//   d    = FP16(d)
//   byte j (0..15) = low nibble q_j | high nibble q_{j+16} << 4
//
// The 32-value quant blocks tile the flat row-major F32 buffer contiguously
// (n_per_row % 32 == 0), so rows need no explicit bookkeeping. Each warp
// quantizes one block independently; the max reduction via shuffles is exact
// and order-independent, the argmax tie-break (lowest index wins) matches the
// reference's sequential scan, and the per-element rounding is
// backend-deterministic. The result is byte-for-byte identical to the
// quantize_row_*_ref implementations on any GPU.
//
// Both public entry points share one chunked host driver (fixed ~128 MiB F32
// device chunks) so single large tensors never need a large contiguous VRAM
// allocation; every CUDA call is checked, and on failure the error is printed
// and 0 returned (the caller aborts the quantization).

#include "quantize_gguf.cuh"

#include <cinttypes>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

static __global__ void quantize_q8_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK8_0

    // grid-stride over quant blocks (nblocks can exceed UINT_MAX on big models)
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK8_0 + lane];

        // exact, order-independent max reduction (max is associative/exact)
        float amax = fabsf(xi);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        }

        // __fdiv_rn: correctly-rounded IEEE division. The build uses
        // -use_fast_math, which makes plain '/' approximate (reciprocal +
        // multiply); the CPU ref divides with exact rounding, and an ulp
        // difference in d/id flips roundf() exactly on k+0.5 ties.
        const float d  = __fdiv_rn(amax, 127.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        block_q8_0 * y = (block_q8_0 *)vy;
        y[ib].qs[lane] = (int8_t)roundf(xi*id);

        if (lane == 0) {
            // store the __half directly. Do NOT round-trip through
            // __half_as_ushort + assignment: block_q8_0.d is __half, and
            // `half = unsigned short` converts the ushort as a *number*
            // (half(float(ushort))), corrupting the scale bits (e.g. 0x29f2
            // -> 0x713e). Assigning the __half copies the raw bits.
            y[ib].d = __float2half_rn(d);
        }
    }
}

static __global__ void quantize_q4_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_0 + lane];

        // argmax of |x|. On |x| ties the *first* (lowest index) element wins,
        // exactly like the ref's sequential scan: quantize_row_q4_0_ref keeps
        // the signed value `max` of the first max-magnitude element, which
        // fixes the sign of d.
        float   bval = fabsf(xi);
        int32_t bidx = lane;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float   oval = __shfl_xor_sync(0xffffffffu, bval, m);
            const int32_t oidx = __shfl_xor_sync(0xffffffffu, bidx, m);
            if (oval > bval || (oval == bval && oidx < bidx)) {
                bval = oval;
                bidx = oidx;
            }
        }

        // signed value of the argmax element, broadcast to the warp
        const float max = __shfl_sync(0xffffffffu, xi, bidx);

        // __fdiv_rn: correctly-rounded IEEE division (see Q8_0 kernel). max/-8
        // is a power-of-2 division (exact anyway); 1/d is the general case.
        const float d  = __fdiv_rn(max, -8.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(15, (int8_t)(x_j*id + 8.5f)): truncation toward zero, then clamp.
        // |x_j*id| <= 8 so x_j*id + 8.5 is in [-0.5, 16.5] and the truncation
        // is always representable (matches the ref's (int8_t) cast exactly).
        // __fmul_rn forces the multiply to round once: with -use_fast_math nvcc
        // otherwise contracts `xi*id + 8.5f` into one FMA (single rounding)
        // while the CPU rounds the product and the add separately, and that
        // 1-ulp difference flips the truncation at integer thresholds.
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 8.5f);
        const int32_t q = v > 15 ? 15 : v;

        block_q4_0 * y = (block_q4_0 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d); // store the __half, see Q8_0 kernel
        }

        // byte j (0..15): low nibble = element j, high nibble = element j+16.
        // lane j<16 writes its byte using the nibble received from lane j+16.
        const uint32_t my   = (uint32_t)q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }
    }
}

// ---------------------------------------------------------------------------
// host driver (shared by all block quants)
// ---------------------------------------------------------------------------

using quantize_kernel_t = void (*)(const float *, void *, int64_t);

static size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, quantize_kernel_t kernel, const char * name) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Fixed-size device buffers; the tensor is processed in chunks so that a
    // single large tensor never needs a huge VRAM allocation (which can fail
    // silently while another context holds most of the memory, e.g. a loaded
    // model in llama-server/webui, and then produce a confusing illegal-access
    // error from the NULL pointers). ~128 MiB of F32 input per chunk.
    const int64_t chunk_blocks = 1 << 20;                  // quant blocks per chunk
    const int64_t chunk_x      = chunk_blocks*qk;          // floats per chunk
    const int64_t chunk_y      = chunk_blocks*blk_size;    // bytes per chunk

    float   * x_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    // one warp per quant block
    const int64_t block_size = qk;

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }

        kernel<<<(unsigned)nblocks, (unsigned)block_size>>>(x_dev, y_dev, nblocks);

        err = cudaMemcpy((char *)dst + base*blk_size, y_dev, nblocks*blk_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK8_0, sizeof(block_q8_0), quantize_q8_0_kernel, "q8_0");
}

size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_0, sizeof(block_q4_0), quantize_q4_0_kernel, "q4_0");
}
