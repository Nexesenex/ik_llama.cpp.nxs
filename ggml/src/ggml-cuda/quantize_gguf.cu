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
// The 32-value quant blocks tile the flat row-major F32 buffer contiguously
// (n_per_row % 32 == 0), so rows need no explicit bookkeeping. Each warp
// quantizes one block independently; max-reduction via shuffles is exact and
// order-independent, and the per-element rounding is backend-deterministic.
// The result is byte-for-byte identical to quantize_row_q8_0_ref on any GPU.

#include "quantize_gguf.cuh"

#include <cinttypes>
#include <cstdio>
#include <algorithm>

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

size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK8_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK8_0);

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
    const int64_t chunk_x      = chunk_blocks*QK8_0;       // floats per chunk
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q8_0); // bytes per chunk

    float      * x_dev = nullptr;
    block_q8_0 * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    // one warp per quant block
    const int64_t block_size = QK8_0;

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK8_0, nblocks*QK8_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        quantize_q8_0_kernel<<<(unsigned)nblocks, (unsigned)block_size>>>(x_dev, y_dev, nblocks);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q8_0), y_dev, nblocks*sizeof(block_q8_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q8_0);
}
