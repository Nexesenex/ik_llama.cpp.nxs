//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// Bit-exact CUDA legacy GGUF quants (RN intrinsics, no FMA; host sigma2 in CPU order),
// base family file: Q8_0, Q6_0, Q5_0, Q4_0, Q5_1, Q4_1 (kernels + drivers here) plus
// the shared helpers (qleg-encoder.cuh) used by qk-encoder.cu and iq-encoder.cu.
// Family entry ggml_cuda_quantize_qleg(), routed from kt-encoder.cu like Joel's KS entry.
// Q8_0 uses Q6_0 fudge quirk (ggml-quants.c:915), ignores imatrix; Q6_0 OLS kept (make_qx+fudge).

#include "qleg-encoder.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <mutex>

#include "../ggml-quants.h"


// --- kernels ---

// Warp helpers shared by the plain kernels (all lanes must call uniformly).

// Argmax |x|, lowest index wins ties (matches ref scan; fixes sign of d).
static __device__ __forceinline__ float warp_argmax_first_device(float xi, int lane) {
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
    return __shfl_sync(0xffffffffu, xi, bidx);
}

// Min/max values (order-independent); used by the x1 plain kernels.
static __device__ __forceinline__ void warp_minmax_device(float xi, float & bmin, float & bmax) {
    bmin = xi;
    bmax = xi;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        bmin = fminf(bmin, __shfl_xor_sync(0xffffffffu, bmin, m));
        bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, m));
    }
}

// Nibble pack: lane j<16 writes byte j from pair j+16.
static __device__ __forceinline__ void pack_nibbles_shfl_device(uint32_t q, uint8_t * qs, int lane) {
    const uint32_t my   = q & 0xF;
    const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
    if (lane < 16) {
        qs[lane] = (uint8_t)(my | (pair << 4));
    }
}

// 5th-bit qh bitmap + nibble pack for one-thread-per-block Q5 kernels
// (L in [0,31]).
static __device__ void pack_q5_qh_device(const uint8_t * L, uint8_t * qs, uint8_t * qh) {
    uint32_t bits = 0;
    for (int j = 0; j < 16; ++j) {
        const uint8_t xi0 = L[j];
        const uint8_t xi1 = L[j + 16];
        qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        bits |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
        bits |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + 16);
    }
    qh[0] = (uint8_t)(bits >>  0);
    qh[1] = (uint8_t)(bits >>  8);
    qh[2] = (uint8_t)(bits >> 16);
    qh[3] = (uint8_t)(bits >> 24);
}

// 5th-bit qh bitmap, little-endian uint32 (Q5_0/Q5_1 warp kernels; all lanes
// must call uniformly).
static __device__ __forceinline__ void pack_qh_ballot_device(uint32_t q, uint8_t * qh, int lane) {
    const uint32_t bits = __ballot_sync(0xffffffffu, (q >> 4) & 1);
    if (lane == 0) {
        qh[0] = (uint8_t)(bits >>  0);
        qh[1] = (uint8_t)(bits >>  8);
        qh[2] = (uint8_t)(bits >> 16);
        qh[3] = (uint8_t)(bits >> 24);
    }
}

static __global__ void quantize_q8_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
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

        // __fdiv_rn for exact rounding under -use_fast_math; 1-ulp flips roundf() on k+0.5 ties.
        const float d  = __fmul_rn(fudge, __fdiv_rn(amax, 127.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        block_q8_0 * y = (block_q8_0 *)vy;
        y[ib].qs[lane] = (int8_t)roundf(xi*id);

        if (lane == 0) {
            // Raw-bit copy (ushort assignment would corrupt bits, e.g. 0x29f2->0x713e).
            // Bit-twiddle (not hardware RN) so NaN payloads match ggml_compute_fp32_to_fp16.
            y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(d));
        }
    }
}

static __global__ void quantize_q4_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_0 + lane];

        const float max = warp_argmax_first_device(xi, lane);
        // __fdiv_rn for 1/d (max/-8 exact pow2); CPU stores FP16(fudge*d).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -8.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // Truncation toward zero + clamp; __fmul_rn avoids FMA (1-ulp flips truncation).
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 8.5f);
        const int32_t q = v > 15 ? 15 : v;

        block_q4_0 * y = (block_q4_0 *)vy;
        if (lane == 0) {
            y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(d)); // raw bits, see Q8_0 kernel
        }

        pack_nibbles_shfl_device((uint32_t)q, y[ib].qs, lane);
    }
}

// --- Q5_0 (quantize_row_q5_0_ref) ---

static __global__ void quantize_q5_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_0 + lane];
        const float max = warp_argmax_first_device(xi, lane);
        // __fdiv_rn for 1/d (max/-16 exact pow2).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -16.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // Truncation + clamp; __fmul_rn avoids FMA (matches CPU roundings).
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 16.5f);
        const int32_t q = v > 31 ? 31 : v;

        block_q5_0 * y = (block_q5_0 *)vy;
        if (lane == 0) {
            y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(d)); // raw bits, see Q8_0 kernel
        }

        pack_nibbles_shfl_device((uint32_t)q, y[ib].qs, lane);
        pack_qh_ballot_device((uint32_t)q, y[ib].qh, lane);
    }
}

// --- Q5_1 (quantize_row_q5_1_ref, no fudge, no clamp) ---

static __global__ void quantize_q5_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q5_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_1 + lane];

        float bmin, bmax;
        warp_minmax_device(xi, bmin, bmax);

        const float d  = __fdiv_rn(bmax - bmin, 31.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // (x-min)*id + 0.5f truncation toward zero, no clamp (matches ref).
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const uint32_t q = (uint32_t)(int32_t)(t + 0.5f);

        block_q5_1 * y = (block_q5_1 *)vy;
        if (lane == 0) {
            // Raw bits via twiddle (see Q8_0 kernel) so NaN payloads match CPU.
            y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                                      __ushort_as_half(fp32_to_fp16_ggml(bmin)));
        }
        pack_nibbles_shfl_device(q, y[ib].qs, lane);
        pack_qh_ballot_device(q, y[ib].qh, lane);
    }
}

// --- Q4_1 (quantize_row_q4_1_ref, no fudge, MIN(15)) ---

static __global__ void quantize_q4_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q4_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_1 + lane];

        float bmin, bmax;
        warp_minmax_device(xi, bmin, bmax);

        const float d  = __fdiv_rn(bmax - bmin, 15.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(15, (int8_t)((x-min)*id + 0.5f)) truncation toward zero.
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const int32_t v = (int32_t)(t + 0.5f);
        const uint32_t q = (uint32_t)(v > 15 ? 15 : v);

        block_q4_1 * y = (block_q4_1 *)vy;
        if (lane == 0) {
            // Raw bits via twiddle (see Q8_0 kernel) so NaN payloads match CPU.
            y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                                      __ushort_as_half(fp32_to_fp16_ggml(bmin)));
        }
        pack_nibbles_shfl_device(q, y[ib].qs, lane);
    }
}

// --- Removable Q4_0 imatrix kernel (one thread/block; base keeps chunk indexing exact) ---
static __global__ void quantize_q4_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_0];
    int8_t L[QK4_0];
    imatrix_weights_device(xb, qb, s2, weight, QK4_0);

    const float d = make_qx_quants_device(QK4_0, 8, xb, L, weight);

    block_q4_0 * y = (block_q4_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));
    pack_q4_nibbles_device((const uint8_t *)L, y[ib].qs);
}

// --- Q4_1 imatrix via make_qkx3 (nmax=15, no fudge) ---
static __global__ void quantize_q4_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge; // no fudge for Q4_1 (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];
    imatrix_weights_device(xb, qb, s2, weight, QK4_1);

    float the_min;
    const float d = make_qkx3_quants_device(QK4_1, 15, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q4_1 * y = (block_q4_1 *)vy;
    // Bit-twiddle FP16 (NaN payload is CPU-vendor semantics).
    y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                              __ushort_as_half(fp32_to_fp16_ggml(-the_min)));
    pack_q4_nibbles_device(L, y[ib].qs);
}

        // --- Removable Q5_0 imatrix (nmax=16 + qh bitmap) ---
static __global__ void quantize_q5_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_0];
    int8_t L[QK5_0];
    imatrix_weights_device(xb, qb, s2, weight, QK5_0);

    const float d = make_qx_quants_device(QK5_0, 16, xb, L, weight);

    block_q5_0 * y = (block_q5_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    pack_q5_qh_device((const uint8_t *)L, y[ib].qs, y[ib].qh);
}

// --- Q5_1 imatrix via make_qkx3 (nmax=31, no fudge) ---
static __global__ void quantize_q5_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge; // no fudge for Q5_1 (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];
    imatrix_weights_device(xb, qb, s2, weight, QK5_1);

    float the_min;
    const float d = make_qkx3_quants_device(QK5_1, 31, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q5_1 * y = (block_q5_1 *)vy;
    // bit-twiddle FP16 (see Q4_1 imatrix kernel): NaN-scale payload must match.
    y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                              __ushort_as_half(fp32_to_fp16_ggml(-the_min)));

    pack_q5_qh_device(L, y[ib].qs, y[ib].qh);
}

// Q6 low nibbles + 2-bit qh pack (L in [0,63]), shared by imatrix/OLS kernels.
static __device__ void pack_q6_0_device(const int8_t * L, block_q6_0 * y, int64_t ib) {
    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
    }
    for (int k = 0; k < QK6_0/4; ++k) {
        const uint8_t a0 = (uint8_t)L[k];
        const uint8_t a1 = (uint8_t)L[k + QK6_0/2];
        const uint8_t b0 = (uint8_t)L[k + QK6_0/4];
        const uint8_t b1 = (uint8_t)L[k + 3*(QK6_0/4)];
        y[ib].qh[k] = (uint8_t)((a0 >> 4) | ((a1 >> 4) << 2)
                              | (((b0 >> 4) | ((b1 >> 4) << 2)) << 4));
    }
}

// --- Q6_0 imatrix (nmax=32, 2-bit qh) ---
static __global__ void quantize_q6_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK6_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK6_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK6_0];
    int8_t L[QK6_0];
    imatrix_weights_device(xb, qb, s2, weight, QK6_0);

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    pack_q6_0_device(L, y, ib);
}

// --- Q6_0 OLS kernel (w=x*x, make_qx+fudge) ---
static __global__ void quantize_q6_0_ols_kernel(
        const float * __restrict__ x,
        void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const float * xb = x + ib*QK6_0;

    float weight[QK6_0];
    int8_t L[QK6_0];
    for (int j = 0; j < QK6_0; ++j) {
        weight[j] = __fmul_rn(xb[j], xb[j]);
    }

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    pack_q6_0_device(L, y, ib);
}

// Shared chunked driver for warp-per-block plain kernels (block_size = 0
// selects one warp per quant block) and one-thread-per-block OLS/codebook
// kernels (block_size = 256). ~128 MiB F32 chunks bound VRAM use.
size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, quantize_kernel_t kernel, const char * name, float fudge, int block_size) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    // Runs on the ambient CUDA device selected by the ggml_cuda_quantize
    // dispatcher (ggml_cuda_set_device); device 0 when called directly.

    // Chunked ~128 MiB F32 input so large tensors avoid huge VRAM allocs.
    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(qk*(int64_t)sizeof(float)));
    // one warp per quant block (block_size 0, grid-stride loop in kernel)
    // or 256-thread one-block-per-thread
    const unsigned int launch_bs = block_size ? (unsigned)block_size : (unsigned)qk;
    const bool warp_path = !block_size;
    auto launch = [&](float * dx, uint8_t * dy, int64_t base, int64_t nblocks, cudaStream_t st) {
        (void) base;
        if (warp_path) {
            kernel<<<(unsigned)nblocks, launch_bs, 0, st>>>(dx, dy, nblocks, fudge);
        } else {
            kernel<<<(unsigned)((nblocks + launch_bs - 1)/launch_bs), launch_bs, 0, st>>>(dx, dy, nblocks, fudge);
        }
    };

    return quantize_exec_chunked(src, dst, nblocks_total, qk, blk_size, chunk_blocks, name, launch);
}


// Shared chunked driver for all imatrix kernels. sigma covers sigma_len values
// per entry: a full row (sum/n_per_row, CPU row order) or one quant unit
// (sum*(2/qk), CPU block order); kernels index it per row or per block.
size_t ggml_cuda_quantize_imatrix_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, int64_t qk, size_t blk_size, imatrix_kernel_t kernel, const char * name, float fudge,
        bool per_block_sigma) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);
    const int32_t blocks_per_row = (int32_t)(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    // Runs on the ambient device (dispatcher selects it).

    const int64_t nsigma = per_block_sigma ? nblocks_total : nrows;
    const int64_t sigma_len = per_block_sigma ? qk : n_per_row;
    std::vector<float> sigma2(nsigma);
    for (int64_t i = 0; i < nsigma; ++i) {
        const float * xr = src + i*sigma_len;
        float sum = 0.0f;
        for (int64_t j = 0; j < sigma_len; ++j) {
            sum += xr[j]*xr[j];
        }
        sigma2[i] = per_block_sigma ? sum*(2.0f/(float)qk) : sum/(float)n_per_row;
    }

    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(qk*(int64_t)sizeof(float)));

    float   * q_dev = nullptr;
    float   * s_dev = nullptr;

    cudaError_t err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(q_dev): %s\n", __func__, name, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&s_dev, nsigma*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(s_dev): %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(q_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMemcpy imatrix H2D: %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nsigma*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMemcpy sigma2 H2D: %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    auto launch = [&](float * dx, uint8_t * dy, int64_t base, int64_t nblocks, cudaStream_t st) {
        kernel<<<(unsigned)((nblocks + 256 - 1)/256), 256, 0, st>>>(
                dx, q_dev, s_dev, dy, base, nblocks, blocks_per_row, fudge);
    };

    const size_t out = quantize_exec_chunked(src, dst, nblocks_total, qk, blk_size, chunk_blocks, name, launch);

    cudaFree(q_dev);
    cudaFree(s_dev);

    return out;
}

// Order Q8_0/Q6_0/Q5_0/Q4_0; Q8_0 keeps Q6_0-fudge quirk (ggml-quants.c:915).
static size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0); // match CPU quirk
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK8_0, sizeof(block_q8_0), quantize_q8_0_kernel, "q8_0", fudge, 0);
}


// --- Removable Q5_0 section (delete kernel+helpers+cases to drop) ---
static size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_0, sizeof(block_q5_0), quantize_q5_0_kernel, "q5_0", fudge, 0);
}

// --- Removable Q4_0 section (delete kernel+helpers+cases to drop) ---
static size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_0, sizeof(block_q4_0), quantize_q4_0_kernel, "q4_0", fudge, 0);
}

// --- Q4_1 (quantize_row_q4_1_ref, no fudge) ---
static size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_1, sizeof(block_q4_1), quantize_q4_1_kernel, "q4_1", 1.0f, 0);
}

// --- Removable Q4_0 imatrix (reuses row imatrix like CPU) ---
static size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_0, sizeof(block_q4_0),
            quantize_q4_0_imatrix_kernel, "q4_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0), false);
}

// --- Removable Q5_0 imatrix (mirror of Q4_0 imatrix) ---
static size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_0, sizeof(block_q5_0),
            quantize_q5_0_imatrix_kernel, "q5_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0), false);
}

// --- Q6_0 OLS imatrix (mirror of Q4_0 imatrix, nmax=32) ---
static size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_imatrix_kernel, "q6_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0), false);
}

// --- Q5_1 (quantize_row_q5_1_ref, no fudge) ---
static size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_1, sizeof(block_q5_1), quantize_q5_1_kernel, "q5_1", 1.0f, 0);
}

// --- Q5_1 imatrix via make_qkx3 (no fudge) ---
static size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_1, sizeof(block_q5_1),
            quantize_q5_1_imatrix_kernel, "q5_1_imatrix", 1.0f, false);
}

// --- Q4_1 imatrix via make_qkx3 (no fudge) ---
static size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_1, sizeof(block_q4_1),
            quantize_q4_1_imatrix_kernel, "q4_1_imatrix", 1.0f, false);
}

// --- Q6_0 OLS driver (w=x*x, make_qx+fudge) ---
static size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_ols_kernel, "q6_0", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0), 256);
}

// Q8_0 imatrix routes to plain (CPU ignores imatrix).
static size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix; // CPU ignores it; match exactly
    return ggml_cuda_quantize_q8_0(src, dst, nrows, n_per_row);
}

// Family entry: plain or imatrix driver per slice (Q8_0 ignores imatrix like CPU).
size_t ggml_cuda_quantize_qleg(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix) {
    (void) device; // ambient device selected by the dispatcher
    const size_t row_size = ggml_row_size(type, n_per_row);
    const int64_t nelements = nrows*n_per_row;
    size_t total = 0;
    for (int64_t s = 0; s < nslice; ++s) {
        const float * s_src = src + s*nelements;
        char * s_dst = (char *)dst + s*nrows*row_size;
        const float * s_im = imatrix ? imatrix + s*n_per_row : nullptr;
        size_t nb = 0;
        switch (type) {
            case GGML_TYPE_Q8_0: nb = ggml_cuda_quantize_q8_0(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q6_0: nb = s_im ? ggml_cuda_quantize_q6_0_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q6_0(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q5_0: nb = s_im ? ggml_cuda_quantize_q5_0_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q5_0(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q4_0: nb = s_im ? ggml_cuda_quantize_q4_0_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q4_0(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q5_1: nb = s_im ? ggml_cuda_quantize_q5_1_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q5_1(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q4_1: nb = s_im ? ggml_cuda_quantize_q4_1_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q4_1(s_src, s_dst, nrows, n_per_row); break;
            default: return 0;
        }
        if (nb == 0) {
            return 0;
        }
        total += nb;
    }
    return total;
}
