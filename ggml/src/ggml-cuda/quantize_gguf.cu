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
// Q4_0 with importance matrix: quantize_row_q4_0_impl (ggml-quants.c:3429).
// Each 32-value block is quantized by make_qx_quants (ggml-quants.c:1786), a
// deterministic sequential greedy optimizer. One thread per block replays it
// in the exact CPU order with correctly-rounded intrinsics, so it is
// byte-identical too. The row-level sigma2 sum is order-dependent, so it is
// pre-computed on the host in the exact CPU summation order.
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
#include <vector>

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
// Q4_0 with importance matrix: make_qx_quants (ggml-quants.c:1786)
// ---------------------------------------------------------------------------

// Byte-exact port of ggml_compute_fp32_to_fp16 (ggml/src/ggml-impl.h:595), the
// fp16 conversion the CPU reference uses when __F16C__ is off (as in this
// build). __float2half_rn agrees with it on finite values and overflow to inf
// (both round to nearest even), but encodes NaN differently: the hardware
// conversion emits the canonical 0x7fff, while this bit-mask path emits
// (sign ? 0xfe00 : 0x7e00). The make_qx_quants scale can be NaN for degenerate
// imatrix blocks, so this must be byte-exact too.
//
// Note: only the NaN *sign/payload* is not normalized here, and that is a
// CPU-vendor semantic difference, not a porting gap: x86 SSE sets the sign of
// a NaN result from the operand signs (so e.g. -inf/+inf and 0*inf render as
// -NaN -> 0xfe00), whereas NVIDIA hardware always emits the default quiet NaN
// (+NaN -> 0x7e00). Even CPU-only llama.cpp produces different bytes for these
// degenerate, NaN-scale blocks across CPU vendors. The harness therefore treats
// any NaN d as equal (spec.nan_d_equal); every finite block must still match
// byte-for-byte.
static __device__ __forceinline__ uint16_t fp32_to_fp16_ggml(float f) {
    const float scale_to_inf  = __int_as_float(0x77800000u);
    const float scale_to_zero = __int_as_float(0x08800000u);
    float base = __fmul_rn(__fmul_rn(fabsf(f), scale_to_inf), scale_to_zero);

    const uint32_t w      = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = __fadd_rn(__int_as_float((bias >> 1) + 0x07800000u), base);
    const uint32_t bits          = __float_as_uint(base);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// round-half-to-even via the 2^23 + 2^22 magic constant (ggml-quants.c:1779)
static __device__ int nearest_int_device(float fval) {
    const unsigned int u = __float_as_uint(__fadd_rn(fval, 12582912.0f));
    return (int)((u & 0x007fffffu) - 0x00400000u);
}

// clamp_l_device(l, nmax) as in make_qx_quants
static __device__ int clamp_l_device(int l, int nmax) {
    return l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
}

// Byte-exact device port of make_qx_quants, restricted to the path the Q4_0
// imatrix quantizer uses: rmse_type == 1 with a non-null weight vector. One
// thread replays the whole sequential algorithm in the exact CPU order, so
// the greedy search and coordinate-descent loop take the identical sequence
// of steps. Every float op is a correctly-rounded intrinsic so the
// -use_fast_math build (approximate sqrt/div) and FMA contraction cannot
// change a single bit: `sumlx += w*x*l` is (w*x)*l + sumlx with separate
// roundings, exactly like the non-contracting host compiler.
static __device__ float make_qx_quants_device(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max  = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)nmax, max);

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = clamp_l_device(l, nmax);
        L[i] = (int8_t)(l + nmax);
        const float w = qw[i]; // rmse_type == 1, qw always provided
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    float best_sumlx = sumlx, best_suml2 = suml2;

    for (int is = -9; is <= 9; ++is) {
        // iscale = -(nmax + 0.1*is)/max
        iscale = __fdiv_rn(-__fadd_rn((float)nmax, __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        // iscale = (nmax-1 + 0.1*is)/max
        iscale = __fdiv_rn(__fadd_rn((float)(nmax-1), __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    // coordinate descent; identical step sequence to the reference
    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0.0f, gmax = 0.0f;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            const float w = qw[j];
            const int l = (int)L[j] - nmax;
            // g = scale*w*(x[j] - scale*l), each op rounded separately
            const float g = __fmul_rn(__fmul_rn(scale, w),
                    __fadd_rn(x[j], -__fmul_rn(scale, (float)l)));
            if ((g > 0.0f && l < nmax-1) || (g < 0.0f && l > -nmax)) {
                const float ag = fabsf(g);
                if (ag > abs_gmax) { abs_gmax = ag; gmax = g; best_j = j; }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        const float w = qw[best_j];
        int l = (int)L[best_j] - nmax;
        if (gmax > 0.0f) {
            new_sumlx = __fadd_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fadd_rn(new_suml2, __fmul_rn(w, (float)(2*l + 1)));
            l += 1;
        } else {
            new_sumlx = __fsub_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fsub_rn(new_suml2, __fmul_rn(w, (float)(2*l - 1)));
            l -= 1;
        }
        if (new_suml2 > 0.0f && __fmul_rn(new_sumlx, new_sumlx) > __fmul_rn(best, new_suml2)) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            L[best_j] = (int8_t)(l + nmax);
        } else {
            break;
        }
    }
    return scale;
}

// One thread per quant block: computes the per-block weights from the shared
// row sigma2 and the (row-reused) importance weights, then runs
// make_qx_quants. `base` is the chunk's first global quant block, so the
// row/weight indexing stays correct when a tensor spans several device chunks
// and the chunk base is not a multiple of blocks_per_row. Block gb = base + ib
// belongs to row gb/blocks_per_row and uses weight block gb%blocks_per_row,
// matching quantize_row_q4_0_impl which is called once per row with the same
// quant_weights pointer.
static __global__ void quantize_q4_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row) {
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
    for (int j = 0; j < QK4_0; ++j) {
        // weight[j] = qw[j]*sqrtf(sigma2 + xb[j]^2); __fsqrt_rn keeps the
        // square root exact under -use_fast_math, __fadd_rn/__fmul_rn keep
        // the sum/multiply uncontracted.
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK4_0, 8, xb, L, weight);

    block_q4_0 * y = (block_q4_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(d));
    for (int j = 0; j < QK4_0/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_0/2] << 4));
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

// Q4_0 with an importance matrix. `imatrix` holds n_per_row weights and is
// reused for every row, exactly like the CPU quantize_row_q4_0_impl which is
// called once per row with the same quant_weights pointer.
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK4_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK4_0);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK4_0);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Per-row sigma2 = sum_x2/n_per_row, summed sequentially in the exact
    // order of quantize_row_q4_0_impl. The host compiler is cl with default
    // /fp:precise (no fast-math, no FMA contraction), so this loop produces
    // the identical float as the CPU reference. A row sum cannot be reduced
    // in parallel on the GPU: different summation order -> different bits.
    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    // Fixed-size device chunks, same rationale as the generic driver.
    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK4_0;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q4_0);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(q_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(n_per_row*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(s_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(nrows*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK4_0, nblocks*QK4_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        // 256 threads per block, one quant block per thread
        const unsigned int block_size = 256;
        quantize_q4_0_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q4_0), y_dev, nblocks*sizeof(block_q4_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q4_0);
}
