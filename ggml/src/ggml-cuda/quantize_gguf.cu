//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// Bit-exact CUDA GGUF quants (RN intrinsics, no FMA; host sigma2 in CPU order),
// via Joel's ggml_cuda_quantize() (kt-encoder.cu).
// Order: Q8_0, Q6_0, Q5_0, Q4_0, Q5_1, Q4_1, IQ4_NL, IQ4_XS. Q4_0/Q5_0 removable (delete kernel+helpers+cases).
// Q8_0 uses Q6_0 fudge quirk (ggml-quants.c:915), ignores imatrix; Q6_0 OLS kept (make_qx+fudge).

#include "quantize_gguf.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <mutex>

#include "../ggml-quants.h"

// Forward declaration: bit-twiddle FP16 matching ggml_compute_fp32_to_fp16
// (defined below with the imatrix section). Declared early so the plain
// kernels above can use it for NaN-payload-exact stores.
static __device__ __forceinline__ uint16_t fp32_to_fp16_ggml(float f);

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

// Q4 nibble pack for one-thread-per-block kernels (L in [0,15]).
static __device__ void pack_q4_nibbles_device(const uint8_t * L, uint8_t * qs) {
    for (int j = 0; j < 16; ++j) {
        qs[j] = (uint8_t)(L[j] | (L[j + 16] << 4));
    }
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

// --- Q6_0 OLS kept (make_qx_quants+fudge; plain max/-32 unused) ---

// --- make_qx_quants device port ---

// Bit-twiddle FP16 matching ggml_compute_fp32_to_fp16; NaN payload is vendor-specific (harness nan_d_equal).
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

// make_qx_quants port (rmse_type==1 only); RN intrinsics, exact CPU order, no FMA.
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

// --- make_qkx3_quants device port (Q4_1/Q5_1; double RN intrinsics, CPU order) ---
static __device__ float make_qkx3_quants_device(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w = __dadd_rn(sum_w, (double)w);
        sum_x = __dadd_rn(sum_x, __dmul_rn((double)w, (double)x[i]));
        sum_x2 = __dadd_rn(sum_x2, __dmul_rn(__dmul_rn((double)w, (double)x[i]), (double)x[i]));
    }
    if (min > 0) {
        min = 0;
    }
    if (max - min < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = __fdiv_rn((float)nmax, __fsub_rn(max, min));
    float scale = __fdiv_rn(1.0f, iscale);
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        double diff = __dadd_rn(__dmul_rn((double)scale, (double)L[i]), (double)min);
        diff = __dsub_rn(diff, (double)x[i]);
        diff = use_mad ? fabs(diff) : __dmul_rn(diff, diff);
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad = __dadd_rn(best_mad, __dmul_rn(w, diff));
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, (float)is)), (float)nmax), __fsub_rn(max, min));
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  = __dadd_rn(sum_l, __dmul_rn((double)w, (double)l));
            sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn((double)w, (double)l), (double)l));
            sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn((double)w, (double)l), (double)x[i]));
        }
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D > 0) {
            double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
            double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
            if (this_min > 0) {
                this_min = 0;
                this_scale = __ddiv_rn(sum_xl, sum_l2);
            }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = __dadd_rn(__dmul_rn((double)this_scale, (double)Laux[i]), (double)this_min);
                    diff = __dsub_rn(diff, (double)x[i]);
                    diff = fabs(diff);
                    double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
                    mad = __dadd_rn(mad, __dmul_rn(w, diff));
                }
            } else {
                mad = __dsub_rn(sum_x2, __dmul_rn(2*this_scale, sum_xl));
                mad = __dsub_rn(mad, __dmul_rn(2*this_min, sum_x));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = (float)this_scale;
                min = (float)this_min;
            }
        }
    }
    if (use_mad) {
        *the_min = -min;
        return scale;
    }

    double sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        int l = L[i];
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)l));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn(w, (double)l), (double)l));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)l), (double)x[i]));
    }
    double best = __dsub_rn(
            __dadd_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), sum_xl),
                      __dmul_rn(__dmul_rn(2.0, (double)min), sum_x)),
            __dmul_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), (double)min), sum_l));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)scale, (double)scale), sum_l2));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)min, (double)min), sum_w));
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0;
        int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            // NOTE float (not double): CPU computes g in float32.
            const float g = __fsub_rn(__fsub_rn(x[j], __fmul_rn(scale, (float)L[j])), min);
            if (g > 0 && L[j] < nmax && g > gmax) {
                gmax = g; best_j = j; dir = 1;
            }
            else if (g < 0 && L[j] > 0 && -g > gmax) {
                gmax = -g; best_j = j; dir = -1;
            }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)dir));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(w, (double)(2*L[best_j]*dir + 1)));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)x[best_j]), (double)dir));
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D <= 0) break;
        double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
        double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
        if (this_min > 0) {
            this_min = 0;
            this_scale = __ddiv_rn(sum_xl, sum_l2);
        }
        if (this_scale < 0) break;
        double score = __dadd_rn(__dmul_rn(2*this_scale, sum_xl), __dmul_rn(2*this_min, (double)sum_x));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
        if (score <= best) break;
        best = score;
        scale = (float)this_scale;
        min = (float)this_min;
        L[best_j] += (uint8_t)dir;
        last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}

// Per-block imatrix weights, shared by all imatrix kernels.
static __device__ void imatrix_weights_device(const float * xb, const float * qb, float s2, float * weight, int n) {
    for (int j = 0; j < n; ++j) {
        // __fsqrt_rn keeps sqrt exact under -use_fast_math, no FMA.
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }
}

// Shared IQ2/IQ3 helpers (same RN op order as ggml-quants.c; templates keep group geometry exact).
template<int GS, typename GridT>
static __device__ int find_best_neighbour_device(const uint16_t * neighbours, const GridT * grid,
        const float * xval, const float * weight, float scale, int8_t * L) {
    const int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = 0;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0.0f;
        for (int i = 0; i < GS; ++i) {
            const float q = (float)pg[i];
            const float diff = __fsub_rn(__fmul_rn(scale, q), xval[i]);
            d2 = __fadd_rn(d2, __fmul_rn(__fmul_rn(weight[i], diff), diff));
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < GS; ++i) {
        L[i] = (int8_t)((pg[i] - 1)/2);
    }
    return grid_index;
}

// IQ3 weights: plain x*x, imatrix qw*sqrt (waux=sqrt, RN, no FMA).
template<int N>
static __device__ void iq3_weights_device(const float * xb, const float * qb, float s2, bool has_imatrix,
        float * weight, float * waux) {
    if (has_imatrix) {
        for (int i = 0; i < N; ++i) {
            weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[i], xb[i]))));
        }
    } else {
        for (int i = 0; i < N; ++i) {
            weight[i] = __fmul_rn(xb[i], xb[i]);
        }
    }
    for (int i = 0; i < N; ++i) {
        waux[i] = __fsqrt_rn(weight[i]);
    }
}

// IQ2 weights: plain 0.25*sigma+x*x, imatrix qw*sqrt (RN, no FMA).
template<int N>
static __device__ void iq2_weights_device(const float * xb, const float * qb, float s2, bool has_imatrix,
        float * weight, float * waux) {
    if (has_imatrix) {
        for (int i = 0; i < N; ++i) {
            weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[i], xb[i]))));
        }
    } else {
        for (int i = 0; i < N; ++i) {
            weight[i] = __fadd_rn(__fmul_rn(0.25f, s2), __fmul_rn(xb[i], xb[i]));
        }
    }
    for (int i = 0; i < N; ++i) {
        waux[i] = __fsqrt_rn(weight[i]);
    }
}

// Direct signs (no parity): xval=abs, s bit per negative like CPU branch.
template<int K>
static __device__ void extract_signs_direct_device(const float * xb, float * xval, uint8_t * bs) {
    for (int k = 0; k < K; ++k) {
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[8*k + i] >= 0) {
                xval[8*k + i] = xb[8*k + i];
            } else {
                xval[8*k + i] = -xb[8*k + i];
                s |= (uint8_t)(1 << i);
            }
        }
        bs[k] = s;
    }
}

// Parity signs: enforce even parity, flip min w*x*x on odd like CPU.
template<int K>
static __device__ void extract_signs_parity_device(const float * xb, const float * weight, float * xval, uint8_t * bs) {
    for (int k = 0; k < K; ++k) {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[8*k + i] >= 0) {
                xval[8*k + i] = xb[8*k + i];
            } else {
                xval[8*k + i] = -xb[8*k + i];
                ++nflip;
                s |= (uint8_t)(1 << i);
            }
        }
        if (nflip % 2) {
            int imin = 0;
            float min = __fmul_rn(__fmul_rn(weight[8*k], xb[8*k]), xb[8*k]);
            for (int i = 1; i < 8; ++i) {
                const float ax = __fmul_rn(__fmul_rn(weight[8*k + i], xb[8*k + i]), xb[8*k + i]);
                if (ax < min) {
                    min = ax;
                    imin = i;
                }
            }
            xval[8*k + imin] = -xval[8*k + imin];
            s ^= (uint8_t)(1 << imin);
        }
        bs[k] = (uint8_t)(s & 127);
    }
}

// Ternary MAX (not fmaxf) to match ggml-impl.h MAX macro, NaN included.
template<int N>
static __device__ float max_abs_device(const float * xval) {
    float max = xval[0];
    for (int i = 1; i < N; ++i) {
        max = max > xval[i] ? max : xval[i];
    }
    return max;
}

// One subgroup quantize: clamp Laux, kmap, neighbour fallback (RN, CPU order).
template<int GS, int SHIFT, typename GridT>
static __device__ int quantize_subgroup_device(float id, float this_scale, const float * xval, const float * waux,
        int8_t * Laux, const int * kmap, const uint16_t * neighbours, const GridT * grid, bool * on_grid_aux, int nmax) {
    for (int i = 0; i < GS; ++i) {
        int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, xval[i]), 1.0f)));
        l = l < 0 ? 0 : (l > nmax - 1 ? nmax - 1 : l);
        Laux[i] = (int8_t)l;
    }
    uint16_t u = 0;
    for (int i = 0; i < GS; ++i) u |= (uint16_t)(Laux[i] << SHIFT*i);
    const int grid_index = kmap[u];
    *on_grid_aux = true;
    if (grid_index < 0) {
        *on_grid_aux = false;
        const uint16_t * nb = neighbours - grid_index - 1;
        return find_best_neighbour_device<GS>(nb, grid, xval, waux, this_scale, Laux);
    }
    return grid_index;
}

// Second-pass subgroup re-quantize: u from id, kmap, neighbour fallback or pg copy into L.
template<int GS, int SHIFT, typename GridT>
static __device__ void requantize_subgroup_device(float id, float scale, const float * xval, const float * waux,
        int8_t * L, const int * kmap, const uint16_t * neighbours, const GridT * grid, int nmax) {
    uint16_t u = 0;
    for (int i = 0; i < GS; ++i) {
        int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, xval[i]), 1.0f)));
        l = l < 0 ? 0 : (l > nmax - 1 ? nmax - 1 : l);
        u |= (uint16_t)(l << SHIFT*i);
    }
    const int grid_index = kmap[u];
    if (grid_index < 0) {
        const uint16_t * nb = neighbours - grid_index - 1;
        find_best_neighbour_device<GS>(nb, grid, xval, waux, scale, L);
    } else {
        const int8_t * pg = (const int8_t *)(grid + grid_index);
        for (int i = 0; i < GS; ++i) L[i] = (int8_t)((pg[i] - 1)/2);
    }
}

// Sweep best update: q=2*L+1 LS fit, strict > like CPU (returns improved).
template<int N>
static __device__ bool try_update_scale_device(const float * weight, const float * xval, const int8_t * Laux,
        float * scale, float * best, int8_t * L, const bool * aux_grid, bool * grid, int ngrp) {
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float q = __fadd_rn(__fmul_rn(2.0f, (float)Laux[i]), 1.0f);
        sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(weight[i], xval[i]), q));
        sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(weight[i], q), q));
    }
    if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(*best, sumq2)) {
        *scale = __fdiv_rn(sumqx, sumq2);
        *best = __fmul_rn(*scale, sumqx);
        for (int i = 0; i < N; ++i) L[i] = Laux[i];
        for (int k = 0; k < ngrp; ++k) grid[k] = aux_grid[k];
        return true;
    }
    return false;
}

// Second-pass refit tail: scale=sumqx/sumq2 if sumq2>0 (RN, CPU order).
template<int N>
static __device__ float refit_scale_device(const float * weight, const float * xval, const int8_t * L, float scale) {
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float q = __fadd_rn(__fmul_rn(2.0f, (float)L[i]), 1.0f);
        sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(weight[i], xval[i]), q));
        sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(weight[i], q), q));
    }
    if (sumq2 > 0.0f) {
        scale = __fdiv_rn(sumqx, sumq2);
    }
    return scale;
}

// Scale<0 flip: full ~ (IQ3_S/IQ2_S) vs masked ~&127 (parity types).
template<int K>
static __device__ void flip_signs_full_device(float * scale, uint8_t * bs) {
    *scale = -*scale;
    for (int k = 0; k < K; ++k) bs[k] = (uint8_t)(~bs[k]);
}

template<int K>
static __device__ void flip_signs_masked_device(float * scale, uint8_t * bs) {
    *scale = -*scale;
    for (int k = 0; k < K; ++k) bs[k] = (uint8_t)((~bs[k]) & 127);
}

// Finalize: d=max/31*fudge, id=1/d (RN, bit-twiddle FP16 for NaN).
static __device__ float finalize_d_device(float max_scale, float fudge, float * id) {
    const float d = __fdiv_rn(max_scale, 31.0f);
    *id = __fdiv_rn(1.0f, d);
    return d;
}

// Scales nibble pack (0..15) like CPU.
static __device__ int pack_scale_nibble_device(float id, float s) {
    int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, s), 1.0f)));
    return l < 0 ? 0 : (l > 15 ? 15 : l);
}

// On-device superblock sigma: 2x (S/XXXS/S) vs 1x (XS/XXS), exact powers of 2.
static __device__ float sigma2_2x_device(const float * xbl) {
    float sum = 0.0f;
    for (int j = 0; j < QK_K; ++j) sum = __fadd_rn(sum, __fmul_rn(xbl[j], xbl[j]));
    return __fmul_rn(sum, 2.0f/(float)QK_K);
}

static __device__ float sigma2_1x_device(const float * xbl) {
    float sum = 0.0f;
    for (int j = 0; j < QK_K; ++j) sum = __fadd_rn(sum, __fmul_rn(xbl[j], xbl[j]));
    return __fdiv_rn(sum, (float)QK_K);
}

// Device table upload, shared by iq2/iq3 getters.
static void cuda_upload_table(const void * host, size_t nbytes, void ** dev) {
    CUDA_CHECK(cudaMalloc(dev, nbytes));
    CUDA_CHECK(cudaMemcpy(*dev, host, nbytes, cudaMemcpyHostToDevice));
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

// --- IQ4_NL (ntry=7, w=x*x) ---

static __device__ const int8_t kvalues_iq4nl_dev[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// LUT from ggml-quants.c:14823; >=16 names boundary pair (ix-16, ix-15).
static __device__ const int kIq4nlIndex[241] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 17, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 18, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 19, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 21, 21, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 23, 23, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 24, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15
};

static __device__ int best_index_iq4nl_device(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = kIq4nlIndex[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

// --- IQ4 shared block optimizer (ntry=7 grid+hill-climb, exact CPU order) ---
static __device__ float iq4nl_opt_block_device(const float * xb, const float * weight, uint8_t * L) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax; max = xb[j];
        }
    }
    if (amax < 1e-15f) {
        return 0.0f;
    }
    float d = __fdiv_rn(-max, (float)kvalues_iq4nl_dev[0]);
    const float id0 = __fdiv_rn(1.0f, d);
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id0, xb[j]));
        L[j] = (uint8_t)l;
        const float q = (float)kvalues_iq4nl_dev[l];
        const float w = weight[j];
        sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
        sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
    }
    d = __fdiv_rn(sumqx, sumq2);
    float best = __fmul_rn(d, sumqx);
    float best_sumqx = sumqx, best_sumq2 = sumq2;
    for (int itry = -7; itry <= 7; ++itry) {
        float id = __fdiv_rn((float)(itry + kvalues_iq4nl_dev[0]), max);
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            const float q = (float)kvalues_iq4nl_dev[l];
            const float w = weight[j];
            sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
            sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
        }
        if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            }
        }
        id = __fdiv_rn((float)(itry + kvalues_iq4nl_dev[15]), max);
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            const float q = (float)kvalues_iq4nl_dev[l];
            const float w = weight[j];
            sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
            sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
        }
        if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            }
        }
    }
    sumqx = best_sumqx; sumq2 = best_sumq2;
    for (int iter = 0; iter < 32*32; ++iter) {
        float min_step = INFINITY;
        int best_j = -1, dir = 0;
        for (int j = 0; j < 32; ++j) {
            const float w = weight[j];
            const float g = __fmul_rn(__fmul_rn(d, w), __fsub_rn(xb[j], __fmul_rn(d, (float)kvalues_iq4nl_dev[L[j]])));
            if (g > 0.0f && L[j] < 15) {
                const float step = __fdiv_rn((float)(kvalues_iq4nl_dev[L[j]+1] - kvalues_iq4nl_dev[L[j]]), g);
                if (step < min_step) {
                    min_step = step; best_j = j; dir = 1;
                }
            }
            else if (g < 0.0f && L[j] > 0) {
                const float step = __fdiv_rn((float)(kvalues_iq4nl_dev[L[j]-1] - kvalues_iq4nl_dev[L[j]]), g);
                if (step < min_step) {
                    min_step = step; best_j = j; dir = -1;
                }
            }
        }
        if (best_j < 0) break;

        const float w = weight[best_j];
        const int l0 = L[best_j];
        const int l1 = l0 + dir;
        const float dq = (float)(kvalues_iq4nl_dev[l1] - kvalues_iq4nl_dev[l0]);
        float new_sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, xb[best_j]), dq));
        const int q1sq = kvalues_iq4nl_dev[l1]*kvalues_iq4nl_dev[l1];
        const int q0sq = kvalues_iq4nl_dev[l0]*kvalues_iq4nl_dev[l0];
        float new_sumq2 = __fadd_rn(sumq2, __fmul_rn(w, (float)(q1sq - q0sq)));
        if (new_sumq2 > 0.0f && __fmul_rn(new_sumqx, new_sumqx) > __fmul_rn(best, new_sumq2)) {
            sumqx = new_sumqx; sumq2 = new_sumq2;
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            L[best_j] = (uint8_t)l1;
        }
        else {
            break;
        }
    }
    return d;
}

static __device__ void iq4nl_requant_pack_device(const float * xb, float scale, uint8_t * L, uint8_t * qs);

static __global__ void quantize_iq4_nl_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const float * xb = x + ib*QK4_NL;

    float weight[QK4_NL];
    uint8_t L[QK4_NL];
    for (int j = 0; j < QK4_NL; ++j) {
        weight[j] = __fmul_rn(xb[j], xb[j]);
    }

    const float scale = iq4nl_opt_block_device(xb, weight, L);

    // Finalize: re-quant with id=0 when scale=0; bit-twiddle FP16 for NaN payload.
    // dh = FP16(fudge*scale) like the CPU (fudge defaults to 1 for IQ4_NL).
    block_iq4_nl * y = (block_iq4_nl *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, scale)));
    iq4nl_requant_pack_device(xb, scale, L, y[ib].qs);
}

// Final re-quant with the fitted scale + nibble pack, shared by NL kernels.
static __device__ void iq4nl_requant_pack_device(const float * xb, float scale, uint8_t * L, uint8_t * qs) {
    const float idf = scale ? __fdiv_rn(1.0f, scale) : 0.0f;
    for (int j = 0; j < 32; ++j) {
        L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idf, xb[j]));
    }
    pack_q4_nibbles_device(L, qs);
}

// --- IQ4_NL imatrix (same optimizer, qw*sqrt(sigma2+x*x)) ---
static __global__ void quantize_iq4_nl_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_NL;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_NL;
    const float s2   = sigma2[gb];

    float weight[QK4_NL];
    uint8_t L[QK4_NL];
    imatrix_weights_device(xb, qb, s2, weight, QK4_NL);

    const float scale = iq4nl_opt_block_device(xb, weight, L);

    // dh = FP16(fudge*scale) like the CPU (bit-twiddle for NaN payload).
    block_iq4_nl * y = (block_iq4_nl *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, scale)));
    iq4nl_requant_pack_device(xb, scale, L, y[ib].qs);
}

// --- IQ4_XS (super_block=256, ntry=7) ---

// XS global scale fit + re-quant + pack, shared by plain/imatrix kernels.
// dh = FP16(fudge*gd) like the CPU (fudge defaults to 1 for IQ4_XS).
static __device__ void iq4xs_finalize_device(const float * xs, float max_scale, float * scales, uint8_t * L,
        block_iq4_xs * y, int64_t sb, float fudge) {
    const float gd = __fmul_rn(fudge, -max_scale/32.0f);
    y[sb].d = __ushort_as_half(fp32_to_fp16_ggml(gd));
    const float gid = gd ? __fdiv_rn(1.0f, gd) : 0.0f;
    uint16_t scales_h = 0;
    for (int ib = 0; ib < 8; ++ib) {
        // Pin l=0 for non-finite scales (deterministic degenerate path, see ggml-quants.c).
        const bool finite_scale = fabsf(scales[ib]) <= FLT_MAX;
        int l = finite_scale ? nearest_int_device(__fmul_rn(gid, scales[ib])) : 0;
        l = l > 31 ? 31 : (l < -32 ? -32 : l);
        const float dl = __fmul_rn(gd, (float)l);
        const float idl = dl ? __fdiv_rn(1.0f, dl) : 0.0f;
        uint8_t * Lb = L + ib*32;
        const float * xb = xs + ib*32;
        for (int j = 0; j < 32; ++j) {
            Lb[j] = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idl, xb[j]));
        }
        l += 32;
        const uint8_t l_l = (uint8_t)(l & 0xf);
        const uint8_t l_h = (uint8_t)((unsigned)l >> 4);
        if (ib % 2 == 0) {
            y[sb].scales_l[ib/2] = l_l;
        } else {
            y[sb].scales_l[ib/2] |= (uint8_t)(l_l << 4);
        }
        scales_h |= (uint16_t)(l_h << (2*(ib % 8)));
    }
    y[sb].scales_h = scales_h;
    for (int i = 0; i < QK_K/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y[sb].qs[16*i + j] = (uint8_t)(L[32*i + j] | (L[32*i + 16 + j] << 4));
        }
    }
}

static __global__ void quantize_iq4_xs_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xs = x + sb*QK_K;

    float weight[32];
    uint8_t L[QK_K];
    float scales[8];
    float max_scale = 0.0f, amax_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xs + ib*32;
        uint8_t * Lb = L + ib*32;
        for (int j = 0; j < 32; ++j) {
            weight[j] = __fmul_rn(xb[j], xb[j]);
        }
        // Shared optimizer (eps 0 is no-op, matches CPU continue).
        const float d = iq4nl_opt_block_device(xb, weight, Lb);
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    // Global scale + re-quant (CPU order, dh = FP16(fudge*gd)).
    block_iq4_xs * y = (block_iq4_xs *)vy;
    iq4xs_finalize_device(xs, max_scale, scales, L, y, sb, fudge);
}

// --- IQ4_XS imatrix (same replay, qw*sqrt(sigma2+x*x)) ---
static __global__ void quantize_iq4_xs_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t sb_per_row,
        const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % sb_per_row)*QK_K;
    const float s2   = sigma2[gb];

    float weight[32];
    uint8_t L[QK_K];
    float scales[8];
    float max_scale = 0.0f, amax_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xs + ib*32;
        const float * qb = qs + ib*32;
        uint8_t * Lb = L + ib*32;
        imatrix_weights_device(xb, qb, s2, weight, 32);
        const float d = iq4nl_opt_block_device(xb, weight, Lb);
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    block_iq4_xs * y = (block_iq4_xs *)vy;
    iq4xs_finalize_device(xs, max_scale, scales, L, y, sb, fudge);
}

// --- IQ3 tables (256 XXS / 512 S, runtime-built; cached per device like KT codebook) ---
struct iq3_cuda_tables {
    const uint32_t * grid = nullptr;
    const int * map = nullptr;
    const uint16_t * neighbours = nullptr;
    int grid_n = 0;
    int neighbours_n = 0;
};

static iq3_cuda_tables iq3_get_tables(int device, int grid_size) {
    static std::mutex mutex;
    static iq3_cuda_tables tables[GGML_CUDA_MAX_DEVICES][2];
    static bool ready[GGML_CUDA_MAX_DEVICES][2] = {};
    const int gi = grid_size == 256 ? 0 : 1;
    std::lock_guard<std::mutex> lock(mutex);
    if (!ready[device][gi]) {
        const uint32_t * grid = nullptr;
        const int * map = nullptr;
        const uint16_t * neighbours = nullptr;
        int grid_n = 0, map_n = 0, neighbours_n = 0;
        // Ensure host tables built (ggml_quantize_init) via accessor.
        if (iq3xs_grid_data(grid_size, &grid, &map, &neighbours, &grid_n, &map_n, &neighbours_n) != 0) {
            fprintf(stderr, "%s: iq3xs_grid_data(%d) failed\n", __func__, grid_size);
            return tables[device][gi];
        }
        uint32_t * d_grid = nullptr;
        int * d_map = nullptr;
        uint16_t * d_neighbours = nullptr;
        cuda_upload_table(grid, grid_n*sizeof(uint32_t), (void **)&d_grid);
        cuda_upload_table(map, 4096*sizeof(int), (void **)&d_map);
        cuda_upload_table(neighbours, neighbours_n*sizeof(uint16_t), (void **)&d_neighbours);
        tables[device][gi].grid = d_grid;
        tables[device][gi].map = d_map;
        tables[device][gi].neighbours = d_neighbours;
        tables[device][gi].grid_n = grid_n;
        tables[device][gi].neighbours_n = neighbours_n;
        ready[device][gi] = true;
    }
    return tables[device][gi];
}

// --- IQ3_S (block 32, QK 256, fudge 1.033; plain w=x*x, imatrix qw*sqrt; qs advances only on processed groups) ---
static __device__ void iq3s_superblock_device(const float * xbl, const float * qw_bl, float sigma2, bool has_imatrix,
        const uint32_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq3_s * y) {
    // Zero like CPU memset + d=0.
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    for (int i = 0; i < 64; ++i) y->qs[i] = 0;
    for (int i = 0; i < 8; ++i) y->qh[i] = 0;
    for (int i = 0; i < 32; ++i) y->signs[i] = 0;
    for (int i = 0; i < 4; ++i) y->scales[i] = 0;

    const int kMaxQ = 8;
    float scales[8];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];
    float max_scale = 0.0f;
    int qs_off = 0;
    int signs_off = 0;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xbl + 32*ib;
        const float * qb = has_imatrix ? qw_bl + 32*ib : nullptr;
        iq3_weights_device<32>(xb, qb, sigma2, has_imatrix, weight, waux);
        extract_signs_direct_device<4>(xb, xval, block_signs);
        const float max = max_abs_device<32>(xval);
        if (max == 0.0f) {
            scales[ib] = 0.0f;
            continue;
        }
        float best = 0.0f;
        float scale = __fdiv_rn(max, (float)(2*kMaxQ - 1));
        for (int k = 0; k < 8; ++k) is_on_grid[k] = false;
        for (int is = -9; is <= 9; ++is) {
            const float id = __fdiv_rn(__fadd_rn((float)(2*kMaxQ - 1), __fmul_rn((float)is, 0.2f)), max);
            const float this_scale = __fdiv_rn(1.0f, id);
            for (int k = 0; k < 8; ++k) {
                quantize_subgroup_device<4, 3>(id, this_scale, xval + 4*k, waux + 4*k, Laux + 4*k,
                        kmap, neighbours, grid, &is_on_grid_aux[k], kMaxQ);
            }
            try_update_scale_device<32>(weight, xval, Laux, &scale, &best, L, is_on_grid_aux, is_on_grid, 8);
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0.0f) {
            const float id = __fdiv_rn(1.0f, scale);
            for (int k = 0; k < 8; ++k) {
                requantize_subgroup_device<4, 3>(id, scale, xval + 4*k, waux + 4*k, L + 4*k,
                        kmap, neighbours, grid, kMaxQ);
            }
            scale = refit_scale_device<32>(weight, xval, L, scale);
        }
        if (scale < 0.0f) {
            flip_signs_full_device<4>(&scale, block_signs);
        }
        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (uint16_t)(L[4*k + i] << 3*i);
            const int grid_index = kmap[u];
            y->qs[qs_off + k] = (uint8_t)(grid_index & 255);
            const int qh_idx = (ib*8 + k)/8;
            const int qh_sh = (ib*8 + k)%8;
            y->qh[qh_idx] |= (uint8_t)(((grid_index >> 8) & 1) << qh_sh);
        }
        qs_off += 8;
        for (int k = 0; k < 4; ++k) y->signs[signs_off + k] = block_signs[k];
        signs_off += 4;
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
    }

    if (max_scale == 0.0f) {
        return;
    }

    float id;
    const float d = finalize_d_device(max_scale, fudge, &id);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    for (int ib = 0; ib < 8; ib += 2) {
        const int l1 = pack_scale_nibble_device(id, scales[ib]);
        const int l2 = pack_scale_nibble_device(id, scales[ib + 1]);
        y->scales[ib/2] = (uint8_t)(l1 | (l2 << 4));
    }
}

static __global__ void quantize_iq3_s_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint32_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xs = x + sb*QK_K;
    block_iq3_s * y = (block_iq3_s *)vy;
    iq3s_superblock_device(xs, nullptr, 0.0f, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq3_s_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge, const uint32_t * __restrict__ grid, const int * __restrict__ kmap,
        const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % blocks_per_row)*QK_K;
    // Device sigma like CPU order (RN, no FMA); matches plain path, avoids host FMA mismatch.
    const float s2 = sigma2_2x_device(xs);
    block_iq3_s * y = (block_iq3_s *)vy;
    iq3s_superblock_device(xs, qs, s2, true, grid, kmap, neighbours, fudge, y + sb);
}

// --- IQ3_XXS (grid 256, block 32, fudge 1.0125; parity, sweep -15..15, 2nd pass skips on-grid) ---
static __device__ void iq3xxs_superblock_device(const float * xbl, const float * qw_bl, float sigma2, bool has_imatrix,
        const uint32_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq3_xxs * y) {
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    uint8_t q3[96];
    for (int i = 0; i < 96; ++i) q3[i] = 0;
    uint32_t * scales_and_signs = (uint32_t *)(q3 + 64);

    const int kMaxQ = 8;
    float scales[8];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];
    float max_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xbl + 32*ib;
        const float * qb = has_imatrix ? qw_bl + 32*ib : nullptr;
        iq3_weights_device<32>(xb, qb, sigma2, has_imatrix, weight, waux);
        extract_signs_parity_device<4>(xb, weight, xval, block_signs);
        const float max = max_abs_device<32>(xval);
        if (max < 1e-8f) {
            scales[ib] = 0.0f;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }
        float best = 0.0f;
        float scale = __fdiv_rn(max, (float)(2*kMaxQ - 1));
        for (int is = -15; is <= 15; ++is) {
            const float id = __fdiv_rn(__fadd_rn((float)(2*kMaxQ - 1), __fmul_rn((float)is, 0.2f)), max);
            const float this_scale = __fdiv_rn(1.0f, id);
            for (int k = 0; k < 8; ++k) {
                quantize_subgroup_device<4, 3>(id, this_scale, xval + 4*k, waux + 4*k, Laux + 4*k,
                        kmap, neighbours, grid, &is_on_grid_aux[k], kMaxQ);
            }
            try_update_scale_device<32>(weight, xval, Laux, &scale, &best, L, is_on_grid_aux, is_on_grid, 8);
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0.0f) {
            const float id = __fdiv_rn(1.0f, scale);
            for (int k = 0; k < 8; ++k) {
                if (is_on_grid[k]) continue;
                requantize_subgroup_device<4, 3>(id, scale, xval + 4*k, waux + 4*k, L + 4*k,
                        kmap, neighbours, grid, kMaxQ);
            }
            scale = refit_scale_device<32>(weight, xval, L, scale);
        }
        if (scale < 0.0f) {
            flip_signs_masked_device<4>(&scale, block_signs);
        }
        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (uint16_t)(L[4*k + i] << 3*i);
            const int grid_index = kmap[u];
            q3[8*ib + k] = (uint8_t)grid_index;
        }
        scales_and_signs[ib] = (uint32_t)block_signs[0] | ((uint32_t)block_signs[1] << 7) |
                ((uint32_t)block_signs[2] << 14) | ((uint32_t)block_signs[3] << 21);
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
    }

    if (max_scale == 0.0f) {
        for (int i = 0; i < 96; ++i) y->qs[i] = 0;
        return;
    }

    float id;
    const float d = finalize_d_device(max_scale, fudge, &id);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    for (int ib = 0; ib < 8; ++ib) {
        scales_and_signs[ib] |= ((uint32_t)pack_scale_nibble_device(id, scales[ib]) << 28);
    }
    for (int i = 0; i < 96; ++i) y->qs[i] = q3[i];
}

static __global__ void quantize_iq3_xxs_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint32_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xs = x + sb*QK_K;
    block_iq3_xxs * y = (block_iq3_xxs *)vy;
    iq3xxs_superblock_device(xs, nullptr, 0.0f, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq3_xxs_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge, const uint32_t * __restrict__ grid, const int * __restrict__ kmap,
        const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % blocks_per_row)*QK_K;
    const float s2 = sigma2_2x_device(xs);
    block_iq3_xxs * y = (block_iq3_xxs *)vy;
    iq3xxs_superblock_device(xs, qs, s2, true, grid, kmap, neighbours, fudge, y + sb);
}

// --- IQ2 tables (XXS 256 / XS 512 / S 1024, runtime-built; cached per device) ---
struct iq2_cuda_tables {
    const uint64_t * grid = nullptr;
    const int * map = nullptr;
    const uint16_t * neighbours = nullptr;
    int grid_n = 0;
    int neighbours_n = 0;
};

static iq2_cuda_tables iq2_get_tables(int device, int type) {
    static std::mutex mutex;
    // gi: 0 XXS, 1 XS, 3 S (match iq2_data_index; 2 unused IQ1).
    static iq2_cuda_tables tables[GGML_CUDA_MAX_DEVICES][4];
    static bool ready[GGML_CUDA_MAX_DEVICES][4] = {};
    int gi = type == GGML_TYPE_IQ2_XXS ? 0 : type == GGML_TYPE_IQ2_XS ? 1 : 3;
    std::lock_guard<std::mutex> lock(mutex);
    if (!ready[device][gi]) {
        const uint64_t * grid = nullptr;
        const int * map = nullptr;
        const uint16_t * neighbours = nullptr;
        int grid_n = 0, map_n = 0, neighbours_n = 0;
        if (iq2xs_grid_data(type, &grid, &map, &neighbours, &grid_n, &map_n, &neighbours_n) != 0) {
            fprintf(stderr, "%s: iq2xs_grid_data(%d) failed\n", __func__, type);
            return tables[device][gi];
        }
        uint64_t * d_grid = nullptr;
        int * d_map = nullptr;
        uint16_t * d_neighbours = nullptr;
        cuda_upload_table(grid, grid_n*sizeof(uint64_t), (void **)&d_grid);
        cuda_upload_table(map, map_n*sizeof(int), (void **)&d_map);
        cuda_upload_table(neighbours, neighbours_n*sizeof(uint16_t), (void **)&d_neighbours);
        tables[device][gi].grid = d_grid;
        tables[device][gi].map = d_map;
        tables[device][gi].neighbours = d_neighbours;
        tables[device][gi].grid_n = grid_n;
        tables[device][gi].neighbours_n = neighbours_n;
        ready[device][gi] = true;
    }
    return tables[device][gi];
}

// --- IQ2_S (group 16, QK 256, fudge 0.9875; direct signs, sweep -9..9, 2nd pass skips on-grid) ---
static __device__ void iq2s_superblock_device(const float * xbl, const float * qw_bl, float sigma2, bool has_imatrix,
        const uint64_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq2_s * y) {
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    for (int i = 0; i < 64; ++i) y->qs[i] = 0;
    for (int i = 0; i < 8; ++i) y->qh[i] = 0;
    for (int i = 0; i < 8; ++i) y->scales[i] = 0;

    const int kMaxQ = 3;
    float scales[16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];
    float max_scale = 0.0f;

    for (int ib = 0; ib < 16; ++ib) {
        const float * xb = xbl + 16*ib;
        const float * qb = has_imatrix ? qw_bl + 16*ib : nullptr;
        iq2_weights_device<16>(xb, qb, sigma2, has_imatrix, weight, waux);
        extract_signs_direct_device<2>(xb, xval, block_signs);
        const float max = max_abs_device<16>(xval);
        if (max < 1e-8f) {
            scales[ib] = 0.0f;
            continue;
        }
        float best = 0.0f;
        float scale = __fdiv_rn(max, (float)(2*kMaxQ - 1));
        is_on_grid[0] = is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            const float id = __fdiv_rn(__fadd_rn((float)(2*kMaxQ - 1), __fmul_rn((float)is, 0.1f)), max);
            const float this_scale = __fdiv_rn(1.0f, id);
            for (int k = 0; k < 2; ++k) {
                quantize_subgroup_device<8, 2>(id, this_scale, xval + 8*k, waux + 8*k, Laux + 8*k,
                        kmap, neighbours, grid, &is_on_grid_aux[k], kMaxQ);
            }
            try_update_scale_device<16>(weight, xval, Laux, &scale, &best, L, is_on_grid_aux, is_on_grid, 2);
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0.0f) {
            const float id = __fdiv_rn(1.0f, scale);
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, xval[8*k + i]), 1.0f)));
                    l = l < 0 ? 0 : (l > kMaxQ - 1 ? kMaxQ - 1 : l);
                    u |= (uint16_t)(l << 2*i);
                    L[8*k + i] = (int8_t)l;
                }
                const int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * nb = neighbours - grid_index - 1;
                    find_best_neighbour_device<8>(nb, grid, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
            }
            scale = refit_scale_device<16>(weight, xval, L, scale);
        }
        if (scale < 0.0f) {
            flip_signs_full_device<2>(&scale, block_signs);
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8*k + i] << 2*i);
            const int grid_index = kmap[u];
            const int i8 = 2*ib + k;
            y->qs[i8] = (uint8_t)(grid_index & 255);
            y->qh[i8/4] |= (uint8_t)(((grid_index >> 8) & 3) << 2*(i8 % 4));
            y->qs[32 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
    }

    if (max_scale == 0.0f) {
        return;
    }

    float id;
    const float d = finalize_d_device(max_scale, fudge, &id);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    for (int ib = 0; ib < 16; ++ib) {
        const int l = pack_scale_nibble_device(id, scales[ib]);
        if (ib % 2 == 0) {
            y->scales[ib/2] = (uint8_t)l;
        } else {
            y->scales[ib/2] |= (uint8_t)(l << 4);
        }
    }
}

static __global__ void quantize_iq2_s_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint64_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xbl = x + sb*QK_K;
    const float sigma2 = sigma2_2x_device(xbl);
    block_iq2_s * y = (block_iq2_s *)vy;
    iq2s_superblock_device(xbl, nullptr, sigma2, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq2_s_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge, const uint64_t * __restrict__ grid, const int * __restrict__ kmap,
        const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % blocks_per_row)*QK_K;
    const float s2 = sigma2_2x_device(xs);
    block_iq2_s * y = (block_iq2_s *)vy;
    iq2s_superblock_device(xs, qs, s2, true, grid, kmap, neighbours, fudge, y + sb);
}

// --- IQ2_XS (group 16, grid 512, fudge 1.05; parity, qsort init, 3-iter refine + LS refit) ---
// Mirrors iq2_xs_impl; stable insertion sort (index tiebreak = iq1_sort_helper); sigma sum/256 both paths.
static __device__ void iq2xs_superblock_device(const float * xbl, const float * qw_bl, float sigma2, bool has_imatrix,
        const uint64_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq2_xs * y) {
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    for (int i = 0; i < 32; ++i) y->qs[i] = 0;
    for (int i = 0; i < 8; ++i) y->scales[i] = 0;

    const int kMaxQ = 3;
    float scales[16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[32];
    uint16_t index[2], aux_index[2];
    float sumx[17], sumw[17];
    float pairs[32];
    int order[16];
    for (int i = 0; i < 32; ++i) q2[i] = 0;
    float max_scale = 0.0f;

    for (int ib = 0; ib < 16; ++ib) {
        const float * xb = xbl + 16*ib;
        const float * qb = has_imatrix ? qw_bl + 16*ib : nullptr;
        iq2_weights_device<16>(xb, qb, sigma2, has_imatrix, weight, waux);
        extract_signs_parity_device<2>(xb, weight, xval, block_signs);
        const float max = max_abs_device<16>(xval);
        if (max < 1e-15f) {
            scales[ib] = 0.0f;
            for (int i = 0; i < 16; ++i) L[i] = 0;
            continue;
        }
        for (int j = 0; j < 16; ++j) {
            pairs[2*j] = xval[j];
            order[j] = j;
        }
        // Stable insertion sort by xval with index tiebreak: matches the CPU
        // qsort (iq1_sort_helper breaks exact ties by original index).
        for (int j = 1; j < 16; ++j) {
            const float v = pairs[2*j];
            const int idx = order[j];
            int k = j - 1;
            while (k >= 0 && (pairs[2*k] > v || (pairs[2*k] == v && order[k] > idx))) {
                pairs[2*(k + 1)] = pairs[2*k];
                order[k + 1] = order[k];
                --k;
            }
            pairs[2*(k + 1)] = v;
            order[k + 1] = idx;
        }
        sumx[0] = 0.0f;
        sumw[0] = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int i = order[j];
            sumx[j + 1] = __fadd_rn(sumx[j], __fmul_rn(weight[i], xval[i]));
            sumw[j + 1] = __fadd_rn(sumw[j], weight[i]);
        }
        float best = 0.0f, scale = 0.0f;
        for (int i1 = 0; i1 <= 16; ++i1) {
            for (int i2 = i1; i2 <= 16; ++i2) {
                const float sumqx = __fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(sumx[i1], sumx[0]), 1.0f),
                        __fmul_rn(__fsub_rn(sumx[i2], sumx[i1]), 3.0f)),
                        __fmul_rn(__fsub_rn(sumx[16], sumx[i2]), 5.0f));
                const float sumq2 = __fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(sumw[i1], sumw[0]), 1.0f),
                        __fmul_rn(__fsub_rn(sumw[i2], sumw[i1]), 9.0f)),
                        __fmul_rn(__fsub_rn(sumw[16], sumw[i2]), 25.0f));
                if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
                    scale = __fdiv_rn(sumqx, sumq2);
                    best = __fmul_rn(scale, sumqx);
                }
            }
        }
        best = 0.0f;
        const float eff_max = __fmul_rn(scale, (float)(2*kMaxQ - 1));
        is_on_grid[0] = is_on_grid[1] = true;
        index[0] = index[1] = 0;
        for (int is = -7; is <= 7; ++is) {
            const float id = __fdiv_rn(__fadd_rn((float)(2*kMaxQ - 1), __fmul_rn((float)is, 0.1f)), eff_max);
            const float this_scale = __fdiv_rn(1.0f, id);
            for (int k = 0; k < 2; ++k) {
                aux_index[k] = (uint16_t)quantize_subgroup_device<8, 2>(id, this_scale, xval + 8*k, waux + 8*k,
                        Laux + 8*k, kmap, neighbours, grid, &is_on_grid_aux[k], kMaxQ);
            }
            if (try_update_scale_device<16>(weight, xval, Laux, &scale, &best, L, is_on_grid_aux, is_on_grid, 2)) {
                for (int k = 0; k < 2; ++k) index[k] = aux_index[k];
            }
        }
        if (scale > 0.0f) {
            for (int iter = 0; iter < 3; ++iter) {
                const float id = __fdiv_rn(1.0f, scale);
                bool changed = false;
                bool dummy;
                for (int k = 0; k < 2; ++k) {
                    aux_index[k] = (uint16_t)quantize_subgroup_device<8, 2>(id, scale, xval + 8*k, waux + 8*k,
                            Laux + 8*k, kmap, neighbours, grid, &dummy, kMaxQ);
                    if ((int)aux_index[k] != (int)index[k]) changed = true;
                }
                if (!changed) break;
                if (try_update_scale_device<16>(weight, xval, Laux, &scale, &best, L, is_on_grid, is_on_grid, 2)) {
                    for (int k = 0; k < 2; ++k) index[k] = aux_index[k];
                } else {
                    break;
                }
            }
        }
        if (scale < 0.0f) {
            flip_signs_masked_device<2>(&scale, block_signs);
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8*k + i] << 2*i);
            const int grid_index = kmap[u];
            q2[2*ib + k] = (uint16_t)grid_index | (uint16_t)((uint16_t)block_signs[k] << 9);
        }
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
    }

    if (max_scale == 0.0f) {
        for (int i = 0; i < 32; ++i) y->qs[i] = 0;
        return;
    }

    float id;
    const float d = finalize_d_device(max_scale, fudge, &id);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int ib = 0; ib < 16; ++ib) {
        int l = pack_scale_nibble_device(id, scales[ib]);
        if (ib % 2 == 0) {
            y->scales[ib/2] = (uint8_t)l;
        } else {
            y->scales[ib/2] |= (uint8_t)(l << 4);
        }
        l = 2*l + 1;
        const float * xb = xbl + 16*ib;
        float weight[16];
        if (has_imatrix) {
            const float * qb = qw_bl + 16*ib;
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))));
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fadd_rn(__fmul_rn(0.25f, sigma2), __fmul_rn(xb[i], xb[i]));
            }
        }
        // Superblock sigma (sum/256); weights recomputed per ib like CPU second pass.
        for (int k = 0; k < 2; ++k) {
            const int grid_index = q2[2*ib + k] & 511;
            const uint8_t * grid8 = (const uint8_t *)(iq2xs_grid + grid_index);
            const uint8_t signs = ksigns_iq2xs[q2[2*ib + k] >> 9];
            for (int j = 0; j < 8; ++j) {
                const float w = weight[8*k + j];
                const float q = __fmul_rn(__fmul_rn(__fmul_rn(0.125f, (float)l), (float)grid8[j]),
                        (signs & kmask_iq2xs[j] ? -1.0f : 1.0f));
                sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[8*k + j]));
                sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
            }
        }
    }
    // q2->qs like CPU memcpy (both uint16_t[32], 64 bytes).
    for (int i = 0; i < 32; ++i) {
        y->qs[i] = q2[i];
    }
    if (sumq2 > 0.0f) {
        y->d = __ushort_as_half(fp32_to_fp16_ggml(__fdiv_rn(__fmul_rn(fudge, sumqx), sumq2)));
    }
}

static __global__ void quantize_iq2_xs_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint64_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xbl = x + sb*QK_K;
    const float sigma2 = sigma2_1x_device(xbl);
    block_iq2_xs * y = (block_iq2_xs *)vy;
    iq2xs_superblock_device(xbl, nullptr, sigma2, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq2_xs_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge, const uint64_t * __restrict__ grid, const int * __restrict__ kmap,
        const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % blocks_per_row)*QK_K;
    const float s2 = sigma2_1x_device(xs);
    block_iq2_xs * y = (block_iq2_xs *)vy;
    iq2xs_superblock_device(xs, qs, s2, true, grid, kmap, neighbours, fudge, y + sb);
}

// make_qp_quants port (ggml-quants.c:2361); uint8 wrap + MIN-only clamps kept bit-exact.
static __device__ float make_qp_quants_device(int n, int nmax, const float * x, uint8_t * L, const float * qw) {
    float max = 0.0f;
    for (int i = 0; i < n; ++i) {
        max = max > x[i] ? max : x[i];
    }
    if (max < 1e-16f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn((float)nmax, max);
    for (int i = 0; i < n; ++i) {
        L[i] = (uint8_t)nearest_int_device(__fmul_rn(iscale, x[i]));
    }
    float scale = __fdiv_rn(1.0f, iscale);
    float best_mse = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float diff = __fsub_rn(x[i], __fmul_rn(scale, (float)L[i]));
        const float w = qw[i];
        best_mse = __fadd_rn(best_mse, __fmul_rn(__fmul_rn(w, diff), diff));
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        const float iscale_is = __fdiv_rn(__fadd_rn(__fmul_rn(0.1f, (float)is), (float)nmax), max);
        const float scale_is = __fdiv_rn(1.0f, iscale_is);
        float mse = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale_is, x[i]));
            l = l > nmax ? nmax : l;
            const float diff = __fsub_rn(x[i], __fmul_rn(scale_is, (float)l));
            const float w = qw[i];
            mse = __fadd_rn(mse, __fmul_rn(__fmul_rn(w, diff), diff));
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = l > nmax ? nmax : l;
        L[i] = (uint8_t)l;
        const float w = qw[i];
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = qw[i];
            const float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)L[i]));
            const float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, (float)L[i]), (float)L[i]));
            if (slx > 0.0f && sl2 > 0.0f) {
                int new_l = nearest_int_device(__fdiv_rn(__fmul_rn(x[i], sl2), slx));
                new_l = new_l > nmax ? nmax : new_l;
                if (new_l != (int)L[i]) {
                    const float nslx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), (float)new_l));
                    const float nsl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, (float)new_l), (float)new_l));
                    if (__fmul_rn(__fmul_rn(nslx, nslx), suml2) > __fmul_rn(__fmul_rn(sumlx, sumlx), nsl2)) {
                        L[i] = (uint8_t)new_l;
                        sumlx = nslx;
                        suml2 = nsl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return __fdiv_rn(sumlx, suml2);
}

// --- IQ2_XXS (group 32, grid 256, fudge 1.0; parity, sweep -6..6, make_qp init, 2nd pass refines all) ---
static __device__ void iq2xxs_superblock_device(const float * xbl, const float * qw_bl, float sigma2, bool has_imatrix,
        const uint64_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq2_xxs * y) {
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    uint32_t q2[16];
    for (int i = 0; i < 16; ++i) q2[i] = 0;

    const int kMaxQ = 3;
    float scales[8];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    uint8_t block_signs[4];
    float max_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xbl + 32*ib;
        const float * qb = has_imatrix ? qw_bl + 32*ib : nullptr;
        iq2_weights_device<32>(xb, qb, sigma2, has_imatrix, weight, waux);
        extract_signs_parity_device<4>(xb, weight, xval, block_signs);
        const float max = max_abs_device<32>(xval);
        if (max < 1e-15f) {
            scales[ib] = 0.0f;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }
        float scale = make_qp_quants_device(32, kMaxQ + 1, xval, (uint8_t *)L, weight);
        float eff_max = __fmul_rn(scale, (float)kMaxQ);
        float best = 0.0f;
        for (int is = -6; is <= 6; ++is) {
            const float id = __fdiv_rn(__fadd_rn((float)(2*kMaxQ - 1), __fmul_rn((float)is, 0.1f)), eff_max);
            const float this_scale = __fdiv_rn(1.0f, id);
            bool dummy;
            for (int k = 0; k < 4; ++k) {
                quantize_subgroup_device<8, 2>(id, this_scale, xval + 8*k, waux + 8*k, Laux + 8*k,
                        kmap, neighbours, grid, &dummy, kMaxQ);
            }
            try_update_scale_device<32>(weight, xval, Laux, &scale, &best, L, nullptr, nullptr, 0);
        }
        if (scale > 0.0f) {
            const float id = __fdiv_rn(1.0f, scale);
            for (int k = 0; k < 4; ++k) {
                requantize_subgroup_device<8, 2>(id, scale, xval + 8*k, waux + 8*k, L + 8*k,
                        kmap, neighbours, grid, kMaxQ);
            }
            scale = refit_scale_device<32>(weight, xval, L, scale);
        }
        if (scale < 0.0f) {
            flip_signs_masked_device<4>(&scale, block_signs);
        }
        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8*k + i] << 2*i);
            const int grid_index = kmap[u];
            q2[2*ib] |= ((uint32_t)grid_index << 8*k);
            q2[2*ib + 1] |= ((uint32_t)block_signs[k] << 7*k);
        }
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
    }

    if (max_scale == 0.0f) {
        for (int i = 0; i < 32; ++i) y->qs[i] = 0;
        return;
    }

    float id;
    const float d = finalize_d_device(max_scale, fudge, &id);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    for (int ib = 0; ib < 8; ++ib) {
        q2[2*ib + 1] |= ((uint32_t)pack_scale_nibble_device(id, scales[ib]) << 28);
    }
    // Byte-exact q2->qs (16 u32 LE = 64 bytes) like CPU memcpy; byte writes avoid u32 misalignment (qs at +2).
    uint8_t * qs8 = (uint8_t *)y->qs;
    for (int i = 0; i < 16; ++i) {
        qs8[4*i] = (uint8_t)(q2[i] & 255);
        qs8[4*i + 1] = (uint8_t)((q2[i] >> 8) & 255);
        qs8[4*i + 2] = (uint8_t)((q2[i] >> 16) & 255);
        qs8[4*i + 3] = (uint8_t)((q2[i] >> 24) & 255);
    }
}

static __global__ void quantize_iq2_xxs_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint64_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xbl = x + sb*QK_K;
    const float sigma2 = sigma2_1x_device(xbl);
    block_iq2_xxs * y = (block_iq2_xxs *)vy;
    iq2xxs_superblock_device(xbl, nullptr, sigma2, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq2_xxs_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge, const uint64_t * __restrict__ grid, const int * __restrict__ kmap,
        const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % blocks_per_row)*QK_K;
    const float s2 = sigma2_1x_device(xs);
    block_iq2_xxs * y = (block_iq2_xxs *)vy;
    iq2xxs_superblock_device(xs, qs, s2, true, grid, kmap, neighbours, fudge, y + sb);
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

// --- generic chunked host drivers (~128 MiB, shared) ---

using quantize_kernel_t = void (*)(const float *, void *, int64_t, float);

// Streaming chunk executor shared by the plain/OLS and imatrix drivers.
// Double-buffered overlap across chunks: while chunk i runs (kernel + D2H)
// on the compute stream, chunk i+1 uploads on the transfer stream and the
// host stages/copies through pinned memory. launch(dx, dy, base, nblocks, st)
// issues the type-specific kernel launch on the given stream. Falls back to
// synchronous copies when streams or pinned staging are unavailable.
// Returns bytes written, or 0 on failure (caller falls back to CPU).
template <typename LaunchFn>
static size_t quantize_exec_chunked(const float * src, void * dst, int64_t nblocks_total,
        int64_t qk, size_t blk_size, int64_t chunk_blocks, const char * name, LaunchFn launch) {
    const int64_t chunk_x = chunk_blocks*qk;
    const int64_t chunk_y = chunk_blocks*(int64_t)blk_size;

    float   * x_dev[2] = {nullptr, nullptr};
    uint8_t * y_dev[2] = {nullptr, nullptr};
    float   * x_pin[2] = {nullptr, nullptr};
    uint8_t * y_pin[2] = {nullptr, nullptr};
    cudaStream_t st_h2d = nullptr, st_exe = nullptr;
    cudaEvent_t ev_h2d[2] = {nullptr, nullptr}, ev_done[2] = {nullptr, nullptr};

    bool ok = true;
    cudaError_t err = cudaSuccess;
    // Single-chunk tensors gain nothing from pipelining; use the synchronous
    // path directly (also avoids pinned staging).
    if (nblocks_total <= chunk_blocks) {
        ok = false;
    }
    for (int b = 0; b < 2 && ok; ++b) {
        if ((err = cudaMalloc(&x_dev[b], chunk_x*sizeof(float))) != cudaSuccess) ok = false;
        if (ok && (err = cudaMalloc(&y_dev[b], chunk_y)) != cudaSuccess) ok = false;
        if (ok && (err = cudaMallocHost(&x_pin[b], chunk_x*sizeof(float))) != cudaSuccess) ok = false;
        if (ok && (err = cudaMallocHost(&y_pin[b], chunk_y)) != cudaSuccess) ok = false;
        if (ok && (err = cudaEventCreateWithFlags(&ev_h2d[b], cudaEventDisableTiming)) != cudaSuccess) ok = false;
        if (ok && (err = cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming)) != cudaSuccess) ok = false;
    }
    if (ok && (err = cudaStreamCreate(&st_h2d)) != cudaSuccess) ok = false;
    if (ok && (err = cudaStreamCreate(&st_exe)) != cudaSuccess) ok = false;

    auto release = [&]() {
        for (int b = 0; b < 2; ++b) {
            if (x_dev[b]) cudaFree(x_dev[b]);
            if (y_dev[b]) cudaFree(y_dev[b]);
            if (x_pin[b]) cudaFreeHost(x_pin[b]);
            if (y_pin[b]) cudaFreeHost(y_pin[b]);
            if (ev_h2d[b]) cudaEventDestroy(ev_h2d[b]);
            if (ev_done[b]) cudaEventDestroy(ev_done[b]);
        }
        if (st_h2d) cudaStreamDestroy(st_h2d);
        if (st_exe) cudaStreamDestroy(st_exe);
    };

    if (!ok) {
        // Synchronous fallback: single buffers, blocking copies (old behavior).
        release();
        float   * xs = nullptr;
        uint8_t * ys = nullptr;
        if ((err = cudaMalloc(&xs, chunk_x*sizeof(float))) != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMalloc(x_dev): %s\n", __func__, name, cudaGetErrorString(err));
            return 0;
        }
        if ((err = cudaMalloc(&ys, chunk_y)) != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMalloc(y_dev): %s\n", __func__, name, cudaGetErrorString(err));
            cudaFree(xs);
            return 0;
        }
        for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
            const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);
            err = cudaMemcpy(xs, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
            launch(xs, ys, base, nblocks, nullptr);
            if ((err = cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr, "%s: %s: kernel launch: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
            err = cudaMemcpy((char *)dst + base*blk_size, ys, nblocks*blk_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
        }
        cudaFree(xs);
        cudaFree(ys);
        if (err != cudaSuccess) {
            return 0;
        }
        return nblocks_total*blk_size;
    }

    for (int64_t base = 0, i = 0; base < nblocks_total; base += chunk_blocks, ++i) {
        const int b = (int)(i & 1);
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        // Buffer b must be fully done (H2D + kernel + D2H of iteration i-2)
        // before its staging and device buffers are reused.
        if (i >= 2) {
            err = cudaStreamWaitEvent(st_h2d, ev_done[b], 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: %s: stream wait: %s\n", __func__, name, cudaGetErrorString(err));
                break;
            }
        }
        memcpy(x_pin[b], src + base*qk, nblocks*qk*sizeof(float));
        err = cudaMemcpyAsync(x_dev[b], x_pin[b], nblocks*qk*sizeof(float), cudaMemcpyHostToDevice, st_h2d);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpyAsync H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventRecord(ev_h2d[b], st_h2d);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event record: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaStreamWaitEvent(st_exe, ev_h2d[b], 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: stream wait: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        launch(x_dev[b], y_dev[b], base, nblocks, st_exe);
        if ((err = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "%s: %s: kernel launch: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaMemcpyAsync(y_pin[b], y_dev[b], nblocks*blk_size, cudaMemcpyDeviceToHost, st_exe);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpyAsync D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventRecord(ev_done[b], st_exe);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event record: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        err = cudaEventSynchronize(ev_done[b]);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: event sync: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
        memcpy((char *)dst + base*blk_size, y_pin[b], nblocks*blk_size);
    }

    release();
    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

// Shared chunked driver for warp-per-block plain kernels (block_size = 0
// selects one warp per quant block) and one-thread-per-block OLS/codebook
// kernels (block_size = 256). ~128 MiB F32 chunks bound VRAM use.
static size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
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

using imatrix_kernel_t = void (*)(const float *, const float *, const float *, void *, int64_t, int64_t, int32_t, float);

// Shared chunked driver for all imatrix kernels. sigma covers sigma_len values
// per entry: a full row (sum/n_per_row, CPU row order) or one quant unit
// (sum*(2/qk), CPU block order); kernels index it per row or per block.
static size_t ggml_cuda_quantize_imatrix_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
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
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0); // match CPU quirk
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK8_0, sizeof(block_q8_0), quantize_q8_0_kernel, "q8_0", fudge, 0);
}


// --- Removable Q5_0 section (delete kernel+helpers+cases to drop) ---
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_0, sizeof(block_q5_0), quantize_q5_0_kernel, "q5_0", fudge, 0);
}

// --- Removable Q4_0 section (delete kernel+helpers+cases to drop) ---
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_0, sizeof(block_q4_0), quantize_q4_0_kernel, "q4_0", fudge, 0);
}

// --- Q4_1 (quantize_row_q4_1_ref, no fudge) ---
size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_1, sizeof(block_q4_1), quantize_q4_1_kernel, "q4_1", 1.0f, 0);
}

// --- Removable Q4_0 imatrix (reuses row imatrix like CPU) ---
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_0, sizeof(block_q4_0),
            quantize_q4_0_imatrix_kernel, "q4_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0), false);
}

// --- Removable Q5_0 imatrix (mirror of Q4_0 imatrix) ---
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_0, sizeof(block_q5_0),
            quantize_q5_0_imatrix_kernel, "q5_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0), false);
}

// --- Q6_0 OLS imatrix (mirror of Q4_0 imatrix, nmax=32) ---
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_imatrix_kernel, "q6_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0), false);
}

// --- Q5_1 (quantize_row_q5_1_ref, no fudge) ---
size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_1, sizeof(block_q5_1), quantize_q5_1_kernel, "q5_1", 1.0f, 0);
}

// --- Q5_1 imatrix via make_qkx3 (no fudge) ---
size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_1, sizeof(block_q5_1),
            quantize_q5_1_imatrix_kernel, "q5_1_imatrix", 1.0f, false);
}

// --- Q4_1 imatrix via make_qkx3 (no fudge) ---
size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_1, sizeof(block_q4_1),
            quantize_q4_1_imatrix_kernel, "q4_1_imatrix", 1.0f, false);
}

// --- IQ4_NL (dh = FP16(fudge*scale)) ---
size_t ggml_cuda_quantize_iq4_nl(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_kernel, "iq4_nl", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_NL), 256);
}

// --- IQ4_NL imatrix (qw*sqrt(sigma2+x*x), host sigma2 per 32-block) ---
size_t ggml_cuda_quantize_iq4_nl_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_imatrix_kernel, "iq4_nl_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_NL), true);
}

// --- IQ4_XS (dh = FP16(fudge*gd)) ---
size_t ggml_cuda_quantize_iq4_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_kernel, "iq4_xs", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_XS), 256);
}

// --- IQ4_XS imatrix (host sigma2 per superblock) ---
size_t ggml_cuda_quantize_iq4_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_imatrix_kernel, "iq4_xs_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_XS), true);
}

// --- Q6_0 OLS driver (w=x*x, make_qx+fudge) ---
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_ols_kernel, "q6_0", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0), 256);
}

// Q8_0 imatrix routes to plain (CPU ignores imatrix).
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix; // CPU ignores it; match exactly
    return ggml_cuda_quantize_q8_0(src, dst, nrows, n_per_row);
}

// Plain/imatrix drivers shared by IQ2/IQ3 (per-device tables, 128MiB chunks, on-device sigma).
template<typename Block, typename Tables, typename Kernel>
static size_t quantize_plain_iq_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        Tables (*get_tables)(int, int), int table_key, ggml_type fudge_type, Kernel kernel, const char * name) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK_K == 0);
    const float fudge = ggml_get_quantize_fudge_factor(fudge_type);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    const Tables t = get_tables(dev, table_key);
    if (!t.grid) return 0;
    const int64_t nblocks_total = nrows*(n_per_row/QK_K);
    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(QK_K*(int64_t)sizeof(float)));
    auto launch = [&](float * dx, uint8_t * dy, int64_t base, int64_t nblocks, cudaStream_t st) {
        (void) base;
        kernel<<<(unsigned)((nblocks + 256 - 1)/256), 256, 0, st>>>(
                dx, dy, nblocks, fudge, t.grid, t.map, t.neighbours);
    };
    return quantize_exec_chunked(src, dst, nblocks_total, QK_K, sizeof(Block), chunk_blocks, name, launch);
}

template<typename Block, typename Tables, typename Kernel>
static size_t quantize_imatrix_iq_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, Tables (*get_tables)(int, int), int table_key, ggml_type fudge_type,
        Kernel kernel, const char * name) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK_K == 0);
    const float fudge = ggml_get_quantize_fudge_factor(fudge_type);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    const Tables t = get_tables(dev, table_key);
    if (!t.grid) return 0;
    const int64_t nblocks_total = nrows*(n_per_row/QK_K);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK_K);
    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(QK_K*(int64_t)sizeof(float)));
    float * q_dev = nullptr;
    if (cudaMalloc(&q_dev, n_per_row*sizeof(float)) != cudaSuccess) return 0;
    if (cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(q_dev); return 0;
    }
    auto launch = [&](float * dx, uint8_t * dy, int64_t base, int64_t nblocks, cudaStream_t st) {
        kernel<<<(unsigned)((nblocks + 256 - 1)/256), 256, 0, st>>>(
                dx, q_dev, dy, base, nblocks, blocks_per_row, fudge, t.grid, t.map, t.neighbours);
    };
    const size_t out = quantize_exec_chunked(src, dst, nblocks_total, QK_K, sizeof(Block),
            chunk_blocks, name, launch);
    cudaFree(q_dev);
    return out;
}

// --- IQ3_S (QK 256, fudge 1.033; plain w=x*x, imatrix qw*sqrt with 2*sum/256 sigma) ---
size_t ggml_cuda_quantize_iq3_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq3_s>(src, dst, nrows, n_per_row,
            iq3_get_tables, 512, GGML_TYPE_IQ3_S, quantize_iq3_s_kernel, "iq3_s");
}

// --- IQ3_S imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
size_t ggml_cuda_quantize_iq3_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq3_s>(src, dst, nrows, n_per_row, imatrix,
            iq3_get_tables, 512, GGML_TYPE_IQ3_S, quantize_iq3_s_imatrix_kernel, "iq3_s_imatrix");
}

// --- IQ3_XXS (grid 256, fudge 1.0125; parity signs, plain w=x*x) ---
size_t ggml_cuda_quantize_iq3_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq3_xxs>(src, dst, nrows, n_per_row,
            iq3_get_tables, 256, GGML_TYPE_IQ3_XXS, quantize_iq3_xxs_kernel, "iq3_xxs");
}

// --- IQ3_XXS imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
size_t ggml_cuda_quantize_iq3_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq3_xxs>(src, dst, nrows, n_per_row, imatrix,
            iq3_get_tables, 256, GGML_TYPE_IQ3_XXS, quantize_iq3_xxs_imatrix_kernel, "iq3_xxs_imatrix");
}

// --- IQ2_S (group 16, fudge 0.9875; direct signs, plain 0.25*sigma+x*x) ---
size_t ggml_cuda_quantize_iq2_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_s>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, quantize_iq2_s_kernel, "iq2_s");
}

// --- IQ2_S imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
size_t ggml_cuda_quantize_iq2_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_s>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, quantize_iq2_s_imatrix_kernel, "iq2_s_imatrix");
}

// --- IQ2_XS (group 16, grid 512, fudge 1.05; parity + qsort init + LS refit, sigma sum/256) ---
size_t ggml_cuda_quantize_iq2_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_xs>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, quantize_iq2_xs_kernel, "iq2_xs");
}

// --- IQ2_XS imatrix (device sigma sum/256 like CPU, not 2x) ---
size_t ggml_cuda_quantize_iq2_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_xs>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, quantize_iq2_xs_imatrix_kernel, "iq2_xs_imatrix");
}

// --- IQ2_XXS (group 32, grid 256, fudge 1.0; parity, sweep -6..6, sigma sum/256) ---
size_t ggml_cuda_quantize_iq2_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_xxs>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, quantize_iq2_xxs_kernel, "iq2_xxs");
}

// --- IQ2_XXS imatrix (device sigma sum/256 like CPU) ---
size_t ggml_cuda_quantize_iq2_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_xxs>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, quantize_iq2_xxs_imatrix_kernel, "iq2_xxs_imatrix");
}
