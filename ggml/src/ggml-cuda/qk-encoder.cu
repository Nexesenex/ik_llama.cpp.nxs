//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//


#include "qk-encoder.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <mutex>

#include "../ggml-quants.h"

// --- QK helpers (after IQx; RN intrinsics, CPU order, no FMA) ---

// make_qkx2 port (Q2_K/Q4_K/Q5_K ref; float sums, RN, CPU order).
static __device__ float make_qkx2_quants_device(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux, float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = __fmul_rn(sum_w, x[0]);
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const float w = weights[i];
        sum_w = __fadd_rn(sum_w, w);
        sum_x = __fadd_rn(sum_x, __fmul_rn(w, x[i]));
    }
    if (min > 0.0f) min = 0.0f;
    if (__fsub_rn(max, min) < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.0f;
    }
    float iscale = __fdiv_rn((float)nmax, __fsub_rn(max, min));
    float scale = __fdiv_rn(1.0f, iscale);
    float best_mad = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        float diff = __fadd_rn(__fmul_rn(scale, (float)L[i]), min);
        diff = __fsub_rn(diff, x[i]);
        diff = use_mad ? fabsf(diff) : __fmul_rn(diff, diff);
        best_mad = __fadd_rn(best_mad, __fmul_rn(weights[i], diff));
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, (float)is)), (float)nmax), __fsub_rn(max, min));
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            const float w = weights[i];
            sum_l  = __fadd_rn(sum_l,  __fmul_rn(w, (float)l));
            sum_l2 = __fadd_rn(sum_l2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
            sum_xl = __fadd_rn(sum_xl, __fmul_rn(__fmul_rn(w, (float)l), x[i]));
        }
        const float D = __fsub_rn(__fmul_rn(sum_w, sum_l2), __fmul_rn(sum_l, sum_l));
        if (D > 0.0f) {
            float this_scale = __fdiv_rn(__fsub_rn(__fmul_rn(sum_w, sum_xl), __fmul_rn(sum_x, sum_l)), D);
            float this_min   = __fdiv_rn(__fsub_rn(__fmul_rn(sum_l2, sum_x), __fmul_rn(sum_l, sum_xl)), D);
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = __fdiv_rn(sum_xl, sum_l2);
            }
            float mad = 0.0f;
            for (int i = 0; i < n; ++i) {
                float diff = __fadd_rn(__fmul_rn(this_scale, (float)Laux[i]), this_min);
                diff = __fsub_rn(diff, x[i]);
                diff = use_mad ? fabsf(diff) : __fmul_rn(diff, diff);
                mad = __fadd_rn(mad, __fmul_rn(weights[i], diff));
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// make_q3 port (Q3_K ref, do_rmse=true only; w=x*x, 5-iter, RN).
static __device__ float make_q3_quants_device(int n, int nmax, const float * x, int8_t * L) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)nmax, max);
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
        L[i] = (int8_t)l;
        const float w = __fmul_rn(x[i], x[i]);
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = __fmul_rn(x[i], x[i]);
            const float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)L[i]));
            if (slx > 0.0f) {
                const float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, (float)L[i]), (float)L[i]));
                int new_l = nearest_int_device(__fdiv_rn(__fmul_rn(x[i], sl2), slx));
                new_l = new_l > nmax-1 ? nmax-1 : (new_l < -nmax ? -nmax : new_l);
                if (new_l != (int)L[i]) {
                    const float nslx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), (float)new_l));
                    const float nsl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, (float)new_l), (float)new_l));
                    if (nsl2 > 0.0f && __fmul_rn(__fmul_rn(nslx, nslx), suml2) > __fmul_rn(__fmul_rn(sumlx, sumlx), nsl2)) {
                        L[i] = (int8_t)new_l; sumlx = nslx; suml2 = nsl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    for (int i = 0; i < n; ++i) L[i] += (int8_t)nmax;
    return __fdiv_rn(sumlx, suml2);
}

// Q2_K 0.75x sigma (0.75*sum/256, exact powers of two).
static __device__ float sigma2_075x_device(const float * xbl) {
    float sum = 0.0f;
    for (int j = 0; j < QK_K; ++j) sum = __fadd_rn(sum, __fmul_rn(xbl[j], xbl[j]));
    return __fdiv_rn(__fmul_rn(0.75f, sum), (float)QK_K);
}

// Q4_K/Q5_K 6-bit super-scale pack (same layout as get_scale_min_k4).
static __device__ void pack_qk_k4_scales_device(const uint8_t * Ls, const uint8_t * Lm, uint8_t * scales) {
    for (int j = 0; j < 8; ++j) {
        const uint8_t ls = Ls[j];
        const uint8_t lm = Lm[j];
        if (j < 4) {
            scales[j] = ls;
            scales[j+4] = lm;
        } else {
            scales[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            scales[j-4] = (uint8_t)(scales[j-4] | ((ls >> 4) << 6));
            scales[j-0] = (uint8_t)(scales[j-0] | ((lm >> 4) << 6));
        }
    }
}

// --- Q6_K (16x16, make_qx + super iscale; plain w=x*x, imatrix raw qw) ---
static __device__ void q6k_superblock_device(const float * xs, const float * qw_bl, bool has_imatrix,
        float fudge, block_q6_K * y) {
    int8_t L[QK_K];
    float scales[QK_K/16];
    float max_scale = 0.0f;
    float max_abs = 0.0f;
    for (int ib = 0; ib < QK_K/16; ++ib) {
        const float * xb = xs + 16*ib;
        float scale;
        if (has_imatrix) {
            scale = make_qx_quants_device(16, 32, xb, L + 16*ib, qw_bl + 16*ib);
        } else {
            float weight[16];
            for (int j = 0; j < 16; ++j) weight[j] = __fmul_rn(xb[j], xb[j]);
            scale = make_qx_quants_device(16, 32, xb, L + 16*ib, weight);
        }
        scales[ib] = scale;
        const float a = fabsf(scale);
        if (a > max_abs) { max_abs = a; max_scale = scale; }
    }
    if (max_abs < 1e-15f) {
        for (int i = 0; i < QK_K/2; ++i) y->ql[i] = 0;
        for (int i = 0; i < QK_K/4; ++i) y->qh[i] = 0;
        for (int i = 0; i < QK_K/16; ++i) y->scales[i] = 0;
        y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
        return;
    }
    const float iscale = __fdiv_rn(-128.0f, max_scale);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fdiv_rn(fudge, iscale)));
    const float d_f = __half2float(y->d);
    for (int ib = 0; ib < QK_K/16; ++ib) {
        int l = nearest_int_device(__fmul_rn(iscale, scales[ib]));
        if (l > 127) l = 127;
        y->scales[ib] = (int8_t)l;
    }
    for (int j = 0; j < QK_K/16; ++j) {
        const float d = __fmul_rn(d_f, (float)y->scales[j]);
        if (d == 0.0f) continue;
        for (int ii = 0; ii < 16; ++ii) {
            int l = nearest_int_device(__fdiv_rn(xs[16*j + ii], d));
            l = l > 31 ? 31 : (l < -32 ? -32 : l);
            L[16*j + ii] = (int8_t)(l + 32);
        }
    }
    uint8_t * ql = y->ql;
    uint8_t * qh = y->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)(L[j + l +  0] & 0xF);
            const uint8_t q2 = (uint8_t)(L[j + l + 32] & 0xF);
            const uint8_t q3 = (uint8_t)(L[j + l + 64] & 0xF);
            const uint8_t q4 = (uint8_t)(L[j + l + 96] & 0xF);
            ql[l+ 0] = (uint8_t)(q1 | (q3 << 4));
            ql[l+32] = (uint8_t)(q2 | (q4 << 4));
            qh[l] = (uint8_t)(((uint8_t)(L[j + l] >> 4)) | (((uint8_t)(L[j + l + 32] >> 4)) << 2) |
                    (((uint8_t)(L[j + l + 64] >> 4)) << 4) | (((uint8_t)(L[j + l + 96] >> 4)) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

static __global__ void quantize_q6_K_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    q6k_superblock_device(x + sb*QK_K, nullptr, false, fudge, (block_q6_K *)vy + sb);
}

static __global__ void quantize_q6_K_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    const int64_t gb = base + sb;
    q6k_superblock_device(x + sb*QK_K, qw + (gb % blocks_per_row)*QK_K, true, fudge, (block_q6_K *)vy + sb);
}

// --- Q4_K (8x32, make_qkx2 ref / make_qkx3+make_qp imatrix) ---
static __device__ void q4k_get_scale_min(const uint8_t * scales, int j, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = (uint8_t)(scales[j] & 63); *m = (uint8_t)(scales[j + 4] & 63);
    } else {
        *sc = (uint8_t)((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4));
        *m  = (uint8_t)((scales[j+4] >>  4) | ((scales[j-0] >> 6) << 4));
    }
}

static __device__ void q4k_superblock_plain_device(const float * xs, block_q4_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K/32; ++j) {
        const float * xb = xs + 32*j;
        float sum_x2 = 0.0f;
        for (int l = 0; l < 32; ++l) sum_x2 = __fadd_rn(sum_x2, __fmul_rn(xb[l], xb[l]));
        const float av = __fsqrt_rn(__fdiv_rn(sum_x2, 32.0f));
        for (int l = 0; l < 32; ++l) weights[l] = __fadd_rn(av, fabsf(xb[l]));
        scales[j] = make_qkx2_quants_device(32, 15, xb, weights, L + 32*j, &mins[j], Laux, -1.0f, 0.1f, 20, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    const float inv_scale = max_scale > 0.0f ? __fdiv_rn(63.0f, max_scale) : 0.0f;
    const float inv_min   = max_min   > 0.0f ? __fdiv_rn(63.0f, max_min)   : 0.0f;
    for (int j = 0; j < QK_K/32; ++j) {
        int ls = nearest_int_device(__fmul_rn(inv_scale, scales[j]));
        int lm = nearest_int_device(__fmul_rn(inv_min, mins[j]));
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            y->scales[j] = (uint8_t)ls;
            y->scales[j+4] = (uint8_t)lm;
        } else {
            y->scales[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            y->scales[j-4] = (uint8_t)(y->scales[j-4] | ((ls >> 4) << 6));
            y->scales[j-0] = (uint8_t)(y->scales[j-0] | ((lm >> 4) << 6));
        }
    }
    y->data.d = __ushort_as_half(fp32_to_fp16_ggml(max_scale > 0.0f ? __fdiv_rn(max_scale, 63.0f) : 0.0f));
    y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(max_min > 0.0f ? __fdiv_rn(max_min, 63.0f) : 0.0f));
    const float d_f = __half2float(y->data.d);
    const float m_f = __half2float(y->data.dmin);
    for (int j = 0; j < QK_K/32; ++j) {
        uint8_t sc, m;
        q4k_get_scale_min(y->scales, j, &sc, &m);
        const float d = __fmul_rn(d_f, (float)sc);
        if (d == 0.0f) continue;
        const float dm = __fmul_rn(m_f, (float)m);
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_device(__fdiv_rn(__fadd_rn(xs[32*j + ii], dm), d));
            l = l > 15 ? 15 : (l < 0 ? 0 : l);
            L[32*j + ii] = (uint8_t)l;
        }
    }
    uint8_t * q = y->qs;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        q += 32;
    }
}

static __device__ void q4k_superblock_imatrix_device(const float * xs, const float * qw_bl, block_q4_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float weights[32];
    float sw[QK_K/32];
    float mins[QK_K/32];
    float scales[QK_K/32];
    const float sigma2 = sigma2_2x_device(xs);
    const float av = __fsqrt_rn(sigma2);
    (void) av;
    for (int j = 0; j < QK_K/32; ++j) {
        const float * xb = xs + 32*j;
        const float * qb = qw_bl + 32*j;
        for (int l = 0; l < 32; ++l) weights[l] = __fmul_rn(qb[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
        float sumw = 0.0f;
        for (int l = 0; l < 32; ++l) sumw = __fadd_rn(sumw, weights[l]);
        sw[j] = sumw;
        scales[j] = make_qkx3_quants_device(32, 15, xb, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }
    const float d_block = make_qp_quants_device(QK_K/32, 63, scales, Ls, sw);
    const float m_block = make_qp_quants_device(QK_K/32, 63, mins, Lm, sw);
    pack_qk_k4_scales_device(Ls, Lm, y->scales);
    y->data.d = __ushort_as_half(fp32_to_fp16_ggml(d_block));
    y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(m_block));
    const float d_f = __half2float(y->data.d);
    const float m_f = __half2float(y->data.dmin);
    for (int j = 0; j < QK_K/32; ++j) {
        uint8_t sc, m;
        q4k_get_scale_min(y->scales, j, &sc, &m);
        const float d = __fmul_rn(d_f, (float)sc);
        if (d == 0.0f) continue;
        const float dm = __fmul_rn(m_f, (float)m);
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_device(__fdiv_rn(__fadd_rn(xs[32*j + ii], dm), d));
            l = l > 15 ? 15 : (l < 0 ? 0 : l);
            L[32*j + ii] = (uint8_t)l;
        }
    }
    uint8_t * q = y->qs;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        q += 32;
    }
}

static __global__ void quantize_q4_K_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    q4k_superblock_plain_device(x + sb*QK_K, (block_q4_K *)vy + sb);
}

static __global__ void quantize_q4_K_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    const int64_t gb = base + sb;
    q4k_superblock_imatrix_device(x + sb*QK_K, qw + (gb % blocks_per_row)*QK_K, (block_q4_K *)vy + sb);
}

// --- Q5_K (8x32+1b, make_qkx2 ref / make_qkx3+make_qp imatrix) ---
static __device__ void q5k_pack_qh_qs_device(const uint8_t * L, uint8_t * qs, uint8_t * qh) {
    for (int i = 0; i < QK_K/8; ++i) qh[i] = 0;
    uint8_t m1 = 1, m2 = 2;
    uint8_t * ql = qs;
    for (int n = 0; n < QK_K; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
            int l2 = L[n + j + 32];
            if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 = (uint8_t)(m1 << 2); m2 = (uint8_t)(m2 << 2);
        ql += 32;
    }
}

static __device__ void q5k_superblock_plain_device(const float * xs, block_q5_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K/32; ++j) {
        const float * xb = xs + 32*j;
        float sum_x2 = 0.0f;
        for (int l = 0; l < 32; ++l) sum_x2 = __fadd_rn(sum_x2, __fmul_rn(xb[l], xb[l]));
        const float av = __fsqrt_rn(__fdiv_rn(sum_x2, 32.0f));
        for (int l = 0; l < 32; ++l) weights[l] = __fadd_rn(av, fabsf(xb[l]));
        scales[j] = make_qkx2_quants_device(32, 31, xb, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    const float inv_scale = max_scale > 0.0f ? __fdiv_rn(63.0f, max_scale) : 0.0f;
    const float inv_min   = max_min   > 0.0f ? __fdiv_rn(63.0f, max_min)   : 0.0f;
    for (int j = 0; j < QK_K/32; ++j) {
        int ls = nearest_int_device(__fmul_rn(inv_scale, scales[j]));
        int lm = nearest_int_device(__fmul_rn(inv_min, mins[j]));
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            y->scales[j] = (uint8_t)ls;
            y->scales[j+4] = (uint8_t)lm;
        } else {
            y->scales[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            y->scales[j-4] = (uint8_t)(y->scales[j-4] | ((ls >> 4) << 6));
            y->scales[j-0] = (uint8_t)(y->scales[j-0] | ((lm >> 4) << 6));
        }
    }
    y->data.d = __ushort_as_half(fp32_to_fp16_ggml(max_scale > 0.0f ? __fdiv_rn(max_scale, 63.0f) : 0.0f));
    y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(max_min > 0.0f ? __fdiv_rn(max_min, 63.0f) : 0.0f));
    const float d_f = __half2float(y->data.d);
    const float m_f = __half2float(y->data.dmin);
    for (int j = 0; j < QK_K/32; ++j) {
        uint8_t sc, m;
        q4k_get_scale_min(y->scales, j, &sc, &m);
        const float d = __fmul_rn(d_f, (float)sc);
        if (d == 0.0f) continue;
        const float dm = __fmul_rn(m_f, (float)m);
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_device(__fdiv_rn(__fadd_rn(xs[32*j + ii], dm), d));
            l = l > 31 ? 31 : (l < 0 ? 0 : l);
            L[32*j + ii] = (uint8_t)l;
        }
    }
    q5k_pack_qh_qs_device(L, y->qs, y->qh);
}

static __device__ void q5k_superblock_imatrix_device(const float * xs, const float * qw_bl, block_q5_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float weights[32];
    float sw[QK_K/32];
    float mins[QK_K/32];
    float scales[QK_K/32];
    const float sigma2 = sigma2_2x_device(xs);
    for (int j = 0; j < QK_K/32; ++j) {
        const float * xb = xs + 32*j;
        const float * qb = qw_bl + 32*j;
        for (int l = 0; l < 32; ++l) weights[l] = __fmul_rn(qb[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
        float sumw = 0.0f;
        for (int l = 0; l < 32; ++l) sumw = __fadd_rn(sumw, weights[l]);
        sw[j] = sumw;
        scales[j] = make_qkx3_quants_device(32, 31, xb, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }
    const float d_block = make_qp_quants_device(QK_K/32, 63, scales, Ls, sw);
    const float m_block = make_qp_quants_device(QK_K/32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K/32; ++j) {
        uint8_t ls = Ls[j] > 63 ? 63 : Ls[j];
        uint8_t lm = Lm[j] > 63 ? 63 : Lm[j];
        if (j < 4) {
            y->scales[j] = ls;
            y->scales[j+4] = lm;
        } else {
            y->scales[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            y->scales[j-4] = (uint8_t)(y->scales[j-4] | ((ls >> 4) << 6));
            y->scales[j-0] = (uint8_t)(y->scales[j-0] | ((lm >> 4) << 6));
        }
    }
    y->data.d = __ushort_as_half(fp32_to_fp16_ggml(d_block));
    y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(m_block));
    const float d_f = __half2float(y->data.d);
    const float m_f = __half2float(y->data.dmin);
    for (int j = 0; j < QK_K/32; ++j) {
        uint8_t sc, m;
        q4k_get_scale_min(y->scales, j, &sc, &m);
        const float d = __fmul_rn(d_f, (float)sc);
        if (d == 0.0f) continue;
        const float dm = __fmul_rn(m_f, (float)m);
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_device(__fdiv_rn(__fadd_rn(xs[32*j + ii], dm), d));
            l = l > 31 ? 31 : (l < 0 ? 0 : l);
            L[32*j + ii] = (uint8_t)l;
        }
    }
    q5k_pack_qh_qs_device(L, y->qs, y->qh);
}

static __global__ void quantize_q5_K_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    q5k_superblock_plain_device(x + sb*QK_K, (block_q5_K *)vy + sb);
}

static __global__ void quantize_q5_K_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    const int64_t gb = base + sb;
    q5k_superblock_imatrix_device(x + sb*QK_K, qw + (gb % blocks_per_row)*QK_K, (block_q5_K *)vy + sb);
}

// --- Q3_K (16x16, make_q3 ref / make_qx+make_qx imatrix, 6-bit scales) ---
static __device__ void q3k_pack_scales_device(const int8_t * Ls, uint8_t * scales) {
    for (int i = 0; i < 12; ++i) scales[i] = 0;
    for (int j = 0; j < QK_K/16; ++j) {
        int l = Ls[j];
        if (j < 8) {
            scales[j] = (uint8_t)(l & 0xF);
        } else {
            scales[j-8] = (uint8_t)(scales[j-8] | ((l & 0xF) << 4));
        }
        l >>= 4;
        scales[j%4 + 8] = (uint8_t)(scales[j%4 + 8] | (l << (2*(j/4))));
    }
}

static __device__ int8_t q3k_get_scale(const uint8_t * scales, int j) {
    int8_t sc = j < 8 ? (int8_t)(scales[j] & 0xF) : (int8_t)(scales[j-8] >> 4);
    sc = (int8_t)((sc | ((((scales[8 + j%4] >> (2*(j/4))) & 3) << 4))) - 32);
    return sc;
}

static __device__ void q3k_requant_pack_device(const float * xs, float d_all, const uint8_t * scales, int8_t * L, block_q3_K * y) {
    for (int j = 0; j < QK_K/16; ++j) {
        const int8_t sc = q3k_get_scale(scales, j);
        const float d = __fmul_rn(d_all, (float)sc);
        if (d == 0.0f) continue;
        for (int ii = 0; ii < 16; ++ii) {
            int l = nearest_int_device(__fdiv_rn(xs[16*j + ii], d));
            l = l > 3 ? 3 : (l < -4 ? -4 : l);
            L[16*j + ii] = (int8_t)(l + 4);
        }
    }
    for (int i = 0; i < QK_K/8; ++i) y->hmask[i] = 0;
    int m = 0;
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; ++j) {
        if (L[j] > 3) {
            y->hmask[m] |= hm;
            L[j] -= 4;
        }
        ++m;
        if (m == QK_K/8) { m = 0; hm = (uint8_t)(hm << 1); }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
        }
    }
}

static __device__ void q3k_superblock_plain_device(const float * xs, float fudge, block_q3_K * y) {
    int8_t L[QK_K];
    float scales_f[QK_K/16];
    float max_scale = 0.0f;
    float amax = 0.0f;
    for (int j = 0; j < QK_K/16; ++j) {
        scales_f[j] = make_q3_quants_device(16, 4, xs + 16*j, L + 16*j);
        const float s = fabsf(scales_f[j]);
        if (s > amax) { amax = s; max_scale = scales_f[j]; }
    }
    if (max_scale == 0.0f) {
        for (int i = 0; i < 12; ++i) y->scales[i] = 0;
        y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    } else {
        const float iscale = __fdiv_rn(-32.0f, max_scale);
        int8_t Ls[QK_K/16];
        for (int j = 0; j < QK_K/16; ++j) {
            int l = nearest_int_device(__fmul_rn(iscale, scales_f[j]));
            l = l > 31 ? 31 : (l < -32 ? -32 : l);
            Ls[j] = (int8_t)(l + 32);
        }
        q3k_pack_scales_device(Ls, y->scales);
        y->d = __ushort_as_half(fp32_to_fp16_ggml(__fdiv_rn(fudge, iscale)));
    }
    const float d_all = __half2float(y->d);
    q3k_requant_pack_device(xs, d_all, y->scales, L, y);
}

static __device__ void q3k_superblock_imatrix_device(const float * xs, const float * qw_bl, float fudge, block_q3_K * y) {
    int8_t L[QK_K];
    float scales_f[QK_K/16];
    float weight[16];
    float sw[QK_K/16];
    int8_t Ls[QK_K/16];
    const float sigma2 = sigma2_2x_device(xs);
    for (int j = 0; j < QK_K/16; ++j) {
        const float * xb = xs + 16*j;
        const float * qb = qw_bl + 16*j;
        for (int l = 0; l < 16; ++l) weight[l] = __fmul_rn(qb[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
        float sumw = 0.0f;
        for (int l = 0; l < 16; ++l) sumw = __fadd_rn(sumw, weight[l]);
        sw[j] = sumw;
        scales_f[j] = make_qx_quants_device(16, 4, xb, L + 16*j, weight);
    }
    const float d_block = make_qx_quants_device(QK_K/16, 32, scales_f, Ls, sw);
    q3k_pack_scales_device(Ls, y->scales);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d_block, fudge)));
    const float d_all = __half2float(y->d);
    q3k_requant_pack_device(xs, d_all, y->scales, L, y);
}

static __global__ void quantize_q3_K_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    q3k_superblock_plain_device(x + sb*QK_K, fudge, (block_q3_K *)vy + sb);
}

static __global__ void quantize_q3_K_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    const int64_t gb = base + sb;
    q3k_superblock_imatrix_device(x + sb*QK_K, qw + (gb % blocks_per_row)*QK_K, fudge, (block_q3_K *)vy + sb);
}

// --- Q2_K (16x16, TriNet ref skipped / make_qkx2+4bit / make_qkx3+make_qp imatrix) ---
// Plain skips the global TriNet shortcut (whole-tensor mse<0.1*mse0, rare ternary
// weights); per-superblock path matches CPU otherwise. TriNet tensors fall back
// via nan_block_equal-style review if ever hit e2e.
static __device__ void q2k_superblock_plain_device(const float * xs, block_q2_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K/16; ++j) {
        const float * xb = xs + 16*j;
        for (int l = 0; l < 16; ++l) weights[l] = fabsf(xb[l]);
        scales[j] = make_qkx2_quants_device(16, 3, xb, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    const float q4scale = 15.0f;
    if (max_scale > 0.0f) {
        const float iscale = __fdiv_rn(q4scale, max_scale);
        for (int j = 0; j < QK_K/16; ++j) {
            const int l = nearest_int_device(__fmul_rn(iscale, scales[j]));
            y->scales[j] = (uint8_t)l;
        }
        y->data.d = __ushort_as_half(fp32_to_fp16_ggml(__fdiv_rn(max_scale, q4scale)));
    } else {
        for (int j = 0; j < QK_K/16; ++j) y->scales[j] = 0;
        y->data.d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    }
    if (max_min > 0.0f) {
        const float iscale = __fdiv_rn(q4scale, max_min);
        for (int j = 0; j < QK_K/16; ++j) {
            const int l = nearest_int_device(__fmul_rn(iscale, mins[j]));
            y->scales[j] = (uint8_t)(y->scales[j] | (l << 4));
        }
        y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(__fdiv_rn(max_min, q4scale)));
    } else {
        y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    }
    const float d_f = __half2float(y->data.d);
    const float m_f = __half2float(y->data.dmin);
    for (int j = 0; j < QK_K/16; ++j) {
        const float d = __fmul_rn(d_f, (float)(y->scales[j] & 0xF));
        if (d == 0.0f) continue;
        const float dm = __fmul_rn(m_f, (float)(y->scales[j] >> 4));
        for (int ii = 0; ii < 16; ++ii) {
            int l = nearest_int_device(__fdiv_rn(__fadd_rn(xs[16*j + ii], dm), d));
            l = l > 3 ? 3 : (l < 0 ? 0 : l);
            L[16*j + ii] = (uint8_t)l;
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
        }
    }
}

static __device__ void q2k_superblock_imatrix_device(const float * xs, const float * qw_bl, block_q2_K * y) {
    uint8_t L[QK_K];
    uint8_t Laux[16];
    uint8_t Ls[QK_K/16], Lm[QK_K/16];
    float mins[QK_K/16];
    float scales[QK_K/16];
    float sw[QK_K/16];
    float weight[16];
    const float sigma2 = sigma2_075x_device(xs);
    for (int j = 0; j < QK_K/16; ++j) {
        const float * xb = xs + 16*j;
        const float * qb = qw_bl + 16*j;
        for (int l = 0; l < 16; ++l) weight[l] = __fmul_rn(qb[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
        float sumw = 0.0f;
        for (int l = 0; l < 16; ++l) sumw = __fadd_rn(sumw, weight[l]);
        sw[j] = sumw;
        scales[j] = make_qkx3_quants_device(16, 3, xb, weight, L + 16*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }
    const float dm = make_qp_quants_device(QK_K/16, 15, scales, Ls, sw);
    const float mm = make_qp_quants_device(QK_K/16, 15, mins, Lm, sw);
    y->data.d = __ushort_as_half(fp32_to_fp16_ggml(dm));
    y->data.dmin = __ushort_as_half(fp32_to_fp16_ggml(mm));
    // CPU impl requants with original dm/mm (not half round-trip, unlike Q4_K/Q5_K); match exactly.
    for (int j = 0; j < QK_K/16; ++j) {
        const float d = __fmul_rn(dm, (float)Ls[j]);
        const float m = __fmul_rn(mm, (float)Lm[j]);
        const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        for (int l = 0; l < 16; ++l) {
            int q = nearest_int_device(__fmul_rn(__fadd_rn(xs[16*j + l], m), id));
            q = q > 3 ? 3 : (q < 0 ? 0 : q);
            L[16*j + l] = (uint8_t)q;
        }
    }
    for (int j = 0; j < QK_K/16; ++j) {
        y->scales[j] = (uint8_t)(Ls[j] | (Lm[j] << 4));
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
        }
    }
}

static __global__ void quantize_q2_K_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    q2k_superblock_plain_device(x + sb*QK_K, (block_q2_K *)vy + sb);
}

static __global__ void quantize_q2_K_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge;
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) return;
    const int64_t gb = base + sb;
    q2k_superblock_imatrix_device(x + sb*QK_K, qw + (gb % blocks_per_row)*QK_K, (block_q2_K *)vy + sb);
}

// --- QK drivers (after IQx; plain via generic, imatrix via on-device sigma) ---
using qk_imatrix_kernel_t = void (*)(const float *, const float *, void *, int64_t, int64_t, int32_t, float);

template<typename Block>
static size_t quantize_qk_imatrix_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, qk_imatrix_kernel_t kernel, const char * name, float fudge) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK_K == 0);
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
                dx, q_dev, dy, base, nblocks, blocks_per_row, fudge);
    };
    const size_t out = quantize_exec_chunked(src, dst, nblocks_total, QK_K, sizeof(Block),
            chunk_blocks, name, launch);
    cudaFree(q_dev);
    return out;
}

// --- Q6_K (fudge 1.0; plain w=x*x, imatrix raw qw) ---
static size_t ggml_cuda_quantize_q6_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_q6_K),
            quantize_q6_K_kernel, "q6_K", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_K), 256);
}

static size_t ggml_cuda_quantize_q6_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_qk_imatrix_generic<block_q6_K>(src, dst, nrows, n_per_row, imatrix,
            quantize_q6_K_imatrix_kernel, "q6_K_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_K));
}

// --- Q5_K (no fudge; plain make_qkx2, imatrix make_qkx3+make_qp) ---
static size_t ggml_cuda_quantize_q5_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_q5_K),
            quantize_q5_K_kernel, "q5_K", 1.0f, 256);
}

static size_t ggml_cuda_quantize_q5_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_qk_imatrix_generic<block_q5_K>(src, dst, nrows, n_per_row, imatrix,
            quantize_q5_K_imatrix_kernel, "q5_K_imatrix", 1.0f);
}

// --- Q4_K (no fudge; plain make_qkx2, imatrix make_qkx3+make_qp) ---
static size_t ggml_cuda_quantize_q4_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_q4_K),
            quantize_q4_K_kernel, "q4_K", 1.0f, 256);
}

static size_t ggml_cuda_quantize_q4_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_qk_imatrix_generic<block_q4_K>(src, dst, nrows, n_per_row, imatrix,
            quantize_q4_K_imatrix_kernel, "q4_K_imatrix", 1.0f);
}

// --- Q3_K (fudge 1.0; plain make_q3, imatrix make_qx+make_qx) ---
static size_t ggml_cuda_quantize_q3_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_q3_K),
            quantize_q3_K_kernel, "q3_K", ggml_get_quantize_fudge_factor(GGML_TYPE_Q3_K), 256);
}

static size_t ggml_cuda_quantize_q3_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_qk_imatrix_generic<block_q3_K>(src, dst, nrows, n_per_row, imatrix,
            quantize_q3_K_imatrix_kernel, "q3_K_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q3_K));
}

// --- Q2_K (no fudge; plain make_qkx2 TriNet-skipped, imatrix make_qkx3+make_qp) ---
static size_t ggml_cuda_quantize_q2_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_q2_K),
            quantize_q2_K_kernel, "q2_K", 1.0f, 256);
}

static size_t ggml_cuda_quantize_q2_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_qk_imatrix_generic<block_q2_K>(src, dst, nrows, n_per_row, imatrix,
            quantize_q2_K_imatrix_kernel, "q2_K_imatrix", 1.0f);
}

// Family entry: plain or imatrix driver per slice.
size_t ggml_cuda_quantize_qk(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
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
            case GGML_TYPE_Q6_K: nb = s_im ? ggml_cuda_quantize_q6_K_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q6_K(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q5_K: nb = s_im ? ggml_cuda_quantize_q5_K_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q5_K(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q4_K: nb = s_im ? ggml_cuda_quantize_q4_K_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q4_K(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q3_K: nb = s_im ? ggml_cuda_quantize_q3_K_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q3_K(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_Q2_K: nb = s_im ? ggml_cuda_quantize_q2_K_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                           : ggml_cuda_quantize_q2_K(s_src, s_dst, nrows, n_per_row); break;
            default: return 0;
        }
        if (nb == 0) {
            return 0;
        }
        total += nb;
    }
    return total;
}
