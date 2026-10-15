//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//


#include "iq-encoder.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <mutex>

#include "../ggml-quants.h"

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
    // gi matches iq2_data_index: 0 XXS, 1 XS, 2 IQ1_S/M, 3 S.
    static iq2_cuda_tables tables[GGML_CUDA_MAX_DEVICES][4];
    static bool ready[GGML_CUDA_MAX_DEVICES][4] = {};
    int gi = type == GGML_TYPE_IQ2_XXS ? 0 : type == GGML_TYPE_IQ2_XS ? 1 :
             (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) ? 2 : 3;
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

// Weighted neighbour search for IQ1 (xg table, (pg-1)/2 index, full-ngrid fallback like CPU).
static __device__ int iq1_find_best_neighbour2_device(const uint16_t * neighbours, const uint64_t * grid,
        const float * xval, const float * weight, float scale, const float * xg, int8_t * L, int ngrid) {
    const int num_neighbors = neighbours[0];
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float q = xg[(pg[i] - 1)/2];
            const float diff = __fsub_rn(__fmul_rn(scale, q), xval[i]);
            d2 = __fadd_rn(d2, __fmul_rn(__fmul_rn(weight[i], diff), diff));
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * gi = (const int8_t *)(grid + i);
            float d2 = 0.0f;
            for (int j = 0; j < 8; ++j) {
                const float q = xg[(gi[j] - 1)/2];
                const float diff = __fsub_rn(__fmul_rn(scale, q), xval[j]);
                d2 = __fadd_rn(d2, __fmul_rn(__fmul_rn(weight[j], diff), diff));
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (int8_t)((pg[i] - 1)/2);
    return grid_index;
}

// Stable insertion sort by (value, index) matching iq1_sort_helper total order.
static __device__ void iq1_sort_pairs_device(float * vals, int * order, int n) {
    for (int j = 0; j < n; ++j) order[j] = j;
    for (int j = 1; j < n; ++j) {
        const float v = vals[j];
        const int idx = order[j];
        int k = j - 1;
        while (k >= 0 && (vals[k] > v || (vals[k] == v && order[k] > idx))) {
            vals[k + 1] = vals[k];
            order[k + 1] = order[k];
            --k;
        }
        vals[k + 1] = v;
        order[k + 1] = idx;
    }
}

// --- IQ1_M (block 16, QK 256, fudge 1.085; 4-variant search, per-32 1.5*sum/32 sigma) ---
// Mirrors quantize_row_iq1_m_impl (no d field; scale folded into scales high bits).
static __device__ void iq1m_superblock_device(const float * xbl, const float * qw_bl, bool has_imatrix,
        const uint64_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq1_m * y) {
    for (int i = 0; i < 32; ++i) y->qs[i] = 0;
    for (int i = 0; i < 16; ++i) y->qh[i] = 0;
    for (int i = 0; i < 8; ++i) y->scales[i] = 0;

    const float x_p[3] = {-1.0f + 0.125f, 0.125f, 1.0f + 0.125f};
    const float x_m[3] = {-1.0f - 0.125f, -0.125f, 1.0f - 0.125f};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    float all_sigma2[8];
    for (int ib = 0; ib < 8; ++ib) {
        const float * xb32 = xbl + 32*ib;
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum = __fadd_rn(sum, __fmul_rn(xb32[i], xb32[i]));
        all_sigma2[ib] = __fdiv_rn(__fmul_rn(1.5f, sum), 32.0f);
    }

    float scales[16];
    float weight[16];
    int8_t L[16];
    float pairs[16];
    int order[16];
    int8_t shifts[16];
    float max_scale = 0.0f;

    for (int ib = 0; ib < 16; ++ib) {
        const float sigma2 = all_sigma2[ib/2];
        const float * xb = xbl + 16*ib;
        if (has_imatrix) {
            const float * qb = qw_bl + 16*ib;
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))));
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fmul_rn(xb[i], xb[i]);
            }
        }
        float max = fabsf(xb[0]);
        float sumwx = 0.0f;
        for (int i = 1; i < 16; ++i) {
            const float ax = fabsf(xb[i]);
            max = max > ax ? max : ax;
            sumwx = __fadd_rn(sumwx, __fmul_rn(weight[i], ax));
        }
        if (max < 1e-7f) {
            scales[ib] = 0.0f;
            shifts[ib] = 0;
            for (int i = 0; i < 16; ++i) L[i] = 1;
            continue;
        }
        if (sumwx == 0.0f) {
            for (int i = 0; i < 16; ++i) weight[i] = __fmul_rn(xb[i], xb[i]);
        }

        int best_k = -1;
        {
            for (int j = 0; j < 16; ++j) pairs[j] = xb[j];
            iq1_sort_pairs_device(pairs, order, 16);
            float sumw1[17], sumw2[17], sumx1[17], sumx2[17];
            sumw1[0] = 0.0f; sumw2[0] = 0.0f; sumx1[0] = 0.0f; sumx2[0] = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const int i = order[j];
                if (i < 8) {
                    sumw1[j + 1] = __fadd_rn(sumw1[j], weight[i]);
                    sumx1[j + 1] = __fadd_rn(sumx1[j], __fmul_rn(weight[i], xb[i]));
                    sumw2[j + 1] = sumw2[j];
                    sumx2[j + 1] = sumx2[j];
                } else {
                    sumw2[j + 1] = __fadd_rn(sumw2[j], weight[i]);
                    sumx2[j + 1] = __fadd_rn(sumx2[j], __fmul_rn(weight[i], xb[i]));
                    sumw1[j + 1] = sumw1[j];
                    sumx1[j + 1] = sumx1[j];
                }
            }
            float sumqx[4], sumq2[4];
            float best_score = 0.0f, scale_loc = 0.0f;
            int besti1 = -1, besti2 = -1;
            best_k = -1;
            for (int i1 = 0; i1 <= 16; ++i1) {
                for (int i2 = i1; i2 <= 16; ++i2) {
                    const float a10 = __fsub_rn(sumx1[i1], sumx1[0]);
                    const float a21 = __fsub_rn(sumx1[i2], sumx1[i1]);
                    const float a32 = __fsub_rn(sumx1[16], sumx1[i2]);
                    const float b10 = __fsub_rn(sumx2[i1], sumx2[0]);
                    const float b21 = __fsub_rn(sumx2[i2], sumx2[i1]);
                    const float b32 = __fsub_rn(sumx2[16], sumx2[i2]);
                    const float c10 = __fsub_rn(sumw1[i1], sumw1[0]);
                    const float c21 = __fsub_rn(sumw1[i2], sumw1[i1]);
                    const float c32 = __fsub_rn(sumw1[16], sumw1[i2]);
                    const float e10 = __fsub_rn(sumw2[i1], sumw2[0]);
                    const float e21 = __fsub_rn(sumw2[i2], sumw2[i1]);
                    const float e32 = __fsub_rn(sumw2[16], sumw2[i2]);
                    // Strict left-to-right like CPU (a+b+c+d+e+f); squares as (c*x)*x, not c*(x*x).
                    sumqx[0] = __fmul_rn(a10, x_p[0]);
                    sumqx[0] = __fadd_rn(sumqx[0], __fmul_rn(a21, x_p[1]));
                    sumqx[0] = __fadd_rn(sumqx[0], __fmul_rn(a32, x_p[2]));
                    sumqx[0] = __fadd_rn(sumqx[0], __fmul_rn(b10, x_p[0]));
                    sumqx[0] = __fadd_rn(sumqx[0], __fmul_rn(b21, x_p[1]));
                    sumqx[0] = __fadd_rn(sumqx[0], __fmul_rn(b32, x_p[2]));
                    sumqx[1] = __fmul_rn(a10, x_p[0]);
                    sumqx[1] = __fadd_rn(sumqx[1], __fmul_rn(a21, x_p[1]));
                    sumqx[1] = __fadd_rn(sumqx[1], __fmul_rn(a32, x_p[2]));
                    sumqx[1] = __fadd_rn(sumqx[1], __fmul_rn(b10, x_m[0]));
                    sumqx[1] = __fadd_rn(sumqx[1], __fmul_rn(b21, x_m[1]));
                    sumqx[1] = __fadd_rn(sumqx[1], __fmul_rn(b32, x_m[2]));
                    sumqx[2] = __fmul_rn(a10, x_m[0]);
                    sumqx[2] = __fadd_rn(sumqx[2], __fmul_rn(a21, x_m[1]));
                    sumqx[2] = __fadd_rn(sumqx[2], __fmul_rn(a32, x_m[2]));
                    sumqx[2] = __fadd_rn(sumqx[2], __fmul_rn(b10, x_p[0]));
                    sumqx[2] = __fadd_rn(sumqx[2], __fmul_rn(b21, x_p[1]));
                    sumqx[2] = __fadd_rn(sumqx[2], __fmul_rn(b32, x_p[2]));
                    sumqx[3] = __fmul_rn(a10, x_m[0]);
                    sumqx[3] = __fadd_rn(sumqx[3], __fmul_rn(a21, x_m[1]));
                    sumqx[3] = __fadd_rn(sumqx[3], __fmul_rn(a32, x_m[2]));
                    sumqx[3] = __fadd_rn(sumqx[3], __fmul_rn(b10, x_m[0]));
                    sumqx[3] = __fadd_rn(sumqx[3], __fmul_rn(b21, x_m[1]));
                    sumqx[3] = __fadd_rn(sumqx[3], __fmul_rn(b32, x_m[2]));
                    sumq2[0] = __fmul_rn(__fmul_rn(c10, x_p[0]), x_p[0]);
                    sumq2[0] = __fadd_rn(sumq2[0], __fmul_rn(__fmul_rn(c21, x_p[1]), x_p[1]));
                    sumq2[0] = __fadd_rn(sumq2[0], __fmul_rn(__fmul_rn(c32, x_p[2]), x_p[2]));
                    sumq2[0] = __fadd_rn(sumq2[0], __fmul_rn(__fmul_rn(e10, x_p[0]), x_p[0]));
                    sumq2[0] = __fadd_rn(sumq2[0], __fmul_rn(__fmul_rn(e21, x_p[1]), x_p[1]));
                    sumq2[0] = __fadd_rn(sumq2[0], __fmul_rn(__fmul_rn(e32, x_p[2]), x_p[2]));
                    sumq2[1] = __fmul_rn(__fmul_rn(c10, x_p[0]), x_p[0]);
                    sumq2[1] = __fadd_rn(sumq2[1], __fmul_rn(__fmul_rn(c21, x_p[1]), x_p[1]));
                    sumq2[1] = __fadd_rn(sumq2[1], __fmul_rn(__fmul_rn(c32, x_p[2]), x_p[2]));
                    sumq2[1] = __fadd_rn(sumq2[1], __fmul_rn(__fmul_rn(e10, x_m[0]), x_m[0]));
                    sumq2[1] = __fadd_rn(sumq2[1], __fmul_rn(__fmul_rn(e21, x_m[1]), x_m[1]));
                    sumq2[1] = __fadd_rn(sumq2[1], __fmul_rn(__fmul_rn(e32, x_m[2]), x_m[2]));
                    sumq2[2] = __fmul_rn(__fmul_rn(c10, x_m[0]), x_m[0]);
                    sumq2[2] = __fadd_rn(sumq2[2], __fmul_rn(__fmul_rn(c21, x_m[1]), x_m[1]));
                    sumq2[2] = __fadd_rn(sumq2[2], __fmul_rn(__fmul_rn(c32, x_m[2]), x_m[2]));
                    sumq2[2] = __fadd_rn(sumq2[2], __fmul_rn(__fmul_rn(e10, x_p[0]), x_p[0]));
                    sumq2[2] = __fadd_rn(sumq2[2], __fmul_rn(__fmul_rn(e21, x_p[1]), x_p[1]));
                    sumq2[2] = __fadd_rn(sumq2[2], __fmul_rn(__fmul_rn(e32, x_p[2]), x_p[2]));
                    sumq2[3] = __fmul_rn(__fmul_rn(c10, x_m[0]), x_m[0]);
                    sumq2[3] = __fadd_rn(sumq2[3], __fmul_rn(__fmul_rn(c21, x_m[1]), x_m[1]));
                    sumq2[3] = __fadd_rn(sumq2[3], __fmul_rn(__fmul_rn(c32, x_m[2]), x_m[2]));
                    sumq2[3] = __fadd_rn(sumq2[3], __fmul_rn(__fmul_rn(e10, x_m[0]), x_m[0]));
                    sumq2[3] = __fadd_rn(sumq2[3], __fmul_rn(__fmul_rn(e21, x_m[1]), x_m[1]));
                    sumq2[3] = __fadd_rn(sumq2[3], __fmul_rn(__fmul_rn(e32, x_m[2]), x_m[2]));
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0.0f && __fmul_rn(sumqx[k], sumqx[k]) > __fmul_rn(best_score, sumq2[k])) {
                            scale_loc = __fdiv_rn(sumqx[k], sumq2[k]);
                            best_score = __fmul_rn(scale_loc, sumqx[k]);
                            besti1 = i1; besti2 = i2; best_k = k;
                        }
                    }
                }
            }
            for (int j = 0; j < besti1; ++j) L[order[j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[order[j]] = 1;
            for (int j = besti2; j < 16; ++j) L[order[j]] = 2;
            float scale = scale_loc;
            if (scale < 0.0f) {
                for (int j = 0; j < 16; ++j) L[j] = (int8_t)(2 - L[j]);
                scale = -scale;
                best_k = 3 - best_k;
            }
            bool all_on_grid = true;
            const float * xx0 = best_k < 2 ? x_p : x_m;
            const float * xx1 = best_k % 2 == 0 ? x_p : x_m;
            uint16_t the_index[2];
            for (int k = 0; k < 2; ++k) {
                const float * xx = k == 0 ? xx0 : xx1;
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (uint16_t)(L[8*k + j] << 2*j);
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t * nb = neighbours - grid_index - 1;
                    grid_index = iq1_find_best_neighbour2_device(nb, grid, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, 2048);
                }
                the_index[k] = (uint16_t)grid_index;
            }
            if (!all_on_grid) {
                float rq[4], r2[4];
                for (int k = 0; k < 4; ++k) { rq[k] = 0.0f; r2[k] = 0.0f; }
                for (int j = 0; j < 16; ++j) {
                    const float w = weight[j];
                    const float qp = x_p[L[j]];
                    const float qm = x_m[L[j]];
                    rq[0] = __fadd_rn(rq[0], __fmul_rn(__fmul_rn(w, xb[j]), qp));
                    r2[0] = __fadd_rn(r2[0], __fmul_rn(__fmul_rn(w, qp), qp));
                    rq[3] = __fadd_rn(rq[3], __fmul_rn(__fmul_rn(w, xb[j]), qm));
                    r2[3] = __fadd_rn(r2[3], __fmul_rn(__fmul_rn(w, qm), qm));
                    if (j < 8) {
                        rq[1] = __fadd_rn(rq[1], __fmul_rn(__fmul_rn(w, xb[j]), qp));
                        r2[1] = __fadd_rn(r2[1], __fmul_rn(__fmul_rn(w, qp), qp));
                        rq[2] = __fadd_rn(rq[2], __fmul_rn(__fmul_rn(w, xb[j]), qm));
                        r2[2] = __fadd_rn(r2[2], __fmul_rn(__fmul_rn(w, qm), qm));
                    } else {
                        rq[2] = __fadd_rn(rq[2], __fmul_rn(__fmul_rn(w, xb[j]), qp));
                        r2[2] = __fadd_rn(r2[2], __fmul_rn(__fmul_rn(w, qp), qp));
                        rq[1] = __fadd_rn(rq[1], __fmul_rn(__fmul_rn(w, xb[j]), qm));
                        r2[1] = __fadd_rn(r2[1], __fmul_rn(__fmul_rn(w, qm), qm));
                    }
                }
                best_score = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    if (rq[k] > 0.0f && r2[k] > 0.0f && __fmul_rn(rq[k], rq[k]) > __fmul_rn(best_score, r2[k])) {
                        scale = __fdiv_rn(rq[k], r2[k]);
                        best_score = __fmul_rn(scale, rq[k]);
                        best_k = k;
                    }
                }
            }
            scales[ib] = scale;
            y->qs[2*ib] = (uint8_t)(the_index[0] & 255);
            y->qs[2*ib + 1] = (uint8_t)(the_index[1] & 255);
            y->qh[ib] = (uint8_t)((the_index[0] >> 8) | ((the_index[1] >> 8) << 4));
            shifts[ib] = (int8_t)best_k;
            max_scale = max_scale > scale ? max_scale : scale;
        }
    }

    if (max_scale == 0.0f) {
        return;
    }

    uint16_t sc[4] = {0, 0, 0, 0};
    float d = __fdiv_rn(max_scale, 15.0f);
    const float id = __fdiv_rn(1.0f, d);
    float sumqx_f = 0.0f, sumq2_f = 0.0f;
    for (int ib = 0; ib < 16; ++ib) {
        const float sigma2 = all_sigma2[ib/2];
        int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, scales[ib]), 1.0f)));
        l = l < 0 ? 0 : (l > 7 ? 7 : l);
        sc[ib/4] |= (uint16_t)((uint16_t)l << 3*(ib % 4));
        y->qh[ib] |= masks[shifts[ib]];
        const float * xb = xbl + 16*ib;
        float weight[16];
        if (has_imatrix) {
            const float * qb = qw_bl + 16*ib;
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))));
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                weight[i] = __fmul_rn(xb[i], xb[i]);
            }
        }
        for (int k = 0; k < 2; ++k) {
            const float * xx = k == 0 ? (shifts[ib] < 2 ? x_p : x_m) : (shifts[ib] % 2 == 0 ? x_p : x_m);
            const int grid_index = (int)y->qs[2*ib + k] + (int)(((uint16_t)y->qh[ib] << (8 - 4*k)) & 0x700);
            const int8_t * pg = (const int8_t *)(grid + grid_index);
            for (int j = 0; j < 8; ++j) {
                const float q = __fmul_rn(xx[(pg[j] - 1)/2], (float)(2*l + 1));
                sumqx_f = __fadd_rn(sumqx_f, __fmul_rn(__fmul_rn(weight[8*k + j], q), xb[8*k + j]));
                sumq2_f = __fadd_rn(sumq2_f, __fmul_rn(__fmul_rn(weight[8*k + j], q), q));
            }
        }
    }
    if (sumq2_f > 0.0f) d = __fdiv_rn(sumqx_f, sumq2_f);
    const uint16_t h = fp32_to_fp16_ggml(__fmul_rn(d, fudge));
    sc[0] |= (uint16_t)((h & 0x000f) << 12);
    sc[1] |= (uint16_t)((h & 0x00f0) << 8);
    sc[2] |= (uint16_t)((h & 0x0f00) << 4);
    sc[3] |= (uint16_t)((h & 0xf000) << 0);
    for (int i = 0; i < 4; ++i) {
        y->scales[2*i] = (uint8_t)(sc[i] & 255);
        y->scales[2*i + 1] = (uint8_t)((sc[i] >> 8) & 255);
    }
}

static __global__ void quantize_iq1_m_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint64_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xbl = x + sb*QK_K;
    block_iq1_m * y = (block_iq1_m *)vy;
    iq1m_superblock_device(xbl, nullptr, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq1_m_imatrix_kernel(
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
    block_iq1_m * y = (block_iq1_m *)vy;
    iq1m_superblock_device(xs, qs, true, grid, kmap, neighbours, fudge, y + sb);
}

// --- IQ1_S (block 32, QK 256, fudge 1.125; 2-shift search, 2*sum/256 sigma) ---
// Mirrors quantize_row_iq1_s_impl. TEST-ONLY plain (revert before mainline): w = x*x like IQ1_M plain.
static __device__ void iq1s_superblock_device(const float * xbl, const float * qw_bl, bool has_imatrix,
        const uint64_t * grid, const int * kmap, const uint16_t * neighbours, float fudge, block_iq1_s * y) {
    y->d = __ushort_as_half(fp32_to_fp16_ggml(0.0f));
    for (int i = 0; i < 32; ++i) y->qs[i] = 0;
    for (int i = 0; i < 8; ++i) y->qh[i] = 0;

    const float x_p[3] = {-1.0f + 0.125f, 0.125f, 1.0f + 0.125f};
    const float x_m[3] = {-1.0f - 0.125f, -0.125f, 1.0f - 0.125f};
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) sumx2 = __fadd_rn(sumx2, __fmul_rn(xbl[i], xbl[i]));
    const float sigma2 = __fmul_rn(sumx2, 2.0f/(float)QK_K);

    float scales[8];
    float weight[32];
    int8_t L[32];
    float sumx[33], sumw[33];
    float pairs[32];
    int order[32];
    uint16_t index[4];
    int8_t shifts[8];
    float max_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xbl + 32*ib;
        if (has_imatrix) {
            const float * qb = qw_bl + 32*ib;
            for (int i = 0; i < 32; ++i) {
                weight[i] = __fmul_rn(qb[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))));
            }
        } else {
            // TEST-ONLY plain (revert before mainline): w = x*x like IQ1_M plain.
            for (int i = 0; i < 32; ++i) {
                weight[i] = __fmul_rn(xb[i], xb[i]);
            }
        }
        // Abs-max like CPU fabsf loop (max_abs_device takes abs values; xb is signed).
        float amax = fabsf(xb[0]);
        for (int i = 1; i < 32; ++i) {
            const float ax = fabsf(xb[i]);
            amax = amax > ax ? amax : ax;
        }
        if (amax < 1e-12f) {
            scales[ib] = 0.0f;
            shifts[ib] = 1;
            continue;
        }
        for (int j = 0; j < 32; ++j) pairs[j] = xb[j];
        iq1_sort_pairs_device(pairs, order, 32);
        sumx[0] = 0.0f; sumw[0] = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int i = order[j];
            sumx[j + 1] = __fadd_rn(sumx[j], __fmul_rn(weight[i], xb[i]));
            sumw[j + 1] = __fadd_rn(sumw[j], weight[i]);
        }
        float best_score = -FLT_MIN, scale = amax;
        int besti1 = -1, besti2 = -1, best_shift = 0;
        for (int i1 = 0; i1 <= 32; ++i1) {
            for (int i2 = i1; i2 <= 32; ++i2) {
                const float a10 = __fsub_rn(sumx[i1], sumx[0]);
                const float a21 = __fsub_rn(sumx[i2], sumx[i1]);
                const float a32 = __fsub_rn(sumx[32], sumx[i2]);
                const float c10 = __fsub_rn(sumw[i1], sumw[0]);
                const float c21 = __fsub_rn(sumw[i2], sumw[i1]);
                const float c32 = __fsub_rn(sumw[32], sumw[i2]);
                float sumqx = __fadd_rn(__fadd_rn(__fmul_rn(a10, x_p[0]), __fmul_rn(a21, x_p[1])), __fmul_rn(a32, x_p[2]));
                float sumq2 = __fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(c10, x_p[0]), x_p[0]),
                        __fmul_rn(__fmul_rn(c21, x_p[1]), x_p[1])), __fmul_rn(__fmul_rn(c32, x_p[2]), x_p[2]));
                if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best_score, sumq2)) {
                    scale = __fdiv_rn(sumqx, sumq2);
                    best_score = __fmul_rn(scale, sumqx);
                    besti1 = i1; besti2 = i2; best_shift = 1;
                }
                sumqx = __fadd_rn(__fadd_rn(__fmul_rn(a10, x_m[0]), __fmul_rn(a21, x_m[1])), __fmul_rn(a32, x_m[2]));
                sumq2 = __fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(c10, x_m[0]), x_m[0]),
                        __fmul_rn(__fmul_rn(c21, x_m[1]), x_m[1])), __fmul_rn(__fmul_rn(c32, x_m[2]), x_m[2]));
                if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best_score, sumq2)) {
                    scale = __fdiv_rn(sumqx, sumq2);
                    best_score = __fmul_rn(scale, sumqx);
                    besti1 = i1; besti2 = i2; best_shift = -1;
                }
            }
        }
        for (int j = 0; j < besti1; ++j) L[order[j]] = 0;
        for (int j = besti1; j < besti2; ++j) L[order[j]] = 1;
        for (int j = besti2; j < 32; ++j) L[order[j]] = 2;
        if (scale < 0.0f) {
            for (int j = 0; j < 32; ++j) L[j] = (int8_t)(2 - L[j]);
            scale = -scale;
            best_shift = -best_shift;
        }
        bool all_on_grid = true;
        const float * xx = best_shift == 1 ? x_p : x_m;
        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int j = 0; j < 8; ++j) u |= (uint16_t)(L[8*k + j] << 2*j);
            int grid_index = kmap[u];
            if (grid_index < 0) {
                all_on_grid = false;
                const uint16_t * nb = neighbours - grid_index - 1;
                grid_index = iq1_find_best_neighbour2_device(nb, grid, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, 2048);
            }
            index[k] = (uint16_t)grid_index;
        }
        if (!all_on_grid) {
            float sumqx = 0.0f, sumq2 = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const int8_t * pg = (const int8_t *)(grid + index[k]);
                for (int j = 0; j < 8; ++j) {
                    const float q = xx[(pg[j] - 1)/2];
                    sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(weight[8*k + j], q), xb[8*k + j]));
                    sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(weight[8*k + j], q), q));
                }
            }
            if (sumqx > 0.0f && sumq2 > 0.0f) scale = __fdiv_rn(sumqx, sumq2);
        }
        uint16_t h = 0;
        for (int k = 0; k < 4; ++k) {
            y->qs[4*ib + k] = (uint8_t)(index[k] & 255);
            h |= (uint16_t)((index[k] >> 8) << 3*k);
        }
        y->qh[ib] = h;
        scales[ib] = scale;
        max_scale = max_scale > scale ? max_scale : scale;
        shifts[ib] = (int8_t)best_shift;
    }

    if (max_scale == 0.0f) {
        return;
    }

    const float d = __fdiv_rn(max_scale, 15.0f);
    y->d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(d, fudge)));
    const float id = __fdiv_rn(1.0f, d);
    for (int ib = 0; ib < 8; ++ib) {
        int l = nearest_int_device(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, scales[ib]), 1.0f)));
        l = l < 0 ? 0 : (l > 7 ? 7 : l);
        if (shifts[ib] == -1) l |= 8;
        y->qh[ib] |= (uint16_t)((uint16_t)l << 12);
    }
}

static __global__ void quantize_iq1_s_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge,
        const uint64_t * __restrict__ grid, const int * __restrict__ kmap, const uint16_t * __restrict__ neighbours) {
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xbl = x + sb*QK_K;
    block_iq1_s * y = (block_iq1_s *)vy;
    // TEST-ONLY plain (revert before mainline).
    iq1s_superblock_device(xbl, nullptr, false, grid, kmap, neighbours, fudge, y + sb);
}

static __global__ void quantize_iq1_s_imatrix_kernel(
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
    block_iq1_s * y = (block_iq1_s *)vy;
    iq1s_superblock_device(xs, qs, true, grid, kmap, neighbours, fudge, y + sb);
}

// --- IQ4_NL (dh = FP16(fudge*scale)) ---
static size_t ggml_cuda_quantize_iq4_nl(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_kernel, "iq4_nl", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_NL), 256);
}

// --- IQ4_NL imatrix (qw*sqrt(sigma2+x*x), host sigma2 per 32-block) ---
static size_t ggml_cuda_quantize_iq4_nl_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_imatrix_kernel, "iq4_nl_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_NL), true);
}

// --- IQ4_XS (dh = FP16(fudge*gd)) ---
static size_t ggml_cuda_quantize_iq4_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_kernel, "iq4_xs", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_XS), 256);
}

// --- IQ4_XS imatrix (host sigma2 per superblock) ---
static size_t ggml_cuda_quantize_iq4_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_imatrix_kernel, "iq4_xs_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_XS), true);
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
static size_t ggml_cuda_quantize_iq3_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq3_s>(src, dst, nrows, n_per_row,
            iq3_get_tables, 512, GGML_TYPE_IQ3_S, quantize_iq3_s_kernel, "iq3_s");
}

// --- IQ3_S imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
static size_t ggml_cuda_quantize_iq3_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq3_s>(src, dst, nrows, n_per_row, imatrix,
            iq3_get_tables, 512, GGML_TYPE_IQ3_S, quantize_iq3_s_imatrix_kernel, "iq3_s_imatrix");
}

// --- IQ3_XXS (grid 256, fudge 1.0125; parity signs, plain w=x*x) ---
static size_t ggml_cuda_quantize_iq3_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq3_xxs>(src, dst, nrows, n_per_row,
            iq3_get_tables, 256, GGML_TYPE_IQ3_XXS, quantize_iq3_xxs_kernel, "iq3_xxs");
}

// --- IQ3_XXS imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
static size_t ggml_cuda_quantize_iq3_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq3_xxs>(src, dst, nrows, n_per_row, imatrix,
            iq3_get_tables, 256, GGML_TYPE_IQ3_XXS, quantize_iq3_xxs_imatrix_kernel, "iq3_xxs_imatrix");
}

// --- IQ2_S (group 16, fudge 0.9875; direct signs, plain 0.25*sigma+x*x) ---
static size_t ggml_cuda_quantize_iq2_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_s>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, quantize_iq2_s_kernel, "iq2_s");
}

// --- IQ2_S imatrix (device sigma per superblock, 2*sum/256 like CPU) ---
static size_t ggml_cuda_quantize_iq2_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_s>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, quantize_iq2_s_imatrix_kernel, "iq2_s_imatrix");
}

// --- IQ2_XS (group 16, grid 512, fudge 1.05; parity + qsort init + LS refit, sigma sum/256) ---
static size_t ggml_cuda_quantize_iq2_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_xs>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, quantize_iq2_xs_kernel, "iq2_xs");
}

// --- IQ2_XS imatrix (device sigma sum/256 like CPU, not 2x) ---
static size_t ggml_cuda_quantize_iq2_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_xs>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, quantize_iq2_xs_imatrix_kernel, "iq2_xs_imatrix");
}

// --- IQ2_XXS (group 32, grid 256, fudge 1.0; parity, sweep -6..6, sigma sum/256) ---
static size_t ggml_cuda_quantize_iq2_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq2_xxs>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, quantize_iq2_xxs_kernel, "iq2_xxs");
}

// --- IQ2_XXS imatrix (device sigma sum/256 like CPU) ---
static size_t ggml_cuda_quantize_iq2_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq2_xxs>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, quantize_iq2_xxs_imatrix_kernel, "iq2_xxs_imatrix");
}

// --- IQ1_M (block 16, QK 256, fudge 1.085; 4-variant search, per-32 1.5*sum/32 sigma) ---
static size_t ggml_cuda_quantize_iq1_m(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq1_m>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, quantize_iq1_m_kernel, "iq1_m");
}

// --- IQ1_M imatrix (device per-32 sigma like CPU) ---
static size_t ggml_cuda_quantize_iq1_m_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq1_m>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, quantize_iq1_m_imatrix_kernel, "iq1_m_imatrix");
}

// --- IQ1_S (block 32, QK 256, fudge 1.125; TEST-ONLY plain, revert before mainline) ---
static size_t ggml_cuda_quantize_iq1_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_plain_iq_generic<block_iq1_s>(src, dst, nrows, n_per_row,
            iq2_get_tables, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, quantize_iq1_s_kernel, "iq1_s");
}

static size_t ggml_cuda_quantize_iq1_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return quantize_imatrix_iq_generic<block_iq1_s>(src, dst, nrows, n_per_row, imatrix,
            iq2_get_tables, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, quantize_iq1_s_imatrix_kernel, "iq1_s_imatrix");
}			

// Family entry: plain or imatrix driver per slice.
size_t ggml_cuda_quantize_iq(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
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
            case GGML_TYPE_IQ4_NL:  nb = s_im ? ggml_cuda_quantize_iq4_nl_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq4_nl(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ4_XS:  nb = s_im ? ggml_cuda_quantize_iq4_xs_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq4_xs(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ3_S:   nb = s_im ? ggml_cuda_quantize_iq3_s_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq3_s(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ3_XXS: nb = s_im ? ggml_cuda_quantize_iq3_xxs_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq3_xxs(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ2_S:   nb = s_im ? ggml_cuda_quantize_iq2_s_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq2_s(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ2_XS:  nb = s_im ? ggml_cuda_quantize_iq2_xs_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq2_xs(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ2_XXS: nb = s_im ? ggml_cuda_quantize_iq2_xxs_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq2_xxs(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ1_M:   nb = s_im ? ggml_cuda_quantize_iq1_m_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq1_m(s_src, s_dst, nrows, n_per_row); break;
            case GGML_TYPE_IQ1_S:   nb = s_im ? ggml_cuda_quantize_iq1_s_imatrix(s_src, s_dst, nrows, n_per_row, s_im)
                                              : ggml_cuda_quantize_iq1_s(s_src, s_dst, nrows, n_per_row); break;
            default: return 0;
        }
        if (nb == 0) {
            return 0;
        }
        total += nb;
    }
    return total;
}
