//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// CUDA quantization of IQ4_KT (IQK legacy "KT" family, ggml/src/iqk/).
//
// Not byte-exact reproducible (see quantize_iq4kt.cuh): the reference
// quantizer quantize_row_iq4_kt_impl sweeps two 32768-point codebooks and
// does a final weighted least-squares scale refinement. This file replays the
// identical search over the same codebooks so the packed output drives the
// same per-row SSE error as the CPU reference within a small tolerance.
//
// Pipeline (all host code chunked by whole rows so no single tensor needs a
// large contiguous VRAM allocation, like the quantize_gguf.cu drivers):
//
//   prep       per 256-value superblock: the row weights (imatrix-aware
//              weight = qw[j]*sqrtf(sigma2 + x^2), else 0.25*sigma2 + x^2)
//              and the per-superblock |x| max.
//   amax_row   per row: max of the per-superblock maxima.
//   scale      per 32-value group: the CPU scale sweep. scale_0 =
//              max(90, 124*amax/amax_row), then the kNtry=2 iteration loop
//              over codebook1 and codebook2 (the ± offset codebook) exactly
//              like quantize_row_iq4_kt_impl, recording the winning scale and
//              the "with offset" bit.
//   row_d      per row: d = -max_scale/64.
//   pack       per 32-value group: ls = nearest_int(id*scale), the codebook
//              refinement best match and the 128-byte block write (shb/8,
//              ql, qh) plus the accumulated weighted sums.
//   final      per row: merge sumsqx/sq2 into the stored scale d (one iloop).
//
// The codebooks, cluster bases and flattened in-cluster point lists are
// fetched from the IQK host quantizer with iq4_kt_get_tables() (lazily, once
// per process) so the GPU never rebuilds the LUT.
//
// Every floating point operation that can change the stored bytes is computed
// with a correctly-rounded intrinsic (__fdiv_rn / __fadd_rn / __fmul_rn /
// __fmaf_rn / __fsqrt_rn) so the search is reproducible and immune to
// -use_fast_math. The cross-group sums are accumulated in the exact CPU
// order per row, so the final scale refinement is deterministic.
//
// Returns the number of bytes written (nrows * ggml_row_size(IQ4_KT,
// n_per_row)), or 0 if the call could not be executed (e.g. no CUDA device).

#include "quantize_iq4kt.cuh"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// constants and device tables
// ---------------------------------------------------------------------------

namespace {

constexpr int IQ4KT_GROUP    = 4;   // values per codebook point
constexpr int IQ4KT_BLOCK    = 32;  // values per 32-value group
constexpr int IQ4KT_NG       = IQ4KT_BLOCK / IQ4KT_GROUP; // groups per 32-block
constexpr int IQ4KT_SUPER    = QK_K;                       // 256
constexpr int IQ4KT_NB_SUPER = IQ4KT_SUPER / IQ4KT_BLOCK;  // 32-blocks per superblock
constexpr int IQ4KT_NTRY     = 2;   // scale sweep iterations either side

constexpr float IQ4KT_KSIGMA_SCALE = 2.0f;
constexpr float IQ4KT_KEPS2        = 1e-14f;
constexpr float IQ4KT_KWEIGHT      = 1e-4f;

struct iq4kt_dtable {
    const float   * values;
    const int32_t * cluster_base;
    const int32_t * points;
    int             nval;
};

// nearest_int replica from iqk_quantize.cpp:39. The +12582912.f addition is
// done with __fadd_rn so it is identical under -use_fast_math.
__device__ __forceinline__ int iq4kt_nearest_int(float fval) {
    const unsigned int u = __float_as_uint(__fadd_rn(fval, 12582912.0f));
    return (int)((u & 0x007fffffu) - 0x00400000u);
}

// bin5 for QuantizerIQKT<32,4,15,false,true>: is_abs == false.
__device__ __forceinline__ int iq4kt_bin5(float x) {
    return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
}

// find_best_match: vx[k] = id*x[k] (id = 1/d), cluster u = base-5 bin code,
// then the weighted best point inside cluster u of the given codebook.
__device__ __forceinline__ void iq4kt_find_best_match(float d,
        const float * xb, const float * weight, int * best_idx,
        const iq4kt_dtable & q) {
    if (!d) {
        for (int l = 0; l < IQ4KT_NG; ++l) best_idx[l] = 0;
        return;
    }
    const float id = __fdiv_rn(1.0f, d);
    for (int l = 0; l < IQ4KT_NG; ++l) {
        const float * xl = xb + IQ4KT_GROUP*l;
        const float * wl = weight + IQ4KT_GROUP*l;
        float vx[IQ4KT_GROUP];
        for (int k = 0; k < IQ4KT_GROUP; ++k) vx[k] = __fmul_rn(id, xl[k]);

        int u = 0, mul = 1;
        for (int k = 0; k < IQ4KT_GROUP; ++k) {
            u += iq4kt_bin5(vx[k]) * mul;
            mul *= 5;
        }
        const int base = q.cluster_base[u];
        const int np   = q.cluster_base[u+1] - base;
        float best = INFINITY;
        int bp = 0;
        // Reference behavior: when cluster u has no points (can happen for the
        // 625-cluster tables), the CPU find_best_match falls back to searching
        // every codebook value.
        const int nsearch = np > 0 ? np : q.nval;
        for (int p = 0; p < nsearch; ++p) {
            const int pi = np > 0 ? q.points[base + p] : p;
            const float * vq = q.values + IQ4KT_GROUP*pi;
            float s = 0.0f;
            for (int k = 0; k < IQ4KT_GROUP; ++k) {
                const float dq = __fsub_rn(vq[k], vx[k]);
                s += __fmul_rn(wl[k], __fmul_rn(dq, dq));
            }
            if (s < best) {
                best = s;
                bp = pi;
            }
        }
        best_idx[l] = bp;
    }
}

// find_best_scale: sumqx = sum w*x*q, sumq2 = sum w*q*q over the chosen
// points; returns (sumqx/sumq2, sumqx*sumqx/sumq2) like the reference.
__device__ __forceinline__ void iq4kt_best_scale(const float * xb,
        const float * weight, const int * best_idx, const iq4kt_dtable & q,
        float & dp, float & score) {
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int l = 0; l < IQ4KT_NG; ++l) {
        const float * xl = xb + IQ4KT_GROUP*l;
        const float * wl = weight + IQ4KT_GROUP*l;
        const float * vq = q.values + IQ4KT_GROUP*best_idx[l];
        for (int k = 0; k < IQ4KT_GROUP; ++k) {
            const float wxk = __fmul_rn(wl[k], xl[k]);
            sumqx += __fmul_rn(wxk, vq[k]);
            sumq2 += __fmul_rn(__fmul_rn(wl[k], vq[k]), vq[k]);
        }
    }
    if (sumq2 > 0.0f) {
        dp    = __fdiv_rn(sumqx, sumq2);
        score = __fdiv_rn(__fmul_rn(sumqx, sumqx), sumq2);
    } else {
        dp    = 0.0f;
        score = 0.0f;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// One thread per 256-value superblock: write the weights and the amax part.
__global__ static void k_iq4kt_prep(const float * __restrict__ x,
        const float * __restrict__ imatrix, float * __restrict__ w,
        float * __restrict__ amax_parts, const int64_t total,
        const int64_t nsuper, const int64_t n_per_row) {
    for (int64_t t = blockIdx.x*blockDim.x + threadIdx.x; t < total; t += gridDim.x*blockDim.x) {
        const int64_t row = t / nsuper;
        const int64_t is  = t % nsuper;
        const float * xr = x + row*n_per_row + is*IQ4KT_SUPER;
        float * wr = w + row*n_per_row + is*IQ4KT_SUPER;
        const float * qw = imatrix ? imatrix + is*IQ4KT_SUPER : nullptr;

        float sumx2 = 0.0f;
        float amax  = 0.0f;
        for (int j = 0; j < IQ4KT_SUPER; ++j) {
            const float xj = xr[j];
            sumx2 += __fmul_rn(xj, xj);
            amax = fmaxf(amax, fabsf(xj));
        }
        amax_parts[t] = amax;
        if (sumx2 < IQ4KT_KEPS2*IQ4KT_SUPER) {
            for (int j = 0; j < IQ4KT_SUPER; ++j) wr[j] = IQ4KT_KWEIGHT;
            continue;
        }
        // sigma2 = (2*sumx2)/256, matching set_weights' expression order
        // (sigma2_scale*sumx2/kSuperBlockSize evaluates left-to-right).
        const float sigma2 = __fdiv_rn(__fmul_rn(IQ4KT_KSIGMA_SCALE, sumx2), (float)IQ4KT_SUPER);
        if (qw) {
            for (int ib = 0; ib < IQ4KT_NB_SUPER; ++ib) {
                const float * xb = xr + IQ4KT_BLOCK*ib;
                const float * qwib = qw + IQ4KT_BLOCK*ib;
                float * wb = wr + IQ4KT_BLOCK*ib;
                float sumwx = 0.0f, sumw2 = 0.0f, sumxb2 = 0.0f;
                for (int j = 0; j < IQ4KT_BLOCK; ++j) {
                    const float wj = __fmul_rn(qwib[j], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[j], xb[j]))));
                    wb[j] = wj;
                    sumwx += __fmul_rn(wj, fabsf(xb[j]));
                    sumw2 += __fmul_rn(wj, wj);
                    sumxb2 += __fmul_rn(xb[j], xb[j]);
                }
                if (sumxb2 < IQ4KT_KEPS2 || sumw2 < IQ4KT_KEPS2 || sumwx < IQ4KT_KEPS2) {
                    for (int j = 0; j < IQ4KT_BLOCK; ++j) wb[j] = IQ4KT_KWEIGHT;
                }
            }
        } else {
            const float w0 = __fmul_rn(0.25f, sigma2);
            for (int j = 0; j < IQ4KT_SUPER; ++j) {
                wr[j] = __fadd_rn(w0, __fmul_rn(xr[j], xr[j]));
            }
        }
    }
}

// One thread per row: max of the per-superblock amax parts.
__global__ static void k_iq4kt_amax_row(const float * __restrict__ amax_parts,
        float * __restrict__ amax_row, const int64_t nrows_c, const int64_t nsuper) {
    for (int64_t row = blockIdx.x*blockDim.x + threadIdx.x; row < nrows_c; row += gridDim.x*blockDim.x) {
        float best = 0.0f;
        const float * ap = amax_parts + row*nsuper;
        for (int64_t is = 0; is < nsuper; ++is) best = fmaxf(best, ap[is]);
        amax_row[row] = best;
    }
}

// One thread per 32-value block: the CPU scale sweep.
__global__ static void k_iq4kt_scale(const float * __restrict__ x,
        const float * __restrict__ w, const float * __restrict__ amax_row,
        float * __restrict__ scales, unsigned char * __restrict__ qsb,
        const int64_t total, const int64_t ngrp, const int64_t n_per_row,
        const iq4kt_dtable q1, const iq4kt_dtable q2) {
    for (int64_t t = blockIdx.x*blockDim.x + threadIdx.x; t < total; t += gridDim.x*blockDim.x) {
        const int64_t row = t / ngrp;
        const int64_t gs  = t % ngrp;
        const float * xb = x + row*n_per_row + gs*IQ4KT_BLOCK;
        const float * wb = w + row*n_per_row + gs*IQ4KT_BLOCK;

        float xl[IQ4KT_BLOCK], wl[IQ4KT_BLOCK];
        float amax = 0.0f;
        for (int j = 0; j < IQ4KT_BLOCK; ++j) {
            xl[j] = xb[j];
            wl[j] = wb[j];
            amax = fmaxf(amax, fabsf(xb[j]));
        }

        float scale = 0.0f;
        unsigned char with_offset = 0;
        if (amax >= 1e-16f) {
            const float arow = amax_row[row];
            if (arow > 0.0f) {
                int best_idx[IQ4KT_NG];
                const float scale_0 = fmaxf(90.f, __fdiv_rn(__fmul_rn(124.f, amax), arow));
                float best = 0.0f;
                for (int itry = -IQ4KT_NTRY; itry <= IQ4KT_NTRY; ++itry) {
                    const float dtry = __fdiv_rn(amax, __fadd_rn((float)(8*itry), scale_0));
                    iq4kt_find_best_match( dtry, xl, wl, best_idx, q1);
                    float dp, sp;
                    iq4kt_best_scale(xl, wl, best_idx, q1, dp, sp);
                    if (sp > best) { best = sp; scale = dp; }
                    iq4kt_find_best_match(-dtry, xl, wl, best_idx, q1);
                    float dm, sm;
                    iq4kt_best_scale(xl, wl, best_idx, q1, dm, sm);
                    if (sm > best) { best = sm; scale = dm; }
                }

                iq4kt_find_best_match(scale, xl, wl, best_idx, q2);
                float d2, s2;
                iq4kt_best_scale(xl, wl, best_idx, q2, d2, s2);
                if (s2 > best) {
                    best = s2;
                    scale = d2;
                    with_offset = 1;
                }
                for (int itry = -IQ4KT_NTRY; itry <= IQ4KT_NTRY; ++itry) {
                    const float dtry = __fdiv_rn(amax, __fadd_rn((float)(8*itry), scale_0));
                    iq4kt_find_best_match( dtry, xl, wl, best_idx, q2);
                    float dp, sp;
                    iq4kt_best_scale(xl, wl, best_idx, q2, dp, sp);
                    if (sp > best) { best = sp; scale = dp; with_offset = 1; }
                    iq4kt_find_best_match(-dtry, xl, wl, best_idx, q2);
                    float dm, sm;
                    iq4kt_best_scale(xl, wl, best_idx, q2, dm, sm);
                    if (sm > best) { best = sm; scale = dm; with_offset = 1; }
                }
            }
        }
        scales[t] = scale;
        qsb[t] = with_offset;
    }
}

// One thread per row: d = -max_scale/64.
__global__ static void k_iq4kt_row_d(const float * __restrict__ scales,
        float * __restrict__ d_row, const int64_t nrows_c, const int64_t ngrp) {
    for (int64_t row = blockIdx.x*blockDim.x + threadIdx.x; row < nrows_c; row += gridDim.x*blockDim.x) {
        float amax_scale = 0.0f, max_scale = 0.0f;
        const float * sc = scales + row*ngrp;
        for (int64_t g = 0; g < ngrp; ++g) {
            const float s = sc[g];
            const float as = fabsf(s);
            if (as > amax_scale) {
                amax_scale = as;
                max_scale = s;
            }
        }
        d_row[row] = __fdiv_rn(-max_scale, 64.0f);
    }
}

// One thread per 32-value block: refine ls + best match, write the bytes and
// accumulate the weighted sums into the partial sum arrays.
__global__ static void k_iq4kt_pack(const float * __restrict__ x,
        const float * __restrict__ w, const float * __restrict__ scales,
        const unsigned char * __restrict__ qsb, const float * __restrict__ d_row,
        float * __restrict__ sumqx, float * __restrict__ sumq2,
        void * __restrict__ out, const int64_t total, const int64_t ngrp,
        const int64_t n_per_row, const int64_t row_size,
        const iq4kt_dtable q1, const iq4kt_dtable q2) {
    for (int64_t t = blockIdx.x*blockDim.x + threadIdx.x; t < total; t += gridDim.x*blockDim.x) {
        const int64_t row = t / ngrp;
        const int64_t gs  = t % ngrp;
        const float d = d_row[row];
        if (!d) {
            sumqx[t] = 0.0f;
            sumq2[t] = 0.0f;
            continue; // whole row was zeroed by the driver memset
        }
        const float id = __fdiv_rn(1.0f, d);
        const float scale = scales[t];
        int ls = iq4kt_nearest_int(__fmul_rn(id, scale));
        if (ls > 63) ls = 63;
        const int qsbv = qsb[t] ? 1 : 0;
        const iq4kt_dtable & q = qsbv ? q2 : q1;
        const float dl = __fmul_rn(d, (float)ls);

        const float * xb = x + row*n_per_row + gs*IQ4KT_BLOCK;
        const float * wb = w + row*n_per_row + gs*IQ4KT_BLOCK;
        float xl[IQ4KT_BLOCK], wl[IQ4KT_BLOCK];
        for (int j = 0; j < IQ4KT_BLOCK; ++j) {
            xl[j] = xb[j];
            wl[j] = wb[j];
        }

        int best_idx[IQ4KT_NG];
        iq4kt_find_best_match(dl, xl, wl, best_idx, q);

        const int64_t super = gs / IQ4KT_NB_SUPER;
        const int ib = (int)(gs % IQ4KT_NB_SUPER);
        uint32_t * shb = (uint32_t *)((char *)out + row*row_size + 4 + super*sizeof(block_iq4_kt));
        uint8_t  * qlb = (uint8_t  *)shb + 8*sizeof(uint32_t);   // byte 32
        uint8_t  * qhb = (uint8_t  *)shb + 8*sizeof(uint32_t) + QK_K/4; // byte 96

        uint32_t word = (uint32_t)((((ls + 64) << 1) | qsbv) & 0xffu);
        float sqx = 0.0f, sq2 = 0.0f;
        for (int j = 0; j < IQ4KT_NG; ++j) {
            const int bid = best_idx[j];
            word |= (uint32_t)((bid >> 12) & 7) << (8 + 3*j);
            const int gi = IQ4KT_NG*ib + j;
            qlb[gi] = (uint8_t)(bid & 255);
            const int hi = (bid >> 8) & 0xf;
            const int qbo = gi % 32;                 // qh byte 0..31
            const int sh = 4*(gi/32) + 8*(qbo & 3);  // bit offset in the word
            atomicOr((unsigned int *)qhb + (qbo >> 2), (unsigned int)hi << sh);

            const float * vq = q.values + IQ4KT_GROUP*bid;
            for (int k = 0; k < IQ4KT_GROUP; ++k) {
                const float qv = __fmul_rn(vq[k], (float)ls);
                const float wxk = __fmul_rn(wl[IQ4KT_GROUP*j + k], xl[IQ4KT_GROUP*j + k]);
                sqx += __fmul_rn(wxk, qv);
                sq2 += __fmul_rn(wl[IQ4KT_GROUP*j + k], __fmul_rn(qv, qv));
            }
        }
        shb[ib] = word;
        sumqx[t] = sqx;
        sumq2[t] = sq2;
    }
}

// One thread per row: merge the per-32-block sums in the CPU order (one iloop)
// and write the final scale.
__global__ static void k_iq4kt_final(const float * __restrict__ sumqx,
        const float * __restrict__ sumq2, const float * __restrict__ d_row,
        void * __restrict__ out, const int64_t nrows_c, const int64_t ngrp,
        const int64_t row_size) {
    for (int64_t row = blockIdx.x*blockDim.x + threadIdx.x; row < nrows_c; row += gridDim.x*blockDim.x) {
        float s1 = 0.0f, s2 = 0.0f;
        const float * sx = sumqx + row*ngrp;
        const float * sy = sumq2 + row*ngrp;
        for (int64_t g = 0; g < ngrp; ++g) {
            s1 += sx[g];
            s2 += sy[g];
        }
        float d = d_row[row];
        if (s2 > 0.0f) d = __fdiv_rn(s1, s2);
        *(float *)((char *)out + row*row_size) = d;
    }
}

// ---------------------------------------------------------------------------
// host driver
// ---------------------------------------------------------------------------

size_t ggml_cuda_quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK_K == 0);

    const int64_t nsuper = n_per_row / IQ4KT_SUPER;  // superblocks per row
    const int64_t ngrp   = n_per_row / IQ4KT_BLOCK;  // 32-blocks per row
    const int64_t row_size = 4 + nsuper*sizeof(block_iq4_kt);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Fetch the two IQK codebooks once (host side, lazily built).
    iq4_kt_tables t1, t2;
    iq4_kt_get_tables(0, &t1);
    iq4_kt_get_tables(1, &t2);
    GGML_ASSERT(t1.nfields == IQ4KT_GROUP && t2.nfields == IQ4KT_GROUP);
    GGML_ASSERT(t1.nval == (1 << 15) && t2.nval == (1 << 15));

    float   * vals1 = nullptr, * vals2 = nullptr;
    int32_t * cb1 = nullptr, * pts1 = nullptr, * cb2 = nullptr, * pts2 = nullptr;

    cudaError_t err = cudaMalloc(&vals1, (size_t)t1.nval*IQ4KT_GROUP*sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&cb1, (size_t)(t1.nclusters+1)*sizeof(int32_t));
    if (err == cudaSuccess) err = cudaMalloc(&pts1, (size_t)t1.total_points*sizeof(int32_t));
    if (err == cudaSuccess) err = cudaMalloc(&vals2, (size_t)t2.nval*IQ4KT_GROUP*sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&cb2, (size_t)(t2.nclusters+1)*sizeof(int32_t));
    if (err == cudaSuccess) err = cudaMalloc(&pts2, (size_t)t2.total_points*sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc codebooks: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(vals1); cudaFree(cb1); cudaFree(pts1);
        cudaFree(vals2); cudaFree(cb2); cudaFree(pts2);
        return 0;
    }

    err = cudaMemcpy(vals1, t1.values,      (size_t)t1.nval*IQ4KT_GROUP*sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(cb1, t1.cluster_base, (size_t)(t1.nclusters+1)*sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(pts1, t1.points,     (size_t)t1.total_points*sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(vals2, t2.values,    (size_t)t2.nval*IQ4KT_GROUP*sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(cb2, t2.cluster_base, (size_t)(t2.nclusters+1)*sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(pts2, t2.points,     (size_t)t2.total_points*sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMemcpy codebooks: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(vals1); cudaFree(cb1); cudaFree(pts1);
        cudaFree(vals2); cudaFree(cb2); cudaFree(pts2);
        return 0;
    }

    const iq4kt_dtable q1 = { vals1, cb1, pts1, t1.nval };
    const iq4kt_dtable q2 = { vals2, cb2, pts2, t2.nval };

    // Optional per-row importance matrix (one row, reused for every row, like
    // the CPU driver).
    float * imatrix_dev = nullptr;
    if (imatrix) {
        err = cudaMalloc(&imatrix_dev, (size_t)n_per_row*sizeof(float));
        if (err == cudaSuccess) err = cudaMemcpy(imatrix_dev, imatrix, (size_t)n_per_row*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: imatrix upload: %s\n", __func__, cudaGetErrorString(err));
            cudaFree(imatrix_dev);
            cudaFree(vals1); cudaFree(cb1); cudaFree(pts1);
            cudaFree(vals2); cudaFree(cb2); cudaFree(pts2);
            return 0;
        }
    }

    // Fixed-size device buffers; the tensor is processed in whole-row chunks so
    // that a single large tensor never needs a huge VRAM allocation.
    const int64_t max_floats = (1 << 25); // ~128 MiB of F32 input per chunk
    const int64_t chunk_rows = std::max<int64_t>(1, std::min(nrows, max_floats/n_per_row));

    const int64_t chunk_x  = chunk_rows*n_per_row;
    const int64_t chunk_y  = chunk_rows*row_size;
    const int64_t chunk_ws = chunk_rows*nsuper;
    const int64_t chunk_wg = chunk_rows*ngrp;

    float   * x_dev    = nullptr;
    float   * w_dev    = nullptr;
    float   * amap_dev = nullptr;
    float   * amrow_dev= nullptr;
    float   * scales_dev= nullptr;
    unsigned char * qsb_dev = nullptr;
    float   * drow_dev = nullptr;
    float   * sqx_dev  = nullptr;
    float   * sq2_dev  = nullptr;
    uint8_t * y_dev    = nullptr;

    bool ok = true;
    if (ok) ok = (err = cudaMalloc(&x_dev, chunk_x*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&w_dev, chunk_x*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&amap_dev, chunk_ws*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&amrow_dev, chunk_rows*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&scales_dev, chunk_wg*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&qsb_dev, chunk_wg)) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&drow_dev, chunk_rows*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&sqx_dev, chunk_wg*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&sq2_dev, chunk_wg*sizeof(float))) == cudaSuccess;
    if (ok) ok = (err = cudaMalloc(&y_dev, chunk_y)) == cudaSuccess;
    if (!ok) {
        fprintf(stderr, "%s: cudaMalloc: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev); cudaFree(w_dev); cudaFree(amap_dev); cudaFree(amrow_dev);
        cudaFree(scales_dev); cudaFree(qsb_dev); cudaFree(drow_dev);
        cudaFree(sqx_dev); cudaFree(sq2_dev); cudaFree(y_dev);
        cudaFree(imatrix_dev);
        cudaFree(vals1); cudaFree(cb1); cudaFree(pts1);
        cudaFree(vals2); cudaFree(cb2); cudaFree(pts2);
        return 0;
    }

    const int block = 256;

    for (int64_t base_row = 0; base_row < nrows; base_row += chunk_rows) {
        const int64_t nrows_c = std::min(chunk_rows, nrows - base_row);
        const int64_t rows_x  = nrows_c*n_per_row;
        const int64_t rows_ws = nrows_c*nsuper;
        const int64_t rows_wg = nrows_c*ngrp;
        const int64_t rows_y  = nrows_c*row_size;

        err = cudaMemcpy(x_dev, src + base_row*n_per_row, rows_x*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: cudaMemcpy x H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
        if (cudaMemset(y_dev, 0, rows_y) != cudaSuccess) {
            fprintf(stderr, "%s: cudaMemset y: %s\n", __func__, cudaGetErrorString(err));
            err = cudaErrorUnknown;
            break;
        }

        {
            const int64_t total = rows_ws;
            k_iq4kt_prep<<<(unsigned)((total + block - 1)/block), (unsigned)block>>>(
                    x_dev, imatrix_dev, w_dev, amap_dev, total, nsuper, n_per_row);
        }
        k_iq4kt_amax_row<<<(unsigned)((nrows_c + block - 1)/block), (unsigned)block>>>(
                amap_dev, amrow_dev, nrows_c, nsuper);
        {
            const int64_t total = rows_wg;
            k_iq4kt_scale<<<(unsigned)((total + block - 1)/block), (unsigned)block>>>(
                    x_dev, w_dev, amrow_dev, scales_dev, qsb_dev, total, ngrp, n_per_row,
                    q1, q2);
        }
        k_iq4kt_row_d<<<(unsigned)((nrows_c + block - 1)/block), (unsigned)block>>>(
                scales_dev, drow_dev, nrows_c, ngrp);
        {
            const int64_t total = rows_wg;
            k_iq4kt_pack<<<(unsigned)((total + block - 1)/block), (unsigned)block>>>(
                    x_dev, w_dev, scales_dev, qsb_dev, drow_dev, sqx_dev, sq2_dev,
                    y_dev, total, ngrp, n_per_row, row_size, q1, q2);
        }
        k_iq4kt_final<<<(unsigned)((nrows_c + block - 1)/block), (unsigned)block>>>(
                sqx_dev, sq2_dev, drow_dev, y_dev, nrows_c, ngrp, row_size);

        err = cudaMemcpy((char *)dst + base_row*row_size, y_dev, rows_y, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: cudaMemcpy y D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev); cudaFree(w_dev); cudaFree(amap_dev); cudaFree(amrow_dev);
    cudaFree(scales_dev); cudaFree(qsb_dev); cudaFree(drow_dev);
    cudaFree(sqx_dev); cudaFree(sq2_dev); cudaFree(y_dev);
    cudaFree(imatrix_dev);
    cudaFree(vals1); cudaFree(cb1); cudaFree(pts1);
    cudaFree(vals2); cudaFree(cb2); cudaFree(pts2);

    if (err != cudaSuccess) {
        return 0;
    }

    return (size_t)nrows*row_size;
}
