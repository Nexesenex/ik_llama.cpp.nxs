//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

// Bit-exact CUDA legacy block quants via Joel's ggml_cuda_quantize() entry in kt-encoder.cu.
// Byte-identical to CPU: exact reductions, fixed rounding, __float2half_rn/__fdiv_rn; sigma2+make_qx_quants replayed in CPU order.

// Order after KT: Q8_0, Q6_0, Q5_0, Q4_0 (+Q5_1/Q4_1/IQ4_NL/IQ4_XS/IQ3_S/IQ3_XXS/IQ2_S/IQ2_XS/IQ2_XXS/IQ1_M/IQ1_S/Q6_K/Q5_K/Q4_K/Q3_K/Q2_K); Q5_0/Q4_0 sections removable; Q6_0 OLS kept, Q8_0 ignores imatrix like CPU.

// Chunked host entries (~128 MiB F32/chunk), checked CUDA calls, return 0 for CPU fallback; returns bytes written or 0 if unavailable.
// Q8_0 (plain ref + fudge; imatrix is ignored to match CPU quantize_q8_0)
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q6_0 OLS (make_qx_quants + fudge; weight = x*x without imatrix)
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// --- Removable Q5_0 section (delete below to drop Q5_0) ---
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// --- Removable Q4_0 section (delete below to drop Q4_0) ---
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q5_1 (plain min/max ref, no fudge; imatrix via make_qkx3, no fudge)
size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q4_1 (plain min/max ref, no fudge; imatrix via make_qkx3, no fudge)
size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ4_NL (ntry=7, w=x*x plain or qw*sqrt imatrix)
size_t ggml_cuda_quantize_iq4_nl(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq4_nl_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ4_XS (ntry=7, w=x*x plain or qw*sqrt imatrix)
size_t ggml_cuda_quantize_iq4_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq4_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ3_S (block 32, QK 256, fudge 1.033; plain w=x*x, imatrix qw*sqrt with 2*sum/256 sigma)
size_t ggml_cuda_quantize_iq3_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq3_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ3_XXS (grid 256, block 32, fudge 1.0125; parity signs, plain w=x*x)
size_t ggml_cuda_quantize_iq3_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq3_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ2_S (group 16, QK 256, fudge 0.9875; direct signs, plain 0.25*sigma+x*x with 2*sum/256 sigma)
size_t ggml_cuda_quantize_iq2_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq2_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ2_XS (group 16, grid 512, fudge 1.05; parity signs, qsort init + 3-iter refine + LS refit)
size_t ggml_cuda_quantize_iq2_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq2_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ2_XXS (group 32, grid 256, fudge 1.0; parity, plain 0.25*sigma+x*x with sum/256 sigma)
size_t ggml_cuda_quantize_iq2_xxs(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq2_xxs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ1_M (block 16, QK 256, fudge 1.085; 4-variant search, per-32 1.5*sum/32 sigma)
size_t ggml_cuda_quantize_iq1_m(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq1_m_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ1_S imatrix-only (block 32, QK 256, fudge 1.125; 2-shift search; plain asserts like CPU)
size_t ggml_cuda_quantize_iq1_s(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq1_s_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q6_K (16x16, make_qx + super iscale; plain w=x*x, imatrix raw qw)
size_t ggml_cuda_quantize_q6_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q6_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q5_K (8x32+1b, make_qkx2 ref / make_qkx3+make_qp imatrix)
size_t ggml_cuda_quantize_q5_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q4_K (8x32, make_qkx2 ref / make_qkx3+make_qp imatrix)
size_t ggml_cuda_quantize_q4_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q3_K (16x16, make_q3 ref / make_qx+make_qx imatrix)
size_t ggml_cuda_quantize_q3_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q3_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q2_K (16x16, TriNet-skipped make_qkx2 ref / make_qkx3+make_qp imatrix)
size_t ggml_cuda_quantize_q2_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q2_K_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
