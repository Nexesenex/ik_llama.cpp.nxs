//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

// Bit-exact CUDA quantization of legacy (non-OLS) block quants for GGUF.
// Guaranteed byte-for-byte identical to the corresponding quantize_row_*_ref
// implementations because every stored byte depends only on:
//   - max/min reductions (exact, order-independent; argmax ties broken by the
//     lowest index, matching the reference's sequential scan),
//   - per-element rounding (fixed rounding mode),
//   - nearest-even FP16 conversion (__float2half_rn),
//   - correctly-rounded division (__fdiv_rn), immune to -use_fast_math.
// No cross-element floating point accumulation is ever introduced, except in
// the imatrix path where the row-level sigma2 sum is order-dependent and is
// therefore computed on the host in the exact CPU summation order, and where
// the per-block make_qx_quants greedy optimizer is replayed sequentially by a
// single thread in the exact CPU order. That path additionally requires the
// reference functions make_qx_quants / quantize_row_q4_0_impl /
// quantize_row_q5_0_impl in ggml-quants.c to be compiled with
// #pragma STDC FP_CONTRACT OFF (as they now are), so a /arch:AVX2 build cannot
// FMA-contract sumlx += w*x*l and drift by ~1 ulp.
//
// Currently implemented: Q8_0 (ref), Q4_0 (ref), Q5_0 (ref), Q6_0 (ref), and
// Q4_0, Q5_0, Q6_0 with an importance matrix.
//
// The host entries process the tensor in fixed-size device chunks (~128 MiB
// F32 per chunk) so single large tensors never require a large contiguous
// VRAM allocation, and check every CUDA return code. On any failure they
// print the CUDA error and return 0 (the caller aborts the quantization).
//
// Returns the number of bytes written to dst (nrows * ggml_row_size(...)),
// or 0 if the quantization could not be executed (e.g. no CUDA device).
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
