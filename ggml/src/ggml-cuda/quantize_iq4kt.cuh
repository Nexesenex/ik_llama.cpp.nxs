//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// CUDA quantization of IQ4_KT (IQK legacy "KT" family, ggml/src/iqk/).
//
// Unlike the legacy block quants in quantize_gguf.cu, IQ4_KT is NOT
// byte-exact reproducible: its reference quantizer (quantize_row_iq4_kt_impl
// in ggml/src/iqk/iqk_quantize.cpp) depends on - in addition to the per-block
// amax/scale heuristics - a swept scale search with two 32768-entry codebooks
// and a final weighted least-squares scale refinement whose accumulation order
// is load/store-architecture-dependent. This CUDA implementation therefore
// targets *bit*... quality parity instead: it replays the exact same search
// over the same codebooks (uploaded via iq4_kt_get_tables) so the packed
// output drives the same per-row SSE error as the CPU reference within a small
// tolerance. The harness in examples/unit_test_cuda.cpp compares the per-row
// quantization SSE error, not the bytes.
//
// Pipeline (all host code chunked so no single tensor needs a large contiguous
// VRAM allocation, like the quantize_gguf.cu drivers):
//
//   prep    : per row + per 256-superblock weights (imatrix-aware weight =
//             qw[j]*sqrtf(sigma2 + x^2), else 0.25*sigma2 + x^2) and the per
//             row amax.
//   scale   : per 32-value group the CPU scale search: scale_0 from
//             max(90, 124*amax/amax_row), then the ±2 iteration loop over
//             codebook1 and codebook2 (± shifted codebook), exactly like
//             quantize_row_iq4_kt_impl.
//   pack    : per 256-value superblock the per-group ls = nearest_int(id*ls),
//             the codebook refinement best match and the 128-byte block write
//             (shb/8, ql, qh) plus the accumulated weighted sums.
//   final   : per row, merge sumsqx/sq2 into the stored scale d (one iloop).
//
// The codebooks, cluster bases and flattened in-cluster point lists are
// fetched from the IQK host quantizer with iq4_kt_get_tables() (lazily, once
// per process) so the GPU never rebuilds the LUT.
//
// Returns the number of bytes written (nrows * ggml_row_size(IQ4_KT,
// n_per_row)), or 0 if the call could not be executed (e.g. no CUDA device).

#pragma once

#include "common.cuh"

#include "iqk/iqk_quantize.h"

size_t ggml_cuda_quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);