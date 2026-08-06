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
//   - max/min reductions (exact, order-independent),
//   - per-element rounding (fixed rounding mode),
//   - nearest-even FP16 conversion (__float2half_rn).
// No cross-element floating point accumulation is ever introduced.
//
// Currently implemented: Q8_0.
//
// The host entry processes the tensor in fixed-size device chunks (~128 MiB
// F32 per chunk) so single large tensors never require a large contiguous
// VRAM allocation, and checks every CUDA return code. On any failure it
// prints the CUDA error and returns 0 (the caller aborts the quantization).
//
// Returns the number of bytes written to dst (nrows * ggml_row_size(...)),
// or 0 if the quantization could not be executed (e.g. no CUDA device).
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
