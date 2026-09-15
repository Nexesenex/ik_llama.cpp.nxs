#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>

bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);

void iqk_gemm_q8kv_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_convert_kquants_q8X_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

// Exported thin wrapper so external tests can drive the IQ4_XS_R8 converter
// directly (it has internal linkage inside the anonymous namespace).
// Mirrors iqk_test_gemm_q8_k_r16 below. Works with and without HAVE_FANCY_SIMD
// (output is Q8_K_R16 when FANCY, Q8_K_R8 otherwise).
extern "C" void iqk_test_convert_iq4_xs_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x);

#ifdef HAVE_FANCY_SIMD
extern "C" void iqk_test_gemm_q8_k_r16(int n, const void * vx, size_t bx,
                                       const DataInfo& info, int nrc_x, int nrc_y);
#endif

#endif
