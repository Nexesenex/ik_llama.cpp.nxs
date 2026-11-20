#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Visual Studio IntelliSense has no CUDA front-end and cannot parse the
// CUDA 12.4+ expansion of __launch_bounds__(func(), n), which becomes
// __declspec(launch_bounds(...)) and breaks on the nested call parentheses.
// That single parse failure cascades into hundreds of bogus errors in the
// .cu sources.  Neutralize the attribute for IntelliSense only; nvcc and
// every real compiler leave __INTELLISENSE__ undefined.
#if defined(__INTELLISENSE__)
#undef __launch_bounds__
#define __launch_bounds__(...)
#endif

#if CUDART_VERSION >= 12050
#include <cuda_fp8.h>
#endif // CUDART_VERSION >= 12050

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
