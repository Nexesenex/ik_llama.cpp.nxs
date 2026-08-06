#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL int  ggml_backend_cuda_get_device_ordinal(int device);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// Set CUDA_SCALE_LAUNCH_QUEUES before buffer type init (must be called before any ggml_backend_cuda_buffer_type call)
// Use: ggml_backend_cuda_set_cslq("2x") or ggml_backend_cuda_set_cslq("4x")
GGML_API GGML_CALL void ggml_backend_cuda_set_cslq(const char * cslq);

// Set pinmem mode for pinned memory allocation (default: 3)
// pinmem=0: Disable both pinning paths - no pinned memory at all
// pinmem=1: Only pin token_embd, CPU tensor overrides use non-pinned allocation
// pinmem=2: Try to pin all host buffers, stop on first failure, rest unpinned
// pinmem=3: Pin all host buffers (default behavior)
// pinmem=4: Cap pinned memory to 1/4 of total system RAM (user mode)
// pinmem=5: TCC full-size portable pinning (no cap, falls back to pinmem=3)
// pinmem=6: TCC full-size non-portable pinning (no cap, bypasses WDDM quota, falls back to pinmem=3)
// pinmem=7: Selective pinning — only pin token_embd + ffn_down* CPU overrides (falls back to pinmem=1, then 0)
GGML_API GGML_CALL void ggml_backend_cuda_set_pinmem(int val);

// Get current pinmem setting (0=disabled, 1=token_embd only, 2=stop on fail, 3=all, 4=quarter, 5=TCC portable, 6=TCC non-portable, 7=selective)
GGML_API GGML_CALL int ggml_backend_cuda_get_pinmem(void);

// Set pindev mode — specify which raw CUDA ordinal to charge for pinned memory.
// pindev=-1: auto-detect TCC device (default)
// pindev=N:  use raw CUDA ordinal N (e.g. pindev=0 for the first device)
GGML_API GGML_CALL void ggml_backend_cuda_set_pindev(int val);
GGML_API GGML_CALL int  ggml_backend_cuda_get_pindev(void);

// Set pinamount cap in GiB — limits how much of the host buffer is actually pinned.
// pinamount=0 (default): no cap, use full pinmem mode behavior.
// pinamount>0:  cap pinned memory at N GiB (e.g. pinamount=52.5 for 52.5 GiB).
// Works with any pinmem mode: the full buffer is still allocated,
// only the capped prefix is registered as pinned.
GGML_API GGML_CALL void ggml_backend_cuda_set_pinamount(float gb);
GGML_API GGML_CALL float ggml_backend_cuda_get_pinamount(void);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);

// Bit-exact CUDA quantization of legacy (non-OLS) block quants for GGUF.
// Currently implemented: Q8_0. Byte-for-byte identical to the corresponding
// quantize_row_*_ref CPU implementation.
// src : host F32 row-major buffer, nrows rows of n_per_row floats
// dst : host output buffer, nrows * ggml_row_size(type, n_per_row) bytes
// Returns the number of bytes written, or 0 if the call could not be executed
// (e.g. no CUDA device). See ggml/src/ggml-cuda/quantize_gguf.cuh.
GGML_API GGML_CALL size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
#ifdef  __cplusplus
}
#endif
