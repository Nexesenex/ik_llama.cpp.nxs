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
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// Get device properties for compute-aware GPU selection
// Returns: compute_capability (e.g. 86 for 8.6), memory_bandwidth_GB/s, num_sm, p2p_supported
GGML_API GGML_CALL void ggml_backend_cuda_get_device_props(int device, int * compute_capability, size_t * memory_bandwidth_GBs, int * num_sm, bool * p2p_supported);

// Set CUDA_SCALE_LAUNCH_QUEUES before buffer type init (must be called before any ggml_backend_cuda_buffer_type call)
// Use: ggml_backend_cuda_set_cslq("2x") or ggml_backend_cuda_set_cslq("4x")
GGML_API GGML_CALL void ggml_backend_cuda_set_cslq(const char * cslq);

// Set stream-k efficiency threshold (0-100, default 75)
// Lower values use stream-k more aggressively, higher values prefer wave attention
// Use: ggml_backend_cuda_set_stream_k_thresh(50) for more stream-k
GGML_API GGML_CALL void ggml_backend_cuda_set_stream_k_thresh(int thresh);

// Get stream-k efficiency threshold for a specific device (default 75)
// device: GPU device index (0, 1, 2, ...)
GGML_API GGML_CALL int ggml_backend_cuda_get_stream_k_thresh(int device);

// Set stream-k efficiency threshold for a specific device
// device: GPU device index (0, 1, 2, ...)
// thresh: efficiency threshold (0-100)
GGML_API GGML_CALL void ggml_backend_cuda_set_stream_k_thresh_for_device(int device, int thresh);

// Get recommended stream-k threshold based on device VRAM (in GiB)
// Returns: 85 for 18-200GiB, 70 for 14-18GiB, 60 for 9-14GiB, 50 for 5-9GiB, 40 for <5GiB
GGML_API GGML_CALL int ggml_backend_cuda_get_default_stream_k_thresh(int device_vram_gib);

// Get nblocks_stream_k_raw threshold multiplier (default 4)
GGML_API GGML_CALL int ggml_backend_cuda_get_nblocks_stream_k_raw_thresh(void);

// Set nblocks_stream_k_raw threshold multiplier (1-64)
// Controls when nblocks_stream_k rounds down to multiple of ntiles_dst
// nblocks_stream_k > X * ntiles_dst triggers rounding
GGML_API GGML_CALL void ggml_backend_cuda_set_nblocks_stream_k_raw_thresh(int thresh);

// Set pin_token_embd_only mode for pinned memory allocation (default: 0)
// When set to 1: only token_embd uses pinned memory, CPU tensor overrides (-ot) use non-pinned allocation
// When set to 0: all host buffers use pinned memory (default behavior)
GGML_API GGML_CALL void ggml_backend_cuda_set_pinemb(int val);

// Get current pin_token_embd_only setting
GGML_API GGML_CALL bool ggml_backend_cuda_get_pin_token_embd_only(void);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);
#ifdef  __cplusplus
}
#endif
