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
GGML_API GGML_CALL bool ggml_backend_cuda_device_is_tcc(int device);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_pci_bus_id(int device, char * pci_bus_id, size_t pci_bus_id_size);
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

// Lightweight per-device event-tickle poller. A background thread records and
// synchronizes a CUDA event on each non-TCC (WDDM) GPU at its own interval -
// the cheapest way to keep a WDDM card from fully idling between tokens, without
// loading it (no kernel, no FMA). intervals[] maps positionally to WDDM GPUs in
// ggml device order (TCC devices don't consume a slot); a single value broadcasts
// to every WDDM GPU; 0 = off for that GPU. All zeros or n <= 0 disables (default).
// Independent of shark/poller-warmup-fma/poller-nvapi: it does not control the heartbeat warmup.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_sync(const int * intervals, int n);

// Autonomous periodic mem-clock stream (--poller-ping-mem N[,N,...]). A background
// thread fires the mem-clock companion burst on each non-TCC (WDDM) GPU at its
// own interval - no NVAPI, no FMA, no temperature dependency. The shared skip
// mask (set_poller_skip) is honored, so a poller-nvapi-fed too-hot card is skipped.
// intervals[] maps positionally to WDDM GPUs in ggml device order (TCC devices
// don't consume a slot); a single value broadcasts to every WDDM GPU; 0 = off
// for that GPU. n_intervals <= 0 or all zeros disables (default).
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_mem(const int * intervals, int n_intervals);

// Set per-GPU number of 2 MiB passes (bursts) for each autonomous mem-clock ping
// (--poller-ping-mem-amplitude). Same broadcast / positional non-TCC (WDDM) mapping and
// semantics as set_poller_ping_fma_amplitude; 0 in the list disables the mem ping on that GPU.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_mem_amplitude(const int * bursts, int n);

// Autonomous periodic FMA ping (--poller-ping-fma N[,N,...]). A background thread fires
// the full-residency FMA burst on each non-TCC (WDDM) GPU at its own interval -
// the core-clock FMA half of the ping load, no NVAPI. Per-GPU chain length comes
// from set_poller_ping_fma_amplitude(); the shared skip mask (set_poller_skip) is honored.
// Same per-GPU mapping as set_poller_ping_mem (single value broadcasts, 0 = off).
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_fma(const int * intervals, int n_intervals);

// Autonomous periodic tensor-core (HMMA) ping (--poller-ping-mma N[,N,...]). The MMA half
// of the core-clock ping load, fired from the same background thread as the FMA half
// (--poller-ping-fma) at its own per-GPU interval. Per-GPU chain length comes from
// set_poller_ping_mma_amplitude(); the shared skip mask (set_poller_skip) is honored.
// Same per-GPU mapping as set_poller_ping_mem (single value broadcasts, 0 = off).
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_mma(const int * intervals, int n_intervals);

// Enable the CUDA heartbeat warmup (--poller-warmup-fma). During TG, launches a full-residency
// FMA burst per WDDM GPU per decode batch to keep the core clock elevated.
// FMA-only: the mem-clock companion is --poller-warmup-mem's / --poller-ping-mem's job. Default: false.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_warmup_fma(bool val);

// Decode-gated mem-clock companion (--poller-warmup-mem). During TG, launches the mem burst
// per enabled WDDM GPU per decode batch, mirroring the poller-warmup-fma cadence
// (mem-only, FMA-free). bursts[] maps positionally to non-TCC (WDDM) GPUs in ggml
// device order; the value is the number of 2 MiB passes over the companion buffer per
// TG batch, 0 = off for that GPU. A single value broadcasts to every WDDM GPU
// (bare --poller-warmup-mem = 1 burst); all zeros or n <= 0 disables (default). The
// shared skip mask (set_poller_skip) is honored. Works independently of --poller-warmup-fma.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_warmup_mem(const int * bursts, int n);

// Decode-solicited FMA probe (--poller-activity-fma). During TG, fires a short FMA burst on
// a WDDM GPU exactly when that GPU actually receives compute nodes in the current
// batch's split graph (the scheduler invokes graph_compute per device that has
// work). Unlike --poller-warmup-fma (fires on every TG batch) it rides along with real kernels,
// so the default 8192 chain suffices to drag the clock governor to boost. FMA-only;
// the mem-clock side is --poller-warmup-mem's / --poller-ping-mem's job. fmas[] maps positionally to
// non-TCC (WDDM) GPUs in ggml device order; 0 = off for that GPU. A single value
// broadcasts to every WDDM GPU (bare --poller-activity-fma = all GPUs at the default); all
// zeros or n <= 0 disables (default). The shared skip mask (set_poller_skip) is honored.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_activity_fma(const int * fmas, int n);

// Tensor-core HMMA warmup (--poller-warmup-mma). Like --poller-warmup-fma (fires on every TG batch) but
// issues tensor-core matrix-multiply-accumulate instructions instead of scalar
// FMAs: each mma.sync.m16n8k16 fp16 is 2048 FLOPs (m16n8k8 = 1024 on
// Volta/Turing), so the same wall-clock burst delivers ~1000x the compute work -
// a much denser power pulse for the clock governor. mmas[] maps positionally to
// non-TCC (WDDM) GPUs in ggml device order; 0 = off for that GPU. A single value
// broadcasts to every WDDM GPU (bare --poller-warmup-mma = all GPUs at the default 8192);
// all zeros or n <= 0 disables (default). Honors the shared skip mask
// (set_poller_skip) and the prompt-length scale. Pre-Volta cards fall back to
// scalar FFMA inside the kernel, so --poller-warmup-mma still works without tensor cores.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_warmup_mma(const int * mmas, int n);

// Decode-solicited HMMA probe (--poller-activity-mma). Like --poller-activity-fma but for the tensor
// cores: during TG, fires a short HMMA burst on a WDDM GPU exactly when that GPU
// actually receives compute nodes in the current batch's split graph (rides
// along with real kernels). ~1000x the FLOPs per instruction of the FMA probe,
// so a far denser boost per ms. mmas[] maps positionally to non-TCC (WDDM) GPUs
// in ggml device order; 0 = off for that GPU. A single value broadcasts to every
// WDDM GPU (bare --poller-activity-mma = all GPUs at the default 8192); all zeros or n <= 0
// disables (default). Honors the shared skip mask (set_poller_skip) and the
// permanent heat penalty.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_activity_mma(const int * mmas, int n);

// Decode-solicited mem burst (--poller-activity-mem). During TG, fires a short mem-clock burst
// on a WDDM GPU exactly when that GPU actually receives compute nodes in the current
// batch's split graph (the scheduler invokes graph_compute per device that has work).
// bursts[] maps positionally to non-TCC (WDDM) GPUs in ggml device order; the value is
// the number of 2 MiB passes, 0 = off for that GPU. A single value broadcasts to every
// WDDM GPU (bare --poller-activity-mem = 1 burst); all zeros or n <= 0 disables (default).
// The shared skip mask (set_poller_skip) is honored.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_activity_mem(const int * bursts, int n);

// Set per-GPU FMA chain length for the heartbeat warmup kernel. A single value
// broadcasts to every WDDM GPU (bare --poller-warmup-fma = all GPUs at the default); more
// values map positionally to non-TCC (WDDM) GPUs in ggml device order (TCC
// devices skip, so fmas[0] => first WDDM GPU). 0 in the list disables the warmup
// on that GPU (e.g. --poller-warmup-fma 0,32768 disables GPU0; a lone 0 disables all GPUs);
// negative values are replaced by the default; missing values keep the default.
// Call before or after set_poller_warmup_fma(true); it is safe either way (set_poller_warmup_fma(true) logs
// each WDDM GPU with the effective FMA, or that it is disabled).
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_warmup_fma_strength(const int * fmas, int n);

// Set per-GPU FMA chain length for the autonomous FMA ping (--poller-ping-fma). Same
// broadcast / positional non-TCC (WDDM) mapping and semantics as set_poller_warmup_fma_strength;
// shorter than the warmup by default (~1 ms), so the ping cycle keeps idle gaps.
// 0 in the list disables the FMA ping on that GPU (e.g. --poller-ping-fma-amplitude 0,8192).
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_fma_amplitude(const int * fmas, int n);

// Set per-GPU tensor-core (HMMA) chain length for the autonomous MMA ping (--poller-ping-mma).
// The MMA half of the core-clock ping, driven by the same thread as the
// FMA half. Same broadcast / positional non-TCC (WDDM) mapping and semantics as
// set_poller_ping_fma_amplitude; 0 in the list disables the MMA ping on that GPU.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_ping_mma_amplitude(const int * mmas, int n);

// Set a per-card skip mask for the warmup and the pings. skip[] is indexed by
// WDDM position (0 = first non-TCC GPU) and, when true, suppresses the warmup
// (--poller-warmup-fma), the mem stream (--poller-ping-mem) and the FMA/MMA pings
// (--poller-ping-fma/--poller-ping-mma) for that
// GPU (e.g. a card that is too hot). The NVAPI poller (--poller-nvapi) feeds this
// from its per-GPU temperature readings. Pass skip == nullptr to clear it.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_skip(const bool * skip, int n);

// Set the per-card permanent FMA heat penalty (in 1/256ths of the full budget),
// indexed by WDDM position like the skip mask. Fed by the NVAPI temp monitor
// (--poller-nvapi / --poller-warmup-fma): each time a card hits the pause temp twice in a row, the
// monitor adds 16 here for that card. Accumulates, never recovers, and lowers
// the FMA of --poller-warmup-fma / --poller-ping-fma-amplitude / --poller-activity-fma (a penalty that zeroes the
// budget disables that card's FMA). Pass penalty == nullptr to clear it.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_penalty(const int * penalty, int n);

// Get current poller-warmup-fma (heartbeat warmup) setting
GGML_API GGML_CALL bool ggml_backend_cuda_get_poller_warmup_fma(void);

// Set poller-warmup-fma active phase (TG=true, PP=false). Only issue the warmup when active.
// Called from llama.cpp at the same points as the shark_callback.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_active(bool val);

// Set the current prompt length (tokens) so the --poller-warmup-fma / --poller-ping-fma-amplitude FMA scales
// down as the prompt fills the context. The 256 brackets are split in 8 slices
// of 32 (scale is in 1/256ths of the full budget, j = bracket within a slice):
//   slice 1 (k 0..31):   256 - 2*j   2x faster than baseline   (256 -> 194)
//   slice 2 (k 32..63):  192 - 2*j   2x faster than baseline   (192 -> 130)
//   slice 3 (k 64..95):  128 - j     baseline pace             (128 -> 97)
//   slice 4 (k 96..127): 96 - j      baseline pace             (96 -> 65)
//   slice 5 (k 128..159): 64 - j/2   half pace                 (64 -> 49)
//   slice 6 (k 160..191): 48 - j/2   half pace                 (48 -> 33)
//   slice 7 (k 192..223): 32 - j/2   half pace                 (32 -> 17)
//   slice 8 (k 224..255): no FMA at all (callers skip the launch)
// k = floor(256 * n_prompt / n_ctx), clamped to 255. A near-full KV cache is
// already busy with real attention work, so the last slice needs no artificial
// pulse. Called per decode by llama.cpp; n_ctx <= 0 or n_prompt <= 0 keeps full
// FMA. A per-card permanent heat penalty is also subtracted (see
// ggml_backend_cuda_set_poller_penalty); a budget below 32 or a result below 1024
// FMA disables the pulse entirely.
GGML_API GGML_CALL void ggml_backend_cuda_set_poller_prompt_len(int n_prompt, int n_ctx);

// Set stream-k efficiency threshold (0-100, default 75)
// Lower values use stream-k more aggressively, higher values prefer wave attention
// Use: ggml_backend_cuda_set_stream_k_thresh(50) for more stream-k
GGML_API GGML_CALL void ggml_backend_cuda_set_stream_k_thresh(int thresh);

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
// Currently implemented: Q8_0, Q4_0, Q5_0 (ref), and Q4_0/Q5_0 with an
// importance matrix.
// Byte-for-byte identical to the corresponding quantize_row_*_ref CPU
// implementation (and to the imatrix make_qx_quants path for Q4_0/Q5_0).
// src : host F32 row-major buffer, nrows rows of n_per_row floats
// dst : host output buffer, nrows * ggml_row_size(type, n_per_row) bytes
// Returns the number of bytes written, or 0 if the call could not be executed
// (e.g. no CUDA device). See ggml/src/ggml-cuda/quantize_gguf.cuh.
GGML_API GGML_CALL size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
GGML_API GGML_CALL size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
#ifdef  __cplusplus
}
#endif
