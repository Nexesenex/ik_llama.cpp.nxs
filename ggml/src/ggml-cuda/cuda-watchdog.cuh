#include "common.cuh"

// CUDA watchdog: monitors GPU progress and detects hangs. The
// cuda_watchdog struct stays nested in ggml_backend_cuda_context (common.cuh);
// the thread proc and the lifecycle helpers live in cuda-watchdog.cu.
void ggml_cuda_watchdog_init(ggml_backend_cuda_context * ctx);
void ggml_cuda_watchdog_cleanup(ggml_backend_cuda_context * ctx);
void ggml_cuda_watchdog_arm(ggml_backend_cuda_context * ctx);
bool ggml_cuda_watchdog_is_hung(const ggml_backend_cuda_context * ctx);
