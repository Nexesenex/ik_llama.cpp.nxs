#include "common.cuh"

// Clock-boosting machinery shared with ggml-cuda.cu. The GGML_CALL setters are
// declared in ggml-cuda.h; the two decode-solicited probes below are called from
// ggml_backend_cuda_graph_compute.
void ggml_cuda_poller_activity_fma_ping_device(int i);
void ggml_cuda_poller_activity_mma_ping_device(int i);
