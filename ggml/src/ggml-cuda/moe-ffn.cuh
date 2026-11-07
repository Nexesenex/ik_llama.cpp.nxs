#pragma once

#include "ggml-cuda-moe-ffn.h"

struct ggml_backend_cuda_context;
struct ggml_tensor;
struct ggml_cgraph;

bool ggml_cuda_moe_ffn_supports_op(ggml_backend_cuda_context & ctx, const ggml_tensor * op);
bool ggml_cuda_moe_ffn_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op);
bool ggml_cuda_moe_ffn_validate_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph);
bool ggml_cuda_moe_ffn_tensor_on_device(const ggml_tensor * tensor, int device);
