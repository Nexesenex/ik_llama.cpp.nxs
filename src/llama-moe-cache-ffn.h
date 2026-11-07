#pragma once

#include "llama-moe-cache.h"
#include "ggml-moe-ffn.h"

#include <cstdint>
#include <vector>

struct llama_model;
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;

struct llama_moe_cache_ffn_state {
    struct layer_binding {
        llama_moe_cache_frozen_layer snapshot;
        ggml_backend * cuda_backend = nullptr;
        ggml_backend_buffer * cpu_buffer = nullptr;
        ggml_context * cpu_ctx = nullptr;
        ggml_cgraph * cpu_graph = nullptr;
        ggml_tensor * cpu_activations = nullptr;
        ggml_tensor * cpu_ids = nullptr;
        ggml_tensor * cpu_routes = nullptr;
        ggml_tensor * cpu_output = nullptr;
        int32_t binding_id = 0;
        ggml_moe_ffn_layout layout{};
    };

    struct device_state {
        ggml_backend * cuda_backend = nullptr;
        ggml_backend * cpu_backend = nullptr;
        int32_t device = -1;
        int32_t cpu_resource_id = 0;
        int32_t n_embd = 0;
        int32_t n_ff = 0;
        int32_t n_experts_used = 0;
        std::vector<ggml_cuda_moe_ffn_resource_signature> resource_signatures;
        std::vector<llama_moe_cache::frozen_lease> groups;
        std::vector<layer_binding> layer_bindings;
    };

    llama_moe_cache * cache = nullptr;
    std::vector<device_state> devices;
    std::vector<const layer_binding *> layer_by_index;
    bool ready = false;

    bool initialize(llama_moe_cache & cache, const llama_model & model, int32_t cpu_threads,
            int32_t token_capacity);
    void shutdown();
    const layer_binding * find_layer(int32_t il) const;
    ggml_tensor * build_node(ggml_context * ctx, ggml_backend_sched_t sched,
            const ggml_moe_ffn_layout & layout);
};
