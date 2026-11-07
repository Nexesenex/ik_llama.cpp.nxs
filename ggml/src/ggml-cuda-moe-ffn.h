#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-moe-ffn.h"

typedef struct ggml_cuda_moe_ffn_fill_part {
    const void * source;
    void * destination;
    size_t bytes;
    // A merged up+gate could store two logical views in separate expert planes
    const void * source2;
    void * destination2;
    size_t bytes2;
} ggml_cuda_moe_ffn_fill_part;

typedef struct ggml_cuda_moe_ffn_fill_job {
    const void * owner;
    int32_t device;
    int32_t group;
    int32_t layer;
    int32_t slot;
    int32_t key;
    uint64_t owner_generation;
    uint64_t generation;
    size_t total_bytes;
} ggml_cuda_moe_ffn_fill_job;

typedef struct ggml_cuda_moe_ffn_pin_token {
    const void * owner;
    int32_t device;
    int32_t group;
    int32_t layer;
    int32_t slot;
    int32_t key;
    uint64_t owner_generation;
    uint64_t generation;
} ggml_cuda_moe_ffn_pin_token;

typedef struct ggml_cuda_moe_ffn_resolution {
    int32_t n_draws;
    int32_t n_pins;
    int32_t * draw_slots;
    size_t draw_capacity;
    ggml_cuda_moe_ffn_pin_token * pins;
    size_t pin_capacity;
} ggml_cuda_moe_ffn_resolution;

typedef bool (*ggml_cuda_moe_ffn_resolve_and_pin_fn)(void * context,
        int32_t device, int32_t group, int32_t layer, uint64_t owner_generation,
        const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result);
typedef bool (*ggml_cuda_moe_ffn_release_pins_fn)(void * context,
        const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins);
typedef bool (*ggml_cuda_moe_ffn_reserve_fill_fn)(void * context,
        int32_t device, int32_t group, int32_t layer, uint64_t owner_generation,
        int32_t expert, ggml_cuda_moe_ffn_fill_job * job,
        ggml_cuda_moe_ffn_fill_part * parts, size_t part_capacity);
typedef bool (*ggml_cuda_moe_ffn_finish_fill_fn)(void * context,
        const ggml_cuda_moe_ffn_fill_job * job, bool copied);

typedef struct ggml_cuda_moe_ffn_cache_callbacks {
    ggml_cuda_moe_ffn_resolve_and_pin_fn resolve_and_pin;
    ggml_cuda_moe_ffn_release_pins_fn release_pins;
    ggml_cuda_moe_ffn_reserve_fill_fn reserve_fill;
    ggml_cuda_moe_ffn_finish_fill_fn finish_fill;
} ggml_cuda_moe_ffn_cache_callbacks;

typedef struct ggml_cuda_moe_ffn_resource_signature {
    int32_t n_embd;
    int32_t n_ff;
    int32_t n_experts_used;
    int32_t token_capacity;
} ggml_cuda_moe_ffn_resource_signature;

typedef struct ggml_cuda_moe_ffn_cpu_resource {
    ggml_backend_t cpu_backend; // ownership transfers on successful registration
    int32_t n_embd;
    int32_t n_ff;
    int32_t n_experts_used;
    int32_t token_capacity;
    int32_t n_threads;
    const ggml_cuda_moe_ffn_resource_signature * signatures; // copied on registration
    size_t n_signatures;
} ggml_cuda_moe_ffn_cpu_resource;

typedef struct ggml_cuda_moe_ffn_cpu_resource_stats {
    uint64_t active_bindings;
    uint64_t pending_pin_active;
    uint64_t fills_accepted;
    uint64_t fills_completed;
    uint64_t fills_failed;
    uint64_t fill_outstanding;
    uint64_t fill_bytes_completed;
} ggml_cuda_moe_ffn_cpu_resource_stats;

typedef struct ggml_cuda_moe_ffn_binding {
    uint64_t generation;
    struct ggml_moe_ffn_layout layout;
    const ggml_cuda_moe_ffn_cache_callbacks * cache_callbacks;
    void * cache_context;
    int32_t cpu_resource_id;
    struct ggml_cgraph * cpu_graph;
    struct ggml_tensor * cpu_activations;
    struct ggml_tensor * cpu_ids;
    struct ggml_tensor * cpu_route_weights;
    struct ggml_tensor * cpu_output;
} ggml_cuda_moe_ffn_binding;

typedef struct ggml_cuda_moe_ffn_stats {
    uint64_t all_hit_calls;
    uint64_t any_miss_calls;
    uint64_t gpu_hit_draws;
    uint64_t cpu_miss_draws;
} ggml_cuda_moe_ffn_stats;

GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_layout_supported(ggml_backend_t backend,
        const struct ggml_moe_ffn_layout * layout);
GGML_API GGML_CALL int32_t ggml_backend_cuda_moe_ffn_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_binding * binding);
GGML_API GGML_CALL int32_t ggml_backend_cuda_moe_ffn_cpu_resource_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_cpu_resource * resource);
GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_unregister(ggml_backend_t backend,
        int32_t resource_id);
GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_stop(ggml_backend_t backend,
        int32_t resource_id);
GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(ggml_backend_t backend,
        int32_t resource_id, ggml_cuda_moe_ffn_cpu_resource_stats * stats);
GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_unregister(ggml_backend_t backend, int32_t binding_id);
GGML_API GGML_CALL bool ggml_backend_cuda_moe_ffn_get_stats(ggml_backend_t backend, int32_t binding_id,
        ggml_cuda_moe_ffn_stats * stats);
