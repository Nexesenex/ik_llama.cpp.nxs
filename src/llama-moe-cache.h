#pragma once

#include "ggml-backend.h"
#include "ggml-cuda-moe-ffn.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

struct llama_model;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer_type;

const ggml_cuda_moe_ffn_cache_callbacks * llama_moe_cache_get_cuda_callbacks();

struct llama_moe_cache_frozen_layer {
    int32_t layer = -1;
    int32_t group = -1;
    int32_t device_group = -1;
    ggml_backend * owner_backend = nullptr;
    int32_t n_experts = 0;
    int32_t n_slots = 0;
    uint64_t generation = 0;
    const ggml_tensor * host_up = nullptr;
    const ggml_tensor * host_gate = nullptr;
    const ggml_tensor * host_down = nullptr;
    const ggml_tensor * host_up_gate_storage = nullptr;
    const ggml_tensor * bank_up = nullptr;
    const ggml_tensor * bank_gate = nullptr;
    const ggml_tensor * bank_down = nullptr;
    const ggml_tensor * bank_up_gate_storage = nullptr;
    ggml_tensor * cached_up = nullptr;
    ggml_tensor * cached_gate = nullptr;
    ggml_tensor * cached_down = nullptr;
    ggml_tensor * cached_up_gate_storage = nullptr;
    std::vector<int32_t> group_layers;
};

class llama_moe_cache {
public:
    static constexpr uint32_t max_batch_tokens = 32;

    class frozen_lease {
    public:
        frozen_lease() = default;
        frozen_lease(const frozen_lease &) = delete;
        frozen_lease & operator=(const frozen_lease &) = delete;
        frozen_lease(frozen_lease && other) noexcept;
        frozen_lease & operator=(frozen_lease && other) noexcept;
        ~frozen_lease();

        const llama_moe_cache_frozen_layer & snapshot() const { return descriptor; }
        bool release();
        explicit operator bool() const { return owner != nullptr; }

    private:
        friend class llama_moe_cache;
        llama_moe_cache * owner = nullptr;
        llama_moe_cache_frozen_layer descriptor;
    };

    llama_moe_cache(const llama_model & model,
            const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts,
            size_t size);
    ~llama_moe_cache();

    ggml_backend * backend(int32_t il) const;
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    ggml_tensor * get_experts(const ggml_tensor * tensor) const;
    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase);
    std::map<ggml_backend_buffer_type *, size_t> memory_breakdown() const;
    bool freeze_group_for_layer(int32_t il, ggml_backend * expected_backend, frozen_lease & lease);
    bool frozen_layer_snapshot(const llama_moe_cache_frozen_layer & group_snapshot, int32_t il,
            llama_moe_cache_frozen_layer & snapshot) const;
    bool frozen_snapshot_valid(const llama_moe_cache_frozen_layer & descriptor) const;
    bool resolve_and_pin(int32_t device, int32_t group, int32_t layer, uint64_t owner_generation,
            const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result);
    bool release_pins(const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins);
    bool reserve_fill(int32_t device, int32_t group, int32_t layer, uint64_t owner_generation,
            int32_t expert, ggml_cuda_moe_ffn_fill_job * job,
            ggml_cuda_moe_ffn_fill_part * parts, size_t part_capacity);
    bool finish_fill(const ggml_cuda_moe_ffn_fill_job * job, bool copied);

private:
    bool release_frozen_group(const llama_moe_cache_frozen_layer & descriptor);
    friend class frozen_lease;
    struct impl;
    std::unique_ptr<impl> pimpl;
};
