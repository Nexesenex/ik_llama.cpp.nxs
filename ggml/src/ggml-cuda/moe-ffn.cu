#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#include "ggml-backend-impl.h"
#include "ggml-moe-ffn.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/moe-ffn.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/multiadd.cuh"
#include "ggml-cuda/quantize.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_cuda_moe_ffn_cpu_resource_state {
    struct fill_candidate {
        int32_t expert;
        size_t first_draw;
    };
    struct fill_request {
        ggml_cuda_moe_ffn_fill_job owner_job{};
        std::vector<ggml_cuda_moe_ffn_fill_part> parts;
        const ggml_cuda_moe_ffn_cache_callbacks * callbacks = nullptr;
        void * cache_context = nullptr;
        int32_t binding_id = -1;
        uint64_t binding_generation = 0;
    };

    struct pending_pin_record {
        const ggml_cuda_moe_ffn_cache_callbacks * callbacks = nullptr;
        void * context = nullptr;
        std::vector<ggml_cuda_moe_ffn_pin_token> pins;
        bool active = false;
    };

    ggml_cuda_moe_ffn_cpu_resource resource{};
    std::vector<ggml_cuda_moe_ffn_resource_signature> signatures;
    int32_t device = -1;
    size_t draw_capacity = 0;
    // Bound the asynchronous fill pipeline so one burst cannot reserve an
    // unbounded number of cache slots or pin an unbounded amount of host RAM.
    static constexpr size_t MAX_FILL_JOBS = 128;
    static constexpr size_t MAX_FILL_QUEUE_BYTES = 512ull * 1024ull * 1024ull;
    static constexpr int32_t MAX_FILLS_PER_OP = 8;

    std::mutex execution_mutex;
    std::mutex fill_mutex;
    std::condition_variable fill_cv;
    std::thread fill_worker;
    ggml_backend_cuda_context * cuda_context = nullptr;
    cudaStream_t fill_stream = nullptr;
    std::deque<fill_request> fill_queue;
    fill_request stuck_fill{};
    size_t queued_fill_bytes = 0;
    uint32_t fill_outstanding = 0;
    bool fill_active = false;
    bool fill_stuck = false;
    bool fill_stopping = false;
    bool transport_stopped = false;
    bool accepting_fills = true;
    uint64_t fills_accepted = 0;
    uint64_t fills_completed = 0;
    uint64_t fills_failed = 0;
    uint64_t fill_bytes_completed = 0;
    void * host_stage = nullptr;
    int32_t * host_ids = nullptr;
    int32_t * host_slots = nullptr;
    int32_t * host_hit_slots = nullptr;
    float * host_activations = nullptr;
    float * host_routes = nullptr;
    float * host_hit_routes = nullptr;
    float * host_output = nullptr;
    int32_t * device_slots = nullptr;
    float * device_routes = nullptr;
    void * device_up_input_q8_1 = nullptr;
    void * device_down_input_q8_1 = nullptr;
    float * device_hidden = nullptr;
    float * device_down = nullptr;
    float * device_output = nullptr;
    cudaEvent_t hit_kernels_done = nullptr;
    size_t device_up_input_q8_1_bytes = 0;
    size_t device_down_input_q8_1_bytes = 0;
    std::atomic<bool> accepting_bindings{true};
    pending_pin_record pending_pins{};
    std::vector<ggml_cuda_moe_ffn_pin_token> resolved_pins;
    std::vector<fill_candidate> fill_candidates;

    ~ggml_cuda_moe_ffn_cpu_resource_state() {
        GGML_ASSERT(!pending_pins.active && "destroying CUDA MoE resource with live cache pins");
        GGML_ASSERT(!fill_worker.joinable() && !fill_outstanding && "destroying CUDA MoE resource with live fill work");
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        if (device >= 0) cudaSetDevice(device);
        if (hit_kernels_done) cudaEventDestroy(hit_kernels_done);
        if (device_slots) cudaFree(device_slots);
        if (device_routes) cudaFree(device_routes);
        if (device_up_input_q8_1) cudaFree(device_up_input_q8_1);
        if (device_down_input_q8_1) cudaFree(device_down_input_q8_1);
        if (device_hidden) cudaFree(device_hidden);
        if (device_down) cudaFree(device_down);
        if (device_output) cudaFree(device_output);
        if (host_stage) cudaFreeHost(host_stage);
        cudaSetDevice(previous_device);
        if (resource.cpu_backend) ggml_backend_free(resource.cpu_backend);
    }
};

struct ggml_cuda_moe_ffn_binding_state {
    ggml_cuda_moe_ffn_binding binding{};
    ggml_cuda_moe_ffn_stats stats{};
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> cpu_resource;
    bool active = true;
};

static size_t ggml_cuda_moe_ffn_physical_part_count(const ggml_moe_ffn_layout & layout) {
    switch (layout.storage_topology) {
        case GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN: return 3;
        case GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN: return 2;
        default: return 0;
    }
}

static std::pair<const ggml_tensor *, const ggml_tensor *> ggml_cuda_moe_ffn_physical_part(
        const ggml_moe_ffn_layout & layout, size_t index) {
    if (layout.storage_topology == GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN) {
        switch (index) {
            case 0: return { layout.up, layout.bank_up };
            case 1: return { layout.gate, layout.bank_gate };
            case 2: return { layout.down, layout.bank_down };
            default: break;
        }
    } else if (layout.storage_topology == GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN) {
        switch (index) {
            case 0: return { layout.up_gate_storage, layout.bank_up_gate_storage };
            case 1: return { layout.down, layout.bank_down };
            default: break;
        }
    }
    return { nullptr, nullptr };
}

static bool ggml_cuda_moe_ffn_validate_fill_job(const ggml_cuda_moe_ffn_binding & binding,
        const ggml_cuda_moe_ffn_fill_job & job, int32_t expected_expert,
        const ggml_cuda_moe_ffn_fill_part * parts, size_t part_count) {
    const auto & layout = binding.layout;
    const size_t expected_parts = ggml_cuda_moe_ffn_physical_part_count(layout);
    if (binding.layout.n_experts <= 0 || binding.layout.n_slots <= 0 || binding.layout.cache_layer < 0 ||
            expected_expert < 0 || expected_expert >= binding.layout.n_experts ||
            job.owner != binding.cache_context || job.device != binding.layout.device ||
            job.group != binding.layout.cache_group || job.layer != binding.layout.cache_layer ||
            job.owner_generation != binding.generation || job.slot < 0 || job.slot >= binding.layout.n_slots ||
            !parts || expected_parts == 0 || part_count != expected_parts || job.generation == 0 ||
            job.key < 0 || job.key / binding.layout.n_experts != binding.layout.cache_layer ||
            job.key % binding.layout.n_experts != expected_expert) return false;

    size_t total = 0;
    for (size_t i = 0; i < expected_parts; ++i) {
        if (layout.storage_topology == GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN && i == 0) {
            const ggml_tensor * host = layout.up_gate_storage;
            const ggml_tensor * bank = layout.bank_up_gate_storage;
            const ggml_tensor * host_root = host && host->view_src ? host->view_src : host;
            const ggml_tensor * bank_root = bank && bank->view_src ? bank->view_src : bank;
            const ggml_tensor * views[2] = { layout.gate, layout.up };
            const ggml_tensor * bank_views[2] = { layout.bank_gate, layout.bank_up };
            if (!host || !host->data || !bank || !bank->data) return false;
            for (size_t v = 0; v < 2; ++v) {
                const ggml_tensor * source_view = views[v];
                const ggml_tensor * destination_view = bank_views[v];
                if (!source_view || !destination_view || source_view->view_src != host_root ||
                        destination_view->view_src != bank_root ||
                        source_view->view_offs < host->view_offs ||
                        destination_view->view_offs < bank->view_offs ||
                        source_view->ne[2] <= expected_expert ||
                        destination_view->ne[2] <= job.slot || source_view->nb[2] == 0 ||
                        destination_view->nb[2] == 0) return false;
                const size_t rows = ggml_row_size(source_view->type, source_view->ne[0]);
                if (!rows || rows > SIZE_MAX / (size_t) source_view->ne[1]) return false;
                const size_t bytes = rows * (size_t) source_view->ne[1];
                if (source_view->type != destination_view->type || bytes == 0 ||
                        (size_t) expected_expert > SIZE_MAX / source_view->nb[2] ||
                        (size_t) job.slot > SIZE_MAX / destination_view->nb[2]) return false;
                const size_t source_step = (size_t) expected_expert * source_view->nb[2];
                const size_t destination_step = (size_t) job.slot * destination_view->nb[2];
                const size_t source_base = source_view->view_offs - host->view_offs;
                const size_t destination_base = destination_view->view_offs - bank->view_offs;
                if (source_base > SIZE_MAX - source_step ||
                        destination_base > SIZE_MAX - destination_step) return false;
                const size_t source_offset = source_base + source_step;
                const size_t destination_offset = destination_base + destination_step;
                const size_t source_size = ggml_nbytes(host);
                const size_t destination_size = ggml_nbytes(bank);
                const void * expected_source = (const uint8_t *) host->data + source_offset;
                void * expected_destination = (uint8_t *) bank->data + destination_offset;
                const size_t descriptor_bytes = v == 0 ? parts[i].bytes : parts[i].bytes2;
                const void * descriptor_source = v == 0 ? parts[i].source : parts[i].source2;
                void * descriptor_destination = v == 0 ? parts[i].destination : parts[i].destination2;
                if (source_offset > source_size || bytes > source_size - source_offset ||
                        destination_offset > destination_size || bytes > destination_size - destination_offset ||
                        descriptor_bytes != bytes || descriptor_source != expected_source ||
                        descriptor_destination != expected_destination || total > SIZE_MAX - bytes) return false;
                total += bytes;
            }
            continue;
        }
        const auto expected = ggml_cuda_moe_ffn_physical_part(layout, i);
        const ggml_tensor * host = expected.first;
        const ggml_tensor * bank = expected.second;
        if (!host || !host->data || !bank || !bank->data ||
                host->ne[2] <= expected_expert || bank->ne[2] <= job.slot ||
                host->nb[2] == 0 || host->nb[2] != bank->nb[2]) return false;
        const size_t bytes = host->nb[2];
        if (parts[i].bytes != bytes || parts[i].source2 || parts[i].destination2 || parts[i].bytes2 != 0 ||
                (size_t) expected_expert > SIZE_MAX / bytes || (size_t) job.slot > SIZE_MAX / bytes) return false;
        const size_t source_offset = (size_t) expected_expert * bytes;
        const size_t destination_offset = (size_t) job.slot * bytes;
        const size_t source_size = ggml_nbytes(host);
        const size_t destination_size = ggml_nbytes(bank);
        if (source_offset > source_size || bytes > source_size - source_offset ||
                destination_offset > destination_size || bytes > destination_size - destination_offset ||
                parts[i].source != (const uint8_t *) host->data + source_offset ||
                parts[i].destination != (uint8_t *) bank->data + destination_offset ||
                total > SIZE_MAX - bytes) return false;
        total += bytes;
    }
    return total != 0 && job.total_bytes == total;
}

enum class ggml_cuda_moe_ffn_fill_result {
    ownership_unresolved,
    rollback_completed,
    committed,
};

static ggml_cuda_moe_ffn_fill_result ggml_cuda_moe_ffn_copy_fill_request(
        ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_cpu_resource_state::fill_request & request) {
    const auto & job = request.owner_job;
    if (!request.callbacks || !request.callbacks->finish_fill || !request.cache_context)
        return ggml_cuda_moe_ffn_fill_result::ownership_unresolved;
    auto rollback = [&]() {
        return request.callbacks->finish_fill(request.cache_context, &job, false) ?
            ggml_cuda_moe_ffn_fill_result::rollback_completed : ggml_cuda_moe_ffn_fill_result::ownership_unresolved;
    };
    if (!resource.fill_stream || !resource.cuda_context || job.device != resource.device) return rollback();
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> binding_state;
    {
        std::lock_guard<std::mutex> lock(resource.cuda_context->moe_ffn_mutex);
        auto it = resource.cuda_context->moe_ffn_bindings.find(request.binding_id);
        if (it != resource.cuda_context->moe_ffn_bindings.end() &&
                it->second->binding.generation == request.binding_generation &&
                it->second->cpu_resource.get() == &resource) binding_state = it->second;
    }
    if (!binding_state || binding_state->binding.layout.n_experts <= 0 || job.key < 0) return rollback();
    const int32_t expert = job.key % binding_state->binding.layout.n_experts;
    if (!ggml_cuda_moe_ffn_validate_fill_job(binding_state->binding, job, expert,
                request.parts.data(), request.parts.size())) return rollback();

    if (cudaSetDevice(resource.device) != cudaSuccess) return ggml_cuda_moe_ffn_fill_result::ownership_unresolved;
    bool copied = true;
    for (size_t i = 0; i < request.parts.size(); ++i) {
        const auto & part = request.parts[i];
        // Model weights are persistent host buffers. CUDA may stage pageable sources synchronously
        if (cudaMemcpyAsync(part.destination, part.source, part.bytes,
                    cudaMemcpyHostToDevice, resource.fill_stream) != cudaSuccess ||
                (part.bytes2 && cudaMemcpyAsync(part.destination2, part.source2, part.bytes2,
                    cudaMemcpyHostToDevice, resource.fill_stream) != cudaSuccess)) {
            copied = false;
            break;
        }
    }
    const cudaError_t drained = cudaStreamSynchronize(resource.fill_stream);
    if (drained != cudaSuccess) {
        return ggml_cuda_moe_ffn_fill_result::ownership_unresolved;
    }
    if (!request.callbacks->finish_fill(request.cache_context, &job, copied))
        return ggml_cuda_moe_ffn_fill_result::ownership_unresolved;
    return copied ? ggml_cuda_moe_ffn_fill_result::committed : ggml_cuda_moe_ffn_fill_result::rollback_completed;
}

static void ggml_cuda_moe_ffn_fill_worker(ggml_cuda_moe_ffn_cpu_resource_state * resource) {
    for (;;) {
        ggml_cuda_moe_ffn_cpu_resource_state::fill_request request{};
        {
            std::unique_lock<std::mutex> lock(resource->fill_mutex);
            resource->fill_cv.wait(lock, [&]() {
                return resource->fill_stopping || !resource->fill_queue.empty();
            });
            if (resource->fill_stopping) {
                // Return every queued reservation without copying.  A slot in
                // COPYING must be released even when the engine is torn down.
                std::deque<ggml_cuda_moe_ffn_cpu_resource_state::fill_request> cancelled;
                cancelled.swap(resource->fill_queue);
                resource->queued_fill_bytes = 0;
                lock.unlock();
                for (auto & pending : cancelled) {
                    if (!pending.callbacks || !pending.callbacks->finish_fill || !pending.cache_context ||
                            !pending.callbacks->finish_fill(pending.cache_context, &pending.owner_job, false)) {
                        std::lock_guard<std::mutex> stuck_lock(resource->fill_mutex);
                        resource->fill_stuck = true;
                        resource->stuck_fill = std::move(pending);
                        return;
                    }
                    std::lock_guard<std::mutex> stats_lock(resource->fill_mutex);
                    if (resource->fill_outstanding > 0) --resource->fill_outstanding;
                    ++resource->fills_failed;
                }
                return;
            }
            request = std::move(resource->fill_queue.front());
            resource->fill_queue.pop_front();
            resource->queued_fill_bytes -= request.owner_job.total_bytes;
            resource->fill_active = true;
        }

        const auto result = ggml_cuda_moe_ffn_copy_fill_request(*resource, request);
        {
            std::lock_guard<std::mutex> lock(resource->fill_mutex);
            resource->fill_active = false;
            if (result == ggml_cuda_moe_ffn_fill_result::ownership_unresolved) {
                resource->fill_stuck = true;
                resource->stuck_fill = std::move(request);
            } else {
                if (result == ggml_cuda_moe_ffn_fill_result::committed) {
                    ++resource->fills_completed;
                    resource->fill_bytes_completed += request.owner_job.total_bytes;
                }
                else ++resource->fills_failed;
                if (resource->fill_outstanding > 0) --resource->fill_outstanding;
            }
        }
        resource->fill_cv.notify_all();
        if (result == ggml_cuda_moe_ffn_fill_result::ownership_unresolved) return;
    }
}

static bool ggml_cuda_moe_ffn_enqueue_fill(ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_binding & binding, int32_t binding_id, int32_t expert,
        int32_t & accepted_this_op) {
    if (accepted_this_op >= ggml_cuda_moe_ffn_cpu_resource_state::MAX_FILLS_PER_OP) return false;
    const auto * callbacks = binding.cache_callbacks;
    if (!callbacks || !callbacks->reserve_fill || !callbacks->finish_fill) return false;
    {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        if (!resource.accepting_fills || resource.fill_stopping ||
                resource.fill_queue.size() >= ggml_cuda_moe_ffn_cpu_resource_state::MAX_FILL_JOBS) {
            return false;
        }
    }

    ggml_cuda_moe_ffn_fill_job job{};
    const size_t part_count = ggml_cuda_moe_ffn_physical_part_count(binding.layout);
    if (part_count == 0) {
        return false;
    }
    std::vector<ggml_cuda_moe_ffn_fill_part> parts;
    try {
        parts.resize(part_count);
    } catch (...) {
        return false;
    }
    if (!callbacks->reserve_fill(binding.cache_context, binding.layout.device, binding.layout.cache_group,
                binding.layout.cache_layer, binding.generation, expert, &job, parts.data(), parts.size())) {
        return false;
    }
    if (!ggml_cuda_moe_ffn_validate_fill_job(binding, job, expert, parts.data(), parts.size())) {
        if (!callbacks->finish_fill(binding.cache_context, &job, false)) {
            GGML_ABORT("failed to roll back invalid MoE cache fill reservation");
        }
        return false;
    }

    ggml_cuda_moe_ffn_cpu_resource_state::fill_request request{};
    request.owner_job = job;
    request.parts = std::move(parts);
    request.callbacks = callbacks;
    request.cache_context = binding.cache_context;
    request.binding_id = binding_id;
    request.binding_generation = binding.generation;
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        if (resource.accepting_fills && !resource.fill_stopping &&
                resource.fill_queue.size() < ggml_cuda_moe_ffn_cpu_resource_state::MAX_FILL_JOBS &&
                job.total_bytes <= ggml_cuda_moe_ffn_cpu_resource_state::MAX_FILL_QUEUE_BYTES -
                    resource.queued_fill_bytes) {
            resource.fill_queue.push_back(std::move(request));
            resource.queued_fill_bytes += job.total_bytes;
            ++resource.fill_outstanding;
            ++resource.fills_accepted;
            accepted = true;
        }
    }
    if (!accepted) {
        if (!callbacks->finish_fill(binding.cache_context, &job, false)) {
            GGML_ABORT("failed to roll back rejected MoE cache fill reservation");
        }
        return false;
    }
    ++accepted_this_op;
    resource.fill_cv.notify_one();
    return true;
}

static std::shared_ptr<ggml_cuda_moe_ffn_binding_state> ggml_cuda_moe_ffn_lookup(
        ggml_backend_cuda_context & ctx, int32_t binding_id, uint64_t generation) {
    std::lock_guard<std::mutex> lock(ctx.moe_ffn_mutex);
    auto it = ctx.moe_ffn_bindings.find(binding_id);
    if (it == ctx.moe_ffn_bindings.end() || it->second->binding.generation != generation) {
        return {};
    }
    return it->second;
}

static bool ggml_cuda_moe_ffn_op_params(const ggml_tensor * op, ggml_moe_ffn_op_params & params) {
    if (!op || op->op != GGML_OP_MOE_FFN) return false;
    memcpy(&params, op->op_params, sizeof(params));
    return params.device >= 0 && params.binding_id > 0 && params.generation > 0 &&
        params.n_tokens > 0 && params.n_embd > 0 && params.n_ff > 0;
}

static bool ggml_cuda_moe_ffn_shape_matches(const ggml_backend_cuda_context & ctx,
        const ggml_tensor * op, const ggml_moe_ffn_op_params & params,
        const ggml_cuda_moe_ffn_binding & b) {
    ggml_moe_ffn_layout graph = b.layout;
    graph.activation = op->src[0];
    graph.selected_ids = op->src[1];
    graph.route_weights = op->src[2];
    graph.n_tokens = params.n_tokens;
    return params.device == ctx.device && b.layout.device == ctx.device &&
        params.n_embd == b.layout.n_embd && params.n_ff == b.layout.n_ff &&
        params.n_tokens > 0 && params.n_tokens <= b.layout.token_capacity &&
        params.unary_op == (int32_t) b.layout.unary_op && params.limit == b.layout.limit &&
        op->type == GGML_TYPE_F32 && op->ne[0] == b.layout.n_embd && op->ne[1] == params.n_tokens &&
        op->ne[2] == 1 && op->ne[3] == 1 && ggml_moe_ffn_graph_inputs_supported(&graph) &&
        ggml_is_contiguous(op);
}

static bool ggml_cuda_moe_ffn_resource_supports_signature(
        const ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_moe_ffn_layout & layout) {
    if (layout.n_embd > resource.resource.n_embd || layout.n_ff > resource.resource.n_ff ||
            layout.n_experts_used > resource.resource.n_experts_used ||
            layout.token_capacity != resource.resource.token_capacity) return false;
    return std::any_of(resource.signatures.begin(), resource.signatures.end(), [&](const auto & signature) {
        return signature.n_embd == layout.n_embd && signature.n_ff == layout.n_ff &&
            signature.n_experts_used == layout.n_experts_used &&
            signature.token_capacity == layout.token_capacity;
    });
}

struct ggml_cuda_moe_ffn_validated_op {
    ggml_moe_ffn_op_params params{};
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> state;
    // Holds the binding/resource stable through preflight and execution.
    std::unique_lock<std::mutex> lock;
};

static bool ggml_cuda_moe_ffn_validate_op(ggml_backend_cuda_context & ctx, const ggml_tensor * op,
        bool require_device_buffers, ggml_cuda_moe_ffn_validated_op & validated) {
    ggml_moe_ffn_op_params params{};
    if (!ggml_cuda_moe_ffn_op_params(op, params) || params.device != ctx.device ||
            !op->src[0] || !op->src[1] || !op->src[2]) return false;
    auto state = ggml_cuda_moe_ffn_lookup(ctx, params.binding_id, params.generation);
    if (!state) return false;
    std::unique_lock<std::mutex> lock(state->cpu_resource->execution_mutex);
    if (!state->active || !ggml_cuda_moe_ffn_shape_matches(ctx, op, params, state->binding)) return false;
    // supports_op precedes allocation; graph validation additionally checks device-buffer placement.
    if (require_device_buffers &&
            (!ggml_cuda_moe_ffn_tensor_on_device(op->src[0], ctx.device) ||
             !ggml_cuda_moe_ffn_tensor_on_device(op->src[1], ctx.device) ||
             !ggml_cuda_moe_ffn_tensor_on_device(op->src[2], ctx.device) ||
             !ggml_cuda_moe_ffn_tensor_on_device(op, ctx.device))) return false;
    validated.params = params;
    validated.state = std::move(state);
    validated.lock = std::move(lock);
    return true;
}

bool ggml_cuda_moe_ffn_supports_op(ggml_backend_cuda_context & ctx, const ggml_tensor * op) {
    ggml_cuda_moe_ffn_validated_op validated;
    return ggml_cuda_moe_ffn_validate_op(ctx, op, false, validated);
}

static bool ggml_cuda_moe_ffn_cpu_draws(const ggml_cuda_moe_ffn_binding & b,
        ggml_cuda_moe_ffn_cpu_resource_state & resource, bool misses_only,
        size_t activation_row_bytes, int32_t n_tokens) {
    const size_t output_row_bytes = size_t(b.layout.n_embd) * sizeof(float);
    memset(resource.host_output, 0, output_row_bytes * (size_t) n_tokens);
    bool success = true;
    for (int32_t token = 0; token < n_tokens && success; ++token) {
        float * total = resource.host_output + size_t(token) * b.layout.n_embd;
        memcpy(b.cpu_activations->data,
                resource.host_activations + size_t(token) * b.layout.n_embd, activation_row_bytes);
        for (int32_t selection = 0; selection < b.layout.n_experts_used; ++selection) {
            const size_t draw = size_t(token) * b.layout.n_experts_used + selection;
            if (misses_only && resource.host_slots[draw] >= 0) continue;
            *(int32_t *) b.cpu_ids->data = resource.host_ids[draw];
            *(float *) b.cpu_route_weights->data = resource.host_routes[draw];
            if (ggml_backend_graph_compute(resource.resource.cpu_backend, b.cpu_graph) != GGML_STATUS_SUCCESS) {
                success = false;
                break;
            }
            const float * contribution = (const float *) b.cpu_output->data;
            for (int32_t j = 0; j < b.layout.n_embd; ++j) total[j] += contribution[j];
        }
    }
    return success;
}

// F16 experts bypass the quantized MMV path, which always feeds a Q8_1
// activation to the kernel.  The arithmetic below mirrors mul_mat_vec_q()'s
// post-dot activation so F16 layers behave identically to quantized ones.
static __device__ __forceinline__ float ggml_cuda_moe_ffn_activate(
        float gate, float up, ggml_unary_op unary_op, float limit) {
    // mmvq_args treats a non-positive limit as "unlimited"; mirror that here.
    if (!(limit > 1e-6f)) limit = INFINITY;
    if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
        constexpr float alpha = 1.702f;
        constexpr float swiglu_limit = 7.0f;
        gate = fminf(gate, swiglu_limit);
        up = fmaxf(fminf(up, swiglu_limit), -swiglu_limit);
        return gate / (1.0f + expf(-gate * alpha)) * (1.0f + up);
    }
    switch (unary_op) {
        case GGML_UNARY_OP_SILU:
            gate = gate / (1.0f + expf(-gate));
            gate = fminf(gate, limit);
            return fmaxf(-limit, fminf(limit, up)) * gate;
        case GGML_UNARY_OP_RELU:
            return fmaxf(gate, 0.0f) * up;
        case GGML_UNARY_OP_GELU: {
            constexpr float GELU_COEF_A    = 0.044715f;
            constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
            return 0.5f*gate*up*(1.0f + tanhf(SQRT_2_OVER_PI*gate*(1.0f + GELU_COEF_A*gate*gate)));
        }
        default:
            return 0.0f;
    }
}

// One warp computes one output row so the half loads along k stay coalesced.
template <int WARPS_PER_BLOCK>
static __global__ void ggml_cuda_moe_ffn_f16_up_gate_kernel(
        const half * up, const half * gate, const float * activation, const int32_t * slots,
        float * output, int64_t n_embd, int64_t n_ff, size_t up_expert_bytes,
        size_t gate_expert_bytes, ggml_unary_op unary_op, float limit, int n_hits) {
    const int hit = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = (int64_t) blockIdx.x * WARPS_PER_BLOCK + warp;
    if (hit >= n_hits || row >= n_ff) return;

    const int32_t slot = slots[hit];
    const half * up_row = (const half *) ((const char *) up + (size_t) slot * up_expert_bytes) +
        (size_t) row * (size_t) n_embd;
    const half * gate_row = (const half *) ((const char *) gate + (size_t) slot * gate_expert_bytes) +
        (size_t) row * (size_t) n_embd;
    float sum_up = 0.0f, sum_gate = 0.0f;
    for (int64_t i = lane; i < n_embd; i += WARP_SIZE) {
        const float x = activation[i];
        sum_up = fmaf(__half2float(up_row[i]), x, sum_up);
        sum_gate = fmaf(__half2float(gate_row[i]), x, sum_gate);
    }
    sum_up = warp_reduce_sum(sum_up);
    sum_gate = warp_reduce_sum(sum_gate);
    if (lane == 0) {
        output[(size_t) hit * (size_t) n_ff + (size_t) row] =
            ggml_cuda_moe_ffn_activate(sum_gate, sum_up, unary_op, limit);
    }
}

template <int WARPS_PER_BLOCK>
static __global__ void ggml_cuda_moe_ffn_f16_down_kernel(
        const half * down, const float * hidden, const int32_t * slots, float * output,
        int64_t n_ff, int64_t n_embd, size_t down_expert_bytes, int n_hits) {
    const int hit = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = (int64_t) blockIdx.x * WARPS_PER_BLOCK + warp;
    if (hit >= n_hits || row >= n_embd) return;

    const int32_t slot = slots[hit];
    const half * down_row = (const half *) ((const char *) down + (size_t) slot * down_expert_bytes) +
        (size_t) row * (size_t) n_ff;
    const float * x = hidden + (size_t) hit * (size_t) n_ff;
    float sum = 0.0f;
    for (int64_t i = lane; i < n_ff; i += WARP_SIZE) {
        sum = fmaf(__half2float(down_row[i]), x[i], sum);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[(size_t) hit * (size_t) n_embd + (size_t) row] = sum;
    }
}

constexpr int GGML_CUDA_MOE_FFN_F16_WARPS = 8;

static bool ggml_cuda_moe_ffn_enqueue_hit_rows(ggml_backend_cuda_context & ctx,
        const ggml_tensor * op, const ggml_cuda_moe_ffn_binding & b,
        ggml_cuda_moe_ffn_cpu_resource_state & resource, int32_t token,
        const void * activation_data, int32_t n_hits, bool & kernels_started) {
    kernels_started = false;
    if (n_hits <= 0 || n_hits > b.layout.n_experts_used) return false;
    cudaStream_t stream = ctx.stream();
    const size_t slots_bytes = size_t(n_hits) * sizeof(int32_t);
    const size_t draw_offset = size_t(token) * b.layout.n_experts_used;
    const size_t up_row_bytes = resource.device_up_input_q8_1_bytes / resource.draw_capacity;
    const size_t down_row_bytes = resource.device_down_input_q8_1_bytes / resource.draw_capacity;
    if (cudaMemcpyAsync(resource.device_slots + draw_offset,
                resource.host_hit_slots + draw_offset, slots_bytes,
                cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;

    const int64_t up_padded = GGML_PAD(b.layout.n_embd, MATRIX_ROW_PADDING);
    const bool up_gate_f16 = b.layout.up->type == GGML_TYPE_F16;
    ggml_tensor ids = *op->src[1];
    ids.ne[0] = n_hits;
    ids.ne[1] = ids.ne[2] = ids.ne[3] = 1;
    ids.nb[0] = sizeof(int32_t);
    ids.data = resource.device_slots + draw_offset;
    ggml_tensor activation = *op->src[0];
    activation.data = const_cast<void *>(activation_data);
    activation.ne[1] = activation.ne[2] = activation.ne[3] = 1;
    activation.nb[1] = activation.nb[2] = activation.nb[3] = 0;
    ggml_tensor hidden{};
    hidden.type = GGML_TYPE_F32;
    hidden.ne[0] = b.layout.n_ff; hidden.ne[1] = 1; hidden.ne[2] = n_hits; hidden.ne[3] = 1;
    hidden.nb[0] = sizeof(float); hidden.nb[1] = b.layout.n_ff * sizeof(float);
    hidden.nb[2] = b.layout.n_ff * sizeof(float);
    hidden.nb[3] = (int64_t) b.layout.n_ff * n_hits * sizeof(float);
    hidden.data = resource.device_hidden + draw_offset * b.layout.n_ff;
    if (up_gate_f16) {
        kernels_started = true;
        const dim3 blocks((unsigned) ((b.layout.n_ff + GGML_CUDA_MOE_FFN_F16_WARPS - 1) /
                    GGML_CUDA_MOE_FFN_F16_WARPS), (unsigned) n_hits, 1);
        ggml_cuda_moe_ffn_f16_up_gate_kernel<GGML_CUDA_MOE_FFN_F16_WARPS>
            <<<blocks, GGML_CUDA_MOE_FFN_F16_WARPS * WARP_SIZE, 0, stream>>>(
                (const half *) b.layout.bank_up->data, (const half *) b.layout.bank_gate->data,
                (const float *) activation_data, resource.device_slots + draw_offset,
                resource.device_hidden + draw_offset * b.layout.n_ff,
                b.layout.n_embd, b.layout.n_ff, b.layout.bank_up->nb[2], b.layout.bank_gate->nb[2],
                b.layout.unary_op, b.layout.limit, n_hits);
        CUDA_CHECK(cudaGetLastError());
    } else {
        quantize_row_q8_1_cuda((const float *) activation_data,
                (char *) resource.device_up_input_q8_1 + draw_offset * up_row_bytes,
                b.layout.n_embd, 1, 1, up_padded, b.layout.up->type, stream);
        kernels_started = true;
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, b.layout.bank_up, &activation, &ids, &hidden, nullptr, nullptr,
                (const char *) b.layout.bank_up->data, (const char *) b.layout.bank_gate->data, nullptr,
                (const char *) resource.device_up_input_q8_1 + draw_offset * up_row_bytes,
                resource.device_hidden + draw_offset * b.layout.n_ff, 0, b.layout.n_ff, 1,
                up_padded, b.layout.unary_op, b.layout.limit, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t down_padded = GGML_PAD(b.layout.n_ff, MATRIX_ROW_PADDING);
    const size_t q8_per_expert = (size_t) GGML_PAD((int64_t) b.layout.n_ff, MATRIX_ROW_PADDING) /
        QK8_1 * sizeof(block_q8_1);
    const size_t down_offset = size_t(token) * (b.layout.n_experts_used + 1) * b.layout.n_embd;
    if (b.layout.down->type == GGML_TYPE_F16) {
        const dim3 blocks((unsigned) ((b.layout.n_embd + GGML_CUDA_MOE_FFN_F16_WARPS - 1) /
                    GGML_CUDA_MOE_FFN_F16_WARPS), (unsigned) n_hits, 1);
        ggml_cuda_moe_ffn_f16_down_kernel<GGML_CUDA_MOE_FFN_F16_WARPS>
            <<<blocks, GGML_CUDA_MOE_FFN_F16_WARPS * WARP_SIZE, 0, stream>>>(
                (const half *) b.layout.bank_down->data,
                resource.device_hidden + draw_offset * b.layout.n_ff,
                resource.device_slots + draw_offset, resource.device_down + down_offset,
                b.layout.n_ff, b.layout.n_embd, b.layout.bank_down->nb[2], n_hits);
        CUDA_CHECK(cudaGetLastError());
    } else {
        quantize_row_q8_1_cuda(resource.device_hidden + draw_offset * b.layout.n_ff,
                (char *) resource.device_down_input_q8_1 + draw_offset * down_row_bytes,
                b.layout.n_ff, n_hits, 1, down_padded, b.layout.down->type, stream);
        CUDA_CHECK(cudaGetLastError());
        ggml_tensor down_input = hidden;
        down_input.ne[1] = 1;
        down_input.ne[2] = 1;
        down_input.nb[1] = q8_per_expert;
        down_input.nb[2] = q8_per_expert;
        ggml_tensor down_output{};
        down_output.type = GGML_TYPE_F32;
        down_output.ne[0] = b.layout.n_embd; down_output.ne[1] = 1;
        down_output.ne[2] = n_hits; down_output.ne[3] = 1;
        down_output.nb[0] = sizeof(float); down_output.nb[1] = b.layout.n_embd * sizeof(float);
        down_output.nb[2] = b.layout.n_embd * sizeof(float);
        down_output.nb[3] = (int64_t) b.layout.n_embd * down_output.ne[2] * sizeof(float);
        down_output.data = resource.device_down + down_offset;
        ggml_cuda_op_mul_mat_vec_q_id(ctx, b.layout.bank_down, &down_input, &ids, &down_output, nullptr,
                (const char *) b.layout.bank_down->data, nullptr,
                (const char *) resource.device_down_input_q8_1 + draw_offset * down_row_bytes,
                resource.device_down + down_offset, 0, b.layout.n_embd, 1, down_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

static bool ggml_cuda_moe_ffn_reduce_commit(ggml_backend_cuda_context & ctx, ggml_tensor * op,
        const ggml_cuda_moe_ffn_binding & b, ggml_cuda_moe_ffn_cpu_resource_state & resource,
        int32_t token, int32_t n_hits, bool include_cpu_partial, bool routes_staged) {
    cudaStream_t stream = ctx.stream();
    const int32_t n_rows = n_hits + (include_cpu_partial ? 1 : 0);
    if (n_rows <= 0 || n_rows > b.layout.n_experts_used + 1) return false;
    const size_t row_offset = size_t(token) * (b.layout.n_experts_used + 1);
    const size_t route_offset = row_offset;
    if (include_cpu_partial) {
        const size_t offset = (row_offset + n_hits) * b.layout.n_embd * sizeof(float);
        const size_t bytes = size_t(b.layout.n_embd) * sizeof(float);
        const float * cpu_partial = resource.host_output + size_t(token) * b.layout.n_embd;
        resource.host_hit_routes[route_offset + n_hits] = 1.0f;
        if (cudaMemcpyAsync((char *) resource.device_down + offset, cpu_partial,
                    bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    }
    if (routes_staged) {
        const size_t route_bytes = size_t(n_rows) * sizeof(float);
        if (cudaMemcpyAsync(resource.device_routes + route_offset,
                    resource.host_hit_routes + route_offset, route_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    }
    ggml_tensor reduce_input{};
    reduce_input.type = GGML_TYPE_F32;
    reduce_input.ne[0] = b.layout.n_embd; reduce_input.ne[1] = n_rows;
    reduce_input.ne[2] = reduce_input.ne[3] = 1;
    reduce_input.nb[0] = sizeof(float); reduce_input.nb[1] = b.layout.n_embd * sizeof(float);
    reduce_input.nb[2] = (int64_t) b.layout.n_embd * n_rows * sizeof(float);
    reduce_input.nb[3] = reduce_input.nb[2];
    reduce_input.data = resource.device_down + row_offset * b.layout.n_embd;
    ggml_tensor routes = *op->src[2];
    routes.ne[0] = 1; routes.ne[1] = n_rows; routes.ne[2] = routes.ne[3] = 1;
    routes.nb[0] = sizeof(float); routes.nb[1] = sizeof(float);
    routes.nb[2] = routes.nb[3] = n_rows * sizeof(float);
    routes.data = routes_staged ? (void *) (resource.device_routes + route_offset) :
        (char *) op->src[2]->data + size_t(token) * op->src[2]->nb[2];
    ggml_tensor reduce_output{};
    reduce_output.type = GGML_TYPE_F32;
    reduce_output.ne[0] = b.layout.n_embd; reduce_output.ne[1] = 1;
    reduce_output.ne[2] = reduce_output.ne[3] = 1;
    reduce_output.nb[0] = sizeof(float); reduce_output.nb[1] = b.layout.n_embd * sizeof(float);
    reduce_output.nb[2] = reduce_output.nb[3] = reduce_output.nb[1];
    reduce_output.data = resource.device_output + size_t(token) * b.layout.n_embd;
    reduce_output.src[0] = &reduce_input;
    reduce_output.src[1] = &routes;
    ggml_cuda_op_mul_multi_add(ctx, &reduce_output);
    CUDA_CHECK(cudaGetLastError());
    const size_t output_bytes = size_t(b.layout.n_embd) * sizeof(float);
    if (cudaMemcpyAsync((char *) op->data + size_t(token) * op->nb[1],
                reduce_output.data, output_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return false;
    return true;
}

static bool ggml_cuda_moe_ffn_release_pin_set(const ggml_cuda_moe_ffn_cache_callbacks * callbacks,
        void * context, const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    if (n_pins == 0) return true;
    return callbacks && callbacks->release_pins && callbacks->release_pins(context, pins, n_pins);
}

static bool ggml_cuda_moe_ffn_release_pending_pins(ggml_cuda_moe_ffn_cpu_resource_state & resource) {
    auto & pending = resource.pending_pins;
    if (!pending.active) return true;
    if (!ggml_cuda_moe_ffn_release_pin_set(pending.callbacks, pending.context, pending.pins.data(),
                (int32_t) pending.pins.size())) {
        return false;
    }
    pending = {};
    return true;
}

static bool ggml_cuda_moe_ffn_checked_mul(size_t a, size_t b, size_t & result);
static bool ggml_cuda_moe_ffn_validate_resolution(const ggml_cuda_moe_ffn_binding & binding,
        const ggml_cuda_moe_ffn_resolution & resolved, int32_t expected_draws) {
    if (expected_draws <= 0 || resolved.n_draws != expected_draws || resolved.n_draws < 0 ||
            !resolved.draw_slots || resolved.draw_capacity < (size_t) resolved.n_draws ||
            resolved.n_pins < 0 || resolved.n_pins > resolved.n_draws || !resolved.pins ||
            resolved.pin_capacity < (size_t) resolved.n_pins) return false;
    int32_t unique_hits = 0;
    for (int32_t i = 0; i < resolved.n_draws; ++i) {
        const int32_t slot = resolved.draw_slots[i];
        if (slot < -1 || slot >= binding.layout.n_slots) return false;
        if (slot < 0) continue;
        bool seen = false;
        for (int32_t j = 0; j < i; ++j) seen |= resolved.draw_slots[j] == slot;
        if (!seen) ++unique_hits;
    }
    if (unique_hits != resolved.n_pins) return false;
    for (int32_t i = 0; i < resolved.n_pins; ++i) {
        const auto & pin = resolved.pins[i];
        if (!pin.owner || pin.device != binding.layout.device || pin.group != binding.layout.cache_group ||
                pin.layer != binding.layout.cache_layer || pin.owner_generation != binding.generation ||
                pin.slot < 0 || pin.slot >= binding.layout.n_slots ||
                pin.generation == 0) return false;
        for (int32_t j = 0; j < i; ++j) {
            if (resolved.pins[j].slot == pin.slot) return false;
        }
        bool used = false;
        for (int32_t j = 0; j < resolved.n_draws; ++j) used |= resolved.draw_slots[j] == pin.slot;
        if (!used) return false;
    }
    return true;
}

static void ggml_cuda_moe_ffn_store_pending_pins(ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_binding & binding, const ggml_cuda_moe_ffn_resolution & resolved) {
    auto & pending = resource.pending_pins;
    GGML_ASSERT(!pending.active);
    if (resolved.n_pins == 0) return;
    pending.callbacks = binding.cache_callbacks;
    pending.context = binding.cache_context;
    pending.pins.assign(resolved.pins, resolved.pins + resolved.n_pins);
    pending.active = true;
}

bool ggml_cuda_moe_ffn_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    ggml_cuda_moe_ffn_validated_op validated;
    if (!ggml_cuda_moe_ffn_validate_op(ctx, op, false, validated)) return false;
    const auto & params = validated.params;
    const auto & state = validated.state;
    auto & resource = *state->cpu_resource;
    auto & b = state->binding;
    auto & stats = state->stats;
    cudaStream_t stream = ctx.stream();
    const int32_t n_tokens = params.n_tokens;
    const int32_t top_k = b.layout.n_experts_used;
    size_t n_draws = 0, activation_row_bytes = 0, activation_bytes = 0, ids_bytes = 0, route_bytes = 0;
    if (!ggml_cuda_moe_ffn_checked_mul((size_t) n_tokens, (size_t) top_k, n_draws) ||
            !ggml_cuda_moe_ffn_checked_mul((size_t) b.layout.n_embd, sizeof(float), activation_row_bytes) ||
            !ggml_cuda_moe_ffn_checked_mul(activation_row_bytes, (size_t) n_tokens, activation_bytes) ||
            !ggml_cuda_moe_ffn_checked_mul(n_draws, sizeof(int32_t), ids_bytes) ||
            !ggml_cuda_moe_ffn_checked_mul(n_draws, sizeof(float), route_bytes)) return false;
    const size_t output_bytes = activation_bytes;
    if (n_draws == 0 || n_draws > resource.draw_capacity || n_draws > INT32_MAX) return false;
    if (cudaMemcpy2DAsync(resource.host_ids, ids_bytes / (size_t) n_tokens,
                op->src[1]->data, op->src[1]->nb[1], ids_bytes / (size_t) n_tokens,
                (size_t) n_tokens,
                cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess ||
            !ggml_cuda_moe_ffn_release_pending_pins(resource)) return false;

    ggml_cuda_moe_ffn_resolution resolved{};
    resolved.draw_slots = resource.host_slots;
    resolved.draw_capacity = n_draws;
    resolved.pins = resource.resolved_pins.data();
    resolved.pin_capacity = n_draws;
    const auto * const draw_storage = resolved.draw_slots;
    auto * const pin_storage = resolved.pins;
    const size_t draw_capacity = resolved.draw_capacity;
    const size_t pin_capacity = resolved.pin_capacity;
    const bool resolved_ok = b.cache_callbacks->resolve_and_pin(b.cache_context, b.layout.device,
            b.layout.cache_group, b.layout.cache_layer, b.generation, resource.host_ids,
            (int32_t) n_draws, &resolved);
    if (!resolved_ok || resolved.draw_slots != draw_storage || resolved.pins != pin_storage ||
            resolved.draw_capacity != draw_capacity || resolved.pin_capacity != pin_capacity) {
        const int32_t acquired = std::clamp<int32_t>(resolved.n_pins, 0, (int32_t) pin_capacity);
        if (acquired > 0 && !ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks,
                    b.cache_context, pin_storage, acquired)) {
            GGML_ABORT("failed to release malformed MoE cache pin result");
        }
        return false;
    }
    const auto release_current_pins = [&]() {
        return ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks, b.cache_context,
                resolved.pins, resolved.n_pins);
    };
    if (!ggml_cuda_moe_ffn_validate_resolution(b, resolved, (int32_t) n_draws)) {
        const int32_t acquired = std::clamp<int32_t>(resolved.n_pins, 0, (int32_t) pin_capacity);
        if (acquired > 0 && !ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks,
                    b.cache_context, pin_storage, acquired))
            GGML_ABORT("failed to release invalid MoE cache pin result");
        return false;
    }
    int32_t n_hits = 0;
    for (size_t i = 0; i < n_draws; ++i) n_hits += resolved.draw_slots[i] >= 0;
    const int32_t n_misses = (int32_t) n_draws - n_hits;
    resource.fill_candidates.clear();
    for (size_t i = 0; i < n_draws; ++i) {
        if (resolved.draw_slots[i] >= 0) continue;
        resource.fill_candidates.push_back({resource.host_ids[i], i});
    }
    std::sort(resource.fill_candidates.begin(), resource.fill_candidates.end(), [](const auto & a, const auto & b) {
        return a.expert != b.expert ? a.expert < b.expert : a.first_draw < b.first_draw;
    });
    resource.fill_candidates.erase(std::unique(resource.fill_candidates.begin(), resource.fill_candidates.end(),
                [](const auto & a, const auto & b) { return a.expert == b.expert; }),
            resource.fill_candidates.end());
    std::sort(resource.fill_candidates.begin(), resource.fill_candidates.end(), [](const auto & a, const auto & b) {
        return a.first_draw < b.first_draw;
    });
    // Admit at most a handful of misses per routed FFN.  A low hit-rate op
    // touches many distinct experts at once; filling them all would let one
    // token evict the entire resident bank before the next layer reuses it.
    int32_t accepted_fills = 0;
    for (const auto & candidate : resource.fill_candidates)
        ggml_cuda_moe_ffn_enqueue_fill(resource, b, params.binding_id, candidate.expert, accepted_fills);

    bool cpu_inputs_ready = false;
    const auto read_cpu_inputs = [&]() {
        if (cudaMemcpyAsync(resource.host_activations, op->src[0]->data, activation_bytes,
                    cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(resource.host_routes, op->src[2]->data, route_bytes,
                    cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }
        cpu_inputs_ready = true;
        return true;
    };
    if (n_misses > 0 && !read_cpu_inputs()) {
        if (!release_current_pins()) GGML_ABORT("failed to release MoE cache pins before hit dispatch");
        return false;
    }
    if (n_misses == 0) {
        ++stats.all_hit_calls;
    } else {
        ++stats.any_miss_calls;
    }
    stats.gpu_hit_draws += n_hits;
    stats.cpu_miss_draws += n_misses;

    const auto complete_cpu_fallback = [&]() {
        if (!cpu_inputs_ready) {
            if (!read_cpu_inputs()) return false;
        }
        if (!ggml_cuda_moe_ffn_cpu_draws(b, resource, false, activation_row_bytes, n_tokens)) return false;
        if (cudaMemcpyAsync(op->data, resource.host_output, output_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        return true;
    };
    // Keep cache pins until submitted bank reads finish before CPU fallback can reuse slot data.
    const auto fallback_after_hit_work = [&](bool submitted) {
        return (!submitted || cudaStreamSynchronize(stream) == cudaSuccess) && complete_cpu_fallback();
    };

    if (n_hits == 0) {
        if (!release_current_pins()) GGML_ABORT("failed to release pins for all-miss MoE operation");
        if (!complete_cpu_fallback()) return false;
        return true;
    }

    // Prepare the packed hit rows before dispatch, the CPU miss graphs below can then overlap the queued bank reads.
    for (int32_t token = 0; token < n_tokens; ++token) {
        int32_t token_hits = 0;
        const size_t token_draw_offset = size_t(token) * top_k;
        for (int32_t selection = 0; selection < top_k; ++selection) {
            const size_t draw = token_draw_offset + selection;
            if (resolved.draw_slots[draw] >= 0) {
                resource.host_hit_slots[token_draw_offset + token_hits] = resolved.draw_slots[draw];
                if (n_misses > 0)
                    resource.host_hit_routes[token * (top_k + 1) + token_hits] = resource.host_routes[draw];
                ++token_hits;
            }
        }
    }

    bool kernels_started = false;
    bool dispatch_ok = true;
    for (int32_t token = 0; token < n_tokens && dispatch_ok; ++token) {
        int32_t token_hits = 0;
        const size_t token_draw_offset = size_t(token) * top_k;
        for (int32_t selection = 0; selection < top_k; ++selection)
            token_hits += resolved.draw_slots[token_draw_offset + selection] >= 0;
        if (token_hits > 0) {
            const void * activation_data = (const char *) op->src[0]->data + size_t(token) * op->src[0]->nb[1];
            bool token_kernels_started = false;
            dispatch_ok = ggml_cuda_moe_ffn_enqueue_hit_rows(ctx, op, b, resource, token,
                    activation_data, token_hits, token_kernels_started);
            kernels_started |= token_kernels_started;
        }
    }

    if (kernels_started) ggml_cuda_moe_ffn_store_pending_pins(resource, b, resolved);
    else if (!release_current_pins()) GGML_ABORT("failed to release MoE cache pins after pre-dispatch rejection");
    if (!dispatch_ok) return fallback_after_hit_work(kernels_started);

    if (n_misses > 0 && !ggml_cuda_moe_ffn_cpu_draws(b, resource, true, activation_row_bytes, n_tokens)) {
        if (kernels_started) {
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return false;
            if (!ggml_cuda_moe_ffn_release_pending_pins(resource))
                GGML_ABORT("failed to release MoE cache pins after CPU miss execution failed");
        }
        return false;
    }

    for (int32_t token = 0; token < n_tokens && dispatch_ok; ++token) {
        int32_t token_hits = 0, token_misses = 0;
        const size_t token_draw_offset = size_t(token) * top_k;
        for (int32_t selection = 0; selection < top_k; ++selection) {
            if (resolved.draw_slots[token_draw_offset + selection] >= 0) ++token_hits;
            else ++token_misses;
        }
        dispatch_ok = ggml_cuda_moe_ffn_reduce_commit(ctx, op, b, resource,
                token, token_hits, token_misses > 0, n_misses > 0);
    }

    if (!dispatch_ok) return fallback_after_hit_work(kernels_started);

    if (n_misses > 0 && n_hits > 0) {
        if (cudaEventRecord(resource.hit_kernels_done, stream) != cudaSuccess) return fallback_after_hit_work(true);
        const cudaError_t hit_state = cudaEventQuery(resource.hit_kernels_done);
        if (hit_state != cudaSuccess && hit_state != cudaErrorNotReady)
            return fallback_after_hit_work(true);
    }
    return true;
}

bool ggml_cuda_moe_ffn_validate_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MOE_FFN) {
            ggml_cuda_moe_ffn_validated_op validated;
            if (!ggml_cuda_moe_ffn_validate_op(ctx, node, true, validated)) return false;
        }
    }
    return true;
}


static bool ggml_cuda_moe_ffn_host_tensor(const ggml_tensor * tensor) {
    return tensor && tensor->data && (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer));
}

static bool ggml_cuda_moe_ffn_device_tensor(const ggml_tensor * tensor, int device) {
    return tensor && tensor->data && ggml_cuda_moe_ffn_tensor_on_device(tensor, device);
}

struct ggml_cuda_moe_ffn_workspace_sizes {
    size_t ids, activation, route, output, host_stage;
    size_t up_input_q8_1, down_input_q8_1, hidden, down, device_routes;
};

static bool ggml_cuda_moe_ffn_checked_mul(size_t a, size_t b, size_t & result) {
    if (b && a > SIZE_MAX / b) return false;
    result = a * b;
    return true;
}

static bool ggml_cuda_moe_ffn_checked_add(size_t a, size_t b, size_t & result) {
    if (a > SIZE_MAX - b) return false;
    result = a + b;
    return true;
}

static bool ggml_cuda_moe_ffn_get_workspace_sizes(int32_t n_embd, int32_t n_ff,
        int32_t n_experts_used, int32_t token_capacity, ggml_cuda_moe_ffn_workspace_sizes & sizes) {
    if (n_embd <= 0 || n_ff <= 0 || n_experts_used <= 0 || token_capacity <= 0 ||
            n_experts_used == INT32_MAX || n_embd % QK8_1 != 0 || n_ff % QK8_1 != 0) return false;

    const int64_t up_padded = GGML_PAD((int64_t) n_embd, MATRIX_ROW_PADDING);
    const int64_t down_padded = GGML_PAD((int64_t) n_ff, MATRIX_ROW_PADDING);
    if (up_padded > INT_MAX || down_padded > INT_MAX) return false;

    size_t draws = 0, up_input_q8_1_row = 0, down_input_q8_1_row = 0;
    if (!ggml_cuda_moe_ffn_checked_mul((size_t) n_experts_used, (size_t) token_capacity, draws) ||
            draws > INT32_MAX) return false;
    if (!ggml_cuda_moe_ffn_checked_mul((size_t) up_padded / QK8_1,
                sizeof(block_q8_1), up_input_q8_1_row) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) down_padded / QK8_1,
                sizeof(block_q8_1), down_input_q8_1_row) ||
        !ggml_cuda_moe_ffn_checked_mul(draws, sizeof(int32_t), sizes.ids) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_embd, (size_t) token_capacity, sizes.activation) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.activation, sizeof(float), sizes.activation) ||
        !ggml_cuda_moe_ffn_checked_mul(draws, sizeof(float), sizes.route)) return false;
    sizes.output = sizes.activation;
    size_t ids_staging = 0, routes_staging = 0, cpu_route_row = 0, host_base = 0;
    if (!ggml_cuda_moe_ffn_checked_mul(sizes.ids, 3, ids_staging) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.route, 2, routes_staging) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) token_capacity, sizeof(float), cpu_route_row) ||
        !ggml_cuda_moe_ffn_checked_add(routes_staging, cpu_route_row, routes_staging) ||
        !ggml_cuda_moe_ffn_checked_add(ids_staging, sizes.activation, host_base) ||
        !ggml_cuda_moe_ffn_checked_add(host_base, routes_staging, host_base) ||
        !ggml_cuda_moe_ffn_checked_add(host_base, sizes.output, sizes.host_stage) ||
        !ggml_cuda_moe_ffn_checked_mul(down_input_q8_1_row, draws, sizes.down_input_q8_1) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_ff, draws, sizes.hidden) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.hidden, sizeof(float), sizes.hidden) ||
        !ggml_cuda_moe_ffn_checked_add(draws, (size_t) token_capacity, sizes.down) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_embd, sizes.down, sizes.down) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.down, sizeof(float), sizes.down) ||
        !ggml_cuda_moe_ffn_checked_add(draws, (size_t) token_capacity, sizes.device_routes) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.device_routes, sizeof(float), sizes.device_routes) ||
        !ggml_cuda_moe_ffn_checked_mul(up_input_q8_1_row, draws, sizes.up_input_q8_1)) return false;
    return true;
}

// F16 experts are executed by the executor's own GEMV kernels; every other
// type still goes through the quantized MMV path.
static bool ggml_cuda_moe_ffn_type_supported(ggml_type type) {
    return type == GGML_TYPE_F16 || ggml_cuda_mmvq_type_supported(type);
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_layout_supported(ggml_backend_t backend,
        const ggml_moe_ffn_layout * layout) {
    if (!ggml_backend_is_cuda(backend) || !layout) return false;
    const auto * ctx = (const ggml_backend_cuda_context *) backend->context;
    ggml_cuda_moe_ffn_workspace_sizes sizes{};
    return layout->owner_backend == backend && (layout->device < 0 || layout->device == ctx->device) &&
        ggml_moe_ffn_layout_supported(layout) && layout->up->type == layout->gate->type &&
        ggml_cuda_moe_ffn_type_supported(layout->up->type) &&
        ggml_cuda_moe_ffn_type_supported(layout->down->type) &&
        ggml_cuda_moe_ffn_get_workspace_sizes(layout->n_embd, layout->n_ff,
                layout->n_experts_used, layout->token_capacity, sizes);
}

GGML_CALL int32_t ggml_backend_cuda_moe_ffn_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_binding * binding) {
    if (!ggml_backend_is_cuda(backend) || !binding) return -1;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    const ggml_moe_ffn_layout & layout = binding->layout;
    if (layout.device != ctx->device || layout.owner_backend != backend ||
        layout.cache_owner != binding->cache_context || binding->generation == 0 ||
        !ggml_backend_cuda_moe_ffn_layout_supported(backend, &layout) || layout.n_slots <= 0 ||
        !binding->cache_callbacks || !binding->cache_callbacks->resolve_and_pin ||
        !binding->cache_callbacks->release_pins || !binding->cache_context ||
        binding->cpu_resource_id <= 0 || !binding->cpu_graph ||
        !binding->cpu_activations || !binding->cpu_ids || !binding->cpu_route_weights || !binding->cpu_output) return -1;

    if (!ggml_cuda_moe_ffn_host_tensor(layout.up) || !ggml_cuda_moe_ffn_host_tensor(layout.gate) ||
        !ggml_cuda_moe_ffn_host_tensor(layout.down) || !ggml_cuda_moe_ffn_device_tensor(layout.bank_up, ctx->device) ||
        !ggml_cuda_moe_ffn_device_tensor(layout.bank_gate, ctx->device) ||
        !ggml_cuda_moe_ffn_device_tensor(layout.bank_down, ctx->device)) return -1;

    if (!binding->cpu_activations->data || !binding->cpu_ids->data || !binding->cpu_route_weights->data || !binding->cpu_output->data ||
        binding->cpu_activations->type != GGML_TYPE_F32 || binding->cpu_activations->ne[0] != layout.n_embd ||
        binding->cpu_ids->type != GGML_TYPE_I32 || binding->cpu_ids->ne[0] != 1 ||
        binding->cpu_route_weights->type != GGML_TYPE_F32 || binding->cpu_route_weights->ne[0] != 1 ||
        binding->cpu_route_weights->ne[1] != 1 ||
        binding->cpu_output->type != GGML_TYPE_F32 || binding->cpu_output->ne[0] != layout.n_embd) return -1;

    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> resource;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(binding->cpu_resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return -1;
        resource = it->second;
    }
    if (resource->device != ctx->device || !ggml_cuda_moe_ffn_resource_supports_signature(*resource, layout)) return -1;

    {
        std::lock_guard<std::mutex> execution_lock(resource->execution_mutex);
        if (!resource->accepting_bindings.load(std::memory_order_acquire) ||
            ggml_backend_graph_compute(resource->resource.cpu_backend, binding->cpu_graph) != GGML_STATUS_SUCCESS) return -1;
    }

    auto state = std::make_shared<ggml_cuda_moe_ffn_binding_state>();
    state->binding = *binding;
    state->cpu_resource = resource;
    std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
    auto resource_it = ctx->moe_ffn_cpu_resources.find(binding->cpu_resource_id);
    if (resource_it == ctx->moe_ffn_cpu_resources.end() || resource_it->second != resource ||
        !resource->accepting_bindings.load(std::memory_order_acquire) ||
        ctx->moe_ffn_next_binding_id <= 0) return -1;
    const int32_t binding_id = ctx->moe_ffn_next_binding_id++;
    ctx->moe_ffn_bindings.emplace(binding_id, std::move(state));
    return binding_id;
}

GGML_CALL int32_t ggml_backend_cuda_moe_ffn_cpu_resource_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_cpu_resource * resource) {
    if (!ggml_backend_is_cuda(backend) || !resource || !ggml_backend_is_cpu(resource->cpu_backend) ||
        resource->n_threads <= 0 || !resource->signatures || resource->n_signatures == 0) return -1;
    ggml_cuda_moe_ffn_workspace_sizes sizes{};
    if (!ggml_cuda_moe_ffn_get_workspace_sizes(resource->n_embd, resource->n_ff,
                resource->n_experts_used, resource->token_capacity, sizes)) return -1;
    for (size_t i = 0; i < resource->n_signatures; ++i) {
        const auto & signature = resource->signatures[i];
        ggml_cuda_moe_ffn_workspace_sizes signature_sizes{};
        if (signature.n_embd <= 0 || signature.n_ff <= 0 || signature.n_experts_used <= 0 ||
                signature.token_capacity != resource->token_capacity ||
                signature.n_embd > resource->n_embd || signature.n_ff > resource->n_ff ||
                signature.n_experts_used > resource->n_experts_used ||
                !ggml_cuda_moe_ffn_get_workspace_sizes(signature.n_embd, signature.n_ff,
                    signature.n_experts_used, signature.token_capacity, signature_sizes)) return -1;
        for (size_t j = 0; j < i; ++j) {
            if (resource->signatures[j].n_embd == signature.n_embd &&
                    resource->signatures[j].n_ff == signature.n_ff &&
                    resource->signatures[j].n_experts_used == signature.n_experts_used &&
                    resource->signatures[j].token_capacity == signature.token_capacity) return -1;
        }
    }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        if (!ctx->moe_ffn_cpu_resources.empty()) return -1;
    }

    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    try {
        state = std::make_shared<ggml_cuda_moe_ffn_cpu_resource_state>();
        state->signatures.assign(resource->signatures, resource->signatures + resource->n_signatures);
        if (!ggml_cuda_moe_ffn_checked_mul((size_t) resource->n_experts_used,
                    (size_t) resource->token_capacity, state->draw_capacity) ||
                state->draw_capacity > INT32_MAX) return -1;
        state->resolved_pins.resize(state->draw_capacity);
        state->pending_pins.pins.reserve(state->draw_capacity);
        state->fill_candidates.reserve(state->draw_capacity);
    } catch (...) {
        return -1;
    }
    state->device = ctx->device;
    state->cuda_context = ctx;
    const size_t ids_bytes = sizes.ids;
    const size_t activation_bytes = sizes.activation;
    const size_t route_bytes = sizes.route;
    const size_t output_bytes = sizes.output;
    const size_t host_bytes = sizes.host_stage;
    ggml_cuda_set_device(ctx->device);
    if (cudaHostAlloc(&state->host_stage, host_bytes, cudaHostAllocPortable) != cudaSuccess) return -1;
    auto * host = (uint8_t *) state->host_stage;
    state->host_ids = (int32_t *) host;
    state->host_slots = (int32_t *) (host + ids_bytes);
    state->host_hit_slots = (int32_t *) (host + 2 * ids_bytes);
    state->host_activations = (float *) (host + 3 * ids_bytes);
    state->host_routes = (float *) (host + 3 * ids_bytes + activation_bytes);
    state->host_hit_routes = (float *) (host + 3 * ids_bytes + activation_bytes + route_bytes);
    state->host_output = (float *) (host + 3 * ids_bytes + activation_bytes + 2 * route_bytes +
            (size_t) resource->token_capacity * sizeof(float));
    state->device_up_input_q8_1_bytes = sizes.up_input_q8_1;
    state->device_down_input_q8_1_bytes = sizes.down_input_q8_1;
    const size_t hidden_bytes = sizes.hidden;
    const size_t down_bytes = sizes.down;
    auto alloc_device = [](void ** ptr, size_t bytes) { return cudaMalloc(ptr, bytes) == cudaSuccess; };
    if (!alloc_device((void **) &state->device_slots, ids_bytes) ||
        !alloc_device((void **) &state->device_routes, sizes.device_routes) ||
        !alloc_device(&state->device_up_input_q8_1, state->device_up_input_q8_1_bytes) ||
        !alloc_device(&state->device_down_input_q8_1, state->device_down_input_q8_1_bytes) ||
        !alloc_device((void **) &state->device_hidden, hidden_bytes) ||
        !alloc_device((void **) &state->device_down, down_bytes) ||
        !alloc_device((void **) &state->device_output, output_bytes) ||
        cudaEventCreateWithFlags(&state->hit_kernels_done, cudaEventDisableTiming) != cudaSuccess) return -1;

    state->fill_stream = ctx->moe_fill_stream();

    ggml_backend_cpu_set_n_threads(resource->cpu_backend, resource->n_threads);
    state->resource = *resource; // Ownership transfers only after successful warm allocation.
    state->resource.signatures = state->signatures.data();
    int32_t resource_id = -1;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        if (!ctx->moe_ffn_cpu_resources.empty() || ctx->moe_ffn_next_cpu_resource_id <= 0) {
            state->resource.cpu_backend = nullptr;
            return -1;
        }
        resource_id = ctx->moe_ffn_next_cpu_resource_id++;
        ctx->moe_ffn_cpu_resources.emplace(resource_id, state);
    }
    try {
        state->fill_worker = std::thread(ggml_cuda_moe_ffn_fill_worker, state.get());
    } catch (...) {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        ctx->moe_ffn_cpu_resources.erase(resource_id);
        state->resource.cpu_backend = nullptr;
        return -1;
    }
    return resource_id;
}

static bool ggml_cuda_moe_ffn_stop_transport(ggml_backend_cuda_context * ctx,
        ggml_cuda_moe_ffn_cpu_resource_state & state) {
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.accepting_fills = false;
        state.fill_stopping = true;
    }
    state.fill_cv.notify_all();
    if (state.fill_worker.joinable()) state.fill_worker.join();

    if (cudaSetDevice(ctx->device) != cudaSuccess) return false;
    if (state.fill_stream && cudaStreamSynchronize(state.fill_stream) != cudaSuccess) return false;
    if (state.fill_stuck) {
        const auto stuck = state.stuck_fill;
        if (!stuck.callbacks || !stuck.callbacks->finish_fill || !stuck.cache_context ||
                !stuck.callbacks->finish_fill(stuck.cache_context, &stuck.owner_job, false)) return false;
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.fill_stuck = false;
        if (state.fill_outstanding > 0) --state.fill_outstanding;
        state.stuck_fill = {};
    }
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        if (!state.fill_queue.empty() || state.fill_outstanding || state.fill_active) return false;
    }

    std::lock_guard<std::mutex> execution_lock(state.execution_mutex);
    const cudaStream_t stream = ctx->streams[ctx->device][0];
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess) return false;
    if (!ggml_cuda_moe_ffn_release_pending_pins(state)) return false;
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.transport_stopped = true;
    }
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_stop(ggml_backend_t backend, int32_t resource_id) {
    if (!ggml_backend_is_cuda(backend) || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        state = it->second;
    }
    return ggml_cuda_moe_ffn_stop_transport(ctx, *state);
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_unregister(ggml_backend_t backend, int32_t resource_id) {
    if (!ggml_backend_is_cuda(backend) || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        for (const auto & entry : ctx->moe_ffn_bindings) {
            if (entry.second->binding.cpu_resource_id == resource_id) return false;
        }
        state = it->second;
        state->accepting_bindings.store(false, std::memory_order_release);
    }
    if (!ggml_cuda_moe_ffn_stop_transport(ctx, *state)) {
        state->accepting_bindings.store(true, std::memory_order_release);
        return false;
    }
    std::lock_guard<std::mutex> registry_lock(ctx->moe_ffn_mutex);
    auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
    if (it == ctx->moe_ffn_cpu_resources.end() || it->second != state) return false;
    for (const auto & entry : ctx->moe_ffn_bindings) {
        if (entry.second->binding.cpu_resource_id == resource_id) {
            state->accepting_bindings.store(true, std::memory_order_release);
            return false;
        }
    }
    ctx->moe_ffn_cpu_resources.erase(it);
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(ggml_backend_t backend,
        int32_t resource_id, ggml_cuda_moe_ffn_cpu_resource_stats * stats) {
    if (!ggml_backend_is_cuda(backend) || !stats || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    uint64_t active_bindings = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        state = it->second;
        for (const auto & entry : ctx->moe_ffn_bindings) {
            active_bindings += entry.second->binding.cpu_resource_id == resource_id;
        }
    }
    std::lock_guard<std::mutex> lock(state->execution_mutex);
    stats->active_bindings = active_bindings;
    stats->pending_pin_active = state->pending_pins.active ? 1 : 0;
    {
        std::lock_guard<std::mutex> fill_lock(state->fill_mutex);
        stats->fills_accepted = state->fills_accepted;
        stats->fills_completed = state->fills_completed;
        stats->fills_failed = state->fills_failed;
        stats->fill_outstanding = state->fill_outstanding;
        stats->fill_bytes_completed = state->fill_bytes_completed;
    }
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_unregister(ggml_backend_t backend, int32_t binding_id) {
    if (!ggml_backend_is_cuda(backend) || binding_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_bindings.find(binding_id);
        if (it == ctx->moe_ffn_bindings.end()) return false;
        state = it->second;
    }
    {
        std::lock_guard<std::mutex> fill_lock(state->cpu_resource->fill_mutex);
        if (!state->cpu_resource->transport_stopped) return false;
    }
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    ggml_cuda_set_device(ctx->device);
    for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
        if (ctx->streams[ctx->device][j] != nullptr && cudaStreamSynchronize(ctx->streams[ctx->device][j]) != cudaSuccess) return false;
    }
    if (!ggml_cuda_moe_ffn_release_pending_pins(*state->cpu_resource)) return false;
    state->active = false;
    std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
    auto it = ctx->moe_ffn_bindings.find(binding_id);
    if (it == ctx->moe_ffn_bindings.end() || it->second != state) return false;
    ctx->moe_ffn_bindings.erase(it);
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_get_stats(ggml_backend_t backend, int32_t binding_id,
        ggml_cuda_moe_ffn_stats * stats) {
    if (!ggml_backend_is_cuda(backend) || !stats || binding_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_bindings.find(binding_id);
        if (it == ctx->moe_ffn_bindings.end()) return false;
        state = it->second;
    }
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    *stats = state->stats;
    return true;
}
