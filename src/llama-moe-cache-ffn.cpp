#include "llama-moe-cache-ffn.h"

#include "llama-model.h"

#include "ggml-moe-ffn.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace {

constexpr size_t CPU_GRAPH_CONTEXT_BYTES = 1024 * 1024;

bool has_routed_expert_tensors(const llama_layer & layer) {
    return layer.ffn_up_exps || layer.ffn_gate_exps || layer.ffn_down_exps;
}

#ifdef GGML_USE_CUDA
void release_resource(llama_moe_cache_ffn_state::device_state & device, bool stopped = false) {
    if (device.cpu_resource_id > 0) {
        if ((!stopped && !ggml_backend_cuda_moe_ffn_cpu_resource_stop(device.cuda_backend, device.cpu_resource_id)) ||
                !ggml_backend_cuda_moe_ffn_cpu_resource_unregister(device.cuda_backend, device.cpu_resource_id)) {
            std::abort();
        }
        device.cpu_resource_id = 0;
        device.cpu_backend = nullptr; // Ownership transferred to the CUDA resource at registration.
    } else if (device.cpu_backend) {
        ggml_backend_free(device.cpu_backend);
        device.cpu_backend = nullptr;
    }
}
#endif

bool model_layer_layout(const llama_model & model, int32_t il, ggml_moe_ffn_layout & layout) {
    if (il < 0 || (size_t) il >= model.layers.size()) return false;
    const auto & w = model.layers[il];
    if (!w.ffn_up_exps || !w.ffn_gate_exps || !w.ffn_down_exps) return false;
    layout = {};
    layout.n_experts = model.hparams.n_expert;
    layout.n_experts_used = model.hparams.n_expert_used;
    layout.n_embd = w.ffn_up_exps->ne[0];
    layout.n_ff = w.ffn_up_exps->ne[1];
    layout.storage_topology = w.ffn_up_gate_exps ? GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN :
        GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN;
    layout.cache_layer = il;
    layout.up = w.ffn_up_exps;
    layout.gate = w.ffn_gate_exps;
    layout.down = w.ffn_down_exps;
    layout.up_gate_storage = w.ffn_up_gate_exps;
    layout.unary_op = GGML_UNARY_OP_SILU;
    layout.limit = model.swiglu_limit(il, false);
    if (layout.up->buffer && !ggml_backend_buffer_is_host(layout.up->buffer)) return false;
    if (layout.gate->buffer && !ggml_backend_buffer_is_host(layout.gate->buffer)) return false;
    if (layout.down->buffer && !ggml_backend_buffer_is_host(layout.down->buffer)) return false;
    layout.flags = GGML_MOE_FFN_LAYOUT_HOST_RESIDENT;
    if (ggml_moe_ffn_expert_execution_inputs_supported(w.ffn_up_exps_b,
            w.ffn_gate_exps_b, w.ffn_down_exps_b, w.ffn_up_gate_exps_b, w.ffn_down_exps_s))
        layout.flags |= GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS;
    layout.flags |= GGML_MOE_FFN_LAYOUT_PHYSICAL_STORAGE_VALID;
    if (!model.hadamard_rotation(w.ffn_up_exps) && !model.hadamard_rotation(w.ffn_gate_exps) &&
            !model.hadamard_rotation(w.ffn_down_exps)) layout.flags |= GGML_MOE_FFN_LAYOUT_NO_HADAMARD;
    return ggml_moe_ffn_layout_supported(&layout);
}

bool layer_is_eligible(const llama_model & model, int32_t il, const llama_moe_cache & cache,
        const llama_moe_cache_frozen_layer & s, int32_t device, ggml_backend * backend,
        int32_t token_capacity, ggml_moe_ffn_layout & layout) {
    if (!model_layer_layout(model, il, layout)) return false;
    layout.token_capacity = token_capacity;
    layout.n_slots = s.n_slots;
    layout.device = s.device_group;
    layout.cache_group = s.group;
    layout.cache_layer = s.layer;
    layout.owner_backend = s.owner_backend;
    layout.cache_owner = &cache;
    layout.bank_up = s.cached_up;
    layout.bank_gate = s.cached_gate;
    layout.bank_down = s.cached_down;
    layout.up_gate_storage = s.host_up_gate_storage;
    layout.bank_up_gate_storage = s.cached_up_gate_storage;
    const bool identity = s.layer == il && s.owner_backend == backend && s.device_group == device &&
            s.n_experts == layout.n_experts && s.host_up == layout.up &&
            s.host_gate == layout.gate && s.host_down == layout.down;
    const bool cuda = ggml_backend_cuda_moe_ffn_layout_supported(backend, &layout);
    return identity && cuda;
}

void add_stats(ggml_cuda_moe_ffn_stats & dst, const ggml_cuda_moe_ffn_stats & src) {
    dst.all_hit_calls += src.all_hit_calls;
    dst.any_miss_calls += src.any_miss_calls;
    dst.gpu_hit_draws += src.gpu_hit_draws;
    dst.cpu_miss_draws += src.cpu_miss_draws;
}

} // namespace

bool llama_moe_cache_ffn_state::initialize(llama_moe_cache & cache, const llama_model & model,
        int32_t cpu_threads, int32_t token_capacity) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(cache);
    GGML_UNUSED(model);
    GGML_UNUSED(cpu_threads);
    GGML_UNUSED(token_capacity);
    return false;
#else
    const auto fail = [](const char * stage) {
        LLAMA_LOG_ERROR("%s: MoE cache FFN initialization failed at %s\n", __func__, stage);
        return false;
    };
    if (cpu_threads <= 0 || token_capacity <= 0) return fail("invalid resource envelope");

    this->cache = &cache;

    const int32_t first_layer = 0;
    const int32_t last_layer = (int32_t) model.layers.size();
    // Establish every device signature and allocate its complete context envelope before freezing cache groups.
    for (int32_t il = first_layer; il < last_layer; ++il) {
        ggml_moe_ffn_layout candidate{};
        if (!model_layer_layout(model, il, candidate)) continue;
        ggml_backend * owner = cache.backend(il);
        if (!owner) continue;
        candidate.owner_backend = owner;
        candidate.device = -1;
        candidate.token_capacity = token_capacity;
        if (!ggml_backend_cuda_moe_ffn_layout_supported(owner, &candidate)) continue;

        auto it = std::find_if(devices.begin(), devices.end(), [&](const auto & d) {
            return d.cuda_backend == owner;
        });
        if (it == devices.end()) {
            devices.emplace_back();
            devices.back().cuda_backend = owner;
            it = std::prev(devices.end());
        }
        const ggml_cuda_moe_ffn_resource_signature signature{
            candidate.n_embd, candidate.n_ff, candidate.n_experts_used, token_capacity };
        if (std::find_if(it->resource_signatures.begin(), it->resource_signatures.end(), [&](const auto & current) {
                    return current.n_embd == signature.n_embd && current.n_ff == signature.n_ff &&
                        current.n_experts_used == signature.n_experts_used &&
                        current.token_capacity == signature.token_capacity;
                }) == it->resource_signatures.end()) it->resource_signatures.push_back(signature);
        it->n_embd = std::max(it->n_embd, candidate.n_embd);
        it->n_ff = std::max(it->n_ff, candidate.n_ff);
        it->n_experts_used = std::max(it->n_experts_used, candidate.n_experts_used);
    }
    for (auto & d : devices) {
        d.cpu_backend = ggml_backend_cpu_init();
        if (!d.cpu_backend) {
            LLAMA_LOG_WARN("%s: CPU resource allocation failed; leaving backend on blocking cache path\n", __func__);
            continue;
        }
        ggml_cuda_moe_ffn_cpu_resource resource = { d.cpu_backend, d.n_embd, d.n_ff,
            d.n_experts_used, token_capacity, cpu_threads, d.resource_signatures.data(),
            d.resource_signatures.size() };
        d.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(d.cuda_backend, &resource);
        if (d.cpu_resource_id <= 0) {
            ggml_backend_free(d.cpu_backend);
            d.cpu_backend = nullptr;
            LLAMA_LOG_WARN("%s: hybrid resource does not cover token capacity %d; leaving backend on blocking cache path\n",
                    __func__, token_capacity);
        }
    }
    devices.erase(std::remove_if(devices.begin(), devices.end(), [](const auto & d) {
        return d.cpu_resource_id <= 0;
    }), devices.end());

    std::vector<bool> rejected_group_layers(model.layers.size(), false);
    for (int32_t il = first_layer; il < last_layer; ++il) {
        if (rejected_group_layers[il]) continue;
        ggml_moe_ffn_layout candidate{};
        if (!model_layer_layout(model, il, candidate)) continue;
        ggml_backend * owner = cache.backend(il);
        if (!owner) continue;
        candidate.owner_backend = owner;
        candidate.device = -1; // The CUDA-owned query resolves and checks this backend's device.
        candidate.token_capacity = token_capacity;
        if (!ggml_backend_cuda_moe_ffn_layout_supported(owner, &candidate)) continue;
        bool group_seen = false;
        for (const auto & d : devices) for (const auto & g : d.groups) {
            const auto & frozen_group = g.snapshot();
            if (std::find(frozen_group.group_layers.begin(), frozen_group.group_layers.end(), il) !=
                    frozen_group.group_layers.end()) {
                group_seen = true;
                break;
            }
        }
        if (group_seen) continue;

        device_state * dstate = nullptr;
        for (auto & d : devices) if (d.cuda_backend == owner) { dstate = &d; break; }
        if (!dstate || dstate->cpu_resource_id <= 0) continue;
        llama_moe_cache::frozen_lease group;
        if (!cache.freeze_group_for_layer(il, owner, group))
            return fail("cache group freeze");
        const auto & frozen_group = group.snapshot();
        if (frozen_group.owner_backend != owner || frozen_group.group_layers.empty())
            return fail("frozen group identity");

        bool group_eligible = true;
        bool group_has_routed_layer = false;
        for (const int32_t group_layer : frozen_group.group_layers) {
            if (group_layer < 0 || (size_t) group_layer >= model.layers.size()) {
                group_eligible = false;
                break;
            }
            const auto & weights = model.layers[group_layer];
            if (!has_routed_expert_tensors(weights)) continue;
            group_has_routed_layer = true;
            if (!weights.ffn_up_exps || !weights.ffn_gate_exps || !weights.ffn_down_exps) {
                group_eligible = false;
                break;
            }
            llama_moe_cache_frozen_layer snapshot;
            ggml_moe_ffn_layout descriptor{};
            if (!cache.frozen_layer_snapshot(frozen_group, group_layer, snapshot) ||
                    !layer_is_eligible(model, group_layer, cache, snapshot, frozen_group.device_group,
                        owner, token_capacity, descriptor)) {
                group_eligible = false;
                break;
            }
        }
        if (!group_eligible || !group_has_routed_layer) {
            for (const int32_t group_layer : frozen_group.group_layers) {
                if (group_layer >= 0 && (size_t) group_layer < rejected_group_layers.size())
                    rejected_group_layers[group_layer] = true;
            }
            if (!group.release()) return fail("ineligible cache group release");
            continue;
        }
        if (dstate->device < 0) dstate->device = frozen_group.device_group;
        if (dstate->device != frozen_group.device_group) return fail("backend spans incompatible cache devices");
        dstate->groups.push_back(std::move(group));
    }
    size_t eligible_count = 0;
    size_t routed_layer_count = 0;
    for (const auto & layer : model.layers) if (has_routed_expert_tensors(layer)) ++routed_layer_count;
    for (auto & d : devices) for (const auto & g : d.groups) {
        const auto & frozen_group = g.snapshot();
        for (const int32_t il : frozen_group.group_layers) {
            if (il < 0 || (size_t) il >= model.layers.size() ||
                    !has_routed_expert_tensors(model.layers[il])) continue;
            llama_moe_cache_frozen_layer layer_snapshot;
            ggml_moe_ffn_layout descriptor{};
            if (!cache.frozen_layer_snapshot(frozen_group, il, layer_snapshot) ||
                    !layer_is_eligible(model, il, cache, layer_snapshot, d.device,
                        d.cuda_backend, token_capacity, descriptor)) return fail("retained group changed after validation");
            ++eligible_count;
        }
    }
    const size_t ineligible_count = routed_layer_count - std::min(routed_layer_count, eligible_count);

    for (auto & d : devices) if (d.groups.empty()) release_resource(d);
    devices.erase(std::remove_if(devices.begin(), devices.end(), [](const auto & d) {
        return d.groups.empty();
    }), devices.end());

    if (eligible_count == 0) {
        // The caller requested asynchronous execution.  Falling back to the
        // blocking cache here would silently reintroduce the synchronous
        // transfers that --moe-cache-async is meant to eliminate.
        LLAMA_LOG_ERROR("%s: no layer is eligible for the CUDA hybrid executor at token capacity %d; "
                "refusing to fall back to the blocking cache\n", __func__, token_capacity);
        return false;
    }

    for (auto & d : devices) for (const auto & g : d.groups) {
        const auto & frozen_group = g.snapshot();
        for (const int32_t il : frozen_group.group_layers) {
            if (il < 0 || (size_t) il >= model.layers.size() ||
                    !has_routed_expert_tensors(model.layers[il])) continue;
            llama_moe_cache_frozen_layer snapshot;
            ggml_moe_ffn_layout layout{};
            if (!cache.frozen_layer_snapshot(frozen_group, il, snapshot) ||
                    !layer_is_eligible(model, il, cache, snapshot, d.device, d.cuda_backend,
                        token_capacity, layout)) return fail("retained group changed before binding");
            layer_binding layer;
            layer.snapshot = std::move(snapshot);
            layer.cuda_backend = d.cuda_backend;
            layer.cpu_backend = d.cpu_backend;
            layer.cpu_resource_id = d.cpu_resource_id;
            layer.layout = layout;
            // The routed activation is only known when the model graph is
            // built.  Defer the CPU miss graph and the CUDA binding to
            // build_node(), which knows the actual unary op.
            d.layer_bindings.push_back(std::move(layer));
        }
    }

    layer_by_index.assign(model.layers.size(), nullptr);
    size_t indexed_bindings = 0;
    for (const auto & d : devices) for (const auto & layer : d.layer_bindings) {
        const int32_t il = layer.snapshot.layer;
        if (il < 0 || (size_t) il >= layer_by_index.size() || layer_by_index[il])
            return fail("duplicate or invalid layer binding index");
        layer_by_index[il] = &layer;
        ++indexed_bindings;
    }
    if (indexed_bindings != eligible_count) return fail("incomplete layer binding index");
    ready = true;

    LLAMA_LOG_INFO("%s: devices=%zu groups=%zu compatible-layers=%zu unsupported-layers=%zu resources=%zu threads/device=%d graph-context=%zu MiB\n",
            __func__, devices.size(), std::accumulate(devices.begin(), devices.end(), size_t(0),
                [](size_t n, const device_state & d) { return n + d.groups.size(); }),
            indexed_bindings, ineligible_count, devices.size(), cpu_threads,
            indexed_bindings * CPU_GRAPH_CONTEXT_BYTES / (1024 * 1024));
    return true;
#endif
}

void llama_moe_cache_ffn_state::shutdown() {
    ready = false;
    layer_by_index.clear();
#ifdef GGML_USE_CUDA
    for (auto & d : devices) if (d.cpu_resource_id > 0 &&
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(d.cuda_backend, d.cpu_resource_id)) std::abort();
    for (const auto & d : devices) {
        ggml_cuda_moe_ffn_stats semantic{};
        ggml_cuda_moe_ffn_cpu_resource_stats resource{};
        bool stats_ok = d.cpu_resource_id > 0 &&
                ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(d.cuda_backend, d.cpu_resource_id, &resource);
        for (const auto & layer : d.layer_bindings) {
            ggml_cuda_moe_ffn_stats current{};
            if (layer.binding_id <= 0 ||
                    !ggml_backend_cuda_moe_ffn_get_stats(d.cuda_backend, layer.binding_id, &current)) {
                stats_ok = false;
            } else {
                add_stats(semantic, current);
            }
        }
        if (stats_ok) {
            LLAMA_LOG_INFO("%s: device=%d layers=%zu calls=%llu all-hit=%llu any-miss=%llu gpu-hit-rows=%llu cpu-miss-rows=%llu fills=%llu/%llu failed=%llu fill-bytes=%llu pending-jobs=%llu pending-pins=%llu bindings=%llu\n",
                    __func__, d.device, d.layer_bindings.size(),
                    (unsigned long long) (semantic.all_hit_calls + semantic.any_miss_calls),
                    (unsigned long long) semantic.all_hit_calls,
                    (unsigned long long) semantic.any_miss_calls,
                    (unsigned long long) semantic.gpu_hit_draws,
                    (unsigned long long) semantic.cpu_miss_draws,
                    (unsigned long long) resource.fills_completed,
                    (unsigned long long) resource.fills_accepted,
                    (unsigned long long) resource.fills_failed,
                    (unsigned long long) resource.fill_bytes_completed,
                    (unsigned long long) resource.fill_outstanding,
                    (unsigned long long) resource.pending_pin_active,
                    (unsigned long long) resource.active_bindings);
        } else {
            LLAMA_LOG_WARN("%s: unable to collect final CUDA hybrid FFN counters for device %d\n", __func__, d.device);
        }
    }
    for (auto & d : devices) for (auto & layer : d.layer_bindings) if (layer.binding_id > 0) {
        if (!ggml_backend_cuda_moe_ffn_unregister(d.cuda_backend, layer.binding_id)) std::abort();
        layer.binding_id = 0;
    }
    for (auto & d : devices) {
        for (auto & layer : d.layer_bindings) {
            if (layer.cpu_buffer) { ggml_backend_buffer_free(layer.cpu_buffer); layer.cpu_buffer = nullptr; }
            if (layer.cpu_ctx) { ggml_free(layer.cpu_ctx); layer.cpu_ctx = nullptr; }
        }
        d.layer_bindings.clear();
        release_resource(d, true);
        for (auto & lease : d.groups) if (lease) {
            if (!lease.release()) std::abort();
        }
        d.groups.clear();
    }
    LLAMA_LOG_INFO("%s: CUDA hybrid FFN teardown complete (devices=%zu, bindings=0, resources=0, leases=0)\n",
            __func__, devices.size());
#endif
    devices.clear();
    cache = nullptr;
}

const llama_moe_cache_ffn_state::layer_binding * llama_moe_cache_ffn_state::find_layer(int32_t il) const {
    if (!ready || il < 0 || (size_t) il >= layer_by_index.size()) return nullptr;
    return layer_by_index[il];
}

llama_moe_cache_ffn_state::layer_binding * llama_moe_cache_ffn_state::find_layer_mutable(int32_t il) {
    if (!ready || il < 0 || (size_t) il >= layer_by_index.size()) return nullptr;
    return const_cast<layer_binding *>(layer_by_index[il]);
}

bool llama_moe_cache_ffn_state::build_layer_cpu_graph(layer_binding & layer,
        ggml_unary_op unary_op, float limit) {
#ifdef GGML_USE_CUDA
    if (layer.binding_id > 0) {
        if (!ggml_backend_cuda_moe_ffn_unregister(layer.cuda_backend, layer.binding_id)) return false;
        layer.binding_id = 0;
    }
    if (layer.cpu_buffer) { ggml_backend_buffer_free(layer.cpu_buffer); layer.cpu_buffer = nullptr; }
    if (layer.cpu_ctx) { ggml_free(layer.cpu_ctx); layer.cpu_ctx = nullptr; }
    layer.cpu_graph = nullptr;
    layer.cpu_activations = nullptr;
    layer.cpu_ids = nullptr;
    layer.cpu_routes = nullptr;
    layer.cpu_output = nullptr;
    layer.layout.unary_op = unary_op;
    layer.layout.limit = limit;
    if (!ggml_moe_ffn_unary_supported(unary_op)) return false;

    const auto fail = [&]() {
        if (layer.cpu_buffer) { ggml_backend_buffer_free(layer.cpu_buffer); layer.cpu_buffer = nullptr; }
        if (layer.cpu_ctx) { ggml_free(layer.cpu_ctx); layer.cpu_ctx = nullptr; }
        layer.cpu_graph = nullptr;
        layer.cpu_activations = nullptr;
        layer.cpu_ids = nullptr;
        layer.cpu_routes = nullptr;
        layer.cpu_output = nullptr;
        return false;
    };

    layer.cpu_ctx = ggml_init({ CPU_GRAPH_CONTEXT_BYTES, nullptr, true });
    if (!layer.cpu_ctx) return false;
    layer.cpu_activations = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_F32, layer.layout.n_embd, 1);
    layer.cpu_ids = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_I32, 1, 1);
    layer.cpu_routes = ggml_new_tensor_3d(layer.cpu_ctx, GGML_TYPE_F32, 1, 1, 1);
    ggml_tensor * up_gate = ggml_moe_up_gate(layer.cpu_ctx,
            const_cast<ggml_tensor *>(layer.snapshot.host_up),
            const_cast<ggml_tensor *>(layer.snapshot.host_gate),
            layer.cpu_activations, layer.cpu_ids, unary_op);
    if (!up_gate) return fail();
    *(float *)(up_gate->op_params + 1) = limit;
    layer.cpu_output = ggml_mul_multi_add(layer.cpu_ctx,
            ggml_mul_mat_id(layer.cpu_ctx, const_cast<ggml_tensor *>(layer.snapshot.host_down),
                    up_gate, layer.cpu_ids), layer.cpu_routes);
    if (!layer.cpu_activations || !layer.cpu_ids || !layer.cpu_routes || !layer.cpu_output ||
            layer.cpu_output->ne[0] != layer.layout.n_embd || layer.cpu_output->ne[1] != 1 ||
            layer.cpu_output->ne[2] != 1) return fail();
    layer.cpu_graph = ggml_new_graph_custom(layer.cpu_ctx, 64, false);
    ggml_build_forward_expand(layer.cpu_graph, layer.cpu_output);
    layer.cpu_buffer = ggml_backend_alloc_ctx_tensors(layer.cpu_ctx, layer.cpu_backend);
    if (!layer.cpu_buffer) return fail();
    const std::vector<float> priming_activations(layer.layout.n_embd, 0.0f);
    const int32_t priming_id = 0;
    const float priming_route = 1.0f;
    ggml_backend_tensor_set(layer.cpu_activations,
            priming_activations.data(), 0, ggml_nbytes(layer.cpu_activations));
    ggml_backend_tensor_set(layer.cpu_ids,
            &priming_id, 0, ggml_nbytes(layer.cpu_ids));
    ggml_backend_tensor_set(layer.cpu_routes,
            &priming_route, 0, ggml_nbytes(layer.cpu_routes));

    ggml_cuda_moe_ffn_binding binding = {};
    binding.generation = layer.snapshot.generation;
    binding.layout = layer.layout;
    binding.cache_callbacks = llama_moe_cache_get_cuda_callbacks();
    binding.cache_context = cache;
    binding.cpu_resource_id = layer.cpu_resource_id;
    binding.cpu_graph = layer.cpu_graph;
    binding.cpu_activations = layer.cpu_activations;
    binding.cpu_ids = layer.cpu_ids;
    binding.cpu_route_weights = layer.cpu_routes;
    binding.cpu_output = layer.cpu_output;
    layer.binding_id = ggml_backend_cuda_moe_ffn_register(layer.cuda_backend, &binding);
    if (layer.binding_id <= 0) return fail();
    return true;
#else
    GGML_UNUSED(layer); GGML_UNUSED(unary_op); GGML_UNUSED(limit); return false;
#endif
}

ggml_tensor * llama_moe_cache_ffn_state::build_node(ggml_context * ctx, ggml_backend_sched_t sched,
        const ggml_moe_ffn_layout & layout) {
#ifdef GGML_USE_CUDA
    layer_binding * layer = find_layer_mutable(layout.cache_layer);
    if (!ready || !layer || !sched || !cache ||
            !cache->frozen_snapshot_valid(layer->snapshot) || !ggml_moe_ffn_layout_supported(&layout) ||
            !ggml_moe_ffn_graph_supported(&layout) || layout.owner_backend != layer->cuda_backend ||
            layout.cache_owner != cache || layout.device != layer->layout.device ||
            layout.cache_group != layer->layout.cache_group || layout.cache_layer != layer->layout.cache_layer ||
            layout.n_experts != layer->layout.n_experts || layout.n_experts_used != layer->layout.n_experts_used ||
            layout.n_embd != layer->layout.n_embd || layout.n_ff != layer->layout.n_ff ||
            layout.n_slots != layer->layout.n_slots || layout.storage_topology != layer->layout.storage_topology ||
            layout.token_capacity != layer->layout.token_capacity ||
            layout.up != layer->layout.up || layout.gate != layer->layout.gate || layout.down != layer->layout.down ||
            layout.bank_up != layer->layout.bank_up || layout.bank_gate != layer->layout.bank_gate ||
            layout.bank_down != layer->layout.bank_down || layout.up_gate_storage != layer->layout.up_gate_storage ||
            layout.bank_up_gate_storage != layer->layout.bank_up_gate_storage ||
            layout.flags != layer->layout.flags) return nullptr;
    if (!ggml_moe_ffn_unary_supported(layout.unary_op)) return nullptr;
    if (layer->binding_id <= 0 || layer->layout.unary_op != layout.unary_op ||
            layer->layout.limit != layout.limit) {
        if (!build_layer_cpu_graph(*layer, layout.unary_op, layout.limit)) return nullptr;
    }
    ggml_tensor * op = ggml_moe_ffn(ctx, const_cast<ggml_tensor *>(layout.activation),
            const_cast<ggml_tensor *>(layout.selected_ids), const_cast<ggml_tensor *>(layout.route_weights),
            layout.device, layer->binding_id, layer->snapshot.generation, layout.n_tokens,
            layout.n_embd, layout.n_ff, layout.unary_op, layout.limit);
    if (!ggml_backend_supports_op(layer->cuda_backend, op)) return nullptr;
    ggml_backend_sched_set_tensor_backend(sched, op, layer->cuda_backend);
    return op;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(sched); GGML_UNUSED(layout); return nullptr;
#endif
}
