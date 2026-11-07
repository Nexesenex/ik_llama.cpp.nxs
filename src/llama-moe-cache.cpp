#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-lru.h"
#include "llama-moe-cache-share.h"
#include "llama-moe-cache-policy.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

using llama_moe_cache_detail::moe_cache_lru;

static std::vector<ggml_tensor *> layer_experts(const llama_layer & layer) {
    if (layer.ffn_up_gate_exps) return { layer.ffn_up_gate_exps, layer.ffn_down_exps };
    return { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps };
}

static bool is_host_weight(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static bool same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) return false;
    }
    return true;
}

static bool checked_mul_size(size_t a, size_t b, size_t & result) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}

static bool tensor_slice_offset(const ggml_tensor * view, const ggml_tensor * storage,
        size_t index, size_t bytes, size_t & offset) {
    if (!view || !storage || !view->nb[2]) return false;
    const ggml_tensor * view_root = view->view_src ? view->view_src : view;
    const ggml_tensor * storage_root = storage->view_src ? storage->view_src : storage;
    if (view_root != storage_root || view->view_offs < storage->view_offs) return false;
    const size_t base = view->view_offs - storage->view_offs;
    size_t step = 0;
    if (!checked_mul_size(index, view->nb[2], step) || base > SIZE_MAX - step) return false;
    offset = base + step;
    const size_t storage_size = ggml_nbytes(storage);
    return offset <= storage_size && bytes <= storage_size - offset;
}

static bool tensor_matrix_bytes(const ggml_tensor * tensor, size_t & bytes) {
    if (!tensor || tensor->ne[1] < 0) return false;
    return checked_mul_size(ggml_row_size(tensor->type, tensor->ne[0]),
            (size_t) tensor->ne[1], bytes);
}

} // namespace

struct llama_moe_cache::impl {
    struct device {
        ggml_backend * backend = nullptr;
        ggml_backend_buffer_type * buft = nullptr;
        int32_t model_device = -1;
        double split = 0.0;
        size_t host_bytes = 0;
        size_t buffer_size = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };
    struct group {
        int32_t device = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        llama_moe_cache_policy_detail::state cache;
        uint64_t generation = 0;
        bool frozen = false;
        moe_cache_lru lru;
    };
    struct layer {
        int32_t group = -1;
        ggml_tensor * slot_map = nullptr;
        ggml_tensor * host_up = nullptr;
        ggml_tensor * host_gate = nullptr;
        ggml_tensor * host_down = nullptr;
        ggml_tensor * host_up_gate_storage = nullptr;
        std::vector<ggml_tensor *> experts;
    };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct stats { size_t hits = 0; size_t misses = 0; size_t evictions = 0; size_t bytes = 0; };

    int32_t n_expert_used;
    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;
    std::vector<int32_t> ids;
    std::vector<llama_moe_cache_policy_detail::state::fill> fills;
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    stats stats_small, stats_large;
    size_t requested_size;
    mutable size_t large_batch_skips = 0;
    mutable size_t incompatible_layer_skips = 0;
    mutable std::mutex state_mutex;

    const binding * find_binding(const ggml_tensor * host, int32_t il, const group & g) const {
        if (!host) return nullptr;
        const auto it = bindings.find(host);
        if (it == bindings.end() || it->second.layer != il || it->second.part < 0 ||
                (size_t) it->second.part >= g.banks.size() || !it->second.cached) return nullptr;
        return &it->second;
    }

    bool make_frozen_snapshot(int32_t group_id, const group & g, int32_t il,
            llama_moe_cache_frozen_layer & snapshot) const {
        if (!g.frozen || g.device < 0 || (size_t) g.device >= devices.size() || il < 0 ||
                (size_t) il >= layers.size() || layers[il].group != group_id ||
                std::find(g.layers.begin(), g.layers.end(), il) == g.layers.end()) return false;
        const layer & l = layers[il];
        if (l.experts.size() != g.banks.size()) return false;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            if (!l.experts[ip] || !l.experts[ip]->data || !g.banks[ip] ||
                    !find_binding(l.experts[ip], il, g)) return false;
        }
        const binding * up = find_binding(l.host_up, il, g);
        const binding * gate = find_binding(l.host_gate, il, g);
        const binding * down = find_binding(l.host_down, il, g);
        const binding * parent = find_binding(l.host_up_gate_storage, il, g);
        if (!up || !gate || !down || (l.host_up_gate_storage && !parent)) return false;

        snapshot = {};
        snapshot.layer = il;
        snapshot.group = group_id;
        snapshot.device_group = g.device;
        snapshot.owner_backend = devices[g.device].backend;
        snapshot.n_experts = g.lru.n_expert;
        snapshot.n_slots = g.n_slots;
        snapshot.generation = g.generation;
        snapshot.host_up = l.host_up;
        snapshot.host_gate = l.host_gate;
        snapshot.host_down = l.host_down;
        snapshot.host_up_gate_storage = l.host_up_gate_storage;
        snapshot.bank_up = g.banks[up->part];
        snapshot.bank_gate = g.banks[gate->part];
        snapshot.bank_down = g.banks[down->part];
        snapshot.bank_up_gate_storage = parent ? g.banks[parent->part] : nullptr;
        snapshot.cached_up = up->cached;
        snapshot.cached_gate = gate->cached;
        snapshot.cached_down = down->cached;
        snapshot.cached_up_gate_storage = parent ? parent->cached : nullptr;
        snapshot.group_layers = g.layers;
        return snapshot.bank_up && snapshot.bank_gate && snapshot.bank_down && snapshot.cached_up &&
                snapshot.cached_gate && snapshot.cached_down &&
                (!l.host_up_gate_storage || (snapshot.bank_up_gate_storage && snapshot.cached_up_gate_storage));
    }

    bool frozen_snapshot_matches(const llama_moe_cache_frozen_layer & snapshot,
            int32_t group_id, const group & g) const {
        if (!g.frozen || g.device < 0 || (size_t) g.device >= devices.size() ||
                snapshot.group != group_id || snapshot.device_group != g.device ||
                snapshot.owner_backend != devices[g.device].backend ||
                snapshot.n_experts != g.lru.n_expert || snapshot.n_slots != g.n_slots ||
                snapshot.generation != g.generation || snapshot.group_layers != g.layers ||
                snapshot.layer < 0 || (size_t) snapshot.layer >= layers.size() ||
                layers[snapshot.layer].group != group_id) return false;
        const layer & l = layers[snapshot.layer];
        auto same_view = [&](const ggml_tensor * host, ggml_tensor * cached, const ggml_tensor * bank) {
            const binding * b = find_binding(host, snapshot.layer, g);
            return b && b->cached == cached && g.banks[b->part] == bank;
        };
        return snapshot.host_up == l.host_up && snapshot.host_gate == l.host_gate &&
                snapshot.host_down == l.host_down && snapshot.host_up_gate_storage == l.host_up_gate_storage &&
                same_view(l.host_up, snapshot.cached_up, snapshot.bank_up) &&
                same_view(l.host_gate, snapshot.cached_gate, snapshot.bank_gate) &&
                same_view(l.host_down, snapshot.cached_down, snapshot.bank_down) &&
                (!l.host_up_gate_storage || same_view(l.host_up_gate_storage,
                    snapshot.cached_up_gate_storage, snapshot.bank_up_gate_storage));
    }

    bool build_fill_descriptor(llama_moe_cache * owner, int32_t device_id, int32_t group_id,
            int32_t il, uint64_t owner_generation, int32_t expert, int32_t slot, int32_t key,
            size_t part_capacity, ggml_cuda_moe_ffn_fill_job & pending,
            ggml_cuda_moe_ffn_fill_part * parts) const {
        const group & g = groups[group_id];
        const layer & l = layers[il];
        const size_t physical_parts = l.experts.size();
        if (part_capacity != physical_parts || physical_parts == 0 || g.banks.size() != physical_parts ||
                l.group != group_id || g.device != device_id || expert < 0 || expert >= g.lru.n_expert ||
                slot < 0 || slot >= g.lru.n_slots) return false;

        pending = {};
        pending.owner = owner;
        pending.device = device_id;
        pending.group = group_id;
        pending.layer = il;
        pending.slot = slot;
        pending.key = key;
        pending.owner_generation = owner_generation;

        for (size_t ip = 0; ip < physical_parts; ++ip) {
            const ggml_tensor * weights = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            if (!weights || !weights->data || !bank || !bank->data) return false;
            if (ip == 0 && l.host_up_gate_storage && l.host_up && l.host_gate) {
                const binding * up_binding = find_binding(l.host_up, il, g);
                const binding * gate_binding = find_binding(l.host_gate, il, g);
                if (!up_binding || !gate_binding || up_binding->part != (int32_t) ip ||
                        gate_binding->part != (int32_t) ip) return false;
                const ggml_tensor * up = l.host_up;
                const ggml_tensor * gate = l.host_gate;
                const ggml_tensor * cached_up = up_binding->cached;
                const ggml_tensor * cached_gate = gate_binding->cached;
                size_t up_bytes = 0, gate_bytes = 0;
                size_t up_source_offset = 0, gate_source_offset = 0;
                size_t up_destination_offset = 0, gate_destination_offset = 0;
                if (!tensor_matrix_bytes(up, up_bytes) || !tensor_matrix_bytes(gate, gate_bytes) ||
                        !up_bytes || !gate_bytes || up_bytes > SIZE_MAX - gate_bytes ||
                        up_bytes + gate_bytes != weights->nb[2] ||
                        up->type != weights->type || gate->type != weights->type ||
                        up->ne[2] <= expert || gate->ne[2] <= expert ||
                        cached_up->ne[2] <= slot || cached_gate->ne[2] <= slot ||
                        !tensor_slice_offset(up, weights, (size_t) expert, up_bytes, up_source_offset) ||
                        !tensor_slice_offset(gate, weights, (size_t) expert, gate_bytes, gate_source_offset) ||
                        !tensor_slice_offset(cached_up, bank, (size_t) slot, up_bytes, up_destination_offset) ||
                        !tensor_slice_offset(cached_gate, bank, (size_t) slot, gate_bytes, gate_destination_offset) ||
                        pending.total_bytes > SIZE_MAX - (up_bytes + gate_bytes)) return false;
                parts[ip] = { (const uint8_t *) weights->data + gate_source_offset,
                    (uint8_t *) bank->data + gate_destination_offset,
                    gate_bytes, (const uint8_t *) weights->data + up_source_offset,
                    (uint8_t *) bank->data + up_destination_offset, up_bytes };
                pending.total_bytes += gate_bytes + up_bytes;
                continue;
            }

            const size_t bytes = weights->nb[2];
            size_t source_offset = 0, destination_offset = 0;
            if (!bytes || !tensor_slice_offset(weights, weights, (size_t) expert, bytes, source_offset) ||
                    !tensor_slice_offset(bank, bank, (size_t) slot, bytes, destination_offset) ||
                    pending.total_bytes > SIZE_MAX - bytes) return false;
            parts[ip] = { (const uint8_t *) weights->data + source_offset,
                    (uint8_t *) bank->data + destination_offset, bytes, nullptr, nullptr, 0 };
            pending.total_bytes += bytes;
        }
        return pending.total_bytes != 0;
    }

    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size,
            enum llama_moe_cache_policy policy)
            : n_expert_used(model.hparams.n_expert_used), layers(model.layers.size()), requested_size(size) {
        if (model.hparams.n_expert == 0 || n_expert_used <= 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (!model.rpc_servers.empty()) throw std::runtime_error("MoE cache does not support RPC devices");
        if (backends.size() != bufts.size()) throw std::runtime_error("MoE cache backend metadata is inconsistent");

        for (size_t i = 0; i < backends.size(); ++i) {
            if (!ggml_backend_is_cpu(backends[i])) {
                device d;
                d.backend = backends[i];
                d.buft = bufts[i];
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty()) throw std::runtime_error("MoE cache requires a GPU backend");

        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), is_host_weight)) continue;
            if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) {
                    return t->ne[2] == experts[0]->ne[2];
                })) continue;
            if (il >= model.default_layer_device.size()) continue;
            const int32_t model_device = model.default_layer_device[il];
            if (model_device < 0 || (size_t) model_device >= model.devices.size()) continue;
            const auto * layer_buft = model.default_buffer_type_offload(model_device);
            auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.buft == layer_buft; });
            if (it_dev == devices.end()) continue;
            const int32_t id = (int32_t) (it_dev - devices.begin());
            it_dev->model_device = model_device;
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.device == id && same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->device = id;
                it->ref = experts;
            }
            it->layers.push_back((int32_t) il);
            for (ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no complete host-resident expert layer is eligible. MoE cache is inactive\n", __func__);
            return;
        }

        std::vector<bool> eligible(model.splits.size(), false);
        for (const device & d : devices) {
            if (d.host_bytes && d.model_device >= 0 && (size_t) d.model_device < eligible.size()) {
                eligible[d.model_device] = true;
            }
        }
        const std::vector<double> shares = llama_moe_cache_detail::device_shares(model.splits, eligible);
        for (device & d : devices) {
            d.split = d.model_device >= 0 && (size_t) d.model_device < shares.size() ? shares[d.model_device] : 0.0;
        }

        double split_sum = 0.0;
        for (const device & d : devices) if (d.host_bytes) split_sum += d.split;
        if (split_sum <= 0.0) {
            for (device & d : devices) if (d.host_bytes) d.split = 1.0;
            split_sum = std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes != 0; });
        }
        auto alloc_size = [&](const group & g, int32_t slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.device].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                const size_t bytes = t->nb[2] * (size_t) (slots + 1);
                if (bytes > SIZE_MAX - alignment) return SIZE_MAX;
                const size_t padded = GGML_PAD(bytes, alignment);
                if (padded > SIZE_MAX - total) return SIZE_MAX;
                total += padded;
            }
            return total;
        };

        size_t host_layer_count = 0;
        std::vector<size_t> device_tensor_counts(devices.size(), 0);
        for (group & g : groups) {
            const device & d = devices[g.device];
            const size_t budget = (size_t) ((double) size * d.split / split_sum * g.host_bytes / d.host_bytes);
            const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
            const int32_t max_slots = (int32_t) std::min<size_t>(INT32_MAX, g.layers.size() * (size_t) n_expert);
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) ++g.n_slots;
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: cache budget leaves %zu expert layers uncached because they cannot hold one token\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.cache.init((int32_t) layers.size(), n_expert, g.n_slots,
                    policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU ? llama_moe_cache_policy_detail::kind::classlru : llama_moe_cache_policy_detail::kind::lru);
            g.lru.init((int32_t) layers.size(), (int32_t) g.layers.size(), n_expert, g.n_slots);
            device_tensor_counts[g.device] += g.ref.size();
            for (const int32_t il : g.layers) {
                const auto & layer = model.layers[il];
                device_tensor_counts[g.device] += layer.ffn_up_gate_exps ?
                    (layer.ffn_up_exps && layer.ffn_gate_exps ? 4 : 2) : 3;
            }
            host_layer_count += g.layers.size();
        }
        if (host_layer_count == 0) throw std::runtime_error("MoE cache budget is too small to hold one token");

        auto make_ctx = [](size_t n_tensors) {
            ggml_init_params params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) throw std::runtime_error("failed to create MoE cache tensor metadata");
            return ctx;
        };
        for (size_t id = 0; id < devices.size(); ++id) if (device_tensor_counts[id]) devices[id].ctx = make_ctx(device_tensor_counts[id]);
        ctx_host = make_ctx(host_layer_count);
        const auto buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (group & g : groups) {
            if (g.n_slots == 0) continue;
            auto & d = devices[g.device];
            for (ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(d.ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                if (bank->nb[2] != t->nb[2]) throw std::runtime_error("MoE cache expert layout is not contiguous by expert");
                ggml_format_name(bank, "moe_cache.%s", t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.group = (int32_t) (&g - groups.data());
                l.experts = layer_experts(model.layers[il]);
                l.host_up = model.layers[il].ffn_up_exps;
                l.host_gate = model.layers[il].ffn_gate_exps;
                l.host_down = model.layers[il].ffn_down_exps;
                l.host_up_gate_storage = model.layers[il].ffn_up_gate_exps;
                auto physical_index = [&](const ggml_tensor * host) -> int32_t {
                    const auto it = std::find(l.experts.begin(), l.experts.end(), host);
                    return it == l.experts.end() ? -1 : (int32_t) std::distance(l.experts.begin(), it);
                };
                auto cached_range = [&](int32_t ip, const ggml_tensor * name_tensor) -> ggml_tensor * {
                    if (ip < 0 || (size_t) ip >= g.banks.size()) return nullptr;
                    ggml_tensor * bank = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots,
                            bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", name_tensor->name);
                    return cached;
                };
                if (l.host_up_gate_storage) {
                    const int32_t parent_ip = physical_index(l.host_up_gate_storage);
                    const int32_t down_ip = physical_index(l.host_down);
                    ggml_tensor * cached_parent = cached_range(parent_ip, l.host_up_gate_storage);
                    ggml_tensor * cached_down = cached_range(down_ip, l.host_down);
                    bindings[l.host_up_gate_storage] = { il, parent_ip, cached_parent };
                    bindings[l.host_down] = { il, down_ip, cached_down };
                    if (l.host_up && l.host_gate) {
                        const size_t up_gate_offset = ggml_row_size(l.host_up->type, l.host_up->ne[0]) *
                                (size_t) l.host_gate->ne[1];
                        ggml_tensor * cached_up = ggml_view_3d(d.ctx.get(), cached_parent, l.host_up->ne[0],
                                l.host_up->ne[1], g.n_slots, l.host_up->nb[1], cached_parent->nb[2],
                                up_gate_offset);
                        ggml_tensor * cached_gate = ggml_view_3d(d.ctx.get(), cached_parent, l.host_gate->ne[0],
                                l.host_gate->ne[1], g.n_slots, l.host_gate->nb[1], cached_parent->nb[2],
                                l.host_gate->view_offs);
                        ggml_format_name(cached_up, "moe_cache.%s", l.host_up->name);
                        ggml_format_name(cached_gate, "moe_cache.%s", l.host_gate->name);
                        bindings[l.host_up] = { il, parent_ip, cached_up };
                        bindings[l.host_gate] = { il, parent_ip, cached_gate };
                    }
                } else {
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        ggml_tensor * cached = cached_range((int32_t) ip, l.experts[ip]);
                        bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                    }
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            d.buffer_size += alloc_size(g, g.n_slots);
        }

        for (device & d : devices) {
            if (!d.ctx) continue;
            d.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buffer) throw std::runtime_error("failed to allocate MoE cache buffers");
            ggml_backend_buffer_clear(d.buffer.get(), 0);
            d.buffer_size = ggml_backend_buffer_get_size(d.buffer.get());
        }
        buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
        if (!buf_host) throw std::runtime_error("failed to allocate MoE cache slot maps");
        ggml_backend_buffer_clear(buf_host.get(), 0xff);
        buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
        for (group & g : groups) if (g.n_slots > 0) for (int32_t il : g.layers) g.cache.slot_map[il] = (int32_t *) layers[il].slot_map->data;
        for (device & d : devices) if (d.buffer) ggml_backend_buffer_set_usage(d.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (buf_host) ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t i = 0; i < devices.size(); ++i) {
            const device & d = devices[i];
            if (!d.host_bytes || !d.buffer_size) continue;
            LLAMA_LOG_INFO("%s: %s MoE cache uses %.2f MiB for %.2f MiB of host experts\n", __func__,
                    ggml_backend_buft_name(d.buft), d.buffer_size / 1048576.0, d.host_bytes / 1048576.0);
            for (const group & g : groups) if (g.device == (int32_t) i && g.n_slots) {
                LLAMA_LOG_INFO("%s: %zu layer(s), %d expert slots (%.1f%% coverage)\n", __func__,
                        g.layers.size(), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
            }
        }
        LLAMA_LOG_INFO("%s: active MoE cache policy: %s\n", __func__,
                policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU ? "classlru" : "lru");
    }

    ~impl() {
        GGML_ASSERT(std::none_of(groups.begin(), groups.end(), [](const group & g) { return g.frozen; }) &&
                "MoE cache destroyed while a frozen group lease is active");
        log_stats();
    }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.group];
        if (g.frozen) return nullptr;
        if (n_tokens <= 0 || n_tokens > llama_moe_cache::max_batch_tokens || n_expert_used <= 0 ||
                n_tokens > std::numeric_limits<int64_t>::max() / n_expert_used) {
            ++large_batch_skips;
            return nullptr;
        }
        if (!llama_moe_cache_policy_detail::slot_map_eligible(n_tokens, n_expert_used,
                l.slot_map->ne[1], g.n_slots, max_batch)) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        std::lock_guard<std::mutex> lock(state_mutex);
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.group];
        if (g.frozen) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const device & d = devices[g.device];
        GGML_ASSERT(backend == d.backend);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
        }
        GGML_ASSERT(lookup && "MoE cache slot map must feed GET_ROWS");
        const ggml_tensor * selected = lookup->src[1];
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == selected || node == selected->view_src) {
                GGML_ABORT("MoE experts for layer %d are selected in the same scheduler split as the cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(selected));
        int64_t n_tokens = 1;
        if (g.cache.policy == llama_moe_cache_policy_detail::kind::classlru &&
                (!llama_moe_cache_policy_detail::recover_selected_tokens(selected, n_expert_used,
                    g.cache.n_expert, n_tokens) || n_tokens > max_batch)) {
            GGML_ABORT("Unable to recover MoE cache token geometry for layer %d", il);
        }
        ids.resize(ggml_nelements(selected));
        ggml_backend_tensor_get_async(backend, selected, ids.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(backend);
        size_t n_hit = 0;
        size_t n_eviction = 0;
        if (!g.cache.plan(il, ids.data(), ids.size(), n_tokens, fills, n_hit, n_eviction)) GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
        if (++g.generation == 0) ++g.generation;
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i+n < fills.size() && fills[i+n].expert == fills[i].expert + (int32_t)n &&
                        fills[i+n].slot == fills[i].slot + (int32_t)n) ++n;
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size,
                        fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits += n_hit;
        st.misses += fills.size();
        st.evictions += n_eviction;
        st.bytes += bytes;
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    std::map<ggml_backend_buffer_type *, size_t> memory_breakdown() const {
        std::map<ggml_backend_buffer_type *, size_t> res;
        for (const device & d : devices) {
            if (d.buffer) res[d.buft] += ggml_backend_buffer_get_size(d.buffer.get());
        }
        if (buf_host) res[ggml_backend_buffer_get_type(buf_host.get())] += ggml_backend_buffer_get_size(buf_host.get());
        return res;
    }

    bool freeze_group_for_layer(int32_t il, ggml_backend * expected_backend,
            llama_moe_cache_frozen_layer & out) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return false;
        layer & l = layers[il];
        group & g = groups[l.group];
        device & d = devices[g.device];
        if (g.frozen || d.backend != expected_backend || !expected_backend ||
                l.host_up == nullptr || l.host_gate == nullptr || l.host_down == nullptr ||
                l.experts.size() != g.banks.size()) return false;

        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            if (!l.experts[ip] || !l.experts[ip]->data || !g.banks[ip]) return false;
            auto it = bindings.find(l.experts[ip]);
            if (it == bindings.end() || !it->second.cached) return false;
        }
        // Hand the blocking-cache residency over to the hybrid slot manager. Groups are frozen before their
        // first routed FFN, so the source map is normally empty; copying it anyway keeps a warmed group
        // coherent and makes the ownership transfer explicit. No group ever runs both planners afterwards.
        g.lru.key_of = g.cache.key_of;
        for (size_t ii = 0; ii < g.lru.slot_map.size() && ii < g.cache.slot_map.size(); ++ii) {
            if (g.cache.slot_map[ii]) g.lru.slot_map[ii] = g.cache.slot_map[ii];
        }
        for (int32_t s = 0; s < g.lru.n_slots; ++s) {
            // Imported residents need a nonzero epoch so pin/lease validation accepts them.
            g.lru.slot_generation[s] = g.lru.key_of[s] >= 0 ? 1 : 0;
        }
        g.lru.freeze();

        if (++g.generation == 0) ++g.generation;
        g.frozen = true;
        GGML_ASSERT(make_frozen_snapshot(l.group, g, il, out));
        return true;
    }

    bool frozen_snapshot_valid(const llama_moe_cache_frozen_layer & snapshot) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (snapshot.group < 0 || (size_t) snapshot.group >= groups.size()) return false;
        return frozen_snapshot_matches(snapshot, snapshot.group, groups[snapshot.group]);
    }

    bool frozen_layer_snapshot(const llama_moe_cache_frozen_layer & group_snapshot, int32_t il,
            llama_moe_cache_frozen_layer & snapshot) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (group_snapshot.group < 0 || (size_t) group_snapshot.group >= groups.size() ||
                il < 0 || (size_t) il >= layers.size()) return false;
        const group & g = groups[group_snapshot.group];
        if (layers[il].group != group_snapshot.group ||
                !frozen_snapshot_matches(group_snapshot, group_snapshot.group, g)) return false;
        return make_frozen_snapshot(group_snapshot.group, g, il, snapshot);
    }

    bool release_frozen_group(const llama_moe_cache_frozen_layer & snapshot) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (snapshot.group < 0 || (size_t) snapshot.group >= groups.size()) return false;
        group & g = groups[snapshot.group];
        if (!g.frozen || snapshot.generation != g.generation) return false;
        for (int32_t s = 0; s < g.lru.n_slots; ++s) {
            if (g.lru.pin_count[s] != 0 || g.lru.state[s] == moe_cache_lru::slot_state::COPYING) return false;
        }
        g.frozen = false;
        if (++g.generation == 0) ++g.generation;
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & s) {
            const size_t n = s.hits + s.misses;
            if (n) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu misses=%zu evictions=%zu hit-rate=%.2f%% uploaded=%.2f MiB\n",
                    name, s.hits, s.misses, s.evictions, 100.0*s.hits/n, s.bytes/1048576.0);
        };
        size_t actual_size = buf_host_size;
        for (const device & d : devices) actual_size += d.buffer_size;
        LLAMA_LOG_INFO("llama_moe_cache: requested %.2f MiB, allocated %.2f MiB\n",
                requested_size / 1048576.0, actual_size / 1048576.0);
        log("ubatch <= 8", stats_small);
        log("ubatch 9-32", stats_large);
        if (large_batch_skips || incompatible_layer_skips) {
            LLAMA_LOG_INFO("llama_moe_cache: skipped %zu large-batch and %zu ineligible layer accesses\n",
                    large_batch_skips, incompatible_layer_skips);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model,
        const std::vector<ggml_backend *> & backends,
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size,
        enum llama_moe_cache_policy policy)
        : pimpl(new impl(model, backends, bufts, size, policy)) {}

llama_moe_cache::~llama_moe_cache() = default;

llama_moe_cache::frozen_lease::frozen_lease(frozen_lease && other) noexcept
        : owner(other.owner), descriptor(std::move(other.descriptor)) {
    other.owner = nullptr;
    other.descriptor = {};
}

llama_moe_cache::frozen_lease & llama_moe_cache::frozen_lease::operator=(frozen_lease && other) noexcept {
    if (this != &other) {
        if (owner && !release()) GGML_ABORT("failed to release frozen MoE cache group during lease move assignment");
        owner = other.owner;
        descriptor = std::move(other.descriptor);
        other.owner = nullptr;
        other.descriptor = {};
    }
    return *this;
}

llama_moe_cache::frozen_lease::~frozen_lease() {
    if (owner && !release()) GGML_ABORT("failed to release frozen MoE cache group during lease destruction");
}

bool llama_moe_cache::frozen_lease::release() {
    if (!owner) return false;
    if (!owner->release_frozen_group(descriptor)) return false;
    owner = nullptr;
    descriptor = {};
    return true;
}

ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }
std::map<ggml_backend_buffer_type *, size_t> llama_moe_cache::memory_breakdown() const { return pimpl->memory_breakdown(); }

bool llama_moe_cache::resolve_and_pin(int32_t device, int32_t group_id, int32_t il, uint64_t owner_generation,
        const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result) {
    if (!ids || !result || n_ids <= 0 || !result->draw_slots || !result->pins ||
            result->draw_capacity < (size_t) n_ids || result->pin_capacity < (size_t) n_ids) return false;
    result->n_draws = 0;
    result->n_pins = 0;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (group_id < 0 || (size_t) group_id >= pimpl->groups.size() ||
            il < 0 || (size_t) il >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[group_id];
    auto & l = pimpl->layers[il];
    if (!g.frozen || g.generation != owner_generation || g.device != device ||
            l.group != group_id || !g.lru.slot_map[il]) return false;

    int32_t n_pins = 0;
    for (int32_t i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        if (expert < 0 || expert >= g.lru.n_expert) return false;
        const int32_t slot = g.lru.slot_map[il][expert];
        if (slot < -1 || slot >= g.lru.n_slots) return false;
        result->draw_slots[i] = slot;
        if (slot < 0) continue;
        if (!g.lru.can_pin(il, expert)) return false;
        bool already_added = false;
        for (int32_t p = 0; p < n_pins; ++p) {
            already_added |= result->pins[p].slot == slot;
        }
        if (!already_added) {
            auto & token = result->pins[n_pins++];
            token.owner = this;
            token.device = device;
            token.group = group_id;
            token.layer = il;
            token.slot = slot;
            token.key = il*g.lru.n_expert + expert;
            token.owner_generation = owner_generation;
        }
    }

    for (int32_t p = 0; p < n_pins; ++p) {
        auto & token = result->pins[p];
        if (!g.lru.pin(token.layer, token.key % g.lru.n_expert, token.generation)) return false;
    }
    result->n_draws = n_ids;
    result->n_pins = n_pins;
    return true;
}

bool llama_moe_cache::release_pins(const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    if (!pins || n_pins <= 0) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    for (int32_t i = 0; i < n_pins; ++i) {
        const auto & token = pins[i];
        if (token.owner != this || token.group < 0 || (size_t) token.group >= pimpl->groups.size() ||
                token.layer < 0 || (size_t) token.layer >= pimpl->layers.size()) return false;
        const auto & g = pimpl->groups[token.group];
        const auto & l = pimpl->layers[token.layer];
        if (!g.frozen || g.device != token.device || l.group != token.group ||
                g.generation != token.owner_generation ||
                token.key < 0 || token.key / g.lru.n_expert != token.layer ||
                !g.lru.can_release(token.layer, token.key % g.lru.n_expert, token.slot,
                    token.generation)) return false;
        for (int32_t j = 0; j < i; ++j) {
            if (pins[j].owner == token.owner && pins[j].group == token.group &&
                    pins[j].slot == token.slot && pins[j].generation == token.generation) return false;
        }
    }
    for (int32_t i = 0; i < n_pins; ++i) pimpl->groups[pins[i].group].lru.release(pins[i].slot);
    return true;
}

bool llama_moe_cache::reserve_fill(int32_t device, int32_t group_id, int32_t il,
        uint64_t owner_generation, int32_t expert, ggml_cuda_moe_ffn_fill_job * job,
        ggml_cuda_moe_ffn_fill_part * parts, size_t part_capacity) {
    if (!job || !parts) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (group_id < 0 || (size_t) group_id >= pimpl->groups.size() ||
            il < 0 || (size_t) il >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[group_id];
    auto & l = pimpl->layers[il];
    const size_t physical_parts = l.experts.size();
    if (!g.frozen || g.generation != owner_generation || g.device != device ||
            l.group != group_id || expert < 0 || expert >= g.lru.n_expert ||
            !g.lru.slot_map[il] || physical_parts == 0 || part_capacity != physical_parts ||
            g.banks.size() != l.experts.size() || g.lru.slot_map[il][expert] >= 0) return false;

    const int64_t key64 = int64_t(il) * g.lru.n_expert + expert;
    if (key64 < 0 || key64 > std::numeric_limits<int32_t>::max()) return false;
    const int32_t key = (int32_t) key64;
    for (int32_t s = 0; s < g.lru.n_slots; ++s) {
        if (g.lru.state[s] == moe_cache_lru::slot_state::COPYING && g.lru.key_of[s] == key) return false;
    }

    int32_t slot = -1;
    bool has_free_slot = false;
    int32_t old_key = -1;
    for (int32_t s = 0; s < g.lru.n_slots; ++s) {
        if (g.lru.state[s] == moe_cache_lru::slot_state::FREE &&
                g.lru.key_of[s] < 0 && g.lru.pin_count[s] == 0) { slot = s; has_free_slot = true; break; }
    }

    // Demand admission.  A first miss is admitted when the group can hold all
    // of its experts anyway, otherwise it must be observed twice before a free
    // slot is spent.  Replacing a resident expert requires eight observations,
    // which keeps one-off routes from thrashing the bank.
    if (!g.lru.admit_miss(key, has_free_slot)) return false;

    if (slot < 0) {
        for (int32_t s = g.lru.head; s >= 0; s = g.lru.next[s]) {
            if (g.lru.state[s] != moe_cache_lru::slot_state::READY ||
                    g.lru.key_of[s] < 0 || g.lru.pin_count[s] != 0) continue;
            const int32_t resident_key = g.lru.key_of[s];
            const int32_t resident_layer = resident_key / g.lru.n_expert;
            const int32_t resident_expert = resident_key % g.lru.n_expert;
            if (resident_layer < 0 || (size_t) resident_layer >= pimpl->layers.size() ||
                    pimpl->layers[resident_layer].group != group_id ||
                    !g.lru.slot_map[resident_layer] ||
                    g.lru.slot_map[resident_layer][resident_expert] != s) return false;
            slot = s;
            old_key = resident_key;
            break;
        }
    }
    if (slot < 0) return false;

    ggml_cuda_moe_ffn_fill_job pending{};
    if (!pimpl->build_fill_descriptor(this, device, group_id, il, owner_generation,
                expert, slot, key, part_capacity, pending, parts)) return false;

    // Validate the full descriptor before removing its old owner or publishing COPYING.
    if (old_key >= 0) {
        const int32_t old_layer = old_key / g.lru.n_expert;
        const int32_t old_expert = old_key % g.lru.n_expert;
        g.lru.slot_map[old_layer][old_expert] = -1;
    }
    pending.generation = g.lru.advance_generation(slot);
    g.lru.key_of[slot] = key;
    g.lru.state[slot] = moe_cache_lru::slot_state::COPYING;
    g.lru.unlink(slot);
    *job = pending;
    return true;
}

bool llama_moe_cache::finish_fill(const ggml_cuda_moe_ffn_fill_job * job, bool copied) {
    if (!job || job->owner != this) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (job->group < 0 || (size_t) job->group >= pimpl->groups.size() ||
            job->layer < 0 || (size_t) job->layer >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[job->group];
    auto & l = pimpl->layers[job->layer];
    if (!g.frozen || g.generation != job->owner_generation || g.device != job->device ||
            l.group != job->group || job->slot < 0 || job->slot >= g.lru.n_slots ||
            job->key < 0 || job->key / g.lru.n_expert != job->layer ||
            !g.lru.slot_map[job->layer] || g.lru.state[job->slot] != moe_cache_lru::slot_state::COPYING ||
            g.lru.key_of[job->slot] != job->key || g.lru.slot_generation[job->slot] != job->generation ||
            g.lru.pin_count[job->slot] != 0 || g.lru.slot_map[job->layer][job->key % g.lru.n_expert] >= 0) return false;

    if (copied) {
        g.lru.slot_map[job->layer][job->key % g.lru.n_expert] = job->slot;
        g.lru.state[job->slot] = moe_cache_lru::slot_state::READY;
        g.lru.link_tail(job->slot);
        // The key is resident now; a future eviction must earn its way back.
        g.lru.reset_miss_demand(job->key);
    } else {
        g.lru.key_of[job->slot] = -1;
        g.lru.state[job->slot] = moe_cache_lru::slot_state::FREE;
        g.lru.link_tail(job->slot);
    }
    return true;
}

namespace {
bool llama_moe_cache_resolve_callback(void * context, int32_t device, int32_t group, int32_t layer,
        uint64_t owner_generation, const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result) {
    return context && static_cast<llama_moe_cache *>(context)->resolve_and_pin(
            device, group, layer, owner_generation, ids, n_ids, result);
}

bool llama_moe_cache_release_callback(void * context, const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    return context && static_cast<llama_moe_cache *>(context)->release_pins(pins, n_pins);
}

bool llama_moe_cache_reserve_fill_callback(void * context, int32_t device, int32_t group,
        int32_t layer, uint64_t owner_generation, int32_t expert, ggml_cuda_moe_ffn_fill_job * job,
        ggml_cuda_moe_ffn_fill_part * parts, size_t part_capacity) {
    return context && static_cast<llama_moe_cache *>(context)->reserve_fill(
            device, group, layer, owner_generation, expert, job, parts, part_capacity);
}

bool llama_moe_cache_finish_fill_callback(void * context, const ggml_cuda_moe_ffn_fill_job * job, bool copied) {
    return context && static_cast<llama_moe_cache *>(context)->finish_fill(job, copied);
}
} // namespace

const ggml_cuda_moe_ffn_cache_callbacks * llama_moe_cache_get_cuda_callbacks() {
    static const ggml_cuda_moe_ffn_cache_callbacks callbacks = {
        llama_moe_cache_resolve_callback,
        llama_moe_cache_release_callback,
        llama_moe_cache_reserve_fill_callback,
        llama_moe_cache_finish_fill_callback,
    };
    return &callbacks;
}

bool llama_moe_cache::freeze_group_for_layer(int32_t il, ggml_backend * expected_backend,
        frozen_lease & lease) {
    if (lease) return false;
    llama_moe_cache_frozen_layer descriptor;
    if (!pimpl->freeze_group_for_layer(il, expected_backend, descriptor)) return false;
    lease.owner = this;
    lease.descriptor = std::move(descriptor);
    return true;
}

bool llama_moe_cache::frozen_snapshot_valid(const llama_moe_cache_frozen_layer & descriptor) const {
    return pimpl->frozen_snapshot_valid(descriptor);
}

bool llama_moe_cache::frozen_layer_snapshot(const llama_moe_cache_frozen_layer & group_snapshot,
        int32_t il, llama_moe_cache_frozen_layer & snapshot) const {
    return pimpl->frozen_layer_snapshot(group_snapshot, il, snapshot);
}

bool llama_moe_cache::release_frozen_group(const llama_moe_cache_frozen_layer & descriptor) {
    return pimpl->release_frozen_group(descriptor);
}
