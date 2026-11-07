#pragma once

#include "ggml.h"

#include <stdint.h>
#include <math.h>

enum ggml_moe_ffn_storage_topology {
    GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN = 0,
    GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN = 1,
};

enum ggml_moe_ffn_layout_flag {
    GGML_MOE_FFN_LAYOUT_PHYSICAL_STORAGE_VALID = 1u << 0,
    GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS = 1u << 1,
    GGML_MOE_FFN_LAYOUT_NO_HADAMARD = 1u << 2,
    GGML_MOE_FFN_LAYOUT_HOST_RESIDENT = 1u << 3,
};

enum ggml_moe_ffn_graph_flag {
    GGML_MOE_FFN_GRAPH_ALL_TOKEN_OUTPUTS = 1u << 0,
    GGML_MOE_FFN_GRAPH_NO_LORA = 1u << 1,
    GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS = 1u << 2,
    GGML_MOE_FFN_GRAPH_SILU_LIMIT = 1u << 3,
    GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS = 1u << 4,
    GGML_MOE_FFN_GRAPH_NO_HADAMARD = 1u << 5,
};

struct ggml_moe_ffn_layout {
    int32_t n_experts, n_experts_used, n_embd, n_ff, n_slots;
    int32_t n_tokens, token_capacity;
    int32_t device, cache_group, cache_layer;
    enum ggml_moe_ffn_storage_topology storage_topology;
    const void * owner_backend;
    const void * cache_owner;
    const struct ggml_tensor * up, * gate, * down;
    const struct ggml_tensor * bank_up, * bank_gate, * bank_down;
    const struct ggml_tensor * up_gate_storage, * bank_up_gate_storage;
    const struct ggml_tensor * activation, * selected_ids, * route_weights;
    enum ggml_unary_op unary_op;
    float limit;
    uint32_t flags;
    uint32_t graph_flags;
};

static inline bool ggml_moe_ffn_expert_execution_inputs_supported(
        const struct ggml_tensor * up_bias, const struct ggml_tensor * gate_bias,
        const struct ggml_tensor * down_bias, const struct ggml_tensor * merged_up_gate_bias,
        const struct ggml_tensor * down_scale) {
    return !up_bias && !gate_bias && !down_bias && !merged_up_gate_bias && !down_scale;
}

static inline bool ggml_moe_ffn_unary_supported(enum ggml_unary_op op) {
    // These are the activations implemented by both the CPU fused up/gate
    // graph and the CUDA hit kernels.  SITU and plain SWIGLU remain outside
    // the semantic contract.
    return op == GGML_UNARY_OP_SILU || op == GGML_UNARY_OP_RELU ||
        op == GGML_UNARY_OP_GELU || op == GGML_UNARY_OP_SWIGLU_OAI;
}

static inline bool ggml_moe_ffn_tensor_shape(const struct ggml_tensor * tensor,
        enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    return tensor && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
        tensor->ne[2] == ne2 && tensor->ne[3] == 1 && ne0 > 0 && ne1 > 0 && ne2 > 0 &&
        ggml_n_dims(tensor) == (ne2 > 1 ? 3 : ne1 > 1 ? 2 : 1);
}

static inline bool ggml_moe_ffn_tensor_layout(const struct ggml_tensor * tensor,
        enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!ggml_moe_ffn_tensor_shape(tensor, type, ne0, ne1, ne2) ||
        tensor->nb[0] != ggml_type_size(type)) {
        return false;
    }

    const size_t row_bytes = ggml_row_size(type, ne0);
    if (row_bytes == 0 || row_bytes > SIZE_MAX / (size_t) ne1) {
        return false;
    }
    const size_t expert_bytes = row_bytes * (size_t) ne1;
    if (expert_bytes > SIZE_MAX / (size_t) ne2) {
        return false;
    }

    // The executor addresses experts with nb[2], so each slice must be exactly its packed row extent
    return tensor->nb[1] == row_bytes && tensor->nb[2] == expert_bytes &&
        tensor->nb[3] == expert_bytes * (size_t) ne2 && ggml_is_contiguous(tensor);
}

static inline bool ggml_moe_ffn_selected_ids_layout(const struct ggml_tensor * tensor,
        int64_t n_selected, int64_t n_tokens) {
    if (!ggml_moe_ffn_tensor_shape(tensor, GGML_TYPE_I32, n_selected, n_tokens, 1) ||
            tensor->nb[0] != sizeof(int32_t)) return false;
    size_t row_bytes = 0, rows_extent = 0, visible_extent = 0;
    if ((size_t) n_selected > SIZE_MAX / sizeof(int32_t)) return false;
    row_bytes = (size_t) n_selected * sizeof(int32_t);
    if (tensor->nb[1] < row_bytes || tensor->nb[1] > SIZE_MAX / (size_t) n_tokens) return false;
    rows_extent = tensor->nb[1] * (size_t) n_tokens;
    if (tensor->nb[2] != rows_extent || tensor->nb[3] != rows_extent ||
            (size_t) (n_tokens - 1) > SIZE_MAX / tensor->nb[1]) return false;
    visible_extent = (size_t) (n_tokens - 1) * tensor->nb[1];
    if (visible_extent > SIZE_MAX - row_bytes) return false;
    visible_extent += row_bytes;
    if (!tensor->view_src) return ggml_is_contiguous(tensor) && tensor->nb[1] == row_bytes;
    if (!ggml_is_contiguous(tensor->view_src) || tensor->view_offs > ggml_nbytes(tensor->view_src) ||
            visible_extent > ggml_nbytes(tensor->view_src) - tensor->view_offs) return false;
    return true;
}

static inline bool ggml_moe_ffn_bank_layout_matches(const struct ggml_tensor * host,
        const struct ggml_tensor * bank) {
    return host && bank && host->type == bank->type && host->ne[0] == bank->ne[0] &&
        host->ne[1] == bank->ne[1] && host->ne[3] == bank->ne[3] &&
        host->nb[0] == bank->nb[0] && host->nb[1] == bank->nb[1] && host->nb[2] == bank->nb[2];
}

static inline bool ggml_moe_ffn_merged_view_fits(const struct ggml_tensor * view,
        size_t view_offset, size_t part_bytes, size_t parent_extent) {
    if (view_offset > parent_extent || view->nb[2] == 0 ||
            (size_t) (view->ne[2] - 1) > (SIZE_MAX - part_bytes) / view->nb[2]) return false;
    const size_t extent = part_bytes + (size_t) (view->ne[2] - 1) * view->nb[2];
    return extent <= parent_extent - view_offset;
}

static inline bool ggml_moe_ffn_merged_views_match(const struct ggml_tensor * up,
        const struct ggml_tensor * gate, const struct ggml_tensor * parent) {
    size_t expected_up_offset = 0, row_bytes = 0, part_bytes = 0, parent_bytes = 0;
    if (!up || !gate || !parent || up->type != gate->type || up->type != parent->type ||
            up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] || up->ne[2] != gate->ne[2] ||
            up->ne[3] != 1 || gate->ne[3] != 1 || parent->ne[0] != up->ne[0] ||
            parent->ne[1] != up->ne[1] + gate->ne[1] || parent->ne[2] != up->ne[2] ||
            parent->ne[3] != 1 || parent->nb[0] != ggml_type_size(parent->type) ||
            !ggml_is_contiguous(parent)) return false;
    row_bytes = ggml_row_size(parent->type, parent->ne[0]);
    if (!row_bytes || row_bytes > SIZE_MAX / (size_t) gate->ne[1]) return false;
    part_bytes = row_bytes * (size_t) gate->ne[1];
    if (part_bytes > SIZE_MAX / 2 || part_bytes * 2 > SIZE_MAX / (size_t) parent->ne[2]) return false;
    parent_bytes = part_bytes * 2;
    if (parent->nb[1] != row_bytes || parent->nb[2] != parent_bytes ||
            up->nb[0] != parent->nb[0] || gate->nb[0] != parent->nb[0] ||
            up->nb[1] != parent->nb[1] || gate->nb[1] != parent->nb[1]) return false;
    const struct ggml_tensor * view_root = parent->view_src ? parent->view_src : parent;
    if (up->view_src != view_root || gate->view_src != view_root ||
            up->view_offs < parent->view_offs || gate->view_offs < parent->view_offs) return false;
    const size_t up_offset = up->view_offs - parent->view_offs;
    const size_t gate_offset = gate->view_offs - parent->view_offs;
    expected_up_offset = part_bytes;
    const size_t parent_extent = parent_bytes * (size_t) parent->ne[2];
    // Native merged tensors expose per-expert interleaved views. The model loader's
    // merge flag instead creates a single parent tensor. Both layouts are supported.
    const bool interleaved = gate_offset == 0 && up_offset == expected_up_offset &&
        up->nb[2] == parent->nb[2] && gate->nb[2] == parent->nb[2];
    const bool expert_planes = gate_offset == 0 &&
        up_offset == part_bytes * (size_t) parent->ne[2] &&
        up->nb[2] == part_bytes && gate->nb[2] == part_bytes;
    if (!interleaved && !expert_planes) return false;
    const size_t plane_bytes = part_bytes * (size_t) parent->ne[2];
    const bool planes_disjoint = up_offset >= gate_offset + plane_bytes ||
        gate_offset >= up_offset + plane_bytes;
    return ggml_moe_ffn_merged_view_fits(up, up_offset, part_bytes, parent_extent) &&
        ggml_moe_ffn_merged_view_fits(gate, gate_offset, part_bytes, parent_extent) &&
        (interleaved || planes_disjoint);
}

static inline bool ggml_moe_ffn_layout_supported(const struct ggml_moe_ffn_layout * l) {
    const uint32_t required = GGML_MOE_FFN_LAYOUT_PHYSICAL_STORAGE_VALID |
        GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS | GGML_MOE_FFN_LAYOUT_NO_HADAMARD |
        GGML_MOE_FFN_LAYOUT_HOST_RESIDENT;
    if (!l || l->n_experts <= 0 || l->n_experts_used < 1 || l->n_experts_used > l->n_experts ||
            l->n_embd <= 0 || l->n_ff <= 0 || !ggml_moe_ffn_unary_supported(l->unary_op) ||
            !isfinite(l->limit) || l->limit < 0.0f || (l->flags & required) != required ||
            !l->up || !l->gate || !l->down ||
            !ggml_moe_ffn_tensor_layout(l->down, l->down->type, l->n_ff, l->n_embd, l->n_experts)) return false;

    bool host_up_gate_ok = false, bank_up_gate_ok = false;
    if (l->storage_topology == GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN) {
        host_up_gate_ok = !l->up_gate_storage &&
            ggml_moe_ffn_tensor_layout(l->up, l->up->type, l->n_embd, l->n_ff, l->n_experts) &&
            ggml_moe_ffn_tensor_layout(l->gate, l->gate->type, l->n_embd, l->n_ff, l->n_experts);
        bank_up_gate_ok = !l->bank_up_gate_storage &&
            ggml_moe_ffn_tensor_layout(l->bank_up, l->up->type, l->n_embd, l->n_ff, l->n_slots) &&
            ggml_moe_ffn_tensor_layout(l->bank_gate, l->gate->type, l->n_embd, l->n_ff, l->n_slots) &&
            ggml_moe_ffn_bank_layout_matches(l->up, l->bank_up) &&
            ggml_moe_ffn_bank_layout_matches(l->gate, l->bank_gate);
    } else if (l->storage_topology == GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN) {
        host_up_gate_ok = ggml_moe_ffn_merged_views_match(l->up, l->gate, l->up_gate_storage);
        bank_up_gate_ok = ggml_moe_ffn_merged_views_match(l->bank_up, l->bank_gate,
                l->bank_up_gate_storage) &&
            ggml_moe_ffn_bank_layout_matches(l->up_gate_storage, l->bank_up_gate_storage);
    }
    const bool down_bank_ok = ggml_moe_ffn_tensor_layout(l->bank_down, l->down->type,
            l->n_ff, l->n_embd, l->n_slots) && ggml_moe_ffn_bank_layout_matches(l->down, l->bank_down);
    const bool no_cache = l->n_slots == 0 && !l->bank_up && !l->bank_gate && !l->bank_down &&
            !l->bank_up_gate_storage;
    const bool cached = l->n_slots > 0 && l->device >= 0 && l->cache_group >= 0 &&
            l->cache_layer >= 0 && l->owner_backend && l->cache_owner && l->bank_up && l->bank_gate &&
            l->bank_down && down_bank_ok;
    return host_up_gate_ok && (no_cache || (cached && bank_up_gate_ok));
}

static inline bool ggml_moe_ffn_graph_inputs_supported(const struct ggml_moe_ffn_layout * l) {
    return ggml_moe_ffn_layout_supported(l) && l->n_slots > 0 && l->n_tokens > 0 &&
        l->token_capacity > 0 && l->n_tokens <= l->token_capacity &&
        ggml_moe_ffn_tensor_layout(l->activation, GGML_TYPE_F32, l->n_embd, l->n_tokens, 1) &&
        ggml_moe_ffn_selected_ids_layout(l->selected_ids, l->n_experts_used, l->n_tokens) &&
        ggml_moe_ffn_tensor_layout(l->route_weights, GGML_TYPE_F32, 1, l->n_experts_used, l->n_tokens);
}

static inline bool ggml_moe_ffn_graph_supported(const struct ggml_moe_ffn_layout * l) {
    // ALL_TOKEN_OUTPUTS means the executor writes every row of the activation
    // batch, not that every row requests logits.  Partial-logits and prompt
    // batches are handled by the ordinary downstream row selection.
    uint32_t required = GGML_MOE_FFN_GRAPH_ALL_TOKEN_OUTPUTS | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS |
        GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS | GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    // Only SILU consumes the routed swiglu limit.
    if (l && l->unary_op == GGML_UNARY_OP_SILU) required |= GGML_MOE_FFN_GRAPH_SILU_LIMIT;
    return l && (l->graph_flags & required) == required && ggml_moe_ffn_graph_inputs_supported(l);
}

struct ggml_moe_ffn_op_params {
    int32_t device;
    int32_t binding_id;
    uint64_t generation;
    int32_t n_tokens;
    int32_t n_embd;
    int32_t n_ff;
    int32_t unary_op;
    float limit;
};
