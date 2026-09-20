#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <vector>

// Full (unsharded) weight for the mirrored K2 tensors in split mode.
// Norms and the MoVA router are replicated on every device, so any
// available split is a complete copy; without splits the tensor itself
// is used (layer mode).
static ggml_tensor * k2_tp_full_weight(ggml_tensor * t) {
    if (t != nullptr && t->extra != nullptr) {
        auto split = (ggml_split_tensor_t *) t->extra;
        for (int id = 0; id < split->n_device; ++id) {
            if (split->splits[id] != nullptr) {
                return split->splits[id];
            }
        }
    }
    return t;
}

// K2 Horizon MoVA: Mixture of Value Attention, operating on explicit tensors
// so both the layer path (full tensors) and the tensor-parallel path
// (mirrored router, head-split value experts) share the same math.
static ggml_tensor * k2_horizon_routed_value_tensors(
    ggml_context * ctx,
    llama_context & lctx,
    ggml_tensor * v_gate,
    ggml_tensor * v_gate_b, // may be nullptr
    ggml_tensor * v_exps,   // full or head-split along ne[1]
    ggml_tensor * cur,
    const llama_hparams & hparams,
    const llm_build_cb & cb,
    int cb_il) {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_values = hparams.n_value_expert;
    const int64_t n_used   = hparams.n_value_expert_used;

    GGML_ASSERT(v_gate != nullptr);
    GGML_ASSERT(v_exps != nullptr);

    // router logits: (n_embd) . (n_embd, n_values) -> (n_values, n_tokens)
    ggml_tensor * logits = llm_build_context::llm_build_lora_mm(lctx, ctx, v_gate, cur);

    // gating function
    ggml_tensor * probs;
    switch (hparams.expert_gating_func) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX:
            probs = ggml_soft_max(ctx, logits);
            break;
        case LLM_EXPERT_GATING_FUNC_SIGMOID:
            probs = ggml_sigmoid(ctx, logits);
            break;
        default:
            GGML_ABORT("Unsupported K2 Horizon value-router gating function");
    }

    cb(logits, "v_moe_logits", cb_il);
    cb(probs, "v_moe_probs", cb_il);

    // the bias selects the experts; the weights come from the unbiased probs
    ggml_tensor * choice_probs = probs;
    if (v_gate_b != nullptr) {
        choice_probs = ggml_add(ctx, probs, v_gate_b);
        cb(choice_probs, "v_moe_probs_biased", cb_il);
    }

    // top-k selection
    ggml_tensor * selected_experts = ggml_top_k(ctx, choice_probs, n_used); // [n_used, n_tokens]
    cb(selected_experts, "v_topk", cb_il);

    // extract selected weights via argsort-style indexing
    ggml_tensor * selection_probs = ggml_reshape_3d(ctx, probs, 1, n_values, n_tokens);
    ggml_tensor * selected_weights = ggml_get_rows(ctx, selection_probs, selected_experts);
    cb(selected_weights, "v_weights", cb_il);
    // [1, n_used, n_tokens]

    // normalize weights (conditional on expert_weights_norm, matching upstream)
    if (hparams.expert_weights_norm) {
        selected_weights = ggml_reshape_2d(ctx, selected_weights, n_used, n_tokens);
        ggml_tensor * wsum = ggml_sum_rows(ctx, selected_weights);
        cb(wsum, "v_sum_rows", cb_il);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        cb(wsum, "v_clamp", cb_il);
        selected_weights = ggml_div(ctx, selected_weights, wsum);
        cb(wsum, "v_div", cb_il);
        selected_weights = ggml_reshape_3d(ctx, selected_weights, 1, n_used, n_tokens);
        cb(selected_weights, "v_moe_weights_norm", cb_il);
    }

    // expert weights scaling (matching upstream)
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        selected_weights = ggml_scale(ctx, selected_weights, hparams.expert_weights_scale);
        cb(selected_weights, "v_moe_weights_scaled", cb_il);
    }

    // compute routed values: indexed matmul + silu + weighted sum
    // v_exps: (n_embd, n_embd_v, n_values)
    // value_inp:   (n_embd, 1, n_tokens)
    ggml_tensor * value_inp = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * values = llm_build_context::llm_build_lora_mm_id(lctx, ctx, v_exps, value_inp, selected_experts);
    cb(values, "v_values", cb_il);
    // values: (n_embd_v, n_used, n_tokens)
    values = ggml_silu(ctx, values);
    cb(values, "v_silu", cb_il);

    auto value_out = ggml_mul_multi_add(ctx, values, selected_weights);

    cb(value_out, "Vcur_routed", cb_il);
    return value_out;
}

// K2 Horizon MoVA: Mixture of Value Attention
static ggml_tensor * k2_horizon_routed_value(
    ggml_context * ctx,
    llama_context & lctx,
    const llama_layer & layer,
    ggml_tensor * cur,
    int il,
    const llama_hparams & hparams,
    const llm_build_cb & cb) {
    GGML_ASSERT(layer.attn_v_gate != nullptr);
    GGML_ASSERT(layer.attn_v_exps != nullptr);

    return k2_horizon_routed_value_tensors(ctx, lctx, layer.attn_v_gate, layer.attn_v_gate_b,
            layer.attn_v_exps, cur, hparams, cb, il);
}

ggml_cgraph * llm_build_context::build_k2horizon() {
    const bool tp_mode = model.split_mode == LLAMA_SPLIT_MODE_TENSOR_PARALLEL ||
                         model.split_mode == LLAMA_SPLIT_MODE_ATTN;

    ggml_cgraph * gf = new_graph_custom();

    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // 1. Embedding
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // 2. Position
    struct ggml_tensor * inp_pos = build_inp_pos();

    // 3. Attention mask
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // 4. Output IDs
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // 5. Scale
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // 6. RoPE cache (precomputed lookup, avoids per-layer recomputation)
    // Not used in tensor-parallel mode: each device applies RoPE to its own
    // head shard directly (same convention as build_qwen3).
    ggml_tensor * rope_cache = nullptr;
    if (!tp_mode && cparams.rope_cache && (hparams.rope_type == LLAMA_ROPE_TYPE_NEOX || hparams.rope_type == LLAMA_ROPE_TYPE_NORM)) {
        const int64_t n_rot = hparams.n_embd_head_k(0);
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, hparams.rope_type,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }

    // 7. Layer loop
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        const bool is_moe_layer = hparams.n_expert > 0 &&
            static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0;

        // Tensor-parallel layer: Q/K split by heads, V split by heads (standard
        // wv or MoVA value experts), per-device KV store into the split caches,
        // wo shard per device, then reduce. Falls back to the layer path when
        // any of the sharded tensors (or the split KV cache) is missing, e.g.
        // for layers offloaded to the CPU.
        bool is_tp_layer = false;
        if (tp_mode && model.layers[il].wo && model.layers[il].wo->extra &&
                model.layers[il].wk && model.layers[il].wk->extra &&
                kv_self.k_l[il] && kv_self.k_l[il]->extra &&
                kv_self.v_l[il] && kv_self.v_l[il]->extra &&
                (!model.layers[il].wqkv_gate || model.layers[il].wqkv_gate->extra) &&
                (!model.layers[il].attn_q_norm || model.layers[il].attn_q_norm->extra) &&
                (!model.layers[il].attn_k_norm || model.layers[il].attn_k_norm->extra)) {
            if (is_mova_layer) {
                is_tp_layer = model.layers[il].attn_v_exps && model.layers[il].attn_v_exps->extra != nullptr;
            } else {
                is_tp_layer = model.layers[il].wv && model.layers[il].wv->extra != nullptr;
            }
        }

        // === grouped RMS norm before attention ===
        // The norm weights are mirrored on every device, so a single norm node
        // on the full hidden state feeds all per-device matmuls below.
        cur = ggml_fused_grouped_rms_norm(ctx0, inpL, k2_tp_full_weight(model.layers[il].attn_norm), hparams.f_norm_rms_eps, hparams.n_norm_groups);
        cb(cur, "attn_norm", il);

        ggml_tensor * attn_inp = cur;

        if (is_tp_layer) {
            // === tensor-parallel attention ===
            auto wq_sp = (ggml_split_tensor_t *) model.layers[il].wq->extra;
            auto wk_sp = (ggml_split_tensor_t *) model.layers[il].wk->extra;
            auto wo_sp = (ggml_split_tensor_t *) model.layers[il].wo->extra;
            auto kl_sp = (ggml_split_tensor_t *) kv_self.k_l[il]->extra;
            auto vl_sp = (ggml_split_tensor_t *) kv_self.v_l[il]->extra;
            const int n_device = wq_sp->n_device;
            GGML_ASSERT(wk_sp->n_device == n_device && wo_sp->n_device == n_device);
            GGML_ASSERT(kl_sp->n_device == n_device && vl_sp->n_device == n_device);

            std::vector<ggml_tensor *> attn_parts(n_device, nullptr);
            int n_have = 0;
            int last_id = -1;
            for (int id = 0; id < n_device; ++id) {
                const int il_cb = 1000*(id+1) + il;
                auto split_wq = wq_sp->splits[id];
                auto split_wk = wk_sp->splits[id];
                auto split_wo = wo_sp->splits[id];
                auto split_kl = kl_sp->splits[id];
                auto split_vl = vl_sp->splits[id];
                ggml_tensor * split_wv    = nullptr;
                ggml_tensor * split_vexps = nullptr;
                if (is_mova_layer) {
                    auto ve_sp = (ggml_split_tensor_t *) model.layers[il].attn_v_exps->extra;
                    GGML_ASSERT(ve_sp && ve_sp->n_device == n_device);
                    split_vexps = ve_sp->splits[id];
                } else {
                    auto wv_sp = (ggml_split_tensor_t *) model.layers[il].wv->extra;
                    GGML_ASSERT(wv_sp && wv_sp->n_device == n_device);
                    split_wv = wv_sp->splits[id];
                }
                const bool have_v = is_mova_layer ? split_vexps != nullptr : split_wv != nullptr;
                const bool have = split_wq && split_wk && split_wo && split_kl && split_vl && have_v;
                const bool none = !split_wq && !split_wk && !split_wo && !split_kl && !split_vl && !have_v;
                GGML_ASSERT(have || none);
                if (!have) {
                    continue;
                }

                // === Q ===
                ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, split_wq, attn_inp);
                cb(Qcur, "Qcur", il_cb);
                if (model.layers[il].attn_q_norm) {
                    auto qn_sp = (ggml_split_tensor_t *) model.layers[il].attn_q_norm->extra;
                    ggml_tensor * qn = qn_sp ? qn_sp->splits[id] : model.layers[il].attn_q_norm;
                    GGML_ASSERT(qn && qn->ne[0] == split_wq->ne[1]);
                    GGML_ASSERT(split_wq->ne[1] % n_embd_head == 0);
                    Qcur = ggml_fused_grouped_rms_norm(ctx0, Qcur, qn, hparams.f_norm_rms_eps, split_wq->ne[1] / n_embd_head);
                    cb(Qcur, "Qcur_normed", il_cb);
                }
                ggml_build_forward_expand(gf, Qcur);

                // === K ===
                ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, split_wk, attn_inp);
                cb(Kcur, "Kcur", il_cb);
                if (model.layers[il].attn_k_norm) {
                    auto kn_sp = (ggml_split_tensor_t *) model.layers[il].attn_k_norm->extra;
                    ggml_tensor * kn = kn_sp ? kn_sp->splits[id] : model.layers[il].attn_k_norm;
                    GGML_ASSERT(kn && kn->ne[0] == split_wk->ne[1]);
                    GGML_ASSERT(split_wk->ne[1] % n_embd_head == 0);
                    Kcur = ggml_fused_grouped_rms_norm(ctx0, Kcur, kn, hparams.f_norm_rms_eps, split_wk->ne[1] / n_embd_head);
                    cb(Kcur, "Kcur_normed", il_cb);
                }
                ggml_build_forward_expand(gf, Kcur);

                // === V: standard or MoVA routed ===
                // The MoVA router is mirrored so every device selects the same
                // experts; the value experts are head-split like wv.
                ggml_tensor * Vcur = nullptr;
                int64_t n_embd_v_local = 0;
                if (is_mova_layer) {
                    auto vg_sp = (ggml_split_tensor_t *) model.layers[il].attn_v_gate->extra;
                    ggml_tensor * vg = vg_sp ? vg_sp->splits[id] : model.layers[il].attn_v_gate;
                    GGML_ASSERT(vg);
                    ggml_tensor * vgb = nullptr;
                    if (model.layers[il].attn_v_gate_b) {
                        auto vgb_sp = (ggml_split_tensor_t *) model.layers[il].attn_v_gate_b->extra;
                        vgb = vgb_sp ? vgb_sp->splits[id] : model.layers[il].attn_v_gate_b;
                    }
                    Vcur = k2_horizon_routed_value_tensors(ctx0, lctx, vg, vgb, split_vexps,
                            attn_inp, hparams, cb, il_cb);
                    n_embd_v_local = split_vexps->ne[1];
                } else {
                    Vcur = llm_build_lora_mm(lctx, ctx0, split_wv, attn_inp);
                    cb(Vcur, "Vcur", il_cb);
                    n_embd_v_local = split_wv->ne[1];
                }
                ggml_build_forward_expand(gf, Vcur);

                const int64_t n_head_local    = split_wq->ne[1] / n_embd_head;
                const int64_t n_head_kv_local = split_wk->ne[1] / n_embd_head;
                GGML_ASSERT(n_embd_v_local == n_head_kv_local * n_embd_head);

                // reshape Q/K/V
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_local, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_local, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_local, n_tokens);

                // RoPE per device (no shared rope_cache in split mode)
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il_cb);
                cb(Kcur, "Kcur_rope", il_cb);
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);

                // === per-device KV store into the split caches ===
                const int idx = 2*n_device*il + 2*id;
                GGML_ASSERT(idx+1 < (int) lctx.cache_copies.size());
                const auto k_row_size = ggml_row_size(split_kl->type, n_embd_head);
                ggml_tensor * k_cache_view = ggml_view_2d(ctx0, split_kl, n_embd_head, n_tokens*n_head_kv_local,
                        k_row_size, k_row_size*n_head_kv_local*kv_head);

                lctx.cache_copies[idx+0].cpy  = ggml_cpy(ctx0, Kcur, k_cache_view);
                lctx.cache_copies[idx+0].step = k_row_size*n_head_kv_local;

                // note: storing RoPE-ed version of K in the KV cache
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+0].cpy);

                ggml_tensor * v_cache_view = nullptr;
                if (cparams.flash_attn) {
                    v_cache_view = ggml_view_1d(ctx0, split_vl, n_tokens*n_embd_v_local,
                            kv_head*ggml_row_size(split_vl->type, n_embd_v_local));
                    lctx.cache_copies[idx+1].step = ggml_row_size(split_vl->type, n_embd_v_local);
                } else {
                    // note: the V cache is transposed when not using flash attention
                    v_cache_view = ggml_view_2d(ctx0, split_vl, n_tokens, n_embd_v_local,
                            n_kv*ggml_element_size(split_vl),
                            kv_head*ggml_element_size(split_vl));
                    lctx.cache_copies[idx+1].step = ggml_element_size(split_vl);

                    Vcur = ggml_transpose(ctx0, Vcur);
                }
                cb(v_cache_view, "v_cache_view", il_cb);

                lctx.cache_copies[idx+1].cpy = ggml_cpy(ctx0, Vcur, v_cache_view);
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+1].cpy);

                // === flash attention on this device's heads ===
                ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q, "q", il_cb);
                auto k = ggml_view_3d(ctx0, split_kl, n_embd_head, n_kv, n_head_kv_local,
                        k_row_size*n_head_kv_local, k_row_size, 0);
                cb(k, "k", il_cb);
                const auto v_row_size = ggml_row_size(split_vl->type, n_embd_head);
                const auto vnb1 = ggml_row_size(split_vl->type, n_embd_v_local);
                auto v = ggml_view_3d(ctx0, split_vl, n_embd_head, n_kv, n_head_kv_local,
                        vnb1, v_row_size, 0);
                cb(v, "v", il_cb);

                // q here is this device's slice, so q->ne[2] is the per-device head
                // count. A per-head mask would be indexed by the device-local head,
                // so single-plane is the only correct case.
                GGML_ASSERT(!KQ_mask || KQ_mask->ne[2] == 1);
                ggml_tensor * fa = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask, kq_scale,
                        hparams.f_max_alibi_bias,
                        hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
                cb(fa, "flash_attn", il_cb);
                ggml_flash_attn_ext_add_sinks(fa, nullptr);
                // K2-Horizon needs F32 attention precision (same as the layer path
                // via llm_build_kv).
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

                cur = ggml_reshape_2d(ctx0, fa, split_wo->ne[0], n_tokens);
                cb(cur, "flash_attn_reshaped", il_cb);

                if (model.layers[il].wqkv_gate) {
                    // attention with softplus gating, per device
                    auto gate_sp = (ggml_split_tensor_t *) model.layers[il].wqkv_gate->extra;
                    ggml_tensor * gate_w = gate_sp ? gate_sp->splits[id] : model.layers[il].wqkv_gate;
                    GGML_ASSERT(gate_w);

                    constexpr float LN2 = 0.6931471805599453f;
                    constexpr float ONE_OVER_LN2 = 1.4426950408889634f;

                    ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, gate_w, attn_inp);
                    gate = ggml_scale(ctx0, gate, LN2);
                    gate = ggml_softplus(ctx0, gate);
                    gate = ggml_scale(ctx0, gate, ONE_OVER_LN2);
                    cb(gate, "attn_gate", il_cb);

                    cur = ggml_mul(ctx0, cur, gate);
                    cb(cur, "attn_gated", il_cb);
                }

                cur = llm_build_lora_mm(lctx, ctx0, split_wo, cur);
                cb(cur, "kqv_wo", il_cb);
                ggml_build_forward_expand(gf, cur);
                attn_parts[id] = cur;
                last_id = id;
                ++n_have;
            }
            GGML_ASSERT(n_have > 0);
            if (n_have > 1) {
                cur = ggml_reduce(ctx0, attn_parts.data(), n_device, GGML_OP_ADD);
            } else {
                cur = attn_parts[last_id];
            }
            cb(cur, "attn_combined", il);
            ggml_build_forward_expand(gf, cur);
        } else {
            // === Q ===
            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            if (model.layers[il].attn_q_norm != nullptr) {
                Qcur = ggml_fused_grouped_rms_norm(ctx0, Qcur, model.layers[il].attn_q_norm, hparams.f_norm_rms_eps, hparams.n_head(il));
            }

            // === K ===
            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            if (model.layers[il].attn_k_norm != nullptr) {
                Kcur = ggml_fused_grouped_rms_norm(ctx0, Kcur, model.layers[il].attn_k_norm, hparams.f_norm_rms_eps, hparams.n_head_kv(il));
            }

            // === V: standard or MoVA routed ===
            ggml_tensor * Vcur;
            if (is_mova_layer) {
                Vcur = k2_horizon_routed_value(ctx0, lctx, model.layers[il], cur, il, hparams, cb);
            } else {
                Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            }

            // reshape Q/K/V
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il), n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

            // RoPE — use fast path when rope_cache is available
            if (rope_cache) {
                Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // === Attention via llm_build_kv (handles KV store + attention in one call) ===
            // Replaces direct ggml_flash_attn_ext() which crashes on k2-horizon's
            // 128 head_dim with quantized KV cache (IQK FA unsupported K-type).
            // llm_build_kv stores K/V into cache AND computes Q*K attention internally.
            // n_kv is the class member (KV cache size), not n_head_kv
            if (model.layers[il].wqkv_gate == nullptr) {
                // standard attention
                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].wo_b,
                        Kcur, Vcur, Qcur,
                        KQ_mask, n_tokens, kv_head, n_kv,
                        kq_scale, cb, il);
            } else {
                // attention with softplus gating — no output projection yet
                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        nullptr, nullptr,
                        Kcur, Vcur, Qcur,
                        KQ_mask, n_tokens, kv_head, n_kv,
                        kq_scale, cb, il);

                // softplus gate
                constexpr float LN2 = 0.6931471805599453f;
                constexpr float ONE_OVER_LN2 = 1.4426950408889634f;

                ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, attn_inp);
                gate = ggml_scale(ctx0, gate, LN2);
                gate = ggml_softplus(ctx0, gate);
                gate = ggml_scale(ctx0, gate, ONE_OVER_LN2);

                cur = ggml_mul(ctx0, cur, gate);
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                if (model.layers[il].wo_b != nullptr) {
                    cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
                }
            }
        }

        // last token selection
        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpSA);
        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // === grouped RMS norm before FFN ===
        // The norm weights are mirrored on every device, so a single norm node
        // on the full hidden state feeds the (possibly split) FFN below.
        cur = ggml_fused_grouped_rms_norm(ctx0, ffn_inp, k2_tp_full_weight(model.layers[il].ffn_norm), hparams.f_norm_rms_eps, hparams.n_norm_groups);
        cb(cur, "ffn_norm", il);

        // === FFN (dense or MoE) ===
        if (is_moe_layer) {
            ggml_tensor * moe_out;
            if (is_tp_layer) {
                // Tensor-parallel MoE: the norm was applied once above, the
                // split-aware helper distributes experts across devices.
                // The shared expert is handled below via llm_build_ffn, which
                // branches to its split path automatically when sharded.
                moe_out = llm_build_std_moe_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_gate_inp,  nullptr,
                        model.layers[il].ffn_up_exps,   nullptr,
                        model.layers[il].ffn_gate_exps, nullptr,
                        model.layers[il].ffn_down_exps, nullptr,
                        model.layers[il].ffn_exp_probs_b,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        nullptr, nullptr,
                        hparams.n_expert, hparams.n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        LLM_FFN_SILU, cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            } else {
                moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        hparams.n_expert, hparams.n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type)hparams.expert_gating_func,
                        cb, il, gf, false,
                        model.layers[il].ffn_up_gate_exps);
            }

            if (model.layers[il].ffn_gate_shexp != nullptr) {
                ggml_tensor * shared_out = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                        model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                        model.layers[il].ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
                cur = ggml_add(ctx0, moe_out, shared_out);
            } else {
                cur = moe_out;
            }
        } else {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
        }
        cb(cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // === Final grouped RMS norm ===
    cur = ggml_fused_grouped_rms_norm(ctx0, inpL, k2_tp_full_weight(model.output_norm), hparams.f_norm_rms_eps, hparams.n_norm_groups);
    cb(cur, "result_norm", -1);

    // === Vocab projection (split-aware when --split-output-tensor is used) ===
    if (model.output->extra) {
        auto out_sp = (ggml_split_tensor_t *) model.output->extra;
        std::vector<ggml_tensor *> o;
        o.reserve(out_sp->n_device);
        for (int id = 0; id < out_sp->n_device; ++id) {
            auto split = out_sp->splits[id];
            if (!split) {
                continue;
            }
            o.push_back(llm_build_lora_mm(lctx, ctx0, split,
                    get_input_tensor_sm_graph(ctx0, cur, id)));
            cb(o.back(), "output", id);
        }
        GGML_ASSERT(!o.empty());
        cur = o.front();
        for (size_t id = 1; id < o.size(); ++id) {
            cur = ggml_concat(ctx0, cur, o[id], 0);
        }
    } else {
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
