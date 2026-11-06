#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_ricursive() {
    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_vres = inpL;

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = kv_self.any_compacted()
        ? build_swa_mask_for_graph(hparams.n_swa, true)
        : build_inp_KQ_mask_swa();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head_k));

    const bool tp_mode   = model.split_mode == LLAMA_SPLIT_MODE_TENSOR_PARALLEL ||
                           model.split_mode == LLAMA_SPLIT_MODE_ATTN;
    const bool add_input = !tp_mode;

    post_norm_data pnd;
    pnd.f_rms_eps = hparams.f_norm_rms_eps;
    post_norm_data * pnd_ptr = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];

        if (tp_mode && il > 0) {
            GGML_ASSERT(pnd.next_input.size() == model.devices.size());
            pnd.norm = model.layers[il-1].ffn_post_norm;
            pnd_ptr = &pnd;
        }

        auto cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                is_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
                is_swa ? hparams.n_swa : 0, il, hparams.has_rope(il), false, add_input, false, false,
                model.layers[il].attn_post_norm, -1, hparams.f_norm_rms_eps, pnd_ptr,
                nullptr, nullptr, nullptr, inp_vres);

        if (tp_mode) {
            pnd_ptr = &pnd;
            if (il == 0) {
                pnd.next_input.resize(model.devices.size(), inpL);
            } else {
                GGML_ASSERT(pnd.next_input.size() == model.devices.size());
            }
            pnd.norm = model.layers[il].attn_post_norm;
        }

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, add_input, false, nullptr,
                model.layers[il].ffn_post_norm, hparams.f_norm_rms_eps, pnd_ptr);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;

    if (tp_mode) {
        // the final ffn_post_norm + residual are deferred to here because no layer follows
        GGML_ASSERT(inpL->op == GGML_OP_REDUCE);
        int idx = model.default_layer_device[n_layer];
        cur = inpL->src[idx];
        if (!cur) {
            for (idx = 0; idx < int(model.devices.size()); ++idx) {
                if (inpL->src[idx]) {
                    cur = inpL->src[idx]; break;
                }
            }
            GGML_ASSERT(cur);
        }
        auto pn_extra = (ggml_split_tensor_t *)model.layers[n_layer-1].ffn_post_norm->extra;
        GGML_ASSERT(pn_extra && pn_extra->splits[idx]);
        cur = ggml_fused_rms_norm(ctx0, cur, pn_extra->splits[idx], pnd.f_rms_eps);
        cb(cur, "ffn_post_norm", n_layer-1);
        GGML_ASSERT(idx < (int)pnd.next_input.size());
        auto add = pnd.next_input[idx];
        if (!add) {
            for (int j = 0; j < int(pnd.next_input.size()); ++j) {
                if (pnd.next_input[j]) {
                    add = pnd.next_input[j]; break;
                }
            }
            GGML_ASSERT(add);
        }
        cur = ggml_add(ctx0, cur, add);
        cb(cur, "ffn_final", -1);
    }

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
