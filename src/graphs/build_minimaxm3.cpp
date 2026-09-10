#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// =============================================================================
// MiniMax-M3 MSA (sparse attention) — block-sparse softmax over GQA.
//
// Per sparse layer a lightweight indexer scores all KV positions with a single
// shared index-key head, those per-token scores are max-pooled into blocks of
// B_k tokens, the top-k blocks per GQA group (plus the forced local block) are
// selected, expanded back to a per-token additive mask (ne[2]=n_head), and that
// mask substitutes for the dense KQ_mask in the main (full-precision softmax)
// attention. Arch-gated to MINIMAX_M3, off by default (--msa).
//
// Returns nullptr to mean "use the dense KQ_mask" — disabled, non-sparse layer,
// indexer tensors absent (e.g. a GGUF that dropped them), or cache missing. This
// makes the path byte-identical to the dense fallback whenever it is off, and
// (because top-k over all blocks selects everything) numerically identical to
// dense whenever k*B_k >= n_kv (short context) — the no-op-exact validation hook.
// =============================================================================
ggml_tensor * llm_build_context::build_minimaxm3_msa_mask(ggml_cgraph * gf,
        ggml_tensor * cur, ggml_tensor * inp_pos, ggml_tensor * KQ_mask, int il, msa_attn_split * msa,
        msa_shared * shared, ggml_tensor ** out_normed) {

    // MSA is off by default; opt in via the --msa CLI flag (cparams.msa). When disabled the
    // minimax-m3 model runs the dense MLA path, byte-identical to not having the feature.
    // Arch-gated to MINIMAX-M3 (mirrors the --dsa / GLM-DSA precedent in PR #2045).
    if (!cparams.msa)                                 return nullptr;
    if (model.arch != LLM_ARCH_MINIMAX_M3)            return nullptr;
    if (!hparams.minimax_sparse_layer(il))            return nullptr;
    const auto & layer = model.layers[il];
    if (!layer.index_q || !layer.index_k) {
        // Warn once: --msa is accepted and logged as enabled, so without this a conversion that
        // dropped the indexer looks like MSA running at dense speed with no explanation.
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: --msa requested but this GGUF has no indexer tensors; running dense. "
                           "Re-convert with a converter that writes the indexer.\n", __func__);
        }
        return nullptr;
    }
    if (kv_self.kr_l.size() <= (size_t) il || !kv_self.kr_l[il]) return nullptr; // no decode cache
    // The per-GQA-group selection emits a per-head mask (ne[2]=n_head). The CPU and CUDA soft_max
    // kernels and the flash-attn path all index the mask by head (i02 % ne[2]); both are validated
    // against dense within noise, with a selection-set diff of 0 vs the HF reference.
    const int64_t d_idx     = hparams.minimax_sparse_index_dim;
    const int64_t idx_heads = hparams.minimax_sparse_index_heads > 0
                                  ? (int64_t) hparams.minimax_sparse_index_heads
                                  : (int64_t) hparams.n_head_kv(il);
    const int64_t B_k       = hparams.minimax_sparse_block_size > 0
                                  ? (int64_t) hparams.minimax_sparse_block_size : 128;
    // --msa-top-k N overrides the model's configured topk_blocks (cparams.msa_top_k < 0 => use
    // the GGUF value; falls back to 16 if the GGUF carries no count). Mirrors --dsa-top-k.
    const int64_t topk_blk  = cparams.msa_top_k >= 0
                                  ? (int64_t) cparams.msa_top_k
                                  : (hparams.minimax_sparse_topk_blocks > 0
                                         ? (int64_t) hparams.minimax_sparse_topk_blocks : 16);

    GGML_ASSERT(KQ_mask && "MSA needs the dense causal mask for the floor");
    const int64_t n_kv_eff = KQ_mask->ne[0];      // == n_kv
    const int64_t n_blocks = (n_kv_eff + B_k - 1) / B_k;

    // Dense fallback, two independent reasons:
    //   (a) the budget covers every block, so MSA == dense and selection is pure cost;
    //   (b) --msa-min-kv N, below which the sparse path is a measured net loss (off by default,
    //       no portable default: the crossover depends on memory bandwidth and core count).
    // (b) attends a superset of the reference selection, trading fidelity for speed.
    //
    // This early-out MUST stay below the index-key cache write above. Early tokens on a growing
    // context are dense, so returning before the write leaves their cache cells uninitialised;
    // once n_kv crosses the block budget the scoring reads zeros and top-k drops genuinely
    // attended blocks, which collapses PPL. The unconditional write is also why the fallback
    // recovers only part of the penalty (~12% residue against pure dense at n_kv 2,240).
    const bool msa_dense = (topk_blk >= n_blocks)
                        || (cparams.msa_min_kv > 0 && n_kv_eff < (int64_t) cparams.msa_min_kv);

    // --- normed hidden (same X the main attention sees) ---
    ggml_tensor * x = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr,
            LLM_NORM_RMS, cb, il);
    if (out_normed) *out_normed = x;   // let the caller skip its identical attn_norm

    // RoPE params for the indexer (must match the reference: NEOX, n_rot=64, theta=5e6).
    // The reference applies the SAME partial RoPE the main attention uses
    // (apply_rotary_pos_emb(idx_q, idx_k, cos[..,:head_dim], sin[..,:head_dim]) where the
    // cos/sin width is rotary_dim=64), i.e. the first n_rot dims of the d_idx-wide index head
    // are rotated NEOX-style and dims [n_rot, d_idx) pass through unrotated.
    const int64_t n_rot_idx = hparams.n_rot;                  // 64 (rope.dimension_count)
    const float   idx_freq_base  = cparams.rope_freq_base;    // 5e6
    const float   idx_freq_scale = freq_scale;

    // --- index_q : {d_idx, idx_heads, n_tokens}, RMSNorm over d_idx, then partial NEOX RoPE ---
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.index_q, x);                 // {d_idx*idx_heads, n_tokens}
    iq = ggml_reshape_3d(ctx0, iq, d_idx, idx_heads, n_tokens);
    iq = llm_build_norm(ctx0, iq, hparams, layer.index_q_norm, nullptr, LLM_NORM_RMS, cb, il);
    iq = build_minimaxm3_index_rope(iq, inp_pos, n_rot_idx, d_idx, idx_heads, idx_freq_base, idx_freq_scale, il);

    // --- index_k : single shared head {d_idx, n_tokens}, RMSNorm, then partial NEOX RoPE ---
    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.index_k, x);                 // {d_idx, n_tokens}
    ik = llm_build_norm(ctx0, ik, hparams, layer.index_k_norm, nullptr, LLM_NORM_RMS, cb, il);
    ik = ggml_reshape_3d(ctx0, ik, d_idx, 1, n_tokens);                      // {d_idx, 1, n_tokens}
    ik = build_minimaxm3_index_rope(ik, inp_pos, n_rot_idx, d_idx, 1, idx_freq_base, idx_freq_scale, il);
    ik = ggml_reshape_2d(ctx0, ik, d_idx, n_tokens);                        // {d_idx, n_tokens}

    // --- write this batch's index keys into the persistent cache at kv_head, read back n_kv ---
    // ROBUSTNESS hardening: clamp the index keys to a safe finite F16 range BEFORE the cache write.
    // On a low-bit base (IQ2_M) the indexer RMSNorm output can saturate to F16 +/-inf (|x|>65504) for
    // some tokens; an inf in kr_l makes a later mul_mat +/-inf. A +inf in a LIVE cell survives the
    // block-max pool and ranks that block first, which is the hazard this removes. (An earlier
    // version of this comment said the danger was NaN surviving the pool. It is not: POOL_MAX
    // initialises to -FLT_MAX and tests `srow_j > drow[i]` (ggml.c), and every comparison against
    // NaN is false, so the pool DROPS NaN. Note also that ggml_clamp maps NaN to `max`, because
    // MIN/MAX are ternaries -- so a clamp is a NaN promoter, not a NaN guard.)
    // NOTE: this is unrelated to the FA -ub128 graph-reuse bug fixed just below (that was a
    // stride/offset bug zeroing recent indexer-key cells, no inf/NaN involved) -- this clamp is
    // defensive only and is a NO-OP on the shipping soft_max path (real keys are well within range).
    // 6e4 is just under the F16 max (65504); the indexer score is scale-free so the clamp value is
    // irrelevant to ranking.
    {
        ggml_tensor * kr = kv_self.kr_l[il];                                  // {d_idx, kv_size} F16
        ggml_tensor * dst = ggml_view_2d(ctx0, kr, d_idx, n_tokens,
                kr->nb[1], (size_t) kv_head * kr->nb[1]);
        // ggml_clamp is in-place and its CPU kernel visits rows ith, ith+nth, ... with ith==0 only,
        // so on a multi-row tensor it clamps 1 row in nth and passes the rest through unclamped.
        // Present the batch as a single row so the write is actually covered.
        ggml_tensor * ik_safe = ggml_clamp(ctx0, ggml_reshape_2d(ctx0, ik, d_idx*n_tokens, 1),
                -6.0e4f, 6.0e4f);                                             // kill F16 inf/nan at the source
        ik_safe = ggml_reshape_2d(ctx0, ik_safe, d_idx, n_tokens);
        ggml_tensor * kr_cpy = ggml_cpy(ctx0, ik_safe, dst);
        // GRAPH-REUSE FIXUP REGISTRATION: the K/V cache_copies fixup re-points the K/V write
        // offsets to the current kv_head when a graph is reused, but it does NOT touch
        // this indexer-key (kr_l) write. Under FA the cache pads to 256, so consecutive ubatches
        // keep the SAME n_kv and the graph IS reused -- without this registration the kr_l write
        // stays baked at the first ubatch's kv_head, so later ubatches never write their recent
        // index keys (those slots read 0.0) and the block-max-pool/top-k drops the genuinely
        // attended recent block (PPL 9.6 -> ~20). Register it like K/V so update_cache_copies()
        // patches view_offs = kv_head * step each reuse. step = one index-key row = kr->nb[1].
        if ((size_t) il < lctx.msa_cache_copies.size()) {
            lctx.msa_cache_copies[il].cpy  = kr_cpy;
            lctx.msa_cache_copies[il].step = kr->nb[1];
        }
        ggml_build_forward_expand(gf, kr_cpy);
    }

    // The index keys for this batch are now written to the cache; only now is it safe to take the dense
    // no-op path (see the note at the top). This is the P0 fix: never skip the index-key write.
    if (msa_dense) return nullptr;

    ggml_tensor * cached_k = ggml_view_2d(ctx0, kv_self.kr_l[il], d_idx, n_kv_eff,
            kv_self.kr_l[il]->nb[1], 0);                                       // {d_idx, n_kv}

    // --- per-(idx-head, token) scores against all cached keys: {n_kv, idx_heads, n_tokens} ---
    // mul_mat(cached_k {d_idx,n_kv}, iq {d_idx,idx_heads,n_tokens}) broadcasts the single
    // key head over the idx_heads dim. The reference uses NO 1/sqrt(d_idx) scale here
    // (scores = idx_q @ idx_k.T directly), so we do not scale either.
    ggml_tensor * scores = ggml_mul_mat(ctx0, cached_k, iq);                  // {n_kv, idx_heads, n_tokens}

    // causal floor: future / padding keys -> -inf so they never win a block max.
    // KQ_mask is {n_kv, n_tokens_padded}; take the first n_tokens cols and broadcast over idx_heads.
    // (reference: scores.masked_fill(k_pos > position_id, -inf) — strict, diagonal kept; the dense
    //  causal KQ_mask carries exactly that pattern in its first n_tokens columns.)
    // On the FA path KQ_mask is F16 (build_inp_KQ_mask casts it); the scoring math is F32, so
    // upcast the causal-floor view to F32. Off the FA path KQ_mask is already F32 (cast is a no-op
    // copy here, removed below by reusing the original when types match).
    const float BIG = 1e30f;
    // KQ_mask, n_kv and n_tokens are graph-level constants, so this chain is identical for every
    // sparse layer; build it on the first one and reuse.
    ggml_tensor * floor3 = shared ? shared->floor3 : nullptr;
    if (!floor3) {
        floor3 = ggml_view_2d(ctx0, KQ_mask, n_kv_eff, n_tokens, KQ_mask->nb[1], 0);
        floor3 = ggml_cont(ctx0, floor3);
        if (floor3->type != GGML_TYPE_F32) {
            floor3 = ggml_cast(ctx0, floor3, GGML_TYPE_F32);
        }
        floor3 = ggml_reshape_3d(ctx0, floor3, n_kv_eff, 1, n_tokens); // {n_kv,1,n_tok} F32
        if (shared) shared->floor3 = floor3;
    }
    // --- BlockMaxPool over kv per idx-head: {n_kv, idx_heads, n_tokens} -> {n_blocks, idx_heads, n_tokens} ---
    // Fold (idx_head, token) into one axis so a single pool_1d covers them all.
    const int64_t HT = idx_heads * n_tokens;                                  // folded (head,token) count
    const int64_t n_pad = n_blocks * B_k - n_kv_eff;
    ggml_tensor * blk;
    if (n_pad == 0) {
        // n_kv is block-aligned (always true under FA: the cache view pads to 256). Fold the block
        // axis out of ne0 BEFORE the causal-floor add: {B_k, n_blocks*idx_heads, n_tokens} against a
        // {B_k, n_blocks, n_tokens} floor is the same per-cell add (row h*n_blocks+b takes floor
        // block b via the % ne11 broadcast), but it hands the add n_blocks*idx_heads rows instead of
        // idx_heads -- at decode that is 4 working threads vs all of them. The sum is already laid
        // out as the {B_k, n_blocks, HT} pool input, so the ggml_cont the flat shape needed goes too.
        ggml_tensor * sc3 = ggml_reshape_3d(ctx0, scores, B_k, n_blocks*idx_heads, n_tokens);
        ggml_tensor * fl3 = ggml_reshape_3d(ctx0, floor3, B_k, n_blocks, n_tokens);
        blk = ggml_reshape_3d(ctx0, ggml_add(ctx0, sc3, fl3), B_k, n_blocks, HT);
    } else {
        // soft_max path only: pad n_kv up to n_blocks*B_k with -BIG (so an all-negative block can't
        // be beaten by zero-pad), reshape to {B_k, n_blocks * idx_heads*n_tokens}.
        scores = ggml_add(ctx0, scores, floor3);                              // broadcast over idx_heads
        ggml_tensor * sc2 = ggml_reshape_2d(ctx0, ggml_cont(ctx0, scores), n_kv_eff, HT);
        sc2 = ggml_pad(ctx0, sc2, n_pad, 0, 0, 0);                            // {n_blocks*B_k, HT}
        ggml_tensor * pos = ggml_arange(ctx0, 0.0f, (float) (n_blocks * B_k), 1.0f);
        ggml_tensor * tail = ggml_scale(ctx0,
                ggml_step(ctx0, ggml_add1(ctx0, pos, ggml_new_f32(ctx0, -((float) n_kv_eff) + 0.5f))),
                -BIG);                                                        // {n_blocks*B_k}
        tail = ggml_reshape_2d(ctx0, tail, n_blocks * B_k, 1);
        sc2 = ggml_add(ctx0, sc2, tail);                                      // broadcast over HT
        blk = ggml_reshape_3d(ctx0, sc2, B_k, n_blocks, HT);
    }
    blk = ggml_pool_1d(ctx0, blk, GGML_OP_POOL_MAX, B_k, B_k, 0);             // {1, n_blocks, HT}
    // Guard the ranking against a +inf that reached the pool from a live cell: clamp AFTER the
    // pool, where it is n_blocks values per (head, token) instead of n_kv, and through a one-row
    // view so the in-place kernel covers all of them (see the index-key clamp above).
    // A fully-masked block pools to -FLT_MAX, not -inf (POOL_MAX's init value is never beaten by
    // -inf, since `-inf > -FLT_MAX` is false); the clamp maps that to -BIG, which still ranks last.
    // Live-block ranking is untouched (real scores are far inside +/-BIG), so selection is
    // unchanged. NaN cannot reach here -- the pool drops it one node earlier -- which matters
    // because ggml_clamp maps NaN to +max and would rank such a block FIRST.
    blk = ggml_clamp(ctx0, ggml_reshape_2d(ctx0, blk, n_blocks*HT, 1), -BIG, BIG);
    blk = ggml_reshape_3d(ctx0, blk, n_blocks, idx_heads, n_tokens);          // {n_blocks, idx_heads, n_tokens}
    // Diagnostic tag (cb is a no-op unless an eval-callback consumes it): the pooled per-block
    // indexer scores, shaped {n_blocks, idx_heads, n_tokens}, BEFORE local-include / top-k.
    cb(blk, "msa_blk_scores", il);

    // --- local-block force-include (reference: block_scores.scatter_(q_block, +inf)) ---
    // For query at position p (absolute kv slot = n_past + p), the local block index is
    // (n_past + p) // B_k. local_blocks=1 in config, so just the single containing block.
    // We add +BIG to block (pos//B_k) for every (head, token). Build a {n_blocks, n_tokens} bump
    // via a one-hot over the local block id, broadcast over idx_heads.
    if (hparams.minimax_sparse_local_block) {
      ggml_tensor * bump = shared ? shared->bump : nullptr;
      if (!bump) {
        // local block id per token = (kv_head + p) / B_k, p in [0,n_tokens).  {n_tokens}
        ggml_tensor * p = ggml_arange(ctx0, (float) kv_head, (float) (kv_head + n_tokens), 1.0f); // abs slot
        // kv_head is baked into op_params here; a reused graph would keep the first ubatch's value.
        // Register so update_cache_copies() can re-point it. One tensor, because this subgraph is
        // now shared by every sparse layer.
        lctx.msa_local_arange = p;
        ggml_tensor * lblk = ggml_scale(ctx0, p, 1.0f / (float) B_k);          // (kv_head+p)/B_k (float)
        // floor via step-sum over block-id thresholds: onehot[b,t] = 1 iff floor(lblk[t])==b.
        // Build {n_blocks, n_tokens}: for block id b, indicator (lblk - b in [0,1)).
        // step(lblk - b) - step(lblk - (b+1)) == 1 exactly on the containing block.
        ggml_tensor * bids = ggml_arange(ctx0, 0.0f, (float) n_blocks, 1.0f);  // {n_blocks}
        bids = ggml_reshape_2d(ctx0, bids, n_blocks, 1);                       // {n_blocks,1}
        ggml_tensor * lt = ggml_reshape_2d(ctx0, lblk, 1, n_tokens);           // {1,n_tokens}
        // diff[b,t] = lblk[t] - b   (broadcast)  -> {n_blocks, n_tokens}
        ggml_tensor * diff = ggml_add(ctx0,
                ggml_repeat(ctx0, lt, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blocks, n_tokens)),
                ggml_scale(ctx0, ggml_repeat(ctx0, bids, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blocks, n_tokens)), -1.0f));
        // onehot = step(diff) - step(diff-1); add an epsilon so the exact integer boundary
        // (diff==0) is firmly >0 and diff==1 is firmly excluded.
        ggml_tensor * onehot = ggml_sub(ctx0,
                ggml_step(ctx0, ggml_add1(ctx0, diff, ggml_new_f32(ctx0,  1e-4f))),
                ggml_step(ctx0, ggml_add1(ctx0, diff, ggml_new_f32(ctx0, -1.0f + 1e-4f))));
        bump = ggml_scale(ctx0, onehot, BIG);                                  // {n_blocks, n_tokens}
        bump = ggml_reshape_3d(ctx0, bump, n_blocks, 1, n_tokens);             // broadcast over idx_heads
        if (shared) shared->bump = bump;
      }
      blk = ggml_add(ctx0, blk, bump);
    }

    // --- per-idx-head top-k block selection -> additive block penalty {n_blocks, idx_heads, n_tokens} ---
    // Independent top-k PER index head (the reference's [B, H_idx, S_q, n_blocks].topk(dim=-1)).
    // Reuse the validated GLM-DSA full-coverage scatter, folding (idx_head, token) into one axis:
    //   base/pen_b {1, n_blocks, HT}, idx {n_blocks, HT, 1}, scatter rank-penalty to sorted[rank].
    const auto & l_early = model.layers[il];
    const bool tp_attn_early = !l_early.wqkv && !l_early.wqk && cparams.flash_attn &&
            l_early.wq && l_early.wq->extra && l_early.wk && l_early.wk->extra &&
            l_early.wv && l_early.wv->extra && l_early.wo && l_early.wo->extra &&
            kv_self.k_l[il] && kv_self.k_l[il]->extra && kv_self.v_l[il] && kv_self.v_l[il]->extra;
    ggml_tensor * blk2 = ggml_reshape_2d(ctx0, ggml_cont(ctx0, blk), n_blocks, HT); // {n_blocks, HT}
    ggml_tensor * sorted = ggml_argsort(ctx0, blk2, GGML_SORT_ORDER_DESC);    // {n_blocks, HT} I32 (block ids)

    // pen[rank] = 0 for rank < topk_blk, else -BIG.  {n_blocks}
    ggml_tensor * rank = ggml_arange(ctx0, 0.0f, (float) n_blocks, 1.0f);     // {n_blocks} F32
    ggml_tensor * sel  = ggml_step(ctx0, ggml_scale_bias(ctx0, rank, -1.0f, (float) topk_blk - 0.5f));
    ggml_tensor * pen  = ggml_scale_bias(ctx0, sel, BIG, -BIG);               // 0 or -BIG, {n_blocks}

    // --- index-list mode (--msa-gather): stop here, no {n_kv, idx_heads} mask is ever built -----
    // `sorted` holds block ids in descending score order per (idx head, token) and the local-block
    // bump is already in the scores, so its first topk_blk rows ARE the selection. Everything below
    // this block exists to turn that into an n_kv-wide additive mask; with an index list on src[5]
    // the FA kernel reads the mask at only topk_blk*B_k positions, and the causal floor it needs is
    // already in KQ_mask, which the graph builds once and every layer shares.
    //
    // The mask chain this skips is not small at decode: ggml_flash_attn_ext requires
    // mask->ne[1] >= GGML_PAD(n_tokens, 16), so a single decode token still pads the mask to 16 rows
    // and then casts it, per sparse layer.
    if (cparams.msa_gather && cparams.msa_split_gqa && cparams.flash_attn && !tp_attn_early &&
            n_kv_eff % B_k == 0 && topk_blk*B_k < n_kv_eff &&
            hparams.n_head(il) % idx_heads == 0 && (int64_t) hparams.n_head_kv(il) == idx_heads) {
        const int64_t n_gather = topk_blk * B_k;
        // Cell-id table {B_k, n_blocks} I32. ggml has no I32 arithmetic and ggml_cast cannot make
        // I32 (ggml_compute_forward_dup handles F32/F16/BF16 sources only), so build it the one way
        // the tree already supports: mask_to_index over an all-zero mask returns [0 .. n_kv).
        // The table is 0..n_kv-1 and does not depend on il, so build it once per graph. Rebuilding
        // it per sparse layer costs 57 redundant O(n_kv) fills and scans, and ~200 extra graph
        // nodes per evaluation, each carrying its own parallel dispatch and barrier.
        ggml_tensor * table = shared ? shared->cell_table : nullptr;
        if (!table) {
            ggml_tensor * zero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_eff, 1), 0.0f);
            table = ggml_mask_to_index(ctx0, zero, (int) n_kv_eff);                 // {n_kv,1} I32
            if (shared) shared->cell_table = table;
        }
        table = ggml_reshape_2d(ctx0, table, B_k, n_blocks);                        // {B_k, n_blocks}
        // Selected block ids, contiguous: {topk_blk, HT} -> flat. ggml_top_k sets the argsort's
        // count so the CPU kernel std::partial_sorts topk_blk of n_blocks instead of full-sorting;
        // the full-sort `sorted` above is not an ancestor of `cells`, so it drops out of the graph.
        ggml_tensor * sel_blk = ggml_cont(ctx0, ggml_top_k(ctx0, blk2, (int) topk_blk)); // {topk_blk, HT} I32
        ggml_tensor * flat = ggml_reshape_1d(ctx0, sel_blk, topk_blk*HT);
        ggml_tensor * cells = ggml_get_rows_ext(ctx0, table, flat, true, false);    // {B_k, topk_blk*HT} I32
        cells = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cells), n_gather, HT);        // {n_gather, HT}
        ggml_build_forward_expand(gf, cells);
        msa->n_groups = (int) idx_heads;
        msa->idx.clear();
        for (int64_t g = 0; g < idx_heads; ++g) {
            // HT folds (idx_head, token) with idx_head fastest, so group g is a strided view.
            msa->idx.push_back(ggml_view_2d(ctx0, cells, n_gather, n_tokens,
                    cells->nb[1]*idx_heads, g*cells->nb[1]));
        }
        cb(cells, "minimax_msa_cells", il);
        return KQ_mask;                                    // causal floor only, shared by all layers
    }

    pen = ggml_reshape_3d(ctx0, pen, 1, n_blocks, 1);
    ggml_tensor * pen_b = ggml_repeat(ctx0, pen,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_blocks, HT));        // {1, n_blocks, HT}
    ggml_tensor * base = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_blocks, HT), -BIG);
    ggml_tensor * idx = ggml_reshape_3d(ctx0, sorted, n_blocks, HT, 1);       // {n_blocks, HT, 1}
    ggml_tensor * blk_mask = ggml_set_rows(ctx0, base, pen_b, idx);           // {1, n_blocks, HT}

    // --- expand block penalty back to per-token {n_kv, idx_heads, n_tokens}: repeat each block over B_k ---
    // blk_mask is {1, n_blocks, HT}. Repeat ne[0] 1->B_k so kv slot p=blk*B_k+b gets block blk's
    // penalty, then crop to n_kv. Result laid out as {n_kv, idx_heads, n_tokens}.
    ggml_tensor * tok_mask = ggml_repeat(ctx0, blk_mask,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, B_k, n_blocks, HT));      // {B_k, n_blocks, HT}
    tok_mask = ggml_reshape_2d(ctx0, ggml_cont(ctx0, tok_mask), n_blocks * B_k, HT);
    ggml_tensor * sparse_ht = ggml_cont(ctx0,
            ggml_view_2d(ctx0, tok_mask, n_kv_eff, HT, tok_mask->nb[1], 0));  // {n_kv, idx_heads*n_tokens}
    sparse_ht = ggml_reshape_3d(ctx0, sparse_ht, n_kv_eff, idx_heads, n_tokens); // {n_kv, idx_heads, n_tokens}
    // re-add causal floor so masked-future positions stay -inf even if their block is selected.
    sparse_ht = ggml_add(ctx0, sparse_ht, floor3);                            // broadcast over idx_heads

    // --- broadcast each GQA group's selection to its query heads: idx_heads(4) -> n_head(64) ---
    // The reference build_block_mask does repeat_interleave(num_attention_heads // n_idx_heads, dim=1).
    // soft_max consumes a mask shaped {n_kv, n_tokens, n_head} (it indexes the mask's ne[2] by the
    // kq head: i12 = i02 % ne12). We need head index h -> idx-group g = h / group_sz, i.e. each
    // group's plane repeated group_sz times CONTIGUOUSLY along the head axis (repeat_interleave),
    // NOT tiled. Lay out as {n_kv, n_tokens, idx_heads} then expand head axis by group_sz.
    const int64_t n_head    = hparams.n_head(il);
    const int64_t group_sz  = n_head / idx_heads;             // 16
    // sparse_ht {n_kv, idx_heads, n_tokens} -> permute to {n_kv, n_tokens, idx_heads}
    ggml_tensor * perm = ggml_cont(ctx0, ggml_permute(ctx0, sparse_ht, 0, 2, 1, 3)); // {n_kv, n_tokens, idx_heads}

    // Default path: stop at idx_heads planes and let llm_build_kqv run one attention call per GQA
    // group. The expansion below is group_sz-fold (16x for M3) and exists only so the kernels'
    // head-plane rule lands on the right plane; it also costs the fast iqk FA kernel, which is
    // skipped whenever mask->ne[2] > 1. --no-msa-split-gqa asks for that wide mask back.
    const auto & l = model.layers[il];
    const bool tp_attn = !l.wqkv && !l.wqk && cparams.flash_attn &&
            l.wq && l.wq->extra && l.wk && l.wk->extra && l.wv && l.wv->extra && l.wo && l.wo->extra &&
            kv_self.k_l[il] && kv_self.k_l[il]->extra && kv_self.v_l[il] && kv_self.v_l[il]->extra;
    // MSA has never run under tensor-parallel attention: that path issues one flash-attention call
    // per device over that device's head slice, and indexes the mask by the DEVICE-LOCAL head, so
    // neither an n_head-wide mask nor a per-group one is read correctly. Before this commit it died
    // inside ggml_flash_attn_ext on `q->ne[2] % mask->ne[2] == 0` (16 % 64). Say so instead.
    if (msa && tp_attn) {
        LLAMA_LOG_ERROR("%s: --msa is not supported with tensor-parallel attention (-sm graph); "
                        "run without --msa, or with -sm layer\n", __func__);
        GGML_ABORT("minimax-m3: --msa with tensor-parallel attention");
    }
    if (msa && cparams.msa_split_gqa && cparams.flash_attn && !tp_attn &&
            hparams.n_head(il) % idx_heads == 0 && (int64_t) hparams.n_head_kv(il) == idx_heads) {
        ggml_tensor * split = perm;
        if (KQ_mask->ne[1] > n_tokens) {
            split = ggml_pad(ctx0, split, 0, KQ_mask->ne[1] - n_tokens, 0, 0);
        }
        if (split->type != GGML_TYPE_F16) {
            split = ggml_cast(ctx0, split, GGML_TYPE_F16);
        }
        msa->n_groups = (int) idx_heads;
        if (cparams.msa_gather) {
            const int64_t n_gather = topk_blk * B_k;
            // The selection names exactly topk_blk*B_k cells per query. Hand that count down and
            // llm_build_kqv attaches the index list the FA kernels already know how to consume.
            msa->n_gather = (int) n_gather;
        }
        cb(split, "minimax_msa_mask", il);
        return split;                                  // {n_kv, n_tok_pad, idx_heads} F16
    }
    // repeat_interleave over the head axis: insert a group_sz axis right after idx_heads-as-ne2,
    // i.e. {n_kv, n_tokens, group_sz, idx_heads}? We need final head order h = g*group_sz + r with
    // group g varying slowest. So expand to {n_kv, n_tokens, group_sz*idx_heads} where the fastest
    // (innermost contiguous along head axis) index is r within group g. Build by repeating each
    // group plane group_sz times contiguously: reshape ne2 idx_heads -> {1, idx_heads} then repeat
    // ne0(the new inner)1->group_sz over a 4D temp, then fold.
    ggml_tensor * m4 = ggml_reshape_4d(ctx0, perm, n_kv_eff, n_tokens, 1, idx_heads); // {n_kv,n_tok,1,idx_heads}
    m4 = ggml_repeat(ctx0, m4,
            ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_eff, n_tokens, group_sz, idx_heads)); // {n_kv,n_tok,group_sz,idx_heads}
    // fold (group_sz, idx_heads) -> n_head with idx_heads slowest, group_sz fastest => head = g*group_sz + r. good.
    ggml_tensor * sparse = ggml_reshape_3d(ctx0, ggml_cont(ctx0, m4), n_kv_eff, n_tokens, n_head); // {n_kv, n_tokens, n_head}

    // pad token axis (ne[1]) back to KQ_mask token layout (GGML_KQ_MASK_PAD) so soft_max's
    // mask->ne[1] >= a->ne[1] holds. (Pad along ne1; head axis ne2 stays n_head.)
    const int64_t n_tok_pad = KQ_mask->ne[1];
    if (n_tok_pad > n_tokens) {
        sparse = ggml_pad(ctx0, sparse, 0, n_tok_pad - n_tokens, 0, 0);
    }
    // Flash-attn requires an F16 mask. Cast the per-head F32 mask to F16; the BIG=1e30 drop value
    // saturates to +inf in F16 which the FA kernels treat as a hard mask (exp(-inf)=0), so the
    // sparse drop is preserved. The CUDA/CPU soft_max path keeps the F32 mask. Shape is
    // {n_kv, n_tok_pad, n_head} in both cases — the kernels index the head plane by i02 % ne[2].
    if (cparams.flash_attn && sparse->type != GGML_TYPE_F16) {
        sparse = ggml_cast(ctx0, sparse, GGML_TYPE_F16);
    }
    cb(sparse, "minimax_msa_mask", il);
    return sparse;
}

// Partial NEOX RoPE for the indexer q/k: rotate the first n_rot dims of each d_idx-wide
// index head, pass dims [n_rot, d_idx) through unrotated, then concat back. Matches the
// reference apply_rotary_pos_emb(idx, cos[..,:head_dim], sin[..,:head_dim]) where the cos/sin
// width is rotary_dim = n_rot. Input/output {d_idx, n_idx_heads, n_tokens}.
ggml_tensor * llm_build_context::build_minimaxm3_index_rope(ggml_tensor * v, ggml_tensor * inp_pos,
        int64_t n_rot_idx, int64_t d_idx, int64_t n_idx_heads, float idx_freq_base, float idx_freq_scale, int il) {
    if (n_rot_idx <= 0 || n_rot_idx >= d_idx) {
        // full-width (or no) rotation
        return ggml_rope_ext(ctx0, ggml_cont(ctx0, v), inp_pos, nullptr, (int) (n_rot_idx > 0 ? n_rot_idx : d_idx),
                LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, idx_freq_base, idx_freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }
    const int64_t n_pass = d_idx - n_rot_idx;
    const int64_t n_tok  = v->ne[2];
    // split [0:n_rot) rope / [n_rot:d_idx) pass, per head (dim0)
    ggml_tensor * pe = ggml_view_3d(ctx0, v, n_rot_idx, n_idx_heads, n_tok,
            v->nb[1], v->nb[2], 0);
    ggml_tensor * pass = ggml_view_3d(ctx0, v, n_pass, n_idx_heads, n_tok,
            v->nb[1], v->nb[2], ggml_row_size(v->type, n_rot_idx));
    pe = ggml_rope_ext(ctx0, ggml_cont(ctx0, pe), inp_pos, nullptr, (int) n_rot_idx,
            LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, idx_freq_base, idx_freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    return ggml_concat(ctx0, pe, ggml_cont(ctx0, pass), 0);                   // {d_idx, n_idx_heads, n_tok}
}

ggml_cgraph* llm_build_context::build_minimaxm3() {
    ggml_cgraph * gf = new_graph_custom();
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // Layer-independent subgraphs, shared by every sparse layer; see build_minimaxm3_msa_mask.
    msa_shared msa_sh;

    // Clear the registered local-block arange before rebuilding. Registration happens during the
    // build below; if a rebuild ever produced a graph without that subgraph, a pointer into the
    // previous graph would otherwise survive and be patched on the next reuse check.
    lctx.msa_local_arange = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        // MiniMax-M3 MSA: build the block-sparse mask for this layer (nullptr => dense).
        msa_attn_split msa;
        // The indexer norms inpL with attn_norm; on a sparse layer the attention would then build
        // a bit-identical second RMSNorm node (57 of them per token, ~0.9 ms). Take the indexer's
        // and pass it as pre_normed. inpL is still handed over as the input, because that is what
        // the add_input residual adds -- passing x_normed there instead would silently change the
        // residual to attn_out + RMSNorm(inpL). On a dense layer x_normed stays null and nothing
        // about this call changes.
        ggml_tensor * x_normed = nullptr;
        ggml_tensor * msa_mask  = build_minimaxm3_msa_mask(gf, inpL, inp_pos, KQ_mask, il, &msa,
                                                           &msa_sh, &x_normed);
        ggml_tensor * attn_mask = msa_mask ? msa_mask : KQ_mask;
        ggml_tensor * ffn_inp = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                attn_mask, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.0f, 0,
                il, true, false, true, false, false, nullptr, -1, 0.0f, nullptr, nullptr, nullptr, &msa, x_normed);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SWIGLU_OAI, LLM_FFN_PAR, cb, il, gf, true);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    nullptr,
                    model.layers[il].ffn_gate_exps,
                    nullptr,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,
                    nullptr,
                    model.layers[il].ffn_gate_shexp,
                    nullptr,
                    model.layers[il].ffn_down_shexp,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SWIGLU_OAI,
                    cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
