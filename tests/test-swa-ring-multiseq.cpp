
#include "llama.h"
#include "llama-model.h"
#include "get-model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fails = 0;

static void check(bool ok, const char * what) {
    printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        ++g_fails;
    }
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return INFINITY;
    }
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

static llama_context * make_ctx(llama_model * model, bool ring, uint32_t n_ctx, uint32_t n_seq_max,
        bool flash_attn = true) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx;
    cparams.n_batch      = n_ctx;
    cparams.n_ubatch     = 48;
    cparams.n_seq_max    = n_seq_max;
    cparams.n_threads    = 2;
    cparams.n_threads_batch = 2;
    cparams.swa_compress = ring;
    cparams.flash_attn   = flash_attn;
    return llama_init_from_model(model, cparams);
}

static bool decode_seq(llama_context * ctx, const llama_token * tokens, int32_t n, llama_pos pos0,
        llama_seq_id seq_id) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static std::vector<float> logits_of(llama_context * ctx, int32_t idx, int32_t n_vocab) {
    const float * l = llama_get_logits_ith(ctx, idx);
    return l ? std::vector<float>(l, l + n_vocab) : std::vector<float>();
}

static std::vector<float> decode_one(llama_context * ctx, llama_token tok, llama_pos pos,
        llama_seq_id seq_id, int32_t n_vocab) {
    if (!decode_seq(ctx, &tok, 1, pos, seq_id)) {
        return {};
    }
    return logits_of(ctx, 0, n_vocab);
}

static bool decode_pair(llama_context * ctx, llama_token t0, llama_pos p0, llama_token t1, llama_pos p1,
        std::vector<float> & out0, std::vector<float> & out1, int32_t n_vocab) {
    llama_batch batch = llama_batch_init(2, 0, 1);
    batch.n_tokens = 2;
    const llama_token toks[2] = { t0, t1 };
    const llama_pos   poss[2] = { p0, p1 };
    for (int32_t i = 0; i < 2; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = poss[i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = i;
        batch.logits[i]    = true;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return false;
    }
    out0 = logits_of(ctx, 0, n_vocab);
    out1 = logits_of(ctx, 1, n_vocab);
    return !out0.empty() && !out1.empty();
}

int main(int argc, char ** argv) {
    const char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    if (const char * ngl = getenv("LLAMACPP_TEST_NGL")) {
        mparams.n_gpu_layers = atoi(ngl);
        printf("LLAMACPP_TEST_NGL: offloading %d layers\n", mparams.n_gpu_layers);
    }
    mparams.swa_compress = true;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model '%s'\n", model_path);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const auto & hparams = model->hparams;
    bool has_swa_layer = false, has_full_layer = false;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        has_swa_layer  = has_swa_layer  ||  hparams.swa_layers[il];
        has_full_layer = has_full_layer || !hparams.swa_layers[il];
    }
    if (hparams.n_swa == 0 || !has_swa_layer || !has_full_layer) {
        fprintf(stderr, "model needs n_swa > 0 and both SWA and full-attention layers to test the ring\n");
        llama_free_model(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const int32_t  n_vocab   = llama_n_vocab(model);
    const uint32_t n_seq     = 2;
    const uint32_t win       = hparams.n_swa + 48;
    const uint32_t n_ctx     = 8 * win + 1024;
    const int32_t  n_prompt  = (int32_t) hparams.n_swa / 2 + 7;   // shorter than one window
    const int32_t  n_skew    = (int32_t) win + 64;                // > one full window past seq 0

    std::vector<llama_token> p0(n_prompt), p1(n_prompt), skew(n_skew);
    for (int32_t i = 0; i < n_prompt; ++i) {
        p0[i] = (llama_token) ((i * 7 + 3) % n_vocab);
        p1[i] = (llama_token) ((i * 11 + 5) % n_vocab);
    }
    for (int32_t i = 0; i < n_skew; ++i) {
        skew[i] = (llama_token) ((i * 13 + 2) % n_vocab);
    }
    const llama_token probe = (llama_token) (23 % n_vocab);

    llama_context * ring  = make_ctx(model, true,  n_ctx, n_seq);
    llama_context * dense = make_ctx(model, false, n_ctx, n_seq);
    check(ring != nullptr && dense != nullptr, "ring and dense multi-sequence contexts created");
    check(ring && llama_kv_self_is_swa_ring(ring),
          "ring KV cache engaged with n_seq_max > 1 (test is not vacuous)");
    check(dense && !llama_kv_self_is_swa_ring(dense), "dense reference context has no ring");
    if (ring == nullptr || dense == nullptr || !llama_kv_self_is_swa_ring(ring)) {
        if (ring)  llama_free(ring);
        if (dense) llama_free(dense);
        llama_free_model(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    std::vector<float> l_ring, l_dense;
    for (llama_context * ctx : { ring, dense }) {
        bool ok = decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        ok = ok && decode_seq(ctx, p1.data(), n_prompt, 0, 1);
        ok = ok && decode_seq(ctx, skew.data(), n_skew, n_prompt, 1);
        check(ok, ctx == ring ? "ring decoded both sequences with skew"
                              : "dense decoded both sequences with skew");
        std::vector<float> l = ok ? decode_one(ctx, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        check(!l.empty(), ctx == ring ? "ring continued the parked sequence"
                                      : "dense continued the parked sequence");
        (ctx == ring ? l_ring : l_dense) = l;
    }
    {
        const float d = max_abs_diff(l_ring, l_dense);
        check(d <= 2e-3f, "parked sequence's window survived a full-window skew by the other sequence");
        printf("     max |logit diff| ring vs dense after skew: %g\n", d);
    }

    {
        bool ok = true;
        std::vector<float> r0, r1, d0, d1;
        for (int step = 0; step < 6 && ok; ++step) {
            const llama_token t0 = (llama_token) ((step * 3 + 1) % n_vocab);
            const llama_token t1 = (llama_token) ((step * 5 + 2) % n_vocab);
            const llama_pos   q0 = n_prompt + 1 + 2*step;
            const llama_pos   q1 = n_prompt + n_skew + step;
            ok = decode_pair(ring,  t0, q0, t1, q1, r0, r1, n_vocab) &&
                 decode_pair(dense, t0, q0, t1, q1, d0, d1, n_vocab);
            ok = ok && decode_seq(ring,  &t0, 1, q0 + 1, 0) && decode_seq(dense, &t0, 1, q0 + 1, 0);
        }
        check(ok, "mixed-sequence ubatches decode on both contexts");
        const float d_a = max_abs_diff(r0, d0);
        const float d_b = max_abs_diff(r1, d1);
        check(std::max(d_a, d_b) <= 2e-3f, "mixed-ubatch writes land in the right stripe for both sequences");
        printf("     max |logit diff| in mixed ubatches: seq0 %g, seq1 %g\n", d_a, d_b);
    }

    {
        llama_context * src = make_ctx(model, true, n_ctx, n_seq);
        llama_context * dst = make_ctx(model, true, n_ctx, n_seq);
        bool ok = src && dst;
        ok = ok && decode_seq(src, p0.data(), n_prompt, 0, 0);

        size_t blob_n = 0;
        std::vector<uint8_t> blob;
        if (ok) {
            blob.resize(llama_state_seq_get_size(src, 0, 0));
            blob_n = blob.empty() ? 0 : llama_state_seq_get_data(src, blob.data(), blob.size(), 0, 0);
        }
        std::vector<float> l_ref = ok ? decode_one(src, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        check(blob_n > 0 && blob_n == blob.size(), "sequence 0 state serializes on a striped ring");

        const size_t nread = (ok && blob_n) ? llama_state_seq_set_data(dst, blob.data(), blob_n, 1, 0) : 0;
        check(nread == blob_n && blob_n > 0, "sequence 0's blob restores into sequence 1's stripe");

        std::vector<float> l_dst = (nread == blob_n && blob_n)
                ? decode_one(dst, probe, n_prompt, 1, n_vocab) : std::vector<float>();
        const float d = max_abs_diff(l_ref, l_dst);
        check(d <= 2e-3f, "restored-into-another-stripe state continues with the same logits");
        printf("     max |logit diff| after cross-stripe restore: %g\n", d);

        if (src) llama_free(src);
        if (dst) llama_free(dst);
    }

    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        bool ok = ctx && decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        }
        check(ok && llama_kv_cache_seq_pos_min(ctx, 1) >= 0,
              "full-range seq_cp populated the destination stripe under the ring");
        std::vector<float> l_a = ok ? decode_one(ctx, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        std::vector<float> l_b = ok ? decode_one(ctx, probe, n_prompt, 1, n_vocab) : std::vector<float>();
        const float d_cp = max_abs_diff(l_a, l_b);
        check(!l_a.empty() && !l_b.empty() && d_cp <= 2e-3f,
              "a cloned sequence continues exactly like its source");
        printf("     max |logit diff| after full-range seq_cp: %g\n", d_cp);
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        bool ok = ctx && decode_seq(ctx, skew.data(), n_skew, 0, 0);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        }
        std::vector<float> l_a = ok ? decode_one(ctx, probe, n_skew, 0, n_vocab) : std::vector<float>();
        std::vector<float> l_b = ok ? decode_one(ctx, probe, n_skew, 1, n_vocab) : std::vector<float>();
        const float d_wrap = max_abs_diff(l_a, l_b);
        check(!l_a.empty() && !l_b.empty() && d_wrap <= 2e-3f,
              "a clone of a WRAPPED window continues exactly like its source");
        printf("     max |logit diff| after cloning a wrapped window: %g\n", d_wrap);
        bool cont = ok;
        for (int32_t i = 0; cont && i < (int32_t) win + 8; ++i) {
            cont = decode_seq(ctx, &skew[i % n_skew], 1, n_skew + 1 + i, 1);
        }
        check(cont, "a cloned sequence keeps decoding past a full window without an occupancy abort");
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        bool ok = ctx && decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        ok = ok && decode_seq(ctx, p1.data(), n_prompt, 0, 1);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        }
        std::vector<float> l_a = ok ? decode_one(ctx, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        std::vector<float> l_b = ok ? decode_one(ctx, probe, n_prompt, 1, n_vocab) : std::vector<float>();
        const float d_clob = max_abs_diff(l_a, l_b);
        check(!l_a.empty() && !l_b.empty() && d_clob <= 2e-3f,
              "seq_cp replaced the destination's own cells with the source's");
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        if (ctx) {
            llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        }
        check(ctx && llama_kv_cache_seq_pos_min(ctx, 1) == -1,
              "cloning an empty sequence leaves the destination empty");
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        bool ok = ctx && decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, 1, 2, n_prompt - 2);
        }
        check(ok && llama_kv_cache_seq_pos_min(ctx, 1) == -1,
              "a partial-range seq_cp is refused and leaves the destination untouched");
        if (ok) {
            check(llama_kv_cache_seq_pos_min(ctx, 0) >= 0,
                  "a refused partial-range seq_cp left the source intact");
        }
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, true, n_ctx, n_seq);
        bool ok = ctx && decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, (llama_seq_id) n_seq, -1, -1);
            llama_kv_cache_seq_cp(ctx, (llama_seq_id) n_seq, 1, -1, -1);
        }
        check(ok && llama_kv_cache_seq_pos_min(ctx, 0) >= 0,
              "an out-of-range seq_cp is refused without disturbing the ring");
        std::vector<float> l = ok ? decode_one(ctx, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        check(!l.empty(), "the ring still decodes after a refused out-of-range seq_cp");
        if (ctx) llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(model, false, n_ctx, n_seq);
        bool ok = ctx && !llama_kv_self_is_swa_ring(ctx);
        ok = ok && decode_seq(ctx, p0.data(), n_prompt, 0, 0);
        if (ok) {
            llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);
        }
        std::vector<float> l_a = ok ? decode_one(ctx, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        std::vector<float> l_b = ok ? decode_one(ctx, probe, n_prompt, 1, n_vocab) : std::vector<float>();
        check(!l_a.empty() && !l_b.empty() && max_abs_diff(l_a, l_b) <= 2e-3f,
              "dense seq_cp still shares a sequence to another slot");
        if (ctx) llama_free(ctx);
    }

    {
        std::vector<float> r0, r1, d0, d1;
        bool ok = true;
        const llama_pos base0 = n_prompt + 2 + 2*6;   // past section (2)'s positions
        const llama_pos base1 = n_prompt + n_skew + 6;
        for (int step = 0; step < 8 && ok; ++step) {
            const llama_token t0 = (llama_token) ((step * 9 + 4) % n_vocab);
            const llama_token t1 = (llama_token) ((step * 7 + 6) % n_vocab);
            ok = decode_pair(ring,  t0, base0 + step, t1, base1 + step, r0, r1, n_vocab) &&
                 decode_pair(dense, t0, base0 + step, t1, base1 + step, d0, d1, n_vocab);
        }
        const float d = std::max(max_abs_diff(r0, d0), max_abs_diff(r1, d1));
        check(ok && d <= 2e-3f, "a repeated mixed-ubatch structure stays correct across graph reuse");
        printf("     max |logit diff| in the steady-state mixed decode: %g\n", d);
    }

    {
        const size_t sz = llama_state_seq_get_size(ring, 1, 0);
        std::vector<uint8_t> blob(sz ? sz : 1);
        const size_t written = sz ? llama_state_seq_get_data(ring, blob.data(), blob.size(), 1, 0) : 0;
        check(sz > 0 && written == sz, "a fragmented-cell-layout sequence serializes");

        llama_context * dst = make_ctx(model, true, n_ctx, n_seq);
        const size_t nread = (dst && written) ? llama_state_seq_set_data(dst, blob.data(), written, 1, 0) : 0;
        check(nread == written && written > 0, "that blob restores into a fresh context");

        const llama_pos next = n_prompt + n_skew + 14;
        std::vector<float> l_ref  = decode_one(ring, probe, next, 1, n_vocab);
        decode_one(dense, probe, next, 1, n_vocab);   // keep the dense reference in lockstep
        std::vector<float> l_test = (nread == written && written)
                ? decode_one(dst, probe, next, 1, n_vocab) : std::vector<float>();
        const float d = max_abs_diff(l_ref, l_test);
        check(d <= 2e-3f, "restored fragmented layout continues with the same logits");
        printf("     max |logit diff| after fragmented-layout restore: %g\n", d);
        if (dst) llama_free(dst);
    }

    {
        llama_context * one = make_ctx(model, true, n_ctx, 1);
        llama_context * two = make_ctx(model, true, n_ctx, n_seq);
        bool ok = one && two && llama_kv_self_is_swa_ring(one) && llama_kv_self_is_swa_ring(two);
        check(ok, "single-slot and multi-slot ring contexts both engaged");
        ok = ok && decode_seq(one, p0.data(), n_prompt, 0, 0);

        std::vector<uint8_t> blob;   // saved before the reference continuation, as above
        size_t blob_n = 0;
        if (ok) {
            blob.resize(llama_state_seq_get_size(one, 0, 0));
            blob_n = blob.empty() ? 0 : llama_state_seq_get_data(one, blob.data(), blob.size(), 0, 0);
        }
        std::vector<float> l_ref = ok ? decode_one(one, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        const size_t nread = blob_n ? llama_state_seq_set_data(two, blob.data(), blob_n, 1, 0) : 0;
        check(blob_n > 0 && nread == blob_n, "a single-slot blob restores into a multi-slot ring");
        std::vector<float> l_two = (nread == blob_n && blob_n)
                ? decode_one(two, probe, n_prompt, 1, n_vocab) : std::vector<float>();
        const float d = max_abs_diff(l_ref, l_two);
        check(d <= 2e-3f, "cross---parallel restore continues with the same logits");
        printf("     max |logit diff| across --parallel values: %g\n", d);
        if (one) llama_free(one);
        if (two) llama_free(two);
    }

    {
        std::vector<float> l_keep_before = decode_one(ring,  probe, n_prompt + n_skew + 16, 1, n_vocab);
        std::vector<float> d_keep_before = decode_one(dense, probe, n_prompt + n_skew + 16, 1, n_vocab);
        check(!l_keep_before.empty() && max_abs_diff(l_keep_before, d_keep_before) <= 2e-3f,
              "surviving sequence matches dense before the other is removed");

        for (llama_context * ctx : { ring, dense }) {
            check(llama_kv_cache_seq_rm(ctx, 0, -1, -1), "whole-sequence removal accepted");
        }
        bool ok = decode_seq(ring, p1.data(), n_prompt, 0, 0) && decode_seq(dense, p1.data(), n_prompt, 0, 0);
        std::vector<float> l_new = ok ? decode_one(ring,  probe, n_prompt, 0, n_vocab) : std::vector<float>();
        std::vector<float> d_new = ok ? decode_one(dense, probe, n_prompt, 0, n_vocab) : std::vector<float>();
        check(ok && !l_new.empty() && max_abs_diff(l_new, d_new) <= 2e-3f,
              "reused slot decodes correctly after the previous sequence was removed");

        std::vector<float> l_keep = decode_one(ring,  probe, n_prompt + n_skew + 17, 1, n_vocab);
        std::vector<float> d_keep = decode_one(dense, probe, n_prompt + n_skew + 17, 1, n_vocab);
        const float d = max_abs_diff(l_keep, d_keep);
        check(d <= 2e-3f, "surviving sequence unaffected by the other's removal and reuse");
        printf("     max |logit diff| for the surviving sequence: %g\n", d);
    }

    {
        llama_kv_cache_seq_keep(ring,  0);
        check(llama_kv_cache_seq_pos_min(ring, 1) != -1,
              "seq_keep is refused under the ring: the other sequence is untouched");

        llama_kv_cache_seq_keep(dense, 0);
        check(llama_kv_cache_seq_pos_min(dense, 1) == -1,
              "dense seq_keep really does drop the other sequence (the test is not vacuous)");

        std::vector<float> l_after = decode_one(ring, probe, n_prompt + n_skew + 18, 1, n_vocab);
        check(!l_after.empty(), "the ring keeps decoding the sequence seq_keep did not remove");
    }

    {
        llama_context * rv = make_ctx(model, true,  n_ctx, n_seq, /* flash_attn */ false);
        llama_context * dv = make_ctx(model, false, n_ctx, n_seq, /* flash_attn */ false);
        bool ok = rv && dv && llama_kv_self_is_swa_ring(rv);
        check(ok, "transposed-V ring context engaged");
        std::vector<float> r0, r1, d0, d1;
        for (llama_context * ctx : { rv, dv }) {
            if (!ok) break;
            ok = decode_seq(ctx, p0.data(), n_prompt, 0, 0) && decode_seq(ctx, p1.data(), n_prompt, 0, 1);
            ok = ok && decode_seq(ctx, skew.data(), n_skew, n_prompt, 1);
        }
        for (int step = 0; step < 4 && ok; ++step) {
            const llama_token t0 = (llama_token) ((step * 3 + 1) % n_vocab);
            const llama_token t1 = (llama_token) ((step * 5 + 2) % n_vocab);
            ok = decode_pair(rv, t0, n_prompt + step, t1, n_prompt + n_skew + step, r0, r1, n_vocab) &&
                 decode_pair(dv, t0, n_prompt + step, t1, n_prompt + n_skew + step, d0, d1, n_vocab);
        }
        const float d = std::max(max_abs_diff(r0, d0), max_abs_diff(r1, d1));
        check(ok && d <= 2e-3f, "transposed-V mixed ubatches match dense for both sequences");
        printf("     max |logit diff| transposed V, mixed ubatch: %g\n", d);
        if (rv) llama_free(rv);
        if (dv) llama_free(dv);
    }

    {
        const size_t sz = llama_state_get_size(ring);
        std::vector<uint8_t> blob(sz ? sz : 1);
        const size_t written = sz ? llama_state_get_data(ring, blob.data(), blob.size()) : 0;
        check(written == 0, "whole-context save is refused while two sequences share a striped ring");
    }

    llama_free(ring);
    llama_free(dense);
    llama_free_model(model);
    llama_backend_free();

    if (g_fails != 0) {
        printf("SWA ring multi-sequence: %d check(s) failed\n", g_fails);
        return EXIT_FAILURE;
    }
    printf("SWA ring multi-sequence OK\n");
    return EXIT_SUCCESS;
}
