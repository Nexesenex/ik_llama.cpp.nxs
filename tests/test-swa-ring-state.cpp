
#include "llama.h"
#include "llama-model.h"
#include "get-model.h"
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

struct ctx_cfg {
    bool     ring;
    uint32_t n_ctx;
    uint32_t n_ubatch;
    bool     flash_attn;
};

static llama_context * make_ctx(llama_model * model, const ctx_cfg & cfg) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = cfg.n_ctx;
    cparams.n_batch      = cfg.n_ctx;
    cparams.n_ubatch     = cfg.n_ubatch;
    cparams.n_seq_max    = 1;        // the ring requires a single sequence
    cparams.n_threads    = 2;
    cparams.n_threads_batch = 2;
    cparams.swa_compress = cfg.ring;
    cparams.flash_attn   = cfg.flash_attn;
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const llama_token * tokens, int32_t n, llama_pos pos0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = pos0 + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static std::vector<float> decode_one(llama_context * ctx, llama_token tok, llama_pos pos, int32_t n_vocab) {
    std::vector<float> out;
    if (!decode_tokens(ctx, &tok, 1, pos)) {
        return out;
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (logits == nullptr) {
        return out;
    }
    out.assign(logits, logits + n_vocab);
    return out;
}

static std::vector<uint8_t> seq_save(llama_context * ctx) {
    const size_t size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> buf(size);
    if (size == 0) {
        return buf;
    }
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0, 0);
    buf.resize(n);   // a short/failed write shrinks the blob and fails the caller's size check
    return buf;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b);

static void check_round_trip(llama_model * model, const ctx_cfg & cfg, int32_t n_prompt,
        int32_t n_vocab, int32_t continue_n, const char * label) {
    std::vector<llama_token> tokens(n_prompt);
    for (int32_t i = 0; i < n_prompt; ++i) {
        tokens[i] = (llama_token) ((i * 5 + 1) % n_vocab);
    }
    const llama_token next_tok = (llama_token) (17 % n_vocab);

    llama_context * ctx_src = make_ctx(model, cfg);
    llama_context * ctx_dst = make_ctx(model, cfg);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        check(false, label);
        if (ctx_src) llama_free(ctx_src);
        if (ctx_dst) llama_free(ctx_dst);
        return;
    }

    bool ok = !cfg.ring || llama_kv_self_is_swa_ring(ctx_src);   // never assert on a dense cache by accident
    ok = ok && decode_tokens(ctx_src, tokens.data(), n_prompt, 0);
    const std::vector<uint8_t> blob = seq_save(ctx_src);
    ok = ok && !blob.empty();
    ok = ok && llama_state_seq_set_data(ctx_dst, blob.data(), blob.size(), 0, 0) == blob.size();

    const std::vector<uint8_t> blob_again = seq_save(ctx_dst);
    ok = ok && blob_again.size() == blob.size() &&
               memcmp(blob_again.data(), blob.data(), blob.size()) == 0;

    std::vector<float> l_ref, l_test;
    const int32_t n_steps = std::max(1, continue_n);
    for (int32_t i = 0; i < n_steps; ++i) {
        const llama_token tok = i == 0 ? next_tok : (llama_token) ((i * 3 + 19) % n_vocab);
        l_ref  = decode_one(ctx_src, tok, n_prompt + i, n_vocab);
        l_test = decode_one(ctx_dst, tok, n_prompt + i, n_vocab);
    }
    const float d = max_abs_diff(l_ref, l_test);
    ok = ok && d <= 2e-3f;

    check(ok, label);
    printf("     %s: %d cells, +%d decoded, blob %zu bytes, max |logit diff| %g\n",
            label, n_prompt, n_steps, blob.size(), d);

    llama_free(ctx_src);
    llama_free(ctx_dst);
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

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;
    if (const char * ngl = getenv("LLAMACPP_TEST_NGL")) {
        mparams.n_gpu_layers = atoi(ngl);
        printf("LLAMACPP_TEST_NGL: offloading %d layers\n", mparams.n_gpu_layers);
    }
    auto * model = (llama_model *) llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model at '%s'\n", model_path);
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

    const int32_t n_vocab = llama_n_vocab(model);

    const ctx_cfg cfg_ring  = { true,  4 * (hparams.n_swa + 48) + 512, 48, true };
    const ctx_cfg cfg_dense = { false, cfg_ring.n_ctx,                 48, true };

    const int32_t n_prompt = (int32_t) cfg_ring.n_ctx - 32;

    std::vector<llama_token> tokens(n_prompt);
    for (int32_t i = 0; i < n_prompt; ++i) {
        tokens[i] = (llama_token) ((i * 7 + 3) % n_vocab);
    }
    const llama_token next_tok  = (llama_token) (11 % n_vocab);
    const llama_token next_tok2 = (llama_token) (13 % n_vocab);

    llama_context * ctx_a = make_ctx(model, cfg_ring);
    check(ctx_a != nullptr, "ring context created");
    check(ctx_a && llama_kv_self_is_swa_ring(ctx_a), "ring KV cache engaged (test is not vacuous)");
    if (ctx_a == nullptr || !llama_kv_self_is_swa_ring(ctx_a)) {
        llama_free_model(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    check(decode_tokens(ctx_a, tokens.data(), n_prompt, 0), "ring context decoded a prompt longer than the window");

    std::vector<uint8_t> blob_ring = seq_save(ctx_a);
    check(!blob_ring.empty(), "ring sequence state serializes (used to refuse with 0)");
    check(blob_ring.size() == llama_state_seq_get_size(ctx_a, 0, 0),
          "llama_state_seq_get_size agrees with the bytes actually written");
    uint32_t ring_magic = 0, ring_size_swa = 0;   // filled in from the blob below

    const std::vector<float> logits_ref = decode_one(ctx_a, next_tok, n_prompt, n_vocab);
    check(!logits_ref.empty(), "reference continuation decoded on the saving context");

    llama_context * ctx_d = make_ctx(model, cfg_dense);
    check(ctx_d != nullptr && !llama_kv_self_is_swa_ring(ctx_d), "dense reference context created without a ring");
    check(ctx_d && decode_tokens(ctx_d, tokens.data(), n_prompt, 0), "dense context decoded the same prompt");
    std::vector<uint8_t> blob_dense = ctx_d ? seq_save(ctx_d) : std::vector<uint8_t>();
    check(!blob_dense.empty(), "dense sequence state serializes");
    check(blob_ring.size() < blob_dense.size(),
          "ring blob is smaller than the dense blob (only the window was written)");
    printf("     ring blob %zu bytes, dense blob %zu bytes, %d cells\n",
            blob_ring.size(), blob_dense.size(), n_prompt);

    llama_context * ctx_b = make_ctx(model, cfg_ring);
    check(ctx_b != nullptr, "fresh ring context for restore");
    if (ctx_b != nullptr && !blob_ring.empty()) {
        const size_t n = llama_state_seq_set_data(ctx_b, blob_ring.data(), blob_ring.size(), 0, 0);
        check(n == blob_ring.size(), "ring sequence state restores fully");

        const std::vector<uint8_t> blob_again = seq_save(ctx_b);
        check(blob_again.size() == blob_ring.size() &&
              memcmp(blob_again.data(), blob_ring.data(), blob_ring.size()) == 0,
              "re-serializing the restored ring state reproduces the blob byte-for-byte");

        const std::vector<float> logits_restored = decode_one(ctx_b, next_tok, n_prompt, n_vocab);
        const float d = max_abs_diff(logits_ref, logits_restored);
        check(d <= 2e-3f, "restored ring state continues with the same logits");
        printf("     max |logit diff| after restore: %g\n", d);
    }

    {
        const size_t full_size = llama_state_get_size(ctx_a);
        check(full_size > 0, "full ring state serializes (used to refuse with 0)");
        std::vector<uint8_t> blob_full(full_size);
        const size_t written = full_size ? llama_state_get_data(ctx_a, blob_full.data(), blob_full.size()) : 0;
        check(written == full_size, "full ring state written completely");

        llama_context * ctx_f = make_ctx(model, cfg_ring);
        check(ctx_f != nullptr, "fresh ring context for full-state restore");
        if (ctx_f != nullptr && written == full_size && full_size > 0) {
            const size_t nread = llama_state_set_data(ctx_f, blob_full.data(), blob_full.size());
            check(nread == full_size, "full ring state restores fully");

            const std::vector<float> l_ref  = decode_one(ctx_a, next_tok2, n_prompt + 1, n_vocab);
            const std::vector<float> l_test = decode_one(ctx_f, next_tok2, n_prompt + 1, n_vocab);
            const float d = max_abs_diff(l_ref, l_test);
            check(d <= 2e-3f, "full-state restore continues with the same logits");
            printf("     max |logit diff| after full-state restore: %g\n", d);
        }
        if (ctx_f) llama_free(ctx_f);
    }

    if (!blob_ring.empty()) {
        llama_context * ctx_x = make_ctx(model, cfg_dense);
        const size_t n = ctx_x ? llama_state_seq_set_data(ctx_x, blob_ring.data(), blob_ring.size(), 0, 0) : 1;
        check(n == 0, "ring blob is refused by a dense KV cache");
        if (ctx_x) llama_free(ctx_x);
    }
    if (blob_ring.size() > 64) {
        llama_context * ctx_x = make_ctx(model, cfg_ring);
        std::vector<uint8_t> trunc(blob_ring.begin(), blob_ring.end() - 16);
        const size_t n = ctx_x ? llama_state_seq_set_data(ctx_x, trunc.data(), trunc.size(), 0, 0) : 1;
        check(n == 0, "truncated ring blob is refused");
        check(ctx_x && decode_tokens(ctx_x, tokens.data(), 16, 0),
              "ring context is still usable after a failed restore");
        if (ctx_x) llama_free(ctx_x);
    }
    if (!blob_dense.empty()) {
        llama_context * ctx_x = make_ctx(model, cfg_ring);
        const size_t n = ctx_x ? llama_state_seq_set_data(ctx_x, blob_dense.data(), blob_dense.size(), 0, 0) : 1;
        check(n == 0, "dense blob is refused by a ring KV cache");
        if (ctx_x) llama_free(ctx_x);
    }
    if (!blob_ring.empty()) {
        const size_t off_descr   = sizeof(uint32_t) + (size_t) n_prompt * 2 * sizeof(uint32_t) + 2 * sizeof(uint32_t);
        const size_t off_size_swa = off_descr + sizeof(uint32_t);
        check(off_size_swa + sizeof(uint32_t) <= blob_ring.size(), "ring descriptor lies inside the blob");

        memcpy(&ring_magic,    blob_ring.data() + off_descr,    sizeof(ring_magic));
        memcpy(&ring_size_swa, blob_ring.data() + off_size_swa, sizeof(ring_size_swa));
        check(ring_magic == 0x474e4952u /* "RING" */ && ring_size_swa > 0,
              "ring descriptor is at the expected offset with the expected magic");
        printf("     ring descriptor: magic=0x%08x size_swa=%u\n", ring_magic, ring_size_swa);

        std::vector<uint8_t> tampered = blob_ring;
        const uint32_t bad_size_swa = ring_size_swa + 32;
        memcpy(tampered.data() + off_size_swa, &bad_size_swa, sizeof(bad_size_swa));

        llama_context * ctx_x = make_ctx(model, cfg_ring);
        const size_t n = ctx_x ? llama_state_seq_set_data(ctx_x, tampered.data(), tampered.size(), 0, 0) : 1;
        check(n == 0, "blob with a mismatched ring window size is refused");
        if (ctx_x) llama_free(ctx_x);
    }

    {
        llama_context * ctx_r = make_ctx(model, cfg_ring);
        const int32_t n_deep = std::min<int32_t>(n_prompt, (int32_t) ring_size_swa * 2);
        bool ok = ctx_r != nullptr && ring_size_swa > 0 && n_deep > (int32_t) ring_size_swa &&
                  decode_tokens(ctx_r, tokens.data(), n_deep, 0);
        check(ok, "context filled well past the ring for the rewind test");

        const bool refused = ok && !llama_kv_cache_seq_rm(ctx_r, 0, 1, -1);
        check(refused, "a tail rewind deeper than the resident window is refused");

        const bool cleared = ok && llama_kv_cache_seq_rm(ctx_r, 0, -1, -1);
        check(cleared, "the whole-sequence removal callers fall back to is accepted");

        llama_context * ctx_c = make_ctx(model, cfg_ring);
        const int32_t n_short = std::min<int32_t>(n_deep, 96);
        std::vector<float> l_reused = (ok && cleared && decode_tokens(ctx_r, tokens.data(), n_short, 0))
                ? decode_one(ctx_r, next_tok, n_short, n_vocab) : std::vector<float>();
        std::vector<float> l_fresh = (ctx_c && decode_tokens(ctx_c, tokens.data(), n_short, 0))
                ? decode_one(ctx_c, next_tok, n_short, n_vocab) : std::vector<float>();
        const float d = max_abs_diff(l_reused, l_fresh);
        check(d <= 2e-3f, "reprocessing after the refusal matches a fresh context");
        printf("     max |logit diff| after refused-rewind reprocess: %g\n", d);
        if (ctx_r) llama_free(ctx_r);
        if (ctx_c) llama_free(ctx_c);
    }

    {
        llama_context * ctx_g = make_ctx(model, cfg_ring);
        bool ok = ctx_g != nullptr && decode_tokens(ctx_g, tokens.data(), 64, 0);
        const std::vector<float> before = ok ? decode_one(ctx_g, next_tok, 64, n_vocab) : std::vector<float>();
        if (ctx_g) {
            llama_kv_cache_defrag(ctx_g);
        }
        const std::vector<float> after = ok ? decode_one(ctx_g, next_tok, 65, n_vocab) : std::vector<float>();
        check(ok && !after.empty(), "explicit llama_kv_cache_defrag() on a ring context is refused, not fatal");

        llama_context * ctx_h = make_ctx(model, cfg_ring);
        bool ok2 = ctx_h != nullptr && decode_tokens(ctx_h, tokens.data(), 64, 0);
        const std::vector<float> ref  = ok2 ? decode_one(ctx_h, next_tok, 64, n_vocab) : std::vector<float>();
        check(!before.empty() && max_abs_diff(before, ref) == 0.0f, "defrag-request baseline matches a clean context");
        const std::vector<float> ref2 = ok2 ? decode_one(ctx_h, next_tok, 65, n_vocab) : std::vector<float>();
        const float d = max_abs_diff(after, ref2);
        check(d <= 2e-3f, "the refused defrag left the ring cache intact");
        printf("     max |logit diff| after a refused defrag: %g\n", d);
        if (ctx_g) llama_free(ctx_g);
        if (ctx_h) llama_free(ctx_h);
    }

    check_round_trip(model, cfg_ring, std::max<int32_t>(8, (int32_t) hparams.n_swa / 2), n_vocab, 0,
            "short-prompt ring round-trip (no wrap yet)");

    ctx_cfg cfg_ring_vtrans = cfg_ring;
    cfg_ring_vtrans.flash_attn = false;
    check_round_trip(model, cfg_ring_vtrans, n_prompt, n_vocab, 0,
            "transposed-V ring round-trip (wrapped)");
    check_round_trip(model, cfg_ring_vtrans, std::max<int32_t>(8, (int32_t) hparams.n_swa / 2), n_vocab, 0,
            "transposed-V ring round-trip (no wrap yet)");

    if (ring_size_swa > 0) {
        const int32_t n_wrap = (int32_t) cfg_ring.n_ctx - (int32_t) ring_size_swa - 48;
        if (n_wrap > (int32_t) ring_size_swa) {
            check_round_trip(model, cfg_ring, n_wrap, n_vocab, (int32_t) ring_size_swa + 8,
                    "ring round-trip, continued past the whole ring (row-major V)");
            check_round_trip(model, cfg_ring_vtrans, n_wrap, n_vocab, (int32_t) ring_size_swa + 8,
                    "ring round-trip, continued past the whole ring (transposed V)");
        } else {
            check(false, "context too small to continue past the whole ring");
        }
    }

    if (ctx_b) llama_free(ctx_b);
    if (ctx_d) llama_free(ctx_d);
    llama_free(ctx_a);
    llama_free_model(model);
    llama_backend_free();

    if (g_fails != 0) {
        printf("SWA ring state save/restore: %d check(s) failed\n", g_fails);
        return EXIT_FAILURE;
    }
    printf("SWA ring state save/restore OK\n");
    return EXIT_SUCCESS;
}
