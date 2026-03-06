#pragma once

#include "llama-impl.h"

#include <cstdint>
#include <algorithm>

inline uint32_t llama_kv_pad_granularity(bool flash_attn) {
    return flash_attn ? 256u : 32u;
}

// Compute the ring-window size for a SWA layer: the number of rows per sequence.
// The window is padded to the granularity so that wraps are deterministic.
inline uint32_t llama_kv_ring_win(uint32_t n_swa, uint32_t n_ubatch, bool flash_attn) {
    const uint32_t pad = std::max<uint32_t>(llama_kv_pad_granularity(flash_attn), 256u);
    return (uint32_t) GGML_PAD(n_swa + n_ubatch, pad);
}

// Compute the total ring size (rows across all sequences).
// Returns 0 if the ring would not undercut the full context (caller should decide to stay dense).
inline uint32_t llama_kv_ring_size(uint32_t n_swa, uint32_t n_ubatch, uint32_t kv_size, uint32_t n_seq_max, bool flash_attn) {
    const uint32_t w = llama_kv_ring_win(n_swa, n_ubatch, flash_attn);
    const uint32_t total = w * n_seq_max;
    return total < kv_size ? total : 0;
}

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    std::vector<std::string> devices;
    std::vector<std::string> devices_draft;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    int  mla_attn;
    int  attn_max_batch;
    bool fused_moe_up_gate;
    bool grouped_expert_routing;
    bool fused_up_gate;
    bool fused_mmad;
    bool rope_cache;
    bool graph_reuse;
    bool prefetch_experts;
    bool k_cache_hadamard;
    bool v_cache_hadamard;
    bool dsa_indexer_hadamard = true; // apply Walsh-Hadamard rotation to DSA indexer q/k (precision)
    bool dsa = false;                 // enable GLM DSA sparse attention (off by default; opt-in via --dsa)
    bool fused_idx_topk = false;      // enable the fused indexer topk op (off by default; opt-in via -fidx or --fused-indexer-topk)
    bool swa_compress = false;
    bool dsv4_cache_cpu = false;      // keep DeepSeek-V4 compressed-attention K caches (CSA/HCA) in host memory
    bool dsv4_lid_cache_cpu = false;  // also keep the DeepSeek-V4 indexer (LID) K cache in host memory
    int  dsa_top_k = -1;              // DSA top-k override (<0 => use the model's configured indexer_top_k)
    bool split_mode_tensor_parallel_scheduling;
    //bool split_mode_f16;
    bool scheduler_async;
    int  min_experts;
    float thresh_experts;
    bool mtp;
    int  worst_graph_tokens;
    int  dflash_query_capacity = 0; // internal DFlash query capacity override

    enum ggml_type reduce_type;
    enum ggml_type graph_attn_precision;
    enum ggml_type idx_type_k = GGML_TYPE_F16;
    enum llama_pooling_type pooling_type;
    enum llama_mtp_op_type mtp_op_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    void * cuda_params;
};
