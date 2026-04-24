#pragma once

#include "llama-build-context.h"

#include <utility>

struct delta_net {
    delta_net(llama_context & lctx, const llama_batch & batch);
    ~delta_net();

    static std::pair<ggml_tensor *, ggml_tensor *> build_fused_delta_net(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      int il, const llm_build_cb & cb, int repeat_type);

    ggml_tensor * build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const;

    // M2 — batched GDN dispatch. Emits ONE gather + ONE ssm_conv (n_kv=n_seqs) + ONE
    // ggml_delta_net (n_seqs>1) + ONE scatter per layer, in place of N per-seq sub-graphs.
    // Runs only when is_batched_dispatch_eligible() returns true (env IK_LLAMA_BATCHED_GDN=1
    // and pure-decode unique-seq conditions hold). Falls back to the per-seq loop otherwise.
    ggml_tensor * build_layer_attn_linear_core_batched(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const;

    ggml_tensor * build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const;

    // M2 — public eligibility predicate so build_qwen3next()/35moe()/35() can decide
    // at graph-build time whether to allocate the batched input tensors.
    bool is_batched_dispatch_eligible() const;

private:

    llama_context     & lctx;
    const llama_batch & batch;
    std::vector<llama_seq_id> token_seq_ids;
    bool all_same_seq;
    bool has_unique_seq_ids;

    // Cached at ctor from env IK_LLAMA_BATCHED_GDN. When false (default) the
    // dispatcher never routes to the batched path; behavior is byte-identical to today.
    bool batched_gdn_enabled = false;

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il, const llm_build_cb & cb);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_beta_gate(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * ssm_beta_alpha, ggml_tensor * ssm_beta, ggml_tensor * ssm_alpha,
            ggml_tensor * ssm_dt, ggml_tensor * ssm_a, int64_t num_k_heads, int64_t num_v_heads, int64_t n_seqs,
            ggml_tensor * cur, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static ggml_tensor * build_qkv(ggml_context * ctx0, ggml_tensor * state_storage, ggml_tensor * ssm_conv1d,
            ggml_tensor * qkv_mixed, ggml_tensor * inp_s_seq_qnext, ggml_tensor * beta, ggml_tensor * gate,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, int64_t ssm_d_conv,
            int64_t state_seq_id_local, uint32_t qnext_state_slots, bool reset_state_local,
            float eps_norm, int repeat_type, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static ggml_tensor * build_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out,
            ggml_tensor * output, ggml_tensor * z, int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb);
};
