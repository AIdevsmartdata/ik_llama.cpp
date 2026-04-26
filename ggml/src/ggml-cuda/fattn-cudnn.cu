//
// Copyright (C) 2024-2026 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// cuDNN flash-attention backend.
//
// J2 scope (2026-04-26):
//   - F16 / BF16 K and V only (quantized K/V → fall through to legacy kernels)
//   - causal or explicit mask (passed as attn_bias)
//   - GQA via H_q != H_k (cuDNN broadcasts automatically)
//   - simple shape-keyed graph cache, mutex-protected
//   - one cudnnHandle per device, lazy-init via ctx.cudnn_handle()
//
// Bypass legacy kernels via env IK_LLAMA_FA_BACKEND=cudnn (handled in fattn.cu).
//

#include "fattn-cudnn.cuh"
#include "convert.cuh"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <vector>

#if defined(GGML_USE_CUDNN)
#include <cudnn.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// Inline F16→F32 cast kernel (D elements). Used to move dense HALF cuDNN output
// into ggml's F32 dst tensor at arbitrary strided byte offset.
static __global__ void k_half_to_float_d(const half * __restrict__ src, float * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
static inline void launch_half_to_float(const half * src, float * dst, int n, cudaStream_t s) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    k_half_to_float_d<<<grid, block, 0, s>>>(src, dst, n);
}

// Stride-aware Q F32 → HALF dense BHSD cast.
// Reads Q via ggml ne[]/nb[] (handles any permutation, including the typical
// `ggml_permute(0, 2, 1, 3)` that produces ne=[D, S, H, B] with non-contiguous nb).
// Writes contiguous BHSD = [B, H, S, D] cuDNN-canonical layout.
static __global__ void k_cast_q_to_dense_bhsd(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t H, const int64_t S, const int64_t D) {
    const int64_t total = ne3 * ne2 * ne1 * ne0;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t i = idx;
    const int64_t d = i % ne0; i /= ne0;
    const int64_t s = i % ne1; i /= ne1;
    const int64_t h = i % ne2;
    const int64_t b = i / ne2;
    const char * p = (const char*)src + b*nb3 + h*nb2 + s*nb1 + d*nb0;
    const float v = *(const float*)p;
    const int64_t out_idx = ((b * H + h) * S + s) * D + d;
    dst[out_idx] = __float2half(v);
}

static inline void launch_cast_q_strided(
        const ggml_tensor * Q, half * q_f16,
        int64_t H, int64_t S, int64_t D, cudaStream_t stream) {
    const int64_t total = Q->ne[0] * Q->ne[1] * Q->ne[2] * Q->ne[3];
    if (total <= 0) return;
    const int block = 256;
    const int grid  = (int)((total + block - 1) / block);
    k_cast_q_to_dense_bhsd<<<grid, block, 0, stream>>>(
        (const float*)Q->data, q_f16,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3],
        H, S, D);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define IK_CUDNN_CHECK(expr) \
    do { cudnnStatus_t _s = (expr); \
         if (_s != CUDNN_STATUS_SUCCESS) { \
             fprintf(stderr, "[fa-cudnn] cuDNN error %d at %s:%d: %s\n", \
                     (int)_s, __FILE__, __LINE__, cudnnGetErrorString(_s)); \
             GGML_ABORT("cuDNN failure"); \
         } } while (0)

static fe::DataType_t ggml_to_cudnn_dtype(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return fe::DataType_t::HALF;
        case GGML_TYPE_BF16: return fe::DataType_t::BFLOAT16;
        case GGML_TYPE_F32:  return fe::DataType_t::FLOAT;
        default: return fe::DataType_t::NOT_SET;
    }
}

#endif // GGML_USE_CUDNN

// ---------------------------------------------------------------------------
// Support probe
// ---------------------------------------------------------------------------

bool ggml_cuda_fattn_cudnn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
#if !defined(GGML_USE_CUDNN)
    (void) ctx; (void) dst;
    return false;
#else
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) return false;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) return false;

    // Sliding window not handled yet by this path.
    const int32_t n_swa = dst->op_params[4];
    if (n_swa != 0) return false;

    // No logit softcap support yet (cuDNN supports via post-scale, leave for later).
    float logit_softcap = 0.0f;
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) return false;

    // No ALiBi for now (max_bias != 0).
    float max_bias = 0.0f;
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return false;

    // Head dim whitelist (matches cuDNN SDPA fast paths). J2 scope: restrict
    // to {64, 128} which are the well-tested paths on sm_120 cuDNN 9.21.
    // 256 head_dim (sometimes used by mtmd/vision layers) hits an unstable
    // SDPA path on Blackwell consumer; defer to J3+ once core path validated.
    const int64_t head_dim = Q->ne[0];
    switch (head_dim) {
        case 64: case 128: break;
        default: return false;
    }

    // Q must be F16/F32, mask must be F16 or absent.
    if (Q->type != GGML_TYPE_F16 && Q->type != GGML_TYPE_F32) return false;
    if (mask && mask->type != GGML_TYPE_F16) return false;

    // J2 scope: only F16 / BF16 KV. Quantized → legacy path.
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_BF16) return false;
    if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_BF16) return false;

    // ggml asserts ne[3]==1 in FA kernel; mirror here.
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;

    // Require cuDNN >= 9.0 at runtime.
    if (cudnnGetVersion() < 9000) return false;

    (void) ctx;
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Graph cache
// ---------------------------------------------------------------------------

#if defined(GGML_USE_CUDNN)
namespace {

// UIDs used inside the graph (stable across cache entries).
constexpr int64_t UID_Q       = 1;
constexpr int64_t UID_K       = 2;
constexpr int64_t UID_V       = 3;
constexpr int64_t UID_BIAS    = 4;
constexpr int64_t UID_O       = 5;
constexpr int64_t UID_SCALE   = 6;

struct fa_cache_key {
    int64_t  B;
    int64_t  H_q, H_k;
    int64_t  S_q, S_kv;
    int64_t  D_qk, D_v;
    // Strides in elements (cuDNN BHSD order = stride[0]=B, stride[1]=H, stride[2]=S, stride[3]=D).
    int64_t  q_stride[4];
    int64_t  k_stride[4];
    int64_t  v_stride[4];
    int64_t  o_stride[4];
    int32_t  q_dtype, kv_dtype;
    int32_t  has_mask;
    int32_t  causal;
    int32_t  pad_;
    bool operator==(const fa_cache_key & o) const noexcept {
        return std::memcmp(this, &o, sizeof(o)) == 0;
    }
};
struct fa_cache_hash {
    size_t operator()(const fa_cache_key & k) const noexcept {
        const uint64_t * p = reinterpret_cast<const uint64_t *>(&k);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(k)/sizeof(uint64_t); ++i) {
            h ^= p[i]; h *= 1099511628211ull;
        }
        return (size_t) h;
    }
};

struct fa_graph_entry {
    std::shared_ptr<fe::graph::Graph> graph;
};

static std::mutex                                                          g_fa_cache_mu;
static std::unordered_map<fa_cache_key, fa_graph_entry, fa_cache_hash>     g_fa_cache;

// Build a fresh graph for the given key. Caller holds the mutex.
static fa_graph_entry build_graph(const fa_cache_key & k, cudnnHandle_t handle, bool has_bias_tensor) {
    auto graph = std::make_shared<fe::graph::Graph>();

    // cuDNN SDPA requires Q to match K/V dtype. ggml passes Q as F32 always
    // (asserted in launch_fattn). We cast Q F32→HALF in a pool buffer before
    // calling cuDNN. So inside the graph, Q is treated as HALF.
    auto kv_dtype = (k.kv_dtype == GGML_TYPE_BF16) ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
    auto q_dtype  = kv_dtype;

    graph->set_io_data_type(kv_dtype)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // ggml convention (D=fastest, B=slowest), translated to cuDNN (B,H,S,D):
    // Q ggml shape [D, S, H, 1] with strides nb[0..3] (bytes).
    // cuDNN dim {B=1, H, S, D}, stride in elements:
    //   stride_D = 1
    //   stride_S = nb[1]/sizeof(elem)
    //   stride_H = nb[2]/sizeof(elem)
    //   stride_B = nb[3]/sizeof(elem)
    // We pass strides as elements; ne are unchanged.

    // Q tensor — strides set at execute time? No: cuDNN bakes them at build time.
    // We set strides assuming dense / packed (S contiguous in token-dim, head as stride S*D).
    // For ggml's actual tensors at execute, those strides match because Q is built fresh each step.
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q").set_uid(UID_Q)
                               .set_dim({k.B, k.H_q, k.S_q, k.D_qk})
                               .set_stride({k.q_stride[0], k.q_stride[1], k.q_stride[2], k.q_stride[3]})
                               .set_data_type(q_dtype));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K").set_uid(UID_K)
                               .set_dim({k.B, k.H_k, k.S_kv, k.D_qk})
                               .set_stride({k.k_stride[0], k.k_stride[1], k.k_stride[2], k.k_stride[3]})
                               .set_data_type(kv_dtype));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V").set_uid(UID_V)
                               .set_dim({k.B, k.H_k, k.S_kv, k.D_v})
                               .set_stride({k.v_stride[0], k.v_stride[1], k.v_stride[2], k.v_stride[3]})
                               .set_data_type(kv_dtype));

    auto opts = fe::graph::SDPA_attributes()
                    .set_name("ggml_fa_sdpa")
                    .set_generate_stats(false);

    // attn_scale baked at graph build time. ggml's scale is in op_params[0]; for
    // standard transformers it equals 1/sqrt(D_qk) but Gemma/MLA may differ. We
    // store the scale as a constant tensor; cuDNN inlines it.
    // (Source of truth at execute time still verified via op_params[0] in entry().)
    opts.set_attn_scale(1.0f / std::sqrt(float(k.D_qk)));

    if (k.causal) {
        opts.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    static const bool no_bias_build = []{
        const char * s = std::getenv("IK_LLAMA_FA_NO_BIAS");
        return s && std::strcmp(s, "0") != 0;
    }();
    if (has_bias_tensor && !no_bias_build) {
        // ggml mask: F16, dim [S_kv, S_q_padded, 1, 1], stride nb1=S_kv*sizeof(half).
        // Broadcast across B and H by setting their strides to 0; for S_q dimension
        // the stride is the row size of the ggml mask in HALF elements.
        // cuDNN dim order is {B, H, S_q, S_kv}.
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias").set_uid(UID_BIAS)
                                      .set_dim({1, 1, k.S_q, k.S_kv})
                                      .set_stride({0, 0, k.S_kv, 1})
                                      .set_data_type(fe::DataType_t::HALF));
        opts.set_bias(bias);
    }

    auto [O, _stats] = graph->sdpa(Q, K, V, opts);

    // Output: ggml dst has shape [D_v, S_q, H_q, 1]. cuDNN computes
    // [B, H_q, S_q, D_v]. Match the strides so cuDNN writes directly into
    // ggml's destination memory with no extra permute.
    // Output: HALF dense BHSD packed (we cast HALF→F32 → ggml dst after).
    O->set_output(true).set_uid(UID_O)
     .set_dim({k.B, k.H_q, k.S_q, k.D_v})
     .set_stride({k.H_q * k.S_q * k.D_v, k.S_q * k.D_v, k.D_v, 1})
     .set_data_type(fe::DataType_t::HALF);

    auto status = graph->build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "[fa-cudnn] graph build failed: %s\n", status.get_message().c_str());
        GGML_ABORT("cuDNN graph build failed");
    }

    fa_graph_entry e;
    e.graph = graph;
    return e;
}

} // namespace
#endif // GGML_USE_CUDNN

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_CUDNN)
    (void) ctx; (void) dst;
    GGML_ABORT("ik_llama was built without GGML_USE_CUDNN");
#else
    GGML_ASSERT(dst && dst->op == GGML_OP_FLASH_ATTN_EXT);

    ggml_tensor * Q    = dst->src[0];
    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    // Build cache key based on shapes + ACTUAL tensor strides.
    // ggml stride convention: nb[0..3] in bytes. cuDNN expects strides in elements,
    // and the dim order is [B, H, S, D] → for ggml [D, S, H, B] this maps as:
    //   cuDNN.stride[0]=B  ← ggml.nb[3]/elemsize
    //   cuDNN.stride[1]=H  ← ggml.nb[2]/elemsize
    //   cuDNN.stride[2]=S  ← ggml.nb[1]/elemsize
    //   cuDNN.stride[3]=D  ← ggml.nb[0]/elemsize  (= 1 for native types)
    // Compute strides in elements of the *runtime* dtype (HALF for cuDNN).
    // For tensors that ggml stores as F32 but we cast to F16 (Q), strides are
    // computed assuming dense BHSD F16 packing of the cast result.
    auto compute_stride = [](const ggml_tensor * t, int64_t * out, bool dense_half_after_cast) {
        if (dense_half_after_cast) {
            // After F32→F16 cast we write dense, ggml-shape-preserving, BHSD layout
            // following ggml ne ordering: out[0]=B-stride=H*S*D, etc.
            const int64_t D = t->ne[0];
            const int64_t S = t->ne[1];
            const int64_t H = t->ne[2];
            out[0] = H * S * D;
            out[1] = S * D;
            out[2] = D;
            out[3] = 1;
            return;
        }
        const size_t es = ggml_type_size(t->type);
        out[0] = (int64_t)(t->nb[3] / es);
        out[1] = (int64_t)(t->nb[2] / es);
        out[2] = (int64_t)(t->nb[1] / es);
        out[3] = (int64_t)(t->nb[0] / es);
        if (out[3] == 0) out[3] = 1;
    };

    fa_cache_key key = {};
    key.B        = Q->ne[3];
    key.H_q      = Q->ne[2];
    key.H_k      = K->ne[2];
    key.S_q      = Q->ne[1];
    key.S_kv     = K->ne[1];
    key.D_qk     = Q->ne[0];
    key.D_v      = V->ne[0];
    // Q gets cast to HALF in dense BHSD (we wrote it that way below). K/V
    // honor their actual ggml strides (they may be views of the KV cache).
    compute_stride(Q,   key.q_stride, /*dense_half_after_cast=*/true);
    compute_stride(K,   key.k_stride, /*dense_half_after_cast=*/false);
    compute_stride(V,   key.v_stride, /*dense_half_after_cast=*/false);
    compute_stride(dst, key.o_stride, /*dense_half_after_cast=*/false);
    key.q_dtype  = (int32_t) Q->type;
    key.kv_dtype = (int32_t) K->type;
    key.has_mask = mask ? 1 : 0;
    key.causal   = 0;

    // Get or build graph.
    fa_graph_entry * entry = nullptr;
    cudnnHandle_t handle = ctx.cudnn_handle(ctx.device);
    IK_CUDNN_CHECK(cudnnSetStream(handle, stream));
    static const bool no_cache = []{
        const char * s = std::getenv("IK_LLAMA_FA_NO_CACHE");
        return s && std::strcmp(s, "0") != 0;
    }();
    fa_graph_entry built_local;
    {
        std::lock_guard<std::mutex> lock(g_fa_cache_mu);
        if (no_cache) {
            built_local = build_graph(key, handle, key.has_mask != 0);
            entry = &built_local;
        } else {
            auto it = g_fa_cache.find(key);
            if (it == g_fa_cache.end()) {
                fa_graph_entry built = build_graph(key, handle, key.has_mask != 0);
                it = g_fa_cache.emplace(key, std::move(built)).first;
            }
            entry = &it->second;
        }
    }

    // Workspace.
    int64_t ws_size = 0;
    auto ws_status = entry->graph->get_workspace_size(ws_size);
    if (!ws_status.is_good()) {
        GGML_ABORT("cuDNN get_workspace_size failed");
    }
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool());
    if (ws_size > 0) workspace.alloc(ws_size);

    // Cast Q F32 → HALF stride-aware (Q is non-contiguous permuted view).
    // We allocate a dense F16 buffer in cuDNN-expected BHSD layout, then call
    // ggml's existing F32→F16 conversion which respects strides via offset
    // arithmetic.
    ggml_cuda_pool_alloc<half> q_f16(ctx.pool());
    void * q_data = Q->data;
    if (Q->type == GGML_TYPE_F32) {
        const int64_t q_nelem = ggml_nelements(Q);
        q_f16.alloc(q_nelem);
        // Stride-aware F32→HALF cast. Respects ggml's ne[]/nb[] (any permutation),
        // writes dense BHSD layout matching what cuDNN's q_stride expects.
        launch_cast_q_strided(Q, q_f16.ptr, key.H_q, key.S_q, key.D_qk, stream);
        q_data = q_f16.ptr;
    }

    // HALF intermediate buffer; cast back to F32 dst after execute.
    const int64_t out_nelem = key.B * key.H_q * key.S_q * key.D_v;
    ggml_cuda_pool_alloc<half> o_f16(ctx.pool());
    o_f16.alloc(out_nelem);

    std::unordered_map<int64_t, void *> variant_pack = {
        {UID_Q, q_data},
        {UID_K, K->data},
        {UID_V, V->data},
        {UID_O, o_f16.ptr},
    };
    static const bool no_bias = []{
        const char * s = std::getenv("IK_LLAMA_FA_NO_BIAS");
        return s && std::strcmp(s, "0") != 0;
    }();
    if (mask && !no_bias) { variant_pack[UID_BIAS] = mask->data; }

    {
        static int debug_n = 0;
        if (debug_n++ < 3) {
            float scale_dbg = 0.f;
            std::memcpy(&scale_dbg, (const float *) dst->op_params + 0, sizeof(float));
            fprintf(stderr, "[fa-cudnn-exec] mask=%s mask.ne1=%lld mask.nb1=%zu  Q->nb={%zu,%zu,%zu,%zu}  K->nb={%zu,%zu,%zu,%zu}  V->nb={%zu,%zu,%zu,%zu}  dst->nb={%zu,%zu,%zu,%zu}  scale=%f  D_qk=%lld H_q=%lld H_k=%lld S_q=%lld S_kv=%lld\n",
                mask ? "YES" : "NO",
                mask ? (long long)mask->ne[1] : 0LL,
                mask ? mask->nb[1] : 0,
                Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                scale_dbg,
                (long long)key.D_qk, (long long)key.H_q, (long long)key.H_k, (long long)key.S_q, (long long)key.S_kv);
        }
    }
    auto exec_status = entry->graph->execute(handle, variant_pack, ws_size > 0 ? workspace.ptr : nullptr);
    if (!exec_status.is_good()) {
        fprintf(stderr, "[fa-cudnn] execute failed: %s\n", exec_status.get_message().c_str());
        GGML_ABORT("cuDNN graph execute failed");
    }

    // Cast HALF dense BHSD output → F32 dst (linear dense, ignoring nb-padding,
    // matches what legacy mma_f16 does).
    {
        const int64_t total = key.B * key.H_q * key.S_q * key.D_v;
        launch_half_to_float(o_f16.ptr, (float*)dst->data, (int)total, stream);
    }

    // ---- Diagnostic tensor dump on first call ----
    {
        static int dump_n = 0;
        if (dump_n++ < 200) {
            cudaStreamSynchronize(stream);
            fprintf(stderr, "[fa-cudnn-dump] CALL %d: ", dump_n - 1);
            auto dump_half = [&](const char * name, const void * dptr, int n) {
                if (!dptr) { fprintf(stderr, "[fa-cudnn-dump] %s: null\n", name); return; }
                std::vector<uint16_t> h(n);
                cudaMemcpy(h.data(), dptr, n*2, cudaMemcpyDeviceToHost);
                fprintf(stderr, "[fa-cudnn-dump] %s (HALF):", name);
                for (int i=0;i<n;++i) {
                    half hv; std::memcpy(&hv, &h[i], 2);
                    fprintf(stderr, " %.4e", (double)__half2float(hv));
                }
                fprintf(stderr, "\n");
            };
            auto dump_f32 = [&](const char * name, const void * dptr, int n) {
                if (!dptr) { fprintf(stderr, "[fa-cudnn-dump] %s: null\n", name); return; }
                std::vector<float> h(n);
                cudaMemcpy(h.data(), dptr, n*4, cudaMemcpyDeviceToHost);
                fprintf(stderr, "[fa-cudnn-dump] %s (F32):", name);
                for (int i=0;i<n;++i) fprintf(stderr, " %.4e", (double)h[i]);
                fprintf(stderr, "\n");
            };
            const int N = 4;
            const int64_t D = key.D_v;
            const int64_t H = key.H_q;
            // Compact one-line hash dump : Q_h0[0..3] + dst_h0[0..3]
            std::vector<uint16_t> qh(N);
            cudaMemcpy(qh.data(), q_data, N*2, cudaMemcpyDeviceToHost);
            fprintf(stderr, "Q_h0=");
            for (int i=0;i<N;++i) { half hv; std::memcpy(&hv,&qh[i],2); fprintf(stderr, "%+.4e,", (double)__half2float(hv)); }
            std::vector<float> dh(N);
            cudaMemcpy(dh.data(), dst->data, N*4, cudaMemcpyDeviceToHost);
            fprintf(stderr, " dst_h0=");
            for (int i=0;i<N;++i) fprintf(stderr, "%+.4e,", (double)dh[i]);
            fprintf(stderr, " S_kv=%lld\n", (long long)key.S_kv);
            fflush(stderr);
        }
    }

    // Verify scale matches our build-time hardcode; abort if model uses different.
    {
        float scale_op = 0.f;
        std::memcpy(&scale_op, (const float*)dst->op_params + 0, sizeof(float));
        const float scale_expected = 1.0f / std::sqrt(float(key.D_qk));
        if (std::fabs(scale_op - scale_expected) > 1e-4f) {
            fprintf(stderr, "[fa-cudnn] scale mismatch: op_params=%f expected=%f — fallback unsupported\n",
                scale_op, scale_expected);
            GGML_ABORT("custom scale not yet supported in cuDNN backend");
        }
    }

    // Diagnostic: dump mask validity across calls to verify causal coverage grows
    {
        static int mask_dump_n = 0;
        if (mask && mask_dump_n++ < 30) {
            cudaStreamSynchronize(stream);
            std::vector<uint16_t> mh(std::min((int64_t)64, mask->ne[0]));
            cudaMemcpy(mh.data(), mask->data, mh.size()*2, cudaMemcpyDeviceToHost);
            int nvalid = 0;
            for (size_t i=0; i<mh.size(); ++i) {
                half hv; std::memcpy(&hv, &mh[i], 2);
                if (__half2float(hv) == 0.0f) nvalid++;
            }
            // Dump first 8 mask values as hex + decoded
            fprintf(stderr, "[fa-cudnn-mask] CALL %d: nvalid_64=%d S_kv=%lld S_q=%lld | hex=", mask_dump_n - 1, nvalid, (long long)key.S_kv, (long long)key.S_q);
            for (int i = 0; i < 8 && i < (int)mh.size(); ++i) {
                half hv; std::memcpy(&hv, &mh[i], 2);
                float f = __half2float(hv);
                fprintf(stderr, "%04x(%s) ", mh[i], std::isinf(f) ? "-inf" : (f == 0.0f ? "0" : "?"));
            }
            // Also dump row 1 (s_q=1) first 8 if mask has multiple rows
            if (mask->ne[1] >= 2) {
                std::vector<uint16_t> mh1(8);
                cudaMemcpy(mh1.data(), (const char*)mask->data + mask->nb[1], 16, cudaMemcpyDeviceToHost);
                fprintf(stderr, "| row1=");
                for (int i = 0; i < 8; ++i) {
                    half hv; std::memcpy(&hv, &mh1[i], 2);
                    float f = __half2float(hv);
                    fprintf(stderr, "%04x(%s) ", mh1[i], std::isinf(f) ? "-inf" : (f == 0.0f ? "0" : "?"));
                }
            }
            fprintf(stderr, "\n");
        }
    }

    // Force sync to get accurate error reporting (for diagnostic; remove later for perf).
    cudaError_t cerr = cudaStreamSynchronize(stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[fa-cudnn] post-execute sync failed: %s | shape Q=%lld,%lld,%lld K=%lld,%lld,%lld V=%lld,%lld,%lld dst.type=%d ws=%lld\n",
            cudaGetErrorString(cerr),
            (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2],
            (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2],
            (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2],
            (int)dst->type, (long long)ws_size);
        GGML_ABORT("cuDNN post-execute CUDA error");
    }
#endif
}
