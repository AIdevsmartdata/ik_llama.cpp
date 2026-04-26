//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn-cudnn.cuh"
#include "fattn.cuh"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "convert.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE 256

static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > CC_TURING;
}

// Diagnostic: dump V[kv=0, h_k=0, d=0..3] (input) + dst[d=0..3] (output) per call.
static inline void ik_fa_dump_dst_post(ggml_tensor * dst, cudaStream_t stream) {
    static const bool enabled = []{
        const char * s = std::getenv("IK_LLAMA_FA_DUMP");
        return s && std::strcmp(s, "0") != 0;
    }();
    if (!enabled) return;
    static int dump_n = 0;
    if (dump_n++ >= 200) return;
    cudaStreamSynchronize(stream);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];
    const int N = 4;
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * M = dst->src[3];
    // Dump |Q|_max amplitude — discriminates double-scaling
    if (Q->type == GGML_TYPE_F32) {
        std::vector<float> qsamp(128);
        cudaMemcpy(qsamp.data(), Q->data, 128*4, cudaMemcpyDeviceToHost);
        float qmax = 0;
        for (auto v : qsamp) qmax = std::max(qmax, std::fabs(v));
        fprintf(stderr, "[ik-fa-dump] CALL %d: |Q|_max=%.4f K_type=%d K_contig=%d ",
            dump_n - 1, qmax, (int)K->type, (int)ggml_is_contiguous(K));
    } else fprintf(stderr, "[ik-fa-dump] CALL %d: ", dump_n - 1);
    fprintf(stderr, "K_ne=%lld,%lld,%lld K_nb=%zu,%zu,%zu V_ne=%lld,%lld,%lld V_nb=%zu,%zu,%zu mask=%p ",
        dump_n - 1,
        (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2],
        K->nb[0], K->nb[1], K->nb[2],
        (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2],
        V->nb[0], V->nb[1], V->nb[2],
        M ? M->data : (void*)0);
    // V[kv=0, h_k=0, d=0..3] — F16 in ggml; first 4 elements at V->data offset 0
    if (V->type == GGML_TYPE_F16) {
        std::vector<uint16_t> vh(N);
        cudaMemcpy(vh.data(), V->data, N*2, cudaMemcpyDeviceToHost);
        fprintf(stderr, "V_h0_kv0=");
        for (int i=0;i<N;++i) { half hv; std::memcpy(&hv,&vh[i],2); fprintf(stderr, "%+.3e,", (double)__half2float(hv)); }
    }
    if (dst->type == GGML_TYPE_F32) {
        std::vector<float> dh(N);
        cudaMemcpy(dh.data(), dst->data, N*4, cudaMemcpyDeviceToHost);
        fprintf(stderr, " dst_h0=");
        for (int i=0;i<N;++i) fprintf(stderr, "%+.3e,", (double)dh[i]);
        // P10/P11: dump at intermediate layers to find drift inflection point
        if (dump_n == 1 || dump_n == 4 || dump_n == 8 || dump_n == 12 || dump_n == 16 || dump_n == 20 || dump_n == 24 || dump_n == 28) {
            const int total = (int)(dst->ne[0] * dst->ne[1] * dst->ne[2]);
            std::vector<float> all(total);
            cudaMemcpy(all.data(), dst->data, total*4, cudaMemcpyDeviceToHost);
            float sum2 = 0, max_abs = 0, mean = 0;
            for (auto v : all) { sum2 += v*v; max_abs = std::max(max_abs, std::fabs(v)); mean += v; }
            fprintf(stderr, " | L2=%.6g max=%.6g mean=%.6g (n=%d)", std::sqrt(sum2), max_abs, mean/total, total);
            // Save raw bytes for element-wise diff
            const char * be = std::getenv("IK_LLAMA_FA_BACKEND");
            char path[256];
            snprintf(path, 256, "/tmp/dst_call%d_%s.bin", dump_n - 1, (be && std::strcmp(be,"cudnn")==0) ? "cudnn" : "legacy");
            FILE* f = fopen(path, "wb");
            if (f) { fwrite(all.data(), 4, total, f); fclose(f); }
        }
        // dst layout probe: head 1 at LINEAR offset D*4 bytes vs nb-STRIDED offset nb[2]
        std::vector<float> dh_lin(N), dh_nb(N);
        cudaMemcpy(dh_lin.data(), (const char*)dst->data + dst->ne[0]*4, N*4, cudaMemcpyDeviceToHost);
        cudaMemcpy(dh_nb.data(), (const char*)dst->data + dst->nb[2], N*4, cudaMemcpyDeviceToHost);
        fprintf(stderr, " dst_h1@D=");
        for (int i=0;i<N;++i) fprintf(stderr, "%+.3e,", (double)dh_lin[i]);
        fprintf(stderr, " dst_h1@nb2=");
        for (int i=0;i<N;++i) fprintf(stderr, "%+.3e,", (double)dh_nb[i]);
    }
    fprintf(stderr, " S_q=%lld H_q=%lld S_kv=%lld\n",
        (long long)Q->ne[1], (long long)Q->ne[2], (long long)dst->src[1]->ne[1]);
    fflush(stderr);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    // RAII dump-on-exit so every return path is covered uniformly.
    struct DumpOnExit {
        ggml_tensor * dst; cudaStream_t s;
        ~DumpOnExit() { ik_fa_dump_dst_post(dst, s); }
    } _ik_dump_guard{dst, ctx.stream()};

    // 2026-04-26: cuDNN SDPA path. Activated by env IK_LLAMA_FA_BACKEND=cudnn.
    // Fixes Blackwell sm_120 multi-seq M>=3 illegal-memory-access bug in legacy
    // mma_f16 / wmma_f16 / tile_f16 kernels (root cause: launch_fattn dequant
    // treats K/V as 1D contiguous, ignoring strides for non-contig views).
    {
        static const bool fa_use_cudnn = []{
            const char * s = std::getenv("IK_LLAMA_FA_BACKEND");
            bool v = s && std::strcmp(s, "cudnn") == 0;
            fprintf(stderr, "[fa-cudnn-dispatch] IK_LLAMA_FA_BACKEND='%s' fa_use_cudnn=%d\n", s ? s : "(null)", (int)v);
            return v;
        }();
        if (fa_use_cudnn) {
            bool sup = ggml_cuda_fattn_cudnn_is_supported(ctx, dst);
            static int log_n = 0;
            if (log_n++ < 5) {
                fprintf(stderr, "[fa-cudnn-dispatch] is_supported=%d  Q={%lld,%lld,%lld,%lld} K={%lld,%lld,%lld,%lld} K->type=%d V->type=%d head_dim=%lld n_swa=%d\n",
                    (int)sup,
                    (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2], (long long)Q->ne[3],
                    (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3],
                    (int)K->type, (int)V->type, (long long)Q->ne[0], (int)n_swa);
            }
            if (sup) {
                ggml_cuda_flash_attn_ext_cudnn(ctx, dst);
                return;
            }
        }
    }

    ggml_tensor local_dst, Kl, Vl, Ml;
    if (n_swa > 0) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        if (first > 0) {
            local_dst = *dst;
            Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
            Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
            Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
            local_dst.src[1] = &Kl;
            local_dst.src[2] = &Vl;
            local_dst.src[3] = &Ml;
            local_dst.op_params[4] = 0;
            dst = &local_dst;
        }
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 && Q->ne[2] / K->ne[2] == 6) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // Diagnostic 2026-04-26: runtime kernel selector for FA dispatch on sm_120.
    // mma_f16 has a confirmed multi-seq batched bug on RTX 5060 Ti (sm_120) that crashes
    // with "illegal memory access" when Q->ne[1] >= 3 and prompts are heterogeneous.
    // Env vars allow forcing alternative kernels for testing/workaround:
    //   IK_LLAMA_FA_FORCE=wmma | tile | vec | mma  (default = mma, the current behavior)
    {
        static const char * fa_force = std::getenv("IK_LLAMA_FA_FORCE");
        if (fa_force) {
            if (!strcmp(fa_force, "wmma")) { ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst); return; }
            if (!strcmp(fa_force, "tile")) {
                if (precision == GGML_PREC_DEFAULT) ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
                else                                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
                return;
            }
            if (!strcmp(fa_force, "vec"))  {
                if (precision == GGML_PREC_DEFAULT) ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                else                                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                return;
            }
            // "mma" → fall through to default
        }
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];
    if (cc >= CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f16_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (new_mma_available(cc) &&
            (Q->ne[0] == 576 || Q->ne[0] == 320 || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (Q->ne[0] == 576 || Q->ne[0] == 320) {
            int gqa_ratio = Q->ne[2]/K->ne[2];
            return (gqa_ratio % 4) == 0;
        }
        return true;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(ctx, dst);
}
