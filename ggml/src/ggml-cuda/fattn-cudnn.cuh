//
// Copyright (C) 2024-2026 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// cuDNN-backed Flash-Attention path for ik_llama.cpp CUDA backend.
//
// Entry points:
//   ggml_cuda_flash_attn_ext_cudnn       - execute FA via cuDNN SDPA graph API
//   ggml_cuda_fattn_cudnn_is_supported   - capability probe (cudnn version,
//                                          dtypes, head_dim, n_swa, ...)
//
// Activation is gated in fattn.cu by the env var IK_LLAMA_FA_BACKEND=cudnn.
// When cuDNN is not built in, both functions are stubbed in the .cu so that
// is_supported() returns false and the dispatcher falls back to the existing
// MMA / WMMA / tile / vec kernels.
//

#pragma once

#include "common.cuh"

// Run flash-attention for `dst` using cuDNN's scaled dot product attention
// graph API. Caller must have already validated support via
// ggml_cuda_fattn_cudnn_is_supported(). On any runtime cuDNN error the
// function falls back via GGML_ABORT (matches existing kernel error policy).
void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Returns true iff the cuDNN backend can handle this op:
//   - cuDNN >= 9.0 available at runtime
//   - K/V dtype in {F16, BF16, Q8_0, Q4_0} (quantized => dequant to F16)
//   - Q dtype F16/F32, mask F16 or null
//   - head_dim in {64, 80, 96, 112, 128, 256}
//   - n_swa == 0 (sliding window not yet wired)
//   - logit softcap == 0 for now
bool ggml_cuda_fattn_cudnn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
