//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

// Stride-aware variant (2026-04-26): honors per-dim strides s01/s02 (in float elements).
// Use this when src tensor may be a non-contiguous view (permute / gather / reshape).
// s01 = stride of row (typically src->nb[1]/sizeof(float))
// s02 = stride of channel (typically src->nb[2]/sizeof(float))
// Existing API (without strides) wraps this assuming contiguous: s01=kx0, s02=kx0*kx1.
void quantize_mmq_q8_1_strided_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const int64_t s01, const int64_t s02,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_id_cuda(
    const float * x, void * vy, const char * row_mapping, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

// For now only applicable for tensors with ne[1] = 1, ne[3] = 1, and useful if ne[2] > 1
void quantize_tensor_q8_1_cuda(const struct ggml_tensor * src, void * vy, const enum ggml_type type, cudaStream_t stream);
