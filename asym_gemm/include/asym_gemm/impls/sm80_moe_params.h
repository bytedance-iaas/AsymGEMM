// asym_gemm/include/asym_gemm/impls/sm80_moe_params.h
// Plain C header — includable from both C++ host code and CUDA device code.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
namespace asym_gemm {
#endif

// Non-templated params struct. Data pointers are void* so this struct is
// layout-stable across the host/device JIT boundary. The kernel casts to Element*.
typedef struct {
    void*    x_ptr;         // [total_tokens, K] row-major
    void*    w_ptr;         // [num_experts, N, K] row-major
    void*    o_ptr;         // [total_tokens, N] row-major
    int32_t* expert_list;   // [list_size] expert IDs
    int32_t* index_list;    // [list_size] cumulative end-token indices
    int32_t  list_size;
    int32_t  expert_size;   // num_experts (outer dim of W)
    int64_t  N;
    int64_t  K;
} SM80MoEParams;

/*
 * FP8 params for SM89 native FP8 MMA (RTX 4090).
 * x_ptr : float8_e4m3fn in HBM          [total_tokens, K]
 * w_ptr : float8_e4m3fn in CPU pinned   [num_experts, N, K]
 * o_ptr : bfloat16 in HBM               [total_tokens, N]
 *         also used as partial-sum accumulation buffer between K-tiles
 * scale_a/b: per-tensor float32 scales applied to FP32 accumulator at the
 *            final K-tile only (intermediate writes store unscaled BF16).
 */
typedef struct {
    void*    x_ptr;
    void*    w_ptr;
    void*    o_ptr;
    int32_t* expert_list;
    int32_t* index_list;
    int32_t  list_size;
    int32_t  expert_size;
    int64_t  N;
    int64_t  K;
    float    scale_a;
    float    scale_b;
    const float* scale_a_ptr;   // [total_tokens] per-token scales, or nullptr
    const float* scale_b_ptr;   // [num_experts] per-expert scales, or nullptr
} SM89MoEFP8Params;

/*
 * Masked FP8 params for SM89 native FP8 MMA.
 * x_ptr : float8_e4m3fn  [num_groups, M_max, K]   padded per-group input
 * w_ptr : float8_e4m3fn  [num_groups, N, K]        weights
 * o_ptr : bfloat16       [num_groups, M_max, N]    padded per-group output
 * masked_m : int32       [num_groups]               valid row count per group
 *
 * Grid Y = num_groups (constant). Each CTA reads masked_m[blockIdx.y] to
 * bound its M-loop, skipping padding rows. CUDA-graph safe.
 */
typedef struct {
    void*    x_ptr;
    void*    w_ptr;
    void*    o_ptr;
    int32_t* masked_m;
    int32_t  num_groups;
    int64_t  M_max;
    int64_t  N;
    int64_t  K;
    float    scale_a;
    float    scale_b;
    const float* scale_a_ptr;   // [num_groups * M_max] per-token scales, or nullptr
    const float* scale_b_ptr;   // [num_groups] per-expert scales, or nullptr
} SM89MoEFP8MaskedParams;

#ifdef __cplusplus
}  // namespace asym_gemm
#endif
