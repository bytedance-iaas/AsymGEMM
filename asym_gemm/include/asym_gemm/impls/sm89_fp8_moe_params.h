// asym_gemm/include/asym_gemm/impls/sm89_fp8_moe_params.h
// Plain C header — includable from both C++ host code and CUDA device code.
// Param structs for the SM89 FP8 MoE kernels (sm89_fp8_moe_gemm.cuh).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
namespace asym_gemm {
#endif

/*
 * FP8 params for SM89 native FP8 MMA (RTX 4090).
 * x_ptr : float8_e4m3fn in HBM          [total_tokens, K]
 * w_ptr : float8_e4m3fn in CPU pinned   [num_experts, N, K]
 * o_ptr : bfloat16 in HBM               [total_tokens, N]
 *         also used as partial-sum accumulation buffer between K-tiles
 * scale_a/b: per-tensor float32 scales applied to FP32 accumulator at the
 *            final K-tile only (intermediate writes store unscaled BF16).
 *
 * Block-scale mode (scale_a_blk_ptr != nullptr): 1x128 activation scales and
 * 128x128 weight scales. Each K-tile's contribution is scaled by its own
 * (token, k-group) x (expert, n-group, k-group) factor before accumulating;
 * intermediate BF16 partials hold the SCALED running sum, and the next K-tile
 * rescales the seed by the reciprocal of its own combined scale.
 * Requires BLOCK_K <= 128 with 128 % BLOCK_K == 0 (tile never straddles a
 * k-group) — enforced by select_sm80_fp8_config(block_scale=true).
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
    const float* scale_a_ptr;     // [total_tokens] per-token scales, or nullptr
    const float* scale_b_ptr;     // [num_experts] per-expert scales, or nullptr
    const float* scale_a_blk_ptr; // [total_tokens, ceil(K/128)] 1x128 scales, or nullptr
    const float* scale_b_blk_ptr; // [num_experts, ceil(N/128), ceil(K/128)], or nullptr
    int32_t  sa_kg;               // ceil(K/128): k-group count (row stride of scale_a_blk)
    int32_t  sb_ng;               // ceil(N/128): n-group count
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
    const float* scale_a_ptr;     // [num_groups * M_max] per-token scales, or nullptr
    const float* scale_b_ptr;     // [num_groups] per-expert scales, or nullptr
    const float* scale_a_blk_ptr; // [num_groups * M_max, ceil(K/128)] 1x128 scales, or nullptr
    const float* scale_b_blk_ptr; // [num_groups, ceil(N/128), ceil(K/128)], or nullptr
    int32_t  sa_kg;               // ceil(K/128)
    int32_t  sb_ng;               // ceil(N/128)
} SM89MoEFP8MaskedParams;

#ifdef __cplusplus
}  // namespace asym_gemm
#endif
