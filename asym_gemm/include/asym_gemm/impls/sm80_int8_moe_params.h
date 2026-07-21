// asym_gemm/include/asym_gemm/impls/sm80_int8_moe_params.h
// Plain C header — includable from both C++ host code and CUDA device code.
// Param structs for the SM80 INT8 asym MoE kernels (sm80_int8_asym_moe_gemm.cuh).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
namespace asym_gemm {
#endif

/*
 * INT8 asym grouped MoE params — SM80 (A100) mma.m16n8k32.s8, S32 accumulate.
 *
 * x_ptr : int8 in HBM                    [total_tokens, K]
 * w_ptr : int8 in CPU pinned (or HBM)    [num_experts, N, K]
 * o_ptr : float32 in HBM                 [total_tokens, N]
 *         partial-sum accumulation buffer between K-tiles AND final output
 *         (matches the SM90 1d1d kernel's FP32-D convention).
 *
 * Scales are the *natural* per-token / per-channel layout with one scale per
 * 128-element K-block (kb = K / 128) — exactly what the unified_moe runtime
 * produces; no K-major transpose:
 *   sfa_ptr : float32 [total_tokens, kb]
 *   sfb_ptr : float32 [num_experts, N, kb]
 *
 * BLOCK_K is locked to 128 (one scale k-group per K-tile), so within a K-tile
 * the S32 accumulation is exact and the single dequant
 *   o += float(acc_s32) * sfa[m, kb] * sfb[e, n, kb]
 * happens once per (tile, K-block) at the epilogue. K must be a multiple of 128.
 */
typedef struct {
    void*    x_ptr;
    void*    w_ptr;
    void*    o_ptr;
    // Segment scheme — SAME convention as the SM90 1d1d asym kernel
    // (asymScheduler.cuh): segment i covers absolute token rows
    // [index_list[2i], index_list[2i+1]) and uses expert_list[i] to index
    // W / sfb. expert_list carries a -1 terminator (and -1 marks gap
    // segments, skipped). Grid Y = num_segments = list_size - 1.
    int32_t* expert_list;   // [list_size] expert IDs, -1 terminated
    int32_t* index_list;    // [2 * (list_size - 1)] (start, end) pairs
    int32_t  list_size;
    int32_t  expert_size;   // num_experts (outer dim of W / sfb)
    int64_t  N;
    int64_t  K;
    const float* sfa_ptr;   // [total_tokens, kb]
    const float* sfb_ptr;   // [num_experts, N, kb]
    int32_t  kb;            // K / 128
} SM80MoEInt8Params;

/*
 * Masked INT8 asym MoE params — padded [G, M_max, ...] layout, CUDA-graph safe.
 * x_ptr : int8    [num_groups, M_max, K]
 * w_ptr : int8    [num_groups, N, K]
 * o_ptr : float32 [num_groups, M_max, N]
 * masked_m : int32 [num_groups]  valid row count per group
 * sfa_ptr : float32 [num_groups, M_max, kb]
 * sfb_ptr : float32 [num_groups, N, kb]
 *
 * Grid Y = num_groups (constant). Rows >= masked_m[g] are never read or
 * written (padding rows keep whatever garbage they held).
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
    const float* sfa_ptr;   // [num_groups, M_max, kb]
    const float* sfb_ptr;   // [num_groups, N, kb]
    int32_t  kb;            // K / 128
} SM80MoEInt8MaskedParams;

#ifdef __cplusplus
}  // namespace asym_gemm
#endif
