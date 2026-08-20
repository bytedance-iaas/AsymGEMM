// asym_gemm/include/asym_gemm/impls/sm80_moe_params.h
// Plain C header — includable from both C++ host code and CUDA device code.
// FP8 (SM89) param structs live in sm89_fp8_moe_params.h.
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

#ifdef __cplusplus
}  // namespace asym_gemm
#endif
