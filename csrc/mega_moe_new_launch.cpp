// Host-side launcher TU for the UMMA two-phase fused MoE kernel.
//
// This file ONLY creates TMA descriptors and calls mega_moe_kernel_and_combine()
// (implemented in mega_moe_new_kernel.cu).  It must NOT include the kernel header
// or any header that includes asym_gemm/common/utils.cuh, because that conflicts
// with csrc/utils/math.hpp (both define asym_gemm::ceil_div in the same namespace).
//
// Sources: csrc/mega_moe_new_launch.cu + csrc/mega_moe_new_kernel.cu (added to setup.py)

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// torch/python.h must come before CuTe headers.
#include <torch/python.h>

// TMA descriptor creation helpers (includes utils/math.hpp but NOT utils.cuh).
#include "jit_kernels/impls/runtime_utils.hpp"

// Forward declaration: kernel launcher implemented in mega_moe_new_kernel.cu.
extern "C" int mega_moe_kernel_and_combine(
        const uint32_t*   d_offsets,
        const CUtensorMap tma_maps[9],
        void*             workspace_ptr,
        uint64_t off_grid_sync, uint64_t off_l1_arrival, uint64_t off_l2_mask,
        uint64_t off_l2_acts,   uint64_t off_l2_sf,      uint64_t off_token_src_map,
        uint64_t off_l1_topk_w, uint64_t off_combine,
        const int32_t* d_topk_map,
        const float*   d_row_topk_w,
        nv_bfloat16*   d_y,
        uint32_t M_total, uint32_t num_tokens, uint32_t num_topk,
        uint32_t H, uint32_t I, uint32_t E,
        float activation_clamp,
        int fast_math,
        cudaStream_t stream);

using namespace asym_gemm;

// ------- C-linkage entry point (called from python_api.cpp) -------------------
extern "C" int asym_mega_moe_new_launch(
        const void* a_fp8,      const void* a_sf,
        const void* l1_w,       const void* l1_w_sf,
        const void* l2_w,       const void* l2_w_sf,
        const void* offsets,    // [2*num_experts] uint32 device ptr (full, zero-padded)
        const void* topk_map,   // [M_total, 2] int32 device ptr
        const void* row_topk_w, // [M_total] float32 device ptr
        void*       workspace_ptr,
        uint64_t    workspace_bytes,
        uint64_t    off_grid_sync,
        uint64_t    off_l1_arrival,
        uint64_t    off_l2_mask,
        uint64_t    off_l2_acts,
        uint64_t    off_l2_sf,
        uint64_t    off_token_src_map,
        uint64_t    off_l1_topk_w,
        uint64_t    off_combine,
        void*       y,
        uint32_t M_total, uint32_t num_tokens, uint32_t num_topk,
        uint32_t hidden, uint32_t intermediate, uint32_t num_experts,
        float   activation_clamp,
        int     fast_math,
        cudaStream_t stream) {

    const int64_t E = (int64_t)num_experts;
    const int64_t H = (int64_t)hidden;
    const int64_t I = (int64_t)intermediate;
    const int64_t M = (int64_t)M_total;

    auto make_t = [&](const void* ptr, std::vector<int64_t> shape, at::ScalarType dt) {
        return torch::from_blob(const_cast<void*>(ptr), shape,
                                torch::dtype(dt).device(torch::kCUDA));
    };

    auto t_a_fp8  = make_t(a_fp8,   {M, H},                torch::kUInt8);
    auto t_a_sf   = make_t(a_sf,    {M, H / 128},          torch::kFloat);
    auto t_l1_w   = make_t(l1_w,    {E, 2 * I, H},         torch::kUInt8);
    auto t_l1_wsf = make_t(l1_w_sf, {E, 2 * I / 128, H / 128}, torch::kFloat);
    auto t_l2_w   = make_t(l2_w,    {E, H, I},             torch::kUInt8);
    auto t_l2_wsf = make_t(l2_w_sf, {E, H / 128, I / 128}, torch::kFloat);

    // Workspace tensor views (for TMA descriptors over device workspace).
    auto* wp = static_cast<uint8_t*>(workspace_ptr);
    auto l2_acts_t = torch::from_blob(wp + off_l2_acts, {M, I},
                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
    auto l2_sf_t   = torch::from_blob(wp + off_l2_sf,   {M, I / 128},
                         torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Fixed tile / swizzle constants (match mega_moe_new_kernel.cu).
    constexpr auto kMajorA       = cute::UMMA::Major::K;
    constexpr auto kMajorB       = cute::UMMA::Major::MN;
    constexpr auto kMajorMN      = cute::UMMA::Major::MN;
    constexpr int  BLOCK_M       = 128;
    constexpr int  BLOCK_N       = 64;
    constexpr int  BLOCK_K       = 128;
    constexpr int  kSwizzleA     = 128;
    constexpr int  kSwizzleB     = 128;
    constexpr int  SF_QUANT_K    = 128;
    constexpr int  SF_BLOCK_M    = 128;
    constexpr int  L1_OUT_N      = BLOCK_N / 2;    // = 32
    constexpr int  STORE_BLOCK_M = BLOCK_M;        // = 128

    // Build the 9 TMA descriptors in the order expected by the kernel:
    //   [0] tma_l1_a, [1] tma_l1_sfa, [2] tma_l1_b,   [3] tma_l1_sfb,
    //   [4] tma_l1_out (store),
    //   [5] tma_l2_a, [6] tma_l2_sfa, [7] tma_l2_b,   [8] tma_l2_sfb
    CUtensorMap tma_maps[9];

    // [0] L1 A: [M_pool, H] FP8 K-major.
    tma_maps[0] = make_tma_a_desc(kMajorA, t_a_fp8,
                                  (int)M, (int)H,
                                  BLOCK_M, BLOCK_K,
                                  (int)H, /*num_groups=*/1, kSwizzleA);

    // [1] L1 SFA: [M_pool, H/128] float32 MN-major.
    tma_maps[1] = make_tma_sf_desc(kMajorMN, t_a_sf,
                                   (int)M, (int)H,
                                   BLOCK_M, SF_QUANT_K,
                                   /*num_groups=*/1, /*swizzle=*/0);

    // [2] L1 B: [E, 2I, H] FP8 MN-major (NT weights).
    tma_maps[2] = make_tma_b_desc(kMajorB, t_l1_w,
                                  (int)(2 * I), (int)H,
                                  BLOCK_N, BLOCK_K,
                                  (int)t_l1_w.stride(-2), (int)E, kSwizzleB);

    // [3] L1 SFB: [E, 2I/128, H/128] float32.
    tma_maps[3] = make_tma_sf_desc(kMajorMN, t_l1_wsf,
                                   (int)(2 * I), (int)H,
                                   BLOCK_N, SF_QUANT_K,
                                   (int)E, /*swizzle=*/0);

    // [4] L1 out (TMA store): write FP8 intermediate to workspace.l2_acts [M, I].
    tma_maps[4] = make_tma_cd_desc(l2_acts_t,
                                   (int)M, (int)I,
                                   STORE_BLOCK_M, L1_OUT_N,
                                   (int)I, /*num_groups=*/1, /*swizzle=*/0);

    // [5] L2 A: read workspace.l2_acts [M, I] FP8 K-major.
    tma_maps[5] = make_tma_a_desc(kMajorA, l2_acts_t,
                                  (int)M, (int)I,
                                  BLOCK_M, BLOCK_K,
                                  (int)I, /*num_groups=*/1, kSwizzleA);

    // [6] L2 SFA: workspace.l2_sf [M, I/128] uint32 (UE8M0 packed).
    tma_maps[6] = make_tma_sf_desc(kMajorMN, l2_sf_t,
                                   (int)M, (int)I,
                                   SF_BLOCK_M, SF_QUANT_K * 4,
                                   /*num_groups=*/1, /*swizzle=*/0);

    // [7] L2 B: [E, H, I] FP8 MN-major (NT weights).
    tma_maps[7] = make_tma_b_desc(kMajorB, t_l2_w,
                                  (int)H, (int)I,
                                  BLOCK_N, BLOCK_K,
                                  (int)t_l2_w.stride(-2), (int)E, kSwizzleB);

    // [8] L2 SFB: [E, H/128, I/128] float32.
    tma_maps[8] = make_tma_sf_desc(kMajorMN, t_l2_wsf,
                                   (int)H, (int)I,
                                   BLOCK_N, SF_QUANT_K,
                                   (int)E, /*swizzle=*/0);

    // ---- Dispatch to kernel TU -----------------------------------------------
    return mega_moe_kernel_and_combine(
        static_cast<const uint32_t*>(offsets),
        tma_maps,
        workspace_ptr,
        off_grid_sync, off_l1_arrival, off_l2_mask,
        off_l2_acts,   off_l2_sf,      off_token_src_map,
        off_l1_topk_w, off_combine,
        static_cast<const int32_t*>(topk_map),
        static_cast<const float*>(row_topk_w),
        static_cast<nv_bfloat16*>(y),
        M_total, num_tokens, num_topk,
        hidden, intermediate, num_experts,
        activation_clamp, fast_math, stream);
}
