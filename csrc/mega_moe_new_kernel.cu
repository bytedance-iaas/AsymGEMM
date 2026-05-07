// Kernel-only TU for the UMMA two-phase fused MoE kernel.
// This file must NOT include runtime_utils.hpp or csrc/utils/math.hpp —
// those conflict with asym_gemm/common/utils.cuh in the same namespace.
// TMA descriptor creation lives in mega_moe_new_launch.cu instead.
//
// This TU:
//   1. Instantiates sm100_fp8_asym_gemm_mega_moe_impl for each test shape.
//   2. Exports extern "C" mega_moe_kernel_and_combine() for the host launcher.

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <asym_gemm/impls/sm100_fp8_asym_gemm_mega_moe.cuh>
#include <asym_gemm/impls/smxx_combine_reduce.cuh>
#include <asym_gemm/common/mega_moe_workspace.cuh>

using namespace asym_gemm;

// Fixed launch constants for all test shapes.
static constexpr auto  kMajorA             = cute::UMMA::Major::K;
static constexpr auto  kMajorB             = cute::UMMA::Major::MN;
static constexpr uint32_t BLOCK_M          = 128;
static constexpr uint32_t BLOCK_N          = 64;
static constexpr uint32_t BLOCK_K          = 128;
static constexpr uint32_t kNumStages       = 2;
static constexpr uint32_t kNumStagesB      = 3;
static constexpr uint32_t kNumNonEpilogueThreads = 128;
static constexpr uint32_t kNumEpilogueThreads    = 128;
static constexpr uint32_t kNumSMs          = 132;
static constexpr uint32_t kSwizzleAMode    = 128;
static constexpr uint32_t kSwizzleBMode    = 128;

// ------- SMEM size helper (host side, pure arithmetic) ------------------------
template <uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t BLOCK_K_,
          uint32_t L1_SHAPE_N, uint32_t kNumStages_, uint32_t kNumStagesB_,
          uint32_t kNumEpilogueThreads_>
__host__ static constexpr uint32_t compute_smem_bytes() {
    constexpr uint32_t LAYOUT_AD_M  = 128;
    constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M_ < LAYOUT_AD_M) ? BLOCK_M_ : LAYOUT_AD_M;
    constexpr uint32_t kNumMWaves   = BLOCK_M_ / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages  = 2;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kSFQuantK    = 128;
    constexpr uint32_t kNumSFPerPack = 4;
    constexpr uint32_t kSFAtomsPerBlockK  = BLOCK_K_ / kSFQuantK;
    constexpr uint32_t kBlockKPerSFLoad   = kNumSFPerPack / kSFAtomsPerBlockK;

    constexpr uint32_t L1_OUT_N   = BLOCK_N_ / 2;
    constexpr uint32_t STORE_BLOCK_M = WAVE_BLOCK_M;
    constexpr uint32_t SMEM_CD_L1_STAGE = STORE_BLOCK_M * L1_OUT_N;
    constexpr uint32_t SMEM_CD_L2       = STORE_BLOCK_M * BLOCK_N_ * 2u;
    constexpr uint32_t SMEM_CD_SIZE     = (SMEM_CD_L1_STAGE * kNumTMAStoreStages > SMEM_CD_L2)
                                          ? SMEM_CD_L1_STAGE * kNumTMAStoreStages : SMEM_CD_L2;
    constexpr uint32_t SMEM_CD_ALIGNED  = (SMEM_CD_SIZE + 1023u) & ~1023u;

    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M_ * BLOCK_K_;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N_ * BLOCK_K_;
    constexpr uint32_t SMEM_B_SIZE = kNumStagesB_ * SMEM_B_SIZE_PER_STAGE;

    constexpr uint32_t SF_BLOCK_M = ((BLOCK_M_ + kNumUTCCPAlignedElems - 1) / kNumUTCCPAlignedElems) * kNumUTCCPAlignedElems;
    constexpr uint32_t SF_BLOCK_N = ((BLOCK_N_ + kNumUTCCPAlignedElems - 1) / kNumUTCCPAlignedElems) * kNumUTCCPAlignedElems;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages =
        (2 * kNumMWaves * BLOCK_N_ + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    constexpr uint32_t ATOM_M = 8u;
    constexpr uint32_t kNumAtomsPerWave = STORE_BLOCK_M / ATOM_M;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads_ / 32;
    constexpr uint32_t SMEM_AMAX_SIZE = kNumEpilogueWarps * kNumAtomsPerWave * sizeof(float2);

    constexpr uint32_t kNumBarriers = kNumStages_ * 3 + kNumEpilogueStages * 2 + kNumStagesB_ * 3;
    constexpr uint32_t BARRIER_REGION = kNumBarriers * 8 + 4;

    return SMEM_CD_ALIGNED
         + kNumStages_  * SMEM_A_SIZE_PER_STAGE
         + SMEM_B_SIZE
         + kNumStages_  * SMEM_SFA_SIZE_PER_STAGE
         + kNumStagesB_ * SMEM_SFB_SIZE_PER_STAGE
         + SMEM_AMAX_SIZE
         + BARRIER_REGION;
}

// ------- Per-shape runner -----------------------------------------------------
template <uint32_t H, uint32_t I, uint32_t E, bool kFastMath>
static int run_shape(
        const uint32_t*    d_offsets,
        const CUtensorMap  tma[9],      // l1_a, l1_sfa, l1_b, l1_sfb, l1_out,
                                        // l2_a, l2_sfa, l2_b, l2_sfb
        void*              workspace_ptr,
        uint64_t off_grid_sync, uint64_t off_l1_arrival, uint64_t off_l2_mask,
        uint64_t off_l2_acts,   uint64_t off_l2_sf,      uint64_t off_token_src_map,
        uint64_t off_l1_topk_w, uint64_t off_combine,
        const int32_t*     d_topk_map,
        const float*       d_row_topk_w,
        nv_bfloat16*       d_y,
        uint32_t M_total, uint32_t num_tokens, uint32_t num_topk,
        float activation_clamp,
        cudaStream_t stream) {

    constexpr uint32_t smem_bytes = compute_smem_bytes<
        BLOCK_M, BLOCK_N, BLOCK_K, 2 * I,
        kNumStages, kNumStagesB, kNumEpilogueThreads>();

    // ---- Fill workspace fields -----------------------------------------------
    cudaMemcpyAsync(static_cast<uint8_t*>(workspace_ptr) + off_token_src_map,
                    d_topk_map,   M_total * 2 * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(static_cast<uint8_t*>(workspace_ptr) + off_l1_topk_w,
                    d_row_topk_w, M_total * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    const uint32_t num_pool_blocks = (M_total + BLOCK_M - 1) / BLOCK_M;
    cudaMemsetAsync(static_cast<uint8_t*>(workspace_ptr) + off_l2_mask,
                    0, num_pool_blocks * sizeof(uint64_t), stream);

    // ---- Build workspace struct ----------------------------------------------
    MegaMoEWorkspace kws;
    kws.base                = workspace_ptr;
    kws.num_experts         = E;
    kws.num_topk            = num_topk;
    kws.num_tokens          = num_tokens;
    kws.num_max_pool_tokens = M_total;
    kws.num_max_pool_blocks = num_pool_blocks;
    kws.hidden              = H;
    kws.intermediate_hidden = I;
    kws.off_grid_sync       = off_grid_sync;
    kws.off_l1_arrival      = off_l1_arrival;
    kws.off_l2_mask         = off_l2_mask;
    kws.off_l2_acts         = off_l2_acts;
    kws.off_l2_sf           = off_l2_sf;
    kws.off_token_src_map   = off_token_src_map;
    kws.off_l1_topk_w       = off_l1_topk_w;
    kws.off_combine         = off_combine;
    kws.activation_clamp    = activation_clamp;

    // ---- Kernel function pointer --------------------------------------------
    const void* kernel_fn = reinterpret_cast<const void*>(
        &sm100_fp8_asym_gemm_mega_moe_impl<
            kMajorA, kMajorB,
            BLOCK_M, BLOCK_N, BLOCK_K,
            E,
            2 * I, H,  // L1_SHAPE_N, L1_SHAPE_K
            H, I,      // L2_SHAPE_N, L2_SHAPE_K
            kSwizzleAMode, kSwizzleBMode,
            kNumStages,
            kNumNonEpilogueThreads, kNumEpilogueThreads,
            kNumSMs,
            kNumStagesB,
            kFastMath>);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    // ---- Launch --------------------------------------------------------------
    // Arg order matches kernel signature:
    //   offsets, tma_l1_a, tma_l1_sfa, tma_l1_b, tma_l1_sfb, tma_l1_out,
    //   tma_l2_a, tma_l2_sfa, tma_l2_b, tma_l2_sfb, workspace
    void* args[] = {
        (void*)&d_offsets,
        (void*)&tma[0], (void*)&tma[1], (void*)&tma[2], (void*)&tma[3],
        (void*)&tma[4],
        (void*)&tma[5], (void*)&tma[6], (void*)&tma[7], (void*)&tma[8],
        (void*)&kws
    };
    dim3 grid(kNumSMs, 1, 1);
    dim3 block(kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1);
    cudaError_t err = cudaLaunchKernel(kernel_fn, grid, block, args, smem_bytes, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[mega_moe] cudaLaunchKernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[mega_moe] kernel error: %s\n", cudaGetErrorString(sync_err));
        return -3;
    }

    // ---- Combine-reduce ------------------------------------------------------
    const void* combine_buf = static_cast<uint8_t*>(workspace_ptr) + off_combine;
    launch_combine_reduce(combine_buf, d_y, num_tokens, num_topk, H, stream);
    cudaError_t cr_err = cudaStreamSynchronize(stream);
    if (cr_err != cudaSuccess) {
        fprintf(stderr, "[mega_moe] combine_reduce error: %s\n", cudaGetErrorString(cr_err));
        return -4;
    }
    return 0;
}

// ------- C-linkage entry point (called from mega_moe_new_launch.cu) -----------
extern "C" int mega_moe_kernel_and_combine(
        const uint32_t*   d_offsets,
        const CUtensorMap tma_maps[9],  // 9 TMA descriptors (see run_shape above)
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
        cudaStream_t stream) {

#define TRY_HIE(Hv, Iv, Ev) \
    if (H == (Hv) && I == (Iv) && E == (Ev)) { \
        if (fast_math) \
            return run_shape<(Hv), (Iv), (Ev), true>( \
                d_offsets, tma_maps, workspace_ptr, \
                off_grid_sync, off_l1_arrival, off_l2_mask, \
                off_l2_acts, off_l2_sf, off_token_src_map, off_l1_topk_w, off_combine, \
                d_topk_map, d_row_topk_w, d_y, \
                M_total, num_tokens, num_topk, activation_clamp, stream); \
        return run_shape<(Hv), (Iv), (Ev), false>( \
            d_offsets, tma_maps, workspace_ptr, \
            off_grid_sync, off_l1_arrival, off_l2_mask, \
            off_l2_acts, off_l2_sf, off_token_src_map, off_l1_topk_w, off_combine, \
            d_topk_map, d_row_topk_w, d_y, \
            M_total, num_tokens, num_topk, activation_clamp, stream); \
    }

    TRY_HIE(256,  128, 4)
    TRY_HIE(512,  256, 4)
    TRY_HIE(512,  256, 8)
    TRY_HIE(1024, 512, 8)
    TRY_HIE(1024, 512, 16)
#undef TRY_HIE

    fprintf(stderr,
            "[mega_moe] Unsupported (H=%u, I=%u, E=%u). "
            "Add to dispatch table in csrc/mega_moe_new_kernel.cu.\n",
            H, I, E);
    return -2;
}
