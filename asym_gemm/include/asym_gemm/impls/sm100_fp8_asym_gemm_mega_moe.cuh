#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <asym_gemm/common/epilogue_utils.cuh>
#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm100_utils.cuh>
#include <asym_gemm/common/mega_moe_scheduler.cuh>
#include <asym_gemm/common/mega_moe_epilogue.cuh>
#include <asym_gemm/common/mega_moe_workspace.cuh>
#include <asym_gemm/common/mega_moe_grid_sync.cuh>
#include <asym_gemm/common/mega_moe_sf_math.cuh>
#include <asym_gemm/common/mega_moe_epilogue_swiglu.cuh>

namespace asym_gemm {

using namespace asym_gemm::sm100;

// Fused mega MoE kernel: L1 GEMM → SwiGLU+FP8 requant → L2 GEMM → BF16 combine-scatter.
//
// CPU-resident expert weights: B lives in CPU DRAM accessed via NVLink-C2C.
// The dedicated B-loader warp (warp 0) + kNumStagesB ring buffer hide this latency.
//
// Warp layout (kNumNonEpilogueThreads = 128, i.e. 4 warps × 32):
//   warp 0: B-loader  — outer k-loop per block, phase-aware TMA descriptor
//   warp 1: A-loader  — inner k-loop per block; spins on l2_arrival_mask for L2
//   warp 2: MMA warp  (leader CTA only)
//   warp 3: UTCCP transposer
//   warps ≥ 4: epilogue (L1: SwiGLU+FP8; L2: BF16 scatter to combine_buffer)
template <
    cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumExperts,
    uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
    uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
    uint32_t kNumStages,
    uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs,
    uint32_t kNumStagesB = 3,
    bool kFastMath = true>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_asym_gemm_mega_moe_impl(
    const uint32_t* offsets,              // [2*kNumExperts] expert token start/end
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_sfa,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_sfb,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_out,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_sfa,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_sfb,
    MegaMoEWorkspace workspace)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;

    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128,
                     "mega MoE requires exactly 4 non-epilogue warps");
    DG_STATIC_ASSERT(kNumStagesB >= 2,
                     "kNumStagesB must be >= 2 for CPU-memory latency hiding");
    DG_STATIC_ASSERT(BLOCK_K % 128 == 0,
                     "BLOCK_K must be a multiple of 128 for FP8");
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "L1_SHAPE_N not divisible by BLOCK_N");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "L2_SHAPE_N not divisible by BLOCK_N");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "L1_SHAPE_K not divisible by BLOCK_K");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "L2_SHAPE_K not divisible by BLOCK_K");
    DG_STATIC_ASSERT(BLOCK_N % 2 == 0, "BLOCK_N must be even for gate/up split");

    // Geometry constants
    constexpr uint32_t LAYOUT_AD_M         = 128;
    constexpr uint32_t WAVE_BLOCK_M        = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves          = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages  = 2;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kSFQuantK           = 128;
    constexpr uint32_t kSFAtomsPerBlockK   = BLOCK_K / kSFQuantK;
    constexpr uint32_t kNumSFPerPack       = sizeof(uint32_t) / sizeof(cutlass::float_ue8m0_t);
    constexpr uint32_t kBlockKPerSFLoad    = kNumSFPerPack / kSFAtomsPerBlockK;
    constexpr uint32_t L1_OUT_N            = BLOCK_N / 2;     // output after SwiGLU
    constexpr uint32_t STORE_BLOCK_M       = WAVE_BLOCK_M;    // rows per store pass
    constexpr uint32_t STORE_BLOCK_N_BF16  = 64u;             // BF16: 128B swizzle / 2 bytes
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;

    DG_STATIC_ASSERT(kNumSFPerPack % kSFAtomsPerBlockK == 0,
                     "SFs per pack must be divisible by SFs per block_k");
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0,
                     "Invalid block M");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Shared memory sizes
    // CD region: max of L1 FP8 staging and L2 BF16 staging
    constexpr uint32_t SMEM_CD_L1_STAGE   = STORE_BLOCK_M * L1_OUT_N;        // FP8, 1 byte/elem
    constexpr uint32_t SMEM_CD_L2         = STORE_BLOCK_M * BLOCK_N * 2u;    // BF16, 2 bytes/elem
    constexpr uint32_t SMEM_CD_SIZE       = cute::max(SMEM_CD_L1_STAGE * kNumTMAStoreStages, SMEM_CD_L2);
    // Round up to 1024 for swizzle alignment
    constexpr uint32_t SMEM_CD_ALIGNED    = (SMEM_CD_SIZE + 1023u) & ~1023u;

    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K; // float_e4m3 = 1 byte
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K;
    constexpr uint32_t SMEM_B_SIZE        = kNumStagesB * SMEM_B_SIZE_PER_STAGE;
    constexpr uint32_t SF_BLOCK_M         = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N         = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE =
        constexpr_align(BLOCK_M, LAYOUT_AD_M) * BLOCK_K;
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE
                     <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStagesB,
                     "Memory Out of bound for UMMA");

    // Epilogue staging
    constexpr uint32_t kNumSFATmemCols    = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols    = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages =
        (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
    constexpr uint32_t kNumAccumTmemCols  = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols       =
        get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512,
                     "Invalid tensor memory columns");

    // Cross-warp amax reduction scratchpad for L1 epilogue
    constexpr uint32_t kNumEpilogueWarps  = kNumEpilogueThreads / 32;
    constexpr uint32_t ATOM_M             = 8u;
    constexpr uint32_t kNumAtomsPerWave   = STORE_BLOCK_M / ATOM_M;
    constexpr uint32_t SMEM_AMAX_SIZE     = kNumEpilogueWarps * kNumAtomsPerWave * sizeof(float2);

    // Warp / thread IDs
    bool is_leader_cta     = cute::block_rank_in_cluster() == 0;
    const auto warp_idx    = cutlass::canonical_warp_idx_sync();
    const auto lane_idx    = get_lane_idx();
    const uint32_t sm_idx  = blockIdx.x;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // --- Shared memory layout -------------------------------------------------
    // [0, SMEM_CD_ALIGNED)                   smem_cd / smem_cd_l2
    // [+kNumStages * SMEM_A_SIZE_PER_STAGE)  smem_a stages
    // [+kNumStagesB * SMEM_B_SIZE_PER_STAGE) smem_b stages
    // [+kNumStages * SMEM_SFA_SIZE_PER_STAGE) smem_sfa stages
    // [+kNumStagesB * SMEM_SFB_SIZE_PER_STAGE) smem_sfb stages
    // [+SMEM_AMAX_SIZE)                      smem_amax_reduction
    // [barrier region]                       barriers (8 bytes each)
    // --------------------------------------------------------------------------

    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + i * SMEM_CD_L1_STAGE;
    });
    auto smem_cd_l2 = smem_buffer;

    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(
            smem_buffer + SMEM_CD_ALIGNED + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(
            smem_buffer + SMEM_CD_ALIGNED
            + kNumStages * SMEM_A_SIZE_PER_STAGE
            + i * SMEM_B_SIZE_PER_STAGE);
    });

    const auto sf_base = smem_buffer + SMEM_CD_ALIGNED
                       + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE;
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_base + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(
            sf_base + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto smem_amax_reduction = reinterpret_cast<float2*>(
        sf_base + kNumStages * SMEM_SFA_SIZE_PER_STAGE
                + kNumStagesB * SMEM_SFB_SIZE_PER_STAGE);

    auto barrier_base = reinterpret_cast<Barrier*>(
        reinterpret_cast<uint8_t*>(smem_amax_reduction) + SMEM_AMAX_SIZE);

    auto full_barriers           = PatternVisitor([=](const uint32_t& i) { return barrier_base + i; });
    auto empty_barriers          = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages + i; });
    auto with_sf_full_barriers   = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 2 + i; });
    auto tmem_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 3 + i; });
    auto tmem_empty_barriers     = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 3 + kNumEpilogueStages + i; });
    auto with_sf_full_barriers_b = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 3 + kNumEpilogueStages * 2 + i; });
    auto full_barriers_b         = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 3 + kNumEpilogueStages * 2 + kNumStagesB + i; });
    auto empty_barriers_b        = PatternVisitor([=](const uint32_t& i) { return barrier_base + kNumStages * 3 + kNumEpilogueStages * 2 + kNumStagesB * 2 + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_base + kNumStages * 3 + kNumEpilogueStages * 2 + kNumStagesB * 3);

    // Initialize barriers (warp 1) and allocate TMEM (warp 2)
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStagesB; ++i) {
            with_sf_full_barriers_b[i]->init(32);
            full_barriers_b[i]->init(1);
            empty_barriers_b[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_a);
        cute::prefetch_tma_descriptor(&tensor_map_l1_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_l1_b);
        cute::prefetch_tma_descriptor(&tensor_map_l1_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_l1_out);
        cute::prefetch_tma_descriptor(&tensor_map_l2_a);
        cute::prefetch_tma_descriptor(&tensor_map_l2_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_l2_b);
        cute::prefetch_tma_descriptor(&tensor_map_l2_sfb);
    }

    // Two-phase scheduler (L1 then L2, round-robin across SMs)
    auto scheduler = MegaMoEAsymScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExperts, kNumSMs>(offsets);

    // Scheduler-derived constants used in L2 A-loader spin
    using SchedT = MegaMoEAsymScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExperts, kNumSMs>;
    constexpr uint32_t kNumL1BlockNs = SchedT::kNumL1BlockNs;
    constexpr uint64_t kL1AllBitsMask = (1ull << kNumL1BlockNs) - 1ull;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "l2_arrival_mask has only 64 bits");

    // =========================================================================
    // Warp 0: B-loader — per-block k-loop, phase-aware descriptor select.
    // =========================================================================
    if (warp_idx == 0 and cute::elect_one_sync()) {
        uint32_t b_empty_phase[kNumStagesB] = {};
        uint32_t b_load_stage = 0;

        scheduler.for_each_block([&](BlockPhase phase,
                                      uint32_t expert_idx,
                                      uint32_t num_k_blocks,
                                      uint32_t /*m_block_idx*/,
                                      uint32_t n_block_idx) {
            const auto* tm_b   = (phase == BlockPhase::Linear2) ? &tensor_map_l2_b   : &tensor_map_l1_b;
            const auto* tm_sfb = (phase == BlockPhase::Linear2) ? &tensor_map_l2_sfb : &tensor_map_l1_sfb;
            const uint32_t shape_n   = (phase == BlockPhase::Linear2) ? L2_SHAPE_N : L1_SHAPE_N;
            const uint32_t shape_k   = (phase == BlockPhase::Linear2) ? L2_SHAPE_K : L1_SHAPE_K;
            const uint32_t sfb_k_stride = ceil_div(shape_k, kSFQuantK * kNumSFPerPack);
            const uint32_t n_idx = expert_idx * shape_n + n_block_idx * BLOCK_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                empty_barriers_b[b_load_stage]->wait(b_empty_phase[b_load_stage] ^ 1);
                b_empty_phase[b_load_stage] ^= 1;

                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sf_k_idx = ceil_div(k_idx, kSFQuantK * kNumSFPerPack);
                const uint32_t sf_grp = k_block_idx % kBlockKPerSFLoad;

                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::float_e4m3_t>(
                        tm_b, full_barriers_b[b_load_stage], smem_b[b_load_stage],
                        k_idx, n_idx, 1, 0);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma_copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::float_e4m3_t>(
                        tm_b, full_barriers_b[b_load_stage], smem_b[b_load_stage],
                        n_idx, k_idx, 1, 0);

                if (sf_grp == 0) {
                    tma_copy<BLOCK_N, 1, 0>(tm_sfb, full_barriers_b[b_load_stage],
                                            smem_sfb[b_load_stage],
                                            n_block_idx * BLOCK_N,
                                            expert_idx * sfb_k_stride + sf_k_idx);
                    full_barriers_b[b_load_stage]->arrive_and_expect_tx(
                        SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t));
                } else {
                    full_barriers_b[b_load_stage]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                b_load_stage = (b_load_stage + 1) % kNumStagesB;
            }
        });

    // =========================================================================
    // Warp 1: A-loader — per-block k-loop, spins on l2_arrival_mask for L2.
    // =========================================================================
    } else if (warp_idx == 1 and cute::elect_one_sync()) {
        uint32_t local_stage_idx = 0, local_phase = 0;
        auto local_advance = [&]() {
            local_stage_idx = (local_stage_idx + 1 == kNumStages) ? 0 : local_stage_idx + 1;
            local_phase ^= (local_stage_idx == 0);
        };

        scheduler.for_each_block([&](BlockPhase phase,
                                      uint32_t /*expert_idx*/,
                                      uint32_t num_k_blocks,
                                      uint32_t /*m_block_idx*/,
                                      uint32_t /*n_block_idx*/) {
            const auto* tm_a   = (phase == BlockPhase::Linear2) ? &tensor_map_l2_a   : &tensor_map_l1_a;
            const auto* tm_sfa = (phase == BlockPhase::Linear2) ? &tensor_map_l2_sfa : &tensor_map_l1_sfa;
            const uint32_t shape_k     = (phase == BlockPhase::Linear2) ? L2_SHAPE_K : L1_SHAPE_K;
            const uint32_t sfa_k_stride = ceil_div(shape_k, kSFQuantK * kNumSFPerPack);
            const uint32_t pool_block  = scheduler.get_current_pool_block_offset();

            // L2: spin until all L1 n-blocks for this pool block are done
            if (phase == BlockPhase::Linear2) {
                const uint64_t* mask_ptr = workspace.get_l2_arrival_mask_ptr(pool_block);
                uint64_t cur;
                do {
                    cur = ptx::ld_acq_gpu(reinterpret_cast<const uint64_t*>(mask_ptr));
                } while ((cur & kL1AllBitsMask) != kL1AllBitsMask);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                empty_barriers[local_stage_idx]->wait(local_phase ^ 1);

                // Both L1 and L2 use pool_block * BLOCK_M as the row base.
                // L1 A is in pool (flat) layout: expert e's rows start at
                // per_expert_m_block_start[e] * BLOCK_M = pool_block * BLOCK_M
                // (true when alignment == BLOCK_M, which the Python dispatch ensures).
                const uint32_t m_base = pool_block * BLOCK_M;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t sf_k_idx = ceil_div(k_idx, kSFQuantK * kNumSFPerPack);
                const uint32_t sf_grp   = k_block_idx % kBlockKPerSFLoad;
                const uint32_t sfa_m_base = pool_block * SF_BLOCK_M;

                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::float_e4m3_t>(
                        tm_a, full_barriers[local_stage_idx], smem_a[local_stage_idx],
                        k_idx, m_base, 1, 0);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma_copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::float_e4m3_t>(
                        tm_a, full_barriers[local_stage_idx], smem_a[local_stage_idx],
                        m_base, k_idx, 1, 0);

                if (sf_grp == 0) {
                    tma_copy<BLOCK_M, 1, 0>(tm_sfa, full_barriers[local_stage_idx],
                                            smem_sfa[local_stage_idx],
                                            sfa_m_base, sf_k_idx);
                    full_barriers[local_stage_idx]->arrive_and_expect_tx(
                        SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(uint32_t));
                } else {
                    full_barriers[local_stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                }
                local_advance();
            }
        });

    // =========================================================================
    // Warp 2: MMA issue warp (leader CTA only)
    // =========================================================================
    } else if (warp_idx == 2 and is_leader_cta) {
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            cutlass::float_e4m3_t, cutlass::float_e4m3_t,
            float, cutlass::float_ue8m0_t,
            UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages  <= 32, "Too many A stages");
        DG_STATIC_ASSERT(kNumStagesB <= 32, "Too many B stages");

        constexpr uint32_t SWIZZLE_ATOM_K_B = kSwizzleBMode / sizeof(cutlass::float_e4m3_t);
        constexpr uint32_t DESC_ATOM_K       = (BLOCK_K > SWIZZLE_ATOM_K_B) ? SWIZZLE_ATOM_K_B : BLOCK_K;
        constexpr uint32_t NUM_K_ATOMS       = BLOCK_K / DESC_ATOM_K;
        constexpr uint32_t UMMA_ITERS_PER_ATOM = DESC_ATOM_K / UMMA_K;

        auto a_desc = make_umma_desc<kMajorA, BLOCK_M, DESC_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, BLOCK_N, DESC_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages  ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStagesB ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;

        auto umma_arrive = [](const uint64_t* barrier) {
            cutlass::arch::umma_arrive(barrier);
        };

        uint32_t b_wsf_phase[kNumStagesB] = {};
        uint32_t b_mma_stage = 0;
        uint32_t local_stage_idx = 0, local_phase = 0;
        uint32_t accum_stage_idx = 0, accum_phase_idx = 0;

        auto local_advance = [&]() {
            local_stage_idx = (local_stage_idx + 1 == kNumStages) ? 0 : local_stage_idx + 1;
            local_phase ^= (local_stage_idx == 0);
        };
        auto accum_advance = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= (accum_stage_idx == 0);
        };

        scheduler.for_each_block([&](BlockPhase /*phase*/,
                                      uint32_t /*expert_idx*/,
                                      uint32_t num_k_blocks,
                                      uint32_t /*m_block_idx*/,
                                      uint32_t /*n_block_idx*/) {
            // Wait for accumulator slot to be released by epilogue
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            tcgen05_after_thread_sync();

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                with_sf_full_barriers_b[b_mma_stage]->wait(b_wsf_phase[b_mma_stage]);
                tcgen05_after_thread_sync();
                b_wsf_phase[b_mma_stage] ^= 1;

                const uint32_t sf_grp = k_block_idx % kBlockKPerSFLoad;
                if (sf_grp == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
                        auto smem_ptr = smem_sfb[b_mma_stage] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }

                with_sf_full_barriers[local_stage_idx]->wait(local_phase);
                tcgen05_after_thread_sync();

                if (sf_grp == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
                        auto smem_ptr = smem_sfa[local_stage_idx] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                }
                __syncwarp();

                using mma_t = SM100_MMA_MXF8F6F4_SS;
                const auto a_base = __shfl_sync(0xffffffff, a_desc_lo,
                                                 static_cast<int>(local_stage_idx));
                const auto b_base = __shfl_sync(0xffffffff, b_desc_lo,
                                                 static_cast<int>(b_mma_stage));

                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t atom = 0; atom < NUM_K_ATOMS; ++atom) {
                        const uint32_t sf_id =
                            (k_block_idx * kSFAtomsPerBlockK + atom) % kNumSFPerPack;
                        const auto rt_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sf_id);
                        const uint32_t b_off = atom * BLOCK_N * DESC_ATOM_K;
                        const uint32_t a_off = atom * BLOCK_M * DESC_ATOM_K;
                        const uint32_t b_atom = b_base + (b_off >> 4);
                        const uint32_t a_atom = a_base + (a_off >> 4);
                        #pragma unroll
                        for (uint32_t ki = 0; ki < UMMA_ITERS_PER_ATOM; ++ki) {
                            b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N,
                                kSwizzleBMode, cutlass::float_e4m3_t>(b_atom, 0, ki * UMMA_K);
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M,
                                    kSwizzleAMode, cutlass::float_e4m3_t>(
                                    a_atom, w * WAVE_BLOCK_M * DESC_ATOM_K, ki * UMMA_K);
                                mma_t::fma(a_desc, b_desc,
                                           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                           (atom > 0 || ki > 0),
                                           rt_desc,
                                           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                                           kTmemStartColOfSFB);
                            }
                        }
                    }
                }

                // Release A slot
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[local_stage_idx]));
                local_advance();

                // After last k-block: signal epilogue that TMEM is ready
                if (k_block_idx == num_k_blocks - 1) {
                    umma_arrive(reinterpret_cast<uint64_t*>(
                        tmem_full_barriers[accum_stage_idx]));
                }

                // Release B slot
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_b[b_mma_stage]));
                b_mma_stage = (b_mma_stage + 1) % kNumStagesB;
            }

            accum_advance();
        });

    // =========================================================================
    // Warp 3: UTCCP transposer
    // =========================================================================
    } else if (warp_idx == 3) {
        auto smem_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t v[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                v[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), v[i]);
        };

        uint32_t b_full_phase[kNumStagesB] = {};
        uint32_t b_trans_stage = 0;
        uint32_t local_stage_idx = 0, local_phase = 0;
        auto local_advance = [&]() {
            local_stage_idx = (local_stage_idx + 1 == kNumStages) ? 0 : local_stage_idx + 1;
            local_phase ^= (local_stage_idx == 0);
        };

        scheduler.for_each_block([&](BlockPhase /*phase*/,
                                      uint32_t /*expert_idx*/,
                                      uint32_t num_k_blocks,
                                      uint32_t /*m_block_idx*/,
                                      uint32_t /*n_block_idx*/) {
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                full_barriers_b[b_trans_stage]->wait(b_full_phase[b_trans_stage]);
                b_full_phase[b_trans_stage] ^= 1;

                const uint32_t sf_grp = k_block_idx % kBlockKPerSFLoad;
                if (sf_grp == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i)
                        smem_transpose(smem_sfb[b_trans_stage] + i * kNumUTCCPAlignedElems);
                    cutlass::arch::fence_view_async_shared();
                }
                with_sf_full_barriers_b[b_trans_stage]->arrive(0u);

                full_barriers[local_stage_idx]->wait(local_phase);
                if (sf_grp == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i)
                        smem_transpose(smem_sfa[local_stage_idx] + i * kNumUTCCPAlignedElems);
                    cutlass::arch::fence_view_async_shared();
                }
                with_sf_full_barriers[local_stage_idx]->arrive(0u);
                local_advance();

                b_trans_stage = (b_trans_stage + 1) % kNumStagesB;
            }
        });

    // =========================================================================
    // Epilogue warps: branch on phase
    //   L1 → SwiGLU + FP8 + TMA to l2_acts + atomicOr l2_arrival_mask
    //   L2 → BF16 scatter to combine_buffer[topk_k][orig_token][n]
    // =========================================================================
    } else if (warp_idx >= kNumNonEpilogueThreads / 32
               and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const uint32_t epi_warp = warp_idx - (kNumNonEpilogueThreads / 32);

        DG_STATIC_ASSERT(kNumEpilogueWarps >= 2 and kNumEpilogueWarps % 2 == 0,
                         "Need an even number of epilogue warps for amax reduction");
        DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Need 128 epilogue threads");

        constexpr uint32_t kNBG = 16u;  // bank group bytes
        constexpr uint32_t kEpBF16 = kNBG / 2u;  // BF16 elems per bank group
        constexpr uint32_t kNStoresN = BLOCK_N / STORE_BLOCK_N_BF16;

        uint32_t tma_stage = 0;
        uint32_t accum_stage_idx = 0, accum_phase_idx = 0;

        auto accum_advance = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= (accum_stage_idx == 0);
        };

        // ---- L1 epilogue: SwiGLU + FP8 requant ----------------------------
        // Processes BLOCK_N FP32 accumulator values per row.
        // Gate: columns [0, L1_OUT_N); Up: columns [L1_OUT_N, BLOCK_N).
        // After SwiGLU, produces L1_OUT_N FP8 values per row.
        //
        // NOTE: The exact gate/up interleave from SM100_TMEM_LOAD_32dp32b4x
        // depends on the UMMA accumulator layout and may need adjustment
        // based on hardware testing.
        auto l1_epilogue = [&](uint32_t valid_m,
                                uint32_t pool_block,
                                uint32_t /*m_block_idx*/,
                                uint32_t n_block_idx) {
            // Number of columns each epilogue warp processes in the gate/up halves
            constexpr uint32_t kColsPerWarp = L1_OUT_N / kNumEpilogueWarps;
            constexpr uint32_t kGateColBase = 0u;
            constexpr uint32_t kUpColBase   = L1_OUT_N;  // second half of BLOCK_N

            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                for (uint32_t i = 0; i < kNumAtomsPerWave; ++i) {
                    const uint32_t base_addr = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N;

                    // Load gate (4 FP32 per lane) and up (4 FP32 per lane)
                    uint32_t g[4], u[4];
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                        base_addr + i * ATOM_M + kGateColBase + epi_warp * kColsPerWarp,
                        g[0], g[1], g[2], g[3]);
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                        base_addr + i * ATOM_M + kUpColBase + epi_warp * kColsPerWarp,
                        u[0], u[1], u[2], u[3]);
                    cutlass::arch::fence_view_async_tmem_load();

                    // Row in the pool block for this lane
                    const uint32_t row = w * STORE_BLOCK_M + i * ATOM_M + (lane_idx % 4) * 2;
                    float weight = 1.0f;
                    if (row < valid_m)
                        weight = *workspace.get_l1_topk_weight_ptr(pool_block * BLOCK_M + row);

                    // Pack gate/up into 8-value layout and apply SwiGLU
                    // Layout: gate=(v[0],v[1]), up=(v[2],v[3]) for k=0
                    //         gate=(v[4],v[5]), up=(v[6],v[7]) for k=1
                    uint32_t vals8[8] = {g[0], g[1], u[0], u[1], g[2], g[3], u[2], u[3]};
                    float2 swiglu[2];
                    float2 amax;
                    mega_moe_epi::swiglu_atom_compute(vals8, weight, workspace.activation_clamp, kFastMath, swiglu, amax);

                    // Cross-warp amax reduction: all kNumEpilogueWarps must agree on the
                    // same SF for the L2 K-block (kSFQuantK=128 elements, 4 L1 n_blocks each).
                    // NamedBarrier ensures cross-warp smem visibility before reads.
                    if (lane_idx < 4)
                        smem_amax_reduction[epi_warp * kNumAtomsPerWave + i] = amax;
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Take max over all epilogue warps (all cover the same n_block K-group).
                    if (lane_idx < 4) {
                        #pragma unroll
                        for (uint32_t ew = 0; ew < kNumEpilogueWarps; ++ew) {
                            const float2 w_amax = smem_amax_reduction[ew * kNumAtomsPerWave + i];
                            amax.x = fmaxf(amax.x, w_amax.x);
                            amax.y = fmaxf(amax.y, w_amax.y);
                        }
                    }
                    // Broadcast to all lanes
                    amax.x = __shfl_sync(0xffffffff, amax.x, lane_idx % 4);
                    amax.y = __shfl_sync(0xffffffff, amax.y, lane_idx % 4);

                    // Pack to FP8 + compute UE8M0 SF
                    uint8_t sf_x, sf_y;
                    const uint32_t packed_fp8 =
                        mega_moe_epi::swiglu_atom_finalize_fp8(swiglu, amax, sf_x, sf_y);

                    // Write FP8 to smem_cd[tma_stage]
                    // Layout: [STORE_BLOCK_M, L1_OUT_N] row-major FP8
                    const uint32_t fp8_row = w * STORE_BLOCK_M + i * ATOM_M + lane_idx % ATOM_M;
                    const uint32_t fp8_col = epi_warp * kColsPerWarp;
                    if (fp8_row < STORE_BLOCK_M and fp8_col + 4 <= L1_OUT_N) {
                        *reinterpret_cast<uint32_t*>(
                            smem_cd[tma_stage] + fp8_row * L1_OUT_N + fp8_col) = packed_fp8;
                    }

                    // Write UE8M0 SF to l2_sf_buffer.
                    // SF granularity = kSFQuantK = 128 K-elements = one L2 K-block.
                    // 4 L1 n_blocks (32 K-elements each) map to one L2 K-block.
                    // k_block = n_block_idx / 4; SF byte = k_block % 4;
                    // uint32 col = k_block / 4; row stride = L2_SHAPE_K / 32 bytes.
                    // All 4 epilogue warps write the same SF byte (last write wins).
                    // NOTE: FP8 values are quantized per-sub-group (not per-K-block),
                    // so the L2 GEMM dequantization may have error for sub-groups whose
                    // local SF differs from the K-block SF.  A two-pass approach is
                    // needed for exact per-K-block quantization.
                    if (lane_idx < 4) {
                        const uint32_t tok = w * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2;
                        const uint32_t sf_row = mega_moe_transform_sf_token_idx(
                            tok, BLOCK_M, SF_BLOCK_M);
                        const uint32_t k_block = n_block_idx / 4;
                        const uint32_t k_col   = k_block / 4;
                        const uint32_t k_byte  = k_block % 4;
                        constexpr uint32_t kSFRowStride = L2_SHAPE_K / 32;
                        auto* sf_u8 = reinterpret_cast<uint8_t*>(workspace.get_l2_sf_ptr());
                        sf_u8[(pool_block * SF_BLOCK_M + sf_row) * kSFRowStride
                              + k_col * 4 + k_byte] = sf_x;
                        sf_u8[(pool_block * SF_BLOCK_M + sf_row + 4) * kSFRowStride
                              + k_col * 4 + k_byte] = sf_y;
                    }
                }

                // Wait for previous TMA store before overwriting smem_cd
                if (epi_warp == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();
                cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                // TMA store smem_cd → l2_acts[pool_block*BLOCK_M + w*STORE_BLOCK_M, :]
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                if (epi_warp == 0 and cute::elect_one_sync()) {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_l1_out,
                        smem_cd[tma_stage],
                        n_block_idx * L1_OUT_N,
                        pool_block * BLOCK_M + w * STORE_BLOCK_M);
                    cute::tma_store_arrive();
                }
                tma_stage = (tma_stage + 1) % kNumTMAStoreStages;
            }

            // Wait all wave TMA stores done before signaling L2
            cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            if (epi_warp == 0 and cute::elect_one_sync()) {
                // Fence ensures l2_acts is visible before setting mask bit
                __threadfence();
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(pool_block),
                    1ull << n_block_idx);
            }
        };

        // ---- L2 epilogue: BF16 scatter to combine_buffer -------------------
        auto l2_epilogue = [&](uint32_t valid_m,
                                uint32_t pool_block,
                                uint32_t n_block_idx) {
            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                for (uint32_t s = 0; s < kNStoresN; ++s) {
                    if (epi_warp == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Load TMEM → smem_cd_l2 (BF16 after cast)
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N_BF16 / kEpBF16; ++i) {
                        const bool kHasShortcut = (128u / kNBG) == 8;
                        auto bg = i + lane_idx * (128u / kNBG);
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bg / 8);
                        auto col = kHasShortcut ? i : (bg % 8);
                        col ^= row % (128u / 16u);

                        const uint32_t tmem_addr =
                            accum_stage_idx * kNumMWaves * BLOCK_N
                            + w * BLOCK_N + s * STORE_BLOCK_N_BF16 + i * kEpBF16;

                        uint32_t v[8];
                        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l2)
                                      + epi_warp * 32 * 128u
                                      + row * (kNBG * 8) + col * kNBG;
                        st_shared(smem_ptr,
                                  cast_into_bf16_and_pack(v[0], v[1]),
                                  cast_into_bf16_and_pack(v[2], v[3]),
                                  cast_into_bf16_and_pack(v[4], v[5]),
                                  cast_into_bf16_and_pack(v[6], v[7]));
                    }

                    // Release TMEM on last store
                    if (w == kNumMWaves - 1 and s == kNStoresN - 1) {
                        tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Scatter each row to combine_buffer[topk_k][orig_token][n]
                    constexpr uint32_t kRowsPerWarp = STORE_BLOCK_M / kNumEpilogueWarps;
                    for (uint32_t r = 0; r < kRowsPerWarp; ++r) {
                        const uint32_t row_in_store = epi_warp * kRowsPerWarp + r;
                        const uint32_t row_global   = pool_block * BLOCK_M
                                                     + w * STORE_BLOCK_M + row_in_store;
                        if (row_in_store >= valid_m) break;

                        const auto meta = *workspace.get_token_src_metadata_ptr(row_global);
                        const uint32_t orig_tok = static_cast<uint32_t>(meta.orig_token_idx);
                        const uint32_t topk_k   = static_cast<uint32_t>(meta.topk_k);

                        // Each lane writes kEpBF16 BF16 values
                        const uint32_t col = s * STORE_BLOCK_N_BF16 + lane_idx * kEpBF16;
                        const auto n_col = n_block_idx * BLOCK_N + col;

                        // Read BF16 from smem_cd_l2
                        const auto smem_src = reinterpret_cast<const uint32_t*>(smem_cd_l2)
                            + row_in_store * (BLOCK_N / 2);  // simplified row stride

                        auto* dst = reinterpret_cast<uint32_t*>(
                            reinterpret_cast<uint8_t*>(workspace.get_combine_buffer_ptr())
                            + (static_cast<uint64_t>(topk_k) * workspace.num_tokens + orig_tok)
                              * L2_SHAPE_N * sizeof(uint16_t)
                            + n_col * sizeof(uint16_t));

                        if (n_col + kEpBF16 <= L2_SHAPE_N and col + kEpBF16 <= STORE_BLOCK_N_BF16) {
                            uint32_t tmp[kEpBF16 / 2];
                            #pragma unroll
                            for (uint32_t t = 0; t < kEpBF16 / 2; ++t)
                                tmp[t] = smem_src[lane_idx * (kEpBF16 / 2) + t];
                            #pragma unroll
                            for (uint32_t t = 0; t < kEpBF16 / 2; ++t)
                                dst[t] = tmp[t];
                        }
                    }
                }
            }
        };

        // ---- Main epilogue dispatch loop ------------------------------------
        scheduler.for_each_block([&](BlockPhase phase,
                                      uint32_t /*expert_idx*/,
                                      uint32_t /*num_k_blocks*/,
                                      uint32_t m_block_idx,
                                      uint32_t n_block_idx) {
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            tcgen05_after_thread_sync();

            const uint32_t valid_m  = scheduler.get_current_valid_m();
            const uint32_t pool_blk = scheduler.get_current_pool_block_offset();

            if (phase == BlockPhase::Linear1) {
                l1_epilogue(valid_m, pool_blk, m_block_idx, n_block_idx);
                // Release TMEM for L1 (L2 releases inside l2_epilogue)
                tcgen05_before_thread_sync();
                tmem_empty_barriers[accum_stage_idx]->arrive(0u);
            } else {
                l2_epilogue(valid_m, pool_blk, n_block_idx);
                // TMEM release happens inside l2_epilogue on last store step
            }

            accum_advance();
        });

        if (epi_warp == kNumUMMAStoreThreads / 32 - 1)
            Allocator().free(0, kNumTmemCols);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

}  // namespace asym_gemm

#pragma clang diagnostic pop
