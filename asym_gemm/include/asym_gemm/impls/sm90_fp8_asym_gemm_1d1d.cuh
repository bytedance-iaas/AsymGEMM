#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm90_utils.cuh>

namespace asym_gemm {

// SM90 FP8 asymmetric GEMM: B-centric access pattern.
// Outer loop over K blocks, inner loop over M blocks.
// B is loaded once per K block and shared across all M blocks.
// Per-element float32 scale factors: SFA[BLOCK_M] per M-block, SFB[BLOCK_N] per K-block.
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm90_fp8_asym_gemm_1d1d_impl(uint32_t* offsets, uint32_t* experts,
                              uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using namespace asym_gemm::sm90;

    // On SM90, we remap the thread budget:
    //   kNumMathThreads = kNumNonEpilogueThreads (math warp-groups do MMA + epilogue)
    //   kNumTMAThreads  = kNumEpilogueThreads    (TMA warp-group does only loading)
    // This keeps the total thread count the same as SM100 for ABI compatibility.
    constexpr uint32_t kNumMathThreads = kNumNonEpilogueThreads;
    constexpr uint32_t kNumTMAThreads = kNumEpilogueThreads;

    // No stage merging for asym kernel on SM90
    constexpr uint32_t kNumStagesPerMerge = 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_;
    constexpr uint32_t kNumStages = kNumStages_;

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // FP8 asym GEMM on SM90 requires FP32 output
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "FP8 asym GEMM on SM90 requires FP32 output");

    // GEMM with accumulation must have FP32 output (already enforced above)
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    DG_STATIC_ASSERT(BLOCK_K % WGMMA::K == 0, "BLOCK_K must be multiple of WGMMA K for FP8");

    // Configs
    constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M <= WGMMA::M) ? BLOCK_M : cute::min<uint32_t>(BLOCK_M, static_cast<uint32_t>(WGMMA::M) * 2);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    DG_STATIC_ASSERT(kNumMWaves == 1, "BLOCK_M > WAVE_BLOCK_M not supported for SM90 asym GEMM");
    constexpr uint32_t kNumTMAStoreStages = 2;
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block M");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA dimensions
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = WAVE_BLOCK_M;
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode == 0 ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support A/D layout without multicast on A");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kIsMulticastOnA, "B-side multicast not supported in SM90 asym GEMM");

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * (kSwizzleCDMode == 0 ? BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)) : kSwizzleCDMode);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE; // single slot for B
    // Scale factor shared memory: float32 per row (SFA) and per column (SFB)
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B/CD must be aligned to 1024 bytes");

    // Smem layout:
    // [smem_cd: kNumTMAStoreStages stages]
    // [smem_a:  kNumStages stages]
    // [smem_b:  1 slot]
    // [smem_sfa: kNumStages stages, BLOCK_M floats each]
    // [smem_sfb: 1 slot, BLOCK_N floats]
    // [barriers: full_barriers, empty_barriers, full_barriers_b, empty_barriers_b]
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE);
    });

    // Scale factor smem (after smem_b)
    const auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE;
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers: full[kNumStages] + empty[kNumStages] + full_b[1] + empty_b[1]
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(sf_start_ptr +
        kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers   = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto full_barriers_b  = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2; });
    auto empty_barriers_b = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + 1; });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMulticast * kNumMathThreads / 32);
        }
        full_barriers_b[0]->init(1);
        empty_barriers_b[0]->init(kNumMulticast * kNumMathThreads / 32);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barriers visible
    (kNumMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    // Block scheduler
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, experts, offsets);

    // Early-exit for inactive expert slots in the masked layout.
    if constexpr (kGemmType == GemmType::MGroupedMasked) {
        if (scheduler.m_end == 0) return;
    }

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, phase_b = 0;
    auto advance_pipeline = [&](uint32_t& block_idx) {
        ++ block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    uint32_t block_k = ceil_div_device(shape_k, BLOCK_K);
    uint32_t n_idx = scheduler.n_idx;

    // Merged stages constants (no merge for asym)
    constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;

    if (warp_idx >= kNumMathThreads / 32) {
        // =====================================================================
        // TMA warp-group: loads A (staged), B (single slot), SFA (per M-block), SFB (per K-block)
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Use the third warp in the TMA warp-group
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                uint32_t k_idx = block_k_iter * BLOCK_K;
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                // Scale factor K index: one scale per BLOCK_K elements
                const uint32_t sf_k_idx = block_k_iter;

                // Wait for B consumer release
                empty_barriers_b[0]->wait(phase_b ^ 1);
                phase_b ^= 1;

                // Load B tile (single slot)
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::float_e4m3_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], k_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::float_e4m3_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], n_idx, k_idx, kNumMulticast, batch_idx);

                // Load SFB (per K-block, single slot) — one float per column of B.
                // SFB is laid out per-group along the K-outer dim (see make_tma_sf_desc),
                // so the group offset goes into the K coordinate, NOT the MN coordinate.
                const uint32_t sfb_n_idx = blockIdx.x * BLOCK_N;
                const uint32_t sfb_k_idx = scheduler.current_group_idx * block_k + sf_k_idx;
                tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers_b[0], smem_sfb[0], sfb_n_idx, sfb_k_idx);

                if (is_leader_cta) {
                    full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                } else {
                    full_barriers_b[0]->arrive(0u);
                }

                // Inner loop over M blocks
                for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                    uint32_t m_idx = (kGemmType == GemmType::MGroupedMasked)
                                     ? (scheduler.current_group_idx * shape_m + (block_m_iter - scheduler.m_start) * BLOCK_M)
                                     : (block_m_iter * BLOCK_M);
                    // local_m_idx for SFA TMA (row index in the per-group token dimension)
                    const uint32_t local_m_idx = (kGemmType == GemmType::MGroupedMasked)
                                                 ? (block_m_iter - scheduler.m_start) * BLOCK_M
                                                 : (block_m_iter * BLOCK_M);

                    // Wait consumer release for A
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                     kMajorA == cute::UMMA::Major::K, "Invalid major");

                    // Add 2-CTA offsets
                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    }

                    // Issue TMA for A
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::float_e4m3_t, kIsBatchedMM>(
                            &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, kNumMulticast, batch_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::MN)
                        tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::float_e4m3_t, kIsBatchedMM>(
                            &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_idx, kNumMulticast, batch_idx);

                    // Load SFA (per M-block per K-block) — one float per row of A
                    const uint32_t sfa_k_idx = (kGemmType == GemmType::MGroupedMasked)
                        ? scheduler.current_group_idx * block_k + sf_k_idx
                        : sf_k_idx;
                    tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], local_m_idx, sfa_k_idx);

                    constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE;
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // =====================================================================
        // Math warp-group(s): WGMMA + scale application + epilogue
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        // Build GMMA descriptors from smem base pointers
        auto a_desc = make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Number of threads participating in WGMMA store (for NamedBarrier)
        DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for BLOCK_M < 64");
        constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
        const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

        // TMA checks for epilogue
        constexpr uint32_t kNumElemBytes = sizeof(cd_dtype_t);
        constexpr uint32_t TMA_D_BLOCK_N = kSwizzleCDMode == 0 ? BLOCK_N : (kSwizzleCDMode / kNumElemBytes);
        constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;

        // Row/column indices for this thread's accumulator elements
        // Each WGMMA thread owns specific rows/cols based on warp+lane position:
        //   r_0 = (warp_idx % 4) * 16 + lane_idx / 4   (first 8 rows of this thread)
        //   r_1 = r_0 + 8                                (second 8 rows)
        //   col_idx = lane_idx % 4
        const uint32_t wg_local_warp_idx = warp_idx % 4;
        const uint32_t row_idx = lane_idx / 4, col_idx_scale = lane_idx % 4;
        const uint32_t r_0 = wg_local_warp_idx * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        // Empty barrier arrival helpers
        auto empty_barrier_arrive_a = [&](uint32_t s) {
            if constexpr (kNumMulticast == 1) {
                lane_idx == 0 ? empty_barriers[s]->arrive() : void();
            } else {
                auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                lane_idx < kNumMulticast ? empty_barriers[s]->arrive(target_cta) : void();
            }
        };

        auto empty_barrier_arrive_b = [&]() {
            if constexpr (kNumMulticast == 1) {
                lane_idx == 0 ? empty_barriers_b[0]->arrive() : void();
            } else {
                auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                lane_idx < kNumMulticast ? empty_barriers_b[0]->arrive(target_cta) : void();
            }
        };

        uint32_t tma_stage_idx = 0;

        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            // Wait for B tile (includes SFB)
            full_barriers_b[0]->wait(phase_b);
            phase_b ^= 1;

            // Inner loop over M blocks
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                // Accumulator in registers -- zero-initialized per M block
                float accum[WGMMA::kNumAccum * kNumMWaves] = {0};

                // Wait for A tile (includes SFA)
                full_barriers[stage_idx]->wait(phase);

                // Compute GMMA descriptor base addresses for this stage
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                // B is always in slot 0 (single slot)
                const auto b_desc_base_lo = b_desc_lo;

                // Read scale factors while smem_sfa/smem_sfb are still owned by the math side
                // (must happen before empty_barrier_arrive_a releases smem_sfa[stage_idx])
                // Load SFB: 2 floats per group-of-4 accumulator columns
                float2 scales_b[WGMMA::kNumAccum / 4];
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = *reinterpret_cast<const float2*>(smem_sfb[0] + i * 8 + col_idx_scale * 2);

                // Load per-token (row) scale factors for rows owned by this thread
                // m_offset accounts for multiple M waves (kNumMWaves==1 here, but kept general)
                float scale_a_0[kNumMWaves], scale_a_1[kNumMWaves];
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < kNumMWaves; ++local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    scale_a_0[local_idx] = ld_shared(smem_sfa[stage_idx] + m_offset + r_0);
                    scale_a_1[local_idx] = ld_shared(smem_sfa[stage_idx] + m_offset + r_1);
                }

                // Issue WGMMA
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumMWaves; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < kNumMWaves; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
                        a_desc.reg32_[0] = advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, cutlass::float_e4m3_t>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, cutlass::float_e4m3_t>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumMWaves; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Release A smem (safe: scale reads already completed above)
                empty_barrier_arrive_a(stage_idx);

                // Skip WGMMA store for threads not participating
                if (not do_wgmma_store) continue;

                // ============================================================
                // Epilogue: apply scale factors, write scaled accumulators to
                // smem_cd via st_shared (FP32 only), then TMA store
                // ============================================================

                // Wait last TMA store to be finished
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                    cute::tma_store_wait<kNumTMAStoreStages - 1>();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

                // Use st_shared for FP32 output with per-element scale application
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < kNumMWaves; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;

                    auto smem_d_0 = reinterpret_cast<float2*>(smem_cd[tma_stage_idx] + (m_offset + wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    auto smem_d_1 = reinterpret_cast<float2*>(smem_cd[tma_stage_idx] + (m_offset + wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);

                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // Apply scale: final = scale_a * scale_b * raw_accum
                        float val_0 = scale_a_0[local_idx] * scales_b[i].x * shifted_accum[i * 4 + 0];
                        float val_1 = scale_a_0[local_idx] * scales_b[i].y * shifted_accum[i * 4 + 1];
                        float val_2 = scale_a_1[local_idx] * scales_b[i].x * shifted_accum[i * 4 + 2];
                        float val_3 = scale_a_1[local_idx] * scales_b[i].y * shifted_accum[i * 4 + 3];
                        st_shared(smem_d_0 + i * 4, make_float2(val_0, val_1));
                        st_shared(smem_d_1 + i * 4, make_float2(val_2, val_3));
                    }
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

                // TMA store to global memory
                // Compute m_idx and n_idx for output
                const auto m_idx_out = (kGemmType == GemmType::MGroupedMasked)
                    ? (scheduler.current_group_idx * shape_m + (block_m_iter - scheduler.m_start) * BLOCK_M)
                    : (BLOCK_M * block_m_iter);
                const auto n_idx_out = blockIdx.x * BLOCK_N;

                DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                    auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                    auto smem_ptr = smem_cd[tma_stage_idx] + in_block_n_offset * STORE_BLOCK_M;

                    if constexpr (kGemmType == GemmType::Batched) {
                        if (block_k_iter == 0) {
                            using cute_tma_t = cute::conditional_t<false,
                                cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                            cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                             n_idx_out + in_block_n_offset, m_idx_out, scheduler.current_group_idx);
                        } else {
                            using cute_tma_t = cute::conditional_t<true,
                                cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                            cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                             n_idx_out + in_block_n_offset, m_idx_out, scheduler.current_group_idx);
                        }
                    } else {
                        if (block_k_iter == 0) {
                            cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr,
                                                          n_idx_out + in_block_n_offset, m_idx_out);
                        } else {
                            cute::SM90_TMA_REDUCE_ADD_2D::copy(&tensor_map_cd, smem_ptr,
                                                               n_idx_out + in_block_n_offset, m_idx_out);
                        }
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();

                tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
            }

            // Release B smem after all M blocks are done for this K block
            empty_barrier_arrive_b();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
