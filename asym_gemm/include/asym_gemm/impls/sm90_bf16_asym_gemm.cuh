#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm90_utils.cuh>

namespace asym_gemm {

// SM90 BF16 asymmetric GEMM: B-centric access pattern.
// Outer loop over K blocks, inner loop over M blocks.
// B is loaded once per K block and shared across all M blocks.
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
sm90_bf16_asym_gemm_impl(uint32_t* offsets, uint32_t* experts,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
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
    using WGMMA = typename BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t WAVE_BLOCK_M = (BLOCK_M <= WGMMA::M) ? BLOCK_M : cute::min<uint32_t>(BLOCK_M, static_cast<uint32_t>(WGMMA::M) * 2);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
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
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA dimensions
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = WAVE_BLOCK_M;
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support A/D layout without multicast on A");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE; // single slot for B
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B/CD must be aligned to 1024 bytes");

    // D/A/B shared memory layout: [smem_cd | smem_a stages | smem_b (1 slot) | barriers]
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE);
    });

    // Barriers: full[kNumStages] + empty[kNumStages] + full_b[1] + empty_b[1]
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE);
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
        // TMA warp-group: loads A (staged) and B (single slot)
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Use the third warp in the TMA warp-group
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                uint32_t k_idx = block_k_iter * BLOCK_K;
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                // Wait for B consumer release
                empty_barriers_b[0]->wait(phase_b ^ 1);
                phase_b ^= 1;

                // Load B tile (single slot)
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], k_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], n_idx, k_idx, kNumMulticast, batch_idx);

                if (is_leader_cta) {
                    full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                } else {
                    full_barriers_b[0]->arrive(0u);
                }

                // Inner loop over M blocks
                for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                    uint32_t m_idx = (kGemmType == GemmType::MGroupedMasked)
                                     ? (scheduler.current_group_idx * shape_m + (block_m_iter - scheduler.m_start) * BLOCK_M)
                                     : (block_m_iter * BLOCK_M);

                    // Wait consumer release for A
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                     kMajorA == cute::UMMA::Major::K, "Invalid major");

                    // Add 2-CTA offsets
                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        // n_idx is const for this block, no need to modify
                    }

                    // Issue TMA for A
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, kNumMulticast, batch_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::MN)
                        tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_idx, kNumMulticast, batch_idx);

                    constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE;
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
        // Math warp-group(s): WGMMA + epilogue
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
            // Wait for B tile
            full_barriers_b[0]->wait(phase_b);
            phase_b ^= 1;

            // Inner loop over M blocks
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                // Accumulator in registers -- zero-initialized per M block
                float accum[WGMMA::kNumAccum * kNumMWaves] = {0};

                // Wait for A tile
                full_barriers[stage_idx]->wait(phase);

                // Compute GMMA descriptor base addresses for this stage
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                // B is always in slot 0 (single slot)
                const auto b_desc_base_lo = b_desc_lo;

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
                        a_desc.reg32_[0] = advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, nv_bfloat16>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumMWaves; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Release A smem
                empty_barrier_arrive_a(stage_idx);

                // Skip WGMMA store for threads not participating
                if (not do_wgmma_store) continue;

                // ============================================================
                // Epilogue: write accumulators to smem_cd via STSM, then TMA store
                // ============================================================

                // Wait last TMA store to be finished
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                    cute::tma_store_wait<kNumTMAStoreStages - 1>();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

                // warp_idx within warp-group for STSM offset computation
                const uint32_t wg_local_warp_idx = warp_idx % 4;

                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    // Write back to shared memory using STSM
                    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "Invalid swizzling type");
                    DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < kNumMWaves; ++ local_idx) {
                        auto m_offset = local_idx * WAVE_BLOCK_M;
                        auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                        #pragma unroll
                        for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            uint8_t* smem_ptr = nullptr;
                            if constexpr (kSwizzleCDMode > 0) {
                                constexpr uint32_t kNumBankGroupBytes = 16;
                                auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                                auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                                auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                                auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                                col ^= row % (kSwizzleCDMode / 16);
                                smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                    wg_local_warp_idx * (WGMMA_M_PER_WARP * kSwizzleCDMode) +
                                    m_offset * kSwizzleCDMode +
                                    atom_offset * STORE_BLOCK_M * kSwizzleCDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                            } else {
                                smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx] + (m_offset + wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                            }
                            SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                                __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                                __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                                smem_ptr
                            );
                        }
                    }
                } else {
                    // Use st.shared for FP32 output
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < kNumMWaves; ++ local_idx) {
                        auto m_offset = local_idx * WAVE_BLOCK_M;
                        auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                        auto smem_d_0 = reinterpret_cast<float2*>(smem_cd[tma_stage_idx] + (m_offset + wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                        auto smem_d_1 = reinterpret_cast<float2*>(smem_cd[tma_stage_idx] + (m_offset + wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            st_shared(smem_d_0 + i * 4, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
                            st_shared(smem_d_1 + i * 4, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
                        }
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
