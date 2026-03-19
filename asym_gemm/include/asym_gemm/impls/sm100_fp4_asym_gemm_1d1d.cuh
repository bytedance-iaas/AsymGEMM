#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/float_subbyte.h>

#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#endif

#include <asym_gemm/common/epilogue_utils.cuh>
#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm100_utils.cuh>

namespace asym_gemm {

using namespace asym_gemm::sm100;
using fp4_input_element_t = cutlass::float_e2m1_t;

DG_STATIC_ASSERT(cutlass::sizeof_bits<fp4_input_element_t>::value == 4, "FP4 kernel expects E2M1 FP4 input");
#if __has_include(<cuda_fp4.h>)
DG_STATIC_ASSERT(sizeof(__nv_fp4_e2m1) == 1, "__nv_fp4_e2m1 must be byte-addressable storage");
#endif

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          typename epilogue_type_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp4_asym_gemm_1d1d_impl(uint32_t* offsets, uint32_t* experts,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // if (blockIdx.x != 0 || blockIdx.y != 0)
    //     return;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(BLOCK_K % 64 == 0, "block K must be a multiple of 64 for FP4");
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");

    // SF (scale factor) indexing: 16-element granularity for NVFP4
    constexpr uint32_t kSFQuantK = 16;                                                     // SF granularity (elements)
    constexpr uint32_t kSFAtomsPerBlockK = BLOCK_K / kSFQuantK;                            // SFs consumed per k-block
    constexpr uint32_t kNumSFPerPack = sizeof(uint32_t) / sizeof(cutlass::float_ue4m3_t);  // 4 (UE4M3 per uint32)
    constexpr uint32_t kSFPacksPerBlockK = kSFAtomsPerBlockK / kNumSFPerPack;              // Paks per block K
    DG_STATIC_ASSERT(kSFAtomsPerBlockK % kNumSFPerPack == 0, "SFs per block_k must be divisible by SF per pack");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sf_k = ceil_div(shape_k, kSFQuantK * kNumSFPerPack);

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();
    const auto smid = get_smid();

    constexpr bool kDebugTrace = true;
    constexpr int kDebugSM = -1;  // set >=0 to filter to one SM
    // Targeted debug tile for large kernel-vs-manual mismatch investigation.
    constexpr int kDbgBlockX = 6;
    constexpr int kDbgBlockY = 1;
    constexpr int kDbgKBlock = 0;
    constexpr int kDbgMBlock = 14;
    auto should_log = [&](uint32_t k_iter, uint32_t m_iter) {
        if constexpr (!kDebugTrace)
            return false;
        const bool cta_ok = (blockIdx.x == 0 && blockIdx.y == 0);
        const bool lane_ok = (lane_idx == 0);
        const bool sm_ok = (kDebugSM < 0) || (static_cast<int>(smid) == kDebugSM);
        const bool iter_ok = true;
        // (k_iter < 2 && m_iter < 2);
        return cta_ok && lane_ok && sm_ok && iter_ok;
    };
    auto should_log_target = [&](uint32_t k_iter, uint32_t m_iter) {
        if constexpr (!kDebugTrace)
            return false;
        return lane_idx == 0 &&
               static_cast<int>(blockIdx.x) == kDbgBlockX &&
               static_cast<int>(blockIdx.y) == kDbgBlockY &&
               static_cast<int>(k_iter) == kDbgKBlock &&
               static_cast<int>(m_iter) == kDbgMBlock;
    };

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K / 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K / 2;
    constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE;
    constexpr uint32_t TMA_A_BYTES_PER_STAGE = LOAD_BLOCK_M * (BLOCK_K / 2) * sizeof(fp4_input_element_t);
    constexpr uint32_t TMA_B_BYTES_PER_STAGE = LOAD_BLOCK_N * (BLOCK_K / 2) * sizeof(fp4_input_element_t);
    DG_STATIC_ASSERT(TMA_A_BYTES_PER_STAGE == SMEM_A_SIZE_PER_STAGE, "FP4 A-stage TMA bytes/SMEM bytes mismatch");
    DG_STATIC_ASSERT(TMA_B_BYTES_PER_STAGE == SMEM_B_SIZE_PER_STAGE, "FP4 B-stage TMA bytes/SMEM bytes mismatch");
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t) * kSFPacksPerBlockK;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t) * kSFPacksPerBlockK;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K / 2;
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Automatically deduce the number of epilogue stages (1 or 2), according to the tensor memory size
    // TODO: test cases of `kNumMWaves == 2 and kNumEpilogueStages == 2`
    constexpr uint32_t kNumSFATmemCols = (SF_BLOCK_M / 32) * kSFPacksPerBlockK;
    constexpr uint32_t kNumSFBTmemCols = (SF_BLOCK_N / 32) * kSFPacksPerBlockK;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // D/A/B shared memory
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<fp4_input_element_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<fp4_input_element_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE;
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * SMEM_A_SIZE_PER_STAGE +
        SMEM_B_SIZE +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto with_sf_full_barriers_b    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages * 2); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto full_barriers_b            = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages * 2 + 1); });
    auto empty_barriers_b           = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages * 2 + 2); });

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 3);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        with_sf_full_barriers_b[0]->init(kNumMulticast * 32);
        full_barriers_b[0]->init(1);
        empty_barriers_b[0]->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler: BF16-style B-centric traversal.
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, experts, offsets);
    const uint32_t num_total_k_blocks = ceil_div_device(shape_k, BLOCK_K);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, phase_b = 0, sf_phase_b = 0;
    uint32_t accum_stage_idx = 0, accum_phase_idx = 0;
    auto advance_pipeline = [&](uint32_t& block_idx) {
        ++ block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    auto advance_accum_pipeline = [&]() {
        accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
        accum_phase_idx ^= accum_stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp (B-centric): for this CTA's B tile, sweep K then all M tiles in its segment.
        const uint32_t n_block_idx = blockIdx.x;
        constexpr bool kIsBatchedB =
            (kGemmType == GemmType::Batched) ||
            (kGemmType == GemmType::MGroupedContiguous) ||
            (kGemmType == GemmType::MGroupedMasked);
        const uint32_t base_n_idx = kIsBatchedB
            ? (scheduler.n_idx - scheduler.current_group_idx * shape_n)
            : scheduler.n_idx;
        uint32_t b_n_idx = base_n_idx;
        if constexpr (kNumMulticast > 1)
            b_n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);

        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; ++k_block_idx) {
            empty_barriers_b[0]->wait(phase_b ^ 1);
            phase_b ^= 1;

            const uint32_t k_idx = k_block_idx * BLOCK_K;
            const uint32_t k_packed_idx = k_block_idx * (BLOCK_K / 2);  // TMA coord in uint8 (byte) units
            const uint32_t sf_k_idx = ceil_div(k_idx, kSFQuantK * kNumSFPerPack);
            constexpr bool kIsBatchedA = (kGemmType == GemmType::Batched);
            const uint32_t batch_idx_a = (kIsBatchedA ? scheduler.current_group_idx : 0);
            const uint32_t batch_idx_b = (kIsBatchedB ? scheduler.current_group_idx : 0);
            if constexpr (kMajorB == cute::UMMA::Major::K)
                tma_copy<BLOCK_K / 2, LOAD_BLOCK_N, kSwizzleBMode, fp4_input_element_t, kIsBatchedB>(
                    &tensor_map_b, full_barriers_b[0], smem_b[0], k_packed_idx, b_n_idx, 1, batch_idx_b);
            if constexpr (kMajorB == cute::UMMA::Major::MN)
                tma_copy<LOAD_BLOCK_N, BLOCK_K / 2, kSwizzleBMode, fp4_input_element_t, kIsBatchedB>(
                    &tensor_map_b, full_barriers_b[0], smem_b[0], b_n_idx, k_packed_idx, 1, batch_idx_b);
            
            const uint32_t sf_stage_in_group_idx = 0;
            if (sf_stage_in_group_idx == 0) {
                if (lane_idx == 0 &&
                    static_cast<int>(blockIdx.x) == kDbgBlockX &&
                    static_cast<int>(blockIdx.y) == kDbgBlockY &&
                    static_cast<int>(k_block_idx) == kDbgKBlock) {
                    const uint32_t sfb_outer_idx = scheduler.current_group_idx * shape_sf_k + sf_k_idx;
                    printf("[FP4DBG][TARGET][W0][SM%u] B/SFB load: block=(%u,%u) group=%u "
                           "m_start=%u m_end=%u n_block_idx=%u b_n_idx=%u k_block_idx=%u "
                           "k_packed_idx=%u sf_k_idx=%u sfb_outer_idx=%u\n",
                           smid, blockIdx.x, blockIdx.y, scheduler.current_group_idx,
                           scheduler.m_start, scheduler.m_end,
                           n_block_idx, b_n_idx, k_block_idx, k_packed_idx, sf_k_idx, sfb_outer_idx);
                }
                if (k_block_idx == 0 && lane_idx == 0) {
                    const uint32_t outer_idx = scheduler.current_group_idx * shape_sf_k + sf_k_idx;
                    const uint32_t outer_dim = kNumGroups * shape_sf_k;
                    printf("[FP4_TMA_SFB] block=(%u,%u) sm=%u n_block_idx=%u b_n_idx=%u "
                           "group=%u sf_k_idx=%u shape_sf_k=%u outer_idx=%u outer_dim=%u n=%u\n",
                           blockIdx.x, blockIdx.y, smid, n_block_idx, b_n_idx,
                           scheduler.current_group_idx, sf_k_idx, shape_sf_k, outer_idx, outer_dim, shape_n);
                }
                tma_copy<BLOCK_N, kSFPacksPerBlockK, 0>(&tensor_map_sfb, full_barriers_b[0], smem_sfb[0],
                                        n_block_idx * BLOCK_N, scheduler.current_group_idx * shape_sf_k + sf_k_idx);
                full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * kSFPacksPerBlockK);
            }

            for (uint32_t m_block_idx = scheduler.m_start; m_block_idx < scheduler.m_end; advance_pipeline(m_block_idx)) {
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W0][SM%u] pre empty.wait k=%u m=%u stage=%u phase=%u\n",
                           smid, k_block_idx, m_block_idx, stage_idx, phase);
                empty_barriers[stage_idx]->wait(phase ^ 1);
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W0][SM%u] post empty.wait k=%u m=%u stage=%u phase=%u\n",
                           smid, k_block_idx, m_block_idx, stage_idx, phase);

                const uint32_t local_m_idx = kGemmType == GemmType::MGroupedMasked 
                                           ? (m_block_idx - scheduler.m_start) * BLOCK_M
                                           : (m_block_idx * BLOCK_M);
                const uint32_t m_idx = kGemmType == GemmType::MGroupedMasked 
                                     ? (scheduler.current_group_idx * shape_m + local_m_idx)
                                     : local_m_idx;

                if constexpr (kNumMulticast > 1)
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;

                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K / 2, LOAD_BLOCK_M, kSwizzleAMode, fp4_input_element_t, kIsBatchedA>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_packed_idx, m_idx, 1, batch_idx_a);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_M, BLOCK_K / 2, kSwizzleAMode, fp4_input_element_t, kIsBatchedA>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_packed_idx, 1, batch_idx_a);

                if (sf_stage_in_group_idx == 0) {
                    const uint32_t sfa_k_idx = kGemmType == GemmType::MGroupedMasked
                                             ? scheduler.current_group_idx * shape_sf_k + sf_k_idx
                                             : sf_k_idx;
                    if (should_log_target(k_block_idx, m_block_idx)) {
                        printf("[FP4DBG][TARGET][W0][SM%u] A/SFA load: block=(%u,%u) "
                               "m_block_idx=%u local_m_idx=%u m_idx=%u sfa_k_idx=%u\n",
                               smid, blockIdx.x, blockIdx.y,
                               m_block_idx, local_m_idx, m_idx, sfa_k_idx);
                    }
                    if (k_block_idx == 0 && m_block_idx == scheduler.m_start && lane_idx == 0) {
                        printf("[FP4_TMA_SFA] block=(%u,%u) sm=%u local_m_idx=%u sfa_k_idx=%u shape_sf_k=%u m=%u\n",
                               blockIdx.x, blockIdx.y, smid, local_m_idx, sfa_k_idx, shape_sf_k, shape_m);
                    }
                    tma_copy<BLOCK_M, kSFPacksPerBlockK, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx],
                                            local_m_idx, sfa_k_idx);
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(uint32_t) * kSFPacksPerBlockK);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 64; // NVFP4 uses UMMA_K = 64
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<fp4_input_element_t, fp4_input_element_t,
                                                                   float, cutlass::float_ue4m3_t,
                                                                   UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");

        // For multi-atom K (block_k > swizzle atom), create descriptors per-atom
        // so LBO=0, and manually rebase between atoms in the inner loop.
        constexpr uint32_t SWIZZLE_ATOM_K_B = kSwizzleBMode * 2;  // 256 for FP4/SW128
        constexpr uint32_t DESC_ATOM_K = (BLOCK_K > SWIZZLE_ATOM_K_B) ? SWIZZLE_ATOM_K_B : BLOCK_K;
        constexpr uint32_t NUM_K_ATOMS = BLOCK_K / DESC_ATOM_K;
        constexpr uint32_t UMMA_ITERS_PER_ATOM = DESC_ATOM_K / UMMA_K;

        auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, DESC_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, DESC_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };

        auto empty_barrier_arrive = [&]() {
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
        };
        using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
            cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; ++k_block_idx) {
            with_sf_full_barriers_b[0]->wait(sf_phase_b);
            tcgen05_after_thread_sync();
            sf_phase_b ^= 1;
            const uint32_t sf_stage_in_group_idx = 0;

            if (sf_stage_in_group_idx == 0)
            {
                #pragma unroll
                for (uint32_t p = 0; p < kSFPacksPerBlockK; ++p) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfb[0] + p * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + p * (SF_BLOCK_N / 32) + i * 4);
                    }
                }
            }
            
            for (uint32_t m_block_idx = scheduler.m_start; m_block_idx < scheduler.m_end; advance_pipeline(m_block_idx), advance_accum_pipeline()) {
                // Ensure this accumulator stage is released by epilogue before UMMA writes into it again.
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                // Wait TMA and SF-transpose arrival
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W1][SM%u] pre with_sf.wait k=%u m=%u stage=%u phase=%u\n",
                            smid, k_block_idx, m_block_idx, stage_idx, phase);
                with_sf_full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W1][SM%u] post with_sf.wait k=%u m=%u stage=%u phase=%u\n",
                            smid, k_block_idx, m_block_idx, stage_idx, phase);

                // ---- Value dump: verify TMA loaded correct FP4 data + scale factors ----
                if (k_block_idx == 0 && should_log(k_block_idx, m_block_idx)) {
                    // FP4 packed: each byte holds 2 FP4 values (lo nibble = elem[0], hi nibble = elem[1])
                    const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(smem_a[stage_idx]);
                    printf("[FP4DBG][SMEM][SM%u][k=%u m=%u] A_bytes[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                           smid, k_block_idx, m_block_idx,
                           a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3],
                           a_ptr[4], a_ptr[5], a_ptr[6], a_ptr[7]);
                    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(smem_b[0]);
                    printf("[FP4DBG][SMEM][SM%u][k=%u m=%u] B_bytes[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                           smid, k_block_idx, m_block_idx,
                           b_ptr[0], b_ptr[1], b_ptr[2], b_ptr[3],
                           b_ptr[4], b_ptr[5], b_ptr[6], b_ptr[7]);
                    // SFA/SFB: uint32 packs of 4 UE4M3 bytes each (raw hex shows packed bytes)
                    const uint32_t* sfa_ptr = smem_sfa[stage_idx];
                    printf("[FP4DBG][SMEM][SM%u][k=%u m=%u] SFA_packs[0..3]: %08x %08x %08x %08x\n",
                           smid, k_block_idx, m_block_idx,
                           sfa_ptr[0], sfa_ptr[1], sfa_ptr[2], sfa_ptr[3]);
                    const uint32_t* sfb_ptr = smem_sfb[0];
                    printf("[FP4DBG][SMEM][SM%u][k=%u m=%u] SFB_packs[0..3]: %08x %08x %08x %08x\n",
                           smid, k_block_idx, m_block_idx,
                           sfb_ptr[0], sfb_ptr[1], sfb_ptr[2], sfb_ptr[3]);
                }
                if (should_log_target(k_block_idx, m_block_idx)) {
                    const uint8_t* a_ptr = reinterpret_cast<const uint8_t*>(smem_a[stage_idx]);
                    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(smem_b[0]);
                    const uint32_t* sfa_ptr = smem_sfa[stage_idx];
                    const uint32_t* sfb_ptr = smem_sfb[0];
                    printf("[FP4DBG][TARGET][SMEM][SM%u][k=%u m=%u] "
                           "A_bytes[0..7]=%02x %02x %02x %02x %02x %02x %02x %02x "
                           "B_bytes[0..7]=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                           smid, k_block_idx, m_block_idx,
                           a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3], a_ptr[4], a_ptr[5], a_ptr[6], a_ptr[7],
                           b_ptr[0], b_ptr[1], b_ptr[2], b_ptr[3], b_ptr[4], b_ptr[5], b_ptr[6], b_ptr[7]);
                    printf("[FP4DBG][TARGET][SMEM][SM%u][k=%u m=%u] "
                           "SFA_packs[0..3]=%08x %08x %08x %08x "
                           "SFB_packs[0..3]=%08x %08x %08x %08x\n",
                           smid, k_block_idx, m_block_idx,
                           sfa_ptr[0], sfa_ptr[1], sfa_ptr[2], sfa_ptr[3],
                           sfb_ptr[0], sfb_ptr[1], sfb_ptr[2], sfb_ptr[3]);
                }
                // -------------------------------------------------------------------------

                if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t p = 0; p < kSFPacksPerBlockK; ++p) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + p * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                            replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + p * (SF_BLOCK_M / 32) + i * 4);
                        }
                    }
                }
                __syncwarp();

                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W1][SM%u] pre mma_t k=%u m=%u stage=%u phase=%u\n",
                            smid, k_block_idx, m_block_idx, stage_idx, phase);

                using mma_t = cute::conditional_t<kNumMulticast == 1, SM100_MMA_MXF4NVF4_SS, SM100_MMA_MXF4NVF4_2x1SM_SS>;
                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(0));
                if (cute::elect_one_sync()) {
                    // Two-level loop: outer over K-atoms, inner over UMMA_K steps per atom.
                    // Each atom gets its own SF index to maintain 128-element dequant granularity.
                    #pragma unroll
                    for (uint32_t atom = 0; atom < NUM_K_ATOMS; ++atom) {
                        // Rebase B descriptor to where TMA placed this K-atom (FP4: half byte per element)
                        const uint32_t b_atom_offset_bytes = atom * LOAD_BLOCK_N * DESC_ATOM_K / 2;
                        const uint32_t b_atom_base = b_desc_base_lo + (b_atom_offset_bytes >> 4);

                        // Rebase A descriptor to where TMA placed this K-atom (FP4: half byte per element)
                        const uint32_t a_atom_offset_bytes = atom * LOAD_BLOCK_M * DESC_ATOM_K / 2;
                        const uint32_t a_atom_base = a_desc_base_lo + (a_atom_offset_bytes >> 4);

                        #pragma unroll
                        for (uint32_t ki = 0; ki < UMMA_ITERS_PER_ATOM; ++ki) {
                            // SF pack index across the full BLOCK_K: atom selects which K-atom,
                            // ki selects which UMMA step within the atom.
                            const uint32_t sf_pack_idx = atom * UMMA_ITERS_PER_ATOM + ki;
                            const uint32_t sf_id = ((k_block_idx * kSFAtomsPerBlockK + atom * (DESC_ATOM_K/16) + ki * (UMMA_K/16))) % kNumSFPerPack;
                            const auto& runtime_instr_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sf_id);

                            b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, fp4_input_element_t>(b_atom_base, 0, ki * UMMA_K / 2);
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                                a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, fp4_input_element_t>(a_atom_base, w * WAVE_BLOCK_M * DESC_ATOM_K / 2, ki * UMMA_K / 2);
                                mma_t::fma(a_desc, b_desc,
                                            accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                            (atom > 0 || ki > 0), // accumulate: false only for atom=0,ki=0
                                            runtime_instr_desc,
                                            kTmemStartColOfSFA + sf_pack_idx * (SF_BLOCK_M / 32) + w * (kNumUTCCPAlignedElems / 32),
                                            kTmemStartColOfSFB + sf_pack_idx * (SF_BLOCK_N / 32));
                            }
                        }
                    }
                }

                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W1][SM%u] pre empty_barrier_arrive k=%u m=%u accum_stage=%u accum_phase=%u\n",
                            smid, k_block_idx, m_block_idx, accum_stage_idx, accum_phase_idx);
                // Publish only after the last K tile of this M tile.
                empty_barrier_arrive();
            }
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_b[0]));
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; ++k_block_idx) {
            full_barriers_b[0]->wait(phase_b);
            phase_b ^= 1;
            const uint32_t sf_stage_in_group_idx = 0;
            if (sf_stage_in_group_idx == 0)
            {
                // printf("[FP8DBG][W2][SM%u] sf_stage_in_group_idx == 0 k=%u stage=%u phase=%u\n",
                #pragma unroll
                for (uint32_t p = 0; p < kSFPacksPerBlockK; ++p) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[0] + p * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                }
                cutlass::arch::fence_view_async_shared();
            }
            with_sf_full_barriers_b[0]->arrive(0u);

            for (uint32_t m_block_idx = scheduler.m_start; m_block_idx < scheduler.m_end; advance_pipeline(m_block_idx)) {
                // Wait TMA arrival
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W2][SM%u] pre full.wait k=%u m=%u stage=%u phase=%u\n",
                           smid, k_block_idx, m_block_idx, stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);
                if (should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][W2][SM%u] post full.wait k=%u m=%u stage=%u phase=%u\n",
                           smid, k_block_idx, m_block_idx, stage_idx, phase);

                // Transpose for UTCCP at certain stages
                if (sf_stage_in_group_idx == 0) {
                    #pragma unroll
                    for (uint32_t p = 0; p < kSFPacksPerBlockK; ++p) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                            utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + p * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                    }                  
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };

        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; ++k_block_idx) {
            // FP8 publishes tmem_full only on the last K tile, so epilogue work happens there.
            // if (k_block_idx != num_total_k_blocks - 1)
            //     continue;

            for (uint32_t m_block_idx = scheduler.m_start; m_block_idx < scheduler.m_end; ++m_block_idx, advance_accum_pipeline()) {
                // Wait UMMA arrival
                if (epilogue_warp_idx == 0 && should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][EPI0][SM%u] pre tmem_full.wait k=%u m=%u accum_stage=%u accum_phase=%u\n",
                           smid, k_block_idx, m_block_idx, accum_stage_idx, accum_phase_idx);
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();
                if (epilogue_warp_idx == 0 && should_log(k_block_idx, m_block_idx))
                    printf("[FP4DBG][EPI0][SM%u] post tmem_full.wait k=%u m=%u accum_stage=%u accum_phase=%u\n",
                           smid, k_block_idx, m_block_idx, accum_stage_idx, accum_phase_idx);

                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                        if (epilogue_warp_idx == 0)
                            cute::tma_store_wait<kNumTMAStoreStages - 1>();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        // const auto m_idx = m_block_idx * BLOCK_M + w * WAVE_BLOCK_M;
                        // const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(blockIdx.x * BLOCK_N + s * STORE_BLOCK_N);

                        // Todo: The pipeline stage
                        const auto base_local_m_idx = kGemmType == GemmType::MGroupedMasked 
                                                    ? (m_block_idx - scheduler.m_start) * BLOCK_M
                                                    : (m_block_idx * BLOCK_M);
                        const auto m_idx = kGemmType == GemmType::MGroupedMasked
                                         ? (scheduler.current_group_idx * shape_m + base_local_m_idx + w * WAVE_BLOCK_M)
                                         : (base_local_m_idx + w * WAVE_BLOCK_M);
                        const auto n_idx = blockIdx.x * BLOCK_N + s * STORE_BLOCK_N;

                        // Store into shared memory
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            // Source and destination memory address
                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +               // Accumulator offset
                                                w * BLOCK_N +                                          // Wave offset
                                                s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                            // Load from tensor memory, store into shared memory
                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                // For FP32 output, read and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                // Debug: print first 4 accumulator floats for CTA(0,0), lane 0, warp 0, w=0, s=0, i=0
                                if (i == 0 && w == 0 && s == 0 && lane_idx == 0 && epilogue_warp_idx == 0 &&
                                    m_block_idx == scheduler.m_start) {
                                    printf("[FP4DBG][OUT][SM%u][k=%u m=%u] tmem_out[0..3]:"
                                           " raw=%08x/%g  %08x/%g  %08x/%g  %08x/%g\n",
                                           smid, k_block_idx, m_block_idx,
                                           values[0], __uint_as_float(values[0]),
                                           values[1], __uint_as_float(values[1]),
                                           values[2], __uint_as_float(values[2]),
                                           values[3], __uint_as_float(values[3]));
                                }
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            } else {
                                // For BF16 output, read, cast and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr,
                                        cast_into_bf16_and_pack(values[0], values[1]),
                                        cast_into_bf16_and_pack(values[2], values[3]),
                                        cast_into_bf16_and_pack(values[4], values[5]),
                                        cast_into_bf16_and_pack(values[6], values[7]));
                                if (lane_idx == 0 && epilogue_warp_idx == 0 && (i == 0 || i == 4 || i == 5) &&
                                    static_cast<int>(blockIdx.x) == kDbgBlockX &&
                                    static_cast<int>(blockIdx.y) == kDbgBlockY &&
                                    static_cast<int>(k_block_idx) == kDbgKBlock &&
                                    static_cast<int>(m_block_idx) == kDbgMBlock) {
                                    printf("[FP4DBG][TARGET][EPI][SM%u] TMEM->SMEM: "
                                           "k=%u m_block=%u w=%u s=%u accum_stage=%u tma_stage=%u "
                                           "i=%u row=%u col=%u bank_group_idx=%u "
                                           "tmem_addr=%u m_idx=%u n_idx=%u "
                                           "vals=%08x %08x %08x %08x %08x %08x %08x %08x\n",
                                           smid, k_block_idx, m_block_idx, w, s,
                                           accum_stage_idx, tma_stage_idx,
                                           i, static_cast<uint32_t>(row), static_cast<uint32_t>(col),
                                           static_cast<uint32_t>(bank_group_index),
                                           tmem_addr, m_idx, n_idx,
                                           values[0], values[1], values[2], values[3],
                                           values[4], values[5], values[6], values[7]);
                                }
                            }
                        }

                        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
                        // NOTES: only the last stage needs to do this
                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Synchronize all threads and issue TMA
                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            if (lane_idx == 0 &&
                                static_cast<int>(blockIdx.x) == kDbgBlockX &&
                                static_cast<int>(blockIdx.y) == kDbgBlockY &&
                                static_cast<int>(k_block_idx) == kDbgKBlock &&
                                static_cast<int>(m_block_idx) == kDbgMBlock) {
                                printf("[FP4DBG][TARGET][EPI][SM%u] TMA store issue: "
                                       "k=%u m_block=%u w=%u s=%u m_idx=%u n_idx=%u group=%u\n",
                                       smid, k_block_idx, m_block_idx, w, s, m_idx, n_idx, scheduler.current_group_idx);
                            }
                            if constexpr (kGemmType == GemmType::Batched) {
                                using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                    cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                                cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx],
                                                n_idx, m_idx, scheduler.current_group_idx);
                            } else {
                                if (k_block_idx == 0)
                                {
                                    using cute_tma_t = cute::conditional_t<false,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                                else
                                {
                                    using cute_tma_t = cute::conditional_t<true,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                                
                            }
                            cute::tma_store_arrive();
                        }
                    }
                }
            }
        }
        // Deallocate tensor memory by the last UMMA store warp
        // NOTES: warp 0 is waiting TMA store
        if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1)
            Allocator().free(0, kNumTmemCols);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
