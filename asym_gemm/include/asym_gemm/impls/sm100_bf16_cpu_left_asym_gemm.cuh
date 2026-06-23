#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm100_utils.cuh>

namespace asym_gemm {

using namespace asym_gemm::sm100;

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
          uint64_t kTensorCoreUtilControl,
          bool kCompactMBlockGrid,
          bool kPairOutput>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_cpu_left_asym_gemm_impl(uint32_t* offsets, uint32_t* experts,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b_pair,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd_pair) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
    DG_STATIC_ASSERT(kNumStages == 2, "This simplified setup requires kNumStages == 2");
    DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "CPU-left BF16 kernel only supports m-grouped contiguous");
    DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K, "CPU-left BF16 kernel requires K-major operands");
    DG_STATIC_ASSERT(kNumMulticast == 1, "CPU-left BF16 kernel does not support multicast yet");

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    DG_STATIC_ASSERT(sizeof(cutlass::bfloat16_t) * LAYOUT_AD_M % kSwizzleAMode == 0, "Invalid swizzle A mode");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

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
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_A_NUM_TILES = 1;
    constexpr uint32_t SMEM_B_NUM_TILES = kNumStages;
    constexpr uint32_t SMEM_A_SIZE = SMEM_A_SIZE_PER_STAGE * SMEM_A_NUM_TILES;
    constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE * SMEM_B_NUM_TILES;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE, "Memory Out of bound for UMMA");

    // Automatically deduce the number of epilogue stages (1 or 2), according to the tensor memory size
    // TODO: test cases of `kNumMWaves == 2 and kNumEpilogueStages == 2`
    constexpr uint32_t kNumEpilogueStages = 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // D/A/B shared memory
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + SMEM_A_SIZE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + SMEM_A_SIZE + SMEM_B_SIZE);
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });

    // Singleton A tile barriers. CPU-left keeps one host-resident A tile in
    // SMEM per (M tile, K tile), while the CUDA-resident B operand is pipelined.
    auto full_barriers_a            = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2); });
    auto empty_barriers_a           = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 1); });

    // Extra barriers for the singleton A tile (full/empty).
    int extend_barrier = 2;
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + extend_barrier;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + extend_barrier + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // A is a single persistent SMEM tile, not a staged pipeline.
        full_barriers_a[0]->init(kNumMulticast);
        empty_barriers_a[0]->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, experts, offsets);
    // Sentinel block (inactive expert or empty M range): skip without entering
    // any TMA / barrier wait paths. All CTAs in a cluster share blockIdx.y, so
    // they all early-exit together and don't deadlock cluster-wide barriers.
    // The init phase already ran Allocator().allocate() unconditionally, so we
    // must release the TMEM before returning, otherwise subsequent kernels see
    // "tensor memory not completely freed". Use the same one-warp-frees pattern
    // as the normal exit path.
    const uint32_t block_m_iter = kCompactMBlockGrid ? (scheduler.m_start + blockIdx.x) : blockIdx.x;
    if (scheduler.m_start >= scheduler.m_end or block_m_iter < scheduler.m_start or block_m_iter >= scheduler.m_end) {
        if (warp_idx == 2) {
            const auto tmem_ptr = ld_shared(tmem_ptr_in_smem);
            Allocator().free(tmem_ptr, kNumTmemCols);
        }
        return;
    }

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0, phase_a = 0;
    auto advance_pipeline = [&](uint32_t& block_idx) {
        ++block_idx;
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    uint32_t block_k = ceil_div_device(shape_k, BLOCK_K);

    // Dispatch warps into different roles. CPU-left owns one M tile per CTA
    // and reuses that host-resident A tile across all CUDA-resident B/N tiles.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        const uint32_t num_n_blocks_per_output = ceil_div_device(shape_n, BLOCK_N);
        const uint32_t num_n_blocks = kPairOutput ? num_n_blocks_per_output * 2 : num_n_blocks_per_output;
        const uint32_t m_idx = block_m_iter * BLOCK_M;

        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            const uint32_t k_idx = block_k_iter * BLOCK_K;

            empty_barriers_a[0]->wait(phase_a ^ 1);
            phase_a ^= 1;
            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, false>(
                &tensor_map_a, full_barriers_a[0], smem_a[0], k_idx, m_idx, kNumMulticast, 0);

            if (is_leader_cta) {
                full_barriers_a[0]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * kNumMulticast);
            } else {
                full_barriers_a[0]->arrive(0u);
            }

            for (uint32_t block_n_iter = 0; block_n_iter < num_n_blocks; advance_pipeline(block_n_iter)) {
                const uint32_t pair_output_idx = kPairOutput ? block_n_iter / num_n_blocks_per_output : 0;
                const uint32_t local_block_n_iter = kPairOutput ? block_n_iter - pair_output_idx * num_n_blocks_per_output : block_n_iter;
                const uint32_t n_idx = local_block_n_iter * BLOCK_N + shape_n * scheduler.current_group_idx;
                const auto* selected_tensor_map_b = (kPairOutput && pair_output_idx != 0) ? &tensor_map_b_pair : &tensor_map_b;

                empty_barriers[stage_idx]->wait(phase ^ 1);
                tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, false>(
                    selected_tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, kNumMulticast, 0);

                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp. A is loaded once per (M tile, K tile) and reused
        // across every N tile; B is double-buffered from CUDA memory across N.
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::bfloat16_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        constexpr uint32_t SWIZZLE_ATOM_K = kSwizzleBMode / sizeof(cutlass::bfloat16_t);
        constexpr uint32_t DESC_ATOM_K = (BLOCK_ATOM_K > SWIZZLE_ATOM_K) ? SWIZZLE_ATOM_K : BLOCK_ATOM_K;
        constexpr uint32_t NUM_K_ATOMS = BLOCK_ATOM_K / DESC_ATOM_K;
        constexpr uint32_t UMMA_ITERS_PER_ATOM = DESC_ATOM_K / UMMA_K;
        constexpr uint32_t B_DESC_K = DESC_ATOM_K;
        auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, DESC_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, B_DESC_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = a_desc.lo;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        auto umma_arrive = [](const uint64_t* barrier) {
            cutlass::arch::umma_arrive(barrier);
        };

        uint32_t accum_stage_idx = 0, accum_phase_idx = 0;
        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };

        const uint32_t num_n_blocks_per_output = ceil_div_device(shape_n, BLOCK_N);
        const uint32_t num_n_blocks = kPairOutput ? num_n_blocks_per_output * 2 : num_n_blocks_per_output;
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            full_barriers_a[0]->wait(phase_a);
            phase_a ^= 1;
            tcgen05_after_thread_sync();

            for (uint32_t block_n_iter = 0; block_n_iter < num_n_blocks; advance_pipeline(block_n_iter)) {
                full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();

                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                using mma_t = SM100_MMA_F16BF16_SS;
                const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(0));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t atom = 0; atom < NUM_K_ATOMS; ++atom) {
                        const uint32_t b_atom_offset_bytes = atom * LOAD_BLOCK_N * DESC_ATOM_K * sizeof(cutlass::bfloat16_t);
                        const uint32_t b_atom_base = b_desc_base_lo + (b_atom_offset_bytes >> 4);
                        const uint32_t a_atom_offset_bytes = atom * LOAD_BLOCK_M * DESC_ATOM_K * sizeof(cutlass::bfloat16_t);
                        const uint32_t a_atom_base = a_desc_base_lo + (a_atom_offset_bytes >> 4);

                        #pragma unroll
                        for (uint32_t ki = 0; ki < UMMA_ITERS_PER_ATOM; ++ki) {
                            b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                b_atom_base, 0, ki * UMMA_K);
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                                a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                    a_atom_base, w * WAVE_BLOCK_M * DESC_ATOM_K, ki * UMMA_K);
                                mma_t::fma(a_desc, b_desc,
                                           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                           (atom > 0 || ki > 0),
                                           runtime_instr_desc);
                            }
                        }
                    }
                }

                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));

                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    constexpr static uint64_t kNumUMMACycles = (2ull * LAYOUT_AD_M * kNumMWaves * BLOCK_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto& start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }

                advance_accum_pipeline();
            }

            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_a[0]));
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups. Store one fixed M tile for every N tile.
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        uint32_t accum_stage_idx = 0, accum_phase_idx = 0, tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };
        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };

        const uint32_t num_n_blocks_per_output = ceil_div_device(shape_n, BLOCK_N);
        const uint32_t num_n_blocks = kPairOutput ? num_n_blocks_per_output * 2 : num_n_blocks_per_output;
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            for (uint32_t block_n_iter = 0; block_n_iter < num_n_blocks; ++block_n_iter, advance_accum_pipeline()) {
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++w) {
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline()) {
                        if (epilogue_warp_idx == 0) {
                            cute::tma_store_wait<kNumTMAStoreStages - 1>();
                        }
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        const auto m_idx = BLOCK_M * block_m_iter + w * WAVE_BLOCK_M;
                        const uint32_t pair_output_idx = kPairOutput ? block_n_iter / num_n_blocks_per_output : 0;
                        const uint32_t local_block_n_iter = kPairOutput ? block_n_iter - pair_output_idx * num_n_blocks_per_output : block_n_iter;
                        const auto n_idx = local_block_n_iter * BLOCK_N + s * STORE_BLOCK_N;
                        const auto* selected_tensor_map_cd = (kPairOutput && pair_output_idx != 0) ? &tensor_map_cd_pair : &tensor_map_cd;

                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +
                                                 w * BLOCK_N +
                                                 s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            } else {
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
                            }
                        }

                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            if (block_k_iter == 0) {
                                cute::SM90_TMA_STORE_2D::copy(selected_tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                            } else {
                                cute::SM90_TMA_REDUCE_ADD_2D::copy(selected_tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                            }
                            cute::tma_store_arrive();
                        }
                    }
                }
            }
        }

        if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1) {
            const auto tmem_ptr = ld_shared(tmem_ptr_in_smem);
            Allocator().free(tmem_ptr, kNumTmemCols);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
