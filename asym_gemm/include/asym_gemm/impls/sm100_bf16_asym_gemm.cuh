#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>

#include <asym_gemm/common/asymScheduler.cuh>
#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/sm100_utils.cuh>

#ifndef ASYM_BF16_KERNEL_NAME
#define ASYM_BF16_KERNEL_NAME sm100_bf16_asym_gemm_impl
#endif

#ifndef ASYM_BF16_KERNEL_EXTRA_ARGS
#define ASYM_BF16_KERNEL_EXTRA_ARGS
#endif

#ifndef ASYM_BF16_ROUTE_GATHER_LEFT
#define ASYM_BF16_ROUTE_GATHER_LEFT 0
#endif

#ifndef ASYM_BF16_ROUTE_SCATTER_ADD
#define ASYM_BF16_ROUTE_SCATTER_ADD 0
#endif

namespace asym_gemm {

using namespace asym_gemm::sm100;

__device__ __forceinline__ float bf16_to_float(cutlass::bfloat16_t v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ float qwen3_moe_route_weight_or_one(
    const void* route_weights,
    uint32_t route_weights_is_bf16,
    uint32_t route_weighted,
    uint32_t row) {
    if (route_weighted == 0u or route_weights == nullptr)
        return 1.0f;
    if (route_weights_is_bf16 != 0u)
        return static_cast<float>(reinterpret_cast<const cutlass::bfloat16_t*>(route_weights)[row]);
    return reinterpret_cast<const float*>(route_weights)[row];
}

template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__ uint32_t qwen3_moe_k_major_smem_offset(uint32_t row, uint32_t col) {
    if constexpr (kSwizzleMode == 128) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        return static_cast<uint32_t>(layout(row, col));
    } else if constexpr (kSwizzleMode == 64) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW64_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        return static_cast<uint32_t>(layout(row, col));
    } else if constexpr (kSwizzleMode == 32) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW32_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        return static_cast<uint32_t>(layout(row, col));
    } else {
        return row * BLOCK_K + col;
    }
}

template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__ void qwen3_moe_store_k_major_smem(
    dtype_t* smem,
    uint32_t row,
    uint32_t col,
    dtype_t value) {
    if constexpr (kSwizzleMode == 128) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW128_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        auto tensor = cute::make_tensor(cute::make_smem_ptr(smem), layout);
        tensor(row, col) = value;
    } else if constexpr (kSwizzleMode == 64) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW64_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        auto tensor = cute::make_tensor(cute::make_smem_ptr(smem), layout);
        tensor(row, col) = value;
    } else if constexpr (kSwizzleMode == 32) {
        auto layout = cute::tile_to_shape(
            cute::GMMA::Layout_K_SW32_Atom<dtype_t>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        auto tensor = cute::make_tensor(cute::make_smem_ptr(smem), layout);
        tensor(row, col) = value;
    } else {
        smem[row * BLOCK_K + col] = value;
    }
}

// 16-byte (8 x bf16) variant of the swizzled store: every SW32/SW64/SW128
// swizzle permutes 16-byte units but keeps their contents contiguous, so 8
// consecutive k of one row (col % 8 == 0) land as one int4 at the first
// element's swizzled address. This is what makes the gather-left staging
// vectorizable (v2 2026-07-28): the scalar per-element form was the routed
// kernel's dominant cost at training M.
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
__device__ __forceinline__ void qwen3_moe_store_k_major_smem_vec8(
    dtype_t* smem,
    uint32_t row,
    uint32_t col,
    int4 value) {
    static_assert(sizeof(dtype_t) == 2, "vec8 store expects 16-bit elements");
    if constexpr (kSwizzleMode == 128 || kSwizzleMode == 64 || kSwizzleMode == 32) {
        auto layout = cute::tile_to_shape(
            cute::conditional_t<kSwizzleMode == 128,
                                cute::GMMA::Layout_K_SW128_Atom<dtype_t>,
                                cute::conditional_t<kSwizzleMode == 64,
                                                    cute::GMMA::Layout_K_SW64_Atom<dtype_t>,
                                                    cute::GMMA::Layout_K_SW32_Atom<dtype_t>>>{},
            cute::make_shape(cute::Int<BLOCK_MN>{}, cute::Int<BLOCK_K>{}));
        auto tensor = cute::make_tensor(cute::make_smem_ptr(smem), layout);
        *reinterpret_cast<int4*>(&tensor(row, col)) = value;
    } else {
        *reinterpret_cast<int4*>(&smem[row * BLOCK_K + col]) = value;
    }
}

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
ASYM_BF16_KERNEL_NAME(uint32_t* offsets, uint32_t* experts,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd
                     ASYM_BF16_KERNEL_EXTRA_ARGS) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    constexpr bool kRouteGatherLeft = ASYM_BF16_ROUTE_GATHER_LEFT != 0;
    constexpr bool kRouteScatterAdd = ASYM_BF16_ROUTE_SCATTER_ADD != 0;
    if constexpr (kRouteGatherLeft)
        DG_STATIC_ASSERT(kMajorA == cute::UMMA::Major::K, "Routed gather-left requires K-major A");
    if constexpr (kRouteScatterAdd)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Routed scatter-add requires FP32 output accumulation");

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

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

#ifdef ASYM_BF16_EP_QUEUED
    // sEP (gb200_ep.md E3): claim a work item from the shared coherent host counters at
    // KERNEL ENTRY — before ANY barrier/TMEM initialization — so a CTA that finds the list
    // empty exits at ~atomic cost. ep_queue: [0]=claimed, [1]=head_taken, [2]=tail_taken.
    // Linearizable: claims are capped at ep_total_items, so front [0..head) and back
    // (total-tail..total-1] can never overlap.
    static_assert(kNumMulticast == 1, "sEP queued kernel must not cluster-launch (HC-EP2)");
    __shared__ int s_ep_item;
    if (threadIdx.x == 0) {
        int ep_ticket = atomicAdd_system(ep_queue + 0, 1);
        int ep_claimed_item = -1;
        if (ep_ticket < static_cast<int>(ep_total_items)) {
            ep_claimed_item = (ep_side == 0)
                ? atomicAdd_system(ep_queue + 1, 1)
                : static_cast<int>(ep_total_items) - 1 - atomicAdd_system(ep_queue + 2, 1);
        }
        s_ep_item = ep_claimed_item;
    }
    __syncthreads();
    const int ep_item = s_ep_item;
    if (ep_item < 0)
        return;
#endif

    // if (threadIdx.x == 0 && blockIdx.y == 3)
    //     printf("blockIdx.x: %d, blockIdx.y: %d \n", blockIdx.x, blockIdx.y);           

    // if (blockIdx.y != 4)
    //     return;

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
    constexpr uint32_t SMEM_B_NUM_TILES = 1;
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
#if ASYM_BF16_ROUTE_GATHER_LEFT
#else
        cute::prefetch_tma_descriptor(&tensor_map_a);
#endif
        cute::prefetch_tma_descriptor(&tensor_map_b);
#if ASYM_BF16_ROUTE_SCATTER_ADD
#else
        cute::prefetch_tma_descriptor(&tensor_map_cd);
#endif
#ifdef ASYM_BF16_EP_STEAL
        cute::prefetch_tma_descriptor(&tensor_map_a_peer);
        cute::prefetch_tma_descriptor(&tensor_map_cd_peer);
#endif
    }

    // D/A/B shared memory
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE);
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    
    auto full_barriers_b            = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2); });
    auto empty_barriers_b           = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 1); });
    // auto tmem_full_barriers_b       = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 2); });
    // auto tmem_empty_barriers_b      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages * 2 + 3); });
    
    // NOTE: Extra barriers for B (full/empty).
    int extend_barrier = 2;
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + extend_barrier;

    // Add near the top of the kernel, after thread/warp setup:
    const bool debug_print = false;
    //  (threadIdx.x % 32 == 0 && blockIdx.x == 0);


    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("information about tensor memory, kNumStages: %d, kNumEpilogueStages: %d, kNumTMAStoreStages: %d, kNumAccumTmemCols: %d, kNumTmemCols: %d \n", kNumStages, kNumEpilogueStages, kNumTMAStoreStages, kNumAccumTmemCols, kNumTmemCols);
    // }

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

        // init barriers for B 
        // Arrive only at the leader CTA
        full_barriers_b[0]->init(kNumMulticast);
        // Arrive at all CTAs
        empty_barriers_b[0]->init(1);
        // tmem_full_barriers_b[0]->init(1);
        // tmem_empty_barriers_b[0]->init(kNumMulticast * kNumUMMAStoreThreads);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
#ifdef ASYM_BF16_EP_QUEUED
    const uint32_t ep_num_n_blocks = ceil_div_device(shape_n, BLOCK_N);
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, experts, offsets,
        static_cast<uint32_t>(ep_item) / ep_num_n_blocks,
        static_cast<uint32_t>(ep_item) % ep_num_n_blocks);
#else
    auto scheduler = asymScheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, experts, offsets);
#endif
#ifdef ASYM_BF16_EP_STEAL
    // sEP steal (fix_gb200_ep.md S2b): items in the PEER's section of the union list
    // read A from the fabric pack and store D to the fabric staging (sysmem TMA both
    // ways — the B path proves host descriptors). ep_n_own is in SEGMENT units: n_blk
    // is a JIT tile choice the host cannot know, and the section boundary is a segment
    // boundary by construction. Side 0 owns the FRONT section, side 1 the BACK.
    const uint32_t ep_segment = static_cast<uint32_t>(ep_item) / ep_num_n_blocks;
    const bool ep_local = (ep_side == 0) ? (ep_segment < ep_n_own) : (ep_segment >= ep_n_own);
    const cute::TmaDescriptor* ep_desc_a = ep_local ? &tensor_map_a : &tensor_map_a_peer;
    const cute::TmaDescriptor* ep_desc_cd = ep_local ? &tensor_map_cd : &tensor_map_cd_peer;
#define ASYM_DESC_A ep_desc_a
#define ASYM_DESC_CD ep_desc_cd
#else
#define ASYM_DESC_A (&tensor_map_a)
#define ASYM_DESC_CD (&tensor_map_cd)
#endif
    // Sentinel block (inactive expert or empty M range): skip without entering
    // any TMA / barrier wait paths. All CTAs in a cluster share blockIdx.y, so
    // they all early-exit together and don't deadlock cluster-wide barriers.
    // The init phase already ran Allocator().allocate() unconditionally — we
    // must release the TMEM before returning, otherwise subsequent kernels see
    // "tensor memory not completely freed". Use the same one-warp-frees pattern
    // as the normal exit path.
    if (scheduler.m_start >= scheduler.m_end) {
        if (warp_idx == 2) {
            const auto tmem_ptr = ld_shared(tmem_ptr_in_smem);
            Allocator().free(tmem_ptr, kNumTmemCols);
        }
        return;
    }

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0, phase_b = 0;
    auto advance_pipeline = [&](uint32_t& block_idx) {
        ++ block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    uint32_t block_k = ceil_div_device(shape_k, BLOCK_K);
    // uint32_t block_k = 1;
    uint32_t n_idx = scheduler.n_idx;

    // if (threadIdx.x == 0) {
    //     printf("information blockIdx.x: %d, blockIdx.y: %d, Stages: %d, BLOCK_K: %d, BLOCK_N: %d, BLOCK_M: %d, block_k: %d, m_start: %d, m_end: %d \n", blockIdx.x, blockIdx.y, kNumStages, BLOCK_K, BLOCK_N, BLOCK_M, block_k, scheduler.m_start, scheduler.m_end);
    // }

    // Dispatch warps into different roles
#if ASYM_BF16_ROUTE_GATHER_LEFT
    if (warp_idx == 0) {
#else
    if (warp_idx == 0 and cute::elect_one_sync()) {
#endif
        // TMA load warp
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
            uint32_t k_idx = block_k_iter * BLOCK_K;
            const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
            uint32_t b_n_idx = n_idx;
            uint32_t b_k_idx = k_idx;
            if constexpr (kGemmType == GemmType::MGroupedContiguous and kMajorB == cute::UMMA::Major::MN) {
                b_n_idx = scheduler.n_blk * BLOCK_N;
                b_k_idx += scheduler.current_group_idx * shape_k;
            }
            
            empty_barriers_b[0]->wait(phase_b ^ 1);
            phase_b ^= 1;
            if (cute::elect_one_sync()) {
                if constexpr (kMajorB == cute::UMMA::Major::K)
                {
                    tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], k_idx, n_idx, kNumMulticast, batch_idx);
                    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                    //     printf("block_k_iter: %d \n", block_k_iter);
                    // }
                }
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers_b[0], smem_b[0], b_n_idx, b_k_idx, kNumMulticast, batch_idx);
            }

            // if (lane_idx == 0 and block_k_iter == 0 and n_idx == 0 and blockIdx.y == 0) {
            //     printf("DBG_BF16_ASYM LOAD_B warp=%u lane=%u k_idx=%u n_idx=%u\n",
            //            (unsigned)warp_idx, (unsigned)lane_idx, (unsigned)k_idx, (unsigned)n_idx);
            // }

            if (cute::elect_one_sync()) {
                if (is_leader_cta) {
                    full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                } else {
                    full_barriers_b[0]->arrive(0u);
                }
            }

            const auto& num_total_k_blocks = ceil_div_device(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter)) {
                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = (kGemmType == GemmType::MGroupedMasked) 
                                 ? (scheduler.current_group_idx * shape_m + (block_m_iter - scheduler.m_start) * BLOCK_M)
                                 : (block_m_iter * BLOCK_M);
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     auto bar_ptr  = full_barriers[stage_idx];
                //     auto smem_ptr = smem_a[stage_idx];

                //     printf("kNumMulticast=%u\n", (unsigned)kNumMulticast);

                //     printf("kIsMulticastOnA=%u phase=%u k_idx=%u m_idx=%u\n",
                //             (unsigned)stage_idx, (unsigned)phase, (unsigned)k_idx, (unsigned)m_idx);

                //     printf("TMA A: stage=%u phase=%u k_idx=%u m_idx=%u\n",
                //             (unsigned)stage_idx, (unsigned)phase, (unsigned)k_idx, (unsigned)m_idx);

                //     printf("  barrier=%p  smem_dst=%p\n", (void*)bar_ptr, (void*)smem_ptr);

                //     printf("  align: barrier%%16=%llu smem%%16=%llu\n",
                //             (unsigned long long)((uintptr_t)bar_ptr & 0xFULL),
                //             (unsigned long long)((uintptr_t)smem_ptr & 0xFULL));
                // }

                // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Within TMA load warp, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);                
#if ASYM_BF16_ROUTE_GATHER_LEFT
                DG_STATIC_ASSERT(kNumMulticast == 1, "Routed gather-left currently supports single-CTA loads only");
                constexpr uint32_t DESC_ATOM_K_FOR_GATHER = (BLOCK_K > (kSwizzleAMode / sizeof(cutlass::bfloat16_t)))
                    ? (kSwizzleAMode / sizeof(cutlass::bfloat16_t)) : BLOCK_K;
                constexpr uint32_t NUM_GATHER_K_ATOMS = BLOCK_K / DESC_ATOM_K_FOR_GATHER;
                // v2 (2026-07-28): vectorized gather staging. The scalar form
                // did a token-index load, a scale lookup, and a 2-byte
                // load/store PER ELEMENT (65K iterations per tile through one
                // warp) and dominated the routed GEMM's runtime at training M.
                // Here each lane moves 8 consecutive k of one row per
                // iteration: one int4 global load, one fused fp32 scale
                // (identical multiply-then-round math, so results are
                // bit-identical), one int4 swizzled store — 8x fewer
                // iterations and per-vector rather than per-element index
                // arithmetic. Real shapes always take the vector path; the
                // scalar tail keeps exotic strides correct.
                constexpr uint32_t GVEC = 8;
                DG_STATIC_ASSERT(DESC_ATOM_K_FOR_GATHER % GVEC == 0, "gather atom must be 16-byte divisible");
                constexpr uint32_t GATHER_VECS_PER_ROW = DESC_ATOM_K_FOR_GATHER / GVEC;
                #pragma unroll
                for (uint32_t atom = 0; atom < NUM_GATHER_K_ATOMS; ++atom) {
                    for (uint32_t linear = lane_idx; linear < LOAD_BLOCK_M * GATHER_VECS_PER_ROW; linear += 32) {
                        const uint32_t local_m = linear / GATHER_VECS_PER_ROW;
                        const uint32_t vj = linear - local_m * GATHER_VECS_PER_ROW;
                        const uint32_t route_row = m_idx + local_m;
                        const uint32_t k_col = k_idx + atom * DESC_ATOM_K_FOR_GATHER + vj * GVEC;
                        const uint32_t out_col = atom * DESC_ATOM_K_FOR_GATHER + vj * GVEC;
                        if (route_row < shape_m and k_col + GVEC <= route_gather_stride) {
                            const int64_t token_row = route_token_indices[route_row];
                            int4 vec = *reinterpret_cast<const int4*>(
                                &route_gather_left[static_cast<uint64_t>(token_row) * route_gather_stride + k_col]);
                            if (route_weighted) {
                                const float scale = qwen3_moe_route_weight_or_one(
                                    route_weights, route_weights_is_bf16, route_weighted, route_row);
                                auto* halves = reinterpret_cast<__nv_bfloat162*>(&vec);
                                #pragma unroll
                                for (uint32_t h = 0; h < GVEC / 2; ++h) {
                                    float2 f = __bfloat1622float2(halves[h]);
                                    f.x *= scale;
                                    f.y *= scale;
                                    halves[h] = __float22bfloat162_rn(f);
                                }
                            }
                            qwen3_moe_store_k_major_smem_vec8<
                                LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t>(
                                    smem_a[stage_idx], local_m, out_col, vec);
                        } else {
                            #pragma unroll
                            for (uint32_t l = 0; l < GVEC; ++l) {
                                const uint32_t kc = k_col + l;
                                cutlass::bfloat16_t value = cutlass::bfloat16_t(0.0f);
                                if (route_row < shape_m and kc < route_gather_stride) {
                                    const int64_t token_row = route_token_indices[route_row];
                                    const float scale = qwen3_moe_route_weight_or_one(
                                        route_weights, route_weights_is_bf16, route_weighted, route_row);
                                    value = cutlass::bfloat16_t(
                                        static_cast<float>(route_gather_left[static_cast<uint64_t>(token_row) * route_gather_stride + kc]) * scale);
                                }
                                qwen3_moe_store_k_major_smem<
                                    LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t>(
                                        smem_a[stage_idx], local_m, out_col + l, value);
                            }
                        }
                    }
                }
                __syncwarp();
                cutlass::arch::fence_view_async_shared();
#else
                {
                    if (cute::elect_one_sync()) {
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                                ASYM_DESC_A, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, kNumMulticast, batch_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                                ASYM_DESC_A, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_idx, kNumMulticast, batch_idx);
                    }
                }
#endif

                // if (lane_idx == 0 and block_k_iter == 0 and m_idx == 0 and n_idx == 0 and blockIdx.y == 0) {
                //     printf("DBG_BF16_ASYM LOAD_A warp=%u lane=%u stage=%u phase=%u m_idx=%u k_idx=%u\n",
                //            (unsigned)warp_idx, (unsigned)lane_idx, (unsigned)stage_idx, (unsigned)phase,
                //            (unsigned)m_idx, (unsigned)k_idx);
                // }
            
                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE;
#if ASYM_BF16_ROUTE_GATHER_LEFT
                if (cute::elect_one_sync())
                    full_barriers[stage_idx]->arrive(0u);
#else
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
#endif
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::bfloat16_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M, UMMA_N, kMajorA, kMajorB>();
        
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 291: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;

        // For multi-atom K (block_k > swizzle atom), create descriptors per-atom
        // so LBO=0, and manually rebase between atoms in the inner loop.
        constexpr uint32_t SWIZZLE_ATOM_K = kSwizzleBMode / sizeof(cutlass::bfloat16_t);  // 64 for BF16/SW128
        constexpr uint32_t DESC_ATOM_K = (BLOCK_ATOM_K > SWIZZLE_ATOM_K) ? SWIZZLE_ATOM_K : BLOCK_ATOM_K;
        constexpr uint32_t NUM_K_ATOMS = BLOCK_ATOM_K / DESC_ATOM_K;  // 2 for block_k=128, 1 for block_k=64
        constexpr uint32_t UMMA_ITERS_PER_ATOM = DESC_ATOM_K / UMMA_K;  // 4 (64/16)

        // MN-major B stores multiple N swizzle atoms separated by the full TMA K span.
        // K-major B can keep the descriptor local to one K atom.
        constexpr uint32_t B_DESC_K = (kMajorB == cute::UMMA::Major::MN) ? BLOCK_K : DESC_ATOM_K;
        auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, DESC_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, B_DESC_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 300: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // UMMA and empty barrier arrival alias
        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };

        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction 327: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }

        uint32_t accum_stage_iter = 0, accum_stage_idx = 0, accum_phase_idx = 0;

        auto empty_barrier_arrive = [&]() {
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

            // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
            // todo: stage_idx should be accum_stage_idx
            umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
        };


        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };
        // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("the MMA warp, before iteraction: stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // uint32_t accum_stage_iter = 0;
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            full_barriers_b[0]->wait(phase_b);
            // if (threadIdx.x == 32 && block_k_iter == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            //     printf("smem_b[0] first4=%f,%f,%f,%f\n",
            //            bf16_to_float(smem_b[0][0]),
            //            bf16_to_float(smem_b[0][1]),
            //            bf16_to_float(smem_b[0][2]),
            //            bf16_to_float(smem_b[0][3]));
            // }
            phase_b ^= 1;
            tcgen05_after_thread_sync();

            // After: full_barriers_b[0]->wait(phase_b);
            if (debug_print && block_k_iter == 0) {
                // Print first few elements of B in shared memory
                auto* b_ptr = reinterpret_cast<nv_bfloat16*>(smem_b[0]);
                printf("B smem[0..7]: ");
                for (int i = 0; i < 8; i++)
                    printf("%.4f ", __bfloat162float(b_ptr[i]));
                printf("\n");
                // Also print elements at the 64-element boundary (start of 2nd K-atom)
                printf("B smem[64..71]: ");
                for (int i = 64; i < 72; i++)
                    printf("%.4f ", __bfloat162float(b_ptr[i]));
                printf("\n");
            }

            // Launch MMAs
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; advance_pipeline(block_m_iter), advance_accum_pipeline()) {
                // auto accum_stage_idx = accum_stage_iter % kNumEpilogueStages;
                // auto accum_phase_idx = (accum_stage_iter / kNumEpilogueStages) & 1;

                // Wait tensor memory empty barrier arrival
                // wait the output in tensor memory has been written to HBM
                // if (threadIdx.x == 32 && blockIdx.x == 0) {
                //     printf("the MMA warp, before wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("the MMA warp 352, after tmem_empty_barriers wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                tcgen05_after_thread_sync();
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("the MMA warp 356, after tmem_empty_barriers wait: block_m_iter: %d, stage_idx: %d, phase: %d \n", block_m_iter, stage_idx, phase);
                // }
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();

                // if (lane_idx == 0 and block_k_iter == 0 and block_m_iter == 0 and blockIdx.x == 0 and blockIdx.y == 0) {
                //     const uint32_t a_row_stride = LOAD_BLOCK_M;
                //     const uint32_t b_row_stride = LOAD_BLOCK_N;
                //     printf("DBG_BF16_ASYM LOAD_A_ALL_COLS block_k_iter=%d stage=%u\n",
                //            block_k_iter, (unsigned)stage_idx);
                //     for (uint32_t col = 0; col < BLOCK_K; ++col) {
                //         const uint32_t idx_r0 = col * a_row_stride + 0;
                //         const uint32_t idx_r1 = col * a_row_stride + 1;
                //         printf("DBG_BF16_ASYM LOAD A r0 c=%u v=%f | r1 c=%u v=%f\n",
                //                (unsigned)col, bf16_to_float(smem_a[stage_idx][idx_r0]),
                //                (unsigned)col, bf16_to_float(smem_a[stage_idx][idx_r1]));
                //     }
                //     printf("DBG_BF16_ASYM LOAD B r0=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //            bf16_to_float(smem_b[0][0]), bf16_to_float(smem_b[0][1]),
                //            bf16_to_float(smem_b[0][2]), bf16_to_float(smem_b[0][3]),
                //            bf16_to_float(smem_b[0][4]), bf16_to_float(smem_b[0][5]),
                //            bf16_to_float(smem_b[0][6]), bf16_to_float(smem_b[0][7]));
                //     printf("DBG_BF16_ASYM LOAD B r1=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //            bf16_to_float(smem_b[0][b_row_stride + 0]), bf16_to_float(smem_b[0][b_row_stride + 1]),
                //            bf16_to_float(smem_b[0][b_row_stride + 2]), bf16_to_float(smem_b[0][b_row_stride + 3]),
                //            bf16_to_float(smem_b[0][b_row_stride + 4]), bf16_to_float(smem_b[0][b_row_stride + 5]),
                //            bf16_to_float(smem_b[0][b_row_stride + 6]), bf16_to_float(smem_b[0][b_row_stride + 7]));
                // }

                // ++accum_stage_iter;
                // if (threadIdx.x == 32 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("after wait the MMA warp, after sync: block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }

                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 32)
                //     printf("Asym before UMMA tmem_empty_barriers block_m_iter=%d \n", block_m_iter);

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, SM100_MMA_F16BF16_SS, SM100_MMA_F16BF16_2x1SM_SS>;
                const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(0));
                if (cute::elect_one_sync()) {

                    if (debug_print && block_k_iter == 0) {
                        printf("BLOCK_K=%d, DESC_ATOM_K=%d, NUM_K_ATOMS=%d, UMMA_K=%d, UMMA_ITERS_PER_ATOM=%d\n",
                            BLOCK_K, DESC_ATOM_K, NUM_K_ATOMS, UMMA_K, UMMA_ITERS_PER_ATOM);
                        printf("b_desc_base_lo=0x%x, a_desc_base_lo=0x%x\n",
                            b_desc_base_lo, a_desc_base_lo);
                    }

                    // Two-level loop: outer over K-atoms, inner over UMMA_K steps per atom.
                    // K-major B rebases the descriptor per K atom. MN-major B keeps the full-K
                    // descriptor span because multiple N swizzle atoms are laid out full-K apart.
                    #pragma unroll
                    for (uint32_t atom = 0; atom < NUM_K_ATOMS; ++atom) {
                        // Rebase B descriptor to where TMA placed this K-atom
                        // B is K-major: TMA placed atom `i` at smem offset i * LOAD_BLOCK_N * DESC_ATOM_K elements
                        const uint32_t b_atom_offset_bytes = atom * LOAD_BLOCK_N * DESC_ATOM_K * sizeof(cutlass::bfloat16_t);
                        const uint32_t b_atom_base = b_desc_base_lo + (b_atom_offset_bytes >> 4);

                        // A is K-major: TMA placed atom `i` at smem offset i * LOAD_BLOCK_M * DESC_ATOM_K elements
                        const uint32_t a_atom_offset_bytes = atom * LOAD_BLOCK_M * DESC_ATOM_K * sizeof(cutlass::bfloat16_t);
                        const uint32_t a_atom_base = a_desc_base_lo + (a_atom_offset_bytes >> 4);

                        if (debug_print && block_k_iter == 0) {
                            printf("  atom=%d: b_atom_base=0x%x (offset=%d B), a_atom_base=0x%x (offset=%d B)\n",
                                atom, b_atom_base, b_atom_offset_bytes, a_atom_base, a_atom_offset_bytes);
                        }

                        #pragma unroll
                        for (uint32_t ki = 0; ki < UMMA_ITERS_PER_ATOM; ++ki) {
                            if constexpr (kMajorB == cute::UMMA::Major::MN) {
                                b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                    b_desc_base_lo, 0, atom * DESC_ATOM_K + ki * UMMA_K);
                            } else {
                                b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                    b_atom_base, 0, ki * UMMA_K);
                            }
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                                a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(a_atom_base, w * WAVE_BLOCK_M * DESC_ATOM_K, ki * UMMA_K);
                                mma_t::fma(a_desc, b_desc,
                                           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                           (atom > 0 || ki > 0), // accumulate: false only for atom=0,ki=0
                                           runtime_instr_desc);
                            }
                        }
                    }
                }

                // if (threadIdx.x == 32) {
                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                //     const uint32_t mb_first = 0;
                //     const uint32_t nb_first = 0;
                //     const uint32_t mb_mid = num_m_blocks / 2;
                //     const uint32_t nb_mid = num_n_blocks / 2;
                //     const uint32_t mb_last = num_m_blocks - 1;
                //     const uint32_t nb_last = num_n_blocks - 1;
                //     int sample_tag = -1;

                //     if (block_m_iter == mb_first && blockIdx.x == nb_first) sample_tag = 0;
                //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid) sample_tag = 1;
                //     else if (block_m_iter == mb_last && blockIdx.x == nb_last) sample_tag = 2;

                //     if (sample_tag >= 0) {
                //         uint32_t result_values[4];
                //         constexpr uint32_t kDebugResultOffset = 0;
                //         cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //             accum_stage_idx * kNumMWaves * BLOCK_N + kDebugResultOffset,
                //             result_values[0], result_values[1], result_values[2], result_values[3]);
                //         cutlass::arch::fence_view_async_tmem_load();
                //         printf("DBG_UMMA_RESULT asym tag=%d bk=%d bm=%u nb=%u accum_stage=%u "
                //                "RESULT=%f,%f,%f,%f\n",
                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, accum_stage_idx,
                //                __uint_as_float(result_values[0]), __uint_as_float(result_values[1]),
                //                __uint_as_float(result_values[2]), __uint_as_float(result_values[3]));
                //     }
                // }

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                // empty_barrier_arrive();

                // if (threadIdx.x == 32) {
                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                //     const uint32_t mb_first = 0;
                //     const uint32_t nb_first = 0;
                //     // const uint32_t mb_mid = num_m_blocks / 2;
                //     // const uint32_t nb_mid = num_n_blocks / 2;
                //     const uint32_t mb_mid = 1;
                //     const uint32_t nb_mid = 0;
                //     // const uint32_t mb_last = num_m_blocks - 1;
                //     // const uint32_t nb_last = num_n_blocks - 1;
                //     const uint32_t mb_last = 2;
                //     const uint32_t nb_last = 0;
                //     int sample_tag = -1;

                //     if (block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                //     else if (block_m_iter == mb_last && blockIdx.x == nb_last  && blockIdx.y == 0) sample_tag = 2;

                //     if (sample_tag >= 0) {
                //         auto a_debug = smem_a[stage_idx];
                //         auto b_debug = smem_b[0];
                //         printf("DBG_UMMA asym tag=%d bk=%d bm=%u nb=%u a_stage=%u "
                //             "A=%f,%f,%f,%f,%f,%f,%f,%f "
                //             "B=%f,%f,%f,%f,%f,%f,%f,%f\n",
                //             sample_tag, block_k_iter, block_m_iter, blockIdx.x, stage_idx,
                //             bf16_to_float(a_debug[0]), bf16_to_float(a_debug[1]), bf16_to_float(a_debug[2]), bf16_to_float(a_debug[3]),
                //             bf16_to_float(a_debug[4]), bf16_to_float(a_debug[5]), bf16_to_float(a_debug[6]), bf16_to_float(a_debug[7]),
                //             bf16_to_float(b_debug[0]), bf16_to_float(b_debug[1]), bf16_to_float(b_debug[2]), bf16_to_float(b_debug[3]),
                //             bf16_to_float(b_debug[4]), bf16_to_float(b_debug[5]), bf16_to_float(b_debug[6]), bf16_to_float(b_debug[7]));
                //     }
                // }

                // if (block_m_iter == 0 and blockIdx.x == 0 and blockIdx.y == 0) {
                //     uint32_t result_values[4];
                //     constexpr uint32_t kDebugResultOffset = 0;
                //     cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //         accum_stage_idx * kNumMWaves * BLOCK_N + kDebugResultOffset,
                //         result_values[0], result_values[1], result_values[2], result_values[3]);
                //     cutlass::arch::fence_view_async_tmem_load();
                //     printf("DBG_UMMA_RESULT asym threadIDx=%u bk=%d bm=%u nb=%u accum_stage=%u "
                //             "RESULT=%f,%f,%f,%f\n",
                //             (unsigned)threadIdx.x, block_k_iter, block_m_iter, blockIdx.x, accum_stage_idx,
                //             __uint_as_float(result_values[0]), __uint_as_float(result_values[1]),
                //             __uint_as_float(result_values[2]), __uint_as_float(result_values[3]));
                // }

                // if (lane_idx == 0 and block_m_iter == 0 and blockIdx.x == 0 and blockIdx.y == 0) {
                //     uint32_t mma_row0[4];
                //     uint32_t mma_row1[4];
                //     const uint32_t tmem_base = accum_stage_idx * kNumMWaves * BLOCK_N;
                //     cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //         tmem_base,
                //         mma_row0[0], mma_row0[1], mma_row0[2], mma_row0[3]);
                //     cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                //         tmem_base + BLOCK_N,
                //         mma_row1[0], mma_row1[1], mma_row1[2], mma_row1[3]);
                //     cutlass::arch::fence_view_async_tmem_load();
                //     printf("DBG_BF16_ASYM MMA_2x2 threadIDx=%u block_k_iter=%u accum_stage=%u "
                //            "C=[[%.6f, %.6f],[%.6f, %.6f]]\n",
                //            (unsigned)threadIdx.x, (unsigned)block_k_iter, (unsigned)accum_stage_idx,
                //            __uint_as_float(mma_row0[0]), __uint_as_float(mma_row0[1]),
                //            __uint_as_float(mma_row1[0]), __uint_as_float(mma_row1[1]));
                // }

                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 32)
                //     printf("Asym after UMMA tmem_empty_barriers block_m_iter=%d \n", block_m_iter);

                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * LAYOUT_AD_M * kNumMWaves * BLOCK_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto& start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
            umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_b[0]));
        }

        // To safely deconstruct barriers, we need another round of waits
        // const auto& iter_idx = scheduler.current_iter - 1;
        // if (kNumMulticast > 1 and iter_idx >= 0) {
        //     const auto& phase = (iter_idx / kNumEpilogueStages) & 1;
        //     tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(phase);
        // }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {        
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("threadIdx.x: %d, epilogue_warp_idx: %d \n", threadIdx.x, epilogue_warp_idx);
        // }

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128) {
        //     const uint32_t tmem_val = ld_shared(tmem_ptr_in_smem);
            // printf("tmem_ptr_in_smem=%p tmem_val=%u warp_idx=%u smem_buffer=%p\n",
            //        (void*)tmem_ptr_in_smem, tmem_val, (unsigned)warp_idx, (void*)smem_buffer);
        // }
        // DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Share store pipeline between blocks
        uint32_t accum_stage_iter = 0, accum_stage_idx = 0, accum_phase_idx = 0, tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };

        auto advance_accum_pipeline = [&]() {
            accum_stage_idx = (accum_stage_idx + 1) % kNumEpilogueStages;
            accum_phase_idx ^= accum_stage_idx == 0;
        };
        // if (threadIdx.x == 256 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("Within the Epilogue warp, stage_idx: %d, phase: %d \n", stage_idx, phase);
        // }
        // uint32_t accum_stage_idx = -1, accum_phase_idx = -1;
        // Persistently schedule over blocks
        for (int block_k_iter = 0; block_k_iter < block_k; ++block_k_iter) {
            for (uint32_t block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; block_m_iter++, advance_accum_pipeline()) {
                // auto accum_stage_idx = accum_stage_iter % kNumEpilogueStages;
                // auto accum_phase_idx = (accum_stage_iter / kNumEpilogueStages) & 1;

                // if (threadIdx.x == 128 && blockIdx.x == 0) {
                //     printf("Within the Epilogue warp, block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }

                // Wait UMMA arrival
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();
                // ++accum_stage_iter;

                // Load from tensor memory into registers, and write shared memory with STSM
                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                // if (blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Epilogue warp after barriers, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                // }

                // Iterate over M waves
                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    // Issue every swizzled atom and pipeline STSM and TMA store
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                        // Wait shared memory to be released
                        if (epilogue_warp_idx == 0)
                        {
                            // if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0) {
                            //     printf("Epilogue warp, scheduler.n_idx: %d, kNumStores: %d, BLOCK_N: %d, STORE_BLOCK_N: %d \n", scheduler.n_idx, kNumStores, BLOCK_N, STORE_BLOCK_N);
                            // }
                            cute::tma_store_wait<kNumTMAStoreStages - 1>();
                        }
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        // The pipeline stage
                        const auto m_idx = (kGemmType == GemmType::MGroupedMasked)
                            ? (scheduler.current_group_idx * shape_m + (block_m_iter - scheduler.m_start) * BLOCK_M + w * WAVE_BLOCK_M)
                            : (BLOCK_M * block_m_iter + w * WAVE_BLOCK_M);
                        const auto n_idx = scheduler.n_blk * BLOCK_N + s * STORE_BLOCK_N;

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
                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +         // Accumulator offset
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
                                // if (lane_idx == 0 and block_m_iter == 0 and blockIdx.x == 0 and w == 0 and s == 0 and i == 0 and blockIdx.y == 0) {
                                //     printf("DBG_BF16_ASYM FP32 EPI_TMEM block_k_iter=%u warp=%u epi_warp=%u lane=%u stage=%u tmem_addr=%u values=%f,%f,%f,%f\n",
                                //            (unsigned)block_k_iter, (unsigned)warp_idx, (unsigned)epilogue_warp_idx, (unsigned)lane_idx, (unsigned)accum_stage_idx, (unsigned)tmem_addr,
                                //            __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //            __uint_as_float(values[2]), __uint_as_float(values[3]));
                                // }
                                // if (epilogue_warp_idx == 0 && lane_idx == 0 && w == 0 && s == 0 && i == 0) {
                                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                                //     const uint32_t mb_first = 0;
                                //     const uint32_t nb_first = 0;
                                //     // const uint32_t mb_mid = num_m_blocks / 2;
                                //     // const uint32_t nb_mid = num_n_blocks / 2;
                                //     const uint32_t mb_mid = 1;
                                //     const uint32_t nb_mid = 0;
                                //     // const uint32_t mb_last = num_m_blocks - 1;
                                //     // const uint32_t nb_last = num_n_blocks - 1;
                                //     const uint32_t mb_last = 2;
                                //     const uint32_t nb_last = 0;
                                //     int sample_tag = -1;
                                //     if (block_k_iter == 0 && block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_last && blockIdx.x == nb_last && blockIdx.y == 0) sample_tag = 2;
                                //     if (sample_tag >= 0) {
                                //         printf("DBG_TMEM_BEFORE_STSM asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u tmem_addr=%u "
                                //                "TMEM=%f,%f,%f,%f\n",
                                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tmem_addr,
                                //                __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //                __uint_as_float(values[2]), __uint_as_float(values[3]));
                                //     }
                                // }
#if ASYM_BF16_ROUTE_SCATTER_ADD
                                const uint32_t local_m = epilogue_warp_idx * 32 + row;
                                const uint32_t route_row = m_idx + local_m;
                                const uint32_t col0 = n_idx + i * kNumElemsPerBankGroup;
                                if (route_row < shape_m) {
                                    const int64_t token_row = route_token_indices[route_row];
                                    const float scale = qwen3_moe_route_weight_or_one(
                                        route_weights, route_weights_is_bf16, route_weighted, route_row);
                                    #pragma unroll
                                    for (uint32_t j = 0; j < kNumElemsPerBankGroup; ++j) {
                                        const uint32_t col_idx = col0 + j;
                                        if (col_idx < shape_n) {
                                            atomicAdd(
                                                &route_scatter_out[static_cast<uint64_t>(token_row) * route_scatter_stride + col_idx],
                                                __uint_as_float(values[j]) * scale);
                                        }
                                    }
                                }
#else
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
#endif
                            } else {
                                // For BF16 output, read, cast and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();
                                // if (lane_idx == 0 and block_m_iter == 0 and blockIdx.x == 0 and w == 0 and s == 0 and i == 0 and blockIdx.y == 0) {
                                //     printf("DBG_BF16_ASYM BF16 EPI_TMEM block_k_iter=%d warp=%u epi_warp=%u lane=%u stage=%u tmem_addr=%u values=%f,%f,%f,%f\n",
                                //            (unsigned)block_k_iter, (unsigned)warp_idx, (unsigned)epilogue_warp_idx, (unsigned)lane_idx, (unsigned)accum_stage_idx, (unsigned)tmem_addr,
                                //            __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //            __uint_as_float(values[2]), __uint_as_float(values[3]));
                                // }
                                // if (epilogue_warp_idx == 0 && lane_idx == 0 && w == 0 && s == 0 && i == 0) {
                                //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                                //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                                //     const uint32_t mb_first = 0;
                                //     const uint32_t nb_first = 0;
                                //     // const uint32_t mb_mid = num_m_blocks / 2;
                                //     // const uint32_t nb_mid = num_n_blocks / 2;
                                //     const uint32_t mb_mid = 1;
                                //     const uint32_t nb_mid = 0;
                                //     // const uint32_t mb_last = num_m_blocks - 1;
                                //     // const uint32_t nb_last = num_n_blocks - 1;
                                //     const uint32_t mb_last = 2;
                                //     const uint32_t nb_last = 0;
                                //     int sample_tag = -1;
                                //     if (block_k_iter == 0 && block_m_iter == mb_first && blockIdx.x == nb_first && blockIdx.y == 0) sample_tag = 0;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_mid && blockIdx.x == nb_mid && blockIdx.y == 0) sample_tag = 1;
                                //     else if (block_k_iter == 0 && block_m_iter == mb_last && blockIdx.x == nb_last && blockIdx.y == 0) sample_tag = 2;
                                //     if (sample_tag >= 0) {
                                //         printf("DBG_TMEM_BEFORE_STSM asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u tmem_addr=%u "
                                //                "TMEM=%f,%f,%f,%f,%f,%f,%f,%f\n",
                                //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tmem_addr,
                                //                __uint_as_float(values[0]), __uint_as_float(values[1]),
                                //                __uint_as_float(values[2]), __uint_as_float(values[3]),
                                //                __uint_as_float(values[4]), __uint_as_float(values[5]),
                                //                __uint_as_float(values[6]), __uint_as_float(values[7]));
                                //     }
                                // }
                                st_shared(smem_ptr,
                                        cast_into_bf16_and_pack(values[0], values[1]),
                                        cast_into_bf16_and_pack(values[2], values[3]),
                                        cast_into_bf16_and_pack(values[4], values[5]),
                                        cast_into_bf16_and_pack(values[6], values[7]));
                            }
                        }

                        // if (epilogue_warp_idx == 0 && lane_idx == 0 &&
                        //     w == 0 && s == 0) {
                        //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                        //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                        //     const uint32_t mb_first = 0;
                        //     const uint32_t nb_first = 0;
                        //     const uint32_t mb_mid = num_m_blocks / 2;
                        //     const uint32_t nb_mid = num_n_blocks / 2;
                        //     const uint32_t mb_last = num_m_blocks - 1;
                        //     const uint32_t nb_last = num_n_blocks - 1;
                        //     int sample_tag = -1;
                        //     if (block_m_iter == mb_first && blockIdx.x == nb_first) sample_tag = 0;
                        //     else if (block_m_iter == mb_mid && blockIdx.x == nb_mid) sample_tag = 1;
                        //     else if (block_m_iter == mb_last && blockIdx.x == nb_last) sample_tag = 2;

                        //     if (sample_tag >= 0) {
                        //         auto d_debug = smem_cd[tma_stage_idx];
                        //         printf("DBG_EPI_CD_AFTER_UMMA asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u cd_stage=%u "
                        //                "CD=%f,%f,%f,%f,%f,%f,%f,%f\n",
                        //                sample_tag, block_k_iter, block_m_iter, blockIdx.x, m_idx, n_idx, tma_stage_idx,
                        //                static_cast<float>(d_debug[0]), static_cast<float>(d_debug[1]),
                        //                static_cast<float>(d_debug[2]), static_cast<float>(d_debug[3]),
                        //                static_cast<float>(d_debug[4]), static_cast<float>(d_debug[5]),
                        //                static_cast<float>(d_debug[6]), static_cast<float>(d_debug[7]));
                        //     }
                        // }

                        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
                        // NOTES: only the last stage needs to do this
                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 128)
                            //     printf("Asym before release tmem_empty_barriers block_m_iter=%d \n", block_m_iter);
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // tcgen05_before_thread_sync();
                        // tmem_empty_barriers[stage_idx]->arrive(0u);
                        //toDo: check whether it is neccessary
                        // __syncwarp();

                        // Synchronize all threads and issue TMA
                        if constexpr (!kRouteScatterAdd)
                            cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        // if (epilogue_warp_idx == 0 && lane_idx == 0 &&
                        //     w == 0 && s == 0) {
                        //     const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
                        //     const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
                        //     const uint32_t mb_first = 0;
                        //     const uint32_t nb_first = 0;
                        //     const uint32_t mb_mid = num_m_blocks / 2;
                        //     const uint32_t nb_mid = num_n_blocks / 2;
                        //     const uint32_t mb_last = num_m_blocks - 1;
                        //     const uint32_t nb_last = num_n_blocks - 1;
                        //     int sample_tag = -1;
                        //     if (block_k_iter == 0 && block_m_iter == mb_first && scheduler.n_idx == nb_first) sample_tag = 0;
                        //     else if (block_k_iter == 0 && block_m_iter == mb_mid && scheduler.n_idx == nb_mid) sample_tag = 1;
                        //     else if (block_k_iter == 0 && block_m_iter == mb_last && scheduler.n_idx == nb_last) sample_tag = 2;
 
                        //     if (sample_tag >= 0) {
                        //         auto d_debug = smem_cd[tma_stage_idx];
                        //         printf("DBG_EPI_CD_PRE_TMA asym tag=%d bk=%d bm=%u nb=%u m=%u n=%u cd_stage=%u "
                        //                "CD=%f,%f,%f,%f,%f,%f,%f,%f\n",
                        //                sample_tag, block_k_iter, block_m_iter, scheduler.n_idx, m_idx, n_idx, tma_stage_idx,
                        //                static_cast<float>(d_debug[0]), static_cast<float>(d_debug[1]), static_cast<float>(d_debug[2]), static_cast<float>(d_debug[3]),
                        //                static_cast<float>(d_debug[4]), static_cast<float>(d_debug[5]), static_cast<float>(d_debug[6]), static_cast<float>(d_debug[7]));
                        //     }
                        // }

                        // if (blockIdx.x == 0 && blockIdx.y == 0) {
                        //     printf("Epilogue warp before store, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                        // }

                        if (debug_print && block_k_iter == 0) {
                            nv_bfloat16* cd_ptr = reinterpret_cast<nv_bfloat16*>(smem_cd[tma_stage_idx]);
                            printf("CD smem[0..7]: ");
                            for (int i = 0; i < 8; i++)
                                printf("%.4f ", __bfloat162float(cd_ptr[i]));
                            printf("\n");
                        }


                        if constexpr (!kRouteScatterAdd) if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {

                            if constexpr (kGemmType == GemmType::Batched) {
                                // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                //     printf("Epilogue warp within store 590, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
                                // }
                                using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                    cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                                cute_tma_t::copy(ASYM_DESC_CD, smem_cd[tma_stage_idx],
                                                n_idx, m_idx, scheduler.current_group_idx);
                            } else {
                          
                                if (block_k_iter == 0)
                                {
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp, kWithAccumulation: false, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    using cute_tma_t = cute::conditional_t<false,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(ASYM_DESC_CD, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                                else
                                {
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp within store, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    // if (blockIdx.x == 0 && blockIdx.y == 0) {
                                    //     printf("Epilogue warp, kWithAccumulation: false, threadIdx.x: %d, tma_stage_idx: %d, n_idx: %d, m_idx: %d \n", threadIdx.x, tma_stage_idx, n_idx, m_idx);
                                    // }
                                    using cute_tma_t = cute::conditional_t<true,
                                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                                    cute_tma_t::copy(ASYM_DESC_CD, smem_cd[tma_stage_idx], n_idx, m_idx);
                                }
                             
                            }
                            cute::tma_store_arrive();
                        }
                    }
                }

                // if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0) {
                //     printf("Within the Epilogue warp, after storing, block_k_iter: %d, block_m_iter: %d, stage_idx: %d, phase: %d \n", block_k_iter, block_m_iter, stage_idx, phase);
                // }
            }
        }

        // Deallocate tensor memory by the last UMMA store warp
        // NOTES: warp 0 is waiting TMA store
        if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1) {
            const auto tmem_ptr = ld_shared(tmem_ptr_in_smem);
            // if (blockIdx.x == 0 && blockIdx.y == 0) {
            //     printf("Within the Epilogue warp, Allocator().free, threadIdx.x: %d, stage_idx: %d, phase: %d \n", threadIdx.x, stage_idx, phase);
            // }
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
