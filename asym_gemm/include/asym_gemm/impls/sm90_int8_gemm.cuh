// SM90 INT8 grouped GEMM for HBM-resident expert weights — deepGEMM iteration
// pattern (hybridGEMM.md Phase A).
//
// Derived from DeepGEMM's sm90_bf16_gemm.cuh persistent design, re-based onto
// the asym_gemm INT8 infrastructure:
//   * persistent 1D grid (kNumSMs CTAs); each CTA strides over the
//     (segment, m_block, n_block) tile space derived from the SAME
//     offsets/experts contiguous grouped layout the asym kernel consumes;
//   * M-outer loop: per tile the FULL K is swept with a kNumStages-deep
//     pipeline on BOTH A and B (the asym kernel's single-slot B is a
//     PCIe trade, wrong for HBM);
//   * INT8 storage, S32 WGMMA exact within each K-block, then per-K-block
//     FFMA promotion into fp32 REGISTER accumulators (the DeepGEMM FP8
//     pattern) — same per-block math as the asym kernel and the reference,
//     but the cross-K-block sum stays in registers instead of chaining fp32
//     TMA_REDUCE_ADD partial sums through HBM;
//   * SFA/SFB are read directly from global memory per K-block (a handful of
//     L2-resident floats per warp — the K-blocked SF TMA machinery the asym
//     kernel needs is pure overhead here).
//     sfa: K-major [Kb, M] fp32; sfb: K-major [Kb, G*N] fp32
//   * FP32 output via plain TMA_STORE (no partial sums, no REDUCE_ADD).
//
// v1 constraints (mirroring the asym launcher's fixed config):
//   BLOCK_M == 64 (single math warp-group), K-major A and B, no multicast.
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
#include <asym_gemm/common/tma_utils.cuh>

namespace asym_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs>
__global__ void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
sm90_int8_gemm_impl(uint32_t* offsets, uint32_t* experts, uint32_t num_segments,
                    const float* __restrict__ sfa, const float* __restrict__ sfb,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using namespace asym_gemm::sm90;

    using WGMMA = typename INT8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // v1 shape constraints (see header comment).
    DG_STATIC_ASSERT(BLOCK_M == 64 and BLOCK_M == WGMMA::M, "v1: single math warp-group");
    DG_STATIC_ASSERT(BLOCK_K % WGMMA::K == 0, "BLOCK_K must be a multiple of WGMMA K");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Per-K-block promotion assumes GRAN_K == BLOCK_K == 128");
    DG_STATIC_ASSERT(kNumStages >= 2, "Deep pipeline expected");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Shared memory: [cd ring: 2][A: kNumStages][B: kNumStages][barriers]
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * BLOCK_N * sizeof(float);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(int8_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(int8_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE_PER_STAGE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0
                     and SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Unaligned shared memory");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<int8_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<int8_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE
                                         + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    // ---------------------------------------------------------------
    // Persistent tile schedule (identical walk on TMA and math sides).
    // Tiles are numbered n-fastest within a segment; CTA `blockIdx.x`
    // owns global tile ids congruent to its rank modulo kNumSMs.
    // ---------------------------------------------------------------
    const uint32_t num_n_blocks = ceil_div_device(shape_n, BLOCK_N);
    const uint32_t num_k_blocks = ceil_div_device(shape_k, BLOCK_K);

    // Pipeline state persists across tiles and segments.
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // for_each_owned_tile(f): f(m_block_global, n_block, expert_id)
    auto for_each_owned_tile = [&](auto&& f) {
        uint32_t tile_mod = 0;   // (total tiles of previous segments) % kNumSMs
        for (uint32_t seg = 0; seg < num_segments; ++ seg) {
            const uint32_t seg_start = __ldg(offsets + 2 * seg);
            const uint32_t seg_end   = __ldg(offsets + 2 * seg + 1);
            const uint32_t eid       = __ldg(experts + seg);
            const uint32_t mb_start  = seg_start / BLOCK_M;        // starts are BLOCK_M-aligned
            const uint32_t mb_end    = ceil_div_device(seg_end, BLOCK_M);
            const uint32_t tiles     = (mb_end - mb_start) * num_n_blocks;
            const uint32_t first     = (blockIdx.x + kNumSMs - tile_mod) % kNumSMs;
            for (uint32_t t = first; t < tiles; t += kNumSMs)
                f(mb_start + t / num_n_blocks, t % num_n_blocks, eid);
            tile_mod = (tile_mod + tiles) % kNumSMs;
        }
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // ============================= TMA side =============================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for the TMA warp-group");
            for_each_owned_tile([&](const uint32_t& m_blk, const uint32_t& n_blk, const uint32_t& eid) {
                const uint32_t m_idx = m_blk * BLOCK_M;
                const uint32_t n_idx = eid * shape_n + n_blk * BLOCK_N;
                for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    const uint32_t k_idx = kb * BLOCK_K;
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, int8_t, false>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1, 0);
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, int8_t, false>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1, 0);
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                    advance_pipeline();
                }
            });
        }
    } else {
        // ============================ math side =============================
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        auto a_desc = make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Epilogue thread mapping (one 128-thread warp-group over 64 rows).
        constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
        const uint32_t wg_local_warp_idx = warp_idx % 4;
        const uint32_t row_idx = lane_idx / 4, col_idx_scale = lane_idx % 4;
        const uint32_t r_0 = wg_local_warp_idx * WGMMA_M_PER_WARP + row_idx;
        const uint32_t r_1 = r_0 + 8;

        auto empty_barrier_arrive = [&](const uint32_t& s) {
            lane_idx == 0 ? empty_barriers[s]->arrive() : void();
        };

        const uint32_t sfb_row_stride = kNumGroups * shape_n;

        uint32_t tma_stage_idx = 0;
        for_each_owned_tile([&](const uint32_t& m_blk, const uint32_t& n_blk, const uint32_t& eid) {
            const uint32_t m_idx = m_blk * BLOCK_M;
            const uint32_t n_idx = eid * shape_n + n_blk * BLOCK_N;

            int32_t accum[WGMMA::kNumAccum];
            float final_accum[WGMMA::kNumAccum] = {0};

            for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                full_barriers[stage_idx]->wait(phase);
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    a_desc.reg32_[0] = advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode, int8_t>(
                        a_desc_base_lo, 0, k * WGMMA::K);
                    b_desc.reg32_[0] = advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode, int8_t>(
                        b_desc_base_lo, 0, k * WGMMA::K);
                    // scale_d = 0 on the K-block's first WGMMA resets the S32
                    // accumulator; each K-block is accumulated exactly in int32.
                    WGMMA::wgmma(a_desc, b_desc, accum, k > 0);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                empty_barrier_arrive(stage_idx);
                advance_pipeline();

                // Per-K-block FFMA promotion into fp32 registers. Scales come
                // straight from global (L2-resident after the first tile) —
                // the smem stage was already released above; only `accum`
                // (registers) is read here. GRAN_K == BLOCK_K == 128 keeps
                // one scale per (row/col, K-block).
                const float scale_a_0 = __ldg(sfa + kb * shape_m + m_idx + r_0);
                const float scale_a_1 = __ldg(sfa + kb * shape_m + m_idx + r_1);
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float2 scale_b = __ldg(
                        reinterpret_cast<const float2*>(sfb + kb * sfb_row_stride + n_idx) + i * 4 + col_idx_scale);
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b.x * static_cast<float>(accum[i * 4 + 0]);
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b.y * static_cast<float>(accum[i * 4 + 1]);
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b.x * static_cast<float>(accum[i * 4 + 2]);
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b.y * static_cast<float>(accum[i * 4 + 3]);
                }
            }

            // ---- epilogue: fp32 accumulators are fully scaled; store as-is.
            if (threadIdx.x < 1)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            auto smem_d_0 = reinterpret_cast<float2*>(
                smem_cd[tma_stage_idx] + (wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
            auto smem_d_1 = reinterpret_cast<float2*>(
                smem_cd[tma_stage_idx] + (wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                st_shared(smem_d_0 + i * 4, make_float2(final_accum[i * 4 + 0], final_accum[i * 4 + 1]));
                st_shared(smem_d_1 + i * 4, make_float2(final_accum[i * 4 + 2], final_accum[i * 4 + 3]));
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Output rows are group-global (contiguous layout): store at (n_local, m_global).
            if (threadIdx.x < 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_cd[tma_stage_idx],
                                              n_blk * BLOCK_N, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        });
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90a");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
