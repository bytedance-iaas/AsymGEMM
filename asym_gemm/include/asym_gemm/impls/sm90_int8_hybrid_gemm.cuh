// SM90 INT8 hybrid grouped GEMM — ONE launch fusing the asym pipeline
// (host-resident expert weights, K-outer, single-slot B, TMA over PCIe/UVA)
// and the deep pipeline (HBM-resident weights, persistent M-outer,
// kNumStages-deep A+B) on disjoint CTA ranges (hybridGEMM.md Phase B).
//
// CTA ranks [0, s_host) run the asym side over the host segment list;
// ranks [s_host, kNumSMs) run the deep side over the HBM segment list.
// `s_host` is a RUNTIME argument — the balance point moves per forward with
// the routing histogram; re-JITting per split is a non-starter.
//
// v1 constraints (hybridGEMM.md §4):
//   * same BLOCK_M/BLOCK_N/BLOCK_K on both sides (64/128/128): one
//     INT8MMASelector instantiation, one epilogue store geometry, one
//     __launch_bounds__ (128 TMA + 128 math);
//   * kNumMulticast == 1 — cluster shape is launch-wide and a runtime s_host
//     could straddle a 2-CTA cluster; every barrier is CTA-local (§9.7);
//   * no stealing (Phase C) — each CTA runs exactly one side;
//   * K-major A and B only (INT8 WGMMA has no MN-major operand form, §9.4).
//
// Shared memory is the UNION of the two sides' plans: the 2-stage CD store
// ring is common (identical geometry on both sides), the region after it is
// overlaid per side, and each CTA initializes only its own side's barriers.
//
// Output contract (§9.6): both sides write the same fp32 [M, N] buffer via
// the same tensor map. The asym side keeps its k=0-STORE / k>0-REDUCE_ADD
// protocol (CTA-internal stream order); the deep side plain-stores fully
// accumulated tiles. Host and HBM segments are disjoint row ranges, so no
// cross-side ordering is needed.
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

// ---------------------------------------------------------------------------
// asym (host) side: K-outer, single-slot B, SFA/SFB via TMA,
// STORE-then-REDUCE_ADD output. Body adapted from
// sm90_int8_asym_gemm_1d1d.cuh specialized to MGroupedContiguous / K-major /
// multicast-off / fp32-out, wrapped in a persistent item loop: flat item
// i = rank + num_host_ctas * iter over (segment, n_block) items — the
// explicit-ids asymScheduler ctor decouples work identity from blockIdx.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages, uint32_t kNumMathThreads, uint32_t kNumSMs>
__device__ __noinline__ void sm90_hybrid_host_side(
        uint32_t* offsets, uint32_t* experts, const uint32_t num_segments,
        const uint32_t rank, const uint32_t num_host_ctas,
        const uint32_t shape_m, const uint32_t shape_n, const uint32_t shape_k,
        const cute::TmaDescriptor* tensor_map_a,
        const cute::TmaDescriptor* tensor_map_b,
        const cute::TmaDescriptor* tensor_map_sfa,
        const cute::TmaDescriptor* tensor_map_sfb,
        const cute::TmaDescriptor* tensor_map_cd,
        uint8_t* smem_buffer) {
    using namespace asym_gemm::sm90;
    using WGMMA = typename INT8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(BLOCK_M == 64 and BLOCK_M == WGMMA::M, "v1: single math warp-group");
    DG_STATIC_ASSERT(BLOCK_K % WGMMA::K == 0, "BLOCK_K must be a multiple of WGMMA K");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Shared memory plan (asym parent layout; CD ring shared with the deep side).
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * BLOCK_N * sizeof(float);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(int8_t);
    constexpr uint32_t SMEM_B_SIZE = BLOCK_N * BLOCK_K * sizeof(int8_t);   // single slot
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    constexpr uint32_t SMEM_SFB_SIZE = BLOCK_N * sizeof(float);            // single slot
    DG_STATIC_ASSERT(SMEM_CD_SIZE_PER_STAGE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0
                     and SMEM_B_SIZE % 1024 == 0, "Unaligned shared memory");

    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<int8_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<int8_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE);
    });
    const auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE;
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(sf_start_ptr +
        kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE);
    auto full_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers   = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto full_barriers_b  = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2; });
    auto empty_barriers_b = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + 1; });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }
        full_barriers_b[0]->init(1);
        empty_barriers_b[0]->init(kNumMathThreads / 32);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    const uint32_t num_n_blocks = ceil_div_device(shape_n, BLOCK_N);
    const uint32_t num_items = num_segments * num_n_blocks;
    const uint32_t num_k_blocks = ceil_div_device(shape_k, BLOCK_K);

    // Pipeline state persists across items (the barriers keep their phase).
    uint32_t stage_idx = 0, phase = 0, phase_b = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    auto make_sched = [&](const uint32_t& item) {
        return asymScheduler<GemmType::MGroupedContiguous, BLOCK_M, BLOCK_N,
                             1, 1, false, kNumSMs>(
            shape_m, shape_n, experts, offsets,
            item / num_n_blocks, item % num_n_blocks);
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // ============================= TMA side =============================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            for (uint32_t item = rank; item < num_items; item += num_host_ctas) {
                const auto& sched = make_sched(item);
                for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                    const uint32_t k_idx = kb * BLOCK_K;

                    // B tile + SFB (single slot, reused across all M blocks).
                    empty_barriers_b[0]->wait(phase_b ^ 1);
                    phase_b ^= 1;
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, int8_t, false>(
                        tensor_map_b, full_barriers_b[0], smem_b[0], k_idx, sched.n_idx, 1, 0);
                    tma_copy<BLOCK_N, 1, 0>(tensor_map_sfb, full_barriers_b[0], smem_sfb[0], sched.n_idx, kb);
                    full_barriers_b[0]->arrive_and_expect_tx(SMEM_B_SIZE + SMEM_SFB_SIZE);

                    for (uint32_t m_it = sched.m_start; m_it < sched.m_end; ++ m_it) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        const uint32_t m_idx = m_it * BLOCK_M;
                        tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, int8_t, false>(
                            tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1, 0);
                        tma_copy<BLOCK_M, 1, 0>(tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], m_idx, kb);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                        advance_pipeline();
                    }
                }
            }
        }
    } else {
        // ============================ math side =============================
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        auto a_desc = make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
        const uint32_t wg_local_warp_idx = warp_idx % 4;
        const uint32_t row_idx = lane_idx / 4, col_idx_scale = lane_idx % 4;
        const uint32_t r_0 = wg_local_warp_idx * WGMMA_M_PER_WARP + row_idx;
        const uint32_t r_1 = r_0 + 8;

        uint32_t tma_stage_idx = 0;
        for (uint32_t item = rank; item < num_items; item += num_host_ctas) {
            const auto& sched = make_sched(item);
            for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                full_barriers_b[0]->wait(phase_b);
                phase_b ^= 1;

                for (uint32_t m_it = sched.m_start; m_it < sched.m_end; ++ m_it) {
                    int32_t accum[WGMMA::kNumAccum] = {0};
                    full_barriers[stage_idx]->wait(phase);
                    const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                    const auto b_desc_base_lo = b_desc_lo;

                    // Read scales while the math side still owns smem_sfa/smem_sfb
                    // (before empty_barrier arrivals release them).
                    float2 scales_b[WGMMA::kNumAccum / 4];
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i)
                        scales_b[i] = *reinterpret_cast<const float2*>(smem_sfb[0] + i * 8 + col_idx_scale * 2);
                    const float scale_a_0 = ld_shared(smem_sfa[stage_idx] + r_0);
                    const float scale_a_1 = ld_shared(smem_sfa[stage_idx] + r_1);

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
                        WGMMA::wgmma(a_desc, b_desc, accum, 1);
                    }
                    warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();

                    // Release A (scale reads already done above).
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                    advance_pipeline();

                    // ---- epilogue: dequant into the CD ring, STORE/REDUCE_ADD.
                    if (threadIdx.x < 1)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    auto smem_d_0 = reinterpret_cast<float2*>(
                        smem_cd[tma_stage_idx] + (wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    auto smem_d_1 = reinterpret_cast<float2*>(
                        smem_cd[tma_stage_idx] + (wg_local_warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        st_shared(smem_d_0 + i * 4, make_float2(
                            scale_a_0 * scales_b[i].x * static_cast<float>(accum[i * 4 + 0]),
                            scale_a_0 * scales_b[i].y * static_cast<float>(accum[i * 4 + 1])));
                        st_shared(smem_d_1 + i * 4, make_float2(
                            scale_a_1 * scales_b[i].x * static_cast<float>(accum[i * 4 + 2]),
                            scale_a_1 * scales_b[i].y * static_cast<float>(accum[i * 4 + 3])));
                    }
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    // Store coordinates: n is group-LOCAL (CD folds groups into M),
                    // via the scheduler's n_blk — NOT blockIdx.x (persistent wrapper).
                    if (threadIdx.x < 1) {
                        if (kb == 0) {
                            cute::SM90_TMA_STORE_2D::copy(tensor_map_cd, smem_cd[tma_stage_idx],
                                                          sched.n_blk * BLOCK_N, m_it * BLOCK_M);
                        } else {
                            cute::SM90_TMA_REDUCE_ADD_2D::copy(tensor_map_cd, smem_cd[tma_stage_idx],
                                                               sched.n_blk * BLOCK_N, m_it * BLOCK_M);
                        }
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }

                // Release B after all M blocks of this K block.
                lane_idx == 0 ? empty_barriers_b[0]->arrive() : void();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// deep (HBM) side: persistent M-outer pipeline — the Phase A kernel body with
// the CTA-stride re-based onto the [s_host, kNumSMs) rank range (runtime
// peer count, so no re-JIT when the split moves).
//
// Stealing (hybridGEMM.md §5 C2, `enable_steal`): tile enumeration switches
// from static rank-striding to popping a device-global atomic ticket counter,
// so asym-side CTAs that exhaust the host segment list can join by calling
// this function against the SAME counter. The TMA warp's elected thread pops
// and decodes each ticket, publishes (m_blk, n_blk, eid) through a 2-slot
// smem mailbox ring (its own full/empty mbarrier pairs — the math warps
// consume tiles in exactly producer order, keeping the A/B stage walk
// identical on both sides), and terminates the math side with a sentinel.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages, uint32_t kNumMathThreads>
__device__ __noinline__ void sm90_hybrid_hbm_side(
        uint32_t* offsets, uint32_t* experts, const uint32_t num_segments,
        const uint32_t rank, const uint32_t num_peers,
        const uint32_t enable_steal, uint32_t* steal_counter,
        const float* __restrict__ sfa, const float* __restrict__ sfb,
        const uint32_t shape_m, const uint32_t shape_n, const uint32_t shape_k,
        const cute::TmaDescriptor* tensor_map_a,
        const cute::TmaDescriptor* tensor_map_b,
        const cute::TmaDescriptor* tensor_map_cd,
        uint8_t* smem_buffer) {
    using namespace asym_gemm::sm90;
    using WGMMA = typename INT8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(BLOCK_M == 64 and BLOCK_M == WGMMA::M, "v1: single math warp-group");
    DG_STATIC_ASSERT(BLOCK_K % WGMMA::K == 0, "BLOCK_K must be a multiple of WGMMA K");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Per-K-block promotion assumes GRAN_K == BLOCK_K == 128");
    DG_STATIC_ASSERT(kNumStages >= 2, "Deep pipeline expected");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Shared memory plan (deep parent layout; CD ring shared with the asym side).
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * BLOCK_N * sizeof(float);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(int8_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(int8_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE_PER_STAGE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0
                     and SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Unaligned shared memory");

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

    // Steal-mode scheduler mailbox: 2 ring slots of (m_blk, n_blk, eid, pad)
    // behind their own full/empty barrier pairs. ~100 bytes; always laid out,
    // only initialized/used when enable_steal is set.
    constexpr uint32_t kNumSchedSlots = 2;
    constexpr uint32_t kSchedSentinel = 0xffffffffu;
    auto full_sched  = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto empty_sched = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumSchedSlots + i; });
    auto sched_ring = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + kNumSchedSlots * 2) + i * 4;
    });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads / 32);
        }
        if (enable_steal) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumSchedSlots; ++ i) {
                full_sched[i]->init(1);
                empty_sched[i]->init(kNumMathThreads / 32);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    const uint32_t num_n_blocks = ceil_div_device(shape_n, BLOCK_N);
    const uint32_t num_k_blocks = ceil_div_device(shape_k, BLOCK_K);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // f(m_block_global, n_block, expert_id) over tiles owned by this CTA:
    // tiles are numbered n-fastest within a segment; this CTA owns global
    // tile ids congruent to its side-local rank modulo the peer count.
    auto for_each_owned_tile = [&](auto&& f) {
        uint32_t tile_mod = 0;   // (total tiles of previous segments) % num_peers
        for (uint32_t seg = 0; seg < num_segments; ++ seg) {
            const uint32_t seg_start = __ldg(offsets + 2 * seg);
            const uint32_t seg_end   = __ldg(offsets + 2 * seg + 1);
            const uint32_t eid       = __ldg(experts + seg);
            const uint32_t mb_start  = seg_start / BLOCK_M;        // starts are BLOCK_M-aligned
            const uint32_t mb_end    = ceil_div_device(seg_end, BLOCK_M);
            const uint32_t tiles     = (mb_end - mb_start) * num_n_blocks;
            const uint32_t first     = (rank + num_peers - tile_mod) % num_peers;
            for (uint32_t t = first; t < tiles; t += num_peers)
                f(mb_start + t / num_n_blocks, t % num_n_blocks, eid);
            tile_mod = (tile_mod + tiles) % num_peers;
        }
    };

    // Steal mode: global ticket -> (m_blk, n_blk, eid); false past the end.
    auto decode_tile = [&](const uint32_t& t, uint32_t& m_blk, uint32_t& n_blk, uint32_t& eid) {
        uint32_t cum = 0;
        for (uint32_t seg = 0; seg < num_segments; ++ seg) {
            const uint32_t seg_start = __ldg(offsets + 2 * seg);
            const uint32_t seg_end   = __ldg(offsets + 2 * seg + 1);
            const uint32_t mb_start  = seg_start / BLOCK_M;
            const uint32_t tiles     = (ceil_div_device(seg_end, BLOCK_M) - mb_start) * num_n_blocks;
            if (t < cum + tiles) {
                const uint32_t tl = t - cum;
                m_blk = mb_start + tl / num_n_blocks;
                n_blk = tl % num_n_blocks;
                eid   = __ldg(experts + seg);
                return true;
            }
            cum += tiles;
        }
        return false;
    };

    // Per-side tile walks. Static mode strides deterministically (both warp
    // groups replay the same walk); steal mode pops the ticket counter ONCE
    // per tile on the TMA side and forwards each tile through the mailbox,
    // ending with a sentinel — so the two sides again walk identical
    // sequences and the persistent A/B stage state stays consistent.
    auto tma_walk = [&](auto&& f) {
        if (not enable_steal)
            return for_each_owned_tile(f);
        uint32_t sched_slot = 0, sched_phase = 0;
        while (true) {
            uint32_t m_blk, n_blk, eid;
            const bool ok = decode_tile(atomicAdd(steal_counter, 1u), m_blk, n_blk, eid);
            empty_sched[sched_slot]->wait(sched_phase ^ 1);
            st_shared(sched_ring[sched_slot] + 0, ok ? m_blk : kSchedSentinel);
            st_shared(sched_ring[sched_slot] + 1, ok ? n_blk : 0u);
            st_shared(sched_ring[sched_slot] + 2, ok ? eid : 0u);
            full_sched[sched_slot]->arrive();
            sched_slot ^= 1;
            sched_phase ^= sched_slot == 0;
            if (not ok)
                break;
            f(m_blk, n_blk, eid);
        }
    };
    auto math_walk = [&](auto&& f) {
        if (not enable_steal)
            return for_each_owned_tile(f);
        uint32_t sched_slot = 0, sched_phase = 0;
        while (true) {
            full_sched[sched_slot]->wait(sched_phase);
            const uint32_t m_blk = ld_shared(sched_ring[sched_slot] + 0);
            const uint32_t n_blk = ld_shared(sched_ring[sched_slot] + 1);
            const uint32_t eid   = ld_shared(sched_ring[sched_slot] + 2);
            __syncwarp();   // whole warp has read the slot before releasing it
            lane_idx == 0 ? empty_sched[sched_slot]->arrive() : void();
            sched_slot ^= 1;
            sched_phase ^= sched_slot == 0;
            if (m_blk == kSchedSentinel)
                break;
            f(m_blk, n_blk, eid);
        }
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // ============================= TMA side =============================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            tma_walk([&](const uint32_t& m_blk, const uint32_t& n_blk, const uint32_t& eid) {
                const uint32_t m_idx = m_blk * BLOCK_M;
                const uint32_t n_idx = eid * shape_n + n_blk * BLOCK_N;
                for (uint32_t kb = 0; kb < num_k_blocks; ++ kb) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    const uint32_t k_idx = kb * BLOCK_K;
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode, int8_t, false>(
                        tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1, 0);
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode, int8_t, false>(
                        tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1, 0);
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
        math_walk([&](const uint32_t& m_blk, const uint32_t& n_blk, const uint32_t& eid) {
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

                // Per-K-block FFMA promotion into fp32 registers (scales from
                // global; the smem stage was already released above).
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

            // ---- epilogue: fp32 accumulators are fully scaled; plain store.
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

            if (threadIdx.x < 1) {
                cute::SM90_TMA_STORE_2D::copy(tensor_map_cd, smem_cd[tma_stage_idx],
                                              n_blk * BLOCK_N, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        });
    }
}

// ---------------------------------------------------------------------------
// The fused kernel: branch once on CTA rank, then run one parent verbatim.
// ---------------------------------------------------------------------------
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroupsHbm,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStagesHost, uint32_t kNumStagesHbm,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs>
__global__ void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
sm90_int8_hybrid_gemm_impl(
        uint32_t* offsets_host, uint32_t* experts_host, uint32_t num_segments_host,
        uint32_t* offsets_hbm,  uint32_t* experts_hbm,  uint32_t num_segments_hbm,
        uint32_t s_host,
        uint32_t enable_steal, uint32_t* steal_counter,
        const float* __restrict__ sfa, const float* __restrict__ sfb_hbm,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor tensor_map_b_host,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa_host,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb_host,
        const __grid_constant__ cute::TmaDescriptor tensor_map_b_hbm,
        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
        if (blockIdx.x < s_host) {
            cute::prefetch_tma_descriptor(&tensor_map_b_host);
            cute::prefetch_tma_descriptor(&tensor_map_sfa_host);
            cute::prefetch_tma_descriptor(&tensor_map_sfb_host);
        } else {
            cute::prefetch_tma_descriptor(&tensor_map_b_hbm);
        }
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    if (blockIdx.x < s_host) {
        if (num_segments_host > 0)
            sm90_hybrid_host_side<BLOCK_M, BLOCK_N, BLOCK_K,
                                  kSwizzleAMode, kSwizzleBMode,
                                  kNumStagesHost, kNumMathThreads, kNumSMs>(
                offsets_host, experts_host, num_segments_host,
                blockIdx.x, s_host,
                shape_m, shape_n, shape_k,
                &tensor_map_a, &tensor_map_b_host,
                &tensor_map_sfa_host, &tensor_map_sfb_host, &tensor_map_cd,
                smem_buffer);
        // Phase C stealing (one-directional, asym -> hbm): once the host
        // item list is exhausted, quiesce this CTA, then join the hbm side's
        // ticket-pop loop (hybridGEMM.md §9.7). Order matters:
        //   1. __syncthreads — every warp is fully out of the host pipeline
        //      (all mbarrier arrivals/waits retired; the init warp would
        //      otherwise race ahead, it does no work in the host loops);
        //   2. tma_store_wait<0> on the storing thread — the CD ring is
        //      drained before the deep side reuses it (its own epilogue only
        //      waits to depth 1, which could still overlap an asym store);
        //   3. mbarrier.inval the host-side barriers — their addresses fall
        //      inside the deep side's A/B stage region, and TMA-writing a
        //      still-valid mbarrier is UB per the PTX spec;
        //   4. the hbm side's own init + fence + __syncthreads gate every
        //      subsequent smem access, so nothing more is needed here.
        if (enable_steal and num_segments_hbm > 0) {
            __syncthreads();
            if (threadIdx.x < 1)
                cute::tma_store_wait<0>();
            if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
                constexpr uint32_t kHostBarrierOffset =
                    BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(float)) * 2   // CD ring
                    + kNumStagesHost * BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K       // A stages + B slot
                    + kNumStagesHost * BLOCK_M * static_cast<uint32_t>(sizeof(float))
                    + BLOCK_N * static_cast<uint32_t>(sizeof(float));              // SFA stages + SFB slot
                const auto host_barriers = reinterpret_cast<const uint64_t*>(smem_buffer + kHostBarrierOffset);
                #pragma unroll
                for (uint32_t i = 0; i < 2 * kNumStagesHost + 2; ++ i)
                    cutlass::arch::ClusterTransactionBarrier::invalidate(host_barriers + i);
            }
            if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync())
                cute::prefetch_tma_descriptor(&tensor_map_b_hbm);
            sm90_hybrid_hbm_side<BLOCK_M, BLOCK_N, BLOCK_K,
                                 kNumGroupsHbm,
                                 kSwizzleAMode, kSwizzleBMode,
                                 kNumStagesHbm, kNumMathThreads>(
                offsets_hbm, experts_hbm, num_segments_hbm,
                /*rank=*/0, /*num_peers=*/1,   // unused in steal mode
                enable_steal, steal_counter,
                sfa, sfb_hbm,
                shape_m, shape_n, shape_k,
                &tensor_map_a, &tensor_map_b_hbm, &tensor_map_cd,
                smem_buffer);
        }
    } else {
        if (num_segments_hbm == 0) return;
        sm90_hybrid_hbm_side<BLOCK_M, BLOCK_N, BLOCK_K,
                             kNumGroupsHbm,
                             kSwizzleAMode, kSwizzleBMode,
                             kNumStagesHbm, kNumMathThreads>(
            offsets_hbm, experts_hbm, num_segments_hbm,
            blockIdx.x - s_host, kNumSMs - s_host,
            enable_steal, steal_counter,
            sfa, sfb_hbm,
            shape_m, shape_n, shape_k,
            &tensor_map_a, &tensor_map_b_hbm, &tensor_map_cd,
            smem_buffer);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90a");
#endif
}

};  // namespace asym_gemm

#pragma clang diagnostic pop
