#pragma once

#include <asym_gemm/common/types.hpp>
#include <asym_gemm/common/utils.cuh>

namespace asym_gemm {

// Two-phase fused MoE computation
enum class BlockPhase : uint8_t {
    None    = 0,
    Linear1 = 1,  // L1: x -> gate+up    (BF16 acc out of FP4xFP4)
    Linear2 = 2   // L2: intermediate -> hidden   (BF16 acc out of FP4xFP4)
};

// Persistent two-phase scheduler for the fused FP4 mega MoE kernel.
//
// The kernel processes all L1 blocks first, does a grid sync, then processes
// all L2 blocks.  Each SM iterates through its assigned blocks via
// `for_each_block(fn)`, which calls `fn(phase, local_expert_idx, num_k_blocks,
// m_block_idx, n_block_idx)`.
//
// Block order within each phase:
//   for local_expert_idx in 0..kNumExperts:
//       for m_block_idx in 0..num_m_blocks[expert]:
//           for n_block_idx in 0..num_n_blocks:
//               emit block
//
// Blocks are assigned round-robin across SMs by `block_global_idx % kNumSMs`.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExperts,
          uint32_t kNumSMs>
struct MegaMoEAsymScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "L1_SHAPE_N not divisible by BLOCK_N");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "L2_SHAPE_N not divisible by BLOCK_N");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "L1_SHAPE_K not divisible by BLOCK_K");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "L2_SHAPE_K not divisible by BLOCK_K");

    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1KBlocks = L1_SHAPE_K / BLOCK_K;
    static constexpr uint32_t kNumL2KBlocks = L2_SHAPE_K / BLOCK_K;

    // Per-expert token count (loaded from offsets buffer in device memory)
    // offsets layout: [start_0, end_0, start_1, end_1, ..., start_{E-1}, end_{E-1}]
    uint32_t per_expert_num_m_blocks[kNumExperts];
    uint32_t per_expert_m_block_start[kNumExperts];  // cumulative M block offset
    uint32_t per_expert_num_valid_tokens[kNumExperts];
    uint32_t total_m_blocks = 0;

    // Current iteration state (populated inside `for_each_block`)
    BlockPhase current_phase     = BlockPhase::None;
    uint32_t current_expert_idx  = 0;
    uint32_t current_m_block_idx = 0;  // m-block index within the current expert
    uint32_t current_n_block_idx = 0;
    uint32_t current_pool_block_offset = 0;  // global m-block offset across experts
    uint32_t current_valid_m     = 0;

    __device__ __forceinline__
    MegaMoEAsymScheduler(const uint32_t* offsets) {
        const uint32_t lane_idx = get_lane_idx();
        // Read offsets sequentially.  TODO: parallelize with one lane per expert.
        uint32_t cum_blocks = 0;
        #pragma unroll
        for (uint32_t e = 0; e < kNumExperts; ++e) {
            const uint32_t m_start = offsets[2 * e];
            const uint32_t m_end   = offsets[2 * e + 1];
            const uint32_t num_tok = (m_end > m_start) ? (m_end - m_start) : 0;
            const uint32_t num_m_blocks = ceil_div<uint32_t>(num_tok, BLOCK_M);
            per_expert_num_valid_tokens[e] = num_tok;
            per_expert_num_m_blocks[e]     = num_m_blocks;
            per_expert_m_block_start[e]    = cum_blocks;
            cum_blocks += num_m_blocks;
        }
        total_m_blocks = cum_blocks;
    }

    // Iterate over all (phase, expert, m_block, n_block) tuples assigned to this SM.
    // The callable `fn` is invoked with:
    //   fn(BlockPhase phase, uint32_t local_expert_idx, uint32_t num_k_blocks,
    //      uint32_t m_block_idx_in_expert, uint32_t n_block_idx)
    template <class Fn>
    __device__ __forceinline__ void for_each_block(Fn&& fn) {
        const uint32_t sm_idx = blockIdx.x;

        // -------- Phase 1: L1 -----------
        current_phase = BlockPhase::Linear1;
        {
            uint32_t block_global_idx = 0;
            #pragma unroll 1
            for (uint32_t e = 0; e < kNumExperts; ++e) {
                const uint32_t num_mblk = per_expert_num_m_blocks[e];
                const uint32_t pool_off = per_expert_m_block_start[e];
                for (uint32_t mb = 0; mb < num_mblk; ++mb) {
                    for (uint32_t nb = 0; nb < kNumL1BlockNs; ++nb) {
                        if (block_global_idx % kNumSMs == sm_idx) {
                            current_expert_idx         = e;
                            current_m_block_idx        = mb;
                            current_n_block_idx        = nb;
                            current_pool_block_offset  = pool_off + mb;
                            const uint32_t m_left = per_expert_num_valid_tokens[e] - mb * BLOCK_M;
                            current_valid_m = m_left < BLOCK_M ? m_left : BLOCK_M;
                            fn(BlockPhase::Linear1, e, kNumL1KBlocks, mb, nb);
                        }
                        ++block_global_idx;
                    }
                }
            }
        }

        // -------- Phase 2: L2 -----------
        current_phase = BlockPhase::Linear2;
        {
            uint32_t block_global_idx = 0;
            #pragma unroll 1
            for (uint32_t e = 0; e < kNumExperts; ++e) {
                const uint32_t num_mblk = per_expert_num_m_blocks[e];
                const uint32_t pool_off = per_expert_m_block_start[e];
                for (uint32_t mb = 0; mb < num_mblk; ++mb) {
                    for (uint32_t nb = 0; nb < kNumL2BlockNs; ++nb) {
                        if (block_global_idx % kNumSMs == sm_idx) {
                            current_expert_idx         = e;
                            current_m_block_idx        = mb;
                            current_n_block_idx        = nb;
                            current_pool_block_offset  = pool_off + mb;
                            const uint32_t m_left = per_expert_num_valid_tokens[e] - mb * BLOCK_M;
                            current_valid_m = m_left < BLOCK_M ? m_left : BLOCK_M;
                            fn(BlockPhase::Linear2, e, kNumL2KBlocks, mb, nb);
                        }
                        ++block_global_idx;
                    }
                }
            }
        }

        current_phase = BlockPhase::None;
    }

    __device__ __forceinline__ uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    __device__ __forceinline__ uint32_t get_current_valid_m() const {
        return current_valid_m;
    }

    __device__ __forceinline__ uint32_t get_pool_block_offset(uint32_t expert_idx) const {
        return per_expert_m_block_start[expert_idx];
    }

    __device__ __forceinline__ uint32_t get_num_tokens(uint32_t expert_idx) const {
        return per_expert_num_valid_tokens[expert_idx];
    }

    // Total number of M-blocks across all experts (for intermediate buffer sizing)
    __device__ __forceinline__ uint32_t get_total_m_blocks() const {
        return total_m_blocks;
    }
};

} // namespace asym_gemm
