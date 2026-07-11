#pragma once

#include <asym_gemm/common/types.hpp>
#include <asym_gemm/common/utils.cuh>

namespace asym_gemm {

enum class IndexType {
    MN,
    K,
    SF_K,
};

// Works for integral types (int, uint32_t, int64_t, etc.)
template <class T>
__host__ __device__ __forceinline__
T ceil_div_device(T a, T b) {
    static_assert(std::is_integral<T>::value, "ceil_div requires integral type");
    // Assumes b > 0
    return (a + b - 1) / b;
}

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto& candidate: {8u, 16u}) {
        const auto& usage = kIsMulticastOnA ?
                    candidate * BLOCK_N + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
                    candidate * BLOCK_M + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,  // for k-grouped GEMM only: 128 (SM90 float SF) or 512 (SM100 UE8M0 SF)
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct asymScheduler {
    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // asym - per-expert offset pairs for M dimension
    // offsets array is laid out as: [start_0, end_0, start_1, end_1, ..., start_N, end_N]
    // For each expert blockIdx.y, reads offsets[blockIdx.y*2] and offsets[blockIdx.y*2+1]
    // m_start = ceil_div(offsets[blockIdx.y*2], BLOCK_M)
    // m_end = ceil_div(offsets[blockIdx.y*2+1], BLOCK_M)
    uint32_t m_start, m_end;
    uint32_t n_start, n_end;
    uint32_t expert_id;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    uint32_t blocks_perExpert = 0;
    uint32_t n_idx = 0;
    // Local n-block index of THIS CTA's work item. Equals blockIdx.x in the static path;
    // equals the popped item's n-block under the sEP queued kernel. All kernel-side uses of
    // blockIdx.x-as-n-position must read this instead (epilogue store, transpose-B path).
    uint32_t n_blk = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    uint32_t next_group_idx, next_shape_k;

    // ReSharper disable once CppPossiblyUninitializedMember
    __device__ __forceinline__ explicit asymScheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                                  uint32_t* experts, uint32_t* offsets)
        : asymScheduler(shape_m, shape_n, experts, offsets, blockIdx.y, blockIdx.x) {}

    // sEP (gb200_ep.md E3): explicit-ids variant — the queued kernel pops a work item
    // (segment, n-block) from shared coherent host counters instead of reading blockIdx.
    // The default ctor above delegates with (blockIdx.y, blockIdx.x), so the static path
    // is byte-identical.
    // ReSharper disable once CppPossiblyUninitializedMember
    __device__ __forceinline__ asymScheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                                  uint32_t* experts, uint32_t* offsets,
                                                  const uint32_t seg_idx, const uint32_t nblk_idx) {
        blocks_perExpert = ceil_div_device(shape_n, BLOCK_N);
        n_blk = nblk_idx;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            // DeepGEMM-style masked scheduler: `offsets` is reused as int32 masked_m[num_groups]
            // (token counts per group). `experts` is unused.
            // gridDim.y == num_groups (compile-time constant) so blockIdx.y IS the expert id.
            // shape_m == max_m (the fixed per-group M allocation, same for every group).
            expert_id = seg_idx;
            current_group_idx = seg_idx;
            n_idx = nblk_idx * BLOCK_N + shape_n * expert_id;
            const int m_count = reinterpret_cast<const int*>(offsets)[seg_idx];
            m_start = 0;
            m_end = ceil_div_device(static_cast<uint32_t>(m_count > 0 ? m_count : 0), BLOCK_M);
        } else {
            expert_id = experts[seg_idx];
            n_start = expert_id * blocks_perExpert;

            // offsets array is laid out as pairs: [start_0, end_0, start_1, end_1, ...]
            // seg_idx indexes into the experts array; multiply by 2 to get offset pair index
            uint32_t offset_pair_idx = seg_idx * 2;
            m_start = ceil_div_device(offsets[offset_pair_idx], BLOCK_M);
            m_end = ceil_div_device(offsets[offset_pair_idx + 1], BLOCK_M);

            // B is laid out as [group, n, k], so N offset must use expert_id (group id),
            // not seg_idx (segment id in offsets/experts list).
            n_idx = nblk_idx * BLOCK_N + shape_n * expert_id;
            current_group_idx = expert_id;
        }
    }

    // template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    // __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
    //                                                    const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
    //     if constexpr (kGemmType == GemmType::Normal) {
    //         return block_idx * block_size;
    //     } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
    //         const auto offset = kWithGroupOffset ? cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M)) : 0;
    //         return offset * shape_dim + block_idx * block_size;
    //     } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
    //         const auto offset = kWithGroupOffset ? current_group_idx : 0;
    //         return offset * shape_dim + block_idx * block_size;
    //     } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
    //         auto offset = 0;
    //         if constexpr (kWithGroupOffset) {
    //             if constexpr (kIndexType == IndexType::MN)
    //                 offset = current_group_idx * shape_dim;
    //             else if constexpr (kIndexType == IndexType::K)
    //                 offset = current_k_cumsum;
    //             else if constexpr (kIndexType == IndexType::SF_K)
    //                 offset = current_sf_k_cumsum;
    //         }
    //         return offset + block_idx * block_size;
    //     } else if constexpr (kGemmType == GemmType::Batched) {
    //         // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
    //         const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
    //         return offset * shape_dim + block_idx * block_size;
    //     }
    // }
};

#pragma clang diagnostic pop

} // namespace asym_gemm
