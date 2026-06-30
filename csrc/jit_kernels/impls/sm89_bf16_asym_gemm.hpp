// csrc/jit_kernels/impls/sm89_bf16_asym_gemm.hpp
#pragma once

#include "sm80_moe_gemm.hpp"

namespace asym_gemm {

// SM89 BF16 masked MoE reuses the shared SM80-style grouped MoE kernel by
// flattening the padded [G, M_max, K] layout and inserting gap segments for
// padded rows. The shared kernel skips those gaps in-kernel.
static void sm89_m_grouped_bf16_moe_gemm_masked(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& masked_m,
    int64_t M_max,
    int64_t N,
    int64_t K,
    int32_t num_groups)
{
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda() && masked_m.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());
    DG_HOST_ASSERT(masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.is_cuda() || (b.device().is_cpu() && b.is_pinned()));

    auto opts_i32 = torch::TensorOptions()
        .device(masked_m.device())
        .dtype(torch::kInt32);
    auto group_ids = torch::arange(num_groups, opts_i32);
    auto gap_ids = torch::full({num_groups}, -1, opts_i32);
    auto experts = torch::stack({group_ids, gap_ids}, 1).reshape({2 * num_groups});

    const auto m_max_i32 = static_cast<int32_t>(M_max);
    auto base = group_ids * m_max_i32;
    auto valid_ends = base + masked_m;
    auto gap_ends = base + m_max_i32;
    auto index_list = torch::stack({valid_ends, gap_ends}, 1).reshape({2 * num_groups});

    auto total_rows = static_cast<int64_t>(num_groups) * M_max;
    auto flat_a = a.view({total_rows, K});
    auto flat_d = d.view({total_rows, N});

    sm80_m_grouped_moe_gemm_contiguous(
        flat_a,
        b,
        flat_d,
        experts,
        index_list,
        N,
        K,
        num_groups,
        2 * num_groups,
        "cutlass::bfloat16_t");
}

}  // namespace asym_gemm
