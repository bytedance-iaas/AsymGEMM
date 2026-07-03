// Copyright (c) 2026.

#include "../apis/qwen3_moe.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "../jit_kernels/impls/sm100_bf16_asym_gemm.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"

namespace asym_gemm::qwen3_moe {
namespace {

void check_sm100() {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10 && "Qwen3 MoE routed kernels require SM100");
}

void check_group_metadata(const torch::Tensor& offsets, const torch::Tensor& experts, int64_t list_size) {
    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt && experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(list_size >= 2);
    DG_HOST_ASSERT(offsets.numel() >= 2 * (list_size - 1));
    DG_HOST_ASSERT(experts.numel() >= list_size);
}

void check_routes(
    const torch::Tensor& token_indices,
    const torch::Tensor& routing_weights,
    int64_t route_rows,
    bool weighted) {
    DG_HOST_ASSERT(token_indices.is_cuda());
    DG_HOST_ASSERT(token_indices.is_contiguous());
    DG_HOST_ASSERT(token_indices.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(token_indices.numel() >= route_rows);
    if (weighted) {
        DG_HOST_ASSERT(routing_weights.defined());
        DG_HOST_ASSERT(routing_weights.is_cuda());
        DG_HOST_ASSERT(routing_weights.is_contiguous());
        DG_HOST_ASSERT(routing_weights.numel() >= route_rows);
        DG_HOST_ASSERT(routing_weights.scalar_type() == torch::kBFloat16 || routing_weights.scalar_type() == torch::kFloat);
    }
}

bool route_weights_are_bf16(const torch::Tensor& routing_weights, bool weighted) {
    return weighted && routing_weights.defined() && routing_weights.scalar_type() == torch::kBFloat16;
}

void check_bf16_2d_cuda(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.is_cuda());
    DG_HOST_ASSERT(t.dim() == 2);
    DG_HOST_ASSERT(t.is_contiguous());
    DG_HOST_ASSERT(t.scalar_type() == torch::kBFloat16);
}

void check_fp32_2d_cuda(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.is_cuda());
    DG_HOST_ASSERT(t.dim() == 2);
    DG_HOST_ASSERT(t.is_contiguous());
    DG_HOST_ASSERT(t.scalar_type() == torch::kFloat);
}

void check_bf16_3d_weight_cpu(const torch::Tensor& t) {
    DG_HOST_ASSERT(!t.is_cuda());
    DG_HOST_ASSERT(t.dim() == 3);
    DG_HOST_ASSERT(t.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(t.is_pinned());
    major_check(t);
}

void launch_routed(
    const torch::Tensor& a,
    const torch::Tensor& weight_cpu,
    const torch::Tensor& d,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const torch::Tensor& token_indices,
    const torch::Tensor& routing_weights,
    int64_t list_size,
    bool weighted,
    const std::string& compiled_dims,
    bool transpose_b,
    bool gather_left,
    bool scatter_add,
    const torch::Tensor& route_scatter_out,
    const torch::Tensor& route_gather_left) {
    check_sm100();
    check_group_metadata(offsets, experts, list_size);

    const auto major_a = get_major_type_ab(a);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    cute::UMMA::Major major_b;
    if (transpose_b) {
        major_check(weight_cpu);
        major_b = cute::UMMA::Major::MN;
    } else {
        major_b = get_major_type_ab(weight_cpu);
    }
    const int m = scatter_add ? static_cast<int>(a.size(0)) : static_cast<int>(d.size(0));
    check_routes(token_indices, routing_weights, static_cast<int64_t>(m), weighted);
    const auto [num_groups, n_phys, k_phys] = get_shape<3>(weight_cpu);
    const int n = transpose_b ? k_phys : n_phys;
    const int k = transpose_b ? n_phys : k_phys;
    DG_HOST_ASSERT(num_groups > 0 && n > 0 && k > 0);
    DG_HOST_ASSERT(a.size(1) == k);
    DG_HOST_ASSERT(d.size(1) == n);
    DG_HOST_ASSERT(offsets.numel() >= 2 * (list_size - 1) && experts.numel() >= list_size);

    check_major_type_cd(d);
    const int b_outer_stride = transpose_b
        ? static_cast<int>(weight_cpu.stride(-2))
        : static_cast<int>(weight_cpu.stride(get_non_contiguous_dim(major_b)));

    const bool weighted_kernel = weighted;
    const bool weights_bf16 = route_weights_are_bf16(routing_weights, weighted_kernel);
    const int grid_y = static_cast<int>(list_size) - 1;
    sm100_m_grouped_bf16_asym_gemm_qwen3_routed(
        a,
        weight_cpu,
        d,
        offsets,
        experts,
        token_indices,
        routing_weights,
        grid_y,
        num_groups,
        m,
        n,
        k,
        major_a,
        major_b,
        compiled_dims,
        b_outer_stride,
        gather_left,
        scatter_add,
        weighted_kernel,
        weights_bf16,
        scatter_add ? static_cast<int>(route_scatter_out.stride(0)) : 0,
        gather_left ? static_cast<int>(route_gather_left.stride(0)) : 0,
        route_scatter_out,
        route_gather_left);
}

}  // namespace

void qwen3_moe_bf16_down_forward_scatter_add_(
    const torch::Tensor& act,
    const torch::Tensor& weight_cpu,
    const torch::Tensor& out_fp32,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const torch::Tensor& token_indices,
    const torch::Tensor& routing_weights,
    int64_t list_size,
    bool weighted,
    const std::string& compiled_dims) {
    c10::cuda::CUDAGuard device_guard(out_fp32.device());
    check_bf16_2d_cuda(act);
    check_bf16_3d_weight_cpu(weight_cpu);
    check_fp32_2d_cuda(out_fp32);
    check_routes(token_indices, routing_weights, static_cast<int64_t>(act.size(0)), weighted);
    DG_HOST_ASSERT(weight_cpu.size(1) == out_fp32.size(1));
    DG_HOST_ASSERT(weight_cpu.size(2) == act.size(1));
    launch_routed(
        act,
        weight_cpu,
        out_fp32,
        offsets,
        experts,
        token_indices,
        routing_weights,
        list_size,
        weighted,
        compiled_dims,
        /*transpose_b=*/false,
        /*gather_left=*/false,
        /*scatter_add=*/true,
        out_fp32,
        torch::Tensor());
}

// Not on the current best benchmark paths; ker101 leaves down-dx gather disabled.
void qwen3_moe_bf16_down_dx_gather_left_(
    const torch::Tensor& grad_token,
    const torch::Tensor& weight_cpu,
    const torch::Tensor& grad_act,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const torch::Tensor& token_indices,
    const torch::Tensor& routing_weights,
    int64_t list_size,
    bool weighted,
    const std::string& compiled_dims) {
    c10::cuda::CUDAGuard device_guard(grad_act.device());
    check_bf16_2d_cuda(grad_token);
    check_bf16_3d_weight_cpu(weight_cpu);
    check_bf16_2d_cuda(grad_act);
    check_routes(token_indices, routing_weights, static_cast<int64_t>(grad_act.size(0)), weighted);
    DG_HOST_ASSERT(weight_cpu.size(1) == grad_token.size(1));
    DG_HOST_ASSERT(weight_cpu.size(2) == grad_act.size(1));
    launch_routed(
        grad_token,
        weight_cpu,
        grad_act,
        offsets,
        experts,
        token_indices,
        routing_weights,
        list_size,
        weighted,
        compiled_dims,
        /*transpose_b=*/true,
        /*gather_left=*/true,
        /*scatter_add=*/false,
        torch::Tensor(),
        grad_token);
}

void qwen3_moe_bf16_gateup_dx_scatter_add_(
    const torch::Tensor& grad_expert,
    const torch::Tensor& weight_cpu,
    const torch::Tensor& grad_hidden_fp32,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    const torch::Tensor& token_indices,
    const torch::Tensor& routing_weights,
    int64_t list_size,
    bool weighted,
    const std::string& compiled_dims) {
    c10::cuda::CUDAGuard device_guard(grad_hidden_fp32.device());
    check_bf16_2d_cuda(grad_expert);
    check_bf16_3d_weight_cpu(weight_cpu);
    check_fp32_2d_cuda(grad_hidden_fp32);
    check_routes(token_indices, routing_weights, static_cast<int64_t>(grad_expert.size(0)), weighted);
    DG_HOST_ASSERT(weight_cpu.size(1) == grad_expert.size(1));
    DG_HOST_ASSERT(weight_cpu.size(2) == grad_hidden_fp32.size(1));
    launch_routed(
        grad_expert,
        weight_cpu,
        grad_hidden_fp32,
        offsets,
        experts,
        token_indices,
        routing_weights,
        list_size,
        weighted,
        compiled_dims,
        /*transpose_b=*/true,
        /*gather_left=*/false,
        /*scatter_add=*/true,
        grad_hidden_fp32,
        torch::Tensor());
}

}  // namespace asym_gemm::qwen3_moe
