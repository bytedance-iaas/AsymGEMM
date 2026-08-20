// csrc/jit_kernels/impls/sm89_fp8_asym_gemm.hpp
#pragma once

#include <torch/python.h>
#include <cstdint>
#include <optional>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm80.hpp"
#include <asym_gemm/impls/sm89_fp8_moe_params.h>

namespace asym_gemm {

// ──────────────────────────────────────────────────────────────────────────────
// FP8 JIT runtime — SM89 native FP8 MMA
// ──────────────────────────────────────────────────────────────────────────────
class SM89MoEFP8GemmRuntime final : public LaunchRuntime<SM89MoEFP8GemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM89MoEFP8Params     params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
// sm89 moe fp8 v4: split header (sm89_fp8_moe_gemm.cuh)
#include <asym_gemm/impls/sm89_fp8_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_impl<{}, {}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps,
            args.params.scale_a_blk_ptr != nullptr);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void sm89_m_grouped_fp8_moe_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_list,
    const torch::Tensor& index_list,
    int64_t N, int64_t K,
    int32_t num_experts, int32_t list_size,
    float scale_a, float scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_a_block = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_block = std::nullopt)
{
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const bool block_scale = scale_a_block.has_value();
    const auto cfg = sm80::select_sm80_fp8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K),
                                                   block_scale);

    const SM89MoEFP8Params params {
        .x_ptr       = a.data_ptr(),
        .w_ptr       = b.data_ptr(),
        .o_ptr       = d.data_ptr(),
        .expert_list = expert_list.data_ptr<int32_t>(),
        .index_list  = index_list.data_ptr<int32_t>(),
        .list_size   = list_size,
        .expert_size = num_experts,
        .N           = N,
        .K           = K,
        .scale_a     = scale_a,
        .scale_b     = scale_b,
        .scale_a_ptr = scale_a_tensor.has_value()
            ? scale_a_tensor->data_ptr<float>() : nullptr,
        .scale_b_ptr = scale_b_tensor.has_value()
            ? scale_b_tensor->data_ptr<float>() : nullptr,
        .scale_a_blk_ptr = block_scale
            ? scale_a_block->data_ptr<float>() : nullptr,
        .scale_b_blk_ptr = scale_b_block.has_value()
            ? scale_b_block->data_ptr<float>() : nullptr,
        .sa_kg       = static_cast<int32_t>((K + 127) / 128),
        .sb_ng       = static_cast<int32_t>((N + 127) / 128),
    };

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k,
                                                block_scale);

    const SM89MoEFP8GemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), list_size},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm89_moe_fp8_gemm_bm{}_bn{}_bk{}_blk{}",
        cfg.block_m, cfg.block_n, cfg.block_k, block_scale ? 1 : 0);

    const auto& code    = SM89MoEFP8GemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM89MoEFP8GemmRuntime::launch(runtime, runtime_args);
}

// ──────────────────────────────────────────────────────────────────────────────
// Masked FP8 JIT runtime — SM89 native FP8 MMA, padded [G, M_max, K] layout
// ──────────────────────────────────────────────────────────────────────────────
class SM89MoEFP8MaskedGemmRuntime final : public LaunchRuntime<SM89MoEFP8MaskedGemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM89MoEFP8MaskedParams params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
// sm89 moe fp8 v4: split header (sm89_fp8_moe_gemm.cuh)
#include <asym_gemm/impls/sm89_fp8_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_masked_impl<{}, {}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps,
            args.params.scale_a_blk_ptr != nullptr);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void sm89_m_grouped_fp8_moe_gemm_masked(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& masked_m,
    int64_t M_max, int64_t N, int64_t K,
    int32_t num_groups,
    float scale_a, float scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_a_block = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_block = std::nullopt)
{
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const bool block_scale = scale_a_block.has_value();
    const auto cfg = sm80::select_sm80_fp8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K),
                                                   block_scale);

    const SM89MoEFP8MaskedParams params {
        .x_ptr       = a.data_ptr(),
        .w_ptr       = b.data_ptr(),
        .o_ptr       = d.data_ptr(),
        .masked_m    = masked_m.data_ptr<int32_t>(),
        .num_groups  = num_groups,
        .M_max       = M_max,
        .N           = N,
        .K           = K,
        .scale_a     = scale_a,
        .scale_b     = scale_b,
        .scale_a_ptr = scale_a_tensor.has_value()
            ? scale_a_tensor->data_ptr<float>() : nullptr,
        .scale_b_ptr = scale_b_tensor.has_value()
            ? scale_b_tensor->data_ptr<float>() : nullptr,
        .scale_a_blk_ptr = block_scale
            ? scale_a_block->data_ptr<float>() : nullptr,
        .scale_b_blk_ptr = scale_b_block.has_value()
            ? scale_b_block->data_ptr<float>() : nullptr,
        .sa_kg       = static_cast<int32_t>((K + 127) / 128),
        .sb_ng       = static_cast<int32_t>((N + 127) / 128),
    };

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k,
                                                block_scale);

    const SM89MoEFP8MaskedGemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), num_groups},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm89_moe_fp8_gemm_masked_bm{}_bn{}_bk{}_blk{}",
        cfg.block_m, cfg.block_n, cfg.block_k, block_scale ? 1 : 0);

    const auto& code    = SM89MoEFP8MaskedGemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM89MoEFP8MaskedGemmRuntime::launch(runtime, runtime_args);
}


// ──────────────────────────────────────────────────────────────────────────────
// Architecture-agnostic FP8 MoE entry points (dispatched from csrc/apis/gemm.hpp).
// ──────────────────────────────────────────────────────────────────────────────
// Validate block-scale tensors shared by the contiguous and masked SM89 APIs.
// rows: total_tokens (contiguous) or num_groups * M_max (masked).
static void check_sm89_block_scales(
    const std::optional<torch::Tensor>& scale_a_block,
    const std::optional<torch::Tensor>& scale_b_block,
    const std::optional<torch::Tensor>& scale_a_tensor,
    const std::optional<torch::Tensor>& scale_b_tensor,
    int64_t rows, int64_t num_experts, int64_t N, int64_t K)
{
    DG_HOST_ASSERT(scale_a_block.has_value() == scale_b_block.has_value());
    if (!scale_a_block.has_value())
        return;
    DG_HOST_ASSERT(!scale_a_tensor.has_value() && !scale_b_tensor.has_value());

    const int64_t kg = (K + 127) / 128;
    const int64_t ng = (N + 127) / 128;
    const auto& sa = *scale_a_block;
    const auto& sb = *scale_b_block;
    DG_HOST_ASSERT(sa.is_cuda() && sa.is_contiguous());
    DG_HOST_ASSERT(sa.is_cuda() && sa.is_contiguous());
    DG_HOST_ASSERT((sb.is_cuda() || (sb.device().is_cpu() && sb.is_pinned())) && sb.is_contiguous());
    DG_HOST_ASSERT(sb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(sa.numel() == rows * kg);
    DG_HOST_ASSERT(sb.numel() == num_experts * ng * kg);
}

static void m_grouped_fp8_asym_gemm_sm89(
    const torch::Tensor& a,        // [total_tokens, K]   float8_e4m3fn  (HBM)
    const torch::Tensor& b,        // [num_experts, N, K] float8_e4m3fn  (CPU pinned or HBM)
    const torch::Tensor& d,        // [total_tokens, N]   bfloat16       (HBM)
    const torch::Tensor& offsets,  // [list_size] int32 cumulative end indices
    const torch::Tensor& experts,  // [list_size] int32 expert IDs
    const int&           list_size,
    const float&         scale_a,
    const float&         scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_a_block = std::nullopt,  // [total_tokens, ceil(K/128)] f32
    const std::optional<torch::Tensor>& scale_b_block = std::nullopt)  // [num_experts, ceil(N/128), ceil(K/128)] f32
{
    DG_HOST_ASSERT(a.dim() == 2 && b.dim() == 3 && d.dim() == 2);

    const int64_t total_tokens = a.size(0);
    const int64_t K            = a.size(1);
    const int64_t num_experts  = b.size(0);
    const int64_t N            = b.size(1);
    DG_HOST_ASSERT(b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == total_tokens && d.size(1) == N);

    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // a and d must be on CUDA; b may be on CPU pinned memory (PCIe) or CUDA
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());

    DG_HOST_ASSERT(offsets.is_cuda() && experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() && experts.is_contiguous());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(experts.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);

    check_sm89_block_scales(scale_a_block, scale_b_block,
                            scale_a_tensor, scale_b_tensor,
                            total_tokens, num_experts, N, K);

    if (total_tokens == 0 || N == 0 || K == 0) return;

    DG_HOST_ASSERT(K % 32 == 0 && K >= 32);   // SM89 FP8 MMA K-atom = 32
    DG_HOST_ASSERT(N % 32 == 0);

    sm89_m_grouped_fp8_moe_gemm_contiguous(
        a, b, d, experts, offsets,
        N, K,
        static_cast<int32_t>(num_experts),
        static_cast<int32_t>(list_size),
        scale_a, scale_b,
        scale_a_tensor, scale_b_tensor,
        scale_a_block, scale_b_block);
}

static void m_grouped_fp8_asym_gemm_sm89_masked(
    const torch::Tensor& a,        // [num_groups, M_max, K]  float8_e4m3fn
    const torch::Tensor& b,        // [num_groups, N, K]      float8_e4m3fn
    const torch::Tensor& d,        // [num_groups, M_max, N]  bfloat16
    const torch::Tensor& masked_m, // [num_groups]            int32
    const int&           expected_m,
    const float&         scale_a,
    const float&         scale_b,
    const std::optional<torch::Tensor>& scale_a_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_b_tensor = std::nullopt,
    const std::optional<torch::Tensor>& scale_a_block = std::nullopt,  // [num_groups, M_max, ceil(K/128)] f32
    const std::optional<torch::Tensor>& scale_b_block = std::nullopt)  // [num_groups, ceil(N/128), ceil(K/128)] f32
{
    DG_HOST_ASSERT(a.dim() == 3 && b.dim() == 3 && d.dim() == 3);

    const int64_t num_groups = a.size(0);
    const int64_t M_max      = a.size(1);
    const int64_t K          = a.size(2);
    const int64_t N          = b.size(1);
    DG_HOST_ASSERT(b.size(0) == num_groups && b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == num_groups && d.size(1) == M_max && d.size(2) == N);

    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // a and d must be on CUDA; b may be on CPU pinned memory (PCIe) or CUDA
    DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() && b.is_contiguous() && d.is_contiguous());

    DG_HOST_ASSERT(masked_m.is_cuda() && masked_m.is_contiguous());
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);

    check_sm89_block_scales(scale_a_block, scale_b_block,
                            scale_a_tensor, scale_b_tensor,
                            num_groups * M_max, num_groups, N, K);

    if (M_max == 0 || N == 0 || K == 0 || expected_m == 0) return;

    DG_HOST_ASSERT(K % 32 == 0 && K >= 32);
    DG_HOST_ASSERT(N % 32 == 0);

    sm89_m_grouped_fp8_moe_gemm_masked(
        a, b, d, masked_m,
        M_max, N, K,
        static_cast<int32_t>(num_groups),
        scale_a, scale_b,
        scale_a_tensor, scale_b_tensor,
        scale_a_block, scale_b_block);
}

}  // namespace asym_gemm
