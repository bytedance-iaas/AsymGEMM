// csrc/jit_kernels/impls/sm89_fp8_moe_gemm.hpp
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
#include <asym_gemm/impls/sm80_moe_params.h>

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
// sm89 moe fp8 v2: block-scale (1x128/128x128) support
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
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

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k);

    const SM89MoEFP8GemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), list_size},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm89_moe_fp8_gemm_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

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
// sm89 moe fp8 v2: block-scale (1x128/128x128) support
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm89_moe_fp8_gemm_masked_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
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

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k);

    const SM89MoEFP8MaskedGemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), num_groups},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm89_moe_fp8_gemm_masked_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM89MoEFP8MaskedGemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM89MoEFP8MaskedGemmRuntime::launch(runtime, runtime_args);
}

}  // namespace asym_gemm
