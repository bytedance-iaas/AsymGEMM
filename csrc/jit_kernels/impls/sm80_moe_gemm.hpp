// csrc/jit_kernels/impls/sm80_moe_gemm.hpp
#pragma once

#include <torch/python.h>
#include <string>
#include <cstdint>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm80.hpp"
#include <asym_gemm/impls/sm80_moe_params.h>

namespace asym_gemm {

class SM80MoEGemmRuntime final : public LaunchRuntime<SM80MoEGemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs launch_args;
        std::string element_type_str;   // "cutlass::half_t" or "cutlass::bfloat16_t"
        SM80MoEParams params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_moe_gemm_impl<{}, {}, {}, {}, {}>);
}};
)",
            c.block_m,
            c.block_n,
            c.block_k,
            c.nwarps,
            args.element_type_str);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        // SM80MoEParams is a plain struct with only scalar and raw-pointer fields,
        // so it can be passed by value through launch_kernel's void* array.
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Free function called from the API layer
// ──────────────────────────────────────────────────────────────────────────────
static void sm80_m_grouped_moe_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_list,   // expert IDs   (= API's "experts")
    const torch::Tensor& index_list,    // cumul. ends  (= API's "offsets")
    int64_t N, int64_t K, int32_t num_experts, int32_t list_size,
    const std::string& element_type_str)
{
    // Validate dtype string early — gives a readable error rather than an NVCC failure
    DG_HOST_ASSERT(element_type_str == "cutlass::half_t" or
                   element_type_str == "cutlass::bfloat16_t");

    // Select block config based on current device arch
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_config(arch_major, arch_minor,
                                              static_cast<int>(N),
                                              static_cast<int>(K));

    const SM80MoEParams params {
        .x_ptr        = a.data_ptr(),
        .w_ptr        = b.data_ptr(),
        .o_ptr        = d.data_ptr(),
        .expert_list  = expert_list.data_ptr<int32_t>(),
        .index_list   = index_list.data_ptr<int32_t>(),
        .list_size    = list_size,
        .expert_size  = num_experts,
        .N            = N,
        .K            = K,
    };

    const SM80MoEGemmRuntime::Args args {
        .gemm_config   = cfg,
        .launch_args   = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                                    cfg.num_threads(),
                                    cfg.smem_bytes()),
        .element_type_str = element_type_str,
        .params        = params,
    };

    // Kernel name encodes dtype + tile dims so different configs get separate CUBINs
    const std::string kernel_name = fmt::format("sm80_moe_gemm_{}_bm{}_bn{}_bk{}",
        (element_type_str == "cutlass::half_t") ? "fp16" : "bf16",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEGemmRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEGemmRuntime::launch(runtime, args);
}

// ──────────────────────────────────────────────────────────────────────────────
// FP8 JIT runtime — SM89 native FP8 MMA
// ──────────────────────────────────────────────────────────────────────────────
class SM80MoEFP8GemmRuntime final : public LaunchRuntime<SM80MoEFP8GemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM80MoEFP8Params     params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
#include <asym_gemm/impls/sm80_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_moe_fp8_gemm_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void sm80_m_grouped_fp8_moe_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& expert_list,
    const torch::Tensor& index_list,
    int64_t N, int64_t K,
    int32_t num_experts, int32_t list_size,
    float scale_a, float scale_b)
{
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_fp8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K));

    const SM80MoEFP8Params params {
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
    };

    const int smem_bytes = sm80::smem_bytes_fp8(cfg.block_m, cfg.block_n, cfg.block_k);

    const SM80MoEFP8GemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                                  cfg.num_threads(),
                                  smem_bytes),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm80_moe_fp8_gemm_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEFP8GemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEFP8GemmRuntime::launch(runtime, runtime_args);
}

}  // namespace asym_gemm
