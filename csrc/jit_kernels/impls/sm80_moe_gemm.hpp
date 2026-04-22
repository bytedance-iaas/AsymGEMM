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

namespace asym_gemm {

// Non-templated params struct — void* data pointers so this struct is layout-stable
// across the type-erased launch_kernel() boundary. The kernel casts to Element* internally.
struct SM80MoEParams {
    void*    x_ptr;         // [total_tokens, K] row-major
    void*    w_ptr;         // [num_experts, N, K] row-major
    void*    o_ptr;         // [total_tokens, N] row-major
    int32_t* expert_list;   // [list_size] expert IDs
    int32_t* index_list;    // [list_size] cumulative end-token indices
    int32_t  list_size;
    int32_t  expert_size;   // num_experts (outer dim of W)
    int64_t  N;
    int64_t  K;
};

class SM80MoEGemmRuntime final : public LaunchRuntime<SM80MoEGemmRuntime> {
public:
    struct Args {
        int64_t n, k;
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
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& o,
    const torch::Tensor& expert_list,
    const torch::Tensor& index_list,
    int64_t N, int64_t K, int32_t num_experts, int32_t list_size,
    const std::string& element_type_str)
{
    // Select block config based on current device arch
    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_config(arch_major, arch_minor,
                                              static_cast<int>(N),
                                              static_cast<int>(K));

    const SM80MoEParams params {
        .x_ptr        = x.data_ptr(),
        .w_ptr        = w.data_ptr(),
        .o_ptr        = o.data_ptr(),
        .expert_list  = expert_list.data_ptr<int32_t>(),
        .index_list   = index_list.data_ptr<int32_t>(),
        .list_size    = list_size,
        .expert_size  = num_experts,
        .N            = N,
        .K            = K,
    };

    const SM80MoEGemmRuntime::Args args {
        .n             = N,
        .k             = K,
        .gemm_config   = cfg,
        .launch_args   = LaunchArgs({cfg.grid_x(static_cast<int>(N)), 1},
                                    cfg.num_threads(),
                                    cfg.smem_bytes()),
        .element_type_str = element_type_str,
        .params        = params,
    };

    // Kernel name encodes dtype for separate CUBIN cache entries
    const std::string kernel_name = fmt::format("sm80_moe_gemm_{}",
        (element_type_str.find("half") != std::string::npos) ? "fp16" : "bf16");

    const auto& code    = SM80MoEGemmRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEGemmRuntime::launch(runtime, args);
}

}  // namespace asym_gemm
