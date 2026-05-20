#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace asym_gemm {

class SM90BF16AsymGemmRuntime final: public LaunchRuntime<SM90BF16AsymGemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm90_bf16_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_asym_gemm_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages,
        args.gemm_config.thread_config.num_math_threads, args.gemm_config.thread_config.num_tma_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        args.gemm_config.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void add_sm90_bf16_cd_double_buffer_smem(GemmConfig& config, const torch::Tensor& d) {
    config.smem_config.smem_size += align(
        config.block_m * config.block_n * static_cast<int>(d.element_size()), 1024);
    DG_HOST_ASSERT(config.smem_config.smem_size <= SM90ArchSpec::smem_capacity);
}

static void sm90_m_grouped_bf16_asym_gemm_contiguous(const torch::Tensor& a,
                                                     const torch::Tensor& b,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& offsets_t,
                                                     const torch::Tensor& experts_t,
                                                     const int& list_size,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims,
                                                     const int b_outer_stride = -1) {
    constexpr int block_m = 64;
    constexpr int block_n = 64;
    const int block_k = (major_b == cute::UMMA::Major::MN) ? 64 : 512;
    const auto& aligned_k = align(k, block_k);

    auto config = get_manual_config_asym<SM90ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, num_groups, major_a, major_b,
        torch::kBFloat16, d.scalar_type(), false,
        device_runtime->get_num_sms(),
        block_m, block_n, block_k);
    config.multicast_config = {1, false};
    config.smem_config = get_smem_config<SM90ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, config.block_m, config.block_n, config.block_k,
        major_a, major_b, torch::kBFloat16, d.scalar_type(),
        config.num_stages, config.multicast_config, true);
    add_sm90_bf16_cd_double_buffer_smem(config, d);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const int outer_b = (b_outer_stride >= 0)
        ? b_outer_stride
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               outer_b, num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);

    if (list_size <= 1)
        return;

    const SM90BF16AsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };

    const auto& code = SM90BF16AsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_m_grouped_asym_gemm_contiguous", code);
    SM90BF16AsymGemmRuntime::launch(runtime, args);
}

static void sm90_m_grouped_bf16_asym_gemm_masked(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m_t,
                                                 const int& expected_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    constexpr int block_m = 64;
    constexpr int block_n = 64;
    constexpr int block_k = 512;
    const auto& aligned_k = align(k, block_k);

    auto config = get_manual_config_asym<SM90ArchSpec>(
        GemmType::MGroupedMasked, KernelType::KernelNoSF,
        expected_m, n, k, num_groups, major_a, major_b,
        torch::kBFloat16, d.scalar_type(), false,
        device_runtime->get_num_sms(),
        block_m, block_n, block_k);
    config.multicast_config = {1, false};
    config.smem_config = get_smem_config<SM90ArchSpec>(
        GemmType::MGroupedMasked, KernelType::KernelNoSF,
        expected_m, n, k, config.block_m, config.block_n, config.block_k,
        major_a, major_b, torch::kBFloat16, d.scalar_type(),
        config.num_stages, config.multicast_config, true);
    add_sm90_bf16_cd_double_buffer_smem(config, d);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);

    const SM90BF16AsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = masked_m_t.data_ptr<int>(),
        .experts = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };

    const auto& code = SM90BF16AsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_bf16_m_grouped_asym_gemm_masked", code);
    SM90BF16AsymGemmRuntime::launch(runtime, args);
}

} // namespace asym_gemm
