#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace asym_gemm {

// kNumStagesB: number of B (expert weight) pipeline stages (Phase 1 double-buffering)
static constexpr int kMegaNumStagesB = 2;

// ============================================================================
// Contiguous MGrouped GEMM — mega kernel (Phase 1+2)
// ============================================================================
class SM100FP8AsymGemmMegaContiguousRuntime final: public LaunchRuntime<SM100FP8AsymGemmMegaContiguousRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;
        const std::optional<std::string>& epilogue_type;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm100_fp8_asym_gemm_mega.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_asym_gemm_mega_impl<
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
        {},
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
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        kMegaNumStagesB);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_fp8_asym_gemm_mega_contiguous(const torch::Tensor& a, const torch::Tensor& sfa,
                                                           const torch::Tensor& b, const torch::Tensor& sfb,
                                                           const torch::Tensor& d,
                                                           const torch::Tensor& offsets_t,
                                                           const torch::Tensor& experts_t,
                                                           const int& list_size,
                                                           const int& num_groups, const int& m, const int& n, const int& k,
                                                           const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                           const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    const int num_sms = device_runtime->get_num_sms();

    auto config = get_best_config_asym<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
        m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        num_sms);

    // Phase 1: adjust smem_size for kMegaNumStagesB B slots.
    // The heuristic allocates 1 B slot (is_asym=true). We need kMegaNumStagesB slots.
    // If the extra smem overflows capacity, reduce block_n (keeping block_k intact).
    // Reducing block_k instead would trigger a stale-TMEM SFA bug when kBlockKPerSFLoad > 1
    // (i.e. when BLOCK_K < 512 for FP8), so we must keep BLOCK_K at its current value.
    {
        const int extra_barrier_smem = (kMegaNumStagesB - 1) * 3 * 8;  // 3 extra B barriers per stage
        const int load_block_n = SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n);
        const int extra_b_smem = (kMegaNumStagesB - 1) * load_block_n * config.block_k;  // fp8 = 1 byte

        if (config.smem_config.smem_size + extra_b_smem + extra_barrier_smem <= SM100ArchSpec::smem_capacity) {
            config.smem_config.smem_size += extra_b_smem + extra_barrier_smem;
        } else {
            // Try smaller block_n (MGroupedContiguous never uses multicast, so load_block_n == block_n)
            const int block_n_candidates[] = {224, 192, 128, 64};
            bool found = false;
            for (const int bn : block_n_candidates) {
                if (bn >= config.block_n) continue;
                if (!SM100ArchSpec::is_block_size_legal(KernelType::Kernel1D1D,
                                                        config.major_a, config.major_b,
                                                        config.ab_dtype, config.cd_dtype,
                                                        m, n, k, config.block_m, bn, config.block_k))
                    continue;
                const MulticastConfig mc{1, false};
                auto candidate_smem = get_smem_config<SM100ArchSpec>(
                    config.gemm_type, config.kernel_type,
                    m, n, k,
                    config.block_m, bn, config.block_k,
                    config.major_a, config.major_b,
                    config.ab_dtype, config.cd_dtype,
                    config.num_stages, mc, true);
                const int extra = (kMegaNumStagesB - 1) * bn * config.block_k + extra_barrier_smem;
                if (candidate_smem.smem_size + extra <= SM100ArchSpec::smem_capacity) {
                    config.block_n = bn;
                    config.multicast_config = mc;
                    config.smem_config = candidate_smem;
                    config.smem_config.smem_size += extra;
                    config.thread_config = SM100ArchSpec::get_thread_config(
                        KernelType::Kernel1D1D, config.block_m, bn);
                    found = true;
                    break;
                }
            }
            DG_HOST_ASSERT(found);
        }
    }

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    constexpr int sf_quant_k = 128;
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, sf_quant_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, sf_quant_k, num_groups, 0);

    if (list_size <= 1)
        return;

    const SM100FP8AsymGemmMegaContiguousRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), list_size - 1}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };

    const auto& code = SM100FP8AsymGemmMegaContiguousRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_asym_gemm_mega_contiguous", code);
    SM100FP8AsymGemmMegaContiguousRuntime::launch(runtime, args);
}

// ============================================================================
// Masked MGrouped GEMM — mega kernel (Phase 1+2)
// ============================================================================
class SM100FP8AsymGemmMegaMaskedRuntime final: public LaunchRuntime<SM100FP8AsymGemmMegaMaskedRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;
        const std::optional<std::string>& epilogue_type;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* masked_m;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm100_fp8_asym_gemm_mega.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_asym_gemm_mega_impl<
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
        {},
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
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        kMegaNumStagesB);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.masked_m, nullptr,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_fp8_asym_gemm_mega_masked(const torch::Tensor& a, const torch::Tensor& sfa,
                                                       const torch::Tensor& b, const torch::Tensor& sfb,
                                                       const torch::Tensor& d,
                                                       const torch::Tensor& masked_m_t,
                                                       const int& expected_m,
                                                       const int& num_groups, const int& m, const int& n, const int& k,
                                                       const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                       const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);

    auto config = get_best_config_asym<SM100ArchSpec>(
        GemmType::MGroupedMasked, KernelType::Kernel1D1D,
        expected_m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Phase 1: adjust smem_size for kMegaNumStagesB B slots.
    // Prefer reducing block_n (keeping block_k) to avoid stale-TMEM SFA when kBlockKPerSFLoad > 1.
    {
        const int extra_barrier_smem = (kMegaNumStagesB - 1) * 3 * 8;
        const int load_block_n = SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n);
        const int extra_b_smem = (kMegaNumStagesB - 1) * load_block_n * config.block_k;

        if (config.smem_config.smem_size + extra_b_smem + extra_barrier_smem <= SM100ArchSpec::smem_capacity) {
            config.smem_config.smem_size += extra_b_smem + extra_barrier_smem;
        } else {
            const int block_n_candidates[] = {224, 192, 128, 64};
            bool found = false;
            for (const int bn : block_n_candidates) {
                if (bn >= config.block_n) continue;
                if (!SM100ArchSpec::is_block_size_legal(KernelType::Kernel1D1D,
                                                        config.major_a, config.major_b,
                                                        config.ab_dtype, config.cd_dtype,
                                                        expected_m, n, k, config.block_m, bn, config.block_k))
                    continue;
                const MulticastConfig mc{1, false};
                auto candidate_smem = get_smem_config<SM100ArchSpec>(
                    config.gemm_type, config.kernel_type,
                    expected_m, n, k,
                    config.block_m, bn, config.block_k,
                    config.major_a, config.major_b,
                    config.ab_dtype, config.cd_dtype,
                    config.num_stages, mc, true);
                const int extra = (kMegaNumStagesB - 1) * bn * config.block_k + extra_barrier_smem;
                if (candidate_smem.smem_size + extra <= SM100ArchSpec::smem_capacity) {
                    config.block_n = bn;
                    config.multicast_config = mc;
                    config.smem_config = candidate_smem;
                    config.smem_config.smem_size += extra;
                    config.thread_config = SM100ArchSpec::get_thread_config(
                        KernelType::Kernel1D1D, config.block_m, bn);
                    found = true;
                    break;
                }
            }
            DG_HOST_ASSERT(found);
        }
    }

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);
    constexpr int sf_quant_k = 128;
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, sf_quant_k, num_groups, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, sf_quant_k, num_groups, 0);

    const SM100FP8AsymGemmMegaMaskedRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .masked_m = masked_m_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };

    const auto& code = SM100FP8AsymGemmMegaMaskedRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_asym_gemm_mega_masked", code);
    SM100FP8AsymGemmMegaMaskedRuntime::launch(runtime, args);
}

} // namespace asym_gemm
