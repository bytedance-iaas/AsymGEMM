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

class SM100FP4AsymGemm1D1DRuntime final: public LaunchRuntime<SM100FP4AsymGemm1D1DRuntime> {
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
#include <asym_gemm/impls/sm100_fp4_asym_gemm_1d1d.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_asym_gemm_1d1d_impl<
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
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& offsets_t,
                                                     const torch::Tensor& experts_t,
                                                     const int& list_size,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    fprintf(stderr, "[FP4_LAUNCH] Enter sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d: m=%d, n=%d, k=%d, num_groups=%d, list_size=%d\n",
            m, n, k, num_groups, list_size);
    fprintf(stderr, "[FP4_LAUNCH] a shape=(%d,%d), sfa shape=(%d,%d)\n",
            (int)a.size(0), (int)a.size(1), (int)sfa.size(0), (int)sfa.size(1));
    fprintf(stderr, "[FP4_LAUNCH] b shape=(%d,%d,%d), sfb shape=(%d,%d,%d)\n",
            (int)b.size(0), (int)b.size(1), (int)b.size(2),
            (int)sfb.size(0), (int)sfb.size(1), (int)sfb.size(2));
    fprintf(stderr, "[FP4_LAUNCH] b device=%s pinned=%d, sfb device=%s pinned=%d\n",
            b.is_cuda() ? "cuda" : "cpu", (int)b.is_pinned(),
            sfb.is_cuda() ? "cuda" : "cpu", (int)sfb.is_pinned());
    if (!b.is_cuda() || !sfb.is_cuda()) {
        fprintf(stderr, "[FP4_LAUNCH][WARN] Using CPU/pinned tensors for FP4 TMA path on SM100; "
                        "if TMA memory-domain mapping is unsupported, this may trigger illegal address.\n");
    }
    fprintf(stderr, "[FP4_LAUNCH] d shape=(%d,%d)\n", (int)d.size(0), (int)d.size(1));

    const auto& aligned_k = align(k, 64);

    const int block_m = 128;
    const int block_n = 128;
    const int block_k = 512;

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
            m, n, k, 1, major_a, major_b,
            torch::kFloat8_e4m3fn, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
            m, n, k, 1, major_a, major_b,
            torch::kFloat8_e4m3fn, d.scalar_type(), false,
            device_runtime->get_num_sms());

    fprintf(stderr, "[FP4_LAUNCH] Config: block=(%d,%d,%d), stages=%d, threads=(%d+%d), multicast=%d, smem=%d, num_sms=%d\n",
            config.block_m, config.block_n, config.block_k,
            config.num_stages,
            config.thread_config.num_non_epilogue_threads, config.thread_config.num_epilogue_threads,
            config.multicast_config.num_multicast,
            config.smem_config.smem_size, config.num_sms);

    // Create tensor descriptors
    // FP4 packed: tensor has k/2 uint8 bytes per row (2 FP4 values per byte).
    // TMA descriptor shape_k must match the actual byte layout (k/2), not the logical element count (k).
    const int k_packed = k / 2;
    fprintf(stderr, "[FP4_LAUNCH] Creating TMA descriptors (k_packed=%d)...\n", k_packed);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k_packed,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k / 2,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    fprintf(stderr, "[FP4_LAUNCH] TMA A done\n");
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto& tensor_map_b = make_tma_3d_desc(
        b,
        k_packed, n, num_groups,
        config.block_k / 2,
        SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
        1,
        static_cast<int>(b.stride(-2)),
        static_cast<int>(b.stride(-3)),
        config.smem_config.swizzle_b_mode);
    fprintf(stderr, "[FP4_LAUNCH] TMA B done\n");
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);
    fprintf(stderr, "[FP4_LAUNCH] TMA CD done\n");
    // FP4 SF: 16-element granularity, 4 UE8M0 per uint32 pack.
    // Box K must be sf_packs_per_block_k so one TMA loads all SF packs for a block.
    constexpr int sf_quant_k = 16;
    constexpr int sf_elem_per_pack = 4;
    const int sf_packs_per_block_k = config.block_k / (sf_quant_k * sf_elem_per_pack);
    const int sf_factor = (sfa.scalar_type() == torch::kFloat) ? 1 : sf_elem_per_pack;
    fprintf(stderr, "[FP4_LAUNCH] SF config: sf_quant_k=%d sf_elem_per_pack=%d sf_packs_per_block_k=%d sf_factor=%d\n",
            sf_quant_k, sf_elem_per_pack, sf_packs_per_block_k, sf_factor);
    // Use actual SF tensor MN dimensions (may differ from m/n if gran_mn > 1 and not broadcast)
    const int sfa_mn = static_cast<int>(sfa.size(sfa.dim() - 2));
    const int sfb_mn = static_cast<int>(sfb.size(sfb.dim() - 2));
    const int sfa_aligned_mn = get_tma_aligned_size(sfa_mn, static_cast<int>(sfa.element_size()));
    const auto& tensor_map_sfa = make_tma_2d_desc(sfa,
                                                  sfa_aligned_mn, ceil_div(k, sf_quant_k * sf_factor),
                                                  config.block_m, sf_packs_per_block_k,
                                                  sfa_aligned_mn, 0, 0, false);
    fprintf(stderr, "[FP4_LAUNCH] TMA SFA done (sfa_mn=%d, aligned=%d)\n", sfa_mn, sfa_aligned_mn);
    const int sfb_aligned_mn = get_tma_aligned_size(sfb_mn, static_cast<int>(sfb.element_size()));
    const auto& tensor_map_sfb = make_tma_2d_desc(sfb,
                                                  sfb_aligned_mn, ceil_div(k, sf_quant_k * sf_factor) * num_groups,
                                                  config.block_n, sf_packs_per_block_k,
                                                  sfb_aligned_mn, 0, 0, false);
    fprintf(stderr, "[FP4_LAUNCH] TMA SFB done (sfb_mn=%d, aligned=%d)\n", sfb_mn, sfb_aligned_mn);

    if (list_size <= 1) {
        fprintf(stderr, "[FP4_LAUNCH] list_size <= 1, early return\n");
        return;
    }

    // Launch kernel
    const auto grid_x = ceil_div(n, config.block_n);
    const auto grid_y = list_size - 1;
    fprintf(stderr, "[FP4_LAUNCH] Grid=(%d,%d), threads=%d, smem=%d, cluster=%d\n",
            grid_x, grid_y, config.thread_config.num_threads,
            config.smem_config.smem_size, config.multicast_config.num_multicast);

    const SM100FP4AsymGemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .epilogue_type = std::nullopt,
        .gemm_config = config,
        .launch_args = LaunchArgs({grid_x, grid_y}, config.thread_config.num_threads,
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

    fprintf(stderr, "[FP4_LAUNCH] JIT compile...\n");
    const auto& code = SM100FP4AsymGemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d", code);
    fprintf(stderr, "[FP4_LAUNCH] JIT done, launching kernel...\n");
    SM100FP4AsymGemm1D1DRuntime::launch(runtime, args);
    fprintf(stderr, "[FP4_LAUNCH] Kernel launched (async)\n");
}

// ============================================================================
// Masked GEMM Implementation for FP4
// ============================================================================
class SM100FP4AsymGemmMaskedRuntime final: public LaunchRuntime<SM100FP4AsymGemmMaskedRuntime> {
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
#include <asym_gemm/impls/sm100_fp4_asym_gemm_1d1d.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_asym_gemm_1d1d_impl<
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
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_m_grouped_fp4_asym_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                 const torch::Tensor& b, const torch::Tensor& sfb,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const int& list_size,
                                                 const int& expected_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 64);

    const int block_m = 128;
    const int block_n = 128;
    const int block_k = 512;

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedMasked, KernelType::Kernel1D1D,
            expected_m, n, k, num_groups, major_a, major_b,
            torch::kFloat8_e4m3fn, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
            GemmType::MGroupedMasked, KernelType::Kernel1D1D,
            expected_m, n, k, num_groups, major_a, major_b,
            torch::kFloat8_e4m3fn, d.scalar_type(), false,
            device_runtime->get_num_sms());

    // Create tensor descriptors with num_groups for grouped layout
    // FP4 packed: tensor has k/2 uint8 bytes per row (2 FP4 values per byte).
    const int k_packed = k / 2;
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k_packed,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k / 2,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto& tensor_map_b = make_tma_3d_desc(
        b,
        k_packed, n, num_groups,
        config.block_k / 2,
        SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
        1,
        static_cast<int>(b.stride(-2)),
        static_cast<int>(b.stride(-3)),
        config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), num_groups,
                                                 config.smem_config.swizzle_cd_mode);
    // FP4 SF: 16-element granularity, 4 UE8M0 per uint32 pack.
    // Box K must be sf_packs_per_block_k so one TMA loads all SF packs for a block.
    constexpr int sf_quant_k = 16;
    constexpr int sf_elem_per_pack = 4;
    const int sf_packs_per_block_k = config.block_k / (sf_quant_k * sf_elem_per_pack);
    const int sf_factor = (sfa.scalar_type() == torch::kFloat) ? 1 : sf_elem_per_pack;
    const int sfa_mn = static_cast<int>(sfa.size(sfa.dim() - 2));
    const int sfb_mn = static_cast<int>(sfb.size(sfb.dim() - 2));
    const int sfa_aligned_mn = get_tma_aligned_size(sfa_mn, static_cast<int>(sfa.element_size()));
    const auto& tensor_map_sfa = make_tma_2d_desc(sfa,
                                                  sfa_aligned_mn, ceil_div(k, sf_quant_k * sf_factor) * num_groups,
                                                  config.block_m, sf_packs_per_block_k,
                                                  sfa_aligned_mn, 0, 0, false);
    const int sfb_aligned_mn = get_tma_aligned_size(sfb_mn, static_cast<int>(sfb.element_size()));
    const auto& tensor_map_sfb = make_tma_2d_desc(sfb,
                                                  sfb_aligned_mn, ceil_div(k, sf_quant_k * sf_factor) * num_groups,
                                                  config.block_n, sf_packs_per_block_k,
                                                  sfb_aligned_mn, 0, 0, false);

    // Launch kernel with masked configuration
    const SM100FP4AsymGemmMaskedRuntime::Args& args = {
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
    const auto& code = SM100FP4AsymGemmMaskedRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp4_asym_gemm_masked_1d1d", code);
    SM100FP4AsymGemmMaskedRuntime::launch(runtime, args);
}

} // namespace asym_gemm
