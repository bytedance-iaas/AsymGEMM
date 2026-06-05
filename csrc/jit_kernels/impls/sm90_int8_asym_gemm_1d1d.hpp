// SM90 (Hopper / H20) INT8 asymmetric grouped GEMM — JIT host launcher.
//
// Drives `sm90_int8_asym_gemm_1d1d_impl`. INT8 storage, int32 (S32) WGMMA
// accumulation, FP32 output. Unlike the SM100 launcher this builds the GemmConfig
// by hand (fixed manual config, multicast=1) and computes the dynamic shared
// memory size with the EXACT formula the kernel uses, so the two never disagree.
//
// Scale-factor tensors are expected pre-transposed into the layout the TMA SF
// descriptors read (done in the Python dispatch layer):
//   * sfa: [num_groups, ceil(k/128), m_max]  (group + k-block major, m contiguous)
//   * sfb: [ceil(k/128), num_groups * n]      (k-block major, (group,n) contiguous)
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

// Exact dynamic-smem size for the SM90 int8 asym kernel (FP32 output, no CD swizzle).
static int sm90_int8_asym_smem_size(const int& block_m, const int& block_n,
                                    const int& block_k, const int& num_stages) {
    constexpr int wgmma_m = 64;
    const int wave_block_m = (block_m <= wgmma_m) ? block_m : std::min(block_m, 2 * wgmma_m);
    const int store_block_m = wave_block_m;
    const int cd_row_bytes = block_n * static_cast<int>(sizeof(float));   // FP32, swizzle=0
    constexpr int kNumTMAStoreStages = 2;

    const int smem_cd = store_block_m * cd_row_bytes * kNumTMAStoreStages;
    const int smem_a = num_stages * block_m * block_k;                    // int8: 1 byte
    const int smem_b = block_n * block_k;                                 // single slot
    const int smem_sfa = num_stages * block_m * static_cast<int>(sizeof(float));
    const int smem_sfb = block_n * static_cast<int>(sizeof(float));
    const int smem_barrier = (2 * num_stages + 2) * 8;                    // ClusterTransactionBarrier == 8B
    return smem_cd + smem_a + smem_b + smem_sfa + smem_sfb + smem_barrier;
}

// Build a fully-controlled GemmConfig: fixed block sizes, 2 stages, no multicast,
// SM90 thread split, hand-matched smem. (cd_dtype = float; ab "dtype" is 1-byte.)
static GemmConfig sm90_int8_asym_make_config(const GemmType& gemm_type,
                                             const int& num_groups, const int& num_sms,
                                             const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                             const int& block_m, const int& block_n, const int& block_k) {
    constexpr int num_stages = 2;
    const int smem_size = sm90_int8_asym_smem_size(block_m, block_n, block_k, num_stages);
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    SharedMemoryConfig smem_config{
        .smem_size = smem_size,
        .swizzle_a_mode = block_k,   // K-major, 1-byte elems -> 128B swizzle == block_k
        .swizzle_b_mode = block_k,
        .swizzle_cd_mode = 0,        // FP32 output: no CD swizzle
    };

    return GemmConfig{
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .ab_dtype = torch::kFloat8_e4m3fn,   // proxy: 1-byte, only element size matters here
        .cd_dtype = torch::kFloat,
        .major_a = major_a,
        .major_b = major_b,
        .with_accumulation = false,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .num_stages = num_stages,
        .num_last_stages = 0,
        .num_sms = num_sms,
        .tc_util = device_runtime->get_tc_util(),
        .multicast_config = MulticastConfig{1, false},
        .smem_config = smem_config,
        .thread_config = SM90ArchSpec::get_thread_config(KernelType::Kernel1D1D, block_m, block_n),
    };
}

class SM90Int8AsymGemm1D1DRuntime final: public LaunchRuntime<SM90Int8AsymGemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* offsets;   // masked_m (masked) or offsets (contiguous)
        void* experts;   // nullptr (masked) or experts (contiguous)
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_int8_asym_gemm_1d1d_impl<
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
        // SM90 asym kernel: kNumNonEpilogueThreads = math, kNumEpilogueThreads = tma
        args.gemm_config.thread_config.num_math_threads, args.gemm_config.thread_config.num_tma_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype),
        /*kTensorCoreUtilControl=*/0);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

// ============================================================================
// Masked INT8 grouped GEMM:  a[G, M_max, K] @ b[G, N, K].mT -> d[G, M_max, N]
// ============================================================================
static void sm90_m_grouped_int8_asym_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                       const torch::Tensor& b, const torch::Tensor& sfb,
                                                       const torch::Tensor& d,
                                                       const torch::Tensor& masked_m_t,
                                                       const int& expected_m,
                                                       const int& num_groups, const int& m, const int& n, const int& k,
                                                       const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                       const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // BLOCK_M = 64 => a single math warp-group (one WGMMA::M tile), which keeps the
    // register->smem epilogue row mapping correct. BLOCK_M = 128 would need a
    // `math_wg_idx * WGMMA::M` row offset the SM90 asym epilogue does not apply.
    const int block_m = 64, block_n = 128, block_k = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(n % block_n == 0);
    DG_HOST_ASSERT(m % block_m == 0);   // M_max must tile evenly (SF TMA in-bounds)

    const auto& config = sm90_int8_asym_make_config(GemmType::MGroupedMasked, num_groups,
                                                    device_runtime->get_num_sms(),
                                                    major_a, major_b, block_m, block_n, block_k);

    constexpr int wgmma_m = 64;
    const int store_block_m = (block_m <= wgmma_m) ? block_m : std::min(block_m, 2 * wgmma_m);
    constexpr int sf_quant_k = 128;

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k, block_m, block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))),
                                               num_groups, config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k, block_n, block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
                                               num_groups, config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n, store_block_m, block_n,
                                                 static_cast<int>(d.stride(-2)), num_groups, 0);
    // SFA stacks groups along the K-block (outer) dim; SFB stacks (group,n) along the inner dim.
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k, block_m, sf_quant_k, num_groups, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n * num_groups, k, block_n, sf_quant_k, 1, 0);

    const SM90Int8AsymGemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, block_n), num_groups},
                                  config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = masked_m_t.data_ptr<int>(),
        .experts = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90Int8AsymGemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_int8_asym_gemm_masked_1d1d", code);
    SM90Int8AsymGemm1D1DRuntime::launch(runtime, args);
}

// ============================================================================
// Contiguous INT8 grouped GEMM:  a[M, K] @ b[G, N, K].mT -> d[M, N]
// ============================================================================
static void sm90_m_grouped_int8_asym_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                          const torch::Tensor& b, const torch::Tensor& sfb,
                                                          const torch::Tensor& d,
                                                          const torch::Tensor& offsets_t,
                                                          const torch::Tensor& experts_t,
                                                          const int& grid_y,
                                                          const int& num_groups, const int& m, const int& n, const int& k,
                                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                          const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // BLOCK_M = 64 (single math warp-group); see masked variant for why.
    const int block_m = 64, block_n = 128, block_k = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(n % block_n == 0);

    const auto& config = sm90_int8_asym_make_config(GemmType::MGroupedContiguous, num_groups,
                                                    device_runtime->get_num_sms(),
                                                    major_a, major_b, block_m, block_n, block_k);

    constexpr int wgmma_m = 64;
    const int store_block_m = (block_m <= wgmma_m) ? block_m : std::min(block_m, 2 * wgmma_m);
    constexpr int sf_quant_k = 128;

    // A and D are a single flat [M, *] tensor (group selected per token row): num_groups = 1.
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k, block_m, block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))),
                                               1, config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k, block_n, block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))),
                                               num_groups, config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n, store_block_m, block_n,
                                                 static_cast<int>(d.stride(-2)), 1, 0);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k, block_m, sf_quant_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n * num_groups, k, block_n, sf_quant_k, 1, 0);

    if (grid_y <= 0)
        return;

    const SM90Int8AsymGemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, block_n), grid_y},
                                  config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90Int8AsymGemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_int8_asym_gemm_contiguous_1d1d", code);
    SM90Int8AsymGemm1D1DRuntime::launch(runtime, args);
}

} // namespace asym_gemm
