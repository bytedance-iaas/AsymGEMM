// SM90 INT8 grouped GEMM for HBM-resident expert weights — JIT host launcher
// for the deepGEMM-pattern kernel (hybridGEMM.md Phase A).
//
// Drives `sm90_int8_gemm_impl`: persistent 1D grid (num_sms CTAs), M-outer
// deep pipeline, INT8 storage, S32 WGMMA full-K accumulation, FP32 output.
// Consumes the SAME contiguous grouped layout (offsets pairs + experts ids)
// as the asym 1d1d launcher, so the runtime's `_build_layout` feeds both.
//
// Scale factors arrive in the same pre-transposed K-major layout as the asym
// kernel's SF TMA descriptors (built in the Python dispatch layer); the kernel
// reads them per K-block via plain global loads (no SF TMA machinery):
//   * sfa: [ceil(k/128), m]              float32
//   * sfb: [ceil(k/128), num_groups * n] float32
#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "../../utils/layout.hpp"
#include "runtime_utils.hpp"

namespace asym_gemm {

// Exact dynamic-smem size for the deep INT8 kernel (FP32 output, no CD swizzle).
static int sm90_int8_deep_smem_size(const int& block_m, const int& block_n,
                                    const int& block_k, const int& num_stages) {
    constexpr int kNumTMAStoreStages = 2;
    const int smem_cd = block_m * block_n * static_cast<int>(sizeof(float)) * kNumTMAStoreStages;
    const int smem_a = num_stages * block_m * block_k;   // int8
    const int smem_b = num_stages * block_n * block_k;   // int8, staged (unlike asym)
    const int smem_barrier = 2 * num_stages * 8;         // ClusterTransactionBarrier == 8B
    return smem_cd + smem_a + smem_b + smem_barrier;
}

// Deepest pipeline that fits shared memory.
static int sm90_int8_deep_num_stages(const int& block_m, const int& block_n, const int& block_k) {
    int num_stages = 2;
    while (sm90_int8_deep_smem_size(block_m, block_n, block_k, num_stages + 1) <= SM90ArchSpec::smem_capacity)
        ++ num_stages;
    return num_stages;
}

class SM90Int8DeepGemmRuntime final: public LaunchRuntime<SM90Int8DeepGemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups, num_segments;
        const std::string& compiled_dims;

        int block_m, block_n, block_k, num_stages, num_sms;
        int num_tma_threads, num_math_threads;
        int smem_size;
        LaunchArgs launch_args;

        void* offsets;
        void* experts;
        void* sfa;
        void* sfb;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm90_int8_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_int8_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.block_m, args.block_n, args.block_k,
        args.num_groups,
        /*kSwizzleAMode=*/args.block_k, /*kSwizzleBMode=*/args.block_k,
        args.num_stages,
        args.num_tma_threads, args.num_math_threads,
        args.num_sms);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, static_cast<uint32_t>(args.num_segments),
            args.sfa, args.sfb,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_cd));
    }
};

// ============================================================================
// Contiguous INT8 grouped GEMM over HBM weights:
//   a[M, K] @ b[G, N, K].mT -> d[M, N]   (deepGEMM iteration pattern)
// ============================================================================
static void sm90_m_grouped_int8_deep_gemm_contiguous(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& offsets_t,
                                                     const torch::Tensor& experts_t,
                                                     const int& num_segments,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const std::string& compiled_dims) {
    // v1 fixed config: single math warp-group (BLOCK_M = 64), K-major, no multicast.
    const int block_m = 64, block_n = 128, block_k = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(n % block_n == 0);

    const int num_stages = sm90_int8_deep_num_stages(block_m, block_n, block_k);
    const int smem_size = sm90_int8_deep_smem_size(block_m, block_n, block_k, num_stages);
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    const int num_sms = device_runtime->get_num_sms();
    const auto& thread_config = SM90ArchSpec::get_thread_config(KernelType::Kernel1D1D, block_m, block_n);

    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k, block_m, block_k,
                                               static_cast<int>(a.stride(0)), 1, block_k);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k, block_n, block_k,
                                               static_cast<int>(b.stride(1)), num_groups, block_k);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n, block_m, block_n,
                                                 static_cast<int>(d.stride(-2)), 1, 0);

    if (num_segments <= 0)
        return;

    const SM90Int8DeepGemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups = num_groups,
        .num_segments = num_segments,
        .compiled_dims = compiled_dims,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_tma_threads = thread_config.num_tma_threads,
        .num_math_threads = thread_config.num_math_threads,
        .smem_size = smem_size,
        .launch_args = LaunchArgs(num_sms,
                                  thread_config.num_threads,
                                  smem_size,
                                  1),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .sfa = sfa.data_ptr<float>(),
        .sfb = sfb.data_ptr<float>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90Int8DeepGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_int8_deep_gemm_contiguous", code);
    SM90Int8DeepGemmRuntime::launch(runtime, args);
}

// ============================================================================
// Validated entry point (dispatched from m_grouped_int8_gemm_nt_contiguous in
// csrc/apis/gemm.hpp). Same contract as the asym contiguous entry, except the
// weights are expected DEVICE-resident (HBM) — that is this kernel's purpose.
// Pinned-host B is still accepted (TMA reads UVA either way), but the asym
// kernel's K-outer pattern is the right choice for that case.
// ============================================================================
static void m_grouped_int8_deep_gemm_sm90_contiguous(
        const torch::Tensor& a,         // [M, K]    int8
        const torch::Tensor& b,         // [G, N, K] int8
        const torch::Tensor& d,         // [M, N]    float32
        const torch::Tensor& offsets,   // [>=2*G]   int32 (start,end) pairs
        const torch::Tensor& experts,   // [>=G+1]   int32 (with -1 terminator)
        const int& list_size,
        const torch::Tensor& sfa,       // [ceil(K/128), M]   float32
        const torch::Tensor& sfb) {     // [ceil(K/128), G*N] float32
    DG_HOST_ASSERT(a.dim() == 2 and b.dim() == 3 and d.dim() == 2);
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t num_groups = b.size(0);
    const int64_t n = b.size(1);
    DG_HOST_ASSERT(b.size(2) == k);
    DG_HOST_ASSERT(d.size(0) == m and d.size(1) == n);

    DG_HOST_ASSERT(a.scalar_type() == torch::kChar and b.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(a.is_cuda() and d.is_cuda() and sfa.is_cuda() and sfb.is_cuda());
    DG_HOST_ASSERT(b.is_cuda() or b.is_pinned());
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(sfa.is_contiguous() and sfb.is_contiguous());
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt and experts.scalar_type() == torch::kInt);

    if (m == 0 or n == 0 or k == 0 or list_size <= 1) return;

    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    sm90_m_grouped_int8_deep_gemm_contiguous(a, sfa, b, sfb, d, offsets, experts,
                                             /*num_segments=*/list_size - 1,
                                             static_cast<int>(num_groups),
                                             static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                             "nk");
}

} // namespace asym_gemm
