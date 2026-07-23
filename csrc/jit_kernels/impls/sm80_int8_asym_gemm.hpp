// csrc/jit_kernels/impls/sm80_int8_asym_gemm.hpp
// Host-side JIT wrappers for the SM80 INT8 asym MoE kernels
// (asym_gemm/include/asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh).
#pragma once

#include <torch/python.h>
#include <cstdint>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm80.hpp"
#include <asym_gemm/impls/sm80_int8_moe_params.h>

namespace asym_gemm {

// ──────────────────────────────────────────────────────────────────────────────
// Contiguous JIT runtime
// ──────────────────────────────────────────────────────────────────────────────
class SM80MoEInt8GemmRuntime final : public LaunchRuntime<SM80MoEInt8GemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM80MoEInt8Params    params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
// sm80 moe int8 v1: 1d1d scales, FP32 partials
#include <asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_int8_asym_moe_gemm_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Deep-pattern JIT runtime (M-outer, HBM-resident B) — Phase 3
// ──────────────────────────────────────────────────────────────────────────────
class SM80MoEInt8DeepGemmRuntime final : public LaunchRuntime<SM80MoEInt8DeepGemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig gemm_config;
        LaunchArgs           launch_args;
        SM80MoEInt8Params    params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
// sm80 moe int8 deep v1: M-outer full-K register accumulation, 2-stage cp.async
#include <asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_int8_deep_moe_gemm_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Masked JIT runtime — padded [G, M_max, ...] layout, constant grid
// ──────────────────────────────────────────────────────────────────────────────
class SM80MoEInt8MaskedGemmRuntime final : public LaunchRuntime<SM80MoEInt8MaskedGemmRuntime> {
public:
    struct Args {
        sm80::SM80GemmConfig     gemm_config;
        LaunchArgs               launch_args;
        SM80MoEInt8MaskedParams  params;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.gemm_config;
        return fmt::format(R"(
// sm80 moe int8 v1: 1d1d scales, FP32 partials
#include <asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh>
using namespace asym_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm80_int8_asym_moe_gemm_masked_impl<{}, {}, {}, {}>);
}};
)",
            c.block_m, c.block_n, c.block_k, c.nwarps);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// ──────────────────────────────────────────────────────────────────────────────
// Shared argument checks
// ──────────────────────────────────────────────────────────────────────────────
static void check_sm80_int8_common(const torch::Tensor& a,
                                   const torch::Tensor& b,
                                   const torch::Tensor& d,
                                   const torch::Tensor& sfa,
                                   const torch::Tensor& sfb,
                                   int64_t N, int64_t K) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kChar and b.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(sfa.is_contiguous() and sfb.is_contiguous());
    DG_HOST_ASSERT(a.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(sfa.is_cuda());
    // W and its scales stream over PCIe from pinned host memory, or live in HBM.
    DG_HOST_ASSERT(b.is_cuda()   or (b.device().is_cpu()   and b.is_pinned()));
    DG_HOST_ASSERT(sfb.is_cuda() or (sfb.device().is_cpu() and sfb.is_pinned()));
    DG_HOST_ASSERT(K >= 128 and K % 128 == 0);   // BLOCK_K = scale granularity
    DG_HOST_ASSERT(N % 32 == 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Free functions called from the API layer
// ──────────────────────────────────────────────────────────────────────────────
// a   : int8  [total_tokens, K] HBM        sfa : fp32 [total_tokens, kb] HBM
// b   : int8  [E, N, K] pinned-host/HBM    sfb : fp32 [E, N, kb] pinned-host/HBM
// d   : fp32  [total_tokens, N] HBM (partials + final)
// offsets : int32 [2*(list_size-1)] (start, end) row pairs per segment
// experts : int32 [list_size] expert ids, -1 terminated (-1 = skip)
// Same segment convention as m_grouped_int8_asym_gemm_sm90_contiguous.
static void sm80_m_grouped_int8_asym_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int32_t list_size,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb)
{
    const int64_t K = a.size(1);
    const int64_t E = b.size(0);
    const int64_t N = b.size(1);
    check_sm80_int8_common(a, b, d, sfa, sfb, N, K);
    DG_HOST_ASSERT(b.size(2) == K and d.size(1) == N);
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt and experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() and experts.is_contiguous());
    DG_HOST_ASSERT(list_size >= 1);
    DG_HOST_ASSERT(offsets.numel() >= 2 * (list_size - 1) and experts.numel() >= list_size);
    const int32_t kb = static_cast<int32_t>(K / 128);
    DG_HOST_ASSERT(sfa.size(0) == a.size(0) and sfa.size(-1) == kb);
    DG_HOST_ASSERT(sfb.size(0) == E and sfb.size(1) == N and sfb.size(2) == kb);

    const int32_t num_segments = list_size - 1;   // last experts entry = -1 terminator
    if (a.size(0) == 0 or num_segments <= 0)
        return;

    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_int8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K));

    const SM80MoEInt8Params params {
        .x_ptr       = a.data_ptr(),
        .w_ptr       = b.data_ptr(),
        .o_ptr       = d.data_ptr(),
        .expert_list = experts.data_ptr<int32_t>(),
        .index_list  = offsets.data_ptr<int32_t>(),
        .list_size   = list_size,
        .expert_size = static_cast<int32_t>(E),
        .N           = N,
        .K           = K,
        .sfa_ptr     = sfa.data_ptr<float>(),
        .sfb_ptr     = sfb.data_ptr<float>(),
        .kb          = kb,
    };

    const SM80MoEInt8GemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), num_segments},
                                  cfg.num_threads(),
                                  sm80::smem_bytes_int8(cfg.block_m, cfg.block_n, cfg.block_k)),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm80_moe_int8_gemm_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEInt8GemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEInt8GemmRuntime::launch(runtime, runtime_args);
}

// Deep-pattern contiguous entry: identical calling convention to
// sm80_m_grouped_int8_asym_gemm_contiguous, but B/sfb must be HBM-resident
// (the M-outer loop re-reads B once per M-tile — ruinous over PCIe, cheap
// from HBM) and D is written exactly once (no FP32 partial round-trips).
static void sm80_m_grouped_int8_deep_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int32_t list_size,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb)
{
    const int64_t K = a.size(1);
    const int64_t E = b.size(0);
    const int64_t N = b.size(1);
    check_sm80_int8_common(a, b, d, sfa, sfb, N, K);
    DG_HOST_ASSERT(b.is_cuda() and sfb.is_cuda());   // HBM-resident B only
    DG_HOST_ASSERT(b.size(2) == K and d.size(1) == N);
    DG_HOST_ASSERT(offsets.scalar_type() == torch::kInt and experts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets.is_cuda() and experts.is_cuda());
    DG_HOST_ASSERT(offsets.is_contiguous() and experts.is_contiguous());
    DG_HOST_ASSERT(list_size >= 1);
    DG_HOST_ASSERT(offsets.numel() >= 2 * (list_size - 1) and experts.numel() >= list_size);
    const int32_t kb = static_cast<int32_t>(K / 128);
    DG_HOST_ASSERT(sfa.size(0) == a.size(0) and sfa.size(-1) == kb);
    DG_HOST_ASSERT(sfb.size(0) == E and sfb.size(1) == N and sfb.size(2) == kb);

    const int32_t num_segments = list_size - 1;
    if (a.size(0) == 0 or num_segments <= 0)
        return;

    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_int8_deep_config(arch_major, arch_minor,
                                                        static_cast<int>(N),
                                                        static_cast<int>(K));

    const SM80MoEInt8Params params {
        .x_ptr       = a.data_ptr(),
        .w_ptr       = b.data_ptr(),
        .o_ptr       = d.data_ptr(),
        .expert_list = experts.data_ptr<int32_t>(),
        .index_list  = offsets.data_ptr<int32_t>(),
        .list_size   = list_size,
        .expert_size = static_cast<int32_t>(E),
        .N           = N,
        .K           = K,
        .sfa_ptr     = sfa.data_ptr<float>(),
        .sfb_ptr     = sfb.data_ptr<float>(),
        .kb          = kb,
    };

    const SM80MoEInt8DeepGemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)), num_segments},
                                  cfg.num_threads(),
                                  sm80::smem_bytes_int8_deep(cfg.block_m, cfg.block_n,
                                                             cfg.block_k)),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm80_moe_int8_deep_gemm_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEInt8DeepGemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEInt8DeepGemmRuntime::launch(runtime, runtime_args);
}

// a   : int8  [G, M_max, K] HBM            sfa : fp32 [G, M_max, kb] HBM
// b   : int8  [G, N, K] pinned-host/HBM    sfb : fp32 [G, N, kb] pinned-host/HBM
// d   : fp32  [G, M_max, N] HBM
static void sm80_m_grouped_int8_asym_gemm_masked(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const torch::Tensor& masked_m,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb)
{
    const int64_t G     = b.size(0);
    const int64_t M_max = a.size(1);
    const int64_t N     = b.size(1);
    const int64_t K     = a.size(2);
    check_sm80_int8_common(a, b, d, sfa, sfb, N, K);
    DG_HOST_ASSERT(a.size(0) == G and b.size(2) == K);
    DG_HOST_ASSERT(d.size(0) == G and d.size(1) == M_max and d.size(2) == N);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt and masked_m.numel() == G);
    DG_HOST_ASSERT(masked_m.is_cuda() and masked_m.is_contiguous());
    const int32_t kb = static_cast<int32_t>(K / 128);
    DG_HOST_ASSERT(sfa.size(0) == G and sfa.size(1) == M_max and sfa.size(2) == kb);
    DG_HOST_ASSERT(sfb.size(0) == G and sfb.size(1) == N and sfb.size(2) == kb);

    if (G == 0 or M_max == 0)
        return;

    const auto& [arch_major, arch_minor] = device_runtime->get_arch_pair();
    const auto cfg = sm80::select_sm80_int8_config(arch_major, arch_minor,
                                                   static_cast<int>(N),
                                                   static_cast<int>(K));

    const SM80MoEInt8MaskedParams params {
        .x_ptr      = a.data_ptr(),
        .w_ptr      = b.data_ptr(),
        .o_ptr      = d.data_ptr(),
        .masked_m   = masked_m.data_ptr<int32_t>(),
        .num_groups = static_cast<int32_t>(G),
        .M_max      = M_max,
        .N          = N,
        .K          = K,
        .sfa_ptr    = sfa.data_ptr<float>(),
        .sfb_ptr    = sfb.data_ptr<float>(),
        .kb         = kb,
    };

    const SM80MoEInt8MaskedGemmRuntime::Args runtime_args {
        .gemm_config = cfg,
        .launch_args = LaunchArgs({cfg.grid_x(static_cast<int>(N)),
                                   static_cast<int32_t>(G)},
                                  cfg.num_threads(),
                                  sm80::smem_bytes_int8(cfg.block_m, cfg.block_n, cfg.block_k)),
        .params      = params,
    };

    const std::string kernel_name = fmt::format("sm80_moe_int8_gemm_masked_bm{}_bn{}_bk{}",
        cfg.block_m, cfg.block_n, cfg.block_k);

    const auto& code    = SM80MoEInt8MaskedGemmRuntime::generate(runtime_args);
    const auto& runtime = compiler->build(kernel_name, code);
    SM80MoEInt8MaskedGemmRuntime::launch(runtime, runtime_args);
}

}  // namespace asym_gemm
