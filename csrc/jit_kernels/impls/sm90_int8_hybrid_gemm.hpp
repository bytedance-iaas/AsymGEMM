// SM90 INT8 hybrid grouped GEMM — JIT host launcher (hybridGEMM.md Phase B).
//
// Drives `sm90_int8_hybrid_gemm_impl`: ONE persistent launch over num_sms
// CTAs; ranks [0, s_host) run the asym K-outer pipeline over host-resident
// (pinned) expert weights, ranks [s_host, num_sms) run the deep M-outer
// pipeline over HBM-resident weights. `s_host` is a runtime kernel argument
// (no re-JIT when the balance point moves).
//
// Both sides consume asym contiguous grouped layouts (offsets pairs +
// experts ids), one list per side, over the SAME quantized activations and
// the SAME fp32 output. Host and HBM segments must be disjoint row ranges.
//
// Scale factors arrive pre-transposed K-major (as for both parents):
//   * sfa:      [ceil(k/128), m]        fp32 — shared: TMA on the asym side,
//                                       plain global loads on the deep side
//   * sfb_host: [ceil(k/128), Gh * n]   fp32 — asym-side TMA
//   * sfb_hbm:  [ceil(k/128), Gd * n]   fp32 — deep-side global loads
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

// Union dynamic-smem size: shared 2-stage fp32 CD ring + max of the two
// sides' staging plans (each side overlays its own region after the ring).
static int sm90_int8_hybrid_smem_size(const int& block_m, const int& block_n, const int& block_k,
                                      const int& stages_host, const int& stages_hbm) {
    constexpr int kNumTMAStoreStages = 2;
    const int smem_cd = block_m * block_n * static_cast<int>(sizeof(float)) * kNumTMAStoreStages;
    // asym side: A staged, B single slot, SFA staged, SFB single slot, 2s+2 barriers.
    const int host_region = stages_host * block_m * block_k + block_n * block_k
                          + stages_host * block_m * static_cast<int>(sizeof(float))
                          + block_n * static_cast<int>(sizeof(float))
                          + (2 * stages_host + 2) * 8;
    // deep side: A and B both staged, 2s barriers, plus the steal-mode
    // scheduler mailbox (2 slots * 4 words + 4 barriers, always laid out).
    const int hbm_region = stages_hbm * (block_m + block_n) * block_k
                         + 2 * stages_hbm * 8
                         + 4 * 8 + 2 * 4 * 4;
    return smem_cd + std::max(host_region, hbm_region);
}

// Deepest hbm-side pipeline that fits the union budget (the asym side's
// fixed 2-stage plan is far smaller, so it never constrains).
static int sm90_int8_hybrid_num_hbm_stages(const int& block_m, const int& block_n, const int& block_k,
                                           const int& stages_host) {
    int num_stages = 2;
    while (sm90_int8_hybrid_smem_size(block_m, block_n, block_k, stages_host, num_stages + 1)
           <= SM90ArchSpec::smem_capacity)
        ++ num_stages;
    return num_stages;
}

class SM90Int8HybridGemmRuntime final: public LaunchRuntime<SM90Int8HybridGemmRuntime> {
public:
    struct Args {
        int m, n, k, num_groups_hbm, num_segments_host, num_segments_hbm, s_host;
        int enable_steal;
        void* steal_counter;
        const std::string& compiled_dims;

        int block_m, block_n, block_k, num_stages_host, num_stages_hbm, num_sms;
        int num_tma_threads, num_math_threads;
        int smem_size;
        LaunchArgs launch_args;

        void* offsets_host;
        void* experts_host;
        void* offsets_hbm;
        void* experts_hbm;
        void* sfa;
        void* sfb_hbm;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b_host;
        CUtensorMap tensor_map_sfa_host;
        CUtensorMap tensor_map_sfb_host;
        CUtensorMap tensor_map_b_hbm;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm90_int8_hybrid_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_int8_hybrid_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.block_m, args.block_n, args.block_k,
        args.num_groups_hbm,
        /*kSwizzleAMode=*/args.block_k, /*kSwizzleBMode=*/args.block_k,
        args.num_stages_host, args.num_stages_hbm,
        args.num_tma_threads, args.num_math_threads,
        args.num_sms);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets_host, args.experts_host, static_cast<uint32_t>(args.num_segments_host),
            args.offsets_hbm, args.experts_hbm, static_cast<uint32_t>(args.num_segments_hbm),
            static_cast<uint32_t>(args.s_host),
            static_cast<uint32_t>(args.enable_steal), args.steal_counter,
            args.sfa, args.sfb_hbm,
            args.m, args.n, args.k,
            args.tensor_map_a,
            args.tensor_map_b_host, args.tensor_map_sfa_host, args.tensor_map_sfb_host,
            args.tensor_map_b_hbm,
            args.tensor_map_cd));
    }
};

// ============================================================================
// Hybrid contiguous INT8 grouped GEMM:
//   a[M, K] @ b_host[Gh, N, K].mT -> d[M, N]  on CTA ranks [0, s_host)
//   a[M, K] @ b_hbm [Gd, N, K].mT -> d[M, N]  on CTA ranks [s_host, num_sms)
// ============================================================================
static void sm90_m_grouped_int8_hybrid_gemm_contiguous(
        const torch::Tensor& a, const torch::Tensor& sfa,
        const torch::Tensor& b_host, const torch::Tensor& sfb_host,
        const torch::Tensor& b_hbm, const torch::Tensor& sfb_hbm,
        const torch::Tensor& d,
        const torch::Tensor& offsets_host_t, const torch::Tensor& experts_host_t, const int& num_segments_host,
        const torch::Tensor& offsets_hbm_t, const torch::Tensor& experts_hbm_t, const int& num_segments_hbm,
        const int& s_host_in, const bool& enable_steal,
        const int& num_groups_host, const int& num_groups_hbm,
        const int& m, const int& n, const int& k,
        const std::string& compiled_dims) {
    // v1 fixed config, shared by both sides: single math warp-group, K-major.
    const int block_m = 64, block_n = 128, block_k = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(n % block_n == 0);

    const int num_stages_host = 2;   // the asym parent's fixed pipeline depth
    const int num_stages_hbm = sm90_int8_hybrid_num_hbm_stages(block_m, block_n, block_k, num_stages_host);
    const int smem_size = sm90_int8_hybrid_smem_size(block_m, block_n, block_k, num_stages_host, num_stages_hbm);
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    const int num_sms = device_runtime->get_num_sms();
    const auto& thread_config = SM90ArchSpec::get_thread_config(KernelType::Kernel1D1D, block_m, block_n);

    if (num_segments_host <= 0 and num_segments_hbm <= 0)
        return;

    // Clamp the split to the work that exists: an empty side gets 0 CTAs;
    // when both sides have work each needs at least one CTA.
    int s_host = s_host_in;
    if (num_segments_host <= 0)
        s_host = 0;
    else if (num_segments_hbm <= 0)
        s_host = num_sms;
    else
        s_host = std::max(1, std::min(s_host, num_sms - 1));

    // Steal-mode ticket counter: one zeroed uint32 per launch (stream-ordered
    // zero-fill, so back-to-back launches never see a stale count). Stealing
    // only matters when both sides have work.
    const bool steal = enable_steal and num_segments_host > 0 and num_segments_hbm > 0;
    torch::Tensor steal_counter_t;
    void* steal_counter = nullptr;
    if (steal) {
        steal_counter_t = torch::zeros({1}, torch::dtype(torch::kInt).device(a.device()));
        steal_counter = steal_counter_t.data_ptr<int>();
    }

    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k, block_m, block_k,
                                               static_cast<int>(a.stride(0)), 1, block_k);
    const auto& tensor_map_b_host = make_tma_b_desc(cute::UMMA::Major::K, b_host, n, k, block_n, block_k,
                                                    static_cast<int>(b_host.stride(1)), num_groups_host, block_k);
    const auto& tensor_map_b_hbm = make_tma_b_desc(cute::UMMA::Major::K, b_hbm, n, k, block_n, block_k,
                                                   static_cast<int>(b_hbm.stride(1)), num_groups_hbm, block_k);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n, block_m, block_n,
                                                 static_cast<int>(d.stride(-2)), 1, 0);
    constexpr int sf_quant_k = 128;
    const auto& tensor_map_sfa_host = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k, block_m, sf_quant_k, 1, 0);
    const auto& tensor_map_sfb_host = make_tma_sf_desc(cute::UMMA::Major::MN, sfb_host, n * num_groups_host, k,
                                                       block_n, sf_quant_k, 1, 0);

    const SM90Int8HybridGemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .num_groups_hbm = num_groups_hbm,
        .num_segments_host = num_segments_host,
        .num_segments_hbm = num_segments_hbm,
        .s_host = s_host,
        .enable_steal = steal ? 1 : 0,
        .steal_counter = steal_counter,
        .compiled_dims = compiled_dims,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_stages_host = num_stages_host,
        .num_stages_hbm = num_stages_hbm,
        .num_sms = num_sms,
        .num_tma_threads = thread_config.num_tma_threads,
        .num_math_threads = thread_config.num_math_threads,
        .smem_size = smem_size,
        .launch_args = LaunchArgs(num_sms,
                                  thread_config.num_threads,
                                  smem_size,
                                  1),
        .offsets_host = offsets_host_t.data_ptr<int>(),
        .experts_host = experts_host_t.data_ptr<int>(),
        .offsets_hbm = offsets_hbm_t.data_ptr<int>(),
        .experts_hbm = experts_hbm_t.data_ptr<int>(),
        .sfa = sfa.data_ptr<float>(),
        .sfb_hbm = sfb_hbm.data_ptr<float>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b_host = tensor_map_b_host,
        .tensor_map_sfa_host = tensor_map_sfa_host,
        .tensor_map_sfb_host = tensor_map_sfb_host,
        .tensor_map_b_hbm = tensor_map_b_hbm,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto& code = SM90Int8HybridGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_int8_hybrid_gemm_contiguous", code);
    SM90Int8HybridGemmRuntime::launch(runtime, args);
}

// ============================================================================
// Validated entry point (dispatched from m_grouped_int8_hybrid_gemm_nt_
// contiguous in csrc/apis/gemm.hpp). The host-side weights may be pinned
// host memory (TMA reads UVA); the hbm-side weights must be device-resident.
// ============================================================================
static void m_grouped_int8_hybrid_gemm_sm90_contiguous(
        const torch::Tensor& a,             // [M, K]     int8
        const torch::Tensor& b_host,        // [Gh, N, K] int8 (pinned or cuda)
        const torch::Tensor& b_hbm,         // [Gd, N, K] int8 (cuda)
        const torch::Tensor& d,             // [M, N]     float32
        const torch::Tensor& offsets_host,  // [2*Sh]     int32 (start,end) pairs
        const torch::Tensor& experts_host,  // [Sh+1]     int32 (with -1 terminator)
        const int& list_size_host,
        const torch::Tensor& offsets_hbm,   // [2*Sd]     int32
        const torch::Tensor& experts_hbm,   // [Sd+1]     int32
        const int& list_size_hbm,
        const int& s_host, const bool& enable_steal,
        const torch::Tensor& sfa,           // [ceil(K/128), M]    float32
        const torch::Tensor& sfb_host,      // [ceil(K/128), Gh*N] float32
        const torch::Tensor& sfb_hbm) {     // [ceil(K/128), Gd*N] float32
    DG_HOST_ASSERT(a.dim() == 2 and b_host.dim() == 3 and b_hbm.dim() == 3 and d.dim() == 2);
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t num_groups_host = b_host.size(0);
    const int64_t num_groups_hbm = b_hbm.size(0);
    const int64_t n = b_host.size(1);
    DG_HOST_ASSERT(b_host.size(2) == k and b_hbm.size(2) == k);
    DG_HOST_ASSERT(b_hbm.size(1) == n);
    DG_HOST_ASSERT(d.size(0) == m and d.size(1) == n);

    DG_HOST_ASSERT(a.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(b_host.scalar_type() == torch::kChar and b_hbm.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfb_host.scalar_type() == torch::kFloat and sfb_hbm.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(a.is_cuda() and d.is_cuda() and sfa.is_cuda());
    DG_HOST_ASSERT(sfb_host.is_cuda() and sfb_hbm.is_cuda());
    DG_HOST_ASSERT(b_host.is_cuda() or b_host.is_pinned());
    DG_HOST_ASSERT(b_hbm.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and b_host.is_contiguous() and b_hbm.is_contiguous()
                   and d.is_contiguous());
    DG_HOST_ASSERT(sfa.is_contiguous() and sfb_host.is_contiguous() and sfb_hbm.is_contiguous());
    DG_HOST_ASSERT(offsets_host.is_cuda() and experts_host.is_cuda());
    DG_HOST_ASSERT(offsets_hbm.is_cuda() and experts_hbm.is_cuda());
    DG_HOST_ASSERT(offsets_host.scalar_type() == torch::kInt and experts_host.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(offsets_hbm.scalar_type() == torch::kInt and experts_hbm.scalar_type() == torch::kInt);

    if (m == 0 or n == 0 or k == 0) return;

    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b_host) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b_hbm) == cute::UMMA::Major::K);
    sm90_m_grouped_int8_hybrid_gemm_contiguous(
        a, sfa, b_host, sfb_host, b_hbm, sfb_hbm, d,
        offsets_host, experts_host, /*num_segments_host=*/list_size_host - 1,
        offsets_hbm, experts_hbm, /*num_segments_hbm=*/list_size_hbm - 1,
        s_host, enable_steal,
        static_cast<int>(num_groups_host), static_cast<int>(num_groups_hbm),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        "nk");
}

} // namespace asym_gemm
