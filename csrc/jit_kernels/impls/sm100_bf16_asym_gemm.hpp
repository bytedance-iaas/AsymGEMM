#pragma once

#include <torch/python.h>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace asym_gemm {

class SM100BF16AsymGemmRuntime final: public LaunchRuntime<SM100BF16AsymGemmRuntime> {
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
#include <asym_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_asym_gemm_impl<
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
        args.gemm_config.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts, 
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

class SM100BF16EpQueuedAsymGemmRuntime final: public LaunchRuntime<SM100BF16EpQueuedAsymGemmRuntime> {
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

        // sEP queued scheduling (gb200_ep.md E3): 3 int32 counters in PINNED HOST memory
        // shared by both devices ([0]=claimed, [1]=head_taken, [2]=tail_taken).
        void* ep_queue;
        uint32_t ep_total_items;
        uint32_t ep_side;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define ASYM_BF16_KERNEL_NAME sm100_bf16_asym_gemm_ep_queued_impl
#define ASYM_BF16_EP_QUEUED 1
#define ASYM_BF16_KERNEL_EXTRA_ARGS , int* ep_queue, uint32_t ep_total_items, uint32_t ep_side
#include <asym_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_asym_gemm_ep_queued_impl<
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
        args.gemm_config.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd,
            args.ep_queue, args.ep_total_items, args.ep_side));
    }
};

class SM100BF16Qwen3RoutedAsymGemmRuntime final: public LaunchRuntime<SM100BF16Qwen3RoutedAsymGemmRuntime> {
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

        void* route_token_indices;
        void* route_weights;
        int route_weights_is_bf16;
        int route_weighted;
        void* route_scatter_out;
        int route_scatter_stride;
        void* route_gather_left;
        int route_gather_stride;

        bool gather_left;
        bool scatter_add;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define ASYM_BF16_KERNEL_NAME qwen3_moe_bf16_asym_routed_impl
#define ASYM_BF16_ROUTE_GATHER_LEFT {}
#define ASYM_BF16_ROUTE_SCATTER_ADD {}
#define ASYM_BF16_KERNEL_EXTRA_ARGS , const int64_t* route_token_indices, const void* route_weights, uint32_t route_weights_is_bf16, uint32_t route_weighted, float* route_scatter_out, uint32_t route_scatter_stride, const cutlass::bfloat16_t* route_gather_left, uint32_t route_gather_stride
#include <asym_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&qwen3_moe_bf16_asym_routed_impl<
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
        args.gather_left ? 1 : 0,
        args.scatter_add ? 1 : 0,
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
        args.gemm_config.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd,
            args.route_token_indices,
            args.route_weights,
            args.route_weights_is_bf16,
            args.route_weighted,
            args.route_scatter_out,
            args.route_scatter_stride,
            args.route_gather_left,
            args.route_gather_stride));
    }
};

class SM100BF16CpuLeftAsymGemmRuntime final: public LaunchRuntime<SM100BF16CpuLeftAsymGemmRuntime> {
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
        CUtensorMap tensor_map_b_pair;
        CUtensorMap tensor_map_cd_pair;

        bool compact_m_block_grid;
        bool pair_output;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <asym_gemm/impls/sm100_bf16_cpu_left_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_cpu_left_asym_gemm_impl<
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
        args.gemm_config.tc_util,
        args.compact_m_block_grid,
        args.pair_output);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.offsets, args.experts,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd,
            args.tensor_map_b_pair,
            args.tensor_map_cd_pair));
    }
};

// int fill_with_sentinel(
//     int* m_indices, int M,
//     int* offsets, int* experts, int capacity
// ) {
//     if (!offsets || !experts || capacity <= 0) return 0;

//     if (M <= 0 || !m_indices) {
//         return 0;
//     }

//     int write = 0;

//     auto maybe_emit = [&](int start_idx) {
//         int e = m_indices[start_idx];
//         if (e != -1) {
//             if (write < capacity) {
//                 offsets[write] = start_idx;
//                 experts[write] = e;
//             }
//             ++write;
//         }
//     };

//     maybe_emit(0);
//     for (int i = 1; i < M; ++i) {
//         if (m_indices[i] != m_indices[i - 1]) {
//             maybe_emit(i);
//         }
//     }

//     // Append sentinel: (M, -1)
//     if (write < capacity) {
//         offsets[write] = M;
//         experts[write] = -1;
//     }
//     ++write;

//     return std::min(write, capacity);
// }

static void sm100_m_grouped_bf16_asym_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const int& grid_y,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const int b_outer_stride = -1) {
    const int block_m = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_M", 64)
        : get_env<int>("DG_BF16_BLOCK_M", 64);
    // SM100 legality requires block_n <= 128 when k <= 256.
    // const int block_n = (k <= 256) ? 128 : 256;
    // const int block_k = 64;

    const int block_n = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_N", 64)
        : get_env<int>("DG_BF16_BLOCK_N", 64);
    const int default_transpose_block_k = (k >= 768) ? 256 : 64;
    const int block_k = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_K", default_transpose_block_k)
        : get_env<int>("DG_BF16_BLOCK_K", 512);
    const auto& aligned_k = align(k, block_k);

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::KernelNoSF,
            // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
            m, n, k, 1, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::KernelNoSF,
            // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
            m, n, k, 1, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms());

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const int outer_b = (b_outer_stride >= 0)
        ? b_outer_stride
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               outer_b, num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);

    int max_len = (int)b.size(0) + 1;

    // 1) read m_indices on CPU
    // auto m_indices_cpu = m_indices.to(torch::kCPU);
    // auto* mi = m_indices_cpu.data_ptr<int>();
    // int M = (int)m_indices_cpu.numel();

    // 2) build offsets/experts on CPU
    // std::vector<int> offsets_h(max_len);
    // std::vector<int> experts_h(max_len);
    // int list_size = fill_with_sentinel(mi, m_indices.size(0), offsets_h.data(), experts_h.data(), max_len);
    // (void)list_size;

    // // 3) allocate offsets/experts on GPU (int32)
    // auto opts_i32_cuda = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
    // auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    // auto experts_t = torch::empty({max_len}, opts_i32_cuda);

    // // 4) copy host -> device (async on current stream)
    // cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    // cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    // cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
    //                 max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    // // 5) pass device pointers to kernel
    // void* offsets = (void*)offsets_t.data_ptr<int>();
    // void* experts = (void*)experts_t.data_ptr<int>();

    // std::cout << "list_size = " << list_size << " (max_len=" << max_len << ")\n";
    // for (int i = 0; i < list_size; ++i) {
    //     std::cout << "pair[" << i << "]: offset=" << offsets_h[i]
    //             << ", expert=" << experts_h[i] << "\n";
    // }

    // printf("ceil_div(n, config.block_n): %d, num_groups: %d \n", ceil_div(n, config.block_n), num_groups);

    if (grid_y <= 0)
        return;

    // Launch
    const SM100BF16AsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), grid_y}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100BF16AsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous", code);
    SM100BF16AsymGemmRuntime::launch(runtime, args);
}

static void sm100_m_grouped_bf16_asym_gemm_contiguous_ep_queued(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const torch::Tensor& ep_queue_t,
                                                 const int& ep_side,
                                                 const int& grid_y,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const int b_outer_stride = -1) {
    // sEP queued launch (gb200_ep.md E3): identical config/grid to the static launcher; the
    // kernel claims (segment, n-block) items from ep_queue_t (pinned host int32[>=3]).
    const int block_m = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_M", 64)
        : get_env<int>("DG_BF16_BLOCK_M", 64);
    const int block_n = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_N", 64)
        : get_env<int>("DG_BF16_BLOCK_N", 64);
    const int default_transpose_block_k = (k >= 768) ? 256 : 64;
    const int block_k = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_K", default_transpose_block_k)
        : get_env<int>("DG_BF16_BLOCK_K", 512);
    const auto& aligned_k = align(k, block_k);

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::KernelNoSF,
            m, n, k, 1, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
            GemmType::MGroupedContiguous, KernelType::KernelNoSF,
            m, n, k, 1, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms());
    // HC-EP2: the queued kernel must not cluster-launch.
    DG_HOST_ASSERT(config.multicast_config.num_multicast == 1);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const int outer_b = (b_outer_stride >= 0)
        ? b_outer_stride
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               outer_b, num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);

    if (grid_y <= 0)
        return;

    const int num_n_blocks = ceil_div(n, config.block_n);
    const uint32_t ep_total_items = static_cast<uint32_t>(grid_y) * static_cast<uint32_t>(num_n_blocks);
    // Per-device CTA budget: a fraction of the item count. Both devices together must cover
    // >= 100% of items (any CTA can claim any item); the excess above 50% per device is the
    // steal margin. 0.75 x 2 = 1.5x coverage; no-ticket CTAs exit at ~atomic cost.
    const int grid_pct = std::max(50, std::min(100, get_env<int>("DG_EP_QUEUE_GRID_PCT", 75)));
    const int grid_y_local = std::max(1, std::min(grid_y, (grid_y * grid_pct + 99) / 100));

    const SM100BF16EpQueuedAsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({num_n_blocks, grid_y_local}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .ep_queue = ep_queue_t.data_ptr<int>(),
        .ep_total_items = ep_total_items,
        .ep_side = static_cast<uint32_t>(ep_side)
    };
    const auto& code = SM100BF16EpQueuedAsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous_ep_queued", code);
    SM100BF16EpQueuedAsymGemmRuntime::launch(runtime, args);
}

static void sm100_m_grouped_bf16_asym_gemm_qwen3_routed(
                                                 const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const torch::Tensor& token_indices_t,
                                                 const torch::Tensor& route_weights_t,
                                                 const int& grid_y,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const int b_outer_stride,
                                                 const bool gather_left,
                                                 const bool scatter_add,
                                                 const bool route_weighted,
                                                 const bool route_weights_is_bf16,
                                                 const int route_scatter_stride,
                                                 const int route_gather_stride,
                                                 const torch::Tensor& route_scatter_out,
                                                 const torch::Tensor& route_gather_left) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(gather_left || scatter_add);
    DG_HOST_ASSERT(!(gather_left && scatter_add));
    if (scatter_add)
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);

    const int block_m = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_M", 64)
        : get_env<int>("DG_BF16_BLOCK_M", 64);
    const int block_n = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_N", 64)
        : get_env<int>("DG_BF16_BLOCK_N", 64);
    const int default_transpose_block_k = (k >= 768) ? 256 : 64;
    const int block_k = (major_b == cute::UMMA::Major::MN)
        ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_K", default_transpose_block_k)
        : get_env<int>("DG_BF16_BLOCK_K", 512);
    const auto& aligned_k = align(k, block_k);

    DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = get_manual_config_asym<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, 1, major_a, major_b,
        torch::kBFloat16, d.scalar_type(), false,
        device_runtime->get_num_sms(),
        block_m, block_n, block_k);
    if (gather_left)
        DG_HOST_ASSERT(config.multicast_config.num_multicast == 1);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               b_outer_stride, num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);

    if (grid_y <= 0)
        return;

    const SM100BF16Qwen3RoutedAsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k, .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), grid_y}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .route_token_indices = const_cast<void*>(reinterpret_cast<const void*>(token_indices_t.data_ptr<int64_t>())),
        .route_weights = route_weights_t.defined() ? const_cast<void*>(route_weights_t.data_ptr()) : nullptr,
        .route_weights_is_bf16 = route_weights_is_bf16 ? 1 : 0,
        .route_weighted = route_weighted ? 1 : 0,
        .route_scatter_out = scatter_add ? route_scatter_out.data_ptr<float>() : nullptr,
        .route_scatter_stride = route_scatter_stride,
        .route_gather_left = gather_left ? const_cast<void*>(reinterpret_cast<const void*>(route_gather_left.data_ptr<at::BFloat16>())) : nullptr,
        .route_gather_stride = route_gather_stride,
        .gather_left = gather_left,
        .scatter_add = scatter_add
    };
    const auto& code = SM100BF16Qwen3RoutedAsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_qwen3_moe_routed_asym_gemm", code);
    SM100BF16Qwen3RoutedAsymGemmRuntime::launch(runtime, args);
}

static void sm100_m_grouped_bf16_cpu_left_asym_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const int& grid_y,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const int b_outer_stride = -1,
                                                 const int compact_m_blocks = 0) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const int block_m = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_M", get_env<int>("DG_BF16_BLOCK_M", 64));
    const int block_n = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_N", get_env<int>("DG_BF16_BLOCK_N", 64));
    const int block_k = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_K", get_env<int>("DG_BF16_BLOCK_K", 512));
    const auto& aligned_k = align(k, block_k);

    DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = get_manual_config_asym<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, 1, major_a, major_b,
        torch::kBFloat16, d.scalar_type(), false,
        device_runtime->get_num_sms(),
        block_m, block_n, block_k);
    DG_HOST_ASSERT(config.multicast_config.num_multicast == 1);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const int outer_b = (b_outer_stride >= 0)
        ? b_outer_stride
        : static_cast<int>(b.stride(get_non_contiguous_dim(major_b)));
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               outer_b, num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                 SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                 SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                 static_cast<int>(d.stride(-2)), 1,
                                                 config.smem_config.swizzle_cd_mode);

    if (grid_y <= 0)
        return;

    const bool compact_m_block_grid = compact_m_blocks > 0;
    const int grid_x = compact_m_block_grid ? compact_m_blocks : ceil_div(m, config.block_m);

    const SM100BF16CpuLeftAsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k, .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({grid_x, grid_y}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
        .tensor_map_b_pair = tensor_map_b,
        .tensor_map_cd_pair = tensor_map_cd,
        .compact_m_block_grid = compact_m_block_grid,
        .pair_output = false
    };
    const auto& code = SM100BF16CpuLeftAsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_cpu_left_asym_gemm_contiguous", code);
    SM100BF16CpuLeftAsymGemmRuntime::launch(runtime, args);
}

// Not on the current best benchmark paths; retained for experimental paired CPU-left LoRA-A runs.
static void sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b_gate,
                                                 const torch::Tensor& b_up,
                                                 const torch::Tensor& d_gate,
                                                 const torch::Tensor& d_up,
                                                 const torch::Tensor& offsets_t,
                                                 const torch::Tensor& experts_t,
                                                 const int& grid_y,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const int b_outer_stride = -1,
                                                 const int compact_m_blocks = 0) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const int block_m = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_M", get_env<int>("DG_BF16_BLOCK_M", 64));
    const int block_n = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_N", get_env<int>("DG_BF16_BLOCK_N", 64));
    const int block_k = get_env<int>("DG_BF16_CPU_LEFT_BLOCK_K", get_env<int>("DG_BF16_BLOCK_K", 512));
    const auto& aligned_k = align(k, block_k);

    DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = get_manual_config_asym<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::KernelNoSF,
        m, n, k, 1, major_a, major_b,
        torch::kBFloat16, d_gate.scalar_type(), false,
        device_runtime->get_num_sms(),
        block_m, block_n, block_k);
    DG_HOST_ASSERT(config.multicast_config.num_multicast == 1);

    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const int outer_b = (b_outer_stride >= 0)
        ? b_outer_stride
        : static_cast<int>(b_gate.stride(get_non_contiguous_dim(major_b)));
    const auto& tensor_map_b_gate = make_tma_b_desc(major_b, b_gate, n, k,
                                                    SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                                    config.block_k,
                                                    outer_b, num_groups,
                                                    config.smem_config.swizzle_b_mode);
    const auto& tensor_map_b_up = make_tma_b_desc(major_b, b_up, n, k,
                                                  SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                                  config.block_k,
                                                  static_cast<int>(b_up.stride(get_non_contiguous_dim(major_b))), num_groups,
                                                  config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d_gate = make_tma_cd_desc(d_gate, m, n,
                                                     SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                     SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                     static_cast<int>(d_gate.stride(-2)), 1,
                                                     config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_d_up = make_tma_cd_desc(d_up, m, n,
                                                   SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                   SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                   static_cast<int>(d_up.stride(-2)), 1,
                                                   config.smem_config.swizzle_cd_mode);

    if (grid_y <= 0)
        return;

    const bool compact_m_block_grid = compact_m_blocks > 0;
    const int grid_x = compact_m_block_grid ? compact_m_blocks : ceil_div(m, config.block_m);

    const SM100BF16CpuLeftAsymGemmRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k, .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({grid_x, grid_y}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = offsets_t.data_ptr<int>(),
        .experts = experts_t.data_ptr<int>(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b_gate,
        .tensor_map_cd = tensor_map_d_gate,
        .tensor_map_b_pair = tensor_map_b_up,
        .tensor_map_cd_pair = tensor_map_d_up,
        .compact_m_block_grid = compact_m_block_grid,
        .pair_output = true
    };
    const auto& code = SM100BF16CpuLeftAsymGemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_cpu_left_pair_asym_gemm_contiguous", code);
    SM100BF16CpuLeftAsymGemmRuntime::launch(runtime, args);
}

// static void sm100_m_grouped_bf16_asym_gemm_contiguous_with_offsets(const torch::Tensor& a,
//                                                 const torch::Tensor& b,
//                                                 const torch::Tensor& d,
//                                                 const torch::Tensor& m_indices,
//                                                 const torch::Tensor& offsets_t,
//                                                 const torch::Tensor& experts_t,
//                                                 const int& list_size,
//                                                 const int& num_groups, const int& m, const int& n, const int& k,
//                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
//                                                 const std::string& compiled_dims) {
//     const auto& aligned_k = align(k, 64);
//     const auto& config = get_best_config_asym<SM100ArchSpec>(
//         GemmType::MGroupedContiguous, KernelType::KernelNoSF,
//         // NOTES: `num_groups` is 1, since the contiguous layout is seen as a whole
//         m, n, k, 1, major_a, major_b,
//         torch::kBFloat16, d.scalar_type(), false,
//         device_runtime->get_num_sms());

//     const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
//                                                SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
//                                                config.block_k,
//                                                static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
//                                                config.smem_config.swizzle_a_mode);
//     const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
//                                                SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
//                                                config.block_k,
//                                                static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
//                                                config.smem_config.swizzle_b_mode);
//     const auto& tensor_map_cd = make_tma_cd_desc(d, m, n,
//                                                  SM100ArchSpec::get_cd_store_block_m(config.block_m),
//                                                  SM100ArchSpec::get_cd_store_block_n(config.block_n),
//                                                  static_cast<int>(d.stride(-2)), 1,
//                                                  config.smem_config.swizzle_cd_mode);

//     DG_HOST_ASSERT(offsets_t.is_cuda() && experts_t.is_cuda());
//     DG_HOST_ASSERT(offsets_t.is_contiguous() && experts_t.is_contiguous());
//     DG_HOST_ASSERT(offsets_t.scalar_type() == torch::kInt && experts_t.scalar_type() == torch::kInt);
//     DG_HOST_ASSERT(offsets_t.numel() >= list_size && experts_t.numel() >= list_size);

//     void* offsets = (void*)offsets_t.data_ptr<int>();
//     void* experts = (void*)experts_t.data_ptr<int>();

//     const SM100BF16AsymGemmRuntime::Args& args = {
//         .m = m, .n = n, .k = aligned_k,
//         .compiled_dims = compiled_dims,
//         .gemm_config = config,
//         .launch_args = LaunchArgs({ceil_div(n, config.block_n), num_groups}, config.thread_config.num_threads,
//                                   config.smem_config.smem_size,
//                                   config.multicast_config.num_multicast),
//         .offsets = offsets,
//         .experts = experts,
//         .tensor_map_a = tensor_map_a,
//         .tensor_map_b = tensor_map_b,
//         .tensor_map_cd = tensor_map_cd
//     };
//     const auto& code = SM100BF16AsymGemmRuntime::generate(args);
//     const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_contiguous_with_offsets", code);
//     SM100BF16AsymGemmRuntime::launch(runtime, args);
// }

// ============================================================================
// Masked GEMM Implementation
// ============================================================================
class SM100BF16AsymGemmMaskedRuntime final: public LaunchRuntime<SM100BF16AsymGemmMaskedRuntime> {
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
#include <asym_gemm/impls/sm100_bf16_asym_gemm.cuh>

using namespace asym_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_asym_gemm_impl<
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

static void sm100_m_grouped_bf16_asym_gemm_masked(const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& grid_y,
                                             const int& expected_m,
                                             const int& num_groups, const int& m, const int& n, const int& k,
                                             const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                             const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 64);
    const int block_m = 64;
    const int block_n = 64;
    const int block_k = 512;

    const bool use_manual_config = block_m > 0 or block_n > 0 or block_k > 0;
    if (use_manual_config)
        DG_HOST_ASSERT(block_m > 0 and block_n > 0 and block_k > 0);
    const auto& config = use_manual_config
        ? get_manual_config_asym<SM100ArchSpec>(
            GemmType::MGroupedMasked, KernelType::KernelNoSF,
            expected_m, n, k, num_groups, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms(),
            block_m, block_n, block_k)
        : get_best_config_asym<SM100ArchSpec>(
            GemmType::MGroupedMasked, KernelType::KernelNoSF,
            expected_m, n, k, num_groups, major_a, major_b,
            torch::kBFloat16, d.scalar_type(), false,
            device_runtime->get_num_sms());

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

    if (grid_y <= 0)
        return;

    // Launch with masked configuration.
    // The kernel scheduler (asymScheduler.cuh, MGroupedMasked branch) reads
    // `offsets[blockIdx.y]` as `masked_m[blockIdx.y]` and ignores `experts`,
    // so we pass `masked_m` as `offsets` and `nullptr` as `experts` — matches
    // the FP8/FP4 masked dispatchers.
    const SM100BF16AsymGemmMaskedRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs({ceil_div(n, config.block_n), grid_y}, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .offsets = masked_m.data_ptr<int>(),
        .experts = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto& code = SM100BF16AsymGemmMaskedRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_asym_gemm_masked", code);
    SM100BF16AsymGemmMaskedRuntime::launch(runtime, args);
}

} // namespace asym_gemm
