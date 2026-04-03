#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <string>
#include <random>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <limits>

#include "../csrc/apis/gemm.hpp"
#include "../csrc/utils/layout.hpp"
#include "../csrc/utils/math.hpp"

// IMPORTANT: include the header that owns `prepare_init` / JIT globals
#include "../csrc/jit/compiler.hpp"   // adjust if your path is different

static torch::Tensor make_fp8_e4m3(const torch::Tensor& x) {
    return x.to(torch::kFloat8_e4m3fn);
}

static bool cuda_check(const char* tag) {
    const auto launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::cerr << "[CUDA] " << tag << " launch error: " << cudaGetErrorString(launch_err) << "\n";
        return false;
    }
    const auto sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[CUDA] " << tag << " sync error: " << cudaGetErrorString(sync_err) << "\n";
        return false;
    }
    return true;
}

static void print_5x5_compare(const torch::Tensor& d_asym, const torch::Tensor& d_deepgemm) {
    auto asym_cpu = d_asym.to(torch::kCPU, torch::kFloat32).contiguous();
    auto deep_cpu = d_deepgemm.to(torch::kCPU, torch::kFloat32).contiguous();
    auto diff_cpu = (asym_cpu - deep_cpu).contiguous();

    const int64_t rows = std::min<int64_t>(5, asym_cpu.size(0));
    const int64_t cols = std::min<int64_t>(5, asym_cpu.size(1));

    auto asym_acc = asym_cpu.accessor<float, 2>();
    auto deep_acc = deep_cpu.accessor<float, 2>();
    auto diff_acc = diff_cpu.accessor<float, 2>();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nD_asym (top-left " << rows << "x" << cols << "):\n";
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            std::cout << asym_acc[i][j] << (j + 1 == cols ? '\n' : ' ');
        }
    }

    std::cout << "\nD_deepGEMM (top-left " << rows << "x" << cols << "):\n";
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            std::cout << deep_acc[i][j] << (j + 1 == cols ? '\n' : ' ');
        }
    }

    std::cout << "\nDiff = D_asym - D_deepGEMM (top-left " << rows << "x" << cols << "):\n";
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            std::cout << diff_acc[i][j] << (j + 1 == cols ? '\n' : ' ');
        }
    }

    const auto abs_diff = diff_cpu.abs();
    const auto finite_mask = torch::isfinite(abs_diff);
    const auto finite_count = finite_mask.sum().item<int64_t>();
    const auto total_count = abs_diff.numel();
    const auto non_finite_count = total_count - finite_count;

    float max_abs_diff = std::numeric_limits<float>::quiet_NaN();
    float mean_abs_diff = std::numeric_limits<float>::quiet_NaN();
    int64_t max_i = -1, max_j = -1;

    if (finite_count > 0) {
        auto finite_abs = abs_diff.masked_select(finite_mask);
        max_abs_diff = finite_abs.max().item<float>();
        mean_abs_diff = finite_abs.mean().item<float>();

        auto is_max = torch::logical_and(finite_mask, abs_diff == max_abs_diff);
        auto max_pos = torch::nonzero(is_max);
        if (max_pos.numel() > 0) {
            max_i = max_pos[0][0].item<int64_t>();
            max_j = max_pos[0][1].item<int64_t>();
        }
    }

    std::cout << "\nAbs diff stats (finite only): max=" << max_abs_diff
              << ", mean=" << mean_abs_diff
              << ", non_finite=" << non_finite_count << "/" << total_count << "\n";

    if (max_i >= 0 and max_j >= 0) {
        std::cout << "Max finite abs diff location: (" << max_i << ", " << max_j << ")\n";

        const int64_t radius = 2;
        const int64_t r0 = std::max<int64_t>(0, max_i - radius);
        const int64_t r1 = std::min<int64_t>(asym_cpu.size(0), max_i + radius + 1);
        const int64_t c0 = std::max<int64_t>(0, max_j - radius);
        const int64_t c1 = std::min<int64_t>(asym_cpu.size(1), max_j + radius + 1);

        std::cout << "Neighborhood around max diff rows [" << r0 << ", " << (r1 - 1)
                  << "], cols [" << c0 << ", " << (c1 - 1) << "]\n";

        std::cout << "\nD_asym neighborhood:\n";
        for (int64_t i = r0; i < r1; ++i) {
            for (int64_t j = c0; j < c1; ++j)
                std::cout << asym_acc[i][j] << (j + 1 == c1 ? '\n' : ' ');
        }

        std::cout << "\nD_deepGEMM neighborhood:\n";
        for (int64_t i = r0; i < r1; ++i) {
            for (int64_t j = c0; j < c1; ++j)
                std::cout << deep_acc[i][j] << (j + 1 == c1 ? '\n' : ' ');
        }

        std::cout << "\nDiff neighborhood:\n";
        for (int64_t i = r0; i < r1; ++i) {
            for (int64_t j = c0; j < c1; ++j)
                std::cout << diff_acc[i][j] << (j + 1 == c1 ? '\n' : ' ');
        }
    }

    if (non_finite_count > 0) {
        auto bad_mask = torch::logical_not(torch::isfinite(diff_cpu));
        auto bad_pos = torch::nonzero(bad_mask);
        if (bad_pos.numel() > 0) {
            const int64_t bi = bad_pos[0][0].item<int64_t>();
            const int64_t bj = bad_pos[0][1].item<int64_t>();
            std::cout << "\nFirst non-finite diff location: (" << bi << ", " << bj << ")"
                      << " D_asym=" << asym_acc[bi][bj]
                      << " D_deepGEMM=" << deep_acc[bi][bj]
                      << " Diff=" << diff_acc[bi][bj] << "\n";
        }
    }
}

// Produce offset PAIRS [start_0, end_0, start_1, end_1, ...] and experts [e0, e1, ..., -1].
// The asymScheduler reads offsets[blockIdx.y * 2] and offsets[blockIdx.y * 2 + 1].
static int fill_with_sentinel(
    int* m_indices, int M,
    int* offsets, int max_offsets,
    int* experts, int max_experts
) {
    if (!offsets || !experts || max_experts <= 0) return 0;
    if (M <= 0 || !m_indices) return 0;

    int num_experts = 0;
    int off_write = 0;

    // Find segments of contiguous expert IDs (skip -1 segments)
    int seg_start = 0;
    while (seg_start < M) {
        int e = m_indices[seg_start];
        // Find end of this segment
        int seg_end = seg_start + 1;
        while (seg_end < M && m_indices[seg_end] == e) ++seg_end;

        if (e != -1) {
            // Emit offset pair (start, end) padded to BLOCK_M alignment
            constexpr int block_m = 128;
            int start_padded = (seg_start / block_m) * block_m;
            int end_padded   = ((seg_end + block_m - 1) / block_m) * block_m;
            if (off_write + 1 < max_offsets && num_experts < max_experts) {
                offsets[off_write]     = start_padded;
                offsets[off_write + 1] = end_padded;
                experts[num_experts]   = e;
                off_write += 2;
                ++num_experts;
            }
        }
        seg_start = seg_end;
    }

    // Sentinel expert
    if (num_experts < max_experts) {
        experts[num_experts] = -1;
    }
    ++num_experts;

    return std::min(num_experts, max_experts);
}

static torch::Tensor build_m_indices_like_generators(
    int64_t expected_m_per_group,
    int64_t num_groups,
    int64_t* out_m,
    int64_t* out_active_m
) {
    const int64_t alignment = asym_gemm::get_mk_alignment_for_contiguous_layout();
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.7f, 1.3f);

    std::vector<int64_t> actual_ms;
    std::vector<int64_t> aligned_ms;
    actual_ms.reserve(num_groups);
    aligned_ms.reserve(num_groups);

    int64_t total_m = 0;
    int64_t active_m = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_m = std::max<int64_t>(1, static_cast<int64_t>(expected_m_per_group * dist(rng)));
        const int64_t aligned_m = asym_gemm::align(actual_m, alignment);
        actual_ms.push_back(actual_m);
        aligned_ms.push_back(aligned_m);
        total_m += aligned_m;
        active_m += actual_m;
    }

    auto m_indices_cpu = torch::empty({total_m},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    auto* mi = m_indices_cpu.data_ptr<int32_t>();

    int64_t start = 0;
    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t actual_end = start + actual_ms[i];
        const int64_t aligned_end = start + aligned_ms[i];
        for (int64_t j = start; j < actual_end; ++j)
            mi[j] = static_cast<int32_t>(i);
        for (int64_t j = actual_end; j < aligned_end; ++j)
            mi[j] = -1;
        start = aligned_end;
    }

    *out_m = total_m;
    *out_active_m = active_m;
    return m_indices_cpu;
}

struct ShapeConfig {
    int64_t num_groups;
    int64_t expected_m_per_group;
    int64_t n;
    int64_t k;
};

static void run_one_shape(const ShapeConfig& shape, torch::Device dev, cudaStream_t stream,
                           bool correctness_check) {
    const int64_t num_groups = shape.num_groups;
    const int64_t expected_m_per_group = shape.expected_m_per_group;
    const int64_t n = shape.n;
    const int64_t k = shape.k;
    int64_t m = 0, active_m = 0;

    constexpr int64_t block_k = 128;
    constexpr int64_t sfb_gran_n = 128;
    const int64_t sf_k = asym_gemm::ceil_div(k, block_k);
    const int64_t sf_n = asym_gemm::ceil_div(n, sfb_gran_n);

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "nk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    auto m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    auto m_indices = m_indices_cpu.to(dev);
    // Offsets are PAIRS: [start_0, end_0, start_1, end_1, ...] → 2*num_groups entries
    // Experts: [e0, e1, ..., -1] → num_groups+1 entries
    const int max_offsets = static_cast<int>(num_groups) * 2;
    const int max_experts = static_cast<int>(num_groups) + 1;
    std::vector<int> offsets_h(max_offsets);
    std::vector<int> experts_h(max_experts);
    const int list_size = fill_with_sentinel(m_indices_cpu.data_ptr<int>(), m_indices_cpu.numel(),
                                             offsets_h.data(), max_offsets,
                                             experts_h.data(), max_experts);
    auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_offsets}, opts_i32_cuda);
    auto experts_t = torch::empty({max_experts}, opts_i32_cuda);
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_offsets * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_experts * sizeof(int), cudaMemcpyHostToDevice, stream);

    auto A_fp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
    // Zero out padding rows (where m_indices == -1) to match Python generators
    {
        auto mi = m_indices_cpu.data_ptr<int32_t>();
        for (int64_t r = 0; r < m; ++r) {
            if (mi[r] == -1) A_fp16[r].zero_();
        }
    }
    auto A = make_fp8_e4m3(A_fp16);

    auto B_fp16_cpu = torch::randn(
        {num_groups, n, k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16).pinned_memory(true));
    auto B_cpu = make_fp8_e4m3(B_fp16_cpu);
    auto B_gpu = B_cpu.to(dev, /*non_blocking=*/true);

    auto D_asym = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));
    auto D_deepGEMM = torch::empty({m, n}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

    auto SFA = torch::ones({m, sf_k}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    auto SFB_cpu = torch::ones(
        {num_groups, sf_n, sf_k},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32).pinned_memory(true));
    auto SFB_gpu = SFB_cpu.to(dev, /*non_blocking=*/true);

    std::pair<torch::Tensor, torch::Tensor> a_pair{A, SFA};
    std::pair<torch::Tensor, torch::Tensor> b_pair_cpu{B_cpu, SFB_cpu};
    std::pair<torch::Tensor, torch::Tensor> b_pair_gpu{B_gpu, SFB_gpu};

    // Correctness check
    if (correctness_check) {
        asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
            a_pair, b_pair_cpu, D_asym, offsets_t, experts_t, list_size,
            recipe, compiled_dims, disable_ue8m0_cast);
        if (!cuda_check("asym")) { std::cerr << "CUDA error in asym kernel\n"; return; }

        asym_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous(
            a_pair, b_pair_gpu, D_deepGEMM, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
        if (!cuda_check("deep")) { std::cerr << "CUDA error in deep kernel\n"; return; }

        auto diff = (D_asym.to(torch::kFloat32) - D_deepGEMM.to(torch::kFloat32)).abs();
        float max_diff = diff.max().item<float>();
        float mean_diff = diff.mean().item<float>();
        std::cout << "  Correctness: max_diff=" << std::scientific << max_diff
                  << ", mean_diff=" << mean_diff << std::fixed << "\n";
    }

    // Performance benchmark
    constexpr int warmup = 5;
    constexpr int iters = 20;
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Asym benchmark
    for (int i = 0; i < warmup; ++i)
        asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
            a_pair, b_pair_cpu, D_asym, offsets_t, experts_t, list_size,
            recipe, compiled_dims, disable_ue8m0_cast);
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev, stream);
    for (int i = 0; i < iters; ++i)
        asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
            a_pair, b_pair_cpu, D_asym, offsets_t, experts_t, list_size,
            recipe, compiled_dims, disable_ue8m0_cast);
    cudaEventRecord(stop_ev, stream);
    cudaEventSynchronize(stop_ev);
    float asym_ms = 0;
    cudaEventElapsedTime(&asym_ms, start_ev, stop_ev);
    asym_ms /= iters;
    double flops = 2.0 * active_m * n * k;
    double asym_tflops = flops / (asym_ms * 1e-3) / 1e12;

    // DeepGEMM benchmark
    for (int i = 0; i < warmup; ++i)
        asym_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous(
            a_pair, b_pair_gpu, D_deepGEMM, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    cudaDeviceSynchronize();

    cudaEventRecord(start_ev, stream);
    for (int i = 0; i < iters; ++i)
        asym_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous(
            a_pair, b_pair_gpu, D_deepGEMM, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
    cudaEventRecord(stop_ev, stream);
    cudaEventSynchronize(stop_ev);
    float deep_ms = 0;
    cudaEventElapsedTime(&deep_ms, start_ev, stop_ev);
    deep_ms /= iters;
    double deep_tflops = flops / (deep_ms * 1e-3) / 1e12;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Asym (B on CPU):     " << std::setw(7) << (asym_ms * 1000) << " us | "
              << std::setw(6) << asym_tflops << " TFLOPS\n";
    std::cout << "  DeepGEMM (B on GPU): " << std::setw(7) << (deep_ms * 1000) << " us | "
              << std::setw(6) << deep_tflops << " TFLOPS\n";
    std::cout << "  Ratio (asym/deep):   " << std::setprecision(3) << (asym_ms / deep_ms) << "x\n";

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
}

int main(int argc, char** argv) {
    // Force unbuffered stdout so output is visible in real-time
    std::cout.setf(std::ios::unitbuf);
    setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);

    asym_gemm::Compiler::prepare_init(
        "/asymGEMMFP8/AsymGEMM/asym_gemm",
        "/usr/local/cuda"
    );
    asym_gemm::KernelRuntime::prepare_init("/usr/local/cuda");

    torch::NoGradGuard ng;
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available.\n";
        return 1;
    }

    auto dev = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(dev);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // MoE-relevant shapes: {num_groups, expected_m_per_group, n, k}
    std::vector<ShapeConfig> shapes = {
        {4,  2048, 4096, 7168},
        {4,  8192, 7168, 2048},
        {8,  4096, 7168, 2048},
    };

    bool check_correctness = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-check") check_correctness = false;
    }

    std::cout << "=== FP8 Asymmetric GEMM Benchmark (B on CPU via NVLink-C2C) ===\n\n";

    for (const auto& shape : shapes) {
        std::cout << "--- Shape: num_groups=" << shape.num_groups
                  << ", m_per_group=" << shape.expected_m_per_group
                  << ", n=" << shape.n << ", k=" << shape.k << " ---\n";
        run_one_shape(shape, dev, stream, check_correctness);
        std::cout << "\n";
    }

    return 0;
}
