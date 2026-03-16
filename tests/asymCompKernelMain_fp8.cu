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

static int fill_with_sentinel(
    int* m_indices, int M,
    int* offsets, int* experts, int capacity
) {
    if (!offsets || !experts || capacity <= 0) return 0;

    if (M <= 0 || !m_indices) {
        return 0;
    }

    int write = 0;

    auto maybe_emit = [&](int start_idx) {
        int e = m_indices[start_idx];
        if (e != -1) {
            if (write < capacity) {
                offsets[write] = start_idx;
                experts[write] = e;
            }
            ++write;
        }
    };

    maybe_emit(0);
    for (int i = 1; i < M; ++i) {
        if (m_indices[i] != m_indices[i - 1]) {
            maybe_emit(i);
        }
    }

    if (write < capacity) {
        offsets[write] = M;
        experts[write] = -1;
    }
    ++write;

    return std::min(write, capacity);
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

int main(int argc, char** argv) {
    // JIT cache + lineinfo (optional but useful)
    setenv("DG_JIT_CACHE_DIR", "/tmp/deepgemm_jit", 1);
    setenv("DG_JIT_WITH_LINEINFO", "1", 1);
    setenv("DG_JIT_DEBUG", "1", 1);

    // CRITICAL: initialize DeepGEMM JIT globals BEFORE any kernel build
    // NOTE: pass the directory whose child is "include/asym_gemm"
    // In this repo that is: <repo>/asym_gemm
    asym_gemm::Compiler::prepare_init(
        "/sgl-workspace/sglang/AsymGEMM/asym_gemm",
        "/usr/local/cuda-12.9"
    );

    asym_gemm::KernelRuntime::prepare_init("/usr/local/cuda-12.9");

    torch::NoGradGuard ng;
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available.\n";
        return 1;
    }


    auto dev = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(dev);

    // -----------------------------
    // 3) Tensors (your current demo settings)
    // -----------------------------
    const int64_t n = 4096, k = 7168, num_groups = 4;
    const int64_t expected_m_per_group = 2048;
    int64_t m = 0, active_m = 0;

    // Scale-factor layout for recipe (1, 128, 128) with BLOCK_K=128:
    //   SFA: [m, ceil(k / 128)]
    //   SFB: [num_groups, ceil(n / 128), ceil(k / 128)]
    constexpr int64_t block_k = 128;
    constexpr int64_t sfb_gran_n = 128;
    const int64_t sf_k = asym_gemm::ceil_div(k, block_k);
    const int64_t sf_n = asym_gemm::ceil_div(n, sfb_gran_n);

    const bool disable_ue8m0_cast = true;
    const std::string compiled_dims = "nk";
    std::optional<std::tuple<int,int,int>> recipe = std::nullopt;

    auto m_indices_cpu = build_m_indices_like_generators(expected_m_per_group, num_groups, &m, &active_m);
    auto m_indices = m_indices_cpu.to(dev);
    const int max_len = static_cast<int>(num_groups) + 1;
    std::vector<int> offsets_h(max_len);
    std::vector<int> experts_h(max_len);
    const int list_size = fill_with_sentinel(m_indices_cpu.data_ptr<int>(), m_indices_cpu.numel(),
                                             offsets_h.data(), experts_h.data(), max_len);
    auto opts_i32_cuda = torch::TensorOptions().device(dev).dtype(torch::kInt32);
    auto offsets_t = torch::empty({max_len}, opts_i32_cuda);
    auto experts_t = torch::empty({max_len}, opts_i32_cuda);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemcpyAsync(offsets_t.data_ptr<int>(), offsets_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(experts_t.data_ptr<int>(), experts_h.data(),
                    max_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    auto A_fp16 = torch::randn({m, k}, torch::TensorOptions().device(dev).dtype(torch::kFloat16));
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

    std::cout << "Calling asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous...\n";
    std::cout << "m=" << m << ", active_m=" << active_m << ", n=" << n << ", k=" << k << ", num_groups=" << num_groups << "\n";
    std::cout << "A=" << A.sizes() << " " << A.scalar_type() << "  SFA=" << SFA.sizes() << " " << SFA.scalar_type() << "\n";
    std::cout << "B_cpu=" << B_cpu.sizes() << " " << B_cpu.scalar_type()
              << "  SFB_cpu=" << SFB_cpu.sizes() << " " << SFB_cpu.scalar_type() << "\n";
    std::cout << "B_gpu=" << B_gpu.sizes() << " " << B_gpu.scalar_type()
              << "  SFB_gpu=" << SFB_gpu.sizes() << " " << SFB_gpu.scalar_type() << "\n";
    std::cout << "D_asym=" << D_asym.sizes() << " " << D_asym.scalar_type()
              << "  D_deepGEMM=" << D_deepGEMM.sizes() << " " << D_deepGEMM.scalar_type()
              << "  m_indices=" << m_indices.sizes() << " " << m_indices.scalar_type()
              << "  list_size=" << list_size << "\n";

    asym_gemm::gemm::m_grouped_fp8_asym_gemm_nt_contiguous(
        a_pair, b_pair_cpu, D_asym, offsets_t, experts_t, list_size,
        recipe, compiled_dims, disable_ue8m0_cast
    );
    if (!cuda_check("m_grouped_fp8_asym_gemm_nt_contiguous")) {
        std::cerr << "Stop after asym kernel failure.\n";
        return 2;
    }
    std::cout << "asym kernel finished.\n";

    asym_gemm::gemm::m_grouped_fp8_gemm_nt_contiguous(
        a_pair, b_pair_gpu, D_deepGEMM, m_indices, recipe, compiled_dims, disable_ue8m0_cast
    );
    if (!cuda_check("m_grouped_fp8_gemm_nt_contiguous")) {
        std::cerr << "Stop after deepGEMM kernel failure.\n";
        return 3;
    }
    std::cout << "deepGEMM kernel finished.\n";

    print_5x5_compare(D_asym, D_deepGEMM);
    std::cout << "Done. D_asym.mean=" << D_asym.to(torch::kFloat32).mean().item<float>()
              << ", D_deepGEMM.mean=" << D_deepGEMM.to(torch::kFloat32).mean().item<float>() << "\n";
    return 0;
}

// ipdb> pp a[0].shape
// torch.Size([35456, 7168])
// ipdb> pp a[1].shape
// torch.Size([35456, 56])
// ipdb> pp b[0].shape
// torch.Size([4, 4096, 7168])
// ipdb> pp b[1].shape
// torch.Size([4, 32, 56])
