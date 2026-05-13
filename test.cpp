// test.cpp — MoE kernel correctness test vs CUTLASS GEMM ground truth
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// CUTLASS 2.x device GEMM
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/arch/arch.h>
#include <cutlass/layout/layout.h>

// ----- Error macros -----
#define CUDA_CHECK(err) do { \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CUTLASS_CHECK(status) do { \
    if ((status) != cutlass::Status::kSuccess) { \
        fprintf(stderr, "CUTLASS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// CUTLASS GEMM: A[M,K] RowMajor * B[N,K] ColumnMajor -> C[M,N] RowMajor
// Passing W[N,K] row-major as B ColumnMajor with ldb=K gives W^T effect.
using GemmNT = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

// Forward declaration of the MoE kernel launcher
void launch_MoECompute(
    cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
    int N, int K,
    int expert_size, int list_size,
    int* expert_list, int* index_list,
    cudaStream_t stream);

// Reference GEMM for one expert slice
void run_ref_gemm(
    cutlass::half_t* X_slice_ptr, int M,
    cutlass::half_t* W_ptr, int N, int K,
    cutlass::half_t* O_ref_ptr)
{
    GemmNT gemm_op;
    cutlass::half_t alpha_h(1.0f);
    cutlass::half_t beta_h(0.0f);

    GemmNT::Arguments args(
        {M, N, K},
        {X_slice_ptr, K},
        {W_ptr, K},
        {O_ref_ptr, N},
        {O_ref_ptr, N},
        {alpha_h, beta_h}
    );
    CUTLASS_CHECK(gemm_op(args));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Fill device buffer with random FP16
void fill_random_fp16(cutlass::half_t* d_ptr, size_t count)
{
    std::vector<cutlass::half_t> h(count);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : h) v = cutlass::half_t(dist(rng));
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), count * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
}

// Compare outputs; return (max_abs_diff, mean_abs_diff)
std::pair<float,float> compare_outputs(
    const std::vector<cutlass::half_t>& moe,
    const std::vector<cutlass::half_t>& ref,
    int N, float threshold = 0.05f)
{
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    int mismatch_count = 0;
    for (size_t i = 0; i < moe.size(); ++i) {
        float a = float(moe[i]);
        float b = float(ref[i]);
        float diff = std::fabs(a - b);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > threshold && mismatch_count < 10) {
            int row = (int)(i / N), col = (int)(i % N);
            printf("  MISMATCH [token=%d, n=%d]: moe=%.4f  ref=%.4f  diff=%.4f\n",
                   row, col, a, b, diff);
            ++mismatch_count;
        }
    }
    return {max_diff, float(sum_diff / moe.size())};
}

bool run_test(const char* name, int N, int K, int num_experts,
              int list_size, int* expert_list_h, int* token_counts)
{
    printf("\n=== %s: N=%d K=%d experts=%d ===\n", name, N, K, num_experts);

    // Build index_list (cumulative end indices)
    std::vector<int> index_list_h(list_size);
    int total_tokens = 0;
    for (int i = 0; i < list_size; ++i) {
        total_tokens += token_counts[i];
        index_list_h[i] = total_tokens;
    }
    printf("total_tokens=%d\n", total_tokens);

    // Allocate device buffers
    cutlass::half_t *d_X, *d_W, *d_O_moe, *d_O_ref;
    CUDA_CHECK(cudaMalloc(&d_X,     (size_t)total_tokens * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_W,     (size_t)num_experts  * N * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_moe, (size_t)total_tokens * N * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_ref, (size_t)total_tokens * N * sizeof(cutlass::half_t)));

    int *d_expert_list, *d_index_list;
    CUDA_CHECK(cudaMalloc(&d_expert_list, list_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_index_list,  list_size * sizeof(int)));

    // Initialize
    fill_random_fp16(d_X, (size_t)total_tokens * K);
    fill_random_fp16(d_W, (size_t)num_experts  * N * K);
    CUDA_CHECK(cudaMemset(d_O_moe, 0, (size_t)total_tokens * N * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemset(d_O_ref, 0, (size_t)total_tokens * N * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemcpy(d_expert_list, expert_list_h,     list_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_index_list,  index_list_h.data(), list_size * sizeof(int), cudaMemcpyHostToDevice));

    // Run MoE kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    launch_MoECompute(d_X, d_W, d_O_moe, N, K, num_experts, list_size,
                      d_expert_list, d_index_list, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run CUTLASS reference per expert
    int len_start = 0;
    for (int e = 0; e < list_size; ++e) {
        int expert_id = expert_list_h[e];
        int len = index_list_h[e] - len_start;
        printf("Expert %d (id=%d): %d tokens\n", e, expert_id, len);
        run_ref_gemm(
            d_X     + (size_t)len_start * K,
            len,
            d_W     + (size_t)expert_id * N * K,
            N, K,
            d_O_ref + (size_t)len_start * N
        );
        len_start = index_list_h[e];
    }

    // Download and compare
    std::vector<cutlass::half_t> h_moe(total_tokens * N);
    std::vector<cutlass::half_t> h_ref(total_tokens * N);
    CUDA_CHECK(cudaMemcpy(h_moe.data(), d_O_moe, total_tokens * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_O_ref, total_tokens * N * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    auto [max_diff, mean_diff] = compare_outputs(h_moe, h_ref, N);
    printf("Max abs diff : %.6f\n", max_diff);
    printf("Mean abs diff: %.6f\n", mean_diff);

    const float THRESHOLD = 0.05f;
    bool passed = (max_diff <= THRESHOLD);
    printf("%s Max diff %.6f %s threshold %.4f\n",
           passed ? "[PASS]" : "[FAIL]", max_diff,
           passed ? "<=" : ">", THRESHOLD);

    // Cleanup
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_O_moe)); CUDA_CHECK(cudaFree(d_O_ref));
    CUDA_CHECK(cudaFree(d_expert_list)); CUDA_CHECK(cudaFree(d_index_list));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return passed;
}

int main()
{
    // Test 1: K=32 (single k-tile, tests k=0 path only)
    {
        int expert_list[] = {0, 3, 5, 7};
        int token_counts[] = {12, 8, 20, 28};  // partial tiles (non-multiples of kBlockM=128)
        bool ok = run_test("SingleKTile", 4096, 32, 8, 4, expert_list, token_counts);
        if (!ok) return 1;
    }

    // Test 2: K=64 (two k-tiles, tests k>0 accumulation path)
    {
        int expert_list[] = {1, 2, 4, 6};
        int token_counts[] = {128, 64, 256, 12};  // mix of full and partial tiles
        bool ok = run_test("MultiKTile", 4096, 64, 8, 4, expert_list, token_counts);
        if (!ok) return 1;
    }

    printf("\nAll tests passed!\n");
    return 0;
}
