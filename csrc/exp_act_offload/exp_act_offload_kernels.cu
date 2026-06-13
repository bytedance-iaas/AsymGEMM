// Copyright (c) 2026.

#include "../apis/exp_act_offload.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace asym_gemm::exp_act_offload {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float to_float(at::BFloat16 v) {
    return static_cast<float>(v);
}

__device__ __forceinline__ at::BFloat16 to_bf16(float v) {
    return static_cast<at::BFloat16>(v);
}

int blocks_for(int64_t total) {
    if (total <= 0) return 0;
    return static_cast<int>((total + kThreads - 1) / kThreads);
}

void check_cuda_bf16_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_bf16_3d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cpu_bf16_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be CPU");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be BF16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_pinned(), name, " must be pinned CPU memory");
}

void check_offsets_experts(const torch::Tensor& offsets, const torch::Tensor& experts, int64_t list_size) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(experts.is_cuda(), "experts must be CUDA");
    TORCH_CHECK(offsets.dim() == 1 && experts.dim() == 1, "offsets/experts must be 1D");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32, "offsets must be int32");
    TORCH_CHECK(experts.scalar_type() == torch::kInt32, "experts must be int32");
    TORCH_CHECK(offsets.is_contiguous() && experts.is_contiguous(), "offsets/experts must be contiguous");
    TORCH_CHECK(list_size >= 1, "list_size must include the sentinel entry");
    TORCH_CHECK(experts.numel() >= list_size, "experts shorter than list_size");
    TORCH_CHECK(offsets.numel() >= 2 * (list_size - 1), "offsets must be pair offsets [2 * groups]");
}

struct GroupPlan {
    int64_t groups;
    int64_t max_rows;
};

GroupPlan validate_group_plan(
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size,
    int64_t rows,
    int64_t num_experts) {
    check_offsets_experts(offsets, experts, list_size);
    const int64_t groups = list_size - 1;
    auto offsets_cpu = offsets.to(torch::kCPU);
    auto experts_cpu = experts.to(torch::kCPU);
    const auto offsets_acc = offsets_cpu.accessor<int32_t, 1>();
    const auto experts_acc = experts_cpu.accessor<int32_t, 1>();
    int64_t max_rows = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t start = offsets_acc[2 * g];
        const int64_t end = offsets_acc[2 * g + 1];
        const int64_t expert = experts_acc[g];
        TORCH_CHECK(0 <= start && start <= end && end <= rows, "invalid grouped row offsets");
        TORCH_CHECK(expert >= 0 && expert < num_experts, "invalid expert id in grouped metadata");
        max_rows = std::max<int64_t>(max_rows, end - start);
    }
    return {.groups = groups, .max_rows = max_rows};
}

__global__ void lora_a_grad_kernel(
    const at::BFloat16* __restrict__ dS0,
    const at::BFloat16* __restrict__ dS1,
    const at::BFloat16* __restrict__ source_cpu,
    float* __restrict__ grad0_acc,
    float* __restrict__ grad1_acc,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t rows_total,
    int32_t rank,
    int32_t k_total,
    int64_t max_linear) {
    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (group >= groups) return;
    const int32_t expert = experts[group];
    const int32_t start = offsets[2 * group];
    const int32_t end = offsets[2 * group + 1];
    if (expert < 0 || start >= end) return;

    const int64_t rows = static_cast<int64_t>(end - start);
    const int64_t total = rows * static_cast<int64_t>(k_total);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total && linear < max_linear;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t k = static_cast<int32_t>(linear % k_total);
        const int32_t row = start + static_cast<int32_t>(linear / k_total);
        const float x = to_float(source_cpu[static_cast<int64_t>(row) * k_total + k]);
        const int64_t ds_base = static_cast<int64_t>(row) * rank;
        const int64_t grad_base = static_cast<int64_t>(expert) * rank * k_total + k;
        for (int32_t r = 0; r < rank; ++r) {
            atomicAdd(&grad0_acc[grad_base + static_cast<int64_t>(r) * k_total], to_float(dS0[ds_base + r]) * x);
            if (dS1 != nullptr && grad1_acc != nullptr) {
                atomicAdd(&grad1_acc[grad_base + static_cast<int64_t>(r) * k_total], to_float(dS1[ds_base + r]) * x);
            }
        }
    }
}

__global__ void lora_b_backward_kernel(
    const at::BFloat16* __restrict__ grad_out_cpu,
    const at::BFloat16* __restrict__ low_rank,
    const at::BFloat16* __restrict__ lora_b,
    float* __restrict__ dS_acc,
    float* __restrict__ grad_b_acc,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ experts,
    int32_t groups,
    int32_t rows_total,
    int32_t out_dim,
    int32_t rank,
    float scale,
    int64_t max_linear) {
    const int32_t group = static_cast<int32_t>(blockIdx.y);
    if (group >= groups) return;
    const int32_t expert = experts[group];
    const int32_t start = offsets[2 * group];
    const int32_t end = offsets[2 * group + 1];
    if (expert < 0 || start >= end) return;

    const int64_t rows = static_cast<int64_t>(end - start);
    const int64_t total = rows * static_cast<int64_t>(out_dim);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total && linear < max_linear;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t i = static_cast<int32_t>(linear % out_dim);
        const int32_t row = start + static_cast<int32_t>(linear / out_dim);
        const float g = scale * to_float(grad_out_cpu[static_cast<int64_t>(row) * out_dim + i]);
        const int64_t lr_base = static_cast<int64_t>(row) * rank;
        const int64_t b_base = (static_cast<int64_t>(expert) * out_dim + i) * rank;
        for (int32_t r = 0; r < rank; ++r) {
            const float s = to_float(low_rank[lr_base + r]);
            const float b = to_float(lora_b[b_base + r]);
            atomicAdd(&dS_acc[lr_base + r], g * b);
            atomicAdd(&grad_b_acc[b_base + r], g * s);
        }
    }
}

__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t total) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    dst[linear] = to_bf16(src[linear]);
}

void cast_fp32_to_bf16(const torch::Tensor& src, const torch::Tensor& dst, cudaStream_t stream) {
    const int64_t total = src.numel();
    if (total == 0) return;
    cast_fp32_to_bf16_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
        src.data_ptr<float>(),
        dst.data_ptr<at::BFloat16>(),
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void require_sm100() {
    const auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr && props->major == 10, "expert activation offload kernels require SM100");
}

}  // namespace

void sm100_grouped_lora_a_grad_bf16_cpu_right(
    const torch::Tensor& grad_low_rank,
    const torch::Tensor& source_cpu,
    const torch::Tensor& grad_a,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size) {
    check_cuda_bf16_2d(grad_low_rank, "grad_low_rank");
    check_cpu_bf16_2d(source_cpu, "source_cpu");
    check_cuda_bf16_3d(grad_a, "grad_a");
    TORCH_CHECK(grad_low_rank.size(0) == source_cpu.size(0), "grad/source row mismatch");
    TORCH_CHECK(grad_a.size(1) == grad_low_rank.size(1), "grad_a rank mismatch");
    TORCH_CHECK(grad_a.size(2) == source_cpu.size(1), "grad_a K mismatch");
    TORCH_CHECK(grad_low_rank.device() == grad_a.device(), "grad_low_rank and grad_a must share device");
    TORCH_CHECK(offsets.device() == grad_low_rank.device() && experts.device() == grad_low_rank.device(),
                "metadata must be on grad_low_rank CUDA device");

    const c10::cuda::CUDAGuard device_guard(grad_low_rank.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, grad_low_rank.size(0), grad_a.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        grad_a.zero_();
        return;
    }

    auto grad_acc = torch::zeros(grad_a.sizes(), grad_a.options().dtype(torch::kFloat32));
    const int64_t max_linear = plan.max_rows * source_cpu.size(1);
    dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lora_a_grad_kernel<<<grid, kThreads, 0, stream>>>(
        grad_low_rank.data_ptr<at::BFloat16>(),
        nullptr,
        source_cpu.data_ptr<at::BFloat16>(),
        grad_acc.data_ptr<float>(),
        nullptr,
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(grad_low_rank.size(0)),
        static_cast<int32_t>(grad_low_rank.size(1)),
        static_cast<int32_t>(source_cpu.size(1)),
        max_linear);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_fp32_to_bf16(grad_acc, grad_a, stream);
}

void sm100_grouped_lora_a_pair_grad_bf16_cpu_right(
    const torch::Tensor& dS_gate,
    const torch::Tensor& dS_up,
    const torch::Tensor& x_cpu,
    const torch::Tensor& grad_gate,
    const torch::Tensor& grad_up,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size) {
    check_cuda_bf16_2d(dS_gate, "dS_gate");
    check_cuda_bf16_2d(dS_up, "dS_up");
    check_cpu_bf16_2d(x_cpu, "x_cpu");
    check_cuda_bf16_3d(grad_gate, "grad_gate");
    check_cuda_bf16_3d(grad_up, "grad_up");
    TORCH_CHECK(dS_gate.sizes() == dS_up.sizes(), "dS pair shape mismatch");
    TORCH_CHECK(grad_gate.sizes() == grad_up.sizes(), "grad pair shape mismatch");
    TORCH_CHECK(dS_gate.size(0) == x_cpu.size(0), "dS/source row mismatch");
    TORCH_CHECK(grad_gate.size(1) == dS_gate.size(1), "grad rank mismatch");
    TORCH_CHECK(grad_gate.size(2) == x_cpu.size(1), "grad K mismatch");
    TORCH_CHECK(dS_gate.device() == dS_up.device() && dS_gate.device() == grad_gate.device() && dS_gate.device() == grad_up.device(),
                "CUDA tensors must share device");
    TORCH_CHECK(offsets.device() == dS_gate.device() && experts.device() == dS_gate.device(),
                "metadata must be on dS CUDA device");

    const c10::cuda::CUDAGuard device_guard(dS_gate.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, dS_gate.size(0), grad_gate.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        grad_gate.zero_();
        grad_up.zero_();
        return;
    }

    auto grad_gate_acc = torch::zeros(grad_gate.sizes(), grad_gate.options().dtype(torch::kFloat32));
    auto grad_up_acc = torch::zeros(grad_up.sizes(), grad_up.options().dtype(torch::kFloat32));
    const int64_t max_linear = plan.max_rows * x_cpu.size(1);
    dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lora_a_grad_kernel<<<grid, kThreads, 0, stream>>>(
        dS_gate.data_ptr<at::BFloat16>(),
        dS_up.data_ptr<at::BFloat16>(),
        x_cpu.data_ptr<at::BFloat16>(),
        grad_gate_acc.data_ptr<float>(),
        grad_up_acc.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(dS_gate.size(0)),
        static_cast<int32_t>(dS_gate.size(1)),
        static_cast<int32_t>(x_cpu.size(1)),
        max_linear);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_fp32_to_bf16(grad_gate_acc, grad_gate, stream);
    cast_fp32_to_bf16(grad_up_acc, grad_up, stream);
}

void sm100_grouped_lora_b_backward_bf16_cpu_source(
    const torch::Tensor& grad_out_cpu,
    const torch::Tensor& low_rank,
    const torch::Tensor& lora_b,
    const torch::Tensor& dS,
    const torch::Tensor& grad_b,
    const torch::Tensor& offsets,
    const torch::Tensor& experts,
    int64_t list_size,
    double scale) {
    check_cpu_bf16_2d(grad_out_cpu, "grad_out_cpu");
    check_cuda_bf16_2d(low_rank, "low_rank");
    check_cuda_bf16_3d(lora_b, "lora_b");
    check_cuda_bf16_2d(dS, "dS");
    check_cuda_bf16_3d(grad_b, "grad_b");
    TORCH_CHECK(grad_out_cpu.size(0) == low_rank.size(0), "grad/low_rank row mismatch");
    TORCH_CHECK(dS.size(0) == low_rank.size(0) && dS.size(1) == low_rank.size(1), "dS shape mismatch");
    TORCH_CHECK(lora_b.size(1) == grad_out_cpu.size(1), "lora_b output dim mismatch");
    TORCH_CHECK(lora_b.size(2) == low_rank.size(1), "lora_b rank mismatch");
    TORCH_CHECK(grad_b.sizes() == lora_b.sizes(), "grad_b shape mismatch");
    TORCH_CHECK(low_rank.device() == lora_b.device() && low_rank.device() == dS.device() && low_rank.device() == grad_b.device(),
                "CUDA tensors must share device");
    TORCH_CHECK(offsets.device() == low_rank.device() && experts.device() == low_rank.device(),
                "metadata must be on low_rank CUDA device");

    const c10::cuda::CUDAGuard device_guard(low_rank.device());
    require_sm100();
    const auto plan = validate_group_plan(offsets, experts, list_size, low_rank.size(0), lora_b.size(0));
    if (plan.groups <= 0 || plan.max_rows <= 0) {
        dS.zero_();
        grad_b.zero_();
        return;
    }

    auto dS_acc = torch::zeros(dS.sizes(), dS.options().dtype(torch::kFloat32));
    auto grad_b_acc = torch::zeros(grad_b.sizes(), grad_b.options().dtype(torch::kFloat32));
    const int64_t max_linear = plan.max_rows * grad_out_cpu.size(1);
    dim3 grid(blocks_for(max_linear), static_cast<unsigned int>(plan.groups));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lora_b_backward_kernel<<<grid, kThreads, 0, stream>>>(
        grad_out_cpu.data_ptr<at::BFloat16>(),
        low_rank.data_ptr<at::BFloat16>(),
        lora_b.data_ptr<at::BFloat16>(),
        dS_acc.data_ptr<float>(),
        grad_b_acc.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(),
        static_cast<int32_t>(plan.groups),
        static_cast<int32_t>(low_rank.size(0)),
        static_cast<int32_t>(grad_out_cpu.size(1)),
        static_cast<int32_t>(low_rank.size(1)),
        static_cast<float>(scale),
        max_linear);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cast_fp32_to_bf16(dS_acc, dS, stream);
    cast_fp32_to_bf16(grad_b_acc, grad_b, stream);
}

}  // namespace asym_gemm::exp_act_offload
