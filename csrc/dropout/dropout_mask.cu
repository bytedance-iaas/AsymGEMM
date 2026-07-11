// Copyright (c) 2026.

#include "../apis/dropout.hpp"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace asym_gemm::dropout {
namespace {

constexpr int kThreads = 256;

__global__ void pack_bool_mask_2d_kernel(const bool* __restrict__ mask, uint8_t* __restrict__ packed, int64_t m, int64_t k, int64_t packed_k) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * packed_k;
    if (linear >= total) return;
    const int64_t row = linear / packed_k;
    const int64_t byte_col = linear - row * packed_k;
    const int64_t k_base = byte_col * 8;
    uint8_t byte = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        const int64_t col = k_base + bit;
        if (col < k && mask[row * k + col]) {
            byte |= static_cast<uint8_t>(1u << bit);
        }
    }
    packed[linear] = byte;
}

__global__ void unpack_bool_mask_2d_kernel(const uint8_t* __restrict__ packed, bool* __restrict__ mask, int64_t m, int64_t k, int64_t packed_k) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * k;
    if (linear >= total) return;
    const int64_t row = linear / k;
    const int64_t col = linear - row * k;
    const uint8_t byte = packed[row * packed_k + (col >> 3)];
    mask[linear] = ((byte >> (col & 7)) & 1u) != 0;
}

template <typename scalar_t>
__global__ void apply_packed_dropout_elementwise_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t k,
    int64_t packed_k,
    float scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * k;
    if (linear >= total) return;
    const int64_t row = linear / k;
    const int64_t col = linear - row * k;
    const uint8_t byte = packed[row * packed_k + (col >> 3)];
    const bool keep = ((byte >> (col & 7)) & 1u) != 0;
    out[linear] = keep ? static_cast<scalar_t>(static_cast<float>(x[linear]) * scale) : static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void apply_packed_dropout_byte_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    scalar_t* __restrict__ out,
    int64_t m,
    int64_t k,
    int64_t packed_k,
    float scale) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * packed_k;
    if (linear >= total) return;
    const int64_t row = linear / packed_k;
    const int64_t byte_col = linear - row * packed_k;
    const int64_t base_col = byte_col * 8;
    const int64_t base = row * k + base_col;
    const uint8_t byte = packed[linear];

#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        const int64_t col = base_col + bit;
        if (col >= k) return;
        const bool keep = ((byte >> bit) & 1u) != 0;
        out[base + bit] = keep ? static_cast<scalar_t>(static_cast<float>(x[base + bit]) * scale) : static_cast<scalar_t>(0);
    }
}

void check_cuda_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}

void check_dropout_p(double dropout_p) {
    TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must satisfy 0.0 <= p < 1.0, got ", dropout_p);
}

}  // namespace

torch::Tensor pack_bool_mask_2d(const torch::Tensor& mask_bool) {
    check_cuda_2d(mask_bool, "mask_bool");
    TORCH_CHECK(mask_bool.scalar_type() == torch::kBool, "mask_bool must have dtype torch.bool");
    auto mask = mask_bool.contiguous();
    const int64_t m = mask.size(0);
    const int64_t k = mask.size(1);
    const int64_t packed_k = (k + 7) / 8;
    auto out = torch::empty({m, packed_k}, mask.options().dtype(torch::kUInt8));
    const int64_t total = m * packed_k;
    if (total == 0) return out;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    pack_bool_mask_2d_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        mask.data_ptr<bool>(),
        out.data_ptr<uint8_t>(),
        m,
        k,
        packed_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor unpack_bool_mask_2d(const torch::Tensor& mask_packed, int64_t width) {
    check_cuda_2d(mask_packed, "mask_packed");
    TORCH_CHECK(mask_packed.scalar_type() == torch::kUInt8, "mask_packed must have dtype torch.uint8");
    TORCH_CHECK(width >= 0, "width must be non-negative, got ", width);
    const int64_t m = mask_packed.size(0);
    const int64_t packed_k = mask_packed.size(1);
    TORCH_CHECK(packed_k == (width + 7) / 8, "mask_packed width mismatch: got ", packed_k, ", expected ", (width + 7) / 8);
    auto packed = mask_packed.contiguous();
    auto out = torch::empty({m, width}, mask_packed.options().dtype(torch::kBool));
    const int64_t total = m * width;
    if (total == 0) return out;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    unpack_bool_mask_2d_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        out.data_ptr<bool>(),
        m,
        width,
        packed_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor apply_packed_dropout_(torch::Tensor x, const torch::Tensor& mask_packed, double dropout_p) {
    check_cuda_2d(x, "x");
    check_cuda_2d(mask_packed, "mask_packed");
    check_dropout_p(dropout_p);
    TORCH_CHECK(mask_packed.scalar_type() == torch::kUInt8, "mask_packed must have dtype torch.uint8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous for apply_packed_dropout_");
    auto packed = mask_packed.contiguous();
    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t packed_k = (k + 7) / 8;
    TORCH_CHECK(packed.size(0) == m, "mask_packed row mismatch: got ", packed.size(0), ", expected ", m);
    TORCH_CHECK(packed.size(1) == packed_k, "mask_packed width mismatch: got ", packed.size(1), ", expected ", packed_k);
    const int64_t total = m * k;
    if (total == 0 || dropout_p == 0.0) return x;
    const float scale = static_cast<float>(1.0 / (1.0 - dropout_p));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "apply_packed_dropout", [&] {
        if (k >= 8) {
            const int64_t packed_total = m * packed_k;
            const int blocks = static_cast<int>((packed_total + kThreads - 1) / kThreads);
            apply_packed_dropout_byte_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                x.data_ptr<scalar_t>(),
                m,
                k,
                packed_k,
                scale);
        } else {
            const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
            apply_packed_dropout_elementwise_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                x.data_ptr<scalar_t>(),
                m,
                k,
                packed_k,
                scale);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x;
}

torch::Tensor apply_packed_dropout(const torch::Tensor& x, const torch::Tensor& mask_packed, double dropout_p) {
    auto out = x.contiguous().clone();
    return apply_packed_dropout_(out, mask_packed, dropout_p);
}

}  // namespace asym_gemm::dropout
