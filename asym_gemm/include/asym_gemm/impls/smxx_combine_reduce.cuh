#pragma once
#include <cuda_bf16.h>
#include <asym_gemm/common/mega_moe_workspace.cuh>

namespace asym_gemm {

// Combine reduce: sum combine_buffer[topk_k][token][:] → output[token][:]
// combine_buffer layout: [num_topk, num_tokens, hidden] BF16
// output layout: [num_tokens, hidden] BF16
// Topk weights were already folded into the L1 SwiGLU epilogue, so this is a plain sum.
__global__ void mega_moe_combine_reduce_impl(
        const void* __restrict__ combine_buffer,
        void* __restrict__ output,
        uint32_t num_tokens,
        uint32_t num_topk,
        uint32_t hidden) {
    const uint32_t token_idx  = blockIdx.x;
    const uint32_t n_idx_base = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
    if (token_idx >= num_tokens || n_idx_base + 1 >= hidden) return;

    float2 acc = {0.0f, 0.0f};
    const auto* cb = reinterpret_cast<const nv_bfloat16*>(combine_buffer);
    for (uint32_t k = 0; k < num_topk; ++k) {
        const auto* row = cb + (static_cast<uint64_t>(k) * num_tokens + token_idx) * hidden + n_idx_base;
        uint32_t packed;
        asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(packed) : "l"(row));
        float2 vals = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
        acc.x += vals.x;
        acc.y += vals.y;
    }

    auto* out_row = reinterpret_cast<nv_bfloat16*>(output) + static_cast<uint64_t>(token_idx) * hidden + n_idx_base;
    __nv_bfloat162 result = __float22bfloat162_rn(acc);
    *reinterpret_cast<__nv_bfloat162*>(out_row) = result;
}

// Launch helper: called from Python C++ binding
// gridDim = (num_tokens, ceil_div(hidden/2, blockDim.x))
// blockDim = 128
inline void launch_combine_reduce(
        const void* combine_buffer,
        void* output,
        uint32_t num_tokens,
        uint32_t num_topk,
        uint32_t hidden,
        cudaStream_t stream = 0) {
    constexpr uint32_t kBlockDim = 128;
    const uint32_t grid_y = (hidden / 2 + kBlockDim - 1) / kBlockDim;
    mega_moe_combine_reduce_impl<<<dim3(num_tokens, grid_y), kBlockDim, 0, stream>>>(
        combine_buffer, output, num_tokens, num_topk, hidden);
}

} // namespace asym_gemm
