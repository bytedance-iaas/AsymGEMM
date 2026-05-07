// C-linkage launcher for the fused single-launch MoE kernel.
// Compiled with nvcc; python_api.cpp calls these via a plain function pointer.

#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>

#include <asym_gemm/impls/sm100_fp8_asym_mega_moe.cuh>

using namespace asym_gemm;

namespace {

template <uint32_t H, uint32_t I,
          uint32_t BLOCK_N_L1, uint32_t BLOCK_N_L2,
          uint32_t kNumThreads, uint32_t kNumSMs, bool kFastMath>
int do_launch(
        const void* a_fp8, const void* a_sf,
        const void* l1_w, const void* l1_w_sf,
        const void* l2_w, const void* l2_w_sf,
        const void* m_indices,
        const void* topk_map,
        const void* row_topk_w,
        void* l1_out,
        void* l2_acts,
        void* l2_sf,
        void* combine_buf,
        void* y,
        void* grid_sync_ctrs,
        uint32_t M_total,
        uint32_t num_tokens,
        uint32_t num_topk,
        float clamp,
        cudaStream_t stream) {
    const void* kernel = reinterpret_cast<const void*>(
        &sm100_fp8_asym_mega_moe_impl<H, I, BLOCK_N_L1, BLOCK_N_L2,
                                       kNumThreads, kNumSMs, kFastMath>);

    // Zero grid-sync counters (4 uint32s) before launch
    cudaMemsetAsync(grid_sync_ctrs, 0, 4 * sizeof(uint32_t), stream);

    const auto* p_a      = static_cast<const __nv_fp8_e4m3*>(a_fp8);
    const auto* p_asf    = static_cast<const float*>(a_sf);
    const auto* p_l1w    = static_cast<const __nv_fp8_e4m3*>(l1_w);
    const auto* p_l1wsf  = static_cast<const float*>(l1_w_sf);
    const auto* p_l2w    = static_cast<const __nv_fp8_e4m3*>(l2_w);
    const auto* p_l2wsf  = static_cast<const float*>(l2_w_sf);
    const auto* p_mi     = static_cast<const int32_t*>(m_indices);
    const auto* p_tm     = static_cast<const int32_t*>(topk_map);
    const auto* p_rw     = static_cast<const float*>(row_topk_w);
    auto*       p_l1o    = static_cast<__nv_bfloat16*>(l1_out);
    auto*       p_l2a    = static_cast<__nv_fp8_e4m3*>(l2_acts);
    auto*       p_l2sf   = static_cast<float*>(l2_sf);
    auto*       p_cb     = static_cast<__nv_bfloat16*>(combine_buf);
    auto*       p_y      = static_cast<__nv_bfloat16*>(y);
    auto*       p_ctr    = static_cast<uint32_t*>(grid_sync_ctrs);

    void* args[] = {
        (void*)&p_a, (void*)&p_asf,
        (void*)&p_l1w, (void*)&p_l1wsf,
        (void*)&p_l2w, (void*)&p_l2wsf,
        (void*)&p_mi, (void*)&p_tm, (void*)&p_rw,
        (void*)&p_l1o, (void*)&p_l2a, (void*)&p_l2sf,
        (void*)&p_cb, (void*)&p_y, (void*)&p_ctr,
        (void*)&M_total, (void*)&num_tokens, (void*)&num_topk,
        (void*)&clamp,
    };
    const size_t smem_bytes = I * sizeof(float);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }
    dim3 grid(kNumSMs, 1, 1);
    dim3 block(kNumThreads, 1, 1);
    auto err = cudaLaunchKernel(kernel, grid, block, args, smem_bytes, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[mega_moe] cudaLaunchKernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    auto sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[mega_moe] kernel runtime error: %s\n", cudaGetErrorString(sync_err));
        return -3;
    }
    return 0;
}

} // namespace

extern "C" int asym_mega_moe_launch(
        const void* a_fp8, const void* a_sf,
        const void* l1_w, const void* l1_w_sf,
        const void* l2_w, const void* l2_w_sf,
        const void* m_indices,
        const void* topk_map,
        const void* row_topk_w,
        void* l1_out,
        void* l2_acts,
        void* l2_sf,
        void* combine_buf,
        void* y,
        void* grid_sync_ctrs,
        uint32_t M_total,
        uint32_t num_tokens,
        uint32_t num_topk,
        uint32_t hidden,
        uint32_t intermediate,
        float clamp,
        int fast_math,
        cudaStream_t stream) {
    constexpr uint32_t kBlockN  = 64;
    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kSMs     = 132;  // conservative — GB200 has 132 SMs

    #define TRY_HI(H, I) \
        if (hidden == (H) && intermediate == (I)) { \
            if (fast_math) \
                return do_launch<(H), (I), kBlockN, kBlockN, kThreads, kSMs, true>( \
                    a_fp8, a_sf, l1_w, l1_w_sf, l2_w, l2_w_sf, \
                    m_indices, topk_map, row_topk_w, \
                    l1_out, l2_acts, l2_sf, combine_buf, y, grid_sync_ctrs, \
                    M_total, num_tokens, num_topk, clamp, stream); \
            return do_launch<(H), (I), kBlockN, kBlockN, kThreads, kSMs, false>( \
                a_fp8, a_sf, l1_w, l1_w_sf, l2_w, l2_w_sf, \
                m_indices, topk_map, row_topk_w, \
                l1_out, l2_acts, l2_sf, combine_buf, y, grid_sync_ctrs, \
                M_total, num_tokens, num_topk, clamp, stream); \
        }

    TRY_HI(256,  128)
    TRY_HI(512,  256)
    TRY_HI(1024, 512)
    #undef TRY_HI

    fprintf(stderr, "[mega_moe] Unsupported (hidden, intermediate) = (%u, %u). "
                    "Add it to the dispatch table in csrc/mega_moe_launch.cu.\n",
            hidden, intermediate);
    return -2;
}
