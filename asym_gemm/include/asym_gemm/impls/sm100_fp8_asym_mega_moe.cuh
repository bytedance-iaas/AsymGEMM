#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

// Single fused CUDA kernel that mirrors DeepGEMM's `sm100_fp8_fp4_mega_moe`
// architectural pattern in one __global__ launch:
//     L1 GEMM  →  SwiGLU + FP8 re-quant  →  L2 GEMM  →  weighted combine
//
// Persistent CTAs iterate a two-phase scheduler; a counter-based grid sync
// (mega_grid_sync) sequences Phase 1 (L1) → Phase 2 (L2) → Phase 3 (combine).
// This first implementation uses CUDA-core (BF16) math rather than UMMA so
// the code is small enough to be audited in one PR; a follow-up can swap in
// the UMMA + TMA path from `sm100_fp8_asym_gemm_mega.cuh` once the Stage 3
// infrastructure is ready.
//
// From the Python side, this is ONE kernel call — identical to what
// DeepGEMM's `fp8_fp4_mega_moe` exposes.  The per-token cast to FP8 and
// per-block cast for weights happen in the caller (same as DeepGEMM).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <asym_gemm/common/utils.cuh>
#include <asym_gemm/common/mega_moe_workspace.cuh>
#include <asym_gemm/common/mega_moe_grid_sync.cuh>

namespace asym_gemm {

// Dequantize one FP8 E4M3 byte to FP32 via the CUDA header's fast path.
__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

// Dequantize a UE8M0 SF byte (from a packed uint32) to FP32.
__device__ __forceinline__ float ue8m0_byte_to_float(uint8_t b) {
    if (b == 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(b) << 23;
    return __uint_as_float(bits);
}

// Load and dequantize an FP8 row chunk (K elements) to BF16 with per-token SF.
// a_ptr: [H] FP8 row
// a_sf : [H/128] FP32 SF (per 128-element block along K)
// For each element at position k, effective value = fp8_to_float * a_sf[k/128]
__device__ __forceinline__ float deq_fp8_per_token(
        const __nv_fp8_e4m3* a_ptr,
        const float* a_sf,
        uint32_t k) {
    float v  = fp8_to_float(a_ptr[k]);
    float sf = a_sf[k / 128];
    return v * sf;
}

// Dequantize an FP8 weight element with per-(block_mn, block_k) SF.
// w_ptr: [N, K] FP8 matrix for ONE expert
// w_sf : [N/128, K/128] FP32 SF
__device__ __forceinline__ float deq_fp8_per_block(
        const __nv_fp8_e4m3* w_ptr,
        const float* w_sf,
        uint32_t n, uint32_t k,
        uint32_t K) {
    float v = fp8_to_float(w_ptr[n * K + k]);
    uint32_t sf_n_dim = /* ceil_div(N, 128) */ 0;  // not needed — SF access uses stride below
    (void)sf_n_dim;
    float sf = w_sf[(n / 128) * (K / 128) + (k / 128)];
    return v * sf;
}

// ---------------------------------------------------------------------------
// Phase 1: L1 GEMM — one row (token) × one N tile of size BLOCK_N at a time.
//   Each CTA computes `stride_m` rows × BLOCK_N columns per pass; walks work
//   items round-robin across SMs.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_N, uint32_t H, uint32_t TwoI>
__device__ __forceinline__ void phase1_l1_gemm(
        // Inputs
        const __nv_fp8_e4m3* a,     // [M_total, H] FP8
        const float*         a_sf,  // [M_total, H/128]
        const __nv_fp8_e4m3* l1_w,  // [E, TwoI, H]
        const float*         l1_w_sf, // [E, TwoI/128, H/128]
        const int32_t*       m_indices, // [M_total] expert id, -1 for padding
        // Output
        __nv_bfloat16*       l1_out, // [M_total, TwoI]
        // Config
        uint32_t M_total,
        uint32_t num_sms) {
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t nb_per_row = TwoI / BLOCK_N;
    const uint32_t total_tiles = M_total * nb_per_row;

    // Each SM processes total_tiles / num_sms tiles.
    for (uint32_t tile = sm_idx; tile < total_tiles; tile += num_sms) {
        const uint32_t m    = tile / nb_per_row;
        const uint32_t n_tile = tile % nb_per_row;
        const int32_t e = m_indices[m];
        if (e < 0) {
            // Padding row: zero the output tile.
            for (uint32_t n = tid; n < BLOCK_N; n += blockDim.x) {
                l1_out[m * TwoI + n_tile * BLOCK_N + n] = __float2bfloat16_rn(0.0f);
            }
            continue;
        }
        const __nv_fp8_e4m3* w_e = l1_w + e * TwoI * H;
        const float*         s_e = l1_w_sf + e * (TwoI / 128) * (H / 128);
        const __nv_fp8_e4m3* a_m = a + m * H;
        const float*         s_m = a_sf + m * (H / 128);

        // Each thread computes one output N column.
        for (uint32_t nl = tid; nl < BLOCK_N; nl += blockDim.x) {
            const uint32_t n = n_tile * BLOCK_N + nl;
            float acc = 0.0f;
            #pragma unroll 8
            for (uint32_t k = 0; k < H; ++k) {
                float a_val = deq_fp8_per_token(a_m, s_m, k);
                float w_val = deq_fp8_per_block(w_e, s_e, n, k, H);
                acc += a_val * w_val;
            }
            l1_out[m * TwoI + n] = __float2bfloat16_rn(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: SwiGLU + weight — per-row fuse and per-token re-quant to FP8
//   (writes into the workspace's L2 activation pool).
//   Gate = l1_out[:, 0:I];  Up = l1_out[:, I:2I]
// ---------------------------------------------------------------------------
template <uint32_t I>
__device__ __forceinline__ void phase2_swiglu_requant(
        const __nv_bfloat16* l1_out,   // [M_total, 2I]
        const int32_t*       m_indices,
        const float*         row_topk_w, // [M_total]
        float                clamp,
        bool                 fast_math,
        __nv_fp8_e4m3*       l2_acts,   // [M_total, I]  (output)
        float*               l2_sf,     // [M_total, I/128]
        uint32_t             M_total,
        uint32_t             num_sms) {
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t blk    = blockDim.x;

    // Each SM handles one row (token) at a time, round-robin.
    for (uint32_t m = sm_idx; m < M_total; m += num_sms) {
        if (m_indices[m] < 0) {
            // Padding: zero everything.
            for (uint32_t i = tid; i < I; i += blk)
                l2_acts[m * I + i] = __nv_fp8_e4m3(0.0f);
            for (uint32_t b = tid; b < I / 128; b += blk)
                l2_sf[m * (I / 128) + b] = 0.0f;
            continue;
        }
        const float w = row_topk_w[m];

        // Each thread processes stride-I output elements; compute SwiGLU then amax.
        // We do a two-pass within this row: first compute SwiGLU values into a temp,
        // then compute per-128-block amax, then re-quantize.
        // Use shared memory for the SwiGLU intermediate.
        extern __shared__ float smem_row[];  // size >= I
        for (uint32_t i = tid; i < I; i += blk) {
            float gate = __bfloat162float(l1_out[m * 2 * I + i]);
            float up   = __bfloat162float(l1_out[m * 2 * I + I + i]);
            // Gate: one-sided clamp to +clamp
            if (clamp > 0.0f) {
                gate = fminf(gate, clamp);
                up   = fmaxf(fminf(up, clamp), -clamp);
            }
            float neg_exp = fast_math ? __expf(-gate) : expf(-gate);
            float silu_g  = fast_math ? __fdividef(gate, 1.0f + neg_exp)
                                      : gate / (1.0f + neg_exp);
            smem_row[i] = silu_g * up * w;
        }
        __syncthreads();

        // For each 128-block, compute amax, UE8M0 SF, quantize.
        const uint32_t nb = I / 128;
        for (uint32_t b = tid; b < nb; b += blk) {
            float amax = 0.0f;
            #pragma unroll 4
            for (uint32_t j = 0; j < 128; ++j)
                amax = fmaxf(amax, fabsf(smem_row[b * 128 + j]));
            float sf = 1.0f;
            if (amax > 0.0f) {
                float raw = amax / 448.0f;
                uint32_t bits = __float_as_uint(raw);
                uint32_t mant = bits & 0x007fffffu;
                uint32_t exp  = (bits >> 23) & 0xffu;
                if (mant != 0) exp += 1u;
                sf = __uint_as_float(exp << 23);
            }
            l2_sf[m * nb + b] = sf;
            float inv_sf = (sf > 0.0f) ? (1.0f / sf) : 1.0f;
            #pragma unroll 4
            for (uint32_t j = 0; j < 128; ++j) {
                float v = smem_row[b * 128 + j] * inv_sf;
                l2_acts[m * I + b * 128 + j] = __nv_fp8_e4m3(v);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Phase 3: L2 GEMM — same structure as Phase 1 but reading from l2_acts / l2_sf.
//   Output: BF16 [M_total, H].  Stored directly into combine buffer at
//   [topk_k, orig_token_idx, :], deterministically — no atomics.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_N, uint32_t I, uint32_t H>
__device__ __forceinline__ void phase3_l2_gemm(
        const __nv_fp8_e4m3* l2_acts,   // [M_total, I]
        const float*         l2_sf,     // [M_total, I/128]
        const __nv_fp8_e4m3* l2_w,      // [E, H, I]
        const float*         l2_w_sf,   // [E, H/128, I/128]
        const int32_t*       m_indices,
        const int32_t*       topk_map,  // [M_total, 2] (orig_token_idx, topk_k)
        __nv_bfloat16*       combine_buf, // [num_topk, num_tokens, H]
        uint32_t M_total,
        uint32_t num_topk,
        uint32_t num_tokens,
        uint32_t num_sms) {
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t nb_per_row = H / BLOCK_N;
    const uint32_t total_tiles = M_total * nb_per_row;

    for (uint32_t tile = sm_idx; tile < total_tiles; tile += num_sms) {
        const uint32_t m      = tile / nb_per_row;
        const uint32_t n_tile = tile % nb_per_row;
        const int32_t e = m_indices[m];
        const int32_t orig = topk_map[m * 2 + 0];
        const int32_t topk_k = topk_map[m * 2 + 1];
        if (e < 0 || orig < 0) continue;

        const __nv_fp8_e4m3* w_e = l2_w + e * H * I;
        const float*         s_e = l2_w_sf + e * (H / 128) * (I / 128);
        const __nv_fp8_e4m3* a_m = l2_acts + m * I;
        const float*         s_m = l2_sf + m * (I / 128);

        for (uint32_t nl = tid; nl < BLOCK_N; nl += blockDim.x) {
            const uint32_t n = n_tile * BLOCK_N + nl;
            float acc = 0.0f;
            #pragma unroll 8
            for (uint32_t k = 0; k < I; ++k) {
                float a_val = deq_fp8_per_token(a_m, s_m, k);
                float w_val = deq_fp8_per_block(w_e, s_e, n, k, I);
                acc += a_val * w_val;
            }
            // Write into combine buffer at deterministic slot — no atomics.
            const uint64_t dst = (uint64_t)topk_k * num_tokens * H
                               + (uint64_t)orig * H + n;
            combine_buf[dst] = __float2bfloat16_rn(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Combine reduce — for each output token, sum over topk slots.
//   This is the only place `y` is written; each CTA handles a subset of
//   (token × N tile) work.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_N, uint32_t H>
__device__ __forceinline__ void phase4_combine_reduce(
        const __nv_bfloat16* combine_buf,  // [num_topk, num_tokens, H]
        __nv_bfloat16*       y,            // [num_tokens, H]
        uint32_t             num_tokens,
        uint32_t             num_topk,
        uint32_t             num_sms) {
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t nb_per_row = H / BLOCK_N;
    const uint32_t total_tiles = num_tokens * nb_per_row;

    for (uint32_t tile = sm_idx; tile < total_tiles; tile += num_sms) {
        const uint32_t t      = tile / nb_per_row;
        const uint32_t n_tile = tile % nb_per_row;

        for (uint32_t nl = tid; nl < BLOCK_N; nl += blockDim.x) {
            const uint32_t n = n_tile * BLOCK_N + nl;
            float acc = 0.0f;
            for (uint32_t k = 0; k < num_topk; ++k) {
                const uint64_t src = (uint64_t)k * num_tokens * H
                                   + (uint64_t)t * H + n;
                acc += __bfloat162float(combine_buf[src]);
            }
            y[t * H + n] = __float2bfloat16_rn(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// One-shot fused MoE kernel.
// Mirrors DeepGEMM's `fp8_fp4_mega_moe` design: persistent CTAs iterating
// through (phase, ...) work with grid-sync barriers between phases.
// ---------------------------------------------------------------------------
template <uint32_t H, uint32_t I,
          uint32_t BLOCK_N_L1, uint32_t BLOCK_N_L2,
          uint32_t kNumThreads,
          uint32_t kNumSMs,
          bool     kFastMath>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_fp8_asym_mega_moe_impl(
        // Inputs
        const __nv_fp8_e4m3* __restrict__ a,          // [M_total, H]
        const float*         __restrict__ a_sf,       // [M_total, H/128]
        const __nv_fp8_e4m3* __restrict__ l1_w,       // [E, 2I, H]
        const float*         __restrict__ l1_w_sf,    // [E, 2I/128, H/128]
        const __nv_fp8_e4m3* __restrict__ l2_w,       // [E, H, I]
        const float*         __restrict__ l2_w_sf,    // [E, H/128, I/128]
        const int32_t*       __restrict__ m_indices,  // [M_total]
        const int32_t*       __restrict__ topk_map,   // [M_total, 2]
        const float*         __restrict__ row_topk_w, // [M_total]
        // Workspace (intermediate FP8 acts + FP32 SFs + combine buffer)
        __nv_bfloat16*       __restrict__ l1_out,     // [M_total, 2I]
        __nv_fp8_e4m3*       __restrict__ l2_acts,    // [M_total, I]
        float*               __restrict__ l2_sf,      // [M_total, I/128]
        __nv_bfloat16*       __restrict__ combine_buf, // [num_topk, num_tokens, H]
        // Output
        __nv_bfloat16*       __restrict__ y,          // [num_tokens, H]
        // Grid-sync counters (4 x uint32 in workspace)
        uint32_t*            __restrict__ grid_sync_ctrs,
        // Scalar config
        uint32_t M_total,
        uint32_t num_tokens,
        uint32_t num_topk,
        float    clamp) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)

    // ---------- Phase 1: L1 GEMM ----------
    phase1_l1_gemm<BLOCK_N_L1, H, 2 * I>(
        a, a_sf, l1_w, l1_w_sf, m_indices, l1_out, M_total, kNumSMs);

    // Grid sync between phases: every SM must see every other SM's writes.
    // Use the workspace's counter[0].
    MegaMoEWorkspace ws_stub{};
    ws_stub.base = grid_sync_ctrs;        // pretend base points at counters
    ws_stub.off_grid_sync = 0;
    mega_grid_sync<kNumSMs, 0>(ws_stub, blockIdx.x, threadIdx.x);

    // ---------- Phase 2: SwiGLU + weight + FP8 requant ----------
    phase2_swiglu_requant<I>(
        l1_out, m_indices, row_topk_w, clamp, kFastMath,
        l2_acts, l2_sf, M_total, kNumSMs);
    mega_grid_sync<kNumSMs, 1>(ws_stub, blockIdx.x, threadIdx.x);

    // ---------- Phase 3: L2 GEMM + scatter to combine buffer ----------
    phase3_l2_gemm<BLOCK_N_L2, I, H>(
        l2_acts, l2_sf, l2_w, l2_w_sf, m_indices,
        topk_map, combine_buf, M_total, num_topk, num_tokens, kNumSMs);
    mega_grid_sync<kNumSMs, 2>(ws_stub, blockIdx.x, threadIdx.x);

    // ---------- Phase 4: Combine reduce ----------
    phase4_combine_reduce<BLOCK_N_L2, H>(
        combine_buf, y, num_tokens, num_topk, kNumSMs);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "sm100_fp8_asym_mega_moe requires sm_90 or higher");
#endif
}

} // namespace asym_gemm

#pragma clang diagnostic pop
