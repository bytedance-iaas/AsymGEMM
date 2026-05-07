#pragma once
// Stage 2 of asym_moe_update.md — L1 epilogue: SwiGLU + FP8 re-quant.
//
// Takes 8 FP32 values (gate/up interleaved) + a per-row topk weight and a
// clamp bound.  Produces:
//   - 4 FP8-packed uint32s per lane (stored via STSM into smem_cd)
//   - 2 UE8M0 SF bytes per lane (written to l2_sf_buffer global)
//
// This matches DeepGEMM's per-atom body (`mega_moe.md §8 Stage C`) but as a
// pure function independent of MMA / TMEM: callers pass in already-loaded
// FP32 values.  The kernel integration (Stage 3) is responsible for the
// TMEM_LOAD and the STSM/TMA_STORE framing around these calls.
//
// Gate/up interleave layout (matches DeepGEMM's SM100_TMEM_LOAD_16dp256b1x
// delivery to each lane):
//   values[0] = gate pair 0 (.x, .y)   values[2] = up pair 0
//   values[1] = gate pair 1 (.x, .y)   values[3] = up pair 1
//   values[4] = gate pair 2             values[6] = up pair 2
//   values[5] = gate pair 3             values[7] = up pair 3

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <asym_gemm/common/mega_moe_sf_math.cuh>
#include <asym_gemm/common/utils.cuh>

namespace asym_gemm {
namespace mega_moe_epi {

// One SwiGLU pair (gate+up), clamp-then-SiLU-then-mul-then-weight.
// Inputs are float2 values; output is float2.
__device__ __forceinline__ float2 swiglu_pair(
        float2 gate, float2 up, float weight, float clamp, bool fast_math) {
    auto bf16_gate = __float22bfloat162_rn(gate);
    auto bf16_up   = __float22bfloat162_rn(up);
    if (clamp > 0.0f) {
        __nv_bfloat162 clamp_pos = __floats2bfloat162_rn(clamp, clamp);
        __nv_bfloat162 clamp_neg = __floats2bfloat162_rn(-clamp, -clamp);
        bf16_gate = __hmin2(bf16_gate, clamp_pos);
        bf16_up   = __hmax2(bf16_up,   clamp_neg);
        bf16_up   = __hmin2(bf16_up,   clamp_pos);
    }
    auto gate_f = __bfloat1622float2(bf16_gate);
    auto up_f   = __bfloat1622float2(bf16_up);

    // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    float2 neg_exp;
    if (fast_math) {
        neg_exp = {__expf(-gate_f.x), __expf(-gate_f.y)};
    } else {
        neg_exp = {expf(-gate_f.x), expf(-gate_f.y)};
    }
    float2 denom = {1.0f + neg_exp.x, 1.0f + neg_exp.y};
    float2 silu_gate;
    if (fast_math) {
        // __fdividef is fast (1 ULP) on recent toolchains
        silu_gate = {__fdividef(gate_f.x, denom.x),
                     __fdividef(gate_f.y, denom.y)};
    } else {
        silu_gate = {gate_f.x / denom.x, gate_f.y / denom.y};
    }
    float2 result = {silu_gate.x * up_f.x * weight,
                     silu_gate.y * up_f.y * weight};
    return result;
}

// Process ONE "atom" (8 FP32 values → 2 SwiGLU float2s → 1 packed uint32
// of FP8 + 2 UE8M0 SF bytes).  The `out_amax` output holds this lane's
// amax contribution, used for cross-warp amax reduction by the caller.
//
// Returns 2 SwiGLU float2 results; caller handles amax reduction and
// FP8 packing in a second pass (see swiglu_atom_finalize_fp8 below).
//
// Gate/up pairing matches SM100_TMEM_LOAD_16dp256b1x delivery layout:
//   k=0: gate=(vf[0],vf[1]), up=(vf[2],vf[3])  → out_swiglu[0]
//   k=1: gate=(vf[4],vf[5]), up=(vf[6],vf[7])  → out_swiglu[1]
__device__ __forceinline__
void swiglu_atom_compute(
        const uint32_t values[8],
        float weight,
        float clamp,
        bool  fast_math,
        float2 out_swiglu[2],
        float2& out_amax) {
    const float* vf = reinterpret_cast<const float*>(values);
    out_swiglu[0] = swiglu_pair(make_float2(vf[0], vf[1]),
                                make_float2(vf[2], vf[3]), weight, clamp, fast_math);
    out_swiglu[1] = swiglu_pair(make_float2(vf[4], vf[5]),
                                make_float2(vf[6], vf[7]), weight, clamp, fast_math);
    float amax_x = fmaxf(fabsf(out_swiglu[0].x), fabsf(out_swiglu[1].x));
    float amax_y = fmaxf(fabsf(out_swiglu[0].y), fabsf(out_swiglu[1].y));
    out_amax = float2{amax_x, amax_y};
}

// Finalize ONE atom: given 2 swiglu float2s and a reduced per-atom amax,
// compute UE8M0 SF, scale values, and pack to 4 FP8 bytes (uint32).
// Returns the packed uint32; also writes UE8M0 SF bytes into out_sf_x, out_sf_y
// (one per float2.x / .y).
__device__ __forceinline__ uint32_t swiglu_atom_finalize_fp8(
        const float2 swiglu[2],
        float2 amax,
        uint8_t& out_sf_x,
        uint8_t& out_sf_y) {
    float2 sf, sf_inv;
    mega_moe_sf::get_e4m3_sf_and_sf_inv(amax, sf, sf_inv);
    out_sf_x = mega_moe_sf::ue8m0_from_float(sf.x);
    out_sf_y = mega_moe_sf::ue8m0_from_float(sf.y);

    float4 scaled;
    scaled.x = swiglu[0].x * sf_inv.x;
    scaled.y = swiglu[0].y * sf_inv.y;
    scaled.z = swiglu[1].x * sf_inv.x;
    scaled.w = swiglu[1].y * sf_inv.y;
    return mega_moe_sf::pack_fp8x4_e4m3(scaled);
}

// Convenience wrapper for testing: full per-atom pipeline on synthetic inputs.
// `values` : 8 FP32 values (gate/up interleaved per SM100_TMEM_LOAD_16dp256b1x)
// Returns packed FP8 and writes SF bytes.
__device__ __forceinline__ uint32_t swiglu_atom_full(
        const uint32_t values[8],
        float weight,
        float clamp,
        bool fast_math,
        uint8_t& out_sf_x,
        uint8_t& out_sf_y) {
    float2 swiglu[2];
    float2 amax;
    swiglu_atom_compute(values, weight, clamp, fast_math, swiglu, amax);
    // For standalone testing, we skip the cross-warp amax reduction and use
    // the per-lane amax directly.  The kernel-integrated version (Stage 3)
    // will replace this with a warp_reduce<4, true>(amax, ReduceMax) and
    // cross-warp rendezvous through smem_amax_reduction.
    return swiglu_atom_finalize_fp8(swiglu, amax, out_sf_x, out_sf_y);
}

} // namespace mega_moe_epi
} // namespace asym_gemm
