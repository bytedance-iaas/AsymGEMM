#pragma once
// Stage 1 scaffolding — UE8M0 scale-factor + FP8x4 pack helpers used by
// the L1 SwiGLU+requant epilogue (Stage 2).
//
// DeepGEMM has these helpers in deep_gemm/common/math.cuh; porting the
// minimal subset that Stage 2 needs.  The code below is inline-only and
// has no AsymGEMM runtime dependencies.

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace asym_gemm {
namespace mega_moe_sf {

// Pack 4 FP32 values into a single uint32 of FP8 E4M3 bytes.
// Mirrors DeepGEMM's __nv_fp8x4_e4m3 path.
__device__ __forceinline__ uint32_t pack_fp8x4_e4m3(float4 v) {
    __nv_fp8x4_e4m3 packed = __nv_fp8x4_e4m3(v);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// UE8M0: 8-bit unsigned exponent; `sf` and its reciprocal computed from a
// per-row amax.  Returns both sf (for storing) and sf_inv (for scaling
// values down before the FP8 cast).
//
// NOTES:
//  - `amax` is assumed already finite and >= 0.
//  - On amax == 0, we return sf = 1 (and sf_inv = 1) as a safe neutral;
//    downstream FP8 cast of zero values is unchanged.
//  - FP8 E4M3 max representable magnitude is 448.0, matching DeepGEMM's
//    convention.
__device__ __forceinline__ void get_e4m3_sf_and_sf_inv(
        float2 amax, float2& sf, float2& sf_inv) {
    constexpr float kFP8Max = 448.0f;
    // Clamp zero to a tiny value so log2 is defined; the sf_inv path
    // compensates.
    auto one_sf = [&](float a, float& o_sf, float& o_sf_inv) {
        if (a <= 0.0f) {
            o_sf = 1.0f; o_sf_inv = 1.0f; return;
        }
        // Desired scale factor so that amax * sf_inv == FP8_max; use UE8M0
        // which rounds up to the next power of 2.
        float raw = a / kFP8Max;
        // Round exponent UP (UE8M0 is `ceil(log2(raw))`).  Reinterpret-cast
        // tricks suffice since the exponent already sits in bits 30..23.
        uint32_t bits = __float_as_uint(raw);
        uint32_t mant = bits & 0x007fffffu;
        uint32_t exp  = (bits >> 23) & 0xffu;
        // If mantissa != 0, round up exponent by 1 (and clear mantissa).
        if (mant != 0) exp += 1u;
        bits = exp << 23;
        float sf_pow2 = __uint_as_float(bits);
        o_sf = sf_pow2;
        o_sf_inv = 1.0f / sf_pow2;
    };
    one_sf(amax.x, sf.x, sf_inv.x);
    one_sf(amax.y, sf.y, sf_inv.y);
}

// Encode a float SF as a UE8M0 byte (just the exponent bits).
__device__ __forceinline__ uint8_t ue8m0_from_float(float sf) {
    if (sf <= 0.0f) return 0;
    uint32_t bits = __float_as_uint(sf);
    // Exponent is already biased with 127; UE8M0 uses the same bias.
    return static_cast<uint8_t>((bits >> 23) & 0xffu);
}

// Warp-reduce max.  NOTES: lanes < 4 hold the reduced float2 afterward
// (DeepGEMM's convention so STSM of 4 lanes writes 4 SF bytes).
__device__ __forceinline__ float2 warp_reduce_max_float2(float2 v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float2 o;
        o.x = __shfl_xor_sync(0xffffffffu, v.x, off);
        o.y = __shfl_xor_sync(0xffffffffu, v.y, off);
        v.x = fmaxf(v.x, o.x);
        v.y = fmaxf(v.y, o.y);
    }
    return v;
}

} // namespace mega_moe_sf
} // namespace asym_gemm
