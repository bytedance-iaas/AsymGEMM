// asym_gemm/include/asym_gemm/impls/smxx_moe_utils.cuh
// Arch-agnostic device helpers shared by the SM80-family grouped MoE kernels
// (sm80_moe_gemm.cuh, sm89_fp8_moe_gemm.cuh, sm80_int8_asym_moe_gemm.cuh).
#pragma once

#include <cstdint>

#include <cutlass/numeric_types.h>      // cutlass::half_t, cutlass::bfloat16_t
#include <cutlass/array.h>              // cutlass::Array<T, N>
#include <cutlass/numeric_conversion.h> // cutlass::NumericArrayConverter (cvt.rn.bf16x2.f32)

#include <cute/tensor.hpp>

namespace asym_gemm {

// ──────────────────────────────────────────────────────────────────────────────
// Cooperative clear helper: zero a raw smem region using all threads
// ──────────────────────────────────────────────────────────────────────────────
template <int NUM_BYTES>
CUTE_DEVICE void clear_smem_region(char* ptr, int tidx, int num_threads) {
    static_assert(NUM_BYTES % 4 == 0, "NUM_BYTES must be 4-byte aligned");
    auto* iptr = reinterpret_cast<int32_t*>(ptr);
    constexpr int n_ints = NUM_BYTES / 4;
    CUTE_UNROLL
    for (int i = tidx; i < n_ints; i += num_threads)
        iptr[i] = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// moe_predicated_copy: vectorised tile copy with optional M-row predication.
//
// Port of FLASH_NAMESPACE::copy (flash-attention/csrc/flash_attn/src/utils.h):
// dispatches one TiledCopy atom per logical (m, k) sub-tile, gated by a per-row
// coordinate predicate.  When the underlying atom is 128-bit (UniversalCopy<uint128_t>
// or SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>), each call emits a single vectorised
// load/store instead of element-by-element scalar accesses.
//
// Template params:
//   Is_even_MN   — if true, all rows are valid; skip coordinate check (fast path).
//   Is_even_K    — if true, all K columns are valid; skip predicate_K check.
//   Clear_OOB_MN — if true, clear destination rows that are out-of-bounds.
//   Clear_OOB_K  — if true, clear destination columns that are out-of-bounds.
//
// Parameters:
//   tiled_copy   — TiledCopy object (must wrap a 128-bit atom for best perf).
//   S            — source tensor, rank-3: (Atom, MMA_M, MMA_N).
//   D            — destination tensor, same shape as S.
//   identity_MN  — coordinate tensor; identity_MN(0, m, 0) gives the M-row index.
//   predicate_K  — bool tensor of length size<2>(S); only consulted when !Is_even_K.
//   max_MN       — exclusive upper bound on valid M rows (ignored when Is_even_MN).
// ──────────────────────────────────────────────────────────────────────────────
template <bool Is_even_MN = true, bool Is_even_K = true,
          bool Clear_OOB_MN = false, bool Clear_OOB_K = false,
          typename TiledCopy,
          typename Engine0, typename Layout0,
          typename Engine1, typename Layout1,
          typename Engine2, typename Layout2,
          typename Engine3, typename Layout3>
CUTE_DEVICE void moe_predicated_copy(
    TiledCopy tiled_copy,
    cute::Tensor<Engine0, Layout0> const& S,
    cute::Tensor<Engine1, Layout1>&       D,
    cute::Tensor<Engine2, Layout2> const& identity_MN,
    cute::Tensor<Engine3, Layout3> const& predicate_K,
    int max_MN = 0)
{
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K),
                  "Clear_OOB_MN requires Clear_OOB_K");
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(S); ++m) {
        if (Is_even_MN || cute::get<0>(identity_MN(cute::_0{}, m, cute::_0{})) < max_MN) {
            CUTE_UNROLL
            for (int k = 0; k < cute::size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k))
                    cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
                else if (Clear_OOB_K)
                    cute::clear(D(cute::_, m, k));
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(cute::_, m, cute::_));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// moe_convert_type: vectorised register-fragment conversion via cutlass
// NumericArrayConverter.
//
// Port of FLASH_NAMESPACE::convert_type (flash-attention/csrc/flash_attn/src/utils.h).
// For FP32 → BF16 the inner 2-elem CUTLASS specialisation emits one
// `cvt.rn.bf16x2.f32` per pair (SM80+); for FP32 → FP16 it emits `cvt.rn.f16x2.f32`.
// The N>2 specialisation unrolls to N/2 SIMD calls.  Same rounding mode
// (round-to-nearest-even) and bit-identical output as the equivalent scalar
// `Element(static_cast<float>(x))` per-element loop.
//
// Requirements:
//   - tensor must be a register-residency tensor (e.g., from partition_fragment_C
//     or make_tensor<T>(layout)) whose data() returns a contiguous register array.
//   - tensor's element count must be known at compile time.
//
// Usage:
//   Tensor rO = moe_convert_type<ElementOut>(tSrO);
// ──────────────────────────────────────────────────────────────────────────────
template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto moe_convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(cute::size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(
        *reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return cute::make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}

}  // namespace asym_gemm
