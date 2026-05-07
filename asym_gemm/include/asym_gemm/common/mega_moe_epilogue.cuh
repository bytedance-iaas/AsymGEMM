#pragma once
// Stage 1 scaffolding — MoE epilogue policy tag types.
//
// The mega kernel (`sm100_fp8_asym_gemm_mega.cuh`) takes an
// `epilogue_type_t` template parameter that is currently unused.  This
// header defines the three policy tags that will drive the future MoE
// fusion (asym_moe_update.md §1.10):
//
//   - EpilogueStoreBF16          :  today's default; TMEM -> BF16 -> TMA store
//   - EpilogueSwiGLUFP8Requant   :  Stage-2 L1 epilogue (SwiGLU + per-token FP8
//                                   re-quant, writes into L2 act pool)
//   - EpilogueCombineScatter     :  Stage-3 L2 epilogue (BF16 scatter into
//                                   per-topk combine buffer)
//
// Stages 2-4 will add the actual CUDA bodies that implement each policy.
// Stage 1 (this file) is header-only and does not change kernel behavior.

#include <asym_gemm/common/mega_moe_scheduler.cuh>  // BlockPhase

namespace asym_gemm {

// The default (and only) policy the existing mega GEMM uses.  Signals
// "no MoE fusion; just store BF16 to tensor_map_cd".
struct EpilogueStoreBF16 {
    static constexpr bool        kIsMegaMoE = false;
    static constexpr BlockPhase  kPhase     = BlockPhase::None;

    // Matches existing `epilogue_type_t::apply_index_n` hook in 1d1d path.
    // For the plain store, the N index is unchanged.
    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }
};

// Stage-2 placeholder.  Carries runtime config for SwiGLU + FP8 requant.
//
// Static config (known at JIT compile time):
//   - kFastMath selects __expf vs expf in the SwiGLU sigmoid
//   - kClampValue is the symmetric activation clamp ( <= 0 disables it )
//
// The kernel body (Stage 2) will branch on `kIsMegaMoE` and
// `kPhase == Linear1`, then invoke the (to-be-added) helper
//   sm100::epilogue_swiglu_fp8(<tile>, clamp=kClampValue, fast_math=kFastMath)
template <bool kFastMathT = true>
struct EpilogueSwiGLUFP8Requant {
    static constexpr bool        kIsMegaMoE = true;
    static constexpr BlockPhase  kPhase     = BlockPhase::Linear1;
    static constexpr bool        kFastMath  = kFastMathT;

    // Runtime clamp bound; 0 disables.  Kept as a runtime float (not a non-type
    // template arg) so the same specialization can be reused at different
    // clamps without re-JIT.
    float kClampValue;

    __device__ __forceinline__ EpilogueSwiGLUFP8Requant(float clamp = 0.0f)
        : kClampValue(clamp) {}

    // N-index hook, unused in the SwiGLU case (it writes to a pool, not CD)
    // but kept for interface parity with EpilogueStoreBF16.
    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }
};

// Stage-3 placeholder.  Signals "L2 epilogue: BF16 values -> combine scatter".
struct EpilogueCombineScatter {
    static constexpr bool        kIsMegaMoE = true;
    static constexpr BlockPhase  kPhase     = BlockPhase::Linear2;

    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }
};

// Convenience concept check (used via SFINAE in kernel bodies).
template <typename T> struct is_mega_moe_epilogue { static constexpr bool value = false; };
template <bool F>     struct is_mega_moe_epilogue<EpilogueSwiGLUFP8Requant<F>> { static constexpr bool value = true; };
template <>           struct is_mega_moe_epilogue<EpilogueCombineScatter>       { static constexpr bool value = true; };

} // namespace asym_gemm
