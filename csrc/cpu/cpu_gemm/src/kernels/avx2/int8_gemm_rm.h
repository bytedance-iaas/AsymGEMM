/*
 * AVX2 INT8 GEMM (row-major B) — last-resort fallback for hosts with
 * neither AMX-INT8 nor AVX-512-VNNI (e.g. AMD Zen 2/3, Intel pre-Ice
 * Lake). Same shared-byte contract as kernels/amx/int8_gemm_rm.h and
 * kernels/avx512/int8_gemm_rm.h:
 *
 *   B [N, K]     row-major int8     (the canonical pinned weight slab)
 *   B scales [N] per-row float32   (per-channel)
 *   A [M, K]     row-major BF16
 *   C [M, N]     row-major float32
 *
 * Implementation:
 *   AVX2 has no int8 dot-product instruction; the widest primitive is
 *   vpmaddubsw (uint8 × int8 → saturating int16 pair-sum). We make the
 *   saturation impossible with the sign trick:
 *     a × b = |a| × sign(a)·b
 *   A is quantized symmetrically to [-127, 127]; per K-chunk we take
 *   |a| via vpsignb(a, a) (unsigned operand, <= 127) and fold a's sign
 *   into b via vpsignb(b, a) (signed operand). B is produced by the
 *   AsymGEMM quantizers, which clamp to [-127, 127], so each int16
 *   pair-sum is bounded by 2 × 127 × 127 = 32258 < 32767 — no
 *   saturation. vpmaddwd against ones then widens to int32 lanes.
 *
 *   NOTE: a B value of -128 would break the bound (and vpsignb's
 *   negation of -128 wraps). All in-tree quantizers clamp to -127; the
 *   contract of this kernel requires B ∈ [-127, 127].
 *
 * Because A stays signed (no offset-binary rebias like the VNNI
 * kernel), no per-column B-sum correction is needed at unpack time.
 *
 * This kernel is correctness-first, not peak-optimal — it exists so the
 * unified CPU+GPU MoE path has a working CPU bucket on AVX2-only hosts.
 */
#ifndef CPU_GEMM_KERNELS_AVX2_INT8_GEMM_RM_H
#define CPU_GEMM_KERNELS_AVX2_INT8_GEMM_RM_H

#if defined(CPU_GEMM_HAS_AVX2_INT8)

#include <cstddef>
#include <cstdint>

#include "cpu_gemm/types.h"

namespace cpu_gemm::kernels::avx2 {

/* Tile parameters chosen for AVX2 ymm register usage:
 *   M_STEP = 4   (4 ymm accumulators per N-tile)
 *   N_STEP = 8   (1 ymm = 8 int32 output lanes)
 *   K_STEP = 4   (one vpmaddubsw+vpmaddwd consumes 4 K-values per lane)
 * N_BLOCK = 64   (8 ymm worth of output cols per inner work unit) */
struct Int8RmTraits {
  static constexpr int M_STEP  = 4;
  static constexpr int N_STEP  = 8;
  static constexpr int K_STEP  = 4;
  static constexpr int N_BLOCK = 64;
};

inline int int8_rm_pad_up(int x, int step) {
  return ((x + step - 1) / step) * step;
}

/* Scratch layout — matches the AMX / AVX-512 kernels' shape so the
 * Int8RmBackend table abstracts uniformly over all three. */
struct Int8RmScratch {
  std::size_t bytes_a;   /* int8 A + per-row scales */
  std::size_t bytes_c;   /* int32 C */
  std::size_t total() const { return bytes_a + bytes_c; }
};

/* Layout inside bytes_a:
 *   [0, m_pad * k_pad)               : int8 A (signed, [-127, 127])
 *   [m_pad * k_pad, +m_pad * 4)      : float32 a_scales[m_pad]
 *
 * Layout inside bytes_c:
 *   [0, m_pad * n_pad * 4)           : int32 C scratch (row-major)
 * (no b_col_sum region — A is signed, no offset-binary fix-up) */
inline Int8RmScratch int8_rm_scratch(int m, int n, int k) {
  using T = Int8RmTraits;
  int m_pad = int8_rm_pad_up(m, T::M_STEP);
  int n_pad = int8_rm_pad_up(n, T::N_STEP);
  int k_pad = int8_rm_pad_up(k, T::K_STEP);
  std::size_t bytes_a = (std::size_t)m_pad * k_pad        /* int8 A */
                      + (std::size_t)m_pad * sizeof(float);
  std::size_t bytes_c = (std::size_t)m_pad * n_pad * sizeof(std::int32_t);
  return {bytes_a, bytes_c};
}

inline std::size_t int8_rm_a_scales_offset(int m_pad, int k) {
  using T = Int8RmTraits;
  int k_pad = int8_rm_pad_up(k, T::K_STEP);
  return (std::size_t)m_pad * k_pad;
}

/* No-op for AVX2; AMX needs ldtilecfg per thread, AVX2 does not. */
void int8_rm_tile_config_init();

/* Per-row quantize BF16 A [m, k] → int8 [-127, 127] + per-row float32
 * scale. Single-threaded — m is small. */
void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a);

/* Core compute. Reads B straight from caller memory with byte stride
 * ldb (>= k). Accumulates int8(A) × int8(B) into int32 C scratch via
 * the vpsignb/vpmaddubsw/vpmaddwd sequence described above. */
void int8_rm_run(int m, int n, int k,
                 const std::int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth);

/* Unpack int32 → float32 with per-row × per-col scale apply.
 * `_transposed` in the name is for AMX contract compatibility; the
 * AVX2 scratch is already row-major so the "transpose" is a no-op. */
void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,
                               const float* b_scales,
                               const void* scratch_c,
                               float alpha,
                               float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth);

}  // namespace cpu_gemm::kernels::avx2

#endif  /* CPU_GEMM_HAS_AVX2_INT8 */
#endif  /* CPU_GEMM_KERNELS_AVX2_INT8_GEMM_RM_H */
