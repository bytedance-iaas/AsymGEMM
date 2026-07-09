/*
 * AVX-512-VNNI INT8 GEMM (row-major B) — fallback path when AMX is
 * unavailable. Same shared-byte contract as kernels/amx/int8_gemm_rm.h:
 *
 *   B [N, K]     row-major int8     (the canonical pinned weight slab)
 *   B scales [N] per-row float32   (per-channel)
 *   A [M, K]     row-major BF16
 *   C [M, N]     row-major float32
 *
 * Implementation:
 *   vpdpbusd computes uint8 × int8 → int32 dot products. We quantize
 *   A to int8 (per-row scale) at pack time, then add 128 to convert
 *   to uint8 (offset binary). At unpack, we correct the bias:
 *     C = sum_k(a_int8 × b_int8)
 *       = sum_k((a_uint8 - 128) × b_int8)
 *       = sum_k(a_uint8 × b_int8) - 128 × sum_k(b_int8)
 *   Where sum_k(b_int8) per output column is precomputed during the
 *   run phase.
 *
 * This kernel is correctness-first, not peak-optimal. On SPR INT8
 * peak is ~25 TFLOPS via vpdpbusd vs ~80 TFLOPS via AMX; this kernel
 * exists for hosts that don't have AMX (Ice Lake, Cascade Lake) or
 * where AMX permission is denied.
 */
#ifndef CPU_GEMM_KERNELS_AVX512_INT8_GEMM_RM_H
#define CPU_GEMM_KERNELS_AVX512_INT8_GEMM_RM_H

#if defined(CPU_GEMM_HAS_AVX512_VNNI)

#include <cstddef>
#include <cstdint>

#include "cpu_gemm/types.h"

namespace cpu_gemm::kernels::avx512 {

/* Tile parameters chosen for AVX-512 zmm register usage:
 *   M_STEP = 4   (4 zmm accumulators per N-tile, leaving 28 zmm free)
 *   N_STEP = 16  (1 zmm = 16 int32 output lanes)
 *   K_STEP = 4   (one vpdpbusd consumes 4 K-values per lane)
 * N_BLOCK = 64   (4 zmm worth of output cols per inner work unit) */
struct Int8RmTraits {
  static constexpr int M_STEP  = 4;
  static constexpr int N_STEP  = 16;
  static constexpr int K_STEP  = 4;
  static constexpr int N_BLOCK = 64;
};

inline int int8_rm_pad_up(int x, int step) {
  return ((x + step - 1) / step) * step;
}

/* Scratch layout — matches AMX's Int8RmScratch shape so the
 * Int8RmBackend table abstracts uniformly over both. */
struct Int8RmScratch {
  std::size_t bytes_a;   /* uint8 A + per-row scales */
  std::size_t bytes_c;   /* int32 C + per-col b_sum (for the
                           * offset-binary correction) */
  std::size_t total() const { return bytes_a + bytes_c; }
};

/* Layout inside bytes_a:
 *   [0, m_pad * k_pad)               : uint8 A (a_int8 + 128)
 *   [m_pad * k_pad, +m_pad * 4)      : float32 a_scales[m_pad]
 *
 * Layout inside bytes_c:
 *   [0, m_pad * n_pad * 4)           : int32 C^T scratch (row-major)
 *   [m_pad*n_pad*4, +n_pad*4)        : int32 b_col_sum[n_pad]
 *                                      (used for offset-binary fix-up) */
inline Int8RmScratch int8_rm_scratch(int m, int n, int k) {
  using T = Int8RmTraits;
  int m_pad = int8_rm_pad_up(m, T::M_STEP);
  int n_pad = int8_rm_pad_up(n, T::N_STEP);
  int k_pad = int8_rm_pad_up(k, T::K_STEP);
  std::size_t bytes_a = (std::size_t)m_pad * k_pad        /* uint8 A */
                      + (std::size_t)m_pad * sizeof(float);
  std::size_t bytes_c = (std::size_t)m_pad * n_pad * sizeof(std::int32_t)
                      + (std::size_t)n_pad * sizeof(std::int32_t);
  return {bytes_a, bytes_c};
}

inline std::size_t int8_rm_a_scales_offset(int m_pad, int k) {
  using T = Int8RmTraits;
  int k_pad = int8_rm_pad_up(k, T::K_STEP);
  return (std::size_t)m_pad * k_pad;
}

/* No-op for AVX-512; AMX needs ldtilecfg per thread, VNNI does not. */
void int8_rm_tile_config_init();

/* Per-row quantize BF16 A [m, k] → uint8 (a_int8 + 128) + per-row
 * float32 scale. Single-threaded — m is small. */
void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a);

/* Core compute. Reads B straight from caller memory with byte stride
 * ldb (>= k). Accumulates uint8(A) × int8(B) into int32 C^T scratch
 * via vpdpbusd. Also computes per-N column sums of B during the
 * stream for use by the unpack-step offset-binary correction. */
void int8_rm_run(int m, int n, int k,
                 const std::int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth);

/* Unpack int32 → float32 with the offset-binary correction and
 * per-row × per-col scale apply. `_transposed` in the name is for
 * AMX contract compatibility; the VNNI scratch is already row-major
 * so the "transpose" is a no-op here. */
void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,
                               const float* b_scales,
                               const void* scratch_c,
                               float alpha,
                               float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth);

}  // namespace cpu_gemm::kernels::avx512

#endif  /* CPU_GEMM_HAS_AVX512_VNNI */
#endif  /* CPU_GEMM_KERNELS_AVX512_INT8_GEMM_RM_H */
