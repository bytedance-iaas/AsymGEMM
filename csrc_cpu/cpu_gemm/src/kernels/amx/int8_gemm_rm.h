/*
 * Stride-aware AMX INT8 GEMM (row-major B).
 *
 * Direct sibling of int8_gemm.{h,cpp}. Same int8.s8.s32 contraction, same
 * tile config, same numerics — but the **operand roles are swapped** so we
 * can feed B straight from row-major pinned memory without a pre-pack:
 *
 *   AMX src1 (rows-major slot)  ← B[N, K] int8 row-major, strided load.
 *   AMX src2 (VNNI/K-row slot)  ← A VNNI-packed (same byte layout the
 *                                  current BufferBInt8 produces for B).
 *   AMX dst                     ← C^T[n, m] int32.
 *
 * Mathematically:  C = A · B^T   ⟺   C^T = B · A^T.
 *
 * The output unpack does an implicit 32×32 int32 transpose while applying
 * (alpha · sA[m] · sB[n]) to recover row-major C[m, n].
 *
 * See `Stride.md` §3 for the design rationale.
 */
#ifndef CPU_GEMM_KERNELS_AMX_INT8_GEMM_RM_H
#define CPU_GEMM_KERNELS_AMX_INT8_GEMM_RM_H

#if defined(CPU_GEMM_HAS_AMX)

#include <cstddef>
#include <cstdint>

#include "cpu_gemm/types.h"
#include "kernels/amx/int8_gemm.h"  /* Int8KernelTraits, int8_pad_up */

namespace cpu_gemm::kernels::amx {

/* Scratch sizes for the stride-aware INT8 path.
 *
 * Differences vs the packed path (int8_scratch):
 *   - No bytes_b: B is read straight from caller memory.
 *   - bytes_a now matches the packed BufferB shape (VNNI), not the row-major
 *     BufferA — A's pack target moved to AMX src2.
 *   - bytes_c is the C^T scratch, sized [n_pad, max_m_pad] int32. */
struct Int8RmScratch {
  std::size_t bytes_a;     /* VNNI-packed A + per-row scales */
  std::size_t bytes_c;     /* C^T int32, blocked */
  std::size_t total() const { return bytes_a + bytes_c; }
};

inline Int8RmScratch int8_rm_scratch(int m, int n, int k) {
  using T = Int8KernelTraits;
  int max_m_pad = int8_pad_up(m, T::M_STEP);
  int n_pad     = int8_pad_up(n, T::N_STEP);
  return {
      sizeof(int8_t)  * (std::size_t)max_m_pad * k + sizeof(float) * max_m_pad,
      sizeof(int32_t) * (std::size_t)n_pad * max_m_pad,
  };
}

/* Offset of the per-row A scales inside scratch_a. */
inline std::size_t int8_rm_a_scales_offset(int max_m_pad, int k) {
  return sizeof(int8_t) * (std::size_t)max_m_pad * k;
}

/* Configure AMX tiles for the rm path on the calling thread. Identical
 * bit-pattern to int8_tile_config_init — the labels swap, the config does
 * not. Kept as a separate symbol so callers don't need to know that. */
void int8_rm_tile_config_init();

/* Per-row quantize BF16 A [m, k] → INT8 + per-row scale; store in VNNI form
 * the AMX src2 slot expects. lda is assumed equal to k (the dispatcher
 * enforces this).  Single-threaded — A is small. */
void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a);

/* Core compute. Reads B straight from row-major caller memory with byte
 * stride ldb (must be ≥ k). Writes C^T int32 accumulators into scratch_c.
 * Threading: partitions whole N_BLOCK stripes across (ith, nth). */
void int8_rm_run(int m, int n, int k,
                 const int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth);

/* Unpack C^T int32 → row-major FP32 C with the implicit 32×32 transpose
 * and (alpha · sA · sB) scale apply. */
void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,
                               const float* b_scales,
                               const void* scratch_c,
                               float alpha,
                               float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth);

}  // namespace cpu_gemm::kernels::amx

#endif  /* CPU_GEMM_HAS_AMX */
#endif  /* CPU_GEMM_KERNELS_AMX_INT8_GEMM_RM_H */
