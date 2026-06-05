/*
 * Public entry for the AMX BF16 GEMM.
 *
 * Takes row-major matrices on the caller's side; packs into the kernel's
 * blocked AMX layout on a caller-supplied scratch buffer. For workloads
 * where the same B is used many times, ktransformers' callers pack B once
 * and reuse — a `cg_pack_b()` C ABI for that is a future addition.
 *
 * Layout, threading:
 *   C[m, n] += alpha * A[m, k] * B^T[k, n]   (i.e. B is provided [n, k])
 *   Each thread is responsible for one slice in n of width K::N_BLOCK.
 *   ith/nth split happens *outside* this function — the caller's thread
 *   pool invokes it once per ith.
 *
 * Scratch sizes:
 *   bytes_a = max_m_pad * k                * sizeof(bf16)
 *   bytes_b = n_pad   * k                  * sizeof(bf16)
 *   bytes_c = max_m_pad * n_pad            * sizeof(float)
 * where max_m_pad = round_up(m, M_STEP), n_pad = round_up(n, N_STEP). The
 * dispatcher computes these via cg_amx_bf16_scratch().
 */
#ifndef CPU_GEMM_KERNELS_AMX_BF16_GEMM_H
#define CPU_GEMM_KERNELS_AMX_BF16_GEMM_H

#if defined(CPU_GEMM_HAS_AMX)

#include <cstddef>

#include "cpu_gemm/types.h"

namespace cpu_gemm::kernels::amx {

/* Kernel block-size constants exposed for scratch sizing. Must match the
 * GemmKernel224BF16 declared in bf16_gemm.cpp. */
struct Bf16KernelTraits {
  static constexpr int M_STEP  = 32;
  static constexpr int N_STEP  = 32;
  static constexpr int K_STEP  = 32;
  static constexpr int N_BLOCK = 256;
  static constexpr int K_BLOCK = 1792;
};

inline int pad_up(int v, int m) { return (v + m - 1) / m * m; }

struct Bf16Scratch {
  std::size_t bytes_a;
  std::size_t bytes_b;
  std::size_t bytes_c;
  std::size_t total() const { return bytes_a + bytes_b + bytes_c; }
};

inline Bf16Scratch bf16_scratch(int m, int n, int k) {
  using T = Bf16KernelTraits;
  int max_m_pad = pad_up(m, T::M_STEP);
  int n_pad     = pad_up(n, T::N_STEP);
  /* k must already be multiple of K_STEP; checked at runtime in dispatcher. */
  return {
      sizeof(cg_bf16_t) * (std::size_t)max_m_pad * k,
      sizeof(cg_bf16_t) * (std::size_t)n_pad   * k,
      sizeof(float)     * (std::size_t)max_m_pad * n_pad,
  };
}

/* Detect whether AMX is currently usable on this thread. Caches the result
 * within the calling thread. */
bool amx_available();

/* Initialize AMX tile configuration on the calling thread for the BF16
 * kernel. Idempotent and cheap to call repeatedly. */
void bf16_tile_config_init();

/* Pack row-major matrices into the kernel layouts on the provided scratch.
 * Multi-threaded: each thread packs its share of B according to (ith, nth).
 * A is packed once (only by ith == 0). */
void bf16_pack(int m, int n, int k,
               const cg_bf16_t* a_rm, std::size_t lda,
               const cg_bf16_t* b_rm, std::size_t ldb,
               void* scratch_a, void* scratch_b, void* scratch_c,
               int ith, int nth);

/* Run one (ith, nth) slice of the packed GEMM. n is split into N_BLOCK
 * chunks; slice ith handles chunks ith, ith+nth, … . */
void bf16_run(int m, int n, int k,
              void* scratch_a, void* scratch_b, void* scratch_c,
              int ith, int nth);

/* Unpack the blocked C buffer back into row-major FP32 with alpha/beta. */
void bf16_unpack(int m, int n,
                 const void* scratch_c,
                 float alpha,
                 float beta, float* c_rm, std::size_t ldc,
                 int ith, int nth);

}  // namespace cpu_gemm::kernels::amx

#endif  // CPU_GEMM_HAS_AMX
#endif
