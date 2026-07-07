/*
 * Public entry for the AMX INT8 GEMM.
 *
 * Compute model:
 *   C[m, n] (FP32) = alpha * (A_int8[m, k] * B_int8^T[n, k]) * a_scale[m] * b_scale[n]
 *                  + beta  * C[m, n]
 *
 * A is BF16 on the caller's side and quantized to int8 row-wise inside
 * pack_a (matches ktransformers' dynamic activation quant).
 *
 * B is provided either as BF16 (we quantize per output channel) or as
 * pre-quantized int8 + scales. Both paths go through the same packed
 * blocked layout the AMX tile loop expects.
 *
 * Threading is the same N_BLOCK split as the BF16 path; the dispatcher
 * runs three sequential parallel_for phases (pack → run → unpack).
 */
#ifndef CPU_GEMM_KERNELS_AMX_INT8_GEMM_H
#define CPU_GEMM_KERNELS_AMX_INT8_GEMM_H

#if defined(CPU_GEMM_HAS_AMX)

#include <cstddef>
#include <cstdint>

#include "cpu_gemm/types.h"

namespace cpu_gemm::kernels::amx {

struct Int8KernelTraits {
  static constexpr int M_STEP  = 32;
  static constexpr int N_STEP  = 32;
  static constexpr int K_STEP  = 64;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 3584;
};

inline int int8_pad_up(int v, int m) { return (v + m - 1) / m * m; }

struct Int8Scratch {
  std::size_t bytes_a;  /* int8 acts + per-row scale */
  std::size_t bytes_b;  /* int8 weights + per-channel scale */
  std::size_t bytes_c;  /* int32 accumulator */
  std::size_t total() const { return bytes_a + bytes_b + bytes_c; }
};

inline Int8Scratch int8_scratch(int m, int n, int k) {
  using T = Int8KernelTraits;
  int max_m_pad = int8_pad_up(m, T::M_STEP);
  int n_pad     = int8_pad_up(n, T::N_STEP);
  return {
      sizeof(int8_t)  * (std::size_t)max_m_pad * k + sizeof(float) * max_m_pad,
      sizeof(int8_t)  * (std::size_t)n_pad   * k + sizeof(float) * n_pad,
      sizeof(int32_t) * (std::size_t)max_m_pad * n_pad,
  };
}

/* Initialize AMX tile shape for the INT8 kernel on the calling thread.
 * Idempotent per thread; reloads tile config on every invocation to be
 * robust against intervening _tile_release() in user code. */
void int8_tile_config_init();

/* Path A: B is BF16; quantize it per output channel during pack. */
void int8_pack_a_bf16(int m, int k,
                      const cg_bf16_t* a_rm,
                      void* scratch_a);

void int8_pack_b_bf16(int n, int k,
                      const cg_bf16_t* b_rm,
                      void* scratch_b,
                      int ith, int nth);

/* Path B: B is int8 + per-channel scales already. */
void int8_pack_b_int8(int n, int k,
                      const int8_t* b_int8,
                      const float* b_scales,
                      void* scratch_b,
                      int ith, int nth);

void int8_run(int m, int n, int k,
              void* scratch_a, void* scratch_b, void* scratch_c,
              int ith, int nth);

/* Unpack the blocked int32 C buffer, apply per-row and per-channel scales,
 * and write the alpha/beta-combined FP32 result to row-major c_rm. The
 * scales pointers are passed explicitly so callers don't need to know the
 * internal scratch_a / scratch_b layout. */
void int8_unpack_explicit(int m, int n,
                          const float* a_scales,
                          const float* b_scales,
                          const void* scratch_c,
                          float alpha,
                          float beta, float* c_rm, std::size_t ldc,
                          int ith, int nth);

/* Return offsets of the scale arrays inside the packed scratch buffers. */
std::size_t int8_a_scales_offset(int max_m_pad, int k);
std::size_t int8_b_scales_offset(int n_pad,     int k);

/* Offline B pre-pack (single-threaded; offline use). */
std::size_t int8_packed_b_size(int n, int k);
void        int8_pack_b_int8_offline(int n, int k,
                                     const int8_t* b_int8,
                                     const float* b_scales,
                                     void* dst);

}  // namespace cpu_gemm::kernels::amx

#endif
#endif
