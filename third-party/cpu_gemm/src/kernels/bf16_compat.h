/*
 * Standalone BF16 <-> FP32 helpers. Bit-for-bit identical to ggml's
 * round-to-nearest-even semantics so kernels ported from ktransformers
 * keep their reference output.
 */
#ifndef CPU_GEMM_KERNELS_BF16_COMPAT_H
#define CPU_GEMM_KERNELS_BF16_COMPAT_H

#include <stdint.h>
#include <string.h>

#include "cpu_gemm/types.h"

static inline float cg_bf16_to_fp32(cg_bf16_t v) {
  uint32_t bits = (uint32_t)v.bits << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static inline cg_bf16_t cg_fp32_to_bf16(float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  /* NaN preserved with mantissa MSB. Other values use round-to-nearest-even. */
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    cg_bf16_t r;
    r.bits = (uint16_t)((bits >> 16) | 64u);
    return r;
  }
  uint32_t tie = ((bits >> 16) & 1u);
  uint32_t rounded = bits + 0x7fffu + tie;
  cg_bf16_t r;
  r.bits = (uint16_t)(rounded >> 16);
  return r;
}

#endif
