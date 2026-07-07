/*
 * Tiny scalar reference + helpers used by tests and examples.
 *
 * No SIMD, no threading, no library dependencies — purely a ground truth.
 */
#ifndef CPU_GEMM_TEST_REFERENCE_H
#define CPU_GEMM_TEST_REFERENCE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_gemm/types.h"

namespace cpu_gemm::test {

inline float bf16_to_f32(cg_bf16_t v) {
  uint32_t bits = (uint32_t)v.bits << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline cg_bf16_t f32_to_bf16(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    cg_bf16_t r;
    r.bits = (uint16_t)((bits >> 16) | 64u);
    return r;
  }
  uint32_t tie = (bits >> 16) & 1u;
  uint32_t rounded = bits + 0x7fffu + tie;
  cg_bf16_t r;
  r.bits = (uint16_t)(rounded >> 16);
  return r;
}

/* Fill a BF16 buffer with values from a small uniform range — keeps the
 * dot-product accumulation well-behaved so we can use tight tolerances. */
inline void fill_bf16(std::vector<cg_bf16_t>& dst, uint64_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : dst) v = f32_to_bf16(dist(rng));
}

/* Naive C[m,n] = alpha * A[m,k] * B[n,k]^T + beta * C[m,n], all FP32 math. */
inline void ref_gemm_bf16_bf16_f32(int m, int n, int k,
                                   float alpha,
                                   const cg_bf16_t* a, size_t lda,
                                   const cg_bf16_t* b, size_t ldb,
                                   float beta,
                                   float* c, size_t ldc) {
  for (int mi = 0; mi < m; ++mi) {
    for (int ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (int ki = 0; ki < k; ++ki) {
        acc += (double)bf16_to_f32(a[(size_t)mi * lda + ki]) *
               (double)bf16_to_f32(b[(size_t)ni * ldb + ki]);
      }
      float& dst = c[(size_t)mi * ldc + ni];
      float prev = beta == 0.0f ? 0.0f : dst * beta;
      dst = prev + alpha * (float)acc;
    }
  }
}

/* Fill an int8 buffer with small signed integers. */
inline void fill_int8(std::vector<int8_t>& dst, uint64_t seed, int lo = -16, int hi = 16) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  for (auto& v : dst) v = (int8_t)dist(rng);
}

/* Fill a per-channel scale buffer with small positive floats. */
inline void fill_scales(std::vector<float>& dst, uint64_t seed, float lo = 0.005f, float hi = 0.05f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : dst) v = dist(rng);
}

/* Round-to-nearest-even + saturate to int8 — matches the kernel's
 * _mm512_cvtps_epi32 (default MXCSR rounding) + _mm512_cvtsepi32_epi8. */
inline int8_t quant_int8_rne(float x) {
  float r = std::nearbyint(x);
  if (r > 127.0f) r = 127.0f;
  if (r < -128.0f) r = -128.0f;
  return (int8_t)r;
}

/* Reference for the AMX INT8 path:
 *   A is BF16 [m,k], quantized per-row to int8 (d = amax/127) exactly as the
 *   kernel does. B is int8 [n,k] with per-channel scales b_scales[n].
 *   C[m,n] = beta*C + alpha * a_scale[i] * b_scale[j] * sum_k(quant(A) * B). */
inline void ref_gemm_bf16_int8_f32(int m, int n, int k,
                                   float alpha,
                                   const cg_bf16_t* a, size_t lda,
                                   const int8_t* b, size_t ldb,
                                   const float* b_scales,
                                   float beta,
                                   float* c, size_t ldc) {
  for (int mi = 0; mi < m; ++mi) {
    float amax = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
      float v = std::fabs(bf16_to_f32(a[(size_t)mi * lda + ki]));
      if (v > amax) amax = v;
    }
    float a_scale = amax / 127.0f;
    float inv = a_scale != 0.0f ? 1.0f / a_scale : 0.0f;
    for (int ni = 0; ni < n; ++ni) {
      long acc = 0;
      for (int ki = 0; ki < k; ++ki) {
        int8_t aq = quant_int8_rne(bf16_to_f32(a[(size_t)mi * lda + ki]) * inv);
        acc += (long)aq * (long)b[(size_t)ni * ldb + ki];
      }
      float& dst = c[(size_t)mi * ldc + ni];
      float prev = beta == 0.0f ? 0.0f : dst * beta;
      dst = prev + alpha * a_scale * b_scales[ni] * (float)acc;
    }
  }
}

inline float max_abs_diff(const float* a, const float* b, size_t n) {
  float worst = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

inline float max_rel_diff(const float* a, const float* b, size_t n) {
  float worst = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float denom = std::fabs(a[i]);
    if (denom < 1e-6f) denom = 1e-6f;
    float d = std::fabs(a[i] - b[i]) / denom;
    if (d > worst) worst = d;
  }
  return worst;
}

}  // namespace cpu_gemm::test
#endif
