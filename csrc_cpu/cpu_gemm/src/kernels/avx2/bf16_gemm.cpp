/*
 * AVX2 BF16 GEMM implementation.
 *
 * Port of operators/avx2/avx2_bf16_gemm.hpp::gemm_bf16 with the buffer
 * wrappers removed. Inner loop unchanged: 4 parallel FP32 accumulators,
 * 32 K-step, scalar tail. Reduction matches ggml_vec_dot_bf16's AVX2 path.
 */
#include "kernels/avx2/bf16_gemm.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "kernels/bf16_compat.h"

namespace cpu_gemm::kernels::avx2 {

namespace {

inline std::pair<int, int> split_range(int total, int ith, int nth) {
  int per = total / nth;
  int rem = total % nth;
  int start = ith * per + std::min(ith, rem);
  int end = start + per + (ith < rem ? 1 : 0);
  return {start, end};
}

inline __m256 load_bf16_to_fp32(const cg_bf16_t* src) {
  __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  __m256i i32 = _mm256_cvtepu16_epi32(bf16);
  return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

inline float hsum(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

}  // namespace

void gemm_bf16_bf16_f32(int m, int n, int k,
                        float alpha,
                        const cg_bf16_t* a, size_t lda,
                        const cg_bf16_t* b, size_t ldb,
                        float beta,
                        float* c, size_t ldc,
                        int ith, int nth) {
  auto [n_start, n_end] = split_range(n, ith, nth);

  for (int ni = n_start; ni < n_end; ++ni) {
    const cg_bf16_t* b_row = b + (size_t)ni * ldb;

    for (int mi = 0; mi < m; ++mi) {
      const cg_bf16_t* a_row = a + (size_t)mi * lda;

      __m256 c1 = _mm256_setzero_ps();
      __m256 c2 = _mm256_setzero_ps();
      __m256 c3 = _mm256_setzero_ps();
      __m256 c4 = _mm256_setzero_ps();

      int ki = 0;
      for (; ki + 32 <= k; ki += 32) {
        c1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki),
                             load_bf16_to_fp32(b_row + ki), c1);
        c2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 8),
                             load_bf16_to_fp32(b_row + ki + 8), c2);
        c3 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 16),
                             load_bf16_to_fp32(b_row + ki + 16), c3);
        c4 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 24),
                             load_bf16_to_fp32(b_row + ki + 24), c4);
      }

      float sum = hsum(_mm256_add_ps(_mm256_add_ps(c1, c3),
                                     _mm256_add_ps(c2, c4)));

      for (; ki < k; ++ki) {
        sum += cg_bf16_to_fp32(a_row[ki]) * cg_bf16_to_fp32(b_row[ki]);
      }

      float* c_elem = c + (size_t)mi * ldc + ni;
      float prev = beta == 0.0f ? 0.0f : *c_elem * beta;
      *c_elem = prev + alpha * sum;
    }
  }
}

}  // namespace cpu_gemm::kernels::avx2
