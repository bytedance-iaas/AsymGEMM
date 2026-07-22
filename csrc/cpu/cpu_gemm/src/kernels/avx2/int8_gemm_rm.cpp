/*
 * AVX2 INT8 GEMM (row-major B) — implementation.
 * See header for the contract and the saturation-safety argument.
 */
#include "kernels/avx2/int8_gemm_rm.h"

#if defined(CPU_GEMM_HAS_AVX2_INT8)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "kernels/bf16_compat.h"

namespace cpu_gemm::kernels::avx2 {

namespace {

constexpr int M_STEP  = Int8RmTraits::M_STEP;
constexpr int N_STEP  = Int8RmTraits::N_STEP;
constexpr int K_STEP  = Int8RmTraits::K_STEP;

inline int pad_up(int x, int step) { return ((x + step - 1) / step) * step; }

}  // namespace

void int8_rm_tile_config_init() {
  /* AMX needs ldtilecfg per thread; AVX2 does not. Keep the symbol so
   * the Int8RmBackend table can hold a non-null function pointer. */
}

void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a) {
  int m_pad = pad_up(m, M_STEP);
  int k_pad = pad_up(k, K_STEP);
  int8_t* a_int8  = static_cast<int8_t*>(scratch_a);
  float*  a_scales = reinterpret_cast<float*>(
      a_int8 + (size_t)m_pad * k_pad);

  /* Per-row: find max-abs in FP32, compute symmetric per-row scale,
   * quantize to int8 in [-127, 127] (kept signed — the run phase folds
   * A's sign into B via vpsignb, so no offset-binary rebias). */
  for (int mi = 0; mi < m; ++mi) {
    float max_abs = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
      float v = cg_bf16_to_fp32(a_rm[mi * k + ki]);
      float av = v < 0 ? -v : v;
      if (av > max_abs) max_abs = av;
    }
    /* Avoid div by zero on all-zero rows. */
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1e-12f;
    float inv_scale = 1.0f / scale;
    a_scales[mi] = scale;

    int8_t* row = a_int8 + (size_t)mi * k_pad;
    for (int ki = 0; ki < k; ++ki) {
      float v = cg_bf16_to_fp32(a_rm[mi * k + ki]);
      int q = (int)std::lrintf(v * inv_scale);
      if (q < -127) q = -127;
      if (q >  127) q =  127;
      row[ki] = static_cast<int8_t>(q);
    }
    /* Pad K with zeros — they contribute nothing to any dot product. */
    for (int ki = k; ki < k_pad; ++ki) row[ki] = 0;
  }
  /* Pad M rows with zeros. */
  for (int mi = m; mi < m_pad; ++mi) {
    a_scales[mi] = 1.0f;
    std::memset(a_int8 + (size_t)mi * k_pad, 0, k_pad);
  }
}

void int8_rm_run(int m, int n, int k,
                 const std::int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth) {
  int m_pad = pad_up(m, M_STEP);
  int n_pad = pad_up(n, N_STEP);
  int k_pad = pad_up(k, K_STEP);

  const int8_t* a_int8  = static_cast<const int8_t*>(scratch_a);
  int32_t*      c_int32 = static_cast<int32_t*>(scratch_c);

  /* Distribute N-blocks across threads. */
  int n_blocks = n_pad / N_STEP;
  int blocks_per_thread = (n_blocks + nth - 1) / nth;
  int nb_lo = ith * blocks_per_thread;
  int nb_hi = std::min(nb_lo + blocks_per_thread, n_blocks);

  const __m256i ones16 = _mm256_set1_epi16(1);

  for (int nb = nb_lo; nb < nb_hi; ++nb) {
    int n_off = nb * N_STEP;

    /* --- Outer M-tile loop. --- */
    for (int mi = 0; mi < m_pad; mi += M_STEP) {
      __m256i acc0 = _mm256_setzero_si256();
      __m256i acc1 = _mm256_setzero_si256();
      __m256i acc2 = _mm256_setzero_si256();
      __m256i acc3 = _mm256_setzero_si256();

      /* For each k-chunk of 4 int8 values: build a 32-byte ymm of B
       * where byte ni × 4 + kj = B[n_off+ni, ki+kj]. The 32 bytes are
       * not contiguous in B's row-major memory (they're 8 strided
       * 4-byte reads), so we assemble them in a stack buffer. This is
       * the simplest correct implementation, mirroring the AVX-512
       * kernel's structure. */
      alignas(32) int8_t b_chunk[32];

      for (int ki = 0; ki < k_pad; ki += K_STEP) {
        /* Pack the 8×4 B subtile. Padding-Ks (ki >= k) get zero. */
        if (ki + K_STEP <= k) {
          for (int ni = 0; ni < N_STEP; ++ni) {
            int n_idx = n_off + ni;
            if (n_idx < n) {
              std::memcpy(b_chunk + ni * 4,
                          b_rm + (size_t)n_idx * ldb + ki, 4);
            } else {
              std::memset(b_chunk + ni * 4, 0, 4);
            }
          }
        } else {
          for (int ni = 0; ni < N_STEP; ++ni) {
            int n_idx = n_off + ni;
            for (int kj = 0; kj < K_STEP; ++kj) {
              int k_idx = ki + kj;
              b_chunk[ni * 4 + kj] =
                  (n_idx < n && k_idx < k)
                      ? b_rm[(size_t)n_idx * ldb + k_idx] : 0;
            }
          }
        }
        __m256i b = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(b_chunk));

        /* A side: 4 K-bytes per (M, K)-chunk, broadcast to all 8 dword
         * lanes of a ymm; the byte within a lane matches the K index
         * of B's byte, so vpsignb pairs a's sign with the right b. */
        const int8_t* arow0 = a_int8 + (size_t)(mi + 0) * k_pad + ki;
        const int8_t* arow1 = a_int8 + (size_t)(mi + 1) * k_pad + ki;
        const int8_t* arow2 = a_int8 + (size_t)(mi + 2) * k_pad + ki;
        const int8_t* arow3 = a_int8 + (size_t)(mi + 3) * k_pad + ki;

        uint32_t a0_4, a1_4, a2_4, a3_4;
        std::memcpy(&a0_4, arow0, 4);
        std::memcpy(&a1_4, arow1, 4);
        std::memcpy(&a2_4, arow2, 4);
        std::memcpy(&a3_4, arow3, 4);

        __m256i a0 = _mm256_set1_epi32(static_cast<int32_t>(a0_4));
        __m256i a1 = _mm256_set1_epi32(static_cast<int32_t>(a1_4));
        __m256i a2 = _mm256_set1_epi32(static_cast<int32_t>(a2_4));
        __m256i a3 = _mm256_set1_epi32(static_cast<int32_t>(a3_4));

        /* a×b = |a| × sign(a)·b. |a| <= 127 and |b| <= 127 (quantizer
         * contract), so vpmaddubsw's int16 pair-sums cannot saturate. */
        __m256i p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(a0, a0),
                                          _mm256_sign_epi8(b, a0));
        __m256i p1 = _mm256_maddubs_epi16(_mm256_sign_epi8(a1, a1),
                                          _mm256_sign_epi8(b, a1));
        __m256i p2 = _mm256_maddubs_epi16(_mm256_sign_epi8(a2, a2),
                                          _mm256_sign_epi8(b, a2));
        __m256i p3 = _mm256_maddubs_epi16(_mm256_sign_epi8(a3, a3),
                                          _mm256_sign_epi8(b, a3));

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones16));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones16));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(p2, ones16));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(p3, ones16));
      }

      /* Store M_STEP × N_STEP int32 accumulators into C scratch. */
      int32_t* crow0 = c_int32 + (size_t)(mi + 0) * n_pad + n_off;
      int32_t* crow1 = c_int32 + (size_t)(mi + 1) * n_pad + n_off;
      int32_t* crow2 = c_int32 + (size_t)(mi + 2) * n_pad + n_off;
      int32_t* crow3 = c_int32 + (size_t)(mi + 3) * n_pad + n_off;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(crow0), acc0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(crow1), acc1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(crow2), acc2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(crow3), acc3);
    }
  }
}

void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,
                               const float* b_scales,
                               const void* scratch_c,
                               float alpha,
                               float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth) {
  int n_pad = pad_up(n, N_STEP);
  const int32_t* c_int32 = static_cast<const int32_t*>(scratch_c);

  /* Distribute M rows across threads. */
  int m_per_thread = (m + nth - 1) / nth;
  int m_lo = ith * m_per_thread;
  int m_hi = std::min(m_lo + m_per_thread, m);

  for (int mi = m_lo; mi < m_hi; ++mi) {
    float a_s = a_scales[mi];
    const int32_t* cint_row = c_int32 + (size_t)mi * n_pad;
    float* c_out_row = c_rm + (size_t)mi * ldc;

    for (int ni = 0; ni < n; ++ni) {
      float val = static_cast<float>(cint_row[ni]) * a_s * b_scales[ni];
      val *= alpha;
      if (beta == 0.0f) {
        c_out_row[ni] = val;
      } else {
        c_out_row[ni] = val + beta * c_out_row[ni];
      }
    }
  }
}

}  // namespace cpu_gemm::kernels::avx2

#endif  /* CPU_GEMM_HAS_AVX2_INT8 */
