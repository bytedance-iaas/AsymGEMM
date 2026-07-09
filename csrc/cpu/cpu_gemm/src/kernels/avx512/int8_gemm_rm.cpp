/*
 * AVX-512-VNNI INT8 GEMM (row-major B) — implementation.
 * See header for the contract and the design rationale.
 */
#include "kernels/avx512/int8_gemm_rm.h"

#if defined(CPU_GEMM_HAS_AVX512_VNNI)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "kernels/bf16_compat.h"

namespace cpu_gemm::kernels::avx512 {

namespace {

constexpr int M_STEP  = Int8RmTraits::M_STEP;
constexpr int N_STEP  = Int8RmTraits::N_STEP;
constexpr int K_STEP  = Int8RmTraits::K_STEP;
constexpr int N_BLOCK = Int8RmTraits::N_BLOCK;

/* Layout helpers — keep header inline-friendly and impl noise here. */
inline int pad_up(int x, int step) { return ((x + step - 1) / step) * step; }

}  // namespace

void int8_rm_tile_config_init() {
  /* AMX needs ldtilecfg per thread; AVX-512 does not. Keep the symbol
   * so the Int8RmBackend table can hold a non-null function pointer
   * even on hosts where AMX was never available. */
}

void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a) {
  int m_pad = pad_up(m, M_STEP);
  int k_pad = pad_up(k, K_STEP);
  uint8_t* a_uint8 = static_cast<uint8_t*>(scratch_a);
  float*   a_scales = reinterpret_cast<float*>(
      a_uint8 + (size_t)m_pad * k_pad);

  /* Per-row: find max-abs in FP32, compute symmetric per-row scale,
   * quantize to int8, then add 128 to get uint8 (offset binary). */
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

    uint8_t* row = a_uint8 + (size_t)mi * k_pad;
    for (int ki = 0; ki < k; ++ki) {
      float v = cg_bf16_to_fp32(a_rm[mi * k + ki]);
      int q = (int)std::lrintf(v * inv_scale);
      if (q < -127) q = -127;
      if (q >  127) q =  127;
      row[ki] = static_cast<uint8_t>(q + 128);
    }
    /* Pad K with 128 (= int8 zero in offset-binary). */
    for (int ki = k; ki < k_pad; ++ki) row[ki] = 128;
  }
  /* Pad M rows with zeros (uint8 128) — they contribute zero to
   * any output via the corrected dot product. */
  for (int mi = m; mi < m_pad; ++mi) {
    a_scales[mi] = 1.0f;
    uint8_t* row = a_uint8 + (size_t)mi * k_pad;
    std::memset(row, 128, k_pad);
  }
}

void int8_rm_run(int m, int n, int k,
                 const std::int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth) {
  int m_pad = pad_up(m, M_STEP);
  int n_pad = pad_up(n, N_STEP);
  int k_pad = pad_up(k, K_STEP);

  const uint8_t* a_uint8 = static_cast<const uint8_t*>(scratch_a);
  int32_t*  c_int32  = static_cast<int32_t*>(scratch_c);
  int32_t*  b_col_sum = c_int32 + (size_t)m_pad * n_pad;

  /* Distribute N-blocks across threads. */
  int n_blocks = n_pad / N_STEP;
  int blocks_per_thread = (n_blocks + nth - 1) / nth;
  int nb_lo = ith * blocks_per_thread;
  int nb_hi = std::min(nb_lo + blocks_per_thread, n_blocks);

  for (int nb = nb_lo; nb < nb_hi; ++nb) {
    int n_off = nb * N_STEP;

    /* --- Precompute b_col_sum for this N-block (16 cols). --- */
    int32_t local_col_sum[N_STEP];
    for (int ni = 0; ni < N_STEP; ++ni) {
      int n_idx = n_off + ni;
      int32_t sum = 0;
      if (n_idx < n) {
        const int8_t* brow = b_rm + (size_t)n_idx * ldb;
        for (int ki = 0; ki < k; ++ki) sum += (int32_t)brow[ki];
      }
      local_col_sum[ni] = sum;
      b_col_sum[n_idx] = sum;
    }

    /* --- Outer M-tile loop. --- */
    for (int mi = 0; mi < m_pad; mi += M_STEP) {
      __m512i acc0 = _mm512_setzero_si512();
      __m512i acc1 = _mm512_setzero_si512();
      __m512i acc2 = _mm512_setzero_si512();
      __m512i acc3 = _mm512_setzero_si512();

      /* For each k-chunk of 4 int8 values: build a 64-byte zmm of B
       * where lane ni × 4 + kj = B[n_off+ni, ki+kj]. The 64 bytes are
       * not contiguous in B's row-major memory (they're 16 strided
       * 4-byte reads), so we assemble them in a stack buffer. This
       * is the simplest correct implementation; an optimized version
       * could prefetch and pipeline. */
      alignas(64) int8_t b_chunk[64];

      for (int ki = 0; ki < k_pad; ki += K_STEP) {
        /* Pack the 16×4 B subtile. Padding-Ks (ki >= k) get zero. */
        for (int ni = 0; ni < N_STEP; ++ni) {
          int n_idx = n_off + ni;
          if (n_idx < n) {
            const int8_t* brow = b_rm + (size_t)n_idx * ldb;
            for (int kj = 0; kj < K_STEP; ++kj) {
              int k_idx = ki + kj;
              b_chunk[ni * 4 + kj] = (k_idx < k) ? brow[k_idx] : 0;
            }
          } else {
            for (int kj = 0; kj < K_STEP; ++kj) b_chunk[ni * 4 + kj] = 0;
          }
        }
        __m512i b = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(b_chunk));

        /* A side: 4 K-bytes per (M, K)-chunk, broadcast to all 16
         * VNNI lanes of a zmm. */
        const uint8_t* arow0 = a_uint8 + (size_t)(mi + 0) * k_pad + ki;
        const uint8_t* arow1 = a_uint8 + (size_t)(mi + 1) * k_pad + ki;
        const uint8_t* arow2 = a_uint8 + (size_t)(mi + 2) * k_pad + ki;
        const uint8_t* arow3 = a_uint8 + (size_t)(mi + 3) * k_pad + ki;

        uint32_t a0_4, a1_4, a2_4, a3_4;
        std::memcpy(&a0_4, arow0, 4);
        std::memcpy(&a1_4, arow1, 4);
        std::memcpy(&a2_4, arow2, 4);
        std::memcpy(&a3_4, arow3, 4);

        __m512i a0 = _mm512_set1_epi32(static_cast<int32_t>(a0_4));
        __m512i a1 = _mm512_set1_epi32(static_cast<int32_t>(a1_4));
        __m512i a2 = _mm512_set1_epi32(static_cast<int32_t>(a2_4));
        __m512i a3 = _mm512_set1_epi32(static_cast<int32_t>(a3_4));

        acc0 = _mm512_dpbusd_epi32(acc0, a0, b);
        acc1 = _mm512_dpbusd_epi32(acc1, a1, b);
        acc2 = _mm512_dpbusd_epi32(acc2, a2, b);
        acc3 = _mm512_dpbusd_epi32(acc3, a3, b);
      }

      /* Store M_STEP × N_STEP int32 accumulators into C scratch. */
      int32_t* crow0 = c_int32 + (size_t)(mi + 0) * n_pad + n_off;
      int32_t* crow1 = c_int32 + (size_t)(mi + 1) * n_pad + n_off;
      int32_t* crow2 = c_int32 + (size_t)(mi + 2) * n_pad + n_off;
      int32_t* crow3 = c_int32 + (size_t)(mi + 3) * n_pad + n_off;
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(crow0), acc0);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(crow1), acc1);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(crow2), acc2);
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(crow3), acc3);
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
  int m_pad = pad_up(m, M_STEP);
  int n_pad = pad_up(n, N_STEP);
  const int32_t* c_int32   = static_cast<const int32_t*>(scratch_c);
  const int32_t* b_col_sum = c_int32 + (size_t)m_pad * n_pad;

  /* Distribute M rows across threads. */
  int m_per_thread = (m + nth - 1) / nth;
  int m_lo = ith * m_per_thread;
  int m_hi = std::min(m_lo + m_per_thread, m);

  for (int mi = m_lo; mi < m_hi; ++mi) {
    float a_s = a_scales[mi];
    const int32_t* cint_row = c_int32 + (size_t)mi * n_pad;
    float* c_out_row = c_rm + (size_t)mi * ldc;

    for (int ni = 0; ni < n; ++ni) {
      /* Offset-binary correction:
       *   raw = sum_k(uint8_a × int8_b)
       *   corrected = raw - 128 × sum_k(int8_b) */
      int32_t corrected = cint_row[ni] - 128 * b_col_sum[ni];
      float val = static_cast<float>(corrected) * a_s * b_scales[ni];
      val *= alpha;
      if (beta == 0.0f) {
        c_out_row[ni] = val;
      } else {
        c_out_row[ni] = val + beta * c_out_row[ni];
      }
    }
  }
}

}  // namespace cpu_gemm::kernels::avx512

#endif  /* CPU_GEMM_HAS_AVX512_VNNI */
