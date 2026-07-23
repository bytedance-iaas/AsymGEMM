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

namespace {

inline int32_t hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

/* Streaming row-dot micro-block: MB A-rows × NU B-rows, K swept in 32-byte
 * chunks straight from B's row-major memory — no B pack. Each B byte is
 * loaded exactly once per call, so for small m the kernel runs at memory
 * speed instead of the tiled path's scalar-pack speed. Same sign trick as
 * the tiled path: a×b = |a| × sign(a)·b, both operands in [-127, 127]. */
template <int MB, int NU>
inline void dot_block_stream(const int8_t* a_int8, int k_pad,
                             const int8_t* b_rm, std::size_t ldb,
                             int n0, int k,
                             int32_t* c_int32, int n_pad) {
  const __m256i ones16 = _mm256_set1_epi16(1);
  const int k32 = k & ~31;

  __m256i acc[MB * NU];
  for (int i = 0; i < MB * NU; ++i) acc[i] = _mm256_setzero_si256();

  const int8_t* brow[NU];
  for (int u = 0; u < NU; ++u) brow[u] = b_rm + (std::size_t)(n0 + u) * ldb;

  for (int kb = 0; kb < k32; kb += 32) {
    __m256i b[NU];
    for (int u = 0; u < NU; ++u)
      b[u] = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(brow[u] + kb));
    for (int j = 0; j < MB; ++j) {
      const __m256i aj = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(a_int8 + (std::size_t)j * k_pad + kb));
      const __m256i pa = _mm256_sign_epi8(aj, aj);
      for (int u = 0; u < NU; ++u) {
        const __m256i p =
            _mm256_maddubs_epi16(pa, _mm256_sign_epi8(b[u], aj));
        acc[j * NU + u] =
            _mm256_add_epi32(acc[j * NU + u], _mm256_madd_epi16(p, ones16));
      }
    }
  }

  /* K tail (< 32 int8): scalar. A is zero-padded past k, B is valid to k. */
  int32_t tail[MB * NU] = {};
  for (int kk = k32; kk < k; ++kk) {
    for (int j = 0; j < MB; ++j) {
      const int32_t av = a_int8[(std::size_t)j * k_pad + kk];
      if (av == 0) continue;
      for (int u = 0; u < NU; ++u)
        tail[j * NU + u] += av * (int32_t)brow[u][kk];
    }
  }

  for (int j = 0; j < MB; ++j)
    for (int u = 0; u < NU; ++u)
      c_int32[(std::size_t)j * n_pad + n0 + u] =
          hsum_epi32(acc[j * NU + u]) + tail[j * NU + u];
}

/* Streaming pass over one block of at most 8 A rows: per thread, walk this
 * thread's slice of output channels in pairs of B rows. The first block
 * streams B from DRAM; later blocks re-read the same per-thread B slice,
 * which for MoE tile sizes (~200 KB) is L2-resident. */
template <int MB>
void run_stream_m(int n_lo, int n_hi, int k, int k_pad,
                  const int8_t* a_int8,
                  const std::int8_t* b_rm, std::size_t ldb,
                  int32_t* c_int32, int n_pad) {
  int n0 = n_lo;
  /* NU=2 for MB <= 4 (8 accumulators), NU=1 above (register pressure). */
  if (MB <= 4) {
    for (; n0 + 2 <= n_hi; n0 += 2)
      dot_block_stream<MB, 2>(a_int8, k_pad, b_rm, ldb, n0, k, c_int32, n_pad);
  }
  for (; n0 < n_hi; ++n0)
    dot_block_stream<MB, 1>(a_int8, k_pad, b_rm, ldb, n0, k, c_int32, n_pad);
}

void run_stream(int m, int n_lo, int n_hi, int k, int k_pad,
                const int8_t* a_int8,
                const std::int8_t* b_rm, std::size_t ldb,
                int32_t* c_int32, int n_pad) {
  for (int mb = 0; mb < m; mb += 8) {
    const int8_t* a = a_int8 + (std::size_t)mb * k_pad;
    int32_t* c = c_int32 + (std::size_t)mb * n_pad;
    switch (m - mb >= 8 ? 8 : m - mb) {
      case 1: run_stream_m<1>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 2: run_stream_m<2>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 3: run_stream_m<3>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 4: run_stream_m<4>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 5: run_stream_m<5>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 6: run_stream_m<6>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      case 7: run_stream_m<7>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
      default: run_stream_m<8>(n_lo, n_hi, k, k_pad, a, b_rm, ldb, c, n_pad); break;
    }
  }
}

}  // namespace

void int8_rm_run(int m, int n, int k,
                 const std::int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth) {
  int m_pad = pad_up(m, M_STEP);
  int n_pad = pad_up(n, N_STEP);
  int k_pad = pad_up(k, K_STEP);

  const int8_t* a_int8  = static_cast<const int8_t*>(scratch_a);
  int32_t*      c_int32 = static_cast<int32_t*>(scratch_c);

  /* Streaming row-dot path for every m. It reads B rows directly from
   * their row-major memory (each B byte once per 8-row A block; later
   * blocks hit L2 for MoE tile sizes) instead of assembling 8×4 B subtiles
   * through a scalar stack buffer like the tiled path below did — that
   * pack cost ~30 cycles per 32 B bytes and dominated everything below
   * m ≈ 50. The tiled path is kept (unreachable) as reference.
   */
  {
    int rows_per_thread = (n + nth - 1) / nth;
    int n_lo = std::min(ith * rows_per_thread, n);
    int n_hi = std::min(n_lo + rows_per_thread, n);
    if (n_lo < n_hi)
      run_stream(m, n_lo, n_hi, k, k_pad, a_int8, b_rm, ldb, c_int32, n_pad);
    return;
  }

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
