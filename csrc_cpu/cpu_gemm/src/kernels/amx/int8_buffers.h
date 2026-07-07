/*
 * Packed-block buffer types for the AMX INT8 GEMM.
 *
 * Ported from operators/amx/la/amx_buffers.hpp::BufferAImpl and the
 * inline BufferB inside operators/amx/la/amx_kernels.hpp::GemmKernel224Int8.
 *
 * Compared to BF16:
 *   - A is int8 (per-row scale d[max_m]) — dynamic quantization of BF16
 *     input on the fly.
 *   - B is int8 (per-output-channel scale d[n]).
 *   - C accumulates as int32, then gets multiplied by (a_scale * b_scale)
 *     to recover FP32.
 *   - K_STEP = 64 (4 bytes per VNNI element vs 2 for BF16), N_BLOCK = 64.
 */
#ifndef CPU_GEMM_KERNELS_AMX_INT8_BUFFERS_H
#define CPU_GEMM_KERNELS_AMX_INT8_BUFFERS_H

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cpu_gemm/types.h"
#include "kernels/amx/amx_utils.h"

namespace cpu_gemm::kernels::amx {

/* BufferA — int8 activation buffer with per-row scale. */
template <typename K>
struct BufferAInt8 {
  int8_t* a;
  float*  d;
  int     max_m;
  int     k;

  static constexpr int M_STEP  = K::M_STEP;
  static constexpr int K_STEP  = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static std::size_t required_size(int max_m, int k) {
    return sizeof(int8_t) * (std::size_t)max_m * k + sizeof(float) * max_m;
  }

  BufferAInt8(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    a = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(a + (std::size_t)max_m * k);
  }

  /* Quantize row-major BF16 [m, k] to int8 + per-row float scale, then
   * pack into blocked layout [k_blocks][m_blocks][k_steps][M_STEP][K_STEP].
   * Single-threaded (matches the reference's `ith == 0 && nth == 1`). */
  void from_bf16(int m, const cg_bf16_t* src) {
    assert(m <= max_m);

    /* Phase 1: per-row amax → scale d[i] = amax / 127. */
    for (int mi = 0; mi < m; ++mi) {
      __m512 amax0 = _mm512_setzero_ps();
      __m512 amax1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j += 32) {
        __m512i src32 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(src + (std::size_t)mi * k + j));
        __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(src32)), 16));
        __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(src32, 1)), 16));
        amax0 = _mm512_max_ps(amax0, _mm512_abs_ps(f0));
        amax1 = _mm512_max_ps(amax1, _mm512_abs_ps(f1));
      }
      float amax = _mm512_reduce_max_ps(_mm512_max_ps(amax0, amax1));
      d[mi] = amax / 127.0f;
    }
    /* d[m..max_m-1] are leftover pad rows; harmless but zero out for
     * deterministic accumulator behavior. */
    for (int mi = m; mi < max_m; ++mi) d[mi] = 0.0f;

    /* Phase 2: quantize & pack into blocked layout. */
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; ++i) {
            float inv_d = d[m_begin + i] != 0.0f ? 1.0f / d[m_begin + i] : 0.0f;
            __m512 id = _mm512_set1_ps(inv_d);
            int8_t* dst = a + (std::size_t)k_block_begin * m_block_size
                            + (std::size_t)m_begin * k_block_size
                            + (std::size_t)k_begin * M_STEP
                            + (std::size_t)i * K_STEP;
            /* Load 64 BF16 values, convert to FP32 in 4 vectors of 16,
             * quantize, pack to int8. */
            const __m512i* src_row = reinterpret_cast<const __m512i*>(
                src + (std::size_t)(m_begin + i) * k + k_block_begin + k_begin);
            __m512i v0 = _mm512_loadu_si512(src_row);
            __m512i v1 = _mm512_loadu_si512(src_row + 1);
            __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v0)), 16));
            __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v0, 1)), 16));
            __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v1)), 16));
            __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v1, 1)), 16));
            __m128i s0 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f0, id)));
            __m128i s1 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f1, id)));
            __m128i s2 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f2, id)));
            __m128i s3 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f3, id)));
            _mm_store_si128(reinterpret_cast<__m128i*>(dst),       s0);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + 16),  s1);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + 32),  s2);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + 48),  s3);
          }
        }
      }
    }
  }

  int8_t* get_submat(int m, int /*k_param*/, int m_begin, int k_begin) const {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + (std::size_t)k_block_begin * m_block_size
             + (std::size_t)m_begin * k_block_size
             + (std::size_t)k_begin * M_STEP;
  }

  const float* get_scale(int m_begin) const { return d + m_begin; }
};

/* BufferB — int8 weight buffer with per-output-channel scale. The packing
 * mirrors BF16's BufferBBF16: blocked by N_BLOCK then K_BLOCK, with each
 * (N_STEP × K_STEP) tile transposed into VNNI form. */
template <typename K>
struct BufferBInt8 {
  int8_t* b;
  float*  d;
  int     n;
  int     k;

  static constexpr int N_STEP  = K::N_STEP;
  static constexpr int K_STEP  = K::K_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static constexpr int TILE_N  = K::TILE_N;

  static std::size_t required_size(int n, int k) {
    return sizeof(int8_t) * (std::size_t)n * k + sizeof(float) * n;
  }

  BufferBInt8(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % K_STEP == 0);
    b = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(b + (std::size_t)n * k);
  }

  /* Pack BF16 weights [n_real, k] → INT8 + per-channel scale, with the AMX
   * VNNI tile transpose. Multi-threaded across N_BLOCK chunks. `n` (the
   * buffer's padded extent) may exceed n_real; padded rows are zero-filled,
   * never read from `src`. */
  void from_bf16(const cg_bf16_t* src, int n_real, int ith, int nth) {
    int n_blocks = (n + N_BLOCK - 1) / N_BLOCK;
    for (int blk = 0; blk < n_blocks; ++blk) {
      if (blk % nth != ith) continue;
      int n_block_begin = blk * N_BLOCK;
      int n_block_end   = std::min(n, n_block_begin + N_BLOCK);
      int n_block_size  = n_block_end - n_block_begin;
      pack_one_block(src + (std::size_t)n_block_begin * k, n_block_begin, n_block_size, n_real);
    }
  }

  /* In-place INT8 pack: caller provides pre-quantized int8 weights
   * [n_real, k] row-major and matching per-channel scales [n_real]. Padded
   * rows/scales beyond n_real are zero-filled, never read. */
  void from_int8(const int8_t* src, const float* scales, int n_real, int ith, int nth) {
    int n_blocks = (n + N_BLOCK - 1) / N_BLOCK;
    for (int blk = 0; blk < n_blocks; ++blk) {
      if (blk % nth != ith) continue;
      int n_block_begin = blk * N_BLOCK;
      int n_block_end   = std::min(n, n_block_begin + N_BLOCK);
      int n_block_size  = n_block_end - n_block_begin;
      pack_one_block_int8(src + (std::size_t)n_block_begin * k,
                          scales + n_block_begin, n_block_begin, n_block_size, n_real);
    }
  }

  int8_t* get_submat(int /*n_param*/, int /*k_param*/, int n_begin, int k_begin) const {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return b + (std::size_t)n_block_begin * k
             + (std::size_t)k_block_begin * n_block_size
             + (std::size_t)n_begin * k_block_size
             + (std::size_t)k_begin * N_STEP;
  }

  const float* get_scale(int n_begin) const { return d + n_begin; }

 private:
  /* Quantize one N_BLOCK of BF16 weights and pack with VNNI transpose.
   * Two-phase: scales first, then quantize+pack+transpose. Rows beyond
   * n_real are padding: their scale is 0 and their packed data is zeroed,
   * and the source is never read for them. */
  void pack_one_block(const cg_bf16_t* src_block, int n_block_begin, int n_block_size, int n_real) {
    /* Real rows in this block; the rest are padding (no source to read). */
    int real_in_block = std::max(0, std::min(n_block_size, n_real - n_block_begin));
    for (int i = real_in_block; i < n_block_size; ++i) d[n_block_begin + i] = 0.0f;
    /* Phase 1: per-row amax → per-channel scale (real rows only). */
    for (int i = 0; i < real_in_block; ++i) {
      __m512 amax0 = _mm512_setzero_ps();
      __m512 amax1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j += 32) {
        __m512i v = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(src_block + (std::size_t)i * k + j));
        __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)), 16));
        __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), 16));
        amax0 = _mm512_max_ps(amax0, _mm512_abs_ps(f0));
        amax1 = _mm512_max_ps(amax1, _mm512_abs_ps(f1));
      }
      float amax = _mm512_reduce_max_ps(_mm512_max_ps(amax0, amax1));
      d[n_block_begin + i] = amax / 127.0f;
    }
    pack_tiles(src_block, n_block_begin, n_block_size, n_real, /*src_is_int8=*/false, nullptr);
  }

  void pack_one_block_int8(const int8_t* src_block, const float* scales_block,
                           int n_block_begin, int n_block_size, int n_real) {
    /* Copy scales as-is for real rows; zero the padding. Use an explicit
     * loop bound (not a per-element ternary) so the compiler cannot
     * auto-vectorize this into an unconditional load of scales_block past
     * n_real, which would read off the end of the caller's scale array. */
    int real_in_block = std::max(0, std::min(n_block_size, n_real - n_block_begin));
    for (int i = 0; i < real_in_block; ++i) d[n_block_begin + i] = scales_block[i];
    for (int i = real_in_block; i < n_block_size; ++i) d[n_block_begin + i] = 0.0f;
    pack_tiles(src_block, n_block_begin, n_block_size, n_real, /*src_is_int8=*/true, scales_block);
  }

  template <typename SrcT>
  void pack_tiles(const SrcT* src_block, int n_block_begin, int n_block_size, int n_real,
                  bool src_is_int8, const float* /*scales_block*/) {
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < N_STEP; ++i) {
            int8_t* dst = b + (std::size_t)n_block_begin * k
                            + (std::size_t)k_block_begin * n_block_size
                            + (std::size_t)n_begin * k_block_size
                            + (std::size_t)k_begin * N_STEP
                            + (std::size_t)i * K_STEP;
            if (n_block_begin + n_begin + i >= n_real) {
              /* Padding row: zero-fill, never read source. */
              std::memset(dst, 0, (std::size_t)K_STEP);
            } else if constexpr (std::is_same<SrcT, cg_bf16_t>::value) {
              float inv_d = d[n_block_begin + n_begin + i] != 0.0f
                                ? 1.0f / d[n_block_begin + n_begin + i]
                                : 0.0f;
              __m512 id = _mm512_set1_ps(inv_d);
              const __m512i* src_row = reinterpret_cast<const __m512i*>(
                  reinterpret_cast<const cg_bf16_t*>(src_block)
                  + (std::size_t)(n_begin + i) * k + k_block_begin + k_begin);
              __m512i v0 = _mm512_loadu_si512(src_row);
              __m512i v1 = _mm512_loadu_si512(src_row + 1);
              __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
                  _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v0)), 16));
              __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
                  _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v0, 1)), 16));
              __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(
                  _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v1)), 16));
              __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(
                  _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v1, 1)), 16));
              __m128i s0 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f0, id)));
              __m128i s1 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f1, id)));
              __m128i s2 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f2, id)));
              __m128i s3 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(f3, id)));
              _mm_store_si128(reinterpret_cast<__m128i*>(dst),      s0);
              _mm_store_si128(reinterpret_cast<__m128i*>(dst + 16), s1);
              _mm_store_si128(reinterpret_cast<__m128i*>(dst + 32), s2);
              _mm_store_si128(reinterpret_cast<__m128i*>(dst + 48), s3);
            } else {
              /* int8 source — pure copy. */
              const int8_t* src_row = reinterpret_cast<const int8_t*>(src_block)
                                    + (std::size_t)(n_begin + i) * k + k_block_begin + k_begin;
              std::memcpy(dst, src_row, (std::size_t)K_STEP);
            }
          }
          /* AMX VNNI tile transpose — same two-16x16 pattern as BF16
           * (the tile is treated as int32 lanes, so each 16x16 of int8
           * within a 16x64 transpose lane works identically). */
          auto* tile0 = reinterpret_cast<__m512i*>(
              b + (std::size_t)n_block_begin * k
                + (std::size_t)k_block_begin * n_block_size
                + (std::size_t)n_begin * k_block_size
                + (std::size_t)k_begin * N_STEP);
          auto* tile1 = reinterpret_cast<__m512i*>(
              b + (std::size_t)n_block_begin * k
                + (std::size_t)k_block_begin * n_block_size
                + (std::size_t)n_begin * k_block_size
                + (std::size_t)k_begin * N_STEP
                + (std::size_t)TILE_N * K_STEP);
          transpose_16x16_32bit(tile0);
          transpose_16x16_32bit(tile1);
          (void)src_is_int8;
        }
      }
    }
  }
};

/* BufferC — int32 accumulator buffer in blocked layout, scaled to FP32 by
 * apply_scale during unpack. */
template <typename K>
struct BufferCInt32 {
  int32_t* c;
  int      max_m;
  int      n;

  static constexpr int M_STEP  = K::M_STEP;
  static constexpr int N_STEP  = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static std::size_t required_size(int max_m, int n) {
    return sizeof(int32_t) * (std::size_t)max_m * n;
  }

  BufferCInt32(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<int32_t*>(ptr);
  }

  int32_t* get_submat(int m, int /*n_param*/, int m_begin, int n_begin) const {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + (std::size_t)m_block_size * n_block_begin
             + (std::size_t)m_begin * n_block_size
             + (std::size_t)n_begin * M_STEP;
  }
};

}  // namespace cpu_gemm::kernels::amx

#endif
#endif
