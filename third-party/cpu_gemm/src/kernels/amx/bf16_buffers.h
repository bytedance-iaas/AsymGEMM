/*
 * Packed-block buffer types for the AMX BF16 GEMM.
 *
 * Ported from operators/amx/la/amx_raw_buffers.hpp::BufferABF16Impl /
 * BufferBBF16Impl / BufferCFP32Impl.
 *
 * Layouts (preserved bit-for-bit):
 *   A: [k_blocks][m_blocks][k_steps][M_STEP][K_STEP]  (5D blocked)
 *   B: [n_blocks][k_blocks][n_steps][k_steps][N_STEP][K_STEP] with each
 *      (N_STEP × K_STEP) tile internally transposed into AMX VNNI form
 *      (two 16x16 32-bit transposes per tile).
 *   C: [n_blocks][m_blocks][n_steps][M_STEP][N_STEP]  (5D blocked)
 *
 * Templated on the kernel K so that block sizes follow the kernel.
 */
#ifndef CPU_GEMM_KERNELS_AMX_BF16_BUFFERS_H
#define CPU_GEMM_KERNELS_AMX_BF16_BUFFERS_H

#if defined(CPU_GEMM_HAS_AMX)

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cpu_gemm/types.h"
#include "kernels/amx/amx_utils.h"
#include "kernels/amx/tile_config.h"

namespace cpu_gemm::kernels::amx {

template <typename K>
struct BufferABF16 {
  cg_bf16_t* a;
  int        max_m;
  int        k;

  static constexpr int M_STEP  = K::M_STEP;
  static constexpr int K_STEP  = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static std::size_t required_size(int max_m, int k) {
    return sizeof(cg_bf16_t) * (std::size_t)max_m * k;
  }

  BufferABF16(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    a = reinterpret_cast<cg_bf16_t*>(ptr);
  }

  /* Pack row-major [m, k] BF16 → blocked layout. Single-threaded; the
   * dispatcher controls outer parallelism. */
  void from_mat(int m, const cg_bf16_t* src) {
    assert(m <= max_m);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; ++i) {
            auto* s = reinterpret_cast<const __m512i*>(
                src + (std::size_t)(m_begin + i) * k + k_block_begin + k_begin);
            auto* d = reinterpret_cast<__m512i*>(
                a + (std::size_t)k_block_begin * m_block_size + (std::size_t)m_begin * k_block_size
                + (std::size_t)k_begin * M_STEP + (std::size_t)i * K_STEP);
            avx512_copy_32xbf16(s, d);
          }
        }
      }
    }
  }

  cg_bf16_t* get_submat(int m, int /*k*/, int m_begin, int k_begin) const {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + (std::size_t)k_block_begin * m_block_size
             + (std::size_t)m_begin * k_block_size
             + (std::size_t)k_begin * M_STEP;
  }
};

template <typename K>
struct BufferBBF16 {
  cg_bf16_t* b;
  int        n;
  int        k;

  static constexpr int N_STEP  = K::N_STEP;
  static constexpr int K_STEP  = K::K_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static constexpr int TILE_N  = K::TILE_N;

  static std::size_t required_size(int n, int k) {
    return sizeof(cg_bf16_t) * (std::size_t)n * k;
  }

  BufferBBF16(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % K_STEP == 0);
    b = reinterpret_cast<cg_bf16_t*>(ptr);
  }

  /* Pack row-major [n_real, k] BF16 → AMX-tile layout. May be threaded across
   * n_blocks. `n` (== n_pad, the buffer's padded extent) may exceed n_real;
   * source rows [n_real, n) do not exist in `src`, so they are zero-filled in
   * the packed buffer rather than read (their output columns are discarded
   * downstream). Reading them would walk past the caller's B buffer. */
  void from_mat(const cg_bf16_t* src, int n_real, int ith, int nth) {
    /* Split the n dimension by N_BLOCK like the original implementation. */
    int n_blocks = (n + N_BLOCK - 1) / N_BLOCK;
    const __m512i zero = _mm512_setzero_si512();
    for (int blk = 0; blk < n_blocks; ++blk) {
      if (blk % nth != ith) continue;
      int n_block_begin = blk * N_BLOCK;
      int n_block_end   = std::min(n, n_block_begin + N_BLOCK);
      int n_block_size  = n_block_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; ++i) {
              auto* d = reinterpret_cast<__m512i*>(
                  b + (std::size_t)n_block_begin * k
                    + (std::size_t)k_block_begin * n_block_size
                    + (std::size_t)n_begin * k_block_size
                    + (std::size_t)k_begin * N_STEP
                    + (std::size_t)i * K_STEP);
              int row = n_block_begin + n_begin + i;
              if (row < n_real) {
                auto* s = reinterpret_cast<const __m512i*>(
                    src + (std::size_t)row * k + k_block_begin + k_begin);
                avx512_copy_32xbf16(s, d);
              } else {
                _mm512_storeu_si512(d, zero);
              }
            }
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
          }
        }
      }
    }
  }

  cg_bf16_t* get_submat(int /*n_param*/, int /*k_param*/, int n_begin, int k_begin) const {
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
};

template <typename K>
struct BufferCFP32 {
  float* c;
  int    max_m;
  int    n;

  static constexpr int M_STEP  = K::M_STEP;
  static constexpr int N_STEP  = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static std::size_t required_size(int max_m, int n) {
    return sizeof(float) * (std::size_t)max_m * n;
  }

  BufferCFP32(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    c = reinterpret_cast<float*>(ptr);
  }

  /* Unpack blocked C → row-major [m, n] FP32 in-place at `dst`. */
  void to_mat(int m, float* dst, int ith, int nth) {
    assert(m <= max_m);
    int n_blocks = (n + N_BLOCK - 1) / N_BLOCK;
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int blk = 0; blk < n_blocks; ++blk) {
      if (blk % nth != ith) continue;
      int n_block_begin = blk * N_BLOCK;
      int n_block_end   = std::min(n, n_block_begin + N_BLOCK);
      int n_block_size  = n_block_end - n_block_begin;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; ++i) {
            const float* src = c + (std::size_t)m_block_size * n_block_begin
                                 + (std::size_t)m_begin * n_block_size
                                 + (std::size_t)n_begin * M_STEP
                                 + (std::size_t)i * N_STEP;
            float* row_dst = dst + (std::size_t)(m_begin + i) * n + n_block_begin + n_begin;
            std::memcpy(row_dst, src, sizeof(float) * N_STEP);
          }
        }
      }
    }
  }

  float* get_submat(int m, int /*n*/, int m_begin, int n_begin) const {
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

#endif  // CPU_GEMM_HAS_AMX
#endif
