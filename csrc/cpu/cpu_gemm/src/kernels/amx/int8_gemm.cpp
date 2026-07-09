/*
 * AMX INT8 GEMM kernel.
 *
 * Direct port of operators/amx/la/amx_kernels.hpp::GemmKernel224Int8 with
 * the buffer wrappers and the integer_mat_mul driver inlined.
 *
 * Tile layout:
 *   tile 0,1 — A int8 tiles,  TILE_M × TILE_K = 16 × 64 bytes
 *   tile 2,3 — B int8 tiles in VNNI form,
 *              (TILE_K/4) × (TILE_N * 4) = 16 × 64 bytes
 *   tile 4..7 — C int32 tiles, TILE_M × TILE_N * 4 = 16 × 64 bytes
 *
 * Inner contraction is _tile_dpbssd → signed × signed → int32 accumulator.
 */
#include "kernels/amx/int8_gemm.h"

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "kernels/amx/amx_utils.h"
#include "kernels/amx/int8_buffers.h"
#include "kernels/amx/tile_config.h"

namespace cpu_gemm::kernels::amx {

struct GemmKernel224Int8 {
  using dt       = int8_t;
  using output_t = int32_t;

  static constexpr int TILE_M    = 16;
  static constexpr int TILE_K    = 64;
  static constexpr int TILE_N    = 16;
  static constexpr int VNNI_BLK  = 4;

  static constexpr int M_STEP    = TILE_M * 2;   // 32
  static constexpr int N_STEP    = TILE_N * 2;   // 32
  static constexpr int K_STEP    = TILE_K;       // 64
  static constexpr int N_BLOCK   = 64;
  static constexpr int K_BLOCK   = 3584;

  static_assert(M_STEP  == Int8KernelTraits::M_STEP);
  static_assert(N_STEP  == Int8KernelTraits::N_STEP);
  static_assert(K_STEP  == Int8KernelTraits::K_STEP);
  static_assert(N_BLOCK == Int8KernelTraits::N_BLOCK);
  static_assert(K_BLOCK == Int8KernelTraits::K_BLOCK);

  using BufferA = BufferAInt8<GemmKernel224Int8>;
  using BufferB = BufferBInt8<GemmKernel224Int8>;
  using BufferC = BufferCInt32<GemmKernel224Int8>;

  static void config() {
    TileConfig cfg;
    for (int i = 0; i < 2; i++) cfg.set_row_col(i, TILE_M,           TILE_K * sizeof(dt));
    for (int i = 2; i < 4; i++) cfg.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));
    for (int i = 4; i < 8; i++) cfg.set_row_col(i, TILE_M,           TILE_N * sizeof(output_t));
    cfg.load();
  }

  static void load_a(const dt* a, std::size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }
  static void load_b(const dt* b, std::size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }
  static void clean_c() { _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7); }
  static void load_c(output_t* c, std::size_t ldc) {
    _tile_loadd(4, c,                                                          ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)),               ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M),                            ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),ldc);
  }
  static void store_c(output_t* c, std::size_t ldc) {
    _tile_stored(4, c,                                                          ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)),               ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M),                            ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),ldc);
  }
  static void run_tile() {
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
  }

  /* Process one (m_begin, n_begin) tile group across the full k. */
  static void amx_kernel(int /*m*/, int /*n*/, int k,
                         int m_begin, int n_begin, int k_block_begin,
                         int32_t* c_sub, BufferA& ba, BufferB& bb) {
    if (k_block_begin == 0) {
      clean_c();
    } else {
      load_c(c_sub, N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      load_a(ba.get_submat(/*m*/ ba.max_m, k, m_begin, k_block_begin + k_begin),
             K_STEP * sizeof(dt));
      load_b(bb.get_submat(/*n*/ bb.n,     k, n_begin, k_block_begin + k_begin),
             K_STEP * sizeof(dt));
      run_tile();
    }
    store_c(c_sub, N_STEP * sizeof(int32_t));
  }
};

/* The integer mat-mul driver (port of integer_mat_mul). C is int32; the
 * (alpha, beta, FP32) combining happens in int8_unpack. */
static void integer_mat_mul(int m, int n, int k,
                            GemmKernel224Int8::BufferA& ba,
                            GemmKernel224Int8::BufferB& bb,
                            GemmKernel224Int8::BufferC& bc,
                            int ith, int nth) {
  using K = GemmKernel224Int8;
  int n_blocks = (n + K::N_BLOCK - 1) / K::N_BLOCK;
  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_start = blk * K::N_BLOCK;
    int n_end   = std::min(n, n_start + K::N_BLOCK);
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
      for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
        for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
          int32_t* c_sub = bc.get_submat(m, n, m_begin, n_begin);
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c_sub, ba, bb);
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Public entry points used by the dispatcher.                        */
/* ------------------------------------------------------------------ */

__attribute__((noinline))
void int8_tile_config_init() {
  static thread_local bool xperm_ok = false;
  if (!xperm_ok) {
    if (!enable_amx()) return;
    xperm_ok = true;
  }
  __asm__ volatile("" ::: "memory");
  GemmKernel224Int8::config();
  __asm__ volatile("" ::: "memory");
}

void int8_pack_a_bf16(int m, int k,
                      const cg_bf16_t* a_rm,
                      void* scratch_a) {
  using K = GemmKernel224Int8;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  GemmKernel224Int8::BufferA ba(max_m_pad, k, scratch_a);
  ba.from_bf16(m, a_rm);
}

void int8_pack_b_bf16(int n, int k,
                      const cg_bf16_t* b_rm,
                      void* scratch_b,
                      int ith, int nth) {
  using K = GemmKernel224Int8;
  int n_pad = int8_pad_up(n, K::N_STEP);
  GemmKernel224Int8::BufferB bb(n_pad, k, scratch_b);
  bb.from_bf16(b_rm, n, ith, nth);
}

void int8_pack_b_int8(int n, int k,
                      const int8_t* b_int8,
                      const float* b_scales,
                      void* scratch_b,
                      int ith, int nth) {
  using K = GemmKernel224Int8;
  int n_pad = int8_pad_up(n, K::N_STEP);
  GemmKernel224Int8::BufferB bb(n_pad, k, scratch_b);
  bb.from_int8(b_int8, b_scales, n, ith, nth);
}

void int8_run(int m, int n, int k,
              void* scratch_a, void* scratch_b, void* scratch_c,
              int ith, int nth) {
  using K = GemmKernel224Int8;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  int n_pad     = int8_pad_up(n, K::N_STEP);

  int8_tile_config_init();

  GemmKernel224Int8::BufferA ba(max_m_pad, k, scratch_a);
  GemmKernel224Int8::BufferB bb(n_pad, k, scratch_b);
  GemmKernel224Int8::BufferC bc(max_m_pad, n_pad, scratch_c);

  integer_mat_mul(m, n_pad, k, ba, bb, bc, ith, nth);
}

std::size_t int8_a_scales_offset(int max_m_pad, int k) {
  return sizeof(int8_t) * (std::size_t)max_m_pad * k;
}
std::size_t int8_b_scales_offset(int n_pad, int k) {
  return sizeof(int8_t) * (std::size_t)n_pad * k;
}

/* Bytes needed for a pre-packed B for an [n, k] weight (n padded up to
 * N_STEP). The buffer is one int8 block followed by the FP32 scale vector. */
std::size_t int8_packed_b_size(int n, int k) {
  using K = GemmKernel224Int8;
  int n_pad = int8_pad_up(n, K::N_STEP);
  return sizeof(int8_t) * (std::size_t)n_pad * k + sizeof(float) * (std::size_t)n_pad;
}

/* Build a pre-packed B from caller-supplied int8 weights + scales. */
void int8_pack_b_int8_offline(int n, int k,
                              const int8_t* b_int8,
                              const float* b_scales,
                              void* dst) {
  using K = GemmKernel224Int8;
  int n_pad = int8_pad_up(n, K::N_STEP);
  GemmKernel224Int8::BufferB bb(n_pad, k, dst);
  bb.from_int8(b_int8, b_scales, n, /*ith=*/0, /*nth=*/1);
}

/* Unpack int32 accumulator → FP32 with per-row × per-channel scales. */
void int8_unpack_explicit(int m, int n,
                          const float* a_scales,
                          const float* b_scales,
                          const void* scratch_c,
                          float alpha,
                          float beta, float* c_rm, std::size_t ldc,
                          int ith, int nth) {
  using K = GemmKernel224Int8;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  int n_pad     = int8_pad_up(n, K::N_STEP);

  const int32_t* c_blk = reinterpret_cast<const int32_t*>(scratch_c);
  int n_blocks = (n_pad + K::N_BLOCK - 1) / K::N_BLOCK;

  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_block_begin = blk * K::N_BLOCK;
    int n_block_end   = std::min(n_pad, n_block_begin + K::N_BLOCK);
    int n_block_size  = n_block_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += K::N_STEP) {
        for (int i = 0; i < K::M_STEP && m_begin + i < m; ++i) {
          float a_scale = a_scales[m_begin + i];
          const int32_t* src = c_blk + (std::size_t)max_m_pad * n_block_begin
                                    + (std::size_t)m_begin * n_block_size
                                    + (std::size_t)n_begin * K::M_STEP
                                    + (std::size_t)i * K::N_STEP;
          int n_valid = std::min(K::N_STEP, n - (n_block_begin + n_begin));
          if (n_valid <= 0) continue;
          float* dst = c_rm + (std::size_t)(m_begin + i) * ldc + n_block_begin + n_begin;
          const float* bs = b_scales + n_block_begin + n_begin;
          for (int j = 0; j < n_valid; ++j) {
            float v = a_scale * bs[j] * (float)src[j];
            float prev = beta == 0.0f ? 0.0f : dst[j] * beta;
            dst[j] = prev + alpha * v;
          }
        }
      }
    }
  }
}

}  // namespace cpu_gemm::kernels::amx

#endif  // CPU_GEMM_HAS_AMX
