/*
 * Stride-aware AMX INT8 GEMM — see int8_gemm_rm.h for the public ABI and
 * Stride.md §3 for the design rationale.
 *
 * Tile assignment (compare against int8_gemm.cpp's GemmKernel224Int8):
 *   tile 0,1 — B int8 row-major, TILE_M × TILE_K = 16 × 64 bytes,
 *              strided load with ldb (caller's row stride).
 *   tile 2,3 — A int8 in VNNI form, (TILE_K/4) × (TILE_N · 4) = 16 × 64 bytes.
 *              Layout matches what BufferBInt8 produces for B today — we
 *              re-use that buffer byte-for-byte (semantics: M lanes, not N).
 *   tile 4-7 — C^T int32, TILE_M × TILE_N · 4 = 16 × 64 bytes.
 *
 * Each `_tile_dpbssd` accumulates `dst[m, n_lane] += B[n_row, k] · A[m_lane, k]`,
 * i.e. C^T[n_begin+m, m_begin+n_lane]. The output is in transposed form;
 * the unpack pass rotates it back during scale application.
 */
#include "kernels/amx/int8_gemm_rm.h"

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstring>

#include "kernels/amx/amx_utils.h"
#include "kernels/amx/int8_buffers.h"
#include "kernels/amx/int8_gemm.h"
#include "kernels/amx/tile_config.h"

namespace cpu_gemm::kernels::amx {

namespace {

/* Tile shapes are bit-identical to the packed kernel — only the semantics
 * (which operand each tile holds) change. */
struct GemmKernel224Int8Rm {
  using dt       = int8_t;
  using output_t = int32_t;

  static constexpr int TILE_M    = 16;
  static constexpr int TILE_K    = 64;
  static constexpr int TILE_N    = 16;
  static constexpr int VNNI_BLK  = 4;

  static constexpr int M_STEP    = TILE_M * 2;   /* 32 */
  static constexpr int N_STEP    = TILE_N * 2;   /* 32 */
  static constexpr int K_STEP    = TILE_K;       /* 64 */
  static constexpr int N_BLOCK   = 64;
  static constexpr int K_BLOCK   = 3584;

  static_assert(M_STEP  == Int8KernelTraits::M_STEP);
  static_assert(N_STEP  == Int8KernelTraits::N_STEP);
  static_assert(K_STEP  == Int8KernelTraits::K_STEP);
  static_assert(N_BLOCK == Int8KernelTraits::N_BLOCK);
  static_assert(K_BLOCK == Int8KernelTraits::K_BLOCK);

  static void config() {
    TileConfig cfg;
    for (int i = 0; i < 2; i++) cfg.set_row_col(i, TILE_M,           TILE_K * sizeof(dt));
    for (int i = 2; i < 4; i++) cfg.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));
    for (int i = 4; i < 8; i++) cfg.set_row_col(i, TILE_M,           TILE_N * sizeof(output_t));
    cfg.load();
  }
};

/* Re-use BufferBInt8 byte-for-byte as the A-VNNI buffer. The byte layout
 * BufferBInt8 produces is exactly what AMX src2 expects; the per-row amax
 * → scale path inside `from_bf16` is the per-token quant we need for A. */
using BufferA_VNNI = BufferBInt8<GemmKernel224Int8Rm>;

/* C^T scratch — same blocked form as BufferCInt32 with M and N swapped:
 *   outer:   stripes of N_BLOCK rows (was M_BLOCK rows).
 *   inner:   for each m_begin (was n_begin) and n_within_block, a 32×32 int32
 *            sub-tile stored row-major with row=N coord, col=M coord, row
 *            stride = M_STEP int32s = 128 bytes.
 *
 * The byte stride of one sub-tile row therefore matches the byte stride
 * the packed kernel passes to store_c (N_STEP · sizeof(int32_t) = 128) —
 * so the AMX tile store/load helpers transfer over unchanged. */
struct BufferCTInt32 {
  int32_t* c;
  int      n_pad;
  int      max_m_pad;

  BufferCTInt32(int n_pad_, int max_m_pad_, void* ptr)
      : n_pad(n_pad_), max_m_pad(max_m_pad_) {
    assert(reinterpret_cast<std::intptr_t>(ptr) % 64 == 0);
    assert(n_pad % GemmKernel224Int8Rm::N_STEP == 0);
    assert(max_m_pad % GemmKernel224Int8Rm::M_STEP == 0);
    c = reinterpret_cast<int32_t*>(ptr);
  }

  int32_t* get_submat(int n_begin, int m_begin) const {
    using K = GemmKernel224Int8Rm;
    int n_block_begin = n_begin / K::N_BLOCK * K::N_BLOCK;
    int n_block_size  = std::min(K::N_BLOCK, n_pad - n_block_begin);
    int n_within      = n_begin - n_block_begin;
    return c + (std::size_t)max_m_pad * n_block_begin
             + (std::size_t)m_begin   * n_block_size
             + (std::size_t)n_within  * K::M_STEP;
  }
};

/* Tile-level helpers. The arithmetic mirrors GemmKernel224Int8::{load_c,
 * store_c, clean_c} — only the semantics change (the loaded/stored value
 * is C^T, not C). The byte strides land identically because M_STEP ==
 * N_STEP == 32. */
inline void rm_clean_c()                              { _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7); }
inline void rm_load_c (int32_t* c, std::size_t ldc) {
  using K = GemmKernel224Int8Rm;
  _tile_loadd(4, c,                                                                       ldc);
  _tile_loadd(5, offset_pointer(c, K::TILE_M * sizeof(int32_t)),                          ldc);
  _tile_loadd(6, offset_pointer(c, ldc * K::TILE_N),                                      ldc);
  _tile_loadd(7, offset_pointer(c, ldc * K::TILE_N + K::TILE_M * sizeof(int32_t)),        ldc);
}
inline void rm_store_c(int32_t* c, std::size_t ldc) {
  using K = GemmKernel224Int8Rm;
  _tile_stored(4, c,                                                                      ldc);
  _tile_stored(5, offset_pointer(c, K::TILE_M * sizeof(int32_t)),                         ldc);
  _tile_stored(6, offset_pointer(c, ldc * K::TILE_N),                                     ldc);
  _tile_stored(7, offset_pointer(c, ldc * K::TILE_N + K::TILE_M * sizeof(int32_t)),       ldc);
}

/* Inner kernel — process one (n_begin, m_begin) 32×32 tile group across the
 * full K-block. Loop body is structurally identical to the packed kernel's
 * `amx_kernel`; only the operand layouts swap. */
inline void amx_kernel_rm(int /*m*/, int /*n*/, int k,
                          int n_begin, int m_begin, int k_block_begin,
                          int32_t* ct_sub,
                          const int8_t* b_rm, std::size_t ldb,
                          BufferA_VNNI& aa) {
  using K = GemmKernel224Int8Rm;
  const std::size_t ldc_sub = K::M_STEP * sizeof(int32_t);

  if (k_block_begin == 0) {
    rm_clean_c();
  } else {
    rm_load_c(ct_sub, ldc_sub);
  }

  for (int k_begin = 0;
       k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
       k_begin += K::K_STEP) {
    /* src1: 32 N-rows × 64 K-bytes of B, read straight from caller memory.
     * Two 16-row tiles, strided by `ldb` bytes between consecutive N-rows. */
    const int8_t* b_ptr = b_rm + (std::size_t)n_begin * ldb
                                 + (std::size_t)(k_block_begin + k_begin);
    _tile_loadd(0, b_ptr,                                ldb);
    _tile_loadd(1, offset_pointer(b_ptr, ldb * K::TILE_N), ldb);

    /* src2: 32 M-lanes × 64 K-bytes of A, in VNNI-packed form. The two
     * tiles are stored back-to-back inside the BufferA sub-block (top half
     * = M lanes 0..15, bottom half = M lanes 16..31). */
    const int8_t* a_ptr = aa.get_submat(/*n_param*/ 0, /*k_param*/ 0,
                                        m_begin, k_block_begin + k_begin);
    _tile_loadd(2, a_ptr,                                          K::K_STEP * sizeof(int8_t));
    _tile_loadd(3, offset_pointer(a_ptr, K::K_STEP * K::TILE_M),   K::K_STEP * sizeof(int8_t));

    _tile_dpbssd(4, 0, 2);   /* C^T[N0..15,  M0..15 ] */
    _tile_dpbssd(5, 0, 3);   /* C^T[N0..15,  M16..31] */
    _tile_dpbssd(6, 1, 2);   /* C^T[N16..31, M0..15 ] */
    _tile_dpbssd(7, 1, 3);   /* C^T[N16..31, M16..31] */
  }

  rm_store_c(ct_sub, ldc_sub);
}

/* Outer driver — partition along N_BLOCK stripes across (ith, nth). For
 * each stripe we sweep all M for the same k-block, keeping B's K-block hot
 * in L2 (B is the large operand under the swap). */
void integer_mat_mul_rm(int m, int n_pad, int k,
                        const int8_t* b_rm, std::size_t ldb,
                        BufferA_VNNI& aa, BufferCTInt32& bct,
                        int ith, int nth) {
  using K = GemmKernel224Int8Rm;
  int n_blocks = (n_pad + K::N_BLOCK - 1) / K::N_BLOCK;
  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_start = blk * K::N_BLOCK;
    int n_end   = std::min(n_pad, n_start + K::N_BLOCK);
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
          int32_t* ct_sub = bct.get_submat(n_begin, m_begin);
          amx_kernel_rm(m, n_pad, k, n_begin, m_begin, k_block_begin,
                        ct_sub, b_rm, ldb, aa);
        }
      }
    }
  }
}

}  // anonymous namespace

/* ------------------------------------------------------------------ */
/* Public entry points used by the dispatcher.                        */
/* ------------------------------------------------------------------ */

__attribute__((noinline))
void int8_rm_tile_config_init() {
  static thread_local bool xperm_ok = false;
  if (!xperm_ok) {
    if (!enable_amx()) return;
    xperm_ok = true;
  }
  __asm__ volatile("" ::: "memory");
  GemmKernel224Int8Rm::config();
  __asm__ volatile("" ::: "memory");
}

void int8_rm_pack_a_bf16(int m, int k,
                         const cg_bf16_t* a_rm,
                         void* scratch_a) {
  using K = GemmKernel224Int8Rm;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  BufferA_VNNI aa(max_m_pad, k, scratch_a);
  /* BufferBInt8::from_bf16 does the per-row amax → scale + VNNI pack we
   * need. The "n_real" arg is the real row count = m; padded rows zero out. */
  aa.from_bf16(a_rm, /*n_real=*/m, /*ith=*/0, /*nth=*/1);
}

void int8_rm_run(int m, int n, int k,
                 const int8_t* b_rm, std::size_t ldb,
                 void* scratch_a, void* scratch_c,
                 int ith, int nth) {
  using K = GemmKernel224Int8Rm;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  int n_pad     = int8_pad_up(n, K::N_STEP);
  /* Dispatcher's eligibility predicate guarantees n == n_pad (caller's B is
   * padded to N_STEP), so the strided _tile_loadd never reads past
   * n_pad-1 — i.e. never past the caller's allocation. */
  assert(n == n_pad);

  int8_rm_tile_config_init();

  BufferA_VNNI  aa (max_m_pad, k, scratch_a);
  BufferCTInt32 bct(n_pad, max_m_pad, scratch_c);

  integer_mat_mul_rm(m, n_pad, k, b_rm, ldb, aa, bct, ith, nth);
}

void int8_rm_unpack_transposed(int m, int n,
                               const float* a_scales,
                               const float* b_scales,
                               const void* scratch_c,
                               float alpha,
                               float beta,
                               float* c_rm, std::size_t ldc,
                               int ith, int nth) {
  using K = GemmKernel224Int8Rm;
  int max_m_pad = int8_pad_up(m, K::M_STEP);
  int n_pad     = int8_pad_up(n, K::N_STEP);

  const int32_t* ct_blk = reinterpret_cast<const int32_t*>(scratch_c);
  int n_blocks = (n_pad + K::N_BLOCK - 1) / K::N_BLOCK;

  /* Scalar implementation of the 32×32 implicit-transpose + scale-apply.
   * Per Stride.md §3.6 we land this first; SIMD-ising the transpose is a
   * follow-up under the same unit tests. */
  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_block_begin = blk * K::N_BLOCK;
    int n_block_end   = std::min(n_pad, n_block_begin + K::N_BLOCK);
    int n_block_size  = n_block_end - n_block_begin;

    for (int n_within = 0; n_within < n_block_size; n_within += K::N_STEP) {
      int n_sub_begin = n_block_begin + n_within;
      int n_valid     = std::min(K::N_STEP, n - n_sub_begin);
      if (n_valid <= 0) continue;

      for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
        const int32_t* src = ct_blk
            + (std::size_t)max_m_pad * n_block_begin
            + (std::size_t)m_begin   * n_block_size
            + (std::size_t)n_within  * K::M_STEP;
        int m_valid = std::min(K::M_STEP, m - m_begin);
        /* Sub-tile layout: row = N (j), col = M (i), row stride = M_STEP. */
        for (int j = 0; j < n_valid; ++j) {
          float b_scale = b_scales[n_sub_begin + j];
          const int32_t* row = src + (std::size_t)j * K::M_STEP;
          for (int i = 0; i < m_valid; ++i) {
            float v = a_scales[m_begin + i] * b_scale * (float)row[i];
            float* dst = c_rm + (std::size_t)(m_begin + i) * ldc + n_sub_begin + j;
            float prev = beta == 0.0f ? 0.0f : *dst * beta;
            *dst = prev + alpha * v;
          }
        }
      }
    }
  }
}

}  // namespace cpu_gemm::kernels::amx

#endif  /* CPU_GEMM_HAS_AMX */
