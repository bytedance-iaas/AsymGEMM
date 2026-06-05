/*
 * AMX BF16 GEMM kernel.
 *
 * Direct port of operators/amx/la/amx_raw_kernels.hpp::GemmKernel224BF16
 * with the tile op helpers (load_a/b, run_tile, clean_c/load_c/store_c)
 * preserved verbatim.
 *
 * The original code embedded these in a struct templated by buffer types
 * and shared with the FP8 / FP4 kernels. Here we narrow it to the BF16
 * path; the abstraction stays the same shape so wiring up FP8 later is
 * one struct definition and one dispatcher arm.
 *
 * Compile flags: -mamx-bf16 -mamx-tile -mavx512f -mavx512bw (set by
 * CMakeLists.txt on the cpu_gemm_amx object library).
 */
#include "kernels/amx/bf16_gemm.h"

#if defined(CPU_GEMM_HAS_AMX)

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "kernels/amx/amx_utils.h"
#include "kernels/amx/bf16_buffers.h"
#include "kernels/amx/tile_config.h"

namespace cpu_gemm::kernels::amx {

/* ------------------------------------------------------------------ */
/* Kernel — tile sizes match Bf16KernelTraits.                        */
/* ------------------------------------------------------------------ */
struct GemmKernel224BF16 {
  using dt       = cg_bf16_t;
  using output_t = float;

  static constexpr int TILE_M    = 16;
  static constexpr int TILE_K    = 32;
  static constexpr int TILE_N    = 16;
  static constexpr int VNNI_BLK  = 2;

  static constexpr int M_STEP    = TILE_M * 2;   // 32
  static constexpr int N_STEP    = TILE_N * 2;   // 32
  static constexpr int K_STEP    = TILE_K;       // 32
  static constexpr int N_BLOCK   = 256;
  static constexpr int K_BLOCK   = 1792;

  static_assert(M_STEP  == Bf16KernelTraits::M_STEP);
  static_assert(N_STEP  == Bf16KernelTraits::N_STEP);
  static_assert(K_STEP  == Bf16KernelTraits::K_STEP);
  static_assert(N_BLOCK == Bf16KernelTraits::N_BLOCK);
  static_assert(K_BLOCK == Bf16KernelTraits::K_BLOCK);

  using BufferA = BufferABF16<GemmKernel224BF16>;
  using BufferB = BufferBBF16<GemmKernel224BF16>;
  using BufferC = BufferCFP32<GemmKernel224BF16>;

  /* Tile configuration:
   *   tile 0,1 — A tiles, TILE_M × (TILE_K * sizeof(bf16)) = 16 × 64
   *   tile 2,3 — B tiles in VNNI form, (TILE_K/2) × (TILE_N * 2 * 2) = 16 × 64
   *   tile 4..7 — C tiles, TILE_M × (TILE_N * sizeof(float)) = 16 × 64
   */
  static void config() {
    TileConfig cfg;
    for (int i = 0; i < 2; i++)  cfg.set_row_col(i, TILE_M, TILE_K * sizeof(dt));
    for (int i = 2; i < 4; i++)  cfg.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));
    for (int i = 4; i < 8; i++)  cfg.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));
    cfg.load();
  }

  static void load_a(const dt* a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b(const dt* b, size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t* c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void store_c(output_t* c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void run_tile() {
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  /* The full GEMM, one (m_begin, n_begin) tile group at a time. */
  static void amx_kernel(int /*m*/, int /*n*/, int k,
                         int m_begin, int n_begin, int k_block_begin,
                         float* c_sub, BufferA& ba, BufferB& bb) {
    if (k_block_begin == 0) {
      clean_c();
    } else {
      load_c(c_sub, N_STEP * sizeof(float));
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      load_a(ba.get_submat(/*m*/ ba.max_m, k, m_begin, k_block_begin + k_begin),
             K_STEP * sizeof(dt));
      load_b(bb.get_submat(/*n*/ bb.n,    k, n_begin, k_block_begin + k_begin),
             K_STEP * sizeof(dt));
      run_tile();
    }
    store_c(c_sub, N_STEP * sizeof(float));
  }
};

/* ------------------------------------------------------------------ */
/* mat_mul driver (port of float_mat_vec from amx_raw_kernels.hpp).   */
/* ------------------------------------------------------------------ */
static void float_mat_mul(int m, int n, int k,
                          GemmKernel224BF16::BufferA& ba,
                          GemmKernel224BF16::BufferB& bb,
                          GemmKernel224BF16::BufferC& bc,
                          int ith, int nth) {
  using K = GemmKernel224BF16;
  /* Each thread handles disjoint N_BLOCK-sized chunks, round-robin. */
  int n_blocks = (n + K::N_BLOCK - 1) / K::N_BLOCK;
  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_start = blk * K::N_BLOCK;
    int n_end   = std::min(n, n_start + K::N_BLOCK);

    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
      for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
        for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
          float* c_sub = bc.get_submat(m, n, m_begin, n_begin);
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c_sub, ba, bb);
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Public entry points used by the dispatcher.                        */
/* ------------------------------------------------------------------ */

bool amx_available() {
  static thread_local int cached = -1;
  if (cached == -1) cached = enable_amx() ? 1 : 0;
  return cached == 1;
}

/* Must be noinline + must contain a clobber-everything asm barrier:
 * the optimizer doesn't model the kernel side effect of ARCH_REQ_XCOMP_PERM,
 * so without these guards AMX intrinsics in callers can be hoisted above
 * the syscall and SIGILL at runtime. */
__attribute__((noinline))
void bf16_tile_config_init() {
  /* Per-thread XSAVE permission grant — only needed once per thread. */
  static thread_local bool xperm_ok = false;
  if (!xperm_ok) {
    if (!enable_amx()) return;
    xperm_ok = true;
  }
  /* Re-load the tile config on every entry. Cheap (single ldtilecfg) and
   * robust against callers that issue _tile_release() between GEMMs. The
   * config struct is stack-allocated, hot in cache. */
  __asm__ volatile("" ::: "memory");
  GemmKernel224BF16::config();
  __asm__ volatile("" ::: "memory");
}

void bf16_pack(int m, int n, int k,
               const cg_bf16_t* a_rm, std::size_t lda,
               const cg_bf16_t* b_rm, std::size_t ldb,
               void* scratch_a, void* scratch_b, void* /*scratch_c*/,
               int ith, int nth) {
  using K = GemmKernel224BF16;
  int max_m_pad = pad_up(m, K::M_STEP);
  int n_pad     = pad_up(n, K::N_STEP);

  GemmKernel224BF16::BufferA ba(max_m_pad, k, scratch_a);
  GemmKernel224BF16::BufferB bb(n_pad, k, scratch_b);

  /* Pack A by ith == 0 only (small relative to B for typical shapes). The
   * dispatcher could split this further; not worth the complexity yet. */
  if (ith == 0) {
    /* If lda differs from k we'd need a stride-aware path; the v0.1
     * dispatcher passes lda == k for now. */
    (void)lda;
    ba.from_mat(m, a_rm);
  }
  /* Pack B across nth threads. Pass the real n so the packer zero-fills
   * padded rows [n, n_pad) instead of reading past the caller's buffer. */
  (void)ldb;
  bb.from_mat(b_rm, n, ith, nth);
}

void bf16_run(int m, int n, int k,
              void* scratch_a, void* scratch_b, void* scratch_c,
              int ith, int nth) {
  using K = GemmKernel224BF16;
  int max_m_pad = pad_up(m, K::M_STEP);
  int n_pad     = pad_up(n, K::N_STEP);

  bf16_tile_config_init();

  GemmKernel224BF16::BufferA ba(max_m_pad, k, scratch_a);
  GemmKernel224BF16::BufferB bb(n_pad, k, scratch_b);
  GemmKernel224BF16::BufferC bc(max_m_pad, n_pad, scratch_c);

  float_mat_mul(m, n_pad, k, ba, bb, bc, ith, nth);
  /* Intentionally NOT calling _tile_release(): keeping the tile config
   * resident lets back-to-back GEMMs skip one ldtilecfg. The tile config
   * is reloaded on every entry to bf16_run anyway, so leaving it warm is
   * a pure win. */
}

void bf16_unpack(int m, int n,
                 const void* scratch_c,
                 float alpha,
                 float beta, float* c_rm, std::size_t ldc,
                 int ith, int nth) {
  using K = GemmKernel224BF16;
  int max_m_pad = pad_up(m, K::M_STEP);
  int n_pad     = pad_up(n, K::N_STEP);
  /* Reinterpret the scratch as a BufferC. We don't need the buffer's
   * stateful methods here — only the layout walk. */
  const float* c_blk = reinterpret_cast<const float*>(scratch_c);
  int n_blocks = (n_pad + K::N_BLOCK - 1) / K::N_BLOCK;
  int m_block_size = max_m_pad;  /* == M_STEP * m_blocks */
  (void)m_block_size;

  for (int blk = 0; blk < n_blocks; ++blk) {
    if (blk % nth != ith) continue;
    int n_block_begin = blk * K::N_BLOCK;
    int n_block_end   = std::min(n_pad, n_block_begin + K::N_BLOCK);
    int n_block_size  = n_block_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += K::N_STEP) {
        for (int i = 0; i < K::M_STEP && m_begin + i < m; ++i) {
          const float* src = c_blk + (std::size_t)max_m_pad * n_block_begin
                                  + (std::size_t)m_begin * n_block_size
                                  + (std::size_t)n_begin * K::M_STEP
                                  + (std::size_t)i * K::N_STEP;
          /* Per-row n range valid in the logical output. */
          int n_valid = std::min(K::N_STEP, n - (n_block_begin + n_begin));
          if (n_valid <= 0) continue;
          float* dst = c_rm + (std::size_t)(m_begin + i) * ldc + n_block_begin + n_begin;
          if (beta == 0.0f && alpha == 1.0f) {
            std::memcpy(dst, src, sizeof(float) * n_valid);
          } else if (beta == 0.0f) {
            for (int j = 0; j < n_valid; ++j) dst[j] = alpha * src[j];
          } else {
            for (int j = 0; j < n_valid; ++j) dst[j] = beta * dst[j] + alpha * src[j];
          }
        }
      }
    }
  }
}

}  // namespace cpu_gemm::kernels::amx

#endif  // CPU_GEMM_HAS_AMX
