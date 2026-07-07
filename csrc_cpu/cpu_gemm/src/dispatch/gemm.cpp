/*
 * GEMM dispatcher.
 *
 * cg_gemm_st() is the leaf — selects a kernel implementation given the dtype
 * tuple and runs it for a single (ith, nth) slice. AVX2-only.
 *
 * cg_gemm() is the parallel front-end. For backends that pack the inputs
 * (AMX, future AVX-512 paths) it owns the scratch arena and the three
 * pack→run→unpack phases. For non-packing backends it fans cg_gemm_st()
 * out across threads.
 *
 * Backend selection priority (when both compiled and supported on host):
 *   AMX  >  AVX2
 */
#include "cpu_gemm/cpu_gemm.h"
#include "runtime/runtime_impl.h"

#include <atomic>
#include <cstdint>
#include <cstring>

#if defined(CPU_GEMM_HAS_AVX2)
#include "kernels/avx2/bf16_gemm.h"
#endif

#if defined(CPU_GEMM_HAS_AMX)
#include "kernels/amx/bf16_gemm.h"
#include "kernels/amx/int8_gemm.h"
#include "kernels/amx/int8_gemm_rm.h"
#endif

namespace {

bool desc_well_formed(const cg_gemm_desc_t* d) {
  if (!d || !d->a || !d->b || !d->c) return false;
  if (d->order != CG_ROW_MAJOR) return false;
  if (d->dtype_c != CG_F32) return false;
  if (d->m == 0 || d->n == 0 || d->k == 0) return false;
  return true;
}

/* INT8 weight path: A is BF16 activations, B is int8 weights with
 * per-channel scales (b_scales), output FP32. B in CG_TRANS layout. */
bool is_bf16_int8_f32_nt(const cg_gemm_desc_t* d) {
  return d->dtype_a == CG_BF16 && d->dtype_b == CG_INT8 &&
         d->dtype_c == CG_F32 &&
         d->trans_a == CG_NO_TRANS && d->trans_b == CG_TRANS &&
         d->b_scales != nullptr;
}

/* INT8 weight path with pre-packed B (AMX layout, scales appended). */
bool is_bf16_int8packed_f32_nt(const cg_gemm_desc_t* d) {
  return d->dtype_a == CG_BF16 && d->dtype_b == CG_INT8_PACKED_AMX &&
         d->dtype_c == CG_F32 &&
         d->trans_a == CG_NO_TRANS && d->trans_b == CG_TRANS;
}

bool is_bf16_bf16_f32_nt(const cg_gemm_desc_t* d) {
  return d->dtype_a == CG_BF16 && d->dtype_b == CG_BF16 &&
         d->dtype_c == CG_F32  &&
         d->trans_a == CG_NO_TRANS && d->trans_b == CG_TRANS;
}

cg_status_t run_avx2_slice(const cg_gemm_desc_t* d, int ith, int nth) {
#if defined(CPU_GEMM_HAS_AVX2)
  cpu_gemm::kernels::avx2::gemm_bf16_bf16_f32(
      (int)d->m, (int)d->n, (int)d->k,
      d->alpha,
      reinterpret_cast<const cg_bf16_t*>(d->a), d->lda,
      reinterpret_cast<const cg_bf16_t*>(d->b), d->ldb,
      d->beta,
      reinterpret_cast<float*>(d->c), d->ldc,
      ith, nth);
  return CG_OK;
#else
  (void)d; (void)ith; (void)nth;
  return CG_E_UNSUPPORTED;
#endif
}

#if defined(CPU_GEMM_HAS_AMX)
/* True when the AMX BF16 path can handle this descriptor on this host. */
bool amx_bf16_eligible(const cg_gemm_desc_t* d) {
  using T = cpu_gemm::kernels::amx::Bf16KernelTraits;
  if (!cpu_gemm::kernels::amx::amx_available()) return false;
  if (!is_bf16_bf16_f32_nt(d)) return false;
  /* k must be a multiple of K_STEP; m and n can be anything (padded). */
  if (d->k % T::K_STEP != 0) return false;
  /* Strides must equal logical row width — the packer is not stride-aware
   * (yet). For typical callers lda == k, ldb == k. */
  if (d->lda != d->k || d->ldb != d->k) return false;
  return true;
}

/* True when the AMX INT8 path can handle this descriptor on this host. */
bool amx_int8_eligible(const cg_gemm_desc_t* d) {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (!cpu_gemm::kernels::amx::amx_available()) return false;
  if (!is_bf16_int8_f32_nt(d)) return false;
  if (d->k % T::K_STEP != 0) return false;   /* K_STEP = 64 for INT8 */
  if (d->lda != d->k || d->ldb != d->k) return false;
  return true;
}

/* True when the stride-aware AMX INT8 path can handle this descriptor —
 * i.e. B is row-major int8 + per-channel scales, K is K_STEP-aligned, A is
 * tightly packed BF16, and B's logical N is N_STEP-aligned (so the
 * strided _tile_loadd never reads past the caller's allocation).
 *
 * The unified-MoE runtime always pads N to 128 (GPU constraint), which is
 * a multiple of N_STEP=32, so this predicate covers it. Unaligned-N
 * callers still hit the older packed-on-touch path below. */
bool amx_int8_rm_eligible(const cg_gemm_desc_t* d) {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (!cpu_gemm::kernels::amx::amx_available()) return false;
  if (!is_bf16_int8_f32_nt(d)) return false;
  if (d->k % T::K_STEP != 0) return false;
  if (d->n % T::N_STEP != 0) return false;
  if (d->lda != d->k)          return false;   /* A still tight row-major BF16 */
  if (d->ldb <  d->k)          return false;   /* B row-major, stride >= k */
  return true;
}

/* True when the AMX INT8 path can consume a pre-packed B on this host. */
bool amx_int8_prepacked_eligible(const cg_gemm_desc_t* d) {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (!cpu_gemm::kernels::amx::amx_available()) return false;
  if (!is_bf16_int8packed_f32_nt(d)) return false;
  if (d->k % T::K_STEP != 0) return false;
  if (d->lda != d->k) return false;
  /* Pre-pack is sized for n padded to N_STEP. ldb is unused (B is the
   * packed buffer, not a row-major matrix). */
  return true;
}

/* Pre-packed B variant: B and its per-channel scales live in desc->b as one
 * AMX-layout buffer. Scratch is allocated for A and C only. */
cg_status_t run_amx_int8_prepacked(cg_runtime_t* rt, const cg_gemm_desc_t* d) {
  namespace amx = cpu_gemm::kernels::amx;
  int m = (int)d->m, n = (int)d->n, k = (int)d->k;

  int max_m_pad = amx::int8_pad_up(m, amx::Int8KernelTraits::M_STEP);
  int n_pad     = amx::int8_pad_up(n, amx::Int8KernelTraits::N_STEP);

  std::size_t bytes_a = sizeof(int8_t)  * (std::size_t)max_m_pad * k + sizeof(float) * max_m_pad;
  std::size_t bytes_c = sizeof(int32_t) * (std::size_t)max_m_pad * n_pad;

  void* base = rt->scratch->reserve(bytes_a + bytes_c);
  auto* p = reinterpret_cast<unsigned char*>(base);
  void* sA = p;            p += bytes_a;
  void* sC = p;

  /* The caller's pre-packed B is already in BufferBInt8 layout: int8 block
   * followed by per-channel scales. Hand both pointers directly to the
   * kernel — no copy, no per-call repack. */
  void* sB = const_cast<void*>(d->b);

  const float* a_scales = reinterpret_cast<const float*>(
      reinterpret_cast<unsigned char*>(sA) + amx::int8_a_scales_offset(max_m_pad, k));
  const float* b_scales = reinterpret_cast<const float*>(
      reinterpret_cast<unsigned char*>(sB) + amx::int8_b_scales_offset(n_pad, k));

  int n_blocks = (n + amx::Int8KernelTraits::N_BLOCK - 1) / amx::Int8KernelTraits::N_BLOCK;
  int nth = rt->threads;
  if (nth > n_blocks) nth = n_blocks;
  if (nth <= 0) nth = 1;

  const cg_bf16_t* a_rm = reinterpret_cast<const cg_bf16_t*>(d->a);
  float*           c_rm = reinterpret_cast<float*>(d->c);

  if (nth == 1) {
    amx::int8_pack_a_bf16(m, k, a_rm, sA);
    amx::int8_run(m, n, k, sA, sB, sC, 0, 1);
    amx::int8_unpack_explicit(m, n, a_scales, b_scales, sC,
                              d->alpha, d->beta, c_rm, d->ldc, 0, 1);
  } else {
    /* A pack on thread 0; the run + unpack go across all threads. */
    rt->pool->parallel_for(nth, [&](int ith) {
      if (ith == 0) amx::int8_pack_a_bf16(m, k, a_rm, sA);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_run(m, n, k, sA, sB, sC, ith, nth);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_unpack_explicit(m, n, a_scales, b_scales, sC,
                                d->alpha, d->beta, c_rm, d->ldc, ith, nth);
    });
  }
  return CG_OK;
}

/* Stride-aware row-major B path. No per-call B pack — B is read straight
 * from caller memory by AMX src1 (strided _tile_loadd). A is packed once
 * into AMX src2's VNNI shape; C lands as C^T and the unpack transposes
 * during scale apply. See Stride.md §3 for the full derivation. */
cg_status_t run_amx_int8_rm(cg_runtime_t* rt, const cg_gemm_desc_t* d) {
  namespace amx = cpu_gemm::kernels::amx;
  int m = (int)d->m, n = (int)d->n, k = (int)d->k;

  auto s = amx::int8_rm_scratch(m, n, k);
  void* base = rt->scratch->reserve(s.total());
  auto* p = reinterpret_cast<unsigned char*>(base);
  void* sA = p;            p += s.bytes_a;
  void* sC = p;

  int max_m_pad = amx::int8_pad_up(m, amx::Int8KernelTraits::M_STEP);
  const float* a_scales = reinterpret_cast<const float*>(
      reinterpret_cast<unsigned char*>(sA) + amx::int8_rm_a_scales_offset(max_m_pad, k));

  int n_blocks = (amx::int8_pad_up(n, amx::Int8KernelTraits::N_STEP)
                  + amx::Int8KernelTraits::N_BLOCK - 1)
                 / amx::Int8KernelTraits::N_BLOCK;
  int nth = rt->threads;
  if (nth > n_blocks) nth = n_blocks;
  if (nth <= 0) nth = 1;

  const cg_bf16_t* a_rm        = reinterpret_cast<const cg_bf16_t*>(d->a);
  const int8_t*    b_rm        = reinterpret_cast<const int8_t*>(d->b);
  const float*     b_scales_in = reinterpret_cast<const float*>(d->b_scales);
  float*           c_rm        = reinterpret_cast<float*>(d->c);

  if (nth == 1) {
    amx::int8_rm_pack_a_bf16(m, k, a_rm, sA);
    amx::int8_rm_run(m, n, k, b_rm, d->ldb, sA, sC, 0, 1);
    amx::int8_rm_unpack_transposed(m, n, a_scales, b_scales_in, sC,
                                   d->alpha, d->beta, c_rm, d->ldc, 0, 1);
  } else {
    /* A pack is cheap and single-threaded (m is small); run-and-unpack
     * fan out across threads along whole N_BLOCK stripes. */
    rt->pool->parallel_for(nth, [&](int ith) {
      if (ith == 0) amx::int8_rm_pack_a_bf16(m, k, a_rm, sA);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_rm_run(m, n, k, b_rm, d->ldb, sA, sC, ith, nth);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_rm_unpack_transposed(m, n, a_scales, b_scales_in, sC,
                                     d->alpha, d->beta, c_rm, d->ldc, ith, nth);
    });
  }
  return CG_OK;
}

cg_status_t run_amx_int8(cg_runtime_t* rt, const cg_gemm_desc_t* d) {
  namespace amx = cpu_gemm::kernels::amx;
  int m = (int)d->m, n = (int)d->n, k = (int)d->k;
  auto s = amx::int8_scratch(m, n, k);

  void* base = rt->scratch->reserve(s.total());
  auto* p = reinterpret_cast<unsigned char*>(base);
  void* sA = p;                  p += s.bytes_a;
  void* sB = p;                  p += s.bytes_b;
  void* sC = p;

  int max_m_pad = amx::int8_pad_up(m, amx::Int8KernelTraits::M_STEP);
  int n_pad     = amx::int8_pad_up(n, amx::Int8KernelTraits::N_STEP);
  const float* a_scales = reinterpret_cast<const float*>(
      reinterpret_cast<unsigned char*>(sA) + amx::int8_a_scales_offset(max_m_pad, k));
  const float* b_scales = reinterpret_cast<const float*>(
      reinterpret_cast<unsigned char*>(sB) + amx::int8_b_scales_offset(n_pad, k));

  int n_blocks = (n + amx::Int8KernelTraits::N_BLOCK - 1) / amx::Int8KernelTraits::N_BLOCK;
  int nth = rt->threads;
  if (nth > n_blocks) nth = n_blocks;
  if (nth <= 0) nth = 1;

  const cg_bf16_t* a_rm     = reinterpret_cast<const cg_bf16_t*>(d->a);
  const int8_t*    b_int8   = reinterpret_cast<const int8_t*>(d->b);
  const float*     b_scales_in = reinterpret_cast<const float*>(d->b_scales);
  float*           c_rm     = reinterpret_cast<float*>(d->c);

  if (nth == 1) {
    amx::int8_pack_a_bf16(m, k, a_rm, sA);
    amx::int8_pack_b_int8(n, k, b_int8, b_scales_in, sB, 0, 1);
    amx::int8_run(m, n, k, sA, sB, sC, 0, 1);
    amx::int8_unpack_explicit(m, n, a_scales, b_scales, sC,
                              d->alpha, d->beta, c_rm, d->ldc, 0, 1);
  } else {
    /* A packing is single-threaded (cheap); do it on thread 0 then run
     * B-pack across threads. */
    rt->pool->parallel_for(nth, [&](int ith) {
      if (ith == 0) amx::int8_pack_a_bf16(m, k, a_rm, sA);
      amx::int8_pack_b_int8(n, k, b_int8, b_scales_in, sB, ith, nth);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_run(m, n, k, sA, sB, sC, ith, nth);
    });
    rt->pool->parallel_for(nth, [&](int ith) {
      amx::int8_unpack_explicit(m, n, a_scales, b_scales, sC,
                                d->alpha, d->beta, c_rm, d->ldc, ith, nth);
    });
  }
  return CG_OK;
}
#endif

cg_status_t run_cg_gemm(cg_runtime_t* rt, const cg_gemm_desc_t* d) {
#if defined(CPU_GEMM_HAS_AMX)
  if (rt && rt->pool && amx_int8_prepacked_eligible(d)) {
    return run_amx_int8_prepacked(rt, d);
  }
  /* Prefer the stride-aware (row-major B) path over the packed-on-touch
   * variant — it avoids the per-call B repack and has strictly less
   * scratch. The packed-on-touch fallback remains for shapes that don't
   * satisfy the stride-aware predicate (e.g. n % N_STEP != 0). */
  if (rt && rt->pool && amx_int8_rm_eligible(d)) {
    return run_amx_int8_rm(rt, d);
  }
  if (rt && rt->pool && amx_int8_eligible(d)) {
    return run_amx_int8(rt, d);
  }
  if (amx_bf16_eligible(d)) {
    namespace amx = cpu_gemm::kernels::amx;
    int m = (int)d->m, n = (int)d->n, k = (int)d->k;
    auto s = amx::bf16_scratch(m, n, k);

    /* Acquire scratch once, share across phases. The arena returns
     * 64-byte-aligned memory; we partition it into A | B | C. */
    void* base = rt->scratch->reserve(s.total());
    auto* p = reinterpret_cast<unsigned char*>(base);
    void* sA = p;                                                 p += s.bytes_a;
    void* sB = p;                                                 p += s.bytes_b;
    void* sC = p;

    /* Choose a thread count compatible with the kernel's N_BLOCK split.
     * Extra threads beyond ceil(n / N_BLOCK) would idle, so cap. */
    int n_blocks = (n + amx::Bf16KernelTraits::N_BLOCK - 1) / amx::Bf16KernelTraits::N_BLOCK;
    int nth = rt->threads;
    if (nth > n_blocks) nth = n_blocks;
    if (nth <= 0) nth = 1;

    const cg_bf16_t* a_rm = reinterpret_cast<const cg_bf16_t*>(d->a);
    const cg_bf16_t* b_rm = reinterpret_cast<const cg_bf16_t*>(d->b);
    float*           c_rm = reinterpret_cast<float*>(d->c);

    if (nth == 1) {
      amx::bf16_pack  (m, n, k, a_rm, d->lda, b_rm, d->ldb, sA, sB, sC, 0, 1);
      amx::bf16_run   (m, n, k, sA, sB, sC, 0, 1);
      amx::bf16_unpack(m, n, sC, d->alpha, d->beta, c_rm, d->ldc, 0, 1);
    } else {
      rt->pool->parallel_for(nth, [&](int ith) {
        amx::bf16_pack(m, n, k, a_rm, d->lda, b_rm, d->ldb, sA, sB, sC, ith, nth);
      });
      rt->pool->parallel_for(nth, [&](int ith) {
        amx::bf16_run(m, n, k, sA, sB, sC, ith, nth);
      });
      rt->pool->parallel_for(nth, [&](int ith) {
        amx::bf16_unpack(m, n, sC, d->alpha, d->beta, c_rm, d->ldc, ith, nth);
      });
    }
    return CG_OK;
  }
#endif

  /* Non-packing AVX2 fan-out. */
  if (is_bf16_bf16_f32_nt(d)) {
    if (!rt || !rt->pool) return run_avx2_slice(d, 0, 1);
    int nth = rt->threads;
    std::atomic<cg_status_t> status{CG_OK};
    rt->pool->parallel_for(nth, [&](int ith) {
      cg_status_t s = run_avx2_slice(d, ith, nth);
      if (s != CG_OK) {
        cg_status_t expected = CG_OK;
        status.compare_exchange_strong(expected, s);
      }
    });
    return status.load();
  }
  return CG_E_UNSUPPORTED;
}

}  // namespace

extern "C" cg_status_t cg_gemm_st(const cg_gemm_desc_t* d, int ith, int nth) {
  if (!desc_well_formed(d)) return CG_E_INVALID;
  if (nth <= 0 || ith < 0 || ith >= nth) return CG_E_INVALID;
  if (is_bf16_bf16_f32_nt(d)) return run_avx2_slice(d, ith, nth);
  return CG_E_UNSUPPORTED;
}

extern "C" cg_status_t cg_gemm(cg_runtime_t* rt, const cg_gemm_desc_t* d) {
  if (!desc_well_formed(d)) return CG_E_INVALID;
  return run_cg_gemm(rt, d);
}

/* ----------------------------------------------------------------------
 * Public offline-pack entry points.
 *
 * Returns the byte size of a pre-packed B buffer for the AMX INT8 path.
 * If AMX is not built in, returns 0 (caller should check status before
 * allocating, e.g. via cg_query_caps).
 * ---------------------------------------------------------------------- */
extern "C" size_t cg_pack_b_int8_amx_size(size_t n, size_t k) {
#if defined(CPU_GEMM_HAS_AMX)
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (n == 0 || k == 0) return 0;
  if (k % T::K_STEP != 0) return 0;
  return cpu_gemm::kernels::amx::int8_packed_b_size((int)n, (int)k);
#else
  (void)n; (void)k;
  return 0;
#endif
}

extern "C" cg_status_t cg_pack_b_int8_amx(size_t n, size_t k,
                                          const int8_t* b_int8,
                                          const float* b_scales,
                                          void* dst) {
#if defined(CPU_GEMM_HAS_AMX)
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  if (!b_int8 || !b_scales || !dst) return CG_E_INVALID;
  if (n == 0 || k == 0) return CG_E_INVALID;
  if (k % T::K_STEP != 0) return CG_E_UNSUPPORTED;
  if (reinterpret_cast<std::uintptr_t>(dst) % 64 != 0) return CG_E_ALIGNMENT;
  cpu_gemm::kernels::amx::int8_pack_b_int8_offline((int)n, (int)k, b_int8, b_scales, dst);
  return CG_OK;
#else
  (void)n; (void)k; (void)b_int8; (void)b_scales; (void)dst;
  return CG_E_UNSUPPORTED;
#endif
}
