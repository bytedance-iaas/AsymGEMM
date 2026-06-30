/*
 * dispatch/int8_rm_backend.h — runtime-selected INT8 row-major backend.
 *
 * Both AMX-INT8 (kernels/amx/int8_gemm_rm) and AVX-512-VNNI
 * (kernels/avx512/int8_gemm_rm) consume the **same** canonical weight
 * bytes: row-major INT8 `[N, K]` + per-channel FP32 scales `[N]`
 * (per Stride.md, layout.md). This header exposes a tiny
 * function-pointer table that selects one of them at runtime.
 *
 * Design contract (avx_512.md §3):
 *   - Selection runs ONCE on first use, cached in a static lambda.
 *     CPUID + per-thread XSAVE permission decide; no per-call probe.
 *   - AMX wins whenever available (`has_amx_int8 && amx_available()`).
 *   - AVX-512-VNNI is the fallback (`has_avx512_vnni && has_avx512f`).
 *   - `ASYM_GEMM_FORCE_BACKEND={amx,avx512,none}` env var overrides
 *     the autodetect — TESTING-ONLY; production never sets it.
 *
 * Both dispatchers (`cg_gemm` and `cg_moe_int8_amx`) call
 * `select_int8_rm_backend()` and route their per-tile kernel calls
 * through the returned struct.
 */
#ifndef CPU_GEMM_DISPATCH_INT8_RM_BACKEND_H
#define CPU_GEMM_DISPATCH_INT8_RM_BACKEND_H

#include <cstddef>
#include <cstdint>

#include "cpu_gemm/types.h"

namespace cpu_gemm {

/* Per-call scratch sizes for the row-major INT8 path. Both backends
 * use a packed-A region (with per-row A scales appended) plus a
 * tiled-C region (INT32 accumulator). The total bytes vary slightly
 * because tile shapes differ (AMX: 32×32; VNNI: 16-lane zmm rows). */
struct Int8RmScratchSizes {
  std::size_t bytes_a;       /* packed A + per-row A scales */
  std::size_t bytes_c;       /* INT32 accumulator scratch */
  std::size_t a_scales_off;  /* byte offset of A scales inside bytes_a */
  std::size_t max_m_pad;     /* padded M used to size the scratch */
  std::size_t total() const { return bytes_a + bytes_c; }
};

/* Function-pointer table populated at first call. ABI stability isn't
 * a concern here — the struct is private to cpu_gemm and is read only
 * inside our own dispatchers. */
struct Int8RmBackend {
  using pack_a_fn  = void(*)(int m, int k,
                             const cg_bf16_t* a_rm, void* scratch_a);
  using run_fn     = void(*)(int m, int n, int k,
                             const std::int8_t* b_rm, std::size_t ldb,
                             void* scratch_a, void* scratch_c,
                             int ith, int nth);
  using unpack_fn  = void(*)(int m, int n,
                             const float* a_scales,
                             const float* b_scales,
                             const void* scratch_c,
                             float alpha, float beta,
                             float* c_rm, std::size_t ldc,
                             int ith, int nth);
  using scratch_fn = Int8RmScratchSizes(*)(int m, int n, int k);
  using tile_cfg_fn = void(*)();   /* may be nullptr (no-op for AVX-512) */

  pack_a_fn   pack_a;
  run_fn      run;
  unpack_fn   unpack_transposed;
  scratch_fn  scratch_sizes;
  tile_cfg_fn tile_config_init;   /* called once per thread before run() */

  int  k_step;       /* k alignment (both backends: 64) */
  int  n_step;       /* n alignment (AMX: 32; VNNI: 16) */
  int  n_block;      /* per-thread N-stripe granularity */
  bool ok;           /* false → no backend usable on this host */
  const char* name;  /* "amx_int8_rm" | "avx512_vnni_int8_rm" | "none" */
};

/* Returns the process-wide-cached backend. The first call probes
 * CPUID + reads ASYM_GEMM_FORCE_BACKEND; subsequent calls return the
 * same struct with zero overhead (static lambda). Thread-safe by
 * C++11 magic-statics initialisation. */
const Int8RmBackend& select_int8_rm_backend();

}  // namespace cpu_gemm

#endif  /* CPU_GEMM_DISPATCH_INT8_RM_BACKEND_H */
