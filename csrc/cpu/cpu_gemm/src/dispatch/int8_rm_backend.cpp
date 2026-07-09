/*
 * dispatch/int8_rm_backend.cpp — implementation of the runtime
 * INT8 row-major backend selector. See header for the contract.
 *
 * PR 1 of avx_512.md: this file populates the AMX branch only. The
 * AVX-512-VNNI branch is wired in PR 3 once kernels/avx512 exposes
 * matching signatures (PR 2).
 */
#include "dispatch/int8_rm_backend.h"

#include <cstdlib>
#include <cstring>

#include "cpu_gemm/runtime.h"

#if defined(CPU_GEMM_HAS_AMX)
#include "kernels/amx/int8_gemm.h"      /* Int8KernelTraits, int8_pad_up */
#include "kernels/amx/int8_gemm_rm.h"   /* int8_rm_* */
#include "kernels/amx/bf16_gemm.h"      /* amx_available() */
#endif

#if defined(CPU_GEMM_HAS_AVX512_VNNI)
#include "kernels/avx512/int8_gemm_rm.h"
#endif

namespace cpu_gemm {

namespace {

#if defined(CPU_GEMM_HAS_AMX)

/* Adapter from the public amx::int8_rm_scratch() shape (which returns
 * its own struct) into the backend-table Int8RmScratchSizes. */
Int8RmScratchSizes amx_int8_rm_scratch_sizes(int m, int n, int k) {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  auto s = cpu_gemm::kernels::amx::int8_rm_scratch(m, n, k);
  Int8RmScratchSizes out{};
  out.bytes_a      = s.bytes_a;
  out.bytes_c      = s.bytes_c;
  out.max_m_pad    = (std::size_t)cpu_gemm::kernels::amx::int8_pad_up(m, T::M_STEP);
  out.a_scales_off = cpu_gemm::kernels::amx::int8_rm_a_scales_offset(
      (int)out.max_m_pad, k);
  return out;
}

Int8RmBackend make_amx_backend() {
  using T = cpu_gemm::kernels::amx::Int8KernelTraits;
  Int8RmBackend be{};
  be.pack_a            = &cpu_gemm::kernels::amx::int8_rm_pack_a_bf16;
  be.run               = &cpu_gemm::kernels::amx::int8_rm_run;
  be.unpack_transposed = &cpu_gemm::kernels::amx::int8_rm_unpack_transposed;
  be.scratch_sizes     = &amx_int8_rm_scratch_sizes;
  be.tile_config_init  = &cpu_gemm::kernels::amx::int8_rm_tile_config_init;
  be.k_step            = T::K_STEP;       /* 64 */
  be.n_step            = T::N_STEP;       /* 32 */
  be.n_block           = T::N_BLOCK;
  be.ok                = true;
  be.name              = "amx_int8_rm";
  return be;
}

#endif  /* CPU_GEMM_HAS_AMX */

#if defined(CPU_GEMM_HAS_AVX512_VNNI)

/* Adapter from avx512::int8_rm_scratch's struct into the table's
 * Int8RmScratchSizes. */
Int8RmScratchSizes avx512_vnni_int8_rm_scratch_sizes(int m, int n, int k) {
  using T = cpu_gemm::kernels::avx512::Int8RmTraits;
  auto s = cpu_gemm::kernels::avx512::int8_rm_scratch(m, n, k);
  Int8RmScratchSizes out{};
  out.bytes_a      = s.bytes_a;
  out.bytes_c      = s.bytes_c;
  out.max_m_pad    = (std::size_t)cpu_gemm::kernels::avx512::int8_rm_pad_up(
      m, T::M_STEP);
  out.a_scales_off = cpu_gemm::kernels::avx512::int8_rm_a_scales_offset(
      (int)out.max_m_pad, k);
  return out;
}

Int8RmBackend make_avx512_vnni_backend() {
  using T = cpu_gemm::kernels::avx512::Int8RmTraits;
  Int8RmBackend be{};
  be.pack_a            = &cpu_gemm::kernels::avx512::int8_rm_pack_a_bf16;
  be.run               = &cpu_gemm::kernels::avx512::int8_rm_run;
  be.unpack_transposed = &cpu_gemm::kernels::avx512::int8_rm_unpack_transposed;
  be.scratch_sizes     = &avx512_vnni_int8_rm_scratch_sizes;
  be.tile_config_init  = &cpu_gemm::kernels::avx512::int8_rm_tile_config_init;
  be.k_step            = T::K_STEP;       /* 4 — vpdpbusd consumes 4 K's per dword */
  be.n_step            = T::N_STEP;       /* 16 */
  be.n_block           = T::N_BLOCK;
  be.ok                = true;
  be.name              = "avx512_vnni_int8_rm";
  return be;
}

#endif  /* CPU_GEMM_HAS_AVX512_VNNI */

Int8RmBackend make_none_backend() {
  return Int8RmBackend{
      /* pack_a            */ nullptr,
      /* run               */ nullptr,
      /* unpack_transposed */ nullptr,
      /* scratch_sizes     */ nullptr,
      /* tile_config_init  */ nullptr,
      /* k_step            */ 0,
      /* n_step            */ 0,
      /* n_block           */ 0,
      /* ok                */ false,
      /* name              */ "none",
  };
}

/* Parse ASYM_GEMM_FORCE_BACKEND. Returns:
 *    0 = no override (autodetect)
 *    1 = force AMX
 *    2 = force AVX-512-VNNI
 *    3 = force none
 *
 * Unknown values are treated as 0 (autodetect) — defensive against
 * typos that would otherwise silently force a no-op binary. */
int parse_force_backend() {
  const char* v = std::getenv("ASYM_GEMM_FORCE_BACKEND");
  if (!v || !*v) return 0;
  if (std::strcmp(v, "amx")    == 0) return 1;
  if (std::strcmp(v, "avx512") == 0) return 2;
  if (std::strcmp(v, "none")   == 0) return 3;
  return 0;
}

Int8RmBackend select_backend_impl() {
  int forced = parse_force_backend();

  if (forced == 3) {
    return make_none_backend();
  }

#if defined(CPU_GEMM_HAS_AMX)
  if (forced == 1) {
    /* User asked for AMX — try even if the runtime might fail later. */
    if (cpu_gemm::kernels::amx::amx_available()) return make_amx_backend();
    return make_none_backend();
  }
#endif

  if (forced == 2) {
#if defined(CPU_GEMM_HAS_AVX512_VNNI)
    cg_caps_t caps = cg_query_caps();
    if (caps.has_avx512_vnni && caps.has_avx512f)
      return make_avx512_vnni_backend();
#endif
    return make_none_backend();
  }

  /* Autodetect path — AMX first, AVX-512-VNNI second. */
  cg_caps_t caps = cg_query_caps();

#if defined(CPU_GEMM_HAS_AMX)
  if (caps.has_amx_int8 && cpu_gemm::kernels::amx::amx_available()) {
    return make_amx_backend();
  }
#endif

#if defined(CPU_GEMM_HAS_AVX512_VNNI)
  if (caps.has_avx512_vnni && caps.has_avx512f) {
    return make_avx512_vnni_backend();
  }
#endif

  (void)caps;
  return make_none_backend();
}

}  // namespace

const Int8RmBackend& select_int8_rm_backend() {
  /* C++11 magic statics make this initialisation thread-safe and
   * one-shot. The cost on every subsequent call is a single load of
   * the cached struct address. */
  static const Int8RmBackend chosen = select_backend_impl();
  return chosen;
}

}  // namespace cpu_gemm

/* ---------------------------------------------------------------------
 * C ABI diagnostic accessors — exported via cpu_gemm/runtime.h.
 * --------------------------------------------------------------------- */
extern "C" const char* cg_int8_rm_backend_name(void) {
  return cpu_gemm::select_int8_rm_backend().name;
}

extern "C" int cg_int8_rm_backend_ok(void) {
  return cpu_gemm::select_int8_rm_backend().ok ? 1 : 0;
}
