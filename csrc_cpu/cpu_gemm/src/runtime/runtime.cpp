#include "cpu_gemm/runtime.h"
#include "runtime/runtime_impl.h"

#include <atomic>
#include <thread>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

extern "C" cg_runtime_t* cg_runtime_create(int n_threads) {
  if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
  if (n_threads <= 0) n_threads = 1;
  auto* rt = new cg_runtime;
  rt->pool = std::make_unique<cpu_gemm::WorkerPool>(n_threads);
  rt->scratch = std::make_unique<cpu_gemm::ScratchArena>();
  rt->threads = n_threads;
  return rt;
}

extern "C" void cg_runtime_destroy(cg_runtime_t* rt) {
  delete rt;
}

extern "C" int cg_runtime_threads(const cg_runtime_t* rt) {
  return rt ? rt->threads : 0;
}

/* ------------------------------------------------------------------------ */
/* CPU capability probe. Returns the union of compile-time + runtime bits.  */
/* ------------------------------------------------------------------------ */
namespace {

struct CapsProbe {
  cg_caps_t bits{};
  CapsProbe() {
#if defined(__GNUC__) || defined(__clang__)
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
      bits.has_fma  = (c & (1u << 12)) != 0;
    }
    /* AVX2, AVX512F live in CPUID leaf 7. */
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
      bits.has_avx2          = (b & (1u << 5))  != 0;
      bits.has_avx512f       = (b & (1u << 16)) != 0;
      bits.has_avx512_bf16   = 0;  /* leaf 7 subleaf 1 */
      bits.has_amx_bf16      = (d & (1u << 22)) != 0;
      bits.has_amx_int8      = (d & (1u << 25)) != 0;
    }
    if (__get_cpuid_count(7, 1, &a, &b, &c, &d)) {
      bits.has_avx_vnni      = (a & (1u << 4))  != 0;
      bits.has_avx512_bf16   = (a & (1u << 5))  != 0;
    }
#endif
  }
};

}  // namespace

extern "C" cg_caps_t cg_query_caps(void) {
  static CapsProbe probe;
  return probe.bits;
}
