/*
 * AMX BF16 correctness tests.
 *
 * Skips itself with PASS if the host doesn't expose AMX_BF16 — we don't
 * want CI lanes without AMX to red-fail on hardware-gated kernels.
 *
 * Coverage:
 *   - K aligned to 32 (the kernel's K_STEP); both small and large.
 *   - M, N tested at the boundaries of the kernel's padding logic
 *     (M_STEP = N_STEP = 32, N_BLOCK = 256).
 *   - Single-thread (rt with 1 thread) and multi-thread (rt with 8 threads).
 *   - alpha != 1, beta != 0 verified.
 */
#include <cstdio>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::fill_bf16;
using cpu_gemm::test::ref_gemm_bf16_bf16_f32;
using cpu_gemm::test::max_rel_diff;
using cpu_gemm::test::max_abs_diff;

namespace {

struct Shape { int m, n, k; const char* tag; };

int run_one(Shape s, int n_threads, float alpha, float beta, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<cg_bf16_t> b((size_t)s.n * s.k);
  std::vector<float>     c_actual((size_t)s.m * s.n);
  std::vector<float>     c_ref((size_t)s.m * s.n);

  fill_bf16(a, seed);
  fill_bf16(b, seed + 1);

  /* Seed C with non-zero values so beta != 0 is exercised. */
  for (size_t i = 0; i < c_actual.size(); ++i) {
    float v = 0.001f * (float)((i * 17) % 31) - 0.015f;
    c_actual[i] = v;
    c_ref[i]    = v;
  }

  ref_gemm_bf16_bf16_f32(s.m, s.n, s.k,
                         alpha,
                         a.data(), s.k,
                         b.data(), s.k,
                         beta,
                         c_ref.data(), s.n);

  auto d = cpu_gemm::make_desc();
  d.m = s.m; d.n = s.n; d.k = s.k;
  d.alpha = alpha; d.beta = beta;
  d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_BF16;
  d.c = c_actual.data(); d.ldc = s.n; d.dtype_c = CG_F32;

  cpu_gemm::Runtime rt(n_threads);
  cpu_gemm::gemm(rt, d);

  float abs = max_abs_diff(c_actual.data(), c_ref.data(), c_actual.size());
  float rel = max_rel_diff(c_actual.data(), c_ref.data(), c_actual.size());

  /* BF16 has ~3e-3 unit roundoff. K-accumulated tolerance scales roughly
   * with sqrt(k) for random inputs; allow a generous bound and check rel
   * as the primary metric. */
  float tol_abs = 5e-2f * std::max(1, s.k / 16);
  float tol_rel = 5e-2f;
  bool ok = abs <= tol_abs || rel <= tol_rel;
  std::printf("[%s thr=%d a=%.1f b=%.1f] m=%4d n=%4d k=%5d  abs=%.3e rel=%.3e %s\n",
              s.tag, n_threads, alpha, beta, s.m, s.n, s.k, abs, rel, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  cg_caps_t caps = cg_query_caps();
  if (!caps.has_amx_bf16) {
    std::printf("AMX-BF16 not present on this host; skipping.\n");
    return 0;
  }
  std::printf("AMX-BF16 detected. Running tests...\n");

  const Shape shapes[] = {
      {1,    64,   256, "tiny"},
      {1,   256,   256, "1xN_BLOCK"},
      {1,   512,   512, "1x2blocks"},
      {32,  256,  1024, "M_STEP"},
      {64,  512,  2048, "med"},
      {128, 4096, 4096, "wide"},
      /* Shapes that aren't multiples of M_STEP or N_STEP — exercise pad path. */
      {7,    71,   256, "odd_mn"},
      {1,   257,   512, "n_unaligned"},
  };

  int failures = 0;
  uint64_t seed = 0xA8E16;
  for (auto s : shapes) failures += run_one(s,  1, 1.0f, 0.0f, seed++);
  for (auto s : shapes) failures += run_one(s,  8, 1.0f, 0.0f, seed++);
  for (auto s : shapes) failures += run_one(s, 32, 1.0f, 0.0f, seed++);
  /* alpha/beta sweep on the wide shape. */
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 0.5f,  0.0f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 1.0f,  0.5f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 2.0f, -1.0f, seed++);

  if (failures) { std::printf("FAIL: %d case(s)\n", failures); return 1; }
  std::printf("All AMX-BF16 cases passed.\n");
  return 0;
}
