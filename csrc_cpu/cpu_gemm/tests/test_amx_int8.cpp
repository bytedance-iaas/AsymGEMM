/*
 * AMX INT8 correctness tests.
 *
 * Exercises the BF16-activation x INT8-weight -> FP32 path through the public
 * C ABI (dtype_a = CG_BF16, dtype_b = CG_INT8, b_scales set). Skips itself
 * with PASS if the host doesn't expose AMX_INT8.
 *
 * Coverage:
 *   - K aligned to the INT8 K_STEP (64); both small and large.
 *   - N at and off the N_STEP (32) boundary — the off-boundary shapes
 *     exercise the padded-B packing path (no out-of-bounds read of the
 *     caller's weight / scale buffers).
 *   - Single-thread (rt with 1 thread) and multi-thread (rt with 8 threads).
 *   - alpha != 1, beta != 0.
 *
 * The reference quantizes A identically to the kernel, so the only modeled
 * error is the int->fp32 rescale; tolerances are tight.
 */
#include <cstdio>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::fill_bf16;
using cpu_gemm::test::fill_int8;
using cpu_gemm::test::fill_scales;
using cpu_gemm::test::ref_gemm_bf16_int8_f32;
using cpu_gemm::test::max_abs_diff;
using cpu_gemm::test::max_rel_diff;

namespace {

struct Shape { int m, n, k; const char* tag; };

int run_one(Shape s, int n_threads, float alpha, float beta, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<int8_t>    b((size_t)s.n * s.k);
  std::vector<float>     b_scales((size_t)s.n);
  std::vector<float>     c_actual((size_t)s.m * s.n);
  std::vector<float>     c_ref((size_t)s.m * s.n);

  fill_bf16(a, seed);
  fill_int8(b, seed + 1);
  fill_scales(b_scales, seed + 2);

  for (size_t i = 0; i < c_actual.size(); ++i) {
    float v = 0.001f * (float)((i * 17) % 31) - 0.015f;
    c_actual[i] = v;
    c_ref[i]    = v;
  }

  ref_gemm_bf16_int8_f32(s.m, s.n, s.k,
                         alpha,
                         a.data(), s.k,
                         b.data(), s.k, b_scales.data(),
                         beta,
                         c_ref.data(), s.n);

  auto d = cpu_gemm::make_desc();
  d.m = s.m; d.n = s.n; d.k = s.k;
  d.alpha = alpha; d.beta = beta;
  d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_INT8;
  d.b_scales = b_scales.data();
  d.c = c_actual.data(); d.ldc = s.n; d.dtype_c = CG_F32;

  cpu_gemm::Runtime rt(n_threads);
  cpu_gemm::gemm(rt, d);

  float abs = max_abs_diff(c_actual.data(), c_ref.data(), c_actual.size());
  float rel = max_rel_diff(c_actual.data(), c_ref.data(), c_actual.size());

  /* The reference mirrors the kernel's quantization exactly, so the residual
   * is just int32->fp32 rescale rounding. Allow a small absolute slack. */
  float tol_abs = 1e-4f;
  float tol_rel = 1e-3f;
  bool ok = abs <= tol_abs || rel <= tol_rel;
  std::printf("[%s thr=%d a=%.1f b=%.1f] m=%4d n=%4d k=%5d  abs=%.3e rel=%.3e %s\n",
              s.tag, n_threads, alpha, beta, s.m, s.n, s.k, abs, rel, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  cg_caps_t caps = cg_query_caps();
  if (!caps.has_amx_int8) {
    std::printf("AMX-INT8 not present on this host; skipping.\n");
    return 0;
  }
  std::printf("AMX-INT8 detected. Running tests...\n");

  const Shape shapes[] = {
      {1,    64,    64, "tiny"},
      {1,    64,   256, "decode"},
      {32,   64,   256, "M_STEP"},
      {16,  128,   512, "med"},
      {64,  256,  1024, "wide"},
      /* N not a multiple of N_STEP (32) — exercises padded-B packing. */
      {8,    70,   256, "n_unaligned"},
      {1,    96,   128, "n96"},
      {17,  130,  1024, "mn_unaligned"},
  };

  int failures = 0;
  uint64_t seed = 0x1278;
  for (auto s : shapes) failures += run_one(s, 1, 1.0f, 0.0f, seed++);
  for (auto s : shapes) failures += run_one(s, 8, 1.0f, 0.0f, seed++);
  /* alpha/beta sweep. */
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 0.5f,  0.0f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 1.0f,  0.5f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 2.0f, -1.0f, seed++);

  if (failures) { std::printf("FAIL: %d case(s)\n", failures); return 1; }
  std::printf("All AMX-INT8 cases passed.\n");
  return 0;
}
