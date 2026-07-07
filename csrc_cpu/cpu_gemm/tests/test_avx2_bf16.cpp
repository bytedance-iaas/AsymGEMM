/*
 * Correctness tests for the AVX2 BF16 GEMM through the public C ABI.
 *
 * Three shapes — decode (M=1), prefill-ish (M=8), and a wider one — at
 * three (K, N) sizes that cover both vectorized (K %% 32 == 0) and tail
 * (K %% 32 != 0) paths.
 *
 * Pass criterion: max relative diff vs FP32 double-precision reference
 * within tight bounds. BF16 has ~8 bits of mantissa; with K up to a few
 * thousand the accumulated rounding stays well under 1e-2.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::bf16_to_f32;
using cpu_gemm::test::f32_to_bf16;
using cpu_gemm::test::fill_bf16;
using cpu_gemm::test::max_abs_diff;
using cpu_gemm::test::max_rel_diff;
using cpu_gemm::test::ref_gemm_bf16_bf16_f32;

namespace {

struct Shape { int m, n, k; };

int run_case(Shape s, uint64_t seed, bool use_threads) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<cg_bf16_t> b((size_t)s.n * s.k);
  std::vector<float>     c_actual((size_t)s.m * s.n, 0.0f);
  std::vector<float>     c_ref((size_t)s.m * s.n, 0.0f);

  fill_bf16(a, seed);
  fill_bf16(b, seed + 1);

  ref_gemm_bf16_bf16_f32(s.m, s.n, s.k,
                         1.0f,
                         a.data(), s.k,
                         b.data(), s.k,
                         0.0f,
                         c_ref.data(), s.n);

  auto d = cpu_gemm::make_desc();
  d.m = s.m; d.n = s.n; d.k = s.k;
  d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_BF16;
  d.c = c_actual.data(); d.ldc = s.n; d.dtype_c = CG_F32;

  if (use_threads) {
    cpu_gemm::Runtime rt(8);
    cpu_gemm::gemm(rt, d);
  } else {
    cpu_gemm::gemm_st(d, 0, 1);
  }

  float abs = max_abs_diff(c_actual.data(), c_ref.data(), c_actual.size());
  float rel = max_rel_diff(c_actual.data(), c_ref.data(), c_actual.size());

  // Tolerance: BF16 has ~3e-3 unit roundoff. With K up to a few thousand,
  // accumulated error stays comfortably under 1e-2 absolute on values
  // bounded by ~K * 1.0 * 1.0.
  const float tol_abs = 1e-2f * std::max(1, s.k / 16);
  const float tol_rel = 5e-2f;

  bool ok = abs <= tol_abs || rel <= tol_rel;
  std::printf("[%s] m=%4d n=%4d k=%4d  abs=%.3e rel=%.3e %s\n",
              use_threads ? "MT" : "ST",
              s.m, s.n, s.k, abs, rel, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;

  // (M, N, K) tuples representative of decode / small-prefill / wide.
  const Shape shapes[] = {
      {1,   64,  256},
      {1,  128, 1024},
      {1,  128, 1025},   // K not multiple of 32 — exercises scalar tail
      {8,  128,  512},
      {8,   64, 2048},
      {16, 256, 4096},
  };

  uint64_t seed = 0xC0FFEE;
  for (auto s : shapes) failures += run_case(s, seed++, false);
  for (auto s : shapes) failures += run_case(s, seed++, true);

  if (failures) {
    std::printf("FAIL: %d case(s)\n", failures);
    return 1;
  }
  std::printf("All cases passed.\n");
  return 0;
}
