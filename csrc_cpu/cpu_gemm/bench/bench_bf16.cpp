/*
 * Microbenchmark for the BF16 GEMM path.
 *
 * Reports GFLOPS and per-call latency at three shapes:
 *   - decode (M=1, N=4096, K=4096)
 *   - small prefill (M=8, N=14336, K=4096)
 *   - balanced (M=32, N=4096, K=4096)
 */
#include <chrono>
#include <cstdio>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::fill_bf16;

namespace {

double bench_once(int m, int n, int k, cpu_gemm::Runtime& rt, int iters) {
  std::vector<cg_bf16_t> a((size_t)m * k);
  std::vector<cg_bf16_t> b((size_t)n * k);
  std::vector<float>     c((size_t)m * n, 0.0f);
  fill_bf16(a, 1);
  fill_bf16(b, 2);

  auto d = cpu_gemm::make_desc();
  d.m = m; d.n = n; d.k = k;
  d.a = a.data(); d.lda = k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = k; d.dtype_b = CG_BF16;
  d.c = c.data(); d.ldc = n; d.dtype_c = CG_F32;

  // Warmup.
  for (int i = 0; i < 3; ++i) cpu_gemm::gemm(rt, d);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) cpu_gemm::gemm(rt, d);
  auto t1 = std::chrono::steady_clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count() / iters;
  double flops = 2.0 * (double)m * n * k;
  return flops / secs / 1e9;  // GFLOPS
}

}  // namespace

int main() {
  cpu_gemm::Runtime rt;
  std::printf("threads: %d\n", rt.threads());

  struct Shape { int m, n, k; const char* tag; };
  const Shape shapes[] = {
      {1,   4096, 4096, "decode"},
      {8,  14336, 4096, "prefill_small"},
      {32,  4096, 4096, "balanced"},
  };

  for (auto s : shapes) {
    double gflops = bench_once(s.m, s.n, s.k, rt, 20);
    std::printf("[%-13s] m=%3d n=%5d k=%4d  %.2f GFLOPS\n",
                s.tag, s.m, s.n, s.k, gflops);
  }
  return 0;
}
