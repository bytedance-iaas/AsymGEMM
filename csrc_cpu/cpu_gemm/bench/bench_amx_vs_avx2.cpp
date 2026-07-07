/*
 * AMX vs AVX2 BF16 head-to-head benchmark.
 *
 * For each shape we run the same descriptor through cg_gemm() (which prefers
 * AMX when eligible) and cg_gemm_st() fanned out by hand (which only uses
 * AVX2). Both use the same runtime thread pool.
 *
 * Numbers reported: GFLOPS (2*M*N*K / time) and median-of-N latency in µs.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::fill_bf16;

namespace {

double median(std::vector<double>& xs) {
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

template <typename F>
double time_one_call(F&& f) {
  auto t0 = std::chrono::steady_clock::now();
  f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

struct Result { double gflops; double median_us; };

Result bench_amx(int m, int n, int k, cpu_gemm::Runtime& rt, int iters) {
  std::vector<cg_bf16_t> a((size_t)m * k);
  std::vector<cg_bf16_t> b((size_t)n * k);
  std::vector<float>     c((size_t)m * n, 0.0f);
  fill_bf16(a, 1); fill_bf16(b, 2);
  auto d = cpu_gemm::make_desc();
  d.m = m; d.n = n; d.k = k;
  d.a = a.data(); d.lda = k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = k; d.dtype_b = CG_BF16;
  d.c = c.data(); d.ldc = n; d.dtype_c = CG_F32;
  for (int i = 0; i < 3; ++i) cpu_gemm::gemm(rt, d);
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) {
    times[i] = time_one_call([&]() { cpu_gemm::gemm(rt, d); });
  }
  double med = median(times);
  double flops = 2.0 * (double)m * n * k;
  return {flops / med / 1e9, med * 1e6};
}

/* AVX2 path with multi-threaded fan-out — we drive cg_gemm_st (AVX2 only)
 * across N std::threads manually because the public Runtime API doesn't
 * expose a generic parallel_for. */
Result bench_avx2_mt(int m, int n, int k, int n_threads, int iters) {
  std::vector<cg_bf16_t> a((size_t)m * k);
  std::vector<cg_bf16_t> b((size_t)n * k);
  std::vector<float>     c((size_t)m * n, 0.0f);
  fill_bf16(a, 1); fill_bf16(b, 2);
  auto d = cpu_gemm::make_desc();
  d.m = m; d.n = n; d.k = k;
  d.a = a.data(); d.lda = k; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = k; d.dtype_b = CG_BF16;
  d.c = c.data(); d.ldc = n; d.dtype_c = CG_F32;

  auto run_once = [&]() {
    std::vector<std::thread> ths;
    ths.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
      ths.emplace_back([&, t]() { cpu_gemm::gemm_st(d, t, n_threads); });
    }
    for (auto& th : ths) th.join();
  };
  for (int i = 0; i < 3; ++i) run_once();
  std::vector<double> times(iters);
  for (int i = 0; i < iters; ++i) times[i] = time_one_call(run_once);
  double med = median(times);
  double flops = 2.0 * (double)m * n * k;
  return {flops / med / 1e9, med * 1e6};
}

}  // namespace

int main() {
  cpu_gemm::Runtime rt;
  cg_caps_t caps = cg_query_caps();
  std::printf("host: avx2=%d fma=%d avx512f=%d avx512bf16=%d avxvnni=%d amx_bf16=%d amx_int8=%d\n",
              caps.has_avx2, caps.has_fma, caps.has_avx512f, caps.has_avx512_bf16,
              caps.has_avx_vnni, caps.has_amx_bf16, caps.has_amx_int8);
  std::printf("threads: %d\n", rt.threads());
  /* Cap AVX2 thread count: the per-call work for small shapes is tiny and
   * fan-out overhead dominates if we use 192 threads. Use min(threads, 32)
   * which lets AVX2 scale wide without being totally absurd. */
  const int avx2_threads = std::min(rt.threads(), 32);

  std::printf("\nShape (M, N, K)             AVX2 %d-thread       AMX %d-thread        speedup\n",
              avx2_threads, rt.threads());
  std::printf("                            GFLOPS    µs/call    GFLOPS    µs/call\n");

  struct S { int m, n, k; const char* tag; };
  const S shapes[] = {
      {1,    4096,  4096, "decode-1"},
      {1,   14336,  4096, "decode-large"},
      {8,    4096,  4096, "prefill-8"},
      {32,   4096,  4096, "balanced"},
      {64,  14336,  4096, "ffn-up"},
      {128,  4096, 14336, "ffn-down"},
      {256,  4096,  4096, "wide"},
  };

  for (auto s : shapes) {
    auto rav = bench_avx2_mt(s.m, s.n, s.k, avx2_threads, 20);
    auto ram = bench_amx    (s.m, s.n, s.k, rt, 20);
    std::printf("[%-13s m=%4d n=%5d k=%5d] %8.1f  %8.1f   %8.1f  %8.1f   x%5.1f\n",
                s.tag, s.m, s.n, s.k,
                rav.gflops, rav.median_us, ram.gflops, ram.median_us,
                ram.gflops / rav.gflops);
  }
  return 0;
}
