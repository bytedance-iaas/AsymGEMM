/*
 * Minimum working example: one BF16 GEMM through the public C++ wrapper.
 *
 * Builds standalone — needs only -lcpu_gemm and the public headers.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"

static cg_bf16_t f2b(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint32_t tie = (bits >> 16) & 1u;
  uint32_t rounded = bits + 0x7fffu + tie;
  cg_bf16_t r;
  r.bits = (uint16_t)(rounded >> 16);
  return r;
}

int main() {
  constexpr int M = 2, N = 3, K = 4;

  // A = 2x4, B = 3x4 (each row is one output channel).
  std::vector<cg_bf16_t> a(M * K), b(N * K);
  for (int i = 0; i < M * K; ++i) a[i] = f2b((float)(i + 1));
  for (int i = 0; i < N * K; ++i) b[i] = f2b((float)(i + 1) * 0.25f);
  std::vector<float> c(M * N, 0.0f);

  cpu_gemm::Runtime rt;  // hardware_concurrency
  std::printf("threads: %d\n", rt.threads());

  auto d = cpu_gemm::make_desc();
  d.m = M; d.n = N; d.k = K;
  d.a = a.data(); d.lda = K; d.dtype_a = CG_BF16;
  d.b = b.data(); d.ldb = K; d.dtype_b = CG_BF16;
  d.c = c.data(); d.ldc = N; d.dtype_c = CG_F32;

  cpu_gemm::gemm(rt, d);

  std::printf("C =\n");
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) std::printf(" %8.3f", c[i * N + j]);
    std::printf("\n");
  }
  return 0;
}
