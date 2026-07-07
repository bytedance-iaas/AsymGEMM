/*
 * AMX INT8 pre-packed-B path correctness.
 *
 * Verifies cg_pack_b_int8_amx + dispatcher with CG_INT8_PACKED_AMX produces
 * results bit-equal to the regular CG_INT8 path. Both paths share the same
 * INT32 accumulator and FP32 dequant, so equality is exact (not just within
 * a tolerance).
 *
 * Tests:
 *   1. Aligned and unaligned N, multiple Ks.
 *   2. Single- and multi-thread.
 *   3. alpha/beta sweep.
 *   4. Pre-pack size matches what the kernel actually consumes.
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu_gemm/cpu_gemm.hpp"
#include "reference.h"

using cpu_gemm::test::fill_bf16;
using cpu_gemm::test::fill_int8;
using cpu_gemm::test::fill_scales;

namespace {

struct Shape { int m, n, k; const char* tag; };

/* Aligned alloc to 64 bytes for the pre-pack buffer. */
void* aligned_alloc64(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  return p;
}

int run_one(Shape s, int n_threads, float alpha, float beta, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<int8_t>    b((size_t)s.n * s.k);
  std::vector<float>     b_scales((size_t)s.n);

  fill_bf16(a, seed);
  fill_int8(b, seed + 1);
  fill_scales(b_scales, seed + 2);

  std::vector<float> c_baseline((size_t)s.m * s.n);
  std::vector<float> c_prepack ((size_t)s.m * s.n);
  for (size_t i = 0; i < c_baseline.size(); ++i) {
    float v = 0.001f * (float)((i * 17) % 31) - 0.015f;
    c_baseline[i] = v;
    c_prepack[i]  = v;
  }

  cpu_gemm::Runtime rt(n_threads);

  /* --- Path A: regular CG_INT8 with per-call B-pack. --- */
  {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = alpha; d.beta = beta;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_INT8;
    d.b_scales = b_scales.data();
    d.c = c_baseline.data(); d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  }

  /* --- Path B: pre-pack B once, then CG_INT8_PACKED_AMX. --- */
  size_t pack_bytes = cg_pack_b_int8_amx_size((size_t)s.n, (size_t)s.k);
  if (pack_bytes == 0) {
    std::printf("pack_bytes==0 for n=%d k=%d (k not multiple of 64?). FAIL\n", s.n, s.k);
    return 1;
  }
  void* packed = aligned_alloc64(pack_bytes);
  cg_status_t st = cg_pack_b_int8_amx((size_t)s.n, (size_t)s.k,
                                      b.data(), b_scales.data(), packed);
  if (st != CG_OK) {
    std::printf("cg_pack_b_int8_amx returned %d. FAIL\n", (int)st);
    free(packed);
    return 1;
  }
  {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = alpha; d.beta = beta;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = packed;   d.ldb = s.k; d.dtype_b = CG_INT8_PACKED_AMX;
    /* b_scales intentionally NULL — packed buffer carries them. */
    d.b_scales = nullptr;
    d.c = c_prepack.data(); d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  }
  free(packed);

  /* Both paths execute the same packed-B → AMX kernel → unpack pipeline,
   * just with the pack done in different places. Output is bit-equal. */
  int mismatches = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < c_baseline.size(); ++i) {
    float d = c_baseline[i] - c_prepack[i];
    float ad = d < 0 ? -d : d;
    if (ad > max_abs) max_abs = ad;
    if (c_baseline[i] != c_prepack[i]) ++mismatches;
  }

  bool ok = mismatches == 0;
  std::printf("[%s thr=%d a=%.1f b=%.1f] m=%4d n=%4d k=%5d  mismatches=%d max_abs=%.3e %s\n",
              s.tag, n_threads, alpha, beta, s.m, s.n, s.k, mismatches, max_abs,
              ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  cg_caps_t caps = cg_query_caps();
  if (!caps.has_amx_int8) {
    std::printf("AMX-INT8 not present on this host; skipping.\n");
    return 0;
  }
  std::printf("AMX-INT8 detected. Running pre-pack tests...\n");

  /* Size sanity: a known shape. n_pad=64 (32-aligned), k=128 ⇒
   *   int8 bytes = 64*128 = 8192, scale bytes = 64*4 = 256 ⇒ 8448. */
  size_t got = cg_pack_b_int8_amx_size(64, 128);
  if (got != 8448) {
    std::printf("size mismatch: got %zu, expected 8448. FAIL\n", got);
    return 1;
  }
  /* Padding: n=33 should round up to 64 (next multiple of N_STEP=32). */
  size_t got2 = cg_pack_b_int8_amx_size(33, 128);
  if (got2 != 64 * 128 + 64 * 4) {
    std::printf("padding mismatch: got %zu, expected %d. FAIL\n",
                got2, 64 * 128 + 64 * 4);
    return 1;
  }

  const Shape shapes[] = {
      {1,    64,    64, "tiny"},
      {1,    64,   256, "decode"},
      {32,   64,   256, "M_STEP"},
      {16,  128,   512, "med"},
      {64,  256,  1024, "wide"},
      {8,    70,   256, "n_unaligned"},   // n→96
      {1,    96,   128, "n96"},
      {17,  130,  1024, "mn_unaligned"},  // n→160
  };

  int failures = 0;
  uint64_t seed = 0x9a2c;
  for (auto s : shapes) failures += run_one(s, 1, 1.0f, 0.0f, seed++);
  for (auto s : shapes) failures += run_one(s, 8, 1.0f, 0.0f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 0.5f,  0.0f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 1.0f,  0.5f, seed++);
  failures += run_one(Shape{32, 256, 1024, "ab"}, 8, 2.0f, -1.0f, seed++);

  if (failures) { std::printf("FAIL: %d case(s)\n", failures); return 1; }
  std::printf("All AMX-INT8 pre-pack cases passed.\n");
  return 0;
}
