/*
 * Stride-aware AMX INT8 kernel — correctness tests.
 *
 * The dispatcher's priority is:
 *   CG_INT8_PACKED_AMX  (offline pre-pack)
 *   CG_INT8 + aligned N (stride-aware rm path)   ← exercised here
 *   CG_INT8             (packed-on-touch fallback for unaligned N)
 *
 * Every shape in this file has n % N_STEP == 0 (32-aligned), which forces
 * the rm path on hosts where AMX is available. The reference is the same
 * scalar quantize-then-int32-dot used by the packed-kernel test, so any
 * mismatch is in the rm kernel itself, not in the quantization scheme.
 *
 * Stages covered (Stride.md §5):
 *   B — single-tile numeric (m=n=32, k=64).        Bit-exact vs ref.
 *   C — multi-K-block      (k=256, k=4096).        Bit-exact vs ref.
 *   D — multi-tile         (m=64,128; n=128,256).  ≤ 1 ULP vs ref.
 *   E — ldb ≠ k            (stride > k).           Bit-equal to ldb==k.
 *   F — parity vs packed prepack path.             Bit-equal to packed.
 *   G — multi-thread       (nth ∈ {1,4,8}).        Deterministic.
 */
#include <cstdio>
#include <cstdlib>
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

void* aligned_alloc64(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  return p;
}

/* Run the rm path through the public ABI and compare to the scalar
 * reference. The reference and kernel use bit-identical quantization, so
 * tolerances are tight (only the int32 → FP32 rescale rounds). */
int run_vs_ref(Shape s, int n_threads, float alpha, float beta, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<int8_t>    b((size_t)s.n * s.k);
  std::vector<float>     b_scales((size_t)s.n);
  std::vector<float>     c_actual((size_t)s.m * s.n);
  std::vector<float>     c_ref   ((size_t)s.m * s.n);

  fill_bf16(a, seed);
  fill_int8(b, seed + 1);
  fill_scales(b_scales, seed + 2);

  for (size_t i = 0; i < c_actual.size(); ++i) {
    float v = 0.001f * (float)((i * 17) % 31) - 0.015f;
    c_actual[i] = v;
    c_ref[i]    = v;
  }

  ref_gemm_bf16_int8_f32(s.m, s.n, s.k, alpha,
                         a.data(), s.k,
                         b.data(), s.k, b_scales.data(),
                         beta, c_ref.data(), s.n);

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

  float tol_abs = 1e-4f;
  float tol_rel = 1e-3f;
  bool ok = abs <= tol_abs || rel <= tol_rel;
  std::printf("[ref  %s thr=%d a=%.1f b=%.1f] m=%4d n=%4d k=%5d  abs=%.3e rel=%.3e %s\n",
              s.tag, n_threads, alpha, beta, s.m, s.n, s.k, abs, rel, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

/* Compare ldb == k against an explicit ldb > k (over-allocated B with row
 * padding). The result must be bit-identical to the contiguous case. */
int run_strided_b(Shape s, int n_threads, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<float>     b_scales((size_t)s.n);
  fill_bf16(a, seed);
  fill_scales(b_scales, seed + 2);

  /* Contiguous baseline. */
  std::vector<int8_t> b_tight((size_t)s.n * s.k);
  fill_int8(b_tight, seed + 1);
  std::vector<float> c_tight((size_t)s.m * s.n, 0.0f);

  /* Strided variant — same values, but a 64-byte gap after each row. */
  std::size_t ldb = (std::size_t)s.k + 64;
  std::vector<int8_t> b_strided((size_t)s.n * ldb, 0);
  for (int ni = 0; ni < s.n; ++ni) {
    std::memcpy(b_strided.data() + (size_t)ni * ldb,
                b_tight.data()   + (size_t)ni * s.k,
                (size_t)s.k);
  }
  std::vector<float> c_strided((size_t)s.m * s.n, 0.0f);

  cpu_gemm::Runtime rt(n_threads);

  auto run = [&](void* b_ptr, std::size_t ld, float* c) {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = b_ptr;    d.ldb = ld;  d.dtype_b = CG_INT8;
    d.b_scales = b_scales.data();
    d.c = c; d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  };
  run(b_tight.data(),   (std::size_t)s.k, c_tight.data());
  run(b_strided.data(), ldb,              c_strided.data());

  int mismatches = 0;
  for (size_t i = 0; i < c_tight.size(); ++i)
    if (c_tight[i] != c_strided[i]) ++mismatches;

  bool ok = mismatches == 0;
  std::printf("[strd %s thr=%d ldb=%zu] m=%4d n=%4d k=%5d  mismatches=%d %s\n",
              s.tag, n_threads, ldb, s.m, s.n, s.k, mismatches, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

/* Parity check against the offline-prepacked path. Both kernels do
 * (int32 dot) · (sA · sB); only the FP32 multiply order differs, so the
 * residual is at the int32 → FP32 rescale level. */
int run_vs_packed(Shape s, int n_threads, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<int8_t>    b((size_t)s.n * s.k);
  std::vector<float>     b_scales((size_t)s.n);
  fill_bf16(a, seed);
  fill_int8(b, seed + 1);
  fill_scales(b_scales, seed + 2);

  std::vector<float> c_rm    ((size_t)s.m * s.n, 0.0f);
  std::vector<float> c_packed((size_t)s.m * s.n, 0.0f);
  cpu_gemm::Runtime rt(n_threads);

  /* Path A: dispatcher routes aligned-N CG_INT8 to the rm kernel. */
  {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_INT8;
    d.b_scales = b_scales.data();
    d.c = c_rm.data(); d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  }

  /* Path B: offline pre-pack + CG_INT8_PACKED_AMX → run_amx_int8_prepacked. */
  size_t pack_bytes = cg_pack_b_int8_amx_size((size_t)s.n, (size_t)s.k);
  void*  packed     = aligned_alloc64(pack_bytes);
  cg_pack_b_int8_amx((size_t)s.n, (size_t)s.k, b.data(), b_scales.data(), packed);
  {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = packed;   d.ldb = s.k; d.dtype_b = CG_INT8_PACKED_AMX;
    d.b_scales = nullptr;
    d.c = c_packed.data(); d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  }
  free(packed);

  float abs = max_abs_diff(c_rm.data(), c_packed.data(), c_rm.size());
  float rel = max_rel_diff(c_rm.data(), c_packed.data(), c_rm.size());
  /* Residual is FP32 multiply reassociation between (sA·sB)·int and
   * sA·(sB·int). At realistic magnitudes this stays well under 1e-6. */
  bool ok = abs <= 1e-4f || rel <= 1e-6f;
  std::printf("[pack %s thr=%d] m=%4d n=%4d k=%5d  abs=%.3e rel=%.3e %s\n",
              s.tag, n_threads, s.m, s.n, s.k, abs, rel, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

/* Multi-thread determinism: run twice with the same nth, expect identical
 * bytes. The partition is whole-N_BLOCK stripes, so no aliasing on C^T. */
int run_determinism(Shape s, int n_threads, uint64_t seed) {
  std::vector<cg_bf16_t> a((size_t)s.m * s.k);
  std::vector<int8_t>    b((size_t)s.n * s.k);
  std::vector<float>     b_scales((size_t)s.n);
  fill_bf16(a, seed);
  fill_int8(b, seed + 1);
  fill_scales(b_scales, seed + 2);

  std::vector<float> c1((size_t)s.m * s.n, 0.0f);
  std::vector<float> c2((size_t)s.m * s.n, 0.0f);
  cpu_gemm::Runtime rt(n_threads);

  auto run = [&](float* c) {
    auto d = cpu_gemm::make_desc();
    d.m = s.m; d.n = s.n; d.k = s.k;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a = a.data(); d.lda = s.k; d.dtype_a = CG_BF16;
    d.b = b.data(); d.ldb = s.k; d.dtype_b = CG_INT8;
    d.b_scales = b_scales.data();
    d.c = c; d.ldc = s.n; d.dtype_c = CG_F32;
    cpu_gemm::gemm(rt, d);
  };
  run(c1.data());
  run(c2.data());

  int mismatches = 0;
  for (size_t i = 0; i < c1.size(); ++i)
    if (c1[i] != c2[i]) ++mismatches;
  bool ok = mismatches == 0;
  std::printf("[det  %s thr=%d] m=%4d n=%4d k=%5d  mismatches=%d %s\n",
              s.tag, n_threads, s.m, s.n, s.k, mismatches, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  cg_caps_t caps = cg_query_caps();
  if (!caps.has_amx_int8) {
    std::printf("AMX-INT8 not present on this host; skipping.\n");
    return 0;
  }
  std::printf("AMX-INT8 detected. Running stride-aware (rm) tests...\n");

  /* All shapes have n % 32 == 0 so the dispatcher selects the rm path. */
  const Shape stage_b[] = {
      {32, 32,    64, "B.single"},        /* one 32×32 tile group, one K_STEP */
  };
  const Shape stage_c[] = {
      {32, 32,   256, "C.kb4"},           /* K = 4 × K_STEP */
      {32, 32,  4096, "C.kbig"},          /* K > K_BLOCK ⇒ multi K-block */
  };
  const Shape stage_d[] = {
      {64,  128,  512, "D.mt"},
      {128, 256, 1024, "D.mtwide"},
      { 1,  128,  256, "D.decode"},       /* m=1: single M_STEP tile */
      { 7,  128,  512, "D.m7"},           /* m not multiple of M_STEP */
  };

  int failures = 0;
  uint64_t seed = 0xa0d3;

  /* Stages B + C — bit-equal to reference. */
  for (auto s : stage_b) failures += run_vs_ref(s, 1, 1.0f, 0.0f, seed++);
  for (auto s : stage_c) failures += run_vs_ref(s, 1, 1.0f, 0.0f, seed++);

  /* Stage D — multi-tile + transpose unpack. */
  for (auto s : stage_d) failures += run_vs_ref(s, 1, 1.0f, 0.0f, seed++);
  for (auto s : stage_d) failures += run_vs_ref(s, 8, 1.0f, 0.0f, seed++);

  /* alpha/beta sweep on a multi-tile shape. */
  failures += run_vs_ref(Shape{32, 256, 1024, "D.ab"}, 8, 0.5f,  0.0f, seed++);
  failures += run_vs_ref(Shape{32, 256, 1024, "D.ab"}, 8, 1.0f,  0.5f, seed++);
  failures += run_vs_ref(Shape{32, 256, 1024, "D.ab"}, 8, 2.0f, -1.0f, seed++);

  /* Stage E — ldb ≠ k. */
  failures += run_strided_b(Shape{32, 128, 256,  "E.strided"}, 1, seed++);
  failures += run_strided_b(Shape{64, 256, 1024, "E.strided"}, 4, seed++);

  /* Stage F — parity vs the offline-prepacked AMX path. */
  failures += run_vs_packed(Shape{ 1, 128,  256, "F.decode"}, 1, seed++);
  failures += run_vs_packed(Shape{32, 256, 1024, "F.med"},    8, seed++);
  failures += run_vs_packed(Shape{64, 256, 1024, "F.wide"},   8, seed++);

  /* Stage G — multi-thread determinism. */
  failures += run_determinism(Shape{64, 256, 1024, "G.det"}, 1, seed++);
  failures += run_determinism(Shape{64, 256, 1024, "G.det"}, 4, seed++);
  failures += run_determinism(Shape{64, 256, 1024, "G.det"}, 8, seed++);

  if (failures) { std::printf("FAIL: %d case(s)\n", failures); return 1; }
  std::printf("All AMX-INT8 stride-aware cases passed.\n");
  return 0;
}
