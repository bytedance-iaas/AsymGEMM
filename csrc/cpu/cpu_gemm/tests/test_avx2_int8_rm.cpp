/*
 * tests/test_avx2_int8_rm.cpp — correctness test for the AVX2 INT8
 * row-major backend against a scalar reference.
 *
 * The scalar reference replicates the kernel's own A quantization
 * (per-row amax/127 symmetric, lrintf, clamp to ±127) and computes the
 * int32 dot product exactly, then applies the same fp32 scale product
 * in the same order — so the outputs must match bit-for-bit up to fp32
 * rounding of identical expressions (tolerance kept small but nonzero
 * to be defensive).
 *
 * The backend selector latches its choice in a function-local static,
 * so ASYM_GEMM_FORCE_BACKEND=avx2 is set before the first
 * backend-touching call.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_gemm/cpu_gemm.h"
#include "cpu_gemm/runtime.h"

namespace {

uint16_t fp32_to_bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t rb = 0x7FFF + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rb) >> 16);
}

float bf16_bits_to_fp32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

}  // namespace

int main() {
  constexpr float TOL = 1e-4f;

  setenv("ASYM_GEMM_FORCE_BACKEND", "avx2", /*overwrite=*/1);

  const char* name = cg_int8_rm_backend_name();
  std::printf("[test_avx2_int8_rm] backend = %s\n", name);
  if (!cg_int8_rm_backend_ok() || std::strcmp(name, "avx2_int8_rm") != 0) {
    std::printf("[SKIP] AVX2 INT8 backend not selectable on this host\n");
    return 0;
  }

  /* m=5 exercises M_STEP padding; k=132 exercises the K tail path
   * (132 % 4 == 0 but the shape is otherwise arbitrary); n=72 is a
   * multiple of N_STEP=8 but not of N_BLOCK=64. */
  const int M = 5, N = 72, K = 132;
  const int N_THREADS = 4;

  std::printf("[test_avx2_int8_rm] m=%d n=%d k=%d threads=%d tol=%.1e\n",
              M, N, K, N_THREADS, TOL);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int>    b_dist(-127, 127);
  std::uniform_real_distribution<float> s_dist(0.01f, 0.5f);

  std::vector<uint16_t> a_bf16(M * K);
  for (auto& v : a_bf16) v = fp32_to_bf16_bits(a_dist(rng));

  std::vector<int8_t> b_int8(N * K);
  for (auto& v : b_int8) v = static_cast<int8_t>(b_dist(rng));

  std::vector<float> b_scales(N);
  for (auto& v : b_scales) v = s_dist(rng);

  /* --- Kernel run. --- */
  cg_runtime_t* rt = cg_runtime_create(N_THREADS);
  if (!rt) {
    std::fprintf(stderr, "cg_runtime_create failed\n");
    return 1;
  }

  std::vector<float> c((size_t)M * N, 0.0f);

  cg_gemm_desc_t d{};
  d.order = CG_ROW_MAJOR;
  d.trans_a = CG_NO_TRANS; d.trans_b = CG_TRANS;
  d.offset_c_mode = CG_OFFSET_C_NONE;
  d.m = (size_t)M; d.n = (size_t)N; d.k = (size_t)K;
  d.alpha = 1.0f; d.beta = 0.0f;
  d.a = a_bf16.data();  d.lda = (size_t)K; d.dtype_a = CG_BF16;
  d.b = b_int8.data();  d.ldb = (size_t)K; d.dtype_b = CG_INT8;
  d.b_scales = b_scales.data();
  d.c = c.data();       d.ldc = (size_t)N; d.dtype_c = CG_F32;

  cg_status_t s = cg_gemm(rt, &d);
  cg_runtime_destroy(rt);
  if (s != CG_OK) {
    std::fprintf(stderr, "cg_gemm returned %d\n", (int)s);
    return 1;
  }

  /* --- Scalar reference (replicates the kernel's A quantization). --- */
  float max_abs_err = 0.0f;
  size_t worst_i = 0;
  float worst_ref = 0.0f, worst_got = 0.0f;
  for (int mi = 0; mi < M; ++mi) {
    float max_abs = 0.0f;
    for (int ki = 0; ki < K; ++ki) {
      float v = bf16_bits_to_fp32(a_bf16[mi * K + ki]);
      float av = std::fabs(v);
      if (av > max_abs) max_abs = av;
    }
    float a_scale = max_abs > 0.0f ? max_abs / 127.0f : 1e-12f;
    float inv_scale = 1.0f / a_scale;

    std::vector<int8_t> qa(K);
    for (int ki = 0; ki < K; ++ki) {
      float v = bf16_bits_to_fp32(a_bf16[mi * K + ki]);
      int q = (int)std::lrintf(v * inv_scale);
      if (q < -127) q = -127;
      if (q >  127) q =  127;
      qa[ki] = static_cast<int8_t>(q);
    }

    for (int ni = 0; ni < N; ++ni) {
      int32_t acc = 0;
      const int8_t* brow = b_int8.data() + (size_t)ni * K;
      for (int ki = 0; ki < K; ++ki)
        acc += (int32_t)qa[ki] * (int32_t)brow[ki];
      float ref = static_cast<float>(acc) * a_scale * b_scales[ni];

      size_t i = (size_t)mi * N + ni;
      float ae = std::fabs(ref - c[i]);
      if (ae > max_abs_err) {
        max_abs_err = ae; worst_i = i; worst_ref = ref; worst_got = c[i];
      }
    }
  }

  std::printf("  max_abs_err = %.6e (worst i=%zu: ref=%.6f got=%.6f)\n",
              max_abs_err, worst_i, worst_ref, worst_got);
  int rc = max_abs_err <= TOL ? 0 : 1;
  std::printf(rc == 0 ? "[PASS] AVX2 INT8 rm matches scalar reference\n"
                      : "[FAIL] divergence exceeds tolerance\n");
  return rc;
}
