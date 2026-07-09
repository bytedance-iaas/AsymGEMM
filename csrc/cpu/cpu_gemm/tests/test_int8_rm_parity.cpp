/*
 * tests/test_int8_rm_parity.cpp — PR 3 of avx_512.md acceptance test.
 *
 * Run identical input through the AMX INT8 row-major backend and the
 * AVX-512-VNNI backend; assert the FP32 outputs agree within tolerance.
 *
 * The selector caches the chosen backend in a function-local static,
 * so once the process has called any backend-using function, the
 * choice is latched for the rest of that process. We work around this
 * by forking one child per backend: each child sets
 * ASYM_GEMM_FORCE_BACKEND, runs the GEMM, writes its FP32 output to a
 * temp file. The parent reads both and compares.
 *
 * Tolerance:
 *   - Both backends compute sum_k(int8_a × int8_b) as int32 → cast to
 *     fp32 × (a_scale × b_scale) per output element. The accumulation
 *     order can differ, but int32 sums are associative; the final
 *     cast to fp32 rounds identically because the scales are shared.
 *   - In practice we see max_abs_err ≈ 0; the gate is 1e-2 to be
 *     defensive against future rounding regressions.
 */
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "cpu_gemm/cpu_gemm.h"
#include "cpu_gemm/runtime.h"

#if defined(CPU_GEMM_HAS_AMX)
#include "kernels/amx/bf16_gemm.h"      /* amx_available() */
#endif

namespace {

uint16_t fp32_to_bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  uint32_t rb = 0x7FFF + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rb) >> 16);
}

bool run_gemm_to_file(const char* force_backend,
                     int m, int n, int k,
                     const std::vector<uint16_t>& a_bf16,
                     const std::vector<int8_t>&   b_int8,
                     const std::vector<float>&    b_scales,
                     int n_threads,
                     const char* out_path) {
  if (force_backend) {
    setenv("ASYM_GEMM_FORCE_BACKEND", force_backend, /*overwrite=*/1);
  } else {
    unsetenv("ASYM_GEMM_FORCE_BACKEND");
  }

  const char* name = cg_int8_rm_backend_name();
  std::fprintf(stderr, "  [child] backend = %s (forced = %s)\n", name,
               force_backend ? force_backend : "(none)");
  if (!cg_int8_rm_backend_ok()) {
    std::fprintf(stderr, "  [child] backend not OK; aborting child\n");
    return false;
  }

  cg_runtime_t* rt = cg_runtime_create(n_threads);
  if (!rt) {
    std::fprintf(stderr, "  [child] cg_runtime_create failed\n");
    return false;
  }

  std::vector<float> c((size_t)m * n, 0.0f);

  cg_gemm_desc_t d{};
  d.order = CG_ROW_MAJOR;
  d.trans_a = CG_NO_TRANS; d.trans_b = CG_TRANS;
  d.offset_c_mode = CG_OFFSET_C_NONE;
  d.m = (size_t)m; d.n = (size_t)n; d.k = (size_t)k;
  d.alpha = 1.0f; d.beta = 0.0f;
  d.a = a_bf16.data();  d.lda = (size_t)k; d.dtype_a = CG_BF16;
  d.b = b_int8.data();  d.ldb = (size_t)k; d.dtype_b = CG_INT8;
  d.b_scales = b_scales.data();
  d.c = c.data();       d.ldc = (size_t)n; d.dtype_c = CG_F32;

  cg_status_t s = cg_gemm(rt, &d);
  cg_runtime_destroy(rt);
  if (s != CG_OK) {
    std::fprintf(stderr, "  [child] cg_gemm returned %d for backend=%s\n",
                 (int)s, name);
    return false;
  }

  /* Persist the result for the parent. */
  FILE* fp = std::fopen(out_path, "wb");
  if (!fp) {
    std::fprintf(stderr, "  [child] fopen %s failed\n", out_path);
    return false;
  }
  std::fwrite(c.data(), sizeof(float), c.size(), fp);
  std::fclose(fp);
  return true;
}

std::vector<float> read_floats(const char* path, size_t expect) {
  std::vector<float> out(expect);
  FILE* fp = std::fopen(path, "rb");
  if (!fp) {
    std::fprintf(stderr, "fopen(%s) failed\n", path);
    out.clear();
    return out;
  }
  size_t got = std::fread(out.data(), sizeof(float), expect, fp);
  std::fclose(fp);
  if (got != expect) {
    std::fprintf(stderr, "short read from %s: %zu/%zu\n", path, got, expect);
    out.clear();
  }
  return out;
}

int compare_results(const std::vector<float>& a,
                    const std::vector<float>& b,
                    float tol) {
  if (a.empty() || b.empty() || a.size() != b.size()) {
    std::fprintf(stderr, "shape mismatch / empty: %zu vs %zu\n",
                 a.size(), b.size());
    return 1;
  }
  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  size_t worst_i = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    float ae = std::fabs(a[i] - b[i]);
    float denom = std::fabs(a[i]) + std::fabs(b[i]) + 1e-9f;
    float re = ae / denom;
    if (ae > max_abs_err) { max_abs_err = ae; worst_i = i; }
    if (re > max_rel_err) max_rel_err = re;
  }
  std::printf("  max_abs_err = %.6e, max_rel_err = %.6e\n",
              max_abs_err, max_rel_err);
  std::printf("  worst-case i=%zu: amx=%.6f vnni=%.6f\n",
              worst_i, a[worst_i], b[worst_i]);
  return max_abs_err <= tol ? 0 : 1;
}

}  // namespace

int main() {
  constexpr float TOL = 1e-2f;

#if !defined(CPU_GEMM_HAS_AMX) || !defined(CPU_GEMM_HAS_AVX512_VNNI)
  std::printf("[SKIP] need both AMX and AVX-512-VNNI builds\n");
  return 0;
#else

  if (!cpu_gemm::kernels::amx::amx_available()) {
    std::printf("[SKIP] AMX not available on this host\n");
    return 0;
  }

  /* Shape: k=128 (mult of 64 and 4); n=64 (mult of 32 and 16);
   * m=5 (exercises M_STEP padding on both sides). */
  const int M = 5, N = 64, K = 128;
  const int N_THREADS = 4;

  std::printf("[test_int8_rm_parity] m=%d n=%d k=%d threads=%d tol=%.1e\n",
              M, N, K, N_THREADS, TOL);

  /* Deterministic random inputs. */
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int>    b_dist(-100, 100);
  std::uniform_real_distribution<float> s_dist(0.01f, 0.5f);

  std::vector<uint16_t> a_bf16(M * K);
  for (auto& v : a_bf16) v = fp32_to_bf16_bits(a_dist(rng));

  std::vector<int8_t> b_int8(N * K);
  for (auto& v : b_int8) v = static_cast<int8_t>(b_dist(rng));

  std::vector<float> b_scales(N);
  for (auto& v : b_scales) v = s_dist(rng);

  const char* amx_path  = "/tmp/test_int8_rm_parity_amx.bin";
  const char* vnni_path = "/tmp/test_int8_rm_parity_vnni.bin";
  /* Best-effort clean-up of stale files from a prior failed run. */
  std::remove(amx_path);
  std::remove(vnni_path);

  pid_t pid_amx = fork();
  if (pid_amx == 0) {
    bool ok = run_gemm_to_file("amx", M, N, K, a_bf16, b_int8, b_scales,
                               N_THREADS, amx_path);
    std::_Exit(ok ? 0 : 1);
  }
  int status_amx = 0;
  waitpid(pid_amx, &status_amx, 0);
  if (!WIFEXITED(status_amx) || WEXITSTATUS(status_amx) != 0) {
    std::fprintf(stderr, "AMX child failed (status %d)\n", status_amx);
    return 1;
  }

  pid_t pid_vnni = fork();
  if (pid_vnni == 0) {
    bool ok = run_gemm_to_file("avx512", M, N, K, a_bf16, b_int8, b_scales,
                               N_THREADS, vnni_path);
    std::_Exit(ok ? 0 : 1);
  }
  int status_vnni = 0;
  waitpid(pid_vnni, &status_vnni, 0);
  if (!WIFEXITED(status_vnni) || WEXITSTATUS(status_vnni) != 0) {
    std::fprintf(stderr, "VNNI child failed (status %d)\n", status_vnni);
    return 1;
  }

  auto c_amx  = read_floats(amx_path,  (size_t)M * N);
  auto c_vnni = read_floats(vnni_path, (size_t)M * N);

  /* Clean up; not fatal if the unlink fails. */
  std::remove(amx_path);
  std::remove(vnni_path);

  std::printf("=== Parity ===\n");
  int rc = compare_results(c_amx, c_vnni, TOL);
  std::printf(rc == 0 ? "[PASS] AMX and AVX-512-VNNI agree within tol\n"
                      : "[FAIL] divergence exceeds tolerance\n");
  return rc;
#endif
}
