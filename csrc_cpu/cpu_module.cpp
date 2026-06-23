/*
 * asym_gemm._cpu_C — pybind11 wrapper around cpu_gemm for use by the
 * Python-side unified MoE runtime (asym_gemm.unified_moe.Layer).
 *
 * This extension is independent of asym_gemm._C (the CUDA extension): it
 * has no torch dependency, and a host without CUDA still builds and loads
 * this module. Runtime availability of the AMX path is reported via
 * caps()['has_amx_int8'].
 *
 * Uses py::array_t<T> typed parameters so pybind11 enforces dtype matches
 * on the Python boundary (auto-converting where safe; rejecting otherwise).
 * Arrays are required C-contiguous via py::array::c_style.
 */
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpu_gemm/cpu_gemm.h"
#include "cpu_gemm/cpu_gemm.hpp"
// Internal headers: the MoE batch entry drives the worker pool directly
// (runtime/runtime_impl.h exposes cg_runtime::pool, worker_pool.h its API).
#include "runtime/runtime_impl.h"
#include "runtime/worker_pool.h"

namespace py = pybind11;

namespace {

void check(cg_status_t s, const char* where) {
  if (s == CG_OK) return;
  throw std::runtime_error(std::string(where) + " failed: status=" + std::to_string((int)s));
}

struct RuntimeHandle {
  cg_runtime_t* rt = nullptr;
  // One single-thread runtime per concurrent MoE work item. A 1-thread
  // runtime executes cg_gemm fully inline (no pool wake) and owns its own
  // scratch arena, so concurrent use from distinct pool workers is safe.
  std::vector<std::unique_ptr<RuntimeHandle>> serial_rts;

  explicit RuntimeHandle(int n_threads) {
    rt = cg_runtime_create(n_threads);
    if (!rt) throw std::runtime_error("cg_runtime_create returned NULL");
  }
  ~RuntimeHandle() { if (rt) cg_runtime_destroy(rt); }
  int threads() const { return cg_runtime_threads(rt); }

  void ensure_serial(size_t count) {
    while (serial_rts.size() < count)
      serial_rts.emplace_back(std::make_unique<RuntimeHandle>(1));
  }
};

inline uint16_t fp32_to_bf16_rne(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)((u + rounding_bias) >> 16);
}

py::dict caps_dict() {
  cg_caps_t c = cg_query_caps();
  py::dict d;
  d["has_avx2"]        = (bool)c.has_avx2;
  d["has_fma"]         = (bool)c.has_fma;
  d["has_avx512f"]     = (bool)c.has_avx512f;
  d["has_avx512_bf16"] = (bool)c.has_avx512_bf16;
  d["has_avx_vnni"]    = (bool)c.has_avx_vnni;
  d["has_amx_bf16"]    = (bool)c.has_amx_bf16;
  d["has_amx_int8"]    = (bool)c.has_amx_int8;
  return d;
}

size_t pack_b_int8_amx_size_py(size_t n, size_t k) {
  return cg_pack_b_int8_amx_size(n, k);
}

py::array_t<uint8_t> pack_b_int8_amx_py(
    py::array_t<int8_t,  py::array::c_style | py::array::forcecast> b_int8,
    py::array_t<float,   py::array::c_style | py::array::forcecast> b_scales) {
  if (b_int8.ndim() != 2)
    throw std::invalid_argument("b_int8 must be 2-D");
  if (b_scales.ndim() != 1)
    throw std::invalid_argument("b_scales must be 1-D");
  ssize_t n = b_int8.shape(0);
  ssize_t k = b_int8.shape(1);
  if (b_scales.shape(0) != n)
    throw std::invalid_argument("b_scales length != b_int8 rows");

  size_t bytes = cg_pack_b_int8_amx_size((size_t)n, (size_t)k);
  if (bytes == 0)
    throw std::invalid_argument("k not multiple of 64 or n==0");

  void* buf = nullptr;
  if (posix_memalign(&buf, 64, bytes) != 0 || !buf)
    throw std::bad_alloc();

  cg_status_t st = cg_pack_b_int8_amx(
      (size_t)n, (size_t)k, b_int8.data(), b_scales.data(), buf);
  if (st != CG_OK) { free(buf); check(st, "cg_pack_b_int8_amx"); }

  py::capsule owner(buf, [](void* p) { free(p); });
  return py::array_t<uint8_t>(
      {(ssize_t)bytes}, {(ssize_t)1}, static_cast<uint8_t*>(buf), owner);
}

void gemm_bf16_int8_py(
    RuntimeHandle& rt,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> a_bf16,
    py::array_t<int8_t,   py::array::c_style | py::array::forcecast> b_int8,
    py::array_t<float,    py::array::c_style | py::array::forcecast> b_scales,
    py::array_t<float,    py::array::c_style>                         c_fp32,
    float alpha, float beta) {
  if (a_bf16.ndim() != 2 || b_int8.ndim() != 2 || c_fp32.ndim() != 2 || b_scales.ndim() != 1)
    throw std::invalid_argument("bad rank");
  ssize_t m = a_bf16.shape(0), k = a_bf16.shape(1);
  ssize_t n = b_int8.shape(0);
  if (b_int8.shape(1) != k) throw std::invalid_argument("b_int8 k mismatch");
  if (c_fp32.shape(0) != m || c_fp32.shape(1) != n)
    throw std::invalid_argument("c_fp32 shape mismatch");
  if (b_scales.shape(0) != n) throw std::invalid_argument("b_scales length mismatch");
  if (!c_fp32.writeable()) throw std::invalid_argument("c_fp32 must be writeable");

  auto d = cpu_gemm::make_desc();
  d.m = (size_t)m; d.n = (size_t)n; d.k = (size_t)k;
  d.alpha = alpha; d.beta = beta;
  d.a = a_bf16.data();   d.lda = (size_t)k; d.dtype_a = CG_BF16;
  d.b = b_int8.data();   d.ldb = (size_t)k; d.dtype_b = CG_INT8;
  d.b_scales = b_scales.data();
  d.c = c_fp32.mutable_data(); d.ldc = (size_t)n; d.dtype_c = CG_F32;

  py::gil_scoped_release rel;
  check(cg_gemm(rt.rt, &d), "cg_gemm (BF16xINT8)");
}

void gemm_bf16_int8_packed_py(
    RuntimeHandle& rt,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> a_bf16,
    py::array_t<uint8_t,  py::array::c_style>                         b_packed,
    py::array_t<float,    py::array::c_style>                         c_fp32,
    size_t n, size_t k,
    float alpha, float beta) {
  if (a_bf16.ndim() != 2 || c_fp32.ndim() != 2 || b_packed.ndim() != 1)
    throw std::invalid_argument("bad rank");
  ssize_t m = a_bf16.shape(0);
  if ((ssize_t)k != a_bf16.shape(1)) throw std::invalid_argument("a_bf16 k mismatch");
  if (c_fp32.shape(0) != m || (size_t)c_fp32.shape(1) != n)
    throw std::invalid_argument("c_fp32 shape mismatch");

  size_t expected = cg_pack_b_int8_amx_size(n, k);
  if ((size_t)b_packed.shape(0) != expected)
    throw std::invalid_argument("b_packed size != cg_pack_b_int8_amx_size(n,k)");
  if (reinterpret_cast<std::uintptr_t>(b_packed.data()) % 64 != 0)
    throw std::invalid_argument("b_packed must be 64-byte aligned");
  if (!c_fp32.writeable()) throw std::invalid_argument("c_fp32 must be writeable");

  auto d = cpu_gemm::make_desc();
  d.m = (size_t)m; d.n = n; d.k = k;
  d.alpha = alpha; d.beta = beta;
  d.a = a_bf16.data();   d.lda = k; d.dtype_a = CG_BF16;
  d.b = b_packed.data(); d.ldb = k; d.dtype_b = CG_INT8_PACKED_AMX;
  d.b_scales = nullptr;
  d.c = c_fp32.mutable_data(); d.ldc = n; d.dtype_c = CG_F32;

  py::gil_scoped_release rel;
  check(cg_gemm(rt.rt, &d), "cg_gemm (BF16xINT8_PACKED)");
}

/*
 * One MoE layer's whole CPU bucket in a single worker-pool job.
 *
 * Each CPU-bucket expert (gate, up, silu, down) is one work item fanned out
 * over the runtime's worker pool via parallel_for — ONE pool wake for the
 * whole bucket instead of (experts x 3 GEMMs) separate cg_gemm launches.
 * This recovers cross-expert parallelism: at single-token decode every
 * routed expert holds one row, so the per-expert GEMMs are tiny and the
 * launch/sync overhead of doing them one-by-one dominated.
 *
 * Row-major (v3) variant: weights are read in place from the shared pinned
 * row-major INT8 slab — the same bytes the SM90 GPU kernel reads — indexed
 * by global expert id. No second AMX-packed copy.
 */
void moe_expert_forward_batch_py(
    RuntimeHandle& rt,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> x_cat,      // [sum_m, hidden] bf16 bits
    py::array_t<int64_t,  py::array::c_style | py::array::forcecast> m_offsets,  // [E+1]
    py::array_t<int64_t,  py::array::c_style | py::array::forcecast> expert_ids, // [E] global expert id per item
    py::array_t<int8_t,   py::array::c_style> gate_int8,   // [G, inter, hidden]
    py::array_t<float,    py::array::c_style> gate_scales, // [G, inter]
    py::array_t<int8_t,   py::array::c_style> up_int8,     // [G, inter, hidden]
    py::array_t<float,    py::array::c_style> up_scales,   // [G, inter]
    py::array_t<int8_t,   py::array::c_style> down_int8,   // [G, hidden, inter]
    py::array_t<float,    py::array::c_style> down_scales, // [G, hidden]
    py::array_t<float,    py::array::c_style> out_cat,     // [sum_m, hidden] fp32
    size_t inter, size_t hidden) {
  if (x_cat.ndim() != 2 || out_cat.ndim() != 2 ||
      m_offsets.ndim() != 1 || expert_ids.ndim() != 1)
    throw std::invalid_argument("bad rank");
  const ssize_t sum_m = x_cat.shape(0);
  if ((size_t)x_cat.shape(1) != hidden) throw std::invalid_argument("x_cat k mismatch");
  if (out_cat.shape(0) != sum_m || (size_t)out_cat.shape(1) != hidden)
    throw std::invalid_argument("out_cat shape mismatch");
  if (!out_cat.writeable()) throw std::invalid_argument("out_cat must be writeable");

  const ssize_t E = m_offsets.shape(0) - 1;
  if (E < 0 || expert_ids.shape(0) != E)
    throw std::invalid_argument("expert_ids length != len(m_offsets)-1");
  if (E == 0) return;

  const int64_t  G          = gate_int8.shape(0);
  const int64_t* off        = m_offsets.data();
  const int64_t* eid        = expert_ids.data();
  const uint16_t* x_base    = x_cat.data();
  float*          out_base  = out_cat.mutable_data();
  const int8_t*  gate_b     = gate_int8.data();
  const float*   gate_s     = gate_scales.data();
  const int8_t*  up_b       = up_int8.data();
  const float*   up_s       = up_scales.data();
  const int8_t*  down_b     = down_int8.data();
  const float*   down_s     = down_scales.data();

  rt.ensure_serial((size_t)E);
  std::atomic<int> first_err{-1};

  {
    py::gil_scoped_release rel;
    rt.rt->pool->parallel_for((int)E, [&](int e) {
      const int64_t g = eid[(size_t)e];
      if (g < 0 || g >= G) { first_err.store(e); return; }
      const ssize_t m = off[e + 1] - off[e];
      if (m <= 0) return;
      cg_runtime_t* srt = rt.serial_rts[(size_t)e]->rt;
      const size_t mi = (size_t)m * inter;

      std::vector<float>    c_gate(mi), c_up(mi);
      std::vector<uint16_t> act(mi);

      auto d = cpu_gemm::make_desc();
      d.alpha = 1.0f; d.beta = 0.0f;
      d.dtype_a = CG_BF16; d.dtype_b = CG_INT8; d.dtype_c = CG_F32;

      // gate = x @ Wgate^T   [m, inter]
      d.m = (size_t)m; d.k = hidden; d.n = inter;
      d.a = x_base + off[e] * (ssize_t)hidden; d.lda = hidden;
      d.b = gate_b + (size_t)g * inter * hidden; d.ldb = hidden;
      d.b_scales = gate_s + (size_t)g * inter;
      d.c = c_gate.data(); d.ldc = inter;
      if (cg_gemm(srt, &d) != CG_OK) { first_err.store(e); return; }

      // up = x @ Wup^T   [m, inter]
      d.b = up_b + (size_t)g * inter * hidden;
      d.b_scales = up_s + (size_t)g * inter;
      d.c = c_up.data();
      if (cg_gemm(srt, &d) != CG_OK) { first_err.store(e); return; }

      // act = silu(gate) * up, rounded to bf16 (matches the GPU path's round-trip)
      for (size_t i = 0; i < mi; ++i) {
        float gg = c_gate[i];
        float a = (gg / (1.0f + std::exp(-gg))) * c_up[i];
        act[i] = fp32_to_bf16_rne(a);
      }

      // down = act @ Wdown^T   [m, hidden]
      d.m = (size_t)m; d.k = inter; d.n = hidden;
      d.a = act.data(); d.lda = inter;
      d.b = down_b + (size_t)g * hidden * inter; d.ldb = inter;
      d.b_scales = down_s + (size_t)g * hidden;
      d.c = out_base + off[e] * (ssize_t)hidden; d.ldc = hidden;
      if (cg_gemm(srt, &d) != CG_OK) { first_err.store(e); return; }
    });
  }

  if (first_err.load() >= 0)
    throw std::runtime_error(
        "moe_expert_forward_batch: failed for expert item " +
        std::to_string(first_err.load()));
}

// ---------------------------------------------------------------------------
// Capturable single-token decode host node.
//
// `moe_decode_host` has the cudaHostFn_t signature (void(void*)) and is the
// function pointer cudaLaunchHostFunc records as a CUDA-graph HOST NODE during
// stream capture. It runs the whole MoE for the decode batch (T tokens, all on
// CPU) reading/writing FIXED pinned buffers whose addresses are bound once into
// a DecodeArgs struct (so graph replay re-reads the same memory). It makes NO
// CUDA calls and touches NO Python state — only cg_gemm over raw pointers — so
// it is legal both as a captured graph node and when invoked on a CUDA driver
// thread at replay.
//
// Math mirrors moe_expert_forward_batch_py exactly (gate/up -> SiLU*up -> bf16
// round -> down, route-weighted FP32 accumulate -> bf16). The caller supplies
// expert_ids clamped >= 0 and route_w masked (invalid slots = 0); routed
// scaling is applied by the caller after the H2D copy, matching the eager path.
struct DecodeArgs {
  RuntimeHandle* rth = nullptr;      // shared pool + per-slot serial runtimes
  const uint16_t* x = nullptr;       // [T, H] bf16 bits
  const int64_t*  eid = nullptr;     // [T, K] int64
  const float*    rw = nullptr;      // [T, K] fp32 (masked)
  uint16_t*       out = nullptr;     // [T, H] bf16 bits
  const int8_t*   gate_b = nullptr;  const float* gate_s = nullptr;  // [G,I,H],[G,I]
  const int8_t*   up_b = nullptr;    const float* up_s = nullptr;    // [G,I,H],[G,I]
  const int8_t*   down_b = nullptr;  const float* down_s = nullptr;  // [G,H,I],[G,H]
  int T = 0, K = 0, G = 0, H = 0, I = 0;
  // Pre-allocated per-slot scratch (one row per top-k slot, so the K experts
  // run concurrently with no write aliasing). single-replay-at-a-time, no race.
  float* c_gate = nullptr;   // [K, I]
  float* c_up = nullptr;     // [K, I]
  uint16_t* act = nullptr;   // [K, I] bf16 bits
  float* c_down = nullptr;   // [K, H]  per-slot expert output (pre-reduce)
  float* acc = nullptr;      // [H] fp32 accumulator
};

// CUDA-graph host node. Mirrors moe_expert_forward_batch_py's parallelism:
// ONE parallel_for over the K routed slots (each expert on its own single-
// thread serial runtime, computing gate/up/silu/down inline), then a serial
// route-weighted reduce. This is one pool wake per token per layer — NOT a
// full-pool fan-out per tiny m=1 GEMM, which is catastrophically slow.
void moe_decode_host(void* user) {
  DecodeArgs* a = static_cast<DecodeArgs*>(user);
  const int H = a->H, I = a->I, K = a->K, G = a->G, T = a->T;
  for (int t = 0; t < T; ++t) {
    const uint16_t* xt   = a->x   + (size_t)t * H;
    const int64_t*  eidt = a->eid + (size_t)t * K;
    const float*    rwt  = a->rw  + (size_t)t * K;

    // Each routed slot computes its expert on a dedicated 1-thread runtime;
    // distinct output rows (c_down + j*H) → no cross-slot races.
    a->rth->rt->pool->parallel_for(K, [&](int j) {
      const float w = rwt[j];
      if (w == 0.0f) return;             // masked / padding slot
      const int64_t e = eidt[j];
      if (e < 0 || e >= G) return;
      cg_runtime_t* srt = a->rth->serial_rts[(size_t)j]->rt;

      float* cg = a->c_gate + (size_t)j * I;
      float* cu = a->c_up   + (size_t)j * I;
      uint16_t* ac = a->act + (size_t)j * I;
      float* cd = a->c_down + (size_t)j * H;

      auto d = cpu_gemm::make_desc();
      d.alpha = 1.0f; d.beta = 0.0f;
      d.dtype_a = CG_BF16; d.dtype_b = CG_INT8; d.dtype_c = CG_F32;

      // gate = x @ Wgate^T   [1, I]
      d.m = 1; d.k = (size_t)H; d.n = (size_t)I;
      d.a = xt; d.lda = (size_t)H;
      d.b = a->gate_b + (size_t)e * I * H; d.ldb = (size_t)H;
      d.b_scales = a->gate_s + (size_t)e * I;
      d.c = cg; d.ldc = (size_t)I;
      if (cg_gemm(srt, &d) != CG_OK) return;

      // up = x @ Wup^T   [1, I]
      d.b = a->up_b + (size_t)e * I * H;
      d.b_scales = a->up_s + (size_t)e * I;
      d.c = cu;
      if (cg_gemm(srt, &d) != CG_OK) return;

      // act = silu(gate) * up, rounded to bf16 (matches the GPU path round-trip)
      for (int i = 0; i < I; ++i) {
        const float gg = cg[i];
        const float v = (gg / (1.0f + std::exp(-gg))) * cu[i];
        ac[i] = fp32_to_bf16_rne(v);
      }

      // down = act @ Wdown^T   [1, H]
      d.m = 1; d.k = (size_t)I; d.n = (size_t)H;
      d.a = ac; d.lda = (size_t)I;
      d.b = a->down_b + (size_t)e * H * I; d.ldb = (size_t)I;
      d.b_scales = a->down_s + (size_t)e * H;
      d.c = cd; d.ldc = (size_t)H;
      if (cg_gemm(srt, &d) != CG_OK) return;
    });

    // Route-weighted reduce over the K slots → bf16 output.
    float* acc = a->acc;
    for (int h = 0; h < H; ++h) acc[h] = 0.0f;
    for (int j = 0; j < K; ++j) {
      const float w = rwt[j];
      if (w == 0.0f) continue;
      const int64_t e = eidt[j];
      if (e < 0 || e >= G) continue;
      const float* cd = a->c_down + (size_t)j * H;
      for (int h = 0; h < H; ++h) acc[h] += w * cd[h];
    }
    uint16_t* outt = a->out + (size_t)t * H;
    for (int h = 0; h < H; ++h) outt[h] = fp32_to_bf16_rne(acc[h]);
  }
}

// Address of the host-node callback, as an int, for ctypes cudaLaunchHostFunc.
std::intptr_t decode_host_fn_ptr_py() {
  return reinterpret_cast<std::intptr_t>(&moe_decode_host);
}

// Bind fixed buffer addresses + weight slab pointers + dims into a heap
// DecodeArgs that must outlive the captured graph (freed via free_decode_args).
std::intptr_t make_decode_args_py(
    RuntimeHandle& rt,
    std::uintptr_t x, std::uintptr_t eid, std::uintptr_t rw, std::uintptr_t out,
    std::uintptr_t gate_b, std::uintptr_t gate_s,
    std::uintptr_t up_b, std::uintptr_t up_s,
    std::uintptr_t down_b, std::uintptr_t down_s,
    int T, int K, int G, int H, int I) {
  if (T <= 0 || K <= 0 || G <= 0 || H <= 0 || I <= 0)
    throw std::invalid_argument("make_decode_args: dims must be positive");
  // One single-thread runtime per routed slot, so the K experts run concurrently
  // under the host node's parallel_for (allocated outside capture).
  rt.ensure_serial((size_t)K);
  auto* a = new DecodeArgs();
  a->rth = &rt;
  a->x = reinterpret_cast<const uint16_t*>(x);
  a->eid = reinterpret_cast<const int64_t*>(eid);
  a->rw = reinterpret_cast<const float*>(rw);
  a->out = reinterpret_cast<uint16_t*>(out);
  a->gate_b = reinterpret_cast<const int8_t*>(gate_b);
  a->gate_s = reinterpret_cast<const float*>(gate_s);
  a->up_b = reinterpret_cast<const int8_t*>(up_b);
  a->up_s = reinterpret_cast<const float*>(up_s);
  a->down_b = reinterpret_cast<const int8_t*>(down_b);
  a->down_s = reinterpret_cast<const float*>(down_s);
  a->T = T; a->K = K; a->G = G; a->H = H; a->I = I;
  a->c_gate = new float[(size_t)K * I];
  a->c_up = new float[(size_t)K * I];
  a->act = new uint16_t[(size_t)K * I];
  a->c_down = new float[(size_t)K * H];
  a->acc = new float[(size_t)H];
  return reinterpret_cast<std::intptr_t>(a);
}

void free_decode_args_py(std::intptr_t p) {
  auto* a = reinterpret_cast<DecodeArgs*>(p);
  if (!a) return;
  delete[] a->c_gate;
  delete[] a->c_up;
  delete[] a->act;
  delete[] a->c_down;
  delete[] a->acc;
  delete a;
}

}  // namespace

PYBIND11_MODULE(_cpu_C, m) {
  m.doc() = "asym_gemm._cpu_C — pybind11 wrapper for cpu_gemm (AMX INT8)";

  py::class_<RuntimeHandle>(m, "Runtime")
      .def(py::init<int>(), py::arg("n_threads") = 0)
      .def_property_readonly("threads", &RuntimeHandle::threads);

  m.def("caps", &caps_dict, "Host CPU capabilities (AMX, AVX-512, ...)");

  m.def("pack_b_int8_amx_size", &pack_b_int8_amx_size_py,
        py::arg("n"), py::arg("k"));
  m.def("pack_b_int8_amx", &pack_b_int8_amx_py,
        py::arg("b_int8"), py::arg("b_scales"));

  m.def("gemm_bf16_int8", &gemm_bf16_int8_py,
        py::arg("rt"), py::arg("a_bf16"), py::arg("b_int8"),
        py::arg("b_scales"), py::arg("c_fp32"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  m.def("gemm_bf16_int8_packed", &gemm_bf16_int8_packed_py,
        py::arg("rt"), py::arg("a_bf16"), py::arg("b_packed"),
        py::arg("c_fp32"), py::arg("n"), py::arg("k"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  m.def("moe_expert_forward_batch", &moe_expert_forward_batch_py,
        py::arg("rt"), py::arg("x_cat"), py::arg("m_offsets"),
        py::arg("expert_ids"),
        py::arg("gate_int8"), py::arg("gate_scales"),
        py::arg("up_int8"), py::arg("up_scales"),
        py::arg("down_int8"), py::arg("down_scales"),
        py::arg("out_cat"), py::arg("inter"), py::arg("hidden"));

  // Capturable single-token decode (CUDA-graph host-node path).
  m.def("decode_host_fn_ptr", &decode_host_fn_ptr_py,
        "Address (int) of the cudaLaunchHostFunc decode callback.");
  m.def("make_decode_args", &make_decode_args_py,
        py::arg("rt"), py::arg("x"), py::arg("eid"), py::arg("rw"),
        py::arg("out"),
        py::arg("gate_int8"), py::arg("gate_scales"),
        py::arg("up_int8"), py::arg("up_scales"),
        py::arg("down_int8"), py::arg("down_scales"),
        py::arg("T"), py::arg("K"), py::arg("G"), py::arg("H"), py::arg("I"));
  m.def("free_decode_args", &free_decode_args_py, py::arg("args"));
}
