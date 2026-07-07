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
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

// ---------------------------------------------------------------------------
// NUMA helpers (no libnuma dependency)
// ---------------------------------------------------------------------------

// CPUs of a NUMA node, parsed from sysfs ("48-95,144-191" style).
static std::vector<int> numa_node_cpus(int node) {
  std::vector<int> cpus;
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                  "/cpulist");
  std::string s;
  if (!f || !std::getline(f, s)) return cpus;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    std::string tok = s.substr(i, j == std::string::npos ? std::string::npos
                                                         : j - i);
    size_t dash = tok.find('-');
    try {
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(tok));
      } else {
        int a = std::stoi(tok.substr(0, dash));
        int b = std::stoi(tok.substr(dash + 1));
        for (int c = a; c <= b; ++c) cpus.push_back(c);
      }
    } catch (...) { return {}; }
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return cpus;
}

// Temporarily restrict the calling thread to `cpus`; spawned std::threads
// inherit the creator's affinity, which is how a WorkerPool gets bound to a
// node without any change to the (portable) pool itself.
struct ScopedAffinity {
  cpu_set_t old_{};
  bool active_ = false;
  explicit ScopedAffinity(const std::vector<int>& cpus) {
    if (cpus.empty()) return;
    if (sched_getaffinity(0, sizeof(old_), &old_) != 0) return;
    cpu_set_t st;
    CPU_ZERO(&st);
    for (int c : cpus) CPU_SET(c, &st);
    active_ = (sched_setaffinity(0, sizeof(st), &st) == 0);
  }
  ~ScopedAffinity() {
    if (active_) sched_setaffinity(0, sizeof(old_), &old_);
  }
};

struct RuntimeHandle {
  cg_runtime_t* rt = nullptr;      // primary pool (node A's in NUMA-TP mode)
  cg_runtime_t* rt_b = nullptr;    // node B's pool (NUMA-TP mode only)
  // One single-thread runtime per concurrent MoE work item. A 1-thread
  // runtime executes cg_gemm fully inline (no pool wake) and owns its own
  // scratch arena, so concurrent use from distinct pool workers is safe.
  std::vector<std::unique_ptr<RuntimeHandle>> serial_rts;

  explicit RuntimeHandle(int n_threads) {
    rt = cg_runtime_create(n_threads);
    if (!rt) throw std::runtime_error("cg_runtime_create returned NULL");
  }

  // NUMA tensor-parallel mode: two pools, each with its worker threads bound
  // to one node (threads inherit the creator's affinity at spawn). Expert
  // weights are split into node-local halves by the caller; each pool only
  // reads its own node's bytes, so the MoE gets both sockets' local memory
  // bandwidth instead of one node's (or half-remote interleaved) bandwidth.
  RuntimeHandle(int n_threads, int node_a, int node_b) {
    const int na = std::max(1, n_threads / 2);
    const int nb = std::max(1, n_threads - na);
    {
      ScopedAffinity aff(numa_node_cpus(node_a));
      rt = cg_runtime_create(na);
    }
    if (!rt) throw std::runtime_error("cg_runtime_create returned NULL");
    {
      ScopedAffinity aff(numa_node_cpus(node_b));
      rt_b = cg_runtime_create(nb);
    }
    if (!rt_b) throw std::runtime_error("cg_runtime_create (node B) returned NULL");
  }

  ~RuntimeHandle() {
    if (rt) cg_runtime_destroy(rt);
    if (rt_b) cg_runtime_destroy(rt_b);
  }
  int threads() const {
    return cg_runtime_threads(rt) + (rt_b ? cg_runtime_threads(rt_b) : 0);
  }
  bool numa_tp() const { return rt_b != nullptr; }

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

/*
 * Per-expert batched MoE compute over a concatenated [sum_m, hidden] layout.
 *
 * Work is split at (expert, N-tile) granularity in two pool phases so a small
 * expert count still engages the whole pool. At decode (8 experts, m_e = 1)
 * the old one-task-per-expert layout left every expert on a single thread,
 * bound by one core's memory bandwidth (~50 GB/s total); N-tiling the weight
 * reads across the pool reaches the socket's full bandwidth.
 *
 *   phase 1: tasks = (expert, tile of `inter`)  — gate/up GEMMs for the tile's
 *            output channels, fused SiLU*up, bf16 round into the shared act
 *            workspace (elementwise in the tiled dim, so no cross-tile deps).
 *   phase 2: tasks = (expert, tile of `hidden`) — down GEMM for the tile,
 *            reading the expert's full act rows (a K-dim read, needs phase 1
 *            complete — hence the two parallel_for barriers).
 *
 * Each task runs its GEMM on its own single-thread serial runtime (inline, no
 * nested pool wake). NO routing weights / reduce (the caller applies those).
 * The caller must already hold no GIL (pybind wrapper releases it; the CUDA
 * host node has none).
 */

// Lower bound of tile `i` of `t` even tiles over `n`, aligned down to 32
// (a multiple of the AMX 16-column tile). tile_lo(n, t, t) == n.
static inline size_t tile_lo(size_t n, int t, int i) {
  if (i >= t) return n;
  return (n * (size_t)i / (size_t)t) & ~(size_t)31;
}

// Tiles per expert for an N-dim of `n`: enough that E experts yield ~4 tasks
// per pool thread (diminishing returns beyond), capped so tiles stay >= 64
// columns (128 B rows — cache-line aligned, and wide enough for the AMX
// kernel to stay efficient).
static inline int pick_tiles(size_t n, int E, int threads) {
  const int max_t = (int)std::max<size_t>(1, n / 64);
  const long want = (4L * threads + E - 1) / E;
  return (int)std::min<long>(max_t, std::max(1L, want));
}

// One slab half's base pointers. In NUMA-TP mode experts [0, g_split) live in
// half A (node A local pages) and experts [g_split, G) in half B (node B),
// indexed by g - g_split. Without NUMA-TP, half A holds all G experts and
// half B is null.
struct SlabHalf {
  const int8_t* gate_b = nullptr; const float* gate_s = nullptr;
  const int8_t* up_b   = nullptr; const float* up_s   = nullptr;
  const int8_t* down_b = nullptr; const float* down_s = nullptr;
};

bool moe_compute_core(
    RuntimeHandle& rt,
    const uint16_t* x_base,      // [sum_m, hidden] bf16 bits
    const int64_t* off,          // [E+1]
    const int64_t* eid,          // [E] global expert id per item
    int E,
    const SlabHalf& ha, const SlabHalf& hb, int64_t g_split,
    int64_t G, float* out_base, size_t inter, size_t hidden) {
  if (E <= 0) return true;
  for (int e = 0; e < E; ++e)
    if (eid[e] < 0 || eid[e] >= G) return false;

  const bool tp = rt.numa_tp() && hb.gate_b != nullptr;

  // Partition work items by owning node (everything to pool A without TP).
  std::vector<int> items_a, items_b;
  items_a.reserve(E);
  for (int e = 0; e < E; ++e) {
    if (tp && eid[e] >= g_split) items_b.push_back(e);
    else items_a.push_back(e);
  }

  const int threads = rt.threads();
  const int t1 = pick_tiles(inter, E, threads);
  const int t2 = pick_tiles(hidden, E, threads);
  rt.ensure_serial((size_t)E * (size_t)std::max(t1, t2));

  // Shared act workspace [sum_m, inter] (bf16 bits), indexed by off[e]*inter.
  // One MoE layer runs at a time (graph nodes are stream-ordered; the eager
  // path is serialized by the caller), so a single growable buffer suffices;
  // the mutex makes an unexpected concurrent caller safe rather than corrupt.
  static std::mutex ws_mu;
  static std::vector<uint16_t> act_ws;
  std::lock_guard<std::mutex> ws_lk(ws_mu);
  const size_t act_elems = (size_t)off[E] * inter;
  if (act_ws.size() < act_elems) act_ws.resize(act_elems);
  uint16_t* act_base = act_ws.data();

  std::atomic<bool> ok{true};

  auto half_of = [&](int64_t g) -> const SlabHalf& {
    return (g < g_split || hb.gate_b == nullptr) ? ha : hb;
  };
  auto local_g = [&](int64_t g) -> size_t {
    return (size_t)((g < g_split || hb.gate_b == nullptr) ? g : g - g_split);
  };

  // ---- phase 1 body: gate/up tiles + fused SiLU*up -> act ----
  auto phase1 = [&](int e, int ti) {
    const int64_t g = eid[e];
    const ssize_t m = off[e + 1] - off[e];
    if (m <= 0) return;
    const size_t ts = tile_lo(inter, t1, ti);
    const size_t te = tile_lo(inter, t1, ti + 1);
    if (te <= ts) return;
    const size_t tn = te - ts;
    cg_runtime_t* srt = rt.serial_rts[(size_t)e * t1 + ti]->rt;
    const SlabHalf& hf = half_of(g);
    const size_t lg = local_g(g);

    std::vector<float> c_gate((size_t)m * tn), c_up((size_t)m * tn);

    auto d = cpu_gemm::make_desc();
    d.alpha = 1.0f; d.beta = 0.0f;
    d.dtype_a = CG_BF16; d.dtype_b = CG_INT8; d.dtype_c = CG_F32;

    // gate tile = x @ Wgate[ts:te]^T   [m, tn]
    d.m = (size_t)m; d.k = hidden; d.n = tn;
    d.a = x_base + off[e] * (ssize_t)hidden; d.lda = hidden;
    d.b = hf.gate_b + (lg * inter + ts) * hidden; d.ldb = hidden;
    d.b_scales = hf.gate_s + lg * inter + ts;
    d.c = c_gate.data(); d.ldc = tn;
    if (cg_gemm(srt, &d) != CG_OK) { ok.store(false); return; }

    // up tile = x @ Wup[ts:te]^T   [m, tn]
    d.b = hf.up_b + (lg * inter + ts) * hidden;
    d.b_scales = hf.up_s + lg * inter + ts;
    d.c = c_up.data();
    if (cg_gemm(srt, &d) != CG_OK) { ok.store(false); return; }

    // act tile = silu(gate) * up, rounded to bf16 (matches the GPU path)
    uint16_t* act_e = act_base + (size_t)off[e] * inter;
    for (ssize_t r = 0; r < m; ++r) {
      const float* pg = &c_gate[(size_t)r * tn];
      const float* pu = &c_up[(size_t)r * tn];
      uint16_t* pa = act_e + (size_t)r * inter + ts;
      for (size_t i = 0; i < tn; ++i) {
        const float gg = pg[i];
        pa[i] = fp32_to_bf16_rne((gg / (1.0f + std::exp(-gg))) * pu[i]);
      }
    }
  };

  // ---- phase 2 body: down tiles ----
  auto phase2 = [&](int e, int ti) {
    const int64_t g = eid[e];
    const ssize_t m = off[e + 1] - off[e];
    if (m <= 0) return;
    const size_t ts = tile_lo(hidden, t2, ti);
    const size_t te = tile_lo(hidden, t2, ti + 1);
    if (te <= ts) return;
    const size_t tn = te - ts;
    cg_runtime_t* srt = rt.serial_rts[(size_t)e * t2 + ti]->rt;
    const SlabHalf& hf = half_of(g);
    const size_t lg = local_g(g);

    auto d = cpu_gemm::make_desc();
    d.alpha = 1.0f; d.beta = 0.0f;
    d.dtype_a = CG_BF16; d.dtype_b = CG_INT8; d.dtype_c = CG_F32;

    // down tile = act @ Wdown[ts:te]^T   [m, tn]
    d.m = (size_t)m; d.k = inter; d.n = tn;
    d.a = act_base + (size_t)off[e] * inter; d.lda = inter;
    d.b = hf.down_b + (lg * hidden + ts) * inter; d.ldb = inter;
    d.b_scales = hf.down_s + lg * hidden + ts;
    d.c = out_base + off[e] * (ssize_t)hidden + ts; d.ldc = hidden;
    if (cg_gemm(srt, &d) != CG_OK) { ok.store(false); return; }
  };

  // Run one phase over both pools concurrently: pool B's job is published
  // asynchronously (its node-bound workers run it alone) while this thread
  // participates in pool A's job, then joins B. No helper std::thread — a
  // pthread_create+join pair per phase costs ~50-100us, which at decode
  // (m_e = 1, ~50us of actual GEMM work) dominated the whole phase; the
  // transient thread was also unbound, so its share of pool B's tasks read
  // node-B weights from the wrong socket.
  auto run_phase = [&](int tiles, auto&& body) {
    if (!tp || items_b.empty()) {
      const std::vector<int>& it = items_a.empty() ? items_b : items_a;
      rt.rt->pool->parallel_for((int)it.size() * tiles, [&](int task) {
        body(it[task / tiles], task % tiles);
      });
      return;
    }
    if (items_a.empty()) {
      rt.rt_b->pool->parallel_for((int)items_b.size() * tiles, [&](int task) {
        body(items_b[task / tiles], task % tiles);
      });
      return;
    }
    // Named object: submit() stores a pointer to it until wait_done().
    std::function<void(int)> body_b = [&](int task) {
      body(items_b[task / tiles], task % tiles);
    };
    rt.rt_b->pool->submit((int)items_b.size() * tiles, body_b);
    rt.rt->pool->parallel_for((int)items_a.size() * tiles, [&](int task) {
      body(items_a[task / tiles], task % tiles);
    });
    rt.rt_b->pool->wait_done();
  };

  run_phase(t1, phase1);
  if (!ok.load()) return false;
  run_phase(t2, phase2);
  return ok.load();
}

/*
 * One MoE layer's whole CPU bucket in a single worker-pool job (pybind entry
 * for the eager unified forward). Thin wrapper over moe_compute_core.
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
    // Optional node-B half (NUMA-TP): experts [g_split, G) with local index
    // g - g_split. Pass size-0 arrays when not split.
    py::array_t<int8_t,   py::array::c_style> gate_int8_b,
    py::array_t<float,    py::array::c_style> gate_scales_b,
    py::array_t<int8_t,   py::array::c_style> up_int8_b,
    py::array_t<float,    py::array::c_style> up_scales_b,
    py::array_t<int8_t,   py::array::c_style> down_int8_b,
    py::array_t<float,    py::array::c_style> down_scales_b,
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

  const int64_t g_split     = gate_int8.shape(0);
  const bool has_b          = gate_int8_b.size() > 0;
  const int64_t G           = g_split + (has_b ? gate_int8_b.shape(0) : 0);
  const int64_t* off        = m_offsets.data();
  const int64_t* eid        = expert_ids.data();
  const uint16_t* x_base    = x_cat.data();
  float*          out_base  = out_cat.mutable_data();

  SlabHalf ha{gate_int8.data(), gate_scales.data(),
              up_int8.data(),   up_scales.data(),
              down_int8.data(), down_scales.data()};
  SlabHalf hb;
  if (has_b) {
    hb = SlabHalf{gate_int8_b.data(), gate_scales_b.data(),
                  up_int8_b.data(),   up_scales_b.data(),
                  down_int8_b.data(), down_scales_b.data()};
  }

  bool ok;
  {
    py::gil_scoped_release rel;
    ok = moe_compute_core(rt, x_base, off, eid, (int)E,
                          ha, hb, g_split, G, out_base, inter, hidden);
  }
  if (!ok)
    throw std::runtime_error("moe_expert_forward_batch: expert compute failed");
}

// ---------------------------------------------------------------------------
// Capturable decode host node (any batch size T, all-CPU).
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
// For T>1 it groups the (token, slot) routing into a per-expert concatenated
// layout and calls moe_compute_core — exactly the eager CPU bucket's per-expert
// batching, so larger batches get cross-expert parallelism (more active experts
// → more pool threads engaged) instead of a slow per-token loop. Math mirrors
// the eager path (gate/up -> SiLU*up -> bf16 round -> down, route-weighted FP32
// accumulate -> bf16). The caller supplies expert_ids clamped >= 0 and route_w
// masked (invalid slots = 0); routed scaling is applied by the caller after the
// H2D copy.
//
// NOTE: this path is all-CPU (no GPU expert offload). For small/moderate decode
// batches that matches the eager unified path; at very large batch the eager
// path offloads big experts to the GPU, so capture should be limited to a
// moderate batch via --cuda-graph-max-bs and larger batches left to eager.
struct DecodeArgs {
  RuntimeHandle* rth = nullptr;      // shared pool(s) + serial runtimes
  const uint16_t* x = nullptr;       // [T, H] bf16 bits
  const int64_t*  eid = nullptr;     // [T, K] int64 (clamped >= 0)
  const float*    rw = nullptr;      // [T, K] fp32 (masked: invalid = 0)
  uint16_t*       out = nullptr;     // [T, H] bf16 bits
  SlabHalf ha;                       // experts [0, g_split)
  SlabHalf hb;                       // experts [g_split, G) (NUMA-TP) or null
  int64_t g_split = 0;
  int T = 0, K = 0, G = 0, H = 0, I = 0;
};

// Scratch for the host node's grouping / concat / reduce. ONE instance shared by
// every layer's host node (layers run sequentially in the captured graph, so no
// overlap), grown to the largest captured T outside capture in make_decode_args.
// Non-pinned (CPU-only), so sharing avoids an O(num_layers) memory blow-up.
struct DecodeWorkspace {
  int cap_T = 0, K = 0, G = 0, H = 0, I = 0;
  std::vector<int64_t>  cnt;         // [G]   tokens routed to each expert
  std::vector<int64_t>  m_offsets;   // [G+1] concat offsets for active experts
  std::vector<int64_t>  expert_ids;  // [G]   active expert ids
  std::vector<int64_t>  wcursor;     // [G]   per-expert write head
  std::vector<uint16_t> x_cat;       // [cap_T*K, H]  gathered activations
  std::vector<int>      cat_token;   // [cap_T*K]     source token per concat row
  std::vector<float>    cat_w;       // [cap_T*K]     route weight per concat row
  std::vector<float>    out_scratch; // [cap_T*K, H]  per-expert output (pre-reduce)
  std::vector<float>    acc;         // [cap_T, H]    fp32 accumulator

  void ensure(int T, int K_, int G_, int H_, int I_) {
    K = K_; G = G_; H = H_; I = I_;
    if (T <= cap_T && !x_cat.empty()) return;
    cap_T = std::max(cap_T, T);
    const size_t sm = (size_t)cap_T * K;
    cnt.assign((size_t)G, 0);
    m_offsets.assign((size_t)G + 1, 0);
    expert_ids.assign((size_t)G, 0);
    wcursor.assign((size_t)G, 0);
    x_cat.assign(sm * (size_t)H, 0);
    cat_token.assign(sm, 0);
    cat_w.assign(sm, 0.0f);
    out_scratch.assign(sm * (size_t)H, 0.0f);
    acc.assign((size_t)cap_T * H, 0.0f);
  }
};
DecodeWorkspace g_ws;

void moe_decode_host(void* user) {
  DecodeArgs* a = static_cast<DecodeArgs*>(user);
  const int T = a->T, K = a->K, G = a->G, H = a->H, I = a->I;
  DecodeWorkspace& w = g_ws;

  // 1. Count tokens routed to each expert (skip masked / invalid slots).
  std::fill_n(w.cnt.data(), G, (int64_t)0);
  for (int t = 0; t < T; ++t) {
    const int64_t* eidt = a->eid + (size_t)t * K;
    const float*   rwt  = a->rw  + (size_t)t * K;
    for (int j = 0; j < K; ++j) {
      if (rwt[j] == 0.0f) continue;
      const int64_t e = eidt[j];
      if (e < 0 || e >= G) continue;
      w.cnt[e]++;
    }
  }
  // 2. Concat offsets for the active experts; wcursor[g] = running write head.
  int E = 0; int64_t running = 0;
  for (int g = 0; g < G; ++g) {
    if (w.cnt[g] == 0) continue;
    w.expert_ids[E] = g;
    w.m_offsets[E] = running;
    w.wcursor[g] = running;
    running += w.cnt[g];
    ++E;
  }
  w.m_offsets[E] = running;
  const int64_t sum_m = running;

  // 3. Scatter each (token, slot) into its expert's block; gather the x row.
  for (int t = 0; t < T; ++t) {
    const int64_t*  eidt = a->eid + (size_t)t * K;
    const float*    rwt  = a->rw  + (size_t)t * K;
    const uint16_t* xt   = a->x   + (size_t)t * H;
    for (int j = 0; j < K; ++j) {
      const float ww = rwt[j];
      if (ww == 0.0f) continue;
      const int64_t e = eidt[j];
      if (e < 0 || e >= G) continue;
      const int64_t p = w.wcursor[e]++;
      w.cat_token[p] = t;
      w.cat_w[p] = ww;
      std::memcpy(&w.x_cat[(size_t)p * H], xt, (size_t)H * sizeof(uint16_t));
    }
  }

  // 4. Per-expert batched compute (all on CPU) into the concat output.
  if (sum_m > 0) {
    moe_compute_core(*a->rth, w.x_cat.data(), w.m_offsets.data(),
                     w.expert_ids.data(), E,
                     a->ha, a->hb, a->g_split,
                     (int64_t)G, w.out_scratch.data(), (size_t)I, (size_t)H);
  }

  // 5. Route-weighted reduce over concat rows → per-token bf16 output.
  std::fill_n(w.acc.data(), (size_t)T * H, 0.0f);
  for (int64_t p = 0; p < sum_m; ++p) {
    const int t = w.cat_token[p];
    const float ww = w.cat_w[p];
    const float* s = &w.out_scratch[(size_t)p * H];
    float* d = &w.acc[(size_t)t * H];
    for (int h = 0; h < H; ++h) d[h] += ww * s[h];
  }
  for (int t = 0; t < T; ++t) {
    uint16_t* o = a->out + (size_t)t * H;
    const float* s = &w.acc[(size_t)t * H];
    for (int h = 0; h < H; ++h) o[h] = fp32_to_bf16_rne(s[h]);
  }
}

// Address of the host-node callback, as an int, for ctypes cudaLaunchHostFunc.
std::intptr_t decode_host_fn_ptr_py() {
  return reinterpret_cast<std::intptr_t>(&moe_decode_host);
}

// Bind fixed buffer addresses + weight slab pointers + dims into a heap
// DecodeArgs that must outlive the captured graph (freed via free_decode_args).
// Also (outside capture) grows the shared workspace and pre-allocates enough
// serial runtimes for the max active experts, so replay never allocates.
std::intptr_t make_decode_args_py(
    RuntimeHandle& rt,
    std::uintptr_t x, std::uintptr_t eid, std::uintptr_t rw, std::uintptr_t out,
    std::uintptr_t gate_b, std::uintptr_t gate_s,
    std::uintptr_t up_b, std::uintptr_t up_s,
    std::uintptr_t down_b, std::uintptr_t down_s,
    std::uintptr_t gate_b2, std::uintptr_t gate_s2,
    std::uintptr_t up_b2, std::uintptr_t up_s2,
    std::uintptr_t down_b2, std::uintptr_t down_s2,
    std::int64_t g_split,
    int T, int K, int G, int H, int I) {
  if (T <= 0 || K <= 0 || G <= 0 || H <= 0 || I <= 0)
    throw std::invalid_argument("make_decode_args: dims must be positive");
  g_ws.ensure(T, K, G, H, I);
  // At most min(G, T*K) experts are active in one decode step.
  rt.ensure_serial((size_t)std::min<int64_t>(G, (int64_t)T * K));
  auto* a = new DecodeArgs();
  a->rth = &rt;
  a->x = reinterpret_cast<const uint16_t*>(x);
  a->eid = reinterpret_cast<const int64_t*>(eid);
  a->rw = reinterpret_cast<const float*>(rw);
  a->out = reinterpret_cast<uint16_t*>(out);
  a->ha = SlabHalf{reinterpret_cast<const int8_t*>(gate_b),
                   reinterpret_cast<const float*>(gate_s),
                   reinterpret_cast<const int8_t*>(up_b),
                   reinterpret_cast<const float*>(up_s),
                   reinterpret_cast<const int8_t*>(down_b),
                   reinterpret_cast<const float*>(down_s)};
  a->hb = SlabHalf{reinterpret_cast<const int8_t*>(gate_b2),
                   reinterpret_cast<const float*>(gate_s2),
                   reinterpret_cast<const int8_t*>(up_b2),
                   reinterpret_cast<const float*>(up_s2),
                   reinterpret_cast<const int8_t*>(down_b2),
                   reinterpret_cast<const float*>(down_s2)};
  a->g_split = g_split ? g_split : (std::int64_t)G;
  a->T = T; a->K = K; a->G = G; a->H = H; a->I = I;
  return reinterpret_cast<std::intptr_t>(a);
}

void free_decode_args_py(std::intptr_t p) {
  auto* a = reinterpret_cast<DecodeArgs*>(p);
  if (!a) return;
  delete a;
}

}  // namespace

PYBIND11_MODULE(_cpu_C, m) {
  m.doc() = "asym_gemm._cpu_C — pybind11 wrapper for cpu_gemm (AMX INT8)";

  py::class_<RuntimeHandle>(m, "Runtime")
      .def(py::init<int>(), py::arg("n_threads") = 0)
      .def(py::init<int, int, int>(), py::arg("n_threads"),
           py::arg("node_a"), py::arg("node_b"),
           "NUMA-TP: two pools with worker threads bound to node_a / node_b")
      .def_property_readonly("numa_tp", &RuntimeHandle::numa_tp)
      .def_property_readonly("threads", &RuntimeHandle::threads);

  m.def("caps", &caps_dict, "Host CPU capabilities (AMX, AVX-512, ...)");

  m.def("moe_expert_forward_batch", &moe_expert_forward_batch_py,
        py::arg("rt"), py::arg("x_cat"), py::arg("m_offsets"),
        py::arg("expert_ids"),
        py::arg("gate_int8"), py::arg("gate_scales"),
        py::arg("up_int8"), py::arg("up_scales"),
        py::arg("down_int8"), py::arg("down_scales"),
        py::arg("gate_int8_b"), py::arg("gate_scales_b"),
        py::arg("up_int8_b"), py::arg("up_scales_b"),
        py::arg("down_int8_b"), py::arg("down_scales_b"),
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
        py::arg("gate_int8_b"), py::arg("gate_scales_b"),
        py::arg("up_int8_b"), py::arg("up_scales_b"),
        py::arg("down_int8_b"), py::arg("down_scales_b"),
        py::arg("g_split"),
        py::arg("T"), py::arg("K"), py::arg("G"), py::arg("H"), py::arg("I"));
  m.def("free_decode_args", &free_decode_args_py, py::arg("args"));
}
