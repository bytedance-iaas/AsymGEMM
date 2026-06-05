# Unified Kernel — CPU GEMM + AsymGEMM, **INT8 on both sides**, per-expert MoE dispatch

**Status:** draft v0.2 (pre-implementation).
**Change from v0.1:** the unified kernel now uses **INT8 end-to-end on both
the CPU and the GPU**. The earlier "dual-format pinned storage (INT8 + FP8)"
design is dropped. A new `sm90_int8_asym_gemm_1d1d` GPU kernel — a port of
the existing `sm90_fp8_asym_gemm_1d1d.cuh` to `s8.s8.s32` WGMMA — becomes
the GPU branch.

**Scope:** one MoE-GEMM façade routing per-expert work to either
`cpu_gemm` (`/workspace/cpu_gemm/`, AMX INT8) or a new
`sm90_int8_asym_gemm_1d1d` (added under
`/workspace/AsymGEMM_Main/AsymGEMM/asym_gemm/include/asym_gemm/impls/`),
with the same INT8 weights pinned once in host memory.

---

## 1. Goals & non-goals

### Goals

1. Single Python/C++ entry point `unified_moe_gemm(...)` for an MoE layer.
2. **One INT8 representation, one pinned-host weight copy.** No dtype
   conversion at the dispatch seam.
3. **Per-expert dispatch** by routed token count `m_e`:
   - `m_e ≤ M_CPU`: AMX INT8 path (cpu_gemm), in place on host.
   - `m_e >  M_CPU`: SM90 INT8 WGMMA path (new AsymGEMM kernel).
4. **Latency-honest threshold.** `M_CPU` is auto-tuned per (N, K, host/GPU
   pair). The dispatcher never picks the slower backend.
5. Follow ktransformers' MoE control flow (per-expert gather → GEMM →
   SwiGLU → scatter) so callers see no new semantics.
6. **CUDA-Graph friendly fast path.** The GPU branch must be replayable;
   the CPU branch may not be (acceptable for short-context decode where
   graphs are usually disabled anyway).

### Non-goals (v1)

- FP4 / MXFP4 paths.
- FP8 paths. We deliberately step off the FP8 ladder so the two backends
  share quantization arithmetic. FP8 can be added later as a parallel
  family (it does not block INT8).
- BF16 asym GEMM.
- Multi-GPU / NCCL routing.

---

## 2. The two source kernels at a glance

| Aspect             | `cpu_gemm` AMX INT8 (exists)                              | `sm90_int8_asym_gemm_1d1d` (NEW; port of FP8 1D1D)               |
|--------------------|------------------------------------------------------------|-------------------------------------------------------------------|
| File               | `cpu_gemm/src/kernels/amx/int8_gemm.cpp`                   | `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh` |
| Data path          | host RAM → AMX tiles (`_tile_dpbssd`, INT32 acc)           | HBM → TMA → smem → WGMMA `s8.s8.s32`                              |
| A dtype            | BF16 in → INT8 after per-row dynamic quant (`amax/127`)    | INT8 (per-token FP32 scale `SFA[BLOCK_M]`)                        |
| B dtype            | INT8 + per-channel FP32 scale (`b_scales[N]`)              | INT8 + per-channel FP32 scale (`SFB[BLOCK_N]`)                    |
| Accumulator        | INT32                                                      | INT32                                                              |
| C dtype            | FP32 (= `sA · sB · int32_acc`)                             | FP32 (= `sA · sB · int32_acc`); BF16 downcast in writeback path  |
| Scale application  | dequant-after-INT32-GEMM, fused into output unpack         | scale-apply-after-WGMMA: cast INT32→FP32, then ×sA×sB, TMA store |
| Threading          | `cg_runtime` worker pool, N-blocked                        | TMA warp-group + math warp-group(s); JIT'd via NVRTC              |
| Pre-pack required  | yes (B in blocked-VNNI INT8)                               | no — TMA reads global tensors via tensor maps                     |
| Caller shape       | dense `[m,k] × [n,k]ᵀ → [m,n]`                             | dense or *grouped MoE*: `[M,K] × [G,N,K]ᵀ → [M,N]`                |

The two now share **identical math** at the accumulator level. CPU and GPU
backends are interchangeable for any single expert's compute; the
dispatcher's choice is a pure latency decision.

---

## 3. High-level architecture

```
                       ┌──────────────────────────────────┐
                       │  unified_moe_gemm(...)           │
                       └──────────────┬───────────────────┘
                                      │ per-MoE-layer
                                      ▼
                       ┌──────────────────────────────────┐
                       │ Router-result decoding           │
                       │  • expert_ids [T, k_top]         │
                       │  • routing_weights [T, k_top]    │
                       │  • per-expert m_e and token-list │
                       └──────────────┬───────────────────┘
                                      ▼
                       ┌──────────────────────────────────┐
                       │ Per-expert dispatch policy       │
                       │   m_e <= M_CPU  → CPU bucket     │
                       │   m_e >  M_CPU  → GPU bucket     │
                       └──────┬───────────────────┬───────┘
                              │                   │
                  ┌───────────▼─────────┐  ┌──────▼───────────────┐
                  │ CPU MoE driver      │  │ GPU MoE driver       │
                  │  • gather tokens    │  │  • gather tokens     │
                  │  • per-row quant    │  │  • per-token quant   │
                  │    A: bf16→int8     │  │    A: bf16→int8 +    │
                  │  • cpu_gemm AMX     │  │    sfa (CUDA prekern)│
                  │  • SwiGLU on CPU    │  │  • H2D activations + │
                  │  • scatter back     │  │    weight if cold    │
                  │                     │  │  • sm90_int8 asym    │
                  │                     │  │    grouped 1D1D      │
                  │                     │  │  • SwiGLU + 2nd GEMM │
                  │                     │  │  • D2H scatter back  │
                  └─────────┬───────────┘  └──────────┬───────────┘
                            └───────────┬─────────────┘
                                        ▼
                        ┌────────────────────────────────────┐
                        │ Weighted reduction (k_top experts) │
                        │  → BF16, on GPU                    │
                        └────────────────────────────────────┘
```

Both branches produce FP32, both use the **same** scale convention
(`sA = amax_row/127`, `sB = amax_col/127`), so per-expert numerical drift
is at the rounding-mode level only.

---

## 4. Weight residency: one pinned-host INT8 copy

```
struct ExpertWeights {              // gate_proj, up_proj, down_proj
  // -- pinned-host primary copy (single source of truth) --
  int8_t*  w_int8;                  // [G, N, K] symmetric per-channel
  float*   w_scales;                // [G, N]    FP32, "amax/127"

  // -- AMX-side pre-pack (also pinned; aliases the same data when possible) --
  void*    w_int8_packed_amx;       // blocked-VNNI layout for cg_gemm
  float*   w_scales_amx;            // same scales, replicated only if
                                    //   layout requires reordering

  // -- optional VRAM mirror for hot experts (LRU, persistent) --
  int8_t*  d_w_int8;                // identical bytes as w_int8 (no repack)
  float*   d_w_scales;
};
```

Notes:

- **One byte per weight element on host, plus a small per-channel scale
  vector.** Compared to v0.1's dual INT8+FP8 design, this halves the
  pinned-memory footprint per layer.
- The AMX layout (`int8_buffers.h::BufferBInt8`, blocked-VNNI) is **not**
  byte-identical to the row-major `[N, K]` the GPU TMA descriptor wants.
  Two options:
  - **(a, default)** Keep both layouts pinned: row-major `w_int8` for
    TMA H2D, blocked-VNNI `w_int8_packed_amx` for `cg_gemm`. Extra cost
    is one byte per weight element, still half of v0.1.
  - **(b, future)** Teach AMX INT8 to consume row-major B with a
    stride-aware packer on first touch (already flagged as future work
    in `cpu_gemm/analysis.md` §5.3). Drops the duplicate but adds
    per-call packing overhead. Defer.
- Quantization is one-shot at `prepare_weights()`: read BF16 master from
  checkpoint, compute per-output-channel `amax`, divide by 127, round
  to INT8, write both layouts to pinned memory.
- The optional VRAM mirror caches **the exact same INT8 bytes** — no
  conversion, just `cudaMemcpyAsync` of `w_int8` plus `w_scales`. LRU
  size defaults to `min(active_experts_per_layer, free_VRAM / per_expert_bytes)`.
- Allocation:
  `cudaHostAlloc(..., cudaHostAllocPortable | cudaHostAllocMapped)` so
  TMA can also DMA directly from pinned host via the mapped-pointer
  fallback in §7.3 if needed.

---

## 5. The unified INT8 numerical contract

Both backends compute the **same** function up to rounding:

```
quantize:  A_int8[i,j] = round(A_bf16[i,j] * 127 / amax_row_i)
           sA[i]       = amax_row_i / 127                       (FP32)
           B_int8[j,n] = round(B_bf16[j,n] * 127 / amax_col_n)  (offline)
           sB[n]       = amax_col_n / 127                       (FP32)

compute:   C_int32[i,n] = Σ_j A_int8[i,j] · B_int8[j,n]
           C_fp32[i,n]  = sA[i] · sB[n] · C_int32[i,n]
```

Per-row symmetric quant on A, per-channel symmetric quant on B, no
zero-points. This is exactly what `cpu_gemm`'s AMX INT8 path already
does (see `int8_buffers.h::BufferAInt8::from_bf16`, line 60). The new
GPU kernel does the same; it inherits the FP8 1D1D's scale-apply skeleton
(`sm90_fp8_asym_gemm_1d1d.cuh`, lines 371–435) and substitutes:

- `cutlass::float_e4m3_t` → `int8_t` everywhere a tile element is held;
- `FP8MMASelector` → `INT8MMASelector` (`wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8.s32`);
- the WGMMA accumulator type stays `s32`; before the scale apply we add
  one `__int2float_rn` (or vectorized equivalent) on each accumulator
  element;
- `SMEM_A_SIZE_PER_STAGE` and `SMEM_B_SIZE_PER_STAGE` formulas keep
  `* sizeof(int8_t)` (same byte width as FP8, so smem sizing tables stay
  identical);
- `BLOCK_K` defaults shift from FP8's typical 128 to INT8's WGMMA k=32
  multiples (e.g. 64 or 128 still work; sized in elements not bytes).

Activation-side quant runs:

- **CPU branch:** already inside `cg_gemm` (per-row, on the gathered
  expert input tensor).
- **GPU branch:** a small `per_token_int8_quant` CUDA kernel writes
  `(d_a_int8, d_sfa)` from `d_a_bf16`. This op exists in SGLang
  (`sgl_kernel.quant.per_token_quant_int8`) and is also trivially
  expressible with reduce-max + `__float2int_rn`. The unified runtime
  ships its own minimal version so we don't take SGLang as a build dep.

**Why this works numerically.** Both backends use INT32 accumulation
of the same INT8 operands, so the only differences are rounding modes
in the per-row `amax` reduction (CPU AVX-512 vs CUDA shuffle reduction).
These differ by ≤ 1 ULP in FP32, and the subsequent INT32→FP32 dequant
is exact for the magnitudes involved (INT32 fits in FP32 mantissa for
all realistic K). Parity test 9.1.2 confirms.

---

## 6. Dispatch policy

### 6.1 Per-expert M decides

Same as v0.1:

```
if m_e <= M_CPU(N, K):           use cpu_gemm
elif m_e <= M_GPU_HOT(N, K):     use AsymGEMM with cached d_w_int8 if present
else:                            use AsymGEMM, H2D the weight slice first
```

### 6.2 How `M_CPU` is chosen

The cross-over depends on:
- AMX INT8 throughput at the live `(N, K, m_e)` — already characterized
  in `cpu_gemm/analysis.md` (AMX > AVX2 from `M ≈ 32` onward; within
  AMX, the kernel scales near-linearly with M past ~64).
- SM90 WGMMA INT8 throughput at `(N, K, m_e)`. **Important:** Hopper's
  INT8 tensor-core peak is roughly **half** of FP8 peak on H100/H20
  (the data-sheet ratio is ~2:1 in favor of FP8). The cross-over will
  therefore sit *higher* than it would for the FP8 design — i.e., the
  CPU bucket can profitably absorb more work before GPU wins. This is
  good news for the CPU branch but means `M_CPU` is not directly
  transferable from any FP8 measurements you have.
- PCIe H2D bandwidth (activations always; weights only on cache miss).
- AsymGEMM kernel launch + (warm) JIT cache lookup, ~10–30 µs.

Implementation: `tune_dispatch(N, K)` runs both backends across
`M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}` once at `prepare_weights()`,
stores cross-over `M_CPU(N, K, weight_in_vram)` in a per-(N,K) LUT.

Default expectation (to be confirmed on H20 + Sapphire Rapids):
`M_CPU ≈ 16–32` weight-cached, `M_CPU ≈ 48–96` weight-cold.

### 6.3 GPU bucket batching

All experts with `m_e > M_CPU` join one grouped call to the new
`m_grouped_int8_asym_gemm_nt_contiguous` (or `_nt_masked`) entry point.
We add this entry point alongside the existing FP8/BF16/FP4 grouped
families in AsymGEMM (see §10.2).

### 6.4 Fallbacks

- **No AMX on host** → fall back to `cpu_gemm`'s AVX-512 INT8 path
  (port from upstream ktransformers's `avx_kernels.hpp`; not currently
  in cpu_gemm — see §10.2). LUT thresholds shift down accordingly.
- **No INT8 WGMMA support** (only relevant on pre-SM75 hardware; SM90
  has it) → degrade to all-CPU. Documented limitation.
- **GPU OOM on VRAM mirror** → disable cache, always H2D, raise `M_CPU`.
- **All `m_e == 0`** → no-op.

---

## 7. Data movement

### 7.1 Pinned arenas

Per-MoE-layer, allocated once:

```
pinned_arena {
  act_bf16          [max_tokens, hidden]    // CPU-bucket A input
  cpu_out_fp32      [max_tokens, hidden]    // CPU-bucket gate/up/down result
  stage_a_int8      [max_tokens, hidden]    // GPU-bucket A (after host quant)
  stage_sfa         [max_tokens]            // FP32, per-token scale
  stage_out_bf16    [max_tokens, hidden]    // GPU result, D2H landing
}
```

Note that compared to v0.1 we no longer keep an FP8 staging area.
`max_tokens = max_seq_len × top_k` (worst-case expansion).

### 7.2 Streams & sync

Two CUDA streams per layer (`stream_compute`, `stream_copy`).
`cudaStreamWaitEvent` enforces:

```
H2D(stage_a_int8 + stage_sfa)  ──┐
H2D(weight slice on cache miss)──┼──> launch GPU gemm
                                  │   on stream_compute
                                  └──> D2H(stage_out_bf16) after kernel
```

CPU bucket runs on the host worker pool concurrently with the GPU stream;
`join()` happens before the weighted reduction.

### 7.3 Zero-copy fallback

Same as v0.1: when an expert is past `M_CPU` *and* its weight is not in
the VRAM cache *and* `weight_bytes / pcie_bandwidth < kernel_time`, the
TMA descriptor is built against the mapped pinned `w_int8`. Default off;
useful only for spillover.

### 7.4 Weighted reduction

After both buckets finish:
- CPU bucket result in `cpu_out_fp32` (host) → `cudaMemcpyAsync` to a
  GPU scratch buffer, downcast to BF16 in the reduce kernel.
- GPU bucket result already in HBM as BF16.

A single CUDA kernel applies the routing weights and accumulates into
the layer's output tensor. Borrowed from SGLang's MoE plumbing
(rewritten in-tree to avoid the SGLang build dep).

---

## 8. Public API

Unchanged in shape from v0.1; documentation reflects the single dtype.

### 8.1 C++ (`unified_moe/include/unified_moe.h`)

```cpp
struct umoe_runtime;          // opaque — owns cg_runtime, JIT cache, LUTs

umoe_status_t umoe_runtime_create(umoe_runtime** out,
                                  int n_cpu_threads,
                                  int cuda_device);
void          umoe_runtime_destroy(umoe_runtime*);

// Owns the three projections of one MoE layer. Quantizes BF16 master to
// INT8 once, populates pinned + AMX-packed buffers, runs dispatch
// auto-tune, optionally seeds the VRAM expert cache.
struct umoe_layer;
umoe_status_t umoe_layer_load(umoe_runtime*,
                              const umoe_layer_desc_t*,   // shapes, expert count
                              const void* gate_bf16,      // [G,N_inter,K]
                              const void* up_bf16,        // [G,N_inter,K]
                              const void* down_bf16,      // [G,K,N_inter]
                              umoe_layer** out);
void          umoe_layer_unload(umoe_layer*);

// Hot path. All pointers are device pointers; outputs land in `y_bf16`.
umoe_status_t umoe_layer_forward(umoe_layer*,
                                 int n_tokens,
                                 int top_k,
                                 const int32_t* expert_ids,   // device
                                 const float*   route_w,      // device
                                 const void*    x_bf16,       // device
                                 void*          y_bf16,       // device
                                 cudaStream_t   stream);
```

### 8.2 Python (`unified_moe.Layer`)

```python
import unified_moe

layer = unified_moe.Layer.from_bf16(
    gate=gate_t, up=up_t, down=down_t,
    num_experts=G, top_k=k, hidden=H, inter=I,
    cpu_threads=192, cuda_device=0,
)

y = layer(x, expert_ids, routing_weights)
```

---

## 9. Correctness & performance verification

### 9.1 Correctness

1. **Per-backend parity vs FP32 reference.**
   For `(N, K, m_e, expert)` grid: assert
   `max(|backend - ref| / |ref|) < 3e-2` (INT8 dynamic-quant noise
   envelope; tighter than the v0.1 cross-precision target).
2. **CPU-vs-GPU parity (the new headline test).**
   Same expert, same inputs, same INT8 weights:
   `max(|cpu_out - gpu_out| / |cpu_out|) < 1e-3` per output element.
   Tighter than (1) because both backends agree on quant + INT32 math.
3. **Dispatch-invariance.** Force `M_CPU = 0` (all-GPU) and
   `M_CPU = ∞` (all-CPU). End-to-end logits differ only within (2).
4. **Mixed-bucket parity.** Half-CPU / half-GPU routing matches
   all-CPU baseline within (2).
5. **Pinned-memory aliasing.** Compute-sanitizer clean under concurrent
   AMX kernels + `cudaMemcpyAsync` from the same pinned region.

(2) is the new test made possible by unifying on INT8 — it pins down
backend equivalence and any regression in the GPU kernel becomes a
loud, structured failure.

### 9.2 Performance benches

1. **Per-backend M-sweep** (`M ∈ {1..256}`) for the model's `(N, K)`.
   Output is the LUT; plot to `bench/cross_over.png`.
2. **VRAM-cache hit vs miss** — pins down `M_GPU_HOT`.
3. **End-to-end MoE layer** vs:
   - all-CPU baseline (ktransformers-style),
   - all-GPU (`M_CPU = 0`, every expert on the new INT8 WGMMA kernel),
   - unified dispatch.
   Goal: unified ≥ best-of-either within 5% at batch sizes
   `{1, 8, 32, 128}`.

### 9.3 CI gating

Build matrix: `(SPR + H20)`, `(SPR no-CUDA)`, `(AMX-less + H20)`.
Tests 9.1.1–9.1.4 are blocking; perf bench is informational.

---

## 10. Repo integration & deliverables

### 10.1 Where the code lives

```
/workspace/
  cpu_gemm/                                  (existing)
    src/kernels/amx/int8_gemm.cpp            ← unchanged math
    NEW: include/cpu_gemm/cpu_gemm.h         + cg_pack_b_int8 entry (§10.2)
  AsymGEMM_Main/AsymGEMM/                    (existing)
    NEW: asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh
    NEW: csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp
    NEW: csrc/apis/gemm.hpp                  + m_grouped_int8_asym_gemm_nt_{contiguous,masked}
    NEW: asym_gemm/dispatch.py               + Python facade for INT8 grouped
  unified_moe/                               NEW top-level
    CMakeLists.txt                             links cpu_gemm + asym_gemm
    include/unified_moe.h                      §8.1 C ABI
    src/
      runtime.cpp                              umoe_runtime, JIT cache, LUT
      layer.cpp                                prepare_weights, dispatch loop
      cpu_bucket.cpp                           cg_gemm wrap + SwiGLU + scatter
      gpu_bucket.cpp                           sm90_int8 grouped + SwiGLU + scatter
      per_token_int8_quant.cu                  A-side activation quant kernel
      reduce.cu                                weighted sum + dtype cast
      autotune.cpp                             M-sweep, LUT persistence
    python/unified_moe/__init__.py             pybind11 facade
    tests/
    bench/
```

### 10.2 Required upstream changes

These are now larger than v0.1 because we have to write a new GPU kernel.

1. **`cpu_gemm` — offline B pre-pack** (small).
   Add a public `cg_pack_b_size` + `cg_pack_b_int8` so the weight is
   pre-packed once into pinned memory in `BufferBInt8` layout; the hot
   path skips B-packing. Same change as v0.1.

2. **`cpu_gemm` — (optional) AVX-512 INT8 fallback** (medium).
   Port `ktransformers/kt-kernel/operators/amx/la/avx_kernels.hpp` INT8
   path so hosts without AMX still hit an INT8 backend. Not blocking
   for SPR + H20 deployments but matters for portability.

3. **AsymGEMM — `sm90_int8_asym_gemm_1d1d.cuh`** (large, the headline
   new deliverable).
   Port `sm90_fp8_asym_gemm_1d1d.cuh` to INT8 WGMMA. Concretely:
   - Same TMA/warp-group structure (TMA loader + math warpgroup; B in a
     single smem slot; A staged across `kNumStages`).
   - Replace `cutlass::float_e4m3_t` with `int8_t` for the data tile
     types; smem sizing formulas already use `sizeof(elem)` and stay
     correct.
   - Replace `FP8MMASelector<BLOCK_N>` with a new
     `INT8MMASelector<BLOCK_N>` that picks
     `cute::SM90_64x{N}x32_S32S8S8_SS` (Hopper INT8 WGMMA atoms).
     `BLOCK_K % 32 == 0` static-asserted (was `% WGMMA::K`, identical
     mechanism — just K=32 instead of K=32 for FP8E4M3, so the constants
     happen to match; sanity-check via CUTLASS atoms).
   - Accumulator stays `int32_t` (was `float`); add `__int2float_rn`
     just before the scale-apply step. Lines 431–434 of the FP8 kernel
     become:
     ```cpp
     float val_0 = scale_a_0[local_idx] * scales_b[i].x * __int2float_rn(shifted_accum[i*4 + 0]);
     // ... and similarly for val_1..val_3
     ```
   - Output dtype: support both FP32 (default; matches FP8 path) and
     BF16 (cheap downcast in the epilogue; saves a separate cast kernel
     in the GPU bucket).
   - SFA/SFB layouts are byte-identical to the FP8 path — both are
     FP32 per-row / per-col, fetched via `tma_copy<BLOCK_M, 1, 0>` /
     `tma_copy<BLOCK_N, 1, 0>`.
   - Reuse `asymScheduler`, barrier counts, and the smem layout
     verbatim. The only structural difference from the FP8 file is the
     two type swaps + the one INT32→FP32 cast.

4. **AsymGEMM — JIT launcher**
   `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp` (port of the
   FP8 launcher in the same directory). One-to-one.

5. **AsymGEMM — Python/C++ facade**
   Add `m_grouped_int8_asym_gemm_nt_contiguous` and `_nt_masked` to
   `csrc/apis/gemm.hpp` and `asym_gemm/dispatch.py`, mirroring the FP8
   entry points. SM89 has no INT8 grouped MoE GEMM in this repo today;
   if added, we'd route SM89 separately. For v1, SM89 falls back to
   all-CPU.

6. **AsymGEMM — test**
   `tests/test_h20_int8_asym.py` mirroring `test_h20_fp8.py`.

### 10.3 Build & packaging

Same as v0.1: top-level `unified_moe/CMakeLists.txt` with
`UMOE_WITH_CUDA` (default on if `nvcc` found) and `UMOE_WITH_AMX`.
Pybind11 wheel via `unified_moe/python/setup.py`. Depends on a built
copy of `cpu_gemm` and a wheel of `asym_gemm` (rebuilt from the
INT8-augmented tree).

---

## 11. Implementation milestones

| #  | Milestone                                                                 | Exit criterion                                                                       |
|----|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 0a | Land `cg_pack_b_int8` in `cpu_gemm`                                       | Pre-packed B used in pinned memory; existing CPU INT8 tests still pass                |
| 0b | **Author `sm90_int8_asym_gemm_1d1d.cuh` + JIT launcher**                  | Standalone unit test against numpy FP32 ref passes within 3e-2; matches CPU INT8 path within 1e-3 |
| 0c | Add `m_grouped_int8_asym_gemm_nt_contiguous` facade to AsymGEMM           | Grouped call works on synthetic G-expert workloads                                    |
| 1  | `umoe_runtime_create` + pinned INT8 alloc + AMX repack                    | `prepare_weights` produces both layouts; round-trip dequant matches BF16 master ≤ 3e-2|
| 2  | CPU bucket end-to-end                                                     | Matches numpy ref on a 32-expert toy MoE                                              |
| 3  | GPU bucket end-to-end (cache hit and cache miss)                          | Matches CPU bucket within 1e-3; H2D overlap visible in Nsight                         |
| 4  | Per-expert dispatch + weighted reduction                                  | Mixed-bucket parity test (§9.1.4) passes                                              |
| 5  | Auto-tune + LUT persistence                                               | LUT loaded on second run; cross-over plot monotonic                                   |
| 6  | Python facade + smoke test on a real MoE checkpoint                       | `unified_moe.Layer(...)` runs under an SGLang-style harness                           |
| 7  | Perf bench vs all-CPU and all-GPU INT8 baselines                          | Unified ≥ best-of-either by ≥0% at every batch size in `{1, 8, 32, 128}`              |

Milestone 0b is the new critical-path item. It's a port of a known-good
kernel (the FP8 1D1D file is 495 lines, structurally complete) with
type swaps and one cast, so it's bounded — but it does need its own
verification before milestone 3 can start.

---

## 12. Risks & open questions

1. **SM90 INT8 WGMMA peak throughput.** Hopper INT8 tensor-core peak is
   roughly **half** of its FP8 peak (data-sheet ratio). On H20 the
   absolute number is further throttled. Concrete number to confirm
   during milestone 0b benchmarking. *Implication:* the GPU path is
   slower per call than the FP8 GPU path would have been; this nudges
   `M_CPU` upward (more work stays on CPU). For decode-dominated
   workloads this is actually fine — the CPU was always going to handle
   the small-M regime, and INT8 WGMMA at large M still trounces AMX.
2. **INT8 quality vs FP8.** Symmetric per-channel INT8 with dynamic
   per-token activation quant is the same recipe used by SmoothQuant,
   GPTQ-INT8, etc. — well-characterized but lossier than block-scaled
   FP8 on some checkpoints. Mitigation: ship a `force_backend = bf16`
   debug switch (CPU bucket only, slow) for offline accuracy bisection.
   We do not commit to publishing eval numbers in v1.
3. **Threshold flapping** at `m_e ≈ M_CPU`. Hysteresis (`±δ`) as in v0.1.
4. **NVRTC cold-start** for the new INT8 kernel. Same mitigation as
   v0.1: warm up the most-likely `(BLOCK_M, BLOCK_N, BLOCK_K)` configs
   inside `prepare_weights()`.
5. **AMX-layout vs row-major duplication** in pinned memory (§4 option
   a). 2× weight bytes per layer relative to a single layout. Bounded
   and far smaller than v0.1's INT8+FP8 dual storage, but still note
   for memory-tight deployments. Track a follow-up to make AMX consume
   row-major B (cpu_gemm `analysis.md` §5.3).
6. **CUDA-Graph capture and the CPU bucket.** A captured graph cannot
   contain host-side AMX work. Documented: graph mode disables the CPU
   bucket (`M_CPU = 0` under capture); the GPU bucket handles
   everything. For decode at batch≤k_top, GPU INT8 WGMMA is still
   acceptable.
7. **Per-token quant on activation A for the GPU branch.** A dedicated
   prekernel writes `(d_a_int8, d_sfa)`. Its overhead must be amortized
   by the WGMMA. At very small `m_e` (where it wouldn't be) the
   dispatcher routes to CPU instead — so this risk is naturally bounded
   by the dispatch policy itself.

---

## 13. What this plan deliberately omits

- An operator-graph abstraction. Scope is one MoE layer's three GEMMs +
  scatter/reduce.
- Heterogeneous quantization (some experts FP8, some INT8). All experts
  share the INT8 scheme. Mixed-precision MoE is a v2 extension via a
  per-expert dtype enum in `ExpertWeights`.
- Multi-GPU. Single CUDA context. ABI keeps the door open.
- FP8 / FP4. Out of scope; the existing AsymGEMM FP8 paths are
  unaffected by this work and remain available.

---

## 14. Appendix — file pointers for implementers

| Topic                                | Where to read                                                                                                       |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| CPU AMX INT8 GEMM                    | `cpu_gemm/src/kernels/amx/int8_gemm.cpp`, `int8_buffers.h`                                                          |
| CPU dispatch + scratch arena         | `cpu_gemm/src/dispatch/gemm.cpp`, `src/runtime/*`                                                                   |
| CPU public ABI                       | `cpu_gemm/include/cpu_gemm/cpu_gemm.h`                                                                              |
| **GPU FP8 1D1D template to port**    | `AsymGEMM_Main/AsymGEMM/asym_gemm/include/asym_gemm/impls/sm90_fp8_asym_gemm_1d1d.cuh`                              |
| **GPU FP8 1D1D JIT launcher to port**| `AsymGEMM_Main/AsymGEMM/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`                                              |
| Existing FP8 grouped facade          | `AsymGEMM_Main/AsymGEMM/csrc/apis/gemm.hpp::m_grouped_fp8_asym_gemm_nt_contiguous`                                  |
| Python dispatch facade               | `AsymGEMM_Main/AsymGEMM/asym_gemm/dispatch.py`                                                                       |
| MoE control-flow reference           | `ktransformers/kt-kernel/operators/amx/moe_base.hpp::forward_prefill` / `forward_decode`                            |
| Prior cpu_gemm analysis              | `cpu_gemm/analysis.md`                                                                                              |
| Prior AsymGEMM SM90 summary          | `AsymGEMM_Main/AsymGEMM/summary_sm90.md`                                                                            |
| Hopper INT8 WGMMA atoms (CUTLASS)    | `cutlass/include/cute/arch/mma_sm90_gmma.hpp` — search `SM90_64x*x32_S32S8S8_SS`                                    |
