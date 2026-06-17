# Unified Parameter Layout — CPU AMX INT8 ↔ GPU SM90 INT8 AsymGEMM

**Status:** design proposal (pre-implementation).
**Scope:** the INT8 weights (`gate_proj`, `up_proj`, `down_proj`) and their
per-channel scales held in pinned host memory by
`asym_gemm.unified_moe.Layer`. Activation quantization is *not* in scope —
it is recomputed at every call by both backends and never persists across
the dispatch seam.

---

## 1. Question this document answers

> Do the CPU kernel (`cpu_gemm` AMX INT8) and the GPU kernel
> (`sm90_int8_asym_gemm_1d1d`) consume the **same byte layout** for the
> pinned INT8 weights they share?

**Short answer: NO.** Two non-overlapping byte layouts coexist in pinned
host memory today — see §2. This document specifies a unified layout that
both backends can read from a single canonical pinned buffer, and lays out
the kernel-side adjustments needed to keep performance honest.

---

## 2. Current state — two layouts, one weight

The unified runtime
(`asym_gemm/unified_moe/runtime.py::ExpertSlab` and
`from_bf16`) currently maintains **two byte-distinct pinned views** of every
expert weight, so that each backend gets the format it prefers without a
per-call repack.

### 2.1 GPU view — row-major

| Field                | Source       | Layout                          | Dtype     | Residency      |
|----------------------|--------------|---------------------------------|-----------|----------------|
| `gate_int8`          | row-major    | `[G, N_inter, K_hidden]`        | int8      | pinned host    |
| `up_int8`            | row-major    | `[G, N_inter, K_hidden]`        | int8      | pinned host    |
| `down_int8`          | row-major    | `[G, K_hidden, N_inter]`        | int8      | pinned host    |
| `gate_scales`        | per-channel  | `[G, N_inter]`                  | float32   | pinned host    |
| `up_scales`          | per-channel  | `[G, N_inter]`                  | float32   | pinned host    |
| `down_scales`        | per-channel  | `[G, K_hidden]`                 | float32   | pinned host    |
| `gate_sfb`           | broadcast    | `[G, N_inter, K_hidden/128]`    | float32   | **device**     |
| `up_sfb`             | broadcast    | `[G, N_inter, K_hidden/128]`    | float32   | **device**     |
| `down_sfb`           | broadcast    | `[G, K_hidden, N_inter/128]`    | float32   | **device**     |

Constraints imposed by `sm90_int8_asym_gemm_1d1d`:

* `K % BLOCK_K == 0`, with `BLOCK_K = 128` (one FP32 scale per 128 K-elements).
* `N % BLOCK_N == 0`, with `BLOCK_N = 128`.
* `M_max % BLOCK_M == 0` (masked) / per-expert offset alignment (contiguous);
  `BLOCK_M = 64` for the kernel as wired, padded to `256` by the runtime to
  cover heuristic candidates {64, 128, 256}.
* Scale tensors must be device-resident (only `B` may be host-pinned and
  fetched via UVA TMA).
* SFA/SFB layouts the kernel actually reads are the **transposed** forms:
  `sfa: [K//128, M]`, `sfb: [K//128, G*N]`. `dispatch.py` performs the
  transpose at the call site (rows 252, 253, 281, 282 of `dispatch.py`).

### 2.2 CPU view — AMX VNNI blocked + transposed

The CPU path consumes `CG_INT8_PACKED_AMX`, a single contiguous buffer per
expert built by `cg_pack_b_int8_amx`
(`asym_gemm/_cpu_C.pack_b_int8_amx`). The runtime stores these as
`_PinnedAmxBuffer` instances in `gate_packed / up_packed / down_packed`.

Per-expert byte layout (see
`third-party/cpu_gemm/src/kernels/amx/int8_buffers.h::BufferBInt8`):

```
int8_t weights[ n_pad * k ]     // blocked-VNNI tiles, see below
float  scales[  n_pad      ]    // per-output-channel scale = amax / 127
```

The `weights` region is laid out as nested blocks:

```
for n_block in [0, N_BLOCK=64) over N:
  for k_block in [0, K_BLOCK=3584) over K:
    for n_step in [0, N_STEP=32):
      for k_step in [0, K_STEP=64):
        # 32 x 64 tile, then transposed in 32-bit lanes (transpose_16x16_32bit)
        # so _tile_loadd streams it directly into the AMX register file.
```

Constraints:

* `K % K_STEP == 0`, with `K_STEP = 64`.
* `N` is padded internally to `N_STEP = 32`; padded rows hold zero data and
  a zero scale.
* The whole buffer requires 64-byte alignment.
* The trailing scale region lives at offset `n_pad * k` inside the same
  pinned blob. CPU code retrieves it via `int8_b_scales_offset`.

### 2.3 Why these two layouts are not interchangeable

1. **AMX `_tile_dpbssd`** reads B as a sequence of `(N_STEP × K_STEP)` tiles
   in VNNI-4 packing (four K-elements packed as one int32 lane) **with**
   the explicit 16×16-dword transpose applied by
   `transpose_16x16_32bit`. Without that pass the tile is not in the order
   the AMX dot-product unit expects.
2. **Hopper WGMMA** atoms (`SM90_64x*x32_S32S8S8_SS_TN`) require the B
   operand in K-major form with the cute swizzle described by the TMA
   descriptor. The TMA descriptor encodes a global-to-shared box copy; it
   cannot undo the `transpose_16x16_32bit` step or the `N_BLOCK`/`K_BLOCK`
   nesting.
3. The two scale **values** match (both backends use symmetric per-channel
   `amax/127`), but the scale **storage** does not: CPU keeps a flat
   `[n_pad]` FP32 vector appended to the weight buffer; GPU keeps a
   separate device tensor broadcast across K-blocks and transposed for TMA
   ingest.

Net effect today:

* Per-expert pinned bytes ≈ **2 × N × K** (row-major) + **N × K + small
  scale tail** (VNNI packed) ≈ **3 × N × K** bytes (dominant term — the
  scale tails are O(N)).
* No per-call repack overhead on either side (both layouts are precomputed
  at `from_bf16`).
* The "two pinned views must agree byte-for-byte modulo permutation"
  invariant is enforced only by a parity test
  (`test_cpu_vs_gpu_single_expert`).

---

## 3. Goal

A single byte layout for INT8 weights + per-channel scales in pinned host
memory, consumed unchanged by **both** the GPU SM90 INT8 kernel and the
CPU AMX INT8 kernel, with **no regression** in CPU AMX throughput beyond
what is explicitly documented and accepted.

Non-goals:

* Unifying the SFA/SFB layout that the SM90 TMA descriptor reads. SFA is
  always recomputed from activations at call time; SFB is a small
  broadcast tensor derived once on load — neither sits on the dispatch
  seam.
* Re-architecting the AMX A-side packing.
* FP8 / FP4 / BF16 paths.

---

## 4. The unified layout

### 4.1 Canonical weight representation

Per expert, per projection, in pinned host memory:

```
weights[G, N, K]    int8           row-major, C-contiguous
scales [G, N]       float32        per-output-channel  =  amax(weight_row) / 127
```

Strictly the layout the GPU kernel already wants (§2.1). Constraints from
both backends combined:

| Constraint                  | Origin                                       |
|-----------------------------|----------------------------------------------|
| `K % 128 == 0`              | GPU `BLOCK_K = 128` (also satisfies CPU's 64)|
| `N % 128 == 0`              | GPU `BLOCK_N = 128` (also satisfies CPU's 32)|
| 64-byte alignment of base   | CPU `BufferBInt8` `posix_memalign(64)`       |
| `torch.pin_memory()`        | GPU UVA TMA + CPU `cudaHostRegister`         |

`K % 128` and `N % 128` are already enforced at
`Layer.from_bf16(...)` and are stricter than the CPU's own minima, so no
new restriction lands on either side.

Per-expert pinned bytes: **N × K + 4 × N**. That is **2.0×** less weight
storage than the current dual-layout design (a 16-billion-parameter MoE
saves on the order of tens of GiB of pinned RAM).

### 4.2 What disappears

* `ExpertSlab.gate_packed`, `up_packed`, `down_packed` and the
  `_PinnedAmxBuffer` helper — no second pinned view.
* `_C.pack_b_int8_amx` is no longer called at load time. Whether the
  symbol stays or becomes internal-only is up to §5's choice.

### 4.3 What stays

* `ExpertSlab.gate_int8 / up_int8 / down_int8` and `*_scales`: these become
  the **single source of truth**.
* `gate_sfb / up_sfb / down_sfb`: still built once at load on the device
  by broadcasting `[G, N]` scales across `K//128`. Device-only, untouched
  by the unification.
* The Python-side SFA/SFB transpose in `dispatch.py`: unchanged.

---

## 5. Reconciling the CPU AMX kernel

The CPU kernel cannot consume row-major B in its current form (§2.3 point
1). We need to choose **one** of three strategies. Recommended is C (with
B available as an interim).

### 5.1 Strategy A — keep both pinned views (rejected)

What it is: status quo. `from_bf16` continues to compute the AMX-packed
twin alongside the row-major copy.

* Footprint: ~2× weight bytes pinned.
* Repack cost: zero (paid once at load).
* Code touched: none.

Rejected because the explicit ask is to use a *unified* layout — keeping
two layouts but calling them "unified" defeats the goal.

### 5.2 Strategy B — pack-on-first-touch + per-runtime LRU cache (interim)

What it is: the pinned canonical layout is row-major. The CPU `Runtime`
gains an LRU cache (default size 16 expert × projection slots) of
AMX-VNNI packed B blobs kept in pinned host memory. On a cache miss the
runtime packs from the canonical row-major weight into a scratch slot
once; on a hit the AMX kernel reuses the packed slot directly.

| Aspect                  | Detail                                                    |
|-------------------------|-----------------------------------------------------------|
| Canonical bytes pinned  | 1× row-major + bounded LRU (≤ active_experts_per_layer × per_expert_pack_bytes) |
| Repack cost (miss)      | ≈ `n_pad·k` bytes streamed through AVX-512: ~150 µs at N=K=4096 |
| Repack cost (hit)       | 0                                                         |
| Cache invalidation      | Tied to `ExpertSlab` lifetime — packs are valid for the lifetime of the row-major buffer they were built from |
| Code touched            | `cpu_module.cpp` (new packed-cache API), `unified_moe/runtime.py` (drop `gate_packed` etc.), `_cpu_C` (passes cached slot to `gemm_bf16_int8_packed`) |

Per-MoE-layer behaviour: gate → up → down all hit the same expert in
quick succession, so within one layer the first call eats the repack and
the next two hit cache. The LRU then evicts when the next layer's
experts come in. Net per-token CPU overhead ≈ (pack cost) ÷ (3 calls) ≈
50 µs per expert per layer — close to the dispatcher's `M_CPU`
break-even, so this strategy effectively *raises* `M_CPU` (pushes more
work to GPU). Tolerable; not great.

This is the right interim landing point if Strategy C cannot be
delivered in the same release as the layout change.

### 5.3 Strategy C — stride-aware AMX kernel (recommended end state)

What it is: replace the CPU kernel's reliance on the precomputed
VNNI/blocked B with a kernel that reads row-major `[N, K]` directly via
strided `_tile_loadd`. Removes the need for any AMX side layout at all.

Why this is feasible (rather than a fantasy):

* `_tile_loadd` takes a 2D base + byte stride. Loading a 32×64 sub-tile
  of B from row-major `[N, K]` with `stride = K bytes` gives you 32 rows
  of N (output channels), 64 K-elements per row, with each row already
  contiguous along K. **A row of int8s contiguous along K IS VNNI-4
  packed** — four consecutive bytes pack into one 32-bit lane, which is
  exactly the `_tile_dpbssd` input shape.
* The only structural piece the current `BufferBInt8::pack_tiles`
  performs that strided load does not is the `transpose_16x16_32bit` step
  (`int8_buffers.h:304`). That transpose exists to make the AMX inner
  loop iterate over `K_STEP` before `N_STEP` in a way that matches the
  packed layout's storage order. A row-major source has the opposite
  order (`K_STEP` is contiguous along memory rows, `N_STEP` is strided),
  which is actually the *natural* AMX inner-loop order.
* The `cpu_gemm` analysis already flags stride-aware AMX as planned work
  (`third-party/cpu_gemm/analysis.md` §4 item 3 + §5.3).

What the work looks like in `cpu_gemm`:

1. **New kernel:** `kernels/amx/int8_gemm_rm.{h,cpp}` —
   `int8_run_rm(...)` that:
   * takes `B` as `(const int8_t*, size_t ldb)` row-major,
     `scales` as `(const float*, size_t)`;
   * pre-touches the row-stride into the AMX tile configuration
     (`tileconfig.colsb` must still be 64);
   * issues `_tile_loadd(tmm_b, B_base + k_step, ldb)` per K-step;
   * **drops** the `transpose_16x16_32bit` permutation by reversing the
     inner loop iteration order (`(n_step, k_step) → (k_step, n_step)`)
     and the corresponding `_tile_dpbssd` operand layout.
   * Threading along `N_BLOCK` chunks unchanged.
2. **Dispatcher hook:** add `CG_INT8` (unpacked) eligibility that no
   longer fails `lda == k && ldb == k`. Specifically allow `ldb >= k`
   for the row-major B; A activations still come row-major from BF16.
3. **Path retired:** `CG_INT8_PACKED_AMX` and `cg_pack_b_int8_amx` can
   stay for callers that opt in offline, but the unified runtime stops
   using them.

| Aspect                  | Detail                                                    |
|-------------------------|-----------------------------------------------------------|
| Canonical bytes pinned  | 1× row-major only                                         |
| Repack cost             | 0 (no repack)                                             |
| Per-tile load overhead  | strided load fetches one cacheline per AMX row — same total bytes the packed load would, with a slightly worse L2 stride pattern in some `K_BLOCK` regimes. Mitigated by sizing `K_BLOCK` so the AMX inner loop reuses L2-resident B slabs. |
| Expected perf delta     | within ±5% of the packed kernel at M ≥ 32 per the upstream `cpu_gemm` plan; small loss expected at M < 8 where the AMX path is already not optimal — measure during integration |
| Code touched            | new file in `cpu_gemm`, dispatcher hook, unit tests against the packed kernel |

This is the right destination because it carries no extra bytes, no
per-call pack overhead, and removes a long-standing cpu_gemm tech-debt
item (`analysis.md` §4.3).

### 5.4 Recommended phasing

```
Phase 1 (this PR)        : land §4 row-major canonical layout in
                           unified_moe/runtime.py + use Strategy B
                           (LRU pack cache) inside the CPU runtime
                           wrapper. Behaviour: weight footprint drops
                           ~33% immediately; CPU throughput unchanged
                           on warm cache, ~+50 µs/expert on first touch.

Phase 2 (cpu_gemm)       : land Strategy C kernel in cpu_gemm
                           (stride-aware AMX INT8). Once landed, retire
                           the LRU cache from Phase 1.

Phase 3 (cleanup)        : drop _PinnedAmxBuffer, gate_packed/up_packed/
                           down_packed dataclass fields, the
                           pack_b_int8_amx pybind binding (or mark
                           internal-only).
```

Phases 1 and 2 can ship independently; Phase 3 depends on Phase 2.

---

## 6. Layout-vs-performance trade-off table

| Strategy | Pinned weight bytes | CPU per-call extra | CPU throughput @ small M | Unified layout? | Effort  |
|----------|--------------------|-------------------:|---------------------------|------------------|---------|
| A: dual layouts (today)    | 2.0×            | 0 µs               | reference                 | NO               | 0       |
| B: LRU pack-on-touch       | 1.0× + LRU      | 0 µs (hit) / ~150 µs (miss) | ≈ reference (post-warmup) | YES         | small   |
| C: stride-aware AMX        | 1.0×            | 0 µs               | within ±5% of reference   | YES              | medium  |

The unification cost is **not** uniform across batch sizes:

* `m_e = 1..4` (decode): per-token CPU cost is small in absolute terms;
  even an extra 50 µs from Strategy B's first-touch path is comparable to
  the AMX compute time. Strategy C is required to keep the small-M
  regime sharp.
* `m_e = 16..32` (`M_CPU` cross-over): Strategy B pays well — the pack
  cost amortizes over a longer kernel.
* `m_e > M_CPU`: the dispatcher routes to GPU; CPU layout is irrelevant.

---

## 7. Scales: unified by construction

The per-channel scale `scales[N] = amax(weight_row) / 127` is identical
across both backends. After §4 lands:

* CPU AMX kernel receives `scales[N]` as a separate `const float*` argument
  (the dispatcher already takes `desc->b_scales` — the appended-vector
  convention only matters for the packed `CG_INT8_PACKED_AMX` path which
  Strategy C retires).
* GPU kernel continues to use `SFB[G, N, K//128]` device tensor built once
  at load by broadcasting `[G, N]` across `K//128`. The transpose to
  `[K//128, G*N]` for the TMA SF descriptor is unchanged
  (`dispatch.py:281-282`).

So the canonical pinned scale layout is `[G, N] float32 pinned host` for
both backends; the GPU additionally maintains a device-side broadcast
view derived from it. The CPU consumes the pinned tensor directly.

Activation-side scales (SFA / per-row A scale) are recomputed live at
every call by `int8_pack_a_bf16` (CPU) and `quantize_per_token_int8_gpu`
(GPU). Neither persists on the dispatch seam, so there is nothing to
unify on the A side.

---

## 8. Concrete change set

### 8.1 In `cpu_gemm` (Phase 2)

| File                                                              | Change |
|-------------------------------------------------------------------|--------|
| `src/kernels/amx/int8_gemm_rm.h` (new)                            | Public entry for the stride-aware INT8 kernel: `int8_run_rm`, `int8_pack_a_bf16_rm`, `int8_unpack_rm`. |
| `src/kernels/amx/int8_gemm_rm.cpp` (new)                          | Strided `_tile_loadd` over row-major B; inverted inner loop relative to the packed kernel; reuses `Int8KernelTraits`. |
| `src/dispatch/gemm.cpp`                                           | New `amx_int8_rm_eligible(d)` accepting `dtype_b == CG_INT8`, `b_scales != nullptr`, `ldb == k` *or* the new stride-aware accepting `ldb >= k`. Wire `run_amx_int8_rm`. |
| `tests/test_amx_int8_rm.cpp` (new)                                | Parity vs the existing packed kernel across the (N,K,M) sweep. |
| `analysis.md`                                                     | Mark §4 item 3 / §5.3 (stride-aware) done. |

### 8.2 In AsymGEMM repo (Phase 1)

| File                                                              | Change |
|-------------------------------------------------------------------|--------|
| `asym_gemm/unified_moe/runtime.py`                                | Drop `gate_packed`, `up_packed`, `down_packed`, `_PinnedAmxBuffer`. CPU expert forward calls `_C.gemm_bf16_int8` (row-major path) instead of `_C.gemm_bf16_int8_packed`. |
| `csrc_cpu/cpu_module.cpp`                                         | Phase 1: add an LRU-cached `gemm_bf16_int8_pack_cached(rt, slab_id, a_bf16, b_int8, b_scales, c_fp32, n, k, alpha, beta)` that owns a Strategy B cache keyed by `(b_data_ptr, n, k)`. Phase 2 (post-C): collapse this back into a plain `gemm_bf16_int8` call. |
| `asym_gemm/unified_moe/__init__.py` docstring                      | Update to note the single canonical layout. |
| `tests/test_unified_moe.py` (parity)                              | The byte-identity invariant test (`test_cpu_vs_gpu_single_expert`) becomes trivial — both backends already read the same bytes. Keep the numerical parity check (`max(|cpu - gpu|) / |cpu| < 1e-3`). |

### 8.3 Verification

The existing parity ladder still applies (see `int8.md` §7 stages A–F and
`unified_kernel.md` §9.1 tests 1–5). Two new checks land alongside the
layout change:

1. **No-second-buffer assertion.** Add a runtime assert in `ExpertSlab`
   that the `*_int8` tensors are the only pinned weight allocations of
   their projection. Catches accidental reintroduction of a side layout
   during future refactors.
2. **Memory regression test.** Track `torch.cuda.cudart().cudaMemGetInfo`
   before / after `Layer.from_bf16` on a fixed `(G=32, N=4096, K=4096)`
   slab; weight footprint should drop by ~`G·N·K` bytes vs the current
   dual-layout baseline.

### 8.4 What does *not* change

* Public Python facade (`asym_gemm.unified_moe.Layer.from_bf16`,
  `Layer.forward`) — same signatures.
* The SM90 GPU kernel `sm90_int8_asym_gemm_1d1d` and its launcher.
* The SFA/SFB on-device transpose convention in `dispatch.py`.
* The `M_CPU` dispatch policy mechanism (its measured value will shift
  per §5.4 phasing, but the LUT structure is unchanged).

---

## 9. Risk register

| Risk                                                               | Mitigation                                                              |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| Strategy C kernel underperforms the packed AMX kernel at large M    | Phase 1 ships Strategy B first; Phase 2 gates retiring B on a measured ≤ 5% throughput delta from C across the M sweep. |
| LRU cache contention under multi-stream CPU calls                  | Cache is per-runtime, lock-protected, evict-on-insert; sized to ≥ `top_k × concurrent_layers` so steady state is hit-only. |
| `cudaHostRegister` lifetime on weight tensors                      | `torch.pin_memory()` already manages pin lifetime; deleting `_PinnedAmxBuffer` removes a parallel pin path and simplifies cleanup. |
| Stride-aware kernel breaks AMX tile config invariants              | `tileconfig.colsb` must stay at 64 (K_STEP). The stride is encoded in `_tile_loadd`'s third argument, not in `tileconfig`. Unit test in `cpu_gemm/tests/test_amx_int8_rm.cpp` covers this. |
| Padded N (multiples of 128) wastes some pinned bytes at small N    | Same padding the GPU already requires; loss is bounded by `(N pad to 128 − N) × K`. For typical MoE shapes (N ≥ 1024) this is < 1%. |
| Per-expert `M_CPU` measurements done with Strategy A no longer hold | Re-run the autotune sweep at the end of Phase 1 and Phase 2; persist the LUT keyed by `(N, K, strategy_version)`. |

---

## 10. Appendix — pointers

| Topic                                       | File / line                                                                                  |
|---------------------------------------------|----------------------------------------------------------------------------------------------|
| Current pinned-host dual-layout setup       | `asym_gemm/unified_moe/runtime.py:155-322`                                                  |
| CPU AMX VNNI-packed B layout                | `third-party/cpu_gemm/src/kernels/amx/int8_buffers.h:138-322`                                |
| CPU AMX INT8 traits + offline pack          | `third-party/cpu_gemm/src/kernels/amx/int8_gemm.h:30-105`                                    |
| CPU dispatcher entry for prepacked B         | `third-party/cpu_gemm/src/dispatch/gemm.cpp:118-170`                                          |
| GPU SM90 INT8 kernel                        | `asym_gemm/include/asym_gemm/impls/sm90_int8_asym_gemm_1d1d.cuh:117-141, 226-275`            |
| GPU SM90 INT8 host launcher                 | `csrc/jit_kernels/impls/sm90_int8_asym_gemm_1d1d.hpp:149-271`                                |
| GPU SFA/SFB transpose at call site          | `asym_gemm/dispatch.py:231-285`                                                              |
| Stride-aware AMX (planned)                  | `third-party/cpu_gemm/analysis.md:165-193`                                                   |
| Unified MoE design intent                   | `unified_kernel.md:116-159, 414-456`                                                          |
| INT8 SM90 design + verification status      | `int8.md:8-95, 366-462`                                                                       |
