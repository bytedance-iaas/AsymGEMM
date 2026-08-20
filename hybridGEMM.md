# hybridGEMM — one SM90 kernel fusing deepGEMM (HBM experts) and asymGEMM (host experts)

**Status:** plan (v2 — supersedes the v1 placement/migration draft; expert
placement is now assumed **given**: the runtime already knows which experts
reside in HBM and which stay in pinned host memory).
**Scope:** one kernel launch per grouped-GEMM call that computes
HBM-resident experts with a deepGEMM-style pipeline and host-resident
experts with the asymGEMM pipeline, on **disjoint SM ranges sized so both
sides finish together**. The CPU AMX bucket stays as-is on top.
**Starting point:** `asym_gemm/include/asym_gemm/impls/sm90_int8_gemm.cuh`
— currently a verbatim copy of DeepGEMM's
`deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh` (BF16, no scales,
`deep_gemm::` headers). Reference tree: `/workspace/AsymGEMM_Main/DeepGEMM`.

---

## 1. Why one kernel

Today `Layer.forward` (`asym_gemm/unified_moe/runtime.py:1428-1507`) runs
the GPU bucket as **2-3 separate launches** — a "cached" partition (HBM),
then "staged"/"slab" partitions (host weights) — each with its own layout
build, TMA descriptor set, and kernel launch. Three costs follow:

1. **Serial SM occupancy.** Partitions run back-to-back on one stream: the
   cached partition's GEMMs finish before the streamed partition starts,
   so while the (compute-light, PCIe-bound) streamed partition runs, most
   SMs idle. The PCIe transfer and the HBM math never overlap *inside the
   GPU* — only against the CPU bucket.
2. **Launch/layout overhead × partitions.** At decode (small T) the fixed
   costs dominate — this is exactly the `b0` intercept the dispatch model
   fights (`adaptive_dispatch.md` §3).
3. **Wrong kernel for HBM.** The cached partition today reuses the *asym*
   INT8 kernel (K-outer, single-slot B, partial-sum `TMA_REDUCE_ADD`
   round-trips through HBM). That loop order is the right trade for a
   21.5 GB/s link and the wrong one for 3 TB/s HBM: B occupies one slot
   (no K-pipelining of weights) and every K-block does a read-modify-write
   of the whole output tile. DeepGEMM's M-outer multi-stage pipeline is
   the proven shape for HBM-resident weights.

The fused kernel fixes all three: HBM experts get a real deepGEMM-class
INT8 pipeline, host experts keep the asym pipeline, both progress
concurrently on disjoint SMs inside one launch, and the SM split is the
load-balancing knob.

```
one launch, gridDim.x = kNumSMs (persistent CTAs)
┌────────────────────────────── GPU ──────────────────────────────┐
│ CTA rank 0 .. S_host-1          │ CTA rank S_host .. kNumSMs-1  │
│ asym side (K-outer, TMA over    │ hbm side (M-outer, kNumStages │
│ PCIe from pinned host slab)     │ deep, B from HBM arena)       │
│ work: host-resident segments    │ work: HBM-resident segments   │
│           └── on own-side exhaustion, steal HBM tiles ──┐       │
└─────────────────────────────────────────────────────────────────┘
   concurrent with: CPU AMX bucket (unchanged, dispatch model on top)
```

## 2. Anatomy of the two parents (what actually differs)

| | deepGEMM `sm90_bf16_gemm.cuh` (the copy) | asym `sm90_int8_asym_gemm_1d1d.cuh` |
|---|---|---|
| grid | persistent: `kNumSMs` CTAs, `sched::Scheduler::get_next_block` loops over (m,n) tiles | non-persistent: `(ceil(N/BN), num_segments)`, CTA = one (n-block, expert) |
| loop order | M-outer: per tile, full K swept, accum lives in registers | K-outer, M-inner: B loaded once per K-block, reused across all M-blocks |
| A/B staging | both A and B `kNumStages`-deep (≥5, stage-merge up to 10+) | A `kNumStages`-deep, **B single slot** (`smem_b[0]`, `full_barriers_b[1]`) |
| dtype / MMA | BF16, `BF16MMASelector`, FP32 accum | INT8, `INT8MMASelector` (S32 accum) |
| scales | none | SFA[M]/SFB[N] fp32 via TMA, dequant in epilogue (lines 368-447) |
| output | BF16 STSM + `TMA_STORE` (or FP32), one store per tile | FP32, `TMA_STORE` on k=0 then `TMA_REDUCE_ADD` per K-block (partial sums) |
| epilogue smem | 1 × D tile | 2-stage CD ring (`kNumTMAStoreStages=2`) |
| scheduler input | `grouped_layout` (DeepGEMM conventions) | `offsets` pairs + `experts` ids (`asymScheduler.cuh:83-124`), matches `_build_layout` |
| warp spec | 128 TMA + `kNumMathThreads` math, regs 48/224-248 | same structure (names swapped for SM100 ABI compat) |

Both already share the barrier discipline, `PatternVisitor` smem layout,
GMMA-descriptor fast path (`advance_gmma_desc_lo`), and warp-group
register reconfig — the merge is structural, not a rewrite of either
pipeline.

## 3. Phase A — make `sm90_int8_gemm.cuh` a real INT8 HBM kernel

Standalone deliverable: replace the cached-partition GEMM (and serve as
the hbm side of the fused kernel). Concrete edit list on the copied file:

1. **Header/namespace port.** Swap `deep_gemm/*` includes for the
   `asym_gemm` equivalents (`asym_gemm/common/utils.cuh`,
   `sm90_utils.cuh` — which already carries `INT8MMASelector`,
   `make_gmma_desc`, `advance_gmma_desc_lo` — and `types.hpp`); namespace
   `deep_gemm` → `asym_gemm`; rename `sm90_bf16_gemm_impl` →
   `sm90_int8_gemm_impl`. DeepGEMM's persistent `sched::Scheduler` is the
   one piece with no asym counterpart — port it into
   `asym_gemm/common/persistentScheduler.cuh` (next item trims it).
2. **Scheduler: contiguous grouped over the asym layout.** Keep the
   persistent tile loop but enumerate `(segment, m_block, n_block)` from
   the **same `offsets`/`experts` arrays** `asymScheduler` consumes
   (`asymScheduler.cuh:110-122`), so `Layer._build_layout` feeds both
   kernels unchanged. `experts[seg]` indexes the B/SFB tensor maps —
   for the HBM kernel the runtime passes **arena slot ids** here (the
   existing `kernel_ids = self._slot_np[part]` convention,
   `runtime.py:1478`). Drop DeepGEMM's `MGroupedMasked`/`KGrouped`/
   `Batched` variants from the port; `Normal` + contiguous-grouped
   suffice.
3. **INT8 data path.**
   - `using WGMMA = INT8MMASelector<BLOCK_N>::type`; accumulators
     `int32_t` (delete the `float accum` and BF16 STSM specialization).
   - smem A/B become `int8_t` — element size halves, so the stage
     heuristic gets ~2× the stages or 2× BLOCK_K for free (see item 6).
   - Static-assert `BLOCK_K % WGMMA::K == 0` as the asym kernel does.
4. **Scales + dequant epilogue.** Add `tensor_map_sfa` / `tensor_map_sfb`
   params, one smem slot each per stage-group, TMA loads next to A/B
   loads, and the register-level dequant from the asym epilogue
   (`sm90_int8_asym_gemm_1d1d.cuh:368-386, 438-447`):
   `out = sfa[r] * sfb[c] * (float)accum`. One simplification vs asym:
   this kernel accumulates the **whole K** in registers, so scales are
   applied once per tile, not per K-block — SFA/SFB are plain `[M]`/`[N]`
   vectors (the runtime's per-K-block broadcast `*_sfb` tensors stay
   usable: read block 0, or pass a 1-block view).
5. **Output.** `cd_dtype_t` = FP32 with plain `TMA_STORE` (no
   REDUCE_ADD — full-K accumulation needs no partial sums). Keep
   `kWithAccumulation` wired to `TMA_REDUCE_ADD` for drop-in parity with
   the asym kernel's output contract when fused (both sides then write
   the same `[M_padded, H]` fp32 buffer the runtime `index_add_`s).
6. **Config heuristic + JIT launcher.** New
   `csrc/jit_kernels/impls/sm90_int8_gemm.hpp` cloned from
   `sm90_int8_asym_gemm_1d1d.hpp`'s runtime class, but with DeepGEMM-side
   config search: maximize `kNumStages` at INT8 element size, standard
   BLOCK_M∈{64,128,256}, TMA multicast on A when the tile grid allows.
   Register under `csrc/apis/gemm.hpp` + python export.
7. **Tests.** `tests/test_sm90_int8.py` gains a `deep` variant: parity vs
   the torch INT8 reference and vs the asym kernel on identical grouped
   inputs; bench vs the asym-kernel-on-HBM cached path (expectation:
   clear win at prefill M, parity or win at decode).

Exit gate A: cached partition in `runtime.py` switched to the new kernel
behind `ASYMGEMM_HBM_KERNEL=deep|asym`, parity suite green, measured
speedup on the cached partition logged.

## 4. Phase B — the fused kernel `sm90_int8_hybrid_gemm.cuh`

One `__global__`, template union of both parents:

```
sm90_int8_hybrid_gemm_impl<...>(
    uint32_t* offsets_host, uint32_t* experts_host,   // asym-side segments (global expert ids → host slab maps)
    uint32_t* offsets_hbm,  uint32_t* experts_hbm,    // hbm-side segments (arena slot ids → arena maps)
    uint32_t  s_host,                                 // CTA-rank split: ranks < s_host run the asym side
    uint32_t* steal_counter,                          // device atomic, §5
    tensor_map_a, tensor_map_sfa,                     // shared (same quantized activations)
    tensor_map_b_host, tensor_map_sfb_host,           // pinned host slab (UVA), asym side
    tensor_map_b_hbm,  tensor_map_sfb_hbm,            // HBM slot arena, hbm side
    tensor_map_cd)                                    // shared fp32 out (+REDUCE_ADD)
```

Design decisions, in dependency order:

- **Grid & side assignment.** `gridDim.x = kNumSMs`, persistent. Branch
  once at kernel top on `blockIdx.x < s_host`. `s_host` is a **runtime
  argument**, not a template constant — the balance point moves per
  forward with the routing histogram, and re-JITting per split is a
  non-starter. Inside each branch the CTA runs its parent's loop
  essentially verbatim.
- **Work enumeration.**
  - asym side: today's CTA identity `(blockIdx.y=segment, blockIdx.x=
    n-block)` becomes a flat item index `i = rank + s_host * iter` over
    `num_host_segments × ceil(N/BN)` items — the explicit-ids
    `asymScheduler` constructor (`asymScheduler.cuh:92`, added for sEP)
    already accepts `(seg_idx, nblk_idx)` decoupled from blockIdx, so the
    persistent wrapper is a ~10-line loop.
  - hbm side: Phase A's persistent scheduler, CTA ranks re-based by
    `rank - s_host` over `kNumSMs - s_host` peers.
- **Shared memory.** One `extern __shared__` buffer laid out as the
  **union** (max) of the two sides' plans; each branch overlays its own
  `PatternVisitor` map. Barrier arrays sized for the max stage count of
  either side; each branch initializes and uses only its own set. Both
  sides are INT8 so the driving budgets are close; the JIT heuristic
  picks the two configs **jointly** under
  `max(smem_asym, smem_hbm) ≤ smem_capacity` (227 KB on H20/H800).
- **Block shapes per side.** Independent BLOCK_M/BLOCK_K per side (asym
  wants max BLOCK_K + 2-ish A stages; hbm wants deep stages), but **same
  BLOCK_N and same math-warp-group count** in v1 — same `INT8MMASelector`
  instantiation, same epilogue store geometry, one `__launch_bounds__`
  (128 TMA + 256 math), no per-side register-pressure cliff. Revisit
  per-side BLOCK_N only if profiling demands it.
- **Cluster/multicast: off in v1** (`kNumMulticast = 1`). Cluster shape
  is launch-wide; with a runtime `s_host` a 2-CTA cluster can straddle
  the side boundary. Cost: the hbm side loses TMA multicast on A —
  acceptable at INT8 arithmetic intensity; v2 can round `s_host` to even
  and keep clusters side-pure.
- **Output contract.** Both sides write the shared fp32 `[M_padded, H]`
  buffer; asym side keeps its k=0-STORE/then-REDUCE_ADD protocol, hbm
  side plain-stores (its segments are disjoint rows from the asym side's,
  so no cross-side ordering is needed).
- **Layout plumbing (runtime.py).** `_build_layout` runs once over
  `gpu_experts` and emits the two segment lists split by the given
  residency mask (`self._cached_mask_np` today): host segments keep
  global expert ids, HBM segments map through `_slot_np`. The
  cached/staged/slab partition loop (`runtime.py:1438-1494`) collapses to
  one hybrid launch.

Exit gate B: fused kernel passes parity vs the two-launch path on forced
splits (all-host / all-HBM / mixed), and beats it end-to-end at decode
sizes (launch-count savings) with `s_host` hand-set.

## 5. Phase C — balancing the SM split

Two mechanisms, coarse then fine:

**C1. Host-side `s_host` model.** The asym side is **link-bound**, not
SM-bound: in-kernel TMA over PCIe tops out ~21.5 GB/s (`runtime.py:113`)
— a handful of CTAs saturate it; extra CTAs add nothing but stolen HBM
throughput. So:

```
T_asym(s)  ≈ bytes_host_experts / BW_link(s)         # BW_link(s) saturates fast: s* ≈ 8-16 CTAs
T_hbm(s)   ≈ max( flops_hbm / (rate_sm · (kNumSMs - s)),
                  bytes_hbm_experts / BW_hbm )
s_host     = argmin_s max(T_asym(s), T_hbm(s)),  s ≥ s*_saturation while host work exists
```

In practice: `s_host = min(s_sat, enough-for-host-items)` then check the
hbm side isn't the bottleneck; all coefficients start as shape-derived
priors and are refit online exactly like `DispatchModel` (`_RATES_PRIOR`
pattern), fed by per-side in-kernel timings (below). This slots into the
existing dispatch machinery: the CPU/GPU makespan partition
(`dispatch_model.partition`) stays the outer loop; its GPU cost model
becomes a two-feature model (host-bytes, hbm-work) whose internal split
is `s_host`.

**C2. Device-side stealing (one-directional).** Prediction error lands as
idle CTAs. Fix inside the launch: the hbm side's tile enumeration is an
**atomic ticket counter** (`steal_counter`) rather than static striding;
asym-side CTAs that exhaust the host segment list fall through to popping
hbm tiles from the same counter. One direction only — hbm CTAs never pull
host segments (the link is already saturated; more consumers just queue
on PCIe and pin SMs doing nothing). The asym→hbm fallthrough reuses the
Phase-B branch as a function call, so code cost is small; smem re-init on
side-switch is a `__syncthreads` + barrier re-init in the already-resident
CTA.

**C3. Per-side observability.** One CUDA event pair around a single fused
launch can no longer separate the regimes for the cost models. Add a tiny
`uint64 stats[4]` device buffer: first/last CTA of each side records
`%globaltimer` at start/exit (min/max via `atomicMin/Max`). Read back
lazily with the existing `harvest()` pattern — this feeds C1's refit and
replaces the two event brackets (`runtime.py:1470-1507`).

**C4. asym-side B double-buffer (independent win, do early).** The
single-slot B (`smem_b[0]`) is why in-kernel TMA sits at 21.5 GB/s — the
fetch of K-block k+1 waits for k's consumption. INT8 halved the B
footprint vs BF16; spend it on `kNumStagesB = 2`. Every GB/s recovered
here shrinks the host-side bucket's makespan *and* lowers `s_sat`,
freeing SMs for the hbm side. Measure against the copy-engine staging
path (~50 GB/s): staging stays the default transport if it still wins —
the fused kernel then simply sees those experts on the hbm side (staged
ring slots are HBM), and the asym side handles only what staging didn't
cover (ring overflow / staging disabled). The fused design is agnostic:
"hbm side" = arena slots ∪ ring slots.

Exit gate C: on the synthetic G=32 sweep and a Qwen3-30B-class serve
bench, fused + modeled `s_host` + stealing ≥ the best hand-tuned static
split at every T, and ≥ the current multi-launch path everywhere
(decode T especially).

## 6. Phase D — runtime integration & cleanup

- `runtime.py`: hybrid launch behind `ASYMGEMM_HYBRID_KERNEL=1`;
  cached/staged/slab loop kept as fallback until gate C holds on both
  H20 and H800. `_cached_gpu_decode` / `_cached_gpu_forward_any`
  (CUDA-graph decode) migrate last — persistent + runtime `s_host` is
  graph-capturable (all args are device tensors/scalars), but validate
  replay explicitly.
- Dispatch model: GPU `_BackendModel` refit against fused timings; the
  serial-enqueue penalty shrinks (one launch) — re-derive `b0` prior.
- NUMA-TP: host slab halves are two tensor maps today (`slab_a`/
  `slab_b` partitions). Fused v1 takes both host maps + a per-segment
  half flag (or two host segment lists); the asym side picks the map per
  segment. No extra launches.
- sEP (`csrc/ep_steal`) queued-kernel mode: the explicit-ids
  `asymScheduler` ctor is shared infrastructure — make the persistent
  wrapper use it so sEP's host-counter pop composes with the hybrid
  kernel later; do not block on it.
- Docs: fold results into `design_overview.md` §4; retire the
  cached-partition description.

## 7. Risks / open questions

- **Register/instruction pressure from the union kernel.** Two full
  pipelines in one `__global__` — if the compiler doesn't fully dead-code
  each branch per warp, spills kill both sides. Mitigation: `if (rank <
  s_host)` at the very top with `__noinline__` per-side functions if
  needed; watch `-Xptxas -v` in the JIT log. Fallback design: two
  cooperative kernels on one stream with an SM-count-limited grid each
  (loses the single-launch win, keeps concurrency).
- **JIT compile time / cache pressure.** The hybrid template multiplies
  config axes (two block-shape sets). Constrain the heuristic to a small
  joint config menu; key the JIT cache on both sides' shapes.
- **Steal-path correctness.** Side-switching CTAs re-initialize barriers
  in live shared memory — needs a careful quiesce (all local TMA stores
  drained, `tma_store_wait<0>`) before re-init; this is the highest-risk
  10 lines of the project. Ship gate B/C with stealing off, enable via
  `ASYMGEMM_HYBRID_STEAL=1` after soak.
- **PCIe contention triangle.** asym-side in-kernel reads, the staging
  ring's copy-engine H2D, and the CPU bucket's D2H activation gather all
  share the link. The dispatch model absorbs steady-state contention into
  fitted rates, but pathological overlap (large gather + big host bucket)
  may want the gather charged to the same budget the `s_host` model uses.
- **REDUCE_ADD atomicity/determinism.** fp32 `TMA_REDUCE_ADD` ordering
  differs run-to-run; parity tests must keep the existing tolerance-based
  comparison (they do today for the asym kernel — no change, just don't
  tighten).
- **H20 vs H800 balance points.** `s_sat`, HBM rates, and SM counts all
  differ; everything tunable must flow through the fitted-rates path, no
  hard-coded splits.

## 8. Milestone summary

| phase | deliverable | gate |
|---|---|---|
| A | `sm90_int8_gemm.cuh` INT8-ized + launcher + tests | parity green; beats asym kernel on HBM cached partition |
| B | `sm90_int8_hybrid_gemm.cuh`, one launch, manual `s_host` | parity on forced splits; decode win from launch collapse |
| C | balance: `s_host` model, stealing, B double-buffer, per-side timers | ≥ best static split ∀T; ≥ multi-launch everywhere |
| D | default-on integration, graph capture, NUMA-TP, docs | serve bench (Qwen3-30B class) regression-free on H20 + H800 |

## 9. Correctness estimate

Point-by-point over the numerically or structurally risky pieces:

1. **INT8 accumulation is exact; overflow has 60× headroom.** The S32
   WGMMA path is integer-exact: `|accum| ≤ K · 127² = K · 16129`, so int32
   (2.1e9) overflows only at K ≳ 133,000; the largest K here is
   `inter = 2048` (down proj) → max 3.3e7. Safe for any realistic shape
   (up to K ≈ 128K).
2. **The deep kernel is *more* accurate than the asym kernel, not less.**
   Phase A accumulates the full K in int32 registers and rounds **once**
   at dequant. The asym kernel rounds per K-block (int32 → fp32 · scales)
   and then chains fp32 `TMA_REDUCE_ADD`s — `ceil(K/BLOCK_K)` roundings
   plus non-associative fp32 adds. `tests/test_sm90_int8.py` already
   observes diff = 0.0 vs the int32 reference on current cases
   (`DIFF_TOL = 0.05` is slack); the deep variant preserves that exactly.
   Parity risk concentrates in the *fused* kernel's plumbing, not in
   numerics.
3. **Scale-once-per-tile is exact by contract.** The runtime's scales are
   per-row (A) / per-channel (B), constant along K (`runtime.py:20-30`
   docstring — the per-K-block SFB layout is a broadcast). Applying
   `sfa[r]·sfb[c]` once after full-K accumulation is algebraically
   identical to the asym kernel's per-K-block application.
4. **Known trap: INT8 WGMMA is K-major-only.** The BF16 copy templates
   `kMajorA/kMajorB` with MN-major smem descriptor paths; SM90 s8 WGMMA
   has no MN-major operand form. The port must `DG_STATIC_ASSERT` both
   majors to K and delete the MN branches (weights `[N, K]` row-major and
   activations `[M, K]` are already K-major — no data change, just a
   template-space trap that would fail at JIT time if left reachable).
   Same reasoning kills the `kDoMergeStages` path (Normal-GEMM-only,
   NT-major-only) — delete it.
5. **Layout sharing is safe; segment math must be shared code.** Both
   sides read `offsets` pairs / `experts` ids produced by
   `_build_layout`; the deep side's `m_start/m_end` must reuse the exact
   `ceil_div(offset, BLOCK_M)` semantics of `asymScheduler.cuh:116-117`
   (lift into a shared helper so the two schedulers cannot drift). One
   subtlety: layout pads segments to 256 rows, but `offsets` can carry
   the *true* end row — then a deep side running BLOCK_M=64 tiles only
   `ceil(true_m/64)` blocks. Correct by construction (pad rows are
   zero-filled and their outputs unread), and 4× less pad compute at
   decode.
6. **Cross-side writes are disjoint.** Placement is per-expert, segments
   are per-expert row ranges → the hbm side's plain `TMA_STORE`s and the
   asym side's STORE-then-REDUCE_ADD protocol never touch the same rows.
   The asym side's RMW ordering is CTA-internal (same CTA issues k=0
   store then k>0 reduce-adds in stream order) — unchanged from today.
7. **v1 multicast-off makes the steal path tractable.** With
   `kNumMulticast = 1` every barrier is CTA-local (no cluster arrivals,
   `is_peer_cta_alive` moot), so a side-switching CTA quiesces with
   `tma_store_wait<0>` + `__syncthreads()` and re-initializes barriers it
   alone owns. The dangerous cross-CTA variant only appears if v2
   re-enables clusters — keep stealing and clusters mutually exclusive
   until proven.
8. **Degenerate splits must equal the parents.** `s_host = 0` (all HBM)
   and `s_host = gridDim.x` (all host) exercise exactly one branch;
   parity tests pin both against the standalone kernels — this isolates
   fusion bugs from pipeline bugs.
9. **CUDA-graph capture freezes `s_host`** if passed as a kernel scalar.
   Pass it via a 4-byte device buffer read at kernel start instead:
   placement-driven rebalance then works under replay (contents-only
   update, same trick as `cached_slot`). Descriptor count is fine: 7 TMA
   descriptors × 128 B = 896 B of `__grid_constant__` params.

Net: Phase A numerics are lower-risk than the existing asym kernel;
fusion risk is concentrated in (7) and in smem/barrier overlay
bookkeeping, both covered by forced-split parity gates.

## 10. Performance estimate

Model-based, using the repo's own measured/fitted rates
(`_RATES_PRIOR`, `runtime.py` comments): PCIe copy-engine staging
52.4 GB/s, in-kernel TMA over PCIe 21.5 GB/s, asym-kernel-on-HBM
320 GB/s, effective WGMMA INT8 251.7 TOPS, GPU launch-side intercept
~150 µs, CPU 209.7 GB/s + 1.57 TOPS. Reference shape: the synthetic
sweep (G=32, H=1024, I=2048, top_k=4, H20 ≈ 78 SMs); per-expert weights
`3·H·I` = 6.29 MB (201 MB/layer), work per routed row `6·H·I` =
12.6 MFLOP.

**Phase A (deep INT8 kernel on HBM-resident experts): the big win, ~3-4×
at decode.**

- Decode, T=64, all 32 experts HBM-resident: asym kernel streams B at
  320 GB/s → 201 MB = 0.63 ms, plus 256-row-padded compute (8192 padded
  rows → 103 GFLOP → 0.41 ms, partially hidden). Deep kernel: weights at
  an expected 1.5-2.5 TB/s (standard multi-stage pipeline; H20 HBM peak
  4 TB/s) → ~0.10 ms, compute at 64-row padding (25.8 GFLOP) → ~0.10 ms.
  **≈ 0.65 ms → ≈ 0.15-0.20 ms** for the cached-partition GEMM.
- Prefill, T=4096, full cache: both kernels go compute-bound
  (206 GFLOP → 0.82 ms at 251.7 TOPS); deep removes the single-slot-B
  serialization and the partial-sum RMW traffic → **~1.1-1.3×**.
- The 320 GB/s → multi-TB/s claim is the load-bearing assumption:
  **validate first** with the Phase A kernel standalone before any fusion
  work (microbench V1 below).

**Fused kernel on PCIe hosts (H20): honest accounting says the win is
launch collapse + overlap, not transport.** Copy-engine staging
(52.4 GB/s) beats in-kernel TMA (21.5, maybe 35-45 GB/s after the B
double-buffer) and both share the same physical link — so when the link
is the constraint, staging remains the right transport and the fused
kernel's asym side covers ring overflow / staging-off configs only.
What the fusion actually buys on PCIe:

- **Launch/layout collapse:** 2-3 launches + per-partition layout builds
  → 1. Worth ~50-90 µs of the fitted 150 µs GPU intercept per layer per
  forward — at decode with ~1 ms MoE layers that is 5-10% per layer, and
  it directly shrinks the `b0` serial-enqueue penalty that today biases
  the CPU/GPU partition away from mixing at decode.
- **SM overlap:** staged-partition GEMM reads (0.16 ms at 320 GB/s today,
  ~25 µs after Phase A for a 50 MB host bucket) run concurrently with the
  hbm side instead of serially after it.
- Mixed prefill example (24 HBM / 8 host, T=4096): today ≈ cached GEMM
  0.62 ms serial-before-streamed + staged copy 0.96 ms (side stream) +
  staged GEMM ≈ **1.3-1.6 ms**; Phase A + fused ≈ max(staged 0.96 ms,
  hbm compute 0.62 ms) ≈ **~1.0 ms** → **1.3-1.5×**.

**Fused kernel on NVLink-C2C hosts (GH200/GB200): this is where the
design pays.** At 450-900 GB/s C2C, in-kernel TMA streaming beats
copy-engine staging (no ring allocation, no double copy, no event
choreography): an 8-expert host bucket (50 MB) streams in ~0.11 ms at
450 GB/s, fully parallel with the hbm side. The fused kernel is the
natural (and only sensible) architecture there; the PCIe H20 path is the
conservative fallback mode of the same binary.

**SM split arithmetic.** The asym side is link-bound: today's grid
already puts ≥16 concurrent CTAs on the link for 21.5 GB/s, so
`s_host ≈ 8-16` (10-20% of 78 SMs) saturates it; the hbm side keeps
80-90% of the machine's TOPS. Stealing covers the tail: at decode the
hbm side finishes in ~0.1 ms while the asym side runs ~1 ms+ — idle-HBM
CTAs have nothing to steal *from* the link-bound side by design, which is
why stealing is one-directional (asym→hbm only, after the host list
drains).

**B double-buffer smem check (C4).** At BLOCK_N=128, BLOCK_K=512:
2×64 KB B slots + 64 KB CD ring + 2×32 KB A stages ≈ 258 KB > 227 KB —
doesn't fit. Drop to BLOCK_K=256: 2×32 KB B + 64 KB CD + 3×16 KB A +
SF/barriers ≈ 180 KB ✓. Twice the K-blocks doubles the partial-sum RMW
and SFB traffic (HBM-side cost, cheap) in exchange for hiding PCIe
latency — expected +50-80% on the 21.5 GB/s in-kernel rate, i.e.
~32-39 GB/s: still below staging on PCIe (confirming C4's "measure, keep
staging if it wins"), but it compounds on C2C. Union smem for the fused
kernel: hbm side at BM=128/BN=128/BK=128 = 64 KB fp32 D + 5×32 KB stages
= 224 KB ✓ (tight; BM=64 gives 7 stages), asym side ≈ 180-200 KB ✓.

**CPU bucket sanity (unchanged executor).** Per the priors, a CPU expert
costs ~30 µs (weights from DRAM) + ~8 µs/row — competitive only at
m ≲ 16-32, exactly the existing crossover. Hybrid doesn't change this;
it changes what the GPU side of the makespan looks like (smaller, so the
solver will hand the CPU slightly less work at decode).

**Validation microbenches — run these before committing to each phase:**

- **V1 (gates Phase A):** standalone deep-INT8 kernel HBM weight-stream
  rate on grouped shapes; target ≥ 1.5 TB/s. If it lands < 800 GB/s,
  Phase A's decode win halves and the plan should pause for a config
  search (stages/BLOCK_K) before proceeding.
- **V2 (gates C4):** toy kernel, 2-slot B TMA-over-PCIe pinned reads;
  measures the real double-buffer recovery vs the 21.5 GB/s baseline.
- **V3 (gates the transport decision):** concurrent copy-engine H2D +
  in-kernel TMA on the same link — how the ~52 GB/s budget splits, i.e.
  whether mixed transport ever beats pure staging on PCIe.
- **V4 (gates Phase B's decode claim):** measure the actual multi-launch
  overhead by timing today's 3-partition GPU bucket vs the same work
  forced into 1 partition — bounds the launch-collapse saving before any
  fusion code is written.

**Summary table (synthetic shape, model-based estimates):**

| scenario | today | + Phase A | + fused (B/C) | driver |
|---|---|---|---|---|
| decode T=64, full HBM residency | ~0.65 ms | ~0.15-0.2 ms | ~0.1-0.15 ms | HBM kernel 3-4×; launch collapse |
| decode T=64, 24 HBM / 8 host (PCIe) | ~1.1 ms | ~1.0 ms | ~0.9-1.0 ms | staged copy 0.96 ms binds (link) |
| prefill T=4096, full HBM | ~0.9 ms | ~0.8 ms | ~0.8 ms | compute-bound both |
| prefill T=4096, 24/8 (PCIe) | ~1.3-1.6 ms | ~1.1 ms | ~1.0 ms | overlap + Phase A |
| decode/prefill mixed on C2C (GH200) | n/a (staging-shaped) | — | link ~10× PCIe | in-kernel streaming wins outright |

Expected end-to-end (48-layer model, decode): Phase A + launch collapse
alone ≈ 0.5 ms × 48 ≈ 20-25 ms/token saved when residency is high — the
fused kernel's incremental PCIe win is real but second-order; its
first-order case is C2C platforms and staging-constrained configs.

### 10.1 Measured validation (H200 + 2×8457C, PCIe Gen5 x16)

Microbenches run on the live box (`scratchpad/hybrid_microbench.py`
pattern: G=64, N=2048, K=1024, 134 MB weights/pass — beats the 50 MB L2;
correctness suite `tests/test_sm90_int8.py` all green with
**diff = 0.00000** on every case including pinned-B and REDUCE_ADD paths,
empirically confirming §9.1-9.2):

| measurement | result | verdict on estimate |
|---|---|---|
| HBM D2D ceiling | 4,271 GB/s | — |
| asym kernel, B=HBM, decode m=64 | 0.185 ms → **726 GB/s** weight stream | baseline better than the 320 GB/s H20-era figure, but still 6× under HBM ceiling → Phase A headroom confirmed |
| asym kernel, B=HBM, prefill m=512 | **184 TFLOPS** effective | far below ceiling → Phase A prefill win **revised up** |
| cuBLAS INT8 fused M=16384 (deep-pipeline compute proxy) | **590 TFLOPS** | Phase A target: 400-550 TFLOPS at prefill, 2-3 TB/s at decode |
| cuBLAS INT8 per-expert loop, decode | 0.563 ms (8.8 µs/launch) | 3× *slower* than the grouped asym kernel — per-expert launches are unviable at decode; validates the single-launch grouped design |
| asym kernel, B=pinned (in-kernel TMA) | **23.5-24.0 GB/s** | confirms the 21.5 GB/s figure (V2 baseline; Gen5 box) |
| copy-engine H2D pinned | **48.3 GB/s** | confirms ~50 GB/s |
| concurrent in-kernel TMA + copy-engine H2D | kernel 5.73→8.38 ms; **32 GB/s combined** vs 48 pure | **V3 answered: mixed transport LOSES on PCIe.** The transports contend, they don't add. Asym side on PCIe = staging-off/ring-overflow fallback only |
| tiny grouped launch | device 24.6 µs + host enqueue 23.3 µs | V4 answered: partition collapse saves ~50-100 µs/layer (2-3 partitions), matching the 150 µs `b0` decomposition |

Revisions to the model-based table above:

- **Phase A decode ≈ 3-4× confirmed** (726 GB/s baseline vs 2-3 TB/s
  target → 0.185 ms → ~0.05-0.07 ms for the 134 MB pass).
- **Phase A prefill revised from 1.1-1.3× to ~2-3×**: the asym kernel is
  compute-limited at 184 TFLOPS (single-slot B stalls + RMW), while the
  same silicon does 590 TFLOPS on a well-pipelined INT8 GEMM. The earlier
  estimate wrongly assumed both kernels share the 251.7 TOPS effective
  prior — that prior *is* the asym kernel's inefficiency, fitted on H20.
- **The C4 double-buffer experiment loses priority on PCIe**: even at
  +80% (24 → 43 GB/s) it stays under the copy engine, and V3 shows the
  two can't productively share the link anyway. C4 is now a C2C-platform
  work item.
- Remaining unvalidated: the Phase A kernel's actual streaming rate
  (V1 proper — needs the kernel to exist; cuBLAS bounds it from above)
  and C2C rates (needs a GH200/GB200 box).

### 10.2 Phase A: implemented and measured (V1 answered)

Phase A is now **implemented**:
`asym_gemm/include/asym_gemm/impls/sm90_int8_gemm.cuh` (the deep-pattern
kernel: persistent 1D grid, per-CTA modular stride over
(segment, m_block, n_block) tiles from the asym offsets/experts layout,
kNumStages-deep A+B pipeline, S32 WGMMA per K-block + FFMA promotion into
fp32 register accumulators, global-load scales, plain TMA_STORE),
launcher `csrc/jit_kernels/impls/sm90_int8_gemm.hpp`, facade
`m_grouped_int8_gemm_nt_contiguous` in `csrc/apis/gemm.hpp`, tests
`tests/test_sm90_int8_deep.py`.

One design revision vs §3.4/§3.5 discovered by the parity suite: the
general INT8 API contract has **per-K-block scales** (GRAN_K=128), not
K-constant ones — full-K int32 accumulation with one dequant is only
valid for the unified_moe runtime's broadcast scales. The kernel now does
per-K-block FFMA promotion (the DeepGEMM FP8 pattern): each K-block is
still integer-exact, the cross-block sum lives in fp32 *registers*
(vs the asym kernel's fp32 TMA_REDUCE_ADD through HBM). Parity:
**diff = 0.00000** on all cases (incl. permuted expert-id segments and
cross-kernel vs asym).

Measured on the H200 box (G=64, N=2048, K=1024, 134 MB weights/pass):

| shape | deep kernel | asym kernel (HBM) | speedup |
|---|---|---|---|
| decode m=64 | 0.084 ms — **1,607 GB/s**, 206 TFLOPS | 0.186 ms — 722 GB/s | **2.23×** |
| prefill m=512 | 0.407 ms — **337 TFLOPS** | 0.754 ms — 182 TFLOPS | **1.85×** |

V1 gate (≥ 1.5 TB/s streaming) met. Remaining headroom vs the 590 TFLOPS
cuBLAS ceiling and 4.3 TB/s HBM ceiling is the v2 config work: BLOCK_M=128
(two math warp-groups), TMA multicast on A, wider BLOCK_N. Next step per
§4: switch the runtime's cached partition to this kernel behind
`ASYMGEMM_HBM_KERNEL=deep|asym`, then Phase B fusion.

### 10.3 Gate A + Phase B: implemented and measured

**Gate A (runtime integration) is closed.** The cached partition's grouped
GEMMs (`_gpu_grouped_forward`, `kind == 'cached'` — fused gate+up AND down)
route through `_hbm_grouped_gemm()`: `ASYMGEMM_HBM_KERNEL=deep` (default) |
`asym`, falling back to asym when the deep facade is absent. Streamed
(pinned-host) partitions stay on the asym kernel — the M-outer loop re-reads
B per m-block, ruinous over PCIe. Parity (H200, G=8 with 4 cached, mixed
routing): deep-vs-asym scale_rel 5.7e-6 end-to-end, both 8.7e-5 vs the
all-CPU reference; the default no-cache path is regression-free on the full
`test_unified_moe.py` suite. `_cached_gpu_decode` / `_cached_gpu_forward_any`
migrate in Phase D as planned.

**Phase B (fused kernel) is implemented** per §4:
`asym_gemm/include/asym_gemm/impls/sm90_int8_hybrid_gemm.cuh` — one
`__global__` (`sm90_int8_hybrid_gemm_impl`), CTA ranks `< s_host` run the
asym K-outer pipeline as a `__noinline__` side function wrapped in a
persistent item loop (`i = rank + s_host·iter` over (segment, n-block)
items, via the explicit-ids `asymScheduler` ctor), remaining ranks run the
Phase A deep pipeline re-based onto a runtime peer count. `s_host` is a
runtime kernel argument (no re-JIT per split; launcher clamps it against
empty sides). Union smem: shared 2-stage CD ring + per-side overlay; hbm
side gets 6 A+B stages under the 227 KB budget. Multicast off, no stealing
(v1). Launcher `csrc/jit_kernels/impls/sm90_int8_hybrid_gemm.hpp`, facade
`m_grouped_int8_hybrid_gemm_nt_contiguous` in `csrc/apis/gemm.hpp`, tests
`tests/test_sm90_int8_hybrid.py`.

**Exit gate B is met** (H200, 132 SMs):

- Parity: diff = 0.00000 vs the float reference on all mixed cases
  (interleaved host/HBM segments, cuda AND pinned host B), invariant across
  `s_host` ∈ {1, 8, 66}. Degenerate splits match the standalone parents to
  diff = 0.0000000 (§9.8: isolates fusion bugs from pipeline bugs — none).
- Register pressure (§7 risk): REG:248 STACK:0 LOCAL:0 — the `__noinline__`
  per-side split holds; no spills.
- Bench vs two-launch (deep-on-HBM + asym-on-pinned, host=8/hbm=24,
  N=2048, K=1024, s_host=16): **1.22×** at decode (m/seg=64, 0.767 →
  0.631 ms) and **1.31×** at prefill (m/seg=256, 0.811 → 0.617 ms) — the
  launch collapse + SM overlap win, on PCIe where §10 predicted the win
  would be smallest.

**Phase C2 (device-side stealing) is implemented and measured.** With
`enable_steal` (a runtime kernel argument; the launcher allocates and
zero-fills the ticket counter per launch), the deep side's tile enumeration
switches from static rank-striding to popping a device-global atomic
counter, and asym-side CTAs that drain the host item list quiesce and join
the same pop loop — one-directional, exactly per §5 C2. Mechanics: the TMA
warp's elected thread pops + decodes each ticket and publishes
(m_blk, n_blk, eid) through a 2-slot smem mailbox ring with its own
mbarrier pairs (so math warps consume tiles in producer order and the A/B
stage walk stays lock-step), terminating with a sentinel. The side-switch
quiesce — the §7 "highest-risk 10 lines" — is: `__syncthreads` (all warps
out of the host pipeline), `tma_store_wait<0>` (CD ring drained; the deep
epilogue only waits to depth 1), `mbarrier.inval` of the host-side barriers
(their addresses fall inside the deep side's A/B stage region; TMA-writing
a valid mbarrier is UB per PTX), then the hbm side's own init/fence/sync
gates the rest. Measured (H200):

- Parity diff = 0.00000 on all steal cases, including `s_host = 131` of
  132 — one native deep CTA, ~everything computed by stealing CTAs.
- Reclaim (prefill m/seg=256, host=2 pinned / hbm=30): well-balanced
  `s_host=16` 0.171 ms; mispredicted `s_host=99` 0.350 ms; mispredicted
  **with stealing 0.172 ms — 2.03×, the full penalty recovered**. (At
  decode the deep side is HBM-BW-bound and even ¼ of the machine saturates
  DRAM, so a bad split costs ~nothing there — stealing's payoff is
  compute-bound shapes.)
- REG:218 STACK:0 LOCAL:0 — still no spills with both enumeration modes
  compiled in.

Remaining, in plan order: Phase C1/C3 (modeled `s_host` + per-side
`%globaltimer` observability feeding the dispatch-model refit), Phase D
(runtime hybrid launch behind `ASYMGEMM_HYBRID_KERNEL=1` with
`ASYMGEMM_HYBRID_STEAL=1` gating steal after soak, collapsing the
cached/staged partition loop, graph-capture validation with `s_host` moved
into a 4-byte device buffer per §9.9).
