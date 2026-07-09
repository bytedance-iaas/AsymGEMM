# Adaptive CPU/GPU dispatch for the unified INT8 MoE layer

**Status:** implemented (v2 — adds shape-derived priors, cross-layer model
sharing, hot-expert row splitting, and row-level D2H gather).
**Code:** `asym_gemm/unified_moe/dispatch_model.py`, hooks in
`asym_gemm/unified_moe/runtime.py`.
**Supersedes:** the static `m_e <= m_cpu` threshold (kept as the default
mode and as the forcing mechanism for tests) and the offline-LUT plan in
`unified_kernel.md` §6.2.

---

## 1. Problem

The unified layer splits each MoE forward into a CPU bucket (AMX INT8,
worker pool) and a GPU bucket (SM90 INT8 grouped WGMMA over pinned host
weights). The static threshold `m_cpu = 16` cannot:

- adapt to the live host/GPU pair (the crossover moves with core count,
  PCIe generation, GPU SKU);
- adapt to live load (CPU thread contention with the serving framework,
  PCIe traffic, clock drift);
- express the *actual* objective. The buckets run **concurrently**, so the
  goal is not "pick the faster backend per expert" but

  ```
  minimize  max(T_cpu(bucket), T_gpu(bucket))        (makespan)
  ```

  Offloading an expert to an idle backend is free even when that backend
  is slower for the expert in isolation — a per-expert threshold can never
  see this; the right split depends on the whole batch's routing histogram.

## 2. Cost models

One linear wall-time model per backend, over one forward's bucket:

```
T_cpu ≈ a0 + a1 · E_cpu + a2 · Σ m_e             (E = experts in bucket)
T_gpu ≈ b0 + b1 · E_gpu + b2 · Σ pad(m_e)        (pad = round up to BLOCK_M=256)
```

The shape mirrors the hardware:

| term | CPU meaning                              | GPU meaning                                   |
|------|-------------------------------------------|-----------------------------------------------|
| ?0   | job launch, gather, numpy plumbing        | quant prekernel + launch + layout build       |
| ?1   | per-expert weight read from DRAM (floor)  | per-expert weight fetch over PCIe (dominant)  |
| ?2   | per-row AMX work                          | per *padded* row of WGMMA                     |

Key asymmetry: `a2 ≫ b2` (CPU cost grows with m much faster) while
`b1 > a1` (GPU carries the big per-expert constant since every expert's
weights cross PCIe). This is what creates a crossover at all.

The GPU model sees **padded** rows because the contiguous grouped layout
pads every expert to a multiple of `BLOCK_M` — padding is real kernel
work, so hiding it from the model would bias the fit.

**Shape-derived priors.** Every coefficient has a physical reading, so the
priors are computed from the layer shape instead of hand-tuned constants:
per-expert cost = weight traffic (3 int8 projections = `3·H·I` bytes) over
a nominal bandwidth (DRAM for CPU, PCIe for GPU); per-row cost = GEMM work
(`6·H·I` MAC-flops) over a nominal throughput (AMX / WGMMA INT8). The
nominal rates (`_RATES_PRIOR`) reproduce the original hand-tuned priors at
the reference shape H=1024, I=2048, so behavior there is unchanged — but
cold-start decisions are now sensible at any shape. Fitted models are
transferable across shapes via `rates()` / `DispatchModel.from_rates()`
(coefficients ↔ hardware rates), which also gives a persistence format.

## 3. Partition solver

Sort active experts by `m_e` ascending; scan all `E+1` prefix→CPU /
suffix→GPU splits with prefix sums; take the split with minimum predicted
makespan. O(E log E) per layer per forward — negligible.

**Serial-enqueue penalty.** Mixed candidates are scored as
`max(T_cpu + b0, T_gpu)` rather than plain `max`: the GPU bucket's launch
work (b0 — quant prekernel, layout build, kernel enqueue) runs on the host
thread *before* the CPU bucket starts, so on mixed splits it delays the
CPU side instead of overlapping it. The per-bucket timings can never teach
the model this (neither bucket observes the other), and without the
penalty the solver over-mixes at decode sizes where b0 dominates both
buckets. b0 also contains some stream-side work, so the penalty slightly
over-charges mixing — erring toward single-backend exactly where mixing's
upside is marginal.

Ascending order is the right family: the benefit/cost ratio of moving an
expert to CPU, `b2·pad(m)/(a2·m)`, is decreasing in `m` (both because
`a2 ≫ b2` and because `pad(m)/m` shrinks), so small-m experts are always
the most profitable CPU residents. All-CPU and all-GPU are members of the
scanned family, so the chosen split is never worse than either baseline
*under the model*.

**Hot-expert row splitting** (`partition_rows`). Whole-expert assignment is
floor-bound by the hottest expert: under skewed routing one expert can
dominate the GPU bucket while the CPU idles. Because both backends read
the *same pinned weight bytes*, one expert's rows can run on both — the
only duplicated cost is a second per-expert weight fetch, which the scan
prices in (the CPU side is charged a full extra expert). Each prefix split
is additionally scored with `c = 1..nblk` `BLOCK_M`-sized chunks of the
largest expert donated to the CPU bucket; the donated row count is chosen
so the GPU remainder exactly fills whole blocks (zero padding waste on the
split expert), and `c = nblk` (full move) also covers non-prefix
"largest expert alone on CPU" assignments. Cost O(E·pad(m_max)/BLOCK_M) —
still negligible. `Layer.forward` sends the first `r` routed rows of the
split expert to the CPU bucket and the remainder to the GPU layout.

## 4. Observations, fitting, hysteresis

- **CPU timing:** `perf_counter` around the worker-pool batch job.
- **GPU timing:** CUDA event pair bracketing the bucket's enqueue,
  **harvested lazily** at a later forward (`event.query()`, no stream
  sync in the hot path). The first GPU bucket call is skipped — it is
  NVRTC/JIT warm-up and would poison the fit.
- **Fit:** ring buffer (64 obs/backend) → ridge-regularized least squares
  toward the priors on column-normalized features (well-posed even when
  all observations share the same expert count), coefficients clamped ≥ 0.
- **Hysteresis:** the windowed refit *is* the anti-flapping mechanism —
  a single noisy observation shifts coefficients by ≤ 1/64 of the window,
  so the split point moves smoothly (cf. `unified_kernel.md` §12.3).
- **Priors** place the single-expert crossover near the old static
  default (`m_e ≈ 16–32`); they only steer the first forwards.

## 5. Execution order change (overlap)

`Layer.forward` previously ran CPU bucket → GPU bucket sequentially. Now:

1. bucket selection (adaptive partition — possibly row-splitting the
   hottest expert — or static threshold);
2. D2H gather of the CPU bucket's activations — *before* the GPU launch,
   because `.to("cpu")` synchronizes the stream. For large batches only
   the bucket's routed rows cross PCIe (deduplicated via `torch.unique`;
   with top_k > 1 a token can hit several CPU experts); small batches
   fall back to one full `[T, H]` copy, which beats the row path's
   several tiny cross-device syncs. Either way the gather is timed and
   charged to the CPU cost model so the partition sees its true cost;
3. **GPU bucket enqueued first** (all ops stream-async, events recorded);
4. CPU bucket runs on the host worker pool — overlapping the GPU stream;
5. join: CPU result `+=`, GPU result `index_add_`.

This makes the makespan objective real: host AMX work now hides under the
GPU kernels instead of serializing in front of them.

## 6. API

```python
layer = Layer.from_bf16(..., adaptive=True)   # or layer.set_adaptive(True)
layer.calibrate()          # optional: forced all-CPU/all-GPU sweeps to seed
layer.dispatch.snapshot()  # fitted coefficients + observation counts
layer.set_m_cpu(k)         # forces static mode (used by parity tests)

# Cross-layer sharing: one model pools observations from every same-shape
# MoE layer (32-layer model → 32× the observations per forward).
shared = DispatchModel(hidden=H, inter=I)
layers = [Layer.from_bf16(..., dispatch_model=shared) for _ in range(n)]

# Cross-shape / cross-process warm start via implied hardware rates.
warm = DispatchModel.from_rates(shared.rates(), hidden=H2, inter=I2)
```

Defaults preserve the old behavior: `adaptive=False`, static threshold 16.
Adaptive mode with a CPU-resident input routes everything to the CPU
bucket (GPU kernels need device tensors). Observations are recorded in
both modes, so a model warmed under static dispatch carries over.

## 7. Limits / future work

- GPU event elapsed time measures the stream segment, so if prior layers'
  work is still queued the observation includes queue wait. Acceptable in
  steady-state decode; a dedicated stream per bucket would isolate it.
- One model per shape; the three projections are aggregated. Fine while
  shapes are fixed per layer. (Cross-layer sharing: pass one
  `dispatch_model` to every same-shape layer — v2.)
- The model only observes backends it dispatches to: once converged to
  all-CPU it never refreshes the GPU fit (and vice versa). A staleness
  probe (occasionally route a minimal bucket to the idle backend, or decay
  toward the shape priors) is future work.
- Row splitting donates rows of the single largest expert only; the
  continuous relaxation says one fractional expert suffices at the
  optimum, but a multi-expert donation scan could shave edge cases.
- CUDA-Graph capture still implies all-GPU (`unified_kernel.md` §12.6);
  the partition honors this if the caller forces `set_m_cpu(0)`.

## 8. Port to main (post-#55 runtime)

The model and solver above are unchanged; the runtime integration differs
from the v2 branch it was developed on:

- **Seam**: replaces the `ASYMGEMM_CPU_PREFILL_FRACTION` fixed-fraction
  split over the *streamed* experts. VRAM-cached experts
  (`ASYMGEMM_GPU_CACHED_EXPERTS`) always stay on the GPU; when the GPU is
  unusable (CPU-resident input) everything — cached included — runs on the
  CPU bucket.
- **GPU timing**: the CUDA event pair brackets only the *streamed*
  partitions (staged/slab halves); the cached partition's HBM GEMMs are a
  different cost regime and stay outside the bracket. Event-waits on the
  staging ring's copies are inside it — that wait is the transfer cost the
  model should see.
- **Row splitting is not wired**: main routes items via a device-side sort
  with one contiguous BLOCK_M-padded segment per expert (`_build_layout`),
  so donating the first r rows of one expert to the CPU needs new layout
  plumbing (reduced count into the GPU layout + diverting
  `seg_start[e]:seg_start[e]+r` into the CPU gather). Until then
  `partition()` (whole experts) is used; `partition_rows` remains for the
  follow-up.
- **Calibrate warm-up skip**: the first forward per (shape, backend) is
  run with `dispatch.paused = True` — per-shape kernel JIT and pool
  warm-up inflate it 5-10x, and one outlier in rings this small wrecks
  the ridge fit (observed: fitted GPU intercept 27 ms vs 5.4 ms real).
- **Measured behavior on main (H20 + AMX host, synthetic G=32 H=1024
  I=2048 top_k=4 sweep)**: no static setting wins across batch sizes
  (`f=0.3` best at small/mid T but 1.7x worse than all-GPU at T=4096);
  adaptive tracks within ~2% of the best static *total* across the sweep
  without hand-tuning and avoids the blow-up cases. The v2 branch's
  1.3-1.6x win over best-static does not reproduce on main because the
  staged/streamed GPU path is ~2.3x faster than the v2 baseline the model
  was measured against — the static heuristic's failure modes shrank.
