# Fix AsymGEMM LoRA-SFT Throughput

**Goal**: AsymGEMM (`asym_cpuadamwds | recomp-off-full-fg`) wins max-seq by +20–30% over
`superoffload_mem | unsloth-off`, but loses throughput badly (measured: ~1.6× slower per
token for q3-30b, ~2.7× for q3-32b — the latter inflated by a thrash-contaminated anchor).
This doc: (1) re-profile both paths fairly, (2) find the stall, (3) fix it. Target:
**beat superoffload unsloth-off tok/s in defensible situations** while keeping the ceiling win.

---

## 0. What we already know (from existing artifacts — no GPU needed)

### 0.1 The smoking gun: backward-pass stall in the asym path

From `lat.md` of the q3-30b asym confirm (`ceiling__q3-30b-a3b__asym_cpuadamwds__…__b8_s173000…/…source…/b8_s173000_ga1/lat.md`):

| phase | measured | expected (no stall) |
|---|---|---|
| forward | 229.6 s | 229.6 s (reference) |
| backward | **1254.5 s** | ~2× fwd ≈ 460 s |
| optimizer (visible substage) | 2.7 s | — |
| optimizer/update side (e2e − fwd/bwd) | 11.5 s | — |

`recomp-off` does **no recompute** in backward, so backward ≈ 2× forward compute.
Measured backward is **5.5× forward** → **~795 s/step of stall (≈53% of the whole step)
hiding inside backward**. The CPU-Adam *visible* optimizer step is tiny (2.7 s) because
asym overlaps grad-offload + CPU Adam inside backward — so "is it our cpu adam?" is really
"is the backward-embedded offload/adam pipeline stalling the GPU?".

### 0.2 The prize (why this is worth fixing)

Per-token cost after attention-normalization (fitted `c_g`, see `scripts/lf/ceiling_table.py -v`):

- q3-30b asym: `1.65e-4 s/tok` vs superoffload: `1.04e-4 s/tok` (+59%)
- If the backward stall were fully removed: step ≈ 229.6 + 460 + 12 ≈ 700 s
  → `c_g ≈ 8.3e-5 s/tok` → **asym would be ~20% FASTER than superoffload** at equal seq,
  while keeping +32% max-seq. The stall is the whole ballgame.

### 0.3 Measurement contamination to fix in the re-profile

- Both asym anchors were taken AT the ceiling (65k, 173k) → near-wall host pressure
  (q3-32b 65k demonstrably thrashed: 5459 s vs 3013 s at 64k). Anchors must move to
  healthy mid-range points.
- Current `unsloth` (plain) rows in `ceiling_table_record.md` are **estimates** (1.30×
  factor) — replace with measurements.
- Pinned-pool eviction churn is NOT the (only) suspect: the 173k run shows
  `max_cpu_pool_cached_bytes=74.2 GiB` vs `limit=192 GiB` → zero-eviction regime.
  Verify per-run anyway (`profile.json → activation_offload → cpu_pool_evictions`).

---

## 1. Phase A — Fair re-profiling protocol (throughput numbers)

### 1.1 Rules (same for both backends)

- Steady-state rule: `WARMUP_STEPS=1`, `MAX_STEPS=4` → 4 measured; **drop warmup + last
  measured step**; report mean of the middle steps (matches `ceiling_table.py`).
- **`PROFILERS=source` for timing runs** (nsys adds overhead; keep nsys for Phase B only).
  Note this means timing leaves land under `profiling/` output root — pass
  `OUTPUT_ROOT=$PWD/profiling_both` (or extend the table script) so anchors stay discoverable.
- One run per GPU, nothing else co-resident (a concurrent run poisons the numbers —
  learned the hard way).
- Healthy-margin criterion: host `MemAvailable` never within 2× watchdog floor (70 GB)
  during measured steps → else re-run 2–4k lower. Near-wall points get a `thrash` flag,
  not a throughput row.
- `RUN_NAME="ceiling__<model>__<backend>__<recompute-base>"` so `ceiling_table.py`
  auto-discovers anchors.

### 1.2 The grid

Per model (llama3.3-70b, q3-32b, q3-30b-a3b), per backend row below, measure at
`s ∈ {8k, 16k, 32k}` ∪ `{min(48k, healthy), ceiling−2k}` — at **B = maxB(s)** (from
`ceiling_table.md`) plus one **B=8** point at 32k for cross-backend apples-to-apples:

1. `asym_cpuadamwds | recomp-off-full-fg-ker000/101-ceil0000-ohbm<winner>` (winners: llama@3, q3-32b@8, q3-30b@0)
2. `superoffload_mem | unsloth-off-ohbm<winner>` (llama@0, q3-32b@4, q3-30b@0)
3. `superoffload_mem | unsloth-ohbm0` (replaces the estimated rows; ceilings ~43k/47k/78k)

Example row (q3-32b asym @32k, maxB=16):

```bash
GPU_ID=0 PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 OVERWRITE=true \
RUN_NAME="ceiling__q3-32b__asym_cpuadamwds__recomp-off-full-fg-ker000-ceil0000" \
RUNS="q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1 ; 32000|16|1 ; none|false|false|false|false|false" \
bash scripts/lf/profile_lora_lf_test_both.sh
```

~30–45 runs total, most 10–30 min → **1.5–2 GPU-days serial, ~half a day on 4 GPUs**
(runs are independent — pin one model per GPU).

### 1.3 "Can the comparison favor AsymGEMM?" — legitimate framings

1. **Beyond-superoffload-ceiling seqs**: at s > so-ceiling (llama >32k, q3-32b >53k,
   q3-30b >131k) asym is the *only* runner — report tok/s there; superoffload cell = `x`.
2. **Adaptive-offload configs at mid seq**: asym at s ≪ its ceiling has 100s of GB of
   unused headroom — spend it on speed: higher ohbm share, disable fine-grained MLP
   offload (`recomp-off` without `full-fg`), keep attn acts on GPU. That's a *different
   recompute label* (new config row) but the honest product story: "asym auto-tunes
   offload aggressiveness to the requested seq".
3. **Tokens-per-step-hour at max usable context** (capacity×throughput product):
   asym wins whenever its extra context matters.
4. At strictly equal (s, B): only winnable by fixing the stall (Phase C).

---

## 2. Phase B — Stall diagnosis (find the ~800 s)

### 2.1 Zero-GPU first: mine existing artifacts

Per anchor leaf (`b8_s*/` under the ceiling dirs), extract and tabulate:

- `step_samples.csv`: `forward_ms`, `backward_ms`, `optimizer_ms`,
  `heartbeat_dataloader_fetch_ms`, per-step RSS deltas → the fwd/bwd stall table for
  ALL configs (0.1 shows q3-30b only).
- `profile.json → activation_offload`: pool cached/limit/evictions.
- `asym_cpu_adamw.csv`, `cpuadam.csv`: per-bucket adam timings; do adam buckets finish
  before backward needs the next grad slot (overlap efficiency)?
- `nsys_stats_cuda_gpu_mem_time_sum.csv` / `_mem_size_sum.csv` (both-mode runs already
  have these): D2H/H2D volume & time → achieved copy bandwidth vs GB200 C2C peak.
- `nsys_stats_osrt_sum.csv`: `pthread_cond_wait`/`sem_wait` totals → host-side blocking.

### 2.2 Targeted nsys runs (2 per backend, 32k @ B=8, `PROFILERS=both`)

Then decompose with the **existing tool**:

```bash
python3 scripts/lf/analyze_stp_bwd.py <leaf>/trace.sqlite
# → per-device kernel busy vs idle, memcpy class volumes/times, largest gaps (stall receipt)
```

### 2.3 Ranked hypotheses and their falsifiers

| # | Hypothesis | Evidence that confirms | Falsifier |
|---|---|---|---|
| H1 | Fine-grained MLP act fetch (H2D) is synchronous in backward — GPU waits per layer | nsys: repeating [gap → H2D → kernels] pattern per layer in bwd; gap total ≈ H2D time | H2D overlapped under compute in timeline |
| H2 | Grad D2H + CPU-Adam pipeline back-pressures backward (bucket not drained before next grad) | `asym_cpu_adamw.csv` bucket latencies ≥ inter-bucket compute time; osrt waits in bwd | adam bucket time ≪ per-layer bwd compute |
| H3 | PCIe/C2C direction contention: grads (D2H) + act fetch (H2D) + root fetch on same engine/stream | mem_time_sum: low achieved BW while both directions active | BW ≈ peak per direction |
| H4 | GC boundary root save/load sync points (ohbm path) | gaps aligned with layer boundaries, size ≈ root bytes/BW | roots prefetched async |
| H5 | Near-wall host pressure only (measurement artifact) | mid-seq re-anchor shows stall shrink to <15% | stall persists at 32k with 800 GB free |
| H6 | `cudaHostAlloc`/pool churn in hot path | `cpu_pool_evictions > 0`; osrt cudaHostAlloc time | evictions=0 (already true @173k) |
| H7 | Liger-loss / logits chunk sync at step tail | gap at step end in timeline | — |

Deliverable: **a stall budget** — "backward = X s compute + Y s H2D-wait + Z s adam-wait
+ W s other" per config, written back into this doc.

---

## 3. Phase C — Fixes (ranked by expected gain / effort, gated on Phase B)

1. **Prefetch/double-buffer offloaded activations** (kills H1): fetch layer *i−1* acts
   during layer *i* backward on a dedicated H2D stream; pool already reuses pinned bufs.
2. **Deepen grad-offload/adam overlap** (H2): more/smaller buckets, dedicated D2H stream,
   adam worker threads pinned to CPU-node cores (numactl), check fp32-master copy path.
3. **Split copy engines / stream priorities** (H3): D2H grads and H2D acts on separate
   engines; verify with `CUDA_DEVICE_MAX_CONNECTIONS`, per-stream nsys occupancy.
4. **Async root prefetch for ohbm shares** (H4).
5. **Adaptive offload level vs seq** (product-level; independent of stall fix): below a
   seq threshold switch to lighter recompute label automatically → instant throughput
   parity at short seq without losing the long-seq ceiling story.
6. Re-validate after each fix with the Phase A protocol (same slots, same rules), and
   re-run `python3 scripts/lf/ceiling_table.py scripts/lf/ceiling_table_configs.txt`.

**Success criteria**
- Backward stall fraction < 20% (from ~63%) on q3-30b @32k.
- asym tok/s ≥ superoffload unsloth-off at equal (s=32k, B=8) for q3-30b; within 15%
  for the dense models — while ceilings stay within 1k of current confirmed values
  (re-run ceiling search after any memory-relevant change; fingerprints will move).

---

## 4. Bookkeeping

- Throughput/ceiling tables: `scripts/lf/ceiling_table.py` (auto) → `ceiling_table.md`;
  estimates + manual notes → `ceiling_table_record.md`.
- Existing per-run artifacts: `profiling_both/asym_long_sft_smoke__lora__lf__bf16/ceiling__*/…/b*_s*_ga*/`.
- The ~"198 GB buffer": `ASYM_EXPACT_CPU_POOL_MAX_BYTES` ran at 192 GiB (206 GB) in these
  runs — pool, not a hard activation cap; NVMe spill (`-ceil<N>` + `_actnvme`) stayed OFF.
- Don't run two jobs on one GPU; driver lock is currently degraded (deleted-inode flock),
  so *nothing* enforces exclusivity — check `pgrep -f run_lf_lora_sft` before launching.
