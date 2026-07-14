# Fix AsymGEMM LoRA-SFT Throughput

**Goal**: AsymGEMM (`asym_cpuadamwds | recomp-off-full-fg`) wins max-seq by +20–30% over
`superoffload_mem | unsloth-off`, but loses throughput badly (measured: ~1.6× slower per
token for q3-30b, ~2.7× for q3-32b — the latter inflated by a thrash-contaminated anchor).
This doc: (1) re-profile both paths fairly, (2) find the stall, (3) fix it. Target:
**beat superoffload unsloth-off tok/s in defensible situations** while keeping the ceiling win.

---

## 0. What we already know (from existing artifacts — no GPU needed)

### 0.1 Measured phase splits (Phase B.1 mining pass — 2026-07-09, see Appendix A)

Same-seq comparison, q3-30b @131k (per steady step, `step_samples.csv`):

| phase | asym | superoffload | asym/so |
|---|---|---|---|
| forward | 103.9 s | 57.7 s | **1.80×** |
| backward | 699.0 s | 470.4 s | **1.49×** |
| bwd/fwd ratio | 6.7 | 8.2 | — |

Three findings that reframe the problem:

1. **asym is uniformly ~1.5–1.8× slower in BOTH phases** — not one poisoned backward.
   The forward gap (no grads, no adam in forward!) points at the per-token activation
   *offload write* path (fg-MLP + attn act D2H) throttling forward, and its mirror
   (act fetch H2D + grad D2H + adam) throttling backward. CPU-Adam alone is NOT the
   story (visible optimizer substage: 2.7 s/step).
2. **Both backends are far above the ~2× compute-only bwd/fwd ratio** (asym 4.4–6.8×,
   superoffload 6.9–8.7×) — backward is transfer/wait-dominated for *everyone*; even
   superoffload leaves large headroom vs pure compute. The true compute floor needs
   nsys kernel-busy% (Phase B.2) — forward itself may already be transfer-bound.
3. **The near-wall thrash is real and huge but localized (H5 partially confirmed)**:
   q3-32b asym @65k: bwd/fwd **11.6**, stall 75% vs @64k: 5.8, 55%. One grid step
   moved ~700 s/step. But stall% for q3-30b is ~60% already at 131k (76% of its
   ceiling, healthy margin) → **the baseline stall is intrinsic, not thrash**.

### 0.2 The prize (why this is worth fixing)

Per-token cost after attention-normalization (fitted `c_g`, see `scripts/lf/ceiling_estimate.py -v`):

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
  measured step**; report mean of the middle steps (matches `ceiling_estimate.py`).
- **`PROFILERS=source` for timing runs** (nsys adds overhead; keep nsys for Phase B only).
  Note this means timing leaves land under `profiling_results/profiling/` output root — pass
  `OUTPUT_ROOT=$PWD/profiling_results/profiling_both` (or extend the table script) so anchors stay discoverable.
- **Strictly serial — one experiment on the whole NODE at a time** (not just one per
  GPU): host RAM bandwidth, CPU cores (adam/offload workers), and C2C paths are shared,
  so even different-GPU runs contaminate each other's latency. 30 s settle between runs.
- Healthy-margin criterion: host `MemAvailable` never within 2× watchdog floor (70 GB)
  during measured steps → else re-run 2–4k lower. Near-wall points get a `thrash` flag,
  not a throughput row.
- `RUN_NAME="ceiling__<model>__<backend>__<recompute-base>"` so `ceiling_estimate.py`
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

**Preliminary decomposition of EXISTING traces (2026-07-09; caveat: capture windows span
different step counts — shares and per-kernel totals are meaningful, raw windows are not):**

| trace | busy% | idle% | copies | GEMM kernels |
|---|---|---|---|---|
| llama asym @32k | **70%** | 30% (87 s; many ~147 ms periodic gaps) | 17.4 TB @ ~200 GB/s | `asym_gemm_impl` 126 s + `cpu_left` 11 s = **66% of busy** |
| llama so @32k | 36% | **64%** (412 s) | **53.6 TB** | stock `nvjet_*` ≈ 44 s/step (~cuBLAS peak) |
| q3-30b asym @131k | 66% | 34% (315 s) | 20 TB @ ~200 GB/s | flash-bprop 232 s (attn) + asym_gemm 84 s + moe 69 s |

Findings (to confirm with the uniform 4-step runs):

- **F1 — asym is largely KERNEL-bound, not copy-bound**: copies are ~200 GB/s and only
  ~100 s busy; the custom `sm100_bf16_asym_gemm_impl` runs at **≈⅓ of cuBLAS peak**
  (llama: ~137 s/step vs ~43 s ideal for 6·P·T). GEMM inefficiency ≈ 93 s/step +
  idle ≈ 87 s ≈ the entire 500 vs 358 s wall-time gap at llama@32k.
- **F2 — superoffload is the opposite**: 64% idle, 3× the copy volume (it streams
  weights+optimizer+acts wholesale), but stock-peak GEMMs. It wins on kernel speed,
  not on discipline — meaning both sides leave headroom.
- **F3 — asym's idle is periodic small gaps** (~147 ms repeating → per-layer
  orchestration), not a few huge stalls: prefetch/pipelining territory.

### 2.3 Ranked hypotheses and their falsifiers

| # | Hypothesis | Evidence that confirms | Falsifier |
|---|---|---|---|
| H0 | Per-token act-offload traffic throttles BOTH phases (fwd D2H write, bwd H2D fetch) — the 1.5–1.8× uniform gap | nsys: fwd shows [compute ‖ D2H] serialization; copy time ≈ phase gap; kernel-busy% low in both phases | fwd gap explained by kernels alone (busy% high) |
| H1 | Fine-grained MLP act fetch (H2D) is synchronous in backward — GPU waits per layer | nsys: repeating [gap → H2D → kernels] pattern per layer in bwd; gap total ≈ H2D time | H2D overlapped under compute in timeline |
| H2 | Grad D2H + CPU-Adam pipeline back-pressures backward (bucket not drained before next grad) | `asym_cpu_adamw.csv` bucket latencies ≥ inter-bucket compute time; osrt waits in bwd | adam bucket time ≪ per-layer bwd compute |
| H3 | PCIe/C2C direction contention: grads (D2H) + act fetch (H2D) + root fetch on same engine/stream | mem_time_sum: low achieved BW while both directions active | BW ≈ peak per direction |
| H4 | GC boundary root save/load sync points (ohbm path) | gaps aligned with layer boundaries, size ≈ root bytes/BW | roots prefetched async |
| H5 | Near-wall host pressure only (measurement artifact) | mid-seq re-anchor shows stall shrink to <15% | stall persists at 32k with 800 GB free |
| H6 | `cudaHostAlloc`/pool churn in hot path | `cpu_pool_evictions > 0`; osrt cudaHostAlloc time | evictions=0 (already true @173k) |
| H7 | Liger-loss / logits chunk sync at step tail | gap at step end in timeline | — |

Deliverable: **a stall budget** — "backward = X s compute + Y s H2D-wait + Z s adam-wait
+ W s other" per config, written back into this doc.

### 2.4 FINAL stall budget @32k×8 (serial 4-step runs, 2026-07-09 — protocol §1.1)

| run | step | fwd | bwd | tok/s | hook-D2H (`hook_grad_copy_ms`) | GPU idle | busy% |
|---|---|---|---|---|---|---|---|
| q3-30b asym | 117.0 s | 18.9 | 93.1 | 2,189 | **30.4 s/step (26%)** | ~20 s/step | 64% |
| q3-30b so | 66.6 s | 8.1 | 56.9 | 3,845 | — | ~18 s/step | 43% |
| q3-32b asym | 349.5 s | 55.6 | 286.7 | 732 | **80.3 s/step (23%)** | ~52 s/step | 69% |
| q3-32b so | 216.0 s | 25.7 | 187.6 | 1,185 | — | ~63 s/step | 39% |

Verdicts (corrected 2026-07-09 after the §3 control experiments — supersedes the first
reading of this table):
- **Hook-blocked time is wall-time SHADOWED, not critical path**: the no-cpu-adam control
  (109.6 vs 117.0 s) and the v3 async A/B (117.0 → 111.7 s) bound the entire cpu-adam
  machinery at ~5–7 s/step, not 30 s. The host blocks in the hook while the GPU keeps
  draining its queue. Fix #0 kept for correctness + the ~5 s; attribution lesson: a
  wall-time counter is a hypothesis until a one-knob control confirms it.
- **The real gap is GPU kernel time on the asym/fg paths.** nsys semantic decomposition
  @32k (ceiling__*32000* nsys leaves, `lat.md`), per step, asym vs so:
  memcpy **24.2 vs 24.2 s — a WASH** (asym moves acts; so moves ZeRO gathers +
  save_on_cpu; both overlapped); attention flash-bwd 5.85 vs 5.83 s — identical;
  host/runtime gap 22.3 vs 14.2 s (+8); **GEMM/MLP kernel work ~30 vs ~5 s (+25 = the
  gap)**: asym-GEMM bins 13.7 + MLP-wrapper-scope 12.5 + base-dX 3.8 vs so's wrapper 3.6
  + cuBLAS 1.4. Forward confirms it operand-free: 19.0 vs 8.1 s with memcpy < 0.6 s on
  both sides — attention projections ~4.5× (SDPA fprop equal), MLP chain ~4×, incl.
  2.27 s/step of token-space LoRA fill/index/scatter (ROUTE_LORA=0 path) and padded
  M=0 expert-group launches.
- Superoffload itself is 57–61% idle at 32k — fast kernels, heavy demand paging. After
  the C1/C2 engine+fg items below, asym parity at 32k is plausible; at long seq asym's
  relative position already improves (gap 1.76× @32k → 1.53× @131k for q3-30b).

---

## 3. Phase C — Fixes

**2026-07-09 CONTROL RESULT (plain `asym` backend, no CPU-Adam at all, q3-30b @32k):
step = 109.6 s vs 117.0 with cpu_adamwds → the ENTIRE cpu-adam+grad-offload machinery
costs only ~7.4 s/step wall.** The 30.4 s/step of host-blocked hook time overlaps GPU
work and does NOT convert to wall time — the "269 s closes the gap" claim is refuted by
experiment. The real gap (109.6 → 66.6) is the asym GEMM tax + act-offload path.
Ranking below re-ordered accordingly; #0 kept for its CORRECTNESS fix (cross-step grad
accumulation bug), expected wall win only ~≤7 s/step.

1′. **[REAL #1 — both models] asym GEMM engine tax + act-offload forward/backward path**
   (control fwd 19.0 s vs superoffload 8.1 s with no optimizer involved): see item 1
   (ncu + hybrid dispatch) and item 2 (act prefetch) — these are now the whole game.

0. **[DONE 2026-07-09, validated] Kill the synchronous per-param grad-D2H hooks** —
   v3 A/B (q3-30b @32k×8): hook block 30.4 → **0.033 s/step**; step 117.0 → **111.7 s**
   (+4.7% tok/s, within 2.1 s of the no-cpu-adam control floor 109.6 s). Implementation:
   bf16 pinned staging for true async D2H + CPU-side fp32 widening at a single per-step
   drain + `step()` now self-clears `grad_buffer_has_data` (also fixes the cross-step
   grad ACCUMULATION correctness bug — zero_grad never reached this optimizer under LF).
   Env kill-switch: `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0`. Original description follows: —
   `asym_gemm/training/cpu_adam.py:413,417` use `copy_(…, non_blocking=False)` in the
   AccumulateGrad hook: 672 stream-draining syncs/step. Self-measured cost
   (`asym_cpu_adamw.csv → hook_grad_copy_ms`): **269.2 s/step** at q3-30b@131k
   (810 s step) for 13.5 GB moved (~50 MB/s effective; buffers already pinned).
   Fix: async copy on a dedicated D2H stream + CUDA event per param/bucket; the CPU-Adam
   step (and the `staging.add_` accumulate path) waits on the event before reading.
   Expected recovery ~200–250 s/step (upper bound 269 — some drain overlapped real work).
   Zero memory cost. Validate with the 32k A/B protocol.
1. **[dense #1] Attack `sm100_bf16_asym_gemm_impl` efficiency (F1, ~90 s/step on
   llama@32k, ~⅓ of cuBLAS peak)**: (a) ncu the kernel in isolation
   (`scripts/lora/profile_ncu_asymgemm.py`); (b) **hybrid dispatch** — route
   HBM-resident operands to stock cuBLAS/nvjet, keep the asym kernel only for genuinely
   CPU-resident operands (mid-seq headroom makes this free capacity→speed trade).
2. **Shave the periodic ~147 ms per-layer gaps (F3, ~80 s/step)**: prefetch/double-buffer
   offloaded acts + roots one layer ahead on a dedicated H2D stream — reuse the existing
   `base_weight_pager.py` 2×-in-flight pattern (weight-side prefetch exists; act-side
   has none).
3. **Deepen grad-offload/adam overlap** (H2): bucket sizing, dedicated D2H stream,
   adam threads pinned to CPU-node cores.
4. **Adaptive offload level vs seq** (product-level): below a seq threshold switch to
   lighter offload config automatically → throughput parity at short seq without losing
   the long-seq ceiling story. (Pairs naturally with 1b.)
5. Re-validate after each fix with the Phase A protocol (same slots, same rules), and
   re-run `python3 scripts/lf/ceiling_estimate.py scripts/lf/ceiling_search_both.sh`.

**Success criteria**
- Backward stall fraction < 20% (from ~63%) on q3-30b @32k.
- asym tok/s ≥ superoffload unsloth-off at equal (s=32k, B=8) for q3-30b; within 15%
  for the dense models — while ceilings stay within 1k of current confirmed values
  (re-run ceiling search after any memory-relevant change; fingerprints will move).

---

## 4. Bookkeeping

- Throughput/ceiling tables: `scripts/lf/ceiling_estimate.py` (auto) → `ceiling_table.md`;
  estimates + manual notes → `ceiling_table_record.md`.
- Existing per-run artifacts: `profiling_results/profiling_both/asym_long_sft_smoke__lora__lf__bf16/ceiling__*/…/b*_s*_ga*/`.
- The ~"198 GB buffer": `ASYM_EXPACT_CPU_POOL_MAX_BYTES` ran at 192 GiB (206 GB) in these
  runs — pool, not a hard activation cap; NVMe spill (`-ceil<N>` + `_actnvme`) stayed OFF.
- Don't run two jobs on one GPU; driver lock is currently degraded (deleted-inode flock),
  so *nothing* enforces exclusivity — check `pgrep -f run_lf_lora_sft` before launching.

---

## Appendix A — Phase B.1 stall-budget mining (existing artifacts, 2026-07-09)

stall := backward − 2×forward (crude compute baseline; refine with nsys busy% in B.2).
n = steady steps used (drop warmup + last where ≥3 measured; else all measured).

```
config                                           seq  n     fwd      bwd bwd/fwd    stall stall%
----------------------------------------------------------------------------------------------------
llama3_3-70b|asym-ker000-ceil0000             32,000  1    91.7    405.9    4.42    222.4  43.9%
llama3_3-70b|asym-ker000-ceil0000             34,000  1   100.9    486.5    4.82    284.7  47.7%
llama3_3-70b|so|unsloth-off                   30,000  2    41.9    307.1    7.33    223.3  63.3%
llama3_3-70b|so|unsloth-off                   32,000  2    45.0    310.9    6.91    220.9  61.5%
q3-30b-a3b|asym-ker101-ceil0000              131,000  2   103.9    699.0    6.73    491.3  60.6%
q3-30b-a3b|asym-ker101-ceil0000              135,000  2   108.0    735.4    6.81    519.5  61.0%
q3-30b-a3b|asym-ker101-ceil0000              143,000  2   117.2    791.6    6.75    557.1  60.7%
q3-30b-a3b|asym-ker101-ceil0000              159,000  2   155.4   1010.2    6.50    699.4  59.5%
q3-30b-a3b|asym-ker101-ceil0000              167,000  2   219.6   1108.2    5.05    669.0  50.0%
q3-30b-a3b|asym-ker101-ceil0000              171,000  2   194.0   1194.0    6.15    806.0  57.7%
q3-30b-a3b|asym-ker101-ceil0000              173,000  2   224.8   1163.8    5.18    714.2  51.0%
q3-30b-a3b|so|unsloth-off                    128,000  2    56.1    490.6    8.74    378.3  68.8%
q3-30b-a3b|so|unsloth-off                    130,000  2    57.3    484.0    8.44    369.4  67.8%
q3-30b-a3b|so|unsloth-off                    131,000  2    57.7    470.4    8.15    355.1  66.8%
q3-32b|asym-ker000-ceil0000                   64,000  2   122.3    713.7    5.83    469.0  55.5%
q3-32b|asym-ker000-ceil0000                   65,000  2   124.8   1440.7   11.55   1191.1  75.5%
q3-32b|so|unsloth-off                         50,000  2    45.2    364.9    8.08    274.6  66.4%
q3-32b|so|unsloth-off                         52,000  2    47.7    365.8    7.67    270.4  64.9%
q3-32b|so|unsloth-off                         53,000  2    48.9    391.8    8.01    294.0  66.2%
```
