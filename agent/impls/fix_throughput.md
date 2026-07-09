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
- **Strictly serial — one experiment on the whole NODE at a time** (not just one per
  GPU): host RAM bandwidth, CPU cores (adam/offload workers), and C2C paths are shared,
  so even different-GPU runs contaminate each other's latency. 30 s settle between runs.
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

Ranked fix list (C0–C5; expected s/step recovery @ q3-30b 32k×8 in brackets; gap to
close: 111.7 → 66.6 s post-C0):

C0. **[DONE 2026-07-09, validated] Kill the synchronous per-param grad-D2H hooks** —
   v3 A/B (q3-30b @32k×8): hook block 30.4 → **0.033 s/step**; step 117.0 → **111.7 s**
   (+4.7% tok/s, within 2.1 s of the no-cpu-adam control floor 109.6 s). Implementation:
   bf16 pinned staging for true async D2H + CPU-side fp32 widening at a single per-step
   drain + `step()` now self-clears `grad_buffer_has_data` (also fixes the cross-step
   grad ACCUMULATION correctness bug — zero_grad never reached this optimizer under LF).
   Env kill-switch: `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0`.
C1. **GEMM engine tax — the real #1, both models** [10–20 s; llama: most of its 3.7×]:
   (a) kernel work: ncu `sm100_bf16_asym_gemm_impl` in isolation
   (`scripts/lora/profile_ncu_asymgemm.py`); tune tiles/block sizes (`DG_BF16_BLOCK_M`,
   cpu-left variants), L2/TMA prefetch depth, occupancy (agent/notes.md engineering
   list). Uncertain ceiling — treat ⅓→½–⅔ peak as the realistic range.
   (b) **hybrid dispatch with STAGED-native as the default at training M**: per-GEMM-site
   choice {stage-once + stock nvjet | asym-stream}. Staging is hidden at SFT shapes
   (1.2 GB/layer ≈ 6 ms @200 GB/s vs 100s-of-ms layer compute; double-buffer ≈ 2–3 GB
   HBM TOTAL — works at every s incl. ceiling, per-expert M ≥ ~10k rows even @173k).
   Residency-cache variant (pin panels in headroom) is the mid-seq special case;
   asym-stream stays the fallback for small-M/zero-headroom. gb200_story T3/B3 math.
   (c) **W-panel reuse across recompute→dX** within one layer backward: stream/stage the
   gate/up panel once, reuse for the dX GEMM (weight reads 2.0×→1.0×; agent/notes.md).
C2. **fg kernel-work diet** [5–8 s]: `ASYMM_QWEN3_MOE_ROUTE_LORA=1` A/B (kills the
   2.27 s/step token-space LoRA fill/index/scatter measured in fwd alone); skip padded
   M=0 expert-group launches; trim MLP wrapper-scope kernels (12.5 vs so 3.6 s bwd);
   per-model ker bits (101 wrong on qwen3.5 — profile, don't hardcode).
C3. **τ discipline: async unpack + prefetch** [5–8 s @32k; ~tens of s at long seq where
   the ~80 s attention-boundary waits live]: decoder/linear-attn `_unpack` are host-
   blocking `ready_event.synchronize()` (attention wrapper already async — unify);
   `stage_concat_columns` is synchronous; prefetch/double-buffer offloaded acts + GC
   roots one layer ahead on the existing restage side stream (LIFO order known;
   `weight_offload.py` prefetch stream allocated, unused; reuse the
   `base_weight_pager.py` 2×-in-flight pattern).
C4. **Within-layer round-trip short-circuit** [8–20 s @32k — A/B decides: the transfer
   itself is overlapped (memcpy is a wash vs so), so the win is sync/stage waits + pool
   overhead + C2C slack, growing with s; the mechanized form of old "adaptive offload vs
   seq"]: in recomp-off-full-fg the outer fwd is the no-grad pure-GPU path — X/gate/up/
   act are offloaded during the *backward recompute* and fetched back within the SAME
   layer's backward. Transient to keep them in HBM ≈ 18 GB @32k, linear in s (~73 GB
   @131k) → config-gated by a per-model threshold s* from the linear memory model: below
   s* skip the round trip (fg code path kept), above s* today's behavior. Ceilings
   preserved by construction.
C5. **Adam overlap depth** (H2, small): bucket sizing, adam threads pinned to CPU-node
   cores (`ASYM_CPU_ADAMW_STEP_THREADS`).

Non-lever (documented so nobody re-derives it): **batch size**. T(B) = c_fix + c_var·B;
only c_fix (~10–15% of step: t0, per-layer orchestration, per-pass weight bytes)
amortizes with B — cannot close a 1.59× per-token gap. Keep B at the knee (per-expert
rows ≥ ~4–8k; b8 suffices at s ≥ 32k); B's value is capacity (maxB/ceiling rows), not
tok/s.

Expected landing @32k: C0 done (111.7) + C2/C3/C4 ≈ 85–100 s (the Phase D A/Bs decide
the exact split — the per-item brackets overlap); **strict parity with so's 66.6 requires
C1b (staged-native dispatch) or C1a reaching near-native** — the streamed-kernel gap is
the irreducible remainder. Re-validate each fix with the Phase A protocol and re-run
`python3 scripts/lf/ceiling_table.py scripts/lf/ceiling_table_configs.txt`.

**Success criteria**
- Backward stall fraction < 20% (from ~63%) on q3-30b @32k.
- asym tok/s ≥ superoffload unsloth-off at equal (s=32k, B=8) for q3-30b; within 15%
  for the dense models — while ceilings stay within 1k of current confirmed values
  (re-run ceiling search after any memory-relevant change; fingerprints will move).
- Timing protocol upgrade (owner request): ≥5 total steps; drop warmup + FIRST measured
  + last measured; NVMe (`-ceil`/`_actnvme`) stays out of scope for this workstream.

---

## Phase D — Diagnose-first, flag-gated A/B execution plan (the order to actually do it)

Principle (from the C0 lesson): **no fix is believed until (a) a pre-build diagnostic
pins the culprit from artifacts or an isolated bench, and (b) a one-flag A/B on the
pinned baseline measures it.** Every fix ships behind an env flag whose default is
today's behavior; the default flips only after gate + loss parity + (if memory-relevant)
ceiling re-search.

### D0. Standing rules for every stage

- Pinned baseline row: `RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false"`
  (primary point 32k×8; long-seq confirmation point 131k×8 = so's ceiling seq).
- Timing: ≥5 total steps (`MAX_STEPS=5 WARMUP_STEPS=1`), drop warmup + FIRST measured +
  LAST measured; strictly serial on the node; healthy-margin rule; `PROFILERS=source`
  (nsys only where the stage needs a timeline).
- ONE flag flips per A/B. After each accepted stage, re-run the **cumulative stack**
  (all accepted flags ON) — interactions are measured, not assumed.
- Loss parity gate for numerics-touching flags (D1 fp32-accum route, D5 staged path):
  loss max/last/train within run-to-run noise of baseline.
- Record every result in the table at the end of this section.

### D1. C2a — ROUTE_LORA A/B  [zero build; flag EXISTS]
- Culprit receipt (already in hand): 2.27 s/step of token-space LoRA fill/index/scatter
  in fwd alone (nsys bin), ×~2–3 in bwd.
- Flag: `ASYMM_QWEN3_MOE_ROUTE_LORA=0|1` (default 0 today).
- Gate: ≥2 s/step @32k + loss parity → flip default for Qwen3-MoE; also decide ker
  label extension (`route101_lora1` already in dir grammar).

### D2. C1 — engine microbench + ncu  [zero build; NO training runs]
- The decisive engine diagnostic, run in isolation: for the flagship's real shapes
  (grouped experts: R≈2.05M rows × [N=768,K=2048]/[2048,768]; dense attention:
  M=256k × [512..4096,2048]), measure the SAME GEMM three ways:
  (i) asym kernel (weights pinned-CPU), (ii) **staged**: H2D copy + stock
  nvjet/`torch._grouped_mm`, (iii) resident nvjet (floor). Plus
  `scripts/lora/profile_ncu_asymgemm.py` on (i) for stall reasons (expect: not enough
  weight tiles in flight to hide C2C latency, not raw-BW-bound at these M).
- Output: per-shape engine-tax table → sizes C1a (kernel tuning) vs C1b (staged
  dispatch) upside BEFORE any integration work. If staged ≈ resident (expected: copy
  6 ms/layer ≪ compute), C1b is confirmed as the parity path.
- Also sweep existing kernel knobs on (i): `DG_BF16_BLOCK_M`, cpu-left variants — the
  free part of C1a.

### D3. C4 — within-layer round-trip short-circuit  [small build]
- Pre-build receipt: `profile.json → activation_offload` per-tag bytes (moe.X/gate/up/
  act) ≈ linear-model prediction (~18 GB/layer-set @32k); nsys stage-wait share.
- Build flag: `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=0|1` (+ dense twin
  `ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM`): when 1, fg skips offload of X/gate/up/act and
  keeps the handles as HBM tensors — sequencing, kernels, everything else identical.
  Default 0 (= today, byte-identical).
- A/B at 32k (transient fits) — do NOT enable near ceiling; the eventual auto rule is
  the s* threshold from the memory model.
- Gate: ≥5 s/step @32k with peak-HBM +≤20 GB → keep flag, wire s* into the launcher.

### D4. C3 — async unpack + prefetch-k  [small build]
- Pre-build receipt: nsys @131k boundary-aligned gaps (~80 s/step family) + the +8 s
  host-gap delta @32k; decoder/lin-attn `_unpack` blocking `synchronize()` in code.
- Build flags: `ASYM_SAVED_TENSOR_ASYNC_UNPACK=0|1` (decoder/linear-attn unpack goes
  event-ordered on the restage side stream, mirroring the attention wrapper) and
  `ASYM_ACT_PREFETCH_LAYERS=0|1|2` (LIFO restage prefetch, reusing the
  base_weight_pager 2×-in-flight pattern). Defaults 0.
- Gate: measured mostly at the 131k point (host gap −50%, boundary gaps gone in
  timeline); @32k expect small (+3–8 s). HBM cost ≤ k × one-layer staged bytes.

### D5. C1b — staged-native dispatch  [the big build; only if D2 confirms]
- Build flag: `ASYM_GEMM_DISPATCH=asym|staged` (+ `ASYM_STAGED_BUFFER_MB`, default
  ~2×largest panel ≈ 2–3 GB), site-class rollout in two steps: dense attention
  projections first (largest per-site tax per D2), then grouped experts.
- Gate: fwd 19.0 → ≤11 s @32k; cumulative step ≤ ~85 s; loss parity; ceilings unchanged
  (re-search — buffers move the fingerprint). C1a kernel tuning continues in parallel
  off D2's ncu findings; per-shape winner becomes the `auto` policy.

### D6. Cumulative close-out
- Full stack (all accepted flags) at {32k, 131k, ceiling−2k} ×8 + the q3-32b and llama
  rows; refresh `ceiling_table.py`; update §2.4 with the final stall budget; declare
  against §3 success criteria.

### Result ledger (fill as stages land)

| stage | flag | Δstep @32k | Δstep @131k | loss parity | ceiling ok | verdict |
|---|---|---|---|---|---|---|
| D1 ROUTE_LORA | `ASYMM_QWEN3_MOE_ROUTE_LORA=1` | . | . | . | n/a | . |
| D2 microbench | (none — bench table) | . | . | n/a | n/a | . |
| D3 keep-acts-HBM | `ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM=1` | . | skip | . | flag-off | . |
| D4 async+prefetch | `ASYM_SAVED_TENSOR_ASYNC_UNPACK=1`, `ASYM_ACT_PREFETCH_LAYERS=k` | . | . | . | . | . |
| D5 staged dispatch | `ASYM_GEMM_DISPATCH=staged` | . | . | . | re-search | . |

---

## 4. Bookkeeping

- Throughput/ceiling tables: `scripts/lf/ceiling_table.py` (auto) → `ceiling_table.md`;
  estimates + manual notes → `ceiling_table_record.md`.
- Existing per-run artifacts: `profiling_both/asym_long_sft_smoke__lora__lf__bf16/ceiling__*/…/b*_s*_ga*/`.
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
