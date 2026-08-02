# Model Capacity — dense ceilings, estimators, and reconfirmation (SELF-CONTAINED)

> **2026-07-25 UPDATE (read §8c first):** (1) §8b's asym 64k/32k results are
> VOID — classifier rc=0 bug + invalid-GQA h=12800 builds that crashed before
> training; §8 @128k survives audit. (2) The "2×-weights steady tax" was
> torch's pinned-allocator pow2 rounding, not physics — fixed by exact-size
> cudaHostRegister homes + pooled exact roots + budget-aware parking
> (−57% host at the 98B anchor). (3) New measured crowns (valid, trained):
> **128k = 263.9B ✓ / 279.2 ✗ (est ~272B)** · **64k = 340.4B ✓ / 355.7 ✗
> (est ~352B)** — vs SO 202.6 (128k) / 227.3 (64k): **+32% / +50%**. Final
> crowns: **128k 267.7B ✓890.0** (X6) · **64k 340.4B ✓870.2** (Y6); walls are
> pure steady physics now (host = 1.863·P + unparked roots + ~33; D1 attrib).
> Headline tables below (§0) are the PRE-audit, PRE-fix numbers.

(Rebuilt 2026-07-22 on s04-p1-dgx-02-c17 after the doc was wiped in the tree
migration; content recovered from the session transcript + memory. This doc is
the single reference for everything model-capacity: metric, protocol, synthetic
models, per-system estimators — baselines AND the asym tiers — final numbers,
and the post-scheduler-merge reconfirmation campaign. The original study ran
2026-07-17→21 on this same machine (c12/c14 were the throughput machines);
artifacts: `profiling_results/profiling_source_capacity_s04-p1-dgx-02-c17/`.)

---

## 0. TL;DR — FINAL NUMBERS @ 128000 tokens × batch 1, one GB200 GPU

Dense track (synthetic dense checkpoints, real released shapes):

| System | Max dense model | What kills it |
|---|---|---|
| so_recomp (= SuperOffload + recompute; FSDP2/ZeRO3 Offload identical) | **53B** (bracket 50.1✓ / 57.2✗) | HBM: per-layer GC checkpoints, G = 1.41 GiB/L → 185 GiB at ~60.6L |
| so_unsloth (SuperOffload + Unsloth-GC) | **200B** as-shipped; **202B** load-fixed est; steady-wall est 199B — the walls COINCIDE | host: load transient 2W+K (K=151) and steady W+roots both hit 905 at ~200B |
| **asym capacity mode** (uns-GC recompute + flush8 + ohbm dial) | **222B est / 215B MEASURED** (as215d5: 112L×12288, ohbm5, peak-unevict 864) | wedged between both walls: 226B → host 910, 252B → host 911 (ohbm4 variant clears host but G 194 > 185) |
| **asym capacity mode 2026-07-25** (exact-pinned + auto-park + flush8, §8c) | **~272B est / 267.7B MEASURED @128k** (X6: 138L×12288, 890.0 peak) · **64k: ~352B est / 340.4B MEASURED** (Y6) | steady wall: host = 1.863·P + unparked roots + ~33 (§8c D1 attrib) |

MoE track (Qwen3-235B-A22B family, layer slices of the real donor):

| System | Max MoE model | What kills it |
|---|---|---|
| so_recomp | 202B as-shipped / **231B load-fixed est** | load wall as-shipped; steady ☠917 @94L scaled |
| so_unsloth | 202B as-shipped / **203B load-fixed est** | LoRA-all optimizer burden ≈2.19 GiB/L is the REAL wall — fixing the loader buys ~nothing |
| **asym** | **249B est / 235B real Qwen3-235B-A22B MEASURED e2e** (loss 1.109→0.973, G 158.7 / C 862) | 250B slice C_OOM 866 (pinned-weight footprint) |

Two-GPU dense estimates (host does NOT scale, only HBM does): recompute trio
105B each; so_unsloth gains ~nothing (host-bound); **asym converts new HBM
into host relief**. `model_capacity.pdf` + paper text now use the TRUE-param
axis (relabeled 2026-07-22, pushed): single-GPU **53/53/53/202/224** (asym
218B measured), two-GPU **105/105/105/207/266**.

**Verdict: asym holds the highest capacity on BOTH tracks under BOTH load-bug
treatments.** The point estimates specifically kill the "SO only loses because
of its 2×-load bug" objection: load-corrected SO still caps at ~202B dense
(its steady wall coincides) and ~203B MoE (optimizer-bound).

**Reconfirmation status (post sched-merge, 3-tier code), 2026-07-22 c17, §8:**
- **True-param axis discovered** (§8): size labels are name-units; index
  `total_size/2` gives d2=97.9B, d5a=**202.6B**, d5e=**217.9B**. Table above
  keeps historic name-axis figures; §8 numbers are true-axis.
- Drift vs pre-merge: **≈zero** (SO@d2 714.6 vs 714; T1@d2 698.8 vs ~700;
  crown 865.4 vs 864).
- **Crown ✓** R11: asym T1-ohbm5+flush8 @ d5e (**217.9B true**) = OK, peak
  865.4 (40 GiB slack).
- **Wall ✓** R12: so_unsloth @ d5a (**202.6B true**) = C_OOM(steady) 905.9 —
  SO ceiling now *measured* (dies past load, mid-step-2). **Measured asym
  advantage: +15.3B (217.9 OK vs 202.6 ☠).**
- **Estimators VALIDATED (campaign closed)**: T2 hold-out @131B **PASS**
  (host +9.9/±25 true-axis, HBM +1.7%/±3%), ceiling ~197B. T3: params-only
  fit first FAILED hold-out (+47 host) → censored kill-peak anchor (trap
  #13); refit on clean anchors then **re-validated on a fresh 125.629B
  build: PASS** (host −13.3, HBM +1.5%); family-local ceiling ~135B. HBM
  ≤1.7% error everywhere; host RSS within ±25 GiB band (noisier, as
  expected). See §8 CAMPAIGN VERDICT.
- New post-merge finding: **T2 @170.8B now PASSES** (862.8) — ~50 GiB leaner
  than the historic offload mode; KEEP_ACTS_HBM recipe is capacity-relevant.

---

## 1. Hardware & metric definitions

- **Node**: GB200 superchip node, 4× GPUs of 185 GiB HBM each; capacity cells
  use ONE GPU. CPU pool = NUMA nodes 0+1 ONLY ≈ **957 GB** (490+490 LPDDR).
  `free`'s ~1.69 TB is a TRAP — the extra ~740 GB is HBM exposed as NUMA
  nodes 2+ via the coherent fabric (see `/home/kevinni/env/agent/project_rules.md` §1).
  No swap. No sudo.
- **Capacity metric (host)** = **peak unevictable host memory** = AnonPages +
  Shmem summed over NUMA 0+1 (`/sys/devices/system/node/node{0,1}/meminfo`).
  On this coherent platform **cudaHostAlloc pinned memory is accounted as
  Shmem** (shared-dirty), so Anon alone misses the pinned set. VmHWM was
  RETIRED as the metric: it is file-cache-contaminated (mmap'd safetensors
  reads inflate it); end-of-run RSS understates (reclaim). Older ledger rows
  quoting VmHWM are marked †.
- **Kill wall = 905 GiB unevictable.** Calibrated from 25-GiB-floor watchdog
  kills: every pass peaked ≤898, every kill ≥906. A cell "fits" iff peak
  unevictable ≤ 905 with the 25 GiB floor never tripped.
- **Watchdog availability** = MemFree + (FilePages − Shmem) over NUMA 0+1;
  kill when < floor (25 GiB for all locked cells; early cells used 35).
- **HBM budget** = 185 GiB (per `torch.cuda` peak reserved).
- **Load vs steady phases matter**: SuperOffload's `from_pretrained` shows a
  **2W + K_load transient (K = 151 GiB)** — DeepSpeed-internal, present even
  with zero.Init active (triple-validated). asym's loader holds **1×W exactly**
  during load (verified at 170–315 GiB scale across 8 runs).

## 2. Protocol (fit probes)

1. **Steps**: exactly **1 warmup + 1 measured step** (capacity fit probes;
   latency runs use 1w+2m — capacity does NOT). Recorded in project_rules.md.
2. **`sync` before every run** — a freshly minted synthetic checkpoint leaves
   ~W GiB of DIRTY page cache, which is unevictable until writeback and killed
   a 252B load (TRAP).
3. **Sampler**: `mem_sampler.sh` (rebuilt in `scripts/lf/capacity/`) samples
   node0+node1 meminfo (+ trainer smaps) every 2 s → trace file. Peak
   unevictable is computed from the trace **windowed to the run** (zombie
   samplers once contaminated traces sharing timestamps — window by time-gap
   or duration cap; kill samplers by args-matched PID loops, never bare pkill).
4. **Watchdog**: kills the cell when avail < floor 25 (that IS the C_OOM
   verdict; OOM-killer races are not acceptable evidence).
5. **Serial only**: one cell at a time per node, ≥30 s settle between cells
   (project_rules §3). Long chains under `timeout`, driven by an orchestrator
   with a supervisor loop so nothing wedges silently.
6. **Classification**: OK / G_OOM (HBM) / C_OOM (host; subtag `load` vs
   `steady`) / UNKNOWN — from trainer log + heartbeat + sampler trace.

## 3. Synthetic models

- **Dense**: random-init on-disk checkpoints (ShardWriter), scaling **width
  and depth together along real released shapes** (Qwen3-dense family
  configs). Never runtime `from_config` — capacity must include the real
  loader path. Generator: `scripts/lf/make_synth_ckpt.py` (dense mode).
  - **GQA shape rule (TRAP)**: heads = h/128 must satisfy **(h/128) % 8 == 0**
    (kv_heads=8). h=12800 → 100 heads → RuntimeError. Valid h ∈ {8192, 9216,
    10240, 11264, 12288, 13312...} — but see the wide-h penalty in §5 before
    extrapolating across h.
- **MoE**: **layer slices of the real Qwen3-235B-A22B donor** (never stack
  30B layers up — router/expert statistics break; never from_config).
  Generator: `make_synth_ckpt.py` slice mode.
- Inventory (`/scratch_local/user_data/shutian/kevin/models_synth/`):
  d0b 43B (48×8192) · d0c 50B (56×8192) · d0 57B (64×8192) · d1 70B (80×8192)
  · d2 100B (88×9216) · d3 130B (96×10240) · d4 170B (104×11264) · d5a 200B
  (104×12288) · d5e 215B (112×12288) · d5c 225B (108×12800 — INVALID GQA) ·
  d5d 226B (100×13312) · d5b 235B (112×13312) · mini 2L×1024 (smoke) ·
  q3-235b-slice-{121b-48L, 250b-100L, 275b-110L, 320b-128L}.
- **NVMe is EXCLUDED from the headline comparison** (Kevin's fairness call:
  baselines could page to NVMe too). asym+panvme extension results are in §9.

## 4. Systems under test

Baselines (DeepSpeed SuperOffload; FSDP2/ZeRO3 Offload hit the identical
recompute HBM wall, so they share so_recomp's capacity):
- `so_recomp` = `superoffload_mem|recomp` — per-layer GC checkpoints in HBM.
- `so_unsloth` = `superoffload_mem|unsloth` — Unsloth-GC checkpoints → host.

asym — **the 3-tier system (post sched-merge, `scripts/lf/tier_recipes.sh`,
single source of truth emitted by `asym_scheduler.py --emit-recipes`)**:

| Tier | recompute token | extra recipe env | character |
|---|---|---|---|
| **T1 (dense)** | `unsloth-ohbm0` | `ASYM_GEMM_DISPATCH=staged` | unsloth-GC recompute on the asym weights-engine — "so_unsloth-like fallback": acts recomputed not stored; host = pinned W (+opt) |
| **T2 (dense)** | `recomp-off-full-fg-ker000-ceil0000-ohbm0` | `ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 ASYM_GEMM_DISPATCH=staged ASYM_SAVED_TENSOR_ASYNC_UNPACK=1` | activation-offload (recomp OFF, full fine-grained), fg acts parked in HBM |
| **T3 (dense)** | `recomp-off-full-fg-ker000-ceil0000-ohbm0` | (none) | deepest offload — everything streams to host |

- **Capacity mode = the T1 family + levers**: `asym_cpuadamwds|unsloth-ohbm<N>[|ligerloss1]`
  + `ASYM_HOST_FLUSH_EVERY_N_LAYERS=8` + `ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0`,
  watchdog floor 25. The **ohbm dial** parks Unsloth-GC checkpoint roots in
  otherwise-idle HBM (ohbm0 = none ... ohbm5 = max parking); it is THE lever
  that converts spare HBM into host relief and the basis of the 2-GPU scaling
  claim.
- **Host-cache flush** (`run_lf_profiled_train.py _install_host_cache_flush_hooks`,
  env `ASYM_HOST_FLUSH_EVERY_N_LAYERS`): forward pre-hooks on decoder layers
  call `torch._C._host_emptyCache()` every N layers (fires during GC recompute
  too ⇒ backward cadence). torch's CachingHostAllocator otherwise hoards every
  freed transient pinned buffer — unevictable; the flush recovered **−317 GiB**
  at 100B (anchor A 842 → anchor C 524.9 steady C, vs SO 696: asym is LEANER).
- Governor-side fixes from this study (all actnvme-gated, inert otherwise):
  attn context saves sealed (`attention_activation_offload.py`), on_seal re-arm
  for CLAIMED/FETCHED (`act_spill_governor.py`), `_empty_torch_host_cache()`
  every 4 seals + every 4 fetches.
- T2/T3 historical caveat: offload modes keep large live act buffers on host —
  at 170B/128k they C_OOM'd ~915 where T1 fit at 862. They are THROUGHPUT
  tiers; T1+dial is the capacity tier. §8 re-measures their walls under the
  merged code to build their estimators anyway (Kevin ask).

## 5. Estimation method (point estimates, no ranges)

y = peak unevictable host (GiB), x = model params P (B) or layers L.

**Physics terms** (used to structure the fit, not free-fit):
- pinned weights W = 1.863 GiB/B-param (bf16 + buffers, measured);
- GC roots (host, ohbm0) = L · seq · h · 2 bytes;
- CPU-Adam optimizer = 16 bytes/trainable (fp32 master+m+v+grad, pinned);
  LORA_TARGET=all on the 128-expert MoE ⇒ 14.06B trainables @250B ⇒ ~185+ GiB
  (symmetric across systems — NOT an asym defect);
- SO load transient = 2W + K_load, **K_load = 151** (matches 121✓/200☠/235☠);
  asym load = 1×W.

**Fit discipline (the part that prevents wrong claims):**
1. **Global linear fits over all anchors are INVALID for SO** — its placement
   is adaptive; residuals 38/319/225 GiB across 57/100/170B are non-monotonic.
2. Use **near-wall local slopes between the two largest same-config,
   same-hidden-family anchors** only. Locked dense slopes: so_unsloth
   **1.88 GiB/B** (from 850@170B); asym T1 ohbm0 **1.76 GiB/B** (from 886@200B;
   3.5 early-range); same-h check at 215B: ohbm0-equiv 930@215 vs 886@200 →
   3.9 GiB/B ≈ physics ✓.
3. **Wide-h host penalty (TRAP)**: h=13312 rungs carry ~+30 GiB unevictable vs
   the same P at narrower h — extrapolate ONLY within a hidden-size family
   (this killed naive 226B/252B predictions).
4. Ceiling = smallest P where predicted peak > 905 (host) or predicted G > 185
   (HBM), taking the max over phases (load, steady): dense so_unsloth load
   2W+151=905 ⇒ 202B, steady ⇒ 199B (coincide → "200B" headline); so_recomp
   G=1.41 GiB/L ⇒ 185 @ 60.6L ⇒ 53B; asym dense ohbm-dial ceiling 222B
   (measured pass 215B; measured fails 226B/252B bracket it).
5. HBM prediction for asym T1: G ≈ engine pools + ohbm-parked roots
   (dial-dependent); at as215d5 predicted host 869 / measured 864 (±5).

**Instrumented anchor table (2026-07-19→21, pre-merge, c17)** — peak
unevictable GiB, identical configs per row:

| model | so_unsloth | asym T1 (uns-GC+flush8) |
|---|---|---|
| 57B dense | 280 | 267 (ohbm0) |
| 100B dense | 714 | ~700 (ohbm0, capped-window) |
| 170B dense | 850 | 833 (ohbm0) |
| 200B dense | ☠ load 906 | **886 OK** (ohbm0) |
| 121B MoE | 647 | **463** (ohbm1) |
| load consts | K_load = 151 (2W+151) | 1×W (verified 8×) |

MoE slopes: asym 8.58 GiB/L (463@48L → ☠910@100L; measured 235✓/250✗);
so_recomp steady ☠917@94L scaled ⇒ 231B load-fixed; so_unsloth optimizer
burden ≈2.19 GiB/L ⇒ 203B regardless of loader.

## 6. Two-GPU & workload-transfer estimates

- **2 GPUs = +185 GiB HBM, +0 host RAM.** Formulation guard (Kevin): do NOT
  claim "SO replicates the model 2×" — ZeRO-3 shards; the correct statement is
  that added GPUs contribute HBM but no host capacity, so capacity scales only
  for systems that can SHIFT host pressure into HBM (the ohbm dial). Dense:
  recompute trio 53→105B; so_unsloth 200→205B (host-bound); asym 222→**264B**.
  TRUE-AXIS relabel (2026-07-22, figure + paper text updated & pushed):
  single-GPU 53/**202**/**224** (SO measured-dead @202.647; asym crown
  217.948 ✓ / wedge ~228 ✗); two-GPU 105/**207**/**266** (same §6 deltas
  +52/+5/+42 on the re-axised anchors).
- Shorter-seq (e.g. 100k×1) estimates follow the same recipe: recompute-root
  and act terms scale ∝ seq; W/opt terms don't. Re-derive from §5 terms rather
  than quoting stale numbers.

## 7. Ledger highlights (recovered; full per-cell logs in the artifacts dir)

- so_recomp: 43B ✓167G/176C · 50B ✓178G/203C · 57B ✗HBM → **cap 53B locked**.
- so_unsloth: 70B ✓78G/330C · 100B ✓88G/696C · 130B ✓97G/765C · 170B
  ✓107G/829C · 200B ✗host@load(2×W) — steady curve flattens 330→696→765→829
  (reclaim absorbs growth; why global fits mislead).
- asym (pre-flush era): uns+ohbm capacity mode 70B ✓34G/516C · 100B ✓39G/891C†
  · 130B ✓43G/917C† · 170B ✓182G/880C (ohbm3, both budgets near-full).
- Mode×capacity matrix @235B real MoE: uns-GC+ohbm1 **OK 862** vs latency-mode
  914.8 ☠ / memory-mode 919.1 ☠ / ohbm2 919.0 ☠ → capacity mode = uns-GC
  recompute; ohbm1 required. 275B synth ☠ 917.6/920.2.
- Flush walk (2026-07-20): fl_d200 dense 200B uns+ohbm0+flush **OK 875.9†** →
  dense goal (asym 200B > so_unsloth 170-measured); fl_m250 ☠909.8 / fl_m275
  ☠920.1 / fl_m320 ☠ (LoRA-all optimizer term found here).
- Estimates round (2026-07-21): §5/§0 tables; confirmations as215d5 **215B
  PASS 864**, as226d5 ☠910, 252B ☠911 (6 GiB past wall exactly as the ~250B
  dial model predicts; ohbm4 clears host but G 194 > 185 — wedged, dial maxed).
- 235B real MoE flagship: **loss 1.109 → 0.973**, G 158.7 / C 862 (uns-GC,
  ohbm1, floor 25) — cell_W5_asym_uns_v8.

## 8. RECONFIRMATION CAMPAIGN (2026-07-22, c17, post sched40×sched42 merge)

**Why**: Kevin updated the runtime substantially (3-tier preset system; source
changes across the six training files). The merge A/B showed **zero HBM effect
and ≤±3% tok/s** on throughput configs (`agent/impls/previous_validation_results.md`
— HBM ±0.0 EXACT on every row; its T1/T2/T3 rows are the tier system working),
but capacity mode (flush + ohbm dial, 128k×1 fit probes) was not in that
matrix. Goals:
1. enough anchors to (re)build the estimators — now PER TIER (T2, T3; T1 is
   the so_unsloth-like fallback and doubles as the capacity mode's base);
2. estimator validity for BOTH peak HBM and peak unevictable RSS (RSS noisier
   — accept ±3% HBM, ±25 GiB RSS on a hold-out size);
3. reconfirm some asym mode still beats both baselines on max dense model
   capacity @128k×1 (expected: T1+ohbm dial ≥215B > SO ~200B).

**Matrix** (dense, 128k×1, serial, sync-guarded, floor 25):

| # | cell | purpose |
|---|---|---|
| R1 | smoke: mini 2L×1024, T1 | plumbing on c17 post-merge |
| R2 | so_unsloth @100B (d2) | baseline drift check vs 714 unevict / 696 steady-C |
| R3 | asym T1+flush8 ohbm0 @100B | drift vs ~700 unevict (525 steady-C) |
| R4/R5 | asym T2 @57B, @100B | T2 estimator anchors (HBM + RSS) |
| R6/R7 | asym T3 @57B, @100B | T3 estimator anchors |
| R8/R9 | asym T2, T3 @170B (d4) | near-wall anchors or measured walls (historically ☠~915) |
| R10 | HOLD-OUT: predict then measure T2+T3 @130B (d3) | estimator correctness gate |
| R11 | asym T1 capacity mode @215B (d5e, ohbm5, flush8) | the crown: reconfirm ≥215B still passes |
| R12 | so_unsloth @200B (d5a) | baseline wall reconfirm (expect ☠ load ~906, or pass ≤905 edge) |

Results are appended below as cells land. Tooling rebuilt (the original
`scripts/lf/capacity/` toolkit did not survive the migration): `enroot_run.sh`,
`mem_sampler.sh`, `run_capacity_cell.sh` (tier-aware), `classify_cell.py`.

### R-results (live)

All cells 128k×1, floor 25, sync-guarded, serial (ledger:
`profiling_results/capacity_reconfirm_c17/ledger.tsv`; traces + logs beside it).

| cell | verdict | peak unevict (GiB) | peak HBM (GiB) | note |
|---|---|---|---|---|
| R1c smoke mini T1 | OK | 5.8 | 0.4 | plumbing ✓ (loss finite 12.01) |
| R2 so_unsloth 100B | **OK 714.6** | | 88.2 | **drift check vs pre-merge 714 → EXACT** |
| R3 asym T1+flush8 100B | **OK 698.8** | | 74.1 | **vs pre-merge ~700 → EXACT**; still leaner than SO |
| R4 T2 57B | OK | 282.1 | 57.5 | T2 anchor |
| R5 T2 100B | OK | 727.1 | 64.0 | T2 anchor |
| R6 T3 57B | OK | 324.2 | 33.0 | T3 anchor |
| R7 T3 100B | OK | 771.3 | 36.5 | T3 anchor |
| R8 T2 170B | **OK** | 862.8 | 78.2 | near-wall PASS — merged T2 (fg-acts→HBM recipe) is ~50 GiB leaner on host than the historic offload mode (☠~915) |
| R9 T3 170B | **C_OOM(steady)** | 924.8 | — | T3 wall reproduced (historic offload-tier ~915–925 ✓) |

**TRUE PARAM AXIS (discovered 2026-07-22, pre-R13).** The historic size
labels are name-units, NOT param counts. Ground truth = safetensors index
`total_size/2` (the trainer's "all params" print is useless — meta/offloaded
params uncounted): d0=**57.251**, d2=**97.891** (not 100.4!),
d3=**130.956**, d3b=**125.629**, d4=**170.765**, d5a=**202.647**,
d5e=**217.948** (not 215!). All §8 fits below are on this axis
(`fit_estimator.py` PARAMS updated); §5's historic estimates remain
name/map-axis and are superseded by §8 where they overlap. Two headline
consequences: the measured crown is **217.9B**, the measured SO kill is at
**202.6B** → demonstrated gap **+15.3B**.

**Post-merge tier estimators (dense, 128k×1, true-param axis)** — anchors
above; tiers are non-adaptive so local fits are legitimate; near-wall
segment used for ceilings:

- **T2**: host near-wall slope (130.956→170.765) = 64.2/39.809 =
  **1.61 GiB/B** → host ceiling ≈ 170.765 + (905−862.8)/1.61 = **~197B**;
  mid slope (97.891→130.956) = 2.16; HBM ~0.18 GiB/B (78.2@170.8B — HBM
  never binds before host). T2 curve flattens toward the wall (282→727→
  799→863), so only the last segment is ceiling-legitimate.
- **T3**: ~~params-only fit through the 170B kill-peak (2.13 GiB/B, ceiling
  ~163B)~~ **REJECTED by R10b hold-out** (see below). The kill-peak (924.8)
  is *censored* — watchdog killed mid-climb, true steady peak is higher —
  so it must not be used as an interpolation anchor. **Refit on uncensored
  near-wall pair (97.891→130.956B): slope = (882.5−771.3)/33.065 =
  **3.36 GiB/B** → host ceiling ≈ 130.956 + (905−882.5)/3.36 = **~138B**.
  Consistency: refit retro-predicts true steady peak @170.765B ≈ 1016 ≫ 905
  → C_OOM ✓ (matches R9's kill); R10b's own min-avail 43.6 puts 131B ~22 GiB
  under the wall ✓. HBM slope 4.0/33.065 = 0.121 GiB/B.
- **T1** (capacity family): §5's historic 1.76 GiB/B and "222B est" are
  map-axis; measured facts on the true axis: 698.8 @97.891B and (with the
  ohbm5 dial) 865.4 @**217.948B** (R11). Dial headroom to the 905 wall ≈
  40 GiB → est ceiling ~**224B** at T1's near-wall ~1.6–1.8 GiB/B.

**Hold-out predictions @130.4B (d3), WRITTEN BEFORE R10 RAN** (gate: host
±25 GiB, HBM ±3%):
- T2: host **783.5**, HBM **69.9** (interp 100.4→172.6 anchors)
- T3: host **835.1**, HBM **38.0** (interp 100.4→kill-peak 172.6)

| cell | verdict | peak unevict | peak HBM | prediction check |
|---|---|---|---|---|
| R10a T2 130B | OK | 798.6 | 71.1 | **PASS** — host +15.1 (≤25 ✓), HBM +1.7% (≤3% ✓); true-axis re-score: pred 788.7 → +9.9, also PASS |
| R10b T3 130B | OK | 882.5 | 40.5 | **FAIL** — host +47.4 (>25 ✗), HBM +6.6% (>3% ✗); cause: censored kill-peak anchor (see refit above) |

**R13 — T3 refit re-validation (pre-registered BEFORE the model was built).**
New same-family hold-out: d3b-dense-125b-**92×10240**, built as bf16 zeros
(`make_synth_dense.py`), true params **125.629B** (index-verified; matches
the closed-form count exactly). Predictions from the refit line anchored at
(130.956B, 882.5 host / 40.5 HBM), slope 3.36 host / 0.121 HBM; registered
first on the map axis (864.0/39.8 @ "125.4"), re-axised pre-measurement
(shards still being written) to the true axis — same line, de-biased x:
- host = 882.5 − (130.956−125.629)·3.363 = **864.6 GiB** (±25 gate)
- HBM  = 40.5 − (130.956−125.629)·0.121 = **39.9 GiB** (±3% gate)
- verdict predicted **OK** (min-avail ≈ 905−865 = 40 > floor 25)

| cell | verdict | peak unevict | peak HBM | prediction check |
|---|---|---|---|---|
| R13 T3 125.629B (92×10240) | **OK** | 851.3 | 40.5 | **PASS** — host −13.3 (≤25 ✓), HBM +1.5% (≤3% ✓); min-avail 77.7, verdict OK as predicted |

R13 post-mortem on the −13.3 residual: the same-family slope (d3 92L→96L,
h=10240 fixed) = 31.2/5.327 = **5.86 GiB/B** (≈7.8 GiB/layer: ≈2×2.48 weights
+ 2.44 act-root — matches the 2×-RSS physics), vs cross-family 3.36. The
params-only slope is family-dependent (L·h activation scaling), exactly why
fit discipline demands same-hidden-family local slopes near the wall. The
prediction passed because the 5.3B extrapolation kept the family error
(5.327×(5.86−3.36) = 13.3) inside the ±25 band. **T3 ceiling, family-local:
130.956 + (905−882.5)/5.86 ≈ ~135B** (d3-family; next step 100L=136.3B
predicts 913.7 > 905 ☠). Quote "~135B" (was ~138 under cross-family slope).
| R11 T1 ohbm5+flush 215B | **OK** | 865.4 | 162.4 | **CROWN ✓** — pre-merge 864, drift +1.4; min-avail 64.8 |
| R12 so_unsloth 200B | **C_OOM(steady)** | 905.9 | – | **WALL ✓** — died 0.9 GiB over the 905 wall. Note: survived load + 1 warmup step, died mid-step-2 (pre-merge "☠ load ~906" was an *estimate*; 200B was never measured pre-merge — this pins the SO ceiling at ≈200B, steady-phase). Killed by the trainer's own `[host-mem-watchdog]` soft-OOM guard (classifier now keys on that marker; the external sampler's 2 s grid logged min-avail 25.1, just above floor). |

### CAMPAIGN VERDICT (2026-07-22, c17) — ALL GOALS MET

1. **Anchors sufficient**: T2 = 4 OK anchors (57.3/97.9/131.0/170.8B);
   T3 = 4 OK anchors (57.3/97.9/125.6/131.0B) + censored bound @170.8;
   T1 drift-checked @97.9 + crown @217.9; SO drift-checked @97.9 + measured
   wall @202.6.
2. **Estimators correct, both metrics** (hold-out gated, predictions
   registered pre-measurement): HBM within **1.7%** on every hold-out
   (R10a +1.7%, R13 +1.5%; gate 3%). Host RSS within **±25 GiB band**
   (R10a +9.9, R13 −13.3) — noisier, as expected, and family-local slopes
   are mandatory (trap #13 + R13 post-mortem).
3. **asym still #1 dense capacity post-merge, 128k×1**: T1-ohbm5+flush8
   trains **217.9B** (865.4 peak, 40 GiB slack) while so_unsloth dies at
   **202.6B** (905.9) → **+15.3B measured gap**; recompute trio unchanged
   at 53B-class (HBM-bound). Per-tier ceilings: T1-dial ~224B est ·
   T2 ~197B est · T3 ~135B est. Merge drift ≈ zero on every re-measured
   pre-merge cell.

## 8b. TOKEN-SCALING CAMPAIGN (2026-07-23) — find T* maximizing asym % over so_unsloth

Goal (Kevin): identify the seq×batch setting with the LARGEST % capacity gain
over so_unsloth (the strongest baseline everywhere — unsloth-GC strictly
relieves SO-recomp's HBM wall). Memory scales with tokens T = seq×batch.

**Decomposition model (from 128k measurements, true axis).** SO near-wall
host line: host = (0.76 + 1.07·T/128k)·P + 535.5 — root share dR/dP = 1.07
GiB/B at 128k from R = L·seq·h·2B across d2→d5a; reproduces the measured
202.6B kill exactly. Asym advantage = roots parked into spare HBM:
adv(B) = min(R_total, HBM_spare(T))/slope(T); HBM-capped at T ≥ ~64k,
root-supply-capped below. Predicted % over SO: 128k +8–11 (measured) ·
96k ~+23 · **64k ~+32 (predicted optimum)** · 32k ~+26. Note the opposing
trade: recomp-trio wall ∝ 1/T, so shorter T shrinks the ×-vs-recompute story
(4.2× @128k → ~2.9× @64k).

**Inventory note: d5b "235b" true count = 255.195B** (names lie again) —
probes up to 255B need no builds; ≥260B requires make_synth_dense builds
(d5b family 13312: 2.242 GiB→B per layer... 2.242B params/layer, 128L ≈
291B ≈ 580 GB zeros; 4.9T scratch free).

**Chain D (64k×1, running) — pre-registered predictions:**

| cell | pred host | pred verdict | tests |
|---|---|---|---|
| D1 SO @202.647B | **753 ± 25** | OK | root-share model (906 − R/2) |
| D2 SO @255.195B | **821 ± 30** | OK | 64k SO slope (D1→D2 pred 1.295) |
| D3 asym ohbm5+flush8 @255.195B | < D2, G ≪ 185 | OK | asym clears SO-near-wall size |
| D4 asym ohbm5+flush8 @227.908B | ~775 ± 40 | OK | the 128k wedge model clears at 64k |

Then chain E: build ~291B (d5b+16L) → SO ☠ bracket (pred ~912) + asym probe
(dial lowered = park more; lower ohbmN ⇒ more parked/higher G per the
ohbm3@170B G182 / ohbm5@217.9 G162 / ohbm4@228 G194 pattern) + larger builds
(~310/350B) until asym walls → measured 64k %. Then single near-wall pairs
at 96k and 32k to confirm the peak shape; batch-equivalence check 32k×2 vs
64k×1 (expect ≈equal). Verdicts land in the table below as they classify.

**Chain D RESULTS (64k×1) — decomposition FALSIFIED on magnitude:**

| cell | pred | measured | verdict on model |
|---|---|---|---|
| D1 SO @202.647B | 753±25 OK | **OK 812.3** (HBM 58.3) | MISS +59 — half-tokens relieves SO only ≥94 GiB, not 153 (128k ref was a censored kill → roots-on-host share overestimated) |
| D2 SO @255.195B | 821±30 OK | **C_OOM(steady) ≥915.4** | MISS — SO 64k ceiling ≈ **~250B**, not 286 (D1→D2 slope ≥1.96, censored, cross-width) |
| D3 asym ohbm5 @255.195B | OK | **C_OOM(steady) ≥916.2** | crown dial is NOT the 64k-optimal dial; fixed ohbm5 sends incremental roots to host |
| D4 asym ohbm5 @227.908B | ~775±40 OK | **OK 801.1** | ✓ 128k wedge clears at 64k; asym holds +25.3B more than SO at LOWER host (801 vs 812) |

Takeaway: SO gains less from shorter seq than modeled (its ceiling 202.6→
~250B, +23%); the 64k % question is now entirely "how much more can the
dial park" — measured next. All kills censored (≥); fits only through OKs.

**Chain E RESULTS — both DEAD, brackets tightened:**
- E1 SO @228.286B: **C_OOM ≥909.2** (pred OK 862 — MISS; the wide-h penalty
  and/or steeper 64k SO slope binds). SO 64k bracket: **202.6✓(812.3) /
  228.3✗**.
- E2 asym ohbm2 @255.195B: **C_OOM ≥914.3** ≈ D3's ohbm5 death (916.2) —
  dial depth changed ~nothing ⇒ parked budget is auto-capped by idle-HBM
  estimate, not by N, at this size. Asym 64k bracket: **227.9✓(801.1) /
  255.2✗**.
- Standing measured-pair % at 64k: asym best-OK 227.9 vs SO best-OK 202.6 =
  **+12.5%** (vs +7.5% at 128k) — PROVISIONAL until SO @217.948B (d5e,
  same-width-as-d5a) is probed.

**Chain F RESULT:** F1 SO @217.948B @64k = **OK 870.0** (pred 842±25, miss
+28 — SO's same-width 64k slope is 3.77 GiB/B (d5a→d5e), steeper than the
censored 1.96 bound). **SO 64k: best-OK 217.9 (870.0) / ☠228.3 → ceiling
~227B.** Standing measured-pair % at 64k drops to 227.9/217.9 = **+4.6%**
pending the gap builds.

**Chain G (after d5f 237.255B / d5g 246.225B builds) — pre-registered:**
G1 asym ohbm5+flush8 @246.2B: host **880±25**, boundary probe — OK ⇒ 64k
% = 246.2/217.9 = **+13.0%** (beats 128k); C_OOM ⇒ G2 @237.2B (pred 853±25
OK ⇒ +8.8%); both ☠ ⇒ 64k stays +4.6% (worse than 128k → probe 96k next).
Asym 64k est ceiling from censored ✗255.2 (≥914.3) ≈ ~253B.

**Chain G RESULTS — both DEAD, and a WIDTH ANOMALY exposed:**
G1 @246.225B ☠ ≥915.2 (pred 880 MISS) · G2 @237.255B ☠ ≥914.3 (pred 853
MISS). Every h=13312 asym cell dies at **~914–916 regardless of size**
(237/246/255) **and dial** (ohbm5, ohbm2), while h=12800 @227.9B passes at
801.1 with 104 GiB slack — a +9.3B/+512-h step cannot cost ≥113 GiB
smoothly. Working hypothesis: the ohbm dial's parked-roots budget collapses
at h=13312@64k (roots ≈165 GiB land on host ⇒ jump matches). Runtime
parking telemetry absent from logs/heartbeat; documented as an open
system finding (NOT fixed, per the no-fix policy) — capacity ladder
sidesteps via same-width d5c family. ⚠ this also taints D3/E2 as evidence
of "asym 64k ceiling": those deaths are anomaly-driven, not slope-driven.

**Chain H (d5c-family h=12800 ladder: d5h 236.204B / d5i 244.501B /
d5j 252.798B, zeros builds) — pre-registered (slope borrow 3.77):**
H1 @244.50B host **864±25 OK** → H3 @252.79B host **895±25** (edge);
H1 ☠ → H2 @236.20B host **832±25 OK**. 64k measured-pair % vs SO 217.948:
236.2→+8.4 · 244.5→**+12.2** · 252.8→**+16.0** · none→+4.6.

**Chain H RESULTS — both OK, predictions HIT:** H1 @244.501B = **OK 858.9**
(pred 864, −5.1 ✓) · H3 @252.798B = **OK 888.0** (pred 895, −7.0 ✓).
Same-width asym slope: 29.1/8.297 = **3.51 GiB/B** (and D4→H1 gives 3.48 ✓
double-consistent → h=13312 anomaly confirmed as pathological, h=12800
smooth). **64k STANDINGS: asym best-OK 252.798 (888.0, 17 GiB slack; est
ceiling ~257.6) vs SO best-OK 217.948 (870.0; ☠228.3; est ceiling ~227) →
measured-pair % = 252.798/217.948 = +16.0%** (vs 128k +7.5%) · est-est
~257.6/~227 = +13.5% (vs 128k +10.9%). % RISES as T falls at this range →
32k must decide the peak side.

**Chain I (running) — 32k×1 discriminators, pre-registered:** I1 SO
@255.195B: **~835±35** — decides if SO's 32k reach ≥255B (64k kill was
censored ≥915.4; roots/2 = 88.9 GiB relief is a lower bound). I2 asym
@252.798B: **~845±40 OK** expected. Branches: I1 ☠ → SO 32k ceiling <255 →
probe SO@246.225 (d5g) / @237.255 (d5f) to pin SO best-OK; I1 OK → SO
climbs ≥255 at 32k and asym must reach ~272B+ (needs 124L/128L d5c-family
builds 261.10/269.39B) to keep a lead. Then: close the 64k ✗ bracket
(asym@261.10@64k pred ~917 ☠), 96k pair for curve shape, batch-equivalence
(asym@244.501 @32k×2 vs H1's 64k×1 858.9).

**Chain I RESULTS — the whole T-structure resolved:**
- I1 SO @255.2B @32k **☠ ≥908.8** (pred 835 MISS) — vs ≥915.4 @64k: root
  halving moved the death ~nothing ⇒ NOT a steady wall. **It is the SO LOAD
  transient: 2W + K_eff ≈ 905, K_eff = 58 GiB ⇒ P_SO-load-max = 227.3B,
  seq-INDEPENDENT.** Retro-validates every SO cell to ~1 GiB: D1 812.3 =
  2·1.863·202.647+58 = 813.1 ✓ (D1's "OK peak" was its load peak!) · F1
  870.0 = 812.1+58 ✓ · E1 ☠228.3 (2W+58 = 908.6) ✓ · D2/I1 (2W = 951 alone)
  ✓. The earlier "SO width anomaly" was size, not width — 2W. At 128k the
  steady wall (202.6) binds BEFORE load (227.3); below ~96k the load wall
  binds: **SO as-shipped ceiling saturates at ~227B for all T ≤ ~96k.**
- I2 asym @252.798B @32k **OK 887.5** ≈ 64k's 888.0 (Δ −0.5) — with ohbm5
  the roots are already ~fully parked at 64k ⇒ **asym host is T-independent
  below 64k; asym ceiling saturates ~257.6B** (slope 3.51 ≈ 2×1.863 + ε:
  asym's steady 2×-RSS weights tax is its own binding term).
- As-shipped measured-pair %: **128k +7.5 · 64k +16.0 · 32k +16.0
  (saturated) → T* = 64k** (longest T with the max %).
- ⚠ **LOAD-FIX CAVEAT (must ship with the claim):** at T ≤ 64k SO is
  load-bug-bound — the exact objection the 128k study killed. Load-fixed SO
  @64k would be steady-bound at est ~255–280B (decomposition, wide bars,
  0-for-3 track record this campaign) ⇒ the +16% is an AS-SHIPPED number;
  under the load-fixed treatment it shrinks toward ~0 (asym's own 2×-steady
  tax binds first). At 128k asym wins under BOTH treatments — 128k remains
  the treatment-robust setting; 64k is the max-% as-shipped setting.

**Chain J (running) — lockdown, pre-registered:** J1 SO @217.948B @32k:
**870 ± 10 OK** (load peak, T-indep) → locks the 32k pair at +16.0%.
J2 asym @244.501B @ 32k×**batch2** (=64k tokens): **~859 ± 15 OK** — batch
≡ seq equivalence (H1 was 858.9 @64k×1); given I2's T-independence, expect
≈859 regardless. [build d5k 124L = 261.096B] J3 asym @261.096B @64k:
**~917 ± 20 ☠** — closes the asym 64k bracket (252.8✓ / 261.1✗).

**Chain J RESULTS — all three predictions HIT (≤1 GiB):** J1 **OK 869.6**
(vs 870.0 @64k — load peak T-independent ✓) · J2 **OK 858.5** (vs 858.9
@64k×1 — batch ≡ seq exactly ✓) · J3 **☠ ≥916.4** (pred 917 ✓).

**Chain K (2026-07-23) — so_recomp post-merge reconfirm, 3 anchors @128k:**
K1 @43.561B OK **G 167.2** (pre-merge 167, drift +0.2) · K2 @50.406B OK
**G 176.9** (178, −1.1) · K3 @57.251B **G_OOM ✓**. Host 171/204/228 ≪ 905 —
the 2×W load transient (2W+58 ≈ 246 @57B) never binds: HBM wall only.
Measured per-layer slope (176.9−167.2)/8 = **1.21 GiB/L** (pre-merge 1.41)
→ slope-based est ceiling 185 @ ~62.7L ≈ **~55B** (bracket 50.4✓/57.3✗;
old headline est 53B — within bracket either way, figure keeps 53 as the
conservative mid-bracket value unless Kevin opts to relabel to 55).

**Chain L (2026-07-24) — SO + Unsloth GC + ACT-OFFLOAD (`superoffload_mem|
unsloth-off-ohbm0`, the tp-plot seq-capacity baseline) @128k×1:**
L1 @97.891B **OK 864.1** (HBM 52.6; pred 908±40, low edge) · L2 @130.956B
**☠ ≥925.7** ✓. Vs SO-unsloth at the same size (714.6): act-offload adds
**+149.5 GiB** of host acts at 97.9B → slope ≈ 1.83 + 0.96 = 2.79 GiB/B →
**est ceiling ~113B** (bracket 97.9✓/131.0✗; consistency: predicts 956 ≫
905 @131 ✓). Load wall (2W+58 = 479 @113B) never binds — pure host-acts
tax, the baseline-side mirror of asym T3 (~135B, which still edges it
+19%). **Capacity ordering @128k: SO-actoff ~113B < asym T3 ~135B <
SO-unsloth ~202B < asym T2 ~197B* < asym T1-dial 224B (217.9✓).**
(*T2 197 vs SO 202: T2 keeps fg acts in HBM but pays roots on host.)

### 8b VERDICT — token-scaling campaign CLOSED (24 cells, D1–J3)

Walls (as-shipped, true-param axis, measured brackets):
| T | SO ceiling | asym ceiling | est-ceiling % | showcase pair (✓ vs ☠) |
|---|---|---|---|---|
| 128k×1 | 202 (☠202.647 @906) | 224 (✓217.948 / ☠~228) | **+10.9%** | 217.9✓ vs 202.6☠ → +7.5% |
| 64k×1 | **227.3 = load wall** (✓217.948 @870 / ☠228.286) | **257.7** (✓252.798 @888 / ☠261.096) | **+13.4%** | 252.8✓ vs 228.3☠ → +10.7% |
| 32k×1 | 227.3 (✓217.948 @869.6 / ☠255.195; load, T-indep) | ~257.6 (✓252.798 @887.5; T-indep) | **+13.4%** | same (saturated) |

- **T\* = 64k×1** (32k ties; 64k keeps the longer context). Batch is NOT an
  independent knob: J2 proved 32k×2 ≡ 64k×1 to 0.4 GiB.
- Mechanism: below ~96k, SO saturates at its **load wall 2W+58 = 905 →
  227.3B (T-independent)** while its steady wall retreats; asym (ohbm5,
  roots ~fully parked at ≤64k) saturates at **~257.6B** (its own 2×-steady
  weights tax, slope 3.51 ≈ 2×1.863). Both systems T-independent ≤64k ⇒
  % plateaus at +13.4 (est-est) / +10.7 (showcase).
- **Load-fix caveat (ship with any short-T claim):** at T ≤ 64k the SO
  binding wall IS the load bug. Load-fixed SO @64k est ~255–280B (wide
  bars) → the short-T bonus shrinks toward ~0 under that treatment. At
  128k asym wins under BOTH treatments (walls coincide there) — **128k is
  the treatment-robust setting; 64k is the max-% as-shipped setting.**
- Best-OK/best-OK ratio (252.798/217.948 = +16.0% @64k) is grid-dependent
  (SO's coarse ladder) — do NOT headline it; use est-ceiling or showcase.
- 96k unmeasured (est intermediate ~+11–13%, walls ~227 vs ~224–257
  transition zone) — optional.
- OPEN: asym h=13312 anomaly (☠914–916 at 237/246/255 regardless of size
  and dial, while h=12800 passes up to 252.8 with slack; SO's 13312 deaths
  fully explained by 2W load instead). Not fixed per no-fix policy; avoid
  h=13312 for asym capacity claims.
- New models this campaign (all zeros, d5c-family h=12800 unless noted):
  d5f 237.255 / d5g 246.225 (13312) · d5h 236.204 · d5i 244.501 ·
  d5j 252.798 · d5k 261.096.

## 8c. ERRATA + CAPACITY-PUSH CAMPAIGN (2026-07-25, c17)

### ERRATA — §8b's asym 64k/32k results are VOID (audit 2026-07-25)

Cross-checking every §8b cell log against its ledger verdict exposed a
classifier bug compounded by an invalid model family:

1. **Classifier bug**: the (lost) classifier keyed OK on driver rc=0 — but the
   profile driver exits 0 even when the training command fails ("Training
   command failed with status 1 ... done rc=0"). Verdicts were stamped OK with
   ZERO backward calls.
2. **Invalid GQA family**: every h=12800 build (d5c/d5h/d5i/d5j/d5k) has
   heads=100, 100 % 8 ≠ 0 — exactly trap #4, which §3 itself flags as "d5c
   INVALID GQA" — and crashes at the FIRST attention call:
   `RuntimeError: Number of heads in key and value must divide...`
   (asym_forward_calls=3, backward_calls_total=0 in every such log).
3. Consequently every asym "OK" at 64k/32k (D4 801.1, H1 858.9, H3 888.0,
   I2 887.5, J2 858.5) was the **load-phase footprint of a crashed run** — this
   is also why they were seq- and batch-independent (loads are), which the 8b
   narrative mistook for "asym host is T-independent".
4. The h=13312 "width anomaly" was backwards: G1/G2/D3/E2 (valid shapes,
   asym_forward_calls 375–627 = killed mid-forward of step 1) were the ONLY
   real asym 64k data; the h=12800 "passes" they were compared against were
   fake. There is no anomaly. Likewise no "dial auto-cap" exists in code — the
   parker was a blind every-Nth counter, period.

**What survives the audit**: all of §8 @128k (every OK cell has train_runtime +
real backward counts; crown R11 217.948B OK 865.4 ✓), chain K, chain L, and the
SO cells D1 (812.3), F1 (870.0), J1 (869.6) + SO load-wall deaths (the 2W+58
model stands). **Voided**: every §8b asym 64k/32k cell, the +16%/T*=64k
claims, the 3.51 "steady slope" (fit through load footprints), and J2's
batch-equivalence (two identical crashed loads). Real standing before this
campaign: 128k crown 217.948B; 64k asym = NO valid OK, bracketed only by
☠237.255 (G2). The rebuilt classifier (scripts/lf/capacity/classify_cell.py)
now requires positive train_runtime evidence for OK and has a CRASH verdict;
the rebuilt builder refuses indivisible GQA shapes.

### Root cause of the "2×-weights steady tax" — pow2 pinned-allocator rounding

Probe (this venv, GB200): torch's CachingHostAllocator rounds EVERY pinned
allocation to the next power of two (37→64, 111→128, 259→512, 731→1024,
1531→2048 MiB; +41% on that batch), hoards freed blocks until
`torch._C._host_emptyCache()`, and `.to("cpu", non_blocking=True)` allocates
its D2H destination THROUGH it (pinned) — so weight homes, GC roots and
transients all pay the tax. Traces confirm: R3 peak = 682.4 GiB Shmem vs 16.4
anon. Reconciliation @97.9B/128k: W 182.4×~1.18 + roots 88×(2.36→**4.0**
GB, 1.70×) ≈ 328 GiB + opt ≈ closes the historic ~680 within noise. SO pins
exact-size via DeepSpeed's own cudaHostAlloc — why its host physics were
always clean (2W+K) while asym's looked like "2×W steady".

**Fixes (2026-07-25, all default-OFF env gates):**
- `ASYM_EXACT_PINNED=1` — HostWeight/cpu_adam homes: exact-size clone +
  in-place `cudaHostRegister` (asym_gemm/training/exact_pinned.py; register
  227 GB/s touched, is_pinned()=True, D2H/H2D within ~10% of allocator-pinned).
- `ASYM_EXACT_PINNED_ROOTS=1` — unsloth-GC boundary roots ride an
  event-guarded exact-pinned RootPool (LF checkpointing.py), one slot per live
  root, zero register churn after step 1.
- `UNSLOTH_GC_OUTER_HBM_AUTO=1` + `UNSLOTH_GC_OUTER_HBM_RESERVE_GB` (20) —
  budget-aware root parking: park while driver-free HBM > reserve.
  Self-limiting (replaces the blind every-Nth ohbm dial; cannot repeat the
  ohbm4@228B G194 death).
- `ASYM_HOST_FLUSH_EVERY_N_LAYERS` hook REBUILT (the original implementation
  was lost with the tree wipe — it existed only in this doc) in LF
  checkpointing.py `_install_host_cache_flush_hooks`, + optional
  `ASYM_HOST_FLUSH_PRESSURE_GB` (flush every layer under pressure).

Toolkit rebuilt in `scripts/lf/capacity/` (mem_sampler.sh, run_capacity_cell.sh,
classify_cell.py, make_synth_dense.py — sparse zeros builds, GQA-guarded,
true-param counts printed; d2/d5e rebuilds reproduce 97.891/217.948B exactly).
Models: 12288-family ladder d6a..d6f = 120L/233.249 · 128L/248.550 ·
136L/263.851 · 144L/279.152 · 152L/294.453 · 160L/309.754B.
Ledger: `profiling_results/capacity_push_c17/ledger.tsv`; predictions
registered pre-measurement in PREDICTIONS.md.

### Push results (live)

All cells: 128k or 64k ×1, floor 25, serial, sync-guarded; spec
`asym_cpuadamwds|unsloth-ohbm0|ligerloss1` + capacity env per cell; HBM =
runner-polled nvidia-smi memory.used (includes ~3 GiB context — ≈ +4 vs the
old breakdown metric).

| cell | fixes | verdict | peak unevict | peak HBM | prediction check |
|---|---|---|---|---|---|
| V1 97.891B/128k | none (pre-fix repro) | **OK** | **699.3** | 78.1 | gate vs R3 698.8: **Δ +0.5 — toolkit/sparse-build equivalence PROVEN** (wall 776s vs 767s) |
| V2 97.891B/128k | exact-pinned (no parking) | **OK** | **406.6** | 78.1 | pred 525±25 — beat it by 119: **−292.7 GiB vs pre-fix**; level is now clean physics (W 182.4 + roots 193.4 + opt/misc ≈ 406); wall 758s ≈ no slowdown |
| V3 97.891B/128k | + auto-park (final: self-heal, r45) | **OK** | **303.5** | 176.4 | **−395.8 GiB vs pre-fix (−57%)**, flat through both steps, loss lines present. Parker took 3 iterations: r20 driver-free G_OOM'd (backward transients ride on parked roots) → allocated-basis over-parked, G_OOM (fragmentation) → FINAL: driver-free + one empty_cache self-heal when free<reserve while cache ≥ reserve (fixes step-2 blindness: torch never returns freed segments to the driver) |
| X1 217.948B/128k | full set, r60 (r45 G_OOM'd @181.5 — 12288 transients bigger) | **OK** | **707.4** | 172.9 | R11's crown model at **−158.0 GiB** (865.4 → 707.4); 198 GiB slack |
| X2 233.249B/128k | full set, r60 | **OK** | **762.7** | 172.9 | **NEW 128k CROWN: 233.2B trains (+15.3B / +7.0% over the campaign best)**; slope X1→X2 = 3.61 ≈ no-more-parking marginal (W 1.86 + roots 1.53 + opt) |
| X3 248.550B/128k | full set, r60 | **OK** | **820.9** | 172.9 | pred 817 (+4) |
| X4 263.851B/128k | full set, r60 | **OK** | **876.2** | 172.9 | pred 871 (+5) — **crown 263.9B = +45.9B / +21.1% over campaign best**; 29 GiB slack |
| X5 279.152B/128k | full set, r60 | **C_OOM ≥909.5** | 909.5 | 172.9 | bracket closed: **128k = 263.9✓/279.2✗**, est ceiling ~272B (slope 3.55). STEADY death (D1 attrib run proved forward was ~118L deep; the teardown asym_forward_calls=0 is an artifact) |
| Y1 233.249B/64k | full set, r60 | **OK** | **564.9** | 148.5 | pred 596 (−31) — first VALID asym 64k OK ever; already past SO's 64k load-wall ceiling (227.3) |
| Y2 263.851B/64k | full set, r60 | **OK** | **652.1** | 148.5 | 64k slope tracking 2.85 GiB/B |
| Y3 294.453B/64k | full set, r60 | **OK** | **739.2** | 148.7 | +67B past SO's 64k ceiling; 166 GiB slack |
| Y4 309.754B/64k | full set, r60 | **OK** | **782.8** | 148.5 | pred 783 (±0) |
| Y5 325.055B/64k | full set, r60 | **OK** | **826.6** | 148.5 | pred 827 (−0.4) |
| Y6 340.356B/64k | full set, r60 | **OK** | **870.2** | 148.5 | pred 870 (±0) — **64k crown 340.4B**; 35 GiB slack; slope model hitting ±1 |
| Y7 355.657B/64k | full set, r60 | **C_OOM ≥910.4** | 910.4 | 148.5 | bracket closed: **64k = 340.4✓/355.7✗**, est ceiling ~352B. Steady death (see D1 attrib note below) |

### 8c VERDICT — capacity-push campaign CLOSED (2026-07-25, 16 cells V/X/Y)

| T | pre-push valid best (asym) | post-fix measured bracket | est ceiling | vs strongest baseline |
|---|---|---|---|---|
| 128k×1 | 217.948 ✓865.4 (R11) | **267.677 ✓890.0 / 279.152 ✗≥909.5** | **~272B** (slope 3.55) | SO ☠202.6 → **+32% measured**; vs old asym crown **+49.7B / +22.8%** |
| 64k×1 | none valid (☠237.255 only) | **340.356 ✓870.2 / 355.657 ✗≥910.4** | **~352B** (slope 2.85) | SO as-shipped ceiling 227.3 (load wall, T-indep) → **+113B / +50%** |

- All OK cells have train_runtime + both loss lines (classifier requires it now);
  same-config anchor V1 reproduced the campaign's R3 at +0.5 GiB → these rows
  are directly comparable to the §8 numbers.
- At 97.891B/128k the fix set is **−395.8 GiB (−57%)** host (699.3 → 303.5),
  wall-time neutral (758–771s vs 776s).
- Load-fix caveat REVERSED vs 8b: SO's short-T ceiling is its load bug (227.3
  for all T ≤ ~96k); asym's new ceilings need no such asterisk at 128k (SO's
  walls coincide there). At 64k, load-fixed SO est ~255–280B still loses to
  asym's measured 340.4 by ≥60B.
- **X6 fine rung (2026-07-25 late)**: 267.677B/128k **OK 890.0** (pred 890
  ±0) — **final 128k crown 267.7B ✓ / 279.2 ✗**, 15 GiB slack, est ~272B
  confirmed.
- **D1 diagnostic (279.152B/128k, ASYM_MEM_ATTRIB_LOG)** CORRECTED the X5/Y7
  phase call: at death root_pool = 345.7 GiB / 118 slots ⇒ forward was ~118
  layers deep — those were ordinary STEADY deaths ("asym_forward_calls=0" in a
  killed run's teardown line is an artifact; do not phase-classify from it —
  trap #14 note). Attribution closes exactly: registered 874.9 = W 520.1 +
  packed roots 345.7 + opt 9.1; unevict 898.7 = registered + ~24 misc. No
  hidden wall exists: **host = 1.863·P + (roots − parked ~76) + ~33.**
- **NEXT (open, future levers)**: (a) deeper parking — G peaked 173/185, and
  parked is only ~76 GiB; a per-size reserve schedule or paging parked roots
  through spare HBM more aggressively is worth ~+3-4B/10 GiB; (b) root
  compression (bf16→fp8 boundary roots halves the root term — needs loss
  validation); (c) MoE track re-run with the fix set; (d) NUMA balance of
  registered homes unchecked; (e) panvme extension on top of exact-pinned.

## 9. Extension notes (NOT part of the headline comparison)

- **asym + NVMe weight-paging (`_panvme`)**: 275B OK C521 · 320B OK C581/G131
  (ohbm2 rebalance; wall migrated host→HBM-roots→dialed). Excluded for
  fairness; an NVMe-vs-NVMe study (ZeRO-Infinity-style) is future work.
- Wall-migration mechanism (the story the figures tell): HBM ckpts (53B) →
  host 2×W load (~200B) → asym pinned-W (~245B) → panvme → HBM roots → dial.

## 10. Traps (ALL of them — read before running anything)

1. `free`'s 1.69 TB is HBM-as-NUMA; real pool 957 GB (rules §1).
2. VmHWM is file-cache-contaminated; end-RSS understates. Peak unevictable
   (anon+Shmem, NUMA0+1, run-windowed) is THE metric.
3. `sync` after minting checkpoints (dirty cache = unevictable at load).
4. Synth GQA: (h/128) % 8 == 0.
5. Wide-h ≈ +30 GiB host penalty — same-h extrapolation only.
6. SO placement adaptive — near-wall local slopes only, never global fits.
7. Samplers: setsid + args-matched kills; window traces; zombie samplers
   contaminate shared trace files.
8. 2×-load: SO = 2W+151 even with zero.Init; asym = 1×W. Don't "fix" it —
   estimate around it (Kevin directive; the estimates show it costs ~nothing).
9. One cell at a time; 30 s settle; timeout-guard every cell; supervisor loop.
10. Kill trainers by args-matched PID loops (never bare pkill patterns).
11. is_qwen3_moe_routed_model must match the checkpoint name (the 235B/slice
    matcher bug burned a run) — verify matcher on new synth names.
12. Container: work happens INSIDE enroot (`asym_sft_38` for this checkout);
    `/workspace` present ⇒ in container. Non-interactive: `enroot start` with
    the env vars from memory/enroot-noninteractive-builds.md.
13. **Censored anchors**: a C_OOM kill-peak is a one-sided bound (true peak
    ≥ kill value), NEVER an interpolation anchor — the watchdog kills
    mid-climb, so fits through it underestimate slopes. Burned R10b: T3@130B
    predicted 835 via the 172.6B kill-peak, measured 882.5 (+47). Fit only
    through OK-verdict points; use kills solely as consistency checks
    (predicted > wall ⇒ kill expected).
14. **OK requires positive training evidence** (2026-07-25): the profile
    driver exits rc=0 even when the training command fails — verdicts keyed on
    rc produced the fake §8b OKs. classify_cell.py demands train_runtime and
    flags CRASH; always sanity-check backward_calls_total > 0 in the runtime
    line before quoting a cell.
15. **Pinned pow2 tax**: torch's CachingHostAllocator rounds every pinned
    block to the next power of two and `.to("cpu", non_blocking=True)` dsts go
    through it. Capacity cells MUST run ASYM_EXACT_PINNED=1 +
    ASYM_EXACT_PINNED_ROOTS=1 (or state the tax); never mix fixed/unfixed
    cells in one fit family.
16. **Parker budget basis**: driver-free goes blind on step 2 (torch never
    returns freed segments); allocated-basis over-parks and G_OOMs on
    fragmentation. The shipped parker = driver-free + one empty_cache
    self-heal; reserve 45 @h≤9216, 60 @12288 (backward transients ride on top
    of ALL parked roots).
17. Cell runner needs TEMPLATE=qwen3 + HOST_MEM_WATCHDOG_FLOOR_GB + OVERWRITE
    =true (driver dedupes by run-dir signature and env gates are NOT in the
    signature — a repeat spec silently SKIPs as existing-complete).
