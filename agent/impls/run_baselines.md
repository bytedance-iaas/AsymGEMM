# PROMPT (standing instruction for this agent — keep at top, work below)

**SCOPE OVERRIDE (user, 2026-08-03): FSDP2-Offload ONLY — the ZeRO3-Offload
queue (chainZ) is DROPPED from this campaign. Ignore zero3 items below; the
zero3_offload series stays as-is (derived) unless a future directive says
otherwise.**

Run the FSDP2-Offload and then the ZeRO3-Offload baselines DIRECTLY (real
runs, not placeholders/derived bars) for the 1-rank and 2-rank throughput
plots. Do the MoE models first, then the dense ones. Probe EVERY rendered
plot column for both systems at both ranks — fit or OOM, no cell may remain
synthesized or inferred. FSDP2 has no backend yet: wire it the same way as
superoffload/zero3 (backend token in the run driver), offloading (almost)
the same state set as ZeRO3-Offload (params + gradients + optimizer to
host). Validate with a loss-parity smoke before trusting any cell.
SANITY CHECK (user 2026-08-02): FSDP2-Offload results should come out
SIMILAR to SuperOffload — same placement class (params+grads+optimizer on
host) — compare every FSDP2 cell against the banked superoffload (so-recomp)
cell at the same column/rank; a wild divergence means the wiring is wrong,
not a result. Investigate before banking. When a
system's placements are refreshed, put the REAL values into the two plot
DATA dicts, drop the derived-bar synthesis for that model, regenerate the
figures, and push to the Overleaf remote (figures only, no prose). Keep
going and do not stop until every FSDP2/ZeRO3 placeholder in both plots is
replaced with a real measured value. Record every run and verdict in this
doc (§8) as the point of record.

================================================================================
## §1 MISSION + MODELS (explicit run order)

Fill the two remaining synthesized series in the throughput figures —
"FSDP2 Offload" and "ZeRO3 Offload" — with measurements, both ranks.
Order: **FSDP2 first, then ZeRO3** (user 2026-08-02); within each system:

MoE FIRST:
1. `q3-30b-a3b`   = Qwen/Qwen3-30B-A3B        (weights cached on c18 ✓)
2. `q3.5-35b-a3b` = Qwen/Qwen3.5-35B-A3B      (cached ✓; FLASH_ATTN=fa4
   auto-venv — the driver switches to .venv-fa4 for qwen3.5, nothing to do)
3. `q3.5-122b-a10b` = Qwen/Qwen3.5-122B-A10B  (cached ✓; fa4 too)
4. `glm4.7-flash` = zai-org/GLM-4.7-Flash (MoE; driver shorthand exists
   [model_integration.md #5]; weights NOT in the c18 cache — download first,
   HF token is in place)
5. `glm4.5-air`   = zai-org/GLM-4.5-Air (MoE; shorthand exists
   [model_integration.md #4]; weights NOT cached — download first)
6. `mixtral-8x22b` = Mixtral-8x22B — **IN the figure but NOT runnable from
   this tree/node**: no model shorthand in scripts/lf/profile_lora_lf_test_source.sh,
   no weights in the c18 cache; its row is c14-native (tpfig campaign,
   agent/anchors_tmp on the c14 tree). SKIP and record the skip, unless you
   are given c14 access or port the shorthand + download weights (~260 GB).

Dense SECOND (after ALL the MoE models above, GLMs included):
7. `q3-32b`       = Qwen/Qwen3-32B             (cached ✓)
8. `llama3.3-70b` = meta-llama/Llama-3.3-70B-Instruct (cached ✓; GATED —
   HF token already at /scratch_local/user_data/shutian/kevin/cache/huggingface/token,
   source of truth /home/kevinni/env/bashrc.sh:97)

## §2 WHAT EXISTS vs WHAT TO BUILD

- **ZeRO3-Offload: ready.** `zero3_offload_mem` is a first-class BACKEND in
  scripts/lf/run_lf_lora_sft.sh (maps to ds_z3_offload_mem_config.json).
  Config string for the profile driver: `zero3_offload_mem|recomp|ligerloss1`
  (the plot series is recompute-class by design — plain HF gradient
  checkpointing, no unsloth-GC).
- **FSDP2-Offload: wire it — INTERFACE CONTRACT IS HARD (user 2026-08-02):
  the backend must be selected EXACTLY like zero3*/superoffload*, i.e. a
  plain BACKEND token; internals may differ, interfaces may not.**
  Concretely, ALL of the following must hold when done:
  * `BACKEND=fsdp2_offload` is a case in run_lf_lora_sft.sh's backend
    switch, sitting beside zero3_offload_mem/superoffload_mem;
  * the profile driver accepts the standard config triple
    `fsdp2_offload|recomp|ligerloss1` in RUNS strings — same grammar,
    same recompute-token semantics as every other backend;
  * `tp_probe_fill.sh <model> <tag> 'fsdp2_offload|recomp|ligerloss1'
    <seq> <gpus> <b...>` works UNCHANGED (no special env needed to select
    the backend; |1 and |2 via the normal model|N spec);
  * artifacts land in the standard layout with
    PROFILE_BACKEND_LABEL=fsdp2_offload (jobs.tsv / summary.md /
    step_samples.json parse with parse_fill_cell.py unmodified);
  * its config json lives beside the ds configs (LF examples/deepspeed
    dir pattern) and is selected BY the branch, not by the caller;
  * the branch itself sets CHECK_SUPEROFFLOAD=false CHECK_CPUADAM=false
    (caller passes nothing special).
  Internal plumbing (verified 2026-08-02 in the container venv):
  transformers 5.6.0 has NATIVE `fsdp`/`fsdp_config` TrainingArguments and
  accelerate 1.11.0 supports `fsdp_version: 2` → wire TRUE FSDP2 (torch
  fully_shard): branch passes `--fsdp "full_shard offload"` +
  `--fsdp_config <json with fsdp_version:2>` (auto transformer wrap;
  sync_module_states; FSDP's own activation checkpointing OFF — recompute
  comes from the `recomp` token; gradient checkpointing needs
  use_reentrant=False under FSDP). torchrun -n2 for |2, torchrun -n1 for
  |1 (FSDP needs dist init; world-size-1 full-shard = pure CPU-offload =
  the zero3-offload-equivalent placement). LF parser has NO fsdp guards
  (verified); if any deepspeed-only assumption trips elsewhere, follow the
  narrow-opt-in precedent at parser.py:260-282, never a blanket removal.
  FSDP1 fallback only if FSDP2 hard-blocks — record which version ran.
  **Gate:** loss-parity smoke vs rc (same model/seq/b/seed: q3-30b 32k b1 ×3
  steps; match banked rc losses within ~0.02/bf16 noise) + memory sanity
  (host RSS carries the optimizer; HBM well below rc's) BEFORE any cell counts.
  **Sanity band (user 2026-08-02): FSDP2-Offload ≈ SuperOffload.** Both
  offload the same state set, so per-cell lat/TP/HBM/RSS should track the
  banked superoffload cells at matched (model, seq, batch, rank). Treat a
  cell far off its superoffload counterpart (rule of thumb: >2× either way
  on TP, or an OOM/fit flip with no placement reason) as a WIRING SUSPECT —
  diagnose (placements, wrap policy, offload flags, dtype) before banking;
  if the deviation survives diagnosis, bank it AND record the explanation
  in §8 next to the cell.

## §3 CELL MATRIX (probe EVERY rendered lean column, fit-or-OOM)

Columns below are exactly what the paper figures render (after the
record-vs-render filter). Batch seeds = rc's measured max (first-fit
descending; deeper columns b1). Expected walls ≈ rc's (same placement
class) — but MEASURE, don't assume: if a system dies earlier/later than rc,
that IS the result. TP at rank 2 = GLOBAL tok/s (2× the per-invocation parse).

RANK-1 (gpus=1):
| model | fit-candidates (batch seed) | expected-OOM columns (probe all) |
|---|---|---|
| q3-30b-a3b | 128k (b2→1) · 320k (b1) | 640k · 800k · 1.1M · 1.4M |
| q3.5-35b-a3b | 128k (b2→1) | 384k · 576k · 1.02M · 1.15M · 1.66M |
| q3.5-122b-a10b | 32k (b6→5→4) · 128k (b1) | 288k · 320k · 512k · 672k |
| glm4.7-flash | 32k · 64k (batch: seed from the row's rc cells; else b2→1) | 96k-192k per its rc wall — read the row comment in plot_tp_vs_seq.py, probe every rendered column beyond the measured rc wall |
| glm4.5-air | 16k · 32k (same seeding rule) | 48k-128k per its rc wall — same rule |
| q3-32b | 128k (b1) · 160k (b1) | 192k · 384k · 576k · 640k |
| llama3.3-70b | 96k (b1) | 192k · 320k · 352k · 384k · 448k |

GLM note: both GLM rows were banked by a parallel campaign — BEFORE running,
read their DATA row comments (both plot scripts) for rc's measured walls and
batch conventions, and split fit-vs-OOM columns accordingly (same fit-or-OOM,
probe-everything standard). Rendered lean columns: glm4.7-flash R1+R2 =
32k·64k·96k·128k·160k·192k; glm4.5-air R1+R2 = 16k·32k·48k·64k·96k·128k.

RANK-2 (gpus=2, GPUs 0,1 — the driver rejects non-same-superchip pairs):
| model | fit-candidates | expected-OOM columns |
|---|---|---|
| q3-30b-a3b | 384k (b1) | 640k · 800k · 880k · 960k · 1.04M |
| q3.5-35b-a3b | 256k (b1) | 384k · 512k · 576k · 640k · 896k |
| q3.5-122b-a10b | 128k (b1) · 192k (b1) | 256k · 288k · 320k · 336k |
| glm4.7-flash | 32k · 64k (seed per row comments) | remaining rendered columns past its measured rc wall |
| glm4.5-air | 16k · 32k (same) | remaining rendered columns past its measured rc wall |
| q3-32b | 128k (b1) · 168k (b1) | 256k · 320k · 384k · 416k |
| llama3.3-70b | 104k (b2→1) | 128k · 168k · 192k · 224k · 256k |

≥900k cells: MAX_SAMPLES=512. OOM-column runs die at load/first-step —
quick; a surprise FIT there is a real cell: bank it, and if it was the last
rendered column, probe one column deeper for the wall.

## §4 PROTOCOL (c18 discipline — non-negotiable)

One run at a time on the whole node. Wrapper:
`scripts/lf/tp_probe_fill.sh <model> <tag> <config> <seq> <gpus> <b...>`
(rank-aware; FILL_POOL=0 singles / 0,1 pairs; artifacts-FIRST verdicts —
the q3.5 teardown false-fail is rescued before the hardfail greps;
freshness-bounded evidence; one-shot DATASET_OVERWRITE retry). Cell
extractor: `scripts/lf/parse_fill_cell.py <rundir> <ranks> <seq> <b>` →
lat s/it · TP · HBM GiB/% (peak reserved / 189471 MiB) · RSS GB · spread%.
Chain-script pattern (guard = live-PID-filtered GPU-empty check — a
compute-app PID with no /proc entry is a dead driver ghost, ignore or clear
via CUDA claim-all+empty_cache; MemAvailable ≥1200 GB fabric-scale; stale
/dev/shm/asym_fabric_* removal; GPU-residual log line): copy
/scratch_local/user_data/shutian/kevin/cache/fillruns/chainY.sh. The ZeRO3
cell list is already scripted at fillruns/chainZ.sh — reuse it AFTER FSDP2.
Verdict TSV: fillruns/results_redo.tsv. Env: NUMACTL membind/cpunodebind
0,1 (CPU NUMA nodes; GPU HBM = nodes 2/10/18/26 — never bind those).
w1+m2 (PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1). Kills: `kill -9
<exact PID>` only; never pkill patterns that can self-match. Watchdog
floors are per-model config (WATCHDOG_FLOOR_GB_BY_MODEL in
run_lf_lora_sft.sh); do NOT change them for baselines — an OOM at the
standing floor is the verdict.

## §5 BANKING + FIGURES

- DATA dicts: scripts/figures/plot_tp_vs_seq.py (R1) and
  plot_tp_vs_seq_2r.py (R2). Add real rows under keys `fsdp2_offload` /
  `zero3_offload` (plain ints; "OOM" measured; NEVER "OOM*" — you probed
  everything). Then patch `_draw_panel` in BOTH scripts: synthesize via
  `_derived_from_recomp` ONLY when the key is absent from the model's spec
  (mixtral exception CLOSED 2026-08-04: c14 made it runnable → measured
  OOM(host) all columns, banked like the rest).
- Row comments carry date, tags, batch, HBM%, and deltas vs the old derived
  values (house style — see the asym/uns rows).
- Regen (host python lacks matplotlib — run in-container):
  `enroot start ... asym_sft_46 bash -c 'cd .../scripts/figures && python3 plot_tp_vs_seq.py && python3 plot_tp_vs_seq_2r.py'`
  Panels must keep: ≤6 columns, no asym-OOM columns (the structural
  record-vs-render filter enforces this), 1.82in height, tight bbox.
- Overleaf: repo at `agent/overleaf/[MLSys 26 Sub] Superchip-based LoRA`
  (remote https://git.overleaf.com/6a41c0d2ac2089c1db81c789). Copy changed
  PDFs from scripts/figures/out/ over figures/, `git pull --rebase` first,
  commit+push FIGURES ONLY (a parallel writing campaign owns the prose —
  do not touch .tex). Verify blob hashes vs origin/main after pushing.

## §6 ENVIRONMENT FACTS + TRAPS (learned the hard way — read once)

- Container: enroot `asym_sft_46`; use the exact mount set from the
  fillruns/chain*.sh launchers (workspace + env + cache). HF_HOME=
  /scratch_local/user_data/shutian/kevin/cache/huggingface (token present).
- Host budget ~957 GB (NUMA 0+1 only; `free`'s 1.69 TB is fabric-inflated).
- Killed asym runs leak /dev/shm/asym_fabric_* files (up to arena-size) —
  the guard cleans them; baselines themselves use no arena.
- Datasets: most §3 seqs already built on c18; missing ones auto-build
  in-run (adds 10-40 min). "validation_ok=False / missing registration" →
  rerun the SAME batch with DATASET_OVERWRITE=true (wrapper does it once).
- Relocated-venv trap: if torchrun paths ever point at SFT-38, STOP — see
  fix_qwen3.5_tp.md [09:34] for the two-layer repair (shebangs + editable
  installs silently importing the wrong tree).
- Rank-2 baselines run DP via torchrun |2 exactly like the banked
  uns/rc/uns-OFF R2 cells (DeepSpeed owns comm for zero3; FSDP owns comm
  for fsdp2 — no manual allreduce path is involved for baselines).
- Stale nvidia-smi ghost (dead PID holding ~1.7 GB): harmless; filter by
  /proc existence; CUDA allocation pressure reclaims it if ever needed.

## §7 COORDINATION / HANDOFF

- A 36-run OOM*-confirm sweep for the EXISTING series (rc/uns/uns-OFF) is
  in flight in the parent session ("chain Y", log fillruns/chainY.log,
  ~10/36 done at handoff). **WAIT for its `CHAIN-Y DONE` marker before your
  first GPU launch** (one-run-at-a-time node rule). FSDP2 WIRING + smoke
  prep is CPU-side and can start immediately. The parent session flips
  those 36 cells OOM*→OOM in DATA and re-pushes — you own ONLY the
  fsdp2_offload / zero3_offload series (+ the derive-only-if-absent patch).
- The parent session's auto-sequencer that would have fired chainZ.sh
  (zero3) after chain Y has been DISARMED — the zero3 queue is yours and
  runs AFTER FSDP2 per the prompt.
- Records/history on this node: fix_plot_placeholders.md (§0-§8 — the whole
  placeholder campaign incl. the max-TP-over-batch cell rule and saturation
  conventions), fix_qwen3.5_tp.md (q3.5 campaigns + the EP2 grad-offload
  port), s04-p1-dgx-02-c12/-c14 dirs (provenance of 30b/32b/llama rows).

## §8 PROGRESS LOG (append-only; every run and verdict, no exceptions)

- [2026-08-02] Doc created (parent session). FSDP2 backend NOT yet wired;
  zero3_offload_mem verified present in the driver; cell matrix frozen from
  the rendered lean columns; chain Y (existing-series OOM sweep) in flight
  on the node.
- [2026-08-03] **FSDP2 WIRED (CPU-side complete, this session — driving c18 over ssh from c14)**:
  * `fsdp2_offload)` case added to run_lf_lora_sft.sh's backend switch (46 tree) beside
    superoffload* — PROFILE_BACKEND_LABEL=fsdp2_offload, BACKEND=torch (torchrun |1/|2 via
    the standard NUM_GPUS path), NO ZERO_BACKEND_LABEL (assert_deepspeed_scope untouched),
    branch sets CHECK_SUPEROFFLOAD=0 CHECK_CPUADAM=0, config json resolved from the LF
    examples/deepspeed dir per the ds-config pattern. bash -n clean.
  * CMD_ARGS: `--fsdp "full_shard offload" --fsdp_config <json>` appended iff
    FSDP_BACKEND_LABEL set (beside the is_zero_backend_run --deepspeed append); hard error
    if the json is missing.
  * `LlamaFactory/examples/deepspeed/fsdp2_offload_config.json` written: fsdp_version 2,
    TRANSFORMER_BASED_WRAP auto-wrap, reshard_after_forward, sync_module_states,
    cpu_ram_efficient_loading, activation_checkpointing false (recompute stays with the
    recomp token; LF itself forces use_reentrant_gc=False under fsdp2 — checkpointing.py:388).
  * SCHEMA VALIDATED in the c18 container venv: TrainingArguments(fsdp="full_shard offload",
    fsdp_config=<json>) parses; transformers 5.6 strips the fsdp_ prefix and resolves
    `version` = **2** (training_args.py:2809) — true FSDP2 confirmed at the args layer.
  * No banked q3-30b 32k·b1 rc run exists on c18 → the parity gate runs its OWN fresh rc
    reference: chainF.sh smoke phase = fsmk32rc (RC) + fsmk32f2 (F2), same seed/protocol,
    then PAUSES for operator parity verification (relaunch with FSMOKE_OK=1).
  * chainF.sh staged at c18 fillruns/ (smokes + full §3 FSDP2 matrix R1+R2, MoE→dense,
    ~57 runs incl. OOM probes; GLM cells deferred to chainF2 pending weight downloads —
    GLM-4.5-Air/4.7-Flash confirmed ABSENT from the c18 HF cache).
  * NODE: parent sweep is chainY2 (not chainY) — LIVE now (fy2b896rc at 01:42Z; 1.66M cells
    still queued). Handover tripwire armed: CHAIN-Y2 DONE marker OR sustained GPU-idle.
    No GPU launches from this campaign until then.
- [2026-08-03 04:2x-05:0xZ] **FSDP2 GATE = PASS (wiring proven; ≈ SuperOffload)**. Debug trail:
  (1) fsmk32f2b HARDFAIL: profile drivers have their OWN backend allowlists (source+both, case
  entry + gpu-count alternation) — fsdp2_offload added to all 4 sites. (2) fsmk32f2c crash:
  accelerate's fsdp2 fp32-upcast `.to(float32)` vs cpu_ram_efficient_loading's broadcast-
  materialized cuda buffers (mixed-device swap) → set cpu_ram_efficient_loading:false (DS
  baselines full-load per rank anyway; revisit if 122b r2 needs it). (3) fsmk32f2d FIT but
  HBM 100.1 GiB: `auto_wrap` missing from --fsdp → whole-model gather; flag now
  "full_shard offload auto_wrap". (4) fsmk32f2e FINAL: losses 1.4688/1.4531/2.0 vs rc
  1.4499/1.4506/1.9990 (Δ≤0.02 ✓); step 14.4s vs rc 12.5s (−13%, ≪2× suspect line ✓);
  RSS 245 vs 194 (fp32 masters on host ✓); breakdown CSV proves params+grads+OPTIMIZER on
  CPU host (routed_experts optimizer_state_cpu 25.3 GB). HBM: allocated 19.5 GB (rc-class ✓);
  reserved 71.2 = +57 GB FSDP2 gather-churn allocator cache, measured WITH
  expandable_segments:True (default for ALL banked cells — fair). House metric = reserved →
  bank reserved, carry allocated in row comments. chainF matrix launched with FSMOKE_OK=1.
- [2026-08-04 ~10:4xZ] **GLM PANELS ADDED TO THE PAPER'S COMBINED FIGURES (user: Overleaf plots
  had NO GLM at all — any series)**. Root cause: the .tex includes ONLY figures/tp_combined.pdf
  + tp2r_combined.pdf, whose COMBINED_KEYS lists stopped at 122b/mixtral; ALL GLM results
  (asym/SO/uns AND the new fsdp2) rendered only into per-model PDFs and the separate
  tp_glm_combined.pdf the paper never includes. Fix: glm4.5-air + glm4.7-flash appended to
  COMBINED_KEYS in BOTH scripts (R1 now 8 panels 4×2; R2 7 panels, 8th blank; scripts
  hardlinked 39↔46 so both trees carry it). Regen + eyeball: GLM panels present with all six
  series (air fsdp2 = OOM cols, flash fsdp2 = measured bars; GLM zero3 stays derived pending
  the other session's GLM chainZ runs). Overleaf commit f750b26 (on top of the writing
  session's d05278a): 3 combined PDFs updated + missing glm4.5-air per-model PDFs +
  refreshed tp_glm_combined added; origin/main verified equal. Figures only.
- [2026-08-04 ~09:5xZ] **CAMPAIGN COMPLETE — BANKED, FIGURES REGENERATED, OVERLEAF PUSHED
  (112adb0)**. Literalism close-out: first pass's 3 rank-2 probes produced NO verdict — the
  one-shot exported FILL_POOL=0 globally so r2 probes bailed pre-launch (no run dir, no @@@
  line; the r1 five were valid). Relaunched with FILL_POOL=0,1 as flit35c2 (35b 1.02M r2) /
  flit32c2 (32b 448k r2) / flitLLb2 (llama 272k r2): **all three ALL-OOM**, artifacts checked
  (genuine CUDA OOM, hostwd 0; llama dirs lowercase flitll*). All 8 literalism probes = OOM →
  ROWS unchanged. BANKED via bank_fsdp2.py: **8 rows into plot_tp_vs_seq.py + 7 into
  plot_tp_vs_seq_2r.py** (46-tree scripts; 122b padded to true column counts 11/7; both
  py_compile clean; derived fsdp2 synthesis now inert for all banked models — derive-only-if-
  absent). Regen in asym_sft_46-on-c14 (46 venv, matplotlib 3.11.0) → env/figures/out; panels
  eyeballed: 30b R1 3038/1446+OOMs, 122b all-OOM, 35b R2 2090/2547+OOMs — exactly the banked
  values. NOTE: parallel session's chainZ had already banked MEASURED zero3 rows into the same
  scripts (their commit 83c7fcd, "FSDP2 pending the baseline agent") — our regen composes both
  measured series; no collision (bank adds only fsdp2 keys). Overleaf: pull --rebase --autostash
  (their unstaged drafts/motivation_plots_v2.md preserved), 22 tp PDFs copied (all differed),
  FIGURES-ONLY commit 112adb0 pushed; origin/main == local verified. §1 mission satisfied under
  the FSDP2-only scope override: every rendered fsdp2 column in both figures is a real
  measured value or a real probed OOM (zero OOM*).
- [2026-08-04 08:2x] **MIXTRAL RUN (bonus — c14 made it runnable) → joins the OOM(host) class**:
  fm1x64 walked b3/2/1, all host-watchdog (min 51 GB, zero CUDA) — 141B fp32 masters ≈ 564 GB
  + loader transients > pool. STRUCTURAL FINDING now three-for-three: HF-FSDP2-offload's host
  floor ≈ 2× model-bf16 (fp32 masters + load double-copy) vs DeepSpeed ≈ 1× (bf16-resident) —
  caps FSDP2 below ~100B on 960-GB nodes while DS carries 106B/122B/141B fine. Remaining 5
  mixtral cells skipped (load-death seq-independent). llama redo: 96k replicate 1091 ≈ 1092
  (stable −3.5% vs SO, banks with note); 128k r1 = measured OOM. 30b 80k completion: 5204@b4
  vs SO 5206@b4 (0.04%!). 35b 256k: 1037 (SO 844, +23%). FINAL literalism pass (8 probes for
  never-probed non-rendered DATA columns) launched — banking follows its verdicts.
- [2026-08-04 07:4x] **MATRIX COMPLETE + GLM-4.7-Flash BOTH ROWS: 12/12 FIT, ALL ≥ SO** —
  R1 tok/s@b: 32k 3609@8 (SO 3477) · 64k 2104@3 (2037) · 96k 1533@3 (1512) · 128k 1186@2
  (1172) · 160k 954@1 (936) · 192k 805@1 (795) → +1.2–3.8%. R2 global: 7277 (6688) · 4231
  (4104) · 3069 (2974) · 2373 (2306) · 1898 (1898 exact tie) · 1603 (1560) → +0.9–8.8%.
  Batch capacities = SO's at every rung. CHAIN-FC14 DONE 07:36:58Z; llama redo pair
  (fc1e96b + fc1e128) running as the last cells before banking.
- [2026-08-04 05:1x] **GLM-4.5-Air joins the 122B verdict class: OOM(host, fp32-master load
  footprint)** — fc1g16 walked b8/6/4, all host-watchdog (min_avail 31 GB, zero CUDA errors):
  106B fp32 masters + the HF loader double-copy ≈ 900+ GB > the 960-GB Grace pool; DS variants
  (bf16-resident, 212 GB) fit fine. Air row banks all-OOM(host), load-phase seq-independent;
  one r2 probe (fc2g16b) completes evidence. Remaining chain trimmed to 15 cells (Air r2
  probe + Flash both ranks + llama redo pair fc1e96b/fc1e128).
- [2026-08-04 04:5x] **SO-CONFIRM PAIR (user-requested cross-check of the 35B 384k fit-flips)**:
  SuperOffload(recomp) 384k FITS on c14 — r1 992 tok/s · 178.3 GiB, r2 global 1877 · 178.1 —
  vs its banked c18 OOMs: the "capacity flip" is NODE-DAY fragile-edge variance (the exact
  class fix_qwen3.5_tp §8 documents between c14/c18), NOT an FSDP2 capacity win. The c18
  SAME-NODE comparison still stands (FSDP2 fit there while SO OOM'd there), and even
  where both fit, FSDP2 leads: r1 1321 vs 992 (+33%), r2 2547 vs 1877 (+36%). Banking:
  SO's banked c18 row unchanged (not my series); both row comments get the dual-node
  provenance note. Also banked: 35B completions 128k·b2 1166 (SO b2 1024, +14%), 256k FIT,
  512k OOM (wall tight). 30B completions all measured (80k fit · 480k/1.6M/720k/1.12M OOM).
- [2026-08-04 02:3x] **llama3.3-70B BOTH ROWS COMPLETE** — R1: 96k FIT 1092 · 174.7 · 566
  (SO 1132 → −3.5%, edge cell at 92% HBM; replicate queued post-chain + unprobed 128k r1
  column queued) · 192k/320k/352k/384k/448k OOM. R2: ALL-OOM incl. 104k b2→b1 (SO fits 104k
  @1687 — rank-2 against-flip, same gather-footprint class as q3-32b 168k; banked measured
  w/ note). POST-CHAIN REDO LIST so far: fc1e96b replicate · fc1e128 r1 probe.
- [2026-08-04 01:0x] **q3-32b BOTH ROWS COMPLETE** — R1: 128k 1097 (SO 1101, −0.4%) · 160k 936
  (SO 941, −0.5%) · 192k/384k/576k/640k OOM (walls = SO's). R2: 128k 2178 global (SO 2131,
  +2.2%) · 168k OOM where SO fit 1738 (fit-flip AGAINST fsdp2 — placement-consistent: fatter
  per-layer gathers at the 189-GiB edge; banked measured w/ note) · 256k/320k/384k/416k OOM.
  Tags fc1d*/fc2d*.
- [2026-08-03 ~16:4x local] **122B × FSDP2-Offload VERDICT: OOM(host) — a real comparative
  result, not a wiring failure.** Elimination trail: CPU-load fix engaged (no GPU warmup);
  fp32-direct load engaged (print evidence); low_cpu_mem_usage forced True (print) — the
  hostmem curve STILL shows a smooth ~2.5 GB/s climb to ~900 GB consumed: HF's loader holds
  bf16 assembly + fp32 conversion concurrently for a 122B (≈732 GB) + prepare transients,
  which a 960-GB Grace pool cannot host. DeepSpeed variants keep the frozen base bf16
  (244 GB) — that's the structural 2× and why rc/uns/zero3 fit. Per §3's own rule ("if a
  system dies earlier/later than rc, that IS the result"): 122B fsdp2 row banks as
  **OOM(host, fp32-master footprint)** at every column — death is load-phase and
  seq-independent (proven identically at 32k/128k/288k/320k/512k r1); one r2 probe
  (fc2c128b) completes the row's evidence. Chain relaunched, 61 cells (dense onward).
- [2026-08-03 ~15:5x local] fc1c32b/128b STILL host-fired at 24<25 GB — NOT accounting this
  time: the HF-FSDP2 load path holds bf16 (244 GB) + the fp32 upcast copy (488 GB)
  SIMULTANEOUSLY @122B (~732 GB + overheads ≈ the whole Grace pool). DeepSpeed never
  double-copies (bf16-only load) — that's why its 122B cells fit. FIX (incident #9, both
  trees): `ASYM_FSDP2_LOAD_FP32=1` (driver fsdp2 branch) → LF patcher loads torch_dtype
  fp32 directly (per-shard cast; accelerate's upcast becomes a no-op; peak ≈ 500 GB;
  end-state numerics identical — fp32 masters either way). Fresh 122B r1 tags (c-suffix);
  relaunched.
- [2026-08-03 ~15:3x local] fc1c32 (122B 32k b6/5/4) = 3× HOST-watchdog fires at 49<50 GB, ZERO
  cuda OOMs — the c14 X1-class accounting artifact (Grace-only MemAvailable; fsdp2 fp32-upcast
  footprints operate at ~50-60 GB avail by design). §4's "don't change floors" rule was written
  for c18 where the floor never fires; on c14 floor 50 misclassifies HEALTHY runs → chain
  relaunched with explicit HOST_MEM_WATCHDOG_FLOOR_GB=25 (kernel-OOM protection intact; this
  RESTORES cross-node comparability rather than breaking it). Fresh tags fc1c32b/fc1c128b;
  a genuine host-OOM below floor 25 still banks as OOM(host).
- [2026-08-03 ~15:1x local] **REPO SEPARATION (user directive): campaign home = the 39 tree**
  (`/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM`, container asym_sft_42) — one machine,
  one repo, isolated running artifacts. ALL fsdp2 wiring PORTED into 39 (driver case+flags+env
  gates; both profile-driver allowlists; wrapper pg-rewrite; 39-LF patcher CPU-load opt-in;
  config json in 39-LF examples/deepspeed). The 46-tree copies are KEPT (user: 46 uses them
  later) — nothing reverted. 8 datasets copied 46→39 LF/data (read-only source); missing 122B
  deep-seq sets auto-build in-run. Gate smoke f39smk (fsdp2 30B 32k·b1) = **FIT on the 39
  tree**. chainF39 launched (72 cells, same matrix as chainF_c14).
- [2026-08-03 ~14:5x local] **CAMPAIGN MOVED TO c14 (user directive: c14 is the FSDP machine;
  c18 belongs to the other session — hands off henceforth)**. Created enroot `asym_sft_46` on
  c14 (base sqsh, 46-workspace + env + cache mounts; venv torch cuda-OK). BONUS: GLM-4.5-Air +
  4.7-Flash weights ALREADY CACHED on c14 → download step dropped. Gate smoke fc14smk (fsdp2
  30B 32k·b1) = FIT on c14. chainF_c14 launched: 72 cells = 122B fresh (both ranks; r2 may
  host-OOM by fp32-upcast × 2-rank load physics on the 960-GB Grace pool — measured verdict
  either way) + dense pair + completion cells + SO-confirm 35B 384k pair + GLM Air/Flash both
  ranks. Guard: squeegee + MemAvail floor adjusted 1200→500 GB (c14 Grace-only accounting).
  BANKING NOTE: 30B/35B rows were measured on c18 (same node as the banked baseline series);
  c14-measured rows get a node-provenance note in the row comments. The c18-contaminated
  fdbg122a/ff1c* tags stay quarantined.
- [2026-08-03 11:4xZ] **122B LOAD-PATH BUG FIXED + NODE CONTENTION EVENT**. (a) ff1c32/ff1c128
  "OOMs" were a LOAD ARTIFACT: transformers' caching_allocator_warmup pre-allocates
  half-model-bytes ON GPU (182.8 GiB @122b → instant OOM) because LF's device_map path fires —
  transformers' is_fsdp_enabled() gate needs FSDP_CPU_RAM_EFFICIENT_LOADING=1 which this
  backend runs false. FIX (narrow-opt-in, parser-precedent): LF patcher.py honors
  `ASYM_FSDP2_CPU_LOAD=1` (driver fsdp2 branch env) → device_map dropped → CPU load →
  accelerate shards+offloads post-load. Both burned ff1c* tags quarantined from banking.
  (b) During the proving walk (fdbg122a), the PARENT session's sweep (fy1uns672 …) RESTARTED
  on the node → concurrent 122B host loads → my watchdog fired at 49<50 GB → fdbg122a ALL
  verdicts CONTAMINATED (quarantined). Per §7/solo rule: my campaign YIELDS; handover watch
  armed; on node-free: rerun 122B block fresh (ff1c*b tags), then dense, completion cells,
  GLMs, zero3. 35B R2 banked meanwhile: 256k FIT 2090 global (+29% vs SO 1620) · 384k FIT
  2547 (SO OOM'd — favorable flip) · 512k/576k/640k/896k OOM.
- [2026-08-03 09:37Z] **35B R1 ROW COMPLETE**: 128k·b1 FIT 585 (SO b1 530 → +10%; b2 max-TP
  cell → completion queue) · 384k·b1 FIT 1321 · 178.5 GiB (SO-recomp measured OOM there —
  fit-flip in FSDP2's favor, banked as measured) · 576k/1.02M/1.15M/1.66M OOM. Squeegee
  validated in production (the twice-dead 384k config trained). Tags fdbg35a, ff1b384b/576b,
  ff1b1020/1150/1660.
- [2026-08-03 08:3xZ] **35B −9 ROOT-CAUSED + FIXED (node-ops, not wiring)**: sampler-instrumented
  repro showed the trainer at 249 GB RSS with NUMA-0 free ~80 GB — the fp32-upcast load
  transient (bf16+fp32 coexist ≈210+ GB @35B) must fit the membind-0,1 free window; chained
  cells leave page-cache residue there and fast large allocs outpace kernel reclaim → SIGKILL.
  Clean-node repro fdbg35a = **FIT** (also banks the walker-skipped 128k·b1 cell).
  cpu_ram_efficient_loading route dead (its swap crash was already at r1). FIX: **cache
  squeegee in the chain guard** — pre-cell touch-and-free of a 280 GB membind-0,1 anon block
  forces cache reclaim; fair (no run-config/placement change). resume3 launched (49 cells,
  fresh tags ff1b384b/576b; burned 35B tags recorded).
- [2026-08-03 07:5xZ] ff1b128 (35b 128k·b2 r1) exit −9 during LOAD (cutlass-import phase,
  ~4 min in; node avail healthy 1525 GB minutes later) — batch-independent death; suspect
  transient NUMA-0,1 membind pressure (numactl binds CPU nodes; page-cache residue from the
  earlier −9 kills). Wrapper classed −9 as HARDFAIL (rc=2) so the b2→b1 walk STOPPED —
  **128k·b1 unprobed → completion queue**. Empirical discriminator = ff1b384 (live): if it
  loads clean, transient; if −9 again, systematic for the fa4/35B family and gets root-caused.
- [2026-08-03 07:4xZ] **q3-30b BOTH ROWS COMPLETE** — R1: 128k·b2 3038 (+1.8% vs SO) · 320k
  1446 (+0.4%) · 640k/800k/1.1M/1.4M OOM. R2 (global): 384k 2381 (+0.5% vs SO 2370) ·
  640k/800k/880k/1.04M OOM · 960k OOM(host, −9 convention). `_draw_panel` patched in BOTH
  figure scripts: derive-only-if-absent (doc §5). BANKING DEFERRED per model until its
  COMPLETION CELLS run (DATA rows carry more seq positions than the 6 rendered columns —
  banking now would need est entries, which the prompt forbids). Completion queue (runs
  after CHAIN-F DONE, before GLM downloads): 30b R1 80k(b2)+480k+1.6M; 30b R2 720k+1.12M;
  equivalents per model computed at each model's completion.
- [2026-08-03 07:2xZ] ff2a960 (30b 960k r2) exit −9 SIGKILL rank-0, watchdog+cgroup silent =
  the §6 recorded host-OOM class → **verdict OOM(host), measured** (2 ranks × fp32-upcast
  full model + masters + 960k overhead > 957 GB). rc=2-with-SIGKILL grades as OOM per the
  standing convention; expect same at 1.04M r2 (also an expected-OOM column).
- [2026-08-03 06:39Z] **RANK-2 PATH CLOSED**. ff2a384 hardfailed: `No backend type associated
  with device type cpu` — FSDP2 CPU-offload grad-norm reduces over CPU DTensors but accelerate
  inits an nccl-only group. Fix path: --ddp_backend rejects torch's multi-backend syntax
  (transformers choices metadata) → env-gated rewrite in OUR wrapper instead
  (`ASYM_FSDP2_MULTIBACKEND=1` from the fsdp2 branch; run_lf_profiled_train.py wraps
  init_process_group: nccl→"cpu:gloo,cuda:nccl"). Ops cost of the two bad attempts: 4 burned
  r2 tags (b-suffix) + torchrun orphans that twice tripped the guard (correctly; killed by
  exact pid). PROOF ff2a384c: FIT, **global 2381 tok/s vs banked R2 SuperOffload 2370 (+0.5%)**,
  179.0 GiB resv, RSS 129/rank. Fresh tags for redone cells: ff2a384c/640c/800b/880b.
- [2026-08-03 05:52Z] **FSDP2 q3-30b R1 ROW COMPLETE (6/6 measured)**: 128k·b2 FIT 3038 tok/s
  · 122.3 resv · RSS 245 (SuperOffload 2985 → +1.8%); 320k·b1 FIT 1446 · 151.9 · 245 (SO 1440
  → +0.4%); 640k/800k/1.1M/1.4M = measured OOM (matches SO's wall shape; SO OOMs from 480k).
  Sanity band: dead-center at both fit cells. Tags ff1a128/320/640/800/1100/1400.
- [2026-08-03 ~04:1xZ] **NODE TAKEOVER (chainY2 wedged at cell 2/25)**: fy2b896rc's driver sat
  in do_wait + sleep-1 poll for 2h25m (cell budget ~85 min) with ZERO trainer/python processes
  for >1h — torchrun died, driver never noticed. Killed the exact wedged PIDs (654025 driver +
  652685/6/92 probe wrappers), node verified clean (only the known §6 ghost 3054277; avail
  1586 GB). chainY2 remains 1/25 — the PARENT session must rerun fy2b896rc onward (its cell
  results before the wedge are intact in results_redo.tsv). chainF smoke phase LAUNCHED:
  fsmk32rc (fresh rc reference) + fsmk32f2 (fsdp2), then pause for parity verification.
