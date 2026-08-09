# PROMPT (standing instruction for this agent — keep at top, work below)

Run the ZeRO3-Offload and then the FSDP2-Offload baselines DIRECTLY (real
runs, not placeholders/derived bars) for the 1-rank and 2-rank throughput
plots. Do the MoE models first, then the dense ones. Probe EVERY rendered
plot column for both systems at both ranks — fit or OOM, no cell may remain
synthesized or inferred. FSDP2 has no backend yet: wire it the same way as
superoffload/zero3 (backend token in the run driver), offloading (almost)
the same state set as ZeRO3-Offload (params + gradients + optimizer to
host). Validate with a loss-parity smoke before trusting any cell. When a
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
Order: **ZeRO3 first, then FSDP2** (user 2026-08-03, supersedes the 08-02
FSDP2-first note — ZeRO3 is wiring-free and lands cells immediately; wire
FSDP2 in parallel/CPU-side while ZeRO3 runs); within each system:

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
cell list is already scripted at fillruns/chainZ.sh — run it FIRST (before FSDP2).
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
  (mixtral keeps derived bars until its runs happen — the flagged exception).
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
  (zero3) after chain Y has been DISARMED. OWNERSHIP SPLIT (user 08-03):
  the PARENT session runs ZeRO3 itself (chainZ.sh + the GLM extension)
  right after its OOM sweep + DATA flips — **this agent owns FSDP2 ONLY**
  (wiring + smoke may start immediately CPU-side; first GPU launch only
  after the parent's zero3 chain logs its DONE marker in fillruns/, and
  never concurrently — one run per node).
- Records/history on this node: fix_plot_placeholders.md (§0-§8 — the whole
  placeholder campaign incl. the max-TP-over-batch cell rule and saturation
  conventions), fix_qwen3.5_tp.md (q3.5 campaigns + the EP2 grad-offload
  port), s04-p1-dgx-02-c12/-c14 dirs (provenance of 30b/32b/llama rows).

## §8 PROGRESS LOG (append-only; every run and verdict, no exceptions)

- [2026-08-02] Doc created (parent session). FSDP2 backend NOT yet wired;
  zero3_offload_mem verified present in the driver; cell matrix frozen from
  the rendered lean columns; chain Y (existing-series OOM sweep) in flight
  on the node.
