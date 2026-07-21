# test_throughpout_v2 — THE FOUR CROSSOVER POINTS (throughput testing target, 2026-07-19)

THE GOAL (user, canonical): per model, find exactly 4 sequence lengths that tell the
whole story — parity on not-so-long sequences, real advantage on long, sole coverage
on ultra-long. Nothing else needs measuring. Start with q3-30b-a3b.

| # | crossover definition | what asym must show |
|---|---|---|
| P1 | so-recomp fits AND so-unsloth fits AND asym-lat fits | asym PARITY (or better) |
| P2 | so-recomp OOM, so-unsloth still fits | asym PARITY with so-unsloth |
| P3 | so-unsloth OOM, so-unsloth-OFF still fits | asym (T1 or T2) BEATS its throughput |
| P4 | ALL baselines fail (incl. unsloth-OFF, G-OOM or host C-OOM) | asym fits and runs |

NOTE P3/P4 re-scope unsloth-off INTO the q3-30b test matrix (user 2026-07-19 —
supersedes the earlier "no unsloth-off" ruling, which remains in force for the other
models' banked verdicts).

## q3-30b-a3b — the 4 points: chosen seqs, status, evidence

| # | seq | so-recomp | so-unsloth | so-unsloth-OFF | asym | status |
|---|---|---|---|---|---|---|
| P1 | **80k** (b8 regime) | fits (b4-5 est; measured 64k b6 5919) | 3424 · 176.9 (96%) · b8 ✓ | (fits, slower) | **T1 3642 · 84.7 (46%) · b8 = +6% BEAT** ✓ | **DONE** (optional: 1 cheap rc@80k run to pin its cell) |
| P2 | **640k** | OOM (wall 392–400k ✓ measured) | 731 · 181.5 (98%) · b1 — its last fit ✓ | (fits, slower) | **T1 732 · 111.4 (60%) · b1 = PARITY +0.1%** ✓ | **DONE** |
| P3 | **800k** | OOM | OOM (wall 640–660k ✓ measured) | **needs 1 run** (b1; pred ~370–400 tok/s, HBM lean, RSS ~500 GB OK) | **T1 597 · 147.5 (80%) · b1 ✓** — pred +50–60% over uns-off | **1 RUN NEEDED** (tputuo-c14, 800k b1) |
| P4 | **1.1M** | OOM | OOM | pred **host C-OOM** (RSS wall ≈1.05M tokens: c12 anchor C-OOM at 131k×8 = 1.048M tok; 1.1M×1 > that) — **needs 1 bracket run** | **T2 pred 436 · ~158 (85%) · b1 — needs the pending run** | **2 RUNS NEEDED** (asym T2 1.1M — interrupted earlier; uns-off 1.1M C-OOM bracket) |

## What we already have vs need — HALF IS ALREADY BANKED
- **P1 ✓ banked** (80k: asym +6% over so-unsloth at half the memory; 120k alt: +23%).
- **P2 ✓ banked** (640k: 732 vs 731, healthy-vs-edge, −70 GiB — the parity showcase).
- **P3: 1 run needed** — so-unsloth-OFF @800k b1 (config
  superoffload_mem|unsloth-off-ohbm0|ligerloss1, tag tputuo-c14; dataset exists;
  ~2.5 h at its slow rate). asym side already measured (597).
- **P4: 2 runs needed** — (a) asym T2-BALANCED @1.1M b1 (the interrupted tputschedb
  run; dataset built; exact recipe in agent/handoffs/prompt.md validation log; ~3 h);
  (b) so-unsloth-OFF @1.1M b1 expecting host C-OOM (the "all others fail" proof;
  watch RSS vs 1693 GB — if it unexpectedly FITS, P4's seq moves right to ~1.3M and
  we re-bracket with one more probe).
- Optional polish: so-recomp @80k b4-b5 (pins P1's rc cell instead of interpolating).

## Definitions/gates (same as v1 protocol)
- "fits" = healthy ≤98% HBM, no thrash, no host watchdog; OOM rows are brackets.
- parity = |Δ tok/s| ≤ 2% (run noise ~1.5%); beat = Δ > +5%.
- All runs: PROFILERS=source, MAX_STEPS=3-4, WARMUP_STEPS=1, host-tagged RUN_NAMEs,
  archive to profiling_tp_s04-p1-dgx-02-c14/, cells in the v1 format
  (tok/s · resv GiB (%HBM) · batch).
- asym T1 stack = the 5-fix latency config (fix_asym.md §5a); T2 = staged+ker000+pins
  (no keep-acts flags, no GC override).

## P5 (added 2026-07-19): the ultra-deep MEMORY-mode capacity point
One more coarse point past T2's wall (~1.25M): a seq where even asym T2 G-OOMs and
only asym T3 MEMORY fits. Does NOT need fine granularity — one healthy fit at
~1.4M b1 shows the last crossover. (T3's leanest slope 0.119 GiB/1k → ~172 GiB @1.4M.)

## THE RUN QUEUE — exactly what's left (4 required + 1 optional)

| # | run | config (RUNS string + env) | pred | est time | for |
|---|---|---|---|---|---|
| R1 | uns-OFF @800k b1 | `superoffload_mem\|unsloth-off-ohbm0\|ligerloss1 ; 800000\|1\|1` — protocol env only, tag tputuo-c14 | TP ~370–400, RSS ~500 GB | ~2.5 h | P3 baseline |
| R2 | asym T2 @1.1M b1 | `asym_cpuadamwds\|recomp-off-full-fg-ker000-ceil0000-ohbm0\|ligerloss1 ; 1100000\|1\|1` + staged + pins, NO keep-acts flags, NO GC override, MAX_SAMPLES=512 | lat ~2520 · TP ~436 · HBM ~158 | ~3 h | P4 asym side |
| R3 | uns-OFF @1.1M b1 | same as R1 at 1100000, MAX_SAMPLES=512 | expect host **C-OOM** (RSS wall ≈1.05M tok) | ~1 h to fail | P4 "all fail" proof |
| R4 | asym T3 @1.4M b1 | `asym_cpuadamwds\|recomp-off-full-fg-ker101-ceil0000-ohbm0\|ligerloss1 ; 1400000\|1\|1` + pins only (NO staged, NO keep-acts, NO fused) — pure memory mode; s1400000 n512 dataset builds on CPU during R1 (~50 min, overlapped — free) | TP ~335 · HBM ~172 (93%) · RSS ~650 GB | ~5 h | P5 sole-coverage crown |
| ~~R5~~ | ~~so-recomp @80k~~ | PRUNED (user): rc's per-sample HBM > unsloth's — its cell adds no information; P1's "rc fits" stands as est | — | — | — |

pins env (asym runs) = ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1, FG_DA_GPU=1,
DOWN_SCATTER_BLOCK_EXPERTS=0, FG_ELEMENTWISE_CHUNK_MB=1024, DOWN_DX_STAGED=1.
protocol env (2026-07-19 rev: user — 1 warmup + 2 measured suffices, measured-step variance ~1%) = PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 DATASET_OVERWRITE=false
OVERWRITE=false (MAX_SAMPLES=512 at ≥900k, else 1024).
Serial order R1→R2→R3→R4; total ≈9.2 h at the rev protocol (1w+2m). R5 pruned.

## STANDING SEARCH RULE (user, 2026-07-19 — DO NOT STOP EARLY)
If any planned point FAILS to show its turning (asym parity/beat/sole-fit not
demonstrated at the chosen seq), the answer is LONGER SEQUENCES, not giving up:
extend the seq (steps of ~1.3x, fast builder makes datasets free) until the turning
point appears — i.e., until asym (whichever tier the scheduler emits) reaches
parity/beat there, or until the baseline is dead and asym still fits. The turning
points EXIST by construction (baseline walls are measured; asym's slopes are
leaner) — the only question each probe answers is WHERE. Concretely:
- R3 uns-OFF @1.1M FITS instead of C-OOM → move P4 right (1.3M, then 1.5M...) and
  re-bracket until uns-OFF dies while asym (T2/T3) still fits.
- R4 @1.4M edges/OOMs → step T3 back to 1.3M (still past T2's wall — P5 survives);
  if it CRUISES (<85%), optionally push 1.6M+ to widen the sole-coverage claim.
- R1 shows uns-OFF closer than pred → push P3 deeper (900k+ has asym T1 banked 519)
  until the beat margin is >5% clear of noise.

## PROGRESS LOG (queue launched 18:11 2026-07-19; chain = crossover_queue.sh)
- [18:07] s1400000 n512 dataset built in ~4 min by the NEW FAST BUILDER
  (build_lf_sft_eval_pair.py fast path: batched rust-parallel verify encodes, ~10x,
  BYTE-IDENTICAL output proven vs legacy on s160000 n1024 train+eval; sample
  manually inspected). Builder change is permanent for all future datasets.
- [18:11] R1 (uns-OFF 800k b1) STARTED under rev protocol MAX_STEPS=2 WARMUP_STEPS=1.
- [22:45 INCIDENT] R1/R2/R3 ALL CONTAMINATED and must rerun: the 17:50 R1 attempt's
  setsid torchrun SURVIVED the 18:10 pkill (project_rules trap: pkill-by-name misses
  setsid trainers; nvidia-smi showed 0 because it was still CPU-loading) and squatted
  119.94 GiB (pid 406892) through R1' (OOM 19:13), R2 (OOM 19:42), R3 (GPU-OOM 20:08 —
  NOT the intended host-C-OOM evidence). Orphan died ~20:35 after finishing its own
  contended run. R4 (started 20:08, trainer 20:33) is likely clean — its measured
  steps postdate the orphan; verify step-time spread at parse before accepting.
- LESSON (now enforced): every launch is preceded by a GPU-EMPTY GUARD
  (nvidia-smi --query-compute-apps must be empty, else abort loudly); kills are by
  GPU PID + verified empty afterwards, never pkill-by-name alone.
- RERUN QUEUE (fires automatically after R4): R2' asym T2 @1.1M -> R1' uns-OFF @800k
  -> R3' uns-OFF @1.1M (each ~2.4/2.1/1 h).
- [00:38] R4 DONE, CLEAN (orphan died ~20:35; R4's trainer + steps all after):
  **asym T3-MEMORY @1.4M b1: lat 4589.9 s/step · TP 305 tok/s · HBM 134.2 (73%) ·
  RSS 940 GB — FITS HEALTHY. P5 BANKED: at 1.4M every baseline AND asym T2 are dead;
  T3 runs at 73% HBM.** Note RSS 940 GB ≈ the ~957 GB host wall — host RAM, not HBM,
  is T3's next binding constraint (matches project_rules §1).
- [01:15] Rerun chain R2'→R1'→R3' LAUNCHED behind the GPU-empty guard.
- R2' result: (pending)  - R1' result: (pending)  - R3' result: (pending)

## After q3-30b
Same 4-point template for q3-32b and llama3.3-70b (c12's models — their P1/P2
partially exist in c12 tables + banked verdicts; do NOT re-run what the STANDING
VERDICTS already bank there).
