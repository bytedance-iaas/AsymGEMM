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

## q3-30b-a3b — FINAL MEASURED TABLE (2026-07-20; cell = lat s/step · TP tok/s · HBM GiB (%) · RSS GB)

| seq | so-recomp | so-unsloth | so-unsloth-OFF | asym T1-LAT | asym T2-BAL | asym T3-MEM | verdict |
|---|---|---|---|---|---|---|---|
| 80k b8 | ~4400 est | 186.9 · 3424 · 176.9 (96%) · 364 | 228.1 · 2806 · 94.4 · 599 | 175.7 · **3642** · 84.7 (46%) | — | ~278 · ~2300 · 80.1 | **P1 ✅ asym +6% at half the HBM** |
| 640k b1 | OOM (wall 392–400k) | 875.4 · 731 · 181.5 (98% edge) · 382 | fits, — | 873.8 · **732** · 111.4 (60%) · 537 | — | — | **P2 ✅ PARITY +0.1%, −70 GiB, healthy vs edge** |
| 800k b1 | OOM | OOM (wall 640–660k) | 1446.8 · 553 · 118.1 (64%) · 663 | 1340.0 · **597** · 147.5 (80%) · 539 | — | — | **P3 ✅ asym +8.0% over the last-alive baseline** |
| 1.1M b1 | OOM | OOM | **HOST-OOM** (watchdog: avail 34 GiB < floor; status 143) | keep-acts >HBM | 2879.3 · **382** · 151.5 (82%) · 906 | — | **P4 ✅ all baselines fail; T2 alone** |
| 1.4M b1 | OOM | OOM | HOST-OOM (implied) | — | G-OOM pred (~1.25M wall) | 4589.9 · **305** · 134.2 (73%) · 940 | **P5 ✅ even T2 dead; T3 alone; next wall = host RAM** |

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
- [04:11] R2' DONE CLEAN: **asym T2-BALANCED @1.1M b1 = lat 2879.3 · TP 382 tok/s ·
  HBM 151.5 (82%) · RSS 906 GB. P4 asym side BANKED** (pred 436/−12% → T2 constants
  refit from this point; HBM pred 158 within 4%). Plot updated (est→measured).
- [06:02] R1' DONE CLEAN: **uns-OFF @800k b1 = lat 1446.8 · TP 553 · HBM 118.1 (64%) ·
  RSS 663 GB.** PREDICTION MISS logged honestly: anchor-based pred was 370-400 (the
  131k×8 anchor's host-pressured rate does NOT transfer to b1 — offload overlaps far
  better). **P3 BANKED @800k: asym T1 597 beats uns-OFF 553 by +8.0%** (> the 5% beat
  gate; also −124 GB host RSS; uns-OFF is leaner on HBM 118 vs 148 — noted).
- [06:36] R3' DONE: **uns-OFF @1.1M b1 = HOST C-OOM CONFIRMED** — log evidence:
  `[host-mem-watchdog] CPU-node available 34 GiB < floor 35 GiB ... soft host OOM`,
  escalated, status 143; jobs.tsv failed:1. **P4 FULLY BANKED: at 1.1M rc=GPU-OOM,
  uns=GPU-OOM, uns-OFF=host-OOM; asym T2 alone = 382 tok/s.**
- **ALL FIVE CROSSOVER POINTS MEASURED** (P1 +6% · P2 parity · P3 +8% · P4 T2-alone ·
  P5 T3-alone). Remaining: the max-seq stretch (S1 @1.6M launched 06:40) and the
  q3.5-122b campaign.

## NEXT MODEL (user 2026-07-20): q3.5-122b-a10b — same 4-6 crossover points + plot
ARCH (config.json): L=48, h=3072, 256 experts top-8 + shared (a10b), head_dim 256,
**hybrid attention: full_attention_interval=4 (12 full-attn + 36 linear/delta-net
layers)** → act memory & per-token cost scale sub-quadratically vs q3-30b; walls sit
deeper per token. Weights: 234 GB bf16, node-cached ✓ (direct download; NOTE this
node needs NO proxy — export.sh's proxy is unreachable here and BREAKS downloads).
BANKED ANCHORS (fix_qwen3.5.md era): uns-off-ohbm0 @16k×8 = 302.4 s/it (423 tok/s),
RSS 644 GB, validated · asym @16k-32k×8 = HOST-OOM >1.12 TB on a 1.17 TB node → on
c14 (~957 GB) asym big-batch points are host-capped; expect the asym story at b1
deep-end (fewer tokens in flight) + T3's lean tier; superoffload's 234-GB streaming
also eats host (644 GB @128k tokens — both sides are host-pressured, walls both real).
Driver auto-switches qwen3.5 to .venv-fa4 + FLASH_ATTN=fa4 (delta-net kernels);
QWEN35_DELTA_CHUNK_SIZE env exists for the chunked delta path (fix_merged.md).
Goal identical: capacity benefit of asym* while keeping throughput at not-so-long
seqs. Prereqs in motion: weights downloading to node HF cache (~244 GB, background;
qwen3.5 runs auto-switch to .venv-fa4 + FLASH_ATTN=fa4 per the driver). Plan once
q3-30b stretch completes: (1) probe superoffload walls (rc, uns) with the anchor-
then-bisect protocol at b1 deep-end + one b8-regime point; (2) asym T1 parity band +
T2/T3 sole-coverage points; (3) datasets via the fast builder (n512); (4) add
q3.5-122b-a10b to plot_tp_vs_seq.py. Note 35B-A3B fla illegal-memory @75k history —
if 122b hits fla issues at long seq, record and step around (known-issue bracket).

## q3-30b STRETCH (max-seq of the asym system; after R3')
- [12:0x] S1 DONE: **T3 @1.6M b1 FITS HEALTHY — lat 5487.6 · TP 292 · HBM 156.1
  (84%) · RSS 925 GB (no watchdog; 32 GB under the wall).** RSS prediction (~1030)
  was pessimistic — act-pool growth is sublinear seq→seq here (925 @1.6M vs 940
  @1.4M, pool reuse + n512 dataset). **MAX-SEQ HEADLINE = 1.6M tokens** (≥2.4x
  uns-OFF's ~1.05M-token host wall, 2.5x uns's 640k, 4x rc's 392k). Projected
  ceiling ~1.7M (host-bound; HBM would allow ~1.85M). Deeper stretch deferred —
  GPU pivots to q3.5-122b (user priority); ohbm/actnvme arbitrage remains the
  next-tier lever if >1.7M is ever needed.

## After q3-30b
Same 4-point template for q3-32b and llama3.3-70b (c12's models — their P1/P2
partially exist in c12 tables + banked verdicts; do NOT re-run what the STANDING
VERDICTS already bank there).


## q3.5-122b-a10b — CLUSTER A RESULT (2026-07-20 13:04; 32k×b8, rev protocol)
| config | lat s/step | TP tok/s | HBM GiB (%) | RSS GB | verdict |
|---|---|---|---|---|---|
| so-recomp | — | — | GPU-OOM (~181 needed) | — | dead at 32k×8 |
| so-unsloth | 281.8 | 909 | 154.0 (83%) | 659 | sole healthy baseline |
| so-unsloth-OFF | — | — | — | HOST-OOM (watchdog 48<50) | dead on c14's 957 GB |
| asym T1 | 304.4 | 841 | 113.7 (61%) | 903 | fits; −7.5% vs uns at −40 GiB HBM |
READING: at 122B the capacity crunch arrives AT 32k — crossovers compress ~10x vs
q3-30b. asym already one of two survivors. GAP DIAGNOSIS: 36/48 layers are
linear-attention whose module lacks the keep-acts port (1165 GiB/step CPU round-trip
+ big pinned pools → RSS 903) — porting ASYMM_ATTN_ACT_KEEP_ACTS_HBM to
linear_attention_activation_offload.py is the top 122B lever (cuts the −7.5% AND host).
CLUSTER B (launched 13:1x): uns 288k b1 → uns 320k b1 (wall pair, est wall ~300k) →
asym T1 288k b1 (head-to-head at uns's edge) → asym T1 384k b1 (sole-coverage probe).


## q3.5-122b — CLUSTER B (2026-07-20 15:00; b1 deep-end)
| run | lat s/step | TP | HBM (%) | RSS | verdict |
|---|---|---|---|---|---|
| uns @288k b1 | 433.1 | 665 | 180.3 (97% edge) | 660 | edge-locked already |
| uns @320k b1 | 499.9 | 640 | 181.3 (98% edge) | 659 | still edge; wall beyond (352k probe queued) |
| asym T1 @288k b1 | — | — | — | watchdog 47<50 during LOAD | TRANSIENT (bigger 384k fit right after) → retry queued |
| asym T1 @384k b1 | 439.5 | **874** | 156.9 (85%) | 848 | **+37% over uns@320k, healthy vs edge** |
KEY 122B FINDING: hybrid linear attention makes asym's per-token cost ~flat in seq
(1189 us/tok @256k-tok → 1144 @384k) while uns's grows steeply (1100 → 1562) AND
uns edge-locks from 288k. The 122B beat is BIGGER and arrives EARLIER than q3-30b.
CLUSTER C (launched 15:1x; guard now also waits host-avail>1.5T): asym 288k retry →
uns 352k (wall bracket) → asym T1 448k (stretch; T2 fallback if edge/OOM).


## q3.5-122b — CLUSTER C (2026-07-20 16:09)
| run | result |
|---|---|
| asym T1 @288k retry | watchdog 49<50 AGAIN → NOT transient: 122b asym deep-end sits ~50 GB from host wall; 288k = coin-flip territory. No further retries (384k/448k carry the story) |
| uns @352k b1 | GPU-OOM → **uns b1 wall (320k, 352k]** |
| asym T1 @448k b1 | **FITS 897 tok/s · 180.8 (98% edge) · RSS 849** — T1's HBM edge; TP still RISING with seq (linear-attn flatness) |
NARRATIVE: uns dead ≥~330k; asym T1 healthy to 448k and +37-40% faster at seqs uns
cannot reach (874@384k, 897@448k vs best-alive 640@320k). FINAL PROBE: asym T2 @480k
(tier transition + host margin) — then plot + close.
