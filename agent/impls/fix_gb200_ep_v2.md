# fix_gb200_ep_v2 — the skewness benchmark tables (micro + e2e), and how to produce them

v1 (fix_gb200_ep.md) is RETIRED as the working doc — it holds all history/receipts.
This doc owns ONE deliverable: the two tables below, filled, with paper-standard
skew injection (Zipf), no invented scenarios.

## GOALS — the tables to produce

FIXED WORKLOAD everywhere:
  model    q3-30b-a3b (Qwen3-30B-A3B: 128 experts, top-8 routing), LoRA-SFT
  micro    one expert-GEMM race, 2 GPUs, 5,120,000 rows, real bank shapes (N=768, K=2048)
  e2e      20000|8|1 per GPU, 2 GPUs, different data; 5 steps (1 warmup + 4),
           report mean of steps 2-4; all offloading on; loss-overlay checked

TABLE 1 — MICRO (each cell: wall ms / GPU-imbalance % / weight-GB streamed per GPU)

  system            | uniform | zipf 0.5 | zipf 0.8 | zipf 1.0 | zipf 1.5 | zipf 2.0 | real median | real worst
  ------------------|---------|----------|----------|----------|----------|----------|-------------|-----------
  EP default split  |         |          |          |          |          |          |             |
  EP smart split    |         |          |          |          |          |          |             |
  ours (queue)      |         |          |          |          |          |          |             |

  - zipf cells: mean + worst over 3 seeded expert-ID shuffles.
  - footnote (caption, not a header): top-8 routing cannot push one expert past
    12.5%; raw-generator columns right of ~0.8 exceed that — kept for literature
    comparability.
  - "EP smart split" = whole experts assigned to GPUs by best-possible load
    bin-packing (placement cannot split an expert). "ours" has no split to choose.

TABLE 2 — E2E (each cell: step seconds / tokens per second)

  system            | natural | zipf 0.5 | zipf 0.8 | zipf 1.0 | zipf 2.0
  ------------------|---------|----------|----------|----------|---------
  EP (owned)        |         |          |          |          |
  ours, no queue    |         |          |          |          |
  ours, queue       |         |          |          |          |
  correctness row   | loss overlay <= 0.01 on natural rows, per system

  - backends: EP (owned) = asym_ep2_cpuadamwds (carries its host-sync footnote
    until the sync-free rebuild); ours-no-queue = asym_sdp2_cpuadamwds;
    ours-queue = asym_sqdp2_cpuadamwds. (The S6 true-sEP backends sep2/sqep2 are
    a separate track, not this table.)
  - already measured: sdp2/sqdp2 natural cells (55.7-56.9 s class at 20k).
  - e2e zipf injection draws 8 DISTINCT experts per token => legal routing at
    every s (busiest expert self-saturates toward the 12.5% ceiling); no stress
    asterisks needed at e2e. z rows are timing-only (loss invalid by design).
  - the old one-hot rows (5-20%) move to the appendix as legacy stress tests.

## NEEDED CHANGES (staged; each lands + validates before the next)

C1  MICRO ZIPF GENERATOR + ID SHUFFLE (scripts/testing/ep_balance_bench.py)
    - scale_counts gains zipf mode: shares_i = (1/rank_i^s)/sum, counts = shares
      x m_total; token form in the ALPHAS list: z0, z0.5, z0.8, z1.0, z1.5, z2.0
      (z0 = the uniform column).
    - per-trial seeded permutation of expert IDs (new arg --seeds, default 3):
      rank->ID mapping shuffled per seed; report mean + worst across seeds.
    - VALIDATE (no GPU): unit check — generated counts match the formula (top1
      share at s=1.0, E=128 ~= 18.4%); shuffles preserve totals and coverage.

C2  MICRO SMART-SPLIT MODE (same file)
    - new mode "owned_smart": whole-expert LPT bin-pack over the case's counts
      (largest expert to the lighter GPU; NO chunk splitting — placement cannot
      split an expert); executes exactly like owned but with the packed map.
    - VALIDATE: unit check — uniform counts pack to bins within one expert of
      equal; a known skewed count vector packs to the known optimal bins.

C3  E2E ZIPF FIELD (scripts/lf/profile_lora_lf_test_both.sh, then _source.sh
    sync with the profiler-default flip preserved)
    - model-spec third field accepts z<s> alongside the legacy numeric alpha:
      q3-30b-a3b|2|z0.8 -> row-scoped env ASYM_EP_SKEW_ZIPF=0.8 + implicit ACK;
      run-dir label gets _zipf08 (dot stripped) so rows never collide.
    - VALIDATE: bash -n both drivers; an invalid value (|z-1) must die with the
      model-spec field error, not launch.

C4  E2E ZIPF SAMPLER (asym_gemm/training/qwen3_moe.py, _compute_routing)
    - when ASYM_EP_SKEW_ZIPF=s: per layer, ONE fixed seeded permutation of the E
      experts (seed 42 + layer name — identical on both ranks and across
      forward/recompute); weights w_i = 1/rank_i^s; REPLACE the router's picks
      with torch.multinomial(w.expand(T, E), 8, replacement=False), generator
      re-seeded per call from (42, layer) so every call of that layer draws the
      SAME picks. Timing-only; reuses the existing ASYM_EP_SKEW_ACK guard.
    - memory note: T x E fp32 weights at T=160k, E=128 ~= 82 MB — fine.
    - VALIDATE (distribution receipt): one |1 step with ASYM_EP_STATS=1 at z1.0;
      the stats json's realized shares must show the busiest expert saturating
      toward ~12.5% and the rank curve matching the saturated-Zipf prediction;
      loss finite; no shape/recompute errors.

C5  FILL TABLE 1 (after C1+C2; GPUs 2,3; ~25 min):
      HIST=profiling_both_epstats/ep_hist_q3_s20000.json \
      ALPHAS=z0,z0.5,z0.8,z1.0,z1.5,z2.0,natural \
      MODES=owned,owned_smart,queue SEEDS=3 MTOTAL=5120000 REPS=3 GPUS=2,3 \
      OUT=profiling_both_skew/table1_micro.json \
      bash scripts/testing/ep_balance_bench.sh
    (natural columns = worst + median layers of HIST, as today. Regenerate the
    histogram any time: ASYM_EP_STATS=1 on a |1 capture run — artifacts land in
    profiling_both_epstats/ automatically; ASYM_EP_STATS_PATH is an optional
    extra COPY destination, the canonical file lives inside the run dir.)

C6  FILL TABLE 2 (after C3+C4; GPUs 0,1; one invocation per system, z-rows
    chained in one RUNS string; /dev/shm cleanup before/after each).
    OUTPUT ROUTING (driver-automatic, 2026-07-10): any invocation whose RUNS rows
    carry a |<alpha> or |z<s> model field (or skew envs) is AUTO-routed to
    profiling_both_skew/ (override: SKEW_OUTPUT_ROOT); ASYM_EP_STATS=1 capture
    runs are auto-routed to profiling_both_epstats/. Plain runs use OUTPUT_ROOT
    as given — skewed, capture, and natural trees can never mix.
      OUTPUT_ROOT=$PWD/profiling_both_skew MAX_STEPS=4 WARMUP_STEPS=1 \
      PROFILERS=source ASYM_GC_SAVE_ON_CPU_OVERRIDE=false \
      ASYM_EXPACT_CPU_POOL_MAX_BYTES=96000000000 GPU_POOL=0,1 \
      RUNS='q3-30b-a3b|2|z0.5 ; asym_sqdp2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false || q3-30b-a3b|2|z0.8 ; ... || q3-30b-a3b|2|z1.0 ; ... || q3-30b-a3b|2|z2.0 ; ...' \
      bash scripts/lf/profile_lora_lf_test_both.sh
    - repeat the invocation with asym_sdp2_cpuadamwds, then asym_ep2_cpuadamwds.
    - natural cells for sdp2/sqdp2 already exist; ep2's natural row carries the
      host-sync footnote (or waits for the sync-free rebuild).

C7  ASSEMBLY: scripts/testing/print_skew_tables.py — reads profiling_both_skew/table1_micro.json +
    the e2e run dirs, prints both tables as markdown (mean/worst for zipf cells,
    steady-mean steps 2-4 for e2e cells, tokens/s = 320000 x 2? NO — tokens/step
    = seq x batch x 2 ranks = 320k; tokens/s = 320000 / step_seconds).

## PROTOCOL REMINDERS (binding; full history in v1)
  - predict-then-measure: log expected bands before each sweep; any surprise =
    stop, diagnose to a receipt, then continue. One change per run.
  - steady state = steps 2-4 of a 1+4 run. /dev/shm fabric files cleaned before
    and after every e2e invocation.
  - archive any same-config run dir before a remeasure (code edits do NOT change
    the config hash — the skip-if-done trap).
  - never edit driver scripts while a driver is running (bash reads by offset).
