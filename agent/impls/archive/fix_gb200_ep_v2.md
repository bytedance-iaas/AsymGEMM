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
FILLED 2026-07-10, RE-BANKED same day under the final grid-aware chunk policy
(artifact profiling_results/profiling_both_skew/table1_micro.json; assembled by
scripts/testing/print_skew_tables.py; zipf wall = mean/worst over 3 ID shuffles)

  system            | uniform        | zipf 0.5       | zipf 0.8       | zipf 1.0       | zipf 1.5       | zipf 2.0       | real median   | real worst
  ------------------|----------------|----------------|----------------|----------------|----------------|----------------|---------------|---------------
  EP default split  | 15.2/15.3 · 1% · 0.20 | 15.7/16.5 · 9% · 0.79 | 17.0/18.2 · 27% · 1.05 | 18.5/19.6 · 40% · 1.21 | 22.2/23.5 · 66% · 1.58 | 24.5/26.0 · 82% · 1.83 | 14.9 · 6% · 0.80 | 17.4 · 25% · 0.96
  (EP e2e baseline DETOXED 2026-07-11 — agent/impls/fix_ep.md: 20k natural
  117.5 -> 93.3 s = inside its own z-floor band; Table 2 EP natural cell and
  the ours-vs-EP ratio (2.04x -> 1.62x) updated accordingly below.)
  EP smart split    | 15.2/15.2 · 0% · 0.20 | 15.4/15.8 · 4% · 0.41 | 15.5/16.2 · 5% · 0.53 | 15.3/16.1 · 6% · 0.56 | 15.5/16.3 · 5% · 0.65 | 16.5/16.6 · 26% · 0.73 | 16.1 · 9% · 0.33 | 15.6 · 4% · 0.35
  ours (plan)       | 15.2/15.2 · 0% · 0.20 | 15.5/15.8 · 4% · 0.41 | 15.5/16.3 · 5% · 0.53 | 15.4/16.2 · 6% · 0.56 | 15.5/16.5 · 5% · 0.65 | 15.2/15.8 · 11% · 0.96 | 16.1 · 9% · 0.33 | 15.6 · 4% · 0.35
  ours (queue)      | 14.6/14.6 · 4% · 0.20 | 15.4/15.6 · 3% · 0.44 | 16.3/16.5 · 7% · 0.67 | 15.9/16.3 · 4% · 0.79 | 16.1/16.4 · 1% · 0.97 | 18.2/19.0 · 7% · 1.07 | 15.5 · 3% · 0.43 | 15.4 · 6% · 0.38

TABLE 1b — MICRO, EXPERTS BLOCK (gate+up GEMMs + SiLU*mul + down GEMM; all banks
streamed; same cells). FILLED 2026-07-10 (profiling_results/profiling_both_skew/table1b_experts.json)

  system            | uniform        | zipf 0.5       | zipf 0.8       | zipf 1.0       | zipf 1.5       | zipf 2.0        | real median   | real worst
  ------------------|----------------|----------------|----------------|----------------|----------------|-----------------|---------------|---------------
  EP default split  | 52.5/52.7 · 1% · 0.59 | 56.5/58.9 · 12% · 2.38 | 62.9/67.3 · 28% · 3.16 | 68.2/73.4 · 39% · 3.63 | 82.8/87.2 · 64% · 4.74 | 93.5/98.4 · 79% · 5.48 | 53.8 · 4% · 2.41 | 61.6 · 26% · 2.89
  EP smart split    | 53.3/53.6 · 0% · 0.59 | 53.9/54.4 · 2% · 1.23 | 54.0/55.6 · 2% · 1.59 | 54.1/55.9 · 4% · 1.67 | 54.2/56.6 · 3% · 1.95 | 62.9/63.5 · 31% · 2.20 | 56.0 · 8% · 1.00 | 53.4 · 1% · 1.04
  ours (plan)       | 53.4/54.3 · 0% · 0.59 | 53.9/54.6 · 2% · 1.23 | 54.4/56.1 · 3% · 1.59 | 54.3/55.5 · 4% · 1.67 | 54.4/56.5 · 4% · 1.95 | 55.4/56.0 · 8% · 2.88 | 56.0 · 7% · 1.00 | 53.7 · 0% · 1.04
  ours (queue)      | 51.3/52.8 · 2% · 0.62 | 53.0/53.4 · 0% · 1.36 | 54.9/55.5 · 1% · 1.88 | 58.1/59.7 · 9% · 2.39 | 60.3/61.8 · 5% · 3.02 | 66.4/67.7 · 8% · 3.25 | 53.8 · 1% · 1.15 | 52.6 · 0% · 1.10

TABLE 1c — MICRO, MOE BLOCK (+ router + token gather + weighted combine; combine
charged at m/2 rows per rank in EVERY mode — post-return combine is balanced in
all real systems; no cross-GPU token movement, favors EP). FILLED 2026-07-10
(profiling_results/profiling_both_skew/table1c_moe.json)

  system            | uniform        | zipf 0.5       | zipf 0.8       | zipf 1.0        | zipf 1.5        | zipf 2.0        | real median   | real worst
  ------------------|----------------|----------------|----------------|-----------------|-----------------|-----------------|---------------|---------------
  EP default split  | 85.9/85.9 · 0% · 0.59 | 90.3/93.0 · 8% · 2.38 | 97.8/102.6 · 21% · 3.16 | 103.5/109.4 · 30% · 3.63 | 119.7/125.7 · 52% · 4.74 | 131.1/136.8 · 66% · 5.48 | 87.1 · 3% · 2.41 | 95.9 · 18% · 2.89
  EP smart split    | 87.6/88.4 · 1% · 0.59 | 87.3/87.8 · 1% · 1.23 | 88.0/89.0 · 2% · 1.59 | 87.7/89.3 · 2% · 1.67 | 88.3/90.1 · 3% · 1.95 | 97.3/97.6 · 22% · 2.20 | 90.1 · 5% · 1.00 | 87.4 · 0% · 1.04
  ours (plan)       | 87.5/88.0 · 1% · 0.59 | 87.7/88.0 · 1% · 1.23 | 87.8/89.3 · 1% · 1.59 | 87.9/89.2 · 2% · 1.67 | 87.8/89.9 · 2% · 1.95 | 88.7/89.7 · 5% · 2.88 | 90.0 · 5% · 1.00 | 87.4 · 1% · 1.04
  ours (queue)      | 84.3/84.4 · 0% · 0.60 | 86.2/86.6 · 0% · 1.33 | 88.6/89.3 · 1% · 1.93 | 88.6/89.9 · 0% · 2.33 | 92.2/93.6 · 1% · 2.99 | 97.4/97.5 · 0% · 3.22 | 87.9 · 0% · 1.18 | 85.9 · 0% · 1.10

TABLES 1d-1l — MULTI-MODEL MICRO (same modes/columns, pure-z only — no real-trace
capture exists for these models). FILLED 2026-07-10; artifacts
profiling_results/profiling_both_skew/table1_{q3235b,q35122b,l4scout}_{gemm,experts,moe}.json;
full cell grids render via scripts/testing/print_skew_tables.py. Geometry
verified from the HF configs; workload row-counts scale to fit HBM.
Summary (mean wall ms · imbalance, uniform -> zipf 2.0):

  (4-MODE RE-RUN 2026-07-10 late: sep = "ours (plan)" added everywhere; grids
  below are the current jsons. plan gets PERFECT counts in the bench — e2e it
  plans from stale counts; the queue needs none.)
  q3-235b-a22b (128E top-8, N=1536 K=4096, 3.84M rows)
    scope    | EP default          | EP smart            | ours (plan)          | ours (queue)
    gemm     | 43.4 · 1% -> 80.3 · 77%  | 43.5 · 2% -> 69.1 · 46% | 43.6 · 1% -> 55.8 · 15% | 42.7 · 1% -> 55.7 · 0%
    experts  | 138.3 · 2% -> 266.4 · 75% | 143.8 · 2% -> 199.2 · 39% | 142.9 · 1% -> 161.5 · 6% | 140.7 · 1% -> 182.4 · 2%
    moe      | 187.5 · 0% -> 315.3 · 69% | 189.4 · 1% -> 249.9 · 32% | 189.7 · 2% -> 210.4 · 4% | 185.8 · 0% -> 228.6 · 0%
  q3.5-122b-a10b (256E top-8, N=1024 K=3072, + shared expert N=1024, 5.12M rows)
    gemm     | 28.1 · 2% -> 50.7 · 55%  | 28.3 · 2% -> 32.8 · 8%  | 28.4 · 1% -> 36.1 · 26% | 29.0 · 8% -> 43.4 · 3%
    experts  | 92.3 · 2% -> 170.3 · 54% | 93.8 · 1% -> 109.5 · 13% | 95.9 · 4% -> 114.8 · 24% | 92.2 · 0% -> 136.2 · 2%
    moe      | 150.3 · 1% -> 227.9 · 47% | 150.3 · 0% -> 168.5 · 12% | 150.5 · 1% -> 171.1 · 15% | 149.3 · 0% -> 191.0 · 0%
  llama4-scout (16E TOP-1, fused gate_up, N=8192 K=5120, + shared expert N=8192, 1.28M rows)
    gemm     | 196.6 · 1% -> 327.8 · 79% | 196.9 · 1% -> 265.5 · 46% | 196.7 · 0% -> 210.6 · 6% | 196.4 · 2% -> 200.7 · 0%
    experts  | 332.7 · 1% -> 537.8 · 79% | 334.0 · 1% -> 436.2 · 46% | 332.9 · 1% -> 346.9 · 5% | 327.1 · 3% -> 332.1 · 1%
    moe      | 494.7 · 2% -> 703.5 · 62% | 493.7 · 1% -> 597.4 · 34% | 494.4 · 1% -> 507.1 · 3% | 485.3 · 1% -> 492.8 · 0%

  - scout is the cleanest separation: at top-1 a 63%-hot expert is ONE
    un-splittable placement unit, so even the oracle hits a 41% structural floor
    (measured 45-46% imb, +32% wall) while the queue stays FLAT across the whole
    dial (196-201 / 326-334 / 484-495 ms); its big mode-flat shared expert then
    dilutes everyone's imbalance at block scope — llama4's own design point.
  - q3.5 caveat (honest limitation, receipts in RUN LOG): with 256 tiny experts
    and 6 MB banks, fine steal grain pays ~25-30% scheduling/re-stream at
    s>=1.5 — queue 43.3 vs oracle 32.9 at gemm z2.0 (still 1.19x faster than
    realizable EP, imbalance 1% vs 55%). Coarsening the grain closes the wall
    gap but costs balance; kept the balance-first grain.

  - zipf cells: mean + worst over 3 seeded expert-ID shuffles.
  - footnote (caption, not a header): top-8 routing cannot push one expert past
    12.5%; raw-generator columns right of ~0.8 exceed that — kept for literature
    comparability.
  - "EP smart split" = whole experts assigned to GPUs by best-possible load
    bin-packing (placement cannot split an expert). "ours" has no split to choose.

TABLE 2 — E2E (each cell: step seconds / tokens per second)
FILLED 2026-07-10 (run dirs in profiling_results/profiling_both_skew/; steady = mean steps 2-4)

  system            | natural       | zipf 0.5      | zipf 0.8      | zipf 1.0      | zipf 2.0
  ------------------|---------------|---------------|---------------|---------------|---------------
  EP (owned, detoxed)| 93.3 s · 3430 | 81.0 s · 3952 | 84.7 s · 3778 | 87.5 s · 3657 | 93.2 s · 3433
  ours, no queue    | 57.8 s · 5540 | 58.1 s · 5512 | 59.4 s · 5389 | 60.9 s · 5257 | 62.6 s · 5115
  ours, queue       | 57.7 s · 5543 | 58.2 s · 5498 | 59.9 s · 5345 | 60.7 s · 5272 | 62.4 s · 5127
  correctness row   | natural losses 1.5465(EP-detoxed) / 1.5446 / 1.5457 — overlay <= 0.01 PASS
  - EP natural was 117.5 s with the host-sync stagger; DETOXED 2026-07-11
    (fix_ep.md: reorder + comm-stream + pad context + damper; Megatron-parity
    audited) to 93.3 s — INSIDE its own z-floor band 81-93, i.e. the baseline
    now runs at its measured stagger-free ceiling. z rows unchanged (z-control
    receipt -1.0%). Ours-vs-EP at natural: 1.62x (honest, vs 2.04x pre-detox).
  - residual note: at larger seq a size-dependent residual remains (32k: 149.1
    vs 136.5 gate) — GC recompute doubles EP's collectives per bwd layer, a
    structural cost Megatron's non-checkpointed dispatch never pays.

  - backends: EP (owned) = asym_ep2_cpuadamwds (carries its host-sync footnote
    until the sync-free rebuild); ours-no-queue = asym_sdp2_cpuadamwds;
    ours-queue = asym_sqdp2_cpuadamwds. (The S6 true-sEP backends are a separate
    track, not this table — NAMING EPOCH 4 2026-07-10: asym_sepqueue2 =
    counter-raced steal (legacy sep2/sqep2/sqeq2 alias here), asym_sepplan2 =
    count-computed cut, ASYM_EP_SEP_MODE=plan.)
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
      HIST=profiling_results/profiling_both_epstats/ep_hist_q3_s20000.json \
      ALPHAS=z0,z0.5,z0.8,z1.0,z1.5,z2.0,natural \
      MODES=owned,owned_smart,plan,queue SEEDS=3 MTOTAL=5120000 REPS=3 GPUS=2,3 \
      OUT=profiling_results/profiling_both_skew/table1_micro.json \
      bash scripts/testing/ep_balance_bench.sh
    (natural columns = worst + median layers of HIST, as today. Regenerate the
    histogram any time: ASYM_EP_STATS=1 on a |1 capture run — artifacts land in
    profiling_results/profiling_both_epstats/ automatically; ASYM_EP_STATS_PATH is an optional
    extra COPY destination, the canonical file lives inside the run dir.)

C6  FILL TABLE 2 (after C3+C4; GPUs 0,1; one invocation per system, z-rows
    chained in one RUNS string; /dev/shm cleanup before/after each).
    OUTPUT ROUTING (driver-automatic, 2026-07-10): any invocation whose RUNS rows
    carry a |<alpha> or |z<s> model field (or skew envs) is AUTO-routed to
    profiling_results/profiling_both_skew/ (override: SKEW_OUTPUT_ROOT); ASYM_EP_STATS=1 capture
    runs are auto-routed to profiling_results/profiling_both_epstats/. Plain runs use OUTPUT_ROOT
    as given — skewed, capture, and natural trees can never mix.
      OUTPUT_ROOT=$PWD/profiling_results/profiling_both_skew MAX_STEPS=4 WARMUP_STEPS=1 \
      PROFILERS=source ASYM_GC_SAVE_ON_CPU_OVERRIDE=false \
      ASYM_EXPACT_CPU_POOL_MAX_BYTES=96000000000 GPU_POOL=0,1 \
      RUNS='q3-30b-a3b|2|z0.5 ; asym_sqdp2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false || q3-30b-a3b|2|z0.8 ; ... || q3-30b-a3b|2|z1.0 ; ... || q3-30b-a3b|2|z2.0 ; ...' \
      bash scripts/lf/profile_lora_lf_test_both.sh
    - repeat the invocation with asym_sdp2_cpuadamwds, then asym_ep2_cpuadamwds.
    - natural cells for sdp2/sqdp2 already exist; ep2's natural row carries the
      host-sync footnote (or waits for the sync-free rebuild).

C7  ASSEMBLY: scripts/testing/print_skew_tables.py — reads profiling_results/profiling_both_skew/table1_micro.json +
    the e2e run dirs, prints both tables as markdown (mean/worst for zipf cells,
    steady-mean steps 2-4 for e2e cells, tokens/s = 320000 x 2? NO — tokens/step
    = seq x batch x 2 ranks = 320k; tokens/s = 320000 / step_seconds).

## MEGATRON-LM / DEEPEP DESIGN NOTES (read 2026-07-10; cites = their repo files)
Facts that matter to us, from a full read of megatron-lm/megatron/core/transformer/moe
+ DeepEP (token_dispatcher.py=td, experts.py=ex, fused_a2a.py=fa, deep_ep legacy=dep):
1. OUTPUT RETURN IS STANDARD: combine = a second all_to_all over the EP group
   (td:837-843); AllGather variant reduce-scatters back (td:334-347). Fresh output
   buffers EVERY call (mappings.py:435-444) — NCCL takes raw pointers, no
   registration, no visibility code. That's the luxury we don't have.
2. THEY PAY OUR HOST-SYNC TAX TOO — a2a needs HOST split sizes, and grouped GEMM
   takes tokens_per_expert as a HOST LIST (.tolist(), ex:658/1234). Their cure is
   a SCHEDULE, not avoidance: DtoH launched EARLY + non-blocking on a dedicated
   side stream (cuda_dtoh_point, default before_permutation_1), the blocking sync
   DEFERRED to the latest viable point (cuda_sync_point priority ladder
   before_permutation_1 < before_ep_alltoall < before_permutation_2 <
   before_finish < no_sync; td:431-451, 893-932). Static-shape modes delete the
   sync entirely (drop_and_pad capacity td:425-514; DeepEP num_worst_tokens
   dep:354; HybridEP static budget) = their CUDA-graph paths.
   -> LESSON for our armed hook: split the offsets DtoH (issue non-blocking at
   pad time, side stream) from the .tolist() consume (sync right before the
   header build) — Megatron-style early-copy/late-sync instead of inline .cpu().
3. NO RUNTIME WORK REBALANCING EXISTS IN THEIR STACK. Balance is (a) prospective
   only — aux losses (rt:286-425), sinkhorn, expert-bias nudging FUTURE routing
   (mu:1079-1109); (b) destructive — capacity dropping (mu:901-966); or
   (c) rerun-the-step on overflow (paged_stash ps:1443-1486). NOTHING moves work
   between ranks inside a step. Our steal/plan queue is outside their design
   space — which is BOTH the novelty claim and the reason we hit transport
   issues they never face (their tokens go to fixed owners; imbalance is eaten).
4. DEEPEP IS THE EXISTENCE PROOF FOR OUR TRANSPORT SHAPE: when they bypass NCCL
   they do exactly what we do — persistent registered buffers + IPC-handle/
   NVSHMEM-id exchange at init (dep:66-135), sized from config HINTS with
   grow-only realloc (fa:47-68), CPU-waits-GPU completion signaling (fa:106-107,
   explicitly not CUDA-graph safe) + event chaining. Our ring slots + host flags
   are the same genus; DeepEP just buried it in C++. Their hint-based buffer
   sizing = the pattern our SLOT_ROWS right-sizing converged to.
5. OVERLAP TRICKS worth stealing later: shared expert runs on its own stream
   INTERLEAVED between the two a2a's via a 5-state machine (se:96-188, td:672-697);
   delayed wgrad on a dedicated stream during backward (ml:737-798); paged_stash
   keeps token counts ON DEVICE with fused Triton freelist kernels (no host sync,
   ps:129-375) and prefetches reloads in the PREVIOUS layer's backward.
6. WEIGHTS NEVER LEAVE THE GPU anywhere in their stack (ex:1002; offload paths
   move activations only) — host-streamed expert weights have no upstream recipe;
   AsymGEMM's premise is genuinely uncovered ground.

## STATUS PAIR — 60000|8|1, 2 GPUs (user-set headline; FILLED 2026-07-10,
## steady = 1w+4m drop first/last measured -> mean of middle 2):
##   sdp2      228.8 s/step · 4196 tok/s   losses 1.6094..1.2971
##   sepplan2  227.5 s/step · 4219 tok/s   losses 1.6099..1.2963
##   VERDICT: PARITY (sepplan2 -0.6%, within noise); loss overlay max delta
##   0.0056; sepplan2 stats 0 armed / 1200 declined (60k is decline regime by
##   the MAX_MPE rule) — the sep machinery tax is INVISIBLE at this step size
##   with the right-sized slots. OPEN NOTE: both rows sit ~7% above the
##   2026-07-09 banked 60k class (213.4-215.3) — same-day pairs are the valid
##   comparison; the day-over-day drift is unattributed (node state/thermals/
##   code accretion) and worth one nsys look if it persists.
##   SIDE DATA: the mistaken 60000|8|2 (ga=2) sdp2 run measured 448.7 s/step —
##   and forced a REAL fix banked on the way: the EP-family manual grad
##   allreduce hard-rejected ga>1; it now syncs ONCE on the accumulation
##   boundary via accelerator.sync_gradients (DDP no_sync->sync semantics).

## FULL 4x4x4 MICRO MATRIX (2026-07-13; user-set deliverable): 4 models x
## {gemm, experts block, MoE block, WHOLE LAYER (attention seq20k + MoE)} x
## {EP owned, DP, plan, queue}, full zipf dial, 3 seeds. Artifacts:
## profiling_results/profiling_both_skew/table1_* (+ *_layer.json); renders via
## print_skew_tables.py (rows now EP/DP/plan/queue; oracle dropped per
## 2026-07-10 decision). Attention + router + combine are mode-flat by
## construction (every scheme shards attention by tokens). Layer-scope
## headline (z2.0 mean wall): q3-30b EP 155.7 vs plan 114.3 (+36%); 235b
## 367.5 vs 264.1 (+39%); q3.5 271.7 vs 214.1 (+27%); scout 908.2 vs queue
## 696.8 (+30%). DP loses to plan by 5-11% on 235b/q3.5 mid-skew (streaming
## economy) and ties elsewhere. Scout layer OOM'd whole-batch attention ->
## chunked to 4-seq groups (same FLOPs), receipted.

## OPTIMIZATION LADDER (proposed 2026-07-10; each rung is GATED — B1
## predict-then-measure, ONE rung per run, keep only what passes its gate,
## revert + log what fails. Baselines to beat, q3-30b 2 GPUs steady:
## 20000|8|1 sqdp2 57.7 s / sepqueue2 61.4; 2048|8|1 sepqueue2 armed 9.7 s
## vs 8.2 declined.)

O1  ARMED-HOOK EARLY-COPY/LATE-SYNC (Megatron td:893 pattern)
    Change: issue the offsets/experts DtoH non-blocking on a side stream at
    pad time; consume (.tolist) only right before the header build.
    Expect: armed 2k 9.7 -> ~9.0-9.3; no decline-regime change.
    GATE: 2k armed >=0.4 s faster AND 20k unchanged +-0.3 AND probe bitwise
    PASS both modes. Cost: small (frozen_linear + ep_sep touch).

O2  X-STAGE SKIP WHEN THE CUT DOESN'T CROSS (plan mode first)
    Change: in plan mode, when k_seg snaps to n_own there are NO cross rows —
    skip the X staging copy AND the peer-done coupling entirely (still consume
    the seq + publish a "local" flag so alignment holds). Queue variant later
    (needs a cheap crossing predictor).
    Expect: the ~1.5 s natural-balance arming cost mostly vanishes: plan 2k
    -> ~8.3-8.6 (the declined floor), skew behavior untouched.
    GATE: plan 2k within 0.3 s of the declined floor AND the 2k+zipf armed win
    (O6) preserved AND probe extended with a snap case, bitwise PASS.

O3  RESIDUAL SEP DECLINE-TAX HUNT (~3.7 s at 20k: 61.4 vs 57.7)
    Change: (a) auto-size slots per workload: rows = min(padded workload m,
    163840) — 20k drops to ...? no: 20k pads to 1.28M -> stays 163840; the
    remaining suspects are the 10.7 GB slots themselves + ctrl page + hook
    remnant. Isolate with SLOT_ROWS=8192 (arming effectively off) vs sqdp2:
    the delta that remains is NOT slot bytes.
    Expect: taxonomy of the residual; possibly 20k sepqueue2 -> <=59.
    GATE: keep any change only if 20k <=60.0 with 2k arming intact (armed
    1200/1200 stats line).

O4  [DONE 2026-07-11 — agent/impls/fix_ep.md; 20k 117.5 -> 93.3 s = its own
    z-floor band; 32k residual receipted/structural; detox config = asym_ep2
    defaults] VANILLA-EP DETOX VIA DEFERRED SYNC (baseline fairness)
    Change: apply the same early-copy/late-sync to ep_vanilla's dispatch host
    reads (its natural-row stagger seeds from habitual per-layer syncs).
    Expect: ep2 natural 117.5 -> toward its z-row base 81-85.
    GATE: ep2 natural <=90 with loss overlay intact. (Strengthens the paper:
    our win over EP must not depend on a gimped baseline.)

O5  DIRECT NVLINK P2P OUTPUT RETURN (DeepEP-style, big rung)
    Change: persistent IPC-exported GPU staging pool per rank; the thief
    writes stolen D GPU->GPU over NVLink; owner gathers with a local device
    copy (host bounce only for flags). X path can follow.
    Expect: armed-regime steal latency down; matters most under skew at 2k
    class and for future >=4-GPU scaling.
    GATE: 2k+z1.0 sepqueue2 wall >=5% faster than the host-bounce path AND
    probe bitwise PASS AND no decline-regime regression. If <5%, revert and
    keep the host fabric (simplicity wins).

O6  SKEW PAYOFF STUDY (not an optimization — the pending S6 demonstration)
    Run: 2048|8|1 (armed regime) x {natural, z0.8, z1.5, z2.0} x
    {sdp2, sqdp2, sepqueue2, sepplan2}; plus one 60k-class natural sanity row.
    Expect: sep* ~= sdp at natural (post O2), sep* WINS under z (steal absorbs
    per-rank imbalance that token-sharded DP cannot see... NOTE: e2e zipf
    injection gives BOTH ranks identical loads — per-rank imbalance needs the
    one-hot α legacy field or natural data skew; design the study around
    UNEQUAL per-rank load, e.g. α on ONE rank or length-skewed shards).
    GATE for the table: sepqueue2 beats sdp2 by >=5% on at least one honest
    skewed setting; otherwise the sep track is documented as
    parity-plus-optionality, not a win.

O7  STREAM-OVERLAP POLISH (low priority)
    Shared-expert-on-side-stream (q3.5/scout backends), delayed LoRA wgrad.
    Expect: small; LoRA grads are tiny.
    GATE: >=2% step win on a shared-expert model else drop.

## PROTOCOL REMINDERS (binding; full history in v1)
  - predict-then-measure: log expected bands before each sweep; any surprise =
    stop, diagnose to a receipt, then continue. One change per run.
  - steady state (UPDATED 2026-07-10, user-set): 1 warmup + 4 measured steps
    (MAX_STEPS=5 WARMUP_STEPS=1); DROP the first and last measured step (first
    still carries allocator/cache warm-in; the last interval ends at trainer
    teardown) => steady = mean of the middle 2. Tables 1-2 above were banked
    under the older mean-of-steps-2-4 rule (1+3) — comparable class, do not
    mix within one table. /dev/shm fabric files cleaned before and after every
    e2e invocation.
  - STATUS WORKLOAD (user-set 2026-07-10): q3-30b-a3b 60000|8|2 on 2 GPUs
    (1.92M tokens/step) replaces 20000|8|1 as the headline e2e row.
  - archive any same-config run dir before a remeasure (code edits do NOT change
    the config hash — the skip-if-done trap).
  - never edit driver scripts while a driver is running (bash reads by offset).

## RUN LOG (append-only)
2026-07-10 TABLE-1 FILLED (4 iterations, artifact profiling_results/profiling_both_skew/table1_micro.json):
  (r1) wrapper dropped MODES -> silently ran default modes; env passthrough added.
  (r2) owned_smart 35-183 ms from z0.8 up: LPT hands rank0 few/one segment(s) ->
  kernel grid (n_block, segment) starvation (~12 CTAs on 148 SMs) — a kernel-shape
  artifact, not placement cost; added _chunk_local (intra-rank hot-segment
  sub-tiling, an execution courtesy every mode gets).
  (r3) all cells sane EXCEPT owned_smart z2.0 still 183.6 ms: avg-based chunk
  threshold can't fire on a ONE-segment list (rank0 = only the 61% mega-expert;
  B=[3.1, 1031.8] MB receipt). Fix: force = len(segs) < 24 chunks any segment >
  HOT_CHUNK.
  (r4, FINAL) owned_smart z2.0 18.1-18.6 ms, time-imb 0.17-0.21 (row floor 0.363
  stands — loads 3.13M/1.99M; wall imb lands lower because the chunked mega-expert
  side re-streams one bank per chunk and runs at better per-row efficiency,
  B=[1201.7, 1031.8] MB). VERDICT vs pre-run expectations: z0 all ~15 ms imb<=5% OK;
  owned degrades 15.4->24.4 ms (worst-seed imb 0.88) OK; smart flat to z1.5 then
  19% imb at z2.0 OK (structural floor, in TIME cheaper than predicted); queue flat
  14.8-18.6 ms / imb<=12% across the ENTIRE dial OK. Real cols reproduce the banked
  class. Assembled markdown: scripts/testing/print_skew_tables.py.
2026-07-10 C3 LANDED + VALIDATED: z<s> third field in both drivers (parse -> row env
  ASYM_EP_SKEW_ZIPF + implicit ACK + mutual-exclusion die with one-hot; label
  _zipf<s-no-dot>; run_env passthrough). bash -n both OK; |z-1 dies with the
  model-spec error pre-launch; _source.sh synced surgically (PROFILERS default
  preserved, live scratch rows untouched).
2026-07-10 C4 LANDED + UNIT-VALIDATED: _EP_SKEW_ZIPF in qwen3_moe._compute_routing
  (per-layer seed-42 ID shuffle; multinomial 8-distinct; generator re-seeded per
  call -> fwd==recompute; stats dump records skew_zipf + loss_invalid). Unit: raw
  top1 weight 18.41% == formula at s=1; realized slot share 10.4% (saturating
  toward the 12.5% cap exactly as predicted); re-seeded draws identical; 8 distinct
  per row; per-layer shuffles differ. E2E CAPTURE VALIDATION PASS (z1.0, |1, stats
  on, 2 steps): all 48 layers realize top1 = 10.42-10.46% vs the saturated-Zipf
  prediction 10.43% (raw 18.4% compressed by 8-distinct draws toward the 12.5%
  cap); rank curve [10.45, 7.55, 5.74, 4.60]% == unit draw; ep_hist.json records
  skew_zipf=1.0 + loss_invalid=true; loss finite (9.50->9.14, garbage by design);
  no shape/recompute errors; run auto-routed to profiling_results/profiling_both_epstats/ with label
  asym_cpuadamwds_zipf10 (stats routing wins over skew routing for capture runs —
  capture timings are never quotable).
2026-07-10 TABLE-2 EXPECTATIONS (pre-run, B1): sdp2/sqdp2 FLAT across every z
  (55.5-58 s class) — token-sharded ranks; z changes segment SHAPES inside each
  GPU's grouped GEMM, not its token count or the 128-bank stream. ep2 inflates
  with s (hot experts land on one owned half; per-layer hot-half share ~52/57/60/
  80% at z0.5/0.8/1.0/2.0): bands z0.5 61-66 s, z0.8 63-70, z1.0 66-75, z2.0
  78-95; natural ~58-63 (prior 60.7). Loss overlay checked on natural rows only
  (z rows loss-invalid by design). Natural cells REMEASURED here (old parity runs
  lived in the deleted stale trees).
2026-07-10 TABLE-2 FILLED (15 runs, 3 ladders, ~2.5 h; artifacts
  profiling_results/profiling_both_skew/.../qwen3-30b-a3b__gpus2__b8_s20000_ga1_w1_s4_r64_a16_drop000/).
  ROUTE BUG + FIX: the 15 runs first landed in profiling_results/profiling/ — the skew auto-route
  tested "${RUNS:-}", but the driver rebuilds RUNS as an ARRAY before that block,
  so only row 1 (natural) was tested. Fix: match on "${RUNS[*]:-}" (both drivers,
  synced); validated — natural-first multi-row RUNS now prints "skew experiment
  detected -> profiling_results/profiling_both_skew" under DRY_RUN=true. Run dirs moved (mv, model-dir
  level) into profiling_results/profiling_both_skew/; nothing else in profiling_results/profiling/ touched.
  VERDICTS vs expectations: ours-no-queue/queue FLAT-ISH 57.7 -> 62.6 s (+8% at
  z2.0, all of it fwd+bwd expert-GEMM segment-shape cost — identical with and
  without queue, opt flat; band was 55.5-58, actual base 57.7-57.8 OK, the +8%
  tail is real and receipted). ep2 z rows 81.0-93.2 vs bands 61-95: bands were
  built on the WRONG natural base (the banked "T_ep2 60.7 s" predates the naming
  epoch — that row was the shared-bank system, not vanilla owned-EP). ep2 NATURAL
  117.5 s = the documented vanilla stagger (bwd 102.4 vs ours 42.8, fwd 12.0
  normal — fwd self-heals, bwd staggers, v1 nsys receipts); z rows DODGE the
  stagger because injected picks are identical on both ranks (same seed/T) ->
  collectives stay phase-locked; what remains is true owned-EP cost: bwd 65.8 ->
  75.3 s as the per-layer hot half grows. Even ep2's best case (81.0 s) is 1.40x
  ours; at natural 2.04x; at z2.0 1.49x.
  LOSS OVERLAY (natural rows): 1.5443 / 1.5446 / 1.5457 (ep2/sdp2/sqdp2), spread
  0.0014 <= 0.01 PASS — matches the banked 1.5446 class.
  C7 LANDED: scripts/testing/print_skew_tables.py assembles both tables (micro
  json + e2e run dirs; tokens/s = 320000/step_s).
2026-07-10 TABLES 1b/1c ADDED (user ask): bench gains --scope gemm|experts|moe.
  experts = gate+up GEMMs + SiLU*mul + down GEMM (three banks streamed; B x3);
  moe = + router (own half tokens, mode-flat) + gather (per EXECUTED rows — the
  owner-side receive volume is what skews) + combine at m/2 rows per rank in
  EVERY mode (real systems combine AFTER token return = balanced) modeled as
  scale+scatter-write. RECEIPT for that choice: torch bf16 index_add_ = 53.6 ms
  per 2.56M rows (2-byte CAS atomics) and its cost varies with CALL GRANULARITY
  (one 2.56M-row call 53.6 ms vs 64 small calls ~22 ms) — a torch artifact that
  charged one-big-run modes ~27 ms unfairly in the first moe smoke (owned 118 vs
  smart 91 at UNIFORM); with the balanced scatter-write combine the z0 parity
  restores (87.0/89.8/84.6 ms). SMOKES: experts z0 52-56 ms (~3.5x one GEMM),
  z2.0 owned 89.1/imb .76, smart 66.0/.20, queue 63.6/.00; moe z2.0 owned
  128.7/.64, smart 103.4/.15, queue 98.9/.00. EXPECTATIONS (full sweeps, 3 seeds):
  experts — z0 all 52-57; owned to ~85-92 worst at z2.0; smart <=68; queue flat
  55-64. moe — z0 all 84-91; owned to ~125-135; smart ~100-108; queue flat
  85-100; natural cols in the z0-to-z0.5 class for smart/queue, owned mildly up.
2026-07-10 TABLES 1b/1c FILLED (3 seeds; artifacts table1b_experts.json /
  table1c_moe.json). VERDICT: on band everywhere (1b owned z2.0 worst-seed 112.4
  slightly above the 85-92 guess — seed variance in what co-locates with the
  mega expert; monotone in s, z0 parity <=2%, queue flat, B-bytes = 3x Table-1
  analytic exactly). The zoom-out story: EP-default z2.0 imbalance 82% -> 79% ->
  65% across scopes (mode-flat router/combine dilute it), queue 12% -> 5% -> 3%;
  wall gap EP-default/queue at z2.0: 1.31x -> 1.56x -> 1.34x.
2026-07-10 MULTI-MODEL MICRO (user ask): bench generalized to --geom E,N,K
  --topk --fused-gateup --shared-n; MODEL presets in the wrapper, geometry
  VERIFIED from HF configs: q3-235b-a22b 128E top-8 N=1536 K=4096 (m 3.84M);
  q3.5-122b-a10b 256E top-8 N=1024 K=3072 + shared expert N=1024 (m 5.12M);
  l4-scout 16E TOP-1 sigmoid, FUSED gate_up (one 2N=16384 GEMM), N=8192 K=5120
  + shared expert N=8192 (m 1.28M rows = 1.28M tokens at top-1). No real-trace
  columns for the new models (no capture exists; guard added: hist E must match
  --geom E). Shared expert + router run on the rank's own half of tokens =
  mode-flat (llama4's design point: the big shared expert dilutes imbalance
  exposure BY CONSTRUCTION at top-1).
  CHUNKER ITERATIONS (receipts): (i) q3-tuned force-chunk + HOT_CHUNK=8192
  re-streamed scout's 252 MB banks 10x at UNIFORM (owned B 20.1 GB vs queue 2.0)
  -> owned-side chunking made grid-aware (chunk only when len(segs)*n_blk_min
  cannot fill ~2 waves; pieces = 296/n_blk_min). (ii) Applying the SAME
  coarsening to the union list broke QUEUE balance at z2.0 (imb 0.41 on 2/3
  seeds: a mega expert parked at one list END in 126k-row units is too lumpy to
  share). The two lists serve different purposes: owned lists need GRID FILL
  (pieces rule), the union list needs SHARING GRAIN with a re-stream bound ->
  step = max(4N, 8192) rows (q3 reduces exactly to the banked 8192; scout gets
  32k-row units, re-stream <= ~25% of chunk A-traffic). Verified: queue z2.0
  imb 0.000/0.000/0.076 across seeds, walls 63-66 ms (experts scope); q3 z0
  cells unchanged. Q3 TABLES RE-BANKED under the final policy for consistency
  (old jsons archived as *_hotchunk8192.bak.json); owned-default inherits the
  union grain (its faster variant on q3), owned_smart uses the pieces rule.
  EXPECTATIONS (new models, full sweeps): 235b gemm z0 ~45-55, owned z2.0
  ~75-95 imb ~.8, queue flat; experts ~3.2x gemm; moe +30-45 flat. q35 gemm z0
  ~22-28 (E=256 -> longer tail, top1 61.5% at z2.0); scout experts z0 ~330,
  owned z2.0 ~550 imb ~.8, queue ~340-350 flat; scout moe adds shared expert
  ~= routed work at top-1 -> imbalance visibly diluted vs experts scope.
2026-07-10 MULTI-MODEL MATRIX FILLED (12 sweeps incl. q3 re-bank, ~11 min wall —
  kernels JIT-cached). VERDICTS vs bands: 235b gemm z0 42.8-44.7 (hair under the
  45-55 guess), owned z2.0 79.8 · 77% IN BAND, queue flat (experts imb <=2%
  everywhere); q35 z0 28.3-29.2 (hair over 22-28), owned z2.0 51.6 · 55% (E=256
  compresses wall-imb: 127 tail experts per side run less efficiently than the
  chunked mega side); scout ON BAND everywhere and queue PERFECTLY flat
  196-201/326-334/484-495 across the entire dial. GOALS section now carries the
  1d-1l summary grids; full tables via print_skew_tables.py.
  STEAL-GRAIN FINAL RULE + RECEIPTS: kept step = max(4N, 8192) rows (balance
  first). Tried unit-count targeting (~1024 units/hot expert) after a q35 probe
  showed 32k grain closes its z2.0 wall gap (43.3 -> 32.9, tied with oracle):
  the unified coarse rule regressed BALANCE broadly (q3 imb 0.18-0.32, 235b
  0.35, scout seed1 0.21 vs <=0.07 banked) — fine grain is what makes stealing
  bulletproof; the wall penalty exists ONLY on q35's 256-tiny-expert shape.
  Reverted; revert verified to reproduce the banked cells (q3 17.8-19.2,
  scout 197-201 · <=0.6%). q35's z>=1.5 queue-vs-oracle gap stands as a
  documented limitation (queue still beats realizable EP at every dial point).
  Q3 RE-BANK deltas vs the morning fill (old jsons archived
  *_hotchunk8192.bak.json): smart z2.0 executes faster under the pieces rule
  (gemm 18.3 -> 16.5 ms, experts 67.3 -> 62.8) at higher printed imb (19 -> 25%,
  18 -> 31% — closer to the 0.363 row floor, i.e. the same structural imbalance
  now shows honestly instead of hiding in re-stream time); owned z2.0 experts
  101.6 -> 93.4 (inherits the union grain, its faster variant); queue cells
  unchanged within noise. Story unchanged: queue is the only row that is flat
  AND balanced at every s, on every scope, on every model.
2026-07-10 SEP ROW ADDED (user ask): full 12-sweep re-run with
  MODES=owned,owned_smart,sep,queue (3-mode jsons archived in
  profiling_results/profiling_both_skew/archive_3mode/). sep = "ours (plan)": chunked-LPT over the
  union with mega-expert splitting (placement cannot split, the plan CAN — union
  A is rank-local), _chunk_local execution courtesy applied like the other
  placed modes; NO per-unit steal costs, NO runtime correction. READINGS: with
  PERFECT counts the plan is near-flat everywhere — q3-30b gemm 15.2-15.5 across
  the whole dial (its z2.0 15.2 is the BEST cell in the column), 235b z2.0
  55.8 ~ queue 55.7, scout z2.0 210.6 vs queue 200.7. Its costs at z2.0: B-bytes
  balloon from mega-splitting (scout 11.1 GB, q35 7.9 GB) and tail-efficiency
  imbalance where the non-mega side gets hundreds of tiny experts (q35 26%,
  receipts: B=[119.5, 2705.3] — row-balanced, time-unbalanced). CAVEAT pinned in
  GOALS: the bench hands sep the exact counts it will execute; e2e sep2 must
  plan from stale counts, so plan-vs-queue e2e is prediction-error-bounded —
  micro plan rows are an ORACLE-INPUT bound, not an achievable e2e claim.
  Queue remains best-balanced (<=3% at z2.0 on all models/scopes) with no
  counts needed. q3.5 fine-grain caveat now reads: plan 36.1 splits the
  smart-vs-queue gap (32.8 / 43.4).
2026-07-10 NAMING EPOCH 4 + PLAN WIRED E2E (user ask): canonical backends
  asym_sepqueue2_cpuadamwds (counter-raced steal; legacy sep2/sqep2/sqeq2
  canonicalize here at BOTH the driver alias map and run_lf, with echo) and
  asym_sepplan2_cpuadamwds (count-computed cut; NEW). Micro bench mode renamed
  sep -> plan ("sep" accepted as alias; printer reads both keys, so archived
  jsons still render). CORRECTION to the entry above: e2e plan does NOT need
  stale counts — token counts are exact at GEMM time and the armed path already
  exchanges both sides' segment headers; the plan cut is computed from the SAME
  union on both ranks with zero extra traffic. IMPLEMENTATION (ep_sep.py,
  ASYM_EP_SEP_MODE=plan): same union/transport/decline/headers; contiguous
  row-balancing cut at a segment boundary (arming rule caps rows/segment at
  MAX_MPE so the cut granularity is fine); each side launches ONLY its sublist
  of the union with a PRIVATE counter block (ints [3b:3b+3) of the ring's 8;
  zero cross-GPU counter traffic), then ONE host write fabricates the
  queue-final meet state so the unchanged spin_gather kernel gathers exactly
  the planned-stolen items; degenerate cuts (k=0/total) fall through to the
  queue race symmetrically.
  BUG FOUND + FIXED while validating: the steal kernel is ONE ITEM PER CTA and
  the launcher under-provisions the grid to DG_EP_QUEUE_GRID_PCT=75% — queue
  mode tolerates it (the two sides' grids overlap on the SHARED counters) but a
  private-block sublist launch must cover all its items alone. Probe receipt:
  zero rows began EXACTLY at grid_y_local*n_blk (claimed=1116 = ceil(123*.75)*12
  at the bal cut; skew cut claimed=768 = 64*12). Fix: plan-mode processes force
  DG_EP_QUEUE_GRID_PCT=100 (queue keeps its tuned 75%).
  VALIDATION: ep_sep_probe --mode queue PASS (regression, bitwise 8/8) and
  --mode plan PASS (bitwise, incl. the 3:1 skew case where the cut lands inside
  rank0's section and side1 executes 43 stolen segments); e2e 2-step smoke of
  both backends in flight (loss parity + label check).
2026-07-10 SEP RENAME E2E VALIDATION (receipts; smokes in profiling_results/profiling_smoke_seprename/):
  CORRECTNESS all green: loss overlay sepplan2 vs sepqueue2 <=0.006 at every
  step (1.5408/1.5460 class at 20k step5); run-dir labels carry the canonicals;
  legacy aliases resolve at the driver (asym_sqep2_cpuadamwds and bare asym_sep2
  -> asym_sepqueue2_cpuadamwds; asym_sepplan2 -> canonical; DRY_RUN receipt);
  bench MODES=sep maps to plan.
  ARMED-REGIME PARITY (2048|8|1, m/segs=1024 <= MAX_MPE, armed 1200/1200):
  sepplan2 10.3 s == sepqueue2 10.2 s steady; plan planned 1200/1200,
  spin_wait 0.4-4.3 s total. HYSTERESIS added on the way: the plan cut snaps to
  n_own when |own_rows - half| <= MAX_MPE — crossing the ownership boundary
  couples the launch on the peer's done flag, so a sub-segment "gain" is pure
  coupling risk; under equal-shard DP (equal m per rank) the cut ALWAYS snaps,
  i.e. e2e plan only crosses under real per-rank load imbalance.
  DECLINE-REGIME (20000|8|1, m/segs=10k > MAX_MPE, declined 1200/1200): the
  95.9 s sepplan2 outlier did NOT reproduce (re-measure 64.2 s, tight steps —
  one-off environmental; first smoke pair was also contaminated by concurrent
  bench/DRY_RUN CPU load, receipted). Stable picture: sepplan2 64.2-64.6,
  sepqueue2 64.4-65.6.
  OPEN FLAG (not the rename): both SEP backends pay +7 s/step vs sqdp2 (57.7)
  at 20k with EVERYTHING declined, ALL of it in backward (bwd 49.8-50.6 vs
  43.0; fwd identical 11.2). Pre-gate landed (host-int decline BEFORE the
  offsets .cpu().tolist() GPU sync + atexit stats line) and is protocol-safe
  (probes PASS) but did NOT close the gap -> the tolist sync is exonerated;
  suspects: the S6 prebuild's ~43 GB of extra pinned fabric slots (RING x ranks
  x X/D at SLOT_ROWS=655360) taxing backward's host-side work, or another
  SEP-gated path. The 2026-07-09 banked 56.9 s parity was NOT reproduced today.
  NEXT INSTRUMENT when picked up: bwd phase decomp sqdp2-vs-sepqueue2 same-day
  + a run with ASYM_EP_SEP_SLOT_ROWS shrunk (e.g. 65536) to test the pin-size
  hypothesis in one flip.
2026-07-10 OPEN FLAG CLOSED — the SEP decline-regime tax IS the pinned slot
  size. One-flip receipts (sepqueue2, 20k, everything declined):
    SLOT_ROWS 655360 (~43 GB pinned slots) -> 65.6 s (bwd 50.6)
    SLOT_ROWS  65536 (~4.3 GB)             -> 59.7 s (bwd 45.0)
    SLOT_ROWS 163840 (~10.7 GB)            -> 61.4 s
  vs sqdp2 57.7 — the pre-gate had already exonerated the .tolist() sync, and
  the slots explain ~6 of the +8 s; residual ~2-3.7 s = ctrl page + remaining
  slot bytes + hook. The 07-09 "56.9 parity" stands unreproduced (conditions
  unknown); today's chain is 3x-reproduced.
  DEFAULT LANDED: ASYM_EP_SEP_SLOT_ROWS = 163840 (run_lf_profiled_train.py).
  131072 exactly was tried and FALSIFIED: BLOCK_M padding pushes the 2k class
  to m ~139k and the capacity check silently declined the backends' own home
  regime (armed 0/1200; and — side receipt — 2k DECLINED at small slots runs
  8.2 s vs 9.7 s armed: natural-balance arming costs ~1.5 s, which is exactly
  why the MAX_MPE gate exists — arming pays under skew, not balance). With
  163840: 2k armed 1200/1200 at 9.7 s; 20k declined at 61.4 s. Raise the env
  deliberately for larger armed workloads.
2026-07-10 C1+C2 LANDED + VALIDATED: zipf generator exact (top1 at s=0/0.5/1/2 =
  0.8/4.7/18.4/61.1% vs formula), seeded shuffles permute-not-reshape, smart-split
  LPT near-optimal (z2.0 loads 3.13M/1.99M = 0.363 imbalance — the STRUCTURAL
  placement floor when one expert holds 61%). TABLE-1 EXPECTATIONS (pre-run):
  z0 all modes equal ~15-16 ms, imb <=5%; owned imbalance grows with s and is
  seed-dependent (mean/worst spread); smart stays row-balanced until ~s1.7 but
  pays the growing mega-segment tail from ~z1.0; z2.0 smart >= 0.36 imb by
  structure; QUEUE flat ~15 ms at every s (chunking + interleaving). Real cols
  should reproduce the banked class (owned 16.7-17.9, queue 14.6-15.9 ms).
