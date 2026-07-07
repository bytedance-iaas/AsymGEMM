# FIX CAMPAIGN: EP=2 must beat EP=1 (substrate latency, gb200_ep.md interlock)

Companion to `agent/impls/gb200_ep.md` (its Decision Log holds every verdict below with
artifacts) and `gb200_tp.md`. Style: staged, gated, ONE change per run, steady-state
timing only. This doc OWNS the sanity gate that unblocks the sEP ladder.

## Goal (ULTRA CLEAR)

```text
SANITY GATE (SG), two-tier (user-set 2026-07-07):
  SG-FINAL (the judgment row, WEAK SCALING — the standard convention): EP-2 with the SAME
    per-device batch (8 per GPU, two DIFFERENT batches = v2 shape, global 16) vs |1 with 8.
    BREAK-EVEN: step <= 2xT1 (any less = net throughput win). TARGET: step <= 1.5xT1
    (=1.33x tokens/sec). GOOD: <= 1.2xT1 (=1.67x).
  SG-INTERIM: the replicated shape is ONLY a cheap debugging vehicle for substrate fixes
    (fix -> rerun -> census); NO performance framing attaches to it (T1/2 framing DROPPED,
    user 2026-07-07 — infeasible and not the goal). The judgment row is SG-FINAL only.
  Workload: q3-30b-a3b | asym | recomp-off-full-fg-ker101 | s20000 (s2048 for fast cycles).
  Steady-state rule: MAX_STEPS=4, drop warmup AND last step.
WHY THIS GATE: (a) with a broken substrate, "sEP beats static" would be a rigged win;
  (b) every fix here is INHERITED by sep1/sep (same code path, one scheduling knob apart);
  (c) EP=2 slower than EP=1 is disqualifying for the paper, full stop.
CONSTRAINTS: STANDARD designs only (Megatron-grade: pools, events, side streams, threads,
  caching) — no numerics changes, no exotic tricks; loss parity after EVERY fix;
  block-parity test after touching metadata/kernels; JIT-warm rule (never quote first-run
  timings); membind=0,1; fresh OUTPUT_ROOT per experiment.
```

## STATUS + HANDOFF (2026-07-07 evening — EXPERIMENTS PAUSED BY USER; read this first)

```text
===================== METRICS (all steady-state unless noted) =====================
MODEL/WORKLOAD: Qwen/Qwen3-30B-A3B (E=128 top-8, ker101 routed kernels), LoRA r64,
  real smoke data, s20000 x b8 (160k tok/step), GPUs 0,1 (pair), static EP-2 (vanilla
  fixed-ownership; ASYM_EP_MODE=static ASYM_STP_MOE=1 ASYM_EP_E2E_LANDED=1).
|1 reference: 15-20 s/step (WITH nsys+stats overhead AND the same GC save-on-cpu waste
  — see D1b; the CLEAN |1 bar has never been measured = D5, DO IT FIRST ON RESUME).
|2 static fix progression (backward seconds, steady):
  245 (cold JIT — artifact) -> 122-145 (warm) -> 97-104 [R1] -> 72-86 [R2] -> ~same [R3]
  -> 58-65 [R4] -> 49-67 [R5]. fwd: 12 -> 9.4-9.7. step: ~62-75 total at R5.
GPU busy (nsys, window): |2 = 7% pre-fixes -> 34% at R5; |1 = 55%. memcpy at R5 window:
  H2D 1970 GB (src pageable!), D2H 1749 GB (dst pageable!), P2P 535 GB, D2D 479 GB.
GIANT GAPS: ~6+ synchronized ~2.5s gaps/step on BOTH devices survived R1-R5 (nothing
  runs during them — no kernels, no copies). Attributed by D1b (below).
LOSSES: stable at 4 decimals across EVERY fix round (1.77/1.70/1.76/1.63/1.57 pattern);
  block-level parity (scripts/testing/stp_moe_block_parity.py) exact after each round.
KERNEL-LEVEL sEP WINS (already banked, probe): balance <=4% at any skew vs static's
  82-94%; static/queue 4.24x/6.94x/8.58x at alpha=.25/.5/.75; balanced overhead 0.969;
  bitwise-exact. Natural skew measured: scout dev-share mean 0.59-0.61 (worst 0.87);
  q3 mild (0.53). These motivate sEP once the substrate is fair.

===================== ROOT-CAUSE FINDINGS (receipts) =====================
FOUND+FIXED (R1-R5, all in gb200_ep.md Decision Log with file anchors):
  R1 offload pool keyed by EXACT shape; EP's per-layer variable routed-row counts ->
     zero reuse -> ~7 GB/layer fresh cudaHostAlloc. FIX: bucketed dim-0 (8k/64k granules)
     + return-base-via-view in activation_offload._alloc_cpu/_return_cpu.
  R2 attention saved-tensor restage was non_blocking=False + event.synchronize per tensor
     (host-blocking). FIX: side-stream + event-based async unpack
     (attention_activation_offload.py; per-device _h2d_restage_stream).
  R3 same for ActivationOffloadManager.stage/wait_cpu_ready (activation_offload.py).
  R4 per-call metadata/pad rebuilds with hidden syncs (.item(), sync D2H, mask-size).
     FIX: memoized on the offsets tensor: _pad_grouped_input_for_asym (frozen_linear),
     _pad_cpu_left_grouped_input_for_asym (cpu_left, + pooled pinned buffer),
     _expert_blocks (qwen3_moe_finegrained), prepare_grouped_lora_metadata (lora).
  R5 the SOURCE producers rebuilt fresh tensors per call so downstream memos keyed on
     throwaway objects. FIX: memoized _pad_route_metadata_for_asym,
     _group_metadata_for_kernel (qwen3_moe_routed_gemm), build_contiguous_route_metadata
     (moe.py, memo on topk_indices).
D1b THE REMAINING ELEPHANT (receipts: run command.txt + gc_boundary_offload.py:98-100 +
   nsys pageable-copy class 1.31GB x ~2300/step + census 'forward' frames inside bwd):
   the 'recomp-off-full-fg' stage FORCES GRADIENT_CHECKPOINTING=true + USE_UNSLOTH_GC=true
   + UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true (profile_lora_lf_test_both.sh:~3268-3273; same
   in _source.sh). Consequences: (a) backward re-runs every layer's forward (RECOMPUTE) —
   under |2 BOTH branches recompute serially on ONE host thread; (b) the recompute itself
   runs under torch.autograd.graph.save_on_cpu -> recompute saves are round-tripped to
   CPU and consumed microseconds later = ~3 TB/step of pointless copies = the giant gaps
   + the pageable-copy class. |1 pays the same waste at half volume.
   RESOLUTION (user-debated, settled): save-on-cpu-during-recompute is ESSENTIAL for
   CAPACITY/FRONTIER rows (its whole point: caps HBM when one layer's activations are
   huge) but PURE OVERHEAD at THROUGHPUT rows (a few GB vs ~155 GB free HBM). POLICY IS
   ROW-TYPED: throughput/ladder rows run it OFF on BOTH |1 and |2 (symmetric, fair);
   frontier rows keep it ON for every system. Override knob LANDED in both drivers:
   ASYM_GC_SAVE_ON_CPU_OVERRIDE=false (see Landed changes).
D1 dead-end note: ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0 in this config — the
   down_dx_gather_left wrapper (instrumented with ASYM_EP_DXTIME timers) is a DISABLED
   branch here; bwd down-dX goes through asym_grouped_frozen_linear dispatch instead.
STRUCTURAL THESIS (dense control case): sTP-2 BEAT |1 on dense q3-32b (234.9 vs 251.8 s)
   — the substrate wins when GPU-bound; this MoE model is HOST-bound, and one python
   thread drives two GPUs (~2x serial host work). Megatron's EP=2 = one RANK PER GPU.
   F2 (branch threads) approximates that in-process; v2's 2-process shape solves it
   structurally.

===================== IMMEDIATE NEXT ACTIONS (in order, on resume) =====================
A1 F1 A/B (the run killed by the pause): same |2 row + ASYM_GC_SAVE_ON_CPU_OVERRIDE=false:
   OUTPUT_ROOT=$PWD/profiling_gb200ep_e2e MAX_STEPS=4 WARMUP_STEPS=1 \
   ASYM_GC_SAVE_ON_CPU_OVERRIDE=false ASYM_EXPACT_CPU_POOL_MAX_BYTES=96000000000 \
   ASYM_STP_MOE=1 ASYM_EP_MODE=static ASYM_EP_E2E_LANDED=1 GPU_POOL=0,1 \
   RUNS='q3-30b-a3b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false' \
   bash scripts/lf/profile_lora_lf_test_both.sh
   READ: step_samples in profile.json (drop warmup+last); check per-device peak HBM
   (expect ~50-90 GiB — must stay < 184); loss column vs prior runs (1.77/1.70/1.76/1.63).
A2 D5 |1 clean bar: same env minus stp knobs, backend asym_cpuadamwds, |1, GPU_POOL=<free>,
   ALSO with ASYM_GC_SAVE_ON_CPU_OVERRIDE=false (symmetric).
A3 If |2 still >> |1 after A1/A2: D4 (time branch0 vs branch1 sections in
   StpDecoderLayer.forward via file-logged timers — NOT stdout, harness swallows it) ->
   F2 branch-parallel host (worker thread for branch1 fwd; risks/rollback in F2 below).
A4 THEN the REAL comparisons the user wants (weak scaling): build v2 (gb200_ep.md E5,
   sharded batches; wiring plan + constraints all verified in ep.md E3.5) and run
   8+8 (global 16) vs |1's 8 -> SG-FINAL verdict; then the sEP ladder
   (static vs hostsplit vs sep) on the fair substrate.

===================== ENVIRONMENT CAVEATS (this box; critical for another machine) =====
- sglang server (EXTERNAL, supervised/auto-respawning, ~163 GiB): originally on GPU 3;
  after a broad pkill during the pause it respawned ON GPU 0 -> the dev pair (0,1) is
  currently OCCUPIED. On resume: either the user moves it back to GPU 3, or switch runs
  to GPU_POOL=2,3 (pair 2,3 is the other same-superchip pair; HC1 membind=0,1 unchanged).
  DO NOT pkill broadly — kill only exact PIDs of our trainers.
- GPU pool env is GPU_POOL=0,1 (NOT GPUS). Driver detaches the trainer: the shell command
  returning does NOT mean training finished; watch profile.json / train.log.
- Trainer stdout is SWALLOWED by the harness — all diagnostics must write to FILES.
- JIT cache: warm on THIS box only. Fresh machine => first run recompiles every kernel
  variant (the 245s artifact); NEVER quote first-run timings (rerun to measure).
- Model caches on THIS box: Qwen3-30B-A3B, Llama-4-Scout-17B-16E, Qwen3-32B, Qwen3.5-35B-A3B
  (HF_HOME=/scratch_local/...). Fresh machine: first run auto-downloads (~60-220 GB).
- Steady-state rule everywhere: MAX_STEPS=4 WARMUP_STEPS=1, report middle steps.
- ALL code changes below are IN THE WORKING TREE, UNCOMMITTED (branch main_kevin). A
  handoff to another machine needs this tree (or commit+push first). _C must be rebuilt
  once (.venv/bin/python setup.py build_ext --inplace) for the queued-kernel entry point.

===================== LANDED CODE CHANGES (inventory; all uncommitted) =====================
KERNEL (sEP queue, probe-validated): asym_gemm/include/asym_gemm/common/asymScheduler.cuh
  (explicit-ids ctor + n_blk field); asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh
  (ASYM_BF16_EP_QUEUED entry-pop variant + n_blk fixes at :380/:950);
  csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp (SM100BF16EpQueuedAsymGemmRuntime +
  launcher + DG_EP_QUEUE_GRID_PCT); csrc/apis/gemm.hpp + asym_gemm/__init__.py
  (m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued).
SUBSTRATE PERF (R1-R5): asym_gemm/training/activation_offload.py (bucketed pool, async
  stage, wait_event); attention_activation_offload.py (async unpack, _h2d_restage_stream);
  frozen_linear.py (pad memo); cpu_left.py (pad rewrite + pooled buffer);
  qwen3_moe_routed_gemm.py (producer memos + ASYM_EP_DXTIME file timers);
  moe.py + lora.py + qwen3_moe_finegrained.py (metadata memos).
I7/EP WIRING (e2e loss-parity PASS at s2048: deltas 0.001-0.009 vs |1):
  asym_gemm/training/stp_moe.py (ep_slice_route_metadata, slice_experts_for_ep,
  build_ep_branch_block; HostWeight routers SHARED not deepcopied); stp_wrap.py (MoE branch
  in build_stp_full_tp, Qwen3MoeAttention for branch1); qwen3_moe.py (ep_expert_range hook,
  empty-range zeros partial, _EpBalanceStats histograms, ASYM_EP_SKEW_HOT knob);
  llama4_moe.py (stats hook).
HARNESS: profile_lora_lf_test_source.sh + _both.sh: router whole->hf downgrade allow-list
  now includes the |2 asym family (~:4529 in each — THE bug that silently disabled MoE
  asymization under sTP); ASYM_EP_* knobs/tags/guards; run_env passthroughs (ASYM_EP_*,
  DXTIME, EP_HOT_CHUNK_ROWS, DG_EP_QUEUE_GRID_PCT); ASYM_GC_SAVE_ON_CPU_OVERRIDE hook in
  the recomp-off stage block (both files). run_lf_lora_sft.sh: sEP guards (HC-EP4, skew ACK).
TESTS/TOOLS: scripts/testing/ep_queue_probe.py (kernel probe, PASS artifacts in
  profiling_gb200ep_e3/); scripts/testing/stp_moe_block_parity.py (block parity, PASS);
  scripts/lf/analyze_stp_bwd.py (nsys busy/memcpy/gap decomposition).
ARTIFACTS: profiling_gb200ep_e3 (probe PASS), _e1/_e1b/_e1c/_e1d (natural-skew histograms +
  MoE fractions: q3 36.9% fwd, scout 23.7%), _e2e (parity + fix-round rows),
  _diag2..8 (censuses, traces). Memory files: ~/.claude-kevin .../gb200-sep-progress.md.
```

## DONE vs REMAINING (the whole sEP campaign — read with gb200_ep.md)

```text
=============================== DONE (validated, banked) ===============================
D-A  sEP queue KERNEL (the mechanism): entry-pop variant of the grouped AsymGEMM kernel,
     bitwise-exact, probe PASS at prod-scale M — balance <=4% at ANY skew (static: 82-94%),
     static/queue 4.24x/6.94x/8.58x, balanced overhead 0.969 (free when balanced).
     Granularity co-design measured (host-B refetch ~25us/chunk; hot-chunk 8192 sweet spot).
D-B  I7 static EP-2 substrate (vanilla EP, the baseline rung) WIRED E2E: two-branch MoE
     layers over shared pinned banks; loss parity vs |1 at s2048 (deltas 0.001-0.009);
     block-level parity exact (fwd ~1 ulp, LoRA grads exact).
D-C  Motivating measurements (real data): scout natural device-share mean 0.59-0.61
     (worst 0.87, persistent 1.5x layers) = the gains model; q3 mild (0.53) = the
     correctness model; MoE fwd fraction: q3 36.9%, scout 23.7%; hottest-expert stats.
D-D  E0 harness: ASYM_EP_MODE knobs/tags/guards, skew injector (ACK-gated), histogram
     collector, run_env passthroughs; router whole->hf allow-list bug FIXED (was silently
     disabling MoE asymization under sTP).
D-E  Substrate perf rounds R1-R5 (bwd 245 -> 49-67s; GPU busy 7% -> 34%): bucketed pinned
     pool, async event-based restage x2 machineries, metadata/pad memoization x7 functions.
     Loss stable at 4 decimals through every round.
D-F  Root cause of the remaining stalls FOUND with receipts (D1b): stage-forced GC +
     recompute-side save_on_cpu (~3 TB/step pointless round-trips). Row-typed policy
     settled + override knob landed in both drivers (UNTESTED — the A/B was killed by
     the pause).
D-G  Megatron-LM deep-read (design rules extracted with file:line receipts) + the fix
     campaign doc (this file) + all state in gb200_ep.md Decision Log + memory files.

=============================== REMAINING (in execution order) ===============================
R-1  F1 A/B run (command = STATUS A1): |2 static with save-on-cpu OFF -> expect giant
     gaps + pageable class to vanish; verify HBM peak + loss parity.        [~15 min]
R-2  D5: CLEAN |1 bar (no nsys, no stats, save-on-cpu OFF, steady-state) — the honest T1.
                                                                            [~15 min]
R-3  If |2 still >> |1: D4 branch-timer receipt -> F2 branch-parallel host (worker thread
     for branch1 fwd) and/or F3 route-plan-per-layer.                       [0.5-1.5 days]
R-4  sep1 e2e wiring (queue kernel into the fg grouped launches; plan + constraints
     verified in gb200_ep.md E3.5: queued fwd-base + routed-kernel EP_QUEUED variant for
     bwd dX + full-pack/row-range-view packing).                            [~1 day]
R-5  sEP-v2 (THE HEADLINE, gb200_ep.md E5): sharded batches (8 per GPU, different data),
     shared host token pool, affinity ordering, steal-traffic accounting.   [~1-2 days]
R-6  SG-FINAL judgment: v2 8+8 weak-scaling vs |1's 8 (break-even 2xT1, target 1.5xT1).
R-7  The REAL comparison ladder (gb200_ep.md E4/E6): static vs hostsplit vs sep1 vs sep
     x q3-30b-a3b + llama4-scout (natural data only, no synthetic) + superoffload/|1
     anchors; EG verdicts; paper rows.
R-8  Deferred (recorded, not blocking): E2 host-side metadata full refactor, I5 dedup of
     replicated [M,H] saves (halves traffic — an optimization once F1 lands), fp32 SDPA
     save volume investigation, scout parity rows, hostsplit e2e mode.
```

## Diagnosis runs (D-track; receipts before fixes)

```text
D1 [CLOSED] dx-phase timers empty — ROUTE_DOWN_DX_GATHER=0: instrumented wrapper is a
   disabled branch in this config. Timers (file-based, ASYM_EP_DXTIME[_PATH]) remain.
D1b [CLOSED — ROOT CAUSE] see STATUS: stage-forced GC + recompute save_on_cpu waste.
D2 [OPEN, only if A1 leaves >1s gaps] giant-gap attribution: per->1s-gap overlapping
   NVTX/memcpys + first kernel after (extend analyze_stp_bwd.py).
D3 [LIKELY CLOSED BY D1b] the pageable-copy class = save_on_cpu round-trips; re-check
   after A1 (should vanish with the override).
D4 [OPEN] host-work budget: branch0 vs branch1 section timers (FILE-logged) in
   StpDecoderLayer.forward + autograd wall -> sets the F2 target.
D5 [OPEN — DO FIRST ON RESUME] |1 clean re-baseline (no nsys, no stats, save-on-cpu OFF,
   steady-state) = the honest T1 for SG-FINAL.
```

## Fix stages (F-track; each gated, one at a time)

```text
F1 [READY — the killed A/B, command in STATUS A1] row-typed save-on-cpu policy:
   OFF for throughput rows on BOTH |1 and |2; ON for frontier rows (its actual purpose).
   GATE: giant gaps -> 0; pageable class -> 0; bwd drops toward the copy total; loss
   parity; per-device HBM peak < 184 GiB with margin.
F2 BRANCH-PARALLEL HOST (structural): branch1 fwd on a persistent worker thread (GIL
   releases during CUDA/C++ enqueue). RISKS: per-thread torch.cuda.device context, join
   ordering at Bcast01Fn/AllReduce2Fn (already event-based), memo thread-safety (branch
   tensors disjoint). GATE: fwd 9.5 -> ~5-6s; loss parity; mini-parity.
F3 ROUTE-PLAN-PER-LAYER (finish E2): one plan object per layer-branch feeding ALL grouped
   calls (kills residual per-call python + remaining .item()s incl. ep_slice int()s).
   GATE: census shows no metadata/pad frames in top-10.
F4 MEGATRON POLISH (only if short): fwd D2H side stream; backward prefetch-1; in-flight cap.
F5 SG-FINAL: v2 8+8 weak-scaling vs |1 clean bar (PASS <= 1.5xT1; break-even 2xT1) ->
   unlock the sEP ladder (static/hostsplit/sep + anchors) per gb200_ep.md.
STRUCTURAL FALLBACK: if the single-process replicated vehicle stays host-floor-bound,
   judge on v2's two-process-equivalent shape only (per-GPU host = Megatron's shape) and
   record honestly in ep.md.
```

## Measurement discipline

```text
Steady-state: MAX_STEPS=4 WARMUP=1, report middle steps (drop warmup + last).
JIT-warm: never quote a first run after kernel-code changes; rerun to measure.
One change per run; loss column checked EVERY run (4-decimal stability = pass).
Census loop: py-spy burst (30-40 samples timed into the backward) -> rank top frames ->
  fix -> steady rerun. nsys only when the census is ambiguous. py-spy CANNOT see native
  frames on ARM (idle samples = host blocked in C/CUDA); gdb bt as fallback.
Diagnostics ALWAYS write to files (harness swallows trainer stdout — two false starts).
Artifacts: profiling_gb200ep_diag*/ + profiling_gb200ep_e2e/; every verdict appended to
  gb200_ep.md's Decision Log.
```

## Decision Log (append-only)

```text
2026-07-07 doc created mid-campaign; R1-R5 receipts in gb200_ep.md's log.
2026-07-07 D1 closed (disabled branch); D1b root cause landed (stage-forced GC +
  recompute save_on_cpu, ~3 TB/step waste); row-typed save-on-cpu policy settled with
  user (OFF for throughput rows both sides, ON for frontier rows); override knob landed
  in both drivers; T1/2 framing dropped (SG-FINAL = weak scaling 8+8 vs 8).
2026-07-07 PAUSED BY USER mid-F1-A/B (run killed; rerun = STATUS A1). During the pause
  sweep the external sglang server respawned from GPU 3 onto GPU 0 (~163 GiB) — pair 0,1
  occupied until it is moved or runs switch to GPU_POOL=2,3. Full handoff state written
  (STATUS + Landed-changes inventory); tree UNCOMMITTED on main_kevin.
```
