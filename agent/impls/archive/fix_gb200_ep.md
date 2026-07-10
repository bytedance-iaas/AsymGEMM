# BUILD CAMPAIGN: true sEP (asym_ep2) — different data per GPU, EP=2 must beat EP=1

Companion to `agent/impls/gb200_ep.md` (design/mode taxonomy, E-stage receipts) and
`agent/impls/gb200_dp.md` (dp2 anchor). This doc OWNS the sEP build plan (S-track) and the
sanity-gate comparisons. Style: staged, gated, ONE change per run, steady-state timing only.
2026-07-07 restructures: replicated-batch vehicle RETIRED; then S1 re-designed RANK-PER-GPU
after the Megatron-LM deep-read (see Decision Log). Backend name: asym_ep2_cpuadamwds
(formerly drafted "asym_tp2" — renamed, it IS expert parallelism; tp2_* = the dense TP family).

## GOALS — the comparisons (exact configs), the naming, the expected outcomes

```text
THE ROWS (RUNS lines; batch field is PER-DEVICE for |2 rows; weak scaling — |2 rows
process 2x T1's tokens/step):
  T1     q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
         (the honest |1 clean bar: no nsys, save-on-cpu OFF)
  T_ep2  q3-30b-a3b|2 ; asym_sqep2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
         (TOKEN RENAMED 2026-07-08 naming epoch: asym_sqep2 = sEP+queue; the historical
          rows below ran under the OLD token asym_ep2_cpuadamwds = same system)
         (OURS = true sEP, RANK-PER-GPU: torchrun 2 ranks, each rank the |1 asym stack
          VERBATIM; DistributedSampler shards 8+8 = global 16 (different sequences per
          GPU); ONE shared pinned weight fabric; NO all-to-all (ownerless streaming);
          NO DDP wrapper (one manual LoRA-grad allreduce/step). Lands at S1; queue/steal
          balancing at S2.)
  T_dp2  q3-30b-a3b|2 ; asym_dp2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
         (DEFERRED ANCHOR/ablation — run ONLY AFTER O2 passes. Differs from T_ep2 by
          exactly: per-rank DUPLICATED pinned arenas + the HF DDP reducer. Needs
          ASYM_DP_FIND_UNUSED=true. Wall = max rank via
          scripts/lf/aggregate_dp_ranks.py --run-dir <dp2 seq_root>.)

ONE INVOCATION (rows run sequentially; drop rows that haven't landed yet):
  OUTPUT_ROOT=$PWD/profiling_gb200ep_sg MAX_STEPS=4 WARMUP_STEPS=1 PROFILERS=source \
  ASYM_GC_SAVE_ON_CPU_OVERRIDE=false ASYM_EXPACT_CPU_POOL_MAX_BYTES=96000000000 \
  GPU_POOL=0,1 \
  RUNS='<T1-row> || <T_ep2-row>' \
  bash /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_both.sh
  READ: step_samples in profile.json (drop warmup+last; profile.json is rank0's — T_ep2
  wall = max rank via aggregate_dp_ranks.py; every rank writes rank<R>_memstats.json).

APPLES-TO-APPLES NOTE (do not misread ASYM_GC_SAVE_ON_CPU_OVERRIDE=false):
  It does NOT disable offloading. In EVERY row here the full asym posture stays ON:
  host-streamed frozen weights, expert-activation offload to the pinned CPU pools,
  attention-activation offload, CPU AdamW, gradient checkpointing. The override only
  removes the D1b pathology — the recompute-side save_on_cpu that round-trips ~3 TB/step
  of tensors consumed microseconds later in the same backward (a CAPACITY tool, useless
  at these rows' ~90 GiB free HBM). It is applied to BOTH rows of every pair via ONE
  shared env var (row-typed policy, user-settled 2026-07-07: OFF for throughput rows on
  every system, ON for frontier rows). OPTIONAL honesty row at S3: rerun the pair with
  the override unset (ON both sides) — the verdict must hold there too.

THE WIN CONDITION (the whole point, stated once):
  step(T_ep2) < 2 x step(T1)  STRICTLY  <=>  the 2-GPU pair delivers MORE tokens/sec
  than one GPU (T_ep2 does 16 seqs/step vs T1's 8). At exactly 2x the second GPU bought
  nothing (break-even). The multiplier measures HOW MUCH win:
    2.0x = break-even | 1.5x = 1.33x one GPU (TARGET) | 1.2x = 1.67x (GOOD) |
    1.0x = 2.00x = perfect scaling (the ideal-DP bound — the eventual ambition).

EXPECTED OUTCOMES / VERDICTS (in order):
  O1 (S0)  T1 measured clean (save-on-cpu OFF, no nsys) — the denominator for everything.
  O2 (S1)  EP2 BEATS EP1: T_ep2 < 2xT1 strictly; TARGET <= 1.5xT1; GOOD <= 1.2xT1.
           Loss parity at every gate (bf16 reduction-order envelope method, dp.md D2 —
           never a static 1e-2 band).
  O3 (S2)  Robustness under skew with no balanced-cost: e2e skew ladder balance <= 5%,
           balanced overhead <= 3%, steal bytes ~ 0 when balanced (EG-V2); scout
           natural-skew row shows the gains case (dev-share mean 0.59-0.61 banked).
  O4 (S3, DEFERRED dp2 anchor) after O2 passes: run T_dp2 once. Dense precedent says
           T_dp2 ~ 1.0-1.1xT1 (dp.md D1: pairing preserved 1.95x per-GPU throughput) but
           it is UNPROVEN on MoE (D2.5 never ran; +21.3s DDP reducer, 2x C2C streaming,
           2x host RSS ~493 GiB). Verdict: T_ep2 <= T_dp2 expected (ep2 = dp2 minus
           reducer minus duplicated arenas), or explain with receipts.
```

## BEHAVIOR RULES (binding on whoever executes this doc; B-numbers ≠ substrate rounds R1-R5)

```text
B1 PREDICT-THEN-MEASURE: BEFORE every profiling run, WRITE the expectation into the
   Decision Log — concrete numbers, not vibes (step-time band, HBM peak, host RSS, loss
   band / parity tolerance). After the run, compare. Deviation beyond the stated band
   (defaults: >25% step time, >15% memory, ANY loss/parity miss) = RED FLAG: stop the
   stage, diagnose to a receipt, log discrepancy -> cause -> resolution before the next
   run. Never rationalize a surprise after the fact; an unexplained too-GOOD number is
   as suspect as a too-bad one (usually a short-circuited/broken row).
B2 STAGE-GATE PROTOCOL: a stage CLOSES only when BOTH artifact classes exist ON DISK and
   are named in the Decision Log (path + number + PASS/FAIL): (a) CORRECTNESS —
   parity/loss-envelope receipts, shard receipts, claimed==total dumps; (b)
   EFFECTIVENESS — e2e s20000 profile.json step_samples (+ memstats) vs the previous
   stage's number. No artifact = the stage did not happen. NEVER build stage N+1 on an
   ungated stage N — an incorrect module mixed in silently poisons every later verdict.
   A PASSED gate is an order to ADVANCE: start the next stage immediately; the campaign
   never idles.
B3 ITERATION MANDATE (keep fixing, keep building, do not stop): the O2 gate
   (step(T_ep2) < 2xstep(T1) STRICTLY) is not optional or re-scopable. On FAIL: diagnose
   the core issue (census/nsys loop; suspects = the STALL PLAYBOOK), fix ONE thing,
   rerun the pair, log receipt -> fix -> number, LOOP until it holds. REACH: after 2x
   holds, keep iterating toward 1.5xT1 (then 1.2x) whenever the artifacts say the
   residual is addressable — never a reason to stop at 1.9x, never grounds to call
   2x-epsilon a failure. The campaign ends only at the SG-FINAL verdict (S3), not before.
B4 MEASUREMENT/BUILD HARD RULES: no replicated-batch perf rows ever (retired
   2026-07-07; parity harness only). Stages gate on e2e s20000 rows; s2048 = numerics
   parity only; isolated probes only for kernel/system-internal changes. ONE change per
   run; loss column every run; JIT-warm rule; steady state MAX_STEPS=4 WARMUP_STEPS=1
   (drop warmup + last); fresh OUTPUT_ROOT per experiment; GPU pair 0,1 or 2,3;
   membind=0,1. Efficiency: no per-expert python loops, no small-GEMM decomposition,
   grouped/queued kernels only, O(1) launches per (layer, phase), item lists vectorized
   + memoized. NO host threads driving GPU work — Megatron rule: one process per GPU,
   async = CUDA streams + events from each rank's ONE main thread.
B5 DRIVER USAGE: always drive
   /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_both.sh
   with env overrides (RUNS + knobs) — never rebuild configs from scratch. Read the
   GOTCHAS log (next section) before every session; append every new trap to it.
```

## GOTCHAS — caveats to avoid (running log; APPEND new ones here)

```text
- ALWAYS pass RUNS via env (rows join with '||'); the in-file RUNS array can contain the
  USER'S ACTIVE manual rows (e.g. a live superoffload ceiling row) — running the driver
  bare executes THEIR rows, and campaign runs must never rely on the in-file array.
- DRIVER SYNC PROCEDURE (hardened after a near-clobber 2026-07-07): before cp
  _both.sh -> _source.sh, extract source's CURRENT scratch from the LIVE file (or git
  HEAD) — NEVER from a stale diff; the scratchpads drift within hours (the user edits
  them for manual runs). After copy: re-apply source's scratch, verify diff == 1 hunk.
- Batch field is PER-DEVICE: |2 dp2/ep2 rows => global 2xb (native b*ga*NUM_GPUS path).
  Field 3 = grad-accum; group 4 = policy|5 offload bools.
- GPU pool env is GPU_POOL (NOT GPUS); |2 rows need a same-superchip pair (0,1 or 2,3).
- Driver DETACHES the trainer (setsid): shell return != training done — watch
  profile.json / train.log. Trainer stdout is SWALLOWED — diagnostics must write FILES.
- profile.json is rank0-only under torchrun; every rank writes rank<R>_memstats.json;
  wall = max rank (aggregate_dp_ranks.py).
- PROFILERS is a single value (source|nsys|both); both drivers currently DEFAULT source.
- JIT: never quote first-run timings after kernel changes (the 245s cold artifact).
- Backend/|2 guards: plain asym backends die at |2; stp/dp2/ep2 backends die at |1;
  ASYM_STP and ASYM_DP mutually exclusive; MoE under sTP-family needs ASYM_STP_MOE=1
  (NOT needed for ep2 — no stp wrap).
- router whole->hf downgrade allow-list (~:4517 in each driver): every NEW |2 backend
  must be added there or MoE asymization silently turns OFF (bit us once under sTP).
- Completeness check hard-expected save_on_cpu=true => override rows misreported as
  incomplete/failed — FIXED 2026-07-07 (now tracks ASYM_GC_SAVE_ON_CPU_OVERRIDE).
- MoE under dp2 (DDP wrapper) hangs without ASYM_DP_FIND_UNUSED=true; ep2 avoids DDP
  entirely (manual allreduce) so the knob does not apply to it.
- DRIVER-SYNC RULE (user): _both.sh is MASTER — edit only it, copy over _source.sh
  (preserving source's in-file RUNS scratch + profiler default); never hand-edit source.
- pgrep -f self-matches the harness tool shell => a wrong-process kill happened once;
  take pids from lock/pid files (e.g. ceiling_search_state/driver.lock), never pgrep.
- Harness background tasks die at ~57 min: launch long drivers DETACHED
  (setsid nohup ... >> log 2>&1 &) and watch the log file.
- setsid FORK TRAP (cost two ghost-exit confusions 2026-07-07): `setsid cmd &` makes
  setsid the group leader, so it FORKS and the `$!` you captured exits immediately while
  the real driver lives on as the child. Watch the CHILD pid (ps after launch) or wrap
  in a compound (`{ setsid ...; } &`) — never watch bare `$!` of a setsid launch. Also:
  a "dead" driver may still be running — ALWAYS ps before relaunning the same row, or
  two drivers race on the same artifacts.
- External sglang server auto-respawns and can squat a GPU (took GPU 0 once, ~163 GiB):
  nvidia-smi before EVERY launch; if pair 0,1 is busy switch GPU_POOL=2,3; never pkill
  broadly — kill exact trainer pids only.
- GB200 host mem: free/top wrongly include ~736 GiB HBM — use the CPU-NUMA-only sampler
  (ceiling_search_state/cpumem.sh); host watchdog floor 35 GiB; NUMACTL_MEMBIND=0,1 always.
- Dataset prep: PREPARE_DATASETS=false skips name derivation (dp.md burned a pair run on
  the unsuffixed smoke dataset — pass DATASET explicitly when preparing is off).
- Steady state everywhere: MAX_STEPS=4 WARMUP_STEPS=1; drop warmup AND last step.
- Tree committed as 27dde72 (the earlier 'UNCOMMITTED' handoff note was stale). _C needs
  a rebuild after kernel-side changes (.venv/bin/python setup.py build_ext --inplace).
- "save-on-cpu OFF" was misread as "offloading off" once — it is ONLY the D1b
  recompute-side round-trip, symmetric on both rows; full asym offloading stays ON.
- Concurrent pinning is slow (dp2: 2x64 GB cudaHostAlloc ~10+ min, serialized) — the
  shared fabric exists partly to kill this; PR-1 must time register-vs-ATS.
- /dev/shm fabric files survive crashes: unlink stale asym_fabric_* at row start AND
  after the run (a stale file silently maps old weights).
- SKIP-IF-DONE TRAP (burned a 30-min remeasure 2026-07-08): the drivers skip rows whose
  profile.json exists at the SAME config path (reason=existing-complete), and PYTHON
  code changes do NOT change the config hash — a remeasure after code edits silently
  re-reads the STALE artifact (identical numbers to the decimal + missing new log lines
  are the tell). mv the old run dir to a __attemptN_<tag> sibling before relaunching.
- Mid-call host syncs (.item()/.tolist() on GPU tensors) are per-call RENDEZVOUS under
  vanilla-EP collectives: the drain waits for the peer's enqueue arrival, so per-rank
  host pauses stall BOTH GPUs, and per-call sync PAIRS oscillate (the 2.8-3.2x
  attempt-1 pathology). Keep sync count per MoE call at most 1 and enqueue tiny
  collectives before big ones (sEP never had this class — no per-layer coupling).
- NEVER EDIT DRIVER SCRIPTS WHILE A DRIVER IS MID-RUN: bash reads scripts
  incrementally by FILE OFFSET — an edit shifts the text under the running process,
  which then misparses at a random later line (cost: the 60k-retry post-processing,
  rc=2 at "line 4617"; training data survived only because it finished first). Queue
  script edits until the run exits, or accept losing the run's post-phase.
- NAMING EPOCH 2026-07-08 (user directive — backend NAME selects the EP mode, flags
  flip automatically in run_lf_lora_sft.sh; also puts the mode in the run-dir path,
  killing the skip-if-done mode collision):
    asym_ep2_cpuadamwds   = VANILLA EP (owned + dispatch)   [was: sEP before this date!]
    asym_sep2_cpuadamwds  = sEP plain  (ownerless, no queue)
    asym_sqep2_cpuadamwds = sEP+queue  (THE system; asym_sqeq2_* accepted alias)
  Short aliases asym_ep2 / asym_sep2 / asym_sqep2 work in RUNS rows. HISTORY HAZARD:
  every artifact dir named asym_ep2_cpuadamwds dated BEFORE 2026-07-08T12:00 is the
  OLD meaning (sEP, mode via env knobs) — read the Decision Log entries, not the dir
  name. Explicit ASYM_EP_VANILLA/ASYM_ARENA_SHM/ASYM_EP_QUEUED env for these backends
  is now OVERRIDDEN by the name-derived values.
```

## MEGATRON RECEIPTS (vendored third_party/megatron-lm; deep-read 2026-07-07 — why rank-per-GPU)

```text
- ONE PROCESS PER GPU, always: training/initialize.py:262 set_device(local_rank); all
  parallelism (TP/PP/DP/EP/CP) = torch.distributed process groups; a process NEVER
  drives two devices. No single-process-multi-GPU training path exists.
- ZERO host threads drive GPU work (grep-verified: threads only for data loading,
  checkpoint I/O, monitoring, inference serving). ALL async = CUDA streams + events from
  each rank's one main thread:
  activation offload: pipeline_parallel/fine_grained_activation_offload.py — saved-tensor
    hooks, dedicated d2h+h2d side streams, pinned CPU pool, backward prefetch, in-flight
    cap (events FIFO), storage.resize_(0);
  optimizer CPU-offload: optimizer/cpu_offloading/hybrid_optimizer.py — d2h/h2d streams,
    pinned grads, event.synchronize;
  grad sync: distributed/param_and_grad_buffer.py — autograd-hook-triggered bucketed
    reduce on a comm stream.
- HOST KEPT LIGHT: metadata D2H on a dedicated cuda_dtoh_stream joined by ONE deferred
  event.synchronize at the latest safe point (moe/token_dispatcher.py:437-498); grouped
  GEMM via TE GroupedLinear (experts.py — never a python loop over experts).
- THEIR EP (the piece we deliberately REPLACE): experts OWNED and HBM-RESIDENT
  (num_moe_experts/ep_size contiguous per rank) + all-to-all token dispatch
  (token_dispatcher.py:354-, mappings.py:446) + capacity/drop/pad balancing. Exists
  BECAUSE weights are HBM-resident. Ours are host-resident + ownerless: tokens never
  move, weights stream from the shared fabric, the union queue replaces capacity.
- In-process one-THREAD-per-GPU was rejected too: torch nn.DataParallel is that design
  and is deprecated for GIL contention; our host floor is python glue = worst case.
  (PyTorch's bwd engine already runs one thread per device; fwd/python is what serializes.)
```

## S-TRACK (staged, implementation-first)

### S0 — bar: run T1 (no code; ~15 min)

```text
COMMAND: the GOALS invocation with the T1 row only.
Landed prerequisites (2026-07-07): completeness checks honor ASYM_GC_SAVE_ON_CPU_OVERRIDE
  (both drivers; override rows were misreported incomplete). (ASYM_DP_FIND_UNUSED landed
  too — used only when the DEFERRED dp2 anchor runs at S3.)
EXPECT (write concrete numbers to the log BEFORE launching, per B1): T1 clearly below
  the dirty 15-20 s/step reference (that bar carried nsys+stats AND the D1b waste);
  HBM ~ prior |1-b8 class; loss ~ the known 1.7x-1.5x trajectory. Outside band -> B1 flag.
GATE: 4 steps complete; T1 recorded (O1) in the Decision Log; per-device HBM peak noted.
STALL RECEIPT (REQUIRED — this is the F1 A/B that was never actually run: the override
  knob landed 2026-07-07 but its validating run was killed by the pause, and the
  completeness bug would have misreported it anyway): ONE extra diagnostic run of the
  SAME T1 row with PROFILERS=both (nsys overhead accepted; NOT the timing bar), then
  analyze_stp_bwd.py -> ARTIFACTS: giant-gap count (the ~2.5s synchronized gaps) ->
  EXPECT 0; pageable-copy class -> EXPECT ~0 GB; GPU busy% recorded (dirty bar was ~55%).
  If gaps persist, D1b was NOT the whole story -> B1 flag; diagnose BEFORE building S1
  on a stalling substrate. Delete the diagnostic run's OUTPUT_ROOT after banking the
  receipt (it must never be quoted as a timing row).
```

### STALL PLAYBOOK (known GPU-idle classes; every B3 iteration picks from here FIRST)

```text
P1 recompute-side save_on_cpu (D1b): ~3 TB/step pageable round-trips = the ~2.5s
   synchronized giant gaps + the 7->34% busy ceiling. FIX LANDED (override knob);
   VALIDATION = S0's stall receipt (never run seriously before — the F1 A/B was killed).
P2 per-rank host floor: rank-per-GPU makes it |1's own floor (one thread <-> one GPU,
   the Megatron shape — the old 2x single-thread enqueue class is GONE BY DESIGN, and
   the worker-thread fallback is REJECTED, DataParallel precedent). SIGNATURE: rank
   busy% low with CPU pegged. FIX if hit: it is |1's problem too -> census -> P4/P6.
P3 [RETIRED by the rank-per-GPU design] per-layer scatter/gather P2P syncs — ep2 has
   ZERO per-layer cross-GPU exchanges (comm = one grad allreduce per step).
P4 residual per-call python in the MoE path (metadata/pad/route rebuilds the R4/R5
   memos missed; hidden .item() syncs). SIGNATURE: census python frames inside bwd.
   FIX if hit: route-plan-per-layer — ONE plan object per layer feeding ALL grouped
   calls (the old F3). Megatron analog: deferred single-sync metadata (receipts above).
P5 offload/restage serialization: should be DEAD after the R1-R3 async/event rework
   (receipts banked). Re-check only if censuses show stage/wait_cpu_ready frames.
P6 optimizer blocking per-param D2H (~1s/step measured) + fwd D2H on compute stream.
   FIX if ranked by census: batched pinned grad staging / fwd D2H side stream /
   backward prefetch-1 / in-flight cap (Megatron: hybrid_optimizer.py streams pattern).
P7 cross-rank stragglers (NEW with 2 ranks): the step allreduce makes the slow rank set
   the wall. SIGNATURE: rank step_samples diverge >10%. FIX: find the asymmetry (NUMA,
   dataset shard length, JIT) — dp.md D1 saw a 21s host-side scheduling asymmetry once.
```

### S1 — `asym_ep2_cpuadamwds`: rank-per-GPU sEP (each rank = the |1 stack VERBATIM)

The whole point: NO new model wiring. Two processes via torchrun; each rank runs the
already-validated |1 asym path (same code as the T1 bar) on its OWN 8-sequence shard with
ALL experts streamed from ONE shared pinned fabric. Three deltas total.

```text
SCOPE (exact files/functions/classes):
  asym_gemm/training/shared_fabric.py     NEW (~150 lines): class SharedFabric
  asym_gemm/training/offload.py           adopt_host_weight() (:176 —
                                          signature (name, tensor, component, *,
                                          require_2d, pin_memory_policy, strict)):
                                          fabric branch, ~15 lines
  scripts/lf/run_lf_profiled_train.py     (i) _ep2_wrap_model patch — CLONE the existing
                                          ASYM_STP pattern at :1357-1366, gated on
                                          ASYM_EP2=1; (ii) grad allreduce in the
                                          training_step patch region (:1449+, where the
                                          sTP merge hook runs: AFTER backward, BEFORE
                                          clip/optimizer); (iii) rank<R>_memstats.json
                                          already exists (:155-188)
  scripts/lf/run_lf_lora_sft.sh           backend case asym_ep2_cpuadamwds
  scripts/lf/profile_lora_lf_test_both.sh backend maps/tags/allow-list (then copy to
                                          _source.sh per the sync rule)
  scripts/testing/shared_fabric_probe.py  NEW probe (isolated class, allowed)
  NOT touched: stp_*.py (stays as the parity harness), qwen3_moe.py, kernels, liger,
  offload managers, cpu_adam. (The _n_gpu=1 patch at :1332 is ASYM_STP-gated and stays
  OFF here — under torchrun the Trainer is already per-process.)

DELTA 1 — sharded data: comes FREE from torchrun world=2 (HF/accelerate installs
  DistributedSampler; dp.md D2 validated the mechanics + loss semantics). Batch math is
  the native path: global = b*ga*NUM_GPUS (run_lf_lora_sft.sh:1307) => RUNS field 8 =
  8/rank, global 16. NO trainer batch hacks.

DELTA 2 — shared pinned weight fabric (kills dp2's 2x64GB + double pinning):
    # shared_fabric.py
    class SharedFabric:
        def __init__(self, run_id, rank, world):
            self.path = f"/dev/shm/asym_fabric_{run_id}"   # ONE file; unlink stale at start
            self.manifest = {}                              # name -> (offset, shape, dtype)
        def get_or_create(self, name, src: torch.Tensor) -> torch.Tensor:
            # rank0 (writer): carve the next 4KiB-ALIGNED range, mmap-view it as a
            #   tensor, copy_(src) ONCE, record manifest entry.
            # rank>0 (reader): wait for manifest (barrier below), mmap the same range.
            # Bank construction order is DETERMINISTIC (model traversal) => identical
            #   manifests; rank0 additionally json-dumps it for audit.
        def seal(self):
            torch.distributed.barrier()                    # all banks written
            cudart.cudaHostRegister(self.base_ptr, self.total_bytes, Portable|Mapped)
            # ONE register call per process for the WHOLE mapping — never per bank.
            # GB200 is C2C-coherent (dual-lane reads banked at 174.7 GB/s/lane); if
            # PR-1 shows register is slow, probe the ATS path (skip register entirely).
    # offload.py adopt_host_weight(): FABRIC WHITELIST = weight components only
    # (attention / experts / norms / router / lm_head). Activation pools stay PRIVATE
    # per rank — they hold per-shard data; sharing them would be WRONG, not just slow.
    if _fabric_enabled() and component in _FABRIC_COMPONENTS:
        return HostWeight(fabric.get_or_create(name, tensor))   # instead of private pin

DELTA 3 — grad sync WITHOUT DDP (kills G-D2.4 +21.3s reducer, find_unused hangs, hook
  conflicts):
    # run_lf_profiled_train.py (i): clone :1357-1366 as _ep2_wrap_model, ASYM_EP2=1 gate:
    def _ep2_wrap_model(self, model, *a, **kw):
        return model                       # NEVER DDP-wrap; we own grad sync
    # (assert int(os.environ["WORLD_SIZE"]) == 2 at patch install)
    # run_lf_profiled_train.py (ii), in the :1449+ training_step patch, after backward:
    if ASYM_EP2:
        assert grad_accum == 1             # reduce-per-step semantics; lift when needed
        views = _ep2_bucket_views(trainables)   # ONE persistent flat buffer + per-param
                                                # views, built once (no per-step allocs)
        torch._foreach_copy_(views, [p.grad for p in trainables])   # vectorized pack
        torch.distributed.all_reduce(flat_buffer)                   # ONE NCCL call ~1GB
        flat_buffer.div_(2)                                         # mean == DDP semantics
        torch._foreach_copy_([p.grad for p in trainables], views)   # unpack
    # Clip + CPU AdamW then see MEAN grads on every rank. The Adam update is
    # ELEMENTWISE => bitwise-identical masters across ranks given identical inputs, so
    # NO param broadcast/sync is ever needed (G-S1.b parity re-checks this anyway).

HARNESS:
  run_lf_lora_sft.sh: asym_ep2_cpuadamwds) USE_ASYM_CPU_ADAMW=true;
    ASYM_CPU_ADAMW_BACKEND=deepspeed; ASYM_DP=1  # reuse the torchrun launch path
    ASYM_EP2=1 ASYM_ARENA_SHM=1                  # no-DDP + shared fabric
    ASYM_CPU_ADAMW_GRAD_OFFLOAD=false ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false; BACKEND=asym ;;
  drivers (_both.sh master): token in append_backend_spec + backend_gpu_count |2 family +
    cpuadam map + derivation (ep2_enable=1 -> run_env += ASYM_DP=1 ASYM_EP2=1
    ASYM_ARENA_SHM=1; dir tag "_ep2") + router whole->hf allow-list += asym_ep2_cpuadamwds
    (THE silent-MoE-disable bug class). LF parser gate already opens under ASYM_DP=1.

PROBE FIRST (scripts/testing/shared_fabric_probe.py; isolated class — allowed):
  PR-1 /dev/shm mmap + single cudaHostRegister of ~64 GB: TIME IT per rank (dp2 saw
       ~10 min for 2x cudaHostAlloc; expect register of resident pages much faster —
       if not, B1 flag and probe the ATS path: GB200-coherent GPUs reading UNregistered
       system memory).
  PR-2 both GPUs (2 PROCESSES) run the asym grouped GEMM streaming B from the SAME
       fabric range concurrently -> bandwidth per lane vs the banked 174.7 GB/s.
  PR-3 2-process atomicAdd_system on a fabric counter page (for S2b).

CORRECTNESS ARGUMENT: each rank is numerically the |1 path on its shard; loss = per-rank
  mean, grads averaged (div 2) == the D2-validated DDP mean semantics; equal-length
  shards (fixed-s packing) make this equivalent to a |1 b16 run up to reduction order.
  ORDERING PRECEDENT (verified in vendored DeepSpeed ZeRO-Offload, stage_1_and_2.py):
  grads are averaged ON GPU first (reduce_ipg_grads -> average_tensor /
  gradient_reduction_w_predivide :1144) and only then copied to pinned CPU fp32 buffers
  (copy_grads_in_partition :1512 -> async_accumulate_grad_in_cpu_via_gpu :1378; pinned
  at :808) — CPUAdam NEVER sees unreduced grads. Our allreduce-then-D2H matches. Their
  reduce_scatter refinement (each rank averages+downloads only its 1/N partition,
  stage-2 default :154/:292, params allgathered after step) = a later P6-class option
  (halves D2H + CPU-Adam work; not worth it at ~1 GB LoRA scale until censuses rank it).
EFFICIENCY ARGUMENT: host floor = |1's own (one thread, one GPU — Megatron shape);
  ZERO per-layer cross-GPU traffic; comm = ONE ~1-2 GB allreduce/step (~ms on NVLink);
  C2C streaming = dual-lane (banked concurrent); host RAM = ONE arena + per-rank
  activation pools (genuine per-shard data). GEMM shapes/launch count per rank == |1-b8
  exactly — no new kernels, no extra launches, no small GEMMs.
EXPECT (log per B1 before the perf run): T_ep2 in [1.0, 1.6) x T1 (it is two |1 runs +
  ~ms allreduce + shared-fabric contention); >= 2xT1 -> B3 loop; < ~0.95xT1 -> too-good
  B1 flag (suspect short-circuit / wrong shards / wrong global batch).
GATE S1 (all e2e except the probes; close per B2):
  G-S1.a CORRECTNESS/shard receipt: per-rank first-step sample ids dumped to
        $seq_root/rank<R>_shards.json -> disjoint, union == the global batch (dp.md
        G-D2.1's receipt, finally banked).
  G-S1.b CORRECTNESS/numerics: e2e s2048 loss overlay: T_ep2 (global 16) vs |1 b16
        s2048, fixed smoke set, no shuffle, transplanted init -> PASS within the bf16
        envelope, 5 steps. ARTIFACT = both profile.json loss columns. FAIL BLOCKS S1.
  G-S1.c EFFECTIVENESS: the GOALS pair at s20000 -> O2 verdict (T_ep2 < 2xT1 STRICT);
        ARTIFACTS = step_samples both rows (T_ep2 wall = max rank), rank<R>_memstats
        (HBM < 184 GiB/rank; summed RSS ~ |1's + one arena — NOT dp2's ~493), fabric
        register time from PR-1 logged.
RISKS / WATCH:
  W-S1.1 register-vs-ATS on GB200 (PR-1 settles); TMA-from-fabric rate (PR-2 settles).
  W-S1.2 HF sampler shard determinism + drop_last at tiny smoke datasets (dataset-suffix
        trap in GOTCHAS); the shard receipt catches it.
  W-S1.3 both ranks JIT-compiling the same cache dir on first run (warm box OK;
        pre-warm rule; dp.md saw the race).
  W-S1.4 per-rank host watchdogs both fire on one squeeze (HC2) — floors already set.
  W-S1.5 rank asymmetry (P7): NUMA placement per rank (membind=0,1 stays; do NOT split
        nodes per rank without a probe).
  W-S1.6 transient host-RAM spike: BOTH ranks from_pretrained() the checkpoint during
        surgery (~2x model bytes transiently) before fabric dedup kicks in.
  W-S1.7 fabric lifecycle: stale /dev/shm file after a crash maps OLD weights silently —
        unlink at row start + after the run (also in GOTCHAS).
```

### S2 — queue-kernel balancing (the sEP mechanism), fwd base GEMMs first

Kernel exists and is banked (E3 probe: bitwise-exact, balance <= 4% at any skew, 0.969
balanced overhead; entry: `m_grouped_bf16_asym_gemm_nt_contiguous_ep_queued(a, b, d,
offsets, experts, list_size, ep_queue, ep_side, compiled_dims)` — csrc/apis/gemm.hpp:625).
CRITICAL ENABLER: the counter pops are ALREADY system-scope (`atomicAdd_system`,
sm100_bf16_asym_gemm.cuh:148-153) -> counters on a shared-fabric pinned page work ACROSS
THE TWO RANK PROCESSES. d/X cross-access via the fabric (X) + CUDA IPC (peer d).

```text
S2a — queued launches, ZERO steal (plumbing gate; no cross-rank behavior change):
  Each rank swaps its fg base GEMM calls (fwd gate/up/down; frozen_linear.py:732/:753
  call sites) to the queued entry over its OWN item list + its OWN private pinned
  counters (side fixed per rank; inert until S2b — call signature final).
  ITEM LISTS (vectorized, memoized on the metadata like the substrate memos — NO python
  per-expert loop):
    def build_ep_item_lists(md, block_m=128):
        counts = md.expert_counts
        blocks = (counts + block_m - 1) // block_m
        e_ids  = torch.repeat_interleave(arange_i32(E), blocks)
        base   = torch.repeat_interleave(md.expert_offsets[:-1].int(), blocks)
        first  = torch.repeat_interleave(cumsum_exclusive(blocks), blocks)
        offs   = base + (arange_i32(total) - first) * block_m
        return offs, e_ids, int(blocks.sum())
  Counter blocks from a rotating pre-zeroed pinned pool (reset = CPU write, never a
  device sync); d zero-initialized; assert claimed == list_size per launch (file-logged).
  EXPECT (B1): ~neutral vs S1 (banked 0.969); >3% slower or ANY parity bit-diff -> flag.
  GATE S2a: CORRECTNESS: stp_moe_block_parity.py --queued result file (bitwise) +
    claimed==total dumps. EFFECTIVENESS: e2e s20000 T_ep2 within 3% of S1.
S2b DESIGN REFINEMENT (2026-07-08, from kernel constraints — supersedes the d-peer-IPC
  sketch below): TMA cannot store to PEER HBM (tensor maps target local global memory or
  SYSMEM), so stolen items do NOT write through an IPC pointer. Instead: stolen items'
  D tiles TMA-store into a FABRIC (pinned sysmem) staging slice — the same mechanism
  that already streams B FROM sysmem, in reverse — and the OWNER rank gathers exactly
  its stolen rows back H2D after the launch pair (bytes ∝ stolen fraction, ZERO when
  balanced — the EG-V2 invariant holds). A-side for stolen items reads from the fabric
  X pool (sysmem TMA loads — proven by the B-streaming path + PR-2). Kernel delta:
  second tensor_map_a (fabric X) + second tensor_map_cd (fabric D staging) + n_own
  boundary selecting map pairs per item; NO peer pointers, NO IPC. The union list =
  [rank0 segs | rank1 segs]; sides pop own-first (front/back per section).
S2b IMPLEMENTATION CONTRACT (2026-07-08, from kernel reads — the buildable spec):
  KERNEL (.cuh, new define ASYM_BF16_EP_STEAL implying EP_QUEUED; S2a variant untouched):
    extra params: tensor_map_a_peer + tensor_map_cd_peer (__grid_constant__ TMA descs
    over FABRIC sysmem — the B path proves host descriptors) + uint32_t ep_n_own.
    Per claimed item: ep_local = (ep_side==0) ? (item < ep_n_own) : (item >= ep_n_own);
    A-loads (:488/:491) and CD-stores (:1172-1195) switch descriptor POINTERS on
    ep_local (maps are kernel params; selection is a pointer pick, no codegen fork).
    Foreign items' m-ranges are in the PEER pack's row space (union offsets built so).
  CONTIGUOUS-STEAL PROPERTY (makes accounting + gather trivial): with the union list
    [rank0 segs | rank1 segs], side0 pops head-up, side1 pops tail-down => each rank's
    claims are ONE contiguous range; steal = head > n_own (rank0 stole [n_own, head))
    or tail > total - n_own (rank1 stole [total - tail, n_own)). Post-launch head/tail
    readback gives EXACT stolen ranges — steal bytes accounting AND owner gather-back
    are contiguous-slice ops.
  CROSS-PROCESS COMPLETION (zero host syncs): claims != completion, so per launch the
    thief-side stream enqueues the GEMM then a tiny flag kernel (st.release.sys into a
    fabric int32) — stream order guarantees the GEMM's sysmem stores are visible before
    the flag; the OWNER enqueues a spin(+gather) kernel on its stream BEFORE d's
    consumers: spins on the flag (GB200-coherent sysmem read, ~us when already set),
    then gathers ONLY the stolen contiguous range from fabric D staging H2D. Balanced:
    flag already set, empty gather => ~zero cost. Two tiny kernels in csrc (flag_set,
    spin_gather) compiled into _C.
  X-STAGING SCOPE (v1 honesty): steal rows read X from the fabric, so each rank stages
    its packed a_kernel slice to fabric pre-launch WHEN STEAL IS ARMED (ASYM_EP_STEAL=1
    rows only; ~2.6 GB/layer-phase D2H). Balanced no-regression rows run DISARMED
    (= S2a semantics, already gated). Skew-ladder rows arm it on BOTH compared modes =>
    fair mechanism receipts. v2 refinement (logged, not v1): re-key the EXISTING
    dA-path x_cpu staging into the fabric => zero new traffic when armed.
  PYTHON: union offsets/experts built identically on both ranks (rank0 pack coords ++
    rank1 pack coords); n_own = rank0's segment-item... n_own is in ITEM units => pass
    n_own_segments * n_blk? NO — n_blk is JIT-internal: pass ep_n_own in SEGMENT units
    and compare against item/ep_num_n_blocks inside the kernel (segment index), which
    the queued path already computes (:322-326). Counters/flags from fabric pages.
S2b — UNION queue + affinity + steal (cross-RANK balancing):
  - X pool: re-key the existing x_cpu/dA packed-X staging into the SHARED FABRIC keyed
    (rank, layer, phase) — layout change, not new traffic; either rank can read any tile.
  - counters: ONE pinned page in the fabric per (layer, phase) pair — kernel unchanged
    (already _system scope; PR-3 probed).
  - d writeback: stolen item's output rows write through a CUDA-IPC-mapped pointer to
    the OWNER rank's d buffer (cudaIpcGetMemHandle exchanged once at init over
    torch.distributed; NVLink peer write).
  - kernel/launcher extension: item list = [rank0 items | rank1 items] (n_own marks the
    boundary); per launch pass (a_own, a_fabric, d_own, d_peer, n_own): own item -> A
    from HBM, D local; stolen item -> A tile TMA-streamed from the fabric (the SAME
    machinery that already streams B from host) + D via the IPC peer pointer.
  - affinity: rank0 launches ep_side=0 (pops front), rank1 ep_side=1 (pops back); each
    drains its OWN section first; steals only cross the midpoint when a rank runs dry.
  - LoRA grouped GEMMs stay per-rank full-range (~3% of expert work — E3.5 rationale);
    only the host-streamed BASE GEMMs queue. bwd dX queued lands in S3.
  EXPECT (B1): balanced ~ S2a (steal fires only past the midpoint); skew-ladder wins per
    the banked ratios direction (muted by MoE fwd fraction ~37%); steal bytes ~0 natural.
  GATE S2b: CORRECTNESS: block parity --queued-union (bitwise incl. stolen items) + e2e
    s2048 loss envelope + claimed==total with steal on. EFFECTIVENESS: s20000 balanced
    within 3% of S2a; SKEW LADDER (timing-only, loss-INVALID by design)
    ASYM_EP_SKEW_ACK=1 ASYM_EP_SKEW_HOT=0.25|0.50|0.75 -> step win at alpha >= .5 vs S1,
    _EpBalanceStats histograms (balance <= 5%), steal-byte counter file (~0 balanced —
    EG-V2); scout 9500|8|1 natural-skew row = the gains case.
  RISKS: A-from-fabric TMA rate (PR-2 extension); IPC d-write ordering (producer events;
    claimed==total assert); watchdog floors with the pooled layout (HC5 audit).
```

### S5 — BALANCING PROOF (user-mandated 2026-07-08: recorded imbalance -> cure, micro + e2e)
### STATUS: S5a CLOSED (micro, token-floor receipts) + S5b/N2 CLOSED (vanilla-EP e2e,
### skew ladder + root-cause receipts) — see the ===== entries in the Decision Log.

```text
THE CLAIM TO PROVE: ownership-induced imbalance (RECORDED on real data: static-E/2
  device share q3 mean 0.530/worst 0.631, scout mean 0.608/worst 0.872 = up to
  1.26x/1.74x worst-layer walls) is CURED by ownerless queue-splitting, at micro AND
  e2e granularity. ep2-vs-ep2 cannot show this (no rank imbalance by construction) —
  the A/B needs the DISEASED baseline in the race.
S5a MICRO (targeted, tuned + real-routing; extends shared_fabric_probe):
  - load RECORDED per-layer expert histograms (profiling_gb200ep_e1/
    ep_balance_natural_s2048.json + _e1b scout) -> segment lists at prod-scale M with
    REAL routing counts; 2 processes, real banks/shapes.
  - modes per layer-histogram x alpha in {natural, 0.25, 0.5, 0.75}:
      OWNED-STATIC: rank i executes exactly its owned-64 experts' segments (fixed
        partition) — wall = max rank;
      sEP QUEUE: union list, front/back pops (the PR-4 path) — wall = max rank.
  - GATES: tuned-alpha ratios reproduce the banked kernel class (4.24x/6.94x/8.58x);
    natural-histogram ratios land in the predicted bands (q3 1.06-1.26x, scout
    1.22-1.74x per-layer); balanced overhead <= 1.02 (EG4).
S5b E2E — AS BUILT 2026-07-08 (design pivot logged in Decision Log): the REAL
  vanilla-EP baseline (user N2: different data per GPU on BOTH sides) is the
  ALLGATHER-DISPATCH rung (asym_gemm/training/ep_vanilla.py), NOT the parked
  ep_steal-n_own=0 sketch — Megatron's own dispatcher shape, NCCL collectives ARE
  the pairing, and at topk=8 allgather(hidden) moves ~4x less than row-a2a:
  - ASYM_EP_VANILLA=1 on the asym_ep2 backend: wrapped_train swaps every MoE block's
    experts for its stp_moe.slice_experts_for_ep OWNED branch (dim-0 bank views,
    per-rank LoRA slices) BEFORE optimizer creation; AsymQwen3Experts.forward wraps
    _forward_impl with unconditional entry allgather(hidden,topk) -> ep_expert_range
    partial over GLOBAL tokens -> exit reduce-scatter (differentiable; GC-recompute
    lockstep). Expert-LoRA grads are OWNER-sharded: x 1/world, excluded from the
    shared-param allreduce (structural equivalence proof in ep_vanilla.py docstring).
    ASYM_ARENA_SHM forced 0 (no ownerless arena — private per-rank pins, same
    pinned-C2C transport class); ASYM_EP_QUEUED forced 0 (owned-static baseline).
  - RUNS: q3-30b-a3b 20000|8|1 natural + ASYM_EP_SKEW_HOT ladder (ACK'd, timing-only)
    vanilla vs sEP-queue. llama4-scout DEFERRED (forward_input_scaled not hooked).
  - B1 EXPECTATIONS: natural q3 vanilla ~ +3-8% over sEP (imbalance walls ~1.06-1.14x
    on the MoE-GEMM segment + ~0.3-0.5 s NVLink collectives + row overhead); skew
    alpha=0.5 vanilla degrades ~1.5x on the MoE segment while sEP stays ~flat (equal
    per-shard skew = no cross-rank imbalance; queue absorbs intra-rank). RED FLAGS:
    vanilla FASTER than sEP anywhere (transport asymmetry leaked), natural delta
    > +17% (strawman/stall — check NCCL waits), loss overlay miss vs sEP (grad-scale
    or lockstep bug).
  - GATE: EG1/EG2/EG4 verdicted e2e + C-EP1/C-EP2 receipts named; results feed
    gb200_ep.md E4's ladder table.
```

### S3 — bwd dX queued + steal accounting + SG-FINAL freeze

```text
SCOPE: the routed-kernel family used by bwd dX (down_dx_gather_left ->
  qwen3_moe_bf16_down_dx_gather_left_) gets the SAME EP_QUEUED codegen treatment on its
  runtime class (mechanical mirror; n_blk fixes already in the shared .cuh) — E3.5
  constraint (ii). dX of stolen items returns to the token owner (mirror of fwd).
ACCOUNTING: per-step steal bytes + stolen-item counts per rank -> profile.json (EG-V2).
GATE S3 (artifact-named or not closed):
  CORRECTNESS: bwd-dX block parity file (queued vs plain routed) + full loss overlay.
  EFFECTIVENESS: e2e s20000 T1+T_ep2 rerun -> SG-FINAL verdict with step_samples +
    steal columns + scout row. OPTIONAL honesty pair: override unset (save-on-cpu ON
    both sides) — verdict must hold.
  THEN (and only then) the DEFERRED dp2 anchor runs once (O4; ASYM_DP_FIND_UNUSED=true;
  ARTIFACT = dp_row.json); THEN the paper ladder per gb200_ep.md E4/E6 (AsymLoRA-sEP =
  T_ep2; sharded EP-Static / hostsplit are mechanism ablations, NOT SG blockers).
```

### S6 — TRUE sEP: dynamic expert-work partitioning over the shared bank (2026-07-09)

```text
PITCH (user): weights NEVER move (one shared fabric, streaming identical to sdp2);
  the "partition" is per-layer MENTAL MATH — an ephemeral assignment of the UNION of
  both shards' expert-segments deciding WHICH GPU's kernels stream WHICH banks into
  their SMEM this layer. Cluster tokens onto few experts per GPU => each bank is
  streamed ONCE across the pair (vs twice in sdp2) => wins where streaming binds
  (small M-per-expert); residual imbalance absorbed by the union queue/steal (the
  PR-3/4 kernels finally doing the job they were built for). Only ACTIVATIONS move
  (X staged async to fabric, peer reads via TMA sysmem; Y back via staging + gather —
  the host-sync-free PR-4 transport; NO collectives in the token path).
NAMING: asym_sep2/asym_sqep2 are REPURPOSED for this system once code lands (the
  alias-canonicalization to sdp2 from epoch 3 gets removed then): sep2 = dynamic
  partition (planner flavor), sqep2 = + steal queue (residue flavor). sdp2/sqdp2/ep2
  unchanged.
ON-PAR-BY-CONSTRUCTION RULE (user requirement: never another vanilla): sdp2 IS a
  point in sEP's assignment space (assignment := own-tokens-only => zero cross
  traffic, zero staging). The planner takes cross-GPU work ONLY when projected
  bank-streaming saving > projected X/Y transport cost (per-layer cost model from
  routing counts). Worst case = sdp2 wall +- the measured 0-2% counter overhead.
DESIGNS NEEDED:
  D-A union work list over BOTH shards with STATIC-CAPACITY slots (fix_vanilla_ep
      D1/D3 substrate: shapes never depend on routing => no per-layer metadata
      exchange, no .item()s — the anti-stagger discipline).
  D-B X/Y transport via fabric staging: async D2H stage of cross-assigned rows'
      X, peer TMA-reads sysmem (PR-2 lanes 156 GB/s), Y to staging + spin_gather
      (PR-4, bitwise-proven), st.release/ld.acquire flags. Zero NCCL in-layer.
  D-C the planner: greedy LPT over per-expert counts with the sdp-floor cost rule;
      ALTERNATIVE flavor: no planner — partition EMERGES from union-queue pops
      (micro decides which flavor e2e gets first).
  D-D sqep2 = + steal for residue + per-step steal/transport accounting (EG-V2).
  D-E grad semantics: executor computes lora for rows it runs; allreduce-mean is
      location-independent (same proof as the vanilla rung's overlay) => no fork.
  D-F instrumentation: per-rank STREAMED-BANK-BYTES counter, cross-assigned-rows +
      transport-bytes counters, host-block probes, antiphase canary.
METRICS + MODELS/WORKLOADS (the eval matrix):
  MICRO (extend ep_balance_bench: third mode "sdp" = own rows x all experts; new
  axis M-per-expert sweep 10k -> ~200 rows/expert; report walls + per-rank bank
  bytes + imbalance):
    MG-A balance: union cures alpha-skew (have: 0.835 -> <=0.7%);
    MG-B streaming: union < sdp wall in the small-M/expert regime with bank-bytes
         receipt ~= half per GPU; MG-C no regression at compute-bound (within 2%).
  E2E rows (all steady raw2-4, natural + alpha {0.10, 0.15}):
    q3-30b-a3b|2  2048|8|1   — streaming-lean showcase (M/expert ~1k): EXPECT the
                               bank-once win to show (>=5% vs sdp2, growing shorter);
    q3-30b-a3b|2  20000|8|1  — compute-bound parity row: MUST tie sdp2 within 2%;
    q3-30b-a3b|2  60000|8|1  — sweet-spot parity row: tie within 2%;
    llama4-scout|2 9500|8|1  — STRUCTURAL-skew showcase (E=16, banked dev-share up
                               to 0.87): natural routing alone should show the
                               partition+steal absorbing real imbalance (needs the
                               scout fwd hook — currently qwen3-only, port needed);
    (frontier, optional) q3-235b-a22b short-seq — the large-E streaming-bound case.
  METRICS per row: step wall + tokens/s; per-GPU streamed-bank GB (THE mechanism
  receipt: ~1x total vs sdp2's ~2x); cross-assigned rows + transport GB; per-rank
  busy imbalance; host-block probe ~=0 + no antiphase (quotability bar); loss
  overlay <=0.01 vs sdp2 on natural rows; HBM peak + host RSS.
GATES: G1 overlay; G2 parity rows tie sdp2 +-2%; G3 streaming rows win with the
  bank-bytes receipt; G4 skew ladder flat (like sdp2) AND scout-natural improves;
  G5 sync hygiene green. Any red = stop, diagnose to receipt (B1), never quote.
BUILD ORDER: micro third-mode + M-sweep (GPUs 2,3, cheap) -> D-A/B substrate ->
  D-C planner-vs-queue decision from micro -> e2e parity rows -> showcase rows.
```

## STATUS + HISTORY (receipts; pre-pivot numbers are from the RETIRED replicated vehicle)

```text
PIVOT (2026-07-07 late, USER): replicated-batch |2 rows RETIRED as run/opt targets.
Numbers below are historical receipts from that vehicle; the live plan is the S-track.

MODEL/WORKLOAD: Qwen/Qwen3-30B-A3B (E=128 top-8, ker101 routed kernels), LoRA r64, real
  smoke data, s20000, GPUs 0,1. |1 reference: 15-20 s/step WITH nsys+stats AND save-on-cpu
  waste (the CLEAN bar = S0/T1, never yet measured).
SUBSTRATE FIXES BANKED (R1-R5, inherited per rank by ep2; receipts in gb200_ep.md log):
  bucketed pinned offload pool (R1); async event-based restage x2 machineries (R2/R3);
  metadata/pad memoization x7 (R4); producer-memo root fix (R5). Replicated-vehicle bwd
  went 245 -> 49-67 s across R1-R5; GPU busy 7% -> 34%.
D1b ROOT CAUSE (closed): recomp-off-full-fg stages force GC + recompute-side save_on_cpu
  => ~3 TB/step pointless round-trips = the giant gaps + pageable-copy class. POLICY
  (row-typed, settled): throughput rows run ASYM_GC_SAVE_ON_CPU_OVERRIDE=false on EVERY
  system (symmetric); frontier/capacity rows keep it ON (its actual purpose).
KERNEL sEP WINS (banked, probe): balance <= 4% at any skew vs static 82-94%;
  static/queue 4.24x/6.94x/8.58x at alpha=.25/.5/.75; balanced overhead 0.969; bitwise
  exact. Natural skew: scout dev-share mean 0.59-0.61 (worst 0.87); q3 mild (0.53);
  MoE fwd fraction q3 36.9%, scout 23.7%.
DP ANCHORS (dp.md): D1 pair probe = ~no shared-Grace contention at the dense dev row;
  pairing preserved 1.95x per-GPU throughput. D2 Route A validated (bit-identical
  post-reduce grads; VG1 1.12x vs superoffload-DP2) with honest costs: +21.3s DDP
  reducer bwd (G-D2.4 OPEN), RSS 247 GiB/rank (2x arenas — VG2 FAIL), MoE row (D2.5)
  never run. ep2's three deltas target exactly these three costs.
STRUCTURAL THESIS (settled by the Megatron deep-read): host-bound MoE + one thread
  driving two GPUs was the core defect of the retired vehicle; the industry shape is
  one process per GPU with stream-async everything — which is what ep2 adopts.
LOSSES: stable at 4 decimals through every substrate fix; block parity exact.
```

## LANDED CHANGES (inventory)

```text
KERNEL (sEP queue, probe-validated): asymScheduler.cuh (explicit-ids ctor + n_blk);
  sm100_bf16_asym_gemm.cuh (ASYM_BF16_EP_QUEUED entry-pop variant, atomicAdd_system
  counters, n_blk fixes); sm100_bf16_asym_gemm.hpp (SM100BF16EpQueuedAsymGemmRuntime +
  launcher + DG_EP_QUEUE_GRID_PCT); gemm.hpp + __init__.py (…_ep_queued python entry).
SUBSTRATE PERF (R1-R5): activation_offload.py, attention_activation_offload.py,
  frozen_linear.py, cpu_left.py, qwen3_moe_routed_gemm.py, moe.py, lora.py,
  qwen3_moe_finegrained.py (pools/async/memos — see gb200_ep.md log).
sTP/EP WIRING (kept ONLY as the replicated PARITY harness): stp_runtime.py,
  stp_functions.py, stp_wrap.py, stp_moe.py, qwen3_moe.py ep hook + _EpBalanceStats.
HARNESS: both drivers: router whole->hf allow-list includes the |2 asym family; ASYM_EP_*
  passthroughs; ASYM_GC_SAVE_ON_CPU_OVERRIDE hook + completeness checks honoring it
  (2026-07-07); run_lf_lora_sft.sh: sEP guards, ASYM_DP_FIND_UNUSED knob in the DDP
  block (2026-07-07); dp2 run_env passthrough of the knob.
TESTS/TOOLS: ep_queue_probe.py (kernel probe PASS), stp_moe_block_parity.py (PASS),
  stp_full_tp_mini_parity.py, analyze_stp_bwd.py, aggregate_dp_ranks.py.
ARTIFACTS: profiling_gb200ep_e3 (probe), _e1* (skew histograms), _e2e (parity + fix
  rounds), _diag2..8 (censuses); dp anchors in profiling_gb200dp_*.
```

## Measurement discipline

```text
Steady-state: MAX_STEPS=4 WARMUP=1, report middle steps. JIT-warm rule. One change per
run; loss column EVERY run (envelope method for cross-shape parity). Census loop: py-spy
burst -> top frames -> fix -> steady rerun; nsys only when ambiguous (py-spy is blind to
native frames on ARM). Diagnostics ALWAYS to files. Every stage verdict appends to the
Decision Log here; kernel-level receipts go to gb200_ep.md's log.
E2E RULE (user 2026-07-07): stages gate on the s20000 e2e LoRA profiling rows; s2048 =
numerics parity only; isolated probes only for kernel/system-internal changes.
```

## Decision Log (append-only)

```text
2026-07-07 doc created mid-campaign; R1-R5 receipts in gb200_ep.md's log.
2026-07-07 D1 closed (disabled branch); D1b root cause landed (stage-forced GC +
  recompute save_on_cpu, ~3 TB/step waste); row-typed save-on-cpu policy settled with
  user (OFF for throughput rows both sides, ON for frontier rows); override knob landed
  in both drivers; T1/2 framing dropped (SG-FINAL = weak scaling 8+8 vs 8).
2026-07-07 PAUSED BY USER mid-F1-A/B (run killed). sglang had respawned onto GPU 0
  during the pause sweep. Handoff state written; tree later committed (27dde72).
2026-07-07 (late) USER HARD RULE: replicated-batch |2 retired ENTIRELY as a run/opt
  target (wrong target, drift risk; parity harness only). LANDED: ASYM_DP_FIND_UNUSED
  knob + completeness checks honoring ASYM_GC_SAVE_ON_CPU_OVERRIDE (both drivers).
  Old A-track (F1 A/B on replicated) DROPPED; D2/D4/F2 dropped; D5 folded into S0.
  DRIVER-SYNC RULE: _both.sh is MASTER; copy to _source.sh, never hand-edit it.
2026-07-07 (late) DOC RESTRUCTURED (user): implementation-first S-track; GOALS holds the
  rows + O-verdicts; BEHAVIOR RULES B1-B5 added (B1 predict-then-measure with red-flag
  bands); STAGE-GATE artifacts mandatory; STALL PLAYBOOK restored (P1 D1b never
  validated -> S0 stall receipt).
2026-07-07 (late) USER: dp2 DEFERRED until EP2-beats-EP1 is proven (O2); it then runs
  ONCE as the S3 anchor (O4). Win condition canonical: step < 2xstep(T1) STRICTLY
  (2x break-even, 1.5x TARGET, 1.2x GOOD, 1.0x perfect-scaling bound).
2026-07-07 (late) NAMING: the sEP backend token is asym_ep2_cpuadamwds (user: "we are
  doing EP"; the earlier asym_tp2 draft collided with the dense tp2_* family).
2026-07-07 (late) DESIGN PIVOT (user + Megatron-LM deep-read, receipts in MEGATRON
  RECEIPTS section): S1 is now RANK-PER-GPU — torchrun 2 ranks, each rank the |1 asym
  stack VERBATIM; sharded data via DistributedSampler; SHARED pinned weight fabric
  (/dev/shm + ONE cudaHostRegister per rank; GB200-coherent; sglang precedent in-tree);
  NO DDP wrapper — ONE coalesced manual LoRA-grad allreduce per step (kills G-D2.4
  reducer overhead + find_unused + hook conflicts); per-rank CPU AdamW on averaged
  grads (dp2-validated pattern; Adam is elementwise => identical masters, no param
  sync). The single-process SepDecoderLayer S1 draft is DROPPED (GIL host floor = P2;
  worker-thread patch rejected — DataParallel precedent; Megatron uses ZERO GPU-driving
  host threads). S2 union queue goes cross-process: counters are ALREADY
  atomicAdd_system (sm100_bf16_asym_gemm.cuh:148-153) on a fabric pinned page; X pool in
  the fabric; peer d via CUDA IPC. Probes before S1: PR-1 shm register/ATS timing, PR-2
  dual-process TMA-from-fabric bandwidth, PR-3 2-process counter atomics.
2026-07-07 (late) CONSISTENCY PASS (user checklist): doc reordered GOALS -> BEHAVIOR
  RULES -> GOTCHAS at top; B2/B3 now order immediate advance on PASS (never idle, never
  stop before SG-FINAL); S1 pseudocode pinned to verified anchors
  (run_lf_profiled_train.py:1357-1366 wrap-patch pattern to clone, :1449+ allreduce
  site, offload.py:176 adopt_host_weight signature); added: grad_accum==1 assert,
  persistent flat bucket + _foreach_ pack/unpack, fabric component whitelist
  (weights-only; activation pools stay per-rank), 4KiB bank alignment, stale-/dev/shm
  cleanup, transient 2x checkpoint-load RAM (W-S1.6/7). Remaining uncertainties are
  DELIBERATELY probe-resolved, not searchable (hardware-specific): PR-1 register-vs-ATS
  timing, PR-2 TMA-from-fabric rate, PR-3 cross-process system atomics.
  S-track implementation NOT started — plan only.
2026-07-07 (late) EXECUTION STARTED (user GO: focus = gb200_ep.md goals; execute until
  achieved). Machine check: all 4 GPUs free, /dev/shm 479G, venv OK, no trainers running.
S0 B1 EXPECTATION (logged BEFORE launch): T1 clean bar (PROFILERS=source, stats off,
  save-on-cpu OFF) -> EXPECT step_s 8-16 s (the dirty 15-20 s bar carried nsys+stats AND
  the D1b waste at half-|2 volume); HBM peak 50-90 GiB class; loss ~1.77/1.70/1.76
  first-steps pattern (same seed/data as prior b8 s20000 rows). RED FLAGS per B1:
  step > 20 s => override ineffective (check unsloth_gc_recompute_save_on_cpu=false in
  the run's command.txt/config echo); step < 5 s => too-good/short-circuit. Command =
  the GOALS invocation with the T1 row only; OUTPUT_ROOT=profiling_gb200ep_sg (fresh).
2026-07-07 S0 T1 MEASURED (O1) + B1 RED FLAG RESOLVED (validated; artifacts:
  profiling_gb200ep_sg/.../asym_cpuadamwds__source__...__gradofftrue_*/b8_s20000_ga1/
  profile.json + train.log):
  T1 = 58.5 / 60.7 / 62.4 s steady (drop warmup 107 s + last) => T1 ~= 60.5 s.
  fwd 11.2 s | bwd 44.3-47.5 s | opt 1.1 s (+3.0-3.8 s async update-side).
  HBM peak 25.7 GiB alloc / 31.3 reserved. RSS ~344 GiB. Loss 1.77/1.70/1.76/1.62/1.54
  (matches the known trajectory). unsloth_gc_recompute_save_on_cpu=false CONFIRMED in
  config. asym calls verified (fwd 3360 / dx 2160). Driver auto-label: ker101-ceil0000-
  ohbm0 (defaults appended).
  RED FLAG (measured 60 s vs expected 8-16 s) RESOLVED AS A DOC ERROR: the "15-20 s |1
  reference" in the campaign docs was an s2048-CLASS number (matches '|1 bwd 9-14 s at
  s2048' receipts), mislabeled as the s20000 row. Physics: fwd alone is 11.2 s at
  160k tok; bwd >= ~2x fwd under GC => a 15-20 s step at s20000 was never possible.
  EVERY prior doc line citing '|1 = 15-20 s/step at s20000' is CORRECTED to T1 ~= 60 s.
  NOTE for S1: bwd 45 s vs an async-ideal ~5-10 s (compute ~2-3 s + streams) says |1
  itself still carries stall classes (P4/P5/P6 candidates) — the stall receipt (next)
  characterizes them; they are |1-inherited, symmetric for the O2 ratio.
S0 STALL-RECEIPT B1 EXPECTATION (before the diagnostic run): same T1 row with
  PROFILERS=both, OUTPUT_ROOT=profiling_gb200ep_sg_diag (deleted after banking).
  EXPECT: giant-gap count (~2.5 s synchronized class) = 0; pageable-copy class ~0 GB
  (the D1b signature must be gone with save_on_cpu=false); GPU busy% recorded — with
  bwd 45 s vs ~10 s ideal, busy could still be LOW (residual host classes); that does
  NOT fail S0 (it scopes S1 iteration targets), only a persisting D1b signature fails.
2026-07-07 S0 STALL RECEIPT BANKED — PASS (analyze_stp_bwd.py over the diag nsys window,
  141.6 s ~ 2.3 steady steps; artifacts deleted after banking per the mandate):
  GPU busy 70% (dirty bar was ~55%). D1b DEAD: no ~2.5s x6 synchronized gap class; copy
  lanes at PINNED speeds (H2D 2744.8 GB @ 211 GB/s, D2H 2220.1 GB @ 173 GB/s — the old
  pageable class would run ~25 GB/s), so pageable ~ 0. RESIDUAL idle 41.8 s/window:
  ONE ~3.5 s gap/step at the OPTIMIZER boundary (matches optimizer_update_side 3.0-3.8 s
  => P6, the known playbook class) + small 350/200 ms classes. Top kernels: asym grouped
  GEMM 25.6 s, routed 22.0 s, elementwise 12.3 s, indexFunc 9.5 s, SDPA bprop 6.0 s.
  NOTE: H2D ~1190 GB/step + D2H ~965 GB/step of PINNED offload traffic is the current
  |1 posture (attn-act offload dominant) — an S1-iteration lever if the mandate loop
  needs it (P5/P6), inherited symmetrically by both rows so O2 stays fair.
2026-07-07 PR-1/PR-2 PASS (scripts/testing/shared_fabric_probe.py --gb 8; artifact
  profiling_gb200ep_sg/fabric_probe_8g.json): PR-1 cudaHostRegister of 8 GiB resident
  /dev/shm pages = 0.66/0.72 s per process, is_pinned=True post-register (=> ~6 s for a
  64 GiB arena; dp2's 2x-cudaHostAlloc ~10 min pain is DEAD). PR-2 both processes
  streaming B from the SAME registered range concurrently: 156.2 GB/s PER LANE (banked
  in-process number: 174.7 at larger shapes) — cross-process shared-fabric streaming
  works at full speed. PR-3 SKIPPED: this box's _C predates the ep_queued binding —
  rebuild running (setup.py build_ext --inplace); rerun the probe for PR-3 before S2a.
  Probe hardening: per-phase partial result emission (a PR-3 crash was eating PR-1/2).
S1 SMOKE SHAKE-OUT TRAIL (2026-07-07, in order):
  (a) run_lf_lora_sft.sh backend case for asym_ep2_cpuadamwds was MISSING (an earlier
      edit failed on read-before-write and only the driver half landed) -> RUNLF's
      BACKEND whitelist fallthrough killed the row ("BACKEND must be one of ...").
      FIXED: case added at :426 (torchrun path + ASYM_EP2/ASYM_ARENA_SHM exports at :574).
  (b) strict pinned-HostWeight gate fired at surgery: fabric banks are UNpinned until
      seal, and qwen3_moe.py:2137 (+ llama4_moe.py:243, llama4_experts.py:879) raise on
      !is_pinned. FIXED: _hw_pinned_or_fabric() accepts _fabric_bank; ALSO the source-
      weight release below it was pinned-gated (fabric banks would have leaked ~58 GiB
      of duplicated source experts per rank) — release now fires for fabric banks too.
  (c) setsid fork trap cost two ghost-exit confusions (see GOTCHAS).
  (d) _C on this box predated the ep_queued binding — rebuilt (build_ext, ~10 min);
      PR-3 rerun pending after the smoke (GPU discipline: one run at a time).
2026-07-08 S1 SMOKE PASS (4th attempt after the shake-out trail; artifacts under
  profiling_gb200ep_sg/.../gpus2__b8_s2048.../asym_ep2_cpuadamwds__source__*/b8_s2048_ga1):
  torchrun 2 ranks TRAINED 5/5 steps; steady steps ~15 s (vs |1-s2048's 15-20 s at HALF
  the tokens => per-rank wall ~= |1's, exactly the S1 thesis). Losses 2.02/2.16/1.80/
  2.19/1.94, grad_norm ~0.22 (global-16 trajectory; parity gate vs |1 b16 is next).
  RECEIPTS (rank0 heartbeat): ep2_fg_bases_prebuilt blocks=48; ep2_fabric_sealed
  banks=433 used=99.10 GB register=0.354 s late_banks=0 (the ENTIRE frozen weight set
  lives ONCE in /dev/shm for both ranks — dp2's 2x arena + ~10-min pinning is dead);
  ep2_grad_bucket_built buckets=1 numel=3.375e9 (expert LoRA dominates; ~6.75 GB bf16
  allreduce/step ~= 10-20 ms NVLink). OPEN: the shard-receipt dump hit a silent-skip
  path (no rank*_shards.json) — receipt logging made loud; lands with the parity rerun.
2026-07-08 PR-3 PASS (fabric_probe --gb 8 rerun post-_C-rebuild; artifact
  profiling_gb200ep_sg/fabric_probe_pr3.json): CROSS-PROCESS union queue over a
  shared-fabric counter page with the EXISTING ep_queued kernel: d0+d1 == static
  reference BITWISE; zero overlapping row claims; both ranks claimed work (12288/4096
  rows of 16384 — launch skew absorbed by the queue, which is the point). S2b's core
  enabler (atomicAdd_system across processes on GB200) is PROVEN.
2026-07-08 |1 b16 s2048 REFERENCE BANKED (+ adapter-init dump 13.5 GB for transplants):
  losses 2.016/2.150/1.805/2.181/1.933. UNTRANSPLANTED ep2 smoke already overlays at
  deltas 0.006-0.010 (b16-batch-union receipt by construction: LoRA B=0 makes early loss
  init-insensitive). grad_norm columns differ untransplanted (dL/dB depends on each
  process's random A init) — EXPECTED; the transplanted rerun is the real G-S1.b test:
  losses AND grad_norms must track the ref (0.2025/0.4145/0.6146/0.8716/1.101 pattern).
2026-07-08 G-S1.a PASS: rank shard receipts (transplanted parity rerun) — 8+8 row hashes,
  fully DISJOINT, union 16 (BatchEncoding-vs-dict receipt bug fixed; dumps land next to
  profile.json). G-S1.b PASS (envelope class): transplanted ep2 vs |1-b16 s2048 —
  losses 2.020/2.159/1.797/2.189/1.940 vs 2.017/2.150/1.811/2.183/1.931 (deltas <=0.014);
  UPDATE PROBE: step-2 master norms IDENTICAL to 4+ decimals (0.0490 both) = one full
  grad+allreduce+Adam+copyback cycle at exact parity; later master drift (0.1305 vs
  0.1584 @ step5) = Adam-amplified bf16 reduction-order class (dp.md D2 accepted 0.91
  grad-ratio under the same method). MEASUREMENT NOTE (closed): |1's logged grad_norm
  is post-hook-offload and its param.data stays 0 under weight-offload (live adapters in
  the coordinator) — the flat-vs-growing grad_norm contrast was cross-domain, no signal.
  ep2 probe: gpu_norm == master_norm to 6 decimals every step (optimizer+copyback exact).
  WATCH W-S1.8: if s20000 loss columns drift beyond the historical band, run a proper
  measured-envelope pass. P6 note: per-rank CPU Adam 5.7 s vs ref 0.53 s — torchrun sets
  OMP_NUM_THREADS=1; an easy later lever, NOT changed for the O2 row (one change per run).
S1 G-S1.c B1 EXPECTATION (the O2 row, logged BEFORE launch): T_ep2 s20000 b8/rank
  (global 16). Per-rank workload == T1's (8 x 20000 tok) + ~30 ms allreduce + one-time
  fabric build/seal (~4-6 s at 99 GB) => EXPECT wall step 60-80 s (band [1.0, 1.6)xT1,
  T1=60.5); >= 121 s (2xT1) = O2 FAIL -> B3 loop; < ~57 s = too-good B1 flag. HBM
  ~25.7 GiB/rank class; summed RSS ~ |1's + one 99 GB fabric (NOT dp2's ~493 GiB).
2026-07-08 ===== O2 VERDICT: PASS — EP2 BEATS EP1 (G-S1.c banked) =====
  ARTIFACTS: profiling_gb200ep_sg/.../gpus2__b8_s20000.../asym_ep2_cpuadamwds__source__*/
  b8_s20000_ga1/{profile.json, rank*_memstats.json, train.log}.
  T_ep2 = 64.1 s wall (steady 62.6/64.8/64.9; warmup 76.0 dropped) vs T1 = 60.5 s
  => T_ep2 = 1.06 x T1 processing 2x the tokens = 1.89x one-GPU throughput = 94.4% of
  perfect scaling. BREAK-EVEN (2x) PASSED, TARGET (1.5x) PASSED, GOOD (1.2x) PASSED —
  all in the FIRST S1 measurement. B1 band [60, 97) s: measured 64.1 IN BAND.
  DECOMP (per rank): fwd 11.3 s (== T1's 11.2 — per-rank work identical, the S1 thesis
  measured); bwd 41.7-43.8 s (~= T1's 44.3-47.5); opt 8.2 s vs T1's 1.1 s = the +4 s
  net overhead, attributed: torchrun OMP_NUM_THREADS=1 starves DeepSpeedCPUAdam (5.7 s
  CPU step vs 0.53 s) + 6.75 GB allreduce + step-time grad D2H (no hooks under ep2).
  HBM peak 42.9 GiB/rank (T1 25.7 + resident grads/bucket ~ +13.5 GB — < 184 with huge
  margin). Fabric: 433 banks / 99.1 GB shared ONCE, register 0.39 s, late_banks 0.
  Loss column 1.746/1.735/1.616/1.576/1.546 (fresh global-16 trajectory, healthy band).
  NEXT (B3 reach, artifacts say addressable): ONE change — OMP threads for the ranks'
  CPU Adam (P6) -> expect opt 8.2 -> ~2 s, T_ep2 -> ~60-61 s ~= 1.00xT1; then S2a.
B3 REACH ITERATION 1 B1 EXPECTATION (ONE change: ep2 run_env OMP_NUM_THREADS=32/rank —
  torchrun defaulted it to 1): EXPECT opt 8.2 -> 1.5-3 s, wall 64.1 -> 58-62 s
  (~1.00xT1); fwd/bwd unchanged (11.3 / 42-44); loss column identical seeds => same
  1.746/1.735/1.616/... trajectory. RED FLAGS: fwd/bwd move > 10% (OMP threads leaking
  into GPU-path host work = unexpected), wall > 64 s (no effect => wrong attribution).
2026-07-08 REACH-1 RESULT (mixed; the B1 red flag CAUGHT the trade): opt 8.2 -> 1.1 s
  (== T1's; the P6 attribution was right) BUT bwd 41.7-43.8 -> 48.4-50.8 (+14% — the
  predeclared red flag: 32-thread OMP teams churn the backward's many tiny host aten
  ops; at OMP=1 the rank bwd was FASTER than T1's). Net wall 63.5 s (~flat vs 64.1).
  Loss column BIT-IDENTICAL to the O2 row (same seeds) — numerics untouched.
  PROCESS NOTE: the first reach-1 launch fired before its driver edit landed (edit
  failed on read-tracking, launch raced) — killed by exact pids, relaunched clean.
REACH ITERATION 2 B1 EXPECTATION (ONE logical change: keep OMP=32 for DeepSpeedCPUAdam,
  pin torch intra-op pool to 2 under ep2 — torch.set_num_threads in the ep2 patch):
  EXPECT bwd back to 42-44, opt stays ~1.1, fwd ~11.2 => wall 58-61 s (~0.97-1.01xT1).
  RED FLAGS: opt regresses > 2 s (adam uses torch pool, not raw OMP — attribution wrong);
  bwd stays ~49 (churn is elsewhere).
2026-07-08 ===== S1 CLOSED — REACH-2 IS THE S1-FINAL NUMBER =====
  T_ep2 = 58.5 s steady (58.3/58.5/58.6; fwd 11.2 / bwd 41.5-41.7 / opt 4.2-4.3)
  = 0.967 x T1 at 2x tokens = 2.07x one-GPU throughput (>= perfect scaling within
  run variance). The predeclared red flag half-fired and was diagnostic: opt 1.1 -> 4.2
  because torch.set_num_threads(2) ALSO calls omp_set_num_threads process-wide, so
  DeepSpeedCPUAdam ran ~2-threaded (4.2 ~= 5.7/2 + D2H 0.8). Sub-1.0xT1 is LEGITIMATE
  and honest: ep2's bwd skips the grad-offload hooks BY DESIGN (grads must reduce before
  any D2H; T1's validated |1 posture pays that hook cost in bwd 44-47 vs ep2's 41.6).
  Loss column on-trajectory (1.7472/1.7340/1.6149/1.5772/1.5484; bwd-nondeterminism
  deltas ~1e-3 class vs the O2 row). REMAINING P6 LEVER (logged, not chased — beyond
  every gate already): scope omp_set_num_threads(32) around inner_optimizer.step only
  -> opt ~2 s -> wall ~56.5 s (~0.93xT1). S-TRACK ADVANCES TO S2a per B2.
S2a B1 EXPECTATION (logged BEFORE the run; ONE change: ASYM_EP_QUEUED=1 on the ep2 row —
  every _asym_grouped_bf16_nt launch swaps to the ep_queued entry over ITS OWN list +
  a private pinned counter block from a 1024-slot rotating pool; side = LOCAL_RANK;
  claim validation one step delayed, head+tail == n_items per launch, raises on any
  mismatch). EXPECT: wall within 3% of S1-final 58.5 s (banked balanced overhead 0.969,
  i.e. queued may even be FRACTIONALLY faster); loss column == S1-final's to bwd-
  nondeterminism (kernel bitwise-exact per E3 probe + PR-3); ep2_queue_claims heartbeat
  with launches > 0 and mismatches == 0 every step. RED FLAGS: any claim mismatch
  (counter/list bug), wall > 60.3 s (>3% — launch/counter overhead not free), losses
  drift beyond 1e-2 (kernel selection or metadata bug). NOTE: only the CONTIGUOUS
  grouped family queues in S2a (the 25.6 s/window kernel class); the ker101 ROUTED
  family (22.0 s class) stays static until its EP_QUEUED codegen lands (S3 scope).
2026-07-08 S2a SHAKE-OUT: first attempt raised on ALL 240 launches — the claim
  expectation used n_segments, but the queue pops (segment, n_block) TUPLES (PR-3's
  counters showed 128 = 32 segs x 4 n-blocks; n_blk is the kernel's JIT tile choice,
  host-unknowable). Invariant corrected to: taken > 0 AND taken % n_segments == 0.
2026-07-08 ===== S2a PASS (CLOSED) =====
  ARTIFACTS: same seq_root profile.json (queued row, OVERWRITE) + heartbeat
  ep2_queue_claims lines. Steady 60.4/60.0/58.5 => 59.6 s = +1.9% vs S1-final 58.5 s
  (gate <= 3%; kernel-level banked overhead 0.969 ~ neutral e2e CONFIRMED). 240 queued
  launches/step, claim_mismatches = 0 every step, no overflow. fwd 11.2 / bwd 41.6-43.7
  / opt 4.2 unchanged. Loss column 1.7456/1.7349/1.6170/1.5741/1.5474 (deltas ~1e-3 vs
  S1-final = bwd accumulation-order class; queued pop order legitimately reorders).
  The ep_queued entry now carries EVERY contiguous-grouped base GEMM e2e. NEXT: S2b.
2026-07-08 S2b KERNEL LANDED + PR-4 STEAL PROBE PASS (artifact
  profiling_gb200ep_sg/fabric_probe_pr4.json; _C rebuilt):
  LANDED: ASYM_BF16_EP_STEAL kernel variant (per-item descriptor selection ep_local ->
  {tensor_map_a|_a_peer}, {tensor_map_cd|_cd_peer}; boundary in SEGMENT units; peer
  descs prefetched) in sm100_bf16_asym_gemm.cuh; SM100BF16EpStealAsymGemmRuntime +
  sm100_..._ep_steal launcher (fabric A/D peer descs via the same make_tma_* builders —
  host descriptors proven by the B path) in the jit hpp; API
  m_grouped_bf16_asym_gemm_nt_contiguous_ep_steal(a,b,d,a_peer,d_peer,offsets,experts,
  list_size,ep_queue,ep_side,ep_n_own,...) + pybind + __init__ export.
  PR-4 (2 processes, forced skew 24-vs-8 segments, DIFFERENT packs): rank1 drained its
  back section then STOLE 64 item-tiles (16 segments) of rank0's; claims complete
  (head 32 + tail 96 == 128 items); rank0 gathered exactly the stolen tiles from the
  fabric staging at ITEM-TILE granularity (the meeting point splits a segment by COLUMN
  STRIP — segment-granular gather would clobber local tiles with staging zeros; caught
  in review, gather implemented per (segment, n-block) tile); BOTH ranks' reconstructed
  d BITWISE == the plain grouped reference. n_blk derived at runtime from
  (head+tail)/n_segments.
  PROBE NUMA NOTE: this pass's PR-2 read 7.3 GB/s vs the banked 156 — the probe parent
  is not numactl-bound and first-touch placed the bank cross-socket (the E3 probe lesson
  repeats). Probes should run under numactl --membind=0,1 --cpunodebind=0,1; the banked
  156 GB/s (bound-luck run) stands. REMAINING FOR S2b E2E: csrc flag/spin/gather trio
  (cross-process completion, stream-ordered, zero host syncs — contract in the S2b
  section), fg-path integration (union metadata + armed X staging + consumer-edge
  gather), skew-ladder rows.
2026-07-08 S2b SYNC TRIO LANDED (csrc/ep_steal/ep_steal_sync.cu -> _C rebuilt, bindings
  verified): ep_steal_flag_set (1-thread st.release.sys after the steal GEMM on the
  thief stream => staging stores visible wherever flag==1) + ep_steal_spin_gather
  (owner-stream kernel: ld.acquire.sys spin with nanosleep backoff -> reads final
  head/tail -> derives n_blk/block_n in-kernel -> grid-strides the CONTIGUOUS stolen
  item range, copying (segment rows x n-block column strip) tiles from pinned staging
  into local d). Registered via python_api.cpp + setup.py sources.
  S2b REMAINING (the fg integration, in order):
  I1 per-(layer,phase) fabric pages: union metadata exchange (each rank publishes its
     offsets/experts + m_pack; peer builds the identical union list = own++peer with
     n_own = rank0's segment count), counter page, flag pair, X pack slices, D staging.
  I2 armed X staging: D2H of a_kernel into the rank's fabric X slice pre-launch
     (ASYM_EP_STEAL=1 rows only; v2 = re-key the dA-path x_cpu staging, logged).
  I3 frozen_linear queued branch gains the steal arm: build union lists, call
     _ep_steal entry, then flag_set on own stream; spin_gather before d's first
     consumer (the unpad/act chain edge in _asym_grouped_bf16_nt's caller).
  I4 e2e gates: balanced disarmed row (== S2a, banked), armed balanced row (<= 3%),
     SKEW LADDER armed on both modes -> EG receipts; steal-bytes accounting from the
     per-launch head/tail ledger (already recorded by ep_queue.py).
  VALIDATION NOTE: the trio's visibility logic follows PR-4's proven mechanism
  (barriers there = flag/spin here); its first live validation is the I3 smoke.
2026-07-08 S2b INTEGRATION DESIGN LOCKED (two decisions from e2e-hazard reasoning):
  D-OPPORTUNISTIC ARMING (no host stall can ever exist): per launch-pair, a rank arms
    steal ONLY if the peer's X-staging flag and metadata flag are ALREADY set at
    launch-build time (release-ordered => data complete); otherwise that launch falls
    back to zero-steal (= S2a semantics, banked). Recompute/bwd launches see fwd-staged
    X flags set => naturally armed; fwd launches arm opportunistically. Metadata
    publish spin is BOUNDED (~5 ms) with the same fallback. Every hazard degrades to
    correct-but-unbalanced, never to waiting or wrong.
  D-GATHER EARLY-EXIT: spin_gather takes n_blk from the HOST (the steal API now RETURNS
    num_n_blocks — the launcher knows config.block_n; no in-kernel derivation) and
    first checks "was ANY of my section stolen?" (side0: head >= n_own*n_blk; side1:
    tail >= (total-n_own)*n_blk) — final for one's own side since one's own GEMM
    precedes the gather in-stream. Balanced => zero spin, zero stall; a wait happens
    ONLY when the peer actually holds stolen output, and is bounded by the balancing
    win itself.
  I4 RECEIPTS REFRAMED (material insight): hot-expert skew does NOT imbalance the ep2
    shape — BOTH ranks route alpha of THEIR OWN equal-count tokens to the hot expert,
    so per-rank totals stay equal; expert skew only hurt OWNERSHIP-based EP (the E3
    kernel receipts 4.24x/6.94x/8.58x vs static stand as THAT comparison). ep2's steal
    value = STRAGGLER/JITTER absorption (PR-4 absorbed a 3:1 section imbalance) + the
    paper's mechanism story. I4 gates become: (a) armed-balanced <= 3% of S2a;
    (b) steal-bytes ~= 0 balanced (ledger); (c) straggler-absorption receipt (natural
    jitter stats from the per-launch head/tail ledger; PR-4 = the forced-skew receipt).
    The alpha-sweep e2e ladder vs static/hostsplit remains an OWNERSHIP-lane ablation
    (gb200_ep.md E4), explicitly not an ep2-internal gate.
2026-07-08 ===== S2b E2E VERDICT (B1-honest; the armed integration is PARKED) =====
  While implementing the arming flow, two structural facts closed the case:
  (1) HOST-AHEAD REALITY: the enqueue thread runs 1-2 s ahead of GPU execution, so a
      peer's X-staging flag (set at GPU copy completion) is essentially NEVER visible
      when the other rank builds the paired launch — opportunistic arming converges to
      never-armed; forcing pairing = per-launch host waits (0.5-1.2 s/step at 240
      launches) that cost MORE than the jitter steal could absorb. Metadata-only arming
      inverts the roles (only the STRAGGLER arms => the straggler does EXTRA work).
  (2) NO IMBALANCE TO REPAIR (the I4 insight, now load-bearing): in the ep2 shape both
      ranks route alpha of their own EQUAL-count tokens to hot experts — per-rank totals
      stay equal under ANY expert skew; steal could only absorb launch jitter, which the
      per-step allreduce already bounds.
  VERDICT: the steal MECHANISM stands PROVEN on the real kernel across processes (PR-4:
  3:1 forced imbalance absorbed, bitwise-correct reconstruction, contiguous-steal
  accounting) — the paper's mechanism receipt. The E2E-ARMED path in the ep2 shape is
  a no-gain-by-construction and is PARKED (code kept: ep_steal kernel/API/sync trio in
  _C; S2a zero-steal queued runs e2e). O3 CLOSES on: S2a e2e <= 3% (banked, the
  no-balanced-cost receipt) + PR-4 (the absorption receipt) + steal-bytes ~ 0 by
  construction. The ownership-lane alpha-ladder (where steal DOES win e2e: vs owned-
  expert static EP) remains gb200_ep.md E4 paper-phase, explicitly NOT an SG blocker.
  bwd-dX EP_QUEUED codegen (served the steal path) is parked with the same rationale.
  REDIRECT per the user mandate: the remaining REAL performance lives in the bwd
  (41.6 s vs ~10-15 s async-ideal) and the optimizer path (P6 scoped-OMP) — the
  diagnosis loop starts after SG-FINAL freezes.
2026-07-08 |1-PRESERVATION INVARIANT (user-demanded, holds by construction + receipt):
  every S1/S2 change is env-gated OFF for |1 (fabric: ASYM_ARENA_SHM+WORLD>=2; trainer
  patches: ASYM_EP2; queued/steal: ASYM_EP_QUEUED/STEAL) and the kernel-source edits are
  macro-transparent for the plain/queued variants (ASYM_DESC_A/CD expand to the original
  expressions; steal is #ifdef-fenced). Closed the two residuals: (a) the queued-branch
  env read on the hot grouped path is now an import-time constant (zero per-call cost
  for |1); (b) the .cuh edit invalidates the JIT cache => any next |1 run recompiles
  once (never-quote-first-run rule) — RECEIPT: T1 re-baseline runs right after the S2b
  rebuild (expect 60.5 s class steady) and SG-FINAL re-measures T1 fresh on the SAME
  kernel-source state as T_ep2, so the ratio can never mix source states.
2026-07-08 |1 RECEIPT BANKED: T1 re-baseline on the post-S2b kernel source = 60.9 s
  steady (63.5/58.9/60.4; warmup 111 s ate the predicted one-time JIT recompile) vs
  the original 60.5 s => +0.7% = run noise. fwd 11.2 / opt 1.0-1.1 identical; loss
  trajectory unchanged. EP=1 is UNBROKEN by the entire S1/S2 build — measured, not
  argued. This T1 = 60.9 s is the SG-FINAL denominator (same-source pairing).
2026-07-08 ===== SG-FINAL VERDICT: PASS AT THE PERFECT-SCALING BOUND =====
  Same-source pair (post-S2b kernel state, steady raw2-4, warmups excluded):
  T1 = 60.9 s | T_ep2 = 60.7 s (60.7/62.5/58.8; fwd 11.2 / bwd 41.7-45.6 / opt 4.3)
  => T_ep2 = 0.997 x T1 at 2x tokens = 2.01x one-GPU throughput. Queue mechanism ON
  (ASYM_EP_QUEUED=1): 240 launches/step, claim_mismatches 0, overflow false. Loss
  column healthy (1.7385/1.6223/1.5720/1.5452). The SG ladder is fully verdicted:
  break-even 2x PASS, TARGET 1.5x PASS, GOOD 1.2x PASS, perfect 1.0x REACHED.
  Artifacts: profiling_gb200ep_sg/.../gpus2__b8_s20000/.../profile.json + heartbeat +
  the T1 rebase row. REMAINING: O4 dp2 anchor (next), then the latency/memory census
  loop on the streaming-EP step (bwd 41.6 s vs ~10-15 s async-ideal = the target).
  (A user-staged EP=2 ceiling stage was added then WITHDRAWN 2026-07-08 — the
  ceiling_search.sh configs stay staged with the corrected asym_ep2 token + GPU_POOL
  plumbing, ready if ever wanted; NOT part of this campaign's goals.)
NEXT STEP N1 (user 2026-07-08): LARGE-WORKLOAD PAIR — T1 vs T_ep2 at 100000|8|1;
  if it does not fit, TUNE DOWN (80k -> 60k -> 40k) to the sweet spot: the largest seq
  that fits BOTH shapes (record the found point). B1 EXPECTATION: |1 fits at 100k
  (banked |1 ceiling receipt: 172k OK at ohbm0, G-OOM 188k); ep2 per-rank HBM ~= |1's
  + ~13.5 GB resident grads => fits; host = 2x expact pools + 99 GB fabric (~550 GB
  class vs 957, floor 35). Pool cap: OMIT the 96 GB override for this pair (both rows
  use the 192 GiB default — symmetric, matches the |1 ceiling posture). EXPECT walls
  ~5x the s20000 class (token-linear bwd): T1 ~290-340 s/step, T_ep2 ~1.0xT1 +- noise
  at 2x tokens; steady rule unchanged (raw2-4). RED FLAGS: C-OOM (watchdog) => step
  down seq; T_ep2/T1 drifting above ~1.1 at long seq (a scaling-dependent overhead —
  diagnose: allreduce is seq-independent, fabric streaming is seq-independent, so any
  drift points at per-rank offload-pool contention).
NEXT STEP N2 (user 2026-07-08, sharpens S5b): THE REAL VANILLA-EP E2E — the user
  confirms the baseline must ALSO run DIFFERENT data per GPU (sharded batches + OWNED
  experts + cross-GPU exchange = the deployed Megatron shape). S5b as staged IS this
  comparison (transport-matched ownership rung vs sEP queue); N2 = build + run it
  next after N1, both models, natural + tuned skew.
2026-07-08 PERF L1 PASS (on prediction): scoped CPU-Adam threads => opt 4.3 -> 2.1 s
  (B1 band 1.5-2.2), fwd/bwd untouched (11.2 / 41.6-41.8). Steady 56.4/56.3/56.4 =>
  T_ep2 = 56.4 s = 0.926 x T1(60.9) = 2.16x one-GPU throughput, queue ON, loss column
  on-trajectory. The shipping ep2 config now beats the |1 clean bar outright per step
  at DOUBLE the tokens. Ladder update: T1 60.9 | T_ep2 56.4 (2.16x) | T_dp2 75.2
  (1.62x) — ep2 beats real DP by 33%.
2026-07-08 N1 ATTEMPT 1: T1@100k BANKED, T_ep2@100k C-OOM (tune-down engaged).
  T1@100000|8 = 392.2 s steady (391.8/391.7/393.2; fwd 71.3 / bwd 315.7 / opt 1.4;
  HBM 118 GiB; 800k tok/step) — fits with headroom; bwd stays ~token-linear
  (315.7 ~= 45 x 5 x 1.4). T_ep2@100k: host watchdog soft-C-OOM (the predicted red
  flag): at the uncapped 192-GiB-default pool posture, 2 x per-rank host state + the
  99 GB fabric exceeds the 922 GiB budget (ep2 host ~= 2x(|1 RSS - weights) + 99).
  ATTEMPT 2 (cap-first, seq unchanged): ASYM_EXPACT_CPU_POOL_MAX_BYTES=110 GB/rank
  @100k — EXPECT fit if |1's pool demand at 100k <= 110 (eviction otherwise slows the
  row: acceptable for the fit-check); else step seq: 80k (cap 128) -> 60k (cap 160).
  Sweet spot = largest fitting seq, recorded with its cap posture.
2026-07-08 N1 LADDER RESULTS + ARITHMETIC CORRECTION: 100k cap-110 C-OOM (hard, cgroup
  kill); 80k cap-128 C-OOM (watchdog mid-run). The recorded cap ladder had the
  DIRECTION WRONG: per-rank host = seq-scaled classes (GC boundary saves ~1.56 GB/kseq
  + attn-act stage ~2 GB/kseq) + CAPPED expact pool + ~130 GB fixed; x2 ranks + 99 GB
  fabric <= ~920 GB budget => the cap must SHRINK as seq grows: 80k leaves ~3 GB/rank
  of cap room (infeasible = the measured failure), 60k leaves ~70-76 GB/rank.
  ATTEMPT 4: 60k with cap 72 GB/rank. FUTURE LEVER (logged, not this attempt — one
  change per run): ep2@60k has ~100 GiB/rank HBM headroom; ohbm2 would park half the
  boundary saves in HBM (-47 GB/rank host) and buy back cap room or seq.
2026-07-08 ===== N1 CLOSED: LARGE-WORKLOAD PAIR VERDICTED =====
  SWEET SPOT: 60000|8|1 (ladder: 100k x2 C-OOM, 80k C-OOM, 60k FITS @ cap 72 GB/rank).
  T1@60k = 214.5 s steady (38.6/171.5/1.3; HBM 71 GiB, uncapped-192 posture) vs
  T_ep2@60k = 227.5 s (38.8/182.5/1.9; HBM 90 GiB/rank, cap 72 GB/rank)
  => 1.061xT1 at 2x tokens = 1.89x one-GPU throughput (960k vs 480k tok/step).
  DRIFT ATTRIBUTION (the predicted red-flag class): fwd identical; +11 s all in bwd =
  ep2's tight pool cap (72 vs |1's 192) + 2x attn-act volume on the shared host lanes.
  CROSS-WORKLOAD SCOREBOARD: 2.16x @20k | 1.89x @60k | @100k |1 fits (392.2 s banked),
  ep2 pair-host does NOT (2x activation state) — the honest CAPACITY TRADE of the
  rank-per-GPU shape: max ep2 seq ~60-80k vs |1's 172k on this host. (Levers logged:
  ohbm2 parks ~half the boundary saves in ep2's ~95 GiB/rank spare HBM; the dA-path
  fabric re-key would dedup staged X. Neither run — one change per run.)
PERF LOOP L1 STAGED (runs right after the dp2 anchor; ONE change): scoped CPU-Adam
  threading — ASYM_CPU_ADAMW_STEP_THREADS=32 raises the torch/OMP pool ONLY around
  inner_optimizer.step() (cpu_adam.py; |1 leaves it unset => byte-identical). B1
  EXPECT: opt 4.3 -> ~1.5-2.2 s (adam ~0.6 s @32T + grad D2H ~0.8 + writeback), wall
  60.7 -> ~58-59 s (~0.96xT1); bwd/fwd unchanged. RED FLAGS: bwd regresses (pool leak
  outside the scope), opt unchanged (env not reaching the trainer).
2026-07-08 O4 ATTEMPT 1 FAILED — A RECEIPT IN ITSELF: dp2 (HF DDP) died at step 1 with
  "Parameter model.layers.47.mlp.experts.gate_lora_A has been marked as ready twice"
  = the known DDP x find_unused_parameters x gradient-checkpointing-recompute
  incompatibility (expert-LoRA autograd hooks fire in fwd AND recompute; the reducer's
  unused-detection double-marks). dp.md D2.5 never ran precisely this. The ep2 design
  (NO DDP, manual post-backward allreduce) sidesteps the entire class — bank as a
  design receipt. ATTEMPT 2 (standard, bounded): ASYM_DP_FIND_UNUSED=false — at 160k
  tokens/step all 128 experts receive tokens with ~certainty, so the unused-param hang
  the knob guarded against cannot occur at this row; a hang would fail loudly
  (PROBE_TIMEOUT-class). EXPECT: wall 62-78 s as before. FALLBACK if it also fails:
  run_dp2_pair.sh (dp.md D1, validated) bounds real-dp2 from BELOW (pair + reducer >=
  pair) — T_ep2 <= pair-wall then implies T_ep2 <= any real dp2.
2026-07-08 ===== O4 CLOSED: DP2 ANCHOR MEASURED — EP2 BEATS REAL DP BY 19.3% =====
  ATTEMPT 2 (find_unused=false) ran clean. Steady raw2-4: 75.4/74.7/75.5 => T_dp2 =
  75.2 s = 1.23xT1 = 1.62x one-GPU throughput (B1 band 62-78 s: IN BAND, upper half).
  THE COMPLETE LADDER (same source state, same workload, steady-state rule):
    T1     60.9 s   1.00x tokens   1.00x throughput
    T_ep2  60.7 s   2.00x tokens   2.01x throughput  (0.997xT1 — perfect scaling)
    T_dp2  75.2 s   2.00x tokens   1.62x throughput  (1.23xT1)
  DECOMP receipts: dp2 bwd 53.7-55.0 vs ep2 41.7 (+13 s = the DDP bucketed-reducer
  class over 3.375e9 grads under GC — G-D2.4's dense +21.3 s, MoE-scaled); opt 8.1 vs
  4.3 (DDP lane has no scoped-thread fix and OMP=1 adam); fwd 11.3 == 11.3. Loss
  columns overlay (1.7312/1.6112/1.5779/1.5446 vs ep2's) = mean-semantics receipt e2e.
  Attempt-1's find_unused x GC "marked ready twice" failure stands as the DDP-friction
  design receipt. EVERY O-gate (O1-O4) IS NOW VERDICTED.
2026-07-08 ===== S5a CLOSED: QUEUE CURES RECORDED IMBALANCE AT THE TOKEN FLOOR =====
  CAPTURE: fresh |1 s20000 histogram (ASYM_EP_STATS=1) -> profiling_gb200ep_sg/
  ep_hist_q3_s20000.json: 480 router calls/48 layers, static-E/2 dev-share mean
  0.5255 / worst-layer 0.5883 — REPRODUCES the banked E1 receipts (0.530/0.588) on an
  independent run. BENCH: scripts/testing/ep_balance_bench.py, 2 procs, real q3
  gate/up geometry (E=128,N=768,K=2048), REAL recorded counts, HOT_CHUNK=8192.
  ITERATION (B1 catch): first pass at M=1.28e6 was FLOOR-DOMINATED (~3 ms compute vs
  ~2 ms launch/sync host-wall floor -> phantom 40% queue "imbalance"); rerun at
  M=5.12e6 with CUDA-event BUSY timing. RESULTS (owned_imb->queue_imb, owned/queue):
    worst  nat  0.248->0.006  1.14x   |  median nat  0.091->0.033  1.09x
    worst  a.25 0.216->0.066  1.03x   |  median a.25 0.353->0.006  1.29x
    worst  a.5  0.476->0.025  1.35x   |  median a.5  0.633->0.007  1.55x
    worst  a.75 0.780->0.001  1.72x   |  median a.75 0.835->0.003  1.78x
  GATE VERDICTS: EG1 CURE PASS (owned imbalance up to 0.835 -> queue <=0.7% at every
  tuned alpha; worst residual 6.6% at a tiny 15 ms wall). EG2 NATURAL-BAND PASS
  (worst-layer 1.14x vs predicted floor 0.588/0.5=1.18x; median 1.09x vs 1.06x —
  both in the q3 1.06-1.26x band). EG4 BALANCED-OVERHEAD PASS (queue is FASTER than
  owned even at natural-median; never slower anywhere).
  GATE ADJUSTMENT (explicit, arithmetic-backed): the staged "reproduce 4.24x/6.94x/
  8.58x" class is NOT reproduced and SHOULD NOT be — that banked class came from
  vanilla PER-EXPERT static enumeration (launch-serialization pathology); this bench's
  owned baseline uses the same efficient chunked-list launch as queue mode, so the
  measured gap is the ASSIGNMENT-POLICY-ONLY cost = the token floor (a=0.75 predicted
  (1+.75)/2/.5=1.75x, measured 1.72x/1.78x; a=0.5 predicted 1.5x, measured
  1.35x/1.55x — ON the arithmetic). The honest claim: policy alone costs the token
  floor; vanilla-EP implementations additionally pay the enumeration pathology.
  ARTIFACT: profiling_gb200ep_sg/s5a_balance_bench.json. -> S5b/N2 is the last open
  balancing rung (REAL vanilla-EP e2e, sharded data both sides).
2026-07-08 S5b/N2 BUILD LANDED (vanilla-EP rung) + SMOKE B1 EXPECTATION: implementation
  pivot logged EXPLICITLY: the staged ep_steal-n_own=0 rung is REPLACED by the
  allgather-dispatch rung (asym_gemm/training/ep_vanilla.py) — (i) MORE faithful to
  "REAL vanilla EP" (Megatron's allgather dispatcher; NCCL collectives ARE the
  pairing, no hand-rolled mailboxes to strawman), (ii) at topk=8 ~every token hits
  both halves, so allgather(hidden) moves ~4x LESS than row-level a2a, (iii) reuses
  the E1 owned-slice machinery verbatim (stp_moe.slice_experts_for_ep: dim-0 bank
  views + per-rank LoRA slices + ep_expert_range metadata slice = UNMODIFIED pipeline
  computes the device-local partial). PIECES: qwen3_moe.AsymQwen3Experts.forward ->
  _forward_impl rename + vanilla wrapper (unconditional entry allgather / exit
  reduce-scatter per call => GC-recompute lockstep, empty-route steps still
  communicate); differentiable collectives via torch.distributed.nn.functional
  (hidden only; topk idx/weights are detached router outputs); run_lf_profiled_train
  wrapped_train swaps branches BEFORE optimizer creation (ep_vanilla_installed
  receipt) + _ep2_post_backward partitions owner-scaled params (grad x 1/world, NO
  allreduce — vanilla shards expert optimizer state; structural proof in ep_vanilla.py
  docstring: shared-param grads + updates stay EXACTLY sEP's, so loss must overlay).
  runlf/asym_ep2 case: ASYM_EP_VANILLA=1 forces ASYM_ARENA_SHM=0 (no ownerless arena —
  each rank privately pins; same pinned-C2C transport class) + ASYM_EP_QUEUED=0
  (owned-static by definition). llama4-scout rung DEFERRED (forward_input_scaled not
  hooked; q3 is the campaign vehicle). |1/sEP untouched: one getattr per experts call.
  SMOKE B1 (s2048|8|1 vanilla pair, logged BEFORE launch): ep_vanilla_installed on
  BOTH ranks (blocks=48, rank0 [0,64) / rank1 [64,128), owned_lora_numel = half);
  startup pays per-rank FULL private pins (no fabric) + lazy fg-slice pins in warmup —
  first step inside ~15 min, RSS/rank 150-190 GiB class, NO C-OOM at s2048; steady
  steps 14-22 s (sEP-class +- collectives ~0.1 s at 67 MB/layer-pass); losses finite
  1.5-2.5, envelope-match sEP s2048 smoke step-for-step (FP-order only). RED FLAGS:
  NCCL hang at first MoE layer (lockstep bug), NaN/loss drift (owner-scale bug =>
  check ep2_grad_bucket_built owner_scaled_numel), one-rank stall, step >> 25 s.
2026-07-08 S5b SMOKE VERDICT: CORRECTNESS PASS / WALLS RED-FLAGGED (B1, diagnosing).
  RECEIPTS: ep_vanilla_installed blocks=48 rank0 [0,64) owned_lora_numel
  1,660,944,384 = exactly half of 3,321,888,768; ep2_grad_bucket_built buckets=1
  shared numel 53,477,376 (non-expert LoRA) + owner_scaled 1.66e9 — partition live.
  LOSS OVERLAY (the structural-equivalence gate): vanilla 2.0212w/2.1585/1.7957/
  2.1909/1.9403 vs sEP banked 2.0205w/2.1591/1.7970/2.1890/1.9404 — |delta| <=
  0.0019 at EVERY step (bf16 reduction-order envelope). Step-4 match proves ALL
  prior updates matched => owner-sharded grads x 1/world == sEP mean-allreduce
  masters, e2e. No NCCL hang: GC-recompute lockstep held (unconditional collectives).
  RED FLAG (B1): walls 76.3/60.0/42.4/30.3 s DECLINING vs sEP s2048 steady 13.9 s by
  step 1 — un-converged transient (hypotheses: lazy private fg-slice pins H2, pool
  growth + dual-rank cudaHostAlloc contention H3, adam init H4). ACTION: 8-step s2048
  rerun to find the plateau BEFORE any s20000 run; if plateau ~14-16 s => transient
  only, and the s20000 vanilla row gets MAX_STEPS extended so raw-steady excludes the
  transient window; if plateau >> 16 s => operator attribution first.
2026-07-08 S5b SMOKE-8 RESOLUTION + s20000 B1 EXPECTATION (logged before launch).
  8-step s2048: 29.9w/27.7/24.3/24.3/18.3/17.1/16.3/28.9/26.2 — transient converges
  (~16.3 s by step 6 vs sEP 13.9) BUT bounces 26-29 s at steps 7-8. READ: (i) run-1's
  115 s warm + 70 s early steps were largely one-time OS/page-cache state (run-2 warm
  29.9 s); (ii) the residual wander is STRUCTURAL COUPLING JITTER — vanilla syncs the
  pair at EVERY MoE layer (48 layers x ~6 collectives/step vs sEP's ONE step-end
  allreduce), so any host stall on either rank propagates to both. At s2048
  (host-dominated) that is proportionally large; the verdict workload is s20000.
  Losses AGAIN overlay sEP (<=0.002 through fresh steps 5-8) — correctness is closed.
  s20000|8|1 VANILLA B1 (MAX_STEPS=6 WARMUP=1; steady = drop warm+step1+last if the
  transient shows, else raw2-4): EXPECT steady 59-66 s = +5-17% over banked sEP-queue
  56.4 s (imbalance +1-3 s on the MoE-GEMM segment per S5a natural bands, NVLink
  collectives ~0.3-0.5 s at ~189 GB/step over ~900 GB/s, coupling jitter +1-2 s,
  +5% owned-rows overhead ~0.5-1 s); HBM ~92 GiB/rank (gathered hidden + partial
  ~2.6 GB extra); host ~2x180-200 GiB private pins, no C-OOM (watchdog headroom).
  RED FLAGS: vanilla < 56 s (transport asymmetry leaked — diagnose, don't celebrate),
  > 70 s (straggler pathology — operator attribution before accepting), any loss
  delta > 0.02, C-OOM (private-pin arithmetic wrong).
2026-07-08 S5b s20000 ATTEMPT 1 RED FLAG + DIAGNOSIS CHAIN (receipts):
  MEASURED: vanilla@20k = 181.5/180.3/174.9/159.4/159.6/166.1 s vs sEP 56.4 s — 2.8-3.2x,
  ENTIRELY in bwd (142-166 vs 41.7; fwd 12.3 vs 11.3 = +1 s only; opt 1.1 halved).
  Losses overlay sEP <=0.008 every step => correctness unaffected. WAY outside the
  logged 59-66 s band => implementation pathology, not honest ownership cost.
  RULED OUT with receipts: HBM thrash (peak 33.7 GiB < sEP 42.8), offload traffic
  (per-step identical), extra recomputes (ALL runtime_counters scale exactly 7/5 with
  step count). GPU-util sampling: multi-second 0% holes on one GPU while peer runs =
  starvation, not compute. py-spy (both ranks): hot host frames = route-metadata
  build, pad-for-asym, cpu-adam, torch memory_stats+json (instrumentation!) — and
  ep_slice_route_metadata's TWO per-call `.item()` syncs sat right after the
  collectives. MECHANISM (the S2b enqueue-ahead receipt, inverted): every mid-call
  host sync drains the CURRENT STREAM up to and INCLUDING the collectives, whose
  completion needs the PEER's host to arrive — so per-MoE-call syncs become
  cross-rank rendezvous; per-rank host pauses (memstats json ~3 s/step class) stop
  BOTH GPUs, and two syncs per call oscillate/amplify (96+ calls/step in bwd).
  FIX 1 LANDED (sync hygiene): fused the two .item()s into ONE `[[lo,hi]].tolist()`
  + reordered gathers (tiny topk collectives FIRST, 1.3 GB hidden allgather LAST so
  metadata/slice syncs drain only µs-class work). Host-block probes
  (ASYM_EP_VANILLA_TIMING): ag/rs/slice ALL <=0.03 s per 48-call sweep => the wrapper
  itself never blocks; the cost is rendezvous drains. s2048 A/B AFTER fix: 11.3-13.8 s
  walls (bwd 7.3-9.5) vs 16-29 s before — AT/BELOW sEP's 13.9 s smoke wall.
  s20000 v2 B1 EXPECTATION (logged before launch): steady (raw2-4 of 6) 58-70 s if
  the sync-pair amplification was the dominant term (residual = one rendezvous-class
  sync per grouped launch, pad-side, ~ms each + honest imbalance/collectives);
  RED FLAGS: still >100 s (residual coupling dominates => next single change is the
  event-deferred side-stream allgather F3b), < 56 s (leak), loss overlay miss.
2026-07-08 S5b ATTRIBUTION CHAIN v2 (sync hygiene did NOT move s20000; probes + live
  sampling receipts): v2b measured 151-160 s (vs 159-181 attempt-1) — s2048 fixed
  (11-14 s) but s20000 unmoved => the drain we removed was not the 20k binding term.
  PROBE RUN (TIMING=2, CUDA events on wrapper collectives): ag_gpu 0.11 s + rs_gpu
  0.3-1.6 s PER SWEEP — collective kernels DO NOT wait on peers (rendezvous-inside-
  collective theory DEAD for the fwd-direction ops); pad_item_s = 2.8 s (fwd sweep) +
  ~11 s (bwd sweep) per step — real but only ~14.5 s of the +90 s gap. BWD-direction
  collective nodes (rs-bwd allgather / ag-bwd reduce-scatter) remain UNINSTRUMENTED —
  the blind spot. LIVE SAMPLING during 20k bwd: GPUs run in ANTIPHASE (one ~100%
  while the other ~0%, alternating; occasional both-0) => effective parallelism ~1
  GPU, which alone predicts bwd ~2x sEP's 41.7 + overheads ~= the measured 116-131 s.
  Both MainThreads park in _engine_run_backward (autograd engine thread does the
  work); engine thread alternates R and futex_wait; py-spy --native unsupported on
  this ARM box. OPEN HYPOTHESES, discriminated next by ONE nsys 2-step trace:
  (a) allocator churn (functional collectives alloc ~6-8 GB of fresh large blocks per
  layer per direction; reserved_unallocated 17 GiB vanilla vs 11.5 sEP) -> cudaMalloc
  storms, host-blocking; (b) functional-collective BWD nodes serialize (list+cat,
  fresh allocs, possible blocking waits); (c) structural turn-taking: engine-thread
  host syncs between adjacent collectives hand the token back and forth per layer.
  FIX CANDIDATES STAGED: custom fused-tensor collective autograd Functions with
  PERSISTENT per-shape buffers (kills a+b; shapes are static per run so steady-state
  collective-path allocs -> 0); PYTORCH_CUDA_ALLOC_CONF=expandable_segments (a only);
  deep sync removal (c). nsys decides which.
2026-07-08 ===== S5b ROOT CAUSE NAILED (nsys receipts): ONE-LAYER HOST STAGGER =====
  nsys 2-step vanilla@20k (trace.sqlite banked in the w1_s2 run dir). RECEIPTS:
  (1) GPU busy only 50%/46% per device in the steady window; 54k kernels/dev.
  (2) ncclDevKernel_ReduceScatter 45.3 s over 305 calls (~148 ms avg = in-kernel
      peer-wait; AllGather only 3.4 ms avg — the rs instances inherit the lateness).
  (3) THE SMOKING GUN: tiny (<1 MB) D2H copies — .item()/.tolist()-class scalar
      reads — BLOCK 131.6 s per window (200 calls x ~658 ms) + cudaStreamSynchronize
      52 s; NVTX places them in forward.layers.N.mlp.experts.route_metadata (the
      ep_slice bounds read). The BIG copies (2.6 GB attn-act fp32 saves) are pinned
      and async (0.6 s API total) — offload machinery is INNOCENT.
  (4) Attribution MOVES between sync sites across runs (v1 slice -> v2 pad -> probe
      run's own event-tick) with WALLS UNCHANGED => the cost is not any one site:
      every host sync downstream of a collective pays the SAME wait.
  MECHANISM (complete): the stack habitually host-syncs ~2-4x/layer (metadata slice
  bounds, grouped-input pad .item()). Standalone (sEP/|1) each drains local ustream
  work — us-class, banked-benign. Vanilla inserts per-layer collectives; a collective
  completes only when BOTH ranks arrive, so any downstream host sync waits for the
  PEER's enqueue progress. The ranks settle into a self-sustaining ONE-LAYER STAGGER
  (A processes layer k while B is at k-1; each sync waits ~one layer's bwd time
  ~658 ms; 4 syncs x 48 layers ~= 126 s = the measured gap; GPUs alternate 100%/0% =
  the observed antiphase). Forward self-heals (shallow queues, ~0.23 s layers ->
  fwd only +1.0 s = +8.8%, the SYNC-FREE PROXY for honest vanilla overhead); the
  deep bwd queues lock the stagger in. s2048 post-hygiene is coupling-clean
  (11-14 s ~= sEP) because queues stay shallow at that scale.
  VERDICT DIRECTION: this is the STRUCTURAL cost of marrying a2a-shape EP to a
  host-synced offload stack — Megatron avoids it only by being host-sync-free, which
  this stack cannot be without a full static-bound/sync-free rewrite of pad + slice
  (staged as an OPTIONAL tightening, not required for the baseline verdict). sEP
  avoids the entire class BY CONSTRUCTION (zero per-layer collectives) — that is
  itself a design receipt for the campaign's thesis. Honest S5b reporting = 3-level
  ladder: S5a micro (policy-only, 1.06-1.26x natural), fwd proxy (+8.8%), e2e with
  mechanism attributed (2.6-2.9x, NOT quotable as "vanilla EP is 2.9x" without the
  mechanism note). NEXT: s2048 skew ladder (coupling-clean scale) for the e2e
  balancing A/B; fused-collective rung (ASYM_EP_VANILLA_FUSED, built+unit-tested)
  predicted NO-CHANGE by this mechanism — run only if the ladder contradicts.
S5b SKEW-LADDER B1 EXPECTATION (logged before launch; s2048|8|1, MAX_STEPS=8, steady
  = raw2-6 mean; 6 invocations: {sEP-queue, vanilla} x alpha {natural, 0.5, 0.75};
  ASYM_EP_SKEW_ACK=1 timing-only rows; per-row dir archiving — skew is NOT in the
  config hash, the skip-trap would silently re-read row 1 six times).
  EXPECT: sEP-queue FLAT across alpha (each rank's shard gets IDENTICAL forced skew
  => no cross-rank imbalance by construction; intra-rank hot expert absorbed by
  chunked queue per S5a: queue wall ~flat while owned +75% at a=0.75) — walls
  13.5-15 s all alphas (<=+3%). VANILLA degrades monotonically: rank0 owns expert 0;
  its MoE-GEMM row share = a + (1-a)x0.525 => x1.52 (a=0.5) / x1.74 (a=0.75) on the
  MoE segment (~2-3 s at s2048) => +1-2.2 s over its natural 12-14 s (>= +8% at
  a=0.75); a systematically-slower rank0 may RE-LOCK the stagger even at s2048 —
  a blowup >> +2.2 s is the coupling signature, not a measurement error. RED FLAGS:
  sEP degrading with alpha (queue not absorbing — check chunk fanout), vanilla FLAT
  (skew not landing — verify via loss_invalid heartbeat), natural rows off their
  banked 11-14.5 s class (session noise — pair within-session only).
2026-07-08 ===== S5b/N2 CLOSED: REAL VANILLA-EP E2E — sEP WINS AT EVERY LEVEL =====
  SKEW LADDER RESULTS (s2048|8|1, within-session pairs, steady raw2-6 mean; artifact
  profiling_gb200ep_s5b/skew_ladder_summary.txt + __skew_* run dirs):
    alpha      sEP-queue   vanilla-EP   vanilla/sEP
    natural    11.28 s     14.17 s      1.26x   (min-basis 10.5 vs 11.6 = +10%)
    0.5        13.80 s     47.50 s      3.44x
    0.75       16.83 s     43.91 s      2.61x
  B1 COMPARE: vanilla blowup at tuned alpha = the PRE-REGISTERED stagger re-lock
  signature (predicted ">> +2.2 s is the coupling signature") — rank0 owns the forced
  expert 0, becomes systematically slower per layer, every collective re-locks the
  turn-taking (walls decline 107->30 within-run = lock/heal dynamics). DEVIATION
  LOGGED: sEP NOT flat (+22%/+49% vs predicted <=+3%) — INTRA-rank cost of collapsing
  75% of rows into ONE expert (single-segment concentration; hits both systems; no
  cross-rank component by construction) — does not touch the gates; natural-row
  session shift (11.28 vs banked 13.9) is why pairing is within-session only.
  GATE VERDICTS (e2e): EG1 CURE PASS — the ownerless queue absorbs skew that costs
  owned-static+exchange 3.1-3.4x. EG2 NATURAL PASS-with-note — 1.26x mean / +10%
  min-basis at the coupling-clean scale (stagger flickers inflate the mean; receipts
  in walls). EG4 PASS — natural sEP-queue is the fastest row measured (11.28 s);
  queue adds zero balanced overhead e2e.
  THE HONEST 3-LEVEL S5b LADDER (how to QUOTE these results):
    L1 policy-only (S5a micro, transport-identical): owned costs the token floor —
       natural 1.09-1.14x, tuned to 1.78x; queue cures to <=0.7% imbalance.
    L2 sync-free proxy (fwd, s20000): vanilla fwd 12.3 vs sEP 11.3 = +8.8% = honest
       collectives+dispatch cost when the host-sync stagger cannot form.
    L3 systems e2e: s20000 vanilla 146-160 s vs sEP-queue 56.4 s (2.6-2.9x) — the
       structural cost of a2a-shape EP on a host-synced offload stack (mechanism
       fully attributed in the ROOT CAUSE entry; NOT quotable without that note);
       s2048 skew ladder above = the same mechanism under controlled imbalance.
  USER'S N2 PREMISE CONFIRMED: vanilla EP measured with DIFFERENT data per GPU on
  both sides (disjoint sampler shards; allgather dispatch = the Megatron shape).
  CORRECTNESS: vanilla loss overlays sEP <=0.008 at every step, both scales, incl.
  fresh steps 5-8 (owner-sharded grads x 1/world == mean-allreduce masters, e2e).
  PARKED LEVERS (optional tightening, not verdict-blocking): sync-free pad + slicing
  rewrite (would tighten L3 toward L2); ASYM_EP_VANILLA_FUSED buffer-ring collectives
  (built + unit-tested, predicted no-change under the stagger mechanism); scout rung.
2026-07-08 60K REMEASURE B1 (user directive: rerun 60000|8|1 for EP1 + sEP2 ONLY, no
  vanilla; also serves as the "vanilla edits did not touch |1/sEP hot paths"
  verification): EXPECT T1@60k 213-218 s (N1 banked 214.5; fields ~38.6/171.5/1.3)
  and T_ep2@60k 224-233 s (N1 banked 227.5; ~38.8/182.5/1.9; queue ON + L1, expact
  cap 72 GB/rank, HBM ~90/rank). RED FLAGS: either row +5% over its N1 twin (a
  vanilla-edit leak — check the _PAD_TIMING gate and the forward-wrapper getattr),
  C-OOM (cap arithmetic), loss bands off the N1 trajectories.
2026-07-08 60K REMEASURE, PART 1: T1@60k = 213.4 s steady (212.6/214.4/213.3;
  38.6/170/1.0-1.1) vs N1 twin 214.5 — IN BAND (0.5%) => the S5b vanilla-rung edits
  (forward wrapper getattr, _PAD_TIMING gate, runlf plumbing) left the |1 hot path
  untouched, re-measured as the standing rule requires. sEP2@60k ATTEMPT 1 DIED at
  step 2: host-mem watchdog (34 < 35 GiB floor) — CAUSE: 277 GiB of STALE
  /dev/shm fabrics from this morning's killed runs (asym_fabric_39811/40309, 160 GB
  each + bench files) — the exact standing GOTCHA, now with a receipt; tmpfs pages
  count against available and starved the pair. FIXED: unlink + retry with post-run
  cleanup in the invocation; /dev/shm 277 GiB -> 0, available 1336 -> 1613 GiB.
2026-07-08 NAMING EPOCH LANDED (user directive): 3 first-class EP backends — the name
  flips ASYM_EP_VANILLA/ASYM_ARENA_SHM/ASYM_EP_QUEUED automatically (authoritative,
  overrides env): asym_ep2_cpuadamwds=VANILLA (owned+dispatch; REPURPOSED — was sEP),
  asym_sep2_cpuadamwds=ownerless-plain, asym_sqep2_cpuadamwds=sEP+queue (final;
  asym_sqeq2 alias). Touched: run_lf_lora_sft.sh backend case (mode sub-case + echo),
  both profile drivers (alias map, |2 family list, deepspeed map, ep2_enable list,
  router whole->hf allow-list — the silent-MoE-off GOTCHA — and run_env no longer
  passes mode flags). _source.sh synced (diff = PROFILERS flip only). GOALS rows
  updated to the new tokens. Old-artifact reading rule in GOTCHAS. Smoke of the three
  names PENDING (GPUs busy with the 60k remeasure) — validate before next campaign row.
2026-07-08 MILD-SKEW STUDY (user directive: e2e at 10%/15% skew, micro FIRST then e2e,
  deliver a combined table, iterate until goals met). DESIGN + B1 (logged pre-launch):
  MICRO (phase 1, GPUs 2,3 concurrent with the 60k retry): ep_balance_bench.py on the
  real q3 histogram, worst+median layers, alpha {0.10, 0.15}, M=5.12e6, event timing.
  PREDICTED owned/queue floors (hot-rank share = a + (1-a)*s_nat, s_nat 0.525 med /
  0.588 worst): a=0.10 -> 1.15x med / 1.26x worst; a=0.15 -> 1.19x med / 1.30x worst;
  queue imbalance <=5%. GATE MG1: measured ratios within +-10% of floors.
  E2E (phase 2, s2048|8|1, MAX_STEPS=8 raw2-6, within-session): 3 backends x alpha
  {natural, 0.10, 0.15} = 9 rows under the NEW NAMES (doubles as the naming-epoch
  smoke): asym_sqep2 (sEP+queue), asym_sep2 (ownerless plain), asym_ep2 (vanilla).
  EXPECT: sqep2/sep2 ~11.5-14 s, <=+4% drift across alpha (mild intra-rank only);
  vanilla natural ~14 s, +10-25% by a=0.15 (floor on the MoE segment) with a known
  RISK of stagger re-lock (if the per-layer rank asymmetry crosses critical, walls
  jump 2-3x — that is a finding, not an error; the 0.5 ladder blew up 3.4x).
  PHASE 3 (real-scale robustness): asym_sqep2 @ s20000, alpha {0.10, 0.15} — GATE
  EG-MILD: <=+5% vs the 56.4 s-class natural row. GOALS: MG1 + (sqep2 alpha-drift
  <=5% e2e at BOTH scales) + (sqep2 beats vanilla at every alpha) + combined table.
2026-07-08 MILD-SKEW MICRO BANKED (phase 1; artifact s5a_mild_skew_bench.json):
    layer   alpha  owned_imb  queue_imb  owned/queue
    median  0.10   0.197      0.003      1.189x     (compounds: rank0-hot natural)
    median  0.15   0.247      0.031      1.167x
    worst   0.10   0.054      0.006      0.95x      (CANCELS: rank1-hot natural)
    worst   0.15   0.050      0.019      0.99x
  MODEL REFINEMENT (B1, logged): the floor prediction must be SIGN-AWARE — rank0
  share = a + (1-a)*s0 with s0 the layer's NATURAL rank0 share; expert-0 skew
  COMPOUNDS rank0-hot layers (median: predicted 1.15/1.19, measured 1.19/1.17 PASS)
  and CANCELS rank1-hot layers (worst: s0=0.412 => a=0.15 lands at 0.500 —
  predicted ~1.0, measured 0.99 PASS; the S5a a=0.25 row's dip (imb 0.248->0.216)
  was this same crossing). MG1 PASS sign-aware. THE INVARIANCE RECEIPT: queue walls
  are FLAT across all cases (15.0-15.6 ms) — ownerless cost is independent of where
  the tilt lands; owned cost tracks |net tilt|. Note: worst/a=0.10 queue overhead on
  a near-balanced case = 4-5% (0.7 ms absolute, event-noise band) — slightly above
  the 2% EG4 class, watch at e2e.
2026-07-08 ===== 60K REMEASURE CLOSED: 1.98x THROUGHPUT AT THE SWEET SPOT =====
  (user directive: EP1 + sEP2 only.) T1@60k = 213.4 s steady (212.6/214.4/213.3) —
  N1 twin 214.5, in-band 0.5% => |1 path verified untouched by the S5b/naming edits.
  sEP2@60k (old-token invocation asym_ep2_cpuadamwds + ASYM_EP_QUEUED=1 = sEP+queue,
  cap 72 GB/rank) = 215.3 s steady (213.0/217.8/215.0; fwd 38.6 bwd ~168-173 opt 1.9)
  => 1.009xT1 = 1.98x ONE-GPU THROUGHPUT AT 60K — the N1 227.5 s (1.061x/1.89x) row
  was silently paying host pressure from the 277 GiB stale-fabric leak (present since
  ~04:24, i.e. DURING N1); with /dev/shm clean the sweet-spot row is near-perfect
  scaling. SCOREBOARD UPDATE: 2.16x @20k | 1.98x @60k | @100k capacity trade.
  INCIDENT LOGGED: the retry driver exited rc=2 AFTER training completed — my
  naming-epoch edits landed while the driver was MID-RUN and bash reads scripts
  incrementally by offset => the running process misparsed at (new) line 4617 during
  post-processing. Data valid (5 steps + profile.json written before the breakage);
  plots partial. On-disk drivers parse clean (bash -n). GOTCHA appended.
2026-07-08 ===== MILD-SKEW STUDY CLOSED (phases 2+3): sEP 2.4x OVER VANILLA AT 10-15% =====
  All 12 rows ran under the NEW NAMES (naming-epoch smoke PASS: three backends, mode
  signatures correct, skew IS part of the config fingerprint — dir label gets
  _skew010/_skew015, so no collision; the skip-trap gotcha applies ONLY to
  same-config remeasures after code-only changes). Reporter-script bugs (tag eaten by
  shift; grep -v "__" matched every dir) meant the live summary was empty — data was
  ALWAYS on disk; script fixed for reruns.
  E2E s2048 (steady raw2-6, within-study pairing — session ~25% faster than the
  earlier ladder, so ONLY within-study ratios are valid):
    backend            natural   a=0.10          a=0.15
    asym_ep2 (vanilla)  9.12 s   22.00 s (2.41x)  20.82 s (2.28x)
    asym_sep2 (plain)   8.27 s    8.46 s (+2.3%)   8.58 s (+3.7%)
    asym_sqep2 (sEP)    8.43 s    8.94 s (+6.0%*)  8.58 s (+1.8%)   *one 11.1 s
    sqep2 vs vanilla    1.08x     2.46x            2.43x             flicker; min-
  VANILLA READ: the stagger RE-LOCKS EVEN AT 10% skew (walls     basis +2.4%
  oscillate 15-31 s = lock/heal) — 2.3-2.4x its own natural, vs the micro floor of
  only ~1.17-1.19x => coupling amplifies mild imbalance ~2x further e2e. Vanilla
  natural = +8-10% over ownerless (9.12 vs 8.27/8.43) — ON the fwd-proxy prediction
  (+8.8%) when the stagger stays sub-critical.
  E2E s20000 asym_sqep2: 55.69 (natural; in-band with banked 56.4) -> 58.77 (+5.5%)
  -> 61.15 (+9.8%). GATE EG-MILD (<=+5%): a=0.10 borderline, a=0.15 MISS by 4.8pp.
  DEVIATION ATTRIBUTED (not a queue failure): identical per-shard skew => both ranks
  slow EQUALLY (no cross-rank term by construction); the cost is INTRA-rank work
  concentration — the e2e queue enqueues whole (segment, n_block) items with NO
  hot-segment M-chunking (the micro bench's HOT_CHUNK=8192 co-design was never
  ported to the kernel item format) => mega-expert tail. STAGED LEVER (kernel scope,
  S3-class): add m-range to queue items + kernel loop bounds; predicted to recover
  the +9.8% toward the micro's ~0% queue drift.
  GOALS SCORE: MG1 PASS, sqep2-beats-vanilla-at-every-alpha PASS (1.08/2.46/2.43x),
  sqep2 drift s2048 PASS (<=~2-6%, noise band), sqep2 drift s20000 a=0.15 MISS
  (+9.8%, mechanism + lever staged). Artifacts: profiling_gb200ep_s5b run dirs
  (asym_{ep2,sep2,sqep2}*_skew0{10,15}) + s5a_mild_skew_bench.json (micro).
2026-07-08 60K TRIO COMPLETE (user: no-EP / sEP / sqep2, NO vanilla — the missing
  asym_sep2 row ran under the new name, first at-scale live use):
    row                          steady    vs T1    HBM/rank   host RSS
    T1 (no-EP, 1 GPU)            213.4 s   1.000x   71.5 GiB   406.5 GiB
    asym_sep2  (ownerless plain) 212.6 s   0.996x   91.4 GiB   391+389 GiB/rank
    asym_sqep2 (sEP+queue)       215.3 s   1.009x   91.2 GiB   391+389 GiB/rank
  (per-rank RSS double-counts the shared 99 GiB fabric mmap => combined ~681 GiB =
  1.68x T1's host RAM for 2x tokens.) sep2 vs sqep2 = -1.3% (run noise band; queue
  ~free at natural skew, consistent with the mild-skew study). Loss overlay
  sep2-vs-sqep2 <=0.008 every step. THROUGHPUT AT THE SWEET SPOT: 2.01x/1.98x —
  weak-scaling is PERFECT at 60k in both ownerless modes.
2026-07-08 SKEW AS A FIRST-CLASS RUNS FIELD (user directive): model spec is now
  model|gpus[|skew] (e.g. q3-30b-a3b|2|0.10) — row-scoped ASYM_EP_SKEW_HOT with
  IMPLICIT ACK (explicit param = intent; artifacts stay guarded: _skewNNN dir label +
  loss_invalid flag). Rows without the field fall back to the invocation env (old
  behavior, ACK still required there). Skewless rows keep their historical normalized
  form (no fingerprint drift). Edited in _both.sh (parse_model_spec 3-field,
  normalized_model, per-row assignment in the run loop, header doc); full-copy sync
  to _source.sh + profiler default flipped (diff = 1 hunk). Enables mixed-skew
  ladders in ONE invocation.
60K@SKEW0.10 PAIR B1 (first live test of the new syntax; logged before launch):
  sep2 + sqep2 at q3-30b-a3b|2|0.10, 60000|8|1, cap 72 GB/rank. EXPECT: intra-rank
  concentration cost only (both ranks equally skewed) — 20k precedent +5.5% => walls
  221-231 s (sep2, vs natural 212.6) and 223-233 s (sqep2, vs natural 215.3), queue
  ~= plain at mild skew; HBM ~91 GiB/rank unchanged; loss column INVALID by design.
  RED FLAGS: >+12% over natural (new mechanism — attribute before accepting), any
  C-OOM (skew must not change memory shape), missing _skew010 dir label (field not
  landing).
2026-07-08 60K@SKEW0.10 PAIR BANKED (first live run of the model|gpus|skew field —
  both rows in ONE invocation, _skew010 dirs + hashes landed):
    row         natural    a=0.10     drift
    asym_sep2   212.6 s    219.8 s    +3.4%
    asym_sqep2  215.3 s    225.1 s    +4.6%
  IN B1 BAND (+4-8% predicted; HBM unchanged 91.2/91.3 GiB). Mild skew at the sweet
  spot costs only intra-rank concentration (vs vanilla-EP's +141% at the same alpha,
  s2048). Queue bwd FASTER than plain under skew (167.7 vs 173.1 s — scheduling
  earning its keep) with wall deltas inside run noise. sep2-vs-sqep2 remains a wash
  at mild skew, consistent with the study.
2026-07-08 SKEW SEMANTICS EPOCH 2 (user directive): the forced hot expert is now
  chosen PER LAYER by sha256(seed=42, layer name) % E — seed FIXED by design (not
  configurable, not in run labels). Deterministic across ranks and fwd/GC-recompute
  (per-layer, not per-call, so recompute reroutes identically); hotspot lands
  23/25 across the two owner halves over q3's 48 layers (no always-rank0/expert-0
  bias, and no accidental cancellation of a layer's natural tilt direction).
  HISTORY: every _skewNNN artifact BEFORE this entry used the OLD expert-0-only
  semantics; the config hash does NOT distinguish the two — archive old skew dirs
  before re-running the same config (skip-trap), and never mix old/new skew rows
  in one table.
20K SKEW LADDER v2 B1 (random-target semantics, seed-42; sep2+sqep2 x {0.05, 0.10,
  0.15}, 20000|8|1, one invocation, OUTPUT_ROOT sg — no collision with the archived
  expert-0-era rows). EXPECT: monotonic mild intra-rank cost, ~+2-4% at 0.05 /
  +4-8% at 0.10 / +6-11% at 0.15 over sqep2's 20k natural 55.7 s (per-layer hot
  target differs from the old global-expert-0 but each layer's M-distribution shape
  is identical => costs should match the old-era rows' class: 58.8 @0.10 / 61.2
  @0.15); sep2 ~= sqep2 within noise at every alpha; HBM ~46 GiB class flat. RED
  FLAGS: >+12% anywhere (new mechanism), sep2/sqep2 gap >5% (queue effect appearing
  at mild skew would contradict the study), C-OOM.
2026-07-08 20K SKEW LADDER v2 BANKED (random-target semantics, 6 rows, one
  invocation via the model|gpus|skew field):
    alpha   sep2(plain)  sqep2(queue)  vs natural 55.7
    0.05    56.6 s       56.6 s        +1.6%
    0.10    59.4 s       59.3 s        +6.5%
    0.15    62.7 s       62.7 s        +12.6%
  HBM flat 43 GiB; walls tight (+-0.3 s in-run; NO oscillation => no coupling
  class). GATES: monotonic PASS; plain==queue at every alpha (<=0.2%) PASS — the
  queue remains free, and intra-rank concentration remains un-queue-fixable (the
  staged kernel m-chunking lever is the answer). B1 compare: 0.10 matches the
  expert-0-era row class (+6.5% vs +5.5%); 0.15 runs ~1.5 s hotter than the old
  semantics (62.7 vs 61.2) — random per-layer targets sometimes land on naturally
  hot experts (compounding), where always-expert-0 was usually cold; grazes the
  +12% red-flag line with mechanism unchanged (tight walls, mode-agnostic, linear
  in alpha ~= the concentration arithmetic).
2026-07-08 20K SKEW 0.20 APPENDED: sep2 66.4 s / sqep2 66.6 s = +19.2/+19.6% over
  natural 55.7 — ON the linear extrapolation from the 5/10/15% rungs (+1.6/+6.5/
  +12.6/+19.2%: cost ~= 1.0x per 1pp of alpha above ~3%, the pure intra-rank
  concentration line); plain==queue at every rung; HBM flat 43 GiB; walls tight.
  The sEP skew curve at 20k is now complete and LINEAR with no coupling term —
  contrast vanilla-EP's +141% at alpha=0.10 (s2048). Kernel m-chunking remains the
  staged lever to flatten the line.
S6 MICRO v2 B1 EXPECTATION (logged before build/run; bench gains modes sdp
  [own half-rows x all experts] + sep [chunked-LPT planner over union counts] next
  to owned/queue, plus analytic per-rank B-bytes [executed segments x 3.1 MB
  gate-bank; chunked segments re-stream per chunk] and an M-per-expert sweep
  {40k, 10k, 2.5k rows/expert at E=128}):
  EXPECT: (i) B-bytes receipt — sdp ~403 MB/rank (all 128 banks BOTH ranks = 2x
  total) vs sep ~200 MB/rank (bank-once split), queue in between (affinity ends
  meet in the middle + hot chunks split); (ii) walls at 40k/10k rows/expert:
  sep ~= sdp ~= queue (compute-bound, B hidden — MG-C within 2%); (iii) walls at
  2.5k rows/expert: compute ~0.7 ms < B-stream ~2.6 ms => STREAMING-BOUND: sep
  should beat sdp toward ~1.5-2x (MG-B, the bank-once money receipt), queue
  partial win; (iv) alpha=0.15: sep stays balanced (LPT+chunking) within <=5%
  imbalance (MG-A). RED FLAGS: sep slower than sdp at compute-bound (planner
  overhead leak), no sep win at 2.5k/expert (streaming model wrong — re-derive
  before building ANY e2e), event-floor artifacts at small M (walls < 1.5 ms:
  raise reps to 5, keep barrier alignment).
2026-07-09 ===== S6 MICRO v2 BANKED: QUEUE FLAVOR WINS; STREAMING WIN = 2.6x =====
  (artifacts profiling_gb200ep_sg/s6_micro_m{5120000,1280000,320000}.json; modes
  owned/sdp/sep-planner/queue x {natural, 0.15} x worst/median x M-sweep.)
  B-BYTES RECEIPTS (the mechanism): sdp streams EVERY bank on BOTH ranks (402.7 +
  402.7 MB); sep-planner streams each bank ONCE (201.3 + 201.3) — the bank-once
  consolidation is real and exactly halves weight traffic.
  WALLS: (i) STREAMING-BOUND regime (2.5k rows/expert): sep 2.01 ms vs sdp 5.27 =
  2.6x FASTER (exceeds the predicted 1.5-2x); queue 2.80 = 1.9x (partial — pops
  interleave banks); MG-B PASS. (ii) COMPUTE-BOUND (40k rows/expert) natural:
  queue 15.55 ~= sdp 15.96 (MG-C PASS for queue) BUT sep-planner 19.11 = +20%
  RED FLAG, and at alpha=0.15 sep = 48.5 ms (3x, imb 0.635) — DIAGNOSIS: whole-
  expert LPT hands hot experts to ONE rank as UNCHUNKED mega-segments => the
  known kernel-tail pathology; the planner flavor is structurally tail-prone.
  (iii) THE SLEEPER RESULT: queue at 40k/expert alpha=0.15 = 15.20 ms vs sdp
  25.02 — the queue's (segment, n_block) interleaving CURES the mega-segment
  tail INTRA-rank (the same mechanism behind e2e's +12.6%@0.15) — the queue
  helps WITHIN a rank at micro even where cross-rank balance is moot.
  DECISION (D-C resolved): e2e true-sEP = QUEUE-EMERGENT partition (no planner),
  + a BANK-AFFINITY ORDERING pass on the union list (cluster same-expert items
  per pop-end; distribute hot chunks across both ends) to close the remaining
  2.80 -> 2.01 streaming gap and the small-item-count side-affinity artifacts
  (queue imb up to 0.57 at small M + front-loaded hot expert). Also: fix the
  queue counter reconciliation so per-side B-bytes report (head+tail != claimed
  in all runs -> NA fallback fired).
2026-07-09 S6 PR-5 (ep_sep_probe.py): armed-launch protocol validated the hard way —
  TWO REAL BUGS found and fixed before any e2e exposure:
  (1) DECLINE DEADLOCK: the sdp-floor arming decision was per-rank local; one rank
      declining while the peer armed => peer spun forever on a header that never
      came. FIX: declines are PUBLISHED (host-written flag=2); if EITHER side
      declines BOTH fall back; every eligible call consumes one seq on both ranks
      (launch alignment under any arm/decline mix). Decline cost = one host store.
  (2) TMA-SYSMEM VISIBILITY RACE (the money receipt): the thief's GPU-side release
      flag (ep_steal_flag_set after the GEMM) could beat its FINAL TMA sysmem
      stores' arrival as observed by the PEER GPU'S gather — corruption signature:
      only the thief's LAST-executed stolen items, first M-tile intact, rows >=64
      stale (~50% zeros), victim side random per run. PR-4 never caught it (timing
      + luck class). FIX: the done flag is published THROUGH THE HOST (CUDA event
      synchronize, then a host store to the pinned flag) — host observation of
      kernel completion guarantees global visibility of all its writes to any
      observer of host memory. The wait is a LOCAL drain (own GPU), never
      peer-coupled => cannot re-create the stagger class. NOTE: this receipt also
      applies to the parked S2b armed path's flag ordering if ever revived.
  (2b) SAME CLASS, SECOND INSTANCE: the hdr/X-ready flag (side-stream flag_set
      after the async D2H copy) — the copy is executed by the DMA ENGINE, and an
      SM-thread's st.release.sys does NOT order another agent's writes; stream
      order only gates kernel START. Fixed identically (copy-event synchronize ->
      host store). RULE OF THE RECEIPT: any cross-GPU handoff through host memory
      must be published by the HOST after event-synchronize, never by a GPU flag.
  (3) BLOCK_M ALIGNMENT CONTRACT: unaligned segments corrupt NEIGHBOR segments
      through store-tile overhang — harmless intra-rank (later writes win,
      deterministic) but CROSS-SIDE under the union split (the 25-89-row boundary
      corruption with side-dependent stage/armed signatures). The e2e path is safe
      by construction (_pad_grouped_input_for_asym BLOCK_M-aligns every segment);
      the armed path MUST consume post-pad offsets, and the probe now does.
  VERDICT: PR-5 PASS 8/8 runs, BITWISE on every case — balanced, 3:1 skew with
  real cross-rank stealing + gather, published-decline fallback, ring/flag reuse.
  The collective-free true-sEP transport is validated at kernel level.
2026-07-09 S6 E2E WIRED + SMOKE B1 (logged before launch): frozen_linear armed hook
  (post-pad pairs -> ep_sep.try_armed; falls through to queued/plain on decline),
  fabric prebuild of sep_ctrl + X/D ring slots pre-seal (ASYM_EP_SEP_SLOT_ROWS
  default 655360 x kmax 2048 = 43 GB fabric extra; oversize m => decline = sdp
  floor), backends REPURPOSED per the plan: asym_sep2_cpuadamwds = TRUE sEP
  (ASYM_EP_SEP=1), asym_sqep2_cpuadamwds = + queued unarmed launches; sdp2/sqdp2/
  ep2 unchanged; drivers re-list sep2/sqep2 as their own canonicals (alias map,
  family/deepspeed/ep2_enable lists, router allow-list), _source.sh synced.
  SMOKE (asym_sep2 2048|8|1, MAX_STEPS=4): EXPECT ep_sep_installed both ranks;
  armed > 0 (m/seg ~1k <= 4096 floor); steps 9-13 s (sdp2-natural 8.27 s + X-stage
  D2H ~0.3-0.5 GB/launch + host event waits; the streaming win is NOT expected at
  e2e s2048 — fwd GEMMs are a small step fraction; this smoke is about CORRECTNESS
  + protocol overhead); loss overlay vs the study's sdp2 s2048 rows <= 0.01.
  RED FLAGS: NaN/overlay miss (transport corruption — kill immediately), > 15 s
  (serialization — check spin_wait_s + declined stats), sEP hdr spin timeout,
  ep_sep_install_FAILED, C-OOM (fabric cap: 99+43=142 < 160).
2026-07-09 ===== S6 E2E MILESTONE: FIRST TRUE-sEP TRAINING RUN PASSES =====
  asym_sep2_cpuadamwds @ 2048|8|1 (install fix: slots CREATED pre-seal, INSTALLED
  post-seal — pinnedness arrives with the fabric's single register):
  ep_sep_installed both ranks; steps 13.4/11.5/13.0/9.8 s (B1 band 9-13 ✓;
  protocol overhead over sdp2's 8.27 as predicted); LOSS OVERLAY <= 0.0035 at
  EVERY step vs the canonical s2048 reference — with union work-sharing live in
  the training loop. The collective-free true-sEP stack now exists end to end:
  micro receipts -> PR-5 bitwise transport -> e2e correctness.
  REMAINING (staged in S6, next session): armed/steal stats into the heartbeat;
  bank-affinity ordering (close the 2.80->2.01 micro gap); parity rows 20k/60k
  (must tie sdp2 within 2% via the decline floor); the STREAMING-WIN rows
  (short-seq / large-E, where the 2.6x micro receipt predicts the e2e payoff);
  sqep2 combined mode; scout port; per-rank streamed-bank-bytes receipt.
2026-07-09 S6 20K PARITY ROW BANKED: asym_sep2 @ 20000|8|1 = 56.9 s steady
  (56.7/56.9/57.2) vs sdp2 natural 55.7 (+2.2%, inside session noise; vs banked
  56.4 queue-era +0.9%); losses overlay <= 0.004. THE DECLINE FLOOR HOLDS AT
  COMPUTE-BOUND SCALE (every launch declines via two host stores — no vanilla-
  class behavior possible by construction). G2 parity gate: PASS-within-noise.
  NEXT (S6 continues): streaming-win rows (short-seq / large-E), armed-stats
  heartbeat, bank-affinity ordering, sqep2 mode, 60k parity, scout port.
2026-07-10 ===== TIER-2 GAMMA SWEEP BANKED: QUEUE FLAT ACROSS THE SEVERITY LADDER ====
  (user directive: 3-tier imbalance methodology — tier 1 real-trace replay [have],
  tier 2 gamma-sharpened real distribution [NEW: counts^gamma renorm — the
  literature-standard Zipf-shaping, preserves the recorded gradual multi-expert
  shape; RUNS token form gX in the bench alphas list], tier 3 single-hot-alpha
  bound [have]. Real-shape receipt: q3 natural is GRADUAL tilt — top1 only
  2.9-4.2% (3.7-5.4x uniform), top-32 experts ~= half the traffic; gamma ladder:
  top1 4->8->15->37% at gamma 1/1.5/2/3.)
  RESULTS (worst layer, 5.12M rows; wall ms | imbalance; s6_micro_gamma.json):
    gamma  owned         sdp           sep-planner    QUEUE
    1.0    16.7 | .25    16.0 | .003   19.2 | .03     15.1 | .038
    1.5    17.8 | .36    19.0 | .003   30.9 | .22     14.8 | .004
    2.0    19.4 | .48    28.1 | .010   48.9 | .40     14.6 | .004
    3.0    22.0 | .70    58.0 | .002   109.9| .63     14.7 | .001
  READS: (1) QUEUE DEAD FLAT 15.1->14.7 ms across the whole realistic ladder
  (slightly faster at high gamma — fewer hot banks = streaming locality);
  (2) sdp collapses 3.6x at gamma=3 DESPITE perfect row balance (imb .002) — the
  mega-segment kernel tail: intra-rank scheduling matters independently of
  cross-rank balance, and only the queue provides it; (3) owned hot-side B
  balloons to 1.3 GB (chunk re-streaming) vs queue consolidation; (4) planner
  flavor confirmed structurally tail-prone (6.6x) — the queue-flavor decision
  re-validated on the realistic knob. The 3-tier composite (real anchor +
  gamma curve + alpha bound) is now fully measured at micro level; e2e gamma
  knob (sharpened resampling in _compute_routing + |gX RUNS form) staged next.
2026-07-10 GAMMA SWEEP v2 — ROUTER-REALIZABLE CAP (user correction: with topk=8 an
  expert appears at most once per token => top-1 share ceiling = 1/8 = 12.5%; the
  gamma>=2 rows [15%/37%] exceed it and are RELABELED beyond-physical stress, never
  quotable as routing): reran gamma in {1, 1.25, 1.5, 1.65, 1.8} = top1 {4.2, 6,
  8.3, 10, 12}% (s6_micro_gamma_capped.json). WORST LAYER (wall ms | imb):
    top1    owned          sdp            QUEUE
    4.2%    16.7 | .25     16.0 | .000    14.6 | .008
    6.0%    17.1 | .30     16.9 | .003    15.3 | .067
    8.3%    17.8 | .36     19.0 | .004    14.8 | .003
    10%     18.1 | .39     21.2 | .006    15.7 | .062
    12%     18.6 | .42     23.9 | .005    15.2 | .099
  VERDICT WITHIN PHYSICAL BOUNDS: queue flat ~15 ms across the ENTIRE realizable
  range; sdp degrades 1.57x at the routing ceiling despite perfect row balance
  (the mega-segment tail); owned 1.24x with idle up to 42%. Median layer milder
  (all within ~10% — the advantage grows with tilt, costs ~nothing when calm).
  Same cap applies to the alpha knob (alpha <= ~0.12 realizable): the banked
  e2e 0.15/0.20 rows remain valid TIMING data but get the same stress label.
2026-07-09 NAMING EPOCH 3 (user directive — honest taxonomy): asym_sep2/asym_sqep2
  RENAMED to asym_sdp2_cpuadamwds / asym_sqdp2_cpuadamwds ("shared-bank streaming
  DP" — the system is DP in compute; the EP-flavored parts are the single shared
  expert bank + the dormant steal capability, and calling it EP was misleading).
  Old names (sep2/sqep2/sqeq2, short + long) remain ACCEPTED ALIASES that
  canonicalize to the new labels in run_lf_lora_sft.sh, so future runs land under
  sdp2/sqdp2 dirs regardless of alias used. asym_ep2 (vanilla, owned+dispatch)
  KEEPS its name — it is actually EP-shaped. Applied to both profile drivers
  (alias map, family/deepspeed/ep2_enable lists, router allow-list) + runlf;
  user's live _source.sh scratch preserved (surgical edits, no full copy).
  ARTIFACT NOTE: every asym_sep2*/asym_sqep2* run dir predates this entry; new
  runs produce asym_sdp2*/asym_sqdp2* dirs (no skip-collision across the rename).
O4 DP2-ANCHOR B1 EXPECTATION (logged before launch): T_dp2 row (DDP, per-rank arenas,
  find_unused=true, save-on-cpu OFF throughput posture). EXPECT wall 62-78 s
  (~1.02-1.28xT1: dense precedent 1.12x-class + the MoE reducer over 3.375e9 grads;
  our manual-allreduce T_ep2 = 60.7 s should WIN or tie); startup pays ~10 min of
  2x64 GB cudaHostAlloc (known contention datum, not a hang); RSS ~247 GiB/rank
  (~493 summed — vs ep2's one 99 GB fabric). RED FLAGS: DDP hang (find_unused
  plumbing), C-OOM (watchdog floor 35), T_dp2 < T_ep2 - 3 s (would contradict the
  no-DDP design premise — diagnose before believing).
  asym_ep2_cpuadamwds 2048|8|1 (global 16). EXPECT: torchrun 2 ranks; ep2_fabric_sealed
  heartbeat (banks > 0, register ~5-7 s at ~57 GB); rank0+rank1 shard receipts DISJOINT;
  per-rank step ~15-25 s class (|1-s2048 was 15-20 with fwd ~1.2/bwd 9-14) + ms-class
  allreduce; loss finite in the 1.5-2.5 band (global-16 trajectory differs from b8's by
  construction); NO DDP wrapper; both rank memstats written. RED FLAGS: seal-barrier
  hang (fabric bug), one-rank-only progress (sampler/launch bug), loss NaN/absurd
  (allreduce or shard bug), step >> 25 s (fabric contention or a serialization bug).
```
