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
  T_ep2  q3-30b-a3b|2 ; asym_ep2_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false
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
- RUNS env is MANDATORY (the in-file RUNS array is fully commented); rows join with '||'.
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
```
