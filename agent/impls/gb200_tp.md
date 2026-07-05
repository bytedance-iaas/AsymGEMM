# GB200 sTP vs TP Baselines: Staged Implementation Plan (`|2`)

Companion: `agent/gb200.md` (design derivation). This doc is self-contained: the
implementation-focused plan for sTP (streamed tensor parallelism)
and its TP baseline ladder. Style/discipline mirrors `fix_finegrained_*.md`: staged,
gated, one experiment at a time, artifacts never overwritten.

## Goal

```text
system:       asym_stp_cpuadamwds  = TP-2 across GPUs 0,1 of ONE superchip; frozen
              base weights stream tile-wise from ONE pinned Grace arena (existing
              asym kernels, per-device shard slices); LoRA + CPUAdamW unchanged
primary row:  llama3.3-70b | 25000|8|1 | ligerloss1 | recomp-off-full-fg-ker000
matrix:       q3-32b 50000|8|1, q2.5-72b 30000|8|1, q3-30b-a3b 80000|8|1 (ker101),
              llama4-scout 9500|8|1
paper names:  TP-Resident (tp2_resident_*), TP-Staged (tp2_offstage_*),
              AsymLoRA-DP (asym_dp2, attribution row), AsymLoRA-sTP (asym_stp_*)
```

## Contribution -> Evidence Map

```text
C1 disjoint-lane streaming: asym_stp vs asym_dp2; per-lane weight bytes ~0.5x,
   step_s ~0.5-0.6x at equal global workload
C2 zero-residency shards: step_H resident > staged > streamed; seq frontier
   asym_stp >= 1.8x tp2_resident (honest: resident WINS step_s where it fits)
C3 shared-arena dedup: arena=1 vs 0; residual host bytes + D2H ~0.5x
C4 tile-wise act consumption: asym_stp vs tp2_offstage boundary (2-3x)
C5 coordination: coord=1 vs 0; 10-30% step_s or DROP the claim
C6 scaling: {|1 b8} vs {asym_dp2 b4x2} vs {asym_stp b8}; dp2 <=1.2x, stp 1.6-2x
```

## Profiling Goals (dev; real models, real workloads)

```text
|1 pace car (established): superoffload_mem|unsloth-off|ligerloss1  b8
|2 memory axis (expect):   zero3_offload_mem|unsloth-off   b4/GPU
|2 time axis   (expect):   superoffload_mem|unsloth-off    b4/GPU
Crown via P0; record winners here:
  |2 memory pace car = [TBD after P0]    |2 time pace car = [TBD after P0]
```

P0 pace-car sweep (runnable BEFORE any sTP code; each row separately, pair 0,1):

```bash
RUNS='llama3.3-70b|2 ; superoffload_mem|unsloth|ligerloss1 ; 25000|4|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_p0 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# repeat with superoffload_mem|unsloth-off and zero3_offload_mem|unsloth-off
# then b4 boundary probes for the two winners: seq 30000 -> 40000 -> 50000, stop at first OOM
```

Dev goals (ALL must hold before paper phase):

```text
P1 llama3.3-70b 25000|8|1: step_H(stp) < step_H(mem pace car b4) AND
   step_s(stp) < step_s(time pace car b4); loss in band
P2 q3-32b 50000|8|1: same        P3 q2.5-72b 30000|8|1: same
P4 q3-30b-a3b 80000|8|1 ker101 (after Stage I7): same
P5 boundary b8 (llama3.3-70b, q3-32b): max seq(stp) >= 1.5x mem pace car
P6 mechanism health at P1: both lanes >= 170 GB/s in streamed windows;
   dup_factor 1.0; per-lane weight bytes ~0.5x of asym_dp2
DECIDABILITY RULE for P1-P4: if a pace car OOMs at the target row (prior |1
   boundaries make b4 completion at these rows non-certain), the comparison
   shifts to that car's max-runnable seq at b4, and the stp row is reported as
   beyond-frontier — P goals stay decidable without renegotiation.
```

## Baselines (summary; full rules unchanged from prior revision)

```text
Tier 1 run-as-is: superoffload_mem|unsloth[-off] b4, zero3_offload_mem|unsloth-off
  b4; Ulysses = LF-SP if available else cited (DECIDED: no in-stack asym-SP —
  SP==DP on the weight axis; at b=1 TP halves the working set same as SP)
Tier 2 stock-API TP — UPDATED after Automodel exploration: run BOTH FW rows via
  vendored NeMo Automodel (torchrun, 2 procs) as pure YAML configs:
  FW1 TP-Resident = distributed:{tp_size:2, dp_size:none} + peft block
    (template: examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml,
    swap model to llama3.3-70b/q3-32b);
  FW2 TP-Staged  = FW1 + fsdp2.offload_policy: torch.distributed.fsdp.CPUOffloadPolicy
    (wired at recipes/_dist_utils.py:148, config.py:221).
  This KILLS the peft-on-DTensor risk (Automodel ships its own DTensor-aware
  LoRA; fw2_feasibility_probe reduces to "does the YAML run + emit metrics").
  Metrics shim: profile.json emitter around their recipe loop.
  BRIDGE RULE unchanged: official baseline number = max(our rung, FW row).
  Megatron-Bridge: paper-required 2 rows (fits-throughput + OOM boundary);
  raw Megatron-LM excluded with reason (no LoRA).
Tier 3 in-codebase ladder: tp2_resident/tp2_offstage/asym_stp one knob apart +
  asym_dp2. Never presented as prior work.
Fairness: TP rows b8 (=global 8); DP rows b4/GPU; pair 0,1 only
  (ALLOW_CROSS_SUPERCHIP=1 for the 0,2 contention study); loss within ~0.05;
  fresh artifacts; KV heads % 2 == 0 verified per model.
```

## Verified Borrow List (2026-07-04 deep exploration of Megatron-Bridge, Automodel, Megatron-Core)

```text
IMPORT AS-IS (pure functions, no dist required):
  Megatron-Bridge param_mapping.py: merge/split_qkv_weights (GQA interleave,
    :3128-3354), GatedMLPMapping per-rank [gate_i;up_i] concat (:2645-2649),
    ColumnParallelMapping chunk-dim0 (:1003), RowParallelMapping chunk-dim1,
    _get_shard_spec KV-replication (:528-571)  -> drives I2 repack-at-load.
    NOTE: our DENSE LF/HF modules are UNFUSED (separate q/k/v, gate, up) ->
    plain dim0/dim1 chunks at head-group boundaries suffice. The EXPERT banks
    ARE fused today (qwen3 [E,2I,H], llama4 [E,H,2I]) but EP-2 slices dim0
    (experts) only, so no fused split is ever performed (see I2 grouped rule).
  Megatron-Core: tensor_parallel/utils.py split fns; moe_utils permute/unpermute/
    sort_chunks_by_idxs + aux-loss fns; random.py checkpoint + RNG tracker.
  Automodel optimized_tp_plans.py (:340 llama, :488 qwen) — authoritative
    col/row plan tables for our exact target models (use as the plan-assert
    source alongside HF base_model_tp_plan).
SEMANTICS TO COPY (wiring yes, transport swapped to our P2P exchange):
  MCore mappings.py 7-collective duality table — every collective's backward.
    Col-parallel: fwd identity / BWD ALLREDUCES dX (placed at the REGION entry,
    Megatron's copy_to f-operator — our TPRegionFn). Row-parallel: fwd allreduce /
    bwd identity. => per decoder block: exactly 2 O(M*H) exchanges fwd AND 2 bwd,
    by construction, even with unfused q/k/v/gate/up.
  LinearWithFrozenWeight (layers.py:350-453): frozen base saves ONLY weight,
    dgrad + one sync allreduce, NO wgrad — our streamed base's backward contract.
  Backward ordering (layers.py:506-655): dgrad -> schedule collective async ->
    wgrad -> wait. RowParallel bias-added-AFTER-reduce (:1350). skip_bias_add.
  LoRA layout, two independent refs AGREE on the invariant: A inherits base
    INPUT sharding, B inherits base OUTPUT sharding, rank-r is the replicated
    rendezvous. Row-case implementation per Automodel lora.py:296-306: scale
    BEFORE B, keep base and LoRA outputs BOTH as partial sums, add, then ONE
    exchange — replaces our earlier "add on dev0 only" hack (cleaner, no
    double-count hazard). Fused adapters: split B only, A shared
    (peft_bridge.py:730-736, :362-374).
  MoE: MCore AllGather dispatcher (token_dispatcher.py:212-351) MINUS its
    all-gather (residual already replicated) = our I7 no-a2a EP-2, verbatim
    mask+permute+combine semantics.
  Activation offload template: fine_grained_activation_offload.py pinned pool +
    saved-tensor hooks + dual D2H/H2D streams (cross-check for our manager).
DO NOT REUSE (incompatible):
  All three training stacks are multi-process SPMD (one GPU per process, NCCL
  process groups); no single-process dual-device path exists anywhere; no
  weight streaming exists anywhere (Bridge cpu_offloading_weights = TE bulk
  per-layer swap, PP=1-only; Automodel = FSDP2 whole-unit offload) — paper
  claim receipts.
```

## Global Efficiency Rules (apply to every stage; violations = design bugs)

```text
E1 NEVER split M (tokens). Shards are N/2 or K/2 — still huge GEMMs.
E2 MoE: ONE grouped kernel per device over its E/2 experts. No per-expert Python
   loops, no per-expert launches beyond what the |1 grouped path already does.
E3 One collective per TP region counts the O(M*H) exchanges: attention -> 1
   AllReduce2Fn fwd + 1 TPRegionFn bwd; MLP -> same. EXEMPT: tiny O(M*r)/O(H*r)
   LoRA-grad exchanges (dS_full per col adapter, row-dB), budget <= 7 per layer
   bwd with unfused modules (q,k,v,gate,up dS + o,down dB); OPTIONAL batching
   (concat qkv dS -> [M,3r], gate/up -> [M,2r]) reduces it to 4 — an E3 audit
   must not flag exchanges within budget. allreduce2 = 1 P2P copy each direction
   on copy streams + local add. No NCCL process groups, no barriers.
E4 Launch pattern per op: enqueue dev0 kernel, enqueue dev1 kernel, back-to-back,
   both async; cross-device ordering via events ONLY at collective points. No
   cudaDeviceSynchronize in steady state; no new .item()/host reads (the only
   host reads remain the existing MoE token counts).
E5 Weight repack happens ONCE at load (weights frozen). Zero runtime relayout.
E6 Backward concurrency is free: torch autograd runs one worker thread per
   device, so dev0/dev1 backward nodes overlap without extra code.
```

## Evidence Discipline

Same as fix_* docs. One experiment at a time (exception: the two ranks inside one
asym_dp2 row). New OUTPUT_ROOT per stage. Before each run write expected
{model, pair, backend, WEIGHT_MODE, arena, coord, per-device+global batch, artifact
tag, comparison row, likely failure}. After: command.txt (all ASYM_STP_* echoed),
train.log, profile.json.config, per-device step_H, loss band, numa_maps, and —
once their emitting stage has landed — lane_bw.json (from I1) and
arena_breakdown dup_factor (from I5); earlier runs (P0, dp2) are judged without
them. Labels: `validated | blocked_by_stage_bug |
inconclusive_wrong_config | inconclusive_partial_profile |
inconclusive_stale_artifact | inconclusive_unexpected_path`. Never advance on
inconclusive. E2E LoRA profiling is the acceptance bar for every stage that touches
training semantics; isolated micro-tests are acceptable ONLY for I1 (pure runtime
plumbing) and kernel-level parity probes.

## Stage I0: Harness Plumbing + P0 Baselines

Intended change: make `|2` runs launchable and auditable before any sTP code.

Scope (files / functions):

```text
scripts/lf/run_lf_profiled_train.py:577-599   add asym_stp, asym_stp_cpuadamwds,
    tp2_resident_cpuadamwds, tp2_offstage_cpuadamwds to the backend sets
scripts/lf/run_lf_lora_sft.sh:390-399         BACKEND whitelist — new names die
    here FIRST (`*) ... exit 2`); add asym_stp*, tp2_* to the whitelist
scripts/lf/run_lf_lora_sft.sh:56,413,445      NUM_GPUS gates:
    asym_stp*/tp2_*  -> require NUM_GPUS=2, launch SINGLE-PROCESS (favorable
                        fact: is_torch_run (:685) keys on BACKEND=="torch", so
                        no torchrun suppression needed)
    asym*/non-stp    -> require NUM_GPUS=1 (unchanged)
scripts/lf/run_lf_lora_sft.sh:55,1836,1858,2176,192   GPU visibility plumbing:
    GPU_ID is scalar and CUDA_VISIBLE_DEVICES="${GPU_ID}" pins ONE GPU; the pair
    launch needs GPU_ID to carry "g0,g1" through :2176 and the nsys metrics
    devices at :192
scripts/lf/run_lf_lora_sft.sh:1144            TP global-batch override:
    if ASYM_STP=1: PROFILE_GLOBAL_BATCH_SIZE=$((BATCH*GA))   # NOT *NUM_GPUS
scripts/lf/run_lf_lora_sft.sh (env echo)      echo ASYM_STP, ASYM_STP_TP_SIZE,
    ASYM_STP_WEIGHT_MODE, ASYM_STP_SHARED_ARENA, ASYM_STP_COORD into command.txt
    + profile.json.config
scripts/lf/profile_lora_lf_test_source.sh:868,886  accept |2 with stp backends;
    forward BOTH pool GPUs to one job (today |2 only feeds the torchrun path);
    refuse --gpus 0,2 unless ALLOW_CROSS_SUPERCHIP=1
scripts/lf/profile_lora_lf_test_source.sh:894-903  backend_gpu_count() — the
    REAL dispatch (dies on unknown backends; hard-codes asym* -> 1 GPU ignoring
    the model's |2, so today asym at |2 silently coerces to 1 GPU): add
    stp/tp2 -> 2 mapping and make asym* at |2 DIE (this is what makes the
    negative guard below actually fire); also the :3392 GPU_ID/NUM_GPUS hand-off
backend->env derivation (defines mode_for_backend, makes the positive dry-run
    self-contained): the harness DERIVES ASYM_STP=1 and defaults
    {WEIGHT_MODE: asym_stp*->stream, tp2_resident*->resident,
     tp2_offstage*->stage; ARENA/COORD: asym_stp*->1/1,
     tp2_offstage*->0/0 pinned, tp2_resident*->0/0 pinned (rung purity: the
     arena/coord claims are ablated on asym_stp only)} FROM the backend name.
    PER-KNOB override semantics (a blanket must-match would make the I5
    arena/coord ablations unlaunchable): ASYM_STP, ASYM_STP_TP_SIZE (=2) and
    WEIGHT_MODE are DERIVED — explicit values must match or die; SHARED_ARENA
    and COORD are DEFAULTS — overridable on asym_stp* (that IS the I5 ablation
    mechanism) BUT explicit overrides DIE until the I5 code lands (otherwise an
    arena0/coord0-tagged artifact runs with arena1/coord1 behavior — the exact
    mislabeled-artifact class Evidence Discipline forbids); pinned on tp2_*
    (explicit override always dies)
artifact tag: append stpW<mode>_arena<0|1>_coord<0|1>_tp2 to config_label
    (:1827) so every ablation gets a distinct directory
```

Pseudocode (bash gate):

```bash
case "${BACKEND}" in
  asym_stp*|tp2_*)
    [[ "${NUM_GPUS}" == 2 ]] || die "${BACKEND} requires NUM_GPUS=2"
    # derive BEFORE the match check (unset env = the normal case, must not die);
    # the derivation exists in BOTH the profile script (dry-run tag) and
    # run_lf_lora_sft.sh (direct launches)
    ASYM_STP_WEIGHT_MODE="${ASYM_STP_WEIGHT_MODE:-$(mode_for_backend "${BACKEND}")}"
    [[ "${ASYM_STP_WEIGHT_MODE}" == "$(mode_for_backend "${BACKEND}")" ]] \
      || die "backend/WEIGHT_MODE mismatch"
    LAUNCH_MODE=single_process_pair ;;   # CUDA_VISIBLE_DEVICES=g0,g1; no torchrun
  asym*|kt_*) [[ "${NUM_GPUS}" == 1 ]] || die "${BACKEND} is single-GPU" ;;
esac
```

Validation gate (all must pass):

```bash
# positive dry run
RUNS='llama3.3-70b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 128|8|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_gb200tp_dryrun RUNS_LOG=profiling_gb200tp_dryrun/runs.log \
GPU_POOL=0,1 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

```text
echo shows num_gpus=2 and single-process launch. Global-batch proof: DRY_RUN
  never executes run_lf_lora_sft.sh (where :1144 computes it) — the I0 change
  must ALSO print the derived global batch in the dry-run echo/command.txt, and
  the real :1144 override is verified by the I3 s2048 run (profile.json global
  batch == 8, NOT 16)
path contains __gpus2__ ... stpWstream_arena1_coord1_tp2
negative guards die: asym_stp at |1; asym_cpuadamwds at |2; mode mismatch;
  --gpus 0,2 without ALLOW_CROSS_SUPERCHIP=1; ASYM_STP=1 with LORA_DROPOUT != 0
  (dropout breaks the x0/x1 bit-identity invariant that I5 dedup and I7
  identical-topk depend on; harness default is 0.00 — guard the sweeps);
  ASYM_STP=1 on llama4 with router_mode != "whole" or router_debug_grad set
  (llama4_moe.py:280-289 detaches only conditionally — a differentiable router
  adds an unsummed per-branch router-path dx and breaks DispatchFn's sum);
  ASYM_STP=1 on ANY MoE model until the I7 code lands (pre-I7 the expert path
  is dev0-only -> x0/x1 silently diverge; gate on an ASYM_STP_MOE flag that I7
  introduces); tp2_* with explicit ASYM_STP_COORD or ASYM_STP_SHARED_ARENA set
  (pinned knobs); asym_stp_cpuadamwds with ASYM_STP_SHARED_ARENA=0 pre-I5
  (override-dies-until-I5 must also be exercised once, same rationale)
P0 pace-car sweep completes and winners are recorded above; per-rank
  profile.json for the DP baselines is sane (fix aggregation if not — this is a
  known unknown of the torchrun path, resolve HERE)
```

Also lands here (zero code dependencies, gives early motivation data):
`scripts/lf/run_dp2_pair.sh` — launch two |1 asym jobs concurrently (GPU0/GPU1,
b4, same seed/dataset), wait both, emit dp2_merged.json {per-rank profile paths,
summed RSS, max wall-clock}. This is the asym_dp2 attribution row AND the
shared-Grace contention evidence, runnable before any sTP code exists.

CRITICAL edge case (RESOLVED design, must implement here): HF Trainer with 2
visible GPUs sets n_gpu=2 and silently wraps the model in nn.DataParallel
(Trainer._wrap_model), which would run our surgery twice per step and corrupt
everything. run_lf_profiled_train.py never builds TrainingArguments itself — it
monkey-patches Trainer (:1243-1341); the mitigation lives in that patch layer,
AFTER HF device setup (an early `_n_gpu` write would be overwritten), and the
assertion must target `model_wrapped` (Trainer wraps into `trainer.model_wrapped`
inside train(); `trainer.model` is NEVER the DP wrapper — asserting on it is
vacuous):

TIMING FACT (verified in vendored transformers): Trainer.__init__ freezes
`self._train_batch_size = per_device * max(1, n_gpu)` (trainer.py:591,
training_args.py:1754) and `_inner_training_loop` builds the dataloader (:1445)
BEFORE `_wrap_model` (:1577) — so an `_n_gpu` write inside `_wrap_model` is TOO
LATE: DP wrap is prevented but the step still consumes batch 16. And
`_setup_devices` is a `@cached_property` (training_args.py:1772): any write
AFTER first access sticks. Therefore patch at `Trainer.__init__`:

```python
# inside the existing Trainer monkey-patch layer (run_lf_profiled_train.py:1243-1341):
if os.environ.get("ASYM_STP") == "1":
    _orig_init = Trainer.__init__
    def _init(self, *a, **k):
        args = k.get("args") or a[1]
        _ = args.n_gpu                 # force the cached _setup_devices first
        args._n_gpu = 1                # sticks; runs BEFORE _train_batch_size freeze
        _orig_init(self, *a, **k)
        assert self._train_batch_size == args.per_device_train_batch_size
    Trainer.__init__ = _init
    _orig_wrap = Trainer._wrap_model   # belt: DP wrap must still never appear
    def _wrap_model(self, model, *a, **k):
        wrapped = _orig_wrap(self, model, *a, **k)
        assert not isinstance(wrapped, torch.nn.DataParallel), "stp: DP wrap leaked"
        return wrapped
    Trainer._wrap_model = _wrap_model
```

Validation add-on (trainer-side receipt, NOT the env-derived value —
profile.json's global batch comes from PROFILE_GLOBAL_BATCH_SIZE (:1144->:2343)
and reads 8 by construction even when the Trainer consumes 16): emit
`trainer._train_batch_size` via the existing heartbeat/partial_writer into
train.log at step 1; the I3 s2048 gate requires it == 8.

Risks/watch: (a) torchrun-path profile.json may report rank0 only — audit and fix
in this stage, not later; (b) dataset registration loss on new roots (known
failure mode — verify dataset_info.json rows exist before runs); (c) accelerate
may have its own device-placement pass — verify with the same assertion.

## Stage I1: STPRuntime (single-process pair; isolated validation allowed)

Intended change: one process owns both GPUs; streams, P2P, collectives, NUMA.

Scope:

```text
asym_gemm/training/stp_runtime.py   NEW: class STPRuntime (singleton via env)
asym_gemm/integrations/lf.py:1149   device resolution: replace
    torch.device("cuda", current_device()) with rt.primary (dev0) when ASYM_STP=1
scripts/testing/stp_runtime_probe.py  NEW probe
scripts/lf/extract_lane_bw.py       NEW (tooling lands HERE, not I6 — I3+ gates
    depend on it): parse nsys export -> lane_bw.json {per-GPU h2d/d2h GB/s
    p50/p95 + bytes, nvlink tx/rx, windows = streamed-GEMM NVTX spans}
```

Pseudocode:

```python
class STPRuntime:
    def __init__(self, dev_ids=(0, 1)):
        self.d = [torch.device("cuda", i) for i in dev_ids]
        for a, b in ((0, 1), (1, 0)):
            assert torch.cuda.can_device_access_peer(dev_ids[a], dev_ids[b])
        # enable peer access NOW (a remote alloc does NOT do it — must be a copy)
        torch.zeros(1, device=self.d[0]).to(self.d[1]); torch.zeros(1, device=self.d[1]).to(self.d[0])
        self.compute = [torch.cuda.Stream(device=x) for x in self.d]
        self.copy    = [torch.cuda.Stream(device=x) for x in self.d]
        self._bind_numa_and_cores()   # membind to pair's node; SiLU pools 4-39/40-71
        # coord DEFAULTS TO 1 from I1 on (this call is unconditional here); the
        # coord=0 path lands in I5 (tp2_* pinning is I0's backend->env
        # derivation) — pre-I5 `coord1` artifact tags are therefore honest

    def allreduce2(self, y0, y1):
        # CONTRACT FACTS: (1) PyTorch enqueues a cross-device copy on the SOURCE
        # device's CURRENT stream (ATen Copy.cu) -> each direction must be issued
        # under the SOURCE side's copy stream or it lands on the default stream
        # and serializes. (2) y_i is both READ (by the outgoing copy) and WRITTEN
        # (by the in-place add) -> the add must wait BOTH copy-done events
        # (write-after-read hazard), not just its incoming one.
        # ready-events on the streams that actually PRODUCED y_i: record on
        # torch.cuda.current_stream(y_i.device) at Fn entry — producers include
        # ambient-stream ops (norms, flash-attn, scatter) and autograd's replay
        # streams, NOT necessarily rt.compute[i]
        e0 = record(current_stream(y0.device)); e1 = record(current_stream(y1.device))
        self.copy[1].wait_event(e1)                                  # src=dev1 reads y1
        with torch.cuda.stream(self.copy[1]): t0 = y1.to(self.d[0], non_blocking=True)
        self.copy[0].wait_event(e0)                                  # src=dev0 reads y0
        with torch.cuda.stream(self.copy[0]): t1 = y0.to(self.d[1], non_blocking=True)
        c_t0 = record(self.copy[1]); c_t1 = record(self.copy[0])     # copy-done
        for ev in (c_t0, c_t1):                                      # BOTH events,
            self.compute[0].wait_event(ev); self.compute[1].wait_event(ev)  # both adds
        with torch.cuda.stream(self.compute[0]): y0 += t0
        with torch.cuda.stream(self.compute[1]): y1 += t1
        # EXIT contract: make each device's AMBIENT stream wait the add's event
        # before returning, so ordinary consumers (cat, residual add, autograd)
        # are ordered after the exchange without knowing about our streams.
        # ALLOCATOR note: `with torch.cuda.stream(copy[1])` changes only dev1's
        # current stream — t0 (a dev0 tensor) is allocated on dev0's AMBIENT
        # stream; call t.record_stream(consuming compute stream) (or take t from
        # the slab pool) to keep the caching allocator honest.
        return y0, y1     # both directions still move concurrently (full-duplex)

    def to0(self, y1):                # dev1 -> dev0 P2P; issued under copy[1]
                                      # (SOURCE side — same Copy.cu rule as above)
    def bcast01(self, x0):            # dev0 -> dev1 P2P; issued under copy[0]
    def to0_sum(self, y0, y1):        # y0 += to0(y1)  (Phase-A return path)
    def bcast01_from_host(self, h):   # pinned host -> dev1 H2D on LANE1 copy stream
                                      # (used by I5 residual restage: both lanes
                                      # pull the ONE pinned copy concurrently)

# out-slab pool: OWNED by STPRuntime, keyed (M, N, dtype). LIFETIME RULE: the
# pool serves ONLY outputs captured by a saved-tensor/offload reference (qkv/gu
# cat slabs) — TAKEN per forward call, RETURNED when that reference is released.
# EXEMPT (fresh torch.empty + record_stream, both directions): any exchange
# output NOT so captured — fwd AllReduce2Fn sums (p_i, q_i feed only the
# residual add, never saved: pool slabs would have no release trigger and
# accumulate ~2 x [M,H] x 80 layers ~ 525 GB/device within ONE forward) and all
# bwd exchange outputs (same arithmetic, no saved-tensor trigger exists in bwd).
# NEVER reuse a slab while autograd still holds it — weight_offload's pool is
# safe only because weights are never saved-for-backward; outputs ARE.
#
# AUTOGRAD CONTRACT (the raw helpers above are NOT autograd-safe; I4 wraps the
# cross-device exchanges as torch.autograd.Functions; regime = SPMD emulation,
# i.e. the REGION BOUNDARY (TPRegionFn bwd) leaves the FULL summed dX on BOTH
# branches — col-Fns themselves never exchange):
#   AllReduce2Fn : fwd = exchange + sum on both devs ; bwd = identity per dev
#   Bcast01Fn    : fwd = copy dev0 -> dev1           ; bwd = pass g0, DROP g1
#                  (g1 is bit-identical to g0 under this regime — DEBUG_HASH
#                  assert; SUMMING would double-count: dev1's contribution
#                  already arrived via the first region boundary's bwd exchange)
#   Join01Fn     : fwd = (x0, x1) -> x0              ; bwd = g -> (g, bcast01(g))
#   TPRegionFn   : fwd = identity on (h0, h1)        ; bwd = allreduce2 of the two
#                  ACCUMULATED LOCAL dX partials — the Megatron f-operator, placed
#                  at each TP-region ENTRY (the norm output feeding qkv / gate_up)
#   DispatchFn   : the MoE instance of TPRegionFn (branch entry — see I7)
# RULE that makes the arithmetic close with UNFUSED q/k/v/gate/up modules:
# col-Fns ALWAYS return LOCAL dX partials (never exchange internally); autograd
# sums q+k+v(+LoRA) partials locally on each device, and the ONE TPRegionFn at
# the region entry performs the single exchange. => exactly 2 O(M*H) bwd
# exchanges per dense block even with 5 separate col modules.
# RULE (two-branch Functions): any op whose backward EXCHANGES (TPRegionFn,
# AllReduce2Fn, DispatchFn, row-dB, dS_full) must be a SINGLE autograd Function
# holding both branches; per-device module clones are allowed ONLY for
# exchange-free ops (norms, rotary, silu, scatter). E6's free concurrency
# applies to the exchange-free nodes.
# STREAM DISCIPLINE (BINDING for EVERY STP primitive, including stp_base_gemm,
# to0, bcast01, and the offload copies — not just allreduce2): at entry, the
# consuming stream waits an event recorded on each operand's PRODUCING stream
# (record it where the producer ran — inside the producing with-block if the
# producer was a compute-stream kernel; current_stream(dev) if ambient); at
# exit, the device's ambient stream waits the result event before the value is
# returned to ordinary consumers. Violations are silent data races (the I3
# fwd x-read, y1 hand-off, and bcast01-consume are exactly this class).
# IN-PLACE RULE (both directions): Fn forwards must NOT do y_i += t_i in place
# on Fn INPUTS (autograd version-counter errors) — write into fresh slab
# outputs, or mark_dirty(). Fn BACKWARDS must NOT mutate their incoming
# grad_outputs either (autograd may alias/reuse those buffers; mark_dirty does
# not exist for backward) — TPRegionFn/DispatchFn bwd use an OUT-OF-PLACE
# allreduce2 variant summing into FRESH torch.empty outputs on the consuming
# compute stream (+ record_stream; the caching allocator handles reuse). Bwd
# outputs are EXEMPT from the slab pool: its lifetime rule is forward-only, and
# a bwd-held slab (no saved-tensor release trigger) would accumulate
# ~2 x [M,H] x 80 layers ~ 525 GB/device at P1. The dS_full/row-dB exchanges
# are exempt from the no-mutation rule only because their operands are freshly
# created inside backward.
# Join01Fn sits at the TOP of the stack (before final-norm/lm_head) and is what
# replaces SPMD's replicated loss seed — without it, dev1's shard grads are zero.
```

Validation (isolated OK — no training semantics):

```bash
python scripts/testing/stp_runtime_probe.py --pair 0,1   # includes topology checks
```

```text
P2P copy >= 700 GB/s/dir sustained, both directions concurrently
both C2C lanes ~190 GB/s H2D concurrently reading the SAME pinned buffer
allreduce2 of a [200000, 8192] bf16 tensor (3.3 GB) < 8 ms
CUDA_DEVICE_MAX_CONNECTIONS A/B: run the probe with =1 and unset; measure
  cross-device kernel overlap + per-kernel enqueue cost; PIN the winner here
  and echo it into command.txt from I3 onward
numa_maps shows the probe's own pinned TEST allocations on the pair's node
  (the weight arena itself only exists from I2 — re-check it there);
  SiLU pools pinned 4-39 / 40-71
|1 smoke row with ASYM_STP unset: zero regression
```

Risks/watch: event/stream deadlock patterns (keep ONE canonical allreduce2, no
ad-hoc syncs); Python launch overhead for 2 devices from 1 thread (measure in
probe: enqueue cost per kernel < 30 us; acceptable because our GEMMs are ms-scale
— E4).

## Stage I2: Sharded Arena — repack-at-load (contiguity is the law)

Intended change: one pinned copy of every frozen weight, physically laid out so
each device's shard is a CONTIGUOUS block. Motivation (verified in code):
`_direct_bf16_reason` requires `b_cpu.is_contiguous()` (frozen_linear.py:415) and
the grouped variants likewise; the C++ side could take outer strides
(gemm.hpp:550-552 uses b.stride(-2); runtime_utils.hpp:109,137 encode strides) but
we do NOT relax the gate — strided-B is an optional later optimization. Weights
are frozen: repack once at load, zero runtime cost (E5).

Scope:

```text
asym_gemm/training/stp_layout.py   NEW:
    plan(model_cfg) -> {module_name: ("col"|"row"|"attn_col"|"attn_row", split_dim)}
    assert_plan_matches_hf(plan, cfg.base_model_tp_plan)   # llama/qwen3
    shard_spec(kind, shape) -> [(dev, slice), (dev, slice)]
asym_gemm/training/host_weight.py  HostWeight gains:
    repack_for_stp(kind): for "col" ([N,K], split N): shards are already
      contiguous dim0 slices — just record offsets, NO copy.
      for "row" ([N,K], split K): allocate ONE pinned buffer sized N*K and copy
      W[:, :K/2] -> block0, W[:, K/2:] -> block1 (each [N, K/2] contiguous);
      free the original.
    GROUPED EXPERT BANKS: EP-2 ONLY — per-device bank = contiguous dim0 slice
      at the E/2 boundary, ZERO-COPY, no repack. NO N/K grouped split exists
      anywhere in this plan. The banks are FUSED today and must never be
      N-bisected (it would separate gate from up): qwen3 [E, 2I, H] out_in
      fused gate_up (qwen3_moe.py:2516 gate=fused[:, :I, :]); llama4 [E, H, 2I]
      in_out fused (llama4_experts.py:803, chunk(2, dim=-1)).
      ARENA-AWARENESS REQUIRED: _ensure_qwen3_moe_finegrained_bases
      (qwen3_moe.py:2509-2543) lazily materializes SPLIT gate/up bank COPIES at
      first fine-grained use — under sTP each per-device clone would allocate
      its own pinned copies, breaking one-copy/dup_factor/RSS gates on P4/scout.
      Fix at load: repack the split gate/up banks INTO the arena once (EP-2 dim0
      slices of the split banks stay zero-copy) and make _ensure… return arena
      views; add the llama4 in_out analog check.
    shard_view(dev) -> pinned, contiguous, aligned tensor view into the ONE buffer
asym_gemm/integrations/lf.py       load path calls repack when ASYM_STP=1
```

Pseudocode (row repack; the only one that copies):

```python
def repack_row(w):                       # w: pinned [N, K] bf16
    buf = pin_alloc(w.numel())           # ONE allocation, same total bytes
    h = K // 2
    b0 = buf[: N * h].view(N, h); b0.copy_(w[:, :h])
    b1 = buf[N * h :].view(N, h); b1.copy_(w[:, h:])
    return b0, b1                        # both contiguous, both pinned, one copy total
```

Validation (unit OK — layout only):

```text
for every wrapped Linear of llama3.3-70b: shard views are pinned, contiguous,
  K/2 and N/2 pass the 8/64-alignment gates in _direct_bf16_reason
host RSS after load == |1 RSS (one copy; repack transient < 1 layer's size —
  repack layer-by-layer, free as you go)
reassembly check PER LAYER AT REPACK TIME, before freeing the original (the
  stage frees originals as it goes — post-load there is nothing left to compare
  against): allclose(cat(shards), original), or record a checksum then
plan assertion passes vs transformers base_model_tp_plan for all matrix models
```

Risks/watch: split points MUST be 64-aligned (mandatory — the transpose_b gate
requires shard-N % 64, see I3). Verified for the matrix models: 28672/2=14336,
25600/2=12800, 29568/2=14784 are all 64-multiples, so even splits work
everywhere today; keep the pad rule (N0=ceil64(N/2), N1=N-N0, kernels accept
uneven shard sizes) for future models. embedding/lm_head stay UNSHARDED.

## Stage I3: sTP Dense Linears, Phase A (fan-out only; model stays on dev0)

Intended change: streamed base GEMMs (fwd base, bwd dX) execute split across both
devices; everything else (attention math, norms, LoRA, residual, optimizer)
untouched on dev0. This is e2e-runnable WITHOUT attention surgery and proves the
lane pooling.

Scope — ONE CHOKE POINT, because the primary backend does NOT go through
AsymFrozenLinear (critical, verified): under `recomp-off-full-fg` the MLP GEMMs
run inside `_FinegrainedDenseMLPFunction` (dense_mlp_finegrained.py:183, base-GEMM
helper calls at :220/:236/:339/:399/:423) and the attention GEMMs inside
`_AsymActivationOffloadLoRALinearFunction` (attention_activation_offload.py:560,
:601/:694) — `AsymFrozenLinear.forward:2062` is never called on that path.
Splitting only frozen_linear would hit the doc's own "lane1 ~0 silent fallback"
symptom on every gate. Therefore:

```text
asym_gemm/training/stp_runtime.py     NEW stp_base_gemm(x_i, shard_i, phase,
    transpose_b, mode) — THE single choke point every base GEMM routes through;
    I6's resident/stage/stream mode dispatch lives INSIDE it
asym_gemm/training/frozen_linear.py   route _dispatch_nt call sites (fwd :1288+,
    bwd-dX :1341+) through stp_base_gemm when ASYM_STP=1 (template semantics
    unchanged); AsymFrozenLinear.forward:2062 dispatch for the non-fg path
asym_gemm/training/dense_mlp_finegrained.py   route the base-GEMM helper calls
    (:220/:236/:339/:399/:423) through stp_base_gemm — this is the path the
    P1 gates actually execute
asym_gemm/training/attention_activation_offload.py   same routing (:601/:694)
asym_gemm/training/cpu_left.py        UNCHANGED in I3 — grouped/expert paths
    keep the |1 single-device path until I7 (EP-2); no grouped N/K split exists
scripts/testing/stp_gemm_parity_probe.py   NEW (declared here; used by this gate)
```

Dataflow + pseudocode, grounded in the REAL kernel calls (verified): the fwd core
is `_dispatch_nt(x_2d, host_weight.weight, phase="forward", ...)` and bwd-dX is
`_dispatch_nt(grad_2d, host_weight.weight, phase="dx", transpose_b=True, ...)`
(frozen_linear.py:1288-1370); the fg Functions call the same kernels via their
helpers. Both fwd and bwd consume the SAME shard slice — col shard
W_i = W[N_i rows, K] serves fwd as B and bwd as B^T. The STP Function is a thin
orchestrator around two per-device calls:

```python
class STPFrozenLinearColFn(Function):
    @staticmethod
    def forward(ctx, x, hw0, hw1, ...):              # x on dev0 [.,K]; hw_i = per-device
        x1 = rt.bcast01(x)                           #   HostWeight shard views (I2)
        with torch.cuda.stream(rt.compute[0]):
            y0 = _dispatch_nt(x.reshape(-1, K), hw0.weight, phase="forward", ...)   # [M,N0]
        with torch.cuda.stream(rt.compute[1]):
            y1 = _dispatch_nt(x1.reshape(-1, K), hw1.weight, phase="forward", ...)  # [M,N1]
        y1_0 = rt.to0(y1)                            # copy stream + event
        return out_slab.cat_(y0, y1_0)               # preallocated [M,N] slab, no realloc
    @staticmethod
    def backward(ctx, g):                            # g [M,N] on dev0
        g1 = rt.bcast01(g[:, N0:].contiguous())
        with torch.cuda.stream(rt.compute[0]):
            dx0 = _dispatch_nt(g[:, :N0].contiguous(), hw0.weight,
                               phase="dx", transpose_b=True, ...)   # partial [M,K]
        with torch.cuda.stream(rt.compute[1]):
            dx1 = _dispatch_nt(g1, hw1.weight, phase="dx", transpose_b=True, ...)
        dx = rt.to0_sum(dx0, dx1)                    # ONE P2P + add (E3)
        return dx, None, None, ...
# row case mirrors with roles swapped: fwd partials -> allreduce2/to0_sum;
# bwd dX_i = _dispatch_nt(g_replicated, hw_i.weight, transpose_b=True) stays local -> cat.
```

STREAM NOTE on the snippet above: it shows the LOGICAL dataflow only — every
operand hand-off obeys the I1 STREAM DISCIPLINE (x's ambient-producer event
waited by compute[0]/copy[0]; y1's ready-event recorded ON compute[1] INSIDE
its with-block, not at ambient after exit; bcast01/to0 results event-ordered
before consumption). A literal transcription without those waits races.

Alignment constraint (verified in `_direct_bf16_reason`): the transpose_b path
requires k % 64 where k = the shard's N_i — so shard split points MUST be
64-aligned (the I2 pad rule is mandatory, not defensive). Matrix models checked:
28672/2, 25600/2, 29568/2 are all 64-multiples — even splits work everywhere in
the current matrix.

Efficiency notes: no M split (E1); the two GEMM launches are back-to-back async
(E4); Phase-A broadcast cost is real (~x [M,K] per call over 900 GB/s NVLink) and
accepted ONLY for this stage — it disappears in I4 when the residual is
replicated. LoRA stays entirely on dev0 in this stage (base GEMMs only).

Validation (kernel parity isolated; acceptance is E2E):

```bash
python scripts/testing/stp_gemm_parity_probe.py --model llama3.3-70b --cases col,row --mode stream
# e2e loss gate
RUNS='llama3.3-70b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I3_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
# e2e target profiling (PROFILERS=both: the lane_bw/nsys gates need the nsys pass;
# this convention applies to EVERY gate below that reads lane_bw/nvlink/class-byte
# artifacts — I4/I5/I6 commands inherit it)
RUNS='llama3.3-70b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 25000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I3_s25000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
PROFILERS=both \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
```

```text
parity MATCH (col, row, uneven shards)
s2048 loss within ~0.05 of |1 asym at same global workload
I0 receipts land HERE: profile.json global batch == 8 (the :1144 override) AND
  train.log step-1 heartbeat shows trainer._train_batch_size == 8 (the
  Trainer.__init__ patch beat the :591 freeze — loss band alone cannot catch a
  silently-doubled consumed batch)
s25000: nsys/lane_bw shows BOTH lanes active in base-GEMM windows; the
  base-GEMM component of bwd_s ~halves vs the |1 run; step_H(dev0) <= |1 step_H
if lane1 ~0 -> silent fallback bug; inconclusive_unexpected_path
```

Risks/watch: cat([y0, y1_to0]) allocates; reuse a preallocated output slab keyed
by (M,N) like weight_offload.py's slab pool; broadcast/bcast01 must ride copy
streams or it serializes compute.

## Stage I4: Full TP Residency (replicated residual; attention head-split; Phase-A
broadcasts deleted)

Intended change: both devices hold the residual stream (replicated, Megatron
pattern); norms/rotary/elementwise run redundantly on both devices (cheap, E4);
attention is head-split; MLP col/row wired to local residual copies; ONE
allreduce2 after o_proj and ONE after down_proj per layer; LoRA layouts final.

Dataflow per decoder layer (fwd; bwd mirrors with the transposed comm points):

```text
x0 [M,H]@dev0, x1 [M,H]@dev1 (bit-identical)
attn: n_i = rmsnorm(x_i)                      # both devs, local
      qkv_i = n_i @^ Wqkv_i^T                 # col-split by heads, own-lane stream
      a_i = flash_attn(rope(qkv_i))           # full seq, half heads, local
      p_i = a_i @^ Wo_i^T                     # row-split, partial [M,H]
      p_i += lora_o_i                         # LoRA partial (scale-before-B, see
                                              #   row-layer rule below) — stays partial
      p0,p1 = allreduce2(p0,p1)               # collective #1 (sums base+LoRA together)
      x_i = x_i + p_i                         # residual add, local
mlp:  m_i = rmsnorm(x_i)
      gu_i = m_i @^ Wgateup_i^T               # col-split [M, 2F/2]
      s_i = silu_mul(gu_i)                    # local halves (CPU or GPU per policy)
      q_i = s_i @^ Wdown_i^T                  # row-split (K=F/2 shard), partial
      q_i += lora_down_i                      # LoRA partial, same rule
      q0,q1 = allreduce2(q0,q1)               # collective #2
      x_i = x_i + q_i
```

Phase-A -> I4 backward delta (do not miss): in Phase A the col-Fn backward ends
with `to0_sum` (residual lives only on dev0; degenerate boundary). In I4 the
col-Fn backward returns its LOCAL dX partial with NO exchange — the exchange
moves to the TPRegionFn at the region entry (I1 rule; per-module exchanges would
be 3 O(M*H) per attention with unfused qkv). `residual_mode` selects
bcast-in/to0_sum-out (Phase A) vs local-in/local-out-partial (I4).

AUTOGRAD REGIME (one regime, stated once — mixing regimes computes wrong grads):
every cross-device exchange in the I4 graph is EITHER one of the four boundary
Functions from the I1 contract (AllReduce2Fn, Bcast01Fn, Join01Fn,
TPRegionFn/DispatchFn) OR a tiny O(M*r)/O(H*r) exchange living INSIDE a single
two-branch module Function (dS_full, row-dB — the I1 two-branch rule, E3
exemption); nothing else may cross devices. Boundary placement:
  bottom: Bcast01Fn right after embedding (fwd copy x0->x1; bwd passes g0 and
    DROPS g1 — under SPMD emulation g1 is bit-identical to g0 because the first
    region boundary (TPRegionFn bwd) already left the full summed dX on both
    branches; summing here would be a latent 2x on embedding-side grads);
  top: Join01Fn right before final-norm/lm_head (fwd passes x0, consumes the x1
    branch; bwd hands the SAME grad to both x0 and x1) — this is what replaces
    SPMD's per-rank replicated loss. WITHOUT Join01Fn, dev1's row-shard partials
    (o/down halves) receive zero gradient and every col-layer allreduce2 sums a
    real partial with a zero: silently ~half-magnitude grads.
  interior: AllReduce2Fn after o_proj and after down_proj (fwd sum, bwd identity);
    TPRegionFn at each norm output feeding a col region (attention entry, MLP
    entry) — col-Fns return LOCAL dX partials and the TPRegionFn does the single
    backward exchange (see the I1 contract rules). lm_head/liger stay dev0-only.
  Exchange budget per dense block: fwd = 2 (the two AllReduce2Fn), bwd = 2 (the
    two TPRegionFn duals) O(M*H) exchanges — holds by construction even with
    unfused q/k/v/gate/up.

LoRA layout (final; the col-only rule is WRONG for row layers):

```text
col layers (qkv, gate_up) — ONE dataflow, primary/fallback never both:
  fwd: A [r,H] replicated; S = x_i @ A^T identical on both devs (x replicated);
       B [N_i,r] col-sharded; y_i += scale * S @ B_i^T locally (no exchange).
  bwd: dB_i = local ([N_i,r], sharded param -> local grad, no exchange).
       dS_i = g_i @ B_i stays PARTIAL for the dX path: dX_lora_i = dS_i @ A is a
       LOCAL partial that autograd sums into the region's accumulated dX; the
       region's single TPRegionFn bwd exchange covers base+LoRA dX for ALL col
       modules of the region at once. dS_full must NOT feed dX.
       dA PRIMARY (X offloaded, = the I5 kernel path): dS_full = allreduce2(dS_i)
       (tiny [M,r]); per-device grouped_lora_a_grad_cpu_right on its X K-half;
       dA = cat(halves). FALLBACK (X not offloaded): dA_i = dS_i^T @ x partial ->
       one tiny allreduce2. NEVER both — double-count otherwise.
  Backward duality (MCore contract): the col region's dX partials MUST be
  summed across devices exactly once — done by the region's TPRegionFn, NOT
  per module (per-module exchanges with 3 unfused qkv linears would be 3
  O(M*H) exchanges; the region boundary keeps it at 1).
row layers (o, down) — Automodel formulation (lora.py:296-306), REPLACES the
  earlier add-on-dev0-only hack:
  fwd: A [r,K_i] row-sharded consumes the local width-shard; apply LoRA scale
       BEFORE B; keep base partial p_i and LoRA partial l_i = B(scale * A_i x_i)
       both UNREDUCED; the block's single allreduce2 sums (p_i + l_i) — one
       exchange covers base+LoRA, no double-count by construction
       (B(sum_i A_i x_i) == sum_i B(A_i x_i)). B [H,r] replicated.
  bwd: dB = allreduce2 of per-device partials dB_i = g^T (scale * S_i) (tiny
       [H,r] — B is REPLICATED, its grad needs the sum); dA_i local ([r,K_i]
       sharded param); dX_lora_i local (row bwd = identity).
```

Trainable-state ownership (single source of truth = the EXISTING pinned CPU LoRA
slabs; resolves who steps the optimizer and how devices stay fresh):

```text
nn.Parameters registered ONCE, on the dev0 model object (Trainer sees n_gpu=1);
  dev1 holds plain-tensor mirrors only. Setup counter: stp_dev1_params_registered=0.
weight_offload.py already re-gathers LoRA banks CPU->GPU per layer per step:
  extend to per-device gathers (dev_i pulls its shards + the replicated pieces
  over its OWN lane). Staleness impossible BY CONSTRUCTION — every step
  re-gathers from the CPU slab CPUAdamW just updated; no post-step push needed.
grads: sharded params (col-B_i, row-A_i, dev1 expert adapters) -> local grad,
  D2H over the owning device's lane into the existing grad-offload path;
  replicated params: row-B -> allreduce2 partials, D2H ONCE from dev0;
  col-A PRIMARY path -> the dA K-halves are cat'd on dev0 and D2H once (the
  dS_full exchange already summed the contributions); col-A FALLBACK ->
  allreduce2 partials, D2H once. Grad-norm/clipping sees each logical param
  exactly once in every case.
CPUAdamW unchanged (steps the CPU masters exactly as in |1).
```

Scope:

```text
asym_gemm/integrations/lf.py:1717-2415   wrap points: instantiate per-device
    module clones sharing the arena shard views (exchange-free ops only — every
    exchanging op is a single two-branch Function per the I1 rule); residual
    duplication at embedding output via Bcast01Fn (the autograd Function, NOT
    the raw helper — a raw .to() leaves a ToCopyBackward edge that SUMS g1,
    contradicting the drop-g1 regime); logits: lm_head on dev0 only, consumes
    x0 via Join01Fn
asym_gemm/training/dense_mlp_finegrained.py + attention_activation_offload.py
    THE EXECUTED PATH for the P1 backend (I3 established AsymFrozenLinear is
    never called under recomp-off-full-fg): restructure both fg Functions into
    two-branch region Functions (per-device silu halves, per-device LoRA
    partials, dS_full/row-dB exchanges inside the Function per the I1 rule);
    fwd-S moves to GPU HERE (S = x_i @ A^T from the live replicated x — the
    cpu-left fwd-S calls at dense_mlp_finegrained.py:227/:243 are dropped in
    I4, NOT in I5; leaving them would lose dev1's LoRA delta silently)
asym_gemm/training/frozen_linear.py      STP fns lose the Phase-A bcast (inputs
    already local); attention QKV/O wiring for the non-fg path
scripts/testing/stp_grad_parity_probe.py  NEW + env ASYM_STP_DUMP_GRADS=1 hook:
    dumps per-adapter grads and 1-batch logits — AT STEP 2, not step 1: PEFT
    default init has B=0, so at step 1 dS=0 => dA identically ZERO in both |1
    and |2 runs, hiding exactly the double-count/placement bugs this gate
    exists to catch (only dB is exercised). Step 2 runs after one CPUAdamW
    update makes B nonzero (alternative: ASYM_STP_PROBE_B_INIT forces nonzero
    B at load). rel-err uses an absolute floor (|a-b| / max(|b|, 1e-8)).
asym_gemm/training/activation_offload.py  per-device pools land HERE (moved
    from I5 — I4's own s25000 gates produce dev1 halves ~17+ GB/layer that
    cannot stay resident, and the existing manager's D2H stream/event
    bookkeeping is dev0-bound: dev1-sourced copies would be unordered):
    pool[dev], per-device D2H/H2D streams + ready events. Byte cap IN I4:
    dev0 = full ASYM_EXPACT_CPU_POOL_MAX_BYTES, dev1 = MAX/2 — dev0 still
    carries the |1-style full-K X blocks until the I5 layout lands, and a /2
    cap would thrash its free-list eviction into repeated cudaHostAlloc during
    the I4 gates. The /2 split (better: ONE shared counter across both pools,
    which I5's coord token bucket introduces anyway) lands WITH I5. I5 keeps
    only the LAYOUT changes (residual dedup, two-block X, split-K dA) + coord.
asym_gemm/training/stp_layout.py         head split DERIVED from config, never a
    hand-written table: num_attention_heads/2, num_key_value_heads/2, assert
    %2==0 (llama3.3-70b: q 64->32/32, kv 8->4/4); rotary caches PER DEVICE
HF Trainer guard: model.to(device) must not migrate dev1 modules/HostWeights —
    HostWeight already refuses .to(cuda) (host_weight.py:317-335); add the same
    guard for dev1-resident buffers
```

Validation (E2E is the bar):

```text
parity WITH BANDS (exact equality is unavailable — fwd-S moved CPU->GPU and
  reduction orders differ): 1-batch logits |1 vs stp max-abs-diff <=
  max(measured envelope, fixed bf16 tolerance) — a plain seed-to-seed |1
  envelope is plausibly ZERO (deterministic fixed-order run + B=0 at step 1),
  which would fail CORRECT code; derive the envelope from a reduction-order-
  perturbed |1 reference (e.g. a different kernel-config run), measure ONCE,
  record in the Decision Log; 5-step seeded loss overlay within band;
  ADAPTER-GRAD PARITY at STEP 2 (see probe note in Scope — step 1 is vacuous
  for dA): per-adapter max-rel-err of dA/dB vs the |1 run <= 1e-2 (bf16) —
  the loss band alone can mask a 2x error on a subset of adapters (the exact
  failure mode of a wrong exchange placement)
e2e P1 run (commands as I3 with OUTPUT_ROOT=profiling_gb200tp_I4_s25000):
  step_H per GPU <= 0.55x of |1 asym step_H at s25000 (intermediates halved)
  step_s(stp) < step_s(|2 time pace car b4)   [P1 first half]
  allreduce2 time < 15% of layer time and overlapped (nsys)
  both-lane weight bytes ~ W/2 each (lane_bw.json)
then P2 (q3-32b 50000|8), P3 (q2.5-72b 30000|8) same gates
```

Risks/watch: (a) per-device autograd thread overlap — verify in nsys that dev0/
dev1 backward kernels overlap; if serialized, the graph has a false dependency
(usually an accidental same-device intermediate); (b) rotary/pos-cache device
mismatch; (c) uneven head counts for q2.5-72b GQA (kv=8 ok); (d) Trainer/accel
device moves — the |1 guards must fire, add a setup-report counter
`stp_dev1_modules_wrapped`; (e) liger loss on dev0 only — logits path unsharded,
unchanged; (f) Qwen3 q_norm/k_norm operate on head-sharded tensors — replicate
per device, never seq-shard them (Automodel optimized_tp_plans.py:512 caveat);
(g) CUDA_DEVICE_MAX_CONNECTIONS: pinned by the I1 A/B probe (a value of 1 may
serialize our cross-device overlap even though Megatron requires it); the pinned
value must appear in command.txt from I3 onward.

## Stage I5: Activation Path + Shared-Arena Dedup + Coordination

Intended change (per-device pools themselves landed in I4): (1) residual
checkpoints offloaded ONCE (dedup — they are bit-identical across devices);
(2) two-block X-offload LAYOUT + bwd LoRA-A grad split-K so both lanes stream
disjoint X halves; (3) coord knob = membind + core split + joint prefetch budget.

Scope + pseudocode:

```text
asym_gemm/training/activation_offload.py   pools/streams landed in I4; I5 only
    re-buckets layouts (below) and adds the residual-dedup path
asym_gemm/training/decoder_activation_offload.py   residual ckpt dedup:
    def offload_residual(x0, x1):        # bit-identical by construction
        h = pool[0].offload(x0)          # ONE D2H, dev0 lane
        return h                         # dev1 does NOT offload
    def restage_residual(h):
        r0 = pool[0].restage(h, rt.d[0]) # H2D lane0
        r1 = rt.bcast01_from_host(h)     # H2D via lane1 directly from the SAME
                                         # pinned buffer (both lanes pull the one
                                         # copy concurrently — no NVLink needed)
    dup_factor accounting -> arena_breakdown.json
WRITER/CALLER SCOPE (the P1 gate is DENSE — scoping only the expert wrapper
    makes the gate unattainable). I5 changes ONLY the X-offload LAYOUT + the dA
    split-K rebalance (the fwd-S move to GPU already happened in I4). Bucket by
    layer kind — the buckets are NOT interchangeable:
    TWO-BLOCK ADOPTERS (replicated-input layers, qkv/gate_up): the ctx.x_cpu
      writers + dA callers at dense_mlp_finegrained.py:397/:421 (via
      _cpu_right_lora_a_grad :805-819) and the qkv-input path in
      attention_activation_offload.py.
    PER-DEVICE LOCAL POOLS (sharded-input layers — their X is the local width
      half ALREADY; a two-block split here is a WRONG transform, K is the local
      F/2): dense_mlp_finegrained.py:335 (down dA on ctx.act_cpu = silu output)
      and the o-proj role in attention_activation_offload.py.
    NOT llama4_experts.py / the expert wrapper: expert X is the per-device
      PACKED gather (llama4_experts.py:681 _rebuild_packed_x_cpu) — local,
      non-replicated, I7's per-device pools own it (pre-I7 the expert path
      stays dev0-only). The kind rule, spelled out:
      replicated-input layers (qkv, gate_up): X identical on both devs ->
        dedup split-K two-lane offload (the layout below);
      sharded-input layers (o, down; silu outputs): input is the per-device
        width-half ALREADY — each device offloads its own local half to its own
        pool over its own lane; nothing shared, no split-K needed.
asym_gemm/training/exp_act_offload_lora.py:231-256   dA split-K. Verified
    signature: grouped_lora_a_grad_cpu_right(dS [M,r] CUDA, source_cpu [M,K]
    pinned CONTIGUOUS, offsets, experts) -> [E, r, K]. A pinned [:, :K/2] view is
    non-contiguous and would be rejected — so the OFFLOAD LAYOUT changes instead:
    # forward: X is replicated on both devices (I4); offload it as TWO contiguous
    # K-half pinned blocks, EACH WRITTEN BY ITS OWN DEVICE over its own lane:
    # .contiguous() launches a compaction kernel — order it after x_i's producer
    # (wait current_stream(x_i.device) event) per the I1 entry/exit discipline:
    with stream(rt.copy[0]): wait_producer(x0); X0_cpu.copy_(x0[:, :K//2].contiguous(), non_blocking=True)
    with stream(rt.copy[1]): wait_producer(x1); X1_cpu.copy_(x1[:, K//2:].contiguous(), non_blocking=True)
    # => one host copy total (dedup), both blocks kernel-legal (contiguous),
    #    AND the D2H traffic itself is lane-balanced. Three birds, one layout.
    # backward:
    with stream0: dA0 = grouped_lora_a_grad_cpu_right(dS,  X0_cpu, ...)  # [E,r,K/2] lane0
    with stream1: dA1 = grouped_lora_a_grad_cpu_right(dS1, X1_cpu, ...)  # [E,r,K/2] lane1
    dA = cat(dA0, dA1.to(dev0), dim=-1)                  # tiny [E,r,K]
    (dS_full is ALREADY on both devices — it is the output of the I4 primary
    rule's allreduce2; no bcast exists. The pair variant
    grouped_lora_a_pair_grad_cpu_right splits identically on its x_cpu.)
    TIMING NOTE: this two-block X layout lands HERE (I5). In I4, K-half views of
    the |1 single-block pinned X are non-contiguous and rejected — so I4 runs
    the |1-style full-K dA on dev0 from dS_full, and I5 rebalances to split-K;
    the "~halves vs I4" gate below measures exactly that rebalance.
coord knob: STPRuntime._bind_numa_and_cores gated by ASYM_STP_COORD; prefetch
    budget: cap concurrent host-touching bytes via a token bucket in the
    offload managers (shared counter, both devices)
```

Validation (E2E):

```text
A-run arena=1 vs A-run arena=0 at P1 workload (fresh roots
  profiling_gb200tp_I5_arena{1,0}):
  residual class D2H bytes ~0.5x; RAM lower by ~(sum residual ckpt bytes);
  dup_factor 1.0 vs 2.0; step_s not worse
bwd LoRA-A grad kernel time: COL-adapter class (q,k,v,gate,up dA) ~halves vs I4
  (it now uses both lanes); whole dA class lands ~0.6-0.7x, NOT 0.5x — row
  adapters (o,down) were already per-device in I4 (their X is the local width
  half), so split-K only rebalances the col share. Must show on the e2e bwd_s.
coord=1 vs coord=0: step_s delta recorded (C5; drop claim if ~0)
P5 boundary sweep b8: max seq >= 1.5x |2 memory pace car
```

Risks/watch: restaging the single residual copy to BOTH devices doubles lane-H2D
for that tensor vs sharded-SP designs — bounded: residual is [M,H], small vs
gate/up traffic; watch it in offload_io class counters. Bit-identity of x0/x1
must be asserted in debug mode (allreduce2 determinism: same add order on both
devices — it is, by construction t+y local order; keep an ASYM_STP_DEBUG_HASH=1
path).

## Stage I6: Ladder Modes + asym_dp2 + Metrics Extraction

Intended change: the two baseline rungs and the measurement tooling.

Scope + pseudocode:

```text
ASYM_STP_WEIGHT_MODE dispatch INSIDE stp_base_gemm (the I3 choke point — NOT
AsymFrozenLinear.forward, which the fg backend never calls):
  stream   -> I3/I4 path (kernels on host shard views)
  resident -> at setup: shard_view(dev).cuda(dev) once; forward = torch.matmul
              (plain GEMM, no staging, no streaming) — vanilla TP rung
  stage    -> per layer: slab[dev].copy_(shard_view(dev), non_blocking=True) into
              a double-buffered HBM slab (2 slabs/device, prefetch next layer on
              copy stream), forward = torch.matmul from slab; per-rank pinned
              buffers, arena=0, coord=0 pinned by backend gate
(run_dp2_pair.sh and extract_lane_bw.py already landed in I0/I1)
```

Validation (E2E ladder at P1 workload, each row separately, fresh root
`profiling_gb200tp_I6`):

```text
step_H: resident > stage > stream (C2)
stage rung: slab prefetch overlaps (nsys: H2D during prior layer's GEMMs)
asym_dp2 completes; RAM ~2x stream rung's host weights; per-lane weight bytes
  ~2x stream rung's (C1); step_s(stream) ~0.5-0.6x dp2 wall (C6)
bridge check when FW rows exist: our rungs >= FW rows or adopt FW numbers
```

Risks/watch: resident rung may OOM at P1 already (70 GB shard + s25000 acts on
189 GB) — that IS a result row, record it and gate its throughput row at s8192.

## Stage I7: MoE — EP-2 (q3-30b-a3b ker101, llama4-scout ker000)

Intended change: experts split E/2 per device; ZERO token all-to-all — the
residual is replicated (I4), so each device already holds ALL tokens; each device
gathers its experts' tokens locally and runs ONE grouped kernel (E2).

Dataflow + pseudocode:

```text
n_i = moe_norm(x_i)                              # pre-MoE norm, local both devs
h0, h1 = DispatchFn(n0, n1)                      # wraps ONLY the MoE-branch input
                                                 # (NOT the residual — wrapping x_i
                                                 # itself would route the residual
                                                 # pass-through grad into the bwd
                                                 # sum and DOUBLE it every layer).
                                                 # fwd identity; bwd = allreduce2 of
                                                 # the two branch-dx PARTIALS: each
                                                 # device's expert path produces
                                                 # only its E/2 experts' share, and
                                                 # without this sum every layer
                                                 # BELOW an MoE layer gets wrong
                                                 # grads (loss band won't catch it)
router: logits_i = h_i @ Wg^T on BOTH devices (Wg tiny, replicated) ->
  identical topk on both -> no communication. INVARIANT the design relies on:
  routing weights are DETACHED from the graph (qwen3_moe.py:2569-2570 raises on
  requires_grad; llama4_moe.py:287-289 detaches) — a non-detached router would
  add an unsummed router-path dx per branch.
dispatch (per device, existing |1 machinery unchanged):
  my_experts = experts[dev]                      # E/2 contiguous bank block (I2)
  idx_i = flatten tokens routed to my_experts    # from the SAME topk both devs hold
  in_i = gather(h_i, idx_i)                      # local, replicated branch input
  out_i = grouped_ker101(in_i, bank_shard[dev])  # ONE grouped launch, E/2 experts
  y_i = zeros[M,H]; scatter_add(y_i, idx_i, out_i * gate_prob)
  y_i += shared_expert_partial_i(h_i)            # scout only — see rule below
combine: y0,y1 = AllReduce2Fn(y0,y1)             # union-sum; AUTOGRAD Function
                                                 # (in the forward graph), never
                                                 # the raw helper
  x_i = x_i + y_i                                # residual add reads the
                                                 # PRE-Dispatch x_i
# exchange budget per MoE layer: 1 fwd (combine) + 1 bwd (DispatchFn dual) —
# identical to a dense block, INCLUDING the shared expert (it rides both).
```

Scope: `asym_gemm/training/moe.py`; `qwen3_moe.py` — entry
`_forward_qwen3_moe_finegrained_offload` :2556-2568, token-threshold logic
`_ThresholdedQwen3ExpertFunction` :1509 + threshold wiring :2764-2785 (per
device); bank repack (I2 grouped rule); routed kerNNN kernels UNCHANGED;
introduces `ASYM_STP_MOE=1` (unlocks the I0 MoE guard — without this landing
here, the I7 parity/P4 commands die at the launcher with no owner).

Validation (E2E):

```text
parity s2048 ker101 vs |1 (loss band; routed counters fire on both devices;
  route bits identical to the |1 MoE plan; ADAPTER-GRAD PARITY on adapters in
  layers BELOW the first MoE layer — the exact tensors a missing DispatchFn
  corrupts while the loss band still passes)
e2e P4: q3-30b-a3b 80000|8|1 gates vs |2 pace cars; expert-imbalance stats
  logged per step (max/mean tokens per device)
llama4-scout 9500|8|1 ker000 row
```

Shared expert (llama4-scout HAS one) and expert-LoRA rules:

```text
shared expert (scout): col/row-split, consumes the DispatchFn OUTPUT h_i, and
  its col-Fns run in PARTIAL mode (local dX partials, NO internal exchange —
  the region's exchange IS DispatchFn's bwd, which sums routed+shared partials
  together); its output partial adds into y_i BEFORE the combine AllReduce2Fn.
  Budget stays 1 fwd + 1 bwd. Forbidden forms: replicated-and-summed
  (double-count); consuming h_i with a SELF-exchanging col-Fn (its full dX would
  be doubled by DispatchFn's sum); computed-on-one-device-then-added-after-
  combine (breaks the x0/x1 bit-identity invariant that I5 dedup and
  identical-topk depend on).
routed-expert LoRA under EP-2: adapters for dev_i's experts are LOCAL to dev_i
  (mirrors gathered from the CPU slab per the ownership rule; grads local ->
  D2H over dev_i's lane). Their X offload is fully LOCAL too — expert inputs
  in_i are per-device token gathers, NOT replicated, so the I5 split-K dedup
  does NOT apply: per-device pools, local dA, no exchange.
```

Risks/watch: expert imbalance -> one device idles (log it; EP fallback: a
starved device may stream any expert's shard directly from the arena — the
tile-stream path makes this free of ownership, add only if imbalance > 20%);
determinism of topk across devices (same dtype/math on both — assert with
DEBUG_HASH); llama4's in_out bank layout ([E,H,2I]) repack rule differs (dim
check per architecture in stp_layout).

## Stage I8: Paper Matrix + Defense Rows (only after I3-I7 validated)

Intended change: no new system code — the external rows, the ablation matrix,
and the reviewer-defense artifacts.

```text
FW rows (Automodel YAMLs, torchrun 2-proc, metrics shim):
  FW1 TP-Resident and FW2 TP-Staged at P1/P2 workloads + their b8 boundaries;
  apply BRIDGE RULE (official baseline number = max(our rung, FW row))
Megatron-Bridge rows (paper-required): one fits-in-HBM throughput point
  (e.g. q3-32b s8192) + its OOM/seq boundary; record integration cost honestly
Ulysses row: LF sequence-parallel if this venv supports it (check once, record);
  else cite Megatron-DeepSpeed-SO with the GPT-only/no-LoRA reason IN the table
Contention study: best baseline + asym_stp on pair 0,2 (ALLOW_CROSS_SUPERCHIP=1)
  vs pair 0,1 — the same-superchip/membind figure
Boundary matrix: b8 (and b1 long-tail) sweeps for every runnable backend on
  llama3.3-70b + q3-32b; OOM cells are REPORTED RESULTS (workload, error, max
  achieved), never omissions
Workload-capability table (paper artifact): system x {LoRA, TP, base-weight
  offload, fine-grained act offload, model > HBM} -> runs/cannot; names
  Megatron-Bridge and NeMo AutoModel explicitly (resident-TP only, no weight
  streaming — receipts in Verified Borrow List); raw Megatron-LM excluded with
  reason (no LoRA)
DeepSpeed-inexpressibility claim: verify against vendored deepspeed source and
  record file:line BEFORE stating (offload_param only in ZeRO-3; ZeRO-3 cannot
  host TP; AutoTP-training composes only with ZeRO-0/1/2)
```

DONE criteria (the whole plan):

```text
P1-P6 all hold (Profiling Goals)
step_H(asym_stp) < step_H(tp2_offstage) < step_H(tp2_resident) at every target row
seq boundary(asym_stp) >= 1.8x tp2_resident and > tp2_offstage on >= 2 models
seq boundary(asym_stp) >= 2x tp2_offstage on >= 1 model (backs C4's headline
  multiplier; if measured lower, soften C4 to "ratio reported" — never claim
  an unbacked number)
step_s(asym_stp) ~0.5-0.6x asym_dp2 at equal global workload
arena ablation ~0.5x residual host bytes + D2H; coord ablation recorded (or C5 dropped)
tp2_resident short-seq throughput win REPORTED, not hidden
every reported row passes the audit checklist and loss band
```

## Memory/BW Decomposition If A Gate Fails

```text
Workload  Backend  stpTag  step_H(g0/g1)  RAM  act_H  lane0/1 GBps  nvlink  dup_factor  top_peak_owner
```

Answer: which device/class owns the peak; live operand vs saved act vs slab vs
allocator reserve; replicated-but-should-be-sharded tensor (dup_factor); lane1
idle (silent fallback); prefetch/slab budget blown; RSS on wrong NUMA node. Only
then propose a change; log the decision in the Decision Log below.

## Decision Log (append-only; date + decision + evidence path)

```text
(empty — first entry lands with the first gate deviation)
```

## Reporting Format

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM  (+ lane0/1, nvlink GB/s, dup_factor on gate stages)
```

Labels exactly as generated (`asym_stp_cpuadamwds | recomp-off-full-fg-ker000`,
stp tag fragment, `__gpus2__`). No row is final without the audit: pair 0,1,
global-batch parity, CPUAdamW family, loss in band, fresh artifact.
