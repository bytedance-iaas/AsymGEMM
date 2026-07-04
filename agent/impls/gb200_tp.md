# GB200 sTP vs TP Baselines: Staged Implementation Plan (`|2`)

Companion: `agent/gb200.md` (design), `agent/impls/gb200_aware.md` (umbrella plan).
This doc is the implementation-focused plan for sTP (streamed tensor parallelism)
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
```

## Baselines (summary; full rules unchanged from prior revision)

```text
Tier 1 run-as-is: superoffload_mem|unsloth[-off] b4, zero3_offload_mem|unsloth-off
  b4; Ulysses = LF-SP if available else cited (DECIDED: no in-stack asym-SP —
  SP==DP on the weight axis; at b=1 TP halves the working set same as SP)
Tier 2 stock-API TP: FW1 = HF tp_plan; FW2 = HF tp_plan x FSDP2 CPUOffloadPolicy
  (dp1 x tp2); gated on fw2_feasibility_probe; BRIDGE RULE: official baseline
  number = max(our rung, FW row). Megatron-Bridge: paper-required 2 rows
  (fits-throughput + OOM boundary); raw Megatron-LM excluded with reason (no LoRA).
Tier 3 in-codebase ladder: tp2_resident/tp2_offstage/asym_stp one knob apart +
  asym_dp2. Never presented as prior work.
Fairness: TP rows b8 (=global 8); DP rows b4/GPU; pair 0,1 only
  (ALLOW_CROSS_SUPERCHIP=1 for the 0,2 contention study); loss within ~0.05;
  fresh artifacts; KV heads % 2 == 0 verified per model.
```

## Global Efficiency Rules (apply to every stage; violations = design bugs)

```text
E1 NEVER split M (tokens). Shards are N/2 or K/2 — still huge GEMMs.
E2 MoE: ONE grouped kernel per device over its E/2 experts. No per-expert Python
   loops, no per-expert launches beyond what the |1 grouped path already does.
E3 One collective per TP block: attention -> 1 allreduce2, MLP -> 1 allreduce2.
   allreduce2 = 1 P2P copy each direction on copy streams + local add. No NCCL
   process groups, no barriers.
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
train.log, profile.json.config, per-device step_H, lane_bw.json, arena_breakdown
dup_factor, loss band, numa_maps. Labels: `validated | blocked_by_stage_bug |
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
scripts/lf/run_lf_lora_sft.sh:56,413,445      NUM_GPUS gates:
    asym_stp*/tp2_*  -> require NUM_GPUS=2, launch SINGLE-PROCESS (no torchrun;
                        the process sees both GPUs of the pair)
    asym*/non-stp    -> require NUM_GPUS=1 (unchanged)
scripts/lf/run_lf_lora_sft.sh:1144            TP global-batch override:
    if ASYM_STP=1: PROFILE_GLOBAL_BATCH_SIZE=$((BATCH*GA))   # NOT *NUM_GPUS
scripts/lf/run_lf_lora_sft.sh (env echo)      echo ASYM_STP, ASYM_STP_TP_SIZE,
    ASYM_STP_WEIGHT_MODE, ASYM_STP_SHARED_ARENA, ASYM_STP_COORD into command.txt
    + profile.json.config
scripts/lf/profile_lora_lf_test_source.sh:868,886  accept |2 with stp backends;
    forward BOTH pool GPUs to one job (today |2 only feeds the torchrun path);
    refuse --gpus 0,2 unless ALLOW_CROSS_SUPERCHIP=1
artifact tag: append stpW<mode>_arena<0|1>_coord<0|1>_tp2 to config_label
    (:1827) so every ablation gets a distinct directory
```

Pseudocode (bash gate):

```bash
case "${BACKEND}" in
  asym_stp*|tp2_*)
    [[ "${NUM_GPUS}" == 2 ]] || die "${BACKEND} requires NUM_GPUS=2"
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
echo shows num_gpus=2, global batch 8 (NOT 16), single-process launch
path contains __gpus2__ ... stpWstream_arena1_coord1_tp2
negative guards die: asym_stp at |1; asym_cpuadamwds at |2; mode mismatch;
  --gpus 0,2 without ALLOW_CROSS_SUPERCHIP=1
P0 pace-car sweep completes and winners are recorded above; per-rank
  profile.json for the DP baselines is sane (fix aggregation if not — this is a
  known unknown of the torchrun path, resolve HERE)
```

Risks/watch: (a) torchrun-path profile.json may report rank0 only — audit and fix
in this stage, not later; (b) dataset registration loss on new roots (known
failure mode — verify dataset_info.json rows exist before runs).

## Stage I1: STPRuntime (single-process pair; isolated validation allowed)

Intended change: one process owns both GPUs; streams, P2P, collectives, NUMA.

Scope:

```text
asym_gemm/training/stp_runtime.py   NEW: class STPRuntime (singleton via env)
asym_gemm/integrations/lf.py:1149   device resolution: replace
    torch.device("cuda", current_device()) with rt.primary (dev0) when ASYM_STP=1
scripts/testing/stp_runtime_probe.py  NEW probe
```

Pseudocode:

```python
class STPRuntime:
    def __init__(self, dev_ids=(0, 1)):
        self.d = [torch.device("cuda", i) for i in dev_ids]
        for a, b in ((0, 1), (1, 0)):
            assert torch.cuda.can_device_access_peer(dev_ids[a], dev_ids[b])
        # peer access is enabled implicitly by torch on first p2p copy; force it:
        with torch.cuda.device(self.d[0]): torch.empty(1, device=self.d[1])
        self.compute = [torch.cuda.Stream(device=x) for x in self.d]
        self.copy    = [torch.cuda.Stream(device=x) for x in self.d]
        self._bind_numa_and_cores()   # membind to pair's node; SiLU pools 4-39/40-71

    def allreduce2(self, y0, y1):
        # y_i produced on compute[i]; exchange on copy streams, add on compute
        e0 = torch.cuda.Event(); e1 = torch.cuda.Event()
        e0.record(self.compute[0]); e1.record(self.compute[1])
        self.copy[0].wait_event(e1); self.copy[1].wait_event(e0)
        with torch.cuda.stream(self.copy[0]): t0 = y1.to(self.d[0], non_blocking=True)
        with torch.cuda.stream(self.copy[1]): t1 = y0.to(self.d[1], non_blocking=True)
        c0 = torch.cuda.Event(); c1 = torch.cuda.Event()
        c0.record(self.copy[0]); c1.record(self.copy[1])
        self.compute[0].wait_event(c0); self.compute[1].wait_event(c1)
        with torch.cuda.stream(self.compute[0]): y0 += t0
        with torch.cuda.stream(self.compute[1]): y1 += t1
        return y0, y1     # both directions move concurrently: full-duplex NVLink

    def bcast01(self, x0):      # dev0 -> dev1, on copy streams (Phase-A only)
    def to0_sum(self, y0, y1):  # y0 += y1.to(dev0)  (Phase-A return path)
```

Validation (isolated OK — no training semantics):

```bash
python scripts/testing/stp_runtime_probe.py --pair 0,1
python scripts/testing/gb200_topology_probe.py
```

```text
P2P copy >= 700 GB/s/dir sustained, both directions concurrently
both C2C lanes ~190 GB/s H2D concurrently reading the SAME pinned buffer
allreduce2 of a [200000, 8192] bf16 tensor (3.3 GB) < 8 ms
numa_maps shows arena on the pair's node; SiLU pools pinned 4-39 / 40-71
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
      free the original. Grouped banks [E,N,K]: same per bank dim (col: dim1
      slice contiguous per-expert only if row-major inner — repack to
      [E, N/2, K] x2 blocks; row: [E, N, K/2] x2 blocks).
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
torch.allclose reassembly: cat(shards) == original weight per module
plan assertion passes vs transformers base_model_tp_plan for all matrix models
```

Risks/watch: models with K or N not divisible by 2x alignment after split
(check q2.5-72b intermediate 29568/2=14784 %64 fail -> pad rule: pad split point
to 64-multiple boundary, uneven shards N0=ceil64(N/2), N1=N-N0 — kernels take
per-shard sizes, nothing requires equality); embedding/lm_head stay UNSHARDED.

## Stage I3: sTP Dense Linears, Phase A (fan-out only; model stays on dev0)

Intended change: streamed base GEMMs (fwd base, bwd dX) execute split across both
devices; everything else (attention math, norms, LoRA, residual, optimizer)
untouched on dev0. This is e2e-runnable WITHOUT attention surgery and proves the
lane pooling.

Scope:

```text
asym_gemm/training/frozen_linear.py
    NEW class STPFrozenLinearColFn / STPFrozenLinearRowFn (siblings of
    AsymFrozenLinearFunction:1279; fwd 1288 / bwd 1341 are the templates —
    reuse their kernel-call helpers verbatim per device)
    AsymFrozenLinear.forward:2062  dispatch: if ASYM_STP and mode==stream ->
    STP path; else existing path
asym_gemm/training/cpu_left.py     same split dispatch for grouped cpu-left calls
```

Dataflow + pseudocode (col case; y = x @ W^T, W [N,K] col-split -> W0,W1 rows):

```python
class STPFrozenLinearColFn(Function):
    @staticmethod
    def forward(ctx, x, W0, W1):                     # x on dev0 [M,K]
        x1 = rt.bcast01(x)                           # dev0->dev1, copy stream (Phase-A cost)
        with torch.cuda.stream(rt.compute[0]):
            y0 = m_grouped_bf16_asym_gemm_nt(x,  W0) # [M, N0]  existing kernel
        with torch.cuda.stream(rt.compute[1]):
            y1 = m_grouped_bf16_asym_gemm_nt(x1, W1) # [M, N1]  concurrent, own lane
        y = torch.cat([y0, y1.to(rt.d[0], non_blocking=True)], dim=1)  # copy stream + event
        ctx.save_for_backward(...)                   # save x1 handle for bwd reuse
        return y                                     # [M, N] on dev0
    @staticmethod
    def backward(ctx, g):                            # g [M,N] on dev0
        g0, g1 = g[:, :N0], rt.bcast01(g[:, N0:])    # slice halves, ship one
        with torch.cuda.stream(rt.compute[0]):
            dx0 = asym_gemm_nn(g0, W0)               # partial [M,K], streams W0 again
        with torch.cuda.stream(rt.compute[1]):
            dx1 = asym_gemm_nn(g1, W1)               # partial [M,K]
        dx = rt.to0_sum(dx0, dx1)                    # ONE P2P + add (E3)
        return dx, None, None
# row case mirrors: fwd partials -> to0_sum; bwd dX halves stay local -> cat.
```

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
# e2e target profiling
RUNS='llama3.3-70b|2 ; asym_stp_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 25000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_gb200tp_I3_s25000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0,1 --overwrite false
```

```text
parity MATCH (col, row, uneven shards)
s2048 loss within ~0.05 of |1 asym at same global workload
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
      p_i += lora_o(...) on dev0 only         # row-LoRA add-once trap (see below)
      p0,p1 = allreduce2(p0,p1)               # collective #1
      x_i = x_i + p_i                         # residual add, local
mlp:  m_i = rmsnorm(x_i)
      gu_i = m_i @^ Wgateup_i^T               # col-split [M, 2F/2]
      s_i = silu_mul(gu_i)                    # local halves (CPU or GPU per policy)
      q_i = s_i @^ Wdown_i^T                  # row-split (K=F/2 shard), partial
      q_i += lora_down on dev0 only
      q0,q1 = allreduce2(q0,q1)               # collective #2
      x_i = x_i + q_i
```

LoRA layout (final; the col-only rule is WRONG for row layers):

```text
col layers (qkv, gate_up): A [r,H] replicated (S computed on both devs, tiny);
  B [N/2,r] col-sharded; y_i += S @ B_i^T locally; bwd: dB_i local,
  dA = allreduce2(partials) (tiny), dS allreduce2 (tiny)
row layers (o, down): A [r,K/2] row-sharded: S_i = x_i_shard @ A_i^T partial ->
  S = allreduce2(S_i) ([M,r], tiny); B [H,r] replicated; the LoRA term
  B-applied is added ON DEV0's PARTIAL ONLY before the big allreduce2 —
  adding on both would double-count after the reduce. Assert-guard this.
```

Scope:

```text
asym_gemm/integrations/lf.py:1717-2415   wrap points: instantiate per-device
    module clones sharing the arena shard views; residual duplication at
    embedding output (one bcast01 per step entry, M*H bytes, negligible);
    logits: lm_head on dev0 only (unsharded), consumes x0
asym_gemm/training/frozen_linear.py      STP fns lose the Phase-A bcast (inputs
    already local); attention QKV/O wiring
asym_gemm/training/stp_layout.py         head-split plan: q 32->16/16, kv 8->4/4
    (llama3.3-70b), rotary caches materialized PER DEVICE
HF Trainer guard: model.to(device) must not migrate dev1 modules/HostWeights —
    HostWeight already refuses .to(cuda) (host_weight.py:317-335); add the same
    guard for dev1-resident buffers
```

Validation (E2E is the bar):

```text
parity: 1-batch logits |1 vs stp (max-abs-diff reported); 5-step seeded loss
  overlay within band
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
unchanged.

## Stage I5: Activation Path + Shared-Arena Dedup + Coordination

Intended change: (1) per-device fine-grained act offload on width-halves (existing
machinery, per device); (2) residual checkpoints offloaded ONCE (dedup — they are
bit-identical across devices); (3) bwd LoRA-A grad split-K so both lanes stream
disjoint X halves; (4) coord knob = membind + core split + joint prefetch budget.

Scope + pseudocode:

```text
asym_gemm/training/activation_offload.py   per-device pools (pool[dev]); byte cap
    split ASYM_EXPACT_CPU_POOL_MAX_BYTES/2 per device
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
asym_gemm/training/exp_act_offload_lora.py:231-256   dA split-K:
    # dA[r, K] = dS^T @^ X_cpu ; X_cpu [M,K] repacked K-halves (I2 rule)
    with stream0: dA0 = lora_a_grad_cpu_right(dS, X0)   # [r, K/2] lane0
    with stream1: dA1 = lora_a_grad_cpu_right(dS1, X1)  # [r, K/2] lane1
    dA = cat(dA0, dA1.to(dev0))                          # tiny
    (dS replicated: [M,r], 25 MB at s25k — bcast once, negligible)
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
bwd LoRA-A grad kernel time ~halves vs I4 (kernel-class table; it now uses both
  lanes) — this is the |1 bottleneck kernel, must show on the e2e bwd_s
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
ASYM_STP_WEIGHT_MODE dispatch in AsymFrozenLinear.forward:
  stream   -> I3/I4 path (kernels on host shard views)
  resident -> at setup: shard_view(dev).cuda(dev) once; forward = torch.matmul
              (plain GEMM, no staging, no streaming) — vanilla TP rung
  stage    -> per layer: slab[dev].copy_(shard_view(dev), non_blocking=True) into
              a double-buffered HBM slab (2 slabs/device, prefetch next layer on
              copy stream), forward = torch.matmul from slab; per-rank pinned
              buffers, arena=0, coord=0 pinned by backend gate
scripts/lf/run_dp2_pair.sh   launch two |1 jobs (GPU0/GPU1, b4, same seed),
  wait both, emit dp2_merged.json {per-rank profile paths, summed RSS, max wall}
scripts/lf/extract_lane_bw.py  parse nsys sqlite/CSV export -> lane_bw.json
  {per-GPU h2d/d2h GB/s p50/p95 + bytes, nvlink tx/rx, windows: streamed-GEMM
  spans from NVTX ranges already emitted by the profiler}
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
router: logits_i = x_i @ Wg^T on BOTH devices (Wg tiny, replicated) ->
  identical topk on both (deterministic) -> no communication, no sync
dispatch (per device, existing |1 machinery unchanged):
  my_experts = experts[dev]                      # E/2 contiguous bank block (I2)
  idx_i = flatten tokens routed to my_experts    # from the SAME topk both devs hold
  in_i = gather(x_i, idx_i)                      # local, replicated residual
  out_i = grouped_ker101(in_i, bank_shard[dev])  # ONE grouped launch, E/2 experts
  y_i = zeros[M,H]; scatter_add(y_i, idx_i, out_i * gate_prob)
combine: y0,y1 = allreduce2(y0,y1)               # union-sum, collective #2 of layer
  x_i += y_i
```

Scope: `asym_gemm/training/moe.py`, `qwen3_moe.py:2557-2568` (token-threshold
logic per device), bank repack (I2 grouped rule), routed kerNNN kernels UNCHANGED.

Validation (E2E):

```text
parity s2048 ker101 vs |1 (loss band; routed counters fire on both devices;
  route bits identical to the |1 MoE plan)
e2e P4: q3-30b-a3b 80000|8|1 gates vs |2 pace cars; expert-imbalance stats
  logged per step (max/mean tokens per device)
llama4-scout 9500|8|1 ker000 row
```

Risks/watch: expert imbalance -> one device idles (log it; EP fallback: a
starved device may stream any expert's shard directly from the arena — the
tile-stream path makes this free of ownership, add only if imbalance > 20%);
determinism of topk across devices (same dtype/math on both — assert with
DEBUG_HASH); llama4's in_out bank layout ([E,H,2I]) repack rule differs (dim
check per architecture in stp_layout).

## Memory/BW Decomposition If A Gate Fails

```text
Workload  Backend  stpTag  step_H(g0/g1)  RAM  act_H  lane0/1 GBps  nvlink  dup_factor  top_peak_owner
```

Answer: which device/class owns the peak; live operand vs saved act vs slab vs
allocator reserve; replicated-but-should-be-sharded tensor (dup_factor); lane1
idle (silent fallback); prefetch/slab budget blown; RSS on wrong NUMA node. Only
then propose a change; log the decision in gb200_aware.md Goals.

## Reporting Format

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM  (+ lane0/1, nvlink GB/s, dup_factor on gate stages)
```

Labels exactly as generated (`asym_stp_cpuadamwds | recomp-off-full-fg-ker000`,
stp tag fragment, `__gpus2__`). No row is final without the audit: pair 0,1,
global-batch parity, CPUAdamW family, loss in band, fresh artifact.
