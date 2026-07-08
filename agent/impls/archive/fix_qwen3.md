# Fix Qwen3-30B-A3B MoE `recomp-off-full-fg` Backward Latency

## Goal (ULTRA CLEAR — read this before any change)

Reduce the end-to-end step latency of the Qwen3-30B-A3B MoE fine-grained
recompute-offload path WITHOUT significantly increasing peak HBM. The target
workload is EXACTLY the same as the q3-30b-a3b rows in the current results
table so every iteration is directly comparable:

```text
model:    q3-30b-a3b  (Qwen/Qwen3-30B-A3B, 48 layers, H=2048, I=768, E=128, top_k=8)
workload: 80000|8|1   (s80000.b8, 640k tokens/step, R = 5.12M routed rows)
loss:     ligerloss1
policy:   none|false|false|false|false|false
target:   asym_cpuadamwds | recomp-off-full-fg-ker000   (primary)
          asym_cpuadamwds | recomp-off-full-fg-ker101   (secondary, HBM-lean variant)
```

### Acceptance rule (hard gate for EVERY candidate change)

A change is ACCEPTED only if, at the target workload above, measured e2e:

```text
EITHER  step_s decreases      AND step_H does not significantly increase (<= ~+5%)
OR      step_H decreases      AND step_s does not increase
```

Anything else is REJECTED, no matter how good the isolated microbenchmark
looks. Never trade a latency win for a significant HBM regression: the HBM
advantage over the SuperOffload baselines is the whole point of this path.

### Baseline scoreboard (2026-07-02, current, all PASS)

```text
Model: q3-30b-a3b
Workload   Backend           Config                     fwd_s  bwd_s   opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ----------------  -------------------------  -----  ------  -----  ------  -----  -----  ------  -----
s80000.b8  superoffload_mem  unsloth                     29.6   130.3    0.0   160.0   91.9  176.9   176.9  360.0
s80000.b8  superoffload_mem  unsloth-off                 33.2   240.7    0.0   274.0   91.9   94.4    94.4  588.5
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000   65.6   977.6    3.8  1043.3   86.0  112.9   112.9  642.0
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101   62.4  1016.6    5.3  1079.1   78.6   73.9    73.9  646.8
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker111   60.2  1179.9    3.9  1240.2   78.6   73.9    73.9  642.2
```

The pain: asym is 3.8x the step time of `superoffload_mem|unsloth-off` at the
same workload (other dense models are only ~1.5-1.7x). The prize: ker101
already beats unsloth-off HBM by 20.5 GiB (73.9 vs 94.4). We must keep that
memory posture while pulling `step_s` toward (and ideally under) the 274s of
unsloth-off.

Artifact locations for the baseline rows (verified 2026-07-02):

```text
ker000: profiling_fix_fgm_v6/.../asym_cpuadamwds__source__recomp-off-full-fg__...__moefg1__.../b8_s80000_ga1/
ker101: outputs/q3_30b_a3b_s80000_ker101_20260702_031830/.../b8_s80000_ga1/
ker111: profiling_q3_moe_routed_real_20260701T111842Z/.../b8_s80000_ga1/
SO unsloth:     profiling_q3_30b_a3b_s80000/.../superoffload_mem__source__unsloth__.../b8_s80000_ga1/
SO unsloth-off: profiling_fix_fgm/.../superoffload_mem__source__unsloth-off__.../b8_s80000_ga1/
```

## Verified Evidence (do not re-derive; artifacts + scripts below)

### E1. All offload traffic and ALL fg work happens inside `step.backward`

`recomp-off-full-fg` = outer Unsloth GC. The measured forward (65.6s) runs
`qwen3_moe_finegrained_nograd_forward` (no manager, no offload). During
backward every layer is recomputed through `_Qwen3MoeFinegrainedFunction`
(offloading forward) and then backwarded. Counters prove it:
`nograd_forward_calls=48, forward_calls=48, backward_calls=48`.
So the 977.6s backward owns: per-layer recompute-fwd + fg-bwd + attention
recompute + attention bwd + all activation offload traffic.

### E2. Transfer volume is huge but transfer bandwidth is NOT the root cause

From `source_profile.json.activation_offload` of the ker000 run:

```text
total_activation_transfer_bytes = 9.93 TB per step (5.09 TB D2H + 4.84 TB H2D)
by tag (D2H side):  moe.X 1006 GB (EXPANDED topk-duplicated X, [R=5.12M, 2048] x48)
                    moe.gate/up/act/dgate/dup  377 GB each
                    saved.float32.8x80000x32x128  1006 GB  (q_norm fp32 saves, x2/layer)
                    saved.float32.8x80000x4x128    126 GB  (k_norm fp32 saves)
                    saved.bfloat16.*               566 GB  (flash-attn q/k/v/o)
                    o_proj.U 252 GB, q_proj.U 126 GB, S_* 110 GB
```

Microbench on this box (scripts kept at /tmp/q3fix/microbench_host_ops.py):

```text
D2H pinned 193 GB/s | H2D pinned 210 GB/s | D2H pageable 26.8 GB/s | H2D pageable 204 GB/s
pinned alloc 21GB: 1.25s (~17 GB/s) | zero_ 21GB: 0.08s | grouped cpu copy 21GB: 0.22s
```

9.93 TB at pinned rates is ~50s if not serialized — real but secondary.

### E3. The step is serialized on ONE host thread for ~930s

`utilization_metrics.cpu` timeseries: from t≈230s to t≈1160s the process sits
at EXACTLY 0.69-0.76% of 144 cores = one core busy, everything else idle.
This is host-side serialization (driver pinning/allocs, synchronizes, python
glue), not CPU-GEMM saturation and not disk (io_samples flat).

### E4. Isolated fg layer at production shape = 4.08s → MoE math is only ~20% of backward

`/tmp/q3fix/isolate_fg_layer.py` (kept; runs one AsymQwen3Experts layer,
R=5.12M, H=2048, I=768, E=128, r=64, fg path, ker000 envs, sync-timed ranges):

```text
fwd+bwd per layer:            4.08 s   (x48 = 196s vs 977.6s e2e backward)
  lora.a_fwd_cpu gate/up/down 1.58 s   <- of which host.cpu_left_pad = 1.14 s
  lora.a_grad_cpu (dA)        0.66 s
  base AsymGEMMs (6 GEMMs)    0.50 s   <- fast; AsymGEMM itself is NOT the problem
  silu fwd+bwd staging        0.48 s
  transfers (offload+stage)   ~0.7 s   (~200 GB/s, pinned, on compute stream)
peak HBM: 110.3 GiB (isolated layer alone!)
```

Key isolated findings:
- `_pad_cpu_left_grouped_input_for_asym` (cpu_left.py) runs on EVERY
  cpu-left LoRA-A forward call (gate, up, down = 3x/layer): allocates a FRESH
  pinned CPU buffer (X-sized 19.5 GiB twice, act-sized 7.3 GiB once),
  `zero_()`s it and copies group-by-group. Pure overhead ~1.14 s/layer
  (~55s per step) + pinned-alloc churn. X is padded TWICE (gate and up pad
  the same handle independently).
- In isolation the 32 GiB CPU buffer pool works (pool hits, alloc time ~0);
  in e2e the pool is GLOBAL and shared with the attention offload + outer GC
  saves (~108 GB/layer of returned buffers vs 32 GiB cap
  `ASYM_EXPACT_CPU_POOL_MAX_BYTES` default) so most per-layer buffers are
  freed and re-cudaHostAlloc'd every layer; e2e RSS grows +212 GB during
  backward. Fresh pinned alloc measured at ~17 GB/s, single-threaded (E3).

### E5. Remaining gap is an e2e-only emergent cost — must be measured, not guessed

Isolated-sum estimate per layer (MoE 4.1s + flash-attn recompute+bwd ~1.6-2s
+ attn asym projections ~1s + glue) ≈ 6-7s/layer ≈ 300-340s. Observed 977s.
The ~600s residual is e2e-only (pinned-pool churn across managers, outer GC
save_on_cpu behavior incl. 1.1 TB fp32 q/k-norm saves, GPU stage-buffer
map/unmap under expandable_segments, allocator syncs near the 147 GiB
reserved ceiling, autograd-thread serialization). Stage B nails it with nsys
before any code beyond v1 is written.

## Root-Cause Summary

```text
R1 (proven, isolated): cpu_left pad = fresh pinned alloc + zero + copy, 3x/layer, X padded twice.
R2 (proven mechanism, e2e magnitude TBC): 32 GiB global CPU pool cap vs ~108 GB/layer working set
    -> cudaHostAlloc/free churn on one thread, every layer, every step.
R3 (proven volume, cost TBC): avoidable transfer volume — expanded moe.X (8x the compact X),
    act offloaded although recomputable from gate/up, dgate/dup CPU round-trip,
    fp32 q/k-norm saves (2x bf16 size).
R4 (open until nsys): everything serializes on one host thread + the compute stream;
    no side-stream overlap of copies with GEMMs; possible allocator sync storms.
```

## Versioned Fix Plan

### v1 — kill R1 + R2 (host-alloc churn) [zero math change, no HBM impact]

1. `cpu_left.py`: cache the padded input per CPU handle so gate+up reuse one
   padded X; allocate the padded buffer through the activation-offload CPU
   pool (reuse across layers/steps) instead of fresh `torch.empty(pin=True)`;
   skip `zero_()` of regions that are fully overwritten (only zero the pad
   tails). Fallback to old behavior if pool unavailable.
2. Raise the CPU pool cap for this path: export
   `ASYM_EXPACT_CPU_POOL_MAX_BYTES` (>= 192 GiB) in the `recomp-off-full-fg`
   stage env of `scripts/lf/run_lf_lora_sft.sh` (env-only config, overridable).
   CPU RAM budget check: RSS peak was 674 GB with churn; retaining ~150-200 GB
   pooled REPLACES the churned allocations, machine budget 958 GB, runs are
   sequential — OK, but verify RAM column in the A/B.
3. Validate: isolated layer (expect ~4.1s -> ~2.9-3.1s, identical loss);
   unit tests for cpu_left/fg path; then e2e gate (Stage A/C below).

Expected e2e effect: -55s (pad) - O(100-250s) (churn) on bwd_s; step_H flat.

### v2 — cut avoidable transfer volume (R3) [HBM-shape aware; each item gated separately]

1. Offload COMPACT X ([M=640k,H] = 2.6 GB) instead of expanded ([R=5.12M,H] =
   21 GB); re-expand on GPU by index_select at stage time for base GEMMs; the
   cpu-left LoRA kernels need the expanded rows -> either feed compact X +
   indices to a new kernel variant, or re-expand once per layer into a pooled
   pinned buffer. Saves ~2 TB/step of D2H+H2D.
2. Do not offload `moe.act`: recompute act = silu(gate)*up on GPU inside
   backward from the already-staged gate/up (they are staged anyway for
   silu-bwd); feed down-LoRA dA from a GPU-resident act chunk or a pooled
   pinned copy. Saves ~755 GB/step.
3. Keep dgate/dup in HBM when headroom allows (they are consumed within the
   same layer backward) instead of CPU round-trip. Saves ~1.5 TB/step.
   HBM cost +7.9-15.7 GB DURING backward of a layer — must verify step_H.
4. Save q/k-norm saved tensors in bf16 (or recompute the norm in backward)
   inside the outer GC path. Saves ~0.5-1.1 TB/step.

### v3 — overlap (R4): side-stream D2H/H2D with events; prefetch next consumer's
stage-in while the current GEMM runs (Megatron-style 2-buffer). Only after v1/v2,
guided by the nsys timeline, since overlap changes stage-buffer lifetimes (HBM).

## Execution Protocol (Stages)

Evidence discipline: run rows ONE AT A TIME, fresh OUTPUT_ROOT per stage,
audit `command.txt` + `profile.json.config` (recomp_label, moefg=1, route bits)
before believing any number. Conclusion labels: validated /
blocked_by_stage_bug / inconclusive_wrong_config / inconclusive_partial_profile.

### Stage A — isolated + unit gate (fast, per code change)

```bash
CUDA_VISIBLE_DEVICES=<free> .venv-fa4/bin/python /tmp/q3fix/isolate_fg_layer.py   # per-phase table
.venv-fa4/bin/python -m pytest tests -k "finegrained or cpu_left" -x -q
```

Pass: per-layer time drops, loss bit-identical vs pre-change, no new HBM peak.

### Stage B — e2e attribution run (small, once per major iteration)

s20000.b8 keeps every mechanism (same 48 layers, same managers) at ~1/16 the
attention cost; use it for nsys + A/B of host-side fixes:

```bash
RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen3_s20000_<tag> MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus <free> --overwrite false
```

plus the SO reference at the same size when needed. If the residual e2e gap
(bwd_s vs isolated-sum) is still >2x after v1, capture nsys (PROFILERS=both
wrapper or nsys launch) and attribute: NVTX fg ranges vs cudaHostAlloc/Free
vs memcpy engines vs kernel gaps. No v2 implementation before this table exists.

### Stage C — target scoreboard (the ONLY numbers that accept/reject a change)

```bash
RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen3_s80000_<tag> MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus <free> --overwrite false
```

Compare against the baseline table rows above (identical workload/config).
Report exactly: `fwd_s bwd_s opt_s step_s fwd_H bwd_H step_H RAM`.
Accept/reject per the Acceptance rule. On accept, update the scoreboard in
this doc with a `vN` row and keep iterating (v2 items next) until
`step_s` is at least under 2x `superoffload_mem|unsloth-off` (stretch: parity,
~274s) with step_H <= baseline asym rows.

## Iteration Log

```text
v0  2026-07-02  baseline verified; root causes R1-R4 established (this doc).
v1  2026-07-02  implemented:
    v1a ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU (qwen3_moe_finegrained.py): recompute-forward
        LoRA-A on GPU — gate/up S from in-HBM `packed`, down S from the GPU act tensor
        inside the activation section (zero added lifetime). Backward dA unchanged
        (cpu_right). Code default OFF; wrappers default it ON only for
        qwen3-moe + recomp-off-full-fg. Kills R1 (cpu-left pad) entirely on this path.
    v1b ASYM_EXPACT_CPU_POOL_MAX_BYTES=192GiB in the same wrapper branch (R2).
    Stage A results: isolated layer 4.08s -> 2.25s (-45%), peakHBM identical 110.3 GiB;
    numerics loss diff 1.9e-9, worst grad rel 8.2e-3 (bf16 kernel swap, abs ~2e-10);
    tests: 136 passed, 1 PRE-EXISTING failure (vision-attention, fails on clean tree too).
    Bench scripts persisted: scripts/testing/isolate_qwen3_fg_layer_timing.py,
    check_qwen3_fg_gpu_lora_numeric.py, microbench_host_transfer_ops.py.
    Stage B (s20000.b8 A/B, same prepared dataset, single measured step):
      control  (v1 off, pool 32GiB):  fwd 15.7  bwd 120.9  step 136.7  peakH 28.5  rss 333.7
      treated  (v1 on,  pool 192GiB): fwd 22.4  bwd 102.1  step 124.6  peakH 28.5  rss 323.8
      bwd -15.6%, step -8.9%, peakH identical, loss 1.7734 vs 1.7750 (bf16 kernel swap).
      bwd delta matches isolated prediction (~0.4 s/layer at R=1.28M). fwd +6.7s is
      single-step noise (identical-config s80000 baselines historically vary ~10%).
      Pool churn is negligible at s20000 (cpu live 12.5 GiB < 32 GiB) — v1b's effect
      is expected only at s80000. GOTCHA recorded: controls must run with
      PREPARE_DATASETS=true or the wrapper silently falls back to the raw
      asym_long_sft_smoke dataset (loss 2.14, peak 6.5 GiB = wrong workload).
    Stage C (s80000.b8 ker000 scoreboard, single measured step, GPU 3, .venv):
      baseline: fwd 65.6  bwd 977.6  opt 3.8  step 1043.3  peakH 112.93  RAM 642  loss 1.6904
      v1:       fwd 62.8  bwd 315.3  opt 3.3  step  378.3  peakH 112.93  RAM 593  loss 1.6874
      step -63.7% (2.76x), bwd -67.7%, peak HBM byte-identical, RSS -49 GB. ACCEPTED.
      Artifact: profiling_fix_qwen3_s80000_v1/.../recomp-off-full-fg-ker000/.../b8_s80000_ga1
      Attribution: at s80000 v1b (pool churn removal) dominates (~550s) — per-layer
      pinned set ~108 GB vs 32 GiB cap meant cudaHostAlloc/free of most buffers every
      layer under a fragmented 600+GB RSS, far slower than clean-process microbench.
      v1a contributes ~88s (pad + cpu-left fwd kernels).
      Now 1.38x superoffload_mem|unsloth-off (274.0s) — under the 2x gate; stretch parity.
```

```text
v2  2026-07-02  implemented (after nsys attribution at s20000/v1: bwd window 59.9s =
    kernels 33.7s + memcpy 15.8s on one stream + idle 10.4s; #1 kernel = cpu-right dA
    8.4s; scatter index_add 5.1s; copies healthy at 178-211 GB/s; cudaHostAlloc down
    to 1.6s -> volume + kernel replacement, not bandwidth):
    v2a ASYMM_QWEN3_MOE_FG_DA_GPU: offload COMPACT X ([M,H], 8x smaller than the
        topk-expanded [R,H]); gate/up dA on GPU via _grouped_lora_weight_grads_torch
        (left=dS [M,r], right=re-gathered X rows [M,H] -> [E,r,H]), chunked along
        expert-group blocks (8 chunks) to bound the gather transient. Requires v1a;
        auto-disabled for down_scatter block path. down dA stays cpu_right.
    v2b ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM: dgate/dup stay in HBM across the layer
        backward (consumed within it; replaces same-size stage buffers -> ~net-zero
        live HBM, kills 4x[R,I] CPU round trips per layer).
    Stage A: isolated layer 2.25s -> 1.50s (-33% vs v1, -63% vs v0), peakHBM
    110.3 -> 103.0 GiB (LOWER); moe.X offload 108.7 -> 13.6 ms; numerics unchanged
    (loss diff 1.9e-9, worst grad rel 8.2e-3); tests 136 pass (same pre-existing
    vision failure), fg subset passes with all flags ON. Wrappers default both ON
    in the qwen3-moe full-fg branch only.
```

### Scoreboard (target workload s80000.b8, single measured step)

```text
Workload   Backend           Config                     fwd_s  bwd_s   opt_s  step_s  step_H    RAM
s80000.b8  superoffload_mem  unsloth                     29.6   130.3    0.0   160.0   176.9  360.0
s80000.b8  superoffload_mem  unsloth-off                 33.2   240.7    0.0   274.0    94.4  588.5
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000   62.9   252.1    3.3   315.1   105.6  557.4   <- v2 ACCEPTED
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker101   61.6   261.0    3.3   322.8    78.6  557.3   <- v2 ACCEPTED
s80000.b8  asym_cpuadamwds   recomp-off-full-fg-ker000   62.8   315.3    3.3   378.3   112.9  593.4   <- v1 ACCEPTED
s80000.b8  (old ker000 base) recomp-off-full-fg-ker000   65.6   977.6    3.8  1043.3   112.9  642.0
s80000.b8  (old ker101 base) recomp-off-full-fg-ker101   62.4  1016.6    5.3  1079.1    73.9  646.8
```

ker101+v2: step -70.1% vs its baseline; peak 78.6 GiB = 15.8 GiB BELOW
superoffload_mem|unsloth-off at 1.18x its step time. The +4.7 GiB vs the old
ker101 (73.9) comes from v2b dgrads retention; set
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=0 to recover the strict-HBM posture while
keeping the v1+v2a wins. Artifacts: profiling_fix_qwen3_s80000_v2/,
profiling_fix_qwen3_s80000_v2_ker101/, s20000 A/Bs in
profiling_fix_qwen3_s20000_{ctl,v1,v2}/, nsys in
profiling_fix_qwen3_s20000_v1_nsys/.

v2 e2e: step -69.8% vs baseline, bwd within 11.4s of unsloth-off, peak HBM DOWN
7.3 GiB, loss 1.6862 (baseline 1.6904 — recompute/kernel-swap band). s20000 A/B
(v1 -> v2): bwd 102.1 -> 42.9 (-58%), step 124.6 -> 58.7, peakH 28.5 -> 26.7.
Remaining gap to unsloth-off is mostly forward (62.9 vs 33.2 — AsymGEMM C2C
weight streaming, the price of keeping ~60 GB of base weights out of HBM).

Remaining v3 candidates (by nsys evidence): routed-kernel scatters (ker101, replaces
index_add ~20s at s80k), fp32 q/k-norm saves -> bf16 (~6-10s), act round-trip
elimination (~5s), side-stream copy/compute overlap (~15-30s), fwd layer-input saves
through the pooled allocator (~5-10s of fwd).
