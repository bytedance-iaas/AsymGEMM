# Fix Fine-Grained Offload: Qwen3.5-27B Dense `ker000`

## Goal

Train dense Qwen3.5-27B LoRA-SFT with the dense fine-grained recompute-offload path and
compare it against the matching SuperOffload baselines at the real target workload:

```text
[q3.5-27b]="Qwen/Qwen3.5-27B"
workload: 50000|8|1
loss: ligerloss1
policy tuple: none|false|false|false|false|false
target artifact label: recomp-off-full-fg-ker000
```

The target is dense, not MoE. Therefore `recomp-off-full-fg` must resolve to the
generated artifact label `recomp-off-full-fg-ker000`, with all Qwen3 MoE routed-kernel
bits off:

```text
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE=000
```

Put `recomp-off-full-fg-ker000` directly in the `RUNS` recompute field. The scripts
canonicalize it internally to `recomp-off-full-fg` for stage setup, but the suffix is
the source of truth for route-bit ownership and artifact labels:

```text
q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

The script must generate the `-ker000` label in artifact paths, `RUN_ID`, echo output,
and `profile.json.config.recomp_label`.

## Required Baselines

The main apples-to-apples scoreboard for `asym_cpuadamwds` is the CPUAdamW /
CPU-optimizer-offload family:

```text
q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

If the no-CPUAdamW backend `asym|recomp-off-full-fg` is tested, it must be compared to
the no-CPUAdamW baselines, not the CPUAdamW SuperOffload rows:

```text
q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; asym|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

The required reported table must include at least:

```text
Model: qwen3.5-27b    LoRA: r64/a16/d0.00
Workload   Backend                         Config                    fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  ------------------------------  ------------------------  ---------------------------  ------------------------  -----
s50000.b8  superoffload_mem                unsloth       [lg+ sd-]
s50000.b8  superoffload_mem                unsloth-off   [lg+ sd-]
s50000.b8  asym_cpuadamwds                 recomp-off-full-fg-ker000  [lg+ sd-]
```

## Why This Is A Separate Dense Plan

This plan is not the Qwen3 MoE routed-kernel plan. Do not copy the Qwen3-30B-A3B
`ker101` default or any expert routed kernels into this work.

Expected dense behavior:

```text
outer Unsloth GC
+ outer save_on_cpu / unsloth-off recompute saved tensors
+ AsymGEMM CPU-resident frozen/base weights
+ dense fine-grained MLP placement
+ attention activation placement where applicable
+ no Qwen3 MoE routed kernels
+ no block-expert or per-expert path
+ no chunked MLP
+ no outer_hbm diagnostic setting
```

The final target path must prove these config facts:

```text
config.recomp_label = recomp-off-full-fg-ker000
config.recomp_off_stage = full-fg
config.use_unsloth_gc = true
config.unsloth_gc_recompute_save_on_cpu = true
config.asymm_dense_mlp_finegrained_offload = 1
config.asymm_dense_mlp_surgical_offload = 0
config.asymm_expert_silu_bwd_gpu = 0
config.asymm_mlp_recompute_chunk = 0
config.unsloth_gc_outer_hbm_every_n = 0
config.asymm_qwen3_moe_finegrained_offload = 0
config.asymm_qwen3_moe_route_kernel_code = 000
```

The setup report must show dense wrapping, not MoE wrapping:

```text
dense_mlp_finegrained_offload_enabled=true
dense_mlp_finegrained_offload_wrapped > 0
dense_mlp_act_offload_wrapped = 0
qwen35_moes_wrapped = 0
```

If Qwen3.5-27B uses a Qwen3.5-specific attention/linear-attention module, do not assume
the exact attention wrapper from Qwen3-32B. Inspect the setup report and runtime
counters. The dense MLP path is the required `ker000` proof; attention counters are
allowed only if `recomp-off-full-fg` explicitly enabled the matching attention wrapper.

## Zero-Centered RMSNorm Caveat (root cause of the historical loss~13 runs)

Every asym q3.5-27b artifact before 2026-07-03 trains at loss ~13.0–13.3 with sane
grad_norm and correct memory. Root cause (fixed 2026-07-03): `Qwen3_5RMSNorm` is
zero-centered — `y = normalize(x) * (1 + w)` (`modeling_qwen3_5.py:724`) — but
`AsymFrozenRMSNorm` keyed the `(1 + w)` handling on the exact class name
`Qwen3_5MoeRMSNorm` only (`asym_gemm/training/offload.py:402`). The dense class fell
through to plain `w * normalize(x)`; with near-zero zero-centered checkpoint weights
every normed output collapsed to ~0, so token mixers and MLPs contributed nothing and
the residual stream carried only embeddings. Fix = include `Qwen3_5RMSNorm` (and
`Qwen3_5RMSNormGated`) in the class-name sets. Validated: s2048 loss 13.27 → 1.715
(baseline 1.718), grad_norm 0.29; probes in
`scripts/testing/qwen35_dense_integration_probe.py` (pre-wrap vs wrapped forward,
all-zeros → MATCH) and `scripts/testing/qwen35_dense_shapes_probe.py` (per-shape
kernel parity — clean before and after, which is what exonerated the kernels).
Liger, dense-fg, attention wrappers and the recomp-off scaffold were each bisected
and exonerated (`profiling_dense27b_bisect_*`). If a future qwen3.5-family class is
added, check its norm convention before trusting any loss.

## Stage 0: Alias And Label Ownership

Required script alias:

```bash
[q3.5-27b]="Qwen/Qwen3.5-27B"
```

Both profiling wrappers must recognize it:

```text
scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
```

Do not run the final source-profile commands unless both wrappers contain this alias
and dry-run resolves it to `MODEL_NAME_OR_PATH=Qwen/Qwen3.5-27B`.

Dry-run proof:

```bash
RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_fix_qwen35_dryrun RUNS_LOG=profiling_fix_qwen35_dryrun/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria:

```text
echo contains: recompute_label=recomp-off-full-fg-ker000
artifact path contains: __recomp-off-full-fg-ker000__
route tag contains: route000_lora0_accfp32
command.txt contains: RUN_ID=...recomp-off-full-fg-ker000...
command.txt contains: ASYM_GEMM_LF_CONFIG_RECOMP_LABEL=recomp-off-full-fg-ker000
command.txt contains: ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE=000
```

Negative guard proof:

```bash
RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_fix_qwen35_dryrun_bad RUNS_LOG=profiling_fix_qwen35_dryrun_bad/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria for the negative guard:

```text
script exits nonzero before writing a valid training command
error states dense model Qwen/Qwen3.5-27B must use recomp-off-full-fg-ker000
```

Do not continue if the positive dry run says `ker101`, if any routed MoE bit is 1, if
the negative guard accepts `ker101`, or if the script resolves the model as a MoE
shorthand.

## Evidence Discipline

Run experiments one at a time. Do not run baselines and targets in parallel while
validating this path. Use a new `OUTPUT_ROOT` for each stage or a unique `RUN_NAME` so
artifacts are never overwritten.

Before each run, write down:

```text
expected model: Qwen/Qwen3.5-27B
expected dense/MoE status: dense, qwen35_moes_wrapped=0
expected backend:
expected recompute input:
expected artifact recompute label:
expected CPUAdamW/optimizer-offload family:
expected dense wrapper count:
expected attention wrapper count:
expected route kernel code: 000
expected comparison baseline:
expected likely failure mode:
```

After each run, inspect:

```text
command.txt
train.log
profile.json.config
profile.json partial/completed fields
source_profile.json step_samples
memory_breakdown_summary.json
memory_breakdown/live activation details
setup report / runtime counters
artifact path labels
```

Treat the result as inconclusive if:

```text
profile is partial and does not identify a stage bug
path label and profile config disagree
recomp_label is not recomp-off-full-fg-ker000
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false
ASYMM_EXPERT_SILU_BWD_GPU=1
ASYMM_MLP_RECOMPUTE_CHUNK != 0
UNSLOTH_GC_OUTER_HBM_EVERY_N != 0
any Qwen3 MoE routed bit is 1
qwen35_moes_wrapped > 0
dense_mlp_finegrained_offload_wrapped == 0
wrong CPUAdamW/SuperOffload family was compared
stale artifact was reused
```

Conclusion labels:

```text
validated
blocked_by_stage_bug
inconclusive_wrong_config
inconclusive_partial_profile
inconclusive_stale_artifact
inconclusive_unexpected_path
```

Do not advance to the next stage on an inconclusive result.

## Stage 1: Small Config Gate

Purpose: prove the dense `ker000` config is real and the model does not trigger MoE
routed paths.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Pass criteria:

```text
both rows complete a backward step
target artifact label is recomp-off-full-fg-ker000
target dense fine-grained wrapper count > 0
target qwen35_moes_wrapped = 0
target route kernel code = 000
target no chunked MLP and no outer_hbm
baseline is not accidentally no-CPUAdamW
```

## Stage 2: Medium Memory-Shape Gate

Purpose: make sure the dense path scales before the real target. This is not the final
scoreboard.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Expected memory shape:

```text
superoffload_mem|unsloth keeps recompute intermediates in HBM during layer backward
superoffload_mem|unsloth-off offloads saved tensors during recompute backward
asym_cpuadamwds|recomp-off-full-fg-ker000 should have dense fine-grained activations
  lower than or comparable to unsloth-off, plus CPU-resident base weights
```

If target HBM is higher than `unsloth-off`, inspect the peak live activation details
before making a conclusion. Identify whether the peak is:

```text
dense MLP live operands
attention / linear-attention saved tensors
LoRA transient or trainable-weight gather
optimizer step / CPUAdamW transfer
wrong no-grad forward path
wrong saved-tensor hook ownership
```

## Stage 3: Bottleneck Gate At s30000

Purpose: prove the mechanism at a workload large enough to expose the real bottleneck
without jumping straight to the final s50000 failure surface.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Pass criteria:

```text
target completes
target step_H is below superoffload_mem|unsloth-off at the same workload
target is also compared to superoffload_mem|unsloth
CPU RSS is within machine budget
runtime counters prove dense fine-grained path fired
memory breakdown peak has no unexpected MoE routed tensors
```

If s30000 does not meet this, do not run s50000 as a scoreboard. First run the memory
decomposition described below and fix the responsible component.

## Stage 4: Final s50000 Scoreboard

Only run this after Stage 3 is validated.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Success criteria:

```text
target completes with loss INSIDE THE BASELINE BAND (within ~0.05 of the
  superoffload rows at the same workload; finite-but-wrong ~13.x = the
  zero-centered-RMSNorm signature, see the norm caveat below)
target artifact label is recomp-off-full-fg-ker000
target peak HBM is below superoffload_mem|unsloth-off at s50000.b8.ga1
target is also reported against superoffload_mem|unsloth
target CPU RSS is within host budget
dense fine-grained counters fire; MoE routed counters stay zero
memory snapshot/live activation details identify no hidden wrong-path peak
```

If the target fails but a baseline also fails, report both failures explicitly. Do not
claim success from "baseline OOM" unless the target completed and the config audit is
clean.

## Optional No-CPUAdamW Cross-Check

Only use this if evaluating `asym|recomp-off-full-fg` without CPUAdamW:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_fix_qwen35_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Do not compare `asym|...` against `superoffload_mem|...` as the only scoreboard. That
mixes optimizer-offload families.

## Memory Decomposition If The Target Is Higher Than Baseline

Produce a concise table for every failed or surprising stage:

```text
Workload   Backend                         Config                      step_H  RAM   act_H  saved_GPU  saved_CPU  top_peak_owner
---------  ------------------------------  --------------------------  ------  ----  -----  ---------  ---------  --------------
s30000.b8  superoffload_mem                unsloth-off                 ...
s30000.b8  asym_cpuadamwds                 recomp-off-full-fg-ker000   ...
```

Then inspect the detailed live tensors and answer:

```text
which module owns the peak?
is it dense MLP, attention/linear-attention, LoRA, optimizer, or allocator reserve?
which exact tensor shape dominates?
is the tensor a live GEMM operand or a saved activation?
is it expected to be CPU-managed by the fine-grained path?
did any raw tensor get stored on ctx outside ActivationOffloadManager?
did the no-grad original forward accidentally save internal tensors?
```

Only after this audit should an implementation change be proposed.

## Reporting Format

The final response/table must be plain text and include these metrics only:

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM
```

Use the generated artifact labels in the backend/config columns:

```text
asym_cpuadamwds    recomp-off-full-fg-ker000
superoffload_mem   unsloth
superoffload_mem   unsloth-off
```

Do not report a run as final unless the artifact audit proves it is `q3.5-27b`,
dense, `ligerloss1`, `50000|8|1`, and `recomp-off-full-fg-ker000`.
