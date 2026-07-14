# Fix Fine-Grained Offload: Qwen3.5-27B Dense `ker000` — Beat BOTH SuperOffload Baselines (FA4, CPUAdamW family)

## Goal

Train dense Qwen3.5-27B LoRA-SFT with the dense fine-grained recompute-offload path and
**beat BOTH same-family SuperOffload baselines on peak HBM** at the real target workload:

```text
[q3.5-27b]="Qwen/Qwen3.5-27B"          # dense, 64 layers
workload: 50000|8|1
loss: ligerloss1
policy tuple: none|false|false|false|false|false
attention runtime: FA4 (mandatory for ALL qwen3.5 — see FA4 Runtime Policy)
target artifact label: recomp-off-full-fg-ker000-ceil0000-ohbm0
```

The scoreboard rows (CPUAdamW optimizer family, one family only):

```text
TARGET   q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
BAR      q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
BAR      q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

**SUCCESS = the target completes with loss in band AND its `step_H` is NOTICEABLY
LOWER THAN BOTH BARS at `s50000.b8.ga1` — not just one of them:**

```text
step_H(asym_cpuadamwds) < step_H(superoffload_mem|unsloth-off)   # the BINDING bar (strongest baseline)
step_H(asym_cpuadamwds) < step_H(superoffload_mem|unsloth)
"noticeably" = at least ~5% below EACH bar (beyond the ~±3% run-to-run noise);
report the % margin against BOTH bars explicitly in the final table.
```

`unsloth-off` (unsloth GC + saved-boundary CPU offload) is the strongest completing
HBM baseline and therefore the binding bar: beating only `unsloth` does NOT close
this plan, and a tie with `unsloth-off` is a FAIL.

The target is dense, not MoE. `recomp-off-full-fg` must resolve with all Qwen3 MoE
routed-kernel bits off:

```text
ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER=0
ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER=0
ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE=000
```

Put `recomp-off-full-fg-ker000` directly in the `RUNS` recompute field. The scripts
canonicalize it internally to `recomp-off-full-fg` for stage setup; the generated
artifact label is the tagged form. Current label convention: asym_* run/config dirs
ALWAYS carry the `-ker<XYZ>-ceil<NNNN>-ohbm<N>` triple (ker000 = no routed-kernel
override; ceil0000 = no per-run CPU activation budget, zero-padded to 4 digits — a
nonzero `-ceil<N>` is only legal on the asym NVMe backends and is out of scope here;
ohbm0 = all outer Unsloth checkpoint roots to CPU, the required setting). Non-asym
Unsloth-GC dirs carry `-ohbm<N>` only (e.g. `unsloth-off-ohbm0`), never `-ceil`:

```text
RUNS token:            recomp-off-full-fg-ker000
target artifact label: recomp-off-full-fg-ker000-ceil0000-ohbm0
bar artifact labels:   unsloth-ohbm0 / unsloth-off-ohbm0
```

The script must generate the tagged label in artifact paths, `RUN_ID`, echo output,
and `ASYM_GEMM_LF_CONFIG_RECOMP_LABEL`.

## Required Baselines

This plan lives entirely in the **CPUAdamW / CPU-optimizer-offload family**:
`asym_cpuadamwds` (DeepSpeed CPUAdamW + grad/weight offload) vs `superoffload_mem`.
Never mix families on one scoreboard:

```text
q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

The no-CPUAdamW family (`asym` vs `superoffload_mem_nocpuadamw`) is a SEPARATE,
optional cross-check (see below). Comparing `asym_cpuadamwds|...` against
`superoffload_mem_nocpuadamw|...` (or `asym|...` against `superoffload_mem|...`) as
the scoreboard is an automatic `inconclusive_wrong_config`.

The required reported table must include at least (both bars + margins):

```text
Model: qwen3.5-27b    LoRA: r64/a16/d0.00
Workload   Backend            Config                                        steps  fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM   loss  grad_norm
---------  -----------------  --------------------------------------------  -----  ---------------------------  --------------------  -----  ----------------
s50000.b8  superoffload_mem   unsloth-ohbm0                [lg+ sd-] fa4
s50000.b8  superoffload_mem   unsloth-off-ohbm0            [lg+ sd-] fa4
s50000.b8  asym_cpuadamwds    recomp-off-full-fg-ker000-ceil0000-ohbm0  [lg+ sd-] fa4
margin:    target vs unsloth-off = -X.X% step_H (binding)   target vs unsloth = -Y.Y% step_H
```

## Why This Is A Separate Dense Plan

This plan is not the Qwen3 MoE routed-kernel plan and not the MoE 35B plan. Do not
copy the `ker101` default, expert routed kernels, or MoE fg wrapping into this work.

Expected dense behavior:

```text
outer Unsloth GC
+ outer save_on_cpu / unsloth-off recompute saved tensors
+ AsymGEMM CPU-resident frozen/base weights
+ DeepSpeed CPUAdamW with grad+weight offload (the family under test)
+ dense fine-grained MLP placement
+ attention activation placement where applicable
+ no Qwen3 MoE routed kernels
+ no block-expert or per-expert path
+ no chunked MLP
+ outer_hbm stays 0 (shown as the -ohbm0 tag; any -ohbm<N>>0 is diagnostic-only)
+ no NVMe roles, no activation-budget ceiling (ceil0000)
```

The final target path must prove these config facts:

```text
config.recomp_label = recomp-off-full-fg-ker000-ceil0000-ohbm0
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
command.txt: BACKEND=asym_cpuadamwds, USE_ASYM_CPU_ADAMW=true,
             ASYM_CPU_ADAMW_BACKEND=deepspeed, ASYM_CPU_ADAMW_GRAD_OFFLOAD=true,
             ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true (the CPUAdamW family under test)
artifact dir: carries the __gradofftrue__weightofftrue suffix (CPUAdamW-family label)
```

The setup report must show dense wrapping, not MoE wrapping:

```text
dense_mlp_finegrained_offload_enabled=true
dense_mlp_finegrained_offload_wrapped > 0     # expect 64 on q3.5-27b
dense_mlp_act_offload_wrapped = 0
qwen35_moes_wrapped = 0
```

If Qwen3.5-27B uses a Qwen3.5-specific attention module, do not assume the exact
attention wrapper from Qwen3-32B. Inspect the setup report and runtime counters. The
dense MLP path is the required `ker000` proof; attention counters are allowed only if
`recomp-off-full-fg` explicitly enabled the matching attention wrapper.

## FA4 Runtime Policy (mandatory — supersedes all pre-FA4 dense evidence)

ALL qwen3.5 results are valid ONLY on the FA4 attention runtime. The profile scripts
already default this (`ASYM_QWEN35_FA4_AUTO=1` auto-selects `LF_FA4_DIR`, `.venv-fa4`,
and `FLASH_ATTN=fa4` for any qwen3.5 model unless explicitly overridden, and
`validate_current_fa4_runtime` dies if FA4 is not importable — bootstrap with
`scripts/lf/bootstrap_lf_venv_fa4.sh` if it does).

Audit consequences:

```text
every final artifact path must contain __attnfa4__
command.txt must show FLASH_ATTN=fa4 and the FA4-capable ENV_DIR
do NOT set FLASH_ATTN/LF_DIR/ENV_DIR in the env for these runs — let the auto-default resolve
any qwen3.5 run without attnfa4 in the path = inconclusive_wrong_config (history only)
```

This retroactively demotes ALL existing dense artifacts: the 2026-07-03 canonical-
attention gate (`profiling_results/profiling_dense27b_fixed_s30000_*`: asym_cpuadamwds 71,228.31 MiB,
loss 1.143, −38.7% vs superoffload_mem|unsloth 116,170.70 MiB — same CPUAdamW family
as this plan) proved the dense fg MECHANISM and the norm fix, but it is non-FA4 and
it was never compared against `unsloth-off`. It is background evidence, not this
scoreboard. There are currently ZERO fa4 dense-27b artifacts of any kind: every stage
below runs fresh.

Lesson imported from the MoE 35B plan: FA4 shrinks attention activations for BOTH
sides, so the baselines get cheaper too — on the MoE model the fg target went from
−43.6% (canonical, vs unsloth) to a TIE with unsloth-off under FA4 (90.7 vs 90.5 GiB
at s70000), which is exactly the failure mode this plan's both-bars criterion
forbids. Expect the same compression here: the dense win must come from what the fg
path uniquely removes (dense MLP live operands + CPU-resident base weights +
saved-activation ownership), not from attention. Plan for the decomposition-first
loop, not for a free win.

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
`scripts/testing/qwen35_dense_integration_probe.py` and
`scripts/testing/qwen35_dense_shapes_probe.py`. If a future qwen3.5-family class is
added, check its norm convention before trusting any loss. A ~13.x loss on ANY stage
below is this signature until proven otherwise.

Second loss trap (qwen3.5-specific, discovered 2026-07-05): at very long cutoffs
(observed at s≥80000 on the 35B MoE, canonical AND fa4, asym AND superoffload) the
packed dataset yields per-step loss EXACTLY 0.0 — zero supervised tokens, not a
numerics win. s50000 has not shown it, but treat `loss == 0.0` at any stage as
`inconclusive_wrong_config` (dataset/template packing), never as a pass.

## Stage 0: Alias, Label, And Runtime Ownership

Required script alias (already present in both wrappers — verify, do not assume):

```bash
[q3.5-27b]="Qwen/Qwen3.5-27B"
```

```text
scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
```

Do not run the final source-profile commands unless both wrappers contain this alias
and dry-run resolves it to `MODEL_NAME_OR_PATH=Qwen/Qwen3.5-27B`.

Dry-run proof (also proves the FA4 auto-default and the CPUAdamW family):

```bash
RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_dryrun RUNS_LOG=profiling_results/profiling_fix_qwen35dense_fa4_dryrun/runs.log \
GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria:

```text
echo contains: recompute_label=recomp-off-full-fg-ker000-ceil0000-ohbm0
echo contains: flash_attn=fa4
artifact path contains: __recomp-off-full-fg-ker000-ceil0000-ohbm0__
artifact path contains: __attnfa4__
artifact path contains: __gradofftrue__weightofftrue (CPUAdamW-family label; may be
  folded into the NAME_MAX truncate+hash tail — verify via RUN_ID if truncated)
route tag contains: route000_lora0_accfp32
command.txt contains: BACKEND=asym_cpuadamwds
command.txt contains: USE_ASYM_CPU_ADAMW=true and ASYM_CPU_ADAMW_BACKEND=deepspeed
command.txt contains: RUN_ID=...recomp-off-full-fg-ker000-ceil0000-ohbm0...
command.txt contains: ASYM_GEMM_LF_CONFIG_RECOMP_LABEL=recomp-off-full-fg-ker000-ceil0000-ohbm0
command.txt contains: ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_ROUTE_KERNEL_CODE=000
```

Negative guard proofs (each must fail loudly):

```bash
# 1. routed kernels on a dense model
RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_dryrun_bad GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh

# 2. a nonzero activation ceiling on a non-NVMe backend
RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil32|ligerloss1 ; 128|1|1 ; none|false|false|false|false|false' \
DRY_RUN=true PREPARE_DATASETS=false PLOT=false RUN_POST=false \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_dryrun_bad GPU_POOL=0 PROFILERS=source MAX_STEPS=1 WARMUP_STEPS=1 \
bash scripts/lf/profile_lora_lf_test_source.sh
```

Pass criteria for the negative guards:

```text
1: script exits nonzero; error states dense model Qwen/Qwen3.5-27B must use recomp-off-full-fg-ker000
2: script exits nonzero; error states -ceil<N> with N > 0 requires an asym NVMe backend
```

Do not continue if the positive dry run says `ker101`, resolves any routed MoE bit to
1, does not say `flash_attn=fa4`, or if a negative guard is accepted.

## Evidence Discipline

Run experiments one at a time. Do not run baselines and targets in parallel while
validating this path, and do not run ANY of this concurrently with a host-RAM-heavy
job on the same box (e.g. NVMe pager bring-up runs): the host-OOM watchdog referees
both and will kill the wrong one. Use a new `OUTPUT_ROOT` per stage (the
`profiling_results/profiling_fix_qwen35dense_fa4_*` names below) so artifacts are never overwritten and
never confused with the pre-FA4 `profiling_results/profiling_dense27b_*` history.

Before each run, write down:

```text
expected model: Qwen/Qwen3.5-27B
expected dense/MoE status: dense, qwen35_moes_wrapped=0
expected backend + optimizer family: asym_cpuadamwds, CPUAdamW (deepspeed, grad+weight offload)
expected recompute input:
expected artifact recompute label: recomp-off-full-fg-ker000-ceil0000-ohbm0
expected attention runtime: fa4 (attnfa4 in path)
expected dense wrapper count: 64
expected attention wrapper count:
expected route kernel code: 000
expected comparison baselines: superoffload_mem|unsloth-off (binding) AND superoffload_mem|unsloth
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
jobs.tsv status row (a completed-looking artifact with status failed is failed)
```

Treat the result as inconclusive if:

```text
profile is partial and does not identify a stage bug
path label and profile config disagree
recomp_label is not recomp-off-full-fg-ker000-ceil0000-ohbm0
artifact path lacks __attnfa4__
loss == 0.0 exactly (packing trap) or loss ~13.x (norm trap)
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false
ASYMM_EXPERT_SILU_BWD_GPU=1
ASYMM_MLP_RECOMPUTE_CHUNK != 0
UNSLOTH_GC_OUTER_HBM_EVERY_N != 0 (label must say -ohbm0)
any Qwen3 MoE routed bit is 1
qwen35_moes_wrapped > 0
dense_mlp_finegrained_offload_wrapped == 0
USE_ASYM_CPU_ADAMW=true missing from the target row (wrong family)
a _nocpuadamw/asym row leaked onto this scoreboard (wrong family)
only one bar was run or reported (BOTH superoffload_mem rows are required)
stale artifact was reused (anything under profiling_results/profiling_dense27b_* or pre-2026-07-03)
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

Purpose: prove the dense `ker000` config is real on the FA4 runtime, the CPUAdamW
family resolves, and the model does not trigger MoE routed paths.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s2048 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Pass criteria:

```text
both rows complete a backward step
both artifact paths contain __attnfa4__
target artifact label is recomp-off-full-fg-ker000-ceil0000-ohbm0
target loss within ~0.05 of the pre-FA4 dense band (1.715/1.718) — NOT 0.0, NOT ~13.x
target dense fine-grained wrapper count = 64
target qwen35_moes_wrapped = 0
target route kernel code = 000
target no chunked MLP; outer_hbm effective value 0 (-ohbm0 in the label)
target row is the CPUAdamW family (USE_ASYM_CPU_ADAMW=true); no _nocpuadamw row leaked in
```

## Stage 2: Medium Memory-Shape Gate

Purpose: make sure the dense path scales under FA4 before the real target. This is
not the final scoreboard.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s8192 MAX_STEPS=1 WARMUP_STEPS=0 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Expected memory shape under FA4:

```text
attention workspace is FA4-compressed for ALL rows — it is no longer the separator
superoffload_mem|unsloth keeps recompute intermediates in HBM during layer backward
superoffload_mem|unsloth-off additionally offloads saved boundary tensors to CPU
asym_cpuadamwds|recomp-off-full-fg-ker000 must additionally remove dense MLP live
  operands and keep base weights CPU-resident — this is the entire margin over
  unsloth-off; the optimizer is CPU-side for the whole family, so it separates nothing
```

If target HBM is not clearly below BOTH bars already at s8192, inspect the peak live
activation details before continuing. Identify whether the peak is:

```text
dense MLP live operands (should be CPU-managed by the fg path)
attention saved tensors (FA4 should have shrunk these — verify fa4 actually ran)
LoRA transient or trainable-weight gather
optimizer step / CPUAdamW grad-offload transfer buffers
logits/loss workspace (liger chunking)
allocator reserve / fragmentation
wrong no-grad forward path
wrong saved-tensor hook ownership
```

## Stage 3: Bottleneck Gate At s30000

Purpose: prove the beat-BOTH-bars mechanism at a workload large enough to expose the
real bottleneck without jumping straight to the final s50000 surface. This replaces —
and must not be confused with — the retired 2026-07-03 canonical-attention s30000
gate (71,228.31 MiB vs unsloth 116,170.70: non-FA4, and never compared against
unsloth-off; background evidence only).

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s30000 MAX_STEPS=2 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Pass criteria:

```text
target completes with loss in the baseline band (± ~0.05), not 0.0, not ~13.x
target step_H is NOTICEABLY (≥~5%) below superoffload_mem|unsloth-off (binding bar)
target step_H is ALSO below superoffload_mem|unsloth — both margins reported
CPU RSS is within machine budget (fg CPU activations + CPU base weights + CPUAdamW
  states all count)
runtime counters prove dense fine-grained path fired (64 wrapped, counters > 0)
memory breakdown peak has no unexpected MoE routed tensors
```

If s30000 does not meet this, do not run s50000 as a scoreboard. Run the memory
decomposition below, fix or gate the responsible component, and repeat s30000. A tie
with unsloth-off here is a FAIL for this plan (the MoE precedent: ties do not become
wins at larger seq — the seq-scaling terms are shared).

## Stage 4: Final s50000 Scoreboard

Only run this after Stage 3 is validated.

Run each row separately:

```bash
RUNS='q3.5-27b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000 MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Success criteria (ALL of these — this is the plan's goal, verbatim):

```text
target completes with loss INSIDE THE BASELINE BAND (within ~0.05 of the
  superoffload_mem rows at the same workload; exactly 0.0 = the packing trap;
  ~13.x = the zero-centered-RMSNorm signature — both are automatic fails)
target artifact label is recomp-off-full-fg-ker000-ceil0000-ohbm0 and path has __attnfa4__
target peak HBM (step_H) is NOTICEABLY (≥~5%) below superoffload_mem|unsloth-off —
  the BINDING bar — at s50000.b8.ga1
target peak HBM (step_H) is ALSO below superoffload_mem|unsloth at the same workload
the final table reports the % margin against BOTH bars; beating one bar only, or
  tying unsloth-off, is a FAIL
target CPU RSS is within host budget
dense fine-grained counters fire; MoE routed counters stay zero
memory snapshot/live activation details identify no hidden wrong-path peak
grad_norm recorded (the known long-seq systemic clip issue is record-don't-gate;
  baselines are equally affected — never frame it as asym-only)
```

If the target fails but a baseline also fails, report both failures explicitly. Do not
claim success from "baseline OOM" unless the target completed and the config audit is
clean.

## Optional No-CPUAdamW Cross-Check

Only if the no-CPUAdamW variant is also wanted on the record (separate scoreboard,
never merged with the main one):

```bash
RUNS='q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false

RUNS='q3.5-27b|1 ; asym|recomp-off-full-fg-ker000|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
OUTPUT_ROOT=profiling_results/profiling_fix_qwen35dense_fa4_s50000_nocpuadamw MAX_STEPS=3 WARMUP_STEPS=1 PLOT=false RUN_POST=false \
bash scripts/lf/profile_lora_lf_test_source.sh --gpus 0 --overwrite false
```

Do not compare `asym|...` against `superoffload_mem|...` as the only scoreboard. That
mixes optimizer-offload families.

## Memory Decomposition If The Target Does Not Beat BOTH Bars

Produce a concise table for every failed or surprising stage (always include BOTH
bars, not just the one that lost):

```text
Workload   Backend            Config                                    step_H  RAM   act_H  saved_GPU  saved_CPU  top_peak_owner
---------  -----------------  ----------------------------------------  ------  ----  -----  ---------  ---------  --------------
s30000.b8  superoffload_mem   unsloth-ohbm0                             ...
s30000.b8  superoffload_mem   unsloth-off-ohbm0                         ...
s30000.b8  asym_cpuadamwds    recomp-off-full-fg-ker000-ceil0000-ohbm0  ...
```

Then inspect the detailed live tensors and answer:

```text
which module owns the peak?
is it dense MLP, attention, LoRA, optimizer/CPUAdamW transfer, logits/loss, or allocator reserve?
which exact tensor shape dominates?
is the tensor a live GEMM operand or a saved activation?
is it expected to be CPU-managed by the fine-grained path?
did any raw tensor get stored on ctx outside ActivationOffloadManager?
did the no-grad original forward accidentally save internal tensors?
is the same term present in the unsloth-off peak (shared ⇒ no margin there)?
```

Candidate dense levers, in escalation order (gate each with an A/B at s30000, report
step_s and step_H together, never stack unvalidated levers):

```text
D1  dense-MLP no-grad forward CPU offload (ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD)
D2  dense-MLP CPU-side activation residency (ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT)
D3  CPU buffer pool cap (ASYM_EXPACT_CPU_POOL_MAX_BYTES) — trades RSS for re-alloc churn,
    HBM-neutral; only for host-budget failures
D4  liger logits chunking review at s50000 (loss workspace scales with S·V)
```

Only after this audit should an implementation change be proposed.

## Reporting Format

The final response/table must be plain text and include these metrics only:

```text
fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H  RAM  loss  grad_norm
```

plus the explicit margin lines:

```text
target vs superoffload_mem|unsloth-off : -X.X% step_H   (binding bar — must be ≥~5%)
target vs superoffload_mem|unsloth     : -Y.Y% step_H
```

Use the generated artifact labels in the backend/config columns:

```text
asym_cpuadamwds    recomp-off-full-fg-ker000-ceil0000-ohbm0
superoffload_mem   unsloth-ohbm0
superoffload_mem   unsloth-off-ohbm0
```

Do not report a run as final unless the artifact audit proves it is `q3.5-27b`,
dense, `ligerloss1`, `50000|8|1`, label `recomp-off-full-fg-ker000-ceil0000-ohbm0`,
`__attnfa4__` in the path, the CPUAdamW family (`BACKEND=asym_cpuadamwds`,
`USE_ASYM_CPU_ADAMW=true`), loss in band (not 0.0, not ~13.x), BOTH bars present on
the same scoreboard with margins reported, and the jobs.tsv-status caveat has been
applied.
