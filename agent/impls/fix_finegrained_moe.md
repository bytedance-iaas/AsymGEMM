# Fix Fine-Grained MoE Offload: `unsloth-off` Semantics + Qwen3 MoE Experts

## Goal

Train Qwen3-30B-A3B MoE LoRA-SFT with lower HBM, and ideally at a longer real
sequence length, than the matching existing-system baselines:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

The intended target family is:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
```

The comparison must stay apples-to-apples on optimizer placement. The target uses the
CPUAdamW/DeepSpeed CPU optimizer family through `asym_cpuadamwds`, so the primary
scoreboard baselines are `superoffload_mem|unsloth` and
`superoffload_mem|unsloth-off`, not the `*_nocpuadamw` variants. No-CPUAdamW runs may
be used only as diagnostics, and must be labeled as diagnostics.

`recomp-off-full-fg` for Qwen3 MoE should mean:

```text
outer Unsloth whole-layer gradient checkpointing
+ outer unsloth-off recompute saved-tensor behavior
+ AsymGEMM CPU-resident frozen expert weights
+ Asym CPUAdamW/grad+weight offload from asym_cpuadamwds
+ attention activation placement from the dense full-fg path
+ new Qwen3-MoE fine-grained routed-expert activation placement
```

The external RUNS policy tuple must remain:

```text
none|false|false|false|false|false
```

That tuple is a request for no old expert-policy axes. The `recomp-off-full-fg`
recompute label owns the internal composition. Do not turn on the old
`ASYMM_EXPERT_ACT_OFFLOAD=true` tuple for the target run.

The key debugging goal is not just to make a mode run. It is to answer why the current
MoE path can fall short even though it appears to offload as much as `unsloth-off`, and
why adding AsymGEMM should reduce memory further. A correct implementation must produce
artifacts that prove which tensors remain live and which path owns them.

## Known Baseline Evidence

The s80000 `superoffload_mem|unsloth` baseline already has a complete source-profile
artifact:

```text
profiling_q3_30b_a3b_s80000/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/superoffload_mem__source__unsloth__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__ligerloss1/b8_s80000_ga1/source_profile.json
```

Current measured row:

```text
Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00
Workload   Backend            Recompute    Config              fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  -----------------  -----------  ------------------  -----  -----  -----  ------  -----  -----  ------  -----
s80000.b8  superoffload_mem   unsloth      none + ligerloss1    29.6  130.3    0.0   162.4   91.9  176.9   176.9  359.9
```

Peak breakdown for that row is dominated by routed experts:

```text
saved_activations/routed_experts      69.75 GiB
temporary_workspace/routed_experts    55.78 GiB
saved_activations/norms               20.84 GiB
saved_activations/attention           13.99 GiB
saved_activations/router               2.81 GiB
live_activation/embed_tokens           2.44 GiB
live_activation/router                 0.16 GiB
actual peak allocated                 176.95 GiB
actual peak reserved                  180.69 GiB
```

This is not enough for a final conclusion. The matching
`superoffload_mem|unsloth-off` s80000 baseline must also be run and audited before any
claim against the target.

## Questions This Plan Must Answer

These are stage gates. Do not skip them.

1. Does `q3-30b-a3b|asym_cpuadamwds|recomp-off-full-fg` really resolve to outer
   Unsloth GC and `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`?

2. Does the original no-grad checkpoint forward stay clean, saving only the layer input
   through Unsloth GC and not doing fine-grained activation CPU copies?

3. Does the Qwen3 MoE fine-grained path dispatch only in the backward recompute forward
   under `torch.is_grad_enabled()`?

4. Is `recomp-off-full-fg` on MoE using a new MoE-owned path, not the dense
   `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD` path and not the old
   `ASYMM_EXPERT_ACT_OFFLOAD` path?

5. Are route-expanded tensors measured with MoE shapes `[R,H]`, `[R,I]`, and `[R,2I]`,
   where `R = batch * seq * top_k`? For q3-30b-a3b at s80000/b8/top_k=8,
   `R = 5,120,000`, `H = 2048`, and `I = 768`.

6. Does the implementation avoid fused live operands that are not necessary for the
   operation being run, especially `gate_up [R,2I]`, `grad_gate_up [R,2I]`, and
   `stage_concat_columns`?

7. Do `memory_breakdown_summary.json`, `memory_live_activation_details.csv`,
   `runtime_counters.json`, and the peak snapshot agree about the peak owner?

8. If the target does not beat `superoffload_mem|unsloth-off`, is the blocker routed
   expert saved activations, temporary workspace, attention/norms, CPUAdamW/LoRA
   staging, or stale/wrong config?

9. Does the target beat or at least meaningfully reduce the named
   `superoffload_mem|unsloth` baseline at the real s80000 workload?

10. Does the implementation keep the dense full-fg path intact? Dense code paths are
    already useful and must not be perturbed while adding MoE behavior.

## Evidence Discipline

Before each run, write the expected result into
`agent/impls/fix_finegrained_moe_validation.md`:

1. resolved config:
   - backend,
   - recompute label and `recomp_off_stage`,
   - `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`,
   - `UNSLOTH_GC_OUTER_HBM_EVERY_N`,
   - new Qwen3 MoE fine-grained flag,
   - old `ASYMM_EXPERT_ACT_OFFLOAD`,
   - attention flag,
   - CPUAdamW grad/weight offload state;
2. expected dispatch:
   - which wrappers should install,
   - which counters should fire,
   - which old counters must stay zero;
3. expected memory shape:
   - expected peak owner,
   - expected movement versus `superoffload_mem|unsloth`,
   - expected movement versus `superoffload_mem|unsloth-off`;
4. expected failure mode:
   - OOM location,
   - wrong flag,
   - stale artifact,
   - unsupported expert shape,
   - custom Function storing raw HBM tensors on `ctx`.

After each run, inspect these artifacts before interpreting numbers:

```text
command.txt
train.log
source_profile.json or profile.json
profile config fields
runtime_counters.json
memory_breakdown_summary.json
memory_live_activation_details.csv
memory_actual_peak_breakdown.csv
peak_snapshot_attrib_allblocks.md/csv/json
process_memory.csv
```

Run experiments one at a time. Do not run profiling jobs in parallel while validating a
stage or debugging a failure. Multi-row `RUNS` is allowed only after the rows are known
to run serially and after earlier stages have passed.

Treat a result as inconclusive if any of these are true:

- artifact is partial and does not identify a useful failure point;
- artifact path label and `source_profile.json.config` disagree;
- `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false`;
- `UNSLOTH_GC_OUTER_HBM_EVERY_N` is nonzero;
- old `ASYMM_EXPERT_ACT_OFFLOAD=true` is enabled for the target path;
- dense fine-grained counters are used to claim a MoE fine-grained result;
- expected MoE fine-grained counters do not fire;
- expected old expert activation-offload counters fire;
- `recomp_off_stage` is missing or wrong;
- `asym_cpuadamwds` is compared only against a `*_nocpuadamw` baseline;
- the run reused stale artifacts.

If a result strongly deviates from expectation, first assume one of these before making
a mechanistic conclusion:

1. stale artifact reuse,
2. wrong resolved config,
3. wrong backend family,
4. wrong recompute alias,
5. missing env forwarding into `RUN_ENV`,
6. missing path tag for the new MoE flag,
7. partial profile mistaken for a completed backward,
8. custom Function stored a large raw HBM tensor on `ctx`,
9. old expert activation-offload path accidentally enabled,
10. dense full-fg path accidentally changed.

## Non-Goals And Fixed Constraints

Do not build a generic producer/offload policy list for this fix. The first correct
implementation should be direct and auditable.

Fixed constraints:

1. no chunked MLP or route chunking as the memory win;
2. no `UNSLOTH_GC_OUTER_HBM_EVERY_N` for final comparisons;
3. no old `ASYMM_EXPERT_ACT_OFFLOAD=true` target path;
4. no dense full-fg behavior changes unless a compile/import issue forces a tiny shared
   plumbing edit;
5. no hidden comparison against `*_nocpuadamw` when target is `asym_cpuadamwds`;
6. no final conclusion before both s80000 baselines and the s80000 target are audited.

## The Two Forwards Must Stay Separate

There are two forwards and they must have different behavior.

### 1. Original training forward

This is the forward run by the outer Unsloth gradient checkpointing wrapper under
`torch.no_grad()`. It should save only the layer input/root on CPU.

For the Qwen3 MoE fine-grained path:

```text
if not torch.is_grad_enabled():
  do not run the fine-grained activation-offload custom Function
  do not offload gate/up/act/S_* handles
  use the normal Asym Qwen3 MoE forward with CPU-resident frozen weights
```

This preserves the actual purpose of `unsloth-off`: the original forward does not keep
internal routed-expert intermediates for backward.

### 2. Backward recompute forward

This is the forward rerun during backward under `torch.enable_grad()` and the outer
`save_on_cpu` behavior. This is the only place the new MoE fine-grained activation
placement should run.

Inside this forward, custom `autograd.Function` internals are not automatically saved by
the outer `save_on_cpu` hook. Anything large that the custom backward needs must be
owned explicitly through `ActivationOffloadManager` CPU handles, not raw HBM tensors
stored on `ctx`.

## Current Code Facts

Current script expansion for `recomp-off-full-fg` is dense-focused:

```text
USE_UNSLOTH_GC=true
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=false
ASYMM_ATTN_ACT_OFFLOAD=true
ASYMM_EXPERT_SILU_BWD_GPU=0
ASYMM_MLP_RECOMPUTE_CHUNK=0
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
```

For MoE this is not sufficient:

- `asym_gemm/integrations/lf.py` gates dense fine-grained wrapping with
  `not expert_prefixes`, so dense fine-grained wrapping is skipped for Qwen3 MoE.
- `asym_gemm/training/qwen3_moe.py` has an old expert activation-offload path behind
  `ASYMM_EXPERT_ACT_OFFLOAD`. That path is `_ActivationOffloadQwen3ExpertFunction`.
- The old path offloads several activations, but it is not the target MoE full-fg
  design. It still uses fused gate/up base output and can materialize fused backward
  operands such as `grad_gate_up [R,2I]` or `stage_concat_columns`.
- Therefore, current `q3-30b-a3b|asym_cpuadamwds|recomp-off-full-fg` should not be
  treated as a completed MoE fine-grained implementation unless a new MoE-specific flag,
  wrapper, counters, and artifacts prove that path is active.

## Required Design

Add a MoE-owned fine-grained path, separate from dense full-fg and separate from old
expert activation offload.

Recommended new flag:

```text
ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1
ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1
```

Recommended artifact tag:

```text
moefg1 / moefg0
```

The tag is important. Without it, old artifacts and new artifacts can share confusingly
similar `recomp-off-full-fg` paths.

The integration rule should be:

```text
if qwen3_moe model and recompute stage is full-fg:
  enable ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD
  keep ASYMM_EXPERT_ACT_OFFLOAD=false
  keep ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD harmless/no-op for MoE
```

`dense-fg` may be used as a diagnostic alias later, but the Qwen3 MoE target in this
document is `recomp-off-full-fg`.

In Python wrapping:

```text
if backend is Asym/Asym CPUAdamW runtime
and Qwen3 MoE routed experts are present
and ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD is true:
  install the new Qwen3 MoE fine-grained routed-expert wrapper
else:
  keep existing behavior
```

Do not key the new path on the external RUNS tuple. The tuple remains
`none|false|false|false|false|false`; the recompute label owns the target composition.

## Module-Level Implementation Plan

### Harness and artifact plumbing

Files:

- `scripts/lf/run_lf_lora_sft.sh`
- `scripts/lf/profile_lora_lf_test_source.sh`
- `scripts/lf/profile_lora_lf_test_both.sh`
- `scripts/lf/run_lf_profiled_train.py`

Needed changes:

1. Add `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD`, default `0`.
2. Forward it through `RUN_ENV`.
3. Record it in `ASYM_GEMM_LF_CONFIG_*`.
4. Include it in `source_profile.json.config`.
5. Add a path tag such as `moefg1`/`moefg0`.
6. Teach profile-complete validation to expect `moefg1` for MoE `recomp-off-full-fg`.
7. Keep `UNSLOTH_GC_OUTER_HBM_EVERY_N=0`, and add no final-comparison command that
   changes it.
8. Keep `ASYMM_EXPERT_ACT_OFFLOAD=false` for `recomp-off-full-fg`.

For dense models, the new MoE flag must be ignored. For MoE models, dense full-fg must
not be used as proof that MoE full-fg is implemented.

### Integration wrapper

File:

- `asym_gemm/integrations/lf.py`

Needed changes:

1. Import a new Qwen3 MoE fine-grained builder.
2. Detect the new flag with both direct and `ASYM_GEMM_LF_CONFIG_*` env names.
3. Install the wrapper only for Qwen3 MoE routed experts or `AsymQwen3MoeBlock`.
4. Do not install it on dense MLP modules.
5. Add setup-report fields:
   - `qwen3_moe_finegrained_offload_enabled`,
   - `qwen3_moe_finegrained_offload_wrapped`.
6. Keep old `qwen3_moes_wrapped` and `packed_experts_wrapped` reporting intact.
7. Ensure `collect_asym_lf_metadata()` records the new mode.

### New MoE fine-grained expert path

Preferred new file:

- `asym_gemm/training/qwen3_moe_finegrained.py`

Minimal dispatch edits may be needed in:

- `asym_gemm/training/qwen3_moe.py`
- `asym_gemm/training/__init__.py`

Do not put this into `dense_mlp_finegrained.py`.

The new path should own the routed-expert activation lifetime for the backward
recompute forward:

```text
input hidden / route metadata
-> route/pack to [R,H]
-> gate base + gate LoRA, offload gate and S_gate, release HBM
-> up base + up LoRA, offload up and S_up, release HBM
-> stage only what is needed to compute act = silu(gate) * up
-> offload act or immediately stage it for down, then release gate/up stages
-> down base + down LoRA
-> scatter routed output back to [M,H]
```

Backward should avoid fused `[R,2I]` operands:

```text
grad_output route gather
-> down base dx + down LoRA grads using staged act only when needed
-> compute grad_act
-> compute grad_up and grad_gate without materializing grad_gate_up [R,2I]
-> gate base dx + gate LoRA grads from grad_gate
-> up base dx + up LoRA grads from grad_up
-> sum route-expanded grad_packed pieces
-> scatter/index_add back to input hidden
```

The first implementation may use a simpler gate/up staging order. If the peak artifact
shows gate/up overlap is the limiter, add the sequential order:

```text
stage gate, compute/offload grad_up ingredients, release gate
stage up, compute intermediate needed for grad_gate, release up
stage gate again, compute grad_gate, release gate
```

Do not add this order blindly as a conclusion. Add it only if memory breakdown proves
the gate/up overlap is still a peak owner at a meaningful sequence length.

### Base weight layout

The old Qwen3 expert path uses a fused `gate_up_base` that returns `[R,2I]`. The target
MoE fine-grained path should split base execution:

```text
gate_base: [E,I,H]
up_base:   [E,I,H]
down_base: [E,H,I]
```

This can be implemented by constructing separate grouped frozen-linear wrappers from
the two halves of the original `gate_up_proj` host weights. CPU host memory should stay
the same order of magnitude as the fused representation; the point is to avoid
route-expanded fused HBM tensors.

### LoRA and CPUAdamW placement

The target backend is `asym_cpuadamwds`, so trainable LoRA weights, gradients, and
optimizer state placement must be recorded. If the new custom Function reads LoRA banks
directly in backward, it must coordinate with the existing CPUAdamW weight/grad offload
logic:

- gather LoRA weights only for the current layer/function,
- release them through the existing post-accumulate or coordinator path,
- do not keep full LoRA banks in HBM across the forward-to-backward gap,
- record counters for gather/release if new code owns them.

If the new class is not an `AsymQwen3Experts` subclass, update
`asym_gemm/training/weight_offload.py` so CPUAdamW weight offload still recognizes it.

### Runtime counters

Add explicit counters so artifacts can prove the path:

```text
qwen3_moe_finegrained_forward_calls
qwen3_moe_finegrained_backward_calls
qwen3_moe_finegrained_gate_base_calls
qwen3_moe_finegrained_up_base_calls
qwen3_moe_finegrained_down_base_calls
qwen3_moe_finegrained_stage_concat_columns_calls
qwen3_moe_finegrained_fused_gate_up_hbm_bytes
qwen3_moe_finegrained_saved_cpu_bytes
qwen3_moe_finegrained_stage_hbm_peak_bytes
```

Required expectation for the target:

```text
qwen3_moe_finegrained_forward_calls > 0
qwen3_moe_finegrained_backward_calls > 0
qwen3_moe_finegrained_stage_concat_columns_calls == 0
old expact counters == 0
dense fine-grained counters == 0 for MoE
```

## Staged Implementation And Validation

Every stage must update `agent/impls/fix_finegrained_moe_validation.md`. Do not move to
the next stage until the current stage has either passed or has a documented blocker.

### Stage 0: config truth and path ownership

Implement only harness/config/reporting changes for the new MoE flag and artifact tag.
Run a tiny dry/smoke config if needed.

Validation:

- `recomp-off-full-fg` records `recomp_off_stage=full-fg`.
- new config field exists and is `true` only for the intended MoE target.
- `UNSLOTH_GC_OUTER_HBM_EVERY_N=0`.
- `ASYMM_EXPERT_ACT_OFFLOAD=false`.
- artifact path includes the new `moefg1` tag when enabled.

### Stage 1: baseline matrix at s80000

Run the two primary baselines serially:

```bash
RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm

RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm
```

`profile_lora_lf_test_both.sh` may be used instead when Nsight artifacts are needed,
but do not run both baselines in parallel.

Validation:

- both complete or have clearly documented OOM/host-OOM status;
- metrics table includes `fwd_s`, `bwd_s`, `opt_s`, `step_s`, `fwd_H`, `bwd_H`,
  `step_H`, `RAM`;
- memory breakdown table includes saved activations, temporary workspace, trainable
  state/optimizer if visible, live activations, and reserved-unallocated;
- no conclusion about the target until this matrix is known.

### Stage 2: current Asym MoE control

Before adding the new expert Function, run the current Asym control at a small workload:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-base|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-attn|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
```

Expected before implementation:

- base/attn prove outer recompute/offload and CPUAdamW ownership;
- `full-fg` does not yet prove MoE fine-grained expert offload unless new counters fire;
- old expert activation-offload counters must stay zero for the target.

### Stage 3: unit/parity tests for the new MoE Function

Add focused tests before LF profiling:

- small static balanced routes,
- skewed routes,
- repeated experts,
- empty expert groups,
- learned router mode if the wrapper owns routing,
- LoRA r64/a16/drop0.00,
- bf16 CUDA path,
- CPUAdamW weight gather/release path if touched.

Validation:

- output finite;
- gradients finite;
- LoRA A/B gradients close to existing Qwen3 MoE reference;
- no raw large HBM tensors saved on `ctx`;
- new counters fire in forward/backward;
- old expact and dense-fg counters stay zero.

### Stage 4: small LF smoke

Run:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
```

Expected:

- complete one measured step;
- new MoE fine-grained wrapper count > 0;
- old `ASYMM_EXPERT_ACT_OFFLOAD` path not active;
- no `stage_concat_columns`;
- no dense full-fg counters;
- memory is not a final claim yet, but peak owner must be understandable.

### Stage 5: memory-shape gate

Run a meaningful intermediate sequence, serially:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
```

Compare against same-sequence baselines if they exist or run them as needed. This stage
answers whether routed-expert saved activations and temporary workspace are moving in
the expected direction before paying for s80000 target runs.

If the target peak is still dominated by live gate/up/act overlap, use the
memory-breakdown and live-activation details to decide whether to add sequential
gate/up staging. Do not add it as an unvalidated trick.

### Stage 6: final s80000 target

Run:

```bash
RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm
```

Use `profile_lora_lf_test_both.sh` if final Nsight artifacts are required, but still run
serially.

Final comparison table must include:

```text
Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00    CPUAdamW family: yes
Workload   Backend            Recompute          Config               fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM
---------  -----------------  -----------------  -------------------  -----  -----  -----  ------  -----  -----  ------  -----
s80000.b8  superoffload_mem   unsloth            none + ligerloss1
s80000.b8  superoffload_mem   unsloth-off        none + ligerloss1
s80000.b8  asym_cpuadamwds    recomp-off-full-fg moefg1 + ligerloss1
```

Final memory decomposition must compare:

```text
saved_activations/routed_experts
temporary_workspace/routed_experts
attention saved/live/temp
router saved/live/temp
norms saved/live/temp
trainable LoRA / grad / optimizer state if visible
allocator reserved-unallocated
actual peak allocated/reserved
RAM
```

## Success Criteria

The implementation is successful only if all are true:

1. `asym_cpuadamwds|recomp-off-full-fg` at s80000 completes or reaches a clearly higher
   sequence-length ceiling than the baselines.
2. The s80000 target uses the new Qwen3 MoE fine-grained path, proven by counters and
   wrapper counts.
3. The target has lower `step_H` than `superoffload_mem|unsloth-off` at the same
   workload, or the doc records a precise audited reason why it cannot.
4. The target meaningfully reduces routed-expert saved activations or temporary
   workspace compared with `superoffload_mem|unsloth`.
5. No final claim relies on `UNSLOTH_GC_OUTER_HBM_EVERY_N`, chunking, old expact tuple
   flags, or stale artifacts.
6. Dense `recomp-off-full-fg` results are not regressed by the MoE changes.

## Summary By Stage

```text
Stage 0  Add MoE flag, config fields, artifact tag, and validation checks.
Stage 1  Establish s80000 superoffload_mem unsloth and unsloth-off baselines.
Stage 2  Measure current Asym MoE controls; prove current full-fg is not MoE-fg unless counters say so.
Stage 3  Implement/test new Qwen3 MoE fine-grained custom Function and wrapper.
Stage 4  Run small LF smoke; verify counters, wrapper counts, and no old paths.
Stage 5  Run s30000/s45000 memory-shape gates; fix real live-activation blockers only when proven.
Stage 6  Run final s80000 target and compare directly against the two named baselines.
```

## Addendum: Remaining Implementation After Current MoE Full-FG Artifacts

The current `qwen3_moe_finegrained.py` path already proves that the new MoE-owned
`moefg1` path can run, but the s80000 artifacts show the remaining blocker:
hidden-width route-expanded tensors are still materialized before scatter.

For q3-30b-a3b at s80000/b8/top_k=8:

```text
M = batch * seq = 640,000
R = M * top_k  = 5,120,000
H = 2048
I = 768

[M,H] bf16 ~= 2.44 GiB
[R,I] bf16 ~= 7.32 GiB
[R,H] bf16 ~= 19.53 GiB
```

The next fix must remove global `[R,H]` tensors. It should not rewrite all MoE
projections as naive per-expert loops. Keep grouped AsymGEMM where the memory shape is
acceptable, and only split the hidden-width down/hidden-gradient work.

### Design Decision: block-wise hidden scatter

Add a block-wise hidden scatter mode:

```text
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=8
ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=8
```

Semantics:

- `0` or unset: current behavior, useful only as a control.
- `1`: closest to torch memory behavior; per active expert segment.
- `8`: recommended first target; about 1.2 GiB average `[R_block,H]` at s80000.
- `16`: faster diagnostic; about 2.4 GiB average `[R_block,H]` at s80000.

The mode should be enabled by the `recomp-off-full-fg` MoE target after smoke testing,
and the output path tag should include the value, for example:

```text
moefg1__dscatter8
```

This is not route chunking or chunked MLP. It uses natural expert groups already
created by routing. The purpose is to scatter hidden-width outputs at the same
granularity that torch's expert loop scatters, while preserving some grouped AsymGEMM
amortization.

### Required helper primitives

Files:

- `asym_gemm/training/activation_offload.py`
- `asym_gemm/training/qwen3_moe_finegrained.py`

Add an `ActivationOffloadManager` row-slice staging helper:

```text
stage_rows(handle, start, end, tag) -> CUDA tensor with handle.tensor[start:end]
```

Requirements:

1. it waits for the CPU handle to be ready;
2. it allocates/caches only `(end - start, width)`, not the full handle shape;
3. it records staged bytes under the supplied tag;
4. `release_stage(..., drop_cache=True)` must release the slice staging buffer;
5. it should be used only for contiguous row ranges.

The Qwen3 MoE packed representation already has contiguous grouped rows. Use
`offsets` and `experts[:-1]` to build block ranges:

```text
group range:    [g0, g1)
row range:      [offsets[g0], offsets[g1])
block_offsets:  offsets[g0:g1+1] - offsets[g0]
block_experts:  concat(experts[g0:g1], sentinel -1)
block_tokens:   token_indices[row0:row1]
block_weights:  routing_weights[row0:row1]
```

Skip empty groups. Do not reorder rows inside the block.

### Stage A: forward down scatter

Current bad forward shape:

```text
down_delta = _lora_b_forward(...)        # [R,H]
output     = _base_forward(down_base...) # [R,H]
scatter both into [M,H] only after the large tensors exist
```

Replace only the down path:

```text
scattered = zeros([M,H])

for each expert block:
  stage act rows [row0:row1] from act_cpu
  compute down LoRA-B chunk from down_low_rank[row0:row1] -> [R_block,H]
  weighted index_add into scattered using block_tokens/block_weights
  release down LoRA-B chunk
  compute down base chunk with block_offsets/block_experts -> [R_block,H]
  weighted index_add into scattered
  release down base chunk and act stage
```

Keep full `[R,r]` `down_low_rank` for the first implementation. It is much smaller than
`[R,H]`, and changing it at the same time would obscure the main result.

Expected artifact movement:

- no live routed expert `[5120000,2048]` down output at peak;
- `qwen3_moe_finegrained_stage_hbm_peak_bytes` should show block-sized down stages;
- `memory_live_activation_details.csv` should not show `down_base [R,H]` as a live
  activation.

### Stage B: backward down without global `grad_2d [R,H]`

Current bad backward shape:

```text
grad_2d = _route_grad_from_tokens(...) # [R,H]
d_s_down, grad_down_lora_B, down_base_dx all consume grad_2d
```

Replace the down-gradient section with block-wise route gather:

```text
grad_act = allocate [R,I] or CPU-offloaded equivalent

for each expert block:
  grad_block = grad_output.index_select(0, block_tokens) -> [R_block,H]
  apply block routing weights if output_weighted
  compute down LoRA dS chunk and accumulate down LoRA-B grad
  compute down base dX chunk -> [R_block,I]
  compute down LoRA dX chunk -> [R_block,I]
  write/add the result into grad_act[row0:row1]
  release grad_block and hidden-width chunks
```

For the first version, keeping `grad_act [R,I]` in HBM is acceptable because it is
about 7.3 GiB at s80000, not 19.5 GiB. If artifacts later prove `[R,I]` is the peak
blocker, add CPU-offloaded `grad_act` as a separate stage.

Important implementation detail: LoRA-B weight gradients must accumulate across
blocks. Initialize `grad_down_lora_B` once, add each block's contribution, and only
return the final tensor.

Expected artifact movement:

- no `scatter_grad` global `[R,H]`;
- no `grad_2d [5120000,2048]` live tensor;
- routed-expert temporary workspace should drop by roughly one hidden-width routed
  tensor, subject to allocator reserve.

### Stage C: backward gate/up dX direct scatter

After Stage A/B, the peak may move to input-gradient reconstruction:

```text
grad_packed = gate dx [R,H] + up dx [R,H] + LoRA dx [R,H]
grad_hidden.index_add_(..., grad_packed)
```

Replace this with direct scatter into `[M,H]`:

```text
grad_hidden = zeros([M,H])

for each expert block:
  stage grad_gate rows for block
  compute gate base dX chunk -> [R_block,H]
  compute gate LoRA dX chunk -> [R_block,H]
  apply input routing weights if input_weighted
  index_add into grad_hidden, release chunks

  stage grad_up rows for block
  compute up base dX chunk -> [R_block,H]
  compute up LoRA dX chunk -> [R_block,H]
  apply input routing weights if input_weighted
  index_add into grad_hidden, release chunks
```

This should remove the remaining global `[R,H]` backward tensor. It is acceptable for
`grad_gate` and `grad_up` themselves to remain CPU handles with block staging. Do not
restore a fused `[R,2I]` gradient path.

### Stage D: counters and artifact proof

Add counters to `AsymExecutionStats` and profile output:

```text
qwen3_moe_finegrained_down_scatter_block_experts
qwen3_moe_finegrained_down_scatter_blocks
qwen3_moe_finegrained_down_scatter_max_block_rows
qwen3_moe_finegrained_hidden_route_global_tensors_avoided
qwen3_moe_finegrained_stage_rows_calls
```

Required target expectations:

```text
qwen3_moe_finegrained_down_scatter_block_experts == 8
qwen3_moe_finegrained_down_scatter_blocks > 0
qwen3_moe_finegrained_down_scatter_max_block_rows << R
qwen3_moe_finegrained_stage_concat_columns_calls == 0
old expact counters == 0
dense fine-grained counters == 0 for MoE
```

The validation doc must record whether the peak still contains any of these shapes:

```text
[5120000,2048]  # global [R,H], should disappear
[5120000,768]   # [R,I], acceptable for the first pass
[640000,2048]   # final [M,H], expected
```

### Stage E: validation sequence for this addendum

Do not run experiments in parallel.

1. Unit/parity tests:
   - block size 1, 8, 16;
   - balanced routes;
   - skewed routes;
   - empty expert groups;
   - LoRA gradients for down/gate/up.

2. LF smoke:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
```

3. Meaningful memory gate:

```text
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
```

4. Target comparison:

```text
q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false
```

The expected result is not necessarily torch-level runtime. The first success criterion
is memory: remove global `[R,H]` while keeping the runtime within a plausible range.
If block size 8 is too slow, try block size 16 only after proving block size 8 removes
the peak. If block size 8 still uses too much HBM, try block size 1 before changing the
algorithm.

### Longer-term kernel, not the next step

If block-wise scatter proves the memory fix but runtime is unacceptable, the principled
final kernel is:

```text
grouped_asym_gemm_scatter_add(
  a_block_or_full,
  weight_cpu,
  offsets,
  experts,
  token_indices,
  routing_weights,
  out[M,H],
)
```

That kernel should write directly into `[M,H]` without creating `[R,H]`, while keeping
the grouped launch shape. Do not start with this kernel. First use block-wise hidden
scatter to prove the memory mechanism and expected peak movement.
