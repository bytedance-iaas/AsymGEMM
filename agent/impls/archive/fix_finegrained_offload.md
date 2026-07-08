# Fix Fine-Grained Offload: `unsloth-off` Semantics + AsymGEMM Placement

## Goal

Train dense Qwen3-32B LoRA-SFT at a strictly longer real sequence length than the
matching existing-system baseline. The comparison must be apples-to-apples on the
CPUAdamW/optimizer-offload axis:

```text
# no CPUAdamW / no CPU optimizer-offload comparison set
q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false

# CPUAdamW / CPU optimizer-offload comparison set
q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
q3-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

The intended final family is:

```text
q3-32b|1 ; asym|recomp-off-full|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
q3-32b|1 ; asym|recomp-off-full-fg|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
q3-32b|1 ; asym_cpuadamwds|recomp-off-full[-fg]|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
```

Comparison rule:

```text
asym|recomp-off-*:
  compare against superoffload_mem_nocpuadamw and zero3_offload_mem_nocpuadamw

asym_cpuadamwds|recomp-off-*:
  compare against superoffload_mem and zero3_offload_mem
```

Do not claim a win from `asym|...` against `superoffload_mem|...` without also showing
the corresponding `*_nocpuadamw` baseline. Conversely, do not use the `*_nocpuadamw`
baseline as the only scoreboard when the Asym run uses CPUAdamW.

`recomp-off-*` should be a composition of:

```text
Unsloth whole-layer gradient checkpointing
+ unsloth-off recompute saved-tensor behavior
+ AsymGEMM CPU-resident base/frozen weights
+ selected AsymGEMM attention/MLP activation placement
```

The key debugging goal is not only to implement a mode. It is to explain why the current
attempt falls short even though, mechanically, it appears to offload at least as much as
the matching `superoffload_mem[_nocpuadamw]|unsloth-off` baseline and should have
additional AsymGEMM advantages. A correct design must produce artifacts that answer that
question.

## Questions This Plan Must Answer

These questions are stage gates. Do not skip them and do not draw conclusions from
artifacts that cannot answer them.

1. Does every `recomp-off-*` run really use outer Unsloth GC with
   `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true` during backward recompute?

2. Is the original no-grad checkpoint forward still clean, saving only the layer input
   on CPU and no internal MLP/attention activations?

3. After module replacement, do the Asym attention/MLP activation-offload paths dispatch
   only in the backward recompute forward under `torch.is_grad_enabled()`?

4. Are large tensors inside custom `autograd.Function`s owned explicitly by
   `ActivationOffloadManager`, instead of being raw HBM tensors on `ctx` that the outer
   `save_on_cpu` hook cannot manage?

5. Is the current dense E=1 expert wrapper actually better than `unsloth-off`, or does
   it lose because it still creates fused live operands such as `gate_up [M,2I]`,
   `grad_gate_up [M,2I]`, `stage_concat_columns`, or down-LoRA `[M,I]` overlap?

6. Is any artifact accidentally measuring the wrong path, especially
   `ASYMM_EXPERT_SILU_BWD_GPU=1`, missing outer `save_on_cpu`, stale profile reuse, or a
   partial after-forward profile?

7. Does the attention path independently help, hurt, or do nothing once outer
   `unsloth-off` semantics are active?

8. Does the baseline gap come from CPU optimizer/CPUAdamW versus CPU param/LoRA offload,
   rather than activation placement? This is why `superoffload_mem_nocpuadamw` and
   `zero3_offload_mem_nocpuadamw` exist as artifact labels.

9. If `asym|recomp-off-full` does not beat the no-CPUAdamW baseline, is the dense MLP
   current wrapper the remaining blocker, and does the fine-grained dense path remove
   the exact blocker seen in the memory snapshot?

10. Once activation behavior is proven, does `asym_cpuadamwds` help or hurt because of
    CPUAdamW/grad-offload contention and trainable LoRA weight gather/release behavior?

## Evidence Discipline

Do not make fast conclusions from a peak number alone. Every stage needs a written
expectation before the run and an artifact audit after the run.

Before each run, write down:

1. expected resolved config:
   - backend,
   - recompute label and `recomp_off_stage`,
   - `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`,
   - attention/dense wrapper flags,
   - CPUAdamW/optimizer-offload state,
   - `ASYMM_EXPERT_SILU_BWD_GPU`;
2. expected dispatch path:
   - which wrappers should install,
   - which custom Functions/counters should fire,
   - which wrappers/counters should stay zero;
3. expected memory shape:
   - which baseline family it is compared against,
   - whether peak should go down, stay similar, or intentionally expose a blocker,
   - which tensors are allowed to dominate the peak;
4. expected failure mode if it fails:
   - OOM location,
   - likely wrong flag,
   - likely unsupported path,
   - likely current-wrapper limitation.

After each run, inspect artifacts before interpreting performance:

```text
command.txt
train.log
profile.json.config
profile.json partial/heartbeat fields
memory snapshot peak frame
memory breakdown summary
runtime counters
artifact path labels
```

Run experiments one at a time while validating a new stage or debugging a failure. Do
not launch multiple GPU jobs in parallel, and do not use a multi-row `RUNS` batch for a
stage whose previous row has not already passed. Multi-row `RUNS` is acceptable only
for already-stable final serial sweeps, and only when the wrapper runs rows strictly one
after another on the intended GPU. If one row fails, stop and audit it before running
the next row.

Treat a result as inconclusive, not as evidence, if any of these are true:

- profile is partial unless the partial artifact clearly identifies the stage bug;
- config does not match the intended stage exactly;
- artifact path label and `profile.json.config` disagree;
- `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false` for any `recomp-off-*` run;
- `ASYMM_EXPERT_SILU_BWD_GPU=true` for Stage 2-6;
- wrong CPUAdamW/optimizer-offload family was compared;
- expected wrapper/counter did not fire;
- unexpected wrapper/counter fired;
- peak frame is from a path that should have been disabled.

If a result deviates strongly from expectation, assume one of these before making a
mechanistic conclusion:

1. stale artifact reuse,
2. wrong resolved config,
3. wrong backend family,
4. wrong recompute label alias,
5. missing env forwarding into `RUN_ENV`,
6. profile-complete validator accepted an old artifact,
7. partial profile was mistaken for a completed backward,
8. the custom Function stored a large raw tensor on `ctx`,
9. nested saved-tensor hooks changed ownership,
10. the current dense wrapper hit a known fused `[M,2I]` path.

Only after these checks pass should the artifact be used to decide whether the current
design works, whether attention/dense is responsible, or whether Stage 6 is required.

## Non-Goals And Fixed Constraints

Important non-goal: do not build or depend on a generic fine-grained producer/offload
list for this fix. No `ASYMM_OFFLOAD_PRODUCERS` policy is needed for the first correct
implementation. The target behavior is fixed:

1. original layer forward is Unsloth GC style and saves only the layer input/root,
2. backward recompute forward is run under blanket `save_on_cpu`, exactly like
   `unsloth-off`,
3. AsymGEMM wrappers replace selected submodules so the wide MLP/attention pieces save
   CPU handles and use CPU-resident base weights instead of saving their recompute graph
   tensors in HBM,
4. no chunked MLP,
5. no optional producer-hold axes while diagnosing correctness.

Do not jump directly to a new dense MLP implementation. First make the current
composition exactly measurable:

```text
unsloth-off blanket recompute saves
+ AsymGEMM backend/weights
+ one module family at a time
```

Only after the clean module-by-module artifacts show that the current dense wrapper is
the remaining problem should we add the new dense-specific code path.

## The two forwards must stay separate

Do not reason about "forward" as one thing. There are two different forwards.

### 1. Original training forward

This is the forward run by `UnslothGradientCheckpointing.forward`.

Code:

- `LlamaFactory/src/llamafactory/model/model_utils/checkpointing.py:63`
- `LlamaFactory/src/llamafactory/model/model_utils/checkpointing.py:64`
- `LlamaFactory/src/llamafactory/model/model_utils/checkpointing.py:67`

Current behavior:

```python
saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
with torch.no_grad():
    outputs = forward_function(hidden_states, *args)
ctx.save_for_backward(saved_hidden_states)
```

This is exactly what we want. The original forward should persist only the layer input
`X_in`/`hidden_states` on CPU. It should not build an autograd graph for MLP, attention,
norm, residual, or LoRA internals.

The Asym wrappers already mostly respect this boundary:

- MLP activation offload dispatch requires `torch.is_grad_enabled()`
  (`asym_gemm/training/qwen3_moe.py:2476`).
- Attention activation offload falls back to plain forward when grad is disabled
  (`asym_gemm/training/attention_activation_offload.py:979`).
- Attention saved-tensor wrapper also skips when grad is disabled
  (`asym_gemm/training/attention_activation_offload.py:236`).

So do not add fine-grained offload into the original no-grad forward. That would only
add CPU traffic and risk changing the checkpoint root semantics.

### 2. Backward recompute forward

This is the forward run inside `UnslothGradientCheckpointing.backward`.

Code:

- `checkpointing.py:75`: reload the saved CPU `hidden_states`
- `checkpointing.py:76`: move it to CUDA
- `checkpointing.py:78`: enter `torch.enable_grad()`
- `checkpointing.py:79-83`: recompute the layer forward
- `checkpointing.py:86`: immediately run backward through that recompute graph

This is the only forward we target. During this recompute, we want:

```text
stage one layer input -> recompute one layer -> offload/release wide intermediates
-> backward that one layer immediately -> release that layer -> next layer
```

The target is not "plain Unsloth recompute forward with everything saved in HBM".
The target is:

```text
outer recompute saved-tensor behavior = unsloth-off
selected submodule internals = AsymGEMM custom autograd with CPU handles
```

## Current status and staged implementation plan

Current `recomp-off` is not yet the clean target. It currently does most of the Asym
module installation/dispatch, but it is missing the exact `unsloth-off` blanket
recompute saved-tensor behavior:

```text
current recomp-off =
  USE_UNSLOTH_GC=true
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1
  ASYMM_EXPERT_ACT_OFFLOAD=true
  ASYMM_ATTN_ACT_OFFLOAD=true
  ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
  UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false   # wrong for this target
```

Some existing artifacts are also dirty or incomplete:

- the s16384 forensic run had `ASYMM_EXPERT_SILU_BWD_GPU=1`, so it measured the
  GPU-SiLU path, not the intended CPU-SiLU memory path;
- the s45000 `expact1/attnact1/loraA=cpu` artifact is only an after-forward partial,
  not a completed backward peak.

Therefore, first fix config truth and then measure baselines and module families one at
a time. Each stage has a validation gate. Do not move to the next stage until the
previous stage either:

1. completes with a profile proving the exact expected config and peak frame, or
2. fails with an artifact that clearly identifies the active config and failure point.

### Stage 0: config truth and artifact validation

Goal: make `recomp-off-*` and baseline artifacts impossible to misread.

Add explicit recompute labels:

```text
recomp-off-base
recomp-off-attn
recomp-off-dense
recomp-off-full
recomp-off-dense-fg
recomp-off-full-fg
```

Keep `recomp-off` as an alias only if it maps to one explicit stage, preferably
`recomp-off-full`. The artifact must record the resolved stage in `profile.json.config`
as `recomp_off_stage`.

For all `recomp-off-*` stages:

```bash
GRADIENT_CHECKPOINTING=true
USE_UNSLOTH_GC=true
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_EXPERT_SILU_BWD_GPU=0
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
ASYMM_LAYER_ACT_OFFLOAD=false
ASYMM_LAYER_GC=false
ASYMM_ATTN_SDPA_RECOMPUTE=false
ASYM_EXPERT_RECOMPUTE_POLICY=none
ASYMM_MLP_RECOMPUTE_CHUNK=0
```

Also forward/log/validate:

```bash
ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE
ASYMM_EXPERT_SILU_BWD_GPU
ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_SILU_BWD_GPU
ASYM_GEMM_LF_CONFIG_UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU
ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_SURGICAL_OFFLOAD
ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD
```

Current harness issues to fix:

Apply these harness changes to both LF profiler wrappers:

```text
scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
```

- the recompute parser should parse the new `recomp-off-*` labels;
- the existing `recomp-off` forced bundle should become a stage resolver that sets only
  the flags for the selected stage;
- every `recomp-off-*` stage must set `unsloth_recompute_save_on_cpu=true`;
- both wrappers currently default `ASYMM_EXPERT_SILU_BWD_GPU=1`; that default is wrong
  for these max-memory stages, so the stage resolver must force `0`.
- `scripts/lf/run_lf_lora_sft.sh` should explicitly include
  `ASYMM_EXPERT_SILU_BWD_GPU` in `RUN_ENV` and in `profile.json.config`.

Stage-0 profile completion should reject a `recomp-off-*` profile unless:

```text
config.use_unsloth_gc == true
config.unsloth_gc_recompute_save_on_cpu == true
config.asymm_expert_silu_bwd_gpu == false
config.recomp_off_stage is one of the explicit stage names
```

No model-code changes in Stage 0.

Validation gate before Stage 1:

```text
At s2048, each `recomp-off-*` label resolves to the expected profile config without
running the wrong module family. Existing profile-complete checks reject stale artifacts
where `unsloth_gc_recompute_save_on_cpu=false` or `asymm_expert_silu_bwd_gpu=true`.
```

### Stage 1: baseline matrix and CPUAdamW parity labels

Goal: separate the baseline effects of activation offload, CPU param offload, and CPU
optimizer/SuperOffload paths, so later stages have matching comparison targets.

Add backend labels:

```text
superoffload_mem_nocpuadamw
zero3_offload_mem_nocpuadamw
```

Definition:

```text
*_nocpuadamw:
  offload_param.device = cpu
  offload_optimizer absent/disabled
```

This keeps parameters/LoRA shards CPU-offloaded for parity with the user's concern, but
removes CPU optimizer/CPUAdamW as an artifact axis.

Implementation changes:

- Add DeepSpeed JSONs:
  - `ds_z3_superoffload_mem_nocpuadamw_config.json`
  - `ds_z3_offload_mem_nocpuadamw_config.json`
- Add backend labels to `scripts/lf/run_lf_lora_sft.sh`:
  - backend parsing/case dispatch,
  - `zero_deepspeed_config`,
  - profile backend labels,
  - run validation.
- Add backend labels to both LF profiler wrappers:
  - backend parser,
  - `is_zero_backend`,
  - `backend_gpu_count`,
  - output path labels,
  - existing-profile validation.
- Add profile config fields that prove:
  - `offload_param=cpu`,
  - `offload_optimizer` is absent/disabled,
  - `super_offload` is absent/disabled for the `nocpuadamw` labels.

Naming caveat: if `offload_optimizer.super_offload` is removed, the
`superoffload_mem_nocpuadamw` label is an artifact-clarity compatibility name, not a true
SuperOffload optimizer run. The profile must make this clear.

Run baselines:

```text
no-CPUAdamW / no CPU optimizer-offload:
superoffload_mem_nocpuadamw|unsloth
superoffload_mem_nocpuadamw|unsloth-off
zero3_offload_mem_nocpuadamw|unsloth
zero3_offload_mem_nocpuadamw|unsloth-off

CPUAdamW / CPU optimizer-offload:
superoffload_mem|unsloth
superoffload_mem|unsloth-off
zero3_offload_mem|unsloth
zero3_offload_mem|unsloth-off
```

Validation gate before Stage 2:

```text
Every baseline has a completed profile or a clean failure artifact. The `nocpuadamw`
profiles prove param offload is on and optimizer offload is off. `superoffload_mem` and
`zero3_offload_mem` profiles prove both param and optimizer offload are on. Later Asym
stages must compare only against the matching baseline family for their CPUAdamW state.
```

### Stage 2: Asym backend and CPU-resident base weights only

Goal: isolate Asym base/frozen weight placement and outer `unsloth-off` semantics with
no custom attention/MLP activation placement and no Asym CPUAdamW axis.

Run:

```text
asym|recomp-off-base|ligerloss1
```

Meaning:

```text
outer Unsloth GC
outer save_on_cpu during recompute
Asym base/frozen weights CPU-resident
LoRA trainable weights stable on GPU
no attention activation wrapper
no dense MLP activation wrapper
```

```bash
USE_UNSLOTH_GC=true
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_ATTN_ACT_OFFLOAD=false
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=false
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false
```

Compare to no-CPUAdamW baselines only:

```text
superoffload_mem_nocpuadamw|unsloth|ligerloss1
superoffload_mem_nocpuadamw|unsloth-off|ligerloss1
zero3_offload_mem_nocpuadamw|unsloth|ligerloss1
zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1
```

This stage answers: "What does Asym base-weight residency do before any custom
activation-placement code is involved?"

Validation gate before Stage 3:

```text
Profile proves `recomp_off_stage=base`, attention wrapper count is zero, dense MLP
activation wrapper count is zero, `unsloth_gc_recompute_save_on_cpu=true`, and the peak
is not from `_silu_backward_gpu`.
```

### Stage 3: add attention activation placement only

Goal: isolate attention wrapper effects.

Run:

```text
asym|recomp-off-attn|ligerloss1
```

Enable:

```bash
ASYMM_ATTN_ACT_OFFLOAD=true
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=false
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
```

Keep all dense MLP activation-offload flags off. This stage answers whether the current
attention path improves or worsens memory on top of `unsloth-off`.

Validation gate before Stage 4:

```text
Profile proves `recomp_off_stage=attn`, attention activation modules are installed,
attention saved-tensor wrapper modules are installed where expected, dense MLP activation
wrapper count is zero, and `attn_act_*` counters are nonzero when attention LoRA is in
the target set.
```

### Stage 4: add current dense MLP wrapper only

Goal: measure the existing E=1 expert-based dense MLP wrapper in isolation.

Run:

```text
asym|recomp-off-dense|ligerloss1
```

Enable:

```bash
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=true
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
ASYMM_EXPERT_SILU_BWD_GPU=0
ASYMM_ATTN_ACT_OFFLOAD=false
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
```

This stage answers whether the current dense wrapper is already good enough when the
outer recompute has exact `unsloth-off` coverage and GPU-SiLU is disabled.

Expected artifact checks:

- no `mlp_dense saved_activation` bulk,
- no `_silu_backward_gpu` peak,
- no HBM LoRA-A input for MLP act,
- if it still loses, inspect whether the peak is `gate_up [M,2I]`,
  `stage_concat_columns`, down-LoRA `[M,I]` overlap, or generic saved tensors.

Validation gate before Stage 5:

```text
Profile proves `recomp_off_stage=dense`, dense wrapper installed, attention wrapper
count is zero, `expact_lora_a_forward_cpu_left_grouped_calls > 0`, no
`_silu_backward_gpu` peak, and the artifact tells whether current-wrapper limits such as
`gate_up [M,2I]` or `stage_concat_columns` remain.
```

### Stage 5: current full composition

Goal: measure the best current implementation before adding new dense code.

Run:

```text
asym|recomp-off-full|ligerloss1
```

Enable:

```bash
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_ATTN_ACT_OFFLOAD=true
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
ASYMM_EXPERT_ACT_OFFLOAD=true
ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
ASYMM_EXPERT_SILU_BWD_GPU=0
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false
```

This is the clean version of the composition we thought we had:

```text
unsloth-off + current Asym attention + current Asym dense MLP wrapper
```

If Stage 5 beats the no-CPUAdamW baselines (`superoffload_mem_nocpuadamw|unsloth-off`
and `zero3_offload_mem_nocpuadamw|unsloth-off`) and has clean counters, keep this as the
first working target. If it does not, use Stage 3/4 artifacts to decide which module is
responsible.

Validation gate before Stage 6:

```text
Profile proves `recomp_off_stage=full`, attention and current dense wrappers are both
installed, outer `save_on_cpu` is active, no GPU-SiLU backward branch ran, and the peak
is attributed to a specific remaining module/operation. Only proceed to Stage 6 if the
dense current wrapper is the remaining blocker.
```

### Stage 6: new dense-specific fine-grained MLP path, only if needed

Do this only if Stage 4 or Stage 5 proves the current dense wrapper is the remaining
problem. The new path should not depend on `ASYMM_EXPERT_ACT_OFFLOAD` for dense MLP
dispatch; it should have dense-specific naming and counters.

Suggested flags:

```bash
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1
ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
```

Run:

```text
asym|recomp-off-dense-fg|ligerloss1
asym|recomp-off-full-fg|ligerloss1
```

The rest of this document's "dense-specific MLP" section describes this Stage-6 path.

Validation gate before Stage 7:

```text
Profiles prove `recomp_off_stage=dense-fg` or `full-fg`, no current E=1 dense wrapper is
installed, dense fine-grained counters are nonzero, `stage_concat_columns` counter is
zero, no full `[M,2I]` dense MLP stage appears, and loss/grad checks match the reference
small-shape tests.
```

### Stage 7: CPUAdamW and trainable-weight offload axis

Do this only after activation behavior is understood. This stage checks whether the
Asym CPUAdamW/LoRA weight-offload axis helps or hurts once the activation path is
correct.

Run:

```text
asym_cpuadamwds|recomp-off-full|ligerloss1
asym_cpuadamwds|recomp-off-full-fg|ligerloss1
```

with explicit artifact labels:

```text
gradofffalse_weightofffalse
gradofftrue_weightofftrue
```

This stage must not be used to diagnose activation placement until Stage 2-6 artifacts
are already understood. The profile must prove whether `ASYM_CPU_ADAMW_GRAD_OFFLOAD`
and `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD` are on or off.

Final acceptance comparison:

```text
asym | recomp-off-full[-fg]
  vs superoffload_mem_nocpuadamw | unsloth
  vs superoffload_mem_nocpuadamw | unsloth-off
  vs zero3_offload_mem_nocpuadamw | unsloth-off

asym_cpuadamwds | recomp-off-full[-fg]
  vs superoffload_mem | unsloth
  vs superoffload_mem | unsloth-off
  vs zero3_offload_mem | unsloth-off
```

## Why `unsloth-off + AsymGEMM` is the right fixed composition

Pure `unsloth-off` proves the important baseline mechanism:

- original forward stores only CPU layer inputs,
- recompute forward is wrapped in `torch.autograd.graph.save_on_cpu`,
- generic recompute saved tensors are moved out of HBM.

AsymGEMM should improve on top of that by changing selected recompute subgraphs:

- base weights stay CPU-resident and are read by AsymGEMM kernels,
- selected wide activations are saved as CPU handles in custom autograd Functions,
- LoRA-A can use CPU-left activations,
- backward can stage only the operand needed for the next GEMM.

This is not a replacement for `unsloth-off`. It is `unsloth-off` as a safety net plus
AsymGEMM custom placement where Asym can do better than generic saved-tensor offload.

## Current MLP wrapper is not the final design

Current dense MLP activation offload is installed through:

- `asym_gemm/integrations/lf.py:1992-2021`
- `asym_gemm/training/dense_mlp.py:87-112`

It wraps a dense MLP as a one-expert `AsymQwen3Experts` engine. That was convenient, but
it is not the exact max-memory fine-grained design.

Current problems:

1. Forward uses fused `gate_up_base`.

   Code:

   - `asym_gemm/training/qwen3_moe.py:1037-1046`

   Current behavior:

   ```python
   gate_up = layer.gate_up_base(...)
   gate, up = gate_up.chunk(2, dim=-1)
   ```

   This creates a full `[M,2I]` HBM result before offloading `gate` and `up`.

2. Backward CPU-SiLU path still rebuilds a fused gradient tensor.

   Code:

   - `asym_gemm/training/qwen3_moe.py:1340-1342`
   - `asym_gemm/training/qwen3_moe.py:1429-1439`

   Current behavior:

   ```python
   grad_gate_up = manager.stage_concat_columns(grad_gate_cpu, grad_up_cpu, ...)
   grad_packed = _grouped_base_dx(layer.gate_up_base, grad_gate_up, ...)
   ```

   This stages full `[M,2I]` in HBM and keeps it through multiple consumers.

3. GPU SiLU backward defeats the intended placement.

   Code:

   - `asym_gemm/training/qwen3_moe.py:1324-1327`

   Current behavior when `ASYMM_EXPERT_SILU_BWD_GPU=1`:

   ```python
   grad_gate_up, grad_gate_stage, grad_up_stage = _silu_backward_gpu(...)
   ```

   `_silu_backward_gpu` stages `gate_cpu` and `up_cpu` to GPU and allocates
   `grad_gate_up [M,2I]`. The s16384 forensic artifact peaked at this line. That run is
   not the intended CPU-SiLU memory design.

4. Down LoRA input-gradient currently materializes a separate `[M,I]`.

   Code:

   - `asym_gemm/training/qwen3_moe.py:1255-1261`
   - `asym_gemm/training/qwen3_moe.py:1296-1307`

   Current order computes `grad_down_lora_x [M,I]`, then computes `grad_act [M,I]`,
   then adds. For strict memory, compute the base `grad_act` first and accumulate the
   LoRA input-gradient into it, or use an output-accumulating kernel.

## Stage 6 dense MLP implementation change

Do not start here. First run Stage 0 through Stage 5 with exact configs and validation
artifacts. Add this dense-specific fine-grained MLP wrapper only if Stage 4 or Stage 5
shows that the current E=1 fused expert wrapper is the remaining memory problem.

The purpose of this Stage-6 path is to preserve the user's intended semantics:

```text
outer Unsloth GC recompute saved tensors on CPU
+ AsymGEMM CPU-resident weights
+ MLP recompute forward that stages only the value needed by the next local backward
```

Suggested files:

```text
asym_gemm/training/dense_mlp_finegrained.py
asym_gemm/integrations/lf.py
```

Suggested opt-in flag:

```bash
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1
```

For Stage 6, this flag should supersede the current
`ASYMM_DENSE_MLP_SURGICAL_OFFLOAD` E=1 wrapper. Keep the old wrapper for Stage 4/5
isolation and existing tests.

### Dense wrapper shape

Wrap each dense MLP as:

```text
AsymFinegrainedDenseMLP
  gate_base: AsymFrozenLinear or HostWeight + asym_bf16_cpu_right_matmul
  up_base:   AsymFrozenLinear or HostWeight + asym_bf16_cpu_right_matmul
  down_base: AsymFrozenLinear or HostWeight + asym_bf16_cpu_right_matmul
  gate_lora_A/B, up_lora_A/B, down_lora_A/B
```

Use separate gate/up base weights. Do not concatenate them into a single `gate_up`
weight for the fine-grained mode.

### Forward inside backward recompute

This forward runs only under `torch.enable_grad()` inside the Unsloth GC backward.

Pseudocode:

```python
class _FinegrainedDenseMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_A, gate_B, up_A, up_B, down_A, down_B, module):
        manager = ActivationOffloadManager(pin_memory=True)
        x_2d = x.reshape(-1, H).contiguous()

        x_cpu = manager.offload(x_2d, "mlp.X")

        # gate path: at most one [M,I] output live
        gate = module.gate_base(x_2d)                         # [M,I] HBM
        gate_s = cpu_left_lora_a(x_cpu.tensor, gate_A)         # [M,r]
        gate_delta = gate_s @ gate_B.T                         # [M,I] HBM
        gate.add_(gate_delta * scale)
        gate_cpu = manager.offload(gate, "mlp.gate")
        gate_s_cpu = manager.offload(gate_s, "mlp.S_gate")
        del gate, gate_delta, gate_s

        # up path: same, separate from gate
        up = module.up_base(x_2d)                              # [M,I] HBM
        up_s = cpu_left_lora_a(x_cpu.tensor, up_A)
        up_delta = up_s @ up_B.T
        up.add_(up_delta * scale)
        up_cpu = manager.offload(up, "mlp.up")
        up_s_cpu = manager.offload(up_s, "mlp.S_up")
        del up, up_delta, up_s

        act_cpu = cpu_silu_mul(gate_cpu, up_cpu)               # CPU [M,I]

        # down LoRA-A uses CPU act; only LoRA-B result is HBM
        down_s = cpu_left_lora_a(act_cpu.tensor, down_A)        # [M,r]
        down_delta = down_s @ down_B.T                          # [M,H] HBM
        down_s_cpu = manager.offload(down_s, "mlp.S_down")
        del down_s

        # down base needs one staged [M,I], then release immediately
        act_stage = manager.stage(act_cpu, tag="mlp.act_for_down_base")
        out = module.down_base(act_stage)                       # [M,H] HBM
        manager.release_stage(act_stage, drop_cache=True)
        out.add_(down_delta * scale)
        del down_delta, act_stage

        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_cpu = act_cpu
        ctx.gate_s_cpu = gate_s_cpu
        ctx.up_s_cpu = up_s_cpu
        ctx.down_s_cpu = down_s_cpu
        ctx.module = module
        ctx.save_for_backward(gate_A, gate_B, up_A, up_B, down_A, down_B)
        return out.reshape_as_input_batch
```

Forward invariants:

- no `gate_up [M,2I]`,
- no HBM LoRA-A input for down act,
- no custom ctx HBM activation saves except small/trainable weights,
- any generic saved tensors outside this Function are still covered by outer
  `save_on_cpu`.

### Backward of the dense fine-grained Function

The backward must be scheduled to avoid `[M,2I]` and avoid extra full `[M,I]` overlap.

Pseudocode:

```python
@staticmethod
def backward(ctx, grad_out):
    gate_A, gate_B, up_A, up_B, down_A, down_B = ctx.saved_tensors
    manager = ctx.manager
    module = ctx.module

    grad_out_2d = grad_out.reshape(-1, H).contiguous()

    # Down LoRA weight grads.
    dS_down = grad_out_2d @ down_B                              # [M,r]
    S_down = manager.stage(ctx.down_s_cpu, tag="mlp.S_down_for_dB")
    d_down_B = grad_out_2d.T @ S_down
    manager.release_stage(S_down, drop_cache=True)
    d_down_A = cpu_right_lora_a_grad(dS_down, ctx.act_cpu.tensor)

    # Down input grad. Compute base first, then accumulate LoRA input grad.
    grad_act = module.down_base_dx(grad_out_2d)                  # [M,I]

    # Requirement: avoid a second persistent [M,I].
    # Best: output-accumulating LoRA input-grad kernel:
    #   add_lora_input_grad_(grad_act, dS_down, down_A)
    # Acceptable first implementation only if measured: temp [M,I], add, delete.
    down_lora_dx = dS_down @ down_A                              # [M,I]
    grad_act.add_(down_lora_dx)
    del down_lora_dx, dS_down

    # Fine-grained SiLU backward. The initial implementation stages gate/up
    # separately on GPU, offloads dup/dgate immediately, and never builds
    # grad_gate_up [M,2I]. A later CPU-SiLU variant can be added behind its own
    # flag if the GPU schedule is still the peak owner at larger sequences.
    gate_stage = manager.stage(ctx.gate_cpu, tag="mlp.gate_for_silu_bwd")
    up_stage = manager.stage(ctx.up_cpu, tag="mlp.up_for_silu_bwd")
    grad_up = silu(gate_stage) * grad_act
    grad_up_cpu = manager.offload(grad_up, "mlp.dup")
    del grad_up
    grad_act.mul_(up_stage)
    grad_gate = silu_backward(grad_act, gate_stage)
    del grad_act
    grad_gate_cpu = manager.offload(grad_gate, "mlp.dgate")
    manager.release_stage(gate_stage, drop_cache=True)
    manager.release_stage(up_stage, drop_cache=True)
    manager.release_cpu(ctx.gate_cpu)
    manager.release_cpu(ctx.up_cpu)

    grad_x = None

    # Gate path. Stage only [M,I] for gate; do not concatenate with up.
    grad_gate = manager.stage(grad_gate_cpu, tag="mlp.dgate")
    S_gate = manager.stage(ctx.gate_s_cpu, tag="mlp.S_gate_for_dB")
    dS_gate = grad_gate @ gate_B
    d_gate_B = grad_gate.T @ S_gate
    manager.release_stage(S_gate, drop_cache=True)
    d_gate_A = cpu_right_lora_a_grad(dS_gate, ctx.x_cpu.tensor)
    grad_x = module.gate_base_dx(grad_gate)                      # [M,H]
    gate_lora_dx = dS_gate @ gate_A                              # [M,H]
    grad_x.add_(gate_lora_dx)
    del gate_lora_dx, dS_gate
    manager.release_stage(grad_gate, drop_cache=True)
    manager.release_cpu(grad_gate_cpu)
    manager.release_cpu(ctx.gate_s_cpu)

    # Up path. Same schedule, add into grad_x.
    grad_up = manager.stage(grad_up_cpu, tag="mlp.dup")
    S_up = manager.stage(ctx.up_s_cpu, tag="mlp.S_up_for_dB")
    dS_up = grad_up @ up_B
    d_up_B = grad_up.T @ S_up
    manager.release_stage(S_up, drop_cache=True)
    d_up_A = cpu_right_lora_a_grad(dS_up, ctx.x_cpu.tensor)
    up_dx = module.up_base_dx(grad_up)                            # [M,H]
    grad_x.add_(up_dx)
    del up_dx
    up_lora_dx = dS_up @ up_A                                      # [M,H]
    grad_x.add_(up_lora_dx)
    del up_lora_dx, dS_up
    manager.release_stage(grad_up, drop_cache=True)
    manager.release_cpu(grad_up_cpu)
    manager.release_cpu(ctx.up_s_cpu)

    manager.release_cpu(ctx.x_cpu)
    manager.release_cpu(ctx.act_cpu)
    manager.release_cpu(ctx.down_s_cpu)

    return grad_x.reshape(input_shape), d_gate_A, d_gate_B, d_up_A, d_up_B, d_down_A, d_down_B, None
```

Backward invariants:

- no `grad_gate_up [M,2I]`,
- no old fused `_silu_backward_gpu` / `grad_gate_up [M,2I]` path,
- no `stage_concat_columns`,
- stage `dgate [M,I]` and `dup [M,I]` separately,
- compute base dX separately for gate and up, accumulating into one `[M,H]` `grad_x`,
- release each stage immediately after its local consumers finish.

### Needed helper APIs

The dense fine-grained wrapper can reuse existing pieces, but a few helper APIs should
be made explicit:

1. CPU-right base matmul:

   - Existing: `asym_bf16_cpu_right_matmul`
     (`asym_gemm/training/frozen_linear.py:1097`)
   - Existing dense module: `AsymFrozenLinear`
     (`asym_gemm/training/frozen_linear.py:1859`)

2. CPU-left LoRA-A forward and CPU-right LoRA-A grad:

   - Existing MLP/expert helpers are already used by `qwen3_moe.py`.
   - Expose/import dense wrappers around:
     - `grouped_lora_a_pair_forward_cpu_left`
     - `grouped_lora_a_forward_cpu_left`
     - `grouped_lora_a_pair_grad_cpu_right`
     - `grouped_lora_a_grad_cpu_right`

3. Output-accumulating LoRA input-grad is strongly preferred:

   ```python
   add_lora_input_grad_(dst, dS, A)
   ```

   Without this, down LoRA input grad creates an extra `[M,I]` temporary. That may still
   fit with `unsloth-off` blanket coverage, but it weakens the "stage only on the spot"
   invariant and must be measured.

## Attention path

The attention activation offload wrapper is closer to the intended design than the
current MLP wrapper.

Forward code:

- `attention_activation_offload.py:599-608`: base projection via CPU-resident weight
- `attention_activation_offload.py:612-618`: offload/share `U`
- `attention_activation_offload.py:619-629`: CPU-left LoRA-A, HBM LoRA-B, offload `S`
- `attention_activation_offload.py:632-650`: save only weights and CPU handles

Backward code:

- `attention_activation_offload.py:691-706`: base/lora input grad
- `attention_activation_offload.py:708-728`: `dA` reads CPU `U`
- `attention_activation_offload.py:730-735`: stage small `S` for `dB`
- `attention_activation_offload.py:737-742`: release staged/CPU handles

Required behavior:

- Keep the no-grad guard at `attention_activation_offload.py:979-981`.
- Keep `ASYMM_ATTN_ACT_OFFLOAD=true` only for `recomp-off-attn`,
  `recomp-off-full`, and `recomp-off-full-fg`.
- Keep outer `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true` so anything not covered by the
  custom attention linear Functions still gets generic `unsloth-off` saved-tensor
  offload.

Potential nesting concern:

The outer recompute uses `torch.autograd.graph.save_on_cpu`. Some attention wrappers
also use `saved_tensors_hooks`. This is acceptable as long as tests prove the inner
wrapper does not reintroduce HBM saved tensors. If nesting creates confusing ownership,
prefer the explicit Asym custom Function for the projection and rely on the outer
blanket `save_on_cpu` for the rest of attention/SDPA/norms.

## Weight placement

The AsymGEMM part should save HBM from base weights independently of activation
offload.

Existing conversion path:

- `asym_gemm/integrations/lf.py:1178-1185`: adopt selected base linear weight as
  `HostWeight`
- `asym_gemm/integrations/lf.py:1197-1211`: attention activation-offload LoRA linear
- `asym_gemm/integrations/lf.py:1212-1224`: regular Asym LoRA linear
- `asym_gemm/integrations/lf.py:1225-1230`: frozen Asym linear
- `AsymFrozenLinear.weight_hbm_saved_bytes` reports CPU-resident frozen weight bytes
  (`asym_gemm/training/frozen_linear.py:1983-1985`)

For dense fine-grained MLP, do not create fused host weight for `gate_up`. Adopt/store
separate CPU host weights for:

```text
gate_proj.weight
up_proj.weight
down_proj.weight
```

This avoids `[2I,H]` fused scheduling and lets backward compute gate/up dX separately.

## Do not use these as hidden axes in the target run

For the first correct implementation, these should stay fixed:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
ASYMM_LAYER_ACT_OFFLOAD=false
ASYMM_LAYER_GC=false
ASYMM_ATTN_SDPA_RECOMPUTE=false
ASYM_EXPERT_RECOMPUTE_POLICY=none
ASYMM_MLP_RECOMPUTE_CHUNK=0
```

Reason: this fix is not about optional producer holds or activation recompute policy.
It is about getting the exact `unsloth-off` blanket recompute behavior, then making
AsymGEMM selected submodules use better placement. Extra axes make artifacts hard to
compare.

## Artifact expectations

A correct staged run should show the flags for its exact stage. Do not compare artifacts
unless the command/config proves which stage was actually active.

### Command/config

For every `recomp-off-*` profile, `command.txt`, `train.log`, and `profile.json.config`
must include:

```text
USE_UNSLOTH_GC=true
UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true
ASYMM_EXPERT_SILU_BWD_GPU=0
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
ASYMM_LAYER_ACT_OFFLOAD=false
ASYMM_LAYER_GC=false
ASYMM_ATTN_SDPA_RECOMPUTE=false
ASYM_EXPERT_RECOMPUTE_POLICY=none
recomp_off_stage=<base|attn|dense|full|dense-fg|full-fg>
```

Stage-specific flags:

```text
Stage 1 baselines:
  superoffload_mem_nocpuadamw:
    offload_param=cpu
    offload_optimizer=false
  zero3_offload_mem_nocpuadamw:
    offload_param=cpu
    offload_optimizer=false
  superoffload_mem:
    offload_param=cpu
    offload_optimizer=cpu
    super_offload=true
  zero3_offload_mem:
    offload_param=cpu
    offload_optimizer=cpu

Stage 2 recomp-off-base:
  backend=asym
  ASYMM_ATTN_ACT_OFFLOAD=false
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
  ASYMM_EXPERT_ACT_OFFLOAD=false
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false

Stage 3 recomp-off-attn:
  backend=asym
  ASYMM_ATTN_ACT_OFFLOAD=true
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
  ASYMM_EXPERT_ACT_OFFLOAD=false
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false

Stage 4 recomp-off-dense:
  backend=asym
  ASYMM_ATTN_ACT_OFFLOAD=false
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
  ASYMM_EXPERT_ACT_OFFLOAD=true
  ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false

Stage 5 recomp-off-full:
  backend=asym
  ASYMM_ATTN_ACT_OFFLOAD=true
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=0
  ASYMM_EXPERT_ACT_OFFLOAD=true
  ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false

Stage 6 recomp-off-dense-fg/full-fg:
  backend=asym
  ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1
  ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0
  ASYMM_EXPERT_ACT_OFFLOAD=true only for real MoE experts, not dense MLP routing
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false

Stage 7 CPUAdamW axis:
  backend=asym_cpuadamwds
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=<false|true>
  ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=<false|true>
```

### Memory breakdown

Expected for all `recomp-off-*` stages:

- `Saved-for-backward activations at peak` should be near the `unsloth-off` level, not
  plain `unsloth`.
- No large `mlp_dense saved_activation` row from the recompute graph.
- No exact live `model.layers.N.mlp.engine.lora_dropout [M,I]` caused by HBM LoRA-A
  when `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu`.
- No peak frame at `_silu_backward_gpu` / `qwen3_moe.py:1326`.

Expected for Stage 6 only:

- No peak allocation from `stage_concat_columns`.
- No full `[M,2I]` dense MLP stage in activation stats.

Acceptable:

- Stage 4/5 may still show current-wrapper limits such as fused `gate_up [M,2I]` or
  `stage_concat_columns`; that is exactly what decides whether Stage 6 is needed.
- Stage 6 may stage one `[M,I]` for `act_for_down_base`.
- Stage 6 may stage one `[M,I]` for `dgate` or `dup`, but not both concatenated into
  `[M,2I]`.
- One `[M,H]` or a few `[M,H]` tensors during residual/norm/input grad.
- CPU RSS larger than baseline, as in `unsloth-off`.

### Runtime counters

Expected when the corresponding module is enabled:

```text
recomp_off_stage matches the run label
reference_fallback_count == 0

Stage 3/5/full-fg attention enabled:
  attn_act_lora_a_forward_calls > 0
  attn_act_lora_a_grad_calls > 0

Stage 4/5 current dense enabled:
  expact_lora_a_forward_hbm_grouped_calls == 0
  expact_lora_a_forward_cpu_left_grouped_calls > 0

Stage 6 fine-grained dense enabled:
  dense_mlp_finegrained_forward_calls > 0
  dense_mlp_finegrained_backward_calls > 0
  dense_mlp_finegrained_gate_base_calls > 0
  dense_mlp_finegrained_up_base_calls > 0
  dense_mlp_finegrained_down_base_calls > 0
  dense_mlp_finegrained_stage_concat_columns_calls == 0
  dense_mlp_finegrained_gpu_silu_bwd_calls > 0  # Stage 6a initial GPU schedule
  dense_mlp_finegrained_cpu_silu_bwd_calls == 0
```

## Test plan

### Unit tests

Stage 0-5 should not need new model unit tests unless the harness/config plumbing has
local tests already. Stage 1 needs JSON/config validation tests if the script test
surface exists.

For Stage 6, add tests for `AsymFinegrainedDenseMLP`.

Small shapes:

```text
B=2, S=8, H=128, I=256, r=8
dtype=bf16
lora_dropout=0
```

Check:

- forward output matches PEFT/torch dense MLP within BF16 tolerance,
- gradients for input and all LoRA A/B weights match within tolerance,
- frozen base weights have no grad,
- CPU handles are released after backward,
- `ASYMM_EXPERT_SILU_BWD_GPU=1` is rejected or ignored for this wrapper in max-memory
  mode.

### Validation runs

Every stage must pass a small validation before larger sequences. Use the existing LF
profiling wrappers, not a new launch path:

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_both.sh
```

Default to `profile_lora_lf_test_source.sh` for fast stage gates. Use
`profile_lora_lf_test_both.sh` when the stage needs the full source-plus-breakdown view
or when producing final comparison artifacts. Both wrappers must receive the exact
`RUNS='model ; backend|recompute|liger ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp'`
row and the normal profiler args:

```bash
--gpus 0 --overwrite true|false --plot false
```

For validation gates, set:

```bash
PROFILE_MEMORY_BREAKDOWN=true
PROFILE_MEMORY_SNAPSHOT=true
PROFILE_SYNC=true
```

Do not interpret a run if these env vars or the selected profiler wrapper are missing
from `command.txt` or the profile config.

For each validation run, create a short expected-outcome note before launching:

```text
stage:
run label:
backend family:
CPUAdamW/optimizer-offload family:
expected wrappers:
expected zero counters:
expected nonzero counters:
expected peak owner:
expected comparison baseline:
```

After the run, attach the artifact audit:

```text
completed or partial:
config matches expected:
path label matches config:
peak frame:
top live tensors:
saved-activation summary:
runtime counters:
conclusion:
next action:
```

The `conclusion` line must be one of:

```text
validated
blocked_by_stage_bug
inconclusive_wrong_config
inconclusive_partial_profile
inconclusive_stale_artifact
inconclusive_unexpected_path
```

Do not advance to the next stage on an inconclusive result.

Suggested ladder:

```text
s2048: config/path validation for every new label
s8192: first memory-shape validation
s16384: forensic/source-memory validation before any s50000 attempt
s30000: first real bottleneck workload; required before final claims
s50000: final scoreboard workload; only after s30000 matches expectations
```

Stage validation matrix:

```text
Stage 0:
  Validate all new labels resolve and stale artifacts are rejected.

Stage 1:
  Validate `*_nocpuadamw` profiles prove param-offload-on and optimizer-offload-off.
  Validate normal `superoffload_mem` / `zero3_offload_mem` profiles prove both
  param-offload-on and optimizer-offload-on.

Stage 2:
  Validate asym|recomp-off-base has no attention/dense activation wrappers.

Stage 3:
  Validate asym|recomp-off-attn has attention wrappers/counters only.

Stage 4:
  Validate asym|recomp-off-dense has current dense wrapper/counters only.

Stage 5:
  Validate asym|recomp-off-full combines Stage 3 and Stage 4 only.

Stage 6:
  Validate fine-grained dense unit tests first, then dense-fg/full-fg profiles.

Stage 7:
  Validate asym_cpuadamwds runs only after Stage 5 or 6 is understood, with explicit
  grad/weight-offload labels.
```

Example validation commands after the labels exist:

```bash
RUNS='q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-base|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-attn|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-dense|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-full|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false
```

Primary final comparisons for `asym|...` without CPUAdamW:

Use s30000 first. It is large enough to expose the real activation/matmul bottleneck
without jumping straight to the final s50000 failure surface. Only run s50000 after the
s30000 artifacts are complete and match the expected path.

```bash
RUNS='q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false
```

Then repeat the same comparison at s50000 only after s30000 is clean:

```bash
RUNS='q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; superoffload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; zero3_offload_mem_nocpuadamw|unsloth-off|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym|recomp-off-full-fg|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false
```

If Stage 7 is evaluated:

The user's original scoreboard row uses CPU optimizer/SuperOffload:

```text
q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 50000|8|1 ; none|false|false|false|false|false
```

So the CPUAdamW/SuperOffload family must also be measured at s30000 before s50000 when
evaluating `asym_cpuadamwds|...`:

```bash
RUNS='q3-32b|1 ; superoffload_mem|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; zero3_offload_mem|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite false --plot false

RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=false ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false

RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false' \
  ASYM_CPU_ADAMW_GRAD_OFFLOAD=true ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true \
  bash "${PROFILE_SCRIPT:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf_test_source.sh}" --gpus 0 --overwrite true --plot false
```

Then repeat the CPUAdamW/SuperOffload family at s50000 only after s30000 is clean.

## Success criteria

At the same `q3-32b`, `b8`, `ga1`, `ligerloss1`, real sequence length, every diagnostic
stage must complete or fail with an artifact that proves the exact config and peak
frame. A stage is not allowed to advance on a partial profile unless the partial profile
identifies a clear implementation bug to fix in that stage.

For the final accepted target stage:

1. Target run completes with finite loss.
2. `profile.json.config` proves exact stage flags above.
3. If the target backend is `asym`, peak allocated/reserved HBM is below
   `superoffload_mem_nocpuadamw|unsloth-off` and
   `zero3_offload_mem_nocpuadamw|unsloth-off` at the same sequence length.
4. If the target backend is `asym_cpuadamwds`, peak allocated/reserved HBM is below
   `superoffload_mem|unsloth-off` and `zero3_offload_mem|unsloth-off` at the same
   sequence length.
5. The accepted target is also compared against the matching `unsloth` scoreboard row:
   `superoffload_mem_nocpuadamw|unsloth` for `asym`, or `superoffload_mem|unsloth` for
   `asym_cpuadamwds`.
6. CPU RSS stays within the machine budget.
7. Memory snapshot has no old fused GPU SiLU / `[M,2I]` dense-MLP peak. The Stage
   6a fine-grained GPU SiLU range is allowed, but must not allocate
   `grad_gate_up [M,2I]` or call `stage_concat_columns`.
8. Runtime counters prove the intended Asym paths were used and no reference fallback
   happened.

For memory tables, report both allocator and breakdown numbers:

```text
top_step_H: profile.json.memory.peak_allocated_hbm_bytes
breakdown_H: memory_breakdown.summary.actual_peak_allocated_hbm_bytes
act_H: memory_breakdown.summary.activation_hbm_bytes_at_peak
saved_H: GPU HBM saved_activation rows at breakdown peak
reserved_H: profile.json.memory.peak_reserved_hbm_bytes
RAM: profile.json.memory.process.rss_peak_bytes
```

Use `breakdown_H`, `act_H`, and `saved_H` for activation-placement conclusions. Use
`top_step_H` and `reserved_H` as allocator sanity checks. If these disagree sharply,
the result is not allowed to support a conclusion until the peak owner is explained.

For Stage 6 specifically, also require no `[M,2I]` dense MLP stage and no
`stage_concat_columns` peak.

## Summary by stage

1. Stage 0: harness/config truth.
   - add explicit `recomp-off-*` labels,
   - make every `recomp-off-*` set `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`,
   - make every `recomp-off-*` set and log `ASYMM_EXPERT_SILU_BWD_GPU=0`,
   - add `recomp_off_stage` and profile validation,
   - avoid optional producer/offload-list flags.

2. Stage 1: baseline matrix.
   - add `superoffload_mem_nocpuadamw`,
   - add `zero3_offload_mem_nocpuadamw`,
   - validate param-offload and optimizer-offload state in profiles.

3. Stage 2: Asym backend/base weights only.
   - run `asym|recomp-off-base`,
   - keep attention and dense activation wrappers off,
   - keep Asym CPUAdamW/LoRA weight-offload axis off.

4. Stage 3: attention activation placement only.
   - run `asym|recomp-off-attn`,
   - enable `ASYMM_ATTN_ACT_OFFLOAD=true`,
   - keep dense activation wrappers off.

5. Stage 4: current dense MLP wrapper only.
   - run `asym|recomp-off-dense`,
   - enable `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1`,
   - enable `ASYMM_EXPERT_ACT_OFFLOAD=true`,
   - force `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu`,
   - force `ASYMM_EXPERT_SILU_BWD_GPU=0`,
   - keep attention off.

6. Stage 5: current full composition.
   - run `asym|recomp-off-full`,
   - combine Stage 3 and Stage 4,
   - this is the clean version of the configuration we thought `recomp-off` already was.

7. Stage 6: new dense-specific MLP path, only if needed.
   - add a dense-specific fine-grained MLP custom Function,
   - use separate gate/up/down base weights,
   - no fused `gate_up [M,2I]`,
   - no `stage_concat_columns`,
   - no old fused GPU SiLU backward; Stage 6a may use the fine-grained GPU SiLU
     schedule and a later CPU-SiLU variant must be separately gated,
   - CPU-left LoRA-A for `X` and `act`,
   - stage only one `[M,I]` at a time and release immediately.

8. Stage 7: CPUAdamW/trainable-weight axis.
   - run `asym_cpuadamwds|recomp-off-full[-fg]`,
   - evaluate `gradofffalse_weightofffalse`,
   - evaluate `gradofftrue_weightofftrue`,
   - do not use this to diagnose activation placement before Stage 2-6 are understood.

The design is therefore not "AsymGEMM instead of unsloth-off". It is:

```text
Unsloth original forward + unsloth-off recompute saved-tensor behavior
+ AsymGEMM CPU-resident base weights
+ AsymGEMM custom MLP/attention activation placement
+ optional CPUAdamW/trainable-weight offload only after activation behavior is proven
```

## Implementation Status After Stage 7

The Stage 6/7 implementation uses a dense-specific `AsymFinegrainedDenseMLP` wrapper
instead of the old E=1 dense surgical path. The wrapper owns a custom autograd Function
for the backward recompute schedule, keeps saved HBM activations at zero, uses separate
gate/up/down projections, and must not call `stage_concat_columns`.

One non-obvious requirement for the CPUAdamW/weight-offload family is parent ownership of
dense MLP LoRA weights. The LoRA weight-offload installer must register each
`AsymFinegrainedDenseMLP` as one parent group containing its six LoRA banks, then mark the
`gate_proj`, `up_proj`, and `down_proj` child `AsymLoRALinear` modules as parent-owned.
If the children are registered independently, the dense custom Function can observe
released 0-size LoRA placeholders and fail in CPU-left LoRA-A staging. This is a weight
staging ownership bug, not an activation-placement failure.

Current validated behavior:

```text
Model: qwen3-32b    LoRA: r64/a16/d0.00    CPUAdam: SuperOffload/Asym CPUAdamWDS
Workload   Backend               Config              fwd_s  bwd_s  opt_s  step_s  br_H  saved_H    RAM
---------- --------------------- ------------------- ------ ------ ----- ------- ----- -------- ------
s30000.b8  superoffload_mem      unsloth               35.1   67.4   0.1   102.6 108.7     77.6  340.0
s30000.b8  superoffload_mem      unsloth-off           35.2  201.2   0.1   236.4  66.5      0.0  528.8
s30000.b8  asym_cpuadamwds       recomp-off-full-fg    52.7  310.9   2.5   366.1  57.9      0.0  545.3
s50000.b8  superoffload_mem      unsloth               57.0  219.2   0.1   276.2 180.9    143.8  340.4
s50000.b8  superoffload_mem      unsloth-off           56.7  368.3   0.1   425.1 110.7      0.0  644.4
s50000.b8  asym_cpuadamwds       recomp-off-full-fg   101.0  561.6   2.5   665.2  96.4      0.0  657.7
```

Use the validation log for the full artifact paths and counter audits. The current
memory result is positive: the fixed Asym CPUAdamWDS row is below
`superoffload_mem|unsloth-off` at s30000 and s50000. The current runtime result is not
positive: the fine-grained path is substantially slower and should be optimized only
after preserving the same config/counter invariants.
