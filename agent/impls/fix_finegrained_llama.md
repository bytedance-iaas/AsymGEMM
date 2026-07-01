# Fix Fine-Grained Llama: Dense Llama3.3-70B `recomp-off-full-fg`

## Goal

Compare these rows at the same Llama3.3-70B LoRA-SFT workload:

```text
llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
```

The comparison must be apples-to-apples on the CPUAdamW / optimizer-offload axis:

```text
asym_cpuadamwds|recomp-off-full-fg
  compare against superoffload_mem|unsloth
  compare against superoffload_mem|unsloth-off
  optionally compare against zero3_offload_mem|unsloth-off
```

Do not compare the final `asym_cpuadamwds` Llama row only against
`*_nocpuadamw` baselines. The final scoreboard for this request is the CPUAdamW family.
Use no-CPUAdamW rows only as diagnostics if the activation path is unclear.

The desired final composition is the same composition validated for dense Qwen:

```text
Unsloth whole-layer gradient checkpointing
+ unsloth-off recompute saved-tensor behavior
+ AsymGEMM CPU-resident base/frozen weights
+ dense fine-grained MLP placement
+ selected attention activation placement
+ Asym CPUAdamW / trainable LoRA weight and grad offload
```

## Answer Up Front

There is no accepted, artifact-proven dense Llama3.3-70B
`asym_cpuadamwds|recomp-off-full-fg` path yet.

There is, however, a generic dense fine-grained MLP path in the code:

```text
asym_gemm/training/dense_mlp_finegrained.py
asym_gemm/integrations/lf.py
```

That path is designed around the standard dense gated MLP shape:

```text
gate_proj: nn.Linear
up_proj:   nn.Linear
down_proj: nn.Linear
act_fn:    silu-compatible
```

That is the same structural family used by Qwen2/Qwen2.5/Qwen3 dense and Llama dense
MLPs. So the likely work is not a brand-new dense algorithm. The likely work is:

1. prove the existing generic dense wrapper installs on Llama3.3;
2. prove it fires under the same `recomp-off-full-fg` semantics;
3. prove attention and Liger-loss paths are active for Llama3.3;
4. add Llama-specific tests / validators for the path;
5. only then run the requested s45000 comparison.

Treat dense Qwen3 and dense Qwen2.5 full-fg as the reference behavior, not as proof that
Llama3.3 is already correct. The Llama path must earn its own artifacts.

## Known Working Reference

Dense Qwen reference behavior:

```text
q3-32b|1 ; asym|recomp-off-full-fg|ligerloss1 ; ...
q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; ...
```

The important established properties are:

- `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`.
- `recomp_off_stage=full-fg`.
- `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1`.
- `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=0`.
- `ASYMM_EXPERT_ACT_OFFLOAD=false` for dense fine-grained.
- no old E=1 dense surgical wrapper.
- no `stage_concat_columns`.
- no fused `gate_up [M,2I]` or `grad_gate_up [M,2I]`.
- dense fine-grained counters are nonzero.
- attention counters are nonzero when full-fg enables attention.
- saved HBM activations are `0.0 GiB` in completed backward artifacts.

The Qwen design and validation discipline are in:

```text
agent/impls/fix_finegrained_offload.md
agent/impls/fix_finegrained_offload_validation.md
agent/impls/isolated_testing.md
```

Llama3.3 must follow the same artifact discipline.

## Questions This Plan Must Answer

These are stage gates. Do not skip them.

1. Does `llama3.3-70b` resolve to the intended model and template, and does the HF
   config report a dense `llama` causal LM rather than a MoE family?

2. Does the LF/Asym wrapper see the model as dense, with no expert prefixes and no
   Qwen3-MoE / Qwen3.5 / Llama4 MoE candidates?

3. Does `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1` wrap every Llama3.3 decoder MLP with
   `AsymFinegrainedDenseMLP`?

4. Does the old dense E=1 surgical wrapper stay off?

5. Does `recomp-off-full-fg` enable attention activation placement on Llama3.3
   attention projections and install saved-tensor wrappers on the text attention
   parents?

6. Does the original no-grad Unsloth checkpoint forward remain clean, saving only the
   layer input/root on CPU and not adding fine-grained activation saves?

7. Does the backward recompute forward run under `torch.enable_grad()` and dispatch the
   fine-grained dense MLP custom Function?

8. Does `ligerloss1` really install the Llama loss-only path and, for Asym, the dense
   Asym Liger lm-head bridge?

9. Does `asym_cpuadamwds` prove CPUAdamW, grad offload, and weight offload state in
   `profile.json.config` and logs?

10. At the same `s45000,b8,ga1,ligerloss1` workload, is the Asym row lower HBM than
    `superoffload_mem|unsloth-off`, or is the remaining peak owner named concretely?

## Evidence Discipline

Before each run, write down:

1. expected resolved config:
   - model label and `model_name_or_path`,
   - backend label,
   - internal backend after alias normalization,
   - recompute label and `recomp_off_stage`,
   - `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`,
   - dense fine-grained / dense surgical flags,
   - attention activation flags,
   - Liger loss state,
   - CPUAdamW grad/weight offload state;
2. expected dispatch path:
   - dense MLP wrapper count,
   - attention projection wrapper count,
   - attention saved-tensor wrapper count,
   - dense fine-grained counters that should fire,
   - counters that must stay zero;
3. expected memory shape:
   - whether this is a CPUAdamW final comparison or a no-CPUAdamW diagnostic,
   - whether saved HBM activations should be zero,
   - which live/temp buckets are allowed to dominate;
4. expected failure mode:
   - bad model-family detection,
   - no dense MLP wrappers,
   - Liger skipped,
   - CPUAdamW label mismatch,
   - attention wrapper skipped,
   - OOM in baseline or Asym row,
   - fine-grained unsupported fallback.

After each run, inspect:

```text
command.txt
train.log
profile.json.config
profile.json status/partial fields
source_profile.json
summary.md
memory_breakdown_summary.json
memory_live_activation_details.csv
memory snapshot peak frame, when available
runtime counters
artifact path labels
```

Treat a result as inconclusive if any of these are true:

- profile is partial and the partial artifact does not identify the failure point;
- `profile.json.config.backend` does not match the intended artifact backend label;
- `recomp_off_stage` is missing or not `full-fg` for the Asym final row;
- `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=false` for `recomp-off-full-fg`;
- `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD` is false for `full-fg`;
- `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD` is true for `full-fg`;
- `ASYMM_EXPERT_ACT_OFFLOAD` is true for dense `full-fg`;
- `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD` is true on Llama3.3;
- dense fine-grained wrapper count is zero;
- attention is expected but attention wrapper count is zero;
- Liger loss-only is skipped while the artifact label says `ligerloss1`;
- CPUAdamW grad/weight offload state is not what the row label intends;
- artifact path labels and `profile.json.config` disagree.

If a result deviates strongly from expectation, assume one of these first:

1. stale artifact reuse,
2. wrong model alias,
3. wrong backend alias normalization,
4. wrong recompute alias,
5. missing env forwarding into `RUN_ENV`,
6. profile-complete validator accepted an old artifact,
7. Liger skipped silently,
8. dense MLP matcher missed Llama's module names,
9. attention matcher skipped Llama's attention parent,
10. trainable weight offload changed LoRA gather/release timing.

## Non-Goals And Fixed Constraints

Do not start by adding a new Llama-specific dense MLP algorithm. The existing
fine-grained dense design is already the right first candidate because Llama3.3 is a
dense gated MLP model.

Do not use the old dense surgical wrapper for the final row. That path is the E=1
expert adapter and is known to be the wrong large-sequence path for dense models.

Do not use Qwen3-MoE fine-grained flags on Llama3.3.

Do not add chunked MLP, producer-list policies, layer activation offload, layer GC, or
SDPA recompute while proving this path. Keep the same fixed `recomp-off-full-fg`
semantics used by dense Qwen.

Do not jump straight to s45000 before the Llama3.3 small and medium gates prove module
replacement and counters.

## Current Code Truth

### Model and backend labels

The profiler wrappers already know the user-facing model alias:

```text
llama3.3-70b -> meta-llama/Llama-3.3-70B-Instruct
```

The run script infers:

```text
llama-3* / llama3-* / meta-llama-3* -> template llama3
```

The `asym_cpuadamwds` backend is an artifact/backend label. The actual module
replacement path should still see the internal backend as `asym`:

```text
BACKEND=asym_cpuadamwds
  -> PROFILE_BACKEND_LABEL=asym_cpuadamwds
  -> USE_ASYM_CPU_ADAMW=true
  -> ASYM_CPU_ADAMW_BACKEND=deepspeed
  -> internal BACKEND=asym
```

So do not treat `backend == "asym"` checks in `asym_gemm/integrations/lf.py` as a
blocker for `asym_cpuadamwds`. The profile must prove both:

```text
config.backend = asym_cpuadamwds
internal Asym module replacement happened
```

### Dense MLP path

The generic dense fine-grained wrapper is:

```text
asym_gemm/training/dense_mlp_finegrained.py
```

It expects:

```text
gate_proj: nn.Linear
up_proj:   nn.Linear
down_proj: nn.Linear
lora_dropout = 0.0
bf16 source weights
bf16 CUDA activation input
silu-compatible activation
```

The generic dense matcher is:

```text
asym_gemm/training/dense_mlp.py::is_dense_mlp_module
```

It accepts modules with:

```text
gate_proj, up_proj, down_proj
```

and rejects known MoE blocks:

```text
Qwen3 packed experts
Qwen3 MoE block
Qwen3.5 MoE block
Llama4 MoE block
```

The LF integration enables dense fine-grained wrapping when:

```text
backend == "asym"
not expert_prefixes
ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1
```

This is structurally compatible with dense Llama, but Llama3.3 still needs explicit
tests and E2E artifacts.

### Attention path

The attention activation path is generic over text attention projections:

```text
q_proj
k_proj
v_proj
o_proj
```

For full-fg, expect:

```text
attention_act_offload_wrapped = number of selected text attention projection leaves
attention_saved_tensor_offload_wrapped = number of text attention parent modules
```

For Llama3.3 with LoRA target `all`, the expected count should be:

```text
attention_act_offload_wrapped = 4 * decoder_layer_count
attention_saved_tensor_offload_wrapped = decoder_layer_count
```

Do not hardcode the layer count in validators. Read it from model config or from the
actual module tree and compare against discovered decoder layers.

### Liger path

The local LlamaFactory helper supports loss-only Liger for dense `llama`, and the Asym
Liger bridge has a dense-model bridge for:

```text
qwen2
qwen3
llama
```

That means `ligerloss1` should be possible for Llama3.3. Still, the label alone is not
evidence. A valid Llama3.3 artifact must show:

```text
config.liger_loss = ligerloss1
train.log contains the loss-only applied message, or equivalent profile metadata
asym_liger_lm_head_bridge.enabled = true for the Asym row
asym_liger_lm_head_bridge.model_type = llama
```

If Liger is skipped, the run is a Liger validation failure, not a memory result.

## Expected Llama3.3 `full-fg` Shape

For:

```text
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
```

the resolved config should be:

```text
backend = asym_cpuadamwds
internal Asym backend = asym
use_asym_cpu_adamw = true
asym_cpu_adamw_backend = deepspeed
asym_cpu_adamw_grad_offload = true
asym_cpu_adamw_weight_offload = true
use_unsloth_gc = true
unsloth_gc_recompute_save_on_cpu = true
activation_recompute = true
recomp_off_stage = full-fg
asymm_dense_mlp_finegrained_offload = true
asymm_dense_mlp_surgical_offload = false
asymm_expert_act_offload = false
asymm_qwen3_moe_finegrained_offload = false
asymm_attn_act_offload = true
asymm_layer_act_offload = false
asymm_layer_gc = false
asymm_attn_sdpa_recompute = false
asym_offload_act_recompute = false
asym_offload_x_unpacked = false
asymm_mlp_recompute_chunk = 0
liger_loss = ligerloss1
```

The wrapper truth should be:

```text
dense_mlp_finegrained_offload_wrapped = decoder_layer_count
dense_mlp_act_offload_wrapped = 0
qwen3_moe_finegrained_offload_wrapped = 0
qwen3_experts_wrapped = 0
qwen35_moes_wrapped = 0
llama4_moes_wrapped = 0
attention_act_offload_wrapped = 4 * decoder_layer_count
attention_saved_tensor_offload_wrapped = decoder_layer_count
reference_fallback_count = 0
```

The runtime counters should include:

```text
dense_mlp_finegrained_forward_calls > 0
dense_mlp_finegrained_backward_calls > 0
dense_mlp_finegrained_gate_base_calls > 0
dense_mlp_finegrained_up_base_calls > 0
dense_mlp_finegrained_down_base_calls > 0
dense_mlp_finegrained_stage_concat_columns_calls = 0
attn_act_lora_a_forward_calls > 0
attn_act_lora_a_grad_calls > 0
```

The runtime counters that must stay zero:

```text
qwen3_moe_finegrained_forward_calls = 0
qwen3_moe_finegrained_backward_calls = 0
expact_lora_a_forward_cpu_left_grouped_calls = 0 for the old dense surgical path
```

The memory shape should be:

```text
saved HBM activations = 0.0 GiB
no peak frame in stage_concat_columns
no peak frame in old qwen3_moe dense E=1 surgical path
no full gate_up [M,2I]
no grad_gate_up [M,2I]
peak owner is live activations, temp/workspace, optimizer, or named attention/MLP stage
```

## Needed Designs

This section is the implementation plan. Keep it staged. A later stage is allowed only
after the previous stage has produced either a passing artifact or a concrete blocker.

Do not make a broad Llama fork first. The working hypothesis is:

```text
dense Llama3.3 can reuse the generic dense fine-grained MLP path
```

The implementation work is therefore mostly truth, routing, tests, and validators. Only
change the dense algorithm if the Llama-specific tests prove a real incompatibility.

### Implementation stage map

```text
Impl 0: static code truth
  confirm current files already contain the generic dense pieces

Impl 1: Llama fake-model unit coverage
  add Llama-shaped tests around the existing dense wrapper and LF installer

Impl 2: Llama Liger truth coverage
  prove llama model_type gets loss-only Liger and the Asym dense lm-head bridge

Impl 3: profile config/counter validators
  make stale or wrong Llama full-fg artifacts impossible to accept

Impl 4: s2048 real-model config gate
  run tiny E2E rows and prove wrappers/counters

Impl 5: s8192 and s16384 memory-shape gates
  prove scaling and inspect owners before tuning

Impl 6: s30000 bottleneck gate
  run Asym first, then baselines only if Asym is clean

Impl 7: s45000 final comparison
  compare the three requested CPUAdamW-family rows
```

Stop rule:

```text
If any implementation stage fails, do not skip forward to the next sequence-length gate.
Write the exact blocker into the validation log first.
```

### Files expected to be touched if implementation is needed

These are the expected files for a future implementation pass. This document does not
change code.

```text
tests/training/test_dense_mlp_finegrained.py
  Add Llama-named dense MLP numerical coverage.

tests/training/test_lf_qwen3_asym_backend.py
  Add FakeLlama3 decoder/model coverage for LF wrapper install, dense MLP wrapping,
  attention activation wrapping, and zero Qwen-MoE counters.

tests/lf/test_liger_loss_only_qwen3_moe.py
  Either extend this file or split a new Liger test file so dense llama model_type is
  explicitly covered.

scripts/lf/profile_lora_lf_test_source.sh
scripts/lf/profile_lora_lf_test_both.sh
  Add Llama3.3-specific profile-complete checks if the current generic validation does
  not reject wrong full-fg artifacts.

scripts/lf/run_lf_profiled_train.py
  Add missing profile fields only if final artifacts cannot prove Liger bridge state,
  dense wrapper state, or CPUAdamW state from existing metadata.

asym_gemm/integrations/lf.py
  Touch only if FakeLlama3 proves dense MLP or attention module discovery misses real
  Llama3.3 names.

asym_gemm/training/dense_mlp_finegrained.py
  Touch only if Llama numerical tests expose a real unsupported activation, dtype, LoRA,
  or shape issue.

asym_gemm/integrations/liger_loss.py
  Touch only if Llama dense Asym lm-head bridge proof fails.
```

Files that should not be changed for the first Llama3.3 implementation pass:

```text
asym_gemm/training/qwen3_moe.py
asym_gemm/training/qwen3_moe_finegrained.py
csrc/qwen3/*
```

Those are Qwen3-MoE/routed paths. A dense Llama3.3 fix should not depend on them.

### Implementation stages in detail

#### Impl 0: static code truth

Purpose:

```text
Confirm the existing generic dense path is the starting point.
```

Required audit:

```text
asym_gemm/training/dense_mlp_finegrained.py exists and exposes AsymFinegrainedDenseMLP
asym_gemm/training/dense_mlp.py::is_dense_mlp_module accepts gate/up/down MLPs
asym_gemm/integrations/lf.py enables dense fine-grained only for dense models
scripts/lf/* parse recomp-off-full-fg and set dense_mlp_finegrained=1
scripts/lf/run_lf_lora_sft.sh maps asym_cpuadamwds to internal BACKEND=asym
```

Pass criteria:

```text
The audit explains whether Llama3.3 is blocked by missing implementation or only by
missing validation.
```

Expected answer:

```text
not missing algorithm; missing Llama3.3 validation and possibly small routing fixes
```

#### Impl 1: Llama fake-model unit coverage

Purpose:

```text
Prove the LF integration can discover dense Llama3-style modules without loading 70B.
```

Implementation:

```text
Add FakeLlama3MLP with gate_proj/up_proj/down_proj.
Add FakeLlama3Attention with q_proj/k_proj/v_proj/o_proj.
Add FakeLlama3DecoderLayer with self_attn, mlp, input_layernorm,
post_attention_layernorm.
Add FakeLlama3Model with layers and config.model_type = "llama".
```

Pass criteria:

```text
is_dense_mlp_module(FakeLlama3MLP()) == true
apply_lf_asym_lora(... backend="asym", offload_modules="all", full-fg env ...)
  dense_mlp_finegrained_offload_wrapped == num_layers
  dense_mlp_act_offload_wrapped == 0
  attention_act_offload_wrapped == 4 * num_layers
  attention_saved_tensor_offload_wrapped == num_layers
  qwen3_moe_finegrained_offload_wrapped == 0
```

Stop if:

```text
expert_prefixes becomes nonempty
dense MLP names are missed
attention parent names are missed
the old dense surgical wrapper installs
```

#### Impl 2: Llama dense numerical coverage

Purpose:

```text
Prove the existing fine-grained custom Function is numerically valid for a
Llama-named dense MLP.
```

Implementation:

```text
Reuse build_finegrained_dense_mlp.
Use a Llama-named toy MLP with the same gate/up/down math.
Compare forward, input grad, and all six LoRA grads against eager reference.
Run both default GPU activation backward and CPU activation mode if that mode is kept
supported.
```

Pass criteria:

```text
forward close to eager
input grad close to eager
gate/up/down LoRA-A/B grads close to eager
dense_mlp_finegrained_stage_concat_columns_calls == 0
old dense surgical counters remain zero
```

Stop if:

```text
activation function is not recognized as SiLU
bf16 path rejects a valid Llama tensor
LoRA shapes differ from dense Qwen assumptions
```

Only after this failure should `dense_mlp_finegrained.py` be changed.

#### Impl 3: Llama Liger truth coverage

Purpose:

```text
Make `ligerloss1` a proven behavior, not a label.
```

Implementation:

```text
Add or extend tests so config.model_type = "llama" reaches LlamaFactory loss-only
Liger.
Add Asym dense lm-head bridge coverage for model_type = "llama".
Ensure final profile metadata records bridge enabled/model_type/bridge_kind.
```

Pass criteria:

```text
loss-only Liger applied for dense llama
non-loss Liger patches stay disabled
Asym dense lm-head bridge installs when lm_head is frozen/Asym-compatible
profile can prove bridge state
```

Stop if:

```text
Liger skips because require_logits is true
Liger apply function is unavailable
lm_head has bias/trainable base/unsupported wrapper
```

#### Impl 4: profile validator hardening

Purpose:

```text
Reject wrong Llama3.3 full-fg artifacts before they enter the comparison table.
```

Implementation:

```text
Add Llama3.3-specific expected fields to existing profile-complete checks only where
the generic checks are insufficient.
Record any missing config fields in run_lf_profiled_train.py only if they cannot be
derived from existing profile metadata.
```

Pass criteria:

```text
wrong backend label is rejected
wrong recomp_off_stage is rejected
missing unsloth save_on_cpu is rejected
missing dense fine-grained flag is rejected
old dense surgical flag is rejected
missing Liger proof is rejected
missing CPUAdamW proof is rejected for asym_cpuadamwds
```

Stop if:

```text
the validator only checks artifact path labels and not profile.json.config
the validator accepts a partial profile as a completed backward
```

#### Impl 5: real-model smoke gates

Purpose:

```text
Prove real Llama3.3 module names and runtime counters before long-context runs.
```

Implementation:

```text
Run s2048 base/dense-fg/full-fg serially.
Use one measured step.
Audit command/config/log/counters before reading memory as a result.
```

Pass criteria:

```text
finite loss
profile complete
dense wrappers and attention wrappers match discovered decoder layer count
dense fine-grained counters fire
attention counters fire for full-fg
saved_H = 0.0 GiB for recomp-off rows
```

Stop if:

```text
any wrapper count is zero unexpectedly
any qwen3_moe fine-grained counter fires
Liger proof is absent
CPUAdamW state is absent or wrong
```

#### Impl 6: memory-shape gates

Purpose:

```text
Prove the implementation scales and identify peak owners before the requested
s45000 comparison.
```

Implementation:

```text
Run s8192 dense-fg/full-fg.
Run s16384 superoffload unsloth, superoffload unsloth-off, and Asym full-fg.
Run s30000 Asym first; run baselines only if Asym is clean.
```

Pass criteria:

```text
completed profiles
saved_H = 0.0 GiB for unsloth-off and Asym full-fg
no old dense surgical peak
no stage_concat_columns
peak owner named if Asym is not lower than baseline
```

Stop if:

```text
Asym is partial at s30000
peak owner is unknown
memory breakdown and top-level peak disagree and no explanation is written
```

#### Impl 7: requested final comparison

Purpose:

```text
Answer the exact requested comparison at s45000.
```

Implementation:

```text
Run the three requested rows serially.
Use the CPUAdamW comparison family.
Produce a validation-log section with config audit, counter audit, memory audit, and
interpretation.
```

Pass criteria:

```text
all three rows are complete or the failed row has a clean failure artifact
all three rows prove ligerloss1
Asym row proves asym_cpuadamwds + full-fg + dense/attention counters
comparison uses breakdown_H, saved_H, live_H, temp_H, reserved_H, RAM, and timing
conclusion names validated or exact blocker
```

### 1. Llama dense wrapper test surface

Add Llama-specific tests before trusting E2E artifacts.

Minimum fake module shape:

```text
FakeLlamaMLP
  gate_proj: nn.Linear
  up_proj: nn.Linear
  down_proj: nn.Linear
  act_fn: F.silu or a Llama SiLU activation object

FakeLlamaDecoderLayer
  self_attn with q_proj/k_proj/v_proj/o_proj
  mlp: FakeLlamaMLP
  input_layernorm
  post_attention_layernorm

FakeLlamaModel
  layers: ModuleList[FakeLlamaDecoderLayer]
  config.model_type = llama
```

Required test assertions:

```text
is_dense_mlp_module(FakeLlamaMLP()) is true
apply_lf_asym_lora(... backend=asym, recomp-off-full-fg flags ...)
  wraps every FakeLlamaMLP with AsymFinegrainedDenseMLP
  installs attention activation wrappers on q/k/v/o
  installs attention saved-tensor wrappers on self_attn
  leaves qwen3_moe_finegrained counters/wrappers zero
```

If this test fails, fix the matcher or model-family guard. Do not proceed to E2E.

### 2. Llama dense fine-grained numerical test

The existing dense fine-grained unit tests use a generic Qwen-style toy MLP. Keep those,
but add a Llama-named variant so future changes cannot accidentally make the path
Qwen-only.

Required checks:

```text
forward matches eager Llama MLP reference
input gradient matches eager reference
gate/up/down LoRA-A and LoRA-B grads match eager reference
dense_mlp_finegrained_stage_concat_columns_calls = 0
dense_mlp_finegrained_gpu_silu_bwd_calls or cpu_silu_bwd_calls matches selected flag
no old dense surgical counters fire
```

This should reuse `build_finegrained_dense_mlp`; do not fork the algorithm unless the
test exposes a real Llama-specific incompatibility.

### 3. Llama Liger truth test

Add a small Llama causal-LM test or profile smoke that proves:

```text
model_type = llama
ligerloss1 requested
loss-only Liger path applied
Asym dense lm-head bridge installed when lm_head is Asym/frozen
```

Required profile fields for final E2E:

```text
config.liger_loss = ligerloss1
asym_liger_lm_head_bridge.enabled = true
asym_liger_lm_head_bridge.model_type = llama
asym_liger_lm_head_bridge.bridge_kind = causal_lm
```

If the bridge is not installed, compare `ligerloss0` rows separately. Do not mix
`ligerloss1` labels with skipped Liger behavior.

### 4. Profile validator additions

For Llama3.3 `recomp-off-full-fg`, profile validation should reject stale or wrong
artifacts unless:

```text
config.model_name_or_path contains Llama-3.3-70B or the exact resolved path
config.template = llama3
config.backend = asym_cpuadamwds for the final row
config.recomp_off_stage = full-fg
config.use_unsloth_gc = true
config.unsloth_gc_recompute_save_on_cpu = true
config.asymm_dense_mlp_finegrained_offload = true
config.asymm_dense_mlp_surgical_offload = false
config.asymm_qwen3_moe_finegrained_offload = false
config.asymm_attn_act_offload = true
config.use_asym_cpu_adamw = true
config.asym_cpu_adamw_backend = deepspeed
```

The validator should also check setup/runtime counters:

```text
dense_mlp_finegrained_offload_wrapped > 0
dense_mlp_act_offload_wrapped = 0
attention_act_offload_wrapped > 0
attention_saved_tensor_offload_wrapped > 0
dense_mlp_finegrained_forward_calls > 0
dense_mlp_finegrained_backward_calls > 0
dense_mlp_finegrained_stage_concat_columns_calls = 0
reference_fallback_count = 0
```

### 5. Dense-only fallback diagnostic

If `full-fg` fails because attention wrappers do not support the Llama attention module,
do not abandon dense fine-grained. Run:

```text
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-dense-fg|ligerloss1 ; <seq>|8|1 ; none|false|false|false|false|false
```

Expected:

```text
dense_mlp_finegrained counters nonzero
attention counters zero
saved HBM activations still controlled by outer unsloth-off
```

If dense-fg works but full-fg fails, the missing design is attention-module support, not
the dense MLP path.

### 6. No-grad forward policy

The original checkpoint forward should remain clean. Dense fine-grained has a no-grad
path, but the target semantics still come from Unsloth GC:

```text
original forward:
  save layer input/root on CPU
  run layer under torch.no_grad()
  do not persist MLP/attention internals

backward recompute forward:
  reload layer input
  run under torch.enable_grad()
  enter dense fine-grained custom Function
  offload wide internals explicitly through ActivationOffloadManager
```

If no-grad fine-grained CPU offload is enabled during the original forward, label that
as a separate experiment. It is not the first Llama3.3 acceptance path.

## Stage Plan

### Stage 0: static truth and small tests

Goal: prove that Llama3.3 can use the existing dense path before spending GPU time.

Required artifacts:

```text
pytest test for FakeLlamaMLP dense fine-grained wrapper
pytest test for LF wrapper install on FakeLlamaModel
pytest or smoke proof for Llama Liger loss-only + Asym dense lm-head bridge
```

Expected pass:

```text
FakeLlamaMLP accepted by is_dense_mlp_module
AsymFinegrainedDenseMLP wraps every fake Llama MLP
attention wrappers install on fake Llama self_attn
qwen3_moe_finegrained flags remain false
Liger dense bridge supports model_type=llama
```

Do not run s45000 before Stage 0 passes.

### Stage 1: s2048 config truth

Goal: prove the actual Llama3.3 model resolves to the intended paths.

Run one row at a time:

```text
llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-base|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-dense-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 2048|8|1 ; none|false|false|false|false|false
```

Expected `base`:

```text
dense_mlp_finegrained_offload_wrapped = 0
attention_act_offload_wrapped = 0
unsloth_gc_recompute_save_on_cpu = true
CPUAdamW state is true/true for grad/weight offload
```

Expected `dense-fg`:

```text
dense_mlp_finegrained_offload_wrapped = decoder_layer_count
attention_act_offload_wrapped = 0
dense_mlp_finegrained_forward_calls > 0
dense_mlp_finegrained_backward_calls > 0
stage_concat_columns = 0
```

Expected `full-fg`:

```text
dense_mlp_finegrained_offload_wrapped = decoder_layer_count
attention_act_offload_wrapped = 4 * decoder_layer_count
attention_saved_tensor_offload_wrapped = decoder_layer_count
dense and attention counters nonzero
```

Stage pass criteria:

```text
all rows complete one measured step
finite loss
Liger proof present for all ligerloss1 rows
profile config and artifact label agree
saved HBM activations are zero for unsloth-off / recomp-off rows
```

### Stage 2: s8192 module-shape gate

Goal: prove the path scales past toy sequence length without old dense surgical
behavior.

Run one row at a time:

```text
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-dense-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false
```

Expected shape:

```text
dense-fg lower than base if base was run
full-fg close to dense-fg unless attention helps/hurts materially
superoffload_mem|unsloth-off saved HBM activations = 0.0 GiB
asym full-fg saved HBM activations = 0.0 GiB
old dense surgical counters remain zero
```

If `full-fg` is worse than `dense-fg`, inspect attention peak owners before changing
dense MLP.

### Stage 3: s16384 forensic gate

Goal: check real memory shape while still keeping the run small enough to debug.

Run:

```text
llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false
```

Expected interpretation:

```text
unsloth has large saved HBM activations
unsloth-off removes saved HBM activations and raises host RAM
asym full-fg also has zero saved HBM activations
asym full-fg peak is live/temp/optimizer, not saved activations
```

If Asym is not lower than `unsloth-off` at this stage, do not change code immediately.
Name the peak owner first:

```text
attention live outputs
dense MLP live outputs
LoRA live/workspace
CPUAdamW gather/release
allocator reserved-unallocated
Liger/lm_head
```

### Stage 4: s30000 bottleneck gate

Goal: run the first meaningful long-context Llama3.3 gate before the requested s45000.

Run Asym first:

```text
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
```

Only if it completes cleanly, run baselines:

```text
llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 30000|8|1 ; none|false|false|false|false|false
```

Stage pass criteria:

```text
all compared rows are complete
Asym config/counters remain clean
Liger proof remains present
Asym saved HBM activations = 0.0 GiB
Asym peak is below superoffload_mem|unsloth-off, or the remaining owner is named
```

If Asym fails at s30000, do not run s45000. Debug from the peak/partial artifact.

### Stage 5: requested s45000 comparison

Goal: answer the user's comparison directly.

Run serially:

```text
llama3.3-70b|1 ; superoffload_mem|unsloth|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
llama3.3-70b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
```

Optional parity row:

```text
llama3.3-70b|1 ; zero3_offload_mem|unsloth-off|ligerloss1 ; 45000|8|1 ; none|false|false|false|false|false
```

The final table should include:

```text
model
workload
backend
recompute/config
fwd_s
bwd_s
opt_s
step_s
top_H
breakdown_H
act_H
saved_H
live_H
temp_H
reserved_H
RAM
loss
status
```

Acceptance:

```text
same model, seq, batch, grad accumulation, rank, alpha, dropout, dataset, max steps
same `ligerloss1` proof
same CPUAdamW comparison family
all rows are fresh or explicitly marked reused with matching command/config evidence
no partial artifact used as a completed result
Asym row has clean dense/attention counters
Asym row has zero saved HBM activations
Asym row beats `superoffload_mem|unsloth-off` on breakdown HBM, or the remaining peak
owner is identified and the implementation is marked incomplete
```

## Failure Interpretation Matrix

Use this before changing implementation.

```text
Symptom:
  dense_mlp_finegrained_offload_wrapped = 0
Likely cause:
  Llama MLP matcher missed module shape, expert_prefixes became nonempty, or the env
  flag did not reach LF integration.
Next action:
  fix matcher/env validation; do not tune memory.
```

```text
Symptom:
  dense wrapper installed, but dense_mlp_finegrained_forward_calls = 0
Likely cause:
  wrapper fallback due to dtype/device/dropout/activation unsupported condition.
Next action:
  inspect `_activation_offload_supported` reasons; add explicit unsupported-reason log.
```

```text
Symptom:
  Liger label present, but no Liger applied proof
Likely cause:
  model_type mismatch, require_logits=true, missing Liger apply function, or bridge not
  installed for Asym lm_head.
Next action:
  fix Liger truth first; memory comparison is invalid.
```

```text
Symptom:
  full-fg fails, dense-fg passes
Likely cause:
  attention wrapper incompatibility or attention saved-tensor hook issue.
Next action:
  debug attention separately; keep dense path accepted if dense-fg counters are clean.
```

```text
Symptom:
  saved_H > 0 for recomp-off-full-fg
Likely cause:
  missing outer save_on_cpu, wrong recompute label, stale artifact, or custom Function
  saved raw HBM tensors.
Next action:
  reject artifact and inspect config/counter truth.
```

```text
Symptom:
  Asym HBM worse than unsloth-off but saved_H = 0
Likely cause:
  live/temp/optimizer peak, not saved activation retention.
Next action:
  inspect memory_live_activation_details and snapshot peak frames before changing dense
  placement.
```

```text
Symptom:
  Asym row has high reserved_H but lower allocated/breakdown_H
Likely cause:
  allocator slack / reserved-unallocated bytes.
Next action:
  do not use reserved HBM alone as failure proof.
```

## Final Expected End State

A successful Llama3.3 result should produce a validation log analogous to
`fix_finegrained_offload_validation.md`:

```text
## Llama3.3 Stage 5: s45000 CPUAdamW Final

run labels:
- superoffload_mem|unsloth|ligerloss1
- superoffload_mem|unsloth-off|ligerloss1
- asym_cpuadamwds|recomp-off-full-fg|ligerloss1

config audit:
- all rows model_name_or_path resolve to meta-llama/Llama-3.3-70B-Instruct
- all rows have ligerloss1 applied, not only labeled
- superoffload_mem rows prove param+optimizer/SuperOffload CPU state
- asym row proves Asym CPUAdamW deepspeed backend, grad offload, weight offload
- asym row proves recomp_off_stage=full-fg and unsloth save_on_cpu
- asym row proves dense fine-grained wrappers and attention wrappers installed

counter audit:
- dense_mlp_finegrained_forward_calls > 0
- dense_mlp_finegrained_backward_calls > 0
- dense_mlp_finegrained_stage_concat_columns_calls = 0
- qwen3_moe_finegrained_forward_calls = 0
- attention counters > 0
- reference_fallback_count = 0

memory audit:
- saved_H = 0.0 GiB for unsloth-off and asym full-fg
- no old dense surgical peak frame
- no fused [M,2I] gate_up/grad_gate_up peak
- asym HBM is lower than superoffload_mem|unsloth-off, or remaining owner is named

conclusion:
- validated, or incomplete with exact blocker named
```

Only after this exists should the Llama3.3 row be placed next to the dense Qwen3 /
Qwen2.5 full-fg results as the same class of evidence.
