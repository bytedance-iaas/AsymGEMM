# Attention Activation Offload Implementation Plan

`agent/attn_math.md` is the source of truth. This plan implements only
projection-side attention activation offload for q/k/v/o LoRA leaves. Do not
change SDPA, FlashAttention, eager attention, RoPE/NoPE, q/k norm, qk_norm,
attention masks, KV-cache semantics, or attention-core backward in v1.

Hard contract:

```text
Target leaves: q_proj, k_proj, v_proj, o_proj
Required gate: ASYMM_ATTN_ACT_OFFLOAD=1
Required wrapper case: component == "attention", selected CPU base offload,
  selected LoRA target, backend == "asym", precision == "bf16"

Profiling comparison axis:
  ASYMM_EXP_ACT_POLICIES items are exactly:
    EXPERT_POLICY|ASYMM_EXPERT_ACT_OFFLOAD|ASYMM_ATTN_ACT_OFFLOAD
  Required comparison entries:
    none|false|false
    gc-exp|false|false
    gc-attn-exp|false|false
    none|true|false
    none|true|true
  gc-exp means selective expert torch checkpointing only.
  gc-attn-exp means selective text-attention torch checkpointing plus
    selective expert torch checkpointing. It must not enable global LF
    gradient_checkpointing.

Per wrapped projection forward:
  1 base CPU-right AsymGEMM:  U @^R W_cpu.T
  1 LoRA-A CPU-left AsymGEMM: U_drop_cpu @^L A.T
  1 LoRA-B HBM GEMM:          S @ B.T

Per wrapped projection backward:
  1 base dx CPU-right AsymGEMM: dZ @^R W_cpu
  1 dS HBM GEMM:                dZ @ B
  1 LoRA input HBM GEMM:        dS @ A
  1 dA CPU-right AsymGEMM:      dS.T @^R U_drop_cpu
  1 dB HBM GEMM with staged S:  dZ.T @ stage(S_cpu)
```

No loops over tokens, rows, heads, KV groups, LoRA rank chunks, row windows, or
experts are allowed. Whole-tensor q/k/v/o projections are the smallest unit.

Reject the implementation if peak HBM is unchanged within noise, the target LF
profile peak drops by less than both 5% and 1 GiB without a credible large-model
projection, step time exceeds 1.25x baseline, AsymGEMM/GEMM counts exceed the
contract above, temporary workspace offsets the memory win, profiling artifacts
cannot separate GC and activation-offload variants, or CPU AdamW stops updating
CUDA LoRA parameters.

## Stage 0 - Profiling Matrix And GC Baseline Wiring

Scope:

```text
Modify:
  scripts/lf/profile_lora_lf.sh
    ASYMM_EXP_ACT_POLICIES default, help, and EXPERT_POLICIES compatibility
    parse_exp_act_policy_pair -> parse_exp_act_policy_tuple
    normalize_expert_policy
    expact_tag plus new attnact_tag
    job_root_path
    legacy_job_root_path
    kt_arm_matching_source_profile_complete
    existing_profile_complete
    run_one_profile run_env/config/run_id
    main ASYMM_EXP_ACT_POLICIES loop

  scripts/lf/run_lf_lora_sft.sh
    ASYMM_ATTN_ACT_OFFLOAD default, bool normalization, tag, logging
    RUN_ENV
    ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD
    ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED

  scripts/lf/run_lf_profiled_train.py
    _config_from_args

  scripts/lf/postprocess_lf_profile_artifacts.py
    summary/config rendering for attention activation offload and attention GC

  asym_gemm/training/moe.py
    parse_expert_recompute_policy_spec

  asym_gemm/integrations/lf.py
    LFAsymReport
    _attention_gc_enabled_for_policy
    _is_text_attention_module_name
    _wrap_attention_checkpoint_modules
    apply_lf_asym_lora
    _infer_adapter_config

Add:
  asym_gemm/training/attention_checkpoint.py
    AttentionCheckpointWrapper
    is_attention_checkpoint_wrapper
    attention_checkpoint_module_names

Add/modify tests:
  tests/lf/test_asym_cpu_adamw_args.py
  tests/lf/test_lf_profile_postprocess.py
  tests/training/test_lf_qwen3_asym_backend.py
```

Implementation steps:

```text
1. Change ASYMM_EXP_ACT_POLICIES from a two-part axis to a three-part axis:
   expert_policy|expert_activation_offload|attention_activation_offload.
   Keep EXPERT_POLICIES backward compatibility by expanding each old policy to
   policy|false|false.
2. Add required accepted tuples:
   none|false|false
   gc-exp|false|false
   gc-attn-exp|false|false
   none|true|false
   none|true|true
3. Keep expert activation offload mutually exclusive with non-none expert
   policies. For v1, keep attention activation offload mutually exclusive with
   gc-exp/gc-attn-exp so GC and activation-offload comparisons stay separable.
4. Add gc-attn-exp to every expert-policy parser. In
   parse_expert_recompute_policy_spec, it must behave like gc-exp for experts:
   policy="gc", torch_checkpoint=True, label="gc-attn-exp".
5. Implement selective attention GC in LF integration. Wrap text attention
   parent modules, not q/k/v/o leaves, with AttentionCheckpointWrapper only
   when expert_recompute_policy == "gc-attn-exp". Do not set global
   GRADIENT_CHECKPOINTING for this policy.
6. Exclude vision and multimodal attention paths before wrapping:
   .vision_model., .vision_tower., .multi_modal_projector., .visual., .vision.
7. Ensure profile path identity, run_id, command.txt, profile config, summary
   markdown, and existing-profile checks include both expact and attnact:
   __expact0__attnact0 / __expact1__attnact1.
8. Add ASYMM_ATTN_ACT_OFFLOAD passthrough to run_lf_lora_sft.sh and record it
   as config["asymm_attn_act_offload"]. Record attention GC as
   config["attention_gc_enabled"].
9. Reject selective GC tuples under BACKEND_SPECS=*|recomp unless an explicit
   developer-only override is set. gc-exp and gc-attn-exp are selective
   policies; pairing them with global LF gradient checkpointing no longer means
   "only experts" or "only attention and experts".
10. Record current attention LoRA wrapping with ASYMM_ATTN_ACT_OFFLOAD unset.
11. Record current q/k/v/o base residency and LoRA parameter residency.
12. Record baseline AsymGEMM counts, peak HBM, and step time for the target LF
   profile shape using the new axis.
13. Confirm the target hidden/output/rank dims satisfy:
   base forward in/out % 8 == 0, base dx out % 64 == 0,
   LoRA-A in/r % 8 == 0, and dA M_grad % 64 == 0.
```

Pseudocode:

```python
def parse_tuple(raw):
    policy, expact, attnact = raw.split("|")
    policy = normalize_expert_policy(policy)
    expact = bool_value(expact)
    attnact = bool_value(attnact)
    if expact and policy != "none":
        fail("expert activation offload requires policy none")
    if attnact and policy != "none":
        fail("attention activation offload is compared without GC in v1")
    return policy, expact, attnact

def apply_lf_asym_lora(..., expert_recompute_policy):
    recompute = parse_expert_recompute_policy_spec(expert_recompute_policy)
    wrap_existing_expert_modules(recompute.label)
    if recompute.label == "gc-attn-exp":
        report.attention_gc_wrapped = wrap_text_attention_modules(model)
```

Resolved by code exploration:

```text
CPU-left LoRA-A exists only as SM100 grouped BF16. Dense attention must call it
as one logical group with offsets=[0,M], experts=[0,-1], A viewed as [1,r,in].
Dense CPU-right BF16 already has single-group launch support through
frozen_linear.py. LF integration is bf16-only.
Existing gc-exp is expert-local torch checkpointing in qwen3_moe.py and friends,
not global LF layer checkpointing. Reusing global GRADIENT_CHECKPOINTING for
gc-attn-exp would checkpoint norms, dense MLP pieces, embeddings, and residual
paths, so it is not the requested baseline.
run_lf_profiled_train.py automatically preserves ASYM_GEMM_LF_CONFIG_* keys, but
the plan still adds explicit config fields so stale-profile checks and summaries
are readable.
```

Risks to watch:

```text
Attention checkpoint wrappers must preserve HF attention kwargs, tuple outputs,
past_key_value/use_cache behavior, and SDPA/FA backend selection.
If attention module inputs have no requires_grad tensors, non-reentrant
checkpoint behavior must still propagate LoRA parameter gradients; test this
with a frozen embedding input.
Model-family attention names can vary. Keep gc-attn-exp strict for known text
families and fail loudly if zero attention modules are wrapped.
Existing profiles without attnact in their path must not be reused for the new
axis.
Selective attention GC is a baseline only. It does not reduce attention-core
temporary work and may increase latency through recompute.
```

Validation before Stage 1:

```bash
python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_three_part_exp_attn_axis_dry_run \
  tests/lf/test_asym_cpu_adamw_args.py::test_profile_lora_lf_rejects_selective_gc_with_global_recomp \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_records_attention_act_offload_config \
  tests/lf/test_lf_profile_postprocess.py::test_profile_config_records_attention_act_and_attention_gc \
  tests/training/test_lf_qwen3_asym_backend.py::test_parse_expert_recompute_policy_gc_attn_exp \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_gc_attn_exp_wraps_text_attention_and_experts_only \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_gc_attn_exp_excludes_vision_attention \
  tests/training/test_lf_qwen3_asym_backend.py::test_lf_offload_module_parser_stage1_contract \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_frozen_base_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_strict_attention_offload_rejects_cuda_source
```

Dry-run matrix validation:

```bash
mkdir -p reports/attn_act_offload
OUTPUT_ROOT=reports/attn_act_offload/stage0_profile_matrix \
ASYM_OFFLOAD_MODULES=all \
scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3-30B-A3B|1' \
  --backend-specs 'asym_cpuadamwds|norecomp' \
  --profilers source \
  --seq-lens 128 \
  --batch-size 1 \
  --max-steps 1 \
  --warmup-steps 0 \
  --max-samples 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.00 \
  --asymm-exp-act-policies 'none|false|false,gc-exp|false|false,gc-attn-exp|false|false,none|true|false,none|true|true' \
  --prepare-datasets false \
  --dry-run true \
  --overwrite true \
  --plot false \
  --plot-memory-breakdown false

grep -R "ASYM_EXPERT_RECOMPUTE_POLICY=gc-attn-exp" \
  reports/attn_act_offload/stage0_profile_matrix
grep -R "ASYMM_ATTN_ACT_OFFLOAD=true" \
  reports/attn_act_offload/stage0_profile_matrix
find reports/attn_act_offload/stage0_profile_matrix -path '*polgc-attn-exp*attnact0*command.txt'
find reports/attn_act_offload/stage0_profile_matrix -path '*polnone*expact1*attnact1*command.txt'
```

Selective-GC profile artifact validation:

```bash
OUTPUT_ROOT=reports/attn_act_offload/stage0_gc_attn_exp_smoke \
ASYM_OFFLOAD_MODULES=all \
scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3-30B-A3B|1' \
  --backend-specs 'asym_cpuadamwds|norecomp' \
  --profilers source \
  --seq-lens 256 \
  --batch-size 1 \
  --max-steps 1 \
  --warmup-steps 1 \
  --max-samples 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.00 \
  --asymm-exp-act-policies 'gc-exp|false|false,gc-attn-exp|false|false' \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-modules attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --profile-level module \
  --profile-sync true \
  --plot false \
  --plot-memory-breakdown false

python - <<'PY'
import json
from pathlib import Path

def flag(value):
    return str(value).lower()

profiles = sorted(Path("reports/attn_act_offload/stage0_gc_attn_exp_smoke").rglob("profile.json"))
assert profiles, "no profile.json files found"
by_policy = {}
for path in profiles:
    data = json.loads(path.read_text())
    cfg = data["config"]
    by_policy[cfg["expert_recompute_policy_spec"]] = cfg
    assert cfg["activation_recompute"] is False, path
    assert flag(cfg["asymm_expert_act_offload"]) == "false", path
    assert flag(cfg["asymm_attn_act_offload"]) == "false", path
assert by_policy["gc-exp"]["expert_recompute_impl"] == "torch_checkpoint"
assert flag(by_policy["gc-exp"]["attention_gc_enabled"]) == "false"
assert by_policy["gc-attn-exp"]["expert_recompute_impl"] == "torch_checkpoint"
assert flag(by_policy["gc-attn-exp"]["attention_gc_enabled"]) == "true"
PY
```

## Stage 1 - Activation GEMM Primitives And Counters

Scope:

```text
Modify:
  asym_gemm/training/frozen_linear.py
    AsymExecutionStats
    asym_bf16_cpu_right_matmul

Add:
  asym_gemm/training/attention_activation_offload.py
    _single_group_offsets_experts
    _dense_lora_a_cpu_left

Add tests:
  tests/training/test_attention_activation_offload_helpers.py
```

Implementation steps:

```text
1. Add AsymExecutionStats fields for attention activation offload:
   attn_act_base_dx_calls
   attn_act_lora_a_forward_calls
   attn_act_lora_a_grad_calls
   attn_act_stage_low_rank_calls
   attn_act_hbm_gemm_calls_by_tag or equivalent shape/call recording
2. Add public asym_bf16_cpu_right_matmul(...) in frozen_linear.py for BF16
   CPU-right operands used by attention activation offload. It wraps the
   existing BF16 CPU-right path, never fp8/fp4 quantized HostWeight paths.
3. Enforce CUDA left, CPU pinned contiguous BF16 right, 2D operands, shape
   checks, and transpose_b constraints. backend="asym" fails loudly.
   backend="torch" is allowed only for focused tests.
4. Add _single_group_offsets_experts(device, M) with cached CUDA int32
   offsets=[0,M] and experts=[0,-1].
5. Add _dense_lora_a_cpu_left(U_drop_cpu, A, stats, tag) that views A as
   [1,r,in] and calls grouped_expert_lora_cpu_left once.
6. Forward LoRA-A is CPU-left only.
```

Pseudocode:

```python
def _dense_lora_a_cpu_left(u_cpu, a, stats, tag):
    offsets, experts = _single_group_offsets_experts(a.device, u_cpu.shape[0])
    weight = a.unsqueeze(0).contiguous()
    out = grouped_expert_lora_cpu_left(
        u_cpu, weight, offsets, experts,
        output_dtype=a.dtype, stats=stats,
    )
    record_attn_act_lora_a_forward(stats, tag, out.shape)
    return out

def asym_bf16_cpu_right_matmul(left_hbm, right_cpu, *, transpose_b, backend, stats, phase, tag):
    validate_bf16_cpu_right(left_hbm, right_cpu, transpose_b)
    return _dispatch_nt(
        left_hbm, right_cpu,
        backend=backend, phase=phase, transpose_b=transpose_b,
        precision="bf16", profile_label=tag,
    )
```

Risks to watch:

```text
CPU-left LoRA-A is SM100-only; non-SM100 tests must assert clear failure or use
backend="torch" reference paths without pretending production support.
_dispatch_nt is private today; expose only the minimal public wrapper needed for
dA so attention code does not depend on private internals.
CPU pinning can be unavailable in CPU-only CI; keep direct-kernel tests skipped
when CUDA/SM100 is unavailable.
```

Validation before Stage 2:

```bash
python -m pytest -q tests/training/test_attention_activation_offload_helpers.py
```

Required helper test cases:

```text
_dense_lora_a_cpu_left(X_cpu, A) matches X @ A.T on SM100 BF16.
asym_bf16_cpu_right_matmul(dS_T, X_cpu, transpose_b=True) matches dS.T @ X.
non-contiguous HBM left is rejected.
unpinned CPU right is rejected under backend="asym".
unsupported dA shape with M_grad % 64 != 0 fails loudly.
no helper test performs row/head/rank/chunk loops.
stats record exact call counts and operand shapes.
```

## Stage 2 - Single Projection Forward

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    AsymActivationOffloadLoRALinear
    _AsymActivationOffloadLoRALinearFunction.forward
    _flatten_last_dim
    _restore_last_dim
    _offload_or_adopt_source_cpu

Use:
  asym_gemm/training/frozen_linear.py
    AsymFrozenLinear
  asym_gemm/training/activation_offload.py
    ActivationOffloadManager

Add/modify tests:
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py
    --mode linear_forward
```

Implementation steps:

```text
1. Match AsymLoRALinear state dict and adapter naming:
   base_layer, lora_A, lora_B, active_adapter, lora_dtype, scaling.
2. Implement from_host_weight(...) with the same base arguments as
   AsymLoRALinear.from_host_weight(...), plus projection_role and optional
   attention_context.
3. Flatten only the last dimension: input [...,in] -> [M,in].
4. Base path calls AsymFrozenLinear and preserves frozen bias.
5. Offload the source U to CPU with ActivationOffloadManager. Do not save HBM U.
6. For p == 0, U_drop_cpu is U_cpu. For p > 0 in this stage, fail with a clear
   NotImplementedError; Stage 4 adds dropout.
7. Run one CPU-left LoRA-A: S = U_drop_cpu @^L A.T.
8. Accumulate LoRA-B into the live base output before offloading S:
   Z += scale * (S @ B.T).
9. Offload S_cpu only after LoRA-B consumes S. The custom Function must receive
   input, A, and B as tensor arguments so backward can return dInput, dA, and dB.
   Save CPU handles, metadata, and A/B via autograd context. Do not save U,
   U_drop, S, base, or delta HBM tensors.
10. Backward may raise NotImplementedError until Stage 3.
```

Pseudocode:

```python
U = flatten_last_dim(input)
Z = base_layer(U)
U_cpu = manager.offload(U, f"{role}.U")
S = _dense_lora_a_cpu_left(U_cpu.tensor, A, stats, f"{role}.lora_a")
Z += scale * (S @ B.T)
S_cpu = manager.offload(S, f"{role}.S")
ctx.save_handles(U_cpu, S_cpu, shape, dtype, role, scale)
return restore_last_dim(Z)
```

Risks to watch:

```text
Flattening a non-contiguous input may materialize an HBM copy; record this in
validation JSON and reject later if it erases memory savings.
Very small M or r may not show memory savings; Stage 2 is correctness only.
Forward q/k/v source sharing is not implemented until Stage 6, so duplicate
CPU source bytes are expected before that stage.
```

Validation before Stage 3:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_forward \
  --device cuda:0 \
  --backend asym \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 13 \
  --compare-to current_asym_lora \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage2_forward.json

ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_linear_forward_matches_current_without_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_linear_forward_saves_no_hbm_source_on_ctx \
  tests/training/test_attention_activation_offload_lora.py::test_linear_forward_launch_counts \
  tests/training/test_attention_activation_offload_lora.py::test_linear_forward_rejects_dropout_until_stage4
```

Advance gate:

```text
Forward diff is within existing BF16 tolerance.
Launch counts are exactly 1 base CPU-right, 1 LoRA-A CPU-left, 1 LoRA-B HBM.
Saved state contains CPU handles and metadata only, not wide HBM source or S.
```

## Stage 3 - Single Projection Backward Without Dropout

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    _AsymActivationOffloadLoRALinearFunction.backward
    _pad_cpu_rows_to
    _pad_hbm_rows_to
    _materialize_low_rank_transpose_for_dA

Modify/add:
  scripts/testing/validate_attention_activation_offload.py
    --mode linear_backward
  tests/training/test_attention_activation_offload_lora.py
```

Implementation steps:

```text
1. Flatten dY to [M,out] and compute base dx explicitly with the base HostWeight:
   dU = asym_bf16_cpu_right_matmul(dY, W_cpu, transpose_b=True, phase="dx").
2. Compute dS = scale * (dY @ B) with one HBM GEMM.
3. Accumulate LoRA input gradient into dU:
   dU += dS @ A.
4. Recompute U_drop_cpu from saved U_cpu. For this stage p == 0, it is U_cpu.
5. Set M_grad = align_up(M, 64). Pad U_drop_cpu rows to [M_grad,in] on CPU.
6. Pad dS rows to [M_grad,r] on HBM, materialize contiguous dS_T [r,M_grad],
   then release dS and the padded dS rows before dB staging.
7. Compute dA = dS_T @^R U_drop_cpu with asym_bf16_cpu_right_matmul(...,
   transpose_b=True, phase="attn_act_dA").
8. Stage S_cpu only for dB, compute dB = scale * (dY.T @ S_stage), then
   release S_stage and S_cpu.
9. Return gradients only for input, A, and B tensor arguments. Frozen base
   weight and bias remain frozen.
```

Risks to watch:

```text
dY.T for dB may create a torch internal temp; record the HBM temp/workspace.
The dA path intentionally materializes only a low-rank contiguous transpose
[r,M_grad]; do not materialize wide transposed activations.
Base dx can fail shape constraints if out % 64 != 0; LF should have filtered or
fallen back before this wrapper is selected.
```

Validation before Stage 4:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_backward \
  --device cuda:0 \
  --backend asym \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 23 \
  --compare-to current_asym_lora \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage3_backward.json

ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_linear_backward_matches_current_without_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_linear_backward_launch_counts \
  tests/training/test_attention_activation_offload_lora.py::test_linear_preserves_frozen_bias \
  tests/training/test_attention_activation_offload_lora.py::test_linear_does_not_grad_base_weight \
  tests/training/test_attention_activation_offload_lora.py::test_linear_releases_cpu_handles_after_backward
```

Advance gate:

```text
dU, dA, and dB match references within BF16 tolerance.
Backward launch counts match the hard contract.
JSON reports M_grad, dS_T shape, S_stage bytes, fallback reasons, and peak HBM.
```

## Stage 4 - LoRA Dropout

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    _cpu_dropout_forward
    _apply_dropout_grad_hbm
    _recompute_dropped_cpu_source

Optionally add only if duplication becomes material:
  asym_gemm/training/dropout.py
    shared packed-mask helpers extracted from qwen3_moe.py

Modify/add:
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py
```

Implementation steps:

```text
1. Support 0 <= p < 1. Keep p == 1 as a clear failure.
2. For p == 0, keep the fast path branch-free.
3. For 0 < p < 1, generate one branch-local CPU mask per projection forward,
   save the exact mask, and apply inverted dropout to U_cpu for LoRA-A.
4. Backward uses the same mask for dU LoRA gradient and dA source recompute.
5. Prefer packed masks or a fused mask kernel for D_bar on HBM. If the first
   implementation stages or expands masks, record those bytes and launch cost.
6. Do not try to match nn.Dropout CUDA RNG. Compare p > 0 only against a masked
   reference built from the saved mask.
```

Risks to watch:

```text
Nonzero dropout can add enough mask memory or latency to fail production gates.
If p > 0 requires expanded HBM masks and the target LF run uses p == 0, keep
p > 0 marked correctness-supported but not performance-accepted until measured.
Extracting shared dropout helpers from qwen3_moe.py can regress expert paths;
prefer local helpers unless duplication blocks maintenance.
```

Validation before Stage 5:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_backward \
  --device cuda:0 \
  --backend asym \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 29 \
  --compare-to masked_reference \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage4_backward_dropout.json

ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_linear_forward_matches_masked_reference_with_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_linear_backward_matches_masked_reference_with_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_linear_rejects_dropout_one \
  tests/training/test_attention_activation_offload_lora.py::test_dropout_mask_bytes_are_reported
```

Advance gate:

```text
Dropout correctness passes for p == 0 and p == 0.1.
Mask bytes and mask-apply launches are visible in JSON.
No extra GEMMs are introduced.
```

## Stage 5 - LF Integration And CPU Adam Contract

Scope:

```text
Modify:
  asym_gemm/integrations/lf.py
    imports
    LFAsymReport
    _attention_act_offload_enabled
    _is_text_attention_projection_name
    _attention_parent_name
    _wrap_lf_linear_leaf
    apply_lf_asym_lora
    _infer_adapter_config

Modify/add:
  tests/training/test_lf_qwen3_asym_backend.py
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Implementation steps:

```text
1. Add ASYMM_ATTN_ACT_OFFLOAD gate. Default unset behavior is unchanged.
2. In _wrap_lf_linear_leaf, choose AsymActivationOffloadLoRALinear only when:
   component == "attention", is_lora_target, selected_cpu_offload,
   backend == "asym", precision == "bf16", and env gate is enabled.
3. Preserve AsymLoRALinear for selected attention LoRA when the gate is unset.
4. Preserve AsymFrozenLinear for selected frozen-only attention projections.
5. Reject or skip known vision paths before generic q/k/v/o matching:
   .vision_model., .vision_tower., .multi_modal_projector., .vision.
6. Keep lora_A/lora_B as CUDA nn.Parameters with the same ModuleDict key names
   as AsymLoRALinear.
7. Add report/config fields:
   attention_act_offload_enabled, attention_act_offload_wrapped,
   attention_act_offload_modules, attention_act_offload_skipped.
8. Add CPU AdamW contract test: optimizer sees CUDA LoRA params, ignores frozen
   CPU base owners, and updates sampled nonzero-gradient LoRA params.
```

Risks to watch:

```text
HF naming for Llama4 text versus vision can vary; keep filters conservative and
strict-mode failures explicit.
State dict compatibility can break PEFT save/load if ModuleDict names differ.
Shape fallback in lf.py currently converts unsupported CPU offload to backend
"torch"; activation offload must not silently claim AsymGEMM savings in that
fallback case.
```

Validation before Stage 6:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_attention_uses_asymgemm

ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_attention_activation_offload_wraps_selected_lora_projection \
  tests/training/test_lf_qwen3_asym_backend.py::test_attention_activation_offload_default_off_is_unchanged \
  tests/training/test_lf_qwen3_asym_backend.py::test_attention_activation_offload_excludes_llama4_vision \
  tests/training/test_lf_qwen3_asym_backend.py::test_attention_activation_offload_report_fields \
  tests/lf/test_asym_cpu_adamw_lf_integration.py::test_cpu_adamw_attention_activation_offload_param_contract
```

Advance gate:

```text
Env unset path is byte-for-byte behaviorally unchanged.
Env set wraps only intended text attention LoRA leaves.
CPU AdamW update health passes.
Adapter state dict keys match AsymLoRALinear conventions.
```

## Stage 6 - Q/K/V Source Sharing

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    AttentionActivationOffloadContext
    AsymActivationOffloadLoRALinear.forward context path

Modify:
  asym_gemm/integrations/lf.py
    _attention_parent_name
    _build_attention_activation_contexts
    apply_lf_asym_lora

Modify/add:
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py
    --mode qkv
```

Implementation steps:

```text
1. Build one AttentionActivationOffloadContext per attention parent only when
   q/k/v are all activation-offload wrappers.
2. Compute source keys before any contiguous materialization:
   device, untyped_storage().data_ptr(), storage_offset(), shape, stride, dtype.
3. Share only X_cpu across q/k/v. Dropout masks and S_q/S_k/S_v handles remain
   branch-local.
4. Clearing the forward lookup after v must not invalidate backward. Each
   autograd node retains its handle reference.
5. If refcounting is not stable, keep the shared CPU handle through the
   attention-layer backward and report retained bytes.
6. Record q/k/v source-sharing hits, misses, duplicate bytes avoided, and
   retained lifetime bytes.
```

Risks to watch:

```text
Model code may pass different views to q/k/v. Misses are allowed, but must be
reported and must not claim sharing savings.
Activation checkpointing can replay forward and disturb context lifetime; test
with checkpointing enabled and disabled.
Sharing must never merge branch-local dropout masks or S_cpu handles.
```

Validation before Stage 7:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode qkv \
  --device cuda:0 \
  --backend asym \
  --hidden-size 128 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 31 \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage6_qkv_share.json

ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_wrappers_share_one_source_handle \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_source_cache_clears_after_v_forward \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_share_backward_keeps_handle_references \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_share_checkpoint_recompute_is_correct
```

Advance gate:

```text
q/k/v share one source handle on matching input storage.
Combined q/k/v backward remains correct.
JSON reports hits, misses, duplicate-source bytes avoided, and retained bytes.
```

## Stage 7 - Full Attention And LF Memory Proof

Scope:

```text
Modify:
  scripts/testing/validate_attention_activation_offload.py
    --mode full_attention
    --mode profile
    --variants current,gc-exp,gc-attn-exp,exp_act_offload,exp_attn_act_offload
    --min-peak-hbm-reduction-ratio
    --min-peak-hbm-reduction-bytes
    --max-step-time-ratio

Modify/add:
  tests/training/test_attention_activation_offload_lora.py
  tests/lf/test_asym_cpu_adamw_lf_integration.py
```

Implementation steps:

```text
1. Validate q/k/v/o inside full attention against the current path for Qwen3
   text and Llama4 text.
2. Reuse the Stage 0 profiling axis. Compare:
   none|false|false as current CPU-base attention/expert baseline,
   gc-exp|false|false as expert-only GC baseline,
   gc-attn-exp|false|false as attention+expert GC baseline,
   none|true|false as expert activation offload,
   none|true|true as expert+attention activation offload.
3. Record peak allocated/reserved HBM, attention saved activations, temporary
   workspace, CPU manager stats, AsymGEMM/GEMM counts, GEMM shapes, source
   sharing counters, and CPU AdamW update health.
4. Check that profile artifacts include expact/attnact state, attention GC
   state, expert policy, global activation_recompute=false for selective GC
   rows, and CPU AdamW enabled.
5. Reject the feature unless target LF memory and latency satisfy the hard
   contract at the top of this file.
```

Risks to watch:

```text
SDPA/FA saved tensors may dominate the peak after projection offload. Report
that separately; do not claim attention-core tensors are removed.
D2H/H2D copies can hurt step time. A latency increase is acceptable only with a
meaningful HBM reduction inside the acceptance gate.
Source-profile attribution can shift bytes from saved activations to temporary
workspace; inspect both before accepting.
gc-attn-exp can reduce HBM but increase recompute latency. It is only a fair
baseline if global activation_recompute remains false.
```

Full-attention validation:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode full_attention \
  --model-family qwen3 \
  --device cuda:0 \
  --backend asym \
  --batch-size 1 \
  --seq-len 128 \
  --hidden-size 256 \
  --num-heads 8 \
  --num-kv-heads 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --attn-implementation sdpa \
  --seed 37 \
  --compare-to current \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage7_qwen3_full_attention.json

ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode full_attention \
  --model-family llama4_text \
  --device cuda:0 \
  --backend asym \
  --batch-size 1 \
  --seq-len 128 \
  --hidden-size 256 \
  --num-heads 8 \
  --num-kv-heads 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --attn-implementation sdpa \
  --seed 41 \
  --compare-to current \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage7_llama4_text_full_attention.json
```

Operator profile gate:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode profile \
  --model-family qwen3 \
  --device cuda:0 \
  --backend asym \
  --batch-size 1 \
  --seq-len 512 \
  --hidden-size 1024 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --attn-implementation sdpa \
  --variants current,gc-exp,gc-attn-exp,exp_act_offload,exp_attn_act_offload \
  --profile-launches true \
  --warmup 5 \
  --iters 10 \
  --min-peak-hbm-reduction-ratio 0.05 \
  --min-peak-hbm-reduction-bytes 1073741824 \
  --max-step-time-ratio 1.25 \
  --output-json reports/attn_act_offload/stage7_profile.json
```

LF CPU AdamW memory proof:

```bash
OUTPUT_ROOT=reports/attn_act_offload/lf_memory \
ASYM_OFFLOAD_MODULES=all \
scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3-30B-A3B|1' \
  --backend-specs 'asym_cpuadamwds|norecomp' \
  --profilers source \
  --seq-lens 4096 \
  --batch-size 1 \
  --max-steps 5 \
  --warmup-steps 5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.00 \
  --asymm-exp-act-policies 'none|false|false,gc-exp|false|false,gc-attn-exp|false|false,none|true|false,none|true|true' \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-modules attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --profile-level module \
  --profile-sync true \
  --plot true \
  --plot-memory-breakdown true

python - <<'PY'
import json
from pathlib import Path

def flag(value):
    return str(value).lower()

profiles = sorted(Path("reports/attn_act_offload/lf_memory").rglob("profile.json"))
assert len(profiles) >= 5, f"expected comparison matrix, found {len(profiles)}"
seen = set()
for path in profiles:
    cfg = json.loads(path.read_text())["config"]
    key = (
        cfg["expert_recompute_policy_spec"],
        flag(cfg["asymm_expert_act_offload"]),
        flag(cfg["asymm_attn_act_offload"]),
    )
    seen.add(key)
    if key[0] in {"gc-exp", "gc-attn-exp"}:
        assert cfg["activation_recompute"] is False, path
        assert cfg["expert_recompute_impl"] == "torch_checkpoint", path
    if key[0] == "gc-attn-exp":
        assert flag(cfg["attention_gc_enabled"]) == "true", path
    if key == ("none", "true", "true"):
        assert flag(cfg["use_asym_cpu_adamw"]) == "true", path
required = {
    ("none", "false", "false"),
    ("gc-exp", "false", "false"),
    ("gc-attn-exp", "false", "false"),
    ("none", "true", "false"),
    ("none", "true", "true"),
}
assert required <= seen, sorted(required - seen)
PY
```

Advance gate:

```text
Full-attention correctness passes for Qwen3 text and Llama4 text.
LF source profile shows meaningful HBM reduction by the hard contract.
Latency ratio is within gate or the feature is rejected.
CPU AdamW update health passes.
attention:saved_activations drops without equivalent temporary/unattributed peak
growth.
Launch counts match the per-projection contract.
Remaining SDPA/FA core saved tensors are reported separately.
```
