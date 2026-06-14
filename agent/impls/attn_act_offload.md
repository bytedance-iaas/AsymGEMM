# Attention Activation Offload Implementation Plan

`agent/attn_math.md` is the source of truth. If this plan conflicts with that
file, fix this plan. The goal is to implement projection-side attention LoRA
activation offload without changing SDPA, FlashAttention, eager attention, or
their backward kernels.

## Target

```text
Models:
  Qwen3 / Qwen3-MoE text attention
  Llama4 text attention

Wrapped leaves:
  q_proj, k_proj, v_proj, o_proj

Required conditions:
  component == "attention"
  projection is selected for LoRA
  projection base weight is CPU-resident HostWeight
  backend == "asym"
  precision == "bf16"
  ASYMM_ATTN_ACT_OFFLOAD=1

Unchanged:
  q/k norm
  RoPE/NoPE
  qk_norm
  NoPE temperature
  attention masks
  attention dropout
  KV-cache semantics
  SDPA/FlashAttention/eager attention forward/backward
```

This is a projection-leaf feature. Do not build a monolithic custom attention
autograd Function in v1. The wrapper returns Q, K, V, or final Y; normal
Transformers/PyTorch autograd owns everything between those projection leaves.

The expected memory win over attention-side gradient checkpointing is on the
projection/LoRA side: checkpointing recomputes q/k/v/o branches and rematerializes
projection intermediates, while this design keeps wide projection sources and
low-rank saved values CPU-resident and fetches CPU operands through AsymGEMM.
If the remaining peak is dominated by SDPA/FlashAttention internal state, that
is a separate stage and must be measured separately.

## Existing Code To Reuse

```text
asym_gemm/training/activation_offload.py
  CPUActivationHandle
  ActivationOffloadManager
  ActivationOffloadStats

asym_gemm/training/frozen_linear.py
  AsymFrozenLinear
  HostWeight dispatch and transpose_b dx path

asym_gemm/training/lora.py
  AsymLoRALinear public API, adapter naming, init, state dict conventions

asym_gemm/integrations/lf.py
  parse_lf_offload_modules(...)
  classify_lf_component(...)
  _wrap_lf_linear_leaf(...)
  apply_lf_asym_lora(...)
```

Reuse `ActivationOffloadManager`; do not add a second activation manager.
Extend counters only when the validation artifacts need a missing field.

## Files

Add:

```text
asym_gemm/training/attention_activation_offload.py
tests/training/test_attention_activation_offload_helpers.py
tests/training/test_attention_activation_offload_lora.py
scripts/testing/validate_attention_activation_offload.py
```

Modify:

```text
asym_gemm/training/frozen_linear.py
asym_gemm/integrations/lf.py
scripts/lf/run_lf_lora_sft.sh
scripts/lf/profile_lora_lf.sh
```

## Public CPU-Right Helper

Add this helper in `asym_gemm/training/frozen_linear.py` and use it from the
attention wrapper. Do not call private `_dispatch_nt(...)` directly from
`attention_activation_offload.py`.

```python
def hbm_cpu_matmul(
    left_hbm: torch.Tensor,
    right_cpu: torch.Tensor,
    *,
    transpose_b: bool = False,
    backend: str = "asym",
    stats: AsymExecutionStats | None = None,
    phase: str = "forward",
    compiled_dims: str = "mnk",
    precision: str = "bf16",
    profile_label: str = "",
    bf16_output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor: ...
```

Semantics:

```text
transpose_b=False: left_hbm @ right_cpu.T
  left_hbm [M_left,K] HBM contiguous bf16
  right_cpu [N,K] CPU contiguous pinned bf16
  out [M_left,N] HBM

transpose_b=True: left_hbm @ right_cpu
  left_hbm [M_left,K] HBM contiguous bf16
  right_cpu [K,N] CPU contiguous pinned bf16
  out [M_left,N] HBM
```

Validation rules:

```text
left_hbm must be CUDA, 2D, contiguous
right_cpu must be CPU, 2D, contiguous
right_cpu must be pinned when direct AsymGEMM is required
precision must be bf16
left_hbm and right_cpu must be bf16
shape checks must account for transpose_b
backend="asym" fails loudly on unsupported direct-kernel shapes
backend="torch" may fallback for tests and must be counted
```

Do not route arbitrary CPU activation operands through fp8/fp4 quantized
HostWeight paths. Those paths are for persistent frozen weights, not saved
activation tensors.

## New Module API

Implement `asym_gemm/training/attention_activation_offload.py`:

```python
class AttentionActivationOffloadContext: ...

class AsymActivationOffloadLoRALinear(nn.Module): ...

class _AsymActivationOffloadLoRALinearFunction(torch.autograd.Function): ...
```

Helper functions:

```python
def _flatten_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]: ...
def _restore_last_dim(x_2d: torch.Tensor, input_shape: tuple[int, ...], out_features: int) -> torch.Tensor: ...
def _row_major_from_transposed(x_t: torch.Tensor) -> torch.Tensor: ...
def _pad_cpu_rows_to(handle_or_tensor: CPUActivationHandle | torch.Tensor, rows: int, tag: str) -> CPUActivationHandle: ...
def _pad_hbm_columns_to(x: torch.Tensor, cols: int, tag: str) -> torch.Tensor: ...
def _cpu_dropout_forward(...): ...
def _apply_dropout_grad_hbm(...): ...
```

`AsymActivationOffloadLoRALinear` owns:

```text
base_layer: AsymFrozenLinear
lora_A/lora_B: ModuleDict with the same adapter naming as AsymLoRALinear
active_adapter
lora_dtype
lora_dropout_p
scaling = lora_alpha / rank
projection_role: q_proj | k_proj | v_proj | o_proj
activation_manager: ActivationOffloadManager
attention_context: AttentionActivationOffloadContext | None
```

It must support `from_host_weight(...)` with the same base arguments as
`AsymLoRALinear.from_host_weight(...)`, plus:

```text
projection_role
activation_manager
attention_context
```

The custom Function must receive the input tensor, LoRA A weight, and LoRA B
weight as tensor arguments. Do not only stash A/B on `ctx`; returned `dA` and
`dB` must attach to the trainable parameters.

Frozen base bias, if present, is preserved in forward and remains frozen. The
LF path must not add any non-LoRA trainable parameter.

## Projection Forward

Implement exactly the primitive in `agent/attn_math.md`:

```text
X_2d = flatten_last_dim(X)                              # [M,in] HBM
base = AsymFrozenLinear(X_2d, W_cpu, frozen_bias)        # [M,out] HBM

X_cpu = offload_or_share_source(X_2d)                    # [M,in] CPU
X_drop_cpu, mask_cpu = D(X_cpu)                          # [M,in] CPU

M_fwd = align_up(M, 8)
X_fwd_cpu = pad_rows(X_drop_cpu, M_fwd)                  # [M_fwd,in] CPU
S_T_pad = hbm_cpu_matmul(A, X_fwd_cpu, transpose_b=False)# [r,M_fwd] HBM
S_T = S_T_pad[:, :M]                                     # [r,M] HBM
S = row_major(S_T.T)                                     # [M,r] HBM
S_cpu = offload(S)                                       # [M,r] CPU

delta = scale * (S @ B.T)                                # [M,out] HBM
Y = restore_last_dim(base + delta)
```

Saved state:

```text
CPU source handle
optional CPU dropout mask handle
CPU S handle
input shape/dtype
projection role
scale, p, M, in, out, rank
base HostWeight metadata
LoRA A/B tensors through autograd
```

Do not save these HBM tensors on ctx:

```text
X_2d
X_drop
S
S_T
base
delta
```

## Projection Backward

Implement exactly the primitive in `agent/attn_math.md`:

```text
dY_2d = flatten_last_dim(dY)                             # [M,out] HBM

dX_base = dY_2d @^R W_cpu                                # [M,in] HBM
dS = scale * (dY_2d @ B)                                 # [M,r] HBM
dX_lora_raw = dS @ A                                     # [M,in] HBM
dX_lora = D_bar(dX_lora_raw, saved_mask)                 # [M,in] HBM
dX = dX_base + dX_lora                                   # [M,in] HBM

X_drop_cpu = recompute D(X_cpu)                          # [M,in] CPU
M_grad = align_up(M, 64)
X_grad_cpu = pad_rows(X_drop_cpu, M_grad)                # [M_grad,in] CPU
dS_T = pad_cols(row_major(dS.T), M_grad)                 # [r,M_grad] HBM
dA = hbm_cpu_matmul(dS_T, X_grad_cpu, transpose_b=True)  # [r,in] HBM

S_stage = stage(S_cpu)                                   # [M,r] HBM
dB = scale * (dY_2d.T @ S_stage)                         # [out,r] HBM
release_stage(S_stage)
```

Important details:

```text
row_major(dS.T) must be materialized before hbm_cpu_matmul
dY_2d.T may be a normal torch GEMM operand for dB; no CPU-right dB in v1
dropout backward must use the saved mask, not a new RNG draw
restore dX to the original input shape before returning
return gradients only for input, A, and B tensor arguments
```

## Dropout

Supported range:

```text
0 <= lora_dropout_p < 1
```

For `p == 0`, no mask is saved. For `0 < p < 1`, generate a branch-local CPU
mask for each projection invocation, save the exact mask, and apply inverted
dropout consistently in:

```text
forward LoRA-A input
backward dX_lora
backward dA recompute
```

CPU-generated masks do not need to reproduce the old CUDA RNG stream. They must
preserve inverted-dropout semantics and validate against a masked reference.

Every CPU handle created for masks or padding must preserve the intended HBM
staging device. Do not call `adopt_cpu(...)` for CUDA-bound tensors without
setting the original device metadata needed by `ActivationOffloadManager.stage`.

## Q/K/V Source Sharing

Implement `AttentionActivationOffloadContext` after the single-projection
wrapper is correct. q/k/v wrappers for the same attention module should share
one CPU source activation handle when they receive the same hidden-state tensor.

Cache key, computed before per-leaf contiguous materialization:

```text
(
  x.device,
  x.untyped_storage().data_ptr(),
  x.storage_offset(),
  tuple(x.shape),
  tuple(x.stride()),
  x.dtype,
)
```

Rules:

```text
q/k/v share only the source X_cpu
q/k/v keep independent dropout masks
q/k/v keep independent S_q_cpu/S_k_cpu/S_v_cpu
clearing the forward lookup after v must not invalidate backward
each autograd node must retain its own handle reference
```

If refcounted release is not stable in the first implementation, keep the shared
handle alive for the attention-layer lifetime and report the retained bytes in
JSON. Do not claim final source-sharing memory until the sharing tests pass.

## LF Integration

Gate:

```text
ASYMM_ATTN_ACT_OFFLOAD=1
```

Modify `_wrap_lf_linear_leaf(...)` only for the gated attention-LoRA case:

```text
component == "attention"
is_lora_target
selected_cpu_offload
backend == "asym"
precision == "bf16"
source is text attention, not vision attention
```

Env unset behavior must remain unchanged:

```text
selected attention LoRA + CPU base offload -> AsymLoRALinear
selected attention frozen CPU base only    -> AsymFrozenLinear
selected LoRA without CPU base offload     -> TorchLoRALinear
```

Add helpers in `lf.py`:

```python
def _attention_act_offload_enabled() -> bool: ...
def _is_text_attention_projection_name(name: str) -> bool: ...
def _attention_parent_name(name: str) -> str | None: ...
def _build_attention_activation_contexts(model: nn.Module, selected_names: set[str]) -> dict[str, AttentionActivationOffloadContext]: ...
```

Reject known vision paths before generic `q_proj/k_proj/v_proj/o_proj`
classification can wrap them:

```text
.vision_model.
.vision_tower.
.multi_modal_projector.
.vision.
```

Strict mode should fail clearly if a selected activation-offload target is a
vision attention projection. Non-strict mode may leave it on the existing path
and report it as skipped.

Report fields:

```text
attention_act_offload_enabled
attention_act_offload_wrapped
attention_act_offload_modules
attention_act_offload_skipped
```

## Launch Contract

Per wrapped projection:

```text
Forward:
  1 base AsymFrozenLinear call
  1 CPU-right LoRA-A AsymGEMM call
  1 HBM LoRA-B GEMM

Backward:
  1 base dx AsymGEMM call
  1 HBM dS GEMM
  1 HBM dX-LoRA GEMM
  1 CPU-right dA AsymGEMM call
  1 HBM dB GEMM with staged S_cpu
```

Hard rules:

```text
No Python loops over tokens or rows
No loops over heads or KV groups
No Python loops over experts; any future grouped path must use grouped kernels
No loops over LoRA rank chunks
No row-window staging in v1
No per-row or per-head AsymGEMM calls
No splitting one q/k/v/o projection except whole-tensor padding
No chunked torch fallback loops
```

Allowed loop level:

```text
model layers
projection leaves q_proj/k_proj/v_proj/o_proj
```

## Validation Artifacts

The validation script must support:

```text
linear_forward
linear_backward
qkv
full_attention
profile
```

Every JSON artifact must include:

```text
correctness diffs by tensor and grad
dropout mode and mask metadata
manager CPU-owned bytes, staged bytes, and per-tag peaks
q/k/v source-sharing hit/miss counters
duplicate source bytes when sharing is disabled or missed
base AsymGEMM call counts by projection
LoRA-A AsymGEMM call counts by projection
dA AsymGEMM call counts by projection
LoRA-B/dS/dX-LoRA/dB HBM GEMM call counts by projection
all GEMM input/output shapes
peak allocated HBM
peak reserved HBM
step or operator timing
attention-core saved tensors still present, if any
```

Timing is reported for review; do not encode arbitrary slowdown thresholds.

## Stage Plan And Gates

Do not advance a stage until the previous stage has a pytest result or JSON
artifact. Stages 1-6 implement `agent/attn_math.md`. Stage 7 is optional and
must not start until Stage 6 proves the remaining HBM peak is attention-core
state, not projection/LoRA state.

Reject criteria for any memory stage:

```text
Reject if peak allocated/reserved HBM is unchanged within measurement noise.
Reject if the full-profile peak drops by less than both 5% and 1 GiB on the
target LF shape, unless a smaller model artifact shows the exact projected
large-model byte savings.
Reject if latency rises without a meaningful memory drop.
Reject if end-to-end step time ratio exceeds 1.25x for the target LF profile
unless the run is explicitly marked investigation-only.
Reject if AsymGEMM/GEMM counts exceed the launch contract in this file.
Reject if CPU AdamW no longer sees CUDA LoRA compute parameters or fails to
update sampled LoRA parameters.
```

### Stage 0 Baseline Lock

Scope:

```text
No code changes.
Read/confirm:
  asym_gemm/integrations/lf.py
    parse_lf_offload_modules
    _wrap_lf_linear_leaf
    apply_lf_asym_lora
  asym_gemm/training/lora.py
    AsymLoRALinear
  asym_gemm/training/frozen_linear.py
    AsymFrozenLinear
  asym_gemm/training/cpu_adam.py
    AsymCPUAdamW
```

Implementation steps:

```text
1. Record current attention LoRA wrapping with ASYMM_ATTN_ACT_OFFLOAD unset.
2. Record current base weight residency and CPU AdamW behavior.
3. Record current peak HBM and step time for the Stage 6 target profile shape.
```

Risks to watch:

```text
Baseline attention peak may already be dominated by SDPA/FA internals, limiting
projection-offload wins.
Some target models may not have q/k/v/o as plain nn.Linear leaves.
CPU-first model loading is required for strict attention base offload.
```

Validation before Stage 1:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_lf_offload_module_parser_stage1_contract \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_frozen_base_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_strict_attention_offload_rejects_cuda_source
```

CUDA target validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_attention_uses_asymgemm
```

CPU AdamW validation:

```bash
python -m pytest -q \
  tests/training/test_asym_cpu_adamw.py \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_asym_cpuadamwtorch_args \
  tests/lf/test_asym_cpu_adamw_args.py::test_run_lf_lora_sft_asym_cpuadamwds_args
```

Baseline profile command using existing LF profiling, before the new validation
script exists:

```bash
mkdir -p reports/attn_act_offload
OUTPUT_ROOT=reports/attn_act_offload/stage0_lf_memory \
ASYM_OFFLOAD_MODULES=attention \
scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3-30B-A3B|1' \
  --backend-specs 'asym_cpuadamwds|recomp' \
  --profilers source \
  --seq-lens 4096 \
  --batch-size 1 \
  --max-steps 5 \
  --warmup-steps 5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.00 \
  --asymm-exp-act-policies 'none|false' \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-modules attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --profile-level module \
  --profile-sync true \
  --plot true \
  --plot-memory-breakdown true
```

### Stage 1 CPU-Right Helper And Counters

Scope:

```text
Modify:
  asym_gemm/training/frozen_linear.py
    hbm_cpu_matmul
    AsymExecutionStats
  tests/training/test_attention_activation_offload_helpers.py

Read only unless a missing counter is proven:
  asym_gemm/training/activation_offload.py
    CPUActivationHandle
    ActivationOffloadManager
```

Implementation steps:

```text
1. Add public hbm_cpu_matmul(...) around _dispatch_nt(...).
2. Enforce 2D, contiguous, bf16, CPU-pinned right operand for backend="asym".
3. Keep backend="torch" as a test/debug whole-tensor fallback only.
4. Add stats counters for CPU-right activation matmuls by phase/profile label:
   lora_a_forward, lora_a_dA, torch_fallback.
5. Do not add loops over rows, heads, rank chunks, or experts.
```

Pseudocode:

```python
def hbm_cpu_matmul(left_hbm, right_cpu, *, transpose_b=False, backend="asym", ...):
    validate_public_contract(left_hbm, right_cpu, transpose_b, precision)
    return _dispatch_nt(
        left_hbm,
        right_cpu,
        backend=backend,
        stats=stats,
        phase=phase,
        compiled_dims=compiled_dims,
        transpose_b=transpose_b,
        precision="bf16",
        profile_label=profile_label,
        bf16_output_dtype=bf16_output_dtype,
    )
```

Risks to watch:

```text
_dispatch_nt expects HostWeight-like CPU operands; verify arbitrary pinned CPU
activation tensors satisfy the same layout assumptions.
transpose_b=True requires the reduction dimension to be 64-aligned in the direct
bf16 path; helper must fail loudly before wrong kernels launch.
CPU pinning may be unavailable in CPU-only CI; tests need torch-backend coverage.
```

Validation before Stage 2:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_activation_offload_manager_tracks_cpu_owners_and_stage_reuse \
  tests/training/test_attention_activation_offload_helpers.py
```

Required helper tests:

```text
hbm_cpu_matmul(A_hbm, X_cpu, transpose_b=False) == A @ X.T
hbm_cpu_matmul(dS_T_hbm, X_cpu, transpose_b=True) == dS_T @ X
non-contiguous left_hbm is rejected
non-bf16 operands are rejected
backend="asym" unsupported shapes fail loudly
backend="torch" fallback uses one whole-tensor matmul, not chunks
stage(S_cpu); dY.T @ S_stage matches dB reference
stats record exact helper call counts and shapes
```

### Stage 2 Projection Wrapper Forward

Scope:

```text
Add:
  asym_gemm/training/attention_activation_offload.py
    AsymActivationOffloadLoRALinear
    _AsymActivationOffloadLoRALinearFunction.forward
    _flatten_last_dim
    _restore_last_dim
    _row_major_from_transposed
    _pad_cpu_rows_to
    _cpu_dropout_forward
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py

Use:
  asym_gemm/training/frozen_linear.py
    AsymFrozenLinear
    hbm_cpu_matmul
  asym_gemm/training/activation_offload.py
    ActivationOffloadManager
```

Implementation steps:

```text
1. Match AsymLoRALinear adapter layout and state-dict naming exactly.
2. Forward flattens only the last dimension and restores the original shape.
3. Base path calls AsymFrozenLinear with frozen bias preserved.
4. Offload source activation to CPU; do not save HBM source on ctx.
5. Apply LoRA dropout on CPU and save the exact CPU mask when p > 0.
6. Pad CPU rows to align_up(M, 8), run one LoRA-A CPU-right AsymGEMM.
7. Materialize S = row_major(S_T.T), offload S_cpu, then run one HBM LoRA-B GEMM.
8. Save only CPU handles, metadata, and A/B tensor references for backward.
9. Backward may raise NotImplementedError in this stage.
```

Risks to watch:

```text
CPU dropout RNG will not match nn.Dropout CUDA RNG; compare p > 0 only to a
masked reference using the saved CPU mask.
For small M/r, offloading S_cpu may not reduce peak; this is acceptable only as
an operator proof, not a production acceptance.
Flattening non-contiguous hidden_states may create an HBM contiguous copy; record
this in JSON and handle q/k/v sharing in Stage 5.
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
```

Dropout validation:

```bash
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
  --lora-dropout 0.1 \
  --seed 17 \
  --compare-to masked_reference \
  --profile-launches true \
  --output-json reports/attn_act_offload/stage2_forward_dropout.json
```

Required tests:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_forward_matches_current_without_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_forward_matches_masked_reference_with_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_forward_saves_no_hbm_source_on_ctx \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_forward_launch_counts
```

Advance gate:

```text
forward diff within existing bf16 tolerance
1 base AsymFrozenLinear call, 1 LoRA-A AsymGEMM, 1 LoRA-B HBM GEMM
no row/head/rank/window loops
JSON records S_cpu bytes, source CPU bytes, mask bytes, and HBM temp peak
```

### Stage 3 Projection Wrapper Backward

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    _AsymActivationOffloadLoRALinearFunction.backward
    _apply_dropout_grad_hbm
    _pad_hbm_columns_to
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py
    --mode linear_backward
```

Implementation steps:

```text
1. Compute dX_base with one frozen base dx AsymGEMM.
2. Compute dS = scale * (dY @ B) with one HBM GEMM.
3. Compute dX_lora_raw = dS @ A with one HBM GEMM.
4. Apply D_bar with the saved CPU mask staged only if p > 0.
5. Recompute D(X_cpu) on CPU for dA; do not stage the wide dropped source.
6. Pad rows to align_up(M, 64), materialize row_major(dS.T), pad columns, and
   run one CPU-right dA AsymGEMM.
7. Stage S_cpu and compute dB = scale * (dY.T @ S_stage) with one HBM GEMM.
8. Return gradients only for input, A, and B tensor arguments.
```

Risks to watch:

```text
dY.T in dB may trigger a torch internal materialization; record temp/workspace
and reject later if it erases HBM gains.
Staging dropout masks for D_bar may be large when p > 0; record mask bytes and
consider bit-packing only after correctness.
Base dx transpose_b constraints can force torch fallback for odd shapes; report
fallback counts and do not hide them.
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
```

Dropout backward:

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
  --output-json reports/attn_act_offload/stage3_backward_dropout.json
```

Required tests:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_backward_matches_current_without_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_backward_matches_masked_reference_with_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_rejects_dropout_one \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_preserves_frozen_bias \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_does_not_grad_base_weight \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_backward_launch_counts
```

Advance gate:

```text
dX, dA, dB match references
dB uses staged S_cpu, not CPU-right AsymGEMM
exact backward launch counts match the contract
no frozen base weight grad appears
JSON records fallback reasons and any dY.T materialization/workspace
```

### Stage 4 LF Integration And CPU Adam Contract

Scope:

```text
Modify:
  asym_gemm/integrations/lf.py
    imports
    LFAsymReport
    _attention_act_offload_enabled
    _is_text_attention_projection_name
    _attention_parent_name
    _build_attention_activation_contexts
    _wrap_lf_linear_leaf
    apply_lf_asym_lora
  tests/training/test_lf_qwen3_asym_backend.py
  tests/training/test_attention_activation_offload_lora.py

Read/validate:
  asym_gemm/training/cpu_adam.py
    AsymCPUAdamW
```

Implementation steps:

```text
1. Add the ASYMM_ATTN_ACT_OFFLOAD gate; default false.
2. Wrap only selected text attention LoRA leaves with CPU base offload.
3. Preserve existing AsymLoRALinear behavior when env is unset.
4. Preserve AsymFrozenLinear for selected frozen-only attention projections.
5. Reject or skip Llama4 vision paths before generic q/k/v/o matching.
6. Ensure lora_A/lora_B remain CUDA nn.Parameters with standard names.
7. Add report fields: enabled, wrapped modules, skipped modules, bytes/counters.
8. Add CPU AdamW contract test: AsymCPUAdamW accepts wrapper params and updates
   sampled nonzero-gradient LoRA parameters.
```

Risks to watch:

```text
HF module names for Llama4 text vs vision may vary; keep path filters explicit
and add a strict-mode failure.
CPU AdamW rejects CPU-resident trainable params; wrapper must never move LoRA
A/B parameters to CPU.
State dict compatibility can break adapter save/load if ModuleDict names differ.
```

Validation before Stage 5:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_attention_uses_asymgemm
```

Gated integration tests:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_wraps_selected_lora_projection \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_default_off_is_unchanged \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_rejects_non_asym_backend \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_excludes_llama4_vision \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_report_fields
```

CPU AdamW contract:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_cpu_adam_param_contract \
  tests/training/test_asym_cpu_adamw.py
```

Advance gate:

```text
env unset path unchanged
env set wraps only intended text attention LoRA leaves
optimizer parameter list contains only LoRA params
AsymCPUAdamW update health passes
adapter state dict keys match AsymLoRALinear conventions
```

### Stage 5 Q/K/V Source Sharing

Scope:

```text
Modify:
  asym_gemm/training/attention_activation_offload.py
    AttentionActivationOffloadContext
    AsymActivationOffloadLoRALinear.forward context path
  asym_gemm/integrations/lf.py
    _attention_parent_name
    _build_attention_activation_contexts
  tests/training/test_attention_activation_offload_lora.py
  scripts/testing/validate_attention_activation_offload.py
    --mode qkv
```

Implementation steps:

```text
1. Build one context per attention parent when q/k/v are all wrapped.
2. Cache source by storage key before any contiguous materialization.
3. Share only X_cpu; keep branch-local dropout masks and S_cpu handles.
4. Clear the forward lookup after v forward.
5. Retain CPU handles for backward until the last q/k/v backward user.
6. Record hits, misses, duplicate bytes avoided, and retained lifetime bytes.
```

Risks to watch:

```text
q/k/v may receive different views after model-specific pre-processing; if keys
differ, fall back to per-leaf offload and record duplicate bytes.
ActivationOffloadManager has no built-in refcount; context must own retention or
intentionally keep handles alive to layer backward.
Checkpointing may replay forward and interact with context lifetime; validate
with and without activation recompute.
```

Validation before Stage 6:

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
  --output-json reports/attn_act_offload/stage5_qkv_share.json
```

Required tests:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_wrappers_share_one_source_handle \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_source_cache_clears_after_v_forward \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_share_backward_keeps_own_handle_references \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_share_checkpoint_recompute_is_correct
```

Advance gate:

```text
q/k/v share one source handle on matching input storage
combined q/k/v backward remains correct
JSON reports source-sharing hits and duplicate-source bytes avoided
misses are explicit and do not silently claim memory savings
```

### Stage 6 Full-Attention And LF Memory Proof

Scope:

```text
Modify:
  scripts/testing/validate_attention_activation_offload.py
    full_attention mode
    profile mode
    --min-peak-hbm-reduction-ratio
    --min-peak-hbm-reduction-bytes
    --max-step-time-ratio
  scripts/lf/run_lf_lora_sft.sh
    ASYMM_ATTN_ACT_OFFLOAD passthrough
    profile config/report field
  scripts/lf/profile_lora_lf.sh
    --asymm-attn-act-offload LIST
    attnact0/attnact1 run identity
    post-run config validation
  tests/lf/test_asym_cpu_adamw_args.py
  tests/lf/test_lf_profile_postprocess.py
```

Implementation steps:

```text
1. Validate full attention forward/backward against current path for q/k/v/o.
2. Compare current, attn_act_offload, and recompute variants.
3. Record peak allocated/reserved HBM, attention saved activations, temporary
   workspace, launch counts, GEMM shapes, CPU manager stats, and CPU AdamW health.
4. Add LF script controls and include attnact state in output paths.
5. Reject the feature unless the full target profile shows meaningful HBM
   reduction without launch-count blowup or unacceptable latency.
```

Risks to watch:

```text
SDPA/FA saved tensors may dominate the peak, making projection offload look
small in end-to-end memory even if projection bytes moved correctly.
Extra D2H/H2D copies can hurt latency; reject unless memory drop is meaningful.
Source-profile memory attribution may move bytes from saved activations to
temporary workspace; inspect both before accepting.
CPU AdamW profile must still show sampled LoRA gradients update after optimizer
step.
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
  --output-json reports/attn_act_offload/stage6_qwen3_full_attention.json
```

Llama4 text validation:

```bash
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
  --output-json reports/attn_act_offload/stage6_llama4_text_full_attention.json
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
  --variants current,attn_act_offload,recompute \
  --profile-launches true \
  --warmup 5 \
  --iters 10 \
  --min-peak-hbm-reduction-ratio 0.05 \
  --max-step-time-ratio 1.25 \
  --output-json reports/attn_act_offload/stage6_profile.json
```

LF CPU AdamW memory proof:

```bash
OUTPUT_ROOT=reports/attn_act_offload/lf_memory \
ASYM_OFFLOAD_MODULES=attention \
scripts/lf/profile_lora_lf.sh \
  --models 'Qwen/Qwen3-30B-A3B|1' \
  --backend-specs 'asym_cpuadamwds|recomp' \
  --profilers source \
  --seq-lens 4096 \
  --batch-size 1 \
  --max-steps 5 \
  --warmup-steps 5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.00 \
  --asymm-exp-act-policies 'none|false' \
  --asymm-attn-act-offload false,true \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-modules attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --profile-level module \
  --profile-sync true \
  --plot true \
  --plot-memory-breakdown true
```

Schema validation:

```bash
python scripts/lf/validate_lf_memory_capacity_schema.py \
  --source-profile-json <run_dir>/source_profile.json \
  --memory-breakdown-summary <run_dir>/memory_breakdown_summary.json \
  --require-breakdown
```

Advance gate:

```text
full-attention correctness passes for Qwen3 and Llama4 text
LF source profile shows meaningful HBM reduction by the reject criteria above
latency ratio is within gate or run is rejected
CPU AdamW optimizer update health passes
attention:saved_activations drops without equivalent unattributed/temporary peak growth
launch counts match the per-projection contract
remaining SDPA/FA core saved tensors are reported separately
```

### Stage 7 Scoped Attention-Core Saved-Tensor Hooks

Scope:

```text
Add only if Stage 6 justifies it:
  asym_gemm/training/attention_core_saved_tensor_offload.py
    scoped saved_tensors_hooks utilities
    model-specific attention wrappers only if hook scoping cannot be local
  scripts/testing/validate_attention_activation_offload.py
    attn_act_offload_core_hooks variant

Do not modify:
  FlashAttention kernels
  PyTorch SDPA kernels
```

Implementation steps:

```text
1. Scope saved_tensors_hooks or save_on_cpu around attention_prepare + core only.
2. Exclude q/k/v/o projection wrappers from the hook scope.
3. Pack saved tensors to CPU and unpack to HBM only when autograd requests them.
4. Record hook bytes separately from projection offload bytes.
```

Pseudocode:

```python
with torch.autograd.graph.saved_tensors_hooks(pack_to_cpu, unpack_to_hbm):
    Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)
    AttnOut = attention_core(Q_attn, K_attn, V_attn)
```

Risks to watch:

```text
Hooks may break fused attention assumptions or add synchronization.
Saved tensor hooks can increase latency sharply for FA kernels.
Hooking too broad a scope can offload projection tensors twice.
```

Validation before Stage 8:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
ASYMM_ATTN_CORE_SAVE_ON_CPU=1 \
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
  --variants current,attn_act_offload,attn_act_offload_core_hooks \
  --profile-launches true \
  --warmup 5 \
  --iters 10 \
  --min-peak-hbm-reduction-ratio 0.05 \
  --max-step-time-ratio 1.25 \
  --output-json reports/attn_act_offload/stage7_core_hooks_profile.json
```

Advance gate:

```text
correctness still matches Stage 6
hook bytes and projection bytes are separated in JSON
remaining HBM peak drops meaningfully
latency gate passes
no FA/SDPA kernel patch exists
```

### Stage 8 Async Copies And Buffer Pools

Scope:

```text
Modify only after Stage 6/7 synchronous correctness:
  asym_gemm/training/activation_offload.py
    optional copy streams/events
    pinned CPU pool controls
    HBM staging pool controls
  asym_gemm/training/attention_activation_offload.py
    stream/event waits at wrapper boundaries if needed
  scripts/testing/validate_attention_activation_offload.py
    async profiling counters
```

Implementation steps:

```text
1. Add optional async D2H/H2D paths guarded by an env/config flag.
2. Use CUDA events so CPU buffers are not reused before GPU reads complete.
3. Reuse pinned CPU and HBM staging buffers by shape/dtype/tag.
4. Add mask packing only if p > 0 mask bytes are material in Stage 6 artifacts.
```

Risks to watch:

```text
Incorrect event ordering can race CPU buffer reuse against AsymGEMM reads.
Async copies can increase reserved HBM through larger staging pools.
Pinned CPU pool growth can hurt host memory pressure and CPU AdamW state.
```

Validation:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
ASYMM_ATTN_ACT_OFFLOAD_ASYNC=1 \
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
  --variants attn_act_offload,attn_act_offload_async \
  --profile-launches true \
  --warmup 5 \
  --iters 10 \
  --max-step-time-ratio 1.10 \
  --output-json reports/attn_act_offload/stage8_async_profile.json
```

Acceptance:

```text
same correctness as Stage 6/7
step time improves or stays within 1.10x of synchronous offload
reserved HBM does not grow enough to erase the memory win
CPU process memory including CPU AdamW state remains within expected host budget
```
