# Attention Activation Offload for AsymGEMM LoRA SFT

This is the concrete implementation plan for `agent/attn_math.md`.

Target scope:

```text
Models:
  Qwen3 / Qwen3-MoE text attention
  Llama4 text attention

Weights:
  frozen q/k/v/o base weights on CPU HostWeight
  trainable LoRA A/B weights on HBM

Activation offload:
  projection and LoRA activations around q_proj/k_proj/v_proj/o_proj

Unchanged in v1:
  q/k norm, RoPE/NoPE, temperature scaling, KV cache update, masks,
  SDPA/FlashAttention/eager attention forward and backward
```

Do not modify FA/SDPA kernels in this plan. Do not build a monolithic custom
attention autograd function in v1. The custom boundary is the projection leaf:
the wrapper returns `Q`, `K`, `V`, or final `Y`, and normal Transformers/PyTorch
autograd owns everything between those projection leaves.

## Current Code Facts

Existing activation manager:

```text
asym_gemm/training/activation_offload.py
  CPUActivationHandle
  ActivationOffloadStats
  ActivationOffloadManager
    empty_cpu(...)
    offload(...)
    adopt_cpu(...)
    stage(...)
    stage_concat_columns(...)
    release_stage(...)
    release_cpu(...)
    snapshot(...)
```

Reuse this manager. Do not add a second activation-offload manager.

Existing attention weight-offload path:

```text
asym_gemm/integrations/lf.py
  parse_lf_offload_modules(...)
  classify_lf_component(...)
  _wrap_lf_linear_leaf(...)
  apply_lf_asym_lora(...)
```

`_wrap_lf_linear_leaf(...)` already converts selected attention leaves to:

```text
AsymLoRALinear    when component == attention, selected for LoRA, and CPU base offload
AsymFrozenLinear  when component == attention, not selected for LoRA, and CPU base offload
TorchLoRALinear   when selected for LoRA but not CPU base offload
```

`asym_gemm/training/lora.py::AsymLoRALinear.forward(...)` still saves LoRA
inputs through normal PyTorch autograd:

```python
base = self.base_layer(x)
lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
lora = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_input)) * self.scaling
return base + lora.to(dtype=base.dtype)
```

The new path replaces this dense attention LoRA wrapper only when all of these
are true:

```text
component == "attention"
is_lora_target == True
selected_cpu_offload == True
backend == "asym"
ASYMM_ATTN_ACT_OFFLOAD=1
```

The existing expert activation-offload env is `ASYMM_EXPERT_ACT_OFFLOAD`.
Use `ASYMM_ATTN_ACT_OFFLOAD` for this feature to match that naming style.

## Implementation Files

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

Only extend `asym_gemm/training/activation_offload.py` if the existing manager
is missing a required counter or tiny utility. Do not change its basic ownership
model in the attention implementation stages.

## Core Symbols to Add

In `asym_gemm/training/attention_activation_offload.py`:

```python
class AsymActivationOffloadLoRALinear(nn.Module): ...
class _AsymActivationOffloadLoRALinearFunction(torch.autograd.Function): ...
class AttentionActivationOffloadContext: ...
```

Helper functions in the same file:

```python
def _flatten_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]: ...
def _restore_last_dim(x_2d: torch.Tensor, input_shape: tuple[int, ...], out_features: int) -> torch.Tensor: ...
def _cpu_dropout_forward(manager: ActivationOffloadManager, handle: CPUActivationHandle, p: float, tag: str) -> tuple[CPUActivationHandle, CPUActivationHandle | None]: ...
def _apply_dropout_grad_hbm(grad: torch.Tensor, mask_handle: CPUActivationHandle | None, p: float, manager: ActivationOffloadManager, tag: str) -> torch.Tensor: ...
def _row_major_from_transposed(result_t: torch.Tensor) -> torch.Tensor: ...
def _contiguous_2d_left(x: torch.Tensor, tag: str) -> torch.Tensor: ...
def _pad_cpu_rows_to(handle_or_tensor: CPUActivationHandle | torch.Tensor, padded_rows: int, tag: str) -> CPUActivationHandle: ...
def _pad_hbm_columns_to(x: torch.Tensor, padded_cols: int, tag: str) -> torch.Tensor: ...
```

`AsymActivationOffloadLoRALinear` owns:

```text
base_layer: AsymFrozenLinear
lora_A/lora_B: same trainable CUDA params and adapter naming as AsymLoRALinear
lora_dropout_p: float
scaling: float
projection_role: q_proj | k_proj | v_proj | o_proj
activation_manager: ActivationOffloadManager
attention_context: AttentionActivationOffloadContext | None
```

It should support `from_host_weight(...)` with the same arguments as
`AsymLoRALinear.from_host_weight(...)`, plus `projection_role`,
`activation_manager`, and `attention_context`.

`_AsymActivationOffloadLoRALinearFunction.apply(...)` must receive the input,
LoRA A weight, and LoRA B weight as tensor arguments. Do not only stash A/B on
`ctx`, or autograd will not attach returned `dA`/`dB` to the trainable
parameters.

Projection bias must be preserved by passing the frozen bias into
`AsymFrozenLinear.from_host_weight(...)`. Bias remains frozen. Backward returns
the normal frozen-bias gradient only if the existing `AsymFrozenLinear` path
requires it; the LF target should still leave no non-LoRA trainable params.

Exclude Llama4 vision attention in v1. Only wrap text attention leaves whose
module path belongs to decoder/text layers. If target matching could hit vision
`q_proj/k_proj/v_proj/o_proj`, leave those modules unchanged or fail in strict
mode with a clear message.

## Required AsymGEMM Helper

Add a public CPU-right helper to `asym_gemm/training/frozen_linear.py`:

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

This helper wraps `_dispatch_nt(...)` and performs public validation:

```text
left_hbm must be CUDA, 2D, contiguous
right_cpu must be CPU, 2D, contiguous, pinned when direct AsymGEMM requires it
precision must be bf16 for this activation-CPU-right helper
left_hbm/right_cpu must both be bf16
shape checks must account for transpose_b
fallback behavior must be explicit and match the existing backend policy
```

Do not use fp8/fp4 paths for arbitrary CPU activation operands in this helper.
Those paths are built around `QuantizedHostWeight`, not saved CPU activations.

Helper semantics must match current `_dispatch_nt(...)`:

```text
transpose_b=False: output = left_hbm @ right_cpu.T
  left_hbm:  [M_left, K]
  right_cpu: [N, K]
  output:    [M_left, N]

transpose_b=True: output = left_hbm @ right_cpu
  left_hbm:  [M_left, K]
  right_cpu: [K, N]
  output:    [M_left, N]
```

Do not call private `_dispatch_nt(...)` directly from
`attention_activation_offload.py`.

Do not add a CPU-left `@^^` dependency. For LoRA-A forward, compute:

```text
M_fwd = align_up(M, 8)
X_pad_cpu = pad_rows_to(X_drop_cpu, M_fwd)            # [M_fwd,in] CPU
S_T_pad = A @^ X_pad_cpu.T                            # left_hbm = A [r,in], right_cpu = X_pad_cpu [M_fwd,in], transpose_b=False
S_T = S_T_pad[:, :M]                                  # [r,M] HBM
S = row_major(S_T.T)                                  # [M,r] HBM
```

For `dA` and `dB`, do not pass non-contiguous CUDA transposed views into
AsymGEMM. Use CPU-right AsymGEMM for `dA`, where the CPU operand is the large
`X_drop_cpu`. Do not use CPU-right AsymGEMM for `dB` in v1: `S_cpu` is small,
and staging it avoids materializing a wide contiguous `dY.T` HBM temporary.

```text
M_grad = align_up(M, 64)
X_grad_cpu = pad_rows_to(X_drop_cpu, M_grad)          # [M_grad,in] CPU
dS_T = pad_cols_to(row_major(dS.T), M_grad)           # [r,M_grad] HBM contiguous
dA = dS_T @^ X_grad_cpu                               # transpose_b=True, [r,in]

S_stage = stage(S_cpu)                                # [M,r] HBM, small
dB = scale * (dY.T @ S_stage)                         # [out,r] HBM GEMM, then release S_stage
```

The helper can own validation, but the wrapper should own padding and left-HBM
materialization so true `M` slicing is explicit. This follows the existing
expert offload pattern for the CPU-right dA path: LoRA-A forward pads rows to an
8 multiple; dA pads the reduction dimension to a 64 multiple when
`transpose_b=True`.

Direct bf16 constraints to validate:

```text
forward S_T: in % 8 == 0 and M_fwd % 8 == 0
dA:          M_grad % 64 == 0 and in % 8 == 0
```

If these fail under `backend == "asym"`, fail loudly. Under an explicit torch
debug backend, fallback may be allowed for tests but must be reported in JSON.
The v1 design saves `S_cpu [M,r]` and stages that small tensor for dB. Do not
save a second `S_T_cpu [r,M]` owner unless a later profile proves the extra CPU
memory is worth it.

## Projection Math Implemented by the Wrapper

Forward for one projection:

```text
X_2d = flatten_last_dim(X)                           # [M,in] HBM
base = asym_frozen_linear(X_2d, W_cpu, bias)          # [M,out] HBM

X_cpu = offload_or_share_source(X_2d)                 # [M,in] CPU
X_drop_cpu, mask_cpu = D(X_cpu)                       # [M,in] CPU, mask only when p > 0

M_fwd = align_up(M, 8)
X_fwd_cpu = pad_rows_to(X_drop_cpu, M_fwd)            # [M_fwd,in] CPU
S_T_pad = hbm_cpu_matmul(A, X_fwd_cpu, transpose_b=False) # [r,M_fwd] HBM
S_T = S_T_pad[:, :M]                                  # [r,M] HBM
S = row_major(S_T.T)                                  # [M,r] HBM
S_cpu = offload(S)                                    # [M,r] CPU

delta = scale * (S @ B.T)                             # [M,out] HBM
Y_2d = base + delta                                   # [M,out] HBM
Y = restore_last_dim(Y_2d)
```

Backward for one projection:

```text
dY_2d = flatten_last_dim(dY)                          # [M,out] HBM

dX_base = dY_2d @^ W_cpu                              # [M,in] HBM
dS = scale * (dY_2d @ B)                              # [M,r] HBM
dX_lora_raw = dS @ A                                  # [M,in] HBM
dX_lora = D_grad(dX_lora_raw, mask_cpu)               # [M,in] HBM
dX = dX_base + dX_lora                                # [M,in] HBM

X_drop_cpu = recompute D(X_cpu)                       # [M,in] CPU
M_grad = align_up(M, 64)
X_grad_cpu = pad_rows_to(X_drop_cpu, M_grad)          # [M_grad,in] CPU
dS_T = pad_cols_to(row_major(dS.T), M_grad)           # [r,M_grad] HBM
dA = hbm_cpu_matmul(dS_T, X_grad_cpu, transpose_b=True) # [r,in] HBM grad
S_stage = stage(S_cpu)                                # [M,r] HBM
dB = scale * (dY_2d.T @ S_stage)                      # [out,r] HBM grad
release_stage(S_stage)
```

Only save CPU handles, scalar metadata, original shape/dtype, and base
HostWeight metadata on `ctx`. Save LoRA A/B through `ctx.save_for_backward(...)`
or as tensor inputs supported by autograd. Do not use
`ctx.save_for_backward(X_2d, S, X_drop, ...)` for wide source activations.

## Launch Efficiency Contract

This design is memory-first, but it must not create a many-small-kernel path.
The implementation must operate at full projection granularity.

Hard rules for v1:

```text
No Python loops over tokens/rows M.
No loops over heads or KV groups.
No loops over LoRA rank chunks.
No row-window staging in v1.
No per-row or per-head AsymGEMM calls.
No splitting one q/k/v/o projection into multiple GEMMs except explicit full-tensor padding.
No chunked torch fallback loops; debug fallback must use one whole-tensor matmul.
```

Allowed loop level:

```text
model layers
attention projection leaves q_proj/k_proj/v_proj/o_proj
```

Expected whole-tensor launch envelope per wrapped projection:

```text
Forward:
  1 base AsymFrozenLinear call:       X @^ W_cpu.T
  1 CPU-right LoRA-A AsymGEMM call:   A @^ X_drop_cpu.T
  1 HBM LoRA-B GEMM call:             S @ B.T
  vectorized whole-tensor copies/transposes/adds only

Backward:
  1 base dx AsymGEMM call:            dY @^ W_cpu
  1 HBM dS GEMM call:                 dY @ B
  1 HBM LoRA input GEMM call:         dS @ A
  1 CPU-right dA AsymGEMM call:       dS.T @^ X_drop_cpu
  1 HBM dB GEMM call with staged S:   dY.T @ stage(S_cpu)
  vectorized whole-tensor padding/transposes/adds/dropout-grad only
```

The current HF model structure already exposes q/k/v/o as separate leaves, so
v1 may have separate projection-level calls for q, k, v, and o. That is not a
license to add smaller loops inside a projection. If launch overhead dominates
after Stage 6, the next optimization is a separate module-level q/k/v packing
stage, not hidden per-row/window loops.

Validation must record:

```text
base_asym_calls_by_projection
lora_a_asym_calls_by_projection
lora_b_gemm_calls_by_projection
backward_base_dx_calls_by_projection
dA_asym_calls_by_projection
dB_gemm_calls_by_projection
small_gemm_shapes
copy_or_transpose_kernel_counts
```

For Stage 2 and Stage 3 unit tests, monkeypatch or instrument
`hbm_cpu_matmul(...)` and the LoRA-B `torch.matmul` call site so a single
projection proves the expected call envelope. For Stage 6 profiling, include
the same counters in JSON alongside timing; do not rely only on wall-clock time.

## Dropout Policy

LoRA dropout is separate from attention-probability dropout. Attention dropout
remains inside SDPA/FlashAttention/eager attention.

Supported range for this wrapper is `0 <= lora_dropout_p < 1`. If
`lora_dropout_p == 1`, fail clearly when `ASYMM_ATTN_ACT_OFFLOAD=1` instead of
silently changing semantics.

For `lora_dropout_p == 0`, `D(x) = x` and no mask is saved.

For `lora_dropout_p > 0`, the wrapper must:

```text
1. generate one branch-local mask for each projection invocation;
2. save that exact mask as a CPU handle;
3. apply inverted dropout to CPU source activations for LoRA-A forward and dA;
4. stage the mask only for D_grad(dX_lora_raw) if the HBM gradient path needs it.
```

Do not silently reject nonzero dropout. Do not compare nonzero-dropout output to
the old `nn.Dropout` path unless both paths are forced to use the same mask.
CPU mask generation does not need to preserve the old CUDA RNG stream exactly;
it must preserve the inverted-dropout distribution and must validate against the
saved mask.
Validation must compare:

```text
p == 0    against current AsymLoRALinear
p > 0    against a masked reference using the saved mask
```

The validation JSON must record mask dtype, shape, CPU bytes, and whether mask
staging occurred.

CPU-generated handles must preserve their intended CUDA staging device:

```text
dropout mask handle original_device = source_handle.original_device
padded source handle original_device = source_handle.original_device
padded S handle original_device = S_cpu.original_device
```

Do not call `manager.adopt_cpu(mask, tag)` without `original_device=...`,
because the manager would stage that mask back to CPU instead of HBM.

## Q/K/V Source Sharing

`agent/attn_math.md` uses one `X_cpu` for q/k/v. A naive leaf wrapper offloads
the same hidden state three times.

Add `AttentionActivationOffloadContext`:

```python
class AttentionActivationOffloadContext:
    def get_or_offload_qkv_source(
        self,
        x_2d: torch.Tensor,
        manager: ActivationOffloadManager,
        tag: str,
    ) -> CPUActivationHandle: ...

    def clear_qkv_source_after_v(self) -> None: ...
```

Cache key:

```python
(
    x_view.device,
    x_view.untyped_storage().data_ptr(),
    x_view.storage_offset(),
    tuple(x_view.shape),
    tuple(x_view.stride()),
    x_view.dtype,
)
```

Compute this key before any per-leaf `.contiguous()` materialization. If q/k/v
need a contiguous source, the context should create one shared contiguous HBM
source and one shared `X_cpu`, not three independent contiguous copies that
miss the storage-key cache.

Each q/k/v autograd node must keep its own reference to the returned
`CPUActivationHandle`; clearing the forward cache after v must not invalidate
backward.

Because `ActivationOffloadManager` does not currently refcount shared handles,
`AttentionActivationOffloadContext` owns the q/k/v retain count:

```text
retain when q/k/v Function ctx stores the shared handle
clear_qkv_source_after_v only removes the forward lookup cache
release the shared CPU handle after the last q/k/v backward user
```

If explicit release ordering is hard in v1, keep the shared source handle alive
until the attention layer backward finishes and report its lifetime in JSON.

If source sharing is unstable in the first prototype, keep per-leaf offload as
a debug fallback and record duplicate CPU source bytes in JSON. Do not block the
first HBM proof on q/k/v source sharing, but do not claim the final math shape
until the sharing stage is validated.

## Stage-Gated Plan

Do not start a stage until the previous stage has a pytest result or JSON
artifact. Do not use arbitrary reduction or slowdown thresholds. Each artifact
must report correctness, peak HBM, reserved HBM, and timing so the next stage can
make an informed decision.

### Stage Scope Matrix

| Stage | Files | Functions/classes in scope | Out of scope |
| --- | --- | --- | --- |
| 0 Baseline | none | none | code edits |
| 1 Helpers | `frozen_linear.py`, tests | `hbm_cpu_matmul`, helper tests, existing `ActivationOffloadManager` counters | attention wrappers, async copies |
| 2 Projection forward | `attention_activation_offload.py`, validation script | `AsymActivationOffloadLoRALinear.forward`, `_AsymActivationOffloadLoRALinearFunction.forward`, CPU dropout forward | backward, LF integration, q/k/v sharing |
| 3 Projection backward | same files | `_AsymActivationOffloadLoRALinearFunction.backward`, dX/dA/dB, dropout grad | LF integration, q/k/v sharing |
| 4 LF integration | `lf.py`, LF tests | `_attention_act_offload_enabled`, `_wrap_lf_linear_leaf`, report fields | attention module rewrites |
| 5 Q/K/V sharing | wrapper + tests | `AttentionActivationOffloadContext` | attention-core hooks |
| 6 Full-attention memory proof | validation/profiling scripts | Qwen3/Llama4 text attention projection-offload validation | FA/SDPA kernel changes |
| STOP | docs only | review Stage 6 artifacts | do not proceed automatically |
| 7 Scoped attention-core saved tensor offload | separate wrapper/util | `saved_tensors_hooks` / `save_on_cpu` around prepare+core only | FA/SDPA kernel changes |
| 8 Async/pool polish | manager/script plumbing | optional streams/events/pinned pools | math changes |

Stages 1-6 implement the current math. Stop after Stage 6 and review the
artifacts before starting Stage 7.

## Stage 0: Baseline Lock

Purpose: prove the current projection weight-offload path is green.

Allowed changes: none.

Validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_lf_offload_module_parser_stage1_contract \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_frozen_base_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_strict_attention_offload_rejects_cuda_source
```

SM100/CUDA validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_attention_uses_asymgemm
```

Dense LoRA parity smoke:

```bash
python -m pytest -q \
  tests/training/test_toy_dense_lora_sft.py::test_dense_llm_parity_by_target_mode_and_checkpointing \
  tests/training/test_toy_dense_lora_sft.py::test_dense_llm_repeated_steps_are_finite_and_track_torch
```

Existing activation manager smoke:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_activation_offload_manager_tracks_cpu_owners_and_stage_reuse
```

Pass criteria:

- tests pass on the target CUDA/SM100 machine;
- selected attention bases are CPU HostWeights;
- env unset behavior remains current `AsymLoRALinear`;
- no attention activation-offload wrapper exists yet.

## Stage 1: Shared Helpers

Purpose: expose the CPU-right GEMM primitive and verify the existing manager is
sufficient.

Allowed changes:

- add `hbm_cpu_matmul(...)` to `asym_gemm/training/frozen_linear.py`;
- extend `ActivationOffloadStats` only if a required counter is missing;
- add focused helper tests.

Tests to add:

```text
tests/training/test_attention_activation_offload_helpers.py
```

Validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_activation_offload_manager_tracks_cpu_owners_and_stage_reuse \
  tests/training/test_attention_activation_offload_helpers.py
```

Helper test coverage:

- `ActivationOffloadManager.offload(...)` returns a `CPUActivationHandle`;
- CPU owners are pinned when CUDA pinning is available;
- `stage(...)` reuses HBM staging buffers and updates counters;
- `hbm_cpu_matmul(A_hbm, X_cpu, transpose_b=False)` matches `A @ X.T`;
- `hbm_cpu_matmul(dS_T_hbm, X_cpu, transpose_b=True)` matches dA when `dS_T`
  and `X_cpu` share the padded reduction dimension;
- staging `S_cpu` to HBM and computing `dY.T @ S_stage` matches dB before
  scale without materializing a full contiguous `dY_T` tensor;
- helper rejects non-bf16 activation CPU-right use for this feature;
- non-contiguous HBM left operands are rejected or materialized explicitly by
  the wrapper before the helper call;
- unsupported direct AsymGEMM shapes fail loudly unless the selected backend
  explicitly allows torch fallback.

Pass criteria:

- helper tests pass;
- existing LF attention offload tests still pass;
- manager snapshot contains offloaded bytes, staged bytes, CPU-owned bytes, and
  per-tag peaks.

## Stage 2: Projection Wrapper Forward

Purpose: prove the forward math for one dense attention LoRA projection.

Allowed changes:

- add `asym_gemm/training/attention_activation_offload.py`;
- implement `AsymActivationOffloadLoRALinear`;
- implement `_AsymActivationOffloadLoRALinearFunction.forward`;
- implement CPU dropout forward and mask ownership;
- backward may raise `NotImplementedError`;
- add `scripts/testing/validate_attention_activation_offload.py --mode linear_forward`.

Forward requirements:

- flatten only the last dim, then restore the original shape;
- compute base with `AsymFrozenLinear` and preserve frozen bias if present;
- offload source activation to CPU;
- apply LoRA dropout on CPU;
- pad `X_drop_cpu` rows to the forward alignment required by AsymGEMM;
- compute `S_T_pad = A @^ X_drop_cpu_pad.T`;
- slice `S_T = S_T_pad[:, :M]`;
- materialize `S = row_major(S_T.T)`;
- offload `S_cpu`;
- compute `delta = scale * (S @ B.T)`;
- return `base + delta` without saving full HBM source activation on `ctx`;
- save CPU handles, mask handle, LoRA params, shape/dtype, and base metadata.

Validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_forward \
  --device cuda:0 \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 13 \
  --compare-to current_asym_lora \
  --output-json reports/attn_act_offload/stage2_forward.json
```

Dropout validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_forward \
  --device cuda:0 \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 17 \
  --compare-to masked_reference \
  --output-json reports/attn_act_offload/stage2_forward_dropout.json
```

Pass criteria:

- `p == 0` output matches current `AsymLoRALinear` within existing bf16
  tolerances;
- `p > 0` output matches the masked reference;
- JSON records offload tags, CPU-owned bytes, mask bytes, and stage peaks;
- JSON records the forward launch envelope for one projection;
- tests prove there is one LoRA-A CPU-right AsymGEMM and one LoRA-B GEMM per
  projection, with no row/head/rank loop;
- no full source activation is saved by `ctx.save_for_backward(...)`;
- `S_cpu` is saved and HBM `S` is released after final forward use.

## Stage 3: Projection Wrapper Backward

Purpose: prove `dX`, `dA`, and `dB` without full model integration.

Allowed changes:

- implement `_AsymActivationOffloadLoRALinearFunction.backward`;
- implement HBM dropout-gradient application;
- implement contiguous HBM-left materialization for `dS.T` and `dY.T`;
- support `p == 0` and `p > 0`;
- add `--mode linear_backward` to the validation script.

Backward requirements:

- compute `dX_base = dY @^ W_cpu`;
- compute `dS = scale * (dY @ B)`;
- compute `dX_lora_raw = dS @ A`;
- apply `D_grad(...)` using the saved mask;
- pad `D(X_cpu)` rows to the dA reduction alignment required by AsymGEMM;
- compute `dA = pad_cols_to(row_major(dS.T), M_grad) @^ D(X_cpu)_pad`;
- stage `S_cpu` to HBM and compute `dB = scale * (dY.T @ S_stage)`;
- sum `dX = dX_base + dX_lora`;
- return gradients only for input and LoRA A/B params;
- base HostWeight and frozen bias remain non-trainable in LF.

Validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_backward \
  --device cuda:0 \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 23 \
  --compare-to current_asym_lora \
  --output-json reports/attn_act_offload/stage3_backward.json
```

Dropout backward validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode linear_backward \
  --device cuda:0 \
  --in-features 128 \
  --out-features 128 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 29 \
  --compare-to masked_reference \
  --output-json reports/attn_act_offload/stage3_backward_dropout.json
```

Pytest:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_backward_matches_current_without_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_backward_matches_masked_reference_with_dropout \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_rejects_dropout_one \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_preserves_frozen_bias \
  tests/training/test_attention_activation_offload_lora.py::test_attention_act_offload_linear_does_not_grad_base_weight
```

Pass criteria:

- `dX`, `dA`, and `dB` match references within existing bf16 tolerances;
- dropout backward uses the saved mask, not a new RNG draw;
- `dA` uses CPU-side dropped source activations;
- `dB` uses saved `S_cpu`;
- JSON records the backward launch envelope for one projection;
- tests prove there is one base-dx AsymGEMM, one dS GEMM, one dX-LoRA GEMM, one
  dA AsymGEMM, and one staged-S dB GEMM per projection, with no row/head/rank
  loop;
- no frozen base weight receives grad;
- JSON shows no unexpected wide HBM saved activation tags.

## Stage 4: LF Integration

Purpose: enable the wrapper only for selected attention LoRA projections.

Allowed changes:

- import `AsymActivationOffloadLoRALinear` in `asym_gemm/integrations/lf.py`;
- add `_attention_act_offload_enabled()` env parser;
- add `_is_text_attention_projection_name(name: str) -> bool`;
- add `_attention_parent_name(name: str) -> str | None`;
- add `_build_attention_activation_contexts(model, selected_names)`;
- modify `_wrap_lf_linear_leaf(...)` only for the gated attention-LoRA case;
- thread `projection_role` from leaf name;
- add report fields if `LFAsymReport` can carry them cleanly.

Gate:

```text
ASYMM_ATTN_ACT_OFFLOAD=1
```

Report fields:

```text
attention_act_offload_wrapped
attention_act_offload_modules
attention_act_offload_enabled
```

Do not replace `AsymFrozenLinear` attention projections in this stage.
Frozen-only projections do not have LoRA activation saves.

The current dense wrapping pass is flat over `named_modules()`. Before replacing
leaves, pre-scan selected dense attention names and build a parent-name to
`AttentionActivationOffloadContext` map:

```text
parent self_attn/attention module has q_proj/k_proj/v_proj selected:
  give q/k/v wrappers the same context
otherwise:
  wrapper gets attention_context=None
```

`_is_text_attention_projection_name(...)` must reject known vision paths such as
names containing `.vision_model.`, `.vision_tower.`, `.multi_modal_projector.`,
or `.vision.` before the generic `q_proj/k_proj/v_proj/o_proj` classifier can
turn them into `"attention"`. In strict mode, fail if a selected activation
offload target is a vision attention projection; otherwise leave it on the
existing wrapper path and report it as skipped.

Validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_lora_adopts_cpu_storage_without_clone \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_attention_uses_asymgemm
```

New integration tests:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_wraps_selected_lora_projection \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_default_off_is_unchanged \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_rejects_non_asym_backend \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_attention_activation_offload_excludes_llama4_vision
```

Pass criteria:

- env unset: selected attention LoRA still wraps as `AsymLoRALinear`;
- env set: selected attention LoRA wraps as
  `AsymActivationOffloadLoRALinear`;
- frozen-only selected attention projections remain `AsymFrozenLinear`;
- residency audit still reports attention base HostWeights on CPU;
- optimizer contains only LoRA parameters;
- Qwen3/Llama4 text attention modules are not otherwise rewritten;
- SDPA/FlashAttention/eager attention implementation selection is unchanged.

## Stage 5: Q/K/V Source Sharing

Purpose: match the single-`X_cpu` q/k/v design from `attn_math.md`.

Allowed changes:

- implement `AttentionActivationOffloadContext`;
- create one context per attention module during LF wrapping when q/k/v are all
  wrapped;
- pass the shared context into q/k/v wrappers;
- clear the forward cache after v;
- retain per-leaf offload fallback behind a debug flag or error fallback.

Validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode qkv \
  --device cuda:0 \
  --hidden-size 128 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --tokens 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 31 \
  --output-json reports/attn_act_offload/stage5_qkv_share.json
```

Pytest:

```bash
ASYMM_ATTN_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_wrappers_share_one_source_handle \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_source_cache_clears_after_v_forward \
  tests/training/test_attention_activation_offload_lora.py::test_qkv_share_backward_keeps_own_handle_references
```

Pass criteria:

- q/k/v wrappers see one shared `X_cpu` handle for the same input tensor;
- q/k/v still have independent dropout masks and independent `S_*_cpu`;
- backward remains correct for a combined q/k/v loss;
- launch counters show projection-level q/k/v calls only, not per-head or
  per-token calls;
- JSON reports duplicate source offload bytes before and after sharing.

## Stage 6: Full-Attention Memory Proof

Purpose: prove projection activation offload reduces HBM in real text attention
without touching SDPA/FA internals.

Validation script:

```text
scripts/testing/validate_attention_activation_offload.py
```

Required modes:

```text
linear_forward
linear_backward
qkv
full_attention
profile
```

Required launch-audit option:

```text
--profile-launches true|false
```

When true, the script must instrument wrapper-level call counts and record GEMM
input/output shapes. It can use Python counters around `hbm_cpu_matmul(...)`,
`AsymFrozenLinear.forward(...)`, and LoRA-B matmul call sites first; Nsight
Systems is optional and only needed if counters disagree with observed timing.

Qwen3 full-attention validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode full_attention \
  --model-family qwen3 \
  --device cuda:0 \
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
  --output-json reports/attn_act_offload/stage6_qwen3_full_attention.json
```

Llama4 text validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode full_attention \
  --model-family llama4_text \
  --device cuda:0 \
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
  --output-json reports/attn_act_offload/stage6_llama4_text_full_attention.json
```

Dedicated profile:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode profile \
  --model-family qwen3 \
  --device cuda:0 \
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
  --output-json reports/attn_act_offload/stage6_profile.json
```

LF profiler integration to add:

```text
scripts/lf/run_lf_lora_sft.sh
  ASYMM_ATTN_ACT_OFFLOAD default false
  config/report field asymm_attn_act_offload

scripts/lf/profile_lora_lf.sh
  --asymm-attn-act-offload LIST
  include attnact1/attnact0 in run identity
  validate profile config matches requested value
```

LF source-profile memory proof:

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
  --expert-policies none \
  --asymm-expert-act-offload false \
  --asymm-attn-act-offload false,true \
  --profile-memory-breakdown true \
  --profile-memory-breakdown-modules attention,router,mlp,experts,shared_experts,lora,embedding,norms,loss \
  --profile-level module \
  --profile-sync true \
  --output-root reports/attn_act_offload/lf_memory \
  --plot true \
  --plot-memory-breakdown true
```

Validate each produced source profile:

```bash
python scripts/lf/validate_lf_memory_capacity_schema.py \
  --source-profile-json <run_dir>/source_profile.json \
  --memory-breakdown-summary <run_dir>/memory_breakdown_summary.json \
  --require-breakdown
```

Optional plot:

```bash
python scripts/plotting/plot_lf_memory_breakdown.py \
  --run-dir <run_dir> \
  --output-dir <run_dir>/memory_plots \
  --clean-output \
  --y-scale shared
```

Pass criteria:

- forward/loss/backward and q/k/v/o LoRA grads match current path within
  existing bf16 tolerances for `p == 0`;
- dropout runs compare against masked references;
- JSON reports current/offload/recompute peak allocated HBM, peak reserved HBM,
  step time, manager counters, source-sharing counters, saved-HBM tags, launch
  counts, and GEMM shapes;
- launch summary shows no row/head/rank/window loops inside q/k/v/o projections;
- profile artifact shows a real peak-HBM reduction for `attn_act_offload` vs
  current;
- timing is reviewed from JSON and is not obviously unusable; no fixed slowdown
  threshold is encoded in the script;
- schema version 2 memory breakdown validates;
- `allocated_closure_ok` and `reserved_closure_ok` are true when present;
- `attention:saved_activations` drops versus baseline due to projection/LoRA
  activations;
- the reduction is not merely shifted into `attention:temporary_workspace` or
  unattributed peak bytes;
- SDPA/FA saved tensors may still appear under attention core. Stage 6 must not
  claim those are removed.

## Hard Stop Before Stage 7

Stop here after Stage 6. Review:

```text
reports/attn_act_offload/stage6_profile.json
reports/attn_act_offload/lf_memory/**/source_profile.json
reports/attn_act_offload/lf_memory/**/memory_breakdown_summary.json
```

Only proceed if the remaining HBM peak is now dominated by attention-core saved
tensors rather than projection/LoRA activations.

## Stage 7: Scoped Attention-Core Saved-Tensor Offload

Purpose: test attention-core saved-tensor offload after projection offload is
already proven.

Do not modify FA/SDPA kernels. Use scoped hooks first:

```python
with torch.autograd.graph.saved_tensors_hooks(pack_to_cpu, unpack_to_hbm):
    # Existing model code still runs here.
    Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)
    AttnOut = attention_core(Q_attn, K_attn, V_attn)
```

or test `torch.autograd.graph.save_on_cpu(...)`.

This stage likely needs a model-specific wrapper because the hook scope must
cover only prepare/core, not q/k/v/o projection wrappers:

```text
asym_gemm/training/attention_core_saved_tensor_offload.py
AsymQwen3AttentionSavedTensorOffload
AsymQwen3MoeAttentionSavedTensorOffload
AsymLlama4TextAttentionSavedTensorOffload
```

Validation:

```bash
mkdir -p reports/attn_act_offload
ASYMM_ATTN_ACT_OFFLOAD=1 \
ASYMM_ATTN_CORE_SAVE_ON_CPU=1 \
python scripts/testing/validate_attention_activation_offload.py \
  --mode profile \
  --model-family qwen3 \
  --device cuda:0 \
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
  --warmup 5 \
  --iters 10 \
  --output-json reports/attn_act_offload/stage7_core_hooks_profile.json
```

Pass criteria:

- correctness still matches Stage 6;
- JSON separates projection-offload bytes from attention-core hook bytes;
- artifact shows whether hook offload reduces the remaining HBM peak;
- timing is reviewed, not threshold-gated;
- no FA/SDPA kernel is patched.

## Stage 8: Async and Buffer Polish

Purpose: reduce transfer overhead after the synchronous memory-first design is
correct.

Allowed changes:

- add optional D2H/H2D streams inside `ActivationOffloadManager`;
- add CUDA events for reuse safety;
- preallocate pinned CPU pools and HBM staging pools;
- add optional mask packing/compression if dropout masks dominate CPU memory.

Validation should rerun Stage 6 and Stage 7 profile commands with extra JSON:

```text
copy_d2h_ms
copy_h2d_ms
compute_ms
overlap_ms
num_async_copies
num_wait_events
```

Do not change math or tolerances in this stage.

## Kernel Modification Policy

Do not modify FlashAttention or PyTorch SDPA kernels unless all of these are
true:

```text
1. Stages 1-6 are correct and produce a real memory artifact.
2. Stage 7 scoped saved-tensor hooks are correct but still insufficient.
3. Memory attribution proves SDPA/FA internals dominate the remaining HBM peak.
4. The project accepts owning forward kernel, backward kernel, masks, dropout
   RNG, GQA semantics, q/k/v layouts, LSE/softmax stats, and HF compatibility.
```

AsymGEMM is useful for CPU-resident frozen projection weights and CPU-saved
projection LoRA activations. It is not naturally useful inside attention core
because attention core is activation-activation math: `Q @ K.T`, softmax, and
`P @ V`. Streaming Q/K/V from CPU into FA would usually be bandwidth-bound and
would lose the main benefit of fused attention kernels.

## Convergence Checklist

Implementation is unambiguous only when every item has an owner and validation
artifact:

- `hbm_cpu_matmul(...)` is implemented and tested;
- projection wrapper forward/backward owns exact saved tensors and gradients;
- transposed HBM views are explicitly materialized before CPU-right AsymGEMM;
- launch-envelope counters prove no per-row/per-head/per-rank/window GEMM loops
  are introduced;
- dropout p > 0 uses saved masks and masked-reference validation;
- LF integration changes only selected attention LoRA projections under
  `ASYMM_ATTN_ACT_OFFLOAD`;
- q/k/v source sharing is implemented or explicitly recorded as a duplicate CPU
  fallback;
- Qwen3/Qwen3-MoE text and Llama4 text paths are validated;
- Llama4 vision attention is excluded or fails clearly in strict mode;
- SDPA/FA/eager attention are unchanged through Stage 6;
- Stage 7 uses hooks, not kernel patches;
- every stage writes either pytest output or JSON under
  `reports/attn_act_offload/`.
