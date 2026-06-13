# Fix Plan: Efficient Expert Activation Offload

This is the implementation plan for `ASYMM_EXPERT_ACT_OFFLOAD=true` in the
Qwen3 packed MoE expert path. The goal is to reduce peak HBM by keeping wide
expert activations CPU-primary while using grouped AsymGEMM-style kernels to
consume CPU operands directly. The normal path must not replace wide HBM saves
with many small GEMMs or CPU matmul loops.

## Scope

Target path:

- `asym_gemm/training/qwen3_moe.py`
- `backend="asym"` with CPU-resident frozen expert base weights
- BF16 on SM100 for the first efficient implementation
- `lora_dropout_p == 0.0`, matching the current activation-offload support gate

Non-goals for this plan:

- Llama4 expert activation offload
- generic saved-tensor hooks
- dropout support beyond the existing `lora_dropout_p == 0.0` activation-offload
  gate

Math source of truth:

- `agent/mlp_math.md` defines the grouped equations.
- This file defines implementation policy, staging, kernels, and validation.

## Current Failure

The current `expact1` code saves some HBM but stalls in backward because it
turns expert LoRA work into per-group operations.

Exact current code behavior in `qwen3_moe.py`:

- `AsymQwen3Experts.forward(...)` and `forward_input_scaled(...)` call
  `_forward_expert_activation_offload(...)` when
  `ASYMM_EXPERT_ACT_OFFLOAD=true`, the module is training, and gradients are
  enabled.
- `_activation_offload_unsupported_reasons(...)` already gates this path to
  `backend="asym"`, expert base CPU offload, BF16 CUDA packed input, contiguous
  CUDA LoRA weights, `lora_dropout_p == 0.0`, SiLU, pinned CPU base weights,
  and current alignment requirements.
- `_ActivationOffloadQwen3ExpertFunction.forward(...)` offloads `X`, runs
  gate/up base, then calls `_activation_offload_lora_a_pair_forward(...)`.
- The same forward calls `_activation_offload_lora_a_forward(...)` for down
  LoRA-A.
- `_ActivationOffloadQwen3ExpertFunction.backward(...)` calls
  `_activation_offload_lora_a_grad(...)` for down, gate, and up LoRA-A
  gradients.
- The same backward calls `_activation_offload_cpu_lora_b_grad(...)` for down,
  gate, and up LoRA-B gradients.
- Backward currently stages `[dgate_cpu, dup_cpu]` before gate/up LoRA work and
  computes gate/up LoRA `dS` from staged HBM slices. The fixed schedule must do
  gate/up LoRA from CPU handles first, then stage only for base `dX`.
- `lora.py::grouped_expert_lora(...)` is not the CPU-left path. It dispatches
  to CUDA grouped-mm when either operand is CUDA, and its CPU/CPU branch is a
  per-group `F.linear` loop. Expert activation offload must use
  `grouped_expert_lora_cpu_left(...)` for CPU activation plus CUDA LoRA-A.
- `lora.py::grouped_expert_lora_pair(...)` concatenates low-rank CUDA tensors
  and is acceptable only for `[M,r]` LoRA-B work, not for wide CPU activation
  inputs.

Bad helpers in that current activation-offload path:

- `_activation_offload_lora_a_pair_forward`: loops over active groups and calls
  `_dispatch_nt()` per group.
- `_activation_offload_lora_a_forward`: same pattern for down LoRA-A.
- `_activation_offload_lora_a_grad`: loops over active groups for `dA`.
- `_activation_offload_cpu_lora_b_grad`: loops over active groups and runs CPU
  FP32 `matmul()` for `dB`.
- the old LoRA-A forward helpers increment a CPU-left-looking stat even though
  they emulate the operation with CPU-right `_dispatch_nt()` and layout repair.

These helpers must not be reachable from normal
`ASYMM_EXPERT_ACT_OFFLOAD=true`.

## Implementation Background

Keep these implementation lessons from Megatron-LM, Megatron Bridge, DeepSpeed,
and this codebase:

- use module-specific activation offload, not generic saved-tensor hooks, because
  the useful memory win comes from knowing exactly which expert tensors can stay
  CPU-resident
- keep activation offload separate from recompute/checkpoint policies until a
  combined schedule is explicitly designed
- use pinned, contiguous CPU buffers with clear ownership, reuse, and release
  points
- predefine the lifetime of each staged tensor before writing the backward order
- avoid double reloads: if a CPU-resident activation is consumed by multiple
  backward operations, schedule those operations together before staging or
  releasing the handle
- make staging profile-visible with per-tag bytes and max live stage bytes
- fail closed during capability checks instead of choosing another training
  implementation
- group work by projection and route metadata, not by individual expert group
- retire old slow code once the grouped path replaces it; do not preserve
  executable compatibility paths that can be accidentally reached

Applicable external lessons:

- Megatron-LM MoE fine-grained activation offload is module-level and targets
  specific modules rather than whole-model hooks. Apply that here by limiting
  this feature to Qwen3 expert LoRA/base tensors under
  `_ActivationOffloadQwen3ExpertFunction`.
- Megatron-LM avoids double reloading CPU-offloaded inputs in checkpoint
  backward. Apply that here by consuming `dgate_cpu`, `dup_cpu`, `X_cpu`, and
  `act_cpu` in the grouped LoRA kernels before any wide staging/release changes
  their lifetime.
- Megatron Bridge documents activation offload as a separate memory tradeoff
  from recompute and CUDA graphs. Apply that here by keeping the existing
  `expert_recompute_config.enabled` rejection and not adding a mixed
  recompute/offload implementation path.
- DeepSpeed activation checkpointing uses contiguous checkpoint buffers and
  explicit buffer reset/release behavior. Apply that here with pinned
  contiguous CPU handles, explicit stage tags, `release_stage(...)`, and
  max-live staging counters.

## Design Invariants

Wide tensors that should survive forward/backward as CPU handles:

```text
X_cpu       [M,H]
gate_cpu    [M,I]
up_cpu      [M,I]
act_cpu     [M,I]
dact_cpu    [M,I]
dgate_cpu   [M,I]
dup_cpu     [M,I]
S_gate_cpu  [M,r]
S_up_cpu    [M,r]
S_down_cpu  [M,r]
```

Allowed HBM materialization:

- low-rank `[M,r]` tensors
- short-lived `dS_*` and LoRA input-gradient tensors
- `act_cpu` staged once for down base
- `[dgate_cpu, dup_cpu]` staged once and reused for gate/up LoRA-B plus
  gate/up base `dX`

Forbidden HBM materialization:

- staging `X_cpu` or `act_cpu` just for LoRA-A `dA`
- staging separate `dgate_cpu` or `dup_cpu` copies in addition to the single
  `[dgate_cpu, dup_cpu]` stage
- `[r,M_g]` LoRA-A intermediates followed by transpose/layout repair
- hidden `.contiguous()` copies of wide `[M,I]` slices for LoRA work
- offloading `dY` to CPU for down `dB`; `dY` is already HBM

Forbidden compute:

- Python/C++ host loops that launch GEMM, AsymGEMM, matmul, CUDA copy, or wide
  CPU copy once per active expert group
- CPU FP32 per-expert LoRA-B gradient loops
- alternate branches that silently run the forbidden helpers
- returning group-indexed `[G,...]` gradients to Python for reduction

Allowed compute:

- one grouped call per projection
- one fused grouped call per gate/up pair
- native tile loops inside one grouped kernel launch
- metadata-only loops for validation or small metadata construction
- reference loops only in test files or explicit reference helpers that are not
  called by `_ActivationOffloadQwen3ExpertFunction`

## Retire Old Activation-Offload Helpers

Do not keep executable slow implementations for compatibility. Delete these
helpers entirely once the grouped replacements are wired. Do not leave
RuntimeError stubs, debug branches, or compatibility shims.

```text
_activation_offload_lora_a_pair_forward
_activation_offload_lora_a_forward
_activation_offload_lora_a_grad
_activation_offload_cpu_lora_b_grad
```

Replacement mapping:

```text
_activation_offload_lora_a_pair_forward
  -> grouped_lora_a_pair_forward_cpu_left

_activation_offload_lora_a_forward
  -> grouped_lora_a_forward_cpu_left

_activation_offload_lora_a_grad
  -> grouped_lora_a_pair_grad_cpu_right
  -> grouped_lora_a_grad_cpu_right

_activation_offload_cpu_lora_b_grad
  -> _grouped_lora_cuda_view + _grouped_lora_weight_grads_torch
```

Rules:

- no env flag restores the old helpers
- no debug branch in normal training calls the old helpers
- no misleading stats are incremented by retired helpers
- the retired helper definitions must not exist in `qwen3_moe.py`
- tests may keep tiny local reference functions, but those references must live
  in test files and must not be imported by `qwen3_moe.py`

## Memory And GEMM Audit

The design is memory-first, but every routed matrix operation must stay grouped.
The implementation must satisfy this operation map:

```text
forward gate/up LoRA-A:
  X_cpu + A_gate/A_up -> two CPU-left grouped calls, no HBM stage of X

forward gate/up LoRA-B:
  S_gate/S_up + B_gate/B_up -> one grouped pair call on CUDA low-rank tensors

forward down LoRA-A:
  act_cpu + A_down -> one CPU-left grouped call, no HBM stage of act

forward down LoRA-B:
  S_down + B_down -> one grouped CUDA call on low-rank tensor

backward down LoRA-B:
  dY + staged S_down -> grouped CUDA weight-gradient path; do not offload dY

backward down LoRA-A dA:
  dS_down + act_cpu -> one CPU-right grouped-reduction kernel

backward gate/up LoRA-B:
  stage [dgate_cpu, dup_cpu] once
  use non-contiguous gate/up views directly in grouped CUDA LoRA-B dS/dB
  reuse the same stage for gate/up base dX

backward gate/up LoRA-A dA:
  dS_gate/dS_up + X_cpu -> one CPU-right grouped pair-reduction kernel

backward gate/up base dX:
  reuse the existing [dgate_cpu, dup_cpu] stage
```

Allowed host-side loops in this feature are metadata-only loops and CPU padding
loops that prepare one grouped kernel call. A host-side loop is forbidden if it
launches GEMM/AsymGEMM/matmul/copy per active expert group or constructs
per-group wide tensors for LoRA compute.

## Risk Register

Known risks from code audit and small local profiling:

- **CPU activation/offload overhead.** The grouped GEMM call explosion is fixed,
  but CPU-resident activation math and D2H/H2D staging still add wall time.
  Profiles must report both memory savings and timing so this tradeoff is
  visible.
- **Current path explodes grouped calls.** A small Qwen3 probe showed current
  expact reduces peak HBM but increases AsymGEMM calls from `4` to `644` and
  slows the step by about `3x`. Retiring the old helpers is required, not
  optional.
- **CPU-left padding overhead.** `grouped_expert_lora_cpu_left` may pad uneven
  routes on CPU. Track padded rows, original rows, CPU copy time, and pinned CPU
  bytes in Stage 1 profiles.
- **Low-rank HBM workspace.** `grouped_expert_lora_pair` materializes
  concatenated `[M,r]` low-rank tensors. This is acceptable only while `r` is
  small relative to `H/I`; Stage 4/5 profiles must confirm it does not dominate
  peak HBM.
- **Rank alignment.** Existing grouped weight-gradient code has reference paths
  for unaligned tiny ranks. Expact validation should use production ranks such
  as `64` and assert no `reference_fallback_count`.
- **Pinned CPU allocation.** CPU-left requires pinned CPU input. Stage 1 and
  E2E profiles must fail on `input_not_pinned` rather than silently continuing.
- **CPU handle lifetime.** `ActivationOffloadManager.release_cpu(...)` returns
  buffers to the pinned CPU pool so CPU memory does not grow across backward
  steps.
- **LoRA-B CPU-source kernel.** A native CPU-source prototype exists for tests,
  but the hot path uses staged grouped CUDA LoRA-B because the prototype was too
  slow on rank-64 profiles.

Risk acceptance:

- none of these risks may be waived by restoring retired helpers
- Stage 5 is not complete unless E2E counters show projection-count grouped
  calls, no retired-path counters, lower peak HBM, and no qualitative timing
  blowup

## Existing And Missing Primitives

Existing:

- grouped BF16 base AsymGEMM:
  `m_grouped_bf16_asym_gemm_nt_contiguous`
- grouped BF16 CPU-left LoRA-A forward:
  `grouped_expert_lora_cpu_left` /
  `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous`
- grouped CUDA LoRA projection:
  `grouped_expert_lora` and `grouped_expert_lora_pair`

Missing:

- further optimization of CPU activation/offload scheduling, likely overlap or
  windowing, if the measured timing tradeoff is not acceptable for a target run

## CPU-Left Integration Contract

The SM100 CPU-left kernel is not a universal replacement for the original
CPU-right AsymGEMM path. It is only used when the left operand is intentionally
CPU-resident and the right operand is a CUDA LoRA-A weight:

```text
X_cpu   [M,H] @ A_gate[e].T [H,r] -> S_gate [M,r]
X_cpu   [M,H] @ A_up[e].T   [H,r] -> S_up   [M,r]
act_cpu [M,I] @ A_down[e].T [I,r] -> S_down [M,r]
```

Training integration required for this plan:

- normal dense/packed CUDA LoRA paths keep using `grouped_expert_lora`
- `ASYMM_EXPERT_ACT_OFFLOAD=true` uses CPU-left only inside
  `_ActivationOffloadQwen3ExpertFunction.forward`
- the integration calls the high-level
  `asym_gemm.training.lora.grouped_expert_lora_cpu_left` helper, not
  `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous` directly
- `ActivationOffloadManager.offload(...)` owns `X_cpu`, `gate_cpu`, `up_cpu`,
  and `act_cpu`; the wrapper validates that the source handle is pinned,
  contiguous, CPU BF16
- LoRA-A weights stay CUDA BF16 contiguous with shape `[E,r,K]`
- route metadata is the same normalized metadata used by `grouped_expert_lora`
- there is no alternate path to the old `_activation_offload_lora_a_*` helpers

Exact forward call-site replacements in `qwen3_moe.py`:

```python
# Replace _activation_offload_lora_a_pair_forward(...)
S_gate, S_up = grouped_lora_a_pair_forward_cpu_left(
    x_cpu.tensor,
    gate_lora_A,
    up_lora_A,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
    tag="gate_up",
)

# Replace _activation_offload_lora_a_forward(...)
S_down = grouped_lora_a_forward_cpu_left(
    act_cpu.tensor,
    down_lora_A,
    offsets,
    experts,
    metadata=lora_metadata,
    stats=layer.stats,
    tag="down",
)
```

Capability gates:

- `require_expert_activation_offload_kernels(scope="forward")` verifies the
  high-level CPU-left helper and native binding are present
- `scope="full"` also verifies the Stage 2 and Stage 3 backward kernels
- `_activation_offload_unsupported_reasons(...)` keeps checking backend,
  dtype, device, pinned base weights, LoRA tensor contiguity, and alignment
- wrappers raise immediately on CPU-left guard failures such as
  `input_not_pinned`, `requires_bf16`, `requires_8_aligned_nk`,
  `requires_sm100`, or missing binding

## Files To Add Or Modify

Python:

- Add `asym_gemm/training/exp_act_offload_lora.py`
- Modify `asym_gemm/training/qwen3_moe.py`
- Modify `asym_gemm/training/frozen_linear.py::AsymExecutionStats`
- Modify `asym_gemm/training/activation_offload.py::ActivationOffloadManager`
- Optionally export helpers in `asym_gemm/training/__init__.py` for tests

Native:

- Add `csrc/apis/exp_act_offload.hpp`
- Add `csrc/exp_act_offload/exp_act_offload_kernels.cu`
- Modify `csrc/python_api.cpp`
- Modify `setup.py`

Tests and scripts:

- Add `scripts/testing/check_exp_act_offload_structure.py`
- Add `tests/training/test_exp_act_offload_native.py`
- Add/modify forward-only activation-offload tests in
  `tests/training/test_lf_qwen3_asym_backend.py`

## Public Python Helpers

All helpers live in `asym_gemm/training/exp_act_offload_lora.py`. The normal
autograd function imports these helpers and does not call native symbols
directly.

### `require_expert_activation_offload_kernels`

```python
from typing import Literal

def require_expert_activation_offload_kernels(
    *,
    scope: Literal["forward", "full"] = "full",
    check_only: bool = False,
) -> str | None:
    ...
```

Behavior:

- verify `grouped_expert_lora_cpu_left` is importable
- when `scope == "forward"`, require only the existing CPU-left LoRA-A forward
  path
- when `scope == "full"`, also verify these native symbols exist:
  - `sm100_grouped_lora_a_grad_bf16_cpu_right`
  - `sm100_grouped_lora_a_pair_grad_bf16_cpu_right`
- return `None` when available
- return `missing_<symbol>` when `check_only=True`
- raise `RuntimeError(f"Qwen3 expert activation offload is unavailable: {reason}")`
  when `check_only=False`
- never return or select a retired implementation

Call sites:

- `AsymQwen3Experts._activation_offload_unsupported_reasons(...)`
- `AsymQwen3Experts._forward_expert_activation_offload(...)`
- normal training path uses `scope="full"`
- forward-only Stage 1 tests may use `scope="forward"` or call the wrapper
  helpers directly

### LoRA-A CPU-left forward wrappers

```python
def grouped_lora_a_forward_cpu_left(
    source_cpu: torch.Tensor,      # [M,K] pinned CPU bf16
    lora_a: torch.Tensor,          # [E,r,K] CUDA bf16
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:                # [M,r] CUDA bf16
    ...

def grouped_lora_a_pair_forward_cpu_left(
    source_cpu: torch.Tensor,      # [M,K] pinned CPU bf16
    lora_a_gate: torch.Tensor,     # [E,r,K] CUDA bf16
    lora_a_up: torch.Tensor,       # [E,r,K] CUDA bf16
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:  # gate/up [M,r]
    ...
```

Implementation:

- validate CPU pinned BF16 input and CUDA BF16 weights
- call existing `asym_gemm.training.lora.grouped_expert_lora_cpu_left(...)`
- pass the existing `GroupedLoRAMetadata` through to the helper
- pair wrapper calls the single wrapper once for gate and once for up; this is
  two grouped calls, not a loop over active groups
- increment `expact_lora_a_forward_grouped_calls` in the wrapper
- allow `cpu_left_lora_a_calls` to be incremented only by the real CPU-left
  helper path
- do not create `[r,M_g]`
- do not call `row_major()` or `.contiguous()` to repair LoRA-A layout
- do not build a fresh `[E,2r,K]` concatenated LoRA-A weight
- do not copy `source_cpu` to CUDA
- do not call `_dispatch_nt()` from LoRA-A forward under activation offload

### LoRA-A `dA` CPU-right wrappers

```python
def grouped_lora_a_grad_cpu_right(
    grad_low_rank: torch.Tensor,   # [M,r] CUDA bf16
    source_cpu: torch.Tensor,      # [M,K] pinned CPU bf16
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    num_experts: int,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:                # [E,r,K] CUDA bf16
    ...

def grouped_lora_a_pair_grad_cpu_right(
    dS_gate: torch.Tensor,         # [M,r] CUDA bf16
    dS_up: torch.Tensor,           # [M,r] CUDA bf16
    x_cpu: torch.Tensor,           # [M,H] pinned CPU bf16
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    num_experts: int,
    stats: AsymExecutionStats | None,
) -> tuple[torch.Tensor, torch.Tensor]:  # gate/up [E,r,H]
    ...
```

Implementation:

- allocate expert-indexed CUDA outputs and zero once
- call the native symbol once
- increment `expact_lora_a_grad_grouped_calls`
- do not stage `source_cpu` or `x_cpu` to CUDA
- do not call `_dispatch_nt()`
- do not reduce group-indexed tensors in Python

Native symbols:

- `sm100_grouped_lora_a_grad_bf16_cpu_right`
- `sm100_grouped_lora_a_pair_grad_bf16_cpu_right`

### Gate/up LoRA-B staged grouped wrapper

```python
def _grouped_lora_cuda_view(
    x: torch.Tensor,               # CUDA view, may be non-contiguous
    weight: torch.Tensor,          # [E,out,in] CUDA bf16
    *,
    metadata: GroupedLoRAMetadata,
) -> torch.Tensor:
    ...

def gate_or_up_lora_b_backward_from_stage(
    grad_stage_view: torch.Tensor, # [M,I] view into staged [M,2I]
    low_rank: torch.Tensor,        # staged [M,r] CUDA bf16
    lora_b: torch.Tensor,          # [E,I,r] CUDA bf16
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    scale: float,
    stats: AsymExecutionStats | None,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:  # dS [M,r], dB [E,I,r]
    ...
```

Implementation:

- caller stages the single `[dgate_cpu, dup_cpu]` tensor before gate/up LoRA-B
- split the staged tensor into gate/up views; do not call `.contiguous()` on
  those views
- caller stages `S_gate` / `S_up` first if saved as CPU handles
- compute `dS` with `_grouped_lora_cuda_view(grad_stage_view, B.T, ...)`
- compute `dB` with `_grouped_lora_weight_grads_torch(grad_stage_view, S, ...)`
- increment `expact_lora_b_backward_grouped_calls`
- consume `lora_b` in existing `[E,I,r]` layout; `B.T` is a view
- reuse the same staged `[dgate, dup]` tensor for gate/up base `dX`
- do not create separate wide HBM copies for gate/up LoRA-B
- do not create CPU FP32 `[E,I,r]` accumulators

Measured design decision:

- A CPU-source fused LoRA-B prototype was added and tested, but its scalar
  atomic reduction was too slow for the rank-64 profile shape.
- The hot path therefore uses one staged `[dgate, dup]` tensor plus grouped CUDA
  LoRA-B work. This keeps the work grouped, avoids per-expert GEMMs, and still
  avoids saving forward wide activations in HBM.

### Low-rank staging helper

```python
def stage_low_rank_from_cpu(
    handle: CPUActivationHandle,
    manager: ActivationOffloadManager,
    *,
    tag: str,
) -> torch.Tensor:
    ...
```

Implementation:

- only accept `[M,r]` low-rank handles
- call `manager.stage(handle, tag=tag)`
- increment `expact_stage_low_rank_calls`
- caller releases with `manager.release_stage(stage, drop_cache=True)`

## Native Kernel Design

Start from existing SM100 BF16 grouped AsymGEMM code:

- host/JIT wrapper: `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`
- kernel framework: `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh`

Reuse:

- grouped scheduler
- `offsets` / `experts` metadata handling
- CPU operand load/pipeline structure
- JIT launch/runtime pattern

Do not implement these as unrelated per-row CUDA loops.

### CPU operand reuse rule

The CPU tile is the expensive operand. The grid must be organized so a CTA
loads a CPU tile once, reuses it across practical rank/output panels, then
evicts it. If rank must be panelized, rank panels are an inner loop under the
same CPU tile load.

Forbidden native shape:

```text
for r_panel:
  load same CPU tile
  compute one rank panel
  discard CPU tile
```

Required native shape:

```text
load CPU tile once
for r_panel while CPU tile is resident:
  compute rank/output panel
evict CPU tile
```

Host-side loops over active experts, CPU row chunks, or rank panels that launch
kernels or AsymGEMMs are not allowed.

### LoRA-A `dA` as AsymGEMM-family adaptation

Existing CPU-right grouped AsymGEMM:

```text
D_g[M_g,N] = A_cuda[M_g,K] @ B_cpu[e_g,N,K].T
```

Needed LoRA-A gradient:

```text
dA[e,r,W] += dS_g[M_g,r].T @ source_cpu,g[M_g,W]
```

Axis mapping:

```text
existing output M -> LoRA rank r
existing output N -> source width W
existing reduction K -> routed rows M_g
CUDA operand -> dS_g read as logical transpose
CPU operand -> source_cpu,g read as logical transpose
output -> grad_a[e,r,W] with grouped accumulation
```

Single `dA` pseudocode:

```text
kernel sm100_grouped_lora_a_grad_bf16_cpu_right(
    dS[M,r], source_cpu[M,W], grad_a[E,r,W], offsets, experts):
  for each grouped CPU-source tile:
    g = group id
    e = experts[g]
    rows = offsets[g] + m_tile
    load source_tile = source_cpu[rows,W_tile] once
    for r_panel while source_tile is resident:
      acc[r_panel,W_tile] = 0
      for m in rows:
        acc += dS[m,r_panel].T * source_tile[m,W_tile]
      atomic_add grad_a[e,r_panel,W_tile] += acc
```

Pair `dA` pseudocode:

```text
kernel sm100_grouped_lora_a_pair_grad_bf16_cpu_right(
    dS_gate[M,r], dS_up[M,r], x_cpu[M,H],
    grad_gate_a[E,r,H], grad_up_a[E,r,H], offsets, experts):
  for each grouped CPU-source tile:
    g = group id
    e = experts[g]
    rows = offsets[g] + m_tile
    load x_tile = x_cpu[rows,H_tile] once
    for r_panel while x_tile is resident:
      acc_gate[r_panel,H_tile] = 0
      acc_up[r_panel,H_tile] = 0
      for m in rows:
        acc_gate += dS_gate[m,r_panel].T * x_tile[m,H_tile]
        acc_up   += dS_up[m,r_panel].T   * x_tile[m,H_tile]
      atomic_add grad_gate_a[e,r_panel,H_tile] += acc_gate
      atomic_add grad_up_a[e,r_panel,H_tile]   += acc_up
```

The pair kernel must not call the single kernel twice. It exists to reuse the
same `x_cpu` tile for gate and up.

### LoRA-B fused CPU-source backward

Needed gate/up equations:

```text
dS_g = scale * grad_out_cpu,g @ B[e_g]
dB[e] += scale * grad_out_cpu,g.T @ S_g
```

This is not a direct existing AsymGEMM substitute because the efficient path
must produce both `dS` and `dB` from the same CPU gradient tile.

Pseudocode:

```text
kernel sm100_grouped_lora_b_backward_bf16_cpu_source(
    grad_out_cpu[M,I], S[M,r], B[E,I,r],
    dS[M,r], dB[E,I,r], offsets, experts, scale):
  zero dS and dB before accumulation
  for each grouped CPU-grad tile:
    g = group id
    e = experts[g]
    rows = offsets[g] + m_tile
    load grad_tile = grad_out_cpu[rows,I_tile] once
    for r_panel while grad_tile is resident:
      load B_panel = B[e,I_tile,r_panel]
      load S_panel = S[rows,r_panel]
      dS_partial[M_tile,r_panel] = 0
      dB_acc[I_tile,r_panel] = 0
      for i in I_tile:
        dS_partial[:,r_panel] += grad_tile[:,i] * B_panel[i,r_panel]
      for m in rows:
        dB_acc[I_tile,r_panel] += grad_tile[m,I_tile].T * S_panel[m,r_panel]
      atomic_add dS[rows,r_panel] += scale * dS_partial
      atomic_add dB[e,I_tile,r_panel] += scale * dB_acc
```

Do not split `dS` and `dB` into independent native passes. That would reload
the same wide CPU gradient tile.

## Autograd Schedule

Modify `qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction` to follow this
order.

Forward:

```python
x_cpu = manager.offload(packed, "X")
gate_up_base = layer.gate_up_base(packed, offsets, experts, dense_experts=True)
gate_base, up_base = gate_up_base.chunk(2, dim=-1)

S_gate, S_up = grouped_lora_a_pair_forward_cpu_left(
    x_cpu.tensor, gate_lora_A, up_lora_A, offsets, experts,
    metadata=metadata, stats=layer.stats, tag="gate_up",
)
gate_delta, up_delta = grouped_expert_lora_pair(
    S_gate, S_up, gate_lora_B, up_lora_B, offsets, experts, metadata=metadata,
)
gate = gate_base + scale * gate_delta
up = up_base + scale * up_delta

gate_cpu = manager.offload(gate, "gate")
up_cpu = manager.offload(up, "up")
S_gate_cpu = manager.offload(S_gate, "S_gate")
S_up_cpu = manager.offload(S_up, "S_up")
act_cpu = _activation_offload_cpu_silu_mul(gate_cpu, up_cpu, manager, tag="act")

S_down = grouped_lora_a_forward_cpu_left(
    act_cpu.tensor, down_lora_A, offsets, experts,
    metadata=metadata, stats=layer.stats, tag="down",
)
down_delta = grouped_expert_lora(S_down, down_lora_B, offsets, experts, metadata=metadata)
S_down_cpu = manager.offload(S_down, "S_down")

act_stage = manager.stage(act_cpu, tag="act_for_down_base")
out = layer.down_base(act_stage, offsets, experts, dense_experts=True) + scale * down_delta
manager.release_stage(act_stage, drop_cache=True)
```

Backward:

```python
dS_down = scale * grouped_expert_lora(
    dY, down_lora_B.transpose(-1, -2), offsets, experts, metadata=metadata,
)
grad_down_lora_x = grouped_expert_lora(
    dS_down, down_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata,
)
S_down = stage_low_rank_from_cpu(ctx.down_low_rank_cpu, manager, tag="S_down_for_dB")
grad_down_lora_B = scale * _grouped_lora_weight_grads_torch(
    dY, S_down, offsets, experts, num_experts, out_dtype=down_lora_B.dtype,
    metadata=metadata, stats=layer.stats,
)
manager.release_stage(S_down, drop_cache=True)
grad_down_lora_A = grouped_lora_a_grad_cpu_right(
    dS_down, ctx.act_cpu.tensor, offsets, experts,
    num_experts=num_experts, stats=layer.stats, tag="down",
)

grad_act = _grouped_base_dx(layer.down_base, dY, offsets, experts, dense_experts=True)
grad_act.add_(grad_down_lora_x.to(dtype=grad_act.dtype))
grad_act_cpu = manager.offload(grad_act, "dact")
grad_gate_cpu, grad_up_cpu = _activation_offload_cpu_silu_backward(
    ctx.gate_cpu, ctx.up_cpu, grad_act_cpu, manager,
)

grad_gate_up = manager.stage_concat_columns(
    grad_gate_cpu, grad_up_cpu, tag="dgate_up_for_gate_up_base",
)
grad_gate_stage, grad_up_stage = grad_gate_up.split(I, dim=-1)

S_gate = stage_low_rank_from_cpu(ctx.gate_low_rank_cpu, manager, tag="S_gate_for_dB")
dS_gate = scale * _grouped_lora_cuda_view(
    grad_gate_stage, gate_lora_B.transpose(-1, -2), metadata=metadata,
)
grad_gate_lora_B = scale * _grouped_lora_weight_grads_torch(
    grad_gate_stage, S_gate, offsets, experts, num_experts,
    out_dtype=gate_lora_B.dtype, metadata=metadata, stats=layer.stats,
)
manager.release_stage(S_gate, drop_cache=True)

S_up = stage_low_rank_from_cpu(ctx.up_low_rank_cpu, manager, tag="S_up_for_dB")
dS_up = scale * _grouped_lora_cuda_view(
    grad_up_stage, up_lora_B.transpose(-1, -2), metadata=metadata,
)
grad_up_lora_B = scale * _grouped_lora_weight_grads_torch(
    grad_up_stage, S_up, offsets, experts, num_experts,
    out_dtype=up_lora_B.dtype, metadata=metadata, stats=layer.stats,
)
manager.release_stage(S_up, drop_cache=True)

grad_gate_lora_x = grouped_expert_lora(
    dS_gate, gate_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata,
)
grad_up_lora_x = grouped_expert_lora(
    dS_up, up_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata,
)
grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(
    dS_gate, dS_up, ctx.x_cpu.tensor, offsets, experts,
    num_experts=num_experts, stats=layer.stats,
)

grad_packed = _grouped_base_dx(layer.gate_up_base, grad_gate_up, offsets, experts, dense_experts=True)
manager.release_stage(grad_gate_up, drop_cache=True)
grad_packed.add_(grad_gate_lora_x).add_(grad_up_lora_x)
```

The wide `[dgate_cpu, dup_cpu]` concat is a single stage. Gate/up LoRA-B uses
views into that stage, and gate/up base `dX` reuses the same stage.

## Stage Plan

Each stage must pass its validation before moving to the next stage.

### Stage 0: Fail Closed, Counters, Structure Checks

Files:

- `asym_gemm/training/qwen3_moe.py`
- `asym_gemm/training/exp_act_offload_lora.py`
- `asym_gemm/training/frozen_linear.py`
- `scripts/testing/check_exp_act_offload_structure.py`

Implement:

- add `require_expert_activation_offload_kernels(...)`
- add `AsymExecutionStats` counters:
  - `expact_lora_a_forward_grouped_calls`
  - `expact_lora_a_grad_grouped_calls`
  - `expact_lora_b_backward_grouped_calls`
  - `expact_stage_low_rank_calls`
- make normal `ASYMM_EXPERT_ACT_OFFLOAD=true` training fail closed when full
  required kernels are missing
- add a retired-helper check that fails if old activation-offload helpers keep
  executable slow bodies
- add structural checker modes:
  - `fail-closed`
  - `grouped-forward`
  - `grouped-lora-b`
  - `grouped-da`
  - `final`
- add focused native tests for the new grouped reduction surfaces

Structural checker pseudocode:

```python
FORBIDDEN = {
    "grouped-forward": [
        "_activation_offload_lora_a_pair_forward",
        "_activation_offload_lora_a_forward",
    ],
    "grouped-lora-b": ["_activation_offload_cpu_lora_b_grad"],
    "grouped-da": ["_activation_offload_lora_a_grad"],
    "final": [
        "_activation_offload_lora_a_pair_forward",
        "_activation_offload_lora_a_forward",
        "_activation_offload_cpu_lora_b_grad",
        "_activation_offload_lora_a_grad",
    ],
}
REQUIRED = {
    "grouped-forward": [
        "grouped_lora_a_pair_forward_cpu_left",
        "grouped_lora_a_forward_cpu_left",
    ],
    "grouped-lora-b": ["_grouped_lora_cuda_view", "_grouped_lora_weight_grads_torch"],
    "grouped-da": [
        "grouped_lora_a_pair_grad_cpu_right",
        "grouped_lora_a_grad_cpu_right",
    ],
    "final": [
        "grouped_lora_a_pair_forward_cpu_left",
        "grouped_lora_a_forward_cpu_left",
        "_grouped_lora_cuda_view",
        "_grouped_lora_weight_grads_torch",
        "grouped_lora_a_pair_grad_cpu_right",
        "grouped_lora_a_grad_cpu_right",
    ],
}
# Parse _ActivationOffloadQwen3ExpertFunction.forward/backward with ast.
# Fail if forbidden calls remain or required calls are missing for the stage.
# Fail if a retired helper still has executable slow code. During staged
# implementation, the only allowed retained helper body is one immediate
# RuntimeError. After all call sites are removed, full helper deletion is also
# valid.
# Fail if normal-path activation-offload helpers contain:
#   for group_idx
#   loops over experts_cpu
#   _dispatch_nt(
#   .matmul(
```

Native wrapper validation is covered by
`tests/training/test_exp_act_offload_native.py`. If kernel-level profiling is
needed, add a profiler script that directly invokes the selected wrapper and
then run `ncu` on that script; do not point the NCU wrapper at a missing helper.

Validation:

```bash
PYTHONPATH="$PWD" python -m py_compile \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/frozen_linear.py \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/lora.py \
  asym_gemm/training/exp_act_offload_lora.py

PYTHONPATH="$PWD" python scripts/testing/check_exp_act_offload_structure.py \
  --require fail-closed

bash -n scripts/testing/ncu_exp_act_offload_kernel_profile.sh

PYTHONPATH="$PWD" python - <<'PY'
from asym_gemm.training.exp_act_offload_lora import require_expert_activation_offload_kernels
print("forward:", require_expert_activation_offload_kernels(scope="forward", check_only=True) or "ok")
print("full:", require_expert_activation_offload_kernels(scope="full", check_only=True) or "ok")
PY
```

Pass condition:

- activation offload does not silently run forbidden helpers
- counters exist in stats output
- structure checker fails on forbidden helper reachability
- structure checker fails if a retired helper definition still exists

### Stage 1: CPU-Left LoRA-A Forward

Files:

- `asym_gemm/training/exp_act_offload_lora.py`
- `asym_gemm/training/qwen3_moe.py`
- `tests/training/test_cpu_left_lora.py`
- `tests/training/test_lf_qwen3_asym_backend.py`

Implement:

- `grouped_lora_a_forward_cpu_left(...)`
- `grouped_lora_a_pair_forward_cpu_left(...)`
- import these wrappers into `qwen3_moe.py`
- replace `_activation_offload_lora_a_pair_forward`
- replace `_activation_offload_lora_a_forward`
- delete the old forward helper definitions after call sites are replaced
- keep low-rank `[M,r]` outputs on HBM and offload only `S_*`
- keep `x_cpu.tensor` and `act_cpu.tensor` as the left operands; do not stage
  them to HBM for LoRA-A
- keep `gate_lora_A`, `up_lora_A`, and `down_lora_A` as CUDA `[E,r,K]`
  weights; do not concatenate or transpose-copy them
- update the structure checker so Stage 1 fails if `_dispatch_nt(` is reachable
  from activation-offload LoRA-A forward

Validation:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_cpu_left_lora.py

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py \
  -k "forward_only_uses_grouped_cpu_left or activation_offload_uses_real_cpu_left_lora_a"

PYTHONPATH="$PWD" python scripts/testing/check_exp_act_offload_structure.py \
  --require grouped-forward
```

Pass condition:

- forward wrapper or forward-only scoped path completes
- normal full training remains fail-closed until Stage 2 and Stage 3 kernels are
  present
- no `[r,M_g]` LoRA-A temporary or layout repair remains
- LoRA-A forward call count scales with projection count, not active groups
- no wide LoRA-A input is staged to HBM
- tests prove `grouped_expert_lora_cpu_left(...)` is called for gate, up, and
  down LoRA-A
- tests prove the old CPU-right LoRA-A forward helpers and their misleading
  stat increments are not used
- structure checker proves the old forward helper bodies no longer contain
  executable `_dispatch_nt` or per-group copy code

### Stage 2: Staged Grouped LoRA-B Backward

Files:

- `asym_gemm/training/exp_act_offload_lora.py`
- `asym_gemm/training/qwen3_moe.py`
- `scripts/testing/check_exp_act_offload_structure.py`

Implement:

- stage `[dgate_cpu, dup_cpu]` once with `stage_concat_columns(...)`
- split that stage into gate/up views and pass the views directly to grouped-mm
- use `_grouped_lora_cuda_view(...)` for `dS`
- use `_grouped_lora_weight_grads_torch(...)` for `dB`
- delete `_activation_offload_cpu_lora_b_grad`
- use grouped GPU `dB_down` because `dY` is already HBM
- stage only low-rank `S_gate` / `S_up` / `S_down` when needed
- do not call `.contiguous()` on gate/up views

Validation:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend

PYTHONPATH="$PWD" python scripts/testing/check_exp_act_offload_structure.py \
  --require grouped-lora-b
```

Pass condition:

- `dS` and `dB` match reference
- no CPU FP32 per-expert `dB` loop
- exactly one `[dgate_cpu, dup_cpu]` stage is used and reused
- no hidden `.contiguous()` copies of the gate/up stage views
- old CPU LoRA-B helper definition is deleted
- kernel reuse profile shows grouped wrapper calls and no retired/reference path
- grouped-mm calls are projection-count sized, not active-expert-count sized

### Stage 3: LoRA-A `dA` CPU-Right Grouped Reduction

Files:

- `asym_gemm/training/exp_act_offload_lora.py`
- `asym_gemm/training/qwen3_moe.py`
- `csrc/apis/gemm.hpp`
- `csrc/jit_kernels/impls/sm100_bf16_exp_act_offload.hpp`
- `asym_gemm/include/asym_gemm/impls/sm100_bf16_exp_act_offload.cuh`
- `tests/training/test_exp_act_offload_grouped_grads.py`
- `tests/training/test_lf_qwen3_asym_backend.py`

Implement:

- native `sm100_grouped_lora_a_grad_bf16_cpu_right`
- native `sm100_grouped_lora_a_pair_grad_bf16_cpu_right`
- Python `grouped_lora_a_grad_cpu_right(...)`
- Python `grouped_lora_a_pair_grad_cpu_right(...)`
- delete `_activation_offload_lora_a_grad`
- implement `dA` as SM100 BF16 AsymGEMM-family CPU-right grouped reduction

Validation:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_exp_act_offload_grouped_grads.py

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py \
  -k "activation_offload"

PYTHONPATH="$PWD" python scripts/testing/check_exp_act_offload_structure.py \
  --require grouped-da

PYTHONPATH="$PWD" python - <<'PY'
import asym_gemm
required = [
    "sm100_grouped_lora_a_grad_bf16_cpu_right",
    "sm100_grouped_lora_a_pair_grad_bf16_cpu_right",
]
missing = [name for name in required if not hasattr(asym_gemm, name)]
if missing:
    raise SystemExit("missing native symbols: " + ", ".join(missing))
PY

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_exp_act_offload_native.py
  --rank 64 \
  --iters 5 \
  --json /tmp/expact_lora_a_down_grad_reuse.json

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
scripts/testing/ncu_exp_act_offload_kernel_profile.sh \
  lora-a-grad \
  /tmp/expact_lora_a_down_grad_ncu \
  --tokens 2048 \
  --num-experts 128 \
  --active-groups 64 \
  --hidden-dim 2048 \
  --intermediate-dim 4096 \
  --rank 64 \
  --json /tmp/expact_lora_a_down_grad_ncu.json

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 2048 \
  --top-k 8 \
  --num-experts 128 \
  --hidden-dim 2048 \
  --intermediate-dim 4096 \
  --rank 64 \
  --warmup 1 \
  --iters 1 \
  --output-json /tmp/qwen3_expact_stage3.json
```

Pass condition:

- `dA_gate`, `dA_up`, and `dA_down` match reference
- no LoRA-A `dA` helper loops over active groups
- old LoRA-A `dA` helper definition is deleted
- native wrapper tests pass on SM100
- E2E counters show projection-count grouped calls
- smoke profile finishes first backward

### Stage 4: Backward Scheduling And Manager Lifetime

Files:

- `asym_gemm/training/qwen3_moe.py`
- `asym_gemm/training/activation_offload.py`

Implement:

- reorder backward so one `[dgate_cpu, dup_cpu]` stage feeds gate/up LoRA-B
  first and gate/up base `dX` second
- release low-rank and wide stages immediately after use
- make `ActivationOffloadManager.release_cpu(temp_handle)` return scratch to a
  pool or drop the reference instead of retaining every temporary
- track `max_stage_bytes_live`, `num_stages`, and stage bytes by tag

Validation:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py \
  -k "activation_offload or manager_tracks_cpu_owners"

PYTHONPATH="$PWD" python scripts/testing/check_exp_act_offload_structure.py \
  --require final

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" \
python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 4096 \
  --top-k 8 \
  --num-experts 128 \
  --hidden-dim 2048 \
  --intermediate-dim 4096 \
  --rank 64 \
  --warmup 1 \
  --iters 1 \
  --output-json /tmp/qwen3_expact_stage4.json
```

Pass condition:

- activation offload finishes backward
- peak HBM is lower than the non-activation-offload variant
- stage accounting shows bounded local stages, not hidden wide copies
- AsymGEMM call counters do not show per-group explosion

### Stage 5: End-To-End LF Comparison

Run after Stage 4 passes.

Files:

- `scripts/lf/profile_lora_lf.sh`
- `scripts/lf/run_lf_lora_sft.sh`

Implement:

- no new core code in this stage
- verify wrapper/run-script wiring records `ASYMM_EXPERT_ACT_OFFLOAD=false,true`
  as `expact0` and `expact1`
- compare short and long Qwen3 LF runs

Validation:

Short sequence:

```bash
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ROUTER_MODES=whole \
EXPERT_POLICIES=none \
SEQ_LENS=512 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
ASYMM_EXPERT_ACT_OFFLOAD=false,true \
PROFILERS=source \
PROFILE_LEVEL=op \
PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Long sequence:

```bash
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ROUTER_MODES=whole \
EXPERT_POLICIES=none \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
ASYMM_EXPERT_ACT_OFFLOAD=false,true \
PROFILERS=source \
PROFILE_LEVEL=op \
PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false \
OVERWRITE=true \
scripts/lf/profile_lora_lf.sh
```

Pass condition:

- `expact0` and `expact1` both complete
- output paths include `expact0` and `expact1`
- `expact1` shows meaningful lower peak HBM on long sequence
- step time is not qualitatively blowing up
- no retired-path counters or recorded reasons indicate per-group
  activation-offload LoRA

Inspection helper:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("outputs").rglob("*profile.json")):
    if "expact" not in str(path):
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    cfg = data.get("config", {})
    metrics = data.get("summary", data.get("metrics", {}))
    print(path)
    print("  expact:", cfg.get("asymm_expert_act_offload"))
    print("  peak_hbm:", metrics.get("peak_hbm_gib") or metrics.get("peak_allocated_gib"))
    print("  step_time:", metrics.get("step_time_s") or metrics.get("mean_step_time_s"))
PY
```

## Final Acceptance Checklist

- `mlp_math.md` stays math-only.
- Normal `ASYMM_EXPERT_ACT_OFFLOAD=true` has no per-active-group GEMM loop.
- Normal `ASYMM_EXPERT_ACT_OFFLOAD=true` has no CPU FP32 per-active-group
  LoRA-B gradient loop.
- Wide forward activations survive as CPU handles.
- LoRA-A forward uses existing grouped CPU-left AsymGEMM.
- LoRA-A forward is integrated only in the Qwen3 expert activation-offload path,
  not as a universal replacement for normal CUDA LoRA.
- CPU-left usage is proven by calls to `grouped_expert_lora_cpu_left`, not by
  old helper stats.
- LoRA-A `dA` uses AsymGEMM-family CPU-right grouped-reduction kernels.
- Gate/up LoRA-B backward uses the single staged `[dgate, dup]` tensor and
  grouped CUDA LoRA-B `dS`/`dB`, with no per-expert GEMMs.
- CPU-source LoRA-B is not used in the hot path until it has a tiled
  tensor-core-quality implementation.
- Wide `[M,H]` and `[M,I]` staging is limited to local base-GEMM needs.
- Qwen3 unit tests, structure checks, native wrapper tests, and profile
  comparisons pass.
