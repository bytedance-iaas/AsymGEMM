# Activation Offload and Staging for AsymGEMM LoRA SFT Experts

This is an implementation design, not an implementation patch. The target is a
memory-first activation offload path for the packed Qwen3 MoE expert body used
by AsymGEMM LoRA SFT.

The goal is not to make CPU offload invisible in latency. The goal is to reduce
peak HBM by keeping expert activations in pinned CPU memory after forward and
staging them back to HBM only when GPU GEMMs need them.

## Bottom Line

Use a small activation offload manager with two public primitives:

```python
handle = offload(tensor, tag)          # GPU tensor -> pinned CPU owner; no HBM tensor saved
handle = empty_cpu(shape, dtype, device, tag)  # pinned CPU owner for CPU-produced activations/grads
view = stage(handle, rows=None)        # pinned CPU tensor/rows -> reusable HBM staging buffer
```

For the first memory-first version, these copies can be blocking. Do not start
with complex async overlap. Still use pinned CPU buffers and reusable HBM
staging buffers so the interface is compatible with a later async version.

The first proof point should not require row-window scheduling. Build it as:

```text
v0: offload full activations to CPU, then stage the full needed tensor back to
    HBM only for the next GPU GEMM, and release it immediately after use.

v1: after v0 proves a real HBM drop with acceptable latency, replace selected
    full stages with row-window stages to reduce the remaining staging peak.
```

In this doc, `stage(window)` means the v1 optimization: copy only a row slice
from a CPU handle into a reusable HBM scratch tensor. It is not required to make
the initial idea work.

The important design difference from Megatron is:

```text
Megatron general activation offload:
  save activation on CPU -> reload activation to GPU for normal backward

AsymGEMM expert activation offload:
  save activation on CPU -> do cheap elementwise CPU work on CPU
  -> stage only the tensor needed by the next GPU GEMM/AsymGEMM call
```

This is what makes the design AsymGEMM-specific and memory-first. Row-window
staging is a later way to make that staging tensor smaller, not a requirement
for the first implementation.

## Readiness Verdict

After checking the local Qwen3 expert code, Megatron-LM activation offload, and
DeepSpeed offload/checkpointing code, the original design direction is sound but
the first draft was not complete enough to implement carefully. The critical
gaps were:

- CPU-produced tensors such as `act_cpu`, `dgate_cpu`, and `dup_cpu` need pinned
  CPU owners, not ad hoc pageable CPU temporaries.
- LoRA dropout must be present in the main forward/backward math, not only in a
  side note. The saved `S_gate`, `S_up`, and `S_down` low-rank tensors are
  dropout-applied low-rank tensors.
- The stage plan must name exact functions/classes per stage and include
  commands that validate each stage before moving on.
- Row-window v1 is the natural next memory stage after full-staging v0 proves
  correctness; async overlap and fused add-and-offload are later optimizations,
  not prerequisites for the first correct path.

This revised plan is intended to be implementation-ready.

## Current Code Facts

### Qwen3 expert layout

`asym_gemm/training/qwen3_moe.py` uses packed expert base weights:

```text
gate_up_proj: [E, 2I, H]
down_proj:    [E, H, I]
```

When `backend == "asym"` and `offload == True`, both are stored in
`AsymGroupedFrozenLinear` with CPU pinned `HostWeight`s. The constructor already
requires CPU-first loading and bf16 source weights in strict mode.

The current forward computes:

```python
gate_up = self.gate_up_base(x, offsets, experts)
gate, up = gate_up.chunk(2, dim=-1)
gate_delta, up_delta = gate/up LoRA
gate = gate + gate_delta
up = up + up_delta
activated = self.act_fn(gate) * up
down = self.down_base(activated)
down_delta = down LoRA
```

The current expert-policy autograd function saves HBM tensors through
`ctx.save_for_backward(...)`, including `packed`, saved `gate`, saved `up`,
saved `activated`, and the low-rank LoRA intermediates. Backward restores full
HBM views and computes full-HBM `grad_activated`, `grad_gate`, `grad_up`, and a
full `[M, 2I]` `grad_gate_up` before the fused base backward.

The current base gate/up backward is already mathematically fused:

```python
grad_gate_up = concat(grad_gate, grad_up)
grad_packed = _grouped_base_dx(layer.gate_up_base, grad_gate_up, ...)
```

This fused layout must be preserved.

### Current AsymGEMM Constraint: No CPU-Left GEMM in This Design

Current AsymGEMM supports the CPU tensor as the right operand. The left operand
must be HBM/CUDA. The existing implementation supports transpose mode:

```text
A @^ B.T   -> transpose_b=False
A @^ B     -> transpose_b=True
```

That transpose option is enough when the CPU tensor is the right operand, but
it does not make a CPU-left operand legal. For this design, CPU-resident LoRA-A
inputs use the right-operand algebraic orientation from `agent/mlp_math.md`:

Allowed pattern:

```text
D(U_cpu)                                  # [M, K] CPU mask/scale
S_T = A @^ D(U_cpu).T                     # [r, M] HBM
S = row_major(S_T.T)                      # [M, r] HBM, small low-rank materialization

dA = dS.T @^ D(U_cpu)                     # [r, K] Grad, CPU operand on right
dB = dOut.T @^ S_cpu                      # [O, r] Grad, CPU operand on right

act = stage(act_cpu)                      # [M, I] HBM, needed for down base
Y_base = act @^ W_down_cpu.T
dgate_up = stage(concat(dgate_cpu, dup_cpu))
dX_base = dgate_up @^ W_gate_up_cpu
```

Do not implement or depend on:

```text
act_cpu @^^ A_down.T
dgate_cpu @^^ B_gate
GPU kernel streaming both operands from CPU
staging X_cpu only to compute dA_gate/dA_up
staging act_cpu only to compute dA_down
staging S_*_cpu only to compute dB_*
```

`@^^` is deliberately deferred to a separate design. This doc should only use
the existing `@^` transpose option. The only wide HBM stages in the core path
should be for base frozen GEMMs and for `dgate_up`; LoRA A/B weight reductions
should use CPU operands directly on the `@^` side.

### LoRA weights stay on GPU

The LoRA expert weights remain normal trainable GPU parameters. The offload
path targets activations and low-rank saved tensors, not LoRA parameter
residency. LoRA GEMMs and LoRA weight-gradient reductions still run on GPU.

## Lessons From DeepSpeed

DeepSpeed's useful lessons are about memory ownership and lifecycle, not about
copying an exact activation design:

1. Use pinned CPU memory for offload buffers.
   `deepspeed/runtime/zero/offload_states.py` uses pinned CPU buffers and
   `copy_()` for offloaded optimizer state when pinning is requested.

2. Use preallocated or reusable buffers instead of allocating at each transfer.
   ZeRO-Offload maintains partition and temporary buffers for CPU offload rather
   than repeatedly moving individual tensors with ad hoc `.to()` calls.

3. Use prefetch/release policy around known future use.
   `PartitionedParameterCoordinator` scans the parameter trace, prefetches into
   an available window, and releases parameters after their reuse distance no
   longer justifies keeping them live.

4. Contiguous buffers matter.
   DeepSpeed activation checkpointing has a contiguous checkpointing path and
   even pre-populates CPU pages before the real transfer. The practical lesson
   is that many small or lazy page faults are bad; allocate and touch stable
   offload buffers during setup or warmup.

5. Trace-based prefetch is useful only when the future use order is generic.
   `PartitionedParameterCoordinator` records the module/parameter trace, uses a
   prefetch bucket, tracks available bytes, and releases parameters based on
   reuse distance. For the Qwen3 expert body we already know the exact tensor
   order, so use a fixed local schedule instead of a generic trace engine.

6. Replacing storage is a last-mile memory action, not a correctness mechanism.
   ZeRO offload clears or replaces parameter data only after references are
   controlled. For activation offload v0, prefer not saving HBM tensors and
   dropping Python references. Use storage resize only in a later audited pass.

For AsymGEMM activation offload, copy the buffer discipline, not the ZeRO param
logic. We know the exact expert tensor lifetimes, so a custom manager is simpler
and more memory efficient than a generic ZeRO-like scheduler.

## Lessons From Megatron-LM

Megatron-LM already has activation offload:

- `cpu_offloading`: layer-level activation offload.
- `fine_grained_activation_offloading`: module-level activation offload for
  selected modules such as `expert_fc1` and `moe_act`.

The fine-grained path uses:

- saved tensor hooks;
- a CPU pinned tensor pool;
- separate D2H/H2D streams;
- CUDA events for offload and reload completion;
- grouped offload/reload by module name;
- min tensor-size filtering;
- optional forced HBM storage release through `untyped_storage().resize_(0)`.

Megatron also treats activation offload as mutually exclusive with several
features when lifetimes become ambiguous. Its config rejects combinations such
as fine-grained activation offload with generic CPU offload, MoE paged stash with
`expert_fc1`/`moe_act` offload, and some CUDA graph/token-drop combinations.
For this repo, the equivalent lesson is to fail closed: activation offload v0
should reject active expert recompute/activation-drop policies and unsupported
backends instead of silently falling back inside the custom autograd function.

This is the closest existing design, but it is still not the right exact path
for AsymGEMM LoRA SFT. Megatron reloads tensors back to HBM so normal GPU
backward can run. Our path should keep `gate`, `up`, `act`, `dact`, `dgate`,
and `dup` CPU-resident as much as possible, then stage only the tensor required
by the next GEMM. Start with full-tensor staging; row-window staging is a later
peak-memory refinement.

Also, this codebase already has a custom expert autograd function, so we do not
need a generic saved-tensor-hook solution for the first implementation. Storing
explicit CPU handles on `ctx` is simpler and gives better control over row
staging lifetime.

## Design Target

The memory-first path should make these tensors CPU-primary after forward:

```text
X_cpu         [M, H]   needed for dA_gate and dA_up
gate_cpu      [M, I]   needed for activation backward
up_cpu        [M, I]   needed for activation backward
act_cpu       [M, I]   needed for down forward/backward and dA_down
S_gate_cpu    [M, r]   dropout-applied low-rank, needed for dB_gate
S_up_cpu      [M, r]   dropout-applied low-rank, needed for dB_up
S_down_cpu    [M, r]   dropout-applied low-rank, needed for dB_down
```

It should avoid saving these full tensors in HBM:

```text
gate
up
activated / act
grad_activated / dact
grad_gate / dgate
grad_up / dup
grad_gate_up [M, 2I]
```

CPU-produced tensors that will later be staged, especially `act_cpu`,
`dgate_cpu`, and `dup_cpu`, should be written into pinned CPU buffers owned by
the manager. Do not let these become anonymous pageable tensors from normal
PyTorch CPU operations.

The unavoidable HBM tensors are:

```text
forward output Y       [M, H]
backward input dY      [M, H]
backward output dX     [M, H]
LoRA parameters/grads  trainable GPU parameters
small metadata/masks   offsets, experts, dropout masks, row plans
```

## Exact Implementation Scope

Keep v0 scoped to the Qwen3 packed expert path only. Do not touch generic dense
LoRA, generic MoE, KT backends, optimizer offload, or the router path.

### Files to add

`asym_gemm/training/activation_offload.py`

- `CPUHandle`
- `ActivationOffloadManager`
- blocking `offload`, `empty_cpu`, `adopt_cpu`, `offload_into`, `cpu_view`,
  `stage`, `stage_into`, `stage_concat_columns`, and `release_stage`
- debug counters: `offloaded_bytes`, `staged_bytes`, `max_stage_bytes_live`,
  `cpu_owned_bytes`, `num_offloads`, `num_cpu_allocs`, `num_stages`

`tests/training/test_activation_offload_manager.py`

- CPU pinned allocation when CUDA is available
- pinned CPU owners for CPU-produced tensors through `empty_cpu` / `adopt_cpu`
- HBM staging buffer reuse
- `offload_into` row overwrite correctness
- `stage_concat_columns` preserves the packed gate/up column order
- no hidden `.to()` allocation in manager hot paths

`scripts/testing/validate_qwen3_activation_offload.py`

- deterministic Qwen3 expert correctness and memory probe
- should build both current AsymGEMM and activation-offload variants from the
  same fake Qwen3 expert weights
- should emit JSON with `max_abs`, `rel_l2`, all LoRA grad errors,
  `peak_hbm_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`, `step_ms`,
  manager counters, and selected variant labels
- should calculate `peak_delta_bytes`, `peak_delta_pct`,
  `slowdown_vs_current`, `max_stage_bytes_live`, and `stage_peak_by_tag` for
  each measured activation-offload variant
- should report `saved_hbm_activation_tags` or an equivalent debug list proving
  that the custom autograd context did not save wide expert activations
- should not bake arbitrary memory or timing cutoffs into the script; emit
  enough data for the stage handoff to judge whether HBM went down and timing
  did not obviously blow up

### Files to modify

`asym_gemm/training/qwen3_moe.py`

- import the activation offload manager
- add `_uses_activation_offload(self) -> bool` on `AsymQwen3Experts`
- add `_check_activation_offload_supported(self, packed, offsets, experts) -> None`
- add `_forward_expert_body_with_activation_offload(...)` beside
  `_forward_expert_body_with_intermediates(...)`
- add `_forward_expert_activation_offload(...)` beside `_forward_expert_policy(...)`
- add `_ActivationOffloadQwen3ExpertFunction(torch.autograd.Function)` beside
  the existing `_ThresholdedQwen3ExpertFunction`
- keep the custom autograd signature parallel to the existing expert-policy
  function, minus recompute masks:

```python
_ActivationOffloadQwen3ExpertFunction.apply(
    packed,
    offsets,
    experts,
    self.gate_lora_A,
    self.gate_lora_B,
    self.up_lora_A,
    self.up_lora_B,
    self.down_lora_A,
    self.down_lora_B,
    self,
)
```

Backward must return:

```python
(
    grad_packed,
    None,
    None,
    grad_gate_lora_A,
    grad_gate_lora_B,
    grad_up_lora_A,
    grad_up_lora_B,
    grad_down_lora_A,
    grad_down_lora_B,
    None,
)
```

- update both `AsymQwen3Experts.forward(...)` and
  `AsymQwen3Experts.forward_input_scaled(...)` to select activation offload
  before recompute:

```python
if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(packed, offsets, experts, metadata)
elif self._uses_expert_recompute():
    down = self._forward_expert_policy(packed, offsets, experts, metadata)
else:
    down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
```

`tests/training/test_lf_qwen3_asym_backend.py`

- add activation-offload tests next to the existing SM100 Qwen3 tests
- do not weaken existing recompute/dropout tests
- new tests should compare activation offload against the current AsymGEMM path
  and, for at least one case, against the torch backend

`scripts/lora/profile_lora_e2e.py`

- only if needed for later profiling: record
  `ASYM_QWEN3_EXPERT_ACT_OFFLOAD` in the profile config and include manager
  counters in `memory` / `stage_memory`
- do not use this script as the first correctness gate; use the dedicated
  validation script above

### Feature gate

Use only explicit opt-in for v0:

```text
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1
```

The default path must remain behaviorally unchanged. With the flag unset or
`0`, no activation-offload manager should be constructed and no new autograd
function should be used.

## Offload Manager API

Add a small internal module, for example:

```text
asym_gemm/training/activation_offload.py
```

Core objects:

```python
@dataclass
class CPUHandle:
    tag: str
    tensor: torch.Tensor          # pinned CPU owner
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device          # original CUDA device
    pinned: bool

class ActivationOffloadManager:
    def offload(self, tensor: torch.Tensor, tag: str) -> CPUHandle: ...
    def empty_cpu(self, shape, dtype, device, tag: str) -> CPUHandle: ...
    def adopt_cpu(self, tensor: torch.Tensor, device, tag: str) -> CPUHandle: ...
    def offload_into(self, handle: CPUHandle, rows: slice, tensor: torch.Tensor) -> None: ...
    def cpu_view(self, handle: CPUHandle, rows: slice | None = None) -> torch.Tensor: ...
    def stage(self, handle: CPUHandle, rows: slice | None = None, *, tag: str) -> torch.Tensor: ...
    def stage_into(self, src_cpu: torch.Tensor, dst: torch.Tensor) -> torch.Tensor: ...
    def stage_concat_columns(self, left_cpu, right_cpu, rows: slice | None = None, *, tag: str) -> torch.Tensor: ...
    def release_stage(self, tensor: torch.Tensor) -> None: ...
```

Implementation policy:

1. CPU buffers are pinned when CUDA is available:

   ```python
   torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
   ```

   If CUDA is unavailable, keep the same API and allocate normal CPU tensors.
   Tests should assert `handle.pinned == torch.cuda.is_available()` only when
   pinning succeeds.

2. Use `copy_`, not `.to()`, for hot-path transfers:

   ```python
   cpu_buf.copy_(gpu_tensor, non_blocking=False)
   gpu_buf.copy_(cpu_view, non_blocking=False)
   ```

   Blocking copies are acceptable in v0. The important part is avoiding hidden
   allocation and making lifetime explicit.

3. `empty_cpu(...)` is for CPU-produced tensors such as `act_cpu`,
   `dgate_cpu`, and `dup_cpu`. CPU code should write into those buffers with
   `out=` operations or copy the result into them immediately.
   `adopt_cpu(...)` may wrap an existing CPU tensor only after making it
   contiguous and pinned if possible.

4. Reuse HBM staging buffers by `(shape, dtype, device, tag_class)`. The Qwen3
   path should call `stage(...)` only for these intentional wide stages:

   ```text
   act_for_down_base            [M or mw, I]
   dgate_up_for_gate_up_base    [M or mw, 2I]
   ```

   These are CPU-only handles, not staging targets for LoRA gradients:

   ```text
   X_cpu                         [M, H]
   act_cpu                       [M, I]
   S_gate_cpu / S_up_cpu         [M, r]
   S_down_cpu                    [M, r]
   ```

   HBM compute temporaries such as `dX_w [mw,H]`, `LoRA_down [mw,H]`, and
   `dS_* [mw,r]` are normal GPU outputs of GEMMs, not manager stages. The
   validation script must fail if `stage_peak_by_tag` contains `X_cpu`,
   `act_cpu_for_dA`, `S_*_cpu`, or other wide tags not listed above.

5. For v0, stage full tensors. For later row-window staging, do not stage
   arbitrary index selections. Build row windows as contiguous slices in packed
   expert-row order, so CPU views are cheap and contiguous.

6. The manager may use blocking copies in v0/v1. It should still centralize all
   copies so a later async implementation only changes the manager internals.

7. Do not call `untyped_storage().resize_(0)` in the first implementation unless
   we can prove the tensor is private and has no aliases. Dropping all Python
   references and not saving the HBM tensor in `ctx` is the safe first step.

8. In v0/v1, all staging and compute should run on the current CUDA stream.
   Reusing a released stage buffer is then stream-ordered. Async D2H/H2D streams
   and CUDA events belong to a later stage.

## Row Window Policy

This section is for the later v1 optimization, not for the first proof point.
Windowing must respect grouped expert metadata. Current grouped kernels expect
packed rows and local `offsets`/`experts`. Therefore, windows should be built
from the existing `offsets` and `experts`, not from arbitrary row indices.

For v1, use per-expert contiguous chunks:

```python
for group_idx, expert in enumerate(experts[:-1]):
    group_start = offsets[group_idx]
    group_end = offsets[group_idx + 1]
    for start in range(group_start, group_end, window_rows):
        end = min(start + window_rows, group_end)
        local_offsets = [0, end - start]
        local_experts = [expert, -1]
        rows = slice(start, end)
```

This is simple and correct. Later, adjacent expert chunks can be coalesced into
a larger window with local multi-group offsets.

The window size should be configured by rows or bytes, for example:

```text
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128
ASYM_ACT_OFFLOAD_WINDOW_BYTES=...
```

The memory target should be:

```text
peak stage memory ~= max_window_rows * max(H, I, 2I) * dtype_size
```

not:

```text
M * I * dtype_size
```

## Forward Math

Use CPU-primary LoRA-A inputs from the start. The implementation can still
materialize the small low-rank `[M, r]` tensors in HBM because current LoRA-B
helpers consume row-major `[M, r]`, but it should not materialize the wide
dropout-applied `[M, H]` or `[M, I]` LoRA-A inputs in HBM. For v0, stage full
`act_cpu` only for the down base projection and release it immediately after
use. The pseudocode below is dropout-aware: `D_gate`, `D_up`, and `D_down` mean
the current packed LoRA dropout masks and scaling, or identity when
`lora_dropout_p == 0`.

```python
X = routed_rows_for_this_expert                         # [M, H] HBM

gate_up_base = X @^ W_gate_up_cpu.T                     # [M, 2I] HBM temp
gate_base, up_base = split(gate_up_base)                # views [M, I]

X_cpu = offload(X)                                      # [M, H] CPU

X_gate_lora_cpu = D_gate(X_cpu)                         # [M, H] CPU
S_gate_T = A_gate @^ X_gate_lora_cpu.T                  # [r, M] HBM
S_gate = row_major(S_gate_T.T)                          # [M, r] HBM

X_up_lora_cpu = D_up(X_cpu)                             # [M, H] CPU
S_up_T = A_up @^ X_up_lora_cpu.T                        # [r, M] HBM
S_up = row_major(S_up_T.T)                              # [M, r] HBM

LoRA_gate = scale * (S_gate @ B_gate.T)                 # [M, I] HBM temp
gate = gate_base + LoRA_gate                            # [M, I] HBM
gate_cpu = offload(gate)

LoRA_up = scale * (S_up @ B_up.T)                       # [M, I] HBM temp
up = up_base + LoRA_up                                  # [M, I] HBM
up_cpu = offload(up)

S_gate_cpu = offload(S_gate)
S_up_cpu   = offload(S_up)

act_cpu = empty_cpu([M, I])
sig_cpu = sigmoid(gate_cpu)                             # [M, I] CPU temp
silu_gate_cpu = sig_cpu * gate_cpu                      # [M, I] CPU temp
act_cpu.copy_(silu_gate_cpu * up_cpu)                   # [M, I] CPU pinned owner
```

Then compute down by staging the full activation for the first prototype:

```python
act_down_lora_cpu = D_down(act_cpu)                     # [M, I] CPU
S_down_T = A_down @^ act_down_lora_cpu.T                # [r, M] HBM
S_down = row_major(S_down_T.T)                          # [M, r] HBM
S_down_cpu = offload(S_down)                            # save for dB_down

LoRA_down = scale * (S_down @ B_down.T)                 # [M, H] HBM temp, same dtype as Y

act = stage(act_cpu, tag="act_for_down_base")           # [M, I] HBM, needed for down base
Y_base = act @^ W_down_cpu.T                            # [M, H] HBM
Y = Y_base                                              # [M, H] HBM output owner
Y.add_(LoRA_down)                                       # no extra [M, H] output tensor

release act, S_down, LoRA_down
```

Notes:

- No `@^^` is part of v0/v1. Even for `S_down`, keep the dropped activation on
  CPU and use `A_down @^ D_down(act_cpu).T`, then materialize only `[M, r]`.
- The first prototype can stage the full `act_cpu` for `Y_base`. If it proves
  the memory idea but still has too much staging peak, v1 can replace this full
  stage with row-window staging.
- Do the down output add in place into `Y_base` inside the custom autograd
  forward. Do not keep `Y_base`, `LoRA_down`, and a third full `[M,H]` sum live
  at the same time.
- For v0, require LoRA compute dtype to match the packed/output dtype. Hidden
  full-size cast temporaries such as `LoRA_down.to(dtype=Y.dtype)` defeat the
  memory accounting and must be rejected or measured explicitly.
- In a later row-window version, every grouped `@^` call must pass the window's
  local `offsets`/`experts`.
- `S_gate_cpu`, `S_up_cpu`, and `S_down_cpu` are the dropout-applied low-rank
  tensors saved for `dB_*`.
- `act_cpu` is saved for `dA_down`.
- `silu_gate_cpu` and `sig_cpu` do not need to be saved. Recompute them on CPU
  during backward to reduce CPU memory and simplify lifetime.

## Backward Math

Backward should use the same staged sequence. For v0, treat `rows` below as all
active rows and stage full tensors. For v1, split `rows` into expert-row windows
to reduce the staging peak. The math is the same; only staging granularity
changes.

Initialize LoRA gradient accumulators on HBM:

```python
dA_gate = zeros_like(A_gate)
dB_gate = zeros_like(B_gate)
dA_up   = zeros_like(A_up)
dB_up   = zeros_like(B_up)
dA_down = zeros_like(A_down)
dB_down = zeros_like(B_down)
dX      = empty_hbm([M, H])
```

For each staged row block:

```python
dY_w = dY[rows]                                         # [mw, H] HBM view/copy

# down backward
dS_down_w = scale * (dY_w @ B_down)                     # [mw, r] HBM
dact_lora_raw_w = dS_down_w @ A_down                    # [mw, I] HBM
dact_lora_w = D_down_bar(dact_lora_raw_w)               # [mw, I] HBM
dact_base_w = dY_w @^ W_down_cpu                        # [mw, I] HBM
dact_w = dact_base_w + dact_lora_w                      # [mw, I] HBM

dact_cpu_w_handle = offload(dact_w, "dact_w")            # temp CPU handle
dact_cpu_w = dact_cpu_w_handle.tensor                   # [mw, I] CPU

act_down_lora_cpu_w = D_down(act_cpu[rows])             # [mw, I] CPU
S_down_cpu_w = S_down_cpu[rows]                         # [mw, r] CPU

dA_down += dS_down_w.T @^ act_down_lora_cpu_w           # [r, I]
dB_down += scale * (dY_w.T @^ S_down_cpu_w)             # [H, r]

release dact_w temporaries
```

Activation backward on CPU:

```python
gate_cpu_w = gate_cpu[rows]
up_cpu_w = up_cpu[rows]

sig_w = sigmoid(gate_cpu_w)
silu_gate_w = sig_w * gate_cpu_w
silu_grad_w = sig_w * (1 + gate_cpu_w * (1 - sig_w))

dgate_cpu_w = empty_cpu([mw, I], tag="dgate_w")
dup_cpu_w   = empty_cpu([mw, I], tag="dup_w")
dgate_cpu_w.copy_(dact_cpu_w * up_cpu_w * silu_grad_w)  # [mw, I] CPU pinned
dup_cpu_w.copy_(dact_cpu_w * silu_gate_w)               # [mw, I] CPU pinned
```

Gate/up base backward should stay fused:

```python
dgate_up_w = stage_concat_columns(
    dgate_cpu_w,
    dup_cpu_w,
    tag="dgate_up_for_gate_up_base",
)                                                            # [mw, 2I] HBM
dX_w = dgate_up_w @^ W_gate_up_cpu                         # [mw, H] HBM
```

Then add LoRA input-gradient and parameter-gradient contributions:

```python
dgate_w = dgate_up_w[:, :I]                              # HBM view
dup_w   = dgate_up_w[:, I:]                              # HBM view
X_gate_lora_cpu_w = D_gate(X_cpu[rows])                  # [mw, H] CPU
X_up_lora_cpu_w   = D_up(X_cpu[rows])                    # [mw, H] CPU

# gate LoRA
dS_gate_w = scale * (dgate_w @ B_gate)                   # [mw, r]
dX_gate_raw_w = dS_gate_w @ A_gate                      # [mw, H]
dX_w += D_gate_bar(dX_gate_raw_w)                        # [mw, H]
S_gate_cpu_w = S_gate_cpu[rows]                          # [mw, r] CPU
dA_gate += dS_gate_w.T @^ X_gate_lora_cpu_w              # [r, H]
dB_gate += scale * (dgate_w.T @^ S_gate_cpu_w)           # [I, r]
release dX_gate_raw_w, dS_gate_w

# up LoRA
dS_up_w = scale * (dup_w @ B_up)                         # [mw, r]
dX_up_raw_w = dS_up_w @ A_up                            # [mw, H]
dX_w += D_up_bar(dX_up_raw_w)                            # [mw, H]
S_up_cpu_w = S_up_cpu[rows]                              # [mw, r] CPU
dA_up += dS_up_w.T @^ X_up_lora_cpu_w                    # [r, H]
dB_up += scale * (dup_w.T @^ S_up_cpu_w)                 # [I, r]
release dX_up_raw_w, dS_up_w

dX[rows] = dX_w

release dX_w, dgate_up_w, dgate_cpu_w, dup_cpu_w, dact_cpu_w_handle
```

This preserves the true math and the current packed gate/up base layout. It
also avoids ever materializing full-HBM `dact`, `dgate`, `dup`, or
`dgate_up`.

As in forward, `A_*` and `B_*` in the pseudocode mean the selected expert LoRA
weight slices when processing one expert group. If a later implementation
coalesces multiple experts in one row window, these operations must use local
grouped metadata and selected expert weight slices.

## Dropout Policy

The current expert path already supports LoRA dropout with packed saved masks:
`gate_mask_packed`, `up_mask_packed`, and `down_mask_packed` are produced in
forward and consumed in backward through `_apply_saved_dropout` /
`_apply_saved_dropout_`. The activation-offload path must preserve that
semantics and should not add a new zero-dropout-only restriction.

- gate/up LoRA A gradients use dropout-applied `X`;
- down LoRA A gradients use dropout-applied `act`;
- LoRA input-gradient contributions must apply the dropout backward mask;
- packed dropout masks should be saved as compact metadata because they are much
  smaller than `[M, I]` activations.

For the staged offload path, select the same rows from the packed mask and apply
the existing helper semantics before the affected GEMM:

```text
X_lora_cpu_w      = D_gate_or_up(X_cpu[rows])
act_lora_cpu_w    = D_down(act_cpu[rows])
dX_lora_raw_w  = ...
dX_lora_w      = D_gate_or_up_bar(dX_lora_raw_w)
dact_lora_raw_w = ...
dact_lora_w     = D_down_bar(dact_lora_raw_w)
```

Do not silently ignore LoRA dropout. If `lora_dropout_p > 0`, the packed masks
must be present, row-selectable, and applied in the same places as the current
path. The dropout-applied forward inputs for LoRA-A gradients stay CPU-side;
do not stage the wide `[mw,H]` or `[mw,I]` dropped inputs just to compute
`dA_*`.

## Recompute Policy Interaction

Activation offload is an alternative to activation recomputation for the expert
body. For the first implementation, do not combine it with the existing expert
recompute policy for the same active expert rows.

Recommended v0/v1 behavior:

```text
if ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1:
    require expert_recompute_policy disables recompute for routed experts
```

or make the new mode a distinct policy:

```text
expert_recompute_policy=act_offload
```

Combining recompute and CPU activation offload makes lifetime and memory
accounting hard to reason about and weakens the purpose of the feature.

## Stage-Gated Implementation Plan

Do not start a stage until the previous stage's validation commands pass. Each
stage should leave behind either a pytest result or a JSON artifact so a bad
assumption is caught before the next layer builds on it.

Stages 0-4 prove baseline correctness, implementation scope, and tensor
lifetime. They do not claim a peak-HBM win. Stage 5 is the first profile review
point for memory and timing. Stages 6-8 each add another profile artifact, but
they should not encode arbitrary numeric gates that block iteration.

### Stage Scope Matrix

This matrix is the authoritative scope. If a change is not listed for the
current stage, defer it unless it is required to make that stage's listed tests
pass.

| Stage | Files | Functions/classes in scope | Out of scope |
| --- | --- | --- | --- |
| 0 Baseline | none | none | all edits |
| 1 Manager | `asym_gemm/training/activation_offload.py`, `tests/training/test_activation_offload_manager.py` | `CPUHandle`, `ActivationOffloadManager.offload`, `empty_cpu`, `adopt_cpu`, `offload_into`, `cpu_view`, `stage`, `stage_into`, `stage_concat_columns`, `release_stage`, counters | Qwen3 math, autograd functions, async streams |
| 2 Forward | `asym_gemm/training/qwen3_moe.py`, validation script | `AsymQwen3Experts._uses_activation_offload`, `_check_activation_offload_supported`, `_forward_expert_body_with_activation_offload`, `_forward_expert_activation_offload`, `_ActivationOffloadQwen3ExpertFunction.forward`, `forward`, `forward_input_scaled` | backward, row windows, async copies |
| 3 Backward | `qwen3_moe.py`, Qwen3 tests, validation script | `_ActivationOffloadQwen3ExpertFunction.backward`, local staged LoRA backward helpers if needed, dropout row-selection helpers if needed | row-window staging, fused add-and-offload |
| 4 Gating/stats | `qwen3_moe.py`, tests | support checks, env parsing, stats/counter exposure on `AsymQwen3Experts` | profiler integration beyond counters |
| 5 Memory proof | validation script, optional profile script | JSON output schema, peak HBM measurement, variant runner | new math |
| 6 Row-window v1 | manager, `qwen3_moe.py`, tests, validation script | row-window iterator from `offsets`/`experts`, local window metadata, row-window backward and selected forward down staging | async streams, fused forward add |
| 7 Forward peak polish | manager, `qwen3_moe.py`, tests | optional chunked/fused add-and-offload for `gate`/`up`; optional direct writes into pinned owners | CPU GEMMs for base/LoRA, `@^^` kernels |
| 8 Async overlap | manager only first, then Qwen3 path | D2H/H2D streams, events, pinned-pool warmup, prefetch/release scheduling | changing math or validation tolerances |

Stage 7 and Stage 8 are later optimization stages. Stages 1-6 are enough for a
careful implementation of extensive activation offload with row-windowed HBM
staging.

### Stage 0: Baseline lock

Purpose: prove the current Qwen3 path is green before changing it.

Allowed changes: none.

Validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_backward_matches_torch_backend \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_recompute_policies_match_none \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_recompute_lora_dropout_matches_none
```

Pass criteria:

- tests pass on an SM100 BF16 AsymGEMM machine;
- no activation-offload env var is set;
- this is the reference for later output/gradient comparisons.

### Stage 1: Manager only

Purpose: implement copy ownership without touching expert math.

Allowed changes:

- add `asym_gemm/training/activation_offload.py`;
- add `tests/training/test_activation_offload_manager.py`;
- do not modify `AsymQwen3Experts.forward(...)` yet.

Validation:

```bash
python -m pytest -q tests/training/test_activation_offload_manager.py
```

Pass criteria:

- CPU buffers are pinned when CUDA is available;
- `empty_cpu` returns pinned CPU owners for CPU-produced tensors;
- `adopt_cpu` either reuses an already pinned contiguous tensor or copies into
  a pinned contiguous owner;
- `stage(...)` copies into reusable HBM buffers;
- `stage_concat_columns(gate_cpu, up_cpu, ...)` preserves packed gate/up column
  order exactly;
- manager counters match expected bytes and live staging bytes;
- no existing Qwen3 tests regress:

```bash
python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_backward_matches_torch_backend
```

### Stage 2: Forward activation offload only

Purpose: verify the forward math and saved CPU handles before writing backward.

Allowed changes:

- in `asym_gemm/training/qwen3_moe.py`, add
  `_uses_activation_offload`, `_check_activation_offload_supported`,
  `_forward_expert_body_with_activation_offload`, and
  `_forward_expert_activation_offload`;
- add `_ActivationOffloadQwen3ExpertFunction.forward`;
- backward may raise `NotImplementedError`;
- update both `forward(...)` and `forward_input_scaled(...)` to select the new
  path only when `ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1`;
- keep full-tensor staging, not row-window staging.

Validation script to add and use:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode forward \
  --device cuda:0 \
  --tokens 16 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 7 \
  --compare-to current \
  --output-json reports/act_offload/stage2_forward.json
```

Dropout forward validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode forward \
  --device cuda:0 \
  --tokens 16 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 557 \
  --compare-to current \
  --output-json reports/act_offload/stage2_forward_dropout.json
```

Pass criteria:

- output `max_abs <= 6e-3` and `rel_l2 <= 2e-2`;
- `X`, `gate`, `up`, `act`, `S_gate`, `S_up`, and `S_down` are represented by
  CPU handles after forward;
- `S_gate`, `S_up`, and `S_down` handles contain the dropout-applied low-rank
  tensors when `lora_dropout_p > 0`;
- no full HBM `gate`, `up`, or `act` is saved on `ctx`;
- `ctx.save_for_backward(...)` contains LoRA parameter tensors, metadata, and
  packed dropout masks only; CPU handles live as normal `ctx` attributes;
- forward JSON includes manager counters, `saved_hbm_activation_tags`, and
  `stage_peak_by_tag`;
- `stage_peak_by_tag` contains no wide stage except `act_for_down_base`;
- with the env var unset, the current path still runs.

### Stage 3: Full-tensor staged backward

Purpose: verify true gradients before optimizing memory further.

Allowed changes:

- implement `_ActivationOffloadQwen3ExpertFunction.backward`;
- use full-tensor `stage(...)` only where current base GEMMs require HBM left
  operands: `act` for down base and `dgate_up` for gate/up base backward;
- do not stage `X_cpu`, `act_cpu`, or `S_*_cpu` just to compute LoRA
  `A/B` gradients; use the CPU-side operands with `@^`;
- implement local LoRA backward math instead of calling
  `_grouped_lora_backward_loop_free(...)` on CPU handles;
- preserve packed dropout masks through `_apply_saved_dropout` and
  `_apply_saved_dropout_`;
- do not add row-window staging yet.

Validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode backward \
  --device cuda:0 \
  --tokens 16 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --seed 11 \
  --compare-to current \
  --output-json reports/act_offload/stage3_backward.json
```

Dropout backward validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode backward \
  --device cuda:0 \
  --tokens 16 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --seed 557 \
  --compare-to current \
  --output-json reports/act_offload/stage3_backward_dropout.json
```

Pytest coverage to add and pass:

```bash
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_backward_matches_current \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_lora_dropout_matches_current \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_forward_input_scaled_matches_current
```

Pass criteria:

- `dX` and every `gate/up/down` LoRA `A/B` grad match current AsymGEMM within
  existing bf16 tolerances;
- dropout backward consumes no RNG;
- gate/up/down LoRA `A` gradients use dropout-applied CPU-side inputs;
- gate/up/down LoRA `B` gradients use saved dropout-applied low-rank handles;
- LoRA input-gradient contributions apply the same dropout masks before being
  added into `dX` / `dact`;
- `dgate_up` layout is `[dgate, dup]`, matching `W_gate_up`;
- full HBM `dact`, `dgate`, `dup`, and `[M, 2I] dgate_up` are not saved across
  backward phases;
- `stage_peak_by_tag` contains only the intentional wide stage tags
  `act_for_down_base` and `dgate_up_for_gate_up_base`;
- `saved_hbm_activation_tags` contains no `X`, `gate`, `up`, `act`, `dact`,
  `dgate`, `dup`, or `dgate_up`;
- base frozen weights still receive no gradients.

### Stage 4: Feature gating and failure modes

Purpose: make the feature hard to enable incorrectly.

Allowed changes:

- add strict support checks inside `_check_activation_offload_supported`;
- reject incompatible recompute/offload combinations;
- record manager counters on the layer or stats object for validation;
- keep default env-off behavior unchanged.

Required support checks:

- `self.training and torch.is_grad_enabled()`; otherwise use the normal forward;
- `backend == "asym"` and `offload is True`;
- `packed.device.type == "cuda"` and `packed.dim() == 2`;
- `packed.dtype == torch.bfloat16`;
- `_is_silu_activation(self.act_fn)`;
- `0.0 <= lora_dropout_p < 1.0`;
- `gate_up_base` and `down_base` are `AsymGroupedFrozenLinear`;
- both base host weights are CPU, contiguous, bf16, and pinned when CUDA is
  available;
- base precision is `"bf16"` and backend is `"asym"`;
- LoRA parameters are CUDA tensors on the same device as `packed`;
- LoRA parameter dtype and LoRA compute dtype match `packed.dtype` for v0, so
  forward adds do not allocate hidden full-size cast tensors;
- `offsets` and `experts` are 1D, contiguous, same device as `packed`, and use
  dense expert metadata for the v0 path;
- `expert_recompute_config.enabled` is false. This includes both recompute and
  activation-drop policies.

Validation:

```bash
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_backward_matches_torch_backend \
  tests/training/test_lf_qwen3_asym_backend.py::test_apply_lf_asym_lora_sm100_accumulates_and_optimizer_updates_only_lora
```

```bash
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_rejects_recompute_policy \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_requires_asym_backend
```

Pass criteria:

- unset env var exactly preserves old behavior;
- setting the env var rejects `backend="torch"`, non-CUDA packed tensors,
  non-`AsymGroupedFrozenLinear` bases, unpinned host base weights, and active
  expert recompute policies;
- error messages identify the failed precondition.

### Stage 5: Memory proof

Purpose: decide whether v0 is worth keeping before adding row windows.

Validation:

```bash
mkdir -p reports/act_offload
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode profile \
  --device cuda:0 \
  --tokens 2048 \
  --hidden-dim 1024 \
  --intermediate-dim 4096 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --warmup 5 \
  --iters 10 \
  --variants current,act_offload,recompute \
  --output-json reports/act_offload/stage5_v0_profile.json
```

Optional integration profiler after the dedicated script is green:

```bash
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
python scripts/lora/profile_lora_e2e.py \
  --workload Qwen/Qwen3-30B-A3B \
  --hf-local-files-only \
  --device cuda:0 \
  --backend asym \
  --warmup-steps 5 \
  --measure-steps 5 \
  --profile-layers 1 \
  --batch-size 1 \
  --seq-len 128 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --precision bf16 \
  --target-modules all \
  --offload-modules routed_experts \
  --expert-recompute-policy none \
  --output-dir reports/act_offload/profile_lora_e2e_qwen3_v0
```

Pass criteria:

- correctness fields in the profile JSON still pass Stage 3 tolerances;
- profile JSON shows whether `act_offload.peak_hbm_bytes` is below
  `current.peak_hbm_bytes`;
- JSON reports, for each variant, `peak_hbm_bytes`, `peak_allocated_bytes`,
  `peak_reserved_bytes`, `step_ms`, and `slowdown_vs_current`;
- JSON reports `current`, `act_offload`, `recompute`,
  `act_offload.peak_delta_bytes`, and `act_offload.peak_delta_pct`;
- compare against `recompute.peak_hbm_bytes`; if recompute beats activation
  offload, the artifact must say that explicitly before Stage 6 begins;
- manager counters include `offloaded_bytes`, `cpu_owned_bytes`,
  `staged_bytes`, `max_stage_bytes_live`, `stage_peak_by_tag`, `num_offloads`,
  `num_cpu_allocs`, and `num_stages`;
- for full-staging v0, `max_stage_bytes_live` is explained by the intentional
  wide stage tags, not by all saved expert activations. For the profile above
  those tags are `act_for_down_base [M,I]` and
  `dgate_up_for_gate_up_base [M,2I]`;
- `stage_peak_by_tag` contains no unexpected wide tags, and
  `saved_hbm_activation_tags` is empty for wide expert activations;
- counters show CPU ownership for `X`, `gate`, `up`, `act`, `S_gate`, `S_up`,
  and `S_down`, and no full HBM save for `gate`, `up`, `act`, `dact`,
  `dgate`, `dup`, or `dgate_up`;
- timing is reviewed from the JSON. If it obviously blows up, record the likely
  cause before moving on, but do not encode a numeric slowdown gate in the
  command.

### Gate Before Row-Window v1

Do not start Stage 6 until Stage 5 has a real correctness, HBM, and timing
artifact. The required handoff between Stage 5 and Stage 6 is:

- summarize Stage 2/3 correctness results;
- summarize Stage 5 `current`, `act_offload`, and `recompute` peak HBM;
- summarize Stage 5 timing, including the slowdown ratio;
- state whether full-staging v0 is already useful enough without row windows;
- state which peak remains dominant and why row windows should address it.

### Stage 6: Row-window v1 only if Stage 5 passes

Purpose: reduce remaining staging peak after the full-staging idea is proven.

Allowed changes:

- add `ASYM_ACT_OFFLOAD_WINDOW_ROWS`;
- add row-window metadata helpers from `offsets` and `experts`;
- replace selected full stages with row-window stages;
- keep the Stage 3 full-staging path available as a debug fallback.

Validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode backward \
  --device cuda:0 \
  --tokens 257 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --route-pattern skewed \
  --seed 701 \
  --compare-to current \
  --output-json reports/act_offload/stage6_window_backward.json
```

Memory run:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode profile \
  --device cuda:0 \
  --tokens 2048 \
  --hidden-dim 1024 \
  --intermediate-dim 4096 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --warmup 5 \
  --iters 10 \
  --variants current,act_offload,recompute \
  --baseline-json reports/act_offload/stage5_v0_profile.json \
  --output-json reports/act_offload/stage6_window_profile.json
```

Pass criteria:

- all Stage 3 correctness tolerances still pass;
- one-row, non-multiple-of-window, empty expert, single active expert, and
  skewed routing cases are covered;
- row-window peak HBM is compared with full-staging peak HBM from
  `reports/act_offload/stage5_v0_profile.json`;
- `max_stage_bytes_live` is compared with the expected row-window scale,
  `dtype_size * window_rows * max(2 * intermediate_dim, hidden_dim, lora_rank)`,
  plus allocator slack;
- `stage_peak_by_tag` shows no full `[M,H]`, `[M,I]`, or `[M,2I]` stage except
  when `ASYM_ACT_OFFLOAD_WINDOW_ROWS` is unset;
- timing is reviewed from the JSON and should not obviously blow up versus the
  Stage 5 full-staging profile;
- no row-window change is allowed to alter the math from Stage 3.

### Stage 7: Forward peak polish

Purpose: reduce the forward-only peak that remains from brief full-HBM
`gate`, `up`, or post-add temporaries. This stage is optional and should happen
only after row-window backward is correct.

Allowed changes:

- in `ActivationOffloadManager`, add narrowly scoped helpers only if needed:
  `offload_add_into(...)` or `offload_columns_into(...)`;
- in `AsymQwen3Experts._forward_expert_body_with_activation_offload`, replace
  `gate = gate_base + LoRA_gate; gate_cpu = offload(gate)` with chunked or
  fused add-and-copy into `gate_cpu`;
- do the same for `up_cpu`;
- keep all base and LoRA GEMMs on GPU/AsymGEMM. Do not move GEMMs to CPU.

Validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode backward \
  --device cuda:0 \
  --tokens 257 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --route-pattern skewed \
  --seed 733 \
  --compare-to current \
  --output-json reports/act_offload/stage7_forward_peak_backward.json
```

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode profile \
  --device cuda:0 \
  --tokens 2048 \
  --hidden-dim 1024 \
  --intermediate-dim 4096 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --warmup 5 \
  --iters 10 \
  --variants current,act_offload,recompute \
  --baseline-json reports/act_offload/stage6_window_profile.json \
  --output-json reports/act_offload/stage7_forward_peak_profile.json
```

Pass criteria:

- all Stage 6 correctness tolerances still pass;
- forward peak HBM and timing are compared with Stage 6. If the artifact does
  not show a useful forward-memory improvement, discard or defer Stage 7 because
  it is optional polish, not required machinery;
- timing should not obviously blow up versus Stage 6;
- no new CPU GEMM or `@^^` dependency is introduced.

### Stage 8: Async overlap after memory is correct

Purpose: recover part of the copy overhead without changing memory ownership or
math. This stage is not needed for correctness.

Allowed changes:

- add optional manager streams for D2H and H2D copies;
- add CUDA events to `CPUHandle` / stage records so stage reuse waits correctly;
- add a small pinned CPU pool warmup path and event-backed release policy;
- add env gates such as `ASYM_ACT_OFFLOAD_ASYNC=1`;
- keep the blocking path as the default debug fallback.

Validation:

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
ASYM_ACT_OFFLOAD_ASYNC=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode backward \
  --device cuda:0 \
  --tokens 257 \
  --hidden-dim 128 \
  --intermediate-dim 128 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --route-pattern skewed \
  --seed 809 \
  --compare-to current \
  --output-json reports/act_offload/stage8_async_backward.json
```

```bash
mkdir -p reports/act_offload
ASYM_QWEN3_EXPERT_ACT_OFFLOAD=1 \
ASYM_ACT_OFFLOAD_WINDOW_ROWS=128 \
ASYM_ACT_OFFLOAD_ASYNC=1 \
python scripts/testing/validate_qwen3_activation_offload.py \
  --mode profile \
  --device cuda:0 \
  --tokens 2048 \
  --hidden-dim 1024 \
  --intermediate-dim 4096 \
  --num-experts 8 \
  --top-k 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --warmup 10 \
  --iters 20 \
  --variants act_offload \
  --baseline-json reports/act_offload/stage6_window_profile.json \
  --output-json reports/act_offload/stage8_async_profile.json
```

Pass criteria:

- async and blocking correctness match the same reference;
- no staging buffer is reused before its H2D copy and dependent GEMMs are safe;
- manager counters distinguish async submitted bytes from waited bytes;
- async timing improves versus the blocking Stage 6 profile or is disabled by
  default with a documented reason.

## Expected HBM Effect

The current path keeps or reconstructs full `[M, I]` tensors in HBM:

```text
gate
up
activated
grad_activated
grad_gate
grad_up
grad_gate_up [M, 2I]
```

It can also reconstruct wide dropout-applied LoRA-A inputs in HBM during
backward, such as gate/up `D(X) [M,H]` and down `D(act) [M,I]`, before LoRA
weight-gradient reductions. The activation-offload path should not do that; it
should apply the saved dropout mask on CPU and use those tensors as the CPU
operand of `@^`.

The v0 activation-offload path should reduce those to one or two full-tensor
HBM staging buffers live at a time:

```text
act stage for down base          [M, I]
dgate_up stage for gate/up base  [M, 2I]
LoRA input-gradient temps        [M, H] or [M, I]
low-rank temps                   [M, r]
```

That can still be a meaningful VRAM drop because the full `[M, I]` tensors are
not all saved in HBM across forward/backward, and the wide dropped LoRA-A
inputs are not rebuilt in HBM just to compute `dA_*`. The full stages are used
only for the next base GPU GEMM and released immediately after use.

The v1 row-window optimization replaces those full staging buffers with:

```text
[mw, I]
[mw, 2I]
[mw, H]
[mw, r]
```

The unavoidable full-HBM tensors remain:

```text
Y       [M, H]
dY      [M, H]
dX      [M, H]
LoRA parameter gradients
```

Forward v0 will still briefly materialize full `gate_up_base`, `gate`, and `up`
before offloading. That is acceptable for the first implementation, but the next
forward peak-memory improvement, after row-window correctness, should be
chunked or fused add-and-offload:

```text
gate_cpu = offload(gate_base + LoRA_gate)
up_cpu   = offload(up_base + LoRA_up)
```

The best later version would avoid a persistent full-HBM `gate` or `up` even
inside forward by writing the post-add result directly into pinned CPU storage.
That requires a fused kernel or at least a careful chunked add/copy path and is
not required for v1 row-window staging.

## Why Not Plain `.to()`

Plain `.to("cpu")` and `.to("cuda")` are acceptable for a correctness
prototype, but not for the intended design:

- `.to()` allocates a new tensor, so allocation becomes part of the hot path;
- it hides ownership, making it easy to accidentally keep both HBM and CPU
  copies alive;
- it does not give a stable place to add future async/event behavior;
- it cannot reuse fixed HBM staging buffers for staged GEMMs.

The implementation should use explicit buffers:

```python
cpu_buf = manager.cpu_pool.alloc(shape, dtype)
cpu_buf.copy_(gpu_tensor, non_blocking=False)

gpu_buf = manager.hbm_pool.alloc(stage_shape, dtype, device)
gpu_buf.copy_(cpu_view, non_blocking=False)
```

This is slower than a fully overlapped path but matches the memory-first goal.

## Validation Rule

The stage-gated plan above is the source of truth for correctness and
implementation order. Do not treat row-window staging, async copies, or
fused add-and-offload as available work until Stage 5 proves the full-staging
activation-offload path is correct and useful.

After Stage 5 passes, Stage 6 row-window staging is the next planned memory
stage. Async copies and fused add-and-offload remain later optimization stages
and must not change the Stage 3/6 math or tolerances.

## Final Position

For AsymGEMM LoRA SFT, the right activation offload design is not a generic
Megatron-style reload mechanism and not a pair of `.to()` helpers. It is a
small manager plus a custom expert autograd path where CPU activations are the
primary saved representation and HBM is used only for explicit staging before
GEMMs. Start with full-tensor staging; add row-window staging only after the
memory win is proven.

That design is slower than recomputation in some cases, but it directly targets
the goal: lower peak HBM without preserving full expert activations on GPU.
