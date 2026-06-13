# V2: Save Gate, Recompute Only Up

Status: recommended next design. This keeps AsymGEMM as the weight-streaming
mechanism, avoids HBM W staging, avoids full-H W panels in SMEM, and reduces
selected gate/up W streaming from `2.0x` to `1.5x`.

## Core Change

Current selected recompute groups save no gate/up/activated tensor from forward:

```text
saved forward state for selected recompute rows:
  no gate
  no up
  no activated
```

Backward currently recomputes gate/up and then rebuilds activated. V2 selected
recompute saves one tensor instead:

```text
gate
```

This adds one saved `[Ms, I]` BF16 tensor relative to the current selected
recompute policy. In backward, V2 recomputes only `up`, rebuilds `activated`,
then computes the activation gradient.

Do not reconstruct:

```text
up = activated / silu(gate)
```

That is numerically unsafe when `silu(gate)` is near zero.

## Math

Forward:

```text
gate = X W_gate^T + LoRA_gate
up   = X W_up^T   + LoRA_up
act  = silu(gate) * up
```

Backward after recomputing `up`:

```text
activated = silu(saved_gate) * up

dact = dY W_down + d down_lora / d activated

dgate = dact * up * silu_grad(saved_gate)
dup   = dact * silu(saved_gate)

dX_base = [dgate, dup] @ W_gate_up
```

## Stream Count

Current selected recompute:

```text
recompute gate+up: 1.0x W_gate_up
base dX:           1.0x W_gate_up
total:             2.0x W_gate_up
```

V2:

```text
recompute up only: 0.5x W_gate_up
base dX:           1.0x W_gate_up
total:             1.5x W_gate_up
```

This is an exact scheduling/policy change, not a math approximation.

## Required Runtime Change: Sub-N AsymGEMM

Packed base weight:

```text
W_gate_up_cpu: [E, 2I, H] CPU-pinned BF16
gate rows:     [0, I)
up rows:       [I, 2I)
```

Need an AsymGEMM runtime/API that streams only the up half directly from the
packed CPU-pinned tensor:

```text
up_base = AsymGEMM_NT_subN(
    X_sel,
    W_gate_up_cpu,
    b_n_offset = I,
    logical_n = I,
    physical_n = 2I,
)
```

Do not implement this as `W_gate_up_cpu[:, I:, :]` plus the existing runtime.
The current TMA flattening assumes compact per-expert `N`; a sliced view would
step from expert 0 up rows into expert 1 gate rows unless the kernel knows the
physical `2I` expert stride.

Runtime requirements:

```text
logical_n controls output shape/grid:        d is [Ms, I]
physical_n controls B group stride:          2I
b_n_offset controls row offset inside expert: I

B tensor map global N span uses physical_n * E.
B tile address uses:
  b_n_idx = expert_id * physical_n + b_n_offset + blockIdx.x * BLOCK_N

Output D address still uses logical_n:
  d_n_idx = blockIdx.x * BLOCK_N
```

First implementation only needs NT sub-N for recomputing `up`. Existing full
transpose-B AsymGEMM can still compute selected `dX_base`.

Suggested binding:

```text
m_grouped_bf16_asym_gemm_nt_contiguous_subn(
    a, b, d,
    offsets, experts, list_size,
    logical_n,
    b_n_offset,
    b_physical_n,
    compiled_dims
)
```

SM100 BF16 only.

## Small CUDA Kernels

### Kernel A: Rebuild Activated

After recomputing `up`, rebuild selected activated rows:

```text
activated_sel = silu(saved_gate_sel) * up_sel
```

Inputs:

```text
saved_gate_sel [Ms, I] bf16
up_sel         [Ms, I] bf16
```

Output:

```text
activated_sel  [Ms, I] bf16
```

Python scatters `activated_sel` into the full/compact activation buffer expected
by existing down LoRA backward, or a later cleanup can make down LoRA backward
consume compact selected activation directly.

### Kernel B: Activation Grad + Pack

After down base and down LoRA backward produce final `dact_sel`:

```text
dgate = dact_sel * up_sel * silu_grad(saved_gate_sel)
dup   = dact_sel * silu(saved_gate_sel)
```

Inputs:

```text
saved_gate_sel [Ms, I] bf16
up_sel         [Ms, I] bf16
dact_sel       [Ms, I] bf16
```

Outputs:

```text
grad_gate_sel    [Ms, I] bf16
grad_up_sel      [Ms, I] bf16
grad_gate_up_sel [Ms, 2I] bf16 = [grad_gate_sel, grad_up_sel]
```

Use FP32 for `silu` and `silu_grad`, then cast outputs to BF16.

## Backward Flow

For selected recompute rows:

```text
1. pack/select X_sel and saved_gate_sel

2. recompute up base only:
   up_base_sel = AsymGEMM_NT_subN(
       X_sel,
       W_gate_up_cpu,
       b_n_offset=I,
       logical_n=I,
       physical_n=2I,
   )

3. add up LoRA:
   up_sel = up_base_sel + LoRA_up(S_up, B_up)

4. rebuild activated:
   activated_sel = silu(saved_gate_sel) * up_sel

5. run existing down backward using rebuilt activated:
   dact = dY W_down + d down_lora / d activated

6. compute activation grad and pack:
   grad_gate_sel, grad_up_sel, grad_gate_up_sel =
       activation_grad_pack(saved_gate_sel, up_sel, dact_sel)

7. selected base dX:
   grad_x_base_sel = AsymGEMM_T(grad_gate_up_sel, W_gate_up_cpu)

8. run existing gate/up LoRA backward using grad_gate_sel/grad_up_sel

9. scatter/add selected grad_x pieces back to full packed grad
```

## LoRA Dropout

Native sub-N AsymGEMM does not own dropout. Follow the dropout fix design:

```text
lora_dropout == 0:
  S_up may be recomputed as dropout-free X A_up^T

lora_dropout > 0:
  S_up must come from saved forward low-rank state
  backward must not rerun dropout
```

Gate LoRA backward still needs the saved/recomputed gate low-rank state as
defined by the existing dropout policy.

## Expected Outcome

V2 does not remove all selected gate/up base W streaming. It removes only the
unneeded half of the recompute stream:

```text
2.0x -> 1.5x selected W_gate_up CPU stream
```

Benefits:

```text
adds one selected forward activation tensor: saved gate [Ms, I]
no HBM W cache
no full-H W panel in SMEM
uses AsymGEMM for all frozen base weight streaming
small custom kernels are elementwise/packing only
```

Validation targets:

```text
correctness vs current recompute for dropout 0.00 and 0.10
profile confirms recompute reads only up half of W_gate_up
profile confirms selected dX still reads full W_gate_up once
peak HBM should stay close to current selected recompute policy
step latency should improve when selected recompute W streaming is material
```
