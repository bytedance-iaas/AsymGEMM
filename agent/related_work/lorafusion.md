# LoRAFusion Kernel Fusion Notes

LoRAFusion does execution fusion for LoRA fine-tuning. It does not merge LoRA
weights into the frozen base weight.

Shapes:

```text
X:  [M, K]   input activations
W:  [N, K]   frozen base weight
A:  [r, K]   trainable LoRA A
B:  [N, r]   trainable LoRA B
S:  [M, r]   low-rank intermediate
Y:  [M, N]   output
dY: [M, N]   upstream gradient
```

`r << K, N`, so LoRAFusion materializes small rank-`r` tensors and avoids extra
large `[M, N]` or `[M, K]` tensors.

## Forward Fusion

### Background

```text
Y = X W^T + alpha * dropout(X) A^T B^T

Xdrop = dropout(X)
S     = Xdrop A^T          # [M, r], small
Y     = X W^T + alpha S B^T
```

`S` is the graph split point:

```text
compute S once
reuse S in the large output fusion
```

### Method

Forward has two stages:

```text
Stage 1:
  S = dropout(X) A^T

Stage 2:
  Y = X W^T + alpha S B^T + bias
```

Optional Stage 1 fusion:

```text
fused_dropout_matmul:
  S = dropout(X) A^T
```

Why separate from the main output kernel:

```text
main output kernel tiles Y over N
S is only [M, r]
computing S inside each Y tile would recompute S many times
```

Main Stage 2 fusion:

```text
fused_lora_xw_sb:
  Y = X W^T + alpha S B^T + bias
```

Unfused output path:

```text
base = X W^T              # large [M, N]
lora = alpha S B^T        # large [M, N]
Y    = base + lora        # large read/write
```

Fused output path:

```text
one output tile accumulator:
  accum += alpha S_tile B_tile^T
  accum += X_tile W_tile^T
  accum += bias
  store Y once
```

Main saving:

```text
avoid materializing base [M, N]
avoid materializing lora [M, N]
avoid separate add kernel
```

## Backward Fusion

### Background

From:

```text
Y = X W^T + alpha S B^T
S = Xdrop A^T
```

Backward:

```text
dB = alpha * dY^T S       # [N, r]
dS = alpha * dY B         # [M, r]
dA = dS^T Xdrop           # [r, K]

dX_base = dY W            # [M, K]
dX_lora = dropout_backward(dS A)
dX      = dX_base + dX_lora

dW is skipped because W is frozen
```

`dX_base` is still required even though `W` is frozen, because gradients must
flow to earlier layers.

### Method

Fused `dB/dS`:

```text
fused_lora_dys_dyb:
  dB = alpha * dY^T S
  dS = alpha * dY B
```

Why:

```text
both consume dY
load dY tile once
produce LoRA-sized outputs [N, r] and [M, r]
```

Fused `dX`:

```text
fused_lora_dyw_dsa:
  dX = dY W + dropout_backward(dS A)
```

Unfused `dX` path:

```text
dX_base = dY W                   # large [M, K]
dX_lora = dropout_backward(dS A) # large [M, K]
dX      = dX_base + dX_lora      # large read/write
```

Main saving:

```text
avoid materializing dX_base [M, K]
avoid materializing dX_lora [M, K]
store final dX once
```

Separate `dA`:

```text
dA = dS^T Xdrop
```

Why separate:

```text
dA needs completed dS
dS is produced by reduction over N
one Triton kernel has no global barrier across all program blocks
```

## Why Not Fuse Everything?

Candidate giant backward:

```text
dB = alpha * dY^T S
dS = alpha * dY B
dA = dS^T Xdrop
dX = dY W + dropout_backward(dS A)
```

Output/reduction mismatch:

```text
dB: [N, r]   reduce over M
dS: [M, r]   reduce over N
dA: [r, K]   reduce over M
dX: [M, K]   reduce over N and r
```

Problems:

```text
different tiling choices
too many accumulators / high register pressure
hard dependency on completed dS
would require storing dS, recomputing dS, or atomics
```

Design rule:

```text
materialize small:
  S, dS              # [M, r]

avoid materializing large:
  base, lora         # [M, N]
  dX_base, dX_lora   # [M, K]
```

## Kernel Summary

```text
Forward:
  optional fused_dropout_matmul:
    S = dropout(X) A^T

  fused_lora_xw_sb:
    Y = X W^T + alpha S B^T + bias

Backward:
  fused_lora_dys_dyb:
    dB = alpha * dY^T S
    dS = alpha * dY B

  normal GEMM:
    dA = dS^T Xdrop

  fused_lora_dyw_dsa:
    dX = dY W + dropout_backward(dS A)

Skipped:
  dW, because W is frozen
```

## Related Kernel Variants

Same math, different scheduling target:

```text
fused_lora_xw_sb_tma:
  TMA/H100-SM90 version of fused_lora_xw_sb.

fused_lora_dyw_dsa_tma:
  TMA/H100-SM90 version of fused_lora_dyw_dsa.

fused_multi_lora_xw_sb:
  multi-adapter version of fused_lora_xw_sb.

fused_multi_lora_dys_dyb:
  multi-adapter version of fused_lora_dys_dyb.

fused_multi_lora_dyw_dsa:
  multi-adapter version of fused_lora_dyw_dsa.
```

Multi-LoRA adds adapter lookup, not new math:

```text
Y_block = X_block W^T + alpha_adapter S_adapter B_adapter^T
```

## System-Level Contribution

Kernel fusion makes one LoRA layer cheaper. LoRAFusion also trains many
independent adapters against one shared frozen base model on the same GPU pool.

System goal:

```text
more concurrent adapter work
fewer pipeline bubbles
better communication overlap
better GPU load balance
```

Scheduler/solver role:

```text
1. group adapters to stagger jobs
2. pack variable-length samples into balanced microbatches
3. respect forward/backward pipeline dependencies
4. trigger per-adapter gradient sync and optimizer steps
```

Claim split:

```text
C1 end-to-end speedup:
  multi-LoRA batching + scheduling + pipeline behavior + fused kernels

C2 fused-kernel speedup:
  per-layer fusion above
```

## Takeaway

LoRAFusion combines selective kernel fusion with multi-adapter scheduling:

```text
kernels reduce large activation memory traffic
scheduler keeps GPUs busy across concurrent LoRA jobs
```
