# Improved LoRA Dropout/S Kernels

```text
# Goal:
#   compute LoRA low-rank state S = dropout(X) A^T efficiently
#   save only S and bitpacked dropout masks
#   do not save or persist full Xdrop / Adrop tensors
```

## Core Idea

This is the same mathematical split as LoRAFusion's `S` kernel:

```text
S = dropout(X) A^T          # [M, r]
Y = X W^T + alpha S B^T
```

But the memory policy should be different from LoRAFusion.

LoRAFusion single-LoRA forward uses a fused dropout-matmul kernel:

```text
S = dropout(X) A^T
```

and saves:

```text
S                 # [M, r]
dropout_mask      # [M, K] bool
masked_scaled_x   # [M, K]
```

For AsymGEMM LoRA SFT, the target policy is:

```text
S                 # [M, r]   save
dropout_mask      # [M, K]/8 save bitpacked
masked_scaled_x   # [M, K]   do not save
```

So this is **LoRAFusion's S idea**, but with **bitpacked-mask replay** and without
persistent full dropped inputs.

## Current Python Path

For `lora_dropout > 0`, current Qwen3 Python runtime is correctness-first:

```text
gate_input = dropout(X)       # [M, H] materialized
S_gate     = gate_input A_gate^T

up_input   = dropout(X)       # [M, H] materialized
S_up       = up_input A_up^T

Adrop      = dropout(A)       # [M, I] materialized
S_down     = Adrop A_down^T
```

This is correct, but it creates transient full-size dropped tensors. The final
efficient path should fuse dropout application into the LoRA-A matmul tile load.

## Target Forward Kernels

```text
mask_gate = gen_pack_dropout_mask(X, p, seed_gate)       # [M, H]/8
S_gate    = masked_grouped_lora_A(X, mask_gate, A_gate)  # [M, r]

mask_up   = gen_pack_dropout_mask(X, p, seed_up)         # [M, H]/8
S_up      = masked_grouped_lora_A(X, mask_up, A_up)      # [M, r]

mask_down = gen_pack_dropout_mask(A, p, seed_down)       # [M, I]/8
S_down    = masked_grouped_lora_A(A, mask_down, A_down)  # [M, r]
```

More fused version:

```text
S_gate, mask_gate = grouped_dropout_lora_A(X, A_gate, p, seed_gate)
S_up,   mask_up   = grouped_dropout_lora_A(X, A_up,   p, seed_up)
S_down, mask_down = grouped_dropout_lora_A(A, A_down, p, seed_down)
```

Tile-level operation:

```text
for each expert group and output tile [BM, r_tile]:
  acc = 0
  for K tile:
    load X_tile                     # [BM, BK]
    generate/load mask bits          # [BM, BK]
    X_tile = where(mask, X_tile/(1-p), 0)
    load A_tile                      # [r_tile, BK]
    acc += X_tile @ A_tile^T
  store S_tile                       # [BM, r_tile]
  store packed mask bits             # [BM, K]/8, if generated here
```

No full `[M, H]` or `[M, I]` dropped tensor is written to HBM.

## Target Backward

For a LoRA projection:

```text
Y_lora = alpha S B^T
S      = dropout(X) A^T
```

Backward:

```text
dB = alpha * dY^T S                  # [N, r]
dS = alpha * dY B                    # [M, r]
dA = dS^T dropout_replay(X, mask)    # [r, K]
dX = dropout_backward(dS A, mask)    # [M, K]
```

Efficient masked kernels should avoid materializing `dropout_replay(X, mask)`:

```text
dA = masked_lora_A_grad(dS, X, mask) # apply mask while reading X tiles
dX = masked_lora_dX(dS, A, mask)     # apply mask while writing dX tiles
```

Tile-level `dA`:

```text
for each expert group and [r_tile, K_tile]:
  acc = 0
  for M tile:
    load dS_tile                     # [BM, r_tile]
    load X_tile                      # [BM, BK]
    load/unpack mask bits             # [BM, BK]
    X_tile = where(mask, X_tile/(1-p), 0)
    acc += dS_tile^T @ X_tile
  store dA_tile                      # [r_tile, BK]
```

Tile-level `dX`:

```text
for each expert group and [M_tile, K_tile]:
  acc = 0
  for r tile:
    load dS_tile                     # [BM, r_tile]
    load A_tile                      # [r_tile, BK]
    acc += dS_tile @ A_tile
  acc = where(mask, acc/(1-p), 0)
  store dX_lora_tile                 # [BM, BK]
```

## What This Saves

Compared with the current Python path:

```text
avoid writing Xdrop_gate [M, H]
avoid reading  Xdrop_gate [M, H]
avoid writing Xdrop_up   [M, H]
avoid reading  Xdrop_up   [M, H]
avoid writing Adrop      [M, I]
avoid reading  Adrop      [M, I]
```

Persistent saved state remains:

```text
S_gate, S_up, S_down                 # each [M, r]
mask_gate, mask_up                   # each [M, H]/8
mask_down                            # [M, I]/8
```

This is mainly a LoRA/dropout efficiency improvement. It does not by itself solve
the AsymGEMM gate/up base-weight recompute streaming problem, but it removes a
known inefficient transient path when `lora_dropout > 0`.

## Implementation Scope

Target scope:

```text
SM100 BF16
Qwen3 expert LoRA path first
grouped experts with existing offsets/experts metadata
dropout_p in [0, 1)
separate masks for gate, up, and down
```

Required validation:

```text
1. forward parity vs current Python path with fixed seeds
2. backward grad parity for A/B and dX
3. memory check: no persistent Xdrop/Adrop saved tensors
4. profile check: reduced dropout+LoRA-A HBM traffic for p > 0
```
