# Qwen3 Expert Recompute Math

```text
# Tags:
#   act_save   = saved forward activation
#   act_drop   = forward activation dropped for selected recompute rows
#   act_recomp = activation rebuilt in backward
#   temp       = short-lived tensor, not saved
#   lora_save  = saved low-rank LoRA state
#   mask_save  = saved bitpacked dropout mask if p > 0

# Shapes.
# X, Y, dX, dY: [M, H]
# gate/up/activated: [M, I]
# LoRA state: [M, r]
# W_gu: [E, 2I, H], W_down: [E, H, I]
# W_gate = W_gu[:, 0:I, :], W_up = W_gu[:, I:2I, :]

# Forward gate/up.
X                                      # [M, H] act_save
G_base = X W_gate^T                    # [M, I] act_drop
U_base = X W_up^T                      # [M, I] act_drop

mask_gate = pack(dropout_mask(X))       # [M, H]/8 mask_save
mask_up   = pack(dropout_mask(X))       # [M, H]/8 mask_save
Xdrop_gate = dropout(X, mask_gate)      # [M, H] temp
Xdrop_up   = dropout(X, mask_up)        # [M, H] temp
S_gate = Xdrop_gate A_gate^T            # [M, r] lora_save
S_up   = Xdrop_up   A_up^T              # [M, r] lora_save
G_lora = alpha * S_gate B_gate^T        # [M, I] act_drop
U_lora = alpha * S_up   B_up^T          # [M, I] act_drop

G = G_base + G_lora                     # [M, I] act_drop
U = U_base + U_lora                     # [M, I] act_drop
A = silu(G) * U                         # [M, I] act_drop

# Forward down.
Y_base = A W_down^T                     # [M, H] out
mask_down = pack(dropout_mask(A))       # [M, I]/8 mask_save
Adrop = dropout(A, mask_down)           # [M, I] act_drop
S_down = Adrop A_down^T                 # [M, r] lora_save
Y_lora = alpha * S_down B_down^T        # [M, H] out
Y = Y_base + Y_lora                     # [M, H] out

------------------------------------------------------------
# Backward input.
dY                                      # [M, H] grad

# Down base backward.
dA_base = dY W_down                     # [M, I] grad
dW_down = dY^T A                        # [H, I] skip_frozen

# Recompute selected gate/up.
G_base = X W_gate^T                     # [M, I] act_recomp
U_base = X W_up^T                       # [M, I] act_recomp
G_lora = alpha * S_gate B_gate^T        # [M, I] act_recomp
U_lora = alpha * S_up   B_up^T          # [M, I] act_recomp
G = G_base + G_lora                     # [M, I] act_recomp
U = U_base + U_lora                     # [M, I] act_recomp
A = silu(G) * U                         # [M, I] act_recomp

# Down LoRA backward.
Adrop = dropout_replay(A, mask_down)    # [M, I] act_recomp
dB_down = alpha * dY^T S_down           # [H, r] grad
dS_down = alpha * dY B_down             # [M, r] grad
dA_down = dS_down^T Adrop               # [r, I] grad
dA_lora = dropout_backward(dS_down A_down, mask_down) # [M, I] grad

# Activation backward.
dA = dA_base + dA_lora                  # [M, I] grad
dU = dA * silu(G)                       # [M, I] grad
dG = dA * U * silu_grad(G)              # [M, I] grad

# Gate/up base dX.
dX_gate_base = dG W_gate                # [M, H] grad
dX_up_base   = dU W_up                  # [M, H] grad
dX_base      = dX_gate_base + dX_up_base # [M, H] grad
dW_gate = dG^T X                        # [I, H] skip_frozen
dW_up   = dU^T X                        # [I, H] skip_frozen

# Gate LoRA backward.
Xdrop_gate = dropout_replay(X, mask_gate) # [M, H] temp
dB_gate = alpha * dG^T S_gate            # [I, r] grad
dS_gate = alpha * dG B_gate              # [M, r] grad
dA_gate = dS_gate^T Xdrop_gate           # [r, H] grad
dX_gate_lora = dropout_backward(dS_gate A_gate, mask_gate) # [M, H] grad

# Up LoRA backward.
Xdrop_up = dropout_replay(X, mask_up)    # [M, H] temp
dB_up = alpha * dU^T S_up                # [I, r] grad
dS_up = alpha * dU B_up                  # [M, r] grad
dA_up = dS_up^T Xdrop_up                 # [r, H] grad
dX_up_lora = dropout_backward(dS_up A_up, mask_up) # [M, H] grad

# Final input gradient.
dX = dX_base + dX_gate_lora + dX_up_lora # [M, H] grad

# Recompute cost.
# G/U act_recomp: heavy, one extra AsymGEMM stream over W_gu [E, 2I, H].
# A/Adrop act_recomp: cheap elementwise plus saved mask replay.
# Current selected gate/up stream: 2.0x W_gu = recompute once + dX once.
```
