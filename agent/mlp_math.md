# Qwen3 MoE MLP Math

### Notation

```
@  = GEMM
@^ = AsymGEMM with a CPU tensor operand

M = routed rows for this expert
H = hidden size
I = intermediate size
r = LoRA rank
scale = lora_alpha / r

X              [M,H]
gate, up       [M,I]
act            [M,I]
Y_down         [M,H]

W_gate_up_cpu  [2I,H]    gate rows first, then up rows
W_down_cpu     [H,I]

A_gate         [r,H]     B_gate     [I,r]
A_up           [r,H]     B_up       [I,r]
A_down         [r,I]     B_down     [H,r]
```

When a LoRA-A input is CPU-resident, this note writes the same math with `@^`
below.

For branch `b in {gate, up, down}`, with saved dropout mask `mask_b`,
`p = lora_dropout_p`, and `q = 1 - p`:

```
D_b(Z)      = Z                         if p == 0
D_b(Z)      = mask_b * Z / q            if 0 < p < 1
D_b_bar(G)  = G                         if p == 0
D_b_bar(G)  = mask_b * G / q            if 0 < p < 1
```

`D_b_bar` is the gradient of the saved inverted-dropout op wrt its input.
`D_b(cpu_tensor)` and `D_b_bar(cpu_tensor)` are CPU mask/scale ops, not GEMMs
and not HBM staging.

When `@^` is used for LoRA-A with a CPU activation input:

```
S_T = A @^ D(U_cpu).T                 # [r,M]
S   = row_major(S_T.T)                # [M,r]
```

`row_major(...)` means materialize the shown low-rank HBM tensor in `[M,r]`
layout.

Only the small low-rank `[M,r]` tensor is materialized in HBM. The wide dropped
LoRA input `D(U_cpu)` stays CPU-side and can be recomputed from the saved source
activation and saved dropout mask.

### Forward

```
X = routed_rows_for_this_expert                         # [M,H] HBM

gate_up_base = X @^ W_gate_up_cpu.T                     # [M,2I] HBM
gate_base, up_base = split(gate_up_base)                # [M,I], [M,I]

X_cpu = offload(X)                                      # [M,H] CPU

X_gate_lora_cpu = D_gate(X_cpu)                         # [M,H] CPU
S_gate_T = A_gate @^ X_gate_lora_cpu.T                  # [r,M] HBM
S_gate = row_major(S_gate_T.T)                          # [M,r] HBM

X_up_lora_cpu = D_up(X_cpu)                             # [M,H] CPU
S_up_T = A_up @^ X_up_lora_cpu.T                        # [r,M] HBM
S_up = row_major(S_up_T.T)                              # [M,r] HBM

LoRA_gate = scale * (S_gate @ B_gate.T)                 # [M,I] HBM
gate = gate_base + LoRA_gate                            # [M,I] HBM
LoRA_up = scale * (S_up @ B_up.T)                       # [M,I] HBM
up = up_base + LoRA_up                                  # [M,I] HBM

gate_cpu = offload(gate)                                # [M,I] CPU
up_cpu = offload(up)                                    # [M,I] CPU
S_gate_cpu = offload(S_gate)                            # [M,r] CPU, save for dB_gate
S_up_cpu = offload(S_up)                                # [M,r] CPU, save for dB_up

sig_cpu = sigmoid(gate_cpu)                             # [M,I] CPU
silu_gate_cpu = gate_cpu * sig_cpu                      # [M,I] CPU
act_cpu = silu_gate_cpu * up_cpu                        # [M,I] CPU

act_down_lora_cpu = D_down(act_cpu)                     # [M,I] CPU
S_down_T = A_down @^ act_down_lora_cpu.T                # [r,M] HBM
S_down = row_major(S_down_T.T)                          # [M,r] HBM
LoRA_down = scale * (S_down @ B_down.T)                 # [M,H] HBM
S_down_cpu = offload(S_down)                            # [M,r] CPU, save for dB_down

act = stage(act_cpu)                                    # [M,I] HBM, needed for down base
Y_down = act @^ W_down_cpu.T + LoRA_down                # [M,H] HBM
```

### Backward

```
dY = dL/dY_down                                         # [M,H] HBM

# ---------------- down backward ----------------

dS_down = scale * (dY @ B_down)                         # [M,r] HBM

dact_lora_raw = dS_down @ A_down                        # [M,I] HBM
dact_lora = D_down_bar(dact_lora_raw)                   # [M,I] HBM
dact_base = dY @^ W_down_cpu                            # [M,I] HBM
dact = dact_base + dact_lora                            # [M,I] HBM

dact_cpu = offload(dact)                                # [M,I] CPU

act_down_lora_cpu = D_down(act_cpu)                     # [M,I] CPU
dA_down = dS_down.T @^ act_down_lora_cpu                # [r,I] Grad
dB_down = scale * (dY.T @^ S_down_cpu)                  # [H,r] Grad


# ---------------- activation backward ----------------

silu_grad_cpu = sig_cpu * (1 + gate_cpu * (1 - sig_cpu))  # [M,I] CPU

dgate_cpu = dact_cpu * up_cpu * silu_grad_cpu           # [M,I] CPU
dup_cpu = dact_cpu * silu_gate_cpu                      # [M,I] CPU


# ---------------- gate/up base backward ----------------

dgate = stage(dgate_cpu)                                # [M,I] HBM
dup = stage(dup_cpu)                                    # [M,I] HBM

dgate_up = concat(dgate, dup)                           # [M,2I] HBM
dX = dgate_up @^ W_gate_up_cpu                          # [M,H] HBM


# ---------------- gate LoRA backward ----------------

dS_gate = scale * (dgate @ B_gate)                      # [M,r] HBM
dX_gate_raw = dS_gate @ A_gate                          # [M,H] HBM
dX_gate_lora = D_gate_bar(dX_gate_raw)                  # [M,H] HBM
dX += dX_gate_lora

X_gate_lora_cpu = D_gate(X_cpu)                         # [M,H] CPU
dA_gate = dS_gate.T @^ X_gate_lora_cpu                  # [r,H] Grad
dB_gate = scale * (dgate.T @^ S_gate_cpu)               # [I,r] Grad


# ---------------- up LoRA backward ----------------

dS_up = scale * (dup @ B_up)                            # [M,r] HBM
dX_up_raw = dS_up @ A_up                                # [M,H] HBM
dX_up_lora = D_up_bar(dX_up_raw)                        # [M,H] HBM
dX += dX_up_lora

X_up_lora_cpu = D_up(X_cpu)                             # [M,H] CPU
dA_up = dS_up.T @^ X_up_lora_cpu                        # [r,H] Grad
dB_up = scale * (dup.T @^ S_up_cpu)                     # [I,r] Grad


# ---------------- final input gradient ----------------
# dX = gate/up base gradient + gate LoRA gradient + up LoRA gradient
```
