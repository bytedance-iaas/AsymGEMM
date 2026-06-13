# Qwen3 MoE MLP Math

### Notation

```
@  = GEMM
@^L = AsymGEMM with a CPU left operand
@^R = AsymGEMM with a CPU right operand
@^L_grp, @^R_grp = grouped forms over active expert groups

E = number of experts
G = number of active expert groups in the packed MoE layer
offsets[0:G+1], experts[0:G]
M_g = offsets[g+1] - offsets[g]
M = sum_g M_g
e_g = experts[g]

Z_g = Z[offsets[g]:offsets[g+1]] for any packed routed-row tensor Z

H = hidden size
I = intermediate size
r = LoRA rank
scale = lora_alpha / r

X              [M,H]
gate, up       [M,I]
act            [M,I]
Y_down         [M,H]

W_gate_up_cpu  [E,2I,H]  gate rows first, then up rows
W_down_cpu     [E,H,I]

A_gate         [E,r,H]   B_gate     [E,I,r]
A_up           [E,r,H]   B_up       [E,I,r]
A_down         [E,r,I]   B_down     [E,H,r]
```

When an operand is CPU-resident, this note marks the operand side in the GEMM
operator. For example, `U_cpu @^L V.T` has a CPU left operand, while
`U @^R V_cpu` has a CPU right operand.

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

`Z_g.T` denotes GEMM orientation in the equation. It is not a requirement to
save or materialize a transposed tensor.

When `@^L_grp` is used for LoRA-A forward with a CPU activation input, each
active group produces the low-rank output directly in routed-row layout:

```
S_b,g = D_b(U_cpu,g) @^L A_b[e_g].T   # [M_g,r]
S_b   = pack_g(S_b,g)                 # [M,r]
```

There is no intermediate `[r,M_g]` low-rank output. The low-rank output layout
is `[M_g,r]` directly.

Only the small low-rank `[M,r]` tensor is materialized in HBM. The wide dropped
LoRA input `D(U_cpu)` stays CPU-side and can be recomputed from the saved source
activation and saved dropout mask.

`S_gate_cpu`, `S_up_cpu`, and `S_down_cpu` store the same low-rank values as
`S_gate`, `S_up`, and `S_down`. Backward equations write `S_b,g` for the
mathematical low-rank value.

`pack_g(...)` denotes the packed routed-row layout of the grouped result. It is
not a requirement to concatenate per-group outputs in Python.

All LoRA-A forward equations, LoRA-A gradient equations, and LoRA-B gradient
reductions below are grouped over `g = 0..G-1`. The `_g` equations are the
expanded math for the grouped operation/reduction, not separate ungrouped work
items.

### Forward

```
X = packed_routed_rows                                  # [M,H] HBM

gate_up_base_g = X_g @^R W_gate_up_cpu[e_g].T           # [M_g,2I] HBM
gate_up_base = pack_g(gate_up_base_g)                   # [M,2I] HBM
gate_base, up_base = split(gate_up_base)                # [M,I], [M,I]

X_cpu = offload(X)                                      # [M,H] CPU

X_gate_lora_cpu,g = D_gate(X_cpu,g)                     # [M_g,H] CPU
S_gate_g = X_gate_lora_cpu,g @^L A_gate[e_g].T          # [M_g,r] HBM
S_gate = pack_g(S_gate_g)                               # [M,r] HBM

X_up_lora_cpu,g = D_up(X_cpu,g)                         # [M_g,H] CPU
S_up_g = X_up_lora_cpu,g @^L A_up[e_g].T                # [M_g,r] HBM
S_up = pack_g(S_up_g)                                   # [M,r] HBM

LoRA_gate_g = scale * (S_gate_g @ B_gate[e_g].T)        # [M_g,I] HBM
LoRA_gate = pack_g(LoRA_gate_g)                         # [M,I] HBM
gate = gate_base + LoRA_gate                            # [M,I] HBM
LoRA_up_g = scale * (S_up_g @ B_up[e_g].T)              # [M_g,I] HBM
LoRA_up = pack_g(LoRA_up_g)                             # [M,I] HBM
up = up_base + LoRA_up                                  # [M,I] HBM

gate_cpu = offload(gate)                                # [M,I] CPU
up_cpu = offload(up)                                    # [M,I] CPU
S_gate_cpu = offload(S_gate)                            # [M,r] CPU, save for dB_gate
S_up_cpu = offload(S_up)                                # [M,r] CPU, save for dB_up

sig_cpu = sigmoid(gate_cpu)                             # [M,I] CPU
silu_gate_cpu = gate_cpu * sig_cpu                      # [M,I] CPU
act_cpu = silu_gate_cpu * up_cpu                        # [M,I] CPU

act_down_lora_cpu,g = D_down(act_cpu,g)                 # [M_g,I] CPU
S_down_g = act_down_lora_cpu,g @^L A_down[e_g].T        # [M_g,r] HBM
S_down = pack_g(S_down_g)                               # [M,r] HBM
LoRA_down_g = scale * (S_down_g @ B_down[e_g].T)        # [M_g,H] HBM
LoRA_down = pack_g(LoRA_down_g)                         # [M,H] HBM
S_down_cpu = offload(S_down)                            # [M,r] CPU, save for dB_down

act = stage(act_cpu)                                    # [M,I] HBM, needed for down base
Y_down_base_g = act_g @^R W_down_cpu[e_g].T             # [M_g,H] HBM
Y_down = pack_g(Y_down_base_g) + LoRA_down              # [M,H] HBM
```

### Backward

```
dY = dL/dY_down                                         # [M,H] HBM

# ---------------- down backward ----------------

dS_down_g = scale * (dY_g @ B_down[e_g])                # [M_g,r] HBM
dS_down = pack_g(dS_down_g)                             # [M,r] HBM

dact_lora_raw_g = dS_down_g @ A_down[e_g]               # [M_g,I] HBM
dact_lora_raw = pack_g(dact_lora_raw_g)                 # [M,I] HBM
dact_lora = D_down_bar(dact_lora_raw)                   # [M,I] HBM
dact_base_g = dY_g @^R W_down_cpu[e_g]                  # [M_g,I] HBM
dact_base = pack_g(dact_base_g)                         # [M,I] HBM
dact = dact_base + dact_lora                            # [M,I] HBM

dact_cpu = offload(dact)                                # [M,I] CPU

act_down_lora_cpu,g = D_down(act_cpu,g)                 # [M_g,I] CPU
dA_down[e] = sum_{g:e_g=e} dS_down_g.T @^R act_down_lora_cpu,g  # [r,I] Grad
dB_down[e] = scale * sum_{g:e_g=e} dY_g.T @ S_down_g    # [H,r] Grad


# ---------------- activation backward ----------------

silu_grad_cpu = sig_cpu * (1 + gate_cpu * (1 - sig_cpu))  # [M,I] CPU

dgate_cpu = dact_cpu * up_cpu * silu_grad_cpu           # [M,I] CPU
dup_cpu = dact_cpu * silu_gate_cpu                      # [M,I] CPU


# ---------------- gate LoRA backward ----------------

dS_gate_g = scale * (dgate_cpu,g @^L B_gate[e_g])       # [M_g,r] HBM
dS_gate = pack_g(dS_gate_g)                             # [M,r] HBM
dX_gate_raw_g = dS_gate_g @ A_gate[e_g]                 # [M_g,H] HBM
dX_gate_raw = pack_g(dX_gate_raw_g)                     # [M,H] HBM
dX_gate_lora = D_gate_bar(dX_gate_raw)                  # [M,H] HBM

X_gate_lora_cpu,g = D_gate(X_cpu,g)                     # [M_g,H] CPU
dA_gate[e] = sum_{g:e_g=e} dS_gate_g.T @^R X_gate_lora_cpu,g  # [r,H] Grad
dB_gate[e] = scale * sum_{g:e_g=e} dgate_cpu,g.T @^L S_gate_g  # [I,r] Grad


# ---------------- up LoRA backward ----------------

dS_up_g = scale * (dup_cpu,g @^L B_up[e_g])             # [M_g,r] HBM
dS_up = pack_g(dS_up_g)                                 # [M,r] HBM
dX_up_raw_g = dS_up_g @ A_up[e_g]                       # [M_g,H] HBM
dX_up_raw = pack_g(dX_up_raw_g)                         # [M,H] HBM
dX_up_lora = D_up_bar(dX_up_raw)                        # [M,H] HBM

X_up_lora_cpu,g = D_up(X_cpu,g)                         # [M_g,H] CPU
dA_up[e] = sum_{g:e_g=e} dS_up_g.T @^R X_up_lora_cpu,g  # [r,H] Grad
dB_up[e] = scale * sum_{g:e_g=e} dup_cpu,g.T @^L S_up_g  # [I,r] Grad


# ---------------- gate/up base backward ----------------

dgate_up = stage_concat(dgate_cpu, dup_cpu)              # [M,2I] HBM
dX_base_g = dgate_up_g @^R W_gate_up_cpu[e_g]           # [M_g,H] HBM
dX_base = pack_g(dX_base_g)                              # [M,H] HBM
dX = dX_base + dX_gate_lora + dX_up_lora                 # [M,H] HBM


# ---------------- final input gradient ----------------
# dX = gate/up base gradient + gate LoRA gradient + up LoRA gradient
```
