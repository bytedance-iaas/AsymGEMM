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

# Logical trainable LoRA weights. With weight offload enabled these are CPU-home
# and are staged to HBM at the `stage(...)` points below.
A_gate         [E,r,H]   B_gate     [E,I,r]
A_up           [E,r,H]   B_up       [E,I,r]
A_down         [E,r,I]   B_down     [E,H,r]
A_gate_up_hbm  [E,2r,H]  temporary cat_r(A_gate_hbm, A_up_hbm) for HBM LoRA-A mode
```

When an operand is CPU-resident, this note marks the operand side in the GEMM
operator. For example, `U_cpu @^L V.T` has a CPU left operand, while
`U @^R V_cpu` has a CPU right operand.

```
offload(Z)   = copy HBM tensor to CPU and save the CPU owner
stage(U)     = make CPU tensor or CPU-home logical LoRA weight U available as an HBM tensor
release(...) = listed tensors, staged weights, or saved handles are no longer live
```

For a saved CPU tensor, `stage(U_cpu)` copies it to HBM for immediate use. For a
LoRA weight `P`, `stage(P)` is a no-op view when LoRA weight offload is
disabled. When LoRA weight offload is enabled, it stages/backfetches the
CPU-home trainable LoRA weight to HBM for the immediate GEMM and the staged
`*_hbm` tensor must be released after its last use. The logical trainable
parameter and CPU optimizer state remain owned by the optimizer/weight-offload
coordinator.

`Y += GEMM(...)` means accumulate into an already-live output buffer. Use a
beta/addmm-style epilogue when the backend supports it. If a backend must
materialize the GEMM result, that result is a one-line temporary consumed by the
add and released immediately.

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
A_b_hbm = stage(A_b)              # [E,r,*] HBM staged
S_b,g   = D_b(U_cpu,g) @^L A_b_hbm[e_g].T # [M_g,r]
S_b     = pack_g(S_b,g)                  # [M,r]
release(A_b_hbm)
```

There is no intermediate `[r,M_g]` low-rank output. The low-rank output layout
is `[M_g,r]` directly.

Expert activation-offload forward LoRA-A is selected by
`ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD`, whose only valid values are `cpu` and
`hbm`.

In `cpu` mode, only the small low-rank `[M,r]` tensor is materialized in HBM.
The wide dropped LoRA input `D(U_cpu)` stays CPU-side and can be recomputed from
the saved source activation and saved dropout mask.

In `hbm` mode, forward LoRA-A uses the already-live or already-staged HBM source
for the LoRA-A GEMM. Gate/up concatenate only LoRA-A weights:
`A_gate_up_hbm = cat_r(A_gate_hbm, A_up_hbm) [E,2r,H]`, not the wide `[M,H]` activation.
The grouped result owner is `S_gate_up [M,2r]`; `S_gate` and `S_up` are views
split from that owner and the owner must stay live until both views have been
consumed by LoRA-B and offloaded. Down LoRA-A reuses the same `act_stage [M,I]`
window that is needed for the down base projection.

The activation-offload path described here is the current v0 path and requires
`lora_dropout=0.0`, so `D_b` is the identity in the validated `cpu`/`hbm` A/B.
The `D_b` notation is kept in the CPU equations to show where dropout would
apply if that constraint changes later.

`S_gate_cpu`, `S_up_cpu`, and `S_down_cpu` store the same low-rank values as
`S_gate`, `S_up`, and `S_down`. Backward equations write `S_b,g` for the
mathematical low-rank value.

`pack_g(...)` denotes the packed routed-row layout of the grouped result. It is
not a requirement to concatenate per-group outputs in Python.

All LoRA-A forward equations, LoRA-A gradient equations, and LoRA-B gradient
reductions below are grouped over `g = 0..G-1`. The `_g` equations are the
expanded math for the grouped operation/reduction, not separate ungrouped work
items.

When a wide gradient is already CPU-resident, LoRA-B backward uses a grouped
CPU-left op to compute both `dS_b` and `dB_b`. Do not stage `[M,I]` or `[M,H]`
wide gradients to HBM solely for `dS_b`/`dB_b`; stage only the saved low-rank
`S_b_cpu [M,r]`.

### Forward

```
X = packed_routed_rows                                  # [M,H] HBM
X_cpu = offload(X)                                      # [M,H] CPU, save for dA_gate/dA_up

gate_up = pack_g(X_g @^R W_gate_up_cpu[e_g].T)          # [M,2I] HBM live
gate, up = split(gate_up)                               # [M,I], [M,I] HBM views


# ---------------- gate/up LoRA-A forward: mode = cpu ----------------

X_gate_lora_cpu,g = D_gate(X_cpu,g)                     # [M_g,H] CPU
A_gate_hbm = stage(A_gate)                       # [E,r,H] HBM staged
S_gate_g = X_gate_lora_cpu,g @^L A_gate_hbm[e_g].T      # [M_g,r] HBM
release(A_gate_hbm)
S_gate = pack_g(S_gate_g)                               # [M,r] HBM
S_gate_cpu = offload(S_gate)                            # [M,r] CPU, save for dB_gate
B_gate_hbm = stage(B_gate)                       # [E,I,r] HBM staged
gate += scale * pack_g(S_gate_g @ B_gate_hbm[e_g].T)    # consume gate delta now
release(S_gate, B_gate_hbm, X_gate_lora_cpu_if_materialized)


# ---------------- up projection ----------------

X_up_lora_cpu,g = D_up(X_cpu,g)                         # [M_g,H] CPU
A_up_hbm = stage(A_up)                           # [E,r,H] HBM staged
S_up_g = X_up_lora_cpu,g @^L A_up_hbm[e_g].T            # [M_g,r] HBM
release(A_up_hbm)
S_up = pack_g(S_up_g)                                   # [M,r] HBM
S_up_cpu = offload(S_up)                                # [M,r] CPU, save for dB_up
B_up_hbm = stage(B_up)                           # [E,I,r] HBM staged
up += scale * pack_g(S_up_g @ B_up_hbm[e_g].T)          # consume up delta now
release(S_up, B_up_hbm, X_up_lora_cpu_if_materialized)


# ---------------- gate/up LoRA-A forward: mode = hbm ----------------

A_gate_hbm = stage(A_gate)                       # [E,r,H] HBM staged
A_up_hbm = stage(A_up)                           # [E,r,H] HBM staged
A_gate_up_hbm = cat_r(A_gate_hbm, A_up_hbm)             # [E,2r,H] HBM temp, weights only
S_gate_up_g = X_g @ A_gate_up_hbm[e_g].T                # [M_g,2r] HBM
release(A_gate_up_hbm, A_gate_hbm, A_up_hbm)
S_gate_g, S_up_g = split_r(S_gate_up_g)                 # [M_g,r], [M_g,r] HBM views
S_gate_up = pack_g(S_gate_up_g)                         # [M,2r] HBM owner
S_gate, S_up = split_r(S_gate_up)                       # [M,r], [M,r] HBM views
B_gate_hbm = stage(B_gate)                       # [E,I,r] HBM staged
B_up_hbm = stage(B_up)                           # [E,I,r] HBM staged
gate += scale * pack_g(S_gate_g @ B_gate_hbm[e_g].T)    # S_gate_g is view of S_gate_up_g
up += scale * pack_g(S_up_g @ B_up_hbm[e_g].T)          # S_up_g is view of S_gate_up_g
S_gate_cpu = offload(S_gate)                            # [M,r] CPU, save for dB_gate
S_up_cpu = offload(S_up)                                # [M,r] CPU, save for dB_up
release(S_gate, S_up, S_gate_up, B_gate_hbm, B_up_hbm)


gate_cpu = offload(gate)                                # [M,I] CPU
up_cpu = offload(up)                                    # [M,I] CPU
release(gate_up)

silu_gate_tmp_cpu = silu(gate_cpu)                      # [M,I] CPU temp
act_cpu = silu_gate_tmp_cpu * up_cpu                    # [M,I] CPU, save for down
release(silu_gate_tmp_cpu)


# ---------------- down projection: mode = cpu ----------------

act_down_lora_cpu,g = D_down(act_cpu,g)                 # [M_g,I] CPU
A_down_hbm = stage(A_down)                       # [E,r,I] HBM staged
S_down_g = act_down_lora_cpu,g @^L A_down_hbm[e_g].T    # [M_g,r] HBM
release(A_down_hbm)
S_down = pack_g(S_down_g)                               # [M,r] HBM
S_down_cpu = offload(S_down)                            # [M,r] CPU, save for dB_down
B_down_hbm = stage(B_down)                       # [E,H,r] HBM staged
down_delta = scale * pack_g(S_down_g @ B_down_hbm[e_g].T) # [M,H] HBM temp

act_stage = stage(act_cpu)                               # [M,I] HBM
Y_down = pack_g(act_stage_g @^R W_down_cpu[e_g].T)       # [M,H] HBM live
release(act_stage)
Y_down += down_delta
release(S_down, down_delta, B_down_hbm, act_down_lora_cpu_if_materialized)


# ---------------- down projection: mode = hbm ----------------

act_stage = stage(act_cpu)                               # [M,I] HBM, shared by LoRA-A and base
A_down_hbm = stage(A_down)                       # [E,r,I] HBM staged
S_down_g = act_stage_g @ A_down_hbm[e_g].T              # [M_g,r] HBM
release(A_down_hbm)
S_down = pack_g(S_down_g)                               # [M,r] HBM
S_down_cpu = offload(S_down)                            # [M,r] CPU, save for dB_down
B_down_hbm = stage(B_down)                       # [E,H,r] HBM staged
down_delta = scale * pack_g(S_down_g @ B_down_hbm[e_g].T) # [M,H] HBM temp

Y_down = pack_g(act_stage_g @^R W_down_cpu[e_g].T)       # [M,H] HBM live
Y_down += down_delta
release(S_down, down_delta, act_stage, B_down_hbm)
```

### Backward

```
# Backward is identical for `cpu` and `hbm` forward LoRA-A modes because the
# saved CPU low-rank tensors S_gate_cpu, S_up_cpu, and S_down_cpu have the same
# mathematical values in both modes.

dY = dL/dY_down                                         # [M,H] HBM

# ---------------- down backward ----------------

dact = pack_g(dY_g @^R W_down_cpu[e_g])                 # [M,I] HBM live
release(dY_if_owned)

S_down = stage(S_down_cpu)                              # [M,r] HBM
S_down_g = S_down[offsets[g]:offsets[g+1]]              # [M_g,r] HBM view
B_down_hbm = stage(B_down)                       # [E,H,r] HBM staged
dS_down_g = scale * (dY_g @ B_down_hbm[e_g])            # [M_g,r] HBM
dS_down = pack_g(dS_down_g)                             # [M,r] HBM
dB_down[e] = scale * sum_{g:e_g=e} dY_g.T @ S_down_g   # [H,r] Grad
release(S_down, S_down_cpu, B_down_hbm)

A_down_hbm = stage(A_down)                       # [E,r,I] HBM staged
dact += D_down_bar(pack_g(dS_down_g @ A_down_hbm[e_g])) # consume down dact delta
release(A_down_hbm)

dact_cpu = offload(dact)                                # [M,I] CPU
release(dact)

act_down_lora_cpu,g = D_down(act_cpu,g)                 # [M_g,I] CPU
dA_down[e] = sum_{g:e_g=e} dS_down_g.T @^R act_down_lora_cpu,g  # [r,I] Grad
release(dS_down, act_down_lora_cpu_if_materialized, act_cpu)


# ---------------- activation backward ----------------

dgate_cpu = silu_backward(dact_cpu * up_cpu, gate_cpu)  # [M,I] CPU
dup_cpu = dact_cpu * silu(gate_cpu)                     # [M,I] CPU
release(dact_cpu, gate_cpu, up_cpu)

dgate_up = stage_concat(dgate_cpu, dup_cpu)             # [M,2I] HBM
dgate_stage_g, dup_stage_g = split(dgate_up_g)          # [M_g,I], [M_g,I] HBM views


# ---------------- gate/up base backward ----------------

dX = pack_g(dgate_up_g @^R W_gate_up_cpu[e_g])          # [M,H] HBM live


# ---------------- gate LoRA backward ----------------

S_gate = stage(S_gate_cpu)                              # [M,r] HBM
S_gate_g = S_gate[offsets[g]:offsets[g+1]]              # [M_g,r] HBM view
B_gate_hbm = stage(B_gate)                       # [E,I,r] HBM staged
dS_gate_g = scale * (dgate_stage_g @ B_gate_hbm[e_g])   # [M_g,r] HBM
dS_gate = pack_g(dS_gate_g)                             # [M,r] HBM
dB_gate[e] = scale * sum_{g:e_g=e} dgate_stage_g.T @ S_gate_g  # [I,r] Grad
release(S_gate, S_gate_cpu, B_gate_hbm)

A_gate_hbm = stage(A_gate)                       # [E,r,H] HBM staged
dX += D_gate_bar(pack_g(dS_gate_g @ A_gate_hbm[e_g]))   # consume gate dX delta
release(A_gate_hbm)

X_gate_lora_cpu,g = D_gate(X_cpu,g)                     # [M_g,H] CPU
dA_gate[e] = sum_{g:e_g=e} dS_gate_g.T @^R X_gate_lora_cpu,g  # [r,H] Grad
release(dS_gate, X_gate_lora_cpu_if_materialized)


# ---------------- up LoRA backward ----------------

S_up = stage(S_up_cpu)                                  # [M,r] HBM
S_up_g = S_up[offsets[g]:offsets[g+1]]                  # [M_g,r] HBM view
B_up_hbm = stage(B_up)                           # [E,I,r] HBM staged
dS_up_g = scale * (dup_stage_g @ B_up_hbm[e_g])         # [M_g,r] HBM
dS_up = pack_g(dS_up_g)                                 # [M,r] HBM
dB_up[e] = scale * sum_{g:e_g=e} dup_stage_g.T @ S_up_g  # [I,r] Grad
release(S_up, S_up_cpu, B_up_hbm)

A_up_hbm = stage(A_up)                           # [E,r,H] HBM staged
dX += D_up_bar(pack_g(dS_up_g @ A_up_hbm[e_g]))         # consume up dX delta
release(A_up_hbm)

X_up_lora_cpu,g = D_up(X_cpu,g)                         # [M_g,H] CPU
dA_up[e] = sum_{g:e_g=e} dS_up_g.T @^R X_up_lora_cpu,g  # [r,H] Grad
release(dS_up, X_up_lora_cpu_if_materialized, X_cpu)
release(dgate_up)


# ---------------- final input gradient ----------------
# dX = gate/up base gradient + gate LoRA gradient + up LoRA gradient
```
