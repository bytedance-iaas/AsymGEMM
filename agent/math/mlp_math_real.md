# Qwen3 MoE Expert MLP Math (Faithful to Current Code)

Faithful to `_ActivationOffloadQwen3ExpertFunction`. Forward and backward are identical
for cpu/hbm LoRA-A modes because saved CPU tensors have the same values in both.

The key difference from `mlp_math.md`: `dY` is never offloaded; `dS_*` and `dB_*` are
plain HBM GEMMs; only `dA_*` use `@^R` because the saved activations are CPU-resident.

## Notation

```text
@   = GEMM (both operands HBM)
@^L = AsymGEMM, CPU left operand
@^R = AsymGEMM, CPU right operand

E = experts; G = active groups; offsets[0:G+1]; experts[0:G]
M_g = offsets[g+1] - offsets[g]; M = sum_g M_g; e_g = experts[g]
Z_g = Z[offsets[g]:offsets[g+1]] for any packed routed-row tensor Z

H = hidden size; I = intermediate size; r = LoRA rank; scale = lora_alpha / r

X              [M,H]     packed routed rows, HBM
gate, up        [M,I]
act             [M,I]     silu(gate) * up, computed on CPU
Y_down          [M,H]

W_gate_up_cpu  [E,2I,H]  frozen, CPU-resident
W_down_cpu     [E,H,I]   frozen, CPU-resident

A_gate [E,r,H]    B_gate [E,I,r]
A_up   [E,r,H]    B_up   [E,I,r]
A_down [E,r,I]    B_down [E,H,r]

offload(Z)             = copy HBM tensor to pinned CPU; owner kept across autograd
stage(U_cpu)           = copy CPU tensor to HBM for immediate use
stage_concat(U,V)      = stage [U;V] concatenated into one HBM buffer
release(...)           = free listed HBM tensors or CPU owners
```

`D_b` / `D_b_bar` are identity (lora_dropout=0 required).

## Forward

### Shared prefix (both LoRA-A modes)

```text
X_cpu = offload(X)                                      # [M,H] CPU — saved for dA_gate, dA_up
gate_up = pack_g(X_g @^R W_gate_up_cpu[e_g].T)         # [M,2I] HBM
gate, up = split(gate_up)                               # [M,I], [M,I] HBM views
```

### gate/up LoRA-A: mode = cpu  (ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu, default)

```text
A_gate_hbm = stage(A_gate)                             # [E,r,H] HBM staged
A_up_hbm   = stage(A_up)                               # [E,r,H] HBM staged
gate_low_rank_g = X_cpu,g @^L A_gate_hbm[e_g].T        # [M_g,r] HBM (cpu-left grouped)
up_low_rank_g   = X_cpu,g @^L A_up_hbm[e_g].T          # [M_g,r] HBM (cpu-left grouped)
release(A_gate_hbm, A_up_hbm)
gate_low_rank = pack_g(gate_low_rank_g)                 # [M,r] HBM
up_low_rank   = pack_g(up_low_rank_g)                   # [M,r] HBM
```

### gate/up LoRA-A: mode = hbm  (ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm)

```text
A_gate_up_hbm = stage_concat(A_gate, A_up)             # [E,2r,H] HBM staged (weights only)
gate_up_low_rank_g = X_g @ A_gate_up_hbm[e_g].T        # [M_g,2r] HBM
release(A_gate_up_hbm)
gate_low_rank_g, up_low_rank_g = split_r(gate_up_low_rank_g)   # [M_g,r] HBM views
gate_low_rank, up_low_rank = split_r(pack_g(gate_up_low_rank_g))  # [M,r] HBM views
```

### gate/up LoRA-B and activation (both modes continue identically)

```text
B_gate_hbm = stage(B_gate)                             # [E,I,r] HBM staged
B_up_hbm   = stage(B_up)                               # [E,I,r] HBM staged
gate += scale * pack_g(gate_low_rank_g @ B_gate_hbm[e_g].T)  # [M,I] accumulate LoRA delta
up   += scale * pack_g(up_low_rank_g   @ B_up_hbm[e_g].T)    # [M,I] accumulate LoRA delta
release(B_gate_hbm, B_up_hbm)

gate_cpu          = offload(gate)                       # [M,I] CPU — saved for silu_bwd
up_cpu            = offload(up)                         # [M,I] CPU — saved for silu_bwd
gate_low_rank_cpu = offload(gate_low_rank)              # [M,r] CPU — saved for dB_gate
up_low_rank_cpu   = offload(up_low_rank)                # [M,r] CPU — saved for dB_up
release(gate, up, gate_low_rank, up_low_rank, gate_up)

act_cpu = silu(gate_cpu) * up_cpu                      # [M,I] CPU — saved for down and dA_down
```

### down projection: mode = cpu

```text
A_down_hbm    = stage(A_down)                          # [E,r,I] HBM staged
down_low_rank_g = act_cpu,g @^L A_down_hbm[e_g].T     # [M_g,r] HBM (cpu-left grouped)
release(A_down_hbm)
down_low_rank = pack_g(down_low_rank_g)                # [M,r] HBM

B_down_hbm    = stage(B_down)                          # [E,H,r] HBM staged
down_delta    = scale * pack_g(down_low_rank_g @ B_down_hbm[e_g].T)  # [M,H] HBM
release(B_down_hbm)
down_low_rank_cpu = offload(down_low_rank)              # [M,r] CPU — saved for dB_down
release(down_low_rank)

act_stage = stage(act_cpu)                             # [M,I] HBM transient
Y_down = pack_g(act_stage_g @^R W_down_cpu[e_g].T)    # [M,H] HBM base
release(act_stage)
Y_down += down_delta; release(down_delta)
```

### down projection: mode = hbm

```text
act_stage = stage(act_cpu)                             # [M,I] HBM, shared by LoRA-A and base

A_down_hbm    = stage(A_down)                          # [E,r,I] HBM staged
down_low_rank_g = act_stage_g @ A_down_hbm[e_g].T     # [M_g,r] HBM
release(A_down_hbm)
down_low_rank = pack_g(down_low_rank_g)                # [M,r] HBM

B_down_hbm    = stage(B_down)                          # [E,H,r] HBM staged
down_delta    = scale * pack_g(down_low_rank_g @ B_down_hbm[e_g].T)  # [M,H] HBM
release(B_down_hbm)
down_low_rank_cpu = offload(down_low_rank)              # [M,r] CPU — saved for dB_down
release(down_low_rank)

Y_down = pack_g(act_stage_g @^R W_down_cpu[e_g].T)    # [M,H] HBM base
Y_down += down_delta; release(down_delta, act_stage)
```

## Backward

```text
dY = dL/dY_down                                        # [M,H] HBM — NOT offloaded


# ---------------- down backward ----------------

dY_lora = dY.to(lora_dtype)                            # [M,H] HBM cast, no copy

B_down_hbm  = stage(B_down)                            # [E,H,r] HBM staged
dS_down_g   = scale * (dY_lora,g @ B_down_hbm[e_g])   # [M_g,r] HBM
dS_down     = pack_g(dS_down_g)                        # [M,r] HBM

S_down      = stage(down_low_rank_cpu)                  # [M,r] HBM transient
dB_down[e]  = scale * sum_{g:e_g=e} dY_lora,g.T @ S_down_g  # [E,H,r] Grad
release(S_down, down_low_rank_cpu, B_down_hbm)

dact        = pack_g(dY_g @^R W_down_cpu[e_g])         # [M,I] HBM base dx
A_down_hbm  = stage(A_down)                            # [E,r,I] HBM staged
dact       += pack_g(dS_down_g @ A_down_hbm[e_g])      # [M,I] LoRA delta
release(A_down_hbm)

dA_down[e]  = sum_{g:e_g=e} dS_down_g.T @^R act_cpu,g  # [E,r,I] Grad (CPU-right)
release(dS_down, act_cpu)


# ---------------- activation backward: CPU path (default) ----------------

dact_cpu      = offload(dact); release(dact)            # [M,I] CPU
dgate_cpu     = silu_backward(dact_cpu * up_cpu, gate_cpu)  # [M,I] CPU
dup_cpu       = dact_cpu * silu(gate_cpu)               # [M,I] CPU
release(dact_cpu, gate_cpu, up_cpu)

dgate_up      = stage_concat(dgate_cpu, dup_cpu)        # [M,2I] HBM
release(dgate_cpu, dup_cpu)
dgate_stage_g, dup_stage_g = split(dgate_up_g)          # [M_g,I], [M_g,I] HBM views


# ---------------- activation backward: GPU path (ASYMM_EXPERT_SILU_BWD_GPU) ----------------

gate_stage    = stage(gate_cpu); release(gate_cpu)       # [M,I] HBM transient
up_stage      = stage(up_cpu);   release(up_cpu)         # [M,I] HBM transient
dgate_up      = empty([M,2I] HBM)                       # [M,2I] HBM preallocated
dgate_stage_g, dup_stage_g = split(dgate_up_g)          # [M_g,I], [M_g,I] HBM views
dup_stage    := dact * silu(gate_stage)                  # in-place fill
dgate_stage  := silu_backward(dact * up_stage, gate_stage)  # in-place fill
release(dact, gate_stage, up_stage)


# ---------------- gate/up base backward ----------------

dX = pack_g(dgate_up_g @^R W_gate_up_cpu[e_g])          # [M,H] HBM live


# ---------------- gate LoRA backward ----------------

B_gate_hbm  = stage(B_gate)                            # [E,I,r] HBM staged
dS_gate_g   = scale * (dgate_stage_g @ B_gate_hbm[e_g])  # [M_g,r] HBM
dS_gate     = pack_g(dS_gate_g)                         # [M,r] HBM

S_gate      = stage(gate_low_rank_cpu)                  # [M,r] HBM transient
dB_gate[e]  = scale * sum_{g:e_g=e} dgate_stage_g.T @ S_gate_g  # [E,I,r] Grad
release(S_gate, gate_low_rank_cpu, B_gate_hbm)

A_gate_hbm  = stage(A_gate)                            # [E,r,H] HBM staged
dX         += pack_g(dS_gate_g @ A_gate_hbm[e_g])      # [M,H] LoRA delta
release(A_gate_hbm)


# ---------------- up LoRA backward ----------------

B_up_hbm    = stage(B_up)                              # [E,I,r] HBM staged
dS_up_g     = scale * (dup_stage_g @ B_up_hbm[e_g])    # [M_g,r] HBM
dS_up       = pack_g(dS_up_g)                           # [M,r] HBM

S_up        = stage(up_low_rank_cpu)                    # [M,r] HBM transient
dB_up[e]    = scale * sum_{g:e_g=e} dup_stage_g.T @ S_up_g  # [E,I,r] Grad
release(S_up, up_low_rank_cpu, B_up_hbm)

A_up_hbm    = stage(A_up)                              # [E,r,H] HBM staged
dX         += pack_g(dS_up_g @ A_up_hbm[e_g])          # [M,H] LoRA delta
release(A_up_hbm)


# ---------------- gate/up dA: CPU-right (X_cpu is CPU-resident) ----------------

dA_gate[e]  = sum_{g:e_g=e} dS_gate_g.T @^R X_cpu,g   # [E,r,H] Grad (CPU-right)
dA_up[e]    = sum_{g:e_g=e} dS_up_g.T   @^R X_cpu,g   # [E,r,H] Grad (CPU-right)
release(dS_gate, dS_up, X_cpu)
release(dgate_up)
```

## Operand placement rule

```text
dA_*  input-weight grad:   input activation is CPU-resident → CPU-right (@^R)
dB_*  output-weight grad:  low-rank S is CPU-resident → stage S to HBM, plain @
dS_*  chain rule:          gradient dY is HBM → plain @
base dx:                   frozen W is CPU-resident → CPU-right (@^R)
```
