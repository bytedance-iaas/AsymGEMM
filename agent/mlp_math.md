###  Forward
```
X = routed_rows_for_this_expert                         # [M, H] HBM

gate_up_base = X @^ W_gate_up_cpu.T                     # [M,2I] Temp
gate_base, up_base = split(gate_up_base)                # [M,I], [M,I] Temp
S_gate = X @ A_gate.T                                   # [M,r] HBM
S_up   = X @ A_up.T                                     # [M,r] HBM

X_cpu      = offload(X)                                 # [M,H] CPU

LoRA_gate = scale * (S_gate @ B_gate.T)                 # [M,I] Temp
gate = gate_base + LoRA_gate                            # [M,I] HBM
LoRA_up   = scale * (S_up   @ B_up.T)                   # [M,I] Temp
up   = up_base   + LoRA_up                              # [M,I] HBM

gate_cpu   = offload(gate)                              # Can later fuse as   gate_cpu = offload(gate_base + LoRA_gate)
up_cpu     = offload(up)                                # Can later fuse as   up_cpu   = offload(up_base   + LoRA_up)
S_gate_cpu = offload(S_gate)                            # [M,r] CPU, save for dB_gate
S_up_cpu   = offload(S_up)                              # [M,r] CPU, save for dB_up

sig_cpu = sigmoid(gate_cpu)                           # [M, I] CPU
silu_gate_cpu = sig_cpu * gate_cpu                    # [M, I] CPU
act_cpu       = silu_gate_cpu * up_cpu                # [M, I] CPU

S_down_T  = A_down @^ act_cpu                         # [r, M] HBM, uses act_cpu.T via @^ transpose mode
LoRA_down = scale * (S_down_T.T @ B_down.T)           # [M, H] Temp
S_down_T_cpu = offload(S_down_T)                      # [r,M] CPU, save for dB_down

act = stage(act_cpu)
Y_down = act @^ W_down_cpu.T + LoRA_down               # [M, H] HBM
```

### Backward
```
dY = dL/dY_down                                      # [M, H] HBM

# ---------------- down backward ----------------



dS_down   = scale * (dY @ B_down)                    # [M, r] Temp
dact_lora = dS_down @ A_down                         # [M, I] Temp
dact_base = dY @^ W_down_cpu                         # [M, I] Temp Can later fuse dact_cpu = offload(dY @^ W_down_cpu + dact_lora)
dact = dact_base + dact_lora                         # [M, I] Temp

dact_cpu = offload(dact)

# S_down_T = A_down @^ act_cpu                        # [r, M] Recomp/Reuse S_down_T_cpu

dA_down = dS_down.T @^ act_cpu                        # [r, I] Grad
dB_down = scale * (dY.T @^ S_down_T_cpu.T)            # [H, r] Grad


# ---------------- activation backward ----------------

# sig = sigmoid(gate_cpu)                              # [M, I] Recomp/Reuse sig_cpu
# silu_gate = gate_tmp * sig                           # [M, I] Recomp/Resue silu_gate_cpu
silu_grad_cpu = sig_cpu * (1 + gate_cpu * (1 - sig_cpu))  # [M, I] CPU

dgate_cpu = dact_cpu * up_cpu * silu_grad_cpu           # [M, I] CPU 
dup_cpu   = dact_cpu * silu_gate_cpu                    # [M, I] CPU

# ---------------- gate/up base backward ----------------

dgate = stage(dgate_cpu)                              # [M, I] HBM
dup   = stage(dup_cpu)                                # [M, I] HBM

dgate_up = concat(dgate, dup)                         # [M, 2I] HBM, gate first then up
dX = dgate_up @^ W_gate_up_cpu                        # [M, H] HBM


# ---------------- gate LoRA backward ----------------

# S_gate = (A_gate @^ X_cpu).T                        # [M, r] Recomp / Reuse S_gate_cpu. Not needed.

dS_gate      = scale * (dgate @ B_gate)               # [M, r] Temp
dX_gate_lora = dS_gate @ A_gate                       # [M, H] Temp
dX += dX_gate_lora

dA_gate = dS_gate.T @^ X_cpu                          # [r, H] Grad
dB_gate = scale * (dgate.T @^ S_gate_cpu)             # [I, r] Grad


# ---------------- up LoRA backward ----------------

# S_up = (A_up @^ X_cpu).T                            # [M, r] Recomp / Reuse S_up_cpu. Not needed.

dS_up      = scale * (dup @ B_up)                     # [M, r] Temp
dX_up_lora = dS_up @ A_up                             # [M, H] Temp
dX += dX_up_lora

dA_up = dS_up.T @^ X_cpu                              # [r, H] Grad
dB_up = scale * (dup.T @^ S_up_cpu)                   # [I, r] Grad


# ---------------- final input gradient ----------------
# dX = dX_gate_base + dX_gate_lora + dX_up_base + dX_up_lora  # [M, H] HBM should have been accumualted along the way
```



