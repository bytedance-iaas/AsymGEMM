Notation:

```
@ = GEMM
@^ = AsymGEMM with a CPU tensor operand. Use the existing transpose option as needed.
CPU = Compute and keep on CPU
HBM = Persistent in HBM until offload / last usage
Temp means this result is in HBM but can be released right after the next reuse.
Grad is the gradient
Tensors with _cpu are on CPU. Those without _cpu are on HBM.
offload is offloading from HBM to CPU
stage is staging from CPU to HBM
row_major(Z_view) materializes a contiguous row-major HBM tensor from a view
Lines that start with # are comments

D_q(Z), D_k(Z), D_v(Z), D_o(Z)
```

For branch `b in {q, k, v, o}` with saved dropout mask `mask_b` and
`p = lora_dropout_p`:

```
D_b(Z)      = Z                          if p == 0
D_b(Z)      = mask_b * Z / (1 - p)       if 0 < p < 1
D_b_grad(G) = G                          if p == 0
D_b_grad(G) = mask_b * G / (1 - p)       if 0 < p < 1
```

`D_b_grad` is the gradient of the saved inverted-dropout op wrt its input. It
uses the exact same saved mask and scale as `D_b`. `D_b(cpu_tensor)` is a CPU
elementwise mask/scale operation and does not stage the activation to HBM.

Shared for Qwen3 and Llama4 dense GQA attention.

```
M   = batch * seq
H   = hidden_size
Dq  = num_q_heads  * head_dim
Dkv = num_kv_heads * head_dim
```

`W_q_cpu/W_k_cpu/W_v_cpu/W_o_cpu` are base weights on CPU.
`A_q/B_q/A_k/B_k/A_v/B_v/A_o/B_o` are LoRA weights on HBM.

Qwen3 `attention_prepare` = q_norm/k_norm + RoPE.
Llama4 `attention_prepare` = RoPE/NoPE + optional qk_norm + NoPE temperature.
`attention_core` = fused SDPA/FlashAttention/GQA core.

Do not split `attention_core` or call `attention_core_backward` in v1.
PyTorch/FlashAttention autograd owns `attention_prepare + attention_core`
backward. Optional later: `saved_tensors_hooks` around `attention_core` only.


### Forward
```
X = hidden_states_flat                                 # [M,H] HBM

Q_base = X @^ W_q_cpu.T                                # [M,Dq] Temp
K_base = X @^ W_k_cpu.T                                # [M,Dkv] Temp
V_base = X @^ W_v_cpu.T                                # [M,Dkv] Temp

X_cpu = offload(X)                                     # [M,H] CPU, save for dA_q/dA_k/dA_v


# ---------------- q projection forward ----------------

X_q_lora_cpu = D_q(X_cpu)                              # [M,H] CPU elementwise
S_q_T = A_q @^ X_q_lora_cpu.T                          # [r,M] HBM, CPU-right X_q_lora_cpu
S_q = row_major(S_q_T.T)                               # [M,r] HBM materialization
LoRA_q = scale * (S_q @ B_q.T)                         # [M,Dq] Temp
S_q_cpu = offload(S_q)                                 # [M,r] CPU, save for dB_q
Q = Q_base + LoRA_q                                    # [M,Dq] HBM


# ---------------- k projection forward ----------------

X_k_lora_cpu = D_k(X_cpu)                              # [M,H] CPU elementwise
S_k_T = A_k @^ X_k_lora_cpu.T                          # [r,M] HBM, CPU-right X_k_lora_cpu
S_k = row_major(S_k_T.T)                               # [M,r] HBM materialization
LoRA_k = scale * (S_k @ B_k.T)                         # [M,Dkv] Temp
S_k_cpu = offload(S_k)                                 # [M,r] CPU, save for dB_k
K = K_base + LoRA_k                                    # [M,Dkv] HBM


# ---------------- v projection forward ----------------

X_v_lora_cpu = D_v(X_cpu)                              # [M,H] CPU elementwise
S_v_T = A_v @^ X_v_lora_cpu.T                          # [r,M] HBM, CPU-right X_v_lora_cpu
S_v = row_major(S_v_T.T)                               # [M,r] HBM materialization
LoRA_v = scale * (S_v @ B_v.T)                         # [M,Dkv] Temp
S_v_cpu = offload(S_v)                                 # [M,r] CPU, save for dB_v
V = V_base + LoRA_v                                    # [M,Dkv] HBM


# ---------------- attention prepare/core forward ----------------

Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)     # HBM, normal PyTorch autograd

AttnOut = attention_core(Q_attn, K_attn, V_attn)        # [M,Dq] HBM, fused SDPA/FA autograd
# No Q_attn/K_attn/V_attn/Core_state manual offload in v1.


# ---------------- o projection forward ----------------

Y_base = AttnOut @^ W_o_cpu.T                           # [M,H] Temp

AttnOut_cpu = offload(AttnOut)                          # [M,Dq] CPU, save for dA_o
AttnOut_o_lora_cpu = D_o(AttnOut_cpu)                   # [M,Dq] CPU elementwise
S_o_T = A_o @^ AttnOut_o_lora_cpu.T                     # [r,M] HBM, CPU-right AttnOut_o_lora_cpu
S_o = row_major(S_o_T.T)                                # [M,r] HBM materialization

LoRA_o = scale * (S_o @ B_o.T)                          # [M,H] Temp
S_o_cpu = offload(S_o)                                  # [M,r] CPU, save for dB_o
Y = Y_base + LoRA_o                                     # [M,H] HBM
```

### Backward
```
dY = dL/dY                                             # [M,H] HBM

# ---------------- o projection backward ----------------

dAttn_base = dY @^ W_o_cpu                             # [M,Dq] Temp

dS_o = scale * (dY @ B_o)                              # [M,r] Temp
dAttn_lora_raw = dS_o @ A_o                            # [M,Dq] Temp
dAttn_lora = D_o_grad(dAttn_lora_raw)                  # [M,Dq] Temp, saved mask_o/scale
dAttnOut = dAttn_base + dAttn_lora                     # [M,Dq] HBM

AttnOut_o_lora_cpu = D_o(AttnOut_cpu)                  # [M,Dq] CPU elementwise
dA_o = dS_o.T @^ AttnOut_o_lora_cpu                    # [r,Dq] Grad, CPU-right
dB_o = scale * (dY.T @^ S_o_cpu)                       # [H,r] Grad, CPU-right


# ---------------- attention prepare/core backward ----------------

# Normal PyTorch/FlashAttention autograd computes this.
# Do not manually stage Q_attn/K_attn/V_attn or call attention_core_backward in v1.
dQ, dK, dV = autograd(attention_prepare + attention_core, dAttnOut)  # [M,Dq], [M,Dkv], [M,Dkv] HBM


# ---------------- q projection backward ----------------

dX = dQ @^ W_q_cpu                                     # [M,H] HBM

# S_q = D_q(X) @ A_q.T                                 # [M,r] Recomp/Reuse S_q_cpu

dS_q      = scale * (dQ @ B_q)                         # [M,r] Temp
dX_q_raw  = dS_q @ A_q                                 # [M,H] Temp
dX_q_lora = D_q_grad(dX_q_raw)                         # [M,H] Temp, saved mask_q/scale
dX += dX_q_lora

X_q_lora_cpu = D_q(X_cpu)                              # [M,H] CPU elementwise
dA_q = dS_q.T @^ X_q_lora_cpu                          # [r,H] Grad, CPU-right
dB_q = scale * (dQ.T @^ S_q_cpu)                       # [Dq,r] Grad, CPU-right


# ---------------- k projection backward ----------------

dX_k_base = dK @^ W_k_cpu                              # [M,H] Temp
dX += dX_k_base

# S_k = D_k(X) @ A_k.T                                 # [M,r] Recomp/Reuse S_k_cpu

dS_k      = scale * (dK @ B_k)                         # [M,r] Temp
dX_k_raw  = dS_k @ A_k                                 # [M,H] Temp
dX_k_lora = D_k_grad(dX_k_raw)                         # [M,H] Temp, saved mask_k/scale
dX += dX_k_lora

X_k_lora_cpu = D_k(X_cpu)                              # [M,H] CPU elementwise
dA_k = dS_k.T @^ X_k_lora_cpu                          # [r,H] Grad, CPU-right
dB_k = scale * (dK.T @^ S_k_cpu)                       # [Dkv,r] Grad, CPU-right


# ---------------- v projection backward ----------------

dX_v_base = dV @^ W_v_cpu                              # [M,H] Temp
dX += dX_v_base

# S_v = D_v(X) @ A_v.T                                 # [M,r] Recomp/Reuse S_v_cpu

dS_v      = scale * (dV @ B_v)                         # [M,r] Temp
dX_v_raw  = dS_v @ A_v                                 # [M,H] Temp
dX_v_lora = D_v_grad(dX_v_raw)                         # [M,H] Temp, saved mask_v/scale
dX += dX_v_lora

X_v_lora_cpu = D_v(X_cpu)                              # [M,H] CPU elementwise
dA_v = dS_v.T @^ X_v_lora_cpu                          # [r,H] Grad, CPU-right
dB_v = scale * (dV.T @^ S_v_cpu)                       # [Dkv,r] Grad, CPU-right


# ---------------- final input gradient ----------------
# dX = q/k/v base gradients + q/k/v LoRA gradients      # [M,H] HBM accumulated along the way
```
