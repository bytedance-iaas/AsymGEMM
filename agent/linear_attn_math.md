# Qwen3.5 Linear Attention Math

## Notation

```text
@    = GEMM
@^L  = AsymGEMM with a CPU left operand
@^R  = AsymGEMM with a CPU right operand

offload(Z)   = copy HBM tensor to CPU and save the CPU owner
stage(U)     = make CPU tensor or CPU-home logical LoRA weight U available as an HBM tensor
release(...) = listed tensors, staged weights, or saved handles are no longer live

CPU tensors have suffix _cpu.
Tensors without _cpu are HBM tensors.
Staged LoRA weights have suffix _hbm.
```

For LoRA branch `p`:

```text
D_p(X)     = X                    if lora_dropout_p == 0
D_p(X)     = mask_p * X / q       if q = 1 - lora_dropout_p
D_p_bar(G) = G                    if lora_dropout_p == 0
D_p_bar(G) = mask_p * G / q
scale      = lora_alpha / r
```

## Shapes

```text
B   = batch
T   = sequence length
M   = B * T
H   = hidden_size
Hk  = linear_num_key_heads
Hv  = linear_num_value_heads
Dk  = linear_key_head_dim
Dv  = linear_value_head_dim
K   = Hk * Dk
V   = Hv * Dv
C   = 2K + V
R   = Hv / Hk
r   = LoRA rank

X         [M,H]
QKV_pre   [M,C]
QKV_conv  [M,C]
Q0,K0     [M,Hk,Dk]
Q,K       [M,Hv,Dk]
Val       [M,Hv,Dv]
Z         [M,Hv,Dv]
Braw      [M,Hv]
Araw      [M,Hv]
Beta      [M,Hv]
G         [M,Hv]
Core      [M,Hv,Dv]
N         [M,V]
Y         [M,H]
```

```text
W_qkv_cpu [C,H]    A_qkv [r,H]    B_qkv [C,r]    logical LoRA weights
W_z_cpu   [V,H]    A_z   [r,H]    B_z   [V,r]    logical LoRA weights
W_b_cpu   [Hv,H]   A_b   [r,H]    B_b   [Hv,r]   logical LoRA weights
W_a_cpu   [Hv,H]   A_a   [r,H]    B_a   [Hv,r]   logical LoRA weights
W_o_cpu   [H,V]    A_o   [r,V]    B_o   [H,r]    logical LoRA weights

W_conv    [C,1,Kconv]
dt_bias   [Hv]
A_log     [Hv]
gamma     [Dv]
```

With LoRA weight offload enabled, `A_*` and `B_*` are CPU-home logical
trainable weights. Every GEMM below uses the staged `A_*_hbm` or `B_*_hbm`
operand returned by `stage(...)`. With LoRA weight offload disabled,
`stage(...)` is a no-op view of the CUDA parameter. Saved CPU activations use
the same `stage(...)` API when they must be copied back to HBM.

## Forward

### Projections

```text
X_cpu = offload(X)

QKV_pre = X @^R W_qkv_cpu.T
A_qkv_hbm = stage(A_qkv)
S_qkv = D_qkv(X_cpu) @^L A_qkv_hbm.T
release(A_qkv_hbm)
S_qkv_cpu = offload(S_qkv)
B_qkv_hbm = stage(B_qkv)
QKV_pre += scale * (S_qkv @ B_qkv_hbm.T)
release(S_qkv, B_qkv_hbm)

Z_flat = X @^R W_z_cpu.T
A_z_hbm = stage(A_z)
S_z = D_z(X_cpu) @^L A_z_hbm.T
release(A_z_hbm)
S_z_cpu = offload(S_z)
B_z_hbm = stage(B_z)
Z_flat += scale * (S_z @ B_z_hbm.T)
release(S_z, B_z_hbm)

Braw = X @^R W_b_cpu.T
A_b_hbm = stage(A_b)
S_b = D_b(X_cpu) @^L A_b_hbm.T
release(A_b_hbm)
S_b_cpu = offload(S_b)
B_b_hbm = stage(B_b)
Braw += scale * (S_b @ B_b_hbm.T)
release(S_b, B_b_hbm)

Araw = X @^R W_a_cpu.T
A_a_hbm = stage(A_a)
S_a = D_a(X_cpu) @^L A_a_hbm.T
release(A_a_hbm)
S_a_cpu = offload(S_a)
B_a_hbm = stage(B_a)
Araw += scale * (S_a @ B_a_hbm.T)
release(S_a, B_a_hbm)
```

### Conv, Gates, And Heads

```text
QKV_conv = causal_conv1d_act(QKV_pre, W_conv)
Q0_flat, K0_flat, V0_flat = split(QKV_conv, [K, K, V])

Q0  = view(Q0_flat, [M,Hk,Dk])
K0  = view(K0_flat, [M,Hk,Dk])
Val = view(V0_flat, [M,Hv,Dv])
Z   = view(Z_flat,  [M,Hv,Dv])

Beta = sigmoid(Braw)
U = Araw.float() + dt_bias
G = -exp(A_log.float()) * softplus(U)

Q = repeat_interleave(Q0, R, dim=head) if R > 1 else Q0
K = repeat_interleave(K0, R, dim=head) if R > 1 else K0
```

### Gated Delta Rule

For each batch/head, with `q_t,k_t [Dk]`, `v_t [Dv]`,
`beta_t,g_t` scalars, and state `H_t [Dk,Dv]`:

```text
q_t = l2norm(Q_t) / sqrt(Dk)
k_t = l2norm(K_t)
a_t = exp(g_t)

H'_t = a_t * H_{t-1}
m_t  = H'_t.T @ k_t
d_t  = beta_t * (v_t - m_t)
H_t  = H'_t + k_t @ d_t.T
o_t  = H_t.T @ q_t
```

```text
Core = stack_t(o_t)                         # [M,Hv,Dv]
```

### Gated RMSNorm And Output Projection

For each row/head vector `c = Core[i,h,:]`, `z = Z[i,h,:]`:

```text
rho = mean(c * c) + eps
u = c * rsqrt(rho)
n = gamma * u * silu(z)
```

```text
N = view(stack_i,h(n), [M,V])
N_cpu = offload(N)

Y = N @^R W_o_cpu.T
A_o_hbm = stage(A_o)
S_o = D_o(N_cpu) @^L A_o_hbm.T
release(A_o_hbm)
S_o_cpu = offload(S_o)
B_o_hbm = stage(B_o)
Y += scale * (S_o @ B_o_hbm.T)
release(S_o, B_o_hbm)
```

## Backward

### Output Projection

```text
dY = dL/dY

dN_base = dY @^R W_o_cpu

S_o = stage(S_o_cpu)
dB_o = scale * (dY.T @ S_o)
B_o_hbm = stage(B_o)
dS_o = scale * (dY @ B_o_hbm)
release(S_o, S_o_cpu, B_o_hbm)

dA_o = dS_o.T @^R D_o(N_cpu)
A_o_hbm = stage(A_o)
dN_lora = D_o_bar(dS_o @ A_o_hbm)
dN = dN_base + dN_lora
release(dS_o, A_o_hbm, N_cpu)
```

### Gated RMSNorm

For each row/head vector:

```text
dn = dL/dn

dgamma += reduce_rows_heads(dn * u * silu(z))
du = dn * gamma * silu(z)
dz = dn * gamma * u * silu'(z)

dot = sum_j du_j * c_j
dc = rsqrt(rho) * du - c * (rho ** -1.5) * dot / Dv
```

```text
dCore = stack(dc)
dZ = stack(dz)
```

### Gated Delta Rule

Reverse over `t = T..1`. Let `dH_t` include gradients from future timesteps.

```text
do_t = dL/do_t

dH_t += q_t @ do_t.T
dq_t += H_t @ do_t

dH'_t += dH_t
dk_t += dH_t @ d_t
dd_t += dH_t.T @ k_t

dbeta_t += dd_t.T @ (v_t - m_t)
dv_t += beta_t * dd_t
dm_t += -beta_t * dd_t

dH'_t += k_t @ dm_t.T
dk_t += H'_t @ dm_t

dg_t += sum(dH'_t * H'_t)
dH_{t-1} += exp(g_t) * dH'_t
```

Backpropagate `dq_t, dk_t` through the normalizations to produce
`dQ_t, dK_t`:

```text
q_t = l2norm(Q_t) / sqrt(Dk)
k_t = l2norm(K_t)
```

Then collect token/head gradients and undo the `R` head repeat:

```text
dQ = stack_t(dQ_t)
dK = stack_t(dK_t)
dVal = stack_t(dv_t)
dBeta = stack_t(dbeta_t)
dG = stack_t(dg_t)

dQ0 = reduce_repeated_heads(dQ, R) if R > 1 else dQ
dK0 = reduce_repeated_heads(dK, R) if R > 1 else dK
```

### Gates And Conv

```text
dBraw = dBeta * Beta * (1 - Beta)

dU = dG * (-exp(A_log.float())) * sigmoid(U)
dAraw = dU.to(Araw.dtype)
dA_log += reduce_tokens(dG * G)
ddt_bias += reduce_tokens(dU)

dQKV_conv = concat(
    view(dQ0, [M,K]),
    view(dK0, [M,K]),
    view(dVal, [M,V]),
)
dQKV_pre, dW_conv = causal_conv1d_act_backward(dQKV_conv, QKV_pre, W_conv)
```

### Input Projections

For `P in {qkv,z,b,a}`:

```text
dOut_qkv = dQKV_pre
dOut_z   = view(dZ, [M,V])
dOut_b   = dBraw
dOut_a   = dAraw
```

```text
dX_P_base = dOut_P @^R W_P_cpu

S_P = stage(S_P_cpu)
dB_P = scale * (dOut_P.T @ S_P)
B_P_hbm = stage(B_P)
dS_P = scale * (dOut_P @ B_P_hbm)
release(S_P, S_P_cpu, B_P_hbm)

dA_P = dS_P.T @^R D_P(X_cpu)
A_P_hbm = stage(A_P)
dX_P_lora = D_P_bar(dS_P @ A_P_hbm)
release(dS_P, A_P_hbm)
```

The generic loop yields:

```text
dX_qkv_base, dX_qkv_lora
dX_z_base,   dX_z_lora
dX_b_base,   dX_b_lora
dX_a_base,   dX_a_lora
```

```text
dX =
  dX_qkv_base + dX_qkv_lora
+ dX_z_base   + dX_z_lora
+ dX_b_base   + dX_b_lora
+ dX_a_base   + dX_a_lora

release(X_cpu)
```
