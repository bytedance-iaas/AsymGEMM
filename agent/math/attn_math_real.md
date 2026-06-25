# Dense Attention LoRA Math (Faithful to Current Code)

This file reflects what `AsymLoRALinear` and `AsymFrozenLinear` actually do in the
current codebase, not the planned activation-offload design in `attn_math.md`.

**What is NOT implemented yet:**
- Activation offload of X or S to CPU for attention projections.
- CPU-left AsymGEMM for LoRA-A forward or dA backward in attention.

The base frozen weight is CPU-resident and fetched via `@^R` (AsymGEMM CPU-right) in
both forward and backward, exactly as `attn_math.md` specifies. LoRA weights may be
CPU-home (weight offload), staged to HBM around each GEMM. LoRA activations
(`x_lora`, `S_*`) remain on HBM and are saved there across the autograd boundary.
No CPU tensor is produced in the current attention projection path.

## Notation

```text
@  = GEMM
@^L = AsymGEMM with a CPU left operand
@^R = AsymGEMM with a CPU right operand

stage(U)       = make CPU-home LoRA weight U available as an HBM tensor via
                 gather_lora_weights(); no-op view when weight offload is disabled
release(...)   = listed tensors or staged weight handles are no longer live

CPU tensors have suffix _cpu.
Tensors without _cpu are HBM tensors.
Staged LoRA weights have suffix _hbm.
Temp means HBM temporary, releasable after last use.
Grad means trainable LoRA gradient.
Save means HBM tensor kept across the autograd boundary in saved_tensors.
```

`Z.T` denotes GEMM orientation only, not a materialized transposed copy.

`stage(A)` / `stage(B)` are gather_lora_weights() call sites. When weight offload is
disabled they are no-op views of the CUDA parameter. When enabled they copy from
pinned CPU home storage to HBM; the returned `*_hbm` tensor is released after its
GEMM. The logical trainable parameter and CPU optimizer state are owned by the
weight-offload coordinator.

`Y += GEMM(...)` means accumulate into an already-live output buffer. A backend that
must materialize a GEMM result produces a one-line temporary consumed by the add.

## Shapes and Parameters

```text
B   = batch
T   = sequence length
M   = B * T
H   = hidden_size
Hq  = num_q_heads
Hkv = num_kv_heads
Dh  = head_dim
Dq  = Hq  * Dh
Dkv = Hkv * Dh
r   = LoRA rank
scale = lora_alpha / r

X        [M,H]     hidden_states flattened over batch and sequence
Q        [M,Dq]
K,V      [M,Dkv]
AttnOut  [M,Dq]
Y        [M,H]
```

```text
# Frozen dense projection weights, CPU-resident HostWeight
W_q_cpu  [Dq,H]
W_k_cpu  [Dkv,H]
W_v_cpu  [Dkv,H]
W_o_cpu  [H,Dq]

# Trainable LoRA weights.
# CPU-home when weight offload enabled; staged to HBM as A_*_hbm / B_*_hbm.
# When weight offload disabled, stage() is a no-op view of the CUDA parameter.
A_q [r,H]       B_q [Dq,r]
A_k [r,H]       B_k [Dkv,r]
A_v [r,H]       B_v [Dkv,r]
A_o [r,Dq]      B_o [H,r]
```

The projection branch map is:

```text
branch  input U   output Z  frozen W_cpu  LoRA A  LoRA B  in   out
q       X         Q         W_q_cpu       A_q     B_q     H    Dq
k       X         K         W_k_cpu       A_k     B_k     H    Dkv
v       X         V         W_v_cpu       A_v     B_v     H    Dkv
o       AttnOut   Y         W_o_cpu       A_o     B_o     Dq   H
```

## Dropout

Not supported in the weight-offload path (`_AsymLoRALinearWeightOffloadFunction`).
When `_weight_offload is not None and self.training`, the forward raises if
`lora_dropout` is not `nn.Identity`. The standard (non-weight-offload) path does
apply `lora_dropout` before casting `U` to `lora_dtype`; that path is not used in
the production activation-offload configuration.

For completeness, notation in the standard path:

```text
D_b(Z)     = Z               if p == 0
D_b(Z)     = mask_b * Z / q  if 0 < p < 1
D_b_bar(G) = G               if p == 0
D_b_bar(G) = mask_b * G / q  if 0 < p < 1
```

In the weight-offload path all equations below assume `p == 0`.

## Forward

There is no shared X_cpu across projections in the current implementation. Each
branch independently casts its input to `lora_dtype` on HBM.

The weight-offload path is used when `_weight_offload is not None and self.training`.
The description below covers that path. When weight offload is disabled, stage() is a
no-op view and the same formula holds without any H2D copies.

```text
X = hidden_states_flat                                      # [M,H] HBM


# ---------------- q projection ----------------

Q = X @^R W_q_cpu.T                                         # [M,Dq] HBM live
X_lora_q = X.to(dtype=lora_dtype)                          # [M,H] HBM, cast only
                                                            # (no CPU copy, no dropout)
A_q_hbm = stage(A_q)                                 # [r,H] HBM staged
B_q_hbm = stage(B_q)                                 # [Dq,r] HBM staged
S_q = X_lora_q @ A_q_hbm.T                                  # [M,r] HBM Save
Q += scale * (S_q @ B_q_hbm.T)                              # accumulate LoRA delta
release(A_q_hbm, B_q_hbm)
# saved across autograd: X_lora_q [M,H] HBM, S_q [M,r] HBM


# ---------------- k projection ----------------

K = X @^R W_k_cpu.T                                         # [M,Dkv] HBM live
X_lora_k = X.to(dtype=lora_dtype)                          # [M,H] HBM, cast only
A_k_hbm = stage(A_k)                                 # [r,H] HBM staged
B_k_hbm = stage(B_k)                                 # [Dkv,r] HBM staged
S_k = X_lora_k @ A_k_hbm.T                                  # [M,r] HBM Save
K += scale * (S_k @ B_k_hbm.T)                              # accumulate LoRA delta
release(A_k_hbm, B_k_hbm)
# saved across autograd: X_lora_k [M,H] HBM, S_k [M,r] HBM


# ---------------- v projection ----------------

V = X @^R W_v_cpu.T                                         # [M,Dkv] HBM live
X_lora_v = X.to(dtype=lora_dtype)                          # [M,H] HBM, cast only
A_v_hbm = stage(A_v)                                 # [r,H] HBM staged
B_v_hbm = stage(B_v)                                 # [Dkv,r] HBM staged
S_v = X_lora_v @ A_v_hbm.T                                  # [M,r] HBM Save
V += scale * (S_v @ B_v_hbm.T)                              # accumulate LoRA delta
release(A_v_hbm, B_v_hbm)
# saved across autograd: X_lora_v [M,H] HBM, S_v [M,r] HBM


# ---------------- attention prepare/core ----------------

Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)          # HBM, normal autograd
AttnOut = attention_core(Q_attn, K_attn, V_attn)             # [M,Dq] HBM


# ---------------- o projection ----------------

Y = AttnOut @^R W_o_cpu.T                                    # [M,H] HBM live
AttnOut_lora = AttnOut.to(dtype=lora_dtype)                 # [M,Dq] HBM, cast only
A_o_hbm = stage(A_o)                                  # [r,Dq] HBM staged
B_o_hbm = stage(B_o)                                  # [H,r] HBM staged
S_o = AttnOut_lora @ A_o_hbm.T                               # [M,r] HBM Save
Y += scale * (S_o @ B_o_hbm.T)                               # accumulate LoRA delta
release(A_o_hbm, B_o_hbm)
# saved across autograd: AttnOut_lora [M,Dq] HBM, S_o [M,r] HBM
```

## Backward

```text
dY = dL/dY                                                  # [M,H] HBM


# ---------------- o projection ----------------

dAttnOut = dY @^R W_o_cpu                                   # [M,Dq] HBM
B_o_hbm = stage(B_o)                                 # [H,r] HBM staged
A_o_hbm = stage(A_o)                                 # [r,Dq] HBM staged
dS_o = scale * (dY @ B_o_hbm)                               # [M,r] Temp HBM
dAttnOut += dS_o @ A_o_hbm                                  # consume LoRA dAttnOut delta
release(A_o_hbm, B_o_hbm)
dB_o = scale * (dY.T @ S_o)                                 # [H,r] Grad (S_o HBM Save)
dA_o = dS_o.T @ AttnOut_lora                                # [r,Dq] Grad (AttnOut_lora HBM Save)
release(dS_o, S_o, AttnOut_lora)


# ---------------- attention prepare/core ----------------

dQ, dK, dV = autograd(attention_prepare + attention_core, dAttnOut)
# dQ [M,Dq], dK [M,Dkv], dV [M,Dkv]


# ---------------- q projection ----------------

dX = dQ @^R W_q_cpu                                         # [M,H] HBM
B_q_hbm = stage(B_q)                                 # [Dq,r] HBM staged
A_q_hbm = stage(A_q)                                 # [r,H] HBM staged
dS_q = scale * (dQ @ B_q_hbm)                               # [M,r] Temp HBM
dX += dS_q @ A_q_hbm                                        # consume LoRA dX delta
release(A_q_hbm, B_q_hbm)
dB_q = scale * (dQ.T @ S_q)                                 # [Dq,r] Grad (S_q HBM Save)
dA_q = dS_q.T @ X_lora_q                                   # [r,H] Grad (X_lora_q HBM Save)
release(dS_q, S_q, X_lora_q)


# ---------------- k projection ----------------

dX += dK @^R W_k_cpu
B_k_hbm = stage(B_k)                                 # [Dkv,r] HBM staged
A_k_hbm = stage(A_k)                                 # [r,H] HBM staged
dS_k = scale * (dK @ B_k_hbm)                               # [M,r] Temp HBM
dX += dS_k @ A_k_hbm                                        # consume LoRA dX delta
release(A_k_hbm, B_k_hbm)
dB_k = scale * (dK.T @ S_k)                                 # [Dkv,r] Grad (S_k HBM Save)
dA_k = dS_k.T @ X_lora_k                                   # [r,H] Grad (X_lora_k HBM Save)
release(dS_k, S_k, X_lora_k)


# ---------------- v projection ----------------

dX += dV @^R W_v_cpu
B_v_hbm = stage(B_v)                                 # [Dkv,r] HBM staged
A_v_hbm = stage(A_v)                                 # [r,H] HBM staged
dS_v = scale * (dV @ B_v_hbm)                               # [M,r] Temp HBM
dX += dS_v @ A_v_hbm                                        # consume LoRA dX delta
release(A_v_hbm, B_v_hbm)
dB_v = scale * (dV.T @ S_v)                                 # [Dkv,r] Grad (S_v HBM Save)
dA_v = dS_v.T @ X_lora_v                                   # [r,H] Grad (X_lora_v HBM Save)
release(dS_v, S_v, X_lora_v)


# ---------------- final input gradient ----------------

# dX = q/k/v base gradients + q/k/v LoRA gradients          # [M,H] HBM
```

## Key Differences from attn_math.md

```text
attn_math.md (planned)                    attn_math_real.md (current code)

X_cpu = offload(X)                        No CPU copy of X; X stays HBM
X_q_cpu = D_q(X_cpu) CPU tensor          X_lora_q = X.to(lora_dtype) HBM tensor
S_q = X_q_cpu @^L A_q_hbm.T             S_q = X_lora_q @ A_q_hbm.T  (HBM GEMM)
S_q_cpu = offload(S_q)                   S_q saved on HBM (saved_tensors)
dA_q = dS_q_T @^R X_q_grad_cpu          dA_q = dS_q.T @ X_lora_q    (HBM GEMM)
dB_q = scale * (dQ.T @ stage(S_q_cpu))  dB_q = scale * (dQ.T @ S_q) (S_q is HBM)
```

The `@^L` LoRA-A forward and the CPU-side activation offload are not yet wired for
attention projections. Only the base frozen projection uses `@^R` (AsymGEMM) today.

## HBM Saved Tensor Budget (per projection branch)

```text
x_lora  [M,in]  = M * in * 2 bytes  (bf16 cast of input, stays HBM)
S       [M,r]   = M * r  * 2 bytes  (low-rank output, stays HBM)
```

Two saves per branch, both HBM. This is worse than the planned design
(attn_math.md) which saves only S on CPU [M,r] and offloads X_cpu shared.

## SDPA/FlashAttention Boundary

Same as attn_math.md. Projection activation offload is not implemented,
so projection-side saves remain HBM. Attention-core saves (LSE, Q/K/V
layout views, masks) are owned by PyTorch/FlashAttention autograd unchanged.

## Launch Contract (Current Code, per branch)

```text
forward:
  1 base CPU-right AsymGEMM:     U @^R W_cpu.T
  1 LoRA-A HBM GEMM:             U_lora @ stage(A).T
  1 LoRA-B HBM GEMM:             S @ stage(B).T

backward:
  1 base dx CPU-right AsymGEMM:  dZ @^R W_cpu
  1 dS HBM GEMM:                 scale * dZ @ stage(B)
  1 dX LoRA HBM GEMM:            dS @ stage(A)
  1 dB HBM GEMM:                 scale * dZ.T @ S
  1 dA HBM GEMM:                 dS.T @ U_lora
```

All five GEMM operands in backward are HBM-resident. No CPU-left or
CPU-right ops for LoRA in the current attention implementation.
