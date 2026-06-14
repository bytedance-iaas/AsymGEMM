# Dense Attention LoRA Activation-Offload Math

This file is the ground-truth math for attention-side activation offload. The
implementation plan in `agent/impls/attn_act_offload.md` must follow this file.

Scope:

```text
Models:
  Qwen3 / Qwen3-MoE text attention
  Llama4 text attention

Weights:
  frozen q/k/v/o base weights on CPU HostWeight
  trainable LoRA A/B weights on HBM

Offloaded activations:
  projection inputs for q/k/v/o LoRA-A and dA
  low-rank S tensors used for dB

Unchanged in v1:
  q/k norm, RoPE/NoPE, temperature scaling, masks, KV-cache semantics,
  SDPA/FlashAttention/eager attention forward and backward
```

Do not split or reimplement attention core in v1. PyTorch/FlashAttention
autograd owns `attention_prepare + attention_core`. This design only changes the
projection leaves around that core.

## Notation

```text
@   = ordinary HBM GEMM
@^R = AsymGEMM with a CPU right operand

stage(Z_cpu)   = copy CPU tensor to an HBM tensor for immediate use
offload(Z)     = copy HBM tensor to CPU and save the CPU owner
row_major(V)   = materialize a contiguous row-major HBM tensor from view V
align_up(n, a) = smallest multiple of a greater than or equal to n

CPU tensors have suffix _cpu.
Tensors without _cpu are HBM tensors.
Temp means HBM temporary, releasable after last use.
Grad means trainable LoRA gradient.
```

`Z.T` denotes GEMM orientation only. It does not imply that a transposed view is
safe to pass into AsymGEMM. Any HBM operand passed to CPU-right AsymGEMM must be
materialized contiguous first.

CPU-right helper contract:

```text
hbm_cpu_matmul(L, R_cpu, transpose_b=False) = L @ R_cpu.T
  L      [M_left,K] HBM contiguous bf16
  R_cpu  [N,K]      CPU contiguous pinned bf16
  out    [M_left,N] HBM

hbm_cpu_matmul(L, R_cpu, transpose_b=True) = L @ R_cpu
  L      [M_left,K] HBM contiguous bf16
  R_cpu  [K,N]      CPU contiguous pinned bf16
  out    [M_left,N] HBM
```

The activation CPU-right helper is bf16-only in v1. Do not use fp8/fp4
quantized host-weight paths for arbitrary saved CPU activations.

## Shapes

```text
M   = batch * seq
H   = hidden_size
Dq  = num_q_heads  * head_dim
Dkv = num_kv_heads * head_dim
r   = LoRA rank
scale = lora_alpha / r

X        [M,H]    hidden_states flattened over batch and sequence
Q        [M,Dq]
K,V      [M,Dkv]
AttnOut  [M,Dq]
Y        [M,H]

W_q_cpu  [Dq,H]     A_q [r,H]    B_q [Dq,r]
W_k_cpu  [Dkv,H]    A_k [r,H]    B_k [Dkv,r]
W_v_cpu  [Dkv,H]    A_v [r,H]    B_v [Dkv,r]
W_o_cpu  [H,Dq]     A_o [r,Dq]   B_o [H,r]
```

`Qwen3 attention_prepare` includes q/k norm and RoPE. `Llama4
attention_prepare` includes RoPE/NoPE, optional qk_norm, and NoPE temperature.
`attention_core` is the selected SDPA/FlashAttention/eager GQA core.

## Dropout

For branch `b in {q,k,v,o}`, with saved dropout mask `mask_b`,
`p = lora_dropout_p`, and `q = 1 - p`:

```text
D_b(Z)     = Z                    if p == 0
D_b(Z)     = mask_b * Z / q       if 0 < p < 1
D_b_bar(G) = G                    if p == 0
D_b_bar(G) = mask_b * G / q       if 0 < p < 1
```

`D_b_bar` is the gradient of the saved inverted-dropout op wrt its input. It
uses the exact same saved mask and scale as `D_b`. `D_b(cpu_tensor)` is a CPU
elementwise mask/scale op and does not stage the wide activation to HBM.

`p == 1` is unsupported for this wrapper and must fail clearly.

## Single Projection Primitive

For a projection with input `U [M,in]`, output `Z [M,out]`, frozen base
`W_cpu [out,in]`, LoRA `A [r,in]`, `B [out,r]`, and branch dropout `D`:

### Forward

```text
Base = U @^R W_cpu.T                                      # [M,out] HBM

U_cpu = offload_or_share(U)                               # [M,in] CPU
U_drop_cpu = D(U_cpu)                                     # [M,in] CPU

M_fwd = align_up(M, 8)
U_fwd_cpu = pad_rows(U_drop_cpu, M_fwd)                   # [M_fwd,in] CPU
S_T_pad = hbm_cpu_matmul(A, U_fwd_cpu, transpose_b=False) # [r,M_fwd] HBM
S_T = S_T_pad[:, :M]                                      # [r,M] HBM Temp
S = row_major(S_T.T)                                      # [M,r] HBM
S_cpu = offload(S)                                        # [M,r] CPU, save for dB

Delta = scale * (S @ B.T)                                 # [M,out] HBM Temp
Z = Base + Delta                                          # [M,out] HBM
```

Only `U_cpu`, the optional dropout mask, `S_cpu`, scalar metadata, shape/dtype
metadata, and LoRA A/B tensor references are saved across the autograd boundary.
Do not save the wide HBM `U`, `U_drop`, `S`, or `S_T`.

### Backward

Given `dZ [M,out]`:

```text
dU_base = dZ @^R W_cpu                                    # [M,in] HBM

dS = scale * (dZ @ B)                                     # [M,r] HBM Temp
dU_lora_raw = dS @ A                                      # [M,in] HBM Temp
dU_lora = D_bar(dU_lora_raw)                              # [M,in] HBM Temp
dU = dU_base + dU_lora                                    # [M,in] HBM

U_drop_cpu = D(U_cpu)                                     # [M,in] CPU recompute
M_grad = align_up(M, 64)
U_grad_cpu = pad_rows(U_drop_cpu, M_grad)                 # [M_grad,in] CPU
dS_T = pad_cols(row_major(dS.T), M_grad)                  # [r,M_grad] HBM
dA = hbm_cpu_matmul(dS_T, U_grad_cpu, transpose_b=True)   # [r,in] Grad

S_stage = stage(S_cpu)                                    # [M,r] HBM small
dB = scale * (dZ.T @ S_stage)                             # [out,r] Grad
release_stage(S_stage)
```

The `dB` path intentionally stages the small `S_cpu [M,r]` tensor and uses an
ordinary HBM GEMM. Do not use CPU-right AsymGEMM for `dB` in v1; it would force
wide transposed HBM materialization while saving little CPU traffic.

Direct bf16 constraints:

```text
Forward LoRA-A: in % 8 == 0 and M_fwd % 8 == 0
dA:             in % 8 == 0 and M_grad % 64 == 0
Base dx:        frozen-linear transpose_b constraints from frozen_linear.py
```

Under `backend == "asym"`, unsupported shapes fail loudly. Test/debug torch
fallbacks must be explicit and recorded.

## Full Attention Forward

```text
X = hidden_states_flat                                    # [M,H] HBM

# q/k/v base projections
Q_base = X @^R W_q_cpu.T                                  # [M,Dq] Temp
K_base = X @^R W_k_cpu.T                                  # [M,Dkv] Temp
V_base = X @^R W_v_cpu.T                                  # [M,Dkv] Temp

# q/k/v share one CPU source activation when the same X reaches all three leaves.
X_cpu = offload_or_share_qkv_source(X)                    # [M,H] CPU

# q LoRA
X_q_cpu = D_q(X_cpu)                                      # [M,H] CPU
M_fwd = align_up(M, 8)
X_q_fwd_cpu = pad_rows(X_q_cpu, M_fwd)                    # [M_fwd,H] CPU
S_q_T_pad = hbm_cpu_matmul(A_q, X_q_fwd_cpu, transpose_b=False)  # [r,M_fwd] HBM
S_q = row_major(S_q_T_pad[:, :M].T)                       # [M,r] HBM
S_q_cpu = offload(S_q)                                    # [M,r] CPU
Q = Q_base + scale * (S_q @ B_q.T)                        # [M,Dq] HBM

# k LoRA
X_k_cpu = D_k(X_cpu)                                      # [M,H] CPU
X_k_fwd_cpu = pad_rows(X_k_cpu, M_fwd)                    # [M_fwd,H] CPU
S_k_T_pad = hbm_cpu_matmul(A_k, X_k_fwd_cpu, transpose_b=False)  # [r,M_fwd] HBM
S_k = row_major(S_k_T_pad[:, :M].T)                       # [M,r] HBM
S_k_cpu = offload(S_k)                                    # [M,r] CPU
K = K_base + scale * (S_k @ B_k.T)                        # [M,Dkv] HBM

# v LoRA
X_v_cpu = D_v(X_cpu)                                      # [M,H] CPU
X_v_fwd_cpu = pad_rows(X_v_cpu, M_fwd)                    # [M_fwd,H] CPU
S_v_T_pad = hbm_cpu_matmul(A_v, X_v_fwd_cpu, transpose_b=False)  # [r,M_fwd] HBM
S_v = row_major(S_v_T_pad[:, :M].T)                       # [M,r] HBM
S_v_cpu = offload(S_v)                                    # [M,r] CPU
V = V_base + scale * (S_v @ B_v.T)                        # [M,Dkv] HBM

# attention prepare/core remains normal model code and normal autograd
Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)        # HBM
AttnOut = attention_core(Q_attn, K_attn, V_attn)           # [M,Dq] HBM

# o projection
Y_base = AttnOut @^R W_o_cpu.T                             # [M,H] Temp
AttnOut_cpu = offload(AttnOut)                             # [M,Dq] CPU
AttnOut_o_cpu = D_o(AttnOut_cpu)                           # [M,Dq] CPU
AttnOut_fwd_cpu = pad_rows(AttnOut_o_cpu, M_fwd)           # [M_fwd,Dq] CPU
S_o_T_pad = hbm_cpu_matmul(A_o, AttnOut_fwd_cpu, transpose_b=False)  # [r,M_fwd] HBM
S_o = row_major(S_o_T_pad[:, :M].T)                        # [M,r] HBM
S_o_cpu = offload(S_o)                                     # [M,r] CPU
Y = Y_base + scale * (S_o @ B_o.T)                         # [M,H] HBM
```

`M_fwd` is the same row padding value for q/k/v/o within one attention call when
all use the same flattened token count.

## Full Attention Backward

```text
dY = dL/dY                                                # [M,H] HBM

# o projection
dAttn_base = dY @^R W_o_cpu                               # [M,Dq] HBM
dS_o = scale * (dY @ B_o)                                 # [M,r] Temp
dAttn_lora = D_o_bar(dS_o @ A_o)                          # [M,Dq] Temp
dAttnOut = dAttn_base + dAttn_lora                        # [M,Dq] HBM

AttnOut_o_cpu = D_o(AttnOut_cpu)                          # [M,Dq] CPU
M_grad = align_up(M, 64)
AttnOut_grad_cpu = pad_rows(AttnOut_o_cpu, M_grad)        # [M_grad,Dq] CPU
dS_o_T = pad_cols(row_major(dS_o.T), M_grad)              # [r,M_grad] HBM
dA_o = hbm_cpu_matmul(dS_o_T, AttnOut_grad_cpu, transpose_b=True)  # [r,Dq] Grad
S_o_stage = stage(S_o_cpu)                                # [M,r] HBM
dB_o = scale * (dY.T @ S_o_stage)                         # [H,r] Grad
release_stage(S_o_stage)

# attention prepare/core backward is PyTorch/FlashAttention autograd
dQ, dK, dV = autograd(attention_prepare + attention_core, dAttnOut)
# dQ [M,Dq], dK [M,Dkv], dV [M,Dkv]

# q projection
dX = dQ @^R W_q_cpu                                       # [M,H] HBM
dS_q = scale * (dQ @ B_q)                                 # [M,r] Temp
dX += D_q_bar(dS_q @ A_q)                                 # [M,H] HBM
X_q_cpu = D_q(X_cpu)                                      # [M,H] CPU
X_q_grad_cpu = pad_rows(X_q_cpu, M_grad)                  # [M_grad,H] CPU
dS_q_T = pad_cols(row_major(dS_q.T), M_grad)              # [r,M_grad] HBM
dA_q = hbm_cpu_matmul(dS_q_T, X_q_grad_cpu, transpose_b=True)  # [r,H] Grad
S_q_stage = stage(S_q_cpu)                                # [M,r] HBM
dB_q = scale * (dQ.T @ S_q_stage)                         # [Dq,r] Grad
release_stage(S_q_stage)

# k projection
dX_k_base = dK @^R W_k_cpu                                # [M,H] Temp
dX += dX_k_base
dS_k = scale * (dK @ B_k)                                 # [M,r] Temp
dX += D_k_bar(dS_k @ A_k)                                 # [M,H] HBM
X_k_cpu = D_k(X_cpu)                                      # [M,H] CPU
X_k_grad_cpu = pad_rows(X_k_cpu, M_grad)                  # [M_grad,H] CPU
dS_k_T = pad_cols(row_major(dS_k.T), M_grad)              # [r,M_grad] HBM
dA_k = hbm_cpu_matmul(dS_k_T, X_k_grad_cpu, transpose_b=True)  # [r,H] Grad
S_k_stage = stage(S_k_cpu)                                # [M,r] HBM
dB_k = scale * (dK.T @ S_k_stage)                         # [Dkv,r] Grad
release_stage(S_k_stage)

# v projection
dX_v_base = dV @^R W_v_cpu                                # [M,H] Temp
dX += dX_v_base
dS_v = scale * (dV @ B_v)                                 # [M,r] Temp
dX += D_v_bar(dS_v @ A_v)                                 # [M,H] HBM
X_v_cpu = D_v(X_cpu)                                      # [M,H] CPU
X_v_grad_cpu = pad_rows(X_v_cpu, M_grad)                  # [M_grad,H] CPU
dS_v_T = pad_cols(row_major(dS_v.T), M_grad)              # [r,M_grad] HBM
dA_v = hbm_cpu_matmul(dS_v_T, X_v_grad_cpu, transpose_b=True)  # [r,H] Grad
S_v_stage = stage(S_v_cpu)                                # [M,r] HBM
dB_v = scale * (dV.T @ S_v_stage)                         # [Dkv,r] Grad
release_stage(S_v_stage)

# final input gradient
# dX = q/k/v base gradients + q/k/v LoRA gradients         # [M,H] HBM
```

## Q/K/V Source Sharing

`X_cpu` should be shared across q/k/v when the same hidden-state tensor reaches
all three projection leaves. Each branch still owns an independent dropout mask
and independent `S_*_cpu`.

Required source key before per-leaf contiguous materialization:

```text
(
  device,
  untyped_storage().data_ptr(),
  storage_offset(),
  shape,
  stride,
  dtype,
)
```

Clearing the forward lookup after v must not invalidate backward. Each q/k/v
autograd node must retain the shared CPU handle until its backward use is done.
If refcounting is not implemented in the first version, keep the shared CPU
source alive for the attention-layer lifetime and report that lifetime.

## SDPA/FlashAttention Boundary

Projection activation offload can remove the projection-side saved HBM tensors:

```text
q/k/v/o LoRA inputs
q/k/v shared source activation duplicate saves
S_q/S_k/S_v/S_o HBM saves for dB
frozen base projection weights from HBM
```

It does not remove attention-core saved tensors in v1:

```text
Q_attn/K_attn/V_attn and layout views needed by autograd
RoPE/qk_norm/NoPE intermediates saved by model code
SDPA/FlashAttention internal backward state, such as LSE/softmax stats
attention masks or dropout state owned by the selected attention backend
```

This can still beat attention-side gradient checkpointing for projection-heavy
LoRA memory because checkpointing recomputes q/k/v/o projection branches and
materializes the same intermediate tensors again during backward. This design
keeps the wide projection sources CPU-resident and fetches only the needed
matrix operands through AsymGEMM. However, for memory dominated by
SDPA/FlashAttention internals, projection offload alone will not remove the
remaining peak. A later stage may test scoped `saved_tensors_hooks` around
`attention_prepare + attention_core`, but FA/SDPA kernels remain unmodified.

## Launch Contract

Per wrapped projection:

```text
Forward:
  1 base CPU-right AsymGEMM:       U @^R W_cpu.T
  1 LoRA-A CPU-right AsymGEMM:     A @^R U_drop_cpu.T
  1 LoRA-B HBM GEMM:               S @ B.T

Backward:
  1 base dx CPU-right AsymGEMM:    dZ @^R W_cpu
  1 dS HBM GEMM:                   dZ @ B
  1 LoRA input HBM GEMM:           dS @ A
  1 dA CPU-right AsymGEMM:         row_major(dS.T) @^R U_drop_cpu
  1 dB HBM GEMM with staged S:     dZ.T @ stage(S_cpu)
```

No loops over tokens, rows, heads, KV groups, LoRA rank chunks, or row windows
are allowed in v1. Padding is whole-tensor padding only.

## Required Validation

Correctness:

```text
p == 0:
  forward, dX, dA, dB match current AsymLoRALinear path

0 < p < 1:
  forward, dX, dA, dB match a masked reference using the saved CPU mask

Full attention:
  q/k/v/o LoRA grads and input grads match current model path within bf16 tolerance
```

Memory and launch audit:

```text
manager CPU-owned bytes and HBM staged bytes by tag
saved HBM tensor tags before and after activation offload
q/k/v source sharing hits and duplicate-source fallback bytes
base/LoRA-A/dA AsymGEMM call counts by projection
LoRA-B/dS/dX-LoRA/dB HBM GEMM call counts by projection
all GEMM input/output shapes
peak allocated HBM and peak reserved HBM
step/profile timing, reported but not threshold-gated
```

A successful v1 artifact may still show attention-core saved tensors under the
attention component. Do not claim those are removed unless a later scoped-core
offload stage proves it separately.

## Production Acceptance

This math is useful only if it lowers training HBM by a meaningful amount
without hiding the cost in launch overhead or temporary workspace.

Reject an implementation if any of these are true:

```text
peak allocated/reserved HBM is unchanged within measurement noise
the target LF profile peak drops by less than both 5% and 1 GiB, unless a
  smaller artifact proves the exact projected large-model byte savings
latency increases without a meaningful HBM reduction
target LF step time exceeds 1.25x baseline unless marked investigation-only
AsymGEMM/GEMM counts exceed the launch contract above
the memory drop is offset by attention:temporary_workspace or unattributed peak
CPU AdamW no longer sees CUDA LoRA compute parameters or fails to update them
```

The implementation must record enough counters to enforce these gates: exact
AsymGEMM/GEMM counts, GEMM shapes, CPU-owned bytes, HBM staged bytes, peak HBM,
step time, and CPU AdamW update health.
