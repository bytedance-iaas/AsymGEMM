# Dense Attention LoRA Activation-Offload Math

This file is the ground-truth math for attention-side activation offload. The
implementation plan in `agent/impls/attn_act_offload.md` must follow this file.

Scope:

```text
Models:
  Qwen3 / Qwen3-MoE text attention
  Llama4 text attention

Changed in v1:
  q/k/v/o projection base weights are CPU HostWeights
  q/k/v/o LoRA-A inputs and dA inputs are CPU-resident
  q/k/v/o low-rank S tensors are saved on CPU for dB

Unchanged in v1:
  q/k norm, qk_norm, RoPE/NoPE, temperature scaling
  masks, KV-cache semantics, attention dropout
  SDPA/FlashAttention/eager attention forward and backward
```

Do not split or reimplement attention core in v1. PyTorch/FlashAttention
autograd owns `attention_prepare + attention_core`. This design only changes the
projection leaves around that core.

## Notation

```text
@  = GEMM
@^L = AsymGEMM with a CPU left operand
@^R = AsymGEMM with a CPU right operand

stage(U)       = make CPU tensor or CPU-home logical LoRA weight U available as an HBM tensor
offload(Z)     = copy HBM tensor to CPU and save the CPU owner
contiguous(V)  = materialize a contiguous HBM tensor from view V
pad_rows(Z, N) = append zero rows so the first dimension is N
align_up(n, a) = smallest multiple of a greater than or equal to n
release(...)   = listed tensors, staged weights, or saved handles are no longer live

CPU tensors have suffix _cpu.
Tensors without _cpu are HBM tensors.
Staged LoRA weights have suffix _hbm.
Temp means HBM temporary, releasable after last use.
Grad means trainable LoRA gradient.
```

`Z.T` denotes GEMM orientation only. It is not a requirement to save a
transposed tensor across autograd. If an AsymGEMM kernel requires a contiguous
HBM left operand, materialize that branch-local low-rank view and release it
immediately.

When an operand is CPU-resident, the operator marks the CPU side. For example,
`U_cpu @^L V.T` has a CPU left operand, while `U @^R V_cpu` has a CPU right
operand.

For a saved CPU tensor, `stage(U_cpu)` copies it to HBM for immediate use. For a
LoRA weight, `stage(A_q)` / `stage(B_q)` are explicit weight-offload lifetime
points. If LoRA weight offload is enabled, the CPU-home trainable weight is
staged/backfetched to HBM and the returned `*_hbm` tensor is the GEMM operand.
If LoRA weight offload is disabled, `stage(...)` is a no-op view of the CUDA
parameter. In both cases, formulas use the staged `*_hbm` operand and release it
after its last use. The logical trainable parameter and optimizer state remain
owned by the optimizer/weight-offload coordinator.

`Y += GEMM(...)` means accumulate into an already-live output buffer. Use a
beta/addmm-style epilogue when the backend supports it. If a backend must
materialize the GEMM result, that result is a one-line temporary consumed by the
add and released immediately.

Activation AsymGEMM on saved CPU activations is bf16-only in v1. Do not use
fp8/fp4 quantized host-weight paths for arbitrary saved CPU activations.

## Shapes And Parameters

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

The attention block has multiple parameter groups:

```text
# Frozen dense projection weights, CPU HostWeight
W_q_cpu  [Dq,H]
W_k_cpu  [Dkv,H]
W_v_cpu  [Dkv,H]
W_o_cpu  [H,Dq]

# Trainable LoRA weights, CPU-home when weight offload is enabled;
# staged HBM operands are named A_*_hbm / B_*_hbm in the schedule.
A_q [r,H]       B_q [Dq,r]
A_k [r,H]       B_k [Dkv,r]
A_v [r,H]       B_v [Dkv,r]
A_o [r,Dq]      B_o [H,r]

# Optional frozen projection bias, if present
bias_q [Dq]     bias_k [Dkv]     bias_v [Dkv]     bias_o [H]

# Optional small prepare/norm parameters, model-specific shape
gamma_q         q_norm weight when present
gamma_k         k_norm weight when present
gamma_qk        Llama4 qk_norm weight when present; may be stateless

# Non-trainable or backend-owned prepare/core data
RoPE/NoPE metadata, cos/sin/cache buffers, temperature constants
attention masks, causal masks, attention-dropout RNG/state
SDPA/FlashAttention internal saved state
```

Residency:

```text
W_q_cpu/W_k_cpu/W_v_cpu/W_o_cpu stay CPU-resident.
A_*/B_* are logical trainable LoRA parameters. With LoRA weight offload enabled,
their home storage is CPU-resident/pinned and they are staged to HBM only around
the GEMMs that consume them. Without LoRA weight offload, the same `stage`
points are no-ops over the CUDA parameter storage.
Projection bias, when present, is a small vector add folded into the base output
before LoRA accumulation.
CPU AdamW may own CPU fp32 masters/state for A_*/B_* and may offload gradients
from CUDA to CPU after backward accumulation. The math still treats dA_*/dB_* as
gradients of the logical trainable LoRA weights.
q_norm/k_norm/qk_norm are small vector ops, not AsymGEMM GEMMs, and are left to
normal model/PyTorch autograd in v1.
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
`D_b_bar(hbm_tensor)` should use a packed saved mask or fused mask kernel; if it
must stage or expand mask data, record those bytes and the launch cost.

`p == 1` is unsupported for this wrapper and must fail clearly.

## AsymGEMM Constraints

```text
For one projection branch with source dim `in` and output dim `out`:

Base forward:   CPU-right BF16, in % 8 == 0, out % 8 == 0
Base dx:        CPU-right BF16 transpose_b, in % 8 == 0, out % 64 == 0
LoRA-A forward: CPU-left BF16, SM100, in % 8 == 0, r % 8 == 0
dA:             CPU-right BF16 transpose_b, in % 8 == 0, M_grad % 64 == 0
```

Under `backend == "asym"`, unsupported shapes fail loudly. Test/debug torch
fallbacks must be explicit and recorded.

Dense attention LoRA-A uses the CPU-left grouped kernel as one logical group:
`offsets=[0,M]`, `experts=[0,-1]`, and `A` is viewed as `[1,r,in]`. The helper
pads CPU rows internally and returns the unpadded `[M,r]` result.

All AsymGEMM CPU operands are pinned contiguous BF16 tensors. HBM operands passed
to AsymGEMM are contiguous when required by the kernel contract.

Only CPU handles, optional dropout masks, scalar metadata, original shape/dtype
metadata, and LoRA A/B tensor references are saved across projection autograd
boundaries. Do not save the wide HBM source activations, dropped activations,
`S`, or transposed low-rank tensors.

## Attention Prepare/Core

The projection wrapper returns flat q/k/v tensors. The model reshapes and
prepares them before calling the selected attention backend:

```text
Q_heads = view(Q, [B,T,Hq,Dh])
K_heads = view(K, [B,T,Hkv,Dh])
V_heads = view(V, [B,T,Hkv,Dh])

# Qwen3 text attention, conceptually
Q_prep = RoPE(q_norm(Q_heads; gamma_q))
K_prep = RoPE(k_norm(K_heads; gamma_k))
V_prep = V_heads

# Llama4 text attention, conceptually
Q_prep, K_prep = RoPE/NoPE_prepare(Q_heads, K_heads)
Q_prep, K_prep = optional_qk_norm(Q_prep, K_prep; gamma_qk)
Q_prep, K_prep = optional_nope_temperature(Q_prep, K_prep)
V_prep = V_heads
```

The attention core is conceptually:

```text
K_gqa, V_gqa = logical_gqa_broadcast(K_prep, V_prep, Hq, Hkv)
Scores = (Q_prep @ K_gqa.T) * attention_scale + mask
P = softmax(Scores)
Context = P @ V_gqa
AttnOut = flatten_heads(Context)                         # [M,Dq]
```

In real execution, `attention_core` may be SDPA, FlashAttention, or eager
attention. Do not manually split it or call a custom attention-core backward in
v1.

## Forward

This is a lifetime schedule, not just algebra. The q/k/v final tensors must be
live together by the time attention starts, but their frozen base temporaries do
not need to be live together. Each branch initializes its final output with the
base CPU-right GEMM, accumulates the LoRA-B GEMM into that same output, then
releases branch-local temporaries before the next branch.

```text
X = hidden_states_flat                                      # [M,H] HBM

# ---------------- shared q/k/v source ----------------

X_cpu = offload(X)                                          # [M,H] CPU


# ---------------- q projection ----------------

Q = X @^R W_q_cpu.T                                         # [M,Dq] HBM live
X_q_cpu = D_q(X_cpu)                                        # [M,H] CPU
A_q_hbm = stage(A_q)                                 # [r,H] HBM staged
B_q_hbm = stage(B_q)                                 # [Dq,r] HBM staged
S_q = X_q_cpu @^L A_q_hbm.T                                 # [M,r] HBM
Q += scale * (S_q @ B_q_hbm.T)                              # consume delta now
S_q_cpu = offload(S_q)                                      # [M,r] CPU, save for dB_q
release(S_q, A_q_hbm, B_q_hbm, X_q_cpu_if_materialized)


# ---------------- k projection ----------------

K = X @^R W_k_cpu.T                                         # [M,Dkv] HBM live
X_k_cpu = D_k(X_cpu)                                        # [M,H] CPU
A_k_hbm = stage(A_k)                                 # [r,H] HBM staged
B_k_hbm = stage(B_k)                                 # [Dkv,r] HBM staged
S_k = X_k_cpu @^L A_k_hbm.T                                 # [M,r] HBM
K += scale * (S_k @ B_k_hbm.T)                              # consume delta now
S_k_cpu = offload(S_k)                                      # [M,r] CPU, save for dB_k
release(S_k, A_k_hbm, B_k_hbm, X_k_cpu_if_materialized)


# ---------------- v projection ----------------

V = X @^R W_v_cpu.T                                         # [M,Dkv] HBM live
X_v_cpu = D_v(X_cpu)                                        # [M,H] CPU
A_v_hbm = stage(A_v)                                 # [r,H] HBM staged
B_v_hbm = stage(B_v)                                 # [Dkv,r] HBM staged
S_v = X_v_cpu @^L A_v_hbm.T                                 # [M,r] HBM
V += scale * (S_v @ B_v_hbm.T)                              # consume delta now
S_v_cpu = offload(S_v)                                      # [M,r] CPU, save for dB_v
release(S_v, A_v_hbm, B_v_hbm, X_v_cpu_if_materialized)


# ---------------- attention prepare/core ----------------

Q_attn, K_attn, V_attn = attention_prepare(Q, K, V)          # HBM, normal autograd
AttnOut = attention_core(Q_attn, K_attn, V_attn)             # [M,Dq] HBM


# ---------------- o projection ----------------

AttnOut_cpu = offload(AttnOut)                               # [M,Dq] CPU
Y = AttnOut @^R W_o_cpu.T                                    # [M,H] HBM live
AttnOut_o_cpu = D_o(AttnOut_cpu)                             # [M,Dq] CPU
A_o_hbm = stage(A_o)                                  # [r,Dq] HBM staged
B_o_hbm = stage(B_o)                                  # [H,r] HBM staged
S_o = AttnOut_o_cpu @^L A_o_hbm.T                            # [M,r] HBM
Y += scale * (S_o @ B_o_hbm.T)                               # consume delta now
S_o_cpu = offload(S_o)                                       # [M,r] CPU, save for dB_o
release(S_o, A_o_hbm, B_o_hbm, AttnOut_o_cpu_if_materialized)
```

## Backward

```text
dY = dL/dY                                                  # [M,H] HBM
M_grad = align_up(M, 64)

# ---------------- o projection ----------------

dAttnOut = dY @^R W_o_cpu                                   # [M,Dq] HBM live
B_o_hbm = stage(B_o)                                 # [H,r] HBM staged
A_o_hbm = stage(A_o)                                 # [r,Dq] HBM staged
dS_o = scale * (dY @ B_o_hbm)                               # [M,r] Temp
dAttnOut += D_o_bar(dS_o @ A_o_hbm)                         # consume dAttn delta
release(A_o_hbm, B_o_hbm)

AttnOut_o_cpu = D_o(AttnOut_cpu)                            # [M,Dq] CPU
AttnOut_grad_cpu = pad_rows(AttnOut_o_cpu, M_grad)          # [M_grad,Dq] CPU
dS_o_grad = pad_rows(dS_o, M_grad)                          # [M_grad,r] HBM
dS_o_T = contiguous(dS_o_grad.T)                             # [r,M_grad] HBM
release(dS_o, dS_o_grad)
dA_o = dS_o_T @^R AttnOut_grad_cpu                          # [r,Dq] Grad
release(dS_o_T, AttnOut_grad_cpu, AttnOut_o_cpu_if_materialized, AttnOut_cpu)
S_o_stage = stage(S_o_cpu)                                  # [M,r] HBM
dB_o = scale * (dY.T @ S_o_stage)                           # [H,r] Grad
release(S_o_stage)
release(S_o_cpu)


# ---------------- attention prepare/core ----------------

dQ, dK, dV = autograd(attention_prepare + attention_core, dAttnOut)
# dQ [M,Dq], dK [M,Dkv], dV [M,Dkv]


# ---------------- q projection ----------------

dX = dQ @^R W_q_cpu                                         # [M,H] HBM
B_q_hbm = stage(B_q)                                 # [Dq,r] HBM staged
A_q_hbm = stage(A_q)                                 # [r,H] HBM staged
dS_q = scale * (dQ @ B_q_hbm)                               # [M,r] Temp
dX += D_q_bar(dS_q @ A_q_hbm)                               # consume q dX delta
release(A_q_hbm, B_q_hbm)

X_q_cpu = D_q(X_cpu)                                        # [M,H] CPU
X_q_grad_cpu = pad_rows(X_q_cpu, M_grad)                    # [M_grad,H] CPU
dS_q_grad = pad_rows(dS_q, M_grad)                          # [M_grad,r] HBM
dS_q_T = contiguous(dS_q_grad.T)                             # [r,M_grad] HBM
release(dS_q, dS_q_grad)
dA_q = dS_q_T @^R X_q_grad_cpu                              # [r,H] Grad
release(dS_q_T, X_q_grad_cpu, X_q_cpu_if_materialized)
S_q_stage = stage(S_q_cpu)                                  # [M,r] HBM
dB_q = scale * (dQ.T @ S_q_stage)                           # [Dq,r] Grad
release(S_q_stage)
release(S_q_cpu)


# ---------------- k projection ----------------

dX += dK @^R W_k_cpu                                        # base term; temp only if no beta path
B_k_hbm = stage(B_k)                                 # [Dkv,r] HBM staged
A_k_hbm = stage(A_k)                                 # [r,H] HBM staged
dS_k = scale * (dK @ B_k_hbm)                               # [M,r] Temp
dX += D_k_bar(dS_k @ A_k_hbm)                               # consume k dX delta
release(A_k_hbm, B_k_hbm)

X_k_cpu = D_k(X_cpu)                                        # [M,H] CPU
X_k_grad_cpu = pad_rows(X_k_cpu, M_grad)                    # [M_grad,H] CPU
dS_k_grad = pad_rows(dS_k, M_grad)                          # [M_grad,r] HBM
dS_k_T = contiguous(dS_k_grad.T)                             # [r,M_grad] HBM
release(dS_k, dS_k_grad)
dA_k = dS_k_T @^R X_k_grad_cpu                              # [r,H] Grad
release(dS_k_T, X_k_grad_cpu, X_k_cpu_if_materialized)
S_k_stage = stage(S_k_cpu)                                  # [M,r] HBM
dB_k = scale * (dK.T @ S_k_stage)                           # [Dkv,r] Grad
release(S_k_stage)
release(S_k_cpu)


# ---------------- v projection ----------------

dX += dV @^R W_v_cpu                                        # base term; temp only if no beta path
B_v_hbm = stage(B_v)                                 # [Dkv,r] HBM staged
A_v_hbm = stage(A_v)                                 # [r,H] HBM staged
dS_v = scale * (dV @ B_v_hbm)                               # [M,r] Temp
dX += D_v_bar(dS_v @ A_v_hbm)                               # consume v dX delta
release(A_v_hbm, B_v_hbm)

X_v_cpu = D_v(X_cpu)                                        # [M,H] CPU
X_v_grad_cpu = pad_rows(X_v_cpu, M_grad)                    # [M_grad,H] CPU
dS_v_grad = pad_rows(dS_v, M_grad)                          # [M_grad,r] HBM
dS_v_T = contiguous(dS_v_grad.T)                             # [r,M_grad] HBM
release(dS_v, dS_v_grad)
dA_v = dS_v_T @^R X_v_grad_cpu                              # [r,H] Grad
release(dS_v_T, X_v_grad_cpu, X_v_cpu_if_materialized, X_cpu)
S_v_stage = stage(S_v_cpu)                                  # [M,r] HBM
dB_v = scale * (dV.T @ S_v_stage)                           # [Dkv,r] Grad
release(S_v_stage)
release(S_v_cpu)


# ---------------- final input gradient ----------------

# dX = q/k/v base gradients + q/k/v LoRA gradients          # [M,H] HBM
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

Each wrapped projection has exactly these launches:

```text
forward pass:
  1 base CPU-right AsymGEMM:       U @^R W_cpu.T
  1 LoRA-A CPU-left AsymGEMM:      U_drop_cpu @^L stage(A).T
  1 LoRA-B HBM GEMM:               S @ stage(B).T

backward pass:
  1 base dx CPU-right AsymGEMM:    dZ @^R W_cpu
  1 dS HBM GEMM:                   dZ @ stage(B)
  1 LoRA input HBM GEMM:           dS @ stage(A)
  1 dA CPU-right AsymGEMM:         dS.T @^R U_drop_cpu
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
step/profile timing, checked against Production Acceptance
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
