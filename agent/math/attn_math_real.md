# Dense Attention LoRA Activation-Offload Math (Faithful to Current Code)

This file describes what the attention projections **actually do** in the
production activation-offload configuration. It supersedes the earlier draft of
this file, which audited the wrong code path (`AsymLoRALinear` /
`_AsymLoRALinearWeightOffloadFunction` in `asym_gemm/training/lora.py`) and
wrongly claimed attention activation offload is "not implemented". Attention
activation offload **is** fully implemented and **is** what runs in production.

The real engine is `AsymActivationOffloadLoRALinear` /
`_AsymActivationOffloadLoRALinearFunction` in
`asym_gemm/training/attention_activation_offload.py`, plus three cooperating
nets installed around the attention module (a generic saved-tensor offload
wrapper, SDPA recompute, and decoder-layer glue-GC). The real code matches the
plan in `attn_math.md` very closely (CPU-resident X shared across q/k/v, CPU-left
`@^L` LoRA-A, S saved on CPU, dA via CPU-right `@^R`, base via `@^R`); the
sections below note every place it differs.

## Production configuration this doc describes

```text
LF profile : asym_cpuadamwds | norecompute | ligerloss1 ; none|true|true|false|true|true
backend    : asym            LoRA target : all     LORA_PARAMS "0.00|64|128|all"
                                                   => lora_dropout=0.0, r=64, alpha=128
                                                   => scale = alpha/r = 128/64 = 2.0
LoRA weight offload + grad offload : true   (A_*/B_* CPU-home, gathered to HBM)
```

The policy tuple `policy|expert_act|attn_act|layer_act|layer_gc|sdpa_recompute`
= `none | true | true | false | true | true` maps to these env gates (production
value in parentheses):

```text
expert_act     ASYMM_EXPERT_ACT_OFFLOAD     (true)   # routed-expert engine, not attention
attn_act       ASYMM_ATTN_ACT_OFFLOAD       (true)   # THIS doc: q/k/v/o projection offload
layer_act      ASYMM_LAYER_ACT_OFFLOAD      (false)  # whole-decoder-layer offload: OFF
layer_gc       ASYMM_LAYER_GC               (true)   # decoder-layer glue-GC wrapper
sdpa_recompute ASYMM_ATTN_SDPA_RECOMPUTE    (true)   # checkpoint the SDPA math
```

Code provenance: gate readers `_attention_act_offload_enabled`
(`lf.py:1306`), `_layer_act_offload_enabled` (`lf.py:1312`),
`_layer_glue_gc_enabled` (`lf.py:1318`); `sdpa_recompute._enabled`
(`sdpa_recompute.py:13`). `apply_lf_asym_lora` reads them at
`lf.py:1719-1724`. `none` policy is required for both glue-GC and layer-act
(`lf.py:1733-1738`).

## The four cooperating mechanisms

```text
1. AsymActivationOffloadLoRALinear   q_proj/k_proj/v_proj/o_proj become this.
   (curated projection Function)     Offloads its own projection input (U) and
                                     low-rank S to pinned CPU; base + dA + LoRA-A
                                     read CPU operands via AsymGEMM. It does NOT
                                     use saved_tensors_hooks; it owns CPU handles
                                     directly via a per-call ActivationOffloadManager.

2. AttentionSavedTensorOffloadWrapper Wraps the whole self_attn forward in
   (generic saved-tensor net)        saved_tensors_hooks. Catches every OTHER big
                                     CUDA saved tensor the curated Function does not
                                     manage (post-RoPE q/k, q_norm/k_norm outputs,
                                     the q/k/v the SDPA checkpoint retains as
                                     recompute inputs) and offloads it to CPU.

3. install_sdpa_recompute            Registers attention interface
   (SDPA recompute)                  "asym_sdpa_recompute" that runs the SDPA math
                                     under torch.utils.checkpoint. SDPA-internal
                                     saved state (softmax LSE, etc.) is RECOMPUTED
                                     in backward, not saved or offloaded.

4. DecoderLayerGlueGCWrapper         Wraps each decoder layer. Recomputes both
   (glue-GC, offload_mode="custom")  RMSNorms (checkpoint) and runs the whole layer
                                     body under a custom DecoderSavedTensorOffload
                                     pack/unpack net for residual/boundary tensors.
                                     The attention module runs INSIDE this wrapper.
```

Provenance: `AsymActivationOffloadLoRALinear` (`attention_activation_offload.py:748`),
its Function (`:560` fwd, `:655` bwd); `AttentionSavedTensorOffloadWrapper`
(`:187`), installed by `_wrap_attention_saved_tensor_offload_modules`
(`lf.py:1514`, called at `lf.py:2251-2256`); `install_sdpa_recompute`
(`sdpa_recompute.py:39`, called at `lf.py:1539`); `DecoderLayerGlueGCWrapper`
(`decoder_layer_glue_gc.py:133`), installed by
`_wrap_decoder_layer_glue_gc_modules` (`lf.py:1491`, called at `lf.py:2305`).

## Notation

```text
@   = GEMM (both operands HBM)
@^L = AsymGEMM with a CPU LEFT operand   (left CPU, right HBM)
@^R = AsymGEMM with a CPU RIGHT operand  (left HBM, right CPU)

offload(Z)     = copy HBM tensor to a pinned CPU buffer and keep the CPU owner
                 (ActivationOffloadManager.offload / acquire_source)
stage(h)       = copy a saved CPU handle back to HBM (ActivationOffloadManager.stage)
gather(A,B)    = gather_lora_weights(): stage CPU-home LoRA weights to HBM (group-level,
                 both A and B at once); no-op when weight offload is disabled
release(...)   = listed CPU handles / staged HBM tensors / staged weights are freed
pad_rows(Z,N)  = append zero rows so dim-0 == N (CPU: _pad_cpu_rows_to, HBM: _pad_hbm_rows_to)
align_up(n,a)  = smallest multiple of a >= n

CPU tensors have suffix _cpu.  Tensors without _cpu are HBM tensors.
Save(CPU)  = kept across the autograd boundary as a pinned-CPU handle on ctx (NOT on HBM).
Temp       = HBM temporary, released after last use.
Grad       = trainable LoRA gradient (dA_* / dB_*).
```

`Z.T` denotes GEMM orientation only, not a materialized transpose, except where
the code explicitly materializes `dS.t().contiguous()` for dA (called out below).

`U @^R W_cpu.T` is `asym_bf16_cpu_right_matmul(U, W_cpu, transpose_b=False)`,
computing `U @ W_cpu.T` with the **pinned CPU** weight as the right operand
(`frozen_linear.py:1097`). `dZ @^R W_cpu` is the same call with
`transpose_b=True`, computing `dZ @ W_cpu` (no transpose of the CPU operand).
`U_cpu @^L A.T` is `_dense_lora_a_cpu_left` → `grouped_expert_lora_cpu_left`, a
single-group CPU-left grouped AsymGEMM with the **pinned CPU** activation as the
left operand and the HBM LoRA-A weight as the right operand
(`attention_activation_offload.py:514`, `cpu_left.py:181`).

## Shapes and parameters

```text
B = batch     T = seq len     M = B*T
H = hidden    Hq/Hkv = q/kv heads     Dh = head_dim
Dq = Hq*Dh    Dkv = Hkv*Dh     r = LoRA rank     scale = alpha/r

X        [M,H]     hidden_states flattened over batch and sequence (post input_layernorm)
Q        [M,Dq]    K,V [M,Dkv]     AttnOut [M,Dq]     Y [M,H]
```

```text
# Frozen dense projection weights: pinned CPU HostWeight (bf16), never on HBM.
W_q_cpu [Dq,H]   W_k_cpu [Dkv,H]   W_v_cpu [Dkv,H]   W_o_cpu [H,Dq]

# Trainable LoRA weights: CPU-home (weight offload), gathered to HBM around their GEMMs.
A_q [r,H]   B_q [Dq,r]      A_k [r,H]   B_k [Dkv,r]
A_v [r,H]   B_v [Dkv,r]     A_o [r,Dq]  B_o [H,r]
```

Branch map (`lf.py` leaf names → role string `name.rsplit(".",1)[-1]`,
`lf.py:1207`):

```text
branch  input U   output Z  frozen W_cpu  LoRA A  LoRA B  in   out   context
q_proj  X         Q         W_q_cpu       A_q     B_q     H    Dq    shared (q/k/v)
k_proj  X         K         W_k_cpu       A_k     B_k     H    Dkv   shared (q/k/v)
v_proj  X         V         W_v_cpu       A_v     B_v     H    Dkv   shared (q/k/v)
o_proj  AttnOut   Y         W_o_cpu       A_o     B_o     Dq   H     None (own)
```

Qwen3-32B dense numbers used throughout (head_dim 128):

```text
H=5120   Dq=Hq*Dh=64*128=8192   Dkv=Hkv*Dh=8*128=1024   r=64   scale=2.0   bf16=2B
```

LoRA init is PEFT-style (`init_lora_weights="peft"`, `lf.py:1205`): A kaiming,
B zero (`lora.py:107-109`), so the delta is zero at step 0. Qwen3 q/k/v/o
projections have **no bias** (`base + bias_cpu` is applied only if present,
`attention_activation_offload.py:609`).

## AsymGEMM alignment contract (derived from the real validators)

For one branch with source dim `in` and output dim `out`, BF16, backend "asym":

```text
base forward   U @^R W_cpu.T   transpose_b=False  needs out%8==0, in%8==0
base dx        dZ @^R W_cpu    transpose_b=True   needs in%8==0,  out%8==0, out%64==0
LoRA-A fwd     U_cpu @^L A.T   CPU-left grouped    needs r%8==0,   in%8==0, SM100 (arch==10)
dA             dS_T @^R U_cpu  transpose_b=True   needs in%8==0,  M_grad%8==0, M_grad%64==0
```

For base dx the kernel's `k`-dim is `out` and `transpose_b` requires `k%64==0`,
hence `out%64==0` (`frozen_linear.py:383-386`). For dA the kernel's `k`-dim is
the row count `M_grad`, and the `transpose_b` `k%64` rule is exactly why M is
padded to `align_up(M,64)` before dA (`attention_activation_offload.py:711`).
The CPU-left LoRA-A path needs `n=r`, `k=in` 8-aligned and **SM100 only**
(`cpu_left.py:73,95`). Under backend "asym" an unsupported shape raises loudly
(`asym_bf16_cpu_right_matmul`, `frozen_linear.py:1130-1141`); `lf.py:1170,1187`
diverts shapes that would fall back to torch to `AsymLoRALinear` instead, so they
never reach this engine.

Qwen3-32B check: H=5120 (`%64=0`), Dq=8192 (`%64=0`), Dkv=1024 (`%64=0`),
r=64 (`%8=0`) — every projection satisfies the full asym contract; no fallback.

## Forward schedule

q/k/v share **one** CPU copy of X via the per-attention-parent
`AttentionActivationOffloadContext`; o_proj has `attention_context=None` and
offloads its own input through a private `ActivationOffloadManager`. Each branch
initializes its output with the base CPU-right GEMM and accumulates the LoRA-B
GEMM into it. LoRA weights are gathered to HBM at Function entry and released at
exit (`_weight_offload_release_after_forward=True` for standalone attention
leaves, `attention_activation_offload.py:914`; the leaf also carries coordinator
forward pre/post gather/release hooks, `weight_offload.py:474-477`).

```text
X = hidden_states_flat                                  # [M,H] HBM bf16

# ============ q projection  (role=q_proj, context = parent ctx) ============
gather(A_q, B_q)                                        # CPU-home -> HBM
U_q   = X.reshape(M,H).contiguous().to(bf16)            # [M,H] HBM (no-op cast if X already bf16)
Q     = U_q @^R W_q_cpu.T                               # [M,Dq] HBM live      tag q_proj.base_forward
X_cpu = acquire_source(key=X, payload=U_q, "q_proj")    # offload U_q -> pinned CPU [M,H]; cache[key(X)]=handle; refcount 1
S_q   = X_cpu @^L A_q.T                                 # [M,r]  HBM           tag q_proj.lora_a_forward
Q    += scale * (S_q @ B_q.T)                           # [M,Dq] HBM accumulate (plain GEMM)  tag q_proj.lora_b_forward
S_q_cpu = offload(S_q)                                  # [M,r] pinned CPU (s_handle)
release(A_q, B_q)                                       # free HBM LoRA copies
# Save(CPU): X_cpu (shared handle), S_q_cpu.  NO HBM activation is saved by this Function.

# ============ k projection  (role=k_proj, context = parent ctx) ============
gather(A_k, B_k)
U_k   = X.reshape(M,H).contiguous().to(bf16)
K     = U_k @^R W_k_cpu.T                               # [M,Dkv] HBM          tag k_proj.base_forward
X_cpu = acquire_source(key=X, payload=U_k, "k_proj")    # cache HIT on key(X): REUSE q's X_cpu; U_k NOT offloaded; refcount 2
S_k   = X_cpu @^L A_k.T                                 # [M,r] HBM            tag k_proj.lora_a_forward
K    += scale * (S_k @ B_k.T)                           # [M,Dkv] HBM          tag k_proj.lora_b_forward
S_k_cpu = offload(S_k)
release(A_k, B_k)

# ============ v projection  (role=v_proj, context = parent ctx) ============
gather(A_v, B_v)
U_v   = X.reshape(M,H).contiguous().to(bf16)
V     = U_v @^R W_v_cpu.T                               # [M,Dkv] HBM          tag v_proj.base_forward
X_cpu = acquire_source(key=X, payload=U_v, "v_proj")    # cache HIT: REUSE X_cpu; refcount 3; then cache cleared after v_proj
S_v   = X_cpu @^L A_v.T                                 # [M,r] HBM            tag v_proj.lora_a_forward
V    += scale * (S_v @ B_v.T)                           # [M,Dkv] HBM          tag v_proj.lora_b_forward
S_v_cpu = offload(S_v)
release(A_v, B_v)

# ============ attention prepare/core (normal autograd; see wrappers below) ====
Q_prep,K_prep,V_prep = RoPE/q_norm/k_norm(Q,K,V)
AttnOut = attention_core(Q_prep,K_prep,V_prep)          # [M,Dq] HBM  (asym_sdpa_recompute: checkpointed)

# ============ o projection  (role=o_proj, context = None) ============
gather(A_o, B_o)
U_o   = AttnOut.reshape(M,Dq).contiguous().to(bf16)     # [M,Dq] HBM
Y     = U_o @^R W_o_cpu.T                               # [M,H] HBM live       tag o_proj.base_forward
AttnOut_cpu = offload(U_o)                              # [M,Dq] pinned CPU (own manager; NOT shared)
S_o   = AttnOut_cpu @^L A_o.T                           # [M,r] HBM            tag o_proj.lora_a_forward
Y    += scale * (S_o @ B_o.T)                           # [M,H] HBM            tag o_proj.lora_b_forward
S_o_cpu = offload(S_o)
release(A_o, B_o)
```

Operand placement summary (forward): the only **CPU** operands are `W_*_cpu`
(base, via `@^R`) and `X_cpu`/`AttnOut_cpu` (LoRA-A, via `@^L`). `A_*`, `B_*`,
`S_*` and both products are HBM. The wide projection inputs (`U_*`) and `S_*`
are immediately offloaded to pinned CPU and held only as CPU handles across the
boundary; the curated Function saves **nothing** on HBM
(`ctx.save_for_backward()` is empty under weight offload,
`attention_activation_offload.py:632-635`).

Code provenance: cast `flat_lora = flat.to(lora_dtype).contiguous()` (`:595`);
base `asym_bf16_cpu_right_matmul(..., tag=f"{role}.base_forward")` (`:599`);
o offloads U via local manager (`:614-615`), q/k/v share via `acquire_source`
(`:616-618`); `_dense_lora_a_cpu_left(u_handle.tensor, a, ...)` (`:619`);
`delta = s @ b.t()`, `out = base + (delta*scaling)` (`:627-628`);
`s_handle = manager.offload(s.contiguous(), ...)` (`:629`). Sharing: cache key is
`_source_key(x)` on the original X storage (`:463`, `:402`), payload offloaded is
`flat_lora` (`:470`); cache cleared after `v_proj` (`:477-479`); refcounted
`_SharedActivationSource` (`:417-440`).

## Backward schedule

Backward runs o first (it is nearest the loss), then attention-core autograd
produces dQ/dK/dV, then q/k/v. Each Function re-gathers its LoRA weights at entry
(`:663-665`); there is no per-Function release in backward (the coordinator
reclaims the HBM copy). Within one branch the order is fixed:
**dS → base dx → LoRA input grad (dX) → dA → dB**
(`attention_activation_offload.py:686-735`).

```text
M_grad = align_up(M, 64)

# ============ o projection backward (first) ============
dY  = grad wrt Y  -> bf16 contiguous d_y                 # [M,H]
gather(A_o, B_o)
dS_o = scale * (dY @ B_o)                                # [M,r] Temp HBM          tag o_proj.dS
dAttnOut = dY @^R W_o_cpu                                # [M,Dq] HBM base dx      tag o_proj.base_dx   (transpose_b=True)
dAttnOut += dS_o @ A_o                                   # [M,Dq] HBM LoRA dX      tag o_proj.lora_input_grad
# --- dA_o (CPU-right, padded) ---
U_pad   = pad_rows(AttnOut_cpu, M_grad)                  # CPU [M_grad,Dq]
dS_o_T  = contiguous(pad_rows(dS_o, M_grad).T)           # HBM [r,M_grad]
dA_o    = dS_o_T @^R U_pad                               # [r,Dq] Grad             tag o_proj.dA  (transpose_b=True; k=M_grad)
# --- dB_o (stage S to HBM, then HBM GEMM) ---
S_o_hbm = stage(S_o_cpu)                                 # [M,r] HBM
dB_o    = scale * (dY.T @ S_o_hbm)                       # [H,r] Grad              tag o_proj.dB
release(S_o_hbm, S_o_cpu, AttnOut_cpu)                   # o owns its U -> release_cpu(u_handle)
# dAttnOut feeds attention-core backward -> dQ [M,Dq], dK [M,Dkv], dV [M,Dkv]

# ============ q projection backward ============
gather(A_q, B_q)
dS_q = scale * (dQ @ B_q)                                # [M,r] Temp HBM          tag q_proj.dS
dX_q = dQ @^R W_q_cpu                                    # [M,H] HBM base dx       tag q_proj.base_dx
dX_q += dS_q @ A_q                                       # [M,H] HBM LoRA dX       tag q_proj.lora_input_grad
U_pad  = pad_rows(X_cpu, M_grad)                         # CPU [M_grad,H]  (shared X_cpu)
dS_q_T = contiguous(pad_rows(dS_q, M_grad).T)            # HBM [r,M_grad]
dA_q   = dS_q_T @^R U_pad                                # [r,H] Grad              tag q_proj.dA
S_q_hbm = stage(S_q_cpu)                                 # [M,r] HBM
dB_q   = scale * (dQ.T @ S_q_hbm)                        # [Dq,r] Grad             tag q_proj.dB
release(S_q_hbm, S_q_cpu); shared_source.release()       # refcount 3 -> 2 (X_cpu kept alive)
# dX_q is accumulated by autograd into X.grad

# ============ k projection backward (same shape pattern; in=H,out=Dkv) ============
gather(A_k, B_k)
dS_k = scale * (dK @ B_k);  dX_k = dK @^R W_k_cpu;  dX_k += dS_k @ A_k
dA_k = contiguous(pad_rows(dS_k,M_grad).T) @^R pad_rows(X_cpu,M_grad)   # [r,H]
dB_k = scale * (dK.T @ stage(S_k_cpu))                                  # [Dkv,r]
release(...); shared_source.release()                    # refcount 2 -> 1

# ============ v projection backward (in=H,out=Dkv) ============
gather(A_v, B_v)
dS_v = scale * (dV @ B_v);  dX_v = dV @^R W_v_cpu;  dX_v += dS_v @ A_v
dA_v = contiguous(pad_rows(dS_v,M_grad).T) @^R pad_rows(X_cpu,M_grad)   # [r,H]
dB_v = scale * (dV.T @ stage(S_v_cpu))                                  # [Dkv,r]
release(...); shared_source.release()                    # refcount 1 -> 0 -> X_cpu freed

# ============ input gradient ============
# X.grad = dX_q + dX_k + dX_v   (autograd accumulates the three branch grad_x)
```

Operand placement summary (backward): base dx is `@^R` (CPU `W`); dA is `@^R`
(CPU `U`, i.e. `X_cpu` for q/k/v or `AttnOut_cpu` for o, read straight from CPU
with no stage-to-HBM); dS, the LoRA-input-grad add, and dB are plain HBM GEMMs;
dB first stages `S_*_cpu` back to HBM. The shared `X_cpu` is held by all three
q/k/v autograd nodes via refcounting and freed only when the last of the three
backwards runs (`_SharedActivationSource.release`, `:430-440`).

Code provenance: `d_s = (d_y @ b)*scaling` (`:687-689`); base dx
`asym_bf16_cpu_right_matmul(d_y, W, transpose_b=True, phase="attn_act_base_dx")`
(`:692-702`); `d_u += d_s @ a` (`:703-705`); `grad_x = d_u.to(input_dtype)`
(`:706`); dA pad+transpose `_pad_cpu_rows_to`/`_pad_hbm_rows_to`/`d_s_rows.t()`
and `asym_bf16_cpu_right_matmul(d_s_t, u_source, transpose_b=True,
phase="attn_act_dA")` (`:711-728`); dB `s_stage = manager.stage(s_handle)` then
`grad_b = (d_y.t() @ s_stage)*scaling` (`:730-735`); release order in `finally`
(`:736-743`).

## AttentionSavedTensorOffloadWrapper (the generic net)

`install_attention_saved_tensor_offload` replaces the self_attn module forward
so the whole forward runs under `saved_tensors_hooks(_pack, _unpack)` when
`module.training and torch.is_grad_enabled()`
(`attention_activation_offload.py:236-241`, installed `lf.py:1535`). `_pack`
offloads a saved tensor to a pinned, **stride-preserving** CPU buffer
(`_empty_strided_cpu_like`, `:275`) and records a CUDA ready-event; `_unpack`
copies it back to HBM in backward (`:308-330`). A tensor is offloaded only if
**all** of (`_should_offload`, `:243-265`):

```text
tensor.device.type == "cuda"
tensor.dtype in {bf16, fp16, fp32}             (ASYM_ATTN_SAVED_TENSOR_OFFLOAD_DTYPES; default these three)
tensor.requires_grad                            (require_grad default True via ASYM_ATTN_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD)
nbytes >= 1 MiB                                 (ASYM_ATTN_SAVED_TENSOR_OFFLOAD_MIN_BYTES; default 1*1024**2)
NOT isinstance(tensor, nn.Parameter)            (skip real weights; keep leaf+grad *activations*, e.g. fla delta-net)
```

What it catches that the curated projection Function does **not**: the curated
Function saves no HBM tensors, so this wrapper is what removes the remaining
attention-glue HBM saves — the **post-RoPE / q_norm / k_norm q and k**, reshaped
head-layout views, and, with sdpa_recompute on, the **q/k/v that the SDPA
checkpoint retains as recompute inputs** (the code comment names this synergy
explicitly: "q/k/v are the recompute inputs (offloaded by attn_act)",
`sdpa_recompute.py:27`). The wrapper is the only place those wide post-prepare
tensors get moved to CPU; everything saved strictly inside the SDPA checkpoint is
recomputed, not offloaded.

## install_sdpa_recompute (recompute, not offload)

When `ASYMM_ATTN_SDPA_RECOMPUTE` is truthy, a HF attention interface
`asym_sdpa_recompute` is registered that wraps the base SDPA/FlashAttention
callable in `torch.utils.checkpoint.checkpoint(_run, q, k, v,
use_reentrant=False)` and points `text_config._attn_implementation` at it
(`sdpa_recompute.py:22-53`). Effect: the SDPA math's internal backward state
(softmax LSE / stats / any large intermediates) is **recomputed** during
backward rather than saved or offloaded; only q/k/v are retained as recompute
inputs (and those are offloaded by mechanism #2). Eval / no-grad is a zero-overhead
passthrough (`sdpa_recompute.py:25-26`).

## DecoderLayerGlueGCWrapper (glue-GC, offload_mode="custom")

Each Qwen3 decoder layer's forward is replaced (`lf.py:2305`,
`decoder_layer_glue_gc.py:252`). On a training, grad-enabled, non-`use_cache`,
bindable call it runs a **manual** layer forward (`_manual_forward`,
`decoder_layer_glue_gc.py:215-242`):

```text
residual = hidden_states
normed   = checkpoint(input_layernorm)(hidden_states)        # RMSNorm RECOMPUTED in backward
mixer    = self_attn(normed)                                 # attention module (nets #1/#2/#3 active here)
hidden   = residual + mixer
residual = hidden
normed   = checkpoint(post_attention_layernorm)(hidden)      # RMSNorm RECOMPUTED
hidden   = residual + mlp(normed)
```

The entire body runs under
`saved_tensors_hooks(DecoderSavedTensorOffloadWrapper._pack, ._unpack)` (the
`"custom"` branch, `decoder_layer_glue_gc.py:210-213`). That net offloads the
layer-level residual/boundary saved tensors (the `normed` feeding attention/MLP,
residual adds, MLP-side activations) under the same ≥1 MiB / allowed-dtype /
non-Parameter rule, but with `require_grad` default **False**
(`decoder_activation_offload.py:120-124,159-181`) — so it also offloads saved
tensors that do not require grad. Because the attention module runs nested inside
this context, its own `saved_tensors_hooks` (net #2) and the SDPA checkpoint
(net #3) take precedence for tensors produced inside them (innermost hooks win);
glue-GC covers what is left at the decoder-layer scope.

## What is offloaded to CPU / stays on HBM / recomputed

```text
Offloaded to pinned CPU (held as handles across backward):
  per q/k/v branch input U .................... ONE shared X_cpu [M,H]      (mechanism #1)
  o_proj input ................................ AttnOut_cpu [M,Dq]          (mechanism #1)
  per branch low-rank S ....................... S_q/S_k/S_v/S_o [M,r]       (mechanism #1)
  post-RoPE/q_norm/k_norm q,k and recompute-input q/k/v ... (mechanism #2)
  decoder-layer residual/boundary + MLP saves ............. (mechanism #4)

Stay CPU-resident the whole step (never on HBM as tensors):
  frozen base weights W_q/W_k/W_v/W_o_cpu       (read by @^R in fwd base, base dx, dA)
  CPU-home LoRA A_*/B_* at rest (gathered to HBM only around their GEMMs)

Live on HBM only transiently:
  Q/K/V/AttnOut/Y outputs; S_* products; dS/dX/dA/dB products
  staged LoRA A_*/B_* during their GEMMs; S_*_hbm staged for dB; pad buffers for dA

Recomputed in backward (not saved, not offloaded):
  both decoder RMSNorms (glue-GC checkpoint)
  the SDPA math internals (sdpa_recompute checkpoint)
```

## Launch contract (per projection branch)

```text
forward:
  1 base CPU-right AsymGEMM @^R   U @^R W_cpu.T               tag {role}.base_forward
  1 LoRA-A CPU-left AsymGEMM @^L  U_cpu @^L A.T               tag {role}.lora_a_forward
  1 LoRA-B HBM GEMM @             S @ B.T                     tag {role}.lora_b_forward

backward:
  1 dS HBM GEMM @                 scale * (dZ @ B)            tag {role}.dS
  1 base dx CPU-right AsymGEMM @^R dZ @^R W_cpu               tag {role}.base_dx
  1 LoRA input-grad HBM GEMM @    dS @ A                      tag {role}.lora_input_grad
  1 dA CPU-right AsymGEMM @^R     dS_T @^R U_cpu (M->M_grad)  tag {role}.dA
  1 dB HBM GEMM @                 scale * (dZ.T @ stage(S))   tag {role}.dB
```

Exactly **3 GEMMs forward / 5 GEMMs backward** per branch, with **no token / row /
head / rank loops** (single-group offsets `[0,M]`, experts `[0,-1]`,
`attention_activation_offload.py:96-110`; the only padding is whole-tensor
M-padding for dA). AsymGEMM (CPU operand) launches: forward `base_forward`(@^R) +
`lora_a_forward`(@^L); backward `base_dx`(@^R) + `dA`(@^R). The other four
(`lora_b_forward`, `dS`, `lora_input_grad`, `dB`) are plain HBM GEMMs, counted in
`stats.attn_act_hbm_gemm_calls_by_tag` (`frozen_linear.py:90-107`,
`attention_activation_offload.py:113`). Per branch: 4 AsymGEMM (2 fwd + 2 bwd
CPU-operand) and 4 plain HBM GEMM (1 fwd + 3 bwd). Across a 4-projection
attention block: 16 AsymGEMM (8 fwd + 8 bwd) and 16 plain HBM GEMM (4 fwd +
12 bwd), plus 4 forward S-offloads and 4 backward S-stages.

## Byte accounting (Qwen3-32B dense, bf16=2B, M=B*T)

CPU activation bytes the curated Function holds per attention layer (the big win
is one **shared** X_cpu for q/k/v instead of three):

```text
shared X_cpu (q/k/v)   M*H*2   = M*5120*2  = 10240*M B
o_proj AttnOut_cpu     M*Dq*2  = M*8192*2  = 16384*M B
4x low-rank S          4*M*r*2 = 4*M*64*2  =   512*M B
------------------------------------------------------
curated CPU total/layer = M*2*(H + Dq + 4r) = 27136*M B  (~26.5 KiB per token)
```

Transient HBM during backward (released after use): dA pads `dS` to
`[align_up(M,64), r]` and `S_*_hbm` stage is `[M,r]` (`M*64*2` B); LoRA weight
stages are tiny — A `[r,in]` and B `[out,r]` (e.g. q: `64*5120*2 + 8192*64*2`).
Versus the OLD (wrong) doc's HBM saved budget, which kept `x_lora [M,in]` **and**
`S [M,r]` on HBM **per branch with no q/k/v sharing**
(`3*(M*H*2) + (M*Dq*2) + 4*(M*r*2)` of HBM saves), this design moves all of it to
CPU and collapses 3 X copies to 1. The generic wrapper (#2) additionally pulls
the post-RoPE q/k and recompute-input q/k/v off HBM, and sdpa_recompute removes
the SDPA-internal HBM saves entirely.

## Differences from attn_math.md (the plan)

```text
plan attn_math.md                         real code (this doc)
-------------------------------------     ------------------------------------------
X_cpu = offload(X)                         X_cpu = offload(X.to(bf16)); the offloaded
                                           payload is flat_lora (the bf16 cast), but the
                                           share cache keys on the ORIGINAL X storage
                                           (_source_key(x)), so q/k/v dedup correctly.
single saved_tensors-hooks design          TWO separate nets: a curated projection
                                           Function that owns CPU handles directly (no
                                           saved_tensors_hooks), PLUS a generic
                                           AttentionSavedTensorOffloadWrapper around the
                                           whole self_attn. The plan does not mention the
                                           generic wrapper at all.
dropout D_b(...) / masks                   dropout is HARD-DISABLED: forward raises if
                                           lora_dropout_p != 0 (:589-590), and lf gating
                                           requires lora_dropout=0.0 (lf.py:1192-1193).
                                           No mask is saved; production p==0.
stage(A_q)/stage(B_q) per weight           gather is GROUP-level: gather_lora_weights()
                                           stages BOTH A and B once at Function entry;
                                           release_lora_weights() frees both at forward
                                           exit. Not per-GEMM.
LoRA weights not saved across boundary      under weight offload, ctx.save_for_backward()
                                           is empty; backward RE-GATHERS A,B from CPU home
                                           (:632-635,:663-665). Without weight offload it
                                           would save (a,b) instead.
M_grad pad for dA                          confirmed: align_up(M,64); pad U on CPU and dS
                                           on HBM, materialize dS.t().contiguous()
                                           (:711-728). r itself needs no row alignment.
"a later stage MAY test scoped              that scoped attention saved-tensors net is
 saved_tensors_hooks around the core"      SHIPPED and ON in production (mechanism #2),
                                           and is paired with sdpa_recompute (#3) and
                                           decoder glue-GC (#4).
```

## Differences from the previous attn_math_real.md (the wrong audit)

```text
old attn_math_real.md (WRONG)             corrected (this doc)
-------------------------------------     ------------------------------------------
"Activation offload NOT implemented"      It IS implemented and is the production path.
described AsymLoRALinear /                 the real engine is AsymActivationOffloadLoRALinear
 _AsymLoRALinearWeightOffloadFunction     / _AsymActivationOffloadLoRALinearFunction in
 (lora.py)                                attention_activation_offload.py.
X stays HBM; x_lora cast kept on HBM       X is offloaded to pinned CPU as X_cpu and SHARED
                                           across q/k/v; nothing is saved on HBM by the
                                           Function.
S_q saved on HBM (saved_tensors)           S_* offloaded to CPU; staged back to HBM only for
                                           dB, then released.
LoRA-A forward is an HBM GEMM              LoRA-A forward is a CPU-LEFT AsymGEMM @^L
                                           (U_cpu @^L A.T), single-group grouped kernel.
dA is an HBM GEMM dS.T @ x_lora            dA is a CPU-RIGHT AsymGEMM @^R dS_T @^R X_cpu,
                                           with M padded to align_up(M,64).
"all 5 backward operands HBM-resident"     2 of the 5 are AsymGEMM with a CPU operand
                                           (base dx @^R W_cpu, dA @^R U_cpu).
no mention of SDPA recompute / glue-GC /   all three additional nets are active in
 generic saved-tensor wrapper              production and remove the remaining attention
                                           HBM saves; documented above.
```
