# MLP Math (Faithful to Current Code)

Two distinct code paths produce the MLP gradient under AsymGEMM LoRA-SFT. Pick by model:

- **MoE expert engine** (`q3-30b-a3b` / `llama4` / `qwen3.5`): the routed experts run through
  `_ActivationOffloadQwen3ExpertFunction` in `asym_gemm/training/qwen3_moe.py`. Per-expert
  `M_g ≪ M` makes CPU compute (silu, `dA` via `@^R`) viable. → **Part A**.
- **Dense MLP** (`q3-32b` / `q2.5-72b` / `llama3.3-70b`): the HF `Qwen3MLP`/`LlamaMLP` does **not**
  touch the expert engine. Its `gate_proj`/`up_proj`/`down_proj` are wrapped leaf-by-leaf as
  `AsymLoRALinear`, and the whole decoder-layer body is offloaded by a **generic saved-tensor
  hook**. → **Part B**.

## Production config (the tuple this doc tracks)

`asym_cpuadamwds | norecompute | ligerloss1 ; none|true|true|false|true|true`
(policy|expert_act|attn_act|layer_act|layer_gc|sdpa_recompute), backend `asym`, LoRA **weight+grad
offload on**, `lora_dropout=0`, `r=64` (`LORA_PARAMS "0.00|64|128|all"`, `scale=alpha/r=128/64=2`).

Env that selects code paths below:

| env | production value | effect |
|---|---|---|
| `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD` | `hbm` | MoE LoRA-A forward = plain HBM `@` (code default is `cpu`: `qwen3_moe.py:235`) |
| `ASYM_OFFLOAD_ACT_RECOMPUTE` | `0` | MoE keeps `act_cpu` on CPU (no fwd-release/bwd-recompute); default `0` (`qwen3_moe.py:244`) |
| `ASYM_OFFLOAD_X_UNPACKED` | `0` | MoE offloads **packed** routed `X`, not the pre-route hidden; default `0` (`qwen3_moe.py:248`) |
| `ASYMM_EXPERT_SILU_BWD_GPU` | `1` | MoE SwiGLU backward on **GPU**; code default OFF (`qwen3_moe.py:921-927`) |
| `ASYMM_LAYER_GC` | `1` (`layer_gc=true`) | installs `DecoderLayerGlueGCWrapper` per decoder layer (`lf.py:2305-2306`) — drives **all** of Part B's offload |
| `ASYMM_DENSE_MLP_SURGICAL_OFFLOAD` | unset → **OFF** | surgical CPU dense-MLP engine NOT used (`lf.py:1981-1988`); dense MLP is Part B, not Part A |

---

# Part A — MoE expert engine (q3-30b-a3b / llama4 / qwen3.5)

Faithful to `_ActivationOffloadQwen3ExpertFunction` (`qwen3_moe.py:995`, forward `997-1214`, backward
`1217-1491`). Forward/backward are identical for the cpu/hbm LoRA-A modes because the saved CPU
tensors have the same values in both; **production uses `hbm`**. The shared engine also serves
Llama4 and Qwen3.5 (the ports are non-expert glue, not the experts).

The key difference from `mlp_math.md`: `dY` is never offloaded; `dS_*` and `dB_*` are plain HBM
GEMMs; only `dA_*` use `@^R` because the saved activations are CPU-resident.

## Notation

```text
@   = GEMM (both operands HBM)
@^L = AsymGEMM, CPU left operand
@^R = AsymGEMM, CPU right operand

E = experts; G = active groups; offsets[0:G+1]; experts[0:G]
M_g = offsets[g+1] - offsets[g]; M = sum_g M_g; e_g = experts[g]   # M = routed rows = tokens·top_k
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
# Production: x_unpacked=False (ASYM_OFFLOAD_X_UNPACKED=0) -> offload the PACKED rows (qwen3_moe.py:1035)
X_cpu = offload(X)                                      # [M,H] CPU — saved for dA_gate, dA_up
gate_up = pack_g(X_g @^R W_gate_up_cpu[e_g].T)         # [M,2I] HBM (layer.gate_up_base, qwen3_moe.py:1038)
gate, up = split(gate_up)                               # [M,I], [M,I] HBM views
```

### gate/up LoRA-A: mode = cpu  (ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu, code default — NOT production)

```text
A_gate_hbm = stage(A_gate)                             # [E,r,H] HBM staged
A_up_hbm   = stage(A_up)                               # [E,r,H] HBM staged
gate_low_rank_g = X_cpu,g @^L A_gate_hbm[e_g].T        # [M_g,r] HBM (cpu-left grouped)
up_low_rank_g   = X_cpu,g @^L A_up_hbm[e_g].T          # [M_g,r] HBM (cpu-left grouped)
release(A_gate_hbm, A_up_hbm)
gate_low_rank = pack_g(gate_low_rank_g)                 # [M,r] HBM
up_low_rank   = pack_g(up_low_rank_g)                   # [M,r] HBM
```

### gate/up LoRA-A: mode = hbm  (ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm — PRODUCTION, qwen3_moe.py:1061)

```text
A_gate_up_hbm = stage_concat(A_gate, A_up)             # [E,2r,H] cat(gate_A,up_A) (qwen3_moe.py:1062)
gate_up_low_rank_g = X_g @ A_gate_up_hbm[e_g].T        # [M_g,2r] HBM — plain @ on the LIVE packed X
release(A_gate_up_hbm)
gate_low_rank_g, up_low_rank_g = split_r(gate_up_low_rank_g)   # [M_g,r] HBM views
gate_low_rank, up_low_rank = split_r(pack_g(gate_up_low_rank_g))  # [M,r] HBM views
```

Note: in `hbm` mode the LoRA-A input is the **live HBM `packed`** (not `X_cpu`); `X_cpu` is offloaded
only to feed the backward `dA_gate/dA_up`.

### gate/up LoRA-B and activation (both modes continue identically)

```text
B_gate_hbm = stage(B_gate)                             # [E,I,r] HBM staged
B_up_hbm   = stage(B_up)                               # [E,I,r] HBM staged
gate += scale * pack_g(gate_low_rank_g @ B_gate_hbm[e_g].T)  # [M,I] accumulate LoRA delta
up   += scale * pack_g(up_low_rank_g   @ B_up_hbm[e_g].T)    # [M,I] accumulate LoRA delta
release(B_gate_hbm, B_up_hbm)

gate_cpu          = offload(gate)                       # [M,I] CPU — saved for silu_bwd
up_cpu            = offload(up)                         # [M,I] CPU — saved for silu_bwd
gate_low_rank_cpu = offload(gate_low_rank)              # [M,r] CPU — saved for dB_gate (S_gate)
up_low_rank_cpu   = offload(up_low_rank)                # [M,r] CPU — saved for dB_up   (S_up)
release(gate, up, gate_low_rank, up_low_rank, gate_up)

act_cpu = silu(gate_cpu) * up_cpu                      # [M,I] CPU (_activation_offload_cpu_silu_mul, qwen3_moe.py:885,1106)
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
down_low_rank_cpu = offload(down_low_rank)              # [M,r] CPU — saved for dB_down (S_down)
release(down_low_rank)

act_stage = stage(act_cpu)                             # [M,I] HBM transient
Y_down = pack_g(act_stage_g @^R W_down_cpu[e_g].T)    # [M,H] HBM base
release(act_stage)
Y_down += down_delta; release(down_delta)
```

### down projection: mode = hbm  (PRODUCTION, qwen3_moe.py:1138-1171)

```text
act_stage = stage(act_cpu)                             # [M,I] HBM, shared by LoRA-A and base

A_down_hbm    = stage(A_down)                          # [E,r,I] HBM staged
down_low_rank_g = act_stage_g @ A_down_hbm[e_g].T     # [M_g,r] HBM — plain @ on staged act
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

### Saved on ctx (qwen3_moe.py:1173-1208)

`x_cpu`, `gate_cpu`, `up_cpu`, `act_cpu` (`None` iff `act_recompute`, which is **off** in production so
`act_cpu` is kept), `gate_low_rank_cpu`/`up_low_rank_cpu`/`down_low_rank_cpu` (= S_gate/S_up/S_down).
Under **weight offload** (production), `save_for_backward` stores only `(offsets, experts)`
(`:1197`); the LoRA banks are **re-gathered** H2D in backward via `layer.gather_lora_weights()`
(`:1224`). The CPU-resident set at the engine's peak is therefore
`{X_cpu[M,H], gate_cpu[M,I], up_cpu[M,I], act_cpu[M,I], 3×S[M,r]}`.

## Backward

```text
dY = dL/dY_down                                        # [M,H] HBM — NOT offloaded

# LoRA banks A_*/B_* (and their *_hbm aliases below) are HBM-resident throughout backward:
# under weight offload they are re-gathered in ONE H2D at backward entry (gather_lora_weights,
# qwen3_moe.py:1224), so stage()/[e_g] on them is just the grouped index-select, not a CPU fetch.


# ---------------- down backward (qwen3_moe.py:1246-1294) ----------------

dY_lora = dY.to(lora_dtype)                            # [M,H] HBM cast

B_down_hbm  = stage(B_down)                            # [E,H,r] HBM staged
dS_down_g   = scale * (dY_lora,g @ B_down_hbm[e_g])   # [M_g,r] HBM
dS_down     = pack_g(dS_down_g)                        # [M,r] HBM
grad_down_lora_x = pack_g(dS_down_g @ A_down_hbm[e_g]) # [M,I] LoRA delta into dact (computed early)

S_down      = stage(down_low_rank_cpu)                  # [M,r] HBM transient
dB_down[e]  = scale * sum_{g:e_g=e} dY_lora,g.T @ S_down_g  # [E,H,r] Grad
release(S_down, down_low_rank_cpu, B_down_hbm)

dA_down[e]  = sum_{g:e_g=e} dS_down_g.T @^R act_cpu,g  # [E,r,I] Grad (CPU-right; act_cpu, no recompute)
release(act_cpu)

dact        = pack_g(dY_g @^R W_down_cpu[e_g])         # [M,I] HBM base dx (_grouped_base_dx)
dact       += grad_down_lora_x                          # [M,I] add LoRA delta


# ---------------- activation backward: GPU path (ASYMM_EXPERT_SILU_BWD_GPU=1 — PRODUCTION, qwen3_moe.py:930,1324-1327) ----------------

# grad_act stays resident on the GPU (NOT offloaded). _silu_backward_gpu stages gate/up back to HBM.
gate_stage    = stage(gate_cpu); up_stage = stage(up_cpu)   # [M,I] HBM transients (~200 MB each)
dgate_up      = empty([M,2I] HBM)                      # [M,2I] preallocated (qwen3_moe.py:954)
dgate_stage_g, dup_stage_g = split(dgate_up_g)         # [M_g,I], [M_g,I] HBM views
dup_stage    := dact * silu(gate_stage)                 # fills [:, I:]  (grad_up)
dgate_stage  := silu_backward(dact * up_stage, gate_stage)  # fills [:, :I] (grad_gate)
release(dact, gate_stage, up_stage, gate_cpu, up_cpu)


# ---------------- activation backward: CPU path (ASYMM_EXPERT_SILU_BWD_GPU unset — legacy, qwen3_moe.py:900,1328-1344) ----------------

dact_cpu      = offload(dact); release(dact)            # [M,I] CPU (extra D2H — the cost the GPU path avoids)
dgate_cpu     = silu_backward(dact_cpu * up_cpu, gate_cpu)  # [M,I] CPU
dup_cpu       = dact_cpu * silu(gate_cpu)               # [M,I] CPU
release(dact_cpu, gate_cpu, up_cpu)
dgate_up      = stage_concat(dgate_cpu, dup_cpu)        # [M,2I] HBM
release(dgate_cpu, dup_cpu)
dgate_stage_g, dup_stage_g = split(dgate_up_g)          # [M_g,I], [M_g,I] HBM views


# ---------------- gate LoRA backward (qwen3_moe.py:1346-1378) ----------------

B_gate_hbm  = stage(B_gate)                            # [E,I,r] HBM staged
dS_gate_g   = scale * (dgate_stage_g @ B_gate_hbm[e_g])  # [M_g,r] HBM
dS_gate     = pack_g(dS_gate_g)                         # [M,r] HBM
S_gate      = stage(gate_low_rank_cpu)                  # [M,r] HBM transient
dB_gate[e]  = scale * sum_{g:e_g=e} dgate_stage_g.T @ S_gate_g  # [E,I,r] Grad
release(S_gate, gate_low_rank_cpu, B_gate_hbm)


# ---------------- up LoRA backward (qwen3_moe.py:1380-1412) ----------------

B_up_hbm    = stage(B_up)                              # [E,I,r] HBM staged
dS_up_g     = scale * (dup_stage_g @ B_up_hbm[e_g])    # [M_g,r] HBM
dS_up       = pack_g(dS_up_g)                           # [M,r] HBM
S_up        = stage(up_low_rank_cpu)                    # [M,r] HBM transient
dB_up[e]    = scale * sum_{g:e_g=e} dup_stage_g.T @ S_up_g  # [E,I,r] Grad
release(S_up, up_low_rank_cpu, B_up_hbm)


# ---------------- gate/up dA: CPU-right (X_cpu is CPU-resident, qwen3_moe.py:1414-1427) ----------------

dA_gate[e]  = sum_{g:e_g=e} dS_gate_g.T @^R X_cpu,g   # [E,r,H] Grad (CPU-right)
dA_up[e]    = sum_{g:e_g=e} dS_up_g.T   @^R X_cpu,g   # [E,r,H] Grad (CPU-right)
release(X_cpu)


# ---------------- gate/up base backward (qwen3_moe.py:1429-1443) ----------------

dX = pack_g(dgate_up_g @^R W_gate_up_cpu[e_g])         # [M,H] HBM base dx
dX += pack_g(dS_gate_g @ A_gate_hbm[e_g])              # LoRA delta (grad_gate_lora_x)
dX += pack_g(dS_up_g   @ A_up_hbm[e_g])                # LoRA delta (grad_up_lora_x)
release(dS_gate, dS_up, dgate_up)
```

## Operand placement rule (Part A)

```text
dA_*  input-weight grad:   input activation is CPU-resident → CPU-right (@^R)
dB_*  output-weight grad:  low-rank S is CPU-resident → stage S to HBM, plain @
dS_*  chain rule:          gradient dY is HBM → plain @
base dx:                   frozen W is CPU-resident → CPU-right (@^R)
SwiGLU bwd (prod):         gate/up CPU-resident → stage to HBM, compute on GPU (@-free elementwise)
```

---

# Part B — Dense MLP (q3-32b / q2.5-72b / llama3.3-70b)

A dense model's MLP is the HF module `Qwen3MLP`/`LlamaMLP`:

```python
def forward(self, x):                 # transformers .../modeling_qwen3.py:81-82
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

It does **not** use `_ActivationOffloadQwen3ExpertFunction`. Two mechanisms produce its gradient:
per-projection `AsymLoRALinear`, and a generic per-layer saved-tensor offload hook.

## How the leaves are wrapped

`classify_lf_component` maps `gate_proj`/`up_proj`/`down_proj` under `.mlp.`/`.feed_forward.` to
component **`mlp_dense`** (`lf.py:592-593`) — **not** `attention`. In `_wrap_lf_linear_leaf`
(`lf.py:1150`) the attention-activation-offload branch is taken only when `component == "attention"`
(`lf.py:1186`), so a dense MLP leaf falls through to `AsymLoRALinear.from_host_weight` (`lf.py:1210`).
It is therefore the **standard** LoRA linear (the `_AsymLoRALinearWeightOffloadFunction` weight-offload
variant under production weight offload), **not** the attention `AsymActivationOffloadLoRALinear` and
**not** the expert engine.

## What `AsymLoRALinear` keeps on HBM (lora.py:396-410)

```text
out = base(x) + lora(x)
base(x):   AsymFrozenLinear → y = x @^R W_cpu     # CPU-resident frozen weight, @^R
           AsymFrozenLinearFunction saves NO activation; backward dx = dY @^R W
           (frozen_linear.py:1236-1335; dx via _dispatch_nt transpose_b=True :1313-1325)
lora(x):   _AsymLoRALinearWeightOffloadFunction (lora.py:177-235)
           flat    = x.reshape(-1, in).contiguous()
           x_lora  = flat.to(bf16).contiguous()    # [M,in]  — saved on HBM (lora.py:196)
           low_rank = x_lora @ A.T                  # [M,r] HBM — plain @ (NOT @^L) (lora.py:197)
           out      = (low_rank @ B.T) * scale      # [M,out]  (lora.py:198)
           save_for_backward(x_lora, low_rank)      # both on HBM (lora.py:204)
```

So per dense projection the LoRA path produces two HBM saves — the cast input `x_lora [M,in]` and the
low-rank `S = low_rank [M,r]`. The base is `@^R` and saves nothing. The LoRA backward (`lora.py:210-235`)
is **all plain HBM `@`** (`grad_b = dY.T @ S`, `dS = dY @ B`, `grad_a = dS.T @ x_lora`,
`grad_x = dS @ A`) — there is no CPU-left/CPU-right grouped AsymGEMM here. The saved `x_lora`/`S` are
the tensors the generic hook below offloads, then **stages back to HBM** before this backward runs.

## What actually offloads to CPU: the generic per-layer hook

With `layer_gc=true` (`ASYMM_LAYER_GC=1`), every decoder layer is wrapped by
`DecoderLayerGlueGCWrapper(offload_mode="custom")` (`decoder_layer_glue_gc.py:133`, install
`lf.py:2305-2306`, default `offload_mode="custom"` `:252`). On the training forward it:

1. **Checkpoints (recomputes) the two RMSNorms** — `input_layernorm` and `post_attention_layernorm`
   are each run inside `torch.utils.checkpoint` (`_checkpoint_norm`, `:172-187`, `:222`, `:233`).
2. Runs the **entire layer body** (attention + MLP, `_manual_forward` `:215-242`) under
   `saved_tensors_hooks(pack, unpack)` of an internal `DecoderSavedTensorOffloadWrapper`
   (`offload_mode == "custom"` → `:210-213`; wrapper built `:151`).

`DecoderSavedTensorOffloadWrapper._pack` (`decoder_activation_offload.py:99,188-222`) offloads to
**pinned CPU** every saved tensor that is:

- a **CUDA** tensor (`:162`),
- dtype ∈ {bf16, fp16, fp32} (default `_DEFAULT_..._DTYPES`, `:15,164`),
- `nbytes ≥ min_bytes` (default **1 MiB**, `:14,171`),
- **not** an `nn.Parameter` (`:177-180`) — skipped regardless of `requires_grad`,

with `require_grad` **default False** (`:121`), i.e. it offloads **regardless of `requires_grad`**.
There is **no selective recompute and no curated CPU-left schedule** for the dense MLP — the hook
offloads whatever the eager graph saved.

## The dense-MLP saved set (per layer) — verified

The eager SwiGLU graph `down_proj(silu(gate_proj(x)) * up_proj(x))` with `AsymLoRALinear` leaves saves
(confirmed by replaying the graph under `saved_tensors_hooks` and binning by shape):

| tensor | shape | who saves it | offloaded? |
|---|---|---|---|
| `gate = gate_proj(x)` | `[M,I]` | SiLU backward needs its input | ✓ CPU |
| `silu(gate)` | `[M,I]` | the `* up` mul backward (left operand) | ✓ CPU |
| `up = up_proj(x)` | `[M,I]` | the `* up` mul backward (right operand) | ✓ CPU |
| `act = silu(gate)·up` | `[M,I]` | `down_proj`'s `x_lora` (cast input) | ✓ CPU |
| `x_lora` of gate_proj | `[M,H]` | `_AsymLoRALinearWeightOffloadFunction` (= cast `normed`) | ✓ CPU |
| `x_lora` of up_proj | `[M,H]` | same (a second copy of `normed`) | ✓ CPU |
| `S` of gate/up/down | `[M,r]` ×3 | `low_rank` of each projection | see threshold |

→ **4 × `[M,I]`** go to CPU per dense MLP layer: `gate`, `silu(gate)`, `up`, `act`. This is the
likely "CPU blows up at long sequence" cause: the generic hook offloads **gate AND silu(gate) AND up
AND act** (no recompute, no fusion). The HF MLP runs `gate_proj` and `up_proj` as **separate** leaves,
so they are 2 separate `[M,I]` saves (never a fused `[M,2I]`); the idealized "fc1 → `[M,2I]` offload +
`mlp_act` recompute" schedule in `module_ops.md:77-80,101` is the **target**, not the wired behavior.

## Byte accounting (Qwen3-32B, `module_ops.md:24-30`; M=B·T=16384, bf16, r=64)

`H=5120, I=25600, L=64`. `[M,I]`=800 MiB, `[M,H]`=160 MiB, `[M,r=64]`=**2 MiB**.

Per dense MLP layer offloaded to CPU:

```text
4 × [M,I]  gate, silu(gate), up, act      = 4 × 800  = 3200 MiB
2 × [M,H]  x_lora(gate), x_lora(up)        = 2 × 160  =  320 MiB
3 × [M,r]  S_gate, S_up, S_down            = 3 ×   2  =    6 MiB
                                            ≈ 3.44 GiB / dense MLP layer
```

`[M,r=64]` = 2 MiB **exceeds** the 1 MiB `min_bytes`, so at production `r=64` (and any `M ≥ 8192`) the
LoRA `S` tensors **also offload to CPU** — they only stay on HBM when `M·r·2 < 1 MiB` (e.g. `module_ops.md`'s
`r=16` gives `[M,r]`=0.5 MiB). Offloaded tensors persist from forward until their layer's backward, so
forward end ≈ `L × per-layer` CPU residency; MLP alone ≈ `3.44 GiB × 64 ≈ 220 GiB`
(cf. `module_ops.md:36` "All-saved ≈3.38 GiB/layer ×64 ≈ 216 GiB (infeasible)").

## Operand placement (Part B)

```text
base proj (gate/up/down):  frozen W CPU-resident → @^R, no activation saved (frozen_linear.py:1313)
LoRA-A / LoRA-B fwd+bwd:   plain HBM @ (AsymLoRALinear, lora.py:197-198, 224-234) — NOT @^L/@^R
                           the CPU x_lora/S are STAGED BACK to HBM by the generic unpack before bwd
[M,I] & [M,H] saves:       offloaded to pinned CPU by the generic decoder hook, not a curated route
```

This is strictly worse than the planned dense design (CPU-resident `X` + CPU-left LoRA-A `@^L`,
never re-materialized): here the wide saves round-trip CPU↔HBM and the LoRA grads run entirely on HBM
(see `issues.md:7-9` issue 1, `attn_math_real.md`).

## fp32 silu transient

The synthetic benchmark scaffold computes the SwiGLU in fp32: `activated = F.silu(gate.float()) * up.float()`
(`asym_gemm/training/dense.py:471`, `issues.md:17-18` issue 4) — transiently `[M,I]` fp32 (~1.6 GiB
each) for `gate` and `up`. The **real** production HF `Qwen3MLP` (`modeling_qwen3.py:81-82`) computes
it in the model compute dtype (bf16), so production has no fp32 silu upcast; the fp32 transient is a
scaffold-only cost to keep in mind when reading `dense.py` throughput numbers.

## Surgical dense-MLP engine: OFF in production

`dense_mlp.py` (`build_dense_mlp_expert_engine` / `AsymDenseMLP`, `:52-112`) can re-use the Part-A
expert engine for a dense MLP as a single `E=1` expert (all tokens → expert 0), which would offload
`act` and run the down-proj LoRA backward on CPU via `@^R`. It is gated by
`ASYMM_DENSE_MLP_SURGICAL_OFFLOAD` (`lf.py:1981-1988`) and is **OFF by default / in production**
(env unset). The code comment is explicit (`lf.py:1975-1980`): the surgical path runs the **full**
dense `silu(gate)*up` + down-proj backward on CPU and **stalls** at dense scale (every token × the
large `I`), so `ASYMM_EXPERT_ACT_OFFLOAD` is a no-op for the dense MLP and `ASYMM_LAYER_GC` (Part B) is
the wired path: it offloads the MLP activations (HBM win) while keeping the backward GEMMs+silu on the
GPU (`issues.md:14-15` issue 3).
