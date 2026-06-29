# Fine-Grained Activation Offload via AsymGEMM Placement (long-sequence LoRA-SFT)

## GOAL (read first)

Extend the **maximum trainable sequence length** of dense **Qwen3-32B** LoRA-SFT on a single GPU **beyond the
`superoffload_mem | unsloth | ligerloss1` baseline**, at the same batch size, by using AsymGEMM's actual capability:
**run the big GEMMs on GPU with CPU-resident operands (`@^R`), and run the wide non-GEMM (elementwise) ops on CPU**,
so the large `[M,I]` intermediates never sit on GPU and the per-layer HBM peak collapses.

Concrete target / acceptance:
- Hardware envelope (measured): **GPU = 184 GiB** (1× Blackwell), **CPU = 958 GiB** (`numactl --membind=0,1`, two
  Grace nodes; shared with other jobs → treat ~870 GiB as the activation budget).
- Baseline to beat: `q3-32b | superoffload_mem | unsloth | ligerloss1 | b8` → peak **178 GiB reserved**, ceiling
  ≈ **s45000** (HBM-bound; OOMs above ~s45k). `profiling/.../superoffload_mem__source__unsloth__.../b8_s45000_ga1/`.
- **Win = train a strictly longer real sequence than s45000 at b8** with finite `loss`, GPU < 184 GiB, CPU < ~870 GiB.
  Latency may be worse (accepted), not pathological. Target ceiling (projection): **~s120–150k** (CPU-bound).

Backend / harness contract:
```
q3-32b | asym | norecompute | ligerloss1 | <seq>|8|1 | none|false|false|false|false|false
```
- `asym` backend: frozen base weights CPU-resident, fetched per-GEMM via CPU-right AsymGEMM `@^R`.
- **Optimizer is an INDEPENDENT choice** (see "Optimizer + contention rule"): CPU AdamW (`asym_cpuadamwds`) **or**
  GPU AdamW (plain `asym`). For the max-seq regime, GPU AdamW is mildly preferred (kills the grad-offload C2C
  contention that slows the CPU silu) at the cost of ~8 GiB HBM; either works.
- `norecompute` = **AsymGEMM owns the offload/recompute itself** (no HF/Unsloth GC, no `ASYMM_LAYER_GC`).
- `ligerloss1` stays on — fused CE removes the `[M,V]` logits apex (≈102 GiB at this M); without it nothing fits.

What was tried and why this is the design (see `MEMORY: asym-unsloth-beats-superoffload`, `finegrained-offload-design`):
- `none|T|T|F|T|T` (offload **everything**, hold all layers) → CPU ~5 TB → OOMs CPU at ~T6.6k. **Wrong.**
- `unsloth` / `asym|unsloth` (recompute **everything** on GPU) → HBM-bound at the whole-layer GPU recompute
  transient (~178 GiB) → ceiling s45k. **No win** (measured `asym|unsloth ≈ superoffload`).
- **This design:** keep the GEMMs on GPU but **do the wide elementwise on CPU and hold only what's cheap**, so the
  per-layer GPU transient is ~one `[M,I]` (~17 GiB), not ~6 (~100 GiB). That is the actual peak lever.

### Two corrections to earlier drafts of this doc (do not regress)
1. **`@^R` + CPU-elementwise DOES reduce the `[M,I]` peak.** Earlier text said offload can't touch the `[M,I]`. That
   was wrong: the GEMMs (`fc1`/`fc2`) must run on GPU (CPU GEMM at M=360k is hours/layer), but their `[M,I]` outputs
   are **offloaded immediately** and the consuming **silu/mul run on CPU** (`mlp_math_real.md` / the expert engine),
   so only **1×`[M,I]` is ever live on GPU**. The MLP transient drops ~100 GiB → ~17 GiB. This is the lever.
2. **"Base weights off HBM" is NOT an asym-specific win.** superoffload_mem (ZeRO-3 `offload_param`) already keeps
   the 64 GiB base on CPU and streams ~1 GiB/layer — its 178 GiB GPU peak is **all activations, `weights … 0.00 MiB`**
   in the breakdown. So base-weight placement is **common, ~1 GiB transient, not the differentiator**. The asym
   advantage is the **activation-side asymmetric ops** (`@^R`-fetch + CPU elementwise), which ZeRO-3 lacks.

---

## The peak lever: AsymGEMM placement (which op runs where)

| op kind | examples | where | rule |
|---|---|---|---|
| **big GEMM** | qkv, o_proj, fc1, fc2 (base) | **GPU**, frozen weight streamed via `@^R` | CPU GEMM is ~10⁴× too slow at dense M; output is one tensor on GPU, offloaded right after |
| **dA (weight-grad GEMM)** | `dA = dS.ᵀ @^R X_cpu` | **GPU**, activation operand on **CPU** | activation **fetched into the GEMM, never materialized** (no-remat) |
| **dB** | `dY.ᵀ @ stage(S)` | GPU | `S` is `[M,r]`, tiny |
| **SDPA** | FlashAttention | **GPU** | fused kernel, O(M) mem, no CPU form |
| **elementwise / norm** | **silu, the `*`**, RMSNorm, RoPE, residual add | **CPU** (on the already-offloaded copies) | memory-bound; doing them on CPU means the `[M,I]` results **never touch GPU** |

**The boundary = (a GEMM-output offload) FUSED WITH the CPU elementwise that consumes it.** A big GEMM's `[M,I]`
output is offloaded D2H *immediately*; the silu/mul that reads it runs **on CPU on that copy (no stage-back)**; the
result is staged to GPU only to feed the *next* GEMM. So at any instant **only one `[M,I]` is live on GPU**.

**Why it cuts the *peak* (trace `mlp_math_real.md` backward, devices annotated):**
```
dY[M,H] GPU
dact  = dY @^R W_down_cpu                 GPU GEMM   -> dact[M,I] on GPU ........ 1 live [M,I]
dA_down = dS_down.ᵀ @^R act_cpu           GPU GEMM, act stays CPU (no remat)
⬇ offload dact -> dact_cpu ; free GPU                                            0 live [M,I]
dgate = silu_bwd(dact_cpu*up_cpu, gate_cpu) ; dup = dact_cpu*silu(gate_cpu)   CPU  (born/consumed on CPU)
⬆ stage dgate ; dX = dgate @^R W_gate_cpu  GPU GEMM .......................... 1 live [M,I]
⬆ stage dup   ; dX += dup   @^R W_up_cpu   GPU GEMM .......................... 1 live [M,I]
dA_gate = dS_gate.ᵀ @^R N2_cpu ; dA_up = dS_up.ᵀ @^R N2_cpu   GPU GEMM, N2 on CPU
```
`gate, up, act, dgate, dup` are never on GPU. Forward is symmetric (`fc1 -> gate_up[M,2I] GPU -> offload -> act=silu*up CPU`).
MLP GPU transient ≈ **1×`[M,I]` (17 GiB)** vs ~100 GiB if all `[M,I]` were GPU-resident.

**Cost (measured, accepted):** the CPU SwiGLU backward is **~640 ms/layer** (`qwen3_moe.py:923` — bandwidth-bound,
contends with grad-offload D2H). ×64×(fwd+bwd) ≈ +80 s/step. `ASYMM_EXPERT_SILU_BWD_GPU` is the **peak↔speed dial**:
GPU-silu = sub-ms but stages `[M,I]` back (higher peak, lower seq); CPU-silu = the peak lever. We take CPU-silu.

The MLP math (`@^R` base + LoRA + CPU silu) is exactly `AsymQwen3Experts` (`mlp_math_real.md`). For a **dense** MLP,
`dense_mlp.py` already wraps it as an E=1 expert. The attention math (`@^R` base, `@^L` LoRA-A on offloaded X, `@^R`
dA, SDPA recompute) is exactly `AsymActivationOffloadLoRALinear` (`attn_math_real.md`). **Reuse both** — this design
is "run them, but recompute per layer so their offloaded activations stay transient".

---

## The per-layer chain — boundaries, devices, LoRA

`⬇`=offload GPU→CPU, `⬆`=stage CPU→GPU. `[T]`=per-layer transient (×1), `[H×64]`=held all layers.

**Attention** (no big elementwise → no silu-boundary; only dA-input offloads):
```
N1 = RMSNorm(X_in)            CPU              ⬆ N1
Q,K,V = qkv(N1) (+LoRA)       GPU @^R          (Q,K,V[M,Dq/Dkv] [T])
Q',K' = qk_norm/RoPE(Q,K)     GPU              (feeds SDPA)
AttnOut = SDPA(Q',K',V)       GPU FlashAttn    ⬇ AttnOut (for o dA)
A = o_proj(AttnOut) (+LoRA)   GPU @^R
H_mid = X_in + A              CPU add (X_in on CPU)
  dA: qkv = dS.ᵀ @^R X_in_cpu ;  o = dS.ᵀ @^R AttnOut_cpu     GPU GEMM, no remat
```
**MLP** (the peak driver; offload each `[M,I]` right after its GEMM):
```
N2 = RMSNorm(H_mid)          CPU              ⬆ N2
gate = N2 @^R W_gate (+LoRA) GPU @^R   ⬇ gate_cpu        | split fc1 into gate/up so only 1×[M,I] on GPU
up   = N2 @^R W_up   (+LoRA) GPU @^R   ⬇ up_cpu          |
act  = silu(gate_cpu)·up_cpu                 CPU  ← attached to the two ⬇ (NOT staged back)
Mout = act @^R W_down (+LoRA) GPU @^R  (⬆ act for base; LoRA = act_cpu @^L A_down)
--- backward: dact ⬇ -> CPU silu_bwd -> dgate,dup ⬆ -> dX GEMMs (as in the trace above) ---
```
LoRA everywhere: **LoRA-A** consumes the projection input on CPU (`X_cpu @^L A`) or GPU; **LoRA-B** is small; the
**low-rank `S` is held** (`[M,r]`) for `dB`; **dA fetches the input via `@^R`** (no remat).

---

## Bounding CPU: per-layer recompute + the offload list (two tiers)

Holding all of a layer's activations to backward is the `none|T|T|F|T|T` blowup: the `[M,I]` alone is
**1099–2197 GiB ×64 = TBs**. So:

- **`[M,I]` (`fc1`/`mlp_act`): never held — regenerated per layer in backward** with the CPU-silu placement above.
  CPU footprint = **one layer's** `[M,I]` (~6×17 ≈ 100 GiB), freed after that layer's bwd. This is fixed, not a choice.
- **Everything else: the offload list picks what to HOLD on CPU (`@^R`-fetch) vs recompute.** Held → skip a recompute
  (speed) at a CPU cost (×64). Recomputed → regenerated from the nearest held boundary.

### Producer ledger (q3-32b: H=5120, Dq=8192, Dkv=1024, I=25600, r=64, L=64; bytes bf16; M=360000 @ s45k)

| producer | shape | GiB/layer | **held×64** | consumer device | disposition |
|---|---|---:|---:|---|---|
| **`layer_input` X_in** | [M,H] | 3.43 | **220** | qkv `dA` (CPU `@^R`) | **mandatory held** (recompute root) |
| **LoRA `S_{q,k,v,o,gate,up,down}`** | [M,r]×7 | 0.30 | **19** | `dB` staged (GPU) | **mandatory held** (small but ×7×64) |
| **`gate`/`up`/`act`/`dact`/`dgate`/`dup`** | [M,I] | 17.2 | — (×1) | **silu on CPU** | **fixed CPU-silu boundary, transient** |
| `mlp_act` act (=down input) | [M,I] | 17.2 | — (×1) | down `dA` CPU `@^R` | recomputed, fetched no-remat |
| `attn_norm` N1 | [M,H] | 3.43 | 220 | qkv `dA` `@^R` | **optional held** vs recompute(CPU norm) |
| `mlp_norm` N2 | [M,H] | 3.43 | 220 | fc1/up `dA` `@^R` | **optional held** vs recompute |
| `core_attn` AttnOut | [M,Dq] | 5.49 | 352 | o_proj `dA` `@^R` | **optional held** vs recompute(SDPA) |
| `post_attn_resid` H_mid | [M,H] | 3.43 | 220 | mlp recompute root | **optional held** (= 2-segment split) |
| `fc1`/`mlp_act` as **held** | [M,2I]/[M,I] | 34/17 | **2197/1099** | — | **forbidden** → NVMe |
| **LoRA optimizer + grads** | total | — | **~8 (GPU)** | — | **keep on GPU** (peak-irrelevant; avoids contention) |

`ASYMM_OFFLOAD_PRODUCERS ⊆ {attn_norm, mlp_norm, core_attn, post_attn_resid}` — a **CPU↔speed** knob (does NOT change
the peak; the peak is the fixed `[M,I]` CPU-silu transient). Default `{}` = hold only `X_in`+`S`, recompute the rest →
min CPU, the ~35–40 GiB peak. `fc1`/`mlp_act` are **rejected** with an NVMe message.

---

## Accounting: ×64 held vs ×1 transient (do not conflate)

- **held / offloaded → ×64** (lives on CPU all layers): always quote the total. `X_in` = 220 GiB; the 7 LoRA `S` =
  **19 GiB** (not "tiny"); each optional `[M,H]` held = +220.
- **recompute-transient → ×1** (one layer live at a time on GPU, freed after its bwd): the `[M,I]` GEMM transient
  (~17 GiB) is NOT ×64.
- **optimizer / params → single all-model total** (~8 GiB), not per-layer.

Envelope @ s45k (projection, must measure):
- **GPU peak** ≈ 1×`[M,I]` (17) + attn FA (~12) + dX (~3) + **8 GiB GPU optimizer/grads** ≈ **~35–40 GiB**.
- **CPU** ≈ `X_in`×64 (220) + `S`×64 (19) + base 64 + one-layer `[M,I]` (~100) ≈ **~400 GiB**; each optional `[M,H]`
  held adds +220 → drives the **NVMe wall** (~s120–150k, where `X_in`×64 + the held list saturates ~870 GiB).

---

## The optimizer + the contention rule

**Rule:** *offload / CPU-compute ONLY what reduces the peak; keep peak-irrelevant work on GPU.* CPU work is not free —
it consumes C2C/CPU bandwidth that the *necessary* offloads (base-weight `@^R` streaming + the `[M,I]` D2H/H2D)
already contend for; the CPU silu is explicitly measured as "contended with the concurrent gradient-offload D2H"
(`qwen3_moe.py:923`). So:
- base weights (64 GiB → CPU) and the wide activations (the `[M,I]`) — **worth the contention** (they cut the peak).
- LoRA optimizer + grads (~8 GiB total, fixed, peak-irrelevant) — **keep on GPU**, *unless* you specifically need the
  HBM. CPU AdamW (`asym_cpuadamwds`, grad-offload) and GPU AdamW (plain `asym`) are an **orthogonal toggle**:
  `USE_ASYM_CPU_ADAMW` + `ASYM_CPU_ADAMW_{GRAD,WEIGHT}_OFFLOAD`. For max-seq, GPU AdamW (no grad-offload) frees C2C
  for the peak-reducing silu — small HBM cost, better latency. Both are valid; pick per run.

---

## The config / arg

`ASYMM_OFFLOAD_PRODUCERS` (comma-separated, `⊆ {attn_norm, mlp_norm, core_attn, post_attn_resid}`), forwarded like the
other `ASYMM_*` via `run_lf_lora_sft.sh` `RUN_ENV`, read in `lf.py`. Rules: `X_in` + LoRA `S` always held (not in the
list); `fc1`/`mlp_act` rejected (NVMe); unknown names error; **uniform across all layers**. Output-dir tag:
`offprod-<letters>` in fixed order (`attn_norm=n, qkv=q, core_attn=c, attn_proj=p, post_attn_resid=r, mlp_norm=m`),
empty = `offprod-none`. Leave `ASYMM_MLP_RECOMPUTE_CHUNK` (chunked MLP) and `ASYMM_LAYER_GC*` registered but **unused**.

---

## Megatron reference (the offload×recompute interleaving — our boundaries differ)

Megatron-Core v0.18 `--fine-grained-activation-offloading` is the mechanical reference for *offloading saved tensors
and reloading in backward*. **Our boundaries are not its "dummy" offload/recompute** — ours are **fused with CPU
compute** (the offloaded `[M,I]` is consumed by a CPU silu, not staged back), and we add `@^R`/`@^L`. But the offload
plumbing is the same; copy it.

Files: `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` (manager + group ops);
`tensor_parallel/random.py:555-829` (`checkpoint` / `CheckpointWithoutOutput`); `transformer/moe/experts.py:769-787`
(the `moe_act` input-offload × output-recompute example); `transformer/transformer_layer.py:617-693` (the `attn_norm`
bracket); `transformer_config.py:1120-1138,1658-1685` (config + valid names `attn_norm,qkv_linear,core_attn,attn_proj,
mlp_norm,expert_fc1,moe_act`).

Mechanism (verified): `with off_interface(flag, x, name) as x: <module>` … `group_commit(out, name,
forced_released_tensors=[x])`. `GroupStart` fwd opens a group + pushes `saved_tensors_hooks`; **bwd triggers H2D
reload**. `GroupCommit` fwd does **D2H on a side stream + `tensor.untyped_storage().resize_(0)` free** (torch GC won't
drop it); **bwd waits the reload event**. The D2H is **deferred past the last forward consumer** (`attn_norm` is
committed only after the residual add "because the residual is needed in self_attn_bda"). Synchronous v1: D2H on a
side stream, `record_stream`, wait the event before `resize_(0)`; reload H2D at the top of backward. **Drop** the PP /
microbatch / double-buffer / VPP / CUDA-graph machinery.

**CRITICAL — explicit autograd op, NOT `torch.utils.checkpoint`.** A prior attempt used `torch.utils.checkpoint`,
whose internal input-holder **bypasses `saved_tensors_hooks`** → the `[M,H]` boundaries piled on GPU (~4.3 GiB/layer)
→ **forward OOM at ~42 layers**. Do the D2H + `resize_(0)` free **explicitly** inside our own Function (Megatron's
`GroupCommit` / Unsloth's `UnslothGradientCheckpointing`, `LlamaFactory/.../checkpointing.py:44-77`). Run the original
forward under `no_grad` per layer so nothing is saved on the forward pass.

**Recompute determinism:** the recompute must be bit-exact. OK here only because `lora_dropout = attention_dropout = 0`
(enforced) and `@^R`/FlashAttention(0) are deterministic → no RNG op in a layer. Do not add RNG-bearing ops without
RNG save/restore. The parity test (below) guards this.

---

## Staged implementation plan (each stage independently testable)

**Stage 0 — plumbing.** Producer enum + `[M,*]` byte table (one place); parse/validate `ASYMM_OFFLOAD_PRODUCERS` in
`lf.py` (reject `fc1`/`mlp_act` → NVMe msg, unknown names); forward via `run_lf` `RUN_ENV`; install gated on `asym`
backend + env. Unit-test parse + no-op-when-unset.

**Stage 1 — per-layer AsymGEMM-placed processing + recompute, parity floor.** Implement the per-layer wrapper: hold
`X_in` + LoRA `S`; recompute the layer in backward via the **expert-engine CPU-silu path for the MLP** (reuse
`dense_mlp.py` / `AsymQwen3Experts` with CPU silu, `@^R` base, `@^L`/`@^R` LoRA) and the **attn `@^R` path** (reuse
`AsymActivationOffloadLoRALinear`). Use the explicit offload Function (not `torch.checkpoint`).
- **Test (CPU, tiny):** reuse `tests/training/test_decoder_layer_glue_gc.py` fakes; `loss`, `dX`, **all param grads**
  match the unwrapped layer in bf16 tol; on CUDA confirm `resize_(0)` actually frees (`memory_allocated` drops).
- **Run (q3-32b s45k b8, `PREPARE_DATASETS=true`, verify real `input_ids` length):** measure GPU peak (expect ~35–40
  GiB) and CPU (~400 GiB). This already beats s45k if it fits — push seq.

**Stage 2 — push seq + optional held list.** Climb seq; if GPU has headroom but CPU is tight, *reduce* the held list
(recompute more); if CPU has headroom but a recompute is slow, *add* an optional `[M,H]` hold. Find the max real seq
with GPU<184, CPU<870, finite loss. Sweep via `scripts/lf/profile_lora_lf_test_both.sh`
(`MAX_STEPS=1 WARMUP_STEPS=1 PROFILERS=source PLOT=false PREPARE_DATASETS=true GPU_POOL=<free>`), one run at a time.

**Stage 3 — record.** `profile.json` + memory breakdown; winning rows into the test script; seq-vs-(GPU,CPU,list,step)
Pareto into MEMORY.

### Guardrails
- Run sequentially; watch `numactl -H` node-0/1 free; if combined < ~6 GiB, **`kill -TERM`** (never `-9` — corrupts
  cpu_adam JIT) to protect other users. Single-GPU asym proc is a bare `python` (kill by PID).
- Always verify real `input_ids` length; identical GPU peak across two seq lengths = stale/capped dataset.

---

## NVMe stop condition (STOP, do not keep going)

CPU (the held `X_in`×64 + `S`×64 + one-layer `[M,I]`) is the binding constraint. **Stop and escalate to NVMe** when:
1. `X_in`×64 + the minimal held list overflows ~870 GiB before HBM (184) binds — i.e. around **~s120–150k**.
2. The sweep needs to hold an `[M,I]`/`[M,2I]` producer (1099–2197 GiB) — never CPU-fittable here.
3. Single-run CPU RSS (minus base/opt) approaches ~870 GiB.

When triggered: **do not implement NVMe inline, do not keep pushing CPU.** Hand off to `agent/impls/nvme_offload.md`
(139 KB, drafted) + the ZeRO-NVMe configs. The refactor: make the activation-offload storage tier pluggable
(HBM→pinned-CPU→NVMe), spill **coldest-first** (early layers, used last in bwd) at ~5–7 GB/s under the long backward.
Record in MEMORY what seq NVMe unlocks, **then stop the autonomous loop**. Until then, **keep going** (climb seq,
tune the held list) until the goal is comfortably beaten or this condition fires.

---

## Constraints / non-goals (v1)

- Backend `asym` (base `@^R`) + `norecompute` + `ligerloss1`. Optimizer CPU-or-GPU AdamW (orthogonal). No HF/Unsloth
  GC, no `ASYMM_LAYER_GC`.
- **MLP elementwise (silu/mul) on CPU; GEMMs on GPU via `@^R`.** This is the peak lever — do NOT "recompute the dense
  MLP on GPU" (that was the stale, peak-bound design).
- **Do NOT use token-chunked MLP** (`ASYMM_MLP_RECOMPUTE_CHUNK`/`chunked_mlp.py`) — registered, unused (different lever).
- No prefetch / async double-buffer / CUDA-graph retention in v1. Synchronous D2H/H2D. Correctness + "fits + finite
  loss" first.
- Uniform across all layers. No NVMe in v1 (hitting its condition is a STOP).

---

## Code hooks & references

- `asym_gemm/training/qwen3_moe.py` — `AsymQwen3Experts` + `_activation_offload_cpu_silu_mul/_backward` (`:885,:900`),
  `_silu_backward_gpu` (`:930`, the speed dial); the MLP `@^R`+CPU-silu engine (= `mlp_math_real.md`).
- `asym_gemm/training/dense_mlp.py` — wraps a dense MLP as an E=1 expert through that engine. **Use this for the MLP.**
- `asym_gemm/training/attention_activation_offload.py` — `AsymActivationOffloadLoRALinear` (`@^R` base, `@^L` LoRA-A on
  offloaded X, `@^R` dA, q/k/v X-sharing). **Use this for attention** (= `attn_math_real.md`).
- `asym_gemm/training/activation_offload.py` — `ActivationOffloadManager` (pinned D2H/H2D/release + byte accounting);
  add the `untyped_storage().resize_(0)` free if it doesn't truly drop GPU storage.
- `asym_gemm/training/decoder_layer_glue_gc.py` `_manual_forward` — per-layer manual forward, the host for the wrapper.
- `asym_gemm/integrations/lf.py` `apply_lf_asym_lora` — install + gate (mirror the `install_chunked_mlp_on_dense_mlps`
  site); `scripts/lf/run_lf_lora_sft.sh` `RUN_ENV` — forward the env.
- `LlamaFactory/.../model_utils/checkpointing.py:44-77` — `UnslothGradientCheckpointing`, the explicit D2H+recompute op.
- `agent/math/mlp_math_real.md`, `attn_math_real.md`, `module_ops.md` — the op placement (which GEMMs are `@^R`/`@^L`,
  which elementwise are CPU). **The math docs ARE the per-op device map; this design is wiring them per-layer + recompute.**
- Megatron: `third_party/megatron-lm/megatron/core/pipeline_parallel/fine_grained_activation_offload.py` and the files
  listed in the Megatron-reference section above.
