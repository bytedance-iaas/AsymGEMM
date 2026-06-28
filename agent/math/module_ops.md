# Dense-LLM Activation Schedule (LoRA SFT)

One row per activation, named by its **producer** module. Drives the per-tensor schedule:
where to compute (CPU/GPU) and whether to persist / recompute / offload / offload+fetch.
Megatron-aligned. Caveats / unwired gaps: see `issues.md`.

## Convention

- **Rows = modules, named by producer** (the `= C` of the Ops column). Megatron names
  recompute by producer, offload by consumer ("input of X"); we use producer for both, since
  `output-of-M ≡ input-of-next`. Map: MG `qkv_linear`-in = `attn_norm`-out · `core_attn`-in =
  `qkv`-out · `attn_proj`-in = `core_attn`-out · `moe_act`-in = `fc1`-out.
- **Tokens** (Megatron): `attn_norm, qkv, core_attn, attn_proj, mlp_norm, fc1, mlp_act, fc2`
  (+ `embed, final_norm, lm_head`). LoRA rows added (MG has none): `*·loraA` → `S[M,r]`;
  LoRA-B delta folds into the base output.
- **Ops**: `@`=GEMM (shared dim = contraction; `W[out,in]` shown as `[…,k]@[k,out]`),
  `* + -`=elementwise, `FA`=FlashAttention.
- **Bound**: compute = AI ≫ ridge (≈330 FLOP/B); memory = AI ≪ ridge.
- **Disposition** (extends MG `persist|offload|recompute`): `offload+fetch ★` = CPU-resident,
  streamed into the bwd GEMM via AsymGEMM `@^L/@^R`, **never re-materialized** (ours; only when
  the sole bwd consumer is a GEMM). `cpu-weight` = frozen base on CPU, `@^R`-fetched.
  `fuse` = never-form. `FA` = FlashAttention-owned.

## Dims & sizes (Qwen3-32B, M=B·T=16384, bf16)

`H=5120 · Dq=8192 · Dkv=1024 · I=25600 · r=16 · V=151936 · L=64`

| `[M,H]` | `[M,Dq]` | `[M,Dkv]` | `[M,2I]` | `[M,I]` | `[M,r]` | `[M,V]` |
|---:|---:|---:|---:|---:|---:|---:|
| 160 MiB | 256 MiB | 32 MiB | 1600 MiB | 800 MiB | 0.5 MiB | 4.64 GiB |

| AI (FLOP/B) | base proj 2643 | lm_head 3803 | LoRA 16 | RMSNorm 1.5 | SwiGLU 0.7 |
|---|---|---|---|---|---|
| → | compute | compute | memory | memory | memory |

All-saved ≈3.38 GiB/layer ×64 ≈ **216 GiB** (infeasible). Frozen base 58 GiB → all `cpu-weight`.
Offload priority: `[M,I]` (2400/layer) ≫ `[M,Dq]` (512) > `[M,H]` (160) ≫ `[M,r]`.

---

## token_embed
`input_ids → embed(gather) → X₀ (=input_stream)`

| Module | Ops | Bound | Disposition | Why |
|---|---|---|---|---|
| `embed` Wₑ | frozen `[V,H]` | — | cpu-weight | 1.45 GiB; not LoRA'd |
| `embed` | `Wₑ[ids]: [V,H]→[M,H]` | memory | recompute | = layer-0 stream; frozen → no save |

## input_stream
The `[M,H]` carry; feeds `attn_norm`/`mlp_norm` (recompute) + skip-add.

| Module | Ops | Bound | Disposition | Why |
|---|---|---|---|---|
| `input_stream` | `embed` / prev `+resid` → `[M,H]` (carry) | memory | persist → offload | survives full layer = checkpoint boundary |

## attn
`input_stream → attn_norm → qkv → qk_norm+rope → core_attn → AttnOut → attn_proj → +resid`

| Module | Ops | Bound | Disposition | Why |
|---|---|---|---|---|
| `attn_norm` | `[M,H] * [M,1] * [H] = [M,H]` | memory | recompute | RMSNorm; = qkv LoRA input → dA plain `@` |
| `qkv` base | `[M,H] @ [H,Dq+2Dkv] = [M,Dq+2Dkv]` | compute | cpu-weight | out Q,K,V → FA |
| `qkv·loraA` | `[M,H] @ [H,r] = [M,r]` ×3 | memory | persist | S for dB; 0.5 MiB |
| `qk_norm+rope` | `rmsnorm·rope(Q,K) = [M,Dq],[M,Dkv]` | memory | recompute (FA) | per-head norm+RoPE (Qwen3; Llama=RoPE only) |
| `core_attn` | `[M,Dq] = FA(Q′,K′,V)`; scores `[B,Hq,T,T]` **never formed** | compute | FA | FA saves Q,K,V,LSE; recomputes scores in bwd |
| `core_attn` out | `AttnOut [M,Dq] → attn_proj in` | compute | offload+fetch ★ | recompute = re-run FA (costly); today HBM (gap) |
| `attn_proj` base | `[M,Dq] @ [Dq,H] = [M,H]` | compute | cpu-weight | out → +resid (unsaved) |
| `attn_proj·loraA` | `[M,Dq] @ [Dq,r] = [M,r]` | memory | persist | S_o for dB |
| `+resid` | `[M,H] + [M,H] = [M,H]` | memory | persist → offload | produces next `input_stream` |

## mlp
`input_stream → mlp_norm → fc1 (gate,up) → mlp_act (silu·up) → fc2 → +resid`

| Module | Ops | Bound | Disposition | Why |
|---|---|---|---|---|
| `mlp_norm` | `[M,H] * [M,1] * [H] = [M,H]` | memory | recompute | RMSNorm; = fc1 LoRA input → dA plain `@` |
| `fc1` base | `[M,H] @ [H,2I] = [M,2I]` (gate‖up) | compute | cpu-weight | out → gate,up |
| `fc1·loraA` | `[M,H] @ [H,r] = [M,r]` ×2 | memory | persist | S_gate/S_up for dB |
| `fc1` out | `gate,up [M,2I] → mlp_act in` | compute | offload | bwd silu (GPU) needs whole gate,up |
| `mlp_act` | `silu([M,I]) * [M,I] = [M,I]` | memory | recompute | from staged gate,up |
| `fc2` base | `[M,I] @ [I,H] = [M,H]` | compute | cpu-weight | out → +resid (unsaved) |
| `fc2·loraA` | `[M,I] @ [I,r] = [M,r]` | memory | persist | S_down for dB |
| `+resid` | `[M,H] + [M,H] = [M,H]` | memory | persist → offload | produces next `input_stream` |

## lm_head
`input_stream(top) → final_norm → lm_head → CE loss`

| Module | Ops | Bound | Disposition | Why |
|---|---|---|---|---|
| `final_norm` | `[M,H] * [M,1] * [H] = [M,H]` | memory | recompute | from top stream |
| `lm_head` Wₗₘ | frozen `[V,H]` | — | cpu-weight | 1.45 GiB |
| `lm_head` Z | `[M,H] @ [H,V] = [M,V]` | compute | fuse | fp32 Zf/P/dZ → 32–42 GiB peak; liger chunks rows |

---

## Policy

1. `input_stream` = backbone → `persist` (→ `offload`/NVMe at depth); norms recompute from it.
2. Frozen base GEMMs (qkv, attn_proj, fc1, fc2, embed, lm_head) → `cpu-weight` `@^R`, GPU.
3. LoRA `S[M,r]` → `persist` (too small to move).
4. Wide + sole-bwd-consumer-a-GEMM (AttnOut; act) → `offload+fetch ★`.
5. Wide + bwd elementwise/softmax consumer (gate,up; FA Q/K/V) → `offload` (stage), no fetch.
6. Memory-bound + inputs kept (norms, qk_norm+rope, act) → `recompute`. Logits → `fuse`.

**Compute (dense):** GEMMs + silu always on **GPU** — dense M = full B·T, so CPU GEMM *and* CPU
silu over `[M,I]` stall. CPU(+NVMe) = storage/streaming tier, not compute. (MoE differs:
per-expert M_g ≪ M makes CPU compute viable — the expert engine, not this.)

**CPU/NVMe:** pinned RAM (GB200 ≈958 GiB, OOMs ≈0.8 TB → offload is selective). Spill
**coldest-first to NVMe** (early layers touched last in bwd, hiding ≈5–7 GB/s). C2C ≈400–900 GB/s = shared cap.
