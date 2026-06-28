# AsymGEMM dense-LLM LoRA-SFT activation scheduling — issues & insights

Audited the math docs (`attn_math.md`, `attn_math_real.md`, `mlp_math.md`, `mlp_math_real.md`, `fused_ce_math.md`) against the wired code. All file:line refs verified.

## Issues / caveats

1. **`attn_math.md` is aspirational — dense attention activation-offload is NOT wired.**
   Planned: projection inputs `X[M,H]` and `AttnOut[M,Dq]` are CPU-resident and LoRA-A uses CPU-left AsymGEMM `@^L` (`attn_math.md:15,254-304`). The wired path `_AsymLoRALinearWeightOffloadFunction` instead does `x_lora = flat.to(lora_dtype).contiguous()` → `low_rank = F.linear(x_lora,a)` → `ctx.save_for_backward(x_lora, low_rank)` (`asym_gemm/training/lora.py:196,197,204`) — both the wide input `[M,H]` and `low_rank [M,r]` saved **on HBM**. `attn_math_real.md:6-8,273` admits it: "What is NOT implemented yet… this is worse than the planned design."
   Impact (Qwen3-32B, H=5120, Dq=8192, M=16384, bf16): ~`x_lora 160MiB + AttnOut 256MiB + ~4×S` ≈ **418 MiB/attn-block** of removable HBM left on the table — the clearest un-built dense win.

2. **The `*_real.md` docs are MoE-shaped, not dense.**
   They carry per-expert grouping `E/G/offsets[g]/experts[g]/pack_g/M_g` (`mlp_math_real.md:16-18`) and compute silu and `dA_down` **on CPU** (`mlp_math_real.md:24,89,155,162-163`). A dense layer is E=1, one group, all tokens — that collapses away. Per-expert CPU compute is cheap only because `M_g << M`; dense `M = B·T` (every token) is exactly what stalls. Do not lift the MoE schedule verbatim for dense.

3. **`dense_mlp.py` reuses the MoE expert engine and is the STALLING path.**
   `build_dense_mlp_expert_engine` wraps the dense MLP as `AsymQwen3Experts` with `num_experts=1`, all tokens routed to expert 0 (`asym_gemm/training/dense_mlp.py:18,42,63,108-109`). Numerically exact (bit-identical LoRA grads, docstring `:1-9`) but runs the full dense `silu(gate)*up` + down-proj backward on CPU → stalls at scale (memory: "dense_mlp.py … STALLS at scale"). Correct dense design keeps GEMMs+silu on GPU, CPU as storage tier only ("dense MLP offload belongs to layer_gc"). Re-confirm production path does not take this surgical branch.

4. **Hidden fp32 silu transient in the MLP.**
   `MLP.forward` does `activated = F.silu(gate.float()) * up.float()` (`asym_gemm/training/dense.py:471`) — transiently upcasts gate and up to fp32 `[M,I]`, ~1.6 GiB **each** at I=25600, M=16384. Must be counted in offload accounting; the win only lands if silu runs in the saved bf16 dtype or in fp32 on CPU (off-HBM).

5. **`dense.py` is a synthetic benchmark scaffold, not the real model.**
   Random init (`_randn`, `dense.py:275-276,292-306`), `nn.LayerNorm` not RMSNorm (`dense.py:493-494,610`), and `SelfAttention.forward` has NO q_norm/k_norm and NO RoPE — just q/k/v → SDPA → o (`dense.py:421-429`). Fine for HBM/throughput parity; NOT the real attention core. Its RoPE/q-k-norm omission is a scaffold limit, not a design choice — the production path (LlamaFactory + engine hooks) has them.

## Correct / keep

- **CPU-resident frozen base weights fetched via CPU-right AsymGEMM (`@^R`)** for every base GEMM (attn + MLP) — weight fetch hidden under compute-bound GEMMs (`attn_math_real.md:10-11`, `attn_math.md:179-181`).
- **GPU silu-backward (`ASYMM_EXPERT_SILU_BWD_GPU`)** — correct direction for dense (keep silu on GPU).
- **Fused CE / liger for the `[M,V]` logits apex** — single biggest HBM win, measured 46.08→4.51 and 50.99→12.33 GiB (`fused_ce_math.md`); mandatory under activation offload, since `[M,V]` is the whole remaining peak.

## Open decisions

- Wire attention-projection activation offload→fetch (`X`, `AttnOut` → CPU; LoRA-A via `@^L`) — issue 1.
- Choose recompute-vs-fetch for the MLP `act[M,I]` intermediate (GPU silu + stream, not CPU silu) — issues 3,4.
- Long-T: offload FlashAttention-saved Q/K/V as a later stage.
