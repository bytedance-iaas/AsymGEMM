"""Minimal upstream repro: fla chunk_gated_delta_rule fails above ~70k tokens/row.

Pure fla call — no AsymGEMM, no LlamaFactory, no model code. Mirrors the stock
transformers Qwen3.5 GatedDeltaNet calling convention (GQA repeat_interleave,
l2norm-in-kernel, fp32 g, bf16 beta). Observed on GB200 (sm100), fla 0.5.0
(flash-linear-attention == fla-core == 0.5.0), triton 3.7.0, torch 2.12.0+cu130:

    S=2048..70000 : clean forward+backward
    S=75000, B=8  : CUDA illegal memory access (chunk_gated_delta_rule_fwd_h)

Inside a full model the same regime silently corrupts instead of faulting
(loss ~0 / NaN grads localized to the delta-net parameters) — layout-dependent
out-of-bounds access. Seen identically under SDPA and FA4 attention stacks and
for non-Asym baselines, so the fault is in the fla kernel path.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/testing/fla_gdn_longseq_repro.py
"""

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def probe(S: int, B: int = 8) -> None:
    Hk, Hv, D = 16, 32, 128  # qwen3.5-35B linear-attention head shapes
    q = (torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16) * 0.5).repeat_interleave(Hv // Hk, dim=2)
    k = (torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16) * 0.5).repeat_interleave(Hv // Hk, dim=2)
    v = torch.randn(B, S, Hv, D, device="cuda", dtype=torch.bfloat16) * 0.5
    g = -torch.rand(B, S, Hv, device="cuda", dtype=torch.float32) * 0.5
    beta = torch.rand(B, S, Hv, device="cuda", dtype=torch.bfloat16)
    q.requires_grad_(True)
    out, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
    )
    out.sum().backward()
    print(f"S={S} B={B}: fwd_nan={int(torch.isnan(out).sum())} bwd_nan={int(torch.isnan(q.grad).sum())}", flush=True)
    del q, k, v, g, beta, out
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for S in (2048, 32768, 60000, 65600, 70000, 75000, 80072):
        try:
            probe(S)
        except Exception as exc:  # noqa: BLE001
            print(f"S={S} B=8: CRASH {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            break
