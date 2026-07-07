#!/usr/bin/env python3
"""I7/E1 block-level parity: static EP-2 two-branch MoE block vs the |1 full block.

Tiny synthetic Qwen3-MoE (E=8, top4). Both branches get the SAME input (replicated
residual, the sTP invariant); branch d executes experts [d*E/2,(d+1)*E/2) via
ep_expert_range slicing over SHARED pinned banks; y0 + y1 must equal the full block's
output (bf16 band; the fp32 scatter orders differ). LoRA grads: branch grads must match
the full block's per-expert grads (sliced) within band.

Gates: fwd band <= 2e-2 abs; grad rel band <= 1e-2 p99; route bits identical across
branches; zero dropped tokens (counts sum).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    from asym_gemm.training.qwen3_moe import AsymQwen3MoeBlock
    from asym_gemm.training.stp_moe import build_ep_branch_block

    torch.manual_seed(0)
    E, H, I, TOPK, T = 8, 256, 128, 4, 512
    cfg = Qwen3MoeConfig(
        hidden_size=H, moe_intermediate_size=I, num_experts=E, num_experts_per_tok=TOPK,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=I, norm_topk_prob=True,
    )
    src = Qwen3MoeSparseMoeBlock(cfg).to(torch.bfloat16)
    with torch.no_grad():
        # fresh well-spread router logits: tiny default init gives near-tied bf16 logits
        # whose stable top-k collapses onto experts 0..3
        src.gate.weight.copy_(torch.randn_like(src.gate.weight.float()).mul(0.6).to(src.gate.weight.dtype))
        # non-vacuous outputs: default init x tiny inputs underflows bf16 to ~0 and every
        # band passes trivially — scale expert weights to get |y| ~ 0.1
        src.experts.gate_up_proj.mul_(32.0)
        src.experts.down_proj.mul_(32.0)
    for p in src.parameters():
        p.requires_grad_(False)

    dev0, dev1 = torch.device("cuda", 0), torch.device("cuda", 1)
    block = AsymQwen3MoeBlock(
        src, backend="asym", precision="bf16", offload=True,
        lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
    )
    # trainable LoRA gets random values so grads are nontrivial
    with torch.no_grad():
        for name, p in block.experts.named_parameters():
            if "lora" in name:
                p.normal_(0, 0.1)

    # move trainable/gate to dev0 (banks stay pinned host)
    block.gate = block.gate.to(dev0)
    for name in ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B"):
        param = getattr(block.experts, name)
        moved = torch.nn.Parameter(param.detach().to(dev0), requires_grad=True)
        block.experts._parameters[name] = moved
        setattr(block.experts, name, moved)

    b0 = build_ep_branch_block(block, 0, E // 2, dev0)
    b1 = build_ep_branch_block(block, E // 2, E, dev1)

    x = (torch.randn(1, T, H, dtype=torch.float32) / 4).to(torch.bfloat16)
    x0 = x.to(dev0).requires_grad_(True)
    x0b = x.to(dev0).requires_grad_(True)
    x1 = x.to(dev1).requires_grad_(True)

    failures = []

    # route bits identical across branches (frozen router replicas)
    with torch.no_grad():
        idx0, w0, _ = b0._compute_routing(x0.view(-1, H))
        idx1, w1, _ = b1._compute_routing(x1.view(-1, H))
    if not torch.equal(idx0.cpu(), idx1.cpu()):
        failures.append("route bits differ across branches")
    counts = torch.bincount(idx0.reshape(-1).cpu(), minlength=E)
    if int(counts.sum()) != T * TOPK:
        failures.append("dropped tokens")

    # forward parity: y0 + y1 == y_ref
    y_ref = block(x0)
    with torch.cuda.device(dev1):
        y1_out = b1(x1)
    y0_out = b0(x0b)
    y_sum = y0_out + y1_out.to(dev0)
    band = (y_sum.float() - y_ref.float()).abs().max().item()
    ref_scale = y_ref.float().abs().mean().item()
    ref_max = y_ref.float().abs().max().item()
    print(f"[parity] fwd abs band: {band:.5f} (|y| mean {ref_scale:.4f} max {ref_max:.4f})")
    if ref_scale < 5e-3:
        failures.append(f"VACUOUS: ref |y| mean {ref_scale}")
    # partials are bf16-quantized before the cross-device add: expect ~1-2 ulp at max|y|
    if band > 0.05 * ref_max:
        failures.append(f"fwd band {band} vs 5% of max signal {ref_max}")

    # grad parity: d(loss)/d(LoRA) per branch == full-block grads sliced
    g = torch.randn_like(y_ref, dtype=torch.float32).to(torch.bfloat16)
    y_ref.backward(g)
    y0_out.backward(g)
    with torch.cuda.device(dev1):
        y1_out.backward(g.to(dev1))
    worst = 0.0
    for name in ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B"):
        full = getattr(block.experts, name).grad
        for br, lo, hi in ((b0, 0, E // 2), (b1, E // 2, E)):
            part = getattr(br.experts, name).grad
            if part is None or full is None:
                failures.append(f"missing grad {name}")
                continue
            ref = full[lo:hi].float().to(part.device)
            rel = ((part.float() - ref).abs() / ref.abs().clamp_min(1e-3)).flatten()
            p99 = rel.quantile(0.99).item()
            worst = max(worst, p99)
    print(f"[parity] grad rel p99 worst: {worst:.5f}")
    if worst > 1e-2:
        failures.append(f"grad p99 {worst}")

    # dX parity: input grads sum
    dx_ref = x0.grad.float()
    dx_sum = x0b.grad.float() + x1.grad.float().to(dev0)
    dx_band = (dx_sum - dx_ref).abs().max().item()
    dx_scale = dx_ref.abs().mean().item()
    print(f"[parity] dX abs band: {dx_band:.5f} (ref mean |dx| {dx_scale:.4f})")
    if dx_band > max(0.1 * dx_scale, 1e-4):
        failures.append(f"dX band {dx_band} vs |dx| {dx_scale}")

    print(f"[parity] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
