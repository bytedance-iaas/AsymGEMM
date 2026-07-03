"""Numeric probe: AsymQwen3Experts fine-grained path at Qwen3.5-35B-A3B shapes.

Compares, at E=256/H=2048/I=512/top_k=8 (and optionally qwen3-30b shapes):
  ref     : plain asym whole-router path (fg disabled)  -- known-good at s45000
  fg000   : fine-grained path, route kernels off (ker000)
  fg101   : fine-grained path, fwd_scatter=1 + gateup_dx_scatter=1 (ker101)
against a pure-torch fp32 reference, forward + backward (input grad + LoRA grads).

Usage: CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py [--qwen3]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asym_gemm.training.qwen3_moe import wrap_qwen3_experts


class FakeExperts(nn.Module):
    def __init__(self, *, num_experts: int, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = F.silu
        g = torch.Generator().manual_seed(1234)
        self.gate_up_proj = nn.Parameter(
            (torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, generator=g) * 0.02).to(torch.bfloat16)
        )
        self.down_proj = nn.Parameter(
            (torch.randn(num_experts, hidden_dim, intermediate_dim, generator=g) * 0.02).to(torch.bfloat16)
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("not used")


def torch_reference(x, top_k_index, top_k_weights, gate_up, down, loras, scale):
    """fp32 dense loop reference with LoRA, output-weighted."""
    T, H = x.shape
    E = gate_up.shape[0]
    I = down.shape[2]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    ga, gb, ua, ub, da, db = [t.float() for t in loras]
    xf = x.float()
    for e in range(E):
        pos, slot = torch.where(top_k_index == e)
        if pos.numel() == 0:
            continue
        rows = xf[pos]
        w = gate_up[e].float()
        gate = rows @ w[:I].T + (rows @ ga[e].T) @ gb[e].T * scale
        up = rows @ w[I:].T + (rows @ ua[e].T) @ ub[e].T * scale
        act = F.silu(gate) * up
        y = act @ down[e].float().T + (act @ da[e].T) @ db[e].T * scale
        out.index_add_(0, pos, y * top_k_weights[pos, slot].float().unsqueeze(1))
    return out


def stats(tag, got, ref):
    got = got.float()
    ref = ref.float()
    nan = int(torch.isnan(got).sum())
    diff = (got - ref).abs()
    rel = diff.norm() / ref.norm().clamp_min(1e-12)
    print(
        f"  {tag:34s} max|d|={diff.max().item():12.6f} rel_fro={rel.item():10.6f} "
        f"nan={nan} ref_absmax={ref.abs().max().item():10.4f} got_absmax={got.abs().max().item():10.4f}"
    )
    return rel.item(), nan


def run_case(engine, x0, idx, w, grad_out, label):
    x = x0.detach().clone().requires_grad_(True)
    for p in engine.parameters():
        if p.grad is not None:
            p.grad = None
    out = engine(x, idx, w)
    out.backward(grad_out)
    grads = {
        "dx": x.grad.detach().clone(),
        **{
            n: p.grad.detach().clone()
            for n, p in engine.named_parameters()
            if p.requires_grad and p.grad is not None
        },
    }
    return out.detach().clone(), grads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen3", action="store_true", help="use qwen3-30b shapes (E=128,I=768) instead of qwen3.5-35b")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--zero-b", action="store_true", help="zero-init LoRA-B (step-1 condition)")
    args = ap.parse_args()

    torch.manual_seed(7)
    dev = torch.device("cuda:0")
    E, H, I, K = (128, 2048, 768, 8) if args.qwen3 else (256, 2048, 512, 8)
    T = args.tokens
    print(f"shapes: E={E} H={H} I={I} top_k={K} T={T} zero_b={args.zero_b}")

    src = FakeExperts(num_experts=E, hidden_dim=H, intermediate_dim=I)
    gate_up_f32 = src.gate_up_proj.detach().float().clone()
    down_f32 = src.down_proj.detach().float().clone()

    engine = wrap_qwen3_experts(
        src,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=64,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_dtype=torch.bfloat16,
        expert_recompute_policy="none",
        stats=None,
        strict=True,
    )
    engine.train()

    with torch.no_grad():
        for bank in (engine.gate_lora_A, engine.up_lora_A, engine.down_lora_A):
            bank.normal_(0, 0.02)
        for bank in (engine.gate_lora_B, engine.up_lora_B, engine.down_lora_B):
            if args.zero_b:
                bank.zero_()
            else:
                bank.normal_(0, 0.02)
    engine = engine.to(dev) if any(p.device.type != "cuda" for p in engine.parameters()) else engine

    x0 = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    logits = torch.randn(T, E, device=dev)
    topw, idx = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    topw = (topw / topw.sum(-1, keepdim=True)).to(torch.bfloat16)
    idx = idx.to(torch.long)
    grad_out = (torch.randn(T, H, device=dev) * 0.1).to(torch.bfloat16)

    loras = [
        engine.gate_lora_A.detach(),
        engine.gate_lora_B.detach(),
        engine.up_lora_A.detach(),
        engine.up_lora_B.detach(),
        engine.down_lora_A.detach(),
        engine.down_lora_B.detach(),
    ]
    ref_out = torch_reference(x0, idx, topw, gate_up_f32.to(dev), down_f32.to(dev), loras, engine.lora_scale)

    def set_route(fwd, gather, dx):
        os.environ["ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER"] = str(fwd)
        os.environ["ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER"] = str(gather)
        os.environ["ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER"] = str(dx)
        os.environ["ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE"] = "fp32"

    results = {}

    engine._qwen3_moe_finegrained_enabled = False
    set_route(0, 0, 0)
    results["plain"] = run_case(engine, x0, idx, topw, grad_out, "plain")

    engine._qwen3_moe_finegrained_enabled = True
    set_route(0, 0, 0)
    results["fg000"] = run_case(engine, x0, idx, topw, grad_out, "fg000")

    set_route(1, 0, 1)
    results["fg101"] = run_case(engine, x0, idx, topw, grad_out, "fg101")

    fg_env = {
        "ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU": "1",
        "ASYMM_QWEN3_MOE_FG_DA_GPU": "1",
        "ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM": "1",
    }
    os.environ.update(fg_env)
    results["fg101+v2"] = run_case(engine, x0, idx, topw, grad_out, "fg101+v2")
    set_route(0, 0, 0)
    results["fg000+v2"] = run_case(engine, x0, idx, topw, grad_out, "fg000+v2")
    for k in fg_env:
        os.environ[k] = "0"

    print("\n== forward vs fp32 torch reference ==")
    worst = {}
    for name, (out, _grads) in results.items():
        rel, nan = stats(f"{name} out", out, ref_out)
        worst[name] = (rel, nan)

    print("\n== per-bank grad norms (explosion localizer) ==")
    for name, (_out, grads) in results.items():
        norms = " ".join(f"{g}={grads[g].float().norm().item():.3e}" for g in sorted(grads))
        print(f"  {name:10s} {norms}")

    print("\n== forward/grads vs plain-asym path (known good) ==")
    p_out, p_grads = results["plain"]
    for name, (out, grads) in results.items():
        if name == "plain":
            continue
        stats(f"{name} out~plain", out, p_out)
        for gname in sorted(p_grads):
            if gname in grads:
                stats(f"{name} {gname}", grads[gname], p_grads[gname])

    bad = {n: v for n, v in worst.items() if v[0] > 0.05 or v[1] > 0}
    print(f"\nverdict: {'BROKEN: ' + ', '.join(sorted(bad)) if bad else 'all forward paths within 5% of fp32 reference'}")


if __name__ == "__main__":
    main()
