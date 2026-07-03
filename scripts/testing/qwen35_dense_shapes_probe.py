"""Numeric probe: asym dense-path modules at Qwen3.5-27B shapes.

Covers, against an fp32 torch reference (fwd out, dx, lora A/B grads):
  1. plain AsymLoRALinear at every 27B projection shape (+qwen2.5-32b controls)
  2. AsymActivationOffloadLoRALinear at the full-attention shapes
  3. AsymFinegrainedDenseMLP (5120 -> 17408) plain vs fine-grained env path

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_dense_shapes_probe.py
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asym_gemm.training.lora import AsymLoRALinear
from asym_gemm.training.attention_activation_offload import AsymActivationOffloadLoRALinear
from asym_gemm.training.dense_mlp_finegrained import build_finegrained_dense_mlp

DEV = torch.device("cuda:0")
T = 4096
RANK, ALPHA = 64, 16.0
FAILS: list[str] = []


def report(tag: str, got: torch.Tensor, ref: torch.Tensor, tol: float = 0.05) -> None:
    got = got.detach().float()
    ref = ref.detach().float()
    nan = int(torch.isnan(got).sum())
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    flag = "OK " if (rel <= tol and nan == 0) else "BAD"
    if flag == "BAD":
        FAILS.append(tag)
    print(f"  [{flag}] {tag:44s} rel_fro={rel:9.5f} nan={nan} ref_absmax={ref.abs().max():8.4f} got_absmax={got.abs().max():8.4f}")


def ref_lora(x, w, a, b, scale):
    xf = x.float()
    return xf @ w.float().t() + (xf @ a.float().t()) @ b.float().t() * scale


def run_linear(kind: str, out_f: int, in_f: int) -> None:
    torch.manual_seed(11)
    w = (torch.randn(out_f, in_f) * 0.02).to(torch.bfloat16)
    if kind == "attnoff":
        layer = AsymActivationOffloadLoRALinear(w.clone(), rank=RANK, alpha=ALPHA, backend="asym", device=DEV, lora_dtype=torch.bfloat16, precision="bf16")
    else:
        layer = AsymLoRALinear(w.clone(), rank=RANK, alpha=ALPHA, backend="asym", device=DEV, dtype=torch.bfloat16, lora_dtype=torch.bfloat16, precision="bf16")
    layer.train()
    a_mod = layer.lora_A[layer.active_adapter] if hasattr(layer, "lora_A") else layer.lora_a
    b_mod = layer.lora_B[layer.active_adapter] if hasattr(layer, "lora_B") else layer.lora_b
    a = a_mod.weight if isinstance(a_mod, nn.Module) else a_mod
    b = b_mod.weight if isinstance(b_mod, nn.Module) else b_mod
    with torch.no_grad():
        a.normal_(0, 0.02)
        b.normal_(0, 0.02)
    scale = float(layer.scaling) if hasattr(layer, "scaling") else float(layer.lora_scale)

    x = (torch.randn(T, in_f, device=DEV) * 0.5).to(torch.bfloat16).requires_grad_(True)
    y = layer(x)
    ref_y = ref_lora(x, w.to(DEV), a, b, scale)
    report(f"{kind} ({out_f},{in_f}) fwd", y, ref_y)

    g = (torch.randn(T, out_f, device=DEV) * 0.1).to(torch.bfloat16)
    y.backward(g)
    gf = g.float()
    xf = x.detach().float()
    ref_dx = gf @ w.to(DEV).float() + (gf @ b.float()) @ a.float() * scale
    ref_db = gf.t() @ (xf @ a.float().t()) * scale
    ref_da = (b.float().t() @ gf.t() @ xf) * scale
    report(f"{kind} ({out_f},{in_f}) dx", x.grad, ref_dx)
    report(f"{kind} ({out_f},{in_f}) dB", b.grad, ref_db)
    report(f"{kind} ({out_f},{in_f}) dA", a.grad, ref_da)


class FakeDenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def run_dense_mlp(hidden: int, inter: int, fg_env: bool) -> None:
    torch.manual_seed(23)
    src = FakeDenseMLP(hidden, inter)
    with torch.no_grad():
        for m in (src.gate_proj, src.up_proj, src.down_proj):
            m.weight.normal_(0, 0.02)
    wg = src.gate_proj.weight.detach().clone()
    wu = src.up_proj.weight.detach().clone()
    wd = src.down_proj.weight.detach().clone()

    os.environ["ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD"] = "1" if fg_env else "0"
    mlp = build_finegrained_dense_mlp(src, backend="asym", precision="bf16", lora_rank=RANK, lora_alpha=ALPHA, lora_dropout=0.0, strict=True)
    mlp.train()
    loras = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(mlp, name)
        a = mod.lora_a if hasattr(mod, "lora_a") else mod.lora_A[mod.active_adapter].weight
        b = mod.lora_b if hasattr(mod, "lora_b") else mod.lora_B[mod.active_adapter].weight
        with torch.no_grad():
            a.normal_(0, 0.02)
            b.normal_(0, 0.02)
        loras[name] = (a, b)
    scale = ALPHA / RANK

    x = (torch.randn(T, hidden, device=DEV) * 0.5).to(torch.bfloat16).requires_grad_(True)
    y = mlp(x)

    def branch(xf, w, ab):
        a, b = ab
        return xf @ w.to(DEV).float().t() + (xf @ a.float().t()) @ b.float().t() * scale

    xf = x.detach().float()
    gate = branch(xf, wg, loras["gate_proj"])
    up = branch(xf, wu, loras["up_proj"])
    act = F.silu(gate) * up
    ref_y = branch(act, wd, loras["down_proj"])
    tag = f"denseMLP({hidden}->{inter}) fg_env={int(fg_env)}"
    report(f"{tag} fwd", y, ref_y)

    g = (torch.randn(T, hidden, device=DEV) * 0.1).to(torch.bfloat16)
    y.backward(g)
    # reference dx via autograd on the fp32 graph
    x2 = xf.clone().requires_grad_(True)
    gate2 = branch(x2, wg, loras["gate_proj"])
    up2 = branch(x2, wu, loras["up_proj"])
    y2 = branch(F.silu(gate2) * up2, wd, loras["down_proj"])
    y2.backward(g.float())
    report(f"{tag} dx", x.grad, x2.grad)
    for name in ("gate_proj", "up_proj", "down_proj"):
        a, b = loras[name]
        if a.grad is not None:
            print(f"      {name} dA_norm={a.grad.float().norm():.4e} dB_norm={b.grad.float().norm():.4e}")


def main() -> None:
    print(f"== plain AsymLoRALinear, qwen3.5-27b shapes (T={T}) ==")
    for out_f, in_f in [(6144, 5120), (1024, 5120), (5120, 6144), (10240, 5120), (17408, 5120), (5120, 17408), (27648, 5120), (5120, 27648)]:
        run_linear("plain", out_f, in_f)
    print("== AsymActivationOffloadLoRALinear, attention shapes ==")
    os.environ["ASYMM_ATTN_ACT_OFFLOAD"] = "1"
    for out_f, in_f in [(6144, 5120), (1024, 5120), (5120, 6144)]:
        run_linear("attnoff", out_f, in_f)
    print("== AsymFinegrainedDenseMLP 5120->17408 ==")
    run_dense_mlp(5120, 17408, fg_env=False)
    run_dense_mlp(5120, 17408, fg_env=True)
    print("== control: qwen2.5-32b MLP 5120->27648 ==")
    run_dense_mlp(5120, 27648, fg_env=True)
    print(f"\nverdict: {'BROKEN: ' + '; '.join(FAILS) if FAILS else 'all dense-path modules match fp32 reference'}")


if __name__ == "__main__":
    main()
