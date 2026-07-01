#!/usr/bin/env python3
"""Focused microbench for Qwen3 MoE routed AsymGEMM kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymGroupedFrozenLinear
from asym_gemm.training.qwen3_moe_routed_gemm import (
    down_dx_gather_left,
    down_forward_scatter_add_,
    gateup_dx_scatter_add_,
)


def _base(weight: torch.Tensor) -> AsymGroupedFrozenLinear:
    return AsymGroupedFrozenLinear(
        weight.detach().cpu().pin_memory(),
        backend="asym",
        pin_memory=True,
        stats=AsymExecutionStats(),
        compiled_dims="nk",
        precision="bf16",
    )


def _metadata(route_rows: int, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = route_rows // num_experts
    rem = route_rows % num_experts
    counts = [base + (1 if i < rem else 0) for i in range(num_experts)]
    offsets = torch.tensor([0] + counts, device="cuda", dtype=torch.long).cumsum(0)
    experts = torch.tensor(list(range(num_experts)) + [-1], device="cuda", dtype=torch.long)
    return offsets.contiguous(), experts.contiguous()


def _events_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(int(iters), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["fwd_scatter", "down_dx_gather", "gateup_dx_scatter", "all"], default="all")
    parser.add_argument("--route-rows", type=int, default=8192)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10:
        raise SystemExit("Qwen3 routed kernels require SM100 CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(args.seed))
    offsets, experts = _metadata(int(args.route_rows), int(args.num_experts))
    route_rows = int(offsets[-1].item())
    num_tokens = int(args.num_tokens)
    token_indices = torch.randint(0, num_tokens, (route_rows,), device="cuda", dtype=torch.long).contiguous()
    routing_weights = torch.rand((route_rows,), device="cuda", dtype=torch.bfloat16).contiguous()
    if not args.weighted:
        routing_weights = torch.empty((0,), device="cuda", dtype=torch.bfloat16)

    down_weight = torch.randn((int(args.num_experts), int(args.hidden), int(args.intermediate)), dtype=torch.bfloat16)
    gate_weight = torch.randn((int(args.num_experts), int(args.intermediate), int(args.hidden)), dtype=torch.bfloat16)
    down_base = _base(down_weight)
    gate_base = _base(gate_weight)
    act = torch.randn((route_rows, int(args.intermediate)), device="cuda", dtype=torch.bfloat16)
    grad_token = torch.randn((num_tokens, int(args.hidden)), device="cuda", dtype=torch.bfloat16)
    grad_expert = torch.randn((route_rows, int(args.intermediate)), device="cuda", dtype=torch.bfloat16)

    results: dict[str, float | int | bool] = {
        "route_rows": route_rows,
        "num_tokens": num_tokens,
        "hidden": int(args.hidden),
        "intermediate": int(args.intermediate),
        "num_experts": int(args.num_experts),
        "weighted": bool(args.weighted),
    }

    def run_fwd():
        out = torch.zeros((num_tokens, int(args.hidden)), device="cuda", dtype=torch.float32)
        down_forward_scatter_add_(
            down_base,
            act,
            out,
            offsets,
            experts,
            token_indices,
            routing_weights if args.weighted else None,
            weighted=bool(args.weighted),
        )
        return out

    def run_down_dx():
        return down_dx_gather_left(
            down_base,
            grad_token,
            (route_rows, int(args.intermediate)),
            offsets,
            experts,
            token_indices,
            routing_weights if args.weighted else None,
            weighted=bool(args.weighted),
        )

    def run_gateup_dx():
        out = torch.zeros((num_tokens, int(args.hidden)), device="cuda", dtype=torch.float32)
        gateup_dx_scatter_add_(
            gate_base,
            grad_expert,
            out,
            offsets,
            experts,
            token_indices,
            routing_weights if args.weighted else None,
            weighted=bool(args.weighted),
        )
        return out

    kernels = []
    if args.kernel in {"fwd_scatter", "all"}:
        kernels.append(("fwd_scatter", run_fwd))
    if args.kernel in {"down_dx_gather", "all"}:
        kernels.append(("down_dx_gather", run_down_dx))
    if args.kernel in {"gateup_dx_scatter", "all"}:
        kernels.append(("gateup_dx_scatter", run_gateup_dx))

    for name, fn in kernels:
        t0 = time.perf_counter()
        ms = _events_time_ms(fn, int(args.warmup), int(args.iters))
        results[f"{name}_ms"] = ms
        results[f"{name}_wall_s"] = float(time.perf_counter() - t0)

    results.update(down_base.stats.as_dict())
    for key, value in gate_base.stats.as_dict().items():
        results[f"gate_{key}"] = value

    out_file = args.output_dir / "profile.json"
    out_file.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
