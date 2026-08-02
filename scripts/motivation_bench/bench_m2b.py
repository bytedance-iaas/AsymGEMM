#!/usr/bin/env python3
"""M2b — scatter fusion (motivation_v2_plots.md).

Qwen3-30B-A3B routed expert GEMMs, both shipped fused legs:
  fwd  : down_forward_scatter_add_  vs unfused (grouped GEMM -> weight
         mul -> index_add_ scatter pass, expert-ordered fp32 output
         materialized)
  bwd  : gateup_dx_scatter_add_     vs unfused analogue
Both executions stream the expert weights from pinned host memory
(the comparison isolates the fusion, not the streaming).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from asym_gemm.training.frozen_linear import (
    AsymExecutionStats,
    AsymGroupedFrozenLinear,
    _asym_grouped_bf16_nt,
)
from asym_gemm.training.qwen3_moe_routed_gemm import (
    down_forward_scatter_add_,
    gateup_dx_scatter_add_,
)

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m2b.json"
TOKENS = 131072          # 128K tokens pre-top-k
TOPK = 8
EXPERTS = 128
HIDDEN = 2048            # Qwen3-30B-A3B hidden
INTER = 768              # per-expert intermediate
ZIPF_Z = 1.0
WARMUP_ITERS = 10
TIMED_ITERS = 50
RUNS = 3  # first discarded


def _route(seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, EXPERTS + 1, dtype=torch.float64)
    probs = (ranks ** -ZIPF_Z)
    probs /= probs.sum()
    rows = TOKENS * TOPK
    expert_of_row = torch.multinomial(probs, rows, replacement=True, generator=g)
    order = torch.argsort(expert_of_row, stable=True)
    expert_sorted = expert_of_row[order]
    token_of_row = (torch.arange(rows) // TOPK)[order]
    counts = torch.bincount(expert_sorted, minlength=EXPERTS)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))).cuda()
    experts = torch.cat((torch.arange(EXPERTS, dtype=torch.long), torch.tensor([-1]))).cuda()
    token_indices = token_of_row.contiguous().cuda()
    routing_weights = torch.rand(rows, generator=g).to(torch.bfloat16).cuda()
    return offsets, experts, token_indices, routing_weights, rows


def _base(shape) -> AsymGroupedFrozenLinear:
    w = torch.randn(shape, dtype=torch.bfloat16)
    return AsymGroupedFrozenLinear(
        w.pin_memory(), backend="asym", pin_memory=True,
        stats=AsymExecutionStats(), precision="bf16",
    )


def _time_loop(fn) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(TIMED_ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / TIMED_ITERS


def main() -> None:
    torch.manual_seed(0)
    offsets, experts, token_indices, routing_weights, rows = _route()
    n_tok = TOKENS

    down_base = _base((EXPERTS, HIDDEN, INTER))
    gate_base = _base((EXPERTS, INTER, HIDDEN))

    act = torch.randn((rows, INTER), device="cuda", dtype=torch.bfloat16)
    grad_expert = torch.randn((rows, INTER), device="cuda", dtype=torch.bfloat16)
    out_fwd = torch.zeros((n_tok, HIDDEN), device="cuda", dtype=torch.float32)
    out_bwd = torch.zeros((n_tok, HIDDEN), device="cuda", dtype=torch.float32)

    def fwd_unfused():
        route_out = _asym_grouped_bf16_nt(
            act, down_base.host_weight.weight, offsets, experts,
            compiled_dims="nk", transpose_b=False, output_dtype=torch.float32,
        )  # expert-ordered output copy materialized
        route_out.mul_(routing_weights.reshape(-1, 1).float())
        out_fwd.zero_()
        out_fwd.index_add_(0, token_indices, route_out)

    def fwd_fused():
        out_fwd.zero_()
        down_forward_scatter_add_(
            down_base, act, out_fwd, offsets, experts,
            token_indices, routing_weights, weighted=True,
        )

    def bwd_unfused():
        route_out = _asym_grouped_bf16_nt(
            grad_expert, gate_base.host_weight.weight, offsets, experts,
            compiled_dims="nk", transpose_b=True, output_dtype=torch.float32,
        )
        route_out.mul_(routing_weights.reshape(-1, 1).float())
        out_bwd.zero_()
        out_bwd.index_add_(0, token_indices, route_out)

    def bwd_fused():
        out_bwd.zero_()
        gateup_dx_scatter_add_(
            gate_base, grad_expert, out_bwd, offsets, experts,
            token_indices, routing_weights, weighted=True,
        )

    # correctness spot-check
    fwd_unfused(); ref = out_fwd.clone()
    fwd_fused(); torch.cuda.synchronize()
    rel = (out_fwd - ref).norm() / ref.norm()
    assert rel < 2e-2, f"fused fwd mismatch rel={rel:.3e}"
    bwd_unfused(); refb = out_bwd.clone()
    bwd_fused(); torch.cuda.synchronize()
    relb = (out_bwd - refb).norm() / refb.norm()
    assert relb < 2e-2, f"fused bwd mismatch rel={relb:.3e}"

    res = {"spec": {
        "model": "Qwen3-30B-A3B", "tokens": TOKENS, "topk": TOPK,
        "experts": EXPERTS, "hidden": HIDDEN, "inter": INTER,
        "zipf_z": ZIPF_Z, "routed_rows": rows,
        "timed_iters": TIMED_ITERS,
        "device": torch.cuda.get_device_name(0),
        "rel_check": float(rel),
    }, "runs": []}
    for _ in range(RUNS):
        res["runs"].append({
            "fwd_unfused_ms": _time_loop(fwd_unfused),
            "fwd_fused_ms": _time_loop(fwd_fused),
            "bwd_unfused_ms": _time_loop(bwd_unfused),
            "bwd_fused_ms": _time_loop(bwd_fused),
        })
    # prose numbers: the expert-ordered output copy the unfused pass holds
    res["copy_bytes_gb"] = rows * HIDDEN * 4 / 1e9  # fp32 route_out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    runs = res["runs"][1:]
    mean = lambda k: sum(r[k] for r in runs) / len(runs)  # noqa: E731
    print(f"fwd: unfused={mean('fwd_unfused_ms'):8.3f} ms  fused={mean('fwd_fused_ms'):8.3f} ms")
    print(f"bwd: unfused={mean('bwd_unfused_ms'):8.3f} ms  fused={mean('bwd_fused_ms'):8.3f} ms")
    print(f"expert-ordered output copy: {res['copy_bytes_gb']:.2f} GB")


if __name__ == "__main__":
    main()
