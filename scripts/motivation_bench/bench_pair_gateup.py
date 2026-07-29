#!/usr/bin/env python3
"""Pair-LoRA salvage A/B — gate/up LoRA-A legs on the standard workload.

Standard workload (agent/impls/aymlora_kernels.md §0): Qwen3-30B-A3B MoE
(d=2048, E=128, top-8), seq 480K x batch 2 => 960K tokens, routed rows
M = 960K x 8 = 7.68M in ragged 128-aligned per-expert segments; X CPU
resident (pinned bf16, ~31.5 GB); gate/up LoRA-A weights [E, 64, 2048]
GPU-resident.

Variants
  fwd_single2  : grouped_expert_lora_cpu_left x2   (shipped default; X streamed twice)
  fwd_pair     : grouped_expert_lora_pair_cpu_left (native kPairOutput kernel; X once)
  fwd_cat      : cat(gate,up) -> single call n=128 -> split (ASYMM_..._PAIR_CAT path)
  grad_single2 : sm100_grouped_lora_a_grad_bf16_cpu_right x2 (X streamed twice)
  grad_pair    : sm100_grouped_lora_a_pair_grad_bf16_cpu_right (X once)
  h2d_slice    : plain pinned->HBM memcpy of a slice (raw link ceiling)

Timing: w1+m2 (3 runs, first discarded), per run WARMUP + TIMED iters
inside CUDA events, device sync around. JSON out with per-run means.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import asym_gemm  # noqa: F401
from asym_gemm.training import cpu_left as cpu_left_impl
from asym_gemm.training import exp_act_offload_lora as expact_impl

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/pair_gateup.json"

D = int(os.environ.get("PAIR_D", "2048"))
RANK = int(os.environ.get("PAIR_RANK", "64"))
EXPERTS = int(os.environ.get("PAIR_EXPERTS", "128"))
TOKENS = int(os.environ.get("PAIR_TOKENS", "960000"))
TOPK = int(os.environ.get("PAIR_TOPK", "8"))
ALIGN = 128  # PR-5 segment alignment contract (CPU_LEFT_GROUPED_BLOCK_M)

WARMUP_ITERS = int(os.environ.get("PAIR_WARMUP", "2"))
TIMED_ITERS = int(os.environ.get("PAIR_TIMED", "5"))
RUNS = int(os.environ.get("PAIR_RUNS", "3"))  # first run discarded (w1+m2)
VARIANTS = os.environ.get(
    "PAIR_VARIANTS", "fwd_single2,fwd_pair,fwd_cat,grad_single2,grad_pair,h2d_slice"
).split(",")
SEED = int(os.environ.get("PAIR_SEED", "0"))


def make_segments(total_rows: int, experts: int, gen: torch.Generator) -> list[int]:
    """Ragged 128-aligned per-expert row counts summing to ~total_rows."""
    probs = torch.full((experts,), 1.0 / experts, dtype=torch.float64)
    counts = torch.multinomial(
        probs, num_samples=total_rows, replacement=True, generator=gen
    ).bincount(minlength=experts)
    aligned = (counts // ALIGN) * ALIGN
    leftover = int(total_rows - int(aligned.sum()))
    order = torch.randperm(experts, generator=gen)
    i = 0
    while leftover >= ALIGN:
        aligned[order[i % experts]] += ALIGN
        leftover -= ALIGN
        i += 1
    return [int(v) for v in aligned]


def _time_loop(fn) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED_ITERS


def main() -> None:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    dev = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)

    lengths = make_segments(TOKENS * TOPK, EXPERTS, gen)
    m = sum(lengths)
    print(f"[setup] routed rows M={m} over {EXPERTS} experts "
          f"(min={min(lengths)}, max={max(lengths)}), d={D}, r={RANK}", flush=True)

    offsets = [0]
    for rows in lengths:
        offsets.append(offsets[-1] + rows)
    offsets_t = torch.tensor(offsets, device=dev, dtype=torch.int32)
    experts_t = torch.tensor(list(range(EXPERTS)) + [-1], device=dev, dtype=torch.int32)

    x_bytes = m * D * 2
    print(f"[setup] allocating pinned X: {x_bytes / 1e9:.2f} GB", flush=True)
    x_pinned = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
    chunk = 1 << 20
    for i in range(0, m, chunk):
        x_pinned[i : i + chunk].normal_()
    print("[setup] X filled", flush=True)

    gate_a = torch.randn((EXPERTS, RANK, D), device=dev, dtype=torch.bfloat16)
    up_a = torch.randn((EXPERTS, RANK, D), device=dev, dtype=torch.bfloat16)
    ds_gate = torch.randn((m, RANK), device=dev, dtype=torch.bfloat16)
    ds_up = torch.randn((m, RANK), device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # ---------------- variant closures (mirror production call paths) --------
    def fwd_single2():
        g = cpu_left_impl.grouped_expert_lora_cpu_left(x_pinned, gate_a, offsets_t, experts_t)
        u = cpu_left_impl.grouped_expert_lora_cpu_left(x_pinned, up_a, offsets_t, experts_t)
        return g, u

    def fwd_pair():
        return cpu_left_impl.grouped_expert_lora_pair_cpu_left(
            x_pinned, gate_a, up_a, offsets_t, experts_t
        )

    def fwd_cat():
        gate_up_a = torch.cat((gate_a, up_a), dim=1).contiguous()
        gu = cpu_left_impl.grouped_expert_lora_cpu_left(x_pinned, gate_up_a, offsets_t, experts_t)
        g, u = gu.split(RANK, dim=-1)
        return g.contiguous(), u.contiguous()

    def grad_single2():
        gg = expact_impl.grouped_lora_a_grad_cpu_right(
            ds_gate, x_pinned, offsets_t, experts_t,
            num_experts=EXPERTS, stats=None, tag="bench.gate")
        gu = expact_impl.grouped_lora_a_grad_cpu_right(
            ds_up, x_pinned, offsets_t, experts_t,
            num_experts=EXPERTS, stats=None, tag="bench.up")
        return gg, gu

    def grad_pair():
        return expact_impl.grouped_lora_a_pair_grad_cpu_right(
            ds_gate, ds_up, x_pinned, offsets_t, experts_t,
            num_experts=EXPERTS, stats=None)

    slice_rows = min(m, (1 << 30) // (D * 2))  # ~1 GiB slice
    stage_buf = torch.empty((slice_rows, D), device=dev, dtype=torch.bfloat16)

    def h2d_slice():
        stage_buf.copy_(x_pinned[:slice_rows], non_blocking=True)

    fns = {
        "fwd_single2": fwd_single2,
        "fwd_pair": fwd_pair,
        "fwd_cat": fwd_cat,
        "grad_single2": grad_single2,
        "grad_pair": grad_pair,
        "h2d_slice": h2d_slice,
    }
    streamed_bytes = {
        "fwd_single2": 2 * x_bytes,
        "fwd_pair": x_bytes,
        "fwd_cat": x_bytes,
        "grad_single2": 2 * x_bytes,
        "grad_pair": x_bytes,
        "h2d_slice": slice_rows * D * 2,
    }

    # ---------------- correctness (once, before timing) ----------------------
    print("[check] correctness spot-checks", flush=True)
    checks: dict[str, float] = {}

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))

    g1, u1 = fwd_single2()
    torch.cuda.synchronize()
    if "fwd_pair" in VARIANTS:
        g2, u2 = fwd_pair()
        torch.cuda.synchronize()
        checks["fwd_pair_vs_single_gate"] = rel(g2, g1)
        checks["fwd_pair_vs_single_up"] = rel(u2, u1)
        del g2, u2
    if "fwd_cat" in VARIANTS:
        g3, u3 = fwd_cat()
        torch.cuda.synchronize()
        checks["fwd_cat_vs_single_gate"] = rel(g3, g1)
        checks["fwd_cat_vs_single_up"] = rel(u3, u1)
        del g3, u3
    # torch reference on 3 segments (largest / smallest / median)
    order = sorted(range(EXPERTS), key=lambda e: lengths[e])
    for label, e in (("small", order[0]), ("mid", order[EXPERTS // 2]), ("large", order[-1])):
        s, t = offsets[e], offsets[e + 1]
        if t <= s:
            continue
        xg = x_pinned[s:t].to(dev).float()
        ref = xg.matmul(gate_a[e].float().t())
        checks[f"fwd_single_vs_torch_{label}"] = rel(g1[s:t], ref.to(torch.bfloat16))
        del xg, ref
    del g1, u1

    gg1, gu1 = grad_single2()
    torch.cuda.synchronize()
    if "grad_pair" in VARIANTS:
        gg2, gu2 = grad_pair()
        torch.cuda.synchronize()
        checks["grad_pair_vs_single_gate"] = rel(gg2, gg1)
        checks["grad_pair_vs_single_up"] = rel(gu2, gu1)
        del gg2, gu2
    e = order[-1]
    s, t = offsets[e], offsets[e + 1]
    xg = x_pinned[s:t].to(dev).float()
    ref_da = ds_gate[s:t].float().t().matmul(xg)
    checks["grad_single_vs_torch_large"] = rel(gg1[e], ref_da.to(torch.bfloat16))
    del xg, ref_da, gg1, gu1
    torch.cuda.empty_cache()
    for k, v in sorted(checks.items()):
        print(f"[check] {k}: {v:.3e}", flush=True)

    # ---------------- timing --------------------------------------------------
    results: dict[str, dict] = {}
    for name in VARIANTS:
        fn = fns[name]
        runs = []
        for r in range(RUNS):
            ms = _time_loop(fn)
            runs.append(ms)
            print(f"[time] {name} run{r}: {ms:.3f} ms", flush=True)
        measured = runs[1:]
        mean_ms = sum(measured) / len(measured)
        gbps = streamed_bytes[name] / 1e9 / (mean_ms / 1e3)
        results[name] = {
            "runs_ms": runs,
            "mean_ms": mean_ms,
            "streamed_gb": streamed_bytes[name] / 1e9,
            "eff_gbps": gbps,
        }
        print(f"[time] {name}: {mean_ms:.3f} ms  ({gbps:.1f} GB/s streamed)", flush=True)
        torch.cuda.empty_cache()

    out = {
        "spec": {
            "site": "MoE gate/up LoRA-A fwd + dA (pair salvage A/B)",
            "model": "Qwen3-30B-A3B-like (d=2048, E=128, top-8)",
            "tokens": TOKENS, "topk": TOPK, "routed_rows": m,
            "rank": RANK, "d": D,
            "segments_min": min(lengths), "segments_max": max(lengths),
            "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS, "runs": RUNS,
            "device": dev_name,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("DG_", "PAIR_", "ASYMM_"))},
        },
        "checks": checks,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)

    if "fwd_single2" in results and "fwd_pair" in results:
        sp = results["fwd_single2"]["mean_ms"] / results["fwd_pair"]["mean_ms"]
        print(f"[summary] fwd pair speedup vs single2: {sp:.3f}x")
    if "grad_single2" in results and "grad_pair" in results:
        sp = results["grad_single2"]["mean_ms"] / results["grad_pair"]["mean_ms"]
        print(f"[summary] grad pair speedup vs single2: {sp:.3f}x")


if __name__ == "__main__":
    main()
