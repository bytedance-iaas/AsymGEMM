#!/usr/bin/env python3
"""M4b — Grace-kernel case study (motivation_v2_plots.md).

Panel (a) deposit dA: MoE attention dA site (Qwen3-30B-A3B, d=2048,
rank 64). Routes per row-count in {64K, 256K, 1M}:
  CPU (ours)  = dS D2H + cpu_grouped_lora_a_grad_bf16 (SVE, 96T),
                fp32 dA lands in the pinned buffer (no dA transfer).
  CPU (stock) = dS D2H + torch CPU bf16 matmul (96T) + fp32 write.
  GPU route   = X restage H2D + cuBLAS bf16 dA + fp32 cast + dA D2H.
Panel (b) ingestion: 1 GB bf16 -> fp32 widen + sqsum.
  ours = cpu_widen_bf16_sqsum (96T); stock = copy_ + pow(2).sum().
Timing: host perf_counter around each full route with cuda.synchronize
fences (routes contain CPU work; CUDA events alone cannot time them).
Run with numactl --membind=0 (data on GPU0's Grace), threads spread.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import asym_gemm

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m4b.json"
D = 2048
RANK = 64
ROWS = [int(x) for x in os.environ.get("M4B_ROWS", "65536,262144,1048576").split(",")]
THREADS = 96
WARMUP_ITERS = 3
TIMED_ITERS = 10
RUNS = 3  # first discarded
ING_ELEMS = 1 << 29  # 512Mi bf16 = 1 GiB


def _time_route(fn) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / TIMED_ITERS * 1e3


def bench_rows(m: int) -> dict:
    torch.set_num_threads(THREADS)
    x_cpu = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
    x_cpu.normal_()
    ds_gpu = torch.randn((m, RANK), device="cuda", dtype=torch.bfloat16)
    ds_cpu = torch.empty((m, RANK), dtype=torch.bfloat16, pin_memory=True)
    out_pinned = torch.zeros((1, RANK, D), dtype=torch.float32, pin_memory=True)
    pr = torch.tensor([0, m], dtype=torch.long)
    ge = torch.zeros(1, dtype=torch.long)
    x_stage = torch.empty((m, D), device="cuda", dtype=torch.bfloat16)
    da_pin = torch.empty((RANK, D), dtype=torch.float32, pin_memory=True)

    def cpu_ours():
        ds_cpu.copy_(ds_gpu, non_blocking=True)
        torch.cuda.synchronize()
        out_pinned.zero_()
        asym_gemm.cpu_grouped_lora_a_grad_bf16(ds_cpu, x_cpu, out_pinned, pr, ge, THREADS)

    def cpu_stock():
        ds_cpu.copy_(ds_gpu, non_blocking=True)
        torch.cuda.synchronize()
        res = torch.matmul(ds_cpu.t(), x_cpu)  # bf16 CPU matmul, THREADS threads
        out_pinned[0].copy_(res.float())

    def gpu_route():
        x_stage.copy_(x_cpu, non_blocking=True)
        da = torch.matmul(ds_gpu.t(), x_stage).float()
        da_pin.copy_(da, non_blocking=True)
        torch.cuda.synchronize()

    # correctness spot-check vs fp32 reference (smallest shape only: the
    # stock bf16 matmul is itself the less-accurate side, so it cannot be
    # the reference)
    rel = 0.0
    if m == ROWS[0]:
        ref = torch.matmul(ds_gpu.t().float(), torch.as_tensor(x_cpu, device="cuda").float()).cpu()
        cpu_ours()
        rel = float((out_pinned[0] - ref).norm() / ref.norm())
        assert rel < 1e-2, f"deposit kernel mismatch vs fp32 ref rel={rel:.3e}"

    res = {"rows": m, "rel_check": float(rel), "runs": []}
    for _ in range(RUNS):
        res["runs"].append(
            {
                "cpu_ours_ms": _time_route(cpu_ours),
                "cpu_stock_ms": _time_route(cpu_stock),
                "gpu_route_ms": _time_route(gpu_route),
            }
        )
    del x_cpu, ds_gpu, ds_cpu, x_stage
    torch.cuda.empty_cache()
    return res


def bench_ingestion() -> dict:
    torch.set_num_threads(THREADS)
    src = torch.empty(ING_ELEMS, dtype=torch.bfloat16, pin_memory=True)
    src.normal_()
    dst = torch.empty(ING_ELEMS, dtype=torch.float32, pin_memory=True)

    def ours():
        asym_gemm.cpu_widen_bf16_sqsum(src, dst, THREADS)

    def stock():
        dst.copy_(src)
        float(dst.pow(2).sum())

    def _t(fn):
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        return (time.perf_counter() - t0) / 5 * 1e3

    res = {"elems": ING_ELEMS, "runs": []}
    for _ in range(RUNS):
        res["runs"].append({"ours_ms": _t(ours), "stock_ms": _t(stock)})
    return res


def main() -> None:
    torch.manual_seed(0)
    out = {
        "spec": {
            "site": "MoE attention dA (Qwen3-30B-A3B)",
            "d": D,
            "rank": RANK,
            "rows": ROWS,
            "threads": THREADS,
            "timed_iters": TIMED_ITERS,
            "device": torch.cuda.get_device_name(0),
        },
        "deposit": [bench_rows(m) for m in ROWS],
        "ingestion": bench_ingestion(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for sh in out["deposit"]:
        runs = sh["runs"][1:]
        mean = lambda k: sum(r[k] for r in runs) / len(runs)  # noqa: E731
        print(
            f"rows={sh['rows']:>8}: ours={mean('cpu_ours_ms'):8.2f}  "
            f"stock={mean('cpu_stock_ms'):8.2f}  gpu={mean('gpu_route_ms'):8.2f} ms"
        )
    ing = out["ingestion"]["runs"][1:]
    mo = sum(r["ours_ms"] for r in ing) / len(ing)
    ms = sum(r["stock_ms"] for r in ing) / len(ing)
    print(f"ingestion 1GiB: ours={mo:.1f} ms  stock={ms:.1f} ms  ({ms/mo:.1f}x)")


if __name__ == "__main__":
    main()
