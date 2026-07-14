#!/usr/bin/env python3
"""Phase D2 engine-tax microbench (agent/impls/fix_throughput.md §Phase D).

Measures the SAME GEMM three ways at the q3-30b flagship shapes:
  asym     — sm100 asym kernel, weight pinned in CPU DRAM (in-kernel C2C streaming)
  staged   — H2D copy of the pinned weight, then native torch mm (cuBLAS/nvjet);
             copy and mm timed separately (double-buffer steady state ~= mm-only)
  resident — weight already in HBM, native torch mm (the floor)

No training, no model load. Run only when the node is otherwise idle (serial rule).

Usage: python3 scripts/gemm/bench_engine_tax.py [--iters 10] [--warmup 3] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys

import torch

from asym_gemm.training.frozen_linear import (
    _asym_bf16_nt,
    _asym_grouped_bf16_nt,
    _grouped_torch_mm,
    _direct_bf16_reason,
    _direct_grouped_bf16_reason,
)


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def bench_dense(name: str, m: int, n: int, k: int, *, transpose_b: bool,
                warmup: int, iters: int) -> list[dict]:
    dev = torch.device("cuda")
    a = torch.randn((m, k) if not transpose_b else (m, n), device=dev, dtype=torch.bfloat16)
    b_cpu = torch.randn((n, k), dtype=torch.bfloat16).pin_memory()
    b_gpu = b_cpu.to(dev)
    flops = 2.0 * m * n * k
    weight_bytes = b_cpu.numel() * b_cpu.element_size()
    rows: list[dict] = []

    reason = _direct_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is None:
        ms = _time_cuda(lambda: _asym_bf16_nt(a, b_cpu, transpose_b=transpose_b),
                        warmup=warmup, iters=iters)
        rows.append(dict(shape=name, engine="asym", ms=ms, tflops=_tflops(flops, ms)))
    else:
        rows.append(dict(shape=name, engine="asym", ms=float("nan"), tflops=0.0, note=reason))

    def _mm(bmat):
        return a @ bmat if transpose_b else a @ bmat.t()

    ms_res = _time_cuda(lambda: _mm(b_gpu), warmup=warmup, iters=iters)
    rows.append(dict(shape=name, engine="resident", ms=ms_res, tflops=_tflops(flops, ms_res)))

    stage_buf = torch.empty_like(b_gpu)
    ms_copy = _time_cuda(lambda: stage_buf.copy_(b_cpu, non_blocking=True),
                         warmup=warmup, iters=iters)
    ms_staged = ms_copy + ms_res
    rows.append(dict(shape=name, engine="staged(copy+mm)", ms=ms_staged,
                     tflops=_tflops(flops, ms_staged),
                     note=f"copy {ms_copy:.2f} ms = {weight_bytes / (ms_copy * 1e-3) / 1e9:.0f} GB/s"))
    return rows


def bench_grouped(name: str, e: int, m_per: int, n: int, k: int, *, transpose_b: bool,
                  warmup: int, iters: int) -> list[dict]:
    dev = torch.device("cuda")
    m = e * m_per
    a = torch.randn((m, k) if not transpose_b else (m, n), device=dev, dtype=torch.bfloat16)
    b_cpu = torch.randn((e, n, k), dtype=torch.bfloat16).pin_memory()
    b_gpu = b_cpu.to(dev)
    offsets = torch.arange(0, e + 1, device=dev, dtype=torch.int32) * m_per
    experts = torch.cat([
        torch.arange(e, device=dev, dtype=torch.int32),
        torch.tensor([-1], device=dev, dtype=torch.int32),
    ])
    flops = 2.0 * m * n * k
    weight_bytes = b_cpu.numel() * b_cpu.element_size()
    rows: list[dict] = []

    reason = _direct_grouped_bf16_reason(a, b_cpu, transpose_b=transpose_b)
    if reason is None:
        ms = _time_cuda(
            lambda: _asym_grouped_bf16_nt(a, b_cpu, offsets, experts, transpose_b=transpose_b),
            warmup=warmup, iters=iters)
        rows.append(dict(shape=name, engine="asym", ms=ms, tflops=_tflops(flops, ms)))
    else:
        rows.append(dict(shape=name, engine="asym", ms=float("nan"), tflops=0.0, note=reason))

    ms_res = _time_cuda(
        lambda: _grouped_torch_mm(a, b_gpu, offsets, experts, transpose_b=transpose_b),
        warmup=warmup, iters=iters)
    rows.append(dict(shape=name, engine="resident", ms=ms_res, tflops=_tflops(flops, ms_res)))

    stage_buf = torch.empty_like(b_gpu)
    ms_copy = _time_cuda(lambda: stage_buf.copy_(b_cpu, non_blocking=True),
                         warmup=warmup, iters=iters)
    ms_staged = ms_copy + ms_res
    rows.append(dict(shape=name, engine="staged(copy+mm)", ms=ms_staged,
                     tflops=_tflops(flops, ms_staged),
                     note=f"copy {ms_copy:.2f} ms = {weight_bytes / (ms_copy * 1e-3) / 1e9:.0f} GB/s"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    rows: list[dict] = []

    # q3-30b-a3b routed experts @32k x 8 (R ~= 2.05M rows, 128 experts, uniform bench split).
    grouped = [
        ("moe.gate_up fwd [E128 m16000 N768 K2048]", 128, 16000, 768, 2048, False),
        ("moe.gate_up dX  [E128 m16000 N768 K2048 tb]", 128, 16000, 768, 2048, True),
        ("moe.down    fwd [E128 m16000 N2048 K768]", 128, 16000, 2048, 768, False),
        ("moe.down    dX  [E128 m16000 N2048 K768 tb]", 128, 16000, 2048, 768, True),
    ]
    # dense attention projections @32k x 8 (M = 256k), q3-30b head geometry.
    dense = [
        ("attn.q  fwd [M256k N4096 K2048]", 256000, 4096, 2048, False),
        ("attn.q  dX  [M256k N4096 K2048 tb]", 256000, 4096, 2048, True),
        ("attn.kv fwd [M256k N512 K2048]", 256000, 512, 2048, False),
        ("attn.o  fwd [M256k N2048 K4096]", 256000, 2048, 4096, False),
    ]

    for name, e, m_per, n, k, tb in grouped:
        rows.extend(bench_grouped(name, e, m_per, n, k, transpose_b=tb,
                                  warmup=args.warmup, iters=args.iters))
    for name, m, n, k, tb in dense:
        rows.extend(bench_dense(name, m, n, k, transpose_b=tb,
                                warmup=args.warmup, iters=args.iters))

    print(f"\n{'shape':50s} {'engine':16s} {'ms':>9s} {'TFLOP/s':>9s}  note")
    by_shape: dict[str, dict[str, float]] = {}
    for r in rows:
        print(f"{r['shape']:50s} {r['engine']:16s} {r['ms']:9.2f} {r['tflops']:9.1f}  {r.get('note', '')}")
        by_shape.setdefault(r["shape"], {})[r["engine"]] = r["ms"]
    print("\nengine tax (asym / resident) and staged overhead (staged / resident):")
    for shape, eng in by_shape.items():
        asym_ms, res_ms = eng.get("asym"), eng.get("resident")
        staged_ms = eng.get("staged(copy+mm)")
        if asym_ms and res_ms and asym_ms == asym_ms:
            print(f"{shape:50s} asym/resident = {asym_ms / res_ms:5.2f}x   "
                  f"staged/resident = {staged_ms / res_ms:5.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["shape", "engine", "ms", "tflops", "note"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
