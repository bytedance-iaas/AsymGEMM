#!/usr/bin/env python3
"""Attention-site LoRA leg micro A/B (shipped-path shapes, single group).

Site: attention projections at seq*batch = N rows, d=2048, r=64; U [N x d]
bf16 pinned host (the shared q/k/v source); per-projection legs as the
shipped attention_activation_offload path runs them.

Variants
  fwd_single3 : 3x grouped_expert_lora_cpu_left (q/k/v each streaming U)
  fwd_cat3    : 1x call with cat'd A [192 x d]  (qkv shared-stream, cat form)
  dA_matmul   : asym_bf16_cpu_right_matmul(dS^T, U, transpose_b=True)  x3
                (the shipped fallback: upstream-form kernel, K2-shaped job)
  dA_tiled    : sm100_grouped_lora_a_grad_bf16_cpu_right (v15 fixed)   x3
  dA_tiled_cat: 1x tiled call with cat'd dS [N x 192]  (qkv dA shared-stream)
  h2d_slice   : raw pinned->HBM copy ceiling

Timing: 3 runs (first discarded), 2 warmup + 5 timed iters, CUDA events.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import asym_gemm  # noqa: F401
from asym_gemm.training import cpu_left as cpu_left_impl
from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/attn_site.json"

D = int(os.environ.get("ATTN_D", "2048"))
RANK = int(os.environ.get("ATTN_RANK", "64"))
N = int(os.environ.get("ATTN_N", "256000"))
WARMUP_ITERS = int(os.environ.get("ATTN_WARMUP", "2"))
TIMED_ITERS = int(os.environ.get("ATTN_TIMED", "5"))
RUNS = int(os.environ.get("ATTN_RUNS", "3"))


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
    torch.manual_seed(0)
    dev = torch.device("cuda")
    assert N % 64 == 0, "keep N 64-aligned (matches the shipped no-pad case)"
    u = torch.empty((N, D), dtype=torch.bfloat16, pin_memory=True)
    for i in range(0, N, 1 << 20):
        u[i : i + (1 << 20)].normal_()

    a3 = [torch.randn((RANK, D), device=dev, dtype=torch.bfloat16) for _ in range(3)]
    ds3 = [torch.randn((N, RANK), device=dev, dtype=torch.bfloat16) for _ in range(3)]
    offsets = torch.tensor([0, N], device=dev, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=dev, dtype=torch.int32)
    grad_native = getattr(asym_gemm, "sm100_grouped_lora_a_grad_bf16_cpu_right")
    pair_grad_native = getattr(asym_gemm, "sm100_grouped_lora_a_pair_grad_bf16_cpu_right")
    torch.cuda.synchronize()

    def fwd_single3():
        outs = []
        for a in a3:
            outs.append(cpu_left_impl.grouped_expert_lora_cpu_left(
                u, a.unsqueeze(0).contiguous(), offsets, experts))
        return outs

    a_cat = torch.cat(a3, dim=0).unsqueeze(0).contiguous()

    def fwd_cat3():
        s = cpu_left_impl.grouped_expert_lora_cpu_left(u, a_cat, offsets, experts)
        return [t.contiguous() for t in s.split(RANK, dim=-1)]

    triple_native = getattr(
        asym_gemm, "sm100_m_grouped_bf16_cpu_left_triple_asym_gemm_nt_contiguous", None)
    b3 = [a.unsqueeze(0).contiguous() for a in a3]
    pair_off = torch.tensor([0, N], device=dev, dtype=torch.int32)

    def fwd_triple():
        outs = [torch.empty((N, RANK), device=dev, dtype=torch.bfloat16) for _ in range(3)]
        triple_native(u, b3[0], b3[1], b3[2], outs[0], outs[1], outs[2],
                      pair_off, experts, 2, "nk")
        return outs

    def dA_matmul():
        # The shipped fallback incl. its real per-call transpose copy.
        outs = []
        for ds in ds3:
            outs.append(asym_bf16_cpu_right_matmul(
                ds.t().contiguous(), u, transpose_b=True,
                output_dtype=torch.bfloat16))
        return outs

    def dA_tiled():
        outs = []
        for ds in ds3:
            g = torch.empty((1, RANK, D), device=dev, dtype=torch.bfloat16)
            grad_native(ds, u, g, offsets, experts, 2)
            outs.append(g[0])
        return outs

    def dA_tiled_pair2p1():
        # qkv dA as pair(q,k) + single(v): 2 X streams instead of 3.
        gq = torch.empty((1, RANK, D), device=dev, dtype=torch.bfloat16)
        gk = torch.empty_like(gq)
        gv = torch.empty_like(gq)
        pair_grad_native(ds3[0], ds3[1], u, gq, gk, offsets, experts, 2)
        grad_native(ds3[2], u, gv, offsets, experts, 2)
        return [gq[0], gk[0], gv[0]]

    slice_rows = min(N, (1 << 30) // (D * 2))
    stage = torch.empty((slice_rows, D), device=dev, dtype=torch.bfloat16)

    def h2d_slice():
        stage.copy_(u[:slice_rows], non_blocking=True)

    fns = {
        "fwd_single3": (fwd_single3, 3 * N * D * 2),
        "fwd_cat3": (fwd_cat3, N * D * 2),
        "fwd_triple": (fwd_triple, N * D * 2),
        "dA_matmul": (dA_matmul, 3 * N * D * 2),
        "dA_tiled": (dA_tiled, 3 * N * D * 2),
        "dA_tiled_pair2p1": (dA_tiled_pair2p1, 2 * N * D * 2),
        "h2d_slice": (h2d_slice, slice_rows * D * 2),
    }

    def rel(x, y):
        return float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))

    checks = {}
    f1, fc = fwd_single3(), fwd_cat3()
    torch.cuda.synchronize()
    checks["fwd_cat_vs_single"] = max(rel(c, s) for c, s in zip(fc, f1))
    if triple_native is not None:
        ft = fwd_triple()
        torch.cuda.synchronize()
        checks["fwd_triple_vs_single"] = max(rel(t, s) for t, s in zip(ft, f1))
        del ft
    else:
        fns.pop("fwd_triple")
    gm, gt, gp = dA_matmul(), dA_tiled(), dA_tiled_pair2p1()
    torch.cuda.synchronize()
    checks["dA_tiled_vs_matmul"] = max(rel(t, m) for t, m in zip(gt, gm))
    checks["dA_pair2p1_vs_tiled"] = max(rel(p, t) for p, t in zip(gp, gt))
    del f1, fc, gm, gt, gp
    torch.cuda.empty_cache()
    for k, v in sorted(checks.items()):
        print(f"[check] {k}: {v:.3e}", flush=True)

    results = {}
    for name, (fn, bytes_) in fns.items():
        runs = [_time_loop(fn) for _ in range(RUNS)]
        mean_ms = sum(runs[1:]) / (RUNS - 1)
        gbps = bytes_ / 1e9 / (mean_ms / 1e3)
        results[name] = {"runs_ms": runs, "mean_ms": mean_ms, "eff_gbps": gbps}
        print(f"[time] {name}: {mean_ms:.3f} ms ({gbps:.1f} GB/s streamed)", flush=True)
        torch.cuda.empty_cache()

    out = {"spec": {"n": N, "d": D, "rank": RANK,
                    "device": torch.cuda.get_device_name(0)},
           "checks": checks, "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[str(N)] = out
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"wrote {OUT}", flush=True)
    if "fwd_single3" in results and "fwd_cat3" in results:
        print(f"[summary] fwd qkv cat speedup: {results['fwd_single3']['mean_ms']/results['fwd_cat3']['mean_ms']:.3f}x")
    if "fwd_single3" in results and "fwd_triple" in results:
        print(f"[summary] fwd qkv IN-KERNEL triple speedup: {results['fwd_single3']['mean_ms']/results['fwd_triple']['mean_ms']:.3f}x")
    if "dA_matmul" in results and "dA_tiled" in results:
        print(f"[summary] dA tiled vs shipped matmul: {results['dA_matmul']['mean_ms']/results['dA_tiled']['mean_ms']:.3f}x")
    if "dA_matmul" in results and "dA_tiled_pair2p1" in results:
        print(f"[summary] dA tiled pair2p1 vs shipped matmul: {results['dA_matmul']['mean_ms']/results['dA_tiled_pair2p1']['mean_ms']:.3f}x")


if __name__ == "__main__":
    main()
