#!/usr/bin/env python3
"""M4b panel (c) — added critical-path time of the dA routes inside a
busy backward context (motivation_v2_plots.md).

Context = the streamed backward: a loop of streamed cpu_left GEMMs
occupies BOTH the SMs and the C2C link (as the real T2B/T3 backward
does). Each route runs CONCURRENTLY with that context; the metric is
added makespan = makespan(context ∥ route) − makespan(context).

CPU route: dS D2H (small) then the SVE deposit kernel on host cores —
should add ≈0 while its work fits inside the context window.
GPU route: X restage + cuBLAS dA + dA D2H on a second stream —
contends for the link and the SMs, so its cost lands on the makespan.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import asym_gemm
from asym_gemm.training import cpu_left as cpu_left_impl

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m4b_window.json"
D = 2048
RANK = 64
ROWS = [int(x) for x in os.environ.get("M4B_ROWS", "65536,262144,1048576").split(",")]
THREADS = 96
BG_ROWS = 262144
BG_D = 5120
RUNS = 4  # first discarded


def make_context():
    """Streamed-GEMM context occupying SMs + C2C link (~tens of ms per call)."""
    a = torch.randn((RANK, BG_D), device="cuda", dtype=torch.bfloat16)
    w = a.unsqueeze(0).contiguous()
    xp = torch.empty((BG_ROWS, BG_D), dtype=torch.bfloat16, pin_memory=True)
    xp.normal_()
    off = torch.tensor([0, BG_ROWS], device="cuda", dtype=torch.int32)
    exp = torch.tensor([0, -1], device="cuda", dtype=torch.int32)

    def one_call():
        cpu_left_impl.grouped_expert_lora_cpu_left(xp, w, off, exp)

    return one_call


def main() -> None:
    torch.manual_seed(0)
    ctx_call = make_context()
    # calibrate context length: enough calls to span the longest CPU route
    CTX_CALLS = 5  # ~5 x ~13 ms = ~65 ms window

    def run_context():
        for _ in range(CTX_CALLS):
            ctx_call()

    # context baseline
    for _ in range(3):
        run_context()
    torch.cuda.synchronize()

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    results = {"spec": {
        "d": D, "rank": RANK, "rows": ROWS, "threads": THREADS,
        "context": f"{CTX_CALLS}x streamed cpu_left GEMM ({BG_ROWS}x{BG_D})",
        "device": torch.cuda.get_device_name(0),
    }, "clusters": []}

    side = torch.cuda.Stream()
    for m in ROWS:
        x_cpu = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
        x_cpu.normal_()
        ds_gpu = torch.randn((m, RANK), device="cuda", dtype=torch.bfloat16)
        ds_cpu = torch.empty((m, RANK), dtype=torch.bfloat16, pin_memory=True)
        out_pinned = torch.zeros((1, RANK, D), dtype=torch.float32, pin_memory=True)
        pr = torch.tensor([0, m], dtype=torch.long)
        ge = torch.zeros(1, dtype=torch.long)
        x_stage = torch.empty((m, D), device="cuda", dtype=torch.bfloat16)
        da_pin = torch.empty((RANK, D), dtype=torch.float32, pin_memory=True)
        torch.set_num_threads(THREADS)

        def ctx_with_cpu_route():
            # dS lands on host first (small), then SVE kernel runs on host
            # cores while the context keeps the GPU busy.
            with torch.cuda.stream(side):
                ds_cpu.copy_(ds_gpu, non_blocking=True)
            side.synchronize()
            out_pinned.zero_()
            import threading
            th = threading.Thread(
                target=asym_gemm.cpu_grouped_lora_a_grad_bf16,
                args=(ds_cpu, x_cpu, out_pinned, pr, ge, THREADS),
            )
            th.start()
            run_context()
            th.join()

        def ctx_with_gpu_route():
            with torch.cuda.stream(side):
                x_stage.copy_(x_cpu, non_blocking=True)
                da = torch.matmul(ds_gpu.t(), x_stage).float()
                da_pin.copy_(da, non_blocking=True)
            run_context()
            side.synchronize()

        cluster = {"rows": m, "runs": []}
        for _ in range(RUNS):
            base = timed(run_context)
            cpu_ms = timed(ctx_with_cpu_route)
            gpu_ms = timed(ctx_with_gpu_route)
            cluster["runs"].append({
                "context_ms": base,
                "cpu_added_ms": cpu_ms - base,
                "gpu_added_ms": gpu_ms - base,
            })
        results["clusters"].append(cluster)
        del x_cpu, ds_gpu, ds_cpu, x_stage
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")
    for cl in results["clusters"]:
        runs = cl["runs"][1:]
        mean = lambda k: sum(r[k] for r in runs) / len(runs)  # noqa: E731
        print(
            f"rows={cl['rows']:>8}: context={mean('context_ms'):7.2f} ms  "
            f"cpu_added={mean('cpu_added_ms'):+7.2f}  gpu_added={mean('gpu_added_ms'):+7.2f}"
        )


if __name__ == "__main__":
    main()
