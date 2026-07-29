#!/usr/bin/env python3
"""M2a — adapter-GEMM case study (motivation_v2_plots.md).

Site: attention LoRA-A forward projection S = X A^T, rank 64, d=5120
(Qwen3-32B attention hidden). X CPU-resident (pinned); A GPU-resident.
Executions: Resident (X in HBM, cuBLAS) / Staged (H2D copy + cuBLAS,
copy timed) / Streamed (grouped_expert_lora_cpu_left, single group).

Timing: CUDA events around the iteration loop, [10] warmup + [100]
timed iters per measured run; 1 warmup run + 2 measured runs; JSON out
carries per-run per-GEMM ms.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import asym_gemm  # noqa: F401  (loads the extension)
from asym_gemm.training import cpu_left as cpu_left_impl
from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m2a.json"
RANK = 64
D = 5120
ROWS = [int(x) for x in os.environ.get("M2A_ROWS", "32768,262144").split(",")]
WARMUP_ITERS = 10
TIMED_ITERS = 100
RUNS = 3  # first is the warmup run, discarded


def _time_loop(fn) -> float:
    """Return mean per-call ms over TIMED_ITERS."""
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


def bench_shape(m: int) -> dict:
    dev = torch.device("cuda")
    a = torch.randn((RANK, D), device=dev, dtype=torch.bfloat16)  # adapter, resident
    weight = a.unsqueeze(0).contiguous()  # [1, 64, 5120] for the grouped kernel
    x_pinned = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
    x_pinned.normal_()
    x_resident = x_pinned.to(dev, non_blocking=False)
    stage_buf = torch.empty((m, D), device=dev, dtype=torch.bfloat16)
    offsets = torch.tensor([0, m], device=dev, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=dev, dtype=torch.int32)
    torch.cuda.synchronize()

    def run_resident():
        torch.matmul(x_resident, a.t())

    def run_staged():
        stage_buf.copy_(x_pinned, non_blocking=True)
        torch.matmul(stage_buf, a.t())

    def run_streamed():
        cpu_left_impl.grouped_expert_lora_cpu_left(x_pinned, weight, offsets, experts)

    def run_direct():
        # naive DIRECT use of the inference-form asym kernel: stream the CPU
        # operand as the "weight" (the only role it can stream), i.e. compute
        # S^T = A . X^T with A as the 64-row resident "batch"; the consumer
        # needs S row-major, so the transpose-back is part of the price.
        st = asym_bf16_cpu_right_matmul(a, x_pinned, backend="asym", tag="m2a_direct")
        st.t().contiguous()

    # correctness spot-checks once
    ref = torch.matmul(x_resident, a.t())
    got = cpu_left_impl.grouped_expert_lora_cpu_left(x_pinned, weight, offsets, experts)
    torch.cuda.synchronize()
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 2e-2, f"streamed kernel mismatch rel={rel:.3e}"
    got_direct = asym_bf16_cpu_right_matmul(a, x_pinned, backend="asym", tag="m2a_direct").t()
    torch.cuda.synchronize()
    rel_d = (got_direct.float() - ref.float()).norm() / ref.float().norm()
    assert rel_d < 2e-2, f"direct-use mismatch rel={rel_d:.3e}"

    res = {"rows": m, "rel_check": float(rel), "rel_check_direct": float(rel_d), "runs": []}
    for _ in range(RUNS):
        res["runs"].append(
            {
                "resident_ms": _time_loop(run_resident),
                "staged_ms": _time_loop(run_staged),
                "streamed_ms": _time_loop(run_streamed),
                "direct_ms": _time_loop(run_direct),
            }
        )
    # analytic operand bytes (GB) for the memory panel
    xb = m * D * 2 / 1e9
    ab = RANK * D * 2 / 1e9
    res["mem_gb"] = {"resident": xb + ab, "staged": xb + ab, "streamed": ab, "direct": ab}
    del x_resident, stage_buf
    torch.cuda.empty_cache()
    return res


def main() -> None:
    torch.manual_seed(0)
    out = {
        "spec": {
            "site": "attention LoRA-A forward S=XA^T",
            "model": "Qwen3-32B (d=5120)",
            "rank": RANK,
            "d": D,
            "rows": ROWS,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "device": torch.cuda.get_device_name(0),
        },
        "shapes": [bench_shape(m) for m in ROWS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for sh in out["shapes"]:
        runs = sh["runs"][1:]
        mean = lambda k: sum(r[k] for r in runs) / len(runs)  # noqa: E731
        print(
            f"rows={sh['rows']:>7}: resident={mean('resident_ms'):7.3f} ms  "
            f"staged={mean('staged_ms'):7.3f} ms  streamed={mean('streamed_ms'):7.3f} ms  "
            f"direct={mean('direct_ms'):7.3f} ms  (rel={sh['rel_check']:.2e})"
        )


if __name__ == "__main__":
    main()
