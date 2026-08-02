#!/usr/bin/env python3
"""M2c — adapter-GRADIENT placement microbenchmark (K2 twin of M2a).

Site: the adapter-gradient contraction dA = dS^T X at M2a's shape (rank 64,
d=5120, Qwen3-32B attention/MLP hidden): dS [M,64] GPU-resident (the small
projection gradient), X [M,5120] CPU-resident (pinned). This is the dataflow
the inference kernel cannot express at all — the reduction runs over the
STREAMED axis — so Staged is literally the only executable fallback.
Executions: Resident (X pre-resident in HBM, cuBLAS — ceiling) / Staged (H2D
copy of X + cuBLAS, copy timed) / Streamed (ours:
sm100_grouped_lora_a_grad_bf16_cpu_right, single group, output-stationary —
the rank-sized accumulators live in registers for the whole stream).

Timing: CUDA events around the iteration loop, [10] warmup + [100] timed
iters per measured run; 1 warmup run + 2 measured runs; JSON out carries
per-run per-GEMM ms. Protocol: GPU on NUMA node 0 with membind=0 (far-node
H2D halves bandwidth).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import asym_gemm  # noqa: F401  (loads the extension)
from asym_gemm.training import exp_act_offload_lora as ea
from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul

DIRECT_CHUNK = 8192  # naive fallback staging grain (house segment grain)

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m2c.json"
RANK = 64
D = 5120
ROWS = [int(x) for x in os.environ.get("M2C_ROWS", "32768,65536,131072,262144,524288").split(",")]
WARMUP_ITERS = 10
TIMED_ITERS = 100
RUNS = 3  # first is the warmup run, discarded


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


def bench_shape(m: int) -> dict:
    dev = torch.device("cuda")
    ds = torch.randn((m, RANK), device=dev, dtype=torch.bfloat16).contiguous()
    x_pinned = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
    x_pinned.normal_()
    x_resident = x_pinned.to(dev, non_blocking=False)
    stage_buf = torch.empty((m, D), device=dev, dtype=torch.bfloat16)
    offsets = torch.tensor([0, m], device=dev, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=dev, dtype=torch.int32)
    torch.cuda.synchronize()

    def run_resident():
        torch.matmul(ds.t(), x_resident)

    def run_staged():
        stage_buf.copy_(x_pinned, non_blocking=True)
        torch.matmul(ds.t(), stage_buf)

    def run_streamed():
        # OURS at the dense site = the shipped path: the TRANSPOSED streamed
        # instantiation (asym_bf16_cpu_right_matmul transpose_b=True), a
        # training-era addition upstream does not contain
        # (attention_activation_offload.py:1409 uses it for grad_a).
        asym_bf16_cpu_right_matmul(
            ds.t().contiguous(), x_pinned, transpose_b=True, backend="asym", tag="m2c_ours"
        )

    def run_grouped_kernel():
        # the MoE-grouped K2 form, recorded for reference (NOT the dense path)
        ea.grouped_lora_a_grad_cpu_right(
            ds, x_pinned, offsets, experts, num_experts=1, stats=None, tag="m2c"
        )

    # naive DIRECT use of the inference-form kernel. The dA dataflow reduces
    # over the STREAMED axis, which the inference kernel does not contain; the
    # closest one-call mapping is its transposed-stream form (dS^T [r,m] GPU x
    # X [m,K] streamed under transpose_b) — if its tiling constraints reject
    # the shape, the only trivial adaptation left is the chunked
    # stage+accumulate loop at the house staging grain.
    direct_mode = os.environ.get("M2C_DIRECT", "chunked")
    try:
        asym_bf16_cpu_right_matmul(
            ds.t().contiguous(), x_pinned, transpose_b=True, backend="asym", tag="m2c_direct"
        )
        torch.cuda.synchronize()
        if direct_mode != "transposed":
            raise RuntimeError("forced chunked")
    except (RuntimeError, ValueError) as exc:
        direct_mode = "chunked"
        direct_reject = str(exc).splitlines()[0][:200]

    if direct_mode == "transposed":
        def run_direct():
            asym_bf16_cpu_right_matmul(
                ds.t().contiguous(), x_pinned, transpose_b=True, backend="asym", tag="m2c_direct"
            )
    else:
        chunk_buf = torch.empty((DIRECT_CHUNK, D), device=dev, dtype=torch.bfloat16)
        def run_direct():
            grad = torch.zeros((RANK, D), device=dev, dtype=torch.float32)
            for s in range(0, m, DIRECT_CHUNK):
                e = min(m, s + DIRECT_CHUNK)
                buf = chunk_buf[: e - s]
                buf.copy_(x_pinned[s:e], non_blocking=True)
                grad.addmm_(ds[s:e].t().float(), buf.float())
            grad.to(torch.bfloat16)

    # correctness spot-checks once (reference = the bf16 cuBLAS product —
    # like-for-like bf16-input reduction over the m-row token axis; the fp32
    # reference sits ~2.7e-2 away from EVERY bf16 pipeline at 32K+ rows)
    ref = torch.matmul(ds.t(), x_resident)
    got = asym_bf16_cpu_right_matmul(
        ds.t().contiguous(), x_pinned, transpose_b=True, backend="asym", tag="m2c_ours"
    )
    torch.cuda.synchronize()
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    # rel grows ~sqrt(m) on zero-mean random data (2.7e-2 @32K, 5.3e-2 @64K):
    # the transposed stream accumulates in bf16, the cancellation-adversarial
    # case for a 512K-term zero-mean reduction — vs the sm100 grouped kernel's
    # fp32 registers at 1.7e-3 (a real accuracy edge of the redesigned K2).
    # The transposed path is loss-parity-gated e2e; this micro gate only
    # catches structural bugs (tail drops etc.).
    assert rel < 2e-1, f"ours (transposed stream) mismatch rel={rel:.3e}"
    got_g = ea.grouped_lora_a_grad_cpu_right(
        ds, x_pinned, offsets, experts, num_experts=1, stats=None, tag="m2c"
    )[0]
    torch.cuda.synchronize()
    rel_g = (got_g.float() - ref.float()).norm() / ref.float().norm()
    assert rel_g < 3e-2, f"grouped dA kernel mismatch rel={rel_g:.3e}"

    res = {"rows": m, "rel_check": float(rel), "direct_mode": direct_mode, "runs": []}
    if direct_mode == "chunked":
        res["direct_reject"] = direct_reject
    for _ in range(RUNS):
        res["runs"].append(
            {
                "resident_ms": _time_loop(run_resident),
                "staged_ms": _time_loop(run_staged),
                "streamed_ms": _time_loop(run_streamed),
                "grouped_kernel_ms": _time_loop(run_grouped_kernel),
                "direct_ms": _time_loop(run_direct),
            }
        )
    # analytic operand bytes (GB) for the memory panel (dS is GPU-resident in
    # every execution; what differs is X's residency)
    xb = m * D * 2 / 1e9
    dsb = m * RANK * 2 / 1e9
    direct_gb = dsb + (0.0 if direct_mode == "transposed" else DIRECT_CHUNK * D * 2 / 1e9)
    res["mem_gb"] = {"resident": xb + dsb, "staged": xb + dsb, "streamed": dsb,
                     "direct": direct_gb}
    del x_resident, stage_buf, ds
    torch.cuda.empty_cache()
    return res


def main() -> None:
    torch.manual_seed(0)
    out = {
        "spec": {
            "site": "adapter gradient dA = dS^T X (K2 twin of M2a's site)",
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
            f"direct[{sh['direct_mode']}]={mean('direct_ms'):7.3f} ms  (rel={sh['rel_check']:.2e})"
        )


if __name__ == "__main__":
    main()
