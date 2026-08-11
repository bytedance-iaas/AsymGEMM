#!/usr/bin/env python3
"""M2a-v2 — adapter-SITE placement microbenchmark (pair-form refresh).

Site: the MLP gate+up pair, d=5120 (Qwen3-32B shapes), rank 64, single
segment. Both panels price ONE SITE's work: two projections forward
(S_g, S_u = X A^T), two gradients backward (dA_g, dA_u = dS^T X).
X CPU-resident (pinned bf16); both adapters GPU-resident.

Executions (3, both rows):
  swap   : H2D copy of X once + 2 cuBLAS GEMMs off the staged copy.
  reaim2 : the library applied directly, one call per adapter —
           fwd: operand-swapped inference-form call x2 (incl.
           transpose-back); bwd: chunked stage+accumulate x2 (the
           only upstream-expressible low-memory dA mapping).
           X crosses the link TWICE (once per adapter).
  ours   : the shipped pair kernels — K1 grouped_expert_lora_pair
           (fwd) and K2 pair grad with auto stream-split (bwd).
           X crosses ONCE for both outputs.

Timing: house style — CUDA events, 10 warmup + 100 timed iters,
3 runs (first discarded). JSON to profiling_results/motivation/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import asym_gemm  # noqa: F401
from asym_gemm.training import cpu_left as cpu_left_impl
from asym_gemm.training import exp_act_offload_lora as expact_impl
from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul

OUT = Path(__file__).resolve().parents[2] / "profiling_results/motivation/m2a_v2.json"
RANK = 64
D = 5120
ROWS = [int(x) for x in os.environ.get("M2A_V2_ROWS", "32768,131072,524288").split(",")]
WARMUP_ITERS = int(os.environ.get("M2A_V2_WARMUP", "10"))
TIMED_ITERS = int(os.environ.get("M2A_V2_TIMED", "100"))
RUNS = 3  # first is the warmup run, discarded
DIRECT_CHUNK = 8192  # house staging grain for the naive dA loop


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
    a_gate = torch.randn((RANK, D), device=dev, dtype=torch.bfloat16)
    a_up = torch.randn((RANK, D), device=dev, dtype=torch.bfloat16)
    gate_w = a_gate.unsqueeze(0).contiguous()  # [1, 64, D] grouped form
    up_w = a_up.unsqueeze(0).contiguous()
    x_pinned = torch.empty((m, D), dtype=torch.bfloat16, pin_memory=True)
    x_pinned.normal_()
    x_resident = x_pinned.to(dev, non_blocking=False)
    stage_buf = torch.empty((m, D), device=dev, dtype=torch.bfloat16)
    ds_gate = torch.randn((m, RANK), device=dev, dtype=torch.bfloat16).contiguous()
    ds_up = torch.randn((m, RANK), device=dev, dtype=torch.bfloat16).contiguous()
    offsets = torch.tensor([0, m], device=dev, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=dev, dtype=torch.int32)
    torch.cuda.synchronize()

    # ---------------- forward: S_g, S_u = X A^T ----------------
    def fwd_swap():
        stage_buf.copy_(x_pinned, non_blocking=True)
        torch.matmul(stage_buf, a_gate.t())
        torch.matmul(stage_buf, a_up.t())

    def fwd_reaim2():
        sg = asym_bf16_cpu_right_matmul(a_gate, x_pinned, backend="asym", tag="v2_dg")
        sg.t().contiguous()
        su = asym_bf16_cpu_right_matmul(a_up, x_pinned, backend="asym", tag="v2_du")
        su.t().contiguous()

    def fwd_ours():
        cpu_left_impl.grouped_expert_lora_pair_cpu_left(
            x_pinned, gate_w, up_w, offsets, experts
        )

    # ---------------- backward: dA_g, dA_u = dS^T X ----------------
    def bwd_swap():
        stage_buf.copy_(x_pinned, non_blocking=True)
        torch.matmul(ds_gate.t(), stage_buf)
        torch.matmul(ds_up.t(), stage_buf)

    chunk_buf = torch.empty((DIRECT_CHUNK, D), device=dev, dtype=torch.bfloat16)

    def _naive_da(ds: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros((RANK, D), device=dev, dtype=torch.float32)
        for s in range(0, m, DIRECT_CHUNK):
            e = min(m, s + DIRECT_CHUNK)
            buf = chunk_buf[: e - s]
            buf.copy_(x_pinned[s:e], non_blocking=True)
            grad.addmm_(ds[s:e].t().float(), buf.float())
        return grad.to(torch.bfloat16)

    def bwd_reaim2():
        _naive_da(ds_gate)
        _naive_da(ds_up)

    def bwd_ours():
        expact_impl.grouped_lora_a_pair_grad_cpu_right(
            ds_gate, ds_up, x_pinned, offsets, experts, num_experts=1, stats=None
        )

    # ---------------- correctness spot-checks (once) ----------------
    ref_g = torch.matmul(x_resident, a_gate.t())
    ref_u = torch.matmul(x_resident, a_up.t())
    got_g, got_u = cpu_left_impl.grouped_expert_lora_pair_cpu_left(
        x_pinned, gate_w, up_w, offsets, experts
    )
    torch.cuda.synchronize()
    rel_fwd = max(
        float((got_g.float() - ref_g.float()).norm() / ref_g.float().norm()),
        float((got_u.float() - ref_u.float()).norm() / ref_u.float().norm()),
    )
    assert rel_fwd < 2e-2, f"pair fwd mismatch rel={rel_fwd:.3e}"

    gref_g = torch.matmul(ds_gate.t().float(), x_resident.float())
    gref_u = torch.matmul(ds_up.t().float(), x_resident.float())
    gg, gu = expact_impl.grouped_lora_a_pair_grad_cpu_right(
        ds_gate, ds_up, x_pinned, offsets, experts, num_experts=1, stats=None
    )
    torch.cuda.synchronize()
    rel_bwd = max(
        float((gg.float() - gref_g).norm() / gref_g.norm()),
        float((gu.float() - gref_u).norm() / gref_u.norm()),
    )
    assert rel_bwd < 2e-2, f"pair grad mismatch rel={rel_bwd:.3e}"

    res = {
        "rows": m,
        "rel_fwd": rel_fwd,
        "rel_bwd": rel_bwd,
        "runs": [],
    }
    for _ in range(RUNS):
        res["runs"].append(
            {
                "fwd_swap_ms": _time_loop(fwd_swap),
                "fwd_reaim2_ms": _time_loop(fwd_reaim2),
                "fwd_ours_ms": _time_loop(fwd_ours),
                "bwd_swap_ms": _time_loop(bwd_swap),
                "bwd_reaim2_ms": _time_loop(bwd_reaim2),
                "bwd_ours_ms": _time_loop(bwd_ours),
            }
        )
    # analytic input-operand bytes (GB): swap holds the X copy + both
    # adapters; the streamed executions hold the adapters only.
    xb = m * D * 2 / 1e9
    ab2 = 2 * RANK * D * 2 / 1e9
    res["mem_gb"] = {"swap": xb + ab2, "reaim2": ab2, "ours": ab2}
    del x_resident, stage_buf, chunk_buf
    torch.cuda.empty_cache()
    return res


def main() -> None:
    torch.manual_seed(0)
    out = {
        "spec": {
            "site": "MLP gate+up pair: S_g,S_u = X A^T (fwd); dA_g,dA_u = dS^T X (bwd)",
            "model": "Qwen3-32B shapes (d=5120)",
            "rank": RANK,
            "d": D,
            "rows": ROWS,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "device": torch.cuda.get_device_name(0),
            "grad_split_env": os.environ.get("ASYMM_LORA_A_GRAD_SPLIT", "<auto>"),
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
            f'rows={sh["rows"]:7d}  '
            f'fwd swap {mean("fwd_swap_ms"):7.3f} | reaim2 {mean("fwd_reaim2_ms"):7.3f} | '
            f'ours {mean("fwd_ours_ms"):7.3f}   ||   '
            f'bwd swap {mean("bwd_swap_ms"):7.3f} | naive2 {mean("bwd_reaim2_ms"):7.3f} | '
            f'ours {mean("bwd_ours_ms"):7.3f}'
        )


if __name__ == "__main__":
    main()
