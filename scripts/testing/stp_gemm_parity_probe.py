#!/usr/bin/env python3
"""Stage I3 kernel-level parity probe (gb200_tp.md): the split stp_base_gemm path must
match the single-device |1 GEMM bit-for-band on col/row, fwd/dX, even/uneven shards.

Reference = the SAME kernel on the UNSPLIT weight (single device). Split outputs differ
only by bf16 partial-sum order on row-fwd/col-dX (sum of two kernel partials), so the
band is a small rel-err, not bit equality.

Usage: .venv/bin/python scripts/testing/stp_gemm_parity_probe.py [--cases col,row]
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ASYM_STP", "1")
os.environ.setdefault("ASYM_STP_TP_SIZE", "2")

import torch  # noqa: E402


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    ref_f = ref.float()
    return ((got.float() - ref_f).abs().max() / ref_f.abs().max().clamp_min(1e-8)).item()


def check_split(kind: str, op: str, got: torch.Tensor, ref: torch.Tensor, fp32: torch.Tensor,
                failures: list[str], label: str) -> None:
    """Gather cases (col fwd, row dX) must be BIT-IDENTICAL to the unsplit kernel.
    Partial-SUM cases (col dX, row fwd) pay exactly one extra bf16 round+add (the
    intrinsic TP partial-sum cost — Megatron's bf16 all-reduce pays the same), so the
    gate is: split error vs fp32 <= 2.5x the UNSPLIT kernel's own error vs fp32."""
    is_gather = (kind == "col" and op == "fwd") or (kind == "row" and op == "dx")
    if is_gather:
        ok = torch.equal(got, ref)
        print(f"[parity] {label}: bit-identical={ok}")
        if not ok:
            failures.append(f"{label} not bit-identical (gather case)")
        return
    err_split = (got.float() - fp32).abs().max().item()
    err_ref = (ref.float() - fp32).abs().max().item()
    ratio = err_split / max(err_ref, 1e-12)
    ok = ratio <= 2.5
    print(f"[parity] {label}: err_vs_fp32 split={err_split:.3e} unsplit={err_ref:.3e} ratio={ratio:.2f} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{label} ratio {ratio:.2f} > 2.5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="col,row")
    parser.add_argument("--m", type=int, default=4096)
    args = parser.parse_args()

    from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul
    from asym_gemm.training.stp_layout import attach_col_shards, repack_row_shards

    dev0 = torch.device("cuda", 0)
    m = args.m
    failures: list[str] = []

    # q3-32b-shaped matrices + one deliberately UNEVEN col split (ceil64 pad rule)
    shapes = {
        "col": [(8192, 5120), (1024, 5120), (25600, 5120), (4160, 5120)],  # 4160/2=2080 -> N0=2112 uneven
        "row": [(5120, 8192), (5120, 25600)],
    }
    tol = 2e-2  # bf16 partial-sum band

    for kind in args.cases.split(","):
        for n, k in shapes[kind]:
            w = (torch.randn(n, k, dtype=torch.bfloat16) / 32.0).pin_memory()
            w_ref = w.clone().pin_memory()  # UNSPLIT reference copy (row repack scrambles w)

            if kind == "col":
                attach_col_shards(w)
                carrier = w
            else:
                carrier, _ = repack_row_shards(w)

            x = torch.randn(m, k, device=dev0, dtype=torch.bfloat16)
            got = asym_bf16_cpu_right_matmul(x, carrier, phase="forward", tag=f"parity.{kind}.fwd")
            ref = asym_bf16_cpu_right_matmul(x, w_ref, phase="forward", tag=f"parity.{kind}.fwd.ref")
            torch.cuda.synchronize(0); torch.cuda.synchronize(1)
            fp32 = x.float() @ w_ref.to(dev0).float().T
            check_split(kind, "fwd", got, ref, fp32, failures, f"{kind} fwd  [{n},{k}]")

            g = torch.randn(m, n, device=dev0, dtype=torch.bfloat16)
            got_dx = asym_bf16_cpu_right_matmul(g, carrier, transpose_b=True, phase="dx", tag=f"parity.{kind}.dx")
            ref_dx = asym_bf16_cpu_right_matmul(g, w_ref, transpose_b=True, phase="dx", tag=f"parity.{kind}.dx.ref")
            torch.cuda.synchronize(0); torch.cuda.synchronize(1)
            fp32_dx = g.float() @ w_ref.to(dev0).float()
            check_split(kind, "dx", got_dx, ref_dx, fp32_dx, failures, f"{kind} dX   [{n},{k}]")

    print(f"[parity] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
