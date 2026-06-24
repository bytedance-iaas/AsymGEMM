"""Microbenchmark: C = A @ B.T  with B resident on CPU.

Two ways to compute it on the GPU, and we measure the **peak HBM of the matmul call ONLY**:

  1. "stage->mm"  : copy the full B to GPU (HBM), then `a @ b.t()`  (torch).
  2. "AsymGEMM"   : hand the CPU-resident B straight to the asym bf16 NT kernel
                    (`_asym_bf16_nt` -> `m_grouped_bf16_asym_gemm_nt_contiguous`), B is never
                    fully materialized in HBM.

A is [N,K] on GPU, B is [N,K] on (pinned) CPU, result C = A @ B.T is [N,N] on GPU.
Square N=K. Isolation: warm up first (cublas/asym autotune allocate workspace once), then
`reset_peak_memory_stats()` right before the single measured call and read
`max_memory_allocated()` right after -> that delta is the HBM the matmul added.

Run:  CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/gemm/gemm.py
"""

from __future__ import annotations

import torch

import asym_gemm  # noqa: F401  (registers the C++ ops)
from asym_gemm.training.frozen_linear import _asym_bf16_nt, _staged_nt

MiB = 1024.0 * 1024.0


def _measure(fn, a, b_cpu, *, warmup: int = 2):
    """Return (peak_hbm_bytes_added_by_call, output_bytes, result_fp32) for one matmul method."""
    # Warm up: first call allocates cublas handles / asym autotune workspace; don't count it.
    for _ in range(warmup):
        d = fn(a, b_cpu, transpose_b=False)
        torch.cuda.synchronize()
        del d
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    d = fn(a, b_cpu, transpose_b=False)  # C = A @ B.T
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    out_bytes = d.numel() * d.element_size()
    res = d.float().clone()
    del d
    torch.cuda.empty_cache()
    return (peak - base), out_bytes, res


def run_size(n: int):
    dev = "cuda"
    dt = torch.bfloat16
    a = torch.randn(n, n, device=dev, dtype=dt)          # A = [N,K] on GPU
    b_cpu = torch.randn(n, n, dtype=dt).pin_memory()     # B = [N,K] on pinned CPU

    # reference (full stage) for a correctness sanity check
    bg = b_cpu.to(dev)
    ref = (a @ bg.t()).float()
    del bg
    torch.cuda.empty_cache()

    b_resident = b_cpu.numel() * b_cpu.element_size()  # what "stage" must bring into HBM
    rows = []
    for label, fn in (("stage->mm", _staged_nt), ("AsymGEMM", _asym_bf16_nt)):
        try:
            hbm, outb, res = _measure(fn, a, b_cpu)
            err = (res - ref).abs().max().item()
            rows.append([label, hbm, outb, err, None])
        except Exception as e:  # noqa: BLE001
            rows.append([label, None, None, None, f"{type(e).__name__}: {e}"])

    del a, b_cpu, ref
    torch.cuda.empty_cache()
    return b_resident, rows


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    dev_name = torch.cuda.get_device_name(0)
    sizes = [4096, 8192, 16384]

    print(f"# A @ B.T  (B on CPU)  | device={dev_name} | bf16 | peak HBM of the matmul call only\n")
    header = (
        f"{'N=K=M':>7} | {'full B (MiB)':>12} | {'method':>9} | "
        f"{'matmul peak HBM (MiB)':>21} | {'output (MiB)':>12} | {'B-stage saved (MiB)':>19} | max|err|"
    )
    print(header)
    print("-" * len(header))
    for n in sizes:
        b_resident, rows = run_size(n)
        hbm_by = {r[0]: r[1] for r in rows}
        for label, hbm, outb, err, errmsg in rows:
            if hbm is None:
                print(f"{n:>7} | {b_resident/MiB:>12.0f} | {label:>9} | {'(failed)':>21} | "
                      f"{'':>12} | {'':>19} | {errmsg}")
                continue
            saved = ""
            if label == "AsymGEMM" and hbm_by.get("stage->mm"):
                saved = f"{(hbm_by['stage->mm'] - hbm)/MiB:.0f}"
            print(f"{n:>7} | {b_resident/MiB:>12.0f} | {label:>9} | {hbm/MiB:>21.1f} | "
                  f"{outb/MiB:>12.0f} | {saved:>19} | {err:.3g}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
