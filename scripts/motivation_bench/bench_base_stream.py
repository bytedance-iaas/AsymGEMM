#!/usr/bin/env python3
"""Dense asym base-GEMM micro: host-streamed frozen weight at training M.

Shape (attention projection at 32K x b8): X [256K x 2048] HBM bf16,
W [2048 x 2048] pinned host bf16, Y = X @ W^T via asym_bf16_cpu_right_matmul
(the shipped attention base fwd). Reference points: cuBLAS with W resident,
and the raw W H2D copy.
"""
from __future__ import annotations

import os

import torch

import asym_gemm  # noqa: F401
from asym_gemm.training.frozen_linear import asym_bf16_cpu_right_matmul

N = int(os.environ.get("BS_N", "256000"))
D = int(os.environ.get("BS_D", "2048"))
O = int(os.environ.get("BS_O", "2048"))
ITERS = int(os.environ.get("BS_ITERS", "5"))


def time_loop(fn, iters=ITERS):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    x = torch.randn(N, D, device=dev, dtype=torch.bfloat16)
    w_host = torch.randn(O, D, dtype=torch.bfloat16).pin_memory()
    w_dev = w_host.to(dev)
    stage = torch.empty_like(w_dev)

    def run_asym():
        return asym_bf16_cpu_right_matmul(x, w_host, output_dtype=torch.bfloat16)

    def run_cublas():
        return x @ w_dev.t()

    def run_w_copy():
        stage.copy_(w_host, non_blocking=True)

    ref = run_cublas()
    got = run_asym()
    torch.cuda.synchronize()
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    print(f"[check] asym vs cublas rel: {rel:.3e}", flush=True)

    flops = 2 * N * D * O
    for name, fn in (("asym_stream", run_asym), ("cublas_resident", run_cublas),
                     ("w_h2d_copy", run_w_copy)):
        ms = time_loop(fn)
        tf = flops / 1e12 / (ms / 1e3)
        print(f"[time] {name}: {ms:.3f} ms ({tf:.0f} TFLOP/s)", flush=True)


if __name__ == "__main__":
    main()
