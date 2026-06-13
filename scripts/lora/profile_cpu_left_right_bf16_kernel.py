#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CPU_LEFT_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one focused SM100 BF16 CPU-left/right AsymGEMM kernel.")
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compiled-dims", default="nk")
    parser.add_argument("--no-profiler-api", action="store_true")
    return parser.parse_args()


def pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    pinned = tensor.detach().cpu().contiguous().pin_memory()
    if tensor.numel() and not pinned.is_pinned():
        raise RuntimeError("pin_memory did not return pinned CPU memory")
    return pinned


def make_metadata(m: int, groups: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    if groups <= 0:
        raise ValueError("groups must be positive")
    if m % groups != 0:
        raise ValueError("m must be divisible by groups for this aligned profiler")
    rows = m // groups
    pair_offsets: list[int] = []
    experts: list[int] = []
    start = 0
    for group in range(groups):
        pair_offsets.extend((start, start + rows))
        experts.append(group)
        start += rows
    return (
        torch.tensor(pair_offsets, device=device, dtype=torch.int32),
        torch.tensor([*experts, -1], device=device, dtype=torch.int32),
        groups + 1,
    )


def measure_ms(fn, *, warmup: int, iters: int, device: torch.device) -> tuple[float, float, float]:
    fn()
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times), min(times), max(times)


def main() -> int:
    args = parse_args()
    if min(args.m, args.n, args.k, args.warmup, args.iters, args.profile_iters) <= 0:
        raise SystemExit("m, n, k, warmup, iters, and profile-iters must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if int(major) != 10:
        raise SystemExit(f"SM100 required, got capability {major}.{minor}")

    import asym_gemm

    if not hasattr(asym_gemm, CPU_LEFT_BINDING):
        raise SystemExit(f"missing binding: {CPU_LEFT_BINDING}")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    a_cuda = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16, generator=generator)
    b_cuda = torch.randn((args.groups, args.n, args.k), device=device, dtype=torch.bfloat16, generator=generator)
    a_cpu = pin_cpu(a_cuda)
    b_cpu = pin_cpu(b_cuda)
    offsets, experts, list_size = make_metadata(args.m, args.groups, device)
    left_out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
    right_out = torch.empty_like(left_out)

    def left() -> None:
        getattr(asym_gemm, CPU_LEFT_BINDING)(
            a_cpu,
            b_cuda,
            left_out,
            offsets,
            experts,
            list_size,
            args.compiled_dims,
        )

    def right() -> None:
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a_cuda,
            b_cpu,
            right_out,
            offsets,
            experts,
            list_size,
            args.compiled_dims,
        )

    selected = {"left": left, "right": right}
    sides = ("left", "right") if args.side == "both" else (args.side,)
    for side in sides:
        median_ms, min_ms, max_ms = measure_ms(
            selected[side],
            warmup=int(args.warmup),
            iters=int(args.iters),
            device=device,
        )
        print(
            f"{side} median_us={median_ms * 1000.0:.3f} "
            f"min_us={min_ms * 1000.0:.3f} max_us={max_ms * 1000.0:.3f}",
            flush=True,
        )

    if not args.no_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(int(args.profile_iters)):
        selected["left" if args.side == "both" else args.side]()
    torch.cuda.synchronize(device)
    if not args.no_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
