#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CPU_LEFT_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"


def parse_shape_list(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            s_text, k_text = item.split("x", 1)
            shapes.append((int(s_text), int(k_text)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid shape {item!r}; expected SxK") from exc
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile SM100 BF16 CPU-left AsymGEMM against CPU-right parity.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--square-cases", type=parse_shape_list, default=parse_shape_list("128x512,256x1024,512x4096"))
    parser.add_argument("--lora-m", type=int, default=256)
    parser.add_argument("--lora-k", type=int, default=1024)
    parser.add_argument("--lora-ranks", type=parse_int_list, default=parse_int_list("8,16,64,128"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-ratio", type=float, default=0.85)
    parser.add_argument("--max-ratio", type=float, default=1.15)
    parser.add_argument("--json", default="", help="optional JSON output path")
    args = parser.parse_args()
    if min(args.lora_m, args.lora_k, args.warmup, args.iters) <= 0:
        raise SystemExit("--lora-m, --lora-k, --warmup, and --iters must be positive")
    for s, k in args.square_cases:
        if min(s, k) <= 0:
            raise SystemExit("--square-cases values must be positive")
    for rank in args.lora_ranks:
        if rank <= 0:
            raise SystemExit("--lora-ranks values must be positive")
    return args


def require_runtime(device: torch.device):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if device.type != "cuda":
        raise SystemExit(f"expected CUDA device, got {device}")
    torch.cuda.set_device(device)
    major, _minor = torch.cuda.get_device_capability(device)
    if int(major) != 10:
        raise SystemExit(f"SM100 required, got capability {major}")

    import asym_gemm

    if not hasattr(asym_gemm, CPU_LEFT_BINDING):
        raise SystemExit(f"missing binding: {CPU_LEFT_BINDING}")
    return asym_gemm


def pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    pinned = tensor.detach().cpu().contiguous().pin_memory()
    if not pinned.is_pinned():
        raise SystemExit("pin_memory did not return a pinned CPU tensor")
    return pinned


def metadata(rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    return (
        torch.tensor([0, rows], device=device, dtype=torch.int32),
        torch.tensor([0, -1], device=device, dtype=torch.int32),
        2,
    )


def measure_ms(fn: Callable[[], None], *, warmup: int, iters: int, device: torch.device) -> list[float]:
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
    return times


def calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    denom = expected_f.abs().max().clamp_min(1e-6)
    return float((actual_f - expected_f).abs().max().div(denom).item())


def run_case(
    asym_gemm,
    *,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    label: str,
    warmup: int,
    iters: int,
    seed: int,
    min_ratio: float,
    max_ratio: float,
) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    a_cuda = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator)
    b_cuda = torch.randn((1, n, k), device=device, dtype=torch.bfloat16, generator=generator)
    a_cpu = pin_cpu(a_cuda)
    b_cpu = pin_cpu(b_cuda)
    offsets, experts, list_size = metadata(m, device)
    left_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    right_out = torch.empty_like(left_out)

    def left() -> None:
        getattr(asym_gemm, CPU_LEFT_BINDING)(a_cpu, b_cuda, left_out, offsets, experts, list_size, "nk")

    def right() -> None:
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a_cuda, b_cpu, right_out, offsets, experts, list_size, "nk"
        )

    _ = float(a_cpu[0, 0].item()) + float(b_cpu[0, 0, 0].item())
    left()
    right()
    torch.cuda.synchronize(device)
    torch_ref = a_cuda.float().matmul(b_cuda[0].float().t()).to(torch.bfloat16)
    diff_torch = calc_diff(left_out, torch_ref)
    diff_left_right = calc_diff(left_out, right_out)

    left_ms = measure_ms(left, warmup=warmup, iters=iters, device=device)
    right_ms = measure_ms(right, warmup=warmup, iters=iters, device=device)
    left_median = float(statistics.median(left_ms))
    right_median = float(statistics.median(right_ms))
    ratio = left_median / right_median if right_median > 0.0 else float("inf")
    square_ratio_ok = label == "lora_rank" or min_ratio <= ratio <= max_ratio
    status = "PASS" if diff_torch < 1e-3 and diff_left_right < 1e-3 and square_ratio_ok else "FAIL"
    return {
        "status": status,
        "case": label,
        "m": m,
        "n": n,
        "k": k,
        "left_us": left_median * 1000.0,
        "right_us": right_median * 1000.0,
        "ratio": ratio,
        "diff_torch": diff_torch,
        "diff_left_right": diff_left_right,
    }


def print_rows(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'status':<6} {'case':<10} {'M':>5} {'N':>5} {'K':>6} "
        f"{'left_us':>10} {'right_us':>10} {'ratio':>7} {'diff_torch':>12} {'diff_left_right':>16}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['status']:<6} {row['case']:<10} {row['m']:>5} {row['n']:>5} {row['k']:>6} "
            f"{row['left_us']:>10.2f} {row['right_us']:>10.2f} {row['ratio']:>7.3f} "
            f"{row['diff_torch']:>12.5e} {row['diff_left_right']:>16.5e}"
        )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    asym_gemm = require_runtime(device)
    rows: list[dict[str, object]] = []
    for idx, (s, k) in enumerate(args.square_cases):
        rows.append(
            run_case(
                asym_gemm,
                device=device,
                m=s,
                n=s,
                k=k,
                label="square",
                warmup=int(args.warmup),
                iters=int(args.iters),
                seed=int(args.seed) + idx,
                min_ratio=float(args.min_ratio),
                max_ratio=float(args.max_ratio),
            )
        )
    for idx, rank in enumerate(args.lora_ranks):
        rows.append(
            run_case(
                asym_gemm,
                device=device,
                m=int(args.lora_m),
                n=int(rank),
                k=int(args.lora_k),
                label="lora_rank",
                warmup=int(args.warmup),
                iters=int(args.iters),
                seed=int(args.seed) + 100 + idx,
                min_ratio=float(args.min_ratio),
                max_ratio=float(args.max_ratio),
            )
        )

    print_rows(rows)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
