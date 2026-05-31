#!/usr/bin/env python3
"""Isolated LoRA operator profiler.

This script measures the LoRA operator path independently from the end-to-end
training-step profiler in `scripts/lora/profile_lora_e2e.py`.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

CSV_FIELDS = [
    "operation",
    "backend",
    "pass",
    "device",
    "tokens",
    "batch_size",
    "seq_len",
    "in_features",
    "out_features",
    "rank",
    "scale",
    "dtype",
    "precision",
    "asym_bf16_output_dtype",
    "dropout_p",
    "warmup",
    "iters",
    "cuda_graph",
    "median_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "peak_hbm_gib",
    "peak_hbm_bytes",
]


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(DTYPES))
        raise SystemExit(f"unsupported dtype {name!r}; allowed: {allowed}") from exc


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx_push(device: torch.device, name: str) -> bool:
    if device.type != "cuda":
        return False
    torch.cuda.nvtx.range_push(name)
    return True


def nvtx_pop(pushed: bool) -> None:
    if pushed:
        torch.cuda.nvtx.range_pop()


def make_asym_base(
    weight: torch.Tensor,
    *,
    pin_memory: bool,
    precision: str,
    bf16_output_dtype: str,
) -> torch.nn.Module:
    from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear

    base = AsymFrozenLinear(
        weight.detach().cpu(),
        backend="asym",
        pin_memory=pin_memory,
        stats=AsymExecutionStats(),
        precision=precision,
        bf16_output_dtype=bf16_output_dtype,
    )
    base.profile_name = "lora_kernel"
    return base


class LoRAInputs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        self.dtype = parse_dtype(args.dtype)
        self.tokens = int(args.tokens) if int(args.tokens) > 0 else int(args.batch_size) * int(args.seq_len)
        if min(self.tokens, args.in_features, args.out_features, args.rank) <= 0:
            raise SystemExit("tokens, feature sizes, and rank must be positive")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(args.seed))
        self.x = torch.randn(self.tokens, args.in_features, device=self.device, dtype=self.dtype, generator=generator)
        self.w: torch.Tensor | None = torch.randn(
            args.out_features,
            args.in_features,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.a = torch.randn(args.rank, args.in_features, device=self.device, dtype=self.dtype, generator=generator)
        self.s: torch.Tensor | None = (
            torch.randn(self.tokens, args.rank, device=self.device, dtype=self.dtype, generator=generator)
            if args.operation == "xw_sb"
            else None
        )
        self.b = torch.randn(args.out_features, args.rank, device=self.device, dtype=self.dtype, generator=generator)
        self.grad: torch.Tensor | None = (
            torch.randn(self.tokens, args.out_features, device=self.device, dtype=self.dtype, generator=generator)
            if args.backward
            else None
        )
        self.base = None
        if args.backend == "asym":
            self.base = make_asym_base(
                self.w,
                pin_memory=self.device.type == "cuda",
                precision=args.precision,
                bf16_output_dtype=args.asym_bf16_output_dtype,
            )
            self.w = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def enable_backward(self) -> None:
        self.x.requires_grad_(True)
        self.a.requires_grad_(True)
        if self.s is not None:
            self.s.requires_grad_(True)
        self.b.requires_grad_(True)

    def clear_grads(self) -> None:
        self.x.grad = None
        self.a.grad = None
        if self.s is not None:
            self.s.grad = None
        self.b.grad = None


def base_linear(inputs: LoRAInputs) -> torch.Tensor:
    if inputs.base is not None:
        return inputs.base(inputs.x)
    assert inputs.w is not None
    return inputs.x @ inputs.w.T


def xw_sb(inputs: LoRAInputs, scale: float, dropout_p: float = 0.0) -> torch.Tensor:
    del dropout_p
    assert inputs.s is not None
    base = base_linear(inputs)
    lora = inputs.s @ inputs.b.T
    return base + (lora * scale).to(dtype=base.dtype)


def full_lora(inputs: LoRAInputs, scale: float, dropout_p: float = 0.0) -> torch.Tensor:
    x_lora = (
        torch.nn.functional.dropout(inputs.x, p=dropout_p, training=True)
        if dropout_p > 0.0
        else inputs.x
    )
    s = x_lora @ inputs.a.T
    base = base_linear(inputs)
    lora = s @ inputs.b.T
    return base + (lora * scale).to(dtype=base.dtype)


def measure(func: Callable[[], None], *, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        func()
    sync(device)
    if device.type != "cuda":
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)
        return times

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        start_events[idx].record()
        func()
        end_events[idx].record()
    sync(device)
    return [start_events[idx].elapsed_time(end_events[idx]) / 1000.0 for idx in range(iters)]


def measure_cuda_graph(func: Callable[[], None], *, warmup: int, iters: int, device: torch.device) -> list[float]:
    if device.type != "cuda":
        raise SystemExit("--cuda-graph requires a CUDA device")

    capture_warmup = 5
    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(capture_warmup):
            func()
    current.wait_stream(stream)
    sync(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        func()
    graph.replay()
    sync(device)

    for _ in range(warmup):
        graph.replay()
    sync(device)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        start_events[idx].record()
        graph.replay()
        end_events[idx].record()
    sync(device)
    return [start_events[idx].elapsed_time(end_events[idx]) / 1000.0 for idx in range(iters)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["xw_sb", "full_lora"], default="xw_sb")
    parser.add_argument("--backend", choices=["torch", "asym"], default="torch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=0, help="Overrides batch-size * seq-len when positive.")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.0,
        help="Dropout probability applied before X @ A.T for full_lora.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--precision",
        default="bf16",
        help="AsymGEMM execution precision and result label. Common values: bf16, fp8, fp4.",
    )
    parser.add_argument(
        "--asym-bf16-output-dtype",
        choices=["bf16", "bfloat16", "fp32", "float32"],
        default="bf16",
        help="Output buffer dtype for direct BF16 AsymGEMM before returning BF16. Default: bf16.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture one static run and benchmark CUDA graph replay.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-format",
        choices=["line", "csv", "none"],
        default="line",
        help="Stdout format. Use csv from driver scripts.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")
    if not 0.0 <= args.dropout_p < 1.0:
        raise SystemExit("--dropout-p must be in [0, 1)")
    return args


def build_result(args: argparse.Namespace, inputs: LoRAInputs, times: list[float], peak_hbm: int) -> dict[str, object]:
    median_ms = statistics.median(times) * 1000.0
    mean_ms = statistics.fmean(times) * 1000.0
    return {
        "operation": args.operation,
        "backend": args.backend,
        "pass": "backward" if args.backward else "forward",
        "device": str(inputs.device),
        "tokens": inputs.tokens,
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "in_features": int(args.in_features),
        "out_features": int(args.out_features),
        "rank": int(args.rank),
        "scale": float(args.scale),
        "dtype": str(inputs.dtype).removeprefix("torch."),
        "precision": str(args.precision),
        "asym_bf16_output_dtype": str(args.asym_bf16_output_dtype),
        "dropout_p": float(args.dropout_p),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "backward": bool(args.backward),
        "cuda_graph": bool(args.cuda_graph),
        "seed": int(args.seed),
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
        "peak_hbm_bytes": peak_hbm,
        "peak_hbm_gib": peak_hbm / (1024.0 ** 3),
    }


def fmt_csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def emit_result(result: dict[str, object], output_format: str) -> None:
    if output_format == "none":
        return
    if output_format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="")
        writer.writerow([fmt_csv_value(result[field]) for field in CSV_FIELDS])
        print(buffer.getvalue(), flush=True)
        return

    print(
        " ".join(
            [
                f"operation={result['operation']}",
                f"backend={result['backend']}",
                f"tokens={result['tokens']}",
                f"in={result['in_features']}",
                f"out={result['out_features']}",
                f"rank={result['rank']}",
                f"dtype=torch.{result['dtype']}",
                f"dropout_p={result['dropout_p']}",
                f"backward={result['backward']}",
                f"cuda_graph={result['cuda_graph']}",
                f"median_ms={float(result['median_ms']):.4f}",
                f"mean_ms={float(result['mean_ms']):.4f}",
                f"peak_hbm_gib={float(result['peak_hbm_gib']):.4f}",
            ]
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    inputs = LoRAInputs(args)
    if args.backward:
        inputs.enable_backward()
    op = xw_sb if args.operation == "xw_sb" else full_lora

    def run_once() -> None:
        pushed = nvtx_push(inputs.device, f"lora.{args.operation}")
        try:
            out = op(inputs, float(args.scale), float(args.dropout_p))
            if args.backward:
                assert inputs.grad is not None
                out.backward(inputs.grad)
                inputs.clear_grads()
        finally:
            nvtx_pop(pushed)

    sync(inputs.device)
    if inputs.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(inputs.device)
    measure_fn = measure_cuda_graph if args.cuda_graph else measure
    times = measure_fn(
        run_once,
        warmup=int(args.warmup),
        iters=int(args.iters),
        device=inputs.device,
    )
    peak_hbm = int(torch.cuda.max_memory_allocated(inputs.device)) if inputs.device.type == "cuda" else 0
    result = build_result(args, inputs, times, peak_hbm)
    emit_result(result, args.output_format)


if __name__ == "__main__":
    main()
