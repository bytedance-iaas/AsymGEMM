#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class Accuracy:
    case: str
    max_abs: float
    mean_abs: float
    max_abs_pct: float
    mean_abs_pct: float
    max_rel: float
    allclose: bool


@dataclass
class Timing:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class TimingAggregate:
    name: str
    median_ms_avg: float
    median_ms_std: float
    mean_ms_avg: float
    min_ms_avg: float
    max_ms_avg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Torch BF16 and AsymGEMM BF16/FP8 2D MM with and without the "
            "dX-style transposed weight path."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--m", type=int, default=8192, help="token rows")
    parser.add_argument("--k", type=int, default=2048, help="hidden/input dim")
    parser.add_argument(
        "--n",
        type=int,
        default=768,
        help="expert output dim; default is one Qwen3-30B-A3B expert projection",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=50, help="number of full timing/accuracy repeats to average")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--compiled-dims", default="mnk")
    parser.add_argument("--atol", type=float, default=5e-1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=1.25,
        help="report transpose as much slower if transpose/nontranspose ratio exceeds this",
    )
    parser.add_argument("--json-output", default="", help="optional path for machine-readable results")
    return parser.parse_args()


def require_cuda(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this benchmark requires a CUDA GPU.")
    if device.type != "cuda":
        raise SystemExit(f"expected a CUDA device, got {device}.")


def require_asym_gemm() -> object:
    try:
        import asym_gemm  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "failed to import asym_gemm. Run this script from the AsymGEMM repo root "
            "or install the package editable first."
        ) from exc
    return asym_gemm


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def make_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    x_cpu = torch.randn(args.m, args.k, generator=gen, dtype=torch.float32) * float(args.scale)
    g_cpu = torch.randn(args.m, args.n, generator=gen, dtype=torch.float32) * float(args.scale)
    w_cpu = torch.randn(args.n, args.k, generator=gen, dtype=torch.float32) * float(args.scale)

    x = x_cpu.to(device=device, dtype=torch.bfloat16)
    g = g_cpu.to(device=device, dtype=torch.bfloat16)
    w_gpu = w_cpu.to(device=device, dtype=torch.bfloat16)
    w_pinned = w_cpu.to(dtype=torch.bfloat16).contiguous().pin_memory()
    return x.contiguous(), g.contiguous(), w_gpu.contiguous(), w_pinned


def pin_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous().pin_memory()


def quantize_fp8_group_weight(weight_cpu: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from asym_gemm.utils import per_block_cast_to_fp8

    weight_cuda = weight_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=weight_cpu.is_pinned()).contiguous()
    values_2d, scales_2d = per_block_cast_to_fp8(weight_cuda, use_ue8m0=True, gran_k=128)
    values = torch.empty((1, *values_2d.shape), device=device, dtype=values_2d.dtype)
    values[0].copy_(values_2d)
    scales = scales_2d.unsqueeze(0).contiguous()
    return pin_cpu(values), pin_cpu(scales)


def quantize_fp8_activation(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from asym_gemm.utils import per_token_cast_to_fp8

    return per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)


def asym_bf16_call(
    asym_gemm: object,
    a: torch.Tensor,
    w_group_cpu: torch.Tensor,
    out: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    transpose_b: bool,
    compiled_dims: str,
) -> None:
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(  # type: ignore[attr-defined]
        a,
        w_group_cpu,
        out,
        offsets,
        experts,
        2,
        compiled_dims,
        transpose_b,
    )


def asym_fp8_call(
    asym_gemm: object,
    a: torch.Tensor,
    qweight_group_cpu: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    compiled_dims: str,
) -> None:
    a_quantized = quantize_fp8_activation(a)
    asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(  # type: ignore[attr-defined]
        a_quantized,
        qweight_group_cpu,
        out,
        offsets,
        experts,
        2,
        recipe=(1, 128, 128),
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=False,
    )


def measure(
    name: str,
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Timing:
    fn()
    sync(device)
    for _ in range(warmup):
        fn()
    sync(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    sync(device)

    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    return Timing(
        name=name,
        median_ms=float(statistics.median(times)),
        mean_ms=float(statistics.fmean(times)),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
    )


def accuracy(case: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> Accuracy:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-6)
    ref_max_abs = float(expected_f.abs().max().item())
    ref_mean_abs = float(expected_f.abs().mean().item())
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    return Accuracy(
        case=case,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_abs_pct=100.0 * max_abs / ref_max_abs if ref_max_abs > 0 else 0.0,
        mean_abs_pct=100.0 * mean_abs / ref_mean_abs if ref_mean_abs > 0 else 0.0,
        max_rel=float((diff / denom).max().item()),
        allclose=bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)),
    )


def format_timing_table(timings: list[Timing]) -> str:
    lines = [
        f"{'op':28s} {'median_ms':>10s} {'mean_ms':>10s} {'min_ms':>10s} {'max_ms':>10s}",
        f"{'-' * 28} {'-' * 10:>10s} {'-' * 10:>10s} {'-' * 10:>10s} {'-' * 10:>10s}",
    ]
    for t in timings:
        lines.append(f"{t.name:28s} {t.median_ms:10.4f} {t.mean_ms:10.4f} {t.min_ms:10.4f} {t.max_ms:10.4f}")
    return "\n".join(lines)


def format_timing_aggregate_table(timings: list[TimingAggregate]) -> str:
    lines = [
        f"{'op':32s} {'median_avg':>11s} {'median_std':>11s} {'mean_avg':>10s} {'min_avg':>10s} {'max_avg':>10s}",
        f"{'-' * 32} {'-' * 11:>11s} {'-' * 11:>11s} {'-' * 10:>10s} {'-' * 10:>10s} {'-' * 10:>10s}",
    ]
    for t in timings:
        lines.append(
            f"{t.name:32s} {t.median_ms_avg:11.4f} {t.median_ms_std:11.4f} "
            f"{t.mean_ms_avg:10.4f} {t.min_ms_avg:10.4f} {t.max_ms_avg:10.4f}"
        )
    return "\n".join(lines)


def format_accuracy_table(rows: list[Accuracy]) -> str:
    lines = [
        (
            f"{'case':18s} {'allclose':>8s} {'max_abs':>22s} {'mean_abs':>22s}"
        ),
        (
            f"{'-' * 18} {'-' * 8:>8s} {'-' * 22:>22s} {'-' * 22:>22s}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row.case:18s} {str(row.allclose):>8s} "
            f"{row.max_abs:10.6g} ({row.max_abs_pct:7.3f}%) "
            f"{row.mean_abs:10.6g} ({row.mean_abs_pct:7.3f}%)"
        )
    return "\n".join(lines)


def average(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def aggregate_accuracy(runs: list[list[Accuracy]]) -> list[Accuracy]:
    if not runs:
        return []
    names = [row.case for row in runs[0]]
    rows: list[Accuracy] = []
    for idx, name in enumerate(names):
        selected = [run[idx] for run in runs]
        rows.append(
            Accuracy(
                case=name,
                max_abs=average([row.max_abs for row in selected]),
                mean_abs=average([row.mean_abs for row in selected]),
                max_abs_pct=average([row.max_abs_pct for row in selected]),
                mean_abs_pct=average([row.mean_abs_pct for row in selected]),
                max_rel=average([row.max_rel for row in selected]),
                allclose=all(row.allclose for row in selected),
            )
        )
    return rows


def aggregate_timings(runs: list[list[Timing]]) -> list[TimingAggregate]:
    if not runs:
        return []
    names = [row.name for row in runs[0]]
    rows: list[TimingAggregate] = []
    for idx, name in enumerate(names):
        selected = [run[idx] for run in runs]
        medians = [row.median_ms for row in selected]
        rows.append(
            TimingAggregate(
                name=name,
                median_ms_avg=average(medians),
                median_ms_std=sample_std(medians),
                mean_ms_avg=average([row.mean_ms for row in selected]),
                min_ms_avg=average([row.min_ms for row in selected]),
                max_ms_avg=average([row.max_ms for row in selected]),
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise SystemExit("--m, --n, and --k must be positive.")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise SystemExit("--warmup must be non-negative and --iters/--repeats must be positive.")
    alignment = 128 if args.precision == "fp8" else 8
    if args.n % alignment or args.k % alignment:
        raise SystemExit(f"{args.precision.upper()} AsymGEMM requires N and K to be multiples of {alignment}.")
    if args.precision == "bf16" and args.n % 64:
        raise SystemExit("transpose_b BF16 AsymGEMM currently requires the transpose inner dim N to be a multiple of 64.")

    device = torch.device(args.device)
    require_cuda(device)
    torch.cuda.set_device(device)
    asym_gemm = require_asym_gemm()

    major, minor = torch.cuda.get_device_capability(device)
    if major not in {9, 10}:
        raise SystemExit(f"AsymGEMM BF16/FP8 path requires SM90/SM100, got sm_{major}{minor}.")
    if args.precision == "bf16" and not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        raise SystemExit("asym_gemm is missing m_grouped_bf16_asym_gemm_nt_contiguous.")
    if args.precision == "fp8" and not hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_contiguous"):
        raise SystemExit("asym_gemm is missing m_grouped_fp8_asym_gemm_nt_contiguous.")

    x, g, w_gpu, w_cpu = make_inputs(args, device)
    w_group_cpu = w_cpu.unsqueeze(0)
    w_t_cpu = pin_cpu(w_cpu.t().contiguous())
    w_t_group_cpu = w_t_cpu.unsqueeze(0)
    w_mT_group_cpu = torch.as_strided(
        w_cpu,
        size=(1, args.k, args.n),
        stride=(args.n * args.k, 1, args.k),
    )
    qweight_group_cpu = None
    qweight_t_group_cpu = None
    if args.precision == "fp8":
        qweight_group_cpu = quantize_fp8_group_weight(w_cpu, device)
        qweight_t_group_cpu = quantize_fp8_group_weight(w_cpu.t().contiguous(), device)
    offsets = torch.tensor([0, args.m], device=device, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=device, dtype=torch.int32)

    torch_nt_out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
    asym_nt_out = torch.empty_like(torch_nt_out)
    torch_t_out = torch.empty((args.m, args.k), device=device, dtype=torch.bfloat16)
    asym_t_out = torch.empty_like(torch_t_out)
    asym_stored_t_out = torch.empty_like(torch_t_out)
    asym_mT_t_out = torch.empty_like(torch_t_out)
    w_gpu_t = w_gpu.t()

    torch_nt = lambda: torch.mm(x, w_gpu_t, out=torch_nt_out)
    if args.precision == "bf16":
        asym_nt = lambda: asym_bf16_call(
            asym_gemm,
            x,
            w_group_cpu,
            asym_nt_out,
            offsets,
            experts,
            transpose_b=False,
            compiled_dims=args.compiled_dims,
        )
    else:
        assert qweight_group_cpu is not None
        asym_nt = lambda: asym_fp8_call(
            asym_gemm,
            x,
            qweight_group_cpu,
            asym_nt_out,
            offsets,
            experts,
            compiled_dims=args.compiled_dims,
        )
    torch_t = lambda: torch.mm(g, w_gpu, out=torch_t_out)
    asym_t = None
    if args.precision == "bf16":
        asym_t = lambda: asym_bf16_call(
            asym_gemm,
            g,
            w_group_cpu,
            asym_t_out,
            offsets,
            experts,
            transpose_b=True,
            compiled_dims=args.compiled_dims,
        )
        asym_stored_t = lambda: asym_bf16_call(
            asym_gemm,
            g,
            w_t_group_cpu,
            asym_stored_t_out,
            offsets,
            experts,
            transpose_b=False,
            compiled_dims=args.compiled_dims,
        )
        asym_mT_t = lambda: asym_bf16_call(
            asym_gemm,
            g,
            w_mT_group_cpu,
            asym_mT_t_out,
            offsets,
            experts,
            transpose_b=False,
            compiled_dims=args.compiled_dims,
        )
    else:
        assert qweight_t_group_cpu is not None
        asym_mT_t = None
        asym_stored_t = lambda: asym_fp8_call(
            asym_gemm,
            g,
            qweight_t_group_cpu,
            asym_stored_t_out,
            offsets,
            experts,
            compiled_dims=args.compiled_dims,
        )

    # Preflight forces JIT compilation before measured repeats.
    torch_nt()
    asym_nt()
    torch_t()
    if asym_t is not None:
        asym_t()
    if asym_mT_t is not None:
        asym_mT_t()
    asym_stored_t()
    sync(device)

    accuracy_runs: list[list[Accuracy]] = []
    timing_runs: list[list[Timing]] = []
    ratio_runs: list[dict[str, float | None]] = []
    for _ in range(args.repeats):
        torch_nt()
        asym_nt()
        torch_t()
        if asym_t is not None:
            asym_t()
        if asym_mT_t is not None:
            asym_mT_t()
        asym_stored_t()
        sync(device)

        acc_rows = [
            accuracy("nontranspose", asym_nt_out, torch_nt_out, atol=args.atol, rtol=args.rtol),
        ]
        if asym_t is not None:
            acc_rows.append(accuracy("transpose", asym_t_out, torch_t_out, atol=args.atol, rtol=args.rtol))
        if asym_mT_t is not None:
            acc_rows.append(accuracy("transpose_mT", asym_mT_t_out, torch_t_out, atol=args.atol, rtol=args.rtol))
        acc_rows.append(accuracy("stored_transpose", asym_stored_t_out, torch_t_out, atol=args.atol, rtol=args.rtol))
        accuracy_runs.append(acc_rows)

        timings = [
            measure("torch_nontranspose", torch_nt, warmup=args.warmup, iters=args.iters, device=device),
            measure(f"asym_{args.precision}_nontranspose", asym_nt, warmup=args.warmup, iters=args.iters, device=device),
            measure("torch_transpose", torch_t, warmup=args.warmup, iters=args.iters, device=device),
        ]
        if asym_t is not None:
            timings.append(measure(f"asym_{args.precision}_transpose", asym_t, warmup=args.warmup, iters=args.iters, device=device))
        if asym_mT_t is not None:
            timings.append(
                measure(
                    f"asym_{args.precision}_transpose_mT",
                    asym_mT_t,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
            )
        timings.append(
            measure(
                f"asym_{args.precision}_stored_transpose",
                asym_stored_t,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
        )
        timing_runs.append(timings)

        by_name = {t.name: t for t in timings}
        torch_ratio_i = by_name["torch_transpose"].median_ms / by_name["torch_nontranspose"].median_ms
        asym_ratio_i = (
            by_name[f"asym_{args.precision}_transpose"].median_ms
            / by_name[f"asym_{args.precision}_nontranspose"].median_ms
            if asym_t is not None
            else None
        )
        asym_stored_ratio_i = (
            by_name[f"asym_{args.precision}_stored_transpose"].median_ms
            / by_name[f"asym_{args.precision}_nontranspose"].median_ms
        )
        asym_mT_ratio_i = (
            by_name[f"asym_{args.precision}_transpose_mT"].median_ms
            / by_name[f"asym_{args.precision}_nontranspose"].median_ms
            if asym_mT_t is not None
            else None
        )
        stored_vs_true_ratio_i = (
            by_name[f"asym_{args.precision}_stored_transpose"].median_ms
            / by_name[f"asym_{args.precision}_transpose"].median_ms
            if asym_t is not None
            else None
        )
        mT_vs_true_ratio_i = (
            by_name[f"asym_{args.precision}_transpose_mT"].median_ms
            / by_name[f"asym_{args.precision}_transpose"].median_ms
            if asym_t is not None and asym_mT_t is not None
            else None
        )
        ratio_runs.append(
            {
                "torch_ratio": torch_ratio_i,
                "asym_ratio": asym_ratio_i,
                "asym_stored_ratio": asym_stored_ratio_i,
                "asym_mT_ratio": asym_mT_ratio_i,
                "stored_vs_true_ratio": stored_vs_true_ratio_i,
                "mT_vs_true_ratio": mT_vs_true_ratio_i,
            }
        )

    acc_rows = aggregate_accuracy(accuracy_runs)
    timing_rows = aggregate_timings(timing_runs)

    def avg_ratio(name: str) -> float | None:
        values = [run[name] for run in ratio_runs if run[name] is not None]
        return average([float(value) for value in values]) if values else None

    torch_ratio = avg_ratio("torch_ratio")
    asym_ratio = avg_ratio("asym_ratio")
    asym_stored_ratio = avg_ratio("asym_stored_ratio")
    asym_mT_ratio = avg_ratio("asym_mT_ratio")
    stored_vs_true_ratio = avg_ratio("stored_vs_true_ratio")
    mT_vs_true_ratio = avg_ratio("mT_vs_true_ratio")
    transpose_correct = all(row.allclose for row in acc_rows)
    asym_much_slower = bool(asym_ratio is not None and asym_ratio > float(args.slow_threshold))
    asym_stored_much_slower = bool(asym_stored_ratio is not None and asym_stored_ratio > float(args.slow_threshold))

    flops = 2.0 * float(args.m) * float(args.n) * float(args.k)
    print("Profile: BF16 Torch 2D MM vs AsymGEMM 2D MM/transposed MM")
    print(f"device=sm_{major}{minor} shape: M={args.m}, N={args.n}, K={args.k}, flops/op={flops / 1e12:.3f} TFLOP")
    print(f"precision: torch=bf16, asym={args.precision}")
    print(f"repeats={args.repeats}, warmup={args.warmup}, iters={args.iters}")
    print("dtype: logical inputs=bf16, torch weights=bf16, outputs=bf16")
    if args.precision == "fp8":
        print("fp8 note: AsymGEMM quantizes activations per call and uses CPU-pinned FP8 quantized weights/scales.")
        print("fp8 note: true transpose_b is not exposed by the FP8 binding; only stored-transposed W is profiled.")
    print(f"nontranspose: X[{args.m},{args.k}] @ W[{args.n},{args.k}].T -> [{args.m},{args.n}]")
    print(f"transpose:    G[{args.m},{args.n}] @ W[{args.n},{args.k}]   -> [{args.m},{args.k}]")
    if args.precision == "bf16":
        print(f"W.mT view:    G[{args.m},{args.n}] @ W.mT[{args.k},{args.n}].T -> [{args.m},{args.k}]")
    print(f"stored W.T:   G[{args.m},{args.n}] @ WT[{args.k},{args.n}].T -> [{args.m},{args.k}]")
    print("tensor sharing: Torch and AsymGEMM use the exact same input and W within each row above.")
    if args.n == args.k:
        print("input note: N == K, so transpose and nontranspose inputs have the same shape but are separately sampled.")
    else:
        print("input note: transpose input has shape [M,N], so it cannot be the same matrix as nontranspose input [M,K].")
    print()
    print("Correctness vs Torch (averaged across repeats)")
    print(format_accuracy_table(acc_rows))
    print()
    print("Timing (averaged across repeats; median columns are per-repeat medians)")
    print(format_timing_aggregate_table(timing_rows))
    print()
    print("Ratios")
    assert torch_ratio is not None
    print(f"torch transpose / nontranspose median avg: {torch_ratio:.3f}x")
    if asym_ratio is not None:
        print(f"asym  transpose / nontranspose median avg: {asym_ratio:.3f}x")
    else:
        print("asym  transpose / nontranspose median avg: N/A (not exposed for this precision)")
    if asym_mT_ratio is not None:
        print(f"asym  transpose_mT / nontranspose median avg: {asym_mT_ratio:.3f}x")
    assert asym_stored_ratio is not None
    print(f"asym  stored-transpose / nontranspose median avg: {asym_stored_ratio:.3f}x")
    if stored_vs_true_ratio is not None:
        print(f"asym  stored-transpose / transpose median avg: {stored_vs_true_ratio:.3f}x")
    if mT_vs_true_ratio is not None:
        print(f"asym  transpose_mT / transpose median avg: {mT_vs_true_ratio:.3f}x")
    print()
    print("Answer")
    print(f"transpose correctness: {'PASS' if transpose_correct else 'FAIL'}")
    print(
        "asym transpose much slower than asym nontranspose: "
        f"{'YES' if asym_much_slower else 'NO'} "
        f"(threshold={float(args.slow_threshold):.2f}x)"
    )
    print(
        "asym stored-transpose much slower than asym nontranspose: "
        f"{'YES' if asym_stored_much_slower else 'NO'} "
        f"(threshold={float(args.slow_threshold):.2f}x)"
    )

    result = {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "device": f"sm_{major}{minor}",
        "atol": args.atol,
        "rtol": args.rtol,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "iters": args.iters,
        "slow_threshold": args.slow_threshold,
        "accuracy": [asdict(row) for row in acc_rows],
        "timing": [asdict(row) for row in timing_rows],
        "raw_repeats": {
            "accuracy": [[asdict(row) for row in run] for run in accuracy_runs],
            "timing": [[asdict(row) for row in run] for run in timing_runs],
            "ratios": ratio_runs,
        },
        "ratios": {
            "torch_transpose_over_nontranspose": torch_ratio,
            f"asym_{args.precision}_transpose_over_nontranspose": asym_ratio,
            f"asym_{args.precision}_transpose_mT_over_nontranspose": asym_mT_ratio,
            f"asym_{args.precision}_stored_transpose_over_nontranspose": asym_stored_ratio,
            f"asym_{args.precision}_stored_transpose_over_transpose": stored_vs_true_ratio,
            f"asym_{args.precision}_transpose_mT_over_transpose": mT_vs_true_ratio,
        },
        "answer": {
            "transpose_correct": transpose_correct,
            "asym_transpose_much_slower": asym_much_slower,
            "asym_stored_transpose_much_slower": asym_stored_much_slower,
        },
    }
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote JSON: {output_path}")

    return 0 if transpose_correct else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        raise
