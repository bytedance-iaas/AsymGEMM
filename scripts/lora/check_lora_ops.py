#!/usr/bin/env python3
"""Check dense LoRA operator gradients for torch vs AsymGEMM.

This is a correctness checker, not a timing profiler. It compares a dense
full-LoRA operator using a torch frozen base against the same operator using
AsymFrozenLinear for the frozen base. Gradients are accumulated over multiple
microbatches before comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

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


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    max_rel: float
    mean_abs: float
    rms_abs: float
    rel_l2: float


@dataclass(frozen=True)
class CheckResult:
    base_output: ErrorStats
    base_dx: ErrorStats
    output: ErrorStats
    x_grad: ErrorStats
    a_grad: ErrorStats
    b_grad: ErrorStats
    asym_forward_calls: int
    asym_dx_calls: int


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(DTYPES))
        raise SystemExit(f"unsupported dtype {name!r}; allowed: {allowed}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--asym-bf16-output-dtype",
        choices=["bf16", "bfloat16", "fp32", "float32"],
        default="bf16",
    )
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--l2-rtol",
        type=float,
        default=5e-3,
        help="Relative L2 tolerance. This catches tensor-level accuracy when BF16 max error is dominated by a few entries.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--zero-b",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero LoRA B to mirror PEFT init. Default keeps B random so A.grad is nonzero.",
    )
    parser.add_argument(
        "--require-asym-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require direct AsymGEMM forward and dx calls for backend=asym.",
    )
    args = parser.parse_args()
    if min(args.tokens, args.in_features, args.out_features, args.rank, args.accum_steps) <= 0:
        raise SystemExit("tokens, feature sizes, rank, and accum-steps must be positive")
    if args.scale == 0.0:
        raise SystemExit("--scale must be nonzero for useful LoRA gradient checking")
    if args.atol < 0.0 or args.rtol < 0.0 or args.l2_rtol < 0.0:
        raise SystemExit("--atol, --rtol, and --l2-rtol must be non-negative")
    return args


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype, generator=generator)


def full_lora(
    x: torch.Tensor,
    base_weight_or_module: torch.Tensor | torch.nn.Module,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    if isinstance(base_weight_or_module, torch.Tensor):
        base = x @ base_weight_or_module.T
    else:
        base = base_weight_or_module(x)
    hidden = x @ a.T
    lora = hidden @ b.T
    return base + (lora * scale).to(dtype=base.dtype)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> ErrorStats:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1.0e-6)
    expected_norm = torch.linalg.vector_norm(expected_f)
    if float(expected_norm.item()) == 0.0:
        rel_l2 = float(torch.linalg.vector_norm(diff).item())
    else:
        rel_l2 = float((torch.linalg.vector_norm(diff) / expected_norm).item())
    return ErrorStats(
        max_abs=float(diff.max().item()),
        max_rel=float((diff / denom).max().item()),
        mean_abs=float(diff.mean().item()),
        rms_abs=float(torch.sqrt((diff * diff).mean()).item()),
        rel_l2=rel_l2,
    )


def assert_close(name: str, stats: ErrorStats, *, atol: float, rtol: float, l2_rtol: float) -> None:
    if stats.max_abs > atol and stats.max_rel > rtol and stats.rel_l2 > l2_rtol:
        raise AssertionError(
            f"{name} mismatch: max_abs={stats.max_abs:.6g} > {atol:.6g} "
            f"and max_rel={stats.max_rel:.6g} > {rtol:.6g} "
            f"and rel_l2={stats.rel_l2:.6g} > {l2_rtol:.6g}"
        )


def require_finite(name: str, tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise AssertionError(f"{name} grad is None")
    if not bool(torch.isfinite(tensor.detach().float()).all().item()):
        raise AssertionError(f"{name} contains non-finite values")
    return tensor


def run_check(args: argparse.Namespace) -> CheckResult:
    from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    dtype = parse_dtype(args.dtype)
    if args.precision == "bf16" and dtype is not torch.bfloat16:
        raise SystemExit("--precision bf16 requires --dtype bf16 for this checker")

    generator = make_generator(device, int(args.seed))
    weight = randn(
        (int(args.out_features), int(args.in_features)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    a_init = randn((int(args.rank), int(args.in_features)), device=device, dtype=dtype, generator=generator)
    b_init = randn((int(args.out_features), int(args.rank)), device=device, dtype=dtype, generator=generator)
    if bool(args.zero_b):
        b_init.zero_()

    torch_xs = []
    asym_xs = []
    grad_outs = []
    for _ in range(int(args.accum_steps)):
        x = randn((int(args.tokens), int(args.in_features)), device=device, dtype=dtype, generator=generator)
        grad = randn((int(args.tokens), int(args.out_features)), device=device, dtype=dtype, generator=generator)
        torch_xs.append(x.detach().clone().requires_grad_(True))
        asym_xs.append(x.detach().clone().requires_grad_(True))
        grad_outs.append(grad.detach().clone())

    torch_w = weight.detach().clone()
    torch_a = a_init.detach().clone().requires_grad_(True)
    torch_b = b_init.detach().clone().requires_grad_(True)

    stats = AsymExecutionStats()
    asym_base = AsymFrozenLinear(
        weight.detach().cpu(),
        backend="asym",
        pin_memory=device.type == "cuda",
        stats=stats,
        precision=str(args.precision),
        bf16_output_dtype=str(args.asym_bf16_output_dtype),
    )
    asym_base.profile_name = "check_lora_ops"
    asym_a = a_init.detach().clone().requires_grad_(True)
    asym_b = b_init.detach().clone().requires_grad_(True)

    base_output_errors: list[ErrorStats] = []
    for torch_x, asym_x, grad_out in zip(torch_xs, asym_xs, grad_outs):
        torch_base_x = torch_x.detach().clone().requires_grad_(True)
        asym_base_x = asym_x.detach().clone().requires_grad_(True)
        torch_base_out = torch_base_x @ torch_w.T
        asym_base_out = asym_base(asym_base_x)
        base_output_errors.append(error_stats(asym_base_out, torch_base_out))
        torch_base_out.backward(grad_out)
        asym_base_out.backward(grad_out)
        torch_x.grad = torch_base_x.grad.detach().clone()
        asym_x.grad = asym_base_x.grad.detach().clone()

    base_dx = error_stats(
        torch.stack([require_finite("asym base x", x.grad).float() for x in asym_xs]),
        torch.stack([require_finite("torch base x", x.grad).float() for x in torch_xs]),
    )

    for x in torch_xs + asym_xs:
        x.grad = None

    output_errors: list[ErrorStats] = []
    for torch_x, asym_x, grad_out in zip(torch_xs, asym_xs, grad_outs):
        torch_out = full_lora(torch_x, torch_w, torch_a, torch_b, scale=float(args.scale))
        asym_out = full_lora(asym_x, asym_base, asym_a, asym_b, scale=float(args.scale))
        output_errors.append(error_stats(asym_out, torch_out))
        torch_out.backward(grad_out)
        asym_out.backward(grad_out)

    if asym_base.host_weight.weight.requires_grad:
        raise AssertionError("AsymGEMM host weight unexpectedly requires grad")
    if asym_base.host_weight.weight.grad is not None:
        raise AssertionError("AsymGEMM host weight unexpectedly received grad")

    torch_x_grad = torch.stack([require_finite("torch x", x.grad).float() for x in torch_xs])
    asym_x_grad = torch.stack([require_finite("asym x", x.grad).float() for x in asym_xs])
    torch_a_grad = require_finite("torch A", torch_a.grad)
    asym_a_grad = require_finite("asym A", asym_a.grad)
    torch_b_grad = require_finite("torch B", torch_b.grad)
    asym_b_grad = require_finite("asym B", asym_b.grad)

    result = CheckResult(
        base_output=ErrorStats(
            max_abs=max(item.max_abs for item in base_output_errors),
            max_rel=max(item.max_rel for item in base_output_errors),
            mean_abs=max(item.mean_abs for item in base_output_errors),
            rms_abs=max(item.rms_abs for item in base_output_errors),
            rel_l2=max(item.rel_l2 for item in base_output_errors),
        ),
        base_dx=base_dx,
        output=ErrorStats(
            max_abs=max(item.max_abs for item in output_errors),
            max_rel=max(item.max_rel for item in output_errors),
            mean_abs=max(item.mean_abs for item in output_errors),
            rms_abs=max(item.rms_abs for item in output_errors),
            rel_l2=max(item.rel_l2 for item in output_errors),
        ),
        x_grad=error_stats(asym_x_grad, torch_x_grad),
        a_grad=error_stats(asym_a_grad, torch_a_grad),
        b_grad=error_stats(asym_b_grad, torch_b_grad),
        asym_forward_calls=int(stats.asym_forward_calls),
        asym_dx_calls=int(stats.asym_dx_calls),
    )

    if bool(args.require_asym_calls):
        expected_calls = 2 * int(args.accum_steps)
        if result.asym_forward_calls != expected_calls:
            raise AssertionError(
                f"expected {expected_calls} AsymGEMM forward calls, got {result.asym_forward_calls}"
            )
        if result.asym_dx_calls != expected_calls:
            raise AssertionError(f"expected {expected_calls} AsymGEMM dx calls, got {result.asym_dx_calls}")

    assert_close("base output", result.base_output, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    assert_close("base dx", result.base_dx, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    assert_close("output", result.output, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    assert_close("x.grad", result.x_grad, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    assert_close("A.grad", result.a_grad, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    assert_close("B.grad", result.b_grad, atol=float(args.atol), rtol=float(args.rtol), l2_rtol=float(args.l2_rtol))
    return result


def print_result(args: argparse.Namespace, result: CheckResult) -> None:
    print("LoRA op gradient check: PASS")
    print(
        "config "
        f"device={args.device} tokens={args.tokens} in={args.in_features} out={args.out_features} "
        f"rank={args.rank} scale={args.scale:g} dtype={args.dtype} precision={args.precision} "
        f"accum_steps={args.accum_steps} atol={args.atol:g} rtol={args.rtol:g} l2_rtol={args.l2_rtol:g}"
    )
    for name, stats in (
        ("base_output", result.base_output),
        ("base_dx", result.base_dx),
        ("output", result.output),
        ("x.grad", result.x_grad),
        ("A.grad", result.a_grad),
        ("B.grad", result.b_grad),
    ):
        print(
            f"{name}: max_abs={stats.max_abs:.6g} max_rel={stats.max_rel:.6g} "
            f"mean_abs={stats.mean_abs:.6g} rms_abs={stats.rms_abs:.6g} rel_l2={stats.rel_l2:.6g}"
        )
    print(f"asym_calls forward={result.asym_forward_calls} dx={result.asym_dx_calls}")


def main() -> None:
    args = parse_args()
    result = run_check(args)
    print_result(args, result)


if __name__ == "__main__":
    main()
