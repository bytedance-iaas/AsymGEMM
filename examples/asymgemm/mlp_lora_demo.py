from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from asym_gemm.training import AsymExecutionStats, measure_gpu_weight_allocation
from asym_gemm.training.mlp import (
    AsymLoRALinear,
    AsymMLP,
    TorchLoRALinear,
    TorchMLP,
    copy_lora,
    lora_parameters,
    optimizer_contains_only,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _process_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _scalar_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(abs(float(a.detach().float().item()) - float(b.detach().float().item())))


_copy_lora = copy_lora
_lora_parameters = lora_parameters
_optimizer_contains_only = optimizer_contains_only


def _grad_parity(asym: AsymMLP, ref: TorchMLP) -> Dict[str, float]:
    return {
        "fc1_lora_a": _max_abs(asym.fc1.lora_a.grad, ref.fc1.lora_a.grad),
        "fc1_lora_b": _max_abs(asym.fc1.lora_b.grad, ref.fc1.lora_b.grad),
        "fc2_lora_a": _max_abs(asym.fc2.lora_a.grad, ref.fc2.lora_a.grad),
        "fc2_lora_b": _max_abs(asym.fc2.lora_b.grad, ref.fc2.lora_b.grad),
    }


def _timed_step(model: nn.Module, x: torch.Tensor, target: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out = model(x)
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    _sync(device)
    elapsed = time.perf_counter() - start
    peak_hbm = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return out, loss, elapsed, peak_hbm


def _memory_probe(
    *,
    mode: str,
    w1: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    alpha: float,
    backend: str,
    asym_precision: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    _clear_cuda(device)

    hbm_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    rss_before = _process_rss_bytes()

    if mode == "normal_gpu_resident":
        model = TorchMLP(w1, w2, rank=rank, alpha=alpha, device=device, dtype=dtype)
    elif mode == "asym_cpu_resident":
        stats = AsymExecutionStats()
        model = AsymMLP(
            w1,
            w2,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            dtype=dtype,
            precision=asym_precision,
        )
    else:
        raise ValueError(f"unknown memory probe mode: {mode}")

    _sync(device)
    model_hbm = int(torch.cuda.memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    model_cpu_accounted = int(getattr(model, "pinned_cpu_bytes", 0))
    cpu_resident_base_weight_bytes = int(getattr(model, "cpu_resident_base_weight_bytes", 0))
    rss_after_build = _process_rss_bytes()

    tokens = 128 if device.type == "cuda" else 8
    in_features = int(w1.shape[1])
    out_features = int(w2.shape[0])
    x = torch.randn(tokens, in_features, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn(tokens, out_features, device=device, dtype=dtype)
    opt = torch.optim.AdamW(_lora_parameters(model), lr=1e-2)

    start = time.perf_counter()
    out = model(x)
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    opt.step()
    _sync(device)
    step_seconds = time.perf_counter() - start

    model_cpu_after_step = int(getattr(model, "pinned_cpu_bytes", 0))
    peak_hbm = int(torch.cuda.max_memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    rss_after_step = _process_rss_bytes()
    if mode == "asym_cpu_resident":
        exec_stats = model.fc1.base.stats.as_dict()
        exec_stats_fc2 = model.fc2.base.stats.as_dict()
        # Both layers share the same stats object. Keep this branch explicit so
        # a future refactor that separates stats does not silently undercount.
        if exec_stats != exec_stats_fc2:
            exec_stats["fc2"] = exec_stats_fc2
    else:
        exec_stats = {}

    del out, loss, opt, x, target, model
    _clear_cuda(device)

    return {
        "mode": mode,
        "asym_precision": asym_precision if mode == "asym_cpu_resident" else None,
        "model_hbm_bytes": max(0, model_hbm),
        "peak_hbm_bytes": max(0, peak_hbm),
        "cpu_model_bytes_after_build": model_cpu_accounted,
        "cpu_model_bytes_after_step": model_cpu_after_step,
        "cpu_resident_base_weight_bytes": cpu_resident_base_weight_bytes,
        "rss_before_bytes": rss_before,
        "rss_after_build_bytes": rss_after_build,
        "rss_after_step_bytes": rss_after_step,
        "rss_build_delta_bytes": rss_after_build - rss_before,
        "rss_step_delta_bytes": rss_after_step - rss_before,
        "step_seconds": step_seconds,
        "execution_stats": exec_stats,
    }


def _memory_comparison(
    *,
    w1: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    alpha: float,
    backend: str,
    asym_precision: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict[str, Any]:
    normal = _memory_probe(
        mode="normal_gpu_resident",
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=device,
        dtype=dtype,
        seed=seed + 101,
    )
    asym = _memory_probe(
        mode="asym_cpu_resident",
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=device,
        dtype=dtype,
        seed=seed + 101,
    )
    return {
        "normal_gpu_resident": normal,
        "asym_cpu_resident": asym,
        "hbm_model_saved_bytes": normal["model_hbm_bytes"] - asym["model_hbm_bytes"],
        "hbm_peak_saved_bytes": normal["peak_hbm_bytes"] - asym["peak_hbm_bytes"],
        "cpu_model_extra_bytes": asym["cpu_model_bytes_after_step"] - normal["cpu_model_bytes_after_step"],
    }


def _warm_memory_paths(
    *,
    w1: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    alpha: float,
    backend: str,
    asym_precision: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    _memory_probe(
        mode="normal_gpu_resident",
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=device,
        dtype=dtype,
        seed=seed + 17,
    )
    _memory_probe(
        mode="asym_cpu_resident",
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=device,
        dtype=dtype,
        seed=seed + 17,
    )
    _clear_cuda(device)


def _run_demo_impl(
    *,
    backend: str = "asym",
    asym_precision: str = "bf16",
    report_path: Optional[Path] = None,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    asym_precision = str(asym_precision).lower()
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device != "cpu":
        dev = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        dev = torch.device("cpu")
        dtype = torch.float32
        if backend == "asym":
            raise RuntimeError("asym requires CUDA SM90/SM100 direct execution")

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    m = 128 if dev.type == "cuda" else 8
    in_features = 128 if dev.type == "cuda" else 16
    hidden = 256 if dev.type == "cuda" else 32
    out_features = 128 if dev.type == "cuda" else 16
    rank = 8 if dev.type == "cuda" else 4
    alpha = 16.0

    w1 = torch.randn(hidden, in_features, dtype=dtype)
    w2 = torch.randn(out_features, hidden, dtype=dtype)
    baseline_weight_hbm = 0
    if dev.type == "cuda":
        baseline_weight_hbm = measure_gpu_weight_allocation(w1, device=dev) + measure_gpu_weight_allocation(w2, device=dev)

    _warm_memory_paths(
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=dev,
        dtype=dtype,
        seed=seed,
    )
    memory_comparison = _memory_comparison(
        w1=w1,
        w2=w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        asym_precision=asym_precision,
        device=dev,
        dtype=dtype,
        seed=seed,
    )

    stats = AsymExecutionStats()
    asym_model = AsymMLP(
        w1,
        w2,
        rank=rank,
        alpha=alpha,
        backend=backend,
        stats=stats,
        device=dev,
        dtype=dtype,
        precision=asym_precision,
    )
    ref_model = TorchMLP(w1, w2, rank=rank, alpha=alpha, device=dev, dtype=dtype)
    _copy_lora(asym_model, ref_model)

    x = torch.randn(m, in_features, device=dev, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    target = torch.randn(m, out_features, device=dev, dtype=dtype)
    target_ref = target.detach().clone()

    opt = torch.optim.AdamW(_lora_parameters(asym_model), lr=1e-2)
    opt_ref = torch.optim.AdamW(_lora_parameters(ref_model), lr=1e-2)

    base_before = [asym_model.fc1.base.host_weight.weight.clone(), asym_model.fc2.base.host_weight.weight.clone()]

    out, loss, asym_time, peak_hbm = _timed_step(asym_model, x, target, dev)
    out_ref, loss_ref, ref_time, ref_peak_hbm = _timed_step(ref_model, x_ref, target_ref, dev)

    grad_parity = _grad_parity(asym_model, ref_model)
    input_grad_parity = _max_abs(x.grad, x_ref.grad)
    forward_parity = _max_abs(out, out_ref)
    loss_parity = _scalar_abs(loss, loss_ref)

    opt.step()
    opt_ref.step()
    opt.zero_grad(set_to_none=True)
    opt_ref.zero_grad(set_to_none=True)

    with torch.no_grad():
        loss_after_step = F.mse_loss(asym_model(x.detach()).float(), target.float())
        loss_ref_after_step = F.mse_loss(ref_model(x_ref.detach()).float(), target_ref.float())

    base_after = [asym_model.fc1.base.host_weight.weight, asym_model.fc2.base.host_weight.weight]
    frozen_base_unchanged = all(torch.equal(before, after) for before, after in zip(base_before, base_after))
    base_absent = _optimizer_contains_only(_lora_parameters(asym_model), opt)

    report: Dict[str, Any] = {
        "seed": seed,
        "backend_requested": backend,
        "asym_precision_requested": asym_precision,
        "asym_precision_effective": asym_model.precision,
        "device": str(dev),
        "dtype": str(dtype),
        "dims": {
            "tokens": m,
            "in_features": in_features,
            "hidden": hidden,
            "out_features": out_features,
            "rank": rank,
        },
        "forward_parity_max_abs": forward_parity,
        "scalar_loss_parity_abs": loss_parity,
        "input_grad_parity_max_abs": input_grad_parity,
        "lora_grad_parity_max_abs": grad_parity,
        "lora_grad_parity_worst_max_abs": max(grad_parity.values()),
        "loss_before_step": float(loss.detach().float().item()),
        "loss_after_step": float(loss_after_step.detach().float().item()),
        "reference_loss_before_step": float(loss_ref.detach().float().item()),
        "reference_loss_after_step": float(loss_ref_after_step.detach().float().item()),
        "optimizer_step_loss_moved": bool(float(loss_after_step.detach().float().item()) != float(loss.detach().float().item())),
        "frozen_base_unchanged": bool(frozen_base_unchanged),
        "base_absent_from_optimizer_state": bool(base_absent),
        "tf32_disabled": bool(
            dev.type != "cuda"
            or (not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32)
        ),
        "memory_warmup_performed": True,
        "number_of_asymgemm_calls": stats.asym_calls,
        "fallback_counts": stats.as_dict(),
        "direct_fetch_forward_used": stats.asym_forward_calls > 0,
        "direct_fetch_dx_used": stats.asym_dx_calls > 0,
        "pinned_cpu_bytes": asym_model.pinned_cpu_bytes,
        "peak_hbm": memory_comparison["asym_cpu_resident"]["peak_hbm_bytes"],
        "reference_peak_hbm": memory_comparison["normal_gpu_resident"]["peak_hbm_bytes"],
        "parity_joint_peak_hbm": peak_hbm,
        "parity_joint_reference_peak_hbm": ref_peak_hbm,
        "gpu_resident_baseline_weight_bytes": int(baseline_weight_hbm),
        "expected_hbm_saved_bytes": int(asym_model.expected_hbm_saved_bytes),
        "memory_comparison": memory_comparison,
        "timings": {
            "asym_or_fallback_step_seconds": asym_time,
            "torch_reference_step_seconds": ref_time,
        },
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_demo(
    *,
    backend: str = "asym",
    asym_precision: str = "bf16",
    report_path: Optional[Path] = None,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    prior_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return _run_demo_impl(
            backend=backend,
            asym_precision=asym_precision,
            report_path=report_path,
            seed=seed,
            device=device,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="asym", choices=["asym", "torch"])
    parser.add_argument("--asym-precision", default="bf16", choices=["bf16", "fp8", "fp4"])
    parser.add_argument("--report", default="reports/mlp_demo.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    report = run_demo(
        backend=args.backend,
        asym_precision=args.asym_precision,
        report_path=Path(args.report),
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
