#!/usr/bin/env python3
"""M4.5 profiling CLI for AsymGEMM SFT micro workloads."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    profiling = importlib.import_module("asym_gemm.training.profiling")
except Exception:
    profiling = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _device(name: str | None) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype_for(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percent(value: float, total: float) -> float:
    return 0.0 if total <= 0.0 else 100.0 * value / total


class TimerBook:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, float] = {}
        self.step_seconds: list[float] = []

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        _sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.values[name] = self.values.get(name, 0.0) + (time.perf_counter() - start)


@contextmanager
def _forward_hooks(model: nn.Module, book: TimerBook, rules: Mapping[str, str]) -> Iterator[None]:
    handles: list[Any] = []
    starts: dict[int, float] = {}

    def pre(_: nn.Module, __: tuple[Any, ...], *, key: str) -> None:
        _sync(book.device)
        starts[id(_)] = time.perf_counter()

    def post(module: nn.Module, __: tuple[Any, ...], ___: Any, *, key: str) -> None:
        _sync(book.device)
        start = starts.pop(id(module), None)
        if start is not None:
            book.values[key] = book.values.get(key, 0.0) + (time.perf_counter() - start)

    for module in model.modules():
        cls_name = type(module).__name__
        for needle, key in rules.items():
            if needle in cls_name:
                handles.append(module.register_forward_pre_hook(lambda m, i, k=key: pre(m, i, key=k)))
                handles.append(module.register_forward_hook(lambda m, i, o, k=key: post(m, i, o, key=k)))
                break
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _patched_timers(module: Any, book: TimerBook, names: Mapping[str, str]) -> Iterator[None]:
    originals: dict[str, Any] = {}

    def make_wrapper(fn: Callable[..., Any], key: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with book.time(key):
                return fn(*args, **kwargs)

        return wrapper

    for attr, key in names.items():
        if hasattr(module, attr):
            originals[attr] = getattr(module, attr)
            setattr(module, attr, make_wrapper(originals[attr], key))
    try:
        yield
    finally:
        for attr, original in originals.items():
            setattr(module, attr, original)


@contextmanager
def _patched_base_dispatch(book: TimerBook) -> Iterator[None]:
    frozen_linear = importlib.import_module("asym_gemm.training.frozen_linear")

    original = frozen_linear._dispatch_nt

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        phase = str(kwargs.get("phase", "forward"))
        key = "base_dx_seconds" if phase == "dx" else "base_forward_seconds"
        with book.time(key):
            return original(*args, **kwargs)

    frozen_linear._dispatch_nt = wrapper
    try:
        yield
    finally:
        frozen_linear._dispatch_nt = original


def _execution_stats(model: nn.Module) -> dict[str, Any]:
    stats = getattr(model, "stats", None)
    if stats is not None and hasattr(stats, "as_dict"):
        return stats.as_dict()
    for module in model.modules():
        stats = getattr(module, "stats", None)
        if stats is not None and hasattr(stats, "as_dict"):
            return stats.as_dict()
    return {}


def _host_weight_bytes(model: nn.Module) -> dict[str, int]:
    weight_bytes = 0
    weight_t_bytes = 0
    pinned_bytes = 0
    count = 0
    seen: set[int] = set()
    for module in model.modules():
        host_weight = getattr(module, "host_weight", None)
        if host_weight is None:
            base = getattr(module, "base_layer", None) or getattr(module, "base", None)
            host_weight = getattr(base, "host_weight", None)
        if host_weight is None:
            continue
        host_id = id(host_weight)
        if host_id in seen:
            continue
        seen.add(host_id)
        count += 1
        weight = getattr(host_weight, "weight", None)
        transpose = getattr(host_weight, "_transpose", None)
        if isinstance(weight, torch.Tensor):
            weight_bytes += int(weight.numel() * weight.element_size())
        if isinstance(transpose, torch.Tensor):
            weight_t_bytes += int(transpose.numel() * transpose.element_size())
        pinned_bytes += int(getattr(host_weight, "pinned_cpu_bytes", 0))
    return {
        "host_weight_count": count,
        "w_host_bytes": weight_bytes,
        "w_t_host_bytes": weight_t_bytes,
        "pinned_cpu_bytes": pinned_bytes,
    }


def _module_bytes(model: nn.Module) -> dict[str, int]:
    gpu = 0
    cpu = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        nbytes = int(tensor.numel() * tensor.element_size())
        if tensor.device.type == "cuda":
            gpu += nbytes
        elif tensor.device.type == "cpu":
            cpu += nbytes
    host = _host_weight_bytes(model)
    return {
        "gpu_parameter_and_buffer_bytes": gpu,
        "cpu_parameter_and_buffer_bytes": cpu,
        **host,
    }


def _memory_report(model: nn.Module, device: torch.device, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    module = _module_bytes(model)
    expected_saved = int(
        getattr(model, "expected_hbm_saved_bytes", 0)
        or getattr(model, "cpu_resident_base_weight_bytes", 0)
        or getattr(model, "frozen_weight_bytes", 0)
    )
    pinned = int(getattr(model, "pinned_cpu_bytes", 0) or module["pinned_cpu_bytes"])
    current_hbm = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    peak_hbm = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    totals = {
        "hbm_current_bytes": current_hbm,
        "hbm_peak_bytes": peak_hbm,
        "expected_hbm_saved_bytes": expected_saved,
        "pinned_cpu_cost_bytes": pinned,
        **module,
        "rss_bytes": _rss_bytes(),
    }
    if extra:
        totals.update({k: v for k, v in extra.items() if isinstance(v, int)})
    total_cpu = totals["cpu_parameter_and_buffer_bytes"] + totals["pinned_cpu_cost_bytes"]
    total_gpu_basis = totals["gpu_parameter_and_buffer_bytes"] + totals["expected_hbm_saved_bytes"]
    total_memory_basis = max(1, total_cpu + total_gpu_basis + totals["rss_bytes"])
    cpu_memory = {
        "total_bytes": total_cpu,
        "total_percent": _percent(total_cpu, total_memory_basis),
        "rss_bytes": totals["rss_bytes"],
        "rss_percent_of_total": _percent(totals["rss_bytes"], total_memory_basis),
        "pinned_bytes": totals["pinned_cpu_cost_bytes"],
        "pinned_percent_of_cpu_total": _percent(totals["pinned_cpu_cost_bytes"], max(1, total_cpu)),
        "w_host_bytes": totals["w_host_bytes"],
        "w_t_host_bytes": totals["w_t_host_bytes"],
        "unattributed_cpu_bytes": max(0, totals["rss_bytes"] - total_cpu),
        "unattributed_cpu_percent": _percent(max(0, totals["rss_bytes"] - total_cpu), max(1, totals["rss_bytes"])),
        "estimated": True,
    }
    gpu_memory = {
        "total_bytes": total_gpu_basis,
        "total_percent": _percent(total_gpu_basis, total_memory_basis),
        "peak_hbm_bytes": totals["hbm_peak_bytes"],
        "peak_hbm_percent_of_total": _percent(totals["hbm_peak_bytes"], max(1, total_gpu_basis)),
        "current_hbm_bytes": totals["hbm_current_bytes"],
        "resident_parameter_and_buffer_bytes": totals["gpu_parameter_and_buffer_bytes"],
        "expected_hbm_saved_bytes": totals["expected_hbm_saved_bytes"],
        "staged_buffer_bytes": int(totals.get("staged_buffer_bytes", 0)),
        "unattributed_gpu_bytes": max(0, totals["hbm_peak_bytes"] - totals["gpu_parameter_and_buffer_bytes"]),
        "unattributed_gpu_percent": _percent(
            max(0, totals["hbm_peak_bytes"] - totals["gpu_parameter_and_buffer_bytes"]),
            max(1, totals["hbm_peak_bytes"]),
        ),
        "estimated": True,
    }
    return {
        "total_bytes": total_memory_basis,
        "total_percent": 100.0,
        "totals": totals,
        "cpu_memory": cpu_memory,
        "gpu_memory": gpu_memory,
        "percentages": {
            "hbm_resident_of_gpu_basis_pct": _percent(totals["gpu_parameter_and_buffer_bytes"], total_gpu_basis),
            "hbm_saved_of_gpu_basis_pct": _percent(totals["expected_hbm_saved_bytes"], total_gpu_basis),
            "pinned_cpu_of_cpu_total_pct": _percent(totals["pinned_cpu_cost_bytes"], total_cpu),
            "w_host_of_pinned_cpu_pct": _percent(totals["w_host_bytes"], max(1, totals["pinned_cpu_cost_bytes"])),
            "w_t_host_of_pinned_cpu_pct": _percent(totals["w_t_host_bytes"], max(1, totals["pinned_cpu_cost_bytes"])),
        },
    }


def _latency_report(book: TimerBook, setup_seconds: float, warmup_steps: int, steps: int) -> dict[str, Any]:
    avg = {key: value / float(max(1, steps)) for key, value in book.values.items()}
    avg["moe_seconds"] = avg.get("moe_seconds", 0.0) + avg.get("expert_moe_seconds", 0.0)
    for key in (
        "base_forward_seconds",
        "base_dx_seconds",
        "lora_seconds",
        "router_seconds",
        "attention_seconds",
        "route_metadata_seconds",
        "route_pack_seconds",
        "route_scatter_seconds",
        "moe_seconds",
        "mlp_seconds",
    ):
        avg.setdefault(key, 0.0)
    avg["lora_grad_seconds_estimated"] = max(0.0, avg.get("backward", 0.0) - avg["base_dx_seconds"])
    forward = avg.get("forward", 0.0)
    loss = avg.get("loss", 0.0)
    backward = avg.get("backward", 0.0)
    optimizer = avg.get("optimizer", 0.0)
    setup_avg = setup_seconds
    total = setup_avg + forward + loss + backward + optimizer
    steady_state = forward + loss + backward + optimizer
    fallback = max(0.0, total - setup_avg - forward - loss - backward - optimizer)
    breakdown = {
        "setup_host_preparation_seconds": setup_avg,
        "forward_seconds": forward,
        "loss_seconds": loss,
        "backward_seconds": backward,
        "optimizer_seconds": optimizer,
        "fallback_overhead_unattributed_seconds": fallback,
    }
    step_values = book.step_seconds or [total]
    if profiling is not None and hasattr(profiling, "timing_summary"):
        timing_stats = profiling.timing_summary(step_values, warmup_steps=warmup_steps, measured_steps=steps)
    else:
        timing_stats = {
            "warmup_steps": warmup_steps,
            "measured_steps": steps,
            "mean_seconds": sum(step_values) / float(len(step_values)),
            "median_seconds": sorted(step_values)[len(step_values) // 2],
            "p90_seconds": max(step_values),
            "p95_seconds": max(step_values),
            "min_seconds": min(step_values),
            "max_seconds": max(step_values),
            "std_seconds": 0.0,
            "coefficient_of_variation": 0.0,
            "values_seconds": step_values,
        }
    return {
        "measure_steps": steps,
        "measured_steps": steps,
        "total_seconds": total,
        "total_percent": 100.0,
        "setup_seconds": setup_avg,
        "steady_state_step_seconds": steady_state,
        "steady_state_step_percent_of_total": _percent(steady_state, total),
        "total_seconds_per_step_plus_setup": total,
        "breakdown_seconds": breakdown,
        "breakdown_percent": {key.replace("_seconds", "_pct"): _percent(value, total) for key, value in breakdown.items()},
        "component_seconds_per_step": {
            key: value
            for key, value in avg.items()
            if key not in {"forward", "loss", "backward", "optimizer"}
        },
        "component_percent_of_total": {
            key: _percent(value, total)
            for key, value in avg.items()
            if key not in {"forward", "loss", "backward", "optimizer"}
        },
        "notes": {
            "component_timers_may_overlap_parent_forward": True,
            "base_dx_seconds_measured": avg["base_dx_seconds"] > 0.0,
            "lora_grad_seconds_estimated": True,
            "profiling_backend_imported": profiling is not None,
        },
        "timing_stats": timing_stats,
    }


def _run_steps(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    make_batch: Callable[[], tuple[Any, Any]],
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    hook_rules: Mapping[str, str],
    patch_context: Callable[[TimerBook], Iterator[None]] | None = None,
) -> TimerBook:
    def one_step(book: TimerBook | None = None) -> None:
        optimizer.zero_grad(set_to_none=True)
        batch, target = make_batch()
        if book is None:
            prediction = forward_fn(model, batch)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            _sync(device)
            return
        _sync(device)
        step_start = time.perf_counter()
        with _forward_hooks(model, book, hook_rules):
            with book.time("forward"):
                prediction = forward_fn(model, batch)
            with book.time("loss"):
                scalar = loss_fn(prediction, target)
            with book.time("backward"):
                scalar.backward()
            with book.time("optimizer"):
                optimizer.step()
        _sync(device)
        book.step_seconds.append(time.perf_counter() - step_start)

    for _ in range(warmup_steps):
        one_step(None)

    book = TimerBook(device)
    ctx = patch_context(book) if patch_context is not None else nullcontext()
    with _patched_base_dispatch(book):
        with ctx:
            for _ in range(measure_steps):
                one_step(book)
    return book


def profile_mlp(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from examples.asymgemm.mlp_lora_demo import AsymMLP, _lora_parameters

    _clear(device)
    start = time.perf_counter()
    stats = AsymExecutionStats()
    tokens = 64 if device.type == "cuda" else 4
    in_features = 128 if device.type == "cuda" else 16
    hidden = 256 if device.type == "cuda" else 32
    out_features = 128 if device.type == "cuda" else 16
    rank = 8 if device.type == "cuda" else 4
    torch.manual_seed(0)
    w1 = torch.randn(hidden, in_features, dtype=dtype)
    w2 = torch.randn(out_features, hidden, dtype=dtype)
    model = AsymMLP(w1, w2, rank=rank, alpha=16.0, backend=args.backend, stats=stats, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(_lora_parameters(model), lr=1e-2)
    setup_seconds = time.perf_counter() - start

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(tokens, in_features, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tokens, out_features, device=device, dtype=dtype),
        )

    def forward_fn(model_: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model_(x)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction.float(), target.float())

    book = _run_steps(
        model=model,
        optimizer=optimizer,
        make_batch=make_batch,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        hook_rules={"AsymLoRALinear": "lora_seconds"},
    )
    return _final_report("mlp", model, device, dtype, args, setup_seconds, book, {"tokens": tokens, "in_features": in_features, "hidden": hidden, "out_features": out_features, "rank": rank})


def profile_dense(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.tiny_dense_llm import AsymTinyDenseLLM, MICRO_DENSE_LLM_CONFIG, make_inputs, make_tiny_dense_weights

    _clear(device)
    start = time.perf_counter()
    config = MICRO_DENSE_LLM_CONFIG
    weights = make_tiny_dense_weights(config, seed=1, dtype=dtype)
    model = AsymTinyDenseLLM(weights, config=config, target_mode="all", backend=args.backend, device=device, dtype=dtype, lora_seed=2)
    optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-3)
    setup_seconds = time.perf_counter() - start

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = make_inputs(config, seed=int(time.perf_counter_ns() % 1_000_000), device=device, dtype=dtype)
        return inputs.detach().clone().requires_grad_(True), labels

    def forward_fn(model_: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        out = model_(inputs_embeds=inputs, labels=None)
        return out["logits"]

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous().to(device=logits.device)
        return F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))

    book = _run_steps(
        model=model,
        optimizer=optimizer,
        make_batch=make_batch,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        hook_rules={"AsymLoRALinear": "lora_seconds", "TinySelfAttention": "attention_seconds", "TinyMLP": "mlp_seconds"},
    )
    return _final_report("dense_llm", model, device, dtype, args, setup_seconds, book, asdict(config))


def profile_moe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    import asym_gemm.training.tiny_moe as tiny_moe
    from asym_gemm.training.tiny_moe import MICRO_MOE_CONFIG, make_static_routes, make_tiny_moe_pair

    _clear(device)
    start = time.perf_counter()
    config = MICRO_MOE_CONFIG
    model, _, _, _ = make_tiny_moe_pair(config=config, seed=3, device=device, base_dtype=dtype, backend=args.backend, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    static_routes = make_static_routes(config, device, pattern="balanced")
    setup_seconds = time.perf_counter() - start

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(config.logical_tokens, config.hidden_size, device=device, dtype=dtype, requires_grad=True) * 0.5
        target = torch.roll(x.detach().float(), shifts=1, dims=0) * 0.25
        return x, target

    def forward_fn(model_: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = model_(x, static_routing=static_routes, mode="contiguous")
        assert isinstance(y, torch.Tensor)
        return y

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction.float(), target)

    def patch_context(book: TimerBook) -> Iterator[None]:
        return _patched_timers(
            tiny_moe,
            book,
            {
                "build_route_metadata": "route_metadata_seconds",
                "pack_tokens_contiguous": "route_pack_seconds",
                "pack_tokens_masked": "route_pack_seconds",
                "scatter_contiguous": "route_scatter_seconds",
                "scatter_masked": "route_scatter_seconds",
            },
        )

    book = _run_steps(
        model=model,
        optimizer=optimizer,
        make_batch=make_batch,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        hook_rules={"AsymTinyExpert": "expert_moe_seconds", "TinySelfAttention": "attention_seconds", "AsymTinyMoELayer": "moe_layer_seconds"},
        patch_context=patch_context,
    )
    report = _final_report("tiny_moe", model, device, dtype, args, setup_seconds, book, asdict(config))
    report["moe_route_mode_comparison"] = {
        mode: _profile_moe_route_mode(
            model=model,
            config=config,
            static_routes=static_routes,
            mode=mode,
            device=device,
            dtype=dtype,
            patch_context=patch_context,
        )
        for mode in ("contiguous", "masked")
    }
    return report


def _profile_moe_route_mode(
    *,
    model: nn.Module,
    config: Any,
    static_routes: list[Any],
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    patch_context: Callable[[TimerBook], Iterator[None]],
) -> dict[str, Any]:
    import asym_gemm.training.tiny_moe as tiny_moe

    model.zero_grad(set_to_none=True)
    book = TimerBook(device)
    x = torch.randn(config.logical_tokens, config.hidden_size, device=device, dtype=dtype, requires_grad=True) * 0.5
    target = torch.roll(x.detach().float(), shifts=1, dims=0) * 0.25
    metadata = tiny_moe.build_route_metadata(
        static_routes[0][0],
        static_routes[0][1],
        num_experts=config.num_experts,
        mode=mode,
    )
    _sync(device)
    step_start = time.perf_counter()
    with _patched_base_dispatch(book):
        with patch_context(book):
            with book.time("forward"):
                y = model(x, static_routing=static_routes, mode=mode)
                assert isinstance(y, torch.Tensor)
            with book.time("loss"):
                loss = F.mse_loss(y.float(), target)
            with book.time("backward"):
                loss.backward()
    _sync(device)
    book.step_seconds.append(time.perf_counter() - step_start)
    return {
        "mode": mode,
        "latency": _latency_report(book, 0.0, 0, 1),
        "route_metadata": tiny_moe.route_metadata_summary(metadata),
    }


def _final_report(
    workload: str,
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    setup_seconds: float,
    book: TimerBook,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    latency = _latency_report(book, setup_seconds, args.warmup_steps, args.measure_steps)
    memory = _memory_report(model, device)
    stats = _execution_stats(model)
    return {
        "milestone": "M4.5 Profiling",
        "workload": workload,
        "generated_at_utc": _now(),
        "device": str(device),
        "dtype": str(dtype),
        "backend": args.backend,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measure_steps,
        "measure_steps": args.measure_steps,
        "config": dict(config),
        "latency": latency,
        "memory": memory,
        "execution_stats": stats,
        "direct_fetch_forward_used": bool(stats.get("asym_forward_calls", 0) > 0),
        "direct_fetch_dx_used": bool(stats.get("asym_dx_calls", 0) > 0),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for name, report in reports.items():
        latency = report["latency"]
        memory = report["memory"]["totals"]
        rows.append(
            {
                "workload": name,
                "total_seconds_per_step_plus_setup": latency["total_seconds_per_step_plus_setup"],
                "forward_seconds": latency["breakdown_seconds"]["forward_seconds"],
                "backward_seconds": latency["breakdown_seconds"]["backward_seconds"],
                "optimizer_seconds": latency["breakdown_seconds"]["optimizer_seconds"],
                "hbm_peak_bytes": memory["hbm_peak_bytes"],
                "expected_hbm_saved_bytes": memory["expected_hbm_saved_bytes"],
                "pinned_cpu_cost_bytes": memory["pinned_cpu_cost_bytes"],
                "w_host_bytes": memory["w_host_bytes"],
                "w_t_host_bytes": memory["w_t_host_bytes"],
                "direct_fetch_forward_used": report["direct_fetch_forward_used"],
                "direct_fetch_dx_used": report["direct_fetch_dx_used"],
            }
        )
    return {
        "milestone": "M4.5 Profiling Summary",
        "generated_at_utc": _now(),
        "reports": rows,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# M4.5 Profile Summary",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "This summary compares latency, GPU/HBM memory, CPU memory, and pinned CPU cost across MLP, dense LLM, and MoE profiling workloads.",
        "",
        "| Workload | Total s | Forward s | Backward s | Optimizer s | Peak HBM | HBM Saved | Pinned CPU | W Host | W.T Host | Direct Fwd/dX |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["reports"]:
        direct = f"{row['direct_fetch_forward_used']}/{row['direct_fetch_dx_used']}"
        lines.append((
            "| {workload} | {total_seconds_per_step_plus_setup:.6f} | {forward_seconds:.6f} | "
            "{backward_seconds:.6f} | {optimizer_seconds:.6f} | {hbm_peak_bytes} | "
            "{expected_hbm_saved_bytes} | {pinned_cpu_cost_bytes} | {w_host_bytes} | "
            "{w_t_host_bytes} | " + direct + " |"
        ).format(**row))
    lines.append("")
    lines.append("Component timers may overlap their parent forward phase; top-level phase percentages are non-overlapping.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["all", "mlp", "dense", "moe"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--backend", choices=["asym_only", "asym_or_staged", "asym_or_torch", "torch_only"], default="asym_or_staged")
    parser.add_argument("--warmup-steps", "--warmup", dest="warmup_steps", type=int, default=1)
    parser.add_argument("--measure-steps", "--measured-steps", "--measured", dest="measure_steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        raise SystemExit("--warmup-steps must be >= 0 and --measure-steps must be > 0")
    device = _device(args.device)
    dtype = _dtype_for(device)
    if device.type == "cpu" and args.backend == "asym_only":
        raise SystemExit("--backend asym_only requires a CUDA device")

    selected = ["mlp", "dense", "moe"] if args.workload == "all" else [args.workload]
    reports: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name == "mlp":
            report = profile_mlp(args, device, dtype)
            filename = "m4_5_mlp_profile.json"
            key = "mlp"
        elif name == "dense":
            report = profile_dense(args, device, dtype)
            filename = "m4_5_dense_llm_profile.json"
            key = "dense_llm"
        else:
            report = profile_moe(args, device, dtype)
            filename = "m4_5_tiny_moe_profile.json"
            key = "tiny_moe"
        reports[key] = report
        _write_json(args.output_dir / filename, report)

    if args.workload == "all":
        summary = _summary(reports)
        _write_json(args.output_dir / "m4_5_profile_summary.json", summary)
        (args.output_dir / "m4_5_profile_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(json.dumps(_summary(reports), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
