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
from types import SimpleNamespace
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


def _memory_point(device: torch.device) -> dict[str, int]:
    allocated = reserved = peak_allocated = peak_reserved = 0
    if device.type == "cuda":
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "cpu_rss_bytes": _rss_bytes(),
        "gpu_allocated_bytes": allocated,
        "gpu_reserved_bytes": reserved,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
    }


def _memory_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        "cpu_rss_delta_bytes": int(after["cpu_rss_bytes"] - before["cpu_rss_bytes"]),
        "gpu_allocated_delta_bytes": int(after["gpu_allocated_bytes"] - before["gpu_allocated_bytes"]),
        "gpu_reserved_delta_bytes": int(after["gpu_reserved_bytes"] - before["gpu_reserved_bytes"]),
    }


def _setup_memory(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, Any]:
    return {"before": dict(before), "after": dict(after), "delta": _memory_delta(before, after)}


def _zero_setup_memory(device: torch.device) -> dict[str, Any]:
    point = _memory_point(device)
    return _setup_memory(point, point)


def _device(name: str | None) -> torch.device:
    if name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _dtype_for(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


QWEN_MODEL_CHOICES = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_30b_a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3_32b": "Qwen/Qwen3-32B",
}


_QWEN_CONFIG_FALLBACKS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-14B": {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "intermediate_size": 17408,
    },
    "Qwen/Qwen3-30B-A3B": {
        "model_type": "qwen3_moe",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "intermediate_size": 768,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percent(value: float, total: float) -> float:
    return 0.0 if total <= 0.0 else 100.0 * value / total


class TimerBook:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, float] = {}
        self.step_seconds: list[float] = []
        self.stage_samples: dict[str, list[dict[str, Any]]] = {}
        self._depth = 0

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        _sync(self.device)
        top_level = self._depth == 0
        before = _memory_point(self.device)
        if top_level and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        self._depth += 1
        try:
            yield
        finally:
            _sync(self.device)
            elapsed = time.perf_counter() - start
            self._depth -= 1
            after = _memory_point(self.device)
            self.values[name] = self.values.get(name, 0.0) + elapsed
            sample: dict[str, Any] = {
                "seconds": elapsed,
                "memory_before": before,
                "memory_after": after,
                "memory_delta": _memory_delta(before, after),
                "gpu_peak_allocated_scope": "stage" if top_level else "enclosing_stage",
                "top_level_stage": top_level,
            }
            self.stage_samples.setdefault(name, []).append(sample)


class DeepProfile:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.setup_seconds: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []
        self.memory_snapshots: list[dict[str, Any]] = []

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        _sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.setup_seconds[name] = self.setup_seconds.get(name, 0.0) + (time.perf_counter() - start)

    def record_event(self, name: str, payload: Mapping[str, Any]) -> None:
        self.events.append({"event": name, **dict(payload)})

    def snapshot(self, label: str, model: nn.Module | None = None, optimizer: torch.optim.Optimizer | None = None) -> None:
        gpu: dict[str, int] = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "reserved_minus_allocated_bytes": 0,
        }
        if self.device.type == "cuda":
            _sync(self.device)
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
            gpu = {
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "reserved_minus_allocated_bytes": max(0, reserved - allocated),
            }
        model_bytes = _module_bytes(model) if model is not None else {}
        optimizer_bytes = 0
        if optimizer is not None and profiling is not None and hasattr(profiling, "optimizer_state_nbytes"):
            optimizer_bytes = int(profiling.optimizer_state_nbytes(optimizer))
        self.memory_snapshots.append(
            {
                "label": label,
                "timestamp_utc": _now(),
                "rss_bytes": _rss_bytes(),
                "gpu": gpu,
                "model": model_bytes,
                "optimizer_state_bytes": optimizer_bytes,
            }
        )

    def report(self, *, execution_stats: Mapping[str, Any]) -> dict[str, Any]:
        host_init_events = [event for event in self.events if event.get("event") == "host_weight_init"]
        transpose_events = [event for event in self.events if event.get("event") == "host_weight_transpose"]

        def sum_float(events: list[dict[str, Any]], key: str) -> float:
            return float(sum(float(event.get(key, 0.0) or 0.0) for event in events))

        def sum_int(events: list[dict[str, Any]], key: str) -> int:
            return int(sum(int(event.get(key, 0) or 0) for event in events))

        setup_total = float(sum(self.setup_seconds.values()))
        setup_breakdown = {
            key: {
                "seconds": value,
                "percent_of_profiled_setup": _percent(value, setup_total),
            }
            for key, value in sorted(self.setup_seconds.items())
        }
        return {
            "enabled": True,
            "setup_breakdown": {
                "total_profiled_setup_seconds": setup_total,
                "sections": setup_breakdown,
                "host_weight_init_seconds": sum_float(host_init_events, "total_seconds"),
                "host_weight_cpu_copy_seconds": sum_float(host_init_events, "cpu_copy_seconds"),
                "host_weight_clone_seconds": sum_float(host_init_events, "clone_seconds"),
                "host_weight_pin_seconds": sum_float(host_init_events, "pin_seconds"),
                "host_weight_count": len(host_init_events),
                "host_weight_bytes": sum_int(host_init_events, "nbytes"),
            },
            "transpose_materialization": {
                "count": len(transpose_events),
                "bytes": sum_int(transpose_events, "nbytes"),
                "total_seconds": sum_float(transpose_events, "total_seconds"),
                "transpose_seconds": sum_float(transpose_events, "transpose_seconds"),
                "pin_seconds": sum_float(transpose_events, "pin_seconds"),
            },
            "copy_accounting": {
                "staged_calls": int(execution_stats.get("staged_calls", 0) or 0),
                "torch_calls": int(execution_stats.get("torch_calls", 0) or 0),
                "explicit_staged_weight_copy_bytes_observed": 0,
                "explicit_gpu_cpu_torch_copy_bytes_observed": 0,
                "notes": "Direct host fetch happens inside AsymGEMM kernels and is not counted as a Python tensor copy. Staged/Torch fallback copy bytes are zero for asym_only reports unless fallback calls are nonzero.",
            },
            "memory_timeline": self.memory_snapshots,
            "host_weight_events": self.events,
        }


@contextmanager
def _host_weight_profile(deep: DeepProfile | None) -> Iterator[None]:
    if deep is None:
        yield
        return
    host_weight = importlib.import_module("asym_gemm.training.host_weight")
    previous = host_weight.set_profile_recorder(deep.record_event)
    try:
        yield
    finally:
        host_weight.set_profile_recorder(previous)


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


def _avg_int(values: list[int]) -> int:
    return int(sum(values) / float(len(values))) if values else 0


def _stage_profile(
    *,
    name: str,
    samples: list[dict[str, Any]],
    total_seconds: float,
    steady_seconds: float,
) -> dict[str, Any]:
    seconds_values = [float(sample["seconds"]) for sample in samples]
    avg_seconds = sum(seconds_values) / float(len(seconds_values)) if seconds_values else 0.0
    before = [sample["memory_before"] for sample in samples]
    after = [sample["memory_after"] for sample in samples]
    delta = [sample["memory_delta"] for sample in samples]
    return {
        "stage": name,
        "sample_count": len(samples),
        "average_seconds": avg_seconds,
        "total_recorded_seconds": sum(seconds_values),
        "percent_of_total": _percent(avg_seconds, total_seconds),
        "percent_of_steady_state_step": _percent(avg_seconds, steady_seconds),
        "average_cpu_rss_before_bytes": _avg_int([int(item["cpu_rss_bytes"]) for item in before]),
        "average_cpu_rss_after_bytes": _avg_int([int(item["cpu_rss_bytes"]) for item in after]),
        "average_cpu_rss_delta_bytes": _avg_int([int(item["cpu_rss_delta_bytes"]) for item in delta]),
        "max_cpu_rss_after_bytes": max([int(item["cpu_rss_bytes"]) for item in after], default=0),
        "average_gpu_allocated_before_bytes": _avg_int([int(item["gpu_allocated_bytes"]) for item in before]),
        "average_gpu_allocated_after_bytes": _avg_int([int(item["gpu_allocated_bytes"]) for item in after]),
        "average_gpu_allocated_delta_bytes": _avg_int([int(item["gpu_allocated_delta_bytes"]) for item in delta]),
        "average_gpu_reserved_before_bytes": _avg_int([int(item["gpu_reserved_bytes"]) for item in before]),
        "average_gpu_reserved_after_bytes": _avg_int([int(item["gpu_reserved_bytes"]) for item in after]),
        "average_gpu_reserved_delta_bytes": _avg_int([int(item["gpu_reserved_delta_bytes"]) for item in delta]),
        "max_gpu_peak_allocated_bytes": max([int(item["memory_after"]["gpu_peak_allocated_bytes"]) for item in samples], default=0),
        "max_gpu_peak_reserved_bytes": max([int(item["memory_after"]["gpu_peak_reserved_bytes"]) for item in samples], default=0),
        "peak_scope": "stage" if any(bool(sample.get("top_level_stage")) for sample in samples) else "enclosing_stage",
        "samples": samples,
    }


def _setup_stage_profile(
    *,
    setup_seconds: float,
    setup_memory: Mapping[str, Any],
    total_seconds: float,
) -> dict[str, Any]:
    before = setup_memory["before"]
    after = setup_memory["after"]
    delta = setup_memory["delta"]
    return {
        "stage": "setup_host_preparation",
        "sample_count": 1,
        "average_seconds": setup_seconds,
        "total_recorded_seconds": setup_seconds,
        "percent_of_total": _percent(setup_seconds, total_seconds),
        "percent_of_steady_state_step": 0.0,
        "average_cpu_rss_before_bytes": int(before["cpu_rss_bytes"]),
        "average_cpu_rss_after_bytes": int(after["cpu_rss_bytes"]),
        "average_cpu_rss_delta_bytes": int(delta["cpu_rss_delta_bytes"]),
        "max_cpu_rss_after_bytes": int(after["cpu_rss_bytes"]),
        "average_gpu_allocated_before_bytes": int(before["gpu_allocated_bytes"]),
        "average_gpu_allocated_after_bytes": int(after["gpu_allocated_bytes"]),
        "average_gpu_allocated_delta_bytes": int(delta["gpu_allocated_delta_bytes"]),
        "average_gpu_reserved_before_bytes": int(before["gpu_reserved_bytes"]),
        "average_gpu_reserved_after_bytes": int(after["gpu_reserved_bytes"]),
        "average_gpu_reserved_delta_bytes": int(delta["gpu_reserved_delta_bytes"]),
        "max_gpu_peak_allocated_bytes": int(after["gpu_peak_allocated_bytes"]),
        "max_gpu_peak_reserved_bytes": int(after["gpu_peak_reserved_bytes"]),
        "peak_scope": "setup",
        "samples": [
            {
                "seconds": setup_seconds,
                "memory_before": dict(before),
                "memory_after": dict(after),
                "memory_delta": dict(delta),
                "gpu_peak_allocated_scope": "setup",
                "top_level_stage": True,
            }
        ],
    }


def _stage_memory_report(
    *,
    book: TimerBook,
    setup_seconds: float,
    setup_memory: Mapping[str, Any],
    total_seconds: float,
    steady_seconds: float,
) -> dict[str, Any]:
    profiles = {
        "setup_host_preparation": _setup_stage_profile(
            setup_seconds=setup_seconds,
            setup_memory=setup_memory,
            total_seconds=total_seconds,
        )
    }
    for name in sorted(book.stage_samples):
        profiles[name] = _stage_profile(
            name=name,
            samples=book.stage_samples[name],
            total_seconds=total_seconds,
            steady_seconds=steady_seconds,
        )
    return {
        "description": "Per-stage timing plus CPU RSS and GPU HBM before/after/delta/peak. Nested component stages can overlap parent stages.",
        "top_level_stages": [
            "setup_host_preparation",
            "forward",
            "loss",
            "backward",
            "optimizer",
        ],
        "component_stages": sorted(name for name in profiles if name not in {"setup_host_preparation", "forward", "loss", "backward", "optimizer"}),
        "stages": profiles,
    }


def _latency_report(
    book: TimerBook,
    setup_seconds: float,
    setup_memory: Mapping[str, Any],
    warmup_steps: int,
    steps: int,
) -> dict[str, Any]:
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
        "per_stage": _stage_memory_report(
            book=book,
            setup_seconds=setup_avg,
            setup_memory=setup_memory,
            total_seconds=total,
            steady_seconds=steady_state,
        ),
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
        "bubble_candidates": _bubble_candidates(avg, total),
    }


def _bubble_candidates(avg: Mapping[str, float], total: float) -> list[dict[str, Any]]:
    """Python-visible idle/underutilization candidates; hardware bubbles need Nsight counters."""

    candidates = [
        (
            "base_forward_frozen_dispatch",
            float(avg.get("base_forward_seconds", 0.0)),
            "CPU-resident base-weight fetch plus GEMM in forward. A high share means PCIe/NVLink fetch latency or small-M GEMM occupancy may be limiting GPU utilization.",
        ),
        (
            "base_dx_frozen_dispatch",
            float(avg.get("base_dx_seconds", 0.0)),
            "CPU-resident original weight fetch for dX. This now uses transpose_b=True and should not include W.T host materialization.",
        ),
        (
            "route_metadata",
            float(avg.get("route_metadata_seconds", 0.0)),
            "MoE route metadata construction. CPU/PyTorch launch gaps here can leave the GPU idle before expert kernels.",
        ),
        (
            "route_pack",
            float(avg.get("route_pack_seconds", 0.0)),
            "MoE token gather/pack. If large, expert GEMMs wait for scattered token layout preparation.",
        ),
        (
            "route_scatter",
            float(avg.get("route_scatter_seconds", 0.0)),
            "MoE output scatter/index_add. This is a common non-GEMM tail after expert compute.",
        ),
        (
            "lora_path",
            float(avg.get("lora_seconds", 0.0)),
            "Low-rank trainable path. Many small GEMMs can underfill the GPU relative to large base GEMMs.",
        ),
        (
            "optimizer",
            float(avg.get("optimizer", 0.0)),
            "Optimizer update. For LoRA-only SFT this is usually small, but Python optimizer overhead can still serialize steps.",
        ),
    ]
    return [
        {
            "name": name,
            "seconds_per_step": seconds,
            "percent_of_total_plus_setup": _percent(seconds, total),
            "why_it_can_create_gpu_bubbles": note,
            "requires_nsight_to_confirm": name.startswith("base_"),
        }
        for name, seconds, note in sorted(candidates, key=lambda item: item[1], reverse=True)
        if seconds > 0.0
    ]


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
    deep: DeepProfile | None = None,
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

    if deep is not None:
        deep.snapshot("before_warmup", model=model, optimizer=optimizer)
    with _host_weight_profile(deep):
        for _ in range(warmup_steps):
            one_step(None)
    if deep is not None:
        deep.snapshot("after_warmup", model=model, optimizer=optimizer)

    book = TimerBook(device)
    ctx = patch_context(book) if patch_context is not None else nullcontext()
    if deep is not None:
        deep.snapshot("before_measured_steps", model=model, optimizer=optimizer)
    with _host_weight_profile(deep):
        with _patched_base_dispatch(book):
            with ctx:
                for _ in range(measure_steps):
                    one_step(book)
    if deep is not None:
        deep.snapshot("after_measured_steps", model=model, optimizer=optimizer)
    return book


def _mlp_architecture_estimate(dtype: torch.dtype) -> dict[str, Any]:
    in_features = 8192
    hidden = 196608
    out_features = 8192
    rank = 128
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    base_elements = hidden * in_features + out_features * hidden
    lora_elements = rank * (in_features + hidden) + rank * (hidden + out_features)
    total_elements = base_elements + lora_elements
    return {
        "name": ">=3B two-layer LoRA MLP reference",
        "profiled_with_scaled_micro_tensors": True,
        "in_features": in_features,
        "hidden": hidden,
        "out_features": out_features,
        "lora_rank": rank,
        "total_model_elements": total_elements,
        "base_weight_elements": base_elements,
        "trainable_lora_elements": lora_elements,
        "expected_hbm_saved_bytes": base_elements * dtype_bytes,
        "expected_pinned_cpu_bytes_without_w_t_copy": base_elements * dtype_bytes,
        "meets_3b_parameter_requirement": total_elements >= 3_000_000_000,
    }


def _dense_toy_architecture_estimate(dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.dense import TinyDenseLLMConfig, estimate_tiny_dense_llm_parameters

    config = TinyDenseLLMConfig(
        vocab_size=32768,
        hidden_size=3072,
        num_layers=24,
        num_heads=24,
        seq_len=128,
        batch_size=1,
        intermediate_size=8192,
        lora_rank=128,
        lora_alpha=256.0,
    )
    estimate = estimate_tiny_dense_llm_parameters(config, target_mode="all", dtype=dtype)
    return {
        "name": ">=3B dense transformer reference",
        "profiled_with_scaled_micro_tensors": True,
        "config": asdict(config),
        "estimate": estimate,
        "meets_3b_parameter_requirement": int(estimate["total_model_elements"]) >= 3_000_000_000,
    }


def _moe_toy_architecture_estimate(dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.moe import TinyMoEConfig, estimate_tiny_moe_parameters

    config = TinyMoEConfig(
        num_layers=24,
        num_experts=16,
        top_k=2,
        hidden_size=2048,
        intermediate_size=4096,
        logical_tokens=128,
        lora_rank=128,
        lora_alpha=256.0,
        residual_scale=0.25,
        num_shared_experts=1,
        vocab_size=32768,
        num_heads=16,
        batch_size=1,
        seq_len=128,
    )
    estimate = estimate_tiny_moe_parameters(config, dtype=dtype)
    return {
        "name": ">=3B sparse MoE transformer reference",
        "profiled_with_scaled_micro_tensors": True,
        "config": asdict(config),
        "estimate": estimate,
        "meets_3b_parameter_requirement": int(estimate["total_model_elements"]) >= 3_000_000_000,
    }


def profile_mlp(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.frozen_linear import AsymExecutionStats
    from asym_gemm.training.mlp import AsymMLP, lora_parameters

    _clear(device)
    setup_before = _memory_point(device)
    deep = DeepProfile(device)
    deep.snapshot("before_setup")
    start = time.perf_counter()
    stats = AsymExecutionStats()
    tokens = 64 if device.type == "cuda" else 4
    in_features = 128 if device.type == "cuda" else 16
    hidden = 256 if device.type == "cuda" else 32
    out_features = 128 if device.type == "cuda" else 16
    rank = 8 if device.type == "cuda" else 4
    with deep.time("setup.seed_and_cpu_weight_tensors"):
        torch.manual_seed(0)
        w1 = torch.randn(hidden, in_features, dtype=dtype)
        w2 = torch.randn(out_features, hidden, dtype=dtype)
    with _host_weight_profile(deep):
        with deep.time("setup.model_and_host_weights"):
            model = AsymMLP(w1, w2, rank=rank, alpha=16.0, backend=args.backend, stats=stats, device=device, dtype=dtype)
    with deep.time("setup.optimizer"):
        optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    _sync(device)
    setup_seconds = time.perf_counter() - start
    setup_memory = _setup_memory(setup_before, _memory_point(device))
    deep.snapshot("after_setup", model=model, optimizer=optimizer)

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
        deep=deep,
    )
    return _final_report(
        "mlp",
        model,
        device,
        dtype,
        args,
        setup_seconds,
        setup_memory,
        book,
        {
            "tokens": tokens,
            "in_features": in_features,
            "hidden": hidden,
            "out_features": out_features,
            "rank": rank,
            "full_scale_reference": _mlp_architecture_estimate(dtype),
        },
        deep=deep,
    )


def profile_dense(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.dense import AsymTinyDenseLLM, MICRO_DENSE_LLM_CONFIG, make_inputs, make_tiny_dense_weights

    _clear(device)
    setup_before = _memory_point(device)
    deep = DeepProfile(device)
    deep.snapshot("before_setup")
    start = time.perf_counter()
    config = MICRO_DENSE_LLM_CONFIG
    with deep.time("setup.cpu_weight_tensors"):
        weights = make_tiny_dense_weights(config, seed=1, dtype=dtype)
    with _host_weight_profile(deep):
        with deep.time("setup.model_and_host_weights"):
            model = AsymTinyDenseLLM(weights, config=config, target_mode="all", backend=args.backend, device=device, dtype=dtype, lora_seed=2)
    with deep.time("setup.optimizer"):
        optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-3)
    _sync(device)
    setup_seconds = time.perf_counter() - start
    setup_memory = _setup_memory(setup_before, _memory_point(device))
    deep.snapshot("after_setup", model=model, optimizer=optimizer)

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
        deep=deep,
    )
    return _final_report(
        "dense_llm",
        model,
        device,
        dtype,
        args,
        setup_seconds,
        setup_memory,
        book,
        {**asdict(config), "full_scale_reference": _dense_toy_architecture_estimate(dtype)},
        deep=deep,
    )


def profile_moe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    import asym_gemm.training.moe as tiny_moe
    from asym_gemm.training.moe import MICRO_MOE_CONFIG, make_static_routes, make_tiny_moe_pair

    _clear(device)
    setup_before = _memory_point(device)
    deep = DeepProfile(device)
    deep.snapshot("before_setup")
    start = time.perf_counter()
    config = MICRO_MOE_CONFIG
    with _host_weight_profile(deep):
        with deep.time("setup.model_pair_and_host_weights"):
            model, _, _, _ = make_tiny_moe_pair(config=config, seed=3, device=device, base_dtype=dtype, backend=args.backend, pin_memory=device.type == "cuda")
    with deep.time("setup.optimizer"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    with deep.time("setup.static_routes"):
        static_routes = make_static_routes(config, device, pattern="balanced")
    _sync(device)
    setup_seconds = time.perf_counter() - start
    setup_memory = _setup_memory(setup_before, _memory_point(device))
    deep.snapshot("after_setup", model=model, optimizer=optimizer)

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
        deep=deep,
    )
    report = _final_report(
        "tiny_moe",
        model,
        device,
        dtype,
        args,
        setup_seconds,
        setup_memory,
        book,
        {**asdict(config), "full_scale_reference": _moe_toy_architecture_estimate(dtype)},
        deep=deep,
    )
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


def _load_hf_config(model_id: str) -> Any:
    fallback = _QWEN_CONFIG_FALLBACKS.get(model_id)
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        if fallback is not None:
            return SimpleNamespace(**fallback)
        raise RuntimeError("transformers is required for Qwen real-config profiling") from exc
    try:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=False, local_files_only=True)
    except Exception:
        if fallback is not None:
            return SimpleNamespace(**fallback)
        raise


def _qwen_dense_profile_config(hf_config: Any, args: argparse.Namespace) -> Any:
    from asym_gemm.training.dense import TinyDenseLLMConfig

    seq_len = int(args.real_seq_len)
    batch_size = int(args.real_batch_size)
    profile_layers = max(1, min(int(args.real_profile_layers), int(hf_config.num_hidden_layers)))
    return TinyDenseLLMConfig(
        vocab_size=min(int(getattr(hf_config, "vocab_size", 151936)), int(args.real_vocab_rows)),
        hidden_size=int(hf_config.hidden_size),
        num_layers=profile_layers,
        num_heads=int(hf_config.num_attention_heads),
        seq_len=seq_len,
        batch_size=batch_size,
        intermediate_size=int(hf_config.intermediate_size),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
    )


def _qwen_moe_profile_config(hf_config: Any, args: argparse.Namespace) -> Any:
    from asym_gemm.training.moe import TinyMoEConfig

    seq_len = int(args.real_seq_len)
    batch_size = int(args.real_batch_size)
    tokens = int(args.real_tokens or (seq_len * batch_size))
    profile_layers = max(1, min(int(args.real_profile_layers), int(hf_config.num_hidden_layers)))
    return TinyMoEConfig(
        num_layers=profile_layers,
        num_experts=int(hf_config.num_experts),
        top_k=int(hf_config.num_experts_per_tok),
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(getattr(hf_config, "moe_intermediate_size", hf_config.intermediate_size)),
        logical_tokens=tokens,
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
        residual_scale=0.25,
        num_shared_experts=int(getattr(hf_config, "num_shared_experts", 0) or 0),
        vocab_size=min(int(getattr(hf_config, "vocab_size", 151936)), int(args.real_vocab_rows)),
        num_heads=int(hf_config.num_attention_heads),
        batch_size=batch_size,
        seq_len=seq_len,
    )


def _qwen_full_dense_estimate(hf_config: Any, args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.dense import TinyDenseLLMConfig, estimate_tiny_dense_llm_parameters

    config = TinyDenseLLMConfig(
        vocab_size=int(getattr(hf_config, "vocab_size", 151936)),
        hidden_size=int(hf_config.hidden_size),
        num_layers=int(hf_config.num_hidden_layers),
        num_heads=int(hf_config.num_attention_heads),
        seq_len=int(args.real_seq_len),
        batch_size=int(args.real_batch_size),
        intermediate_size=int(hf_config.intermediate_size),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
    )
    estimate = estimate_tiny_dense_llm_parameters(config, target_mode="all", dtype=dtype)
    return {"config": asdict(config), "estimate": estimate}


def _qwen_full_moe_estimate(hf_config: Any, args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    from asym_gemm.training.moe import TinyMoEConfig, estimate_tiny_moe_parameters

    config = TinyMoEConfig(
        num_layers=int(hf_config.num_hidden_layers),
        num_experts=int(hf_config.num_experts),
        top_k=int(hf_config.num_experts_per_tok),
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(getattr(hf_config, "moe_intermediate_size", hf_config.intermediate_size)),
        logical_tokens=int(args.real_tokens or (int(args.real_seq_len) * int(args.real_batch_size))),
        lora_rank=int(args.real_lora_rank),
        lora_alpha=float(args.real_lora_alpha),
        residual_scale=0.25,
        num_shared_experts=int(getattr(hf_config, "num_shared_experts", 0) or 0),
        vocab_size=int(getattr(hf_config, "vocab_size", 151936)),
        num_heads=int(hf_config.num_attention_heads),
        batch_size=int(args.real_batch_size),
        seq_len=int(args.real_seq_len),
    )
    estimate = estimate_tiny_moe_parameters(config, dtype=dtype)
    return {"config": asdict(config), "estimate": estimate}


def profile_qwen_dense(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_key: str = "qwen3_8b",
) -> dict[str, Any]:
    from asym_gemm.training.dense import AsymTinyDenseLLM, make_inputs, make_tiny_dense_weights

    model_id = QWEN_MODEL_CHOICES[model_key]
    _clear(device)
    setup_before = _memory_point(device)
    deep = DeepProfile(device)
    deep.snapshot("before_setup")
    start = time.perf_counter()
    with deep.time("setup.hf_config"):
        hf_config = _load_hf_config(model_id)
    profile_config = _qwen_dense_profile_config(hf_config, args)
    with deep.time("setup.config_matched_cpu_weight_tensors"):
        weights = make_tiny_dense_weights(profile_config, seed=11, dtype=dtype)
    with _host_weight_profile(deep):
        with deep.time("setup.model_and_host_weights"):
            model = AsymTinyDenseLLM(weights, config=profile_config, target_mode="all", backend=args.backend, device=device, dtype=dtype, lora_seed=12)
    with deep.time("setup.optimizer"):
        optimizer = torch.optim.AdamW(model.lora_parameters(), lr=3e-4)
    _sync(device)
    setup_seconds = time.perf_counter() - start
    setup_memory = _setup_memory(setup_before, _memory_point(device))
    deep.snapshot("after_setup", model=model, optimizer=optimizer)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = make_inputs(profile_config, seed=int(time.perf_counter_ns() % 1_000_000), device=device, dtype=dtype)
        return inputs.detach().clone().requires_grad_(True), labels

    def forward_fn(model_: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        out = model_(inputs_embeds=inputs, labels=None)
        return out["logits"]

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous().to(device=logits.device)
        return F.cross_entropy(shift_logits.view(-1, profile_config.vocab_size), shift_labels.view(-1))

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
        deep=deep,
    )
    config_report = {
        **asdict(profile_config),
        "hf_model_id": model_id,
        "hf_model_type": str(getattr(hf_config, "model_type", "")),
        "hf_num_hidden_layers": int(hf_config.num_hidden_layers),
        "profiled_layers": int(profile_config.num_layers),
        "weight_source": "random_config_matched",
        "full_model": _qwen_full_dense_estimate(hf_config, args, dtype),
    }
    return _final_report(model_key, model, device, dtype, args, setup_seconds, setup_memory, book, config_report, deep=deep)


def profile_qwen_moe(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_key: str = "qwen3_30b_a3b",
) -> dict[str, Any]:
    import asym_gemm.training.moe as tiny_moe
    from asym_gemm.training.moe import make_static_routes, make_tiny_moe_pair

    model_id = QWEN_MODEL_CHOICES[model_key]
    _clear(device)
    setup_before = _memory_point(device)
    deep = DeepProfile(device)
    deep.snapshot("before_setup")
    start = time.perf_counter()
    with deep.time("setup.hf_config"):
        hf_config = _load_hf_config(model_id)
    profile_config = _qwen_moe_profile_config(hf_config, args)
    with _host_weight_profile(deep):
        with deep.time("setup.model_pair_and_host_weights"):
            model, _, _, _ = make_tiny_moe_pair(
                config=profile_config,
                seed=13,
                device=device,
                base_dtype=dtype,
                backend=args.backend,
                pin_memory=device.type == "cuda",
            )
    with deep.time("setup.optimizer"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    with deep.time("setup.static_routes"):
        static_routes = make_static_routes(profile_config, device, pattern="balanced")
    _sync(device)
    setup_seconds = time.perf_counter() - start
    setup_memory = _setup_memory(setup_before, _memory_point(device))
    deep.snapshot("after_setup", model=model, optimizer=optimizer)

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(profile_config.logical_tokens, profile_config.hidden_size, device=device, dtype=dtype, requires_grad=True) * 0.5
        target = torch.roll(x.detach().float(), shifts=1, dims=0) * 0.25
        return x, target

    def forward_fn(model_: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = model_(x, static_routing=static_routes, mode=args.moe_mode)
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
        deep=deep,
    )
    report = _final_report(
        model_key,
        model,
        device,
        dtype,
        args,
        setup_seconds,
        setup_memory,
        book,
        {
            **asdict(profile_config),
            "hf_model_id": model_id,
            "hf_model_type": str(getattr(hf_config, "model_type", "")),
            "hf_num_hidden_layers": int(hf_config.num_hidden_layers),
            "profiled_layers": int(profile_config.num_layers),
            "weight_source": "random_config_matched",
            "full_model": _qwen_full_moe_estimate(hf_config, args, dtype),
        },
        deep=deep,
    )
    report["moe_route_mode_comparison"] = {
        mode: _profile_moe_route_mode(
            model=model,
            config=profile_config,
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
    import asym_gemm.training.moe as tiny_moe

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
        "latency": _latency_report(book, 0.0, _zero_setup_memory(device), 0, 1),
        "route_metadata": tiny_moe.route_metadata_summary(metadata),
    }


def _final_report(
    workload: str,
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    setup_seconds: float,
    setup_memory: Mapping[str, Any],
    book: TimerBook,
    config: Mapping[str, Any],
    *,
    deep: DeepProfile | None = None,
) -> dict[str, Any]:
    latency = _latency_report(book, setup_seconds, setup_memory, args.warmup_steps, args.measure_steps)
    memory = _memory_report(model, device)
    stats = _execution_stats(model)
    if deep is not None:
        deep.snapshot("final_report", model=model)
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
        "deep_profile": {} if deep is None else deep.report(execution_stats=stats),
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
        deep = report.get("deep_profile", {})
        deep_setup = deep.get("setup_breakdown", {}) if isinstance(deep, Mapping) else {}
        deep_transpose = deep.get("transpose_materialization", {}) if isinstance(deep, Mapping) else {}
        timing = latency.get("timing_stats", {})
        rows.append(
            {
                "workload": name,
                "total_seconds_per_step_plus_setup": latency["total_seconds_per_step_plus_setup"],
                "steady_state_step_seconds": latency["steady_state_step_seconds"],
                "mean_wall_step_seconds": timing.get("mean_seconds", 0.0),
                "p95_wall_step_seconds": timing.get("p95_seconds", 0.0),
                "setup_seconds": latency["setup_seconds"],
                "forward_seconds": latency["breakdown_seconds"]["forward_seconds"],
                "backward_seconds": latency["breakdown_seconds"]["backward_seconds"],
                "optimizer_seconds": latency["breakdown_seconds"]["optimizer_seconds"],
                "host_weight_init_seconds": deep_setup.get("host_weight_init_seconds", 0.0),
                "host_weight_pin_seconds": deep_setup.get("host_weight_pin_seconds", 0.0),
                "transpose_materialization_seconds": deep_transpose.get("total_seconds", 0.0),
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
        "| Workload | Total+Setup s | Steady s | Mean s | p95 s | Setup s | Forward s | Backward s | Optimizer s | Host Init s | Host Pin s | W.T Mat s | Peak HBM | HBM Saved | Pinned CPU | W Host | W.T Host | Direct Fwd/dX |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["reports"]:
        direct = f"{row['direct_fetch_forward_used']}/{row['direct_fetch_dx_used']}"
        lines.append((
            "| {workload} | {total_seconds_per_step_plus_setup:.6f} | {steady_state_step_seconds:.6f} | "
            "{mean_wall_step_seconds:.6f} | {p95_wall_step_seconds:.6f} | {setup_seconds:.6f} | "
            "{forward_seconds:.6f} | {backward_seconds:.6f} | {optimizer_seconds:.6f} | "
            "{host_weight_init_seconds:.6f} | {host_weight_pin_seconds:.6f} | {transpose_materialization_seconds:.6f} | "
            "{hbm_peak_bytes} | {expected_hbm_saved_bytes} | {pinned_cpu_cost_bytes} | {w_host_bytes} | "
            "{w_t_host_bytes} | " + direct + " |"
        ).format(**row))
    lines.append("")
    lines.append("Component timers may overlap their parent forward phase; top-level phase percentages are non-overlapping.")
    lines.append("Deep profile fields are Python-observed setup and host-weight costs; direct host fetch traffic inside kernels requires Nsight/NCU for hardware counters.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=["all", "toy", "mlp", "dense", "moe", "qwen", "qwen3_8b", "qwen3_14b", "qwen3_30b_a3b", "qwen3_32b"],
        default="all",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backend", choices=["asym_only", "asym_or_staged", "asym_or_torch", "torch_only"], default="asym_only")
    parser.add_argument("--warmup-steps", "--warmup", dest="warmup_steps", type=int, default=10)
    parser.add_argument("--measure-steps", "--measured-steps", "--measured", dest="measure_steps", type=int, default=50)
    parser.add_argument("--profile-layers", dest="real_profile_layers", metavar="N", type=int, default=1)
    parser.add_argument("--batch-size", dest="real_batch_size", metavar="N", type=int, default=1)
    parser.add_argument("--seq-len", dest="real_seq_len", metavar="N", type=int, default=64)
    parser.add_argument("--tokens", dest="real_tokens", metavar="N", type=int, default=0)
    parser.add_argument("--lora-rank", dest="real_lora_rank", metavar="N", type=int, default=64)
    parser.add_argument("--lora-alpha", dest="real_lora_alpha", metavar="FLOAT", type=float, default=128.0)
    parser.add_argument("--vocab-rows", dest="real_vocab_rows", metavar="N", type=int, default=4096)
    parser.add_argument("--moe-mode", choices=["contiguous", "masked"], default="contiguous")
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

    if args.real_profile_layers <= 0:
        raise SystemExit("--profile-layers must be > 0")
    if args.real_batch_size <= 0 or args.real_seq_len <= 1:
        raise SystemExit("--batch-size must be > 0 and --seq-len must be > 1")
    if args.real_lora_rank <= 0 or args.real_lora_alpha <= 0:
        raise SystemExit("--lora-rank and --lora-alpha must be > 0")

    if args.workload == "all":
        selected = ["mlp", "dense", "moe", "qwen3_8b", "qwen3_14b", "qwen3_30b_a3b", "qwen3_32b"]
    elif args.workload == "toy":
        selected = ["mlp", "dense", "moe"]
    elif args.workload == "qwen":
        selected = ["qwen3_8b", "qwen3_14b", "qwen3_30b_a3b", "qwen3_32b"]
    else:
        selected = [args.workload]
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
        elif name == "moe":
            report = profile_moe(args, device, dtype)
            filename = "m4_5_tiny_moe_profile.json"
            key = "tiny_moe"
        elif name in {"qwen3_8b", "qwen3_14b", "qwen3_32b"}:
            report = profile_qwen_dense(args, device, dtype, model_key=name)
            filename = f"m4_5_{name}_profile.json"
            key = name
        elif name == "qwen3_30b_a3b":
            report = profile_qwen_moe(args, device, dtype, model_key=name)
            filename = "m4_5_qwen3_30b_a3b_profile.json"
            key = "qwen3_30b_a3b"
        else:
            raise AssertionError(f"unhandled workload {name!r}")
        reports[key] = report
        _write_json(args.output_dir / filename, report)

    if len(reports) > 1:
        summary = _summary(reports)
        _write_json(args.output_dir / "m4_5_profile_summary.json", summary)
        (args.output_dir / "m4_5_profile_summary.md").write_text(_markdown(summary), encoding="utf-8")

    print(json.dumps(_summary(reports), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
