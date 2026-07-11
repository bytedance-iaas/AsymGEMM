from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import torch


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def process_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def process_peak_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensor_tree_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_nbytes(value)
    if isinstance(value, Mapping):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    return tensor_tree_nbytes(optimizer.state)


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "reserved_minus_allocated_bytes": 0,
        }
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "reserved_minus_allocated_bytes": max(0, reserved - allocated),
    }


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "process_rss_bytes": process_rss_bytes(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "gpu": cuda_memory_snapshot(device),
    }


def _percent(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return float(value) * 100.0 / float(total)


def make_breakdown(
    components: Iterable[Mapping[str, Any]],
    *,
    total: float | int | None = None,
    unit: str,
    unattributed_name: str = "unattributed",
) -> dict[str, Any]:
    """Return component accounting with percentages and an explicit gap bucket."""

    normalized: list[dict[str, Any]] = []
    component_sum = 0.0
    for component in components:
        value = float(component.get("value", 0.0))
        component_sum += value
        normalized.append(
            {
                "name": str(component.get("name", "unnamed")),
                "value": value,
                "unit": unit,
                "estimated": bool(component.get("estimated", False)),
                "notes": str(component.get("notes", "")),
            }
        )

    resolved_total = float(component_sum if total is None else total)
    unattributed = max(0.0, resolved_total - component_sum)
    if unattributed > max(resolved_total, 1.0) * 1e-9:
        normalized.append(
            {
                "name": unattributed_name,
                "value": unattributed,
                "unit": unit,
                "estimated": True,
                "notes": "total minus attributed components",
            }
        )
    over_attributed = max(0.0, component_sum - resolved_total)
    denominator = resolved_total if resolved_total > 0.0 else component_sum
    for component in normalized:
        component["percent_of_total"] = _percent(float(component["value"]), denominator)

    accounted_sum = sum(float(component["value"]) for component in normalized)
    return {
        "total": resolved_total,
        "unit": unit,
        "components": normalized,
        "component_sum": accounted_sum,
        "attributed_without_unattributed": component_sum,
        "unattributed": unattributed,
        "unattributed_percent": _percent(unattributed, denominator),
        "over_attributed": over_attributed,
        "over_attributed_percent": _percent(over_attributed, denominator),
        "component_count": len(normalized),
    }


def timing_summary(values: Iterable[float], *, warmup_steps: int, measured_steps: int) -> dict[str, Any]:
    series = [float(value) for value in values]
    if not series:
        return {
            "warmup_steps": int(warmup_steps),
            "measured_steps": int(measured_steps),
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p90_seconds": 0.0,
            "p95_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "std_seconds": 0.0,
            "coefficient_of_variation": 0.0,
            "values_seconds": [],
        }

    sorted_values = sorted(series)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower]
        weight = position - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    mean = statistics.fmean(series)
    std = statistics.pstdev(series) if len(series) > 1 else 0.0
    return {
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(measured_steps),
        "count": len(series),
        "mean_seconds": mean,
        "median_seconds": statistics.median(series),
        "p90_seconds": percentile(0.90),
        "p95_seconds": percentile(0.95),
        "min_seconds": min(series),
        "max_seconds": max(series),
        "std_seconds": std,
        "coefficient_of_variation": (std / mean) if mean > 0.0 else 0.0,
        "values_seconds": series,
    }


@dataclass
class StageTimer:
    device: torch.device
    records: dict[str, float]

    def __call__(self, name: str) -> "_StageTimerContext":
        return _StageTimerContext(self.device, self.records, name)


class _StageTimerContext:
    def __init__(self, device: torch.device, records: dict[str, float], name: str) -> None:
        self.device = device
        self.records = records
        self.name = name
        self.start = 0.0

    def __enter__(self) -> None:
        sync_device(self.device)
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        sync_device(self.device)
        elapsed = time.perf_counter() - self.start
        self.records[self.name] = self.records.get(self.name, 0.0) + elapsed


def _run_command(command: list[str], *, cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def hardware_metadata(*, root: str | os.PathLike[str] | None = None, device: torch.device | None = None) -> dict[str, Any]:
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
            "total_memory_bytes": int(props.total_memory),
        }
    cpus_allowed = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpus_allowed = None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(resolved_device),
        "gpu": gpu,
        "cpu": platform.processor(),
        "cpus_allowed_list": cpus_allowed,
        "pid": os.getpid(),
        "git_commit": _run_command(["git", "rev-parse", "HEAD"], cwd=root),
        "command": " ".join(sys.argv),
    }


def named_parameter_nbytes(model: torch.nn.Module) -> dict[str, int]:
    return {name: tensor_nbytes(param) for name, param in model.named_parameters()}


def named_buffer_nbytes(model: torch.nn.Module) -> dict[str, int]:
    return {name: tensor_nbytes(buffer) for name, buffer in model.named_buffers()}


def sum_named_bytes(items: Mapping[str, int], *needles: str) -> int:
    return int(sum(value for name, value in items.items() if any(needle in name for needle in needles)))


def write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    import json
    from pathlib import Path

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
