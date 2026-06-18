#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


GIB = 1024**3


@dataclass(frozen=True)
class ProfileMetrics:
    run_dir: Path
    backend: str
    liger_loss: str
    peak_allocated_hbm_bytes: int
    lm_head_loss_hbm_bytes: int
    median_step_ms: float
    median_forward_ms: float
    median_backward_ms: float
    sources: dict[str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ligerloss0 and ligerloss1 LF profile artifacts.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--baseline-liger-loss", required=True, choices=("ligerloss0", "ligerloss1"))
    parser.add_argument("--candidate-liger-loss", required=True, choices=("ligerloss0", "ligerloss1"))
    parser.add_argument("--min-peak-drop-gib", type=float, default=10.0)
    parser.add_argument("--min-lm-head-loss-drop-gib", type=float, default=20.0)
    parser.add_argument("--max-step-ratio", type=float, default=1.10)
    parser.add_argument("--max-forward-ratio", type=float, default=1.15)
    parser.add_argument("--max-backward-ratio", type=float, default=1.15)
    return parser.parse_args()


def _fail(message: str, *, payload: dict[str, Any] | None = None) -> None:
    if payload:
        print(json.dumps({"ok": False, "error": message, **payload}, indent=2, sort_keys=True))
    else:
        print(json.dumps({"ok": False, "error": message}, indent=2, sort_keys=True))
    raise SystemExit(2)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _first_existing(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def load_profile_config(run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = _first_existing(run_dir, ("source_profile.json", "profile.json"))
    if path is None:
        raise FileNotFoundError(f"{run_dir} has no source_profile.json or profile.json")
    profile = _load_json(path)
    config = profile.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no config object")
    return config, path


def _validate_path_labels(run_dir: Path, backend: str, liger_loss: str) -> None:
    text = "/".join(part.name for part in [*run_dir.parents[:5], run_dir])
    if backend not in text:
        raise ValueError(f"path labels for {run_dir} do not contain backend {backend!r}")
    if liger_loss not in text:
        raise ValueError(f"path labels for {run_dir} do not contain {liger_loss!r}")


def _int_field(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _float_field(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _memory_summary_path(run_dir: Path) -> Path:
    path = _first_existing(
        run_dir,
        (
            "memory_breakdown_summary.json",
            "memory_breakdown/memory_breakdown_summary.json",
        ),
    )
    if path is None:
        raise FileNotFoundError(f"{run_dir} has no memory_breakdown_summary.json")
    return path


def _peak_allocated_hbm(summary: dict[str, Any]) -> tuple[int, str]:
    for key in ("actual_peak_allocated_hbm_bytes", "peak_allocated_hbm_bytes", "allocated_stack_sum_bytes"):
        value = _int_field(summary.get(key))
        if value is not None:
            return value, key
    raise ValueError("memory summary has no peak allocated HBM field")


def _row_is_gpu_hbm(row: dict[str, Any]) -> bool:
    memory_space = str(row.get("memory_space", "")).lower()
    return "gpu" in memory_space or "hbm" in memory_space


def _row_matches_lm_head_or_loss(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, ""))
        for key in (
            "component",
            "group",
            "kind",
            "module",
            "path",
            "name",
            "label",
            "stage",
        )
    ).lower()
    return "lm_head" in haystack or "loss" in haystack


def load_memory_metrics(run_dir: Path) -> tuple[int, int, dict[str, str]]:
    path = _memory_summary_path(run_dir)
    summary = _load_json(path)
    peak, peak_field = _peak_allocated_hbm(summary)
    rows = summary.get("actual_peak_breakdown_rows")
    row_field = "actual_peak_breakdown_rows"
    if not isinstance(rows, list) or not rows:
        rows = summary.get("breakdown_rows")
        row_field = "breakdown_rows"
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no memory breakdown rows")

    lm_head_loss = 0
    matched_rows = 0
    for row in rows:
        if not isinstance(row, dict) or not _row_is_gpu_hbm(row) or not _row_matches_lm_head_or_loss(row):
            continue
        value = _int_field(row.get("bytes"))
        if value is None:
            continue
        lm_head_loss += value
        matched_rows += 1
    if matched_rows == 0 or lm_head_loss <= 0:
        raise ValueError(f"{path} has no GPU HBM rows attributed to lm_head or loss")

    return peak, lm_head_loss, {
        "memory_summary": str(path),
        "peak_allocated_hbm_field": peak_field,
        "lm_head_loss_hbm_field": f"{row_field}[component/module/path contains lm_head|loss]",
    }


def _read_step_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _median(values: list[float], field: str, source: Path) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        raise ValueError(f"{source} has no finite {field} values")
    return float(statistics.median(clean))


def _timing_from_step_samples(run_dir: Path) -> tuple[float, float, float, dict[str, str]] | None:
    path = _first_existing(run_dir, ("step_samples.csv",))
    if path is None:
        return None
    rows = _read_step_rows(path)
    measured = [row for row in rows if not _truthy(row.get("is_warmup"))]
    if not measured:
        measured = rows
    if not measured:
        raise ValueError(f"{path} has no step sample rows")

    step_values = [_float_field(row.get("step_milliseconds")) for row in measured]
    forward_values = [_float_field(row.get("forward_milliseconds")) for row in measured]
    backward_values = [_float_field(row.get("backward_milliseconds")) for row in measured]
    return (
        _median([value for value in step_values if value is not None], "step_milliseconds", path),
        _median([value for value in forward_values if value is not None], "forward_milliseconds", path),
        _median([value for value in backward_values if value is not None], "backward_milliseconds", path),
        {
            "timing": str(path),
            "step_ms_field": "step_milliseconds",
            "forward_ms_field": "forward_milliseconds",
            "backward_ms_field": "backward_milliseconds",
        },
    )


def _timing_from_profile_summary(run_dir: Path) -> tuple[float, float, float, dict[str, str]]:
    profile_path = _first_existing(run_dir, ("source_profile.json", "profile.json"))
    if profile_path is None:
        raise FileNotFoundError(f"{run_dir} has no profile JSON for timing fallback")
    profile = _load_json(profile_path)
    trainer = profile.get("trainer", {})
    timing = trainer.get("timing", {}) if isinstance(trainer, dict) else {}
    step_ms = _float_field(timing.get("measured_e2e_step_milliseconds")) if isinstance(timing, dict) else None
    forward = profile.get("forward", {})
    backward = profile.get("backward", {})
    forward_ms = _float_field(forward.get("total_milliseconds")) if isinstance(forward, dict) else None
    backward_ms = _float_field(backward.get("total_milliseconds")) if isinstance(backward, dict) else None
    missing = [
        name
        for name, value in (
            ("trainer.timing.measured_e2e_step_milliseconds", step_ms),
            ("forward.total_milliseconds", forward_ms),
            ("backward.total_milliseconds", backward_ms),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"{profile_path} missing timing fallback fields: {', '.join(missing)}")
    return (
        float(step_ms),
        float(forward_ms),
        float(backward_ms),
        {
            "timing": str(profile_path),
            "timing_fallback": "profile_summary",
            "step_ms_field": "trainer.timing.measured_e2e_step_milliseconds",
            "forward_ms_field": "forward.total_milliseconds",
            "backward_ms_field": "backward.total_milliseconds",
        },
    )


def load_timing_metrics(run_dir: Path) -> tuple[float, float, float, dict[str, str]]:
    timing = _timing_from_step_samples(run_dir)
    if timing is not None:
        return timing
    return _timing_from_profile_summary(run_dir)


def load_profile_metrics(run_dir: Path, *, backend: str, liger_loss: str) -> ProfileMetrics:
    config, config_path = load_profile_config(run_dir)
    actual_backend = str(config.get("backend", ""))
    actual_liger_loss = str(config.get("liger_loss", ""))
    if actual_backend != backend:
        raise ValueError(f"{config_path} backend mismatch: expected {backend!r}, got {actual_backend!r}")
    if actual_liger_loss != liger_loss:
        raise ValueError(
            f"{config_path} liger_loss mismatch: expected {liger_loss!r}, got {actual_liger_loss!r}"
        )
    _validate_path_labels(run_dir, backend, liger_loss)

    peak, lm_head_loss, memory_sources = load_memory_metrics(run_dir)
    step_ms, forward_ms, backward_ms, timing_sources = load_timing_metrics(run_dir)
    return ProfileMetrics(
        run_dir=run_dir,
        backend=actual_backend,
        liger_loss=actual_liger_loss,
        peak_allocated_hbm_bytes=peak,
        lm_head_loss_hbm_bytes=lm_head_loss,
        median_step_ms=step_ms,
        median_forward_ms=forward_ms,
        median_backward_ms=backward_ms,
        sources={"config": str(config_path), **memory_sources, **timing_sources},
    )


def _ratio(candidate: float, baseline: float, label: str) -> float:
    if baseline <= 0:
        raise ValueError(f"baseline {label} must be positive")
    return candidate / baseline


def compare_metrics(baseline: ProfileMetrics, candidate: ProfileMetrics, args: argparse.Namespace) -> dict[str, Any]:
    if baseline.liger_loss == candidate.liger_loss:
        raise ValueError("baseline and candidate use the same liger_loss")
    if baseline.backend != candidate.backend:
        raise ValueError("baseline and candidate backend differ")

    peak_drop = baseline.peak_allocated_hbm_bytes - candidate.peak_allocated_hbm_bytes
    lm_head_loss_drop = baseline.lm_head_loss_hbm_bytes - candidate.lm_head_loss_hbm_bytes
    step_ratio = _ratio(candidate.median_step_ms, baseline.median_step_ms, "step ms")
    forward_ratio = _ratio(candidate.median_forward_ms, baseline.median_forward_ms, "forward ms")
    backward_ratio = _ratio(candidate.median_backward_ms, baseline.median_backward_ms, "backward ms")

    checks = {
        "peak_drop_gib": peak_drop / GIB,
        "lm_head_loss_drop_gib": lm_head_loss_drop / GIB,
        "step_ratio": step_ratio,
        "forward_ratio": forward_ratio,
        "backward_ratio": backward_ratio,
    }
    failures = []
    if peak_drop < int(float(args.min_peak_drop_gib) * GIB):
        failures.append("peak HBM drop below threshold")
    if lm_head_loss_drop < int(float(args.min_lm_head_loss_drop_gib) * GIB):
        failures.append("lm_head/loss HBM drop below threshold")
    if step_ratio > float(args.max_step_ratio):
        failures.append("step latency ratio above threshold")
    if forward_ratio > float(args.max_forward_ratio):
        failures.append("forward latency ratio above threshold")
    if backward_ratio > float(args.max_backward_ratio):
        failures.append("backward latency ratio above threshold")

    return {
        "ok": not failures,
        "failures": failures,
        "checks": checks,
        "thresholds": {
            "min_peak_drop_gib": args.min_peak_drop_gib,
            "min_lm_head_loss_drop_gib": args.min_lm_head_loss_drop_gib,
            "max_step_ratio": args.max_step_ratio,
            "max_forward_ratio": args.max_forward_ratio,
            "max_backward_ratio": args.max_backward_ratio,
        },
        "baseline": asdict(baseline) | {"run_dir": str(baseline.run_dir)},
        "candidate": asdict(candidate) | {"run_dir": str(candidate.run_dir)},
    }


def main() -> None:
    args = _parse_args()
    try:
        baseline = load_profile_metrics(
            args.baseline,
            backend=args.backend,
            liger_loss=args.baseline_liger_loss,
        )
        candidate = load_profile_metrics(
            args.candidate,
            backend=args.backend,
            liger_loss=args.candidate_liger_loss,
        )
        result = compare_metrics(baseline, candidate, args)
    except Exception as exc:
        _fail(str(exc))

    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
