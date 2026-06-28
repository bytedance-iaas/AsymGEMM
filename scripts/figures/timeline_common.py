#!/usr/bin/env python3
"""Shared helpers for paper-style per-run timeline figures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def opt_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def resolve_timeseries_path(input_path: Path, metric: str) -> Path:
    """Resolve run dir, metric dir, existing png, or CSV to data/timeseries.csv."""
    p = input_path.expanduser().resolve()
    candidates: list[Path] = []
    if p.is_file():
        if p.name == "timeseries.csv":
            candidates.append(p)
        candidates.extend([
            p.parent / "data" / "timeseries.csv",
            p.parent / "timeseries.csv",
            p.parent.parent / "data" / "timeseries.csv",
        ])
    else:
        candidates.extend([
            p / "timeseries.csv",
            p / "data" / "timeseries.csv",
            p / metric / "data" / "timeseries.csv",
            p / "metrics" / metric / "data" / "timeseries.csv",
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {metric}/data/timeseries.csv from input path: {input_path}"
    )


def metric_dir_from_timeseries(path: Path) -> Path:
    # .../metrics/<metric>/data/timeseries.csv -> .../metrics/<metric>
    return path.parent.parent


def run_dir_from_metric_dir(path: Path) -> Path:
    # .../<run>/metrics/<metric> -> .../<run>
    return path.parent.parent


def infer_step_count(metric_dir: Path) -> int:
    bounds = measured_step_bounds(metric_dir)
    if bounds:
        return len(bounds)

    run_dir = run_dir_from_metric_dir(metric_dir)
    timing_csv = run_dir / "metrics" / "timing" / "data" / "step_samples.csv"
    if timing_csv.is_file():
        rows = read_csv_rows(timing_csv)
        measured = set()
        count = 0
        for row in rows:
            if str(row.get("is_warmup", "")).lower() in {"true", "1", "yes"}:
                continue
            step = row.get("measured_step") or row.get("step")
            if step not in (None, ""):
                measured.add(str(step))
            count += 1
        return max(len(measured), count, 1)

    step_summary = metric_dir / "data" / "step_summary.csv"
    if step_summary.is_file():
        steps = {
            str(row.get("step"))
            for row in read_csv_rows(step_summary)
            if row.get("scope") == "step" and row.get("step") not in (None, "")
        }
        if steps:
            return len(steps)
    return 1


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _step_rows_from_profile(profile: dict) -> list[dict]:
    source_profile = profile.get("source_profile")
    if isinstance(source_profile, dict):
        step_samples = source_profile.get("step_samples")
        rows = step_samples.get("rows") if isinstance(step_samples, dict) else None
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    step_samples = profile.get("step_samples")
    rows = step_samples.get("rows") if isinstance(step_samples, dict) else None
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def measured_step_bounds(metric_dir: Path) -> list[tuple[float, float]]:
    profile_json = run_dir_from_metric_dir(metric_dir) / "profile.json"
    if not profile_json.is_file():
        return []
    try:
        profile = json.loads(profile_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    bounds: list[tuple[float, float]] = []
    for row in _step_rows_from_profile(profile):
        if _truthy(row.get("is_warmup")):
            continue
        start = opt_float(row.get("trainer_e2e_start_seconds"))
        end = opt_float(row.get("trainer_e2e_end_seconds"))
        if start is None or end is None or end <= start:
            continue
        bounds.append((start, end))
    return sorted(bounds, key=lambda item: item[0])


def _phase_duration_seconds(row: dict, *keys: str) -> float:
    for key in keys:
        value = opt_float(row.get(key))
        if value is not None and value > 0.0:
            return value / 1000.0
    return 0.0


def measured_step_phase_spans(
    metric_dir: Path,
    bounds: list[tuple[float, float]],
) -> list[tuple[str, float, float]]:
    """Return phase spans in measured-step coordinates: (phase, x0, x1)."""
    profile_json = run_dir_from_metric_dir(metric_dir) / "profile.json"
    if not profile_json.is_file():
        return []
    try:
        profile = json.loads(profile_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    rows = [row for row in _step_rows_from_profile(profile) if not _truthy(row.get("is_warmup"))]
    spans: list[tuple[str, float, float]] = []
    for index, row in enumerate(rows):
        have_bounds = index < len(bounds)
        denom = (bounds[index][1] - bounds[index][0]) if have_bounds else 0.0
        if denom <= 0.0:
            denom = _phase_duration_seconds(row, "trainer_e2e_step_milliseconds", "step_milliseconds")
        if denom <= 0.0:
            continue
        fwd_s = _phase_duration_seconds(row, "forward_milliseconds", "source_forward_milliseconds")
        bwd_s = _phase_duration_seconds(row, "backward_milliseconds", "source_backward_milliseconds")
        fwd = fwd_s / denom
        bwd = bwd_s / denom
        if fwd + bwd > 1.0:
            scale = fwd + bwd
            fwd, bwd = fwd / scale, bwd / scale
        x = float(index)
        if fwd > 0.0:
            spans.append(("forward", x, x + fwd))
        if bwd > 0.0:
            spans.append(("backward", x + fwd, x + fwd + bwd))
        if (1.0 - fwd - bwd) > 1e-4:
            spans.append(("optimizer", x + fwd + bwd, x + 1.0))
    return spans


def aggregate_max_by_x(
    rows: Iterable[dict[str, str]],
    *,
    metric_column: str,
    x_column: str,
    y_column: str = "value_percent",
) -> dict[str, list[tuple[float, float]]]:
    """Group duplicate timestamps/GPU rows by metric with max y per x."""
    grouped: dict[str, dict[float, float]] = {}
    for row in rows:
        metric = str(row.get(metric_column) or "")
        x = opt_float(row.get(x_column))
        y = opt_float(row.get(y_column))
        if not metric or x is None or y is None:
            continue
        y = min(100.0, max(0.0, y))
        series = grouped.setdefault(metric, {})
        series[x] = max(series.get(x, 0.0), y)
    return {
        metric: sorted(points.items(), key=lambda item: item[0])
        for metric, points in grouped.items()
    }


def resample_to_measured_steps(
    series: list[tuple[float, float]],
    *,
    n_steps: int,
    points_per_step: int,
    agg: str,
) -> list[tuple[float, float]]:
    """Affine-map a raw timeline to measured-step x coordinates and bucket it."""
    if not series or n_steps <= 0 or points_per_step <= 0:
        return []
    lo, hi = series[0][0], series[-1][0]
    total_buckets = n_steps * points_per_step
    if total_buckets <= 0:
        return []
    buckets: list[list[float]] = [[] for _ in range(total_buckets)]
    span = hi - lo
    if span <= 0:
        buckets[0] = [y for _, y in series]
    else:
        for x, y in series:
            frac = (x - lo) / span
            slot = min(total_buckets - 1, max(0, int(frac * total_buckets)))
            buckets[slot].append(y)
    out: list[tuple[float, float]] = []
    for slot, values in enumerate(buckets):
        if not values:
            continue
        value = max(values) if agg == "max" else sum(values) / len(values)
        out.append(((slot + 0.5) / points_per_step, value))
    return out


def affine_to_bounds_window(
    series: list[tuple[float, float]],
    bounds: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Map a raw series' x-span onto the measured-step time window."""
    if not series or not bounds:
        return series
    start, end = bounds[0][0], bounds[-1][1]
    lo, hi = series[0][0], series[-1][0]
    if end <= start or hi <= lo:
        return series
    scale = (end - start) / (hi - lo)
    return [(start + (x - lo) * scale, y) for x, y in series]


def resample_with_bounds(
    series: list[tuple[float, float]],
    *,
    bounds: list[tuple[float, float]],
    points_per_step: int,
    agg: str,
) -> list[tuple[float, float]]:
    """Bucket samples inside explicit step bounds; x = measured step index."""
    if not series or not bounds or points_per_step <= 0:
        return []
    out: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(bounds):
        if end <= start:
            continue
        buckets: list[list[float]] = [[] for _ in range(points_per_step)]
        for x, y in series:
            if start <= x < end:
                slot = int((x - start) / (end - start) * points_per_step)
                buckets[min(slot, points_per_step - 1)].append(y)
        for slot, values in enumerate(buckets):
            if values:
                value = max(values) if agg == "max" else sum(values) / len(values)
                out.append((index + (slot + 0.5) / points_per_step, value))
    return out
