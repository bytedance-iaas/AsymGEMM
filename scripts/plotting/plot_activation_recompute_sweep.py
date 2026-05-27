#!/usr/bin/env python3
"""Plot activation-recompute sequence-length sweeps from LoRA driver outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MIB = 1024.0**2
RESULT_RE = re.compile(
    r"^(?P<precision>.+?)_lora-sft_b(?P<batch_size>[0-9]+)_s(?P<seq_len>[0-9]+)_"
    r"(?P<recompute>recomp|norecomp)(?:_expertthr(?P<expert_threshold>[0-9]+))?_(?P<tail>.+)$"
)
PROFILERS = ("source", "nsys", "cpu", "ncu")
LINEAR_REGION_R2_THRESHOLD = 0.99
LINEAR_REGION_RATIO_CV_THRESHOLD = 0.08
MIN_LINEAR_REGION_POINTS = 4
SUBLINEAR_SLOPE_TOLERANCE = 0.08
SUBLINEAR_COLOR = "green"
SUBLINEAR_ALPHA = 0.055
ROOT_OUTPUT_FILES = (
    "activation_recompute_sweep_index.csv",
    "activation_recompute_sweep_index.json",
)
COMBINED_OUTPUT_FILES = (
    "combined_forward_end_memory_vs_seq.png",
    "combined_forward_peak_memory_vs_seq.png",
    "combined_backward_start_memory_vs_seq.png",
    "combined_backward_peak_memory_vs_seq.png",
    "combined_peak_hbm_vs_seq.png",
    "combined_timing_vs_seq.png",
    "combined_forward_end_memory_vs_expert_threshold.png",
    "combined_forward_peak_memory_vs_expert_threshold.png",
    "combined_backward_start_memory_vs_expert_threshold.png",
    "combined_backward_peak_memory_vs_expert_threshold.png",
    "combined_peak_hbm_vs_expert_threshold.png",
    "combined_timing_vs_expert_threshold.png",
)
GROUP_OUTPUT_FILES = (
    "sweep_summary.csv",
    "sweep_summary.json",
    "forward_end_memory_vs_seq.png",
    "forward_peak_memory_vs_seq.png",
    "backward_start_memory_vs_seq.png",
    "backward_peak_memory_vs_seq.png",
    "peak_hbm_vs_seq.png",
    "timing_vs_seq.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "profiling")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <input-root>/activation_recompute_plots.",
    )
    parser.add_argument("--precision", default="")
    parser.add_argument("--workload", action="append", default=[], help="Workload label to include, e.g. moe-604m-a38m-l2.")
    parser.add_argument("--backend", action="append", default=[], choices=["asym", "torch", "kt"])
    parser.add_argument("--profiler", action="append", default=[], choices=list(PROFILERS))
    parser.add_argument(
        "--recompute",
        action="append",
        default=[],
        choices=["norecomp", "recomp", "no_recompute", "recompute"],
        help="Activation recompute mode to include. Repeat for both.",
    )
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--seq-len", "--seq-lens", dest="seq_lens", nargs="+", type=int, default=[])
    parser.add_argument("--expert-recompute-threshold", action="append", type=int, default=[])
    parser.add_argument(
        "--expert-recompute-thresholds",
        nargs="+",
        type=int,
        default=[],
        help="Expert threshold filter. 0 means no fine-grained expert recompute.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previously generated plot artifacts from the output directory before writing this filtered run.",
    )
    return parser.parse_args()


def safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch == "-" else "_" for ch in value).strip("_-")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def output_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    return resolve_path(args.input_root) / "activation_recompute_plots"


def parse_result_dir(path: Path) -> dict[str, Any] | None:
    match = RESULT_RE.match(path.name)
    if match is None:
        return None
    tail = match.group("tail")
    profiler = next((candidate for candidate in PROFILERS if tail.endswith(f"_{candidate}")), "")
    if not profiler:
        return None
    backend = tail[: -(len(profiler) + 1)]
    if not backend:
        return None
    return {
        "precision": match.group("precision"),
        "batch_size": int(match.group("batch_size")),
        "seq_len": int(match.group("seq_len")),
        "mode": "recompute" if match.group("recompute") == "recomp" else "no_recompute",
        "activation_recompute": match.group("recompute") == "recomp",
        "expert_recompute_threshold": int(match.group("expert_threshold") or 0),
        "backend": backend,
        "profiler": profiler,
        "workload": path.parent.name,
    }


def profile_json_path(result_dir: Path) -> Path | None:
    candidates = [result_dir / "profile.json", *sorted(result_dir.glob("*_profile.json"))]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_views(profile: dict[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for candidate in (profile.get("source_profile"), profile, profile.get("memory_profile")):
        if isinstance(candidate, dict):
            views.append(candidate)
    return views


def first_dict(profile: dict[str, Any], key: str) -> dict[str, Any]:
    for view in profile_views(profile):
        value = view.get(key)
        if isinstance(value, dict):
            return value
    return {}


def stage_row(profile: dict[str, Any], name: str) -> dict[str, Any]:
    for view in profile_views(profile):
        rows = view.get("stage_memory", {}).get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("name") == name:
                return row
    return {}


def numeric_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def numeric_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def nested_float(mapping: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = mapping.get(section)
    if not isinstance(value, dict):
        return default
    return numeric_float(value.get(key), default)


def nested_int(mapping: dict[str, Any], section: str, key: str, default: int = 0) -> int:
    value = mapping.get(section)
    if not isinstance(value, dict):
        return default
    return numeric_int(value.get(key), default)


def to_mib(value: Any) -> float:
    return numeric_float(value) / MIB


def step_ms(profile: dict[str, Any]) -> float:
    stages = profile.get("stages")
    if isinstance(stages, list):
        total = 0.0
        found = False
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            value = stage.get("total_milliseconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += float(value)
                found = True
        if found:
            return total
    return numeric_float(first_dict(profile, "step").get("total_milliseconds"))


def passes_filters(args: argparse.Namespace, meta: dict[str, Any]) -> bool:
    recompute_modes = {
        {"norecomp": "no_recompute", "recomp": "recompute"}.get(mode, mode)
        for mode in args.recompute
    }
    if args.precision and meta["precision"] != args.precision:
        return False
    if args.workload and meta["workload"] not in set(args.workload):
        return False
    if args.backend and meta["backend"] not in set(args.backend):
        return False
    if args.profiler and meta["profiler"] not in set(args.profiler):
        return False
    if recompute_modes and meta["mode"] not in recompute_modes:
        return False
    if args.batch_size and meta["batch_size"] not in set(args.batch_size):
        return False
    if args.seq_lens and meta["seq_len"] not in set(args.seq_lens):
        return False
    threshold_filter = set(args.expert_recompute_threshold) | set(args.expert_recompute_thresholds)
    if threshold_filter and int(meta["expert_recompute_threshold"]) not in threshold_filter:
        return False
    return True


def result_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*_lora-sft_*") if path.is_dir())


def row_from_result_dir(args: argparse.Namespace, result_dir: Path) -> dict[str, Any] | None:
    meta = parse_result_dir(result_dir)
    if meta is None or not passes_filters(args, meta):
        return None
    if int(meta["expert_recompute_threshold"]) > 0 and bool(meta["activation_recompute"]):
        # Current driver semantics reserve layer recompute for threshold 0.
        # Ignore stale dirs from older runs that combined layer and expert recompute.
        return None
    profile_path = profile_json_path(result_dir)
    if profile_path is None:
        return None
    profile = load_json(profile_path)
    config = first_dict(profile, "config")
    forward = stage_row(profile, "step.forward")
    backward = stage_row(profile, "step.backward")
    memory = first_dict(profile, "memory")
    memory_gpu = memory.get("gpu", {})
    if not isinstance(memory_gpu, dict):
        memory_gpu = {}
    batch_size = int(config.get("batch_size", meta["batch_size"]))
    route_stats = first_dict(profile, "expert_token_distribution")
    threshold_effect = route_stats.get("threshold_effect", {})
    if not isinstance(threshold_effect, dict):
        threshold_effect = {}
    return {
        "workload": meta["workload"],
        "precision": meta["precision"],
        "batch_size": batch_size,
        "seq_len": int(meta["seq_len"]),
        "logical_tokens": int(config.get("logical_tokens", batch_size * int(meta["seq_len"]))),
        "mode": meta["mode"],
        "activation_recompute": bool(meta["activation_recompute"]),
        "expert_recompute_threshold": int(meta["expert_recompute_threshold"]),
        "backend": meta["backend"],
        "profiler": meta["profiler"],
        "step_ms": step_ms(profile),
        "forward_ms": numeric_float(first_dict(profile, "forward").get("total_milliseconds")),
        "backward_ms": numeric_float(first_dict(profile, "backward").get("total_milliseconds")),
        "peak_hbm_mib": to_mib(memory_gpu.get("peak_hbm_bytes")),
        "stage_local_peak_hbm_mib": to_mib(memory_gpu.get("stage_local_peak_hbm_bytes")),
        "forward_alloc_start_mib": to_mib(forward.get("avg_allocated_start_bytes")),
        "forward_alloc_end_mib": to_mib(forward.get("avg_allocated_end_bytes")),
        "forward_live_delta_mib": to_mib(forward.get("avg_allocated_delta_bytes")),
        "forward_local_peak_mib": to_mib(forward.get("avg_local_peak_bytes")),
        "forward_local_peak_delta_mib": to_mib(forward.get("avg_local_peak_delta_bytes")),
        "backward_alloc_start_mib": to_mib(backward.get("avg_allocated_start_bytes")),
        "backward_alloc_end_mib": to_mib(backward.get("avg_allocated_end_bytes")),
        "backward_alloc_delta_mib": to_mib(backward.get("avg_allocated_delta_bytes")),
        "backward_local_peak_mib": to_mib(backward.get("avg_local_peak_bytes")),
        "backward_local_peak_delta_mib": to_mib(backward.get("avg_local_peak_delta_bytes")),
        "route_samples": numeric_int(route_stats.get("samples")),
        "route_num_tokens": numeric_int(route_stats.get("num_tokens")),
        "route_top_k": numeric_int(route_stats.get("top_k")),
        "route_num_experts": numeric_int(route_stats.get("num_experts")),
        "route_all_expert_tokens_avg": nested_float(route_stats, "all_expert_tokens", "avg"),
        "route_all_expert_tokens_median": nested_float(route_stats, "all_expert_tokens", "median"),
        "route_all_expert_tokens_min": nested_int(route_stats, "all_expert_tokens", "min"),
        "route_all_expert_tokens_max": nested_int(route_stats, "all_expert_tokens", "max"),
        "route_all_expert_tokens_p0": nested_float(route_stats, "all_expert_tokens", "p0"),
        "route_all_expert_tokens_p25": nested_float(route_stats, "all_expert_tokens", "p25"),
        "route_all_expert_tokens_p50": nested_float(route_stats, "all_expert_tokens", "p50"),
        "route_all_expert_tokens_p75": nested_float(route_stats, "all_expert_tokens", "p75"),
        "route_all_expert_tokens_p90": nested_float(route_stats, "all_expert_tokens", "p90"),
        "route_all_expert_tokens_p100": nested_float(route_stats, "all_expert_tokens", "p100"),
        "route_active_expert_tokens_avg": nested_float(route_stats, "active_expert_tokens", "avg"),
        "route_active_expert_tokens_median": nested_float(route_stats, "active_expert_tokens", "median"),
        "route_active_expert_tokens_min": nested_int(route_stats, "active_expert_tokens", "min"),
        "route_active_expert_tokens_max": nested_int(route_stats, "active_expert_tokens", "max"),
        "route_active_expert_tokens_p0": nested_float(route_stats, "active_expert_tokens", "p0"),
        "route_active_expert_tokens_p25": nested_float(route_stats, "active_expert_tokens", "p25"),
        "route_active_expert_tokens_p50": nested_float(route_stats, "active_expert_tokens", "p50"),
        "route_active_expert_tokens_p75": nested_float(route_stats, "active_expert_tokens", "p75"),
        "route_active_expert_tokens_p90": nested_float(route_stats, "active_expert_tokens", "p90"),
        "route_active_expert_tokens_p100": nested_float(route_stats, "active_expert_tokens", "p100"),
        "route_active_experts_avg": nested_float(route_stats, "active_experts", "avg"),
        "route_active_experts_median": nested_float(route_stats, "active_experts", "median"),
        "route_active_experts_min": nested_int(route_stats, "active_experts", "min"),
        "route_active_experts_max": nested_int(route_stats, "active_experts", "max"),
        "route_active_experts_p0": nested_float(route_stats, "active_experts", "p0"),
        "route_active_experts_p25": nested_float(route_stats, "active_experts", "p25"),
        "route_active_experts_p50": nested_float(route_stats, "active_experts", "p50"),
        "route_active_experts_p75": nested_float(route_stats, "active_experts", "p75"),
        "route_active_experts_p90": nested_float(route_stats, "active_experts", "p90"),
        "route_active_experts_p100": nested_float(route_stats, "active_experts", "p100"),
        "route_samples_with_recompute": numeric_int(threshold_effect.get("samples_with_recompute")),
        "route_samples_all_active_recomputed": numeric_int(threshold_effect.get("samples_all_active_recomputed")),
        "route_recomputed_experts_avg": numeric_float(threshold_effect.get("recomputed_experts_avg")),
        "route_recomputed_experts_min": numeric_int(threshold_effect.get("recomputed_experts_min")),
        "route_recomputed_experts_max": numeric_int(threshold_effect.get("recomputed_experts_max")),
        "route_recomputed_experts_p0": numeric_float(threshold_effect.get("recomputed_experts_p0")),
        "route_recomputed_experts_p25": numeric_float(threshold_effect.get("recomputed_experts_p25")),
        "route_recomputed_experts_p50": numeric_float(threshold_effect.get("recomputed_experts_p50")),
        "route_recomputed_experts_p75": numeric_float(threshold_effect.get("recomputed_experts_p75")),
        "route_recomputed_experts_p90": numeric_float(threshold_effect.get("recomputed_experts_p90")),
        "route_recomputed_experts_p100": numeric_float(threshold_effect.get("recomputed_experts_p100")),
        "route_kept_experts_avg": numeric_float(threshold_effect.get("kept_experts_avg")),
        "route_kept_experts_min": numeric_int(threshold_effect.get("kept_experts_min")),
        "route_kept_experts_max": numeric_int(threshold_effect.get("kept_experts_max")),
        "route_kept_experts_p0": numeric_float(threshold_effect.get("kept_experts_p0")),
        "route_kept_experts_p25": numeric_float(threshold_effect.get("kept_experts_p25")),
        "route_kept_experts_p50": numeric_float(threshold_effect.get("kept_experts_p50")),
        "route_kept_experts_p75": numeric_float(threshold_effect.get("kept_experts_p75")),
        "route_kept_experts_p90": numeric_float(threshold_effect.get("kept_experts_p90")),
        "route_kept_experts_p100": numeric_float(threshold_effect.get("kept_experts_p100")),
        "route_recomputed_routes_avg": numeric_float(threshold_effect.get("recomputed_routes_avg")),
        "route_recomputed_routes_min": numeric_int(threshold_effect.get("recomputed_routes_min")),
        "route_recomputed_routes_max": numeric_int(threshold_effect.get("recomputed_routes_max")),
        "route_recomputed_routes_p0": numeric_float(threshold_effect.get("recomputed_routes_p0")),
        "route_recomputed_routes_p25": numeric_float(threshold_effect.get("recomputed_routes_p25")),
        "route_recomputed_routes_p50": numeric_float(threshold_effect.get("recomputed_routes_p50")),
        "route_recomputed_routes_p75": numeric_float(threshold_effect.get("recomputed_routes_p75")),
        "route_recomputed_routes_p90": numeric_float(threshold_effect.get("recomputed_routes_p90")),
        "route_recomputed_routes_p100": numeric_float(threshold_effect.get("recomputed_routes_p100")),
        "route_kept_routes_avg": numeric_float(threshold_effect.get("kept_routes_avg")),
        "route_kept_routes_min": numeric_int(threshold_effect.get("kept_routes_min")),
        "route_kept_routes_max": numeric_int(threshold_effect.get("kept_routes_max")),
        "route_kept_routes_p0": numeric_float(threshold_effect.get("kept_routes_p0")),
        "route_kept_routes_p25": numeric_float(threshold_effect.get("kept_routes_p25")),
        "route_kept_routes_p50": numeric_float(threshold_effect.get("kept_routes_p50")),
        "route_kept_routes_p75": numeric_float(threshold_effect.get("kept_routes_p75")),
        "route_kept_routes_p90": numeric_float(threshold_effect.get("kept_routes_p90")),
        "route_kept_routes_p100": numeric_float(threshold_effect.get("kept_routes_p100")),
        "output_dir": str(result_dir),
        "profile_json": str(profile_path),
    }


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_root = resolve_path(args.input_root)
    rows = [row for path in result_dirs(input_root) if (row := row_from_result_dir(args, path)) is not None]
    return sorted(
        rows,
        key=lambda row: (
            row["workload"],
            row["batch_size"],
            row["backend"],
            row["profiler"],
            row["seq_len"],
            row["mode"],
            row["expert_recompute_threshold"],
        ),
    )


def group_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (str(row["workload"]), str(row["precision"]), int(row["batch_size"]), str(row["backend"]), str(row["profiler"]))


def threshold_group_key(row: dict[str, Any]) -> tuple[str, str, int, int, str, str]:
    return (
        str(row["workload"]),
        str(row["precision"]),
        int(row["batch_size"]),
        int(row["seq_len"]),
        str(row["backend"]),
        str(row["profiler"]),
    )


def varied_fields(rows: list[dict[str, Any]]) -> set[str]:
    fields = ("workload", "precision", "batch_size", "backend", "profiler", "mode")
    return {field for field in fields if len({row[field] for row in rows}) > 1}


def varied_threshold_fields(rows: list[dict[str, Any]]) -> set[str]:
    fields = ("workload", "precision", "batch_size", "seq_len", "backend", "profiler", "mode")
    return {field for field in fields if len({row[field] for row in rows}) > 1}


def combined_label(group: tuple[str, str, int, str, str], mode: str, varied: set[str]) -> str:
    workload, precision, batch_size, backend, profiler = group
    mode_labels = {"no_recompute": "No recompute", "recompute": "Activation recompute"}
    parts: list[str] = []
    if "workload" in varied:
        parts.append(workload)
    if "batch_size" in varied:
        parts.append(f"b{batch_size}")
    if "precision" in varied:
        parts.append(precision)
    if "backend" in varied:
        parts.append(backend)
    if "profiler" in varied:
        parts.append(profiler)
    if "mode" in varied:
        parts.append(mode_labels.get(mode, mode))
    if parts:
        return " / ".join(parts)
    return f"{backend} / {mode_labels.get(mode, mode)}"


def combined_threshold_label(group: tuple[str, str, int, int, str, str], mode: str, varied: set[str]) -> str:
    workload, precision, batch_size, seq_len, backend, profiler = group
    mode_labels = {"no_recompute": "No layer recompute", "recompute": "Layer recompute"}
    parts: list[str] = []
    if "workload" in varied:
        parts.append(workload)
    if "batch_size" in varied:
        parts.append(f"b{batch_size}")
    if "seq_len" in varied:
        parts.append(f"s{seq_len}")
    if "precision" in varied:
        parts.append(precision)
    if "backend" in varied:
        parts.append(backend)
    if "profiler" in varied:
        parts.append(profiler)
    if "mode" in varied:
        parts.append(mode_labels.get(mode, mode))
    if parts:
        return " / ".join(parts)
    return f"s{seq_len} / {backend} / {mode_labels.get(mode, mode)}"


def write_table(rows: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    # Combined plots used to be written at the root. Remove those legacy files
    # so the root only contains the sweep index.
    for name in ROOT_OUTPUT_FILES + COMBINED_OUTPUT_FILES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "_combined":
            for name in COMBINED_OUTPUT_FILES:
                path = child / name
                if path.is_file():
                    path.unlink()
        for name in GROUP_OUTPUT_FILES:
            path = child / name
            if path.is_file():
                path.unlink()
        for path in child.glob("expert_threshold_summary_s*.csv"):
            path.unlink()
        for path in child.glob("expert_threshold_summary_s*.json"):
            path.unlink()
        for path in child.glob("*_vs_expert_threshold_s*.png"):
            path.unlink()
        try:
            child.rmdir()
        except OSError:
            pass


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) <= 1e-12:
        return 0.0 if all(abs(value) <= 1e-12 for value in values) else float("inf")
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    count = len(points)
    if count < 2:
        return 0.0, points[0][1] if points else 0.0, 1.0
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    ss_x = sum((x - mean_x) ** 2 for x, _ in points)
    if ss_x <= 0.0:
        return 0.0, mean_y, 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / ss_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    if ss_tot <= 0.0:
        return slope, intercept, 1.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    return slope, intercept, 1.0 - ss_res / ss_tot


def offset_normalized_ratio_cv(points: list[tuple[float, float]], intercept: float) -> float:
    ratios: list[float] = []
    for x, y in points:
        shifted_y = y - intercept
        if x <= 0.0 or shifted_y <= 0.0:
            return float("inf")
        ratios.append(shifted_y / x)
    return coefficient_of_variation(ratios)


def series_points(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in sorted(rows, key=lambda item: item["seq_len"]):
        x = float(row["seq_len"])
        y = float(row[key]) / scale
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def linear_region(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> tuple[float, float] | None:
    points = series_points(rows, key, scale=scale)
    if not points:
        return None
    if len(points) == 1:
        return points[0][0], points[0][0]

    min_points = min(MIN_LINEAR_REGION_POINTS, len(points))
    best_region: tuple[float, float] | None = None
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    fallback_region = (points[0][0], points[-1][0])
    fallback_key = (float("inf"), float("inf"), float("-inf"))

    for start in range(0, len(points) - min_points + 1):
        for stop in range(start + min_points - 1, len(points)):
            window = points[start : stop + 1]
            slope, intercept, r2 = linear_fit(window)
            ratio_cv = offset_normalized_ratio_cv(window, intercept)
            span = window[-1][0] - window[0][0]
            score = (ratio_cv, max(0.0, LINEAR_REGION_R2_THRESHOLD - r2), -span)
            if score < fallback_key:
                fallback_region = (window[0][0], window[-1][0])
                fallback_key = score
            if slope <= 0.0:
                continue
            if r2 < LINEAR_REGION_R2_THRESHOLD or ratio_cv > LINEAR_REGION_RATIO_CV_THRESHOLD:
                continue
            valid_key = (span, float(len(window)), -ratio_cv, r2)
            if valid_key > best_key:
                best_region = (window[0][0], window[-1][0])
                best_key = valid_key
    return best_region if best_region is not None else fallback_region


def reference_line(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> tuple[float, float] | None:
    points = series_points(rows, key, scale=scale)
    region = linear_region(rows, key, scale=scale)
    if region is None:
        return None
    left, right = region
    window = [(x, y) for x, y in points if left <= x <= right]
    slope, intercept, _ = linear_fit(window)
    if slope <= 0.0:
        return None
    return slope, intercept


def sublinear_regions(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> list[tuple[float, float]]:
    points = series_points(rows, key, scale=scale)
    reference = reference_line(rows, key, scale=scale)
    if len(points) < 2 or reference is None:
        return []
    slope, _intercept = reference
    regions: list[tuple[float, float]] = []
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= 0.0 or right_x <= left_x:
            continue
        # Treat each adjacent pair as its own region. Translating the left point
        # to (0, 0) leaves dx and dy, so sublinearity is local slope vs. the
        # fitted linear reference slope.
        local_slope = (right_y - left_y) / (right_x - left_x)
        if 0.0 < local_slope < slope * (1.0 - SUBLINEAR_SLOPE_TOLERANCE):
            regions.append((left_x, right_x))
    return regions


def merge_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for left, right in sorted(regions):
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def draw_sublinear_regions(ax: Any, regions: list[tuple[float, float]]) -> bool:
    unique_regions = sorted({(left, right) for left, right in regions if right > left})
    merged_regions = merge_regions(unique_regions)
    for left, right in merged_regions:
        ax.axvspan(left, right, color=SUBLINEAR_COLOR, alpha=SUBLINEAR_ALPHA, zorder=0.1)
        ax.axvline(left, color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4, alpha=0.8, zorder=0.8)
        ax.axvline(right, color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4, alpha=0.8, zorder=0.8)
    return bool(merged_regions)


def add_legend(ax: Any, *, sublinear_region: bool, fontsize: int | None = None) -> None:
    if not sublinear_region:
        ax.legend(fontsize=fontsize)
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4))
    labels.append("Sublinear boundary")
    handles.append(Patch(facecolor=SUBLINEAR_COLOR, alpha=SUBLINEAR_ALPHA, edgecolor="none"))
    labels.append("Sublinear region")
    ax.legend(handles, labels, fontsize=fontsize)


def plot_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    by_mode = {
        "no_recompute": sorted((row for row in rows if row["mode"] == "no_recompute"), key=lambda row: row["seq_len"]),
        "recompute": sorted((row for row in rows if row["mode"] == "recompute"), key=lambda row: row["seq_len"]),
    }
    labels = {"no_recompute": "No recompute", "recompute": "Activation recompute"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    sublinear_spans: list[tuple[float, float]] = []
    for mode, mode_rows in by_mode.items():
        if not mode_rows:
            continue
        ax.plot(
            [row["seq_len"] for row in mode_rows],
            [float(row[key]) / scale for row in mode_rows],
            marker="o",
            linewidth=2,
            label=labels[mode],
        )
        sublinear_spans.extend(sublinear_regions(mode_rows, key, scale=scale))
    has_sublinear_region = draw_sublinear_regions(ax, sublinear_spans)
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    add_legend(ax, sublinear_region=has_sublinear_region)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_combined_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[tuple[tuple[str, str, int, str, str], str], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((group_key(row), str(row["mode"])), []).append(row)
    varied = varied_fields(rows)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    sublinear_spans: list[tuple[float, float]] = []
    for (group, mode), group_rows in sorted(series.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row["seq_len"])
        label = combined_label(group, mode, varied)
        ax.plot(
            [row["seq_len"] for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            marker="o",
            linewidth=1.8,
            label=label,
        )
        sublinear_spans.extend(sublinear_regions(sorted_rows, key, scale=scale))
    has_sublinear_region = draw_sublinear_regions(ax, sublinear_spans)
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    add_legend(ax, sublinear_region=has_sublinear_region, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def threshold_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[tuple[tuple[str, str, int, int, str, str], str], set[int]] = {}
    for row in rows:
        by_series.setdefault((threshold_group_key(row), str(row["mode"])), set()).add(int(row["expert_recompute_threshold"]))
    eligible = {key for key, thresholds in by_series.items() if len(thresholds) >= 2}
    return [row for row in rows if (threshold_group_key(row), str(row["mode"])) in eligible]


def plot_threshold_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    by_mode = {
        "no_recompute": sorted((row for row in rows if row["mode"] == "no_recompute"), key=lambda row: row["expert_recompute_threshold"]),
        "recompute": sorted((row for row in rows if row["mode"] == "recompute"), key=lambda row: row["expert_recompute_threshold"]),
    }
    labels = {"no_recompute": "No layer recompute", "recompute": "Layer recompute"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    plotted = False
    for mode, mode_rows in by_mode.items():
        if len({int(row["expert_recompute_threshold"]) for row in mode_rows}) < 2:
            continue
        ax.plot(
            [row["expert_recompute_threshold"] for row in mode_rows],
            [float(row[key]) / scale for row in mode_rows],
            marker="o",
            linewidth=2,
            label=labels[mode],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Expert recompute threshold (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_combined_threshold_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[tuple[tuple[str, str, int, int, str, str], str], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((threshold_group_key(row), str(row["mode"])), []).append(row)
    varied = varied_threshold_fields(rows)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    plotted = False
    for (group, mode), group_rows in sorted(series.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row["expert_recompute_threshold"])
        if len({int(row["expert_recompute_threshold"]) for row in sorted_rows}) < 2:
            continue
        label = combined_threshold_label(group, mode, varied)
        ax.plot(
            [row["expert_recompute_threshold"] for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            marker="o",
            linewidth=1.8,
            label=label,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Expert recompute threshold (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def write_group_plots(rows: list[dict[str, Any]], output_dir: Path, key: tuple[str, str, int, str, str]) -> None:
    workload, precision, batch_size, backend, profiler = key
    title_base = f"{workload} LoRA SFT"
    suffix = f", batch size {batch_size}, {precision}, {backend}/{profiler}"
    plot_metric(
        rows,
        output_dir,
        "forward_end_memory_vs_seq.png",
        f"{title_base} memory after forward{suffix}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "forward_peak_memory_vs_seq.png",
        f"{title_base} forward local peak{suffix}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "backward_start_memory_vs_seq.png",
        f"{title_base} memory carried into backward{suffix}",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "backward_peak_memory_vs_seq.png",
        f"{title_base} backward local peak{suffix}",
        "Backward local peak allocation (GiB)",
        "backward_local_peak_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "peak_hbm_vs_seq.png",
        f"{title_base} whole-step peak HBM{suffix}",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "timing_vs_seq.png",
        f"{title_base} step time{suffix}",
        "Step time (ms)",
        "step_ms",
    )


def write_group_threshold_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    key: tuple[str, str, int, str, str],
    seq_len: int,
) -> None:
    workload, precision, batch_size, backend, profiler = key
    title_base = f"{workload} LoRA SFT"
    suffix = f", batch size {batch_size}, seq {seq_len}, {precision}, {backend}/{profiler}"
    plot_threshold_metric(
        rows,
        output_dir,
        f"forward_end_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} memory after forward{suffix}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"forward_peak_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} forward local peak{suffix}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"backward_start_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} memory carried into backward{suffix}",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"peak_hbm_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} whole-step peak HBM{suffix}",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"timing_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} step time vs expert threshold{suffix}",
        "Step time (ms)",
        "step_ms",
    )


def write_combined_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_metric(
        rows,
        output_dir,
        "combined_forward_end_memory_vs_seq.png",
        "All workloads: memory after forward",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_forward_peak_memory_vs_seq.png",
        "All workloads: forward local peak",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_backward_start_memory_vs_seq.png",
        "All workloads: memory carried into backward",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_backward_peak_memory_vs_seq.png",
        "All workloads: backward local peak",
        "Backward local peak allocation (GiB)",
        "backward_local_peak_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_peak_hbm_vs_seq.png",
        "All workloads: whole-step peak HBM",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_timing_vs_seq.png",
        "All workloads: step time",
        "Step time (ms)",
        "step_ms",
    )


def write_combined_threshold_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_forward_end_memory_vs_expert_threshold.png",
        "All workloads: memory after forward vs expert threshold",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_forward_peak_memory_vs_expert_threshold.png",
        "All workloads: forward local peak vs expert threshold",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_backward_start_memory_vs_expert_threshold.png",
        "All workloads: memory carried into backward vs expert threshold",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_peak_hbm_vs_expert_threshold.png",
        "All workloads: whole-step peak HBM vs expert threshold",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_timing_vs_expert_threshold.png",
        "All workloads: step time vs expert threshold",
        "Step time (ms)",
        "step_ms",
    )


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    if not rows:
        raise SystemExit(f"no driver result directories found under {resolve_path(args.input_root)}")

    root = output_root(args)
    if args.clean_output:
        clean_output_dir(root)
    write_table(rows, root, "activation_recompute_sweep_index")

    seq_rows = [row for row in rows if int(row["expert_recompute_threshold"]) == 0]
    if seq_rows:
        combined_dir = root / "_combined"
        write_combined_plots(seq_rows, combined_dir)

    groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for row in seq_rows:
        groups.setdefault(group_key(row), []).append(row)
    for key, group_rows in sorted(groups.items()):
        workload, precision, batch_size, backend, profiler = key
        group_dir = root / safe_label(f"{workload}-b{batch_size}-{precision}-{backend}-{profiler}")
        write_table(group_rows, group_dir, "sweep_summary")
        write_group_plots(group_rows, group_dir, key)
        print(f"wrote {group_dir}", flush=True)

    threshold_rows = threshold_sweep_rows(rows)
    if threshold_rows:
        combined_dir = root / "_combined"
        write_combined_threshold_plots(threshold_rows, combined_dir)
        threshold_groups: dict[tuple[tuple[str, str, int, str, str], int], list[dict[str, Any]]] = {}
        for row in threshold_rows:
            threshold_groups.setdefault((group_key(row), int(row["seq_len"])), []).append(row)
        for (key, seq_len), group_rows in sorted(threshold_groups.items()):
            workload, precision, batch_size, backend, profiler = key
            group_dir = root / safe_label(f"{workload}-b{batch_size}-{precision}-{backend}-{profiler}")
            write_table(group_rows, group_dir, f"expert_threshold_summary_s{seq_len}")
            write_group_threshold_plots(group_rows, group_dir, key, seq_len)
            print(f"wrote {group_dir} expert threshold s{seq_len}", flush=True)
    print(f"wrote {root / 'activation_recompute_sweep_index.json'}", flush=True)


if __name__ == "__main__":
    main()
