#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GIB = 1024.0**3
COMPONENT_ORDER = [
    "attention",
    "router",
    "shared_experts",
    "routed_experts",
    "mlp_dense",
    "lora",
    "embedding",
    "lm_head",
    "norms",
    "loss",
    "other",
    "unknown_saved_activation",
]
GROUP_ORDER = ["weights", "gradients", "optimizer", "saved_activations", "persistent"]
COMPONENT_LABELS = {
    "attention": "Attention",
    "router": "Router",
    "shared_experts": "Shared experts",
    "routed_experts": "Routed experts",
    "mlp_dense": "Dense MLP",
    "lora": "LoRA",
    "embedding": "Embedding",
    "lm_head": "LM head",
    "norms": "Norms",
    "loss": "Loss",
    "other": "Other",
    "unknown_saved_activation": "Unknown saved activations",
}
GROUP_LABELS = {
    "weights": "weights",
    "gradients": "gradients",
    "optimizer": "optimizer",
    "saved_activations": "saved activations",
    "persistent": "persistent",
}
SPECIAL_SEGMENT_LABELS = {
    "unattributed_allocated_peak": "Unattributed allocated peak",
    "allocator_reserved_unallocated": "Reserved but unallocated",
}
SPECIAL_SEGMENT_COLORS = {
    "unattributed_allocated_peak": "#72b7b2",
    "allocator_reserved_unallocated": "#9d9da1",
    "external_cuda_or_driver": "#666666",
}
GROUP_BASE_COLORS = {
    "weights": "#4c78a8",
    "gradients": "#f58518",
    "optimizer": "#54a24b",
    "saved_activations": "#e45756",
    "persistent": "#b279a2",
}
SEGMENT_PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#8cd17d",
    "#b6992d",
    "#499894",
    "#86bcb6",
    "#fabfd2",
    "#d37295",
    "#a0cbe8",
]
RUN_DIR_RE = re.compile(r"^(?:b(?P<batch_size>[0-9]+)_)?s(?P<seq_len>[0-9]+)$")
PHASE_PRIORITY = {
    "after_backward": 60,
    "before_optimizer_step": 50,
    "after_forward": 40,
    "after_optimizer_step": 30,
    "step_begin": 0,
}


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    summary_path: Path
    jsonl_path: Path | None
    summary: dict[str, Any]
    metadata: dict[str, str]

    @property
    def label(self) -> str:
        parts = [
            self.metadata.get("workload", ""),
            self.metadata.get("backend", ""),
            f"router={self.metadata.get('router_mode', '')}" if self.metadata.get("router_mode") else "",
            self.metadata.get("recompute", ""),
            self.metadata.get("expert_policy", ""),
            self.metadata.get("seq_len", ""),
        ]
        return " ".join(part for part in parts if part)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LF source memory-breakdown artifacts.")
    parser.add_argument("--input-root", type=Path, action="append", default=[], help="Root to scan for memory_breakdown_summary.json files.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit source run directory.")
    parser.add_argument("--output-dir", type=Path, help="Directory for combined plots, or the per-run output dir when a single --run-dir is given.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--include-non-source", action="store_true", help="Include runs whose path does not contain __source__.")
    parser.add_argument("--y-scale", choices=("shared", "per-plot", "global"), default="shared")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--profiler", action="append", default=[])
    parser.add_argument("--router-mode", action="append", default=[], choices=["hf", "whole"])
    parser.add_argument("--seq-lens", nargs="+", default=[])
    parser.add_argument("--expert-recompute-policies", nargs="+", default=[])
    return parser.parse_args()


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _seq_len_from_run_dir_name(name: str) -> str:
    match = RUN_DIR_RE.match(name)
    return match.group("seq_len") if match is not None else ""


def _infer_metadata(run_dir: Path, summary: dict[str, Any]) -> dict[str, str] | None:
    source_profile = _safe_read_json(run_dir / "source_profile.json")
    config = source_profile.get("config", {}) if isinstance(source_profile.get("config"), dict) else {}
    job_root = run_dir.parent
    config_root = job_root.parent
    job_parts = job_root.name.split("__")
    if len(job_parts) != 5:
        return None
    backend_part, profiler_part, recompute_part, policy_part, router_part = job_parts
    if not policy_part.startswith("pol") or not router_part.startswith("router"):
        return None
    router_mode = str(config.get("router_mode") or router_part[len("router") :])
    if router_mode not in {"hf", "whole"}:
        return None
    expert_policy = str(config.get("expert_policy") or policy_part[len("pol") :] or "none")

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__")[0]),
        "backend": str(config.get("backend") or backend_part),
        "profiler": str(profiler_part),
        "recompute": str(recompute_part),
        "expert_policy": expert_policy,
        "router_mode": router_mode,
        "seq_len": str(config.get("seq_len") or ""),
        "config": config_root.name,
    }
    if not metadata["seq_len"]:
        metadata["seq_len"] = _seq_len_from_run_dir_name(run_dir.name)
    if not metadata["profiler"]:
        metadata["profiler"] = "source"
    return metadata


def _find_summary_paths(input_roots: list[Path], run_dirs: list[Path], include_non_source: bool) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        candidates = [
            run_dir / "memory_breakdown_summary.json",
            run_dir / "memory_breakdown" / "memory_breakdown_summary.json",
        ]
        found = _first_existing(candidates)
        if found is not None:
            paths.append(found)
    for root in input_roots:
        if root.is_file() and root.name == "memory_breakdown_summary.json":
            paths.append(root)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("memory_breakdown_summary.json")):
            if include_non_source or "__source__" in str(path.parent.parent.name) or "__source__" in str(path):
                paths.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _load_runs(args: argparse.Namespace) -> list[RunRecord]:
    paths = _find_summary_paths(args.input_root, args.run_dir, args.include_non_source)
    runs: list[RunRecord] = []
    for summary_path in paths:
        summary = _safe_read_json(summary_path)
        if not summary.get("enabled", bool(summary.get("breakdown_rows"))):
            continue
        if int(summary.get("schema_version") or 0) != 2:
            continue
        run_dir = summary_path.parent
        jsonl_path = _first_existing(
            [
                run_dir / "memory_breakdown.jsonl",
                run_dir / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
            ]
        )
        metadata = _infer_metadata(run_dir, summary)
        if metadata is None:
            continue
        record = RunRecord(run_dir=run_dir, summary_path=summary_path, jsonl_path=jsonl_path, summary=summary, metadata=metadata)
        if not _matches_filters(record, args):
            continue
        runs.append(record)
    return sorted(runs, key=lambda run: (run.metadata.get("workload", ""), run.metadata.get("seq_len", ""), run.label, str(run.run_dir)))


def _no_runs_message(args: argparse.Namespace) -> str:
    paths = _find_summary_paths(args.input_root, args.run_dir, args.include_non_source)
    if not paths:
        return "no source memory_breakdown_summary.json files matched the requested filters"
    legacy_paths: list[Path] = []
    disabled_paths: list[Path] = []
    metadata_failures = 0
    filter_failures = 0
    for summary_path in paths:
        summary = _safe_read_json(summary_path)
        if not summary.get("enabled", bool(summary.get("breakdown_rows"))):
            disabled_paths.append(summary_path)
            continue
        if int(summary.get("schema_version") or 0) != 2:
            legacy_paths.append(summary_path)
            continue
        run_dir = summary_path.parent
        metadata = _infer_metadata(run_dir, summary)
        if metadata is None:
            metadata_failures += 1
            continue
        jsonl_path = _first_existing(
            [
                run_dir / "memory_breakdown.jsonl",
                run_dir / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
            ]
        )
        record = RunRecord(
            run_dir=run_dir,
            summary_path=summary_path,
            jsonl_path=jsonl_path,
            summary=summary,
            metadata=metadata,
        )
        if not _matches_filters(record, args):
            filter_failures += 1
    details: list[str] = []
    if legacy_paths:
        details.append(
            f"found {len(legacy_paths)} legacy/non-v2 source memory breakdown summary file(s); "
            "rerun the source profiler so memory_breakdown_summary.json is schema_version 2"
        )
    if disabled_paths:
        details.append(f"found {len(disabled_paths)} disabled/empty summary file(s)")
    if metadata_failures:
        details.append(f"{metadata_failures} schema-v2 summary file(s) had unparseable run metadata")
    if filter_failures:
        details.append(f"{filter_failures} schema-v2 summary file(s) were excluded by filters")
    if details:
        return "no schema-v2 source memory breakdown summaries matched; " + "; ".join(details)
    return "no source memory_breakdown_summary.json files matched the requested filters"


def _filter_values(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                result.add(part.lower())
    return result


def _matches_filters(run: RunRecord, args: argparse.Namespace) -> bool:
    filters = {
        "workload": _filter_values(args.workload),
        "backend": _filter_values(args.backend),
        "profiler": _filter_values(args.profiler),
        "router_mode": _filter_values(args.router_mode),
        "seq_len": _filter_values(args.seq_lens),
        "expert_policy": _filter_values(args.expert_recompute_policies),
    }
    for key, allowed in filters.items():
        if not allowed:
            continue
        value = run.metadata.get(key, "").lower()
        if key == "workload":
            config_workload = run.metadata.get("config", "").split("__", 1)[0].lower()
            if value in allowed or config_workload in allowed:
                continue
        if value not in allowed:
            return False
    return True


def _selection_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0),
        PHASE_PRIORITY.get(str(row.get("phase") or ""), 10),
        int(row.get("allocated_bytes") or 0),
        int(row.get("step") or 0),
    )


def _segment_key(row: dict[str, Any]) -> str | None:
    component = str(row.get("component") or "")
    group = str(row.get("group") or "persistent")
    memory_space = str(row.get("memory_space") or "")
    if component == "allocator_reserved_unallocated":
        return "allocator_reserved_unallocated"
    if component == "external_cuda_or_driver":
        return "external_cuda_or_driver"
    if memory_space != "GPU HBM":
        return None
    if group == "unattributed_allocated_peak" or component == "unattributed_allocated_peak":
        return "unattributed_allocated_peak"
    if group not in GROUP_ORDER:
        group = "persistent"
    return f"{component or 'other'}:{group}"


def _component_sort_index(component: str) -> int:
    try:
        return COMPONENT_ORDER.index(component)
    except ValueError:
        return len(COMPONENT_ORDER)


def _group_sort_index(group: str) -> int:
    try:
        return GROUP_ORDER.index(group)
    except ValueError:
        return len(GROUP_ORDER)


def _segment_sort_key(key: str) -> tuple[int, int, str]:
    if key == "unattributed_allocated_peak":
        return (10_000, 0, key)
    if key == "allocator_reserved_unallocated":
        return (10_001, 0, key)
    if key == "external_cuda_or_driver":
        return (10_002, 0, key)
    component, _, group = key.partition(":")
    return (_component_sort_index(component), _group_sort_index(group), key)


def _segment_label(key: str) -> str:
    if key in SPECIAL_SEGMENT_LABELS:
        return SPECIAL_SEGMENT_LABELS[key]
    if key == "external_cuda_or_driver":
        return "External CUDA/driver"
    component, _, group = key.partition(":")
    component_label = COMPONENT_LABELS.get(component, component.replace("_", " "))
    group_label = GROUP_LABELS.get(group, group.replace("_", " "))
    return f"{component_label} {group_label}"


def _segment_color(key: str) -> str:
    if key in SPECIAL_SEGMENT_COLORS:
        return SPECIAL_SEGMENT_COLORS[key]
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return SEGMENT_PALETTE[int(digest[:8], 16) % len(SEGMENT_PALETTE)]


def _aggregate_rows(rows: list[dict[str, Any]], *, include_external: bool = False) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _segment_key(row)
        if key is None:
            continue
        if key == "external_cuda_or_driver" and not include_external:
            continue
        values[key] = values.get(key, 0) + int(row.get("bytes", 0) or 0)
    return values


def _aggregate_summary(summary: dict[str, Any]) -> dict[str, int]:
    rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return {}
    return _aggregate_rows(rows)


def _flatten_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    peak_allocated = int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)
    peak_reserved = int(row.get("peak_reserved_since_step_begin") or row.get("reserved_bytes") or 0)
    peak_reserved = max(peak_reserved, peak_allocated)
    persistent = row.get("persistent_bytes", {})
    saved_activation = row.get("saved_activation_bytes_at_peak", {})
    if not isinstance(saved_activation, dict) or not saved_activation:
        saved_activation = row.get("saved_activation_bytes", {})
    closure = row.get("closure_bytes", {})
    external_memory = row.get("external_memory", {})
    rows: list[dict[str, Any]] = []

    def add(memory_space: str, group: str, component: str, kind: str, value: int, *, keep_zero: bool = False) -> None:
        if value > 0 or (value == 0 and keep_zero):
            rows.append({"memory_space": memory_space, "group": group, "component": component, "kind": kind, "bytes": int(value)})

    if isinstance(persistent, dict):
        for component, kinds in persistent.items():
            if not isinstance(kinds, dict):
                continue
            for kind, value in kinds.items():
                value_int = int(value or 0)
                if value_int <= 0:
                    continue
                kind_str = str(kind)
                if kind_str.endswith("_cpu") or kind_str.endswith("_cpu_pinned"):
                    add("CPU host", "host", str(component), kind_str, value_int)
                elif kind_str in {"weight", "frozen_weight", "buffer"}:
                    add("GPU HBM", "weights", str(component), kind_str, value_int)
                elif kind_str == "grad":
                    add("GPU HBM", "gradients", str(component), kind_str, value_int)
                elif kind_str == "optimizer_state":
                    add("GPU HBM", "optimizer", str(component), kind_str, value_int)
                else:
                    add("GPU HBM", "persistent", str(component), kind_str, value_int)

    saved_activation_items = [
        (str(component), int(value or 0))
        for component, value in (saved_activation.items() if isinstance(saved_activation, dict) else [])
        if int(value or 0) > 0
    ]
    for component, value in saved_activation_items:
        add("GPU HBM", "saved_activations", component, "saved_activation", value)
    known = sum(int(item["bytes"]) for item in rows if item["memory_space"] == "GPU HBM")
    unattributed = max(0, peak_allocated - known)
    if isinstance(closure, dict):
        unattributed = min(
            max(unattributed, int(closure.get("unattributed_allocated_peak") or 0)),
            max(0, peak_allocated - known),
        )
    add("GPU HBM", "unattributed_allocated_peak", "unattributed_allocated_peak", "allocated_residual", unattributed)
    add(
        "GPU reserved",
        "allocator",
        "allocator_reserved_unallocated",
        "reserved_unallocated",
        max(0, peak_reserved - peak_allocated),
        keep_zero=True,
    )
    external_value = 0
    if isinstance(external_memory, dict):
        external_value = int(external_memory.get("external_cuda_or_driver_bytes") or 0)
    if isinstance(closure, dict):
        external_value = max(external_value, int(closure.get("external_cuda_or_driver") or 0))
    add("External CUDA", "external", "external_cuda_or_driver", "process_or_driver_gap", external_value)
    return rows, peak_allocated, peak_reserved


def _step_series(run: RunRecord) -> tuple[list[int], dict[str, list[int]], list[int]]:
    rows = _load_jsonl(run.jsonl_path)
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("is_warmup") or int(row.get("schema_version") or 0) != 2:
            continue
        try:
            step = int(row.get("step", 0))
        except (TypeError, ValueError):
            continue
        current = selected.get(step)
        if current is None or _selection_key(row) > _selection_key(current):
            selected[step] = row
    if not selected:
        step = int(run.summary.get("selected_step", 1) or 1)
        return (
            [step],
            {key: [value] for key, value in _aggregate_summary(run.summary).items()},
            [int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0)],
        )

    steps = sorted(selected)
    per_step_values: list[dict[str, int]] = []
    peak_allocated_values: list[int] = []
    keys: set[str] = set()
    for step in steps:
        flat, peak_allocated, _peak_reserved = _flatten_row(selected[step])
        values = _aggregate_rows(flat)
        per_step_values.append(values)
        peak_allocated_values.append(peak_allocated)
        keys.update(values)
    ordered_keys = sorted(keys, key=_segment_sort_key)
    series = {key: [] for key in ordered_keys}
    for values in per_step_values:
        for key in ordered_keys:
            series[key].append(values.get(key, 0))
    return steps, series, peak_allocated_values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _summary_csv_rows(run: RunRecord) -> list[dict[str, Any]]:
    rows = run.summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return []
    peak_allocated = int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0)
    peak_reserved = int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0)
    reserved_unallocated = int(run.summary.get("reserved_unallocated_bytes", 0) or 0)
    external_cuda = int(run.summary.get("external_cuda_or_driver_bytes", 0) or 0)
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        is_reserved_stack_row = row.get("memory_space") == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
        result.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "schema_version": run.summary.get("schema_version", ""),
                "selected_metric": run.summary.get("selected_metric", ""),
                "selected_step": run.summary.get("selected_step", ""),
                "selected_phase": run.summary.get("selected_phase", ""),
                "peak_allocated_hbm_bytes": peak_allocated,
                "peak_reserved_hbm_bytes": peak_reserved,
                "reserved_unallocated_bytes": reserved_unallocated,
                "external_cuda_or_driver_bytes": external_cuda,
                "allocated_stack_sum_bytes": int(run.summary.get("allocated_stack_sum_bytes", 0) or 0),
                "reserved_stack_sum_bytes": int(run.summary.get("reserved_stack_sum_bytes", 0) or 0),
                "saved_activation_hbm_bytes_at_peak": int(
                    run.summary.get("saved_activation_hbm_bytes_at_peak", 0) or 0
                ),
                "unattributed_allocated_peak_bytes": int(
                    run.summary.get("unattributed_allocated_peak_bytes", 0) or 0
                ),
                "memory_space": row.get("memory_space", "-"),
                "group": row.get("group", "-"),
                "component": row.get("component", "-"),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "gib": value / GIB,
                "percent_peak_reserved_hbm": (value * 100.0 / peak_reserved)
                if peak_reserved > 0 and is_reserved_stack_row
                else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "allocated_closure_ok": bool(run.summary.get("allocated_closure_ok", False)),
                "reserved_closure_ok": bool(run.summary.get("reserved_closure_ok", False)),
                "allocated_closure_error_bytes": int(run.summary.get("allocated_closure_error_bytes", 0) or 0),
                "reserved_closure_error_bytes": int(run.summary.get("reserved_closure_error_bytes", 0) or 0),
            }
        )
    return result


def _prepare_output(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _peak_ylim_gib(runs: list[RunRecord]) -> float:
    peak = max((int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0) for run in runs), default=0)
    if peak <= 0:
        return 1.0
    return max(1.0, math.ceil((peak / GIB) * 1.08))


def _plot_single_peak(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    values = _aggregate_summary(run.summary)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    bottom = 0.0
    x_label = run.metadata.get("backend", "run")
    for key in sorted(values, key=_segment_sort_key):
        value = values[key] / GIB
        if value <= 0:
            continue
        ax.bar([x_label], [value], bottom=bottom, label=_segment_label(key), color=_segment_color(key))
        bottom += value
    peak_allocated = int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0) / GIB
    if peak_allocated > 0:
        ax.axhline(peak_allocated, color="#222222", linestyle="--", linewidth=1.0, label="Peak allocated")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(run.label or run.run_dir.name)
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(out_dir / "memory_peak_stack.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_stacked_bar(ax: Any, steps: list[int], series: dict[str, list[int]], peak_allocated: list[int]) -> None:
    label = str(steps[0]) if steps else "step"
    bottom = 0.0
    for key in sorted(series, key=_segment_sort_key):
        values = series.get(key, [])
        value = (values[0] / GIB) if values else 0.0
        if value <= 0.0:
            continue
        ax.bar([label], [value], bottom=bottom, width=0.55, label=_segment_label(key), color=_segment_color(key))
        bottom += value
    if peak_allocated:
        ax.axhline(peak_allocated[0] / GIB, color="#222222", linestyle="--", linewidth=1.0, label="Peak allocated")


def _plot_single_steps(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    steps, series, peak_allocated = _step_series(run)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    if len(steps) == 1:
        _plot_step_stacked_bar(ax, steps, series, peak_allocated)
    else:
        keys = sorted(series, key=_segment_sort_key)
        stacks = [[value / GIB for value in series[key]] for key in keys]
        ax.stackplot(steps, stacks, labels=[_segment_label(key) for key in keys], colors=[_segment_color(key) for key in keys])
        if peak_allocated:
            ax.plot(steps, [value / GIB for value in peak_allocated], color="#222222", linestyle="--", linewidth=1.0, label="Peak allocated")
    ax.set_xlabel("Measured step")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(run.label or run.run_dir.name)
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(out_dir / "memory_over_steps_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_per_run(run: RunRecord, out_dir: Path, clean: bool, y_limit_gib: float | None) -> None:
    _prepare_output(out_dir, clean)
    _write_csv(out_dir / "memory_breakdown.csv", _summary_csv_rows(run))
    (out_dir / "memory_breakdown_index.json").write_text(
        json.dumps({"run_dir": str(run.run_dir), "summary_path": str(run.summary_path), "jsonl_path": str(run.jsonl_path or "")}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_single_peak(run, out_dir, y_limit_gib)
    _plot_single_steps(run, out_dir, y_limit_gib)


def _plot_combined_peak(runs: list[RunRecord], out_dir: Path, y_limit_gib: float) -> None:
    labels = [run.label or run.run_dir.name for run in runs]
    x_positions = list(range(len(runs)))
    fig_width = max(9.0, min(28.0, 1.2 * len(runs) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0), constrained_layout=True)
    bottoms = [0.0 for _run in runs]
    keys = sorted({key for run in runs for key in _aggregate_summary(run.summary)}, key=_segment_sort_key)
    for key in keys:
        values = [_aggregate_summary(run.summary).get(key, 0) / GIB for run in runs]
        ax.bar(x_positions, values, bottom=bottoms, label=_segment_label(key), color=_segment_color(key))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    for x_position, run in zip(x_positions, runs):
        peak_allocated = int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0) / GIB
        if peak_allocated > 0:
            ax.hlines(peak_allocated, x_position - 0.35, x_position + 0.35, colors="#222222", linestyles="--", linewidth=1.0)
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(0, y_limit_gib)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("LF Source Reserved-Capacity Memory Breakdown")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(out_dir / "combined_memory_peak_stack.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_steps(runs: list[RunRecord], out_dir: Path, y_limit_gib: float) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 3 else 1
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 3.2 * nrows), sharey=True, squeeze=False, constrained_layout=True)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        steps, series, peak_allocated = _step_series(run)
        if len(steps) == 1:
            _plot_step_stacked_bar(ax, steps, series, peak_allocated)
        else:
            keys = sorted(series, key=_segment_sort_key)
            stacks = [[value / GIB for value in series[key]] for key in keys]
            ax.stackplot(steps, stacks, labels=[_segment_label(key) for key in keys], colors=[_segment_color(key) for key in keys])
            if peak_allocated:
                ax.plot(steps, [value / GIB for value in peak_allocated], color="#222222", linestyle="--", linewidth=1.0, label="Peak allocated")
        ax.set_title(run.label or run.run_dir.name, fontsize=9)
        ax.set_xlabel("Measured step")
        ax.set_ylim(0, y_limit_gib)
        if idx % ncols == 0:
            ax.set_ylabel("Memory (GiB)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(out_dir / "combined_memory_over_steps_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_combined(runs: list[RunRecord], out_dir: Path, clean: bool, y_limit_gib: float) -> None:
    _prepare_output(out_dir, clean)
    all_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for run in runs:
        all_rows.extend(_summary_csv_rows(run))
        index_rows.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "summary_path": str(run.summary_path),
                "jsonl_path": str(run.jsonl_path or ""),
                "schema_version": run.summary.get("schema_version", ""),
                "selected_metric": run.summary.get("selected_metric", ""),
                "peak_allocated_hbm_bytes": int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0),
                "peak_reserved_hbm_bytes": int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0),
                "reserved_unallocated_bytes": int(run.summary.get("reserved_unallocated_bytes", 0) or 0),
                "external_cuda_or_driver_bytes": int(run.summary.get("external_cuda_or_driver_bytes", 0) or 0),
                "allocated_stack_sum_bytes": int(run.summary.get("allocated_stack_sum_bytes", 0) or 0),
                "reserved_stack_sum_bytes": int(run.summary.get("reserved_stack_sum_bytes", 0) or 0),
            }
        )
    _write_csv(out_dir / "combined_memory_breakdown.csv", all_rows)
    _write_csv(out_dir / "memory_breakdown_index.csv", index_rows)
    (out_dir / "memory_breakdown_index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_combined_peak(runs, out_dir, y_limit_gib)
    _plot_combined_steps(runs, out_dir, y_limit_gib)


def main() -> None:
    args = _parse_args()
    runs = _load_runs(args)
    if not runs:
        raise SystemExit(_no_runs_message(args))

    shared_ylim = _peak_ylim_gib(runs)
    single_run_output = bool(args.output_dir and len(runs) == 1 and args.run_dir and not args.input_root and not args.combined_only)
    if not args.combined_only:
        for run in runs:
            out_dir = args.output_dir if single_run_output else run.run_dir / "memory_plots"
            y_limit = None if args.y_scale == "per-plot" else shared_ylim
            _write_per_run(run, out_dir, args.clean_output, y_limit)

    if args.output_dir and (args.combined_only or len(runs) > 1 or args.input_root):
        _write_combined(runs, args.output_dir, args.clean_output, shared_ylim)

    print(f"Plotted {len(runs)} source memory run(s).")


if __name__ == "__main__":
    main()
