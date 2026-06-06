#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GIB = 1024.0**3
CATEGORY_ORDER = [
    "weights",
    "gradients",
    "optimizer",
    "activations",
    "temp_workspace",
    "persistent",
]
CATEGORY_LABELS = {
    "weights": "Weights",
    "gradients": "Gradients",
    "optimizer": "Optimizer",
    "activations": "Activations",
    "temp_workspace": "Temp/workspace",
    "persistent": "Other persistent",
}
COLORS = {
    "weights": "#4c78a8",
    "gradients": "#f58518",
    "optimizer": "#54a24b",
    "activations": "#e45756",
    "temp_workspace": "#72b7b2",
    "persistent": "#b279a2",
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


def _infer_metadata(run_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    source_profile = _safe_read_json(run_dir / "source_profile.json")
    config = source_profile.get("config", {}) if isinstance(source_profile.get("config"), dict) else {}
    job_root = run_dir.parent
    config_root = job_root.parent
    job_parts = job_root.name.split("__")

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__")[0]),
        "backend": str(config.get("backend") or (job_parts[0] if len(job_parts) > 0 else "")),
        "profiler": str(job_parts[1] if len(job_parts) > 1 else "source"),
        "recompute": str(job_parts[2] if len(job_parts) > 2 else ""),
        "expert_policy": str(config.get("expert_policy") or ""),
        "seq_len": str(config.get("seq_len") or ""),
        "config": config_root.name,
    }
    for part in job_parts:
        if part.startswith("pol") and not metadata["expert_policy"]:
            metadata["expert_policy"] = part[len("pol") :]
    if not metadata["seq_len"] and run_dir.name.startswith("s") and run_dir.name[1:].isdigit():
        metadata["seq_len"] = run_dir.name[1:]
    if not metadata["profiler"]:
        metadata["profiler"] = "source"
    if not metadata["expert_policy"]:
        metadata["expert_policy"] = "none"
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
        run_dir = summary_path.parent
        jsonl_path = _first_existing(
            [
                run_dir / "memory_breakdown.jsonl",
                run_dir / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
            ]
        )
        metadata = _infer_metadata(run_dir, summary)
        record = RunRecord(run_dir=run_dir, summary_path=summary_path, jsonl_path=jsonl_path, summary=summary, metadata=metadata)
        if not _matches_filters(record, args):
            continue
        runs.append(record)
    return sorted(runs, key=lambda run: (run.metadata.get("workload", ""), run.metadata.get("seq_len", ""), run.label, str(run.run_dir)))


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


def _plot_category(row: dict[str, Any]) -> str:
    group = str(row.get("group", "persistent"))
    if group in CATEGORY_ORDER:
        return group
    return "persistent"


def _aggregate_summary(summary: dict[str, Any]) -> dict[str, int]:
    values = {category: 0 for category in CATEGORY_ORDER}
    rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return values
    for row in rows:
        if not isinstance(row, dict) or row.get("memory_space") != "GPU HBM":
            continue
        category = _plot_category(row)
        values[category] += int(row.get("bytes", 0) or 0)
    return values


def _flatten_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    peak = int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)
    persistent = row.get("persistent_bytes", {})
    activation = row.get("activation_bytes", {})
    closure = row.get("closure_bytes", {})
    rows: list[dict[str, Any]] = []

    def add(memory_space: str, group: str, component: str, kind: str, value: int) -> None:
        if value > 0:
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

    known = sum(int(item["bytes"]) for item in rows if item["memory_space"] == "GPU HBM")
    activation_items = [
        (str(component), int(value or 0))
        for component, value in (activation.items() if isinstance(activation, dict) else [])
        if int(value or 0) > 0
    ]
    activation_total = sum(value for _component, value in activation_items)
    activation_scale = 1.0
    if activation_total > 0:
        available = max(0, peak - known)
        activation_scale = min(1.0, float(available) / float(activation_total))
    for component, value in activation_items:
        add("GPU HBM", "activations", component, "activation", int(round(value * activation_scale)))
    known = sum(int(item["bytes"]) for item in rows if item["memory_space"] == "GPU HBM")
    framework = max(0, peak - known)
    if isinstance(closure, dict):
        framework = min(max(framework, int(closure.get("framework_temp_workspace") or 0)), max(0, peak - known))
    add("GPU HBM", "temp_workspace", "framework_temp_workspace", "temp_workspace", framework)
    return rows, peak


def _step_series(run: RunRecord) -> tuple[list[int], dict[str, list[int]]]:
    rows = _load_jsonl(run.jsonl_path)
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("is_warmup"):
            continue
        try:
            step = int(row.get("step", 0))
        except (TypeError, ValueError):
            continue
        current = selected.get(step)
        peak = int(row.get("peak_allocated_since_step_begin") or 0)
        if current is None or peak > int(current.get("peak_allocated_since_step_begin") or 0):
            selected[step] = row
    if not selected:
        step = int(run.summary.get("selected_step", 1) or 1)
        return [step], {category: [value] for category, value in _aggregate_summary(run.summary).items()}

    steps = sorted(selected)
    series = {category: [] for category in CATEGORY_ORDER}
    for step in steps:
        flat, _peak = _flatten_row(selected[step])
        values = {category: 0 for category in CATEGORY_ORDER}
        for row in flat:
            if row.get("memory_space") == "GPU HBM":
                values[_plot_category(row)] += int(row.get("bytes", 0) or 0)
        for category in CATEGORY_ORDER:
            series[category].append(values[category])
    return steps, series


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
    peak = int(run.summary.get("peak_hbm_bytes", 0) or 0)
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        result.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "selected_step": run.summary.get("selected_step", ""),
                "selected_phase": run.summary.get("selected_phase", ""),
                "memory_space": row.get("memory_space", "-"),
                "group": row.get("group", "-"),
                "component": row.get("component", "-"),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "gib": value / GIB,
                "percent_peak_hbm": (value * 100.0 / peak) if peak > 0 and row.get("memory_space") == "GPU HBM" else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "closure_ok": bool(run.summary.get("closure_ok", False)),
                "closure_error_bytes": int(run.summary.get("closure_error_bytes", 0) or 0),
            }
        )
    return result


def _prepare_output(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _peak_ylim_gib(runs: list[RunRecord]) -> float:
    peak = max((int(run.summary.get("peak_hbm_bytes", 0) or 0) for run in runs), default=0)
    if peak <= 0:
        return 1.0
    return max(1.0, math.ceil((peak / GIB) * 1.08))


def _plot_single_peak(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    values = _aggregate_summary(run.summary)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    bottom = 0.0
    for category in CATEGORY_ORDER:
        value = values[category] / GIB
        if value <= 0:
            continue
        ax.bar([run.metadata.get("backend", "run")], [value], bottom=bottom, label=CATEGORY_LABELS[category], color=COLORS[category])
        bottom += value
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(run.label or run.run_dir.name)
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(out_dir / "memory_peak_stack.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_stacked_bar(ax: Any, steps: list[int], series: dict[str, list[int]]) -> None:
    label = str(steps[0]) if steps else "step"
    bottom = 0.0
    for category in CATEGORY_ORDER:
        values = series.get(category, [])
        value = (values[0] / GIB) if values else 0.0
        if value <= 0.0:
            continue
        ax.bar([label], [value], bottom=bottom, width=0.55, label=CATEGORY_LABELS[category], color=COLORS[category])
        bottom += value


def _plot_single_steps(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    steps, series = _step_series(run)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    if len(steps) == 1:
        _plot_step_stacked_bar(ax, steps, series)
    else:
        stacks = [[value / GIB for value in series[category]] for category in CATEGORY_ORDER]
        ax.stackplot(steps, stacks, labels=[CATEGORY_LABELS[category] for category in CATEGORY_ORDER], colors=[COLORS[category] for category in CATEGORY_ORDER])
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
    for category in CATEGORY_ORDER:
        values = [_aggregate_summary(run.summary)[category] / GIB for run in runs]
        ax.bar(x_positions, values, bottom=bottoms, label=CATEGORY_LABELS[category], color=COLORS[category])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(0, y_limit_gib)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("LF Source Memory Peak Breakdown")
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
        steps, series = _step_series(run)
        if len(steps) == 1:
            _plot_step_stacked_bar(ax, steps, series)
        else:
            stacks = [[value / GIB for value in series[category]] for category in CATEGORY_ORDER]
            ax.stackplot(steps, stacks, labels=[CATEGORY_LABELS[category] for category in CATEGORY_ORDER], colors=[COLORS[category] for category in CATEGORY_ORDER])
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
                "peak_hbm_bytes": int(run.summary.get("peak_hbm_bytes", 0) or 0),
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
        raise SystemExit("no source memory_breakdown_summary.json files matched the requested filters")

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
