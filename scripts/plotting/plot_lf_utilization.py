#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plotting.plot_lf_interconnect_ctc import (
    _apply_workload_filters,
    _find_profile_paths,
    _infer_metadata,
    _matches_filters,
    _parse_job_dir_parts,
    _safe_label,
    _safe_read_json,
)


SUMMARY_FIELDS = [
    "workload",
    "backend",
    "router_mode",
    "asymm_expert_act_offload",
    "expact",
    "asymm_attn_act_offload",
    "attnact",
    "asymm_layer_act_offload",
    "layeract",
    "asymm_layer_gc",
    "layergc",
    "liger_loss",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_dropout",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "config",
    "run_label",
    "gpu_samples",
    "mean_gpu_util_percent",
    "p95_gpu_util_percent",
    "max_gpu_util_percent",
    "cpu_samples",
    "mean_cpu_util_percent",
    "p95_cpu_util_percent",
    "max_cpu_util_percent",
    "utilization_plot",
    "profile_json",
    "run_dir",
]
INDEX_FIELDS = [
    "workload",
    "backend",
    "router_mode",
    "asymm_expert_act_offload",
    "expact",
    "asymm_attn_act_offload",
    "attnact",
    "asymm_layer_act_offload",
    "layeract",
    "asymm_layer_gc",
    "layergc",
    "liger_loss",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_dropout",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "config",
    "run_label",
    "profile_json",
    "run_dir",
]
TIMELINE_BIN_SECONDS = 0.25


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    profile_path: Path
    metadata: dict[str, str]
    summary: dict[str, Any]
    series: dict[str, list[tuple[float, float]]]

    @property
    def label(self) -> str:
        parts = [
            self.metadata.get("workload", ""),
            self.metadata.get("backend", ""),
            f"router={self.metadata.get('router_mode', '')}" if self.metadata.get("router_mode") else "",
            self.metadata.get("expact", ""),
            self.metadata.get("attnact", ""),
            self.metadata.get("layeract", ""),
            self.metadata.get("layergc", ""),
            self.metadata.get("liger_loss", ""),
            self.metadata.get("recompute", ""),
            self.metadata.get("expert_policy", ""),
            f"s{self.metadata.get('seq_len', '')}" if self.metadata.get("seq_len") else "",
        ]
        return " ".join(part for part in parts if part)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined LF GPU/CPU utilization artifacts.")
    parser.add_argument("--input-root", type=Path, action="append", default=[], help="Root to scan for profile.json files.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit run directory containing profile.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined utilization plots.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--combined-only", action="store_true", help="Accepted for parity with other LF plotting scripts.")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--profiler", action="append", default=[])
    parser.add_argument("--router-mode", action="append", default=[])
    parser.add_argument("--expact", action="append", default=[])
    parser.add_argument("--attnact", action="append", default=[])
    parser.add_argument("--layeract", action="append", default=[])
    parser.add_argument("--layergc", action="append", default=[])
    parser.add_argument("--liger-loss", action="append", default=[])
    parser.add_argument("--recompute", action="append", default=[])
    parser.add_argument("--workloads", nargs="+", default=[])
    parser.add_argument("--expert-recompute-policies", nargs="+", default=[])
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _device_rows(profile: dict[str, Any], device: str) -> list[dict[str, Any]]:
    metrics = profile.get("utilization_metrics")
    if not isinstance(metrics, dict) or not metrics.get("available"):
        return []
    rows = metrics.get("summary", {}).get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and str(row.get("device")) == device]


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "mean_percent": None, "p95_percent": None, "max_percent": None}
    sample_count = 0
    mean_values: list[float] = []
    p95_values: list[float] = []
    max_values: list[float] = []
    for row in rows:
        try:
            sample_count += int(row.get("samples", 0) or 0)
        except (TypeError, ValueError):
            pass
        mean = _float_or_none(row.get("mean_percent", row.get("avg_percent")))
        p95 = _float_or_none(row.get("p95_percent"))
        max_value = _float_or_none(row.get("max_percent"))
        if mean is not None:
            mean_values.append(mean)
        if p95 is not None:
            p95_values.append(p95)
        if max_value is not None:
            max_values.append(max_value)
    return {
        "samples": sample_count,
        "mean_percent": sum(mean_values) / len(mean_values) if mean_values else None,
        "p95_percent": sum(p95_values) / len(p95_values) if p95_values else None,
        "max_percent": max(max_values) if max_values else None,
    }


def _run_summary(profile: dict[str, Any]) -> dict[str, Any]:
    gpu = _aggregate_rows(_device_rows(profile, "gpu"))
    cpu = _aggregate_rows(_device_rows(profile, "cpu"))
    return {
        "gpu_samples": gpu["samples"],
        "mean_gpu_util_percent": gpu["mean_percent"],
        "p95_gpu_util_percent": gpu["p95_percent"],
        "max_gpu_util_percent": gpu["max_percent"],
        "cpu_samples": cpu["samples"],
        "mean_cpu_util_percent": cpu["mean_percent"],
        "p95_cpu_util_percent": cpu["p95_percent"],
        "max_cpu_util_percent": cpu["max_percent"],
    }


def _time_series(profile: dict[str, Any], device: str) -> list[tuple[float, float]]:
    metrics = profile.get("utilization_metrics")
    if not isinstance(metrics, dict) or not metrics.get("available"):
        return []
    rows = metrics.get("timeseries", {}).get("rows", [])
    if not isinstance(rows, list):
        return []
    points: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("device") or "") != device:
            continue
        try:
            elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
            value = float(row.get("value_percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(elapsed) or not math.isfinite(value):
            continue
        points.append((elapsed, min(100.0, max(0.0, value))))
    return sorted(points, key=lambda item: item[0])


def _measured_window(profile: dict[str, Any]) -> tuple[float, float] | None:
    source_profile = profile.get("source_profile")
    if not isinstance(source_profile, dict):
        return None
    step_samples = source_profile.get("step_samples")
    if not isinstance(step_samples, dict):
        return None
    rows = step_samples.get("rows")
    if not isinstance(rows, list):
        return None
    measured_rows = [row for row in rows if isinstance(row, dict) and not row.get("is_warmup")]
    if not measured_rows:
        return None
    start = _float_or_none(measured_rows[0].get("trainer_e2e_start_seconds"))
    end = _float_or_none(measured_rows[-1].get("trainer_e2e_end_seconds"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def _shift_series_to_window(series: list[tuple[float, float]], window: tuple[float, float] | None) -> list[tuple[float, float]]:
    if window is None:
        return series
    start, end = window
    shifted = [(elapsed - start, value) for elapsed, value in series if start <= elapsed <= end]
    return sorted(shifted, key=lambda item: item[0])


def _channel_series(
    profile: dict[str, Any],
    measured_window: tuple[float, float] | None,
) -> dict[str, list[tuple[float, float]]]:
    """One time-series per channel: one entry per GPU id (keyed "GPU <id>") plus "CPU".

    GPU points keep their Nsight capture-relative elapsed axis; CPU points are shifted to the
    post-warmup measured-step window. The two come from different clocks (Nsight capture vs the
    in-process sampler) and are not co-registered, so GPU is left on its own axis.
    """
    metrics = profile.get("utilization_metrics")
    if not isinstance(metrics, dict) or not metrics.get("available"):
        return {}
    rows = metrics.get("timeseries", {}).get("rows", [])
    if not isinstance(rows, list):
        return {}
    gpu_points: dict[int, list[tuple[float, float]]] = {}
    cpu_points: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        device = str(row.get("device") or "")
        try:
            elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
            value = float(row.get("value_percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(elapsed) or not math.isfinite(value):
            continue
        value = min(100.0, max(0.0, value))
        if device == "gpu":
            try:
                gpu_id = int(row.get("gpu_id"))
            except (TypeError, ValueError):
                gpu_id = 0
            gpu_points.setdefault(gpu_id, []).append((elapsed, value))
        elif device == "cpu":
            cpu_points.append((elapsed, value))
    channels: dict[str, list[tuple[float, float]]] = {}
    for gpu_id in sorted(gpu_points):
        channels[f"GPU {gpu_id}"] = sorted(gpu_points[gpu_id], key=lambda item: item[0])
    if cpu_points:
        channels["CPU"] = _shift_series_to_window(
            sorted(cpu_points, key=lambda item: item[0]), measured_window
        )
    return channels


def _load_runs(args: argparse.Namespace) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for profile_path in _find_profile_paths(args.input_root, args.run_dir):
        profile = _safe_read_json(profile_path)
        summary = _run_summary(profile)
        measured_window = _measured_window(profile)
        series = _channel_series(profile, measured_window)
        if not any(channel != "CPU" for channel in series):
            # Skip CPU-only/empty profiles so source-only runs don't create GPU-less panels.
            continue
        metadata = _infer_metadata(profile_path, profile)
        if metadata is None:
            continue
        record = RunRecord(
            run_dir=profile_path.parent,
            profile_path=profile_path,
            metadata=metadata,
            summary=summary,
            series=series,
        )
        if _matches_filters(record, args):
            runs.append(record)
    return sorted(
        runs,
        key=lambda run: (
            run.metadata.get("workload", ""),
            int(run.metadata.get("seq_len") or 0),
            run.metadata.get("backend", ""),
            run.metadata.get("router_mode", ""),
            run.metadata.get("liger_loss", ""),
            run.metadata.get("recompute", ""),
            run.metadata.get("expert_policy", ""),
            str(run.run_dir),
        ),
    )


def _prepare_output(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _summary_rows(runs: list[RunRecord]) -> list[dict[str, Any]]:
    return [
        {
            **run.metadata,
            "run_label": run.label,
            **run.summary,
            "utilization_plot": str(run.run_dir / "utilization.png"),
            "profile_json": str(run.profile_path),
            "run_dir": str(run.run_dir),
        }
        for run in runs
    ]


def _plot_label(run: RunRecord) -> str:
    metadata = run.metadata
    config = metadata.get("config", "")
    config_step = ""
    match = re.search(r"_w(?P<warmup>[0-9]+)_s(?P<measured>[0-9]+)", config)
    if match is not None:
        config_step = f"w{match.group('warmup')}+m{match.group('measured')}"
    parts = [
        metadata.get("workload", ""),
        metadata.get("backend", ""),
        metadata.get("recompute", ""),
        f"s{metadata.get('seq_len', '')}" if metadata.get("seq_len") else "",
        f"b{metadata.get('batch_size', '')}" if metadata.get("batch_size") else "",
        f"ga{metadata.get('gradient_accumulation_steps', '')}" if metadata.get("gradient_accumulation_steps") else "",
        config_step,
    ]
    for key in ("expact", "attnact", "layeract", "layergc"):
        value = metadata.get(key, "")
        if value and not value.endswith("0"):
            parts.append(value)
    return " ".join(part for part in parts if part)


def _bin_series(series: list[tuple[float, float]], bin_seconds: float) -> list[tuple[float, float]]:
    if bin_seconds <= 0.0 or len(series) <= 1:
        return series
    bins: dict[int, list[float]] = {}
    for elapsed, value in series:
        bucket = int(elapsed // bin_seconds)
        bins.setdefault(bucket, []).append(value)
    return [
        ((bucket + 0.5) * bin_seconds, sum(values) / len(values))
        for bucket, values in sorted(bins.items())
        if values
    ]


def _bin_label(bin_seconds: float) -> str:
    if bin_seconds < 1.0:
        return f"{bin_seconds * 1000.0:g} ms"
    return f"{bin_seconds:g} s"


# Distinct reds/oranges per GPU id; CPU is always blue. Color keyed by gpu id so a given
# physical GPU keeps the same color across subplots and the shared legend stays consistent.
_GPU_LINE_COLORS = ["#ff2d2d", "#ff8c00", "#9b1d1d", "#ff5c8a", "#d2691e", "#7a0000", "#ffae42", "#c71585"]
_CPU_LINE_COLOR = "#1f4aff"


def _gpu_id_from_channel(channel: str) -> int:
    try:
        return int(channel.split()[-1])
    except (ValueError, IndexError):
        return 0


def _channel_color(channel: str) -> str:
    if channel == "CPU":
        return _CPU_LINE_COLOR
    return _GPU_LINE_COLORS[_gpu_id_from_channel(channel) % len(_GPU_LINE_COLORS)]


def _channel_sort_key(channel: str) -> tuple[int, int]:
    # GPUs first (ordered by numeric id), CPU last.
    return (1, 0) if channel == "CPU" else (0, _gpu_id_from_channel(channel))


def _plot_timeline(runs: list[RunRecord], out_dir: Path) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 2 else 1
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0 * ncols, 3.2 * nrows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    bin_label = _bin_label(TIMELINE_BIN_SECONDS)
    legend_handles: dict[str, Any] = {}
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        # One line per GPU id, CPU last.
        for channel in sorted(run.series, key=_channel_sort_key):
            series = run.series.get(channel, [])
            if not series:
                continue
            binned = _bin_series(series, TIMELINE_BIN_SECONDS)
            xs = [point[0] for point in binned]
            ys = [point[1] for point in binned]
            (line,) = ax.plot(xs, ys, color=_channel_color(channel), linewidth=1.4, label=channel)
            legend_handles.setdefault(channel, line)
        ax.set_title(_plot_label(run) or run.run_dir.name, fontsize=9)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="both", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Utilization (%)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if legend_handles:
        labels = sorted(legend_handles, key=_channel_sort_key)
        fig.legend(
            [legend_handles[name] for name in labels],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=max(1, min(len(labels), 5)),
            frameon=False,
        )
    fig.suptitle(f"Per-GPU and CPU Utilization Timeline ({bin_label} mean)", fontsize=12)
    fig.savefig(out_dir / "combined_utilization_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, *, runs: list[RunRecord], reason: str | None = None) -> None:
    bin_label = _bin_label(TIMELINE_BIN_SECONDS)
    lines = [
        "# LF Utilization Combined Artifacts",
        "",
        "These files combine per-run GPU and CPU utilization timelines across runs.",
        f"`combined_utilization_timeline.png` renders {bin_label} mean utilization bins and stacks one subplot per run, drawing one line per physical GPU id plus a CPU line.",
        "The per-run timeline plot is `utilization.png` next to each `profile.json`.",
        "GPU values come from Nsight SM Active / SMs Active when available (one line per physical GPU); CPU values come from process CPU time normalized by available CPU affinity cores.",
        "Runs without GPU utilization samples are excluded so CPU-only source profiles do not create extra panels.",
        "GPU lines use the Nsight capture-relative time axis; CPU samples are shifted to the post-warmup measured-step window, so GPU and CPU are not co-registered in wall-clock time.",
        "",
    ]
    if reason:
        lines.extend(["## Status", "", reason, ""])
    else:
        lines.extend(
            [
                "## Files",
                "",
                "- `combined_utilization_timeline.png`: stacked per-run GPU/CPU measured-window timeline plot.",
                "- `combined_utilization_summary.csv`: one row per run.",
                "- `combined_utilization_index.csv` / `combined_utilization_index.json`: input run index.",
                "",
                f"Runs included: {len(runs)}.",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_empty_outputs(out_dir: Path, clean: bool, reason: str) -> None:
    _prepare_output(out_dir, clean)
    _write_csv(out_dir / "combined_utilization_summary.csv", [], SUMMARY_FIELDS)
    _write_csv(out_dir / "combined_utilization_index.csv", [], INDEX_FIELDS)
    (out_dir / "combined_utilization_index.json").write_text("[]\n", encoding="utf-8")
    _write_readme(out_dir, runs=[], reason=reason)


def _group_label(run: RunRecord) -> str:
    metadata = run.metadata
    parts = [
        metadata.get("workload", ""),
        f"b{metadata.get('batch_size', '')}" if metadata.get("batch_size") else "",
        f"ga{metadata.get('gradient_accumulation_steps', '')}" if metadata.get("gradient_accumulation_steps") else "",
        f"drop{metadata.get('lora_dropout', '').replace('.', '')}" if metadata.get("lora_dropout") else "",
        metadata.get("precision", ""),
        metadata.get("backend", ""),
        metadata.get("profiler", ""),
        f"router{metadata.get('router_mode', '')}" if metadata.get("router_mode") else "",
        metadata.get("expact", ""),
        metadata.get("attnact", ""),
        metadata.get("layeract", ""),
        metadata.get("layergc", ""),
        metadata.get("liger_loss", ""),
        metadata.get("recompute", ""),
        f"pol{metadata.get('expert_policy', '')}" if metadata.get("expert_policy") else "",
    ]
    return _safe_label("-".join(part for part in parts if part))


def _write_grouped_outputs(runs: list[RunRecord], out_dir: Path, clean: bool) -> None:
    groups: dict[str, list[RunRecord]] = {}
    for run in runs:
        groups.setdefault(_group_label(run), []).append(run)
    for label, group_runs in sorted(groups.items()):
        if len(group_runs) <= 1:
            continue
        _write_outputs(group_runs, out_dir / label, clean, write_groups=False)


def _write_outputs(runs: list[RunRecord], out_dir: Path, clean: bool, *, write_groups: bool = True) -> None:
    _prepare_output(out_dir, clean)
    summary_rows = _summary_rows(runs)
    index_rows = [
        {
            **run.metadata,
            "run_label": run.label,
            "profile_json": str(run.profile_path),
            "run_dir": str(run.run_dir),
        }
        for run in runs
    ]
    _write_csv(out_dir / "combined_utilization_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(out_dir / "combined_utilization_index.csv", index_rows, INDEX_FIELDS)
    (out_dir / "combined_utilization_index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_timeline(runs, out_dir)
    _write_readme(out_dir, runs=runs)
    if write_groups:
        _write_grouped_outputs(runs, out_dir, clean)


def main() -> None:
    args = _parse_args()
    _apply_workload_filters(args)
    runs = _load_runs(args)
    if not runs:
        _write_empty_outputs(
            args.output_dir,
            args.clean_output,
            "No runs with usable GPU utilization metrics were found for the requested filters.",
        )
        print(f"[plot_lf_utilization] no runs matched; wrote empty artifacts to {args.output_dir}")
        return
    _write_outputs(runs, args.output_dir, args.clean_output)
    print(f"[plot_lf_utilization] wrote utilization artifacts for {len(runs)} run(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
