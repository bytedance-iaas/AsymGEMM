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


CTC_METRICS = ("ctc_rx", "ctc_tx")
SUMMARY_FIELDS = [
    "workload",
    "backend",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "lora_dropout",
    "config",
    "metric",
    "metric_name",
    "run_label",
    "step_count",
    "p95_mean_percent",
    "p95_peak_percent",
    "p95_peak_step",
    "max_peak_percent",
    "max_peak_step",
    "p95_saturated_steps_ge90",
    "max_saturated_steps_ge90",
    "profile_json",
    "run_dir",
]
STEP_FIELDS = [
    "workload",
    "backend",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "lora_dropout",
    "config",
    "metric",
    "metric_name",
    "run_label",
    "step",
    "avg_percent",
    "p50_percent",
    "p95_percent",
    "max_percent",
    "profile_json",
    "run_dir",
]
INDEX_FIELDS = [
    "workload",
    "backend",
    "profiler",
    "recompute",
    "expert_policy",
    "seq_len",
    "precision",
    "lora_dropout",
    "config",
    "run_label",
    "profile_json",
    "run_dir",
]
METRIC_LABELS = {
    "ctc_rx": "C2C RX",
    "ctc_tx": "C2C TX",
}
METRIC_COLORS = {
    "ctc_rx": "#1f77b4",
    "ctc_tx": "#d62728",
}


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    profile_path: Path
    metadata: dict[str, str]
    by_step: dict[int, dict[str, dict[str, float]]]

    @property
    def label(self) -> str:
        parts = [
            self.metadata.get("workload", ""),
            self.metadata.get("backend", ""),
            self.metadata.get("recompute", ""),
            self.metadata.get("expert_policy", ""),
            f"s{self.metadata.get('seq_len', '')}" if self.metadata.get("seq_len") else "",
        ]
        return " ".join(part for part in parts if part)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined LF Nsight C2C/CTC saturation artifacts.")
    parser.add_argument("--input-root", type=Path, action="append", default=[], help="Root to scan for nsys profile.json files.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit run directory containing profile.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for combined C2C plots.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--combined-only", action="store_true", help="Accepted for parity with other LF plotting scripts.")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--profiler", action="append", default=[])
    parser.add_argument("--recompute", action="append", default=[])
    parser.add_argument("--seq-lens", nargs="+", default=[])
    parser.add_argument("--expert-recompute-policies", nargs="+", default=[])
    return parser.parse_args()


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_label(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "run"


def _filter_values(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                result.add(part.lower())
    return result


def _find_profile_paths(input_roots: list[Path], run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        profile_path = run_dir / "profile.json"
        if profile_path.exists():
            paths.append(profile_path)
    for root in input_roots:
        if root.is_file() and root.name == "profile.json":
            paths.append(root)
            continue
        if not root.exists():
            continue
        paths.extend(sorted(root.rglob("profile.json")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _source_config(run_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    source_profile = profile.get("source_profile")
    if not isinstance(source_profile, dict) or not source_profile:
        source_profile = _safe_read_json(run_dir / "source_profile.json")
    config = source_profile.get("config") if isinstance(source_profile, dict) else {}
    return config if isinstance(config, dict) else {}


def _infer_metadata(profile_path: Path, profile: dict[str, Any]) -> dict[str, str]:
    run_dir = profile_path.parent
    job_root = run_dir.parent
    config_root = job_root.parent
    job_parts = job_root.name.split("__")
    config = _source_config(run_dir, profile)

    expert_policy = str(config.get("expert_policy") or "")
    for part in job_parts:
        if part.startswith("pol") and not expert_policy:
            expert_policy = part[len("pol") :]

    seq_len = str(config.get("seq_len") or config.get("cutoff_len") or "")
    if not seq_len and run_dir.name.startswith("s") and run_dir.name[1:].isdigit():
        seq_len = run_dir.name[1:]

    recompute = str(job_parts[2] if len(job_parts) > 2 else "")
    if not recompute:
        recompute = "recomp" if bool(config.get("activation_recompute", False)) else "norecomp"

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__", 1)[0]),
        "backend": str(config.get("backend") or (job_parts[0] if len(job_parts) > 0 else "")),
        "profiler": str(job_parts[1] if len(job_parts) > 1 else "nsys"),
        "recompute": recompute,
        "expert_policy": expert_policy or "none",
        "seq_len": seq_len,
        "precision": str(config.get("precision") or ""),
        "lora_dropout": str(config.get("lora_dropout") if config.get("lora_dropout") is not None else ""),
        "config": config_root.name,
    }
    return metadata


def _matches_filters(record: RunRecord, args: argparse.Namespace) -> bool:
    filters = {
        "workload": _filter_values(args.workload),
        "backend": _filter_values(args.backend),
        "profiler": _filter_values(args.profiler),
        "recompute": _filter_values(args.recompute),
        "seq_len": _filter_values(args.seq_lens),
        "expert_policy": _filter_values(args.expert_recompute_policies),
    }
    for key, allowed in filters.items():
        if not allowed:
            continue
        value = record.metadata.get(key, "").lower()
        if key == "workload":
            config_workload = record.metadata.get("config", "").split("__", 1)[0].lower()
            if value in allowed or config_workload in allowed:
                continue
        if value not in allowed:
            return False
    return True


def _aggregate_step_rows(rows: list[Any]) -> dict[int, dict[str, dict[str, float]]]:
    by_step: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("scope") != "step":
            continue
        metric = str(row.get("metric") or "")
        if metric not in CTC_METRICS:
            continue
        try:
            step = int(row.get("step"))
        except (TypeError, ValueError):
            continue
        entry = by_step.setdefault(step, {}).setdefault(
            metric,
            {
                "avg_percent": 0.0,
                "p50_percent": 0.0,
                "p95_percent": 0.0,
                "max_percent": 0.0,
            },
        )
        for key in ("avg_percent", "p50_percent", "p95_percent", "max_percent"):
            entry[key] = max(entry[key], _to_float(row.get(key)))
    return by_step


def _load_runs(args: argparse.Namespace) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for profile_path in _find_profile_paths(args.input_root, args.run_dir):
        profile = _safe_read_json(profile_path)
        metrics = profile.get("interconnect_metrics")
        if not isinstance(metrics, dict) or not metrics.get("available"):
            continue
        range_summary = metrics.get("range_summary")
        rows = range_summary.get("rows") if isinstance(range_summary, dict) else []
        if not isinstance(rows, list):
            continue
        by_step = _aggregate_step_rows(rows)
        if not by_step:
            continue
        record = RunRecord(
            run_dir=profile_path.parent,
            profile_path=profile_path,
            metadata=_infer_metadata(profile_path, profile),
            by_step=by_step,
        )
        if _matches_filters(record, args):
            runs.append(record)
    return sorted(
        runs,
        key=lambda run: (
            run.metadata.get("workload", ""),
            int(run.metadata.get("seq_len") or 0),
            run.metadata.get("backend", ""),
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
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _metric_step_values(run: RunRecord, metric: str, stat: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for step in sorted(run.by_step):
        values.append((step, float(run.by_step[step].get(metric, {}).get(stat, 0.0))))
    return values


def _summary_rows(runs: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for metric in CTC_METRICS:
            p95_values = _metric_step_values(run, metric, "p95_percent")
            max_values = _metric_step_values(run, metric, "max_percent")
            if not any(value for _step, value in p95_values) and not any(value for _step, value in max_values):
                continue
            peak_p95_step, peak_p95_value = max(p95_values, key=lambda item: item[1], default=(0, 0.0))
            peak_max_step, peak_max_value = max(max_values, key=lambda item: item[1], default=(0, 0.0))
            nonzero_p95 = [value for _step, value in p95_values if value > 0.0]
            rows.append(
                {
                    **run.metadata,
                    "metric": metric,
                    "metric_name": METRIC_LABELS[metric],
                    "run_label": run.label,
                    "step_count": len(run.by_step),
                    "p95_mean_percent": sum(nonzero_p95) / len(nonzero_p95) if nonzero_p95 else 0.0,
                    "p95_peak_percent": peak_p95_value,
                    "p95_peak_step": peak_p95_step,
                    "max_peak_percent": peak_max_value,
                    "max_peak_step": peak_max_step,
                    "p95_saturated_steps_ge90": sum(1 for _step, value in p95_values if value >= 90.0),
                    "max_saturated_steps_ge90": sum(1 for _step, value in max_values if value >= 90.0),
                    "profile_json": str(run.profile_path),
                    "run_dir": str(run.run_dir),
                }
            )
    return rows


def _step_rows(runs: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for step in sorted(run.by_step):
            for metric in CTC_METRICS:
                stats = run.by_step[step].get(metric)
                if not stats:
                    continue
                rows.append(
                    {
                        **run.metadata,
                        "metric": metric,
                        "metric_name": METRIC_LABELS[metric],
                        "run_label": run.label,
                        "step": step,
                        **stats,
                        "profile_json": str(run.profile_path),
                        "run_dir": str(run.run_dir),
                    }
                )
    return rows


def _plot_by_step(runs: list[RunRecord], out_dir: Path) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 2 else 1
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 3.4 * nrows), sharey=True, squeeze=False, constrained_layout=True)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        for metric in CTC_METRICS:
            p95 = _metric_step_values(run, metric, "p95_percent")
            max_values = _metric_step_values(run, metric, "max_percent")
            if not any(value for _step, value in p95) and not any(value for _step, value in max_values):
                continue
            steps = [step for step, _value in p95]
            ax.plot(
                steps,
                [value for _step, value in p95],
                label=f"{METRIC_LABELS[metric]} p95",
                color=METRIC_COLORS[metric],
                linewidth=1.7,
            )
            ax.plot(
                [step for step, _value in max_values],
                [value for _step, value in max_values],
                label=f"{METRIC_LABELS[metric]} max",
                color=METRIC_COLORS[metric],
                linestyle=":",
                linewidth=1.3,
            )
        ax.axhline(90.0, color="#444444", linestyle="--", linewidth=0.9)
        ax.set_title(run.label or run.run_dir.name, fontsize=9)
        ax.set_xlabel("Measured step")
        ax.set_ylim(bottom=0.0, top=max(100.0, ax.get_ylim()[1]))
        ax.grid(True, axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Saturation (%)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Combined C2C Saturation by Measured Step", fontsize=12)
    fig.savefig(out_dir / "combined_c2c_saturation_by_step.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_summary(runs: list[RunRecord], out_dir: Path) -> None:
    labels = [run.label or run.run_dir.name for run in runs]
    summary = _summary_rows(runs)
    by_run_metric = {(row["profile_json"], row["metric"]): row for row in summary}
    y_positions = list(range(len(runs)))
    height = max(4.5, min(28.0, 0.52 * len(runs) + 2.5))
    fig, ax = plt.subplots(figsize=(10.0, height), constrained_layout=True)
    for metric, offset in (("ctc_rx", -0.18), ("ctc_tx", 0.18)):
        p95 = [
            float(by_run_metric.get((str(run.profile_path), metric), {}).get("p95_peak_percent", 0.0))
            for run in runs
        ]
        max_values = [
            float(by_run_metric.get((str(run.profile_path), metric), {}).get("max_peak_percent", 0.0))
            for run in runs
        ]
        ys = [y + offset for y in y_positions]
        ax.barh(ys, p95, height=0.3, color=METRIC_COLORS[metric], alpha=0.72, label=f"{METRIC_LABELS[metric]} p95 peak")
        ax.scatter(max_values, ys, color=METRIC_COLORS[metric], marker="D", s=18, label=f"{METRIC_LABELS[metric]} max peak")
    ax.axvline(90.0, color="#444444", linestyle="--", linewidth=1.0, label="90%")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Saturation (%)")
    ax.set_title("Combined C2C Peak Saturation")
    all_values = [float(row.get("p95_peak_percent", 0.0)) for row in summary] + [float(row.get("max_peak_percent", 0.0)) for row in summary]
    ax.set_xlim(left=0.0, right=max(100.0, max(all_values, default=0.0) * 1.08))
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_dir / "combined_c2c_peak_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, *, runs: list[RunRecord], reason: str | None = None) -> None:
    lines = [
        "# LF C2C / CTC Combined Artifacts",
        "",
        "These files summarize Nsight Systems GPU metric samples for NVLink-C2C/CTC throughput saturation.",
        "Values are percent of peak throughput reported by Nsight GPU metrics, not absolute GB/s.",
        "Per-step values aggregate with max across GPUs so one saturated GPU is not hidden by idle GPUs.",
        "",
    ]
    if reason:
        lines.extend(
            [
                "## Status",
                "",
                reason,
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Files",
                "",
                "- `combined_c2c_saturation_by_step.png`: subplots per run; x-axis is measured step, y-axis is C2C RX/TX saturation percent.",
                "- `combined_c2c_peak_summary.png`: per-run peak p95 bars and max markers.",
                "- `combined_c2c_summary.csv`: one row per run and C2C direction.",
                "- `combined_c2c_step_summary.csv`: one row per run, step, and C2C direction.",
                "- `combined_c2c_index.csv` / `combined_c2c_index.json`: input run index.",
                "",
                f"Runs included: {len(runs)}.",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_empty_outputs(out_dir: Path, clean: bool, reason: str) -> None:
    _prepare_output(out_dir, clean)
    _write_csv(out_dir / "combined_c2c_summary.csv", [], SUMMARY_FIELDS)
    _write_csv(out_dir / "combined_c2c_step_summary.csv", [], STEP_FIELDS)
    _write_csv(out_dir / "combined_c2c_index.csv", [], INDEX_FIELDS)
    (out_dir / "combined_c2c_index.json").write_text("[]\n", encoding="utf-8")
    _write_readme(out_dir, runs=[], reason=reason)


def _write_outputs(runs: list[RunRecord], out_dir: Path, clean: bool) -> None:
    _prepare_output(out_dir, clean)
    summary_rows = _summary_rows(runs)
    step_rows = _step_rows(runs)
    index_rows = [
        {
            **run.metadata,
            "run_label": run.label,
            "profile_json": str(run.profile_path),
            "run_dir": str(run.run_dir),
        }
        for run in runs
    ]
    _write_csv(out_dir / "combined_c2c_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(out_dir / "combined_c2c_step_summary.csv", step_rows, STEP_FIELDS)
    _write_csv(out_dir / "combined_c2c_index.csv", index_rows, INDEX_FIELDS)
    (out_dir / "combined_c2c_index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_by_step(runs, out_dir)
    _plot_peak_summary(runs, out_dir)
    _write_readme(out_dir, runs=runs)


def main() -> None:
    args = _parse_args()
    runs = _load_runs(args)
    if not runs:
        reason = (
            "No nsys `profile.json` files with C2C/CTC step metrics matched the requested filters. "
            "Fresh traces must be collected with Nsight GPU metrics enabled; older traces without "
            "`GPU_METRICS` tables cannot be converted into C2C saturation plots."
        )
        _write_empty_outputs(args.output_dir, args.clean_output, reason)
        print(reason)
        return
    _write_outputs(runs, args.output_dir, args.clean_output)
    print(f"Plotted {len(runs)} C2C/CTC run(s).")


if __name__ == "__main__":
    main()
