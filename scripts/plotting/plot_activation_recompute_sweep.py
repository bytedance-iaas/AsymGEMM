#!/usr/bin/env python3
"""Plot activation-recompute sequence-length sweeps from LoRA driver outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MIB = 1024.0**2
RESULT_RE = re.compile(
    r"^(?P<precision>.+?)_lora-sft_b(?P<batch_size>[0-9]+)_s(?P<seq_len>[0-9]+)_"
    r"(?P<recompute>recomp|norecomp)_(?P<tail>.+)$"
)
PROFILERS = ("source", "nsys", "cpu", "ncu")


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
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--seq-len", "--seq-lens", dest="seq_lens", nargs="+", type=int, default=[])
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
    if args.precision and meta["precision"] != args.precision:
        return False
    if args.workload and meta["workload"] not in set(args.workload):
        return False
    if args.backend and meta["backend"] not in set(args.backend):
        return False
    if args.profiler and meta["profiler"] not in set(args.profiler):
        return False
    if args.batch_size and meta["batch_size"] not in set(args.batch_size):
        return False
    if args.seq_lens and meta["seq_len"] not in set(args.seq_lens):
        return False
    return True


def result_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*_lora-sft_*") if path.is_dir())


def row_from_result_dir(args: argparse.Namespace, result_dir: Path) -> dict[str, Any] | None:
    meta = parse_result_dir(result_dir)
    if meta is None or not passes_filters(args, meta):
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
    return {
        "workload": meta["workload"],
        "precision": meta["precision"],
        "batch_size": batch_size,
        "seq_len": int(meta["seq_len"]),
        "logical_tokens": int(config.get("logical_tokens", batch_size * int(meta["seq_len"]))),
        "mode": meta["mode"],
        "activation_recompute": bool(meta["activation_recompute"]),
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
        "output_dir": str(result_dir),
        "profile_json": str(profile_path),
    }


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_root = resolve_path(args.input_root)
    rows = [row for path in result_dirs(input_root) if (row := row_from_result_dir(args, path)) is not None]
    return sorted(rows, key=lambda row: (row["workload"], row["batch_size"], row["backend"], row["profiler"], row["seq_len"], row["mode"]))


def group_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (str(row["workload"]), str(row["precision"]), int(row["batch_size"]), str(row["backend"]), str(row["profiler"]))


def write_table(rows: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()
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

    labels = {"no_recompute": "No recompute", "recompute": "Activation recompute"}
    series: dict[tuple[tuple[str, str, int, str, str], str], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((group_key(row), str(row["mode"])), []).append(row)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for (group, mode), group_rows in sorted(series.items()):
        workload, precision, batch_size, backend, profiler = group
        sorted_rows = sorted(group_rows, key=lambda row: row["seq_len"])
        label = f"{workload} b{batch_size} {precision} {backend}/{profiler} {labels.get(mode, mode)}"
        ax.plot(
            [row["seq_len"] for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="best")
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


def write_combined_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
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


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    if not rows:
        raise SystemExit(f"no driver result directories found under {resolve_path(args.input_root)}")

    root = output_root(args)
    write_table(rows, root, "activation_recompute_sweep_index")
    write_combined_plots(rows, root)

    groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    for key, group_rows in sorted(groups.items()):
        workload, precision, batch_size, backend, profiler = key
        group_dir = root / safe_label(f"{workload}-b{batch_size}-{precision}-{backend}-{profiler}")
        write_table(group_rows, group_dir, "sweep_summary")
        write_group_plots(group_rows, group_dir, key)
        print(f"wrote {group_dir}", flush=True)
    print(f"wrote {root / 'activation_recompute_sweep_index.json'}", flush=True)


if __name__ == "__main__":
    main()
