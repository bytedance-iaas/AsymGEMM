#!/usr/bin/env python3
"""Summarize Nsight Compute CSV output for AsymGEMM kernels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_OPS = {
    "matrix_1b": [
        "forward.matrix.base_frozen_asymgemm",
        "backward.matrix.base_dx_asymgemm",
    ],
    "mlp_1b": [
        "forward.fc1.base_frozen_asymgemm",
        "forward.fc2.base_frozen_asymgemm",
        "backward.fc2.base_dx_asymgemm",
        "backward.fc1.base_dx_asymgemm",
    ],
}


KEY_METRICS = [
    ("duration_ns", "gpu__time_duration.sum"),
    ("sm_throughput_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("tensor_pipe_pct", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("tensor_type_pct", "sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("memory_throughput_pct", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram_throughput_pct", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l2_throughput_pct", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1tex_throughput_pct", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("issue_active_pct", "sm__issue_active.avg.pct_of_peak_sustained_elapsed"),
    ("warps_active_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("max_warps_active_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("registers_per_thread", "launch__registers_per_thread"),
    ("shared_mem_per_block_bytes", "launch__shared_mem_per_block"),
    ("dynamic_shared_mem_per_block_bytes", "launch__shared_mem_per_block_dynamic"),
    ("block_size", "launch__block_size"),
    ("grid_size", "launch__grid_size"),
    ("waves_per_sm", "launch__waves_per_multiprocessor"),
    ("replay_passes", "profiler__replayer_passes"),
]


CATEGORY_PATTERNS = {
    "tensor_core_util": ("tensor", "sm__pipe_tensor", "smsp__inst_executed_pipe_tensor"),
    "memory_throughput": ("gpu__compute_memory", "gpu__dram", "dram__", "l1tex__throughput", "lts__throughput"),
    "l2_dram_behavior": ("lts__", "dram__", "sysmem", "fill_sysmem", "sector"),
    "occupancy": ("occupancy", "warps_active", "waves_per_multiprocessor", "maximum_warps"),
    "warp_stall_reasons": ("warp_issue_stalled", "warp_cycles_per_stall", "stall"),
    "scheduler_stats": ("issue_active", "scheduler", "eligible", "issued", "smsp__issue"),
    "roofline_position": ("roofline", "flop", "op_hmma", "op_dmma", "op_fma"),
    "launch_stats": ("launch__",),
}


def _numeric(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value or value in {"n/a", "N/A", "nan", "NaN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_ncu_csv(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for idx, line in enumerate(raw_lines):
        if "Kernel Name" in line and line.lstrip().startswith('"ID"'):
            header_index = idx
            break
    if header_index is None:
        raise RuntimeError(f"could not find NCU CSV header in {path}")

    parsed = list(csv.reader(raw_lines[header_index:]))
    if len(parsed) < 2:
        raise RuntimeError(f"NCU CSV has no data rows: {path}")
    header = parsed[0]
    units = parsed[1] if parsed[1] and not parsed[1][0].strip() else ["" for _ in header]
    rows: list[dict[str, str]] = []
    for row in parsed[2:]:
        if not row or len(row) < len(header):
            continue
        row = row[: len(header)]
        if not row[0].strip().isdigit():
            continue
        rows.append(dict(zip(header, row)))
    return header, units, rows


def _metric(row: dict[str, str], metric_name: str) -> float | None:
    return _numeric(row.get(metric_name))


def _metric_unit(header: list[str], units: list[str], metric_name: str) -> str:
    try:
        return units[header.index(metric_name)]
    except ValueError:
        return ""


def _short_kernel(name: str, limit: int = 120) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def _category_rows(header: list[str], units: list[str], rows: list[dict[str, str]], patterns: tuple[str, ...], limit: int = 20) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    lower_patterns = tuple(pattern.lower() for pattern in patterns)
    for metric_name in header:
        lower = metric_name.lower()
        if not any(pattern in lower for pattern in lower_patterns):
            continue
        if lower.startswith("device__attribute"):
            continue
        if "cycles_elapsed" in lower and "pct_of_peak" not in lower:
            continue
        values = [_metric(row, metric_name) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        unit = _metric_unit(header, units, metric_name)
        priority = 0 if unit == "%" or "pct_of_peak" in lower or "pct" in lower else 1
        candidates.append(
            {
                "metric": metric_name,
                "unit": unit,
                "avg": sum(numeric_values) / len(numeric_values),
                "max": max(numeric_values),
                "min": min(numeric_values),
                "_priority": priority,
            }
        )
    candidates.sort(key=lambda item: (int(item["_priority"]), -abs(float(item["avg"]))))
    for item in candidates:
        item.pop("_priority", None)
    return candidates[:limit]


def _top_warp_stalls(header: list[str], units: list[str], row: dict[str, str], limit: int = 10) -> list[dict[str, Any]]:
    stalls: list[dict[str, Any]] = []
    for metric_name in header:
        lower = metric_name.lower()
        if "warps_issue_stalled" not in lower and "warp_issue_stalled" not in lower and "warp_cycles_per_stall" not in lower:
            continue
        value = _metric(row, metric_name)
        if value is None or value <= 0:
            continue
        stalls.append({"metric": metric_name, "value": value, "unit": _metric_unit(header, units, metric_name)})
    stalls.sort(key=lambda item: float(item["value"]), reverse=True)
    return stalls[:limit]


def summarize(path: Path, *, workload: str | None = None) -> dict[str, Any]:
    header, units, rows = _read_ncu_csv(path)
    expected_ops = EXPECTED_OPS.get(workload or "", [])
    kernel_summaries: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        key_metrics: dict[str, Any] = {}
        for label, metric_name in KEY_METRICS:
            value = _metric(row, metric_name)
            if value is not None:
                key_metrics[label] = value
        duration_ns = key_metrics.get("duration_ns")
        if duration_ns is not None:
            key_metrics["duration_ms"] = float(duration_ns) / 1_000_000.0
        kernel_summaries.append(
            {
                "id": row.get("ID", str(idx)),
                "operation": expected_ops[idx] if idx < len(expected_ops) else f"asymgemm_kernel_{idx}",
                "kernel_name": row.get("Kernel Name", ""),
                "key_metrics": key_metrics,
                "top_warp_stalls": _top_warp_stalls(header, units, row),
            }
        )

    categories = {
        name: _category_rows(header, units, rows, patterns)
        for name, patterns in CATEGORY_PATTERNS.items()
    }
    return {
        "source": str(path),
        "workload": workload,
        "kernel_count": len(rows),
        "kernels": kernel_summaries,
        "categories": categories,
        "notes": [
            "Nsight Compute replays kernels and is for kernel-internal diagnosis, not end-to-end wall-time claims.",
            "Use Nsight Systems tables for true timeline percentages and GPU no-kernel bubbles.",
            "Source-line correlation requires JIT cubins compiled with lineinfo.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# NCU AsymGEMM Kernel Report: {report['workload']}", "", f"Source: `{report['source']}`", ""]
    lines += [
        "## Kernel Summary",
        "",
        "| ID | Operation | duration ms | tensor pipe % | SM throughput % | memory throughput % | DRAM % | L2 % | issue active % | active warps % | regs/thread | smem/block | replay passes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel in report["kernels"]:
        metrics = kernel["key_metrics"]
        tensor = metrics.get("tensor_pipe_pct", metrics.get("tensor_type_pct"))
        lines.append(
            "| "
            f"{kernel['id']} | "
            f"`{kernel['operation']}` | "
            f"{_fmt(metrics.get('duration_ms'))} | "
            f"{_fmt(tensor)} | "
            f"{_fmt(metrics.get('sm_throughput_pct'))} | "
            f"{_fmt(metrics.get('memory_throughput_pct'))} | "
            f"{_fmt(metrics.get('dram_throughput_pct'))} | "
            f"{_fmt(metrics.get('l2_throughput_pct'))} | "
            f"{_fmt(metrics.get('issue_active_pct'))} | "
            f"{_fmt(metrics.get('warps_active_pct'))} | "
            f"{_fmt(metrics.get('registers_per_thread'))} | "
            f"{_fmt(metrics.get('shared_mem_per_block_bytes'))} | "
            f"{_fmt(metrics.get('replay_passes'))} |"
        )
    lines.append("")

    for kernel in report["kernels"]:
        lines += [
            f"## Kernel {kernel['id']}: `{kernel['operation']}`",
            "",
            f"`{_short_kernel(kernel['kernel_name'])}`",
            "",
            "### Top Warp Stalls",
            "",
            "| Metric | value | unit |",
            "|---|---:|---|",
        ]
        for row in kernel["top_warp_stalls"]:
            lines.append(f"| `{row['metric']}` | {row['value']:.4f} | {row['unit']} |")
        if not kernel["top_warp_stalls"]:
            lines.append("| - | - | - |")
        lines.append("")

    for category, rows in report["categories"].items():
        lines += [f"## {category}", "", "| Metric | avg | min | max | unit |", "|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| `{row['metric']}` | {row['avg']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['unit']} |")
        if not rows:
            lines.append("| - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--workload", choices=sorted(EXPECTED_OPS))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    report = summarize(args.csv_path, workload=args.workload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(report), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
