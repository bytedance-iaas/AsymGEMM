#!/usr/bin/env python3
"""Postprocess an Nsight Systems SQLite export for M4/LoRA profiling.

Run the workload with `scripts/profile_m4_steps.py --timing-mode profile` or
with the same `asym_gemm.training.profile_ranges.prof_range()` NVTX labels in a
larger integration such as LLaMA-Factory.  This postprocessor reads the Nsight
Systems database and reports, per `step.forward` / `step.backward`:

* CUDA kernel busy time, as a union of kernel intervals.
* memcpy time.
* CUDA runtime API time.
* CUDA synchronization API time.
* GPU no-kernel time.
* named NVTX operation ranges with correlated kernel/API time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sqlite3
from pathlib import Path
from typing import Any


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def _interval_union(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted((s, e) for s, e in intervals if e > s)
    total = 0
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    total += cur_e - cur_s
    return total


def _percent(value: float, total: float) -> float:
    return 0.0 if total <= 0.0 else value * 100.0 / total


def _ms(ns: int | float) -> float:
    return float(ns) / 1_000_000.0


def _rows(values: dict[str, float], total: float) -> list[dict[str, Any]]:
    return [
        {"name": name, "milliseconds": ms, "percent": _percent(ms, total)}
        for name, ms in sorted(values.items(), key=lambda item: item[1], reverse=True)
        if ms > 0.0
    ]


def _normalize_range_name(name: str) -> str | None:
    if not (name.startswith("forward.") or name.startswith("backward.")):
        return None
    # Source-debug child ranges are emitted only to build synchronized coverage
    # tables.  The parent NVTX range is the Nsight truth attribution.
    if ".call_" in name:
        return None
    if name.endswith(".dispatch_loop"):
        return None
    if name.startswith("forward.base_frozen_asymgemm"):
        return None
    if name.startswith("backward."):
        suffixes = (
            ".input_grad",
            ".weight_grad",
            ".bias_grad",
            ".grad",
            ".scale_cast_grad",
            ".gate_activation_grad",
            ".up_mul_grad",
        )
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
    return name


def _fetch_ranges(con: sqlite3.Connection, pattern: str) -> list[tuple[int, int, str]]:
    return [
        (int(start), int(end), str(text))
        for start, end, text in con.execute(
            "select start,end,text from NVTX_EVENTS where text like ? and end is not null order by start",
            (pattern,),
        )
    ]


def _last_step(con: sqlite3.Connection, name: str) -> tuple[int, int]:
    row = con.execute(
        "select start,end from NVTX_EVENTS where text=? and end is not null order by start desc limit 1",
        (name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing NVTX range {name!r}")
    return int(row[0]), int(row[1])


def _runtime_rows(con: sqlite3.Connection, start: int, end: int) -> list[tuple[int, int, int]]:
    if not _table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    return [
        (int(s), int(e), int(corr))
        for s, e, corr in con.execute(
            "select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?",
            (start, end),
        )
    ]


def _correlated_intervals(con: sqlite3.Connection, table: str, correlation_ids: set[int]) -> list[tuple[int, int]]:
    if not correlation_ids or not _table_exists(con, table):
        return []
    placeholders = ",".join("?" for _ in correlation_ids)
    return [
        (int(s), int(e))
        for s, e in con.execute(
            f"select start,end from {table} where correlationId in ({placeholders})",
            tuple(correlation_ids),
        )
    ]


def _kernel_name_rows(con: sqlite3.Connection, correlation_ids: set[int]) -> dict[str, float]:
    if not correlation_ids or not _table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return {}
    placeholders = ",".join("?" for _ in correlation_ids)
    query = f"""
        select coalesce(s.value, '<unknown>'), sum(k.end-k.start)/1000000.0
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.correlationId in ({placeholders})
        group by s.value
    """
    return {str(name): float(ms or 0.0) for name, ms in con.execute(query, tuple(correlation_ids))}


def summarize_stage(con: sqlite3.Connection, stage_name: str) -> dict[str, Any]:
    start, end = _last_step(con, stage_name)
    total_ms = _ms(end - start)
    runtime = _runtime_rows(con, start, end)
    runtime_corr = {corr for _, _, corr in runtime if corr is not None}
    kernel_intervals = _correlated_intervals(con, "CUPTI_ACTIVITY_KIND_KERNEL", runtime_corr)
    memcpy_intervals = _correlated_intervals(con, "CUPTI_ACTIVITY_KIND_MEMCPY", runtime_corr)
    sync_ns = 0
    if _table_exists(con, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        sync_ns = int(con.execute(
            "select coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and end<=?",
            (start, end),
        ).fetchone()[0])

    kernel_ms = _ms(_interval_union(kernel_intervals))
    memcpy_ms = _ms(_interval_union(memcpy_intervals))
    runtime_ms = _ms(sum(e - s for s, e, _ in runtime))
    sync_ms = _ms(sync_ns)
    gpu_idle_or_no_kernel_ms = max(0.0, total_ms - kernel_ms - memcpy_ms)

    prefix = stage_name.replace("step.", "") + ".%"
    op_kernel: dict[str, float] = defaultdict(float)
    op_api: dict[str, float] = defaultdict(float)
    for r_start, r_end, text in _fetch_ranges(con, prefix):
        if r_start < start or r_end > end:
            continue
        name = _normalize_range_name(text)
        if name is None or name == stage_name:
            continue
        corr = {corr for s, e, corr in runtime if s >= r_start and e <= r_end}
        op_api[name] += _ms(sum(e - s for s, e, corr_id in runtime if corr_id in corr))
        op_kernel[name] += _ms(_interval_union(_correlated_intervals(con, "CUPTI_ACTIVITY_KIND_KERNEL", corr)))

    summary = {
        "stage": stage_name,
        "total_milliseconds": total_ms,
        "stage_breakdown": {
            "total_milliseconds": total_ms,
            "rows": _rows(
                {
                    "cuda_kernel_busy_union": kernel_ms,
                    "cuda_memcpy_union": memcpy_ms,
                    "gpu_no_kernel_time": gpu_idle_or_no_kernel_ms,
                },
                total_ms,
            ),
        },
        "host_api_breakdown": {
            "total_milliseconds": total_ms,
            "rows": _rows(
                {
                    "cuda_runtime_api_sum_overlaps_gpu_timeline": runtime_ms,
                    "cuda_synchronization_api_sum_overlaps_gpu_timeline": sync_ms,
                },
                total_ms,
            ),
        },
        "operation_kernel_time": {
            "total_milliseconds": total_ms,
            "rows": _rows(op_kernel, total_ms),
        },
        "operation_cuda_api_time": {
            "total_milliseconds": total_ms,
            "rows": _rows(op_api, total_ms),
        },
        "top_kernels": _rows(_kernel_name_rows(con, runtime_corr), total_ms)[:20],
    }
    return summary


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Nsight M4 Trace: {report['source']}", ""]
    for stage in report["stages"]:
        lines += [f"## {stage['stage']}", ""]
        for title, key in [
            ("Stage Timeline", "stage_breakdown"),
            ("Host CUDA API", "host_api_breakdown"),
            ("Operation Kernel Time", "operation_kernel_time"),
            ("Operation CUDA API Time", "operation_cuda_api_time"),
        ]:
            lines += [f"### {title}", "", "| Component | ms | % stage |", "|---|---:|---:|"]
            for row in stage[key]["rows"]:
                lines.append(f"| {row['name']} | {row['milliseconds']:.4f} | {row['percent']:.2f}% |")
            lines.append("")
        lines += ["### Top Kernels", "", "| Kernel | ms | % stage |", "|---|---:|---:|"]
        for row in stage["top_kernels"]:
            lines.append(f"| `{row['name']}` | {row['milliseconds']:.4f} | {row['percent']:.2f}% |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    con = sqlite3.connect(str(args.sqlite_path))
    report = {
        "source": str(args.sqlite_path),
        "stages": [summarize_stage(con, "step.forward"), summarize_stage(con, "step.backward")],
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(markdown(report), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
