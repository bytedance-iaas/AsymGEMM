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


def _merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted((s, e) for s, e in intervals if e > s)
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = start, end
    merged.append((cur_s, cur_e))
    return merged


def _overlap_ns(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    total = 0
    for other_start, other_end in intervals:
        overlap = min(end, other_end) - max(start, other_start)
        if overlap > 0:
            total += overlap
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


def _sync_intervals(con: sqlite3.Connection, start: int, end: int) -> list[tuple[int, int]]:
    if not _table_exists(con, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        return []
    return [
        (int(s), int(e))
        for s, e in con.execute(
            "select start,end from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and end<=?",
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


def _kernel_events(con: sqlite3.Connection, correlation_ids: set[int]) -> list[dict[str, Any]]:
    if not correlation_ids or not _table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    placeholders = ",".join("?" for _ in correlation_ids)
    query = f"""
        select k.start,k.end,coalesce(s.value, '<unknown>')
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.correlationId in ({placeholders})
        order by k.start
    """
    return [
        {"start": int(start), "end": int(end), "name": str(name)}
        for start, end, name in con.execute(query, tuple(correlation_ids))
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


def _enclosing_range(gap_start: int, gap_end: int, ranges: list[tuple[int, int, str]], fallback: str) -> str:
    enclosing: list[tuple[int, str]] = []
    for start, end, text in ranges:
        if start <= gap_start and gap_end <= end:
            name = _normalize_range_name(text)
            if name is not None:
                enclosing.append((end - start, name))
    if not enclosing:
        return fallback
    enclosing.sort(key=lambda item: item[0])
    return enclosing[0][1]


def _kernel_before(kernel_events: list[dict[str, Any]], gap_start: int) -> str:
    before = [event for event in kernel_events if int(event["end"]) <= gap_start]
    if not before:
        return "<stage_start>"
    return str(max(before, key=lambda event: int(event["end"]))["name"])


def _kernel_after(kernel_events: list[dict[str, Any]], gap_end: int) -> str:
    after = [event for event in kernel_events if int(event["start"]) >= gap_end]
    if not after:
        return "<stage_end>"
    return str(min(after, key=lambda event: int(event["start"]))["name"])


def _no_kernel_gaps(
    *,
    stage_start: int,
    stage_end: int,
    total_ms: float,
    stage_name: str,
    ranges: list[tuple[int, int, str]],
    runtime: list[tuple[int, int, int]],
    sync_intervals: list[tuple[int, int]],
    kernel_events: list[dict[str, Any]],
    memcpy_intervals: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    busy = _merged_intervals([(int(event["start"]), int(event["end"])) for event in kernel_events] + memcpy_intervals)
    gaps: list[tuple[int, int]] = []
    cursor = stage_start
    for start, end in busy:
        clipped_start = max(stage_start, start)
        clipped_end = min(stage_end, end)
        if clipped_end <= stage_start or clipped_start >= stage_end:
            continue
        if clipped_start > cursor:
            gaps.append((cursor, clipped_start))
        cursor = max(cursor, clipped_end)
    if cursor < stage_end:
        gaps.append((cursor, stage_end))

    runtime_intervals = [(start, end) for start, end, _ in runtime]
    rows: list[dict[str, Any]] = []
    for gap_start, gap_end in gaps:
        gap_ms = _ms(gap_end - gap_start)
        if gap_ms <= 0.0:
            continue
        rows.append(
            {
                "previous_kernel": _kernel_before(kernel_events, gap_start),
                "next_kernel": _kernel_after(kernel_events, gap_end),
                "gap_milliseconds": gap_ms,
                "percent": _percent(gap_ms, total_ms),
                "enclosing_nvtx": _enclosing_range(gap_start, gap_end, ranges, stage_name),
                "cuda_api_overlap_milliseconds": _ms(_overlap_ns(gap_start, gap_end, runtime_intervals)),
                "sync_overlap_milliseconds": _ms(_overlap_ns(gap_start, gap_end, sync_intervals)),
                "start_offset_milliseconds": _ms(gap_start - stage_start),
                "end_offset_milliseconds": _ms(gap_end - stage_start),
            }
        )
    return rows


def summarize_stage(con: sqlite3.Connection, stage_name: str) -> dict[str, Any]:
    start, end = _last_step(con, stage_name)
    total_ms = _ms(end - start)
    runtime = _runtime_rows(con, start, end)
    runtime_corr = {corr for _, _, corr in runtime if corr is not None}
    kernel_intervals = _correlated_intervals(con, "CUPTI_ACTIVITY_KIND_KERNEL", runtime_corr)
    memcpy_intervals = _correlated_intervals(con, "CUPTI_ACTIVITY_KIND_MEMCPY", runtime_corr)
    sync_intervals = _sync_intervals(con, start, end)
    sync_ns = sum(sync_end - sync_start for sync_start, sync_end in sync_intervals)
    kernel_events = _kernel_events(con, runtime_corr)

    kernel_ms = _ms(_interval_union(kernel_intervals))
    memcpy_ms = _ms(_interval_union(memcpy_intervals))
    runtime_ms = _ms(sum(e - s for s, e, _ in runtime))
    sync_ms = _ms(sync_ns)
    gpu_idle_or_no_kernel_ms = max(0.0, total_ms - kernel_ms - memcpy_ms)

    prefix = stage_name.replace("step.", "") + ".%"
    op_kernel: dict[str, float] = defaultdict(float)
    op_api: dict[str, float] = defaultdict(float)
    ranges = _fetch_ranges(con, prefix)
    for r_start, r_end, text in ranges:
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
        "gpu_no_kernel_gaps": {
            "total_milliseconds": gpu_idle_or_no_kernel_ms,
            "rows": _no_kernel_gaps(
                stage_start=start,
                stage_end=end,
                total_ms=total_ms,
                stage_name=stage_name,
                ranges=ranges,
                runtime=runtime,
                sync_intervals=sync_intervals,
                kernel_events=kernel_events,
                memcpy_intervals=memcpy_intervals,
            ),
        },
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
        lines += [
            "### GPU No-Kernel Gaps",
            "",
            "| Previous kernel | Next kernel | gap ms | % stage | enclosing NVTX | CUDA API overlap ms | sync overlap ms | stage offset ms |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for row in stage["gpu_no_kernel_gaps"]["rows"]:
            lines.append(
                "| "
                f"`{row['previous_kernel']}` | "
                f"`{row['next_kernel']}` | "
                f"{row['gap_milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"`{row['enclosing_nvtx']}` | "
                f"{row['cuda_api_overlap_milliseconds']:.4f} | "
                f"{row['sync_overlap_milliseconds']:.4f} | "
                f"{row['start_offset_milliseconds']:.4f} |"
            )
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
