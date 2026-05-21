#!/usr/bin/env python3
"""Postprocess Nsight Systems CPU-debug traces for GPU no-kernel gaps.

This is a debug companion to ``postprocess_nsys_lora.py``.  The base LoRA-SFT
postprocessor remains the low-overhead GPU timeline truth.  This script reads a
Nsight Systems SQLite export captured with CUDA + NVTX + OS runtime + CPU
sampling/context-switch tracing and explains the no-kernel gaps with:

* CPU sample stack buckets on the stage submission threads.
* OS runtime API overlap on those same threads.
* CUDA runtime and synchronization overlap.
* Scheduler/off-CPU context-switch intervals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.postprocess_nsys_lora import (  # noqa: E402
    _correlated_intervals,
    _fetch_ranges,
    _kernel_events,
    _ms,
    _no_kernel_gaps,
    _percent,
    _runtime_rows,
    _sync_intervals,
    _table_exists,
    summarize_stage,
)


def _clip_overlap_ns(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start))


def _clip_sum_ns(start: int, end: int, intervals: Iterable[tuple[int, int]]) -> int:
    return sum(_clip_overlap_ns(start, end, other_start, other_end) for other_start, other_end in intervals)


def _string(con: sqlite3.Connection, value_id: int | None) -> str:
    if value_id is None:
        return ""
    row = con.execute("select value from StringIds where id=?", (int(value_id),)).fetchone()
    return "" if row is None else str(row[0])


def _enum_label(con: sqlite3.Connection, table: str, value_id: int | None) -> str:
    if value_id is None or not _table_exists(con, table):
        return "unknown"
    row = con.execute(f"select coalesce(label, name, id) from {table} where id=?", (int(value_id),)).fetchone()
    return "unknown" if row is None else str(row[0])


def _stage_row(con: sqlite3.Connection, stage_name: str) -> tuple[int, int, int | None]:
    row = con.execute(
        "select start,end,globalTid from NVTX_EVENTS where text=? and end is not null order by start desc limit 1",
        (stage_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing NVTX range {stage_name!r}")
    return int(row[0]), int(row[1]), None if row[2] is None else int(row[2])


def _runtime_details(con: sqlite3.Connection, start: int, end: int) -> list[dict[str, Any]]:
    if not _table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    return [
        {
            "start": int(row[0]),
            "end": int(row[1]),
            "correlation_id": None if row[2] is None else int(row[2]),
            "global_tid": None if row[3] is None else int(row[3]),
            "name": str(row[4]),
        }
        for row in con.execute(
            """
            select r.start,r.end,r.correlationId,r.globalTid,coalesce(s.value, '<unknown>')
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            left join StringIds s on s.id = r.nameId
            where r.end>? and r.start<?
            order by r.start
            """,
            (start, end),
        )
    ]


def _stage_tids(con: sqlite3.Connection, stage_name: str, start: int, end: int, stage_tid: int | None) -> set[int]:
    tids: set[int] = set()
    if stage_tid is not None:
        tids.add(stage_tid)
    if _table_exists(con, "NVTX_EVENTS"):
        prefix = stage_name.replace("step.", "") + ".%"
        for (tid,) in con.execute(
            """
            select distinct globalTid
            from NVTX_EVENTS
            where globalTid is not null and end is not null and end>? and start<? and (text=? or text like ?)
            """,
            (start, end, stage_name, prefix),
        ):
            if tid is not None:
                tids.add(int(tid))
    for row in _runtime_details(con, start, end):
        if row["global_tid"] is not None:
            tids.add(int(row["global_tid"]))
    return tids


def _samples(
    con: sqlite3.Connection,
    start: int,
    end: int,
    tids: set[int] | None,
) -> list[dict[str, Any]]:
    if not _table_exists(con, "COMPOSITE_EVENTS"):
        return []
    params: list[Any] = [start, end]
    tid_clause = ""
    if tids:
        placeholders = ",".join("?" for _ in tids)
        tid_clause = f" and globalTid in ({placeholders})"
        params.extend(sorted(tids))
    return [
        {
            "id": int(row[0]),
            "start": int(row[1]),
            "global_tid": None if row[2] is None else int(row[2]),
            "thread_state": None if row[3] is None else int(row[3]),
        }
        for row in con.execute(
            f"""
            select id,start,globalTid,threadState
            from COMPOSITE_EVENTS
            where start>=? and start<?{tid_clause}
            order by start
            """,
            params,
        )
    ]


def _stack_frames(con: sqlite3.Connection, sample_id: int, table: str = "SAMPLING_CALLCHAINS") -> list[dict[str, Any]]:
    if not _table_exists(con, table):
        return []
    rows = con.execute(
        f"""
        select c.stackDepth, c.symbol, c.module, c.kernelMode, c.specialEntry
        from {table} c
        where c.id=?
        order by c.stackDepth
        """,
        (sample_id,),
    ).fetchall()
    frames: list[dict[str, Any]] = []
    for depth, symbol_id, module_id, kernel_mode, special_entry in rows:
        symbol = _string(con, int(symbol_id))
        module = _string(con, int(module_id))
        if symbol == "[Max depth]" or module == "[Max depth]":
            continue
        frames.append(
            {
                "depth": int(depth),
                "symbol": symbol or "<unknown>",
                "module": module or "<unknown>",
                "kernel_mode": bool(kernel_mode),
                "special_entry": None if special_entry is None else int(special_entry),
            }
        )
    return frames


def _compact_symbol(symbol: str, *, limit: int = 100) -> str:
    symbol = " ".join(symbol.split())
    if len(symbol) <= limit:
        return symbol
    return symbol[: limit - 3] + "..."


def _stack_signature(frames: list[dict[str, Any]], *, max_frames: int = 5) -> str:
    if not frames:
        return "<no stack>"
    return " <- ".join(_compact_symbol(str(frame["symbol"]), limit=80) for frame in frames[:max_frames])


def _leaf_symbol(frames: list[dict[str, Any]]) -> str:
    if not frames:
        return "<no stack>"
    return _compact_symbol(str(frames[0]["symbol"]))


def _classify_stack(frames: list[dict[str, Any]]) -> str:
    symbol_text = " ".join(str(frame["symbol"]) for frame in frames).lower()
    module_text = " ".join(str(frame["module"]) for frame in frames).lower()
    text = f"{symbol_text} {module_text}"
    if not text:
        return "unsampled_or_unresolved"
    if "cudacachingallocator" in text or "allocator" in text or "malloc" in text or "free" in text:
        return "allocator_or_memory"
    if "autograd" in text or "thpfunction_apply" in text or "engine::evaluate_function" in text:
        return "pytorch_autograd_engine"
    if "cuda" in text or "cudart" in text or "libcuda" in text or "nvcuda" in text:
        return "cuda_runtime_or_driver"
    if "aten::" in text or "at::native" in text or "libtorch_cpu" in text or "libtorch_python" in text or "c10::" in text:
        return "pytorch_dispatch_or_aten"
    if "_pyeval_evalframe" in text or "pyobject" in text or "python" in text:
        return "python_interpreter_or_model_code"
    if "pthread_cond" in text or "futex" in text or "poll" in text or "nanosleep" in text or "wait" in text:
        return "os_runtime_or_thread_wait"
    if "blas_thread_server" in text or "openblas" in text or "mkl" in text:
        return "blas_or_background_thread"
    if "libnsys" in module_text or "libcupti" in module_text or "libtoolsinjection" in module_text or "nsight" in module_text:
        return "nsight_tooling"
    return "other_cpu_stack"


def _rows_from_counter(counter: Counter[str], total: int, *, value_key: str = "samples", max_rows: int = 20) -> list[dict[str, Any]]:
    return [
        {"name": name, value_key: count, "percent": _percent(float(count), float(total))}
        for name, count in counter.most_common(max_rows)
        if count > 0
    ]


def _sample_summary(con: sqlite3.Connection, samples: list[dict[str, Any]], max_stacks: int) -> dict[str, Any]:
    bucket_counter: Counter[str] = Counter()
    leaf_counter: Counter[str] = Counter()
    stack_counter: Counter[tuple[str, str]] = Counter()
    state_counter: Counter[str] = Counter()
    for sample in samples:
        frames = _stack_frames(con, int(sample["id"]))
        bucket = _classify_stack(frames)
        bucket_counter[bucket] += 1
        leaf_counter[_leaf_symbol(frames)] += 1
        stack_counter[(bucket, _stack_signature(frames))] += 1
        state_counter[_enum_label(con, "ENUM_SAMPLING_THREAD_STATE", sample["thread_state"])] += 1

    total = len(samples)
    stack_rows = [
        {
            "bucket": bucket,
            "signature": signature,
            "samples": count,
            "percent": _percent(float(count), float(total)),
        }
        for (bucket, signature), count in stack_counter.most_common(max_stacks)
    ]
    return {
        "sample_count": total,
        "bucket_rows": _rows_from_counter(bucket_counter, total, max_rows=max_stacks),
        "leaf_rows": _rows_from_counter(leaf_counter, total, max_rows=max_stacks),
        "thread_state_rows": _rows_from_counter(state_counter, total, max_rows=max_stacks),
        "stack_rows": stack_rows,
    }


def _osrt_events(
    con: sqlite3.Connection,
    start: int,
    end: int,
    tids: set[int] | None,
) -> list[dict[str, Any]]:
    if not _table_exists(con, "OSRT_API"):
        return []
    params: list[Any] = [start, end]
    tid_clause = ""
    if tids:
        placeholders = ",".join("?" for _ in tids)
        tid_clause = f" and o.globalTid in ({placeholders})"
        params.extend(sorted(tids))
    return [
        {
            "start": int(row[0]),
            "end": int(row[1]),
            "global_tid": None if row[2] is None else int(row[2]),
            "name": str(row[3]),
        }
        for row in con.execute(
            f"""
            select o.start,o.end,o.globalTid,coalesce(s.value, '<unknown>')
            from OSRT_API o
            left join StringIds s on s.id = o.nameId
            where o.end>? and o.start<?{tid_clause}
            order by o.start
            """,
            params,
        )
    ]


def _group_overlap_rows(
    events: list[dict[str, Any]],
    start: int,
    end: int,
    total_ms: float,
    *,
    key: str = "name",
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, int] = defaultdict(int)
    for event in events:
        grouped[str(event[key])] += _clip_overlap_ns(start, end, int(event["start"]), int(event["end"]))
    rows = [
        {"name": name, "milliseconds": _ms(ns), "percent_of_gap": _percent(_ms(ns), total_ms)}
        for name, ns in grouped.items()
        if ns > 0
    ]
    return sorted(rows, key=lambda row: row["milliseconds"], reverse=True)[:max_rows]


def _runtime_overlap_rows(
    runtime: list[dict[str, Any]],
    start: int,
    end: int,
    total_ms: float,
    *,
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    return _group_overlap_rows(runtime, start, end, total_ms, max_rows=max_rows)


def _scheduler_intervals(con: sqlite3.Connection, start: int, end: int, tids: set[int]) -> list[dict[str, Any]]:
    if not tids or not _table_exists(con, "SCHED_EVENTS"):
        return []
    placeholders = ",".join("?" for _ in tids)
    rows = list(
        con.execute(
            f"""
            select start,isSchedIn,globalTid,threadState,threadBlock
            from SCHED_EVENTS
            where globalTid in ({placeholders}) and start>=? and start<=?
            order by globalTid,start
            """,
            (*sorted(tids), start, end),
        )
    )
    for tid in sorted(tids):
        previous = con.execute(
            """
            select start,isSchedIn,globalTid,threadState,threadBlock
            from SCHED_EVENTS
            where globalTid=? and start<?
            order by start desc
            limit 1
            """,
            (tid, start),
        ).fetchone()
        if previous is not None:
            rows.append(previous)
    rows.sort(key=lambda row: (int(row[2]), int(row[0])))

    intervals: list[dict[str, Any]] = []
    open_out: dict[int, tuple[int, int | None, int | None]] = {}
    for event_start, is_sched_in, tid, thread_state, thread_block in rows:
        event_start = int(event_start)
        tid = int(tid)
        if int(is_sched_in) == 0:
            open_out[tid] = (
                event_start,
                None if thread_state is None else int(thread_state),
                None if thread_block is None else int(thread_block),
            )
        elif tid in open_out:
            out_start, out_state, out_block = open_out.pop(tid)
            if event_start > out_start:
                block_label = _enum_label(con, "ENUM_SCHEDULING_THREAD_BLOCK", out_block)
                state_label = _enum_label(con, "ENUM_SAMPLING_THREAD_STATE", out_state)
                intervals.append(
                    {
                        "start": max(start, out_start),
                        "end": min(end, event_start),
                        "global_tid": tid,
                        "name": f"{block_label}/{state_label}",
                    }
                )
    for tid, (out_start, out_state, out_block) in open_out.items():
        if end > out_start:
            block_label = _enum_label(con, "ENUM_SCHEDULING_THREAD_BLOCK", out_block)
            state_label = _enum_label(con, "ENUM_SAMPLING_THREAD_STATE", out_state)
            intervals.append(
                {
                    "start": max(start, out_start),
                    "end": end,
                    "global_tid": tid,
                    "name": f"{block_label}/{state_label}",
                }
            )
    return intervals


def _scheduler_event_count(con: sqlite3.Connection, start: int, end: int, tids: set[int]) -> int:
    if not tids or not _table_exists(con, "SCHED_EVENTS"):
        return 0
    placeholders = ",".join("?" for _ in tids)
    return int(
        con.execute(
            f"select count(*) from SCHED_EVENTS where globalTid in ({placeholders}) and start>=? and start<?",
            (*sorted(tids), start, end),
        ).fetchone()[0]
    )


def _dominant_name(rows: list[dict[str, Any]], name_key: str = "name", value_key: str = "samples") -> str:
    if not rows:
        return "-"
    row = rows[0]
    return f"{row[name_key]} ({row[value_key]})"


def _gap_root_hint(
    sample_summary: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    osrt_rows: list[dict[str, Any]],
    scheduler_rows: list[dict[str, Any]],
    sync_overlap_ms: float,
) -> str:
    if sample_summary["bucket_rows"]:
        return _dominant_name(sample_summary["bucket_rows"])
    if sync_overlap_ms > 0.0:
        return "cuda_sync_wait_no_cpu_sample"
    if scheduler_rows:
        return f"off_cpu_wait_no_cpu_sample: {scheduler_rows[0]['name']}"
    if osrt_rows:
        return f"os_runtime_no_cpu_sample: {osrt_rows[0]['name']}"
    if runtime_rows:
        return f"cuda_api_no_cpu_sample: {runtime_rows[0]['name']}"
    return "below_cpu_sample_resolution"


def _summarize_gap(
    con: sqlite3.Connection,
    gap: dict[str, Any],
    *,
    stage_start: int,
    stage_end: int,
    stage_tids: set[int],
    runtime: list[dict[str, Any]],
    sync_intervals: list[tuple[int, int]],
    osrt_events: list[dict[str, Any]],
    scheduler_intervals: list[dict[str, Any]],
    max_stacks: int,
) -> dict[str, Any]:
    gap_start = stage_start + int(round(float(gap["start_offset_milliseconds"]) * 1_000_000.0))
    gap_end = stage_start + int(round(float(gap["end_offset_milliseconds"]) * 1_000_000.0))
    gap_start = max(stage_start, min(gap_start, stage_end))
    gap_end = max(gap_start, min(gap_end, stage_end))
    gap_ms = _ms(gap_end - gap_start)

    samples = _samples(con, gap_start, gap_end, stage_tids)
    sample_scope = "stage_submission_threads"
    if not samples:
        samples = _samples(con, gap_start, gap_end, None)
        sample_scope = "all_process_threads_fallback" if samples else "below_cpu_sample_resolution"
    sample_summary = _sample_summary(con, samples, max_stacks=max_stacks)
    runtime_rows = _runtime_overlap_rows(runtime, gap_start, gap_end, gap_ms, max_rows=8)
    osrt_rows = _group_overlap_rows(osrt_events, gap_start, gap_end, gap_ms, max_rows=8)
    scheduler_rows = _group_overlap_rows(scheduler_intervals, gap_start, gap_end, gap_ms, max_rows=8)
    sync_overlap_ms = _ms(_clip_sum_ns(gap_start, gap_end, sync_intervals))
    root_hint = _gap_root_hint(sample_summary, runtime_rows, osrt_rows, scheduler_rows, sync_overlap_ms)

    enriched = dict(gap)
    enriched.update(
        {
            "start_ns": gap_start,
            "end_ns": gap_end,
            "sample_scope": sample_scope,
            "cpu_sample_count": sample_summary["sample_count"],
            "dominant_cpu_bucket": root_hint,
            "top_cpu_leaf": _dominant_name(sample_summary["leaf_rows"]) if sample_summary["leaf_rows"] else "no_cpu_sample_in_gap",
            "top_osrt_api": "-" if not osrt_rows else f"{osrt_rows[0]['name']} ({osrt_rows[0]['milliseconds']:.4f} ms)",
            "top_cuda_api": "-" if not runtime_rows else f"{runtime_rows[0]['name']} ({runtime_rows[0]['milliseconds']:.4f} ms)",
            "scheduler_context_switch_count": _scheduler_event_count(con, gap_start, gap_end, stage_tids),
            "scheduler_offcpu_overlap_milliseconds": sum(row["milliseconds"] for row in scheduler_rows),
            "sync_overlap_milliseconds": sync_overlap_ms,
            "cpu_sample_buckets": sample_summary["bucket_rows"],
            "cpu_stack_signatures": sample_summary["stack_rows"],
            "cuda_api_overlap": runtime_rows,
            "osrt_overlap": osrt_rows,
            "scheduler_offcpu_overlap": scheduler_rows,
            "root_cause_hint": root_hint,
        }
    )
    return enriched


def summarize_cpu_gaps(sqlite_path: Path, *, max_gap_rows: int = 50, max_stacks: int = 20) -> dict[str, Any]:
    con = sqlite3.connect(str(sqlite_path))
    stages: list[dict[str, Any]] = []
    for stage_name in ("step.forward", "step.backward"):
        stage_start, stage_end, stage_tid = _stage_row(con, stage_name)
        total_ms = _ms(stage_end - stage_start)
        stage_tids = _stage_tids(con, stage_name, stage_start, stage_end, stage_tid)

        runtime_simple = _runtime_rows(con, stage_start, stage_end)
        runtime_corr = {corr for _, _, corr in runtime_simple if corr is not None}
        kernel_events = _kernel_events(con, runtime_corr)
        memcpy_intervals = _correlated_intervals(con, "CUPTI_ACTIVITY_KIND_MEMCPY", runtime_corr)
        sync = _sync_intervals(con, stage_start, stage_end)
        ranges = _fetch_ranges(con, stage_name.replace("step.", "") + ".%")
        base = summarize_stage(con, stage_name)
        runtime = _runtime_details(con, stage_start, stage_end)
        osrt = _osrt_events(con, stage_start, stage_end, stage_tids)
        scheduler = _scheduler_intervals(con, stage_start, stage_end, stage_tids)
        base_gaps = _no_kernel_gaps(
            stage_start=stage_start,
            stage_end=stage_end,
            total_ms=total_ms,
            stage_name=stage_name,
            ranges=ranges,
            runtime=runtime_simple,
            sync_intervals=sync,
            kernel_events=kernel_events,
            memcpy_intervals=memcpy_intervals,
        )
        enriched_gaps = [
            _summarize_gap(
                con,
                gap,
                stage_start=stage_start,
                stage_end=stage_end,
                stage_tids=stage_tids,
                runtime=runtime,
                sync_intervals=sync,
                osrt_events=osrt,
                scheduler_intervals=scheduler,
                max_stacks=max_stacks,
            )
            for gap in base_gaps
        ]

        gap_start_end = [(int(gap["start_ns"]), int(gap["end_ns"])) for gap in enriched_gaps]
        samples_in_gaps = []
        for gap_start, gap_end in gap_start_end:
            samples_in_gaps.extend(_samples(con, gap_start, gap_end, stage_tids))
        sample_summary = _sample_summary(con, samples_in_gaps, max_stacks=max_stacks)
        osrt_rows = _group_overlap_rows(osrt, stage_start, stage_end, total_ms, max_rows=max_stacks)
        runtime_rows = _runtime_overlap_rows(runtime, stage_start, stage_end, total_ms, max_rows=max_stacks)
        scheduler_rows = _group_overlap_rows(scheduler, stage_start, stage_end, total_ms, max_rows=max_stacks)

        stages.append(
            {
                "stage": stage_name,
                "total_milliseconds": total_ms,
                "stage_thread_id": stage_tid,
                "analyzed_thread_ids": sorted(stage_tids),
                "thread_scope": "NVTX stage thread plus CUDA runtime submission threads",
                "timeline": {
                    "stage_breakdown": base["stage_breakdown"],
                    "host_api_breakdown": base["host_api_breakdown"],
                    "operation_kernel_time": base["operation_kernel_time"],
                    "operation_cuda_api_time": base["operation_cuda_api_time"],
                    "top_kernels": base["top_kernels"],
                },
                "gpu_no_kernel_gaps": {
                    "total_milliseconds": base["gpu_no_kernel_gaps"]["total_milliseconds"],
                    "percent_of_stage": _percent(base["gpu_no_kernel_gaps"]["total_milliseconds"], total_ms),
                    "gap_count": len(enriched_gaps),
                    "rows": sorted(enriched_gaps, key=lambda row: row["gap_milliseconds"], reverse=True)[:max_gap_rows],
                },
                "cpu_samples_during_gaps": sample_summary,
                "cuda_api_overlap_stage": runtime_rows,
                "osrt_overlap_stage": osrt_rows,
                "scheduler_offcpu_overlap_stage": scheduler_rows,
            }
        )
    return {
        "source": str(sqlite_path),
        "mode": "nsight_systems_cpu_gap_debug",
        "notes": [
            "CPU sample percentages are sampling shares on stage submission threads, not exact elapsed-time shares.",
            "Small gaps can be below the CPU sampling period; those rows still keep CUDA API, OSRT, scheduler, and enclosing NVTX attribution.",
            "Use the regular Nsight table.md/profile.json for low-overhead end-to-end timing truth.",
        ],
        "stages": stages,
    }


def _emit_rows(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Nsight CPU Gap Debug: {report['source']}", ""]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    for stage in report["stages"]:
        gap_total = stage["gpu_no_kernel_gaps"]["total_milliseconds"]
        lines += [
            f"## {stage['stage']}",
            "",
            f"- Stage total: {stage['total_milliseconds']:.4f} ms",
            f"- GPU no-kernel gap total: {gap_total:.4f} ms ({stage['gpu_no_kernel_gaps']['percent_of_stage']:.2f}% of stage)",
            f"- Gap count: {stage['gpu_no_kernel_gaps']['gap_count']}",
            f"- Thread scope: {stage['thread_scope']} ({len(stage['analyzed_thread_ids'])} tids)",
            "",
            "### CPU Sample Buckets During Gaps",
            "",
        ]
        _emit_rows(
            lines,
            ["Bucket", "Samples", "% samples"],
            [
                [row["name"], str(row["samples"]), f"{row['percent']:.2f}%"]
                for row in stage["cpu_samples_during_gaps"]["bucket_rows"]
            ],
        )

        lines += ["### OS Runtime API Overlap In Stage", ""]
        _emit_rows(
            lines,
            ["OSRT API", "overlap ms", "summed % stage"],
            [
                [row["name"], f"{row['milliseconds']:.4f}", f"{row['percent_of_gap']:.2f}%"]
                for row in stage["osrt_overlap_stage"]
            ],
        )

        lines += ["### CUDA Runtime API Overlap In Stage", ""]
        _emit_rows(
            lines,
            ["CUDA API", "overlap ms", "summed % stage"],
            [
                [row["name"], f"{row['milliseconds']:.4f}", f"{row['percent_of_gap']:.2f}%"]
                for row in stage["cuda_api_overlap_stage"]
            ],
        )

        lines += ["### Scheduler Off-CPU Overlap In Stage", ""]
        _emit_rows(
            lines,
            ["Block/state", "overlap ms", "summed % stage"],
            [
                [row["name"], f"{row['milliseconds']:.4f}", f"{row['percent_of_gap']:.2f}%"]
                for row in stage["scheduler_offcpu_overlap_stage"]
            ],
        )

        lines += ["### Largest GPU No-Kernel Gaps", ""]
        _emit_rows(
            lines,
            [
                "offset ms",
                "gap ms",
                "% stage",
                "enclosing NVTX",
                "previous kernel",
                "next kernel",
                "CPU bucket",
                "top CPU leaf",
                "top CUDA API",
                "top OSRT",
                "ctxsw",
            ],
            [
                [
                    f"{row['start_offset_milliseconds']:.4f}",
                    f"{row['gap_milliseconds']:.4f}",
                    f"{row['percent']:.2f}%",
                    f"`{row['enclosing_nvtx']}`",
                    f"`{row['previous_kernel']}`",
                    f"`{row['next_kernel']}`",
                    row["dominant_cpu_bucket"],
                    f"`{row['top_cpu_leaf']}`",
                    row["top_cuda_api"],
                    row["top_osrt_api"],
                    str(row["scheduler_context_switch_count"]),
                ]
                for row in stage["gpu_no_kernel_gaps"]["rows"]
            ],
        )

        lines += ["### Top CPU Stack Signatures During Gaps", ""]
        _emit_rows(
            lines,
            ["Bucket", "Samples", "% samples", "Stack leaf-to-root"],
            [
                [row["bucket"], str(row["samples"]), f"{row['percent']:.2f}%", f"`{row['signature']}`"]
                for row in stage["cpu_samples_during_gaps"]["stack_rows"]
            ],
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-gap-rows", type=int, default=50)
    parser.add_argument("--max-stacks", type=int, default=20)
    args = parser.parse_args()

    report = summarize_cpu_gaps(args.sqlite_path, max_gap_rows=args.max_gap_rows, max_stacks=args.max_stacks)
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
