#!/usr/bin/env python3
"""Postprocess an Nsight Systems SQLite export for LoRA-SFT profiling.

Run the workload with `scripts/profile_lora.py --timing-mode profile` or
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
import re
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


def _fmt_mib(value: int | float) -> str:
    return f"{float(value) / (1024.0 ** 2):.2f}"


def _source_profile_candidates(path: Path) -> list[Path]:
    if path.is_dir():
        return [path / "profile.json", *sorted(path.glob("*_profile.json"))]
    return [path]


def _load_source_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    for candidate in _source_profile_candidates(path):
        if candidate.exists() and candidate.is_file():
            profile = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(profile, dict):
                profile["_source_profile_path"] = str(candidate)
                return profile
    return {}


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


def _display_operation_name(name: str) -> str:
    direction = ""
    body = name
    if body.startswith("forward."):
        direction = "FWD "
        body = body[len("forward.") :]
    elif body.startswith("backward."):
        direction = "BWD "
        body = body[len("backward.") :]

    replacements = {
        "base_frozen_asymgemm": "base AsymGEMM",
        "base_dx_asymgemm": "base dX AsymGEMM",
        "grouped_base_frozen_asymgemm": "grouped base AsymGEMM",
        "grouped_base_dx_asymgemm": "grouped base dX AsymGEMM",
        "grouped_base_torch": "grouped base torch",
        "grouped_base_dx_torch": "grouped base dX torch",
        "routed_expert": "routed MoE",
        "shared_expert": "shared MoE",
        "gate_base": "gate base",
        "up_base": "up base",
        "down_base": "down base",
        "gate_proj": "gate proj",
        "up_proj": "up proj",
        "down_proj": "down proj",
        "q_proj": "q proj",
        "k_proj": "k proj",
        "v_proj": "v proj",
        "o_proj": "o proj",
        "scores_matmul": "scores matmul",
        "value_matmul": "value matmul",
        "causal_mask": "causal mask",
        "route_metadata": "route metadata",
        "pack_tokens": "pack tokens",
        "scatter_combine": "scatter combine",
        "residual_add": "residual add",
        "combine_shared_routed": "combine shared+routed",
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    body = re.sub(r"\blayers\.(\d+)\b", r"layer \1", body)
    body = body.replace(".", " / ").replace("_", " ")
    body = re.sub(r"\s+", " ", body).strip()
    return direction + body


def _display_kernel_name(name: str) -> str:
    if name in {"<stage_start>", "<stage_end>", "<unknown>"}:
        return name

    lower = name.lower()
    if "asym_gemm::sm100_bf16" in lower:
        dims = re.findall(r"\(unsigned int\)(\d+)", name)
        shape = f" M={dims[0]} N={dims[1]} K={dims[2]}" if len(dims) >= 3 else ""
        return f"AsymGEMM SM100 BF16{shape}"
    if "asym_gemm::sm90" in lower:
        dims = re.findall(r"\(unsigned int\)(\d+)", name)
        shape = f" M={dims[0]} N={dims[1]} K={dims[2]}" if len(dims) >= 3 else ""
        return f"AsymGEMM SM90{shape}"
    if "asym_gemm::" in lower:
        return "AsymGEMM kernel"
    if "cutlass" in lower or "nvjet" in lower:
        return "Torch/CUTLASS GEMM kernel"
    if "cublas" in lower:
        return "cuBLAS/cuBLASLt GEMM kernel"
    if "softmax" in lower:
        return "Torch softmax kernel"
    if "layer_norm" in lower or "layernorm" in lower:
        return "Torch layernorm kernel"
    if "indexfunc" in lower or "indexselect" in lower or "scatter" in lower or "gather" in lower:
        return "Torch index/scatter kernel"
    if "catarraybatchedcopy" in lower or "copy" in lower:
        return "Torch copy/cast kernel"
    if (
        "vectorized_elementwise" in lower
        or "unrolled_elementwise" in lower
        or "elementwise_kernel" in lower
    ):
        if "fillfunctor" in lower:
            return "Torch fill elementwise kernel"
        if "add" in lower:
            return "Torch add elementwise kernel"
        if "mul" in lower:
            return "Torch mul elementwise kernel"
        if "masked_fill" in lower:
            return "Torch masked-fill elementwise kernel"
        return "Torch elementwise kernel"
    if "radixsort" in lower or "histogram" in lower:
        return "Torch routing/sort kernel"
    return "Other CUDA kernel"


def _display_kernel_rows(rows: list[dict[str, Any]], total_ms: float) -> list[dict[str, Any]]:
    values: dict[str, float] = defaultdict(float)
    for row in rows:
        values[_display_kernel_name(str(row["name"]))] += float(row["milliseconds"])
    return _rows(values, total_ms)


def _operation_kernel_class_rows(
    *,
    kernel_events: list[dict[str, Any]],
    ranges: list[tuple[int, int, str]],
    runtime_operations: dict[int, str],
    stage_name: str,
    total_ms: float,
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], float] = defaultdict(float)
    for event in kernel_events:
        start = int(event["start"])
        end = int(event["end"])
        if end <= start:
            continue
        operation = runtime_operations.get(int(event["correlation_id"]))
        if operation is None:
            operation = _enclosing_range(start, end, ranges, stage_name)
        kernel_class = _display_kernel_name(str(event["name"]))
        values[(operation, kernel_class)] += _ms(end - start)
    return [
        {
            "operation": operation,
            "kernel_class": kernel_class,
            "milliseconds": milliseconds,
            "percent": _percent(milliseconds, total_ms),
        }
        for (operation, kernel_class), milliseconds in sorted(values.items(), key=lambda item: item[1], reverse=True)
        if milliseconds > 0.0
    ]


def _runtime_operation_map(
    runtime: list[tuple[int, int, int]],
    ranges: list[tuple[int, int, str]],
    stage_name: str,
) -> dict[int, str]:
    result: dict[int, str] = {}
    for start, end, corr in runtime:
        result[int(corr)] = _enclosing_range(int(start), int(end), ranges, stage_name)
    return result


def _stage_breakdown_ms(stage: dict[str, Any], name: str) -> float:
    for row in stage["stage_breakdown"]["rows"]:
        if row["name"] == name:
            return float(row["milliseconds"])
    return 0.0


def _expert_scope(op: str) -> str:
    if "routed_expert" in op:
        return "routed "
    if "shared_expert" in op:
        return "shared "
    return ""


def _projection_scope(op: str) -> str:
    if "gate_up_lora" in op:
        return "gate/up "
    if "gate_base" in op or "gate_lora" in op or "gate_proj" in op:
        return "gate "
    if "up_base" in op or "up_lora" in op or "up_proj" in op:
        return "up "
    if "down_base" in op or "down_lora" in op or "down_proj" in op:
        return "down "
    return ""


def _attention_scope(op: str) -> str:
    if "q_proj" in op:
        return "q proj"
    if "k_proj" in op:
        return "k proj"
    if "v_proj" in op:
        return "v proj"
    if "o_proj" in op:
        return "o proj"
    if "scores_matmul" in op:
        return "scores matmul"
    if "value_matmul" in op:
        return "value matmul"
    if "softmax" in op:
        return "softmax"
    if "causal_mask" in op:
        return "causal mask"
    if "layernorm" in op:
        return "layernorm"
    if "residual_add" in op:
        return "residual add"
    return "other"


def _semantic_operation_scope(op: str) -> str:
    expert = _expert_scope(op)
    projection = _projection_scope(op)
    if projection:
        return f"{expert}{projection}".strip()
    if ".fc1." in op or op.endswith(".fc1"):
        return "fc1"
    if ".fc2." in op or op.endswith(".fc2"):
        return "fc2"
    if ".matrix." in op or op.endswith(".matrix"):
        return "matrix"
    if "attention" in op:
        return f"attention {_attention_scope(op)}"
    if "lm_head" in op:
        return "LM-head"
    if "router" in op:
        return "router"
    if "route_metadata" in op:
        return "route metadata"
    if "pack_tokens" in op:
        return "pack tokens"
    if "scatter_combine" in op:
        return "scatter/combine"
    if "final_norm" in op:
        return "final norm"
    if "moe.layernorm" in op:
        return "MoE layernorm"
    if "moe.residual_add" in op:
        return "MoE residual add"
    if "combine_shared_routed" in op:
        return "MoE combine shared+routed"
    return expert.strip()


def _scoped_label(scope: str, label: str) -> str:
    return f"{scope} {label}".strip() if scope else label


def _kernel_family(kernel: str) -> str:
    if "gemm kernel" in kernel:
        return "GEMM"
    if "copy/cast" in kernel:
        return "copy/cast"
    if "mul elementwise" in kernel or "add elementwise" in kernel or "elementwise" in kernel:
        return "elementwise"
    if "softmax" in kernel:
        return "softmax"
    if "layernorm" in kernel:
        return "layernorm"
    if "index/scatter" in kernel:
        return "index/scatter"
    if "routing/sort" in kernel:
        return "routing/sort"
    if "memcpy" in kernel:
        return "memcpy"
    return "other CUDA"


def _gap_operation_bucket(operation: str, stage_name: str) -> str:
    op = operation.lower()
    if operation == stage_name:
        if stage_name == "step.backward":
            return "backward top-level / unlabeled ops"
        return "forward top-level / unlabeled ops"

    expert = _expert_scope(op)
    projection = _projection_scope(op)
    scope = _semantic_operation_scope(op)

    if "lora" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), "LoRA")
    if "grouped_base_dx_asymgemm" in op or "base_dx_asymgemm" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), "base dX AsymGEMM")
    if "grouped_base_frozen_asymgemm" in op or "base_frozen_asymgemm" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), "base AsymGEMM")
    if "activation_silu" in op or "silu_mul" in op:
        return f"{expert}activation/silu".strip()
    if "attention" in op:
        return f"attention {_attention_scope(op)}"
    if "router" in op:
        return "router"
    if "route_metadata" in op:
        return "route metadata"
    if "pack_tokens" in op:
        return "pack tokens"
    if "scatter_combine" in op:
        return "scatter/combine"
    if "lm_head" in op:
        return "LM-head"
    if "final_norm" in op:
        return "final norm"
    if "moe.layernorm" in op:
        return "MoE layernorm"
    if "moe.residual_add" in op:
        return "MoE residual add"
    if "combine_shared_routed" in op:
        return "MoE combine shared+routed"
    return _display_operation_name(operation).removeprefix("FWD ").removeprefix("BWD ")


def _top_level_gap_operation_bucket(row: dict[str, Any], stage_name: str) -> str:
    prev_kernel = _display_kernel_name(str(row["previous_kernel"]))
    next_kernel = _display_kernel_name(str(row["next_kernel"]))
    stage = "backward" if stage_name == "step.backward" else "forward"
    return f"{stage} unlabeled kernel chain ({prev_kernel} -> {next_kernel})"


def _stage_direction(stage_name: str) -> str:
    return "backward" if stage_name == "step.backward" else "forward"


def _canonical_no_kernel_label(stage_name: str, bucket: str) -> str:
    stage = _stage_direction(stage_name)
    bucket = re.sub(r"^no-kernel\s+", "", bucket.strip(), flags=re.IGNORECASE)
    lower_bucket = bucket.lower()
    if lower_bucket.startswith(("forward ", "backward ")):
        return f"No-kernel {bucket}"
    return f"No-kernel {stage} {bucket}"


def _compact_no_kernel_gap_rows(
    values: dict[str, float],
    *,
    total_ms: float,
    no_kernel_ms: float,
) -> list[dict[str, Any]]:
    compacted: dict[str, float] = {}
    misc_ms = 0.0
    for name, ms in values.items():
        no_kernel_percent = _percent(ms, no_kernel_ms)
        always_keep = (
            "LoRA" in name
            or "AsymGEMM" in name
            or no_kernel_percent >= 1.0
        )
        if always_keep:
            compacted[name] = ms
        else:
            misc_ms += ms
    if misc_ms > 0.0:
        compacted["No-kernel misc small gaps"] = misc_ms

    return [
        {
            "name": name,
            "milliseconds": ms,
            "percent": _percent(ms, total_ms),
            "percent_no_kernel": _percent(ms, no_kernel_ms),
        }
        for name, ms in sorted(compacted.items(), key=lambda item: item[1], reverse=True)
        if ms > 0.0
    ]


def _no_kernel_gap_attribution_rows(
    gap_rows: list[dict[str, Any]],
    *,
    stage_name: str,
    total_ms: float,
    no_kernel_ms: float,
) -> list[dict[str, Any]]:
    values: dict[str, float] = defaultdict(float)
    for row in gap_rows:
        gap_ms = float(row["gap_milliseconds"])
        if gap_ms <= 0.0:
            continue
        enclosing = str(row["enclosing_nvtx"])
        if enclosing == stage_name:
            op_bucket = _top_level_gap_operation_bucket(row, stage_name)
        else:
            op_bucket = _gap_operation_bucket(enclosing, stage_name)
        sync_ms = min(gap_ms, max(0.0, float(row["sync_overlap_milliseconds"])))
        remaining_ms = max(0.0, gap_ms - sync_ms)
        runtime_ms = min(remaining_ms, max(0.0, float(row["cuda_api_overlap_milliseconds"])))
        remaining_ms = max(0.0, remaining_ms - runtime_ms)

        if sync_ms > 0.0:
            values[f"No-kernel CUDA sync/wait: {op_bucket}"] += sync_ms
        if runtime_ms > 0.0:
            values[f"No-kernel CUDA runtime/API: {op_bucket}"] += runtime_ms
        if remaining_ms > 0.0:
            values[f"No-kernel host/autograd/Python: {op_bucket}"] += remaining_ms

    return _compact_no_kernel_gap_rows(values, total_ms=total_ms, no_kernel_ms=no_kernel_ms)


def _semantic_kernel_bucket(operation: str, kernel_class: str, stage_name: str) -> str:
    op = operation.lower()
    kernel = kernel_class.lower()

    expert = _expert_scope(op)
    projection = _projection_scope(op)
    scope = _semantic_operation_scope(op)
    family = _kernel_family(kernel)

    if "asymgemm" in kernel:
        suffix = "dX AsymGEMM" if "base_dx_asymgemm" in op or "grouped_base_dx_asymgemm" in op else "AsymGEMM"
        return _scoped_label(scope or f"{expert}{projection}".strip(), f"base {suffix}")

    if "grouped_base_dx_torch" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), f"base dX torch {family} kernels")

    if "grouped_base_torch" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), f"base torch {family} kernels")

    if "activation_silu" in op or "silu_mul" in op:
        return f"{expert}activation/silu kernels".strip()

    if "lora" in op:
        return _scoped_label(scope or f"{expert}{projection}".strip(), f"LoRA torch {family} kernels")

    if "attention" in op:
        return _scoped_label(scope, f"torch {family} kernels")

    if "lm_head" in op:
        return f"LM-head torch {family} kernels"

    if "router" in op:
        return f"router torch {family} kernels"

    if "route_metadata" in op:
        return f"route metadata torch {family} kernels"

    if "pack_tokens" in op:
        return f"pack tokens torch {family} kernels"

    if "scatter_combine" in op:
        return f"scatter/combine torch {family} kernels"

    if projection and "base" in op:
        return f"{expert}{projection}base support torch {family} kernels".strip()

    if "gemm kernel" in kernel:
        if "attention" in op:
            return _scoped_label(scope, "torch GEMM kernels")
        if "lm_head" in op:
            return "LM-head torch GEMM kernels"
        if "router" in op:
            return "router torch GEMM kernels"
        if stage_name == "step.backward" and operation == "step.backward":
            return "unattributed backward torch GEMM kernels (mostly LoRA)"
        if scope:
            return _scoped_label(scope, "torch GEMM kernels")
        return "other non-Asym torch GEMM kernels"

    if "silu" in op or "activation" in op or "elementwise" in kernel:
        return "activation/elementwise kernels"
    return "other CUDA kernels"


def _semantic_stage_summary(stage: dict[str, Any]) -> list[dict[str, Any]]:
    total_ms = float(stage["total_milliseconds"])
    kernel_busy_ms = _stage_breakdown_ms(stage, "cuda_kernel_busy_union")
    memcpy_ms = _stage_breakdown_ms(stage, "cuda_memcpy_union")
    no_kernel_ms = _stage_breakdown_ms(stage, "gpu_no_kernel_time")

    buckets: dict[str, float] = defaultdict(float)
    for row in stage["operation_kernel_classes"]["rows"]:
        bucket = _semantic_kernel_bucket(str(row["operation"]), str(row["kernel_class"]), str(stage["stage"]))
        buckets[bucket] += float(row["milliseconds"])

    explicit = {name: ms for name, ms in buckets.items() if name != "other CUDA kernels" and ms > 0.0}
    explicit_ms = sum(explicit.values())
    other_cuda_ms = max(0.0, kernel_busy_ms - explicit_ms)

    values = dict(explicit)
    if other_cuda_ms > 0.0:
        values["other CUDA kernels"] = other_cuda_ms
    if memcpy_ms > 0.0:
        values["CUDA memcpy / transfer"] = memcpy_ms
    no_kernel_attribution_rows = stage.get("gpu_no_kernel_gap_attribution", {}).get("rows", [])
    if no_kernel_attribution_rows:
        for row in no_kernel_attribution_rows:
            values[str(row["name"])] = float(row["milliseconds"])
    elif no_kernel_ms > 0.0:
        values["CPU/runtime/no-kernel gap"] = no_kernel_ms

    compacted: dict[str, float] = {}
    misc_ms = 0.0
    for name, ms in values.items():
        percent = (100.0 * ms / total_ms) if total_ms else 0.0
        lower_name = name.lower()
        always_keep = (
            name in {"CPU/runtime/no-kernel gap", "CUDA memcpy / transfer", "other CUDA kernels"}
            or lower_name.startswith("no-kernel ")
            or "AsymGEMM" in name
            or "LoRA" in name
            or "activation/silu" in name
            or "unattributed backward" in name
        )
        if always_keep or percent >= 0.05:
            compacted[name] = ms
        else:
            misc_ms += ms
    if misc_ms > 0.0:
        compacted["misc small CUDA kernels"] = misc_ms

    return _rows(compacted, total_ms)


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
        select k.start,k.end,k.correlationId,coalesce(s.value, '<unknown>')
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.correlationId in ({placeholders})
        order by k.start
    """
    return [
        {"start": int(start), "end": int(end), "correlation_id": int(corr), "name": str(name)}
        for start, end, corr, name in con.execute(query, tuple(correlation_ids))
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
    runtime_operations = _runtime_operation_map(runtime, ranges, stage_name)
    for r_start, r_end, text in ranges:
        if r_start < start or r_end > end:
            continue
        name = _normalize_range_name(text)
        if name is None or name == stage_name:
            continue
        corr = {corr for s, e, corr in runtime if s >= r_start and e <= r_end}
        op_api[name] += _ms(sum(e - s for s, e, corr_id in runtime if corr_id in corr))
        op_kernel[name] += _ms(_interval_union(_correlated_intervals(con, "CUPTI_ACTIVITY_KIND_KERNEL", corr)))

    no_kernel_gap_rows = _no_kernel_gaps(
        stage_start=start,
        stage_end=end,
        total_ms=total_ms,
        stage_name=stage_name,
        ranges=ranges,
        runtime=runtime,
        sync_intervals=sync_intervals,
        kernel_events=kernel_events,
        memcpy_intervals=memcpy_intervals,
    )

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
        "operation_kernel_classes": {
            "total_milliseconds": total_ms,
            "rows": _operation_kernel_class_rows(
                kernel_events=kernel_events,
                ranges=ranges,
                runtime_operations=runtime_operations,
                stage_name=stage_name,
                total_ms=total_ms,
            ),
        },
        "operation_cuda_api_time": {
            "total_milliseconds": total_ms,
            "rows": _rows(op_api, total_ms),
        },
        "top_kernels": _rows(_kernel_name_rows(con, runtime_corr), total_ms)[:20],
        "gpu_no_kernel_gap_attribution": {
            "total_milliseconds": gpu_idle_or_no_kernel_ms,
            "rows": _no_kernel_gap_attribution_rows(
                no_kernel_gap_rows,
                stage_name=stage_name,
                total_ms=total_ms,
                no_kernel_ms=gpu_idle_or_no_kernel_ms,
            ),
        },
        "gpu_no_kernel_gaps": {
            "total_milliseconds": gpu_idle_or_no_kernel_ms,
            "rows": no_kernel_gap_rows,
        },
    }
    summary["semantic_stage_summary"] = {
        "total_milliseconds": total_ms,
        "rows": _semantic_stage_summary(summary),
    }
    return summary


def _semantic_rows(stage: dict[str, Any]) -> list[dict[str, Any]]:
    semantic = stage.get("semantic_stage_summary", {})
    rows = semantic.get("rows") if isinstance(semantic, dict) else None
    if isinstance(rows, list):
        return rows
    return _semantic_stage_summary(stage)


def _stage_by_name(report: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    for stage in report.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage") == stage_name:
            return stage
    return None


def _compact_semantic_text(text: str) -> str:
    compact = text.lower()
    compact = re.sub(r"^(forward|backward)\.", "", compact)
    compact = re.sub(r"\blayers\.\d+\.", "", compact)
    compact = compact.replace("/", " ").replace(".", " ")
    compact = re.sub(r"[_\s]+", " ", compact).strip()
    return compact


def _semantic_projection(compact: str) -> str:
    patterns = [
        ("q proj", "q_proj"),
        ("k proj", "k_proj"),
        ("v proj", "v_proj"),
        ("o proj", "o_proj"),
        ("gate proj", "gate_proj"),
        ("up proj", "up_proj"),
        ("down proj", "down_proj"),
        ("mlp gate", "gate_proj"),
        ("mlp up", "up_proj"),
        ("mlp down", "down_proj"),
        ("gate base", "gate"),
        ("up base", "up"),
        ("down base", "down"),
        ("gate up lora", "gate_up"),
        ("gate lora", "gate"),
        ("up lora", "up"),
        ("down lora", "down"),
        ("fc1", "fc1"),
        ("fc2", "fc2"),
        ("matrix", "matrix"),
    ]
    for pattern, projection in patterns:
        if pattern in compact:
            return projection
    return ""


def _semantic_operation(compact: str) -> str:
    if "attention" in compact and (
        "base asymgemm" in compact or "base frozen asymgemm" in compact or "base dx asymgemm" in compact
    ):
        return "base_torch"
    if "base asymgemm" in compact or "base frozen asymgemm" in compact or "base dx asymgemm" in compact:
        return "base_asymgemm"
    if "grouped base asymgemm" in compact or "grouped base frozen asymgemm" in compact or "grouped base dx asymgemm" in compact:
        return "base_asymgemm"
    if "base torch" in compact or "grouped base torch" in compact or "grouped base dx torch" in compact:
        return "base_torch"
    if "q proj base" in compact or "k proj base" in compact or "v proj base" in compact or "o proj base" in compact:
        return "base_torch"
    if "add cast scale" in compact:
        return "add_cast_scale"
    if "base lora add" in compact:
        return "base_lora_add"
    if "lora a" in compact:
        return "lora_A"
    if "lora b" in compact:
        return "lora_B"
    if "lora" in compact:
        return "lora"
    if "silu mul activation" in compact or "activation silu mul" in compact:
        return "silu_mul_activation"
    if "relu" in compact:
        return "relu_activation"
    for pattern, operation in [
        ("scores matmul", "scores_matmul"),
        ("value matmul", "value_matmul"),
        ("causal mask", "causal_mask"),
        ("softmax", "softmax"),
        ("layernorm", "layernorm"),
        ("residual add", "residual_add"),
        ("route metadata", "route_metadata"),
        ("pack tokens", "pack_tokens"),
        ("scatter combine", "scatter_combine"),
        ("combine shared routed", "combine_shared_routed"),
        ("forward sft", "forward_sft"),
        ("kt lora update", "kt_lora_update"),
        ("cross entropy", "cross_entropy"),
        ("mse", "mse"),
    ]:
        if pattern in compact:
            return operation
    if "final norm" in compact:
        return "final_norm"
    if "lm head" in compact:
        return "lm_head"
    if "embedding" in compact:
        return "embeddings"
    if "router" in compact:
        return "router"
    if "loss" in compact:
        return "loss"
    return ""


def _semantic_leaf_key(text: str, *, stage_name: str | None = None) -> str:
    compact = _compact_semantic_text(text)
    projection = _semantic_projection(compact)
    operation = _semantic_operation(compact)

    if compact in {"step forward", "step backward"}:
        stage = "backward" if compact.endswith("backward") else "forward"
        return f"unattributed.{stage}_top_level"

    if "routed expert" in compact or "shared expert" in compact:
        domain = "routed_expert" if "routed expert" in compact else "shared_expert"
        parts = [domain]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if "attention" in compact:
        parts = ["attention"]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if "mlp" in compact or projection in {"gate_proj", "up_proj", "down_proj"}:
        parts = ["mlp"]
        if projection:
            parts.append(projection)
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if projection in {"gate", "up", "down"}:
        parts = ["mlp", f"{projection}_proj"]
        if operation:
            parts.append(operation)
        return ".".join(parts)

    if projection:
        parts = [projection]
        if operation and operation != projection:
            parts.append(operation)
        return ".".join(parts)

    if operation:
        return operation

    if stage_name:
        stage = "backward" if stage_name == "step.backward" else "forward"
        return f"unattributed.{stage}_top_level"
    return re.sub(r"[^a-z0-9]+", "_", compact).strip("_") or "unattributed"


def _gap_semantic_key(row_name: str, stage_name: str) -> tuple[str, str | None]:
    bucket = row_name.split(": ", 1)[1] if ": " in row_name else row_name
    lower_bucket = bucket.lower()
    if "unlabeled kernel chain" in lower_bucket:
        label = _canonical_no_kernel_label(stage_name, bucket)
        key = "no_kernel." + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return key, label

    key = _semantic_leaf_key(bucket, stage_name=stage_name)
    if key.startswith("unattributed."):
        label = _canonical_no_kernel_label(stage_name, bucket)
        key = "no_kernel." + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return key, label
    return key, None


def _semantic_leaf_label(key: str) -> str:
    if key.startswith("no_kernel."):
        body = key[len("no_kernel.") :].replace("_", " ")
        return body[:1].upper() + body[1:]
    labels = {
        "mlp": "MLP",
        "attention": "Attention",
        "routed_expert": "Routed expert",
        "shared_expert": "Shared expert",
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "base_asymgemm": "base AsymGEMM",
        "base_torch": "base torch",
        "lora_A": "LoRA A",
        "lora_B": "LoRA B",
        "lora": "LoRA",
        "add_cast_scale": "add/cast/scale",
        "base_lora_add": "base+LoRA add",
        "silu_mul_activation": "SiLU * up activation",
        "relu_activation": "ReLU activation",
        "scores_matmul": "scores matmul",
        "value_matmul": "value matmul",
        "causal_mask": "causal mask",
        "softmax": "softmax",
        "layernorm": "LayerNorm",
        "residual_add": "residual add",
        "route_metadata": "route metadata",
        "pack_tokens": "pack tokens",
        "scatter_combine": "scatter/combine",
        "combine_shared_routed": "combine shared+routed",
        "forward_sft": "KT forward SFT",
        "kt_lora_update": "KT LoRA update",
        "cross_entropy": "cross entropy",
        "mse": "MSE",
        "final_norm": "final norm",
        "lm_head": "LM head",
        "embeddings": "embeddings",
        "router": "router",
        "loss": "loss",
        "cuda": "CUDA",
        "memcpy_transfer": "memcpy / transfer",
        "unattributed": "unattributed",
        "forward_top_level": "forward top-level",
        "backward_top_level": "backward top-level",
    }
    return " ".join(labels.get(part, part.replace("_", " ")) for part in key.split("."))


def _kernel_class_summary(classes: set[str]) -> str:
    if not classes:
        return "-"
    ordered = sorted(classes)
    visible = ordered[:3]
    suffix = f" +{len(ordered) - len(visible)}" if len(ordered) > len(visible) else ""
    return ", ".join(visible) + suffix


def _summary_entry(rows: dict[str, dict[str, Any]], key: str, label: str | None = None) -> dict[str, Any]:
    if key not in rows:
        rows[key] = {
            "key": key,
            "label": label or _semantic_leaf_label(key),
            "forward_gpu_ms": 0.0,
            "forward_gap_ms": 0.0,
            "backward_gpu_ms": 0.0,
            "backward_gap_ms": 0.0,
            "saved_activation_bytes": 0,
            "kernel_classes": set(),
        }
    elif label and rows[key]["label"] == _semantic_leaf_label(key):
        rows[key]["label"] = label
    return rows[key]


def _memory_profile_for_summary(report: dict[str, Any]) -> dict[str, Any]:
    for name in ("memory_profile", "source_profile"):
        profile = report.get(name)
        if not isinstance(profile, dict):
            continue
        attribution = profile.get("memory_attribution")
        if isinstance(attribution, dict):
            saved = attribution.get("saved_activations")
            if isinstance(saved, dict) and isinstance(saved.get("rows"), list):
                return profile
    return {}


def _semantic_timing_memory_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for stage in report.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage", ""))
        prefix = "backward" if stage_name == "step.backward" else "forward"
        for row in stage.get("operation_kernel_classes", {}).get("rows", []):
            key = _semantic_leaf_key(str(row.get("operation", "")), stage_name=stage_name)
            entry = _summary_entry(rows, key)
            entry[f"{prefix}_gpu_ms"] += float(row.get("milliseconds", 0.0))
            entry["kernel_classes"].add(str(row.get("kernel_class", "-")))

        memcpy_ms = _stage_breakdown_ms(stage, "cuda_memcpy_union")
        if memcpy_ms > 0.0:
            entry = _summary_entry(rows, "cuda.memcpy_transfer")
            entry[f"{prefix}_gpu_ms"] += memcpy_ms
            entry["kernel_classes"].add("CUDA memcpy / transfer")

        for row in stage.get("gpu_no_kernel_gap_attribution", {}).get("rows", []):
            key, label = _gap_semantic_key(str(row.get("name", "")), stage_name)
            entry = _summary_entry(rows, key, label)
            entry[f"{prefix}_gap_ms"] += float(row.get("milliseconds", 0.0))

    profile = _memory_profile_for_summary(report)
    attribution = profile.get("memory_attribution") if isinstance(profile, dict) else None
    saved = attribution.get("saved_activations") if isinstance(attribution, dict) else None
    saved_rows = saved.get("rows") if isinstance(saved, dict) else None
    if isinstance(saved_rows, list):
        for row in saved_rows:
            key = _semantic_leaf_key(str(row.get("owner", "")))
            entry = _summary_entry(rows, key)
            entry["saved_activation_bytes"] += int(row.get("unique_bytes", 0))

    result: list[dict[str, Any]] = []
    for entry in rows.values():
        total_ms = (
            float(entry["forward_gpu_ms"])
            + float(entry["forward_gap_ms"])
            + float(entry["backward_gpu_ms"])
            + float(entry["backward_gap_ms"])
        )
        saved_bytes = int(entry["saved_activation_bytes"])
        if total_ms <= 0.0 and saved_bytes <= 0:
            continue
        result.append(
            {
                "key": entry["key"],
                "label": entry["label"],
                "forward_gpu_ms": float(entry["forward_gpu_ms"]),
                "forward_gap_ms": float(entry["forward_gap_ms"]),
                "backward_gpu_ms": float(entry["backward_gpu_ms"]),
                "backward_gap_ms": float(entry["backward_gap_ms"]),
                "total_milliseconds": total_ms,
                "saved_activation_bytes": saved_bytes,
                "saved_activation_mib": saved_bytes / (1024.0**2),
                "kernel_classes": _kernel_class_summary(entry["kernel_classes"]),
            }
        )
    total_saved_activation_bytes = sum(int(row["saved_activation_bytes"]) for row in result)
    for row in result:
        row["saved_activation_percent"] = _percent(int(row["saved_activation_bytes"]), total_saved_activation_bytes)
    return sorted(result, key=lambda row: (float(row["total_milliseconds"]), int(row["saved_activation_bytes"])), reverse=True)


def _semantic_stage_rows(report: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    stage = _stage_by_name(report, stage_name)
    if stage is None:
        return []
    total_ms = float(stage.get("total_milliseconds", 0.0))
    prefix = "backward" if stage_name == "step.backward" else "forward"
    result: list[dict[str, Any]] = []
    for row in _semantic_timing_memory_rows(report):
        gpu_ms = float(row[f"{prefix}_gpu_ms"])
        gap_ms = float(row[f"{prefix}_gap_ms"])
        stage_ms = gpu_ms + gap_ms
        if stage_ms <= 0.0:
            continue
        result.append(
            {
                "label": row["label"],
                "gpu_ms": gpu_ms,
                "gap_ms": gap_ms,
                "total_milliseconds": stage_ms,
                "percent": _percent(stage_ms, total_ms),
                "kernel_classes": row["kernel_classes"],
            }
        )
    return sorted(result, key=lambda row: float(row["total_milliseconds"]), reverse=True)


def _fmt_summary_ms(value: float) -> str:
    return f"{value:.4f}" if value > 0.0 else "-"


def _fmt_summary_mib(value: float) -> str:
    return f"{value:.2f}" if value > 0.0 else "-"


def _timing_summary_markdown(report: dict[str, Any], stage_name: str, title: str) -> list[str]:
    stage = _stage_by_name(report, stage_name)
    if stage is None:
        return []
    lines = [
        f"## {title} Timing Summary",
        "",
        f"Total: `{float(stage['total_milliseconds']):.4f} ms`",
        "",
        "| Semantic op | GPU ms | no-kernel gap ms | total ms | % stage | Kernel classes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in _semantic_stage_rows(report, stage_name):
        lines.append(
            f"| {row['label']} | "
            f"{_fmt_summary_ms(float(row['gpu_ms']))} | "
            f"{_fmt_summary_ms(float(row['gap_ms']))} | "
            f"{row['total_milliseconds']:.4f} | "
            f"{row['percent']:.2f}% | "
            f"{row['kernel_classes']} |"
        )
    lines.append("")
    return lines


def _semantic_timing_memory_markdown(report: dict[str, Any]) -> list[str]:
    rows = _semantic_timing_memory_rows(report)
    if not rows:
        return []
    total_ms = sum(float(row["total_milliseconds"]) for row in rows)
    lines = [
        "## Semantic Timing + Memory Summary",
        "",
        "GPU ms is attributed from kernel/memcpy activity. No-kernel gap ms is attributed to the nearest semantic range when Nsight labels allow it.",
        "",
        "| Semantic op | FWD GPU ms | FWD gap ms | BWD GPU ms | BWD gap ms | Total ms | % listed | Saved act MiB | % saved GPU | Kernel classes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        total = float(row["total_milliseconds"])
        lines.append(
            f"| {row['label']} | "
            f"{_fmt_summary_ms(float(row['forward_gpu_ms']))} | "
            f"{_fmt_summary_ms(float(row['forward_gap_ms']))} | "
            f"{_fmt_summary_ms(float(row['backward_gpu_ms']))} | "
            f"{_fmt_summary_ms(float(row['backward_gap_ms']))} | "
            f"{total:.4f} | "
            f"{_percent(total, total_ms):.2f}% | "
            f"{_fmt_summary_mib(float(row['saved_activation_mib']))} | "
            f"{float(row['saved_activation_percent']):.2f}% | "
            f"{row['kernel_classes']} |"
        )
    lines.append("")
    return lines


def _top_bottlenecks_markdown(report: dict[str, Any]) -> list[str]:
    rows = _semantic_timing_memory_rows(report)
    if not rows:
        return []
    lines = ["## Top Bottlenecks", ""]

    lines.extend(["### Top Timing", "", "| Semantic op | Total ms | FWD ms | BWD ms | Saved act MiB | % saved GPU |", "|---|---:|---:|---:|---:|---:|"])
    for row in rows[:10]:
        fwd_ms = float(row["forward_gpu_ms"]) + float(row["forward_gap_ms"])
        bwd_ms = float(row["backward_gpu_ms"]) + float(row["backward_gap_ms"])
        lines.append(
            f"| {row['label']} | {row['total_milliseconds']:.4f} | "
            f"{_fmt_summary_ms(fwd_ms)} | {_fmt_summary_ms(bwd_ms)} | "
            f"{_fmt_summary_mib(float(row['saved_activation_mib']))} | "
            f"{float(row['saved_activation_percent']):.2f}% |"
        )
    lines.append("")

    memory_rows = [row for row in rows if int(row["saved_activation_bytes"]) > 0]
    memory_rows.sort(key=lambda row: int(row["saved_activation_bytes"]), reverse=True)
    if memory_rows:
        lines.extend(["### Top Saved Activation Memory", "", "| Semantic op | unique MiB | % saved GPU | Total timing ms |", "|---|---:|---:|---:|"])
        for row in memory_rows[:10]:
            lines.append(
                f"| {row['label']} | {row['saved_activation_mib']:.2f} | "
                f"{float(row['saved_activation_percent']):.2f}% | "
                f"{_fmt_summary_ms(float(row['total_milliseconds']))} |"
            )
        lines.append("")

    gap_rows = [row for row in rows if float(row["forward_gap_ms"]) + float(row["backward_gap_ms"]) > 0.0]
    gap_rows.sort(key=lambda row: float(row["forward_gap_ms"]) + float(row["backward_gap_ms"]), reverse=True)
    if gap_rows:
        lines.extend(["### Top No-Kernel Gaps", "", "| Semantic op | Gap ms | FWD gap ms | BWD gap ms |", "|---|---:|---:|---:|"])
        for row in gap_rows[:10]:
            fwd_gap = float(row["forward_gap_ms"])
            bwd_gap = float(row["backward_gap_ms"])
            lines.append(f"| {row['label']} | {fwd_gap + bwd_gap:.4f} | {_fmt_summary_ms(fwd_gap)} | {_fmt_summary_ms(bwd_gap)} |")
        lines.append("")
    return lines


def _top_latency_markdown(report: dict[str, Any]) -> list[str]:
    rows = _semantic_timing_memory_rows(report)
    if not rows:
        return []
    total_ms = sum(float(row["total_milliseconds"]) for row in rows)
    lines = [
        "## Top Latency",
        "",
        "Sorted by total timing from largest to smallest.",
        "",
        "| Semantic op | Total ms | FWD ms | BWD ms | no-kernel gap ms | % listed | Kernel classes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fwd_ms = float(row["forward_gpu_ms"]) + float(row["forward_gap_ms"])
        bwd_ms = float(row["backward_gpu_ms"]) + float(row["backward_gap_ms"])
        gap_ms = float(row["forward_gap_ms"]) + float(row["backward_gap_ms"])
        total = float(row["total_milliseconds"])
        lines.append(
            f"| {row['label']} | {total:.4f} | "
            f"{_fmt_summary_ms(fwd_ms)} | {_fmt_summary_ms(bwd_ms)} | "
            f"{_fmt_summary_ms(gap_ms)} | {_percent(total, total_ms):.2f}% | "
            f"{row['kernel_classes']} |"
        )
    lines.append("")
    return lines


def _top_memory_rows(source_profile: dict[str, Any], memory_profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attribution = memory_profile.get("memory_attribution") if isinstance(memory_profile, dict) else None
    if isinstance(attribution, dict):
        categories = attribution.get("categories", {}) if isinstance(attribution.get("categories", {}), dict) else {}
        category_rows = categories.get("rows") if isinstance(categories, dict) else None
        if isinstance(category_rows, list):
            for row in category_rows:
                rows.append(
                    {
                        "source": "memory attribution pass",
                        "item": str(row.get("category", "-")),
                        "memory_space": str(row.get("memory_space", "-")),
                        "bytes": int(row.get("bytes", 0)),
                    }
                )

        saved = attribution.get("saved_activations", {}) if isinstance(attribution.get("saved_activations", {}), dict) else {}
        saved_rows = saved.get("rows") if isinstance(saved, dict) else None
        if isinstance(saved_rows, list):
            for row in saved_rows:
                rows.append(
                    {
                        "source": "memory attribution pass saved activation",
                        "item": _semantic_leaf_key(str(row.get("owner", "-"))),
                        "memory_space": "GPU HBM",
                        "bytes": int(row.get("unique_bytes", 0)),
                    }
                )

    memory = source_profile.get("memory") if isinstance(source_profile, dict) else None
    if isinstance(memory, dict):
        gpu = memory.get("gpu", {}) if isinstance(memory.get("gpu", {}), dict) else {}
        cpu = memory.get("cpu", {}) if isinstance(memory.get("cpu", {}), dict) else {}
        for item, memory_space, value in [
            ("GPU peak HBM", "GPU HBM", gpu.get("peak_hbm_bytes", 0)),
            ("GPU parameters", "GPU HBM", gpu.get("parameter_bytes", 0)),
            ("GPU buffers", "GPU HBM", gpu.get("buffer_bytes", 0)),
            ("GPU unattributed peak", "GPU HBM", gpu.get("unattributed_peak_bytes", 0)),
            ("CPU host W", "CPU", cpu.get("host_w_bytes", 0)),
            ("CPU pinned W", "CPU pinned", cpu.get("pinned_w_bytes", 0)),
            ("CPU pinned total", "CPU pinned", cpu.get("pinned_total_bytes", 0)),
        ]:
            rows.append({"source": "source timing pass", "item": item, "memory_space": memory_space, "bytes": int(value or 0)})

    return sorted((row for row in rows if int(row["bytes"]) > 0), key=lambda row: int(row["bytes"]), reverse=True)


def _top_memory_markdown(source_profile: dict[str, Any], memory_profile: dict[str, Any]) -> list[str]:
    rows = _top_memory_rows(source_profile, memory_profile)
    if not rows:
        return []
    lines = [
        "## Top Memory",
        "",
        "Sorted by bytes from largest to smallest.",
        "",
        "| Source | Item | Memory space | bytes | MiB |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        value = int(row["bytes"])
        lines.append(
            f"| {row['source']} | {row['item']} | {row['memory_space']} | "
            f"{value} | {_fmt_mib(value)} |"
        )
    lines.append("")
    return lines


def _overall_memory_markdown(source_profile: dict[str, Any]) -> list[str]:
    memory = source_profile.get("memory") if isinstance(source_profile, dict) else None
    if not isinstance(memory, dict):
        return [
            "## Overall Memory Summary",
            "",
            "No source memory report found. Rerun the profile with the current `scripts/profile_lora.py` and postprocess with `--source-profile-json` or `--source-profile-dir`.",
            "",
        ]
    gpu = memory.get("gpu", {}) if isinstance(memory.get("gpu", {}), dict) else {}
    cpu = memory.get("cpu", {}) if isinstance(memory.get("cpu", {}), dict) else {}
    rows = [
        ("GPU peak HBM", gpu.get("peak_hbm_bytes", 0)),
        ("GPU parameters", gpu.get("parameter_bytes", 0)),
        ("GPU buffers", gpu.get("buffer_bytes", 0)),
        ("GPU unattributed peak", gpu.get("unattributed_peak_bytes", 0)),
        ("CPU host W", cpu.get("host_w_bytes", 0)),
        ("CPU pinned W", cpu.get("pinned_w_bytes", 0)),
        ("CPU pinned total", cpu.get("pinned_total_bytes", 0)),
    ]
    lines = ["## Overall Memory Summary", "", "| Component | bytes | MiB |", "|---|---:|---:|"]
    for name, value in rows:
        lines.append(f"| {name} | {int(value)} | {_fmt_mib(value)} |")
    source_path = source_profile.get("_source_profile_path")
    if source_path:
        lines += ["", f"Source memory report: `{source_path}`"]
    lines.append("")
    return lines


def _stage_memory_markdown(source_profile: dict[str, Any]) -> list[str]:
    stage_memory = source_profile.get("stage_memory") if isinstance(source_profile, dict) else None
    rows = stage_memory.get("rows") if isinstance(stage_memory, dict) else None
    if not isinstance(rows, list) or not rows:
        return [
            "## Forward/Backward Memory Summary",
            "",
            "No forward/backward allocator snapshots found in the source profile. Existing older runs need to be rerun to populate this table.",
            "",
        ]
    lines = [
        "## Forward/Backward Memory Summary",
        "",
        "CUDA allocator snapshots are averaged over measured steps. `global peak after` is the process-wide CUDA allocated peak observed after that stage.",
        "",
        "| Stage | samples | alloc start MiB | alloc end MiB | alloc delta MiB | local peak MiB | local peak delta MiB | reserved start MiB | reserved end MiB | reserved delta MiB | global peak after MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stage_order = {"step.forward": 0, "step.backward": 1}
    for row in sorted(rows, key=lambda item: (stage_order.get(str(item.get("name", "")), 99), str(item.get("name", "")))):
        lines.append(
            f"| {row['name']} | "
            f"{row['samples']} | "
            f"{_fmt_mib(row['avg_allocated_start_bytes'])} | "
            f"{_fmt_mib(row['avg_allocated_end_bytes'])} | "
            f"{_fmt_mib(row['avg_allocated_delta_bytes'])} | "
            f"{_fmt_mib(row.get('avg_local_peak_bytes', row['max_global_peak_after_bytes']))} | "
            f"{_fmt_mib(row.get('avg_local_peak_delta_bytes', 0))} | "
            f"{_fmt_mib(row['avg_reserved_start_bytes'])} | "
            f"{_fmt_mib(row['avg_reserved_end_bytes'])} | "
            f"{_fmt_mib(row['avg_reserved_delta_bytes'])} | "
            f"{_fmt_mib(row['max_global_peak_after_bytes'])} |"
        )
    lines.append("")
    return lines


def _memory_attribution_markdown(profile: dict[str, Any]) -> list[str]:
    attribution = profile.get("memory_attribution") if isinstance(profile, dict) else None
    if not isinstance(attribution, dict):
        return [
            "## Fine-Grained Memory Attribution",
            "",
            "No memory attribution report found. Rerun the driver with memory attribution enabled.",
            "",
        ]
    categories = attribution.get("categories", {}) if isinstance(attribution.get("categories", {}), dict) else {}
    category_rows = categories.get("rows") if isinstance(categories, dict) else None
    lines = [
        "## Fine-Grained Memory Attribution",
        "",
        "Model, gradient, and optimizer rows are tensor-size accounting. Saved activation attribution is collected in a separate memory-only source pass and must not be used for timing claims.",
        "",
    ]
    if isinstance(category_rows, list) and category_rows:
        memory = profile.get("memory") if isinstance(profile, dict) else None
        gpu = memory.get("gpu", {}) if isinstance(memory, dict) and isinstance(memory.get("gpu", {}), dict) else {}
        gpu_peak_hbm = int(gpu.get("peak_hbm_bytes", 0) or 0)
        if gpu_peak_hbm <= 0:
            gpu_peak_hbm = sum(int(row.get("bytes", 0)) for row in category_rows if row.get("memory_space") == "GPU HBM")
        if gpu_peak_hbm > 0:
            lines.extend([f"Percent denominator: memory-attribution pass peak HBM `{gpu_peak_hbm}` bytes.", ""])
        lines.extend(["| Category | Memory space | bytes | MiB | % peak HBM | Accuracy |", "|---|---|---:|---:|---:|---|"])
        for row in sorted(category_rows, key=lambda item: int(item.get("bytes", 0)), reverse=True):
            value = int(row.get("bytes", 0))
            if value <= 0:
                continue
            memory_space = row.get("memory_space", "-")
            gpu_percent = f"{_percent(value, gpu_peak_hbm):.2f}%" if memory_space == "GPU HBM" else "-"
            lines.append(
                f"| {row.get('category', '-')} | {memory_space} | "
                f"{value} | {_fmt_mib(value)} | {gpu_percent} | {row.get('accuracy', '-')} |"
            )
        lines.append("")
    else:
        lines.extend(["No category rows found.", ""])
    source_path = profile.get("_source_profile_path") if isinstance(profile, dict) else None
    if source_path:
        lines.extend([f"Memory attribution report: `{source_path}`", ""])

    saved = attribution.get("saved_activations", {}) if isinstance(attribution.get("saved_activations", {}), dict) else {}
    saved_rows = saved.get("rows") if isinstance(saved, dict) else None
    if isinstance(saved_rows, list) and saved_rows:
        saved_total_unique = int(saved.get("total_unique_bytes", 0))
        if saved_total_unique <= 0:
            saved_total_unique = sum(int(row.get("unique_bytes", 0)) for row in saved_rows)
        lines.extend(
            [
                "## Saved Activation Memory by Semantic Owner",
                "",
                "`unique_bytes` deduplicates repeated saves of the same CUDA tensor. `% saved GPU` uses unique CUDA saved activation bytes only. CPU memory is excluded.",
                "",
                "| Owner | unique bytes | unique MiB | % saved GPU | reference bytes | reference MiB | saves | unique tensors |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(saved_rows, key=lambda item: int(item.get("unique_bytes", 0)), reverse=True):
            unique = int(row.get("unique_bytes", 0))
            if unique <= 0:
                continue
            refs = int(row.get("reference_bytes", 0))
            owner = _semantic_leaf_key(str(row.get("owner", "-")))
            lines.append(
                f"| {owner} | {unique} | {_fmt_mib(unique)} | "
                f"{_percent(unique, saved_total_unique):.2f}% | "
                f"{refs} | {_fmt_mib(refs)} | {int(row.get('save_count', 0))} | {int(row.get('unique_tensor_count', 0))} |"
            )
        lines.append("")
    elif not bool(attribution.get("enabled", False)):
        lines.extend(
            [
                "## Saved Activation Memory by Semantic Owner",
                "",
                "Not collected in this profile. The normal Nsight timing run intentionally leaves saved-tensor hooks disabled.",
                "",
            ]
        )
    return lines


def _front_summary_markdown(report: dict[str, Any]) -> list[str]:
    source_profile = report.get("source_profile", {})
    memory_profile = report.get("memory_profile", {})
    if not isinstance(memory_profile, dict) or not memory_profile:
        memory_profile = source_profile
    lines: list[str] = []
    lines.extend(_semantic_timing_memory_markdown(report))
    lines.extend(_top_bottlenecks_markdown(report))
    lines.extend(_timing_summary_markdown(report, "step.forward", "Forward"))
    lines.extend(_timing_summary_markdown(report, "step.backward", "Backward"))
    lines.extend(_stage_memory_markdown(source_profile if isinstance(source_profile, dict) else {}))
    lines.extend(_memory_attribution_markdown(memory_profile if isinstance(memory_profile, dict) else {}))
    lines.extend(_overall_memory_markdown(source_profile if isinstance(source_profile, dict) else {}))
    return lines


def latency_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Nsight LoRA Latency: {report['source']}", ""]
    lines.extend(_top_latency_markdown(report))
    lines.extend(_timing_summary_markdown(report, "step.forward", "Forward"))
    lines.extend(_timing_summary_markdown(report, "step.backward", "Backward"))
    for stage in report["stages"]:
        lines += [f"## {stage['stage']}", ""]
        lines += [
            "### Semantic Stage Summary",
            "",
            "| Semantic op | GPU ms | no-kernel gap ms | total ms | % stage | Kernel classes |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in _semantic_stage_rows(report, str(stage["stage"])):
            lines.append(
                f"| {row['label']} | "
                f"{_fmt_summary_ms(float(row['gpu_ms']))} | "
                f"{_fmt_summary_ms(float(row['gap_ms']))} | "
                f"{row['total_milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{row['kernel_classes']} |"
            )
        lines.append("")
        lines += [
            "### No-Kernel Gap Attribution",
            "",
            "| Gap category | ms | % stage | % no-kernel gap |",
            "|---|---:|---:|---:|",
        ]
        for row in stage["gpu_no_kernel_gap_attribution"]["rows"]:
            lines.append(
                f"| {row['name']} | "
                f"{row['milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{row['percent_no_kernel']:.2f}% |"
            )
        lines.append("")
        for title, key in [
            ("Stage Timeline", "stage_breakdown"),
            ("Host CUDA API", "host_api_breakdown"),
            ("Operation Kernel Time", "operation_kernel_time"),
            ("Operation CUDA API Time", "operation_cuda_api_time"),
        ]:
            lines += [f"### {title}", "", "| Component | ms | % stage |", "|---|---:|---:|"]
            for row in stage[key]["rows"]:
                lines.append(
                    f"| {_display_operation_name(str(row['name']))} | "
                    f"{row['milliseconds']:.4f} | {row['percent']:.2f}% |"
                )
            lines.append("")
        lines += [
            "### Operation Kernel Classes",
            "",
            "| Operation | Kernel class | ms | % stage |",
            "|---|---|---:|---:|",
        ]
        for row in stage["operation_kernel_classes"]["rows"]:
            lines.append(
                f"| {_display_operation_name(str(row['operation']))} | "
                f"{row['kernel_class']} | "
                f"{row['milliseconds']:.4f} | "
                f"{row['percent']:.2f}% |"
            )
        lines.append("")
        lines += [
            "### GPU No-Kernel Gaps",
            "",
            "| Previous kernel class | Next kernel class | gap ms | % stage | enclosing operation | CUDA API overlap ms | sync overlap ms | stage offset ms |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for row in stage["gpu_no_kernel_gaps"]["rows"]:
            lines.append(
                "| "
                f"{_display_kernel_name(str(row['previous_kernel']))} | "
                f"{_display_kernel_name(str(row['next_kernel']))} | "
                f"{row['gap_milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{_display_operation_name(str(row['enclosing_nvtx']))} | "
                f"{row['cuda_api_overlap_milliseconds']:.4f} | "
                f"{row['sync_overlap_milliseconds']:.4f} | "
                f"{row['start_offset_milliseconds']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def memory_markdown(report: dict[str, Any]) -> str:
    source_profile = report.get("source_profile", {})
    memory_profile = report.get("memory_profile", {})
    if not isinstance(memory_profile, dict) or not memory_profile:
        memory_profile = source_profile
    source_profile = source_profile if isinstance(source_profile, dict) else {}
    memory_profile = memory_profile if isinstance(memory_profile, dict) else {}
    lines = [f"# Nsight LoRA Memory: {report['source']}", ""]
    lines.extend(_top_memory_markdown(source_profile, memory_profile))
    lines.extend(_stage_memory_markdown(source_profile))
    lines.extend(_memory_attribution_markdown(memory_profile))
    lines.extend(_overall_memory_markdown(source_profile))
    return "\n".join(lines)


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Nsight LoRA Trace: {report['source']}", ""]
    lines.extend(_front_summary_markdown(report))
    for stage in report["stages"]:
        lines += [f"## {stage['stage']}", ""]
        lines += [
            "### Semantic Stage Summary",
            "",
            "| Semantic op | GPU ms | no-kernel gap ms | total ms | % stage | Kernel classes |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in _semantic_stage_rows(report, str(stage["stage"])):
            lines.append(
                f"| {row['label']} | "
                f"{_fmt_summary_ms(float(row['gpu_ms']))} | "
                f"{_fmt_summary_ms(float(row['gap_ms']))} | "
                f"{row['total_milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{row['kernel_classes']} |"
            )
        lines.append("")
        lines += [
            "### No-Kernel Gap Attribution",
            "",
            "| Gap category | ms | % stage | % no-kernel gap |",
            "|---|---:|---:|---:|",
        ]
        for row in stage["gpu_no_kernel_gap_attribution"]["rows"]:
            lines.append(
                f"| {row['name']} | "
                f"{row['milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{row['percent_no_kernel']:.2f}% |"
            )
        lines.append("")
        for title, key in [
            ("Stage Timeline", "stage_breakdown"),
            ("Host CUDA API", "host_api_breakdown"),
            ("Operation Kernel Time", "operation_kernel_time"),
            ("Operation CUDA API Time", "operation_cuda_api_time"),
        ]:
            lines += [f"### {title}", "", "| Component | ms | % stage |", "|---|---:|---:|"]
            for row in stage[key]["rows"]:
                lines.append(
                    f"| {_display_operation_name(str(row['name']))} | "
                    f"{row['milliseconds']:.4f} | {row['percent']:.2f}% |"
                )
            lines.append("")
        lines += [
            "### Operation Kernel Classes",
            "",
            "| Operation | Kernel class | ms | % stage |",
            "|---|---|---:|---:|",
        ]
        for row in stage["operation_kernel_classes"]["rows"]:
            lines.append(
                f"| {_display_operation_name(str(row['operation']))} | "
                f"{row['kernel_class']} | "
                f"{row['milliseconds']:.4f} | "
                f"{row['percent']:.2f}% |"
            )
        lines.append("")
        lines += [
            "### GPU No-Kernel Gaps",
            "",
            "| Previous kernel class | Next kernel class | gap ms | % stage | enclosing operation | CUDA API overlap ms | sync overlap ms | stage offset ms |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for row in stage["gpu_no_kernel_gaps"]["rows"]:
            lines.append(
                "| "
                f"{_display_kernel_name(str(row['previous_kernel']))} | "
                f"{_display_kernel_name(str(row['next_kernel']))} | "
                f"{row['gap_milliseconds']:.4f} | "
                f"{row['percent']:.2f}% | "
                f"{_display_operation_name(str(row['enclosing_nvtx']))} | "
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
    parser.add_argument("--source-profile-json", type=Path, help="Source profiler JSON to merge memory attribution into the Nsight table.")
    parser.add_argument("--source-profile-dir", type=Path, help="Directory containing a source profiler *_profile.json report.")
    parser.add_argument("--memory-profile-json", type=Path, help="Memory-only source profiler JSON with saved activation attribution.")
    parser.add_argument("--memory-profile-dir", type=Path, help="Directory containing a memory-only source profiler *_profile.json report.")
    args = parser.parse_args()

    con = sqlite3.connect(str(args.sqlite_path))
    source_profile = _load_source_profile(args.source_profile_json) or _load_source_profile(args.source_profile_dir)
    memory_profile = _load_source_profile(args.memory_profile_json) or _load_source_profile(args.memory_profile_dir)
    report = {
        "source": str(args.sqlite_path),
        "source_profile": source_profile,
        "memory_profile": memory_profile,
        "stages": [summarize_stage(con, "step.forward"), summarize_stage(con, "step.backward")],
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(markdown(report), encoding="utf-8")
        args.output_md.with_name("lat.md").write_text(latency_markdown(report), encoding="utf-8")
        args.output_md.with_name("memory.md").write_text(memory_markdown(report), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
