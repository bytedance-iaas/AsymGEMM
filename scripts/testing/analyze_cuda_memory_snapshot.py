#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any


@dataclass(frozen=True)
class LiveBlock:
    addr: int
    size: int
    frames: tuple[dict[str, Any], ...]
    event_index: int


_EXPERT_FILES = (
    "asym_gemm/training/qwen3_moe.py",
    "asym_gemm/training/moe.py",
    "asym_gemm/training/exp_act_offload",
    "asym_gemm/training/activation_offload.py",
    "asym_gemm/training/frozen_linear.py",
    "asym_gemm/training/cpu_left.py",
)


def _frame_filename(frame: Any) -> str:
    if isinstance(frame, dict):
        return str(frame.get("filename") or frame.get("file") or "")
    return str(getattr(frame, "filename", "") or getattr(frame, "file", "") or "")


def _frame_name(frame: Any) -> str:
    if isinstance(frame, dict):
        return str(frame.get("name") or frame.get("function") or "")
    return str(getattr(frame, "name", "") or getattr(frame, "function", "") or "")


def _frame_line(frame: Any) -> int | None:
    value = frame.get("line") if isinstance(frame, dict) else getattr(frame, "line", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _component_for_frame(frame: Any) -> str:
    filename = _frame_filename(frame).replace("\\", "/")
    lowered = filename.lower()
    name = _frame_name(frame).lower()
    line = _frame_line(frame)
    if "asym_gemm/training/offload.py" in lowered and line is not None and 339 <= line <= 415:
        return "norms"
    if any(part in lowered for part in _EXPERT_FILES) or "expert" in name:
        return "routed_experts"
    if "attention" in lowered or "attn" in lowered or "attention" in name or "attn" in name:
        return "attention"
    if "norm" in lowered or "norm" in name:
        return "norms"
    if (
        "loss" in lowered
        or "cross_entropy" in lowered
        or "cross_entropy" in name
        or "causallmloss" in name
        or ("modeling_qwen3_moe.py" in lowered and line == 688)
    ):
        return "loss"
    if "lm_head" in lowered or "lm_head" in name:
        return "lm_head"
    if "optim" in lowered or "optimizer" in lowered or "adam" in lowered:
        return "optimizer"
    if "embedding" in lowered or "embed_tokens" in lowered:
        return "embedding"
    if "torch/" in lowered or "site-packages/torch" in lowered:
        return "torch_internal"
    if not filename:
        return "allocator_unframed"
    return "other_python"


def _frame_score(frame: Any, index: int) -> tuple[int, int]:
    component = _component_for_frame(frame)
    scores = {
        "routed_experts": 90,
        "loss": 80,
        "attention": 70,
        "norms": 75,
        "lm_head": 60,
        "embedding": 55,
        "optimizer": 50,
        "other_python": 20,
        "torch_internal": 5,
        "allocator_unframed": 0,
    }
    return scores.get(component, 0), index


def _best_frame(frames: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    if not frames:
        return None
    indexed = list(enumerate(frames))
    _, frame = max(indexed, key=lambda item: _frame_score(item[1], item[0]))
    return frame


def _frame_key(frame: Any | None) -> str:
    if frame is None:
        return "<no python frame>"
    filename = _frame_filename(frame) or "<unknown>"
    name = _frame_name(frame) or "<unknown>"
    line = _frame_line(frame)
    if line is None:
        return f"{filename}:{name}"
    return f"{filename}:{line}:{name}"


def _event_action(event: Any) -> str:
    if isinstance(event, dict):
        action = str(event.get("action") or event.get("kind") or "").lower()
    else:
        action = str(getattr(event, "action", "") or getattr(event, "kind", "")).lower()
    if "." in action:
        action = action.rsplit(".", 1)[-1]
    return action


def _event_addr(event: Any) -> int | None:
    value = event.get("addr") if isinstance(event, dict) else getattr(event, "addr", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_size(event: Any) -> int:
    if isinstance(event, dict):
        value = event.get("size") or event.get("requested_size") or event.get("allocation_size")
    else:
        value = getattr(event, "size", None) or getattr(event, "requested_size", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _event_frames(event: Any) -> tuple[dict[str, Any], ...]:
    frames = event.get("frames") if isinstance(event, dict) else getattr(event, "frames", None)
    if not isinstance(frames, (list, tuple)):
        return ()
    normalized: list[dict[str, Any]] = []
    for frame in frames:
        normalized.append(
            {
                "filename": _frame_filename(frame),
                "name": _frame_name(frame),
                "line": _frame_line(frame),
            }
        )
    return tuple(normalized)


def _device_traces(snapshot: dict[str, Any], device: int | None) -> list[tuple[int, list[Any]]]:
    raw = snapshot.get("device_traces") or []
    traces: list[tuple[int, list[Any]]] = []
    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = enumerate(raw)
    for device_index, trace in iterable:
        try:
            device_int = int(device_index)
        except (TypeError, ValueError):
            continue
        if device is not None and device_int != device:
            continue
        if isinstance(trace, list):
            traces.append((device_int, trace))
    return traces


def analyze_snapshot(snapshot: dict[str, Any], *, device: int | None = None, min_bytes: int = 0, top: int = 40) -> dict[str, Any]:
    live: dict[int, LiveBlock] = {}
    free_requested_addrs: set[int] = set()
    peak_live: dict[int, LiveBlock] = {}
    current_live_bytes = 0
    peak_live_bytes = 0
    alloc_events = 0
    free_events = 0
    unknown_free_events = 0
    event_index = 0

    traces = _device_traces(snapshot, device)
    for _, trace in traces:
        for event in trace:
            event_index += 1
            action = _event_action(event)
            addr = _event_addr(event)
            if addr is None:
                continue
            if action == "alloc":
                size = _event_size(event)
                if size <= 0:
                    continue
                previous = live.pop(addr, None)
                if previous is not None:
                    current_live_bytes -= previous.size
                live[addr] = LiveBlock(addr=addr, size=size, frames=_event_frames(event), event_index=event_index)
                current_live_bytes += size
                alloc_events += 1
                if current_live_bytes > peak_live_bytes:
                    peak_live_bytes = current_live_bytes
                    peak_live = dict(live)
            elif action in {"free_requested", "free_completed", "free"}:
                block = live.pop(addr, None)
                if block is None:
                    if action == "free_completed" and addr in free_requested_addrs:
                        free_requested_addrs.discard(addr)
                        continue
                    unknown_free_events += 1
                    continue
                current_live_bytes -= block.size
                free_events += 1
                if action == "free_requested":
                    free_requested_addrs.add(addr)

    bucket_bytes: dict[str, int] = defaultdict(int)
    bucket_blocks: dict[str, int] = defaultdict(int)
    frame_bytes: dict[str, int] = defaultdict(int)
    frame_blocks: dict[str, int] = defaultdict(int)
    top_blocks: list[dict[str, Any]] = []
    for block in peak_live.values():
        if block.size < min_bytes:
            continue
        frame = _best_frame(block.frames)
        component = _component_for_frame(frame) if frame is not None else "allocator_unframed"
        frame_key = _frame_key(frame)
        bucket_bytes[component] += block.size
        bucket_blocks[component] += 1
        frame_bytes[frame_key] += block.size
        frame_blocks[frame_key] += 1
        top_blocks.append(
            {
                "addr": hex(block.addr),
                "bytes": block.size,
                "component": component,
                "frame": frame_key,
                "event_index": block.event_index,
                "frames": [_frame_key(frame) for frame in block.frames[:12]],
            }
        )
    top_blocks.sort(key=lambda row: int(row["bytes"]), reverse=True)

    bucket_rows = [
        {"component": key, "bytes": value, "blocks": bucket_blocks[key]}
        for key, value in sorted(bucket_bytes.items(), key=lambda item: item[1], reverse=True)
    ]
    frame_rows = [
        {"frame": key, "bytes": value, "blocks": frame_blocks[key]}
        for key, value in sorted(frame_bytes.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "schema_version": 1,
        "device": device,
        "trace_count": len(traces),
        "event_count": event_index,
        "alloc_events": alloc_events,
        "free_events": free_events,
        "unknown_free_events": unknown_free_events,
        "peak_live_bytes": peak_live_bytes,
        "final_live_bytes": current_live_bytes,
        "peak_live_block_count": len(peak_live),
        "min_reported_block_bytes": int(min_bytes),
        "bucket_rows": bucket_rows,
        "frame_rows": frame_rows[: int(top)],
        "top_blocks": top_blocks[: int(top)],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CUDA Memory Snapshot Peak Attribution",
        "",
        f"- peak live bytes: `{report['peak_live_bytes']}`",
        f"- final live bytes: `{report['final_live_bytes']}`",
        f"- peak live blocks: `{report['peak_live_block_count']}`",
        f"- trace count: `{report['trace_count']}`",
        f"- events: `{report['event_count']}`",
        "",
        "## Components",
        "",
        "| component | bytes | blocks |",
        "| --- | ---: | ---: |",
    ]
    for row in report.get("bucket_rows", []):
        lines.append(f"| {row['component']} | {row['bytes']} | {row['blocks']} |")
    lines.extend(["", "## Top Frames", "", "| frame | bytes | blocks |", "| --- | ---: | ---: |"])
    for row in report.get("frame_rows", []):
        lines.append(f"| {row['frame']} | {row['bytes']} | {row['blocks']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay torch CUDA memory_snapshot device_traces and attribute the peak live-set.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--min-bytes", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    with args.snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not isinstance(snapshot, dict):
        raise SystemExit("snapshot root must be a dict")
    report = analyze_snapshot(snapshot, device=args.device, min_bytes=args.min_bytes, top=args.top)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(report), encoding="utf-8")
    if args.output_json is None and args.output_md is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
