#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from asym_gemm.profiling.lf_trace import build_memory_breakdown_summary
except Exception:  # pragma: no cover - plotting still works with existing summary files.
    build_memory_breakdown_summary = None


GIB = 1024.0**3
COMPONENT_ORDER = [
    "attention",
    "router",
    "shared_experts",
    "routed_experts",
    "mlp_dense",
    "lora",
    "embedding",
    "lm_head",
    "norms",
    "loss",
    "other",
    "other_saved_activations",
    "source_runtime",
]
GROUP_ORDER = ["weights", "gradients", "optimizer", "saved_activations", "temporary_workspace", "persistent"]
COMPONENT_LABELS = {
    "attention": "Attention",
    "router": "Router",
    "shared_experts": "Shared experts",
    "routed_experts": "Routed experts",
    "mlp_dense": "Dense MLP",
    "lora": "LoRA",
    "embedding": "Embedding",
    "lm_head": "LM head",
    "norms": "Norms",
    "loss": "Loss",
    "other": "Other",
    "other_saved_activations": "Other activations",
    "unknown_saved_activation": "Other activations",
    "source_runtime": "Source/runtime",
}
GROUP_LABELS = {
    "weights": "weights",
    "gradients": "gradients",
    "optimizer": "optimizer",
    "saved_activations": "activations",
    "temporary_workspace": "temporary workspace",
    "persistent": "persistent",
}
SPECIAL_SEGMENT_LABELS = {
    "unattributed_allocated_peak": "Unattributed allocated peak",
    "allocator_reserved_unallocated": "Reserved but unallocated",
}
SPECIAL_SEGMENT_COLORS = {
    "unattributed_allocated_peak": "#00a6a6",
    "allocator_reserved_unallocated": "#4f46e5",
    "external_cuda_or_driver": "#111827",
}
SEGMENT_PALETTE = [
    "#005f73",
    "#ca6702",
    "#2a9d8f",
    "#d62828",
    "#7b2cbf",
    "#118ab2",
    "#9b5de5",
    "#f15bb5",
    "#386641",
    "#f77f00",
    "#06d6a0",
    "#ef476f",
    "#264653",
    "#e76f51",
    "#3a86ff",
    "#ffbe0b",
    "#8338ec",
    "#fb5607",
    "#0081a7",
    "#c1121f",
    "#588157",
    "#ff006e",
    "#4361ee",
    "#ffb703",
]
SEGMENT_COLORS = {
    "attention:weights": "#005f73",
    "attention:gradients": "#ca6702",
    "attention:optimizer": "#2a9d8f",
    "attention:saved_activations": "#d62828",
    "attention:temporary_workspace": "#ef4444",
    "router:weights": "#7b2cbf",
    "router:gradients": "#118ab2",
    "router:optimizer": "#9b5de5",
    "router:saved_activations": "#f15bb5",
    "router:temporary_workspace": "#db2777",
    "shared_experts:weights": "#386641",
    "shared_experts:gradients": "#f77f00",
    "shared_experts:optimizer": "#06d6a0",
    "shared_experts:saved_activations": "#ef476f",
    "shared_experts:temporary_workspace": "#84cc16",
    "routed_experts:weights": "#264653",
    "routed_experts:gradients": "#e76f51",
    "routed_experts:optimizer": "#3a86ff",
    "routed_experts:saved_activations": "#ffbe0b",
    "routed_experts:temporary_workspace": "#f97316",
    "mlp_dense:weights": "#8338ec",
    "mlp_dense:gradients": "#fb5607",
    "mlp_dense:optimizer": "#0081a7",
    "mlp_dense:saved_activations": "#c1121f",
    "mlp_dense:temporary_workspace": "#dc2626",
    "lora:weights": "#588157",
    "lora:gradients": "#ff006e",
    "lora:optimizer": "#4361ee",
    "lora:saved_activations": "#ffb703",
    "lora:temporary_workspace": "#eab308",
    "other_saved_activations:saved_activations": "#a855f7",
    "other_saved_activations:temporary_workspace": "#9333ea",
    "embedding:weights": "#0f766e",
    "lm_head:weights": "#ea580c",
    "lm_head:gradients": "#facc15",
    "lm_head:optimizer": "#14b8a6",
    "lm_head:saved_activations": "#f43f5e",
    "lm_head:temporary_workspace": "#fb923c",
    "norms:weights": "#2563eb",
    "norms:gradients": "#0891b2",
    "norms:optimizer": "#7c3aed",
    "norms:saved_activations": "#22c55e",
    "norms:temporary_workspace": "#65a30d",
    "loss:persistent": "#dc2626",
    "loss:saved_activations": "#ec4899",
    "loss:temporary_workspace": "#be123c",
    "other:persistent": "#16a34a",
    "source_runtime:temporary_workspace": "#0f172a",
}
RUN_DIR_RE = re.compile(r"^(?:b(?P<batch_size>[0-9]+)_)?s(?P<seq_len>[0-9]+)$")
PHASE_PRIORITY = {
    "after_backward": 60,
    "before_optimizer_step": 50,
    "after_forward": 40,
    "after_optimizer_step": 30,
    "step_begin": 0,
}
PHASE_ORDER = ["step_begin", "after_forward", "after_backward", "before_optimizer_step", "after_optimizer_step"]
STANDARD_BAR_WIDTH = 0.78
MIN_BAR_SLOTS = 4
BAR_SLOT_INCHES = 0.95
PLOT_EXTRA_INCHES = 3.9
PEAK_LINE_COLOR = "#111827"


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
            f"router={self.metadata.get('router_mode', '')}" if self.metadata.get("router_mode") else "",
            self.metadata.get("expact", ""),
            self.metadata.get("attnact", ""),
            self.metadata.get("recompute", ""),
            self.metadata.get("expert_policy", ""),
            self.metadata.get("seq_len", ""),
        ]
        return " ".join(part for part in parts if part)


CONFIG_SUFFIX_RE = re.compile(
    r"(?:^|__)gpus(?P<gpus>[0-9]+)__b(?P<batch>[0-9]+)_s(?P<seq>[0-9]+)_"
    r"w(?P<warmup>[0-9]+)_s(?P<steps>[0-9]+)_r(?P<rank>[0-9]+)_a(?P<alpha>[^_]+)_(?P<drop>drop[0-9]+)"
)


def _run_config_label(run: RunRecord) -> str:
    config = str(run.metadata.get("config") or run.run_dir.parent.parent.name)
    match = CONFIG_SUFFIX_RE.search(config)
    if match is None:
        return config
    return (
        f"gpus{match.group('gpus')} b{match.group('batch')} s{match.group('seq')} "
        f"w{match.group('warmup')}_s{match.group('steps')} "
        f"r{match.group('rank')} a{match.group('alpha')} {match.group('drop')}"
    )


def _run_plot_label(run: RunRecord) -> str:
    return run.label or run.run_dir.name


def _wrap_plot_label(label: str, *, width: int = 32) -> str:
    if len(label) <= width:
        return label
    return "\n".join(
        textwrap.wrap(
            label,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _run_plot_labels(runs: list[RunRecord]) -> list[str]:
    base_labels = [_run_plot_label(run) for run in runs]
    counts: dict[str, int] = {}
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1
    return [
        _wrap_plot_label(f"{label}\n{_run_config_label(run)}" if counts.get(label, 0) > 1 else label)
        for run, label in zip(runs, base_labels)
    ]


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
    parser.add_argument("--router-mode", action="append", default=[], choices=["hf", "whole"])
    parser.add_argument("--expact", action="append", default=[], choices=["expact0", "expact1"])
    parser.add_argument("--attnact", action="append", default=[], choices=["attnact0", "attnact1"])
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


def _saved_activation_component(component: Any) -> str:
    component_str = str(component or "").strip()
    if component_str in {"", "unknown_saved_activation"}:
        return "other_saved_activations"
    return component_str


def _repair_summary_from_jsonl(summary: dict[str, Any], jsonl_path: Path | None) -> dict[str, Any]:
    if build_memory_breakdown_summary is None or jsonl_path is None or not jsonl_path.exists():
        return summary
    rows = _load_jsonl(jsonl_path)
    if not rows:
        return summary
    rebuilt = build_memory_breakdown_summary(rows)
    if int(rebuilt.get("schema_version") or 0) != 2 or not rebuilt.get("enabled", False):
        return summary
    old_saved = int(summary.get("saved_activation_hbm_bytes_at_peak", 0) or 0)
    new_saved = int(rebuilt.get("saved_activation_hbm_bytes_at_peak", 0) or 0)
    old_unattributed = int(summary.get("unattributed_allocated_peak_bytes", 0) or 0)
    new_unattributed = int(rebuilt.get("unattributed_allocated_peak_bytes", 0) or 0)
    required_fields = {
        "live_activation_hbm_bytes_at_peak",
        "activation_hbm_bytes_at_peak",
        "temporary_workspace_hbm_bytes_at_peak",
        "actual_peak_breakdown_rows",
    }
    if any(field not in summary for field in required_fields):
        return rebuilt
    if new_saved > old_saved or new_unattributed < old_unattributed:
        return rebuilt
    return summary


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _seq_len_from_run_dir_name(name: str) -> str:
    match = RUN_DIR_RE.match(name)
    return match.group("seq_len") if match is not None else ""


def _safe_label(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "run"


def _normalize_bool_config(value: Any, default: str = "false") -> str:
    text = str(value if value is not None else default).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    return default


def _expact_label(value: Any) -> str:
    return "expact1" if _normalize_bool_config(value) == "true" else "expact0"


def _attnact_label(value: Any) -> str:
    return "attnact1" if _normalize_bool_config(value) == "true" else "attnact0"


def _parse_expact_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"expact1", "expacttrue"}:
        return "true", "expact1"
    if value in {"expact0", "expactfalse"}:
        return "false", "expact0"
    return None


def _parse_attnact_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"attnact1", "attnacttrue"}:
        return "true", "attnact1"
    if value in {"attnact0", "attnactfalse"}:
        return "false", "attnact0"
    return None


def _known_optional_job_axis(part: str) -> bool:
    value = part.strip().lower()
    return (
        value in {"layeract0", "layeract1", "layeractfalse", "layeracttrue"}
        or value in {"actrecomp0", "actrecomp1", "actrecompfalse", "actrecomptrue"}
        or value in {"xunpack0", "xunpack1", "xunpackfalse", "xunpacktrue"}
        or value.startswith("loraafwd")
        or value.startswith("gradoff")
        or value.startswith("weightoff")
    )


def _parse_job_dir_parts(job_dir_name: str) -> dict[str, str] | None:
    parts = job_dir_name.split("__")
    if len(parts) < 4:
        return None
    backend_part, profiler_part, recompute_part, policy_part = parts[:4]
    tail = parts[4:]
    router_part = "routerhf"
    expact_value = "false"
    expact = "expact0"
    attnact_value = "false"
    attnact = "attnact0"

    if tail:
        router_part = tail.pop(0)
        if not router_part.startswith("router"):
            return None

    for part in tail:
        parsed_expact = _parse_expact_part(part)
        if parsed_expact is not None:
            expact_value, expact = parsed_expact
            continue
        parsed_attnact = _parse_attnact_part(part)
        if parsed_attnact is not None:
            attnact_value, attnact = parsed_attnact
            continue
        if _known_optional_job_axis(part):
            continue
        return None

    return {
        "backend": backend_part,
        "profiler": profiler_part,
        "recompute": recompute_part,
        "policy_part": policy_part,
        "router_part": router_part,
        "asymm_expert_act_offload": expact_value,
        "expact": expact,
        "asymm_attn_act_offload": attnact_value,
        "attnact": attnact,
    }


def _run_dir_for_summary(summary_path: Path) -> Path:
    run_dir = summary_path.parent
    if run_dir.name == "memory_breakdown" and RUN_DIR_RE.match(run_dir.parent.name):
        return run_dir.parent
    return run_dir


def _infer_metadata(run_dir: Path, summary: dict[str, Any]) -> dict[str, str] | None:
    source_profile = _safe_read_json(run_dir / "source_profile.json")
    config = source_profile.get("config", {}) if isinstance(source_profile.get("config"), dict) else {}
    job_root = run_dir.parent
    config_root = job_root.parent
    job_meta = _parse_job_dir_parts(job_root.name)
    if job_meta is None:
        return None
    backend_part = job_meta["backend"]
    profiler_part = job_meta["profiler"]
    recompute_part = job_meta["recompute"]
    policy_part = job_meta["policy_part"]
    router_part = job_meta["router_part"]
    expact_value = job_meta["asymm_expert_act_offload"]
    expact = job_meta["expact"]
    attnact_value = job_meta["asymm_attn_act_offload"]
    attnact = job_meta["attnact"]
    if not policy_part.startswith("pol") or not router_part.startswith("router"):
        return None
    router_mode = str(config.get("router_mode") or router_part[len("router") :])
    if router_mode not in {"hf", "whole"}:
        return None
    expert_policy = str(config.get("expert_policy") or policy_part[len("pol") :] or "none")
    expact_value = _normalize_bool_config(config.get("asymm_expert_act_offload", expact_value))
    expact = _expact_label(expact_value)
    attnact_value = _normalize_bool_config(config.get("asymm_attn_act_offload", attnact_value))
    attnact = _attnact_label(attnact_value)

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__")[0]),
        "backend": str(config.get("backend") or backend_part),
        "profiler": str(profiler_part),
        "recompute": str(recompute_part),
        "expert_policy": expert_policy,
        "router_mode": router_mode,
        "asymm_expert_act_offload": expact_value,
        "expact": expact,
        "asymm_attn_act_offload": attnact_value,
        "attnact": attnact,
        "seq_len": str(config.get("seq_len") or ""),
        "precision": str(config.get("precision") or ""),
        "batch_size": str(config.get("batch_size") or ""),
        "lora_dropout": str(config.get("lora_dropout") if config.get("lora_dropout") is not None else ""),
        "config": config_root.name,
    }
    if not metadata["seq_len"]:
        metadata["seq_len"] = _seq_len_from_run_dir_name(run_dir.name)
    if not metadata["profiler"]:
        metadata["profiler"] = "source"
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
        if int(summary.get("schema_version") or 0) != 2:
            continue
        run_dir = _run_dir_for_summary(summary_path)
        jsonl_path = _first_existing(
            [
                run_dir / "memory_breakdown.jsonl",
                run_dir / "memory_breakdown" / "memory_breakdown.jsonl",
                run_dir / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
                summary_path.parent / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
            ]
        )
        summary = _repair_summary_from_jsonl(summary, jsonl_path)
        metadata = _infer_metadata(run_dir, summary)
        if metadata is None:
            continue
        record = RunRecord(run_dir=run_dir, summary_path=summary_path, jsonl_path=jsonl_path, summary=summary, metadata=metadata)
        if not _matches_filters(record, args):
            continue
        runs.append(record)
    return sorted(runs, key=lambda run: (run.metadata.get("workload", ""), run.metadata.get("seq_len", ""), run.label, str(run.run_dir)))


def _no_runs_message(args: argparse.Namespace) -> str:
    paths = _find_summary_paths(args.input_root, args.run_dir, args.include_non_source)
    if not paths:
        return "no source memory_breakdown_summary.json files matched the requested filters"
    legacy_paths: list[Path] = []
    disabled_paths: list[Path] = []
    metadata_failures = 0
    filter_failures = 0
    for summary_path in paths:
        summary = _safe_read_json(summary_path)
        if not summary.get("enabled", bool(summary.get("breakdown_rows"))):
            disabled_paths.append(summary_path)
            continue
        if int(summary.get("schema_version") or 0) != 2:
            legacy_paths.append(summary_path)
            continue
        run_dir = _run_dir_for_summary(summary_path)
        metadata = _infer_metadata(run_dir, summary)
        if metadata is None:
            metadata_failures += 1
            continue
        jsonl_path = _first_existing(
            [
                run_dir / "memory_breakdown.jsonl",
                run_dir / "memory_breakdown" / "memory_breakdown.jsonl",
                run_dir / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
                summary_path.parent / f"{summary_path.stem.removesuffix('_summary')}.jsonl",
            ]
        )
        record = RunRecord(
            run_dir=run_dir,
            summary_path=summary_path,
            jsonl_path=jsonl_path,
            summary=summary,
            metadata=metadata,
        )
        if not _matches_filters(record, args):
            filter_failures += 1
    details: list[str] = []
    if legacy_paths:
        details.append(
            f"found {len(legacy_paths)} legacy/non-v2 source memory breakdown summary file(s); "
            "rerun the source profiler so memory_breakdown_summary.json is schema_version 2"
        )
    if disabled_paths:
        details.append(f"found {len(disabled_paths)} disabled/empty summary file(s)")
    if metadata_failures:
        details.append(f"{metadata_failures} schema-v2 summary file(s) had unparseable run metadata")
    if filter_failures:
        details.append(f"{filter_failures} schema-v2 summary file(s) were excluded by filters")
    if details:
        return "no schema-v2 source memory breakdown summaries matched; " + "; ".join(details)
    return "no source memory_breakdown_summary.json files matched the requested filters"


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
        "router_mode": _filter_values(args.router_mode),
        "expact": _filter_values(args.expact),
        "attnact": _filter_values(args.attnact),
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


def _selection_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0),
        PHASE_PRIORITY.get(str(row.get("phase") or ""), 10),
        int(row.get("allocated_bytes") or 0),
        int(row.get("step") or 0),
    )


def _segment_key(row: dict[str, Any]) -> str | None:
    component = str(row.get("component") or "")
    group = str(row.get("group") or "persistent")
    memory_space = str(row.get("memory_space") or "")
    if component == "allocator_reserved_unallocated":
        return "allocator_reserved_unallocated"
    if component == "external_cuda_or_driver":
        return "external_cuda_or_driver"
    if memory_space != "GPU HBM":
        return None
    if group == "unattributed_allocated_peak" or component == "unattributed_allocated_peak":
        return "unattributed_allocated_peak"
    if group not in GROUP_ORDER:
        group = "persistent"
    return f"{component or 'other'}:{group}"


def _component_sort_index(component: str) -> int:
    try:
        return COMPONENT_ORDER.index(component)
    except ValueError:
        return len(COMPONENT_ORDER)


def _group_sort_index(group: str) -> int:
    try:
        return GROUP_ORDER.index(group)
    except ValueError:
        return len(GROUP_ORDER)


def _segment_sort_key(key: str) -> tuple[int, int, str]:
    if key == "unattributed_allocated_peak":
        return (10_000, 0, key)
    if key == "allocator_reserved_unallocated":
        return (10_001, 0, key)
    if key == "external_cuda_or_driver":
        return (10_002, 0, key)
    component, _, group = key.partition(":")
    return (_component_sort_index(component), _group_sort_index(group), key)


def _segment_label(key: str) -> str:
    if key in SPECIAL_SEGMENT_LABELS:
        return SPECIAL_SEGMENT_LABELS[key]
    if key == "external_cuda_or_driver":
        return "External CUDA/driver"
    if key == "other_saved_activations:saved_activations":
        return "Other activations"
    component, _, group = key.partition(":")
    component_label = COMPONENT_LABELS.get(component, component.replace("_", " "))
    group_label = GROUP_LABELS.get(group, group.replace("_", " "))
    return f"{component_label} {group_label}"


def _segment_color(key: str) -> str:
    if key in SPECIAL_SEGMENT_COLORS:
        return SPECIAL_SEGMENT_COLORS[key]
    if key in SEGMENT_COLORS:
        return SEGMENT_COLORS[key]
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return SEGMENT_PALETTE[int(digest[:8], 16) % len(SEGMENT_PALETTE)]


def _positive_segment_keys(series: dict[str, list[int]] | dict[str, int]) -> list[str]:
    keys: list[str] = []
    for key, values in series.items():
        if isinstance(values, list):
            if any(int(value or 0) > 0 for value in values):
                keys.append(key)
        elif int(values or 0) > 0:
            keys.append(key)
    return sorted(keys, key=_segment_sort_key)


def _bar_axis_slots(n_bars: int, *, min_slots: int = MIN_BAR_SLOTS) -> int:
    return max(int(n_bars), int(min_slots), 1)


def _bar_plot_width(n_bars: int, *, min_slots: int = MIN_BAR_SLOTS) -> float:
    slots = _bar_axis_slots(n_bars, min_slots=min_slots)
    return max(7.0, min(36.0, PLOT_EXTRA_INCHES + BAR_SLOT_INCHES * slots))


def _subplot_plot_width(n_bars: int, *, min_slots: int = MIN_BAR_SLOTS) -> float:
    slots = _bar_axis_slots(n_bars, min_slots=min_slots)
    return max(6.0, min(18.0, 2.6 + BAR_SLOT_INCHES * slots))


def _set_standard_bar_geometry(ax: Any, n_bars: int, *, min_slots: int = MIN_BAR_SLOTS) -> None:
    slots = _bar_axis_slots(n_bars, min_slots=min_slots)
    side_padding = max(0.5, (slots - int(n_bars)) / 2.0 + 0.5)
    ax.set_xlim(-side_padding, max(int(n_bars) - 1, 0) + side_padding)


def _nice_y_limit_gib(peak_bytes: int) -> float:
    peak_gib = max(float(peak_bytes) / GIB, 0.0)
    if peak_gib <= 0.0:
        return 1.0
    target = peak_gib * 1.08
    if target >= 1.0:
        return float(math.ceil(target))
    magnitude = 10.0 ** math.floor(math.log10(target))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = multiplier * magnitude
        if candidate >= target:
            return candidate
    return 10.0 * magnitude


def _legend_handles(keys: list[str], *, include_peak: bool = True) -> list[Any]:
    handles: list[Any] = [
        Patch(facecolor=_segment_color(key), edgecolor="#ffffff", linewidth=0.4, label=_segment_label(key))
        for key in keys
    ]
    if include_peak:
        handles.append(Line2D([0], [0], color=PEAK_LINE_COLOR, linestyle="--", linewidth=1.2, label="Peak allocated"))
    return handles


def _plot_legend(ax: Any, keys: list[str], *, include_peak: bool = True) -> None:
    handles = _legend_handles(keys, include_peak=include_peak)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8,
        labelspacing=0.35,
        handlelength=1.4,
        borderaxespad=0.0,
    )


def _aggregate_rows(rows: list[dict[str, Any]], *, include_external: bool = False) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _segment_key(row)
        if key is None:
            continue
        if key == "external_cuda_or_driver" and not include_external:
            continue
        values[key] = values.get(key, 0) + int(row.get("bytes", 0) or 0)
    return values


def _aggregate_summary(summary: dict[str, Any]) -> dict[str, int]:
    rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return {}
    return _aggregate_rows(rows)


def _aggregate_actual_peak_summary(summary: dict[str, Any]) -> dict[str, int]:
    rows = summary.get("actual_peak_breakdown_rows", [])
    if not isinstance(rows, list) or not rows:
        rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return {}
    return _aggregate_rows(rows)


def _measured_jsonl_rows(run: RunRecord) -> list[dict[str, Any]]:
    rows = []
    for row in _load_jsonl(run.jsonl_path):
        if not isinstance(row, dict) or row.get("is_warmup") or int(row.get("schema_version") or 0) != 2:
            continue
        rows.append(row)
    return rows


def _flatten_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    peak_allocated = int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)
    peak_reserved = int(row.get("peak_reserved_since_step_begin") or row.get("reserved_bytes") or 0)
    peak_reserved = max(peak_reserved, peak_allocated)
    persistent = row.get("persistent_bytes", {})
    saved_activation = row.get("saved_activation_bytes_at_peak", {})
    if not isinstance(saved_activation, dict) or not saved_activation:
        saved_activation = row.get("saved_activation_bytes", {})
    live_activation = row.get("live_activation_bytes_at_peak", {})
    if not isinstance(live_activation, dict) or not live_activation:
        live_activation = row.get("live_activation_bytes", {})
    peak_growth = row.get("peak_growth_bytes_at_peak", {})
    if not isinstance(peak_growth, dict) or not peak_growth:
        peak_growth = row.get("peak_growth_bytes", {})
    closure = row.get("closure_bytes", {})
    external_memory = row.get("external_memory", {})
    rows: list[dict[str, Any]] = []

    def add(memory_space: str, group: str, component: str, kind: str, value: int, *, keep_zero: bool = False) -> None:
        if value > 0 or (value == 0 and keep_zero):
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
                if kind_str.endswith("_cpu_pinned"):
                    # Page-locked SUBSET of the matching *_cpu total -- already counted in "CPU host".
                    # Skip it so the CPU host bar does not double-count offloaded weights.
                    continue
                if kind_str.endswith("_cpu"):
                    add("CPU host", "host", str(component), kind_str, value_int)
                elif kind_str in {"weight", "frozen_weight", "buffer"}:
                    add("GPU HBM", "weights", str(component), kind_str, value_int)
                elif kind_str == "grad":
                    add("GPU HBM", "gradients", str(component), kind_str, value_int)
                elif kind_str == "optimizer_state":
                    add("GPU HBM", "optimizer", str(component), kind_str, value_int)
                else:
                    add("GPU HBM", "persistent", str(component), kind_str, value_int)

    saved_activation_totals: dict[str, int] = {}
    for component, value in saved_activation.items() if isinstance(saved_activation, dict) else []:
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        saved_activation_totals[component_name] = saved_activation_totals.get(component_name, 0) + value_int
    saved_activation_items = sorted(saved_activation_totals.items())
    for component, value in saved_activation_items:
        add("GPU HBM", "saved_activations", component, "saved_activation", value)

    live_activation_totals: dict[str, int] = {}
    for component, value in live_activation.items() if isinstance(live_activation, dict) else []:
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        live_activation_totals[component_name] = live_activation_totals.get(component_name, 0) + value_int
    for component, value in sorted(live_activation_totals.items()):
        add("GPU HBM", "saved_activations", component, "live_activation", value)

    known_before_workspace = sum(int(item["bytes"]) for item in rows if item["memory_space"] == "GPU HBM")
    workspace_residual = max(0, peak_allocated - known_before_workspace)
    if isinstance(closure, dict):
        workspace_residual = min(workspace_residual, max(0, int(closure.get("unattributed_allocated_peak") or 0)))
    peak_growth_totals: dict[str, int] = {}
    for component, value in peak_growth.items() if isinstance(peak_growth, dict) else []:
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        peak_growth_totals[component_name] = peak_growth_totals.get(component_name, 0) + value_int
    total_peak_growth = sum(peak_growth_totals.values())
    if workspace_residual > 0 and total_peak_growth > 0:
        remaining = int(workspace_residual)
        growth_items = sorted(peak_growth_totals.items(), key=lambda item: int(item[1]), reverse=True)
        for index, (component, growth_bytes) in enumerate(growth_items):
            if remaining <= 0:
                break
            if index == len(growth_items) - 1:
                value = remaining
            else:
                value = min(remaining, int(round(workspace_residual * (int(growth_bytes) / total_peak_growth))))
            if value <= 0:
                continue
            add("GPU HBM", "temporary_workspace", component, "inferred_peak_workspace", value)
            remaining -= value

    known = sum(int(item["bytes"]) for item in rows if item["memory_space"] == "GPU HBM")
    unattributed = max(0, peak_allocated - known)
    add("GPU HBM", "unattributed_allocated_peak", "unattributed_allocated_peak", "allocated_residual", unattributed)
    add(
        "GPU reserved",
        "allocator",
        "allocator_reserved_unallocated",
        "reserved_unallocated",
        max(0, peak_reserved - peak_allocated),
        keep_zero=True,
    )
    external_value = 0
    if isinstance(external_memory, dict):
        external_value = int(external_memory.get("external_cuda_or_driver_bytes") or 0)
    if isinstance(closure, dict):
        external_value = max(external_value, int(closure.get("external_cuda_or_driver") or 0))
    add("External CUDA", "external", "external_cuda_or_driver", "process_or_driver_gap", external_value)
    return rows, peak_allocated, peak_reserved


def _phase_sort_key(phase: str) -> tuple[int, str]:
    try:
        return (PHASE_ORDER.index(phase), phase)
    except ValueError:
        return (len(PHASE_ORDER), phase)


def _phase_csv_rows(run: RunRecord) -> list[dict[str, Any]]:
    source_rows = _measured_jsonl_rows(run)
    if not source_rows:
        source_rows = [
            {
                "schema_version": run.summary.get("schema_version", 2),
                "step": run.summary.get("selected_step", 0),
                "raw_step": run.summary.get("selected_step", 0),
                "phase": run.summary.get("selected_phase", "selected"),
                "peak_allocated_since_step_begin": run.summary.get("peak_allocated_hbm_bytes", 0),
                "peak_reserved_since_step_begin": run.summary.get("peak_reserved_hbm_bytes", 0),
                "allocated_bytes": run.summary.get("allocated_bytes", 0),
                "reserved_bytes": run.summary.get("reserved_bytes", 0),
                "persistent_bytes": {},
                "saved_activation_bytes_at_peak": {},
                "live_activation_bytes_at_peak": {},
                "peak_growth_bytes_at_peak": {},
            }
        ]
    result: list[dict[str, Any]] = []
    selected_step = str(run.summary.get("selected_step", ""))
    selected_phase = str(run.summary.get("selected_phase", ""))
    for source_row in sorted(
        source_rows,
        key=lambda row: (int(row.get("step") or 0), _phase_sort_key(str(row.get("phase") or ""))),
    ):
        flat_rows, peak_allocated, peak_reserved = _flatten_row(source_row)
        phase = str(source_row.get("phase") or "")
        step = str(source_row.get("step") or "")
        reserved_unallocated = max(0, peak_reserved - peak_allocated)
        allocated_sum = sum(int(row.get("bytes") or 0) for row in flat_rows if row.get("memory_space") == "GPU HBM")
        reserved_gap_sum = sum(
            int(row.get("bytes") or 0)
            for row in flat_rows
            if row.get("component") == "allocator_reserved_unallocated"
        )
        for row in flat_rows:
            if not isinstance(row, dict):
                continue
            value = int(row.get("bytes", 0) or 0)
            is_reserved_stack_row = (
                row.get("memory_space") == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
            )
            result.append(
                {
                    **run.metadata,
                    "run_dir": str(run.run_dir),
                    "schema_version": source_row.get("schema_version", ""),
                    "step": step,
                    "raw_step": source_row.get("raw_step", step),
                    "phase": phase,
                    "is_selected_summary_phase": str(step) == selected_step and phase == selected_phase,
                    "allocated_bytes": int(source_row.get("allocated_bytes", 0) or 0),
                    "reserved_bytes": int(source_row.get("reserved_bytes", 0) or 0),
                    "peak_allocated_hbm_bytes": peak_allocated,
                    "peak_reserved_hbm_bytes": peak_reserved,
                    "reserved_unallocated_bytes": reserved_unallocated,
                    "allocated_stack_sum_bytes": int(allocated_sum),
                    "reserved_stack_sum_bytes": int(allocated_sum + reserved_gap_sum),
                    "memory_space": row.get("memory_space", "-"),
                    "group": row.get("group", "-"),
                    "component": row.get("component", "-"),
                    "kind": row.get("kind", "-"),
                    "bytes": value,
                    "gib": value / GIB,
                    "percent_peak_reserved_hbm": (value * 100.0 / peak_reserved)
                    if peak_reserved > 0 and is_reserved_stack_row
                    else "",
                    "method": row.get("method", "-"),
                    "accuracy": row.get("accuracy", "-"),
                    "allocated_closure_error_bytes": int(peak_allocated) - int(allocated_sum),
                    "reserved_closure_error_bytes": int(peak_reserved) - int(allocated_sum + reserved_gap_sum),
                }
            )
    return result


def _representative_phase_rows(run: RunRecord) -> list[dict[str, Any]]:
    by_phase: dict[str, dict[str, Any]] = {}
    for row in _measured_jsonl_rows(run):
        phase = str(row.get("phase") or "")
        current = by_phase.get(phase)
        if current is None or _selection_key(row) > _selection_key(current):
            by_phase[phase] = row
    return [by_phase[phase] for phase in sorted(by_phase, key=_phase_sort_key)]


def _phase_plot_data(run: RunRecord) -> tuple[list[str], dict[str, list[int]], list[int]]:
    rows = _representative_phase_rows(run)
    if not rows:
        values = _aggregate_summary(run.summary)
        return (
            [str(run.summary.get("selected_phase", "selected") or "selected")],
            {key: [value] for key, value in values.items()},
            [int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0)],
        )
    labels: list[str] = []
    per_phase_values: list[dict[str, int]] = []
    peak_allocated_values: list[int] = []
    keys: set[str] = set()
    for row in rows:
        phase = str(row.get("phase") or "")
        labels.append(phase)
        flat, peak_allocated, _peak_reserved = _flatten_row(row)
        values = _aggregate_rows(flat)
        per_phase_values.append(values)
        peak_allocated_values.append(peak_allocated)
        keys.update(values)
    ordered_keys = sorted(keys, key=_segment_sort_key)
    series = {key: [] for key in ordered_keys}
    for values in per_phase_values:
        for key in ordered_keys:
            series[key].append(values.get(key, 0))
    return labels, series, peak_allocated_values


def _step_series(run: RunRecord) -> tuple[list[int], dict[str, list[int]], list[int]]:
    rows = _load_jsonl(run.jsonl_path)
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("is_warmup") or int(row.get("schema_version") or 0) != 2:
            continue
        try:
            step = int(row.get("step", 0))
        except (TypeError, ValueError):
            continue
        current = selected.get(step)
        if current is None or _selection_key(row) > _selection_key(current):
            selected[step] = row
    if not selected:
        step = int(run.summary.get("selected_step", 1) or 1)
        return (
            [step],
            {key: [value] for key, value in _aggregate_summary(run.summary).items()},
            [int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0)],
        )

    steps = sorted(selected)
    per_step_values: list[dict[str, int]] = []
    peak_allocated_values: list[int] = []
    keys: set[str] = set()
    for step in steps:
        flat, peak_allocated, _peak_reserved = _flatten_row(selected[step])
        values = _aggregate_rows(flat)
        per_step_values.append(values)
        peak_allocated_values.append(peak_allocated)
        keys.update(values)
    ordered_keys = sorted(keys, key=_segment_sort_key)
    series = {key: [] for key in ordered_keys}
    for values in per_step_values:
        for key in ordered_keys:
            series[key].append(values.get(key, 0))
    return steps, series, peak_allocated_values


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
    peak_allocated = int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0)
    peak_reserved = int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0)
    actual_peak_allocated = int(run.summary.get("actual_peak_allocated_hbm_bytes", peak_allocated) or 0)
    actual_peak_reserved = int(run.summary.get("actual_peak_reserved_hbm_bytes", peak_reserved) or 0)
    reserved_unallocated = int(run.summary.get("reserved_unallocated_bytes", 0) or 0)
    external_cuda = int(run.summary.get("external_cuda_or_driver_bytes", 0) or 0)
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        is_reserved_stack_row = row.get("memory_space") == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
        result.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "schema_version": run.summary.get("schema_version", ""),
                "selected_metric": run.summary.get("selected_metric", ""),
                "selected_step": run.summary.get("selected_step", ""),
                "selected_phase": run.summary.get("selected_phase", ""),
                "actual_peak_step": run.summary.get("actual_peak_step", ""),
                "actual_peak_phase": run.summary.get("actual_peak_phase", ""),
                "actual_peak_allocated_hbm_bytes": actual_peak_allocated,
                "actual_peak_reserved_hbm_bytes": actual_peak_reserved,
                "peak_allocated_hbm_bytes": peak_allocated,
                "peak_reserved_hbm_bytes": peak_reserved,
                "reserved_unallocated_bytes": reserved_unallocated,
                "external_cuda_or_driver_bytes": external_cuda,
                "allocated_stack_sum_bytes": int(run.summary.get("allocated_stack_sum_bytes", 0) or 0),
                "reserved_stack_sum_bytes": int(run.summary.get("reserved_stack_sum_bytes", 0) or 0),
                "saved_activation_hbm_bytes_at_peak": int(
                    run.summary.get("saved_activation_hbm_bytes_at_peak", 0) or 0
                ),
                "live_activation_hbm_bytes_at_peak": int(
                    run.summary.get("live_activation_hbm_bytes_at_peak", 0) or 0
                ),
                "activation_hbm_bytes_at_peak": int(
                    run.summary.get("activation_hbm_bytes_at_peak", 0) or 0
                ),
                "temporary_workspace_hbm_bytes_at_peak": int(
                    run.summary.get("temporary_workspace_hbm_bytes_at_peak", 0) or 0
                ),
                "unattributed_allocated_peak_bytes": int(
                    run.summary.get("unattributed_allocated_peak_bytes", 0) or 0
                ),
                "memory_space": row.get("memory_space", "-"),
                "group": row.get("group", "-"),
                "component": row.get("component", "-"),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "gib": value / GIB,
                "percent_peak_reserved_hbm": (value * 100.0 / peak_reserved)
                if peak_reserved > 0 and is_reserved_stack_row
                else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "allocated_closure_ok": bool(run.summary.get("allocated_closure_ok", False)),
                "reserved_closure_ok": bool(run.summary.get("reserved_closure_ok", False)),
                "allocated_closure_error_bytes": int(run.summary.get("allocated_closure_error_bytes", 0) or 0),
                "reserved_closure_error_bytes": int(run.summary.get("reserved_closure_error_bytes", 0) or 0),
            }
        )
    return result


def _actual_peak_csv_rows(run: RunRecord) -> list[dict[str, Any]]:
    rows = run.summary.get("actual_peak_breakdown_rows", [])
    if not isinstance(rows, list) or not rows:
        rows = run.summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return []
    actual_peak_allocated = int(run.summary.get("actual_peak_allocated_hbm_bytes", run.summary.get("peak_allocated_hbm_bytes", 0)) or 0)
    actual_peak_reserved = int(run.summary.get("actual_peak_reserved_hbm_bytes", run.summary.get("peak_reserved_hbm_bytes", 0)) or 0)
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        is_reserved_stack_row = row.get("memory_space") == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
        result.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "schema_version": run.summary.get("schema_version", ""),
                "actual_peak_step": run.summary.get("actual_peak_step", ""),
                "actual_peak_phase": run.summary.get("actual_peak_phase", ""),
                "actual_peak_allocated_hbm_bytes": actual_peak_allocated,
                "actual_peak_reserved_hbm_bytes": actual_peak_reserved,
                "actual_peak_reserved_unallocated_bytes": int(
                    run.summary.get("actual_peak_reserved_unallocated_bytes", max(0, actual_peak_reserved - actual_peak_allocated)) or 0
                ),
                "memory_space": row.get("memory_space", "-"),
                "group": row.get("group", "-"),
                "component": row.get("component", "-"),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "gib": value / GIB,
                "percent_actual_peak_reserved_hbm": (value * 100.0 / actual_peak_reserved)
                if actual_peak_reserved > 0 and is_reserved_stack_row
                else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "actual_peak_allocated_closure_error_bytes": int(
                    run.summary.get("actual_peak_allocated_closure_error_bytes", 0) or 0
                ),
                "actual_peak_reserved_closure_error_bytes": int(
                    run.summary.get("actual_peak_reserved_closure_error_bytes", 0) or 0
                ),
            }
        )
    return result


def _prepare_output(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _peak_ylim_gib(runs: list[RunRecord]) -> float:
    peak = 0
    for run in runs:
        peak = max(
            peak,
            int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0),
            int(run.summary.get("actual_peak_reserved_hbm_bytes", 0) or 0),
        )
    return _nice_y_limit_gib(peak)


def _phase_ylim_gib(runs: list[RunRecord]) -> float:
    peak = 0
    for run in runs:
        for row in _representative_phase_rows(run):
            peak = max(peak, int(row.get("peak_reserved_since_step_begin") or row.get("reserved_bytes") or 0))
        peak = max(peak, int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0))
    return _nice_y_limit_gib(peak)


def _plot_single_peak(run: RunRecord, out_dir: Path, y_limit_gib: float | None, *, actual: bool = False) -> None:
    values = _aggregate_actual_peak_summary(run.summary) if actual else _aggregate_summary(run.summary)
    keys = _positive_segment_keys(values)
    fig, ax = plt.subplots(figsize=(_bar_plot_width(1), 5.8), constrained_layout=True)
    bottom = 0.0
    x_label = run.metadata.get("backend", "run")
    for key in keys:
        value = values[key] / GIB
        if value <= 0:
            continue
        ax.bar(
            [0],
            [value],
            bottom=bottom,
            width=STANDARD_BAR_WIDTH,
            color=_segment_color(key),
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        bottom += value
    peak_allocated_key = "actual_peak_allocated_hbm_bytes" if actual else "peak_allocated_hbm_bytes"
    peak_allocated = int(run.summary.get(peak_allocated_key, 0) or 0) / GIB
    if peak_allocated > 0:
        ax.hlines(
            peak_allocated,
            -STANDARD_BAR_WIDTH / 2.0,
            STANDARD_BAR_WIDTH / 2.0,
            colors=PEAK_LINE_COLOR,
            linestyles="--",
            linewidth=1.2,
        )
    _set_standard_bar_geometry(ax, 1)
    ax.set_xticks([0])
    ax.set_xticklabels([x_label])
    ax.set_ylabel("Memory (GiB)")
    ax.set_title((run.label or run.run_dir.name) + (" actual CUDA peak" if actual else " selected attribution"))
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    _plot_legend(ax, keys, include_peak=peak_allocated > 0)
    fig.savefig(out_dir / ("memory_actual_peak_stack.png" if actual else "memory_peak_stack.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_stacked_bars_on_axis(
    ax: Any,
    steps: list[int],
    series: dict[str, list[int]],
    peak_allocated: list[int],
) -> None:
    labels = [str(step) for step in steps] or ["step"]
    x_positions = list(range(len(labels)))
    bottoms = [0.0 for _label in labels]
    keys = _positive_segment_keys(series)
    for key in keys:
        values = series.get(key, [])
        values_gib = [(int(value or 0) / GIB) for value in values]
        if not any(value > 0.0 for value in values_gib):
            continue
        ax.bar(
            x_positions,
            values_gib,
            bottom=bottoms,
            width=STANDARD_BAR_WIDTH,
            color=_segment_color(key),
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values_gib)]
    for x_position, value in zip(x_positions, peak_allocated):
        peak_gib = int(value or 0) / GIB
        if peak_gib <= 0:
            continue
        ax.hlines(
            peak_gib,
            x_position - STANDARD_BAR_WIDTH / 2.0,
            x_position + STANDARD_BAR_WIDTH / 2.0,
            colors=PEAK_LINE_COLOR,
            linestyles="--",
            linewidth=1.2,
        )
    _set_standard_bar_geometry(ax, len(labels))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)


def _plot_step_stacked_bar(ax: Any, steps: list[int], series: dict[str, list[int]], peak_allocated: list[int]) -> None:
    _plot_step_stacked_bars_on_axis(ax, steps[:1], {key: values[:1] for key, values in series.items()}, peak_allocated[:1])


def _plot_single_steps(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    steps, series, peak_allocated = _step_series(run)
    fig, ax = plt.subplots(figsize=(_bar_plot_width(max(len(steps), 1)), 5.8), constrained_layout=True)
    keys = _positive_segment_keys(series)
    _plot_step_stacked_bars_on_axis(ax, steps, series, peak_allocated)
    ax.set_xlabel("Measured step")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(run.label or run.run_dir.name)
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    _plot_legend(ax, keys, include_peak=bool(peak_allocated))
    fig.savefig(out_dir / "memory_over_steps_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_stacks_on_axis(ax: Any, labels: list[str], series: dict[str, list[int]], peak_allocated: list[int]) -> None:
    x_positions = list(range(len(labels)))
    bottoms = [0.0 for _label in labels]
    for key in _positive_segment_keys(series):
        values = [value / GIB for value in series.get(key, [])]
        if not any(value > 0.0 for value in values):
            continue
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=STANDARD_BAR_WIDTH,
            color=_segment_color(key),
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    for x_position, value in zip(x_positions, peak_allocated):
        value_gib = value / GIB
        if value_gib > 0:
            ax.hlines(
                value_gib,
                x_position - STANDARD_BAR_WIDTH / 2.0,
                x_position + STANDARD_BAR_WIDTH / 2.0,
                colors=PEAK_LINE_COLOR,
                linestyles="--",
                linewidth=1.2,
            )
    _set_standard_bar_geometry(ax, len(labels), min_slots=5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)


def _plot_single_phases(run: RunRecord, out_dir: Path, y_limit_gib: float | None) -> None:
    labels, series, peak_allocated = _phase_plot_data(run)
    fig, ax = plt.subplots(figsize=(_bar_plot_width(len(labels), min_slots=5), 5.8), constrained_layout=True)
    _plot_phase_stacks_on_axis(ax, labels, series, peak_allocated)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(f"{run.label or run.run_dir.name} phase memory attribution")
    if y_limit_gib is not None:
        ax.set_ylim(0, y_limit_gib)
    _plot_legend(ax, _positive_segment_keys(series), include_peak=bool(peak_allocated))
    fig.savefig(out_dir / "memory_by_phase_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_per_run(run: RunRecord, out_dir: Path, clean: bool, y_limit_gib: float | None) -> None:
    _prepare_output(out_dir, clean)
    _write_csv(out_dir / "memory_breakdown.csv", _summary_csv_rows(run))
    _write_csv(out_dir / "memory_actual_peak_breakdown.csv", _actual_peak_csv_rows(run))
    _write_csv(out_dir / "memory_breakdown_by_phase.csv", _phase_csv_rows(run))
    (out_dir / "memory_breakdown_index.json").write_text(
        json.dumps({"run_dir": str(run.run_dir), "summary_path": str(run.summary_path), "jsonl_path": str(run.jsonl_path or "")}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_single_peak(run, out_dir, y_limit_gib)
    _plot_single_peak(run, out_dir, y_limit_gib, actual=True)
    _plot_single_steps(run, out_dir, y_limit_gib)
    phase_y_limit = max(y_limit_gib or 0.0, _phase_ylim_gib([run])) if y_limit_gib is not None else None
    _plot_single_phases(run, out_dir, phase_y_limit)


def _plot_combined_peak(runs: list[RunRecord], out_dir: Path, y_limit_gib: float, *, actual: bool = False) -> None:
    labels = _run_plot_labels(runs)
    x_positions = list(range(len(runs)))
    fig, ax = plt.subplots(figsize=(_bar_plot_width(len(runs)), 6.2), constrained_layout=True)
    bottoms = [0.0 for _run in runs]
    aggregate = _aggregate_actual_peak_summary if actual else _aggregate_summary
    per_run_values = [aggregate(run.summary) for run in runs]
    keys = sorted({key for values in per_run_values for key, value in values.items() if int(value or 0) > 0}, key=_segment_sort_key)
    for key in keys:
        values = [run_values.get(key, 0) / GIB for run_values in per_run_values]
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=STANDARD_BAR_WIDTH,
            color=_segment_color(key),
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    for x_position, run in zip(x_positions, runs):
        peak_allocated_key = "actual_peak_allocated_hbm_bytes" if actual else "peak_allocated_hbm_bytes"
        peak_allocated = int(run.summary.get(peak_allocated_key, 0) or 0) / GIB
        if peak_allocated > 0:
            ax.hlines(
                peak_allocated,
                x_position - STANDARD_BAR_WIDTH / 2.0,
                x_position + STANDARD_BAR_WIDTH / 2.0,
                colors=PEAK_LINE_COLOR,
                linestyles="--",
                linewidth=1.2,
            )
    ax.set_ylabel("Memory (GiB)")
    ax.set_ylim(0, y_limit_gib)
    _set_standard_bar_geometry(ax, len(runs))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(
        "LF Source Actual CUDA-Peak Memory Breakdown"
        if actual
        else "LF Source Selected-Attribution Memory Breakdown"
    )
    ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
    _plot_legend(ax, keys, include_peak=True)
    fig.savefig(out_dir / ("combined_memory_actual_peak_stack.png" if actual else "combined_memory_peak_stack.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_steps(runs: list[RunRecord], out_dir: Path, y_limit_gib: float) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 3 else 1
    nrows = math.ceil(n_runs / ncols)
    plot_data = [(run, *_step_series(run)) for run in runs]
    max_step_count = max((len(steps) for _run, steps, _series, _peak_allocated in plot_data), default=1)
    legend_keys = sorted(
        {
            key
            for _run, _steps, series, _peak_allocated in plot_data
            for key in _positive_segment_keys(series)
        },
        key=_segment_sort_key,
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_subplot_plot_width(max_step_count) * ncols, 4.0 * nrows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        _run, steps, series, peak_allocated = plot_data[idx]
        _plot_step_stacked_bars_on_axis(ax, steps, series, peak_allocated)
        ax.set_title(_run_plot_labels(runs)[idx], fontsize=9)
        ax.set_xlabel("Measured step")
        ax.set_ylim(0, y_limit_gib)
        if idx % ncols == 0:
            ax.set_ylabel("Memory (GiB)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.legend(
        handles=_legend_handles(legend_keys, include_peak=True),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8,
        labelspacing=0.35,
        handlelength=1.4,
    )
    fig.savefig(out_dir / "combined_memory_over_steps_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_phases(runs: list[RunRecord], out_dir: Path, y_limit_gib: float) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 3 else 1
    nrows = math.ceil(n_runs / ncols)
    plot_data = [(run, *_phase_plot_data(run)) for run in runs]
    max_phase_count = max((len(labels) for _run, labels, _series, _peak_allocated in plot_data), default=1)
    legend_keys = sorted(
        {
            key
            for _run, _labels, series, _peak_allocated in plot_data
            for key in _positive_segment_keys(series)
        },
        key=_segment_sort_key,
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_subplot_plot_width(max_phase_count, min_slots=5) * ncols, 4.3 * nrows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        _run, phase_labels, series, peak_allocated = plot_data[idx]
        _plot_phase_stacks_on_axis(ax, phase_labels, series, peak_allocated)
        ax.set_title(_run_plot_labels(runs)[idx], fontsize=9)
        ax.set_xlabel("Phase")
        ax.set_ylim(0, y_limit_gib)
        if idx % ncols == 0:
            ax.set_ylabel("Memory (GiB)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.legend(
        handles=_legend_handles(legend_keys, include_peak=True),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8,
        labelspacing=0.35,
        handlelength=1.4,
    )
    fig.savefig(out_dir / "combined_memory_by_phase_stacked.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _group_label(run: RunRecord) -> str:
    metadata = run.metadata
    parts = [
        metadata.get("workload", ""),
        f"b{metadata.get('batch_size', '')}" if metadata.get("batch_size") else "",
        f"drop{metadata.get('lora_dropout', '').replace('.', '')}" if metadata.get("lora_dropout") else "",
        metadata.get("precision", ""),
        metadata.get("backend", ""),
        metadata.get("profiler", ""),
        f"router{metadata.get('router_mode', '')}" if metadata.get("router_mode") else "",
        metadata.get("expact", ""),
        metadata.get("attnact", ""),
        metadata.get("recompute", ""),
        f"pol{metadata.get('expert_policy', '')}" if metadata.get("expert_policy") else "",
    ]
    return _safe_label("-".join(part for part in parts if part))


def _write_grouped_combined(runs: list[RunRecord], out_dir: Path, clean: bool, y_limit_gib: float) -> None:
    groups: dict[str, list[RunRecord]] = {}
    for run in runs:
        groups.setdefault(_group_label(run), []).append(run)
    for label, group_runs in sorted(groups.items()):
        _write_combined(group_runs, out_dir / label, clean, y_limit_gib, write_groups=False)


def _write_combined(
    runs: list[RunRecord],
    out_dir: Path,
    clean: bool,
    y_limit_gib: float,
    *,
    write_groups: bool = True,
) -> None:
    _prepare_output(out_dir, clean)
    all_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for run in runs:
        all_rows.extend(_summary_csv_rows(run))
        actual_rows.extend(_actual_peak_csv_rows(run))
        phase_rows.extend(_phase_csv_rows(run))
        index_rows.append(
            {
                **run.metadata,
                "run_dir": str(run.run_dir),
                "summary_path": str(run.summary_path),
                "jsonl_path": str(run.jsonl_path or ""),
                "schema_version": run.summary.get("schema_version", ""),
                "selected_metric": run.summary.get("selected_metric", ""),
                "peak_allocated_hbm_bytes": int(run.summary.get("peak_allocated_hbm_bytes", 0) or 0),
                "peak_reserved_hbm_bytes": int(run.summary.get("peak_reserved_hbm_bytes", 0) or 0),
                "actual_peak_step": run.summary.get("actual_peak_step", ""),
                "actual_peak_phase": run.summary.get("actual_peak_phase", ""),
                "actual_peak_allocated_hbm_bytes": int(run.summary.get("actual_peak_allocated_hbm_bytes", 0) or 0),
                "actual_peak_reserved_hbm_bytes": int(run.summary.get("actual_peak_reserved_hbm_bytes", 0) or 0),
                "reserved_unallocated_bytes": int(run.summary.get("reserved_unallocated_bytes", 0) or 0),
                "external_cuda_or_driver_bytes": int(run.summary.get("external_cuda_or_driver_bytes", 0) or 0),
                "allocated_stack_sum_bytes": int(run.summary.get("allocated_stack_sum_bytes", 0) or 0),
                "reserved_stack_sum_bytes": int(run.summary.get("reserved_stack_sum_bytes", 0) or 0),
                "activation_hbm_bytes_at_peak": int(run.summary.get("activation_hbm_bytes_at_peak", 0) or 0),
                "temporary_workspace_hbm_bytes_at_peak": int(
                    run.summary.get("temporary_workspace_hbm_bytes_at_peak", 0) or 0
                ),
                "unattributed_allocated_peak_bytes": int(run.summary.get("unattributed_allocated_peak_bytes", 0) or 0),
            }
        )
    _write_csv(out_dir / "combined_memory_breakdown.csv", all_rows)
    _write_csv(out_dir / "combined_memory_actual_peak_breakdown.csv", actual_rows)
    _write_csv(out_dir / "combined_memory_breakdown_by_phase.csv", phase_rows)
    _write_csv(out_dir / "memory_breakdown_index.csv", index_rows)
    (out_dir / "memory_breakdown_index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_combined_peak(runs, out_dir, y_limit_gib)
    _plot_combined_peak(runs, out_dir, y_limit_gib, actual=True)
    _plot_combined_steps(runs, out_dir, y_limit_gib)
    _plot_combined_phases(runs, out_dir, _phase_ylim_gib(runs))
    if write_groups:
        _write_grouped_combined(runs, out_dir, clean, y_limit_gib)


def main() -> None:
    args = _parse_args()
    runs = _load_runs(args)
    if not runs:
        raise SystemExit(_no_runs_message(args))

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
