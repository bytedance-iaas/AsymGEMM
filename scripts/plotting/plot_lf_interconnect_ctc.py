#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


CTC_METRICS = ("ctc_rx", "ctc_tx")
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
    "route",
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
    "router_mode",
    "asymm_expert_act_offload",
    "expact",
    "asymm_attn_act_offload",
    "attnact",
    "asymm_layer_act_offload",
    "layeract",
    "asymm_layer_gc",
    "layergc",
    "route",
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
    "router_mode",
    "asymm_expert_act_offload",
    "expact",
    "asymm_attn_act_offload",
    "attnact",
    "asymm_layer_act_offload",
    "layeract",
    "asymm_layer_gc",
    "layergc",
    "route",
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
METRIC_LABELS = {
    "ctc_rx": "C2C RX",
    "ctc_tx": "C2C TX",
}
METRIC_COLORS = {
    "ctc_rx": "#1f77b4",
    "ctc_tx": "#d62728",
}
# Phase background shading for the per-step timeline (forward -> backward -> optimizer).
# Hues chosen distinct from the RX (blue) and TX (red) lines so the lines stay readable.
PHASE_SHADE = (
    ("forward", "Forward", "#4C9A4C"),      # green
    ("backward", "Backward", "#8A6FC1"),    # purple
    ("optimizer", "Optimizer", "#D9A300"),  # gold
)
PHASE_SHADE_ALPHA = 0.17
RUN_DIR_RE = re.compile(r"^b(?P<batch_size>[0-9]+)_s(?P<seq_len>[0-9]+)_ga(?P<grad_accum>[0-9]+)$")
POINTS_PER_STEP = 100  # resample each measured step to this many points
TIMELINE_VARIANTS = (
    ("avg", "mean", "avg per bucket", "timeline_avg.png"),
    ("max", "max", "max per bucket", "timeline_max.png"),
)


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    profile_path: Path
    metadata: dict[str, str]
    by_step: dict[int, dict[str, dict[str, float]]]
    timelines: dict[str, dict[str, list[tuple[float, float]]]]
    phase_spans: tuple[tuple[str, float, float], ...] = ()

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
            self.metadata.get("route", ""),
            self.metadata.get("liger_loss", ""),
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


def _layeract_label(value: Any) -> str:
    return "layeract1" if _normalize_bool_config(value) == "true" else "layeract0"


def _layergc_label(value: Any) -> str:
    return "layergc1" if _normalize_bool_config(value) == "true" else "layergc0"


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


def _parse_layeract_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"layeract1", "layeracttrue"}:
        return "true", "layeract1"
    if value in {"layeract0", "layeractfalse"}:
        return "false", "layeract0"
    return None


def _parse_layergc_part(part: str) -> tuple[str, str] | None:
    value = part.strip().lower()
    if value in {"layergc1", "layergctrue"}:
        return "true", "layergc1"
    if value in {"layergc0", "layergcfalse"}:
        return "false", "layergc0"
    return None


def _parse_liger_loss_part(part: str) -> str | None:
    value = part.strip().lower()
    if value in {"ligerloss0", "ligerloss1"}:
        return value
    return None


def _known_optional_job_axis(part: str) -> bool:
    value = part.strip().lower()
    return (
        value in {
            "layeract0",
            "layeract1",
            "layeractfalse",
            "layeracttrue",
            "layergc0",
            "layergc1",
            "layergcfalse",
            "layergctrue",
        }
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
    layeract_value = "false"
    layeract = "layeract0"
    layergc_value = "false"
    layergc = "layergc0"
    route = "route_missing"
    liger_loss = "ligerloss0"

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
        parsed_layeract = _parse_layeract_part(part)
        if parsed_layeract is not None:
            layeract_value, layeract = parsed_layeract
            continue
        parsed_layergc = _parse_layergc_part(part)
        if parsed_layergc is not None:
            layergc_value, layergc = parsed_layergc
            continue
        parsed_liger_loss = _parse_liger_loss_part(part)
        if parsed_liger_loss is not None:
            liger_loss = parsed_liger_loss
            continue
        if part.startswith("route"):
            route = part
            continue
        # Unknown tail parts are config axes. Plotting should not reject a run
        # just because the driver grew a new folder label.
        continue

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
        "asymm_layer_act_offload": layeract_value,
        "layeract": layeract,
        "asymm_layer_gc": layergc_value,
        "layergc": layergc,
        "route": route,
        "liger_loss": liger_loss,
    }


def _filter_values(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part:
                result.add(part.lower())
    return result


def _apply_workload_filters(args: argparse.Namespace) -> None:
    workload_tuples: set[tuple[str, str, str]] = set()
    for value in args.workloads:
        for workload in str(value).replace(",", " ").split():
            fields = workload.split("|")
            if len(fields) != 3 or not all(field.isdigit() and int(field) > 0 for field in fields):
                raise SystemExit(
                    "--workloads items must be seq_len|per_device_batch_size|gradient_accumulation_steps, "
                    f"got {workload!r}"
                )
            workload_tuples.add((fields[0], fields[1], fields[2]))
    args.workload_tuples = workload_tuples


def _seq_len_from_run_dir_name(name: str) -> str:
    match = RUN_DIR_RE.match(name)
    return match.group("seq_len") if match is not None else ""


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


def _infer_metadata(profile_path: Path, profile: dict[str, Any]) -> dict[str, str] | None:
    run_dir = profile_path.parent
    job_root = run_dir.parent
    config_root = job_root.parent
    config = _source_config(run_dir, profile)
    run_dir_match = RUN_DIR_RE.match(run_dir.name)
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
    layeract_value = job_meta["asymm_layer_act_offload"]
    layeract = job_meta["layeract"]
    layergc_value = job_meta["asymm_layer_gc"]
    layergc = job_meta["layergc"]
    if not policy_part.startswith("pol") or not router_part.startswith("router"):
        return None
    if recompute_part not in {"norecomp", "recomp", "unsloth"}:
        return None

    expert_policy = str(config.get("expert_policy") or policy_part[len("pol") :] or "none")
    router_mode = str(config.get("router_mode") or router_part[len("router") :])

    seq_len = str(
        config.get("seq_len")
        or config.get("cutoff_len")
        or (run_dir_match.group("seq_len") if run_dir_match else "")
    )
    if not seq_len:
        seq_len = _seq_len_from_run_dir_name(run_dir.name)
    expact_value = _normalize_bool_config(config.get("asymm_expert_act_offload", expact_value))
    expact = _expact_label(expact_value)
    attnact_value = _normalize_bool_config(config.get("asymm_attn_act_offload", attnact_value))
    attnact = _attnact_label(attnact_value)
    layeract_value = _normalize_bool_config(config.get("asymm_layer_act_offload", layeract_value))
    layeract = _layeract_label(layeract_value)
    layergc_config_value = config.get("asymm_layer_gc", config.get("asym_layer_glue_gc_enabled", layergc_value))
    layergc_value = _normalize_bool_config(layergc_config_value)
    layergc = _layergc_label(layergc_value)
    liger_loss = str(config.get("liger_loss") or job_meta.get("liger_loss") or "ligerloss0").lower()
    if liger_loss not in {"ligerloss0", "ligerloss1"}:
        liger_loss = "ligerloss0"

    metadata = {
        "workload": str(config.get("workload") or config_root.name.split("__", 1)[0]),
        "backend": str(config.get("backend") or backend_part),
        "router_mode": router_mode,
        "asymm_expert_act_offload": expact_value,
        "expact": expact,
        "asymm_attn_act_offload": attnact_value,
        "attnact": attnact,
        "asymm_layer_act_offload": layeract_value,
        "layeract": layeract,
        "asymm_layer_gc": layergc_value,
        "layergc": layergc,
        "route": job_meta.get("route", "route_missing"),
        "liger_loss": liger_loss,
        "profiler": str(profiler_part),
        "recompute": recompute_part,
        "expert_policy": expert_policy,
        "seq_len": seq_len,
        "precision": str(config.get("precision") or ""),
        "batch_size": str(config.get("batch_size") or (run_dir_match.group("batch_size") if run_dir_match else "")),
        "gradient_accumulation_steps": str(
            config.get("gradient_accumulation_steps") or (run_dir_match.group("grad_accum") if run_dir_match else "")
        ),
        "lora_dropout": str(config.get("lora_dropout") if config.get("lora_dropout") is not None else ""),
        "lora_rank": str(config.get("lora_rank") if config.get("lora_rank") is not None else ""),
        "lora_alpha": str(config.get("lora_alpha") if config.get("lora_alpha") is not None else ""),
        "lora_target": str(config.get("lora_target") or ""),
        "config": config_root.name,
    }
    return metadata


def _matches_filters(record: RunRecord, args: argparse.Namespace) -> bool:
    if getattr(args, "workload_tuples", set()):
        workload_tuple = (
            record.metadata.get("seq_len", ""),
            record.metadata.get("batch_size", ""),
            record.metadata.get("gradient_accumulation_steps", ""),
        )
        if workload_tuple not in args.workload_tuples:
            return False
    filters = {
        "workload": _filter_values(args.workload),
        "backend": _filter_values(args.backend),
        "profiler": _filter_values(args.profiler),
        "router_mode": _filter_values(args.router_mode),
        "expact": _filter_values(getattr(args, "expact", [])),
        "attnact": _filter_values(getattr(args, "attnact", [])),
        "layeract": _filter_values(getattr(args, "layeract", [])),
        "layergc": _filter_values(getattr(args, "layergc", [])),
        "liger_loss": _filter_values(getattr(args, "liger_loss", [])),
        "recompute": _filter_values(args.recompute),
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


def _phase_spans(rows: list[Any], bounds: list[tuple[float, float]]) -> list[tuple[str, float, float]]:
    """Per-step forward/backward/optimizer x-bands (step-index coords) for timeline shading.

    Each measured step occupies [k, k+1). Forward/backward come from the NVTX phase ranges
    (durations in ns, clock-rate-invariant), drawn contiguous forward -> backward; the leftover
    tail of the measured step (bounds[k], which spans the full trainer step incl. the optimizer)
    is the post-backward optimizer+overhead region. Without measured-step bounds the optimizer
    tail is unknown, so only forward/backward are shaded.
    """
    per_step: dict[int, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("scope") != "phase":
            continue
        if str(row.get("metric") or "") != "ctc_rx":  # one metric only -> no double count
            continue
        phase = str(row.get("phase") or "")
        if phase not in ("forward", "backward"):
            continue
        try:
            step = int(row.get("step"))
            dur = (int(row.get("end_ns")) - int(row.get("start_ns"))) / 1e9
        except (TypeError, ValueError):
            continue
        if dur > 0:
            per_step.setdefault(step, {})[phase] = dur
    spans: list[tuple[str, float, float]] = []
    for index, step in enumerate(sorted(per_step)):
        durs = per_step[step]
        fwd_s = durs.get("forward", 0.0)
        bwd_s = durs.get("backward", 0.0)
        have_bounds = index < len(bounds)
        denom = (bounds[index][1] - bounds[index][0]) if have_bounds else (fwd_s + bwd_s)
        if denom <= 0:
            continue
        f = fwd_s / denom
        b = bwd_s / denom
        if f + b > 1.0:  # guard clock skew so phases never overflow the step
            f, b = f / (f + b), b / (f + b)
        x = float(index)
        if f > 0:
            spans.append(("forward", x, x + f))
        if b > 0:
            spans.append(("backward", x + f, x + f + b))
        # optimizer = tail of the measured step after backward; only defined when backward exists
        # (a missing backward range -> incomplete step, leave the tail unshaded rather than mislabel).
        if have_bounds and b > 0 and (1.0 - f - b) > 1e-4:
            spans.append(("optimizer", x + f + b, x + 1.0))
    return spans


def _phase_duration_seconds(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _to_float(row.get(key))
        if value > 0.0:
            return value / 1000.0
    return 0.0


def _phase_spans_from_step_samples(
    profile: dict[str, Any],
    bounds: list[tuple[float, float]],
) -> list[tuple[str, float, float]]:
    source_profile = profile.get("source_profile")
    if not isinstance(source_profile, dict):
        return []
    step_samples = source_profile.get("step_samples")
    rows = step_samples.get("rows") if isinstance(step_samples, dict) else None
    if not isinstance(rows, list):
        return []
    measured_rows = [row for row in rows if isinstance(row, dict) and not row.get("is_warmup")]
    spans: list[tuple[str, float, float]] = []
    for index, row in enumerate(measured_rows):
        have_bounds = index < len(bounds)
        denom = (bounds[index][1] - bounds[index][0]) if have_bounds else 0.0
        if denom <= 0.0:
            denom = _phase_duration_seconds(row, "trainer_e2e_step_milliseconds", "step_milliseconds")
        if denom <= 0.0:
            continue
        fwd_s = _phase_duration_seconds(row, "forward_milliseconds", "source_forward_milliseconds")
        bwd_s = _phase_duration_seconds(row, "backward_milliseconds", "source_backward_milliseconds")
        f = fwd_s / denom
        b = bwd_s / denom
        if f + b > 1.0:
            f, b = f / (f + b), b / (f + b)
        x = float(index)
        if f > 0.0:
            spans.append(("forward", x, x + f))
        if b > 0.0:
            spans.append(("backward", x + f, x + f + b))
        if (1.0 - f - b) > 1e-4:
            spans.append(("optimizer", x + f + b, x + 1.0))
    return spans


def _phase_spans_from_profile(
    profile: dict[str, Any],
    range_rows: list[Any],
    bounds: list[tuple[float, float]],
) -> list[tuple[str, float, float]]:
    spans = _phase_spans_from_step_samples(profile, bounds)
    if spans:
        return spans
    return _phase_spans(range_rows, bounds)


def _measured_step_bounds(profile: dict[str, Any]) -> list[tuple[float, float]]:
    """[(start, end), ...] trainer-e2e seconds for non-warmup steps (source_profile.step_samples)."""
    source_profile = profile.get("source_profile")
    if not isinstance(source_profile, dict):
        return []
    step_samples = source_profile.get("step_samples")
    rows = step_samples.get("rows") if isinstance(step_samples, dict) else None
    if not isinstance(rows, list):
        return []
    bounds: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("is_warmup"):
            continue
        try:
            start = float(row.get("trainer_e2e_start_seconds"))
            end = float(row.get("trainer_e2e_end_seconds"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            bounds.append((start, end))
    return sorted(bounds, key=lambda item: item[0])


def _affine_to_window(
    series: list[tuple[float, float]], window: tuple[float, float] | None
) -> list[tuple[float, float]]:
    """Map a series' own elapsed range onto [window.start, window.end] (CTC Nsight clock -> steps)."""
    if not series or window is None:
        return series
    start, end = window
    lo, hi = series[0][0], series[-1][0]
    if hi <= lo:
        return series
    scale = (end - start) / (hi - lo)
    return [(start + (x - lo) * scale, y) for x, y in series]


def _resample_per_step(
    series: list[tuple[float, float]], bounds: list[tuple[float, float]], points_per_step: int, agg: str = "max"
) -> list[tuple[float, float]]:
    """Resample (elapsed, value) to points_per_step per measured step; x = step_index + fraction.

    agg "max" reports each bucket's peak (right for saturation); "mean" reports the average. Falls
    back to one synthetic step over the series' own span when bounds are unavailable.
    """
    if not series or points_per_step <= 0:
        return []
    if not bounds:
        bounds = [(series[0][0], series[-1][0])]
    out: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(bounds):
        if end <= start:
            continue
        buckets: list[list[float]] = [[] for _ in range(points_per_step)]
        for x, y in series:
            if start <= x < end:
                slot = int((x - start) / (end - start) * points_per_step)
                buckets[min(slot, points_per_step - 1)].append(y)
        for slot, values in enumerate(buckets):
            if values:
                value = max(values) if agg == "max" else sum(values) / len(values)
                out.append((index + (slot + 0.5) / points_per_step, value))
    return out


def _ctc_timeline_series(
    metrics: dict[str, Any], bounds: list[tuple[float, float]]
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Per-step-normalized CTC RX/TX timelines keyed by timeline variant, then metric label."""
    rows = metrics.get("timeseries", {}).get("rows", [])
    if not isinstance(rows, list):
        return {}
    raw: dict[str, dict[int, float]] = {"ctc_rx": {}, "ctc_tx": {}}
    for row in rows:
        if not isinstance(row, dict) or row.get("metric_key") not in raw:
            continue
        try:
            timestamp = int(row.get("timestamp_ns"))
            value = float(row.get("value_percent"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = row["metric_key"]
        raw[key][timestamp] = max(raw[key].get(timestamp, 0.0), min(100.0, max(0.0, value)))
    window = (bounds[0][0], bounds[-1][1]) if bounds else None
    timelines: dict[str, dict[str, list[tuple[float, float]]]] = {name: {} for name, _agg, _suffix, _file in TIMELINE_VARIANTS}
    for key in ("ctc_rx", "ctc_tx"):
        points = [(float(ts), val) for ts, val in sorted(raw[key].items())]
        normalized = _affine_to_window(points, window)
        for variant, agg, _suffix, _filename in TIMELINE_VARIANTS:
            resampled = _resample_per_step(normalized, bounds, POINTS_PER_STEP, agg)
            if resampled:
                timelines[variant][METRIC_LABELS[key]] = resampled
    return {variant: series for variant, series in timelines.items() if series}


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
        metadata = _infer_metadata(profile_path, profile)
        if metadata is None:
            continue
        bounds = _measured_step_bounds(profile)
        record = RunRecord(
            run_dir=profile_path.parent,
            profile_path=profile_path,
            metadata=metadata,
            by_step=by_step,
            timelines=_ctc_timeline_series(metrics, bounds),
            phase_spans=tuple(_phase_spans_from_profile(profile, rows, bounds)),
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
            run.metadata.get("route", ""),
            run.metadata.get("liger_loss", ""),
            run.metadata.get("recompute", ""),
            run.metadata.get("expert_policy", ""),
            str(run.run_dir),
        ),
    )


def _prepare_output(path: Path, clean: bool) -> tuple[Path, Path]:
    if clean and path.exists():
        shutil.rmtree(path)
    plots_dir = path  # plots sit directly in the metric folder; data/ stays nested
    data_dir = path / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


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


def _plot_timeline(
    runs: list[RunRecord],
    plots_dir: Path,
    *,
    variant: str,
    title_suffix: str,
    filename: str,
) -> None:
    n_runs = len(runs)
    ncols = 2 if n_runs > 2 else 1
    nrows = math.ceil(n_runs / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 3.4 * nrows), sharey=True, squeeze=False, constrained_layout=True)
    color_for = {METRIC_LABELS["ctc_rx"]: METRIC_COLORS["ctc_rx"], METRIC_LABELS["ctc_tx"]: METRIC_COLORS["ctc_tx"]}
    legend_handles: dict[str, Any] = {}
    for idx, run in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        n_steps = 0
        # Phase background bands (forward -> backward -> optimizer) behind the RX/TX lines.
        for phase_key, phase_label, phase_color in PHASE_SHADE:
            drawn = False
            for span_key, x0, x1 in run.phase_spans:
                if span_key != phase_key or x1 <= x0:
                    continue
                ax.axvspan(x0, x1, color=phase_color, alpha=PHASE_SHADE_ALPHA, linewidth=0, zorder=0)
                drawn = True
            if drawn:
                legend_handles.setdefault(
                    phase_label, Patch(facecolor=phase_color, alpha=PHASE_SHADE_ALPHA, label=phase_label)
                )
        # CTC RX/TX already per-step-normalized (x = measured step).
        for label in (METRIC_LABELS["ctc_rx"], METRIC_LABELS["ctc_tx"]):
            series = run.timelines.get(variant, {}).get(label, [])
            if not series:
                continue
            xs = [point[0] for point in series]
            ys = [point[1] for point in series]
            n_steps = max(n_steps, int(math.ceil(max(xs))))
            (line,) = ax.plot(xs, ys, label=label, color=color_for[label], linewidth=1.5)
            legend_handles.setdefault(label, line)
        for boundary in range(1, n_steps):
            ax.axvline(boundary, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0.0, float(max(n_steps, 1)))
        ax.set_xticks(list(range(n_steps + 1)))
        ax.set_title(run.label or run.run_dir.name, fontsize=9)
        ax.set_xlabel("Measured step")
        ax.set_ylim(bottom=0.0, top=max(100.0, ax.get_ylim()[1]))
        ax.grid(True, axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("C2C Saturation (%)")
    for idx in range(n_runs, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if legend_handles:
        labels = list(legend_handles)
        fig.legend(
            [legend_handles[name] for name in labels],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=max(1, min(len(labels), 6)),
            frameon=False,
        )
    fig.suptitle(f"C2C Saturation Timeline by Measured Step ({POINTS_PER_STEP} pts/step, {title_suffix})", fontsize=12)
    fig.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timelines(runs: list[RunRecord], plots_dir: Path) -> None:
    for variant, _agg, title_suffix, filename in TIMELINE_VARIANTS:
        _plot_timeline(runs, plots_dir, variant=variant, title_suffix=title_suffix, filename=filename)


def _write_readme(out_dir: Path, *, runs: list[RunRecord], reason: str | None = None) -> None:
    lines = [
        "# LF C2C / CTC Artifacts",
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
                f"- `timeline_avg.png`: subplots per run; x-axis is the measured step ({POINTS_PER_STEP} pts/step, mean per bucket), y-axis is C2C RX/TX saturation percent.",
                f"- `timeline_max.png`: subplots per run; x-axis is the measured step ({POINTS_PER_STEP} pts/step, max per bucket), y-axis is C2C RX/TX saturation percent.",
                "- `data/summary.csv`: one row per run and C2C direction.",
                "- `data/step_summary.csv`: one row per run, step, and C2C direction.",
                "- `data/index.csv` / `data/index.json`: input run index.",
                "",
                f"Runs included: {len(runs)}.",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_empty_outputs(out_dir: Path, clean: bool, reason: str) -> None:
    _plots_dir, data_dir = _prepare_output(out_dir, clean)
    _write_csv(data_dir / "summary.csv", [], SUMMARY_FIELDS)
    _write_csv(data_dir / "step_summary.csv", [], STEP_FIELDS)
    _write_csv(data_dir / "index.csv", [], INDEX_FIELDS)
    (data_dir / "index.json").write_text("[]\n", encoding="utf-8")
    _write_readme(out_dir, runs=[], reason=reason)


def _write_outputs(runs: list[RunRecord], out_dir: Path, clean: bool) -> None:
    plots_dir, data_dir = _prepare_output(out_dir, clean)
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
    _write_csv(data_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(data_dir / "step_summary.csv", step_rows, STEP_FIELDS)
    _write_csv(data_dir / "index.csv", index_rows, INDEX_FIELDS)
    (data_dir / "index.json").write_text(json.dumps(index_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_timelines(runs, plots_dir)
    _write_readme(out_dir, runs=runs)


def main() -> None:
    args = _parse_args()
    _apply_workload_filters(args)
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
