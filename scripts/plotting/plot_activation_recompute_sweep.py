#!/usr/bin/env python3
"""Plot activation-recompute sequence-length sweeps from LoRA driver outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MIB = 1024.0**2
RESULT_RE = re.compile(
    r"^(?P<precision>.+?)_lora-sft_b(?P<batch_size>[0-9]+)_s(?P<seq_len>[0-9]+)_"
    r"(?P<recompute>recomp|norecomp)"
    r"(?:_expertpolicy(?P<expert_policy>[A-Za-z0-9.-]+))?"
    r"_(?P<tail>.+)$"
)
FLAT_SEQ_RE = re.compile(r"^s(?P<seq_len>[0-9]+)$")
PROFILERS = ("source", "nsys", "cpu", "ncu")
BACKENDS = ("asym", "torch", "kt")
LINEAR_REGION_R2_THRESHOLD = 0.99
LINEAR_REGION_RATIO_CV_THRESHOLD = 0.08
MIN_LINEAR_REGION_POINTS = 4
SUBLINEAR_SLOPE_TOLERANCE = 0.08
SUBLINEAR_COLOR = "green"
SUBLINEAR_ALPHA = 0.055
STYLE_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#000000",
    "#6A3D9A",
    "#A6761D",
    "#E31A1C",
    "#17BECF",
    "#F0E442",
    "#7F7F7F",
    "#1B9E77",
    "#E7298A",
    "#8DD3C7",
    "#FB8072",
    "#80B1D3",
    "#FDB462",
    "#B3DE69",
    "#FCCDE5",
    "#BC80BD",
    "#CCEBC5",
    "#FFED6F",
    "#1F78B4",
    "#33A02C",
    "#E31A1C",
    "#FF7F00",
    "#6A3D9A",
    "#B15928",
    "#A6CEE3",
    "#B2DF8A",
    "#FB9A99",
    "#FDBF6F",
    "#CAB2D6",
    "#FFFF99",
    "#8C564B",
    "#17BECF",
    "#7F7F7F",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
)
BACKEND_MARKERS = {
    "asym": "^",
    "torch": "o",
    "kt": "s",
}
ROOT_OUTPUT_FILES = (
    "activation_recompute_sweep_index.csv",
    "activation_recompute_sweep_index.json",
    "step_samples_index.csv",
    "step_samples_index.json",
)
COMBINED_OUTPUT_FILES = (
    "combined_step_samples_index.csv",
    "combined_step_samples_index.json",
    "combined_forward_end_memory_vs_seq.png",
    "combined_forward_peak_memory_vs_seq.png",
    "combined_backward_start_memory_vs_seq.png",
    "combined_backward_peak_memory_vs_seq.png",
    "combined_peak_hbm_vs_seq.png",
    "combined_timing_vs_seq.png",
    "combined_forward_end_memory_vs_step.png",
    "combined_forward_peak_memory_vs_step.png",
    "combined_backward_start_memory_vs_step.png",
    "combined_backward_peak_memory_vs_step.png",
    "combined_peak_hbm_vs_step.png",
    "combined_timing_vs_step.png",
    "combined_forward_timing_vs_step.png",
    "combined_backward_timing_vs_step.png",
    "combined_loss_vs_step.png",
    "combined_forward_end_memory_vs_expert_threshold.png",
    "combined_forward_peak_memory_vs_expert_threshold.png",
    "combined_backward_start_memory_vs_expert_threshold.png",
    "combined_backward_peak_memory_vs_expert_threshold.png",
    "combined_peak_hbm_vs_expert_threshold.png",
    "combined_timing_vs_expert_threshold.png",
    "combined_forward_end_memory_vs_expert_tok_threshold.png",
    "combined_forward_peak_memory_vs_expert_tok_threshold.png",
    "combined_backward_start_memory_vs_expert_tok_threshold.png",
    "combined_backward_peak_memory_vs_expert_tok_threshold.png",
    "combined_peak_hbm_vs_expert_tok_threshold.png",
    "combined_timing_vs_expert_tok_threshold.png",
    "combined_forward_end_memory_vs_expert_tok_act_threshold.png",
    "combined_forward_peak_memory_vs_expert_tok_act_threshold.png",
    "combined_backward_start_memory_vs_expert_tok_act_threshold.png",
    "combined_backward_peak_memory_vs_expert_tok_act_threshold.png",
    "combined_peak_hbm_vs_expert_tok_act_threshold.png",
    "combined_timing_vs_expert_tok_act_threshold.png",
)
GROUP_OUTPUT_FILES = (
    "sweep_summary.csv",
    "sweep_summary.json",
    "step_samples.csv",
    "step_samples.json",
    "forward_end_memory_vs_seq.png",
    "forward_peak_memory_vs_seq.png",
    "backward_start_memory_vs_seq.png",
    "backward_peak_memory_vs_seq.png",
    "peak_hbm_vs_seq.png",
    "timing_vs_seq.png",
    "forward_end_memory_vs_step.png",
    "forward_peak_memory_vs_step.png",
    "backward_start_memory_vs_step.png",
    "backward_peak_memory_vs_step.png",
    "peak_hbm_vs_step.png",
    "timing_vs_step.png",
    "forward_timing_vs_step.png",
    "backward_timing_vs_step.png",
    "loss_vs_step.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "profiling")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <input-root>/activation_recompute_plots.",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=None,
        help="Default: <output-dir>/combined.",
    )
    parser.add_argument("--precision", default="")
    parser.add_argument("--workload", action="append", default=[], help="Workload label to include, e.g. moe-604m-a38m-l2.")
    parser.add_argument("--backend", action="append", default=[], choices=list(BACKENDS))
    parser.add_argument("--profiler", action="append", default=[], choices=list(PROFILERS))
    parser.add_argument(
        "--recompute",
        action="append",
        default=[],
        choices=["norecomp", "recomp", "no_recompute", "recompute"],
        help="Activation recompute mode to include. Repeat for both.",
    )
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--seq-len", "--seq-lens", dest="seq_lens", nargs="+", type=int, default=[])
    parser.add_argument("--expert-recompute-policy", action="append", default=[])
    parser.add_argument(
        "--expert-recompute-policies",
        nargs="+",
        default=[],
        help="Expert policy filter. Accepts none, tok-le0, tok-le128, tok-ge128, tok64-256, or -act variants.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previously generated plot artifacts from the output directory before writing this filtered run.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not write combined plots.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the combined index and combined plots.",
    )
    return parser.parse_args()


def safe_label(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch == "-" else "_" for ch in value).strip("_-")


def split_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tokens


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def output_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return resolve_path(args.output_dir)
    return resolve_path(args.input_root) / "activation_recompute_plots"


def combined_output_root(args: argparse.Namespace, root: Path) -> Path:
    if args.combined_output_dir is not None:
        combined_dir = resolve_path(args.combined_output_dir)
        if combined_dir.name == "combined" and combined_dir.parent.name == "_combined":
            return combined_dir.parent
        return combined_dir
    return root / "combined"


def precision_from_path(path: Path) -> str:
    for parent in (path, *path.parents):
        for prefix in ("lora_e2e_", "lora_lf_"):
            if parent.name.startswith(prefix):
                return parent.name.removeprefix(prefix)
    return ""


def parse_expert_policy_spec(value: str | None) -> dict[str, Any]:
    def result(
        *,
        spec: str,
        label: str,
        policy: str,
        token_threshold: int = 0,
        token_min: int = 1,
        token_max: int | None = None,
        activation_policy: str = "save_all",
        activation_threshold: int = 0,
        activation_min: int = 1,
        activation_max: int | None = None,
    ) -> dict[str, Any]:
        return {
            "expert_recompute_policy_spec": spec,
            "expert_policy_label": label,
            "expert_recompute_impl": "none",
            "expert_recompute_policy": policy,
            "expert_recompute_threshold": int(token_threshold),
            "expert_recompute_token_min": int(token_min),
            "expert_recompute_token_max": None if token_max is None else int(token_max),
            "expert_activation_save_policy": activation_policy,
            "expert_activation_save_threshold": int(activation_threshold),
            "expert_activation_save_token_min": int(activation_min),
            "expert_activation_save_token_max": None if activation_max is None else int(activation_max),
        }

    if value is None:
        return result(spec="none", label="none", policy="none")
    spec = str(value).strip()
    if spec == "none":
        return result(spec="none", label="none", policy="none")
    if spec == "tok-le0":
        return result(spec="tok-le0", label="tok-le0", policy="none")
    if spec == "tok-le0-act":
        return result(spec="tok-le0-act", label="tok-le0-act", policy="none")

    def parse_range(kind: str, first: str, second: str | None = None) -> tuple[int, int | None, str]:
        if kind == "le":
            upper = int(first)
            return 1, upper, f"tok-le{upper}"
        if kind == "ge":
            lower = int(first)
            return lower, None, f"tok-ge{lower}"
        lower = int(first)
        upper = int(second or 0)
        if lower > upper:
            raise ValueError(f"invalid expert recompute policy spec {value!r}; lower bound exceeds upper bound")
        return lower, upper, f"tok{lower}-{upper}"

    match = re.fullmatch(r"tok-le([1-9][0-9]*)", spec)
    kind = "le" if match else ""
    if match is None:
        match = re.fullmatch(r"tok-ge([1-9][0-9]*)", spec)
        kind = "ge" if match else kind
    if match is None:
        match = re.fullmatch(r"tok([1-9][0-9]*)-([1-9][0-9]*)", spec)
        kind = "range" if match else kind
    if match is not None:
        lower, upper, label = parse_range(kind, match.group(1), match.group(2) if kind == "range" else None)
        return result(
            spec=label,
            label=label,
            policy="tok",
            token_threshold=upper or 0,
            token_min=lower,
            token_max=upper,
        )

    match = re.fullmatch(r"tok-le([1-9][0-9]*)-act", spec)
    kind = "le" if match else ""
    if match is None:
        match = re.fullmatch(r"tok-ge([1-9][0-9]*)-act", spec)
        kind = "ge" if match else kind
    if match is None:
        match = re.fullmatch(r"tok([1-9][0-9]*)-([1-9][0-9]*)-act", spec)
        kind = "range" if match else kind
    if match is not None:
        lower, upper, label = parse_range(kind, match.group(1), match.group(2) if kind == "range" else None)
        return result(
            spec=f"{label}-act",
            label=f"{label}-act",
            policy="none",
            activation_policy="tok_act",
            activation_threshold=upper or 0,
            activation_min=lower,
            activation_max=upper,
        )
    raise ValueError(
        f"invalid expert recompute policy spec {value!r}; expected none, tok-le0, tok-le0-act, tok-leN, tok-geN, tokA-B, or -act variants"
    )


def parse_flat_result_dir(path: Path) -> dict[str, Any] | None:
    seq_match = FLAT_SEQ_RE.match(path.name)
    if seq_match is None:
        return None
    job_dir = path.parent
    parts = job_dir.name.split("__")
    if len(parts) != 4:
        return None
    backend, profiler, recompute, policy_part = parts
    if profiler not in PROFILERS or recompute not in {"recomp", "norecomp"}:
        return None
    if policy_part.startswith("pol"):
        try:
            policy_meta = parse_expert_policy_spec(policy_part[3:])
        except ValueError:
            return None
    else:
        return None
    config_name = job_dir.parent.name
    batch_match = re.search(r"(?:^|__)b(?P<batch>[0-9]+)_", config_name)
    return {
        "precision": precision_from_path(path),
        "batch_size": int(batch_match.group("batch")) if batch_match is not None else 0,
        "seq_len": int(seq_match.group("seq_len")),
        "mode": "recompute" if recompute == "recomp" else "no_recompute",
        "activation_recompute": recompute == "recomp",
        "backend": backend,
        "profiler": profiler,
        "workload": config_name.split("__", 1)[0],
        **policy_meta,
    }


def parse_result_dir(path: Path) -> dict[str, Any] | None:
    match = RESULT_RE.match(path.name)
    if match is None:
        return parse_flat_result_dir(path)
    tail = match.group("tail")
    profiler = next((candidate for candidate in PROFILERS if tail.endswith(f"_{candidate}")), "")
    if not profiler:
        return None
    backend = tail[: -(len(profiler) + 1)]
    if backend not in BACKENDS:
        return None
    try:
        policy_meta = parse_expert_policy_spec(
            match.group("expert_policy"),
        )
    except ValueError:
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
        **policy_meta,
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


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def nested_float(mapping: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = mapping.get(section)
    if not isinstance(value, dict):
        return default
    return numeric_float(value.get(key), default)


def nested_int(mapping: dict[str, Any], section: str, key: str, default: int = 0) -> int:
    value = mapping.get(section)
    if not isinstance(value, dict):
        return default
    return numeric_int(value.get(key), default)


def to_mib(value: Any) -> float:
    return numeric_float(value) / MIB


def optional_mib(value: Any) -> float | None:
    result = optional_float(value)
    return None if result is None else result / MIB


def first_optional(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        result = optional_float(mapping.get(key))
        if result is not None:
            return result
    return None


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
    recompute_modes = {
        {"norecomp": "no_recompute", "recomp": "recompute"}.get(mode, mode)
        for mode in args.recompute
    }
    if args.precision and str(meta["precision"]).lower() != str(args.precision).lower():
        return False
    if args.workload and meta["workload"] not in set(args.workload):
        return False
    if args.backend and meta["backend"] not in set(args.backend):
        return False
    if args.profiler and meta["profiler"] not in set(args.profiler):
        return False
    if recompute_modes and meta["mode"] not in recompute_modes:
        return False
    if args.batch_size and int(meta["batch_size"]) != 0 and meta["batch_size"] not in set(args.batch_size):
        return False
    if args.seq_lens and meta["seq_len"] not in set(args.seq_lens):
        return False
    policy_values = split_tokens(list(args.expert_recompute_policy) + list(args.expert_recompute_policies))
    if policy_values:
        parsed_policy_filter = [parse_expert_policy_spec(value) for value in policy_values]
        policy_filter = {
            str(policy["expert_recompute_policy_spec"])
            for policy in parsed_policy_filter
        } | {
            str(policy["expert_policy_label"])
            for policy in parsed_policy_filter
        }
        if str(meta["expert_recompute_policy_spec"]) not in policy_filter and str(meta.get("expert_policy_label", "")) not in policy_filter:
            return False
    return True


def skip_search_path(path: Path, input_root: Path) -> bool:
    try:
        parts = path.relative_to(input_root).parts
    except ValueError:
        parts = path.parts
    return any(part in {"_combined", "combined"} or part.startswith("_") for part in parts)


def result_dirs(input_root: Path) -> list[Path]:
    root_named_dirs = [
        path
        for path in input_root.rglob("*_lora-sft_*")
        if path.is_dir() and not skip_search_path(path, input_root)
    ]
    flat_dirs = [
        path
        for path in input_root.rglob("s*")
        if path.is_dir()
        and FLAT_SEQ_RE.match(path.name)
        and not skip_search_path(path, input_root)
        and profile_json_path(path) is not None
    ]
    return sorted({*root_named_dirs, *flat_dirs})


def row_from_result_dir(args: argparse.Namespace, result_dir: Path) -> dict[str, Any] | None:
    meta = parse_result_dir(result_dir)
    if meta is None or not passes_filters(args, meta):
        return None
    if str(meta["expert_recompute_policy"]) != "none" and bool(meta["activation_recompute"]):
        # Current driver semantics reserve layer recompute for threshold 0.
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
    if args.batch_size and batch_size not in set(args.batch_size):
        return None
    route_stats = first_dict(profile, "expert_token_distribution")
    threshold_effect = route_stats.get("threshold_effect", {})
    if not isinstance(threshold_effect, dict):
        threshold_effect = {}
    expert_recompute_policy_spec = str(
        route_stats.get("expert_recompute_policy_spec", config.get("expert_recompute_policy_spec", meta["expert_recompute_policy_spec"]))
    )
    expert_policy_label = str(
        route_stats.get("expert_policy_label", config.get("expert_policy_label", meta.get("expert_policy_label", expert_recompute_policy_spec)))
    )
    expert_recompute_policy = str(
        route_stats.get("expert_recompute_policy", config.get("expert_recompute_policy", meta["expert_recompute_policy"]))
    )
    expert_recompute_impl = str(
        route_stats.get("expert_recompute_impl", config.get("expert_recompute_impl", meta.get("expert_recompute_impl", "none")))
    )
    expert_recompute_threshold = numeric_int(
        route_stats.get("expert_recompute_threshold", config.get("expert_recompute_threshold", meta["expert_recompute_threshold"]))
    )
    expert_activation_save_policy = str(
        route_stats.get(
            "expert_activation_save_policy",
            config.get("expert_activation_save_policy", meta.get("expert_activation_save_policy", "save_all")),
        )
    )
    expert_activation_save_threshold = numeric_int(
        route_stats.get(
            "expert_activation_save_threshold",
            config.get("expert_activation_save_threshold", meta.get("expert_activation_save_threshold", 0)),
        )
    )
    # Expert-policy sweeps are an AsymGEMM feature.  Torch/KT comparison runs
    # may live under policy-named directories for pairing, but the policy is not
    # applied to those backends.  Canonicalize them to one baseline series so
    # plots do not imply torch changes with AsymGEMM expert recompute policy.
    if str(meta["backend"]) != "asym":
        expert_recompute_policy_spec = "none"
        expert_policy_label = "none"
        expert_recompute_policy = "none"
        expert_recompute_impl = "none"
        expert_recompute_threshold = 0
        expert_activation_save_policy = "save_all"
        expert_activation_save_threshold = 0
    if expert_activation_save_policy == "tok_act":
        expert_recompute_sweep_x = float(expert_activation_save_threshold)
        expert_recompute_sweep_x_name = "activation_token_threshold"
    elif expert_recompute_policy == "tok":
        expert_recompute_sweep_x = float(expert_recompute_threshold)
        expert_recompute_sweep_x_name = "token_threshold"
    else:
        expert_recompute_sweep_x = 0.0
        expert_recompute_sweep_x_name = "none"
    return {
        "workload": meta["workload"],
        "precision": meta["precision"],
        "batch_size": batch_size,
        "seq_len": int(meta["seq_len"]),
        "logical_tokens": int(config.get("logical_tokens", batch_size * int(meta["seq_len"]))),
        "mode": meta["mode"],
        "activation_recompute": bool(meta["activation_recompute"]),
        "expert_recompute_policy_spec": expert_recompute_policy_spec,
        "expert_policy_label": expert_policy_label,
        "expert_recompute_policy": expert_recompute_policy,
        "expert_recompute_impl": expert_recompute_impl,
        "expert_recompute_threshold": expert_recompute_threshold,
        "expert_activation_save_policy": expert_activation_save_policy,
        "expert_activation_save_threshold": expert_activation_save_threshold,
        "expert_recompute_sweep_x": expert_recompute_sweep_x,
        "expert_recompute_sweep_x_name": expert_recompute_sweep_x_name,
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
        "route_samples": numeric_int(route_stats.get("samples")),
        "route_num_tokens": numeric_int(route_stats.get("num_tokens")),
        "route_top_k": numeric_int(route_stats.get("top_k")),
        "route_num_experts": numeric_int(route_stats.get("num_experts")),
        "route_all_expert_tokens_avg": nested_float(route_stats, "all_expert_tokens", "avg"),
        "route_all_expert_tokens_median": nested_float(route_stats, "all_expert_tokens", "median"),
        "route_all_expert_tokens_min": nested_int(route_stats, "all_expert_tokens", "min"),
        "route_all_expert_tokens_max": nested_int(route_stats, "all_expert_tokens", "max"),
        "route_all_expert_tokens_p0": nested_float(route_stats, "all_expert_tokens", "p0"),
        "route_all_expert_tokens_p25": nested_float(route_stats, "all_expert_tokens", "p25"),
        "route_all_expert_tokens_p50": nested_float(route_stats, "all_expert_tokens", "p50"),
        "route_all_expert_tokens_p75": nested_float(route_stats, "all_expert_tokens", "p75"),
        "route_all_expert_tokens_p90": nested_float(route_stats, "all_expert_tokens", "p90"),
        "route_all_expert_tokens_p100": nested_float(route_stats, "all_expert_tokens", "p100"),
        "route_active_expert_tokens_avg": nested_float(route_stats, "active_expert_tokens", "avg"),
        "route_active_expert_tokens_median": nested_float(route_stats, "active_expert_tokens", "median"),
        "route_active_expert_tokens_min": nested_int(route_stats, "active_expert_tokens", "min"),
        "route_active_expert_tokens_max": nested_int(route_stats, "active_expert_tokens", "max"),
        "route_active_expert_tokens_p0": nested_float(route_stats, "active_expert_tokens", "p0"),
        "route_active_expert_tokens_p25": nested_float(route_stats, "active_expert_tokens", "p25"),
        "route_active_expert_tokens_p50": nested_float(route_stats, "active_expert_tokens", "p50"),
        "route_active_expert_tokens_p75": nested_float(route_stats, "active_expert_tokens", "p75"),
        "route_active_expert_tokens_p90": nested_float(route_stats, "active_expert_tokens", "p90"),
        "route_active_expert_tokens_p100": nested_float(route_stats, "active_expert_tokens", "p100"),
        "route_active_experts_avg": nested_float(route_stats, "active_experts", "avg"),
        "route_active_experts_median": nested_float(route_stats, "active_experts", "median"),
        "route_active_experts_min": nested_int(route_stats, "active_experts", "min"),
        "route_active_experts_max": nested_int(route_stats, "active_experts", "max"),
        "route_active_experts_p0": nested_float(route_stats, "active_experts", "p0"),
        "route_active_experts_p25": nested_float(route_stats, "active_experts", "p25"),
        "route_active_experts_p50": nested_float(route_stats, "active_experts", "p50"),
        "route_active_experts_p75": nested_float(route_stats, "active_experts", "p75"),
        "route_active_experts_p90": nested_float(route_stats, "active_experts", "p90"),
        "route_active_experts_p100": nested_float(route_stats, "active_experts", "p100"),
        "route_samples_with_recompute": numeric_int(threshold_effect.get("samples_with_recompute")),
        "route_samples_all_active_recomputed": numeric_int(threshold_effect.get("samples_all_active_recomputed")),
        "route_recomputed_experts_avg": numeric_float(threshold_effect.get("recomputed_experts_avg")),
        "route_recomputed_experts_min": numeric_int(threshold_effect.get("recomputed_experts_min")),
        "route_recomputed_experts_max": numeric_int(threshold_effect.get("recomputed_experts_max")),
        "route_recomputed_experts_p0": numeric_float(threshold_effect.get("recomputed_experts_p0")),
        "route_recomputed_experts_p25": numeric_float(threshold_effect.get("recomputed_experts_p25")),
        "route_recomputed_experts_p50": numeric_float(threshold_effect.get("recomputed_experts_p50")),
        "route_recomputed_experts_p75": numeric_float(threshold_effect.get("recomputed_experts_p75")),
        "route_recomputed_experts_p90": numeric_float(threshold_effect.get("recomputed_experts_p90")),
        "route_recomputed_experts_p100": numeric_float(threshold_effect.get("recomputed_experts_p100")),
        "route_kept_experts_avg": numeric_float(threshold_effect.get("kept_experts_avg")),
        "route_kept_experts_min": numeric_int(threshold_effect.get("kept_experts_min")),
        "route_kept_experts_max": numeric_int(threshold_effect.get("kept_experts_max")),
        "route_kept_experts_p0": numeric_float(threshold_effect.get("kept_experts_p0")),
        "route_kept_experts_p25": numeric_float(threshold_effect.get("kept_experts_p25")),
        "route_kept_experts_p50": numeric_float(threshold_effect.get("kept_experts_p50")),
        "route_kept_experts_p75": numeric_float(threshold_effect.get("kept_experts_p75")),
        "route_kept_experts_p90": numeric_float(threshold_effect.get("kept_experts_p90")),
        "route_kept_experts_p100": numeric_float(threshold_effect.get("kept_experts_p100")),
        "route_recomputed_routes_avg": numeric_float(threshold_effect.get("recomputed_routes_avg")),
        "route_recomputed_routes_min": numeric_int(threshold_effect.get("recomputed_routes_min")),
        "route_recomputed_routes_max": numeric_int(threshold_effect.get("recomputed_routes_max")),
        "route_recomputed_routes_p0": numeric_float(threshold_effect.get("recomputed_routes_p0")),
        "route_recomputed_routes_p25": numeric_float(threshold_effect.get("recomputed_routes_p25")),
        "route_recomputed_routes_p50": numeric_float(threshold_effect.get("recomputed_routes_p50")),
        "route_recomputed_routes_p75": numeric_float(threshold_effect.get("recomputed_routes_p75")),
        "route_recomputed_routes_p90": numeric_float(threshold_effect.get("recomputed_routes_p90")),
        "route_recomputed_routes_p100": numeric_float(threshold_effect.get("recomputed_routes_p100")),
        "route_kept_routes_avg": numeric_float(threshold_effect.get("kept_routes_avg")),
        "route_kept_routes_min": numeric_int(threshold_effect.get("kept_routes_min")),
        "route_kept_routes_max": numeric_int(threshold_effect.get("kept_routes_max")),
        "route_kept_routes_p0": numeric_float(threshold_effect.get("kept_routes_p0")),
        "route_kept_routes_p25": numeric_float(threshold_effect.get("kept_routes_p25")),
        "route_kept_routes_p50": numeric_float(threshold_effect.get("kept_routes_p50")),
        "route_kept_routes_p75": numeric_float(threshold_effect.get("kept_routes_p75")),
        "route_kept_routes_p90": numeric_float(threshold_effect.get("kept_routes_p90")),
        "route_kept_routes_p100": numeric_float(threshold_effect.get("kept_routes_p100")),
        "route_activated_drop_experts_avg": numeric_float(threshold_effect.get("activated_drop_experts_avg")),
        "route_activated_drop_experts_min": numeric_int(threshold_effect.get("activated_drop_experts_min")),
        "route_activated_drop_experts_max": numeric_int(threshold_effect.get("activated_drop_experts_max")),
        "route_activated_drop_routes_avg": numeric_float(threshold_effect.get("activated_drop_routes_avg")),
        "route_activated_drop_routes_min": numeric_int(threshold_effect.get("activated_drop_routes_min")),
        "route_activated_drop_routes_max": numeric_int(threshold_effect.get("activated_drop_routes_max")),
        "route_estimated_activated_saved_mib_avg": to_mib(threshold_effect.get("estimated_activated_saved_bytes_avg")),
        "route_estimated_activated_saved_mib_min": to_mib(threshold_effect.get("estimated_activated_saved_bytes_min")),
        "route_estimated_activated_saved_mib_max": to_mib(threshold_effect.get("estimated_activated_saved_bytes_max")),
        "route_recomputed_paid_rows_avg": numeric_float(threshold_effect.get("recomputed_paid_rows_avg")),
        "route_recomputed_paid_rows_min": numeric_int(threshold_effect.get("recomputed_paid_rows_min")),
        "route_recomputed_paid_rows_max": numeric_int(threshold_effect.get("recomputed_paid_rows_max")),
        "route_kept_paid_rows_avg": numeric_float(threshold_effect.get("kept_paid_rows_avg")),
        "route_kept_paid_rows_min": numeric_int(threshold_effect.get("kept_paid_rows_min")),
        "route_kept_paid_rows_max": numeric_int(threshold_effect.get("kept_paid_rows_max")),
        "route_recomputed_util_avg": numeric_float(threshold_effect.get("recomputed_util_avg")),
        "route_kept_util_avg": numeric_float(threshold_effect.get("kept_util_avg")),
        "output_dir": str(result_dir),
        "profile_json": str(profile_path),
    }


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_root = resolve_path(args.input_root)
    rows = [row for path in result_dirs(input_root) if (row := row_from_result_dir(args, path)) is not None]
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["workload"],
            row["precision"],
            row["batch_size"],
            row["seq_len"],
            row["backend"],
            row["profiler"],
            row["mode"],
            row["expert_recompute_policy"],
            row["expert_recompute_threshold"],
            row["expert_activation_save_policy"],
            row["expert_activation_save_threshold"],
            row["expert_policy_label"],
        )
        deduped.setdefault(key, row)
    rows = list(deduped.values())
    return sorted(
        rows,
        key=lambda row: (
            row["workload"],
            row["batch_size"],
            row["backend"],
            row["profiler"],
            row["seq_len"],
            row["mode"],
            row["expert_recompute_policy"],
            row["expert_recompute_threshold"],
            row["expert_activation_save_policy"],
            row["expert_activation_save_threshold"],
            row["expert_policy_label"],
        ),
    )


def profile_step_samples(profile: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [profile, profile.get("source_profile"), profile.get("memory_profile")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        step_samples = candidate.get("step_samples", {})
        rows = step_samples.get("rows", []) if isinstance(step_samples, dict) else []
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)]
    return []


def is_warmup_sample(sample: dict[str, Any]) -> bool:
    value = sample.get("is_warmup", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def sample_mib(sample: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        result = optional_mib(sample.get(key))
        if result is not None:
            return result
    return None


def add_optional_ms(row: dict[str, Any], output_key: str, sample: dict[str, Any], *keys: str) -> None:
    row[output_key] = first_optional(sample, *keys)


def step_rows_from_result_dir(args: argparse.Namespace, result_dir: Path) -> list[dict[str, Any]]:
    base = row_from_result_dir(args, result_dir)
    if base is None:
        return []
    profile = load_json(Path(str(base["profile_json"])))
    samples = profile_step_samples(profile)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if is_warmup_sample(sample):
            continue
        step = numeric_int(sample.get("measured_step"), numeric_int(sample.get("step")))
        if step <= 0:
            continue
        row = dict(base)
        row["step"] = step
        row["raw_step"] = numeric_int(sample.get("raw_step"), step)
        row["measured_step"] = step
        row["is_warmup"] = False
        row["loss"] = first_optional(sample, "loss")
        add_optional_ms(row, "forward_ms", sample, "forward_milliseconds", "forward_ms")
        add_optional_ms(row, "backward_ms", sample, "backward_milliseconds", "backward_ms")
        forward_ms = optional_float(row.get("forward_ms"))
        backward_ms = optional_float(row.get("backward_ms"))
        step_value = first_optional(sample, "step_milliseconds", "step_ms")
        if step_value is None:
            if forward_ms is not None or backward_ms is not None:
                step_value = (forward_ms or 0.0) + (backward_ms or 0.0)
        if forward_ms is None and backward_ms is None:
            continue
        row["step_ms"] = step_value
        add_optional_ms(row, "step_wall_ms", sample, "wall_milliseconds", "wall_ms")
        row["forward_cuda_kernel_busy_ms"] = first_optional(
            sample,
            "forward_cuda_kernel_busy_milliseconds",
            "forward_cuda_kernel_busy_ms",
        )
        row["backward_cuda_kernel_busy_ms"] = first_optional(
            sample,
            "backward_cuda_kernel_busy_milliseconds",
            "backward_cuda_kernel_busy_ms",
        )
        row["forward_cuda_memcpy_ms"] = first_optional(sample, "forward_cuda_memcpy_milliseconds", "forward_cuda_memcpy_ms")
        row["backward_cuda_memcpy_ms"] = first_optional(sample, "backward_cuda_memcpy_milliseconds", "backward_cuda_memcpy_ms")
        row["forward_cuda_runtime_api_ms"] = first_optional(
            sample,
            "forward_cuda_runtime_api_milliseconds",
            "forward_cuda_runtime_api_ms",
        )
        row["backward_cuda_runtime_api_ms"] = first_optional(
            sample,
            "backward_cuda_runtime_api_milliseconds",
            "backward_cuda_runtime_api_ms",
        )
        row["forward_gpu_no_kernel_ms"] = first_optional(sample, "forward_gpu_no_kernel_milliseconds", "forward_gpu_no_kernel_ms")
        row["backward_gpu_no_kernel_ms"] = first_optional(sample, "backward_gpu_no_kernel_milliseconds", "backward_gpu_no_kernel_ms")
        row["cuda_kernel_busy_ms"] = (
            (row["forward_cuda_kernel_busy_ms"] or 0.0) + (row["backward_cuda_kernel_busy_ms"] or 0.0)
            if row["forward_cuda_kernel_busy_ms"] is not None or row["backward_cuda_kernel_busy_ms"] is not None
            else None
        )
        row["peak_hbm_mib"] = sample_mib(sample, "peak_hbm_bytes", "global_peak_after_bytes")
        row["forward_alloc_start_mib"] = sample_mib(sample, "forward_allocated_start_bytes", "forward_alloc_start_bytes")
        row["forward_alloc_end_mib"] = sample_mib(sample, "forward_allocated_end_bytes", "forward_alloc_end_bytes")
        row["forward_live_delta_mib"] = sample_mib(sample, "forward_allocated_delta_bytes", "forward_live_delta_bytes")
        row["forward_local_peak_mib"] = sample_mib(sample, "forward_local_peak_bytes", "forward_local_peak_hbm_bytes")
        row["forward_local_peak_delta_mib"] = sample_mib(sample, "forward_local_peak_delta_bytes")
        row["backward_alloc_start_mib"] = sample_mib(sample, "backward_allocated_start_bytes", "backward_alloc_start_bytes")
        row["backward_alloc_end_mib"] = sample_mib(sample, "backward_allocated_end_bytes", "backward_alloc_end_bytes")
        row["backward_alloc_delta_mib"] = sample_mib(sample, "backward_allocated_delta_bytes", "backward_alloc_delta_bytes")
        row["backward_local_peak_mib"] = sample_mib(sample, "backward_local_peak_bytes", "backward_local_peak_hbm_bytes")
        row["backward_local_peak_delta_mib"] = sample_mib(sample, "backward_local_peak_delta_bytes")
        rows.append(row)
    return rows


def collect_step_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_root = resolve_path(args.input_root)
    rows: list[dict[str, Any]] = []
    for path in result_dirs(input_root):
        rows.extend(step_rows_from_result_dir(args, path))
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["workload"],
            row["precision"],
            row["batch_size"],
            row["seq_len"],
            row["backend"],
            row["profiler"],
            row["mode"],
            row["expert_recompute_policy"],
            row["expert_recompute_threshold"],
            row["expert_activation_save_policy"],
            row["expert_activation_save_threshold"],
            row["expert_policy_label"],
            row["raw_step"],
        )
        deduped.setdefault(key, row)
    rows = list(deduped.values())
    return sorted(
        rows,
        key=lambda row: (
            row["workload"],
            row["batch_size"],
            row["backend"],
            row["profiler"],
            row["seq_len"],
            row["mode"],
            row["expert_recompute_policy"],
            row["expert_recompute_threshold"],
            row["expert_activation_save_policy"],
            row["expert_activation_save_threshold"],
            row["expert_policy_label"],
            row["step"],
        ),
    )


def group_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (str(row["workload"]), str(row["precision"]), int(row["batch_size"]), str(row["backend"]), str(row["profiler"]))


def threshold_group_key(row: dict[str, Any]) -> tuple[str, str, int, int, str, str]:
    return (
        str(row["workload"]),
        str(row["precision"]),
        int(row["batch_size"]),
        int(row["seq_len"]),
        str(row["backend"]),
        str(row["profiler"]),
    )


def varied_fields(rows: list[dict[str, Any]]) -> set[str]:
    fields = ("workload", "precision", "batch_size", "backend", "profiler", "mode")
    return {field for field in fields if len({row[field] for row in rows}) > 1}


def varied_threshold_fields(rows: list[dict[str, Any]]) -> set[str]:
    fields = ("workload", "precision", "batch_size", "seq_len", "backend", "profiler", "mode")
    return {field for field in fields if len({row[field] for row in rows}) > 1}


def combined_label(group: tuple[str, str, int, str, str], mode: str, varied: set[str]) -> str:
    workload, precision, batch_size, backend, profiler = group
    mode_labels = {"no_recompute": "No recompute", "recompute": "Activation recompute"}
    parts: list[str] = []
    if "workload" in varied:
        parts.append(workload)
    if "batch_size" in varied:
        parts.append(f"b{batch_size}")
    if "precision" in varied:
        parts.append(precision)
    if "backend" in varied:
        parts.append(backend)
    if "profiler" in varied:
        parts.append(profiler)
    if "mode" in varied:
        parts.append(mode_labels.get(mode, mode))
    if parts:
        return " / ".join(parts)
    return f"{backend} / {mode_labels.get(mode, mode)}"


def combined_threshold_label(group: tuple[str, str, int, int, str, str], mode: str, varied: set[str]) -> str:
    workload, precision, batch_size, seq_len, backend, profiler = group
    mode_labels = {"no_recompute": "No layer recompute", "recompute": "Layer recompute"}
    parts: list[str] = []
    if "workload" in varied:
        parts.append(workload)
    if "batch_size" in varied:
        parts.append(f"b{batch_size}")
    if "seq_len" in varied:
        parts.append(f"s{seq_len}")
    if "precision" in varied:
        parts.append(precision)
    if "backend" in varied:
        parts.append(backend)
    if "profiler" in varied:
        parts.append(profiler)
    if "mode" in varied:
        parts.append(mode_labels.get(mode, mode))
    if parts:
        return " / ".join(parts)
    return f"s{seq_len} / {backend} / {mode_labels.get(mode, mode)}"


def write_table(rows: list[dict[str, Any]], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    # Keep the root output directory limited to the sweep index.
    for name in ROOT_OUTPUT_FILES + COMBINED_OUTPUT_FILES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in {"_combined", "combined"}:
            for name in COMBINED_OUTPUT_FILES:
                path = child / name
                if path.is_file():
                    path.unlink()
        for name in GROUP_OUTPUT_FILES:
            path = child / name
            if path.is_file():
                path.unlink()
        for path in child.glob("expert_threshold_summary_s*.csv"):
            path.unlink()
        for path in child.glob("expert_threshold_summary_s*.json"):
            path.unlink()
        for path in child.glob("*_vs_expert_threshold_s*.png"):
            path.unlink()
        for path in child.glob("expert_*_threshold_summary_s*.csv"):
            path.unlink()
        for path in child.glob("expert_*_threshold_summary_s*.json"):
            path.unlink()
        for path in child.glob("*_vs_expert_*_threshold_s*.png"):
            path.unlink()
        try:
            child.rmdir()
        except OSError:
            pass


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) <= 1e-12:
        return 0.0 if all(abs(value) <= 1e-12 for value in values) else float("inf")
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    count = len(points)
    if count < 2:
        return 0.0, points[0][1] if points else 0.0, 1.0
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    ss_x = sum((x - mean_x) ** 2 for x, _ in points)
    if ss_x <= 0.0:
        return 0.0, mean_y, 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / ss_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    if ss_tot <= 0.0:
        return slope, intercept, 1.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    return slope, intercept, 1.0 - ss_res / ss_tot


def offset_normalized_ratio_cv(points: list[tuple[float, float]], intercept: float) -> float:
    ratios: list[float] = []
    for x, y in points:
        shifted_y = y - intercept
        if x <= 0.0 or shifted_y <= 0.0:
            return float("inf")
        ratios.append(shifted_y / x)
    return coefficient_of_variation(ratios)


def series_points(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in sorted(rows, key=lambda item: item["seq_len"]):
        x = float(row["seq_len"])
        y = float(row[key]) / scale
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    return points


def linear_region(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> tuple[float, float] | None:
    points = series_points(rows, key, scale=scale)
    if not points:
        return None
    if len(points) == 1:
        return points[0][0], points[0][0]

    min_points = min(MIN_LINEAR_REGION_POINTS, len(points))
    best_region: tuple[float, float] | None = None
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    fallback_region = (points[0][0], points[-1][0])
    fallback_key = (float("inf"), float("inf"), float("-inf"))

    for start in range(0, len(points) - min_points + 1):
        for stop in range(start + min_points - 1, len(points)):
            window = points[start : stop + 1]
            slope, intercept, r2 = linear_fit(window)
            ratio_cv = offset_normalized_ratio_cv(window, intercept)
            span = window[-1][0] - window[0][0]
            score = (ratio_cv, max(0.0, LINEAR_REGION_R2_THRESHOLD - r2), -span)
            if score < fallback_key:
                fallback_region = (window[0][0], window[-1][0])
                fallback_key = score
            if slope <= 0.0:
                continue
            if r2 < LINEAR_REGION_R2_THRESHOLD or ratio_cv > LINEAR_REGION_RATIO_CV_THRESHOLD:
                continue
            valid_key = (span, float(len(window)), -ratio_cv, r2)
            if valid_key > best_key:
                best_region = (window[0][0], window[-1][0])
                best_key = valid_key
    return best_region if best_region is not None else fallback_region


def reference_line(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> tuple[float, float] | None:
    points = series_points(rows, key, scale=scale)
    region = linear_region(rows, key, scale=scale)
    if region is None:
        return None
    left, right = region
    window = [(x, y) for x, y in points if left <= x <= right]
    slope, intercept, _ = linear_fit(window)
    if slope <= 0.0:
        return None
    return slope, intercept


def sublinear_regions(rows: list[dict[str, Any]], key: str, *, scale: float = 1.0) -> list[tuple[float, float]]:
    points = series_points(rows, key, scale=scale)
    reference = reference_line(rows, key, scale=scale)
    if len(points) < 2 or reference is None:
        return []
    slope, _intercept = reference
    regions: list[tuple[float, float]] = []
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= 0.0 or right_x <= left_x:
            continue
        # Treat each adjacent pair as its own region. Translating the left point
        # to (0, 0) leaves dx and dy, so sublinearity is local slope vs. the
        # fitted linear reference slope.
        local_slope = (right_y - left_y) / (right_x - left_x)
        if 0.0 < local_slope < slope * (1.0 - SUBLINEAR_SLOPE_TOLERANCE):
            regions.append((left_x, right_x))
    return regions


def merge_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for left, right in sorted(regions):
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def draw_sublinear_regions(ax: Any, regions: list[tuple[float, float]]) -> bool:
    unique_regions = sorted({(left, right) for left, right in regions if right > left})
    merged_regions = merge_regions(unique_regions)
    for left, right in merged_regions:
        ax.axvspan(left, right, color=SUBLINEAR_COLOR, alpha=SUBLINEAR_ALPHA, zorder=0.1)
        ax.axvline(left, color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4, alpha=0.8, zorder=0.8)
        ax.axvline(right, color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4, alpha=0.8, zorder=0.8)
    return bool(merged_regions)


def add_legend(ax: Any, *, sublinear_region: bool, fontsize: int | None = None) -> None:
    if not sublinear_region:
        ax.legend(fontsize=fontsize)
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=SUBLINEAR_COLOR, linestyle=":", linewidth=1.4))
    labels.append("Sublinear boundary")
    handles.append(Patch(facecolor=SUBLINEAR_COLOR, alpha=SUBLINEAR_ALPHA, edgecolor="none"))
    labels.append("Sublinear region")
    ax.legend(handles, labels, fontsize=fontsize)


def series_color_map(labels: list[str]) -> dict[str, str]:
    return {label: STYLE_PALETTE[index % len(STYLE_PALETTE)] for index, label in enumerate(labels)}


def marker_for_backend(backend: Any) -> str:
    return BACKEND_MARKERS.get(str(backend), "D")


def plot_line(
    ax: Any,
    x_values: list[Any],
    y_values: list[float],
    *,
    label: str,
    backend: Any,
    linewidth: float,
    color: str,
) -> None:
    ax.plot(
        x_values,
        y_values,
        marker=marker_for_backend(backend),
        color=color,
        linewidth=linewidth,
        markersize=6,
        markeredgewidth=0.9,
        markeredgecolor="#222222",
        label=label,
    )


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
    color_by_label = series_color_map([labels[mode] for mode, mode_rows in by_mode.items() if mode_rows])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    sublinear_spans: list[tuple[float, float]] = []
    for mode, mode_rows in by_mode.items():
        if not mode_rows:
            continue
        label = labels[mode]
        plot_line(
            ax,
            [row["seq_len"] for row in mode_rows],
            [float(row[key]) / scale for row in mode_rows],
            label=label,
            backend=mode_rows[0].get("backend", ""),
            linewidth=2,
            color=color_by_label[label],
        )
        sublinear_spans.extend(sublinear_regions(mode_rows, key, scale=scale))
    has_sublinear_region = draw_sublinear_regions(ax, sublinear_spans)
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    add_legend(ax, sublinear_region=has_sublinear_region)
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

    series: dict[tuple[tuple[str, str, int, str, str], str], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((group_key(row), str(row["mode"])), []).append(row)
    varied = varied_fields(rows)
    series_items: list[tuple[str, list[dict[str, Any]]]] = []
    for (group, mode), group_rows in sorted(series.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row["seq_len"])
        series_items.append((combined_label(group, mode, varied), sorted_rows))
    color_by_label = series_color_map([label for label, _ in series_items])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    sublinear_spans: list[tuple[float, float]] = []
    for label, sorted_rows in series_items:
        plot_line(
            ax,
            [row["seq_len"] for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            label=label,
            backend=sorted_rows[0].get("backend", ""),
            linewidth=1.8,
            color=color_by_label[label],
        )
        sublinear_spans.extend(sublinear_regions(sorted_rows, key, scale=scale))
    has_sublinear_region = draw_sublinear_regions(ax, sublinear_spans)
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    add_legend(ax, sublinear_region=has_sublinear_region, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def step_series_label(row: dict[str, Any]) -> str:
    mode_labels = {"no_recompute": "No recompute", "recompute": "Activation recompute"}
    parts = [f"s{row['seq_len']}", mode_labels.get(str(row["mode"]), str(row["mode"]))]
    policy = str(row.get("expert_policy_label", "none"))
    if policy != "none":
        parts.append(policy)
    return " / ".join(parts)


def combined_step_series_label(row: dict[str, Any], varied: set[str]) -> str:
    label = combined_label(group_key(row), str(row["mode"]), varied)
    parts = [label, f"s{row['seq_len']}"]
    policy = str(row.get("expert_policy_label", "none"))
    if policy != "none":
        parts.append(policy)
    return " / ".join(parts)


def step_plot_specs(*, combined: bool) -> list[tuple[str, str, str, str, float]]:
    prefix = "combined_" if combined else ""
    title_prefix = "All workloads: " if combined else ""
    return [
        (
            f"{prefix}forward_end_memory_vs_step.png",
            f"{title_prefix}memory after forward over steps",
            "GPU allocated after forward (GiB)",
            "forward_alloc_end_mib",
            1024.0,
        ),
        (
            f"{prefix}forward_peak_memory_vs_step.png",
            f"{title_prefix}forward local peak over steps",
            "Forward local peak allocation (GiB)",
            "forward_local_peak_mib",
            1024.0,
        ),
        (
            f"{prefix}backward_start_memory_vs_step.png",
            f"{title_prefix}memory carried into backward over steps",
            "GPU allocated at backward start (GiB)",
            "backward_alloc_start_mib",
            1024.0,
        ),
        (
            f"{prefix}backward_peak_memory_vs_step.png",
            f"{title_prefix}backward local peak over steps",
            "Backward local peak allocation (GiB)",
            "backward_local_peak_mib",
            1024.0,
        ),
        (
            f"{prefix}peak_hbm_vs_step.png",
            f"{title_prefix}whole-step peak HBM over steps",
            "Peak HBM allocation (GiB)",
            "peak_hbm_mib",
            1024.0,
        ),
        (
            f"{prefix}timing_vs_step.png",
            f"{title_prefix}step time over steps",
            "Step time (ms)",
            "step_ms",
            1.0,
        ),
        (
            f"{prefix}forward_timing_vs_step.png",
            f"{title_prefix}forward time over steps",
            "Forward time (ms)",
            "forward_ms",
            1.0,
        ),
        (
            f"{prefix}backward_timing_vs_step.png",
            f"{title_prefix}backward time over steps",
            "Backward time (ms)",
            "backward_ms",
            1.0,
        ),
        (
            f"{prefix}loss_vs_step.png",
            f"{title_prefix}loss over steps",
            "Loss",
            "loss",
            1.0,
        ),
    ]


def plot_step_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    scale: float = 1.0,
    combined: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        value = optional_float(row.get(key))
        if value is None:
            continue
        if combined:
            series_key = (
                group_key(row),
                int(row["seq_len"]),
                str(row["mode"]),
                str(row.get("expert_policy_label", "none")),
            )
        else:
            series_key = (
                int(row["seq_len"]),
                str(row["mode"]),
                str(row.get("expert_policy_label", "none")),
            )
        series.setdefault(series_key, []).append(row)
    if not series:
        return

    varied = varied_fields(rows)
    series_items: list[tuple[str, list[dict[str, Any]]]] = []
    for _, series_rows in sorted(series.items()):
        sorted_rows = sorted(series_rows, key=lambda row: numeric_int(row.get("raw_step"), int(row["step"])))
        label = combined_step_series_label(sorted_rows[0], varied) if combined else step_series_label(sorted_rows[0])
        series_items.append((label, sorted_rows))
    color_by_label = series_color_map([label for label, _ in series_items])

    fig, ax = plt.subplots(figsize=(11 if combined else 8, 5.5), dpi=160)
    for label, sorted_rows in series_items:
        plot_line(
            ax,
            [numeric_int(row.get("raw_step"), int(row["step"])) for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            label=label,
            backend=sorted_rows[0].get("backend", ""),
            linewidth=1.8,
            color=color_by_label[label],
        )
    ax.set_title(title)
    ax.set_xlabel("Raw trainer step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7 if combined else None)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def threshold_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[tuple[tuple[str, str, int, int, str, str], str], set[int]] = {}
    token_rows = [
        row
        for row in rows
        if str(row.get("expert_recompute_policy", "none")) in {"none", "tok"}
        and str(row.get("expert_activation_save_policy", "save_all")) == "save_all"
    ]
    for row in token_rows:
        by_series.setdefault((threshold_group_key(row), str(row["mode"])), set()).add(int(row["expert_recompute_threshold"]))
    eligible = {key for key, thresholds in by_series.items() if len(thresholds) >= 2}
    return [row for row in token_rows if (threshold_group_key(row), str(row["mode"])) in eligible]


def policy_family_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    if family == "tok":
        return [
            row
            for row in rows
            if str(row.get("expert_recompute_policy", "none")) in {"none", "tok"}
            and str(row.get("expert_activation_save_policy", "save_all")) == "save_all"
        ]
    if family == "tok_act":
        return [
            row
            for row in rows
            if (
                str(row.get("expert_policy_label", "none")) == "none"
                and str(row.get("expert_activation_save_policy", "save_all")) == "save_all"
            )
            or str(row.get("expert_activation_save_policy", "save_all")) == "tok_act"
        ]
    return []


def policy_sweep_x(row: dict[str, Any], family: str) -> float:
    if family == "tok_act":
        if str(row.get("expert_activation_save_policy", "save_all")) == "tok_act":
            return float(row["expert_activation_save_threshold"])
        return 0.0
    if str(row.get("expert_recompute_policy", "none")) == "none":
        return 0.0
    if family == "tok":
        return float(row["expert_recompute_threshold"])
    return 0.0


def policy_series_suffix(row: dict[str, Any], family: str) -> int:
    return 0


def policy_sweep_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    family_rows = policy_family_rows(rows, family)
    by_series: dict[tuple[tuple[str, str, int, int, str, str], str, int], set[float]] = {}
    for row in family_rows:
        key = (threshold_group_key(row), str(row["mode"]), policy_series_suffix(row, family))
        by_series.setdefault(key, set()).add(policy_sweep_x(row, family))
    eligible = {key for key, xs in by_series.items() if len(xs) >= 2}
    return [
        row
        for row in family_rows
        if (threshold_group_key(row), str(row["mode"]), policy_series_suffix(row, family)) in eligible
    ]


def policy_x_label(family: str) -> str:
    if family == "tok":
        return "Expert recompute threshold (tokens)"
    if family == "tok_act":
        return "Expert activated-drop threshold (tokens)"
    return "Expert recompute threshold (tokens)"


def policy_filename_suffix(family: str) -> str:
    return {
        "tok": "expert_tok_threshold",
        "tok_act": "expert_tok_act_threshold",
    }[family]


def plot_threshold_metric(
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
        "no_recompute": sorted((row for row in rows if row["mode"] == "no_recompute"), key=lambda row: row["expert_recompute_threshold"]),
        "recompute": sorted((row for row in rows if row["mode"] == "recompute"), key=lambda row: row["expert_recompute_threshold"]),
    }
    labels = {"no_recompute": "No layer recompute", "recompute": "Layer recompute"}
    color_by_label = series_color_map(
        [
            labels[mode]
            for mode, mode_rows in by_mode.items()
            if len({int(row["expert_recompute_threshold"]) for row in mode_rows}) >= 2
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    plotted = False
    for mode, mode_rows in by_mode.items():
        if len({int(row["expert_recompute_threshold"]) for row in mode_rows}) < 2:
            continue
        label = labels[mode]
        plot_line(
            ax,
            [row["expert_recompute_threshold"] for row in mode_rows],
            [float(row[key]) / scale for row in mode_rows],
            label=label,
            backend=mode_rows[0].get("backend", ""),
            linewidth=2,
            color=color_by_label[label],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Expert recompute threshold (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_combined_threshold_metric(
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

    series: dict[tuple[tuple[str, str, int, int, str, str], str], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((threshold_group_key(row), str(row["mode"])), []).append(row)
    varied = varied_threshold_fields(rows)
    series_items: list[tuple[str, list[dict[str, Any]]]] = []
    for (group, mode), group_rows in sorted(series.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row["expert_recompute_threshold"])
        if len({int(row["expert_recompute_threshold"]) for row in sorted_rows}) >= 2:
            series_items.append((combined_threshold_label(group, mode, varied), sorted_rows))
    color_by_label = series_color_map([label for label, _ in series_items])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    plotted = False
    for label, sorted_rows in series_items:
        plot_line(
            ax,
            [row["expert_recompute_threshold"] for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            label=label,
            backend=sorted_rows[0].get("backend", ""),
            linewidth=1.8,
            color=color_by_label[label],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Expert recompute threshold (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_policy_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    family: str,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((str(row["mode"]), policy_series_suffix(row, family)), []).append(row)
    labels = {"no_recompute": "No layer recompute", "recompute": "Layer recompute"}
    series_items: list[tuple[str, list[dict[str, Any]]]] = []
    for (mode, token_cap), series_rows in sorted(series.items()):
        sorted_rows = sorted(series_rows, key=lambda row: policy_sweep_x(row, family))
        if len({policy_sweep_x(row, family) for row in sorted_rows}) >= 2:
            series_items.append((labels.get(mode, mode), sorted_rows))
    color_by_label = series_color_map([label for label, _ in series_items])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    plotted = False
    for label, sorted_rows in series_items:
        plot_line(
            ax,
            [policy_sweep_x(row, family) for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            label=label,
            backend=sorted_rows[0].get("backend", ""),
            linewidth=2,
            color=color_by_label[label],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel(policy_x_label(family))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_combined_policy_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    key: str,
    *,
    family: str,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt

    series: dict[tuple[tuple[str, str, int, int, str, str], str, int], list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault((threshold_group_key(row), str(row["mode"]), policy_series_suffix(row, family)), []).append(row)
    varied = varied_threshold_fields(rows)
    series_items: list[tuple[str, list[dict[str, Any]]]] = []
    for (group, mode, token_cap), group_rows in sorted(series.items()):
        sorted_rows = sorted(group_rows, key=lambda row: policy_sweep_x(row, family))
        if len({policy_sweep_x(row, family) for row in sorted_rows}) >= 2:
            series_items.append((combined_threshold_label(group, mode, varied), sorted_rows))
    color_by_label = series_color_map([label for label, _ in series_items])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    plotted = False
    for label, sorted_rows in series_items:
        plot_line(
            ax,
            [policy_sweep_x(row, family) for row in sorted_rows],
            [float(row[key]) / scale for row in sorted_rows],
            label=label,
            backend=sorted_rows[0].get("backend", ""),
            linewidth=1.8,
            color=color_by_label[label],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel(policy_x_label(family))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
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
        "forward_end_memory_vs_seq.png",
        f"{title_base} memory after forward{suffix}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_metric(
        rows,
        output_dir,
        "forward_peak_memory_vs_seq.png",
        f"{title_base} forward local peak{suffix}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
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


def write_group_step_plots(rows: list[dict[str, Any]], output_dir: Path, key: tuple[str, str, int, str, str]) -> None:
    if not rows:
        return
    workload, precision, batch_size, backend, profiler = key
    title_base = f"{workload} LoRA SFT"
    suffix = f", batch size {batch_size}, {precision}, {backend}/{profiler}"
    write_table(rows, output_dir, "step_samples")
    for filename, title, ylabel, metric_key, scale in step_plot_specs(combined=False):
        plot_step_metric(
            rows,
            output_dir,
            filename,
            f"{title_base} {title}{suffix}",
            ylabel,
            metric_key,
            scale=scale,
            combined=False,
        )


def write_group_threshold_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    key: tuple[str, str, int, str, str],
    seq_len: int,
) -> None:
    workload, precision, batch_size, backend, profiler = key
    title_base = f"{workload} LoRA SFT"
    suffix = f", batch size {batch_size}, seq {seq_len}, {precision}, {backend}/{profiler}"
    plot_threshold_metric(
        rows,
        output_dir,
        f"forward_end_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} memory after forward{suffix}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"forward_peak_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} forward local peak{suffix}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"backward_start_memory_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} memory carried into backward{suffix}",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"peak_hbm_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} whole-step peak HBM{suffix}",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_threshold_metric(
        rows,
        output_dir,
        f"timing_vs_expert_threshold_s{seq_len}.png",
        f"{title_base} step time vs expert threshold{suffix}",
        "Step time (ms)",
        "step_ms",
    )


def write_group_policy_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    key: tuple[str, str, int, str, str],
    seq_len: int,
    family: str,
) -> None:
    workload, precision, batch_size, backend, profiler = key
    title_base = f"{workload} LoRA SFT"
    suffix = f", batch size {batch_size}, seq {seq_len}, {precision}, {backend}/{profiler}"
    name = policy_filename_suffix(family)
    title = {
        "tok": "expert token threshold",
        "tok_act": "expert activated-drop threshold",
    }[family]
    plot_policy_metric(
        rows,
        output_dir,
        f"forward_end_memory_vs_{name}_s{seq_len}.png",
        f"{title_base} memory after forward vs {title}{suffix}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        family=family,
        scale=1024.0,
    )
    plot_policy_metric(
        rows,
        output_dir,
        f"forward_peak_memory_vs_{name}_s{seq_len}.png",
        f"{title_base} forward local peak vs {title}{suffix}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        family=family,
        scale=1024.0,
    )
    plot_policy_metric(
        rows,
        output_dir,
        f"backward_start_memory_vs_{name}_s{seq_len}.png",
        f"{title_base} memory carried into backward vs {title}{suffix}",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        family=family,
        scale=1024.0,
    )
    plot_policy_metric(
        rows,
        output_dir,
        f"backward_peak_memory_vs_{name}_s{seq_len}.png",
        f"{title_base} backward local peak vs {title}{suffix}",
        "Backward local peak allocation (GiB)",
        "backward_local_peak_mib",
        family=family,
        scale=1024.0,
    )
    plot_policy_metric(
        rows,
        output_dir,
        f"peak_hbm_vs_{name}_s{seq_len}.png",
        f"{title_base} whole-step peak HBM vs {title}{suffix}",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        family=family,
        scale=1024.0,
    )
    plot_policy_metric(
        rows,
        output_dir,
        f"timing_vs_{name}_s{seq_len}.png",
        f"{title_base} step time vs {title}{suffix}",
        "Step time (ms)",
        "step_ms",
        family=family,
    )


def write_combined_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_metric(
        rows,
        output_dir,
        "combined_forward_end_memory_vs_seq.png",
        "All workloads: memory after forward",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_combined_metric(
        rows,
        output_dir,
        "combined_forward_peak_memory_vs_seq.png",
        "All workloads: forward local peak",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
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


def write_combined_step_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, output_dir, "combined_step_samples_index")
    for filename, title, ylabel, metric_key, scale in step_plot_specs(combined=True):
        plot_step_metric(
            rows,
            output_dir,
            filename,
            title,
            ylabel,
            metric_key,
            scale=scale,
            combined=True,
        )


def write_combined_threshold_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_forward_end_memory_vs_expert_threshold.png",
        "All workloads: memory after forward vs expert threshold",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_forward_peak_memory_vs_expert_threshold.png",
        "All workloads: forward local peak vs expert threshold",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_backward_start_memory_vs_expert_threshold.png",
        "All workloads: memory carried into backward vs expert threshold",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_peak_hbm_vs_expert_threshold.png",
        "All workloads: whole-step peak HBM vs expert threshold",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        scale=1024.0,
    )
    plot_combined_threshold_metric(
        rows,
        output_dir,
        "combined_timing_vs_expert_threshold.png",
        "All workloads: step time vs expert threshold",
        "Step time (ms)",
        "step_ms",
    )


def write_combined_policy_plots(rows: list[dict[str, Any]], output_dir: Path, family: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = policy_filename_suffix(family)
    title = {
        "tok": "expert token threshold",
        "tok_act": "expert activated-drop threshold",
    }[family]
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_forward_end_memory_vs_{name}.png",
        f"All workloads: memory after forward vs {title}",
        "GPU allocated after forward (GiB)",
        "forward_alloc_end_mib",
        family=family,
        scale=1024.0,
    )
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_forward_peak_memory_vs_{name}.png",
        f"All workloads: forward local peak vs {title}",
        "Forward local peak allocation (GiB)",
        "forward_local_peak_mib",
        family=family,
        scale=1024.0,
    )
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_backward_start_memory_vs_{name}.png",
        f"All workloads: memory carried into backward vs {title}",
        "GPU allocated at backward start (GiB)",
        "backward_alloc_start_mib",
        family=family,
        scale=1024.0,
    )
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_backward_peak_memory_vs_{name}.png",
        f"All workloads: backward local peak vs {title}",
        "Backward local peak allocation (GiB)",
        "backward_local_peak_mib",
        family=family,
        scale=1024.0,
    )
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_peak_hbm_vs_{name}.png",
        f"All workloads: whole-step peak HBM vs {title}",
        "Peak HBM allocation (GiB)",
        "peak_hbm_mib",
        family=family,
        scale=1024.0,
    )
    plot_combined_policy_metric(
        rows,
        output_dir,
        f"combined_timing_vs_{name}.png",
        f"All workloads: step time vs {title}",
        "Step time (ms)",
        "step_ms",
        family=family,
    )


def main() -> None:
    args = parse_args()
    if args.skip_combined and args.combined_only:
        raise SystemExit("--skip-combined and --combined-only cannot be used together")
    rows = collect_rows(args)
    if not rows:
        raise SystemExit(f"no driver result directories found under {resolve_path(args.input_root)}")
    step_rows = collect_step_rows(args)

    root = output_root(args)
    combined_dir = combined_output_root(args, root)
    if args.clean_output:
        if not args.combined_only:
            clean_output_dir(root)
        if not args.skip_combined:
            clean_output_dir(combined_dir)
    write_table(rows, root, "activation_recompute_sweep_index")
    if step_rows:
        write_table(step_rows, root, "step_samples_index")

    seq_rows = [
        row
        for row in rows
        if str(row.get("expert_recompute_policy", "none")) == "none"
        and str(row.get("expert_activation_save_policy", "save_all")) == "save_all"
    ]
    seq_step_rows = [
        row
        for row in step_rows
        if str(row.get("expert_recompute_policy", "none")) == "none"
        and str(row.get("expert_activation_save_policy", "save_all")) == "save_all"
    ]
    if seq_rows and not args.skip_combined:
        write_combined_plots(seq_rows, combined_dir)
    if step_rows and not args.skip_combined:
        write_combined_step_plots(step_rows, combined_dir)

    if not args.combined_only:
        groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
        for row in seq_rows:
            groups.setdefault(group_key(row), []).append(row)
        step_groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
        for row in step_rows:
            step_groups.setdefault(group_key(row), []).append(row)
        for key in sorted(set(groups) | set(step_groups)):
            group_rows = groups.get(key, [])
            group_step_rows = step_groups.get(key, [])
            workload, precision, batch_size, backend, profiler = key
            group_dir = root / safe_label(f"{workload}-b{batch_size}-{precision}-{backend}-{profiler}")
            if group_rows:
                write_table(group_rows, group_dir, "sweep_summary")
                write_group_plots(group_rows, group_dir, key)
            if group_step_rows:
                write_group_step_plots(group_step_rows, group_dir, key)
            print(f"wrote {group_dir}", flush=True)

    for family in ("tok", "tok_act"):
        family_rows = policy_sweep_rows(rows, family)
        if not family_rows:
            continue
        if not args.skip_combined:
            write_combined_policy_plots(family_rows, combined_dir, family)
        if args.combined_only:
            continue
        family_groups: dict[tuple[tuple[str, str, int, str, str], int], list[dict[str, Any]]] = {}
        for row in family_rows:
            family_groups.setdefault((group_key(row), int(row["seq_len"])), []).append(row)
        suffix = policy_filename_suffix(family)
        for (key, seq_len), group_rows in sorted(family_groups.items()):
            workload, precision, batch_size, backend, profiler = key
            group_dir = root / safe_label(f"{workload}-b{batch_size}-{precision}-{backend}-{profiler}")
            write_table(group_rows, group_dir, f"{suffix}_summary_s{seq_len}")
            write_group_policy_plots(group_rows, group_dir, key, seq_len, family)
            print(f"wrote {group_dir} {suffix} s{seq_len}", flush=True)
    print(f"wrote {root / 'activation_recompute_sweep_index.json'}", flush=True)


if __name__ == "__main__":
    main()
