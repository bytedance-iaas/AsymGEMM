#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


LEGACY_KEYS = {
    "activation_bytes",
    "framework_temp_workspace",
    "peak_hbm_bytes",
    "stage_local_peak_hbm_bytes",
    "global_peak_after_bytes",
    "avg_local_peak_bytes",
    "avg_local_peak_delta_bytes",
    "max_local_peak_bytes",
    "max_global_peak_after_bytes",
    "local_peak_bytes",
    "local_peak_delta_bytes",
}
GPU_FIELDS = {
    "peak_allocated_hbm_bytes",
    "peak_reserved_hbm_bytes",
    "reserved_unallocated_bytes",
}
STAGE_FIELDS = {
    "avg_peak_allocated_bytes",
    "max_peak_allocated_bytes",
    "avg_peak_allocated_delta_bytes",
    "avg_peak_reserved_bytes",
    "max_peak_reserved_bytes",
    "avg_peak_reserved_delta_bytes",
    "max_global_peak_allocated_after_bytes",
    "max_global_peak_reserved_after_bytes",
    "avg_reserved_unallocated_bytes",
}
WHOLE_STEP_FIELDS = {
    "peak_allocated_hbm_bytes",
    "peak_reserved_hbm_bytes",
    "reserved_unallocated_bytes",
}
PER_STAGE_STEP_SUFFIXES = {
    "peak_allocated_bytes",
    "peak_allocated_delta_bytes",
    "peak_reserved_bytes",
    "peak_reserved_delta_bytes",
    "global_peak_allocated_after_bytes",
    "global_peak_reserved_after_bytes",
    "reserved_unallocated_bytes",
}
BREAKDOWN_FIELDS = {
    "schema_version",
    "selected_metric",
    "peak_allocated_hbm_bytes",
    "peak_reserved_hbm_bytes",
    "reserved_unallocated_bytes",
    "allocated_stack_sum_bytes",
    "reserved_stack_sum_bytes",
    "saved_activation_hbm_bytes_at_peak",
    "unattributed_allocated_peak_bytes",
    "allocated_bytes",
    "reserved_bytes",
    "allocated_closure_ok",
    "reserved_closure_ok",
    "allocated_closure_error_bytes",
    "reserved_closure_error_bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LF allocated/reserved memory-capacity schema artifacts.")
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--source-profile-json", type=Path)
    parser.add_argument("--memory-breakdown-summary", type=Path)
    parser.add_argument("--require-nsys-step-bytes", action="store_true")
    parser.add_argument("--require-breakdown", action="store_true")
    args = parser.parse_args()
    if not any((args.profile_json, args.source_profile_json, args.memory_breakdown_summary)):
        parser.error("provide at least one artifact path")
    return args


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def profile_views(profile: dict[str, Any]) -> list[dict[str, Any]]:
    views = [profile]
    nested = profile.get("source_profile")
    if isinstance(nested, dict):
        views.append(nested)
    return views


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _missing(mapping: dict[str, Any], fields: set[str]) -> list[str]:
    return sorted(field for field in fields if field not in mapping)


def _legacy_key_errors(value: Any, path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in LEGACY_KEYS:
                errors.append(f"{child_path}: legacy key is not allowed")
            errors.extend(_legacy_key_errors(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_legacy_key_errors(child, f"{path}[{index}]"))
    return errors


def _validate_gap(mapping: dict[str, Any], path: str, allocated_key: str, reserved_key: str, gap_key: str) -> list[str]:
    errors: list[str] = []
    missing = _missing(mapping, {allocated_key, reserved_key, gap_key})
    if missing:
        return [f"{path}: missing {', '.join(missing)}"]
    allocated = mapping.get(allocated_key)
    reserved = mapping.get(reserved_key)
    gap = mapping.get(gap_key)
    if not all(_is_number(value) for value in (allocated, reserved, gap)):
        return [f"{path}: allocated/reserved/gap fields must be numeric"]
    expected_gap = max(0, int(reserved) - int(allocated))
    if int(reserved) < int(allocated):
        errors.append(f"{path}: reserved bytes {reserved} is less than allocated bytes {allocated}")
    if int(gap) != expected_gap:
        errors.append(f"{path}: {gap_key}={gap} does not equal max(0, {reserved_key} - {allocated_key})={expected_gap}")
    return errors


def _memory_gpu_views(profile: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    views: list[tuple[str, dict[str, Any]]] = []
    for index, view in enumerate(profile_views(profile)):
        memory = view.get("memory")
        gpu = memory.get("gpu") if isinstance(memory, dict) else None
        if isinstance(gpu, dict):
            views.append((f"profile_views[{index}].memory.gpu", gpu))
    return views


def validate_memory_gpu(profile: dict[str, Any]) -> list[str]:
    errors = _legacy_key_errors(profile, "")
    views = _memory_gpu_views(profile)
    if not views:
        errors.append("memory.gpu: missing memory GPU view with allocated/reserved fields")
        return errors
    for path, gpu in views:
        missing = _missing(gpu, GPU_FIELDS)
        if missing:
            errors.append(f"{path}: missing {', '.join(missing)}")
            continue
        errors.extend(
            _validate_gap(gpu, path, "peak_allocated_hbm_bytes", "peak_reserved_hbm_bytes", "reserved_unallocated_bytes")
        )
    return errors


def _stage_rows_views(profile: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    views: list[tuple[str, list[dict[str, Any]]]] = []
    for index, view in enumerate(profile_views(profile)):
        stage_memory = view.get("stage_memory")
        rows = stage_memory.get("rows") if isinstance(stage_memory, dict) else None
        if isinstance(rows, list):
            views.append((f"profile_views[{index}].stage_memory.rows", [row for row in rows if isinstance(row, dict)]))
    return views


def validate_stage_rows(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    views = _stage_rows_views(profile)
    if not views:
        return ["stage_memory.rows: missing stage memory rows"]
    for path, rows in views:
        if not rows:
            errors.append(f"{path}: empty stage memory rows")
            continue
        for index, row in enumerate(rows):
            row_path = f"{path}[{index}]"
            missing = _missing(row, STAGE_FIELDS)
            if missing:
                errors.append(f"{row_path}: missing {', '.join(missing)}")
            reserved = row.get("max_peak_reserved_bytes")
            allocated = row.get("max_peak_allocated_bytes")
            if _is_number(reserved) and _is_number(allocated) and int(reserved) < int(allocated):
                errors.append(f"{row_path}: max peak reserved is less than max peak allocated")
    return errors


def _step_rows_views(profile: dict[str, Any], *, top_level_only: bool) -> list[tuple[str, list[dict[str, Any]]]]:
    raw_views = [profile] if top_level_only else profile_views(profile)
    views: list[tuple[str, list[dict[str, Any]]]] = []
    for index, view in enumerate(raw_views):
        step_samples = view.get("step_samples")
        rows = step_samples.get("rows") if isinstance(step_samples, dict) else None
        if isinstance(rows, list):
            view_path = "profile.step_samples.rows" if top_level_only else f"profile_views[{index}].step_samples.rows"
            views.append((view_path, [row for row in rows if isinstance(row, dict)]))
    return views


def validate_step_rows(profile: dict[str, Any], require_nsys_step_bytes: bool) -> list[str]:
    errors: list[str] = []
    views = _step_rows_views(profile, top_level_only=require_nsys_step_bytes)
    if not views:
        target = "top-level profile.step_samples.rows" if require_nsys_step_bytes else "step_samples.rows"
        return [f"{target}: missing step sample rows"]
    for path, rows in views:
        if not rows:
            errors.append(f"{path}: empty step sample rows")
            continue
        for index, row in enumerate(rows):
            row_path = f"{path}[{index}]"
            missing = _missing(row, WHOLE_STEP_FIELDS)
            if missing:
                errors.append(f"{row_path}: missing {', '.join(missing)}")
            else:
                errors.extend(
                    _validate_gap(row, row_path, "peak_allocated_hbm_bytes", "peak_reserved_hbm_bytes", "reserved_unallocated_bytes")
                )
            for prefix in ("forward", "backward"):
                required = {f"{prefix}_{suffix}" for suffix in PER_STAGE_STEP_SUFFIXES}
                stage_missing = _missing(row, required)
                if stage_missing:
                    errors.append(f"{row_path}: missing {', '.join(stage_missing)}")
                allocated = row.get(f"{prefix}_peak_allocated_bytes")
                reserved = row.get(f"{prefix}_peak_reserved_bytes")
                gap = row.get(f"{prefix}_reserved_unallocated_bytes")
                if all(_is_number(value) for value in (allocated, reserved, gap)):
                    expected_gap = max(0, int(reserved) - int(allocated))
                    if int(gap) != expected_gap:
                        errors.append(f"{row_path}: {prefix}_reserved_unallocated_bytes does not match reserved-allocated gap")
    return errors


def validate_breakdown(summary: dict[str, Any]) -> list[str]:
    errors = _legacy_key_errors(summary, "memory_breakdown_summary")
    missing = _missing(summary, BREAKDOWN_FIELDS)
    if missing:
        errors.append(f"memory_breakdown_summary: missing {', '.join(missing)}")
        return errors
    if int(summary.get("schema_version") or 0) != 2:
        errors.append("memory_breakdown_summary: schema_version must be 2")
    if summary.get("selected_metric") != "peak_allocated_hbm_bytes":
        errors.append("memory_breakdown_summary: selected_metric must be peak_allocated_hbm_bytes")
    errors.extend(
        _validate_gap(
            summary,
            "memory_breakdown_summary",
            "peak_allocated_hbm_bytes",
            "peak_reserved_hbm_bytes",
            "reserved_unallocated_bytes",
        )
    )
    rows = summary.get("breakdown_rows")
    if not isinstance(rows, list) or not rows:
        errors.append("memory_breakdown_summary.breakdown_rows: missing or empty")
        return errors
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if row.get("group") == "activations":
            errors.append(f"memory_breakdown_summary.breakdown_rows[{index}]: old activations group is not allowed")
        if row.get("component") == "framework_temp_workspace":
            errors.append(f"memory_breakdown_summary.breakdown_rows[{index}]: old framework_temp_workspace component is not allowed")
    allocator_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("component") == "allocator_reserved_unallocated"
    ]
    if not allocator_rows:
        errors.append("memory_breakdown_summary.breakdown_rows: missing allocator_reserved_unallocated row")
    allocated_sum = sum(
        int(row.get("bytes", 0) or 0)
        for row in rows
        if isinstance(row, dict) and row.get("memory_space") == "GPU HBM"
    )
    reserved_gap_sum = sum(int(row.get("bytes", 0) or 0) for row in allocator_rows)
    saved_activation_sum = sum(
        int(row.get("bytes", 0) or 0)
        for row in rows
        if isinstance(row, dict)
        and row.get("memory_space") == "GPU HBM"
        and row.get("group") == "saved_activations"
    )
    unattributed_sum = sum(
        int(row.get("bytes", 0) or 0)
        for row in rows
        if isinstance(row, dict)
        and row.get("memory_space") == "GPU HBM"
        and row.get("group") == "unattributed_allocated_peak"
    )
    external_sum = sum(
        int(row.get("bytes", 0) or 0)
        for row in rows
        if isinstance(row, dict) and row.get("component") == "external_cuda_or_driver"
    )
    peak_allocated = int(summary["peak_allocated_hbm_bytes"])
    peak_reserved = int(summary["peak_reserved_hbm_bytes"])
    allocated_error = peak_allocated - allocated_sum
    reserved_error = peak_reserved - allocated_sum - reserved_gap_sum
    if allocated_sum != int(summary.get("allocated_stack_sum_bytes", -1)):
        errors.append("memory_breakdown_summary: allocated_stack_sum_bytes does not match GPU HBM rows")
    if allocated_sum + reserved_gap_sum != int(summary.get("reserved_stack_sum_bytes", -1)):
        errors.append("memory_breakdown_summary: reserved_stack_sum_bytes does not match GPU HBM rows plus allocator gap")
    if saved_activation_sum != int(summary.get("saved_activation_hbm_bytes_at_peak", -1)):
        errors.append("memory_breakdown_summary: saved_activation_hbm_bytes_at_peak does not match saved_activations rows")
    if unattributed_sum != int(summary.get("unattributed_allocated_peak_bytes", -1)):
        errors.append("memory_breakdown_summary: unattributed_allocated_peak_bytes does not match unattributed row")
    if "external_cuda_or_driver_bytes" in summary and external_sum != int(summary.get("external_cuda_or_driver_bytes") or 0):
        errors.append("memory_breakdown_summary: external_cuda_or_driver_bytes does not match external diagnostic rows")
    if allocated_error != int(summary.get("allocated_closure_error_bytes", 0)):
        errors.append("memory_breakdown_summary: allocated_closure_error_bytes does not match rows")
    if reserved_error != int(summary.get("reserved_closure_error_bytes", 0)):
        errors.append("memory_breakdown_summary: reserved_closure_error_bytes does not match rows")
    if bool(summary.get("allocated_closure_ok")) is not True:
        errors.append("memory_breakdown_summary: allocated_closure_ok is not true")
    if bool(summary.get("reserved_closure_ok")) is not True:
        errors.append("memory_breakdown_summary: reserved_closure_ok is not true")
    return errors


def _breakdown_from_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    for view in profile_views(profile):
        breakdown = view.get("memory_breakdown")
        summary = breakdown.get("summary") if isinstance(breakdown, dict) else None
        if isinstance(summary, dict):
            return summary
        summary = view.get("memory_breakdown_summary")
        if isinstance(summary, dict):
            return summary
    return None


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    profiles: list[tuple[str, dict[str, Any]]] = []

    for label, path in (("profile-json", args.profile_json), ("source-profile-json", args.source_profile_json)):
        if path is None:
            continue
        try:
            profile = load_json(path)
        except Exception as exc:
            errors.append(f"{label}: failed to load {path}: {exc}")
            continue
        profiles.append((label, profile))
        errors.extend(f"{label}: {error}" for error in validate_memory_gpu(profile))
        errors.extend(f"{label}: {error}" for error in validate_stage_rows(profile))
        errors.extend(
            f"{label}: {error}"
            for error in validate_step_rows(profile, require_nsys_step_bytes=(label == "profile-json" and args.require_nsys_step_bytes))
        )

    if args.memory_breakdown_summary is not None:
        try:
            summary = load_json(args.memory_breakdown_summary)
            errors.extend(validate_breakdown(summary))
        except Exception as exc:
            errors.append(f"memory-breakdown-summary: failed to load {args.memory_breakdown_summary}: {exc}")
    elif args.require_breakdown:
        summary = None
        for _label, profile in profiles:
            summary = _breakdown_from_profile(profile)
            if summary is not None:
                break
        if summary is None:
            errors.append("memory-breakdown-summary: required but not found")
        else:
            errors.extend(validate_breakdown(summary))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
