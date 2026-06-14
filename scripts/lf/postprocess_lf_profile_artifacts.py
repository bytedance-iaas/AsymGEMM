#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write LF profile artifacts.")
    artifacts = parser.add_argument_group("artifact output")
    artifacts.add_argument("--profile-json", type=Path, help="Profile JSON to convert into CSV artifacts.")
    artifacts.add_argument("--source-profile-json", type=Path, help="Source-profile JSON to convert into markdown artifacts.")
    artifacts.add_argument("--output-dir", type=Path, help="Directory for profile artifacts.")
    args = parser.parse_args()
    if not any((args.profile_json, args.source_profile_json)):
        parser.error("provide --profile-json or --source-profile-json")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    return args


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = value.get("rows", [])
    else:
        rows = value
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if fieldnames:
            writer.writerows(rows)


def _fmt_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def _fmt_mib(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) / (1024.0 ** 2):.2f}"
    return "-"


def _fmt_pct(value: Any, total: Any) -> str:
    if isinstance(value, (int, float)) and isinstance(total, (int, float)) and float(total) > 0.0:
        return f"{float(value) * 100.0 / float(total):.2f}%"
    return "-"


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration_to_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            result = hours * 3600.0 + minutes * 60.0 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            result = minutes * 60.0 + seconds
        else:
            result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _trainer_log_records_for_timing(profile: dict[str, Any]) -> list[dict[str, Any]]:
    trainer = profile.get("trainer", {})
    if not isinstance(trainer, dict):
        return []
    records = trainer.get("records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    log_path = str(trainer.get("trainer_log") or "").strip()
    if not log_path:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _trainer_elapsed_by_step(profile: dict[str, Any]) -> dict[int, float]:
    elapsed_by_step: dict[int, float] = {}
    for record in _trainer_log_records_for_timing(profile):
        step = _int_value(record.get("current_steps", record.get("step")))
        elapsed_seconds = _duration_to_seconds(record.get("elapsed_time"))
        if step is None or step <= 0 or elapsed_seconds is None:
            continue
        elapsed_by_step[step] = max(elapsed_seconds, elapsed_by_step.get(step, 0.0))
    return elapsed_by_step


def _heartbeat_records_for_timing(profile: dict[str, Any]) -> list[dict[str, Any]]:
    heartbeat = profile.get("heartbeat", {})
    if not isinstance(heartbeat, dict):
        return []
    path = Path(str(heartbeat.get("jsonl") or "").strip())
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            rows.append(record)
    return rows


def _record_elapsed_seconds(record: dict[str, Any]) -> float | None:
    return _float_value(record.get("elapsed_seconds"))


def _heartbeat_step_interval_rows(
    profile: dict[str, Any], records: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if records is None:
        records = _heartbeat_records_for_timing(profile)
    if not records:
        return []

    dataloader_start: dict[int, float] = {}
    dataloader_end: dict[int, float] = {}
    training_start: dict[int, float] = {}
    training_end: dict[int, float] = {}
    optimizer_start: dict[int, float] = {}
    optimizer_end: dict[int, float] = {}
    trainer_end: float | None = None

    stage_maps: dict[str, dict[int, float]] = {
        "dataloader_batch_fetch_start": dataloader_start,
        "dataloader_batch_fetch_end": dataloader_end,
        "training_step_start": training_start,
        "training_step_end": training_end,
        "optimizer_step_start": optimizer_start,
        "optimizer_step_end": optimizer_end,
    }
    start_stages = {"dataloader_batch_fetch_start", "training_step_start", "optimizer_step_start"}
    for record in records:
        stage = record.get("stage")
        elapsed_seconds = _record_elapsed_seconds(record)
        if elapsed_seconds is None:
            continue
        if stage == "trainer_end":
            trainer_end = max(elapsed_seconds, trainer_end or 0.0)
            continue
        stage_map = stage_maps.get(str(stage))
        if stage_map is None:
            continue
        step = _int_value(record.get("count"))
        if step is None or step <= 0:
            global_step = _int_value(record.get("global_step"))
            step = global_step + 1 if global_step is not None and global_step >= 0 else None
        if step is None or step <= 0:
            continue
        previous = stage_map.get(step)
        if previous is None:
            stage_map[step] = elapsed_seconds
        elif str(stage) in start_stages:
            stage_map[step] = min(previous, elapsed_seconds)
        else:
            stage_map[step] = max(previous, elapsed_seconds)

    if not dataloader_start:
        return []
    rows: list[dict[str, Any]] = []
    for raw_step in sorted(dataloader_start):
        start = dataloader_start[raw_step]
        end = dataloader_start.get(raw_step + 1)
        end_source = "next_dataloader_batch_fetch_start"
        if end is None:
            end = trainer_end
            end_source = "trainer_end"
        if end is None or end < start:
            continue
        row: dict[str, Any] = {
            "raw_step": raw_step,
            "trainer_elapsed_seconds": end,
            "trainer_e2e_step_milliseconds": (end - start) * 1000.0,
            "trainer_e2e_start_seconds": start,
            "trainer_e2e_end_seconds": end,
            "trainer_e2e_end_source": end_source,
            "step_milliseconds_source": "heartbeat_dataloader_interval",
        }
        fetch_end = dataloader_end.get(raw_step)
        if fetch_end is not None and fetch_end >= start:
            row["heartbeat_dataloader_fetch_milliseconds"] = (fetch_end - start) * 1000.0
        train_start = training_start.get(raw_step)
        train_end = training_end.get(raw_step)
        if train_start is not None and train_end is not None and train_end >= train_start:
            row["heartbeat_training_step_milliseconds"] = (train_end - train_start) * 1000.0
        opt_start = optimizer_start.get(raw_step)
        opt_end = optimizer_end.get(raw_step)
        if opt_start is not None and opt_end is not None and opt_end >= opt_start:
            row["heartbeat_optimizer_step_milliseconds"] = (opt_end - opt_start) * 1000.0
        rows.append(row)
    return rows


def _heartbeat_step_intervals_by_step(
    profile: dict[str, Any], rows: list[dict[str, Any]] | None = None
) -> dict[int, dict[str, Any]]:
    if rows is None:
        rows = _heartbeat_step_interval_rows(profile)
    return {int(row["raw_step"]): row for row in rows if row.get("raw_step") is not None}


def _heartbeat_timing(profile: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = _heartbeat_step_interval_rows(profile)
    if not rows:
        return {}
    config = profile.get("config", {})
    warmup_steps = max(_int_value(config.get("warmup_steps")) or 0, 0) if isinstance(config, dict) else 0
    measured_rows = [row for row in rows if (_int_value(row.get("raw_step")) or 0) > warmup_steps]
    total_ms = sum(float(row["trainer_e2e_step_milliseconds"]) for row in rows)
    measured_ms = sum(float(row["trainer_e2e_step_milliseconds"]) for row in measured_rows)
    return {
        "available": True,
        "source": "heartbeat_dataloader_interval",
        "final_steps": len(rows),
        "final_elapsed_seconds": total_ms / 1000.0,
        "total_e2e_step_milliseconds": total_ms / float(len(rows)),
        "warmup_steps": warmup_steps,
        "measured_steps": len(measured_rows),
        "measured_elapsed_seconds": measured_ms / 1000.0,
        "measured_e2e_step_milliseconds": measured_ms / float(len(measured_rows)) if measured_rows else None,
        "note": "batch-fetch-start to next-batch-fetch-start intervals; final step ends at trainer_end",
    }


def _stage_memory_row(profile: dict[str, Any], name: str) -> dict[str, Any]:
    rows = profile.get("stage_memory", {}).get("rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _stage_timing_ms(profile: dict[str, Any], name: str) -> float | None:
    step = profile.get("step", {})
    rows = step.get("rows", []) if isinstance(step, dict) else []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("name") == name:
                value = row.get("milliseconds")
                if isinstance(value, (int, float)):
                    return float(value)
    row = _stage_memory_row(profile, name)
    value = row.get("milliseconds")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _trainer_timing(profile: dict[str, Any]) -> dict[str, Any]:
    trainer = profile.get("trainer", {})
    if not isinstance(trainer, dict):
        return {}
    timing = trainer.get("timing", {})
    if isinstance(timing, dict) and timing.get("available") and timing.get("source") == "heartbeat_dataloader_interval":
        return timing
    heartbeat_timing = _heartbeat_timing(profile)
    if heartbeat_timing.get("available"):
        return heartbeat_timing
    return timing if isinstance(timing, dict) else {}


def _kt_counter_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    kt = profile.get("kt", {})
    if not isinstance(kt, dict):
        return []
    rows = kt.get("rows", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _lora_counter_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    lora = profile.get("lora", {})
    if not isinstance(lora, dict):
        return []
    row = {key: value for key, value in lora.items() if not isinstance(value, (dict, list))}
    return [row] if row else []


def _cpuadam_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cpuadam = profile.get("cpuadam", {})
    if not isinstance(cpuadam, dict):
        return []
    row = {key: value for key, value in cpuadam.items() if not isinstance(value, (dict, list))}
    return [row] if row else []


def _asym_cpu_adamw_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cpuadamw = profile.get("asym_cpu_adamw", {})
    if not isinstance(cpuadamw, dict):
        return []
    row = {key: value for key, value in cpuadamw.items() if not isinstance(value, (dict, list))}
    return [row] if row else []


def _source_profile_for_artifacts(profile: dict[str, Any]) -> dict[str, Any]:
    source_profile = profile.get("source_profile")
    return source_profile if isinstance(source_profile, dict) and source_profile else profile


def _counter_value(container: Any, key: str, default: Any = "-") -> Any:
    if not isinstance(container, dict):
        return "missing"
    if container.get("available") is False:
        reason = container.get("reason")
        return f"missing ({reason})" if reason else "missing"
    if key not in container:
        return default
    value = container.get(key)
    return default if value is None else value


def _trainer_log_trainable_params(profile: dict[str, Any]) -> int | None:
    trainer = profile.get("trainer", {})
    if not isinstance(trainer, dict):
        return None
    log_path = str(trainer.get("trainer_log") or "").strip()
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    patterns = (
        r"trainable params:\s*([0-9][0-9,]*)",
        r"Number of trainable parameters\s*=\s*([0-9][0-9,]*)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return int(matches[-1].replace(",", ""))
    return None


def _trainable_param_value(profile: dict[str, Any]) -> int | None:
    value = _int_counter(profile.get("lora", {}), "trainable_parameters")
    return value if value is not None else _trainer_log_trainable_params(profile)


def _trainable_param_display(profile: dict[str, Any]) -> Any:
    value = _int_counter(profile.get("lora", {}), "trainable_parameters")
    if value is not None:
        return value
    fallback = _trainer_log_trainable_params(profile)
    if fallback is not None:
        return f"{fallback} (trainer log fallback)"
    return _counter_value(profile.get("lora", {}), "trainable_parameters")


def _int_counter(container: Any, key: str) -> int | None:
    if not isinstance(container, dict) or container.get("available") is False:
        return None
    value = container.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _trainable_surface_summary(profile: dict[str, Any]) -> dict[str, Any]:
    config = profile.get("config", {})
    lora = profile.get("lora", {})

    def unknown_surface(reason: str, fallback_trainable: int | None = None, source: str | None = None) -> dict[str, Any]:
        if fallback_trainable is None:
            fallback_trainable = _trainer_log_trainable_params(profile)
        structured_trainable = _int_counter(lora, "trainable_parameters") if isinstance(lora, dict) else None
        if fallback_trainable is None:
            fallback_trainable = structured_trainable
        if source is None:
            if structured_trainable is not None:
                source = "structured_lora"
            elif fallback_trainable is not None:
                source = "trainer_log"
            else:
                source = "missing"
        return {
            "available": fallback_trainable is not None,
            "reason": reason,
            "surface": "unknown",
            "comparison_note": "only trainer-log trainable parameter count is available"
            if source == "trainer_log"
            else "structured trainable parameter count is available but LoRA surface counters are incomplete"
            if source == "structured_lora"
            else reason,
            "trainable_parameters": fallback_trainable,
            "trainable_parameters_source": source,
        }

    if not isinstance(lora, dict):
        return unknown_surface("missing lora counters")
    if not lora:
        return unknown_surface("missing lora counters")
    if lora.get("available") is False:
        reason = str(lora.get("reason") or "lora counters unavailable")
        return unknown_surface(reason, source="trainer_log" if _trainer_log_trainable_params(profile) is not None else None)

    peft_lora = _int_counter(lora, "peft_lora_parameters")
    surface_counter_keys = (
        "peft_lora_parameters",
        "peft_expert_lora_parameters",
        "kt_peft_expert_lora_parameters",
        "kt_expert_lora_parameters",
        "lf_fused_expert_lora_parameters",
        "kt_fused_expert_lora_parameters",
    )
    if not any(_int_counter(lora, key) is not None for key in surface_counter_keys):
        return unknown_surface("missing LoRA surface counters")

    peft_expert = _int_counter(lora, "peft_expert_lora_parameters")
    kt_peft_expert = _int_counter(lora, "kt_peft_expert_lora_parameters") or 0
    if peft_expert is None:
        peft_expert = kt_peft_expert
    peft_expert = peft_expert or 0
    kt_expert = _int_counter(lora, "kt_expert_lora_parameters") or 0
    lf_fused_expert = _int_counter(lora, "lf_fused_expert_lora_parameters") or 0
    expert_lora = max(kt_expert, peft_expert + lf_fused_expert)
    non_expert_peft = None if peft_lora is None else max(0, peft_lora - peft_expert)

    if expert_lora > 0 and (non_expert_peft or 0) > 0:
        surface = "attention+expert LoRA"
    elif expert_lora > 0:
        surface = "expert LoRA"
    elif (non_expert_peft or 0) > 0:
        surface = "attention-only LoRA"
    else:
        surface = "no trainable LoRA detected"

    backend = str(config.get("backend") or "")
    is_kt = backend.startswith("kt_") or str(config.get("kt_backend") or "").strip() not in {"", "-"}
    if expert_lora > 0:
        comparison_note = "requires a baseline that also trains expert LoRA"
    elif (non_expert_peft or 0) > 0:
        comparison_note = "attention-only surface; do not compare directly to expert-LoRA KT runs"
    elif is_kt:
        comparison_note = "KT run has no captured expert LoRA; verify the adapter surface before comparison"
    else:
        comparison_note = "no trainable LoRA surface captured"

    return {
        "available": True,
        "surface": surface,
        "comparison_note": comparison_note,
        "trainable_parameters": _trainable_param_value(profile),
        "trainable_parameters_source": "structured_lora"
        if _int_counter(lora, "trainable_parameters") is not None
        else "trainer_log",
        "peft_lora_parameters": peft_lora,
        "non_expert_peft_lora_parameters": non_expert_peft,
        "expert_lora_parameters": expert_lora,
        "peft_expert_lora_parameters": peft_expert,
        "lf_fused_expert_lora_parameters": lf_fused_expert,
        "kt_expert_lora_parameters": kt_expert,
        "kt_peft_expert_lora_parameters": kt_peft_expert,
        "kt_fused_expert_lora_parameters": _int_counter(lora, "kt_fused_expert_lora_parameters") or 0,
    }


def _trainable_surface_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [_trainable_surface_summary(profile)]


def _kt_lora_update_health(profile: dict[str, Any]) -> dict[str, Any]:
    optimizer_memory = profile.get("optimizer_memory", {})
    if not isinstance(optimizer_memory, dict):
        return {}
    health = optimizer_memory.get("kt_lora_update_health", {})
    if not isinstance(health, dict):
        return {}
    return _normalize_kt_lora_update_health(health)


def _normalize_kt_lora_update_health(health: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(health)
    if "passed" in normalized:
        normalized.setdefault("input_passed", normalized.get("passed"))
    if "reason" in normalized:
        normalized.setdefault("input_reason", normalized.get("reason"))
    rows = normalized.get("rows", [])
    row_dicts = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    updated_grad_tensors = sum(
        int(
            bool(row.get("nonzero_grad_changed_after_step"))
            or (bool(row.get("grad_nonzero_before_step")) and bool(row.get("param_changed_after_step")))
        )
        for row in row_dicts
    )
    grad_nonzero_unchanged_tensors = sum(
        int(bool(row.get("grad_nonzero_before_step")) and not bool(row.get("param_changed_after_step")))
        for row in row_dicts
    )

    normalized["updated_grad_tensors"] = int(normalized.get("updated_grad_tensors", updated_grad_tensors) or 0)
    normalized["grad_nonzero_unchanged_tensors"] = int(
        normalized.get("grad_nonzero_unchanged_tensors", grad_nonzero_unchanged_tensors) or 0
    )
    compared_tensors = int(normalized.get("compared_tensors", len(row_dicts)) or 0)
    grad_nonzero_tensors = int(normalized.get("grad_nonzero_tensors", 0) or 0)
    sampled_tensors = int(normalized.get("sampled_tensors", len(row_dicts)) or 0)
    total_fused_tensors = int(normalized.get("total_fused_tensors", 0) or 0)
    after_sampled_tensors = normalized.get("after_sampled_tensors")
    after_total_fused_tensors = normalized.get("after_total_fused_tensors")
    try:
        after_sampled_tensors_int = int(after_sampled_tensors) if after_sampled_tensors not in (None, "") else None
    except (TypeError, ValueError):
        after_sampled_tensors_int = None
    try:
        after_total_fused_tensors_int = (
            int(after_total_fused_tensors) if after_total_fused_tensors not in (None, "") else None
        )
    except (TypeError, ValueError):
        after_total_fused_tensors_int = None
    missing_after_tensors = int(normalized.get("missing_after_tensors", 0) or 0)
    unexpected_after_tensors = int(normalized.get("unexpected_after_tensors", 0) or 0)
    missing_snapshot_fields = [
        key
        for key in (
            "after_sampled_tensors",
            "after_total_fused_tensors",
            "missing_after_tensors",
            "unexpected_after_tensors",
        )
        if key not in normalized or normalized.get(key) in (None, "")
    ]
    normalized.setdefault("exhaustive", bool(total_fused_tensors > 0 and sampled_tensors == total_fused_tensors))
    normalized.setdefault("exhaustive_elements", False)
    if normalized.get("available") is False:
        derived_passed = False
        derived_reason = "KT fused LoRA update health unavailable"
    elif missing_snapshot_fields:
        derived_passed = False
        derived_reason = (
            "KT fused LoRA update health missing after-step snapshot fields: " + ", ".join(missing_snapshot_fields)
        )
    elif after_sampled_tensors_int is None or after_total_fused_tensors_int is None:
        derived_passed = False
        derived_reason = "KT fused LoRA update health has invalid after-step snapshot counts"
    elif missing_after_tensors > 0 or unexpected_after_tensors > 0:
        derived_passed = False
        derived_reason = (
            "sampled fused LoRA tensor set changed between before/after optimizer snapshots "
            f"(missing_after={missing_after_tensors}, unexpected_after={unexpected_after_tensors})"
        )
    elif sampled_tensors != after_sampled_tensors_int or total_fused_tensors != after_total_fused_tensors_int:
        derived_passed = False
        derived_reason = (
            "fused LoRA tensor counts changed between before/after optimizer snapshots "
            f"(sampled {sampled_tensors}->{after_sampled_tensors_int}, "
            f"total {total_fused_tensors}->{after_total_fused_tensors_int})"
        )
    elif normalized.get("exhaustive") and compared_tensors != sampled_tensors:
        derived_passed = False
        derived_reason = f"exhaustive fused LoRA health compared {compared_tensors} of {sampled_tensors} tensors"
    elif compared_tensors <= 0:
        derived_passed = False
        derived_reason = "no sampled fused LoRA tensors were comparable"
    elif grad_nonzero_tensors <= 0:
        derived_passed = False
        derived_reason = "no sampled fused LoRA tensors had nonzero gradients before optimizer step"
    elif normalized["grad_nonzero_unchanged_tensors"] > 0:
        derived_passed = False
        derived_reason = "one or more sampled nonzero-gradient fused LoRA tensors did not change after optimizer step"
    elif normalized["updated_grad_tensors"] <= 0:
        derived_passed = False
        derived_reason = "no sampled nonzero-gradient fused LoRA tensors changed after optimizer step"
    else:
        derived_passed = True
        if normalized.get("exhaustive_elements"):
            derived_reason = "all nonzero-gradient fused LoRA tensors changed after optimizer step with full-element checksums"
        elif normalized.get("exhaustive"):
            derived_reason = "all nonzero-gradient fused LoRA tensors changed after optimizer step"
        else:
            derived_reason = "all sampled nonzero-gradient fused LoRA tensors changed after optimizer step"

    if normalized.get("input_passed") is False and derived_passed:
        normalized["passed"] = False
        normalized["reason"] = str(normalized.get("input_reason") or "input KT fused LoRA update health failed")
    else:
        normalized["passed"] = derived_passed
        normalized["reason"] = derived_reason
    return normalized


def _kt_lora_update_health_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    health = _kt_lora_update_health(profile)
    if not health:
        return []
    summary = {
        "available": health.get("available"),
        "passed": health.get("passed"),
        "reason": health.get("reason"),
        "sampled_tensors": health.get("sampled_tensors"),
        "total_fused_tensors": health.get("total_fused_tensors"),
        "after_sampled_tensors": health.get("after_sampled_tensors"),
        "after_total_fused_tensors": health.get("after_total_fused_tensors"),
        "exhaustive": health.get("exhaustive"),
        "exhaustive_elements": health.get("exhaustive_elements"),
        "requested_max_tensors": health.get("requested_max_tensors"),
        "requested_max_elements": health.get("requested_max_elements"),
        "compared_tensors": health.get("compared_tensors"),
        "missing_after_tensors": health.get("missing_after_tensors"),
        "unexpected_after_tensors": health.get("unexpected_after_tensors"),
        "grad_nonzero_tensors": health.get("grad_nonzero_tensors"),
        "updated_grad_tensors": health.get("updated_grad_tensors"),
        "grad_nonzero_unchanged_tensors": health.get("grad_nonzero_unchanged_tensors"),
        "changed_tensors": health.get("changed_tensors"),
        "input_passed": health.get("input_passed"),
        "input_reason": health.get("input_reason"),
    }
    rows = health.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return [summary]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append({**summary, **row})
    return normalized or [summary]


def _optimizer_memory_preflight(profile: dict[str, Any]) -> dict[str, Any]:
    preflight = profile.get("optimizer_memory_preflight", {})
    if isinstance(preflight, dict):
        return preflight
    return {}


def _optimizer_memory_preflight_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = _optimizer_memory_preflight(profile)
    if not preflight:
        return []
    return [{key: value for key, value in preflight.items() if not isinstance(value, (dict, list))}]


def _grad_clip_summary(profile: dict[str, Any]) -> dict[str, Any]:
    grad_clip = profile.get("grad_clip", {})
    return grad_clip if isinstance(grad_clip, dict) else {}


def _grad_clip_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    grad_clip = _grad_clip_summary(profile)
    if not grad_clip:
        return []
    return [{key: value for key, value in grad_clip.items() if not isinstance(value, (dict, list))}]


def _process_memory(profile: dict[str, Any]) -> dict[str, Any]:
    memory = profile.get("memory", {})
    process = memory.get("process", {}) if isinstance(memory, dict) else {}
    return process if isinstance(process, dict) else {}


def _process_memory_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    process = _process_memory(profile)
    if process:
        rows.append({"name": "report", **{key: value for key, value in process.items() if not isinstance(value, (dict, list))}})
    stage_rows = profile.get("stage_memory", {}).get("rows", [])
    if isinstance(stage_rows, list):
        for row in stage_rows:
            if not isinstance(row, dict):
                continue
            process_items = {
                key: value
                for key, value in row.items()
                if key == "name" or key.startswith("avg_process_") or key.startswith("max_process_")
            }
            if len(process_items) > 1:
                rows.append(process_items)
    optimizer_memory = profile.get("optimizer_memory", {})
    if isinstance(optimizer_memory, dict):
        at_start = optimizer_memory.get("process_memory_at_start", {})
        before = optimizer_memory.get("process_memory_before_step", {})
        after = optimizer_memory.get("process_memory_after_step", {})
        if isinstance(at_start, dict) and at_start:
            rows.append(
                {
                    "name": "optimizer_step_start",
                    "process_rss_pre_step_overhead_delta_bytes": optimizer_memory.get(
                        "process_rss_pre_step_overhead_delta_bytes"
                    ),
                    **{key: value for key, value in at_start.items() if not isinstance(value, (dict, list))},
                }
            )
        if isinstance(before, dict) and before:
            rows.append(
                {
                    "name": "optimizer_step_before",
                    "process_rss_pre_step_overhead_delta_bytes": optimizer_memory.get(
                        "process_rss_pre_step_overhead_delta_bytes"
                    ),
                    **{key: value for key, value in before.items() if not isinstance(value, (dict, list))},
                }
            )
        if isinstance(after, dict) and after:
            rows.append(
                {
                    "name": "optimizer_step_after",
                    "process_rss_delta_bytes": optimizer_memory.get("process_rss_delta_bytes"),
                    **{key: value for key, value in after.items() if not isinstance(value, (dict, list))},
                }
            )
    return rows


def _source_summary_markdown(profile: dict[str, Any]) -> str:
    config = profile.get("config", {})
    warmup_steps = int(config.get("warmup_steps", 0) or 0)
    measure_steps = config.get("measure_steps", config.get("max_steps", "-"))
    memory = profile.get("memory", {})
    gpu = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    trainer = profile.get("trainer", {})
    kt = profile.get("kt", {})
    lora = profile.get("lora", {})
    trainable_surface = _trainable_surface_summary(profile)
    kt_lora_update_health = _kt_lora_update_health(profile)
    optimizer_memory_preflight = _optimizer_memory_preflight(profile)
    grad_clip = _grad_clip_summary(profile)
    process_memory = _process_memory(profile)
    process_memory_rows = _process_memory_rows(profile)
    timing = _trainer_timing(profile)
    forward_ms = _float_value(profile.get("forward", {}).get("total_milliseconds"))
    backward_ms = _float_value(profile.get("backward", {}).get("total_milliseconds"))
    forward_backward_ms = forward_ms + backward_ms if forward_ms is not None and backward_ms is not None else None
    lines = [
        "# LF LoRA-SFT Source Profile",
        "",
        f"Workload: `{profile.get('workload', '-')}`  ",
        f"Backend: `{config.get('backend', '-')}`  ",
        f"Router mode: `{config.get('router_mode', '-')}`  ",
        f"Expert activation offload: `{config.get('asymm_expert_act_offload', '-')}`  ",
        f"Attention activation offload: `{config.get('asymm_attn_act_offload', '-')}`  ",
        f"Layer activation offload: `{config.get('asymm_layer_act_offload', '-')}`  ",
        f"Attention GC: `{config.get('attention_gc_enabled', config.get('attn_gc_enabled', '-'))}`  ",
        f"Layer GC: `{config.get('layer_gc_enabled', '-')}`  ",
        f"Precision: `{config.get('precision', '-')}`  ",
        f"Seq len: `{config.get('seq_len', '-')}`",
        f"Steps: `{warmup_steps}` warmup + `{measure_steps}` measured",
        "",
        "| Stage | host ms | avg allocated start MiB | avg allocated end MiB | avg peak allocated MiB | avg peak reserved MiB | samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stage_names: list[tuple[str, Any]] = [
        ("trainer.e2e.measured_step", timing.get("measured_e2e_step_milliseconds")),
        ("trainer.e2e.total_step", timing.get("total_e2e_step_milliseconds")),
        ("lf.training_step.total", _stage_timing_ms(profile, "lf.step.total")),
        ("step.forward + step.backward", forward_backward_ms),
        ("step.forward", forward_ms),
        ("step.backward", backward_ms),
        ("lf.grad_clip", _stage_timing_ms(profile, "lf.grad_clip")),
        ("lf.optimizer.step", _stage_timing_ms(profile, "lf.optimizer.step")),
        ("lf.scheduler.step", _stage_timing_ms(profile, "lf.scheduler.step")),
    ]
    seen_stage_names: set[str] = set()
    for name, total_ms in stage_names:
        if name in seen_stage_names:
            continue
        seen_stage_names.add(name)
        if total_ms is None:
            continue
        row = _stage_memory_row(profile, name)
        if not row and name == "lf.training_step.total":
            row = _stage_memory_row(profile, "lf.step.total")
        lines.append(
            "| {name} | {ms} | {start} | {end} | {peak_allocated} | {peak_reserved} | {samples} |".format(
                name=name,
                ms=_fmt_ms(total_ms),
                start=_fmt_mib(row.get("avg_allocated_start_bytes")),
                end=_fmt_mib(row.get("avg_allocated_end_bytes")),
                peak_allocated=_fmt_mib(row.get("avg_peak_allocated_bytes")),
                peak_reserved=_fmt_mib(row.get("avg_peak_reserved_bytes")),
                samples=row.get("samples", "-"),
            )
        )
    lines += [
        "",
        f"Peak allocated HBM: `{_fmt_mib(gpu.get('peak_allocated_hbm_bytes'))} MiB`",
        f"Peak reserved HBM: `{_fmt_mib(gpu.get('peak_reserved_hbm_bytes'))} MiB`",
        f"Reserved but unallocated: `{_fmt_mib(gpu.get('reserved_unallocated_bytes'))} MiB`",
        f"Timing source: `{timing.get('source', '-')}`",
        f"Trainer log: `{trainer.get('trainer_log', '')}`",
        "",
    ]
    if process_memory:
        lines += [
            "## Process Memory",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| available | {process_memory.get('available', '-')} |",
            f"| RSS bytes | {process_memory.get('rss_bytes', '-')} |",
            f"| RSS MiB | {_fmt_mib(process_memory.get('rss_bytes'))} |",
            f"| RSS peak bytes | {process_memory.get('rss_peak_bytes', '-')} |",
            f"| RSS peak MiB | {_fmt_mib(process_memory.get('rss_peak_bytes'))} |",
            f"| virtual memory bytes | {process_memory.get('virtual_memory_bytes', '-')} |",
            f"| source | {process_memory.get('source', '-')} |",
            "",
        ]
        if process_memory_rows:
            lines += [
                "| Sample | RSS bytes | RSS peak bytes | virtual memory bytes | RSS delta bytes |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in process_memory_rows:
                name = row.get("name", "-")
                rss_bytes = row.get("rss_bytes", row.get("avg_process_rss_end_bytes", "-"))
                rss_peak_bytes = row.get("rss_peak_bytes", row.get("max_process_rss_peak_end_bytes", "-"))
                virtual_memory_bytes = row.get(
                    "virtual_memory_bytes", row.get("avg_process_virtual_memory_end_bytes", "-")
                )
                rss_delta_bytes = row.get("process_rss_delta_bytes", row.get("avg_process_rss_delta_bytes", "-"))
                lines.append(f"| {name} | {rss_bytes} | {rss_peak_bytes} | {virtual_memory_bytes} | {rss_delta_bytes} |")
            lines.append("")
    if grad_clip:
        lines += [
            "## Gradient Clipping",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| available | {grad_clip.get('available', '-')} |",
            f"| path | {grad_clip.get('path', '-')} |",
            f"| operation | {grad_clip.get('operation', '-')} |",
            f"| method | {grad_clip.get('method', '-')} |",
            f"| max norm | {grad_clip.get('max_norm', '-')} |",
            f"| total norm | {grad_clip.get('total_norm', '-')} |",
            f"| result norm | {grad_clip.get('result_norm', '-')} |",
            f"| clip coefficient | {grad_clip.get('clip_coef', '-')} |",
            f"| clipped | {grad_clip.get('clipped', '-')} |",
            f"| nonfinite | {grad_clip.get('nonfinite', '-')} |",
            f"| elapsed ms | {_fmt_ms(grad_clip.get('elapsed_ms'))} |",
            f"| unique grad tensors | {grad_clip.get('unique_grad_tensors', '-')} |",
            f"| duplicate grad tensors | {grad_clip.get('duplicate_grad_tensors', '-')} |",
            f"| CPU grad tensors | {grad_clip.get('cpu_grad_tensors', '-')} |",
            f"| CPU grad elements | {grad_clip.get('cpu_grad_numel', '-')} |",
            f"| CUDA grad tensors | {grad_clip.get('cuda_grad_tensors', '-')} |",
            f"| CUDA grad elements | {grad_clip.get('cuda_grad_numel', '-')} |",
            f"| chunk elements | {grad_clip.get('chunk_elements', '-')} |",
            "",
        ]
    if isinstance(kt, dict) or isinstance(lora, dict):
        lines += [
            "## KT / LoRA Counters",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        if isinstance(kt, dict):
            lines += [
                f"| KT backend | {config.get('kt_backend', '-')} |",
                f"| KT wrappers | {_counter_value(kt, 'wrapper_count')} |",
                f"| KT forward calls | {_counter_value(kt, 'total_forward_calls')} |",
                f"| KT backward calls | {_counter_value(kt, 'total_backward_calls')} |",
            ]
        if isinstance(lora, dict):
            lines += [
                f"| trainable params | {_trainable_param_display(profile)} |",
                f"| PEFT LoRA params | {_counter_value(lora, 'peft_lora_parameters')} |",
                f"| LF fused expert LoRA params | {_counter_value(lora, 'lf_fused_expert_lora_parameters')} |",
                f"| LF fused expert LoRA tensors | {_counter_value(lora, 'lf_fused_expert_lora_tensors')} |",
                f"| KT expert LoRA params | {_counter_value(lora, 'kt_expert_lora_parameters')} |",
                f"| KT PEFT-view expert LoRA params | {_counter_value(lora, 'kt_peft_expert_lora_parameters')} |",
                f"| KT fused expert LoRA params | {_counter_value(lora, 'kt_fused_expert_lora_parameters')} |",
                f"| KT fused expert LoRA tensors | {_counter_value(lora, 'kt_fused_expert_lora_tensors')} |",
                f"| KT fused expert LoRA sidecar expected tensors | {_counter_value(lora, 'kt_fused_expert_lora_tensors')} |",
                f"| KT fused expert LoRA sidecar expected params | {_counter_value(lora, 'kt_fused_expert_lora_parameters')} |",
            ]
            lines += [
                f"| trainable surface | {trainable_surface.get('surface', '-')} |",
                f"| non-expert PEFT LoRA params | {trainable_surface.get('non_expert_peft_lora_parameters', '-')} |",
                f"| PEFT expert LoRA params | {trainable_surface.get('peft_expert_lora_parameters', '-')} |",
                f"| expert LoRA params | {trainable_surface.get('expert_lora_parameters', '-')} |",
                f"| backend comparison note | {trainable_surface.get('comparison_note', '-')} |",
            ]
        lines.append("")
    if optimizer_memory_preflight:
        lines += [
            "## Optimizer Memory Preflight",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| available | {optimizer_memory_preflight.get('available', '-')} |",
            f"| source | {optimizer_memory_preflight.get('source', '-')} |",
            f"| reason | {optimizer_memory_preflight.get('reason', '-')} |",
            f"| assumed param dtype | {optimizer_memory_preflight.get('assumed_param_dtype', '-')} |",
            f"| logical qlen | {optimizer_memory_preflight.get('logical_qlen', '-')} |",
            f"| LoRA rank | {optimizer_memory_preflight.get('lora_rank', '-')} |",
            f"| KT ARM top-k | {optimizer_memory_preflight.get('kt_arm_sft_top_k', '-')} |",
            f"| KT ARM token chunk size | {optimizer_memory_preflight.get('kt_arm_sft_token_chunk_size', '-')} |",
            f"| KT ARM effective route qlen | {optimizer_memory_preflight.get('kt_arm_effective_route_qlen', '-')} |",
            f"| KT ARM token chunks | {optimizer_memory_preflight.get('kt_arm_token_chunks', '-')} |",
            f"| KT ARM route-rank work | {optimizer_memory_preflight.get('kt_arm_route_rank_work', '-')} |",
            f"| KT ARM route-rank cap | {optimizer_memory_preflight.get('kt_arm_sft_max_route_rank_work', '-')} |",
            f"| trainable params | {optimizer_memory_preflight.get('trainable_parameters', '-')} |",
            f"| expert LoRA params | {optimizer_memory_preflight.get('expert_lora_parameters', '-')} |",
            f"| LF fused expert LoRA params | {optimizer_memory_preflight.get('lf_fused_expert_lora_parameters', '-')} |",
            f"| KT fused expert LoRA params | {optimizer_memory_preflight.get('kt_fused_expert_lora_parameters', '-')} |",
            f"| KT expert LoRA params | {optimizer_memory_preflight.get('kt_expert_lora_parameters', '-')} |",
            f"| non-expert PEFT LoRA params | {optimizer_memory_preflight.get('non_expert_peft_lora_parameters', '-')} |",
            f"| BF16 param bytes | {optimizer_memory_preflight.get('param_bf16_bytes', '-')} |",
            f"| BF16 grad bytes | {optimizer_memory_preflight.get('grad_bf16_bytes', '-')} |",
            f"| AdamW BF16 moments bytes | {optimizer_memory_preflight.get('adamw_bf16_moments_bytes', '-')} |",
            f"| AdamW FP32 moments bytes | {optimizer_memory_preflight.get('adamw_fp32_moments_bytes', '-')} |",
            f"| FP32 master bytes | {optimizer_memory_preflight.get('adamw_fp32_master_bytes', '-')} |",
            f"| total BF16 params/grads + BF16 moments bytes | {optimizer_memory_preflight.get('total_bf16_params_grads_adamw_bf16_moments_bytes', '-')} |",
            f"| total BF16 params/grads + FP32 moments bytes | {optimizer_memory_preflight.get('total_bf16_params_grads_adamw_fp32_moments_bytes', '-')} |",
            f"| total with FP32 moments + master bytes | {optimizer_memory_preflight.get('total_bf16_params_grads_adamw_fp32_moments_master_bytes', '-')} |",
            f"| large surface warning | {optimizer_memory_preflight.get('large_surface_warning', '-')} |",
            "",
        ]
    if kt_lora_update_health:
        lines += [
            "## KT Fused LoRA Update Health",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| available | {kt_lora_update_health.get('available', '-')} |",
            f"| passed | {kt_lora_update_health.get('passed', '-')} |",
            f"| reason | {kt_lora_update_health.get('reason', '-')} |",
            f"| total fused tensors | {kt_lora_update_health.get('total_fused_tensors', '-')} |",
            f"| sampled tensors | {kt_lora_update_health.get('sampled_tensors', '-')} |",
            f"| after total fused tensors | {kt_lora_update_health.get('after_total_fused_tensors', '-')} |",
            f"| after sampled tensors | {kt_lora_update_health.get('after_sampled_tensors', '-')} |",
            f"| exhaustive | {kt_lora_update_health.get('exhaustive', '-')} |",
            f"| exhaustive elements | {kt_lora_update_health.get('exhaustive_elements', '-')} |",
            f"| requested max tensors | {kt_lora_update_health.get('requested_max_tensors', '-')} |",
            f"| requested max elements | {kt_lora_update_health.get('requested_max_elements', '-')} |",
            f"| compared tensors | {kt_lora_update_health.get('compared_tensors', '-')} |",
            f"| missing after tensors | {kt_lora_update_health.get('missing_after_tensors', '-')} |",
            f"| unexpected after tensors | {kt_lora_update_health.get('unexpected_after_tensors', '-')} |",
            f"| nonzero-gradient sampled tensors | {kt_lora_update_health.get('grad_nonzero_tensors', '-')} |",
            f"| nonzero-gradient tensors changed | {kt_lora_update_health.get('updated_grad_tensors', '-')} |",
            f"| nonzero-gradient tensors unchanged | {kt_lora_update_health.get('grad_nonzero_unchanged_tensors', '-')} |",
            f"| changed sampled tensors | {kt_lora_update_health.get('changed_tensors', '-')} |",
            "",
        ]
    losses = trainer.get("losses", [])
    if isinstance(losses, list) and losses:
        measured_losses = [row for row in losses if isinstance(row, dict) and not row.get("is_warmup")]
        lines += [
            "## Measured Losses",
            "",
            "| Step | Raw step | Loss |",
            "|---:|---:|---:|",
        ]
        for row in measured_losses:
            lines.append(f"| {row.get('measured_step', row.get('step', '-'))} | {row.get('raw_step', '-')} | {row.get('loss', '-')} |")
        lines.append("")
    return "\n".join(lines)


def _source_latency_markdown(profile: dict[str, Any]) -> str:
    timing = _trainer_timing(profile)
    e2e_measured_ms = timing.get("measured_e2e_step_milliseconds")
    e2e_total_ms = timing.get("total_e2e_step_milliseconds")
    training_step_ms = _stage_timing_ms(profile, "lf.step.total")
    optimizer_ms = _stage_timing_ms(profile, "lf.optimizer.step")
    grad_clip_ms = _stage_timing_ms(profile, "lf.grad_clip")
    scheduler_ms = _stage_timing_ms(profile, "lf.scheduler.step")
    forward_ms = profile.get("forward", {}).get("total_milliseconds")
    backward_ms = profile.get("backward", {}).get("total_milliseconds")
    forward_backward_ms = (
        float(forward_ms) + float(backward_ms)
        if isinstance(forward_ms, (int, float)) and isinstance(backward_ms, (int, float))
        else None
    )
    optimizer_side_ms = (
        float(e2e_measured_ms) - float(forward_backward_ms)
        if isinstance(e2e_measured_ms, (int, float)) and isinstance(forward_backward_ms, (int, float))
        else None
    )
    return "\n".join(
        [
            "# LF LoRA-SFT Latency",
            "",
            "| Metric | ms |",
            "|---|---:|",
            f"| trainer e2e measured step incl optimizer | {_fmt_ms(e2e_measured_ms)} |",
            f"| trainer e2e total step incl warmup | {_fmt_ms(e2e_total_ms)} |",
            f"| optimizer/update side = e2e measured - fwd/bwd | {_fmt_ms(optimizer_side_ms)} |",
            f"| lf.training_step.total | {_fmt_ms(training_step_ms)} |",
            f"| step.forward + step.backward | {_fmt_ms(forward_backward_ms)} |",
            f"| step.forward | {_fmt_ms(forward_ms)} |",
            f"| step.backward | {_fmt_ms(backward_ms)} |",
            f"| lf.grad_clip | {_fmt_ms(grad_clip_ms)} |",
            f"| lf.optimizer.step substage | {_fmt_ms(optimizer_ms)} |",
            f"| lf.scheduler.step | {_fmt_ms(scheduler_ms)} |",
            "",
            f"Timing source: `{timing.get('source', '-')}`",
            "",
        ]
    )


def _memory_breakdown_summary(profile: dict[str, Any]) -> dict[str, Any]:
    breakdown = profile.get("memory_breakdown", {})
    if isinstance(breakdown, dict):
        summary = breakdown.get("summary", {})
        if isinstance(summary, dict):
            return summary
    summary = profile.get("memory_breakdown_summary", {})
    return summary if isinstance(summary, dict) else {}


def _memory_breakdown_component(component: Any) -> str:
    component_str = str(component or "").strip()
    if component_str in {"", "unknown_saved_activation"}:
        return "other_saved_activations"
    return component_str


def _memory_breakdown_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return []
    peak_allocated = int(summary.get("peak_allocated_hbm_bytes", 0) or 0)
    peak_reserved = int(summary.get("peak_reserved_hbm_bytes", 0) or 0)
    actual_peak_allocated = int(summary.get("actual_peak_allocated_hbm_bytes", peak_allocated) or 0)
    actual_peak_reserved = int(summary.get("actual_peak_reserved_hbm_bytes", peak_reserved) or 0)
    reserved_unallocated = int(summary.get("reserved_unallocated_bytes", 0) or 0)
    external_cuda = int(summary.get("external_cuda_or_driver_bytes", 0) or 0)
    selected_step = summary.get("selected_step", "")
    selected_phase = summary.get("selected_phase", "")
    allocated_closure_error = int(summary.get("allocated_closure_error_bytes", 0) or 0)
    reserved_closure_error = int(summary.get("reserved_closure_error_bytes", 0) or 0)
    allocated_closure_ok = bool(summary.get("allocated_closure_ok", False))
    reserved_closure_ok = bool(summary.get("reserved_closure_ok", False))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        memory_space = row.get("memory_space", "-")
        is_reserved_stack_row = memory_space == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
        normalized.append(
            {
                "selected_step": selected_step,
                "selected_phase": selected_phase,
                "actual_peak_step": summary.get("actual_peak_step", ""),
                "actual_peak_phase": summary.get("actual_peak_phase", ""),
                "schema_version": summary.get("schema_version", ""),
                "selected_metric": summary.get("selected_metric", ""),
                "actual_peak_allocated_hbm_bytes": actual_peak_allocated,
                "actual_peak_reserved_hbm_bytes": actual_peak_reserved,
                "peak_allocated_hbm_bytes": peak_allocated,
                "peak_reserved_hbm_bytes": peak_reserved,
                "reserved_unallocated_bytes": reserved_unallocated,
                "external_cuda_or_driver_bytes": external_cuda,
                "allocated_stack_sum_bytes": int(summary.get("allocated_stack_sum_bytes", 0) or 0),
                "reserved_stack_sum_bytes": int(summary.get("reserved_stack_sum_bytes", 0) or 0),
                "saved_activation_hbm_bytes_at_peak": int(
                    summary.get("saved_activation_hbm_bytes_at_peak", 0) or 0
                ),
                "live_activation_hbm_bytes_at_peak": int(summary.get("live_activation_hbm_bytes_at_peak", 0) or 0),
                "activation_hbm_bytes_at_peak": int(summary.get("activation_hbm_bytes_at_peak", 0) or 0),
                "temporary_workspace_hbm_bytes_at_peak": int(
                    summary.get("temporary_workspace_hbm_bytes_at_peak", 0) or 0
                ),
                "unattributed_allocated_peak_bytes": int(
                    summary.get("unattributed_allocated_peak_bytes", 0) or 0
                ),
                "memory_space": memory_space,
                "group": row.get("group", "-"),
                "component": _memory_breakdown_component(row.get("component", "-")),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "mib": value / (1024.0 ** 2),
                "percent_peak_reserved_hbm": (value * 100.0 / peak_reserved)
                if peak_reserved > 0 and is_reserved_stack_row
                else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "allocated_closure_ok": allocated_closure_ok,
                "reserved_closure_ok": reserved_closure_ok,
                "allocated_closure_error_bytes": allocated_closure_error,
                "reserved_closure_error_bytes": reserved_closure_error,
            }
        )
    return normalized


def _actual_peak_memory_breakdown_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("actual_peak_breakdown_rows", [])
    if not isinstance(rows, list) or not rows:
        rows = summary.get("breakdown_rows", [])
    if not isinstance(rows, list):
        return []
    actual_peak_allocated = int(summary.get("actual_peak_allocated_hbm_bytes", summary.get("peak_allocated_hbm_bytes", 0)) or 0)
    actual_peak_reserved = int(summary.get("actual_peak_reserved_hbm_bytes", summary.get("peak_reserved_hbm_bytes", 0)) or 0)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = int(row.get("bytes", 0) or 0)
        memory_space = row.get("memory_space", "-")
        is_reserved_stack_row = memory_space == "GPU HBM" or row.get("component") == "allocator_reserved_unallocated"
        normalized.append(
            {
                "actual_peak_step": summary.get("actual_peak_step", ""),
                "actual_peak_phase": summary.get("actual_peak_phase", ""),
                "schema_version": summary.get("schema_version", ""),
                "actual_peak_allocated_hbm_bytes": actual_peak_allocated,
                "actual_peak_reserved_hbm_bytes": actual_peak_reserved,
                "actual_peak_reserved_unallocated_bytes": int(
                    summary.get("actual_peak_reserved_unallocated_bytes", max(0, actual_peak_reserved - actual_peak_allocated)) or 0
                ),
                "memory_space": memory_space,
                "group": row.get("group", "-"),
                "component": _memory_breakdown_component(row.get("component", "-")),
                "kind": row.get("kind", "-"),
                "bytes": value,
                "mib": value / (1024.0 ** 2),
                "percent_actual_peak_reserved_hbm": (value * 100.0 / actual_peak_reserved)
                if actual_peak_reserved > 0 and is_reserved_stack_row
                else "",
                "method": row.get("method", "-"),
                "accuracy": row.get("accuracy", "-"),
                "actual_peak_allocated_closure_error_bytes": int(
                    summary.get("actual_peak_allocated_closure_error_bytes", 0) or 0
                ),
                "actual_peak_reserved_closure_error_bytes": int(
                    summary.get("actual_peak_reserved_closure_error_bytes", 0) or 0
                ),
            }
        )
    return normalized


def _source_memory_breakdown_markdown(profile: dict[str, Any], *, top_level: bool = False) -> str:
    summary = _memory_breakdown_summary(profile)
    rows = _memory_breakdown_csv_rows(summary)
    if not rows:
        return ""
    peak_allocated = int(summary.get("peak_allocated_hbm_bytes", 0) or 0)
    peak_reserved = int(summary.get("peak_reserved_hbm_bytes", 0) or 0)
    actual_peak_allocated = int(summary.get("actual_peak_allocated_hbm_bytes", peak_allocated) or 0)
    actual_peak_reserved = int(summary.get("actual_peak_reserved_hbm_bytes", peak_reserved) or 0)
    reserved_unallocated = int(summary.get("reserved_unallocated_bytes", 0) or 0)
    external_cuda = int(summary.get("external_cuda_or_driver_bytes", 0) or 0)
    allocated_closure_error = int(summary.get("allocated_closure_error_bytes", 0) or 0)
    reserved_closure_error = int(summary.get("reserved_closure_error_bytes", 0) or 0)
    title = "# LF LoRA-SFT Source Memory Breakdown" if top_level else "## Source Memory Breakdown"
    lines = [
        title,
        "",
        f"Schema version: `{summary.get('schema_version', '-')}`  ",
        f"Selected metric: `{summary.get('selected_metric', '-')}`  ",
        f"Selected step: `{summary.get('selected_step', '-')}`  ",
        f"Selected phase: `{summary.get('selected_phase', '-')}`  ",
        f"Selected-attribution peak allocated HBM: `{_fmt_mib(peak_allocated)} MiB`  ",
        f"Selected-attribution peak reserved HBM: `{_fmt_mib(peak_reserved)} MiB`  ",
        f"Actual peak step: `{summary.get('actual_peak_step', '-')}`  ",
        f"Actual peak phase: `{summary.get('actual_peak_phase', '-')}`  ",
        f"Actual peak allocated HBM: `{_fmt_mib(actual_peak_allocated)} MiB`  ",
        f"Actual peak reserved HBM: `{_fmt_mib(actual_peak_reserved)} MiB`  ",
        f"Reserved but unallocated: `{_fmt_mib(reserved_unallocated)} MiB`  ",
        f"External CUDA/driver diagnostic: `{_fmt_mib(external_cuda)} MiB`  ",
        f"Allocated stack sum: `{_fmt_mib(summary.get('allocated_stack_sum_bytes'))} MiB`  ",
        f"Reserved stack sum: `{_fmt_mib(summary.get('reserved_stack_sum_bytes'))} MiB`  ",
        f"Activations at peak: `{_fmt_mib(summary.get('activation_hbm_bytes_at_peak', summary.get('saved_activation_hbm_bytes_at_peak')))} MiB`  ",
        f"Saved-for-backward activations at peak: `{_fmt_mib(summary.get('saved_activation_hbm_bytes_at_peak'))} MiB`  ",
        f"Live-output activations at peak: `{_fmt_mib(summary.get('live_activation_hbm_bytes_at_peak'))} MiB`  ",
        f"Temporary/workspace at peak: `{_fmt_mib(summary.get('temporary_workspace_hbm_bytes_at_peak'))} MiB`  ",
        f"Unattributed allocated peak: `{_fmt_mib(summary.get('unattributed_allocated_peak_bytes'))} MiB`  ",
        f"Allocated closure error: `{_fmt_mib(allocated_closure_error)} MiB`  ",
        f"Allocated closure OK: `{bool(summary.get('allocated_closure_ok', False))}`  ",
        f"Reserved closure error: `{_fmt_mib(reserved_closure_error)} MiB`  ",
        f"Reserved closure OK: `{bool(summary.get('reserved_closure_ok', False))}`",
        "",
        "GPU HBM semantic rows below are the selected-attribution rows. Actual peak rows are emitted separately in memory_actual_peak_breakdown.csv. The allocator reserved-unallocated row extends allocated HBM to reserved HBM. External CUDA/driver diagnostics are separate and excluded from both closures.",
        "",
        "| Group | Component | Kind | Memory space | MiB | % peak reserved HBM | Method | Accuracy |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["memory_space"]) not in {"GPU HBM", "GPU reserved"},
            str(item["memory_space"]) == "GPU reserved",
            -float(item["bytes"]),
        ),
    ):
        value = int(row["bytes"])
        is_reserved_stack_row = (
            row["memory_space"] == "GPU HBM" or row["component"] == "allocator_reserved_unallocated"
        )
        lines.append(
            "| {group} | {component} | {kind} | {memory_space} | {mib} | {pct} | {method} | {accuracy} |".format(
                group=row["group"],
                component=row["component"],
                kind=row["kind"],
                memory_space=row["memory_space"],
                mib=_fmt_mib(value),
                pct=_fmt_pct(value, peak_reserved) if is_reserved_stack_row else "-",
                method=row["method"],
                accuracy=row["accuracy"],
            )
        )
    notes = summary.get("notes", [])
    if isinstance(notes, list) and notes:
        lines += ["", "Notes:"]
        for note in notes:
            lines.append(f"- {note}")
    snapshot = profile.get("memory_snapshot", {})
    if isinstance(snapshot, dict) and snapshot.get("enabled"):
        lines += [
            "",
            "Snapshot validation:",
            f"- dumped: `{bool(snapshot.get('dumped', False))}`",
            f"- path: `{snapshot.get('path', '')}`",
        ]
        if snapshot.get("error"):
            lines.append(f"- error: `{snapshot.get('error')}`")
    lines.append("")
    return "\n".join(lines)


def _source_memory_markdown(profile: dict[str, Any]) -> str:
    memory = profile.get("memory", {})
    gpu = memory.get("gpu", {}) if isinstance(memory, dict) else {}
    memory_attribution = profile.get("memory_attribution", {})
    category_rows = memory_attribution.get("rows", []) if isinstance(memory_attribution, dict) else []
    saved_tensors = memory_attribution.get("saved_tensors", {}) if isinstance(memory_attribution, dict) else {}
    saved_rows = saved_tensors.get("by_owner", []) if isinstance(saved_tensors, dict) else []
    lines = [
        "# LF LoRA-SFT Memory",
        "",
        "| Metric | MiB |",
        "|---|---:|",
        f"| peak_allocated_hbm_bytes | {_fmt_mib(gpu.get('peak_allocated_hbm_bytes'))} |",
        f"| peak_reserved_hbm_bytes | {_fmt_mib(gpu.get('peak_reserved_hbm_bytes'))} |",
        f"| reserved_unallocated_bytes | {_fmt_mib(gpu.get('reserved_unallocated_bytes'))} |",
        "",
    ]
    breakdown_markdown = _source_memory_breakdown_markdown(profile)
    if breakdown_markdown:
        lines.append(breakdown_markdown)
    if isinstance(category_rows, list) and category_rows:
        lines += [
            "## Persistent Tensor Accounting",
            "",
            "These rows are exact tensor-size accounting for parameters, buffers, gradients, and host/pinned tensors. They are not a full peak allocated HBM attribution by themselves.",
            "",
            "| Category | Component | Device | MiB |",
            "|---|---|---|---:|",
        ]
        sorted_categories = sorted(
            (row for row in category_rows if isinstance(row, dict)),
            key=lambda row: float(row.get("bytes", 0) or 0),
            reverse=True,
        )
        for row in sorted_categories[:12]:
            lines.append(
                "| {category} | {component} | {device} | {mib} |".format(
                    category=row.get("category", "-"),
                    component=row.get("component", "-"),
                    device=row.get("device", "-"),
                    mib=_fmt_mib(row.get("bytes")),
                )
            )
        lines.append("")
    if isinstance(saved_tensors, dict) and saved_tensors.get("enabled"):
        lines += [
            "## Saved Activation Owners",
            "",
            "| Owner | MiB |",
            "|---|---:|",
        ]
        sorted_saved = sorted(
            (row for row in saved_rows if isinstance(row, dict)),
            key=lambda row: float(row.get("bytes", 0) or 0),
            reverse=True,
        )
        for row in sorted_saved[:20]:
            lines.append(f"| {row.get('owner', '-')} | {_fmt_mib(row.get('bytes'))} |")
        lines += [
            "",
            f"Unique saved tensors: `{saved_tensors.get('unique_tensors', '-')}`  ",
            f"Unique saved tensor bytes: `{_fmt_mib(saved_tensors.get('total_unique_bytes'))} MiB`  ",
            f"Reference-counted saved tensor bytes: `{_fmt_mib(saved_tensors.get('total_reference_bytes'))} MiB`",
            "",
        ]
    elif isinstance(memory_attribution, dict) and not memory_attribution.get("enabled", False):
        lines += ["Saved activation attribution was disabled for this run.", ""]
    return "\n".join(lines)


def _write_step_samples(profile: dict[str, Any], output_dir: Path) -> None:
    step_samples = profile.get("step_samples", {})
    rows = step_samples.get("rows", []) if isinstance(step_samples, dict) else []
    if not isinstance(rows, list) or not rows:
        return
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return
    fieldnames = list(normalized_rows[0].keys())
    for row in normalized_rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    (output_dir / "step_samples.json").write_text(
        json.dumps(normalized_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "step_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _augment_step_sample_rows(
    profile: dict[str, Any], heartbeat_rows: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    step_samples = profile.get("step_samples", {})
    rows = step_samples.get("rows", []) if isinstance(step_samples, dict) else []
    if not isinstance(rows, list):
        return []

    heartbeat_by_step = _heartbeat_step_intervals_by_step(profile, heartbeat_rows)
    elapsed_by_step = _trainer_elapsed_by_step(profile)
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        raw_step = _int_value(row.get("raw_step", row.get("step")))

        forward_ms = _float_value(row.get("forward_milliseconds"))
        backward_ms = _float_value(row.get("backward_milliseconds"))
        forward_backward_ms = _float_value(row.get("forward_backward_milliseconds"))
        if forward_backward_ms is None and (forward_ms is not None or backward_ms is not None):
            forward_backward_ms = (forward_ms or 0.0) + (backward_ms or 0.0)
        if forward_backward_ms is None and row.get("step_milliseconds_source") in (None, "", "forward_plus_backward_only"):
            forward_backward_ms = _float_value(row.get("step_milliseconds"))
        if forward_backward_ms is not None:
            row["forward_backward_milliseconds"] = forward_backward_ms

        profiled_training_step_ms = _float_value(row.get("profiled_training_step_milliseconds"))
        if profiled_training_step_ms is None:
            profiled_training_step_ms = _float_value(row.get("training_step_milliseconds"))
        if profiled_training_step_ms is not None:
            row["profiled_training_step_milliseconds"] = profiled_training_step_ms

        heartbeat_row = heartbeat_by_step.get(raw_step) if raw_step is not None else None
        elapsed_seconds = elapsed_by_step.get(raw_step) if raw_step is not None else None
        previous_elapsed_seconds = None
        if raw_step is not None:
            previous_elapsed_seconds = 0.0 if raw_step == 1 else elapsed_by_step.get(raw_step - 1)
        if heartbeat_row is not None:
            trainer_step_ms = _float_value(heartbeat_row.get("trainer_e2e_step_milliseconds"))
            if trainer_step_ms is not None:
                row.update(heartbeat_row)
                if forward_backward_ms is not None:
                    row["optimizer_update_side_milliseconds"] = trainer_step_ms - forward_backward_ms
                row["step_milliseconds"] = trainer_step_ms
                row["step_milliseconds_source"] = "heartbeat_dataloader_interval"
        elif elapsed_seconds is not None and previous_elapsed_seconds is not None:
            trainer_step_ms = max(0.0, elapsed_seconds - previous_elapsed_seconds) * 1000.0
            row["trainer_elapsed_seconds"] = elapsed_seconds
            row["trainer_e2e_step_milliseconds"] = trainer_step_ms
            if forward_backward_ms is not None:
                row["optimizer_update_side_milliseconds"] = trainer_step_ms - forward_backward_ms
            row["step_milliseconds"] = trainer_step_ms
            row["step_milliseconds_source"] = "trainer_log_elapsed_time"
        else:
            if row.get("step_milliseconds_source") != "trainer_log_elapsed_time":
                if forward_backward_ms is not None:
                    row["step_milliseconds"] = forward_backward_ms
                row["step_milliseconds_source"] = "forward_plus_backward_only"
        normalized_rows.append(row)
    return normalized_rows


def _install_augmented_step_samples(profile: dict[str, Any]) -> None:
    heartbeat_records = _heartbeat_records_for_timing(profile)
    heartbeat_rows = _heartbeat_step_interval_rows(profile, heartbeat_records)
    heartbeat_timing = _heartbeat_timing(profile, heartbeat_rows)
    if heartbeat_timing.get("available"):
        trainer = profile.get("trainer", {})
        if not isinstance(trainer, dict):
            trainer = {}
        profile["trainer"] = {**trainer, "timing": heartbeat_timing}
    rows = _augment_step_sample_rows(profile, heartbeat_rows)
    if not rows:
        return
    step_samples = profile.get("step_samples", {})
    if not isinstance(step_samples, dict):
        step_samples = {}
    profile["step_samples"] = {**step_samples, "rows": rows}


def _stage_value(stage: dict[str, Any], name: str) -> float:
    for row in _as_rows(stage.get("stage_breakdown", {})):
        if row.get("name") == name:
            try:
                return float(row.get("milliseconds", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _host_api_value(stage: dict[str, Any], name: str) -> float:
    for row in _as_rows(stage.get("host_api_breakdown", {})):
        if row.get("name") == name:
            try:
                return float(row.get("milliseconds", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _layer(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return match.group(1) if match else ""


def _module(name: str) -> str:
    if ".self_attn" in name or ".attention" in name:
        return "attention"
    if ".mlp.experts" in name or ".experts" in name:
        return "experts"
    if ".mlp" in name:
        return "mlp"
    if "lora" in name.lower():
        return "lora"
    if "optimizer" in name.lower():
        return "optimizer"
    if "router" in name.lower():
        return "router"
    return "other"


def _profile_memory(profile: dict[str, Any]) -> dict[str, Any]:
    for container in (profile, profile.get("source_profile", {}), profile.get("memory_profile", {})):
        if isinstance(container, dict) and isinstance(container.get("memory_attribution"), dict):
            return container["memory_attribution"]
    return {}


def _timing_by_stage(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(profile.get("stages"), list):
        for stage in profile["stages"]:
            if not isinstance(stage, dict):
                continue
            rows.append(
                {
                    "stage": stage.get("stage", ""),
                    "total_milliseconds": stage.get("total_milliseconds", 0.0),
                    "cuda_kernel_busy_milliseconds": _stage_value(stage, "cuda_kernel_busy_union"),
                    "cuda_memcpy_milliseconds": _stage_value(stage, "cuda_memcpy_union"),
                    "gpu_no_kernel_milliseconds": _stage_value(stage, "gpu_no_kernel_time"),
                    "cuda_runtime_api_milliseconds": _host_api_value(stage, "cuda_runtime_api_sum_overlaps_gpu_timeline"),
                    "cuda_sync_api_milliseconds": _host_api_value(stage, "cuda_synchronization_api_sum_overlaps_gpu_timeline"),
                }
            )
        return rows
    for row in _as_rows(profile.get("step", {})):
        rows.append(
            {
                "stage": row.get("name", ""),
                "host_milliseconds": row.get("milliseconds", 0.0),
                "samples": row.get("samples", ""),
            }
        )
    return rows


def _timing_by_op(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("operation_kernel_time", {})):
            name = str(row.get("name", ""))
            rows.append(
                {
                    "stage": stage_name,
                    "operation": name,
                    "layer": _layer(name),
                    "module": _module(name),
                    "gpu_kernel_milliseconds": row.get("milliseconds", 0.0),
                    "percent_of_stage": row.get("percent", 0.0),
                }
            )
    return rows


def _timing_by_layer(op_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], float] = {}
    for row in op_rows:
        layer = str(row.get("layer", ""))
        if not layer:
            continue
        key = (str(row.get("stage", "")), layer)
        totals[key] = totals.get(key, 0.0) + float(row.get("gpu_kernel_milliseconds") or 0.0)
    return [
        {"stage": stage, "layer": layer, "gpu_kernel_milliseconds": value}
        for (stage, layer), value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _timing_by_module(op_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], float] = {}
    for row in op_rows:
        key = (str(row.get("stage", "")), str(row.get("module", "other")))
        totals[key] = totals.get(key, 0.0) + float(row.get("gpu_kernel_milliseconds") or 0.0)
    return [
        {"stage": stage, "module": module, "gpu_kernel_milliseconds": value}
        for (stage, module), value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _kernel_by_op(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("operation_kernel_classes", {})):
            op = str(row.get("operation", ""))
            rows.append(
                {
                    "stage": stage_name,
                    "operation": op,
                    "layer": _layer(op),
                    "module": _module(op),
                    "kernel_class": row.get("kernel_class", ""),
                    "milliseconds": row.get("milliseconds", 0.0),
                    "percent_of_stage": row.get("percent", 0.0),
                }
            )
    return rows


def _unattributed(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage", "")
        for row in _as_rows(stage.get("gpu_no_kernel_gap_attribution", {})):
            name = str(row.get("name", ""))
            if "unattributed" in name.lower() or "no-kernel" in name.lower():
                rows.append(
                    {
                        "stage": stage_name,
                        "name": name,
                        "milliseconds": row.get("milliseconds", 0.0),
                        "percent_of_stage": row.get("percent", 0.0),
                    }
                )
        for row in _as_rows(stage.get("operation_kernel_time", {})):
            name = str(row.get("name", ""))
            if "unattributed" in name.lower():
                rows.append(
                    {
                        "stage": stage_name,
                        "name": name,
                        "milliseconds": row.get("milliseconds", 0.0),
                        "percent_of_stage": row.get("percent", 0.0),
                    }
                )
    return rows


def _memory_by_category(profile: dict[str, Any]) -> list[dict[str, Any]]:
    memory = _profile_memory(profile)
    return _as_rows(memory.get("rows", []))


def _memory_by_module(profile: dict[str, Any]) -> list[dict[str, Any]]:
    memory = _profile_memory(profile)
    saved = memory.get("saved_tensors", {}) if isinstance(memory, dict) else {}
    rows = _as_rows(saved.get("by_owner", [])) if isinstance(saved, dict) else []
    if rows:
        return rows
    return _as_rows(profile.get("stage_memory", {}))


def _write_source_artifacts(source_profile_json: Path, output_dir: Path, profile_json: Path | None) -> None:
    profile = json.loads(source_profile_json.read_text(encoding="utf-8"))
    profile["trainable_surface"] = _trainable_surface_summary(profile)
    kt_lora_health = _kt_lora_update_health(profile)
    if kt_lora_health and isinstance(profile.get("optimizer_memory"), dict):
        profile["optimizer_memory"]["kt_lora_update_health"] = kt_lora_health
    _install_augmented_step_samples(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_profile = profile_json or output_dir / "profile.json"
    target_profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = _source_summary_markdown(profile)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "table.md").write_text(summary, encoding="utf-8")
    (output_dir / "lat.md").write_text(_source_latency_markdown(profile), encoding="utf-8")
    (output_dir / "memory.md").write_text(_source_memory_markdown(profile), encoding="utf-8")
    breakdown = _source_memory_breakdown_markdown(profile, top_level=True)
    memory_breakdown_summary = _memory_breakdown_summary(profile)
    breakdown_rows = _memory_breakdown_csv_rows(memory_breakdown_summary)
    actual_breakdown_rows = _actual_peak_memory_breakdown_csv_rows(memory_breakdown_summary)
    if breakdown and breakdown_rows:
        (output_dir / "memory_breakdown.md").write_text(breakdown, encoding="utf-8")
        _write_csv(output_dir / "memory_breakdown.csv", breakdown_rows)
    if actual_breakdown_rows:
        _write_csv(output_dir / "memory_actual_peak_breakdown.csv", actual_breakdown_rows)
    process_memory_rows = _process_memory_rows(profile)
    if process_memory_rows:
        _write_csv(output_dir / "process_memory.csv", process_memory_rows)
    kt_rows = _kt_counter_rows(profile)
    if kt_rows:
        _write_csv(output_dir / "kt_counters.csv", kt_rows)
    lora_rows = _lora_counter_rows(profile)
    if lora_rows:
        _write_csv(output_dir / "lora_counters.csv", lora_rows)
    cpuadam_rows = _cpuadam_rows(profile)
    if cpuadam_rows:
        _write_csv(output_dir / "cpuadam.csv", cpuadam_rows)
    asym_cpu_adamw_rows = _asym_cpu_adamw_rows(profile)
    if asym_cpu_adamw_rows:
        _write_csv(output_dir / "asym_cpu_adamw.csv", asym_cpu_adamw_rows)
    trainable_surface_rows = _trainable_surface_rows(profile)
    if trainable_surface_rows:
        _write_csv(output_dir / "trainable_surface.csv", trainable_surface_rows)
    grad_clip_rows = _grad_clip_rows(profile)
    if grad_clip_rows:
        _write_csv(output_dir / "grad_clip.csv", grad_clip_rows)
    kt_lora_health_rows = _kt_lora_update_health_rows(profile)
    if kt_lora_health_rows:
        _write_csv(output_dir / "kt_lora_update_health.csv", kt_lora_health_rows)
    optimizer_preflight_rows = _optimizer_memory_preflight_rows(profile)
    if optimizer_preflight_rows:
        _write_csv(output_dir / "optimizer_memory_preflight.csv", optimizer_preflight_rows)
    _write_step_samples(profile, output_dir)


def _write_profile_csv_artifacts(profile_json: Path, output_dir: Path) -> None:
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    source_profile = _source_profile_for_artifacts(profile)
    source_profile.setdefault("trainable_surface", _trainable_surface_summary(source_profile))
    kt_lora_health = _kt_lora_update_health(source_profile)
    if kt_lora_health and isinstance(source_profile.get("optimizer_memory"), dict):
        source_profile["optimizer_memory"]["kt_lora_update_health"] = kt_lora_health
    _install_augmented_step_samples(source_profile)
    profile_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = _timing_by_stage(profile)
    op_rows = _timing_by_op(profile)
    _write_csv(output_dir / "timing_by_stage.csv", stage_rows)
    _write_csv(output_dir / "timing_by_op.csv", op_rows)
    _write_csv(output_dir / "timing_by_layer.csv", _timing_by_layer(op_rows))
    _write_csv(output_dir / "timing_by_module.csv", _timing_by_module(op_rows))
    _write_csv(output_dir / "kernel_by_op.csv", _kernel_by_op(profile))
    _write_csv(output_dir / "memory_by_category.csv", _memory_by_category(profile))
    _write_csv(output_dir / "memory_by_module.csv", _memory_by_module(profile))
    memory_breakdown_summary = _memory_breakdown_summary(profile)
    if not _memory_breakdown_csv_rows(memory_breakdown_summary):
        memory_breakdown_summary = _memory_breakdown_summary(source_profile)
    breakdown_rows = _memory_breakdown_csv_rows(memory_breakdown_summary)
    if breakdown_rows:
        _write_csv(output_dir / "memory_breakdown.csv", breakdown_rows)
    actual_breakdown_rows = _actual_peak_memory_breakdown_csv_rows(memory_breakdown_summary)
    if actual_breakdown_rows:
        _write_csv(output_dir / "memory_actual_peak_breakdown.csv", actual_breakdown_rows)
    process_memory_rows = _process_memory_rows(source_profile)
    if process_memory_rows:
        _write_csv(output_dir / "process_memory.csv", process_memory_rows)
    kt_rows = _kt_counter_rows(source_profile)
    if kt_rows:
        _write_csv(output_dir / "kt_counters.csv", kt_rows)
    lora_rows = _lora_counter_rows(source_profile)
    if lora_rows:
        _write_csv(output_dir / "lora_counters.csv", lora_rows)
    cpuadam_rows = _cpuadam_rows(source_profile)
    if cpuadam_rows:
        _write_csv(output_dir / "cpuadam.csv", cpuadam_rows)
    asym_cpu_adamw_rows = _asym_cpu_adamw_rows(source_profile)
    if asym_cpu_adamw_rows:
        _write_csv(output_dir / "asym_cpu_adamw.csv", asym_cpu_adamw_rows)
    trainable_surface_rows = _trainable_surface_rows(source_profile)
    if trainable_surface_rows:
        _write_csv(output_dir / "trainable_surface.csv", trainable_surface_rows)
    grad_clip_rows = _grad_clip_rows(source_profile)
    if grad_clip_rows:
        _write_csv(output_dir / "grad_clip.csv", grad_clip_rows)
    kt_lora_health_rows = _kt_lora_update_health_rows(source_profile)
    if kt_lora_health_rows:
        _write_csv(output_dir / "kt_lora_update_health.csv", kt_lora_health_rows)
    optimizer_preflight_rows = _optimizer_memory_preflight_rows(source_profile)
    if optimizer_preflight_rows:
        _write_csv(output_dir / "optimizer_memory_preflight.csv", optimizer_preflight_rows)
    _write_step_samples(source_profile, output_dir)
    _write_csv(output_dir / "unattributed_timing.csv", _unattributed(profile))


def main() -> None:
    args = _parse_args()
    if args.source_profile_json:
        _write_source_artifacts(args.source_profile_json, args.output_dir, args.profile_json)
    if args.profile_json:
        _write_profile_csv_artifacts(args.profile_json, args.output_dir)


if __name__ == "__main__":
    main()
