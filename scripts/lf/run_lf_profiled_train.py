#!/usr/bin/env python
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import torch

from asym_gemm.profiling.lf_trace import LFTraceConfig, LFTraceHandle, install_lf_trace, uninstall_lf_trace
from asym_gemm.training.moe import parse_expert_recompute_policy_spec
from asym_gemm.training.profile_ranges import prof_range, set_profile_enabled


PROFILE_SOURCE_JSON_ENV = "ASYM_GEMM_LF_PROFILE_SOURCE_JSON"
PROFILE_MEMORY_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY"
PROFILE_LEVEL_ENV = "ASYM_GEMM_LF_PROFILE_LEVEL"
PROFILE_LAYERS_ENV = "ASYM_GEMM_LF_PROFILE_LAYERS"
PROFILE_MEMORY_ATTRIBUTION_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION"
PROFILE_MEMORY_BREAKDOWN_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN"
PROFILE_MEMORY_BREAKDOWN_OUTPUT_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_OUTPUT"
PROFILE_MEMORY_SNAPSHOT_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT"
PROFILE_MEMORY_SNAPSHOT_PATH_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_PATH"
PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES"
PROFILE_EXTERNAL_MEMORY_ENV = "ASYM_GEMM_LF_PROFILE_EXTERNAL_MEMORY"
PROFILE_SYNC_ENV = "ASYM_GEMM_LF_PROFILE_SYNC"
PROFILE_MODULE_FILTER_ENV = "ASYM_GEMM_LF_PROFILE_MODULE_FILTER"
PROFILE_HEARTBEAT_ENV = "ASYM_GEMM_LF_HEARTBEAT_JSON"
PROFILE_PARTIAL_INTERVAL_ENV = "ASYM_GEMM_LF_PROFILE_PARTIAL_INTERVAL_SECONDS"
CONFIG_ENV_PREFIX = "ASYM_GEMM_LF_CONFIG_"
KT_LORA_HEALTH_MAX_TENSORS_ENV = "ASYM_GEMM_LF_KT_LORA_HEALTH_MAX_TENSORS"
KT_LORA_HEALTH_MAX_ELEMENTS_ENV = "ASYM_GEMM_LF_KT_LORA_HEALTH_MAX_ELEMENTS"

_LAST_LF_MODEL: Any | None = None
_DEEPSPEED_RUNTIME_MARKER: dict[str, Any] = {}
_OPTIMIZER_MEMORY_MARKER: dict[str, Any] = {}
_GRAD_CLIP_MARKER: dict[str, Any] = {}
_MODEL_CAPTURE_HEARTBEAT: Any | None = None
_MODEL_CAPTURE_PARTIAL_WRITER: Any | None = None
KT_REQUIRE_STARTUP_ENV = "ASYM_GEMM_LF_REQUIRE_KT_STARTUP"
KT_REQUIRE_FUSED_LORA_STARTUP_ENV = "ASYM_GEMM_LF_REQUIRE_KT_FUSED_LORA_STARTUP"


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _value_enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _source_profile_notes(config: dict[str, Any]) -> list[str]:
    if _value_enabled(config.get("profile_sync"), default=True):
        timing_note = (
            "LF source timings are host wall-clock ranges with torch.cuda.synchronize() "
            "before and after each profiled range."
        )
    else:
        timing_note = "LF source timings are host wall-clock ranges without per-range CUDA synchronization."
    return [
        timing_note,
        "Use heartbeat e2e timing for full training-step wall time including optimizer-side and trainer overhead.",
        "Use Nsight Systems postprocessed profile.json for CUDA-kernel attribution, not as the sole source of source-stage wall timing.",
    ]


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _is_rank0() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _process_memory_snapshot() -> dict[str, Any]:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return {"available": False, "reason": "/proc/self/status unavailable"}
    keys = {
        "VmRSS": "rss_bytes",
        "VmHWM": "rss_peak_bytes",
        "VmSize": "virtual_memory_bytes",
        "VmData": "data_bytes",
        "VmStk": "stack_bytes",
    }
    snapshot: dict[str, Any] = {"available": True, "source": str(status_path)}
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            label, _, raw_value = line.partition(":")
            key = keys.get(label)
            if key is None:
                continue
            parts = raw_value.strip().split()
            if not parts:
                continue
            value = _safe_int(parts[0])
            if value is not None:
                snapshot[key] = value * 1024
    except OSError as exc:
        return {"available": False, "reason": repr(exc)}
    return snapshot


def _process_memory_value(snapshot: dict[str, Any], key: str) -> int:
    value = snapshot.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


class LFHeartbeat:
    def __init__(self, path: str | None, config: dict[str, Any]):
        self.path = Path(path) if path and _is_rank0() else None
        self.latest_path = self.path.with_name(f"{self.path.stem}.latest.json") if self.path is not None else None
        self.started = time.perf_counter()
        self.config = config
        self.counts: dict[str, int] = {}
        self.latest: dict[str, Any] | None = None

    def emit(self, stage: str, **fields: Any) -> None:
        if self.path is None:
            return
        count = self.counts.get(stage, 0) + 1
        self.counts[stage] = count
        record = {
            "timestamp": _utc_timestamp(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            "pid": os.getpid(),
            "stage": stage,
            "count": count,
            "backend": self.config.get("backend"),
            "kt_backend": self.config.get("kt_backend"),
            **fields,
        }
        self.latest = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.latest_path is not None:
            _atomic_write_json(self.latest_path, record)


class PartialProfileWriter:
    def __init__(
        self,
        source_json: str | None,
        recorder: "LFProfileRecorder",
        trace_handle: LFTraceHandle,
        heartbeat: LFHeartbeat,
    ):
        self.path = Path(source_json).with_name("source_profile.partial.json") if source_json and _is_rank0() else None
        self.recorder = recorder
        self.trace_handle = trace_handle
        self.heartbeat = heartbeat
        self.last_write = 0.0
        self.interval_seconds = max(_safe_float(os.environ.get(PROFILE_PARTIAL_INTERVAL_ENV)) or 5.0, 0.0)

    def write(self, reason: str, *, force: bool = False, extra: dict[str, Any] | None = None) -> None:
        if self.path is None:
            return
        now = time.perf_counter()
        if not force and now - self.last_write < self.interval_seconds:
            return
        self.last_write = now
        try:
            report = self.recorder.report(self.trace_handle)
        except Exception as exc:
            report = {"partial_report_error": repr(exc)}
        report["partial"] = True
        report["partial_reason"] = reason
        report["partial_timestamp"] = _utc_timestamp()
        report["heartbeat_latest"] = self.heartbeat.latest
        if _GRAD_CLIP_MARKER:
            report["grad_clip"] = _GRAD_CLIP_MARKER
        if extra:
            report["partial_extra"] = extra
        _atomic_write_json(self.path, report)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _lora_surface_sidecar() -> dict[str, Any] | None:
    explicit_path = os.environ.get("ASYM_GEMM_LF_LORA_SURFACE_JSON", "").strip()
    source_json = os.environ.get(PROFILE_SOURCE_JSON_ENV, "").strip()
    if explicit_path:
        path = Path(explicit_path)
    elif source_json:
        path = Path(source_json).with_name("lora_surface.json")
    else:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload.get("available") is True else None


def _kt_lora_health_max_tensors() -> int:
    raw = os.environ.get(KT_LORA_HEALTH_MAX_TENSORS_ENV, "12").strip().lower()
    if raw in {"", "default"}:
        return 12
    if raw in {"all", "full", "exhaustive", "0"}:
        return 0
    value = _safe_int(raw)
    if value is None:
        raise ValueError(
            f"{KT_LORA_HEALTH_MAX_TENSORS_ENV} must be a positive integer, 0, or 'all', got {raw!r}"
        )
    if value < 0:
        raise ValueError(
            f"{KT_LORA_HEALTH_MAX_TENSORS_ENV} must be a positive integer, 0, or 'all', got {raw!r}"
        )
    return value


def _kt_lora_health_max_elements() -> int:
    raw = os.environ.get(KT_LORA_HEALTH_MAX_ELEMENTS_ENV, "4096").strip().lower()
    if raw in {"", "default"}:
        return 4096
    if raw in {"all", "full", "exhaustive", "0"}:
        return 0
    value = _safe_int(raw)
    if value is None or value < 0:
        raise ValueError(
            f"{KT_LORA_HEALTH_MAX_ELEMENTS_ENV} must be a positive integer, 0, or 'all', got {raw!r}"
        )
    return value


def _trainer_log_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _heartbeat_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


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
            return hours * 3600.0 + minutes * 60.0 + seconds
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        seconds = float(text)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def _heartbeat_step_interval_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        elapsed_seconds = _safe_float(record.get("elapsed_seconds"))
        if elapsed_seconds is None:
            continue
        stage = str(record.get("stage") or "")
        if stage == "trainer_end":
            trainer_end = max(elapsed_seconds, trainer_end or 0.0)
            continue
        stage_map = stage_maps.get(stage)
        if stage_map is None:
            continue
        step = _safe_int(record.get("count"))
        if step is None or step <= 0:
            global_step = _safe_int(record.get("global_step"))
            step = global_step + 1 if global_step is not None and global_step >= 0 else None
        if step is None or step <= 0:
            continue
        previous = stage_map.get(step)
        if previous is None:
            stage_map[step] = elapsed_seconds
        elif stage in start_stages:
            stage_map[step] = min(previous, elapsed_seconds)
        else:
            stage_map[step] = max(previous, elapsed_seconds)

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


def _heartbeat_timing_summary(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if rows is None:
        rows = _heartbeat_step_interval_rows(records)
    if not rows:
        return None
    warmup_steps = max(_safe_int(config.get("warmup_steps")) or 0, 0)
    measured_rows = [row for row in rows if int(row["raw_step"]) > warmup_steps]
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


def _trainer_timing_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    step_records: list[dict[str, Any]] = []
    for record in records:
        current_steps = _safe_int(record.get("current_steps"))
        elapsed_seconds = _duration_to_seconds(record.get("elapsed_time"))
        if current_steps is None or current_steps <= 0 or elapsed_seconds is None:
            continue
        step_records.append({**record, "current_steps": current_steps, "elapsed_seconds": elapsed_seconds})
    if not step_records:
        return {"available": False, "reason": "trainer log has no elapsed step records"}

    final_record = max(step_records, key=lambda row: (int(row["current_steps"]), float(row["elapsed_seconds"])))
    final_steps = int(final_record["current_steps"])
    final_elapsed_seconds = float(final_record["elapsed_seconds"])
    warmup_steps = max(_safe_int(config.get("warmup_steps")) or 0, 0)
    warmup_record = None
    if warmup_steps > 0:
        candidates = [row for row in step_records if int(row["current_steps"]) == warmup_steps]
        if candidates:
            warmup_record = max(candidates, key=lambda row: float(row["elapsed_seconds"]))

    measured_steps = final_steps
    measured_elapsed_seconds = final_elapsed_seconds
    if warmup_record is not None and final_steps > warmup_steps:
        measured_steps = final_steps - warmup_steps
        measured_elapsed_seconds = final_elapsed_seconds - float(warmup_record["elapsed_seconds"])

    total_step_ms = final_elapsed_seconds * 1000.0 / float(final_steps) if final_steps > 0 else None
    measured_step_ms = (
        measured_elapsed_seconds * 1000.0 / float(measured_steps) if measured_steps > 0 else None
    )
    return {
        "available": True,
        "source": "trainer_log_elapsed_time",
        "final_steps": final_steps,
        "final_elapsed_seconds": final_elapsed_seconds,
        "total_e2e_step_milliseconds": total_step_ms,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measured_elapsed_seconds": measured_elapsed_seconds,
        "measured_e2e_step_milliseconds": measured_step_ms,
    }


def _env_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(CONFIG_ENV_PREFIX):
            field_name = key[len(CONFIG_ENV_PREFIX) :].lower()
            config[field_name] = value
    return config


def _format_cpu_ranges(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _cpu_affinity_summary() -> dict[str, Any]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return {"available": False, "reason": "os.sched_getaffinity unavailable"}
    try:
        cpus = sorted(int(cpu) for cpu in get_affinity(0))
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}
    return {
        "available": True,
        "count": len(cpus),
        "cpus": _format_cpu_ranges(cpus),
    }


def _config_from_args(args: list[str]) -> dict[str, Any]:
    env_config = _env_config()
    cpu_affinity = _cpu_affinity_summary()
    model_name = _option_value(args, "--model_name_or_path")
    model_label = model_name.rstrip("/").rsplit("/", 1)[-1] if model_name else "lf"
    batch_size = _safe_int(_option_value(args, "--per_device_train_batch_size"))
    cutoff_len = _safe_int(_option_value(args, "--cutoff_len"))
    gradient_accumulation_steps = _safe_int(_option_value(args, "--gradient_accumulation_steps"))
    logical_qlen = batch_size * cutoff_len if batch_size is not None and cutoff_len is not None else None
    lora_rank = _safe_int(_option_value(args, "--lora_rank"))
    lora_alpha = _safe_float(_option_value(args, "--lora_alpha"))
    max_steps = _safe_int(_option_value(args, "--max_steps"))
    warmup_steps = max(_safe_int(env_config.get("warmup_steps")) or 0, 0)
    total_steps = _safe_int(env_config.get("total_steps")) or max_steps
    measure_steps = _safe_int(env_config.get("measure_steps"))
    if measure_steps is None and total_steps is not None:
        measure_steps = max(int(total_steps) - warmup_steps, 0)
    asym_backend = _option_value(args, "--asym_backend")
    backend = os.environ.get("ASYM_GEMM_LF_CONFIG_BACKEND") or ("torch" if asym_backend == "torch" else asym_backend or "hf")
    liger_loss = os.environ.get("ASYM_GEMM_LF_CONFIG_LIGER_LOSS", "ligerloss0").strip().lower()
    if liger_loss not in {"ligerloss0", "ligerloss1"}:
        raise SystemExit(
            f"ASYM_GEMM_LF_CONFIG_LIGER_LOSS must be ligerloss0 or ligerloss1, got {liger_loss!r}"
        )
    expert_policy = parse_expert_recompute_policy_spec(os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none"))
    attention_gc_enabled = os.environ.get("ASYM_GEMM_LF_CONFIG_ATTN_GC_ENABLED")
    if attention_gc_enabled is None:
        attention_gc_enabled = "true" if expert_policy.label == "gc-attn-exp" else "false"
    layer_gc_enabled = os.environ.get("ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED")
    if layer_gc_enabled is None:
        layer_gc_enabled = "true" if expert_policy.label == "gc-layer" else "false"
    is_superoffload_backend = backend in {"superoffload", "superoffload_mem"}
    is_cpuadam_backend = backend == "zero3_cpuadam"
    is_asym_deepspeed_cpuadamw = backend == "asym_cpuadamwds" or (
        str(env_config.get("use_asym_cpu_adamw", "")).lower() == "true"
        and str(env_config.get("asym_cpu_adamw_backend", "")).lower() == "deepspeed"
    )
    is_deepspeed_backend = backend in {
        "zero2",
        "zero3",
        "zero3_offload",
        "zero3_offload_mem",
        "zero3_offload_opnvme",
        "zero3_offload_pnvme",
        "zero3_cpuadam",
        "superoffload",
        "superoffload_mem",
    }
    config = {
        "workflow": "lora_lf_sft",
        "workload": os.environ.get("ASYM_GEMM_LF_CONFIG_WORKLOAD", model_label),
        "model_name_or_path": model_name,
        "backend": backend,
        "liger_loss": liger_loss,
        "kt_backend": os.environ.get("ASYM_GEMM_LF_CONFIG_KT_BACKEND") or _option_value(args, "--kt_backend"),
        "precision": os.environ.get("ASYM_GEMM_LF_CONFIG_PRECISION") or _option_value(args, "--asym_precision") or "bf16",
        "dataset": _option_value(args, "--dataset"),
        "template": _option_value(args, "--template"),
        "batch_size": batch_size,
        "per_device_train_batch_size": batch_size,
        "seq_len": _safe_int(os.environ.get("ASYM_GEMM_LF_CONFIG_SEQ_LEN")) or cutoff_len,
        "cutoff_len": cutoff_len,
        "logical_qlen": _safe_int(env_config.get("logical_qlen")) or logical_qlen,
        "max_samples": _safe_int(_option_value(args, "--max_samples")),
        "max_steps": measure_steps,
        "measure_steps": measure_steps,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "trainer_max_steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_target": _option_value(args, "--lora_target"),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": _safe_float(_option_value(args, "--lora_dropout")),
        "qwen_moe_expert_lora_impl": os.environ.get("ASYM_GEMM_LF_CONFIG_QWEN_EXPERT_LORA_IMPL")
        or os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", "split-target-parameters"),
        "kt_max_cache_depth": _safe_int(_option_value(args, "--kt_max_cache_depth")),
        "max_grad_norm": _safe_float(env_config.get("max_grad_norm"))
        if _safe_float(env_config.get("max_grad_norm")) is not None
        else _safe_float(_option_value(args, "--max_grad_norm")),
        "activation_recompute": os.environ.get("ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE", "false").lower()
        in {"1", "true", "yes", "on"},
        "asymm_expert_act_offload": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD", "false"),
        "asymm_expert_act_offload_lora_a_fwd": os.environ.get(
            "ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD",
            os.environ.get("ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD", "cpu"),
        ),
        "asymm_attn_act_offload": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD", "false"),
        "asymm_layer_act_offload": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD", "false"),
        "asymm_layer_gc": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC", "false"),
        "attention_gc_enabled": attention_gc_enabled,
        "layer_gc_enabled": layer_gc_enabled,
        "expert_recompute_policy_spec": expert_policy.label,
        "expert_recompute_policy": expert_policy.policy,
        "expert_recompute_impl": (
            "torch_checkpoint"
            if expert_policy.torch_checkpoint_enabled
            else "custom"
            if expert_policy.custom_autograd_enabled
            else "none"
        ),
        "expert_gc_use_reentrant": os.environ.get("ASYM_EXPERT_GC_USE_REENTRANT", os.environ.get("ASYM_EXPERT_GC_REENTRANT", "true")).lower()
        in {"1", "true", "yes", "on"},
        "expert_recompute_threshold": expert_policy.token_threshold,
        "expert_recompute_token_min": expert_policy.token_min,
        "expert_recompute_token_max": expert_policy.token_max,
        "expert_activation_save_policy": expert_policy.activation_save_policy,
        "expert_activation_save_threshold": expert_policy.activation_save_threshold,
        "expert_activation_save_token_min": expert_policy.activation_save_min,
        "expert_activation_save_token_max": expert_policy.activation_save_max,
        "expert_policy_label": expert_policy.label,
        "profile_level": os.environ.get(PROFILE_LEVEL_ENV, "stage"),
        "profile_layers": os.environ.get(PROFILE_LAYERS_ENV, "all"),
        "profile_memory_attribution": os.environ.get(PROFILE_MEMORY_ATTRIBUTION_ENV, "0"),
        "profile_memory_breakdown": os.environ.get(PROFILE_MEMORY_BREAKDOWN_ENV, "0"),
        "profile_memory_snapshot": os.environ.get(PROFILE_MEMORY_SNAPSHOT_ENV, "0"),
        "profile_external_memory": os.environ.get(PROFILE_EXTERNAL_MEMORY_ENV, "0"),
        "profile_sync": os.environ.get(PROFILE_SYNC_ENV, "0"),
        "profile_module_filter": os.environ.get(PROFILE_MODULE_FILTER_ENV, ""),
        "cpu_affinity": cpu_affinity,
        "cpu_affinity_count": cpu_affinity.get("count"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
        "omp_places": os.environ.get("OMP_PLACES", ""),
        "kt_lora_health_max_tensors": os.environ.get(KT_LORA_HEALTH_MAX_TENSORS_ENV, "12"),
        "kt_lora_health_max_elements": os.environ.get(KT_LORA_HEALTH_MAX_ELEMENTS_ENV, "4096"),
        "kt_grad_clip_chunk_elements": os.environ.get("ASYM_GEMM_LF_KT_GRAD_CLIP_CHUNK_ELEMENTS", ""),
        "asym_cpu_adamw_grad_offload": (
            os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_GRAD_OFFLOAD")
            or _option_value(args, "--asym_cpu_adamw_grad_offload")
            or "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        "asym_cpu_adamw_weight_offload": (
            os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD")
            or _option_value(args, "--asym_cpu_adamw_weight_offload")
            or "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        "nsys_capture_range": _env_enabled("ASYM_GEMM_LF_NSYS_CAPTURE_RANGE"),
        "superoffload_config": os.environ.get("ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG")
        if is_superoffload_backend
        else None,
        "cpuadam_config": os.environ.get("ASYM_GEMM_LF_CONFIG_CPUADAM_CONFIG") if is_cpuadam_backend else None,
        "deepspeed_config": os.environ.get("ASYM_GEMM_LF_CONFIG_DEEPSPEED_CONFIG")
        if is_deepspeed_backend
        else None,
        "deepspeed_dir": os.environ.get("ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR")
        if is_deepspeed_backend or is_asym_deepspeed_cpuadamw
        else None,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
        "output_dir": _option_value(args, "--output_dir"),
    }
    for key, value in env_config.items():
        config.setdefault(key, value)
    return {key: value for key, value in config.items() if value is not None and value != ""}


def _kt_startup_validation_errors(kt: dict[str, Any], lora: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _env_enabled(KT_REQUIRE_STARTUP_ENV):
        wrapper_count = _safe_int(kt.get("wrapper_count")) or 0
        if not kt.get("available", False):
            errors.append(f"KT startup counters unavailable: {kt.get('reason', 'unknown')}")
        elif wrapper_count <= 0:
            errors.append(f"KT startup wrapper_count must be positive, got {wrapper_count}")
    if _env_enabled(KT_REQUIRE_FUSED_LORA_STARTUP_ENV):
        fused_lora_params = _safe_int(lora.get("kt_fused_expert_lora_parameters")) or 0
        if not lora.get("available", False):
            errors.append(f"KT fused LoRA startup counters unavailable: {lora.get('reason', 'unknown')}")
        elif fused_lora_params <= 0:
            errors.append(f"KT fused expert LoRA params must be positive at startup, got {fused_lora_params}")
    return errors


def _kt_optimizer_memory_preflight(lora: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(lora, dict) or lora.get("available") is False:
        return {
            "available": False,
            "reason": f"lora counters unavailable: {lora.get('reason', 'unknown') if isinstance(lora, dict) else 'missing'}",
        }

    trainable_params = _safe_int(lora.get("trainable_parameters"))
    if trainable_params is None:
        return {
            "available": False,
            "reason": "lora counter trainable_parameters is missing or non-numeric",
        }

    kt_fused_params = _safe_int(lora.get("kt_fused_expert_lora_parameters"))
    kt_expert_params = _safe_int(lora.get("kt_expert_lora_parameters"))
    peft_lora_params = _safe_int(lora.get("peft_lora_parameters"))
    peft_expert_params = _safe_int(lora.get("peft_expert_lora_parameters"))
    qwen_moe_expert_params = _safe_int(lora.get("qwen_moe_expert_lora_parameters"))
    kt_peft_expert_params = _safe_int(lora.get("kt_peft_expert_lora_parameters"))
    if peft_expert_params is None:
        peft_expert_params = kt_peft_expert_params
    expert_lora_params = max(
        kt_expert_params or 0,
        peft_expert_params or 0,
        qwen_moe_expert_params or 0,
    )
    non_expert_peft_params = (
        max(0, peft_lora_params - (peft_expert_params or 0)) if peft_lora_params is not None else None
    )

    def bf16_bytes(params: int) -> int:
        return int(params) * 2

    def fp32_bytes(params: int) -> int:
        return int(params) * 4

    adamw_bf16_moments_bytes = 2 * bf16_bytes(trainable_params)
    adamw_fp32_moments_bytes = 2 * fp32_bytes(trainable_params)
    param_bf16_bytes = bf16_bytes(trainable_params)
    grad_bf16_bytes = bf16_bytes(trainable_params)
    total_bf16_moment_policy = param_bf16_bytes + grad_bf16_bytes + adamw_bf16_moments_bytes
    total_fp32_moment_policy = param_bf16_bytes + grad_bf16_bytes + adamw_fp32_moments_bytes
    total_fp32_moment_master_policy = total_fp32_moment_policy + fp32_bytes(trainable_params)

    return {
        "available": True,
        "source": "lora_counters_pre_optimizer_step",
        "assumed_param_dtype": "bf16",
        "trainable_parameters": trainable_params,
        "expert_lora_parameters": expert_lora_params,
        "kt_fused_expert_lora_parameters": kt_fused_params,
        "kt_expert_lora_parameters": kt_expert_params,
        "peft_expert_lora_parameters": peft_expert_params,
        "qwen_moe_expert_lora_parameters": qwen_moe_expert_params,
        "non_expert_peft_lora_parameters": non_expert_peft_params,
        "param_bf16_bytes": param_bf16_bytes,
        "grad_bf16_bytes": grad_bf16_bytes,
        "adamw_bf16_moments_bytes": adamw_bf16_moments_bytes,
        "adamw_fp32_moments_bytes": adamw_fp32_moments_bytes,
        "adamw_fp32_master_bytes": fp32_bytes(trainable_params),
        "total_bf16_params_grads_adamw_bf16_moments_bytes": total_bf16_moment_policy,
        "total_bf16_params_grads_adamw_fp32_moments_bytes": total_fp32_moment_policy,
        "total_bf16_params_grads_adamw_fp32_moments_master_bytes": total_fp32_moment_master_policy,
        "large_surface_warning": bool(
            trainable_params >= 1_000_000_000
            or expert_lora_params >= 1_000_000_000
            or (qwen_moe_expert_params is not None and qwen_moe_expert_params >= 1_000_000_000)
            or (kt_fused_params is not None and kt_fused_params >= 1_000_000_000)
        ),
        "logical_qlen": _safe_int((config or {}).get("logical_qlen")),
        "lora_rank": _safe_int((config or {}).get("lora_rank")),
        "kt_arm_sft_top_k": _safe_int((config or {}).get("kt_arm_sft_top_k")),
        "kt_arm_sft_token_chunk_size": _safe_int((config or {}).get("kt_arm_sft_token_chunk_size")),
        "kt_arm_effective_route_qlen": _safe_int((config or {}).get("kt_arm_effective_route_qlen")),
        "kt_arm_token_chunks": _safe_int((config or {}).get("kt_arm_token_chunks")),
        "kt_arm_route_rank_work": _safe_int((config or {}).get("kt_arm_route_rank_work")),
        "kt_arm_sft_max_route_rank_work": _safe_int((config or {}).get("kt_arm_sft_max_route_rank_work")),
    }


def _capture_loaded_model(model: Any) -> Any:
    global _LAST_LF_MODEL
    _LAST_LF_MODEL = model
    heartbeat = _MODEL_CAPTURE_HEARTBEAT
    partial_writer = _MODEL_CAPTURE_PARTIAL_WRITER
    if heartbeat is not None:
        try:
            kt = _kt_counters_from_model(getattr(heartbeat, "config", {}))
            lora = _lora_counters_from_model()
            optimizer_memory_preflight = _kt_optimizer_memory_preflight(lora, getattr(heartbeat, "config", {}))
        except Exception as exc:
            heartbeat.emit("model_loaded_summary_exception", error=repr(exc))
            if partial_writer is not None:
                partial_writer.write("model_loaded_summary_exception", force=True, extra={"error": repr(exc)})
            if _env_enabled(KT_REQUIRE_STARTUP_ENV) or _env_enabled(KT_REQUIRE_FUSED_LORA_STARTUP_ENV):
                raise
            return model

        heartbeat.emit(
            "model_loaded",
            kt_available=kt.get("available"),
            kt_wrappers=kt.get("wrapper_count"),
            kt_forward_calls=kt.get("total_forward_calls"),
            kt_backward_calls=kt.get("total_backward_calls"),
            trainable_parameters=lora.get("trainable_parameters"),
            expert_lora_parameters=optimizer_memory_preflight.get("expert_lora_parameters"),
            kt_expert_lora_parameters=lora.get("kt_expert_lora_parameters"),
            kt_fused_expert_lora_parameters=lora.get("kt_fused_expert_lora_parameters"),
            optimizer_preflight_total_bf16_moments_bytes=optimizer_memory_preflight.get(
                "total_bf16_params_grads_adamw_bf16_moments_bytes"
            ),
            optimizer_preflight_total_fp32_moments_master_bytes=optimizer_memory_preflight.get(
                "total_bf16_params_grads_adamw_fp32_moments_master_bytes"
            ),
        )
        if partial_writer is not None:
            partial_writer.write(
                "model_loaded",
                force=True,
                extra={"kt": kt, "lora": lora, "optimizer_memory_preflight": optimizer_memory_preflight},
            )

        validation_errors = _kt_startup_validation_errors(kt, lora)
        if validation_errors:
            message = "; ".join(validation_errors)
            heartbeat.emit("model_loaded_startup_validation_failed", error=message)
            if partial_writer is not None:
                partial_writer.write(
                    "model_loaded_startup_validation_failed",
                    force=True,
                    extra={"error": message, "kt": kt, "lora": lora},
                )
            raise RuntimeError(message)
    return model


def _install_model_capture_hook() -> None:
    try:
        import llamafactory.model as model_module
        import llamafactory.model.loader as loader_module
    except Exception:
        return

    original_load_model = getattr(model_module, "load_model", None)
    if original_load_model is None:
        return
    if getattr(original_load_model, "_asym_gemm_profile_capture", False):
        return

    def wrapped_load_model(*args: Any, **kwargs: Any) -> Any:
        return _capture_loaded_model(original_load_model(*args, **kwargs))

    wrapped_load_model._asym_gemm_profile_capture = True  # type: ignore[attr-defined]
    model_module.load_model = wrapped_load_model
    loader_module.load_model = wrapped_load_model

    for module_name in (
        "llamafactory.train.sft.workflow",
        "llamafactory.train.tuner",
        "llamafactory.train.trainer_utils",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "load_model"):
            setattr(module, "load_model", wrapped_load_model)

    try:
        import llamafactory.train.sft.workflow as sft_workflow

        sft_workflow.load_model = wrapped_load_model
    except Exception:
        pass


def _model_and_base_model() -> tuple[Any | None, Any | None]:
    model = _LAST_LF_MODEL
    base_model = None
    if model is not None and hasattr(model, "get_base_model"):
        try:
            base_model = model.get_base_model()
        except Exception:
            base_model = None
    return model, base_model


def _find_kt_wrappers(model: Any | None, base_model: Any | None) -> list[Any] | None:
    for candidate in (model, base_model):
        wrappers = getattr(candidate, "_kt_wrappers", None)
        if wrappers is not None:
            return list(wrappers)
    return None


def _kt_counters_from_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    wrappers = _find_kt_wrappers(model, base_model)
    rows: list[dict[str, Any]] = []
    config = config or {}
    logical_qlen = _safe_int(config.get("logical_qlen"))
    logical_max_cache_depth = _safe_int(config.get("kt_max_cache_depth"))
    kt_arm_token_chunks = _safe_int(config.get("kt_arm_token_chunks"))
    expected_effective_max_cache_depth = (
        logical_max_cache_depth * kt_arm_token_chunks
        if logical_max_cache_depth is not None and kt_arm_token_chunks is not None
        else None
    )
    kt_arm_row_context = {
        "kt_arm_logical_qlen": logical_qlen,
        "kt_arm_sft_token_chunk_size": _safe_int(config.get("kt_arm_sft_token_chunk_size")),
        "kt_arm_effective_route_qlen": _safe_int(config.get("kt_arm_effective_route_qlen")),
        "kt_arm_token_chunks": kt_arm_token_chunks,
        "kt_arm_route_rank_work": _safe_int(config.get("kt_arm_route_rank_work")),
        "logical_max_cache_depth": logical_max_cache_depth,
        "expected_effective_max_cache_depth": expected_effective_max_cache_depth,
    }
    for index, layer in enumerate(wrappers or []):
        wrapper = getattr(layer, "wrapper", None)
        effective_max_cache_depth = int(getattr(wrapper, "max_cache_depth", 0) or 0)
        rows.append(
            {
                "index": index,
                "layer_idx": getattr(layer, "layer_idx", index),
                "method": getattr(wrapper, "method", ""),
                "forward_calls": int(getattr(wrapper, "_kt_forward_calls", 0) or 0),
                "backward_calls": int(getattr(wrapper, "_kt_backward_calls", 0) or 0),
                "lora_initialized": bool(getattr(wrapper, "_lora_initialized", False)),
                "cache_depth": int(getattr(wrapper, "_cache_depth", 0) or 0),
                "max_cache_depth": effective_max_cache_depth,
                "effective_max_cache_depth": effective_max_cache_depth,
                "max_cache_depth_observed": int(getattr(wrapper, "_max_cache_depth_observed", 0) or 0),
                **kt_arm_row_context,
                "staging_buffer_scope": "wrapper",
                "staging_buffer_capacity_qlen": int(getattr(getattr(wrapper, "_buffer", None), "qlen", 0) or 0),
                "staging_buffer_capacity_bytes": int(
                    getattr(getattr(wrapper, "_buffer", None), "capacity_bytes", lambda: 0)() or 0
                ),
                "staging_buffer_allocation_count": int(getattr(wrapper, "_buffer_allocation_count", 0) or 0),
                "native_cache_stack_depth": int(getattr(wrapper, "cache_stack_depth", 0) or 0),
                "native_max_cache_stack_depth_observed": int(
                    getattr(wrapper, "max_cache_stack_depth_observed", 0) or 0
                ),
                "native_cache_save_count": int(getattr(wrapper, "cache_save_count", 0) or 0),
                "native_cache_pop_count": int(getattr(wrapper, "cache_pop_count", 0) or 0),
                "native_last_cache_entry_bytes": int(getattr(wrapper, "last_cache_entry_bytes", 0) or 0),
                "native_total_cache_entry_bytes_saved": int(
                    getattr(wrapper, "total_cache_entry_bytes_saved", 0) or 0
                ),
                "native_last_cache_route_bytes": int(getattr(wrapper, "last_cache_route_bytes", 0) or 0),
                "native_total_cache_route_bytes_saved": int(
                    getattr(wrapper, "total_cache_route_bytes_saved", 0) or 0
                ),
                "timing": dict(getattr(wrapper, "_kt_timing", {}) or {}),
            }
        )

    return {
        "available": wrappers is not None,
        "wrapper_count": len(wrappers or []),
        "rows": rows,
        "total_forward_calls": sum(int(row["forward_calls"]) for row in rows),
        "total_backward_calls": sum(int(row["backward_calls"]) for row in rows),
    }


def _superoffload_summary_from_config(config: dict[str, Any]) -> dict[str, Any]:
    is_superoffload_backend = str(config.get("backend") or "").lower() in {"superoffload", "superoffload_mem"}
    config_path = str(config.get("superoffload_config") or "") if is_superoffload_backend else ""
    config_super_offload = False
    cpuadam_cores_perc = None
    if config_path:
        path = Path(config_path)
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    ds_config = json.load(handle)
                zero = ds_config.get("zero_optimization", {}) if isinstance(ds_config, dict) else {}
                optimizer = zero.get("offload_optimizer", {}) if isinstance(zero, dict) else {}
                if isinstance(optimizer, dict):
                    config_super_offload = optimizer.get("super_offload") is True
                    cpuadam_cores_perc = _safe_float(optimizer.get("cpuadam_cores_perc")) or cpuadam_cores_perc
            except (OSError, json.JSONDecodeError):
                config_super_offload = False

    optimizer_class = _DEEPSPEED_RUNTIME_MARKER.get("optimizer_class")
    runtime_verified = optimizer_class == "SuperOffloadOptimizer_Stage3"
    return {
        "enabled": bool(config_super_offload or runtime_verified),
        "runtime_verified": bool(runtime_verified),
        "optimizer_class": optimizer_class,
        "engine_class": _DEEPSPEED_RUNTIME_MARKER.get("engine_class"),
        "engine_super_offload": _DEEPSPEED_RUNTIME_MARKER.get("engine_super_offload"),
        "config_super_offload": bool(config_super_offload),
        "cpuadam_cores_perc": cpuadam_cores_perc,
        "deepspeed_config": config_path or None,
        "deepspeed_dir": config.get("deepspeed_dir"),
        "marker_source": _DEEPSPEED_RUNTIME_MARKER.get("marker_source"),
    }


def _load_deepspeed_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _cpuadam_summary_from_config(config: dict[str, Any]) -> dict[str, Any]:
    is_cpuadam_backend = str(config.get("backend") or "").lower() == "zero3_cpuadam"
    config_path = str(config.get("cpuadam_config") or config.get("deepspeed_config") or "") if is_cpuadam_backend else ""
    ds_config = _load_deepspeed_config(config_path)
    zero = ds_config.get("zero_optimization", {}) if isinstance(ds_config, dict) else {}
    optimizer = ds_config.get("optimizer", {}) if isinstance(ds_config, dict) else {}
    optimizer_params = optimizer.get("params", {}) if isinstance(optimizer, dict) else {}
    offload_optimizer = zero.get("offload_optimizer", {}) if isinstance(zero, dict) else {}
    offload_param = zero.get("offload_param", {}) if isinstance(zero, dict) else {}

    optimizer_type = optimizer.get("type") if isinstance(optimizer, dict) else None
    offload_optimizer_device = offload_optimizer.get("device") if isinstance(offload_optimizer, dict) else None
    offload_param_device = offload_param.get("device") if isinstance(offload_param, dict) else None
    torch_adam = optimizer_params.get("torch_adam") if isinstance(optimizer_params, dict) else None
    adam_w_mode = optimizer_params.get("adam_w_mode") if isinstance(optimizer_params, dict) else None
    fp32_optimizer_states = optimizer_params.get("fp32_optimizer_states") if isinstance(optimizer_params, dict) else None

    optimizer_class = _DEEPSPEED_RUNTIME_MARKER.get("optimizer_class")
    optimizer_wrapper_class = _DEEPSPEED_RUNTIME_MARKER.get("optimizer_wrapper_class")
    basic_optimizer_class = _DEEPSPEED_RUNTIME_MARKER.get("basic_optimizer_class")
    runtime_verified = basic_optimizer_class == "DeepSpeedCPUAdam" or optimizer_class == "DeepSpeedCPUAdam"
    config_enabled = (
        str(optimizer_type or "").lower() == "adamw"
        and str(offload_optimizer_device or "").lower() == "cpu"
        and torch_adam is False
    )
    return {
        "enabled": bool(config_enabled or runtime_verified),
        "runtime_verified": bool(runtime_verified),
        "optimizer_class": optimizer_class,
        "optimizer_wrapper_class": optimizer_wrapper_class,
        "basic_optimizer_class": basic_optimizer_class,
        "engine_class": _DEEPSPEED_RUNTIME_MARKER.get("engine_class"),
        "engine_super_offload": _DEEPSPEED_RUNTIME_MARKER.get("engine_super_offload"),
        "config_optimizer_type": optimizer_type,
        "config_offload_optimizer_device": offload_optimizer_device,
        "config_offload_param_device": offload_param_device,
        "config_torch_adam": torch_adam,
        "config_adam_w_mode": adam_w_mode,
        "config_fp32_optimizer_states": fp32_optimizer_states,
        "deepspeed_config": config_path or None,
        "deepspeed_dir": config.get("deepspeed_dir"),
        "marker_source": _DEEPSPEED_RUNTIME_MARKER.get("marker_source"),
    }


def _asym_cpu_adamw_summary_from_trace(trace_handle: Any | None) -> dict[str, Any]:
    if trace_handle is None:
        return {"enabled": False}
    optimizer = getattr(trace_handle, "optimizer", None) or getattr(trace_handle, "prepared_optimizer", None)
    seen: set[int] = set()
    for _ in range(8):
        if optimizer is None:
            break
        optimizer_id = id(optimizer)
        if optimizer_id in seen:
            break
        seen.add(optimizer_id)
        summary_fn = getattr(optimizer, "asym_cpu_adamw_summary", None)
        if callable(summary_fn):
            try:
                summary = summary_fn()
            except Exception as exc:
                return {"enabled": True, "summary_error": str(exc)}
            if isinstance(summary, dict):
                return {"enabled": True, **summary}
        next_optimizer = None
        for attr in ("optimizer", "base_optimizer", "wrapped_optimizer", "inner_optimizer"):
            candidate = getattr(optimizer, attr, None)
            if candidate is not None and candidate is not optimizer:
                next_optimizer = candidate
                break
        optimizer = next_optimizer
    config = getattr(trace_handle, "config", {}) or {}
    if isinstance(config, dict):
        enabled_value = config.get("use_asym_cpu_adamw", "")
    else:
        enabled_value = getattr(config, "use_asym_cpu_adamw", "")
    enabled = str(enabled_value).lower() == "true"
    return {"enabled": bool(enabled)}


def _install_deepspeed_optimizer_capture_hook() -> None:
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except Exception:
        return

    original_configure_optimizer = getattr(DeepSpeedEngine, "_configure_optimizer", None)
    if original_configure_optimizer is None:
        return
    if getattr(original_configure_optimizer, "_asym_gemm_deepspeed_optimizer_capture", False):
        return

    def wrapped_configure_optimizer(self: Any, client_optimizer: Any, model_parameters: Any) -> Any:
        result = original_configure_optimizer(self, client_optimizer, model_parameters)
        optimizer = getattr(self, "optimizer", None)
        basic_optimizer = getattr(self, "basic_optimizer", None)
        nested_optimizer = getattr(optimizer, "optimizer", None) if optimizer is not None else None
        optimizer_class = optimizer.__class__.__name__ if optimizer is not None else None
        basic_optimizer_class = basic_optimizer.__class__.__name__ if basic_optimizer is not None else None
        nested_optimizer_class = nested_optimizer.__class__.__name__ if nested_optimizer is not None else None
        try:
            engine_super_offload = bool(self.super_offload())
        except Exception:
            engine_super_offload = None
        _DEEPSPEED_RUNTIME_MARKER.update(
            {
                "optimizer_class": optimizer_class,
                "optimizer_wrapper_class": optimizer_class,
                "basic_optimizer_class": basic_optimizer_class or nested_optimizer_class,
                "nested_optimizer_class": nested_optimizer_class,
                "engine_class": self.__class__.__name__,
                "engine_super_offload": engine_super_offload,
                "marker_source": "deepspeed_engine_hook",
            }
        )
        if _is_rank0() and optimizer_class:
            print(f"DeepSpeed Final Optimizer = {optimizer_class}", flush=True)
            if basic_optimizer_class:
                print(f"DeepSpeed Basic Optimizer = {basic_optimizer_class}", flush=True)
            if engine_super_offload:
                print("DeepSpeed SuperOffload runtime enabled = true", flush=True)
            if basic_optimizer_class == "DeepSpeedCPUAdam" or nested_optimizer_class == "DeepSpeedCPUAdam":
                print("DeepSpeed CPUAdam runtime enabled = true", flush=True)
        return result

    wrapped_configure_optimizer._asym_gemm_deepspeed_optimizer_capture = True  # type: ignore[attr-defined]
    DeepSpeedEngine._configure_optimizer = wrapped_configure_optimizer


def _should_install_deepspeed_hook(args: list[str], config: dict[str, Any]) -> bool:
    if _option_value(args, "--deepspeed"):
        return True
    backend = str(config.get("backend") or "").lower()
    if backend in {"zero2", "zero3", "zero3_offload", "zero3_offload_mem", "zero3_offload_opnvme", "zero3_offload_pnvme", "zero3_cpuadam", "superoffload", "superoffload_mem"}:
        return True
    return bool(config.get("superoffload_config") or config.get("cpuadam_config") or config.get("deepspeed_config"))


def _install_trainer_heartbeat_hooks(heartbeat: LFHeartbeat, partial_writer: PartialProfileWriter) -> None:
    try:
        from transformers import Trainer
    except Exception:
        return

    def _result_to_float(result: Any) -> Any:
        if hasattr(result, "item"):
            try:
                return float(result.item())
            except (TypeError, ValueError):
                return result
        return result

    def _wrap_grad_norm_method(owner: Any, method_name: str, fallback_operation: str) -> None:
        original = getattr(owner, method_name, None)
        if original is None or getattr(original, "_asym_gemm_heartbeat", False):
            return

        def wrapped_grad_norm_method(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
            depth = int(getattr(self, "_asym_gemm_grad_norm_heartbeat_depth", 0) or 0)
            if depth > 0:
                return original(self, model, *args, **kwargs)

            global _GRAD_CLIP_MARKER
            step = getattr(getattr(self, "state", None), "global_step", None)
            started = time.perf_counter()
            start_record = {
                "trainer_class": self.__class__.__name__,
                "global_step": step,
                "operation": fallback_operation,
                "method": f"{owner.__name__}.{method_name}",
            }
            heartbeat.emit("grad_clip_start", **start_record)
            partial_writer.write("grad_clip_start", force=True, extra={"grad_clip": start_record})
            setattr(self, "_asym_gemm_grad_norm_heartbeat_depth", depth + 1)
            try:
                result = original(self, model, *args, **kwargs)
            except BaseException as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                record = {
                    **start_record,
                    "available": False,
                    "elapsed_ms": elapsed_ms,
                    "error": repr(exc),
                    "exception_type": exc.__class__.__name__,
                    "timestamp": _utc_timestamp(),
                }
                _GRAD_CLIP_MARKER = record
                heartbeat.emit("grad_clip_exception", **record)
                partial_writer.write("grad_clip_exception", force=True, extra={"grad_clip": record, "error": repr(exc)})
                raise
            finally:
                if depth:
                    setattr(self, "_asym_gemm_grad_norm_heartbeat_depth", depth)
                else:
                    try:
                        delattr(self, "_asym_gemm_grad_norm_heartbeat_depth")
                    except AttributeError:
                        pass

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record = dict(start_record)
            summary = getattr(self, "_asym_cpu_adamw_grad_clip_last_summary", None)
            if not isinstance(summary, dict):
                summary = getattr(self, "_kt_grad_clip_last_summary", None)
            if isinstance(summary, dict):
                record.update(summary)
            else:
                record.update({"enabled": False, "path": "default"})
            record.update(
                {
                    "available": True,
                    "elapsed_ms": elapsed_ms,
                    "result_norm": _result_to_float(result),
                    "timestamp": _utc_timestamp(),
                }
            )
            _GRAD_CLIP_MARKER = record
            heartbeat.emit("grad_clip_end", **record)
            partial_writer.write("grad_clip_end", force=True, extra={"grad_clip": record})
            return result

        wrapped_grad_norm_method._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        setattr(owner, method_name, wrapped_grad_norm_method)

    if not getattr(Trainer.train, "_asym_gemm_heartbeat", False):
        original_train = Trainer.train

        def wrapped_train(self: Any, *args: Any, **kwargs: Any) -> Any:
            heartbeat.emit("trainer_start", trainer_class=self.__class__.__name__)
            partial_writer.write("trainer_start", force=True)
            try:
                result = original_train(self, *args, **kwargs)
            except BaseException as exc:
                heartbeat.emit("trainer_exception", trainer_class=self.__class__.__name__, error=repr(exc))
                partial_writer.write("trainer_exception", force=True, extra={"error": repr(exc)})
                raise
            heartbeat.emit("trainer_end", trainer_class=self.__class__.__name__)
            partial_writer.write("trainer_end", force=True)
            return result

        wrapped_train._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        Trainer.train = wrapped_train

    if hasattr(Trainer, "get_batch_samples") and not getattr(Trainer.get_batch_samples, "_asym_gemm_heartbeat", False):
        original_get_batch_samples = Trainer.get_batch_samples

        def wrapped_get_batch_samples(self: Any, *args: Any, **kwargs: Any) -> Any:
            heartbeat.emit("dataloader_batch_fetch_start", trainer_class=self.__class__.__name__)
            partial_writer.write("dataloader_batch_fetch_start")
            result = original_get_batch_samples(self, *args, **kwargs)
            batch_samples = result[0] if isinstance(result, tuple) and result else result
            batch_count = len(batch_samples) if hasattr(batch_samples, "__len__") else None
            heartbeat.emit(
                "dataloader_batch_fetch_end",
                trainer_class=self.__class__.__name__,
                batch_count=batch_count,
            )
            partial_writer.write("dataloader_batch_fetch_end")
            return result

        wrapped_get_batch_samples._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        Trainer.get_batch_samples = wrapped_get_batch_samples

    if not getattr(Trainer.training_step, "_asym_gemm_heartbeat", False):
        original_training_step = Trainer.training_step

        def wrapped_training_step(self: Any, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            step = getattr(getattr(self, "state", None), "global_step", None)
            heartbeat.emit("training_step_start", trainer_class=self.__class__.__name__, global_step=step)
            partial_writer.write("training_step_start", force=True)
            try:
                result = original_training_step(self, model, inputs, *args, **kwargs)
            except BaseException as exc:
                heartbeat.emit(
                    "training_step_exception",
                    trainer_class=self.__class__.__name__,
                    global_step=step,
                    error=repr(exc),
                )
                partial_writer.write("training_step_exception", force=True, extra={"error": repr(exc)})
                raise
            heartbeat.emit("training_step_end", trainer_class=self.__class__.__name__, global_step=step)
            partial_writer.write("training_step_end")
            return result

        wrapped_training_step._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        Trainer.training_step = wrapped_training_step

    _wrap_grad_norm_method(Trainer, "_clip_grad_norm", "clip")
    _wrap_grad_norm_method(Trainer, "_get_grad_norm", "norm")

    if hasattr(Trainer, "create_optimizer") and not getattr(Trainer.create_optimizer, "_asym_gemm_heartbeat", False):
        original_create_optimizer = Trainer.create_optimizer

        def wrapped_create_optimizer(self: Any, *args: Any, **kwargs: Any) -> Any:
            optimizer = original_create_optimizer(self, *args, **kwargs)
            target_optimizer = getattr(self, "optimizer", optimizer)
            if target_optimizer is not None and not getattr(target_optimizer.step, "_asym_gemm_heartbeat", False):
                original_step = target_optimizer.step

                def wrapped_optimizer_step(*step_args: Any, **step_kwargs: Any) -> Any:
                    global _OPTIMIZER_MEMORY_MARKER
                    step = getattr(getattr(self, "state", None), "global_step", None)
                    process_memory_at_start = _process_memory_snapshot()
                    heartbeat.emit(
                        "optimizer_step_start",
                        trainer_class=self.__class__.__name__,
                        global_step=step,
                        process_rss_bytes=process_memory_at_start.get("rss_bytes"),
                        process_rss_peak_bytes=process_memory_at_start.get("rss_peak_bytes"),
                    )
                    partial_writer.write("optimizer_step_start", force=True)
                    kt_lora_health_max_tensors = _kt_lora_health_max_tensors()
                    kt_lora_health_max_elements = _kt_lora_health_max_elements()
                    kt_lora_before = _kt_fused_lora_update_snapshot(
                        getattr(self, "model", None),
                        max_tensors=kt_lora_health_max_tensors,
                        max_elements=kt_lora_health_max_elements,
                    )
                    process_memory_before = _process_memory_snapshot()
                    try:
                        result = original_step(*step_args, **step_kwargs)
                    except BaseException as exc:
                        process_memory_after = _process_memory_snapshot()
                        try:
                            summary = _optimizer_memory_summary(target_optimizer, getattr(self, "model", None))
                        except BaseException as summary_exc:
                            summary = {
                                "available": False,
                                "reason": f"optimizer memory summary failed after optimizer.step exception: {summary_exc!r}",
                            }
                        summary["process_memory_at_start"] = process_memory_at_start
                        summary["process_memory_before_step"] = process_memory_before
                        summary["process_memory_after_step"] = process_memory_after
                        start_rss = _process_memory_value(process_memory_at_start, "rss_bytes")
                        before_rss = _process_memory_value(process_memory_before, "rss_bytes")
                        after_rss = _process_memory_value(process_memory_after, "rss_bytes")
                        summary["process_rss_pre_step_overhead_delta_bytes"] = before_rss - start_rss
                        summary["process_rss_delta_bytes"] = after_rss - before_rss
                        summary["kt_lora_update_health"] = {
                            "available": False,
                            "passed": False,
                            "reason": f"optimizer.step raised before after-step fused LoRA snapshot: {exc!r}",
                            "before": kt_lora_before,
                        }
                        summary.update(
                            {
                                "timestamp": _utc_timestamp(),
                                "trainer_class": self.__class__.__name__,
                                "global_step": step,
                                "error": repr(exc),
                                "exception_type": exc.__class__.__name__,
                            }
                        )
                        _OPTIMIZER_MEMORY_MARKER = summary
                        heartbeat.emit("optimizer_step_exception", **summary)
                        partial_writer.write(
                            "optimizer_step_exception",
                            force=True,
                            extra={"optimizer_memory": summary, "error": repr(exc)},
                        )
                        raise
                    process_memory_after = _process_memory_snapshot()
                    kt_lora_after = _kt_fused_lora_update_snapshot(
                        getattr(self, "model", None),
                        max_tensors=kt_lora_health_max_tensors,
                        max_elements=kt_lora_health_max_elements,
                    )
                    summary = _optimizer_memory_summary(target_optimizer, getattr(self, "model", None))
                    summary["process_memory_at_start"] = process_memory_at_start
                    summary["process_memory_before_step"] = process_memory_before
                    summary["process_memory_after_step"] = process_memory_after
                    start_rss = _process_memory_value(process_memory_at_start, "rss_bytes")
                    before_rss = _process_memory_value(process_memory_before, "rss_bytes")
                    after_rss = _process_memory_value(process_memory_after, "rss_bytes")
                    summary["process_rss_pre_step_overhead_delta_bytes"] = before_rss - start_rss
                    summary["process_rss_delta_bytes"] = after_rss - before_rss
                    summary["kt_lora_update_health"] = _kt_fused_lora_update_health(kt_lora_before, kt_lora_after)
                    summary.update(
                        {
                            "timestamp": _utc_timestamp(),
                            "trainer_class": self.__class__.__name__,
                            "global_step": step,
                        }
                    )
                    _OPTIMIZER_MEMORY_MARKER = summary
                    heartbeat.emit("optimizer_step_end", **summary)
                    partial_writer.write("optimizer_step_end", force=True, extra={"optimizer_memory": summary})
                    return result

                wrapped_optimizer_step._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
                target_optimizer.step = wrapped_optimizer_step
            return optimizer

        wrapped_create_optimizer._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        Trainer.create_optimizer = wrapped_create_optimizer

    try:
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    except Exception:
        return

    _wrap_grad_norm_method(CustomSeq2SeqTrainer, "_clip_grad_norm", "clip")
    _wrap_grad_norm_method(CustomSeq2SeqTrainer, "_get_grad_norm", "norm")

    if not getattr(CustomSeq2SeqTrainer.compute_loss, "_asym_gemm_heartbeat", False):
        original_compute_loss = CustomSeq2SeqTrainer.compute_loss

        def wrapped_compute_loss(self: Any, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            step = getattr(getattr(self, "state", None), "global_step", None)
            heartbeat.emit("model_forward_enter", trainer_class=self.__class__.__name__, global_step=step)
            partial_writer.write("model_forward_enter", force=True)
            try:
                result = original_compute_loss(self, model, inputs, *args, **kwargs)
            except BaseException as exc:
                heartbeat.emit(
                    "model_forward_exception",
                    trainer_class=self.__class__.__name__,
                    global_step=step,
                    error=repr(exc),
                )
                partial_writer.write("model_forward_exception", force=True, extra={"error": repr(exc)})
                raise
            heartbeat.emit("model_forward_exit", trainer_class=self.__class__.__name__, global_step=step)
            partial_writer.write("model_forward_exit")
            return result

        wrapped_compute_loss._asym_gemm_heartbeat = True  # type: ignore[attr-defined]
        CustomSeq2SeqTrainer.compute_loss = wrapped_compute_loss


def _start_memory_snapshot_recording(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    if not torch.cuda.is_available():
        return {"enabled": True, "record_started": False, "error": "cuda unavailable"}
    max_entries = _env_int(PROFILE_MEMORY_SNAPSHOT_MAX_ENTRIES_ENV, 1_000_000, minimum=1)
    try:
        torch.cuda.memory._record_memory_history(  # type: ignore[attr-defined]
            max_entries=max_entries,
            stacks="python",
            context="all",
        )
    except TypeError:
        try:
            torch.cuda.memory._record_memory_history()  # type: ignore[attr-defined]
        except Exception as exc:
            return {"enabled": True, "record_started": False, "error": str(exc)}
    except Exception as exc:
        return {"enabled": True, "record_started": False, "error": str(exc)}
    return {"enabled": True, "record_started": True, "max_entries": max_entries}


def _dump_memory_snapshot(snapshot_info: dict[str, Any], path: Path) -> dict[str, Any]:
    if not snapshot_info.get("record_started"):
        return snapshot_info
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(str(path))  # type: ignore[attr-defined]
    except Exception as exc:
        result = dict(snapshot_info)
        result.update({"path": str(path), "dumped": False, "error": str(exc)})
        return result
    result = dict(snapshot_info)
    result.update({"path": str(path), "dumped": True})
    return result


def _iter_unique_parameters(value: Any) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def add(param: Any) -> None:
        if not isinstance(param, torch.nn.Parameter):
            return
        key = id(param)
        if key in seen:
            return
        seen.add(key)
        params.append(param)

    def visit(obj: Any) -> None:
        if isinstance(obj, torch.nn.Parameter):
            add(obj)
        elif isinstance(obj, torch.nn.ModuleDict):
            for module in obj.values():
                visit(module)
        elif isinstance(obj, torch.nn.ParameterDict):
            for param in obj.values():
                visit(param)
        elif isinstance(obj, torch.nn.ParameterList):
            for param in obj:
                visit(param)
        elif isinstance(obj, torch.nn.Module):
            for param in obj.parameters(recurse=True):
                visit(param)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)

    visit(value)
    return params


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    return 0


def _optimizer_memory_summary(optimizer: Any, model: Any | None = None) -> dict[str, Any]:
    param_count = 0
    trainable_param_count = 0
    param_bytes = 0
    trainable_param_bytes = 0
    grad_bytes = 0
    cpu_param_bytes = 0
    cpu_grad_bytes = 0
    cuda_param_bytes = 0
    cuda_grad_bytes = 0

    params = list(model.parameters()) if isinstance(model, torch.nn.Module) else []
    seen_params: set[int] = set()
    for param in params:
        key = id(param)
        if key in seen_params:
            continue
        seen_params.add(key)
        param_count += int(param.numel())
        bytes_ = _tensor_nbytes(param)
        param_bytes += bytes_
        if param.device.type == "cpu":
            cpu_param_bytes += bytes_
        elif param.device.type == "cuda":
            cuda_param_bytes += bytes_
        if param.requires_grad:
            trainable_param_count += int(param.numel())
            trainable_param_bytes += bytes_
        if param.grad is not None:
            grad_nbytes = _tensor_nbytes(param.grad)
            grad_bytes += grad_nbytes
            if param.grad.device.type == "cpu":
                cpu_grad_bytes += grad_nbytes
            elif param.grad.device.type == "cuda":
                cuda_grad_bytes += grad_nbytes

    state_tensor_count = 0
    state_bytes = 0
    cpu_state_bytes = 0
    cuda_state_bytes = 0
    state = getattr(optimizer, "state", None)
    if isinstance(state, dict):
        for state_value in state.values():
            values = state_value.values() if isinstance(state_value, dict) else []
            for item in values:
                if isinstance(item, torch.Tensor):
                    state_tensor_count += 1
                    nbytes = _tensor_nbytes(item)
                    state_bytes += nbytes
                    if item.device.type == "cpu":
                        cpu_state_bytes += nbytes
                    elif item.device.type == "cuda":
                        cuda_state_bytes += nbytes

    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "param_bytes": param_bytes,
        "trainable_param_bytes": trainable_param_bytes,
        "grad_bytes": grad_bytes,
        "cpu_param_bytes": cpu_param_bytes,
        "cpu_grad_bytes": cpu_grad_bytes,
        "cuda_param_bytes": cuda_param_bytes,
        "cuda_grad_bytes": cuda_grad_bytes,
        "optimizer_state_tensor_count": state_tensor_count,
        "optimizer_state_bytes": state_bytes,
        "cpu_optimizer_state_bytes": cpu_state_bytes,
        "cuda_optimizer_state_bytes": cuda_state_bytes,
    }


def _sample_reduction_stats(sample: torch.Tensor) -> dict[str, Any]:
    sample_count = int(sample.numel())
    if sample_count == 0:
        return {
            "sample_count": 0,
            "sample_sum": 0.0,
            "sample_abs_sum": 0.0,
            "sample_weighted_sum": 0.0,
            "sample_max_abs": 0.0,
        }

    sample_sum = 0.0
    sample_abs_sum = 0.0
    sample_weighted_sum = 0.0
    sample_max_abs = 0.0
    chunk_size = 1_048_576
    for offset in range(0, sample_count, chunk_size):
        chunk = sample[offset : offset + chunk_size].float()
        sample_sum += float(chunk.sum().item())
        abs_chunk = chunk.abs()
        sample_abs_sum += float(abs_chunk.sum().item())
        sample_max_abs = max(sample_max_abs, float(abs_chunk.max().item()))
        weights = torch.arange(offset + 1, offset + 1 + int(chunk.numel()), device=chunk.device, dtype=chunk.dtype)
        sample_weighted_sum += float((chunk * weights).sum().item())
    return {
        "sample_count": sample_count,
        "sample_sum": sample_sum,
        "sample_abs_sum": sample_abs_sum,
        "sample_weighted_sum": sample_weighted_sum,
        "sample_max_abs": sample_max_abs,
    }


def _sample_tensor_stats(tensor: torch.Tensor, *, max_samples: int = 4096) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    numel = int(flat.numel())
    if numel == 0:
        return {
            "numel": 0,
            "max_samples": "all" if max_samples <= 0 else max_samples,
            "sample_count": 0,
            "sample_all_elements": True,
            "sample_sum": 0.0,
            "sample_abs_sum": 0.0,
            "sample_weighted_sum": 0.0,
            "sample_max_abs": 0.0,
        }
    if max_samples <= 0 or numel <= max_samples:
        sample = flat
        sample_all_elements = True
    else:
        if max_samples == 1:
            indices = torch.tensor([numel - 1], device=flat.device, dtype=torch.long)
        else:
            indices = torch.arange(max_samples, device=flat.device, dtype=torch.long)
            indices = torch.div(indices * (numel - 1), max_samples - 1, rounding_mode="floor")
            indices[-1] = numel - 1
        sample = flat[indices]
        sample_all_elements = False
    stats = _sample_reduction_stats(sample)
    return {
        "numel": numel,
        "max_samples": "all" if max_samples <= 0 else max_samples,
        "sample_all_elements": sample_all_elements,
        **stats,
    }


def _kt_fused_lora_update_snapshot(
    model: Any | None, *, max_tensors: int | None = None, max_elements: int | None = None
) -> dict[str, Any]:
    if not isinstance(model, torch.nn.Module):
        return {"available": False, "reason": "model unavailable"}
    if max_tensors is None:
        max_tensors = _kt_lora_health_max_tensors()
    if max_elements is None:
        max_elements = _kt_lora_health_max_elements()
    limit = int(max_tensors)
    element_limit = int(max_elements)
    unlimited = limit <= 0
    base_model = None
    if hasattr(model, "get_base_model"):
        try:
            base_model = model.get_base_model()
        except Exception:
            base_model = None
    wrappers = _find_kt_wrappers(model, base_model)
    if wrappers is None:
        return {"available": False, "reason": "KT wrappers unavailable"}

    rows: list[dict[str, Any]] = []
    total_fused_tensors = 0
    for layer_index, layer in enumerate(wrappers):
        params = _iter_unique_parameters(getattr(layer, "_fused_expert_lora_params", None))
        total_fused_tensors += len(params)
        for param_index, param in enumerate(params):
            if not unlimited and len(rows) >= limit:
                break
            row = {
                "layer_index": layer_index,
                "layer_idx": getattr(layer, "layer_idx", layer_index),
                "param_index": param_index,
                "shape": list(param.shape),
                "device": str(param.device),
                "requires_grad": bool(param.requires_grad),
                "param": _sample_tensor_stats(param, max_samples=element_limit),
                "grad": _sample_tensor_stats(param.grad, max_samples=element_limit)
                if isinstance(param.grad, torch.Tensor)
                else None,
            }
            rows.append(row)
        if not unlimited and len(rows) >= limit:
            continue

    exhaustive = total_fused_tensors > 0 and len(rows) == total_fused_tensors
    exhaustive_elements = exhaustive and all(
        bool((row.get("param") or {}).get("sample_all_elements"))
        and (row.get("grad") is None or bool((row.get("grad") or {}).get("sample_all_elements")))
        for row in rows
    )
    return {
        "available": True,
        "sampled_tensors": len(rows),
        "total_fused_tensors": total_fused_tensors,
        "requested_max_tensors": "all" if unlimited else limit,
        "requested_max_elements": "all" if element_limit <= 0 else element_limit,
        "exhaustive": exhaustive,
        "exhaustive_elements": exhaustive_elements,
        "rows": rows,
    }


def _kt_fused_lora_update_health(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        before_reason = before.get("reason")
        after_reason = after.get("reason")
        return {
            "available": False,
            "passed": False,
            "reason": f"before unavailable: {before_reason}; after unavailable: {after_reason}",
            "before_reason": before_reason,
            "after_reason": after_reason,
        }

    def key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (row.get("layer_idx"), row.get("layer_index"), row.get("param_index"))

    before_row_items = [row for row in before.get("rows", []) if isinstance(row, dict)]
    after_row_items = [row for row in after.get("rows", []) if isinstance(row, dict)]
    before_keys = {key(row) for row in before_row_items}
    after_keys = {key(row) for row in after_row_items}
    missing_after_keys = sorted(str(item) for item in (before_keys - after_keys))
    unexpected_after_keys = sorted(str(item) for item in (after_keys - before_keys))
    after_rows = {key(row): row for row in after_row_items}
    changed_tensors = 0
    grad_nonzero_tensors = 0
    updated_grad_tensors = 0
    grad_nonzero_unchanged_tensors = 0
    compared_tensors = 0
    rows: list[dict[str, Any]] = []
    for before_row in before_row_items:
        after_row = after_rows.get(key(before_row))
        if after_row is None:
            continue
        compared_tensors += 1
        before_param = before_row.get("param") or {}
        after_param = after_row.get("param") or {}
        before_grad = before_row.get("grad") or {}
        param_abs_delta = abs(float(after_param.get("sample_sum", 0.0)) - float(before_param.get("sample_sum", 0.0)))
        param_abs_sum_delta = abs(
            float(after_param.get("sample_abs_sum", 0.0)) - float(before_param.get("sample_abs_sum", 0.0))
        )
        param_weighted_sum_delta = abs(
            float(after_param.get("sample_weighted_sum", 0.0))
            - float(before_param.get("sample_weighted_sum", 0.0))
        )
        param_max_abs_delta = abs(
            float(after_param.get("sample_max_abs", 0.0)) - float(before_param.get("sample_max_abs", 0.0))
        )
        grad_abs_sum = float(before_grad.get("sample_abs_sum", 0.0))
        changed = (
            param_abs_delta > 0.0
            or param_abs_sum_delta > 0.0
            or param_weighted_sum_delta > 0.0
            or param_max_abs_delta > 0.0
        )
        grad_nonzero = grad_abs_sum > 0.0
        changed_tensors += int(changed)
        grad_nonzero_tensors += int(grad_nonzero)
        updated_grad_tensors += int(grad_nonzero and changed)
        grad_nonzero_unchanged_tensors += int(grad_nonzero and not changed)
        rows.append(
            {
                "layer_idx": before_row.get("layer_idx"),
                "param_index": before_row.get("param_index"),
                "shape": before_row.get("shape"),
                "grad_sample_abs_sum_before_step": grad_abs_sum,
                "param_sample_sum_delta": param_abs_delta,
                "param_sample_abs_sum_delta": param_abs_sum_delta,
                "param_sample_weighted_sum_delta": param_weighted_sum_delta,
                "param_sample_max_abs_delta": param_max_abs_delta,
                "grad_nonzero_before_step": grad_nonzero,
                "param_changed_after_step": changed,
                "nonzero_grad_changed_after_step": grad_nonzero and changed,
            }
        )

    sampled_tensors = int(before.get("sampled_tensors", len(before_row_items)) or 0)
    total_fused_tensors = int(before.get("total_fused_tensors", 0) or 0)
    after_sampled_tensors = int(after.get("sampled_tensors", len(after_row_items)) or 0)
    after_total_fused_tensors = int(after.get("total_fused_tensors", 0) or 0)
    exhaustive = bool(before.get("exhaustive")) and bool(after.get("exhaustive"))
    exhaustive_elements = bool(before.get("exhaustive_elements")) and bool(after.get("exhaustive_elements"))

    if missing_after_keys or unexpected_after_keys:
        passed = False
        reason = (
            "sampled fused LoRA tensor set changed between before/after optimizer snapshots "
            f"(missing_after={missing_after_keys[:3]}, unexpected_after={unexpected_after_keys[:3]})"
        )
    elif sampled_tensors != after_sampled_tensors or total_fused_tensors != after_total_fused_tensors:
        passed = False
        reason = (
            "fused LoRA tensor counts changed between before/after optimizer snapshots "
            f"(sampled {sampled_tensors}->{after_sampled_tensors}, total {total_fused_tensors}->{after_total_fused_tensors})"
        )
    elif exhaustive and compared_tensors != sampled_tensors:
        passed = False
        reason = f"exhaustive fused LoRA health compared {compared_tensors} of {sampled_tensors} tensors"
    elif compared_tensors <= 0:
        passed = False
        reason = "no sampled fused LoRA tensors were comparable"
    elif grad_nonzero_tensors <= 0:
        passed = False
        reason = "no sampled fused LoRA tensors had nonzero gradients before optimizer step"
    elif grad_nonzero_unchanged_tensors > 0:
        passed = False
        reason = "one or more sampled nonzero-gradient fused LoRA tensors did not change after optimizer step"
    else:
        passed = True
        if exhaustive_elements:
            reason = "all nonzero-gradient fused LoRA tensors changed after optimizer step with full-element checksums"
        elif exhaustive:
            reason = "all nonzero-gradient fused LoRA tensors changed after optimizer step"
        else:
            reason = "all sampled nonzero-gradient fused LoRA tensors changed after optimizer step"

    return {
        "available": True,
        "passed": passed,
        "reason": reason,
        "sampled_tensors": sampled_tensors,
        "total_fused_tensors": total_fused_tensors,
        "after_sampled_tensors": after_sampled_tensors,
        "after_total_fused_tensors": after_total_fused_tensors,
        "requested_max_tensors": before.get("requested_max_tensors"),
        "requested_max_elements": before.get("requested_max_elements"),
        "exhaustive": exhaustive,
        "exhaustive_elements": exhaustive_elements,
        "compared_tensors": compared_tensors,
        "missing_after_tensors": len(missing_after_keys),
        "unexpected_after_tensors": len(unexpected_after_keys),
        "grad_nonzero_tensors": grad_nonzero_tensors,
        "changed_tensors": changed_tensors,
        "updated_grad_tensors": updated_grad_tensors,
        "grad_nonzero_unchanged_tensors": grad_nonzero_unchanged_tensors,
        "rows": rows,
    }


def _lora_counters_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    trainable_params = 0
    all_params = 0
    peft_lora_params = 0
    peft_expert_lora_tensors = 0
    peft_expert_lora_params = 0

    def is_peft_lora_name(name: str) -> bool:
        return ".lora_" in name or name.startswith("lora_") or "lora_A" in name or "lora_B" in name

    def is_expert_lora_name(name: str) -> bool:
        lowered = name.lower()
        if not is_peft_lora_name(name):
            return False
        expert_markers = (
            ".experts.",
            ".expert.",
            ".shared_expert",
            "shared_experts",
            "block_sparse_moe",
            "moe.experts",
        )
        return any(marker in lowered for marker in expert_markers)
    for name, param in model.named_parameters():
        numel = int(param.numel())
        all_params += numel
        if param.requires_grad:
            trainable_params += numel
        if is_peft_lora_name(name) and param.requires_grad:
            peft_lora_params += numel
            if is_expert_lora_name(name):
                peft_expert_lora_tensors += 1
                peft_expert_lora_params += numel

    qwen_moe_expert_modules = 0
    qwen_moe_expert_tensors = 0
    qwen_moe_expert_params = 0
    kt_peft_expert_modules = 0
    kt_peft_expert_tensors = 0
    kt_peft_expert_params = 0
    kt_fused_modules = 0
    kt_fused_tensors = 0
    kt_fused_params = 0
    kt_expert_modules = 0
    kt_expert_tensors = 0
    kt_expert_params = 0

    module_roots = [root for root in (model, base_model) if root is not None]
    seen_modules: set[int] = set()
    qwen_moe_layer_type: Any = ()
    try:
        from llamafactory.model.model_utils.fused_moe_lora import QwenSplitMoeExpertParamWrapper

        qwen_moe_layer_type = QwenSplitMoeExpertParamWrapper
    except Exception:
        qwen_moe_layer_type = ()
    for root in module_roots:
        for module in root.modules():
            module_key = id(module)
            if module_key in seen_modules:
                continue
            seen_modules.add(module_key)

            if qwen_moe_layer_type and isinstance(module, qwen_moe_layer_type):
                qwen_params = [
                    param
                    for layer_name in getattr(module, "adapter_layer_names", ())
                    for param in getattr(module, layer_name).values()
                    if isinstance(param, torch.nn.Parameter) and param.requires_grad
                ]
                if qwen_params:
                    qwen_moe_expert_modules += 1
                    qwen_moe_expert_tensors += len(qwen_params)
                    qwen_moe_expert_params += sum(int(param.numel()) for param in qwen_params)

    wrappers = _find_kt_wrappers(model, base_model) or []
    seen_kt_params: set[int] = set()

    def add_kt_param(bucket: list[torch.nn.Parameter], param: Any) -> None:
        if isinstance(param, torch.nn.Parameter) and param.requires_grad:
            bucket.append(param)

    for layer in wrappers:
        layer_peft_params: list[torch.nn.Parameter] = []
        peft_lora_modules = getattr(layer, "_peft_lora_modules", None)
        if isinstance(peft_lora_modules, dict):
            for expert_loras in peft_lora_modules.values():
                if not isinstance(expert_loras, dict):
                    continue
                for pair in expert_loras.values():
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    lora_a, lora_b = pair
                    add_kt_param(layer_peft_params, getattr(lora_a, "weight", None))
                    add_kt_param(layer_peft_params, getattr(lora_b, "weight", None))

        layer_fused_params = _iter_unique_parameters(getattr(layer, "_fused_expert_lora_params", None))

        def unique_new(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
            unique: list[torch.nn.Parameter] = []
            for param in params:
                key = id(param)
                if key in seen_kt_params:
                    continue
                seen_kt_params.add(key)
                unique.append(param)
            return unique

        layer_peft_params = unique_new(layer_peft_params)
        layer_fused_params = unique_new(layer_fused_params)
        layer_kt_params = layer_peft_params + layer_fused_params

        if layer_peft_params:
            kt_peft_expert_modules += 1
            kt_peft_expert_tensors += len(layer_peft_params)
            kt_peft_expert_params += sum(int(param.numel()) for param in layer_peft_params)
        if layer_fused_params:
            kt_fused_modules += 1
            kt_fused_tensors += len(layer_fused_params)
            kt_fused_params += sum(int(param.numel()) for param in layer_fused_params)
        if layer_kt_params:
            kt_expert_modules += 1
            kt_expert_tensors += len(layer_kt_params)
            kt_expert_params += sum(int(param.numel()) for param in layer_kt_params)

    result = {
        "available": True,
        "trainable_parameters": trainable_params,
        "all_parameters": all_params,
        "peft_lora_parameters": peft_lora_params,
        "peft_expert_lora_tensors": peft_expert_lora_tensors,
        "peft_expert_lora_parameters": peft_expert_lora_params,
        "qwen_moe_expert_lora_modules": qwen_moe_expert_modules,
        "qwen_moe_expert_lora_tensors": qwen_moe_expert_tensors,
        "qwen_moe_expert_lora_parameters": qwen_moe_expert_params,
        "kt_peft_expert_lora_modules": kt_peft_expert_modules,
        "kt_peft_expert_lora_tensors": kt_peft_expert_tensors,
        "kt_peft_expert_lora_parameters": kt_peft_expert_params,
        "kt_fused_expert_lora_modules": kt_fused_modules,
        "kt_fused_expert_lora_tensors": kt_fused_tensors,
        "kt_fused_expert_lora_parameters": kt_fused_params,
        "kt_expert_lora_modules": kt_expert_modules,
        "kt_expert_lora_tensors": kt_expert_tensors,
        "kt_expert_lora_parameters": kt_expert_params,
    }
    sidecar = _lora_surface_sidecar()
    if sidecar is not None and (_safe_int(sidecar.get("trainable_parameters")) or 0) > trainable_params:
        for key in (
            "trainable_parameters",
            "all_parameters",
            "peft_lora_parameters",
            "peft_expert_lora_tensors",
            "peft_expert_lora_parameters",
            "qwen_moe_expert_lora_modules",
            "qwen_moe_expert_lora_tensors",
            "qwen_moe_expert_lora_parameters",
        ):
            if key in sidecar:
                result[key] = sidecar[key]
        result["sidecar_source"] = sidecar.get("source", "lora_surface_sidecar")
    return result


def _int_dict_sum(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(_safe_int(item) or 0 for item in value.values())


def _merge_int_dict(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, raw_count in value.items():
        count = _safe_int(raw_count) or 0
        if count:
            target[str(key)] = target.get(str(key), 0) + count


def _activation_stage_bytes(stats: dict[str, Any]) -> int:
    by_tag = stats.get("stage_bytes_by_tag")
    if isinstance(by_tag, dict):
        return _int_dict_sum(by_tag)
    return _safe_int(stats.get("staged_bytes")) or 0


def _activation_transfer_weight(stats: dict[str, Any]) -> int:
    return (
        (_safe_int(stats.get("offloaded_bytes")) or 0)
        + _activation_stage_bytes(stats)
        + (_safe_int(stats.get("num_offloads")) or 0)
        + (_safe_int(stats.get("num_stages")) or 0)
    )


def _activation_source_context_key(row: dict[str, Any]) -> str:
    source = str(row.get("profile_prefix") or row.get("name") or "")
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        suffix = f".{leaf}"
        if source.endswith(suffix):
            return source[: -len(suffix)]
    return source


def _activation_transfer_totals(stats_rows: list[dict[str, Any]]) -> dict[str, Any]:
    offload_bytes_by_tag: dict[str, int] = {}
    stage_bytes_by_tag: dict[str, int] = {}
    total_offload_calls = 0
    total_stage_calls = 0
    total_offloaded_bytes = 0
    total_staged_bytes = 0

    for stats in stats_rows:
        total_offload_calls += _safe_int(stats.get("num_offloads")) or 0
        total_stage_calls += _safe_int(stats.get("num_stages")) or 0
        total_offloaded_bytes += _safe_int(stats.get("offloaded_bytes")) or 0
        total_staged_bytes += _activation_stage_bytes(stats)
        _merge_int_dict(offload_bytes_by_tag, stats.get("offload_bytes_by_tag"))
        _merge_int_dict(stage_bytes_by_tag, stats.get("stage_bytes_by_tag"))

    return {
        "total_d2h_offload_copy_calls": total_offload_calls,
        "total_h2d_stage_copy_calls": total_stage_calls,
        "total_d2h_offloaded_bytes": total_offloaded_bytes,
        "total_h2d_staged_bytes": total_staged_bytes,
        "total_forward_offload_copy_calls": total_offload_calls,
        "total_backward_stage_copy_calls": total_stage_calls,
        "total_forward_offloaded_bytes": total_offloaded_bytes,
        "total_backward_staged_bytes": total_staged_bytes,
        "total_activation_transfer_calls": total_offload_calls + total_stage_calls,
        "total_activation_transfer_bytes": total_offloaded_bytes + total_staged_bytes,
        "offload_bytes_by_tag": offload_bytes_by_tag,
        "stage_bytes_by_tag": stage_bytes_by_tag,
    }


def _activation_offload_counters_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    rows: list[dict[str, Any]] = []
    source_context_by_key: dict[str, dict[str, Any]] = {}
    seen_modules: set[int] = set()
    module_roots = [root for root in (model, base_model) if root is not None]
    for root in module_roots:
        for name, module in root.named_modules():
            module_key = id(module)
            if module_key in seen_modules:
                continue
            seen_modules.add(module_key)

            stats = getattr(module, "_last_activation_offload_stats", None)
            pre_release_stats = getattr(module, "_last_activation_offload_stats_pre_release", None)
            execution_stats = getattr(module, "stats", None)
            if not isinstance(stats, dict) or not stats:
                continue

            row: dict[str, Any] = {
                "name": name,
                "class": module.__class__.__name__,
                "profile_prefix": getattr(module, "profile_prefix", ""),
            }
            if isinstance(stats, dict):
                row["activation_offload_stats"] = dict(stats)
                row["cpu_owned_bytes"] = _safe_int(stats.get("cpu_owned_bytes")) or 0
                row["cpu_live_bytes"] = _safe_int(stats.get("cpu_live_bytes")) or 0
                row["cpu_peak_bytes_live"] = _safe_int(stats.get("cpu_peak_bytes_live")) or 0
                row["max_stage_bytes_live"] = _safe_int(stats.get("max_stage_bytes_live")) or 0
                row["cpu_pool_cached_bytes"] = _safe_int(stats.get("cpu_pool_cached_bytes")) or 0
                row["cpu_pool_limit_bytes"] = _safe_int(stats.get("cpu_pool_limit_bytes")) or 0
                row["pre_final_cleanup_cpu_owned_bytes"] = (
                    _safe_int(stats.get("pre_final_cleanup_cpu_owned_bytes")) or 0
                )
                row["final_cleanup_released_bytes"] = _safe_int(stats.get("final_cleanup_released_bytes")) or 0
            if isinstance(pre_release_stats, dict):
                row["activation_offload_stats_pre_release"] = dict(pre_release_stats)

            as_dict = getattr(execution_stats, "as_dict", None)
            if callable(as_dict):
                try:
                    execution_stats_dict = as_dict()
                except Exception as exc:
                    row["execution_stats_error"] = repr(exc)
                else:
                    if isinstance(execution_stats_dict, dict):
                        row["execution_stats"] = execution_stats_dict
            rows.append(row)

            source_context = stats.get("source_context")
            if isinstance(source_context, dict) and source_context:
                source_key = _activation_source_context_key(row)
                previous = source_context_by_key.get(source_key)
                if previous is None or _activation_transfer_weight(source_context) >= _activation_transfer_weight(previous):
                    source_context_by_key[source_key] = dict(source_context)

    row_stats = [
        row["activation_offload_stats"]
        for row in rows
        if isinstance(row.get("activation_offload_stats"), dict)
    ]
    transfer_totals = _activation_transfer_totals(row_stats + list(source_context_by_key.values()))

    return {
        "available": bool(rows),
        "module_count": len(rows),
        "rows": rows,
        "source_context_count": len(source_context_by_key),
        **transfer_totals,
        "total_cpu_owned_bytes": sum(int(row.get("cpu_owned_bytes", 0) or 0) for row in rows),
        "total_cpu_live_bytes": sum(int(row.get("cpu_live_bytes", 0) or 0) for row in rows),
        "max_cpu_peak_bytes_live": max([int(row.get("cpu_peak_bytes_live", 0) or 0) for row in rows] or [0]),
        "max_stage_bytes_live": max([int(row.get("max_stage_bytes_live", 0) or 0) for row in rows] or [0]),
        "max_cpu_pool_cached_bytes": max([int(row.get("cpu_pool_cached_bytes", 0) or 0) for row in rows] or [0]),
        "max_cpu_pool_limit_bytes": max([int(row.get("cpu_pool_limit_bytes", 0) or 0) for row in rows] or [0]),
        "total_final_cleanup_released_bytes": sum(
            int(row.get("final_cleanup_released_bytes", 0) or 0) for row in rows
        ),
    }


def _asym_execution_stats_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    stats = None
    source = ""
    for label, candidate in (("model", model), ("base_model", base_model)):
        if candidate is None:
            continue
        candidate_stats = getattr(candidate, "_asym_execution_stats", None)
        if candidate_stats is not None:
            stats = candidate_stats
            source = label
            break
    if stats is None:
        return {"available": False, "reason": "model has no _asym_execution_stats"}

    as_dict = getattr(stats, "as_dict", None)
    if not callable(as_dict):
        return {"available": False, "reason": "_asym_execution_stats has no as_dict"}
    try:
        data = as_dict()
    except Exception as exc:
        return {"available": False, "reason": f"as_dict failed: {exc!r}"}
    if not isinstance(data, dict):
        return {"available": False, "reason": "as_dict did not return a dict"}
    return {"available": True, "source": source, **data}


def _asym_liger_lm_head_bridge_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"enabled": False, "reason": "model hook did not capture a model"}

    try:
        from asym_gemm.integrations.liger_loss import asym_liger_lm_head_bridge_metadata
    except Exception as exc:
        return {"enabled": False, "reason": f"metadata import failed: {exc!r}"}

    for label, candidate in (("model", model), ("base_model", base_model)):
        if candidate is None:
            continue
        try:
            metadata = asym_liger_lm_head_bridge_metadata(candidate)
        except Exception as exc:
            return {"enabled": False, "source": label, "reason": f"metadata failed: {exc!r}"}
        if metadata.get("enabled"):
            return {"source": label, **metadata}

    try:
        metadata = asym_liger_lm_head_bridge_metadata(model)
    except Exception as exc:
        return {"enabled": False, "reason": f"metadata failed: {exc!r}"}
    return {"source": "model", **metadata}


@dataclass
class StageRecord:
    milliseconds: float
    allocated_start_bytes: int
    allocated_end_bytes: int
    reserved_start_bytes: int
    reserved_end_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    global_peak_allocated_after_bytes: int
    global_peak_reserved_after_bytes: int
    process_rss_start_bytes: int
    process_rss_end_bytes: int
    process_rss_peak_start_bytes: int
    process_rss_peak_end_bytes: int
    process_virtual_memory_start_bytes: int
    process_virtual_memory_end_bytes: int

    @property
    def allocated_delta_bytes(self) -> int:
        return self.allocated_end_bytes - self.allocated_start_bytes

    @property
    def reserved_delta_bytes(self) -> int:
        return self.reserved_end_bytes - self.reserved_start_bytes

    @property
    def peak_allocated_delta_bytes(self) -> int:
        return self.peak_allocated_bytes - self.allocated_start_bytes

    @property
    def peak_reserved_delta_bytes(self) -> int:
        return self.peak_reserved_bytes - self.reserved_start_bytes

    @property
    def reserved_unallocated_bytes(self) -> int:
        return max(0, self.peak_reserved_bytes - self.peak_allocated_bytes)

    @property
    def process_rss_delta_bytes(self) -> int:
        return self.process_rss_end_bytes - self.process_rss_start_bytes

    @property
    def process_virtual_memory_delta_bytes(self) -> int:
        return self.process_virtual_memory_end_bytes - self.process_virtual_memory_start_bytes


@dataclass
class LFProfileRecorder:
    config: dict[str, Any]
    measure_memory: bool = True
    reset_stage_peak_stats: bool = True
    records: dict[str, list[StageRecord]] = field(default_factory=dict)
    global_peak_allocated_bytes: int = 0
    global_peak_reserved_bytes: int = 0

    @contextmanager
    def stage(self, name: str, *, sync: bool = False) -> Iterator[None]:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        process_memory_start = _process_memory_snapshot()
        cuda_available = self.measure_memory and torch.cuda.is_available()
        allocated_start = reserved_start = 0
        if cuda_available:
            allocated_start = int(torch.cuda.memory_allocated())
            reserved_start = int(torch.cuda.memory_reserved())
            if self.reset_stage_peak_stats:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except RuntimeError:
                    pass
        try:
            with prof_range(name):
                yield
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            process_memory_end = _process_memory_snapshot()
            allocated_end = reserved_end = peak_allocated = peak_reserved = 0
            global_peak_allocated = global_peak_reserved = 0
            if cuda_available:
                allocated_end = int(torch.cuda.memory_allocated())
                reserved_end = int(torch.cuda.memory_reserved())
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
                global_peak_allocated = max(peak_allocated, allocated_end)
                global_peak_reserved = max(peak_reserved, reserved_end)
                self.global_peak_allocated_bytes = max(self.global_peak_allocated_bytes, global_peak_allocated)
                self.global_peak_reserved_bytes = max(self.global_peak_reserved_bytes, global_peak_reserved)
            self.records.setdefault(name, []).append(
                StageRecord(
                    milliseconds=elapsed_ms,
                    allocated_start_bytes=allocated_start,
                    allocated_end_bytes=allocated_end,
                    reserved_start_bytes=reserved_start,
                    reserved_end_bytes=reserved_end,
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    global_peak_allocated_after_bytes=global_peak_allocated,
                    global_peak_reserved_after_bytes=global_peak_reserved,
                    process_rss_start_bytes=_process_memory_value(process_memory_start, "rss_bytes"),
                    process_rss_end_bytes=_process_memory_value(process_memory_end, "rss_bytes"),
                    process_rss_peak_start_bytes=_process_memory_value(process_memory_start, "rss_peak_bytes"),
                    process_rss_peak_end_bytes=_process_memory_value(process_memory_end, "rss_peak_bytes"),
                    process_virtual_memory_start_bytes=_process_memory_value(
                        process_memory_start, "virtual_memory_bytes"
                    ),
                    process_virtual_memory_end_bytes=_process_memory_value(process_memory_end, "virtual_memory_bytes"),
                )
            )

    def _stage_total_ms(self, name: str) -> float:
        records = self._measured_records(name)
        if not records:
            return 0.0
        return sum(record.milliseconds for record in records) / float(len(records))

    def _warmup_steps(self) -> int:
        value = _safe_int(self.config.get("warmup_steps"))
        return max(value or 0, 0)

    def _measured_records(self, name: str) -> list[StageRecord]:
        records = self.records.get(name, [])
        warmup_steps = min(self._warmup_steps(), len(records))
        return records[warmup_steps:]

    def _stage_memory_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in sorted(self.records):
            raw_records = self.records.get(name, [])
            records = self._measured_records(name)
            if not records:
                continue

            def avg(field_name: str) -> float:
                return sum(float(getattr(record, field_name)) for record in records) / float(len(records))

            rows.append(
                {
                    "name": name,
                    "samples": len(records),
                    "raw_samples": len(raw_records),
                    "warmup_samples_skipped": len(raw_records) - len(records),
                    "avg_allocated_start_bytes": avg("allocated_start_bytes"),
                    "avg_allocated_end_bytes": avg("allocated_end_bytes"),
                    "avg_allocated_delta_bytes": avg("allocated_delta_bytes"),
                    "avg_reserved_start_bytes": avg("reserved_start_bytes"),
                    "avg_reserved_end_bytes": avg("reserved_end_bytes"),
                    "avg_reserved_delta_bytes": avg("reserved_delta_bytes"),
                    "avg_peak_allocated_bytes": avg("peak_allocated_bytes"),
                    "max_peak_allocated_bytes": max(record.peak_allocated_bytes for record in records),
                    "avg_peak_allocated_delta_bytes": avg("peak_allocated_delta_bytes"),
                    "avg_peak_reserved_bytes": avg("peak_reserved_bytes"),
                    "max_peak_reserved_bytes": max(record.peak_reserved_bytes for record in records),
                    "avg_peak_reserved_delta_bytes": avg("peak_reserved_delta_bytes"),
                    "max_global_peak_allocated_after_bytes": max(
                        record.global_peak_allocated_after_bytes for record in records
                    ),
                    "max_global_peak_reserved_after_bytes": max(
                        record.global_peak_reserved_after_bytes for record in records
                    ),
                    "avg_reserved_unallocated_bytes": avg("reserved_unallocated_bytes"),
                    "avg_process_rss_start_bytes": avg("process_rss_start_bytes"),
                    "avg_process_rss_end_bytes": avg("process_rss_end_bytes"),
                    "avg_process_rss_delta_bytes": avg("process_rss_delta_bytes"),
                    "max_process_rss_end_bytes": max(record.process_rss_end_bytes for record in records),
                    "max_process_rss_peak_end_bytes": max(record.process_rss_peak_end_bytes for record in records),
                    "avg_process_virtual_memory_start_bytes": avg("process_virtual_memory_start_bytes"),
                    "avg_process_virtual_memory_end_bytes": avg("process_virtual_memory_end_bytes"),
                    "avg_process_virtual_memory_delta_bytes": avg("process_virtual_memory_delta_bytes"),
                    "max_process_virtual_memory_end_bytes": max(
                        record.process_virtual_memory_end_bytes for record in records
                    ),
                }
            )
        return rows

    def _step_sample_rows(
        self,
        losses: list[dict[str, Any]],
        trainer_records: list[dict[str, Any]],
        heartbeat_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        loss_by_step: dict[int, float] = {}
        for loss_row in losses:
            if not isinstance(loss_row, dict):
                continue
            step = _safe_int(loss_row.get("raw_step", loss_row.get("step")))
            loss = _safe_float(loss_row.get("loss"))
            if step is not None and loss is not None:
                loss_by_step[step] = loss
        elapsed_by_step: dict[int, float] = {}
        for record in trainer_records:
            if not isinstance(record, dict):
                continue
            step = _safe_int(record.get("current_steps", record.get("step")))
            elapsed_seconds = _duration_to_seconds(record.get("elapsed_time"))
            if step is None or step <= 0 or elapsed_seconds is None:
                continue
            elapsed_by_step[step] = max(elapsed_seconds, elapsed_by_step.get(step, 0.0))
        heartbeat_by_step = {
            int(row["raw_step"]): row
            for row in (heartbeat_rows or [])
            if row.get("raw_step") is not None
        }

        def add_stage(row: dict[str, Any], prefix: str, record: StageRecord | None) -> None:
            if record is None:
                return
            row.update(
                {
                    f"{prefix}_milliseconds": record.milliseconds,
                    f"{prefix}_allocated_start_bytes": record.allocated_start_bytes,
                    f"{prefix}_allocated_end_bytes": record.allocated_end_bytes,
                    f"{prefix}_allocated_delta_bytes": record.allocated_delta_bytes,
                    f"{prefix}_reserved_start_bytes": record.reserved_start_bytes,
                    f"{prefix}_reserved_end_bytes": record.reserved_end_bytes,
                    f"{prefix}_reserved_delta_bytes": record.reserved_delta_bytes,
                    f"{prefix}_peak_allocated_bytes": record.peak_allocated_bytes,
                    f"{prefix}_peak_allocated_delta_bytes": record.peak_allocated_delta_bytes,
                    f"{prefix}_peak_reserved_bytes": record.peak_reserved_bytes,
                    f"{prefix}_peak_reserved_delta_bytes": record.peak_reserved_delta_bytes,
                    f"{prefix}_global_peak_allocated_after_bytes": record.global_peak_allocated_after_bytes,
                    f"{prefix}_global_peak_reserved_after_bytes": record.global_peak_reserved_after_bytes,
                    f"{prefix}_reserved_unallocated_bytes": record.reserved_unallocated_bytes,
                    f"{prefix}_process_rss_start_bytes": record.process_rss_start_bytes,
                    f"{prefix}_process_rss_end_bytes": record.process_rss_end_bytes,
                    f"{prefix}_process_rss_delta_bytes": record.process_rss_delta_bytes,
                    f"{prefix}_process_rss_peak_end_bytes": record.process_rss_peak_end_bytes,
                    f"{prefix}_process_virtual_memory_start_bytes": record.process_virtual_memory_start_bytes,
                    f"{prefix}_process_virtual_memory_end_bytes": record.process_virtual_memory_end_bytes,
                    f"{prefix}_process_virtual_memory_delta_bytes": record.process_virtual_memory_delta_bytes,
                }
            )

        forward_records = self.records.get("step.forward", [])
        backward_records = self.records.get("step.backward", [])
        training_step_records = self.records.get("lf.step.total", [])
        max_elapsed_step = max(elapsed_by_step) if elapsed_by_step else 0
        max_heartbeat_step = max(heartbeat_by_step) if heartbeat_by_step else 0
        sample_count = max(
            len(forward_records),
            len(backward_records),
            len(training_step_records),
            max_elapsed_step,
            max_heartbeat_step,
        )
        rows: list[dict[str, Any]] = []
        warmup_steps = self._warmup_steps()
        for index in range(sample_count):
            raw_step = index + 1
            measured_step = max(raw_step - warmup_steps, 0)
            is_warmup = raw_step <= warmup_steps
            forward = forward_records[index] if index < len(forward_records) else None
            backward = backward_records[index] if index < len(backward_records) else None
            training_step = training_step_records[index] if index < len(training_step_records) else None
            row: dict[str, Any] = {
                "step": measured_step if measured_step > 0 else raw_step,
                "raw_step": raw_step,
                "measured_step": measured_step,
                "is_warmup": is_warmup,
            }
            if raw_step in loss_by_step:
                row["loss"] = loss_by_step[raw_step]
            add_stage(row, "forward", forward)
            add_stage(row, "backward", backward)
            add_stage(row, "training_step", training_step)
            forward_backward_ms = sum(
                record.milliseconds for record in (forward, backward) if record is not None
            )
            row["forward_backward_milliseconds"] = forward_backward_ms
            row["profiled_training_step_milliseconds"] = training_step.milliseconds if training_step is not None else None
            heartbeat_row = heartbeat_by_step.get(raw_step)
            elapsed_seconds = elapsed_by_step.get(raw_step)
            previous_elapsed_seconds = 0.0 if raw_step == 1 else elapsed_by_step.get(raw_step - 1)
            if heartbeat_row is not None:
                trainer_step_ms = float(heartbeat_row["trainer_e2e_step_milliseconds"])
                row.update(heartbeat_row)
                row["optimizer_update_side_milliseconds"] = trainer_step_ms - forward_backward_ms
                row["step_milliseconds"] = trainer_step_ms
                row["step_milliseconds_source"] = "heartbeat_dataloader_interval"
            elif elapsed_seconds is not None and previous_elapsed_seconds is not None:
                trainer_step_ms = max(0.0, elapsed_seconds - previous_elapsed_seconds) * 1000.0
                row["trainer_elapsed_seconds"] = elapsed_seconds
                row["trainer_e2e_step_milliseconds"] = trainer_step_ms
                row["optimizer_update_side_milliseconds"] = trainer_step_ms - forward_backward_ms
                row["step_milliseconds"] = trainer_step_ms
                row["step_milliseconds_source"] = "trainer_log_elapsed_time"
            else:
                row["step_milliseconds"] = forward_backward_ms
                row["step_milliseconds_source"] = "forward_plus_backward_only"
            row["peak_allocated_hbm_bytes"] = max(
                [record.peak_allocated_bytes for record in (forward, backward) if record is not None] or [0]
            )
            row["peak_reserved_hbm_bytes"] = max(
                [record.peak_reserved_bytes for record in (forward, backward) if record is not None] or [0]
            )
            row["reserved_unallocated_bytes"] = max(
                0, int(row["peak_reserved_hbm_bytes"]) - int(row["peak_allocated_hbm_bytes"])
            )
            rss_values = [record.process_rss_end_bytes for record in (forward, backward) if record is not None]
            peak_rss_values = [record.process_rss_peak_end_bytes for record in (forward, backward) if record is not None]
            row["process_rss_end_bytes"] = max(rss_values or [0])
            row["process_rss_peak_bytes"] = max(peak_rss_values or [0])
            rows.append(row)
        return rows

    def _loss_rows(self, trainer_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        warmup_steps = self._warmup_steps()
        losses: list[dict[str, Any]] = []
        for record in trainer_records:
            if record.get("loss") is None:
                continue
            raw_step = _safe_int(record.get("current_steps", record.get("step")))
            if raw_step is None:
                raw_step = len(losses) + 1
            measured_step = max(raw_step - warmup_steps, 0)
            losses.append(
                {
                    "step": measured_step if measured_step > 0 else raw_step,
                    "raw_step": raw_step,
                    "measured_step": measured_step,
                    "is_warmup": raw_step <= warmup_steps,
                    "loss": record.get("loss"),
                }
            )
        return losses

    def _stage_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in sorted(self.records):
            raw_records = self.records.get(name, [])
            records = self._measured_records(name)
            if raw_records:
                rows.append(
                    {
                        "name": name,
                        "milliseconds": self._stage_total_ms(name),
                        "samples": len(records),
                        "raw_samples": len(raw_records),
                        "warmup_samples_skipped": len(raw_records) - len(records),
                    }
                )
        return rows

    def report(self, trace_handle: LFTraceHandle | None = None) -> dict[str, Any]:
        forward_ms = self._stage_total_ms("step.forward")
        backward_ms = self._stage_total_ms("step.backward")
        stage_rows = self._stage_rows()
        output_dir = Path(str(self.config.get("output_dir", ""))) if self.config.get("output_dir") else None
        trainer_log = output_dir / "trainer_log.jsonl" if output_dir is not None else None
        trainer_records = _trainer_log_records(trainer_log) if trainer_log is not None else []
        heartbeat_path = Path(str(self.config.get("heartbeat_jsonl", ""))) if self.config.get("heartbeat_jsonl") else None
        heartbeat_records = _heartbeat_records(heartbeat_path)
        heartbeat_rows = _heartbeat_step_interval_rows(heartbeat_records)
        trainer_timing = _heartbeat_timing_summary(heartbeat_records, self.config, heartbeat_rows) or _trainer_timing_summary(
            trainer_records, self.config
        )
        losses = self._loss_rows(trainer_records)
        lora = _lora_counters_from_model()
        kt = _kt_counters_from_model(self.config)
        activation_offload = _activation_offload_counters_from_model()
        asym_execution_stats = _asym_execution_stats_from_model()
        asym_liger_lm_head_bridge = _asym_liger_lm_head_bridge_from_model()
        process_memory = _process_memory_snapshot()
        return {
            "workload": self.config.get("workload", "lf"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "asym_precision_requested": self.config.get("precision", "bf16"),
            "asym_precision_effective": self.config.get("precision", "bf16"),
            "config": self.config,
            "step": {
                "total_milliseconds": forward_ms + backward_ms,
                "rows": stage_rows or [
                    {"name": "step.forward", "milliseconds": forward_ms},
                    {"name": "step.backward", "milliseconds": backward_ms},
                ],
            },
            "forward": {"total_milliseconds": forward_ms, "rows": []},
            "backward": {"total_milliseconds": backward_ms, "rows": []},
            "memory": {
                "gpu": {
                    "peak_allocated_hbm_bytes": self.global_peak_allocated_bytes,
                    "peak_reserved_hbm_bytes": self.global_peak_reserved_bytes,
                    "reserved_unallocated_bytes": max(
                        0, self.global_peak_reserved_bytes - self.global_peak_allocated_bytes
                    ),
                },
                "process": process_memory,
                "peak_allocated_hbm_bytes": self.global_peak_allocated_bytes,
                "peak_reserved_hbm_bytes": self.global_peak_reserved_bytes,
                "reserved_unallocated_bytes": max(0, self.global_peak_reserved_bytes - self.global_peak_allocated_bytes),
            },
            "stage_memory": {
                "rows": self._stage_memory_rows(),
                "max_stage_peak_allocated_bytes": self.global_peak_allocated_bytes,
                "max_stage_peak_reserved_bytes": self.global_peak_reserved_bytes,
            },
            "memory_attribution": (
                trace_handle.memory_summary(trace_handle.model)
                if trace_handle is not None
                else {"enabled": False, "rows": []}
            ),
            "memory_breakdown": {"enabled": False},
            "step_samples": {
                "source": "lf_source_recorder",
                "warmup_steps": self._warmup_steps(),
                "measure_steps": _safe_int(self.config.get("measure_steps")),
                "total_steps": _safe_int(self.config.get("total_steps")),
                "rows": self._step_sample_rows(losses, trainer_records, heartbeat_rows),
            },
            "trainer": {
                "trainer_log": str(trainer_log) if trainer_log is not None else "",
                "records": len(trainer_records),
                "timing": trainer_timing,
                "losses": losses,
            },
            "lora": lora,
            "kt": kt,
            "activation_offload": activation_offload,
            "asym_execution_stats": asym_execution_stats,
            "asym_liger_lm_head_bridge": asym_liger_lm_head_bridge,
            "grad_clip": _GRAD_CLIP_MARKER,
            "optimizer_memory_preflight": _kt_optimizer_memory_preflight(lora, self.config),
            "optimizer_memory": _OPTIMIZER_MEMORY_MARKER,
            "superoffload": _superoffload_summary_from_config(self.config),
            "cpuadam": _cpuadam_summary_from_config(self.config),
            "asym_cpu_adamw": _asym_cpu_adamw_summary_from_trace(trace_handle),
            "expert_token_distribution": {"samples": 0, "per_expert": []},
            "notes": _source_profile_notes(self.config),
        }


def main() -> None:
    global _GRAD_CLIP_MARKER, _MODEL_CAPTURE_HEARTBEAT, _MODEL_CAPTURE_PARTIAL_WRITER

    lf_args = sys.argv[1:]
    if lf_args and lf_args[0] == "train":
        lf_args = lf_args[1:]
    if "-h" in lf_args or "--help" in lf_args:
        print("Usage: run_lf_profiled_train.py [LLaMA-Factory train options]")
        print()
        print("Runs LLaMA-Factory train with LF/AsymGEMM NVTX ranges enabled.")
        print("Set ASYM_GEMM_LF_PROFILE_SOURCE_JSON to write the source profile JSON.")
        return

    _GRAD_CLIP_MARKER = {}
    config = _config_from_args(lf_args)
    heartbeat_path = os.environ.get(PROFILE_HEARTBEAT_ENV)
    if heartbeat_path:
        config["heartbeat_jsonl"] = heartbeat_path
    trace_config = LFTraceConfig.from_env(os.environ)
    recorder = LFProfileRecorder(
        config=config,
        measure_memory=_env_enabled(PROFILE_MEMORY_ENV, default=True),
        # Reset the torch peak counter per stage so forward/backward report true within-stage peaks.
        reset_stage_peak_stats=True,
    )
    source_json = os.environ.get(PROFILE_SOURCE_JSON_ENV)
    if source_json and _is_rank0():
        source_path = Path(source_json)
        _unlink_if_exists(source_path)
        _unlink_if_exists(source_path.with_name("source_profile.partial.json"))
    if heartbeat_path and _is_rank0():
        heartbeat_file = Path(heartbeat_path)
        _unlink_if_exists(heartbeat_file)
        _unlink_if_exists(heartbeat_file.with_name(f"{heartbeat_file.stem}.latest.json"))
    snapshot_enabled = _env_enabled(PROFILE_MEMORY_SNAPSHOT_ENV, default=False) and _is_rank0()
    snapshot_info = _start_memory_snapshot_recording(snapshot_enabled)

    set_profile_enabled(True)
    trace_handle = install_lf_trace(trace_config, recorder=recorder)
    heartbeat = LFHeartbeat(heartbeat_path, config)
    partial_writer = PartialProfileWriter(source_json, recorder, trace_handle, heartbeat)
    _MODEL_CAPTURE_HEARTBEAT = heartbeat
    _MODEL_CAPTURE_PARTIAL_WRITER = partial_writer
    heartbeat.emit("profile_launcher_start", source_json=source_json, partial_json=str(partial_writer.path or ""))
    partial_writer.write("profile_launcher_start", force=True)
    _install_model_capture_hook()
    _install_trainer_heartbeat_hooks(heartbeat, partial_writer)
    if _should_install_deepspeed_hook(lf_args, config):
        _install_deepspeed_optimizer_capture_hook()

    run_succeeded = False
    run_exception: BaseException | None = None
    try:
        from llamafactory.train.tuner import run_exp

        with trace_handle.saved_tensor_context():
            heartbeat.emit("run_exp_enter")
            partial_writer.write("run_exp_enter", force=True)
            run_exp(lf_args)
        run_succeeded = True
    except BaseException as exc:
        run_exception = exc
        heartbeat.emit("profile_launcher_exception", error=repr(exc))
        partial_writer.write("profile_launcher_exception", force=True, extra={"error": repr(exc)})
        raise
    finally:
        heartbeat.emit("profile_launcher_finally", success=run_succeeded)
        partial_writer.write("profile_launcher_finally", force=True)
        if source_json and _is_rank0():
            path = Path(source_json)
            write_path = path if run_succeeded else path.with_name("source_profile.partial.json")
            write_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path_raw = os.environ.get(PROFILE_MEMORY_SNAPSHOT_PATH_ENV, "").strip()
            snapshot_path = Path(snapshot_path_raw) if snapshot_path_raw else write_path.parent / "memory_snapshot.pickle"
            snapshot_info = _dump_memory_snapshot(snapshot_info, snapshot_path)
            report = recorder.report(trace_handle)
            if not run_succeeded:
                report["partial"] = True
                report["partial_reason"] = "profile_launcher_exception"
                if run_exception is not None:
                    report["error"] = repr(run_exception)
            if snapshot_info.get("enabled"):
                report["memory_snapshot"] = snapshot_info
            memory_breakdown = trace_handle.memory_breakdown_report()
            if memory_breakdown.get("enabled"):
                output_base = os.environ.get(PROFILE_MEMORY_BREAKDOWN_OUTPUT_ENV, "memory_breakdown").strip() or "memory_breakdown"
                breakdown_jsonl = write_path.parent / f"{output_base}.jsonl"
                breakdown_summary_json = write_path.parent / f"{output_base}_summary.json"
                rows = memory_breakdown.get("rows", [])
                row_items = rows if isinstance(rows, list) else []
                with breakdown_jsonl.open("w", encoding="utf-8") as handle:
                    for row in row_items:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                summary = memory_breakdown.get("summary", {})
                breakdown_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                report["memory_breakdown"] = {
                    "enabled": True,
                    "jsonl": str(breakdown_jsonl),
                    "summary_json": str(breakdown_summary_json),
                    "summary": summary,
                    "rows": len(row_items),
                }
            if run_succeeded:
                heartbeat.emit("source_profile_written", source_json=str(path))
            else:
                heartbeat.emit("source_profile_partial_written", source_json=str(write_path), error=repr(run_exception))
            report["heartbeat"] = {
                "jsonl": str(heartbeat.path) if heartbeat.path is not None else "",
                "latest": heartbeat.latest,
            }
            write_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _MODEL_CAPTURE_HEARTBEAT = None
        _MODEL_CAPTURE_PARTIAL_WRITER = None
        uninstall_lf_trace(trace_handle)


if __name__ == "__main__":
    main()
