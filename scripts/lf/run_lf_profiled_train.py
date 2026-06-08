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
PROFILE_EXTERNAL_MEMORY_ENV = "ASYM_GEMM_LF_PROFILE_EXTERNAL_MEMORY"
PROFILE_SYNC_ENV = "ASYM_GEMM_LF_PROFILE_SYNC"
PROFILE_MODULE_FILTER_ENV = "ASYM_GEMM_LF_PROFILE_MODULE_FILTER"
CONFIG_ENV_PREFIX = "ASYM_GEMM_LF_CONFIG_"

_LAST_LF_MODEL: Any | None = None
_DEEPSPEED_RUNTIME_MARKER: dict[str, Any] = {}


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_rank0() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _option_value(args: list[str], name: str) -> str:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return ""


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


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


def _env_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(CONFIG_ENV_PREFIX):
            field_name = key[len(CONFIG_ENV_PREFIX) :].lower()
            config[field_name] = value
    return config


def _config_from_args(args: list[str]) -> dict[str, Any]:
    env_config = _env_config()
    model_name = _option_value(args, "--model_name_or_path")
    model_label = model_name.rstrip("/").rsplit("/", 1)[-1] if model_name else "lf"
    batch_size = _safe_int(_option_value(args, "--per_device_train_batch_size"))
    cutoff_len = _safe_int(_option_value(args, "--cutoff_len"))
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
    expert_policy = parse_expert_recompute_policy_spec(os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none"))
    is_superoffload_backend = backend == "superoffload"
    config = {
        "workflow": "lora_lf_sft",
        "workload": os.environ.get("ASYM_GEMM_LF_CONFIG_WORKLOAD", model_label),
        "model_name_or_path": model_name,
        "backend": backend,
        "kt_backend": os.environ.get("ASYM_GEMM_LF_CONFIG_KT_BACKEND") or _option_value(args, "--kt_backend"),
        "precision": os.environ.get("ASYM_GEMM_LF_CONFIG_PRECISION") or _option_value(args, "--asym_precision") or "bf16",
        "dataset": _option_value(args, "--dataset"),
        "template": _option_value(args, "--template"),
        "batch_size": batch_size,
        "seq_len": _safe_int(os.environ.get("ASYM_GEMM_LF_CONFIG_SEQ_LEN")) or cutoff_len,
        "cutoff_len": cutoff_len,
        "max_samples": _safe_int(_option_value(args, "--max_samples")),
        "max_steps": measure_steps,
        "measure_steps": measure_steps,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "trainer_max_steps": max_steps,
        "gradient_accumulation_steps": _safe_int(_option_value(args, "--gradient_accumulation_steps")),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": _safe_float(_option_value(args, "--lora_dropout")),
        "activation_recompute": os.environ.get("ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE", "false").lower()
        in {"1", "true", "yes", "on"},
        "expert_recompute_policy_spec": expert_policy.label,
        "expert_recompute_policy": expert_policy.policy,
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
        "nsys_capture_range": _env_enabled("ASYM_GEMM_LF_NSYS_CAPTURE_RANGE"),
        "superoffload_config": os.environ.get("ASYM_GEMM_LF_CONFIG_SUPEROFFLOAD_CONFIG")
        if is_superoffload_backend
        else None,
        "deepspeed_dir": os.environ.get("ASYM_GEMM_LF_CONFIG_DEEPSPEED_DIR"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "output_dir": _option_value(args, "--output_dir"),
    }
    for key, value in env_config.items():
        config.setdefault(key, value)
    return {key: value for key, value in config.items() if value not in {"", None}}


def _capture_loaded_model(model: Any) -> Any:
    global _LAST_LF_MODEL
    _LAST_LF_MODEL = model
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


def _kt_counters_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    wrappers = _find_kt_wrappers(model, base_model)
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(wrappers or []):
        wrapper = getattr(layer, "wrapper", None)
        rows.append(
            {
                "index": index,
                "layer_idx": getattr(layer, "layer_idx", index),
                "method": getattr(wrapper, "method", ""),
                "forward_calls": int(getattr(wrapper, "_kt_forward_calls", 0) or 0),
                "backward_calls": int(getattr(wrapper, "_kt_backward_calls", 0) or 0),
                "lora_initialized": bool(getattr(wrapper, "_lora_initialized", False)),
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
    is_superoffload_backend = str(config.get("backend") or "").lower() == "superoffload"
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


def _install_deepspeed_optimizer_capture_hook() -> None:
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except Exception:
        return

    original_configure_optimizer = getattr(DeepSpeedEngine, "_configure_optimizer", None)
    if original_configure_optimizer is None:
        return
    if getattr(original_configure_optimizer, "_asym_gemm_superoffload_capture", False):
        return

    def wrapped_configure_optimizer(self: Any, client_optimizer: Any, model_parameters: Any) -> Any:
        result = original_configure_optimizer(self, client_optimizer, model_parameters)
        optimizer = getattr(self, "optimizer", None)
        optimizer_class = optimizer.__class__.__name__ if optimizer is not None else None
        try:
            engine_super_offload = bool(self.super_offload())
        except Exception:
            engine_super_offload = None
        _DEEPSPEED_RUNTIME_MARKER.update(
            {
                "optimizer_class": optimizer_class,
                "engine_class": self.__class__.__name__,
                "engine_super_offload": engine_super_offload,
                "marker_source": "deepspeed_engine_hook",
            }
        )
        if _is_rank0() and optimizer_class:
            print(f"DeepSpeed Final Optimizer = {optimizer_class}", flush=True)
            if engine_super_offload:
                print("DeepSpeed SuperOffload runtime enabled = true", flush=True)
        return result

    wrapped_configure_optimizer._asym_gemm_superoffload_capture = True  # type: ignore[attr-defined]
    DeepSpeedEngine._configure_optimizer = wrapped_configure_optimizer


def _start_memory_snapshot_recording(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    if not torch.cuda.is_available():
        return {"enabled": True, "record_started": False, "error": "cuda unavailable"}
    try:
        torch.cuda.memory._record_memory_history()  # type: ignore[attr-defined]
    except Exception as exc:
        return {"enabled": True, "record_started": False, "error": str(exc)}
    return {"enabled": True, "record_started": True}


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


def _lora_counters_from_model() -> dict[str, Any]:
    model, base_model = _model_and_base_model()
    if model is None:
        return {"available": False, "reason": "model hook did not capture a model"}

    trainable_params = 0
    all_params = 0
    peft_lora_params = 0
    for name, param in model.named_parameters():
        numel = int(param.numel())
        all_params += numel
        if param.requires_grad:
            trainable_params += numel
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad:
            peft_lora_params += numel

    lf_fused_modules = 0
    lf_fused_tensors = 0
    lf_fused_params = 0
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
    for root in module_roots:
        for module in root.modules():
            module_key = id(module)
            if module_key in seen_modules:
                continue
            seen_modules.add(module_key)

            lf_params = _iter_unique_parameters(getattr(module, "_lf_fused_lora_params", None))
            if lf_params:
                lf_fused_modules += 1
                lf_fused_tensors += len(lf_params)
                lf_fused_params += sum(int(param.numel()) for param in lf_params)

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

    return {
        "available": True,
        "trainable_parameters": trainable_params,
        "all_parameters": all_params,
        "peft_lora_parameters": peft_lora_params,
        "lf_fused_expert_lora_modules": lf_fused_modules,
        "lf_fused_expert_lora_tensors": lf_fused_tensors,
        "lf_fused_expert_lora_parameters": lf_fused_params,
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
        if name not in {"step.forward", "step.backward"}:
            return records
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
                }
            )
        return rows

    def _step_sample_rows(self, losses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        loss_by_step: dict[int, float] = {}
        for loss_row in losses:
            if not isinstance(loss_row, dict):
                continue
            step = _safe_int(loss_row.get("raw_step", loss_row.get("step")))
            loss = _safe_float(loss_row.get("loss"))
            if step is not None and loss is not None:
                loss_by_step[step] = loss

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
                }
            )

        forward_records = self.records.get("step.forward", [])
        backward_records = self.records.get("step.backward", [])
        sample_count = max(len(forward_records), len(backward_records))
        rows: list[dict[str, Any]] = []
        warmup_steps = self._warmup_steps()
        for index in range(sample_count):
            raw_step = index + 1
            measured_step = max(raw_step - warmup_steps, 0)
            is_warmup = raw_step <= warmup_steps
            forward = forward_records[index] if index < len(forward_records) else None
            backward = backward_records[index] if index < len(backward_records) else None
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
            row["step_milliseconds"] = sum(
                record.milliseconds for record in (forward, backward) if record is not None
            )
            row["peak_allocated_hbm_bytes"] = max(
                [record.peak_allocated_bytes for record in (forward, backward) if record is not None] or [0]
            )
            row["peak_reserved_hbm_bytes"] = max(
                [record.peak_reserved_bytes for record in (forward, backward) if record is not None] or [0]
            )
            row["reserved_unallocated_bytes"] = max(
                0, int(row["peak_reserved_hbm_bytes"]) - int(row["peak_allocated_hbm_bytes"])
            )
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
            records = self.records.get(name, [])
            if records:
                rows.append({"name": name, "milliseconds": self._stage_total_ms(name), "samples": len(records)})
        return rows

    def report(self, trace_handle: LFTraceHandle | None = None) -> dict[str, Any]:
        forward_ms = self._stage_total_ms("step.forward")
        backward_ms = self._stage_total_ms("step.backward")
        stage_rows = self._stage_rows()
        output_dir = Path(str(self.config.get("output_dir", ""))) if self.config.get("output_dir") else None
        trainer_log = output_dir / "trainer_log.jsonl" if output_dir is not None else None
        trainer_records = _trainer_log_records(trainer_log) if trainer_log is not None else []
        losses = self._loss_rows(trainer_records)
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
                "rows": self._step_sample_rows(losses),
            },
            "trainer": {
                "trainer_log": str(trainer_log) if trainer_log is not None else "",
                "records": len(trainer_records),
                "losses": losses,
            },
            "lora": _lora_counters_from_model(),
            "kt": _kt_counters_from_model(),
            "superoffload": _superoffload_summary_from_config(self.config),
            "expert_token_distribution": {"samples": 0, "per_expert": []},
            "notes": [
                "LF source timings are host wall-clock ranges without per-range CUDA synchronization.",
                "Use the Nsight Systems postprocessed profile.json for low-overhead GPU timing truth.",
            ],
        }


def main() -> None:
    lf_args = sys.argv[1:]
    if lf_args and lf_args[0] == "train":
        lf_args = lf_args[1:]
    if "-h" in lf_args or "--help" in lf_args:
        print("Usage: run_lf_profiled_train.py [LLaMA-Factory train options]")
        print()
        print("Runs LLaMA-Factory train with LF/AsymGEMM NVTX ranges enabled.")
        print("Set ASYM_GEMM_LF_PROFILE_SOURCE_JSON to write the source profile JSON.")
        return

    config = _config_from_args(lf_args)
    trace_config = LFTraceConfig.from_env(os.environ)
    recorder = LFProfileRecorder(
        config=config,
        measure_memory=_env_enabled(PROFILE_MEMORY_ENV, default=True),
        reset_stage_peak_stats=not trace_config.memory_breakdown,
    )
    source_json = os.environ.get(PROFILE_SOURCE_JSON_ENV)
    snapshot_enabled = _env_enabled(PROFILE_MEMORY_SNAPSHOT_ENV, default=False) and _is_rank0()
    snapshot_info = _start_memory_snapshot_recording(snapshot_enabled)

    set_profile_enabled(True)
    trace_handle = install_lf_trace(trace_config, recorder=recorder)
    _install_model_capture_hook()
    _install_deepspeed_optimizer_capture_hook()

    try:
        from llamafactory.train.tuner import run_exp

        with trace_handle.saved_tensor_context():
            run_exp(lf_args)
    finally:
        if source_json and _is_rank0():
            path = Path(source_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path_raw = os.environ.get(PROFILE_MEMORY_SNAPSHOT_PATH_ENV, "").strip()
            snapshot_path = Path(snapshot_path_raw) if snapshot_path_raw else path.parent / "memory_snapshot.pickle"
            snapshot_info = _dump_memory_snapshot(snapshot_info, snapshot_path)
            report = recorder.report(trace_handle)
            if snapshot_info.get("enabled"):
                report["memory_snapshot"] = snapshot_info
            memory_breakdown = trace_handle.memory_breakdown_report()
            if memory_breakdown.get("enabled"):
                output_base = os.environ.get(PROFILE_MEMORY_BREAKDOWN_OUTPUT_ENV, "memory_breakdown").strip() or "memory_breakdown"
                breakdown_jsonl = path.parent / f"{output_base}.jsonl"
                breakdown_summary_json = path.parent / f"{output_base}_summary.json"
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
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        uninstall_lf_trace(trace_handle)


if __name__ == "__main__":
    main()
