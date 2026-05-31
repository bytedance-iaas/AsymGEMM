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

from asym_gemm.training.profile_ranges import prof_range, set_profile_enabled


PROFILE_SOURCE_JSON_ENV = "ASYM_GEMM_LF_PROFILE_SOURCE_JSON"
PROFILE_MEMORY_ENV = "ASYM_GEMM_LF_PROFILE_MEMORY"
CONFIG_ENV_PREFIX = "ASYM_GEMM_LF_CONFIG_"


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    model_name = _option_value(args, "--model_name_or_path")
    model_label = model_name.rstrip("/").rsplit("/", 1)[-1] if model_name else "lf"
    batch_size = _safe_int(_option_value(args, "--per_device_train_batch_size"))
    cutoff_len = _safe_int(_option_value(args, "--cutoff_len"))
    lora_rank = _safe_int(_option_value(args, "--lora_rank"))
    lora_alpha = _safe_float(_option_value(args, "--lora_alpha"))
    max_steps = _safe_int(_option_value(args, "--max_steps"))
    asym_backend = _option_value(args, "--asym_backend")
    backend = os.environ.get("ASYM_GEMM_LF_CONFIG_BACKEND") or ("torch" if asym_backend == "torch" else asym_backend or "hf")
    config = {
        "workflow": "lora_lf_sft",
        "workload": os.environ.get("ASYM_GEMM_LF_CONFIG_WORKLOAD", model_label),
        "model_name_or_path": model_name,
        "backend": backend,
        "precision": os.environ.get("ASYM_GEMM_LF_CONFIG_PRECISION") or _option_value(args, "--asym_precision") or "bf16",
        "dataset": _option_value(args, "--dataset"),
        "template": _option_value(args, "--template"),
        "batch_size": batch_size,
        "seq_len": _safe_int(os.environ.get("ASYM_GEMM_LF_CONFIG_SEQ_LEN")) or cutoff_len,
        "cutoff_len": cutoff_len,
        "max_samples": _safe_int(_option_value(args, "--max_samples")),
        "max_steps": max_steps,
        "gradient_accumulation_steps": _safe_int(_option_value(args, "--gradient_accumulation_steps")),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": _safe_float(_option_value(args, "--lora_dropout")),
        "activation_recompute": os.environ.get("ASYM_GEMM_LF_CONFIG_ACTIVATION_RECOMPUTE", "false").lower()
        in {"1", "true", "yes", "on"},
        "expert_recompute_policy_spec": os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none"),
        "expert_recompute_policy": "none",
        "expert_recompute_threshold": 0,
        "expert_recompute_util_threshold": 0.0,
        "expert_activation_save_policy": "save_all",
        "expert_activation_save_threshold": 0,
        "expert_policy_label": os.environ.get("ASYM_GEMM_LF_CONFIG_EXPERT_POLICY", "none"),
        "output_dir": _option_value(args, "--output_dir"),
    }
    for key, value in _env_config().items():
        config.setdefault(key, value)
    return {key: value for key, value in config.items() if value not in {"", None}}


@dataclass
class StageRecord:
    milliseconds: float
    allocated_start_bytes: int
    allocated_end_bytes: int
    reserved_start_bytes: int
    reserved_end_bytes: int
    local_peak_bytes: int
    global_peak_after_bytes: int

    @property
    def allocated_delta_bytes(self) -> int:
        return self.allocated_end_bytes - self.allocated_start_bytes

    @property
    def reserved_delta_bytes(self) -> int:
        return self.reserved_end_bytes - self.reserved_start_bytes

    @property
    def local_peak_delta_bytes(self) -> int:
        return self.local_peak_bytes - self.allocated_start_bytes


@dataclass
class LFProfileRecorder:
    config: dict[str, Any]
    measure_memory: bool = True
    records: dict[str, list[StageRecord]] = field(default_factory=lambda: {"step.forward": [], "step.backward": []})
    global_peak_bytes: int = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start_time = time.perf_counter()
        cuda_available = self.measure_memory and torch.cuda.is_available()
        allocated_start = reserved_start = 0
        if cuda_available:
            allocated_start = int(torch.cuda.memory_allocated())
            reserved_start = int(torch.cuda.memory_reserved())
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
        try:
            with prof_range(name):
                yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            allocated_end = reserved_end = local_peak = global_peak = 0
            if cuda_available:
                allocated_end = int(torch.cuda.memory_allocated())
                reserved_end = int(torch.cuda.memory_reserved())
                local_peak = int(torch.cuda.max_memory_allocated())
                global_peak = max(local_peak, allocated_end)
                self.global_peak_bytes = max(self.global_peak_bytes, global_peak)
            self.records.setdefault(name, []).append(
                StageRecord(
                    milliseconds=elapsed_ms,
                    allocated_start_bytes=allocated_start,
                    allocated_end_bytes=allocated_end,
                    reserved_start_bytes=reserved_start,
                    reserved_end_bytes=reserved_end,
                    local_peak_bytes=local_peak,
                    global_peak_after_bytes=global_peak,
                )
            )

    def _stage_total_ms(self, name: str) -> float:
        records = self.records.get(name, [])
        if not records:
            return 0.0
        return sum(record.milliseconds for record in records) / float(len(records))

    def _stage_memory_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in ("step.forward", "step.backward"):
            records = self.records.get(name, [])
            if not records:
                continue

            def avg(field_name: str) -> float:
                return sum(float(getattr(record, field_name)) for record in records) / float(len(records))

            rows.append(
                {
                    "name": name,
                    "samples": len(records),
                    "avg_allocated_start_bytes": avg("allocated_start_bytes"),
                    "avg_allocated_end_bytes": avg("allocated_end_bytes"),
                    "avg_allocated_delta_bytes": avg("allocated_delta_bytes"),
                    "avg_reserved_start_bytes": avg("reserved_start_bytes"),
                    "avg_reserved_end_bytes": avg("reserved_end_bytes"),
                    "avg_reserved_delta_bytes": avg("reserved_delta_bytes"),
                    "avg_local_peak_bytes": avg("local_peak_bytes"),
                    "avg_local_peak_delta_bytes": avg("local_peak_delta_bytes"),
                    "max_local_peak_bytes": max(record.local_peak_bytes for record in records),
                    "max_global_peak_after_bytes": max(record.global_peak_after_bytes for record in records),
                }
            )
        return rows

    def _step_sample_rows(self, losses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        loss_by_step: dict[int, float] = {}
        for loss_row in losses:
            if not isinstance(loss_row, dict):
                continue
            step = _safe_int(loss_row.get("step"))
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
                    f"{prefix}_local_peak_bytes": record.local_peak_bytes,
                    f"{prefix}_local_peak_delta_bytes": record.local_peak_delta_bytes,
                    f"{prefix}_global_peak_after_bytes": record.global_peak_after_bytes,
                }
            )

        forward_records = self.records.get("step.forward", [])
        backward_records = self.records.get("step.backward", [])
        sample_count = max(len(forward_records), len(backward_records))
        rows: list[dict[str, Any]] = []
        for index in range(sample_count):
            step = index + 1
            forward = forward_records[index] if index < len(forward_records) else None
            backward = backward_records[index] if index < len(backward_records) else None
            row: dict[str, Any] = {"step": step}
            if step in loss_by_step:
                row["loss"] = loss_by_step[step]
            add_stage(row, "forward", forward)
            add_stage(row, "backward", backward)
            row["step_milliseconds"] = sum(
                record.milliseconds for record in (forward, backward) if record is not None
            )
            row["peak_hbm_bytes"] = max(
                [record.local_peak_bytes for record in (forward, backward) if record is not None] or [0]
            )
            row["global_peak_after_bytes"] = max(
                [record.global_peak_after_bytes for record in (forward, backward) if record is not None] or [0]
            )
            rows.append(row)
        return rows

    def report(self) -> dict[str, Any]:
        forward_ms = self._stage_total_ms("step.forward")
        backward_ms = self._stage_total_ms("step.backward")
        output_dir = Path(str(self.config.get("output_dir", ""))) if self.config.get("output_dir") else None
        trainer_log = output_dir / "trainer_log.jsonl" if output_dir is not None else None
        trainer_records = _trainer_log_records(trainer_log) if trainer_log is not None else []
        losses = [
            {"step": record.get("current_steps", record.get("step")), "loss": record.get("loss")}
            for record in trainer_records
            if record.get("loss") is not None
        ]
        return {
            "workload": self.config.get("workload", "lf"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "asym_precision_requested": self.config.get("precision", "bf16"),
            "asym_precision_effective": self.config.get("precision", "bf16"),
            "config": self.config,
            "step": {
                "total_milliseconds": forward_ms + backward_ms,
                "rows": [
                    {"name": "step.forward", "milliseconds": forward_ms},
                    {"name": "step.backward", "milliseconds": backward_ms},
                ],
            },
            "forward": {"total_milliseconds": forward_ms, "rows": []},
            "backward": {"total_milliseconds": backward_ms, "rows": []},
            "memory": {
                "gpu": {
                    "peak_hbm_bytes": self.global_peak_bytes,
                    "stage_local_peak_hbm_bytes": self.global_peak_bytes,
                },
                "peak_hbm_bytes": self.global_peak_bytes,
            },
            "stage_memory": {
                "rows": self._stage_memory_rows(),
                "max_stage_peak_bytes": self.global_peak_bytes,
            },
            "step_samples": {
                "source": "lf_source_recorder",
                "rows": self._step_sample_rows(losses),
            },
            "trainer": {
                "trainer_log": str(trainer_log) if trainer_log is not None else "",
                "records": len(trainer_records),
                "losses": losses,
            },
            "expert_token_distribution": {"samples": 0, "per_expert": []},
            "notes": [
                "LF source timings are host wall-clock ranges without per-range CUDA synchronization.",
                "Use the Nsight Systems postprocessed profile.json for low-overhead GPU timing truth.",
            ],
        }


def _patch_training(recorder: LFProfileRecorder) -> None:
    from accelerate import Accelerator
    from transformers import Trainer

    original_backward = Accelerator.backward

    def patch_compute_loss(cls: type[Any]) -> bool:
        original_compute_loss = getattr(cls, "compute_loss", None)
        if original_compute_loss is None or getattr(original_compute_loss, "_asym_lf_profile_wrapped", False):
            return False

        def compute_loss_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            with recorder.stage("step.forward"):
                return original_compute_loss(self, *args, **kwargs)

        compute_loss_with_profile._asym_lf_profile_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, "compute_loss", compute_loss_with_profile)
        return True

    patched_forward = False
    try:
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

        patched_forward = patch_compute_loss(CustomSeq2SeqTrainer) or patched_forward
    except Exception:
        patched_forward = False
    if not patched_forward:
        patch_compute_loss(Trainer)

    def backward_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
        with recorder.stage("step.backward"):
            return original_backward(self, *args, **kwargs)

    Accelerator.backward = backward_with_profile


def main() -> None:
    lf_args = sys.argv[1:]
    if lf_args and lf_args[0] == "train":
        lf_args = lf_args[1:]
    if "-h" in lf_args or "--help" in lf_args:
        print("Usage: run_lf_profiled_train.py [LLaMA-Factory train options]")
        print()
        print("Runs LLaMA-Factory train with AsymGEMM NVTX ranges enabled around step.forward and step.backward.")
        print("Set ASYM_GEMM_LF_PROFILE_SOURCE_JSON to write the source profile JSON.")
        return

    config = _config_from_args(lf_args)
    recorder = LFProfileRecorder(config=config, measure_memory=_env_enabled(PROFILE_MEMORY_ENV, default=True))
    source_json = os.environ.get(PROFILE_SOURCE_JSON_ENV)

    set_profile_enabled(True)
    _patch_training(recorder)

    try:
        from llamafactory.train.tuner import run_exp

        run_exp(lf_args)
    finally:
        if source_json:
            path = Path(source_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(recorder.report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
