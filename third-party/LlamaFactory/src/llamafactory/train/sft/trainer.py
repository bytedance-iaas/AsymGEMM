# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)

_KT_ARM_BACKENDS = {"ARMBF16", "ARMBF16_SFT", "KT_ARM"}
_KT_GRAD_CLIP_CHUNK_ELEMENTS_ENV = "ASYM_GEMM_LF_KT_GRAD_CLIP_CHUNK_ELEMENTS"
_KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS = 8 * 1024 * 1024
_ASYM_DROP_TRAINING_LOGITS_ENV = "ASYM_DROP_TRAINING_LOGITS"


def _emit_asym_gemm_heartbeat(stage: str, **fields: Any) -> None:
    path = os.environ.get("ASYM_GEMM_LF_HEARTBEAT_JSON", "").strip()
    if not path:
        return
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "stage": stage,
        **fields,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        latest_path = f"{os.path.splitext(path)[0]}.latest.json"
        tmp_path = f"{latest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, latest_path)
    except OSError:
        return


def _is_kt_arm_backend(model_args: Optional["ModelArguments"]) -> bool:
    if model_args is None or not getattr(model_args, "use_kt", False):
        return False

    backend = str(getattr(model_args, "kt_backend", "") or "").upper()
    return backend in _KT_ARM_BACKENDS


def _asym_cpu_adamw_grad_offload_optimizer(optimizer: Any) -> Optional[Any]:
    current = optimizer
    seen: set[int] = set()
    for _ in range(8):
        if current is None:
            return None
        current_id = id(current)
        if current_id in seen:
            return None
        seen.add(current_id)

        enabled_fn = getattr(current, "asym_cpu_adamw_grad_offload_enabled", None)
        if callable(enabled_fn) and bool(enabled_fn()):
            return current

        next_optimizer = None
        for attr in ("optimizer", "base_optimizer", "wrapped_optimizer", "inner_optimizer"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                next_optimizer = candidate
                break
        current = next_optimizer

    return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _is_asym_backend(model_args: Optional["ModelArguments"]) -> bool:
    if model_args is None:
        return False
    backend = str(getattr(model_args, "asym_backend", "") or "").lower()
    return backend == "asym"


def _log_asym_device_residency(stage: str, model: "torch.nn.Module", model_args: Optional["ModelArguments"]) -> None:
    if not _env_bool("ASYM_GEMM_LF_AUDIT_DEVICE_PLACEMENT", True):
        return
    try:
        from asym_gemm.integrations.lf import summarize_lf_tensor_devices

        summary = summarize_lf_tensor_devices(
            model,
            offload_modules=getattr(model_args, "asym_offload_modules", None),
            max_examples_per_component=3,
        )
    except Exception as exc:
        logger.warning_rank0(f"AsymGEMM device residency audit failed at {stage}: {exc!r}.")
        return
    logger.info_rank0(f"AsymGEMM device residency at {stage}: {summary}.")


def _clear_output_logits(outputs: Any) -> None:
    try:
        if isinstance(outputs, dict) and "logits" in outputs:
            outputs["logits"] = None
    except Exception:
        pass
    try:
        if hasattr(outputs, "logits"):
            outputs.logits = None
    except Exception:
        pass


def _loss_from_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
    else:
        loss = outputs[0] if isinstance(outputs, (tuple, list)) and outputs else getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError("The model did not return a loss from the inputs; cannot drop training logits.")
    return loss


def _kt_grad_clip_chunk_elements() -> int:
    raw_value = os.environ.get(_KT_GRAD_CLIP_CHUNK_ELEMENTS_ENV, "").strip()
    if not raw_value:
        return _KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning_rank0(
            f"{_KT_GRAD_CLIP_CHUNK_ELEMENTS_ENV}={raw_value!r} is not an integer; "
            f"using {_KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS}."
        )
        return _KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS

    if value <= 0:
        logger.warning_rank0(
            f"{_KT_GRAD_CLIP_CHUNK_ELEMENTS_ENV} must be positive; "
            f"using {_KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS}."
        )
        return _KT_GRAD_CLIP_DEFAULT_CHUNK_ELEMENTS

    return value


def _grad_view_key(grad: torch.Tensor) -> tuple[Any, ...]:
    try:
        storage_ptr = grad.untyped_storage().data_ptr()
    except RuntimeError:
        storage_ptr = grad.data_ptr()

    return (
        grad.device.type,
        grad.device.index,
        storage_ptr,
        grad.storage_offset(),
        tuple(grad.shape),
        tuple(grad.stride()),
        str(grad.dtype),
    )


def _collect_unique_dense_grads(parameters: Any) -> tuple[list[torch.Tensor], dict[str, Any]]:
    grads: list[torch.Tensor] = []
    seen: set[tuple[Any, ...]] = set()
    stats: dict[str, Any] = {
        "parameter_count": 0,
        "parameters_with_grad": 0,
        "unique_grad_tensors": 0,
        "duplicate_grad_tensors": 0,
        "cpu_grad_tensors": 0,
        "cuda_grad_tensors": 0,
        "other_device_grad_tensors": 0,
        "cpu_grad_numel": 0,
        "cuda_grad_numel": 0,
        "other_device_grad_numel": 0,
    }

    for param in parameters:
        stats["parameter_count"] += 1
        if param is None or not getattr(param, "requires_grad", False):
            continue

        grad = getattr(param, "grad", None)
        if grad is None:
            continue

        if grad.is_sparse:
            raise RuntimeError("KT-aware gradient clipping does not support sparse gradients.")

        stats["parameters_with_grad"] += 1
        key = _grad_view_key(grad)
        if key in seen:
            stats["duplicate_grad_tensors"] += 1
            continue

        seen.add(key)
        grads.append(grad)
        stats["unique_grad_tensors"] += 1
        numel = int(grad.numel())
        if grad.device.type == "cpu":
            stats["cpu_grad_tensors"] += 1
            stats["cpu_grad_numel"] += numel
        elif grad.device.type == "cuda":
            stats["cuda_grad_tensors"] += 1
            stats["cuda_grad_numel"] += numel
        else:
            stats["other_device_grad_tensors"] += 1
            stats["other_device_grad_numel"] += numel

    return grads, stats


def _iter_grad_chunks(grad: torch.Tensor, chunk_elements: int):
    flat = grad.detach().reshape(-1)
    for start in range(0, int(flat.numel()), chunk_elements):
        yield flat.narrow(0, start, min(chunk_elements, int(flat.numel()) - start))


def _grad_chunk_norm(grad: torch.Tensor, norm_type: float, chunk_elements: int) -> torch.Tensor:
    if norm_type == math.inf:
        max_abs = torch.zeros((), dtype=torch.float64, device="cpu")
        for chunk in _iter_grad_chunks(grad, chunk_elements):
            if chunk.numel() == 0:
                continue
            value = chunk.abs().max().to(device="cpu", dtype=torch.float64)
            max_abs = torch.maximum(max_abs, value)
        return max_abs

    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for chunk in _iter_grad_chunks(grad, chunk_elements):
        if chunk.numel() == 0:
            continue
        chunk_f32 = chunk if chunk.dtype in (torch.float32, torch.float64) else chunk.float()
        value = torch.sum(torch.abs(chunk_f32) ** norm_type, dtype=torch.float64)
        total += value.to(device="cpu", dtype=torch.float64)

    return total.pow(1.0 / norm_type)


def _kt_clip_grad_norm_(
    parameters: Any,
    max_norm: float,
    norm_type: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grads, stats = _collect_unique_dense_grads(parameters)
    chunk_elements = _kt_grad_clip_chunk_elements()
    norm_only = math.isinf(float(max_norm))
    stats.update(
        {
            "enabled": True,
            "path": "kt_aware_dense",
            "operation": "norm_only" if norm_only else "clip",
            "max_norm": "inf" if norm_only else float(max_norm),
            "norm_type": float(norm_type) if norm_type != math.inf else "inf",
            "chunk_elements": chunk_elements,
        }
    )

    if not grads:
        total_norm = torch.zeros((), dtype=torch.float32)
        stats.update({"total_norm": 0.0, "clip_coef": 1.0, "clipped": False, "nonfinite": False})
        return total_norm, stats

    if norm_type == math.inf:
        total_norm = torch.zeros((), dtype=torch.float64, device="cpu")
        for grad in grads:
            total_norm = torch.maximum(total_norm, _grad_chunk_norm(grad, norm_type, chunk_elements))
    else:
        total = torch.zeros((), dtype=torch.float64, device="cpu")
        for grad in grads:
            grad_norm = _grad_chunk_norm(grad, norm_type, chunk_elements)
            total += grad_norm.pow(norm_type)
        total_norm = total.pow(1.0 / norm_type)

    total_norm_value = float(total_norm.item())
    nonfinite = not math.isfinite(total_norm_value)
    if norm_only:
        clip_coef = 1.0
        clip_coef_clamped = 1.0
        clipped = False
    elif math.isnan(total_norm_value):
        clip_coef = float("nan")
        clip_coef_clamped = 1.0
        clipped = False
    else:
        clip_coef = float(max_norm) / (total_norm_value + 1e-6)
        clip_coef_clamped = min(clip_coef, 1.0) if math.isfinite(clip_coef) else clip_coef
        clipped = bool(clip_coef_clamped < 1.0)

    if clipped:
        for grad in grads:
            grad.mul_(clip_coef_clamped)

    stats.update(
        {
            "total_norm": total_norm_value,
            "clip_coef": clip_coef,
            "clip_coef_clamped": clip_coef_clamped,
            "clipped": clipped,
            "nonfinite": nonfinite,
        }
    )
    return total_norm.to(dtype=torch.float32), stats


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self._patch_asym_accelerator_prepare_model()
        if _is_kt_arm_backend(model_args) and getattr(training_args, "max_grad_norm", 0) > 0:
            logger.info_rank0(
                "KT ARM BF16 will use KT-aware dense gradient clipping for CPU fused expert LoRA gradients."
            )
        elif getattr(model_args, "use_kt", False) and getattr(training_args, "max_grad_norm", 0) > 0:
            logger.warning_rank0(
                "KT fused expert LoRA keeps part of the trainable surface outside normal CUDA parameter "
                "gradients. Set `max_grad_norm=0` unless KT-aware clipping has been validated for this run."
            )
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def _move_model_to_device(self, model: "torch.nn.Module", device: "torch.device") -> None:
        if getattr(model, "_asym_cpu_first_selective_device_move", False):
            logger.info_rank0(
                "Skipping Trainer full-device placement because AsymGEMM CPU-first selective placement is active."
            )
            return
        return super()._move_model_to_device(model, device)

    def _patch_asym_accelerator_prepare_model(self) -> None:
        if not _is_asym_backend(getattr(self, "model_args", None)):
            return
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or getattr(accelerator, "_asym_cpu_first_prepare_model_patched", False):
            return

        original_prepare_model = accelerator.prepare_model

        def prepare_model_with_asym_cpu_first(model, device_placement=None, evaluation_mode=False):
            if getattr(model, "_asym_cpu_first_selective_device_move", False):
                _log_asym_device_residency(
                    "before_accelerator_prepare_model",
                    model,
                    getattr(self, "model_args", None),
                )
                logger.info_rank0(
                    "Calling accelerator.prepare_model with device_placement=False because AsymGEMM "
                    "CPU-first selective placement is active."
                )
                prepared = original_prepare_model(
                    model,
                    device_placement=False,
                    evaluation_mode=evaluation_mode,
                )
                _log_asym_device_residency(
                    "after_accelerator_prepare_model",
                    prepared,
                    getattr(self, "model_args", None),
                )
                return prepared
            return original_prepare_model(
                model,
                device_placement=device_placement,
                evaluation_mode=evaluation_mode,
            )

        accelerator.prepare_model = prepare_model_with_asym_cpu_first
        accelerator._asym_cpu_first_prepare_model_patched = True

    def _wrap_kt_optimizer_step(self) -> None:
        if not getattr(self.model_args, "use_kt", False) or self.optimizer is None:
            return
        if getattr(self.optimizer, "_kt_lora_pointer_refresh_wrapped", False):
            return

        original_step = self.optimizer.step

        def step_with_kt_lora_refresh(*args, **kwargs):
            result = original_step(*args, **kwargs)
            from kt_kernel.sft import update_kt_lora_pointers

            model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
            _emit_asym_gemm_heartbeat("kt_lora_pointer_refresh_start", trainer_class=self.__class__.__name__)
            try:
                update_kt_lora_pointers(model)
            except BaseException as exc:
                _emit_asym_gemm_heartbeat(
                    "kt_lora_pointer_refresh_exception",
                    trainer_class=self.__class__.__name__,
                    error=repr(exc),
                )
                raise
            _emit_asym_gemm_heartbeat("kt_lora_pointer_refresh_end", trainer_class=self.__class__.__name__)
            return result

        self.optimizer.step = step_with_kt_lora_refresh
        self.optimizer._kt_lora_pointer_refresh_wrapped = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        optimizer = super().create_optimizer()
        self._wrap_kt_optimizer_step()
        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _clip_grad_norm(self, model: "torch.nn.Module") -> "torch.Tensor":
        asym_optimizer = _asym_cpu_adamw_grad_offload_optimizer(getattr(self, "optimizer", None))
        if asym_optimizer is not None:
            if hasattr(self.accelerator, "unscale_gradients"):
                self.accelerator.unscale_gradients()

            total_norm, summary = asym_optimizer.asym_cpu_adamw_clip_grad_norm_(
                float(self.args.max_grad_norm),
                chunk_elements=_kt_grad_clip_chunk_elements(),
            )
            self._asym_cpu_adamw_grad_clip_last_summary = summary
            _emit_asym_gemm_heartbeat(
                "asym_cpu_adamw_grad_clip",
                trainer_class=self.__class__.__name__,
                **summary,
            )
            return total_norm

        if not _is_kt_arm_backend(self.model_args):
            return super()._clip_grad_norm(model)

        if hasattr(self.accelerator, "unscale_gradients"):
            self.accelerator.unscale_gradients()

        total_norm, summary = _kt_clip_grad_norm_(model.parameters(), float(self.args.max_grad_norm))
        self._kt_grad_clip_last_summary = summary
        _emit_asym_gemm_heartbeat("kt_aware_grad_clip", trainer_class=self.__class__.__name__, **summary)
        return total_norm

    @override
    def _get_grad_norm(self, model: "torch.nn.Module", grad_norm: Optional[Any] = None) -> Any:
        asym_optimizer = _asym_cpu_adamw_grad_offload_optimizer(getattr(self, "optimizer", None))
        if asym_optimizer is not None:
            if grad_norm is None:
                if hasattr(self.accelerator, "unscale_gradients"):
                    self.accelerator.unscale_gradients()
                grad_norm, summary = asym_optimizer.asym_cpu_adamw_clip_grad_norm_(
                    float("inf"),
                    chunk_elements=_kt_grad_clip_chunk_elements(),
                )
                summary["operation"] = "norm_only"
                self._asym_cpu_adamw_grad_clip_last_summary = summary
                _emit_asym_gemm_heartbeat(
                    "asym_cpu_adamw_grad_norm",
                    trainer_class=self.__class__.__name__,
                    **summary,
                )
                return grad_norm

            summary = getattr(self, "_asym_cpu_adamw_grad_clip_last_summary", None)
            if isinstance(summary, dict):
                summary["returned_grad_norm"] = float(grad_norm.item()) if hasattr(grad_norm, "item") else grad_norm
                self._asym_cpu_adamw_grad_clip_last_summary = summary
            return grad_norm

        if not _is_kt_arm_backend(self.model_args):
            return super()._get_grad_norm(model, grad_norm=grad_norm)

        if grad_norm is None:
            if hasattr(self.accelerator, "unscale_gradients"):
                self.accelerator.unscale_gradients()
            grad_norm, summary = _kt_clip_grad_norm_(model.parameters(), float("inf"))
            self._kt_grad_clip_last_summary = summary
            _emit_asym_gemm_heartbeat("kt_aware_grad_norm", trainer_class=self.__class__.__name__, **summary)
            return grad_norm

        summary = getattr(self, "_kt_grad_clip_last_summary", None)
        if isinstance(summary, dict):
            summary["returned_grad_norm"] = float(grad_norm.item()) if hasattr(grad_norm, "item") else grad_norm
            self._kt_grad_clip_last_summary = summary
        return grad_norm

    @override
    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Optional["torch.nn.Module"] = None) -> None:
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if not getattr(self.model_args, "use_kt", False):
            return
        if not resume_from_checkpoint:
            return

        from kt_kernel.sft import load_kt_moe_from_adapter

        target_model = model if model is not None else self.model
        unwrapped_model = (
            self.accelerator.unwrap_model(target_model) if hasattr(self, "accelerator") else target_model
        )
        load_kt_moe_from_adapter(unwrapped_model, resume_from_checkpoint)
        logger.info_rank0(f"Loaded KTransformers fused expert LoRA sidecar from {resume_from_checkpoint}.")

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict[str, Any]] = None) -> None:
        if not getattr(self.model_args, "use_asym_gemm", False):
            super()._save(output_dir=output_dir, state_dict=state_dict)
            if not getattr(self.model_args, "use_kt", False):
                return
            if not self.is_world_process_zero():
                return

            kt_output_dir = output_dir if output_dir is not None else self.args.output_dir
            model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model

            from kt_kernel.sft import save_kt_moe_to_adapter

            save_kt_moe_to_adapter(model, kt_output_dir)
            logger.info_rank0(f"Saved KTransformers fused expert LoRA sidecar to {kt_output_dir}.")
            return

        if not self.is_world_process_zero():
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model

        from asym_gemm.integrations.lf import save_asym_peft_adapter

        metadata = {
            "base_model_name_or_path": getattr(self.model_args, "model_name_or_path", None),
            "r": self.finetuning_args.lora_rank,
            "lora_alpha": self.finetuning_args.lora_alpha,
            "lora_dropout": self.finetuning_args.lora_dropout,
            "target_modules": self.finetuning_args.lora_target,
            "asym_backend": getattr(self.model_args, "asym_backend", None),
            "asym_precision": getattr(self.model_args, "asym_precision", None),
            "asym_offload_modules": getattr(self.model_args, "asym_offload_modules", None),
            "asym_expert_recompute_policy": getattr(self.model_args, "asym_expert_recompute_policy", None),
            "asym_router_mode": getattr(self.model_args, "asym_router_mode", None),
        }
        save_asym_peft_adapter(model, output_dir, metadata=metadata)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info_rank0(f"Saved AsymGEMM adapter to {output_dir}.")

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return_outputs = bool(kwargs.get("return_outputs", args[0] if args else False))
            if (
                _env_bool(_ASYM_DROP_TRAINING_LOGITS_ENV, True)
                and _is_asym_backend(getattr(self, "model_args", None))
                and bool(getattr(model, "training", False))
                and not return_outputs
                and "labels" in inputs
                and getattr(self, "label_smoother", None) is None
            ):
                outputs = model(**inputs)
                loss = _loss_from_outputs(outputs)
                _clear_output_logits(outputs)
                return loss
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
