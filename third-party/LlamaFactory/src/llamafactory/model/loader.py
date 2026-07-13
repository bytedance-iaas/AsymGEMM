# Copyright 2025 the LlamaFactory team.
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

import gc
import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, get_current_device, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_torch_version_greater_than
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _use_asym_cpu_first_load(model_args: "ModelArguments") -> bool:
    if not (
        getattr(model_args, "use_asym_gemm", False)
        and getattr(model_args, "asym_backend", None) == "asym"
    ):
        return False
    from asym_gemm.integrations.lf import parse_lf_offload_modules

    selection = parse_lf_offload_modules(getattr(model_args, "asym_offload_modules", None))
    return selection.any_cpu_offload


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _drop_asym_frozen_vision_for_text_profile(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if not _env_bool("ASYM_GEMM_LF_DROP_FROZEN_VISION", False):
        return
    if not bool(getattr(finetuning_args, "freeze_vision_tower", True)):
        raise RuntimeError("ASYM_GEMM_LF_DROP_FROZEN_VISION=1 requires freeze_vision_tower=true")

    from asym_gemm.integrations.lf import drop_lf_frozen_vision_tower_for_text_only_profile

    summary = drop_lf_frozen_vision_tower_for_text_only_profile(
        model,
        strict=bool(getattr(model_args, "asym_strict", True)),
    )
    setattr(model, "_asym_text_only_vision_dropped", True)
    logger.info_rank0(f"AsymGEMM text-only frozen vision drop summary: {summary}.")


def _move_asym_cpu_first_model_to_device(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    device = get_current_device()
    logger.info_rank0(f"Moving AsymGEMM non-HostWeight model state to {device}.")
    from asym_gemm.integrations.lf import move_lf_asym_cpu_first_model_to_device

    summary = move_lf_asym_cpu_first_model_to_device(
        model,
        device,
        offload_modules=getattr(model_args, "asym_offload_modules", None),
        strict=bool(getattr(model_args, "asym_strict", True)),
        keep_frozen_vision_on_cpu=bool(getattr(finetuning_args, "freeze_vision_tower", True)),
        keep_frozen_projector_on_cpu=bool(getattr(finetuning_args, "freeze_multi_modal_projector", True)),
    )
    setattr(model, "_asym_cpu_first_selective_device_move", True)
    setattr(model, "_asym_cpu_first_target_device", str(device))
    logger.info_rank0(f"AsymGEMM CPU-first selective device move summary: {summary}.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))
    asym_cpu_first_load = _use_asym_cpu_first_load(model_args)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"
        if asym_cpu_first_load:
            init_kwargs.pop("device_map", None)
            init_kwargs["low_cpu_mem_usage"] = True
            logger.info_rank0("AsymGEMM CPU offload enabled: loading pretrained model on CPU first.")

        if model_args.use_kt:
            if not is_trainable:
                raise ValueError("KTransformers SFT backend is only supported while training.")
            if model_args.mixture_of_depths is not None:
                raise ValueError("KTransformers SFT backend is incompatible with mixture_of_depths.")

            from kt_kernel.sft import load_kt_model

            model = load_kt_model(
                config=config,
                model_args=model_args,
                finetuning_args=finetuning_args,
                model_name_or_path=model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                token=model_args.hf_hub_token,
                torch_dtype=model_args.compute_dtype or torch.bfloat16,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
            )
        elif model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    if model_args.use_kt and is_trainable:
        from kt_kernel.sft import kt_adapt_peft_lora, load_kt_moe_from_adapter

        kt_adapt_peft_lora(model)
        if model_args.adapter_name_or_path is not None:
            kt_adapter_paths = model_args.adapter_name_or_path
            kt_adapter_path = kt_adapter_paths if isinstance(kt_adapter_paths, str) else kt_adapter_paths[-1]
            load_kt_moe_from_adapter(model, kt_adapter_path)

    if asym_cpu_first_load:
        _drop_asym_frozen_vision_for_text_profile(model, model_args, finetuning_args)
        _move_asym_cpu_first_model_to_device(model, model_args, finetuning_args)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # Conv3D is not recommended when using torch 2.9.x
    if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
        if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
            raise ValueError(
                "Unsupported torch version detected: torch 2.9.x with Conv3D. "
                "This combination is known to cause severe performance regression. "
                "Please downgrade torch to <2.9 or remove Conv3D. "
                "See https://github.com/pytorch/pytorch/issues/166122"
            )

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model, include_kernels=model_args.use_v1_kernels)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
