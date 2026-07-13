# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
import os
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES, EngineName
from ..extras.misc import check_dependencies, check_version, get_current_device, is_env_enabled
from ..extras.packages import is_mcore_adapter_available
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .training_args import RayArguments, TrainingArguments


logger = logging.get_logger(__name__)

check_dependencies()

_LOCAL_KT_BACKENDS = {"TORCHBF16", "TORCHBF16_SFT", "ARMBF16", "ARMBF16_SFT", "KT_ARM"}
_ARM_KT_BACKENDS = {"ARMBF16", "ARMBF16_SFT", "KT_ARM"}
_SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS = {"TORCHBF16", "TORCHBF16_SFT", "ARMBF16", "ARMBF16_SFT", "KT_ARM"}
_ARM_KT_DEFAULT_TOP_K = 8
_ARM_KT_DEFAULT_MAX_ROUTE_RANK_WORK = 1_048_576
_ARM_KT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES = 34_359_738_368


_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_EVAL_CLS = tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]

if is_mcore_adapter_available() and is_env_enabled("USE_MCA"):
    from mcore_adapter import TrainingArguments as McaTrainingArguments

    _TRAIN_MCA_ARGS = [ModelArguments, DataArguments, McaTrainingArguments, FinetuningArguments, GeneratingArguments]
    _TRAIN_MCA_CLS = tuple[
        ModelArguments, DataArguments, McaTrainingArguments, FinetuningArguments, GeneratingArguments
    ]
else:
    _TRAIN_MCA_ARGS = []
    _TRAIN_MCA_CLS = tuple()


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        return sys.argv[1:]


def _parse_args(
    parser: "HfArgumentParser", args: dict[str, Any] | list[str] | None = None, allow_extra_keys: bool = False
) -> tuple[Any]:
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)


def _verify_trackio_args(training_args: "TrainingArguments") -> None:
    """Validates Trackio-specific arguments.

    Args:
        training_args: TrainingArguments instance (not a dictionary)
    """
    report_to = training_args.report_to
    if not report_to:
        return

    if isinstance(report_to, str):
        report_to = [report_to]

    if "trackio" not in report_to:
        return

    # --- Enforce project (required by Trackio) ---
    if not training_args.project:
        raise ValueError("`--project` must be specified when using Trackio.")

    # --- Validate trackio_space_id format ---
    space_id = training_args.trackio_space_id
    if space_id:
        if space_id != "trackio" and "/" not in space_id:
            logger.warning(
                f"trackio_space_id '{space_id}' should typically be in format "
                "'org/space' for Hugging Face Spaces deployment."
            )

    # --- Inform about default project usage ---
    if training_args.project == "huggingface":
        logger.info(
            "Using default project name 'huggingface'. "
            "Consider setting a custom project name with --project "
            "for better organization."
        )

    # --- Validate hub repo privacy flag ---
    if training_args.hub_private_repo:
        logger.info("Repository will be created as private on Hugging Face Hub.")

    # --- Recommend run_name for experiment clarity ---
    if not training_args.run_name:
        logger.warning("Consider setting --run_name for better experiment tracking clarity.")


def _set_transformers_logging() -> None:
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _set_env_vars() -> None:
    if is_torch_npu_available():
        # avoid JIT compile on NPU devices, see https://zhuanlan.zhihu.com/p/660875458
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("NPU_JIT_COMPILE"))
        # avoid use fork method on NPU devices, see https://github.com/hiyouga/LLaMA-Factory/issues/7447
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantization is only compatible with the LoRA or OFT method.")

        if finetuning_args.pissa_init:
            raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA for a quantized model.")

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")


def _is_local_kt_backend(model_args: "ModelArguments") -> bool:
    return str(getattr(model_args, "kt_backend", "")).upper() in _LOCAL_KT_BACKENDS


def _kt_backend_name(model_args: "ModelArguments") -> str:
    return str(getattr(model_args, "kt_backend", "")).upper()


def _positive_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, "")
    if value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"`{name}` must be a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")
    return parsed


def _field_default(dataclass_type: type, name: str) -> Any:
    for field_info in fields(dataclass_type):
        if field_info.name != name:
            continue
        if field_info.default is not MISSING:
            return field_info.default
        if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            return field_info.default_factory()  # type: ignore[misc]
    return None


def _optimizer_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _verify_asym_cpu_adamw_args(
    model_args: "ModelArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if not model_args.use_asym_gemm:
        raise ValueError("`use_asym_cpu_adamw=true` requires `use_asym_gemm=true`.")
    if finetuning_args.stage != "sft":
        raise ValueError("AsymGEMM CPU AdamW only supports SFT stage.")
    if not training_args.do_train:
        raise ValueError("AsymGEMM CPU AdamW requires `do_train=true`.")
    if finetuning_args.finetuning_type != "lora":
        raise ValueError("AsymGEMM CPU AdamW only supports LoRA finetuning.")
    if not finetuning_args.pure_bf16:
        raise ValueError("AsymGEMM CPU AdamW requires `pure_bf16=true`.")
    if training_args.deepspeed is not None or is_deepspeed_zero3_enabled():
        raise ValueError("AsymGEMM CPU AdamW cannot be combined with DeepSpeed or ZeRO.")
    if training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
        # gb200_dp.md D2: per-rank CPU AdamW under 2-rank DDP (asym_dp2) is supported; the
        # hook-based grad/weight offload stays disabled (see the guards below).
        if os.environ.get("ASYM_DP") != "1":
            raise ValueError("AsymGEMM CPU AdamW requires single-process single-device training.")
    if finetuning_args.asym_cpu_adamw_backend not in {"torch", "deepspeed"}:
        raise ValueError("`asym_cpu_adamw_backend` must be either `torch` or `deepspeed`.")
    if not finetuning_args.asym_cpu_adamw_fp32_master:
        raise ValueError("AsymGEMM CPU AdamW v1 requires `asym_cpu_adamw_fp32_master=true`.")
    if finetuning_args.asym_cpu_adamw_grad_offload and training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
        raise ValueError("AsymGEMM CPU AdamW grad offload requires single-process single-device training.")
    if finetuning_args.asym_cpu_adamw_weight_offload and training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
        raise ValueError("AsymGEMM CPU AdamW weight offload requires single-process single-device training.")
    if finetuning_args.asym_cpu_adamw_weight_offload and not finetuning_args.asym_cpu_adamw_grad_offload:
        raise ValueError("`asym_cpu_adamw_weight_offload=true` requires `asym_cpu_adamw_grad_offload=true`.")
    default_optim = _field_default(TrainingArguments, "optim")
    if default_optim is not None and _optimizer_name(training_args.optim) != _optimizer_name(default_optim):
        raise ValueError(
            "AsymGEMM CPU AdamW replaces the Trainer optimizer; keep `optim` at the installed Transformers default."
        )
    if training_args.load_best_model_at_end:
        raise ValueError("AsymGEMM CPU AdamW does not support `load_best_model_at_end` before checkpoint support.")
    if training_args.resume_from_checkpoint is not None:
        raise ValueError("AsymGEMM CPU AdamW does not support checkpoint resume before Stage 4.")
    if finetuning_args.loraplus_lr_ratio is not None:
        raise ValueError("AsymGEMM CPU AdamW is incompatible with LoRA+.")
    custom_optimizer_flags = {
        "use_galore": finetuning_args.use_galore,
        "use_apollo": finetuning_args.use_apollo,
        "use_badam": finetuning_args.use_badam,
        "use_adam_mini": finetuning_args.use_adam_mini,
        "use_muon": finetuning_args.use_muon,
        "use_mca": finetuning_args.use_mca,
        "use_hyper_parallel": finetuning_args.use_hyper_parallel,
    }
    enabled_custom = [name for name, enabled in custom_optimizer_flags.items() if enabled]
    if enabled_custom:
        raise ValueError(
            "AsymGEMM CPU AdamW is incompatible with other custom optimizer paths: "
            + ", ".join(enabled_custom)
            + "."
        )


def _arm_kt_route_rank_limit() -> int | None:
    explicit_limit = _positive_int_env("KT_ARM_SFT_MAX_ROUTE_RANK_WORK", None)
    if explicit_limit is not None:
        return explicit_limit
    if is_env_enabled("KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK"):
        return None
    return int(_positive_int_env("KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK", _ARM_KT_DEFAULT_MAX_ROUTE_RANK_WORK))


def _arm_kt_effective_route_qlen(logical_qlen: int) -> int:
    token_chunk_size = _positive_int_env("KT_ARM_SFT_TOKEN_CHUNK_SIZE", None)
    if token_chunk_size is not None and token_chunk_size < logical_qlen:
        return token_chunk_size
    return logical_qlen


def _ensure_arm_kt_backward_scratch_limit() -> None:
    _positive_int_env("KT_ARM_SFT_BACKWARD_THREADS", None)
    _positive_int_env("KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES", None)
    default_limit = _positive_int_env(
        "KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES", _ARM_KT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES
    )
    if os.environ.get("KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES", "").strip():
        return
    if is_env_enabled("KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH"):
        return
    os.environ["KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES"] = str(default_limit)


def _verify_arm_kt_route_rank_work(
    data_args: "DataArguments", training_args: "TrainingArguments", finetuning_args: "FinetuningArguments"
) -> None:
    top_k = _positive_int_env("KT_ARM_SFT_TOP_K", _ARM_KT_DEFAULT_TOP_K)
    limit = _arm_kt_route_rank_limit()
    logical_qlen = int(data_args.cutoff_len) * int(training_args.per_device_train_batch_size)
    effective_route_qlen = _arm_kt_effective_route_qlen(logical_qlen)
    route_rank_work = effective_route_qlen * int(top_k or _ARM_KT_DEFAULT_TOP_K) * int(finetuning_args.lora_rank)
    if limit is not None and route_rank_work > limit:
        raise ValueError(
            f"KT ARMBF16/ARM route-rank work {route_rank_work} exceeds "
            f"KT_ARM_SFT_MAX_ROUTE_RANK_WORK={limit}. Reduce per-device batch size, cutoff length, "
            "or LoRA rank; set KT_ARM_SFT_TOKEN_CHUNK_SIZE; raise the explicit limit; "
            "or set KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 "
            "only for validation."
        )


def _check_local_kt_import() -> None:
    try:
        import kt_kernel  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Local KT backend requires kt_kernel to be importable from the production source tree. "
            "Set PYTHONPATH to include "
            "/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel before launch."
        ) from exc


def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["TrainingArguments"] = None,
) -> None:
    if model_args.use_kt:
        if _is_local_kt_backend(model_args):
            _check_local_kt_import()
        else:
            check_version("kt-kernel", mandatory=True)
            check_version("transformers-kt", mandatory=True)
            check_version("accelerate-kt", mandatory=True)

    if model_args.use_asym_gemm:
        check_version("asym_gemm", mandatory=True)

    if model_args.use_unsloth:
        check_version("unsloth", mandatory=True)

    if model_args.enable_liger_kernel:
        check_version("liger-kernel", mandatory=True)

    if model_args.mixture_of_depths is not None:
        check_version("mixture-of-depth>=1.1.6", mandatory=True)

    if model_args.infer_backend == EngineName.VLLM:
        check_version("vllm>=0.4.3,<=0.11.0")
        check_version("vllm", mandatory=True)
    elif model_args.infer_backend == EngineName.SGLANG:
        check_version("sglang>=0.4.5")
        check_version("sglang", mandatory=True)

    if finetuning_args.use_galore:
        check_version("galore_torch", mandatory=True)

    if finetuning_args.use_apollo:
        check_version("apollo_torch", mandatory=True)

    if finetuning_args.use_badam:
        check_version("badam>=1.2.1", mandatory=True)

    if finetuning_args.use_adam_mini:
        check_version("adam-mini", mandatory=True)

    if finetuning_args.use_swanlab:
        check_version("swanlab", mandatory=True)

    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)

    if training_args is not None:
        if training_args.deepspeed:
            check_version("deepspeed", mandatory=True)

        if training_args.predict_with_generate:
            check_version("jieba", mandatory=True)
            check_version("nltk", mandatory=True)
            check_version("rouge_chinese", mandatory=True)


def _parse_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_train_mca_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_MCA_CLS:
    parser = HfArgumentParser(_TRAIN_MCA_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )

    _configure_mca_training_args(training_args, data_args, finetuning_args)

    return model_args, data_args, training_args, finetuning_args, generating_args


def _configure_mca_training_args(training_args, data_args, finetuning_args) -> None:
    """Patch training args to avoid args checking errors and sync MCA settings."""
    training_args.predict_with_generate = False
    training_args.generation_max_length = data_args.cutoff_len
    training_args.generation_num_beams = 1
    training_args.use_mca = True
    finetuning_args.use_mca = True


def _parse_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def get_ray_args(args: dict[str, Any] | list[str] | None = None) -> RayArguments:
    parser = HfArgumentParser(RayArguments)
    (ray_args,) = _parse_args(parser, args, allow_extra_keys=True)
    return ray_args


def get_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_kt:
            raise ValueError("KTransformers does not support lora reward model.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")

        if training_args.report_to and any(
            logger not in ("wandb", "tensorboard", "trackio", "none") for logger in training_args.report_to
        ):
            raise ValueError("PPO only accepts wandb, tensorboard, or trackio logger.")

    if finetuning_args.use_asym_cpu_adamw and not model_args.use_asym_gemm:
        raise ValueError("`use_asym_cpu_adamw=true` requires `use_asym_gemm=true`.")
    if finetuning_args.asym_cpu_adamw_grad_offload and not finetuning_args.use_asym_cpu_adamw:
        raise ValueError("`asym_cpu_adamw_grad_offload=true` requires `use_asym_cpu_adamw=true`.")
    if finetuning_args.asym_cpu_adamw_weight_offload and not finetuning_args.use_asym_cpu_adamw:
        raise ValueError("`asym_cpu_adamw_weight_offload=true` requires `use_asym_cpu_adamw=true`.")
    if finetuning_args.asym_cpu_adamw_weight_offload and not finetuning_args.asym_cpu_adamw_grad_offload:
        raise ValueError("`asym_cpu_adamw_weight_offload=true` requires `asym_cpu_adamw_grad_offload=true`.")

    if model_args.use_asym_gemm:
        os.environ.setdefault("USE_ASYM_GEMM", "1")
        if finetuning_args.stage != "sft":
            raise ValueError("AsymGEMM only supports SFT stage in the first implementation.")
        if not training_args.do_train:
            raise ValueError("AsymGEMM requires `do_train=true`.")
        if model_args.asym_backend == "asym" and training_args.parallel_mode == ParallelMode.DISTRIBUTED:
            # gb200_dp.md D2 (asym_dp2, Route A): 2-rank DDP over the LoRA params IS supported —
            # per-rank asym surgery + DDP reducer on adapters; CPUAdamW uses the step-time
            # (non-hook) grad path, so grads are read AFTER DDP finalization.
            if os.environ.get("ASYM_DP") != "1":
                raise ValueError("AsymGEMM CPU-offload LoRA-SFT does not support distributed/DDP training yet (set ASYM_DP=1 for the asym_dp2 backend).")
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("AsymGEMM only supports LoRA finetuning.")
        if model_args.asym_backend not in {"asym", "torch"}:
            raise ValueError("`asym_backend` must be either `asym` or `torch`.")
        if model_args.asym_precision != "bf16":
            raise ValueError("AsymGEMM first implementation supports `asym_precision=bf16` only.")
        from asym_gemm.integrations.lf import parse_lf_offload_modules

        parse_lf_offload_modules(model_args.asym_offload_modules)
        if model_args.asym_router_mode not in {"hf", "whole"}:
            raise ValueError("`asym_router_mode` must be either `hf` or `whole`.")
        if model_args.asym_router_mode == "whole" and model_args.moe_aux_loss_coef:
            raise ValueError("asym_router_mode=whole does not support moe_aux_loss_coef yet.")
        from asym_gemm.training.moe import parse_expert_recompute_policy_spec

        parse_expert_recompute_policy_spec(model_args.asym_expert_recompute_policy)
        if not (training_args.bf16 or finetuning_args.pure_bf16):
            raise ValueError("AsymGEMM BF16 LoRA-SFT requires `bf16=true` or `pure_bf16=true`.")
        if not finetuning_args.pure_bf16:
            raise ValueError("AsymGEMM smoke training requires `pure_bf16=true` to keep LoRA params in bf16.")
        if model_args.use_kt:
            raise ValueError("AsymGEMM is incompatible with KTransformers in the first implementation.")
        if model_args.use_unsloth:
            raise ValueError("AsymGEMM is incompatible with Unsloth in the first implementation.")
        if model_args.quantization_bit is not None:
            raise ValueError("AsymGEMM first implementation does not support quantization.")
        if finetuning_args.use_dora:
            raise ValueError("AsymGEMM first implementation does not support DoRA.")
        if finetuning_args.use_rslora:
            raise ValueError("AsymGEMM first implementation does not support RS-LoRA.")
        if finetuning_args.pissa_init:
            raise ValueError("AsymGEMM first implementation does not support PiSSA initialization.")
        if finetuning_args.additional_target is not None:
            raise ValueError("AsymGEMM first implementation does not support `additional_target`.")
        if finetuning_args.loraplus_lr_ratio is not None:
            raise ValueError("AsymGEMM first implementation does not support LoRA+.")
        if model_args.adapter_name_or_path is not None:
            raise ValueError("AsymGEMM first implementation does not support adapter resume/load.")
        if training_args.resume_from_checkpoint is not None:
            raise ValueError("AsymGEMM first implementation does not support checkpoint resume.")
        if training_args.deepspeed is not None or is_deepspeed_zero3_enabled():
            raise ValueError("AsymGEMM first implementation does not support DeepSpeed or ZeRO-3.")
        if finetuning_args.use_asym_cpu_adamw:
            _verify_asym_cpu_adamw_args(model_args, training_args, finetuning_args)

    if not (model_args.use_kt or model_args.use_asym_gemm) and training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict or training_args.predict_with_generate) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please make sure eval_dataset be provided or val_size >1e-6")

    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")

    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.use_galore and finetuning_args.galore_layerwise:
            raise ValueError("Distributed training does not support layer-wise GaLore.")

        if finetuning_args.use_apollo and finetuning_args.apollo_layerwise:
            raise ValueError("Distributed training does not support layer-wise APOLLO.")

        if finetuning_args.use_badam:
            if finetuning_args.badam_mode == "ratio":
                raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
            elif not is_deepspeed_zero3_enabled():
                raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")

    if training_args.deepspeed is not None and (finetuning_args.use_galore or finetuning_args.use_apollo):
        raise ValueError("GaLore and APOLLO are incompatible with DeepSpeed yet.")

    if not finetuning_args.use_mca and training_args.fp8 and model_args.quantization_bit is not None:
        raise ValueError("FP8 training is not compatible with quantization. Please disable one of them.")

    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if model_args.use_kt and is_deepspeed_zero3_enabled():
        raise ValueError("KTransformers is incompatible with DeepSpeed ZeRO-3.")

    if model_args.use_kt and finetuning_args.lora_dropout != 0:
        kt_backend = _kt_backend_name(model_args)
        if kt_backend not in _SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS:
            validated = ", ".join(sorted(_SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS))
            raise ValueError(
                f"KT backend {kt_backend} does not yet support lora_dropout > 0. "
                f"Validated backends: {validated}."
            )

    if model_args.use_kt and _is_local_kt_backend(model_args):
        if finetuning_args.stage != "sft":
            raise ValueError("Local KT backend currently supports only SFT.")
        if not training_args.do_train:
            raise ValueError("Local KT backend requires `do_train=true`.")
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Local KT backend currently supports only LoRA SFT.")
        if model_args.use_asym_gemm:
            raise ValueError("KTransformers is incompatible with AsymGEMM in the first implementation.")
        if training_args.deepspeed:
            raise ValueError("Local KT backend Phase 1 is single-GPU only; disable DeepSpeed.")
        if training_args.parallel_mode == ParallelMode.DISTRIBUTED:
            raise ValueError("Local KT backend Phase 1 is single-GPU only; launch with one visible GPU.")
        is_arm_kt_backend = _kt_backend_name(model_args) in _ARM_KT_BACKENDS
        if is_arm_kt_backend and bool(getattr(model_args, "kt_tp_enabled", False)):
            raise ValueError("KT ARMBF16/ARM aliases do not support `kt_tp_enabled=true`.")
        if (
            is_arm_kt_backend
            and getattr(model_args, "kt_threadpool_count", None) is not None
            and int(getattr(model_args, "kt_threadpool_count")) != 1
        ):
            raise ValueError("KT ARMBF16/ARM aliases require `kt_threadpool_count` to be unset or 1.")
        if (
            is_arm_kt_backend
            and getattr(model_args, "kt_num_gpu_experts", None) is not None
            and int(getattr(model_args, "kt_num_gpu_experts")) != 0
        ):
            raise ValueError("KT ARMBF16/ARM aliases do not support GPU experts; set `kt_num_gpu_experts=0`.")
        if is_arm_kt_backend and int(training_args.gradient_accumulation_steps) != 1:
            raise ValueError(
                "KT ARMBF16/ARM aliases currently require `gradient_accumulation_steps=1` because native expert-LoRA "
                "backward clears fused expert gradients on each backward pass."
            )
        if is_arm_kt_backend:
            _verify_arm_kt_route_rank_work(data_args, training_args, finetuning_args)
            _ensure_arm_kt_backward_scratch_limit()

    if model_args.use_asym_gemm and is_deepspeed_zero3_enabled():
        raise ValueError("AsymGEMM is incompatible with DeepSpeed ZeRO-3.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)
    _verify_trackio_args(training_args)

    if not finetuning_args.use_mca and training_args.fp8_enable_fsdp_float8_all_gather and not training_args.fp8:
        logger.warning_rank0("fp8_enable_fsdp_float8_all_gather requires fp8=True. Setting fp8=True.")
        model_args.fp8 = True

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning_rank0(
            "Remember to add embedding layers to `additional_target` to make the added tokens trainable."
        )

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning_rank0("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning_rank0("We recommend enable mixed precision training.")

    if (
        training_args.do_train
        and (finetuning_args.use_galore or finetuning_args.use_apollo)
        and not finetuning_args.pure_bf16
    ):
        logger.warning_rank0(
            "Using GaLore or APOLLO with mixed precision training may significantly increases GPU memory usage."
        )

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning_rank0("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning_rank0("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    if finetuning_args.finetuning_type == "lora":
        # https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/trainer.py#L782
        training_args.label_names = training_args.label_names or ["labels"]

    if "swanlab" in training_args.report_to and finetuning_args.use_swanlab:
        training_args.report_to.remove("swanlab")

    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.info_rank0("Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning_rank0("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not getattr(training_args, "overwrite_output_dir", False)  # for mca training args and transformers >= 5.0
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {training_args.resume_from_checkpoint}.")
            logger.info_rank0("Change `output_dir` or use `overwrite_output_dir` to avoid.")

    if (
        model_args.use_asym_gemm
        and finetuning_args.use_asym_cpu_adamw
        and training_args.resume_from_checkpoint is not None
    ):
        raise ValueError("AsymGEMM CPU AdamW does not support checkpoint resume before Stage 4.")

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning_rank0(
            f"Add {training_args.resume_from_checkpoint} to `adapter_name_or_path` to resume training from checkpoint."
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary
    logger.info(
        f"Process rank: {training_args.process_index}, "
        f"world size: {training_args.world_size}, device: {training_args.device}, "
        f"distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)

    if model_args.use_kt:
        kt_model_max_length = model_args.model_max_length
        if kt_model_max_length is not None:
            kt_model_max_length *= int(training_args.per_device_train_batch_size)
        logger.info(
            "KT logical length config: cutoff_len=%s per_device_train_batch_size=%s logical_qlen=%s",
            data_args.cutoff_len,
            training_args.per_device_train_batch_size,
            kt_model_max_length,
        )
        model_args.apply_kt_config(finetuning_args, training_args, kt_model_max_length)

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support bnb quantization (GPTQ and AWQ are supported).")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("vLLM only accepts a single adapter. Merge them first.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    # Post-process model arguments
    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        if data_args.cutoff_len != DataArguments().cutoff_len:  # override cutoff_len if it is not default
            model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
