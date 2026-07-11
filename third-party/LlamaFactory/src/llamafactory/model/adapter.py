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

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, OFTConfig, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _get_qwen_moe_expert_lora_impl(default: str = "split-target-parameters") -> str:
    return os.environ.get("LF_QWEN_MOE_EXPERT_LORA_IMPL", default).strip().lower()


def _zero_aware_numel(param: torch.nn.Parameter) -> int:
    numel = int(param.numel())
    if numel == 0 and hasattr(param, "ds_numel"):
        try:
            numel = int(param.ds_numel)
        except (TypeError, ValueError):
            numel = 0
    return numel


def _maybe_write_lora_surface_sidecar(model: "PreTrainedModel") -> None:
    source_json = os.environ.get("ASYM_GEMM_LF_PROFILE_SOURCE_JSON", "").strip()
    explicit_path = os.environ.get("ASYM_GEMM_LF_LORA_SURFACE_JSON", "").strip()
    if not source_json and not explicit_path:
        return
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        rank = 0
    if rank != 0:
        return

    path = Path(explicit_path) if explicit_path else Path(source_json).with_name("lora_surface.json")
    trainable_params = 0
    all_params = 0
    peft_lora_params = 0
    peft_expert_lora_tensors = 0
    peft_expert_lora_params = 0

    def is_lora_name(name: str) -> bool:
        return ".lora_" in name or name.startswith("lora_") or "lora_A" in name or "lora_B" in name

    def is_expert_lora_name(name: str) -> bool:
        lowered = name.lower()
        return is_lora_name(name) and any(
            marker in lowered
            for marker in (
                ".experts.",
                ".expert.",
                ".shared_expert",
                "shared_experts",
                "block_sparse_moe",
                "moe.experts",
            )
        )

    for name, param in model.named_parameters():
        numel = _zero_aware_numel(param)
        all_params += numel
        if param.requires_grad:
            trainable_params += numel
        if is_lora_name(name) and param.requires_grad:
            peft_lora_params += numel
            if is_expert_lora_name(name):
                peft_expert_lora_tensors += 1
                peft_expert_lora_params += numel

    qwen_moe_expert_modules = 0
    qwen_moe_expert_tensors = 0
    qwen_moe_expert_params = 0
    try:
        from .model_utils.fused_moe_lora import QwenSplitMoeExpertParamWrapper

        for module in model.modules():
            if not isinstance(module, QwenSplitMoeExpertParamWrapper):
                continue
            module_params = [
                param
                for layer_name in getattr(module, "adapter_layer_names", ())
                for param in getattr(module, layer_name).values()
                if isinstance(param, torch.nn.Parameter) and param.requires_grad
            ]
            if module_params:
                qwen_moe_expert_modules += 1
                qwen_moe_expert_tensors += len(module_params)
                qwen_moe_expert_params += sum(_zero_aware_numel(param) for param in module_params)
    except Exception:
        pass

    payload = {
        "available": True,
        "source": "llamafactory_adapter_setup",
        "qwen_moe_expert_lora_impl": _get_qwen_moe_expert_lora_impl("split-target-parameters"),
        "trainable_parameters": trainable_params,
        "all_parameters": all_params,
        "peft_lora_parameters": peft_lora_params,
        "peft_expert_lora_tensors": peft_expert_lora_tensors,
        "peft_expert_lora_parameters": peft_expert_lora_params,
        "qwen_moe_expert_lora_modules": qwen_moe_expert_modules,
        "qwen_moe_expert_lora_tensors": qwen_moe_expert_tensors,
        "qwen_moe_expert_lora_parameters": qwen_moe_expert_params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_peft_with_qwen_expert_lora(
    model: "PreTrainedModel",
    adapter_path: str,
    is_trainable: bool,
    init_kwargs: dict,
    requested_mode: str = "auto",
) -> "PeftModel":
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(adapter_path, **init_kwargs)
    peft_type = getattr(peft_config, "peft_type", None)
    peft_type_value = str(getattr(peft_type, "value", peft_type) or "").upper()
    if peft_type_value == "LORA":
        from .model_utils.fused_moe_lora import infer_qwen_expert_lora_impl, prepare_qwen_moe_expert_lora_config

        mode = infer_qwen_expert_lora_impl(peft_config, requested_mode)
        saved_targets = getattr(peft_config, "target_modules", None) or []
        peft_config = prepare_qwen_moe_expert_lora_config(
            model,
            peft_config,
            mode,
            raw_lora_target=saved_targets,
            resolved_target_modules=saved_targets,
        )

    return PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=is_trainable,
        config=peft_config,
        **init_kwargs,
    )


def _is_asym_router_module_name(name: str) -> bool:
    parts = name.split(".")
    return "router" in parts or name.endswith(".mlp.gate") or ".mlp.gate." in name


def _target_matches_linear_name(name: str, target: str) -> bool:
    if "." in target:
        return name == target or name.endswith(f".{target}")

    leaf = target.rsplit(".", 1)[-1]
    return name == target or name.endswith(f".{target}") or name.rsplit(".", 1)[-1] == leaf


def _filter_asym_dense_peft_targets(model: "PreTrainedModel", target_modules: list[str]) -> list[str]:
    filtered = []
    for target in target_modules:
        leaf = target.rsplit(".", 1)[-1]
        if leaf in {"gate", "router"}:
            continue

        matching = [
            name
            for name, module in model.named_modules()
            if "Linear" in module.__class__.__name__
            and "Embedding" not in module.__class__.__name__
            and _target_matches_linear_name(name, target)
        ]
        if matching and all(_is_asym_router_module_name(name) for name in matching):
            continue

        filtered.append(target)

    return filtered


def split_asym_peft_dense_targets(
    model: "PreTrainedModel", target_modules: list[str], selection
) -> tuple[list[str], list[str]]:
    from asym_gemm.integrations.lf import classify_lf_component, component_is_selected

    filtered_targets = _filter_asym_dense_peft_targets(model, target_modules)
    peft_targets: list[str] = []
    asym_owned_targets: list[str] = []

    for target in filtered_targets:
        leaf = target.rsplit(".", 1)[-1]
        matching = [
            (name, module)
            for name, module in model.named_modules()
            if "Linear" in module.__class__.__name__
            and "Embedding" not in module.__class__.__name__
            and _target_matches_linear_name(name, target)
        ]
        if not matching:
            peft_targets.append(target)
            continue

        selected_names: list[str] = []
        peft_names: list[str] = []
        for name, module in matching:
            component = classify_lf_component(name, module)
            module_leaf = name.rsplit(".", 1)[-1]
            if component in {"attention", "linear_attention", "shared_experts", "lm_head", "mlp_dense"} and component_is_selected(
                component, module_leaf, selection
            ):
                selected_names.append(name)
            else:
                peft_names.append(name)

        asym_owned_targets.extend(selected_names)
        if peft_names:
            if selected_names:
                peft_targets.extend(peft_names)
            else:
                peft_targets.append(target)

    return list(dict.fromkeys(peft_targets)), sorted(set(asym_owned_targets))


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.extend(COMPOSITE_MODELS[model_type].projector_keys)

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        if finetuning_args.finetuning_type == "oft":
            logger.info_rank0("Fine-tuning method: OFT")
        else:
            logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_kt:
            assert len(model_args.adapter_name_or_path) == 1, "KTransformers model only accepts a single adapter"
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        requested_expert_lora_impl = _get_qwen_moe_expert_lora_impl("auto")

        for adapter in adapter_to_merge:
            model: LoraModel = _load_peft_with_qwen_expert_lora(
                model,
                adapter,
                is_trainable=False,
                init_kwargs=init_kwargs,
                requested_mode=requested_expert_lora_impl,
            )
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, finetuning_args, is_trainable=is_trainable)
            else:
                model = _load_peft_with_qwen_expert_lora(
                    model,
                    adapter_to_resume,
                    is_trainable=is_trainable,
                    init_kwargs=init_kwargs,
                    requested_mode=requested_expert_lora_impl,
                )

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        if finetuning_args.finetuning_type == "lora":
            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,
                "use_dora": finetuning_args.use_dora,
                "modules_to_save": finetuning_args.additional_target,
            }
        elif finetuning_args.finetuning_type == "oft":
            peft_kwargs = {
                "r": finetuning_args.oft_rank,
                "oft_block_size": finetuning_args.oft_block_size,
                "target_modules": target_modules,
                "module_dropout": finetuning_args.module_dropout,
                "modules_to_save": finetuning_args.additional_target,
            }

        if model_args.use_asym_gemm:
            if finetuning_args.finetuning_type != "lora":
                raise ValueError("AsymGEMM only supports LoRA finetuning.")

            from asym_gemm.integrations.lf import parse_lf_offload_modules
            from asym_gemm.integrations.liger_loss import install_asym_liger_loss_bridge
            from asym_gemm.integrations.peft_lf import adapt_lf_asym_peft_lora

            selection = parse_lf_offload_modules(model_args.asym_offload_modules)
            peft_dense_target_modules, asym_owned_dense_target_modules = split_asym_peft_dense_targets(
                model, list(target_modules), selection
            )
            if peft_dense_target_modules:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **{**peft_kwargs, "target_modules": peft_dense_target_modules},
                )
                model = get_peft_model(model, peft_config)

            model, report = adapt_lf_asym_peft_lora(
                model,
                raw_lora_target=finetuning_args.lora_target,
                dense_target_modules=peft_dense_target_modules,
                asym_owned_dense_target_modules=asym_owned_dense_target_modules,
                lora_rank=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                backend=model_args.asym_backend,
                precision=model_args.asym_precision,
                offload_modules=model_args.asym_offload_modules,
                expert_recompute_policy=model_args.asym_expert_recompute_policy,
                router_mode=model_args.asym_router_mode,
                strict=model_args.asym_strict,
            )
            if model_args.enable_liger_kernel:
                bridge_installed = install_asym_liger_loss_bridge(
                    model,
                    strict=bool(model_args.asym_strict and selection.lm_head),
                )
                if bridge_installed:
                    logger.info_rank0("Asym Liger loss bridge has been installed.")
            logger.info_rank0(report.to_log_string())
            return model
        elif model_args.use_kt:
            if finetuning_args.finetuning_type != "lora":
                raise ValueError("KTransformers only supports LoRA finetuning.")

            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **peft_kwargs)
            model = get_peft_model(model, peft_config)
        elif model_args.use_unsloth:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("Unsloth is currently not supported for OFT.")

            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
                from .model_utils.fused_moe_lora import prepare_qwen_moe_expert_lora_config

                peft_config = prepare_qwen_moe_expert_lora_config(
                    model,
                    peft_config,
                    _get_qwen_moe_expert_lora_impl("split-target-parameters"),
                    raw_lora_target=finetuning_args.lora_target,
                    resolved_target_modules=target_modules,
                )
            elif finetuning_args.finetuning_type == "oft":
                peft_config = OFTConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            model = get_peft_model(model, peft_config)

            # Only the top-level Llama4 conditional-generation path needs this post-load
            # normal-parameter bridge; Qwen3-MoE and llama4_text rely on the class patch.
            if model_args.enable_liger_kernel and getattr(model.config, "model_type", None) == "llama4":
                from asym_gemm.integrations.liger_loss import install_asym_liger_loss_bridge

                bridge_installed = install_asym_liger_loss_bridge(model, strict=False)
                if bridge_installed:
                    logger.info_rank0("Post-load Liger loss bridge has been installed.")

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if is_trainable:
        _maybe_write_lora_surface_sidecar(model)
        # AsymGEMM activation-offload BASELINES (off-layer / generic same-policy) for NON-asym backends
        # (zero3_offload / superoffload_mem). No-op unless the harness env requests a baseline; the
        # use_asym_gemm path returned early above, so this never affects the asym integration.
        if not model_args.use_asym_gemm:
            from asym_gemm.integrations.generic_offload_lf import maybe_apply_generic_offload

            maybe_apply_generic_offload(model)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantized models can only be used for the LoRA or OFT tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and is_deepspeed_zero3_enabled():
        logger.info_rank0("DeepSpeed ZeRO3 detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model
