from __future__ import annotations

import atexit
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from asym_gemm.training.frozen_linear import AsymExecutionStats
from asym_gemm.training.lora import (
    TorchLoRALinear,
    freeze_non_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    lora_parameters,
)
from asym_gemm.training.moe import parse_expert_recompute_policy_spec
from asym_gemm.training.qwen3_moe import AsymQwen3Experts, is_qwen3_experts, wrap_qwen3_experts
from asym_gemm.training.llama4_moe import AsymLlama4Moe, is_llama4_moe, wrap_llama4_moe


ASYM_LF_ADAPTER_FORMAT = "asym_gemm_lf_v1"


@dataclass
class LFAsymReport:
    packed_experts_wrapped: int = 0
    llama4_moes_wrapped: int = 0
    dense_lora_wrapped: int = 0
    trainable_lora_params: int = 0
    cpu_resident_base_bytes: int = 0
    gpu_resident_base_bytes: int = 0
    expert_recompute_policy: str = "none"
    skipped: list[str] = field(default_factory=list)
    stats: AsymExecutionStats | None = field(default=None, repr=False)

    @property
    def qwen3_experts_wrapped(self) -> int:
        return self.packed_experts_wrapped

    def to_log_string(self) -> str:
        skipped = "; ".join(self.skipped) if self.skipped else "none"
        return (
            "AsymGEMM LoRA-SFT setup: "
            f"packed_experts_wrapped={self.packed_experts_wrapped}, "
            f"llama4_moes_wrapped={self.llama4_moes_wrapped}, "
            f"dense_lora_wrapped={self.dense_lora_wrapped}, "
            f"trainable_lora_params={self.trainable_lora_params}, "
            f"cpu_resident_base_bytes={self.cpu_resident_base_bytes}, "
            f"gpu_resident_base_bytes={self.gpu_resident_base_bytes}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"skipped={skipped}"
        )

    def runtime_log_string(self) -> str:
        if self.stats is None:
            return "AsymGEMM LoRA-SFT runtime: stats=unavailable"
        fallbacks = (
            ";".join(f"{reason}:{count}" for reason, count in sorted(self.stats.fallback_reasons.items()))
            if self.stats.fallback_reasons
            else "none"
        )
        return (
            "AsymGEMM LoRA-SFT runtime: "
            f"asym_forward_calls={self.stats.asym_forward_calls}, "
            f"asym_dx_calls={self.stats.asym_dx_calls}, "
            f"torch_forward_calls={self.stats.torch_forward_calls}, "
            f"torch_dx_calls={self.stats.torch_dx_calls}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"reference_fallback_count={self.stats.reference_fallback_count}, "
            f"fallback_reasons={fallbacks}"
        )


def _env_true(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "y", "on"}


def _register_runtime_report(report: LFAsymReport) -> None:
    if not _env_true(os.environ.get("ASYM_GEMM_LF_LOG_RUNTIME_STATS")):
        return

    def _emit() -> None:
        print(report.runtime_log_string(), flush=True)

    atexit.register(_emit)


def _as_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [part.strip() for part in values.split(",") if part.strip()]
    return [str(value) for value in values]


def _is_all_target(raw_lora_target: Sequence[str] | str | None) -> bool:
    return any(target == "all" for target in _as_list(raw_lora_target))


def _targets_experts(raw_lora_target: Sequence[str] | str | None) -> bool:
    targets = set(_as_list(raw_lora_target))
    return bool(targets.intersection({"all", "experts", "gate", "up", "down", "gate_up_proj", "down_proj"}))


def _matches_target(name: str, module: nn.Module, targets: Sequence[str] | str | None) -> bool:
    target_list = _as_list(targets)
    if not target_list:
        return False
    child = name.rsplit(".", 1)[-1]
    return any(name == target or name.endswith(f".{target}") or child == target for target in target_list)


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _layer_profile_prefix_from_module_name(name: str, suffix: str) -> str:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return f"layers.{parts[index + 1]}.{suffix}"
    return f"layers.unknown.{suffix}"


def _qwen3_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "mlp.experts")


def _gemma4_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "experts")


def _llama4_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "feed_forward")


def _packed_expert_family(module: nn.Module) -> str:
    class_name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    config = getattr(module, "config", None)
    config_type = str(getattr(config, "model_type", "")).lower()
    if "gemma4" in class_name or "gemma4" in module_name or config_type == "gemma4":
        return "gemma4"
    if "qwen3" in class_name or "qwen3" in module_name or config_type in {"qwen3_moe", "qwen3_vl_moe"}:
        return "qwen3"
    return "packed"


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters(recurse=False):
        return torch.device(param.device), param.dtype
    for buffer in module.buffers(recurse=False):
        return torch.device(buffer.device), buffer.dtype
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device()), torch.bfloat16
    return torch.device("cpu"), torch.bfloat16


def _is_under(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _is_router_module_name(name: str) -> bool:
    parts = name.split(".")
    return "router" in parts or name.endswith(".mlp.gate") or ".mlp.gate." in name


def _validate_trainable_params(model: nn.Module) -> None:
    bad = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name and ".lora_A." not in name and ".lora_B." not in name]
    if bad:
        raise RuntimeError(f"AsymGEMM setup left non-LoRA trainable params: {bad[:20]}")
    router_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (
            ".mlp.gate." in name
            or name.endswith(".mlp.gate.weight")
            or ".router." in name
            or name.endswith(".router.weight")
            or ".feed_forward.router." in name
            or name.endswith(".feed_forward.router.weight")
        )
    ]
    if router_trainable:
        raise RuntimeError(f"AsymGEMM setup must not train router params: {router_trainable[:20]}")


def count_lora_wrapped_modules(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and not isinstance(module, AsymQwen3Experts)
    )


def apply_lf_asym_lora(
    model: nn.Module,
    *,
    raw_lora_target: Sequence[str] | str,
    dense_target_modules: Sequence[str] | str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload_modules: Literal["routed_experts", "none"],
    expert_recompute_policy: str = "none",
    wrap_dense: bool = True,
    preexisting_dense_lora_wrapped: int = 0,
    strict: bool = True,
) -> tuple[nn.Module, LFAsymReport]:
    if backend not in {"asym", "torch"}:
        raise ValueError("backend must be 'asym' or 'torch'")
    if precision != "bf16":
        raise ValueError("AsymGEMM LLaMA-Factory integration supports bf16 only")
    if offload_modules not in {"routed_experts", "none"}:
        raise ValueError("offload_modules must be 'routed_experts' or 'none'")

    recompute_config = parse_expert_recompute_policy_spec(expert_recompute_policy)
    report = LFAsymReport(expert_recompute_policy=recompute_config.label)
    report.dense_lora_wrapped = int(preexisting_dense_lora_wrapped)
    stats = AsymExecutionStats()
    report.stats = stats
    wrap_experts = _targets_experts(raw_lora_target)
    offload_experts = backend == "asym" and offload_modules == "routed_experts"

    expert_replacements: list[tuple[str, nn.Module, nn.Module, str]] = []
    if wrap_experts:
        for name, module in list(model.named_modules()):
            if is_qwen3_experts(module):
                family = _packed_expert_family(module)
                wrapped = wrap_qwen3_experts(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    stats=stats,
                    strict=strict,
                )
                wrapped.asym_expert_family = family
                if family == "gemma4":
                    wrapped.profile_prefix = _gemma4_profile_prefix_from_module_name(name)
                else:
                    wrapped.profile_prefix = _qwen3_profile_prefix_from_module_name(name)
                expert_replacements.append((name, module, wrapped, name))
            elif is_llama4_moe(module):
                wrapped = wrap_llama4_moe(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    expert_recompute_policy=recompute_config.label,
                    stats=stats,
                    strict=strict,
                )
                wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                expert_replacements.append((name, module, wrapped, f"{name}.experts"))

        if not expert_replacements and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.")

        for name, _old_module, new_module, _skip_prefix in expert_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)
            if isinstance(new_module, AsymLlama4Moe):
                report.llama4_moes_wrapped += 1
            else:
                report.packed_experts_wrapped += 1
            report.cpu_resident_base_bytes += new_module.cpu_resident_base_bytes
            report.gpu_resident_base_bytes += new_module.gpu_resident_base_bytes

    expert_prefixes = [skip_prefix for _name, _old_module, _new_module, skip_prefix in expert_replacements]
    dense_replacements: list[tuple[str, nn.Module, TorchLoRALinear]] = []
    if wrap_dense:
        for name, module in list(model.named_modules()):
            if not name or _is_under(name, expert_prefixes):
                continue
            if _is_router_module_name(name):
                report.skipped.append(f"{name}:router")
                continue
            if ".lora_A." in name or ".lora_B." in name or isinstance(module, TorchLoRALinear):
                continue
            if not _matches_target(name, module, dense_target_modules):
                continue
            if not isinstance(module, nn.Linear):
                report.skipped.append(f"{name}:not_nn_linear:{type(module).__name__}")
                continue
            device, dtype = _module_device_dtype(module)
            dense_replacements.append(
                (
                    name,
                    module,
                    TorchLoRALinear(
                        module,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        device=device,
                        dtype=dtype,
                        lora_dtype=torch.bfloat16,
                        init_lora_weights="peft",
                        lora_dropout=lora_dropout,
                    ),
                )
            )

    if wrap_dense and _is_all_target(raw_lora_target) and not dense_replacements and report.dense_lora_wrapped == 0 and strict:
        raise ValueError("AsymGEMM requested dense LoRA target=all but found no dense nn.Linear modules.")

    for name, _old_module, new_module in dense_replacements:
        parent, child_name = _parent_and_child(model, name)
        _replace_child(parent, child_name, new_module)
        report.dense_lora_wrapped += 1
        report.gpu_resident_base_bytes += int(new_module.gpu_resident_base_weight_bytes)

    freeze_non_lora_params(model)
    _validate_trainable_params(model)
    report.trainable_lora_params = sum(param.numel() for param in lora_parameters(model))
    if report.trainable_lora_params == 0 and strict:
        raise ValueError("AsymGEMM setup produced zero trainable LoRA parameters.")
    _register_runtime_report(report)
    return model, report


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _infer_adapter_config(model: nn.Module, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "inference_mode": False,
        "asym_gemm": True,
        "asym_adapter_format": ASYM_LF_ADAPTER_FORMAT,
    }
    for module in model.modules():
        if isinstance(module, AsymQwen3Experts):
            family = getattr(module, "asym_expert_family", "qwen3")
            config.update(
                {
                    "asym_expert_format": "packed_gate_up_down",
                    "asym_expert_family": family,
                    "r": module.lora_rank,
                    "lora_alpha": module.lora_alpha,
                    "lora_dropout": module.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.expert_recompute_config.label,
                }
            )
            break
        if isinstance(module, AsymLlama4Moe):
            config.update(
                {
                    "asym_expert_format": "llama4_packed_moe",
                    "asym_expert_family": "llama4",
                    "r": module.experts.lora_rank,
                    "lora_alpha": module.experts.lora_alpha,
                    "lora_dropout": module.experts.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.experts.expert_recompute_config.label,
                }
            )
            break
    if metadata:
        config.update(_jsonable(dict(metadata)))
    return config


def get_asym_lora_state_dict(
    model: nn.Module,
    *,
    adapter_name: str = "default",
) -> OrderedDict[str, torch.Tensor]:
    state = get_lora_state_dict(model, adapter_name=adapter_name)
    return OrderedDict((name, tensor.detach().to(device="cpu").contiguous()) for name, tensor in state.items())


def save_asym_peft_adapter(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
    metadata: Mapping[str, Any] | None = None,
    safe_serialization: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = get_asym_lora_state_dict(model, adapter_name=adapter_name)
    if not state:
        raise ValueError("AsymGEMM adapter save found no LoRA parameters")

    config = _infer_adapter_config(model, metadata)
    (output_path / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("save_asym_peft_adapter requires safetensors when safe_serialization=True") from exc
        save_file(state, str(output_path / "adapter_model.safetensors"))
    else:
        torch.save(state, output_path / "adapter_model.bin")


def load_asym_peft_adapter(
    model: nn.Module,
    adapter_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
    strict: bool = True,
) -> None:
    adapter_path = Path(adapter_dir)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("load_asym_peft_adapter requires safetensors for adapter_model.safetensors") from exc
        state = load_file(str(safetensors_path))
    else:
        state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
    load_lora_state_dict(model, state, adapter_name=adapter_name, strict=strict)


__all__ = [
    "ASYM_LF_ADAPTER_FORMAT",
    "LFAsymReport",
    "apply_lf_asym_lora",
    "count_lora_wrapped_modules",
    "get_asym_lora_state_dict",
    "load_asym_peft_adapter",
    "save_asym_peft_adapter",
]
