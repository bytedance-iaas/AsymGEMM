from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from asym_gemm.training.frozen_linear import AsymExecutionStats
from asym_gemm.training.lora import TorchLoRALinear, freeze_non_lora_params, lora_parameters
from asym_gemm.training.qwen3_moe import AsymQwen3Experts, is_qwen3_experts, wrap_qwen3_experts


@dataclass
class LFAsymReport:
    qwen3_experts_wrapped: int = 0
    dense_lora_wrapped: int = 0
    trainable_lora_params: int = 0
    cpu_resident_base_bytes: int = 0
    gpu_resident_base_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def to_log_string(self) -> str:
        skipped = "; ".join(self.skipped) if self.skipped else "none"
        return (
            "AsymGEMM LoRA-SFT setup: "
            f"qwen3_experts_wrapped={self.qwen3_experts_wrapped}, "
            f"dense_lora_wrapped={self.dense_lora_wrapped}, "
            f"trainable_lora_params={self.trainable_lora_params}, "
            f"cpu_resident_base_bytes={self.cpu_resident_base_bytes}, "
            f"gpu_resident_base_bytes={self.gpu_resident_base_bytes}, "
            f"skipped={skipped}"
        )


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


def _validate_trainable_params(model: nn.Module) -> None:
    bad = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name and ".lora_A." not in name and ".lora_B." not in name]
    if bad:
        raise RuntimeError(f"AsymGEMM setup left non-LoRA trainable params: {bad[:20]}")
    router_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (".mlp.gate." in name or name.endswith(".mlp.gate.weight"))
    ]
    if router_trainable:
        raise RuntimeError(f"AsymGEMM setup must not train Qwen3 router params: {router_trainable[:20]}")


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
    strict: bool,
) -> tuple[nn.Module, LFAsymReport]:
    if backend not in {"asym", "torch"}:
        raise ValueError("backend must be 'asym' or 'torch'")
    if precision != "bf16":
        raise ValueError("AsymGEMM LLaMA-Factory integration supports bf16 only")
    if offload_modules not in {"routed_experts", "none"}:
        raise ValueError("offload_modules must be 'routed_experts' or 'none'")

    report = LFAsymReport()
    stats = AsymExecutionStats()
    wrap_experts = _targets_experts(raw_lora_target)
    offload_experts = backend == "asym" and offload_modules == "routed_experts"

    expert_replacements: list[tuple[str, nn.Module, AsymQwen3Experts]] = []
    if wrap_experts:
        for name, module in list(model.named_modules()):
            if is_qwen3_experts(module):
                wrapped = wrap_qwen3_experts(
                    module,
                    backend=backend,
                    precision=precision,
                    offload=offload_experts,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    lora_dtype=torch.bfloat16,
                    stats=stats,
                    strict=strict,
                )
                expert_replacements.append((name, module, wrapped))

        if not expert_replacements and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no Qwen3 packed expert modules.")

        for name, _old_module, new_module in expert_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)
            report.qwen3_experts_wrapped += 1
            report.cpu_resident_base_bytes += new_module.cpu_resident_base_bytes
            report.gpu_resident_base_bytes += new_module.gpu_resident_base_bytes

    expert_prefixes = [name for name, _old_module, _new_module in expert_replacements]
    dense_replacements: list[tuple[str, nn.Module, TorchLoRALinear]] = []
    for name, module in list(model.named_modules()):
        if not name or _is_under(name, expert_prefixes):
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

    if _is_all_target(raw_lora_target) and not dense_replacements and strict:
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
    return model, report


__all__ = ["LFAsymReport", "apply_lf_asym_lora"]
