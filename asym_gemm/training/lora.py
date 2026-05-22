from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import (
    AsymExecutionStats,
    AsymFrozenLinear,
    VALID_ASYM_PRECISIONS,
    _TORCH_GROUPED_MM,
    _TORCH_GROUPED_MM_NAME,
)


QWEN_TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "attention_only": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp_only": ("gate_proj", "up_proj", "down_proj"),
    "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}

TOY_TARGET_MODULES: dict[str, dict[str, tuple[str, ...]]] = {
    "mm": {"all": ("linear", "matrix", "proj")},
    "mlp": {"all": ("fc1", "fc2")},
    "dense": QWEN_TARGET_MODULES,
    "moe": {"all": ("gate_proj", "up_proj", "down_proj", "gate", "up", "down")},
}

TARGET_MODULE_PRESETS: dict[str, tuple[str, ...]] = {
    "attention_only": QWEN_TARGET_MODULES["attention_only"],
    "mlp_only": QWEN_TARGET_MODULES["mlp_only"],
    "all": tuple(
        dict.fromkeys(
            (
                *QWEN_TARGET_MODULES["all"],
                *TOY_TARGET_MODULES["mm"]["all"],
                *TOY_TARGET_MODULES["mlp"]["all"],
                *TOY_TARGET_MODULES["moe"]["all"],
            )
        )
    ),
}


LORA_DTYPE_NAMES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
}


@dataclass
class LoRASetupReport:
    matched_module_names: list[str] = field(default_factory=list)
    skipped_module_reasons: OrderedDict[str, str] = field(default_factory=OrderedDict)
    replaced_module_count: int = 0
    trainable_adapter_parameter_count: int = 0
    frozen_base_parameter_count: int = 0


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(dtype=dtype).contiguous()


def normalize_lora_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in LORA_DTYPE_NAMES:
        raise ValueError(f"unsupported lora_dtype={dtype!r}; expected one of {tuple(LORA_DTYPE_NAMES)}")
    return LORA_DTYPE_NAMES[key]


def _source_weight(source: nn.Linear | torch.Tensor) -> torch.Tensor:
    if isinstance(source, nn.Linear):
        return source.weight
    if isinstance(source, torch.Tensor):
        return source
    raise TypeError(f"expected nn.Linear or torch.Tensor, got {type(source)!r}")


def _source_bias(source: nn.Linear | torch.Tensor) -> torch.Tensor | None:
    return source.bias if isinstance(source, nn.Linear) else None


def _resolve_device_dtype(
    source: nn.Linear | torch.Tensor,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    weight = _source_weight(source)
    return (
        torch.device(weight.device if device is None else device),
        weight.dtype if dtype is None else dtype,
    )


class FrozenTorchLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        weight = _source_weight(source)
        bias = _source_bias(source)
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(device=device, dtype=dtype).contiguous())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def weight_nbytes(self) -> int:
        return int(self.weight.numel() * self.weight.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class AsymLoRALinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear | torch.Tensor,
        *,
        rank: int,
        alpha: float,
        backend: str,
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        pin_memory: bool | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        precision = str(precision).lower()
        if precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        resolved_device, resolved_dtype = _resolve_device_dtype(source, device=device, dtype=dtype)
        resolved_lora_dtype = normalize_lora_dtype(lora_dtype)
        weight = _source_weight(source)
        bias = _source_bias(source)
        self.base_layer = AsymFrozenLinear(
            weight.detach().to(dtype=resolved_dtype),
            bias=None if bias is None else bias.detach().to(dtype=resolved_dtype),
            backend=backend,
            pin_memory=resolved_device.type == "cuda" if pin_memory is None else pin_memory,
            stats=stats,
            precision=precision,
        )
        self.lora_A = nn.ModuleDict(
            {adapter_name: nn.Linear(weight.shape[1], rank, bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        self.lora_B = nn.ModuleDict(
            {adapter_name: nn.Linear(rank, weight.shape[0], bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        self.active_adapter = adapter_name
        self.lora_dtype = resolved_lora_dtype
        self.scaling = float(alpha) / float(rank)
        self.precision = precision
        self._reset_lora(adapter_name, lora_generator)

    @property
    def base(self) -> AsymFrozenLinear:
        return self.base_layer

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    def _reset_lora(self, adapter_name: str, generator: torch.Generator | None) -> None:
        with torch.no_grad():
            a_weight = self.lora_A[adapter_name].weight
            b_weight = self.lora_B[adapter_name].weight
            a = _randn(tuple(a_weight.shape), generator=generator, dtype=a_weight.dtype, scale=0.01)
            b = _randn(tuple(b_weight.shape), generator=generator, dtype=b_weight.dtype, scale=0.01)
            a_weight.copy_(a.to(device=a_weight.device))
            b_weight.copy_(b.to(device=b_weight.device))

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.base_layer.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_hbm_saved_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = x.to(dtype=self.lora_dtype)
        lora = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_input)) * self.scaling
        return base + lora.to(dtype=base.dtype)


class TorchLoRALinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear | torch.Tensor,
        *,
        rank: int,
        alpha: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        adapter_name: str = "default",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        resolved_device, resolved_dtype = _resolve_device_dtype(source, device=device, dtype=dtype)
        resolved_lora_dtype = normalize_lora_dtype(lora_dtype)
        weight = _source_weight(source)
        self.base_layer = FrozenTorchLinear(source, device=resolved_device, dtype=resolved_dtype)
        self.lora_A = nn.ModuleDict(
            {adapter_name: nn.Linear(weight.shape[1], rank, bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        self.lora_B = nn.ModuleDict(
            {adapter_name: nn.Linear(rank, weight.shape[0], bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        self.active_adapter = adapter_name
        self.lora_dtype = resolved_lora_dtype
        self.scaling = float(alpha) / float(rank)
        self._reset_lora(adapter_name, lora_generator)

    @property
    def base_weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    def _reset_lora(self, adapter_name: str, generator: torch.Generator | None) -> None:
        with torch.no_grad():
            a_weight = self.lora_A[adapter_name].weight
            b_weight = self.lora_B[adapter_name].weight
            a = _randn(tuple(a_weight.shape), generator=generator, dtype=a_weight.dtype, scale=0.01)
            b = _randn(tuple(b_weight.shape), generator=generator, dtype=b_weight.dtype, scale=0.01)
            a_weight.copy_(a.to(device=a_weight.device))
            b_weight.copy_(b.to(device=b_weight.device))

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_nbytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = x.to(dtype=self.lora_dtype)
        lora = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_input)) * self.scaling
        return base + lora.to(dtype=base.dtype)


def _is_lora_parameter_name(name: str, *, adapter_name: str = "default") -> bool:
    if f".lora_A.{adapter_name}." in name or f".lora_B.{adapter_name}." in name:
        return True
    return "lora_" in name.lower()


def lora_parameters(model: nn.Module, *, adapter_name: str = "default") -> list[torch.nn.Parameter]:
    return [param for name, param in model.named_parameters() if _is_lora_parameter_name(name, adapter_name=adapter_name)]


def freeze_non_lora_params(model: nn.Module, *, adapter_name: str = "default") -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(bool(_is_lora_parameter_name(name, adapter_name=adapter_name)))


def get_lora_state_dict(model: nn.Module, *, adapter_name: str = "default") -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
        if _is_lora_parameter_name(name, adapter_name=adapter_name)
    )


def get_lora_state_names(model: nn.Module, *, adapter_name: str = "default") -> list[str]:
    return list(get_lora_state_dict(model, adapter_name=adapter_name).keys())


def lora_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def load_lora_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    adapter_name: str = "default",
    strict: bool = True,
) -> None:
    params = dict(model.named_parameters())
    missing = [name for name in state if name not in params]
    if missing:
        raise KeyError(f"adapter state contains unknown keys: {missing}")
    if strict:
        expected = set(get_lora_state_names(model, adapter_name=adapter_name))
        provided = set(state.keys())
        if expected != provided:
            raise KeyError(f"adapter key mismatch: missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def copy_lora_state(src: nn.Module, dst: nn.Module, *, adapter_name: str = "default") -> None:
    load_lora_state_dict(dst, get_lora_state_dict(src, adapter_name=adapter_name), adapter_name=adapter_name, strict=True)


adapter_state_dict = get_lora_state_dict
adapter_state_names = get_lora_state_names
adapter_state_hash = lora_state_hash
load_adapter_state_dict = load_lora_state_dict
copy_adapter_state = copy_lora_state


def _iter_lora_modules(model: nn.Module) -> list[nn.Module]:
    return [
        module
        for module in model.modules()
        if (hasattr(module, "lora_A") and hasattr(module, "lora_B")) or (hasattr(module, "lora_a") and hasattr(module, "lora_b"))
    ]


def copy_lora(src: nn.Module, dst: nn.Module, *, adapter_name: str = "default") -> None:
    src_modules = _iter_lora_modules(src)
    dst_modules = _iter_lora_modules(dst)
    if len(src_modules) != len(dst_modules):
        raise ValueError(f"LoRA module count mismatch: src={len(src_modules)} dst={len(dst_modules)}")
    with torch.no_grad():
        for src_module, dst_module in zip(src_modules, dst_modules):
            if hasattr(src_module, "lora_A") and hasattr(dst_module, "lora_A"):
                dst_module.lora_A[adapter_name].weight.copy_(src_module.lora_A[adapter_name].weight)
                dst_module.lora_B[adapter_name].weight.copy_(src_module.lora_B[adapter_name].weight)
            else:
                dst_module.lora_a.copy_(src_module.lora_a)
                dst_module.lora_b.copy_(src_module.lora_b)


def _target_suffixes_from_preset(target_preset: str) -> tuple[str, ...]:
    if target_preset not in TARGET_MODULE_PRESETS:
        raise ValueError(f"target_preset={target_preset!r} must be one of {tuple(TARGET_MODULE_PRESETS)}")
    return TARGET_MODULE_PRESETS[target_preset]


def _target_matcher(target_modules: Sequence[str] | str | None, target_preset: str) -> Callable[[str], bool]:
    if target_modules is None:
        suffixes = _target_suffixes_from_preset(target_preset)
        return lambda name: any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)
    if isinstance(target_modules, str):
        if "," in target_modules:
            suffixes = tuple(part.strip() for part in target_modules.split(",") if part.strip())
            return lambda name: any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)
        pattern = re.compile(target_modules)
        return lambda name: bool(pattern.search(name))
    suffixes = tuple(str(target) for target in target_modules)
    return lambda name: any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)


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


def add_asym_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_modules: Sequence[str] | str | None = None,
    target_preset: str = "all",
    backend: str = "asym",
    precision: str = "bf16",
    adapter_name: str = "default",
    lora_dtype: torch.dtype | str | None = torch.bfloat16,
    module_filter: Callable[[str, nn.Module], bool] | None = None,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> LoRASetupReport:
    matcher = _target_matcher(target_modules, target_preset)
    shared_stats = stats if stats is not None else AsymExecutionStats()
    report = LoRASetupReport()
    replacements: list[tuple[str, nn.Linear, AsymLoRALinear]] = []

    for name, module in list(model.named_modules()):
        if not name or ".lora_A." in name or ".lora_B." in name:
            continue
        if not matcher(name):
            continue
        if module_filter is not None and not module_filter(name, module):
            report.skipped_module_reasons[name] = "filtered"
            continue
        if isinstance(module, AsymLoRALinear):
            report.skipped_module_reasons[name] = "already_asym_lora"
            continue
        if not isinstance(module, nn.Linear):
            report.skipped_module_reasons[name] = f"not_nn_linear:{type(module).__name__}"
            continue
        report.frozen_base_parameter_count += int(module.weight.numel())
        if module.bias is not None:
            report.frozen_base_parameter_count += int(module.bias.numel())
        wrapped = AsymLoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=shared_stats,
            precision=precision,
            adapter_name=adapter_name,
            lora_dtype=lora_dtype,
            pin_memory=module.weight.device.type == "cuda",
        )
        replacements.append((name, module, wrapped))

    non_linear_matches = {
        name: reason
        for name, reason in report.skipped_module_reasons.items()
        if reason.startswith("not_nn_linear:")
    }
    if non_linear_matches and strict:
        raise TypeError(f"matched target modules must be nn.Linear instances: {non_linear_matches}")

    if not replacements and strict:
        skipped = dict(report.skipped_module_reasons)
        raise ValueError(f"no target modules were adapted for target_modules={target_modules!r} target_preset={target_preset!r}; skipped={skipped}")

    for name, _old_module, new_module in replacements:
        parent, child_name = _parent_and_child(model, name)
        _replace_child(parent, child_name, new_module)
        report.matched_module_names.append(name)

    report.replaced_module_count = len(replacements)
    report.trainable_adapter_parameter_count = sum(param.numel() for param in lora_parameters(model, adapter_name=adapter_name))
    return report


def save_peft_adapter(
    model: nn.Module,
    output_dir: Path | str,
    *,
    base_model_name_or_path: str,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    adapter_name: str = "default",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }
    (output_path / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("save_peft_adapter requires safetensors") from exc
    save_file(get_lora_state_dict(model, adapter_name=adapter_name), str(output_path / "adapter_model.safetensors"))


def load_peft_adapter(
    model: nn.Module,
    adapter_dir: Path | str,
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
            raise ImportError("load_peft_adapter requires safetensors for adapter_model.safetensors") from exc
        state = load_file(str(safetensors_path))
    else:
        state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
    load_lora_state_dict(model, state, adapter_name=adapter_name, strict=strict)


def _state_tensor(state: Mapping[str, Any], name: str) -> torch.Tensor:
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state field {name!r} is not a tensor")
    return value


def _stack_expert_weight(expert_states: Sequence[Mapping[str, Any]], name: str) -> torch.Tensor:
    if not expert_states:
        raise ValueError("cannot build grouped expert weight from an empty expert list")
    return torch.stack([_state_tensor(expert_state, name) for expert_state in expert_states], dim=0).contiguous()


def _grouped_lora_torch_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda" or x.device != weight.device:
        raise RuntimeError(
            f"grouped LoRA torch path using {_TORCH_GROUPED_MM_NAME} "
            "requires input and LoRA weight on the same CUDA device"
        )
    if offsets.device != x.device or experts.device != x.device:
        raise RuntimeError("grouped LoRA torch path requires offsets and experts on the input CUDA device")
    if x.dtype != weight.dtype:
        raise RuntimeError(
            f"grouped LoRA torch path requires matching input/weight dtype, got {x.dtype} and {weight.dtype}"
        )
    if x.dim() != 2 or weight.dim() != 3:
        raise ValueError("grouped LoRA torch path expects x=[M,K] and weight=[E,N,K]")
    if offsets.dim() != 1 or experts.dim() != 1 or int(offsets.numel()) != int(experts.numel()):
        raise ValueError("grouped LoRA torch path expects cumulative offsets and experts with matching sentinel length")

    starts = offsets[:-1]
    ends = offsets[1:]
    active = ends > starts
    active_experts = experts[:-1][active].to(dtype=torch.long)
    active_offsets = ends[active].to(dtype=torch.int32).contiguous()
    if int(active_offsets.numel()) == 0:
        return x.new_empty((0, int(weight.shape[1])))

    mat1 = x.contiguous()
    mat2 = weight.index_select(0, active_experts).transpose(-1, -2)
    return _TORCH_GROUPED_MM(mat1, mat2, offs=active_offsets)


def _grouped_lora_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
) -> torch.Tensor:
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    experts_cpu = experts.detach().to(device="cpu", dtype=torch.long).tolist()
    chunks: list[torch.Tensor] = []
    for group_idx, expert_idx in enumerate(experts_cpu[:-1]):
        start = int(offsets_cpu[group_idx])
        end = int(offsets_cpu[group_idx + 1])
        if end <= start:
            continue
        expert_weight = weight[int(expert_idx)].to(device=x.device, dtype=x.dtype)
        chunks.append(F.linear(x[start:end], expert_weight))
    if chunks:
        return torch.cat(chunks, dim=0)
    return x.new_empty((0, int(weight.shape[1])))


def grouped_expert_lora(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
) -> torch.Tensor:
    """Run an expert-grouped LoRA projection with [E, out, in] weights."""

    if x.numel() == 0:
        return x.new_empty((0, int(weight.shape[1])))
    if x.device.type == "cuda" or weight.device.type == "cuda":
        return _grouped_lora_torch_mm(x, weight, offsets, experts)
    return _grouped_lora_fallback(x, weight, offsets, experts)


class PackedExpertLoRA(nn.Module):
    """Layer-local expert-indexed LoRA banks for grouped MoE execution."""

    _LORA_NAMES = (
        "gate_lora_a",
        "gate_lora_b",
        "up_lora_a",
        "up_lora_b",
        "down_lora_a",
        "down_lora_b",
    )

    def __init__(
        self,
        expert_states: Sequence[Mapping[str, Any]],
        *,
        config: Any,
        device: torch.device,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not expert_states:
            raise ValueError("PackedExpertLoRA requires at least one expert")
        self.config = config
        self.num_experts = len(expert_states)
        self.lora_scale = config.lora_scale
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        for name in self._LORA_NAMES:
            self.register_parameter(
                name,
                nn.Parameter(
                    _stack_expert_weight(expert_states, name).to(device=device, dtype=self.lora_dtype).clone()
                ),
            )

    def _weight(self, prefix: str, suffix: str) -> torch.Tensor:
        return getattr(self, f"{prefix}_lora_{suffix}")

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        prefix: str,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if prefix not in {"gate", "up", "down"}:
            raise ValueError(f"unsupported LoRA projection prefix {prefix!r}")
        out_features = self.config.intermediate_size if prefix in {"gate", "up"} else self.config.hidden_size
        if x.numel() == 0:
            return x.new_empty((0, out_features))

        compute_dtype = self.lora_dtype
        a = self._weight(prefix, "a")
        b = self._weight(prefix, "b")
        if x.device.type == "cuda":
            if x.dtype != compute_dtype:
                raise RuntimeError(f"CUDA grouped LoRA expects input dtype {compute_dtype}, got {x.dtype}")
            if a.device != x.device or b.device != x.device:
                raise RuntimeError("CUDA grouped LoRA expects LoRA parameters on the input device")
            if a.dtype != compute_dtype or b.dtype != compute_dtype:
                raise RuntimeError("CUDA grouped LoRA expects LoRA parameters in the configured LoRA dtype")
            x_compute = x
        else:
            x_compute = x.to(dtype=compute_dtype)
            a = a.to(device=x.device, dtype=compute_dtype)
            b = b.to(device=x.device, dtype=compute_dtype)
        low_rank = grouped_expert_lora(x_compute, a, offsets, experts)
        out = grouped_expert_lora(low_rank, b, offsets, experts) * self.lora_scale
        return out.to(dtype=out_dtype)


__all__ = [
    "AsymLoRALinear",
    "FrozenTorchLinear",
    "LoRASetupReport",
    "PackedExpertLoRA",
    "QWEN_TARGET_MODULES",
    "TARGET_MODULE_PRESETS",
    "TOY_TARGET_MODULES",
    "TorchLoRALinear",
    "add_asym_lora",
    "adapter_state_dict",
    "adapter_state_hash",
    "adapter_state_names",
    "copy_adapter_state",
    "copy_lora",
    "copy_lora_state",
    "freeze_non_lora_params",
    "get_lora_state_dict",
    "get_lora_state_names",
    "grouped_expert_lora",
    "load_adapter_state_dict",
    "load_lora_state_dict",
    "load_peft_adapter",
    "lora_parameters",
    "lora_state_hash",
    "normalize_lora_dtype",
    "save_peft_adapter",
]
