from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import (
    AsymExecutionStats,
    AsymFrozenLinear,
    VALID_ASYM_PRECISIONS,
    _TORCH_GROUPED_MM as _FROZEN_TORCH_GROUPED_MM,
)
from .host_weight import HostWeight


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

_TORCH_GROUPED_MM = _FROZEN_TORCH_GROUPED_MM


def _require_lora_grouped_mm():
    if _TORCH_GROUPED_MM is None:
        raise RuntimeError("PyTorch grouped LoRA path requires torch.nn.functional.grouped_mm or torch._grouped_mm")
    return _TORCH_GROUPED_MM


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


def _reset_lora_weights(
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    *,
    init_lora_weights: Literal["asym", "peft"],
    generator: torch.Generator | None,
) -> None:
    if init_lora_weights == "asym":
        a = _randn(tuple(a_weight.shape), generator=generator, dtype=a_weight.dtype, scale=0.01)
        b = _randn(tuple(b_weight.shape), generator=generator, dtype=b_weight.dtype, scale=0.01)
        a_weight.copy_(a.to(device=a_weight.device))
        b_weight.copy_(b.to(device=b_weight.device))
        return
    if init_lora_weights == "peft":
        nn.init.kaiming_uniform_(a_weight, a=math.sqrt(5))
        nn.init.zeros_(b_weight)
        return
    raise ValueError("init_lora_weights must be 'asym' or 'peft'")


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


class _AsymLoRALinearWeightOffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, module: "AsymLoRALinear") -> torch.Tensor:
        if not isinstance(module.lora_dropout, nn.Identity):
            raise RuntimeError("Asym LoRA weight offload requires lora_dropout=0.0")
        module.gather_lora_weights()
        a = module.lora_a
        b = module.lora_b
        if a.dtype != module.lora_dtype or b.dtype != module.lora_dtype:
            raise RuntimeError("Asym LoRA weight offload found staged weights with the wrong dtype")
        if a.dim() != 2 or b.dim() != 2:
            raise RuntimeError("Asym LoRA weight offload expects 2D LoRA A/B weights")
        if int(a.shape[0]) != int(b.shape[1]):
            raise RuntimeError(f"Asym LoRA rank mismatch: A={tuple(a.shape)} B={tuple(b.shape)}")
        if x.shape[-1] != int(module.base_layer.in_features):
            raise ValueError(f"expected input last dim {module.base_layer.in_features}, got {x.shape[-1]}")

        input_shape = tuple(int(dim) for dim in x.shape)
        flat = x.reshape(-1, int(module.base_layer.in_features)).contiguous()
        x_lora = flat.to(dtype=module.lora_dtype).contiguous()
        low_rank = F.linear(x_lora, a)
        out = F.linear(low_rank, b) * float(module.scaling)

        ctx.module = module
        ctx.input_shape = input_shape
        ctx.input_dtype = x.dtype
        ctx.out_features = int(module.base_layer.out_features)
        ctx.save_for_backward(x_lora, low_rank)
        if bool(getattr(module, "_weight_offload_release_after_forward", True)):
            module.release_lora_weights()
        return out.reshape(*input_shape[:-1], ctx.out_features).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        module: AsymLoRALinear = ctx.module
        x_lora, low_rank = ctx.saved_tensors
        # With non-reentrant checkpointing, materializing saved tensors can replay
        # forward and release the offloaded LoRA group. Gather after that replay so
        # the weights used below are full staged tensors, not 0-size placeholders.
        module.gather_lora_weights()
        a = module.lora_a
        b = module.lora_b

        grad_x = grad_a = grad_b = None
        grad_flat = grad_output.reshape(-1, int(ctx.out_features)).to(dtype=module.lora_dtype).contiguous()
        grad_scaled = grad_flat * float(module.scaling)
        if ctx.needs_input_grad[2]:
            grad_b = grad_scaled.t().contiguous().matmul(low_rank).to(dtype=b.dtype)
        needs_low_rank_grad = bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
        d_low_rank = grad_scaled.matmul(b) if needs_low_rank_grad else None
        if ctx.needs_input_grad[1]:
            if d_low_rank is None:
                raise RuntimeError("internal error: missing low-rank gradient for LoRA-A")
            grad_a = d_low_rank.t().contiguous().matmul(x_lora).to(dtype=a.dtype)
        if ctx.needs_input_grad[0]:
            if d_low_rank is None:
                raise RuntimeError("internal error: missing low-rank gradient for input")
            grad_x = d_low_rank.matmul(a).reshape(ctx.input_shape).to(dtype=ctx.input_dtype)
        return grad_x, grad_a, grad_b, None


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
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
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
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        self._weight_offload = None
        object.__setattr__(self, "_weight_offload_owner", self)
        self._weight_offload_release_after_forward = True
        self._reset_lora(adapter_name, lora_generator, init_lora_weights=init_lora_weights)

    @classmethod
    def from_host_weight(
        cls,
        host_weight: HostWeight,
        *,
        bias: torch.Tensor | None = None,
        rank: int,
        alpha: float,
        backend: str,
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
    ) -> "AsymLoRALinear":
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        precision = str(precision).lower()
        if precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        resolved_device = device if device is not None else torch.device("cpu")
        resolved_lora_dtype = normalize_lora_dtype(lora_dtype)
        obj.base_layer = AsymFrozenLinear.from_host_weight(
            host_weight,
            bias=bias,
            backend=backend,
            stats=stats,
            precision=precision,
        )
        obj.lora_A = nn.ModuleDict(
            {adapter_name: nn.Linear(host_weight.in_features, rank, bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        obj.lora_B = nn.ModuleDict(
            {adapter_name: nn.Linear(rank, host_weight.out_features, bias=False, device=resolved_device, dtype=resolved_lora_dtype)}
        )
        obj.active_adapter = adapter_name
        obj.lora_dtype = resolved_lora_dtype
        obj.scaling = float(alpha) / float(rank)
        obj.precision = precision
        obj.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        obj._weight_offload = None
        object.__setattr__(obj, "_weight_offload_owner", obj)
        obj._weight_offload_release_after_forward = True
        obj._reset_lora(adapter_name, lora_generator, init_lora_weights=init_lora_weights)
        return obj

    @property
    def base(self) -> AsymFrozenLinear:
        return self.base_layer

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    def _reset_lora(
        self,
        adapter_name: str,
        generator: torch.Generator | None,
        *,
        init_lora_weights: Literal["asym", "peft"],
    ) -> None:
        with torch.no_grad():
            a_weight = self.lora_A[adapter_name].weight
            b_weight = self.lora_B[adapter_name].weight
            _reset_lora_weights(a_weight, b_weight, init_lora_weights=init_lora_weights, generator=generator)

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.base_layer.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_hbm_saved_bytes

    def gather_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.gather_group(getattr(self, "_weight_offload_owner", self))

    def release_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.release_group(getattr(self, "_weight_offload_owner", self))

    def _uses_lora_weight_offload(self) -> bool:
        # NOTE: do not gate on torch.is_grad_enabled(). A checkpoint recompute runs the forward
        # under no_grad (inside an autograd.Function or reentrant checkpoint); the non-offload
        # path then hits the 0-numel at-rest LoRA weights ("vec (0)" size mismatch). The offload
        # path stages the weights itself (see forward: gather_lora_weights before apply), so it is
        # the correct path whenever weight offload is configured and we're training.
        return getattr(self, "_weight_offload", None) is not None and self.training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self._uses_lora_weight_offload():
            # Stage the LoRA bank BEFORE apply() captures self.lora_a/self.lora_b, so the
            # differentiable inputs are the full [r, k]/[n, r] weights, not the 0-numel
            # at-rest placeholders. Without this, a checkpoint recompute pass (gradient
            # checkpointing / sdpa-recompute) captures [0]-shaped inputs and backward returns
            # full-shaped grads -> "invalid gradient ... expected shape compatible with [0]". gather_group
            # is idempotent (no-op if already staged), so the normal path is unchanged.
            self.gather_lora_weights()
            lora = _AsymLoRALinearWeightOffloadFunction.apply(x, self.lora_a, self.lora_b, self)
            return base + lora.to(dtype=base.dtype)
        lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
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
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
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
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        self._reset_lora(adapter_name, lora_generator, init_lora_weights=init_lora_weights)

    @property
    def base_weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    def _reset_lora(
        self,
        adapter_name: str,
        generator: torch.Generator | None,
        *,
        init_lora_weights: Literal["asym", "peft"],
    ) -> None:
        with torch.no_grad():
            a_weight = self.lora_A[adapter_name].weight
            b_weight = self.lora_B[adapter_name].weight
            _reset_lora_weights(a_weight, b_weight, init_lora_weights=init_lora_weights, generator=generator)

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_nbytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
        lora = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_input)) * self.scaling
        return base + lora.to(dtype=base.dtype)


def _is_lora_parameter_name(name: str, *, adapter_name: str = "default") -> bool:
    if f".lora_A.{adapter_name}." in name or f".lora_B.{adapter_name}." in name:
        return True
    return "lora_" in name.lower()


def named_lora_parameters(model: nn.Module, *, adapter_name: str = "default") -> list[tuple[str, torch.nn.Parameter]]:
    try:
        named_params = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_params = model.named_parameters()
    return [
        (name, param)
        for name, param in named_params
        if _is_lora_parameter_name(name, adapter_name=adapter_name)
    ]


def lora_parameters(model: nn.Module, *, adapter_name: str = "default") -> list[torch.nn.Parameter]:
    return [param for _, param in named_lora_parameters(model, adapter_name=adapter_name)]


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
    lora_dropout: float = 0.0,
    init_lora_weights: Literal["asym", "peft"] = "asym",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    pin_memory: bool | None = None,
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
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            device=device,
            dtype=dtype,
            pin_memory=module.weight.device.type == "cuda" if pin_memory is None else pin_memory,
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


@dataclass(frozen=True)
class GroupedLoRAMetadata:
    offsets: torch.Tensor
    experts: torch.Tensor
    active_experts: torch.Tensor
    active_offsets: torch.Tensor
    dense_expert_weights: bool = False

    @property
    def active_groups(self) -> int:
        return int(self.active_offsets.numel())


def prepare_grouped_lora_metadata(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    dense_experts: bool = False,
) -> GroupedLoRAMetadata:
    memo = getattr(offsets, "_asym_lora_meta_memo", None)
    if memo is not None and bool(dense_experts) in memo:
        return memo[bool(dense_experts)]
    starts = offsets[:-1]
    ends = offsets[1:]
    active = ends > starts
    active_experts = experts[:-1][active].to(dtype=torch.long)
    active_offsets = ends[active].to(dtype=torch.int32).contiguous()
    if dense_experts:
        dense_expert_weights = int(active_offsets.numel()) == int(ends.numel())
        result = GroupedLoRAMetadata(
            offsets=offsets,
            experts=experts,
            active_experts=active_experts,
            active_offsets=active_offsets,
            dense_expert_weights=dense_expert_weights,
        )
    else:
        result = GroupedLoRAMetadata(
            offsets=offsets,
            experts=experts,
            active_experts=active_experts,
            active_offsets=active_offsets,
            dense_expert_weights=False,
        )
    if memo is None:
        try:
            offsets._asym_lora_meta_memo = {bool(dense_experts): result}  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        memo[bool(dense_experts)] = result
    return result


def _metadata_or_prepare(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata: GroupedLoRAMetadata | None,
) -> GroupedLoRAMetadata:
    return prepare_grouped_lora_metadata(offsets, experts) if metadata is None else metadata


def _grouped_lora_torch_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None = None,
) -> torch.Tensor:
    metadata = _metadata_or_prepare(offsets, experts, metadata)
    if int(metadata.active_offsets.numel()) == 0:
        return x.new_empty((0, int(weight.shape[1])))

    mat1 = x.contiguous()
    if metadata.dense_expert_weights and int(weight.shape[0]) == metadata.active_groups:
        selected = weight
    else:
        selected = weight.index_select(0, metadata.active_experts)
    mat2 = selected.transpose(-1, -2)
    grouped_mm = _require_lora_grouped_mm()
    return grouped_mm(mat1, mat2, offs=metadata.active_offsets)


def _grouped_lora_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None = None,
) -> torch.Tensor:
    metadata = _metadata_or_prepare(offsets, experts, metadata)
    offsets_cpu = metadata.offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    experts_cpu = metadata.experts.detach().to(device="cpu", dtype=torch.long).tolist()
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
    *,
    metadata: GroupedLoRAMetadata | None = None,
) -> torch.Tensor:
    """Run an expert-grouped LoRA projection with [E, out, in] weights."""

    if x.numel() == 0:
        return x.new_empty((0, int(weight.shape[1])))
    if x.device.type == "cuda" or weight.device.type == "cuda":
        return _grouped_lora_torch_mm(x, weight, offsets, experts, metadata=metadata)
    return _grouped_lora_fallback(x, weight, offsets, experts, metadata=metadata)


def grouped_expert_lora_cpu_left(
    x_cpu: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None = None,
    compiled_dims: str = "nk",
    output_dtype: torch.dtype | str = torch.bfloat16,
    stats: AsymExecutionStats | None = None,
) -> torch.Tensor:
    """Run expert LoRA-A with a pinned CPU left operand and CUDA weights."""

    from .cpu_left import grouped_expert_lora_cpu_left as _impl

    return _impl(
        x_cpu,
        weight,
        offsets,
        experts,
        metadata=metadata,
        compiled_dims=compiled_dims,
        output_dtype=output_dtype,
        stats=stats,
    )


def grouped_expert_lora_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two same-routed expert LoRA projections as one grouped_mm on CUDA."""

    if x0.numel() == 0:
        empty = x0.new_empty((0, int(weight0.shape[1])))
        return empty, empty
    metadata = _metadata_or_prepare(offsets, experts, metadata)
    if x0.device.type != "cuda" or weight0.device.type != "cuda":
        return (
            grouped_expert_lora(x0, weight0, offsets, experts, metadata=metadata),
            grouped_expert_lora(x1, weight1, offsets, experts, metadata=metadata),
        )
    if int(metadata.active_offsets.numel()) == 0:
        empty = x0.new_empty((0, int(weight0.shape[1])))
        return empty, empty

    rows = int(x0.shape[0])
    if metadata.dense_expert_weights and int(weight0.shape[0]) == metadata.active_groups:
        selected0 = weight0
        selected1 = weight1
    else:
        selected0 = weight0.index_select(0, metadata.active_experts)
        selected1 = weight1.index_select(0, metadata.active_experts)
    mat1 = torch.cat((x0.contiguous(), x1.contiguous()), dim=0)
    mat2 = torch.cat((selected0, selected1), dim=0).transpose(-1, -2)
    offs = torch.cat((metadata.active_offsets, metadata.active_offsets + rows), dim=0).contiguous()
    grouped_mm = _require_lora_grouped_mm()
    out = grouped_mm(mat1, mat2, offs=offs)
    return out[:rows], out[rows:]


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

    def prepare_metadata(
        self,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool = False,
    ) -> GroupedLoRAMetadata:
        return prepare_grouped_lora_metadata(offsets, experts, dense_experts=dense_experts)

    def _prepare_compute(
        self,
        x: torch.Tensor,
        weights: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        compute_dtype = self.lora_dtype
        if x.device.type == "cuda":
            return x, tuple(weights)
        return (
            x.to(dtype=compute_dtype),
            tuple(weight.to(device=x.device, dtype=compute_dtype) for weight in weights),
        )

    def forward_gate_up(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        out_dtype: torch.dtype,
        *,
        metadata: GroupedLoRAMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            empty = x.new_empty((0, self.config.intermediate_size))
            return empty, empty

        gate_a = self.gate_lora_a
        up_a = self.up_lora_a
        gate_b = self.gate_lora_b
        up_b = self.up_lora_b
        x_compute, prepared = self._prepare_compute(x, (gate_a, up_a, gate_b, up_b))
        gate_a, up_a, gate_b, up_b = prepared
        metadata = self.prepare_metadata(offsets, experts) if metadata is None else metadata

        rank = int(gate_a.shape[1])
        gate_up_a = torch.cat((gate_a, up_a), dim=1)
        low_rank = grouped_expert_lora(x_compute, gate_up_a, offsets, experts, metadata=metadata)
        gate_low_rank, up_low_rank = low_rank.split(rank, dim=-1)
        gate_out, up_out = grouped_expert_lora_pair(
            gate_low_rank,
            up_low_rank,
            gate_b,
            up_b,
            offsets,
            experts,
            metadata=metadata,
        )
        if self.lora_scale != 1.0:
            gate_out = gate_out.mul_(self.lora_scale)
            up_out = up_out.mul_(self.lora_scale)
        return gate_out.to(dtype=out_dtype), up_out.to(dtype=out_dtype)

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        prefix: str,
        out_dtype: torch.dtype,
        *,
        metadata: GroupedLoRAMetadata | None = None,
    ) -> torch.Tensor:
        if prefix not in {"gate", "up", "down"}:
            raise ValueError(f"unsupported LoRA projection prefix {prefix!r}")
        out_features = self.config.intermediate_size if prefix in {"gate", "up"} else self.config.hidden_size
        if x.numel() == 0:
            return x.new_empty((0, out_features))

        a = self._weight(prefix, "a")
        b = self._weight(prefix, "b")
        x_compute, (a, b) = self._prepare_compute(x, (a, b))
        metadata = self.prepare_metadata(offsets, experts) if metadata is None else metadata
        low_rank = grouped_expert_lora(x_compute, a, offsets, experts, metadata=metadata)
        out = grouped_expert_lora(low_rank, b, offsets, experts, metadata=metadata)
        if self.lora_scale != 1.0:
            out = out.mul_(self.lora_scale)
        return out.to(dtype=out_dtype)


__all__ = [
    "AsymLoRALinear",
    "FrozenTorchLinear",
    "GroupedLoRAMetadata",
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
    "grouped_expert_lora_cpu_left",
    "grouped_expert_lora_pair",
    "load_adapter_state_dict",
    "load_lora_state_dict",
    "load_peft_adapter",
    "lora_parameters",
    "lora_state_hash",
    "named_lora_parameters",
    "normalize_lora_dtype",
    "prepare_grouped_lora_metadata",
    "save_peft_adapter",
]
