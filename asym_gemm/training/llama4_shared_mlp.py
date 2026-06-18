from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .activation_offload import ActivationOffloadManager
from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, asym_bf16_cpu_right_matmul
from .lora import AsymLoRALinear, TorchLoRALinear, normalize_lora_dtype
from .offload import adopt_host_weight
from .profile_ranges import prof_range, scoped_name


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _shared_mlp_activation_offload_enabled() -> bool:
    return _env_enabled("ASYMM_EXPERT_ACT_OFFLOAD") or _env_enabled("ASYM_GEMM_LF_CONFIG_ASYMM_EXPERT_ACT_OFFLOAD")


def _is_silu_activation(fn: Any) -> bool:
    name = getattr(fn, "__name__", "")
    return fn is F.silu or name in {"silu", "silu_python"}


@dataclass(frozen=True)
class _SharedLinearLeaf:
    base: nn.Linear
    lora_a: torch.Tensor | None = None
    lora_b: torch.Tensor | None = None
    scaling: float | None = None
    preexisting_lora: bool = False


def _first_adapter_name(module: nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str) and active:
        return active
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(active_adapters, (list, tuple)) and active_adapters:
        return str(active_adapters[0])
    lora_a = getattr(module, "lora_A", None)
    if isinstance(lora_a, (nn.ModuleDict, dict)):
        if "default" in lora_a:
            return "default"
        keys = list(lora_a.keys())
        if keys:
            return str(keys[0])
    return "default"


def _maybe_get_mapping_value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, (nn.ModuleDict, dict)):
        if key in mapping:
            return mapping[key]
        keys = list(mapping.keys())
        if keys:
            return mapping[keys[0]]
    return None


def _linear_base_from_leaf(module: nn.Module) -> nn.Linear | None:
    if isinstance(module, nn.Linear):
        return module
    get_base_layer = getattr(module, "get_base_layer", None)
    if callable(get_base_layer):
        base = get_base_layer()
        if isinstance(base, nn.Linear):
            return base
    base = getattr(module, "base_layer", None)
    if isinstance(base, nn.Linear):
        return base
    return None


def _extract_shared_linear_leaf(module: nn.Module, *, strict: bool) -> _SharedLinearLeaf | None:
    base = _linear_base_from_leaf(module)
    if base is None:
        if strict:
            raise TypeError(f"Llama4 shared expert leaf must be nn.Linear or PEFT LoRA-over-Linear, got {type(module).__name__}")
        return None

    lora_a_map = getattr(module, "lora_A", None)
    lora_b_map = getattr(module, "lora_B", None)
    if lora_a_map is None or lora_b_map is None:
        return _SharedLinearLeaf(base=base)

    adapter_name = _first_adapter_name(module)
    lora_a = _maybe_get_mapping_value(lora_a_map, adapter_name)
    lora_b = _maybe_get_mapping_value(lora_b_map, adapter_name)
    lora_a_weight = getattr(lora_a, "weight", None)
    lora_b_weight = getattr(lora_b, "weight", None)
    if not isinstance(lora_a_weight, torch.Tensor) or not isinstance(lora_b_weight, torch.Tensor):
        if strict:
            raise TypeError(f"Llama4 shared expert PEFT leaf {type(module).__name__} has unsupported LoRA tensors")
        return None

    scaling = None
    scaling_map = getattr(module, "scaling", None)
    if isinstance(scaling_map, dict):
        raw_scaling = scaling_map.get(adapter_name)
        if raw_scaling is None and scaling_map:
            raw_scaling = next(iter(scaling_map.values()))
        if raw_scaling is not None:
            scaling = float(raw_scaling)
    elif scaling_map is not None:
        try:
            scaling = float(scaling_map)
        except (TypeError, ValueError):
            scaling = None
    return _SharedLinearLeaf(
        base=base,
        lora_a=lora_a_weight,
        lora_b=lora_b_weight,
        scaling=scaling,
        preexisting_lora=True,
    )


def is_llama4_shared_mlp_leaf(module: nn.Module) -> bool:
    return _extract_shared_linear_leaf(module, strict=False) is not None


def _copy_preexisting_lora(dst: AsymLoRALinear | TorchLoRALinear, spec: _SharedLinearLeaf, *, strict: bool) -> None:
    if spec.lora_a is None or spec.lora_b is None:
        return
    dst_a = dst.lora_A[dst.active_adapter].weight
    dst_b = dst.lora_B[dst.active_adapter].weight
    if tuple(dst_a.shape) != tuple(spec.lora_a.shape) or tuple(dst_b.shape) != tuple(spec.lora_b.shape):
        message = (
            "preexisting Llama4 shared expert LoRA shape mismatch: "
            f"A {tuple(spec.lora_a.shape)} -> {tuple(dst_a.shape)}, "
            f"B {tuple(spec.lora_b.shape)} -> {tuple(dst_b.shape)}"
        )
        if strict:
            raise ValueError(message)
        return
    with torch.no_grad():
        dst_a.copy_(spec.lora_a.detach().to(device=dst_a.device, dtype=dst_a.dtype))
        dst_b.copy_(spec.lora_b.detach().to(device=dst_b.device, dtype=dst_b.dtype))


def _asym_base_forward(
    base: AsymFrozenLinear,
    x: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    out = asym_bf16_cpu_right_matmul(
        x,
        base.host_weight.weight,
        backend=base.backend,
        stats=stats,
        phase="forward",
        tag=tag,
        compiled_dims=base.compiled_dims,
        output_dtype=base.bf16_output_dtype,
    )
    if base.bias_cpu is not None:
        out = out + base.bias_cpu.to(device=out.device, dtype=out.dtype, non_blocking=base.bias_cpu.is_pinned())
    return out


def _asym_base_dx(
    base: AsymFrozenLinear,
    grad_output: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    return asym_bf16_cpu_right_matmul(
        grad_output,
        base.host_weight.weight,
        transpose_b=True,
        backend=base.backend,
        stats=stats,
        phase="dx",
        tag=tag,
        compiled_dims=base.compiled_dims,
        output_dtype=input_dtype,
    )


def _lora_forward(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_rank = x @ a.t()
    delta = low_rank @ b.t()
    if float(scale) != 1.0:
        delta = delta.mul(float(scale))
    return delta, low_rank


def _lora_backward(
    grad_output: torch.Tensor,
    low_rank: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_lora = grad_output if grad_output.dtype == b.dtype else grad_output.to(dtype=b.dtype)
    d_low_rank = grad_lora @ b
    if float(scale) != 1.0:
        d_low_rank = d_low_rank.mul(float(scale))
    grad_b = grad_lora.t().contiguous() @ low_rank
    if float(scale) != 1.0:
        grad_b = grad_b.mul(float(scale))
    grad_a = d_low_rank.t().contiguous() @ x
    grad_x = d_low_rank @ a
    return grad_x, grad_a.to(dtype=a.dtype), grad_b.to(dtype=b.dtype)


class _Llama4SharedMLPActivationOffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        down_a: torch.Tensor,
        down_b: torch.Tensor,
        layer: "AsymLlama4SharedMLP",
    ) -> torch.Tensor:
        if x.dim() < 1:
            raise ValueError("Llama4 shared expert MLP expects a tensor with a hidden dimension")
        weight_offload = getattr(layer, "_weight_offload", None) is not None
        if weight_offload:
            layer.gather_lora_weights()
            gate_a = layer.gate_proj.lora_a
            gate_b = layer.gate_proj.lora_b
            up_a = layer.up_proj.lora_a
            up_b = layer.up_proj.lora_b
            down_a = layer.down_proj.lora_a
            down_b = layer.down_proj.lora_b
        flat = x.reshape(-1, layer.hidden_size).contiguous()
        if flat.dtype != torch.bfloat16:
            raise ValueError(f"Llama4 shared expert activation offload requires bf16 input, got {flat.dtype}")

        manager = ActivationOffloadManager(pin_memory=True)
        with prof_range(layer._forward_range("activation_offload", "x_to_cpu")):
            x_cpu = manager.offload(flat, "shared.X")

        with prof_range(layer._forward_range("activation_offload", "gate_up")):
            gate_base = _asym_base_forward(
                layer.gate_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("shared_gate", "base_forward"),
            )
            up_base = _asym_base_forward(
                layer.up_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("shared_up", "base_forward"),
            )
            gate_delta, _ = _lora_forward(flat, gate_a, gate_b, scale=layer.lora_scale)
            up_delta, _ = _lora_forward(flat, up_a, up_b, scale=layer.lora_scale)
            gate = gate_base + gate_delta.to(dtype=gate_base.dtype)
            up = up_base + up_delta.to(dtype=up_base.dtype)
            del gate_base, up_base, gate_delta, up_delta

        with prof_range(layer._forward_range("activation_offload", "activation")):
            activated = layer.activation_fn(gate) * up
            del gate, up

        with prof_range(layer._forward_range("activation_offload", "down")):
            down_base = _asym_base_forward(
                layer.down_proj.base_layer,
                activated.to(dtype=torch.bfloat16).contiguous(),
                stats=layer.stats,
                tag=layer._profile_name("shared_down", "base_forward"),
            )
            down_delta, _ = _lora_forward(activated.to(dtype=down_a.dtype).contiguous(), down_a, down_b, scale=layer.lora_scale)
            out = down_base + down_delta.to(dtype=down_base.dtype)
            del down_base, down_delta, activated

        if weight_offload:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(gate_a, gate_b, up_a, up_b, down_a, down_b)
        ctx.weight_offload = bool(weight_offload)
        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.input_shape = tuple(int(dim) for dim in x.shape)
        ctx.input_dtype = x.dtype
        layer._last_activation_offload_stats = manager.snapshot()
        if weight_offload and bool(getattr(layer, "_weight_offload_release_after_forward", True)):
            layer.release_lora_weights()
        return out.reshape(*ctx.input_shape[:-1], layer.hidden_size)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        layer: AsymLlama4SharedMLP = ctx.layer
        if getattr(ctx, "weight_offload", False):
            layer.gather_lora_weights()
            gate_a = layer.gate_proj.lora_a
            gate_b = layer.gate_proj.lora_b
            up_a = layer.up_proj.lora_a
            up_b = layer.up_proj.lora_b
            down_a = layer.down_proj.lora_a
            down_b = layer.down_proj.lora_b
        else:
            gate_a, gate_b, up_a, up_b, down_a, down_b = ctx.saved_tensors
        manager: ActivationOffloadManager = ctx.manager
        x_stage = None

        grad_x = None
        try:
            with prof_range(layer._backward_range("activation_offload", "stage_x")):
                x_stage = manager.stage(ctx.x_cpu, tag="shared.X_for_backward")
                x_lora = x_stage.to(dtype=gate_a.dtype)

            with prof_range(layer._backward_range("activation_offload", "recompute_gate_up")):
                gate_base = _asym_base_forward(
                    layer.gate_proj.base_layer,
                    x_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("shared_gate", "recompute_base"),
                )
                up_base = _asym_base_forward(
                    layer.up_proj.base_layer,
                    x_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("shared_up", "recompute_base"),
                )
                gate_delta, gate_low_rank = _lora_forward(x_lora, gate_a, gate_b, scale=layer.lora_scale)
                up_delta, up_low_rank = _lora_forward(x_lora, up_a, up_b, scale=layer.lora_scale)
                gate = gate_base + gate_delta.to(dtype=gate_base.dtype)
                up = up_base + up_delta.to(dtype=up_base.dtype)
                del gate_base, up_base, gate_delta, up_delta

            with prof_range(layer._backward_range("activation_offload", "recompute_activation")):
                activated = layer.activation_fn(gate) * up
                activated_lora = activated.to(dtype=down_a.dtype).contiguous()
                _, down_low_rank = _lora_forward(activated_lora, down_a, down_b, scale=layer.lora_scale)

            with prof_range(layer._backward_range("activation_offload", "down_backward")):
                grad_2d = grad_output.reshape(-1, layer.hidden_size).to(dtype=torch.bfloat16).contiguous()
                grad_down_base_x = _asym_base_dx(
                    layer.down_proj.base_layer,
                    grad_2d,
                    stats=layer.stats,
                    tag=layer._profile_name("shared_down", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                grad_down_lora_x, grad_down_a, grad_down_b = _lora_backward(
                    grad_2d,
                    down_low_rank,
                    activated_lora,
                    down_a,
                    down_b,
                    scale=layer.lora_scale,
                )
                grad_act = grad_down_base_x + grad_down_lora_x.to(dtype=grad_down_base_x.dtype)
                del grad_down_base_x, grad_down_lora_x, down_low_rank, activated_lora

            with prof_range(layer._backward_range("activation_offload", "activation_backward")):
                silu_gate = layer.activation_fn(gate)
                grad_up = grad_act * silu_gate
                grad_gate = torch.ops.aten.silu_backward(grad_act * up, gate)
                del grad_act, silu_gate, activated, up

            with prof_range(layer._backward_range("activation_offload", "gate_up_backward")):
                grad_gate_base_x = _asym_base_dx(
                    layer.gate_proj.base_layer,
                    grad_gate.to(dtype=torch.bfloat16).contiguous(),
                    stats=layer.stats,
                    tag=layer._profile_name("shared_gate", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                grad_up_base_x = _asym_base_dx(
                    layer.up_proj.base_layer,
                    grad_up.to(dtype=torch.bfloat16).contiguous(),
                    stats=layer.stats,
                    tag=layer._profile_name("shared_up", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                grad_gate_lora_x, grad_gate_a, grad_gate_b = _lora_backward(
                    grad_gate,
                    gate_low_rank,
                    x_lora,
                    gate_a,
                    gate_b,
                    scale=layer.lora_scale,
                )
                grad_up_lora_x, grad_up_a, grad_up_b = _lora_backward(
                    grad_up,
                    up_low_rank,
                    x_lora,
                    up_a,
                    up_b,
                    scale=layer.lora_scale,
                )
                grad_x_2d = grad_gate_base_x + grad_up_base_x
                grad_x_2d.add_(grad_gate_lora_x.to(dtype=grad_x_2d.dtype))
                grad_x_2d.add_(grad_up_lora_x.to(dtype=grad_x_2d.dtype))
                grad_x = grad_x_2d.to(dtype=ctx.input_dtype).reshape(ctx.input_shape)
                del grad_gate, grad_up, gate_low_rank, up_low_rank, x_lora
        finally:
            manager.release_stage(x_stage, drop_cache=True)
            manager.release_cpu(ctx.x_cpu)
            layer._last_activation_offload_stats = manager.snapshot()

        return (
            grad_x,
            grad_gate_a,
            grad_gate_b,
            grad_up_a,
            grad_up_b,
            grad_down_a,
            grad_down_b,
            None,
        )


class AsymLlama4SharedMLP(nn.Module):
    """Llama4 shared expert MLP with Llama4-owned activation offload."""

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        offload: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if backend not in {"asym", "torch"}:
            raise ValueError("backend must be 'asym' or 'torch'")
        if precision != "bf16":
            raise ValueError("AsymLlama4SharedMLP supports bf16 only")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        gate_spec = _extract_shared_linear_leaf(getattr(source, "gate_proj", None), strict=strict)
        up_spec = _extract_shared_linear_leaf(getattr(source, "up_proj", None), strict=strict)
        down_spec = _extract_shared_linear_leaf(getattr(source, "down_proj", None), strict=strict)
        if gate_spec is None or up_spec is None or down_spec is None:
            raise TypeError(f"Llama4 shared expert source must expose gate/up/down linear leaves, got {type(source).__name__}")
        gate = gate_spec.base
        up = up_spec.base
        down = down_spec.base
        if gate.bias is not None or up.bias is not None or down.bias is not None:
            raise NotImplementedError("Llama4 shared expert wrapper currently supports bias-free MLPs only")
        self.hidden_size = int(gate.in_features)
        self.intermediate_size = int(gate.out_features)
        if int(up.in_features) != self.hidden_size or int(down.out_features) != self.hidden_size:
            raise ValueError("Llama4 shared expert gate/up/down shapes are inconsistent")
        if int(up.out_features) != self.intermediate_size or int(down.in_features) != self.intermediate_size:
            raise ValueError("Llama4 shared expert intermediate sizes are inconsistent")
        self.activation_fn = getattr(source, "activation_fn", None)
        if not callable(self.activation_fn):
            raise TypeError("Llama4 shared expert source must expose activation_fn")
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        preexisting_scalings = [
            spec.scaling
            for spec in (gate_spec, up_spec, down_spec)
            if spec.preexisting_lora and spec.scaling is not None
        ]
        if preexisting_scalings:
            first_scaling = float(preexisting_scalings[0])
            if strict and any(abs(float(value) - first_scaling) > 1e-6 for value in preexisting_scalings[1:]):
                raise ValueError(f"Llama4 shared expert LoRA scaling mismatch across gate/up/down: {preexisting_scalings}")
            if strict and abs(first_scaling - self.lora_scale) > 1e-6:
                raise ValueError(
                    f"Llama4 shared expert PEFT scaling {first_scaling} does not match requested alpha/r {self.lora_scale}"
                )
            self.lora_scale = first_scaling
            self.lora_alpha = first_scaling * float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        self.lora_dropout_p = float(lora_dropout)
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.profile_prefix = "layers.unknown.feed_forward.shared_expert"
        self._last_activation_offload_stats: dict[str, Any] = {}
        self._weight_offload = None
        self._weight_offload_release_after_forward = True
        self.preexisting_lora_leaf_count = sum(
            int(spec.preexisting_lora) for spec in (gate_spec, up_spec, down_spec)
        )

        if backend == "asym" and offload:
            for leaf_name, module in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                if strict and module.weight.device.type != "cpu":
                    raise RuntimeError(f"Llama4 shared expert {leaf_name} CPU offload requires CPU-first model loading")
                if strict and module.weight.dtype != torch.bfloat16:
                    raise RuntimeError(f"Llama4 shared expert {leaf_name} CPU offload requires bf16 source weights")

            self.gate_proj = AsymLoRALinear.from_host_weight(
                adopt_host_weight("shared_expert.gate_proj.weight", gate.weight, "shared_experts", pin_memory_policy="auto", strict=strict),
                rank=lora_rank,
                alpha=lora_alpha,
                backend=backend,
                stats=self.stats,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                lora_dtype=self.lora_dtype,
                precision=precision,
                init_lora_weights="peft",
                lora_dropout=lora_dropout,
            )
            self.up_proj = AsymLoRALinear.from_host_weight(
                adopt_host_weight("shared_expert.up_proj.weight", up.weight, "shared_experts", pin_memory_policy="auto", strict=strict),
                rank=lora_rank,
                alpha=lora_alpha,
                backend=backend,
                stats=self.stats,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                lora_dtype=self.lora_dtype,
                precision=precision,
                init_lora_weights="peft",
                lora_dropout=lora_dropout,
            )
            self.down_proj = AsymLoRALinear.from_host_weight(
                adopt_host_weight("shared_expert.down_proj.weight", down.weight, "shared_experts", pin_memory_policy="auto", strict=strict),
                rank=lora_rank,
                alpha=lora_alpha,
                backend=backend,
                stats=self.stats,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                lora_dtype=self.lora_dtype,
                precision=precision,
                init_lora_weights="peft",
                lora_dropout=lora_dropout,
            )
        elif backend == "asym":
            self.gate_proj = AsymLoRALinear(gate, rank=lora_rank, alpha=lora_alpha, backend=backend, stats=self.stats, lora_dtype=self.lora_dtype, precision=precision, init_lora_weights="peft", lora_dropout=lora_dropout)
            self.up_proj = AsymLoRALinear(up, rank=lora_rank, alpha=lora_alpha, backend=backend, stats=self.stats, lora_dtype=self.lora_dtype, precision=precision, init_lora_weights="peft", lora_dropout=lora_dropout)
            self.down_proj = AsymLoRALinear(down, rank=lora_rank, alpha=lora_alpha, backend=backend, stats=self.stats, lora_dtype=self.lora_dtype, precision=precision, init_lora_weights="peft", lora_dropout=lora_dropout)
        else:
            device = torch.device(gate.weight.device)
            self.gate_proj = TorchLoRALinear(gate, rank=lora_rank, alpha=lora_alpha, device=device, lora_dtype=self.lora_dtype, init_lora_weights="peft", lora_dropout=lora_dropout)
            self.up_proj = TorchLoRALinear(up, rank=lora_rank, alpha=lora_alpha, device=device, lora_dtype=self.lora_dtype, init_lora_weights="peft", lora_dropout=lora_dropout)
            self.down_proj = TorchLoRALinear(down, rank=lora_rank, alpha=lora_alpha, device=device, lora_dtype=self.lora_dtype, init_lora_weights="peft", lora_dropout=lora_dropout)
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            module.scaling = self.lora_scale
        _copy_preexisting_lora(self.gate_proj, gate_spec, strict=strict)
        _copy_preexisting_lora(self.up_proj, up_spec, strict=strict)
        _copy_preexisting_lora(self.down_proj, down_spec, strict=strict)

    @property
    def cpu_resident_base_bytes(self) -> int:
        return sum(int(getattr(module, "cpu_resident_base_weight_bytes", 0)) for module in (self.gate_proj, self.up_proj, self.down_proj))

    @property
    def gpu_resident_base_bytes(self) -> int:
        return sum(int(getattr(module, "gpu_resident_base_weight_bytes", 0)) for module in (self.gate_proj, self.up_proj, self.down_proj))

    @property
    def trainable_lora_params(self) -> int:
        return sum(param.numel() for name, param in self.named_parameters() if "lora_" in name or ".lora_A." in name or ".lora_B." in name)

    def _lora_weight_banks(self) -> list[tuple[str, torch.nn.Parameter]]:
        if not (
            isinstance(self.gate_proj, AsymLoRALinear)
            and isinstance(self.up_proj, AsymLoRALinear)
            and isinstance(self.down_proj, AsymLoRALinear)
        ):
            return []
        return [
            ("gate_lora_A", self.gate_proj.lora_a),
            ("gate_lora_B", self.gate_proj.lora_b),
            ("up_lora_A", self.up_proj.lora_a),
            ("up_lora_B", self.up_proj.lora_b),
            ("down_lora_A", self.down_proj.lora_a),
            ("down_lora_B", self.down_proj.lora_b),
        ]

    def gather_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.gather_group(self)

    def release_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.release_group(self)

    def _profile_name(self, *parts: object) -> str:
        return scoped_name(self.profile_prefix, *parts)

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def _backward_range(self, *parts: object) -> str:
        return scoped_name("backward", self.profile_prefix, *parts)

    def _activation_offload_supported(self, x: torch.Tensor) -> bool:
        return (
            self.backend == "asym"
            and self.offload
            and self.training
            and torch.is_grad_enabled()
            and _shared_mlp_activation_offload_enabled()
            and self.lora_dropout_p == 0.0
            and self.lora_dtype == torch.bfloat16
            and x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and _is_silu_activation(self.activation_fn)
            and isinstance(self.gate_proj, AsymLoRALinear)
            and isinstance(self.up_proj, AsymLoRALinear)
            and isinstance(self.down_proj, AsymLoRALinear)
            and self.gate_proj.base_layer.precision == "bf16"
            and self.up_proj.base_layer.precision == "bf16"
            and self.down_proj.base_layer.precision == "bf16"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation_offload_supported(x):
            return _Llama4SharedMLPActivationOffloadFunction.apply(
                x,
                self.gate_proj.lora_a,
                self.gate_proj.lora_b,
                self.up_proj.lora_a,
                self.up_proj.lora_b,
                self.down_proj.lora_a,
                self.down_proj.lora_b,
                self,
            )
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation_fn(gate) * up)


def is_llama4_shared_mlp(module: nn.Module) -> bool:
    return isinstance(module, AsymLlama4SharedMLP)


__all__ = ["AsymLlama4SharedMLP", "is_llama4_shared_mlp", "is_llama4_shared_mlp_leaf"]
