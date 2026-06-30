"""Fine-grained activation offload for dense Qwen-style MLPs.

This path is intentionally separate from ``dense_mlp.py``.  The older dense path
adapts the MLP to the one-expert Qwen3 expert engine, which fuses gate/up and can
run the full dense activation backward on CPU.  This module keeps gate, up, and
down as separate dense projections so the backward can stage one logical operand
at a time and never call ``stage_concat_columns``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .exp_act_offload_lora import (
    grouped_lora_a_forward_cpu_left,
    grouped_lora_a_grad_cpu_right,
    require_expert_activation_offload_kernels,
)
from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, asym_bf16_cpu_right_matmul
from .lora import AsymLoRALinear, normalize_lora_dtype
from .offload import adopt_host_weight
from .profile_ranges import prof_range, scoped_name


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _finegrained_enabled() -> bool:
    return _env_enabled("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD"
    )


def _is_silu_activation(fn: Any) -> bool:
    if fn is F.silu or isinstance(fn, torch.nn.SiLU):
        return True
    name = getattr(fn, "__name__", "") or type(fn).__name__
    return name in {"silu", "silu_python", "SiLU", "SiLUActivation"}


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


def _lora_b_forward(low_rank: torch.Tensor, b: torch.Tensor, *, scale: float) -> torch.Tensor:
    out = low_rank @ b.t()
    return out.mul(float(scale)) if float(scale) != 1.0 else out


def _lora_b_grad(grad_output: torch.Tensor, low_rank: torch.Tensor, *, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    grad = grad_output.t().contiguous() @ low_rank
    if float(scale) != 1.0:
        grad = grad.mul(float(scale))
    return grad.to(dtype=out_dtype)


def _lora_ds(grad_output: torch.Tensor, b: torch.Tensor, *, scale: float) -> torch.Tensor:
    grad_lora = grad_output if grad_output.dtype == b.dtype else grad_output.to(dtype=b.dtype)
    out = grad_lora @ b
    return out.mul(float(scale)) if float(scale) != 1.0 else out


def _dense_lora_a(lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
    if lora_a.dim() == 2:
        return lora_a
    if lora_a.dim() == 3 and int(lora_a.shape[0]) == 1:
        return lora_a.squeeze(0)
    raise RuntimeError(f"{tag}: fine-grained dense MLP expected LoRA-A [r,K], got {tuple(lora_a.shape)}")


def _grouped_lora_a(lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
    dense = _dense_lora_a(lora_a, tag=tag)
    return dense.unsqueeze(0).contiguous()


@dataclass(frozen=True)
class _OneExpertPlan:
    offsets: torch.Tensor
    experts: torch.Tensor


class _FinegrainedDenseMLPFunction(torch.autograd.Function):
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
        layer: "AsymFinegrainedDenseMLP",
    ) -> torch.Tensor:
        if x.dim() < 1:
            raise ValueError("fine-grained dense MLP expects a hidden dimension")
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
            raise ValueError(f"fine-grained dense MLP requires bf16 input, got {flat.dtype}")

        manager = ActivationOffloadManager(pin_memory=True)
        layer.stats.dense_mlp_finegrained_forward_calls += 1

        with prof_range(layer._forward_range("finegrained", "x_to_cpu")):
            x_cpu = manager.offload(flat, "mlp.X")

        with prof_range(layer._forward_range("finegrained", "gate")):
            layer.stats.dense_mlp_finegrained_gate_base_calls += 1
            gate = _asym_base_forward(
                layer.gate_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("gate", "base_forward"),
            )
            manager.wait_cpu_ready(x_cpu)
            gate_low_rank = layer._cpu_left_lora_a(x_cpu, gate_a, tag="gate")
            gate_delta = _lora_b_forward(gate_low_rank, gate_b, scale=layer.lora_scale)
            gate.add_(gate_delta.to(dtype=gate.dtype))
            gate_cpu = manager.offload(gate, "mlp.gate")
            gate_low_rank_cpu = manager.offload(gate_low_rank, "mlp.S_gate")
            del gate, gate_delta, gate_low_rank

        with prof_range(layer._forward_range("finegrained", "up")):
            layer.stats.dense_mlp_finegrained_up_base_calls += 1
            up = _asym_base_forward(
                layer.up_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("up", "base_forward"),
            )
            manager.wait_cpu_ready(x_cpu)
            up_low_rank = layer._cpu_left_lora_a(x_cpu, up_a, tag="up")
            up_delta = _lora_b_forward(up_low_rank, up_b, scale=layer.lora_scale)
            up.add_(up_delta.to(dtype=up.dtype))
            up_cpu = manager.offload(up, "mlp.up")
            up_low_rank_cpu = manager.offload(up_low_rank, "mlp.S_up")
            del up, up_delta, up_low_rank

        with prof_range(layer._forward_range("finegrained", "activation")):
            gate_stage = manager.stage(gate_cpu, tag="mlp.gate_for_act")
            up_stage = manager.stage(up_cpu, tag="mlp.up_for_act")
            act = layer.activation_fn(gate_stage) * up_stage
            act_cpu = manager.offload(act.to(dtype=torch.bfloat16).contiguous(), "mlp.act")
            manager.release_stage(gate_stage, drop_cache=True)
            manager.release_stage(up_stage, drop_cache=True)
            del gate_stage, up_stage, act

        with prof_range(layer._forward_range("finegrained", "down_lora")):
            manager.wait_cpu_ready(act_cpu)
            down_low_rank = layer._cpu_left_lora_a(act_cpu, down_a, tag="down")
            down_delta = _lora_b_forward(down_low_rank, down_b, scale=layer.lora_scale)
            down_low_rank_cpu = manager.offload(down_low_rank, "mlp.S_down")
            del down_low_rank

        with prof_range(layer._forward_range("finegrained", "down_base")):
            act_stage = manager.stage(act_cpu, tag="mlp.act_for_down_base")
            layer.stats.dense_mlp_finegrained_down_base_calls += 1
            out = _asym_base_forward(
                layer.down_proj.base_layer,
                act_stage,
                stats=layer.stats,
                tag=layer._profile_name("down", "base_forward"),
            )
            manager.release_stage(act_stage, drop_cache=True)
            out.add_(down_delta.to(dtype=out.dtype))
            del act_stage, down_delta

        if weight_offload:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(gate_a, gate_b, up_a, up_b, down_a, down_b)
        ctx.weight_offload = bool(weight_offload)
        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_cpu = act_cpu
        ctx.gate_low_rank_cpu = gate_low_rank_cpu
        ctx.up_low_rank_cpu = up_low_rank_cpu
        ctx.down_low_rank_cpu = down_low_rank_cpu
        ctx.input_shape = tuple(int(dim) for dim in x.shape)
        ctx.input_dtype = x.dtype
        layer._last_activation_offload_stats = manager.snapshot()
        if weight_offload and bool(getattr(layer, "_weight_offload_release_after_forward", True)):
            layer.release_lora_weights()
        return out.reshape(*ctx.input_shape[:-1], layer.hidden_size)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        layer: AsymFinegrainedDenseMLP = ctx.layer
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
        layer.stats.dense_mlp_finegrained_backward_calls += 1
        grad_x = None
        grad_gate_a = grad_gate_b = grad_up_a = grad_up_b = grad_down_a = grad_down_b = None
        grad_gate_cpu = grad_up_cpu = None

        try:
            grad_2d = grad_output.reshape(-1, layer.hidden_size).to(dtype=torch.bfloat16).contiguous()

            with prof_range(layer._backward_range("finegrained", "down_lora")):
                down_low_rank = manager.stage(ctx.down_low_rank_cpu, tag="mlp.S_down_for_dB")
                dS_down = _lora_ds(grad_2d, down_b, scale=layer.lora_scale)
                grad_down_b = _lora_b_grad(grad_2d, down_low_rank, scale=layer.lora_scale, out_dtype=down_b.dtype)
                manager.release_stage(down_low_rank, drop_cache=True)
                manager.release_cpu(ctx.down_low_rank_cpu)
                manager.wait_cpu_ready(ctx.act_cpu)
                grad_down_a = layer._cpu_right_lora_a_grad(dS_down, ctx.act_cpu, down_a, tag="down")

            with prof_range(layer._backward_range("finegrained", "down_base_dx")):
                layer.stats.dense_mlp_finegrained_down_base_calls += 1
                grad_act = _asym_base_dx(
                    layer.down_proj.base_layer,
                    grad_2d,
                    stats=layer.stats,
                    tag=layer._profile_name("down", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                down_lora_dx = dS_down @ down_a
                grad_act.add_(down_lora_dx.to(dtype=grad_act.dtype))
                del down_lora_dx, dS_down

            with prof_range(layer._backward_range("finegrained", "activation_bwd_gpu")):
                layer.stats.dense_mlp_finegrained_gpu_silu_bwd_calls += 1
                gate_stage = manager.stage(ctx.gate_cpu, tag="mlp.gate_for_silu_bwd")
                up_stage = manager.stage(ctx.up_cpu, tag="mlp.up_for_silu_bwd")
                grad_up = layer.activation_fn(gate_stage)
                grad_up.mul_(grad_act)
                grad_up_cpu = manager.offload(grad_up.to(dtype=torch.bfloat16).contiguous(), "mlp.dup")
                del grad_up
                grad_act.mul_(up_stage)
                grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                del grad_act
                grad_gate_cpu = manager.offload(grad_gate.to(dtype=torch.bfloat16).contiguous(), "mlp.dgate")
                manager.release_stage(gate_stage, drop_cache=True)
                manager.release_stage(up_stage, drop_cache=True)
                manager.release_cpu(ctx.gate_cpu)
                manager.release_cpu(ctx.up_cpu)
                del gate_stage, up_stage, grad_gate

            with prof_range(layer._backward_range("finegrained", "gate")):
                grad_gate_stage = manager.stage(grad_gate_cpu, tag="mlp.dgate")
                gate_low_rank = manager.stage(ctx.gate_low_rank_cpu, tag="mlp.S_gate_for_dB")
                dS_gate = _lora_ds(grad_gate_stage, gate_b, scale=layer.lora_scale)
                grad_gate_b = _lora_b_grad(
                    grad_gate_stage,
                    gate_low_rank,
                    scale=layer.lora_scale,
                    out_dtype=gate_b.dtype,
                )
                manager.release_stage(gate_low_rank, drop_cache=True)
                manager.release_cpu(ctx.gate_low_rank_cpu)
                manager.wait_cpu_ready(ctx.x_cpu)
                grad_gate_a = layer._cpu_right_lora_a_grad(dS_gate, ctx.x_cpu, gate_a, tag="gate")
                layer.stats.dense_mlp_finegrained_gate_base_calls += 1
                grad_x = _asym_base_dx(
                    layer.gate_proj.base_layer,
                    grad_gate_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("gate", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                gate_lora_dx = dS_gate @ gate_a
                grad_x.add_(gate_lora_dx.to(dtype=grad_x.dtype))
                del gate_lora_dx, dS_gate
                manager.release_stage(grad_gate_stage, drop_cache=True)
                manager.release_cpu(grad_gate_cpu)
                grad_gate_cpu = None

            with prof_range(layer._backward_range("finegrained", "up")):
                grad_up_stage = manager.stage(grad_up_cpu, tag="mlp.dup")
                up_low_rank = manager.stage(ctx.up_low_rank_cpu, tag="mlp.S_up_for_dB")
                dS_up = _lora_ds(grad_up_stage, up_b, scale=layer.lora_scale)
                grad_up_b = _lora_b_grad(grad_up_stage, up_low_rank, scale=layer.lora_scale, out_dtype=up_b.dtype)
                manager.release_stage(up_low_rank, drop_cache=True)
                manager.release_cpu(ctx.up_low_rank_cpu)
                manager.wait_cpu_ready(ctx.x_cpu)
                grad_up_a = layer._cpu_right_lora_a_grad(dS_up, ctx.x_cpu, up_a, tag="up")
                layer.stats.dense_mlp_finegrained_up_base_calls += 1
                up_dx = _asym_base_dx(
                    layer.up_proj.base_layer,
                    grad_up_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("up", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                grad_x.add_(up_dx.to(dtype=grad_x.dtype))
                del up_dx
                up_lora_dx = dS_up @ up_a
                grad_x.add_(up_lora_dx.to(dtype=grad_x.dtype))
                del up_lora_dx, dS_up
                manager.release_stage(grad_up_stage, drop_cache=True)
                manager.release_cpu(grad_up_cpu)
                grad_up_cpu = None

            grad_x = grad_x.to(dtype=ctx.input_dtype).reshape(ctx.input_shape)
        finally:
            for handle in (
                ctx.x_cpu,
                ctx.gate_cpu,
                ctx.up_cpu,
                ctx.act_cpu,
                ctx.gate_low_rank_cpu,
                ctx.up_low_rank_cpu,
                ctx.down_low_rank_cpu,
                grad_gate_cpu,
                grad_up_cpu,
            ):
                manager.release_cpu(handle)
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


class AsymFinegrainedDenseMLP(nn.Module):
    """Dense MLP replacement with separate gate/up/down fine-grained scheduling."""

    _is_asym_finegrained_dense_mlp = True

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
        profile_prefix: str = "layers.unknown.mlp",
    ) -> None:
        super().__init__()
        gate = getattr(source, "gate_proj", None)
        up = getattr(source, "up_proj", None)
        down = getattr(source, "down_proj", None)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            raise TypeError("fine-grained dense MLP expects nn.Linear gate_proj/up_proj/down_proj leaves")
        if float(lora_dropout) != 0.0:
            raise NotImplementedError("fine-grained dense MLP currently requires lora_dropout=0.0")
        if backend == "asym":
            require_expert_activation_offload_kernels(scope="full")
        if strict:
            for leaf_name, module in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                if module.weight.device.type != "cpu":
                    raise RuntimeError(f"{leaf_name} fine-grained dense MLP CPU offload requires CPU-first weights")
                if module.weight.dtype != torch.bfloat16:
                    raise RuntimeError(f"{leaf_name} fine-grained dense MLP requires bf16 source weights")

        self.backend = backend
        self.precision = precision
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.activation_fn = getattr(source, "act_fn", F.silu)
        self.hidden_size = int(gate.weight.shape[1])
        self.intermediate_size = int(gate.weight.shape[0])
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(torch.bfloat16)
        self.lora_dropout_p = float(lora_dropout)
        self.profile_prefix = profile_prefix
        self._last_activation_offload_stats: dict[str, Any] = {}
        self._weight_offload = None
        self._weight_offload_release_after_forward = True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.gate_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.gate_proj.weight", gate.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if gate.bias is None else gate.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )
        self.up_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.up_proj.weight", up.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if up.bias is None else up.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )
        self.down_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.down_proj.weight", down.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if down.bias is None else down.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )

    @property
    def cpu_resident_base_bytes(self) -> int:
        return sum(
            int(getattr(module, "cpu_resident_base_weight_bytes", 0))
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        )

    @property
    def gpu_resident_base_bytes(self) -> int:
        return sum(
            int(getattr(module, "gpu_resident_base_weight_bytes", 0))
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        )

    @property
    def trainable_lora_params(self) -> int:
        return sum(param.numel() for name, param in self.named_parameters() if "lora_" in name or ".lora_A." in name or ".lora_B." in name)

    def _one_expert_plan(self, rows: int, device: torch.device) -> _OneExpertPlan:
        offsets = torch.tensor([0, int(rows)], device=device, dtype=torch.long)
        experts = torch.tensor([0, -1], device=device, dtype=torch.long)
        return _OneExpertPlan(offsets=offsets, experts=experts)

    def _cpu_left_lora_a(self, source: CPUActivationHandle, lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
        lora_a = _dense_lora_a(lora_a, tag=tag)
        if self.backend == "torch":
            return source.tensor.to(device=lora_a.device, dtype=lora_a.dtype, non_blocking=source.tensor.is_pinned()).matmul(lora_a.t())
        plan = self._one_expert_plan(int(source.tensor.shape[0]), lora_a.device)
        return grouped_lora_a_forward_cpu_left(
            source.tensor,
            _grouped_lora_a(lora_a, tag=tag),
            plan.offsets,
            plan.experts,
            metadata=None,
            stats=self.stats,
            tag=tag,
        )

    def _cpu_right_lora_a_grad(
        self,
        dS: torch.Tensor,
        source: CPUActivationHandle,
        lora_a: torch.Tensor,
        *,
        tag: str,
    ) -> torch.Tensor:
        lora_a = _dense_lora_a(lora_a, tag=tag)
        if self.backend == "torch":
            return dS.t().contiguous().matmul(
                source.tensor.to(device=dS.device, dtype=dS.dtype, non_blocking=source.tensor.is_pinned())
            ).to(dtype=lora_a.dtype)
        plan = self._one_expert_plan(int(source.tensor.shape[0]), dS.device)
        grad = grouped_lora_a_grad_cpu_right(
            dS.contiguous(),
            source.tensor,
            plan.offsets,
            plan.experts,
            num_experts=1,
            stats=self.stats,
            tag=tag,
        )
        return grad.squeeze(0)

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
            self.backend in {"asym", "torch"}
            and self.training
            and torch.is_grad_enabled()
            and _finegrained_enabled()
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
            return _FinegrainedDenseMLPFunction.apply(
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


def build_finegrained_dense_mlp(
    mlp: nn.Module,
    *,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
    profile_prefix: str = "layers.unknown.mlp",
) -> AsymFinegrainedDenseMLP:
    return AsymFinegrainedDenseMLP(
        mlp,
        backend=backend,
        precision=precision,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        stats=stats,
        strict=strict,
        profile_prefix=profile_prefix,
    )


__all__ = ["AsymFinegrainedDenseMLP", "build_finegrained_dense_mlp"]
