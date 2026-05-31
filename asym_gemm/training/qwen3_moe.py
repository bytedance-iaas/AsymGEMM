from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Literal

import torch
from torch import nn

from .frozen_linear import AsymExecutionStats, AsymGroupedFrozenLinear, TorchGroupedFrozenLinear
from .lora import grouped_expert_lora, grouped_expert_lora_pair, normalize_lora_dtype, prepare_grouped_lora_metadata
from .moe import build_contiguous_route_metadata, make_dense_group_metadata, pack_tokens_contiguous, scatter_contiguous


@dataclass(frozen=True)
class Qwen3ExpertReport:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cpu_resident_base_bytes: int
    gpu_resident_base_bytes: int
    trainable_lora_params: int


def _is_3d_parameter(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, nn.Parameter) and value.dim() == 3


def is_qwen3_experts(module: nn.Module) -> bool:
    if not (_is_3d_parameter(module, "gate_up_proj") and _is_3d_parameter(module, "down_proj")):
        return False
    for attr in ("num_experts", "hidden_dim", "intermediate_dim"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    if not callable(getattr(module, "act_fn", None)):
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in ("hidden_states", "top_k_index", "top_k_weights"))


def _resolve_device(source: torch.Tensor) -> torch.device:
    return torch.device(source.device)


def _reset_lora_bank(weight: torch.Tensor, *, is_b: bool) -> None:
    if is_b:
        nn.init.zeros_(weight)
    else:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class AsymQwen3Experts(nn.Module):
    """Qwen3 packed MoE experts with frozen grouped bases and trainable LoRA."""

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
        init_lora_weights: Literal["peft"] = "peft",
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if strict and not is_qwen3_experts(source):
            raise TypeError(f"source does not look like a Qwen3 packed expert module: {type(source).__name__}")
        if backend not in {"asym", "torch"}:
            raise ValueError("backend must be 'asym' or 'torch'")
        if precision != "bf16":
            raise ValueError("AsymQwen3Experts first pass supports bf16 only")
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")
        if init_lora_weights != "peft":
            raise ValueError("AsymQwen3Experts first pass only supports PEFT-compatible LoRA init")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")

        gate_up = getattr(source, "gate_up_proj").detach()
        down = getattr(source, "down_proj").detach()
        self.num_experts = int(getattr(source, "num_experts"))
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.intermediate_dim = int(getattr(source, "intermediate_dim"))
        expected_gate_up = (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        expected_down = (self.num_experts, self.hidden_dim, self.intermediate_dim)
        if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
            raise ValueError(
                "unexpected Qwen3 expert shapes: "
                f"gate_up={tuple(gate_up.shape)} expected={expected_gate_up}, "
                f"down={tuple(down.shape)} expected={expected_down}"
            )

        self.config = getattr(source, "config", None)
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.act_fn = getattr(source, "act_fn")
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        self.stats = stats if stats is not None else AsymExecutionStats()

        base_dtype = torch.bfloat16
        if backend == "asym" and self.offload:
            self.gate_up_base = AsymGroupedFrozenLinear(
                gate_up.to(dtype=base_dtype),
                backend="asym",
                precision=precision,
                stats=self.stats,
            )
            self.down_base = AsymGroupedFrozenLinear(
                down.to(dtype=base_dtype),
                backend="asym",
                precision=precision,
                stats=self.stats,
            )
        else:
            device = _resolve_device(gate_up)
            self.gate_up_base = TorchGroupedFrozenLinear(gate_up, device=device, dtype=base_dtype)
            self.down_base = TorchGroupedFrozenLinear(down, device=device, dtype=base_dtype)

        device = _resolve_device(gate_up)
        self.gate_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.gate_lora_B = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype)
        )
        self.up_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.up_lora_B = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.down_lora_A = nn.Parameter(
            torch.empty(self.num_experts, self.lora_rank, self.intermediate_dim, device=device, dtype=self.lora_dtype)
        )
        self.down_lora_B = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        with torch.no_grad():
            for weight in (self.gate_lora_A, self.up_lora_A, self.down_lora_A):
                _reset_lora_bank(weight, is_b=False)
            for weight in (self.gate_lora_B, self.up_lora_B, self.down_lora_B):
                _reset_lora_bank(weight, is_b=True)

    @property
    def cpu_resident_base_bytes(self) -> int:
        total = 0
        for module in (self.gate_up_base, self.down_base):
            if isinstance(module, AsymGroupedFrozenLinear):
                total += int(getattr(module, "weight_hbm_saved_bytes", 0))
        return total

    @property
    def gpu_resident_base_bytes(self) -> int:
        total = 0
        for module in (self.gate_up_base, self.down_base):
            total += int(getattr(module, "gpu_resident_weight_bytes", 0))
        return total

    @property
    def trainable_lora_params(self) -> int:
        return sum(param.numel() for name, param in self.named_parameters() if "lora_" in name)

    def report(self) -> Qwen3ExpertReport:
        return Qwen3ExpertReport(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            cpu_resident_base_bytes=self.cpu_resident_base_bytes,
            gpu_resident_base_bytes=self.gpu_resident_base_bytes,
            trainable_lora_params=self.trainable_lora_params,
        )

    def _forward_gate_up_lora(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, metadata) -> tuple[torch.Tensor, torch.Tensor]:
        x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
        gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)
        low_rank = grouped_expert_lora(x_lora, gate_up_a, offsets, experts, metadata=metadata)
        gate_low_rank, up_low_rank = low_rank.split(self.lora_rank, dim=-1)
        gate_delta, up_delta = grouped_expert_lora_pair(
            gate_low_rank,
            up_low_rank,
            self.gate_lora_B,
            self.up_lora_B,
            offsets,
            experts,
            metadata=metadata,
        )
        if self.lora_scale != 1.0:
            gate_delta = gate_delta.mul(self.lora_scale)
            up_delta = up_delta.mul(self.lora_scale)
        return gate_delta, up_delta

    def _forward_down_lora(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, metadata) -> torch.Tensor:
        x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
        low_rank = grouped_expert_lora(x_lora, self.down_lora_A, offsets, experts, metadata=metadata)
        delta = grouped_expert_lora(low_rank, self.down_lora_B, offsets, experts, metadata=metadata)
        if self.lora_scale != 1.0:
            delta = delta.mul(self.lora_scale)
        return delta

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts=self.num_experts)
        packed = pack_tokens_contiguous(hidden_states, metadata)
        offsets, experts = make_dense_group_metadata(
            metadata.expert_offsets,
            num_groups=self.num_experts,
            device=packed.device,
        )
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)

        gate_up = self.gate_up_base(packed, offsets, experts, dense_experts=True)
        gate, up = gate_up.chunk(2, dim=-1)
        gate_delta, up_delta = self._forward_gate_up_lora(packed, offsets, experts, lora_metadata)
        gate = gate + gate_delta.to(dtype=gate.dtype)
        up = up + up_delta.to(dtype=up.dtype)

        activated = self.act_fn(gate) * up
        down = self.down_base(activated.to(dtype=packed.dtype), offsets, experts, dense_experts=True)
        down_delta = self._forward_down_lora(activated, offsets, experts, lora_metadata)
        down = down + down_delta.to(dtype=down.dtype)
        return scatter_contiguous(down, metadata).to(dtype=input_dtype)


def wrap_qwen3_experts(
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
) -> AsymQwen3Experts:
    return AsymQwen3Experts(
        source,
        backend=backend,
        precision=precision,
        offload=offload,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_dtype=lora_dtype,
        init_lora_weights="peft",
        stats=stats,
        strict=strict,
    )


__all__ = [
    "AsymQwen3Experts",
    "Qwen3ExpertReport",
    "is_qwen3_experts",
    "wrap_qwen3_experts",
]
