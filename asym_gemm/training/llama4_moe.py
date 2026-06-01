from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .frozen_linear import AsymExecutionStats
from .packed_moe import AsymPackedExperts, PackedExpertSource, PackedMoELayout, wrap_packed_experts
from .profile_ranges import prof_range, scoped_name


@dataclass(frozen=True)
class Llama4MoeReport:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cpu_resident_base_bytes: int
    gpu_resident_base_bytes: int
    trainable_lora_params: int
    expert_recompute_policy: str = "none"


def _is_3d_parameter(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, nn.Parameter) and value.dim() == 3


def is_llama4_moe(module: nn.Module) -> bool:
    experts = getattr(module, "experts", None)
    router = getattr(module, "router", None)
    shared_expert = getattr(module, "shared_expert", None)
    if not isinstance(experts, nn.Module) or not isinstance(router, nn.Module) or not isinstance(shared_expert, nn.Module):
        return False
    if not (_is_3d_parameter(experts, "gate_up_proj") and _is_3d_parameter(experts, "down_proj")):
        return False
    for attr in ("num_experts", "hidden_dim", "top_k"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    for attr in ("hidden_size", "intermediate_size", "expert_dim", "num_experts"):
        if not isinstance(getattr(experts, attr, None), int):
            return False
    if not callable(getattr(experts, "act_fn", None)):
        return False

    gate_up = getattr(experts, "gate_up_proj")
    down = getattr(experts, "down_proj")
    num_experts = int(getattr(experts, "num_experts"))
    hidden_size = int(getattr(experts, "hidden_size"))
    expert_dim = int(getattr(experts, "expert_dim"))
    return tuple(gate_up.shape) == (num_experts, hidden_size, 2 * expert_dim) and tuple(down.shape) == (
        num_experts,
        expert_dim,
        hidden_size,
    )


class AsymLlama4Moe(nn.Module):
    """Llama 4 MoE wrapper using the shared packed expert LoRA path."""

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
        expert_recompute_policy: str = "none",
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if strict and not is_llama4_moe(source):
            raise TypeError(f"source does not look like a Llama 4 MoE module: {type(source).__name__}")
        if backend not in {"asym", "torch"}:
            raise ValueError("backend must be 'asym' or 'torch'")
        if precision != "bf16":
            raise ValueError("AsymLlama4Moe first pass supports bf16 only")

        source_experts = getattr(source, "experts")
        gate_up = getattr(source_experts, "gate_up_proj").detach()
        down = getattr(source_experts, "down_proj").detach()
        self.config = getattr(source, "config", None)
        self.top_k = int(getattr(source, "top_k"))
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.num_experts = int(getattr(source, "num_experts"))
        self.router = getattr(source, "router")
        self.shared_expert = getattr(source, "shared_expert")
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.profile_prefix = "layers.unknown.feed_forward"

        layout = PackedMoELayout(
            family="llama4",
            gate_up_layout="experts,hidden,2*expert",
            down_layout="experts,expert,hidden",
            hidden_size=int(getattr(source_experts, "hidden_size")),
            expert_size=int(getattr(source_experts, "expert_dim")),
            num_experts=int(getattr(source_experts, "num_experts")),
            top_k=self.top_k,
        )
        if layout.hidden_size != self.hidden_dim or layout.num_experts != self.num_experts:
            raise ValueError(
                "Llama 4 MoE/expert metadata mismatch: "
                f"moe hidden/experts=({self.hidden_dim}, {self.num_experts}) "
                f"experts hidden/experts=({layout.hidden_size}, {layout.num_experts})"
            )
        packed_source = PackedExpertSource(
            gate_up_proj=gate_up.transpose(-1, -2).contiguous(),
            down_proj=down.transpose(-1, -2).contiguous(),
            layout=layout,
            act_fn=getattr(source_experts, "act_fn"),
            config=getattr(source_experts, "config", self.config),
        )
        self.experts: AsymPackedExperts = wrap_packed_experts(
            packed_source,
            backend=backend,
            precision=precision,
            offload=offload,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_dtype=lora_dtype,
            expert_recompute_policy=expert_recompute_policy,
            stats=stats,
        )

    @property
    def cpu_resident_base_bytes(self) -> int:
        return int(self.experts.cpu_resident_base_bytes)

    @property
    def gpu_resident_base_bytes(self) -> int:
        return int(self.experts.gpu_resident_base_bytes)

    @property
    def trainable_lora_params(self) -> int:
        return int(self.experts.trainable_lora_params)

    def report(self) -> Llama4MoeReport:
        return Llama4MoeReport(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.experts.intermediate_dim,
            cpu_resident_base_bytes=self.cpu_resident_base_bytes,
            gpu_resident_base_bytes=self.gpu_resident_base_bytes,
            trainable_lora_params=self.trainable_lora_params,
            expert_recompute_policy=self.experts.expert_recompute_config.label,
        )

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        with prof_range(self._forward_range("router")):
            router_scores, router_logits = self.router(flat)
            _top_values, top_k_index = torch.topk(router_logits, self.top_k, dim=1)
            input_weights = router_scores.gather(1, top_k_index)
        with prof_range(self._forward_range("shared_expert")):
            out = self.shared_expert(flat)
        with prof_range(self._forward_range("experts")):
            routed = self.experts.forward_input_scaled(flat, top_k_index, input_weights)
        return out + routed.to(dtype=out.dtype), router_logits


def wrap_llama4_moe(
    source: nn.Module,
    *,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_dtype: torch.dtype | str | None = torch.bfloat16,
    expert_recompute_policy: str = "none",
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymLlama4Moe:
    return AsymLlama4Moe(
        source,
        backend=backend,
        precision=precision,
        offload=offload,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_dtype=lora_dtype,
        expert_recompute_policy=expert_recompute_policy,
        stats=stats,
        strict=strict,
    )


__all__ = ["AsymLlama4Moe", "Llama4MoeReport", "is_llama4_moe", "wrap_llama4_moe"]
