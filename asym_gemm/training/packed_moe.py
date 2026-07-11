from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from .frozen_linear import AsymExecutionStats
from .qwen3_moe import AsymQwen3Experts as AsymPackedExperts


@dataclass(frozen=True)
class PackedMoELayout:
    family: str
    gate_up_layout: str
    down_layout: str
    hidden_size: int
    expert_size: int
    num_experts: int
    top_k: int


class PackedExpertSource(nn.Module):
    """Small normalized source object for packed expert wrappers.

    `AsymPackedExperts` can consume either Qwen3-style `[experts, out, in]`
    packed weights or model-native `[experts, in, out]` weights when the
    layout declares it. The grouped GEMM wrapper then selects the correct
    transpose mode without materializing a full layout-conversion copy.
    """

    def __init__(
        self,
        *,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        layout: PackedMoELayout,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        config: object | None = None,
    ) -> None:
        super().__init__()
        if layout.gate_up_layout == "experts,2*expert,hidden":
            expected_gate_up = (layout.num_experts, 2 * layout.expert_size, layout.hidden_size)
            self.gate_up_weight_layout = "out_in"
        elif layout.gate_up_layout == "experts,hidden,2*expert":
            expected_gate_up = (layout.num_experts, layout.hidden_size, 2 * layout.expert_size)
            self.gate_up_weight_layout = "in_out"
        else:
            raise ValueError(f"unsupported gate_up layout {layout.gate_up_layout!r}")
        if layout.down_layout == "experts,hidden,expert":
            expected_down = (layout.num_experts, layout.hidden_size, layout.expert_size)
            self.down_weight_layout = "out_in"
        elif layout.down_layout == "experts,expert,hidden":
            expected_down = (layout.num_experts, layout.expert_size, layout.hidden_size)
            self.down_weight_layout = "in_out"
        else:
            raise ValueError(f"unsupported down layout {layout.down_layout!r}")
        if tuple(gate_up_proj.shape) != expected_gate_up:
            raise ValueError(f"normalized gate_up shape {tuple(gate_up_proj.shape)} != {expected_gate_up}")
        if tuple(down_proj.shape) != expected_down:
            raise ValueError(f"normalized down shape {tuple(down_proj.shape)} != {expected_down}")
        self.config = config
        self.num_experts = int(layout.num_experts)
        self.hidden_dim = int(layout.hidden_size)
        self.intermediate_dim = int(layout.expert_size)
        self.act_fn = act_fn
        self.gate_up_proj = nn.Parameter(gate_up_proj.detach().contiguous(), requires_grad=False)
        self.down_proj = nn.Parameter(down_proj.detach().contiguous(), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("PackedExpertSource is only used to normalize packed weights")


def wrap_packed_experts(
    source: PackedExpertSource,
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
) -> AsymPackedExperts:
    return AsymPackedExperts(
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
        strict=False,
    )


__all__ = [
    "AsymPackedExperts",
    "PackedExpertSource",
    "PackedMoELayout",
    "wrap_packed_experts",
]
