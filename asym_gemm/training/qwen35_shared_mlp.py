from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .frozen_linear import AsymExecutionStats
from .llama4_shared_mlp import AsymLlama4SharedMLP, is_llama4_shared_mlp_leaf


class AsymQwen35SharedMLP(AsymLlama4SharedMLP):
    """Qwen3.5 shared expert MLP using the proven Llama4 shared-MLP execution path."""

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
        activation_fn = getattr(source, "activation_fn", None)
        added_activation_alias = False
        if not callable(activation_fn):
            activation_fn = getattr(source, "act_fn", None)
            if callable(activation_fn):
                setattr(source, "activation_fn", activation_fn)
                added_activation_alias = True
        try:
            super().__init__(
                source,
                backend=backend,
                precision=precision,
                offload=offload,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_dtype=lora_dtype,
                stats=stats,
                strict=strict,
            )
        finally:
            if added_activation_alias:
                try:
                    delattr(source, "activation_fn")
                except AttributeError:
                    pass
        self.profile_prefix = "layers.unknown.mlp.shared_expert"


def is_qwen35_shared_mlp(module: nn.Module) -> bool:
    return isinstance(module, AsymQwen35SharedMLP)


def is_qwen35_shared_mlp_leaf(module: nn.Module) -> bool:
    return is_llama4_shared_mlp_leaf(module)


__all__ = ["AsymQwen35SharedMLP", "is_qwen35_shared_mlp", "is_qwen35_shared_mlp_leaf"]
