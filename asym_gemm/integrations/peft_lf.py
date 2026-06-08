from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from torch import nn

from .lf import LFAsymReport, apply_lf_asym_lora, count_lora_wrapped_modules


def adapt_lf_asym_peft_lora(
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
    router_mode: Literal["hf", "whole"] = "whole",
    strict: bool = True,
) -> tuple[nn.Module, LFAsymReport]:
    """Adapt packed experts after PEFT has attached ordinary dense LoRA modules."""

    return apply_lf_asym_lora(
        model,
        raw_lora_target=raw_lora_target,
        dense_target_modules=dense_target_modules,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        backend=backend,
        precision=precision,
        offload_modules=offload_modules,
        expert_recompute_policy=expert_recompute_policy,
        router_mode=router_mode,
        wrap_dense=False,
        preexisting_dense_lora_wrapped=count_lora_wrapped_modules(model),
        strict=strict,
    )


__all__ = ["adapt_lf_asym_peft_lora"]
