"""Jamba MoE family wrapper (AI21 Jamba2-Mini; model_integration.md family #7).

REPLICATED code path per the integration campaign rule: standalone family
layer (detector + block + wrap fn) delegating to the shared engine
`AsymQwen3Experts`, exactly like mixtral_moe.py — transformers 5.6's
`JambaExperts` carries the identical packed layout (`gate_up_proj
[E, 2I, H]`, `down_proj [E, H, I]`, act_fn, forward(hidden, top_k_index,
top_k_weights)), so the engine consumes it unchanged.

Family specifics kept here:
- detectors are NAME-GATED on `Jamba` (qwen3's structural detector would
  match Jamba blocks; the name gate makes reverse capture impossible);
- the HF block's router is a bare `nn.Linear(hidden, num_experts)` and the
  block owns `hidden_dim`/`top_k`/`num_experts` (Mixtral keeps them on the
  gate) — routing math is replicated out-of-place for numerical parity:
  softmax(logits, fp32) -> topk -> cast to activation dtype, NO
  renormalization (HF `route_tokens_to_experts`), no jitter;
- router stays frozen on GPU (router_mode="whole"; a 16xhidden Linear is
  negligible), no router offload;
- Jamba interleaves Mamba/attention decoder layers and puts the FFN under
  `.feed_forward` (dense `JambaMLP` on even layers, `JambaSparseMoeBlock`
  every `expert_layer_period`) — layer-level dispatch lives in
  integrations/lf.py; this module only wraps the sparse block.
No EP-skew/EP-stats debug instrumentation (qwen3-family diagnostics stay
qwen3-only).
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import AsymExecutionStats
from .profile_ranges import prof_range, scoped_name
from .qwen3_moe import AsymQwen3Experts


def _is_3d_parameter(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, torch.Tensor) and value.dim() == 3


def is_jamba_experts(module: nn.Module) -> bool:
    if "jamba" not in type(module).__name__.lower():
        return False
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


def is_jamba_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_jamba_moe_block", False):
        return False
    if "jamba" not in type(module).__name__.lower():
        return False
    if hasattr(module, "shared_expert") or hasattr(module, "shared_experts"):
        return False
    router = getattr(module, "router", None)
    experts = getattr(module, "experts", None)
    if not isinstance(router, nn.Linear) or not is_jamba_experts(experts):
        return False
    for attr in ("hidden_dim", "top_k", "num_experts"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    return router.out_features == int(getattr(module, "num_experts"))


class AsymJambaMoeBlock(nn.Module):
    """Jamba sparse-MoE block wrapper that owns frozen router execution."""

    _is_asym_jamba_moe_block = True

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
        router_mode: Literal["whole"] = "whole",
        router_debug_grad: bool = False,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if router_mode != "whole":
            raise ValueError(f"AsymJambaMoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_jamba_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a Jamba sparse MoE block with router/experts: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.feed_forward"

        # Preserve the installed module order: router first, experts second.
        self.router = getattr(source, "router")
        self.experts = AsymQwen3Experts(
            getattr(source, "experts"),
            backend=backend,
            precision=precision,
            offload=offload,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_dtype=lora_dtype,
            expert_recompute_policy=expert_recompute_policy,
            init_lora_weights="peft",
            stats=stats,
            strict=False,  # family detector already ran; engine still validates shapes
        )

        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.top_k = int(getattr(source, "top_k"))
        self.num_experts = int(getattr(source, "num_experts"))
        self.router.requires_grad_(False)

    @property
    def cpu_resident_base_bytes(self) -> int:
        return int(self.experts.cpu_resident_base_bytes)

    @property
    def gpu_resident_base_bytes(self) -> int:
        return int(self.experts.gpu_resident_base_bytes)

    @property
    def trainable_lora_params(self) -> int:
        return int(self.experts.trainable_lora_params)

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def _compute_routing(self, hidden_states_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # HF JambaSparseMoeBlock.route_tokens_to_experts, out-of-place:
        # softmax over ALL experts in fp32, topk, cast; NO renormalization.
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_logits = self.router(hidden_states_2d)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
            top_k_weights = top_k_weights.to(hidden_states_2d.dtype)
        if not self.router_debug_grad:
            top_k_weights = top_k_weights.detach()
            top_k_index = top_k_index.detach()
        return top_k_index, top_k_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymJambaMoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        flat = hidden_states.reshape(-1, input_shape[-1])
        top_k_index, top_k_weights = self._compute_routing(flat)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
        with prof_range(self._forward_range("experts")):
            out = self.experts(flat, top_k_index, top_k_weights)
        return out.view(input_shape)


def wrap_jamba_moe_block(
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
    router_mode: Literal["whole"] = "whole",
    router_debug_grad: bool = False,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymJambaMoeBlock:
    return AsymJambaMoeBlock(
        source,
        backend=backend,
        precision=precision,
        offload=offload,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_dtype=lora_dtype,
        expert_recompute_policy=expert_recompute_policy,
        router_mode=router_mode,
        router_debug_grad=router_debug_grad,
        stats=stats,
        strict=strict,
    )


__all__ = [
    "AsymJambaMoeBlock",
    "is_jamba_experts",
    "is_jamba_moe_block",
    "wrap_jamba_moe_block",
]
