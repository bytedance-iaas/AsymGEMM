"""Phi-3.5-MoE family wrapper (model_integration.md, family #2 of 6).

REPLICATED code path (campaign rule): standalone family-facing layer; the deep
engine is the shared `AsymQwen3Experts` (packed_moe precedent — tf-5.6
`PhimoeExperts` carries the identical packed layout `[E, 2I, H]/[E, H, I]`,
silu, and the (hidden, top_k_index, top_k_weights) forward contract).

Family deltas vs the Mixtral clone this was copied from:
- the block's router attr is `.router` (a `PhimoeTopKRouter(nn.Linear)` running
  sparsemixer-v2 with its own training-time jitter) — kept intact on GPU
  (router_mode="whole"); LF's LoRA-all targeting already excludes it by name
  ("router" in module path);
- `hidden_dim/top_k/num_experts` ints live on the BLOCK, not the gate;
- input jitter attr is `input_jitter_noise` (block-level, training-only),
  replicated out-of-place for fidelity.
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


def _resolve_phimoe_expert_act_fn(module: nn.Module):
    act_fn = getattr(module, "act_fn", None)
    if callable(act_fn):
        return act_fn
    hidden_act = str(getattr(getattr(module, "config", None), "hidden_act", "")).lower()
    if hidden_act in {"silu", "swish"}:
        return F.silu
    return None


def is_phimoe_experts(module: nn.Module) -> bool:
    if "phimoe" not in type(module).__name__.lower():
        return False
    if not (_is_3d_parameter(module, "gate_up_proj") and _is_3d_parameter(module, "down_proj")):
        return False
    for attr in ("num_experts", "hidden_dim", "intermediate_dim"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    if not callable(_resolve_phimoe_expert_act_fn(module)):
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in ("hidden_states", "top_k_index", "top_k_weights"))


def is_phimoe_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_phimoe_moe_block", False):
        return False
    if "phimoe" not in type(module).__name__.lower():
        return False
    if hasattr(module, "shared_expert") or hasattr(module, "shared_expert_gate") or hasattr(module, "shared_experts"):
        return False
    router = getattr(module, "router", None)
    experts = getattr(module, "experts", None)
    if not isinstance(router, nn.Module) or not is_phimoe_experts(experts):
        return False
    for attr in ("hidden_dim", "top_k", "num_experts"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    return callable(getattr(router, "forward", None))


class AsymPhimoeMoeBlock(nn.Module):
    """Phi-MoE sparse block wrapper that owns frozen router execution."""

    _is_asym_phimoe_moe_block = True

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
            raise ValueError(f"AsymPhimoeMoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_phimoe_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a Phi-MoE sparse block with router/expert routing: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"
        # HF applies multiplicative input jitter in training mode (Phi-3.5-MoE
        # configures 0.01); replicated out-of-place for parity.
        self.input_jitter_noise = float(getattr(source, "input_jitter_noise", 0.0) or 0.0)

        # Preserve the installed Phi-MoE module order: router first, experts second.
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

    def _compute_routing(self, hidden_states_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_out = self.router(hidden_states_2d)

        if isinstance(router_out, tuple) and len(router_out) >= 3:
            top_k_weights = router_out[1]
            top_k_index = router_out[2]
            if not self.router_debug_grad:
                top_k_weights = top_k_weights.detach()
                top_k_index = top_k_index.detach()
            if top_k_weights.dtype != hidden_states_2d.dtype:
                top_k_weights = top_k_weights.to(dtype=hidden_states_2d.dtype)
            return top_k_index, top_k_weights, None

        raise TypeError(
            "AsymPhimoeMoeBlock requires a PhimoeTopKRouter-style router returning "
            "(router_logits, top_k_weights, top_k_index); "
            f"got {type(router_out).__name__}"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymPhimoeMoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        if self.training and self.input_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        flat = hidden_states.reshape(-1, input_shape[-1])
        top_k_index, top_k_weights, _router_logits = self._compute_routing(flat)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
        with prof_range(self._forward_range("experts")):
            out = self.experts(flat, top_k_index, top_k_weights)
        return out.view(input_shape)


def wrap_phimoe_moe_block(
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
) -> AsymPhimoeMoeBlock:
    return AsymPhimoeMoeBlock(
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
    "AsymPhimoeMoeBlock",
    "is_phimoe_experts",
    "is_phimoe_moe_block",
    "wrap_phimoe_moe_block",
]
