"""Hunyuan-A13B (hunyuan_v1_moe) family wrapper (model_integration.md, #3 of 6).

REPLICATED code path (campaign rule): standalone family-facing layer; engine =
shared `AsymQwen3Experts` (tf-5.6 `HunYuanMoEV1Experts` carries the identical
packed layout `[E, 2I, H]/[E, H, I]`, silu, standard forward contract).

Family deltas vs the Mixtral/Phimoe clones:
- the HF gate (`HunYuanMoEV1Gate`, fp32 `wg` Linear) returns raw LOGITS only;
  top-k happens in the block. This wrapper replicates the block's
  `route_tokens_to_experts` verbatim (softmax over experts in fp32, top-k,
  renormalize, cast back) under no-grad;
- the block carries a SHARED dense MLP (`shared_mlp`, plain gate/up/down
  Linears) whose output is ADDED to the routed output. It is kept as the
  original module inside this wrapper: it stays GPU-resident and receives
  standard PEFT LoRA via the normal dense targeting (`...mlp.shared_mlp.*`),
  exactly like any dense MLP — no asym offload for it in this first pass;
- `num_experts`/`top_k` ints live on the BLOCK (per-layer configurable in the
  arch; scalar on A13B), `hidden_dim` comes from the experts module.
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


def _resolve_hunyuan_expert_act_fn(module: nn.Module):
    act_fn = getattr(module, "act_fn", None)
    if callable(act_fn):
        return act_fn
    hidden_act = str(getattr(getattr(module, "config", None), "hidden_act", "")).lower()
    if hidden_act in {"silu", "swish"}:
        return F.silu
    return None


def is_hunyuan_experts(module: nn.Module) -> bool:
    if "hunyuan" not in type(module).__name__.lower():
        return False
    if not (_is_3d_parameter(module, "gate_up_proj") and _is_3d_parameter(module, "down_proj")):
        return False
    for attr in ("num_experts", "hidden_dim", "intermediate_dim"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    if not callable(_resolve_hunyuan_expert_act_fn(module)):
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in ("hidden_states", "top_k_index", "top_k_weights"))


def is_hunyuan_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_hunyuan_moe_block", False):
        return False
    if "hunyuan" not in type(module).__name__.lower():
        return False
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    shared = getattr(module, "shared_mlp", None)
    if not isinstance(gate, nn.Module) or not is_hunyuan_experts(experts):
        return False
    if not isinstance(shared, nn.Module):
        return False
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(shared, leaf, None), nn.Module):
            return False
    for attr in ("num_experts", "top_k"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    return callable(getattr(gate, "forward", None))


class AsymHunyuanMoeBlock(nn.Module):
    """Hunyuan MoE block wrapper: frozen gate + replicated top-k + shared MLP."""

    _is_asym_hunyuan_moe_block = True

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
            raise ValueError(f"AsymHunyuanMoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_hunyuan_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a Hunyuan MoE block with gate/experts/shared_mlp: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        # Preserve the installed Hunyuan module order: gate, experts, shared_mlp.
        self.gate = getattr(source, "gate")
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
        # Shared dense MLP stays the ORIGINAL module: GPU-resident, standard PEFT
        # LoRA targets (...mlp.shared_mlp.{gate,up,down}_proj), grads flow through.
        self.shared_mlp = getattr(source, "shared_mlp")

        self.num_experts = int(getattr(source, "num_experts"))
        self.top_k = int(getattr(source, "top_k"))
        self.hidden_dim = int(self.experts.hidden_dim)
        self.gate.requires_grad_(False)

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

    def _route_tokens_to_experts(self, router_logits: torch.Tensor, out_dtype: torch.dtype):
        # Verbatim replication of HunYuanMoEV1Moe.route_tokens_to_experts:
        # fp32 softmax over experts, top-k, renormalize, cast to activations dtype.
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return selected_experts, routing_weights.to(out_dtype)

    def _compute_routing(self, hidden_states_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_logits = self.gate(hidden_states_3d)
            if isinstance(router_logits, tuple):
                router_logits = router_logits[0]
            top_k_index, top_k_weights = self._route_tokens_to_experts(
                router_logits, hidden_states_3d.dtype
            )
        if not self.router_debug_grad:
            top_k_weights = top_k_weights.detach()
            top_k_index = top_k_index.detach()
        return top_k_index, top_k_weights, None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymHunyuanMoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        with prof_range(self._forward_range("shared_mlp")):
            shared_out = self.shared_mlp(hidden_states)
        top_k_index, top_k_weights, _router_logits = self._compute_routing(hidden_states)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
        flat = hidden_states.reshape(-1, input_shape[-1])
        with prof_range(self._forward_range("experts")):
            routed = self.experts(flat, top_k_index, top_k_weights)
        return routed.view(input_shape) + shared_out


def wrap_hunyuan_moe_block(
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
) -> AsymHunyuanMoeBlock:
    return AsymHunyuanMoeBlock(
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
    "AsymHunyuanMoeBlock",
    "is_hunyuan_experts",
    "is_hunyuan_moe_block",
    "wrap_hunyuan_moe_block",
]
