"""GLM-4.7-Flash (glm4_moe_lite) family wrapper (model_integration.md, #5 of 6).

REPLICATED code path (campaign rule): standalone family-facing layer; engine =
shared `AsymQwen3Experts` (tf-5.6 `Glm4MoeLiteNaiveMoe` carries the identical
packed layout `[E, 2I, H]/[E, H, I]`, silu, standard forward contract).

Family deltas vs the Mixtral/Phimoe/Hunyuan clones:
- the gate (`Glm4MoeLiteTopkRouter`) returns raw fp32 LOGITS; DeepSeek-V3-style
  routing happens in the BLOCK: sigmoid scores + `e_score_correction_bias` for
  CHOICE only, group-limited top-k (n_group/topk_group), weights gathered from
  the UN-corrected sigmoid scores, optional renorm, `routed_scaling_factor`.
  Replicated verbatim under no-grad; weights cast to activation dtype for the
  grouped engine;
- shared experts (`.shared_experts`, dense gate/up/down MLP) kept as the
  original module: GPU-resident, standard PEFT LoRA, grads flow, output added;
- first-k-dense / "dense" `mlp_layer_types` layers are plain `Glm4MoeLiteMLP`
  and never reach this wrapper (generic dense path handles them).
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


def _name_matches(obj: object) -> bool:
    lowered = type(obj).__name__.lower()
    return "glm4moelite" in lowered


def _is_3d_parameter(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, torch.Tensor) and value.dim() == 3


def _resolve_glm47_expert_act_fn(module: nn.Module):
    act_fn = getattr(module, "act_fn", None)
    if callable(act_fn):
        return act_fn
    hidden_act = str(getattr(getattr(module, "config", None), "hidden_act", "")).lower()
    if hidden_act in {"silu", "swish"}:
        return F.silu
    return None


def is_glm47_experts(module: nn.Module) -> bool:
    if not _name_matches(module):
        return False
    if not (_is_3d_parameter(module, "gate_up_proj") and _is_3d_parameter(module, "down_proj")):
        return False
    for attr in ("num_experts", "hidden_dim", "intermediate_dim"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    if not callable(_resolve_glm47_expert_act_fn(module)):
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in ("hidden_states", "top_k_index", "top_k_weights"))


def is_glm47_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_glm47_moe_block", False):
        return False
    if not _name_matches(module):
        return False
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    shared = getattr(module, "shared_experts", None)
    if not isinstance(gate, nn.Module) or not is_glm47_experts(experts):
        return False
    if not isinstance(shared, nn.Module):
        return False
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(shared, leaf, None), nn.Module):
            return False
    for attr in ("n_routed_experts", "n_group", "topk_group", "top_k"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    if not isinstance(getattr(gate, "e_score_correction_bias", None), torch.Tensor):
        return False
    return callable(getattr(gate, "forward", None))


class AsymGlm47MoeBlock(nn.Module):
    """GLM-4.7-Flash MoE block wrapper: frozen gate + replicated group top-k + shared experts."""

    _is_asym_glm47_moe_block = True

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
            raise ValueError(f"AsymGlm47MoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_glm47_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a GLM-4.7-Flash MoE block with gate/experts/shared_experts: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        # Preserve the installed GLM module order: experts, gate, shared_experts.
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
        self.gate = getattr(source, "gate")
        # Shared dense MLP stays the ORIGINAL module: GPU-resident, standard PEFT
        # LoRA targets (...mlp.shared_experts.*), grads flow through.
        self.shared_experts = getattr(source, "shared_experts")

        self.n_routed_experts = int(getattr(source, "n_routed_experts"))
        self.n_group = int(getattr(source, "n_group"))
        self.topk_group = int(getattr(source, "topk_group"))
        self.norm_topk_prob = bool(getattr(source, "norm_topk_prob"))
        self.routed_scaling_factor = float(getattr(source, "routed_scaling_factor"))
        self.top_k = int(getattr(source, "top_k"))
        self.hidden_dim = int(self.experts.hidden_dim)
        self.num_experts = int(self.experts.num_experts)
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

    def _route_tokens_to_experts(self, router_logits: torch.Tensor):
        # Verbatim replication of Glm4MoeLiteMoE.route_tokens_to_experts.
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _compute_routing(self, hidden_states_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_logits = self.gate(hidden_states_3d)
            if isinstance(router_logits, tuple):
                router_logits = router_logits[0]
            top_k_index, top_k_weights = self._route_tokens_to_experts(router_logits)
            top_k_weights = top_k_weights.to(dtype=hidden_states_3d.dtype)
        if not self.router_debug_grad:
            top_k_weights = top_k_weights.detach()
            top_k_index = top_k_index.detach()
        return top_k_index, top_k_weights, None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymGlm47MoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        residuals = hidden_states
        top_k_index, top_k_weights, _router_logits = self._compute_routing(hidden_states)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
        flat = hidden_states.reshape(-1, input_shape[-1])
        with prof_range(self._forward_range("experts")):
            routed = self.experts(flat, top_k_index, top_k_weights)
        with prof_range(self._forward_range("shared_experts")):
            shared_out = self.shared_experts(residuals)
        return routed.view(input_shape) + shared_out


def wrap_glm47_moe_block(
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
) -> AsymGlm47MoeBlock:
    return AsymGlm47MoeBlock(
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
    "AsymGlm47MoeBlock",
    "is_glm47_experts",
    "is_glm47_moe_block",
    "wrap_glm47_moe_block",
]
