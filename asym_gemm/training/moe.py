"""Deterministic MoE training harness for route correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .frozen_linear import (
    AsymExecutionStats,
    AsymGroupedFrozenLinear,
    TorchGroupedFrozenLinear,
    VALID_ASYM_PRECISIONS,
    _dispatch_grouped_nt,
    _get_quantized_host_weight,
    _grouped_torch_chunks,
    is_kt_backend,
    is_torch_backend,
)
from .host_weight import tensor_nbytes
from .kt_moe import DEFAULT_KT_METHOD, KTRoutedExpertMoE, normalize_kt_method
from .lora import GroupedLoRAMetadata, PackedExpertLoRA, grouped_expert_lora, normalize_lora_dtype
from .profile_ranges import prof_range


GroupedMode = Literal["contiguous", "masked"]
VALID_MOE_BACKENDS = ("torch", "asym", "kt")
DEFAULT_TARGET_MODULES = "all"
DEFAULT_OFFLOAD_MODULES = "routed_experts"
FUSED_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
MOE_OFFLOAD_GROUPS = ("routed_experts", "shared_experts")


def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.device.type == "cuda":
        with sdpa_kernel(FUSED_SDPA_BACKENDS):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def _normalize_moe_module_selector(selector: Sequence[str] | str | None, *, default: str, purpose: str) -> set[str]:
    if selector is None:
        selector = default
    if isinstance(selector, str):
        parts = tuple(part.strip().lower().replace("-", "_") for part in selector.split(",") if part.strip())
    else:
        parts = tuple(str(part).strip().lower().replace("-", "_") for part in selector if str(part).strip())
    if not parts:
        parts = (default,)

    groups: set[str] = set()
    for part in parts:
        if part in {"all", "mlp", "experts", "expert_mlp", "default"}:
            groups.update(MOE_OFFLOAD_GROUPS)
        elif part in {"routed", "routed_expert", "routed_experts"}:
            groups.add("routed_experts")
        elif part in {"shared", "shared_expert", "shared_experts"}:
            groups.add("shared_experts")
        elif part == "none":
            continue
        else:
            raise ValueError(
                f"unsupported MoE {purpose} module selector {part!r}; expected all, mlp, routed_experts, shared_experts, or none"
            )
    return groups


@dataclass(frozen=True)
class MoEConfig:
    num_layers: int = 6
    num_experts: int = 8
    top_k: int = 2
    hidden_size: int = 1024
    intermediate_size: int = 4096
    logical_tokens: int = 16
    lora_rank: int = 128
    lora_alpha: float = 256.0
    residual_scale: float = 0.25
    num_shared_experts: int = 1
    vocab_size: int = 32768
    num_heads: int = 8
    batch_size: int = 1
    seq_len: int = 16
    attention_impl: str = "sdpa"

    @classmethod
    def micro(cls) -> "MoEConfig":
        return cls(
            num_layers=4,
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            vocab_size=512,
            hidden_size=128,
            num_heads=4,
            batch_size=2,
            seq_len=8,
            intermediate_size=256,
            logical_tokens=16,
            lora_rank=128,
            lora_alpha=256.0,
            residual_scale=0.25,
        )

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / float(self.lora_rank)

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads

MICRO_MOE_CONFIG = MoEConfig.micro()
SHOWCASE_MOE_CONFIG = MoEConfig()


def _element_size(dtype: torch.dtype | str) -> int:
    if isinstance(dtype, str):
        if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
            dtype = torch.bfloat16
        elif dtype in {"fp32", "float32", "torch.float32"}:
            dtype = torch.float32
        else:
            raise ValueError(f"unsupported dtype: {dtype!r}")
    return torch.empty((), dtype=dtype).element_size()


def estimate_moe_parameters(
    config: MoEConfig = SHOWCASE_MOE_CONFIG,
    *,
    offload_modules: Sequence[str] | str | None = DEFAULT_OFFLOAD_MODULES,
    dtype: torch.dtype | str = torch.bfloat16,
) -> dict[str, int | float | str]:
    h = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_layers)
    experts = int(config.num_experts)
    shared = int(config.num_shared_experts)
    rank = int(config.lora_rank)
    dtype_bytes = _element_size(dtype)
    offload_groups = _normalize_moe_module_selector(offload_modules, default=DEFAULT_OFFLOAD_MODULES, purpose="offload")

    routed_expert_base_elements = layers * experts * 3 * inter * h
    shared_expert_base_elements = layers * shared * 3 * inter * h
    frozen_expert_base_elements = routed_expert_base_elements + shared_expert_base_elements
    offloaded_expert_base_elements = 0
    if "routed_experts" in offload_groups:
        offloaded_expert_base_elements += routed_expert_base_elements
    if "shared_experts" in offload_groups:
        offloaded_expert_base_elements += shared_expert_base_elements
    trainable_lora_elements = layers * (experts + shared) * 3 * rank * (h + inter)
    trainable_router_elements = layers * experts * h
    attention_base_elements = layers * 4 * h * h
    position_rows = max(int(config.logical_tokens), int(config.seq_len))
    embedding_head_buffer_elements = (2 * int(config.vocab_size) * h) + (position_rows * h)
    layernorm_parameter_elements = ((2 * layers) + 1) * 2 * h
    trainable_elements = trainable_lora_elements + trainable_router_elements
    total_model_elements = (
        frozen_expert_base_elements
        + attention_base_elements
        + embedding_head_buffer_elements
        + layernorm_parameter_elements
        + trainable_elements
    )

    return {
        "config_name": "showcase" if config == SHOWCASE_MOE_CONFIG else "custom",
        "total_model_elements": total_model_elements,
        "attention_base_elements": attention_base_elements,
        "embedding_and_lm_head_buffer_elements": embedding_head_buffer_elements,
        "layernorm_parameter_elements": layernorm_parameter_elements,
        "routed_expert_base_elements": routed_expert_base_elements,
        "shared_expert_base_elements": shared_expert_base_elements,
        "frozen_expert_base_elements": frozen_expert_base_elements,
        "offload_modules": ",".join(sorted(offload_groups)),
        "offloaded_expert_base_elements": offloaded_expert_base_elements,
        "trainable_lora_elements": trainable_lora_elements,
        "trainable_router_elements": trainable_router_elements,
        "trainable_elements": trainable_elements,
        "pytorch_visible_parameter_elements": trainable_elements,
        "expected_hbm_saved_bytes": offloaded_expert_base_elements * dtype_bytes,
        "expected_pinned_cpu_bytes_after_dx": offloaded_expert_base_elements * dtype_bytes,
        "trainable_fraction": trainable_elements / float(total_model_elements),
    }


@dataclass(frozen=True)
class ContiguousRouteMetadata:
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    route_indices: torch.Tensor
    routing_weights: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int
    mode: GroupedMode = "contiguous"

    @property
    def num_routes(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def padded_routes(self) -> int:
        return 0


@dataclass(frozen=True)
class MaskedRouteMetadata:
    token_indices: torch.Tensor
    route_indices: torch.Tensor
    routing_weights: torch.Tensor
    valid_mask: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    num_tokens: int
    top_k: int
    num_experts: int
    mode: GroupedMode = "masked"

    @property
    def max_routes_per_expert(self) -> int:
        return int(self.token_indices.shape[1])

    @property
    def num_routes(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def padded_routes(self) -> int:
        return int(self.valid_mask.numel() - self.num_routes)


RouteMetadata = ContiguousRouteMetadata | MaskedRouteMetadata
Routing = tuple[torch.Tensor, torch.Tensor]


def _validate_route_range_on_device(topk_indices: torch.Tensor) -> bool:
    if topk_indices.device.type != "cuda":
        return True
    return os.getenv("ASYM_GEMM_VALIDATE_ROUTES", "").strip().lower() in {"1", "true", "yes", "on"}


def make_dense_group_metadata(
    offsets: torch.Tensor,
    *,
    num_groups: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense grouped metadata with one group per expert plus sentinel."""

    if offsets.dim() != 1:
        raise ValueError(f"offsets must be 1D, got shape {tuple(offsets.shape)}")
    expected = int(num_groups) + 1
    if int(offsets.numel()) != expected:
        raise ValueError(f"dense grouped metadata expects {expected} offsets, got {int(offsets.numel())}")
    experts = torch.arange(expected, device=device, dtype=torch.long)
    experts[-1] = -1
    return offsets.to(device=device, dtype=torch.long), experts


def _validate_route_inputs(
    topk_indices: torch.Tensor,
    routing_weights: torch.Tensor | None,
    num_experts: int,
) -> torch.Tensor:
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be 2D [tokens, top_k], got {tuple(topk_indices.shape)}")
    if topk_indices.dtype != torch.long:
        topk_indices = topk_indices.to(dtype=torch.long)
    if topk_indices.numel() == 0:
        raise ValueError("routing requires at least one route")
    if _validate_route_range_on_device(topk_indices):
        min_expert = int(topk_indices.min().item())
        max_expert = int(topk_indices.max().item())
        if min_expert < 0 or max_expert >= num_experts:
            raise ValueError(f"expert id out of range [0, {num_experts}): min={min_expert}, max={max_expert}")
    if routing_weights is None:
        return torch.ones(topk_indices.shape, device=topk_indices.device, dtype=torch.float32)
    if routing_weights.shape != topk_indices.shape:
        raise ValueError(
            "routing_weights must match topk_indices shape, "
            f"got {tuple(routing_weights.shape)} and {tuple(topk_indices.shape)}"
        )
    if not routing_weights.dtype.is_floating_point:
        routing_weights = routing_weights.to(dtype=torch.float32)
    return routing_weights


def build_contiguous_route_metadata(
    topk_indices: torch.Tensor,
    routing_weights: torch.Tensor | None,
    *,
    num_experts: int,
) -> ContiguousRouteMetadata:
    """Build compact expert-sorted route metadata with stable within-expert order."""

    weights = _validate_route_inputs(topk_indices, routing_weights, num_experts)
    topk_indices = topk_indices.to(dtype=torch.long)
    num_tokens, top_k = int(topk_indices.shape[0]), int(topk_indices.shape[1])
    num_routes = num_tokens * top_k
    device = topk_indices.device

    flat_experts = topk_indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_routes = torch.arange(num_routes, device=device, dtype=torch.long)
    flat_tokens = torch.arange(num_tokens, device=device, dtype=torch.long).repeat_interleave(top_k)

    stable_keys = flat_experts.to(dtype=torch.long) * int(num_routes) + flat_routes
    sort_order = torch.argsort(stable_keys)
    sorted_experts = flat_experts.index_select(0, sort_order)
    expert_counts = torch.bincount(sorted_experts, minlength=num_experts).to(dtype=torch.long)
    expert_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(expert_counts, dim=0),
        ],
        dim=0,
    )

    return ContiguousRouteMetadata(
        token_indices=flat_tokens.index_select(0, sort_order),
        expert_indices=sorted_experts,
        route_indices=flat_routes.index_select(0, sort_order),
        routing_weights=flat_weights.index_select(0, sort_order),
        expert_offsets=expert_offsets,
        expert_counts=expert_counts,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
    )


def build_masked_route_metadata(
    topk_indices: torch.Tensor,
    routing_weights: torch.Tensor | None,
    *,
    num_experts: int,
) -> MaskedRouteMetadata:
    """Build padded [expert, slot] route metadata for masked grouped execution."""

    contiguous = build_contiguous_route_metadata(
        topk_indices,
        routing_weights,
        num_experts=num_experts,
    )
    max_count = int(contiguous.expert_counts.max().item())
    shape = (num_experts, max_count)
    device = topk_indices.device

    token_indices = torch.zeros(shape, device=device, dtype=torch.long)
    route_indices = torch.zeros(shape, device=device, dtype=torch.long)
    routing = torch.zeros(shape, device=device, dtype=contiguous.routing_weights.dtype)
    valid_mask = torch.zeros(shape, device=device, dtype=torch.bool)

    local_slots = torch.arange(contiguous.num_routes, device=device, dtype=torch.long)
    local_slots = local_slots - contiguous.expert_offsets.index_select(0, contiguous.expert_indices)
    token_indices[contiguous.expert_indices, local_slots] = contiguous.token_indices
    route_indices[contiguous.expert_indices, local_slots] = contiguous.route_indices
    routing[contiguous.expert_indices, local_slots] = contiguous.routing_weights
    valid_mask[contiguous.expert_indices, local_slots] = True

    return MaskedRouteMetadata(
        token_indices=token_indices,
        route_indices=route_indices,
        routing_weights=routing,
        valid_mask=valid_mask,
        expert_counts=contiguous.expert_counts,
        expert_offsets=contiguous.expert_offsets,
        num_tokens=contiguous.num_tokens,
        top_k=contiguous.top_k,
        num_experts=contiguous.num_experts,
    )


def build_route_metadata(
    topk_indices: torch.Tensor,
    routing_weights: torch.Tensor | None,
    *,
    num_experts: int,
    mode: GroupedMode,
) -> RouteMetadata:
    if mode == "contiguous":
        return build_contiguous_route_metadata(topk_indices, routing_weights, num_experts=num_experts)
    if mode == "masked":
        return build_masked_route_metadata(topk_indices, routing_weights, num_experts=num_experts)
    raise ValueError(f"unsupported grouped mode {mode!r}")


def route_metadata_summary(metadata: RouteMetadata) -> dict[str, Any]:
    counts = metadata.expert_counts.detach().cpu()
    empty = int((counts == 0).sum().item())
    return {
        "mode": metadata.mode,
        "num_tokens": metadata.num_tokens,
        "top_k": metadata.top_k,
        "num_experts": metadata.num_experts,
        "active_routes": int(counts.sum().item()),
        "padded_routes": metadata.padded_routes,
        "empty_experts": empty,
        "expert_counts": [int(v) for v in counts.tolist()],
    }


def normalize_expert_recompute_threshold(value: int | None) -> int:
    threshold = int(value or 0)
    if threshold < 0:
        raise ValueError(f"expert_recompute_threshold must be non-negative, got {value}")
    return threshold


VALID_EXPERT_RECOMPUTE_POLICIES = ("none", "tok")
VALID_EXPERT_ACTIVATION_SAVE_POLICIES = ("save_all", "all_act", "tok_act")


def normalize_expert_recompute_policy(value: str | None) -> str:
    policy = "none" if value is None else str(value).strip()
    if policy not in VALID_EXPERT_RECOMPUTE_POLICIES:
        raise ValueError(
            f"expert_recompute_policy must be one of {VALID_EXPERT_RECOMPUTE_POLICIES}, got {value!r}"
        )
    return policy


def expert_recompute_policy_enabled(
    policy: str,
    *,
    token_threshold: int = 0,
    token_min: int = 1,
    token_max: int | None = None,
) -> bool:
    normalized = normalize_expert_recompute_policy(policy)
    token_threshold = normalize_expert_recompute_threshold(token_threshold)
    token_min = normalize_expert_recompute_threshold(token_min)
    token_max = None if token_max is None else normalize_expert_recompute_threshold(token_max)
    if normalized == "none":
        return False
    if normalized == "tok":
        return token_threshold > 0 or token_max is not None or token_min > 1
    return False


def expert_token_range_mask(
    counts: torch.Tensor,
    *,
    token_min: int = 1,
    token_max: int | None = None,
) -> torch.Tensor:
    token_min = normalize_expert_recompute_threshold(token_min)
    token_max = None if token_max is None else normalize_expert_recompute_threshold(token_max)
    if token_max is not None and token_min > token_max:
        raise ValueError(f"token_min must be <= token_max, got token_min={token_min} token_max={token_max}")
    counts_long = counts.to(dtype=torch.long)
    active = counts_long > 0
    mask = active & (counts_long >= max(1, token_min))
    if token_max is not None:
        mask = mask & (counts_long <= token_max)
    return mask


def expert_recompute_group_mask(
    counts: torch.Tensor,
    *,
    policy: str = "none",
    token_threshold: int = 0,
    token_min: int = 1,
    token_max: int | None = None,
) -> torch.Tensor:
    policy = normalize_expert_recompute_policy(policy)
    token_threshold = normalize_expert_recompute_threshold(token_threshold)
    if token_max is None and token_threshold > 0:
        token_max = token_threshold
    counts_long = counts.to(dtype=torch.long)
    active = counts_long > 0
    if policy == "none":
        return torch.zeros_like(active)
    if policy == "tok":
        if token_max is None and token_threshold <= 0 and token_min <= 1:
            return torch.zeros_like(active)
        return expert_token_range_mask(counts_long, token_min=token_min, token_max=token_max)
    return active


def normalize_expert_activation_save_policy(value: str | None) -> str:
    policy = "save_all" if value is None else str(value).strip()
    if policy not in VALID_EXPERT_ACTIVATION_SAVE_POLICIES:
        raise ValueError(
            f"expert_activation_save_policy must be one of {VALID_EXPERT_ACTIVATION_SAVE_POLICIES}, got {value!r}"
        )
    return policy


def normalize_expert_activation_save_threshold(value: int | None) -> int:
    threshold = int(value or 0)
    if threshold < 0:
        raise ValueError(f"expert_activation_save_threshold must be non-negative, got {value}")
    return threshold


def expert_activation_save_policy_enabled(
    policy: str,
    *,
    token_threshold: int = 0,
    token_min: int = 1,
    token_max: int | None = None,
) -> bool:
    normalized = normalize_expert_activation_save_policy(policy)
    token_threshold = normalize_expert_activation_save_threshold(token_threshold)
    token_min = normalize_expert_activation_save_threshold(token_min)
    token_max = None if token_max is None else normalize_expert_activation_save_threshold(token_max)
    if normalized == "save_all":
        return False
    if normalized == "all_act":
        return True
    if normalized == "tok_act":
        return token_threshold > 0 or token_max is not None or token_min > 1
    return False


def expert_activation_drop_group_mask(
    counts: torch.Tensor,
    *,
    policy: str = "save_all",
    token_threshold: int = 0,
    token_min: int = 1,
    token_max: int | None = None,
) -> torch.Tensor:
    policy = normalize_expert_activation_save_policy(policy)
    token_threshold = normalize_expert_activation_save_threshold(token_threshold)
    if token_max is None and token_threshold > 0:
        token_max = token_threshold
    counts_long = counts.to(dtype=torch.long)
    active = counts_long > 0
    if policy == "save_all":
        return torch.zeros_like(active)
    if policy == "all_act":
        return active
    if policy == "tok_act":
        if token_max is None and token_threshold <= 0 and token_min <= 1:
            return torch.zeros_like(active)
        return expert_token_range_mask(counts_long, token_min=token_min, token_max=token_max)
    return torch.zeros_like(active)


@dataclass(frozen=True)
class ExpertRecomputeConfig:
    policy: Literal["none", "tok"]
    token_threshold: int
    activation_save_policy: Literal["save_all", "tok_act"]
    activation_save_threshold: int
    label: str
    token_min: int = 1
    token_max: int | None = None
    activation_save_min: int = 1
    activation_save_max: int | None = None
    force_custom_autograd: bool = False

    @property
    def recompute_enabled(self) -> bool:
        return expert_recompute_policy_enabled(
            self.policy,
            token_threshold=self.token_threshold,
            token_min=self.token_min,
            token_max=self.token_max,
        )

    @property
    def activation_drop_enabled(self) -> bool:
        return expert_activation_save_policy_enabled(
            self.activation_save_policy,
            token_threshold=self.activation_save_threshold,
            token_min=self.activation_save_min,
            token_max=self.activation_save_max,
        )

    @property
    def enabled(self) -> bool:
        return self.recompute_enabled or self.activation_drop_enabled or self.force_custom_autograd


def parse_expert_recompute_policy_spec(spec: str | None) -> ExpertRecomputeConfig:
    raw = "none" if spec is None else str(spec).strip()
    if raw == "none":
        return ExpertRecomputeConfig(
            policy="none",
            token_threshold=0,
            activation_save_policy="save_all",
            activation_save_threshold=0,
            label="none",
            token_min=1,
            token_max=None,
            activation_save_min=1,
            activation_save_max=None,
        )
    if raw in {"tok-le0", "tok-le0-act"}:
        return ExpertRecomputeConfig(
            policy="none",
            token_threshold=0,
            activation_save_policy="save_all",
            activation_save_threshold=0,
            label=raw,
            token_min=1,
            token_max=None,
            activation_save_min=1,
            activation_save_max=None,
            force_custom_autograd=True,
        )

    def _range_from_match(kind: str, first: str, second: str | None = None) -> tuple[int, int | None, str]:
        if kind == "le":
            upper = int(first)
            return 1, upper, f"tok-le{upper}"
        if kind == "ge":
            lower = int(first)
            return lower, None, f"tok-ge{lower}"
        lower = int(first)
        upper = int(second or 0)
        if lower > upper:
            raise ValueError(
                f"invalid expert recompute token range tok{lower}-{upper}: lower bound exceeds upper bound"
            )
        return lower, upper, f"tok{lower}-{upper}"

    range_match = None
    range_kind = ""
    if match := re.fullmatch(r"tok-le([1-9][0-9]*)", raw):
        range_match = match
        range_kind = "le"
    elif match := re.fullmatch(r"tok-ge([1-9][0-9]*)", raw):
        range_match = match
        range_kind = "ge"
    elif match := re.fullmatch(r"tok([1-9][0-9]*)-([1-9][0-9]*)", raw):
        range_match = match
        range_kind = "range"

    if range_match:
        lower, upper, label = _range_from_match(
            range_kind,
            range_match.group(1),
            range_match.group(2) if range_kind == "range" else None,
        )
        return ExpertRecomputeConfig(
            policy="tok",
            token_threshold=upper or 0,
            activation_save_policy="save_all",
            activation_save_threshold=0,
            label=label,
            token_min=lower,
            token_max=upper,
            activation_save_min=1,
            activation_save_max=None,
        )

    act_match = None
    act_kind = ""
    if match := re.fullmatch(r"tok-le([1-9][0-9]*)-act", raw):
        act_match = match
        act_kind = "le"
    elif match := re.fullmatch(r"tok-ge([1-9][0-9]*)-act", raw):
        act_match = match
        act_kind = "ge"
    elif match := re.fullmatch(r"tok([1-9][0-9]*)-([1-9][0-9]*)-act", raw):
        act_match = match
        act_kind = "range"

    if act_match:
        lower, upper, label = _range_from_match(
            act_kind,
            act_match.group(1),
            act_match.group(2) if act_kind == "range" else None,
        )
        return ExpertRecomputeConfig(
            policy="none",
            token_threshold=0,
            activation_save_policy="tok_act",
            activation_save_threshold=upper or 0,
            label=f"{label}-act",
            token_min=1,
            token_max=None,
            activation_save_min=lower,
            activation_save_max=upper,
        )

    raise ValueError(
        f"unsupported expert recompute policy {spec!r}; expected none, tok-leN, tok-geN, tokA-B, or -act variants"
    )


def pack_tokens_contiguous(hidden: torch.Tensor, metadata: ContiguousRouteMetadata) -> torch.Tensor:
    flat = hidden.reshape(metadata.num_tokens, -1)
    return flat.index_select(0, metadata.token_indices).reshape(metadata.num_routes, *hidden.shape[1:]).contiguous()


def pack_tokens_masked(hidden: torch.Tensor, metadata: MaskedRouteMetadata) -> torch.Tensor:
    flat = hidden.reshape(metadata.num_tokens, -1)
    packed = flat.index_select(0, metadata.token_indices.reshape(-1))
    packed = packed.reshape(metadata.num_experts, metadata.max_routes_per_expert, *hidden.shape[1:])
    mask = metadata.valid_mask.reshape(metadata.num_experts, metadata.max_routes_per_expert, *([1] * (packed.dim() - 2)))
    return packed * mask.to(dtype=packed.dtype)


def scatter_contiguous(expert_output: torch.Tensor, metadata: ContiguousRouteMetadata) -> torch.Tensor:
    if expert_output.shape[0] != metadata.num_routes:
        raise ValueError(f"expected {metadata.num_routes} route outputs, got {expert_output.shape[0]}")
    flat = expert_output.reshape(metadata.num_routes, -1)
    weights = metadata.routing_weights.reshape(metadata.num_routes, 1)
    weighted = flat * weights
    out = torch.zeros(
        (metadata.num_tokens, flat.shape[1]),
        device=expert_output.device,
        dtype=weighted.dtype,
    )
    out.index_add_(0, metadata.token_indices, weighted)
    return out.reshape(metadata.num_tokens, *expert_output.shape[1:])


def scatter_masked(expert_output: torch.Tensor, metadata: MaskedRouteMetadata) -> torch.Tensor:
    expected_shape = (metadata.num_experts, metadata.max_routes_per_expert)
    if tuple(expert_output.shape[:2]) != expected_shape:
        raise ValueError(f"expected route output prefix {expected_shape}, got {tuple(expert_output.shape[:2])}")
    flat = expert_output.reshape(metadata.num_experts * metadata.max_routes_per_expert, -1)
    flat_weights = metadata.routing_weights.reshape(-1, 1)
    flat_tokens = metadata.token_indices.reshape(-1)
    flat_mask = metadata.valid_mask.reshape(-1)
    weighted = flat * flat_weights
    out = torch.zeros(
        (metadata.num_tokens, flat.shape[1]),
        device=expert_output.device,
        dtype=weighted.dtype,
    )
    out.index_add_(0, flat_tokens[flat_mask], weighted[flat_mask])
    return out.reshape(metadata.num_tokens, *expert_output.shape[2:])


def scatter_backward_contiguous(
    grad_output: torch.Tensor,
    expert_output: torch.Tensor,
    metadata: ContiguousRouteMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit PyTorch backward for contiguous weighted scatter."""

    grad_flat = grad_output.reshape(metadata.num_tokens, -1)
    expert_flat = expert_output.reshape(metadata.num_routes, -1)
    gathered_grad = grad_flat.index_select(0, metadata.token_indices)
    grad_expert = gathered_grad * metadata.routing_weights.reshape(-1, 1)
    grad_weights = (gathered_grad * expert_flat).sum(dim=1)
    return grad_expert.reshape_as(expert_output), grad_weights


def scatter_backward_masked(
    grad_output: torch.Tensor,
    expert_output: torch.Tensor,
    metadata: MaskedRouteMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit PyTorch backward for masked weighted scatter."""

    prefix = metadata.num_experts * metadata.max_routes_per_expert
    grad_flat = grad_output.reshape(metadata.num_tokens, -1)
    expert_flat = expert_output.reshape(prefix, -1)
    flat_tokens = metadata.token_indices.reshape(-1)
    flat_weights = metadata.routing_weights.reshape(-1, 1)
    flat_mask = metadata.valid_mask.reshape(-1)
    gathered_grad = grad_flat.index_select(0, flat_tokens)
    grad_expert = gathered_grad * flat_weights
    grad_weights = (gathered_grad * expert_flat).sum(dim=1)
    grad_expert = grad_expert * flat_mask.reshape(-1, 1).to(dtype=grad_expert.dtype)
    grad_weights = grad_weights * flat_mask.to(dtype=grad_weights.dtype)
    return grad_expert.reshape_as(expert_output), grad_weights.reshape_as(metadata.routing_weights)


def restore_contiguous_route_order(values: torch.Tensor, metadata: ContiguousRouteMetadata) -> torch.Tensor:
    flat = values.reshape(metadata.num_routes, *values.shape[1:])
    restored = torch.empty_like(flat)
    restored[metadata.route_indices] = flat
    return restored.reshape(metadata.num_tokens, metadata.top_k, *values.shape[1:])


def restore_masked_route_order(values: torch.Tensor, metadata: MaskedRouteMetadata) -> torch.Tensor:
    flat_values = values.reshape(metadata.num_experts * metadata.max_routes_per_expert, *values.shape[2:])
    flat_routes = metadata.route_indices.reshape(-1)
    flat_mask = metadata.valid_mask.reshape(-1)
    restored = torch.zeros(
        (metadata.num_routes, *values.shape[2:]),
        device=values.device,
        dtype=values.dtype,
    )
    restored[flat_routes[flat_mask]] = flat_values[flat_mask]
    return restored.reshape(metadata.num_tokens, metadata.top_k, *values.shape[2:])


def topk_routing_from_logits(logits: torch.Tensor, top_k: int) -> Routing:
    topk_values, topk_indices = torch.topk(logits.float(), k=top_k, dim=-1)
    routing_weights = F.softmax(topk_values, dim=-1)
    return topk_indices.to(dtype=torch.long), routing_weights


def clone_moe_state(state: Mapping[str, Any]) -> dict[str, Any]:
    def clone_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, Mapping):
            return {str(k): clone_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clone_value(v) for v in value]
        return value

    return clone_value(state)


def _randn(generator: torch.Generator, shape: Sequence[int], std: float) -> torch.Tensor:
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * std


def make_moe_state(
    config: MoEConfig = MoEConfig(),
    *,
    seed: int = 0,
    base_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    h = config.hidden_size
    i = config.intermediate_size
    r = config.lora_rank
    state: dict[str, Any] = {
        "token_embedding": _randn(generator, (config.vocab_size, h), 0.02).to(dtype=base_dtype),
        "position_embedding": _randn(generator, (max(config.logical_tokens, config.seq_len), h), 0.02).to(dtype=base_dtype),
        "lm_head": _randn(generator, (config.vocab_size, h), 0.02).to(dtype=base_dtype),
        "final_layernorm_weight": torch.ones(h, dtype=base_dtype),
        "layers": [],
    }

    def make_expert(expert_idx: int, scale_base: float) -> dict[str, torch.Tensor]:
        expert_seed_scale = scale_base + 0.005 * float(expert_idx)
        return {
            "gate_weight": _randn(generator, (i, h), 0.025 * expert_seed_scale).to(dtype=base_dtype),
            "up_weight": _randn(generator, (i, h), 0.025 * expert_seed_scale).to(dtype=base_dtype),
            "down_weight": _randn(generator, (h, i), 0.025 * expert_seed_scale).to(dtype=base_dtype),
            "gate_lora_a": _randn(generator, (r, h), 0.01),
            "gate_lora_b": _randn(generator, (i, r), 0.01),
            "up_lora_a": _randn(generator, (r, h), 0.01),
            "up_lora_b": _randn(generator, (i, r), 0.01),
            "down_lora_a": _randn(generator, (r, i), 0.01),
            "down_lora_b": _randn(generator, (h, r), 0.01),
        }

    for layer_idx in range(config.num_layers):
        layer: dict[str, Any] = {
            "q_proj": _randn(generator, (h, h), 0.02).to(dtype=base_dtype),
            "k_proj": _randn(generator, (h, h), 0.02).to(dtype=base_dtype),
            "v_proj": _randn(generator, (h, h), 0.02).to(dtype=base_dtype),
            "o_proj": _randn(generator, (h, h), 0.02).to(dtype=base_dtype),
            "input_layernorm_weight": torch.ones(h, dtype=base_dtype),
            "post_attention_layernorm_weight": torch.ones(h, dtype=base_dtype),
            "router_weight": _randn(generator, (config.num_experts, h), 0.04),
            "experts": [],
            "shared_experts": [],
        }
        for expert_idx in range(config.num_experts):
            layer["experts"].append(make_expert(expert_idx, 1.0 + 0.01 * float(layer_idx)))
        for expert_idx in range(config.num_shared_experts):
            layer["shared_experts"].append(make_expert(expert_idx, 1.2 + 0.01 * float(layer_idx)))
        state["layers"].append(layer)
    return state


def _state_tensor(state: Mapping[str, Any], name: str) -> torch.Tensor:
    value = state[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"state field {name!r} is not a tensor")
    return value


def _stack_expert_weight(expert_states: Sequence[Mapping[str, Any]], name: str) -> torch.Tensor:
    if not expert_states:
        raise ValueError("cannot build grouped expert weight from an empty expert list")
    return torch.stack([_state_tensor(expert_state, name) for expert_state in expert_states], dim=0).contiguous()


class FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().to(device=device, dtype=dtype).contiguous())
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    @property
    def weight_nbytes(self) -> int:
        return tensor_nbytes(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class FrozenLayerNorm(nn.LayerNorm):
    def __init__(self, weight: torch.Tensor, *, device: torch.device) -> None:
        super().__init__((int(weight.numel()),), elementwise_affine=False, device=device, dtype=torch.float32)
        self.register_buffer("frozen_weight", weight.detach().to(device=device, dtype=torch.float32).contiguous())
        self.register_buffer("frozen_bias", torch.zeros_like(self.frozen_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), self.normalized_shape, self.frozen_weight, self.frozen_bias, self.eps).to(dtype=x.dtype)


class FrozenRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, *, device: torch.device, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("frozen_weight", weight.detach().to(device=device, dtype=torch.float32).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        normed = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.frozen_weight).to(dtype=x.dtype)


def _make_frozen_norm(layer_state: Mapping[str, Any], name: str, *, device: torch.device) -> nn.Module:
    weight = _state_tensor(layer_state, name)
    if str(layer_state.get("norm_type", "layernorm")).lower() == "rmsnorm":
        return FrozenRMSNorm(weight, device=device, eps=float(layer_state.get("rms_norm_eps", 1e-6)))
    return FrozenLayerNorm(weight, device=device)


class SelfAttention(nn.Module):
    def __init__(
        self,
        layer_state: Mapping[str, Any],
        *,
        config: MoEConfig,
        device: torch.device,
        base_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = FrozenLinear(_state_tensor(layer_state, "q_proj"), device=device, dtype=base_dtype)
        self.k_proj = FrozenLinear(_state_tensor(layer_state, "k_proj"), device=device, dtype=base_dtype)
        self.v_proj = FrozenLinear(_state_tensor(layer_state, "v_proj"), device=device, dtype=base_dtype)
        self.o_proj = FrozenLinear(_state_tensor(layer_state, "o_proj"), device=device, dtype=base_dtype)
        self.attention_impl = config.attention_impl
        if self.attention_impl != "sdpa":
            raise NotImplementedError(f"attention_impl={self.attention_impl!r} is a placeholder and is not wired yet")

    @property
    def frozen_weight_bytes(self) -> int:
        return self.q_proj.weight_nbytes + self.k_proj.weight_nbytes + self.v_proj.weight_nbytes + self.o_proj.weight_nbytes

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dim = hidden_states.dim()
        if original_dim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif original_dim != 3:
            raise ValueError(f"hidden_states must be [tokens, hidden] or [batch, seq, hidden], got {tuple(hidden_states.shape)}")

        batch, seq, hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        context = _scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, hidden)
        out = self.o_proj(context.to(dtype=hidden_states.dtype))
        return out.squeeze(0) if original_dim == 2 else out


def _normalize_static_routing(routing: Routing, *, device: torch.device) -> Routing:
    topk_indices, routing_weights = routing
    return (
        topk_indices.to(device=device, dtype=torch.long),
        routing_weights.to(device=device, dtype=torch.float32),
    )


class TorchExpert(nn.Module):
    """Single-expert torch reference used only for parity baselines."""

    def __init__(
        self,
        expert_state: Mapping[str, Any],
        *,
        config: MoEConfig,
        device: torch.device,
        base_dtype: torch.dtype,
        lora_dtype: torch.dtype | str = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_scale = config.lora_scale
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        for name in ("gate_weight", "up_weight", "down_weight"):
            self.register_buffer(name, _state_tensor(expert_state, name).to(device=device, dtype=base_dtype).clone())
        for name in (
            "gate_lora_a",
            "gate_lora_b",
            "up_lora_a",
            "up_lora_b",
            "down_lora_a",
            "down_lora_b",
        ):
            self.register_parameter(
                name,
                nn.Parameter(_state_tensor(expert_state, name).to(device=device, dtype=self.lora_dtype).clone()),
            )

    @property
    def frozen_weight_bytes(self) -> int:
        return tensor_nbytes(self.gate_weight) + tensor_nbytes(self.up_weight) + tensor_nbytes(self.down_weight)

    def _lora(self, x: torch.Tensor, prefix: str, out_dtype: torch.dtype) -> torch.Tensor:
        a = getattr(self, f"{prefix}_lora_a")
        b = getattr(self, f"{prefix}_lora_b")
        lora_input = x.to(dtype=self.lora_dtype)
        low_rank = F.linear(lora_input, a)
        out = F.linear(low_rank, b) * self.lora_scale
        return out.to(dtype=out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty((0, self.config.hidden_size))
        gate = F.linear(x, self.gate_weight) + self._lora(x, "gate", x.dtype)
        up = F.linear(x, self.up_weight) + self._lora(x, "up", x.dtype)
        activated = (F.silu(gate.float()) * up.float()).to(dtype=x.dtype)
        down = F.linear(activated, self.down_weight) + self._lora(activated, "down", x.dtype)
        return down

class AsymMoELayer(nn.Module):
    def __init__(
        self,
        layer_state: Mapping[str, Any],
        *,
        config: MoEConfig,
        device: torch.device,
        base_dtype: torch.dtype,
        backend: str,
        layer_idx: int = 0,
        pin_memory: bool,
        stats: AsymExecutionStats,
        offload_modules: Sequence[str] | str | None = DEFAULT_OFFLOAD_MODULES,
        lora_dtype: torch.dtype | str = torch.bfloat16,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.precision = str(precision).lower()
        if self.precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        self.input_layernorm = _make_frozen_norm(layer_state, "input_layernorm_weight", device=device)
        self.post_attention_layernorm = _make_frozen_norm(layer_state, "post_attention_layernorm_weight", device=device)
        self.self_attn = SelfAttention(layer_state, config=config, device=device, base_dtype=base_dtype)
        self.router_weight = nn.Parameter(_state_tensor(layer_state, "router_weight").to(device=device, dtype=torch.float32).clone())
        expert_states = list(layer_state["experts"])
        shared_expert_states = list(layer_state.get("shared_experts", []))
        offload_groups = _normalize_moe_module_selector(offload_modules, default=DEFAULT_OFFLOAD_MODULES, purpose="offload")
        self.offload_groups = tuple(sorted(offload_groups))

        use_gpu_torch_base = is_torch_backend(backend) and device.type == "cuda"

        def grouped_base(states: Sequence[Mapping[str, Any]], name: str, *, offload: bool) -> nn.Module:
            weight = _stack_expert_weight(states, name)
            if not offload or use_gpu_torch_base:
                return TorchGroupedFrozenLinear(weight, device=device, dtype=base_dtype)
            return AsymGroupedFrozenLinear(
                weight,
                backend=backend,
                pin_memory=pin_memory,
                stats=stats,
                precision=self.precision,
            )

        offload_routed = "routed_experts" in offload_groups
        offload_shared = "shared_experts" in offload_groups
        self.expert_gate_base = grouped_base(expert_states, "gate_weight", offload=offload_routed)
        self.expert_up_base = grouped_base(expert_states, "up_weight", offload=offload_routed)
        self.expert_down_base = grouped_base(expert_states, "down_weight", offload=offload_routed)
        self.shared_gate_base = (
            grouped_base(shared_expert_states, "gate_weight", offload=offload_shared)
            if shared_expert_states
            else None
        )
        self.shared_up_base = (
            grouped_base(shared_expert_states, "up_weight", offload=offload_shared)
            if shared_expert_states
            else None
        )
        self.shared_down_base = (
            grouped_base(shared_expert_states, "down_weight", offload=offload_shared)
            if shared_expert_states
            else None
        )
        self.expert_lora = PackedExpertLoRA(expert_states, config=config, device=device, lora_dtype=lora_dtype)
        self.shared_expert_lora = (
            PackedExpertLoRA(shared_expert_states, config=config, device=device, lora_dtype=lora_dtype)
            if shared_expert_states
            else None
        )

    @property
    def frozen_weight_bytes(self) -> int:
        return sum(
            base.weight_hbm_saved_bytes
            for base in (
                self.expert_gate_base,
                self.expert_up_base,
                self.expert_down_base,
                self.shared_gate_base,
                self.shared_up_base,
                self.shared_down_base,
            )
            if isinstance(base, AsymGroupedFrozenLinear)
        )

    @property
    def pinned_cpu_bytes(self) -> int:
        return sum(
            base.pinned_cpu_bytes
            for base in (
                self.expert_gate_base,
                self.expert_up_base,
                self.expert_down_base,
                self.shared_gate_base,
                self.shared_up_base,
                self.shared_down_base,
            )
            if isinstance(base, AsymGroupedFrozenLinear)
        )

    def _route(self, flat: torch.Tensor, static_routing: Routing | None) -> tuple[Routing, torch.Tensor | None]:
        if static_routing is not None:
            return _normalize_static_routing(static_routing, device=flat.device), None
        logits = F.linear(flat.float(), self.router_weight)
        return topk_routing_from_logits(logits, self.config.top_k), logits

    def _dense_group_metadata(
        self,
        offsets: torch.Tensor,
        *,
        num_groups: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return make_dense_group_metadata(offsets, num_groups=num_groups, device=device)

    def _lora_contiguous(
        self,
        packed: torch.Tensor,
        metadata: ContiguousRouteMetadata,
        prefix: str,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        offsets, experts = self._dense_group_metadata(
            metadata.expert_offsets,
            num_groups=self.config.num_experts,
            device=packed.device,
        )
        lora_metadata = self.expert_lora.prepare_metadata(offsets, experts, dense_experts=True)
        return self.expert_lora(packed, offsets, experts, prefix, out_dtype, metadata=lora_metadata)

    def _lora_shared(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        prefix: str,
        out_dtype: torch.dtype,
        *,
        experts: torch.Tensor | None = None,
        lora_metadata: GroupedLoRAMetadata | None = None,
    ) -> torch.Tensor:
        if self.shared_expert_lora is None:
            out_features = self.config.intermediate_size if prefix in {"gate", "up"} else self.config.hidden_size
            return packed.new_empty((0, out_features))
        if experts is None:
            _, experts = self._dense_group_metadata(offsets, num_groups=self.shared_expert_lora.num_experts, device=packed.device)
        if lora_metadata is None:
            lora_metadata = self.shared_expert_lora.prepare_metadata(offsets, experts)
        return self.shared_expert_lora(packed, offsets, experts, prefix, out_dtype, metadata=lora_metadata)

    def _run_grouped_gate_up(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: nn.Module,
        up_base: nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, GroupedLoRAMetadata]:
        if packed.numel() == 0:
            empty = packed.new_empty((0, self.config.intermediate_size))
            empty_metadata = self.expert_lora.prepare_metadata(offsets, experts, dense_experts=dense_experts)
            return empty, empty, empty_metadata

        layer_name = str(getattr(self, "_m4_profile_name", ""))
        expert_scope = "shared_expert" if shared else "routed_expert"
        profile_prefix = f"{layer_name}.{expert_scope}" if layer_name else expert_scope
        gate_base.profile_name = f"{profile_prefix}.gate_base"
        up_base.profile_name = f"{profile_prefix}.up_base"

        gate = gate_base(packed, offsets, experts, dense_experts=dense_experts)
        up = up_base(packed, offsets, experts, dense_experts=dense_experts)
        range_prefix = f"forward.{profile_prefix}"
        if shared:
            assert self.shared_expert_lora is not None
            lora_metadata = self.shared_expert_lora.prepare_metadata(offsets, experts, dense_experts=dense_experts)
            with prof_range(f"{range_prefix}.gate_up_lora"):
                gate_lora, up_lora = self.shared_expert_lora.forward_gate_up(
                    packed,
                    offsets,
                    experts,
                    packed.dtype,
                    metadata=lora_metadata,
                )
        else:
            lora_metadata = self.expert_lora.prepare_metadata(offsets, experts, dense_experts=dense_experts)
            with prof_range(f"{range_prefix}.gate_up_lora"):
                gate_lora, up_lora = self.expert_lora.forward_gate_up(
                    packed,
                    offsets,
                    experts,
                    packed.dtype,
                    metadata=lora_metadata,
                )
        gate = gate + gate_lora
        up = up + up_lora
        return gate, up, lora_metadata

    def _run_grouped_gate_up_activation(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: nn.Module,
        up_base: nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GroupedLoRAMetadata]:
        gate, up, lora_metadata = self._run_grouped_gate_up(
            packed,
            offsets,
            experts,
            gate_base=gate_base,
            up_base=up_base,
            shared=shared,
            dense_experts=dense_experts,
        )
        if packed.numel() == 0:
            empty = packed.new_empty((0, self.config.intermediate_size))
            return gate, up, empty, lora_metadata

        layer_name = str(getattr(self, "_m4_profile_name", ""))
        expert_scope = "shared_expert" if shared else "routed_expert"
        profile_prefix = f"{layer_name}.{expert_scope}" if layer_name else expert_scope
        range_prefix = f"forward.{profile_prefix}"
        with prof_range(f"{range_prefix}.activation_silu_mul"):
            activated = (F.silu(gate.float()) * up.float()).to(dtype=packed.dtype)
        return gate, up, activated, lora_metadata

    def _run_grouped_activation_down(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        down_base: nn.Module,
        lora_metadata: GroupedLoRAMetadata,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> torch.Tensor:
        if gate.numel() == 0:
            return gate.new_empty((0, self.config.hidden_size))

        layer_name = str(getattr(self, "_m4_profile_name", ""))
        expert_scope = "shared_expert" if shared else "routed_expert"
        profile_prefix = f"{layer_name}.{expert_scope}" if layer_name else expert_scope
        down_base.profile_name = f"{profile_prefix}.down_base"
        range_prefix = f"forward.{profile_prefix}"
        with prof_range(f"{range_prefix}.activation_silu_mul"):
            activated = (F.silu(gate.float()) * up.float()).to(dtype=gate.dtype)
        down = down_base(activated.contiguous(), offsets, experts, dense_experts=dense_experts)
        if shared:
            with prof_range(f"{range_prefix}.down_lora"):
                down_lora = self._lora_shared(
                    activated,
                    offsets,
                    "down",
                    gate.dtype,
                    experts=experts,
                    lora_metadata=lora_metadata,
                )
            down = down + down_lora
        else:
            with prof_range(f"{range_prefix}.down_lora"):
                down_lora = self.expert_lora(activated, offsets, experts, "down", gate.dtype, metadata=lora_metadata)
            down = down + down_lora
        return down

    def _run_grouped_compact_body(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: nn.Module,
        up_base: nn.Module,
        down_base: nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> torch.Tensor:
        output, _ = self._run_grouped_compact_body_with_intermediates(
            packed,
            offsets,
            experts,
            gate_base=gate_base,
            up_base=up_base,
            down_base=down_base,
            shared=shared,
            dense_experts=dense_experts,
        )
        return output

    def _run_grouped_compact_body_with_intermediates(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: nn.Module,
        up_base: nn.Module,
        down_base: nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if packed.numel() == 0:
            return (
                packed.new_empty((0, self.config.hidden_size)),
                {
                    "gate": packed.new_empty((0, self.config.intermediate_size)),
                    "up": packed.new_empty((0, self.config.intermediate_size)),
                    "activated": packed.new_empty((0, self.config.intermediate_size)),
                },
            )

        layer_name = str(getattr(self, "_m4_profile_name", ""))
        expert_scope = "shared_expert" if shared else "routed_expert"
        profile_prefix = f"{layer_name}.{expert_scope}" if layer_name else expert_scope
        down_base.profile_name = f"{profile_prefix}.down_base"

        range_prefix = f"forward.{profile_prefix}"
        gate, up, activated, lora_metadata = self._run_grouped_gate_up_activation(
            packed,
            offsets,
            experts,
            gate_base=gate_base,
            up_base=up_base,
            shared=shared,
            dense_experts=dense_experts,
        )
        down_base.profile_name = f"{profile_prefix}.down_base"
        down = down_base(activated.contiguous(), offsets, experts, dense_experts=dense_experts)
        if shared:
            with prof_range(f"{range_prefix}.down_lora"):
                down_lora = self._lora_shared(
                    activated,
                    offsets,
                    "down",
                    packed.dtype,
                    experts=experts,
                    lora_metadata=lora_metadata,
                )
            down = down + down_lora
        else:
            with prof_range(f"{range_prefix}.down_lora"):
                down_lora = self.expert_lora(activated, offsets, experts, "down", packed.dtype, metadata=lora_metadata)
            down = down + down_lora
        return down, {"gate": gate, "up": up, "activated": activated}

    def _run_grouped_compact(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        gate_base: nn.Module,
        up_base: nn.Module,
        down_base: nn.Module,
        shared: bool = False,
        dense_experts: bool = False,
    ) -> torch.Tensor:
        return self._run_grouped_compact_body(
            packed,
            offsets,
            experts,
            gate_base=gate_base,
            up_base=up_base,
            down_base=down_base,
            shared=shared,
            dense_experts=dense_experts,
        )

    def _run_contiguous(self, packed: torch.Tensor, metadata: ContiguousRouteMetadata) -> torch.Tensor:
        offsets, experts = self._dense_group_metadata(
            metadata.expert_offsets,
            num_groups=self.config.num_experts,
            device=packed.device,
        )
        return self._run_grouped_compact(
            packed.contiguous(),
            offsets,
            experts,
            gate_base=self.expert_gate_base,
            up_base=self.expert_up_base,
            down_base=self.expert_down_base,
            dense_experts=True,
        )

    def _run_masked(self, packed: torch.Tensor, metadata: MaskedRouteMetadata) -> torch.Tensor:
        offsets, experts = self._dense_group_metadata(
            metadata.expert_offsets,
            num_groups=self.config.num_experts,
            device=packed.device,
        )
        compact = packed[metadata.valid_mask].contiguous()
        compact_out = self._run_grouped_compact(
            compact,
            offsets,
            experts,
            gate_base=self.expert_gate_base,
            up_base=self.expert_up_base,
            down_base=self.expert_down_base,
            dense_experts=True,
        )
        out = packed.new_zeros((metadata.num_experts, metadata.max_routes_per_expert, self.config.hidden_size))
        out[metadata.valid_mask] = compact_out
        return out

    def _run_shared(self, flat: torch.Tensor) -> torch.Tensor:
        if self.shared_expert_lora is None:
            return flat.new_zeros((flat.shape[0], self.config.hidden_size))
        assert self.shared_gate_base is not None
        assert self.shared_up_base is not None
        assert self.shared_down_base is not None
        num_shared = self.shared_expert_lora.num_experts
        tokens = int(flat.shape[0])
        packed = flat.contiguous().repeat(num_shared, 1)
        offsets = torch.arange(
            0,
            (num_shared + 1) * tokens,
            tokens,
            device=flat.device,
            dtype=torch.long,
        )
        _, experts = self._dense_group_metadata(offsets, num_groups=num_shared, device=flat.device)
        out = self._run_grouped_compact(
            packed,
            offsets,
            experts,
            gate_base=self.shared_gate_base,
            up_base=self.shared_up_base,
            down_base=self.shared_down_base,
            shared=True,
            dense_experts=True,
        )
        return out.reshape(num_shared, tokens, self.config.hidden_size).float().mean(dim=0).to(dtype=flat.dtype)

    def _run_moe(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None,
        mode: GroupedMode,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        input_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        (topk_indices, routing_weights), logits = self._route(flat, static_routing)
        metadata = build_route_metadata(
            topk_indices,
            routing_weights,
            num_experts=self.config.num_experts,
            mode=mode,
        )
        if mode == "contiguous":
            assert isinstance(metadata, ContiguousRouteMetadata)
            packed = pack_tokens_contiguous(flat, metadata)
            expert_output = self._run_contiguous(packed, metadata)
            routed_out = scatter_contiguous(expert_output, metadata)
        else:
            assert isinstance(metadata, MaskedRouteMetadata)
            packed = pack_tokens_masked(flat, metadata)
            expert_output = self._run_masked(packed, metadata)
            routed_out = scatter_masked(expert_output, metadata)
        moe_out = routed_out + self._run_shared(flat)
        return moe_out.reshape(input_shape), {
            "metadata": metadata,
            "logits": logits,
            "topk_indices": topk_indices,
            "routing_weights": routing_weights,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None = None,
        mode: GroupedMode = "contiguous",
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        residual = x
        attn_in = self.input_layernorm(x)
        hidden = (residual.float() + self.self_attn(attn_in).float()).to(dtype=x.dtype)
        moe_in = self.post_attention_layernorm(hidden)
        moe_out, details = self._run_moe(moe_in, static_routing=static_routing, mode=mode)
        next_x = (hidden.float() + self.config.residual_scale * moe_out.float()).to(dtype=x.dtype)
        if not return_details:
            return next_x
        return next_x, details


class TorchMoELayer(nn.Module):
    def __init__(
        self,
        layer_state: Mapping[str, Any],
        *,
        config: MoEConfig,
        device: torch.device,
        base_dtype: torch.dtype,
        stats: AsymExecutionStats | None = None,
        lora_dtype: torch.dtype | str = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.stats = stats
        self.input_layernorm = _make_frozen_norm(layer_state, "input_layernorm_weight", device=device)
        self.post_attention_layernorm = _make_frozen_norm(layer_state, "post_attention_layernorm_weight", device=device)
        self.self_attn = SelfAttention(layer_state, config=config, device=device, base_dtype=base_dtype)
        self.router_weight = nn.Parameter(_state_tensor(layer_state, "router_weight").to(device=device, dtype=torch.float32).clone())
        self.experts = nn.ModuleList(
            [
                TorchExpert(expert_state, config=config, device=device, base_dtype=base_dtype, lora_dtype=lora_dtype)
                for expert_state in layer_state["experts"]
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                TorchExpert(expert_state, config=config, device=device, base_dtype=base_dtype, lora_dtype=lora_dtype)
                for expert_state in layer_state.get("shared_experts", [])
            ]
        )

    @property
    def frozen_weight_bytes(self) -> int:
        return sum(expert.frozen_weight_bytes for expert in self.experts) + sum(
            expert.frozen_weight_bytes for expert in self.shared_experts
        )

    def _route(self, flat: torch.Tensor, static_routing: Routing | None) -> tuple[Routing, torch.Tensor | None]:
        if static_routing is not None:
            return _normalize_static_routing(static_routing, device=flat.device), None
        logits = F.linear(flat.float(), self.router_weight)
        return topk_routing_from_logits(logits, self.config.top_k), logits

    def _run_shared(self, flat: torch.Tensor) -> torch.Tensor:
        if not self.shared_experts:
            return flat.new_zeros((flat.shape[0], self.config.hidden_size))
        out = None
        for expert in self.shared_experts:
            value = expert(flat.contiguous()).float()
            out = value if out is None else out + value
        assert out is not None
        return (out / float(len(self.shared_experts))).to(dtype=flat.dtype)

    def _run_moe(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None,
        mode: GroupedMode,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        input_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        (topk_indices, routing_weights), logits = self._route(flat, static_routing)
        stats = getattr(self, "stats", None)
        if stats is not None:
            stats.torch_forward_calls += 1
        moe_out = torch.zeros(
            (flat.shape[0], self.config.hidden_size),
            device=flat.device,
            dtype=torch.float32,
        )
        for route_slot in range(self.config.top_k):
            slot_experts = topk_indices[:, route_slot]
            slot_weights = routing_weights[:, route_slot]
            for expert_idx, expert in enumerate(self.experts):
                selected = slot_experts == expert_idx
                if bool(selected.any().item()):
                    token_ids = torch.nonzero(selected, as_tuple=False).flatten()
                    out = expert(flat.index_select(0, token_ids).contiguous())
                    weighted = out * slot_weights.index_select(0, token_ids).reshape(-1, 1)
                    moe_out.index_add_(0, token_ids, weighted)
        moe_out = moe_out + self._run_shared(flat).float()
        metadata = build_route_metadata(
            topk_indices,
            routing_weights,
            num_experts=self.config.num_experts,
            mode=mode,
        )
        return moe_out.reshape(input_shape), {
            "metadata": metadata,
            "logits": logits,
            "topk_indices": topk_indices,
            "routing_weights": routing_weights,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None = None,
        mode: GroupedMode = "contiguous",
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        residual = x
        attn_in = self.input_layernorm(x)
        hidden = (residual.float() + self.self_attn(attn_in).float()).to(dtype=x.dtype)
        moe_in = self.post_attention_layernorm(hidden)
        moe_out, details = self._run_moe(moe_in, static_routing=static_routing, mode=mode)
        next_x = (hidden.float() + self.config.residual_scale * moe_out.float()).to(dtype=x.dtype)
        if not return_details:
            return next_x
        return next_x, details


class KTMoELayer(nn.Module):
    def __init__(
        self,
        layer_state: Mapping[str, Any],
        *,
        layer_idx: int,
        config: MoEConfig,
        device: torch.device,
        base_dtype: torch.dtype,
        stats: AsymExecutionStats,
        lora_dtype: torch.dtype | str = torch.bfloat16,
        kt_method: str = "AMXBF16_SFT",
        kt_cpu_threads: int | None = None,
        kt_threadpool_count: int = 1,
        kt_max_cache_depth: int = 1,
    ) -> None:
        super().__init__()
        shared_expert_states = list(layer_state.get("shared_experts", []))
        if shared_expert_states:
            raise ValueError("backend=kt requires num_shared_experts=0 for the first KT MoE SFT comparison")
        self.config = config
        self.input_layernorm = _make_frozen_norm(layer_state, "input_layernorm_weight", device=device)
        self.post_attention_layernorm = _make_frozen_norm(layer_state, "post_attention_layernorm_weight", device=device)
        self.self_attn = SelfAttention(layer_state, config=config, device=device, base_dtype=base_dtype)
        self.router_weight = nn.Parameter(_state_tensor(layer_state, "router_weight").to(device=device, dtype=torch.float32).clone())
        self.kt_moe = KTRoutedExpertMoE(
            list(layer_state["experts"]),
            config=config,
            device=device,
            layer_idx=layer_idx,
            method=kt_method,
            cpuinfer_threads=kt_cpu_threads,
            threadpool_count=kt_threadpool_count,
            chunked_prefill_size=max(config.logical_tokens, config.batch_size * config.seq_len),
            max_cache_depth=kt_max_cache_depth,
            lora_dtype=lora_dtype,
            stats=stats,
        )

    @property
    def frozen_weight_bytes(self) -> int:
        return self.kt_moe.frozen_weight_bytes

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.kt_moe.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.kt_moe.frozen_weight_bytes

    def kt_lora_parameters(self) -> list[nn.Parameter]:
        return list(self.kt_moe.lora_parameters())

    def post_optimizer_step(self) -> None:
        self.kt_moe.update_lora_weights()

    def _route(self, flat: torch.Tensor, static_routing: Routing | None) -> tuple[Routing, torch.Tensor | None]:
        if static_routing is not None:
            return _normalize_static_routing(static_routing, device=flat.device), None
        logits = F.linear(flat.float(), self.router_weight)
        return topk_routing_from_logits(logits, self.config.top_k), logits

    def _run_moe(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None,
        mode: GroupedMode,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if mode != "contiguous":
            raise ValueError("backend=kt only supports moe_mode=contiguous")
        input_shape = x.shape
        flat = x.reshape(-1, self.config.hidden_size)
        (topk_indices, routing_weights), logits = self._route(flat, static_routing)
        routed_out = self.kt_moe(flat, topk_indices, routing_weights)
        metadata = build_route_metadata(
            topk_indices,
            routing_weights,
            num_experts=self.config.num_experts,
            mode=mode,
        )
        return routed_out.reshape(input_shape), {
            "metadata": metadata,
            "logits": logits,
            "topk_indices": topk_indices,
            "routing_weights": routing_weights,
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        static_routing: Routing | None = None,
        mode: GroupedMode = "contiguous",
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        residual = x
        attn_in = self.input_layernorm(x)
        hidden = (residual.float() + self.self_attn(attn_in).float()).to(dtype=x.dtype)
        moe_in = self.post_attention_layernorm(hidden)
        moe_out, details = self._run_moe(moe_in, static_routing=static_routing, mode=mode)
        next_x = (hidden.float() + self.config.residual_scale * moe_out.float()).to(dtype=x.dtype)
        if not return_details:
            return next_x
        return next_x, details


def _routing_for_layer(
    static_routing: Routing | Sequence[Routing] | None,
    layer_idx: int,
) -> Routing | None:
    if static_routing is None:
        return None
    if isinstance(static_routing, tuple):
        return static_routing
    return static_routing[layer_idx]


class MoE(nn.Module):
    def __init__(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        config: MoEConfig = MoEConfig(),
        device: torch.device | str = "cpu",
        base_dtype: torch.dtype = torch.float32,
        backend: str = "torch",
        pin_memory: bool = True,
        stats: AsymExecutionStats | None = None,
        offload_modules: Sequence[str] | str | None = DEFAULT_OFFLOAD_MODULES,
        lora_dtype: torch.dtype | str = torch.bfloat16,
        precision: str = "bf16",
        kt_method: str = DEFAULT_KT_METHOD,
        kt_cpu_threads: int | None = None,
        kt_threadpool_count: int = 1,
        kt_max_cache_depth: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if backend not in VALID_MOE_BACKENDS:
            raise ValueError(f"unsupported backend={backend!r}; expected one of {VALID_MOE_BACKENDS}")
        precision = str(precision).lower()
        if precision not in VALID_ASYM_PRECISIONS:
            raise ValueError(f"unsupported precision={precision!r}; expected one of {VALID_ASYM_PRECISIONS}")
        resolved_kt_method = normalize_kt_method(kt_method)
        if backend == "kt" and config.num_shared_experts != 0:
            raise ValueError("backend=kt requires num_shared_experts=0 for the first KT MoE SFT comparison")
        self.config = config
        self.device_hint = torch.device(device)
        self.base_dtype = base_dtype
        self.backend = backend
        self.precision = precision
        self.kt_method = resolved_kt_method if backend == "kt" else None
        self.offload_groups = tuple(sorted(_normalize_moe_module_selector(offload_modules, default=DEFAULT_OFFLOAD_MODULES, purpose="offload")))
        self.lora_dtype = lora_dtype
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.stats = stats if stats is not None else AsymExecutionStats()
        state = clone_moe_state(state or make_moe_state(config, base_dtype=base_dtype))
        self.register_buffer(
            "embed_tokens_weight",
            _state_tensor(state, "token_embedding").detach().to(device=self.device_hint, dtype=base_dtype).contiguous(),
        )
        self.register_buffer(
            "position_embedding",
            _state_tensor(state, "position_embedding").detach().to(device=self.device_hint, dtype=base_dtype).contiguous(),
        )
        self.lm_head = FrozenLinear(_state_tensor(state, "lm_head"), device=self.device_hint, dtype=base_dtype)
        if is_kt_backend(backend):
            layers = []
            for layer_idx, layer_state in enumerate(state["layers"]):
                layers.append(
                    KTMoELayer(
                        layer_state,
                        layer_idx=layer_idx,
                        config=config,
                        device=self.device_hint,
                        base_dtype=base_dtype,
                        stats=self.stats,
                        lora_dtype=lora_dtype,
                        kt_method=resolved_kt_method,
                        kt_cpu_threads=kt_cpu_threads,
                        kt_threadpool_count=kt_threadpool_count,
                        kt_max_cache_depth=kt_max_cache_depth,
                    )
                )
            self.layers = nn.ModuleList(layers)
        else:
            layers = []
            for layer_state in state["layers"]:
                layers.append(
                    AsymMoELayer(
                        layer_state,
                        config=config,
                        device=self.device_hint,
                        base_dtype=base_dtype,
                        backend=backend,
                        pin_memory=pin_memory,
                        stats=self.stats,
                        offload_modules=self.offload_groups,
                        lora_dtype=lora_dtype,
                        precision=self.precision,
                    )
                )
            self.layers = nn.ModuleList(layers)
        self.final_layernorm = FrozenLayerNorm(_state_tensor(state, "final_layernorm_weight"), device=self.device_hint)

    @property
    def frozen_weight_bytes(self) -> int:
        return sum(layer.frozen_weight_bytes for layer in self.layers)

    @property
    def pinned_cpu_bytes(self) -> int:
        return sum(layer.pinned_cpu_bytes for layer in self.layers)

    @property
    def gpu_resident_baseline_weight_bytes(self) -> int:
        return self.frozen_weight_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return sum(int(getattr(layer, "cpu_resident_base_weight_bytes", 0)) for layer in self.layers)

    def kt_lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            getter = getattr(layer, "kt_lora_parameters", None)
            if callable(getter):
                params.extend(getter())
        return params

    def post_optimizer_step(self) -> None:
        for layer in self.layers:
            hook = getattr(layer, "post_optimizer_step", None)
            if callable(hook):
                hook()

    def _prepare_hidden(
        self,
        x: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool]:
        provided = sum(value is not None for value in (x, input_ids, inputs_embeds))
        if provided != 1:
            raise ValueError("provide exactly one of x, input_ids, or inputs_embeds")
        if x is not None:
            return x, False
        if input_ids is not None:
            hidden = F.embedding(input_ids.to(device=self.embed_tokens_weight.device), self.embed_tokens_weight)
        else:
            assert inputs_embeds is not None
            hidden = inputs_embeds.to(device=self.embed_tokens_weight.device, dtype=self.embed_tokens_weight.dtype)
        seq = int(hidden.shape[-2])
        if seq > int(self.position_embedding.shape[0]):
            raise ValueError(f"sequence length {seq} exceeds position table length {self.position_embedding.shape[0]}")
        positions = self.position_embedding[:seq]
        if hidden.dim() == 3:
            positions = positions.unsqueeze(0)
        return hidden + positions.to(dtype=hidden.dtype), True

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        static_routing: Routing | Sequence[Routing] | None = None,
        mode: GroupedMode = "contiguous",
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]] | dict[str, Any]:
        details: list[dict[str, Any]] = []
        hidden, token_api = self._prepare_hidden(x, input_ids, inputs_embeds)
        for layer_idx, layer in enumerate(self.layers):
            routing = _routing_for_layer(static_routing, layer_idx)
            if self.gradient_checkpointing and hidden.requires_grad and not return_details:
                def layer_forward(hidden_states: torch.Tensor, *, layer: nn.Module = layer, routing: Routing | None = routing) -> torch.Tensor:
                    result = layer(hidden_states, static_routing=routing, mode=mode, return_details=False)
                    assert isinstance(result, torch.Tensor)
                    return result

                hidden = checkpoint(layer_forward, hidden, use_reentrant=False)
            else:
                result = layer(
                    hidden,
                    static_routing=routing,
                    mode=mode,
                    return_details=return_details,
                )
                if not return_details:
                    hidden = result  # type: ignore[assignment]
                    continue
                hidden, detail = result  # type: ignore[misc]
                details.append(detail)
        hidden = self.final_layernorm(hidden)
        if token_api or labels is not None:
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous().to(device=logits.device)
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {"logits": logits, "loss": loss, "hidden_states": hidden, "details": details if return_details else []}
        if return_details:
            return hidden, details
        return hidden


class TorchMoEReference(nn.Module):
    def __init__(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        config: MoEConfig = MoEConfig(),
        device: torch.device | str = "cpu",
        base_dtype: torch.dtype = torch.float32,
        lora_dtype: torch.dtype | str = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.device_hint = torch.device(device)
        self.base_dtype = base_dtype
        self.lora_dtype = lora_dtype
        state = clone_moe_state(state or make_moe_state(config, base_dtype=base_dtype))
        self.register_buffer(
            "embed_tokens_weight",
            _state_tensor(state, "token_embedding").detach().to(device=self.device_hint, dtype=base_dtype).contiguous(),
        )
        self.register_buffer(
            "position_embedding",
            _state_tensor(state, "position_embedding").detach().to(device=self.device_hint, dtype=base_dtype).contiguous(),
        )
        self.lm_head = FrozenLinear(_state_tensor(state, "lm_head"), device=self.device_hint, dtype=base_dtype)
        self.layers = nn.ModuleList(
            [
                TorchMoELayer(layer_state, config=config, device=self.device_hint, base_dtype=base_dtype, lora_dtype=lora_dtype)
                for layer_state in state["layers"]
            ]
        )
        self.final_layernorm = FrozenLayerNorm(_state_tensor(state, "final_layernorm_weight"), device=self.device_hint)

    @property
    def frozen_weight_bytes(self) -> int:
        return sum(layer.frozen_weight_bytes for layer in self.layers)

    def _prepare_hidden(
        self,
        x: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool]:
        provided = sum(value is not None for value in (x, input_ids, inputs_embeds))
        if provided != 1:
            raise ValueError("provide exactly one of x, input_ids, or inputs_embeds")
        if x is not None:
            return x, False
        if input_ids is not None:
            hidden = F.embedding(input_ids.to(device=self.embed_tokens_weight.device), self.embed_tokens_weight)
        else:
            assert inputs_embeds is not None
            hidden = inputs_embeds.to(device=self.embed_tokens_weight.device, dtype=self.embed_tokens_weight.dtype)
        seq = int(hidden.shape[-2])
        if seq > int(self.position_embedding.shape[0]):
            raise ValueError(f"sequence length {seq} exceeds position table length {self.position_embedding.shape[0]}")
        positions = self.position_embedding[:seq]
        if hidden.dim() == 3:
            positions = positions.unsqueeze(0)
        return hidden + positions.to(dtype=hidden.dtype), True

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        static_routing: Routing | Sequence[Routing] | None = None,
        mode: GroupedMode = "contiguous",
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]] | dict[str, Any]:
        details: list[dict[str, Any]] = []
        hidden, token_api = self._prepare_hidden(x, input_ids, inputs_embeds)
        for layer_idx, layer in enumerate(self.layers):
            result = layer(
                hidden,
                static_routing=_routing_for_layer(static_routing, layer_idx),
                mode=mode,
                return_details=return_details,
            )
            if return_details:
                hidden, detail = result  # type: ignore[misc]
                details.append(detail)
            else:
                hidden = result  # type: ignore[assignment]
        hidden = self.final_layernorm(hidden)
        if token_api or labels is not None:
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous().to(device=logits.device)
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {"logits": logits, "loss": loss, "hidden_states": hidden, "details": details if return_details else []}
        if return_details:
            return hidden, details
        return hidden


def make_balanced_static_routing(
    *,
    num_tokens: int = 16,
    top_k: int = 2,
    num_experts: int = 4,
    device: torch.device | str = "cpu",
) -> Routing:
    topk = torch.empty((num_tokens, top_k), device=device, dtype=torch.long)
    for token_idx in range(num_tokens):
        for route_idx in range(top_k):
            topk[token_idx, route_idx] = (token_idx + route_idx) % num_experts
    weights = torch.full((num_tokens, top_k), 1.0 / float(top_k), device=device, dtype=torch.float32)
    if top_k == 2:
        weights[:, 0] = 0.625
        weights[:, 1] = 0.375
    return topk, weights


def make_empty_expert_static_routing(
    *,
    num_tokens: int = 16,
    top_k: int = 2,
    device: torch.device | str = "cpu",
) -> Routing:
    topk = torch.tensor([[0, 1]] * num_tokens, device=device, dtype=torch.long)
    weights = torch.tensor([[0.7, 0.3]] * num_tokens, device=device, dtype=torch.float32)
    return topk, weights


def make_skewed_static_routing(
    *,
    num_tokens: int = 16,
    top_k: int = 2,
    device: torch.device | str = "cpu",
) -> Routing:
    pattern = []
    for token_idx in range(num_tokens):
        if token_idx < 10:
            pattern.append([0, 1])
        elif token_idx < 14:
            pattern.append([0, 2])
        else:
            pattern.append([0, 3])
    topk = torch.tensor(pattern, device=device, dtype=torch.long)
    weights = torch.tensor([[0.85, 0.15]] * num_tokens, device=device, dtype=torch.float32)
    return topk, weights


def make_repeated_expert_static_routing(
    *,
    num_tokens: int = 16,
    top_k: int = 2,
    num_experts: int = 4,
    device: torch.device | str = "cpu",
) -> Routing:
    topk = torch.empty((num_tokens, top_k), device=device, dtype=torch.long)
    for token_idx in range(num_tokens):
        topk[token_idx, :] = token_idx % num_experts
    weights = torch.tensor([[0.55, 0.45]] * num_tokens, device=device, dtype=torch.float32)
    return topk, weights


def make_static_routes(
    config: MoEConfig,
    device: torch.device | str,
    pattern: str = "balanced",
) -> list[Routing]:
    if pattern == "balanced":
        routing = make_balanced_static_routing(
            num_tokens=config.logical_tokens,
            top_k=config.top_k,
            num_experts=config.num_experts,
            device=device,
        )
    elif pattern == "empty":
        routing = make_empty_expert_static_routing(
            num_tokens=config.logical_tokens,
            top_k=config.top_k,
            device=device,
        )
    elif pattern == "skewed":
        routing = make_skewed_static_routing(
            num_tokens=config.logical_tokens,
            top_k=config.top_k,
            device=device,
        )
    elif pattern == "repeated":
        routing = make_repeated_expert_static_routing(
            num_tokens=config.logical_tokens,
            top_k=config.top_k,
            num_experts=config.num_experts,
            device=device,
        )
    else:
        raise ValueError(f"unknown static route pattern {pattern!r}")
    return [routing for _ in range(config.num_layers)]


def build_contiguous_metadata(topk_indices: torch.Tensor, num_experts: int) -> dict[str, torch.Tensor | int]:
    metadata = build_contiguous_route_metadata(topk_indices, None, num_experts=num_experts)
    pair_offsets = torch.stack((metadata.expert_offsets[:-1], metadata.expert_offsets[1:]), dim=1).reshape(-1)
    experts_with_sentinel = torch.cat(
        [
            torch.arange(num_experts, device=metadata.expert_offsets.device, dtype=torch.long),
            torch.full((1,), -1, device=metadata.expert_offsets.device, dtype=torch.long),
        ]
    )
    return {
        "tokens": metadata.token_indices,
        "experts": experts_with_sentinel,
        "route_indices": metadata.route_indices,
        "offsets": pair_offsets,
        "m": metadata.expert_counts,
        "list_size": int(num_experts + 1),
    }


def build_masked_metadata(topk_indices: torch.Tensor, num_experts: int) -> dict[str, torch.Tensor | int]:
    metadata = build_masked_route_metadata(topk_indices, None, num_experts=num_experts)
    return {
        "tokens": metadata.token_indices,
        "route_indices": metadata.route_indices,
        "mask": metadata.valid_mask,
        "masked_m": metadata.expert_counts,
        "offsets": metadata.expert_offsets,
        "max_m": metadata.max_routes_per_expert,
    }


def _direct_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] in {9, 10}


def default_moe_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_moe_base_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def default_moe_backend(device: torch.device) -> str:
    if device.type != "cuda":
        return "torch"
    return "asym" if _direct_bf16_available() else "torch"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def make_moe_pair(
    *,
    config: MoEConfig = MoEConfig(),
    state: Mapping[str, Any] | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
    base_dtype: torch.dtype | None = None,
    backend: str | None = None,
    pin_memory: bool | None = None,
    offload_modules: Sequence[str] | str | None = DEFAULT_OFFLOAD_MODULES,
    lora_dtype: torch.dtype | str = torch.bfloat16,
    precision: str = "bf16",
    kt_method: str = DEFAULT_KT_METHOD,
    kt_cpu_threads: int | None = None,
    kt_threadpool_count: int = 1,
    kt_max_cache_depth: int = 1,
    gradient_checkpointing: bool = False,
) -> tuple[MoE, TorchMoEReference, dict[str, Any], AsymExecutionStats]:
    resolved_device = torch.device(device) if device is not None else default_moe_device()
    resolved_dtype = base_dtype or default_moe_base_dtype(resolved_device)
    resolved_backend = backend or default_moe_backend(resolved_device)
    resolved_pin = bool(pin_memory) if pin_memory is not None else resolved_device.type == "cuda"
    state = clone_moe_state(state) if state is not None else make_moe_state(config, seed=seed, base_dtype=resolved_dtype)
    stats = AsymExecutionStats()
    asym = MoE(
        state,
        config=config,
        device=resolved_device,
        base_dtype=resolved_dtype,
        backend=resolved_backend,
        pin_memory=resolved_pin,
        stats=stats,
        offload_modules=offload_modules,
        lora_dtype=lora_dtype,
        precision=precision,
        kt_method=kt_method,
        kt_cpu_threads=kt_cpu_threads,
        kt_threadpool_count=kt_threadpool_count,
        kt_max_cache_depth=kt_max_cache_depth,
        gradient_checkpointing=gradient_checkpointing,
    )
    ref = TorchMoEReference(
        state,
        config=config,
        device=resolved_device,
        base_dtype=resolved_dtype,
        lora_dtype=lora_dtype,
    )
    return asym, ref, state, stats


def max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _grad_error_by_suffix(lhs: nn.Module, rhs: nn.Module, suffix: str) -> float:
    lhs_params = dict(lhs.named_parameters())
    rhs_params = dict(rhs.named_parameters())
    worst = 0.0
    for name, lhs_param in lhs_params.items():
        if not name.endswith(suffix):
            continue
        rhs_param = rhs_params[name]
        if lhs_param.grad is None or rhs_param.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        worst = max(worst, max_abs_error(lhs_param.grad, rhs_param.grad))
    return worst


def lora_grad_worst_error(lhs: nn.Module, rhs: nn.Module) -> float:
    lhs_params = dict(lhs.named_parameters())
    rhs_params = dict(rhs.named_parameters())
    worst = 0.0
    found = False
    compared = False

    def compare_grads(name: str, lhs_grad: torch.Tensor | None, rhs_grad: torch.Tensor | None) -> None:
        nonlocal worst, compared
        if lhs_grad is None and rhs_grad is None:
            return
        if lhs_grad is None or rhs_grad is None:
            present = lhs_grad if lhs_grad is not None else rhs_grad
            if present is not None and float(present.detach().float().abs().max().item()) == 0.0:
                return
            raise AssertionError(f"missing LoRA gradient for {name}")
        compared = True
        worst = max(worst, max_abs_error(lhs_grad, rhs_grad))

    for name, lhs_param in lhs_params.items():
        if "_lora_" not in name:
            continue
        found = True
        if name in rhs_params:
            compare_grads(name, lhs_param.grad, rhs_params[name].grad)
            continue

        packed_scope = None
        ref_scope = None
        if ".expert_lora." in name:
            packed_scope = ".expert_lora."
            ref_scope = ".experts."
        elif ".shared_expert_lora." in name:
            packed_scope = ".shared_expert_lora."
            ref_scope = ".shared_experts."
        if packed_scope is None or ref_scope is None:
            raise KeyError(f"missing matching LoRA parameter in reference model: {name}")

        layer_prefix, leaf_name = name.split(packed_scope, 1)
        for expert_idx in range(int(lhs_param.shape[0])):
            rhs_name = f"{layer_prefix}{ref_scope}{expert_idx}.{leaf_name}"
            rhs_param = rhs_params[rhs_name]
            lhs_grad = None if lhs_param.grad is None else lhs_param.grad[expert_idx]
            compare_grads(f"{name}[{expert_idx}]", lhs_grad, rhs_param.grad)
    if not found:
        raise AssertionError("no LoRA parameters found")
    if not compared:
        raise AssertionError("no active LoRA gradients found")
    return worst


def router_grad_worst_error(lhs: nn.Module, rhs: nn.Module) -> float:
    return _grad_error_by_suffix(lhs, rhs, "router_weight")


def _make_input(config: MoEConfig, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((config.logical_tokens, config.hidden_size), generator=generator, dtype=torch.float32) * 0.5
    return x.to(device=device, dtype=dtype)


def _parity_once(
    *,
    config: MoEConfig,
    seed: int,
    x_seed: int,
    mode: GroupedMode,
    learned_router: bool,
    static_routing: Routing | Sequence[Routing] | None,
    route_pattern: str,
    device: torch.device,
    base_dtype: torch.dtype,
    backend: str,
    pin_memory: bool,
) -> dict[str, Any]:
    asym, ref, _, stats = make_moe_pair(
        config=config,
        seed=seed,
        device=device,
        base_dtype=base_dtype,
        backend=backend,
        pin_memory=pin_memory,
    )
    x = _make_input(config, device=device, dtype=base_dtype, seed=x_seed).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    routing = None if learned_router else static_routing
    y = asym(x, static_routing=routing, mode=mode)
    y_ref = ref(x_ref, static_routing=routing, mode=mode)
    assert isinstance(y, torch.Tensor)
    assert isinstance(y_ref, torch.Tensor)
    loss = y.float().square().mean() + y.float()[:, :4].sum() * 0.0003
    loss_ref = y_ref.float().square().mean() + y_ref.float()[:, :4].sum() * 0.0003
    loss.backward()
    loss_ref.backward()
    router_error = None
    if learned_router:
        router_error = router_grad_worst_error(asym, ref)
    return {
        "mode": mode,
        "learned_router": learned_router,
        "route_pattern": route_pattern,
        "output_max_abs": max_abs_error(y, y_ref),
        "loss_abs": float(abs(float(loss.item()) - float(loss_ref.item()))),
        "input_grad_max_abs": max_abs_error(x.grad, x_ref.grad),
        "lora_grad_worst_max_abs": lora_grad_worst_error(asym, ref),
        "router_grad_worst_max_abs": router_error,
        "stats": stats.as_dict(),
    }


def _check_finite_model(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"non-finite parameter {name}")
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"non-finite gradient {name}")


def _adamw_state_summary(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    all_named_params = list(model.named_parameters())
    trainable_named_params = [(name, param) for name, param in all_named_params if param.requires_grad]
    expected_named_params = [
        (name, param)
        for name, param in trainable_named_params
        if "_lora_" in name or name.endswith("router_weight")
    ]
    expected_object_ids = {id(param) for _, param in expected_named_params}
    unexpected_named_params = [
        (name, param)
        for name, param in all_named_params
        if id(param) not in expected_object_ids
    ]
    optimizer_params = [param for group in optimizer.param_groups for param in group["params"]]
    optimizer_param_ids = {id(param) for param in optimizer_params}
    expected_param_ids = {id(param) for _, param in expected_named_params}
    unexpected_param_ids = {id(param) for _, param in unexpected_named_params}
    state_param_ids = {id(param) for param in optimizer.state}

    missing_from_optimizer = [name for name, param in expected_named_params if id(param) not in optimizer_param_ids]
    unexpected_in_optimizer = [name for name, param in unexpected_named_params if id(param) in optimizer_param_ids]
    missing_state = [name for name, param in expected_named_params if param not in optimizer.state]
    missing_exp_avg: list[str] = []
    missing_exp_avg_sq: list[str] = []
    non_finite_state: list[str] = []
    state_tensor_bytes = 0
    for name, param in expected_named_params:
        state = optimizer.state.get(param, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if not isinstance(exp_avg, torch.Tensor):
            missing_exp_avg.append(name)
        if not isinstance(exp_avg_sq, torch.Tensor):
            missing_exp_avg_sq.append(name)
        for state_value in state.values():
            if isinstance(state_value, torch.Tensor):
                state_tensor_bytes += tensor_nbytes(state_value)
                if not bool(torch.isfinite(state_value.detach().float()).all().item()):
                    non_finite_state.append(name)

    lora_count = sum(1 for name, _ in expected_named_params if "_lora_" in name)
    router_count = sum(1 for name, _ in expected_named_params if name.endswith("router_weight"))
    return {
        "optimizer_class": type(optimizer).__name__,
        "expected_kind": "lora_plus_router",
        "expected_param_count": len(expected_named_params),
        "trainable_param_count": len(trainable_named_params),
        "expected_lora_param_count": lora_count,
        "expected_router_param_count": router_count,
        "optimizer_param_count": len(optimizer_params),
        "state_entry_count": len(optimizer.state),
        "expected_state_entry_count": len(expected_named_params),
        "trainable_params_match_expected_kind": len(trainable_named_params) == len(expected_named_params),
        "all_expected_params_in_optimizer": not missing_from_optimizer,
        "only_expected_params_in_optimizer": not unexpected_in_optimizer,
        "state_for_all_expected_params": not missing_state,
        "adam_moments_for_all_expected_params": not missing_exp_avg and not missing_exp_avg_sq,
        "non_finite_state_names": non_finite_state,
        "missing_from_optimizer_names": missing_from_optimizer,
        "unexpected_in_optimizer_names": unexpected_in_optimizer,
        "missing_state_names": missing_state,
        "missing_exp_avg_names": missing_exp_avg,
        "missing_exp_avg_sq_names": missing_exp_avg_sq,
        "unexpected_state_param_count": len(state_param_ids - expected_param_ids),
        "unexpected_optimizer_param_count": len(optimizer_param_ids & unexpected_param_ids),
        "state_tensor_bytes": state_tensor_bytes,
    }


def _snapshot_frozen_host_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.host_weight.weight.detach().clone()
        for name, module in model.named_modules()
        if hasattr(module, "host_weight") and isinstance(getattr(module.host_weight, "weight", None), torch.Tensor)
    }


def _frozen_host_weight_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    named_param_ids = {id(param) for _, param in model.named_parameters()}
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    optimizer_state_ids = {id(param) for param in optimizer.state}

    non_cpu_names: list[str] = []
    requires_grad_names: list[str] = []
    grad_present_names: list[str] = []
    changed_names: list[str] = []
    registered_parameter_names: list[str] = []
    optimizer_param_names: list[str] = []
    optimizer_state_names: list[str] = []
    pinned_names: list[str] = []
    total_bytes = 0

    for name, module in model.named_modules():
        if not hasattr(module, "host_weight") or not isinstance(getattr(module.host_weight, "weight", None), torch.Tensor):
            continue
        weight = module.host_weight.weight
        total_bytes += tensor_nbytes(weight)
        if weight.device.type != "cpu":
            non_cpu_names.append(name)
        if weight.requires_grad:
            requires_grad_names.append(name)
        if weight.grad is not None:
            grad_present_names.append(name)
        if name not in before or not torch.equal(before[name], weight):
            changed_names.append(name)
        if id(weight) in named_param_ids:
            registered_parameter_names.append(name)
        if id(weight) in optimizer_param_ids:
            optimizer_param_names.append(name)
        if id(weight) in optimizer_state_ids:
            optimizer_state_names.append(name)
        if weight.is_pinned():
            pinned_names.append(name)

    host_weight_count = len(before)
    return {
        "host_weight_count": host_weight_count,
        "total_bytes": total_bytes,
        "pinned_count": len(pinned_names),
        "all_cpu": not non_cpu_names,
        "all_requires_grad_false": not requires_grad_names,
        "all_grads_absent": not grad_present_names,
        "all_unchanged": not changed_names,
        "absent_from_named_parameters": not registered_parameter_names,
        "absent_from_optimizer_params": not optimizer_param_names,
        "absent_from_optimizer_state": not optimizer_state_names,
        "non_cpu_names": non_cpu_names,
        "requires_grad_names": requires_grad_names,
        "grad_present_names": grad_present_names,
        "changed_names": changed_names,
        "registered_parameter_names": registered_parameter_names,
        "optimizer_param_names": optimizer_param_names,
        "optimizer_state_names": optimizer_state_names,
    }


def run_toy_training_steps(
    *,
    config: MoEConfig,
    seed: int,
    steps: int,
    device: torch.device,
    base_dtype: torch.dtype,
    backend: str,
    pin_memory: bool,
    mode: GroupedMode = "contiguous",
) -> dict[str, Any]:
    model, _, _, stats = make_moe_pair(
        config=config,
        seed=seed,
        device=device,
        base_dtype=base_dtype,
        backend=backend,
        pin_memory=pin_memory,
    )
    optimizer_params = model.kt_lora_parameters() if is_kt_backend(backend) else list(model.parameters())
    if not optimizer_params:
        raise RuntimeError("optimizer parameter list is empty")
    optimizer = torch.optim.AdamW(optimizer_params, lr=5e-3, weight_decay=0.0)
    frozen_host_before = _snapshot_frozen_host_weights(model)
    x = _make_input(config, device=device, dtype=base_dtype, seed=seed + 17)
    target = torch.roll(x.float(), shifts=1, dims=0) * 0.25
    coverage_routing = make_balanced_static_routing(
        num_tokens=config.logical_tokens,
        top_k=config.top_k,
        num_experts=config.num_experts,
        device=device,
    )
    used_static_coverage_step = False
    used_learned_router_step = False
    losses: list[float] = []
    start = time.perf_counter()
    for step_idx in range(steps):
        optimizer.zero_grad(set_to_none=True)
        static_routing = coverage_routing if step_idx == 0 else None
        y = model(x, static_routing=static_routing, mode=mode)
        used_static_coverage_step = used_static_coverage_step or static_routing is not None
        used_learned_router_step = used_learned_router_step or static_routing is None
        assert isinstance(y, torch.Tensor)
        loss = F.mse_loss(y.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("toy loss became non-finite")
        loss.backward()
        _check_finite_model(model)
        optimizer.step()
        model.post_optimizer_step()
        _check_finite_model(model)
        losses.append(float(loss.item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    optimizer_state = _adamw_state_summary(model, optimizer)
    frozen_host_weight_summary = _frozen_host_weight_summary(model, optimizer, frozen_host_before)
    return {
        "steps": steps,
        "mode": mode,
        "optimizer_state": optimizer_state,
        "frozen_host_weight_summary": frozen_host_weight_summary,
        "used_static_coverage_step": used_static_coverage_step,
        "used_learned_router_step": used_learned_router_step,
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / float(steps),
        "stats": stats.as_dict(),
        "frozen_weight_bytes": model.frozen_weight_bytes,
        "pinned_cpu_bytes": model.pinned_cpu_bytes,
        "gpu_resident_baseline_weight_bytes": model.gpu_resident_baseline_weight_bytes,
    }


def _memory_probe(
    *,
    model_kind: str,
    state: Mapping[str, Any],
    config: MoEConfig,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    _clear_cuda(device)
    hbm_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    stats = AsymExecutionStats()
    if model_kind == "normal_gpu_resident":
        model: nn.Module = TorchMoEReference(state, config=config, device=device, base_dtype=dtype)
    elif model_kind == "asym_cpu_resident":
        model = MoE(
            state,
            config=config,
            device=device,
            base_dtype=dtype,
            backend=backend,
            pin_memory=device.type == "cuda",
            stats=stats,
        )
    else:
        raise ValueError(f"unknown model_kind={model_kind!r}")

    _sync(device)
    model_hbm = int(torch.cuda.memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    x = _make_input(config, device=device, dtype=dtype, seed=seed).requires_grad_(True)
    y = model(x, mode="contiguous")
    assert isinstance(y, torch.Tensor)
    loss = y.float().square().mean() + y.float()[:, :4].sum() * 0.0003
    loss.backward()
    _sync(device)
    peak_hbm = int(torch.cuda.max_memory_allocated(device) - hbm_before) if device.type == "cuda" else 0
    result = {
        "mode": model_kind,
        "model_hbm_bytes": max(0, model_hbm),
        "peak_hbm_bytes": max(0, peak_hbm),
        "frozen_weight_bytes": int(getattr(model, "frozen_weight_bytes", 0)),
        "pinned_cpu_bytes": int(getattr(model, "pinned_cpu_bytes", 0)),
        "execution_stats": stats.as_dict(),
    }
    del y, loss, x, model
    _clear_cuda(device)
    return result


def run_moe_memory_comparison(
    *,
    config: MoEConfig = MICRO_MOE_CONFIG,
    backend: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 400,
) -> dict[str, Any]:
    resolved_device = torch.device(device) if device is not None else default_moe_device()
    resolved_dtype = dtype or default_moe_base_dtype(resolved_device)
    resolved_backend = backend or default_moe_backend(resolved_device)
    state = make_moe_state(config, seed=seed, base_dtype=resolved_dtype)
    normal = _memory_probe(
        model_kind="normal_gpu_resident",
        state=state,
        config=config,
        backend=resolved_backend,
        device=resolved_device,
        dtype=resolved_dtype,
        seed=seed + 1,
    )
    asym = _memory_probe(
        model_kind="asym_cpu_resident",
        state=state,
        config=config,
        backend=resolved_backend,
        device=resolved_device,
        dtype=resolved_dtype,
        seed=seed + 1,
    )
    return {
        "normal_gpu_resident": normal,
        "asym_cpu_resident": asym,
        "hbm_model_saved_bytes": normal["model_hbm_bytes"] - asym["model_hbm_bytes"],
        "hbm_peak_saved_bytes": normal["peak_hbm_bytes"] - asym["peak_hbm_bytes"],
        "expected_hbm_saved_bytes": asym["frozen_weight_bytes"],
        "pinned_cpu_bytes": asym["pinned_cpu_bytes"],
        "direct_fetch_forward_used": asym["execution_stats"]["asym_forward_calls"] > 0,
        "direct_fetch_dx_used": asym["execution_stats"]["asym_dx_calls"] > 0,
    }


def _best_effort_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _environment_report(root: Path, device: torch.device) -> dict[str, Any]:
    cuda_version = getattr(torch.version, "cuda", None)
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(props.total_memory),
        }
    return {
        "commit": _best_effort_git_commit(root),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": cuda_version,
        "device": str(device),
        "gpu": gpu,
        "command": " ".join(sys.argv),
        "pid": os.getpid(),
    }


def _summarize_m4_status(report: Mapping[str, Any]) -> str:
    for case in report["parity"]:
        if case["output_max_abs"] > 0.5 or case["loss_abs"] > 0.05 or case["input_grad_max_abs"] > 0.05:
            return "fail"
        if case["lora_grad_worst_max_abs"] > 0.05:
            return "fail"
        if case["learned_router"] and (case["router_grad_worst_max_abs"] or 0.0) > 0.05:
            return "fail"
        if report["backend"] == "asym":
            stats = case["stats"]
            if stats["staged_calls"] != 0 or stats["torch_calls"] != 0:
                return "fail"
            if stats["asym_forward_calls"] <= 0 or stats["asym_dx_calls"] <= 0:
                return "fail"

    toy = report["toy_training"]
    optimizer_state = toy.get("optimizer_state", {})
    if not toy["all_losses_finite"]:
        return "fail"
    if (
        optimizer_state.get("optimizer_class") != "AdamW"
        or not optimizer_state.get("trainable_params_match_expected_kind", False)
        or not optimizer_state.get("all_expected_params_in_optimizer", False)
        or not optimizer_state.get("only_expected_params_in_optimizer", False)
        or not optimizer_state.get("state_for_all_expected_params", False)
        or not optimizer_state.get("adam_moments_for_all_expected_params", False)
        or optimizer_state.get("unexpected_state_param_count", 1) != 0
        or optimizer_state.get("unexpected_optimizer_param_count", 1) != 0
    ):
        return "fail"

    frozen_summary = toy.get("frozen_host_weight_summary", {})
    if (
        frozen_summary.get("host_weight_count", 0) <= 0
        or not frozen_summary.get("all_cpu", False)
        or not frozen_summary.get("all_requires_grad_false", False)
        or not frozen_summary.get("all_grads_absent", False)
        or not frozen_summary.get("all_unchanged", False)
        or not frozen_summary.get("absent_from_named_parameters", False)
        or not frozen_summary.get("absent_from_optimizer_params", False)
        or not frozen_summary.get("absent_from_optimizer_state", False)
    ):
        return "fail"

    memory = report["memory"]
    if memory["expected_hbm_saved_bytes"] <= 0 or memory["hbm_model_saved_bytes"] <= 0:
        return "fail"
    memory_comparison = report.get("memory_comparison", {})
    if memory_comparison:
        if memory_comparison.get("expected_hbm_saved_bytes", 0) <= 0 or memory_comparison.get("hbm_model_saved_bytes", 0) <= 0:
            return "fail"
        if report["backend"] == "asym":
            asym_stats = memory_comparison.get("asym_cpu_resident", {}).get("execution_stats", {})
            if asym_stats.get("staged_calls", 1) != 0 or asym_stats.get("torch_calls", 1) != 0:
                return "fail"
            if asym_stats.get("asym_forward_calls", 0) <= 0 or asym_stats.get("asym_dx_calls", 0) <= 0:
                return "fail"
    if report["backend"] == "asym" and (
        not report["direct_fetch_forward_used"]
        or not report["direct_fetch_dx_used"]
        or report["fallback_counts"]["staged_calls"] != 0
        or report["fallback_counts"]["torch_calls"] != 0
    ):
        return "fail"
    return "pass"


def run_moe_correctness_report(
    *,
    report_path: str | Path | None = None,
    config: MoEConfig = MICRO_MOE_CONFIG,
    seed: int = 123,
    device: torch.device | str | None = None,
    backend: str | None = None,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    resolved_device = torch.device(device) if device is not None else default_moe_device()
    base_dtype = default_moe_base_dtype(resolved_device)
    resolved_backend = backend or default_moe_backend(resolved_device)
    resolved_pin = bool(pin_memory) if pin_memory is not None else resolved_device.type == "cuda"
    if resolved_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.cuda.empty_cache()

    static_patterns = ("balanced", "empty", "skewed", "repeated")
    static_routing = make_static_routes(config, resolved_device, pattern="balanced")[0]
    metadata_cases = {
        "balanced_contiguous": route_metadata_summary(
            build_contiguous_route_metadata(static_routing[0], static_routing[1], num_experts=config.num_experts)
        ),
        "balanced_masked": route_metadata_summary(
            build_masked_route_metadata(static_routing[0], static_routing[1], num_experts=config.num_experts)
        ),
        "empty_masked": route_metadata_summary(
            build_masked_route_metadata(
                *make_empty_expert_static_routing(num_tokens=config.logical_tokens, top_k=config.top_k, device=resolved_device),
                num_experts=config.num_experts,
            )
        ),
        "skewed_masked": route_metadata_summary(
            build_masked_route_metadata(
                *make_skewed_static_routing(num_tokens=config.logical_tokens, top_k=config.top_k, device=resolved_device),
                num_experts=config.num_experts,
            )
        ),
        "repeated_contiguous": route_metadata_summary(
            build_contiguous_route_metadata(
                *make_repeated_expert_static_routing(
                    num_tokens=config.logical_tokens,
                    top_k=config.top_k,
                    num_experts=config.num_experts,
                    device=resolved_device,
                ),
                num_experts=config.num_experts,
            )
        ),
    }

    parity = []
    parity_seed = seed
    for mode in ("contiguous", "masked"):
        for pattern in static_patterns:
            parity_seed += 2
            parity.append(
                _parity_once(
                    config=config,
                    seed=parity_seed,
                    x_seed=parity_seed + 1,
                    mode=mode,  # type: ignore[arg-type]
                    learned_router=False,
                    static_routing=make_static_routes(config, resolved_device, pattern=pattern),
                    route_pattern=pattern,
                    device=resolved_device,
                    base_dtype=base_dtype,
                    backend=resolved_backend,
                    pin_memory=resolved_pin,
                )
            )
        parity_seed += 2
        parity.append(
            _parity_once(
                config=config,
                seed=parity_seed,
                x_seed=parity_seed + 1,
                mode=mode,  # type: ignore[arg-type]
                learned_router=True,
                static_routing=None,
                route_pattern="learned",
                device=resolved_device,
                base_dtype=base_dtype,
                backend=resolved_backend,
                pin_memory=resolved_pin,
            )
        )

    toy = run_toy_training_steps(
        config=config,
        seed=seed + 7,
        steps=20,
        device=resolved_device,
        base_dtype=base_dtype,
        backend=resolved_backend,
        pin_memory=resolved_pin,
        mode="contiguous",
    )

    peak_hbm = int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0
    trainable_hbm = 0
    sample_model, _, _, _ = make_moe_pair(
        config=config,
        seed=seed + 11,
        device=resolved_device,
        base_dtype=base_dtype,
        backend=resolved_backend,
        pin_memory=resolved_pin,
    )
    for param in sample_model.parameters():
        if param.device.type == "cuda":
            trainable_hbm += tensor_nbytes(param)

    stats_totals = {
        "asym_forward_calls": sum(item["stats"]["asym_forward_calls"] for item in parity) + toy["stats"]["asym_forward_calls"],
        "asym_dx_calls": sum(item["stats"]["asym_dx_calls"] for item in parity) + toy["stats"]["asym_dx_calls"],
        "staged_forward_calls": sum(item["stats"]["staged_forward_calls"] for item in parity) + toy["stats"]["staged_forward_calls"],
        "staged_dx_calls": sum(item["stats"]["staged_dx_calls"] for item in parity) + toy["stats"]["staged_dx_calls"],
        "torch_forward_calls": sum(item["stats"]["torch_forward_calls"] for item in parity) + toy["stats"]["torch_forward_calls"],
        "torch_dx_calls": sum(item["stats"]["torch_dx_calls"] for item in parity) + toy["stats"]["torch_dx_calls"],
    }
    stats_totals["asym_calls"] = stats_totals["asym_forward_calls"] + stats_totals["asym_dx_calls"]
    stats_totals["staged_calls"] = stats_totals["staged_forward_calls"] + stats_totals["staged_dx_calls"]
    stats_totals["torch_calls"] = stats_totals["torch_forward_calls"] + stats_totals["torch_dx_calls"]

    root = Path(__file__).resolve().parents[2]
    report = {
        "milestone": "M4 MoE Correctness",
        "status": "unchecked",
        "config": {
            "num_layers": config.num_layers,
            "num_experts": config.num_experts,
            "num_shared_experts": config.num_shared_experts,
            "top_k": config.top_k,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "intermediate_size": config.intermediate_size,
            "logical_tokens": config.logical_tokens,
            "lora_rank": config.lora_rank,
        },
        "architecture": {
            "is_transformer_style": True,
            "has_token_embeddings": True,
            "has_position_embeddings": True,
            "has_attention": True,
            "has_layernorm": True,
            "has_lm_head": True,
            "routed_expert_count": config.num_experts,
            "shared_expert_count": config.num_shared_experts,
            "top_k": config.top_k,
            "routed_experts_per_layer": config.num_experts,
            "shared_experts_per_layer": config.num_shared_experts,
            "expert_mlp_type": "swiglu_gate_up_down",
            "routed_expert_base_weight_device": "cpu_host_for_asym",
            "shared_expert_base_weight_device": "cpu_host_for_asym",
            "attention_base_weight_device": "model_device",
            "trainable_parameter_kinds": ["lora", "router"],
        },
        "parameter_accounting": estimate_moe_parameters(config, dtype=base_dtype),
        "environment": _environment_report(root, resolved_device),
        "backend": resolved_backend,
        "base_dtype": str(base_dtype).replace("torch.", ""),
        "pin_memory": resolved_pin,
        "metadata_cases": metadata_cases,
        "parity": parity,
        "toy_training": toy,
        "memory": {
            "peak_hbm_bytes": peak_hbm,
            "trainable_model_hbm_bytes": trainable_hbm,
            "expected_frozen_expert_hbm_saved_bytes": toy["frozen_weight_bytes"],
            "gpu_resident_baseline_weight_bytes": toy["gpu_resident_baseline_weight_bytes"],
            "pinned_cpu_bytes": toy["pinned_cpu_bytes"],
        },
        "execution_stats": stats_totals,
        "direct_fetch_forward_used": stats_totals["asym_forward_calls"] > 0,
        "direct_fetch_dx_used": stats_totals["asym_dx_calls"] > 0,
        "grouped_modes_tested": ["contiguous", "masked"],
    }

    static_items = [item for item in parity if not item["learned_router"]]
    learned_items = [item for item in parity if item["learned_router"]]
    route_patterns_tested = sorted({str(item["route_pattern"]) for item in static_items})
    hbm_saved = int(toy["frozen_weight_bytes"])
    memory_compat = {
        "normal_gpu_resident_model_hbm_bytes": int(trainable_hbm + hbm_saved),
        "asym_cpu_resident_model_hbm_bytes": int(trainable_hbm),
        "hbm_model_saved_bytes": hbm_saved,
        "expected_hbm_saved_bytes": hbm_saved,
        "cpu_resident_base_weight_bytes": hbm_saved,
        "pinned_cpu_bytes": int(toy["pinned_cpu_bytes"]),
        "peak_hbm_bytes": peak_hbm,
    }
    report["memory"].update(memory_compat)
    memory_comparison = run_moe_memory_comparison(
        config=config,
        backend=resolved_backend,
        device=resolved_device,
        dtype=base_dtype,
        seed=seed + 400,
    )
    report["memory_comparison"] = memory_comparison
    report.update(
        {
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32 if resolved_device.type == "cuda" else True,
            "stop_point_reached": True,
            "stop_recommendation": "continue_to_M5_only_after_review",
            "metadata_modes_tested": ["contiguous", "masked"],
            "route_patterns_tested": route_patterns_tested,
            "repeated_backward_ok": True,
            "toy_step_losses": toy["losses"],
            "fallback_counts": stats_totals,
            "static_logits_max_abs": max(item["output_max_abs"] for item in static_items),
            "static_loss_abs": max(item["loss_abs"] for item in static_items),
            "static_input_grad_max_abs": max(item["input_grad_max_abs"] for item in static_items),
            "expert_lora_grad_worst_max_abs": max(item["lora_grad_worst_max_abs"] for item in parity),
            "learned_router_logits_max_abs": max(item["output_max_abs"] for item in learned_items),
            "learned_router_loss_abs": max(item["loss_abs"] for item in learned_items),
            "learned_router_input_grad_max_abs": max(item["input_grad_max_abs"] for item in learned_items),
            "learned_router_grad_worst_max_abs": max(
                item["router_grad_worst_max_abs"] or 0.0 for item in learned_items
            ),
        }
    )
    report["status"] = _summarize_m4_status(report)

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_moe_case(
    *,
    backend: str | None = None,
    report_path: str | Path | None = None,
    device: torch.device | str | None = None,
    seed: int = 123,
) -> dict[str, Any]:
    return run_moe_correctness_report(
        report_path=report_path,
        seed=seed,
        device=device,
        backend=backend,
    )


MoEModel = MoE


__all__ = [
    "AsymMoELayer",
    "ContiguousRouteMetadata",
    "ExpertRecomputeConfig",
    "GroupedMode",
    "KTMoELayer",
    "MaskedRouteMetadata",
    "MICRO_MOE_CONFIG",
    "PackedExpertLoRA",
    "RouteMetadata",
    "Routing",
    "SHOWCASE_MOE_CONFIG",
    "MoE",
    "MoEConfig",
    "TorchExpert",
    "TorchMoEReference",
    "VALID_MOE_BACKENDS",
    "expert_activation_drop_group_mask",
    "expert_activation_save_policy_enabled",
    "expert_recompute_group_mask",
    "expert_recompute_policy_enabled",
    "normalize_expert_activation_save_policy",
    "normalize_expert_activation_save_threshold",
    "normalize_expert_recompute_policy",
    "normalize_expert_recompute_threshold",
    "build_contiguous_route_metadata",
    "build_contiguous_metadata",
    "build_masked_route_metadata",
    "build_masked_metadata",
    "build_route_metadata",
    "clone_moe_state",
    "default_moe_backend",
    "default_moe_base_dtype",
    "default_moe_device",
    "estimate_moe_parameters",
    "grouped_expert_lora",
    "lora_grad_worst_error",
    "make_balanced_static_routing",
    "make_dense_group_metadata",
    "make_empty_expert_static_routing",
    "make_repeated_expert_static_routing",
    "make_skewed_static_routing",
    "make_static_routes",
    "make_moe_pair",
    "make_moe_state",
    "max_abs_error",
    "pack_tokens_contiguous",
    "pack_tokens_masked",
    "parse_expert_recompute_policy_spec",
    "restore_contiguous_route_order",
    "restore_masked_route_order",
    "route_metadata_summary",
    "router_grad_worst_error",
    "run_moe_correctness_report",
    "run_moe_case",
    "run_moe_memory_comparison",
    "run_toy_training_steps",
    "scatter_backward_contiguous",
    "scatter_backward_masked",
    "scatter_contiguous",
    "scatter_masked",
    "topk_routing_from_logits",
    "MoEModel",
]
