from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import AsymExecutionStats
from .profile_ranges import prof_range, scoped_name
from .qwen3_moe import AsymQwen3Experts, AsymQwen3Router, is_qwen3_experts, wrap_qwen3_experts


@dataclass(frozen=True)
class Qwen35MoeReport:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cpu_resident_base_bytes: int
    gpu_resident_base_bytes: int
    trainable_lora_params: int
    expert_recompute_policy: str = "none"


def _is_qwen35_family(module: nn.Module) -> bool:
    class_name = type(module).__name__.lower().replace("_", "")
    module_name = type(module).__module__.lower()
    if "qwen35" in class_name:
        return True
    if "qwen3_5_moe" in module_name:
        return True

    for candidate in (
        getattr(module, "config", None),
        getattr(getattr(module, "experts", None), "config", None),
        getattr(getattr(module, "gate", None), "config", None),
        getattr(getattr(module, "shared_expert", None), "config", None),
    ):
        config_type = str(getattr(candidate, "model_type", "")).lower()
        if config_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
            return True
    return False


def is_qwen35_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_qwen35_moe_block", False):
        return False
    if not _is_qwen35_family(module):
        return False

    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    shared_expert = getattr(module, "shared_expert", None)
    shared_expert_gate = getattr(module, "shared_expert_gate", None)

    if not isinstance(gate, nn.Module):
        return False
    if not is_qwen3_experts(experts):
        return False
    if not isinstance(shared_expert, nn.Module):
        return False
    if not isinstance(shared_expert_gate, nn.Module):
        return False

    for attr in ("hidden_dim", "top_k", "num_experts"):
        if not isinstance(getattr(gate, attr, None), int):
            return False

    if not callable(getattr(gate, "forward", None)):
        return False
    if not callable(getattr(shared_expert, "forward", None)):
        return False
    if not callable(getattr(shared_expert_gate, "forward", None)):
        return False

    if hasattr(shared_expert_gate, "in_features") and int(shared_expert_gate.in_features) != int(gate.hidden_dim):
        return False
    if hasattr(shared_expert_gate, "out_features") and int(shared_expert_gate.out_features) != 1:
        return False
    return True


class AsymQwen35MoeBlock(nn.Module):
    """Qwen3.5 MoE block wrapper preserving the shared expert branch."""

    _is_asym_qwen35_moe_block = True

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
        offload_router: bool = False,
        router_debug_grad: bool = False,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if router_mode != "whole":
            raise ValueError(f"AsymQwen35MoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_qwen35_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a Qwen3.5 MoE block with gate/experts/shared_expert: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        source_gate = getattr(source, "gate")
        self.gate = (
            AsymQwen3Router(source_gate, backend=backend, precision=precision, stats=stats, strict=strict)
            if backend == "asym" and offload_router
            else source_gate
        )
        self.experts: AsymQwen3Experts = wrap_qwen3_experts(
            getattr(source, "experts"),
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
        self.shared_expert = getattr(source, "shared_expert")
        self.shared_expert_gate = getattr(source, "shared_expert_gate")

        self.hidden_dim = int(getattr(self.gate, "hidden_dim"))
        self.top_k = int(getattr(self.gate, "top_k"))
        self.num_experts = int(getattr(self.gate, "num_experts"))
        self.gate.requires_grad_(False)

    @property
    def cpu_resident_base_bytes(self) -> int:
        gate_bytes = int(getattr(getattr(self.gate, "proj", None), "cpu_resident_base_weight_bytes", 0))
        return int(self.experts.cpu_resident_base_bytes) + gate_bytes

    @property
    def gpu_resident_base_bytes(self) -> int:
        gate_bytes = int(getattr(getattr(self.gate, "proj", None), "gpu_resident_base_weight_bytes", 0))
        return int(self.experts.gpu_resident_base_bytes) + gate_bytes

    @property
    def trainable_lora_params(self) -> int:
        return int(self.experts.trainable_lora_params)

    def report(self) -> Qwen35MoeReport:
        return Qwen35MoeReport(
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

    def _compute_routing(self, hidden_states_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_out = self.gate(hidden_states_2d)

        if isinstance(router_out, tuple) and len(router_out) >= 3:
            router_logits = router_out[0]
            top_k_weights = router_out[1]
            top_k_index = router_out[2]
            if not self.router_debug_grad:
                router_logits = router_logits.detach()
                top_k_weights = top_k_weights.detach()
                top_k_index = top_k_index.detach()
            if top_k_weights.dtype != hidden_states_2d.dtype:
                top_k_weights = top_k_weights.to(dtype=hidden_states_2d.dtype)
            return top_k_index, top_k_weights, router_logits

        raise TypeError(
            "AsymQwen35MoeBlock requires a Qwen3_5MoeTopKRouter-style gate returning "
            "(router_logits, top_k_weights, top_k_index); "
            f"got {type(router_out).__name__}"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymQwen35MoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")

        flat = hidden_states.view(-1, input_shape[-1])
        with prof_range(self._forward_range("shared_expert")):
            shared = self.shared_expert(flat)

        top_k_index, top_k_weights, _router_logits = self._compute_routing(flat)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")

        with prof_range(self._forward_range("experts")):
            routed = self.experts(flat, top_k_index, top_k_weights)

        with prof_range(self._forward_range("shared_expert_gate")):
            shared = F.sigmoid(self.shared_expert_gate(flat)) * shared

        out = routed + shared
        return out.reshape(input_shape)


def wrap_qwen35_moe_block(
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
    offload_router: bool = False,
    router_debug_grad: bool = False,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymQwen35MoeBlock:
    return AsymQwen35MoeBlock(
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
        offload_router=offload_router,
        router_debug_grad=router_debug_grad,
        stats=stats,
        strict=strict,
    )


__all__ = [
    "AsymQwen35MoeBlock",
    "Qwen35MoeReport",
    "is_qwen35_moe_block",
    "wrap_qwen35_moe_block",
]
