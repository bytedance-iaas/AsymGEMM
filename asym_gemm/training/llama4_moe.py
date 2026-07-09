from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .frozen_linear import AsymExecutionStats, AsymFrozenLinear
from .llama4_experts import AsymLlama4Experts
from .llama4_shared_mlp import AsymLlama4SharedMLP
from .offload import adopt_host_weight
from .packed_moe import PackedExpertSource, PackedMoELayout
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


def _drop_parameter_storage(module: nn.Module, *names: str) -> None:
    for name in names:
        value = getattr(module, name, None)
        if not isinstance(value, nn.Parameter):
            continue
        value.data = torch.empty(0, dtype=value.dtype, device=value.device)
        value.requires_grad_(False)


class AsymLlama4Router(nn.Module):
    """Llama 4 router with a CPU-resident frozen projection."""

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        weight = getattr(source, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            raise TypeError(f"AsymLlama4Router requires a 2D router weight, got {type(source).__name__}")
        if strict and weight.device.type != "cpu":
            raise RuntimeError("Llama 4 router CPU offload requires CPU-first model loading")
        if strict and weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Llama 4 router CPU offload requires bf16 source weight, got {weight.dtype}")
        host_weight = adopt_host_weight(
            "router.weight",
            weight,
            "router",
            require_2d=True,
            pin_memory_policy="auto",
            strict=strict,
        )
        bias = getattr(source, "bias", None)
        self.proj = AsymFrozenLinear.from_host_weight(
            host_weight,
            bias=None if bias is None else bias.detach(),
            backend=backend,
            stats=stats,
            precision=precision,
        )
        self.num_experts = int(getattr(source, "num_experts", self.proj.out_features))
        self.top_k = int(getattr(source, "top_k"))

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.proj.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.proj(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)
        return router_scores, router_logits


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
        router_mode: Literal["hf", "whole"] = "whole",
        offload_router: bool = False,
        wrap_shared_expert: bool = False,
        offload_shared_expert: bool = False,
        router_debug_grad: bool = False,
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
        if router_mode not in {"hf", "whole"}:
            raise ValueError(f"router_mode must be 'hf' or 'whole', got {router_mode!r}")

        source_experts = getattr(source, "experts")
        gate_up = getattr(source_experts, "gate_up_proj").detach()
        down = getattr(source_experts, "down_proj").detach()
        if backend == "asym" and offload and strict:
            if gate_up.device.type != "cpu" or down.device.type != "cpu":
                raise RuntimeError("Llama 4 expert CPU offload requires CPU-first model loading")
            if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
                raise RuntimeError(
                    "Llama 4 expert CPU offload requires bf16 source weights: "
                    f"gate_up={gate_up.dtype}, down={down.dtype}"
                )
        self.config = getattr(source, "config", None)
        self.top_k = int(getattr(source, "top_k"))
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.num_experts = int(getattr(source, "num_experts"))
        source_router = getattr(source, "router")
        self.router = (
            AsymLlama4Router(source_router, backend=backend, precision=precision, stats=stats, strict=strict)
            if backend == "asym" and offload_router
            else source_router
        )
        source_shared_expert = getattr(source, "shared_expert")
        self.shared_expert = (
            AsymLlama4SharedMLP(
                source_shared_expert,
                backend=backend,
                precision=precision,
                offload=offload_shared_expert,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_dtype=lora_dtype,
                stats=stats,
                strict=strict,
            )
            if wrap_shared_expert
            else source_shared_expert
        )
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.feed_forward"
        self.router.requires_grad_(False)

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
            gate_up_proj=gate_up,
            down_proj=down,
            layout=layout,
            act_fn=getattr(source_experts, "act_fn"),
            config=getattr(source_experts, "config", self.config),
        )
        self.experts = AsymLlama4Experts(
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
            strict=False,
        )
        if backend == "asym" and offload and strict and torch.cuda.is_available():
            def _hw_ok(hw) -> bool:  # fabric banks pin at seal() (fix_gb200_ep.md S1)
                return hw.weight.is_pinned() or bool(getattr(hw, "_fabric_bank", False))

            if not _hw_ok(self.experts.gate_up_base.host_weight) or not _hw_ok(self.experts.down_base.host_weight):
                raise RuntimeError("Llama 4 expert CPU offload requires pinned CPU HostWeights for AsymGEMM")
        if backend == "asym" and offload:
            _drop_parameter_storage(source_experts, "gate_up_proj", "down_proj")
        del packed_source, gate_up, down

    @property
    def cpu_resident_base_bytes(self) -> int:
        router_bytes = int(getattr(getattr(self.router, "proj", None), "cpu_resident_base_weight_bytes", 0))
        shared_bytes = int(getattr(self.shared_expert, "cpu_resident_base_bytes", 0))
        return int(self.experts.cpu_resident_base_bytes) + router_bytes + shared_bytes

    @property
    def gpu_resident_base_bytes(self) -> int:
        router_bytes = int(getattr(getattr(self.router, "proj", None), "gpu_resident_base_weight_bytes", 0))
        shared_bytes = int(getattr(self.shared_expert, "gpu_resident_base_bytes", 0))
        return int(self.experts.gpu_resident_base_bytes) + router_bytes + shared_bytes

    @property
    def trainable_lora_params(self) -> int:
        shared_params = int(getattr(self.shared_expert, "trainable_lora_params", 0))
        return int(self.experts.trainable_lora_params) + shared_params

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

    def _compute_routing(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        no_grad_router = self.router_mode == "whole" and not self.router_debug_grad
        context = torch.no_grad() if no_grad_router else nullcontext()
        with context, prof_range(self._forward_range("router")):
            router_scores, router_logits = self.router(flat)
            _top_values, top_k_index = torch.topk(router_logits, self.top_k, dim=1)
            input_weights = router_scores.gather(1, top_k_index)
        if no_grad_router:
            router_logits = router_logits.detach()
            top_k_index = top_k_index.detach()
            input_weights = input_weights.detach()
        # gb200_ep.md E0: natural-skew histogram collection (shared collector; llama4 site)
        from .qwen3_moe import _EP_STATS

        if _EP_STATS is not None:
            n_exp = getattr(self, "num_experts", None) or getattr(self.experts, "num_experts")
            _EP_STATS.record(self.profile_prefix, top_k_index, int(n_exp))
        return top_k_index, input_weights, router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymLlama4Moe expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        flat = hidden_states.view(-1, self.hidden_dim)
        top_k_index, input_weights, router_logits = self._compute_routing(flat)
        if self.router_mode == "whole" and not self.router_debug_grad and input_weights.requires_grad:
            raise RuntimeError("Llama 4 router no-grad mode produced differentiable input_weights")
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
    router_mode: Literal["hf", "whole"] = "whole",
    offload_router: bool = False,
    wrap_shared_expert: bool = False,
    offload_shared_expert: bool = False,
    router_debug_grad: bool = False,
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
        router_mode=router_mode,
        offload_router=offload_router,
        wrap_shared_expert=wrap_shared_expert,
        offload_shared_expert=offload_shared_expert,
        router_debug_grad=router_debug_grad,
        stats=stats,
        strict=strict,
    )


__all__ = ["AsymLlama4Moe", "AsymLlama4Router", "Llama4MoeReport", "is_llama4_moe", "wrap_llama4_moe"]
