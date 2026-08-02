"""gpt-oss family wrapper (model_integration.md, family #6 of 6 — the hard one).

REPLICATED code path (campaign rule). gpt-oss is the ONE family whose experts
cannot ride the shared `AsymQwen3Experts` engine: transposed packing
(`gate_up_proj [E, H, 2I]`, `down_proj [E, I, H]`), per-expert BIASES on both
projections, INTERLEAVED gate/up (`[..., ::2]/[..., 1::2]`), and a clamped GLU
(`gate·σ(1.702·gate)`, ±limit clamps, `(up+1)·glu`) instead of silu·mul — the
engine's elementwise core and its silu-specific backward kernels do not apply.

First-pass engine here (`AsymGptOssExperts`) — capacity-real, correctness-first:
- all four base banks live PINNED on host (adopted from the HF module, GPU
  copies freed); per-expert weights are small (H×2I ≈ 33 MB at 120B), so
  streaming per active expert is cheap;
- each active expert's compute runs under non-reentrant
  `torch.utils.checkpoint`: the base weights are fetched (no-grad) INSIDE the
  checkpointed fn, so backward re-streams them transiently instead of autograd
  retaining them — one expert's weights resident at a time;
- trainable grouped LoRA (peft-style init: kaiming A, zero B) on gate_up and
  down; LoRA-B learns the model-native interleaved output layout directly;
- no fused/tuned grouped kernels yet — cuBLAS GEMMs on streamed weights
  (T1-class behavior). Tuned kernels are follow-up work, not this pass.

Router (`GptOssTopKRouter`) already returns the standard
(logits, top_k_weights, top_k_index) triple → block wrapper mirrors the other
families; NOTE the HF block returns a `(hidden, router_scores)` TUPLE and the
decoder unpacks it — the wrapper preserves that contract exactly.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import math
from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .frozen_linear import AsymExecutionStats
from .profile_ranges import prof_range, scoped_name


def _is_param_like(module: nn.Module, name: str, dims: int) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, torch.Tensor) and value.dim() == dims


def is_gptoss_experts(module: nn.Module) -> bool:
    if "gptoss" not in type(module).__name__.lower():
        return False
    if not (_is_param_like(module, "gate_up_proj", 3) and _is_param_like(module, "down_proj", 3)):
        return False
    if not (_is_param_like(module, "gate_up_proj_bias", 2) and _is_param_like(module, "down_proj_bias", 2)):
        return False
    for attr in ("num_experts", "hidden_size", "intermediate_size"):
        if not isinstance(getattr(module, attr, None), int):
            return False
    for attr in ("alpha", "limit"):
        if not isinstance(getattr(module, attr, None), (int, float)):
            return False
    return callable(getattr(module, "forward", None))


def is_gptoss_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_gptoss_moe_block", False):
        return False
    if "gptoss" not in type(module).__name__.lower():
        return False
    router = getattr(module, "router", None)
    experts = getattr(module, "experts", None)
    if not isinstance(router, nn.Module) or not is_gptoss_experts(experts):
        return False
    for attr in ("top_k", "num_experts", "hidden_dim"):
        if not isinstance(getattr(router, attr, None), int):
            return False
    return callable(getattr(router, "forward", None))


class AsymGptOssExperts(nn.Module):
    """gpt-oss packed experts: pinned host base banks + streamed checkpointed compute."""

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
    ) -> None:
        super().__init__()
        if precision != "bf16":
            raise ValueError("AsymGptOssExperts first pass supports bf16 only")
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")

        self.num_experts = int(source.num_experts)
        self.hidden_dim = int(source.hidden_size)
        self.intermediate_dim = int(source.intermediate_size)
        self.alpha = float(source.alpha)
        self.limit = float(source.limit)
        self.backend = backend
        self.offload = bool(offload)
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.profile_prefix = "layers.unknown.mlp.experts"

        base_dtype = torch.bfloat16
        pin = torch.cuda.is_available() and self.offload

        def _adopt(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().to(dtype=base_dtype)
            if self.offload:
                t = t.to("cpu")
                if pin and not t.is_pinned():
                    t = t.contiguous().pin_memory()
            return t.contiguous()

        # Plain python attrs (NOT registered): module.to()/state_dict must not
        # touch the host banks; residency is reported via cpu_resident_base_bytes.
        self._gate_up = _adopt(source.gate_up_proj)          # [E, H, 2I]
        self._gate_up_bias = _adopt(source.gate_up_proj_bias)  # [E, 2I]
        self._down = _adopt(source.down_proj)                # [E, I, H]
        self._down_bias = _adopt(source.down_proj_bias)      # [E, H]

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0.0 else nn.Identity()
        device = source.gate_up_proj.device if not self.offload else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        e, h, i, r = self.num_experts, self.hidden_dim, self.intermediate_dim, self.lora_rank
        self.lora_A_gate_up = nn.Parameter(torch.empty(e, h, r, dtype=base_dtype, device=device))
        self.lora_B_gate_up = nn.Parameter(torch.zeros(e, r, 2 * i, dtype=base_dtype, device=device))
        self.lora_A_down = nn.Parameter(torch.empty(e, i, r, dtype=base_dtype, device=device))
        self.lora_B_down = nn.Parameter(torch.zeros(e, r, h, dtype=base_dtype, device=device))
        for a in (self.lora_A_gate_up, self.lora_A_down):
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    @property
    def cpu_resident_base_bytes(self) -> int:
        if not self.offload:
            return 0
        return sum(
            t.numel() * t.element_size()
            for t in (self._gate_up, self._gate_up_bias, self._down, self._down_bias)
        )

    @property
    def gpu_resident_base_bytes(self) -> int:
        if self.offload:
            return 0
        return sum(
            t.numel() * t.element_size()
            for t in (self._gate_up, self._gate_up_bias, self._down, self._down_bias)
        )

    @property
    def trainable_lora_params(self) -> int:
        return sum(
            p.numel()
            for p in (self.lora_A_gate_up, self.lora_B_gate_up, self.lora_A_down, self.lora_B_down)
        )

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Verbatim GptOssExperts._apply_gate: interleaved split, clamps, α-sigmoid GLU.
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1) * glu

    def _expert_fn(self, expert_idx: int):
        def fn(current_state: torch.Tensor) -> torch.Tensor:
            device = current_state.device
            with torch.no_grad():
                w_gu = self._gate_up[expert_idx].to(device, non_blocking=True)
                b_gu = self._gate_up_bias[expert_idx].to(device, non_blocking=True)
                w_dn = self._down[expert_idx].to(device, non_blocking=True)
                b_dn = self._down_bias[expert_idx].to(device, non_blocking=True)
            gate_up = current_state @ w_gu + b_gu
            if self.lora_rank > 0:
                dropped = self.lora_dropout(current_state)
                gate_up = gate_up + (dropped @ self.lora_A_gate_up[expert_idx]) @ self.lora_B_gate_up[
                    expert_idx
                ] * self.lora_scale
            activated = self._apply_gate(gate_up)
            out = activated @ w_dn + b_dn
            if self.lora_rank > 0:
                dropped_act = self.lora_dropout(activated)
                out = out + (dropped_act @ self.lora_A_down[expert_idx]) @ self.lora_B_down[
                    expert_idx
                ] * self.lora_scale
            return out

        return fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        out = torch.zeros_like(hidden_states)
        needs_grad = torch.is_grad_enabled() and (
            hidden_states.requires_grad or self.lora_A_gate_up.requires_grad
        )
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            fn = self._expert_fn(expert_idx)
            if needs_grad:
                expert_out = checkpoint(fn, current_state, use_reentrant=False)
            else:
                expert_out = fn(current_state)
            weighted = expert_out * top_k_weights[token_idx, top_k_pos, None]
            out = out.index_add(0, token_idx, weighted.to(out.dtype))
        return out


class AsymGptOssMoeBlock(nn.Module):
    """gpt-oss MLP block wrapper; preserves the HF (hidden, router_scores) return."""

    _is_asym_gptoss_moe_block = True

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
            raise ValueError(f"AsymGptOssMoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_gptoss_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a GptOss MLP block with router/experts: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        self.router = getattr(source, "router")
        self.experts = AsymGptOssExperts(
            getattr(source, "experts"),
            backend=backend,
            precision=precision,
            offload=offload,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_dtype=lora_dtype,
            stats=stats,
        )

        self.hidden_dim = int(getattr(self.router, "hidden_dim"))
        self.top_k = int(getattr(self.router, "top_k"))
        self.num_experts = int(getattr(self.router, "num_experts"))
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

    def forward(self, hidden_states: torch.Tensor):
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymGptOssMoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        flat = hidden_states.reshape(-1, input_shape[-1])
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_out = self.router(flat)
        if not (isinstance(router_out, tuple) and len(router_out) >= 3):
            raise TypeError(
                "AsymGptOssMoeBlock requires a GptOssTopKRouter-style router returning "
                f"(router_logits, top_k_weights, top_k_index); got {type(router_out).__name__}"
            )
        router_scores = router_out[1]
        top_k_index = router_out[2]
        top_k_weights = router_scores
        if not self.router_debug_grad:
            top_k_weights = top_k_weights.detach()
            top_k_index = top_k_index.detach()
        if top_k_weights.dtype != flat.dtype:
            top_k_weights = top_k_weights.to(dtype=flat.dtype)
        with prof_range(self._forward_range("experts")):
            out = self.experts(flat, top_k_index, top_k_weights)
        # HF GptOssMLP returns (hidden, router_scores); the decoder unpacks the pair.
        return out.view(input_shape), router_scores


def wrap_gptoss_moe_block(
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
) -> AsymGptOssMoeBlock:
    return AsymGptOssMoeBlock(
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
    "AsymGptOssExperts",
    "AsymGptOssMoeBlock",
    "is_gptoss_experts",
    "is_gptoss_moe_block",
    "wrap_gptoss_moe_block",
]
