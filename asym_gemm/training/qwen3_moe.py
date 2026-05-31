from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .frozen_linear import AsymExecutionStats, AsymGroupedFrozenLinear, TorchGroupedFrozenLinear
from .lora import grouped_expert_lora, grouped_expert_lora_pair, normalize_lora_dtype, prepare_grouped_lora_metadata
from .moe import (
    ExpertRecomputeConfig,
    build_contiguous_route_metadata,
    expert_activation_drop_group_mask,
    expert_recompute_group_mask,
    make_dense_group_metadata,
    pack_tokens_contiguous,
    parse_expert_recompute_policy_spec,
    scatter_contiguous,
)


@dataclass(frozen=True)
class Qwen3ExpertReport:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cpu_resident_base_bytes: int
    gpu_resident_base_bytes: int
    trainable_lora_params: int
    expert_recompute_policy: str = "none"


@dataclass(frozen=True)
class Qwen3ExpertSubset:
    packed: torch.Tensor
    offsets: torch.Tensor
    experts: torch.Tensor
    row_indices: torch.Tensor


def _is_3d_parameter(module: nn.Module, name: str) -> bool:
    value = getattr(module, name, None)
    return isinstance(value, nn.Parameter) and value.dim() == 3


def is_qwen3_experts(module: nn.Module) -> bool:
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


def _resolve_device(source: torch.Tensor) -> torch.device:
    return torch.device(source.device)


def _reset_lora_bank(weight: torch.Tensor, *, is_b: bool) -> None:
    if is_b:
        nn.init.zeros_(weight)
    else:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class AsymQwen3Experts(nn.Module):
    """Qwen3 packed MoE experts with frozen grouped bases and trainable LoRA."""

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
        init_lora_weights: Literal["peft"] = "peft",
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if strict and not is_qwen3_experts(source):
            raise TypeError(f"source does not look like a Qwen3 packed expert module: {type(source).__name__}")
        if backend not in {"asym", "torch"}:
            raise ValueError("backend must be 'asym' or 'torch'")
        if precision != "bf16":
            raise ValueError("AsymQwen3Experts first pass supports bf16 only")
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")
        if init_lora_weights != "peft":
            raise ValueError("AsymQwen3Experts first pass only supports PEFT-compatible LoRA init")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")

        gate_up = getattr(source, "gate_up_proj").detach()
        down = getattr(source, "down_proj").detach()
        self.num_experts = int(getattr(source, "num_experts"))
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.intermediate_dim = int(getattr(source, "intermediate_dim"))
        expected_gate_up = (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        expected_down = (self.num_experts, self.hidden_dim, self.intermediate_dim)
        if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
            raise ValueError(
                "unexpected Qwen3 expert shapes: "
                f"gate_up={tuple(gate_up.shape)} expected={expected_gate_up}, "
                f"down={tuple(down.shape)} expected={expected_down}"
            )

        self.config = getattr(source, "config", None)
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.act_fn = getattr(source, "act_fn")
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0.0 else nn.Identity()
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.expert_recompute_config = parse_expert_recompute_policy_spec(expert_recompute_policy)

        base_dtype = torch.bfloat16
        if backend == "asym" and self.offload:
            self.gate_up_base = AsymGroupedFrozenLinear(
                gate_up.to(dtype=base_dtype),
                backend="asym",
                precision=precision,
                stats=self.stats,
            )
            self.down_base = AsymGroupedFrozenLinear(
                down.to(dtype=base_dtype),
                backend="asym",
                precision=precision,
                stats=self.stats,
            )
        else:
            device = _resolve_device(gate_up)
            self.gate_up_base = TorchGroupedFrozenLinear(gate_up, device=device, dtype=base_dtype)
            self.down_base = TorchGroupedFrozenLinear(down, device=device, dtype=base_dtype)

        device = _resolve_device(gate_up)
        self.gate_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.gate_lora_B = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype)
        )
        self.up_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.up_lora_B = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.down_lora_A = nn.Parameter(
            torch.empty(self.num_experts, self.lora_rank, self.intermediate_dim, device=device, dtype=self.lora_dtype)
        )
        self.down_lora_B = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        with torch.no_grad():
            for weight in (self.gate_lora_A, self.up_lora_A, self.down_lora_A):
                _reset_lora_bank(weight, is_b=False)
            for weight in (self.gate_lora_B, self.up_lora_B, self.down_lora_B):
                _reset_lora_bank(weight, is_b=True)

    @property
    def cpu_resident_base_bytes(self) -> int:
        total = 0
        for module in (self.gate_up_base, self.down_base):
            if isinstance(module, AsymGroupedFrozenLinear):
                total += int(getattr(module, "weight_hbm_saved_bytes", 0))
        return total

    @property
    def gpu_resident_base_bytes(self) -> int:
        total = 0
        for module in (self.gate_up_base, self.down_base):
            total += int(getattr(module, "gpu_resident_weight_bytes", 0))
        return total

    @property
    def trainable_lora_params(self) -> int:
        return sum(param.numel() for name, param in self.named_parameters() if "lora_" in name)

    def report(self) -> Qwen3ExpertReport:
        return Qwen3ExpertReport(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            cpu_resident_base_bytes=self.cpu_resident_base_bytes,
            gpu_resident_base_bytes=self.gpu_resident_base_bytes,
            trainable_lora_params=self.trainable_lora_params,
            expert_recompute_policy=self.expert_recompute_config.label,
        )

    def _forward_gate_up_lora(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, metadata) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lora_dropout_p == 0.0:
            x_lora = x.to(dtype=self.lora_dtype)
            gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)
            low_rank = grouped_expert_lora(x_lora, gate_up_a, offsets, experts, metadata=metadata)
            gate_low_rank, up_low_rank = low_rank.split(self.lora_rank, dim=-1)
        else:
            gate_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            up_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            gate_low_rank = grouped_expert_lora(gate_input, self.gate_lora_A, offsets, experts, metadata=metadata)
            up_low_rank = grouped_expert_lora(up_input, self.up_lora_A, offsets, experts, metadata=metadata)
        gate_delta, up_delta = grouped_expert_lora_pair(
            gate_low_rank,
            up_low_rank,
            self.gate_lora_B,
            self.up_lora_B,
            offsets,
            experts,
            metadata=metadata,
        )
        if self.lora_scale != 1.0:
            gate_delta = gate_delta.mul(self.lora_scale)
            up_delta = up_delta.mul(self.lora_scale)
        return gate_delta, up_delta

    def _forward_down_lora(self, x: torch.Tensor, offsets: torch.Tensor, experts: torch.Tensor, metadata) -> torch.Tensor:
        x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
        low_rank = grouped_expert_lora(x_lora, self.down_lora_A, offsets, experts, metadata=metadata)
        delta = grouped_expert_lora(low_rank, self.down_lora_B, offsets, experts, metadata=metadata)
        if self.lora_scale != 1.0:
            delta = delta.mul(self.lora_scale)
        return delta

    def _lora_metadata(self, offsets: torch.Tensor, experts: torch.Tensor, *, dense_experts: bool):
        return prepare_grouped_lora_metadata(offsets, experts, dense_experts=dense_experts)

    def _forward_gate_up(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        lora_metadata = self._lora_metadata(offsets, experts, dense_experts=dense_experts)
        gate_up = self.gate_up_base(x, offsets, experts, dense_experts=dense_experts, profile_name="gate_up")
        gate, up = gate_up.chunk(2, dim=-1)
        gate_delta, up_delta = self._forward_gate_up_lora(x, offsets, experts, lora_metadata)
        gate = gate + gate_delta.to(dtype=gate.dtype)
        up = up + up_delta.to(dtype=up.dtype)
        return gate, up, lora_metadata

    def _forward_activation_down(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        lora_metadata,
        *,
        dense_experts: bool,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        activated = self.act_fn(gate) * up
        down = self.down_base(activated.to(dtype=input_dtype), offsets, experts, dense_experts=dense_experts, profile_name="down")
        down_delta = self._forward_down_lora(activated, offsets, experts, lora_metadata)
        return down + down_delta.to(dtype=down.dtype)

    def _forward_expert_body(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool,
    ) -> torch.Tensor:
        gate, up, lora_metadata = self._forward_gate_up(
            packed,
            offsets,
            experts,
            dense_experts=dense_experts,
        )
        return self._forward_activation_down(
            gate,
            up,
            offsets,
            experts,
            lora_metadata,
            dense_experts=dense_experts,
            input_dtype=packed.dtype,
        )

    def _checkpoint_preserve_rng_state(self) -> bool:
        return self.lora_dropout_p > 0.0

    def _uses_expert_recompute(self) -> bool:
        return bool(self.expert_recompute_config.enabled and self.training and torch.is_grad_enabled())

    def _select_subset(self, packed: torch.Tensor, metadata, group_mask: torch.Tensor) -> Qwen3ExpertSubset | None:
        group_mask = group_mask.to(device=metadata.expert_indices.device, dtype=torch.bool)
        route_mask = group_mask.index_select(0, metadata.expert_indices)
        row_indices = torch.nonzero(route_mask, as_tuple=False).flatten()
        if int(row_indices.numel()) == 0:
            return None

        subset_packed = packed.index_select(0, row_indices)
        subset_experts_for_rows = metadata.expert_indices.index_select(0, row_indices)
        counts = torch.bincount(subset_experts_for_rows, minlength=self.num_experts).to(device=packed.device, dtype=torch.long)
        offsets = torch.cat(
            (
                torch.zeros(1, device=packed.device, dtype=torch.long),
                torch.cumsum(counts, dim=0),
            ),
            dim=0,
        )
        experts = torch.arange(self.num_experts + 1, device=packed.device, dtype=torch.long)
        experts[-1] = -1
        return Qwen3ExpertSubset(packed=subset_packed, offsets=offsets, experts=experts, row_indices=row_indices)

    def _run_subset_body(
        self,
        subset: Qwen3ExpertSubset | None,
        *,
        checkpoint_body: bool = False,
        checkpoint_activation_down: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if subset is None:
            return None
        if checkpoint_body and checkpoint_activation_down:
            raise ValueError("checkpoint_body and checkpoint_activation_down are mutually exclusive")

        if checkpoint_body:
            def body_fn(x: torch.Tensor) -> torch.Tensor:
                return self._forward_expert_body(
                    x,
                    subset.offsets,
                    subset.experts,
                    dense_experts=True,
                )

            output = checkpoint(
                body_fn,
                subset.packed,
                use_reentrant=False,
                preserve_rng_state=self._checkpoint_preserve_rng_state(),
            ) if subset.packed.requires_grad else body_fn(subset.packed)
            return subset.row_indices, output

        if checkpoint_activation_down:
            gate, up, lora_metadata = self._forward_gate_up(
                subset.packed,
                subset.offsets,
                subset.experts,
                dense_experts=True,
            )

            def activation_down_fn(gate_arg: torch.Tensor, up_arg: torch.Tensor) -> torch.Tensor:
                return self._forward_activation_down(
                    gate_arg,
                    up_arg,
                    subset.offsets,
                    subset.experts,
                    lora_metadata,
                    dense_experts=True,
                    input_dtype=subset.packed.dtype,
                )

            output = checkpoint(
                activation_down_fn,
                gate,
                up,
                use_reentrant=False,
                preserve_rng_state=self._checkpoint_preserve_rng_state(),
            ) if gate.requires_grad or up.requires_grad else activation_down_fn(gate, up)
            return subset.row_indices, output

        output = self._forward_expert_body(
            subset.packed,
            subset.offsets,
            subset.experts,
            dense_experts=True,
        )
        return subset.row_indices, output

    def _run_dense_checkpoint_body(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
    ) -> torch.Tensor:
        def body_fn(x: torch.Tensor) -> torch.Tensor:
            return self._forward_expert_body(x, offsets, experts, dense_experts=True)

        return checkpoint(
            body_fn,
            packed,
            use_reentrant=False,
            preserve_rng_state=self._checkpoint_preserve_rng_state(),
        ) if packed.requires_grad else body_fn(packed)

    def _run_dense_checkpoint_activation_down(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
    ) -> torch.Tensor:
        gate, up, lora_metadata = self._forward_gate_up(
            packed,
            offsets,
            experts,
            dense_experts=True,
        )

        def activation_down_fn(gate_arg: torch.Tensor, up_arg: torch.Tensor) -> torch.Tensor:
            return self._forward_activation_down(
                gate_arg,
                up_arg,
                offsets,
                experts,
                lora_metadata,
                dense_experts=True,
                input_dtype=packed.dtype,
            )

        return checkpoint(
            activation_down_fn,
            gate,
            up,
            use_reentrant=False,
            preserve_rng_state=self._checkpoint_preserve_rng_state(),
        ) if gate.requires_grad or up.requires_grad else activation_down_fn(gate, up)

    def _forward_expert_policy(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        config: ExpertRecomputeConfig = self.expert_recompute_config
        counts = metadata.expert_counts.to(device=packed.device, dtype=torch.long)
        active_groups = counts > 0
        empty_groups = torch.zeros_like(active_groups)
        selected_by_recompute = (
            expert_recompute_group_mask(
                counts,
                policy=config.policy,
                token_threshold=config.token_threshold,
                util_threshold=config.util_threshold,
            )
            if config.recompute_enabled
            else empty_groups
        )
        split_groups = selected_by_recompute if config.policy == "split" else empty_groups
        recompute_groups = selected_by_recompute if config.policy != "split" else empty_groups
        activation_drop_groups = (
            expert_activation_drop_group_mask(
                counts,
                policy=config.activation_save_policy,
                token_threshold=config.activation_save_threshold,
            )
            if config.activation_drop_enabled
            else empty_groups
        ) & ~(recompute_groups | split_groups)
        selected_groups = recompute_groups | activation_drop_groups | split_groups
        if not bool(selected_groups.any().item()):
            return self._forward_expert_body(packed, offsets, experts, dense_experts=True)

        kept_groups = active_groups & ~selected_groups
        output = packed.new_empty((packed.shape[0], self.hidden_dim))
        pieces = (
            self._run_subset_body(self._select_subset(packed, metadata, kept_groups)),
            self._run_subset_body(self._select_subset(packed, metadata, split_groups)),
            self._run_subset_body(self._select_subset(packed, metadata, recompute_groups), checkpoint_body=True),
            self._run_subset_body(
                self._select_subset(packed, metadata, activation_drop_groups),
                checkpoint_activation_down=True,
            ),
        )
        for piece in pieces:
            if piece is None:
                continue
            row_indices, values = piece
            output.index_copy_(0, row_indices, values)
        return output

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts=self.num_experts)
        packed = pack_tokens_contiguous(hidden_states, metadata)
        offsets, experts = make_dense_group_metadata(
            metadata.expert_offsets,
            num_groups=self.num_experts,
            device=packed.device,
        )
        if self._uses_expert_recompute():
            down = self._forward_expert_policy(packed, offsets, experts, metadata)
        else:
            down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
        return scatter_contiguous(down, metadata).to(dtype=input_dtype)


def wrap_qwen3_experts(
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
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
) -> AsymQwen3Experts:
    return AsymQwen3Experts(
        source,
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
        strict=strict,
    )


__all__ = [
    "AsymQwen3Experts",
    "Qwen3ExpertReport",
    "is_qwen3_experts",
    "wrap_qwen3_experts",
]
