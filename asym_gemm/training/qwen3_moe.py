from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .frozen_linear import (
    AsymExecutionStats,
    AsymGroupedFrozenLinear,
    TorchGroupedFrozenLinear,
    _dispatch_grouped_nt,
    _get_quantized_host_weight,
    _grouped_torch_chunks,
)
from .lora import (
    GroupedLoRAMetadata,
    _require_lora_grouped_mm,
    grouped_expert_lora,
    grouped_expert_lora_pair,
    normalize_lora_dtype,
    prepare_grouped_lora_metadata,
)
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
from .profile_ranges import prof_range, scoped_name


@dataclass(frozen=True)
class Qwen3ExpertReport:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    cpu_resident_base_bytes: int
    gpu_resident_base_bytes: int
    trainable_lora_params: int
    expert_recompute_policy: str = "none"


def _scatter_contiguous_sum(expert_output: torch.Tensor, metadata) -> torch.Tensor:
    if expert_output.shape[0] != metadata.num_routes:
        raise ValueError(f"expected {metadata.num_routes} route outputs, got {expert_output.shape[0]}")
    flat = expert_output.reshape(metadata.num_routes, -1)
    out = torch.zeros(
        (metadata.num_tokens, flat.shape[1]),
        device=expert_output.device,
        dtype=flat.dtype,
    )
    out.index_add_(0, metadata.token_indices, flat)
    return out.reshape(metadata.num_tokens, *expert_output.shape[1:])


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


def _is_silu_activation(fn) -> bool:
    if fn is F.silu or isinstance(fn, nn.SiLU):
        return True
    name = getattr(fn, "__name__", "")
    cls_name = type(fn).__name__
    return "silu" in name.lower() or "silu" in cls_name.lower() or "swish" in cls_name.lower()


SAVE_EMPTY = 0
SAVE_FULL = 1
SAVE_COMPACT = 2
SAVE_UNKNOWN = -1


def _empty_group_metadata_like(offsets: torch.Tensor, experts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return offsets.new_zeros((1,)), experts.new_full((1,), -1)


def _save_values_for_plan(values: torch.Tensor, rows: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == SAVE_FULL:
        return values.detach()
    if mode == SAVE_EMPTY:
        return values.new_empty((0, values.shape[-1]))
    return values.detach().index_select(0, rows)


def _make_group_plan(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    group_mask: torch.Tensor,
    active_groups: torch.Tensor,
    *,
    rows_total: int,
    mode_hint: int = SAVE_UNKNOWN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rows = torch.empty((0,), device=offsets.device, dtype=torch.long)
    if mode_hint == SAVE_FULL:
        return offsets, experts, rows, SAVE_FULL
    if mode_hint == SAVE_EMPTY:
        empty_offsets, empty_experts = _empty_group_metadata_like(offsets, experts)
        return empty_offsets, empty_experts, rows, SAVE_EMPTY

    selected_groups = group_mask & active_groups
    counts = (offsets[1:] - offsets[:-1]).to(dtype=torch.long)
    row_mask = torch.repeat_interleave(selected_groups, counts, output_size=int(rows_total))
    rows = torch.nonzero(row_mask, as_tuple=False).flatten()
    compact_counts = counts[selected_groups]
    zero = offsets.new_zeros((1,))
    compact_offsets = torch.cat((zero, compact_counts.to(dtype=offsets.dtype).cumsum(dim=0)))
    compact_experts = torch.cat((experts[:-1][selected_groups], experts.new_full((1,), -1)))
    return compact_offsets, compact_experts, rows, SAVE_COMPACT


def _make_group_row_plan(
    values: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    group_mask: torch.Tensor,
    active_groups: torch.Tensor,
    *,
    mode_hint: int = SAVE_UNKNOWN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    compact_offsets, compact_experts, rows, mode = _make_group_plan(
        offsets,
        experts,
        group_mask,
        active_groups,
        rows_total=int(values.shape[0]),
        mode_hint=mode_hint,
    )
    saved = _save_values_for_plan(values, rows, mode)
    return saved, compact_offsets, compact_experts, rows, mode


def _restore_saved_rows(
    saved: torch.Tensor,
    rows: torch.Tensor,
    mode: int,
    *,
    rows_total: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == SAVE_FULL:
        return saved
    restored = torch.empty((rows_total, width), device=device, dtype=dtype)
    if mode == SAVE_COMPACT and int(rows.shape[0]) > 0:
        restored[rows] = saved
    return restored


def _forward_gate_up_selected_or_empty(
    layer: "AsymQwen3Experts",
    packed: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(packed.shape[0]) == 0:
        empty = packed.new_empty((0, layer.intermediate_dim))
        low_rank_empty = packed.new_empty((0, layer.lora_rank), dtype=layer.lora_dtype)
        return empty, empty, low_rank_empty, low_rank_empty
    gate, up, _, gate_low_rank, up_low_rank = layer._forward_gate_up(
        packed,
        offsets,
        experts,
        dense_experts=False,
        compiled_dims="nk",
        return_low_rank=True,
    )
    return gate, up, gate_low_rank, up_low_rank


def _policy_selects_all_active_groups(config: ExpertRecomputeConfig) -> bool:
    return bool(config.policy == "tok" and config.token_min <= 1 and config.token_max is None)


def _activation_drop_selects_all_active_groups(config: ExpertRecomputeConfig) -> bool:
    return bool(
        config.activation_save_policy == "tok_act"
        and config.activation_save_min <= 1
        and config.activation_save_max is None
    )


def _grouped_base_dx(
    base: nn.Module,
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    dense_experts: bool,
    input_dtype: torch.dtype,
    profile_name: str,
) -> torch.Tensor:
    if isinstance(base, TorchGroupedFrozenLinear):
        grad_2d = grad_output.reshape(-1, int(base.weight.shape[1])).contiguous()
        with prof_range(f"backward.{profile_name}.grouped_base_dx_torch"):
            grad_x = _grouped_torch_chunks(
                grad_2d,
                base.weight,
                offsets,
                experts,
                transpose_b=True,
                dense_experts=dense_experts,
            )
        return grad_x.reshape(*grad_output.shape[:-1], int(base.weight.shape[2]))

    if isinstance(base, AsymGroupedFrozenLinear):
        grad_2d = grad_output.reshape(-1, int(base.host_weight.weight.shape[1])).contiguous()
        if base.precision == "bf16" and base.backend != "torch" and grad_2d.dtype != base.host_weight.weight.dtype:
            grad_2d = grad_2d.to(dtype=base.host_weight.weight.dtype)
        quantized_weight_t = (
            _get_quantized_host_weight(base.host_weight, base.precision, transpose=True)
            if base.backend != "torch" and base.precision != "bf16"
            else None
        )
        with prof_range(f"backward.{profile_name}.grouped_base_dx_asymgemm"):
            grad_x = _dispatch_grouped_nt(
                grad_2d,
                base.host_weight.weight,
                offsets,
                experts,
                backend=base.backend,
                stats=base.stats,
                phase="dx",
                compiled_dims=base.compiled_dims,
                transpose_b=True,
                precision=base.precision,
                quantized_weight=quantized_weight_t,
                dense_experts=dense_experts,
                bf16_output_dtype=input_dtype if base.precision == "bf16" else base.bf16_output_dtype,
            )
        return grad_x.reshape(*grad_output.shape[:-1], int(base.host_weight.weight.shape[2]))

    raise TypeError(f"unsupported grouped base module {type(base).__name__}")


def _lora_backward_group(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    *,
    scale: float,
    need_grad_x: bool,
    precomputed_low_rank: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    x_lora = x.to(dtype=a_weight.dtype)
    grad_lora = grad_y.to(dtype=b_weight.dtype)
    low_rank = F.linear(x_lora, a_weight) if precomputed_low_rank is None else precomputed_low_rank.to(dtype=a_weight.dtype)
    grad_b = grad_lora.transpose(0, 1).matmul(low_rank).mul(scale).to(dtype=b_weight.dtype)
    grad_low_rank = grad_lora.matmul(b_weight).mul(scale)
    grad_a = grad_low_rank.transpose(0, 1).matmul(x_lora).to(dtype=a_weight.dtype)
    grad_x = grad_low_rank.matmul(a_weight).to(dtype=x.dtype) if need_grad_x else None
    return grad_x, grad_a, grad_b


def _record_reference_fallback(stats: AsymExecutionStats | None, reason: str) -> None:
    if stats is not None:
        stats.record_reference_fallback(reason)


def _grouped_lora_backward(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    scale: float,
    need_grad_x: bool,
    precomputed_low_rank: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    grad_x = torch.zeros_like(x) if need_grad_x else None
    grad_a = torch.zeros_like(a_weight)
    grad_b = torch.zeros_like(b_weight)
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    experts_cpu = experts.detach().to(device="cpu", dtype=torch.long).tolist()
    for group_idx, expert_idx in enumerate(experts_cpu[:-1]):
        start = int(offsets_cpu[group_idx])
        end = int(offsets_cpu[group_idx + 1])
        if end <= start:
            continue
        expert = int(expert_idx)
        grad_x_group, grad_a_group, grad_b_group = _lora_backward_group(
            x[start:end],
            grad_y[start:end],
            a_weight[expert],
            b_weight[expert],
            scale=scale,
            need_grad_x=need_grad_x,
            precomputed_low_rank=None if precomputed_low_rank is None else precomputed_low_rank[start:end],
        )
        if grad_x is not None and grad_x_group is not None:
            grad_x[start:end].add_(grad_x_group)
        grad_a[expert].add_(grad_a_group)
        grad_b[expert].add_(grad_b_group)
    return grad_x, grad_a, grad_b


def _grouped_lora_weight_grads_torch(
    left: torch.Tensor,
    right: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    *,
    out_dtype: torch.dtype,
    metadata: GroupedLoRAMetadata | None = None,
    stats: AsymExecutionStats | None = None,
) -> torch.Tensor:
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"left/right row mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    if int(left.shape[0]) == 0:
        return torch.zeros(
            (num_experts, int(left.shape[1]), int(right.shape[1])),
            device=left.device,
            dtype=out_dtype,
        )
    # Local torch._grouped_mm requires some row strides to be at least
    # 16-byte aligned. Production LoRA ranks such as 8/64 satisfy this;
    # tiny test ranks like 2/3 do not, so keep the reference path there.
    if (int(right.shape[1]) * int(right.element_size())) % 16 != 0:
        _record_reference_fallback(stats, "lora_weight_grad_unaligned")
        return _grouped_lora_weight_grads_reference(
            left,
            right,
            offsets,
            experts,
            num_experts,
            out_dtype=out_dtype,
        )
    metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=False) if metadata is None else metadata
    grouped_mm = _require_lora_grouped_mm()
    left_t = left.transpose(0, 1)
    grouped = grouped_mm(left_t, right, offs=metadata.active_offsets)
    grouped = grouped.to(dtype=out_dtype)
    if metadata.dense_expert_weights and int(grouped.shape[0]) == int(num_experts):
        return grouped
    out = torch.zeros(
        (num_experts, int(left.shape[1]), int(right.shape[1])),
        device=left.device,
        dtype=out_dtype,
    )
    out.index_add_(0, metadata.active_experts.to(device=left.device, dtype=torch.long, non_blocking=True), grouped)
    return out


def _grouped_lora_weight_grads_reference(
    left: torch.Tensor,
    right: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros(
        (num_experts, int(left.shape[1]), int(right.shape[1])),
        device=left.device,
        dtype=out_dtype,
    )
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long).tolist()
    experts_cpu = experts.detach().to(device="cpu", dtype=torch.long).tolist()
    for group_idx, expert_idx in enumerate(experts_cpu[:-1]):
        start = int(offsets_cpu[group_idx])
        end = int(offsets_cpu[group_idx + 1])
        if end <= start:
            continue
        out[int(expert_idx)].add_(left[start:end].transpose(0, 1).matmul(right[start:end]).to(dtype=out_dtype))
    return out


def _grouped_lora_backward_loop_free(
    x: torch.Tensor,
    grad_y: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    scale: float,
    need_grad_x: bool,
    precomputed_low_rank: torch.Tensor | None = None,
    metadata: GroupedLoRAMetadata | None = None,
    stats: AsymExecutionStats | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda" or grad_y.device.type != "cuda":
        _record_reference_fallback(stats, "lora_backward_non_cuda")
        return _grouped_lora_backward(
            x,
            grad_y,
            a_weight,
            b_weight,
            offsets,
            experts,
            scale=scale,
            need_grad_x=need_grad_x,
            precomputed_low_rank=precomputed_low_rank,
        )
    if (int(a_weight.shape[1]) * int(a_weight.element_size())) % 16 != 0:
        _record_reference_fallback(stats, "lora_backward_unaligned")
        return _grouped_lora_backward(
            x,
            grad_y,
            a_weight,
            b_weight,
            offsets,
            experts,
            scale=scale,
            need_grad_x=need_grad_x,
            precomputed_low_rank=precomputed_low_rank,
        )

    x_lora = x if x.dtype == a_weight.dtype else x.to(dtype=a_weight.dtype)
    grad_lora = grad_y if grad_y.dtype == b_weight.dtype else grad_y.to(dtype=b_weight.dtype)
    lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True) if metadata is None else metadata
    low_rank = (
        grouped_expert_lora(x_lora, a_weight, offsets, experts, metadata=lora_metadata)
        if precomputed_low_rank is None
        else (precomputed_low_rank if precomputed_low_rank.dtype == a_weight.dtype else precomputed_low_rank.to(dtype=a_weight.dtype))
    )
    grad_b = _grouped_lora_weight_grads_torch(
        grad_lora,
        low_rank,
        offsets,
        experts,
        int(b_weight.shape[0]),
        out_dtype=b_weight.dtype,
        metadata=lora_metadata,
        stats=stats,
    ).mul_(scale)
    grad_low_rank = grouped_expert_lora(
        grad_lora,
        b_weight.transpose(-1, -2),
        offsets,
        experts,
        metadata=lora_metadata,
    ).mul_(scale)
    grad_a = _grouped_lora_weight_grads_torch(
        grad_low_rank,
        x_lora,
        offsets,
        experts,
        int(a_weight.shape[0]),
        out_dtype=a_weight.dtype,
        metadata=lora_metadata,
        stats=stats,
    )
    grad_x = None
    if need_grad_x:
        grad_x = grouped_expert_lora(
            grad_low_rank,
            a_weight.transpose(-1, -2),
            offsets,
            experts,
            metadata=lora_metadata,
        ).to(dtype=x.dtype)
    return grad_x, grad_a, grad_b


class _ThresholdedQwen3ExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        recompute_groups: torch.Tensor,
        activation_drop_groups: torch.Tensor,
        gate_lora_A: torch.Tensor,
        gate_lora_B: torch.Tensor,
        up_lora_A: torch.Tensor,
        up_lora_B: torch.Tensor,
        down_lora_A: torch.Tensor,
        down_lora_B: torch.Tensor,
        layer: "AsymQwen3Experts",
    ) -> torch.Tensor:
        with prof_range("forward.mlp.expert_policy.body_with_intermediates"):
            output, gate, up, activated, gate_low_rank, up_low_rank, down_low_rank = layer._forward_expert_body_with_intermediates(
                packed,
                offsets,
                experts,
                dense_experts=True,
            )
        with prof_range("forward.mlp.expert_policy.prepare_masks"):
            offsets = offsets.detach().contiguous()
            experts = experts.detach().contiguous()
            active_groups = offsets[1:] > offsets[:-1]
            config = layer.expert_recompute_config
            recompute_enabled = config.recompute_enabled
            activation_drop_enabled = config.activation_drop_enabled
            recompute_all_active = recompute_enabled and _policy_selects_all_active_groups(config)
            activation_drop_all_active = activation_drop_enabled and _activation_drop_selects_all_active_groups(config)
            empty_groups = torch.empty((0,), device=packed.device, dtype=torch.bool)
            recompute_groups = (
                recompute_groups.detach().to(device=packed.device, dtype=torch.bool).contiguous()
                if recompute_enabled
                else empty_groups
            )
            activation_drop_groups = (
                activation_drop_groups.detach().to(device=packed.device, dtype=torch.bool).contiguous()
                if activation_drop_enabled
                else empty_groups
            )
            if recompute_enabled and activation_drop_enabled:
                activation_drop_groups = activation_drop_groups & ~recompute_groups
            keep_gate_up_groups = active_groups if not recompute_enabled else active_groups & ~recompute_groups
            activated_save_groups = (
                keep_gate_up_groups if not activation_drop_enabled else keep_gate_up_groups & ~activation_drop_groups
            )
            if recompute_enabled and activation_drop_enabled:
                activation_rebuild_groups = recompute_groups | activation_drop_groups
            elif recompute_enabled:
                activation_rebuild_groups = recompute_groups
            elif activation_drop_enabled:
                activation_rebuild_groups = activation_drop_groups
            else:
                activation_rebuild_groups = empty_groups
            gate_up_mode_hint = SAVE_FULL if not recompute_enabled else SAVE_EMPTY if recompute_all_active else SAVE_UNKNOWN
            activated_mode_hint = (
                SAVE_FULL
                if not recompute_enabled and not activation_drop_enabled
                else SAVE_EMPTY
                if recompute_all_active or (activation_drop_all_active and not recompute_enabled)
                else SAVE_UNKNOWN
            )
            recompute_mode_hint = SAVE_EMPTY if not recompute_enabled else SAVE_FULL if recompute_all_active else SAVE_UNKNOWN
            activation_rebuild_mode_hint = (
                SAVE_EMPTY
                if not recompute_enabled and not activation_drop_enabled
                else SAVE_FULL
                if recompute_all_active or (activation_drop_all_active and not recompute_enabled)
                else SAVE_UNKNOWN
            )
        with prof_range("forward.mlp.expert_policy.save_gate_up_plan"):
            gate_saved, _, _, gate_saved_rows, gate_saved_mode = _make_group_row_plan(
                gate.detach(),
                offsets,
                experts,
                keep_gate_up_groups,
                active_groups,
                mode_hint=gate_up_mode_hint,
            )
            up_saved = _save_values_for_plan(up, gate_saved_rows, gate_saved_mode)
            gate_low_rank_saved = _save_values_for_plan(gate_low_rank, gate_saved_rows, gate_saved_mode)
            up_low_rank_saved = _save_values_for_plan(up_low_rank, gate_saved_rows, gate_saved_mode)
        with prof_range("forward.mlp.expert_policy.save_activated_plan"):
            activated_saved, _, _, activated_saved_rows, activated_saved_mode = _make_group_row_plan(
                activated.detach(),
                offsets,
                experts,
                activated_save_groups,
                active_groups,
                mode_hint=activated_mode_hint,
            )
            down_low_rank_saved = _save_values_for_plan(down_low_rank, activated_saved_rows, activated_saved_mode)
        with prof_range("forward.mlp.expert_policy.save_recompute_plan"):
            recompute_offsets, recompute_experts, recompute_rows, recompute_mode = _make_group_plan(
                offsets,
                experts,
                recompute_groups,
                active_groups,
                rows_total=int(packed.shape[0]),
                mode_hint=recompute_mode_hint,
            )
        with prof_range("forward.mlp.expert_policy.save_activation_rebuild_plan"):
            activation_offsets, activation_experts, activation_rows, activation_rebuild_mode = _make_group_plan(
                offsets,
                experts,
                activation_rebuild_groups,
                active_groups,
                rows_total=int(packed.shape[0]),
                mode_hint=activation_rebuild_mode_hint,
            )
        ctx.layer = layer
        ctx.input_dtype = packed.dtype
        ctx.gate_saved_mode = gate_saved_mode
        ctx.activated_saved_mode = activated_saved_mode
        ctx.recompute_mode = recompute_mode
        ctx.activation_rebuild_mode = activation_rebuild_mode
        with prof_range("forward.mlp.expert_policy.save_context"):
            ctx.save_for_backward(
                packed.detach(),
                offsets,
                experts,
                gate_saved,
                up_saved,
                activated_saved,
                gate_saved_rows,
                activated_saved_rows,
                recompute_offsets,
                recompute_experts,
                recompute_rows,
                activation_offsets,
                activation_experts,
                activation_rows,
                gate_low_rank_saved,
                up_low_rank_saved,
                down_low_rank_saved,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            packed,
            offsets,
            experts,
            gate_saved,
            up_saved,
            activated_saved,
            gate_saved_rows,
            activated_saved_rows,
            recompute_offsets,
            recompute_experts,
            recompute_rows,
            activation_offsets,
            activation_experts,
            activation_rows,
            gate_low_rank_saved,
            up_low_rank_saved,
            down_low_rank_saved,
            gate_lora_A,
            gate_lora_B,
            up_lora_A,
            up_lora_B,
            down_lora_A,
            down_lora_B,
        ) = ctx.saved_tensors
        layer: AsymQwen3Experts = ctx.layer
        need_grad_packed = ctx.needs_input_grad[0]
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)

        with prof_range("backward.mlp.expert_policy.down_base_dx"):
            grad_activated = _grouped_base_dx(
                layer.down_base,
                grad_output,
                offsets,
                experts,
                dense_experts=True,
                input_dtype=ctx.input_dtype,
                profile_name=layer._profile_name("down", "base"),
            )

        with prof_range("backward.mlp.expert_policy.restore_saved"):
            rows_total = int(packed.shape[0])
            gate_full = _restore_saved_rows(
                gate_saved,
                gate_saved_rows,
                ctx.gate_saved_mode,
                rows_total=rows_total,
                width=layer.intermediate_dim,
                device=packed.device,
                dtype=grad_output.dtype,
            )
            up_full = _restore_saved_rows(
                up_saved,
                gate_saved_rows,
                ctx.gate_saved_mode,
                rows_total=rows_total,
                width=layer.intermediate_dim,
                device=packed.device,
                dtype=grad_output.dtype,
            )
            activated_full = _restore_saved_rows(
                activated_saved,
                activated_saved_rows,
                ctx.activated_saved_mode,
                rows_total=rows_total,
                width=layer.intermediate_dim,
                device=packed.device,
                dtype=grad_output.dtype,
            )
            gate_low_rank_full = _restore_saved_rows(
                gate_low_rank_saved,
                gate_saved_rows,
                ctx.gate_saved_mode,
                rows_total=rows_total,
                width=layer.lora_rank,
                device=packed.device,
                dtype=gate_lora_A.dtype,
            )
            up_low_rank_full = _restore_saved_rows(
                up_low_rank_saved,
                gate_saved_rows,
                ctx.gate_saved_mode,
                rows_total=rows_total,
                width=layer.lora_rank,
                device=packed.device,
                dtype=up_lora_A.dtype,
            )
            down_low_rank_full = _restore_saved_rows(
                down_low_rank_saved,
                activated_saved_rows,
                ctx.activated_saved_mode,
                rows_total=rows_total,
                width=layer.lora_rank,
                device=packed.device,
                dtype=down_lora_A.dtype,
            )

        with prof_range("backward.mlp.expert_policy.pack_recompute_selected"):
            if ctx.recompute_mode == SAVE_FULL:
                recompute_packed = packed
            elif ctx.recompute_mode == SAVE_COMPACT:
                recompute_packed = packed.index_select(0, recompute_rows)
            else:
                recompute_packed = packed.new_empty((0, packed.shape[-1]))
        with prof_range("backward.mlp.expert_policy.recompute_gate_up_selected"):
            if ctx.recompute_mode != SAVE_EMPTY:
                gate_recompute, up_recompute, gate_low_rank_recompute, up_low_rank_recompute = _forward_gate_up_selected_or_empty(
                    layer,
                    recompute_packed,
                    recompute_offsets,
                    recompute_experts,
                )
                if ctx.recompute_mode == SAVE_FULL:
                    gate_full = gate_recompute
                    up_full = up_recompute
                    gate_low_rank_full = gate_low_rank_recompute
                    up_low_rank_full = up_low_rank_recompute
                elif int(recompute_rows.shape[0]) > 0:
                    gate_full[recompute_rows] = gate_recompute
                    up_full[recompute_rows] = up_recompute
                    gate_low_rank_full[recompute_rows] = gate_low_rank_recompute
                    up_low_rank_full[recompute_rows] = up_low_rank_recompute

        with prof_range("backward.mlp.expert_policy.rebuild_activation_selected"):
            if ctx.activation_rebuild_mode == SAVE_FULL:
                activated_full = layer.act_fn(gate_full) * up_full
                down_low_rank_full = grouped_expert_lora(
                    activated_full.to(dtype=down_lora_A.dtype),
                    down_lora_A,
                    activation_offsets,
                    activation_experts,
                    metadata=lora_metadata,
                )
            elif ctx.activation_rebuild_mode == SAVE_COMPACT and int(activation_rows.shape[0]) > 0:
                gate_need = gate_full.index_select(0, activation_rows)
                up_need = up_full.index_select(0, activation_rows)
                activated_need = layer.act_fn(gate_need) * up_need
                activated_full[activation_rows] = activated_need
                activation_lora_metadata = prepare_grouped_lora_metadata(
                    activation_offsets,
                    activation_experts,
                    dense_experts=False,
                )
                down_low_rank_need = grouped_expert_lora(
                    activated_need.to(dtype=down_lora_A.dtype),
                    down_lora_A,
                    activation_offsets,
                    activation_experts,
                    metadata=activation_lora_metadata,
                )
                down_low_rank_full[activation_rows] = down_low_rank_need

        with prof_range("backward.mlp.expert_policy.down_lora_backward"):
            grad_down_lora_x, grad_down_lora_A, grad_down_lora_B = _grouped_lora_backward_loop_free(
                activated_full,
                grad_output,
                down_lora_A,
                down_lora_B,
                offsets,
                experts,
                scale=layer.lora_scale,
                need_grad_x=True,
                precomputed_low_rank=down_low_rank_full,
                metadata=lora_metadata,
                stats=layer.stats,
            )
            if grad_down_lora_x is not None:
                grad_activated.add_(
                    grad_down_lora_x if grad_down_lora_x.dtype == grad_activated.dtype else grad_down_lora_x.to(dtype=grad_activated.dtype)
                )

        with prof_range("backward.mlp.expert_policy.activation_grad_silu"):
            silu = F.silu(gate_full)
            grad_up = grad_activated * silu
            if grad_up.dtype != grad_output.dtype:
                grad_up = grad_up.to(dtype=grad_output.dtype)
            grad_gate_input = grad_activated.mul(up_full)
            grad_gate = torch.ops.aten.silu_backward(grad_gate_input, gate_full)
            if grad_gate.dtype != grad_output.dtype:
                grad_gate = grad_gate.to(dtype=grad_output.dtype)

        grad_packed = None
        if need_grad_packed:
            with prof_range("backward.mlp.expert_policy.gate_up_base_dx"):
                grad_gate_up = torch.empty((packed.shape[0], 2 * layer.intermediate_dim), device=packed.device, dtype=grad_gate.dtype)
                grad_gate_up[:, : layer.intermediate_dim] = grad_gate
                grad_gate_up[:, layer.intermediate_dim :] = grad_up
                grad_packed = _grouped_base_dx(
                    layer.gate_up_base,
                    grad_gate_up,
                    offsets,
                    experts,
                    dense_experts=True,
                    input_dtype=ctx.input_dtype,
                    profile_name=layer._profile_name("gate_up", "base"),
                )

        with prof_range("backward.mlp.expert_policy.gate_lora_backward"):
            grad_gate_lora_x, grad_gate_lora_A, grad_gate_lora_B = _grouped_lora_backward_loop_free(
                packed,
                grad_gate,
                gate_lora_A,
                gate_lora_B,
                offsets,
                experts,
                scale=layer.lora_scale,
                need_grad_x=need_grad_packed,
                precomputed_low_rank=gate_low_rank_full,
                metadata=lora_metadata,
                stats=layer.stats,
            )
        with prof_range("backward.mlp.expert_policy.up_lora_backward"):
            grad_up_lora_x, grad_up_lora_A, grad_up_lora_B = _grouped_lora_backward_loop_free(
                packed,
                grad_up,
                up_lora_A,
                up_lora_B,
                offsets,
                experts,
                scale=layer.lora_scale,
                need_grad_x=need_grad_packed,
                precomputed_low_rank=up_low_rank_full,
                metadata=lora_metadata,
                stats=layer.stats,
            )
        with prof_range("backward.mlp.expert_policy.merge_lora_dx"):
            if grad_packed is not None and grad_gate_lora_x is not None:
                grad_packed.add_(
                    grad_gate_lora_x if grad_gate_lora_x.dtype == grad_packed.dtype else grad_gate_lora_x.to(dtype=grad_packed.dtype)
                )
            if grad_packed is not None and grad_up_lora_x is not None:
                grad_packed.add_(
                    grad_up_lora_x if grad_up_lora_x.dtype == grad_packed.dtype else grad_up_lora_x.to(dtype=grad_packed.dtype)
                )

        return (
            grad_packed,
            None,
            None,
            None,
            None,
            grad_gate_lora_A,
            grad_gate_lora_B,
            grad_up_lora_A,
            grad_up_lora_B,
            grad_down_lora_A,
            grad_down_lora_B,
            None,
        )


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
        self.profile_prefix = "layers.unknown.mlp.experts"

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

    def _forward_gate_up_lora(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        metadata,
        *,
        return_low_rank: bool = False,
    ):
        if self.lora_dropout_p == 0.0:
            x_lora = x.to(dtype=self.lora_dtype)
            gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)
            with prof_range(self._forward_range("gate_up", "lora_a")):
                low_rank = grouped_expert_lora(x_lora, gate_up_a, offsets, experts, metadata=metadata)
            gate_low_rank, up_low_rank = low_rank.split(self.lora_rank, dim=-1)
        else:
            gate_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            up_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            with prof_range(self._forward_range("gate_up", "gate_lora_a")):
                gate_low_rank = grouped_expert_lora(gate_input, self.gate_lora_A, offsets, experts, metadata=metadata)
            with prof_range(self._forward_range("gate_up", "up_lora_a")):
                up_low_rank = grouped_expert_lora(up_input, self.up_lora_A, offsets, experts, metadata=metadata)
        with prof_range(self._forward_range("gate_up", "lora_b")):
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
        if return_low_rank:
            return gate_delta, up_delta, gate_low_rank, up_low_rank
        return gate_delta, up_delta

    def _forward_down_lora(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        metadata,
        *,
        return_low_rank: bool = False,
    ):
        x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
        with prof_range(self._forward_range("down", "lora_a")):
            low_rank = grouped_expert_lora(x_lora, self.down_lora_A, offsets, experts, metadata=metadata)
        with prof_range(self._forward_range("down", "lora_b")):
            delta = grouped_expert_lora(low_rank, self.down_lora_B, offsets, experts, metadata=metadata)
        if self.lora_scale != 1.0:
            delta = delta.mul(self.lora_scale)
        if return_low_rank:
            return delta, low_rank
        return delta

    def _lora_metadata(self, offsets: torch.Tensor, experts: torch.Tensor, *, dense_experts: bool):
        with prof_range(self._forward_range("route_metadata", "lora_metadata")):
            return prepare_grouped_lora_metadata(offsets, experts, dense_experts=dense_experts)

    def _profile_name(self, *parts: object) -> str:
        return scoped_name(self.profile_prefix, *parts)

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def _forward_gate_up(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool,
        compiled_dims: str | None = None,
        return_low_rank: bool = False,
    ):
        lora_metadata = self._lora_metadata(offsets, experts, dense_experts=dense_experts)
        gate_up_kwargs = {
            "dense_experts": dense_experts,
            "profile_name": self._profile_name("gate_up", "base"),
        }
        if compiled_dims is not None and isinstance(self.gate_up_base, AsymGroupedFrozenLinear):
            gate_up_kwargs["compiled_dims"] = compiled_dims
        gate_up = self.gate_up_base(x, offsets, experts, **gate_up_kwargs)
        gate, up = gate_up.chunk(2, dim=-1)
        gate_up_lora = self._forward_gate_up_lora(
            x,
            offsets,
            experts,
            lora_metadata,
            return_low_rank=return_low_rank,
        )
        if return_low_rank:
            gate_delta, up_delta, gate_low_rank, up_low_rank = gate_up_lora
        else:
            gate_delta, up_delta = gate_up_lora
        gate = gate + gate_delta.to(dtype=gate.dtype)
        up = up + up_delta.to(dtype=up.dtype)
        if return_low_rank:
            return gate, up, lora_metadata, gate_low_rank, up_low_rank
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
        with prof_range(self._forward_range("activation_silu_mul")):
            activated = self.act_fn(gate) * up
        down = self.down_base(
            activated.to(dtype=input_dtype),
            offsets,
            experts,
            dense_experts=dense_experts,
            profile_name=self._profile_name("down", "base"),
        )
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

    def _forward_expert_body_with_intermediates(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        dense_experts: bool,
    ):
        gate, up, lora_metadata, gate_low_rank, up_low_rank = self._forward_gate_up(
            packed,
            offsets,
            experts,
            dense_experts=dense_experts,
            return_low_rank=True,
        )
        with prof_range(self._forward_range("activation_silu_mul")):
            activated = self.act_fn(gate) * up
        down = self.down_base(
            activated.to(dtype=packed.dtype),
            offsets,
            experts,
            dense_experts=dense_experts,
            profile_name=self._profile_name("down", "base"),
        )
        down_delta, down_low_rank = self._forward_down_lora(
            activated,
            offsets,
            experts,
            lora_metadata,
            return_low_rank=True,
        )
        return down + down_delta.to(dtype=down.dtype), gate, up, activated, gate_low_rank, up_low_rank, down_low_rank

    def _uses_expert_recompute(self) -> bool:
        return bool(self.expert_recompute_config.enabled and self.training and torch.is_grad_enabled())

    def _forward_expert_policy(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        config: ExpertRecomputeConfig = self.expert_recompute_config
        if self.lora_dropout_p > 0.0:
            raise NotImplementedError("Qwen3 expert recompute requires lora_dropout=0.0; no slow checkpoint fallback is used")
        if not _is_silu_activation(self.act_fn):
            raise NotImplementedError("AsymGEMM expert recompute supports only SiLU expert activation")
        with prof_range(self._forward_range("expert_policy")):
            counts = metadata.expert_counts.to(device=packed.device, dtype=torch.long)
            recompute_groups = (
                expert_recompute_group_mask(
                    counts,
                    policy=config.policy,
                    token_threshold=config.token_threshold,
                    token_min=config.token_min,
                    token_max=config.token_max,
                )
                if config.recompute_enabled
                else counts.new_empty((0,), dtype=torch.bool)
            )
            activation_drop_groups = (
                expert_activation_drop_group_mask(
                    counts,
                    policy=config.activation_save_policy,
                    token_threshold=config.activation_save_threshold,
                    token_min=config.activation_save_min,
                    token_max=config.activation_save_max,
                )
                if config.activation_drop_enabled
                else counts.new_empty((0,), dtype=torch.bool)
            )
            if config.recompute_enabled and config.activation_drop_enabled:
                activation_drop_groups = activation_drop_groups & ~recompute_groups

        return _ThresholdedQwen3ExpertFunction.apply(
            packed,
            offsets,
            experts,
            recompute_groups,
            activation_drop_groups,
            self.gate_lora_A,
            self.gate_lora_B,
            self.up_lora_A,
            self.up_lora_B,
            self.down_lora_A,
            self.down_lora_B,
            self,
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        with prof_range(self._forward_range("route_metadata")):
            metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts=self.num_experts)
        with prof_range(self._forward_range("pack_tokens")):
            packed = pack_tokens_contiguous(hidden_states, metadata)
        with prof_range(self._forward_range("route_metadata", "dense_groups")):
            offsets, experts = make_dense_group_metadata(
                metadata.expert_offsets,
                num_groups=self.num_experts,
                device=packed.device,
            )
        if self._uses_expert_recompute():
            down = self._forward_expert_policy(packed, offsets, experts, metadata)
        else:
            down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
        with prof_range(self._forward_range("scatter_combine")):
            return scatter_contiguous(down, metadata).to(dtype=input_dtype)

    def forward_input_scaled(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        input_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run routed experts where route weights scale inputs before the expert body.

        Qwen3 multiplies each expert output by the router weight. Llama 4 instead
        multiplies the expert input by the selected router score and then sums
        outputs. This entry point keeps the same packed expert kernels, LoRA
        path, and recompute policies while preserving that routing semantic.
        """

        if input_weights.shape != top_k_index.shape:
            raise ValueError(
                "input_weights must match top_k_index shape, "
                f"got {tuple(input_weights.shape)} and {tuple(top_k_index.shape)}"
            )
        input_dtype = hidden_states.dtype
        with prof_range(self._forward_range("route_metadata")):
            metadata = build_contiguous_route_metadata(top_k_index, input_weights, num_experts=self.num_experts)
        with prof_range(self._forward_range("pack_tokens")):
            packed = pack_tokens_contiguous(hidden_states, metadata)
            route_scale = metadata.routing_weights.reshape(metadata.num_routes, *([1] * (packed.dim() - 1)))
            packed = packed * route_scale.to(device=packed.device, dtype=packed.dtype)
        with prof_range(self._forward_range("route_metadata", "dense_groups")):
            offsets, experts = make_dense_group_metadata(
                metadata.expert_offsets,
                num_groups=self.num_experts,
                device=packed.device,
            )
        if self._uses_expert_recompute():
            down = self._forward_expert_policy(packed, offsets, experts, metadata)
        else:
            down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
        with prof_range(self._forward_range("scatter_combine")):
            return _scatter_contiguous_sum(down, metadata).to(dtype=input_dtype)


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
