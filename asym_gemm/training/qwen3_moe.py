from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace as _dataclass_replace
import inspect
import math
import os
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .frozen_linear import (
    AsymExecutionStats,
    AsymFrozenLinear,
    AsymGroupedFrozenLinear,
    TorchGroupedFrozenLinear,
    _dispatch_grouped_nt,
    _dispatch_nt,
    _get_quantized_host_weight,
    _grouped_torch_chunks,
)
from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .exp_act_offload_lora import (
    grouped_lora_a_forward_cpu_left,
    grouped_lora_a_forward_hbm,
    grouped_lora_a_grad_cpu_right,
    grouped_lora_a_pair_forward_cpu_left,
    grouped_lora_a_pair_grad_cpu_right,
    require_expert_activation_offload_kernels,
    stage_low_rank_from_cpu,
)
from .lora import (
    GroupedLoRAMetadata,
    _require_lora_grouped_mm,
    grouped_expert_lora,
    grouped_expert_lora_pair,
    normalize_lora_dtype,
    prepare_grouped_lora_metadata,
)
from .offload import adopt_host_weight
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
    if not callable(_resolve_qwen3_expert_act_fn(module)):
        return False
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in ("hidden_states", "top_k_index", "top_k_weights"))


def _resolve_qwen3_expert_act_fn(module: nn.Module):
    act_fn = getattr(module, "act_fn", None)
    if callable(act_fn):
        return act_fn
    hidden_act = str(getattr(getattr(module, "config", None), "hidden_act", "")).lower()
    if hidden_act in {"silu", "swish"}:
        return F.silu
    # Liger's packed Experts module stores the same Qwen3/Qwen3.5 weight layout
    # but does not carry act_fn/config; its constructor has already rejected
    # non-SiLU activations.
    if type(module).__name__ == "LigerExperts":
        return F.silu
    return None


def is_qwen3_moe_block(module: nn.Module) -> bool:
    if getattr(module, "_is_asym_qwen3_moe_block", False):
        return False
    if hasattr(module, "shared_expert") or hasattr(module, "shared_expert_gate"):
        return False
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    if not isinstance(gate, nn.Module) or not is_qwen3_experts(experts):
        return False
    for attr in ("hidden_dim", "top_k", "num_experts"):
        if not isinstance(getattr(gate, attr, None), int):
            return False
    return callable(getattr(gate, "forward", None))


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


def _empty_packed_mask(device: torch.device) -> torch.Tensor:
    return torch.empty((0, 0), device=device, dtype=torch.uint8)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV = "ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD"
VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD = {"cpu", "hbm"}


def _expert_act_offload_lora_a_fwd_mode() -> str:
    raw = os.environ.get(EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV, "cpu")
    mode = str(raw)
    if mode not in VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD:
        valid = ", ".join(sorted(VALID_EXPERT_ACT_OFFLOAD_LORA_A_FWD))
        raise ValueError(f"{EXPERT_ACT_OFFLOAD_LORA_A_FWD_ENV} must be one of {valid}, got {raw!r}")
    return mode


def _expert_act_offload_act_recompute() -> bool:
    return _env_flag("ASYM_OFFLOAD_ACT_RECOMPUTE", False)


def _expert_act_offload_x_unpacked() -> bool:
    return _env_flag("ASYM_OFFLOAD_X_UNPACKED", False)


def _qwen3_window_param(name: str, default: int) -> int:
    value = os.environ.get(f"ASYM_QWEN3_GATE_UP_WINDOWED_BWD_{name}")
    if value is None or not value.strip():
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _qwen3_window_tile_config(intermediate_dim: int, hidden_dim: int) -> tuple[int, int, int, int, int]:
    p = min(_qwen3_window_param("P", 32), max(1, int(intermediate_dim)))
    q_default = _qwen3_window_param("Q", 8)
    q = min(q_default, max(1, int(intermediate_dim) // p))
    bm = _qwen3_window_param("BM", 64)
    bk = min(_qwen3_window_param("BK", 512), max(1, int(hidden_dim)))
    g_work = _qwen3_window_param("G_WORK", 128)
    return p, q, bm, bk, g_work


def _pack_bool_mask_2d(mask_bool: torch.Tensor) -> torch.Tensor:
    if mask_bool.dtype != torch.bool or mask_bool.dim() != 2:
        raise ValueError(f"mask_bool must be a 2D bool tensor, got shape={tuple(mask_bool.shape)} dtype={mask_bool.dtype}")
    if mask_bool.device.type == "cuda":
        import asym_gemm

        pack = getattr(getattr(asym_gemm, "_C", None), "pack_bool_mask_2d", None)
        if pack is not None:
            return pack(mask_bool.contiguous())
    mask_u8 = mask_bool.contiguous().to(dtype=torch.uint8)
    rows, width = int(mask_u8.shape[0]), int(mask_u8.shape[1])
    packed_width = (width + 7) // 8
    if packed_width == 0:
        return torch.empty((rows, 0), device=mask_bool.device, dtype=torch.uint8)
    padded_width = packed_width * 8
    if padded_width != width:
        mask_u8 = F.pad(mask_u8, (0, padded_width - width))
    bits = (1 << torch.arange(8, device=mask_bool.device, dtype=torch.uint8)).view(1, 1, 8)
    return mask_u8.view(rows, packed_width, 8).mul(bits).sum(dim=-1).to(dtype=torch.uint8)


def _unpack_bool_mask_2d(mask_packed: torch.Tensor, width: int) -> torch.Tensor:
    if mask_packed.dtype != torch.uint8 or mask_packed.dim() != 2:
        raise ValueError(f"mask_packed must be a 2D uint8 tensor, got shape={tuple(mask_packed.shape)} dtype={mask_packed.dtype}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    expected = (int(width) + 7) // 8
    if int(mask_packed.shape[1]) != expected:
        raise ValueError(f"mask_packed width mismatch: got {int(mask_packed.shape[1])}, expected {expected}")
    if mask_packed.device.type == "cuda":
        import asym_gemm

        unpack = getattr(getattr(asym_gemm, "_C", None), "unpack_bool_mask_2d", None)
        if unpack is not None:
            return unpack(mask_packed.contiguous(), int(width))
    if width == 0:
        return torch.empty((int(mask_packed.shape[0]), 0), device=mask_packed.device, dtype=torch.bool)
    bit_ids = torch.arange(8, device=mask_packed.device, dtype=torch.long).view(1, 1, 8)
    unpacked = ((mask_packed.contiguous().unsqueeze(-1).to(dtype=torch.long) >> bit_ids) & 1).to(dtype=torch.bool)
    return unpacked.reshape(int(mask_packed.shape[0]), expected * 8)[:, : int(width)]


def _apply_saved_dropout(
    x: torch.Tensor,
    mask_packed: torch.Tensor | None,
    dropout_p: float,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if dropout_p == 0.0 or mask_packed is None or int(mask_packed.numel()) == 0:
        return x if x.dtype == out_dtype else x.to(dtype=out_dtype)
    if not (0.0 < float(dropout_p) < 1.0):
        raise ValueError(f"dropout_p must satisfy 0.0 < p < 1.0 when a mask is provided, got {dropout_p}")
    x_out = x if x.dtype == out_dtype else x.to(dtype=out_dtype)
    if x_out.device.type == "cuda":
        import asym_gemm

        apply = getattr(getattr(asym_gemm, "_C", None), "apply_packed_dropout", None)
        if apply is not None:
            return apply(x_out.contiguous(), mask_packed.contiguous(), float(dropout_p))
    mask = _unpack_bool_mask_2d(mask_packed, int(x_out.shape[1]))
    scale = 1.0 / (1.0 - float(dropout_p))
    return torch.where(mask, x_out * scale, torch.zeros((), device=x_out.device, dtype=x_out.dtype))


def _apply_saved_dropout_(
    x: torch.Tensor,
    mask_packed: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    if dropout_p == 0.0 or mask_packed is None or int(mask_packed.numel()) == 0:
        return x
    if x.device.type == "cuda":
        import asym_gemm

        apply_inplace = getattr(getattr(asym_gemm, "_C", None), "apply_packed_dropout_", None)
        if apply_inplace is not None and x.is_contiguous():
            return apply_inplace(x, mask_packed.contiguous(), float(dropout_p))
    mask = _unpack_bool_mask_2d(mask_packed, int(x.shape[1]))
    x.mul_(mask.to(dtype=x.dtype)).mul_(1.0 / (1.0 - float(dropout_p)))
    return x


def _native_dropout_with_packed_mask(
    x: torch.Tensor,
    p: float,
    *,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if p == 0.0:
        return x.to(dtype=out_dtype), _empty_packed_mask(x.device)
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"lora_dropout must satisfy 0.0 <= p < 1.0 in the custom expert path, got {p}")
    x_drop, mask_bool = torch.ops.aten.native_dropout(x, float(p), True)
    return x_drop.to(dtype=out_dtype), _pack_bool_mask_2d(mask_bool)


def _forward_gate_up_with_saved_low_rank(
    layer: "AsymQwen3Experts",
    packed: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    gate_low_rank: torch.Tensor,
    up_low_rank: torch.Tensor,
    gate_lora_B: torch.Tensor,
    up_lora_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(packed.shape[0]) == 0:
        empty = packed.new_empty((0, layer.intermediate_dim))
        return empty, empty
    gate_up_kwargs = {
        "dense_experts": False,
        "profile_name": layer._profile_name("gate_up", "base"),
    }
    if isinstance(layer.gate_up_base, AsymGroupedFrozenLinear):
        gate_up_kwargs["compiled_dims"] = "nk"
    gate_up = layer.gate_up_base(packed, offsets, experts, **gate_up_kwargs)
    gate, up = gate_up.chunk(2, dim=-1)
    metadata = layer._lora_metadata(offsets, experts, dense_experts=False)
    gate_delta, up_delta = grouped_expert_lora_pair(
        gate_low_rank,
        up_low_rank,
        gate_lora_B,
        up_lora_B,
        offsets,
        experts,
        metadata=metadata,
    )
    if layer.lora_scale != 1.0:
        gate_delta = gate_delta.mul(layer.lora_scale)
        up_delta = up_delta.mul(layer.lora_scale)
    return gate + gate_delta.to(dtype=gate.dtype), up + up_delta.to(dtype=up.dtype)


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
    dropout_mask_packed: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    x_lora = _apply_saved_dropout(x, dropout_mask_packed, dropout_p, out_dtype=a_weight.dtype)
    grad_lora = grad_y.to(dtype=b_weight.dtype)
    low_rank = F.linear(x_lora, a_weight) if precomputed_low_rank is None else precomputed_low_rank.to(dtype=a_weight.dtype)
    grad_b = grad_lora.transpose(0, 1).matmul(low_rank).mul(scale).to(dtype=b_weight.dtype)
    grad_low_rank = grad_lora.matmul(b_weight).mul(scale)
    grad_a = grad_low_rank.transpose(0, 1).matmul(x_lora).to(dtype=a_weight.dtype)
    grad_x = None
    if need_grad_x:
        grad_x_raw = grad_low_rank.matmul(a_weight)
        grad_x = _apply_saved_dropout(grad_x_raw, dropout_mask_packed, dropout_p, out_dtype=x.dtype)
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
    dropout_mask_packed: torch.Tensor | None = None,
    dropout_p: float = 0.0,
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
            dropout_mask_packed=None if dropout_mask_packed is None or int(dropout_mask_packed.numel()) == 0 else dropout_mask_packed[start:end],
            dropout_p=dropout_p,
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
    record_lora_b_backward: bool = False,
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
        out = _grouped_lora_weight_grads_reference(
            left,
            right,
            offsets,
            experts,
            num_experts,
            out_dtype=out_dtype,
        )
        if stats is not None and record_lora_b_backward:
            stats.expact_lora_b_backward_grouped_calls += 1
        return out
    metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=False) if metadata is None else metadata
    grouped_mm = _require_lora_grouped_mm()
    left_t = left.transpose(0, 1)
    grouped = grouped_mm(left_t, right, offs=metadata.active_offsets)
    grouped = grouped.to(dtype=out_dtype)
    if stats is not None and record_lora_b_backward:
        stats.expact_lora_b_backward_grouped_calls += 1
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
    dropout_mask_packed: torch.Tensor | None = None,
    dropout_p: float = 0.0,
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
            dropout_mask_packed=dropout_mask_packed,
            dropout_p=dropout_p,
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
            dropout_mask_packed=dropout_mask_packed,
            dropout_p=dropout_p,
        )

    x_lora = _apply_saved_dropout(x, dropout_mask_packed, dropout_p, out_dtype=a_weight.dtype)
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
        record_lora_b_backward=True,
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
        grad_x_raw = grouped_expert_lora(
            grad_low_rank,
            a_weight.transpose(-1, -2),
            offsets,
            experts,
            metadata=lora_metadata,
        )
        grad_x = _apply_saved_dropout_(grad_x_raw.contiguous(), dropout_mask_packed, dropout_p).to(dtype=x.dtype)
    return grad_x, grad_a, grad_b


def _selected_rows_for_mode(
    rows_total: int,
    rows: torch.Tensor,
    mode: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if mode == SAVE_FULL:
        return torch.arange(rows_total, device=device, dtype=torch.long)
    if mode == SAVE_COMPACT:
        return rows
    return torch.empty((0,), device=device, dtype=torch.long)


def _nonselected_group_plan(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    selected_experts: torch.Tensor,
    selected_mode: int,
    *,
    rows_total: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if selected_mode == SAVE_FULL:
        empty_offsets, empty_experts = _empty_group_metadata_like(offsets, experts)
        return empty_offsets, empty_experts, torch.empty((0,), device=offsets.device, dtype=torch.long), SAVE_EMPTY
    if selected_mode == SAVE_EMPTY:
        return offsets, experts, torch.empty((0,), device=offsets.device, dtype=torch.long), SAVE_FULL
    active_groups = offsets[1:] > offsets[:-1]
    selected_ids = selected_experts[:-1]
    selected_group_mask = torch.isin(experts[:-1], selected_ids) if int(selected_ids.numel()) > 0 else torch.zeros_like(active_groups)
    return _make_group_plan(
        offsets,
        experts,
        active_groups & ~selected_group_mask,
        active_groups,
        rows_total=rows_total,
    )


def _select_packed_mask_rows(mask_packed: torch.Tensor, rows: torch.Tensor, mode: int, device: torch.device) -> torch.Tensor:
    if mask_packed is None or int(mask_packed.numel()) == 0:
        return _empty_packed_mask(device)
    if mode == SAVE_FULL:
        return mask_packed
    if int(rows.numel()) == 0:
        return _empty_packed_mask(device)
    return mask_packed.index_select(0, rows)


def _grouped_down_lora_backward_split_loop_free(
    activated_for_nonselected: torch.Tensor,
    grad_y: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    nonselected_offsets: torch.Tensor,
    nonselected_experts: torch.Tensor,
    nonselected_rows: torch.Tensor,
    *,
    scale: float,
    precomputed_low_rank: torch.Tensor | None,
    dropout_mask_packed: torch.Tensor | None,
    dropout_p: float,
    metadata: GroupedLoRAMetadata,
    stats: AsymExecutionStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_lora = grad_y if grad_y.dtype == b_weight.dtype else grad_y.to(dtype=b_weight.dtype)
    low_rank = (
        precomputed_low_rank if precomputed_low_rank is not None and precomputed_low_rank.dtype == a_weight.dtype
        else None if precomputed_low_rank is None
        else precomputed_low_rank.to(dtype=a_weight.dtype)
    )
    if low_rank is None:
        raise RuntimeError("down-LoRA split backward requires saved down low-rank tensor")

    grad_b = _grouped_lora_weight_grads_torch(
        grad_lora,
        low_rank,
        offsets,
        experts,
        int(b_weight.shape[0]),
        out_dtype=b_weight.dtype,
        metadata=metadata,
        stats=stats,
        record_lora_b_backward=True,
    ).mul_(scale)
    dS_down = grouped_expert_lora(
        grad_lora,
        b_weight.transpose(-1, -2),
        offsets,
        experts,
        metadata=metadata,
    ).mul_(scale)

    grad_x_raw = grouped_expert_lora(
        dS_down,
        a_weight.transpose(-1, -2),
        offsets,
        experts,
        metadata=metadata,
    )
    grad_x = _apply_saved_dropout_(grad_x_raw.contiguous(), dropout_mask_packed, dropout_p).to(dtype=activated_for_nonselected.dtype)

    grad_a = torch.zeros_like(a_weight)
    if int(nonselected_rows.numel()) > 0:
        act_nonselected = activated_for_nonselected.index_select(0, nonselected_rows)
        mask_nonselected = _select_packed_mask_rows(dropout_mask_packed, nonselected_rows, SAVE_COMPACT, activated_for_nonselected.device)
        act_lora_nonselected = _apply_saved_dropout(act_nonselected, mask_nonselected, dropout_p, out_dtype=a_weight.dtype)
        dS_nonselected = dS_down.index_select(0, nonselected_rows)
        nonselected_metadata = prepare_grouped_lora_metadata(nonselected_offsets, nonselected_experts, dense_experts=False)
        grad_a.add_(
            _grouped_lora_weight_grads_torch(
                dS_nonselected,
                act_lora_nonselected,
                nonselected_offsets,
                nonselected_experts,
                int(a_weight.shape[0]),
                out_dtype=a_weight.dtype,
                metadata=nonselected_metadata,
                stats=stats,
            )
        )
    return grad_x, grad_a, grad_b, dS_down


def _grouped_lora_cuda_view(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise RuntimeError("activation-offload grouped LoRA-B view path requires CUDA tensors")
    if int(x.shape[0]) == 0:
        return x.new_empty((0, int(weight.shape[1])))
    if int(metadata.active_offsets.numel()) == 0:
        return x.new_empty((0, int(weight.shape[1])))
    if metadata.dense_expert_weights and int(weight.shape[0]) == metadata.active_groups:
        selected = weight
    else:
        selected = weight.index_select(0, metadata.active_experts.to(device=weight.device, dtype=torch.long, non_blocking=True))
    grouped_mm = _require_lora_grouped_mm()
    return grouped_mm(x, selected.transpose(-1, -2), offs=metadata.active_offsets)


def _activation_offload_cpu_silu_mul(
    gate: CPUActivationHandle,
    up: CPUActivationHandle,
    manager: ActivationOffloadManager,
    *,
    tag: str,
) -> CPUActivationHandle:
    manager.wait_cpu_ready_host(gate)
    manager.wait_cpu_ready_host(up)
    out = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, tag)
    with torch.no_grad():
        out.tensor.copy_(F.silu(gate.tensor).mul(up.tensor), non_blocking=False)
    return out


def _activation_offload_cpu_silu_backward(
    gate: CPUActivationHandle,
    up: CPUActivationHandle,
    grad_act: CPUActivationHandle,
    manager: ActivationOffloadManager,
) -> tuple[CPUActivationHandle, CPUActivationHandle]:
    with prof_range("backward.mlp.activation_offload.activation_cpu.wait"):
        manager.wait_cpu_ready_host(gate)
        manager.wait_cpu_ready_host(up)
        manager.wait_cpu_ready_host(grad_act)
    with prof_range("backward.mlp.activation_offload.activation_cpu.alloc"):
        grad_gate = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, "dgate")
        grad_up = manager.empty_cpu(tuple(up.tensor.shape), up.tensor.dtype, up.original_device, "dup")
    with prof_range("backward.mlp.activation_offload.activation_cpu.math"):
        with torch.no_grad():
            silu = F.silu(gate.tensor)
            grad_up.tensor.copy_(grad_act.tensor.mul(silu), non_blocking=False)
            grad_gate.tensor.copy_(torch.ops.aten.silu_backward(grad_act.tensor.mul(up.tensor), gate.tensor), non_blocking=False)
    return grad_gate, grad_up


def _use_gpu_silu_bwd() -> bool:
    """v14: opt-in. Compute the expert SwiGLU backward on the GPU instead of on the
    CPU. The CPU path is ~640 ms/layer (CPU-bandwidth-contended with the concurrent
    gradient-offload D2H copies); the GPU is ~90% idle and does the same math in
    sub-millisecond. Default OFF (preserves the all-CPU activation-offload behavior)."""
    v = os.environ.get("ASYMM_EXPERT_SILU_BWD_GPU", "")
    return v.lower() not in {"", "0", "false", "no", "off"}


def _silu_backward_gpu(
    gate_cpu: CPUActivationHandle,
    up_cpu: CPUActivationHandle,
    grad_act: torch.Tensor,
    manager: ActivationOffloadManager,
):
    """GPU SwiGLU backward. Stages the offloaded gate/up activations back to the GPU
    (transient ~200 MB each, released immediately), keeps grad_act on the GPU, and
    computes grad_gate/grad_up there. Returns the same layout the CPU path produced:
    a concatenated grad_gate_up = [grad_gate | grad_up] plus its two split views.

    Takes the gate/up CPU handles explicitly (symmetric with the CPU
    `_activation_offload_cpu_silu_backward`) so the Qwen3 and Llama4 expert
    backwards can share it.

    Numerically identical to `_activation_offload_cpu_silu_backward` (same BF16 ops):
      grad_up   = grad_act * silu(gate)
      grad_gate = silu_backward(grad_act * up, gate)
    """
    with prof_range("backward.mlp.activation_offload.activation_cpu.stage"):
        gate_gpu = manager.stage(gate_cpu, tag="gate_for_silu_bwd")
        up_gpu = manager.stage(up_cpu, tag="up_for_silu_bwd")
    with prof_range("backward.mlp.activation_offload.activation_cpu.math"):
        inter = int(gate_gpu.shape[1])
        grad_gate_up = torch.empty(
            (int(gate_gpu.shape[0]), 2 * inter), device=gate_gpu.device, dtype=gate_gpu.dtype
        )
        with torch.no_grad():
            silu = F.silu(gate_gpu)
            grad_gate_up[:, inter:].copy_(grad_act.mul(silu))
            grad_gate_up[:, :inter].copy_(torch.ops.aten.silu_backward(grad_act.mul(up_gpu), gate_gpu))
        grad_gate_stage, grad_up_stage = grad_gate_up.split(inter, dim=-1)
    manager.release_stage(gate_gpu, drop_cache=True)
    manager.release_stage(up_gpu, drop_cache=True)
    manager.release_cpu(gate_cpu)
    manager.release_cpu(up_cpu)
    return grad_gate_up, grad_gate_stage, grad_up_stage


def _rebuild_qwen3_packed_x_cpu(
    ctx,
    manager: ActivationOffloadManager,
) -> tuple[CPUActivationHandle, bool]:
    if not getattr(ctx, "x_unpacked", False):
        return ctx.x_cpu, False

    hidden_cpu = ctx.x_cpu.tensor
    token_indices = getattr(ctx, "x_token_indices_cpu", None)
    if token_indices is None:
        raise RuntimeError("Qwen3 unpacked-X reconstruction requires token indices")

    manager.wait_cpu_ready_host(ctx.x_cpu)
    rebuilt = manager.empty_cpu(
        (int(token_indices.numel()), int(hidden_cpu.shape[1])),
        hidden_cpu.dtype,
        ctx.x_cpu.original_device,
        "X",
    )
    torch.index_select(hidden_cpu, 0, token_indices, out=rebuilt.tensor)
    route_scale = getattr(ctx, "x_route_scale_cpu", None)
    if route_scale is not None:
        rebuilt.tensor.mul_(route_scale.reshape(-1, 1).to(dtype=rebuilt.tensor.dtype))
    return rebuilt, True


class _ActivationOffloadQwen3ExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        x_src_hidden: torch.Tensor | None,
        x_token_indices: torch.Tensor | None,
        x_route_scale: torch.Tensor | None,
        gate_lora_A: torch.Tensor,
        gate_lora_B: torch.Tensor,
        up_lora_A: torch.Tensor,
        up_lora_B: torch.Tensor,
        down_lora_A: torch.Tensor,
        down_lora_B: torch.Tensor,
        layer: "AsymQwen3Experts",
    ) -> torch.Tensor:
        manager = ActivationOffloadManager(pin_memory=True)
        offsets = offsets.detach().contiguous()
        experts = experts.detach().contiguous()
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
        lora_a_forward_mode = _expert_act_offload_lora_a_fwd_mode()
        act_recompute = _expert_act_offload_act_recompute()
        x_unpacked = (
            _expert_act_offload_x_unpacked()
            and lora_a_forward_mode == "hbm"
            and x_src_hidden is not None
            and x_token_indices is not None
        )

        x_token_indices_cpu = None
        x_route_scale_cpu = None
        with prof_range(layer._forward_range("activation_offload", "x_to_cpu")):
            if x_unpacked:
                x_cpu = manager.offload(x_src_hidden.detach(), "X")
                x_token_indices_cpu = x_token_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
                if x_route_scale is not None:
                    x_route_scale_cpu = x_route_scale.detach().to(device="cpu", dtype=x_cpu.tensor.dtype).contiguous()
            else:
                x_cpu = manager.offload(packed, "X")

        with prof_range(layer._forward_range("activation_offload", "gate_up_base")):
            gate_up = layer.gate_up_base(
                packed,
                offsets,
                experts,
                dense_experts=True,
                compiled_dims="nk",
                profile_name=layer._profile_name("gate_up", "base"),
            )
            gate, up = gate_up.chunk(2, dim=-1)

        with prof_range(layer._forward_range("activation_offload", "gate_up_lora_a")):
            gate_up_low_rank_owner = None
            if lora_a_forward_mode == "cpu":
                gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
                    x_cpu.tensor,
                    gate_lora_A,
                    up_lora_A,
                    offsets,
                    experts,
                    metadata=lora_metadata,
                    stats=layer.stats,
                    tag="gate_up",
                )
            elif lora_a_forward_mode == "hbm":
                gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1).contiguous()
                try:
                    gate_up_low_rank_owner = grouped_lora_a_forward_hbm(
                        packed,
                        gate_up_a,
                        offsets,
                        experts,
                        metadata=lora_metadata,
                        stats=layer.stats,
                        tag="gate_up",
                    )
                    gate_low_rank, up_low_rank = gate_up_low_rank_owner.split(layer.lora_rank, dim=-1)
                finally:
                    del gate_up_a
            else:
                raise AssertionError(f"unreachable LoRA-A forward mode {lora_a_forward_mode}")

        with prof_range(layer._forward_range("activation_offload", "gate_up_lora_b")):
            gate_delta, up_delta = grouped_expert_lora_pair(
                gate_low_rank,
                up_low_rank,
                gate_lora_B,
                up_lora_B,
                offsets,
                experts,
                metadata=lora_metadata,
            )
            if layer.lora_scale != 1.0:
                gate_delta = gate_delta.mul(layer.lora_scale)
                up_delta = up_delta.mul(layer.lora_scale)
            gate.add_(gate_delta.to(dtype=gate.dtype))
            up.add_(up_delta.to(dtype=up.dtype))
            del gate_delta, up_delta

        with prof_range(layer._forward_range("activation_offload", "save_gate_up_cpu")):
            gate_cpu = manager.offload(gate, "gate")
            up_cpu = manager.offload(up, "up")
            gate_low_rank_cpu = manager.offload(gate_low_rank, "S_gate")
            up_low_rank_cpu = manager.offload(up_low_rank, "S_up")
            del gate, up, gate_up, gate_low_rank, up_low_rank
            if gate_up_low_rank_owner is not None:
                del gate_up_low_rank_owner

        with prof_range(layer._forward_range("activation_offload", "activation_cpu")):
            act_cpu = _activation_offload_cpu_silu_mul(gate_cpu, up_cpu, manager, tag="act")

        if lora_a_forward_mode == "cpu":
            with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
                down_low_rank = grouped_lora_a_forward_cpu_left(
                    act_cpu.tensor,
                    down_lora_A,
                    offsets,
                    experts,
                    metadata=lora_metadata,
                    stats=layer.stats,
                    tag="down",
                )
            with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
                down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
                if layer.lora_scale != 1.0:
                    down_delta = down_delta.mul(layer.lora_scale)
                down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
                del down_low_rank

            with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
                act_stage = manager.stage(act_cpu, tag="act_for_down_base")
                output = layer.down_base(
                    act_stage,
                    offsets,
                    experts,
                    dense_experts=True,
                    profile_name=layer._profile_name("down", "base"),
                )
                manager.release_stage(act_stage, drop_cache=True)
                output.add_(down_delta.to(dtype=output.dtype))
                del down_delta, act_stage
        else:
            act_stage = None
            try:
                with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
                    act_stage = manager.stage(act_cpu, tag="act_for_down_base")
                with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
                    down_low_rank = grouped_lora_a_forward_hbm(
                        act_stage,
                        down_lora_A,
                        offsets,
                        experts,
                        metadata=lora_metadata,
                        stats=layer.stats,
                        tag="down",
                    )
                with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
                    down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
                    if layer.lora_scale != 1.0:
                        down_delta = down_delta.mul(layer.lora_scale)
                    down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
                    del down_low_rank
                with prof_range(layer._forward_range("activation_offload", "down_base")):
                    output = layer.down_base(
                        act_stage,
                        offsets,
                        experts,
                        dense_experts=True,
                        profile_name=layer._profile_name("down", "base"),
                    )
                    output.add_(down_delta.to(dtype=output.dtype))
                    del down_delta
            finally:
                if act_stage is not None:
                    manager.release_stage(act_stage, drop_cache=True)

        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_recompute = act_recompute
        ctx.x_unpacked = x_unpacked
        ctx.x_token_indices_cpu = x_token_indices_cpu
        ctx.x_route_scale_cpu = x_route_scale_cpu
        if act_recompute:
            manager.release_cpu(act_cpu)
            ctx.act_cpu = None
        else:
            ctx.act_cpu = act_cpu
        ctx.gate_low_rank_cpu = gate_low_rank_cpu
        ctx.up_low_rank_cpu = up_low_rank_cpu
        ctx.down_low_rank_cpu = down_low_rank_cpu
        ctx.input_dtype = packed.dtype
        ctx.lora_dropout_p = float(layer.lora_dropout_p)
        ctx.expert_lora_a_forward_mode = lora_a_forward_mode
        ctx.weight_offload = getattr(layer, "_weight_offload", None) is not None
        if ctx.weight_offload:
            # Weight offload: do NOT keep the GPU weights alive across the fwd->bwd gap.
            # The forward hook releases them; backward re-gathers from ctx.layer.
            ctx.save_for_backward(offsets, experts)
        else:
            ctx.save_for_backward(
                offsets,
                experts,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            )
        activation_offload_stats = manager.snapshot()
        activation_offload_stats["expert_lora_a_forward_mode"] = lora_a_forward_mode
        activation_offload_stats["qwen3_act_recompute"] = bool(act_recompute)
        activation_offload_stats["qwen3_x_unpacked"] = bool(x_unpacked)
        layer._last_activation_offload_stats = activation_offload_stats
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        layer: AsymQwen3Experts = ctx.layer
        if getattr(ctx, "weight_offload", False):
            # Re-gather the layer's LoRA banks to GPU (single H2D) before reading them. param.data is
            # full-shaped here (AccumulateGrad needs it); per-bank release is owned by the optimizer's
            # post-accumulate grad hook, which runs strictly after grad accumulation.
            offsets, experts = ctx.saved_tensors
            layer.gather_lora_weights()
            gate_lora_A = layer.gate_lora_A
            gate_lora_B = layer.gate_lora_B
            up_lora_A = layer.up_lora_A
            up_lora_B = layer.up_lora_B
            down_lora_A = layer.down_lora_A
            down_lora_B = layer.down_lora_B
        else:
            (
                offsets,
                experts,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            ) = ctx.saved_tensors
        manager: ActivationOffloadManager = ctx.manager
        need_grad_packed = ctx.needs_input_grad[0]
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)

        with prof_range("backward.mlp.activation_offload.down_lora"):
            grad_lora = grad_output if grad_output.dtype == down_lora_B.dtype else grad_output.to(dtype=down_lora_B.dtype)
            dS_down = grouped_expert_lora(
                grad_lora,
                down_lora_B.transpose(-1, -2),
                offsets,
                experts,
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_down_lora_x = grouped_expert_lora(
                dS_down,
                down_lora_A.transpose(-1, -2),
                offsets,
                experts,
                metadata=lora_metadata,
            )
            down_low_rank = stage_low_rank_from_cpu(
                ctx.down_low_rank_cpu,
                manager,
                tag="S_down_for_dB",
                stats=layer.stats,
            )
            grad_down_lora_B = _grouped_lora_weight_grads_torch(
                grad_lora,
                down_low_rank,
                offsets,
                experts,
                int(down_lora_B.shape[0]),
                out_dtype=down_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(down_low_rank, drop_cache=True)
            manager.release_cpu(ctx.down_low_rank_cpu)
            if getattr(ctx, "act_recompute", False):
                act_handle = _activation_offload_cpu_silu_mul(ctx.gate_cpu, ctx.up_cpu, manager, tag="act")
            else:
                act_handle = ctx.act_cpu
            grad_down_lora_A = grouped_lora_a_grad_cpu_right(
                dS_down,
                act_handle.tensor,
                offsets,
                experts,
                num_experts=int(down_lora_A.shape[0]),
                stats=layer.stats,
                tag="down",
            )
            manager.release_cpu(act_handle)

        with prof_range("backward.mlp.activation_offload.down_base_dx"):
            grad_act = _grouped_base_dx(
                layer.down_base,
                grad_output,
                offsets,
                experts,
                dense_experts=True,
                input_dtype=ctx.input_dtype,
                profile_name=layer._profile_name("down", "base"),
            )
            grad_act.add_(grad_down_lora_x.to(dtype=grad_act.dtype))
            del grad_down_lora_x
            # GPU SwiGLU-backward keeps grad_act resident on the GPU; the CPU path
            # offloads it (D2H) so the SwiGLU backward can run on the host.
            if _use_gpu_silu_bwd():
                grad_act_cpu = None
            else:
                grad_act_cpu = manager.offload(grad_act, "dact")
                del grad_act

        grad_packed = None
        grad_gate_lora_x = None
        grad_up_lora_x = None
        # The GPU SwiGLU-backward path never materializes these CPU handles; keep them
        # defined so the shared final-cleanup loop (release_cpu tolerates None) is happy.
        grad_gate_cpu = None
        grad_up_cpu = None

        if _use_gpu_silu_bwd():
            with prof_range("backward.mlp.activation_offload.activation_cpu"):
                grad_gate_up, grad_gate_stage, grad_up_stage = _silu_backward_gpu(ctx.gate_cpu, ctx.up_cpu, grad_act, manager)
                del grad_act
        else:
            with prof_range("backward.mlp.activation_offload.activation_cpu"):
                grad_gate_cpu, grad_up_cpu = _activation_offload_cpu_silu_backward(
                    ctx.gate_cpu,
                    ctx.up_cpu,
                    grad_act_cpu,
                    manager,
                )
                manager.release_cpu(grad_act_cpu)
                manager.release_cpu(ctx.gate_cpu)
                manager.release_cpu(ctx.up_cpu)

            with prof_range("backward.mlp.activation_offload.gate_up_stage"):
                grad_gate_up = manager.stage_concat_columns(grad_gate_cpu, grad_up_cpu, tag="dgate_up_for_gate_up_base")
                grad_gate_stage, grad_up_stage = grad_gate_up.split(int(grad_gate_cpu.tensor.shape[1]), dim=-1)
                manager.release_cpu(grad_gate_cpu)
                manager.release_cpu(grad_up_cpu)

        with prof_range("backward.mlp.activation_offload.gate_lora"):
            gate_low_rank = stage_low_rank_from_cpu(
                ctx.gate_low_rank_cpu,
                manager,
                tag="S_gate_for_dB",
                stats=layer.stats,
            )
            dS_gate = _grouped_lora_cuda_view(
                grad_gate_stage,
                gate_lora_B.transpose(-1, -2),
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_gate_lora_B = _grouped_lora_weight_grads_torch(
                grad_gate_stage,
                gate_low_rank,
                offsets,
                experts,
                int(gate_lora_B.shape[0]),
                out_dtype=gate_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(gate_low_rank, drop_cache=True)
            manager.release_cpu(ctx.gate_low_rank_cpu)
            if need_grad_packed:
                grad_gate_lora_x = grouped_expert_lora(
                    dS_gate,
                    gate_lora_A.transpose(-1, -2),
                    offsets,
                    experts,
                    metadata=lora_metadata,
                )

        with prof_range("backward.mlp.activation_offload.up_lora"):
            up_low_rank = stage_low_rank_from_cpu(
                ctx.up_low_rank_cpu,
                manager,
                tag="S_up_for_dB",
                stats=layer.stats,
            )
            dS_up = _grouped_lora_cuda_view(
                grad_up_stage,
                up_lora_B.transpose(-1, -2),
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_up_lora_B = _grouped_lora_weight_grads_torch(
                grad_up_stage,
                up_low_rank,
                offsets,
                experts,
                int(up_lora_B.shape[0]),
                out_dtype=up_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(up_low_rank, drop_cache=True)
            manager.release_cpu(ctx.up_low_rank_cpu)
            if need_grad_packed:
                grad_up_lora_x = grouped_expert_lora(
                    dS_up,
                    up_lora_A.transpose(-1, -2),
                    offsets,
                    experts,
                    metadata=lora_metadata,
                )

        with prof_range("backward.mlp.activation_offload.gate_up_lora_a_grad"):
            x_handle, release_x_handle = _rebuild_qwen3_packed_x_cpu(ctx, manager)
            grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(
                dS_gate,
                dS_up,
                x_handle.tensor,
                offsets,
                experts,
                num_experts=int(gate_lora_A.shape[0]),
                stats=layer.stats,
            )
            if release_x_handle:
                manager.release_cpu(x_handle)
            manager.release_cpu(ctx.x_cpu)

        if need_grad_packed:
            with prof_range("backward.mlp.activation_offload.gate_up_base_dx"):
                grad_packed = _grouped_base_dx(
                    layer.gate_up_base,
                    grad_gate_up,
                    offsets,
                    experts,
                    dense_experts=True,
                    input_dtype=ctx.input_dtype,
                    profile_name=layer._profile_name("gate_up", "base"),
                )
            if grad_gate_lora_x is not None:
                grad_packed.add_(grad_gate_lora_x.to(dtype=grad_packed.dtype))
            if grad_up_lora_x is not None:
                grad_packed.add_(grad_up_lora_x.to(dtype=grad_packed.dtype))

        manager.release_stage(grad_gate_up, drop_cache=True)
        activation_offload_stats_pre_release = manager.snapshot()
        activation_offload_stats_pre_release["expert_lora_a_forward_mode"] = getattr(
            ctx,
            "expert_lora_a_forward_mode",
            "cpu",
        )
        activation_offload_stats_pre_release["qwen3_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
        activation_offload_stats_pre_release["qwen3_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
        final_cleanup_released_bytes = 0
        for handle in (
            ctx.x_cpu,
            ctx.gate_cpu,
            ctx.up_cpu,
            ctx.act_cpu,
            ctx.gate_low_rank_cpu,
            ctx.up_low_rank_cpu,
            ctx.down_low_rank_cpu,
            grad_act_cpu,
            grad_gate_cpu,
            grad_up_cpu,
        ):
            final_cleanup_released_bytes += manager.release_cpu(handle)
        activation_offload_stats = manager.snapshot()
        activation_offload_stats["expert_lora_a_forward_mode"] = getattr(ctx, "expert_lora_a_forward_mode", "cpu")
        activation_offload_stats["qwen3_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
        activation_offload_stats["qwen3_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
        activation_offload_stats["pre_final_cleanup_cpu_owned_bytes"] = activation_offload_stats_pre_release.get("cpu_owned_bytes", 0)
        activation_offload_stats["final_cleanup_released_bytes"] = final_cleanup_released_bytes
        layer._last_activation_offload_stats_pre_release = activation_offload_stats_pre_release
        layer._last_activation_offload_stats = activation_offload_stats

        return (
            grad_packed,
            None,
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
            (
                output,
                gate,
                up,
                activated,
                gate_low_rank,
                up_low_rank,
                down_low_rank,
                gate_mask_packed,
                up_mask_packed,
                down_mask_packed,
            ) = layer._forward_expert_body_with_intermediates(
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
            gate_low_rank_saved = gate_low_rank.detach()
            up_low_rank_saved = up_low_rank.detach()
        with prof_range("forward.mlp.expert_policy.save_activated_plan"):
            activated_saved, _, _, activated_saved_rows, activated_saved_mode = _make_group_row_plan(
                activated.detach(),
                offsets,
                experts,
                activated_save_groups,
                active_groups,
                mode_hint=activated_mode_hint,
            )
            down_low_rank_saved = down_low_rank.detach()
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
        ctx.lora_dropout_p = float(layer.lora_dropout_p)
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
                gate_mask_packed,
                up_mask_packed,
                down_mask_packed,
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
            gate_mask_packed,
            up_mask_packed,
            down_mask_packed,
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
            gate_low_rank_full = gate_low_rank_saved
            up_low_rank_full = up_low_rank_saved
            down_low_rank_full = down_low_rank_saved

        with prof_range("backward.mlp.expert_policy.pack_recompute_selected"):
            if ctx.recompute_mode == SAVE_FULL:
                recompute_packed = packed
                gate_low_rank_recompute = gate_low_rank_full
                up_low_rank_recompute = up_low_rank_full
            elif ctx.recompute_mode == SAVE_COMPACT:
                recompute_packed = packed.index_select(0, recompute_rows)
                gate_low_rank_recompute = gate_low_rank_full.index_select(0, recompute_rows)
                up_low_rank_recompute = up_low_rank_full.index_select(0, recompute_rows)
            else:
                recompute_packed = packed.new_empty((0, packed.shape[-1]))
                gate_low_rank_recompute = gate_low_rank_full.new_empty((0, layer.lora_rank))
                up_low_rank_recompute = up_low_rank_full.new_empty((0, layer.lora_rank))

        native_api = None
        use_native_selected = False
        if (
            _env_flag("ASYM_QWEN3_GATE_UP_WINDOWED_BWD", False)
            and need_grad_packed
            and ctx.recompute_mode != SAVE_EMPTY
            and int(recompute_packed.shape[0]) > 0
            and packed.device.type == "cuda"
            and packed.dtype == torch.bfloat16
            and isinstance(layer.gate_up_base, AsymGroupedFrozenLinear)
            and layer.gate_up_base.precision == "bf16"
            and layer.gate_up_base.host_weight.weight.is_pinned()
        ):
            try:
                import asym_gemm

                native_api = getattr(asym_gemm, "qwen3_gate_up_recompute_bwd_sm100_bf16_windowed", None)
                use_native_selected = native_api is not None and torch.cuda.get_device_capability(packed.device)[0] >= 10
            except Exception:
                native_api = None
                use_native_selected = False

        selected_rows = _selected_rows_for_mode(
            rows_total,
            recompute_rows,
            ctx.recompute_mode,
            device=packed.device,
        )
        nonselected_offsets, nonselected_experts, nonselected_rows, _ = _nonselected_group_plan(
            offsets,
            experts,
            recompute_experts,
            ctx.recompute_mode if use_native_selected else SAVE_EMPTY,
            rows_total=rows_total,
        )

        with prof_range("backward.mlp.expert_policy.recompute_gate_up_selected"):
            if ctx.recompute_mode != SAVE_EMPTY and not use_native_selected:
                gate_recompute, up_recompute = _forward_gate_up_with_saved_low_rank(
                    layer,
                    recompute_packed,
                    recompute_offsets,
                    recompute_experts,
                    gate_low_rank_recompute,
                    up_low_rank_recompute,
                    gate_lora_B,
                    up_lora_B,
                )
                if ctx.recompute_mode == SAVE_FULL:
                    gate_full = gate_recompute
                    up_full = up_recompute
                elif int(recompute_rows.shape[0]) > 0:
                    gate_full[recompute_rows] = gate_recompute
                    up_full[recompute_rows] = up_recompute

        with prof_range("backward.mlp.expert_policy.rebuild_activation_selected"):
            if not use_native_selected and ctx.activation_rebuild_mode == SAVE_FULL:
                activated_full = layer.act_fn(gate_full) * up_full
            elif ctx.activation_rebuild_mode == SAVE_COMPACT and int(activation_rows.shape[0]) > 0:
                if use_native_selected and int(selected_rows.numel()) > 0:
                    rebuild_mask = ~torch.isin(activation_rows, selected_rows)
                    rows_to_rebuild = activation_rows[rebuild_mask]
                else:
                    rows_to_rebuild = activation_rows
                if int(rows_to_rebuild.numel()) == 0:
                    rows_to_rebuild = activation_rows.new_empty((0,))
                gate_need = gate_full.index_select(0, rows_to_rebuild)
                up_need = up_full.index_select(0, rows_to_rebuild)
                activated_need = layer.act_fn(gate_need) * up_need
                activated_full[rows_to_rebuild] = activated_need

        with prof_range("backward.mlp.expert_policy.down_lora_backward"):
            if use_native_selected:
                grad_down_lora_x, grad_down_lora_A, grad_down_lora_B, dS_down_full = _grouped_down_lora_backward_split_loop_free(
                    activated_full,
                    grad_output,
                    down_lora_A,
                    down_lora_B,
                    offsets,
                    experts,
                    nonselected_offsets,
                    nonselected_experts,
                    nonselected_rows,
                    scale=layer.lora_scale,
                    precomputed_low_rank=down_low_rank_full,
                    dropout_mask_packed=down_mask_packed,
                    dropout_p=ctx.lora_dropout_p,
                    metadata=lora_metadata,
                    stats=layer.stats,
                )
            else:
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
                    dropout_mask_packed=down_mask_packed,
                    dropout_p=ctx.lora_dropout_p,
                    metadata=lora_metadata,
                    stats=layer.stats,
                )
                dS_down_full = None
            if grad_down_lora_x is not None:
                grad_activated.add_(
                    grad_down_lora_x if grad_down_lora_x.dtype == grad_activated.dtype else grad_down_lora_x.to(dtype=grad_activated.dtype)
                )

        native_grad_x_base_sel = None
        native_stats = None
        if use_native_selected:
            with prof_range("backward.mlp.expert_policy.gate_up_windowed_native_selected"):
                assert native_api is not None
                assert dS_down_full is not None
                dact_sel = grad_activated if ctx.recompute_mode == SAVE_FULL else grad_activated.index_select(0, recompute_rows)
                dS_down_sel = dS_down_full if ctx.recompute_mode == SAVE_FULL else dS_down_full.index_select(0, recompute_rows)
                down_mask_sel = _select_packed_mask_rows(down_mask_packed, recompute_rows, ctx.recompute_mode, packed.device)
                p_tile, q_tile, bm_tile, bk_tile, g_work_tile = _qwen3_window_tile_config(layer.intermediate_dim, layer.hidden_dim)
                native_result = native_api(
                    recompute_packed,
                    dact_sel.contiguous(),
                    gate_low_rank_recompute.contiguous(),
                    up_low_rank_recompute.contiguous(),
                    gate_lora_B,
                    up_lora_B,
                    layer.gate_up_base.host_weight.weight,
                    recompute_offsets,
                    recompute_experts,
                    dS_down_sel.contiguous(),
                    down_mask_sel.contiguous(),
                    float(ctx.lora_dropout_p),
                    p=p_tile,
                    q=q_tile,
                    bm=bm_tile,
                    bk=bk_tile,
                    g_work=g_work_tile,
                    lora_scale=float(layer.lora_scale),
                    mode="cache_first_window",
                    return_stats=True,
                )
                native_grad_x_base_sel, native_grad_gate_sel, native_grad_up_sel, native_grad_down_lora_A, native_stats = native_result
                grad_down_lora_A.add_(native_grad_down_lora_A.to(dtype=grad_down_lora_A.dtype))
                layer._last_gate_up_windowed_bwd_stats = dict(native_stats)
                layer.stats.asym_dx_calls += 1

        with prof_range("backward.mlp.expert_policy.activation_grad_silu"):
            if use_native_selected:
                grad_gate = torch.zeros((rows_total, layer.intermediate_dim), device=packed.device, dtype=grad_output.dtype)
                grad_up = torch.zeros_like(grad_gate)
                if int(nonselected_rows.numel()) > 0:
                    gate_nonselected = gate_full.index_select(0, nonselected_rows)
                    up_nonselected = up_full.index_select(0, nonselected_rows)
                    dact_nonselected = grad_activated.index_select(0, nonselected_rows)
                    grad_up_nonselected = dact_nonselected * F.silu(gate_nonselected)
                    grad_gate_input = dact_nonselected.mul(up_nonselected)
                    grad_gate_nonselected = torch.ops.aten.silu_backward(grad_gate_input, gate_nonselected)
                    grad_gate[nonselected_rows] = grad_gate_nonselected.to(dtype=grad_output.dtype)
                    grad_up[nonselected_rows] = grad_up_nonselected.to(dtype=grad_output.dtype)
                if ctx.recompute_mode == SAVE_FULL:
                    grad_gate.copy_(native_grad_gate_sel.to(dtype=grad_gate.dtype))
                    grad_up.copy_(native_grad_up_sel.to(dtype=grad_up.dtype))
                elif int(recompute_rows.shape[0]) > 0:
                    grad_gate[recompute_rows] = native_grad_gate_sel.to(dtype=grad_gate.dtype)
                    grad_up[recompute_rows] = native_grad_up_sel.to(dtype=grad_up.dtype)
            else:
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
                if use_native_selected:
                    grad_packed = torch.zeros((rows_total, layer.hidden_dim), device=packed.device, dtype=ctx.input_dtype)
                    if native_grad_x_base_sel is not None:
                        if ctx.recompute_mode == SAVE_FULL:
                            grad_packed.add_(native_grad_x_base_sel.to(dtype=grad_packed.dtype))
                        elif int(recompute_rows.shape[0]) > 0:
                            grad_packed[recompute_rows] = native_grad_x_base_sel.to(dtype=grad_packed.dtype)
                    if int(nonselected_rows.numel()) > 0:
                        grad_gate_up_nonselected = torch.empty(
                            (int(nonselected_rows.numel()), 2 * layer.intermediate_dim),
                            device=packed.device,
                            dtype=grad_gate.dtype,
                        )
                        grad_gate_up_nonselected[:, : layer.intermediate_dim] = grad_gate.index_select(0, nonselected_rows)
                        grad_gate_up_nonselected[:, layer.intermediate_dim :] = grad_up.index_select(0, nonselected_rows)
                        grad_nonselected = _grouped_base_dx(
                            layer.gate_up_base,
                            grad_gate_up_nonselected,
                            nonselected_offsets,
                            nonselected_experts,
                            dense_experts=False,
                            input_dtype=ctx.input_dtype,
                            profile_name=layer._profile_name("gate_up", "base"),
                        )
                        grad_packed[nonselected_rows] = grad_nonselected.to(dtype=grad_packed.dtype)
                else:
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
                dropout_mask_packed=gate_mask_packed,
                dropout_p=ctx.lora_dropout_p,
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
                dropout_mask_packed=up_mask_packed,
                dropout_p=ctx.lora_dropout_p,
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
        self.act_fn = _resolve_qwen3_expert_act_fn(source)
        if not callable(self.act_fn):
            raise TypeError(f"source expert module does not expose a supported activation: {type(source).__name__}")
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
        self._last_activation_offload_stats: dict[str, object] = {}
        self._weight_offload = None  # LoRAWeightOffloadCoordinator when JIT LoRA weight offload is installed

        base_dtype = torch.bfloat16
        if backend == "asym" and self.offload:
            if strict and (gate_up.device.type != "cpu" or down.device.type != "cpu"):
                raise RuntimeError("Qwen3 expert CPU offload requires CPU-first model loading")
            if strict and (gate_up.dtype != base_dtype or down.dtype != base_dtype):
                raise RuntimeError(
                    "Qwen3 expert CPU offload requires bf16 source weights: "
                    f"gate_up={gate_up.dtype}, down={down.dtype}"
                )
            self.gate_up_base = AsymGroupedFrozenLinear(
                gate_up.to(dtype=base_dtype),
                backend="asym",
                pin_memory=torch.cuda.is_available(),
                clone=False,
                precision=precision,
                stats=self.stats,
            )
            self.down_base = AsymGroupedFrozenLinear(
                down.to(dtype=base_dtype),
                backend="asym",
                pin_memory=torch.cuda.is_available(),
                clone=False,
                precision=precision,
                stats=self.stats,
            )
            def _hw_pinned_or_fabric(hw) -> bool:
                # asym_ep2 shared-fabric banks (fix_gb200_ep.md S1) are independent /dev/shm
                # copies that become pinned at the collective seal() BEFORE the first GEMM —
                # treat them as pinned here (both for the strict gate and the source release).
                return hw.weight.is_pinned() or bool(getattr(hw, "_fabric_bank", False))

            if strict and torch.cuda.is_available():
                if not _hw_pinned_or_fabric(self.gate_up_base.host_weight) or not _hw_pinned_or_fabric(self.down_base.host_weight):
                    raise RuntimeError("Qwen3 expert CPU offload requires pinned CPU HostWeights for AsymGEMM")
            # Release the duplicated source base weights now that the pinned HostWeight copies are
            # independent (pin_memory() always copies; the fabric path copies into /dev/shm). Frozen
            # experts would otherwise stay resident a second time (~1.2 GiB/layer, ~58 GiB total).
            # Gated on the copies actually being pinned (or fabric) so a silent pin failure
            # (HostWeight swallows the error) cannot drop weights still aliased.
            if _hw_pinned_or_fabric(self.gate_up_base.host_weight) and _hw_pinned_or_fabric(self.down_base.host_weight):
                for _src_attr in ("gate_up_proj", "down_proj"):
                    _src = getattr(source, _src_attr, None)
                    if isinstance(_src, torch.nn.Parameter):
                        _src.data = torch.empty(0, dtype=_src.dtype, device=_src.device)
                    elif _src is not None:
                        try:
                            setattr(source, _src_attr, None)
                        except Exception:
                            pass
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
        return_dropout_state: bool = False,
    ):
        if self.lora_dropout_p == 0.0:
            x_lora = x.to(dtype=self.lora_dtype)
            gate_mask_packed = _empty_packed_mask(x.device)
            up_mask_packed = _empty_packed_mask(x.device)
            gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)
            with prof_range(self._forward_range("gate_up", "lora_a")):
                low_rank = grouped_expert_lora(x_lora, gate_up_a, offsets, experts, metadata=metadata)
            gate_low_rank, up_low_rank = low_rank.split(self.lora_rank, dim=-1)
        elif return_dropout_state:
            gate_input, gate_mask_packed = _native_dropout_with_packed_mask(
                x,
                self.lora_dropout_p,
                out_dtype=self.lora_dtype,
            )
            up_input, up_mask_packed = _native_dropout_with_packed_mask(
                x,
                self.lora_dropout_p,
                out_dtype=self.lora_dtype,
            )
            with prof_range(self._forward_range("gate_up", "gate_lora_a")):
                gate_low_rank = grouped_expert_lora(gate_input, self.gate_lora_A, offsets, experts, metadata=metadata)
            with prof_range(self._forward_range("gate_up", "up_lora_a")):
                up_low_rank = grouped_expert_lora(up_input, self.up_lora_A, offsets, experts, metadata=metadata)
        else:
            gate_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            up_input = self.lora_dropout(x).to(dtype=self.lora_dtype)
            gate_mask_packed = _empty_packed_mask(x.device)
            up_mask_packed = _empty_packed_mask(x.device)
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
        if return_dropout_state:
            return gate_delta, up_delta, gate_low_rank, up_low_rank, gate_mask_packed, up_mask_packed
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
        return_dropout_state: bool = False,
    ):
        if return_dropout_state:
            x_lora, mask_packed = _native_dropout_with_packed_mask(
                x,
                self.lora_dropout_p,
                out_dtype=self.lora_dtype,
            )
        else:
            x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
            mask_packed = _empty_packed_mask(x.device)
        with prof_range(self._forward_range("down", "lora_a")):
            low_rank = grouped_expert_lora(x_lora, self.down_lora_A, offsets, experts, metadata=metadata)
        with prof_range(self._forward_range("down", "lora_b")):
            delta = grouped_expert_lora(low_rank, self.down_lora_B, offsets, experts, metadata=metadata)
        if self.lora_scale != 1.0:
            delta = delta.mul(self.lora_scale)
        if return_dropout_state:
            return delta, low_rank, mask_packed
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
        return_dropout_state: bool = False,
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
            return_dropout_state=return_dropout_state,
        )
        if return_dropout_state:
            gate_delta, up_delta, gate_low_rank, up_low_rank, gate_mask_packed, up_mask_packed = gate_up_lora
        elif return_low_rank:
            gate_delta, up_delta, gate_low_rank, up_low_rank = gate_up_lora
        else:
            gate_delta, up_delta = gate_up_lora
        gate = gate + gate_delta.to(dtype=gate.dtype)
        up = up + up_delta.to(dtype=up.dtype)
        if return_dropout_state:
            return gate, up, lora_metadata, gate_low_rank, up_low_rank, gate_mask_packed, up_mask_packed
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
        gate, up, lora_metadata, gate_low_rank, up_low_rank, gate_mask_packed, up_mask_packed = self._forward_gate_up(
            packed,
            offsets,
            experts,
            dense_experts=dense_experts,
            return_low_rank=True,
            return_dropout_state=True,
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
        down_delta, down_low_rank, down_mask_packed = self._forward_down_lora(
            activated,
            offsets,
            experts,
            lora_metadata,
            return_low_rank=True,
            return_dropout_state=True,
        )
        return (
            down + down_delta.to(dtype=down.dtype),
            gate,
            up,
            activated,
            gate_low_rank,
            up_low_rank,
            down_low_rank,
            gate_mask_packed,
            up_mask_packed,
            down_mask_packed,
        )

    def _uses_expert_gc(self) -> bool:
        config = self.expert_recompute_config
        return bool(config.torch_checkpoint_enabled and self.training and torch.is_grad_enabled())

    def _uses_expert_recompute(self) -> bool:
        config = self.expert_recompute_config
        return bool(config.custom_autograd_enabled and self.training and torch.is_grad_enabled())

    def _expert_gc_use_reentrant(self) -> bool:
        default = _env_flag("ASYM_EXPERT_GC_REENTRANT", True)
        return _env_flag("ASYM_EXPERT_GC_USE_REENTRANT", default)

    def _lora_banks(self) -> tuple[torch.nn.Parameter, ...]:
        return (
            self.gate_lora_A,
            self.gate_lora_B,
            self.up_lora_A,
            self.up_lora_B,
            self.down_lora_A,
            self.down_lora_B,
        )

    def gather_lora_weights(self) -> None:
        """Stage this layer's LoRA banks CPU->GPU (single H2D) when JIT weight offload is on."""
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.gather_group(self)

    def release_lora_weights(self) -> None:
        """Release this layer's LoRA banks to 0-size CUDA placeholders (frees the staged slab)."""
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            for param in self._lora_banks():
                coordinator.release(param)

    def _asym_weight_offload_release_after_forward(self) -> bool:
        if not self.training or not torch.is_grad_enabled():
            return True
        # Plain expert autograd reads trainable LoRA banks through regular torch
        # ops, so AccumulateGrad needs the parameters to stay full-shaped until
        # the optimizer's post-accumulate hooks release them. The custom expert
        # paths below gather in their backward and can release after forward.
        return bool(
            self._uses_qwen3_moe_finegrained_offload()
            or self._uses_activation_offload()
            or self._uses_expert_gc()
            or self._uses_expert_recompute()
        )

    def _uses_qwen3_moe_finegrained_offload(self) -> bool:
        return bool(
            getattr(self, "_qwen3_moe_finegrained_enabled", False)
            and self.training
            and torch.is_grad_enabled()
        )

    def _uses_qwen3_moe_finegrained_nograd_forward(self) -> bool:
        return bool(
            getattr(self, "_qwen3_moe_finegrained_enabled", False)
            and self.training
            and not torch.is_grad_enabled()
        )

    def _ensure_qwen3_moe_finegrained_bases(self) -> tuple[AsymGroupedFrozenLinear, AsymGroupedFrozenLinear]:
        gate_base = getattr(self, "_qwen3_moe_finegrained_gate_base", None)
        up_base = getattr(self, "_qwen3_moe_finegrained_up_base", None)
        if isinstance(gate_base, AsymGroupedFrozenLinear) and isinstance(up_base, AsymGroupedFrozenLinear):
            return gate_base, up_base
        if not isinstance(self.gate_up_base, AsymGroupedFrozenLinear):
            raise NotImplementedError("Qwen3 MoE fine-grained offload requires an AsymGroupedFrozenLinear gate/up base")
        fused = self.gate_up_base.host_weight.weight
        gate_weight = fused[:, : self.intermediate_dim, :].contiguous()
        up_weight = fused[:, self.intermediate_dim :, :].contiguous()
        gate_base = AsymGroupedFrozenLinear(
            gate_weight,
            backend=self.gate_up_base.backend,
            pin_memory=torch.cuda.is_available(),
            clone=False,
            precision=self.gate_up_base.precision,
            stats=self.stats,
            compiled_dims=self.gate_up_base.compiled_dims,
            bf16_output_dtype=self.gate_up_base.bf16_output_dtype,
            weight_layout=self.gate_up_base.weight_layout,
        )
        up_base = AsymGroupedFrozenLinear(
            up_weight,
            backend=self.gate_up_base.backend,
            pin_memory=torch.cuda.is_available(),
            clone=False,
            precision=self.gate_up_base.precision,
            stats=self.stats,
            compiled_dims=self.gate_up_base.compiled_dims,
            bf16_output_dtype=self.gate_up_base.bf16_output_dtype,
            weight_layout=self.gate_up_base.weight_layout,
        )
        self._qwen3_moe_finegrained_gate_base = gate_base
        self._qwen3_moe_finegrained_up_base = up_base
        # The full-fg path runs every base GEMM (fwd, nograd-fwd, and backward dX)
        # through the SPLIT gate/up bases above — the fused [E, 2I, H] home is then
        # only read by capability checks, yet its pinned copy stays resident: with
        # the split copies that is 5*I*H per expert instead of 3 (measured 5/3 of
        # the experts' bf16 bytes on qwen3-30b/35B/122B; +144 GiB host at 122B).
        # ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME frees the fused pinned storage once
        # the splits are pinned. DEFAULT ON since 2026-07-15 (clear bug fix; set 0
        # to keep the old duplicated-home behavior).
        # NB: the LF driver forwards unset knobs as EMPTY strings — empty must mean
        # "default (on)", not "off" (same convention as DOWN_DX_STAGED's reader).
        _release_raw = os.environ.get("ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME")
        if _release_raw is None or _release_raw.strip() == "":
            _release_raw = os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_RELEASE_FUSED_HOME", "")
        _release_on = (_release_raw or "").strip().lower() not in {"0", "false", "no", "off"}
        if (
            _release_on
            # Never release a shared-fabric bank view (multi-rank arena-shm backends):
            # the fused "home" there is a view into SharedFabric's persistent bank —
            # swapping _tensor frees nothing and desyncs the bank's accounting.
            and not getattr(self.gate_up_base.host_weight, "_fabric_bank", False)
            and gate_base.host_weight.weight.is_pinned()
            and up_base.host_weight.weight.is_pinned()
        ):
            fused_hw = self.gate_up_base.host_weight
            released = torch.empty(0, dtype=fused_hw.weight.dtype, device="cpu")
            fused_hw._tensor = released
            # Zero the byte telemetry (nbytes -> pinned_cpu_bytes/weight_nbytes) so the
            # freed fused bytes stop being reported as pinned-resident — otherwise the
            # memory summaries hide exactly the savings this release delivers. is_pinned
            # is deliberately LEFT TRUE: _load_from_state_dict (frozen_linear.py) reads
            # it as the pin intent for a reloaded fused weight, and a good pre-release
            # checkpoint restored onto a released module must come back pinned.
            fused_hw._metadata = _dataclass_replace(fused_hw._metadata, nbytes=0)
            setattr(fused_hw, "_asym_released_fused_home", True)
            setattr(self.gate_up_base, "_asym_released_fused_home", True)
            self.stats.qwen3_moe_finegrained_fused_home_released = (
                getattr(self.stats, "qwen3_moe_finegrained_fused_home_released", 0) + 1
            )
        return gate_base, up_base

    def _qwen3_moe_finegrained_unsupported_reasons(self, packed: torch.Tensor) -> list[str]:
        from .qwen3_moe_finegrained import qwen3_moe_finegrained_unsupported_reasons

        return qwen3_moe_finegrained_unsupported_reasons(self, packed)

    def _check_qwen3_moe_finegrained_supported(self, packed: torch.Tensor) -> None:
        reasons = self._qwen3_moe_finegrained_unsupported_reasons(packed)
        if reasons:
            joined = "; ".join(dict.fromkeys(reasons))
            raise NotImplementedError(f"Qwen3 MoE fine-grained offload is unsupported for this configuration: {joined}")

    def _forward_qwen3_moe_finegrained_offload(
        self,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        token_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        input_weighted: bool,
        output_weighted: bool,
    ) -> torch.Tensor:
        from .qwen3_moe_finegrained import qwen3_moe_finegrained_forward

        if routing_weights.requires_grad:
            raise NotImplementedError("Qwen3 MoE fine-grained offload requires detached router weights")
        self._check_qwen3_moe_finegrained_supported(hidden_states)
        return qwen3_moe_finegrained_forward(
            self,
            hidden_states,
            offsets,
            experts,
            token_indices,
            routing_weights,
            input_weighted=bool(input_weighted),
            output_weighted=bool(output_weighted),
        )

    def _forward_qwen3_moe_finegrained_nograd(
        self,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        token_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        input_weighted: bool,
        output_weighted: bool,
    ) -> torch.Tensor:
        from .qwen3_moe_finegrained import qwen3_moe_finegrained_nograd_forward

        if routing_weights.requires_grad:
            raise NotImplementedError("Qwen3 MoE fine-grained no-grad forward requires detached router weights")
        self._check_qwen3_moe_finegrained_supported(hidden_states)
        return qwen3_moe_finegrained_nograd_forward(
            self,
            hidden_states,
            offsets,
            experts,
            token_indices,
            routing_weights,
            input_weighted=bool(input_weighted),
            output_weighted=bool(output_weighted),
        )

    def _activation_offload_requested(self) -> bool:
        return _env_flag("ASYMM_EXPERT_ACT_OFFLOAD", False)

    def _uses_activation_offload(self) -> bool:
        return bool(self._activation_offload_requested() and self.training and torch.is_grad_enabled())

    def _activation_offload_unsupported_reasons(self, packed: torch.Tensor) -> list[str]:
        reasons: list[str] = []
        if self.backend != "asym" or not self.offload:
            reasons.append("requires backend='asym' with expert base CPU offload")
        if not isinstance(self.gate_up_base, AsymGroupedFrozenLinear) or not isinstance(self.down_base, AsymGroupedFrozenLinear):
            reasons.append("requires AsymGroupedFrozenLinear gate/up and down bases")
        if self.expert_recompute_config.enabled:
            reasons.append("expert recompute/activation-drop policy must be disabled")
        if self.lora_dropout_p != 0.0:
            reasons.append("v0 supports lora_dropout=0.0 only")
        if not _is_silu_activation(self.act_fn):
            reasons.append("requires SiLU activation")
        if packed.device.type != "cuda":
            reasons.append("requires CUDA packed expert input")
        if packed.dtype != torch.bfloat16:
            reasons.append(f"requires bf16 packed expert input, got {packed.dtype}")
        if not packed.is_contiguous():
            reasons.append("requires contiguous packed expert input")
        if self.lora_dtype != packed.dtype:
            reasons.append(f"requires LoRA dtype to match packed dtype, got {self.lora_dtype} vs {packed.dtype}")
        if self.hidden_dim % 64 != 0 or self.intermediate_dim % 64 != 0:
            reasons.append("requires hidden_dim and intermediate_dim multiples of 64 for CPU-right transpose AsymGEMM")
        try:
            lora_a_forward_mode = _expert_act_offload_lora_a_fwd_mode()
        except ValueError as exc:
            reasons.append(str(exc))
            lora_a_forward_mode = "cpu"
        if lora_a_forward_mode == "hbm":
            try:
                _require_lora_grouped_mm()
            except RuntimeError as exc:
                reasons.append(str(exc))
        for name, param in (
            ("gate_lora_A", self.gate_lora_A),
            ("gate_lora_B", self.gate_lora_B),
            ("up_lora_A", self.up_lora_A),
            ("up_lora_B", self.up_lora_B),
            ("down_lora_A", self.down_lora_A),
            ("down_lora_B", self.down_lora_B),
        ):
            if param.device != packed.device:
                reasons.append(f"{name} must be on {packed.device}, got {param.device}")
            if param.dtype != packed.dtype:
                reasons.append(f"{name} must have dtype {packed.dtype}, got {param.dtype}")
            if not param.is_contiguous():
                reasons.append(f"{name} must be contiguous")
        if isinstance(self.gate_up_base, AsymGroupedFrozenLinear):
            if self.gate_up_base.precision != "bf16" or self.gate_up_base.backend != "asym":
                reasons.append("gate/up base must use direct bf16 AsymGEMM")
            if torch.cuda.is_available() and not self.gate_up_base.host_weight.weight.is_pinned():
                reasons.append("gate/up host weight must be pinned CPU memory")
        if isinstance(self.down_base, AsymGroupedFrozenLinear):
            if self.down_base.precision != "bf16" or self.down_base.backend != "asym":
                reasons.append("down base must use direct bf16 AsymGEMM")
            if torch.cuda.is_available() and not self.down_base.host_weight.weight.is_pinned():
                reasons.append("down host weight must be pinned CPU memory")
        kernel_reason = require_expert_activation_offload_kernels(scope="full", check_only=True)
        if kernel_reason is not None:
            reasons.append(kernel_reason)
        return reasons

    def _check_activation_offload_supported(self, packed: torch.Tensor) -> None:
        reasons = self._activation_offload_unsupported_reasons(packed)
        if reasons:
            joined = "; ".join(dict.fromkeys(reasons))
            raise NotImplementedError(f"Qwen3 expert activation offload is unsupported for this configuration: {joined}")

    def _forward_expert_activation_offload(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        x_src_hidden: torch.Tensor | None = None,
        x_token_indices: torch.Tensor | None = None,
        x_route_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_activation_offload_supported(packed)
        return _ActivationOffloadQwen3ExpertFunction.apply(
            packed,
            offsets,
            experts,
            x_src_hidden,
            x_token_indices,
            x_route_scale,
            self.gate_lora_A,
            self.gate_lora_B,
            self.up_lora_A,
            self.up_lora_B,
            self.down_lora_A,
            self.down_lora_B,
            self,
        )

    def _forward_expert_gc(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
    ) -> torch.Tensor:
        if self.lora_dropout_p >= 1.0:
            raise NotImplementedError("gc-exp requires 0.0 <= lora_dropout < 1.0")
        if not packed.is_floating_point():
            raise NotImplementedError("gc-exp requires floating packed expert input")
        if not packed.requires_grad and any(param.requires_grad for param in self.parameters()):
            packed = packed.requires_grad_(True)

        def expert_body(packed_arg: torch.Tensor) -> torch.Tensor:
            if getattr(self, "_weight_offload", None) is not None:
                # Checkpoint recompute calls this closure directly, bypassing module
                # forward pre-hooks. Gather here so released 0-size LoRA placeholders
                # are restored before grouped LoRA GEMMs run. Do not release here:
                # original forward is released by the module hook, and recompute
                # backward is released by the post-accumulate grad hook.
                self.gather_lora_weights()
            return self._forward_expert_body(
                packed_arg,
                offsets,
                experts,
                dense_experts=True,
            )

        with prof_range(self._forward_range("expert_gc")):
            return checkpoint(
                expert_body,
                packed,
                use_reentrant=self._expert_gc_use_reentrant(),
                preserve_rng_state=True,
            )

    def _forward_expert_policy(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        config: ExpertRecomputeConfig = self.expert_recompute_config
        if self.lora_dropout_p >= 1.0:
            raise NotImplementedError("Qwen3 expert recompute supports 0.0 <= lora_dropout < 1.0")
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
        if not getattr(self, "ep_vanilla_a2a", False):
            return self._forward_impl(hidden_states, top_k_index, top_k_weights)
        # S5b VANILLA-EP rung (ep_vanilla.py): allgather-dispatch -> owned-slice
        # partial over GLOBAL tokens -> reduce-scatter combine. Collectives are
        # unconditional per call so both ranks stay lockstep through GC recompute.
        # Megatron-grade sync schedule: the BIG
        # hidden allgather is deferred via hidden_provider until AFTER the layer's
        # host reads (slice bounds + pad prewarm) — no CPU ever blocks on the peer
        # mid-layer, which was the entire stagger pathology. Legacy order kept
        # behind ASYM_EP_VANILLA_LEGACY_ORDER=1 for the A/B receipt.
        import os as _os

        from .ep_vanilla import (
            gather_moe_hidden,
            gather_moe_inputs,
            gather_moe_routing,
            reduce_scatter_partial,
        )

        from .frozen_linear import pad_memo_context

        local_tokens = hidden_states.shape[0]
        if _os.environ.get("ASYM_EP_VANILLA_LEGACY_ORDER") == "1":
            hidden_g, idx_g, w_g = gather_moe_inputs(hidden_states, top_k_index, top_k_weights)
            partial = self._forward_impl(hidden_g, idx_g, w_g)
        else:
            idx_g, w_g = gather_moe_routing(top_k_index, top_k_weights)
            # fix_ep: one pad .item() per LAYER — the context lets every grouped
            # call of this invocation (fg entries rebuild offsets tensors) reuse
            # the prewarm's metadata by VALUE key instead of tensor identity.
            with pad_memo_context():
                partial = self._forward_impl(
                    hidden_states, idx_g, w_g,
                    hidden_provider=lambda: gather_moe_hidden(hidden_states),
                )
        return reduce_scatter_partial(partial, local_tokens)

    def _forward_impl(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, hidden_provider=None) -> torch.Tensor:
        # fix_ep D1: with hidden_provider set (vanilla EP), `hidden_states` arrives
        # as the LOCAL tensor (dtype/device donors only) and the GLOBAL gathered
        # hidden is produced by the provider strictly AFTER the metadata host
        # reads below — the layer's contract is "no host sync after the big
        # collective enqueues".
        input_dtype = hidden_states.dtype
        ep_range = getattr(self, "ep_expert_range", None)
        with prof_range(self._forward_range("route_metadata")):
            metadata = build_contiguous_route_metadata(
                top_k_index, top_k_weights,
                num_experts=getattr(self, "ep_num_experts_full", None) or self.num_experts,
            )
            if ep_range is not None:
                # sEP static EP-2 (gb200_ep.md E1/tp.md I7): contiguous slice to this
                # device's expert range; scatter output becomes the device-local PARTIAL.
                from .stp_moe import ep_slice_route_metadata

                metadata = ep_slice_route_metadata(metadata, ep_range[0], ep_range[1])
                if metadata.num_routes == 0:
                    # no tokens routed to this device's experts this microbatch: the
                    # partial is exactly zero (dX contribution zero; combine adds peer's).
                    # The peer STILL enqueues its allgather — ours must too (NCCL
                    # collectives must stay call-count aligned) even though the
                    # result is unused.
                    if hidden_provider is not None:
                        hidden_provider()
                    return hidden_states.new_zeros(metadata.num_tokens, self.hidden_dim)
        with prof_range(self._forward_range("route_metadata", "dense_groups")):
            offsets, experts = make_dense_group_metadata(
                metadata.expert_offsets,
                num_groups=self.num_experts,
                device=hidden_states.device,
            )
        if hidden_provider is not None:
            # fix_ep D2: the pad memo's one .item() fires NOW (pre-collective,
            # drains µs of tiny-gather work) via a zero-width dummy through the
            # production padder; every later grouped call hits the memo.
            from .frozen_linear import prewarm_pad_memo

            prewarm_pad_memo(offsets, experts, int(metadata.num_routes),
                             device=hidden_states.device, dtype=hidden_states.dtype)
            with prof_range(self._forward_range("vanilla_hidden_allgather")):
                hidden_states = hidden_provider()
        if self._uses_qwen3_moe_finegrained_offload():
            return self._forward_qwen3_moe_finegrained_offload(
                hidden_states,
                offsets,
                experts,
                metadata.token_indices,
                metadata.routing_weights,
                input_weighted=False,
                output_weighted=True,
            ).to(dtype=input_dtype)
        if self._uses_qwen3_moe_finegrained_nograd_forward():
            return self._forward_qwen3_moe_finegrained_nograd(
                hidden_states,
                offsets,
                experts,
                metadata.token_indices,
                metadata.routing_weights,
                input_weighted=False,
                output_weighted=True,
            ).to(dtype=input_dtype)
        with prof_range(self._forward_range("pack_tokens")):
            packed = pack_tokens_contiguous(hidden_states, metadata)
        if self._uses_activation_offload():
            x_src_hidden = None
            x_token_indices = None
            if _expert_act_offload_x_unpacked():
                x_src_hidden = hidden_states.reshape(metadata.num_tokens, -1)
                x_token_indices = metadata.token_indices
            down = self._forward_expert_activation_offload(
                packed,
                offsets,
                experts,
                x_src_hidden=x_src_hidden,
                x_token_indices=x_token_indices,
                x_route_scale=None,
            )
        elif self._uses_expert_gc():
            down = self._forward_expert_gc(packed, offsets, experts)
        elif self._uses_expert_recompute():
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
        with prof_range(self._forward_range("route_metadata", "dense_groups")):
            offsets, experts = make_dense_group_metadata(
                metadata.expert_offsets,
                num_groups=self.num_experts,
                device=hidden_states.device,
            )
        if self._uses_qwen3_moe_finegrained_offload():
            return self._forward_qwen3_moe_finegrained_offload(
                hidden_states,
                offsets,
                experts,
                metadata.token_indices,
                metadata.routing_weights,
                input_weighted=True,
                output_weighted=False,
            ).to(dtype=input_dtype)
        if self._uses_qwen3_moe_finegrained_nograd_forward():
            return self._forward_qwen3_moe_finegrained_nograd(
                hidden_states,
                offsets,
                experts,
                metadata.token_indices,
                metadata.routing_weights,
                input_weighted=True,
                output_weighted=False,
            ).to(dtype=input_dtype)
        with prof_range(self._forward_range("pack_tokens")):
            packed = pack_tokens_contiguous(hidden_states, metadata)
            route_scale = metadata.routing_weights.reshape(metadata.num_routes, *([1] * (packed.dim() - 1)))
            packed = packed * route_scale.to(device=packed.device, dtype=packed.dtype)
        if self._uses_activation_offload():
            x_src_hidden = None
            x_token_indices = None
            x_route_scale = None
            if _expert_act_offload_x_unpacked():
                x_src_hidden = hidden_states.reshape(metadata.num_tokens, -1)
                x_token_indices = metadata.token_indices
                x_route_scale = metadata.routing_weights
            down = self._forward_expert_activation_offload(
                packed,
                offsets,
                experts,
                x_src_hidden=x_src_hidden,
                x_token_indices=x_token_indices,
                x_route_scale=x_route_scale,
            )
        elif self._uses_expert_gc():
            down = self._forward_expert_gc(packed, offsets, experts)
        elif self._uses_expert_recompute():
            down = self._forward_expert_policy(packed, offsets, experts, metadata)
        else:
            down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
        with prof_range(self._forward_range("scatter_combine")):
            return _scatter_contiguous_sum(down, metadata).to(dtype=input_dtype)


class AsymQwen3Router(nn.Module):
    """Qwen3/Qwen3.5 top-k router with a CPU-resident frozen projection."""

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
            raise TypeError(f"AsymQwen3Router requires a 2D router weight, got {type(source).__name__}")
        if strict and weight.device.type != "cpu":
            raise RuntimeError("Qwen3 router CPU offload requires CPU-first model loading")
        if strict and weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Qwen3 router CPU offload requires bf16 source weight, got {weight.dtype}")
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
        self.hidden_dim = int(getattr(source, "hidden_dim", self.proj.in_features))
        self.num_experts = int(getattr(source, "num_experts", self.proj.out_features))
        self.top_k = int(getattr(source, "top_k"))
        self.norm_topk_prob = bool(getattr(source, "norm_topk_prob", True))

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.proj.bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.proj(hidden_states)
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class _EpBalanceStats:
    """gb200_ep.md E0: per-layer expert token histograms + hypothetical static-E/2 device
    shares, accumulated across the run and dumped at exit (ASYM_EP_STATS=1;
    path via ASYM_EP_STATS_PATH). Cheap: one bincount per MoE forward."""

    def __init__(self) -> None:
        self.counts: dict[str, torch.Tensor] = {}
        self.dev_share_max: dict[str, list[float]] = {}
        self.calls = 0
        import atexit

        atexit.register(self.dump)

    def record(self, key: str, top_k_index: torch.Tensor, num_experts: int) -> None:
        counts = torch.bincount(top_k_index.reshape(-1), minlength=num_experts).cpu()
        acc = self.counts.get(key)
        self.counts[key] = counts if acc is None else acc + counts
        half = num_experts // 2
        d0 = int(counts[:half].sum())
        total = int(counts.sum())
        if total > 0:
            self.dev_share_max.setdefault(key, []).append(max(d0, total - d0) / total)
        self.calls += 1

    def dump(self) -> None:
        if not self.counts:
            return
        path = os.environ.get("ASYM_EP_STATS_PATH", "ep_balance_stats.json")
        try:
            import json

            layers = {}
            for key, counts in sorted(self.counts.items()):
                c = counts.to(torch.float64)
                total = float(c.sum())
                if total <= 0:
                    continue
                shares = self.dev_share_max.get(key, [])
                layers[key] = {
                    "tokens": int(total),
                    "hottest_expert_share": float(c.max() / total),
                    "static_e2_device_share_mean": (sum(shares) / len(shares)) if shares else None,
                    "static_e2_device_share_max": max(shares) if shares else None,
                    "counts": [int(x) for x in counts.tolist()],
                }
            agg = [v["static_e2_device_share_max"] for v in layers.values() if v["static_e2_device_share_max"]]
            report = {
                "calls": self.calls,
                "skew_hot": _EP_SKEW_HOT,
                "skew_zipf": _EP_SKEW_ZIPF,
                "loss_invalid": _EP_SKEW_HOT > 0 or _EP_SKEW_ZIPF is not None,
                "worst_layer_static_e2_device_share": max(agg) if agg else None,
                "layers": layers,
            }
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - atexit best-effort
            print(f"[asym-ep] stats dump failed: {exc}")


_EP_SKEW_HOT = float(os.environ.get("ASYM_EP_SKEW_HOT", "0") or 0.0)
_EP_SKEW_ZIPF_RAW = (os.environ.get("ASYM_EP_SKEW_ZIPF") or "").strip()
_EP_SKEW_ZIPF: float | None = float(_EP_SKEW_ZIPF_RAW) if _EP_SKEW_ZIPF_RAW else None
_EP_STATS = _EpBalanceStats() if os.environ.get("ASYM_EP_STATS") == "1" else None
if _EP_SKEW_HOT > 0 and os.environ.get("ASYM_EP_SKEW_ACK") != "1":
    raise RuntimeError(
        "ASYM_EP_SKEW_HOT forces routing (loss-INVALID, timing-only rows; HC-EP1). "
        "Set ASYM_EP_SKEW_ACK=1 to acknowledge."
    )
if _EP_SKEW_ZIPF is not None:
    if os.environ.get("ASYM_EP_SKEW_ACK") != "1":
        raise RuntimeError(
            "ASYM_EP_SKEW_ZIPF forces routing (loss-INVALID, timing-only rows). "
            "Set ASYM_EP_SKEW_ACK=1 to acknowledge."
        )
    if _EP_SKEW_HOT > 0:
        raise RuntimeError("ASYM_EP_SKEW_HOT and ASYM_EP_SKEW_ZIPF are mutually exclusive")
    if _EP_SKEW_ZIPF < 0:
        raise RuntimeError(f"ASYM_EP_SKEW_ZIPF must be >= 0, got {_EP_SKEW_ZIPF}")

_SKEW_SEED = 42  # fixed by design (2026-07-08): not configurable, not in run labels
_SKEW_HOT_CACHE: dict[str, int] = {}


def _skew_hot_expert_for_layer(layer_key: str, num_experts: int) -> int:
    """Deterministic per-layer hot-expert target: sha256(seed, layer name) % E.
    Same on every rank and across fwd/GC-recompute; varies across layers so the
    forced hotspot lands on different owner-halves (no always-expert-0 bias)."""
    hot = _SKEW_HOT_CACHE.get(layer_key)
    if hot is None:
        import hashlib

        digest = hashlib.sha256(f"{_SKEW_SEED}:{layer_key}".encode()).digest()
        hot = int.from_bytes(digest[:8], "little") % max(1, num_experts)
        _SKEW_HOT_CACHE[layer_key] = hot
    return hot


_SKEW_ZIPF_CACHE: dict[str, tuple[int, torch.Tensor]] = {}


def _skew_zipf_state_for_layer(layer_key: str, num_experts: int) -> tuple[int, torch.Tensor]:
    """Deterministic per-layer Zipf state: (draw seed, per-expert-ID weight vector).
    Popularity ranks (share ~ 1/rank^s) are assigned to a fixed seed-42 shuffle of
    the expert IDs — identical on every rank and across fwd/GC-recompute, and the
    hot experts land on different owner-halves across layers."""
    state = _SKEW_ZIPF_CACHE.get(layer_key)
    if state is None:
        import hashlib

        digest = hashlib.sha256(f"{_SKEW_SEED}:zipf:{layer_key}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") >> 1
        gen = torch.Generator()
        gen.manual_seed(seed)
        perm = torch.randperm(num_experts, generator=gen)
        shares = torch.arange(1, num_experts + 1, dtype=torch.float64).pow_(-float(_EP_SKEW_ZIPF or 0.0))
        weights = torch.empty(num_experts, dtype=torch.float32)
        weights[perm] = (shares / shares.sum()).to(torch.float32)
        state = (seed, weights)
        _SKEW_ZIPF_CACHE[layer_key] = state
    return state


class AsymQwen3MoeBlock(nn.Module):
    """Qwen3 MoE block wrapper that owns frozen router execution."""

    _is_asym_qwen3_moe_block = True

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
            raise ValueError(f"AsymQwen3MoeBlock only implements router_mode='whole', got {router_mode!r}")
        if strict and not is_qwen3_moe_block(source):
            source_file = inspect.getsourcefile(type(source)) or "unknown"
            raise TypeError(
                "source does not look like a Qwen3 MoE block with gate/expert routing: "
                f"{type(source).__name__} from {source_file}"
            )

        self.config = getattr(source, "config", None)
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.router_mode = router_mode
        self.router_debug_grad = bool(router_debug_grad)
        self.profile_prefix = "layers.unknown.mlp"

        # Preserve the installed Qwen3 module order: gate first, experts second.
        source_gate = getattr(source, "gate")
        self.gate = (
            AsymQwen3Router(source_gate, backend=backend, precision=precision, stats=stats, strict=strict)
            if backend == "asym" and offload_router
            else source_gate
        )
        self.experts = wrap_qwen3_experts(
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

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def _compute_routing(self, hidden_states_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        context = nullcontext() if self.router_debug_grad else torch.no_grad()
        with context, prof_range(self._forward_range("router")):
            router_out = self.gate(hidden_states_2d)

        if isinstance(router_out, tuple) and len(router_out) >= 3:
            top_k_weights = router_out[1]
            top_k_index = router_out[2]
            if not self.router_debug_grad:
                top_k_weights = top_k_weights.detach()
                top_k_index = top_k_index.detach()
            if top_k_weights.dtype != hidden_states_2d.dtype:
                top_k_weights = top_k_weights.to(dtype=hidden_states_2d.dtype)
            if _EP_SKEW_HOT > 0.0:
                # HC-EP1: synthetic hot-expert skew AFTER detach — forward values garbage,
                # kernel work real. First ceil(alpha * slots) routed slots -> ONE hot
                # expert chosen PER LAYER by a fixed-seed(42) hash of the layer name:
                # deterministic, identical on every rank and across fwd/recompute, and
                # the hotspot lands on different owner-halves across layers (no
                # always-rank0 bias). Seed is intentionally NOT configurable.
                flat_idx = top_k_index.reshape(-1)
                n_force = int(_EP_SKEW_HOT * flat_idx.numel())
                if n_force > 0:
                    num_experts = getattr(self.config, "num_experts", None) or int(
                        getattr(self.gate, "num_experts", 0)
                    )
                    hot = _skew_hot_expert_for_layer(self.profile_prefix, int(num_experts))
                    top_k_index = top_k_index.clone()
                    top_k_index.reshape(-1)[:n_force] = hot
            if _EP_SKEW_ZIPF is not None:
                # Paper-standard Zipf skew AFTER detach — forward values garbage, kernel
                # work real. Expert popularity ~ 1/rank^s over a fixed seed-42 per-layer
                # ID shuffle; every token draws top_k DISTINCT experts (multinomial
                # without replacement), so routing stays legal at any s and the busiest
                # expert self-saturates toward top_k/E. The generator is re-seeded per
                # call from (seed, layer): fwd and GC-recompute draw identical picks.
                num_experts = getattr(self.config, "num_experts", None) or int(
                    getattr(self.gate, "num_experts", 0)
                )
                seed, weights = _skew_zipf_state_for_layer(self.profile_prefix, int(num_experts))
                dev = top_k_index.device
                gen = torch.Generator(device=dev)
                gen.manual_seed(seed)
                probs = weights.to(dev).expand(top_k_index.shape[0], -1)
                top_k_index = torch.multinomial(
                    probs, top_k_index.shape[-1], replacement=False, generator=gen
                ).to(dtype=top_k_index.dtype)
            if _EP_STATS is not None:
                num_experts = getattr(self.config, "num_experts", None) or int(top_k_index.max()) + 1
                _EP_STATS.record(self.profile_prefix, top_k_index, int(num_experts))
            return top_k_index, top_k_weights, None

        raise TypeError(
            "AsymQwen3MoeBlock requires a Qwen3MoeTopKRouter-style gate returning "
            "(router_logits, top_k_weights, top_k_index); "
            f"got {type(router_out).__name__}"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        if hidden_states.dim() != 3:
            raise ValueError(f"AsymQwen3MoeBlock expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        flat = hidden_states.view(-1, input_shape[-1])
        top_k_index, top_k_weights, _router_logits = self._compute_routing(flat)
        if not self.router_debug_grad and top_k_weights.requires_grad:
            raise RuntimeError("router no-grad mode produced differentiable top_k_weights")
        with prof_range(self._forward_range("experts")):
            out = self.experts(flat, top_k_index, top_k_weights)
        return out.view(input_shape)


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


def wrap_qwen3_moe_block(
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
) -> AsymQwen3MoeBlock:
    return AsymQwen3MoeBlock(
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
    "AsymQwen3Experts",
    "AsymQwen3MoeBlock",
    "AsymQwen3Router",
    "Qwen3ExpertReport",
    "is_qwen3_experts",
    "is_qwen3_moe_block",
    "wrap_qwen3_experts",
    "wrap_qwen3_moe_block",
]
