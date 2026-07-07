from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from .frozen_linear import AsymGroupedFrozenLinear, _pad_grouped_input_for_asym, _unpad_grouped_output


_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in _TRUE_VALUES


def route_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() in _TRUE_VALUES
    return env_flag("ASYMM_QWEN3_MOE_ROUTE_MAPPED_GEMM", False)


@dataclass(frozen=True)
class RoutedKernelFlags:
    fwd_scatter: bool
    down_dx_gather: bool
    gateup_dx_scatter: bool
    lora: bool
    accum_dtype: str
    debug: bool

    @property
    def any_base(self) -> bool:
        return self.fwd_scatter or self.down_dx_gather or self.gateup_dx_scatter


def routed_kernel_flags() -> RoutedKernelFlags:
    return RoutedKernelFlags(
        fwd_scatter=route_flag("ASYMM_QWEN3_MOE_ROUTE_FWD_SCATTER"),
        down_dx_gather=route_flag("ASYMM_QWEN3_MOE_ROUTE_DOWN_DX_GATHER"),
        gateup_dx_scatter=route_flag("ASYMM_QWEN3_MOE_ROUTE_GATEUP_DX_SCATTER"),
        lora=env_flag("ASYMM_QWEN3_MOE_ROUTE_LORA", False),
        accum_dtype=os.environ.get("ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE", "fp32").strip().lower(),
        debug=env_flag("ASYMM_QWEN3_MOE_ROUTE_KERNEL_DEBUG", False),
    )


def _normalize_token_indices(token_indices: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if token_indices.device != device:
        token_indices = token_indices.to(device=device, non_blocking=True)
    if token_indices.dtype != torch.long:
        token_indices = token_indices.to(dtype=torch.long)
    if not token_indices.is_contiguous():
        token_indices = token_indices.contiguous()
    return token_indices


def _normalize_routing_weights(
    routing_weights: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if routing_weights is None:
        return None
    if routing_weights.device != device:
        routing_weights = routing_weights.to(device=device, non_blocking=True)
    if not routing_weights.is_contiguous():
        routing_weights = routing_weights.contiguous()
    return routing_weights


def _group_metadata_for_kernel(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if offsets.dim() != 1 or experts.dim() != 1:
        raise ValueError("offsets and experts must be 1D tensors")
    if experts.numel() < 2:
        raise ValueError("grouped metadata requires at least one group and a sentinel")
    num_groups = int(experts.numel() - 1)
    memo = getattr(offsets, "_asym_kernel_meta_memo", None)
    if memo is not None and str(device) in memo:
        return memo[str(device)]
    if offsets.numel() == experts.numel():
        starts = offsets[:-1]
        ends = offsets[1:]
        pair_offsets = torch.stack((starts, ends), dim=1).reshape(-1)
    elif offsets.numel() >= 2 * num_groups:
        pair_offsets = offsets[: 2 * num_groups]
    else:
        raise ValueError(
            "offsets must be cumulative [num_groups + 1] or pairs [2 * num_groups], "
            f"got offsets={offsets.numel()} experts={experts.numel()}"
        )
    offsets_i32 = pair_offsets.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    experts_i32 = experts.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    result = (offsets_i32, experts_i32, int(experts_i32.numel()))
    if memo is None:
        try:
            offsets._asym_kernel_meta_memo = {str(device): result}  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        memo[str(device)] = result
    return result


def _check_supported_base(base, device: torch.device, *, compiled_dims: str) -> None:
    if getattr(base, "backend", None) != "asym" or getattr(base, "precision", None) != "bf16":
        raise RuntimeError("Qwen3 routed kernels require AsymGroupedFrozenLinear backend='asym', precision='bf16'")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("Qwen3 routed kernels are SM100-only")
    if compiled_dims != "nk":
        raise RuntimeError("Qwen3 routed kernels currently require compiled_dims='nk'")


def _packed_cpu_weight(base) -> torch.Tensor:
    host_weight = getattr(base, "host_weight", None)
    weight = None if host_weight is None else getattr(host_weight, "weight", None)
    if weight is None:
        raise RuntimeError("Qwen3 routed kernel requires the existing packed CPU Asym host weight")
    return weight


def _require_symbol(name: str):
    import asym_gemm

    symbol = getattr(getattr(asym_gemm, "_C", None), name, None)
    if symbol is None:
        raise RuntimeError(f"`{name}` is not available; rebuild asym_gemm with Qwen3 routed kernels")
    return symbol


@dataclass(frozen=True)
class _RoutePadding:
    padded_rows: torch.Tensor
    original_rows: torch.Tensor
    output_m: int


def _normalize_left(left: torch.Tensor) -> torch.Tensor:
    if left.dtype != torch.bfloat16:
        left = left.to(dtype=torch.bfloat16)
    if not left.is_contiguous():
        left = left.contiguous()
    return left


def _empty_route_weights(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=torch.bfloat16)


def _pad_route_vectors_from_unpad(
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    *,
    device: torch.device,
    padded_rows: int,
    unpad,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    token_indices = _normalize_token_indices(token_indices, device=device)
    routing_weights = _normalize_routing_weights(routing_weights, device=device)
    if unpad is None:
        return token_indices, routing_weights

    padded_tokens = torch.zeros((int(padded_rows),), device=device, dtype=torch.long)
    padded_tokens.index_copy_(
        0,
        unpad.padded_rows,
        token_indices.index_select(0, unpad.original_rows),
    )
    if routing_weights is None:
        return padded_tokens, None

    padded_weights = torch.zeros((int(padded_rows),), device=device, dtype=routing_weights.dtype)
    padded_weights.index_copy_(
        0,
        unpad.padded_rows,
        routing_weights.index_select(0, unpad.original_rows),
    )
    return padded_tokens, padded_weights


def _pad_route_metadata_for_asym(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    *,
    device: torch.device,
    block_m: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], _RoutePadding | None]:
    token_indices = _normalize_token_indices(token_indices, device=device)
    routing_weights = _normalize_routing_weights(routing_weights, device=device)
    if offsets.numel() != experts.numel():
        return offsets, token_indices, routing_weights, None

    # The SAME (offsets, token_indices, routing_weights) trio serves every routed call of a
    # layer (fwd + all backward variants): memoize the padded forms on the offsets tensor.
    # Rebuilding per call cost one .item() sync + several GPU derivations each — measured as
    # the dominant host time in the sTP backward. Downstream memos (pad/group-metadata) hit
    # BECAUSE the returned tensors are now stable objects per layer.
    memo = getattr(offsets, "_asym_route_pad_memo", None)
    memo_key = (str(device), int(block_m), int(token_indices.numel()))
    cached = None if memo is None else memo.get(memo_key)
    if cached is not None:
        return cached

    def _remember(result):
        holder = getattr(offsets, "_asym_route_pad_memo", None)
        if holder is None:
            holder = {}
            try:
                offsets._asym_route_pad_memo = holder  # type: ignore[attr-defined]
            except Exception:
                return result
        holder[memo_key] = result
        return result

    num_groups = int(experts.numel() - 1)
    offsets_long = offsets.to(device=device, dtype=torch.long, non_blocking=True)
    starts = offsets_long[:-1]
    counts = (offsets_long[1:] - starts).clamp_min(0)
    padded_counts = torch.div(counts + int(block_m) - 1, int(block_m), rounding_mode="floor") * int(block_m)
    padded_offsets_long = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(padded_counts, dim=0),
        ),
        dim=0,
    )
    total_padded = int(padded_offsets_long[-1].item())
    if total_padded == int(token_indices.numel()):
        return _remember((offsets, token_indices, routing_weights, None))

    padded_rows = torch.arange(total_padded, device=device, dtype=torch.long)
    group_idx = torch.bucketize(padded_rows, padded_offsets_long[1:], right=True)
    group_idx = group_idx.clamp_max(max(num_groups - 1, 0))
    group_starts = starts.index_select(0, group_idx)
    group_counts = counts.index_select(0, group_idx)
    local_rows = padded_rows - padded_offsets_long.index_select(0, group_idx)
    valid_rows = local_rows < group_counts
    source_rows = group_starts + local_rows
    valid_padded_rows = torch.nonzero(valid_rows, as_tuple=False).flatten()
    original_rows = source_rows.index_select(0, valid_padded_rows)

    padded_tokens = torch.zeros((total_padded,), device=device, dtype=torch.long)
    padded_tokens.index_copy_(0, valid_padded_rows, token_indices.index_select(0, original_rows))
    padded_weights = None
    if routing_weights is not None:
        padded_weights = torch.zeros((total_padded,), device=device, dtype=routing_weights.dtype)
        padded_weights.index_copy_(0, valid_padded_rows, routing_weights.index_select(0, original_rows))

    return _remember((
        padded_offsets_long.to(device=offsets.device, dtype=offsets.dtype),
        padded_tokens,
        padded_weights,
        _RoutePadding(padded_rows=valid_padded_rows, original_rows=original_rows, output_m=int(token_indices.numel())),
    ))


def _routing_weights_arg(routing_weights: Optional[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    return routing_weights if routing_weights is not None else _empty_route_weights(device)


def _record_avoided(base: AsymGroupedFrozenLinear, width: int) -> None:
    stats = getattr(base, "stats", None)
    if stats is not None:
        stats.qwen3_moe_routed_route_space_h_tensors_avoided += 1


def down_forward_scatter_add_(
    base: AsymGroupedFrozenLinear,
    act: torch.Tensor,
    out_fp32: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    *,
    weighted: bool,
    compiled_dims: str = "nk",
) -> torch.Tensor:
    """Compute down expert base projection and scatter-add directly into token-space fp32 output."""
    _check_supported_base(base, act.device, compiled_dims=compiled_dims)
    if out_fp32.dtype != torch.float32:
        raise ValueError("down_forward_scatter_add_ requires a float32 token-space output accumulator")
    act = _normalize_left(act)
    act_kernel, offsets_kernel, unpad = _pad_grouped_input_for_asym(act, offsets, experts)
    token_kernel, weights_kernel = _pad_route_vectors_from_unpad(
        token_indices,
        routing_weights,
        device=act_kernel.device,
        padded_rows=int(act_kernel.shape[0]),
        unpad=unpad,
    )
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets_kernel, experts, device=act_kernel.device)
    _require_symbol("qwen3_moe_bf16_down_forward_scatter_add_")(
        act_kernel,
        _packed_cpu_weight(base),
        out_fp32,
        offsets_i32,
        experts_i32,
        token_kernel,
        _routing_weights_arg(weights_kernel, device=act_kernel.device),
        int(list_size),
        bool(weighted),
        compiled_dims,
    )
    stats = getattr(base, "stats", None)
    if stats is not None:
        stats.qwen3_moe_routed_base_forward_scatter_calls += 1
    _record_avoided(base, int(out_fp32.shape[1]))
    return out_fp32


def down_dx_gather_left(
    base: AsymGroupedFrozenLinear,
    grad_token: torch.Tensor,
    output_shape: tuple[int, int],
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    *,
    weighted: bool,
    compiled_dims: str = "nk",
) -> torch.Tensor:
    """Compute down-base dX from token-space gradient without materializing grad_routes [R,H]."""
    import os as _os
    import time as _time
    _t = _time.perf_counter if _os.environ.get("ASYM_EP_DXTIME") == "1" else None
    _t0 = _t() if _t else 0
    _check_supported_base(base, grad_token.device, compiled_dims=compiled_dims)
    grad_token = _normalize_left(grad_token)
    output_m, output_n = int(output_shape[0]), int(output_shape[1])
    _t1 = _t() if _t else 0
    offsets_kernel, token_kernel, weights_kernel, route_unpad = _pad_route_metadata_for_asym(
        offsets,
        experts,
        token_indices,
        routing_weights,
        device=grad_token.device,
    )
    _t2 = _t() if _t else 0
    padded_m = int(token_kernel.numel())
    grad_act_padded = torch.empty((padded_m, output_n), device=grad_token.device, dtype=torch.bfloat16)
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets_kernel, experts, device=grad_token.device)
    _t3 = _t() if _t else 0
    _require_symbol("qwen3_moe_bf16_down_dx_gather_left_")(
        grad_token,
        _packed_cpu_weight(base),
        grad_act_padded,
        offsets_i32,
        experts_i32,
        token_kernel,
        _routing_weights_arg(weights_kernel, device=grad_token.device),
        int(list_size),
        bool(weighted),
        compiled_dims,
    )
    _t4 = _t() if _t else 0
    grad_act = _unpad_grouped_output(grad_act_padded, route_unpad, output_m=output_m)
    if _t:
        _t5 = _t()
        _p = _os.environ.get("ASYM_EP_DXTIME_PATH", "/tmp/asym_dxtime.log")
        with open(_p, "a") as _fh:
            _fh.write(f"norm={1e3*(_t1-_t0):.1f} pad={1e3*(_t2-_t1):.1f} meta={1e3*(_t3-_t2):.1f} "
                      f"launch={1e3*(_t4-_t3):.1f} unpad={1e3*(_t5-_t4):.1f}\n")
    stats = getattr(base, "stats", None)
    if stats is not None:
        stats.qwen3_moe_routed_base_gather_left_calls += 1
    _record_avoided(base, int(grad_token.shape[1]))
    return grad_act


def gateup_dx_scatter_add_(
    base: AsymGroupedFrozenLinear,
    grad_expert: torch.Tensor,
    grad_hidden_fp32: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: Optional[torch.Tensor],
    *,
    weighted: bool,
    compiled_dims: str = "nk",
) -> torch.Tensor:
    """Compute gate/up base dX and scatter-add directly into token-space fp32 gradient."""
    _check_supported_base(base, grad_expert.device, compiled_dims=compiled_dims)
    if grad_hidden_fp32.dtype != torch.float32:
        raise ValueError("gateup_dx_scatter_add_ requires a float32 token-space output accumulator")
    grad_expert = _normalize_left(grad_expert)
    grad_kernel, offsets_kernel, unpad = _pad_grouped_input_for_asym(grad_expert, offsets, experts)
    token_kernel, weights_kernel = _pad_route_vectors_from_unpad(
        token_indices,
        routing_weights,
        device=grad_kernel.device,
        padded_rows=int(grad_kernel.shape[0]),
        unpad=unpad,
    )
    offsets_i32, experts_i32, list_size = _group_metadata_for_kernel(offsets_kernel, experts, device=grad_kernel.device)
    _require_symbol("qwen3_moe_bf16_gateup_dx_scatter_add_")(
        grad_kernel,
        _packed_cpu_weight(base),
        grad_hidden_fp32,
        offsets_i32,
        experts_i32,
        token_kernel,
        _routing_weights_arg(weights_kernel, device=grad_kernel.device),
        int(list_size),
        bool(weighted),
        compiled_dims,
    )
    stats = getattr(base, "stats", None)
    if stats is not None:
        stats.qwen3_moe_routed_base_dx_scatter_calls += 1
    _record_avoided(base, int(grad_hidden_fp32.shape[1]))
    return grad_hidden_fp32
