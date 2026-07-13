"""sEP/I7 static EP-2 building blocks (gb200_ep.md E1 / gb200_tp.md I7).

Expert-sorted flat route metadata makes device-local filtering a contiguous SLICE:
zero wasted compute, local 0-based ids fall out of offset rebasing. Branch modules
share the ORIGINAL pinned banks via dim-0 views (ownerless arena — the sEP substrate);
only LoRA params are per-device copies.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from dataclasses import dataclass

from .moe import ContiguousRouteMetadata
from .frozen_linear import AsymGroupedFrozenLinear


@dataclass(frozen=True)
class EpSlicedRouteMetadata(ContiguousRouteMetadata):
    """Sliced metadata: num_tokens stays FULL (scatter output shape) but the route list is
    the local expert range only — num_routes must reflect the actual flat length."""

    @property
    def num_routes(self) -> int:  # type: ignore[override]
        return int(self.token_indices.numel())


def ep_slice_route_metadata(metadata: ContiguousRouteMetadata, lo: int, hi: int) -> ContiguousRouteMetadata:
    """Restrict expert-sorted flat metadata to experts [lo, hi) with LOCAL 0-based ids.

    Rows for one expert are contiguous (expert-major sort), so the slice is exact and
    the resulting scatter produces the device-local PARTIAL output (zeros elsewhere) —
    the cross-device sum (allreduce2) completes it.
    """
    import time as _time

    _t0 = _time.perf_counter()
    # one fused D2H read instead of two .item() syncs
    start, end = metadata.expert_offsets[[lo, hi]].tolist()
    start, end = int(start), int(end)
    try:
        from .ep_vanilla import _timing_add

        _timing_add("slice_item_s", _time.perf_counter() - _t0)
    except Exception:
        pass
    return EpSlicedRouteMetadata(
        token_indices=metadata.token_indices[start:end],
        expert_indices=metadata.expert_indices[start:end] - lo,
        route_indices=metadata.route_indices[start:end],
        routing_weights=metadata.routing_weights[start:end],
        expert_offsets=metadata.expert_offsets[lo : hi + 1] - start,
        expert_counts=metadata.expert_counts[lo:hi],
        num_tokens=metadata.num_tokens,
        top_k=metadata.top_k,
        num_experts=hi - lo,
    )


def _shallow_module_copy(mod: nn.Module) -> nn.Module:
    out = copy.copy(mod)
    out._parameters = dict(mod._parameters)
    out._buffers = dict(mod._buffers)
    out._modules = dict(mod._modules)
    return out


def _slice_grouped_frozen(base: AsymGroupedFrozenLinear, lo: int, hi: int) -> AsymGroupedFrozenLinear:
    """New grouped frozen linear over a dim-0 VIEW of the original pinned bank (zero-copy)."""
    weight = base.host_weight.weight[lo:hi]
    return AsymGroupedFrozenLinear(
        weight,
        backend=base.backend,
        pin_memory=False,  # already a view of pinned memory; do not re-pin/copy
        clone=False,
        precision=base.precision,
        stats=base.stats,
        compiled_dims=base.compiled_dims,
        bf16_output_dtype=base.bf16_output_dtype,
        weight_layout=base.weight_layout,
    )


_LORA_NAMES = ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B")


def slice_experts_for_ep(experts: nn.Module, lo: int, hi: int, device: torch.device) -> nn.Module:
    """Branch experts module for static EP-2: experts [lo, hi) on `device`.

    Shares the original pinned banks (dim-0 views); LoRA params become per-device
    trainable copies of the slice. The forward filter is driven by `ep_expert_range`.
    """
    branch = _shallow_module_copy(experts)
    branch.num_experts = hi - lo
    branch.ep_expert_range = (lo, hi)
    branch.ep_num_experts_full = experts.num_experts
    branch.gate_up_base = _slice_grouped_frozen(experts.gate_up_base, lo, hi)
    branch.down_base = _slice_grouped_frozen(experts.down_base, lo, hi)
    branch._modules["gate_up_base"] = branch.gate_up_base
    branch._modules["down_base"] = branch.down_base
    # fg split caches belong to the FULL bank — the branch rebuilds from its slice
    branch._qwen3_moe_finegrained_gate_base = None
    branch._qwen3_moe_finegrained_up_base = None
    branch._weight_offload = None
    for name in _LORA_NAMES:
        src = getattr(experts, name)
        param = nn.Parameter(src.detach()[lo:hi].to(device=device).clone(), requires_grad=True)
        branch._parameters[name] = param
        setattr(branch, name, param)
    return branch


def build_ep_branch_block(block: nn.Module, lo: int, hi: int, device: torch.device) -> nn.Module:
    """Branch AsymQwen3MoeBlock: replicated frozen router on `device` + sliced experts.

    Both branches see the SAME tokens (replicated residual) and compute IDENTICAL
    routing; each executes only its expert range; outputs are partials to be summed.
    """
    branch = _shallow_module_copy(block)
    gate_params = list(block.gate.parameters())
    if not gate_params:
        # AsymQwen3Router-style: weight is a pinned HostWeight plain attr (no Parameters),
        # staged to the input's device per call — device-agnostic, and untouched by
        # finalize_stp_placement. SHARE it (deepcopy would un-pin the host weight).
        branch.gate = block.gate
        branch._modules["gate"] = block.gate
    else:
        # param-carrying router (nn.Linear / HF TopKRouter): frozen replica pinned to the
        # branch device (placement-independent; tiny [E, H]).
        replica = copy.deepcopy(block.gate).to(device)
        for p in replica.parameters():
            p.requires_grad_(False)
        branch.gate = replica
        branch._modules["gate"] = replica
    branch.experts = slice_experts_for_ep(block.experts, lo, hi, device)
    branch._modules["experts"] = branch.experts
    branch.profile_prefix = f"{block.profile_prefix}.ep{lo}_{hi}"
    branch.experts.profile_prefix = f"{branch.profile_prefix}.experts"
    return branch
