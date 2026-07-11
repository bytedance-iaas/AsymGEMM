"""Stage I2 (gb200_tp.md): sharded-arena layout planning + repack-at-load.

One pinned copy of every frozen weight, laid out so each device's shard is a CONTIGUOUS
block (the `_direct_bf16_reason` kernel gate requires contiguous pinned 2D operands):

  COL layers (out_features N split; W=[N,K], dim0): shards are already-contiguous dim0
      slices -> zero-copy VIEWS of the original pinned tensor. Each N_i must be %64
      (the dX transpose contraction dim) and %8 for the forward.
  ROW layers (in_features K split; W=[N,K], dim1): a [:, :K0] view is NON-contiguous ->
      repack ONCE into a same-size pinned carrier buffer holding [N,K0] then [N,K1]
      contiguously. The carrier is reshaped [N,K] as a SHAPE-preserving stand-in (its
      element order is NOT the logical [N,K]); every runtime consumer must go through
      the shard views. `_dispatch_nt` asserts no un-routed consumer ever reads it.

Shard metadata rides ON the pinned weight tensor as `tensor._stp` =
StpShardInfo(kind, split, shards) because every primary-path GEMM call passes
`base.host_weight.weight` verbatim into `asym_bf16_cpu_right_matmul` (the I3 choke point).

Weights are frozen: repack happens exactly once at load (E5), zero runtime relayout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = [
    "StpShardInfo",
    "kind_for_module",
    "split_points",
    "attach_col_shards",
    "repack_row_shards",
    "repack_host_weight_for_stp",
    "stp_shard_info",
]

_COL_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW_SUFFIXES = ("o_proj", "down_proj")
_UNSHARDED_MARKERS = ("lm_head", "embed_tokens")


@dataclass(frozen=True)
class StpShardInfo:
    kind: str                      # "col" | "row"
    split: int                     # N0 for col, K0 for row
    shards: tuple[torch.Tensor, torch.Tensor]
    bias_shards: tuple[torch.Tensor, torch.Tensor] | None = None


def _ceil64(value: int) -> int:
    return (value + 63) // 64 * 64


def kind_for_module(name: str) -> str:
    """Classify a wrapped Linear leaf by module name. 'none' = stays unsharded on dev0.

    Matches path COMPONENTS anywhere in the name (wrapped leaves nest, e.g.
    "...self_attn.q_proj.base"), not just the final component.
    """
    lowered = name.lower()
    if any(marker in lowered for marker in _UNSHARDED_MARKERS):
        return "none"
    if re.search(r"(?:^|\.)experts?(?:\.|$)", lowered):
        return "grouped"  # EP-2 bank slicing lands in Stage I7
    for suffix in _COL_SUFFIXES:
        if re.search(rf"(?:^|\.){suffix}(?:\.|$)", lowered):
            return "col"
    for suffix in _ROW_SUFFIXES:
        if re.search(rf"(?:^|\.){suffix}(?:\.|$)", lowered):
            return "row"
    return "none"


def split_points(kind: str, shape: torch.Size) -> int:
    """Split point for a [N,K] weight. col -> N0 (dim0), row -> K0 (dim1).

    COL: BOTH N0 and N1 = N-N0 must be 64-multiples (each shard's dX GEMM contracts over
    its own N_i with the transpose_b k%64 gate) -> requires N % 64 == 0, N0 = ceil64(N/2).
    ROW: K0/K1 must be 8-multiples (forward contraction alignment).
    """
    n, k = int(shape[0]), int(shape[1])
    if kind == "col":
        if n % 64:
            raise ValueError(f"col split needs N%64==0, got N={n}")
        n0 = _ceil64(n // 2)
        if (n - n0) % 64:
            raise ValueError(f"col split N={n} -> N0={n0} leaves misaligned N1={n - n0}")
        return n0
    if kind == "row":
        half = k // 2
        if half % 8 or (k - half) % 8:
            raise ValueError(f"row split needs K/2 %8==0, got K={k}")
        return half
    raise ValueError(f"no split for kind '{kind}'")


def stp_shard_info(tensor: torch.Tensor) -> StpShardInfo | None:
    return getattr(tensor, "_stp", None)


def attach_col_shards(weight: torch.Tensor, bias: torch.Tensor | None = None) -> StpShardInfo:
    """Zero-copy dim0 shard views on the ORIGINAL pinned tensor (stays fully valid)."""
    n0 = split_points("col", weight.shape)
    shards = (weight[:n0], weight[n0:])
    for shard in shards:
        if not (shard.is_contiguous() and shard.is_pinned()):
            raise RuntimeError("col shard view lost contiguity/pinning")
    bias_shards = None
    if bias is not None:
        bias_shards = (bias[:n0], bias[n0:])
    info = StpShardInfo("col", n0, shards, bias_shards)
    weight._stp = info  # type: ignore[attr-defined]
    return info


def repack_row_shards(weight: torch.Tensor) -> tuple[torch.Tensor, StpShardInfo]:
    """Copy the two K-halves into ONE pinned carrier (same total bytes), contiguous each.

    Returns (carrier, info). The carrier is [N,K]-shaped for shape-consumers only; its
    element ORDER is shard-major, so any GEMM must use info.shards. Reassembly is checked
    BEFORE the caller drops the original (nothing to compare against post-load).
    """
    n, k = int(weight.shape[0]), int(weight.shape[1])
    k0 = split_points("row", weight.shape)
    carrier = torch.empty(n * k, dtype=weight.dtype).pin_memory()
    b0 = carrier[: n * k0].view(n, k0)
    b1 = carrier[n * k0 :].view(n, k - k0)
    b0.copy_(weight[:, :k0])
    b1.copy_(weight[:, k0:])
    if not torch.equal(torch.cat([b0, b1], dim=1), weight):
        raise RuntimeError("row repack reassembly mismatch")
    carrier = carrier.view(n, k)
    info = StpShardInfo("row", k0, (b0, b1))
    carrier._stp = info  # type: ignore[attr-defined]
    return carrier, info


def repack_host_weight_for_stp(host_weight, name: str) -> str:
    """Repack ONE HostWeight in place. Returns the kind acted on ('col'/'row'/'none').

    col: attach shard views to the existing pinned tensor (zero copy, original intact).
    row: replace hw._tensor with the carrier; the original's storage is freed when the
         last reference drops (callers fetch hw.weight per call, so the swap propagates).
    """
    kind = kind_for_module(name)
    if kind in ("none",):
        return kind
    if kind == "grouped":
        raise RuntimeError(f"{name}: grouped/expert repack is Stage I7 scope")
    weight = host_weight.weight
    if not (weight.is_pinned() and weight.dim() == 2 and weight.dtype == torch.bfloat16):
        raise RuntimeError(f"{name}: expected pinned 2D bf16 host weight, got {weight.shape} {weight.dtype}")
    if stp_shard_info(weight) is not None:
        return kind  # already repacked
    if kind == "col":
        attach_col_shards(weight)
        return kind
    carrier, _ = repack_row_shards(weight)
    host_weight._tensor = carrier
    return kind


def repack_model_for_stp(named_host_weights: Iterable[tuple[str, object]]) -> dict[str, int]:
    """Repack every wrapped leaf; layer-by-layer so the transient stays < one layer."""
    counts = {"col": 0, "row": 0, "none": 0}
    for name, host_weight in named_host_weights:
        acted = repack_host_weight_for_stp(host_weight, name)
        counts[acted] = counts.get(acted, 0) + 1
    return counts
