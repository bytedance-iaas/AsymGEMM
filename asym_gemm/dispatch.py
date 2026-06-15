# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM
"""Architecture dispatch for the AsymGEMM Python facade.

This module centralizes the SM89 / SM90 / SM100 routing so that callers (e.g.
SGLang's MoE backend) use a single, architecture-agnostic API and never inspect
raw SM numbers themselves. It mirrors DeepGEMM's design where the library — not
the application — decides which kernel runs for the running GPU, and follows
DeepGEMM's naming: kernels are identified by SM number, and the capability split
is framed as Blackwell vs. non-Blackwell.

Two FP8 kernel families coexist in the C++ core:

* ``*_sm89`` / ``*_sm89_masked`` — FP8-tensor-core grouped GEMMs used on Ada
  (SM89) and Hopper (SM90/H20). They consume a *per-token* activation scale and
  a *per-expert* weight scale, passed as separate tensors.
* ``*_nt_contiguous`` / ``*_nt_masked`` — TMA/UMMA grouped GEMMs used on
  Blackwell (SM100+). They consume *per-token-group* (block) scales, optionally
  in the packed UE8M0 layout, carried alongside the data in ``(tensor, scale)``
  pairs.

The unified entry points below accept the ``(data, scale)`` pair form for both
families and unwrap the scales as needed for the SM89/SM90 kernels, so a caller
can issue one call regardless of architecture.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import _C

__all__ = [
    "get_arch_pair",
    "get_arch_major",
    "is_blackwell",
    "supported_dtypes",
    "is_dtype_supported",
    "supported_archs",
    "m_grouped_fp8_asym_gemm_nt_contiguous",
    "m_grouped_fp8_asym_gemm_nt_masked",
]

# Architecture support matrix for the grouped-MoE GEMM kernels, keyed by the
# logical input dtype family. This is the single source of truth for "can
# AsymGEMM run this dtype on this GPU?" and MUST be kept in sync with the
# per-arch dispatch in csrc/apis/gemm.hpp.
#
# Each entry maps a dtype family to the SM targets that have a working kernel:
#   * "bf16" / "fp8" — SM89 (Ada), SM90 (Hopper/H20), SM100+ (Blackwell)
#   * "fp4"  (NVFP4) — SM100+ (Blackwell) only; the block-scaled TMA/UMMA path
#                      has no SM89/SM90 implementation.
# SM targets are encoded as ``major * 10 + minor`` (89, 90) for pre-Blackwell,
# and by major alone for Blackwell+ (any minor of SM100/SM120 counts).
_SUPPORTED_PRE_BLACKWELL_SM = (89, 90)


def _arch_supports(dtype: str, major: int, minor: int) -> bool:
    if dtype == "fp4":
        return major >= 10  # NVFP4 is Blackwell-only
    if dtype in ("bf16", "fp8"):
        return major >= 10 or (major * 10 + minor) in _SUPPORTED_PRE_BLACKWELL_SM
    return False


def get_arch_pair() -> Tuple[int, int]:
    """Return the running GPU's ``(major, minor)`` compute capability.

    Prefers the C++ ``device_runtime`` (authoritative, matches the kernel JIT
    target) and falls back to torch if the extension predates the export.
    """
    if hasattr(_C, "get_arch_pair"):
        major, minor = _C.get_arch_pair()
        return int(major), int(minor)
    return torch.cuda.get_device_capability()


def get_arch_major() -> int:
    if hasattr(_C, "get_arch_major"):
        return int(_C.get_arch_major())
    return get_arch_pair()[0]


def is_blackwell() -> bool:
    """True on Blackwell (SM100+), which uses the TMA/UMMA ``*_nt_*`` kernels and
    packed UE8M0 block scales. Ada (SM89) and Hopper (SM90/H20) are non-Blackwell
    and share the ``*_sm89`` FP8 grouped-GEMM kernels.
    """
    return get_arch_major() >= 10


def supported_dtypes() -> Tuple[str, ...]:
    """The dtype families AsymGEMM can run on *the current* GPU.

    Returns a subset of ``("bf16", "fp8", "fp4")``. Intended for callers (e.g.
    SGLang) to fail fast when a model's GEMM dtype has no kernel on this arch,
    instead of aborting deep inside the first forward pass.
    """
    major, minor = get_arch_pair()
    return tuple(d for d in ("bf16", "fp8", "fp4") if _arch_supports(d, major, minor))


def is_dtype_supported(dtype: str) -> bool:
    """Whether AsymGEMM has a working grouped-MoE kernel for ``dtype`` on the
    current GPU. ``dtype`` is one of ``"bf16"``, ``"fp8"``, ``"fp4"``.
    """
    major, minor = get_arch_pair()
    return _arch_supports(dtype, major, minor)


def supported_archs(dtype: str) -> Tuple[str, ...]:
    """Human-readable SM targets that support ``dtype`` (for error messages),
    e.g. ``("SM100 (Blackwell)",)`` for fp4 or
    ``("SM89", "SM90", "SM100 (Blackwell)")`` for bf16/fp8. Arch-independent.
    """
    if dtype == "fp4":
        return ("SM100 (Blackwell)",)
    if dtype in ("bf16", "fp8"):
        return ("SM89", "SM90/H20", "SM100 (Blackwell)")
    return ()


def _flatten_scale(scale: torch.Tensor) -> torch.Tensor:
    """Flatten a per-token scale tensor to the contiguous 1-D form the SM89
    kernel expects ([total_tokens] for contiguous, [G * M_max] for masked).
    """
    return scale.reshape(-1).contiguous()


def _as_int(list_size) -> int:
    if isinstance(list_size, int):
        return list_size
    return int(list_size.item())


def _is_block_scaled(a_scale: torch.Tensor, b_scale: torch.Tensor) -> bool:
    """Distinguish 1x128/128x128 block scales from legacy per-token/per-expert.

    The weight scale is the discriminator: per-expert is 1-D ``[G]``, block is
    3-D ``[G, ceil(N/128), ceil(K/128)]``.
    """
    if a_scale is None or b_scale is None or b_scale.dim() != 3:
        return False
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise TypeError(
            "SM89/SM90 block-scale path requires float32 scales, got "
            f"a={a_scale.dtype}, b={b_scale.dtype} (UE8M0/packed scales are "
            "Blackwell-only)"
        )
    return True


def m_grouped_fp8_asym_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    list_size,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    """Architecture-agnostic contiguous FP8 grouped GEMM ``[M, K] @ [G, N, K].mT``.

    ``a``/``b`` are ``(data, scale)`` pairs. Block scales (``a``: [M, ceil(K/128)]
    1x128, ``b``: [G, ceil(N/128), ceil(K/128)] 128x128, both float32) are accepted
    on every architecture. The SM89/SM90 path also accepts the legacy per-token /
    per-expert scale pair.
    """
    if is_blackwell():
        _C.m_grouped_fp8_asym_gemm_nt_contiguous(
            a, b, d, offsets, experts, list_size, recipe, compiled_dims, disable_ue8m0_cast
        )
        return

    a_data, a_scale = a
    b_data, b_scale = b
    if _is_block_scaled(a_scale, b_scale):
        _C.m_grouped_fp8_asym_gemm_sm89(
            a_data,
            b_data,
            d,
            offsets,
            experts,
            _as_int(list_size),
            scale_a_block=a_scale.contiguous(),
            scale_b_block=b_scale.contiguous(),
        )
        return
    _C.m_grouped_fp8_asym_gemm_sm89(
        a_data,
        b_data,
        d,
        offsets,
        experts,
        _as_int(list_size),
        1.0,
        1.0,
        _flatten_scale(a_scale),
        b_scale,
    )


def m_grouped_fp8_asym_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    """Architecture-agnostic masked FP8 grouped GEMM ``[G, M, K] @ [G, N, K].mT``.

    ``a``/``b`` are ``(data, scale)`` pairs. Block scales (``a``: [G, M, ceil(K/128)]
    1x128, ``b``: [G, ceil(N/128), ceil(K/128)] 128x128, both float32) are accepted
    on every architecture. The SM89/SM90 path also accepts the legacy per-token
    ([G, M, 1]) / per-expert scale pair.
    """
    if is_blackwell():
        _C.m_grouped_fp8_asym_gemm_nt_masked(
            a, b, d, masked_m, expected_m, recipe, compiled_dims, disable_ue8m0_cast
        )
        return

    a_data, a_scale = a
    b_data, b_scale = b
    if _is_block_scaled(a_scale, b_scale):
        _C.m_grouped_fp8_asym_gemm_sm89_masked(
            a_data,
            b_data,
            d,
            masked_m,
            expected_m,
            scale_a_block=a_scale.contiguous(),
            scale_b_block=b_scale.contiguous(),
        )
        return
    _C.m_grouped_fp8_asym_gemm_sm89_masked(
        a_data,
        b_data,
        d,
        masked_m,
        expected_m,
        1.0,
        1.0,
        _flatten_scale(a_scale),
        b_scale,
    )
