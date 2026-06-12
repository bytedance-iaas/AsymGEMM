from __future__ import annotations

from typing import Any

import torch

from .frozen_linear import (
    AsymExecutionStats,
    _group_metadata_tensors,
    _normalize_bf16_output_dtype,
)


CPU_LEFT_BF16_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"


def _arch_major(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_capability(device)[0])


def cpu_left_grouped_bf16_reason(
    a_cpu: torch.Tensor,
    b_cuda: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    output_dtype: torch.dtype | str = torch.bfloat16,
) -> str | None:
    """Return a named reason when the SM100 BF16 CPU-left path is unavailable."""

    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if a_cpu.device.type != "cpu":
        return "input_not_cpu"
    if b_cuda.device.type != "cuda":
        return "weight_not_cuda"
    if _arch_major(b_cuda.device) != 10:
        return "requires_sm100"
    if a_cpu.dim() != 2 or b_cuda.dim() != 3:
        return "requires_2d_input_3d_weight"
    if a_cpu.dtype != torch.bfloat16 or b_cuda.dtype != torch.bfloat16:
        return "requires_bf16"
    try:
        _normalize_bf16_output_dtype(output_dtype)
    except ValueError:
        return "requires_bf16_or_fp32_output"
    if not a_cpu.is_pinned():
        return "input_not_pinned"
    if not a_cpu.is_contiguous() or not b_cuda.is_contiguous():
        return "requires_contiguous"
    if int(b_cuda.shape[0]) <= 0:
        return "requires_positive_groups"
    if int(a_cpu.shape[1]) != int(b_cuda.shape[2]):
        return "shape_mismatch"
    n = int(b_cuda.shape[1])
    k = int(b_cuda.shape[2])
    if n <= 0 or k <= 0:
        return "requires_positive_nk"
    if n % 8 != 0 or k % 8 != 0:
        return "requires_8_aligned_nk"
    if offsets.dim() != 1 or experts.dim() != 1 or experts.numel() < 2:
        return "metadata_mismatch"
    num_groups = int(experts.numel() - 1)
    if offsets.numel() != experts.numel() and offsets.numel() < 2 * num_groups:
        return "metadata_mismatch"

    import asym_gemm

    if not hasattr(asym_gemm, CPU_LEFT_BF16_BINDING):
        return "missing_sm100_cpu_left_bf16_binding"
    return None


def grouped_expert_lora_cpu_left(
    x_cpu: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: Any | None = None,
    compiled_dims: str = "nk",
    output_dtype: torch.dtype | str = torch.bfloat16,
    stats: AsymExecutionStats | None = None,
) -> torch.Tensor:
    """Run grouped LoRA-A as pinned CPU-left BF16 @ CUDA BF16.T on SM100."""

    if x_cpu.dim() != 2:
        raise RuntimeError("SM100 CPU-left grouped BF16 AsymGEMM is unavailable: requires_2d_input_3d_weight")
    if weight.dim() != 3:
        raise RuntimeError("SM100 CPU-left grouped BF16 AsymGEMM is unavailable: requires_2d_input_3d_weight")

    try:
        out_dtype = _normalize_bf16_output_dtype(output_dtype)
    except ValueError as exc:
        raise RuntimeError(
            "SM100 CPU-left grouped BF16 AsymGEMM is unavailable: requires_bf16_or_fp32_output"
        ) from exc
    m = int(x_cpu.shape[0])
    n = int(weight.shape[1])
    if m == 0:
        return torch.empty((0, n), device=weight.device, dtype=out_dtype)

    call_offsets = getattr(metadata, "offsets", offsets) if metadata is not None else offsets
    call_experts = getattr(metadata, "experts", experts) if metadata is not None else experts

    reason = cpu_left_grouped_bf16_reason(
        x_cpu,
        weight,
        call_offsets,
        call_experts,
        output_dtype=out_dtype,
    )
    if reason is not None:
        raise RuntimeError(f"SM100 CPU-left grouped BF16 AsymGEMM is unavailable: {reason}")

    import asym_gemm

    out = torch.empty((m, n), device=weight.device, dtype=out_dtype)
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(call_offsets, call_experts, device=weight.device)
    getattr(asym_gemm, CPU_LEFT_BF16_BINDING)(
        x_cpu,
        weight,
        out,
        offsets_i32,
        experts_i32,
        list_size,
        compiled_dims,
    )
    if stats is not None:
        stats.cpu_left_lora_a_calls += 1
    return out


__all__ = [
    "CPU_LEFT_BF16_BINDING",
    "cpu_left_grouped_bf16_reason",
    "grouped_expert_lora_cpu_left",
]
