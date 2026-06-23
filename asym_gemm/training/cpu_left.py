from __future__ import annotations

import os
from typing import Any

import torch

from .frozen_linear import (
    AsymExecutionStats,
    _GroupedPadding,
    _group_metadata_tensors,
    _normalize_bf16_output_dtype,
    _unpad_grouped_output,
)


CPU_LEFT_BF16_BINDING = "sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous"
CPU_LEFT_BF16_PAIR_BINDING = "sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_nt_contiguous"
CPU_LEFT_GROUPED_BLOCK_M = 128


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _cpu_left_kernel_block_m() -> int:
    return _env_positive_int(
        "DG_BF16_CPU_LEFT_BLOCK_M",
        _env_positive_int("DG_BF16_BLOCK_M", 64),
    )


def _compact_grid_enabled() -> bool:
    return _env_positive_int("DG_BF16_CPU_LEFT_COMPACT_GRID", 0) != 0


def _binding_supports_compact_m_blocks(native: Any, min_argcount: int = 8) -> bool:
    code = getattr(native, "__code__", None)
    if code is None:
        return True
    return int(getattr(code, "co_argcount", 0)) >= int(min_argcount)


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


def _pad_cpu_left_grouped_input_for_asym(
    x_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    index_device: torch.device,
    block_m: int = CPU_LEFT_GROUPED_BLOCK_M,
) -> tuple[torch.Tensor, torch.Tensor, _GroupedPadding | None, int]:
    if offsets.numel() != experts.numel():
        return x_cpu, offsets, None, 0

    num_groups = int(experts.numel() - 1)
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.long)
    starts = offsets_cpu[:-1]
    counts = (offsets_cpu[1:] - starts).clamp_min(0)
    padded_counts = torch.div(counts + int(block_m) - 1, int(block_m), rounding_mode="floor") * int(block_m)
    kernel_block_m = _cpu_left_kernel_block_m()
    compact_m_blocks = int(
        torch.div(
            padded_counts.max() + int(kernel_block_m) - 1,
            int(kernel_block_m),
            rounding_mode="floor",
        ).item()
    ) if int(padded_counts.numel()) > 0 else 0
    padded_offsets_cpu = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(padded_counts, dim=0),
        ),
        dim=0,
    )

    total_padded = int(padded_offsets_cpu[-1].item())
    if total_padded == int(x_cpu.shape[0]):
        return x_cpu, offsets, None, compact_m_blocks

    padded = torch.empty(
        (total_padded, int(x_cpu.shape[1])),
        device="cpu",
        dtype=x_cpu.dtype,
        pin_memory=True,
    )
    padded.zero_()

    padded_row_chunks: list[torch.Tensor] = []
    original_row_chunks: list[torch.Tensor] = []
    for group in range(num_groups):
        rows = int(counts[group].item())
        if rows <= 0:
            continue
        src_start = int(starts[group].item())
        dst_start = int(padded_offsets_cpu[group].item())
        padded[dst_start : dst_start + rows].copy_(x_cpu[src_start : src_start + rows])
        padded_row_chunks.append(torch.arange(dst_start, dst_start + rows, dtype=torch.long))
        original_row_chunks.append(torch.arange(src_start, src_start + rows, dtype=torch.long))

    if padded_row_chunks:
        padded_rows = torch.cat(padded_row_chunks).to(device=index_device, non_blocking=True)
        original_rows = torch.cat(original_row_chunks).to(device=index_device, non_blocking=True)
    else:
        padded_rows = torch.empty((0,), device=index_device, dtype=torch.long)
        original_rows = torch.empty((0,), device=index_device, dtype=torch.long)

    return (
        padded,
        padded_offsets_cpu.to(device=offsets.device, dtype=offsets.dtype, non_blocking=True),
        _GroupedPadding(padded_rows=padded_rows, original_rows=original_rows),
        compact_m_blocks,
    )


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

    x_kernel, offsets_kernel, unpad, compact_m_blocks = _pad_cpu_left_grouped_input_for_asym(
        x_cpu,
        call_offsets,
        call_experts,
        index_device=weight.device,
    )
    out = torch.empty((int(x_kernel.shape[0]), n), device=weight.device, dtype=out_dtype)
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, call_experts, device=weight.device)
    native = getattr(asym_gemm, CPU_LEFT_BF16_BINDING)
    if _compact_grid_enabled() and compact_m_blocks > 0 and _binding_supports_compact_m_blocks(native, min_argcount=8):
        native(
            x_kernel,
            weight,
            out,
            offsets_i32,
            experts_i32,
            list_size,
            compiled_dims,
            int(compact_m_blocks),
        )
    else:
        native(
            x_kernel,
            weight,
            out,
            offsets_i32,
            experts_i32,
            list_size,
            compiled_dims,
        )
    out = _unpad_grouped_output(out, unpad, output_m=m)
    if stats is not None:
        stats.cpu_left_lora_a_calls += 1
    return out


def grouped_expert_lora_pair_cpu_left(
    x_cpu: torch.Tensor,
    weight_gate: torch.Tensor,
    weight_up: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: Any | None = None,
    compiled_dims: str = "nk",
    output_dtype: torch.dtype | str = torch.bfloat16,
    stats: AsymExecutionStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dtype = _normalize_bf16_output_dtype(output_dtype)
    if weight_gate.shape != weight_up.shape:
        raise RuntimeError("SM100 CPU-left grouped BF16 pair AsymGEMM is unavailable: shape_mismatch")
    m = int(x_cpu.shape[0])
    n = int(weight_gate.shape[1])
    if m == 0:
        empty = torch.empty((0, n), device=weight_gate.device, dtype=out_dtype)
        return empty, torch.empty_like(empty)

    call_offsets = getattr(metadata, "offsets", offsets) if metadata is not None else offsets
    call_experts = getattr(metadata, "experts", experts) if metadata is not None else experts

    reason = cpu_left_grouped_bf16_reason(
        x_cpu,
        weight_gate,
        call_offsets,
        call_experts,
        output_dtype=out_dtype,
    )
    if reason is None:
        reason = cpu_left_grouped_bf16_reason(
            x_cpu,
            weight_up,
            call_offsets,
            call_experts,
            output_dtype=out_dtype,
        )
    if reason is not None:
        raise RuntimeError(f"SM100 CPU-left grouped BF16 pair AsymGEMM is unavailable: {reason}")

    import asym_gemm

    if not hasattr(asym_gemm, CPU_LEFT_BF16_PAIR_BINDING):
        raise RuntimeError("SM100 CPU-left grouped BF16 pair AsymGEMM is unavailable: missing_sm100_cpu_left_pair_bf16_binding")

    x_kernel, offsets_kernel, unpad, compact_m_blocks = _pad_cpu_left_grouped_input_for_asym(
        x_cpu,
        call_offsets,
        call_experts,
        index_device=weight_gate.device,
    )
    out_gate = torch.empty((int(x_kernel.shape[0]), n), device=weight_gate.device, dtype=out_dtype)
    out_up = torch.empty_like(out_gate)
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, call_experts, device=weight_gate.device)
    native = getattr(asym_gemm, CPU_LEFT_BF16_PAIR_BINDING)
    if _compact_grid_enabled() and compact_m_blocks > 0 and _binding_supports_compact_m_blocks(native, min_argcount=10):
        native(
            x_kernel,
            weight_gate,
            weight_up,
            out_gate,
            out_up,
            offsets_i32,
            experts_i32,
            list_size,
            compiled_dims,
            int(compact_m_blocks),
        )
    else:
        native(
            x_kernel,
            weight_gate,
            weight_up,
            out_gate,
            out_up,
            offsets_i32,
            experts_i32,
            list_size,
            compiled_dims,
        )
    out_gate = _unpad_grouped_output(out_gate, unpad, output_m=m)
    out_up = _unpad_grouped_output(out_up, unpad, output_m=m)
    if stats is not None:
        stats.cpu_left_lora_a_calls += 1
    return out_gate, out_up


__all__ = [
    "CPU_LEFT_BF16_BINDING",
    "CPU_LEFT_BF16_PAIR_BINDING",
    "CPU_LEFT_GROUPED_BLOCK_M",
    "cpu_left_grouped_bf16_reason",
    "grouped_expert_lora_cpu_left",
    "grouped_expert_lora_pair_cpu_left",
]
