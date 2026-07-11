from __future__ import annotations

import os
from typing import Literal

import torch

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .cpu_left import CPU_LEFT_BF16_BINDING, grouped_expert_lora_pair_cpu_left
from .frozen_linear import AsymExecutionStats, _group_metadata_tensors
from .lora import GroupedLoRAMetadata, grouped_expert_lora, grouped_expert_lora_cpu_left


LORA_A_GRAD_CPU_RIGHT = "sm100_grouped_lora_a_grad_bf16_cpu_right"
LORA_A_PAIR_GRAD_CPU_RIGHT = "sm100_grouped_lora_a_pair_grad_bf16_cpu_right"
LORA_B_BACKWARD_CPU_SOURCE = "sm100_grouped_lora_b_backward_bf16_cpu_source"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _missing_symbol(name: str) -> str | None:
    try:
        import asym_gemm
    except Exception:
        return f"missing_{name}"
    return None if hasattr(asym_gemm, name) else f"missing_{name}"


def require_expert_activation_offload_kernels(
    *,
    scope: Literal["forward", "full"] = "full",
    check_only: bool = False,
) -> str | None:
    """Check native support required by Qwen3 expert activation offload."""

    required = [CPU_LEFT_BF16_BINDING]
    if scope == "full":
        required.extend(
            [
                LORA_A_GRAD_CPU_RIGHT,
                LORA_A_PAIR_GRAD_CPU_RIGHT,
            ]
        )
    elif scope != "forward":
        raise ValueError(f"unsupported expert activation-offload scope {scope!r}")

    for name in required:
        reason = _missing_symbol(name)
        if reason is None:
            continue
        if check_only:
            return reason
        raise RuntimeError(f"Qwen3 expert activation offload is unavailable: {reason}")
    return None


def _check_cpu_left_inputs(source_cpu: torch.Tensor, lora_a: torch.Tensor, tag: str) -> None:
    if source_cpu.device.type != "cpu":
        raise RuntimeError(f"{tag}: CPU-left LoRA-A source must be CPU, got {source_cpu.device}")
    if lora_a.device.type != "cuda":
        raise RuntimeError(f"{tag}: CPU-left LoRA-A weight must be CUDA, got {lora_a.device}")
    if source_cpu.dtype != torch.bfloat16 or lora_a.dtype != torch.bfloat16:
        raise RuntimeError(f"{tag}: CPU-left LoRA-A requires BF16 source and weight")
    if not source_cpu.is_contiguous() or not lora_a.is_contiguous():
        raise RuntimeError(f"{tag}: CPU-left LoRA-A requires contiguous source and weight")
    if torch.cuda.is_available() and not source_cpu.is_pinned():
        raise RuntimeError(f"{tag}: CPU-left LoRA-A source must be pinned CPU memory")
    if source_cpu.dim() != 2 or lora_a.dim() != 3:
        raise RuntimeError(f"{tag}: CPU-left LoRA-A expects source [M,K] and weight [E,r,K]")
    if int(source_cpu.shape[1]) != int(lora_a.shape[2]):
        raise RuntimeError(f"{tag}: CPU-left LoRA-A shape mismatch")


def _check_hbm_lora_a_inputs(source_hbm: torch.Tensor, lora_a: torch.Tensor, tag: str) -> None:
    if source_hbm.device.type != "cuda":
        raise RuntimeError(f"{tag}: HBM LoRA-A source must be CUDA, got {source_hbm.device}")
    if lora_a.device.type != "cuda":
        raise RuntimeError(f"{tag}: HBM LoRA-A weight must be CUDA, got {lora_a.device}")
    if source_hbm.dtype != torch.bfloat16 or lora_a.dtype != torch.bfloat16:
        raise RuntimeError(f"{tag}: HBM LoRA-A requires BF16 source and weight")
    if not source_hbm.is_contiguous() or not lora_a.is_contiguous():
        raise RuntimeError(f"{tag}: HBM LoRA-A requires contiguous source and weight")
    if source_hbm.dim() != 2 or lora_a.dim() != 3:
        raise RuntimeError(f"{tag}: HBM LoRA-A expects source [M,K] and weight [E,r,K]")
    if int(source_hbm.shape[1]) != int(lora_a.shape[2]):
        raise RuntimeError(f"{tag}: HBM LoRA-A shape mismatch")


def _check_pinned_cpu_bf16_2d(source_cpu: torch.Tensor, tag: str) -> None:
    if source_cpu.device.type != "cpu":
        raise RuntimeError(f"{tag}: expected CPU tensor, got {source_cpu.device}")
    if source_cpu.dtype != torch.bfloat16:
        raise RuntimeError(f"{tag}: expected CPU BF16 tensor")
    if source_cpu.dim() != 2:
        raise RuntimeError(f"{tag}: expected CPU tensor [M,K]")
    if not source_cpu.is_contiguous():
        raise RuntimeError(f"{tag}: expected contiguous CPU tensor")
    if torch.cuda.is_available() and not source_cpu.is_pinned():
        raise RuntimeError(f"{tag}: expected pinned CPU tensor")


def grouped_lora_a_forward_cpu_left(
    source_cpu: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    _check_cpu_left_inputs(source_cpu, lora_a, tag)
    out = grouped_expert_lora_cpu_left(
        source_cpu,
        lora_a,
        offsets,
        experts,
        metadata=metadata,
        output_dtype=lora_a.dtype,
        stats=stats,
    )
    if stats is not None:
        stats.expact_lora_a_forward_grouped_calls += 1
        stats.expact_lora_a_forward_cpu_left_grouped_calls += 1
    return out


def grouped_lora_a_forward_hbm(
    source_hbm: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    _check_hbm_lora_a_inputs(source_hbm, lora_a, tag)
    out = grouped_expert_lora(
        source_hbm,
        lora_a,
        offsets,
        experts,
        metadata=metadata,
    )
    if stats is not None:
        stats.expact_lora_a_forward_grouped_calls += 1
        stats.expact_lora_a_forward_hbm_grouped_calls += 1
    return out


def grouped_lora_a_pair_forward_cpu_left(
    source_cpu: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: GroupedLoRAMetadata | None,
    stats: AsymExecutionStats | None,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _env_flag("ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE"):
        _check_cpu_left_inputs(source_cpu, lora_a_gate, f"{tag}.gate")
        _check_cpu_left_inputs(source_cpu, lora_a_up, f"{tag}.up")
        gate, up = grouped_expert_lora_pair_cpu_left(
            source_cpu,
            lora_a_gate,
            lora_a_up,
            offsets,
            experts,
            metadata=metadata,
            stats=stats,
        )
        if stats is not None:
            stats.expact_lora_a_forward_grouped_calls += 1
            stats.expact_lora_a_forward_cpu_left_grouped_calls += 1
        return gate, up

    if _env_flag("ASYMM_CPU_LEFT_LORA_A_PAIR_CAT"):
        _check_cpu_left_inputs(source_cpu, lora_a_gate, f"{tag}.gate")
        _check_cpu_left_inputs(source_cpu, lora_a_up, f"{tag}.up")
        if lora_a_gate.shape != lora_a_up.shape:
            raise RuntimeError(f"{tag}: gate/up LoRA-A weights must have matching shapes")
        gate_up_lora_a = torch.cat((lora_a_gate, lora_a_up), dim=1).contiguous()
        gate_up = grouped_lora_a_forward_cpu_left(
            source_cpu,
            gate_up_lora_a,
            offsets,
            experts,
            metadata=metadata,
            stats=stats,
            tag=f"{tag}.gate_up",
        )
        gate, up = gate_up.split(int(lora_a_gate.shape[1]), dim=-1)
        return gate.contiguous(), up.contiguous()

    gate = grouped_lora_a_forward_cpu_left(
        source_cpu,
        lora_a_gate,
        offsets,
        experts,
        metadata=metadata,
        stats=stats,
        tag=f"{tag}.gate",
    )
    up = grouped_lora_a_forward_cpu_left(
        source_cpu,
        lora_a_up,
        offsets,
        experts,
        metadata=metadata,
        stats=stats,
        tag=f"{tag}.up",
    )
    return gate, up


def _native_symbol(name: str):
    reason = _missing_symbol(name)
    if reason is not None:
        raise RuntimeError(f"Qwen3 expert activation offload is unavailable: {reason}")
    import asym_gemm

    return getattr(asym_gemm, name)


def grouped_lora_a_grad_cpu_right(
    grad_low_rank: torch.Tensor,
    source_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    num_experts: int,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    native = _native_symbol(LORA_A_GRAD_CPU_RIGHT)
    _check_pinned_cpu_bf16_2d(source_cpu, tag)
    if grad_low_rank.device.type != "cuda" or grad_low_rank.dtype != torch.bfloat16 or not grad_low_rank.is_contiguous():
        raise RuntimeError(f"{tag}: LoRA-A grad requires contiguous CUDA BF16 dS")
    if grad_low_rank.dim() != 2 or int(grad_low_rank.shape[0]) != int(source_cpu.shape[0]):
        raise RuntimeError(f"{tag}: LoRA-A grad expects dS [M,r] and source [M,K]")
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets, experts, device=grad_low_rank.device)
    grad_a = torch.empty(
        (int(num_experts), int(grad_low_rank.shape[1]), int(source_cpu.shape[1])),
        device=grad_low_rank.device,
        dtype=grad_low_rank.dtype,
    )
    native(grad_low_rank, source_cpu, grad_a, offsets_i32, experts_i32, list_size)
    if stats is not None:
        stats.expact_lora_a_grad_grouped_calls += 1
    return grad_a


def grouped_lora_a_pair_grad_cpu_right(
    dS_gate: torch.Tensor,
    dS_up: torch.Tensor,
    x_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    num_experts: int,
    stats: AsymExecutionStats | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = _native_symbol(LORA_A_PAIR_GRAD_CPU_RIGHT)
    if dS_gate.device.type != "cuda" or dS_up.device.type != "cuda":
        raise RuntimeError("gate/up LoRA-A grad requires CUDA dS tensors")
    if dS_gate.dtype != torch.bfloat16 or dS_up.dtype != torch.bfloat16:
        raise RuntimeError("gate/up LoRA-A grad requires BF16 dS tensors")
    if not dS_gate.is_contiguous() or not dS_up.is_contiguous():
        raise RuntimeError("gate/up LoRA-A grad requires contiguous dS tensors")
    _check_pinned_cpu_bf16_2d(x_cpu, "gate/up LoRA-A grad")
    if dS_gate.dim() != 2 or dS_gate.shape != dS_up.shape or int(dS_gate.shape[0]) != int(x_cpu.shape[0]):
        raise RuntimeError("gate/up LoRA-A grad expects dS tensors [M,r] and X [M,H]")
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets, experts, device=dS_gate.device)
    grad_gate = torch.empty(
        (int(num_experts), int(dS_gate.shape[1]), int(x_cpu.shape[1])),
        device=dS_gate.device,
        dtype=dS_gate.dtype,
    )
    grad_up = torch.empty_like(grad_gate)
    native(dS_gate, dS_up, x_cpu, grad_gate, grad_up, offsets_i32, experts_i32, list_size)
    if stats is not None:
        stats.expact_lora_a_grad_grouped_calls += 1
    return grad_gate, grad_up


def grouped_lora_b_backward_cpu_source(
    grad_out_cpu: torch.Tensor,
    low_rank: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    scale: float,
    stats: AsymExecutionStats | None,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = _native_symbol(LORA_B_BACKWARD_CPU_SOURCE)
    _check_pinned_cpu_bf16_2d(grad_out_cpu, f"{tag}: LoRA-B backward")
    if low_rank.device.type != "cuda" or lora_b.device.type != "cuda":
        raise RuntimeError(f"{tag}: LoRA-B backward requires CUDA low_rank and LoRA-B")
    if low_rank.dtype != torch.bfloat16 or lora_b.dtype != torch.bfloat16:
        raise RuntimeError(f"{tag}: LoRA-B backward requires BF16 low_rank and LoRA-B")
    if not low_rank.is_contiguous() or not lora_b.is_contiguous():
        raise RuntimeError(f"{tag}: LoRA-B backward requires contiguous low_rank and LoRA-B")
    if low_rank.dim() != 2 or lora_b.dim() != 3:
        raise RuntimeError(f"{tag}: LoRA-B backward expects low_rank [M,r] and LoRA-B [E,I,r]")
    if int(low_rank.shape[0]) != int(grad_out_cpu.shape[0]) or int(low_rank.shape[1]) != int(lora_b.shape[2]):
        raise RuntimeError(f"{tag}: LoRA-B backward shape mismatch")
    if int(grad_out_cpu.shape[1]) != int(lora_b.shape[1]):
        raise RuntimeError(f"{tag}: LoRA-B backward output-dim mismatch")
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets, experts, device=low_rank.device)
    dS = torch.empty(
        (int(grad_out_cpu.shape[0]), int(lora_b.shape[2])),
        device=lora_b.device,
        dtype=lora_b.dtype,
    )
    grad_b = torch.empty_like(lora_b)
    native(grad_out_cpu, low_rank, lora_b, dS, grad_b, offsets_i32, experts_i32, list_size, float(scale))
    if stats is not None:
        stats.expact_lora_b_backward_grouped_calls += 1
    return dS, grad_b


def stage_low_rank_from_cpu(
    handle: CPUActivationHandle,
    manager: ActivationOffloadManager,
    *,
    tag: str,
    stats: AsymExecutionStats | None = None,
) -> torch.Tensor:
    if handle.tensor.dim() != 2:
        raise RuntimeError(f"{tag}: expected low-rank CPU handle [M,r]")
    stage = manager.stage(handle, tag=tag)
    if stats is not None:
        stats.expact_stage_low_rank_calls += 1
    return stage


__all__ = [
    "grouped_lora_a_forward_cpu_left",
    "grouped_lora_a_forward_hbm",
    "grouped_lora_a_grad_cpu_right",
    "grouped_lora_a_pair_forward_cpu_left",
    "grouped_lora_a_pair_grad_cpu_right",
    "grouped_lora_b_backward_cpu_source",
    "require_expert_activation_offload_kernels",
    "stage_low_rank_from_cpu",
]
