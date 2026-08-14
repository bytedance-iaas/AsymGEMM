from __future__ import annotations

import os
from typing import Literal

import torch

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .cpu_left import (
    CPU_LEFT_BF16_BINDING,
    CPU_LEFT_BF16_PAIR_BINDING,
    grouped_expert_lora_pair_cpu_left,
)
from .frozen_linear import AsymExecutionStats, _group_metadata_tensors
from .lora import GroupedLoRAMetadata, grouped_expert_lora, grouped_expert_lora_cpu_left


LORA_A_GRAD_CPU_RIGHT = "sm100_grouped_lora_a_grad_bf16_cpu_right"
LORA_A_PAIR_GRAD_CPU_RIGHT = "sm100_grouped_lora_a_pair_grad_bf16_cpu_right"
LORA_B_BACKWARD_CPU_SOURCE = "sm100_grouped_lora_b_backward_bf16_cpu_source"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _pair_native_enabled() -> bool:
    """Native gate/up pair kernel selector — DEFAULT ON (flipped 2026-07-27).

    Measured on the standard workload (Qwen3-30B-A3B shapes, 7.68M routed
    rows, X 31.5 GB pinned): pair 224.7 ms vs 2x single 450.9 ms (2.007x;
    2.000x at link saturation with DG_BF16_CPU_LEFT_COMPACT_GRID=1), outputs
    bit-identical. Set ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=0 to fall back to
    the ASYMM_CPU_LEFT_LORA_A_PAIR_CAT / two-single-call paths.
    """
    value = os.environ.get("ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE")
    if value is None or value == "":
        return True
    return value.lower() not in {"0", "false", "no", "off"}


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
    import asym_gemm

    if (
        _pair_native_enabled()
        and hasattr(asym_gemm, CPU_LEFT_BF16_PAIR_BINDING)
        and lora_a_gate.shape == lora_a_up.shape
    ):
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


_REAIM_GRAD_CHUNK = 8192  # house staging grain (M2a-v2 DIRECT_CHUNK)
_REAIM_STAGE_BUFS: dict[tuple[int, int], torch.Tensor] = {}


def _staged_grouped_grad(
    grads_low_rank: tuple[torch.Tensor, ...],
    source_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    site: str,
) -> tuple[torch.Tensor, ...]:
    """Swap-back dA (M2a-v2 'swap' / table middle row): one full H2D re-stage
    of X per leg call, then native per-group weight-grad GEMMs on the staged
    copy. Re-buys the offloaded bytes at the backward peak."""
    from .cpu_left import _reaim_groups_cpu, _reaim_log_once

    _reaim_log_once(f"staged:{site}")
    dev = grads_low_rank[0].device
    x_stage = source_cpu.to(dev, non_blocking=source_cpu.is_pinned())
    k = int(source_cpu.shape[1])
    # bf16 GEMMs on the staged copy (cuBLAS fp32 accumulate) — M2a-v2 'swap'
    outs = tuple(
        torch.zeros((int(num_experts), int(g.shape[1]), k), device=dev, dtype=grads_low_rank[0].dtype)
        for g in grads_low_rank
    )
    for start, end, eid in _reaim_groups_cpu(offsets, experts):
        seg = x_stage[start:end]
        for g, out in zip(grads_low_rank, outs):
            torch.matmul(g[start:end].t(), seg, out=out[eid])
    if os.environ.get("ASYMM_LORA_KERNELS_DEBUG", "") == "1":
        bad = [int(torch.isnan(o).sum()) + int(torch.isinf(o).sum()) for o in outs]
        gbad = [int(torch.isnan(g).sum()) for g in grads_low_rank]
        xbad = int(torch.isnan(x_stage).sum())
        print(f"[staged-dbg] {site} out_nan_inf={bad} dS_nan={gbad} x_nan={xbad}", flush=True)
    del x_stage
    return outs


def _reaim_grouped_grad(
    grad_low_rank: torch.Tensor,
    source_cpu: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    site: str,
) -> torch.Tensor:
    """Re-aim of the adapter gradient: chunked stage+accumulate (M2a-v2
    'reaim2'), segment-sliced for the grouped site. dA's reduction runs over
    the streamed axis, which no operand mapping of the inference kernel
    expresses — the upstream-expressible low-memory fallback re-stages X in
    chunks and accumulates dA_e = dS_e^T X_e in fp32 on the staged copy."""
    from .cpu_left import _reaim_groups_cpu, _reaim_log_once

    _reaim_log_once(site)
    k = int(source_cpu.shape[1])
    r = int(grad_low_rank.shape[1])
    dev = grad_low_rank.device
    grad = torch.zeros((int(num_experts), r, k), device=dev, dtype=torch.float32)
    key = (k, dev.index or 0)
    buf = _REAIM_STAGE_BUFS.get(key)
    if buf is None:
        buf = torch.empty((_REAIM_GRAD_CHUNK, k), device=dev, dtype=torch.bfloat16)
        _REAIM_STAGE_BUFS[key] = buf
    for start, end, eid in _reaim_groups_cpu(offsets, experts):
        for c0 in range(start, end, _REAIM_GRAD_CHUNK):
            c1 = min(end, c0 + _REAIM_GRAD_CHUNK)
            n_rows = c1 - c0
            stage = buf[:n_rows]
            stage.copy_(source_cpu[c0:c1], non_blocking=True)
            grad[eid].addmm_(grad_low_rank[c0:c1].t().float(), stage.float())
    return grad.to(grad_low_rank.dtype)


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
    from .cpu_left import _reaim_enabled, _staged_enabled

    if _reaim_enabled():
        grad_a = _reaim_grouped_grad(grad_low_rank, source_cpu, offsets, experts, num_experts, f"lora_a_grad.{tag}")
        if stats is not None:
            stats.expact_lora_a_grad_grouped_calls += 1
        return grad_a
    if _staged_enabled():
        (grad_a,) = _staged_grouped_grad((grad_low_rank,), source_cpu, offsets, experts, num_experts, f"lora_a_grad.{tag}")
        if stats is not None:
            stats.expact_lora_a_grad_grouped_calls += 1
        return grad_a
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
    from .cpu_left import _reaim_enabled, _staged_enabled

    if _reaim_enabled():
        # X re-staged once per adapter: the upstream mapping has one consumer per pass
        grad_gate = _reaim_grouped_grad(dS_gate, x_cpu, offsets, experts, num_experts, "lora_a_pair_grad.gate")
        grad_up = _reaim_grouped_grad(dS_up, x_cpu, offsets, experts, num_experts, "lora_a_pair_grad.up")
        if stats is not None:
            stats.expact_lora_a_grad_grouped_calls += 1
        return grad_gate, grad_up
    if _staged_enabled():
        grad_gate, grad_up = _staged_grouped_grad((dS_gate, dS_up), x_cpu, offsets, experts, num_experts, "lora_a_pair_grad")
        if stats is not None:
            stats.expact_lora_a_grad_grouped_calls += 1
        return grad_gate, grad_up
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
