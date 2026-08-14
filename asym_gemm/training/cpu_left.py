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


def _lora_kernels_mode() -> str:
    """Fig-12 kernel ablation arm-B selector (ASYMM_LORA_KERNELS):
    - ""       : shipped AsymLoRA kernels (arm A).
    - "reaim"  : re-aimed inference-form mapping (M2a/M2a-v2 'direct'/'reaim2'):
      fwd = operand-swapped upstream call per adapter per expert segment
      (S_e^T = A_e . X_e^T, transpose-back included); grad = chunked
      stage+accumulate (the only upstream-expressible low-memory dA mapping).
    - "staged" : the component table's middle row ('swap-backs'): stage the
      whole CPU-resident operand back to HBM once per leg call and run native
      GEMMs on the staged copy (M2a-v2 'swap'). Fast but re-buys the offloaded
      bytes at the peak — the capacity cost the kernels remove.
    Everything else (offload schedule, base GEMMs, optimizer) is untouched."""
    return os.environ.get("ASYMM_LORA_KERNELS", "").strip().lower()


def _reaim_enabled() -> bool:
    return _lora_kernels_mode() == "reaim"


def _staged_enabled() -> bool:
    return _lora_kernels_mode() == "staged"


def _staged_grouped_forward(
    x_cpu: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    offsets: torch.Tensor,
    experts: torch.Tensor,
    out_dtype: torch.dtype,
    site: str,
) -> tuple[torch.Tensor, ...]:
    """Swap-back fwd: one full H2D stage of X per leg call, then native GEMMs
    per group per adapter on the staged copy (M2a-v2 'swap' semantics)."""
    _reaim_log_once(f"staged:{site}")
    dev = weights[0].device
    x_stage = x_cpu.to(dev, non_blocking=x_cpu.is_pinned())
    m = int(x_cpu.shape[0])
    outs = tuple(
        torch.empty((m, int(w.shape[1])), device=dev, dtype=out_dtype) for w in weights
    )
    covered = 0
    for start, end, eid in _reaim_groups_cpu(offsets, experts):
        seg = x_stage[start:end]
        covered += end - start
        for w, out in zip(weights, outs):
            out[start:end] = (seg @ w[eid].t()).to(out_dtype)
    if os.environ.get("ASYMM_LORA_KERNELS_DEBUG", "") == "1":
        bad = [int(torch.isnan(o).sum()) + int(torch.isinf(o).sum()) for o in outs]
        xbad = int(torch.isnan(x_stage).sum())
        print(f"[staged-dbg] {site} m={m} covered={covered} out_nan_inf={bad} x_nan={xbad}", flush=True)
    del x_stage
    return outs


_REAIM_LOGGED: set[str] = set()


def _reaim_log_once(site: str) -> None:
    if site not in _REAIM_LOGGED:
        _REAIM_LOGGED.add(site)
        print(f"[asym-reaim] ENGAGED {site}", flush=True)


def _reaim_groups_cpu(offsets: torch.Tensor, experts: torch.Tensor) -> list[tuple[int, int, int]]:
    """Memoized (start, end, expert_id) triples for the contiguous-group layout."""
    memo = getattr(offsets, "_asym_reaim_groups_memo", None)
    if memo is not None:
        return memo
    if offsets.numel() != experts.numel():
        raise RuntimeError("reaim grouped mapping requires the contiguous offsets layout")
    offs = offsets.detach().to("cpu", torch.long).tolist()
    exps = experts.detach().to("cpu", torch.long).tolist()
    groups: list[tuple[int, int, int]] = []
    for g in range(len(exps) - 1):
        start, end, eid = int(offs[g]), int(offs[g + 1]), int(exps[g])
        if end > start and eid >= 0:
            groups.append((start, end, eid))
    try:
        offsets._asym_reaim_groups_memo = groups  # type: ignore[attr-defined]
    except Exception:
        pass
    return groups


_REAIM_TAIL_BUFS: dict[tuple[int, int], torch.Tensor] = {}


def _reaim_grouped_forward(
    x_cpu: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    out_dtype: torch.dtype,
    site: str,
) -> torch.Tensor:
    """Direct re-aim of the inference-form kernel at the LoRA-A fwd site.

    Per expert segment: S_e^T = A_e . X_e^T with A_e as the resident "batch"
    and the pinned CPU rows streamed in the kernel's weight role, then the
    transpose-back the consumer needs (row-major S). Segments are ragged, so
    each expert is its own launch (the grouped inference form requires uniform
    weights); rows beyond the 8-row alignment the binding requires are staged.
    """
    from .frozen_linear import asym_bf16_cpu_right_matmul

    _reaim_log_once(site)
    m = int(x_cpu.shape[0])
    n = int(weight.shape[1])
    out = torch.empty((m, n), device=weight.device, dtype=out_dtype)
    for start, end, eid in _reaim_groups_cpu(offsets, experts):
        rows = end - start
        body = rows - (rows % 8)
        a_e = weight[eid]
        if body > 0:
            seg = x_cpu[start : start + body]
            st = asym_bf16_cpu_right_matmul(
                a_e, seg, backend="asym", tag=f"{site}.reaim_direct"
            )
            out[start : start + body] = st.t().to(out_dtype)
        if body < rows:
            key = (int(x_cpu.shape[1]), weight.device.index or 0)
            buf = _REAIM_TAIL_BUFS.get(key)
            if buf is None or buf.shape[0] < 8:
                buf = torch.empty((8, int(x_cpu.shape[1])), device=weight.device, dtype=torch.bfloat16)
                _REAIM_TAIL_BUFS[key] = buf
            tail = rows - body
            buf[:tail].copy_(x_cpu[start + body : end], non_blocking=True)
            out[start + body : end] = (buf[:tail] @ a_e.t()).to(out_dtype)
    return out


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
    # metadata (group starts/counts/padded offsets/index maps) depends only on
    # (offsets, block_m): memoize on the offsets tensor. The un-memoized version cost one
    # sync D2H + ~4 .item()s PER GROUP per call — a top backward hotspot under sTP.
    memo = getattr(offsets, "_asym_cpuleft_pad_memo", None)
    memo_key = (int(block_m), int(x_cpu.shape[0]), str(index_device))
    cached = None if memo is None else memo.get(memo_key)
    if cached is None:
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
        total_padded = int(padded_offsets_cpu[-1])
        copy_plan = []
        padded_row_chunks: list[torch.Tensor] = []
        original_row_chunks: list[torch.Tensor] = []
        if total_padded != int(x_cpu.shape[0]):
            for group in range(num_groups):
                rows = int(counts[group])
                if rows <= 0:
                    continue
                src_start = int(starts[group])
                dst_start = int(padded_offsets_cpu[group])
                copy_plan.append((dst_start, src_start, rows))
                padded_row_chunks.append(torch.arange(dst_start, dst_start + rows, dtype=torch.long))
                original_row_chunks.append(torch.arange(src_start, src_start + rows, dtype=torch.long))
        if padded_row_chunks:
            padded_rows = torch.cat(padded_row_chunks).to(device=index_device, non_blocking=True)
            original_rows = torch.cat(original_row_chunks).to(device=index_device, non_blocking=True)
        else:
            padded_rows = torch.empty((0,), device=index_device, dtype=torch.long)
            original_rows = torch.empty((0,), device=index_device, dtype=torch.long)
        padded_offsets_dev = padded_offsets_cpu.to(device=offsets.device, dtype=offsets.dtype, non_blocking=True)
        cached = (total_padded, compact_m_blocks, copy_plan, padded_rows, original_rows, padded_offsets_dev)
        if memo is None:
            memo = {}
            try:
                offsets._asym_cpuleft_pad_memo = memo  # type: ignore[attr-defined]
            except Exception:
                memo = None
        if memo is not None:
            memo[memo_key] = cached

    total_padded, compact_m_blocks, copy_plan, padded_rows, original_rows, padded_offsets_dev = cached
    if total_padded == int(x_cpu.shape[0]):
        return x_cpu, offsets, None, compact_m_blocks

    from .activation_offload import _alloc_cpu

    padded = _alloc_cpu((total_padded, int(x_cpu.shape[1])), x_cpu.dtype, pin_memory=True)
    padded.zero_()
    for dst_start, src_start, rows in copy_plan:
        padded[dst_start : dst_start + rows].copy_(x_cpu[src_start : src_start + rows])

    return (
        padded,
        padded_offsets_dev,
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

    if _reaim_enabled():
        out = _reaim_grouped_forward(x_cpu, weight, call_offsets, call_experts, out_dtype, "lora_a_fwd")
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return out
    if _staged_enabled():
        (out,) = _staged_grouped_forward(x_cpu, (weight,), call_offsets, call_experts, out_dtype, "lora_a_fwd")
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return out

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

    if _reaim_enabled():
        # one stream per adapter: the inference form serves one consumer per pass
        out_gate = _reaim_grouped_forward(x_cpu, weight_gate, call_offsets, call_experts, out_dtype, "lora_a_pair_fwd.gate")
        out_up = _reaim_grouped_forward(x_cpu, weight_up, call_offsets, call_experts, out_dtype, "lora_a_pair_fwd.up")
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return out_gate, out_up
    if _staged_enabled():
        out_gate, out_up = _staged_grouped_forward(
            x_cpu, (weight_gate, weight_up), call_offsets, call_experts, out_dtype, "lora_a_pair_fwd")
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return out_gate, out_up

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


def grouped_expert_lora_triple_cpu_left(
    x_cpu: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    metadata: Any | None = None,
    compiled_dims: str = "nk",
    stats: AsymExecutionStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """N1: one host X stream serves three adapters (in-kernel kNumOutputs=3)."""
    if not (w0.shape == w1.shape == w2.shape):
        raise RuntimeError("SM100 CPU-left grouped BF16 triple AsymGEMM is unavailable: shape_mismatch")
    m = int(x_cpu.shape[0])
    n = int(w0.shape[1])
    if m == 0:
        e = torch.empty((0, n), device=w0.device, dtype=torch.bfloat16)
        return e, torch.empty_like(e), torch.empty_like(e)
    call_offsets = getattr(metadata, "offsets", offsets) if metadata is not None else offsets
    call_experts = getattr(metadata, "experts", experts) if metadata is not None else experts
    reason = cpu_left_grouped_bf16_reason(x_cpu, w0, call_offsets, call_experts)
    if reason is not None:
        raise RuntimeError(f"SM100 CPU-left grouped BF16 triple AsymGEMM is unavailable: {reason}")
    if _reaim_enabled():
        outs_reaim = tuple(
            _reaim_grouped_forward(x_cpu, w, call_offsets, call_experts, torch.bfloat16, f"lora_a_triple_fwd.{i}")
            for i, w in enumerate((w0, w1, w2))
        )
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return outs_reaim
    if _staged_enabled():
        outs_staged = _staged_grouped_forward(
            x_cpu, (w0, w1, w2), call_offsets, call_experts, torch.bfloat16, "lora_a_triple_fwd")
        if stats is not None:
            stats.cpu_left_lora_a_calls += 1
        return outs_staged
    import asym_gemm

    binding = "sm100_m_grouped_bf16_cpu_left_triple_asym_gemm_nt_contiguous"
    if not hasattr(asym_gemm, binding):
        raise RuntimeError("SM100 CPU-left grouped BF16 triple AsymGEMM is unavailable: missing_binding")
    x_kernel, offsets_kernel, unpad, _ = _pad_cpu_left_grouped_input_for_asym(
        x_cpu, call_offsets, call_experts, index_device=w0.device)
    outs = [torch.empty((int(x_kernel.shape[0]), n), device=w0.device, dtype=torch.bfloat16) for _ in range(3)]
    offsets_i32, experts_i32, list_size = _group_metadata_tensors(offsets_kernel, call_experts, device=w0.device)
    getattr(asym_gemm, binding)(
        x_kernel, w0, w1, w2, outs[0], outs[1], outs[2],
        offsets_i32, experts_i32, list_size, compiled_dims)
    outs = [_unpad_grouped_output(o, unpad, output_m=m) for o in outs]
    if stats is not None:
        stats.cpu_left_lora_a_calls += 1
    return outs[0], outs[1], outs[2]


__all__ = [
    "CPU_LEFT_BF16_BINDING",
    "CPU_LEFT_BF16_PAIR_BINDING",
    "CPU_LEFT_GROUPED_BLOCK_M",
    "cpu_left_grouped_bf16_reason",
    "grouped_expert_lora_cpu_left",
    "grouped_expert_lora_pair_cpu_left",
    "grouped_expert_lora_triple_cpu_left",
]
