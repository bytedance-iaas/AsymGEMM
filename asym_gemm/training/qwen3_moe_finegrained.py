"""Fine-grained activation offload for Qwen3 routed MoE experts.

This path is separate from the older ``ASYMM_EXPERT_ACT_OFFLOAD`` implementation.
It keeps gate/up base projections split so the backward never materializes a fused
``[R, 2I]`` gate-up activation or ``dgate_up`` tensor.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from .activation_offload import ActivationOffloadManager
from . import activation_offload as _act_offload
from . import cpu_ops
from . import cpu_worker
from . import placement_policy
from .exp_act_offload_lora import (
    LORA_A_GRAD_CPU_RIGHT,
    _missing_symbol,
    grouped_lora_a_forward_cpu_left,
    grouped_lora_a_grad_cpu_right,
    require_expert_activation_offload_kernels,
)
from .frozen_linear import AsymGroupedFrozenLinear
from .lora import grouped_expert_lora, prepare_grouped_lora_metadata
from .profile_ranges import prof_range
from .qwen3_moe_routed_gemm import (
    down_dx_gather_left,
    down_forward_scatter_add_,
    gateup_dx_scatter_add_,
    routed_kernel_flags,
)

if TYPE_CHECKING:  # pragma: no cover
    from .qwen3_moe import AsymQwen3Experts


def _scatter_routes_inplace(
    expert_output: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    num_tokens: int,
    weighted: bool,
) -> torch.Tensor:
    if int(expert_output.shape[0]) != int(token_indices.numel()):
        raise ValueError(
            f"route output/token index mismatch: {tuple(expert_output.shape)} vs {tuple(token_indices.shape)}"
        )
    flat = expert_output.reshape(int(token_indices.numel()), -1)
    if weighted:
        flat.mul_(routing_weights.reshape(-1, 1).to(device=flat.device, dtype=flat.dtype))
    out = torch.zeros((int(num_tokens), int(flat.shape[1])), device=flat.device, dtype=flat.dtype)
    out.index_add_(0, token_indices.to(device=flat.device, dtype=torch.long, non_blocking=True), flat)
    return out.reshape(int(num_tokens), *expert_output.shape[1:])


def _scatter_routes_add_(
    out: torch.Tensor,
    expert_output: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    weighted: bool,
) -> torch.Tensor:
    if int(expert_output.shape[0]) != int(token_indices.numel()):
        raise ValueError(
            f"route output/token index mismatch: {tuple(expert_output.shape)} vs {tuple(token_indices.shape)}"
        )
    flat = expert_output.reshape(int(token_indices.numel()), -1)
    out_flat = out.reshape(int(out.shape[0]), -1)
    if int(flat.shape[1]) != int(out_flat.shape[1]):
        raise ValueError(f"route output width mismatch: {int(flat.shape[1])} vs {int(out_flat.shape[1])}")
    if weighted:
        flat.mul_(routing_weights.reshape(-1, 1).to(device=flat.device, dtype=flat.dtype))
    out_flat.index_add_(0, token_indices.to(device=flat.device, dtype=torch.long, non_blocking=True), flat)
    return out


def _route_grad_from_tokens(
    grad_output: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    weighted: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    grad_flat = grad_output.reshape(int(grad_output.shape[0]), -1)
    grad_routes = grad_flat.index_select(0, token_indices.to(device=grad_flat.device, dtype=torch.long, non_blocking=True))
    grad_routes = grad_routes.to(dtype=out_dtype).contiguous()
    if weighted:
        grad_routes.mul_(routing_weights.reshape(-1, 1).to(device=grad_routes.device, dtype=grad_routes.dtype))
    return grad_routes


def _record_manager_peaks(layer: "AsymQwen3Experts", manager: ActivationOffloadManager) -> dict[str, Any]:
    snapshot = manager.snapshot()
    route_flags = routed_kernel_flags()
    snapshot["qwen3_moe_route_fwd_scatter"] = bool(route_flags.fwd_scatter)
    snapshot["qwen3_moe_route_down_dx_gather"] = bool(route_flags.down_dx_gather)
    snapshot["qwen3_moe_route_gateup_dx_scatter"] = bool(route_flags.gateup_dx_scatter)
    snapshot["qwen3_moe_route_lora"] = bool(route_flags.lora)
    snapshot["qwen3_moe_route_accum_dtype"] = route_flags.accum_dtype
    layer.stats.qwen3_moe_finegrained_saved_cpu_bytes = max(
        int(layer.stats.qwen3_moe_finegrained_saved_cpu_bytes),
        int(snapshot.get("cpu_peak_bytes_live", 0)),
    )
    layer.stats.qwen3_moe_finegrained_stage_hbm_peak_bytes = max(
        int(layer.stats.qwen3_moe_finegrained_stage_hbm_peak_bytes),
        int(snapshot.get("max_stage_bytes_live", 0)),
    )
    return snapshot


def _down_scatter_block_experts() -> int:
    raw = os.environ.get(
        "ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS", "0"),
    )
    try:
        return max(0, int(str(raw).strip() or "0"))
    except ValueError:
        return 0


def _fg_lora_a_fwd_gpu_enabled() -> bool:
    """Compute recompute-forward LoRA-A on GPU from the in-HBM operands.

    The grad-enabled fg forward only runs inside backward recompute, where
    packed X and the GPU act tensor are live anyway, so the GPU grouped GEMM
    adds no HBM lifetime while skipping the CPU-left kernel + its pinned
    padding copies. Backward dA stays on the CPU-right kernel.
    """
    raw = os.environ.get(
        "ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU", "0"),
    )
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_flag_default_off(name: str) -> bool:
    raw = os.environ.get(name, os.environ.get(f"ASYM_GEMM_LF_CONFIG_{name}", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fg_da_gpu_enabled() -> bool:
    """Offload compact X ([M,H], not the topk-expanded [R,H]) and compute the
    gate/up LoRA-A weight grads on GPU from chunked re-gathered rows instead of
    the CPU-right kernel. Requires the GPU LoRA-A recompute-forward."""
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_DA_GPU")


def _fg_keep_dgrads_hbm_enabled() -> bool:
    """Keep dgate/dup in HBM across the layer backward instead of the CPU
    round trip; they are consumed within the same backward and their stage
    buffers were the same size, so live HBM is ~unchanged."""
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM")



_DIRECT_REUSE_COUNTS = {
    "up_direct": 0,
    "act_direct": 0,
    "gate_direct": 0,
    "silu_bwd_single_stage": 0,
    "guard_denied": 0,
}
_DIRECT_REUSE_ENGAGED = False


def direct_reuse_stats() -> dict:
    """P15 (byte-diet) per-mechanism engage counters for the profile."""
    out = dict(_DIRECT_REUSE_COUNTS)
    out["enabled"] = _fg_direct_reuse_enabled()
    return out


def _fg_direct_reuse_enabled() -> bool:
    """128k moe-backward byte-diet (fix_cpu_compute.md, accepted round): DIRECT
    REUSE of the GPU-born gate/up/act tensors in place of their offload->restage
    roundtrips (bit-identical by construction: the skipped roundtrip is lossless).
    Env ``ASYMM_MOE_FG_DIRECT_REUSE`` force-arms; else policy rule P15 (False
    until gated). Every mechanism is additionally G-guarded per call."""
    raw = os.environ.get(
        "ASYMM_MOE_FG_DIRECT_REUSE",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_MOE_FG_DIRECT_REUSE", ""),
    )
    if raw.strip().lower() in {"1", "true", "yes", "y", "on"}:
        return True
    return placement_policy.enabled() and placement_policy.moe_direct_reuse()


def _direct_reuse_ok(mech: str, extra_bytes: int) -> bool:
    """Per-mechanism gate: flag + G headroom (holding the GPU tensor through the
    mechanism's window costs `extra_bytes` of HBM the roundtrip would have freed)."""
    global _DIRECT_REUSE_ENGAGED
    if not _fg_direct_reuse_enabled():
        return False
    if not _act_offload.prefetch_free_ok(int(extra_bytes)):
        _DIRECT_REUSE_COUNTS["guard_denied"] += 1
        return False
    _DIRECT_REUSE_COUNTS[mech] += 1
    if not _DIRECT_REUSE_ENGAGED:
        _DIRECT_REUSE_ENGAGED = True
        import sys

        print(f"[asym-direct-reuse] moe fg direct-reuse ENGAGED (first={mech})",
              file=sys.stderr, flush=True)
    return True



# NOTE: helpers below this point consult asym_gemm.training.placement_policy when the
# ASYM_PLACEMENT_POLICY master switch is ON (fix_cpu_compute.md item 1); the env flags
# remain the manual mechanism for policy-OFF runs.


def _cpu_act_max_bytes() -> int:
    """Bytes-based CPU-act guard (2026-07-15): the rows-only threshold missed dense-32B
    ([256k, 25600] = 13.1 GB with few rows). Cap = rows*cols*2 bytes; default 6.4 GB
    (the measured 30B win regime is <=3.2 GB/tensor; 128k loss regime is 12.6 GB)."""
    raw = os.environ.get(
        "ASYMM_CPU_ACT_MAX_BYTES",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_CPU_ACT_MAX_BYTES", "6400000000"),
    )
    try:
        return int(str(raw).strip() or "6400000000")
    except ValueError:
        return 6400000000


def _fg_cpu_act_max_rows() -> int:
    """Routed-rows ceiling for the CPU act placement (cpu_compute.md CEIL findings,
    2026-07-13): CPU act WINS at R≈2.1M (32k×b8, −2.3%) and LOSES at R≈8.2M
    (128k×b8, −0.7%; 16T worse) — the CPU sweep outgrows the GPU up-block window.
    Above this threshold the GPU act path is used regardless of the flags."""
    raw = os.environ.get(
        "ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS", "4194304"),
    )
    try:
        return int(str(raw).strip() or "4194304")
    except ValueError:
        return 4194304


def _fg_cpu_silu_bwd_enabled() -> bool:
    """K-5 (cpu_compute.md): SwiGLU backward on the CPU worker — dgrads born on CPU
    (enables dB deposits with zero new transfers). Window analysis predicts the silu-bwd
    itself is exposed (~73 ms/layer @32k); measured gate decides."""
    if placement_policy.enabled():
        return placement_policy.moe_cpu_silu_bwd()  # P9: rejected (+7.7% measured)
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_CPU_SILU_BWD")


def _fg_lora_b_grad_cpu_deposit_enabled() -> bool:
    """K-5: grouped dB deposit (dB = dgrad^T @ S, both CPU-resident; same kernel with
    swapped operands). Requires CPU-born dgrads (ASYMM_QWEN3_MOE_FG_CPU_SILU_BWD)."""
    if placement_policy.enabled():
        return placement_policy.moe_lora_b_deposit()  # P9: rejected
    return _env_flag_default_off("ASYMM_LORA_B_GRAD_CPU")


def _fg_stage_dedup_enabled() -> bool:
    """K-9 (cpu_compute.md): reuse ONE act stage for down_lora AND down_base."""
    if placement_policy.enabled():
        return placement_policy.stage_dedup()  # P5: never-hurts, always ON
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_STAGE_DEDUP")


def _fg_cpu_act_chunked_enabled() -> bool:
    """K-4 (cpu_compute.md): worker computes the mul in row-chunks and fills the
    down_base stage buffer per-chunk on the restage side stream (H2D rides the mul)."""
    if placement_policy.enabled():
        return placement_policy.moe_cpu_act_chunked()  # rides P1's gate
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_CPU_ACT_CHUNKED")


def _fg_cpu_act_async_enabled() -> bool:
    """Stage-2 (cpu_compute.md): run the CPU SwiGLU on the CpuWorker thread, split
    into silu(gate) — overlapped with the whole GPU up-block — and ``*= up`` after
    the up D2H (worker FIFO chains the passes). Requires ASYMM_QWEN3_MOE_FG_CPU_ACT
    + ASYMM_CPU_WORKER + the split kernels; falls back to the sync fused path."""
    if placement_policy.enabled():
        return placement_policy.moe_cpu_act_async()  # the winning P1 arm
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_CPU_ACT_ASYNC")


def _fg_cpu_act_enabled() -> bool:
    """Compute the recompute-forward SwiGLU on the Grace CPU with the fused
    SVE kernel (cpu_compute.md Stage 1): gate_cpu/up_cpu are already pinned-CPU
    right after their D2H, so the CPU act skips the gate+up H2D stages and the
    act D2H (3 of 4 stage copies of [R, I]); only the act H2D for the down
    block remains. Requires ASYMM_CPU_FUSED_SILU + a build with the kernels;
    falls back to the GPU path otherwise."""
    if placement_policy.enabled():
        return placement_policy.moe_cpu_act_feature()  # P1 (per-call guard below)
    return _env_flag_default_off("ASYMM_QWEN3_MOE_FG_CPU_ACT")


def _cpu_act_fits(rows: int, nbytes: int) -> bool:
    """P1's per-call automatic gate: rows/bytes window fit. Policy-ON routes through
    the policy module (single authority + P11 tracing); policy-OFF keeps the legacy
    env-threshold behaviour bit-for-bit."""
    if placement_policy.enabled():
        return placement_policy.moe_cpu_act(rows, nbytes)
    return int(rows) <= _fg_cpu_act_max_rows() and int(nbytes) <= _cpu_act_max_bytes()


def _grouped_da_gpu(
    layer: "AsymQwen3Experts",
    d_s: torch.Tensor,
    x_compact: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    input_weighted: bool,
    num_experts: int,
    out_dtype: torch.dtype,
    chunks: int = 8,
) -> torch.Tensor:
    """dA[E,r,H] = sum_e dS_e^T @ X_e with X re-gathered on GPU per group-chunk."""
    from .qwen3_moe import _grouped_lora_weight_grads_torch

    groups = max(int(offsets.numel()) - 1, 0)
    block_experts = max(1, (groups + int(chunks) - 1) // int(chunks))
    grad_a = None
    for row_start, row_end, block_offsets, block_expert_ids, row_slice in _expert_blocks(offsets, experts, block_experts):
        block_metadata = prepare_grouped_lora_metadata(block_offsets, block_expert_ids, dense_experts=True)
        x_rows = x_compact.index_select(
            0,
            token_indices[row_slice].to(device=x_compact.device, dtype=torch.long, non_blocking=True),
        )
        if input_weighted:
            x_rows = x_rows * routing_weights[row_slice].reshape(-1, 1).to(device=x_rows.device, dtype=x_rows.dtype)
        part = _grouped_lora_weight_grads_torch(
            d_s[row_slice].contiguous(),
            x_rows,
            block_offsets,
            block_expert_ids,
            int(num_experts),
            out_dtype=out_dtype,
            metadata=block_metadata,
            stats=None,
            record_lora_b_backward=False,
        )
        del x_rows
        grad_a = part if grad_a is None else grad_a.add_(part)
    if grad_a is None:
        grad_a = torch.zeros((int(num_experts), int(d_s.shape[1]), int(x_compact.shape[1])), device=d_s.device, dtype=out_dtype)
    layer.stats.qwen3_moe_finegrained_lora_a_grad_calls += 1
    layer.stats.qwen3_moe_finegrained_da_gpu_calls += 1
    return grad_a


def _validate_routed_flags_for_current_path(*, down_scatter_block_experts: int) -> None:
    route_flags = routed_kernel_flags()
    if not (route_flags.any_base or route_flags.lora):
        return
    if route_flags.accum_dtype != "fp32":
        raise RuntimeError("Qwen3 routed kernels currently require ASYMM_QWEN3_MOE_ROUTE_ACCUM_DTYPE=fp32")
    if int(down_scatter_block_experts) != 0:
        raise RuntimeError("Qwen3 routed kernels must not be combined with ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS")


def _expert_blocks(
    offsets: torch.Tensor,
    experts: torch.Tensor,
    block_experts: int,
) -> list[tuple[int, int, torch.Tensor, torch.Tensor, slice]]:
    groups = max(int(offsets.numel()) - 1, 0)
    if groups == 0 or int(block_experts) <= 0:
        return []
    memo = getattr(offsets, "_asym_expert_blocks_memo", None)
    if memo is not None and int(block_experts) in memo:
        return memo[int(block_experts)]
    blocks: list[tuple[int, int, torch.Tensor, torch.Tensor, slice]] = []
    for group_start in range(0, groups, int(block_experts)):
        group_end = min(groups, group_start + int(block_experts))
        row_start = int(offsets[group_start].item())
        row_end = int(offsets[group_end].item())
        if row_end <= row_start:
            continue
        block_offsets = (offsets[group_start : group_end + 1] - row_start).contiguous()
        block_expert_ids = torch.cat(
            (experts[group_start:group_end], experts.new_full((1,), -1)),
            dim=0,
        ).contiguous()
        blocks.append((row_start, row_end, block_offsets, block_expert_ids, slice(row_start, row_end)))
    if memo is None:
        try:
            offsets._asym_expert_blocks_memo = {int(block_experts): blocks}  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        memo[int(block_experts)] = blocks
    return blocks


def _record_down_scatter_block(
    layer: "AsymQwen3Experts",
    block_experts: int,
    blocks: list[tuple[int, int, torch.Tensor, torch.Tensor, slice]],
) -> None:
    if int(block_experts) <= 0:
        return
    layer.stats.qwen3_moe_finegrained_down_scatter_block_experts = int(block_experts)
    layer.stats.qwen3_moe_finegrained_down_scatter_blocks += len(blocks)
    for row_start, row_end, _block_offsets, _block_experts, _row_slice in blocks:
        layer.stats.qwen3_moe_finegrained_down_scatter_max_block_rows = max(
            int(layer.stats.qwen3_moe_finegrained_down_scatter_max_block_rows),
            int(row_end) - int(row_start),
        )


def _stage_rows(
    manager: ActivationOffloadManager,
    layer: "AsymQwen3Experts",
    handle,
    row_start: int,
    row_end: int,
    *,
    tag: str,
) -> torch.Tensor:
    layer.stats.qwen3_moe_finegrained_stage_rows_calls += 1
    return manager.stage_rows(handle, row_start, row_end, tag=tag)


def _accumulate_grad(current: torch.Tensor | None, update: torch.Tensor) -> torch.Tensor:
    if current is None:
        return update
    current.add_(update.to(dtype=current.dtype))
    return current


def _base_forward(
    layer: "AsymQwen3Experts",
    base: AsymGroupedFrozenLinear,
    x: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    part: str,
) -> torch.Tensor:
    return base(
        x,
        offsets,
        experts,
        dense_experts=True,
        compiled_dims="nk",
        profile_name=layer._profile_name(part, "finegrained_base"),
    )


def _lora_a_forward_cpu(
    layer: "AsymQwen3Experts",
    source_cpu: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata,
    *,
    tag: str,
) -> torch.Tensor:
    out = grouped_lora_a_forward_cpu_left(
        source_cpu,
        lora_a,
        offsets,
        experts,
        metadata=metadata,
        stats=None,
        tag=tag,
    )
    layer.stats.qwen3_moe_finegrained_lora_a_forward_calls += 1
    return out


def _lora_a_forward_gpu(
    layer: "AsymQwen3Experts",
    source_gpu: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata,
) -> torch.Tensor:
    out = grouped_expert_lora(source_gpu, lora_a, offsets, experts, metadata=metadata)
    layer.stats.qwen3_moe_finegrained_lora_a_forward_calls += 1
    layer.stats.qwen3_moe_finegrained_lora_a_forward_gpu_calls += 1
    return out


def _lora_b_forward(
    low_rank: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata,
    *,
    scale: float,
) -> torch.Tensor:
    delta = grouped_expert_lora(low_rank, lora_b, offsets, experts, metadata=metadata)
    return delta.mul(float(scale)) if float(scale) != 1.0 else delta


def _lora_ds(
    grad_output: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata,
    *,
    scale: float,
) -> torch.Tensor:
    grad_lora = grad_output if grad_output.dtype == lora_b.dtype else grad_output.to(dtype=lora_b.dtype)
    d_s = grouped_expert_lora(
        grad_lora,
        lora_b.transpose(-1, -2),
        offsets,
        experts,
        metadata=metadata,
    )
    return d_s.mul(float(scale)) if float(scale) != 1.0 else d_s


def _lora_b_grad(
    layer: "AsymQwen3Experts",
    grad_output: torch.Tensor,
    low_rank: torch.Tensor,
    lora_b: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    metadata,
) -> torch.Tensor:
    from .qwen3_moe import _grouped_lora_weight_grads_torch

    grad_lora = grad_output if grad_output.dtype == lora_b.dtype else grad_output.to(dtype=lora_b.dtype)
    grad_b = _grouped_lora_weight_grads_torch(
        grad_lora,
        low_rank,
        offsets,
        experts,
        int(lora_b.shape[0]),
        out_dtype=lora_b.dtype,
        metadata=metadata,
        stats=None,
        record_lora_b_backward=False,
    ).mul_(layer.lora_scale)
    layer.stats.qwen3_moe_finegrained_lora_b_backward_calls += 1
    return grad_b


def _lora_a_grad_cpu_deposit_enabled() -> bool:
    """Stage-3 (cpu_compute.md): compute LoRA-A wgrad on the Grace CPU (CpuWorker,
    fire-and-forget) writing fp32 directly into the optimizer's grad buffer. The
    returned autograd grad is a dummy; the grad hook still fires (bank release),
    but its copies are skipped and the drain waits the worker task. Removes the
    per-layer C2C X-read of the GPU cpu-right wgrad kernel."""
    if placement_policy.enabled():
        return placement_policy.moe_wgrad_deposit()  # P2: ON for MoE, all contexts
    return _env_flag_default_off("ASYMM_LORA_A_GRAD_CPU")


def _cpu_group_meta(offsets: torch.Tensor, experts: torch.Tensor):
    cached = getattr(offsets, "_asym_cpu_wgrad_meta", None)
    if cached is not None:
        return cached
    off = offsets.detach().to(device="cpu", dtype=torch.long)
    num_groups = int(off.numel()) - 1
    pairs = torch.stack([off[:-1], off[1:]], dim=1).reshape(-1).contiguous()
    exp = experts.detach().to(device="cpu", dtype=torch.long)
    ge = exp[:num_groups].contiguous() if int(exp.numel()) >= num_groups else torch.arange(num_groups)
    meta = (pairs, ge)
    try:
        offsets._asym_cpu_wgrad_meta = meta
    except Exception:
        pass
    return meta


class _DsSlots:
    """Two rotating pinned dS staging buffers per shape (double-buffer; waiting the
    slot's previous worker task before reuse makes the fire-and-forget safe)."""

    def __init__(self) -> None:
        self.bufs: list = [None, None]
        self.tasks: list = [None, None]
        self.i = 0

    def acquire(self, like: torch.Tensor):
        self.i ^= 1
        if self.tasks[self.i] is not None:
            cpu_worker.wait(self.tasks[self.i])
            self.tasks[self.i] = None
        buf = self.bufs[self.i]
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            from . import pinned_ledger

            nbytes = like.numel() * like.element_size()
            pin = pinned_ledger.try_reserve("deposit", nbytes)
            buf = torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=pin)
            if pin:
                pinned_ledger.register_tensor(buf, "deposit")
            self.bufs[self.i] = buf
        return self.i, buf


_DS_SLOTS: dict = {}
_DEPOSIT_DIAG_PRINTED = False


def _try_deposit_lora_a_grad(layer, d_s, source_cpu, lora_a, offsets, experts, ctx):
    import asym_gemm as _ag
    from . import cpu_adam as _cpu_adam

    kernel = getattr(_ag, "cpu_grouped_lora_a_grad_bf16", None)
    adam = _cpu_adam.get_active_adamw()
    if kernel is None or adam is None or not cpu_worker.enabled():
        return None
    buf = adam.get_grad_deposit_buffer(lora_a)
    if buf is None or buf.dtype != torch.float32 or tuple(buf.shape) != (
        int(lora_a.shape[0]), int(d_s.shape[1]), int(source_cpu.shape[1])
    ):
        return None
    pairs, ge = _cpu_group_meta(offsets, experts)
    slots = _DS_SLOTS.setdefault((tuple(d_s.shape), d_s.dtype), _DsSlots())
    slot_i, ds_pin = slots.acquire(d_s)
    d_s_src = d_s if d_s.is_contiguous() else d_s.contiguous()
    ds_pin.copy_(d_s_src, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream())
    nt = cpu_ops.wgrad_threads()

    def _job(ev=ev, ds=ds_pin, x=source_cpu, out=buf, p=pairs, g=ge, n=nt, k=kernel):
        ev.synchronize()  # host wait: dS D2H must be complete
        k(ds, x, out, p, g, n)

    task = cpu_worker.submit_deposit(_job, tag="deposit.dA.moe")
    slots.tasks[slot_i] = task
    global _DEPOSIT_DIAG_PRINTED
    if not _DEPOSIT_DIAG_PRINTED:
        _DEPOSIT_DIAG_PRINTED = True
        import sys
        print("[asym-cpu-wgrad] Stage-3 deposit path ENGAGED (CPU LoRA-A wgrad -> optimizer grad buffer)",
              file=sys.stderr, flush=True)
    if not adam.register_grad_deposit(lora_a, task):
        cpu_worker.wait(task)  # defensive; single-threaded caller makes this unreachable
        return None
    if ctx is not None:
        tasks = getattr(ctx, "_asym_deposit_tasks", None)
        if tasks is None:
            tasks = []
            ctx._asym_deposit_tasks = tasks
        tasks.append(task)
    layer.stats.qwen3_moe_finegrained_lora_a_grad_calls += 1
    # dummy CUDA grad keeps the autograd/hook contract; its content is never read
    # (the hook's copies are skipped for deposited mappings)
    return torch.empty(buf.shape, device=d_s.device, dtype=torch.bfloat16)


def _lora_a_grad_cpu(
    layer: "AsymQwen3Experts",
    d_s: torch.Tensor,
    source_cpu: torch.Tensor,
    lora_a: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    tag: str,
    allow_deposit: bool = False,
    ctx: Any = None,
) -> torch.Tensor:
    _mode = os.environ.get(
        "ASYMM_LORA_A_GRAD_CPU", os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LORA_A_GRAD_CPU", "0")
    ).strip().lower()
    if placement_policy.enabled():
        _mode = "policy"  # K-2b arm-B (gpu_d2h) is an ablation arm; policy picks deposit
    if allow_deposit and _mode == "gpu_d2h":
        # K-2b arm B: GPU wgrad kernel + direct async D2H into the optimizer staging
        from . import cpu_adam as _cpu_adam

        grad_a = grouped_lora_a_grad_cpu_right(
            d_s.contiguous(), source_cpu, offsets, experts,
            num_experts=int(lora_a.shape[0]), stats=None, tag=tag,
        )
        layer.stats.qwen3_moe_finegrained_lora_a_grad_calls += 1
        adam = _cpu_adam.get_active_adamw()
        if adam is not None and adam.direct_gpu_grad_offload(lora_a, grad_a):
            global _DEPOSIT_DIAG_PRINTED
            if not _DEPOSIT_DIAG_PRINTED:
                _DEPOSIT_DIAG_PRINTED = True
                import sys
                print("[asym-cpu-wgrad] K-2b arm-B (GPU wgrad + direct D2H) ENGAGED", file=sys.stderr, flush=True)
            return torch.empty(grad_a.shape, device=d_s.device, dtype=torch.bfloat16)
        return grad_a
    if allow_deposit and _lora_a_grad_cpu_deposit_enabled():
        deposited = _try_deposit_lora_a_grad(layer, d_s, source_cpu, lora_a, offsets, experts, ctx)
        if deposited is not None:
            return deposited
    grad_a = grouped_lora_a_grad_cpu_right(
        d_s.contiguous(),
        source_cpu,
        offsets,
        experts,
        num_experts=int(lora_a.shape[0]),
        stats=None,
        tag=tag,
    )
    layer.stats.qwen3_moe_finegrained_lora_a_grad_calls += 1
    return grad_a


def _try_deposit_lora_b_grad(layer, dgrad_cpu, s_cpu, lora_b, offsets, experts, ctx, manager):
    """K-5: grouped dB deposit — dB[e] = dgrad_g^T @ S_g (both operands CPU-resident,
    zero new transfers; same kernel as dA with swapped operands), scaled by lora_scale,
    written fp32 into the optimizer buffer on the background worker."""
    import asym_gemm as _ag
    from . import cpu_adam as _cpu_adam

    kernel = getattr(_ag, "cpu_grouped_lora_a_grad_bf16", None)
    adam = _cpu_adam.get_active_adamw()
    if kernel is None or adam is None or not cpu_worker.enabled():
        return None
    dg, sc = dgrad_cpu.tensor, s_cpu.tensor
    if dg.dtype != torch.bfloat16 or sc.dtype != torch.bfloat16 or not sc.is_contiguous() or not dg.is_contiguous():
        return None
    buf = adam.get_grad_deposit_buffer(lora_b)
    if buf is None or buf.dtype != torch.float32 or tuple(buf.shape) != (
        int(lora_b.shape[0]), int(dg.shape[1]), int(sc.shape[1])
    ):
        return None
    pairs, ge = _cpu_group_meta(offsets, experts)
    nt = cpu_ops.wgrad_threads()
    scale = float(layer.lora_scale)

    def _job(dg=dg, sc=sc, out=buf, p=pairs, g=ge, n=nt, k=kernel, s=scale):
        k(dg, sc, out, p, g, n)
        if s != 1.0:
            out.mul_(s)

    task = cpu_worker.submit_deposit(_job, tag="deposit.dB.moe")
    if not adam.register_grad_deposit(lora_b, task):
        cpu_worker.wait(task)
        return None
    tasks = getattr(ctx, "_asym_deposit_tasks", None)
    if tasks is None:
        tasks = []
        ctx._asym_deposit_tasks = tasks
    tasks.append(task)
    # defer the S handle release until the job is done (drain force-sweeps)
    from .attention_activation_offload import _defer_deposit_release

    _defer_deposit_release(task, manager, s_cpu, None)
    return torch.empty(buf.shape, device="cuda", dtype=torch.bfloat16), task


def _base_dx(
    layer: "AsymQwen3Experts",
    base: AsymGroupedFrozenLinear,
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    part: str,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    from .qwen3_moe import _grouped_base_dx

    return _grouped_base_dx(
        base,
        grad_output,
        offsets,
        experts,
        dense_experts=True,
        input_dtype=input_dtype,
        profile_name=layer._profile_name(part, "finegrained_base"),
    )


class _Qwen3MoeFinegrainedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        token_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_lora_A: torch.Tensor,
        gate_lora_B: torch.Tensor,
        up_lora_A: torch.Tensor,
        up_lora_B: torch.Tensor,
        down_lora_A: torch.Tensor,
        down_lora_B: torch.Tensor,
        layer: "AsymQwen3Experts",
        input_weighted: bool,
        output_weighted: bool,
    ) -> torch.Tensor:
        placement_policy.register_model_class("moe")
        weight_offload = getattr(layer, "_weight_offload", None) is not None
        if weight_offload:
            layer.gather_lora_weights()
            gate_lora_A = layer.gate_lora_A
            gate_lora_B = layer.gate_lora_B
            up_lora_A = layer.up_lora_A
            up_lora_B = layer.up_lora_B
            down_lora_A = layer.down_lora_A
            down_lora_B = layer.down_lora_B

        offsets = offsets.detach().contiguous()
        experts = experts.detach().contiguous()
        token_indices = token_indices.detach().contiguous()
        routing_weights = routing_weights.detach().contiguous()
        input_dtype = hidden_states.dtype
        num_tokens = int(hidden_states.shape[0])
        with prof_range(layer._forward_range("moe_finegrained", "pack_tokens")):
            packed = hidden_states.index_select(
                0,
                token_indices.to(device=hidden_states.device, dtype=torch.long, non_blocking=True),
            ).contiguous()
            if input_weighted:
                packed.mul_(routing_weights.reshape(-1, 1).to(device=packed.device, dtype=packed.dtype))
        metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
        gate_base, up_base = layer._ensure_qwen3_moe_finegrained_bases()
        manager = ActivationOffloadManager(pin_memory=True)
        lora_a_fwd_gpu = _fg_lora_a_fwd_gpu_enabled()
        layer.stats.qwen3_moe_finegrained_forward_calls += 1
        down_scatter_block_experts = _down_scatter_block_experts()
        _validate_routed_flags_for_current_path(down_scatter_block_experts=down_scatter_block_experts)
        route_flags = routed_kernel_flags()
        down_scatter_blocks = _expert_blocks(offsets, experts, down_scatter_block_experts)
        _record_down_scatter_block(layer, down_scatter_block_experts, down_scatter_blocks)
        # Compact-X dA needs the GPU LoRA-A forward (nothing reads expanded X on
        # CPU then) and is row-sliced per expert block, so exclude the block path.
        da_gpu = _fg_da_gpu_enabled() and lora_a_fwd_gpu and down_scatter_block_experts == 0

        with prof_range(layer._forward_range("moe_finegrained", "x_to_cpu")):
            if da_gpu:
                x_cpu = manager.offload(hidden_states.detach().contiguous(), "moe.X")
            else:
                x_cpu = manager.offload(packed, "moe.X")

        with prof_range(layer._forward_range("moe_finegrained", "gate")):
            layer.stats.qwen3_moe_finegrained_gate_base_calls += 1
            gate = _base_forward(layer, gate_base, packed, offsets, experts, part="gate")
            if lora_a_fwd_gpu:
                gate_low_rank = _lora_a_forward_gpu(layer, packed, gate_lora_A, offsets, experts, metadata)
            else:
                manager.wait_cpu_ready(x_cpu)
                gate_low_rank = _lora_a_forward_cpu(
                    layer,
                    x_cpu.tensor,
                    gate_lora_A,
                    offsets,
                    experts,
                    metadata,
                    tag="moe.gate",
                )
            gate_delta = _lora_b_forward(
                gate_low_rank,
                gate_lora_B,
                offsets,
                experts,
                metadata,
                scale=layer.lora_scale,
            )
            gate.add_(gate_delta.to(dtype=gate.dtype))
            gate_cpu = manager.offload(gate, "moe.gate")
            # byte-diet mech 3 (gate-direct): provisionally keep the GPU-born gate;
            # committed (guard+counter) only after the act-path decision below, so
            # CPU-act regimes stay allocation-identical to the flag-off arm.
            _dr_gate_cand = gate if _fg_direct_reuse_enabled() else None
            gate_low_rank_cpu = manager.offload(gate_low_rank, "moe.S_gate")
            del gate_delta, gate_low_rank
            del gate  # _dr_gate may still hold the storage (mech 3)

        # Stage-2 async CPU act: start silu(gate) on the worker NOW so it overlaps
        # the whole GPU up-block below; `*= up` is chained after the up D2H.
        act_task = None
        act_cpu_async = None
        split_silu = None
        if (
            _fg_cpu_act_enabled()
            and _fg_cpu_act_async_enabled()
            and cpu_worker.enabled()
            and cpu_ops.fused_silu_applicable(gate_cpu.tensor)
            and _cpu_act_fits(
                int(gate_cpu.tensor.shape[0]), int(gate_cpu.tensor.numel()) * 2
            )
        ):
            split_silu = cpu_ops.split_silu_kernels()
        if split_silu is not None:
            _silu_k, _mul_k, _cpu_nt = split_silu
            act_cpu_async = manager.empty_cpu(
                tuple(gate_cpu.tensor.shape), gate_cpu.tensor.dtype, gate_cpu.original_device, "moe.act"
            )
            _ev_g = manager.take_cpu_ready_event(gate_cpu)
            _g_t, _a_t = gate_cpu.tensor, act_cpu_async.tensor

            def _silu_job(ev=_ev_g, g=_g_t, o=_a_t, k=_silu_k, n=_cpu_nt):
                if ev is not None:
                    ev.synchronize()  # host wait: D2H of gate must be complete
                k(g, o, n)

            act_task = cpu_worker.submit(_silu_job, tag="moe.silu_fwd")


        # mech 3 commit point: keep gate only if the GPU-act path will consume it
        # (out-of-place silu writes a fresh tensor, so demand 2x for the window).
        _dr_gate = (
            _dr_gate_cand
            if (
                _dr_gate_cand is not None
                and act_task is None
                and _direct_reuse_ok("gate_direct", 2 * gate_cpu.nbytes)
            )
            else None
        )
        _dr_gate_cand = None

        with prof_range(layer._forward_range("moe_finegrained", "up")):
            layer.stats.qwen3_moe_finegrained_up_base_calls += 1
            up = _base_forward(layer, up_base, packed, offsets, experts, part="up")
            if lora_a_fwd_gpu:
                up_low_rank = _lora_a_forward_gpu(layer, packed, up_lora_A, offsets, experts, metadata)
            else:
                manager.wait_cpu_ready(x_cpu)
                up_low_rank = _lora_a_forward_cpu(
                    layer,
                    x_cpu.tensor,
                    up_lora_A,
                    offsets,
                    experts,
                    metadata,
                    tag="moe.up",
                )
            up_delta = _lora_b_forward(
                up_low_rank,
                up_lora_B,
                offsets,
                experts,
                metadata,
                scale=layer.lora_scale,
            )
            up.add_(up_delta.to(dtype=up.dtype))
            up_cpu = manager.offload(up, "moe.up")
            up_low_rank_cpu = manager.offload(up_low_rank, "moe.S_up")
            # byte-diet mech 1 (up-direct): the up restage sits a few statements
            # after this offload on the GPU-act path — keep the tensor instead.
            _dr_up = (
                up
                if act_task is None and _direct_reuse_ok("up_direct", up_cpu.nbytes)
                else None
            )
            del up_delta, up_low_rank
            del up  # _dr_up may still hold the storage (mech 1)
        del packed

        act_stage_prefilled = None
        act_stage_done_ev = None
        if act_task is not None:
            # chain `act *= up` behind the silu pass (worker FIFO orders them)
            _ev_u = manager.take_cpu_ready_event(up_cpu)
            _u_t, _a_t2 = up_cpu.tensor, act_cpu_async.tensor
            if _fg_cpu_act_chunked_enabled() and lora_a_fwd_gpu:
                # K-4: the worker fills the down_base stage buffer chunk-by-chunk on the
                # restage side stream while computing the mul — the [R,I] H2D never
                # appears as a separate serialized copy, and K-9 reuses this buffer for
                # BOTH down_lora and down_base (single stage per layer).
                from .attention_activation_offload import _h2d_restage_stream

                act_stage_prefilled = manager.stage_buffer(act_cpu_async, tag="moe.act_for_down_base")
                _side = _h2d_restage_stream(gate_cpu.original_device)
                act_stage_done_ev = torch.cuda.Event()

                def _mul_stage_job(
                    ev=_ev_u, u=_u_t, o=_a_t2, k=_mul_k, n=_cpu_nt,
                    stage_buf=act_stage_prefilled, side=_side, done_ev=act_stage_done_ev,
                ):
                    if ev is not None:
                        ev.synchronize()
                    rows = int(o.shape[0])
                    step = max(65536, -(-rows // 4))  # ~4 chunks, floor 64Ki rows
                    with torch.cuda.stream(side):
                        for c0 in range(0, rows, step):
                            c1 = min(rows, c0 + step)
                            k(o[c0:c1], u[c0:c1], n)
                            stage_buf[c0:c1].copy_(o[c0:c1], non_blocking=True)
                        done_ev.record(side)

                act_task = cpu_worker.submit(_mul_stage_job, tag="moe.mul_stage")
            else:
                def _mul_job(ev=_ev_u, u=_u_t, o=_a_t2, k=_mul_k, n=_cpu_nt):
                    if ev is not None:
                        ev.synchronize()
                    k(o, u, n)

                act_task = cpu_worker.submit(_mul_job, tag="moe.mul")

        down_low_rank = None
        carried_act_stage = None  # K-9: one act stage reused by down_lora AND down_base
        carried_act_gpu = None    # byte-diet mech 2: act kept on-GPU for down_base
        fused_silu = cpu_ops.fused_silu_kernels() if _fg_cpu_act_enabled() else None
        with prof_range(layer._forward_range("moe_finegrained", "activation")):
            if act_task is not None:
                # Stage-2 async path: the worker computed act while the GPU ran the
                # up block; only the residual tail (if any) blocks here.
                cpu_worker.wait(act_task)
                act_cpu = act_cpu_async
                if act_stage_prefilled is not None:
                    # K-4: worker already filled the stage buffer chunk-by-chunk
                    torch.cuda.current_stream().wait_event(act_stage_done_ev)
                    carried_act_stage = act_stage_prefilled
                    down_low_rank = _lora_a_forward_gpu(layer, carried_act_stage, down_lora_A, offsets, experts, metadata)
                elif lora_a_fwd_gpu:
                    _st = manager.stage(act_cpu, tag="moe.act_for_down_base")
                    down_low_rank = _lora_a_forward_gpu(layer, _st, down_lora_A, offsets, experts, metadata)
                    if _fg_stage_dedup_enabled():
                        carried_act_stage = _st
                    else:
                        manager.release_stage(_st)
                    del _st
            elif (
                fused_silu is not None
                and cpu_ops.fused_silu_applicable(gate_cpu.tensor, up_cpu.tensor)
                and _cpu_act_fits(
                    int(gate_cpu.tensor.shape[0]), int(gate_cpu.tensor.numel()) * 2
                )
            ):
                # CPU fused SwiGLU on the already-resident pinned gate/up
                # (skips gate+up H2D stages and the act D2H entirely).
                fused_fwd, _, cpu_act_threads = fused_silu
                manager.host_wait_cpu_ready(gate_cpu)
                manager.host_wait_cpu_ready(up_cpu)
                act_cpu = manager.empty_cpu(
                    tuple(gate_cpu.tensor.shape), gate_cpu.tensor.dtype, gate_cpu.original_device, "moe.act"
                )
                fused_fwd(gate_cpu.tensor, up_cpu.tensor, act_cpu.tensor, cpu_act_threads)
                if lora_a_fwd_gpu:
                    _st = manager.stage(act_cpu, tag="moe.act_for_down_base")
                    down_low_rank = _lora_a_forward_gpu(layer, _st, down_lora_A, offsets, experts, metadata)
                    if _fg_stage_dedup_enabled():
                        carried_act_stage = _st
                    else:
                        manager.release_stage(_st)
                    del _st
            else:
                if _dr_gate is not None:
                    # mech 3: out-of-place silu — the kept gate is the source of an
                    # in-flight side-stream D2H, so it must never be written to.
                    gate_stage = F.silu(_dr_gate)
                    _dr_gate = None
                    _gate_was_stage = False
                else:
                    gate_stage = manager.stage(gate_cpu, tag="moe.gate_for_act")
                    F.silu(gate_stage, inplace=True)
                    _gate_was_stage = True
                if _dr_up is not None:
                    up_stage = _dr_up
                    _dr_up = None
                    _up_was_stage = False
                else:
                    up_stage = manager.stage(up_cpu, tag="moe.up_for_act")
                    _up_was_stage = True
                gate_stage.mul_(up_stage)
                act_gpu = gate_stage.to(dtype=torch.bfloat16).contiguous()
                act_cpu = manager.offload(act_gpu, "moe.act")
                if lora_a_fwd_gpu:
                    down_low_rank = _lora_a_forward_gpu(layer, act_gpu, down_lora_A, offsets, experts, metadata)
                # byte-diet mech 2 (act-direct): keep act_gpu for the down_base GEMM
                # (the act D2H above STAYS — backward's dA deposit host-reads it).
                if down_scatter_block_experts == 0 and _direct_reuse_ok(
                    "act_direct", act_cpu.nbytes
                ):
                    carried_act_gpu = act_gpu
                del act_gpu
                if _gate_was_stage:
                    manager.release_stage(gate_stage, drop_cache=True)
                if _up_was_stage:
                    manager.release_stage(up_stage, drop_cache=True)
                del gate_stage, up_stage

        _dr_gate = _dr_gate_cand = _dr_up = None  # drop unused byte-diet keeps

        with prof_range(layer._forward_range("moe_finegrained", "down_lora")):
            if down_low_rank is None:
                manager.wait_cpu_ready(act_cpu)
                down_low_rank = _lora_a_forward_cpu(
                    layer,
                    act_cpu.tensor,
                    down_lora_A,
                    offsets,
                    experts,
                    metadata,
                    tag="moe.down",
                )
            down_low_rank_cpu = manager.offload(down_low_rank, "moe.S_down")
            scattered = torch.zeros(
                (int(num_tokens), int(layer.hidden_dim)),
                device=hidden_states.device,
                dtype=input_dtype,
            )
            if down_scatter_block_experts > 0:
                layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
                for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                    block_metadata = prepare_grouped_lora_metadata(block_offsets, block_experts, dense_experts=True)
                    down_delta = _lora_b_forward(
                        down_low_rank[row_slice].contiguous(),
                        down_lora_B,
                        block_offsets,
                        block_experts,
                        block_metadata,
                        scale=layer.lora_scale,
                    )
                    _scatter_routes_add_(
                        scattered,
                        down_delta,
                        token_indices[row_slice],
                        routing_weights[row_slice],
                        weighted=bool(output_weighted),
                    )
                    del down_delta
            else:
                down_delta = _lora_b_forward(
                    down_low_rank,
                    down_lora_B,
                    offsets,
                    experts,
                    metadata,
                    scale=layer.lora_scale,
                )
                _scatter_routes_add_(
                    scattered,
                    down_delta,
                    token_indices,
                    routing_weights,
                    weighted=bool(output_weighted),
                )
                del down_delta
            del down_low_rank

        with prof_range(layer._forward_range("moe_finegrained", "down_base")):
            if down_scatter_block_experts > 0:
                if carried_act_stage is not None:
                    manager.release_stage(carried_act_stage, drop_cache=True)
                    carried_act_stage = None
                layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
                for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                    act_stage = _stage_rows(manager, layer, act_cpu, row_start, row_end, tag="moe.act_for_down_base_block")
                    layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                    output = _base_forward(layer, layer.down_base, act_stage, block_offsets, block_experts, part="down")
                    _scatter_routes_add_(
                        scattered,
                        output,
                        token_indices[row_slice],
                        routing_weights[row_slice],
                        weighted=bool(output_weighted),
                    )
                    manager.release_stage(act_stage, drop_cache=True)
                    del act_stage, output
            else:
                if carried_act_gpu is not None:
                    act_stage = carried_act_gpu  # byte-diet mech 2: no restage at all
                elif carried_act_stage is not None:
                    act_stage = carried_act_stage  # K-9: reuse, skip the second H2D copy
                else:
                    act_stage = manager.stage(act_cpu, tag="moe.act_for_down_base")
                layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                if route_flags.fwd_scatter:
                    base_scattered_fp32 = torch.zeros(
                        (int(num_tokens), int(layer.hidden_dim)),
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    down_forward_scatter_add_(
                        layer.down_base,
                        act_stage,
                        base_scattered_fp32,
                        offsets,
                        experts,
                        token_indices,
                        routing_weights,
                        weighted=bool(output_weighted),
                    )
                    scattered.add_(base_scattered_fp32.to(dtype=scattered.dtype))
                    del base_scattered_fp32
                    output = None
                else:
                    output = _base_forward(layer, layer.down_base, act_stage, offsets, experts, part="down")
                if carried_act_gpu is not None:
                    carried_act_gpu = None  # plain GPU tensor, not a managed stage
                else:
                    manager.release_stage(act_stage, drop_cache=True)
                del act_stage

        if weight_offload:
            ctx.save_for_backward(offsets, experts, token_indices, routing_weights)
        else:
            ctx.save_for_backward(
                offsets,
                experts,
                token_indices,
                routing_weights,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            )
        ctx.weight_offload = bool(weight_offload)
        ctx.layer = layer
        ctx.manager = manager
        ctx.num_tokens = int(num_tokens)
        ctx.input_weighted = bool(input_weighted)
        ctx.output_weighted = bool(output_weighted)
        ctx.x_compact = bool(da_gpu)
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_cpu = act_cpu
        ctx.gate_low_rank_cpu = gate_low_rank_cpu
        ctx.up_low_rank_cpu = up_low_rank_cpu
        ctx.down_low_rank_cpu = down_low_rank_cpu
        ctx.input_dtype = input_dtype
        manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)
        layer._last_activation_offload_stats = _record_manager_peaks(layer, manager)
        if down_scatter_block_experts <= 0 and not route_flags.fwd_scatter:
            with prof_range(layer._forward_range("moe_finegrained", "scatter_down_base")):
                _scatter_routes_add_(
                    scattered,
                    output,
                    token_indices,
                    routing_weights,
                    weighted=bool(output_weighted),
                )
                del output
        return scattered

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        layer: AsymQwen3Experts = ctx.layer
        if getattr(ctx, "weight_offload", False):
            offsets, experts, token_indices, routing_weights = ctx.saved_tensors
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
                token_indices,
                routing_weights,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            ) = ctx.saved_tensors

        manager: ActivationOffloadManager = ctx.manager
        metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
        gate_base, up_base = layer._ensure_qwen3_moe_finegrained_bases()
        layer.stats.qwen3_moe_finegrained_backward_calls += 1
        down_scatter_block_experts = _down_scatter_block_experts()
        _validate_routed_flags_for_current_path(down_scatter_block_experts=down_scatter_block_experts)
        route_flags = routed_kernel_flags()
        down_scatter_blocks = _expert_blocks(offsets, experts, down_scatter_block_experts)
        _record_down_scatter_block(layer, down_scatter_block_experts, down_scatter_blocks)
        grad_packed = None
        grad_hidden = None
        grad_hidden_base_fp32 = None
        grad_hidden_lora = None
        grad_gate_cpu = None
        grad_up_cpu = None
        grad_gate_hbm = None
        grad_up_hbm = None
        x_stage = None
        grad_gate_lora_A = grad_gate_lora_B = None
        grad_up_lora_A = grad_up_lora_B = None
        grad_down_lora_A = grad_down_lora_B = None
        x_compact = bool(getattr(ctx, "x_compact", False))
        keep_dgrads_hbm = _fg_keep_dgrads_hbm_enabled() and down_scatter_block_experts == 0

        # R5 restage prefetch (fix_cpu_compute.md): early-issue the gate/up H2D
        # restages on the DEDICATED prefetch stream at backward entry so they run
        # under the multi-GEMM down blocks instead of stalling the silu-bwd block
        # (measured 3x ~0.71 s/step exposed @32k, ~21.6 @128k). Per-tensor events;
        # consumers commit at use. G-guarded (holds the two stage buffers through
        # the down blocks): live @32k, auto-off @128k (G~180/186).
        _pref_gate = _pref_up = None
        if down_scatter_block_experts == 0 and _act_offload.restage_prefetch_enabled():
            _will_cpu_silu = False
            if _fg_cpu_silu_bwd_enabled():
                _fp = cpu_ops.fused_silu_kernels()
                _will_cpu_silu = (
                    _fp is not None
                    and cpu_worker.enabled()
                    and cpu_ops.fused_silu_applicable(ctx.gate_cpu.tensor, ctx.up_cpu.tensor)
                )
            _extra = 2 * int(ctx.gate_cpu.nbytes)
            if not _will_cpu_silu and _act_offload.prefetch_free_ok(_extra):
                _act_offload.prefetch_engaged_once("moe.silu_bwd")
                _pref_gate = manager.stage_begin(ctx.gate_cpu, tag="moe.gate_for_silu_bwd")
                _pref_up = manager.stage_begin(ctx.up_cpu, tag="moe.up_for_silu_bwd_dgate")

        try:
            if down_scatter_block_experts > 0:
                layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
                grad_act = torch.empty(
                    (int(token_indices.numel()), int(layer.intermediate_dim)),
                    device=grad_output.device,
                    dtype=ctx.input_dtype,
                )
                manager.wait_cpu_ready(ctx.act_cpu)
                grad_flat = grad_output.reshape(int(grad_output.shape[0]), -1)
                with prof_range(layer._forward_range("moe_finegrained_backward", "down_block_scatter_grad")):
                    for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                        block_metadata = prepare_grouped_lora_metadata(block_offsets, block_experts, dense_experts=True)
                        block_tokens = token_indices[row_slice]
                        grad_block = grad_flat.index_select(
                            0,
                            block_tokens.to(device=grad_flat.device, dtype=torch.long, non_blocking=True),
                        ).to(dtype=ctx.input_dtype).contiguous()
                        if bool(getattr(ctx, "output_weighted", True)):
                            grad_block.mul_(routing_weights[row_slice].reshape(-1, 1).to(device=grad_block.device, dtype=grad_block.dtype))

                        down_low_rank = _stage_rows(
                            manager,
                            layer,
                            ctx.down_low_rank_cpu,
                            row_start,
                            row_end,
                            tag="moe.S_down_for_dB_block",
                        )
                        d_s_down = _lora_ds(
                            grad_block,
                            down_lora_B,
                            block_offsets,
                            block_experts,
                            block_metadata,
                            scale=layer.lora_scale,
                        )
                        grad_down_lora_B = _accumulate_grad(
                            grad_down_lora_B,
                            _lora_b_grad(
                                layer,
                                grad_block,
                                down_low_rank,
                                down_lora_B,
                                block_offsets,
                                block_experts,
                                block_metadata,
                            ),
                        )
                        manager.release_stage(down_low_rank, drop_cache=True)

                        grad_down_lora_A = _accumulate_grad(
                            grad_down_lora_A,
                            _lora_a_grad_cpu(
                                layer,
                                d_s_down,
                                ctx.act_cpu.tensor[row_slice].contiguous(),
                                down_lora_A,
                                block_offsets,
                                block_experts,
                                tag="moe.down",
                            ),
                        )

                        layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                        grad_act_block = _base_dx(
                            layer,
                            layer.down_base,
                            grad_block,
                            block_offsets,
                            block_experts,
                            part="down",
                            input_dtype=ctx.input_dtype,
                        )
                        down_lora_dx = grouped_expert_lora(
                            d_s_down,
                            down_lora_A.transpose(-1, -2),
                            block_offsets,
                            block_experts,
                            metadata=block_metadata,
                        )
                        grad_act_block.add_(down_lora_dx.to(dtype=grad_act_block.dtype))
                        grad_act[row_slice].copy_(grad_act_block.to(dtype=grad_act.dtype))
                        del grad_block, d_s_down, down_lora_dx, grad_act_block
                manager.release_cpu(ctx.down_low_rank_cpu)
            else:
                if route_flags.down_dx_gather:
                    with prof_range(layer._forward_range("moe_finegrained_backward", "down_base_dx")):
                        grad_token = grad_output.reshape(int(ctx.num_tokens), -1)
                        layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                        grad_act = down_dx_gather_left(
                            layer.down_base,
                            grad_token,
                            (int(token_indices.numel()), int(layer.intermediate_dim)),
                            offsets,
                            experts,
                            token_indices,
                            routing_weights,
                            weighted=bool(getattr(ctx, "output_weighted", True)),
                        )
                else:
                    grad_act = None

                with prof_range(layer._forward_range("moe_finegrained_backward", "scatter_grad")):
                    grad_2d = _route_grad_from_tokens(
                        grad_output,
                        token_indices,
                        routing_weights,
                        weighted=bool(getattr(ctx, "output_weighted", True)),
                        out_dtype=ctx.input_dtype,
                    )

                with prof_range(layer._forward_range("moe_finegrained_backward", "down_lora")):
                    down_low_rank = manager.stage(ctx.down_low_rank_cpu, tag="moe.S_down_for_dB")
                    d_s_down = _lora_ds(
                        grad_2d,
                        down_lora_B,
                        offsets,
                        experts,
                        metadata,
                        scale=layer.lora_scale,
                    )
                    grad_down_lora_B = _lora_b_grad(
                        layer,
                        grad_2d,
                        down_low_rank,
                        down_lora_B,
                        offsets,
                        experts,
                        metadata,
                    )
                    manager.release_stage(down_low_rank, drop_cache=True)
                    manager.release_cpu(ctx.down_low_rank_cpu)
                    if _lora_a_grad_cpu_deposit_enabled():
                        # deposit path reads act_cpu from the HOST — host wait
                        # (wait_cpu_ready would pop the event with only a stream wait)
                        manager.host_wait_cpu_ready(ctx.act_cpu)
                    else:
                        manager.wait_cpu_ready(ctx.act_cpu)
                    grad_down_lora_A = _lora_a_grad_cpu(
                        layer,
                        d_s_down,
                        ctx.act_cpu.tensor,
                        down_lora_A,
                        offsets,
                        experts,
                        tag="moe.down",
                        allow_deposit=True,
                        ctx=ctx,
                    )

                with prof_range(layer._forward_range("moe_finegrained_backward", "down_base_dx")):
                    if grad_act is None:
                        layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                        grad_act = _base_dx(
                            layer,
                            layer.down_base,
                            grad_2d,
                            offsets,
                            experts,
                            part="down",
                            input_dtype=ctx.input_dtype,
                        )
                    down_lora_dx = grouped_expert_lora(
                        d_s_down,
                        down_lora_A.transpose(-1, -2),
                        offsets,
                        experts,
                        metadata=metadata,
                    )
                    grad_act.add_(down_lora_dx.to(dtype=grad_act.dtype))
                    del down_lora_dx, d_s_down, grad_2d

            cpu_silu_bwd_task = None
            _fused_pair = cpu_ops.fused_silu_kernels() if _fg_cpu_silu_bwd_enabled() else None
            if (
                _fused_pair is not None
                and cpu_worker.enabled()
                and down_scatter_block_experts == 0
                and cpu_ops.fused_silu_applicable(ctx.gate_cpu.tensor, ctx.up_cpu.tensor)
            ):
                with prof_range(layer._forward_range("moe_finegrained_backward", "activation")):
                    _, _fused_bwd_k, _nt5 = _fused_pair
                    dact_cpu = manager.offload(grad_act.to(dtype=torch.bfloat16).contiguous(), "moe.dact")
                    del grad_act
                    grad_gate_cpu = manager.empty_cpu(
                        tuple(ctx.gate_cpu.tensor.shape), torch.bfloat16, ctx.gate_cpu.original_device, "moe.dgate"
                    )
                    grad_up_cpu = manager.empty_cpu(
                        tuple(ctx.up_cpu.tensor.shape), torch.bfloat16, ctx.up_cpu.original_device, "moe.dup"
                    )
                    _ev_d = manager.take_cpu_ready_event(dact_cpu)
                    _g5, _u5, _d5 = ctx.gate_cpu.tensor, ctx.up_cpu.tensor, dact_cpu.tensor
                    _dg5, _du5 = grad_gate_cpu.tensor, grad_up_cpu.tensor

                    def _silu_bwd_job(ev=_ev_d, g=_g5, u=_u5, d=_d5, dg=_dg5, du=_du5, k=_fused_bwd_k, n=_nt5):
                        if ev is not None:
                            ev.synchronize()
                        k(g, u, d, dg, du, n)

                    cpu_silu_bwd_task = cpu_worker.submit(_silu_bwd_job, tag="moe.silu_bwd")
                    keep_dgrads_hbm = False
                    # K-5 dB deposits: same kernel, swapped operands, both CPU-resident.
                    if _fg_lora_b_grad_cpu_deposit_enabled():
                        ctx._asym_db_plan = (cpu_silu_bwd_task, grad_gate_cpu, grad_up_cpu)
                    # dact handle: released in the cleanup path after task waits
                    ctx._asym_dact_cpu = dact_cpu
            elif _pref_gate is not None:
              with prof_range(layer._forward_range("moe_finegrained_backward", "activation")):
                # R5 prefetch path: commit the early-issued stages (per-tensor events)
                # and reuse ONE gate stage for both dup and dgate (out-of-place silu
                # keeps gate intact) — same kernels, same operand orders as the legacy
                # sequence, bitwise-identical dgrads (unit-gated); eliminates the
                # third H2D restage entirely.
                layer.stats.qwen3_moe_finegrained_gpu_silu_bwd_calls += 1
                gate_stage = manager.stage_commit(
                    *_pref_gate, nbytes=ctx.gate_cpu.nbytes, tag="moe.gate_for_silu_bwd"
                )
                silu_gate = F.silu(gate_stage)
                silu_gate.mul_(grad_act)
                if keep_dgrads_hbm:
                    grad_up_hbm = silu_gate.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_up_cpu = manager.offload(silu_gate.to(dtype=torch.bfloat16).contiguous(), "moe.dup")
                del silu_gate

                up_stage = manager.stage_commit(
                    *_pref_up, nbytes=ctx.up_cpu.nbytes, tag="moe.up_for_silu_bwd_dgate"
                )
                grad_act.mul_(up_stage)
                manager.release_stage(up_stage, drop_cache=True)
                manager.release_cpu(ctx.up_cpu)
                del up_stage
                _pref_up = None

                grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                del grad_act
                if keep_dgrads_hbm:
                    grad_gate_hbm = grad_gate.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_gate_cpu = manager.offload(grad_gate.to(dtype=torch.bfloat16).contiguous(), "moe.dgate")
                manager.release_stage(gate_stage, drop_cache=True)
                manager.release_cpu(ctx.gate_cpu)
                del gate_stage, grad_gate
                _pref_gate = None
            elif _direct_reuse_ok("silu_bwd_single_stage", ctx.gate_cpu.nbytes):
              with prof_range(layer._forward_range("moe_finegrained_backward", "activation")):
                # byte-diet mech 4 (128k-sized): ONE gate stage held across the up
                # window (demand 1x, unlike P13's 2x early-issue guard); out-of-place
                # silu keeps it intact for silu_backward — same kernels + operand
                # orders as the legacy sequence, bitwise-identical dgrads.
                layer.stats.qwen3_moe_finegrained_gpu_silu_bwd_calls += 1
                gate_stage = manager.stage(ctx.gate_cpu, tag="moe.gate_for_silu_bwd")
                silu_gate = F.silu(gate_stage)
                silu_gate.mul_(grad_act)
                if keep_dgrads_hbm:
                    grad_up_hbm = silu_gate.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_up_cpu = manager.offload(silu_gate.to(dtype=torch.bfloat16).contiguous(), "moe.dup")
                del silu_gate

                up_stage = manager.stage(ctx.up_cpu, tag="moe.up_for_silu_bwd_dgate")
                grad_act.mul_(up_stage)
                manager.release_stage(up_stage, drop_cache=True)
                manager.release_cpu(ctx.up_cpu)
                del up_stage

                grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                del grad_act
                if keep_dgrads_hbm:
                    grad_gate_hbm = grad_gate.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_gate_cpu = manager.offload(grad_gate.to(dtype=torch.bfloat16).contiguous(), "moe.dgate")
                manager.release_stage(gate_stage, drop_cache=True)
                manager.release_cpu(ctx.gate_cpu)
                del gate_stage, grad_gate
            else:
              with prof_range(layer._forward_range("moe_finegrained_backward", "activation")):
                layer.stats.qwen3_moe_finegrained_gpu_silu_bwd_calls += 1
                gate_stage = manager.stage(ctx.gate_cpu, tag="moe.gate_for_silu_bwd_dup")
                F.silu(gate_stage, inplace=True)
                gate_stage.mul_(grad_act)
                if keep_dgrads_hbm:
                    # drop_cache release below makes this reference the sole owner
                    grad_up_hbm = gate_stage.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_up_cpu = manager.offload(gate_stage.to(dtype=torch.bfloat16).contiguous(), "moe.dup")
                manager.release_stage(gate_stage, drop_cache=True)
                del gate_stage

                up_stage = manager.stage(ctx.up_cpu, tag="moe.up_for_silu_bwd_dgate")
                grad_act.mul_(up_stage)
                manager.release_stage(up_stage, drop_cache=True)
                manager.release_cpu(ctx.up_cpu)
                del up_stage

                gate_stage = manager.stage(ctx.gate_cpu, tag="moe.gate_for_silu_bwd_dgate")
                grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                del grad_act
                if keep_dgrads_hbm:
                    grad_gate_hbm = grad_gate.to(dtype=torch.bfloat16).contiguous()
                    layer.stats.qwen3_moe_finegrained_dgrads_hbm_kept += 1
                else:
                    grad_gate_cpu = manager.offload(grad_gate.to(dtype=torch.bfloat16).contiguous(), "moe.dgate")
                manager.release_stage(gate_stage, drop_cache=True)
                manager.release_cpu(ctx.gate_cpu)
                del gate_stage, grad_gate

            if cpu_silu_bwd_task is not None:
                cpu_worker.wait(cpu_silu_bwd_task)
                _d = getattr(ctx, "_asym_dact_cpu", None)
                if _d is not None:
                    manager.release_cpu(_d)
                    ctx._asym_dact_cpu = None

            if down_scatter_block_experts > 0:
                manager.wait_cpu_ready(ctx.x_cpu)
                if ctx.needs_input_grad[0]:
                    layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
                    grad_hidden = torch.zeros(
                        (int(ctx.num_tokens), int(layer.hidden_dim)),
                        device=grad_output.device,
                        dtype=ctx.input_dtype,
                    )
                with prof_range(layer._forward_range("moe_finegrained_backward", "gate_block_dx")):
                    for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                        block_metadata = prepare_grouped_lora_metadata(block_offsets, block_experts, dense_experts=True)
                        grad_gate_stage = _stage_rows(
                            manager,
                            layer,
                            grad_gate_cpu,
                            row_start,
                            row_end,
                            tag="moe.dgate_block",
                        )
                        gate_low_rank = _stage_rows(
                            manager,
                            layer,
                            ctx.gate_low_rank_cpu,
                            row_start,
                            row_end,
                            tag="moe.S_gate_for_dB_block",
                        )
                        d_s_gate = _lora_ds(
                            grad_gate_stage,
                            gate_lora_B,
                            block_offsets,
                            block_experts,
                            block_metadata,
                            scale=layer.lora_scale,
                        )
                        grad_gate_lora_B = _accumulate_grad(
                            grad_gate_lora_B,
                            _lora_b_grad(
                                layer,
                                grad_gate_stage,
                                gate_low_rank,
                                gate_lora_B,
                                block_offsets,
                                block_experts,
                                block_metadata,
                            ),
                        )
                        manager.release_stage(gate_low_rank, drop_cache=True)
                        grad_gate_lora_A = _accumulate_grad(
                            grad_gate_lora_A,
                            _lora_a_grad_cpu(
                                layer,
                                d_s_gate,
                                ctx.x_cpu.tensor[row_slice].contiguous(),
                                gate_lora_A,
                                block_offsets,
                                block_experts,
                                tag="moe.gate",
                            ),
                        )
                        if ctx.needs_input_grad[0]:
                            layer.stats.qwen3_moe_finegrained_gate_base_calls += 1
                            gate_dx = _base_dx(
                                layer,
                                gate_base,
                                grad_gate_stage,
                                block_offsets,
                                block_experts,
                                part="gate",
                                input_dtype=ctx.input_dtype,
                            )
                            gate_lora_dx = grouped_expert_lora(
                                d_s_gate,
                                gate_lora_A.transpose(-1, -2),
                                block_offsets,
                                block_experts,
                                metadata=block_metadata,
                            )
                            gate_dx.add_(gate_lora_dx.to(dtype=gate_dx.dtype))
                            if bool(getattr(ctx, "input_weighted", False)):
                                gate_dx.mul_(routing_weights[row_slice].reshape(-1, 1).to(device=gate_dx.device, dtype=gate_dx.dtype))
                            grad_hidden.index_add_(
                                0,
                                token_indices[row_slice].to(device=gate_dx.device, dtype=torch.long, non_blocking=True),
                                gate_dx.to(dtype=grad_hidden.dtype),
                            )
                            del gate_dx, gate_lora_dx
                        del d_s_gate
                        manager.release_stage(grad_gate_stage, drop_cache=True)
                manager.release_cpu(ctx.gate_low_rank_cpu)
                manager.release_cpu(grad_gate_cpu)
                grad_gate_cpu = None

                with prof_range(layer._forward_range("moe_finegrained_backward", "up_block_dx")):
                    for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                        block_metadata = prepare_grouped_lora_metadata(block_offsets, block_experts, dense_experts=True)
                        grad_up_stage = _stage_rows(
                            manager,
                            layer,
                            grad_up_cpu,
                            row_start,
                            row_end,
                            tag="moe.dup_block",
                        )
                        up_low_rank = _stage_rows(
                            manager,
                            layer,
                            ctx.up_low_rank_cpu,
                            row_start,
                            row_end,
                            tag="moe.S_up_for_dB_block",
                        )
                        d_s_up = _lora_ds(
                            grad_up_stage,
                            up_lora_B,
                            block_offsets,
                            block_experts,
                            block_metadata,
                            scale=layer.lora_scale,
                        )
                        grad_up_lora_B = _accumulate_grad(
                            grad_up_lora_B,
                            _lora_b_grad(
                                layer,
                                grad_up_stage,
                                up_low_rank,
                                up_lora_B,
                                block_offsets,
                                block_experts,
                                block_metadata,
                            ),
                        )
                        manager.release_stage(up_low_rank, drop_cache=True)
                        grad_up_lora_A = _accumulate_grad(
                            grad_up_lora_A,
                            _lora_a_grad_cpu(
                                layer,
                                d_s_up,
                                ctx.x_cpu.tensor[row_slice].contiguous(),
                                up_lora_A,
                                block_offsets,
                                block_experts,
                                tag="moe.up",
                            ),
                        )
                        if ctx.needs_input_grad[0]:
                            layer.stats.qwen3_moe_finegrained_up_base_calls += 1
                            up_dx = _base_dx(
                                layer,
                                up_base,
                                grad_up_stage,
                                block_offsets,
                                block_experts,
                                part="up",
                                input_dtype=ctx.input_dtype,
                            )
                            up_lora_dx = grouped_expert_lora(
                                d_s_up,
                                up_lora_A.transpose(-1, -2),
                                block_offsets,
                                block_experts,
                                metadata=block_metadata,
                            )
                            up_dx.add_(up_lora_dx.to(dtype=up_dx.dtype))
                            if bool(getattr(ctx, "input_weighted", False)):
                                up_dx.mul_(routing_weights[row_slice].reshape(-1, 1).to(device=up_dx.device, dtype=up_dx.dtype))
                            grad_hidden.index_add_(
                                0,
                                token_indices[row_slice].to(device=up_dx.device, dtype=torch.long, non_blocking=True),
                                up_dx.to(dtype=grad_hidden.dtype),
                            )
                            del up_dx, up_lora_dx
                        del d_s_up
                        manager.release_stage(grad_up_stage, drop_cache=True)
                manager.release_cpu(ctx.up_low_rank_cpu)
                manager.release_cpu(grad_up_cpu)
                grad_up_cpu = None
            else:
                if route_flags.gateup_dx_scatter and ctx.needs_input_grad[0]:
                    grad_hidden_base_fp32 = torch.zeros(
                        (int(ctx.num_tokens), int(layer.hidden_dim)),
                        device=grad_output.device,
                        dtype=torch.float32,
                    )
                    grad_hidden_lora = torch.zeros(
                        (int(ctx.num_tokens), int(layer.hidden_dim)),
                        device=grad_output.device,
                        dtype=ctx.input_dtype,
                    )

                with prof_range(layer._forward_range("moe_finegrained_backward", "gate")):
                    if keep_dgrads_hbm:
                        grad_gate_stage = grad_gate_hbm
                    else:
                        grad_gate_stage = manager.stage(grad_gate_cpu, tag="moe.dgate")
                    d_s_gate = _lora_ds(
                        grad_gate_stage,
                        gate_lora_B,
                        offsets,
                        experts,
                        metadata,
                        scale=layer.lora_scale,
                    )
                    _db_gate = None
                    if getattr(ctx, "_asym_db_plan", None) is not None and not keep_dgrads_hbm:
                        _db_gate = _try_deposit_lora_b_grad(
                            layer, grad_gate_cpu, ctx.gate_low_rank_cpu, gate_lora_B, offsets, experts, ctx, manager
                        )
                    if _db_gate is not None:
                        grad_gate_lora_B, _db_gate_task = _db_gate
                    else:
                        gate_low_rank = manager.stage(ctx.gate_low_rank_cpu, tag="moe.S_gate_for_dB")
                        grad_gate_lora_B = _lora_b_grad(
                            layer,
                            grad_gate_stage,
                            gate_low_rank,
                            gate_lora_B,
                            offsets,
                            experts,
                            metadata,
                        )
                        manager.release_stage(gate_low_rank, drop_cache=True)
                        manager.release_cpu(ctx.gate_low_rank_cpu)
                    if x_compact:
                        x_stage = manager.stage(ctx.x_cpu, tag="moe.X_for_dA")
                        grad_gate_lora_A = _grouped_da_gpu(
                            layer,
                            d_s_gate,
                            x_stage,
                            token_indices,
                            routing_weights,
                            offsets,
                            experts,
                            input_weighted=bool(getattr(ctx, "input_weighted", False)),
                            num_experts=int(gate_lora_A.shape[0]),
                            out_dtype=gate_lora_A.dtype,
                        )
                    else:
                        manager.wait_cpu_ready(ctx.x_cpu)
                        grad_gate_lora_A = _lora_a_grad_cpu(
                            layer,
                            d_s_gate,
                            ctx.x_cpu.tensor,
                            gate_lora_A,
                            offsets,
                            experts,
                            tag="moe.gate",
                        )
                    if ctx.needs_input_grad[0]:
                        layer.stats.qwen3_moe_finegrained_gate_base_calls += 1
                        if route_flags.gateup_dx_scatter:
                            gateup_dx_scatter_add_(
                                gate_base,
                                grad_gate_stage,
                                grad_hidden_base_fp32,
                                offsets,
                                experts,
                                token_indices,
                                routing_weights,
                                weighted=bool(getattr(ctx, "input_weighted", False)),
                            )
                        else:
                            grad_packed = _base_dx(
                                layer,
                                gate_base,
                                grad_gate_stage,
                                offsets,
                                experts,
                                part="gate",
                                input_dtype=ctx.input_dtype,
                            )
                        gate_lora_dx = grouped_expert_lora(
                            d_s_gate,
                            gate_lora_A.transpose(-1, -2),
                            offsets,
                            experts,
                            metadata=metadata,
                        )
                        if route_flags.gateup_dx_scatter:
                            _scatter_routes_add_(
                                grad_hidden_lora,
                                gate_lora_dx,
                                token_indices,
                                routing_weights,
                                weighted=bool(getattr(ctx, "input_weighted", False)),
                            )
                        else:
                            grad_packed.add_(gate_lora_dx.to(dtype=grad_packed.dtype))
                        del gate_lora_dx
                    del d_s_gate
                    if keep_dgrads_hbm:
                        grad_gate_hbm = None
                        grad_gate_stage = None
                    else:
                        manager.release_stage(grad_gate_stage, drop_cache=True)
                        if _db_gate is not None:
                            from .attention_activation_offload import _defer_deposit_release
                            _defer_deposit_release(_db_gate_task, manager, grad_gate_cpu, None)
                        else:
                            manager.release_cpu(grad_gate_cpu)
                        grad_gate_cpu = None

                with prof_range(layer._forward_range("moe_finegrained_backward", "up")):
                    if keep_dgrads_hbm:
                        grad_up_stage = grad_up_hbm
                    else:
                        grad_up_stage = manager.stage(grad_up_cpu, tag="moe.dup")
                    d_s_up = _lora_ds(
                        grad_up_stage,
                        up_lora_B,
                        offsets,
                        experts,
                        metadata,
                        scale=layer.lora_scale,
                    )
                    _db_up = None
                    if getattr(ctx, "_asym_db_plan", None) is not None and not keep_dgrads_hbm:
                        _db_up = _try_deposit_lora_b_grad(
                            layer, grad_up_cpu, ctx.up_low_rank_cpu, up_lora_B, offsets, experts, ctx, manager
                        )
                    if _db_up is not None:
                        grad_up_lora_B, _db_up_task = _db_up
                    else:
                        up_low_rank = manager.stage(ctx.up_low_rank_cpu, tag="moe.S_up_for_dB")
                        grad_up_lora_B = _lora_b_grad(
                            layer,
                            grad_up_stage,
                            up_low_rank,
                            up_lora_B,
                            offsets,
                            experts,
                            metadata,
                        )
                        manager.release_stage(up_low_rank, drop_cache=True)
                        manager.release_cpu(ctx.up_low_rank_cpu)
                    if x_compact:
                        grad_up_lora_A = _grouped_da_gpu(
                            layer,
                            d_s_up,
                            x_stage,
                            token_indices,
                            routing_weights,
                            offsets,
                            experts,
                            input_weighted=bool(getattr(ctx, "input_weighted", False)),
                            num_experts=int(up_lora_A.shape[0]),
                            out_dtype=up_lora_A.dtype,
                        )
                        manager.release_stage(x_stage, drop_cache=True)
                        x_stage = None
                    else:
                        manager.wait_cpu_ready(ctx.x_cpu)
                        grad_up_lora_A = _lora_a_grad_cpu(
                            layer,
                            d_s_up,
                            ctx.x_cpu.tensor,
                            up_lora_A,
                            offsets,
                            experts,
                            tag="moe.up",
                        )
                    if ctx.needs_input_grad[0]:
                        if grad_packed is None and not route_flags.gateup_dx_scatter:
                            grad_packed = torch.zeros(
                                (int(grad_up_stage.shape[0]), int(layer.hidden_dim)),
                                device=grad_up_stage.device,
                                dtype=ctx.input_dtype,
                            )
                        layer.stats.qwen3_moe_finegrained_up_base_calls += 1
                        if route_flags.gateup_dx_scatter:
                            gateup_dx_scatter_add_(
                                up_base,
                                grad_up_stage,
                                grad_hidden_base_fp32,
                                offsets,
                                experts,
                                token_indices,
                                routing_weights,
                                weighted=bool(getattr(ctx, "input_weighted", False)),
                            )
                        else:
                            up_dx = _base_dx(
                                layer,
                                up_base,
                                grad_up_stage,
                                offsets,
                                experts,
                                part="up",
                                input_dtype=ctx.input_dtype,
                            )
                            grad_packed.add_(up_dx.to(dtype=grad_packed.dtype))
                            del up_dx
                        up_lora_dx = grouped_expert_lora(
                            d_s_up,
                            up_lora_A.transpose(-1, -2),
                            offsets,
                            experts,
                            metadata=metadata,
                        )
                        if route_flags.gateup_dx_scatter:
                            _scatter_routes_add_(
                                grad_hidden_lora,
                                up_lora_dx,
                                token_indices,
                                routing_weights,
                                weighted=bool(getattr(ctx, "input_weighted", False)),
                            )
                        else:
                            grad_packed.add_(up_lora_dx.to(dtype=grad_packed.dtype))
                        del up_lora_dx
                    del d_s_up
                    if keep_dgrads_hbm:
                        grad_up_hbm = None
                        grad_up_stage = None
                    else:
                        manager.release_stage(grad_up_stage, drop_cache=True)
                        if _db_up is not None:
                            from .attention_activation_offload import _defer_deposit_release
                            _defer_deposit_release(_db_up_task, manager, grad_up_cpu, None)
                        else:
                            manager.release_cpu(grad_up_cpu)
                        grad_up_cpu = None

                if route_flags.gateup_dx_scatter and ctx.needs_input_grad[0]:
                    grad_hidden = grad_hidden_base_fp32.to(dtype=ctx.input_dtype)
                    grad_hidden.add_(grad_hidden_lora)
                    del grad_hidden_base_fp32, grad_hidden_lora
                    grad_hidden_base_fp32 = None
                    grad_hidden_lora = None

            manager.release_cpu(ctx.x_cpu)
            if grad_packed is not None:
                grad_packed = grad_packed.to(dtype=ctx.input_dtype)
                with prof_range(layer._forward_range("moe_finegrained_backward", "pack_grad")):
                    if bool(getattr(ctx, "input_weighted", False)):
                        grad_packed.mul_(routing_weights.reshape(-1, 1).to(device=grad_packed.device, dtype=grad_packed.dtype))
                    grad_hidden = torch.zeros(
                        (int(ctx.num_tokens), int(layer.hidden_dim)),
                        device=grad_packed.device,
                        dtype=grad_packed.dtype,
                    )
                    grad_hidden.index_add_(
                        0,
                        token_indices.to(device=grad_packed.device, dtype=torch.long, non_blocking=True),
                        grad_packed,
                    )
                    del grad_packed
            elif grad_hidden is None:
                grad_hidden = None
        finally:
            # Stage-3 deposit: the worker may still be reading act_cpu / dS — wait
            # before the handles go back to the (non-thread-safe) pinned pool.
            for _task in getattr(ctx, "_asym_deposit_tasks", ()):
                cpu_worker.wait(_task)
            if x_stage is not None:
                manager.release_stage(x_stage, drop_cache=True)
            for handle in (
                ctx.x_cpu,
                ctx.gate_cpu,
                ctx.up_cpu,
                ctx.act_cpu,
                ctx.gate_low_rank_cpu,
                ctx.up_low_rank_cpu,
                ctx.down_low_rank_cpu,
                grad_gate_cpu,
                grad_up_cpu,
            ):
                manager.release_cpu(handle)
            layer._last_activation_offload_stats = _record_manager_peaks(layer, manager)

        return (
            grad_hidden,
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
            None,
            None,
        )


def qwen3_moe_finegrained_forward(
    layer: "AsymQwen3Experts",
    hidden_states: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    input_weighted: bool,
    output_weighted: bool,
) -> torch.Tensor:
    return _Qwen3MoeFinegrainedFunction.apply(
        hidden_states,
        offsets,
        experts,
        token_indices,
        routing_weights,
        layer.gate_lora_A,
        layer.gate_lora_B,
        layer.up_lora_A,
        layer.up_lora_B,
        layer.down_lora_A,
        layer.down_lora_B,
        layer,
        bool(input_weighted),
        bool(output_weighted),
    )


def qwen3_moe_finegrained_nograd_forward(
    layer: "AsymQwen3Experts",
    hidden_states: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    input_weighted: bool,
    output_weighted: bool,
) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("qwen3_moe_finegrained_nograd_forward must run under torch.no_grad()")

    offsets = offsets.detach().contiguous()
    experts = experts.detach().contiguous()
    token_indices = token_indices.detach().contiguous()
    routing_weights = routing_weights.detach().contiguous()
    num_tokens = int(hidden_states.shape[0])
    metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
    gate_base, up_base = layer._ensure_qwen3_moe_finegrained_bases()
    layer.stats.qwen3_moe_finegrained_nograd_forward_calls += 1
    down_scatter_block_experts = _down_scatter_block_experts()
    _validate_routed_flags_for_current_path(down_scatter_block_experts=down_scatter_block_experts)
    route_flags = routed_kernel_flags()
    down_scatter_blocks = _expert_blocks(offsets, experts, down_scatter_block_experts)
    _record_down_scatter_block(layer, down_scatter_block_experts, down_scatter_blocks)

    with prof_range(layer._forward_range("moe_finegrained_nograd", "pack_tokens")):
        packed = hidden_states.index_select(
            0,
            token_indices.to(device=hidden_states.device, dtype=torch.long, non_blocking=True),
        ).contiguous()
        if input_weighted:
            packed.mul_(routing_weights.reshape(-1, 1).to(device=packed.device, dtype=packed.dtype))

    with prof_range(layer._forward_range("moe_finegrained_nograd", "gate")):
        layer.stats.qwen3_moe_finegrained_gate_base_calls += 1
        gate = _base_forward(layer, gate_base, packed, offsets, experts, part="gate")
        gate_low_rank = grouped_expert_lora(packed, layer.gate_lora_A, offsets, experts, metadata=metadata)
        gate_delta = _lora_b_forward(
            gate_low_rank,
            layer.gate_lora_B,
            offsets,
            experts,
            metadata,
            scale=layer.lora_scale,
        )
        gate.add_(gate_delta.to(dtype=gate.dtype))
        del gate_low_rank, gate_delta

    with prof_range(layer._forward_range("moe_finegrained_nograd", "up")):
        layer.stats.qwen3_moe_finegrained_up_base_calls += 1
        up = _base_forward(layer, up_base, packed, offsets, experts, part="up")
        up_low_rank = grouped_expert_lora(packed, layer.up_lora_A, offsets, experts, metadata=metadata)
        up_delta = _lora_b_forward(
            up_low_rank,
            layer.up_lora_B,
            offsets,
            experts,
            metadata,
            scale=layer.lora_scale,
        )
        up.add_(up_delta.to(dtype=up.dtype))
        del packed, up_low_rank, up_delta

    with prof_range(layer._forward_range("moe_finegrained_nograd", "activation")):
        F.silu(gate, inplace=True)
        gate.mul_(up)
        act = gate
        del up, gate

    with prof_range(layer._forward_range("moe_finegrained_nograd", "down_lora")):
        down_low_rank = grouped_expert_lora(act, layer.down_lora_A, offsets, experts, metadata=metadata)
        scattered = torch.zeros(
            (int(num_tokens), int(layer.hidden_dim)),
            device=act.device,
            dtype=act.dtype,
        )
        if down_scatter_block_experts > 0:
            layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
            for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                block_metadata = prepare_grouped_lora_metadata(block_offsets, block_experts, dense_experts=True)
                down_delta = _lora_b_forward(
                    down_low_rank[row_slice].contiguous(),
                    layer.down_lora_B,
                    block_offsets,
                    block_experts,
                    block_metadata,
                    scale=layer.lora_scale,
                )
                _scatter_routes_add_(
                    scattered,
                    down_delta,
                    token_indices[row_slice],
                    routing_weights[row_slice],
                    weighted=bool(output_weighted),
                )
                del down_delta
        else:
            down_delta = _lora_b_forward(
                down_low_rank,
                layer.down_lora_B,
                offsets,
                experts,
                metadata,
                scale=layer.lora_scale,
            )
            _scatter_routes_add_(
                scattered,
                down_delta,
                token_indices,
                routing_weights,
                weighted=bool(output_weighted),
            )
            del down_delta
        del down_low_rank

    with prof_range(layer._forward_range("moe_finegrained_nograd", "down_base")):
        if down_scatter_block_experts > 0:
            layer.stats.qwen3_moe_finegrained_hidden_route_global_tensors_avoided += 1
            for row_start, row_end, block_offsets, block_experts, row_slice in down_scatter_blocks:
                layer.stats.qwen3_moe_finegrained_down_base_calls += 1
                output = _base_forward(layer, layer.down_base, act[row_slice].contiguous(), block_offsets, block_experts, part="down")
                _scatter_routes_add_(
                    scattered,
                    output,
                    token_indices[row_slice],
                    routing_weights[row_slice],
                    weighted=bool(output_weighted),
                )
                del output
            del act
        else:
            layer.stats.qwen3_moe_finegrained_down_base_calls += 1
            if route_flags.fwd_scatter:
                base_scattered_fp32 = torch.zeros(
                    (int(num_tokens), int(layer.hidden_dim)),
                    device=act.device,
                    dtype=torch.float32,
                )
                down_forward_scatter_add_(
                    layer.down_base,
                    act,
                    base_scattered_fp32,
                    offsets,
                    experts,
                    token_indices,
                    routing_weights,
                    weighted=bool(output_weighted),
                )
                scattered.add_(base_scattered_fp32.to(dtype=scattered.dtype))
                del base_scattered_fp32
                output = None
            else:
                output = _base_forward(layer, layer.down_base, act, offsets, experts, part="down")
            del act

    if down_scatter_block_experts <= 0 and not route_flags.fwd_scatter:
        with prof_range(layer._forward_range("moe_finegrained_nograd", "scatter_down_base")):
            _scatter_routes_add_(
                scattered,
                output,
                token_indices,
                routing_weights,
                weighted=bool(output_weighted),
            )
            del output

    return scattered


def qwen3_moe_finegrained_unsupported_reasons(layer: "AsymQwen3Experts", hidden_states: torch.Tensor) -> list[str]:
    from .qwen3_moe import _is_silu_activation

    reasons: list[str] = []
    if layer.backend != "asym" or not layer.offload:
        reasons.append("requires backend='asym' with expert base CPU offload")
    if not isinstance(layer.gate_up_base, AsymGroupedFrozenLinear) or not isinstance(layer.down_base, AsymGroupedFrozenLinear):
        reasons.append("requires AsymGroupedFrozenLinear fused gate/up and down bases")
    if layer.expert_recompute_config.enabled:
        reasons.append("expert recompute/activation-drop policy must be disabled")
    if layer.lora_dropout_p != 0.0:
        reasons.append("v0 supports lora_dropout=0.0 only")
    if not _is_silu_activation(layer.act_fn):
        reasons.append("requires SiLU activation")
    if hidden_states.device.type != "cuda":
        reasons.append("requires CUDA packed expert input")
    if hidden_states.dtype != torch.bfloat16:
        reasons.append(f"requires bf16 packed expert input, got {hidden_states.dtype}")
    if not hidden_states.is_contiguous():
        reasons.append("requires contiguous packed expert input")
    if layer.lora_dtype != hidden_states.dtype:
        reasons.append(f"requires LoRA dtype to match packed dtype, got {layer.lora_dtype} vs {hidden_states.dtype}")
    if layer.hidden_dim % 64 != 0 or layer.intermediate_dim % 64 != 0:
        reasons.append("requires hidden_dim and intermediate_dim multiples of 64 for direct grouped BF16 AsymGEMM")
    for name, param in (
        ("gate_lora_A", layer.gate_lora_A),
        ("gate_lora_B", layer.gate_lora_B),
        ("up_lora_A", layer.up_lora_A),
        ("up_lora_B", layer.up_lora_B),
        ("down_lora_A", layer.down_lora_A),
        ("down_lora_B", layer.down_lora_B),
    ):
        if param.device != hidden_states.device:
            reasons.append(f"{name} must be on {hidden_states.device}, got {param.device}")
        if param.dtype != hidden_states.dtype:
            reasons.append(f"{name} must have dtype {hidden_states.dtype}, got {param.dtype}")
        if not param.is_contiguous():
            reasons.append(f"{name} must be contiguous")
    if isinstance(layer.gate_up_base, AsymGroupedFrozenLinear):
        if layer.gate_up_base.precision != "bf16" or layer.gate_up_base.backend != "asym":
            reasons.append("gate/up base must use direct bf16 AsymGEMM")
        if torch.cuda.is_available() and not layer.gate_up_base.host_weight.weight.is_pinned():
            reasons.append("gate/up host weight must be pinned CPU memory")
    if isinstance(layer.down_base, AsymGroupedFrozenLinear):
        if layer.down_base.precision != "bf16" or layer.down_base.backend != "asym":
            reasons.append("down base must use direct bf16 AsymGEMM")
        if torch.cuda.is_available() and not layer.down_base.host_weight.weight.is_pinned():
            reasons.append("down host weight must be pinned CPU memory")
    kernel_reason = require_expert_activation_offload_kernels(scope="forward", check_only=True)
    if kernel_reason is not None:
        reasons.append(kernel_reason)
    grad_kernel_reason = _missing_symbol(LORA_A_GRAD_CPU_RIGHT)
    if grad_kernel_reason is not None:
        reasons.append(grad_kernel_reason)
    return reasons
