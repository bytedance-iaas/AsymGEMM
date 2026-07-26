"""Fine-grained activation offload for dense Qwen-style MLPs.

This path is intentionally separate from ``dense_mlp.py``.  The older dense path
adapts the MLP to the one-expert Qwen3 expert engine, which fuses gate/up and can
run the full dense activation backward on CPU.  This module keeps gate, up, and
down as separate dense projections so the backward can stage one logical operand
at a time and never call ``stage_concat_columns``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .activation_offload import ActivationOffloadManager, CPUActivationHandle, fg_chunk_rows
from . import activation_offload as _act_offload
from . import cpu_ops
from . import cpu_worker
from . import placement_policy
from .exp_act_offload_lora import (
    grouped_lora_a_forward_cpu_left,
    grouped_lora_a_grad_cpu_right,
    require_expert_activation_offload_kernels,
)
from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, asym_bf16_cpu_right_matmul
from .lora import AsymLoRALinear, normalize_lora_dtype
from .offload import adopt_host_weight
from .profile_ranges import prof_range, scoped_name


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _finegrained_enabled() -> bool:
    return _env_enabled("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD"
    )


def _finegrained_cpu_activation_enabled() -> bool:
    if placement_policy.enabled():
        # P8 kill-switch (fix_cpu_compute.md item 1): dense CPU compute stays OFF
        # until the pinned-bytes cap exists (every 32B arm host-OOM'd).
        return placement_policy.dense_cpu_compute()
    return _env_enabled("ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT"
    )


def _finegrained_nograd_cpu_offload_enabled() -> bool:
    return _env_enabled("ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_NOGRAD_CPU_OFFLOAD"
    )


def _finegrained_keep_acts_hbm_enabled() -> bool:
    """Dense twin of ASYMM_QWEN3_MOE_FG_KEEP_ACTS_HBM (fix_throughput D12 nsys receipt:
    dense backward was 69% host-gap at the fg staging boundaries — tensors here are
    [M, inter] = 20 GB-class on q3-32b). Keep X/gate/up/act/S_* as HBM tensors across
    the within-layer fwd->bwd window; LoRA-A fwd and dA fall through to plain GPU
    matmuls via the is_cuda branches in _cpu_left_lora_a/_cpu_right_lora_a_grad."""
    return _env_enabled("ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM"
    )


def _dense_cpu_act_async_enabled() -> bool:
    """K-3 (cpu_compute.md): dense SwiGLU on the CPU worker — silu(gate) hidden under the
    GPU up-block, `*= up` chained after the up D2H (mirrors the MoE FG async path)."""
    if placement_policy.enabled():
        return placement_policy.dense_cpu_compute()  # P8 kill-switch
    return _env_enabled("ASYMM_DENSE_MLP_FG_CPU_ACT_ASYNC") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FG_CPU_ACT_ASYNC"
    )


def _dense_lora_a_grad_cpu_deposit_enabled() -> bool:
    """K-3: dense LoRA-A wgrad deposit (same flag as the MoE deposit)."""
    if placement_policy.enabled():
        return placement_policy.dense_cpu_compute()  # P8 kill-switch
    return _env_enabled("ASYMM_LORA_A_GRAD_CPU") or _env_enabled(
        "ASYM_GEMM_LF_CONFIG_ASYMM_LORA_A_GRAD_CPU"
    )


_DENSE_PAIR_META: dict = {}
_DENSE_DEPOSIT_DIAG = False


def _try_deposit_dense_lora_a_grad(layer, lora_a_2d, d_s, source, tag, deposit_ctx):
    """Single-group CPU wgrad deposit: dA = dS^T @ source_cpu written fp32 into the
    optimizer buffer. On success records (handle, task) in deposit_ctx and returns a
    dummy CUDA grad; the caller must NOT release `source` (deferred sweep owns it)."""
    import asym_gemm as _ag
    from . import cpu_adam as _cpu_adam
    from .qwen3_moe_finegrained import _DS_SLOTS, _DsSlots

    kernel = getattr(_ag, "cpu_grouped_lora_a_grad_bf16", None)
    adam = _cpu_adam.get_active_adamw()
    if kernel is None or adam is None or not cpu_worker.enabled():
        return None
    # 2026-07-15 fix (32B OOM): sweep the deferred-release list at EVERY dense deposit —
    # with attention deposits off, the old cadence (attn sites + drain only) retained
    # up to 64 layers x ~15.7 GB of pinned handles before the step-end sweep.
    from .attention_activation_offload import _sweep_attn_deposit_releases

    _sweep_attn_deposit_releases()
    src = source.tensor
    if d_s.dtype != torch.bfloat16 or src.dtype != torch.bfloat16 or not src.is_contiguous():
        return None
    buf = adam.get_grad_deposit_buffer(lora_a_2d)
    if (
        buf is None
        or buf.dtype != torch.float32
        or tuple(buf.shape) != (int(d_s.shape[1]), int(src.shape[1]))
    ):
        return None
    m = int(d_s.shape[0])
    meta = _DENSE_PAIR_META.get(m)
    if meta is None:
        meta = (torch.tensor([0, m], dtype=torch.long), torch.zeros(1, dtype=torch.long))
        _DENSE_PAIR_META[m] = meta
    pairs, ge = meta
    slots = _DS_SLOTS.setdefault((tuple(d_s.shape), d_s.dtype, f"dense.{tag}"), _DsSlots())
    slot_i, ds_pin = slots.acquire(d_s)
    ds_pin.copy_(d_s if d_s.is_contiguous() else d_s.contiguous(), non_blocking=True)
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream())
    nt = int(os.environ.get("ASYM_CPU_OPS_THREADS", "32"))
    out3d = buf.view(1, int(buf.shape[0]), int(buf.shape[1]))

    def _job(ev=ev, ds=ds_pin, x=src, out=out3d, p=pairs, g=ge, n=nt, k=kernel):
        ev.synchronize()  # same-stream FIFO also covers the source's earlier D2H
        k(ds, x, out, p, g, n)

    task = cpu_worker.submit_deposit(_job, tag="deposit.dA.dense")
    slots.tasks[slot_i] = task
    if not adam.register_grad_deposit(lora_a_2d, task):
        cpu_worker.wait(task)
        return None
    deposit_ctx[id(source)] = (source, task)  # last task wins (FIFO covers earlier ones)
    global _DENSE_DEPOSIT_DIAG
    if not _DENSE_DEPOSIT_DIAG:
        _DENSE_DEPOSIT_DIAG = True
        import sys
        print("[asym-cpu-wgrad] K-3 dense deposit path ENGAGED", file=sys.stderr, flush=True)
    return torch.empty(lora_a_2d.shape, device=d_s.device, dtype=torch.bfloat16)


def _is_silu_activation(fn: Any) -> bool:
    if fn is F.silu or isinstance(fn, torch.nn.SiLU):
        return True
    name = getattr(fn, "__name__", "") or type(fn).__name__
    return name in {"silu", "silu_python", "SiLU", "SiLUActivation"}


def _asym_base_forward(
    base: AsymFrozenLinear,
    x: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
) -> torch.Tensor:
    out = asym_bf16_cpu_right_matmul(
        x,
        base.host_weight.weight,
        backend=base.backend,
        stats=stats,
        phase="forward",
        tag=tag,
        compiled_dims=base.compiled_dims,
        output_dtype=base.bf16_output_dtype,
    )
    if base.bias_cpu is not None:
        out = out + base.bias_cpu.to(device=out.device, dtype=out.dtype, non_blocking=base.bias_cpu.is_pinned())
    return out


def _asym_base_dx(
    base: AsymFrozenLinear,
    grad_output: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    return asym_bf16_cpu_right_matmul(
        grad_output,
        base.host_weight.weight,
        transpose_b=True,
        backend=base.backend,
        stats=stats,
        phase="dx",
        tag=tag,
        compiled_dims=base.compiled_dims,
        output_dtype=input_dtype,
    )


def _lora_b_forward(low_rank: torch.Tensor, b: torch.Tensor, *, scale: float) -> torch.Tensor:
    out = low_rank @ b.t()
    return out.mul(float(scale)) if float(scale) != 1.0 else out


def _fused_lora_addmm_enabled() -> bool:
    """C2 elementwise diet (agent/impls/fix_asym.md S2/S0'): fuse `dest += (x@W)*scale`
    into a single addmm_ epilogue instead of matmul -> scale-mul -> cast -> add_ (two
    extra full-width elementwise sweeps + a [M,out] temp per call; the launch-bound
    elementwise_kernel bucket). alpha applies in fp32 accumulation — numerics-touching
    (slightly BETTER rounding), so flag-gated. Default off = pre-fix behavior."""
    value = os.environ.get("ASYMM_FUSED_LORA_ADDMM")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _add_lora_b_delta_(dest: torch.Tensor, low_rank: torch.Tensor, b: torch.Tensor, *, scale: float) -> None:
    """dest += (low_rank @ b^T) * scale without materializing a full-width delta:
    the [M, out] product is the same size as dest (20 GB-class at long seq), so
    compute and add it in row chunks."""
    rows = int(dest.shape[0])
    fused = (
        _fused_lora_addmm_enabled()
        and dest.dim() == 2
        and low_rank.dim() == 2
        and dest.dtype == low_rank.dtype == b.dtype
    )
    chunk = fg_chunk_rows(rows, int(dest.shape[1]), dest.element_size())
    if chunk <= 0:
        if fused:
            dest.addmm_(low_rank, b.t(), alpha=float(scale))
            return
        dest.add_(_lora_b_forward(low_rank, b, scale=scale).to(dtype=dest.dtype))
        return
    for row_start in range(0, rows, chunk):
        row_end = min(rows, row_start + chunk)
        if fused:
            dest[row_start:row_end].addmm_(low_rank[row_start:row_end], b.t(), alpha=float(scale))
            continue
        delta = _lora_b_forward(low_rank[row_start:row_end], b, scale=scale)
        dest[row_start:row_end].add_(delta.to(dtype=dest.dtype))
        del delta


def _release_chunk_stages(manager: ActivationOffloadManager, stages: dict[int, torch.Tensor]) -> None:
    for staged in stages.values():
        manager.release_stage(staged, drop_cache=True)
    stages.clear()


def _lora_b_grad(grad_output: torch.Tensor, low_rank: torch.Tensor, *, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    grad = grad_output.t().contiguous() @ low_rank
    if float(scale) != 1.0:
        grad = grad.mul(float(scale))
    return grad.to(dtype=out_dtype)


def _lora_ds(grad_output: torch.Tensor, b: torch.Tensor, *, scale: float) -> torch.Tensor:
    grad_lora = grad_output if grad_output.dtype == b.dtype else grad_output.to(dtype=b.dtype)
    out = grad_lora @ b
    return out.mul(float(scale)) if float(scale) != 1.0 else out


def _dense_lora_a(lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
    if lora_a.dim() == 2:
        return lora_a
    if lora_a.dim() == 3 and int(lora_a.shape[0]) == 1:
        return lora_a.squeeze(0)
    raise RuntimeError(f"{tag}: fine-grained dense MLP expected LoRA-A [r,K], got {tuple(lora_a.shape)}")


def _gpu_lora_a_forward(source: torch.Tensor, lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
    dense = _dense_lora_a(lora_a, tag=tag)
    lhs = source if source.dtype == dense.dtype else source.to(dtype=dense.dtype)
    return lhs.matmul(dense.t())


def _grouped_lora_a(lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
    dense = _dense_lora_a(lora_a, tag=tag)
    return dense.unsqueeze(0).contiguous()


def _cpu_silu_mul(
    gate: CPUActivationHandle,
    up: CPUActivationHandle,
    manager: ActivationOffloadManager,
    *,
    tag: str,
) -> CPUActivationHandle:
    fused = cpu_ops.fused_silu_kernels()
    if fused is not None and cpu_ops.fused_silu_applicable(gate.tensor, up.tensor):
        fused_fwd, _, num_threads = fused
        # host math needs a HOST wait on the producing D2H (stream wait is not enough)
        manager.host_wait_cpu_ready(gate)
        manager.host_wait_cpu_ready(up)
        out = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, tag)
        fused_fwd(gate.tensor, up.tensor, out.tensor, num_threads)
        return out
    manager.wait_cpu_ready_host(gate)
    manager.wait_cpu_ready_host(up)
    out = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, tag)
    with torch.no_grad():
        out.tensor.copy_(F.silu(gate.tensor).mul(up.tensor), non_blocking=False)
    return out


def _cpu_silu_backward(
    gate: CPUActivationHandle,
    up: CPUActivationHandle,
    grad_act: CPUActivationHandle,
    manager: ActivationOffloadManager,
) -> tuple[CPUActivationHandle, CPUActivationHandle]:
    fused = cpu_ops.fused_silu_kernels()
    if fused is not None and cpu_ops.fused_silu_applicable(gate.tensor, up.tensor, grad_act.tensor):
        _, fused_bwd, num_threads = fused
        manager.host_wait_cpu_ready(gate)
        manager.host_wait_cpu_ready(up)
        manager.host_wait_cpu_ready(grad_act)
        grad_gate = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, "mlp.dgate")
        grad_up = manager.empty_cpu(tuple(up.tensor.shape), up.tensor.dtype, up.original_device, "mlp.dup")
        fused_bwd(gate.tensor, up.tensor, grad_act.tensor, grad_gate.tensor, grad_up.tensor, num_threads)
        return grad_gate, grad_up
    manager.wait_cpu_ready_host(gate)
    manager.wait_cpu_ready_host(up)
    manager.wait_cpu_ready_host(grad_act)
    grad_gate = manager.empty_cpu(tuple(gate.tensor.shape), gate.tensor.dtype, gate.original_device, "mlp.dgate")
    grad_up = manager.empty_cpu(tuple(up.tensor.shape), up.tensor.dtype, up.original_device, "mlp.dup")
    with torch.no_grad():
        silu = F.silu(gate.tensor)
        grad_up.tensor.copy_(grad_act.tensor.mul(silu), non_blocking=False)
        grad_gate.tensor.copy_(torch.ops.aten.silu_backward(grad_act.tensor.mul(up.tensor), gate.tensor), non_blocking=False)
    return grad_gate, grad_up


@dataclass(frozen=True)
class _OneExpertPlan:
    offsets: torch.Tensor
    experts: torch.Tensor


class _FinegrainedDenseMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        down_a: torch.Tensor,
        down_b: torch.Tensor,
        layer: "AsymFinegrainedDenseMLP",
    ) -> torch.Tensor:
        if x.dim() < 1:
            raise ValueError("fine-grained dense MLP expects a hidden dimension")
        placement_policy.register_model_class("dense")
        weight_offload = getattr(layer, "_weight_offload", None) is not None
        if weight_offload:
            layer.gather_lora_weights()
            gate_a = layer.gate_proj.lora_a
            gate_b = layer.gate_proj.lora_b
            up_a = layer.up_proj.lora_a
            up_b = layer.up_proj.lora_b
            down_a = layer.down_proj.lora_a
            down_b = layer.down_proj.lora_b

        flat = x.reshape(-1, layer.hidden_size).contiguous()
        if flat.dtype != torch.bfloat16:
            raise ValueError(f"fine-grained dense MLP requires bf16 input, got {flat.dtype}")

        manager = ActivationOffloadManager(pin_memory=True)
        if _finegrained_keep_acts_hbm_enabled():
            if layer.cpu_activation:
                raise RuntimeError(
                    "ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM requires "
                    "ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT=0 (the CPU silu paths read "
                    "pinned CPU handles and are not rerouted)."
                )
            from .qwen3_moe_finegrained import _HBMKeepManager

            manager = _HBMKeepManager()
        layer.stats.dense_mlp_finegrained_forward_calls += 1

        with prof_range(layer._forward_range("finegrained", "x_to_cpu")):
            x_cpu = manager.offload(flat, "mlp.X")

        with prof_range(layer._forward_range("finegrained", "gate")):
            layer.stats.dense_mlp_finegrained_gate_base_calls += 1
            gate = _asym_base_forward(
                layer.gate_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("gate", "base_forward"),
            )
            manager.wait_cpu_ready_host(x_cpu)
            gate_low_rank = layer._cpu_left_lora_a(x_cpu, gate_a, tag="gate")
            _add_lora_b_delta_(gate, gate_low_rank, gate_b, scale=layer.lora_scale)
            gate_cpu = manager.offload(gate, "mlp.gate")
            gate_low_rank_cpu = manager.offload(gate_low_rank, "mlp.S_gate")
            del gate, gate_low_rank

        # K-3 async CPU act: silu(gate) on the worker, hidden under the GPU up-block
        act_task = None
        act_cpu_async = None
        _dense_split = None
        if (
            not layer.cpu_activation
            and not _finegrained_keep_acts_hbm_enabled()
            and _dense_cpu_act_async_enabled()
            and cpu_worker.enabled()
            and cpu_ops.fused_silu_applicable(gate_cpu.tensor)
        ):
            from .qwen3_moe_finegrained import _cpu_act_max_bytes

            if int(gate_cpu.tensor.numel()) * 2 <= _cpu_act_max_bytes():
                _dense_split = cpu_ops.split_silu_kernels()
        if _dense_split is not None:
            _silu_k, _mul_k, _cpu_nt = _dense_split
            act_cpu_async = manager.empty_cpu(
                tuple(gate_cpu.tensor.shape), gate_cpu.tensor.dtype, gate_cpu.original_device, "mlp.act"
            )
            _ev_g = manager.take_cpu_ready_event(gate_cpu)
            _g_t, _a_t = gate_cpu.tensor, act_cpu_async.tensor

            def _dense_silu_job(ev=_ev_g, g=_g_t, o=_a_t, k=_silu_k, n=_cpu_nt):
                if ev is not None:
                    ev.synchronize()
                k(g, o, n)

            act_task = cpu_worker.submit(_dense_silu_job, tag="dense.silu_fwd")

        with prof_range(layer._forward_range("finegrained", "up")):
            layer.stats.dense_mlp_finegrained_up_base_calls += 1
            up = _asym_base_forward(
                layer.up_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("up", "base_forward"),
            )
            manager.wait_cpu_ready_host(x_cpu)
            up_low_rank = layer._cpu_left_lora_a(x_cpu, up_a, tag="up")
            _add_lora_b_delta_(up, up_low_rank, up_b, scale=layer.lora_scale)
            up_cpu = manager.offload(up, "mlp.up")
            up_low_rank_cpu = manager.offload(up_low_rank, "mlp.S_up")
            del up, up_low_rank

        act_rows = int(gate_cpu.tensor.shape[0])
        act_width = int(gate_cpu.tensor.shape[1])
        act_chunk = fg_chunk_rows(act_rows, act_width) if hasattr(manager, "stage_rows") else 0
        if act_task is not None:
            _ev_u = manager.take_cpu_ready_event(up_cpu)
            _u_t, _a_t2 = up_cpu.tensor, act_cpu_async.tensor

            def _dense_mul_job(ev=_ev_u, u=_u_t, o=_a_t2, k=_mul_k, n=_cpu_nt):
                if ev is not None:
                    ev.synchronize()
                k(o, u, n)

            act_task = cpu_worker.submit(_dense_mul_job, tag="dense.mul")

        if act_task is not None:
            with prof_range(layer._forward_range("finegrained", "activation_cpu_async")):
                cpu_worker.wait(act_task)
                act_cpu = act_cpu_async
        elif layer.cpu_activation:
            with prof_range(layer._forward_range("finegrained", "activation_cpu")):
                act_cpu = _cpu_silu_mul(gate_cpu, up_cpu, manager, tag="mlp.act")
        elif act_chunk > 0:
            with prof_range(layer._forward_range("finegrained", "activation")):
                # Row-chunked silu·mul: never stage full-width gate AND up together.
                act_cpu = manager.empty_cpu((act_rows, act_width), torch.bfloat16, gate_cpu.original_device, "mlp.act")
                manager.wait_cpu_ready(gate_cpu)
                manager.wait_cpu_ready(up_cpu)
                chunk_stages: dict[int, torch.Tensor] = {}
                for row_start in range(0, act_rows, act_chunk):
                    row_end = min(act_rows, row_start + act_chunk)
                    gate_chunk = manager.stage_rows(gate_cpu, row_start, row_end, tag="mlp.gate_for_act_chunk")
                    chunk_stages[int(gate_chunk.data_ptr())] = gate_chunk
                    F.silu(gate_chunk, inplace=True)
                    up_chunk = manager.stage_rows(up_cpu, row_start, row_end, tag="mlp.up_for_act_chunk")
                    chunk_stages[int(up_chunk.data_ptr())] = up_chunk
                    gate_chunk.mul_(up_chunk)
                    act_cpu.tensor[row_start:row_end].copy_(
                        gate_chunk.to(dtype=torch.bfloat16), non_blocking=bool(act_cpu.tensor.is_pinned())
                    )
                    del gate_chunk, up_chunk
                manager.record_cpu_ready(act_cpu)
                _release_chunk_stages(manager, chunk_stages)
        else:
            with prof_range(layer._forward_range("finegrained", "activation")):
                gate_stage = manager.stage(gate_cpu, tag="mlp.gate_for_act")
                F.silu(gate_stage, inplace=True)
                up_stage = manager.stage(up_cpu, tag="mlp.up_for_act", mutable=False)
                gate_stage.mul_(up_stage)
                act_cpu = manager.offload(gate_stage.to(dtype=torch.bfloat16).contiguous(), "mlp.act")
                manager.release_stage(gate_stage, drop_cache=True)
                manager.release_stage(up_stage, drop_cache=True)
                del gate_stage, up_stage

        with prof_range(layer._forward_range("finegrained", "down_lora")):
            manager.wait_cpu_ready_host(act_cpu)
            down_low_rank = layer._cpu_left_lora_a(act_cpu, down_a, tag="down")
            down_low_rank_cpu = manager.offload(down_low_rank, "mlp.S_down")

        with prof_range(layer._forward_range("finegrained", "down_base")):
            act_stage = manager.stage(act_cpu, tag="mlp.act_for_down_base", mutable=False)
            layer.stats.dense_mlp_finegrained_down_base_calls += 1
            out = _asym_base_forward(
                layer.down_proj.base_layer,
                act_stage,
                stats=layer.stats,
                tag=layer._profile_name("down", "base_forward"),
            )
            manager.release_stage(act_stage, drop_cache=True)
            _add_lora_b_delta_(out, down_low_rank, down_b, scale=layer.lora_scale)
            del act_stage, down_low_rank

        if weight_offload:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(gate_a, gate_b, up_a, up_b, down_a, down_b)
        ctx.weight_offload = bool(weight_offload)
        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_cpu = act_cpu
        ctx.gate_low_rank_cpu = gate_low_rank_cpu
        ctx.up_low_rank_cpu = up_low_rank_cpu
        ctx.down_low_rank_cpu = down_low_rank_cpu
        ctx.input_shape = tuple(int(dim) for dim in x.shape)
        ctx.input_dtype = x.dtype
        manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)
        layer._last_activation_offload_stats = manager.snapshot()
        if weight_offload and bool(getattr(layer, "_weight_offload_release_after_forward", True)):
            layer.release_lora_weights()
        return out.reshape(*ctx.input_shape[:-1], layer.hidden_size)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        layer: AsymFinegrainedDenseMLP = ctx.layer
        if getattr(ctx, "weight_offload", False):
            layer.gather_lora_weights()
            gate_a = layer.gate_proj.lora_a
            gate_b = layer.gate_proj.lora_b
            up_a = layer.up_proj.lora_a
            up_b = layer.up_proj.lora_b
            down_a = layer.down_proj.lora_a
            down_b = layer.down_proj.lora_b
        else:
            gate_a, gate_b, up_a, up_b, down_a, down_b = ctx.saved_tensors

        manager: ActivationOffloadManager = ctx.manager
        layer.stats.dense_mlp_finegrained_backward_calls += 1
        grad_x = None
        grad_gate_a = grad_gate_b = grad_up_a = grad_up_b = grad_down_a = grad_down_b = None
        grad_gate_cpu = grad_up_cpu = None
        grad_gate_hbm = grad_up_hbm = None
        deposit_ctx: dict = {}  # K-3: id(source_handle) -> (handle, last worker task)

        # R5 restage prefetch (fix_cpu_compute.md): the dense mlp stage family is the
        # single largest measured exposed-wait class (~50 s/step of the 32B step, with
        # ~78 GiB G headroom). Early-issue gate/up H2D on the dedicated prefetch
        # stream at backward entry (hidden under the down blocks), reuse ONE gate
        # stage, and keep dgate/dup ON-GPU (skips their offload->restage roundtrip,
        # the mlp.dgate/mlp.dup classes). G-guarded: needs ~4x gate bytes headroom.
        _pref_gate = _pref_up = None
        _keep_dgrads = False
        if not layer.cpu_activation and _act_offload.restage_prefetch_enabled() and hasattr(manager, "stage_begin"):
            # guard demand: the two held stages (gate+up). The kept dgrads replace
            # same-size offload staging transients, so demanding 4x double-counted —
            # measured on the b32_r5 pair: free flapped at ~64-78 GiB against a
            # 64.8 GiB requirement => only partial engagement (gap 58.5->54.3 of a
            # ~50 s/step target). 2x + the min-free floor is the corrected demand.
            _extra = 2 * int(ctx.gate_cpu.nbytes)
            if _act_offload.prefetch_free_ok(_extra):
                _act_offload.prefetch_engaged_once("mlp.silu_bwd")
                _pref_gate = manager.stage_begin(ctx.gate_cpu, tag="mlp.gate_for_silu_bwd")
                _pref_up = manager.stage_begin(ctx.up_cpu, tag="mlp.up_for_silu_bwd_dgate")
                _keep_dgrads = True

        try:
            grad_2d = grad_output.reshape(-1, layer.hidden_size).to(dtype=torch.bfloat16).contiguous()

            with prof_range(layer._backward_range("finegrained", "down_lora")):
                down_low_rank = manager.stage(ctx.down_low_rank_cpu, tag="mlp.S_down_for_dB", mutable=False)
                dS_down = _lora_ds(grad_2d, down_b, scale=layer.lora_scale)
                grad_down_b = _lora_b_grad(grad_2d, down_low_rank, scale=layer.lora_scale, out_dtype=down_b.dtype)
                manager.release_stage(down_low_rank, drop_cache=True)
                manager.release_cpu(ctx.down_low_rank_cpu)
                manager.wait_cpu_ready_host(ctx.act_cpu)
                grad_down_a = layer._cpu_right_lora_a_grad(dS_down, ctx.act_cpu, down_a, tag="down", deposit_ctx=deposit_ctx)

            with prof_range(layer._backward_range("finegrained", "down_base_dx")):
                layer.stats.dense_mlp_finegrained_down_base_calls += 1
                grad_act = _asym_base_dx(
                    layer.down_proj.base_layer,
                    grad_2d,
                    stats=layer.stats,
                    tag=layer._profile_name("down", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                down_lora_dx = dS_down @ down_a
                grad_act.add_(down_lora_dx.to(dtype=grad_act.dtype))
                del down_lora_dx, dS_down

            if layer.cpu_activation:
                with prof_range(layer._backward_range("finegrained", "activation_bwd_cpu")):
                    layer.stats.dense_mlp_finegrained_cpu_silu_bwd_calls += 1
                    grad_act_cpu = manager.offload(grad_act.to(dtype=torch.bfloat16).contiguous(), "mlp.dact")
                    del grad_act
                    grad_gate_cpu, grad_up_cpu = _cpu_silu_backward(ctx.gate_cpu, ctx.up_cpu, grad_act_cpu, manager)
                    manager.release_cpu(grad_act_cpu)
                    manager.release_cpu(ctx.gate_cpu)
                    manager.release_cpu(ctx.up_cpu)
            elif _pref_gate is not None:
                with prof_range(layer._backward_range("finegrained", "activation_bwd_gpu")):
                    # R5 prefetch path: commit the early-issued stages, reuse ONE gate
                    # stage (out-of-place silu keeps it intact), and keep dgate/dup
                    # ON-GPU (skips the mlp.dup/mlp.dgate offload->restage roundtrip).
                    # Same kernels + operand orders as the legacy sequence => bitwise-
                    # identical dgrads (unit-gated).
                    layer.stats.dense_mlp_finegrained_gpu_silu_bwd_calls += 1
                    gate_stage = manager.stage_commit(
                        *_pref_gate, nbytes=ctx.gate_cpu.nbytes, tag="mlp.gate_for_silu_bwd"
                    )
                    silu_gate = F.silu(gate_stage)
                    silu_gate.mul_(grad_act)
                    grad_up_hbm = silu_gate.to(dtype=torch.bfloat16).contiguous()
                    del silu_gate

                    up_stage = manager.stage_commit(
                        *_pref_up, nbytes=ctx.up_cpu.nbytes, tag="mlp.up_for_silu_bwd_dgate"
                    )
                    grad_act.mul_(up_stage)
                    manager.release_stage(up_stage, drop_cache=True)
                    manager.release_cpu(ctx.up_cpu)
                    del up_stage
                    _pref_up = None

                    grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                    del grad_act
                    grad_gate_hbm = grad_gate.to(dtype=torch.bfloat16).contiguous()
                    manager.release_stage(gate_stage, drop_cache=True)
                    manager.release_cpu(ctx.gate_cpu)
                    del gate_stage, grad_gate
                    _pref_gate = None
            else:
                bwd_rows = int(grad_act.shape[0])
                bwd_width = int(grad_act.shape[1])
                bwd_chunk = fg_chunk_rows(bwd_rows, bwd_width) if hasattr(manager, "stage_rows") else 0
                if bwd_chunk > 0:
                    with prof_range(layer._backward_range("finegrained", "activation_bwd_gpu")):
                        layer.stats.dense_mlp_finegrained_gpu_silu_bwd_calls += 1
                        # Row-chunked silu backward: grad_act stays full-width (it is the
                        # down-dx GEMM output) but gate/up are staged per chunk and dgate/dup
                        # are written straight to their pinned CPU rows.
                        grad_up_cpu = manager.empty_cpu((bwd_rows, bwd_width), torch.bfloat16, grad_act.device, "mlp.dup")
                        grad_gate_cpu = manager.empty_cpu((bwd_rows, bwd_width), torch.bfloat16, grad_act.device, "mlp.dgate")
                        manager.wait_cpu_ready(ctx.gate_cpu)
                        manager.wait_cpu_ready(ctx.up_cpu)
                        chunk_stages: dict[int, torch.Tensor] = {}
                        dup_nb = bool(grad_up_cpu.tensor.is_pinned())
                        dgate_nb = bool(grad_gate_cpu.tensor.is_pinned())
                        for row_start in range(0, bwd_rows, bwd_chunk):
                            row_end = min(bwd_rows, row_start + bwd_chunk)
                            grad_slice = grad_act[row_start:row_end]
                            gate_chunk = manager.stage_rows(ctx.gate_cpu, row_start, row_end, tag="mlp.gate_for_silu_bwd_chunk")
                            chunk_stages[int(gate_chunk.data_ptr())] = gate_chunk
                            dup_chunk = F.silu(gate_chunk)
                            dup_chunk.mul_(grad_slice)
                            grad_up_cpu.tensor[row_start:row_end].copy_(dup_chunk.to(dtype=torch.bfloat16), non_blocking=dup_nb)
                            del dup_chunk
                            up_chunk = manager.stage_rows(ctx.up_cpu, row_start, row_end, tag="mlp.up_for_silu_bwd_chunk")
                            chunk_stages[int(up_chunk.data_ptr())] = up_chunk
                            grad_slice.mul_(up_chunk)
                            dgate_chunk = torch.ops.aten.silu_backward(grad_slice, gate_chunk)
                            grad_gate_cpu.tensor[row_start:row_end].copy_(dgate_chunk.to(dtype=torch.bfloat16), non_blocking=dgate_nb)
                            del dgate_chunk, gate_chunk, up_chunk, grad_slice
                        manager.record_cpu_ready(grad_up_cpu)
                        manager.record_cpu_ready(grad_gate_cpu)
                        _release_chunk_stages(manager, chunk_stages)
                        manager.release_cpu(ctx.gate_cpu)
                        manager.release_cpu(ctx.up_cpu)
                        del grad_act
                else:
                    with prof_range(layer._backward_range("finegrained", "activation_bwd_gpu")):
                        layer.stats.dense_mlp_finegrained_gpu_silu_bwd_calls += 1
                        # Stage gate twice instead of keeping gate and up live together.
                        gate_stage = manager.stage(ctx.gate_cpu, tag="mlp.gate_for_silu_bwd_dup")
                        F.silu(gate_stage, inplace=True)
                        gate_stage.mul_(grad_act)
                        grad_up_cpu = manager.offload(gate_stage.to(dtype=torch.bfloat16).contiguous(), "mlp.dup")
                        manager.release_stage(gate_stage, drop_cache=True)
                        del gate_stage

                        up_stage = manager.stage(ctx.up_cpu, tag="mlp.up_for_silu_bwd_dgate", mutable=False)
                        grad_act.mul_(up_stage)
                        manager.release_stage(up_stage, drop_cache=True)
                        manager.release_cpu(ctx.up_cpu)
                        del up_stage

                        gate_stage = manager.stage(ctx.gate_cpu, tag="mlp.gate_for_silu_bwd_dgate", mutable=False)
                        grad_gate = torch.ops.aten.silu_backward(grad_act, gate_stage)
                        del grad_act
                        grad_gate_cpu = manager.offload(grad_gate.to(dtype=torch.bfloat16).contiguous(), "mlp.dgate")
                        manager.release_stage(gate_stage, drop_cache=True)
                        manager.release_cpu(ctx.gate_cpu)
                        del gate_stage, grad_gate

            with prof_range(layer._backward_range("finegrained", "gate")):
                if grad_gate_hbm is not None:
                    grad_gate_stage = grad_gate_hbm  # R5: dgate kept on-GPU, no restage
                else:
                    grad_gate_stage = manager.stage(grad_gate_cpu, tag="mlp.dgate", mutable=False)
                gate_low_rank = manager.stage(ctx.gate_low_rank_cpu, tag="mlp.S_gate_for_dB", mutable=False)
                dS_gate = _lora_ds(grad_gate_stage, gate_b, scale=layer.lora_scale)
                grad_gate_b = _lora_b_grad(
                    grad_gate_stage,
                    gate_low_rank,
                    scale=layer.lora_scale,
                    out_dtype=gate_b.dtype,
                )
                manager.release_stage(gate_low_rank, drop_cache=True)
                manager.release_cpu(ctx.gate_low_rank_cpu)
                manager.wait_cpu_ready_host(ctx.x_cpu)
                grad_gate_a = layer._cpu_right_lora_a_grad(dS_gate, ctx.x_cpu, gate_a, tag="gate", deposit_ctx=deposit_ctx)
                layer.stats.dense_mlp_finegrained_gate_base_calls += 1
                grad_x = _asym_base_dx(
                    layer.gate_proj.base_layer,
                    grad_gate_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("gate", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                gate_lora_dx = dS_gate @ gate_a
                grad_x.add_(gate_lora_dx.to(dtype=grad_x.dtype))
                del gate_lora_dx, dS_gate
                if grad_gate_hbm is not None:
                    del grad_gate_stage
                    grad_gate_hbm = None
                else:
                    manager.release_stage(grad_gate_stage, drop_cache=True)
                    manager.release_cpu(grad_gate_cpu)
                    grad_gate_cpu = None

            with prof_range(layer._backward_range("finegrained", "up")):
                if grad_up_hbm is not None:
                    grad_up_stage = grad_up_hbm  # R5: dup kept on-GPU, no restage
                else:
                    grad_up_stage = manager.stage(grad_up_cpu, tag="mlp.dup", mutable=False)
                up_low_rank = manager.stage(ctx.up_low_rank_cpu, tag="mlp.S_up_for_dB", mutable=False)
                dS_up = _lora_ds(grad_up_stage, up_b, scale=layer.lora_scale)
                grad_up_b = _lora_b_grad(grad_up_stage, up_low_rank, scale=layer.lora_scale, out_dtype=up_b.dtype)
                manager.release_stage(up_low_rank, drop_cache=True)
                manager.release_cpu(ctx.up_low_rank_cpu)
                manager.wait_cpu_ready_host(ctx.x_cpu)
                grad_up_a = layer._cpu_right_lora_a_grad(dS_up, ctx.x_cpu, up_a, tag="up", deposit_ctx=deposit_ctx)
                layer.stats.dense_mlp_finegrained_up_base_calls += 1
                up_dx = _asym_base_dx(
                    layer.up_proj.base_layer,
                    grad_up_stage,
                    stats=layer.stats,
                    tag=layer._profile_name("up", "base_dx"),
                    input_dtype=torch.bfloat16,
                )
                grad_x.add_(up_dx.to(dtype=grad_x.dtype))
                del up_dx
                up_lora_dx = dS_up @ up_a
                grad_x.add_(up_lora_dx.to(dtype=grad_x.dtype))
                del up_lora_dx, dS_up
                if grad_up_hbm is not None:
                    del grad_up_stage
                    grad_up_hbm = None
                else:
                    manager.release_stage(grad_up_stage, drop_cache=True)
                    manager.release_cpu(grad_up_cpu)
                    grad_up_cpu = None

            grad_x = grad_x.to(dtype=ctx.input_dtype).reshape(ctx.input_shape)
        finally:
            deferred_ids = set(deposit_ctx.keys())
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
                if handle is not None and id(handle) in deferred_ids:
                    continue  # K-3: worker wgrad still reads it; deferred sweep releases
                manager.release_cpu(handle)
            if deposit_ctx:
                from .attention_activation_offload import _defer_deposit_release

                for handle, task in deposit_ctx.values():
                    _defer_deposit_release(task, manager, handle, None)
            layer._last_activation_offload_stats = manager.snapshot()

        return (
            grad_x,
            grad_gate_a,
            grad_gate_b,
            grad_up_a,
            grad_up_b,
            grad_down_a,
            grad_down_b,
            None,
        )


def _finegrained_dense_mlp_no_grad_gpu_forward(layer: "AsymFinegrainedDenseMLP", x: torch.Tensor) -> torch.Tensor:
    weight_offload = getattr(layer, "_weight_offload", None) is not None
    if weight_offload:
        layer.gather_lora_weights()
    gate_a = layer.gate_proj.lora_a
    gate_b = layer.gate_proj.lora_b
    up_a = layer.up_proj.lora_a
    up_b = layer.up_proj.lora_b
    down_a = layer.down_proj.lora_a
    down_b = layer.down_proj.lora_b

    flat = x.reshape(-1, layer.hidden_size).contiguous()
    if flat.dtype != torch.bfloat16:
        raise ValueError(f"fine-grained dense MLP requires bf16 input, got {flat.dtype}")

    manager = ActivationOffloadManager(pin_memory=True)
    layer.stats.dense_mlp_finegrained_forward_calls += 1
    try:
        with prof_range(layer._forward_range("finegrained_nograd_gpu", "gate")):
            layer.stats.dense_mlp_finegrained_gate_base_calls += 1
            gate = _asym_base_forward(
                layer.gate_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("gate", "base_forward"),
            )
            gate_low_rank = _gpu_lora_a_forward(flat, gate_a, tag="gate")
            _add_lora_b_delta_(gate, gate_low_rank, gate_b, scale=layer.lora_scale)
            del gate_low_rank

        with prof_range(layer._forward_range("finegrained_nograd_gpu", "up")):
            layer.stats.dense_mlp_finegrained_up_base_calls += 1
            up = _asym_base_forward(
                layer.up_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("up", "base_forward"),
            )
            up_low_rank = _gpu_lora_a_forward(flat, up_a, tag="up")
            _add_lora_b_delta_(up, up_low_rank, up_b, scale=layer.lora_scale)
            del up_low_rank, flat

        with prof_range(layer._forward_range("finegrained_nograd_gpu", "activation")):
            F.silu(gate, inplace=True)
            gate.mul_(up)
            act = gate
            del up

        with prof_range(layer._forward_range("finegrained_nograd_gpu", "down_lora")):
            down_low_rank = _gpu_lora_a_forward(act, down_a, tag="down")

        with prof_range(layer._forward_range("finegrained_nograd_gpu", "down_base")):
            layer.stats.dense_mlp_finegrained_down_base_calls += 1
            out = _asym_base_forward(
                layer.down_proj.base_layer,
                act,
                stats=layer.stats,
                tag=layer._profile_name("down", "base_forward"),
            )
            del act
            _add_lora_b_delta_(out, down_low_rank, down_b, scale=layer.lora_scale)
            del down_low_rank

        stats = manager.snapshot()
        stats["finegrained_no_grad_gpu_forward"] = True
        stats["finegrained_no_grad_cpu_offload"] = False
        layer._last_activation_offload_stats = stats
        return out.reshape(*x.shape[:-1], layer.hidden_size)
    finally:
        if weight_offload and bool(getattr(layer, "_weight_offload_release_after_forward", True)):
            layer.release_lora_weights()


def _finegrained_dense_mlp_no_grad_cpu_offload_forward(layer: "AsymFinegrainedDenseMLP", x: torch.Tensor) -> torch.Tensor:
    weight_offload = getattr(layer, "_weight_offload", None) is not None
    if weight_offload:
        layer.gather_lora_weights()
    gate_a = layer.gate_proj.lora_a
    gate_b = layer.gate_proj.lora_b
    up_a = layer.up_proj.lora_a
    up_b = layer.up_proj.lora_b
    down_a = layer.down_proj.lora_a
    down_b = layer.down_proj.lora_b

    flat = x.reshape(-1, layer.hidden_size).contiguous()
    if flat.dtype != torch.bfloat16:
        raise ValueError(f"fine-grained dense MLP requires bf16 input, got {flat.dtype}")

    manager = ActivationOffloadManager(pin_memory=True)
    layer.stats.dense_mlp_finegrained_forward_calls += 1
    cpu_handles: list[CPUActivationHandle | None] = []
    stage_tensors: list[torch.Tensor | None] = []
    try:
        with prof_range(layer._forward_range("finegrained_nograd", "x_to_cpu")):
            x_cpu = manager.offload(flat, "mlp.X")
            cpu_handles.append(x_cpu)

        with prof_range(layer._forward_range("finegrained_nograd", "gate")):
            layer.stats.dense_mlp_finegrained_gate_base_calls += 1
            gate = _asym_base_forward(
                layer.gate_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("gate", "base_forward"),
            )
            manager.wait_cpu_ready_host(x_cpu)
            gate_low_rank = layer._cpu_left_lora_a(x_cpu, gate_a, tag="gate")
            gate_delta = _lora_b_forward(gate_low_rank, gate_b, scale=layer.lora_scale)
            gate.add_(gate_delta.to(dtype=gate.dtype))
            gate_cpu = manager.offload(gate, "mlp.gate")
            cpu_handles.append(gate_cpu)
            del gate, gate_delta, gate_low_rank

        with prof_range(layer._forward_range("finegrained_nograd", "up")):
            layer.stats.dense_mlp_finegrained_up_base_calls += 1
            up = _asym_base_forward(
                layer.up_proj.base_layer,
                flat,
                stats=layer.stats,
                tag=layer._profile_name("up", "base_forward"),
            )
            manager.wait_cpu_ready_host(x_cpu)
            up_low_rank = layer._cpu_left_lora_a(x_cpu, up_a, tag="up")
            up_delta = _lora_b_forward(up_low_rank, up_b, scale=layer.lora_scale)
            up.add_(up_delta.to(dtype=up.dtype))
            up_cpu = manager.offload(up, "mlp.up")
            cpu_handles.append(up_cpu)
            del up, up_delta, up_low_rank

        if layer.cpu_activation:
            with prof_range(layer._forward_range("finegrained_nograd", "activation_cpu")):
                act_cpu = _cpu_silu_mul(gate_cpu, up_cpu, manager, tag="mlp.act")
                cpu_handles.append(act_cpu)
                manager.release_cpu(gate_cpu)
                manager.release_cpu(up_cpu)
                gate_cpu = up_cpu = None
        else:
            with prof_range(layer._forward_range("finegrained_nograd", "activation")):
                gate_stage = manager.stage(gate_cpu, tag="mlp.gate_for_act")
                gate_stage_idx = len(stage_tensors)
                stage_tensors.append(gate_stage)
                up_stage = None
                up_stage_idx: int | None = None
                try:
                    F.silu(gate_stage, inplace=True)
                    up_stage = manager.stage(up_cpu, tag="mlp.up_for_act", mutable=False)
                    up_stage_idx = len(stage_tensors)
                    stage_tensors.append(up_stage)
                    gate_stage.mul_(up_stage)
                    act_cpu = manager.offload(gate_stage.to(dtype=torch.bfloat16).contiguous(), "mlp.act")
                    cpu_handles.append(act_cpu)
                finally:
                    if up_stage is not None:
                        manager.release_stage(up_stage, drop_cache=True)
                        if up_stage_idx is not None:
                            stage_tensors[up_stage_idx] = None
                    manager.release_stage(gate_stage, drop_cache=True)
                    stage_tensors[gate_stage_idx] = None
                del gate_stage, up_stage
                manager.release_cpu(gate_cpu)
                manager.release_cpu(up_cpu)
                gate_cpu = up_cpu = None

        with prof_range(layer._forward_range("finegrained_nograd", "down_lora")):
            # act_cpu was offloaded (non-blocking D2H) just above; the cpu-left
            # LoRA-A pads it with a HOST memcpy — this site had NO wait at all.
            manager.wait_cpu_ready_host(act_cpu)
            down_low_rank = layer._cpu_left_lora_a(act_cpu, down_a, tag="down")
            down_delta = _lora_b_forward(down_low_rank, down_b, scale=layer.lora_scale)
            del down_low_rank
            manager.release_cpu(x_cpu)
            x_cpu = None

        with prof_range(layer._forward_range("finegrained_nograd", "down_base")):
            act_stage = manager.stage(act_cpu, tag="mlp.act_for_down_base", mutable=False)
            act_stage_idx = len(stage_tensors)
            stage_tensors.append(act_stage)
            layer.stats.dense_mlp_finegrained_down_base_calls += 1
            out = _asym_base_forward(
                layer.down_proj.base_layer,
                act_stage,
                stats=layer.stats,
                tag=layer._profile_name("down", "base_forward"),
            )
            manager.release_stage(act_stage, drop_cache=True)
            stage_tensors[act_stage_idx] = None
            out.add_(down_delta.to(dtype=out.dtype))
            del act_stage, down_delta

        manager.release_cpu(act_cpu)
        layer._last_activation_offload_stats = manager.snapshot()
        return out.reshape(*x.shape[:-1], layer.hidden_size)
    finally:
        for tensor in stage_tensors:
            manager.release_stage(tensor, drop_cache=True)
        for handle in cpu_handles:
            manager.release_cpu(handle)
        if weight_offload and bool(getattr(layer, "_weight_offload_release_after_forward", True)):
            layer.release_lora_weights()


def _finegrained_dense_mlp_no_grad_forward(layer: "AsymFinegrainedDenseMLP", x: torch.Tensor) -> torch.Tensor:
    if _finegrained_nograd_cpu_offload_enabled():
        return _finegrained_dense_mlp_no_grad_cpu_offload_forward(layer, x)
    return _finegrained_dense_mlp_no_grad_gpu_forward(layer, x)


class AsymFinegrainedDenseMLP(nn.Module):
    """Dense MLP replacement with separate gate/up/down fine-grained scheduling."""

    _is_asym_finegrained_dense_mlp = True

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
        profile_prefix: str = "layers.unknown.mlp",
    ) -> None:
        super().__init__()
        gate = getattr(source, "gate_proj", None)
        up = getattr(source, "up_proj", None)
        down = getattr(source, "down_proj", None)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            raise TypeError("fine-grained dense MLP expects nn.Linear gate_proj/up_proj/down_proj leaves")
        if float(lora_dropout) != 0.0:
            raise NotImplementedError("fine-grained dense MLP currently requires lora_dropout=0.0")
        if backend == "asym":
            require_expert_activation_offload_kernels(scope="full")
        if strict:
            for leaf_name, module in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                if module.weight.device.type != "cpu":
                    raise RuntimeError(f"{leaf_name} fine-grained dense MLP CPU offload requires CPU-first weights")
                if module.weight.dtype != torch.bfloat16:
                    raise RuntimeError(f"{leaf_name} fine-grained dense MLP requires bf16 source weights")

        self.backend = backend
        self.precision = precision
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.activation_fn = getattr(source, "act_fn", F.silu)
        self.hidden_size = int(gate.weight.shape[1])
        self.intermediate_size = int(gate.weight.shape[0])
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(torch.bfloat16)
        self.lora_dropout_p = float(lora_dropout)
        self.profile_prefix = profile_prefix
        self._last_activation_offload_stats: dict[str, Any] = {}
        self._weight_offload = None
        self._weight_offload_release_after_forward = True
        # P8: dense layers put the whole run in the host-RAM-bound regime — register
        # BEFORE evaluating any policy-consulting gate helper below.
        placement_policy.register_model_class("dense")
        self.cpu_activation = _finegrained_cpu_activation_enabled()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.gate_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.gate_proj.weight", gate.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if gate.bias is None else gate.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )
        self.up_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.up_proj.weight", up.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if up.bias is None else up.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )
        self.down_proj = AsymLoRALinear.from_host_weight(
            adopt_host_weight(f"{profile_prefix}.down_proj.weight", down.weight, "mlp_dense", pin_memory_policy="auto", strict=strict),
            bias=None if down.bias is None else down.bias.detach(),
            rank=lora_rank,
            alpha=lora_alpha,
            backend=backend,
            stats=self.stats,
            device=device,
            lora_dtype=self.lora_dtype,
            precision=precision,
            init_lora_weights="peft",
            lora_dropout=lora_dropout,
        )

    @property
    def cpu_resident_base_bytes(self) -> int:
        return sum(
            int(getattr(module, "cpu_resident_base_weight_bytes", 0))
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        )

    @property
    def gpu_resident_base_bytes(self) -> int:
        return sum(
            int(getattr(module, "gpu_resident_base_weight_bytes", 0))
            for module in (self.gate_proj, self.up_proj, self.down_proj)
        )

    @property
    def trainable_lora_params(self) -> int:
        return sum(param.numel() for name, param in self.named_parameters() if "lora_" in name or ".lora_A." in name or ".lora_B." in name)

    def _one_expert_plan(self, rows: int, device: torch.device) -> _OneExpertPlan:
        offsets = torch.tensor([0, int(rows)], device=device, dtype=torch.long)
        experts = torch.tensor([0, -1], device=device, dtype=torch.long)
        return _OneExpertPlan(offsets=offsets, experts=experts)

    def _cpu_left_lora_a(self, source: CPUActivationHandle, lora_a: torch.Tensor, *, tag: str) -> torch.Tensor:
        lora_a = _dense_lora_a(lora_a, tag=tag)
        if source.tensor.is_cuda:
            # keep-acts-HBM: the "offloaded" handle is an HBM tensor — plain GPU matmul.
            return _gpu_lora_a_forward(source.tensor, lora_a, tag=tag)
        if self.backend == "torch" or not source.tensor.is_pinned():
            # item 4 (fix_cpu_compute.md): a pinned-ledger cap denial yields an
            # UNPINNED act handle, which the CPU-left C2C kernel cannot read (the GPU
            # maps page-locked host memory directly). Fall back to the mathematically
            # identical staged-GPU GEMM (lossless; slower — the cap converts OOM risk
            # into bounded slowdown as designed).
            return source.tensor.to(device=lora_a.device, dtype=lora_a.dtype, non_blocking=source.tensor.is_pinned()).matmul(lora_a.t())
        plan = self._one_expert_plan(int(source.tensor.shape[0]), lora_a.device)
        return grouped_lora_a_forward_cpu_left(
            source.tensor,
            _grouped_lora_a(lora_a, tag=tag),
            plan.offsets,
            plan.experts,
            metadata=None,
            stats=self.stats,
            tag=tag,
        )

    def _cpu_right_lora_a_grad(
        self,
        dS: torch.Tensor,
        source: CPUActivationHandle,
        lora_a: torch.Tensor,
        *,
        tag: str,
        deposit_ctx: dict | None = None,
    ) -> torch.Tensor:
        lora_a_param = lora_a  # keep Parameter identity for the deposit lookup
        lora_a = _dense_lora_a(lora_a, tag=tag)
        if source.tensor.is_cuda:
            # keep-acts-HBM: dA = dS^T @ source directly on GPU.
            src = source.tensor if source.tensor.dtype == dS.dtype else source.tensor.to(dtype=dS.dtype)
            return dS.t().contiguous().matmul(src).to(dtype=lora_a.dtype)
        if (
            deposit_ctx is not None
            and self.backend != "torch"
            and _dense_lora_a_grad_cpu_deposit_enabled()
            and lora_a.dim() == 2
        ):
            dummy = _try_deposit_dense_lora_a_grad(self, lora_a_param if lora_a_param.dim() == 2 else lora_a, dS, source, tag, deposit_ctx)
            if dummy is not None:
                return dummy
        if self.backend == "torch" or not source.tensor.is_pinned():
            # item 4: cap-denial fallback (see _cpu_left_lora_a) — the cpu-right wgrad
            # kernel also reads page-locked host memory from the GPU.
            return dS.t().contiguous().matmul(
                source.tensor.to(device=dS.device, dtype=dS.dtype, non_blocking=source.tensor.is_pinned())
            ).to(dtype=lora_a.dtype)
        plan = self._one_expert_plan(int(source.tensor.shape[0]), dS.device)
        grad = grouped_lora_a_grad_cpu_right(
            dS.contiguous(),
            source.tensor,
            plan.offsets,
            plan.experts,
            num_experts=1,
            stats=self.stats,
            tag=tag,
        )
        return grad.squeeze(0)

    def _lora_weight_banks(self) -> list[tuple[str, torch.nn.Parameter]]:
        if not (
            isinstance(self.gate_proj, AsymLoRALinear)
            and isinstance(self.up_proj, AsymLoRALinear)
            and isinstance(self.down_proj, AsymLoRALinear)
        ):
            return []
        return [
            ("gate_lora_A", self.gate_proj.lora_a),
            ("gate_lora_B", self.gate_proj.lora_b),
            ("up_lora_A", self.up_proj.lora_a),
            ("up_lora_B", self.up_proj.lora_b),
            ("down_lora_A", self.down_proj.lora_a),
            ("down_lora_B", self.down_proj.lora_b),
        ]

    def gather_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.gather_group(self)

    def release_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.release_group(self)

    def _profile_name(self, *parts: object) -> str:
        return scoped_name(self.profile_prefix, *parts)

    def _forward_range(self, *parts: object) -> str:
        return scoped_name("forward", self.profile_prefix, *parts)

    def _backward_range(self, *parts: object) -> str:
        return scoped_name("backward", self.profile_prefix, *parts)

    def _activation_offload_supported(self, x: torch.Tensor) -> bool:
        return (
            self.backend in {"asym", "torch"}
            and self.training
            and _finegrained_enabled()
            and self.lora_dropout_p == 0.0
            and self.lora_dtype == torch.bfloat16
            and x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and _is_silu_activation(self.activation_fn)
            and isinstance(self.gate_proj, AsymLoRALinear)
            and isinstance(self.up_proj, AsymLoRALinear)
            and isinstance(self.down_proj, AsymLoRALinear)
            and self.gate_proj.base_layer.precision == "bf16"
            and self.up_proj.base_layer.precision == "bf16"
            and self.down_proj.base_layer.precision == "bf16"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation_offload_supported(x):
            if not torch.is_grad_enabled():
                return _finegrained_dense_mlp_no_grad_forward(self, x)
            return _FinegrainedDenseMLPFunction.apply(
                x,
                self.gate_proj.lora_a,
                self.gate_proj.lora_b,
                self.up_proj.lora_a,
                self.up_proj.lora_b,
                self.down_proj.lora_a,
                self.down_proj.lora_b,
                self,
            )
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation_fn(gate) * up)


def build_finegrained_dense_mlp(
    mlp: nn.Module,
    *,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    stats: AsymExecutionStats | None = None,
    strict: bool = True,
    profile_prefix: str = "layers.unknown.mlp",
) -> AsymFinegrainedDenseMLP:
    return AsymFinegrainedDenseMLP(
        mlp,
        backend=backend,
        precision=precision,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        stats=stats,
        strict=strict,
        profile_prefix=profile_prefix,
    )


__all__ = ["AsymFinegrainedDenseMLP", "build_finegrained_dense_mlp"]
