from __future__ import annotations

from typing import Literal

import os

import torch
from torch import nn

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .exp_act_offload_lora import (
    grouped_lora_a_forward_cpu_left,
    grouped_lora_a_forward_hbm,
    grouped_lora_a_grad_cpu_right,
    grouped_lora_a_pair_forward_cpu_left,
    grouped_lora_a_pair_grad_cpu_right,
    stage_low_rank_from_cpu,
)
from .lora import (
    GroupedLoRAMetadata,
    grouped_expert_lora,
    grouped_expert_lora_pair,
    normalize_lora_dtype,
    prepare_grouped_lora_metadata,
)
from .frozen_linear import (
    AsymExecutionStats,
    AsymGroupedFrozenLinear,
    TorchGroupedFrozenLinear,
    _dispatch_grouped_nt,
    _get_quantized_host_weight,
    _grouped_torch_chunks,
)
from .moe import build_contiguous_route_metadata, make_dense_group_metadata, pack_tokens_contiguous, parse_expert_recompute_policy_spec
from .profile_ranges import prof_range
from .qwen3_moe import (
    AsymQwen3Experts,
    _activation_offload_cpu_silu_backward,
    _activation_offload_cpu_silu_mul,
    _expert_act_offload_lora_a_fwd_mode,
    _grouped_lora_cuda_view,
    _grouped_lora_weight_grads_torch,
    _is_silu_activation,
    _reset_lora_bank,
    _resolve_device,
    _scatter_contiguous_sum,
)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


def _llama4_act_recompute() -> bool:
    return _env_enabled("ASYM_OFFLOAD_ACT_RECOMPUTE")


def _llama4_x_unpacked() -> bool:
    return _env_enabled("ASYM_OFFLOAD_X_UNPACKED")


def _llama4_grouped_base_dx(
    base: nn.Module,
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    dense_experts: bool,
    input_dtype: torch.dtype,
    profile_name: str,
) -> torch.Tensor:
    if isinstance(base, TorchGroupedFrozenLinear):
        grad_2d = grad_output.reshape(-1, int(base.out_features)).contiguous()
        with prof_range(f"backward.{profile_name}.grouped_base_dx_torch"):
            grad_x = _grouped_torch_chunks(
                grad_2d,
                base.weight,
                offsets,
                experts,
                transpose_b=bool(base.backward_transpose_b),
                dense_experts=dense_experts,
            )
        return grad_x.reshape(*grad_output.shape[:-1], int(base.in_features))

    if isinstance(base, AsymGroupedFrozenLinear):
        grad_2d = grad_output.reshape(-1, int(base.out_features)).contiguous()
        if base.precision == "bf16" and base.backend != "torch" and grad_2d.dtype != base.host_weight.weight.dtype:
            grad_2d = grad_2d.to(dtype=base.host_weight.weight.dtype)
        quantized_weight_t = (
            _get_quantized_host_weight(base.host_weight, base.precision, transpose=bool(base.backward_transpose_b))
            if base.backend != "torch" and base.precision != "bf16"
            else None
        )
        with prof_range(f"backward.{profile_name}.grouped_base_dx_asymgemm"):
            grad_x = _dispatch_grouped_nt(
                grad_2d,
                base.host_weight.weight,
                offsets,
                experts,
                backend=base.backend,
                stats=base.stats,
                phase="dx",
                compiled_dims=base.compiled_dims,
                transpose_b=bool(base.backward_transpose_b),
                precision=base.precision,
                quantized_weight=quantized_weight_t,
                dense_experts=dense_experts,
                bf16_output_dtype=input_dtype if base.precision == "bf16" else base.bf16_output_dtype,
            )
        return grad_x.reshape(*grad_output.shape[:-1], int(base.in_features))

    raise TypeError(f"unsupported grouped base module {type(base).__name__}")


def _rebuild_packed_x_cpu(
    ctx,
    manager: ActivationOffloadManager,
) -> tuple[CPUActivationHandle, bool]:
    if not getattr(ctx, "x_unpacked", False):
        return ctx.x_cpu, False
    hidden_cpu = ctx.x_cpu.tensor
    token_indices = getattr(ctx, "x_token_indices_cpu", None)
    if token_indices is None:
        raise RuntimeError("Llama4 gate/up recompute from unpacked X requires token indices")
    manager.wait_cpu_ready(ctx.x_cpu)
    rebuilt = manager.empty_cpu(
        (int(token_indices.shape[0]), int(hidden_cpu.shape[1])),
        hidden_cpu.dtype,
        ctx.x_cpu.original_device,
        "X",
    )
    torch.index_select(hidden_cpu, 0, token_indices, out=rebuilt.tensor)
    route_scale = getattr(ctx, "x_route_scale_cpu", None)
    if route_scale is not None:
        rebuilt.tensor.mul_(route_scale.reshape(-1, 1).to(dtype=rebuilt.tensor.dtype))
    return rebuilt, True


def _recompute_gate_up_cpu(
    ctx,
    layer: "AsymLlama4Experts",
    manager: ActivationOffloadManager,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    gate_lora_A: torch.Tensor,
    gate_lora_B: torch.Tensor,
    up_lora_A: torch.Tensor,
    up_lora_B: torch.Tensor,
    lora_metadata: GroupedLoRAMetadata,
) -> tuple[CPUActivationHandle, CPUActivationHandle]:
    x_handle, release_x_handle = _rebuild_packed_x_cpu(ctx, manager)
    x_stage = None
    gate_up_low_rank_owner = None
    lora_a_forward_mode = getattr(ctx, "expert_lora_a_forward_mode", _expert_act_offload_lora_a_fwd_mode())
    try:
        with prof_range("backward.llama4_mlp.activation_offload.gate_up_recompute_stage_x"):
            x_stage = manager.stage(x_handle, tag="X_for_gate_up_recompute")
        with prof_range("backward.llama4_mlp.activation_offload.gate_up_recompute_base"):
            gate_up = layer.gate_up_base(
                x_stage,
                offsets,
                experts,
                dense_experts=True,
                compiled_dims="nk",
                profile_name=layer._profile_name("gate_up", "base"),
            )
            gate, up = gate_up.chunk(2, dim=-1)
        with prof_range("backward.llama4_mlp.activation_offload.gate_up_recompute_lora_a"):
            if lora_a_forward_mode == "cpu":
                gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
                    x_handle.tensor,
                    gate_lora_A,
                    up_lora_A,
                    offsets,
                    experts,
                    metadata=lora_metadata,
                    stats=layer.stats,
                    tag="gate_up.recompute",
                )
            elif lora_a_forward_mode == "hbm":
                gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1).contiguous()
                try:
                    gate_up_low_rank_owner = grouped_lora_a_forward_hbm(
                        x_stage,
                        gate_up_a,
                        offsets,
                        experts,
                        metadata=lora_metadata,
                        stats=layer.stats,
                        tag="gate_up.recompute",
                    )
                    gate_low_rank, up_low_rank = gate_up_low_rank_owner.split(layer.lora_rank, dim=-1)
                finally:
                    del gate_up_a
            else:
                raise AssertionError(f"unreachable LoRA-A forward mode {lora_a_forward_mode}")
        with prof_range("backward.llama4_mlp.activation_offload.gate_up_recompute_lora_b"):
            gate_delta, up_delta = grouped_expert_lora_pair(
                gate_low_rank,
                up_low_rank,
                gate_lora_B,
                up_lora_B,
                offsets,
                experts,
                metadata=lora_metadata,
            )
            if layer.lora_scale != 1.0:
                gate_delta = gate_delta.mul(layer.lora_scale)
                up_delta = up_delta.mul(layer.lora_scale)
            gate.add_(gate_delta.to(dtype=gate.dtype))
            up.add_(up_delta.to(dtype=up.dtype))
            del gate_delta, up_delta
        with prof_range("backward.llama4_mlp.activation_offload.gate_up_recompute_to_cpu"):
            gate_cpu = manager.offload(gate, "gate")
            up_cpu = manager.offload(up, "up")
        del gate, up, gate_up, gate_low_rank, up_low_rank
        if gate_up_low_rank_owner is not None:
            del gate_up_low_rank_owner
        return gate_cpu, up_cpu
    finally:
        if x_stage is not None:
            manager.release_stage(x_stage, drop_cache=True)
        if release_x_handle:
            manager.release_cpu(x_handle)


class _ActivationOffloadLlama4ExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        x_src_hidden: torch.Tensor | None,
        x_token_indices: torch.Tensor | None,
        x_route_scale: torch.Tensor | None,
        gate_lora_A: torch.Tensor,
        gate_lora_B: torch.Tensor,
        up_lora_A: torch.Tensor,
        up_lora_B: torch.Tensor,
        down_lora_A: torch.Tensor,
        down_lora_B: torch.Tensor,
        layer: "AsymLlama4Experts",
    ) -> torch.Tensor:
        manager = ActivationOffloadManager(pin_memory=True)
        offsets = offsets.detach().contiguous()
        experts = experts.detach().contiguous()
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
        lora_a_forward_mode = _expert_act_offload_lora_a_fwd_mode()
        act_recompute = _llama4_act_recompute()
        # Reserved for a future gate/up recompute lever. Keep disabled until
        # e2e profiling shows gate/up CPU storage is the actual bottleneck.
        gate_up_recompute = False
        x_unpacked = (
            _llama4_x_unpacked()
            and lora_a_forward_mode == "hbm"
            and x_src_hidden is not None
            and x_token_indices is not None
        )

        x_token_indices_cpu = None
        x_route_scale_cpu = None
        with prof_range(layer._forward_range("activation_offload", "x_to_cpu")):
            if x_unpacked:
                x_cpu = manager.offload(x_src_hidden.detach(), "X")
                x_token_indices_cpu = x_token_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
                if x_route_scale is not None:
                    x_route_scale_cpu = x_route_scale.detach().to(device="cpu", dtype=x_cpu.tensor.dtype).contiguous()
            else:
                x_cpu = manager.offload(packed, "X")

        with prof_range(layer._forward_range("activation_offload", "gate_up_base")):
            gate_up = layer.gate_up_base(
                packed,
                offsets,
                experts,
                dense_experts=True,
                compiled_dims="nk",
                profile_name=layer._profile_name("gate_up", "base"),
            )
            gate, up = gate_up.chunk(2, dim=-1)

        with prof_range(layer._forward_range("activation_offload", "gate_up_lora_a")):
            gate_up_low_rank_owner = None
            if lora_a_forward_mode == "cpu":
                gate_low_rank, up_low_rank = grouped_lora_a_pair_forward_cpu_left(
                    x_cpu.tensor,
                    gate_lora_A,
                    up_lora_A,
                    offsets,
                    experts,
                    metadata=lora_metadata,
                    stats=layer.stats,
                    tag="gate_up",
                )
            elif lora_a_forward_mode == "hbm":
                gate_up_a = torch.cat((gate_lora_A, up_lora_A), dim=1).contiguous()
                try:
                    gate_up_low_rank_owner = grouped_lora_a_forward_hbm(
                        packed,
                        gate_up_a,
                        offsets,
                        experts,
                        metadata=lora_metadata,
                        stats=layer.stats,
                        tag="gate_up",
                    )
                    gate_low_rank, up_low_rank = gate_up_low_rank_owner.split(layer.lora_rank, dim=-1)
                finally:
                    del gate_up_a
            else:
                raise AssertionError(f"unreachable LoRA-A forward mode {lora_a_forward_mode}")

        with prof_range(layer._forward_range("activation_offload", "gate_up_lora_b")):
            gate_delta, up_delta = grouped_expert_lora_pair(
                gate_low_rank,
                up_low_rank,
                gate_lora_B,
                up_lora_B,
                offsets,
                experts,
                metadata=lora_metadata,
            )
            if layer.lora_scale != 1.0:
                gate_delta = gate_delta.mul(layer.lora_scale)
                up_delta = up_delta.mul(layer.lora_scale)
            gate.add_(gate_delta.to(dtype=gate.dtype))
            up.add_(up_delta.to(dtype=up.dtype))
            del gate_delta, up_delta

        with prof_range(layer._forward_range("activation_offload", "save_gate_up_cpu")):
            gate_cpu = manager.offload(gate, "gate")
            up_cpu = manager.offload(up, "up")
            gate_low_rank_cpu = manager.offload(gate_low_rank, "S_gate")
            up_low_rank_cpu = manager.offload(up_low_rank, "S_up")
            del gate, up, gate_up, gate_low_rank, up_low_rank
            if gate_up_low_rank_owner is not None:
                del gate_up_low_rank_owner

        with prof_range(layer._forward_range("activation_offload", "activation_cpu")):
            act_cpu = _activation_offload_cpu_silu_mul(gate_cpu, up_cpu, manager, tag="act")
            if gate_up_recompute:
                manager.release_cpu(gate_cpu)
                manager.release_cpu(up_cpu)
                gate_cpu = None
                up_cpu = None

        if lora_a_forward_mode == "cpu":
            with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
                down_low_rank = grouped_lora_a_forward_cpu_left(
                    act_cpu.tensor,
                    down_lora_A,
                    offsets,
                    experts,
                    metadata=lora_metadata,
                    stats=layer.stats,
                    tag="down",
                )
            with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
                down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
                if layer.lora_scale != 1.0:
                    down_delta = down_delta.mul(layer.lora_scale)
                down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
                del down_low_rank
            with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
                act_stage = manager.stage(act_cpu, tag="act_for_down_base")
                output = layer.down_base(
                    act_stage,
                    offsets,
                    experts,
                    dense_experts=True,
                    profile_name=layer._profile_name("down", "base"),
                )
                manager.release_stage(act_stage, drop_cache=True)
                output.add_(down_delta.to(dtype=output.dtype))
                del down_delta, act_stage
        else:
            act_stage = None
            try:
                with prof_range(layer._forward_range("activation_offload", "down_base_stage")):
                    act_stage = manager.stage(act_cpu, tag="act_for_down_base")
                with prof_range(layer._forward_range("activation_offload", "down_lora_a")):
                    down_low_rank = grouped_lora_a_forward_hbm(
                        act_stage,
                        down_lora_A,
                        offsets,
                        experts,
                        metadata=lora_metadata,
                        stats=layer.stats,
                        tag="down",
                    )
                with prof_range(layer._forward_range("activation_offload", "down_lora_b")):
                    down_delta = grouped_expert_lora(down_low_rank, down_lora_B, offsets, experts, metadata=lora_metadata)
                    if layer.lora_scale != 1.0:
                        down_delta = down_delta.mul(layer.lora_scale)
                    down_low_rank_cpu = manager.offload(down_low_rank, "S_down")
                    del down_low_rank
                with prof_range(layer._forward_range("activation_offload", "down_base")):
                    output = layer.down_base(
                        act_stage,
                        offsets,
                        experts,
                        dense_experts=True,
                        profile_name=layer._profile_name("down", "base"),
                    )
                    output.add_(down_delta.to(dtype=output.dtype))
                    del down_delta
            finally:
                if act_stage is not None:
                    manager.release_stage(act_stage, drop_cache=True)

        ctx.layer = layer
        ctx.manager = manager
        ctx.x_cpu = x_cpu
        ctx.gate_cpu = gate_cpu
        ctx.up_cpu = up_cpu
        ctx.act_recompute = act_recompute
        ctx.gate_up_recompute = gate_up_recompute
        ctx.x_unpacked = x_unpacked
        ctx.x_token_indices_cpu = x_token_indices_cpu
        ctx.x_route_scale_cpu = x_route_scale_cpu
        if act_recompute:
            manager.release_cpu(act_cpu)
            ctx.act_cpu = None
        else:
            ctx.act_cpu = act_cpu
        ctx.gate_low_rank_cpu = gate_low_rank_cpu
        ctx.up_low_rank_cpu = up_low_rank_cpu
        ctx.down_low_rank_cpu = down_low_rank_cpu
        ctx.input_dtype = packed.dtype
        ctx.lora_dropout_p = float(layer.lora_dropout_p)
        ctx.expert_lora_a_forward_mode = lora_a_forward_mode
        ctx.weight_offload = getattr(layer, "_weight_offload", None) is not None
        if ctx.weight_offload:
            ctx.save_for_backward(offsets, experts)
        else:
            ctx.save_for_backward(
                offsets,
                experts,
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            )
        activation_offload_stats = manager.snapshot()
        activation_offload_stats["expert_lora_a_forward_mode"] = lora_a_forward_mode
        activation_offload_stats["llama4_act_recompute"] = bool(act_recompute)
        activation_offload_stats["llama4_x_unpacked"] = bool(x_unpacked)
        activation_offload_stats["llama4_gate_up_recompute"] = bool(gate_up_recompute)
        layer._last_activation_offload_stats = activation_offload_stats
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        layer: AsymLlama4Experts = ctx.layer
        if getattr(ctx, "weight_offload", False):
            offsets, experts = ctx.saved_tensors
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
                gate_lora_A,
                gate_lora_B,
                up_lora_A,
                up_lora_B,
                down_lora_A,
                down_lora_B,
            ) = ctx.saved_tensors
        manager: ActivationOffloadManager = ctx.manager
        need_grad_packed = ctx.needs_input_grad[0]
        lora_metadata = prepare_grouped_lora_metadata(offsets, experts, dense_experts=True)
        gate_cpu = ctx.gate_cpu
        up_cpu = ctx.up_cpu

        def ensure_gate_up_cpu() -> tuple[CPUActivationHandle, CPUActivationHandle]:
            nonlocal gate_cpu, up_cpu
            if gate_cpu is None or up_cpu is None:
                gate_cpu, up_cpu = _recompute_gate_up_cpu(
                    ctx,
                    layer,
                    manager,
                    offsets,
                    experts,
                    gate_lora_A,
                    gate_lora_B,
                    up_lora_A,
                    up_lora_B,
                    lora_metadata,
                )
            return gate_cpu, up_cpu

        with prof_range("backward.llama4_mlp.activation_offload.down_lora"):
            grad_lora = grad_output if grad_output.dtype == down_lora_B.dtype else grad_output.to(dtype=down_lora_B.dtype)
            dS_down = grouped_expert_lora(
                grad_lora,
                down_lora_B.transpose(-1, -2),
                offsets,
                experts,
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_down_lora_x = grouped_expert_lora(
                dS_down,
                down_lora_A.transpose(-1, -2),
                offsets,
                experts,
                metadata=lora_metadata,
            )
            down_low_rank = stage_low_rank_from_cpu(
                ctx.down_low_rank_cpu,
                manager,
                tag="S_down_for_dB",
                stats=layer.stats,
            )
            grad_down_lora_B = _grouped_lora_weight_grads_torch(
                grad_lora,
                down_low_rank,
                offsets,
                experts,
                int(down_lora_B.shape[0]),
                out_dtype=down_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(down_low_rank, drop_cache=True)
            manager.release_cpu(ctx.down_low_rank_cpu)
            if getattr(ctx, "act_recompute", False):
                gate_handle, up_handle = ensure_gate_up_cpu()
                act_handle = _activation_offload_cpu_silu_mul(gate_handle, up_handle, manager, tag="act")
            else:
                act_handle = ctx.act_cpu
            grad_down_lora_A = grouped_lora_a_grad_cpu_right(
                dS_down,
                act_handle.tensor,
                offsets,
                experts,
                num_experts=int(down_lora_A.shape[0]),
                stats=layer.stats,
                tag="down",
            )
            manager.release_cpu(act_handle)

        with prof_range("backward.llama4_mlp.activation_offload.down_base_dx"):
            grad_act = _llama4_grouped_base_dx(
                layer.down_base,
                grad_output,
                offsets,
                experts,
                dense_experts=True,
                input_dtype=ctx.input_dtype,
                profile_name=layer._profile_name("down", "base"),
            )
            grad_act.add_(grad_down_lora_x.to(dtype=grad_act.dtype))
            del grad_down_lora_x
            grad_act_cpu = manager.offload(grad_act, "dact")
            del grad_act

        with prof_range("backward.llama4_mlp.activation_offload.activation_cpu"):
            gate_handle, up_handle = ensure_gate_up_cpu()
            grad_gate_cpu, grad_up_cpu = _activation_offload_cpu_silu_backward(
                gate_handle,
                up_handle,
                grad_act_cpu,
                manager,
            )
            manager.release_cpu(grad_act_cpu)
            manager.release_cpu(gate_handle)
            manager.release_cpu(up_handle)
            gate_cpu = None
            up_cpu = None

        grad_packed = None
        grad_gate_lora_x = None
        grad_up_lora_x = None

        with prof_range("backward.llama4_mlp.activation_offload.gate_up_stage"):
            grad_gate_up = manager.stage_concat_columns(grad_gate_cpu, grad_up_cpu, tag="dgate_up_for_gate_up_base")
            grad_gate_stage, grad_up_stage = grad_gate_up.split(int(grad_gate_cpu.tensor.shape[1]), dim=-1)
            manager.release_cpu(grad_gate_cpu)
            manager.release_cpu(grad_up_cpu)

        with prof_range("backward.llama4_mlp.activation_offload.gate_lora"):
            gate_low_rank = stage_low_rank_from_cpu(
                ctx.gate_low_rank_cpu,
                manager,
                tag="S_gate_for_dB",
                stats=layer.stats,
            )
            dS_gate = _grouped_lora_cuda_view(
                grad_gate_stage,
                gate_lora_B.transpose(-1, -2),
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_gate_lora_B = _grouped_lora_weight_grads_torch(
                grad_gate_stage,
                gate_low_rank,
                offsets,
                experts,
                int(gate_lora_B.shape[0]),
                out_dtype=gate_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(gate_low_rank, drop_cache=True)
            manager.release_cpu(ctx.gate_low_rank_cpu)
            if need_grad_packed:
                grad_gate_lora_x = grouped_expert_lora(
                    dS_gate,
                    gate_lora_A.transpose(-1, -2),
                    offsets,
                    experts,
                    metadata=lora_metadata,
                )

        with prof_range("backward.llama4_mlp.activation_offload.up_lora"):
            up_low_rank = stage_low_rank_from_cpu(
                ctx.up_low_rank_cpu,
                manager,
                tag="S_up_for_dB",
                stats=layer.stats,
            )
            dS_up = _grouped_lora_cuda_view(
                grad_up_stage,
                up_lora_B.transpose(-1, -2),
                metadata=lora_metadata,
            ).mul_(layer.lora_scale)
            grad_up_lora_B = _grouped_lora_weight_grads_torch(
                grad_up_stage,
                up_low_rank,
                offsets,
                experts,
                int(up_lora_B.shape[0]),
                out_dtype=up_lora_B.dtype,
                metadata=lora_metadata,
                stats=layer.stats,
                record_lora_b_backward=True,
            ).mul_(layer.lora_scale)
            manager.release_stage(up_low_rank, drop_cache=True)
            manager.release_cpu(ctx.up_low_rank_cpu)
            if need_grad_packed:
                grad_up_lora_x = grouped_expert_lora(
                    dS_up,
                    up_lora_A.transpose(-1, -2),
                    offsets,
                    experts,
                    metadata=lora_metadata,
                )

        with prof_range("backward.llama4_mlp.activation_offload.gate_up_lora_a_grad"):
            x_handle, release_x_handle = _rebuild_packed_x_cpu(ctx, manager)
            grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(
                dS_gate,
                dS_up,
                x_handle.tensor,
                offsets,
                experts,
                num_experts=int(gate_lora_A.shape[0]),
                stats=layer.stats,
            )
            if release_x_handle:
                manager.release_cpu(x_handle)
            manager.release_cpu(ctx.x_cpu)

        if need_grad_packed:
            with prof_range("backward.llama4_mlp.activation_offload.gate_up_base_dx"):
                grad_packed = _llama4_grouped_base_dx(
                    layer.gate_up_base,
                    grad_gate_up,
                    offsets,
                    experts,
                    dense_experts=True,
                    input_dtype=ctx.input_dtype,
                    profile_name=layer._profile_name("gate_up", "base"),
                )
            if grad_gate_lora_x is not None:
                grad_packed.add_(grad_gate_lora_x.to(dtype=grad_packed.dtype))
            if grad_up_lora_x is not None:
                grad_packed.add_(grad_up_lora_x.to(dtype=grad_packed.dtype))

        manager.release_stage(grad_gate_up, drop_cache=True)
        activation_offload_stats_pre_release = manager.snapshot()
        activation_offload_stats_pre_release["expert_lora_a_forward_mode"] = getattr(
            ctx,
            "expert_lora_a_forward_mode",
            "cpu",
        )
        activation_offload_stats_pre_release["llama4_gate_up_recompute"] = bool(
            getattr(ctx, "gate_up_recompute", False)
        )
        activation_offload_stats_pre_release["llama4_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
        activation_offload_stats_pre_release["llama4_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
        final_cleanup_released_bytes = 0
        for handle in (
            ctx.x_cpu,
            gate_cpu,
            up_cpu,
            ctx.act_cpu,
            ctx.gate_low_rank_cpu,
            ctx.up_low_rank_cpu,
            ctx.down_low_rank_cpu,
            grad_act_cpu,
            grad_gate_cpu,
            grad_up_cpu,
        ):
            final_cleanup_released_bytes += manager.release_cpu(handle)
        activation_offload_stats = manager.snapshot()
        activation_offload_stats["expert_lora_a_forward_mode"] = getattr(ctx, "expert_lora_a_forward_mode", "cpu")
        activation_offload_stats["llama4_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
        activation_offload_stats["llama4_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
        activation_offload_stats["llama4_gate_up_recompute"] = bool(getattr(ctx, "gate_up_recompute", False))
        activation_offload_stats["pre_final_cleanup_cpu_owned_bytes"] = activation_offload_stats_pre_release.get(
            "cpu_owned_bytes",
            0,
        )
        activation_offload_stats["final_cleanup_released_bytes"] = final_cleanup_released_bytes
        layer._last_activation_offload_stats_pre_release = activation_offload_stats_pre_release
        layer._last_activation_offload_stats = activation_offload_stats

        return (
            grad_packed,
            None,
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
        )


class AsymLlama4Experts(AsymQwen3Experts):
    """Llama4-owned packed expert wrapper.

    This class owns Llama4's native grouped weight layouts. It reuses the stable
    Qwen3 expert body methods, but constructor shape checks and activation
    offload backward shape logic live here so Qwen3 remains Qwen3-specific.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        backend: Literal["asym", "torch"],
        precision: Literal["bf16"],
        offload: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        expert_recompute_policy: str = "none",
        init_lora_weights: Literal["peft"] = "peft",
        stats: AsymExecutionStats | None = None,
        strict: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if backend not in {"asym", "torch"}:
            raise ValueError("backend must be 'asym' or 'torch'")
        if precision != "bf16":
            raise ValueError("AsymLlama4Experts supports bf16 only")
        if lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {lora_rank}")
        if init_lora_weights != "peft":
            raise ValueError("AsymLlama4Experts only supports PEFT-compatible LoRA init")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")

        gate_up = getattr(source, "gate_up_proj").detach()
        down = getattr(source, "down_proj").detach()
        self.num_experts = int(getattr(source, "num_experts"))
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.intermediate_dim = int(getattr(source, "intermediate_dim"))
        self.gate_up_weight_layout = str(getattr(source, "gate_up_weight_layout", "in_out"))
        self.down_weight_layout = str(getattr(source, "down_weight_layout", "in_out"))
        if self.gate_up_weight_layout == "out_in":
            expected_gate_up = (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        elif self.gate_up_weight_layout == "in_out":
            expected_gate_up = (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
        else:
            raise ValueError(f"unsupported Llama4 gate_up_weight_layout={self.gate_up_weight_layout!r}")
        if self.down_weight_layout == "out_in":
            expected_down = (self.num_experts, self.hidden_dim, self.intermediate_dim)
        elif self.down_weight_layout == "in_out":
            expected_down = (self.num_experts, self.intermediate_dim, self.hidden_dim)
        else:
            raise ValueError(f"unsupported Llama4 down_weight_layout={self.down_weight_layout!r}")
        if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
            raise ValueError(
                "unexpected Llama4 expert shapes: "
                f"gate_up={tuple(gate_up.shape)} expected={expected_gate_up}, "
                f"down={tuple(down.shape)} expected={expected_down}"
            )

        self.config = getattr(source, "config", None)
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = self.gate_up_weight_layout == "in_out"
        self.is_concatenated = True
        self.act_fn = getattr(source, "act_fn")
        self.backend = backend
        self.precision = precision
        self.offload = bool(offload)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = float(lora_alpha) / float(lora_rank)
        self.lora_dtype = normalize_lora_dtype(lora_dtype)
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0.0 else nn.Identity()
        self.stats = stats if stats is not None else AsymExecutionStats()
        self.expert_recompute_config = parse_expert_recompute_policy_spec(expert_recompute_policy)
        self.profile_prefix = "layers.unknown.feed_forward.experts"
        self._last_activation_offload_stats: dict[str, object] = {}
        self._weight_offload = None

        base_dtype = torch.bfloat16
        if backend == "asym" and self.offload:
            if strict and (gate_up.device.type != "cpu" or down.device.type != "cpu"):
                raise RuntimeError("Llama4 expert CPU offload requires CPU-first model loading")
            if strict and (gate_up.dtype != base_dtype or down.dtype != base_dtype):
                raise RuntimeError(
                    "Llama4 expert CPU offload requires bf16 source weights: "
                    f"gate_up={gate_up.dtype}, down={down.dtype}"
                )
            self.gate_up_base = AsymGroupedFrozenLinear(
                gate_up.to(dtype=base_dtype),
                backend="asym",
                pin_memory=torch.cuda.is_available(),
                clone=False,
                precision=precision,
                stats=self.stats,
                weight_layout=self.gate_up_weight_layout,
            )
            self.down_base = AsymGroupedFrozenLinear(
                down.to(dtype=base_dtype),
                backend="asym",
                pin_memory=torch.cuda.is_available(),
                clone=False,
                precision=precision,
                stats=self.stats,
                weight_layout=self.down_weight_layout,
            )
            if strict and torch.cuda.is_available():
                if not self.gate_up_base.host_weight.weight.is_pinned() or not self.down_base.host_weight.weight.is_pinned():
                    raise RuntimeError("Llama4 expert CPU offload requires pinned CPU HostWeights for AsymGEMM")
        else:
            device = _resolve_device(gate_up)
            self.gate_up_base = TorchGroupedFrozenLinear(
                gate_up,
                device=device,
                dtype=base_dtype,
                weight_layout=self.gate_up_weight_layout,
            )
            self.down_base = TorchGroupedFrozenLinear(
                down,
                device=device,
                dtype=base_dtype,
                weight_layout=self.down_weight_layout,
            )

        device = _resolve_device(gate_up)
        self.gate_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.gate_lora_B = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype)
        )
        self.up_lora_A = nn.Parameter(torch.empty(self.num_experts, self.lora_rank, self.hidden_dim, device=device, dtype=self.lora_dtype))
        self.up_lora_B = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.down_lora_A = nn.Parameter(
            torch.empty(self.num_experts, self.lora_rank, self.intermediate_dim, device=device, dtype=self.lora_dtype)
        )
        self.down_lora_B = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.lora_rank, device=device, dtype=self.lora_dtype))
        self.reset_lora_parameters()

    def _forward_expert_activation_offload(
        self,
        packed: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        *,
        x_src_hidden: torch.Tensor | None = None,
        x_token_indices: torch.Tensor | None = None,
        x_route_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_activation_offload_supported(packed)
        return _ActivationOffloadLlama4ExpertFunction.apply(
            packed,
            offsets,
            experts,
            x_src_hidden,
            x_token_indices,
            x_route_scale,
            self.gate_lora_A,
            self.gate_lora_B,
            self.up_lora_A,
            self.up_lora_B,
            self.down_lora_A,
            self.down_lora_B,
            self,
        )

    def forward_input_scaled(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        input_weights: torch.Tensor,
    ) -> torch.Tensor:
        if top_k_index.dim() != 2:
            raise ValueError(f"top_k_index must be [tokens, top_k], got {tuple(top_k_index.shape)}")
        if input_weights.shape != top_k_index.shape:
            raise ValueError(
                "input_weights must match top_k_index shape, "
                f"got {tuple(input_weights.shape)} and {tuple(top_k_index.shape)}"
            )

        self.gather_lora_weights()
        try:
            input_dtype = hidden_states.dtype
            with prof_range(self._forward_range("route_metadata")):
                metadata = build_contiguous_route_metadata(top_k_index, input_weights, num_experts=self.num_experts)
            with prof_range(self._forward_range("pack_tokens")):
                packed = pack_tokens_contiguous(hidden_states, metadata)
                route_scale = metadata.routing_weights.reshape(metadata.num_routes, *([1] * (packed.dim() - 1)))
                packed = packed * route_scale.to(device=packed.device, dtype=packed.dtype)
            with prof_range(self._forward_range("route_metadata", "dense_groups")):
                offsets, experts = make_dense_group_metadata(
                    metadata.expert_offsets,
                    num_groups=self.num_experts,
                    device=packed.device,
            )
            if self._uses_activation_offload():
                x_src_hidden = None
                x_token_indices = None
                x_route_scale = None
                if _llama4_x_unpacked():
                    x_src_hidden = hidden_states.reshape(metadata.num_tokens, -1)
                    x_token_indices = metadata.token_indices
                    x_route_scale = metadata.routing_weights
                down = self._forward_expert_activation_offload(
                    packed,
                    offsets,
                    experts,
                    x_src_hidden=x_src_hidden,
                    x_token_indices=x_token_indices,
                    x_route_scale=x_route_scale,
                )
            elif self._uses_expert_gc():
                down = self._forward_expert_gc(packed, offsets, experts)
            elif self._uses_expert_recompute():
                down = self._forward_expert_policy(packed, offsets, experts, metadata)
            else:
                down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
            with prof_range(self._forward_range("scatter_combine")):
                return _scatter_contiguous_sum(down, metadata).to(dtype=input_dtype)
        finally:
            self.release_lora_weights()


__all__ = ["AsymLlama4Experts"]
