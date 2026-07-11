from __future__ import annotations

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from asym_gemm.training.dense_mlp_finegrained import build_finegrained_dense_mlp
from asym_gemm.training.frozen_linear import AsymExecutionStats
from asym_gemm.training.weight_offload import LoRAWeightOffloadCoordinator, install_lora_weight_offload


class _ToyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _lora_linear(x: torch.Tensor, base_weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    base = F.linear(x, base_weight.to(device=x.device, dtype=x.dtype))
    low_rank = x.to(dtype=a.dtype).matmul(a.t())
    delta = low_rank.matmul(b.t()).mul(scale)
    return base + delta.to(dtype=base.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finegrained_dense_mlp_matches_reference_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD", "1")
    torch.manual_seed(123)
    hidden = 32
    intermediate = 64
    rank = 8
    mlp = _ToyMLP(hidden, intermediate)
    stats = AsymExecutionStats()
    wrapped = build_finegrained_dense_mlp(
        mlp,
        backend="torch",
        precision="bf16",
        lora_rank=rank,
        lora_alpha=16.0,
        lora_dropout=0.0,
        stats=stats,
        strict=True,
        profile_prefix="test.layers.0.mlp",
    ).cuda()
    wrapped.train()

    with torch.no_grad():
        for param in (
            wrapped.gate_proj.lora_a,
            wrapped.gate_proj.lora_b,
            wrapped.up_proj.lora_a,
            wrapped.up_proj.lora_b,
            wrapped.down_proj.lora_a,
            wrapped.down_proj.lora_b,
        ):
            param.normal_(mean=0.0, std=0.02)

    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    gate_a_ref = wrapped.gate_proj.lora_a.detach().clone().requires_grad_(True)
    gate_b_ref = wrapped.gate_proj.lora_b.detach().clone().requires_grad_(True)
    up_a_ref = wrapped.up_proj.lora_a.detach().clone().requires_grad_(True)
    up_b_ref = wrapped.up_proj.lora_b.detach().clone().requires_grad_(True)
    down_a_ref = wrapped.down_proj.lora_a.detach().clone().requires_grad_(True)
    down_b_ref = wrapped.down_proj.lora_b.detach().clone().requires_grad_(True)

    out = wrapped(x)
    scale = wrapped.lora_scale
    gate_ref = _lora_linear(
        x_ref,
        wrapped.gate_proj.base_layer.host_weight.weight,
        gate_a_ref,
        gate_b_ref,
        scale,
    )
    up_ref = _lora_linear(
        x_ref,
        wrapped.up_proj.base_layer.host_weight.weight,
        up_a_ref,
        up_b_ref,
        scale,
    )
    act_ref = F.silu(gate_ref) * up_ref
    out_ref = _lora_linear(
        act_ref,
        wrapped.down_proj.base_layer.host_weight.weight,
        down_a_ref,
        down_b_ref,
        scale,
    )

    assert torch.allclose(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    assert torch.allclose(x.grad, x_ref.grad, atol=4e-2, rtol=4e-2)
    for got, ref in (
        (wrapped.gate_proj.lora_a.grad, gate_a_ref.grad),
        (wrapped.gate_proj.lora_b.grad, gate_b_ref.grad),
        (wrapped.up_proj.lora_a.grad, up_a_ref.grad),
        (wrapped.up_proj.lora_b.grad, up_b_ref.grad),
        (wrapped.down_proj.lora_a.grad, down_a_ref.grad),
        (wrapped.down_proj.lora_b.grad, down_b_ref.grad),
    ):
        assert got is not None
        assert ref is not None
        assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2)

    assert stats.dense_mlp_finegrained_forward_calls == 1
    assert stats.dense_mlp_finegrained_backward_calls == 1
    assert stats.dense_mlp_finegrained_stage_concat_columns_calls == 0
    assert stats.dense_mlp_finegrained_gpu_silu_bwd_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finegrained_dense_mlp_cpu_activation_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD", "1")
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT", "1")
    torch.manual_seed(456)
    hidden = 32
    intermediate = 64
    rank = 8
    mlp = _ToyMLP(hidden, intermediate)
    stats = AsymExecutionStats()
    wrapped = build_finegrained_dense_mlp(
        mlp,
        backend="torch",
        precision="bf16",
        lora_rank=rank,
        lora_alpha=16.0,
        lora_dropout=0.0,
        stats=stats,
        strict=True,
        profile_prefix="test.layers.0.mlp",
    ).cuda()
    wrapped.train()

    with torch.no_grad():
        for param in (
            wrapped.gate_proj.lora_a,
            wrapped.gate_proj.lora_b,
            wrapped.up_proj.lora_a,
            wrapped.up_proj.lora_b,
            wrapped.down_proj.lora_a,
            wrapped.down_proj.lora_b,
        ):
            param.normal_(mean=0.0, std=0.02)

    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    gate_a_ref = wrapped.gate_proj.lora_a.detach().clone().requires_grad_(True)
    gate_b_ref = wrapped.gate_proj.lora_b.detach().clone().requires_grad_(True)
    up_a_ref = wrapped.up_proj.lora_a.detach().clone().requires_grad_(True)
    up_b_ref = wrapped.up_proj.lora_b.detach().clone().requires_grad_(True)
    down_a_ref = wrapped.down_proj.lora_a.detach().clone().requires_grad_(True)
    down_b_ref = wrapped.down_proj.lora_b.detach().clone().requires_grad_(True)

    out = wrapped(x)
    scale = wrapped.lora_scale
    gate_ref = _lora_linear(x_ref, wrapped.gate_proj.base_layer.host_weight.weight, gate_a_ref, gate_b_ref, scale)
    up_ref = _lora_linear(x_ref, wrapped.up_proj.base_layer.host_weight.weight, up_a_ref, up_b_ref, scale)
    act_ref = F.silu(gate_ref) * up_ref
    out_ref = _lora_linear(act_ref, wrapped.down_proj.base_layer.host_weight.weight, down_a_ref, down_b_ref, scale)

    assert torch.allclose(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    assert torch.allclose(x.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
    for got, ref in (
        (wrapped.gate_proj.lora_a.grad, gate_a_ref.grad),
        (wrapped.gate_proj.lora_b.grad, gate_b_ref.grad),
        (wrapped.up_proj.lora_a.grad, up_a_ref.grad),
        (wrapped.up_proj.lora_b.grad, up_b_ref.grad),
        (wrapped.down_proj.lora_a.grad, down_a_ref.grad),
        (wrapped.down_proj.lora_b.grad, down_b_ref.grad),
    ):
        assert got is not None
        assert ref is not None
        assert torch.allclose(got, ref, atol=6e-2, rtol=6e-2)

    assert stats.dense_mlp_finegrained_forward_calls == 1
    assert stats.dense_mlp_finegrained_backward_calls == 1
    assert stats.dense_mlp_finegrained_cpu_silu_bwd_calls == 1
    assert stats.dense_mlp_finegrained_gpu_silu_bwd_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finegrained_dense_mlp_cpu_activation_no_grad_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD", "1")
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT", "1")
    torch.manual_seed(789)
    hidden = 32
    intermediate = 64
    rank = 8
    mlp = _ToyMLP(hidden, intermediate)
    stats = AsymExecutionStats()
    wrapped = build_finegrained_dense_mlp(
        mlp,
        backend="torch",
        precision="bf16",
        lora_rank=rank,
        lora_alpha=16.0,
        lora_dropout=0.0,
        stats=stats,
        strict=True,
        profile_prefix="test.layers.0.mlp",
    ).cuda()
    wrapped.train()

    with torch.no_grad():
        for param in (
            wrapped.gate_proj.lora_a,
            wrapped.gate_proj.lora_b,
            wrapped.up_proj.lora_a,
            wrapped.up_proj.lora_b,
            wrapped.down_proj.lora_a,
            wrapped.down_proj.lora_b,
        ):
            param.normal_(mean=0.0, std=0.02)
        x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)
        out = wrapped(x)
        scale = wrapped.lora_scale
        gate_ref = _lora_linear(x, wrapped.gate_proj.base_layer.host_weight.weight, wrapped.gate_proj.lora_a, wrapped.gate_proj.lora_b, scale)
        up_ref = _lora_linear(x, wrapped.up_proj.base_layer.host_weight.weight, wrapped.up_proj.lora_a, wrapped.up_proj.lora_b, scale)
        out_ref = _lora_linear(
            F.silu(gate_ref) * up_ref,
            wrapped.down_proj.base_layer.host_weight.weight,
            wrapped.down_proj.lora_a,
            wrapped.down_proj.lora_b,
            scale,
        )

    assert torch.allclose(out, out_ref, atol=3e-2, rtol=3e-2)
    assert not out.requires_grad
    assert stats.dense_mlp_finegrained_forward_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finegrained_dense_mlp_weight_offload_registers_parent_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD", "1")
    torch.manual_seed(321)
    hidden = 32
    intermediate = 64
    rank = 8
    mlp = _ToyMLP(hidden, intermediate)
    stats = AsymExecutionStats()
    wrapped = build_finegrained_dense_mlp(
        mlp,
        backend="torch",
        precision="bf16",
        lora_rank=rank,
        lora_alpha=16.0,
        lora_dropout=0.0,
        stats=stats,
        strict=True,
        profile_prefix="test.layers.0.mlp",
    ).cuda()
    wrapped.train()

    coordinator = LoRAWeightOffloadCoordinator(pin_memory=False, persistence_threshold_numel=0)
    installed = install_lora_weight_offload(wrapped, coordinator)
    summary = coordinator.summary()

    assert installed == 6
    assert summary["weight_offload_group_count"] == 1
    assert summary["weight_offload_param_count"] == 6
    assert summary["weight_offload_group_count_by_component"]["mlp_dense"] == 1
    assert getattr(wrapped, "_weight_offload", None) is coordinator
    assert getattr(wrapped.gate_proj, "_weight_offload_owner", None) is wrapped
    assert getattr(wrapped.up_proj, "_weight_offload_owner", None) is wrapped
    assert getattr(wrapped.down_proj, "_weight_offload_owner", None) is wrapped
    assert wrapped.gate_proj.lora_a.numel() == 0
    assert wrapped.up_proj.lora_a.numel() == 0
    assert wrapped.down_proj.lora_b.numel() == 0

    x = torch.randn(2, 4, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = wrapped(x).float().square().mean()

    assert wrapped.gate_proj.lora_a.numel() == 0
    assert wrapped.up_proj.lora_a.numel() == 0
    assert wrapped.down_proj.lora_b.numel() == 0

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
    for param, shape in (
        (wrapped.gate_proj.lora_a, (rank, hidden)),
        (wrapped.gate_proj.lora_b, (intermediate, rank)),
        (wrapped.up_proj.lora_a, (rank, hidden)),
        (wrapped.up_proj.lora_b, (intermediate, rank)),
        (wrapped.down_proj.lora_a, (rank, intermediate)),
        (wrapped.down_proj.lora_b, (hidden, rank)),
    ):
        assert param.grad is not None
        assert tuple(param.grad.shape) == shape
