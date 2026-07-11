from __future__ import annotations

import pytest
import torch

from asym_gemm.training.exp_act_offload_lora import (
    grouped_lora_a_grad_cpu_right,
    grouped_lora_a_pair_grad_cpu_right,
    grouped_lora_b_backward_cpu_source,
    require_expert_activation_offload_kernels,
)


def _sm100_expact_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and require_expert_activation_offload_kernels(scope="full", check_only=True) is None
    )


def _route_metadata(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor([0, *torch.cumsum(torch.tensor(lengths, dtype=torch.int64), dim=0).tolist()], device="cuda", dtype=torch.int32)
    experts = torch.tensor([*range(len(lengths)), -1], device="cuda", dtype=torch.int32)
    return offsets, experts


@pytest.mark.skipif(not _sm100_expact_available(), reason="SM100 expert activation-offload kernels required")
def test_grouped_lora_a_grad_cpu_right_matches_reference() -> None:
    torch.manual_seed(123)
    lengths = [4, 5, 3]
    num_experts = len(lengths)
    rows = sum(lengths)
    hidden = 16
    rank = 8
    offsets, experts = _route_metadata(lengths)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    x_cpu = x.cpu().pin_memory()
    dS = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).contiguous()

    grad = grouped_lora_a_grad_cpu_right(dS, x_cpu, offsets, experts, num_experts=num_experts, stats=None, tag="test")

    ref = torch.zeros(num_experts, rank, hidden, device="cuda", dtype=torch.float32)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        ref[expert] = dS[start:end].float().T @ x[start:end].float()
        start = end
    assert torch.allclose(grad.float(), ref, atol=0.08, rtol=0.08)


@pytest.mark.skipif(not _sm100_expact_available(), reason="SM100 expert activation-offload kernels required")
def test_grouped_lora_a_pair_grad_cpu_right_matches_reference() -> None:
    torch.manual_seed(124)
    lengths = [3, 6, 2]
    num_experts = len(lengths)
    rows = sum(lengths)
    hidden = 16
    rank = 8
    offsets, experts = _route_metadata(lengths)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    x_cpu = x.cpu().pin_memory()
    dS_gate = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).contiguous()
    dS_up = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).contiguous()

    grad_gate, grad_up = grouped_lora_a_pair_grad_cpu_right(
        dS_gate,
        dS_up,
        x_cpu,
        offsets,
        experts,
        num_experts=num_experts,
        stats=None,
    )

    ref_gate = torch.zeros(num_experts, rank, hidden, device="cuda", dtype=torch.float32)
    ref_up = torch.zeros_like(ref_gate)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        ref_gate[expert] = dS_gate[start:end].float().T @ x[start:end].float()
        ref_up[expert] = dS_up[start:end].float().T @ x[start:end].float()
        start = end
    assert torch.allclose(grad_gate.float(), ref_gate, atol=0.08, rtol=0.08)
    assert torch.allclose(grad_up.float(), ref_up, atol=0.08, rtol=0.08)


@pytest.mark.skipif(not _sm100_expact_available(), reason="SM100 expert activation-offload kernels required")
def test_grouped_lora_b_backward_cpu_source_matches_reference() -> None:
    torch.manual_seed(125)
    lengths = [4, 5, 3]
    num_experts = len(lengths)
    rows = sum(lengths)
    intermediate = 24
    rank = 8
    scale = 0.25
    offsets, experts = _route_metadata(lengths)
    grad_out = torch.randn(rows, intermediate, device="cuda", dtype=torch.bfloat16)
    grad_out_cpu = grad_out.cpu().pin_memory()
    low_rank = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).contiguous()
    lora_b = torch.randn(num_experts, intermediate, rank, device="cuda", dtype=torch.bfloat16).contiguous()

    dS, grad_b = grouped_lora_b_backward_cpu_source(
        grad_out_cpu,
        low_rank,
        lora_b,
        offsets,
        experts,
        scale=scale,
        stats=None,
        tag="test",
    )

    ref_dS = torch.zeros(rows, rank, device="cuda", dtype=torch.float32)
    ref_grad_b = torch.zeros(num_experts, intermediate, rank, device="cuda", dtype=torch.float32)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        ref_dS[start:end] = grad_out[start:end].float() @ lora_b[expert].float() * scale
        ref_grad_b[expert] = grad_out[start:end].float().T @ low_rank[start:end].float() * scale
        start = end
    assert torch.allclose(dS.float(), ref_dS, atol=0.08, rtol=0.08)
    assert torch.allclose(grad_b.float(), ref_grad_b, atol=0.08, rtol=0.08)
