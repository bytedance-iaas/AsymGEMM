from __future__ import annotations

import pytest
import torch

from asym_gemm.training.attention_activation_offload import (
    _dense_lora_a_cpu_left,
    _single_group_offsets_experts,
)
from asym_gemm.training.frozen_linear import AsymExecutionStats, asym_bf16_cpu_right_matmul


def test_single_group_offsets_experts_are_cached_and_int32() -> None:
    offsets, experts = _single_group_offsets_experts("cpu", 17)
    offsets_again, experts_again = _single_group_offsets_experts(torch.device("cpu"), 17)

    assert offsets.tolist() == [0, 17]
    assert experts.tolist() == [0, -1]
    assert offsets.dtype == torch.int32
    assert experts.dtype == torch.int32
    assert offsets.data_ptr() == offsets_again.data_ptr()
    assert experts.data_ptr() == experts_again.data_ptr()


def test_dense_lora_a_cpu_left_torch_matches_reference_and_records_stats() -> None:
    torch.manual_seed(11)
    stats = AsymExecutionStats()
    u_cpu = torch.randn(9, 16, dtype=torch.bfloat16).contiguous()
    a = torch.randn(4, 16, dtype=torch.bfloat16).contiguous()

    out = _dense_lora_a_cpu_left(u_cpu, a, stats=stats, tag="q.lora_a", backend="torch")
    ref = u_cpu @ a.t()

    assert torch.allclose(out.float(), ref.float(), atol=0.0, rtol=0.0)
    assert stats.attn_act_lora_a_forward_calls == 1
    assert stats.attn_act_hbm_gemm_calls_by_tag == {"q.lora_a": 1}


def test_asym_bf16_cpu_right_matmul_torch_matches_reference_and_records_base_dx() -> None:
    torch.manual_seed(13)
    stats = AsymExecutionStats()
    left = torch.randn(7, 16, dtype=torch.bfloat16).contiguous()
    right = torch.randn(32, 16, dtype=torch.bfloat16).contiguous()

    out = asym_bf16_cpu_right_matmul(
        left,
        right,
        backend="torch",
        stats=stats,
        phase="attn_act_base_dx",
        tag="q.base_dx",
    )
    ref = left @ right.t()

    assert torch.allclose(out.float(), ref.float(), atol=0.0, rtol=0.0)
    assert stats.torch_dx_calls == 1
    assert stats.attn_act_base_dx_calls == 1
    assert stats.attn_act_hbm_gemm_calls_by_tag == {"q.base_dx": 1}


def test_asym_bf16_cpu_right_matmul_torch_transpose_matches_reference_and_records_dA() -> None:
    torch.manual_seed(17)
    stats = AsymExecutionStats()
    left = torch.randn(4, 8, dtype=torch.bfloat16).contiguous()
    right = torch.randn(8, 12, dtype=torch.bfloat16).contiguous()

    out = asym_bf16_cpu_right_matmul(
        left,
        right,
        transpose_b=True,
        backend="torch",
        stats=stats,
        phase="attn_act_dA",
        tag="q.dA",
    )
    ref = left @ right

    assert torch.allclose(out.float(), ref.float(), atol=0.0, rtol=0.0)
    assert stats.torch_dx_calls == 1
    assert stats.attn_act_lora_a_grad_calls == 1
    assert stats.attn_act_hbm_gemm_calls_by_tag == {"q.dA": 1}


def test_asym_bf16_cpu_right_matmul_rejects_noncontiguous_operands() -> None:
    left = torch.randn(8, 4, dtype=torch.bfloat16).t()
    right = torch.randn(16, 8, dtype=torch.bfloat16).contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        asym_bf16_cpu_right_matmul(left, right, backend="asym", phase="attn_act_base_dx")


def test_asym_bf16_cpu_right_matmul_asym_fails_loudly_when_direct_path_unavailable() -> None:
    left = torch.randn(4, 8, dtype=torch.bfloat16).contiguous()
    right = torch.randn(16, 8, dtype=torch.bfloat16).contiguous()

    with pytest.raises(RuntimeError, match="direct BF16 AsymGEMM is unavailable"):
        asym_bf16_cpu_right_matmul(left, right, backend="asym", phase="attn_act_base_dx")
