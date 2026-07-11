import pytest
import torch

from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymGroupedFrozenLinear, _asym_grouped_bf16_nt
from asym_gemm.training.qwen3_moe_routed_gemm import (
    down_dx_gather_left,
    down_forward_scatter_add_,
    gateup_dx_scatter_add_,
)


def _skip_without_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability(0)[0] != 10:
        pytest.skip("Qwen3 routed Asym kernels are SM100-only")


def _metadata():
    counts = torch.tensor([129, 17, 256, 3], device="cuda", dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.long), counts.cumsum(0)))
    experts = torch.tensor([0, 1, 2, 3, -1], device="cuda", dtype=torch.long)
    return offsets, experts, int(offsets[-1].item())


def _base(weight: torch.Tensor) -> AsymGroupedFrozenLinear:
    return AsymGroupedFrozenLinear(
        weight.detach().cpu().pin_memory(),
        backend="asym",
        pin_memory=True,
        stats=AsymExecutionStats(),
        precision="bf16",
    )


def _scatter_ref(
    route_output: torch.Tensor,
    token_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    num_tokens: int,
) -> torch.Tensor:
    flat = route_output.to(dtype=torch.float32).contiguous()
    flat.mul_(routing_weights.reshape(-1, 1).to(dtype=torch.float32, device=flat.device))
    out = torch.zeros((int(num_tokens), int(flat.shape[1])), device=flat.device, dtype=torch.float32)
    out.index_add_(0, token_indices.to(dtype=torch.long), flat)
    return out


def test_qwen3_moe_routed_base_kernels_match_grouped_asym_reference():
    _skip_without_sm100()
    torch.manual_seed(123)

    offsets, experts, route_rows = _metadata()
    num_tokens = route_rows + 23
    hidden = 128
    intermediate = 128
    num_experts = 4
    token_indices = torch.randperm(num_tokens, device="cuda", dtype=torch.long)[:route_rows].contiguous()
    routing_weights = torch.rand((route_rows,), device="cuda", dtype=torch.bfloat16).contiguous()

    down_weight = torch.randn((num_experts, hidden, intermediate), dtype=torch.bfloat16)
    gate_weight = torch.randn((num_experts, intermediate, hidden), dtype=torch.bfloat16)
    down_base = _base(down_weight)
    gate_base = _base(gate_weight)

    act = torch.randn((route_rows, intermediate), device="cuda", dtype=torch.bfloat16)
    down_route = _asym_grouped_bf16_nt(
        act,
        down_base.host_weight.weight,
        offsets,
        experts,
        compiled_dims="nk",
        transpose_b=False,
        output_dtype=torch.float32,
    )
    expected_down = _scatter_ref(down_route, token_indices, routing_weights, num_tokens=num_tokens)
    got_down = torch.zeros_like(expected_down)
    down_forward_scatter_add_(
        down_base,
        act,
        got_down,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=True,
    )
    torch.testing.assert_close(got_down, expected_down, rtol=2e-2, atol=2e-1)

    grad_token = torch.randn((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)
    grad_routes = grad_token.index_select(0, token_indices)
    grad_routes.mul_(routing_weights.reshape(-1, 1).to(dtype=torch.bfloat16, device=grad_routes.device))
    expected_grad_act = _asym_grouped_bf16_nt(
        grad_routes.contiguous(),
        down_base.host_weight.weight,
        offsets,
        experts,
        compiled_dims="nk",
        transpose_b=True,
        output_dtype=torch.bfloat16,
    )
    got_grad_act = down_dx_gather_left(
        down_base,
        grad_token,
        (route_rows, intermediate),
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=True,
    )
    torch.testing.assert_close(got_grad_act, expected_grad_act, rtol=2e-2, atol=2e-1)

    grad_expert = torch.randn((route_rows, intermediate), device="cuda", dtype=torch.bfloat16)
    gate_route = _asym_grouped_bf16_nt(
        grad_expert,
        gate_base.host_weight.weight,
        offsets,
        experts,
        compiled_dims="nk",
        transpose_b=True,
        output_dtype=torch.float32,
    )
    expected_hidden = _scatter_ref(gate_route, token_indices, routing_weights, num_tokens=num_tokens)
    got_hidden = torch.zeros_like(expected_hidden)
    gateup_dx_scatter_add_(
        gate_base,
        grad_expert,
        got_hidden,
        offsets,
        experts,
        token_indices,
        routing_weights,
        weighted=True,
    )
    torch.testing.assert_close(got_hidden, expected_hidden, rtol=2e-2, atol=2e-1)

    assert down_base.stats.qwen3_moe_routed_base_forward_scatter_calls == 1
    assert down_base.stats.qwen3_moe_routed_base_gather_left_calls == 1
    assert gate_base.stats.qwen3_moe_routed_base_dx_scatter_calls == 1
