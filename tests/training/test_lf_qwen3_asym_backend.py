from __future__ import annotations

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from asym_gemm.integrations.lf import apply_lf_asym_lora
from asym_gemm.training.frozen_linear import TorchGroupedFrozenLinear
from asym_gemm.training.qwen3_moe import AsymQwen3Experts, is_qwen3_experts


class FakeQwen3Experts(nn.Module):
    def __init__(self, *, num_experts: int = 4, hidden_dim: int = 8, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16) * 0.02)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        self.mlp.experts = FakeQwen3Experts()


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(), FakeBlock()])
        self.lm_head = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)


def _routing() -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.tensor(
        [
            [0, 1],
            [2, 2],
            [3, 0],
            [1, 3],
            [0, 0],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.9, 0.1],
        ],
        dtype=torch.bfloat16,
    )
    return indices, weights


def test_is_qwen3_experts_accepts_packed_fake_and_rejects_linear() -> None:
    assert is_qwen3_experts(FakeQwen3Experts())
    assert not is_qwen3_experts(nn.Linear(8, 8))


def test_asym_qwen3_experts_torch_matches_eager_at_zero_delta() -> None:
    torch.manual_seed(0)
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16)
    top_k_index, top_k_weights = _routing()

    expected = source(x, top_k_index, top_k_weights)
    actual = wrapped(x, top_k_index, top_k_weights)

    assert torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(wrapped.gate_lora_B) == 0
    assert torch.count_nonzero(wrapped.up_lora_B) == 0
    assert torch.count_nonzero(wrapped.down_lora_B) == 0


def test_asym_qwen3_experts_torch_backward_trains_only_lora() -> None:
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    for name, param in wrapped.named_parameters():
        param.requires_grad_("lora_" in name)

    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16, requires_grad=True)
    top_k_index, top_k_weights = _routing()
    loss = wrapped(x, top_k_index, top_k_weights).float().square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
    trainable = {name for name, param in wrapped.named_parameters() if param.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert any(param.grad is not None for name, param in wrapped.named_parameters() if name.endswith("lora_B"))


def test_apply_lf_asym_lora_wraps_experts_dense_and_freezes_router() -> None:
    model = FakeModel()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        strict=True,
    )

    assert report.qwen3_experts_wrapped == 2
    assert report.dense_lora_wrapped == 8
    assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)
    assert isinstance(model.layers[0].mlp.experts.gate_up_base, TorchGroupedFrozenLinear)
    assert not model.layers[0].mlp.gate.weight.requires_grad
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in trainable)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_asym_qwen3_experts_sm100_smoke() -> None:
    source = FakeQwen3Experts(hidden_dim=64, intermediate_dim=64).cuda()
    wrapped = AsymQwen3Experts(
        source,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    x = torch.randn(5, source.hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    top_k_index, top_k_weights = _routing()
    loss = wrapped(x, top_k_index.cuda(), top_k_weights.cuda()).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert wrapped.stats.asym_calls > 0
