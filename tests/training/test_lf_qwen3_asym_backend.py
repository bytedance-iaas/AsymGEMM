from __future__ import annotations

import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from asym_gemm.integrations.lf import (
    apply_lf_asym_lora,
    get_asym_lora_state_dict,
    load_asym_peft_adapter,
    save_asym_peft_adapter,
)
from asym_gemm.integrations.peft_lf import adapt_lf_asym_peft_lora
from asym_gemm.training.frozen_linear import TorchGroupedFrozenLinear
from asym_gemm.training.llama4_moe import AsymLlama4Moe, is_llama4_moe
from asym_gemm.training.moe import (
    build_contiguous_route_metadata,
    make_dense_group_metadata,
    pack_tokens_contiguous,
    parse_expert_recompute_policy_spec,
)
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


class FakeGemma4TextExperts(FakeQwen3Experts):
    pass


class FakeBlock(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden_dim, 4, bias=False, dtype=torch.bfloat16)
        self.mlp.experts = FakeQwen3Experts(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)


class FakeModel(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeBlock(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            dense = layer.q_proj(hidden_states) + layer.k_proj(hidden_states) + layer.v_proj(hidden_states) + layer.o_proj(hidden_states)
            routed = layer.mlp.experts(hidden_states, top_k_index, top_k_weights)
            hidden_states = (hidden_states + dense.mul(0.125).to(hidden_states.dtype) + routed).to(dtype=hidden_states.dtype)
        return self.lm_head(hidden_states)


class FakeGemma4Block(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.router = nn.Module()
        self.router.proj = nn.Linear(hidden_dim, 4, bias=False, dtype=torch.bfloat16)
        self.experts = FakeGemma4TextExperts(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)


class FakeGemma4Model(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeGemma4Block(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim) for _ in range(num_layers)]
        )


class FakeLlama4Experts(nn.Module):
    def __init__(self, *, num_experts: int = 4, hidden_size: int = 8, intermediate_size: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_dim = intermediate_size
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=torch.bfloat16) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16) * 0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = torch.bmm(up * self.act_fn(gate), self.down_proj)
        return next_states.view(-1, self.hidden_size)


class FakeLlama4Router(nn.Linear):
    def __init__(self, *, hidden_size: int = 8, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__(hidden_size, num_experts, bias=False, dtype=torch.bfloat16)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = super().forward(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)
        return router_scores, router_logits


class FakeLlama4MLP(nn.Module):
    def __init__(self, *, hidden_size: int = 8, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.activation_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeLlama4Moe(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int = 4,
        hidden_size: int = 8,
        intermediate_size: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.hidden_dim = hidden_size
        self.num_experts = num_experts
        self.experts = FakeLlama4Experts(num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.router = FakeLlama4Router(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        self.shared_expert = FakeLlama4MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        routed_in = hidden_states.repeat(router_scores.shape[1], 1)
        routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)
        out = out + routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0)
        return out, router_logits


class FakeLlama4Block(nn.Module):
    def __init__(self, *, hidden_size: int = 8, intermediate_size: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16)
        self.feed_forward = FakeLlama4Moe(hidden_size=hidden_size, intermediate_size=intermediate_size)


class FakeLlama4Model(nn.Module):
    def __init__(self, *, hidden_size: int = 8, intermediate_size: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeLlama4Block(hidden_size=hidden_size, intermediate_size=intermediate_size) for _ in range(num_layers)]
        )


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


def _copy_random_lora_params(lhs: nn.Module, rhs: nn.Module, *, seed: int = 99) -> None:
    rhs_params = dict(rhs.named_parameters())
    with torch.no_grad():
        for name, param in lhs.named_parameters():
            if "lora_" not in name:
                continue
            generator_device = param.device if param.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(seed + len(name))
            value = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator) * 0.01
            param.copy_(value)
            rhs_params[name].copy_(value)


def _assert_tensor_close_l2(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_tol: float = 5e-3,
    rel_l2_tol: float = 1e-2,
) -> None:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    expected_norm = torch.linalg.vector_norm(expected_f)
    rel_l2 = float(torch.linalg.vector_norm(diff).item()) if float(expected_norm.item()) == 0.0 else float((torch.linalg.vector_norm(diff) / expected_norm).item())
    max_abs = float(diff.max().item())
    assert max_abs <= max_abs_tol, f"{name} max_abs={max_abs:.6g} > {max_abs_tol:.6g}, rel_l2={rel_l2:.6g}"
    assert rel_l2 <= rel_l2_tol, f"{name} rel_l2={rel_l2:.6g} > {rel_l2_tol:.6g}, max_abs={max_abs:.6g}"


def _assert_grad_close(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    assert actual is not None, f"{name} actual grad is None"
    assert expected is not None, f"{name} expected grad is None"
    assert torch.isfinite(actual.float()).all(), f"{name} actual grad has non-finite values"
    assert torch.isfinite(expected.float()).all(), f"{name} expected grad has non-finite values"
    _assert_tensor_close_l2(name, actual, expected)


def test_is_qwen3_experts_accepts_packed_fake_and_rejects_linear() -> None:
    assert is_qwen3_experts(FakeQwen3Experts())
    assert not is_qwen3_experts(nn.Linear(8, 8))


def test_is_llama4_moe_accepts_fake_and_rejects_qwen3_experts() -> None:
    assert is_llama4_moe(FakeLlama4Moe())
    assert not is_llama4_moe(FakeQwen3Experts())


def test_parse_expert_recompute_policy_spec() -> None:
    none = parse_expert_recompute_policy_spec(None)
    lower = parse_expert_recompute_policy_spec("tok-le2")
    zero = parse_expert_recompute_policy_spec("tok-le0")
    upper = parse_expert_recompute_policy_spec("tok-ge2")
    upper_all = parse_expert_recompute_policy_spec("tok-ge1")
    bounded = parse_expert_recompute_policy_spec("tok2-4")
    activation = parse_expert_recompute_policy_spec("tok-le2-act")
    activation_zero = parse_expert_recompute_policy_spec("tok-le0-act")
    activation_upper = parse_expert_recompute_policy_spec("tok-ge2-act")
    activation_upper_all = parse_expert_recompute_policy_spec("tok-ge1-act")
    activation_bounded = parse_expert_recompute_policy_spec("tok2-4-act")

    assert none.label == "none"
    assert lower.label == "tok-le2"
    assert lower.policy == "tok"
    assert lower.token_threshold == 2
    assert lower.token_min == 1
    assert lower.token_max == 2
    assert zero.label == "tok-le0"
    assert zero.policy == "none"
    assert zero.force_custom_autograd
    assert upper.label == "tok-ge2"
    assert upper.token_threshold == 0
    assert upper.token_min == 2
    assert upper.token_max is None
    assert upper_all.recompute_enabled
    assert upper_all.token_min == 1
    assert upper_all.token_max is None
    assert bounded.label == "tok2-4"
    assert bounded.token_threshold == 4
    assert bounded.token_min == 2
    assert bounded.token_max == 4
    assert activation.policy == "none"
    assert activation.activation_save_policy == "tok_act"
    assert activation.activation_save_threshold == 2
    assert activation.activation_save_min == 1
    assert activation.activation_save_max == 2
    assert activation.label == "tok-le2-act"
    assert activation_zero.label == "tok-le0-act"
    assert activation_zero.policy == "none"
    assert activation_zero.force_custom_autograd
    assert activation_upper.label == "tok-ge2-act"
    assert activation_upper.activation_save_threshold == 0
    assert activation_upper.activation_save_min == 2
    assert activation_upper.activation_save_max is None
    assert activation_upper_all.activation_drop_enabled
    assert activation_upper_all.activation_save_min == 1
    assert activation_upper_all.activation_save_max is None
    assert activation_bounded.label == "tok2-4-act"
    assert activation_bounded.activation_save_threshold == 4
    assert activation_bounded.activation_save_min == 2
    assert activation_bounded.activation_save_max == 4
    for invalid in ("tok2", "tok2-act", "tok0", "tok0-act"):
        with pytest.raises(ValueError, match="expected none, tok-leN, tok-geN, tokA-B"):
            parse_expert_recompute_policy_spec(invalid)
    with pytest.raises(ValueError, match="expected none, tok-leN, tok-geN, tokA-B"):
        parse_expert_recompute_policy_spec("bad-policy")


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


def test_asym_llama4_moe_torch_matches_eager_at_zero_delta() -> None:
    torch.manual_seed(3)
    source = FakeLlama4Moe()
    wrapped = AsymLlama4Moe(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    x = torch.randn(2, 5, source.hidden_dim, dtype=torch.bfloat16)

    expected, expected_logits = source(x)
    actual, actual_logits = wrapped(x)

    _assert_tensor_close_l2("llama4 moe output", actual, expected, max_abs_tol=3e-3, rel_l2_tol=2e-2)
    torch.testing.assert_close(actual_logits, expected_logits)
    assert torch.count_nonzero(wrapped.experts.gate_lora_B) == 0
    assert torch.count_nonzero(wrapped.experts.up_lora_B) == 0
    assert torch.count_nonzero(wrapped.experts.down_lora_B) == 0


@pytest.mark.parametrize("policy", ["tok-le2", "tok-le2-act"])
def test_asym_qwen3_experts_torch_recompute_policies_match_none(policy: str) -> None:
    torch.manual_seed(5)
    source_ref = FakeQwen3Experts()
    source_policy = FakeQwen3Experts()
    source_policy.load_state_dict(source_ref.state_dict())
    reference = AsymQwen3Experts(
        source_ref,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        expert_recompute_policy="none",
        init_lora_weights="peft",
    )
    candidate = AsymQwen3Experts(
        source_policy,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        expert_recompute_policy=policy,
        init_lora_weights="peft",
    )
    _copy_random_lora_params(reference, candidate)

    top_k_index, top_k_weights = _routing()
    for seed in (31, 32):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        x_ref = torch.randn(5, source_ref.hidden_dim, dtype=torch.bfloat16, generator=generator, requires_grad=True)
        x_candidate = x_ref.detach().clone().requires_grad_(True)
        out_ref = reference(x_ref, top_k_index, top_k_weights)
        out_candidate = candidate(x_candidate, top_k_index, top_k_weights)
        _assert_tensor_close_l2(f"{policy} output", out_candidate, out_ref)

        loss_ref = out_ref.float().square().mean() / 2.0
        loss_candidate = out_candidate.float().square().mean() / 2.0
        loss_ref.backward()
        loss_candidate.backward()
        _assert_grad_close(f"{policy} input seed={seed}", x_candidate.grad, x_ref.grad)

    candidate_params = dict(candidate.named_parameters())
    for name, param in reference.named_parameters():
        if "lora_" not in name:
            continue
        _assert_grad_close(f"{policy} {name}", candidate_params[name].grad, param.grad)


def test_qwen3_gate_up_lora_dropout_uses_independent_masks() -> None:
    torch.manual_seed(23)
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=2.0,
        lora_dropout=0.5,
        init_lora_weights="peft",
    )
    wrapped.train()
    with torch.no_grad():
        value_a = torch.randn_like(wrapped.gate_lora_A) * 0.1
        value_b = torch.randn_like(wrapped.gate_lora_B) * 0.1
        wrapped.gate_lora_A.copy_(value_a)
        wrapped.up_lora_A.copy_(value_a)
        wrapped.gate_lora_B.copy_(value_b)
        wrapped.up_lora_B.copy_(value_b)

    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16)
    top_k_index, top_k_weights = _routing()
    metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts=source.num_experts)
    packed = pack_tokens_contiguous(x, metadata)
    offsets, experts = make_dense_group_metadata(metadata.expert_offsets, num_groups=source.num_experts, device=packed.device)
    lora_metadata = wrapped._lora_metadata(offsets, experts, dense_experts=True)

    torch.manual_seed(29)
    gate_delta, up_delta = wrapped._forward_gate_up_lora(packed, offsets, experts, lora_metadata)

    assert not torch.equal(gate_delta, up_delta)


def test_qwen3_tok_act_does_not_run_subset_gate_up_recompute() -> None:
    torch.manual_seed(37)
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        expert_recompute_policy="tok-le2-act",
        init_lora_weights="peft",
    )
    subset_gate_up_calls = 0
    original = wrapped._forward_gate_up

    def counted_forward_gate_up(*args, **kwargs):
        nonlocal subset_gate_up_calls
        if kwargs.get("dense_experts") is False:
            subset_gate_up_calls += 1
        return original(*args, **kwargs)

    wrapped._forward_gate_up = counted_forward_gate_up  # type: ignore[method-assign]
    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16, requires_grad=True)
    top_k_index, top_k_weights = _routing()
    loss = wrapped(x, top_k_index, top_k_weights).float().square().mean()
    loss.backward()

    assert subset_gate_up_calls == 0


def test_qwen3_tok_policy_uses_saved_low_rank_without_subset_lora_a_recompute() -> None:
    torch.manual_seed(38)
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        expert_recompute_policy="tok-le2",
        init_lora_weights="peft",
    )
    subset_gate_up_calls = 0
    original = wrapped._forward_gate_up

    def counted_forward_gate_up(*args, **kwargs):
        nonlocal subset_gate_up_calls
        if kwargs.get("dense_experts") is False:
            subset_gate_up_calls += 1
        return original(*args, **kwargs)

    wrapped._forward_gate_up = counted_forward_gate_up  # type: ignore[method-assign]
    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16, requires_grad=True)
    top_k_index, top_k_weights = _routing()
    loss = wrapped(x, top_k_index, top_k_weights).float().square().mean()
    loss.backward()

    assert subset_gate_up_calls == 0


@pytest.mark.parametrize("policy", ["tok-le0", "tok-le2", "tok-le2-act"])
def test_asym_qwen3_experts_torch_recompute_lora_dropout_matches_none(policy: str) -> None:
    torch.manual_seed(39)
    source_ref = FakeQwen3Experts()
    source_policy = FakeQwen3Experts()
    source_policy.load_state_dict(source_ref.state_dict())
    reference = AsymQwen3Experts(
        source_ref,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.5,
        expert_recompute_policy="none",
        init_lora_weights="peft",
    )
    candidate = AsymQwen3Experts(
        source_policy,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.5,
        expert_recompute_policy=policy,
        init_lora_weights="peft",
    )
    _copy_random_lora_params(reference, candidate)
    reference.train()
    candidate.train()

    top_k_index, top_k_weights = _routing()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(191)
    x_ref = torch.randn(5, source_ref.hidden_dim, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    x_candidate = x_ref.detach().clone().requires_grad_(True)

    torch.manual_seed(313)
    out_ref = reference(x_ref, top_k_index, top_k_weights)
    torch.manual_seed(313)
    out_candidate = candidate(x_candidate, top_k_index, top_k_weights)
    _assert_tensor_close_l2(f"{policy} dropout output", out_candidate, out_ref)

    loss_ref = out_ref.float().square().mean() / 2.0
    loss_candidate = out_candidate.float().square().mean() / 2.0
    loss_ref.backward()
    loss_candidate.backward()
    _assert_grad_close(f"{policy} dropout input", x_candidate.grad, x_ref.grad)

    candidate_params = dict(candidate.named_parameters())
    for name, param in reference.named_parameters():
        if "lora_" not in name:
            continue
        _assert_grad_close(f"{policy} dropout {name}", candidate_params[name].grad, param.grad)


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
        expert_recompute_policy="tok-le2-act",
        strict=True,
    )

    assert report.qwen3_experts_wrapped == 2
    assert report.dense_lora_wrapped == 8
    assert report.expert_recompute_policy == "tok-le2-act"
    assert report.stats is not None
    assert "asym_forward_calls=0" in report.runtime_log_string()
    assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)
    assert isinstance(model.layers[0].mlp.experts.gate_up_base, TorchGroupedFrozenLinear)
    assert not model.layers[0].mlp.gate.weight.requires_grad
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in trainable)


def test_apply_lf_asym_lora_tags_gemma4_packed_experts() -> None:
    model = FakeGemma4Model()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        strict=True,
    )

    assert report.packed_experts_wrapped == 2
    assert report.qwen3_experts_wrapped == 2
    assert report.llama4_moes_wrapped == 0
    assert isinstance(model.layers[0].experts, AsymQwen3Experts)
    assert model.layers[0].experts.asym_expert_family == "gemma4"
    assert model.layers[0].experts.profile_prefix == "layers.0.experts"
    assert not model.layers[0].router.proj.weight.requires_grad
    assert not hasattr(model.layers[0].router.proj, "lora_A")


def test_apply_lf_asym_lora_skips_gemma4_router_proj_target() -> None:
    model = FakeGemma4Model()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "proj"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        strict=True,
    )

    assert report.packed_experts_wrapped == 2
    assert report.dense_lora_wrapped == 8
    assert "layers.0.router.proj:router" in report.skipped
    assert not model.layers[0].router.proj.weight.requires_grad
    assert not hasattr(model.layers[0].router.proj, "lora_A")


def test_apply_lf_asym_lora_wraps_llama4_moe_and_dense_without_router_lora() -> None:
    model = FakeLlama4Model()
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
        expert_recompute_policy="tok-le2-act",
        strict=True,
    )

    assert report.qwen3_experts_wrapped == 0
    assert report.llama4_moes_wrapped == 2
    assert report.dense_lora_wrapped == 14
    assert isinstance(model.layers[0].feed_forward, AsymLlama4Moe)
    assert isinstance(model.layers[0].feed_forward.experts, AsymQwen3Experts)
    assert not model.layers[0].feed_forward.router.weight.requires_grad
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("router" not in name for name in trainable)
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in trainable)


def test_adapt_lf_asym_peft_lora_only_wraps_packed_experts_after_dense_peft() -> None:
    model = FakeModel()
    model, report = adapt_lf_asym_peft_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        strict=True,
    )

    assert report.qwen3_experts_wrapped == 2
    assert report.dense_lora_wrapped == 0
    assert isinstance(model.layers[0].q_proj, nn.Linear)
    assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)
    assert not model.layers[0].q_proj.weight.requires_grad
    assert any("mlp.experts.gate_lora_A" in name for name, param in model.named_parameters() if param.requires_grad)


def test_asym_lf_adapter_state_saves_only_lora_and_loads(tmp_path) -> None:
    pytest.importorskip("safetensors.torch")
    torch.manual_seed(31)
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
    assert report.trainable_lora_params > 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" not in name and ".lora_A." not in name and ".lora_B." not in name:
                continue
            param.copy_(torch.randn(param.shape, device=param.device, dtype=param.dtype) * 0.01)

    state = get_asym_lora_state_dict(model)
    assert state
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in state)
    assert not any("host_weight" in name or "base_layer.weight" in name for name in state)

    save_asym_peft_adapter(
        model,
        tmp_path,
        metadata={
            "base_model_name_or_path": "fake-qwen3",
            "target_modules": ["all"],
            "r": 2,
            "lora_alpha": 4.0,
        },
    )
    assert (tmp_path / "adapter_model.safetensors").exists()
    config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["asym_gemm"] is True
    assert config["asym_adapter_format"] == "asym_gemm_lf_v1"
    assert config["asym_expert_format"] == "packed_gate_up_down"
    assert config["asym_expert_family"] == "qwen3"
    assert config["base_model_name_or_path"] == "fake-qwen3"

    reloaded = FakeModel()
    reloaded, _ = apply_lf_asym_lora(
        reloaded,
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
    load_asym_peft_adapter(reloaded, tmp_path)
    reloaded_state = get_asym_lora_state_dict(reloaded)
    assert list(reloaded_state) == list(state)
    for name, tensor in state.items():
        torch.testing.assert_close(reloaded_state[name], tensor)


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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_asym_qwen3_experts_sm100_backward_matches_torch_backend() -> None:
    torch.manual_seed(7)
    source_torch = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_asym = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_asym.load_state_dict(source_torch.state_dict())
    torch_backend = AsymQwen3Experts(
        source_torch,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    asym_backend = AsymQwen3Experts(
        source_asym,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
    )
    _copy_random_lora_params(torch_backend, asym_backend)

    x_torch = torch.randn(5, source_torch.hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_asym = x_torch.detach().clone().requires_grad_(True)
    top_k_index, top_k_weights = _routing()
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()

    out_torch = torch_backend(x_torch, top_k_index, top_k_weights)
    out_asym = asym_backend(x_asym, top_k_index, top_k_weights)
    _assert_tensor_close_l2("qwen3 output", out_asym, out_torch)

    grad_out = torch.randn_like(out_torch)
    out_torch.backward(grad_out)
    out_asym.backward(grad_out)

    _assert_grad_close("qwen3 input", x_asym.grad, x_torch.grad)
    asym_params = dict(asym_backend.named_parameters())
    for name, param in torch_backend.named_parameters():
        if "lora_" not in name:
            continue
        _assert_grad_close(name, asym_params[name].grad, param.grad)

    assert asym_backend.gate_up_base.host_weight.weight.grad is None
    assert asym_backend.down_base.host_weight.weight.grad is None
    assert asym_backend.stats.asym_forward_calls >= 2
    assert asym_backend.stats.asym_dx_calls >= 2


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
@pytest.mark.parametrize("policy", ["tok-le2", "tok-le2-act"])
def test_asym_qwen3_experts_sm100_recompute_policies_match_none(policy: str) -> None:
    torch.manual_seed(17)
    source_ref = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_policy = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_policy.load_state_dict(source_ref.state_dict())
    reference = AsymQwen3Experts(
        source_ref,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        expert_recompute_policy="none",
        init_lora_weights="peft",
    )
    candidate = AsymQwen3Experts(
        source_policy,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        expert_recompute_policy=policy,
        init_lora_weights="peft",
    )
    _copy_random_lora_params(reference, candidate)

    top_k_index, top_k_weights = _routing()
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()
    for seed in (41, 42):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        x_ref = torch.randn(5, source_ref.hidden_dim, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
        x_candidate = x_ref.detach().clone().requires_grad_(True)
        out_ref = reference(x_ref, top_k_index, top_k_weights)
        out_candidate = candidate(x_candidate, top_k_index, top_k_weights)
        _assert_tensor_close_l2(f"{policy} output", out_candidate, out_ref)

        loss_ref = out_ref.float().square().mean() / 2.0
        loss_candidate = out_candidate.float().square().mean() / 2.0
        loss_ref.backward()
        loss_candidate.backward()
        _assert_grad_close(f"{policy} input seed={seed}", x_candidate.grad, x_ref.grad)

    candidate_params = dict(candidate.named_parameters())
    for name, param in reference.named_parameters():
        if "lora_" not in name:
            continue
        _assert_grad_close(f"{policy} {name}", candidate_params[name].grad, param.grad)
    assert reference.stats.asym_forward_calls > 0
    assert candidate.stats.asym_forward_calls > 0
    assert candidate.stats.asym_dx_calls > 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
@pytest.mark.parametrize("policy", ["tok-le0", "tok-le2", "tok-le2-act"])
def test_asym_qwen3_experts_sm100_recompute_lora_dropout_matches_none(policy: str) -> None:
    torch.manual_seed(23)
    source_ref = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_policy = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    source_policy.load_state_dict(source_ref.state_dict())
    reference = AsymQwen3Experts(
        source_ref,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        expert_recompute_policy="none",
        init_lora_weights="peft",
    )
    candidate = AsymQwen3Experts(
        source_policy,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        expert_recompute_policy=policy,
        init_lora_weights="peft",
    )
    _copy_random_lora_params(reference, candidate)
    reference.train()
    candidate.train()

    top_k_index, top_k_weights = _routing()
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(47)
    x_ref = torch.randn(5, source_ref.hidden_dim, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    x_candidate = x_ref.detach().clone().requires_grad_(True)

    torch.manual_seed(557)
    out_ref = reference(x_ref, top_k_index, top_k_weights)
    torch.manual_seed(557)
    out_candidate = candidate(x_candidate, top_k_index, top_k_weights)
    _assert_tensor_close_l2(f"{policy} dropout output", out_candidate, out_ref, max_abs_tol=6e-3, rel_l2_tol=2e-2)

    loss_ref = out_ref.float().square().mean() / 2.0
    loss_candidate = out_candidate.float().square().mean() / 2.0
    loss_ref.backward()
    loss_candidate.backward()
    _assert_grad_close(f"{policy} dropout input", x_candidate.grad, x_ref.grad)

    candidate_params = dict(candidate.named_parameters())
    for name, param in reference.named_parameters():
        if "lora_" not in name:
            continue
        _assert_grad_close(f"{policy} dropout {name}", candidate_params[name].grad, param.grad)
    assert reference.stats.asym_forward_calls > 0
    assert candidate.stats.asym_forward_calls > 0
    assert candidate.stats.asym_dx_calls > 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_asym_qwen3_experts_sm100_recompute_lora_dropout_backward_consumes_no_rng() -> None:
    torch.manual_seed(29)
    source = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    wrapped = AsymQwen3Experts(
        source,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        expert_recompute_policy="tok-le2",
        init_lora_weights="peft",
    )
    _copy_random_lora_params(wrapped, wrapped)
    top_k_index, top_k_weights = _routing()
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(59)
    x = torch.randn(5, source.hidden_dim, device="cuda", dtype=torch.bfloat16, generator=generator, requires_grad=True)

    torch.manual_seed(701)
    out = wrapped(x, top_k_index, top_k_weights)
    loss = out.float().square().mean()
    rng_before = torch.cuda.get_rng_state()
    loss.backward()
    rng_after = torch.cuda.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_asym_qwen3_experts_sm100_recompute_offload_uses_custom_dense_path() -> None:
    torch.manual_seed(19)
    source = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
    wrapped = AsymQwen3Experts(
        source,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        expert_recompute_policy="tok-le2",
        init_lora_weights="peft",
    )
    assert not hasattr(wrapped, "_run_subset_body")
    assert not hasattr(wrapped, "_forward_expert_policy_subset")

    x = torch.randn(5, source.hidden_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    top_k_index, top_k_weights = _routing()
    loss = wrapped(x, top_k_index.cuda(), top_k_weights.cuda()).float().square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert wrapped.stats.asym_forward_calls > 0
    assert wrapped.stats.asym_dx_calls > 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_apply_lf_asym_lora_sm100_accumulates_and_optimizer_updates_only_lora() -> None:
    torch.manual_seed(11)
    model = FakeModel(hidden_dim=128, intermediate_dim=128).cuda()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="routed_experts",
        strict=True,
    )
    assert report.qwen3_experts_wrapped == 2
    assert report.trainable_lora_params > 0

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    before = {name: param.detach().clone() for name, param in trainable}
    optimizer = torch.optim.SGD([param for _name, param in trainable], lr=1e-3)
    top_k_index, top_k_weights = _routing()
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()

    for seed in (101, 102):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
        loss = model(x, top_k_index, top_k_weights).float().square().mean() / 2.0
        loss.backward()

    active_grads = [(name, param.grad) for name, param in trainable if param.grad is not None]
    assert active_grads
    for name, grad in active_grads:
        assert torch.isfinite(grad.float()).all(), f"{name} grad has non-finite values"
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"frozen parameter unexpectedly received grad: {name}"

    optimizer.step()
    changed = [
        name
        for name, param in trainable
        if not torch.equal(param.detach().cpu(), before[name].detach().cpu())
    ]
    assert changed
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in changed)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")
def test_asym_qwen3_experts_sm100_checkpoint_and_anomaly_smoke() -> None:
    torch.manual_seed(13)
    source = FakeQwen3Experts(hidden_dim=128, intermediate_dim=128).cuda()
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
    top_k_index = top_k_index.cuda()
    top_k_weights = top_k_weights.cuda()

    def run_experts(hidden_states: torch.Tensor) -> torch.Tensor:
        return wrapped(hidden_states, top_k_index, top_k_weights)

    with torch.autograd.detect_anomaly(check_nan=True):
        out = checkpoint(run_experts, x, use_reentrant=False, preserve_rng_state=False)
        loss = out.float().square().mean()
        loss.backward()

    assert torch.isfinite(loss)
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
    assert wrapped.stats.asym_forward_calls >= 2
    assert wrapped.stats.asym_dx_calls >= 2
    for name, param in wrapped.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} missing grad"
            assert torch.isfinite(param.grad.float()).all(), f"{name} grad has non-finite values"
