from __future__ import annotations

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from asym_gemm.integrations.lf import apply_lf_asym_lora
from asym_gemm.training.frozen_linear import TorchGroupedFrozenLinear
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


def test_parse_expert_recompute_policy_spec() -> None:
    none = parse_expert_recompute_policy_spec(None)
    split = parse_expert_recompute_policy_spec("split2")
    full = parse_expert_recompute_policy_spec("tok2-ckpt")
    activation = parse_expert_recompute_policy_spec("tok2-act-ckpt")

    assert none.label == "none"
    assert split.policy == "split"
    assert split.token_threshold == 2
    assert full.policy == "tok"
    assert full.token_threshold == 2
    assert activation.policy == "none"
    assert activation.activation_save_policy == "tok_act"
    assert activation.activation_save_threshold == 2
    with pytest.raises(ValueError, match="ambiguous"):
        parse_expert_recompute_policy_spec("tok2")


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


@pytest.mark.parametrize("policy", ["split2", "tok2-ckpt", "tok2-act-ckpt"])
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
        expert_recompute_policy="split2",
        strict=True,
    )

    assert report.qwen3_experts_wrapped == 2
    assert report.dense_lora_wrapped == 8
    assert report.expert_recompute_policy == "split2"
    assert report.stats is not None
    assert "asym_forward_calls=0" in report.runtime_log_string()
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
@pytest.mark.parametrize("policy", ["split2", "tok2-ckpt", "tok2-act-ckpt"])
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
def test_asym_qwen3_experts_sm100_recompute_offload_uses_subset_path() -> None:
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
        expert_recompute_policy="tok2-ckpt",
        init_lora_weights="peft",
    )

    def fail_dense_checkpoint(*_args, **_kwargs):
        raise AssertionError("asym+offload recompute should use per-group subsets, not dense checkpoint")

    wrapped._run_dense_checkpoint_body = fail_dense_checkpoint
    wrapped._run_dense_checkpoint_activation_down = fail_dense_checkpoint

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
