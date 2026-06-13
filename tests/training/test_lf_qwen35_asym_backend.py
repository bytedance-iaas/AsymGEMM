from __future__ import annotations

import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from asym_gemm.integrations.lf import apply_lf_asym_lora, get_asym_lora_state_dict, load_asym_peft_adapter, save_asym_peft_adapter
from asym_gemm.integrations.peft_lf import adapt_lf_asym_peft_lora
from asym_gemm.training.offload import AsymFrozenEmbedding, AsymFrozenLayerNorm, AsymFrozenRMSNorm
from asym_gemm.training.lora import AsymLoRALinear, TorchLoRALinear
from asym_gemm.training.qwen3_moe import AsymQwen3Experts, AsymQwen3Router, is_qwen3_moe_block
from asym_gemm.training.qwen35_moe import AsymQwen35MoeBlock, is_qwen35_moe_block


class FakeQwen3_5Experts(nn.Module):
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


class FakeQwen3_5Gate(nn.Linear):
    def __init__(self, *, hidden_dim: int = 8, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__(hidden_dim, num_experts, bias=False, dtype=torch.bfloat16)
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = super().forward(hidden_states)
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class FakeQwen3_5SharedExpert(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeQwen3_5MoeBlock(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.gate = FakeQwen3_5Gate(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
        self.experts = FakeQwen3_5Experts(num_experts=num_experts, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
        self.shared_expert = FakeQwen3_5SharedExpert(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
        self.shared_expert_gate = nn.Linear(hidden_dim, 1, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, input_shape[-1])
        shared = self.shared_expert(flat)
        _router_logits, weights, indices = self.gate(flat)
        routed = self.experts(flat, indices, weights)
        shared = F.sigmoid(self.shared_expert_gate(flat)) * shared
        return (routed + shared).reshape(input_shape)


class FakeQwen3NextMoeBlock(FakeQwen3_5MoeBlock):
    pass


class FakeQwen3_5Block(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.mlp = FakeQwen3_5MoeBlock(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)


class FakeQwen3_5Model(nn.Module):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeQwen3_5Block(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)


class FakeQwen3_5WholeOffloadModel(FakeQwen3_5Model):
    def __init__(self, *, hidden_dim: int = 8, intermediate_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers)
        self.embed_tokens = nn.Embedding(hidden_dim, hidden_dim, dtype=torch.bfloat16)
        self.input_layernorm = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def _dense_targets() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "shared_expert_gate"]


def _assert_cpu_owner_adopted_or_pinned_replacement(tensor: torch.Tensor, source_data_ptr: int) -> None:
    assert tensor.device.type == "cpu"
    if torch.cuda.is_available() and tensor.is_pinned():
        return
    assert tensor.untyped_storage().data_ptr() == source_data_ptr


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=4e-3, rtol=2e-2)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch") from exc


def test_qwen35_matcher_accepts_qwen35_and_rejects_qwen3_and_lookalike() -> None:
    qwen35 = FakeQwen3_5MoeBlock()
    lookalike = FakeQwen3NextMoeBlock()

    assert is_qwen35_moe_block(qwen35)
    assert not is_qwen3_moe_block(qwen35)
    assert not is_qwen35_moe_block(lookalike)
    assert not is_qwen3_moe_block(lookalike)


def test_asym_qwen35_moe_whole_matches_source_and_detaches_router() -> None:
    torch.manual_seed(7)
    source = FakeQwen3_5MoeBlock()
    x = torch.randn(2, 5, source.gate.hidden_dim, dtype=torch.bfloat16, requires_grad=True)
    expected = source(x)

    wrapped = AsymQwen35MoeBlock(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    actual = wrapped(x)

    _assert_close("qwen35 whole moe", actual, expected)
    flat = x.view(-1, source.gate.hidden_dim)
    _indices, weights, router_logits = wrapped._compute_routing(flat)
    assert not weights.requires_grad
    assert router_logits is not None and not router_logits.requires_grad
    assert list(wrapped._modules)[:4] == ["gate", "experts", "shared_expert", "shared_expert_gate"]


def test_asym_qwen35_gc_exp_accepts_and_runs_packed_experts() -> None:
    torch.manual_seed(11)
    source = FakeQwen3_5MoeBlock()
    wrapped = AsymQwen35MoeBlock(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        expert_recompute_policy="gc-exp",
    )
    assert wrapped.experts.expert_recompute_config.label == "gc-exp"
    assert wrapped.experts.expert_recompute_config.torch_checkpoint_enabled

    x = torch.randn(2, 5, source.gate.hidden_dim, dtype=torch.bfloat16, requires_grad=True)
    loss = wrapped(x).float().square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
    for name, param in wrapped.experts.named_parameters():
        if "lora_" in name:
            assert param.grad is not None, f"{name} missing grad"


def test_real_transformers_qwen35_sparse_moe_block_matches_wrapper() -> None:
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")

    torch.manual_seed(17)
    config = configuration.Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    source = modeling.Qwen3_5MoeSparseMoeBlock(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for param in source.parameters():
            param.normal_(mean=0.0, std=0.02)

    x = torch.randn(2, 3, config.hidden_size, dtype=torch.bfloat16)
    expected = source(x)

    assert is_qwen35_moe_block(source)
    assert not is_qwen3_moe_block(source)
    wrapped = AsymQwen35MoeBlock(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    actual = wrapped(x)

    _assert_close("real transformers qwen35 sparse moe", actual, expected)


def test_asym_frozen_qwen35_rmsnorm_matches_shifted_weight_formula() -> None:
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")

    torch.manual_seed(19)
    source = modeling.Qwen3_5MoeRMSNorm(8).to(dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.normal_(mean=0.0, std=0.1)
    source_key = source.weight.untyped_storage().data_ptr()
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    expected = source(x)

    wrapped = AsymFrozenRMSNorm(source)
    actual = wrapped(x)

    assert wrapped.host_weight.weight.untyped_storage().data_ptr() == source_key
    assert wrapped.shifted_weight
    assert not wrapped.gated
    torch.testing.assert_close(actual, expected)


def test_asym_frozen_qwen35_gated_rmsnorm_matches_transformers() -> None:
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")

    torch.manual_seed(23)
    source = modeling.Qwen3_5MoeRMSNormGated(8).to(dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.normal_(mean=1.0, std=0.1)
    source_key = source.weight.untyped_storage().data_ptr()
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    gate = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    expected = source(x, gate)

    wrapped = AsymFrozenRMSNorm(source)
    actual = wrapped(x, gate)

    assert wrapped.host_weight.weight.untyped_storage().data_ptr() == source_key
    assert wrapped.gated
    assert not wrapped.shifted_weight
    torch.testing.assert_close(actual, expected)


def test_asym_qwen35_moe_keeps_shared_branch_when_routed_experts_are_zero() -> None:
    torch.manual_seed(11)
    source = FakeQwen3_5MoeBlock()
    with torch.no_grad():
        source.experts.gate_up_proj.zero_()
        source.experts.down_proj.zero_()
    x = torch.randn(2, 3, source.gate.hidden_dim, dtype=torch.bfloat16)
    expected = source(x)

    wrapped = AsymQwen35MoeBlock(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    actual = wrapped(x)

    assert expected.abs().max() > 0
    _assert_close("qwen35 shared-only moe", actual, expected)


def test_asym_qwen35_moe_keeps_routed_branch_when_shared_expert_is_zero() -> None:
    torch.manual_seed(13)
    source = FakeQwen3_5MoeBlock()
    with torch.no_grad():
        for param in source.shared_expert.parameters():
            param.zero_()
    x = torch.randn(2, 3, source.gate.hidden_dim, dtype=torch.bfloat16)
    expected = source(x)

    wrapped = AsymQwen35MoeBlock(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    actual = wrapped(x)

    assert expected.abs().max() > 0
    _assert_close("qwen35 routed-only moe", actual, expected)


def test_apply_lf_asym_lora_whole_wraps_qwen35_and_dense_shared_modules() -> None:
    model = FakeQwen3_5Model()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        expert_recompute_policy="tok-le2-act",
        router_mode="whole",
        strict=True,
    )

    assert report.qwen35_moes_wrapped == 2
    assert report.qwen3_moes_wrapped == 0
    assert report.qwen3_experts_wrapped == 0
    assert report.dense_lora_wrapped == 16
    assert isinstance(model.layers[0].mlp, AsymQwen35MoeBlock)
    assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)
    assert model.layers[0].mlp.experts.profile_prefix == "layers.0.mlp.experts"
    assert hasattr(model.layers[0].mlp.shared_expert.gate_proj, "lora_A")
    assert hasattr(model.layers[0].mlp.shared_expert_gate, "lora_A")
    router_trainable = [name for name, param in model.named_parameters() if ".mlp.gate." in name and param.requires_grad]
    assert router_trainable == []


def test_apply_lf_asym_lora_qwen35_adopts_cpu_expert_storage_without_clone() -> None:
    model = FakeQwen3_5Model()
    first_experts = model.layers[0].mlp.experts
    gate_up_key = first_experts.gate_up_proj.untyped_storage().data_ptr()
    down_key = first_experts.down_proj.untyped_storage().data_ptr()

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["experts"],
        dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="routed_experts",
        router_mode="whole",
        strict=True,
    )

    wrapped = model.layers[0].mlp.experts
    assert isinstance(wrapped, AsymQwen3Experts)
    _assert_cpu_owner_adopted_or_pinned_replacement(wrapped.gate_up_base.host_weight.weight, gate_up_key)
    _assert_cpu_owner_adopted_or_pinned_replacement(wrapped.down_base.host_weight.weight, down_key)
    assert report.cpu_resident_base_bytes_by_component["routed_experts"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert not any(
        ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name
        for name, _param in model.named_parameters()
    )


def test_apply_lf_asym_lora_qwen35_shared_experts_adopt_cpu_storage_without_clone() -> None:
    model = FakeQwen3_5Model()
    gate_proj_key = model.layers[0].mlp.shared_expert.gate_proj.weight.untyped_storage().data_ptr()
    shared_gate_key = model.layers[0].mlp.shared_expert_gate.weight.untyped_storage().data_ptr()

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["gate_proj", "shared_expert_gate"],
        dense_target_modules=["gate_proj", "shared_expert_gate"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="shared_experts",
        router_mode="whole",
        strict=True,
    )

    shared_expert = model.layers[0].mlp.shared_expert
    assert isinstance(shared_expert.gate_proj, AsymLoRALinear)
    assert isinstance(model.layers[0].mlp.shared_expert_gate, AsymLoRALinear)
    _assert_cpu_owner_adopted_or_pinned_replacement(shared_expert.gate_proj.base_layer.host_weight.weight, gate_proj_key)
    _assert_cpu_owner_adopted_or_pinned_replacement(
        model.layers[0].mlp.shared_expert_gate.base_layer.host_weight.weight,
        shared_gate_key,
    )
    assert report.cpu_resident_base_bytes_by_component["shared_experts"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert not any(".mlp.shared_expert.gate_proj.weight" in name for name, _param in model.named_parameters())
    assert not any(".mlp.shared_expert_gate.weight" in name for name, _param in model.named_parameters())


def test_qwen35_shared_expert_gate_uses_torch_cpu_fetch_when_shape_is_not_direct_bf16_compatible() -> None:
    model = FakeQwen3_5Model(hidden_dim=64, intermediate_dim=64)
    gate_proj_key = model.layers[0].mlp.shared_expert.gate_proj.weight.untyped_storage().data_ptr()
    shared_gate_key = model.layers[0].mlp.shared_expert_gate.weight.untyped_storage().data_ptr()

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["gate_proj", "shared_expert_gate"],
        dense_target_modules=["gate_proj", "shared_expert_gate"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="shared_experts",
        router_mode="whole",
        strict=True,
    )

    gate_proj = model.layers[0].mlp.shared_expert.gate_proj
    shared_gate = model.layers[0].mlp.shared_expert_gate
    assert isinstance(gate_proj, AsymLoRALinear)
    assert isinstance(shared_gate, AsymLoRALinear)
    assert gate_proj.base_layer.backend == "asym"
    assert shared_gate.base_layer.backend == "torch"
    _assert_cpu_owner_adopted_or_pinned_replacement(gate_proj.base_layer.host_weight.weight, gate_proj_key)
    _assert_cpu_owner_adopted_or_pinned_replacement(shared_gate.base_layer.host_weight.weight, shared_gate_key)
    assert any(
        "shared_expert_gate:torch_cpu_fetched:requires_8_aligned_nk" in skipped for skipped in report.skipped
    )


def test_apply_lf_asym_lora_qwen35_router_adopts_cpu_storage_without_clone() -> None:
    model = FakeQwen3_5Model()
    router_key = model.layers[0].mlp.gate.weight.untyped_storage().data_ptr()

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["experts"],
        dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="router",
        router_mode="whole",
        strict=True,
    )

    wrapped = model.layers[0].mlp.gate
    assert isinstance(wrapped, AsymQwen3Router)
    _assert_cpu_owner_adopted_or_pinned_replacement(wrapped.proj.host_weight.weight, router_key)
    assert report.cpu_resident_base_bytes_by_component["router"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert not any(".mlp.gate.weight" in name for name, _param in model.named_parameters())


def test_apply_lf_asym_lora_qwen35_all_selector_covers_whole_model_buckets() -> None:
    model = FakeQwen3_5WholeOffloadModel()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="all",
        router_mode="whole",
        strict=True,
    )

    expected_components = {
        "routed_experts",
        "router",
        "shared_experts",
        "attention",
        "embed_tokens",
        "lm_head",
        "norms",
    }
    assert expected_components <= set(report.cpu_resident_base_bytes_by_component)
    assert all(report.cpu_resident_base_bytes_by_component[component] > 0 for component in expected_components)
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert isinstance(model.embed_tokens, AsymFrozenEmbedding)
    assert isinstance(model.input_layernorm, AsymFrozenLayerNorm)
    assert hasattr(model.lm_head, "host_weight")


def test_adapt_lf_asym_peft_lora_preserves_preexisting_shared_expert_lora() -> None:
    model = FakeQwen3_5Model()
    prewrapped_gate_proj = TorchLoRALinear(
        model.layers[0].mlp.shared_expert.gate_proj,
        rank=2,
        alpha=4.0,
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
    )
    prewrapped_shared_gate = TorchLoRALinear(
        model.layers[0].mlp.shared_expert_gate,
        rank=2,
        alpha=4.0,
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=0.0,
    )
    model.layers[0].mlp.shared_expert.gate_proj = prewrapped_gate_proj
    model.layers[0].mlp.shared_expert_gate = prewrapped_shared_gate

    model, report = adapt_lf_asym_peft_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=True,
    )

    assert report.qwen35_moes_wrapped == 2
    assert report.dense_lora_wrapped == 2
    assert model.layers[0].mlp.shared_expert.gate_proj is prewrapped_gate_proj
    assert model.layers[0].mlp.shared_expert_gate is prewrapped_shared_gate
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any("shared_expert.gate_proj.lora_A" in name for name in trainable)
    assert any("shared_expert_gate.lora_A" in name for name in trainable)
    assert all("router" not in name and ".mlp.gate." not in name for name in trainable)


def test_apply_lf_asym_lora_hf_wraps_only_qwen35_experts() -> None:
    model = FakeQwen3_5Model()
    original_mlp = model.layers[0].mlp
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
        router_mode="hf",
        strict=True,
    )

    assert report.packed_experts_wrapped == 2
    assert report.qwen35_moes_wrapped == 0
    assert model.layers[0].mlp is original_mlp
    assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)
    assert model.layers[0].mlp.experts.asym_expert_family == "qwen3_5"


def test_apply_lf_asym_lora_whole_rejects_nested_router_logits_config() -> None:
    model = FakeQwen3_5Model()
    model.config = type("Config", (), {"text_config": type("TextConfig", (), {"output_router_logits": True})()})()
    with pytest.raises(ValueError, match="output_router_logits=False"):
        apply_lf_asym_lora(
            model,
            raw_lora_target=["all"],
            dense_target_modules=_dense_targets(),
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            backend="torch",
            precision="bf16",
            offload_modules="none",
            router_mode="whole",
            strict=True,
        )


def test_asym_qwen35_whole_adapter_state_saves_and_loads(tmp_path) -> None:
    pytest.importorskip("safetensors.torch")
    torch.manual_seed(31)
    model = FakeQwen3_5Model()
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        router_mode="whole",
        strict=True,
    )
    assert report.trainable_lora_params > 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" not in name and ".lora_A." not in name and ".lora_B." not in name:
                continue
            param.copy_(torch.randn(param.shape, device=param.device, dtype=param.dtype) * 0.01)

    state = get_asym_lora_state_dict(model)
    assert any("mlp.experts.gate_lora_A" in name for name in state)
    assert any("mlp.shared_expert.gate_proj.lora_A" in name for name in state)
    assert any("mlp.shared_expert_gate.lora_A" in name for name in state)

    save_asym_peft_adapter(
        model,
        tmp_path,
        metadata={
            "base_model_name_or_path": "fake-qwen35",
            "target_modules": ["all"],
            "r": 2,
            "lora_alpha": 4.0,
        },
    )
    config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["asym_expert_format"] == "qwen3_5_owned_moe"
    assert config["asym_expert_family"] == "qwen3_5"
    assert config["asym_router_mode"] == "whole"

    reloaded = FakeQwen3_5Model()
    reloaded, _ = apply_lf_asym_lora(
        reloaded,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="torch",
        precision="bf16",
        offload_modules="none",
        router_mode="whole",
        strict=True,
    )
    load_asym_peft_adapter(reloaded, tmp_path)
    reloaded_state = get_asym_lora_state_dict(reloaded)
    assert list(reloaded_state) == list(state)
    for name, tensor in state.items():
        torch.testing.assert_close(reloaded_state[name], tensor)
