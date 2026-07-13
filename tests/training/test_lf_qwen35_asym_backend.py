from __future__ import annotations

import json

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import asym_gemm
from asym_gemm.integrations.lf import (
    _infer_adapter_config,
    apply_lf_asym_lora,
    audit_lf_frozen_cuda_residue,
    classify_lf_component,
    component_is_selected,
    get_asym_lora_state_dict,
    load_asym_peft_adapter,
    move_lf_asym_cpu_first_model_to_device,
    parse_lf_offload_modules,
    save_asym_peft_adapter,
)
from asym_gemm.integrations.peft_lf import adapt_lf_asym_peft_lora
from asym_gemm.training.attention_activation_offload import AsymActivationOffloadLoRALinear
from asym_gemm.training.decoder_activation_offload import (
    decoder_saved_tensor_offload_module_names,
    is_decoder_saved_tensor_offload_wrapper,
)
from asym_gemm.training.linear_attention_activation_offload import (
    is_linear_attention_saved_tensor_offload_wrapper,
    linear_attention_saved_tensor_offload_module_names,
)
from asym_gemm.training.decoder_checkpoint import decoder_checkpoint_module_names, is_decoder_checkpoint_wrapper
from asym_gemm.training.offload import AsymFrozenEmbedding, AsymFrozenLayerNorm, AsymFrozenRMSNorm
from asym_gemm.training.lora import AsymLoRALinear, TorchLoRALinear
from asym_gemm.training.qwen3_moe import AsymQwen3Experts, AsymQwen3Router, is_qwen3_moe_block
from asym_gemm.training.qwen35_moe import AsymQwen35MoeBlock, is_qwen35_moe_block
from asym_gemm.training.qwen35_shared_mlp import AsymQwen35SharedMLP
from asym_gemm.training.weight_offload import LoRAWeightOffloadCoordinator, install_lora_weight_offload


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
        self.act_fn = F.silu
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakePeftLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, *, rank: int = 2, alpha: float = 4.0) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(base_layer.in_features, rank, bias=False, dtype=torch.bfloat16)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(rank, base_layer.out_features, bias=False, dtype=torch.bfloat16)}
        )
        self.active_adapter = "default"
        self.scaling = {"default": float(alpha) / float(rank)}

    def get_base_layer(self) -> nn.Linear:
        return self.base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x)) * self.scaling["default"]


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


class FakeQwen3_5LinearAttention(nn.Module):
    def __init__(self, *, hidden_dim: int = 8) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.in_proj_z = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.in_proj_b = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.in_proj_a = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        projected = self.in_proj_qkv(hidden_states)
        projected = projected + self.in_proj_z(hidden_states)
        projected = projected + self.in_proj_b(hidden_states)
        projected = projected + self.in_proj_a(hidden_states)
        return self.out_proj(projected)


class FakeQwen3_5SelfAttention(nn.Module):
    def __init__(self, *, hidden_dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor, None]:
        return self.o_proj(self.v_proj(hidden_states)), None


class FakeQwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        mixer: str,
        hidden_dim: int = 8,
        intermediate_dim: int = 8,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "qwen3_5_moe_text"})()
        if mixer == "linear_attn":
            self.linear_attn = FakeQwen3_5LinearAttention(hidden_dim=hidden_dim)
        elif mixer == "self_attn":
            self.self_attn = FakeQwen3_5SelfAttention(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")
        self.mlp = FakeQwen3_5MoeBlock(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if hasattr(self, "linear_attn"):
            mixer_out = self.linear_attn(hidden_states, **kwargs)
        else:
            mixer_out, _ = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + mixer_out.to(dtype=residual.dtype)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states.to(dtype=residual.dtype)


class FakeQwen3_5DecoderModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        intermediate_dim: int = 8,
        layer_mixers: tuple[str, ...] = ("linear_attn", "self_attn"),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FakeQwen3_5DecoderLayer(
                    mixer=mixer,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                )
                for mixer in layer_mixers
            ]
        )
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return self.lm_head(hidden_states)


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
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "shared_expert_gate",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ]


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


def _sm100_bf16_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 10
        and hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous")
    )


def test_qwen35_matcher_accepts_qwen35_and_rejects_qwen3_and_lookalike() -> None:
    qwen35 = FakeQwen3_5MoeBlock()
    lookalike = FakeQwen3NextMoeBlock()

    assert is_qwen35_moe_block(qwen35)
    assert not is_qwen3_moe_block(qwen35)
    assert not is_qwen35_moe_block(lookalike)
    assert not is_qwen3_moe_block(lookalike)


def test_qwen35_matcher_rejects_bad_shared_expert_shapes() -> None:
    qwen35 = FakeQwen3_5MoeBlock(hidden_dim=8, intermediate_dim=8)
    qwen35.shared_expert.down_proj = nn.Linear(7, 8, bias=False, dtype=torch.bfloat16)

    assert not is_qwen35_moe_block(qwen35)


def test_decoder_layer_matcher_accepts_qwen35_hybrid_layers() -> None:
    from asym_gemm.integrations.lf import _is_qwen3_decoder_layer_module_name as _is_decoder

    assert _is_decoder("model.layers.0", FakeQwen3_5DecoderLayer(mixer="linear_attn"))
    assert _is_decoder("model.layers.1", FakeQwen3_5DecoderLayer(mixer="self_attn"))
    assert not _is_decoder("model.vision_model.layers.0", FakeQwen3_5DecoderLayer(mixer="linear_attn"))
    assert not _is_decoder("model.layers.0", nn.Linear(8, 8))


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


def test_qwen35_shared_mlp_wrapper_matches_source() -> None:
    torch.manual_seed(37)
    source = FakeQwen3_5SharedExpert(hidden_dim=8, intermediate_dim=8)
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    expected = source(x)

    wrapped = AsymQwen35SharedMLP(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    actual = wrapped(x)

    _assert_close("qwen35 shared mlp", actual, expected)
    assert wrapped.profile_prefix == "layers.unknown.mlp.shared_expert"
    assert wrapped.preexisting_lora_leaf_count == 0


def test_apply_lf_asym_lora_qwen35_shared_mlp_wrapper_adopts_all_shared_leaves() -> None:
    model = FakeQwen3_5Model(hidden_dim=64, intermediate_dim=64, num_layers=1)
    source_keys = {
        leaf: getattr(model.layers[0].mlp.shared_expert, leaf).weight.untyped_storage().data_ptr()
        for leaf in ("gate_proj", "up_proj", "down_proj")
    }

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="shared_experts",
        router_mode="whole",
        strict=True,
    )

    shared = model.layers[0].mlp.shared_expert
    assert isinstance(shared, AsymQwen35SharedMLP)
    assert shared.profile_prefix == "layers.0.mlp.shared_expert"
    assert report.cpu_resident_base_bytes_by_component["shared_experts"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    for leaf, source_key in source_keys.items():
        wrapped_leaf = getattr(shared, leaf)
        assert isinstance(wrapped_leaf, AsymLoRALinear)
        assert wrapped_leaf.base_layer.backend == "asym"
        _assert_cpu_owner_adopted_or_pinned_replacement(wrapped_leaf.base_layer.host_weight.weight, source_key)
        assert not any(f".mlp.shared_expert.{leaf}.weight" in name for name, _param in model.named_parameters())
    assert isinstance(model.layers[0].mlp.shared_expert_gate, AsymLoRALinear)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any("shared_expert.gate_proj.lora_A" in name for name in trainable)
    assert any("shared_expert.up_proj.lora_A" in name for name in trainable)
    assert any("shared_expert.down_proj.lora_A" in name for name in trainable)
    assert any("shared_expert_gate.lora_A" in name for name in trainable)
    assert all(".mlp.gate." not in name for name in trainable)


def test_adapt_lf_asym_peft_lora_qwen35_shared_mlp_replaces_preexisting_peft_leaves() -> None:
    model = FakeQwen3_5Model(hidden_dim=64, intermediate_dim=64, num_layers=1)
    shared = model.layers[0].mlp.shared_expert
    shared.gate_proj = FakePeftLinear(shared.gate_proj, rank=2, alpha=4.0)
    shared.up_proj = FakePeftLinear(shared.up_proj, rank=2, alpha=4.0)
    shared.down_proj = FakePeftLinear(shared.down_proj, rank=2, alpha=4.0)
    gate_a_before = shared.gate_proj.lora_A["default"].weight.detach().clone()
    up_a_before = shared.up_proj.lora_A["default"].weight.detach().clone()
    down_a_before = shared.down_proj.lora_A["default"].weight.detach().clone()

    model, report = adapt_lf_asym_peft_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        asym_owned_dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="shared_experts",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=True,
    )

    wrapped = model.layers[0].mlp.shared_expert
    assert isinstance(wrapped, AsymQwen35SharedMLP)
    assert wrapped.preexisting_lora_leaf_count == 3
    assert torch.equal(wrapped.gate_proj.lora_a.detach().cpu(), gate_a_before)
    assert torch.equal(wrapped.up_proj.lora_a.detach().cpu(), up_a_before)
    assert torch.equal(wrapped.down_proj.lora_a.detach().cpu(), down_a_before)
    assert report.cpu_resident_base_bytes_by_component["shared_experts"] > 0


@pytest.mark.skipif(not _sm100_bf16_available(), reason="Qwen3.5 shared activation offload requires SM100 CUDA kernels")
def test_qwen35_shared_mlp_activation_offload_matches_torch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_EXPERT_ACT_OFFLOAD", "1")
    torch.manual_seed(41)
    source_state = FakeQwen3_5SharedExpert(hidden_dim=64, intermediate_dim=64).state_dict()
    source_ref = FakeQwen3_5SharedExpert(hidden_dim=64, intermediate_dim=64).to(device="cuda", dtype=torch.bfloat16)
    source_asym = FakeQwen3_5SharedExpert(hidden_dim=64, intermediate_dim=64).to(dtype=torch.bfloat16)
    source_ref.load_state_dict(source_state)
    source_asym.load_state_dict(source_state)

    ref = AsymQwen35SharedMLP(
        source_ref,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    ).train()
    asym = AsymQwen35SharedMLP(
        source_asym,
        backend="asym",
        precision="bf16",
        offload=True,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    ).train()

    with torch.no_grad():
        for ref_leaf, asym_leaf in (
            (ref.gate_proj, asym.gate_proj),
            (ref.up_proj, asym.up_proj),
            (ref.down_proj, asym.down_proj),
        ):
            ref_leaf.lora_a.normal_(mean=0.0, std=0.02)
            ref_leaf.lora_b.normal_(mean=0.0, std=0.02)
            asym_leaf.lora_a.copy_(ref_leaf.lora_a)
            asym_leaf.lora_b.copy_(ref_leaf.lora_b)

    x_ref = torch.randn(3, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_asym = x_ref.detach().clone().requires_grad_(True)
    y_ref = ref(x_ref)
    y_asym = asym(x_asym)
    _assert_close("qwen35 shared activation offload forward", y_asym, y_ref)

    loss_ref = y_ref.float().square().mean()
    loss_asym = y_asym.float().square().mean()
    loss_ref.backward()
    loss_asym.backward()

    _assert_close("qwen35 shared activation offload input grad", x_asym.grad, x_ref.grad)
    assert asym._last_activation_offload_stats
    for name, param in asym.named_parameters():
        if "lora_" in name or ".lora_A." in name or ".lora_B." in name:
            assert param.grad is not None, f"{name} missing grad"


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


def test_qwen35_linear_attention_selector_adopts_gdn_projection_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_ATTN_ACT_OFFLOAD", "1")
    model = FakeQwen3_5DecoderModel(hidden_dim=64, intermediate_dim=64, layer_mixers=("linear_attn",))
    source_keys = {
        leaf: getattr(model.layers[0].linear_attn, leaf).weight.untyped_storage().data_ptr()
        for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    }

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["all"],
        dense_target_modules=_dense_targets(),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="linear_attention",
        router_mode="whole",
        strict=True,
    )

    selection = parse_lf_offload_modules("linear_attention")
    assert selection.implemented_components == frozenset({"linear_attention"})
    assert component_is_selected("linear_attention", "in_proj_qkv", selection)
    assert classify_lf_component("layers.0.linear_attn.in_proj_qkv") == "linear_attention"
    assert report.cpu_resident_base_bytes_by_component["linear_attention"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert not report.attention_act_offload_modules
    for leaf, source_key in source_keys.items():
        wrapped = getattr(model.layers[0].linear_attn, leaf)
        assert isinstance(wrapped, AsymLoRALinear)
        assert not isinstance(wrapped, AsymActivationOffloadLoRALinear)
        assert wrapped.base_layer.backend == "asym"
        _assert_cpu_owner_adopted_or_pinned_replacement(wrapped.base_layer.host_weight.weight, source_key)
        assert not any(f".linear_attn.{leaf}.weight" in name for name, _param in model.named_parameters())


def test_qwen35_all_selector_includes_linear_attention() -> None:
    model = FakeQwen3_5DecoderModel(hidden_dim=64, intermediate_dim=64, layer_mixers=("linear_attn",))
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

    assert parse_lf_offload_modules("all").linear_attention
    assert report.cpu_resident_base_bytes_by_component["linear_attention"] > 0
    assert report.selected_gpu_resident_base_bytes_by_component == {}
    assert isinstance(model.layers[0].linear_attn.in_proj_qkv, AsymLoRALinear)


def test_qwen35_vision_paths_are_not_misclassified_as_text_offload_components() -> None:
    assert classify_lf_component("visual.pos_embed") == "vision"
    assert classify_lf_component("model.visual.blocks.0.mlp.fc1.weight") == "vision"
    assert classify_lf_component("model.visual.merger.mlp.0.weight") == "vision"
    assert classify_lf_component("multi_modal_projector.linear.weight") == "vision"
    assert classify_lf_component("language_model.layers.0.linear_attn.in_proj_qkv.weight") == "linear_attention"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selective Asym CPU-first move requires CUDA")
def test_qwen35_asym_cpu_first_move_keeps_frozen_visual_state_on_cpu() -> None:
    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pos_embed = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8, bias=False, dtype=torch.bfloat16))])
            self.merger = nn.Sequential(nn.Linear(8, 8, bias=False, dtype=torch.bfloat16))

    class FakeConditional(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = FakeQwen3_5WholeOffloadModel(hidden_dim=8, intermediate_dim=8, num_layers=1)
            self.visual = FakeVisual()
            self.multi_modal_projector = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.language_model.get_input_embeddings()

        def get_output_embeddings(self) -> nn.Linear:
            return self.language_model.get_output_embeddings()

    model = FakeConditional()
    model, _report = apply_lf_asym_lora(
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

    summary = move_lf_asym_cpu_first_model_to_device(
        model,
        torch.device("cuda"),
        offload_modules="all",
        strict=True,
        keep_frozen_vision_on_cpu=True,
        keep_frozen_projector_on_cpu=True,
    )

    assert summary["kept_cpu_bytes_by_component"]["vision_or_projector"] > 0
    assert model.visual.pos_embed.device.type == "cpu"
    assert model.visual.blocks[0][0].weight.device.type == "cpu"
    assert model.multi_modal_projector.weight.device.type == "cpu"
    lora_params = [param for name, param in model.named_parameters() if "lora_" in name]
    assert lora_params and all(param.device.type == "cuda" for param in lora_params)
    residue = audit_lf_frozen_cuda_residue(model, offload_modules="all", strict=True)
    assert residue.get("vision", 0) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="frozen multimodal CPU staging requires CUDA")
def test_qwen35_frozen_visual_cpu_staging_runs_forward_and_releases_to_cpu() -> None:
    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pos_embed = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
            self.patch_embed = nn.Identity()
            self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)])
            self.merger = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.merger(self.blocks[0](x + self.pos_embed[: x.shape[0]]))

    class FakeConditional(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = FakeQwen3_5WholeOffloadModel(hidden_dim=8, intermediate_dim=8, num_layers=1)
            self.visual = FakeVisual()

        def get_input_embeddings(self) -> nn.Embedding:
            return self.language_model.get_input_embeddings()

        def get_output_embeddings(self) -> nn.Linear:
            return self.language_model.get_output_embeddings()

    model = FakeConditional()
    model, _report = apply_lf_asym_lora(
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
    move_lf_asym_cpu_first_model_to_device(
        model,
        torch.device("cuda"),
        offload_modules="all",
        strict=True,
        keep_frozen_vision_on_cpu=True,
        keep_frozen_projector_on_cpu=True,
    )

    assert getattr(model.visual, "stage_calls", 0) == 0
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    out = model.visual(x)

    assert out.device.type == "cuda"
    assert out.requires_grad is False
    assert model.visual.stage_calls == 1
    assert model.visual.release_calls == 1
    assert model.visual.pos_embed.device.type == "cpu"
    assert model.visual.blocks[0].weight.device.type == "cpu"
    assert model.visual.merger.weight.device.type == "cpu"


def test_qwen35_weight_offload_registers_routed_shared_and_linear_attention_banks() -> None:
    model = FakeQwen3_5DecoderModel(hidden_dim=64, intermediate_dim=64, layer_mixers=("linear_attn",))
    model, _report = apply_lf_asym_lora(
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

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all("lora_" in name or ".lora_A." in name or ".lora_B." in name for name in trainable_names)
    assert any(".mlp.experts.gate_lora_A" in name for name in trainable_names)
    assert any(".mlp.shared_expert.gate_proj.lora_A" in name for name in trainable_names)
    assert any(".mlp.shared_expert.up_proj.lora_A" in name for name in trainable_names)
    assert any(".mlp.shared_expert.down_proj.lora_A" in name for name in trainable_names)
    assert any(".mlp.shared_expert_gate.lora_A" in name for name in trainable_names)
    assert any(".linear_attn.in_proj_qkv.lora_A" in name for name in trainable_names)
    assert any(".linear_attn.out_proj.lora_A" in name for name in trainable_names)
    assert all(".mlp.gate." not in name for name in trainable_names)

    coordinator = LoRAWeightOffloadCoordinator(pin_memory=False, persistence_threshold_numel=0)
    installed = install_lora_weight_offload(model, coordinator)
    summary = coordinator.summary()
    registered_ids = {id(param) for param in coordinator.registered_params}
    registered_names = [name for name, param in model.named_parameters() if id(param) in registered_ids]
    resident_trainable_names = [name for name, param in model.named_parameters() if param.requires_grad and id(param) not in registered_ids]

    assert installed == 24
    assert summary["weight_offload_group_count"] == 4
    assert summary["weight_offload_param_count"] == 24
    assert summary["weight_offload_group_count_by_component"]["routed_experts"] == 1
    assert summary["weight_offload_group_count_by_component"]["shared_experts"] == 2
    assert summary["weight_offload_group_count_by_component"]["linear_attention"] == 1
    assert any(".mlp.experts." in name for name in registered_names)
    assert any(".mlp.shared_expert.gate_proj.lora_A" in name for name in registered_names)
    assert any(".mlp.shared_expert_gate.lora_A" in name for name in registered_names)
    assert any(".linear_attn.in_proj_qkv.lora_A" in name for name in registered_names)
    assert any(".linear_attn.out_proj.lora_A" in name for name in registered_names)
    assert not resident_trainable_names


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


def test_apply_lf_asym_lora_gc_layer_wraps_qwen35_hybrid_decoder_layers() -> None:
    model = FakeQwen3_5DecoderModel(layer_mixers=("linear_attn", "self_attn"))
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
        expert_recompute_policy="gc-layer",
        router_mode="whole",
        strict=True,
    )

    assert report.layer_gc_enabled
    assert report.layer_gc_wrapped == 2
    assert report.layer_gc_modules == ("layers.0", "layers.1")
    assert decoder_checkpoint_module_names(model) == ("layers.0", "layers.1")
    assert is_decoder_checkpoint_wrapper(model.layers[0])
    assert is_decoder_checkpoint_wrapper(model.layers[1])

    model.train()
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    loss = model(x).float().square().mean()
    loss.backward()

    wrapper = getattr(model.layers[0], "_asym_decoder_checkpoint_wrapper")
    assert wrapper.checkpoint_calls == 1
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()


def test_qwen35_hybrid_decoder_layer_activation_offload_wraps_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "1")
    model = FakeQwen3_5DecoderModel(layer_mixers=("linear_attn", "self_attn"))
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["experts"],
        dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="none",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=False,
    )

    assert report.layer_act_offload_enabled
    assert report.layer_act_offload_wrapped == 2
    assert report.layer_act_offload_modules == ("layers.0", "layers.1")
    assert decoder_saved_tensor_offload_module_names(model) == ("layers.0", "layers.1")
    assert is_decoder_saved_tensor_offload_wrapper(model.layers[0])
    assert is_decoder_saved_tensor_offload_wrapper(model.layers[1])
    assert getattr(model, "_asym_layer_act_offload_modules") == ("layers.0", "layers.1")


def test_qwen35_layer_activation_offload_wraps_linear_attention_saved_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "1")
    model = FakeQwen3_5DecoderModel(layer_mixers=("linear_attn", "self_attn"))
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["experts"],
        dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="linear_attention",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=False,
    )

    assert report.layer_act_offload_enabled
    assert report.linear_attention_saved_tensor_offload_wrapped == 1
    assert report.linear_attention_saved_tensor_offload_modules == ("layers.0.linear_attn",)
    assert linear_attention_saved_tensor_offload_module_names(model) == ("layers.0.linear_attn",)
    assert is_linear_attention_saved_tensor_offload_wrapper(model.layers[0].linear_attn)
    assert getattr(model, "_asym_linear_attention_saved_tensor_offload_modules") == ("layers.0.linear_attn",)


def test_qwen35_full_fg_wraps_linear_attention_without_layer_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE", "full-fg")
    monkeypatch.setenv("ASYMM_ATTN_ACT_OFFLOAD", "1")
    model = FakeQwen3_5DecoderModel(layer_mixers=("linear_attn", "self_attn"))
    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["experts"],
        dense_target_modules=[],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="linear_attention",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=False,
    )

    assert report.attention_act_offload_enabled
    assert not report.layer_act_offload_enabled
    assert not report.layer_glue_gc_enabled
    assert report.linear_attention_saved_tensor_offload_wrapped == 1
    assert report.linear_attention_saved_tensor_offload_modules == ("layers.0.linear_attn",)
    assert linear_attention_saved_tensor_offload_module_names(model) == ("layers.0.linear_attn",)
    assert is_linear_attention_saved_tensor_offload_wrapper(model.layers[0].linear_attn)
    assert getattr(model, "_asym_linear_attention_saved_tensor_offload_modules") == ("layers.0.linear_attn",)

    config = _infer_adapter_config(model, {})
    assert config["asym_linear_attention_saved_tensor_offload_modules"] == ["layers.0.linear_attn"]


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
