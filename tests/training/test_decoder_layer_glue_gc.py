from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from asym_gemm.integrations.lf import apply_lf_asym_lora
from asym_gemm.training.decoder_layer_glue_gc import (
    decoder_layer_glue_gc_module_names,
    install_decoder_layer_glue_gc,
    is_decoder_layer_glue_gc_wrapper,
)
from asym_gemm.training.linear_attention_activation_offload import (
    install_linear_attention_saved_tensor_offload,
    is_linear_attention_saved_tensor_offload_wrapper,
    linear_attention_saved_tensor_offload_module_names,
)


class CountingAttention(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> tuple[torch.Tensor, None]:
        self.calls += 1
        mixed = self.q_proj(hidden_states) + self.k_proj(hidden_states) + self.v_proj(hidden_states)
        return self.o_proj(mixed), None


class CountingMLP(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.down_proj(torch.nn.functional.silu(self.up_proj(hidden_states)))


class FakeQwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.self_attn = CountingAttention(hidden_dim)
        self.mlp = CountingMLP(hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class FakeQwen3DecoderModel(nn.Module):
    def __init__(self, hidden_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeQwen3DecoderLayer(hidden_dim) for _ in range(num_layers))

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states


class FakeLinearAttention(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.calls = 0
        self.position_ids = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: object = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cache_params, cache_position, attention_mask
        self.calls += 1
        self.position_ids = position_ids
        return self.proj(hidden_states)


class FakeLinearAttentionWithoutCachePosition(FakeLinearAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: object = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cache_params, attention_mask
        self.calls += 1
        return self.proj(hidden_states)


class FakeQwen35DecoderLayer(FakeQwen3DecoderLayer):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__(hidden_dim)
        self.layer_type = "linear_attention"
        self.linear_attn = FakeLinearAttention(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=kwargs.get("past_key_values"),
            cache_position=kwargs.get("cache_position"),
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class FakeQwen35DecoderLayerWithoutCachePosition(FakeQwen35DecoderLayer):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__(hidden_dim)
        self.linear_attn = FakeLinearAttentionWithoutCachePosition(hidden_dim)


class FakeLlama4FeedForward(CountingMLP):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = super().forward(hidden_states)
        return out.reshape(-1, out.shape[-1]), None


class FakeLlama4DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.self_attn = CountingAttention(hidden_dim)
        self.feed_forward = FakeLlama4FeedForward(hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.is_moe_layer = True

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.feed_forward(hidden_states)
        return residual + hidden_states.view(residual.shape)


def test_qwen3_glue_gc_matches_original_forward_backward_and_does_not_recompute_core_modules() -> None:
    torch.manual_seed(0)
    original = FakeQwen3DecoderLayer()
    wrapped = copy.deepcopy(original)
    install_decoder_layer_glue_gc(wrapped)
    wrapped.train()
    original.train()

    x_ref = torch.randn(2, 4, 8, requires_grad=True)
    x_wrapped = x_ref.detach().clone().requires_grad_(True)

    loss_ref = original(x_ref).float().square().mean()
    loss_wrapped = wrapped(x_wrapped).float().square().mean()
    loss_ref.backward()
    loss_wrapped.backward()

    assert torch.allclose(loss_wrapped, loss_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_wrapped.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert wrapped.self_attn.calls == 1
    assert wrapped.mlp.calls == 1
    wrapper = wrapped._asym_decoder_layer_glue_gc_wrapper
    assert wrapper.checkpoint_norm_calls == 2


def test_qwen35_linear_attention_signature_receives_position_ids() -> None:
    layer = FakeQwen35DecoderLayer()
    install_decoder_layer_glue_gc(layer)
    layer.train()

    position_ids = torch.arange(4).view(1, 4)
    x = torch.randn(2, 4, 8, requires_grad=True)
    loss = layer(x, position_ids=position_ids).float().sum()
    loss.backward()

    assert layer.linear_attn.calls == 1
    assert layer.linear_attn.position_ids is position_ids
    assert layer.self_attn.calls == 0
    assert x.grad is not None


def test_qwen35_linear_attention_wrapper_filters_kwargs_against_original_signature() -> None:
    layer = FakeQwen35DecoderLayerWithoutCachePosition()
    install_linear_attention_saved_tensor_offload(layer.linear_attn, min_bytes=0)
    install_decoder_layer_glue_gc(layer)
    layer.train()

    x = torch.randn(2, 4, 8, requires_grad=True)
    cache_position = torch.arange(4)
    loss = layer(x, cache_position=cache_position).float().sum()
    loss.backward()

    assert layer.linear_attn.calls == 1
    assert x.grad is not None


def test_llama4_feed_forward_tuple_is_reshaped_to_residual_shape() -> None:
    layer = FakeLlama4DecoderLayer()
    install_decoder_layer_glue_gc(layer)
    layer.train()

    x = torch.randn(2, 4, 8, requires_grad=True)
    out = layer(x)
    out.float().sum().backward()

    assert out.shape == x.shape
    assert layer.self_attn.calls == 1
    assert layer.feed_forward.calls == 1


def test_apply_lf_asym_lora_installs_layer_glue_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_LAYER_GC", "1")
    model = FakeQwen3DecoderModel(num_layers=2)

    model, report = apply_lf_asym_lora(
        model,
        raw_lora_target=["q_proj"],
        dense_target_modules=["q_proj"],
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="none",
        expert_recompute_policy="none",
        router_mode="hf",
        strict=False,
    )

    assert report.layer_glue_gc_enabled
    assert report.layer_glue_gc_wrapped == 2
    assert report.layer_glue_gc_modules == ("layers.0", "layers.1")
    assert decoder_layer_glue_gc_module_names(model) == ("layers.0", "layers.1")
    assert is_decoder_layer_glue_gc_wrapper(model.layers[0])


def test_apply_lf_asym_lora_layer_glue_gc_wraps_qwen35_linear_attention_saved_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASYMM_LAYER_GC", "1")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([FakeQwen35DecoderLayer()])

    model, report = apply_lf_asym_lora(
        Model(),
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

    assert report.layer_glue_gc_enabled
    assert report.layer_glue_gc_wrapped == 1
    assert report.linear_attention_saved_tensor_offload_wrapped == 1
    assert report.linear_attention_saved_tensor_offload_modules == ("layers.0.linear_attn",)
    assert linear_attention_saved_tensor_offload_module_names(model) == ("layers.0.linear_attn",)
    assert is_linear_attention_saved_tensor_offload_wrapper(model.layers[0].linear_attn)


def test_apply_lf_asym_lora_rejects_layer_glue_gc_with_layer_act(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASYMM_LAYER_GC", "1")
    monkeypatch.setenv("ASYMM_LAYER_ACT_OFFLOAD", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        apply_lf_asym_lora(
            FakeQwen3DecoderModel(num_layers=1),
            raw_lora_target=["q_proj"],
            dense_target_modules=["q_proj"],
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            backend="asym",
            precision="bf16",
            offload_modules="none",
            expert_recompute_policy="none",
            router_mode="hf",
            strict=False,
        )
