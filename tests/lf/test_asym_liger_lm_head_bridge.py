from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast

from asym_gemm.integrations import liger_loss
from asym_gemm.integrations.liger_loss import (
    asym_liger_lm_head_bridge_metadata,
    asym_qwen3_5_moe_conditional_lce_forward,
    asym_qwen3_moe_lce_forward,
    install_asym_liger_qwen3_5_moe_loss_bridge,
    install_asym_liger_qwen3_moe_loss_bridge,
)
from asym_gemm.training.frozen_linear import AsymFrozenLinear


class TinyBackbone(nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **kwargs) -> MoeModelOutputWithPast:
        return MoeModelOutputWithPast(
            last_hidden_state=self.hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            router_logits=None,
        )


class TinyQwen3MoeForCausalLM(nn.Module):
    def __init__(self, lm_head: nn.Module, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            use_return_dict=True,
            hidden_size=hidden_states.shape[-1],
        )
        self.model = TinyBackbone(hidden_states)
        self.lm_head = lm_head
        self.vocab_size = lm_head.out_features
        self.num_experts = 1
        self.num_experts_per_tok = 1
        self.router_aux_loss_coef = 0.0


def test_asym_frozen_linear_liger_weight_stages_without_changing_weight_property() -> None:
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
    wrapped = AsymFrozenLinear(weight, bias=None, pin_memory=False)

    staged = wrapped.asym_liger_lm_head_weight(device=torch.device("cpu"), dtype=torch.float32)

    assert wrapped.weight.device.type == "cpu"
    assert wrapped.weight.dtype == torch.bfloat16
    assert staged.device.type == "cpu"
    assert staged.dtype == torch.float32
    assert staged.shape == (5, 4)
    assert not hasattr(wrapped, "_cached_liger_lm_head_weight")


def test_asym_frozen_linear_liger_weight_rejects_bias() -> None:
    weight = torch.randn(5, 4, dtype=torch.bfloat16)
    bias = torch.randn(5, dtype=torch.bfloat16)
    wrapped = AsymFrozenLinear(weight, bias=bias, pin_memory=False)

    with pytest.raises(RuntimeError, match="bias-free"):
        wrapped.asym_liger_lm_head_weight(device=torch.device("cpu"), dtype=torch.bfloat16)


def test_install_bridge_patches_instance_only_and_uses_asym_staged_weight(monkeypatch) -> None:
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    lm_head = AsymFrozenLinear(torch.randn(5, 4, dtype=torch.bfloat16), bias=None, pin_memory=False)
    model = TinyQwen3MoeForCausalLM(lm_head, hidden_states)
    model.train()

    calls: list[dict[str, object]] = []

    def fake_liger_loss(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].float().sum()

    monkeypatch.setattr(liger_loss, "LigerForCausalLMLoss", fake_liger_loss)

    assert install_asym_liger_qwen3_moe_loss_bridge(model, strict=True) is True
    assert model.forward.__func__ is asym_qwen3_moe_lce_forward
    assert TinyQwen3MoeForCausalLM.forward is not asym_qwen3_moe_lce_forward

    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output = model(labels=labels)

    assert output.loss is not None
    assert len(calls) == 1
    assert calls[0]["lm_head_weight"].device.type == "cpu"
    assert calls[0]["lm_head_weight"].dtype == torch.bfloat16
    assert calls[0]["hidden_size"] == 4
    assert asym_liger_lm_head_bridge_metadata(model) == {
        "enabled": True,
        "weight_source": "asym_host_staged",
        "staged_bytes": lm_head.cpu_resident_base_weight_bytes,
        "lm_head_type": "AsymFrozenLinear",
        "model_type": "qwen3_moe",
        "bridge_kind": "causal_lm",
    }


def test_install_bridge_rejects_trainable_or_biased_lm_head() -> None:
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    trainable = nn.Linear(4, 5, bias=False, dtype=torch.bfloat16)
    biased = nn.Linear(4, 5, bias=True, dtype=torch.bfloat16)
    biased.weight.requires_grad_(False)
    biased.bias.requires_grad_(False)

    with pytest.raises(RuntimeError, match="frozen lm_head"):
        install_asym_liger_qwen3_moe_loss_bridge(TinyQwen3MoeForCausalLM(trainable, hidden_states), strict=True)

    with pytest.raises(RuntimeError, match="bias-free"):
        install_asym_liger_qwen3_moe_loss_bridge(TinyQwen3MoeForCausalLM(biased, hidden_states), strict=True)


def test_install_bridge_skips_unvalidated_model_when_not_strict() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(model_type="llama4")

    assert install_asym_liger_qwen3_moe_loss_bridge(model, strict=False) is False


# ---------------------------------------------------------------------------
# Qwen3.5-MoE conditional-generation bridge (Qwen3_5MoeForConditionalGeneration).
# ---------------------------------------------------------------------------
class TinyCondOutput:
    def __init__(self, hidden_states: torch.Tensor) -> None:
        self._hidden_states = hidden_states
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
        self.rope_deltas = None
        self.router_logits = None

    def __getitem__(self, index: int) -> torch.Tensor:
        assert index == 0
        return self._hidden_states


class TinyCondBackbone(nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, **kwargs) -> TinyCondOutput:
        return TinyCondOutput(self.hidden_states)


class TinyQwen3_5MoeForConditionalGeneration(nn.Module):
    def __init__(self, lm_head: nn.Module, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5_moe",
            use_return_dict=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            text_config=SimpleNamespace(
                hidden_size=hidden_states.shape[-1],
                vocab_size=lm_head.out_features,
                num_experts=1,
                num_experts_per_tok=1,
                router_aux_loss_coef=0.0,
            ),
        )
        self.model = TinyCondBackbone(hidden_states)
        self.lm_head = lm_head

    def loss_function(self, **kwargs):  # pragma: no cover - must not run in training skip_logits path
        raise AssertionError("loss_function must not be called when fused CE fires")


def test_install_qwen3_5_moe_conditional_bridge_stages_weight(monkeypatch) -> None:
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    lm_head = AsymFrozenLinear(torch.randn(5, 4, dtype=torch.bfloat16), bias=None, pin_memory=False)
    model = TinyQwen3_5MoeForConditionalGeneration(lm_head, hidden_states)
    model.train()

    calls: list[dict[str, object]] = []

    def fake_liger_loss(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].float().sum()

    monkeypatch.setattr(liger_loss, "LigerForCausalLMLoss", fake_liger_loss)

    assert install_asym_liger_qwen3_5_moe_loss_bridge(model, strict=True) is True
    assert model.forward.__func__ is asym_qwen3_5_moe_conditional_lce_forward
    assert TinyQwen3_5MoeForConditionalGeneration.forward is not asym_qwen3_5_moe_conditional_lce_forward

    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output = model(labels=labels)

    assert output.loss is not None
    assert len(calls) == 1
    # fused CE fires (no logits materialized) and the lm_head weight is Asym-staged
    assert calls[0]["lm_head_weight"].device.type == "cpu"
    assert calls[0]["lm_head_weight"].dtype == torch.bfloat16
    assert calls[0]["hidden_size"] == 4

    metadata = asym_liger_lm_head_bridge_metadata(model)
    assert metadata["enabled"] is True
    assert metadata["weight_source"] == "asym_host_staged"
    assert metadata["bridge_kind"] == "conditional_generation"
    assert metadata["model_type"] == "qwen3_5_moe"
    assert metadata["lm_head_type"] == "AsymFrozenLinear"


def test_install_qwen3_5_moe_conditional_bridge_rejects_trainable_lm_head() -> None:
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    trainable = nn.Linear(4, 5, bias=False, dtype=torch.bfloat16)
    model = TinyQwen3_5MoeForConditionalGeneration(trainable, hidden_states)

    with pytest.raises(RuntimeError, match="frozen lm_head"):
        install_asym_liger_qwen3_5_moe_loss_bridge(model, strict=True)
