from __future__ import annotations

import pytest
import torch
from torch import nn

from asym_gemm.training.lora import AsymLoRALinear, add_asym_lora, freeze_non_lora_params, get_lora_state_dict


class ProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=True)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.norm(self.q_proj(x)))


def test_add_asym_lora_with_exact_target_list() -> None:
    model = ProjectionBlock()

    report = add_asym_lora(
        model,
        rank=2,
        alpha=4.0,
        target_modules=["q_proj", "v_proj"],
        backend="torch",
        precision="bf16",
    )

    assert report.replaced_module_count == 2
    assert report.matched_module_names == ["q_proj", "v_proj"]
    assert isinstance(model.q_proj, AsymLoRALinear)
    assert isinstance(model.v_proj, AsymLoRALinear)
    assert model.q_proj.lora_A["default"].weight.dtype == torch.bfloat16
    assert model.q_proj.lora_B["default"].weight.dtype == torch.bfloat16
    assert report.trainable_adapter_parameter_count == (2 * 4 + 4 * 2) + (2 * 4 + 2 * 2)

    out = model(torch.randn(3, 4))
    assert tuple(out.shape) == (3, 2)


def test_add_asym_lora_accepts_explicit_lora_dtype() -> None:
    model = ProjectionBlock()

    add_asym_lora(
        model,
        rank=2,
        alpha=4.0,
        target_modules=["q_proj"],
        backend="torch",
        precision="bf16",
        lora_dtype=torch.float32,
    )

    assert isinstance(model.q_proj, AsymLoRALinear)
    assert model.q_proj.lora_A["default"].weight.dtype == torch.float32
    assert model.q_proj.lora_B["default"].weight.dtype == torch.float32


def test_add_asym_lora_with_regex_targeting() -> None:
    model = ProjectionBlock()

    report = add_asym_lora(
        model,
        rank=2,
        alpha=4.0,
        target_modules=r".*_proj$",
        backend="torch",
        precision="bf16",
    )

    assert report.matched_module_names == ["q_proj", "v_proj"]
    assert isinstance(model.q_proj, AsymLoRALinear)
    assert isinstance(model.v_proj, AsymLoRALinear)


def test_add_asym_lora_defaults_to_all_preset() -> None:
    model = ProjectionBlock()

    report = add_asym_lora(model, rank=2, alpha=4.0, backend="torch", precision="bf16")

    assert report.matched_module_names == ["q_proj", "v_proj"]


def test_add_asym_lora_strict_no_match_raises() -> None:
    model = ProjectionBlock()

    with pytest.raises(ValueError, match="no target modules"):
        add_asym_lora(
            model,
            rank=2,
            alpha=4.0,
            target_modules=["missing_proj"],
            backend="torch",
            precision="bf16",
        )


def test_add_asym_lora_strict_non_linear_match_raises() -> None:
    model = ProjectionBlock()

    with pytest.raises(TypeError, match="nn.Linear"):
        add_asym_lora(
            model,
            rank=2,
            alpha=4.0,
            target_modules=["norm"],
            backend="torch",
            precision="bf16",
        )


def test_freeze_non_lora_params_only_leaves_adapter_trainable() -> None:
    model = ProjectionBlock()
    add_asym_lora(
        model,
        rank=2,
        alpha=4.0,
        target_modules=["q_proj"],
        backend="torch",
        precision="bf16",
    )

    freeze_non_lora_params(model)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names == ["q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight"]


def test_lora_state_dict_filters_base_weights() -> None:
    model = ProjectionBlock()
    add_asym_lora(
        model,
        rank=2,
        alpha=4.0,
        target_modules=["q_proj"],
        backend="torch",
        precision="bf16",
    )

    state = get_lora_state_dict(model)

    assert list(state) == ["q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight"]
    assert all("base_layer" not in name for name in state)
