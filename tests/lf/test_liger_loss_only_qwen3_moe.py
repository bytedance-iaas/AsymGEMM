from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
LF_SRC = ROOT.parent / "LlamaFactory" / "src"
if str(LF_SRC) not in sys.path:
    sys.path.insert(0, str(LF_SRC))


def test_build_liger_loss_only_kwargs_disables_non_loss_patches() -> None:
    from llamafactory.model.model_utils import liger_kernel

    def fake_apply(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model=None,
    ) -> None:
        return None

    assert liger_kernel._build_liger_loss_only_kwargs(fake_apply) == {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": False,
        "swiglu": False,
    }


def test_apply_liger_kernel_uses_loss_only_for_qwen3_moe(monkeypatch) -> None:
    from llamafactory.model.model_utils import liger_kernel

    calls: list[dict[str, bool]] = []

    def fake_apply(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model=None,
    ) -> None:
        calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
            }
        )

    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda model_type: fake_apply)

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type="qwen3_moe"),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=False,
    )

    assert calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
        }
    ]


def test_apply_liger_kernel_skips_when_logits_are_required(monkeypatch) -> None:
    from llamafactory.model.model_utils import liger_kernel

    calls: list[dict[str, bool]] = []
    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda model_type: lambda **kwargs: calls.append(kwargs))

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type="qwen3_moe"),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=True,
    )

    assert calls == []


def test_apply_liger_kernel_skips_unvalidated_model_types(monkeypatch) -> None:
    from llamafactory.model.model_utils import liger_kernel

    calls: list[dict[str, bool]] = []
    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda model_type: lambda **kwargs: calls.append(kwargs))

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type="unknown_model"),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=False,
    )

    assert calls == []


def test_local_liger_qwen3_moe_signature_still_has_expected_loss_axis() -> None:
    import inspect
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

    signature = inspect.signature(apply_liger_kernel_to_qwen3_moe)
    assert {"rope", "cross_entropy", "fused_linear_cross_entropy", "rms_norm", "swiglu", "model"} <= set(
        signature.parameters
    )


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text"])
def test_apply_liger_kernel_uses_loss_only_for_qwen3_5_dense(monkeypatch, model_type) -> None:
    from llamafactory.model.model_utils import liger_kernel

    calls: list[dict[str, bool]] = []

    def fake_apply(
        rope: bool = False,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model=None,
    ) -> None:
        calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
            }
        )

    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda mt: fake_apply)

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type=model_type),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=False,
    )

    assert calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
        }
    ]


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_apply_liger_kernel_uses_loss_only_for_qwen3_5_moe(monkeypatch, model_type) -> None:
    from llamafactory.model.model_utils import liger_kernel

    calls: list[dict[str, bool]] = []

    def fake_apply(
        rope: bool = False,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model=None,
    ) -> None:
        calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
            }
        )

    monkeypatch.setattr(liger_kernel, "_resolve_liger_apply_fn", lambda mt: fake_apply)

    liger_kernel.apply_liger_kernel(
        SimpleNamespace(model_type=model_type),
        SimpleNamespace(enable_liger_kernel=True),
        is_trainable=True,
        require_logits=False,
    )

    assert calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
        }
    ]


def test_qwen3_5_moe_is_in_supported_model_types() -> None:
    from llamafactory.model.model_utils import liger_kernel

    assert {"qwen3_5", "qwen3_5_text"} <= liger_kernel._LOSS_ONLY_SUPPORTED_MODEL_TYPES
    assert liger_kernel._resolve_liger_apply_fn("qwen3_5").__name__ == "apply_liger_kernel_to_qwen3_5"
    assert {"qwen3_5_moe", "qwen3_5_moe_text"} <= liger_kernel._LOSS_ONLY_SUPPORTED_MODEL_TYPES
    assert liger_kernel._resolve_liger_apply_fn("qwen3_5_moe").__name__ == "apply_liger_kernel_to_qwen3_5_moe"


def test_local_liger_qwen3_5_signature_still_has_expected_loss_axis() -> None:
    import inspect
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    signature = inspect.signature(apply_liger_kernel_to_qwen3_5)
    assert {"rope", "cross_entropy", "fused_linear_cross_entropy", "rms_norm", "swiglu", "model"} <= set(
        signature.parameters
    )


def test_local_liger_qwen3_5_moe_signature_still_has_expected_loss_axis() -> None:
    import inspect
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe

    signature = inspect.signature(apply_liger_kernel_to_qwen3_5_moe)
    assert {"rope", "cross_entropy", "fused_linear_cross_entropy", "rms_norm", "swiglu", "model"} <= set(
        signature.parameters
    )
