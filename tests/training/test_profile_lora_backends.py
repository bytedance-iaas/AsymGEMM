from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from asym_gemm.training.kt_moe import KTBackendUnavailable, _import_kt_moe_wrapper


ROOT = Path(__file__).resolve().parents[2]
PROFILE_LORA_PATH = ROOT / "scripts" / "profile_lora.py"
PROFILE_LORA_DRIVER_PATH = ROOT / "scripts" / "profile_lora_driver.py"


def _load_profile_lora_module():
    spec = importlib.util.spec_from_file_location("profile_lora_under_test", PROFILE_LORA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_profile_lora_driver_module():
    spec = importlib.util.spec_from_file_location("profile_lora_driver_under_test", PROFILE_LORA_DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_lora_public_backends_are_canonical() -> None:
    profile_lora = _load_profile_lora_module()

    assert profile_lora.BACKEND_CHOICES == ("torch", "asym", "kt")


def test_profile_lora_public_workload_names_are_blockwise_for_moe() -> None:
    profile_lora = _load_profile_lora_module()

    assert "dense_14b" in profile_lora.WORKLOAD_CHOICES
    assert "moe-604m-a75m" in profile_lora.WORKLOAD_CHOICES
    assert "moe-604m-a38m" in profile_lora.WORKLOAD_CHOICES
    legacy_names = {"moe" + "_3b", "qwen3" + "_30b" + "_a3b"}
    assert not legacy_names.intersection(profile_lora.WORKLOAD_CHOICES)
    assert profile_lora.KT_MOE_WORKLOADS == ("moe", "moe-604m-a75m", "moe-604m-a38m")


def test_profile_lora_driver_supports_workload_layer_specs() -> None:
    driver = _load_profile_lora_driver_module()

    specs = driver._expand_workloads(["moe-604m-a75m|2", "qwen|4"])

    assert [(spec.name, spec.profile_layers, spec.label) for spec in specs] == [
        ("moe-604m-a75m", 2, "moe-604m-a75m-l2"),
        ("dense_14b", 4, "dense_14b-l4"),
        ("moe-604m-a38m", 4, "moe-604m-a38m-l4"),
    ]


def test_kt_backend_is_restricted_to_moe_workloads() -> None:
    profile_lora = _load_profile_lora_module()

    with pytest.raises(ValueError, match="backend=kt is only implemented for MoE LoRA SFT workloads"):
        profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="mlp", lora_dtype="bf16"))

    profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="moe", lora_dtype="bf16"))


def test_kt_backend_rejects_non_bf16_lora_dtype() -> None:
    profile_lora = _load_profile_lora_module()

    with pytest.raises(ValueError, match="backend=kt currently supports BF16 LoRA buffers only"):
        profile_lora.validate_backend_workload(argparse.Namespace(backend="kt", workload="moe", lora_dtype="fp16"))


def test_profile_lora_cli_rejects_kt_for_non_moe_workload(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROFILE_LORA_PATH),
            "--workload",
            "mlp",
            "--backend",
            "kt",
            "--device",
            "cpu",
            "--warmup-steps",
            "0",
            "--measure-steps",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "backend=kt is only implemented for MoE LoRA SFT workloads" in result.stderr


def test_kt_moe_wrapper_import_reports_clear_availability() -> None:
    try:
        wrapper_cls = _import_kt_moe_wrapper()
    except KTBackendUnavailable as exc:
        assert "backend=kt requires kt-kernel with SFT AMX support" in str(exc)
    else:
        assert wrapper_cls.__name__ == "KTMoEWrapper"
