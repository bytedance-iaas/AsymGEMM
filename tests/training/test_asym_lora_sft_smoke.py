from pathlib import Path
from importlib import metadata

import pytest
import torch


PACKAGE_MIRRORED_C_BINDINGS = {
    "set_num_sms",
    "get_num_sms",
    "set_tc_util",
    "get_tc_util",
    "set_compile_mode",
    "get_compile_mode",
    "fp8_gemm_nt",
    "k_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_asym_gemm_nt_contiguous",
    "m_grouped_fp8_asym_gemm_nt_masked",
    "m_grouped_fp4_asym_gemm_nt_contiguous",
    "m_grouped_fp4_asym_gemm_nt_masked",
    "m_grouped_bf16_asym_gemm_nt_contiguous",
    "m_grouped_bf16_asym_gemm_nt_masked",
    "einsum",
    "fp8_einsum",
    "transform_sf_into_required_layout",
    "get_mk_alignment_for_contiguous_layout",
}

LEGACY_ALIASES = {
    "fp8_m_grouped_asym_gemm_nt_masked": "m_grouped_fp8_asym_gemm_nt_masked",
    "fp8_m_grouped_gemm_nt_masked": "m_grouped_fp8_asym_gemm_nt_masked",
    "bf16_m_grouped_asym_gemm_nt_masked": "m_grouped_bf16_asym_gemm_nt_masked",
    "bf16_m_grouped_gemm_nt_masked": "m_grouped_bf16_asym_gemm_nt_masked",
}

BF16_M_GROUPED_CONTIGUOUS = "m_grouped_bf16_asym_gemm_nt_contiguous"


def _public_names(module):
    return {name for name in dir(module) if not name.startswith("_")}


def _require_c_extension(asym_gemm):
    if hasattr(asym_gemm, "_C"):
        return asym_gemm._C
    if torch.cuda.is_available():
        pytest.fail("CUDA is available, but asym_gemm._C is not importable")
    pytest.skip("asym_gemm._C is unavailable; skipping extension binding checks")


def _cuda_h200_sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA device: {name}, capability=sm_{capability[0]}{capability[1]}")
    if capability != (9, 0) or "H200" not in name:
        pytest.skip(f"H200/SM90 not present on this runner: {name}, capability={capability}")
    return name, capability


def test_import_and_version() -> None:
    import asym_gemm

    assert isinstance(asym_gemm.__version__, str)
    assert len(asym_gemm.__version__) > 0
    assert asym_gemm.__version__ == metadata.version("asym_gemm")


def test_expected_h200_forward_exports() -> None:
    import asym_gemm

    _require_c_extension(asym_gemm)
    required = [
        "m_grouped_bf16_asym_gemm_nt_contiguous",
        "m_grouped_bf16_asym_gemm_nt_masked",
    ]
    missing = [name for name in required if not hasattr(asym_gemm, name)]
    assert not missing, f"missing expected H200 BF16 exports: {missing}"


def test_c_extension_bindings_match_package_exports(record_property) -> None:
    import asym_gemm

    c_extension = _require_c_extension(asym_gemm)
    c_names = _public_names(c_extension)
    package_names = _public_names(asym_gemm)
    mirrored = sorted(PACKAGE_MIRRORED_C_BINDINGS & c_names)
    missing_mirrors = [
        name
        for name in mirrored
        if not hasattr(asym_gemm, name)
        or getattr(asym_gemm, name) is not getattr(c_extension, name)
    ]

    record_property("c_extension_exports", ",".join(sorted(c_names)))
    record_property("package_mirrored_c_exports", ",".join(mirrored))
    record_property("c_extension_only_exports", ",".join(sorted(c_names - package_names)))
    print(f"_C exports: {sorted(c_names)}")
    print(f"Package mirrors from _C: {mirrored}")
    print(f"_C-only exports: {sorted(c_names - package_names)}")

    assert not missing_mirrors

    for alias_name, target_name in LEGACY_ALIASES.items():
        if hasattr(c_extension, target_name):
            assert getattr(asym_gemm, alias_name) is getattr(asym_gemm, target_name)
        else:
            assert hasattr(asym_gemm, alias_name)


def test_hardware_scope_reports_h200_when_cuda_is_available() -> None:
    name, capability = _cuda_h200_sm90_or_skip()

    assert capability == (9, 0)
    assert "H200" in name


def test_sm90_bf16_m_grouped_contiguous_forward_smoke() -> None:
    _cuda_h200_sm90_or_skip()

    import asym_gemm

    _require_c_extension(asym_gemm)
    assert hasattr(asym_gemm, BF16_M_GROUPED_CONTIGUOUS), (
        f"{BF16_M_GROUPED_CONTIGUOUS} is required for the H200 BF16 smoke"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    m = 64
    n = 64
    k = 512
    num_groups = 1

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_gpu = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    b_host = b_gpu.detach().cpu().pin_memory()
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    experts = torch.tensor([0, -1], device="cuda", dtype=torch.int32)

    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a,
        b_host,
        d,
        offsets,
        experts,
        2,
        "nk",
    )
    torch.cuda.synchronize()

    expected = a @ b_gpu[0].t()
    torch.testing.assert_close(d, expected, rtol=0, atol=0)


def test_lora_sft_agent_notes_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    instructions = root / "agent" / "instructions.md"
    action_items = root / "agent" / "action_items.md"

    assert instructions.exists()
    assert action_items.exists()
    assert "LoRA/SFT Precision Integration Plan" in instructions.read_text(encoding="utf-8")
    assert "Unified Toy/HF Asym LoRA Integration" in action_items.read_text(encoding="utf-8")
