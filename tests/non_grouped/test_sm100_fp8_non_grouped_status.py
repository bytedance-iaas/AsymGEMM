import sys
from pathlib import Path

import pytest
import torch

import asym_gemm

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

from asym_gemm.testing import get_arch_major  # noqa: E402


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch_major() != 10,
    reason="SM100/GB200 required",
)
def test_non_grouped_fp8_sm100_not_implemented() -> None:
    assert hasattr(asym_gemm, "fp8_gemm_nt")
    pytest.skip("fp8_gemm_nt is currently wired to the SM90 implementation only")
