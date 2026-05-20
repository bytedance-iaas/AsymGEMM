import sys
from pathlib import Path

import pytest
import torch

import asym_gemm

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

from asym_gemm.testing import calc_diff, get_arch_major  # noqa: E402
from generators import KernelType, MajorTypeAB, generate_normal  # noqa: E402


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90/H20 required")
def test_fp8_gemm_nt_sm90():
    if not hasattr(asym_gemm, "fp8_gemm_nt"):
        pytest.skip("fp8_gemm_nt is not exported")

    a, b, c, d, ref_d = generate_normal(
        m=128,
        n=256,
        k=512,
        major_a=MajorTypeAB.KMajor,
        major_b=MajorTypeAB.KMajor,
        accumulate=True,
        out_dtype=torch.float,
        kernel_type=KernelType.Kernel1D1D,
    )

    asym_gemm.fp8_gemm_nt(a, b, d, c=c, recipe=(1, 1, 128))
    diff = calc_diff(d, ref_d)
    print(f"fp8_gemm_nt diff={diff:.5e}")
    assert diff < 0.001


if __name__ == "__main__":
    torch.manual_seed(0)
    test_fp8_gemm_nt_sm90()
