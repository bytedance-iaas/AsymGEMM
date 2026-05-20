import sys
from pathlib import Path

import pytest
import torch

import asym_gemm

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))

from asym_gemm.testing import calc_diff, get_arch_major  # noqa: E402
from generators import MajorTypeAB, generate_k_grouped_contiguous  # noqa: E402


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90/H20 required")
def test_k_grouped_fp8_gemm_nt_contiguous_sm90():
    if not hasattr(asym_gemm, "k_grouped_fp8_gemm_nt_contiguous"):
        pytest.skip("k_grouped_fp8_gemm_nt_contiguous is not exported")

    num_groups = 2
    m, n = 128, 256
    ks = [256, 384]
    k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
        num_groups=num_groups,
        m=m,
        n=n,
        major_a=MajorTypeAB.KMajor,
        major_b=MajorTypeAB.KMajor,
        ks=ks,
    )
    ks_tensor = torch.tensor(ks, dtype=torch.int32, device="cuda")

    asym_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c)
    diff = calc_diff(d, ref_d)
    print(f"k_grouped_fp8_gemm_nt_contiguous k={k} diff={diff:.5e}")
    assert diff < 0.001


if __name__ == "__main__":
    torch.manual_seed(0)
    test_k_grouped_fp8_gemm_nt_contiguous_sm90()
