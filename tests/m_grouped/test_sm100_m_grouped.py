import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import asym_gemm

from asym_gemm.testing import calc_diff, get_arch_major  # noqa: E402
from asym_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch_major() != 10,
    reason="SM100/GB200 required",
)


def _seed() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _dense_offsets(segment_rows: list[int], *, device: torch.device | str = "cuda"):
    offsets: list[int] = []
    experts: list[int] = []
    start = 0
    for expert, rows in enumerate(segment_rows):
        offsets.extend([start, start + rows])
        experts.append(expert)
        start += rows
    experts.append(-1)
    return (
        torch.tensor(offsets, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(experts)], dtype=torch.int32, device=device),
    )


def _offsets_from_m_indices_pairs(m_indices: torch.Tensor, block_m: int = 128):
    m_indices_cpu = m_indices.to("cpu", torch.int32).contiguous()
    values = m_indices_cpu.tolist()
    offsets: list[int] = []
    experts: list[int] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            expert = values[start]
            if expert != -1:
                offsets.extend([
                    (start // block_m) * block_m,
                    ((idx + block_m - 1) // block_m) * block_m,
                ])
                experts.append(expert)
            start = idx
    experts.append(-1)
    return (
        torch.tensor(offsets, dtype=torch.int32, device=m_indices.device),
        torch.tensor(experts, dtype=torch.int32, device=m_indices.device),
        torch.tensor([len(experts)], dtype=torch.int32, device=m_indices.device),
    )


def _assert_grouped_close(out: torch.Tensor, ref: torch.Tensor, lengths: list[int], tol: float) -> None:
    start = 0
    max_diff = 0.0
    for rows in lengths:
        if rows > 0:
            max_diff = max(max_diff, calc_diff(out[start:start + rows], ref[start:start + rows]))
        start += rows
    assert max_diff < tol, f"max grouped diff {max_diff:.5e} >= {tol:.5e}"


def _bf16_contiguous_case(lengths: list[int], n: int, k: int):
    num_groups = len(lengths)
    m = sum(lengths)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    start = 0
    for group, rows in enumerate(lengths):
        ref[start:start + rows] = a[start:start + rows].float().matmul(b[group].float().t()).to(torch.bfloat16)
        start += rows
    return a, b, ref


def _fp8_contiguous_case(lengths: list[int], n: int, k: int):
    a_bf16, b_bf16, ref = _bf16_contiguous_case(lengths, n, k)
    a = per_token_cast_to_fp8(a_bf16, use_ue8m0=True)
    b_values = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    b_scales = []
    for group in range(b_bf16.size(0)):
        b_values[group], scales = per_block_cast_to_fp8(b_bf16[group], use_ue8m0=True)
        b_scales.append(scales)
    return a, (b_values, torch.stack(b_scales)), ref


def _bf16_masked_case(num_groups: int, max_m: int, n: int, k: int, masked_m: torch.Tensor):
    a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    ref = torch.einsum("gmk,gnk->gmn", a.float(), b.float()).to(torch.bfloat16)
    return a, b, ref


def _fp8_masked_case(num_groups: int, max_m: int, n: int, k: int, masked_m: torch.Tensor):
    a_bf16, b_bf16, ref = _bf16_masked_case(num_groups, max_m, n, k, masked_m)
    a_values = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    a_scales = []
    b_values = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    b_scales = []
    for group in range(num_groups):
        a_values[group], scales_a = per_token_cast_to_fp8(a_bf16[group], use_ue8m0=True)
        b_values[group], scales_b = per_block_cast_to_fp8(b_bf16[group], use_ue8m0=True)
        a_scales.append(scales_a)
        b_scales.append(scales_b)
    return (a_values, torch.stack(a_scales)), (b_values, torch.stack(b_scales)), ref


def test_m_grouped_bf16_contiguous_sm100_tensor_list_size() -> None:
    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_contiguous"):
        pytest.skip("BF16 m-grouped contiguous kernel is not exported")

    _seed()
    lengths = [128, 128]
    n, k = 128, 512
    a, b, ref = _bf16_contiguous_case(lengths, n, k)
    d = torch.empty_like(ref)

    offsets, experts, list_size = _dense_offsets(lengths)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b, d, offsets, experts, list_size, compiled_dims="nk"
    )
    torch.cuda.synchronize()
    _assert_grouped_close(d, ref, lengths, tol=1e-3)


def test_m_grouped_bf16_masked_sm100_masked_m() -> None:
    if not hasattr(asym_gemm, "m_grouped_bf16_asym_gemm_nt_masked"):
        pytest.skip("BF16 m-grouped masked kernel is not exported")

    _seed()
    num_groups, max_m, expected_m, n, k = 2, 128, 96, 128, 512
    masked_m = torch.tensor([64, 96], dtype=torch.int32, device="cuda")
    a, b, ref = _bf16_masked_case(num_groups, max_m, n, k, masked_m)
    d = torch.empty_like(ref)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
        a, b, d, masked_m, expected_m, compiled_dims="nk"
    )
    torch.cuda.synchronize()

    max_diff = 0.0
    for group in range(num_groups):
        rows = int(masked_m[group].item())
        if rows > 0:
            max_diff = max(max_diff, calc_diff(d[group, :rows], ref[group, :rows]))
    assert max_diff < 1e-3, f"max masked BF16 diff {max_diff:.5e}"


def test_m_grouped_fp8_contiguous_sm100_tensor_list_size() -> None:
    if not hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_contiguous"):
        pytest.skip("FP8 m-grouped contiguous kernel is not exported")

    _seed()
    lengths = [128, 128]
    n, k = 128, 512
    a, b, ref = _fp8_contiguous_case(lengths, n, k)
    d = torch.empty_like(ref)
    offsets, experts, list_size = _dense_offsets(lengths)
    asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
        a,
        b,
        d,
        offsets,
        experts,
        list_size,
        recipe=(1, 128, 128),
        disable_ue8m0_cast=False,
    )
    torch.cuda.synchronize()
    _assert_grouped_close(d, ref, lengths, tol=2e-2)


def test_m_grouped_fp8_masked_sm100_masked_m() -> None:
    if not hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_masked"):
        pytest.skip("FP8 m-grouped masked kernel is not exported")

    _seed()
    num_groups, max_m, expected_m, n, k = 2, 128, 96, 128, 512
    masked_m = torch.tensor([64, 96], dtype=torch.int32, device="cuda")
    a, b, ref = _fp8_masked_case(num_groups, max_m, n, k, masked_m)
    d = torch.empty_like(ref)
    asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
        a,
        b,
        d,
        masked_m,
        expected_m,
        recipe=(1, 128, 128),
        disable_ue8m0_cast=False,
    )
    torch.cuda.synchronize()

    max_diff = 0.0
    for group in range(num_groups):
        rows = int(masked_m[group].item())
        if rows > 0:
            max_diff = max(max_diff, calc_diff(d[group, :rows], ref[group, :rows]))
    assert max_diff < 2e-2, f"max masked FP8 diff {max_diff:.5e}"


def test_m_grouped_fp4_exports_sm100() -> None:
    missing = [
        name for name in (
            "m_grouped_fp4_asym_gemm_nt_contiguous",
            "m_grouped_fp4_asym_gemm_nt_masked",
        )
        if not hasattr(asym_gemm, name)
    ]
    assert not missing, f"missing SM100 FP4 exports: {missing}"
