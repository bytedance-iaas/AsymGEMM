"""SM90 deep-pattern INT8 grouped GEMM tests (hybridGEMM.md Phase A).

Parity of `m_grouped_int8_gemm_nt_contiguous` (persistent M-outer kernel for
HBM-resident expert weights) against the float reference and the existing
asym K-outer kernel on identical grouped inputs, plus a quick bench at the
hybridGEMM.md §10 shapes.

Runs two ways (AsymGEMM convention):
    python tests/test_sm90_int8_deep.py       # scripts/test.sh path; skips via exit 0
    pytest tests/test_sm90_int8_deep.py -s    # skips via pytest marks
"""
import random
import sys

import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major
from asym_gemm.utils.math import ceil_div, per_channel_cast_to_int8, per_token_cast_to_int8

from test_sm90_int8 import (
    GRAN_K,
    DIFF_TOL,
    build_offsets_experts_from_m_indices_pairs,
    ref_2d,
)


def _deep_fn():
    fn = getattr(asym_gemm, "m_grouped_int8_gemm_nt_contiguous", None)
    if fn is None:
        pytest.skip("m_grouped_int8_gemm_nt_contiguous not exported by this build")
    return fn


def _asym_fn():
    return getattr(asym_gemm, "m_grouped_int8_asym_gemm_nt_contiguous", None)


def _build_case(num_groups, m_per_group, n, k, permuted=False):
    """Quantized grouped inputs + layout + float reference.

    With `permuted=True` the expert ids of the segments are a shuffled subset
    (exercising expert_id != segment_id indexing into B/SFB, as in a real
    hybrid split where only SOME experts are HBM-resident).
    """
    m = num_groups * m_per_group
    a = torch.randn((m, k), device="cuda", dtype=torch.float32) / k ** 0.25
    total_experts = num_groups * 2 if permuted else num_groups
    b = torch.randn((total_experts, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

    a_q, sfa = per_token_cast_to_int8(a)
    b_q = torch.empty_like(b, dtype=torch.int8)
    sfb = torch.empty((total_experts, n, ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
    for g in range(total_experts):
        b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

    ids = list(range(total_experts))
    random.shuffle(ids)
    seg_expert = ids[:num_groups]
    m_indices = torch.tensor(seg_expert, device="cuda").repeat_interleave(m_per_group).to(torch.int32)
    # block_m=64: the deep kernel's launch granularity (segments here are
    # 64-row-aligned by construction; decode-like cases use 64-row segments).
    offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(m_indices, block_m=64)

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    for g in set(seg_expert):
        rows = (m_indices == g)
        ref[rows] = ref_2d(a_q[rows], sfa[rows], b_q[g], sfb[g], k)
    return a_q, sfa, b_q, sfb, offsets, experts, list_size, ref


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_m_grouped_int8_deep_contiguous_sm90() -> None:
    fn = _deep_fn()
    random.seed(0)
    torch.manual_seed(0)
    print("Testing SM90 m-grouped contiguous deep INT8 GEMM:")
    recipe = (1, GRAN_K, GRAN_K)

    cases = [
        # (num_groups, m_per_group, n, k, permuted)
        (2, 256, 256, 512, False),
        (4, 256, 128, 256, False),
        (8, 64, 256, 512, False),      # decode-like: one m-block per segment
        (4, 512, 2048, 1024, False),   # hybridGEMM.md reference projection shape
        (8, 128, 256, 512, True),      # expert_id != segment_id (hybrid split)
        (3, 320, 384, 640, True),      # odd sizes: pipeline wrap across segments
    ]
    for num_groups, m_per_group, n, k, permuted in cases:
        a_q, sfa, b_q, sfb, offsets, experts, list_size, ref = _build_case(
            num_groups, m_per_group, n, k, permuted=permuted)
        d = torch.empty((ref.shape[0], n), device="cuda", dtype=torch.float32)
        fn((a_q, sfa), (b_q, sfb), d, offsets, experts, list_size, recipe=recipe)

        diff = calc_diff(d, ref)
        tag = "permuted" if permuted else "identity"
        print(f"   > ({num_groups=}, m={ref.shape[0]}, n={n}, k={k}, {tag}): diff={diff:.5f}")
        assert diff < DIFF_TOL, f"{num_groups=}, n={n}, k={k}, {tag}, diff={diff:.5f}"
    print()


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_deep_matches_asym_kernel() -> None:
    """Cross-kernel parity: both kernels on identical inputs must agree to
    fp32-reduction-order tolerance (deep accumulates full K in int32, asym
    chains per-K-block fp32 REDUCE_ADDs — deep is the more exact of the two)."""
    deep, asym = _deep_fn(), _asym_fn()
    if asym is None:
        pytest.skip("asym contiguous kernel not available")
    random.seed(1)
    torch.manual_seed(1)
    recipe = (1, GRAN_K, GRAN_K)

    a_q, sfa, b_q, sfb, offsets, experts, list_size, _ = _build_case(4, 256, 512, 1024)
    m = a_q.shape[0]
    d_deep = torch.empty((m, 512), device="cuda", dtype=torch.float32)
    d_asym = torch.empty((m, 512), device="cuda", dtype=torch.float32)
    deep((a_q, sfa), (b_q, sfb), d_deep, offsets, experts, list_size, recipe=recipe)
    asym((a_q, sfa), (b_q, sfb), d_asym, offsets, experts, list_size, recipe=recipe)

    diff = calc_diff(d_deep, d_asym)
    print(f"   > deep vs asym: diff={diff:.7f}")
    assert diff < 1e-3, f"cross-kernel diff={diff:.7f}"


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_deep_bench_vs_asym() -> None:
    """hybridGEMM.md §10.1 V1: HBM weight-stream rate of the deep kernel vs
    the asym baseline (726 GB/s decode / 184 TFLOPS prefill on H200)."""
    deep, asym = _deep_fn(), _asym_fn()
    if asym is None:
        pytest.skip("asym contiguous kernel not available")
    torch.manual_seed(0)
    recipe = (1, GRAN_K, GRAN_K)
    G, N, K = 64, 2048, 1024          # 134 MB weights/pass (beats 50 MB L2)

    def bench(fn, *args, iters=20):
        for _ in range(5):
            fn(*args, recipe=recipe)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn(*args, recipe=recipe)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    print("Deep vs asym INT8 grouped GEMM on HBM weights (G=64, N=2048, K=1024):")
    for m_per_group, tag in ((64, "decode m=64 "), (512, "prefill m=512")):
        a_q, sfa, b_q, sfb, offsets, experts, list_size, _ = _build_case(G, m_per_group, N, K)
        m = a_q.shape[0]
        d = torch.empty((m, N), device="cuda", dtype=torch.float32)
        args = ((a_q, sfa), (b_q, sfb), d, offsets, experts, list_size)
        ms_deep = bench(deep, *args)
        ms_asym = bench(asym, *args)
        wbytes = G * N * K
        flops = 2 * m * N * K
        print(f"   > {tag}: deep {ms_deep:7.3f} ms ({wbytes/ms_deep/1e6:7.0f} GB/s, {flops/ms_deep/1e9:6.1f} TFLOPS) | "
              f"asym {ms_asym:7.3f} ms ({wbytes/ms_asym/1e6:7.0f} GB/s, {flops/ms_asym/1e9:6.1f} TFLOPS) | "
              f"{ms_asym/ms_deep:4.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available() or get_arch_major() != 9:
        print("Skip: SM90 GPU required")
        sys.exit(0)
    if getattr(asym_gemm, "m_grouped_int8_gemm_nt_contiguous", None) is None:
        print("Skip: m_grouped_int8_gemm_nt_contiguous not exported by this build")
        sys.exit(0)
    test_m_grouped_int8_deep_contiguous_sm90()
    test_deep_matches_asym_kernel()
    test_deep_bench_vs_asym()
