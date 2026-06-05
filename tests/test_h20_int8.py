"""SM90 (H20/Hopper) INT8 1D1D asymmetric grouped GEMM tests.

Self-contained: builds int8 inputs + FP32 per-token (A) / per-channel (B) scales,
computes a float reference, calls the new int8 entry points, and compares with
`calc_diff`. See `int8.md` for the implementation plan and verification stages.

The int8 kernel/API may not be built yet; tests skip gracefully if the entry point
is missing so this file stays valid before the kernel lands.
"""
import random

import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major


GRAN_K = 128
DIFF_TOL = 0.05  # int8 quant is coarser than fp8; see int8.md §7


# --------------------------------------------------------------------------- #
# Quantization helpers (symmetric, round-to-nearest, clamp to ±127)
# --------------------------------------------------------------------------- #
def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def per_token_cast_to_int8(x: torch.Tensor, gran_k: int = GRAN_K):
    """x: [M, K] -> (int8 [M, K], fp32 sf [M, K//gran_k]).

    One scale per (row, gran_k-block along K). sf = amax / 127.
    """
    assert x.dim() == 2
    m, n = x.shape
    padded_n = _align(n, gran_k)
    xp = x.new_zeros((m, padded_n))
    xp[:, :n] = x
    xv = xp.view(m, -1, gran_k)
    amax = xv.abs().float().amax(dim=2).clamp(1e-4)
    sf = amax / 127.0
    q = (xv / sf.unsqueeze(2)).round().clamp(-127, 127).to(torch.int8)
    return q.view(m, padded_n)[:, :n].contiguous(), sf


def per_channel_cast_to_int8(x: torch.Tensor, gran_k: int = GRAN_K):
    """B is [N, K]; each output channel (row) is scaled independently. Identical
    math to per_token_cast_to_int8 but named to document B's semantics."""
    return per_token_cast_to_int8(x, gran_k)


def _expand_sf(sf: torch.Tensor, k: int, gran_k: int = GRAN_K) -> torch.Tensor:
    """[*, K//gran_k] -> [*, K] by repeating each block scale gran_k times."""
    return sf.repeat_interleave(gran_k, dim=-1)[..., :k]


def dequant(q: torch.Tensor, sf: torch.Tensor, k: int) -> torch.Tensor:
    return q.float() * _expand_sf(sf, k)


# --------------------------------------------------------------------------- #
# Float reference: int32 accumulate-per-block then SFA*SFB is numerically
# identical to dequantize-then-matmul because each scale is constant within its
# K-block.
# --------------------------------------------------------------------------- #
def ref_2d(a_q, sfa, b_q, sfb, k):
    a = dequant(a_q, sfa, k)            # [M, K]
    b = dequant(b_q, sfb, k)            # [N, K]
    return a @ b.transpose(-1, -2)      # [M, N]


# --------------------------------------------------------------------------- #
# API availability guard
# --------------------------------------------------------------------------- #
def _int8_masked_fn():
    fn = getattr(asym_gemm, "m_grouped_int8_asym_gemm_nt_masked", None)
    if fn is None:
        pytest.skip("int8 SM90 masked entry point not built yet (see int8.md)")
    return fn


def _int8_contiguous_fn():
    fn = getattr(asym_gemm, "m_grouped_int8_asym_gemm_nt_contiguous", None)
    if fn is None:
        pytest.skip("int8 SM90 contiguous entry point not built yet (see int8.md)")
    return fn


def build_offsets_experts_from_m_indices_pairs(m_indices: torch.Tensor, block_m: int = 128):
    """Convert a 1D expert-id array (contiguous layout) into block_m-aligned
    (start, end) offset pairs + experts (with -1 terminator)."""
    assert m_indices.dim() == 1
    M = m_indices.numel()
    device = m_indices.device
    if M == 0:
        return (torch.empty((0,), device=device, dtype=torch.int32),
                torch.tensor([-1], device=device, dtype=m_indices.dtype), 1)

    change = (m_indices[1:] != m_indices[:-1])
    starts = torch.nonzero(change, as_tuple=False).flatten().to(torch.long) + 1
    seg_starts = torch.cat([torch.zeros((1,), device=device, dtype=torch.long), starts])
    seg_ends = torch.cat([starts, torch.tensor([M], device=device, dtype=torch.long)])
    seg_experts = m_indices[seg_starts]

    offsets, experts = [], []
    for s, e, eid in zip(seg_starts.tolist(), seg_ends.tolist(), seg_experts.tolist()):
        if eid == -1:
            continue
        offsets.append((s // block_m) * block_m)
        offsets.append(((e + block_m - 1) // block_m) * block_m)
        experts.append(eid)
    experts.append(-1)
    return (torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(experts, dtype=m_indices.dtype, device=device),
            len(experts))


# --------------------------------------------------------------------------- #
# Stage B/C: single-group smoke test (isolate MMA + scale-apply, then REDUCE_ADD)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
@pytest.mark.parametrize("k", [128, 512])  # 128 = single K-block (STORE only); 512 = REDUCE_ADD
def test_int8_smoke_sm90(k: int) -> None:
    fn = _int8_masked_fn()
    random.seed(0)
    torch.manual_seed(0)

    num_groups, m, n = 1, 128, 128
    a = torch.randn((num_groups, m, k), device="cuda", dtype=torch.float32) / k ** 0.25
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

    a_q = torch.empty_like(a, dtype=torch.int8)
    b_q = torch.empty_like(b, dtype=torch.int8)
    sfa = torch.empty((num_groups, m, _ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
    sfb = torch.empty((num_groups, n, _ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
    for g in range(num_groups):
        a_q[g], sfa[g] = per_token_cast_to_int8(a[g])
        b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

    masked_m = torch.full((num_groups,), m, device="cuda", dtype=torch.int32)
    d = torch.empty((num_groups, m, n), device="cuda", dtype=torch.float32)

    fn((a_q, sfa), (b_q, sfb), d, masked_m, m, recipe=(1, GRAN_K, GRAN_K))

    ref = ref_2d(a_q[0], sfa[0], b_q[0], sfb[0], k)
    diff = calc_diff(d[0], ref)
    print(f"   > smoke (k={k}): diff={diff:.5f}")
    assert diff < DIFF_TOL, f"k={k}, diff={diff:.5f}"


# --------------------------------------------------------------------------- #
# PCIe-resident expert weights: B lives in CUDA-pinned host memory and is fetched
# by Hopper TMA over the UVA address space. Activations (A), output (D) and BOTH
# scale tensors (SFA, SFB) stay in device memory — only the weights B are host-pinned.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_int8_pinned_weights_sm90() -> None:
    fn = _int8_masked_fn()
    random.seed(0)
    torch.manual_seed(0)
    print("Testing SM90 masked INT8 asym GEMM with pinned-CPU expert weights:")

    for num_groups, m, n, k in [(1, 256, 256, 512), (4, 512, 128, 256), (8, 256, 256, 512)]:
        kb = _ceil_div(k, GRAN_K)
        a = torch.randn((num_groups, m, k), device="cuda", dtype=torch.float32) / k ** 0.25
        b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

        a_q = torch.empty_like(a, dtype=torch.int8)
        b_q_gpu = torch.empty_like(b, dtype=torch.int8)
        sfa = torch.empty((num_groups, m, kb), device="cuda", dtype=torch.float32)
        sfb = torch.empty((num_groups, n, kb), device="cuda", dtype=torch.float32)
        for g in range(num_groups):
            a_q[g], sfa[g] = per_token_cast_to_int8(a[g])
            b_q_gpu[g], sfb[g] = per_channel_cast_to_int8(b[g])

        # Expert weights in CUDA-pinned host memory; SFB stays on the GPU.
        b_q = b_q_gpu.cpu().pin_memory()
        assert b_q.is_pinned() and sfb.is_cuda

        masked_m = torch.full((num_groups,), m, device="cuda", dtype=torch.int32)
        d = torch.empty((num_groups, m, n), device="cuda", dtype=torch.float32)
        fn((a_q, sfa), (b_q, sfb), d, masked_m, m, recipe=(1, GRAN_K, GRAN_K))

        max_diff = 0.0
        for g in range(num_groups):
            ref = ref_2d(a_q[g], sfa[g], b_q_gpu[g], sfb[g], k)
            max_diff = max(max_diff, calc_diff(d[g], ref))
        print(f"   > (pinned-B, {num_groups=}, m={m}, n={n}, k={k}): max_diff={max_diff:.5f}")
        assert max_diff < DIFF_TOL, f"pinned-B {num_groups=}, m={m}, {n=}, {k=}, max_diff={max_diff:.5f}"
    print()


# --------------------------------------------------------------------------- #
# Stage D: masked grouped sweep
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_int8_masked_sm90() -> None:
    fn = _int8_masked_fn()
    random.seed(0)
    torch.manual_seed(0)
    print("Testing SM90 m-grouped masked INT8 asym GEMM:")

    for num_groups, max_m, expected_m, n, k in [
        (1, 1024, 1024, 256, 512),
        (4, 1024, 1024, 256, 512),
        (8, 512, 512, 128, 256),
    ]:
        a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.float32) / k ** 0.25
        b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

        a_q = torch.empty_like(a, dtype=torch.int8)
        b_q = torch.empty_like(b, dtype=torch.int8)
        sfa = torch.empty((num_groups, max_m, _ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
        sfb = torch.empty((num_groups, n, _ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
        for g in range(num_groups):
            a_q[g], sfa[g] = per_token_cast_to_int8(a[g])
            b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

        masked_m = torch.tensor(
            [int(expected_m * random.uniform(0.7, 1.3)) for _ in range(num_groups)],
            device="cuda", dtype=torch.int32,
        ).clamp_max(max_m)

        d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.float32)
        fn((a_q, sfa), (b_q, sfb), d, masked_m, expected_m, recipe=(1, GRAN_K, GRAN_K))

        max_diff = 0.0
        for g in range(num_groups):
            vm = masked_m[g].item()
            if vm > 0:
                ref = ref_2d(a_q[g, :vm], sfa[g, :vm], b_q[g], sfb[g], k)
                max_diff = max(max_diff, calc_diff(d[g, :vm], ref))

        print(f"   > ({num_groups=}, expected_m={expected_m}, n={n}, k={k}): max_diff={max_diff:.5f}")
        assert max_diff < DIFF_TOL, f"{num_groups=}, expected_m={expected_m}, {n=}, {k=}, max_diff={max_diff:.5f}"
    print()


# --------------------------------------------------------------------------- #
# Stage E: contiguous grouped sweep
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_int8_contiguous_sm90() -> None:
    fn = _int8_contiguous_fn()
    random.seed(0)
    torch.manual_seed(0)
    print("Testing SM90 m-grouped contiguous INT8 asym GEMM:")
    recipe = (1, GRAN_K, GRAN_K)

    for num_groups, m_per_group, n, k in [(2, 256, 256, 512), (4, 256, 128, 256)]:
        m = num_groups * m_per_group
        a = torch.randn((m, k), device="cuda", dtype=torch.float32) / k ** 0.25
        b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

        a_q, sfa = per_token_cast_to_int8(a)
        b_q = torch.empty_like(b, dtype=torch.int8)
        sfb = torch.empty((num_groups, n, _ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
        for g in range(num_groups):
            b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

        # Block-aligned contiguous expert assignment.
        m_indices = torch.arange(num_groups, device="cuda").repeat_interleave(m_per_group).to(torch.int32)
        offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(m_indices)

        d = torch.empty((m, n), device="cuda", dtype=torch.float32)
        fn((a_q, sfa), (b_q, sfb), d, offsets, experts, list_size, recipe=recipe)

        # Reference per group, masking invalid rows.
        ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        for g in range(num_groups):
            rows = (m_indices == g)
            ref[rows] = ref_2d(a_q[rows], sfa[rows], b_q[g], sfb[g], k)

        valid = (m_indices != -1).unsqueeze(1)
        diff = calc_diff(torch.where(valid, d, 0.0), torch.where(valid, ref, 0.0))
        print(f"   > ({num_groups=}, m={m}, n={n}, k={k}): diff={diff:.5f}")
        assert diff < DIFF_TOL, f"{num_groups=}, m={m}, {n=}, {k=}, diff={diff:.5f}"
    print()


if __name__ == "__main__":
    test_int8_smoke_sm90(128)
    test_int8_smoke_sm90(512)
    test_int8_pinned_weights_sm90()
    test_m_grouped_int8_masked_sm90()
    test_m_grouped_int8_contiguous_sm90()
