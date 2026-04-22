import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major
from generators import (
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
)


# ---------------------------------------------------------------------------
# Helpers (copied from test_bf16.py — not importable as a package)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_offsets_and_experts_start(m_indices: torch.Tensor, drop_invalid: bool = True):
    """
    Convert a 1D expert-id array m_indices (length M) into:
      offsets: 1D int64 tensor of segment start positions (0-based)
      experts: 1D tensor of expert ids for each segment

    If drop_invalid=True, segments with expert == -1 are removed.
    A sentinel pair (M, -1) is appended to the output when M > 0.
    """
    assert m_indices.dim() == 1, f"expected 1D m_indices, got {m_indices.shape}"
    M = m_indices.numel()
    device = m_indices.device

    if M == 0:
        offsets = torch.empty((0,), device=device, dtype=torch.long)
        experts = torch.empty((0,), device=device, dtype=m_indices.dtype)
        return offsets, experts

    change = (m_indices[1:] != m_indices[:-1])
    starts = torch.nonzero(change, as_tuple=False).flatten().to(torch.long) + 1
    offsets = torch.cat([torch.zeros((1,), device=device, dtype=torch.long), starts], dim=0)
    experts = m_indices[offsets]

    if drop_invalid:
        keep = (experts != -1)
        offsets = offsets[keep]
        experts = experts[keep]

    offsets = torch.cat(
        [offsets, torch.tensor([M], device=device, dtype=torch.long)],
        dim=0,
    )
    experts = torch.cat(
        [experts, torch.tensor([-1], device=device, dtype=m_indices.dtype)],
        dim=0,
    )

    return offsets, experts


def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128):
    """
    Build offsets and experts for sparse m-grouped masked GEMM with fixed per-group allocation.

    Each group gets fixed allocation of max_m space. Only groups with masked_m[g] > 0 are
    included in the output mapping. Each active group generates a pair of offsets (start, end).
    """
    offsets = []
    experts = []

    for g in range(num_groups):
        v = masked_m[g].item()
        if v > 0:
            start = g * max_m
            end = start + ((v + block_m - 1) // block_m) * block_m
            offsets.append(start)
            offsets.append(end)
            experts.append(g)

    experts.append(-1)

    return (torch.tensor(offsets, dtype=torch.int32, device=masked_m.device),
            torch.tensor(experts, dtype=torch.int32, device=masked_m.device),
            len(experts))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_bf16_contiguous_sm90():
    print("\nTesting SM90 m-grouped BF16 contiguous GEMM:")
    compiled_dims = "nk"

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        # Only K-major layouts are supported
        if not major_a.is_k_major() or not major_b.is_k_major():
            continue

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True
        )

        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        d_asym = torch.empty_like(d)

        offsets, experts = extract_offsets_and_experts_start(m_indices)
        experts_i32 = experts.to(dtype=torch.int32, device="cuda").contiguous()
        offsets_i32 = offsets.to(dtype=torch.int32, device="cuda").contiguous()
        list_size = experts_i32.numel()

        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a, b_pinned, d_asym, offsets_i32, experts_i32, list_size, compiled_dims
        )

        # Mask out padding rows before computing diff
        d_asym = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        ref_d_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

        diff = calc_diff(d_asym, ref_d_masked)
        print(f"  num_groups={num_groups:2}, expected_m={expected_m_per_group:4}, "
              f"n={n:5}, k={k:5}, m={m:6}: diff={diff:.5e}")
        assert diff < 0.001, (
            f"diff={diff:.5e} too large for "
            f"num_groups={num_groups}, expected_m_per_group={expected_m_per_group}, n={n}, k={k}"
        )


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_bf16_masked_sm90():
    print("\nTesting SM90 m-grouped BF16 masked GEMM:")
    compiled_dims = "nk"

    for _, _, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.bfloat16):
        # psum layout is not supported by asym GEMM
        if use_psum_layout:
            continue

        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k, use_bf16=True, use_psum_layout=False
        )

        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        d_asym = torch.empty_like(d)

        asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
            a, b_pinned, d_asym, masked_m, expected_m_per_group, compiled_dims=compiled_dims
        )

        max_diff = 0.0
        for j in range(num_groups):
            actual_m = masked_m[j].item()
            if actual_m == 0:
                continue
            d_slice = d_asym[j, :actual_m]
            ref_slice = ref_d[j, :actual_m]
            diff = calc_diff(d_slice, ref_slice)
            max_diff = max(max_diff, diff)

        status = "PASS" if max_diff < 0.001 else "FAIL"
        print(f"  [{status}] num_groups={num_groups:2}, expected_m={expected_m_per_group:4}, "
              f"n={n:5}, k={k:5}: max_diff={max_diff:.5e}")
        assert max_diff < 0.001, (
            f"max_diff={max_diff:.5e} too large for "
            f"num_groups={num_groups}, expected_m_per_group={expected_m_per_group}, n={n}, k={k}"
        )
