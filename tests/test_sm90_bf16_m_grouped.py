import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major
from generators import (
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_offsets_experts_from_m_indices_pairs(m_indices: torch.Tensor, block_m: int = 128):
    """
    Convert 1D expert-id array m_indices (contiguous layout) into offset pairs and experts.

    Groups contiguous tokens by expert, creating (start, end) offset pairs for each group.
    Each pair is padded to block_m alignment.

    Args:
        m_indices: (M,) tensor with expert IDs, where -1 indicates invalid tokens
        block_m: block alignment (default 128)

    Returns:
        offsets: flat tensor with pairs [start_0, end_0, start_1, end_1, ...]
        experts: expert IDs for each group + terminator (-1)
        list_size: number of experts (including terminator)
    """
    assert m_indices.dim() == 1, f"expected 1D m_indices, got {m_indices.shape}"
    M = m_indices.numel()
    device = m_indices.device

    if M == 0:
        offsets = torch.empty((0,), device=device, dtype=torch.int32)
        experts = torch.tensor([-1], device=device, dtype=m_indices.dtype)
        return offsets, experts, 1

    # Find boundaries where expert id changes
    change = (m_indices[1:] != m_indices[:-1])

    # Segment start indices
    starts = torch.nonzero(change, as_tuple=False).flatten().to(torch.long) + 1
    segment_starts = torch.cat([torch.zeros((1,), device=device, dtype=torch.long), starts], dim=0)

    # Segment end indices
    segment_ends = torch.cat([starts, torch.tensor([M], device=device, dtype=torch.long)], dim=0)

    # Expert id for each segment
    segment_experts = m_indices[segment_starts]

    offsets = []
    experts = []

    for start, end, expert_id in zip(segment_starts.tolist(), segment_ends.tolist(), segment_experts.tolist()):
        if expert_id == -1:  # Skip invalid segments
            continue

        # Pad to block_m alignment
        start_padded = (start // block_m) * block_m
        end_padded = ((end + block_m - 1) // block_m) * block_m

        offsets.append(start_padded)
        offsets.append(end_padded)
        experts.append(expert_id)

    experts.append(-1)  # Terminator

    return (torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(experts, dtype=m_indices.dtype, device=device),
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

        offsets_i32, experts_i32, list_size = build_offsets_experts_from_m_indices_pairs(m_indices)

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
def test_m_grouped_bf16_contiguous_transpose_b_sm90():
    device = torch.device("cuda")
    m, n_phys, k_phys = 64, 128, 512
    a = torch.randn((m, n_phys), device=device, dtype=torch.bfloat16)
    b = torch.randn((1, n_phys, k_phys), device=device, dtype=torch.bfloat16)
    b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
    d_asym = torch.empty((m, k_phys), device=device, dtype=torch.float32)
    offsets_i32 = torch.tensor([0, m], device=device, dtype=torch.int32)
    experts_i32 = torch.tensor([0, -1], device=device, dtype=torch.int32)

    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b_pinned, d_asym, offsets_i32, experts_i32, 2, "mnk", True
    )

    ref_d = a.float() @ b[0].float()
    diff = calc_diff(d_asym, ref_d)
    assert diff < 0.001, f"transpose_b diff={diff:.5e} too large"


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


if __name__ == '__main__':
    import random
    random.seed(0)
    torch.manual_seed(0)
    test_m_grouped_bf16_contiguous_sm90()
    test_m_grouped_bf16_masked_sm90()
