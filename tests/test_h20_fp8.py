import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major
from generators import (
    KernelType, get_ue8m0_usage,
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
)


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


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_fp8_contiguous_sm90() -> None:
    print('Testing SM90 m-grouped contiguous FP8 asym GEMM:')
    recipe = (1, 1, 128)

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = '1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Only K-major A and B are supported
        if not major_a.is_k_major() or not major_b.is_k_major():
            continue

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b,
            use_ue8m0=use_ue8m0
        )
        offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(m_indices)

        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a, b, d, offsets, experts, list_size,
            recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast
        )

        d_asym_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
        ref_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

        diff = calc_diff(d_asym_masked, ref_masked)
        print(f'   > ({num_groups=}, m={m}, n={n}, k={k}, {kernel_opt}, layout={major_opt}): diff={diff:.5f}')
        assert diff < 0.001, (
            f'{num_groups=}, {m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, diff={diff:.5f}'
        )

    print()


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 (H20) required")
def test_m_grouped_fp8_masked_sm90() -> None:
    print('Testing SM90 m-grouped masked FP8 asym GEMM:')

    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        # psum layout not supported on SM90
        if use_psum_layout:
            continue

        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        kernel_opt = '1D1D' if kernel_type.is_1d1d() else '1D2D'

        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k,
            use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout
        )

        d_asym = torch.empty_like(d)
        asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
            a, b, d_asym, masked_m, expected_m_per_group,
            disable_ue8m0_cast=disable_ue8m0_cast,
            recipe=(1, 1, 128)
        )

        max_diff = 0.0
        for j in range(num_groups):
            vm = masked_m[j].item()
            if vm > 0:
                diff = calc_diff(d_asym[j, :vm], ref_d[j, :vm])
                max_diff = max(max_diff, diff)

        print(f'   > ({kernel_opt}, {num_groups=}, expected_m={expected_m_per_group}, n={n}, k={k}): max_diff={max_diff:.5f}')
        assert max_diff < 0.001, (
            f'{kernel_opt}, {num_groups=}, expected_m={expected_m_per_group}, {n=}, {k=}, max_diff={max_diff:.5f}'
        )

    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_m_grouped_fp8_contiguous_sm90()
    test_m_grouped_fp8_masked_sm90()
