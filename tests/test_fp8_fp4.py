import copy
import numpy as np
import random
import torch

import deep_gemm
import asym_gemm
from asym_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major
)

from generators import (
    KernelType, get_ue8m0_usage, align,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
)

def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, max_m: int, block_m: int = 128):
    """
    Build offsets/experts for asymScheduler (pairs format) from a masked_m tensor.

    `a` is stored as (num_groups, max_m, k), so each group occupies `max_m` slots
    starting at `g * max_m`. Only groups with masked_m[g] > 0 are emitted.
    Each active group contributes a (start, end) pair. `experts` has a trailing -1.

    Layout matches asymScheduler: offsets[blockIdx.y*2] = start, offsets[blockIdx.y*2+1] = end.
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
    return (torch.tensor(offsets, dtype=torch.int32, device='cuda'),
            torch.tensor(experts, dtype=torch.int32, device='cuda'),
            len(experts))

def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    fp8_kernel = asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe = (1, 128, 128)

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0
        )
        # Convert m_indices (contiguous layout) into pairs-format offsets/experts
        # matching asymScheduler's expected layout (see tests/test_fp8.py).
        from test_fp8 import build_offsets_experts_from_m_indices_pairs
        offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(m_indices)
        fp8_kernel(a, b, d, offsets, experts, list_size,
                   recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast)
        diff = calc_diff(d, ref_d)
        assert diff < 0.05, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0
        )
        offsets, experts, list_size = build_offsets_experts_from_m_indices_pairs(m_indices)

        # noinspection PyShadowingNames
        def test_func():
            fp8_kernel(a, b, d, offsets, experts, list_size,
                       recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')

        break
    print()

def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, enable_overlap, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        if enable_overlap and (use_psum_layout or get_arch_major() != 9):
            continue

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0

        for i in range(num_tests):
            a, b, masked_m, psum_m, d, ref_d, signal = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                                 use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                 quant_config=quant_config, enable_overlap=enable_overlap)
            if use_psum_layout:
                a_psum = (layout_masked_to_psum(a[0], psum_m), layout_masked_to_psum(a[1], psum_m))
                d_psum = layout_masked_to_psum(d, psum_m)


            # Construct full cases
            d_asym = torch.empty_like(d)

            # noinspection PyShadowingNames
            def test_func():
                if use_psum_layout:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a_psum, b, d_psum, psum_m, disable_ue8m0_cast=disable_ue8m0_cast,
                                                                   use_psum_layout=True, expected_m_for_psum_layout=expected_m_per_group,
                                                                   recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                else:
                    result = deep_gemm.m_grouped_fp8_gemm_nt_masked(
                        a, b, d, masked_m, expected_m_per_group,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                        enable_overlap=enable_overlap, signal=signal,
                    )
                    if enable_overlap:
                        block_m, threshold = result
                        check_signal(num_groups, max_m, block_m, threshold, signal, masked_m)

            # noinspection PyShadowingNames
            def test_func_asym():
                asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)


            test_func()
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                if use_psum_layout:
                    d_slice = d_psum[: psum_m[j]] if j == 0 else d_psum[align(psum_m[j - 1], 128): psum_m[j]]
                else:
                    d_slice = d[j, :masked_m[j].item()]
                diff = calc_diff(d_slice, ref_d[j, :masked_m[j].item()])
                assert diff < quant_config.max_diff(), f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'{kernel_opt}, psum={1 if use_psum_layout else 0}, enable_overlap={enable_overlap}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    # test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
