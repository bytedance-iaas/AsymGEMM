import copy
import numpy as np
import random
import torch

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

        m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0
        )
        try:
            fp8_kernel(a, b, d, grouped_layout, recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast)
        except TypeError:
            fp8_kernel(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast)
        diff = calc_diff(d, ref_d)
        assert diff < 0.05, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}'
        m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0
        )

        # noinspection PyShadowingNames
        def test_func():
            try:
                fp8_kernel(a, b, d, grouped_layout, recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast)
            except TypeError:
                fp8_kernel(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast)

        import ipdb
        ipdb.set_trace()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')

        break
    print()

if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    test_m_grouped_gemm_contiguous()
