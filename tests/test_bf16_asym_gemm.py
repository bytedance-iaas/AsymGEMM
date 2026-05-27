# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

import random
import torch

import asym_gemm
from asym_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major,
)
from generators import (
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    build_offsets_experts_from_m_indices,
)


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, _, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.bfloat16):
        if use_psum_layout:
            # psum layout has no asym kernel path here; skip rather than running a perf-only deep path.
            continue

        num_tests = 8
        sum_t_asym, max_t_asym = 0, 0
        sum_ops, sum_bytes = 0, 0
        asym_diff_max = 0.0

        for i in range(num_tests):
            a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k,
                use_bf16=True, use_psum_layout=False)

            b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
            d_asym = torch.empty_like(d)
            torch.cuda.synchronize()

            # noinspection PyShadowingNames
            def test_func_asym():
                asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
                    a, b_pinned, d_asym, masked_m,
                    expected_m_per_group, compiled_dims="nk")

            test_func_asym()

            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                asym_diff = calc_diff(d_asym[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                asym_diff_max = max(asym_diff_max, asym_diff)

            valid_m = masked_m.sum().item()
            t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True)

            sum_t_asym += t_asym
            max_t_asym = max(max_t_asym, t_asym)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        if sum_t_asym > 0:
            print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, '
                  f'n={n:4}, k={k:4}): '
                  f'asym_gpu={sum_t_asym / num_tests * 1e6:4.0f} (max: {max_t_asym * 1e6:3.0f}) us | '
                  f'asym_TFLOPS={sum_ops / sum_t_asym / 1e12:4.0f} | '
                  f'{sum_bytes / sum_t_asym / 1e9:4.0f} GB/s')
        print(f'   asym_gpu diff={asym_diff_max:.5e}')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')
    compiled_dims = "mnk"

    for _, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        # we only support k_major until now. vLLM and Sglang mainly use K major
        if not major_a.is_k_major() or not major_b.is_k_major():
            continue

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_bf16=True)
        b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
        d_asym = torch.empty_like(d)
        torch.cuda.synchronize()

        offsets, experts, list_size = build_offsets_experts_from_m_indices(m_indices, num_groups)

        # noinspection PyShadowingNames
        def test_func_asym():
            asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                a, b_pinned, d_asym, offsets, experts, list_size, compiled_dims
            )

        test_func_asym()
        t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True)

        d_asym = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        asym_diff = calc_diff(d_asym, ref_d)
        active_m = int((m_indices != -1).sum().item())
        flops_active = 2 * active_m * n * k

        print(f'   asym_gemm diff={asym_diff:.5e}')
        if t_asym > 0:
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
                  f'active_m={active_m:5} | '
                  f'asym={t_asym * 1e6:6.0f} us | '
                  f'asym_tflops={flops_active / t_asym / 1e12:4.0f} | '
                  f'{count_bytes(a, b, d_asym) / 1e9 / t_asym:4.0f} GB/s')
        else:
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
                  'bench_kineto returned 0, skip TFLOPS')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    if get_arch_major() >= 9:
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()
