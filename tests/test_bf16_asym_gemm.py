# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

import random
import torch

import asym_gemm
import deep_gemm
from asym_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major,
)
from generators import (
    align,
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    layout_masked_to_psum,
    build_offsets_experts_from_m_indices, build_offsets_experts_from_masked_m,
)


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, _, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.bfloat16):
        num_tests = 8
        sum_t_deep, max_t_deep = 0, 0
        sum_t_asym, max_t_asym = 0, 0
        sum_ops, sum_bytes = 0, 0

        for i in range(num_tests):
            a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k,
                use_bf16=True, use_psum_layout=use_psum_layout)

            if use_psum_layout:
                a_psum = layout_masked_to_psum(a, psum_m)
                d_psum_deep = layout_masked_to_psum(d, psum_m)

            b_pinned = b.detach().to("cpu", non_blocking=False).pin_memory()
            b_gpu = b_pinned.to(device="cuda", non_blocking=True)
            d_deep = torch.empty_like(d)
            d_asym = torch.empty_like(d)
            torch.cuda.synchronize()

            offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

            # noinspection PyShadowingNames
            def test_func_deep():
                if use_psum_layout:
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_psum, b_gpu, d_psum_deep, psum_m,
                                                                use_psum_layout=True,
                                                                expected_m_for_psum_layout=expected_m_per_group)
                else:
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b_gpu, d_deep, masked_m, expected_m_per_group)

            # noinspection PyShadowingNames
            def test_func_asym():
                if not use_psum_layout:
                    asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
                        a, b_pinned, d_asym, offsets, experts, list_size,
                        expected_m_per_group, compiled_dims="nk")

            test_func_deep()
            if not use_psum_layout:
                test_func_asym()

            deep_diff_max = 0.0
            asym_diff_max = 0.0

            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                if use_psum_layout:
                    d_slice_deep = (d_psum_deep[: psum_m[j]] if j == 0
                                    else d_psum_deep[align(psum_m[j - 1], 128): psum_m[j]])
                else:
                    d_slice_deep = d_deep[j, :masked_m[j].item()]
                    d_slice_asym = d_asym[j, :masked_m[j].item()]

                deep_diff = calc_diff(d_slice_deep, ref_d[j, :masked_m[j].item()])
                deep_diff_max = max(deep_diff_max, deep_diff)

                if not use_psum_layout:
                    asym_diff = calc_diff(d_slice_asym, ref_d[j, :masked_m[j].item()])
                    asym_diff_max = max(asym_diff_max, asym_diff)

            valid_m = masked_m.sum().item()
            t_deep_gpu = bench_kineto(test_func_deep, 'bf16_gemm', suppress_kineto_output=True)
            t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True) if not use_psum_layout else 0

            sum_t_deep += t_deep_gpu
            max_t_deep = max(max_t_deep, t_deep_gpu)
            sum_t_asym += t_asym
            max_t_asym = max(max_t_asym, t_asym)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        if sum_t_deep > 0:
            print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, '
                  f'n={n:4}, k={k:4}, psum={1 if use_psum_layout else 0}): '
                  f'deep_gpu={sum_t_deep / num_tests * 1e6:4.0f} (max: {max_t_deep * 1e6:3.0f}) us | '
                  f'{sum_ops / sum_t_deep / 1e12:4.0f} TFLOPS | '
                  f'asym_gpu={sum_t_asym / num_tests * 1e6:4.0f} (max: {max_t_asym * 1e6:3.0f}) us | '
                  f'asym_TFLOPS={sum_ops / sum_t_asym / 1e12 if sum_t_asym > 0 else 0:4.0f} | '
                  f'{sum_bytes / sum_t_deep / 1e9:4.0f} GB/s')
        if not use_psum_layout:
            print(f'   deep_gpu diff={deep_diff_max:.5e} | asym_gpu diff={asym_diff_max:.5e}')
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
        d_deep_gpu = torch.empty_like(d)
        d_asym = torch.empty_like(d)

        b_gpu = b_pinned.to(device="cuda", non_blocking=True)
        torch.cuda.synchronize()

        offsets, experts, list_size = build_offsets_experts_from_m_indices(m_indices, num_groups)

        # noinspection PyShadowingNames
        def test_func_deep():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b_gpu, d_deep_gpu, m_indices)

        # noinspection PyShadowingNames
        def test_func_asym():
            asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
                a, b_pinned, d_asym, offsets, experts, list_size, compiled_dims
            )

        t_deep_gpu = bench_kineto(test_func_deep, 'bf16_gemm', suppress_kineto_output=True)
        t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True)

        d_deep_gpu = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_deep_gpu), d_deep_gpu)
        d_asym = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)

        deep_gpu_diff = calc_diff(d_deep_gpu, ref_d)
        asym_diff = calc_diff(d_asym, ref_d)
        active_m = int((m_indices != -1).sum().item())
        flops_total = 2 * m * n * k
        flops_active = 2 * active_m * n * k

        print(f'   deep_gpu diff={deep_gpu_diff:.5e} | asym_gemm diff={asym_diff:.5e}')
        if t_deep_gpu > 0 and t_asym > 0:
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}): '
                  f'active_m={active_m:5} | '
                  f'deep_gpu={t_deep_gpu * 1e6:6.0f} us | '
                  f'deep_tflops={flops_active / t_deep_gpu / 1e12:4.0f} | '
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
