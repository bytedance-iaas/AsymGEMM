# Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
# Modified by Bytedance Inc., 2026.
# Original: https://github.com/deepseek-ai/DeepGEMM

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
    KernelType, get_ue8m0_usage,
    enumerate_m_grouped_contiguous, enumerate_m_grouped_masked,
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    build_offsets_experts_from_m_indices, build_offsets_experts_from_masked_m,
)


def print_5x5_matrix_diff(tag: str, out: torch.Tensor, ref: torch.Tensor):
    out_cpu = out.to(torch.float32).cpu().contiguous()
    ref_cpu = ref.to(torch.float32).cpu().contiguous()
    diff_cpu = (out_cpu - ref_cpu).contiguous()
    rows = min(5, out_cpu.size(0))
    cols = min(5, out_cpu.size(1))
    print(f'\n[{tag}] Out (top-left {rows}x{cols}):')
    for i in range(rows):
        print(' '.join(f'{float(out_cpu[i, j]):.6f}' for j in range(cols)))
    print(f'\n[{tag}] Ref (top-left {rows}x{cols}):')
    for i in range(rows):
        print(' '.join(f'{float(ref_cpu[i, j]):.6f}' for j in range(cols)))
    print(f'\n[{tag}] Diff = Out - Ref (top-left {rows}x{cols}):')
    for i in range(rows):
        print(' '.join(f'{float(diff_cpu[i, j]):.6f}' for j in range(cols)))


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    def build_groundtruth_from_original(a_bf16: torch.Tensor, b_bf16: torch.Tensor, m_indices: torch.Tensor):
        gt = torch.zeros((a_bf16.size(0), b_bf16.size(1)), device=a_bf16.device, dtype=torch.bfloat16)
        for g in range(b_bf16.size(0)):
            mask = (m_indices == g)
            if mask.any():
                gt[mask] = (a_bf16[mask].float() @ b_bf16[g].float().t()).to(torch.bfloat16)
        return gt

    def print_5x5_compare(tag: str, out: torch.Tensor, ref: torch.Tensor):
        out_cpu = out.to(torch.float32).cpu().contiguous()
        ref_cpu = ref.to(torch.float32).cpu().contiguous()
        diff_cpu = (out_cpu - ref_cpu).contiguous()
        rows = min(5, out_cpu.size(0))
        cols = min(5, out_cpu.size(1))
        print(f'\n[{tag}] Out (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(out_cpu[i, j]):.6f}' for j in range(cols)))
        print(f'\n[{tag}] Ref (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(ref_cpu[i, j]):.6f}' for j in range(cols)))
        print(f'\n[{tag}] Diff = Out - Ref (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(diff_cpu[i, j]):.6f}' for j in range(cols)))

    recipe_asym = (1, 128, 128)
    recipe_deepgemm = (1, 128, 128)

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # we only support k_major until now. vLLM and Sglang mainly use K major
        if not major_a.is_k_major() or not major_b.is_k_major():
            continue

        m, a, b, m_indices, d_asym, _, a_bf16, b_bf16 = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b,
            use_ue8m0=use_ue8m0, return_original=True
        )
        groundtruth = build_groundtruth_from_original(a_bf16, b_bf16, m_indices)
        d_deep = torch.empty_like(d_asym)
        offsets, experts, list_size = build_offsets_experts_from_m_indices(m_indices, num_groups)

        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a, b, d_asym, offsets, experts, list_size,
            recipe=recipe_asym, disable_ue8m0_cast=disable_ue8m0_cast
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d_deep, m_indices, recipe=recipe_deepgemm, disable_ue8m0_cast=disable_ue8m0_cast
        )

        d_asym_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        d_deep_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_deep), d_deep)
        diff_asym_ref = calc_diff(d_asym_masked, groundtruth)
        diff_deep_ref = calc_diff(d_deep_masked, groundtruth)
        diff_asym_deep = calc_diff(d_asym_masked, d_deep_masked)
        print(f'   > Precision ({major_opt}, {kernel_opt}): '
              f'asym128-vs-gt={diff_asym_ref:.5f}, deep128-vs-gt={diff_deep_ref:.5f}, '
              f'asym128-vs-deep128={diff_asym_deep:.5f}')
        print_5x5_compare('asym128-vs-sym128', d_asym_masked, d_deep_masked)
        print_5x5_compare('asym128-vs-gt', d_asym_masked, groundtruth)
        print_5x5_compare('sym128-vs-gt', d_deep_masked, groundtruth)
        assert diff_deep_ref < 0.001, (
            f'deep128 baseline drifted: {m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, '
            f'{diff_deep_ref:.5f}'
        )

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0
        )
        offsets, experts, list_size = build_offsets_experts_from_m_indices(m_indices, num_groups)

        # noinspection PyShadowingNames
        def test_func():
            asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
                a, b, d, offsets, experts, list_size,
                recipe=recipe_asym, disable_ue8m0_cast=disable_ue8m0_cast
            )

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        if t <= 0:
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
                  'bench_kineto returned 0, skip TFLOPS/GBps')
        else:
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0

        # Test correctness
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k,
            use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout)
        offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)

        deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group,
                                               disable_ue8m0_cast=disable_ue8m0_cast)

        d_asym = torch.empty_like(d)
        asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, offsets, experts, list_size,
                                                    expected_m_per_group,
                                                    disable_ue8m0_cast=disable_ue8m0_cast)

        max_diff_baseline = 0.0
        max_diff_asym = 0.0
        max_diff_asym_group = -1
        for j in range(num_groups):
            if masked_m[j].item() > 0:
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                max_diff_baseline = max(max_diff_baseline, diff)

                diff_asym = calc_diff(d_asym[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                if diff_asym > max_diff_asym:
                    max_diff_asym = diff_asym
                    max_diff_asym_group = j

        print(f'   > Precision ({kernel_opt}): baseline-vs-gt={max_diff_baseline:.5f}, asym-vs-gt={max_diff_asym:.5f}')

        for j in range(num_groups):
            if masked_m[j].item() > 0:
                print_5x5_matrix_diff('deep_gemm', d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                print_5x5_matrix_diff('asym_gemm', d_asym[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                break

        if max_diff_asym_group >= 0:
            j = max_diff_asym_group
            vm = masked_m[j].item()
            out_g = d_asym[j, :vm].to(torch.float32).cpu().contiguous()
            ref_g = ref_d[j, :vm].to(torch.float32).cpu().contiguous()
            abs_diff = (out_g - ref_g).abs()
            max_val = abs_diff.max().item()
            max_pos = (abs_diff == max_val).nonzero()[0]
            mi, ni_ = int(max_pos[0].item()), int(max_pos[1].item())
            radius = 3
            r0, r1 = max(0, mi - radius), min(vm, mi + radius + 1)
            c0, c1 = max(0, ni_ - radius), min(out_g.size(1), ni_ + radius + 1)
            print(f'\n[asym_gemm max diff] group={j}, location=({mi},{ni_}), max_abs_diff={max_val:.6f}')
            print(f'[asym_gemm max diff] Neighborhood rows [{r0},{r1-1}], cols [{c0},{c1-1}]:')
            print('  Out:')
            for i in range(r0, r1):
                print('  ' + ' '.join(f'{float(out_g[i, c]):.6f}' for c in range(c0, c1)))
            print('  Ref:')
            for i in range(r0, r1):
                print('  ' + ' '.join(f'{float(ref_g[i, c]):.6f}' for c in range(c0, c1)))
            print('  Diff:')
            for i in range(r0, r1):
                print('  ' + ' '.join(f'{float(out_g[i,c]-ref_g[i,c]):.6f}' for c in range(c0, c1)))

        # Performance with fixed shapes
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
        offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups, max_m)
        d_asym = torch.empty_like(d)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group,
                                                   disable_ue8m0_cast=disable_ue8m0_cast)

        # noinspection PyShadowingNames
        def test_func_asym():
            asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, offsets, experts, list_size,
                                                        expected_m_per_group,
                                                        disable_ue8m0_cast=disable_ue8m0_cast)

        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True)

        print(f' > Perf ({num_groups=}, expected_m={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): ')
        if t > 0:
            print(f'   Baseline: {t * 1e6:4.0f} us | {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS')
        else:
            print(f'   Baseline: bench_kineto returned 0, skip')
        if t_asym > 0:
            print(f'   Asym:     {t_asym * 1e6:4.0f} us | {2 * valid_m * n * k / t_asym / 1e12:4.0f} TFLOPS')
        else:
            print(f'   Asym:     bench_kineto returned 0, skip')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
