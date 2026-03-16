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
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
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


def gemm_tflops(m: int, n: int, k: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return 2 * m * n * k / elapsed_s / 1e12

@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 9)
def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    for kernel_type, quant_config, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe = (1, 1, 128) if kernel_type.is_1d1d() and accumulate else None

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
            func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(asym_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                  f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
        t = bench_kineto(lambda: asym_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe),
                         'fp8_gemm', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: asym_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        tflops = gemm_tflops(m, n, k, t)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {tflops:4.0f} TFLOPS | {tflops:4.0f} FP4-TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
        if cublas_t > 0:
            scores.append((cublas_t + split_k_t) / t)
    print(f"Average speedup over cuBLASLt: {float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x\n")

def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    def build_offsets_experts_from_m_indices(m_indices: torch.Tensor, num_groups: int):
        m_indices_cpu = m_indices.to('cpu', torch.int32).contiguous()
        m_list = m_indices_cpu.tolist()
        m = len(m_list)
        capacity = num_groups + 1

        offsets = []
        experts = []
        if m > 0:
            if m_list[0] != -1:
                offsets.append(0)
                experts.append(m_list[0])
            for i in range(1, m):
                if m_list[i] != m_list[i - 1] and m_list[i] != -1:
                    offsets.append(i)
                    experts.append(m_list[i])

        offsets.append(m)
        experts.append(-1)

        offsets = offsets[:capacity]
        experts = experts[:capacity]
        list_size = len(offsets)

        offsets_t = torch.tensor(offsets, dtype=torch.int32, device=m_indices.device)
        experts_t = torch.tensor(experts, dtype=torch.int32, device=m_indices.device)
        return offsets_t, experts_t, list_size

    def build_groundtruth_from_original(a_bf16: torch.Tensor, b_bf16: torch.Tensor, m_indices: torch.Tensor):
        gt = torch.zeros((a_bf16.size(0), b_bf16.size(1)), device=a_bf16.device, dtype=torch.bfloat16)
        for g in range(b_bf16.size(0)):
            mask = (m_indices == g)
            if mask.any():
                gt[mask] = (a_bf16[mask].float() @ b_bf16[g].float().t()).to(torch.bfloat16)
        return gt

    def print_5x5_compare_and_max_neighborhood(tag: str, out: torch.Tensor, ref: torch.Tensor):
        out_cpu = out.to(torch.float32).cpu().contiguous()
        ref_cpu = ref.to(torch.float32).cpu().contiguous()
        diff_cpu = (out_cpu - ref_cpu).contiguous()

        rows = min(5, out_cpu.size(0))
        cols = min(5, out_cpu.size(1))
        out_acc = out_cpu
        ref_acc = ref_cpu
        diff_acc = diff_cpu

        print(f'\n[{tag}] Out (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(out_acc[i, j]):.6f}' for j in range(cols)))

        print(f'\n[{tag}] Ref (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(ref_acc[i, j]):.6f}' for j in range(cols)))

        print(f'\n[{tag}] Diff = Out - Ref (top-left {rows}x{cols}):')
        for i in range(rows):
            print(' '.join(f'{float(diff_acc[i, j]):.6f}' for j in range(cols)))

        abs_diff = diff_cpu.abs()
        finite_mask = torch.isfinite(abs_diff)
        finite_count = int(finite_mask.sum().item())
        total_count = abs_diff.numel()
        non_finite_count = total_count - finite_count

        max_abs_diff = float('nan')
        mean_abs_diff = float('nan')
        max_i = -1
        max_j = -1

        if finite_count > 0:
            finite_abs = abs_diff.masked_select(finite_mask)
            max_abs_diff = float(finite_abs.max().item())
            mean_abs_diff = float(finite_abs.mean().item())
            max_pos = torch.nonzero(torch.logical_and(finite_mask, abs_diff == max_abs_diff))
            if max_pos.numel() > 0:
                max_i = int(max_pos[0, 0].item())
                max_j = int(max_pos[0, 1].item())

        # print(f'\n[{tag}] Abs diff stats (finite only): '
        #       f'max={max_abs_diff:.6f}, mean={mean_abs_diff:.6f}, non_finite={non_finite_count}/{total_count}')

        # if max_i >= 0 and max_j >= 0:
        #     print(f'[{tag}] Max finite abs diff location: ({max_i}, {max_j})')
        #     radius = 2
        #     r0 = max(0, max_i - radius)
        #     r1 = min(out_cpu.size(0), max_i + radius + 1)
        #     c0 = max(0, max_j - radius)
        #     c1 = min(out_cpu.size(1), max_j + radius + 1)

        #     print(f'[{tag}] Neighborhood rows [{r0}, {r1 - 1}], cols [{c0}, {c1 - 1}]')
        #     print(f'\n[{tag}] Out neighborhood:')
        #     for i in range(r0, r1):
        #         print(' '.join(f'{float(out_acc[i, j]):.6f}' for j in range(c0, c1)))
        #     print(f'\n[{tag}] Ref neighborhood:')
        #     for i in range(r0, r1):
        #         print(' '.join(f'{float(ref_acc[i, j]):.6f}' for j in range(c0, c1)))
        #     print(f'\n[{tag}] Diff neighborhood:')
        #     for i in range(r0, r1):
        #         print(' '.join(f'{float(diff_acc[i, j]):.6f}' for j in range(c0, c1)))

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
        b_pinned = (b[0].cpu().pin_memory(), b[1].cpu().pin_memory())
        import ipdb;ipdb.set_trace()
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a, b, d_asym, offsets, experts, list_size,
            recipe=recipe_asym, disable_ue8m0_cast=disable_ue8m0_cast
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d_deep, m_indices, recipe=recipe_deepgemm, disable_ue8m0_cast=disable_ue8m0_cast
        )
        # import ipdb;ipdb.set_trace()

        d_asym_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_asym), d_asym)
        d_deep_masked = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d_deep), d_deep)
        diff_asym_ref = calc_diff(d_asym_masked, groundtruth)
        diff_deep_ref = calc_diff(d_deep_masked, groundtruth)
        diff_asym_deep = calc_diff(d_asym_masked, d_deep_masked)
        print(f'   > Precision ({major_opt}, {kernel_opt}): '
              f'asym128-vs-gt={diff_asym_ref:.5f}, deep128-vs-gt={diff_deep_ref:.5f}, '
              f'asym128-vs-deep128={diff_asym_deep:.5f}')
        print_5x5_compare_and_max_neighborhood('asym128-vs-sym128', d_asym_masked, d_deep_masked)
        print_5x5_compare_and_max_neighborhood('asym128-vs-gt', d_asym_masked, groundtruth)
        print_5x5_compare_and_max_neighborhood('sym128-vs-gt', d_deep_masked, groundtruth)
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
            tflops = gemm_tflops(m, n, k, t)
            print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
                  f'{t * 1e6:4.0f} us | '
                  f'{tflops:4.0f} TFLOPS | '
                  f'{tflops:4.0f} FP4-TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def build_offsets_experts_from_masked_m(masked_m: torch.Tensor, num_groups: int, block_m: int = 128):
    offsets = []
    experts = []
    curr_offset = 0
    for g in range(num_groups):
        v = masked_m[g].item()
        if v > 0:
            offsets.append(curr_offset)
            experts.append(g)
            curr_offset += ((v + block_m - 1) // block_m) * block_m
    offsets.append(curr_offset)
    experts.append(-1)
    return (torch.tensor(offsets, dtype=torch.int32, device='cuda'), 
            torch.tensor(experts, dtype=torch.int32, device='cuda'), 
            len(offsets))

def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    def _tensor_to_int_list(x: torch.Tensor):
        return [int(v) for v in x.detach().cpu().tolist()]

    def _print_asym_case_debug(
        tag: str,
        num_groups: int,
        max_m: int,
        expected_m_per_group: int,
        n: int,
        k: int,
        masked_m: torch.Tensor,
        offsets: torch.Tensor,
        experts: torch.Tensor,
        list_size: int,
    ) -> None:
        valid_m = int(masked_m.sum().item())
        print(
            f'[{tag}] num_groups={num_groups}, m_max={max_m}, m_valid={valid_m}, '
            f'expected_m_per_group={expected_m_per_group}, n={n}, k={k}'
        )
        print(f'[{tag}] masked_m={_tensor_to_int_list(masked_m)}')
        print(f'[{tag}] offsets(list_size={list_size})={_tensor_to_int_list(offsets)}')
        print(f'[{tag}] experts(list_size={list_size})={_tensor_to_int_list(experts)}')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0
        # Test correctness
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout)
        offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups)
        
        deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
        # import ipdb
        # ipdb.set_trace()
        d_asym = torch.empty_like(d)
        _print_asym_case_debug(
            'asym_gemm correctness',
            num_groups,
            max_m,
            expected_m_per_group,
            n,
            k,
            masked_m,
            offsets,
            experts,
            list_size,
        )

        b_pinned = (b[0].cpu().pin_memory(), b[1].cpu().pin_memory())

        asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b_pinned, d_asym, offsets, experts, list_size, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

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

        # Output the 5x5 submatrix and neighborhood of max diff for asym_gemm
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

        # Construct full cases
        # a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
        # offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, num_groups)
        # d_asym = torch.empty_like(d)
        # _print_asym_case_debug(
        #     'asym_gemm perf',
        #     num_groups,
        #     max_m,
        #     expected_m_per_group,
        #     n,
        #     k,
        #     masked_m,
        #     offsets,
        #     experts,
        #     list_size,
        # )

        # # noinspection PyShadowingNames
        # def test_func():
        #     deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # # noinspection PyShadowingNames
        # def test_func_asym():
        #     asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(a, b, d_asym, offsets, experts, list_size, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # # Test performance with fixed shapes
        # valid_m = masked_m.sum().item()
        # import ipdb;ipdb.set_trace()
        # t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        # t_asym = bench_kineto(test_func_asym, 'asym_gemm', suppress_kineto_output=True)
        
        # print(f' > Perf ({num_groups=}, expected_m={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): ')
        # if t > 0:
        #     baseline_tflops = gemm_tflops(valid_m, n, k, t)
        #     print(f'   Baseline: {t * 1e6:4.0f} us | {baseline_tflops:4.0f} TFLOPS | {baseline_tflops:4.0f} FP4-TFLOPS')
        # else:
        #     print(f'   Baseline: bench_kineto returned 0, skip')
        # if t_asym > 0:
        #     asym_tflops = gemm_tflops(valid_m, n, k, t_asym)
        #     print(f'   Asym:     {t_asym * 1e6:4.0f} us | {asym_tflops:4.0f} TFLOPS | {asym_tflops:4.0f} FP4-TFLOPS')
        # else:
        #     print(f'   Asym:     bench_kineto returned 0, skip')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    k_grouped_fp8_gemm_contiguous = asym_gemm.k_grouped_fp8_gemm_nt_contiguous if get_arch_major() == 9 \
                                    else asym_gemm.k_grouped_fp8_gemm_tn_contiguous
    for num_groups, m, n, major_a, major_b, ks, expected_k_per_group in enumerate_k_grouped_contiguous(torch.float8_e4m3fn):
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups and len(ks) > 1:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, new_ks, use_ue8m0=use_ue8m0)
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device='cuda')
            k_grouped_fp8_gemm_contiguous(a, b, d, new_ks, new_ks_tensor, c)

            diff = calc_diff(d, ref_d)
            print(f'{m=}, {n=}, {k=}, {ks=}, {diff:.5f}')
            if not test_empty_groups:
                print_5x5_matrix_diff('asym_gemm', d, ref_d)

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, ks, use_ue8m0=use_ue8m0)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            k_grouped_fp8_gemm_contiguous(a, b, d, ks, ks_tensor, c)
    
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        tflops = gemm_tflops(m, n, k, t)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{tflops:4.0f} TFLOPS | '
              f'{tflops:4.0f} FP4-TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
        break
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {asym_gemm.__path__}\n')

    # test_gemm()
    test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
    # test_k_grouped_gemm_contiguous()
