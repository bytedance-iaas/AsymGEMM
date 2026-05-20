import random
import gc
import os
import sys
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS_DIR))

import asym_gemm  # noqa: E402
from generators import (  # noqa: E402
    enumerate_m_grouped_contiguous,
    enumerate_m_grouped_masked,
    generate_m_grouped_contiguous,
    generate_m_grouped_masked,
    get_ue8m0_usage,
)
from test_h20_bf16 import build_offsets_experts_from_m_indices_pairs as build_bf16_offsets  # noqa: E402
from test_h20_fp8 import build_offsets_experts_from_m_indices_pairs as build_fp8_offsets  # noqa: E402

try:
    import deep_gemm  # noqa: E402
except ImportError:
    deep_gemm = None


WARMUPS = 3
ITERS = 10


def tensor_nbytes(t):
    return t.numel() * t.element_size()


def tree_nbytes(x):
    if isinstance(x, torch.Tensor):
        return tensor_nbytes(x)
    return sum(tree_nbytes(item) for item in x)


def fmt_bytes(num_bytes):
    num = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB")
    for unit in units:
        if abs(num) < 1024.0 or unit == units[-1]:
            return f"{num:.2f} {unit}"
        num /= 1024.0


def clear_cuda_memory_stats():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cuda_allocated():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def time_cuda_us(fn, warmups=WARMUPS, iters=ITERS):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()

    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def first_k_major_contiguous(dtype):
    for item in enumerate_m_grouped_contiguous(dtype):
        _, _, _, _, _, major_a, major_b = item
        if major_a.is_k_major() and major_b.is_k_major():
            return item
    raise RuntimeError(f"no K-major contiguous case for {dtype}")


def first_masked(dtype):
    for item in enumerate_m_grouped_masked(dtype):
        if not item[-1]:
            return item
    raise RuntimeError(f"no masked case for {dtype}")


def speedup(baseline_us, candidate_us):
    return baseline_us / candidate_us if candidate_us > 0 else float("nan")


def print_ratio(label, baseline_us, candidate_us):
    print(f"    {label}: {speedup(baseline_us, candidate_us):.2f}x")


def print_hbm_delta(label, without_b, with_b, expected_b_bytes):
    measured = with_b - without_b
    tolerance = max(1 << 20, int(expected_b_bytes * 0.02))
    if measured <= 0 or abs(measured - expected_b_bytes) > tolerance:
        raise AssertionError(
            f"{label}: measured GPU B delta {fmt_bytes(measured)} does not match "
            f"expected {fmt_bytes(expected_b_bytes)}"
        )
    print(f"  {label}")
    print(f"    live CUDA alloc without GPU B: {fmt_bytes(without_b)}")
    print(f"    live CUDA alloc with GPU B:    {fmt_bytes(with_b)}")
    print(f"    measured GPU B delta:         {fmt_bytes(measured)}")
    print(f"    expected B tensor bytes:      {fmt_bytes(expected_b_bytes)}")


def memory_bf16_contiguous():
    _, num_groups, expected_m, n, k, major_a, major_b = first_k_major_contiguous(torch.bfloat16)
    m, a, b_gpu, m_indices, d, _ = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, major_a, major_b, use_bf16=True
    )
    b_bytes = tensor_nbytes(b_gpu)
    b_pinned = b_gpu.detach().to("cpu", non_blocking=False).pin_memory()
    offsets, experts, list_size = build_bf16_offsets(m_indices)
    del b_gpu

    clear_cuda_memory_stats()
    without_b = cuda_allocated()
    d_asym = torch.empty_like(d)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        a, b_pinned, d_asym, offsets, experts, list_size, "nk"
    )
    torch.cuda.synchronize()
    without_b_after_call = cuda_allocated()
    b_gpu_resident = b_pinned.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    with_b = cuda_allocated()

    active_m = int((m_indices != -1).sum().item())
    print_hbm_delta(
        f"BF16 contiguous pinned-B: groups={num_groups}, active_m={active_m}, n={n}, k={k}",
        without_b_after_call,
        with_b,
        b_bytes,
    )
    print(f"    B is CPU pinned in asym path:  {b_pinned.is_pinned()}")
    del b_gpu_resident


def memory_bf16_masked():
    _, _, num_groups, max_m, expected_m, n, k, _ = first_masked(torch.bfloat16)
    a, b_gpu, masked_m, _, d, _ = generate_m_grouped_masked(
        num_groups, max_m, expected_m, n, k, use_bf16=True, use_psum_layout=False
    )
    b_bytes = tensor_nbytes(b_gpu)
    b_pinned = b_gpu.detach().to("cpu", non_blocking=False).pin_memory()
    del b_gpu

    clear_cuda_memory_stats()
    without_b = cuda_allocated()
    d_asym = torch.empty_like(d)
    asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
        a, b_pinned, d_asym, masked_m, expected_m, compiled_dims="nk"
    )
    torch.cuda.synchronize()
    without_b_after_call = cuda_allocated()
    b_gpu_resident = b_pinned.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    with_b = cuda_allocated()

    active_m = int(masked_m.sum().item())
    print_hbm_delta(
        f"BF16 masked pinned-B: groups={num_groups}, active_m={active_m}, n={n}, k={k}",
        without_b_after_call,
        with_b,
        b_bytes,
    )
    print(f"    B is CPU pinned in asym path:  {b_pinned.is_pinned()}")
    del b_gpu_resident


def memory_fp8_contiguous():
    kernel_type, num_groups, expected_m, n, k, major_a, major_b = first_k_major_contiguous(torch.float8_e4m3fn)
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    m, a, b_gpu, m_indices, d, _ = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, major_a, major_b, use_ue8m0=use_ue8m0
    )
    b_bytes = tree_nbytes(b_gpu)
    bf16_b_bytes = num_groups * n * k * torch.tensor([], dtype=torch.bfloat16).element_size()
    b_cpu = tuple(t.detach().to("cpu", non_blocking=False).pin_memory() for t in b_gpu)
    del b_gpu

    clear_cuda_memory_stats()
    without_b = cuda_allocated()
    b_gpu_resident = tuple(t.to("cuda", non_blocking=False) for t in b_cpu)
    torch.cuda.synchronize()
    with_b = cuda_allocated()

    active_m = int((m_indices != -1).sum().item())
    print_hbm_delta(
        f"FP8 contiguous current GPU-B API: groups={num_groups}, active_m={active_m}, n={n}, k={k}",
        without_b,
        with_b,
        b_bytes,
    )
    print(f"    equivalent BF16 B bytes:      {fmt_bytes(bf16_b_bytes)}")
    print(f"    FP8 B HBM vs BF16 B:          {b_bytes / bf16_b_bytes:.2%}")
    del b_gpu_resident


def memory_fp8_masked():
    kernel_type, _, num_groups, max_m, expected_m, n, k, _ = first_masked(torch.float8_e4m3fn)
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    a, b_gpu, masked_m, _, d, _ = generate_m_grouped_masked(
        num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0, use_psum_layout=False
    )
    b_bytes = tree_nbytes(b_gpu)
    bf16_b_bytes = num_groups * n * k * torch.tensor([], dtype=torch.bfloat16).element_size()
    b_cpu = tuple(t.detach().to("cpu", non_blocking=False).pin_memory() for t in b_gpu)
    del b_gpu

    clear_cuda_memory_stats()
    without_b = cuda_allocated()
    b_gpu_resident = tuple(t.to("cuda", non_blocking=False) for t in b_cpu)
    torch.cuda.synchronize()
    with_b = cuda_allocated()

    active_m = int(masked_m.sum().item())
    print_hbm_delta(
        f"FP8 masked current GPU-B API: groups={num_groups}, active_m={active_m}, n={n}, k={k}",
        without_b,
        with_b,
        b_bytes,
    )
    print(f"    equivalent BF16 B bytes:      {fmt_bytes(bf16_b_bytes)}")
    print(f"    FP8 B HBM vs BF16 B:          {b_bytes / bf16_b_bytes:.2%}")
    del b_gpu_resident


def memory_report():
    print("\nH200 grouped HBM residency smoke:")
    memory_bf16_contiguous()
    memory_bf16_masked()
    memory_fp8_contiguous()
    memory_fp8_masked()


def benchmark_bf16_contiguous():
    _, num_groups, expected_m, n, k, major_a, major_b = first_k_major_contiguous(torch.bfloat16)
    m, a, b_gpu, m_indices, d, _ = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, major_a, major_b, use_bf16=True
    )
    b_pinned = b_gpu.detach().to("cpu", non_blocking=False).pin_memory()
    d_asym = torch.empty_like(d)
    d_torch = torch.empty_like(d)
    b_staging = torch.empty_like(b_gpu)
    offsets, experts, list_size = build_bf16_offsets(m_indices)
    offsets_h = offsets.cpu().tolist()
    experts_h = experts.cpu().tolist()

    def asym_fn():
        asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
            a, b_pinned, d_asym, offsets, experts, list_size, "nk"
        )

    def torch_loop_with_b(b_src):
        for i, expert in enumerate(experts_h[:-1]):
            start = offsets_h[2 * i]
            end = offsets_h[2 * i + 1]
            d_torch[start:end].copy_(a[start:end] @ b_src[expert].t())

    def torch_loop_gpu_fn():
        torch_loop_with_b(b_gpu)

    def torch_loop_copy_fn():
        b_staging.copy_(b_pinned, non_blocking=True)
        torch_loop_with_b(b_staging)

    asym_us = time_cuda_us(asym_fn)
    torch_gpu_us = time_cuda_us(torch_loop_gpu_fn)
    torch_copy_us = time_cuda_us(torch_loop_copy_fn)

    active_m = int((m_indices != -1).sum().item())
    print(f"  BF16 contiguous: groups={num_groups}, active_m={active_m}, n={n}, k={k}")
    print(f"    asym pinned-B API:       {asym_us:9.1f} us")
    print(f"    torch loop, B on GPU:    {torch_gpu_us:9.1f} us")
    print(f"    torch loop, copy B first:{torch_copy_us:9.1f} us")
    print_ratio("speedup vs torch GPU-B loop", torch_gpu_us, asym_us)
    print_ratio("speedup vs torch copy-B loop", torch_copy_us, asym_us)


def benchmark_bf16_masked():
    _, _, num_groups, max_m, expected_m, n, k, _ = first_masked(torch.bfloat16)
    a, b_gpu, masked_m, _, d, _ = generate_m_grouped_masked(
        num_groups, max_m, expected_m, n, k, use_bf16=True, use_psum_layout=False
    )
    b_pinned = b_gpu.detach().to("cpu", non_blocking=False).pin_memory()
    d_asym = torch.empty_like(d)
    d_torch = torch.empty_like(d)
    b_staging = torch.empty_like(b_gpu)
    masked_h = [int(v) for v in masked_m.cpu().tolist()]

    def asym_fn():
        asym_gemm.m_grouped_bf16_asym_gemm_nt_masked(
            a, b_pinned, d_asym, masked_m, expected_m, compiled_dims="nk"
        )

    def torch_loop_with_b(b_src):
        for group, actual_m in enumerate(masked_h):
            if actual_m > 0:
                d_torch[group, :actual_m].copy_(a[group, :actual_m] @ b_src[group].t())

    def torch_loop_gpu_fn():
        torch_loop_with_b(b_gpu)

    def torch_loop_copy_fn():
        b_staging.copy_(b_pinned, non_blocking=True)
        torch_loop_with_b(b_staging)

    asym_us = time_cuda_us(asym_fn)
    torch_gpu_us = time_cuda_us(torch_loop_gpu_fn)
    torch_copy_us = time_cuda_us(torch_loop_copy_fn)

    active_m = int(masked_m.sum().item())
    print(f"  BF16 masked: groups={num_groups}, active_m={active_m}, n={n}, k={k}")
    print(f"    asym pinned-B API:       {asym_us:9.1f} us")
    print(f"    torch loop, B on GPU:    {torch_gpu_us:9.1f} us")
    print(f"    torch loop, copy B first:{torch_copy_us:9.1f} us")
    print_ratio("speedup vs torch GPU-B loop", torch_gpu_us, asym_us)
    print_ratio("speedup vs torch copy-B loop", torch_copy_us, asym_us)


def benchmark_fp8_contiguous():
    if deep_gemm is None:
        print("  FP8 contiguous: deep_gemm unavailable, skipping baseline timing")
        return

    kernel_type, num_groups, expected_m, n, k, major_a, major_b = first_k_major_contiguous(torch.float8_e4m3fn)
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0
    recipe = (1, 128, 128)

    m, a, b, m_indices, d, _ = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k, major_a, major_b, use_ue8m0=use_ue8m0
    )
    d_asym = torch.empty_like(d, dtype=torch.float)
    d_deep = torch.empty_like(d)
    offsets, experts, list_size = build_fp8_offsets(m_indices)

    def asym_fn():
        asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
            a, b, d_asym, offsets, experts, list_size,
            recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast
        )

    def deep_fn():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d_deep, m_indices, recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast
        )

    asym_us = time_cuda_us(asym_fn)
    deep_us = time_cuda_us(deep_fn)

    active_m = int((m_indices != -1).sum().item())
    print(f"  FP8 contiguous: groups={num_groups}, active_m={active_m}, n={n}, k={k}")
    print(f"    asym FP32-output API:    {asym_us:9.1f} us")
    print(f"    deep_gemm BF16 baseline: {deep_us:9.1f} us")
    print_ratio("speedup vs deep_gemm baseline", deep_us, asym_us)


def benchmark_fp8_masked():
    if deep_gemm is None:
        print("  FP8 masked: deep_gemm unavailable, skipping baseline timing")
        return

    kernel_type, _, num_groups, max_m, expected_m, n, k, _ = first_masked(torch.float8_e4m3fn)
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0
    recipe = (1, 128, 128)

    a, b, masked_m, _, d, _ = generate_m_grouped_masked(
        num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0, use_psum_layout=False
    )
    d_asym = torch.empty_like(d, dtype=torch.float)
    d_deep = torch.empty_like(d)

    def asym_fn():
        asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
            a, b, d_asym, masked_m, expected_m,
            recipe=recipe, disable_ue8m0_cast=disable_ue8m0_cast
        )

    def deep_fn():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a, b, d_deep, masked_m, expected_m, disable_ue8m0_cast=disable_ue8m0_cast
        )

    asym_us = time_cuda_us(asym_fn)
    deep_us = time_cuda_us(deep_fn)

    active_m = int(masked_m.sum().item())
    print(f"  FP8 masked: groups={num_groups}, active_m={active_m}, n={n}, k={k}")
    print(f"    asym FP32-output API:    {asym_us:9.1f} us")
    print(f"    deep_gemm BF16 baseline: {deep_us:9.1f} us")
    print_ratio("speedup vs deep_gemm baseline", deep_us, asym_us)


def main():
    if torch.cuda.get_device_capability()[0] != 9:
        print("Skipping H200 smoke report: SM90/H200 GPU is required")
        return

    if os.environ.get("RUN_MEMORY", "1") != "0":
        random.seed(0)
        torch.manual_seed(0)
        memory_report()

    if os.environ.get("RUN_TIMING", "1") != "0":
        random.seed(0)
        torch.manual_seed(0)
        print("\nH200 grouped timing smoke (warm API calls, no pass/fail threshold):")
        benchmark_bf16_contiguous()
        benchmark_bf16_masked()
        benchmark_fp8_contiguous()
        benchmark_fp8_masked()


if __name__ == "__main__":
    main()
