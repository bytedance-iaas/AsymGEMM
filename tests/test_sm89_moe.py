# tests/test_sm89_moe.py
"""
Correctness + performance test for m_grouped_fp8_asym_gemm_sm89.
Reference: per-expert torch.matmul in float32, scaled by scale_a * scale_b.

Run:
    python tests/test_sm89_moe.py            # correctness + benchmark
    python tests/test_sm89_moe.py --perf     # benchmark only (faster)
"""
import gc
import itertools
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm


# ── Reference implementation ──────────────────────────────────────────────────

@torch.no_grad()
def ref_fp8_moe_gemm(a_fp8, b_fp8, scale_a, scale_b, experts, offsets):
    """
    Dequantize FP8 inputs and compute MoE GEMM in float32.
    Matches kernel semantics: d = (a_fp8 @ b_fp8.T) * scale_a * scale_b

    a_fp8:   [total_tokens, K]    torch.float8_e4m3fn
    b_fp8:   [num_experts, N, K]  torch.float8_e4m3fn
    experts: [list_size]          int32 — expert IDs
    offsets: [list_size]          int32 — cumulative end-token indices
    returns: [total_tokens, N]    torch.float32
    """
    a_f32 = a_fp8.float()
    b_f32 = b_fp8.float()
    combined = scale_a * scale_b

    total_tokens = a_f32.shape[0]
    _, N, _ = b_f32.shape
    out = torch.zeros(total_tokens, N, dtype=torch.float32, device=a_fp8.device)

    elist = experts.tolist()
    ilist = offsets.tolist()
    start = 0
    for i, expert_id in enumerate(elist):
        end = ilist[i]
        out[start:end] = (a_f32[start:end] @ b_f32[expert_id].t()) * combined
        start = end
    return out


def calc_diff(a, b):
    """Root-mean-squared relative diff normalised by max-abs of reference."""
    scale = b.abs().max().clamp(min=1e-6)
    return ((a - b) / scale).norm() / (a.numel() ** 0.5)


def estimate_case_bytes(num_experts, total_tokens, n, k):
    fp8_size = 1   # float8_e4m3fn: 1 byte/elem
    bf16_size = 2  # bfloat16:      2 bytes/elem
    int_size  = 4  # int32:         4 bytes/elem
    x_bytes   = total_tokens * k * fp8_size
    w_bytes   = num_experts * n * k * fp8_size
    o_bytes   = total_tokens * n * bf16_size
    ref_bytes = total_tokens * n * 4   # fp32 reference
    list_bytes = (total_tokens + num_experts) * int_size
    return x_bytes + w_bytes + o_bytes + ref_bytes + list_bytes


def moe_flops(total_tokens, N, K):
    """
    Effective FLOPs for MoE GEMM.
    Each of the total_tokens active tokens does one dot-product of length K
    against N output features:  2 * total_tokens * N * K  (multiply-add).
    Experts that receive no tokens contribute 0 FLOPs.
    """
    return 2 * total_tokens * N * K


def time_kernel(kernel_fn, a, b, d, offsets_t, experts_t, list_size,
                scale_a, scale_b, num_warmup=10, num_iters=50):
    """
    Returns median kernel wall-time in milliseconds using CUDA events.
    """
    # Warm-up: allow JIT compilation and cache warming
    for _ in range(num_warmup):
        kernel_fn(a, b, d, offsets_t, experts_t, list_size,
                  scale_a=scale_a, scale_b=scale_b)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(num_iters):
        start_ev.record()
        kernel_fn(a, b, d, offsets_t, experts_t, list_size,
                  scale_a=scale_a, scale_b=scale_b)
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev))

    times_ms.sort()
    # Return median (robust to occasional scheduling spikes)
    return times_ms[len(times_ms) // 2]


# ── Test cases ─────────────────────────────────────────────────────────────────
# All K values are multiples of 32 (SM89 FP8 MMA K-atom constraint).
# (num_experts, N, K, token_counts_per_active_expert)
TEST_CASES = [
    # Single BLOCK_K tile (k_max = 1)
    (8,  4096,   256, [12, 8, 20, 28]),
    # Two BLOCK_K tiles
    (8,  4096,   512, [128, 64, 256, 12]),
    # 16 K-tiles — common production shape
    (8,  4096,  4096, [256, 128, 512, 64, 300, 100, 200, 400]),
    # 28 K-tiles — DeepSeek FFN hidden dim
    (8,  4096,  7168, [128, 256, 64, 192]),
    # K=128 — 4 K-tiles at BLOCK_K=32
    (8,  4096,   128, [100, 200, 50, 150]),
    # Large N — exercises block_n=128 path
    (8, 16384,  4096, [64, 128, 32, 96]),
    # Partial M-tiles — boundary predication
    (4,  4096,   256, [7, 13, 31, 127]),
    # Single expert
    (4,  4096,  4096, [512]),
]

# ── Benchmark cases ─────────────────────────────────────────────────────────
# Larger / more production-representative shapes for TFLOPS measurement.
# (num_experts, N, K, token_counts_per_active_expert, label)
BENCH_CASES = [
    # DeepSeek-V3 / MLA-style: 256 experts, top-8 active, 2048 tokens/expert
    (256, 7168, 4096, [2048] * 8,   "DS-V3 e256/top8  M=16384"),
    # Smaller token budget (decode phase)
    (256, 7168, 4096, [16]   * 8,   "DS-V3 e256/top8  M=128  (decode)"),
    # Mixtral-8x7B-style: 8 experts, top-2, varied load
    (8,  14336, 4096, [512, 256, 128, 64, 32, 16, 8, 4],
                                    "Mixtral-8x7B e8  M=1020"),
    # Wide N (projection up)
    (8,  28672, 4096, [256] * 8,    "Wide-N e8        M=2048"),
    # Tall K (projection down-up fusion)
    (8,   4096, 7168, [256] * 8,    "Tall-K e8        M=2048"),
    # Single large expert (sanity / baseline for serial behaviour)
    (1,   7168, 4096, [4096],       "Single expert    M=4096"),
]


@torch.no_grad()
def test_fp8_moe_gemm(with_timing: bool = True):
    print("=" * 72)
    print("Correctness: m_grouped_fp8_asym_gemm_sm89 (SM89 native FP8 MMA)")
    print("=" * 72)
    kernel_fn = asym_gemm.m_grouped_fp8_asym_gemm_sm89

    # Non-trivial scales verify that scale is actually applied
    scale_a, scale_b = 0.5, 2.0

    header = (f"{'Status':<6}  {'N':>6} {'K':>5} {'tokens':>7} "
              f"{'experts':>7}  {'diff':>10}")
    if with_timing:
        header += f"  {'ms':>7}  {'TFLOPS':>7}"
    print(header)
    print("-" * len(header))

    all_passed = True
    for (num_experts, N, K, token_counts) in TEST_CASES:
        torch.cuda.empty_cache()
        gc.collect()

        expert_ids   = list(range(len(token_counts)))
        total_tokens = sum(token_counts)
        offsets_h    = list(itertools.accumulate(token_counts))

        required_bytes = int(estimate_case_bytes(num_experts, total_tokens, N, K) * 1.15)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes:
            print(f"[SKIP]  N={N:5d} K={K:5d} tokens={total_tokens:5d} "
                  f"experts={len(token_counts):>5}  "
                  f"free={free_bytes/2**20:.1f} MiB  needed~={required_bytes/2**20:.1f} MiB")
            continue

        a = b = d = experts_t = offsets_t = ref = None
        try:
            a_bf16 = torch.randn(total_tokens, K,
                                 dtype=torch.bfloat16, device="cuda").clamp(-1, 1)
            b_bf16 = torch.randn(num_experts, N, K,
                                 dtype=torch.bfloat16, device="cuda").clamp(-1, 1)
            a = a_bf16.to(torch.float8_e4m3fn)
            b = b_bf16.to(torch.float8_e4m3fn)
            d = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")

            experts_t = torch.tensor(expert_ids, dtype=torch.int32, device="cuda")
            offsets_t = torch.tensor(offsets_h,  dtype=torch.int32, device="cuda")

            kernel_fn(a, b, d, offsets_t, experts_t, len(expert_ids),
                      scale_a=scale_a, scale_b=scale_b)
            torch.cuda.synchronize()

            ref  = ref_fp8_moe_gemm(a, b, scale_a, scale_b, experts_t, offsets_t)
            diff = calc_diff(d.float(), ref)

            threshold = 0.01
            status = "PASS" if diff < threshold else "FAIL"
            if diff >= threshold:
                all_passed = False

            row = (f"[{status}]  {N:>6} {K:>5} {total_tokens:>7} "
                   f"{len(token_counts):>7}  {diff:>10.6f}")

            if with_timing:
                ms = time_kernel(kernel_fn, a, b, d, offsets_t, experts_t,
                                 len(expert_ids), scale_a, scale_b,
                                 num_warmup=5, num_iters=30)
                flops   = moe_flops(total_tokens, N, K)
                tflops  = flops / (ms * 1e-3) / 1e12
                row += f"  {ms:>7.3f}  {tflops:>7.3f}"

            print(row)
        finally:
            del a, b, d, experts_t, offsets_t, ref
            torch.cuda.empty_cache()
            gc.collect()

    return all_passed


@torch.no_grad()
def benchmark_fp8_moe_gemm(num_warmup: int = 10, num_iters: int = 100):
    """
    Standalone TFLOPS benchmark across production-representative shapes.

    TFLOPS formula
    ──────────────
    Each of the `total_tokens` active tokens performs a dot-product of
    length K against N output channels, yielding one multiply-add per
    element:

        FLOPs = 2 × total_tokens × N × K

    Experts that receive no tokens contribute 0 FLOPs.  This counts only
    the compute work done by the active (selected) experts.

    Reported time is the **median** of `num_iters` CUDA-event measurements
    after `num_warmup` warm-up iterations.
    """
    print()
    print("=" * 72)
    print("Performance: m_grouped_fp8_asym_gemm_sm89  "
          f"(warmup={num_warmup}, iters={num_iters})")
    print("=" * 72)
    kernel_fn = asym_gemm.m_grouped_fp8_asym_gemm_sm89
    scale_a, scale_b = 1.0, 1.0

    col_w = [38, 8, 8, 8, 8, 9, 8]
    header = (f"{'Case':<{col_w[0]}}  {'N':>{col_w[1]}} {'K':>{col_w[2]}} "
              f"{'tokens':>{col_w[3]}} {'experts':>{col_w[4]}}  "
              f"{'ms':>{col_w[5]}}  {'TFLOPS':>{col_w[6]}}")
    print(header)
    print("-" * len(header))

    for (num_experts, N, K, token_counts, label) in BENCH_CASES:
        torch.cuda.empty_cache()
        gc.collect()

        expert_ids   = list(range(len(token_counts)))
        total_tokens = sum(token_counts)
        offsets_h    = list(itertools.accumulate(token_counts))

        # Rough memory estimate (no ref tensor needed for benchmark)
        mem_bytes = (total_tokens * K            # a FP8
                     + num_experts * N * K       # b FP8
                     + total_tokens * N * 2)     # d BF16
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < int(mem_bytes * 1.2):
            print(f"[SKIP]  {label}  — not enough GPU memory "
                  f"({free_bytes/2**20:.0f} MiB free, "
                  f"~{mem_bytes/2**20:.0f} MiB needed)")
            continue

        a = b = d = experts_t = offsets_t = None
        try:
            a_bf16 = torch.randn(total_tokens, K,
                                 dtype=torch.bfloat16, device="cuda").clamp(-1, 1)
            b_bf16 = torch.randn(num_experts, N, K,
                                 dtype=torch.bfloat16, device="cuda").clamp(-1, 1)
            a = a_bf16.to(torch.float8_e4m3fn)
            b = b_bf16.to(torch.float8_e4m3fn)
            d = torch.empty(total_tokens, N, dtype=torch.bfloat16, device="cuda")

            experts_t = torch.tensor(expert_ids, dtype=torch.int32, device="cuda")
            offsets_t = torch.tensor(offsets_h,  dtype=torch.int32, device="cuda")

            ms = time_kernel(kernel_fn, a, b, d, offsets_t, experts_t,
                             len(expert_ids), scale_a, scale_b,
                             num_warmup=num_warmup, num_iters=num_iters)

            flops  = moe_flops(total_tokens, N, K)
            tflops = flops / (ms * 1e-3) / 1e12

            print(f"{label:<{col_w[0]}}  {N:>{col_w[1]}} {K:>{col_w[2]}} "
                  f"{total_tokens:>{col_w[3]}} {len(token_counts):>{col_w[4]}}  "
                  f"{ms:>{col_w[5]}.3f}  {tflops:>{col_w[6]}.3f}")
        finally:
            del a, b, d, experts_t, offsets_t
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true",
                        help="Run benchmark only, skip correctness test")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warm-up iterations for benchmark (default: 10)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations for benchmark (default: 100)")
    args = parser.parse_args()

    torch.manual_seed(42)

    if not args.perf:
        ok = test_fp8_moe_gemm(with_timing=True)
        if ok:
            print("\nAll correctness tests passed.")
        else:
            raise SystemExit("One or more correctness tests FAILED.")

    benchmark_fp8_moe_gemm(num_warmup=args.warmup, num_iters=args.iters)
