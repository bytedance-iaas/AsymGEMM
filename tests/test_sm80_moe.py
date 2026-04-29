# tests/test_sm80_moe.py
"""
Correctness test for m_grouped_fp8_asym_gemm_sm80.
Reference: per-expert torch.matmul in float32, scaled by scale_a * scale_b.

Run:
    python tests/test_sm80_moe.py
"""
import gc
import itertools
import os
import sys

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


@torch.no_grad()
def test_fp8_moe_gemm():
    print("Testing m_grouped_fp8_asym_gemm_sm80 (SM89 native FP8 MMA):")
    kernel_fn = asym_gemm.m_grouped_fp8_asym_gemm_sm80

    # Non-trivial scales verify that scale is actually applied
    scale_a, scale_b = 0.5, 2.0

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
                  f"experts={len(token_counts)}  "
                  f"free={free_bytes/2**20:.1f} MiB  needed~={required_bytes/2**20:.1f} MiB")
            continue

        a = b = d = experts_t = offsets_t = ref = None
        try:
            # Random BF16 in [-1, 1], cast to FP8 E4M3 (max representable ≈ 448)
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

            ref = ref_fp8_moe_gemm(a, b, scale_a, scale_b, experts_t, offsets_t)

            diff = calc_diff(d.float(), ref)
            threshold = 0.01
            status = "PASS" if diff < threshold else "FAIL"
            print(f"[{status}]  N={N:5d} K={K:5d} tokens={total_tokens:5d} "
                  f"experts={len(token_counts)}  diff={diff:.6f}")
            if diff >= threshold:
                all_passed = False
        finally:
            del a, b, d, experts_t, offsets_t, ref
            torch.cuda.empty_cache()
            gc.collect()

    return all_passed


if __name__ == "__main__":
    torch.manual_seed(42)
    ok = test_fp8_moe_gemm()
    if ok:
        print("\nAll tests passed.")
    else:
        raise SystemExit("One or more tests FAILED.")
