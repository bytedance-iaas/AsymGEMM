# tests/test_sm80_moe.py
"""
Correctness test for sm80_moe_gemm_nt_contiguous.
Reference: per-expert torch.matmul in float32.

Run:
    python tests/test_sm80_moe.py
"""
import itertools
import os
import sys
import gc

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm


# ── Reference implementation ──────────────────────────────────────────────────

@torch.no_grad()
def ref_moe_gemm(a, b, experts, offsets):
    """
    a:       [total_tokens, K]   fp16 or bf16
    b:       [num_experts, N, K] fp16 or bf16
    experts: [list_size]         int32 — expert IDs
    offsets: [list_size]         int32 — cumulative end indices
    returns: [total_tokens, N]   fp32
    """
    total_tokens, K = a.shape
    num_experts, N, K_ = b.shape
    assert K == K_, "K mismatch"

    out = torch.zeros(total_tokens, N, dtype=torch.float32, device=a.device)
    start = 0
    elist = experts.tolist() if isinstance(experts, torch.Tensor) else experts
    ilist = offsets.tolist() if isinstance(offsets, torch.Tensor) else offsets

    for i, expert_id in enumerate(elist):
        end = ilist[i]
        out[start:end] = a[start:end].float() @ b[expert_id].float().t()
        start = end
    return out


def calc_diff(a, b):
    """Root-mean-squared relative diff after scaling by max-abs."""
    scale = b.abs().max().clamp(min=1e-6)
    return ((a - b) / scale).norm() / (a.numel() ** 0.5)


def estimate_case_bytes(num_experts, total_tokens, n, k, dtype: torch.dtype):
    elem_size = torch.empty((), dtype=dtype).element_size()
    int_size = torch.empty((), dtype=torch.int32).element_size()
    x_bytes = total_tokens * k * elem_size
    w_bytes = num_experts * n * k * elem_size
    o_bytes = total_tokens * n * elem_size
    ref_bytes = total_tokens * n * torch.empty((), dtype=torch.float32).element_size()
    list_bytes = (total_tokens + num_experts) * int_size
    return x_bytes + w_bytes + o_bytes + ref_bytes + list_bytes


# ── Test cases ─────────────────────────────────────────────────────────────────
# (num_experts, N, K, token_counts_per_active_expert)
TEST_CASES = [
    # Single BLOCK_K tile (k=0 path)
    (8,  4096,   256, [12, 8, 20, 28]),
    # Two BLOCK_K tiles (k>0 accumulation)
    (8,  4096,   512, [128, 64, 256, 12]),
    # 16 K-tiles — common production shape
    (8,  4096,  4096, [256, 128, 512, 64, 300, 100, 200, 400]),
    # 28 K-tiles — DeepSeek FFN hidden dim
    (8,  4096,  7168, [128, 256, 64, 192]),
    # K=128 → block_k halving heuristic
    (8,  4096,   128, [100, 200, 50, 150]),
    # Large N — exercises block_n=128 path
    (8, 16384,  4096, [64, 128, 32, 96]),
    # Partial M-tiles — boundary predication
    (4,  4096,   256, [7, 13, 31, 127]),
    # Single expert
    (4,  4096,  4096, [512]),
]


@torch.no_grad()
def test_moe_gemm(dtype: torch.dtype):
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    kernel_fn = asym_gemm.m_grouped_moe_gemm_nt_contiguous

    all_passed = True
    for (num_experts, N, K, token_counts) in TEST_CASES:
        torch.cuda.empty_cache()
        gc.collect()

        expert_ids   = list(range(len(token_counts)))
        total_tokens = sum(token_counts)
        offsets_h    = list(itertools.accumulate(token_counts))  # cumulative end indices
        required_bytes = int(estimate_case_bytes(num_experts, total_tokens, N, K, dtype) * 1.15)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes:
            print(f"[SKIP] {dtype_str}  N={N:5d} K={K:5d} tokens={total_tokens:5d} "
                  f"experts={len(token_counts)}  free_mem={free_bytes / 2**20:.1f} MiB "
                  f"required~={required_bytes / 2**20:.1f} MiB")
            continue

        a = b = d = experts = offsets = ref = None
        try:
            a       = torch.randn(total_tokens, K,    dtype=dtype,       device='cuda')
            b       = torch.randn(num_experts,  N, K, dtype=dtype,       device='cuda')
            d       = torch.empty(total_tokens, N,    dtype=dtype,       device='cuda')
            experts = torch.tensor(expert_ids, dtype=torch.int32, device='cuda')
            offsets = torch.tensor(offsets_h,  dtype=torch.int32, device='cuda')
            list_size = experts.numel()

            kernel_fn(a, b, d, offsets, experts, list_size)
            torch.cuda.synchronize()

            ref = ref_moe_gemm(a, b, experts, offsets)  # float32 reference

            diff = calc_diff(d.float(), ref)
            threshold = 0.01  # relative, normalised
            status = "PASS" if diff < threshold else "FAIL"
            print(f"[{status}] {dtype_str}  N={N:5d} K={K:5d} "
                  f"tokens={total_tokens:5d} experts={len(token_counts)}  diff={diff:.6f}")
            if diff >= threshold:
                all_passed = False
        finally:
            del a, b, d, experts, offsets, ref
            torch.cuda.empty_cache()
            gc.collect()

    return all_passed


if __name__ == '__main__':
    torch.manual_seed(42)
    ok_fp16 = test_moe_gemm(torch.float16)
    ok_bf16 = test_moe_gemm(torch.bfloat16)
    if ok_fp16 and ok_bf16:
        print("\nAll tests passed.")
    else:
        raise SystemExit("One or more tests FAILED.")
