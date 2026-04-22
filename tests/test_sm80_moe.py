# tests/test_sm80_moe.py
"""
Correctness test for sm80_moe_gemm_nt_contiguous.
Reference: per-expert torch.matmul in float32.

Run:
    python tests/test_sm80_moe.py
"""
import itertools
import torch
import asym_gemm


# ── Reference implementation ──────────────────────────────────────────────────

@torch.no_grad()
def ref_moe_gemm(x, w, expert_list, index_list):
    """
    x:           [total_tokens, K]   fp16 or bf16
    w:           [num_experts, N, K] fp16 or bf16
    expert_list: [list_size]         int32 tensor or list
    index_list:  [list_size]         int32 tensor or list, cumulative end indices
    returns:     [total_tokens, N]   fp32
    """
    total_tokens, K = x.shape
    num_experts, N, K_ = w.shape
    assert K == K_, "K mismatch"

    out = torch.zeros(total_tokens, N, dtype=torch.float32, device=x.device)
    start = 0
    elist = expert_list.tolist() if isinstance(expert_list, torch.Tensor) else expert_list
    ilist = index_list.tolist()  if isinstance(index_list,  torch.Tensor) else index_list

    for i, expert_id in enumerate(elist):
        end = ilist[i]
        out[start:end] = x[start:end].float() @ w[expert_id].float().t()
        start = end
    return out


def calc_diff(a, b):
    """Root-mean-squared relative diff after scaling by max-abs."""
    scale = b.abs().max().clamp(min=1e-6)
    return ((a - b) / scale).norm() / (a.numel() ** 0.5)


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
        expert_ids   = list(range(len(token_counts)))   # experts 0..list_size-1
        total_tokens = sum(token_counts)
        index_list_h = list(itertools.accumulate(token_counts))  # cumulative end indices

        x           = torch.randn(total_tokens, K,    dtype=dtype,       device='cuda')
        w           = torch.randn(num_experts,  N, K, dtype=dtype,       device='cuda')
        o           = torch.empty(total_tokens, N,    dtype=dtype,       device='cuda')
        expert_list = torch.tensor(expert_ids,        dtype=torch.int32, device='cuda')
        index_list  = torch.tensor(index_list_h,      dtype=torch.int32, device='cuda')

        kernel_fn(x, w, o, expert_list, index_list)
        torch.cuda.synchronize()

        ref = ref_moe_gemm(x, w, expert_list, index_list)  # float32 reference

        diff = calc_diff(o.float(), ref)
        threshold = 0.01  # relative, normalised
        status = "PASS" if diff < threshold else "FAIL"
        print(f"[{status}] {dtype_str}  N={N:5d} K={K:5d} "
              f"tokens={total_tokens:5d} experts={len(token_counts)}  diff={diff:.6f}")
        if diff >= threshold:
            all_passed = False

    return all_passed


if __name__ == '__main__':
    torch.manual_seed(42)
    ok_fp16 = test_moe_gemm(torch.float16)
    ok_bf16 = test_moe_gemm(torch.bfloat16)
    if ok_fp16 and ok_bf16:
        print("\nAll tests passed.")
    else:
        raise SystemExit("One or more tests FAILED.")
