# tests/test_sm89_block_scale.py
"""
Correctness tests for SM89 FP8 MoE GEMM with native block scales
(1x128 activation, 128x128 weight).

Covers:
  1. contiguous kernel vs float32 block-dequant reference
  2. masked kernel vs float32 block-dequant reference
  3. block path vs legacy per-token path (broadcast-equivalent scales)
  4. arch-agnostic dispatch entrypoints route block scales correctly

Run:
    python tests/test_sm89_block_scale.py
"""
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm


def calc_diff(a, b):
    """Root-mean-squared relative diff normalised by max-abs of reference."""
    scale = b.abs().max().clamp(min=1e-6)
    return (((a - b) / scale).norm() / (a.numel() ** 0.5)).item()


def rand_fp8(shape, device="cuda"):
    return (torch.rand(shape, device=device) * 2 - 1).to(torch.float8_e4m3fn)


def rand_scale(shape, device="cuda"):
    return (torch.rand(shape, device=device) * 1.5 + 0.5).float()


def dequant_a(a_fp8, sa):
    """[M, K] fp8 with [M, ceil(K/128)] 1x128 scales -> [M, K] fp32."""
    M, K = a_fp8.shape
    sa_full = sa.repeat_interleave(128, dim=1)[:, :K]
    return a_fp8.float() * sa_full


def dequant_b(b_fp8, sb):
    """[E, N, K] fp8 with [E, ceil(N/128), ceil(K/128)] scales -> fp32."""
    E, N, K = b_fp8.shape
    sb_full = sb.repeat_interleave(128, dim=1)[:, :N, :]
    sb_full = sb_full.repeat_interleave(128, dim=2)[:, :, :K]
    return b_fp8.float() * sb_full


def build_contig_case(token_counts, N, K, device="cuda"):
    """Returns (a, b, sa, sb, experts, offsets, list_size, total_tokens)."""
    num_experts = len(token_counts)
    total = sum(token_counts)
    a = rand_fp8((total, K), device)
    b = rand_fp8((num_experts, N, K), device)
    sa = rand_scale((total, (K + 127) // 128), device)
    sb = rand_scale((num_experts, (N + 127) // 128, (K + 127) // 128), device)

    experts, ends, acc = [], [], 0
    for e, cnt in enumerate(token_counts):
        if cnt == 0:
            continue
        acc += cnt
        experts.append(e)
        ends.append(acc)
    experts_t = torch.tensor(experts, dtype=torch.int32, device=device)
    offsets_t = torch.tensor(ends, dtype=torch.int32, device=device)
    return a, b, sa, sb, experts_t, offsets_t, len(experts), total


def ref_contig(a, b, sa, sb, experts, offsets):
    a_deq = dequant_a(a, sa)
    b_deq = dequant_b(b, sb)
    out = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32, device=a.device)
    start = 0
    for i, eid in enumerate(experts.tolist()):
        end = int(offsets[i])
        out[start:end] = a_deq[start:end] @ b_deq[eid].t()
        start = end
    return out


def test_contiguous_block_scale():
    torch.manual_seed(0)
    cases = [
        # token_counts, N, K
        ([128], 256, 512),                       # single expert, aligned
        ([3, 0, 200, 64, 1, 127, 128, 33], 256, 512),   # zero/partial tiles
        ([256, 512, 64, 96], 1024, 2048),        # deeper K accumulation
        ([300, 5], 512, 7168),                   # DeepSeek-like K, 56 K-tiles
    ]
    for token_counts, N, K in cases:
        a, b, sa, sb, experts, offsets, list_size, total = build_contig_case(
            token_counts, N, K
        )
        d = torch.empty(total, N, dtype=torch.bfloat16, device="cuda")
        asym_gemm.m_grouped_fp8_asym_gemm_sm89(
            a, b, d, offsets, experts, list_size,
            scale_a_block=sa, scale_b_block=sb,
        )
        ref = ref_contig(a, b, sa, sb, experts, offsets)
        diff = calc_diff(d.float(), ref)
        print(f"  contig tokens={token_counts} N={N} K={K}: diff={diff:.3e}")
        assert diff < 2e-2, f"contiguous block-scale mismatch: {diff}"


def test_masked_block_scale():
    torch.manual_seed(1)
    G, M_max, N, K = 4, 384, 256, 512
    masked = torch.tensor([5, 0, 384, 130], dtype=torch.int32, device="cuda")
    a = rand_fp8((G, M_max, K))
    b = rand_fp8((G, N, K))
    sa = rand_scale((G, M_max, (K + 127) // 128))
    sb = rand_scale((G, (N + 127) // 128, (K + 127) // 128))
    d = torch.empty(G, M_max, N, dtype=torch.bfloat16, device="cuda")

    asym_gemm.m_grouped_fp8_asym_gemm_sm89_masked(
        a, b, d, masked, int(masked.max()),
        scale_a_block=sa, scale_b_block=sb,
    )

    for g in range(G):
        m = int(masked[g])
        if m == 0:
            continue
        a_deq = dequant_a(a[g, :m], sa[g, :m])
        b_deq = dequant_b(b[g : g + 1], sb[g : g + 1])[0]
        ref = a_deq @ b_deq.t()
        diff = calc_diff(d[g, :m].float(), ref)
        print(f"  masked group={g} m={m}: diff={diff:.3e}")
        assert diff < 2e-2, f"masked block-scale mismatch g={g}: {diff}"


def test_block_vs_per_token_equivalence():
    """Broadcast per-token/per-expert scales into block form; outputs must agree."""
    torch.manual_seed(2)
    token_counts, N, K = [64, 130, 7], 256, 1024
    a, b, _, _, experts, offsets, list_size, total = build_contig_case(
        token_counts, N, K
    )
    kg, ng = (K + 127) // 128, (N + 127) // 128
    sa_tok = rand_scale((total,))
    sb_exp = rand_scale((len(token_counts),))

    d_legacy = torch.empty(total, N, dtype=torch.bfloat16, device="cuda")
    asym_gemm.m_grouped_fp8_asym_gemm_sm89(
        a, b, d_legacy, offsets, experts, list_size,
        scale_a_tensor=sa_tok, scale_b_tensor=sb_exp,
    )

    d_block = torch.empty_like(d_legacy)
    asym_gemm.m_grouped_fp8_asym_gemm_sm89(
        a, b, d_block, offsets, experts, list_size,
        scale_a_block=sa_tok[:, None].expand(total, kg).contiguous(),
        scale_b_block=sb_exp[:, None, None].expand(len(token_counts), ng, kg).contiguous(),
    )

    diff = calc_diff(d_block.float(), d_legacy.float())
    print(f"  block vs per-token: diff={diff:.3e}")
    assert diff < 5e-3, f"block path diverges from per-token path: {diff}"


def test_dispatch_routing():
    """Arch-agnostic entrypoints must route (data, block-scale) pairs to SM89."""
    torch.manual_seed(3)
    token_counts, N, K = [40, 88], 256, 512
    a, b, sa, sb, experts, offsets, list_size, total = build_contig_case(
        token_counts, N, K
    )
    d = torch.empty(total, N, dtype=torch.bfloat16, device="cuda")
    asym_gemm.m_grouped_fp8_asym_gemm_nt_contiguous(
        (a, sa), (b, sb), d, offsets, experts, list_size
    )
    ref = ref_contig(a, b, sa, sb, experts, offsets)
    diff = calc_diff(d.float(), ref)
    print(f"  dispatch contig: diff={diff:.3e}")
    assert diff < 2e-2, f"dispatch contiguous mismatch: {diff}"

    G, M_max = 2, 256
    masked = torch.tensor([100, 256], dtype=torch.int32, device="cuda")
    am = rand_fp8((G, M_max, K))
    bm = rand_fp8((G, N, K))
    sam = rand_scale((G, M_max, (K + 127) // 128))
    sbm = rand_scale((G, (N + 127) // 128, (K + 127) // 128))
    dm = torch.empty(G, M_max, N, dtype=torch.bfloat16, device="cuda")
    asym_gemm.m_grouped_fp8_asym_gemm_nt_masked(
        (am, sam), (bm, sbm), dm, masked, int(masked.max())
    )
    for g in range(G):
        m = int(masked[g])
        ref_g = dequant_a(am[g, :m], sam[g, :m]) @ dequant_b(bm[g:g+1], sbm[g:g+1])[0].t()
        diff = calc_diff(dm[g, :m].float(), ref_g)
        print(f"  dispatch masked g={g}: diff={diff:.3e}")
        assert diff < 2e-2, f"dispatch masked mismatch g={g}: {diff}"


if __name__ == "__main__":
    print("test_contiguous_block_scale:")
    test_contiguous_block_scale()
    print("test_masked_block_scale:")
    test_masked_block_scale()
    print("test_block_vs_per_token_equivalence:")
    test_block_vs_per_token_equivalence()
    print("test_dispatch_routing:")
    test_dispatch_routing()
    print("ALL PASSED")
