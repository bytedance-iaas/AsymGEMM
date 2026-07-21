# tests/test_sm80_int8_asym.py
"""
Correctness tests for the SM80 (A100) INT8 asym MoE kernels via the direct
entry points m_grouped_int8_asym_gemm_sm80_{contiguous,masked}.

The kernels' arch guard is __CUDA_ARCH__ >= 800 and they use only SM80-era
instructions, so these tests exercise the exact A100 code path on any
SM80/SM90/SM100 device (see tests/test_arch_compile_gates.py for the
sm_80 compile-only guarantee).

Semantics under test (1d1d, per-128-K-block scales, FP32 D):
    d[m, n] = sum_kb  (a[m, kb*128:(kb+1)*128].int() @ b[e, n, ...].int())
              * sfa[m, kb] * sfb[e, n, kb]

Run:  python -m pytest tests/test_sm80_int8_asym.py -v
"""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import asym_gemm  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rand_int8(*shape, device="cpu"):
    return torch.randint(-127, 128, shape, dtype=torch.int8, device=device)


def _rand_scales(*shape, device="cpu"):
    # Positive, spread over ~2 decades like real per-token/per-channel scales.
    return (torch.rand(*shape, device=device) * 0.9 + 0.1) * 0.01


def ref_int8_moe(a, sfa, b, sfb, segments):
    """Float64 reference of the per-K-block dequant semantics.

    segments: list of (start, end, expert_id) — the pair-style convention the
    SM80/SM90 asym kernels share (expert_id < 0 = gap, skipped).
    """
    M, K = a.shape
    _, N, _ = b.shape
    kb = K // 128
    d = torch.zeros(M, N, dtype=torch.float64, device=a.device)
    for start, end, e in segments:
        if e >= 0 and end > start:
            a_e = a[start:end].to(torch.float64)
            b_e = b[e].to(torch.float64)
            for k in range(kb):
                sl = slice(k * 128, (k + 1) * 128)
                blk = a_e[:, sl] @ b_e[:, sl].T   # exact in f64 (int8 inputs)
                d[start:end] += (blk
                                 * sfa[start:end, k, None].to(torch.float64)
                                 * sfb[e, None, :, k].to(torch.float64))
    return d


def _check(d, ref, valid_mask=None, tag=""):
    d64 = d.to(torch.float64)
    if valid_mask is not None:
        d64, ref = d64[valid_mask], ref[valid_mask]
    denom = ref.abs().max().clamp(min=1e-30)
    rel = ((d64 - ref).abs().max() / denom).item()
    assert rel < 1e-4, f"{tag}: max rel err {rel:.3e}"


@requires_cuda
@pytest.mark.parametrize("b_pinned", [True, False], ids=["b_pinned_host", "b_hbm"])
@pytest.mark.parametrize(
    "N,K,lens,experts",
    [
        # partial M-tiles, gap segment, zero-token expert
        (256, 256, [37, 0, 128, 200], [2, -1, 0, 3]),
        # single expert, len < one M-tile
        (128, 128, [5], [0]),
        # bn=64 config (N % 128 != 0), deeper K
        (192, 512, [130, 256], [1, 0]),
        # expert reuse across segments (same expert id twice)
        (384, 384, [64, 96, 300], [1, 1, 2]),
    ],
)
def test_contiguous_parity(N, K, lens, experts, b_pinned):
    torch.manual_seed(0)
    dev = "cuda"
    G = max(e for e in experts) + 1
    total = sum(lens)
    kb = K // 128

    # Pair-style segment layout (SM90-compatible): offsets [2S] interleaved
    # (start, end), experts [S+1] with -1 terminator, list_size = S+1.
    starts_l = [sum(lens[:i]) for i in range(len(lens))]
    ends_l = [sum(lens[: i + 1]) for i in range(len(lens))]
    segments = list(zip(starts_l, ends_l, experts))
    offsets = torch.tensor(
        [v for se in zip(starts_l, ends_l) for v in se],
        dtype=torch.int32, device=dev)
    experts_t = torch.tensor(experts + [-1], dtype=torch.int32, device=dev)

    a = _rand_int8(total, K, device=dev)
    sfa = _rand_scales(total, kb, device=dev)
    if b_pinned:
        b = _rand_int8(G, N, K).pin_memory()
        sfb = _rand_scales(G, N, kb).pin_memory()
    else:
        b = _rand_int8(G, N, K, device=dev)
        sfb = _rand_scales(G, N, kb, device=dev)
    d = torch.full((total, N), float("nan"), dtype=torch.float32, device=dev)

    asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
        a, b, d, offsets, experts_t, len(experts) + 1, sfa, sfb)
    torch.cuda.synchronize()

    ref = ref_int8_moe(a, sfa, b.cuda(), sfb.cuda(), segments)

    # Gap/zero segments must stay untouched (NaN canary): build validity mask.
    valid = torch.zeros(total, dtype=torch.bool, device=dev)
    for start, end, e in segments:
        if e >= 0:
            valid[start:end] = True
    assert not d[valid].isnan().any(), "kernel left valid rows unwritten"
    assert d[~valid].isnan().all(), "kernel wrote gap-segment rows"
    _check(d, ref, valid, tag=f"contiguous N={N} K={K}")


@requires_cuda
def test_contiguous_padded_segment_layout():
    """Mirror unified_moe's cached-path layout: segment starts BLOCK_M-aligned
    with dead (unwritten) rows between a segment's end and the next start."""
    torch.manual_seed(2)
    dev = "cuda"
    G, N, K, BM = 3, 256, 256, 256
    kb = K // 128
    counts = [37, 256, 130]
    starts_l = [i * BM for i in range(len(counts))]
    ends_l = [s + c for s, c in zip(starts_l, counts)]
    experts = [1, 0, 2]
    total = len(counts) * BM
    segments = list(zip(starts_l, ends_l, experts))

    offsets = torch.tensor(
        [v for se in zip(starts_l, ends_l) for v in se],
        dtype=torch.int32, device=dev)
    experts_t = torch.tensor(experts + [-1], dtype=torch.int32, device=dev)

    a = _rand_int8(total, K, device=dev)
    sfa = _rand_scales(total, kb, device=dev)
    b = _rand_int8(G, N, K).pin_memory()
    sfb = _rand_scales(G, N, kb).pin_memory()
    d = torch.full((total, N), float("nan"), dtype=torch.float32, device=dev)

    asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
        a, b, d, offsets, experts_t, len(experts) + 1, sfa, sfb)
    torch.cuda.synchronize()

    ref = ref_int8_moe(a, sfa, b.cuda(), sfb.cuda(), segments)
    valid = torch.zeros(total, dtype=torch.bool, device=dev)
    for start, end, _ in segments:
        valid[start:end] = True
    assert not d[valid].isnan().any()
    assert d[~valid].isnan().all(), "kernel wrote padding rows between segments"
    _check(d, ref, valid, tag="padded layout")


@requires_cuda
@pytest.mark.parametrize("b_pinned", [True, False], ids=["b_pinned_host", "b_hbm"])
def test_masked_parity(b_pinned):
    torch.manual_seed(1)
    dev = "cuda"
    G, M_max, N, K = 5, 192, 256, 384
    kb = K // 128

    masked_l = [0, 192, 37, 128, 191]   # zero group, full group, partial tiles
    masked_m = torch.tensor(masked_l, dtype=torch.int32, device=dev)

    a = _rand_int8(G, M_max, K, device=dev)
    sfa = _rand_scales(G, M_max, kb, device=dev)
    if b_pinned:
        b = _rand_int8(G, N, K).pin_memory()
        sfb = _rand_scales(G, N, kb).pin_memory()
    else:
        b = _rand_int8(G, N, K, device=dev)
        sfb = _rand_scales(G, N, kb, device=dev)
    d = torch.full((G, M_max, N), float("nan"), dtype=torch.float32, device=dev)

    asym_gemm.m_grouped_int8_asym_gemm_sm80_masked(a, b, d, masked_m, sfa, sfb)
    torch.cuda.synchronize()

    b_dev, sfb_dev = b.cuda(), sfb.cuda()
    for g in range(G):
        m = masked_l[g]
        if m == 0:
            assert d[g].isnan().all(), f"group {g}: wrote rows of empty group"
            continue
        ref = ref_int8_moe(
            a[g, :m], sfa[g, :m], b_dev[g:g + 1], sfb_dev[g:g + 1],
            [(0, m, 0)])
        assert not d[g, :m].isnan().any(), f"group {g}: valid rows unwritten"
        assert d[g, m:].isnan().all(), f"group {g}: wrote padding rows"
        _check(d[g, :m], ref, tag=f"masked g={g} m={m}")


@requires_cuda
def test_unified_layer_gpu_bucket_on_sm80_kernel(monkeypatch):
    """End-to-end: unified_moe.Layer's GPU bucket produces the same output
    through the SM80 INT8 kernel as through the device's native backend.

    The facade routes by real device arch, so on H100/H200 boxes we force the
    SM80 path by monkeypatching the facade name the runtime calls — proving
    the A100 kernel is drop-in compatible with the unified runtime's layout
    (pair offsets, -1 terminated experts, [M,kb]/[G,N,kb] scales, FP32 D).
    """
    if asym_gemm.unified_moe is None:
        pytest.skip("unified_moe sub-package not available")
    from asym_gemm.unified_moe import Layer

    torch.manual_seed(3)
    G, H, I, top_k, T = 8, 256, 512, 2, 96
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.05
    up = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.05
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=4, m_cpu=0)

    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    expert_ids = torch.rand(T, G, device="cuda").topk(top_k, dim=1).indices
    route_w = torch.rand(T, top_k, device="cuda")
    route_w = route_w / route_w.sum(dim=1, keepdim=True)

    y_native = layer.forward(x, expert_ids, route_w).float()

    calls = {"n": 0}

    def sm80_shim(a, b, d, offsets, experts, list_size, recipe=None,
                  compiled_dims="nk", **kw):
        calls["n"] += 1
        asym_gemm.m_grouped_int8_asym_gemm_sm80_contiguous(
            a[0], b[0], d, offsets, experts, list_size, a[1], b[1])

    monkeypatch.setattr(
        asym_gemm, "m_grouped_int8_asym_gemm_nt_contiguous", sm80_shim)
    y_sm80 = layer.forward(x, expert_ids, route_w).float()

    assert calls["n"] > 0, "monkeypatched facade was never called"
    denom = y_native.abs().max().clamp(min=1e-30)
    rel = ((y_sm80 - y_native).abs().max() / denom).item()
    cos = torch.nn.functional.cosine_similarity(
        y_sm80.flatten(), y_native.flatten(), dim=0).item()
    assert rel < 5e-3 and cos > 0.9999, f"rel={rel:.3e} cos={cos:.6f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
