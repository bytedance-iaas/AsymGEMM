"""Fused Triton kernels for the VRAM-cached MoE paths.

Two hotspots in the cached forward were death-by-tiny-kernels (~40 tensor ops
per layer, thousands per decode step even under CUDA graph replay — replay
removes launch overhead but still executes every kernel):

- per-token INT8 A-quantization (torch: ~7 kernels per call, two calls/layer)
  → one kernel: bf16 rows in, int8 rows + fp32 scales out.
- the decode routing/layout build (torch: ~18 kernels: one-hot, cumsums,
  scatters, arange/where chains) → one single-block kernel producing dest /
  offsets / experts directly. Ranks come from atomics, so a token's row
  within its expert segment varies run-to-run — harmless, because every
  per-row computation is independent and results are gathered back by dest.

Both match the torch reference semantics: scales = amax/127 clamped to 1e-12;
values round half-away-from-zero (torch rounds half-to-even — ties are
essentially absent in real activations and the INT8 contract tolerates them).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_rows_kernel(
    x_ptr, q_ptr, s_ptr,
    K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    amax = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        m = (k0 + offs) < K
        v = tl.load(x_ptr + row * K + k0 + offs, mask=m, other=0.0)
        amax = tl.maximum(amax, tl.abs(v.to(tl.float32)))
    a = tl.max(amax, axis=0)
    s = tl.maximum(a / 127.0, 1e-12)
    tl.store(s_ptr + row, s)
    inv = 1.0 / s
    for k0 in range(0, K, BLOCK_K):
        m = (k0 + offs) < K
        v = tl.load(x_ptr + row * K + k0 + offs, mask=m, other=0.0)
        vs = v.to(tl.float32) * inv
        # round half away from zero via truncating int cast of vs +/- 0.5
        ri = (vs + tl.where(vs >= 0, 0.5, -0.5)).to(tl.int32)
        ri = tl.minimum(tl.maximum(ri, -127), 127)
        tl.store(q_ptr + row * K + k0 + offs, ri.to(tl.int8), mask=m)


def quant_rows(x_bf16: torch.Tensor):
    """Per-row symmetric INT8 quant. x [N, K] bf16 contiguous ->
    (int8 [N, K], fp32 scales [N]). One kernel launch."""
    assert x_bf16.dtype == torch.bfloat16 and x_bf16.is_cuda
    x = x_bf16.contiguous()
    N, K = x.shape
    q = torch.empty(N, K, dtype=torch.int8, device=x.device)
    s = torch.empty(N, dtype=torch.float32, device=x.device)
    if N == 0:
        return q, s
    BLOCK_K = min(2048, triton.next_power_of_2(K))
    _quant_rows_kernel[(N,)](x, q, s, K, BLOCK_K=BLOCK_K, num_warps=8)
    return q, s


@triton.jit
def _decode_layout_kernel(
    eids_ptr,          # [TK] int64, clamped >= 0
    cached_slot_ptr,   # [G] int64 (-1 = not cached)
    dest_ptr,          # [TK] int64 out
    offsets_ptr,       # [2*S] int32 out
    experts_ptr,       # [S+1] int32 out
    counts_ptr,        # [NC] int32 scratch
    seg_of_ptr,        # [NC] int32 scratch
    TK, NC, S, M, BLOCK_M_PAD,
    BLOCK_TK: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    # Single program: TK <= BLOCK_TK items, NC <= BLOCK_NC cache slots.
    offs_tk = tl.arange(0, BLOCK_TK)
    mtk = offs_tk < TK
    e = tl.load(eids_ptr + offs_tk, mask=mtk, other=0)
    slot = tl.load(cached_slot_ptr + e, mask=mtk, other=-1)
    valid = (slot >= 0) & mtk
    sc = tl.where(valid, slot, 0).to(tl.int32)

    offs_nc = tl.arange(0, BLOCK_NC)
    mnc = offs_nc < NC
    tl.store(counts_ptr + offs_nc, tl.zeros([BLOCK_NC], dtype=tl.int32), mask=mnc)
    tl.debug_barrier()

    rank = tl.atomic_add(counts_ptr + sc, 1, mask=valid)
    tl.debug_barrier()

    c = tl.load(counts_ptr + offs_nc, mask=mnc, other=0)
    active = c > 0
    seg = tl.cumsum(active.to(tl.int32), axis=0) - 1        # [BLOCK_NC]
    tl.store(seg_of_ptr + offs_nc, seg, mask=mnc)

    # Defaults: segment j -> expert 0, empty range [j*BM, j*BM); sentinel -1.
    ms = offs_nc < S
    start = (offs_nc * BLOCK_M_PAD).to(tl.int32)
    tl.store(experts_ptr + offs_nc, tl.zeros([BLOCK_NC], dtype=tl.int32), mask=ms)
    tl.store(offsets_ptr + 2 * offs_nc, start, mask=ms)
    tl.store(offsets_ptr + 2 * offs_nc + 1, start, mask=ms)
    one = tl.arange(0, 1)
    tl.store(experts_ptr + S + one, tl.full([1], -1, dtype=tl.int32))
    tl.debug_barrier()

    # Active slots claim their segment: expert id + real end offset.
    ma = active & mnc
    tl.store(experts_ptr + seg, offs_nc.to(tl.int32), mask=ma)
    tl.store(offsets_ptr + 2 * seg + 1, (seg * BLOCK_M_PAD + c).to(tl.int32), mask=ma)
    tl.debug_barrier()

    segi = tl.load(seg_of_ptr + sc, mask=mtk, other=0)
    d = segi.to(tl.int64) * BLOCK_M_PAD + rank.to(tl.int64)
    park = M + offs_tk.to(tl.int64)
    tl.store(dest_ptr + offs_tk, tl.where(valid, d, park), mask=mtk)


def decode_layout(eids_flat: torch.Tensor, cached_slot: torch.Tensor,
                  n_cache: int, S: int, block_m: int):
    """Build the decode contiguous layout in ONE kernel launch.

    eids_flat [TK] int64 (clamped >= 0), cached_slot [G] int64. Returns
    (dest [TK] int64 — parking rows at M+i for non-cached items,
    offsets [2S] int32, experts [S+1] int32) where M = S*block_m.
    """
    dev = eids_flat.device
    TK = eids_flat.numel()
    M = S * block_m
    dest = torch.empty(TK, dtype=torch.int64, device=dev)
    offsets = torch.empty(2 * S, dtype=torch.int32, device=dev)
    experts = torch.empty(S + 1, dtype=torch.int32, device=dev)
    counts = torch.empty(n_cache, dtype=torch.int32, device=dev)
    seg_of = torch.empty(n_cache, dtype=torch.int32, device=dev)
    BLOCK_TK = triton.next_power_of_2(max(TK, 2))
    BLOCK_NC = triton.next_power_of_2(max(n_cache, 2))
    _decode_layout_kernel[(1,)](
        eids_flat, cached_slot, dest, offsets, experts, counts, seg_of,
        TK, n_cache, S, M, block_m,
        BLOCK_TK=BLOCK_TK, BLOCK_NC=BLOCK_NC,
        num_warps=max(1, min(16, BLOCK_TK // 128)),
    )
    return dest, offsets, experts
