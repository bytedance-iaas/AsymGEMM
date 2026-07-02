"""Unified INT8 MoE layer (asym_gemm.unified_moe.Layer) — v3.

Per-expert dispatch on routed token count:
- m_e <= m_cpu  → CPU AMX INT8 path (cpu_gemm, stride-aware row-major B)
- m_e >  m_cpu  → SM90 INT8 grouped-MoE WGMMA kernel (asym_gemm)

**All expert parameters live in pinned host memory** — no VRAM weight mirror.
The GPU path reads weights from pinned host via Hopper TMA over PCIe (UVA).
The CPU path reads from **the same pinned row-major INT8 bytes** via the
stride-aware AMX kernel — no second permuted view, no per-call repack.
See `Stride.md` for the kernel rework that enables this unification.

Both backends compute the same INT8 → INT32 → FP32 dequant contract:

    sA[i]       = amax_row_i / 127
    sB[n]       = amax_col_n / 127
    C_int32     = A_int8 @ B_int8.T
    C_fp32      = sA · sB · C_int32       (outer broadcast)

The SM90 kernel expects per-K-block scales (granularity `GRAN_K=128`); we
satisfy this by broadcasting our per-row / per-channel scales across the
K-block dimension — mathematically identical because the scales are
constant along K.

See docs unified_kernel_pinned_CPU_memory.md for the full design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import asym_gemm
from .. import _cpu_C as _C

# Granularity required by sm90_int8_asym_gemm_1d1d (one scale per 128 K elems).
GRAN_K = 128
# Per-expert M alignment for the SM90 INT8 1D1D contiguous-layout kernel.
# The kernel's asymScheduler computes m_start/m_end = ceil_div(offsets/BLOCK_M),
# so each expert's offset range must be a multiple of the kernel-chosen BLOCK_M
# or adjacent expert ranges collapse together. SM90 INT8 heuristic candidates
# are {64, 128, 256} — we pad to the maximum so the layout is safe regardless
# of which the heuristic picks for the live (M, N, K) shape.
BLOCK_M = 256

# Fraction of routed rows the CPU bucket takes at prefill (large batches),
# where the two buckets run concurrently: the GPU streams each of its experts'
# weights over PCIe (cost ~per expert), the CPU cost scales ~per row, so the
# CPU takes the smallest-count experts. 0 disables the split (GPU-only
# prefill, CPU idle). Override via ASYMGEMM_CPU_PREFILL_FRACTION.
#
# Default from a sweep on 2x8457C (48-thread pool, node-bound) + H200,
# Qwen3-30B-A3B, 3500-token prefill — TTFT: 0.0 -> 867 ms, 0.05 -> 710 ms,
# 0.07 -> 728 ms, 0.10 -> 748 ms, 0.15 -> 805 ms, 0.25+ -> CPU-bound and
# worse. Small fractions win twice over: the smallest experts also carry the
# worst BLOCK_M padding waste in the GPU contiguous layout.
_CPU_PREFILL_FRACTION: Optional[float] = None


def _cpu_prefill_fraction() -> float:
    global _CPU_PREFILL_FRACTION
    if _CPU_PREFILL_FRACTION is None:
        _CPU_PREFILL_FRACTION = float(
            os.getenv("ASYMGEMM_CPU_PREFILL_FRACTION", "0.05")
        )
    return _CPU_PREFILL_FRACTION


# ---------------------------------------------------------------------------
# Quantization helpers (the unified INT8 contract)
# ---------------------------------------------------------------------------

def quantize_per_channel_int8(
    w_bf16: torch.Tensor,  # [N, K] BF16
) -> tuple[np.ndarray, np.ndarray]:
    """Offline weight quantization. Returns (int8 weights [N, K], FP32 scales [N])."""
    assert w_bf16.dtype == torch.bfloat16
    assert w_bf16.ndim == 2
    w_fp32 = w_bf16.to(torch.float32)
    amax = w_fp32.abs().amax(dim=1)            # [N]
    scales = (amax / 127.0).clamp(min=1e-12)   # [N]
    inv = 1.0 / scales
    q = (w_fp32 * inv[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.numpy(force=True), scales.to(torch.float32).numpy(force=True)


def quantize_per_token_int8_gpu(
    x_bf16: torch.Tensor,  # [M, K] BF16 on CUDA
) -> tuple[torch.Tensor, torch.Tensor]:
    """A-side per-row quant on GPU. Returns (int8 [M, K], scales [M] FP32)."""
    assert x_bf16.is_cuda and x_bf16.dtype == torch.bfloat16
    x = x_bf16.to(torch.float32)
    amax = x.abs().amax(dim=1)
    scales = (amax / 127.0).clamp(min=1e-12)
    inv = (1.0 / scales)[:, None]
    q = (x * inv).round().clamp(-127, 127).to(torch.int8)
    return q, scales.to(torch.float32)


# ---------------------------------------------------------------------------
# BF16 ↔ FP32 helpers (BF16 carried as uint16 bit pattern for CPU path)
# ---------------------------------------------------------------------------

def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 → bf16. Returns uint16."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def bf16_bits_to_fp32(u16: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(u16, dtype=np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def torch_bf16_to_np_bits(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16 and not t.is_cuda
    return t.contiguous().view(torch.uint16).numpy(force=True)


def silu_fp32(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Expert weight container
# ---------------------------------------------------------------------------

@dataclass
class ExpertSlab:
    """One MoE layer's three projections, INT8 quantized, all in pinned host.

    A **single** pinned-host byte layout per expert — row-major
    `[G, N, K]` int8 + `[G, N]` fp32 scales — feeds both backends:

      *_int8 / *_scales   — row-major torch tensors, pinned via pin_memory().
                            Consumed directly by the SM90 INT8 GPU kernel via
                            Hopper TMA UVA fetches over PCIe, *and* directly
                            by the stride-aware cpu_gemm AMX INT8 kernel.

    Plus the per-K-block scale broadcast on device:
      *_sfb [G, N, K//128] fp32 — small device tensor, built once at load,
                                  consumed by the GPU kernel.
    """
    num_experts: int
    hidden: int
    inter: int
    kb_hidden: int                      # hidden // GRAN_K
    kb_inter:  int                      # inter  // GRAN_K

    # row-major INT8 in pinned host (one source of truth for both backends)
    gate_int8: torch.Tensor             # int8, pinned, [G, inter, hidden]
    gate_scales: torch.Tensor           # float32, pinned, [G, inter]
    up_int8: torch.Tensor               # int8, pinned, [G, inter, hidden]
    up_scales: torch.Tensor             # float32, pinned, [G, inter]
    down_int8: torch.Tensor             # int8, pinned, [G, hidden, inter]
    down_scales: torch.Tensor           # float32, pinned, [G, hidden]

    # Device-resident SFB broadcasts for the SM90 kernel
    gate_sfb: Optional[torch.Tensor] = None   # [G, inter, kb_hidden] fp32 on CUDA
    up_sfb:   Optional[torch.Tensor] = None   # [G, inter, kb_hidden] fp32 on CUDA
    down_sfb: Optional[torch.Tensor] = None   # [G, hidden, kb_inter] fp32 on CUDA


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """Unified INT8 MoE layer with pinned-CPU weight residency.

    Construct via ``Layer.from_bf16(...)``. Forward signature:

        y = layer.forward(x_bf16, expert_ids, route_w)

    where:
        x_bf16     : (T, hidden) torch.bfloat16, cuda
        expert_ids : (T, top_k) torch.int64 / int32
        route_w    : (T, top_k) torch.float32
        y          : (T, hidden) torch.bfloat16 on the same device as x

    ``hidden`` and ``inter`` must each be multiples of ``GRAN_K=128`` (the
    K-block granularity required by sm90_int8_asym_gemm_1d1d).
    """

    weight_residency = "pinned_host"
    gpu_backend = "asym_gemm_sm90_int8_1d1d"

    def __init__(
        self,
        slab: ExpertSlab,
        *,
        top_k: int,
        cpu_threads: int = 0,
        cuda_device: int = 0,
        m_cpu: int = 16,
        runtime: Optional["_C.Runtime"] = None,
    ):
        self.slab = slab
        self.top_k = top_k
        self.cuda_device = cuda_device
        # An injected `runtime` (e.g. a single pool shared by every MoE layer
        # in a serving framework) takes precedence; `cpu_threads` is only used
        # for the build-your-own default and is ignored when `runtime` is given.
        self.rt = runtime if runtime is not None else _C.Runtime(cpu_threads)
        self.m_cpu = m_cpu
        self.cache_n = 0
        self._build_gpu_cache()

    # -----------------------------------------------------------
    # VRAM expert cache (decode GPU bucket)
    # -----------------------------------------------------------

    def _build_gpu_cache(self) -> None:
        """Copy the first ASYMGEMM_GPU_CACHED_EXPERTS experts' INT8 weights to
        the GPU.

        The cache is a memory/speed dial: cached experts are computed at
        decode by the SM90 INT8 kernel reading HBM (~320 GB/s measured)
        concurrently with the CPU AMX bucket, instead of on the CPU. 0 (the
        default) keeps the all-CPU decode path; num_experts moves the whole
        decode MoE onto the GPU (still 2x smaller than BF16 weights). Cost:
        n * (2*inter+hidden)*... bytes per layer — e.g. Qwen3-30B-A3B,
        n=128: ~600 MB/layer, 29 GB total.
        """
        n = int(os.getenv("ASYMGEMM_GPU_CACHED_EXPERTS", "0"))
        if n <= 0 or not torch.cuda.is_available():
            return
        s = self.slab
        n = min(n, s.num_experts)
        dev = f"cuda:{self.cuda_device}"
        self.cache_gate_int8 = s.gate_int8[:n].to(dev)
        self.cache_up_int8 = s.up_int8[:n].to(dev)
        self.cache_down_int8 = s.down_int8[:n].to(dev)
        # SFBs are already device-resident; slice a contiguous copy so the
        # kernel indexes them 0..n-1 like the cached INT8 slabs.
        self.cache_gate_sfb = s.gate_sfb[:n].contiguous()
        self.cache_up_sfb = s.up_sfb[:n].contiguous()
        self.cache_down_sfb = s.down_sfb[:n].contiguous()
        cached_slot = torch.full((s.num_experts,), -1, dtype=torch.int64,
                                 device=dev)
        cached_slot[:n] = torch.arange(n, dtype=torch.int64, device=dev)
        self.cached_slot = cached_slot
        self.cache_n = n

    def _cached_gpu_decode(
        self,
        x_bf16: torch.Tensor,       # [T, H] bf16 device
        expert_ids: torch.Tensor,   # [T, K] int64 device (clamped >= 0)
        route_w: torch.Tensor,      # [T, K] fp32 device
    ) -> torch.Tensor:              # [T, H] fp32, route-weighted, cached items only
        """Decode-time GPU bucket over the VRAM expert cache.

        Pure stream-ordered tensor ops + the grouped INT8 kernel, so it is
        CUDA-graph capturable: the contiguous layout is computed on the device
        every step (segment s = 256 rows at offset s*256; offsets/experts
        tensor *contents* are dynamic, their shapes and `list_size` static).
        Non-cached items contribute 0 here and are computed by the CPU bucket
        (their route weight is zeroed before the host node reads it).
        """
        slab = self.slab
        dev = x_bf16.device
        H, I = slab.hidden, slab.inter
        kb_h, kb_i = slab.kb_hidden, slab.kb_inter
        Nc = self.cache_n
        T, K = expert_ids.shape
        TK = T * K
        S = min(TK, Nc)                     # max simultaneously-active experts
        M = S * BLOCK_M

        slot = self.cached_slot[expert_ids.reshape(-1)]        # [TK]
        valid = slot >= 0
        sc = slot.clamp_min(0)

        # rank of each item within its expert; per-slot counts.
        onehot = torch.zeros(TK, Nc, dtype=torch.int32, device=dev)
        onehot.scatter_(1, sc.unsqueeze(1), valid.to(torch.int32).unsqueeze(1))
        cum = onehot.cumsum(0)                                  # int64 (promoted)
        rank = cum.gather(1, sc.unsqueeze(1)).squeeze(1) - 1
        counts = cum[-1]                                        # [Nc] int64

        # Pack active experts into segments [seg*256, seg*256+count).
        active = counts > 0
        seg_of = active.to(torch.int64).cumsum(0) - 1           # [Nc]
        dump = torch.full_like(seg_of, S)                       # park inactive
        seg_idx = torch.where(active, seg_of, dump)
        seg_counts = torch.zeros(S + 1, dtype=torch.int64, device=dev)
        seg_counts.scatter_(0, seg_idx, counts)
        experts_t = torch.zeros(S + 1, dtype=torch.int32, device=dev)
        experts_t.scatter_(
            0, seg_idx, torch.arange(Nc, dtype=torch.int32, device=dev)
        )
        experts_t.narrow(0, S, 1).fill_(-1)                     # sentinel
        starts = (
            torch.arange(S, dtype=torch.int32, device=dev) * BLOCK_M
        )
        offsets = torch.empty(2 * S, dtype=torch.int32, device=dev)
        offsets[0::2] = starts
        offsets[1::2] = starts + seg_counts[:S].to(torch.int32)

        # Item destinations in the grouped layout; invalid items park past M.
        arange_tk = torch.arange(TK, dtype=torch.int64, device=dev)
        dest_valid = seg_of.index_select(0, sc) * BLOCK_M + rank
        dest = torch.where(valid, dest_valid, M + arange_tk)
        dest_safe = torch.where(valid, dest_valid, torch.zeros_like(dest_valid))

        # Gather + quantize only the TK routed rows, scatter into the layout
        # (quantizing the full padded buffer would read ~30x the bytes).
        tok = torch.div(arange_tk, K, rounding_mode="floor")
        x_items = x_bf16.index_select(0, tok)                   # [TK, H]
        q_items, s_items = quantize_per_token_int8_gpu(x_items)
        a_int8 = torch.zeros(M + TK, H, dtype=torch.int8, device=dev)
        a_int8.index_copy_(0, dest, q_items)
        sfa = torch.zeros(M + TK, kb_h, dtype=torch.float32, device=dev)
        sfa.index_copy_(0, dest, s_items.unsqueeze(1).expand(TK, kb_h).contiguous())

        d_gate = torch.empty(M, I, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8[:M], sfa[:M]), (self.cache_gate_int8, self.cache_gate_sfb),
            d_gate, offsets, experts_t, S + 1, recipe=(1, 1, GRAN_K),
        )
        d_up = torch.empty(M, I, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8[:M], sfa[:M]), (self.cache_up_int8, self.cache_up_sfb),
            d_up, offsets, experts_t, S + 1, recipe=(1, 1, GRAN_K),
        )
        act = torch.nn.functional.silu(d_gate) * d_up
        act_bf16 = act.to(torch.bfloat16)

        # Down: re-quantize the routed rows (BF16 round-trip matches the CPU
        # path). Rows of inactive segments are uninitialized — gather via
        # dest_safe so invalid items read a real row and land in parking.
        a2_items, s2_items = quantize_per_token_int8_gpu(
            act_bf16.index_select(0, dest_safe)
        )
        a2_int8 = torch.zeros(M + TK, I, dtype=torch.int8, device=dev)
        a2_int8.index_copy_(0, dest, a2_items)
        sfa2 = torch.zeros(M + TK, kb_i, dtype=torch.float32, device=dev)
        sfa2.index_copy_(0, dest, s2_items.unsqueeze(1).expand(TK, kb_i).contiguous())

        d_down = torch.empty(M, H, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a2_int8[:M], sfa2[:M]), (self.cache_down_int8, self.cache_down_sfb),
            d_down, offsets, experts_t, S + 1, recipe=(1, 1, GRAN_K),
        )

        w = route_w.reshape(-1) * valid.to(torch.float32)
        y_items = d_down.index_select(0, dest_safe) * w[:, None]
        return y_items.view(T, K, H).sum(dim=1)                 # [T, H] fp32

    # -----------------------------------------------------------
    # construction
    # -----------------------------------------------------------

    @classmethod
    def from_bf16(
        cls,
        gate: torch.Tensor,   # [G, inter, hidden] bf16
        up: torch.Tensor,     # [G, inter, hidden] bf16
        down: torch.Tensor,   # [G, hidden, inter] bf16
        *,
        top_k: int,
        cpu_threads: int = 0,
        cuda_device: int = 0,
        m_cpu: int = 16,
        runtime: Optional["_C.Runtime"] = None,
    ) -> "Layer":
        assert gate.shape == up.shape
        G, N_inter, K_hidden = gate.shape
        assert down.shape == (G, K_hidden, N_inter)
        for t in (gate, up, down):
            assert t.dtype == torch.bfloat16

        # SM90 INT8 kernel requires K % 128 == 0 on every K dim it sees.
        # In our layer that's both `hidden` (gate/up's K, down's N) and
        # `inter` (gate/up's N, down's K).
        assert K_hidden % GRAN_K == 0, \
            f"hidden ({K_hidden}) must be a multiple of {GRAN_K}"
        assert N_inter % GRAN_K == 0, \
            f"inter ({N_inter}) must be a multiple of {GRAN_K}"
        kb_h = K_hidden // GRAN_K
        kb_i = N_inter  // GRAN_K

        # --- pinned host: row-major INT8 + per-channel FP32 scales ---
        gate_int8 = torch.empty(G, N_inter, K_hidden, dtype=torch.int8).pin_memory()
        gate_s    = torch.empty(G, N_inter,            dtype=torch.float32).pin_memory()
        up_int8   = torch.empty_like(gate_int8).pin_memory()
        up_s      = torch.empty_like(gate_s).pin_memory()
        down_int8 = torch.empty(G, K_hidden, N_inter, dtype=torch.int8).pin_memory()
        down_s    = torch.empty(G, K_hidden,           dtype=torch.float32).pin_memory()

        # --- quantize once into the pinned row-major buffers ---
        # No second permuted view: both backends read these bytes directly.
        for g in range(G):
            q, s = quantize_per_channel_int8(gate[g])
            gate_int8[g] = torch.from_numpy(q)
            gate_s[g]    = torch.from_numpy(s)

            q, s = quantize_per_channel_int8(up[g])
            up_int8[g] = torch.from_numpy(q)
            up_s[g]    = torch.from_numpy(s)

            q, s = quantize_per_channel_int8(down[g])
            down_int8[g] = torch.from_numpy(q)
            down_s[g]    = torch.from_numpy(s)

        return cls._assemble(
            gate_int8=gate_int8, gate_s=gate_s,
            up_int8=up_int8, up_s=up_s,
            down_int8=down_int8, down_s=down_s,
            top_k=top_k, cpu_threads=cpu_threads,
            cuda_device=cuda_device, m_cpu=m_cpu, runtime=runtime,
        )

    @classmethod
    def from_int8(
        cls,
        gate_int8: torch.Tensor,    # [G, inter, hidden] int8
        gate_scales: torch.Tensor,  # [G, inter] fp32
        up_int8: torch.Tensor,      # [G, inter, hidden] int8
        up_scales: torch.Tensor,    # [G, inter] fp32
        down_int8: torch.Tensor,    # [G, hidden, inter] int8
        down_scales: torch.Tensor,  # [G, hidden] fp32
        *,
        top_k: int,
        cpu_threads: int = 0,
        cuda_device: int = 0,
        m_cpu: int = 16,
        runtime: Optional["_C.Runtime"] = None,
    ) -> "Layer":
        """Build a layer from **pre-quantized** INT8 expert weights.

        This is the fast path that skips the per-expert quantization loop
        ``from_bf16`` performs. The tensors must be the exact bytes
        ``from_bf16`` would have produced — per-channel symmetric INT8 with
        FP32 ``amax/127`` scales, row-major, one scale per output channel:

            gate_int8/up_int8 : [G, inter, hidden] int8   (K dim = hidden)
            down_int8         : [G, hidden, inter] int8   (K dim = inter)
            *_scales          : per output row, fp32

        Produce them offline with ``scripts/convert_int8_weights.py``, which
        calls the same :func:`quantize_per_channel_int8` used here, so an
        offline-converted checkpoint is byte-identical to the online path.
        """
        assert gate_int8.shape == up_int8.shape, \
            f"gate {tuple(gate_int8.shape)} != up {tuple(up_int8.shape)}"
        G, N_inter, K_hidden = gate_int8.shape
        assert down_int8.shape == (G, K_hidden, N_inter), \
            f"down_int8 {tuple(down_int8.shape)} != {(G, K_hidden, N_inter)}"
        assert gate_scales.shape == (G, N_inter), \
            f"gate_scales {tuple(gate_scales.shape)} != {(G, N_inter)}"
        assert up_scales.shape == (G, N_inter), \
            f"up_scales {tuple(up_scales.shape)} != {(G, N_inter)}"
        assert down_scales.shape == (G, K_hidden), \
            f"down_scales {tuple(down_scales.shape)} != {(G, K_hidden)}"
        for name, t in (("gate_int8", gate_int8), ("up_int8", up_int8),
                        ("down_int8", down_int8)):
            assert t.dtype == torch.int8, f"{name} must be int8, got {t.dtype}"
        for name, t in (("gate_scales", gate_scales), ("up_scales", up_scales),
                        ("down_scales", down_scales)):
            assert t.dtype == torch.float32, f"{name} must be float32, got {t.dtype}"
        assert K_hidden % GRAN_K == 0, \
            f"hidden ({K_hidden}) must be a multiple of {GRAN_K}"
        assert N_inter % GRAN_K == 0, \
            f"inter ({N_inter}) must be a multiple of {GRAN_K}"

        # Match from_bf16's residency: the row-major bytes must live in pinned
        # host memory so the SM90 kernel can TMA them over PCIe (UVA) and the
        # CPU AMX kernel can read them directly.
        def _pin(t: torch.Tensor) -> torch.Tensor:
            t = t.contiguous()
            if torch.cuda.is_available() and not t.is_pinned():
                t = t.pin_memory()
            return t

        return cls._assemble(
            gate_int8=_pin(gate_int8), gate_s=_pin(gate_scales),
            up_int8=_pin(up_int8), up_s=_pin(up_scales),
            down_int8=_pin(down_int8), down_s=_pin(down_scales),
            top_k=top_k, cpu_threads=cpu_threads,
            cuda_device=cuda_device, m_cpu=m_cpu, runtime=runtime,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        gate_int8: torch.Tensor,
        gate_s: torch.Tensor,
        up_int8: torch.Tensor,
        up_s: torch.Tensor,
        down_int8: torch.Tensor,
        down_s: torch.Tensor,
        top_k: int,
        cpu_threads: int = 0,
        cuda_device: int = 0,
        m_cpu: int = 16,
        runtime: Optional["_C.Runtime"] = None,
    ) -> "Layer":
        """Assemble the device SFB broadcasts, the ExpertSlab and the Layer from
        the six pinned-host INT8/scale tensors. Shared by ``from_bf16`` (which
        quantizes first) and ``from_int8`` (which loads pre-quantized bytes)."""
        G, N_inter, K_hidden = gate_int8.shape
        kb_h = K_hidden // GRAN_K
        kb_i = N_inter // GRAN_K

        # --- device-resident SFB broadcasts ---
        # Build on whatever CUDA device the user picked.
        if torch.cuda.is_available():
            dev = f"cuda:{cuda_device}"
            # gate_sfb: per-channel scale broadcast over K-blocks. K_dim = hidden.
            gate_sfb = gate_s.to(dev).unsqueeze(-1).expand(G, N_inter, kb_h).contiguous()
            up_sfb   = up_s  .to(dev).unsqueeze(-1).expand(G, N_inter, kb_h).contiguous()
            # down_sfb: K_dim = inter
            down_sfb = down_s.to(dev).unsqueeze(-1).expand(G, K_hidden, kb_i).contiguous()
        else:
            gate_sfb = up_sfb = down_sfb = None

        slab = ExpertSlab(
            num_experts=G, hidden=K_hidden, inter=N_inter,
            kb_hidden=kb_h, kb_inter=kb_i,
            gate_int8=gate_int8, gate_scales=gate_s,
            up_int8=up_int8,     up_scales=up_s,
            down_int8=down_int8, down_scales=down_s,
            gate_sfb=gate_sfb, up_sfb=up_sfb, down_sfb=down_sfb,
        )
        return cls(slab, top_k=top_k, cpu_threads=cpu_threads,
                   cuda_device=cuda_device, m_cpu=m_cpu, runtime=runtime)

    def set_m_cpu(self, m_cpu: int) -> None:
        self.m_cpu = int(m_cpu)

    # -----------------------------------------------------------
    # CPU bucket — reads the same pinned row-major bytes as the GPU path
    # -----------------------------------------------------------

    def _cpu_expert_forward(
        self,
        e: int,
        x_bf16_bits: np.ndarray,    # [m_e, hidden] uint16
    ) -> np.ndarray:                # [m_e, hidden] fp32
        slab = self.slab
        m_e = x_bf16_bits.shape[0]
        H, I = slab.hidden, slab.inter

        # numpy views into the pinned tensors — same bytes, no copy.
        gate_b   = slab.gate_int8[e].numpy()
        gate_s   = slab.gate_scales[e].numpy()
        up_b     = slab.up_int8[e].numpy()
        up_s     = slab.up_scales[e].numpy()
        down_b   = slab.down_int8[e].numpy()
        down_s   = slab.down_scales[e].numpy()

        c_gate = np.empty((m_e, I), dtype=np.float32)
        c_up   = np.empty((m_e, I), dtype=np.float32)
        _C.gemm_bf16_int8(self.rt, x_bf16_bits, gate_b, gate_s, c_gate, 1.0, 0.0)
        _C.gemm_bf16_int8(self.rt, x_bf16_bits, up_b,   up_s,   c_up,   1.0, 0.0)

        act = silu_fp32(c_gate) * c_up
        act_bf16_bits = fp32_to_bf16_bits(act)

        c_down = np.empty((m_e, H), dtype=np.float32)
        _C.gemm_bf16_int8(self.rt, act_bf16_bits, down_b, down_s, c_down, 1.0, 0.0)
        return c_down

    # -----------------------------------------------------------
    # GPU bucket — grouped INT8 over pinned weights, one launch / projection
    # -----------------------------------------------------------

    @staticmethod
    def _build_layout(
        part_experts: np.ndarray,   # expert ids in this partition (sorted)
        counts_np: np.ndarray,      # [G] routed item count per expert
        seg_start_np: np.ndarray,   # [G] start of each expert's sorted segment
        sorted_e: torch.Tensor,     # [T*K] expert id per sorted item (device)
        rows_sorted: torch.Tensor,  # [T*K] token row per sorted item (device)
        slots_sorted: torch.Tensor, # [T*K] top-k slot per sorted item (device)
        G: int,
        device,
    ):
        """Vectorized AsymGEMM contiguous layout for one expert partition.

        Expert g's items land at pad_start[g] + rank-within-segment; every
        expert's block is padded to BLOCK_M for the kernel's per-expert row
        alignment. Returns the (M_grouped, idx_to_orig, slot_to_orig, offsets,
        experts, list_size) tuple consumed by _gpu_grouped_forward.
        """
        m_padded = ((counts_np[part_experts] + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        pad_start_np = np.zeros(G, dtype=np.int64)
        pad_start_np[part_experts] = np.concatenate(
            ([0], np.cumsum(m_padded[:-1]))
        )
        M_grouped = int(m_padded.sum())

        offsets_np = np.empty(2 * len(part_experts), dtype=np.int32)
        offsets_np[0::2] = pad_start_np[part_experts]
        offsets_np[1::2] = pad_start_np[part_experts] + m_padded
        experts_np = np.concatenate((part_experts, [-1])).astype(np.int32)

        part_t = torch.from_numpy(part_experts).to(device)
        seg_start = torch.from_numpy(seg_start_np).to(device)
        pad_start = torch.from_numpy(pad_start_np).to(device)

        is_part_item = torch.isin(sorted_e, part_t)
        pos = torch.nonzero(is_part_item, as_tuple=False).squeeze(-1)
        e_sel = sorted_e[pos]
        dest = pad_start[e_sel] + (pos - seg_start[e_sel])

        idx_to_orig = torch.full((M_grouped,), -1, dtype=torch.long, device=device)
        slot_to_orig = torch.full_like(idx_to_orig, -1)
        idx_to_orig[dest] = rows_sorted[pos]
        slot_to_orig[dest] = slots_sorted[pos]

        return (
            M_grouped,
            idx_to_orig,
            slot_to_orig,
            torch.from_numpy(offsets_np).to(device),
            torch.from_numpy(experts_np).to(device),
            len(experts_np),
        )

    def _gpu_grouped_forward(
        self,
        x_gpu: torch.Tensor,        # [T, H] bf16 device
        layout,                     # contiguous layout built in forward()
        route_w_gpu: torch.Tensor,  # [T, top_k] fp32 device
        cached: bool = False,       # weights from the VRAM cache vs pinned host
    ):
        """Run gate/up/down for one GPU-bucket partition in three grouped calls.

        ``layout`` is (M_grouped, idx_to_orig, slot_to_orig, offsets, experts,
        list_size): the AsymGEMM contiguous layout, built vectorized on the
        GPU in forward(). With ``cached`` the kernels read the VRAM expert
        cache (HBM, ~14x the bandwidth of the pinned-host PCIe path); the
        cache holds experts 0..cache_n-1, so global expert ids double as
        cache-local ids and the same layout format works for both. Returns
        (orig_rows_for_valid, y_weighted_fp32) where orig_rows_for_valid is a
        [n_valid] long tensor of source rows in the original [T, H]
        activations, and y_weighted_fp32 is [n_valid, H] fp32 ready for
        ``out_fp32.index_add_(0, orig_rows, y_weighted_fp32)``.
        """
        slab = self.slab
        dev = x_gpu.device
        H, I = slab.hidden, slab.inter
        kb_h, kb_i = slab.kb_hidden, slab.kb_inter

        M_grouped, idx_to_orig, slot_to_orig, offsets, experts, list_size = layout
        if M_grouped == 0:
            return None, None

        if cached:
            gate_b, gate_sfb = self.cache_gate_int8, self.cache_gate_sfb
            up_b, up_sfb = self.cache_up_int8, self.cache_up_sfb
            down_b, down_sfb = self.cache_down_int8, self.cache_down_sfb
        else:
            gate_b, gate_sfb = slab.gate_int8, slab.gate_sfb
            up_b, up_sfb = slab.up_int8, slab.up_sfb
            down_b, down_sfb = slab.down_int8, slab.down_sfb

        # Gather activations into the contiguous layout. Padding rows (idx=-1)
        # stay zero so their activation amax → 0 → scale clamped to 1e-12.
        valid_mask = (idx_to_orig >= 0)
        a_bf16 = torch.zeros(M_grouped, H, device=dev, dtype=torch.bfloat16)
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        a_bf16.index_copy_(0, valid_idx,
                            x_gpu.index_select(0, idx_to_orig[valid_idx]))

        # Per-token A quant.
        a_int8, sA = quantize_per_token_int8_gpu(a_bf16)
        # SFA for gate/up: broadcast per-token scale across kb_h K-blocks.
        sfa_h = sA.unsqueeze(1).expand(M_grouped, kb_h).contiguous()

        # gate
        d_gate = torch.empty(M_grouped, I, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8, sfa_h), (gate_b, gate_sfb),
            d_gate, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
        )

        # up
        d_up = torch.empty(M_grouped, I, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8, sfa_h), (up_b, up_sfb),
            d_up, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
        )

        # SwiGLU
        act = torch.nn.functional.silu(d_gate) * d_up

        # Down: re-quantize act (BF16 round-trip to match the CPU path).
        act_bf16 = act.to(torch.bfloat16)
        a2_int8, sA2 = quantize_per_token_int8_gpu(act_bf16)
        sfa_i = sA2.unsqueeze(1).expand(M_grouped, kb_i).contiguous()

        d_down = torch.empty(M_grouped, H, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a2_int8, sfa_i), (down_b, down_sfb),
            d_down, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
        )

        # Apply routing weights, return for scatter-add.
        orig_rows  = idx_to_orig[valid_idx]
        orig_slots = slot_to_orig[valid_idx]
        w = route_w_gpu[orig_rows, orig_slots]              # [n_valid]
        y_w = d_down[valid_idx] * w[:, None]
        return orig_rows, y_w

    # -----------------------------------------------------------
    # main forward
    # -----------------------------------------------------------

    def forward(
        self,
        x_bf16: torch.Tensor,        # [T, hidden] BF16
        expert_ids: torch.Tensor,    # [T, top_k] int
        route_w: torch.Tensor,       # [T, top_k] fp32
    ) -> torch.Tensor:
        T, H = x_bf16.shape
        assert H == self.slab.hidden
        assert expert_ids.shape == (T, self.top_k)
        assert route_w.shape    == (T, self.top_k)
        in_device = x_bf16.device
        G = self.slab.num_experts
        K = self.top_k

        # ---- routing dispatch, vectorized on the device ----
        # Sort the T*K routed items by expert id; each expert's items form one
        # contiguous segment of the sorted order (stable sort keeps token order
        # within an expert). The only host round-trip is the [G] counts vector
        # — everything per-item stays on the GPU. (The old per-token Python
        # loop here cost ~10 ms/layer at prefill.)
        flat_e = expert_ids.reshape(-1)
        if flat_e.dtype != torch.int64:
            flat_e = flat_e.to(torch.int64)
        counts = torch.bincount(flat_e, minlength=G)          # [G] device
        order = torch.argsort(flat_e, stable=True)            # [T*K] device
        rows_sorted = torch.div(order, K, rounding_mode="floor")
        slots_sorted = order - rows_sorted * K
        sorted_e = flat_e[order]

        counts_np = counts.cpu().numpy()                      # one small D2H
        active = np.nonzero(counts_np)[0]
        cpu_frac = _cpu_prefill_fraction()
        # Only streamed (non-VRAM-cached) experts are candidates for the CPU
        # prefill split: the CPU's job there is to relieve the PCIe weight
        # stream, and cached experts read HBM. With a full cache the CPU
        # share drops to zero automatically.
        streamed = active[active >= self.cache_n]
        if (
            cpu_frac > 0.0
            and len(streamed)
            and int(counts_np.sum()) > self.m_cpu * max(1, len(active))
        ):
            # Large-batch (prefill) split: the streamed GPU bucket's cost is
            # dominated by streaming each expert's weights over PCIe
            # (per-expert, not per-row), the CPU bucket's by rows. Give the
            # CPU the smallest-count streamed experts until it holds
            # ~cpu_frac of their routed rows, and run the buckets
            # concurrently (GPU kernels are launched before the blocking AMX
            # call below). This engages the otherwise-idle CPU during prefill.
            by_count = streamed[np.argsort(counts_np[streamed], kind="stable")]
            csum = np.cumsum(counts_np[by_count])
            n_cpu = int(np.searchsorted(csum, cpu_frac * csum[-1], side="right"))
            cpu_experts = np.sort(by_count[:n_cpu])
            gpu_experts = np.sort(np.concatenate(
                (by_count[n_cpu:], active[active < self.cache_n])
            ))
        else:
            cpu_experts = active[counts_np[active] <= self.m_cpu]
            gpu_experts = active[counts_np[active] > self.m_cpu]

        # Per-expert start of its segment in the sorted order.
        seg_start_np = np.zeros(G, dtype=np.int64)
        np.cumsum(counts_np[:-1], out=seg_start_np[1:])

        out_fp32 = torch.zeros((T, H), dtype=torch.float32, device=in_device)

        # ---- CPU bucket, stage 1: gather routed rows and land them on the
        # host NOW, before any GPU-bucket kernel is enqueued — the D2H copy is
        # stream-ordered, so issuing it later would serialize the CPU bucket
        # behind the GPU bucket's grouped GEMMs instead of overlapping them.
        rows_cpu = slots_cpu = x_cat = None
        if len(cpu_experts):
            cpu_e_t = torch.from_numpy(cpu_experts).to(in_device)
            is_cpu_item = torch.isin(sorted_e, cpu_e_t)
            rows_cpu = rows_sorted[is_cpu_item]
            slots_cpu = slots_sorted[is_cpu_item]
            x_cat_t = (
                x_bf16.index_select(0, rows_cpu).to("cpu").contiguous()
            )
            x_cat = torch_bf16_to_np_bits(x_cat_t)

        # ---- GPU bucket (grouped INT8): enqueue first, asynchronously; the
        # AMX bucket below then runs on the host while the GPU works through
        # these kernels. When a VRAM expert cache exists, the bucket splits
        # into a cached partition (kernels read HBM) and a streamed partition
        # (kernels read pinned host over PCIe) — the cache holds experts
        # 0..cache_n-1 so the split is a simple id threshold.
        gpu_parts = []
        if len(gpu_experts):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU bucket non-empty but no CUDA device available.")

            x_gpu_bf16 = x_bf16.to(in_device, dtype=torch.bfloat16).contiguous()
            route_w_gpu = route_w.to(in_device, dtype=torch.float32)
            if self.cache_n > 0:
                partitions = (
                    (gpu_experts[gpu_experts < self.cache_n], True),
                    (gpu_experts[gpu_experts >= self.cache_n], False),
                )
            else:
                partitions = ((gpu_experts, False),)
            for part, cached in partitions:
                if not len(part):
                    continue
                layout = self._build_layout(
                    part, counts_np, seg_start_np, sorted_e,
                    rows_sorted, slots_sorted, G, in_device,
                )
                rows_p, y_p = self._gpu_grouped_forward(
                    x_gpu_bf16, layout, route_w_gpu, cached=cached,
                )
                if rows_p is not None:
                    gpu_parts.append((rows_p, y_p))

        # ---- CPU bucket, stage 2: the blocking AMX call. The GIL and the
        # CUDA stream are both free here, so this host work overlaps the GPU
        # bucket's grouped GEMMs enqueued above.
        if x_cat is not None:
            slab = self.slab
            m_offsets = np.zeros(len(cpu_experts) + 1, dtype=np.int64)
            np.cumsum(counts_np[cpu_experts], out=m_offsets[1:])
            out_cat = np.empty((int(m_offsets[-1]), H), dtype=np.float32)
            _C.moe_expert_forward_batch(
                self.rt, x_cat, m_offsets, cpu_experts.astype(np.int64),
                slab.gate_int8.numpy(), slab.gate_scales.numpy(),
                slab.up_int8.numpy(),   slab.up_scales.numpy(),
                slab.down_int8.numpy(), slab.down_scales.numpy(),
                out_cat, slab.inter, slab.hidden,
            )
            y_cpu = torch.from_numpy(out_cat).to(in_device, non_blocking=True)
            w = route_w[rows_cpu, slots_cpu].to(torch.float32)
            out_fp32.index_add_(0, rows_cpu, y_cpu * w[:, None])

        for rows_p, y_p in gpu_parts:
            out_fp32.index_add_(0, rows_p, y_p)

        return out_fp32.to(torch.bfloat16)
