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

    def _gpu_grouped_forward(
        self,
        x_gpu: torch.Tensor,        # [T, H] bf16 device
        layout,                     # contiguous layout built in forward()
        route_w_gpu: torch.Tensor,  # [T, top_k] fp32 device
    ):
        """Run gate/up/down for all GPU-bucket experts in three grouped calls.

        ``layout`` is (M_grouped, idx_to_orig, slot_to_orig, offsets, experts,
        list_size): the AsymGEMM contiguous layout, built vectorized on the
        GPU in forward(). Returns (orig_rows_for_valid, y_weighted_fp32) where
        orig_rows_for_valid is a [n_valid] long tensor of source rows in the
        original [T, H] activations, and y_weighted_fp32 is [n_valid, H] fp32
        ready for ``out_fp32.index_add_(0, orig_rows, y_weighted_fp32)``.
        """
        slab = self.slab
        dev = x_gpu.device
        H, I = slab.hidden, slab.inter
        kb_h, kb_i = slab.kb_hidden, slab.kb_inter

        M_grouped, idx_to_orig, slot_to_orig, offsets, experts, list_size = layout
        if M_grouped == 0:
            return None, None

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

        # gate  — pinned B (slab.gate_int8), device SFB (slab.gate_sfb)
        d_gate = torch.empty(M_grouped, I, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8, sfa_h), (slab.gate_int8, slab.gate_sfb),
            d_gate, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
        )

        # up
        d_up = torch.empty(M_grouped, I, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8, sfa_h), (slab.up_int8, slab.up_sfb),
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
            (a2_int8, sfa_i), (slab.down_int8, slab.down_sfb),
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
        if cpu_frac > 0.0 and int(counts_np.sum()) > self.m_cpu * max(1, len(active)):
            # Large-batch (prefill) split: the GPU bucket's cost is dominated
            # by streaming each expert's weights over PCIe (per-expert, not
            # per-row), the CPU bucket's by rows. Give the CPU the
            # smallest-count experts until it holds ~cpu_frac of the routed
            # rows, and run the two buckets concurrently (GPU kernels are
            # launched before the blocking AMX call below). This engages the
            # otherwise-idle CPU during prefill.
            by_count = active[np.argsort(counts_np[active], kind="stable")]
            csum = np.cumsum(counts_np[by_count])
            n_cpu = int(np.searchsorted(csum, cpu_frac * csum[-1], side="right"))
            cpu_experts = np.sort(by_count[:n_cpu])
            gpu_experts = np.sort(by_count[n_cpu:])
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

        # ---- GPU bucket (grouped INT8 over pinned weights): enqueue first,
        # asynchronously; the AMX bucket below then runs on the host while the
        # GPU works through these kernels.
        gpu_rows = gpu_y = None
        if len(gpu_experts):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU bucket non-empty but no CUDA device available.")

            # Contiguous layout, vectorized: expert g's items land at
            # pad_start[g] + rank-within-segment; every expert's block is
            # padded to BLOCK_M for the kernel's per-expert row alignment.
            m_padded = ((counts_np[gpu_experts] + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            pad_start_np = np.zeros(G, dtype=np.int64)
            pad_start_np[gpu_experts] = np.concatenate(
                ([0], np.cumsum(m_padded[:-1]))
            )
            M_grouped = int(m_padded.sum())

            offsets_np = np.empty(2 * len(gpu_experts), dtype=np.int32)
            offsets_np[0::2] = pad_start_np[gpu_experts]
            offsets_np[1::2] = pad_start_np[gpu_experts] + m_padded
            experts_np = np.concatenate(
                (gpu_experts, [-1])
            ).astype(np.int32)

            gpu_e_t = torch.from_numpy(gpu_experts).to(in_device)
            seg_start = torch.from_numpy(seg_start_np).to(in_device)
            pad_start = torch.from_numpy(pad_start_np).to(in_device)

            is_gpu_item = torch.isin(sorted_e, gpu_e_t)
            pos = torch.nonzero(is_gpu_item, as_tuple=False).squeeze(-1)
            e_sel = sorted_e[pos]
            dest = pad_start[e_sel] + (pos - seg_start[e_sel])

            idx_to_orig = torch.full(
                (M_grouped,), -1, dtype=torch.long, device=in_device
            )
            slot_to_orig = torch.full_like(idx_to_orig, -1)
            idx_to_orig[dest] = rows_sorted[pos]
            slot_to_orig[dest] = slots_sorted[pos]

            layout = (
                M_grouped,
                idx_to_orig,
                slot_to_orig,
                torch.from_numpy(offsets_np).to(in_device),
                torch.from_numpy(experts_np).to(in_device),
                len(experts_np),
            )

            x_gpu_bf16 = x_bf16.to(in_device, dtype=torch.bfloat16).contiguous()
            route_w_gpu = route_w.to(in_device, dtype=torch.float32)
            gpu_rows, gpu_y = self._gpu_grouped_forward(
                x_gpu_bf16, layout, route_w_gpu,
            )

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

        if gpu_rows is not None:
            out_fp32.index_add_(0, gpu_rows, gpu_y)

        return out_fp32.to(torch.bfloat16)
