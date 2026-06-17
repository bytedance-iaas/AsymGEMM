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
    ):
        self.slab = slab
        self.top_k = top_k
        self.cuda_device = cuda_device
        self.rt = _C.Runtime(cpu_threads)
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
                   cuda_device=cuda_device, m_cpu=m_cpu)

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

    def _build_contiguous_layout(
        self,
        gpu_experts: list[int],
        per_expert: list[list[int]],
        per_expert_slot: list[list[int]],
        device: torch.device,
        block_m: int = BLOCK_M,
    ):
        """Pack the GPU-bucket experts into the AsymGEMM contiguous layout.

        Returns:
            M_grouped     : total padded row count
            idx_to_orig   : [M_grouped] long device — original token row per
                            grouped row (-1 if padding)
            slot_to_orig  : [M_grouped] long device — original top-k slot per
                            grouped row (-1 if padding)
            offsets       : [2*K] int32 device — flat (start, end) per expert
            experts       : [K+1] int32 device — expert ids + -1 sentinel
            list_size     : int                   — len(experts)

        Each expert's m_e is padded up to a multiple of block_m so the kernel's
        per-expert row range satisfies its alignment requirement.
        """
        idx_list:  list[int] = []
        slot_list: list[int] = []
        offsets:   list[int] = []
        experts:   list[int] = []
        cur = 0
        for e in gpu_experts:
            rows  = per_expert[e]
            slots = per_expert_slot[e]
            m_e = len(rows)
            if m_e == 0:
                continue
            m_padded = ((m_e + block_m - 1) // block_m) * block_m
            idx_list.extend(rows)
            slot_list.extend(slots)
            if m_padded > m_e:
                idx_list.extend([-1] * (m_padded - m_e))
                slot_list.extend([-1] * (m_padded - m_e))
            offsets.append(cur)
            offsets.append(cur + m_padded)
            experts.append(e)
            cur += m_padded
        experts.append(-1)

        if cur == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return (0, empty, empty,
                    torch.empty(0, dtype=torch.int32, device=device),
                    torch.tensor([-1], dtype=torch.int32, device=device), 1)

        return (
            cur,
            torch.tensor(idx_list,  dtype=torch.long, device=device),
            torch.tensor(slot_list, dtype=torch.long, device=device),
            torch.tensor(offsets,   dtype=torch.int32, device=device),
            torch.tensor(experts,   dtype=torch.int32, device=device),
            len(experts),
        )

    def _gpu_grouped_forward(
        self,
        x_gpu: torch.Tensor,        # [T, H] bf16 device
        gpu_experts: list[int],
        per_expert: list[list[int]],
        per_expert_slot: list[list[int]],
        route_w_gpu: torch.Tensor,  # [T, top_k] fp32 device
    ):
        """Run gate/up/down for all GPU-bucket experts in three grouped calls.

        Returns (orig_rows_for_valid, y_weighted_fp32) where
        orig_rows_for_valid is a [n_valid] long tensor of source rows in the
        original [T, H] activations, and y_weighted_fp32 is [n_valid, H] fp32
        ready for ``out_fp32.index_add_(0, orig_rows, y_weighted_fp32)``.
        """
        slab = self.slab
        dev = x_gpu.device
        H, I = slab.hidden, slab.inter
        kb_h, kb_i = slab.kb_hidden, slab.kb_inter

        M_grouped, idx_to_orig, slot_to_orig, offsets, experts, list_size = (
            self._build_contiguous_layout(gpu_experts, per_expert,
                                          per_expert_slot, dev))
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

        # Per-expert token / slot lists (host-side dispatch decisions).
        expert_ids_cpu = expert_ids.to("cpu").to(torch.int64)
        per_expert:      list[list[int]] = [[] for _ in range(self.slab.num_experts)]
        per_expert_slot: list[list[int]] = [[] for _ in range(self.slab.num_experts)]
        ei = expert_ids_cpu.numpy()
        for t in range(T):
            for j in range(self.top_k):
                e = int(ei[t, j])
                per_expert[e].append(t)
                per_expert_slot[e].append(j)

        cpu_experts, gpu_experts = [], []
        for e, rows in enumerate(per_expert):
            if not rows:
                continue
            (cpu_experts if len(rows) <= self.m_cpu else gpu_experts).append(e)

        out_fp32 = torch.zeros((T, H), dtype=torch.float32, device=in_device)

        # ---- CPU bucket (unchanged from v1) ----
        if cpu_experts:
            x_cpu_bf16 = x_bf16.to("cpu", dtype=torch.bfloat16).contiguous()
            x_bits_all = torch_bf16_to_np_bits(x_cpu_bf16)
            route_w_cpu = route_w.to("cpu", dtype=torch.float32).numpy()

            cpu_out = np.zeros((T, H), dtype=np.float32)
            for e in cpu_experts:
                rows  = per_expert[e]
                slots = per_expert_slot[e]
                gathered = np.ascontiguousarray(x_bits_all[rows])
                y = self._cpu_expert_forward(e, gathered)
                w = route_w_cpu[rows, slots][:, None]
                np.add.at(cpu_out, rows, y * w)
            out_fp32 += torch.from_numpy(cpu_out).to(in_device, non_blocking=True)

        # ---- GPU bucket (grouped INT8 over pinned weights) ----
        if gpu_experts:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU bucket non-empty but no CUDA device available.")
            x_gpu_bf16 = x_bf16.to(in_device, dtype=torch.bfloat16).contiguous()
            route_w_gpu = route_w.to(in_device, dtype=torch.float32)
            orig_rows, y_w = self._gpu_grouped_forward(
                x_gpu_bf16, gpu_experts, per_expert, per_expert_slot, route_w_gpu,
            )
            if orig_rows is not None:
                out_fp32.index_add_(0, orig_rows, y_w)

        return out_fp32.to(torch.bfloat16)
