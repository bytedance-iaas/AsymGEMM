"""Unified INT8 MoE layer (asym_gemm.unified_moe.Layer).

One pinned-host INT8 weight slab, per-expert dispatch on routed token
count, CPU path via cpu_gemm (AMX), GPU path via torch._int_mm
(CUTLASS-backed). Both backends compute the same INT8 → INT32 → FP32
dequant — verified by tests/test_unified_moe.py.

This is the v1 implementation. The hand-written SM90 INT8 WGMMA kernel
(see docs/unified_moe.md milestone 0b) is the eventual GPU backend;
until it lands, _int_mm gives us the same arithmetic at a possibly
lower throughput (the qualitative dispatch result still holds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .. import _cpu_C as _C


# ---------------------------------------------------------------------------
# Quantization helpers (the §5 contract)
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


def quantize_per_token_int8_cpu(
    x_bf16_bits: np.ndarray,  # [M, K] uint16
) -> tuple[np.ndarray, np.ndarray]:
    """A-side per-row quant on CPU. Used for the GPU bucket's host-staged path
    (the GPU bucket actually re-quantizes on GPU via the helper below; this
    one is a reference for tests)."""
    x_fp32 = bf16_bits_to_fp32(x_bf16_bits)
    amax = np.abs(x_fp32).max(axis=1)
    scales = np.maximum(amax / 127.0, 1e-12).astype(np.float32)
    inv = (1.0 / scales)[:, None]
    q = np.clip(np.round(x_fp32 * inv), -127, 127).astype(np.int8)
    return q, scales


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
# BF16 <-> FP32 helpers (BF16 carried as uint16 bit pattern)
# ---------------------------------------------------------------------------

def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 → bf16. Returns uint16."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    # RTE: add (0x7FFF + (lower-half lsb of upper-half)) before truncation.
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    u32_rounded = u32 + rounding_bias
    return (u32_rounded >> 16).astype(np.uint16)


def bf16_bits_to_fp32(u16: np.ndarray) -> np.ndarray:
    """uint16 bit pattern → fp32 (exact upcast)."""
    u16 = np.ascontiguousarray(u16, dtype=np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def torch_bf16_to_np_bits(t: torch.Tensor) -> np.ndarray:
    """Torch BF16 tensor (CPU) → uint16 numpy view of the same bytes."""
    assert t.dtype == torch.bfloat16 and not t.is_cuda
    # bfloat16 storage is 2 bytes per elem; reinterpret as uint16.
    return t.contiguous().view(torch.uint16).numpy(force=True)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

def silu_fp32(x: np.ndarray) -> np.ndarray:
    # numerically safe: silu(x) = x * sigmoid(x)
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Expert weight container
# ---------------------------------------------------------------------------

@dataclass
class ExpertSlab:
    """One MoE layer's three projections (gate, up, down), quantized INT8,
    each stored row-major in pinned host memory plus an AMX-packed twin.

    Shapes (per expert):
        gate, up: [N_inter, K_hidden]  (B is N-major)
        down:    [K_hidden, N_inter]

    The AMX path consumes the *_packed buffers; the GPU path reads the
    row-major *_int8 + scales (mirrored to VRAM on first GPU use of each
    expert).
    """
    num_experts: int
    hidden: int
    inter: int

    # row-major INT8 (host, pinned via torch). [G, N, K] shaped tensors.
    gate_int8: torch.Tensor      # int8, pinned, [G, inter, hidden]
    gate_scales: torch.Tensor    # float32, pinned, [G, inter]
    up_int8: torch.Tensor        # int8, pinned, [G, inter, hidden]
    up_scales: torch.Tensor      # float32, pinned, [G, inter]
    down_int8: torch.Tensor      # int8, pinned, [G, hidden, inter]
    down_scales: torch.Tensor    # float32, pinned, [G, hidden]

    # AMX-packed twins (uint8 numpy arrays, 64-byte aligned, per expert).
    gate_packed: list = field(default_factory=list)  # list[np.ndarray]
    up_packed: list   = field(default_factory=list)
    down_packed: list = field(default_factory=list)

    # GPU mirror cache (expert_idx → {"gate": int8 tensor, "gate_s": fp32, …}).
    gpu_cache: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The unified layer
# ---------------------------------------------------------------------------

class Layer:
    """Unified INT8 MoE layer.

    Construct via ``Layer.from_bf16(...)``. Forward signature:

        y = layer.forward(x_bf16, expert_ids, route_w)

    where:
        x_bf16     : (T, hidden) torch.bfloat16, cuda or cpu
        expert_ids : (T, top_k) torch.int64 / int32
        route_w    : (T, top_k) torch.float32
        y          : (T, hidden) torch.bfloat16 on the same device as x
    """

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

        # Quantize all experts.
        gate_int8 = torch.empty(G, N_inter, K_hidden, dtype=torch.int8).pin_memory()
        gate_s    = torch.empty(G, N_inter,            dtype=torch.float32).pin_memory()
        up_int8   = torch.empty_like(gate_int8).pin_memory()
        up_s      = torch.empty_like(gate_s).pin_memory()
        down_int8 = torch.empty(G, K_hidden, N_inter, dtype=torch.int8).pin_memory()
        down_s    = torch.empty(G, K_hidden,           dtype=torch.float32).pin_memory()

        gate_packed, up_packed, down_packed = [], [], []
        for g in range(G):
            q, s = quantize_per_channel_int8(gate[g])
            gate_int8[g] = torch.from_numpy(q)
            gate_s[g]    = torch.from_numpy(s)
            gate_packed.append(_C.pack_b_int8_amx(q, s))

            q, s = quantize_per_channel_int8(up[g])
            up_int8[g] = torch.from_numpy(q)
            up_s[g]    = torch.from_numpy(s)
            up_packed.append(_C.pack_b_int8_amx(q, s))

            q, s = quantize_per_channel_int8(down[g])
            down_int8[g] = torch.from_numpy(q)
            down_s[g]    = torch.from_numpy(s)
            down_packed.append(_C.pack_b_int8_amx(q, s))

        slab = ExpertSlab(
            num_experts=G, hidden=K_hidden, inter=N_inter,
            gate_int8=gate_int8, gate_scales=gate_s,
            up_int8=up_int8,     up_scales=up_s,
            down_int8=down_int8, down_scales=down_s,
            gate_packed=gate_packed, up_packed=up_packed, down_packed=down_packed,
        )
        return cls(slab, top_k=top_k, cpu_threads=cpu_threads,
                   cuda_device=cuda_device, m_cpu=m_cpu)

    # -----------------------------------------------------------
    # dispatch knobs (debug/eval)
    # -----------------------------------------------------------

    def set_m_cpu(self, m_cpu: int) -> None:
        self.m_cpu = int(m_cpu)

    # -----------------------------------------------------------
    # the per-expert backend kernels
    # -----------------------------------------------------------

    def _cpu_expert_forward(
        self,
        e: int,
        x_bf16_bits: np.ndarray,    # [m_e, hidden] uint16 (rows for this expert)
    ) -> np.ndarray:                # [m_e, hidden] fp32
        """One expert's gate-up-SwiGLU-down on CPU via AMX INT8."""
        slab = self.slab
        m_e = x_bf16_bits.shape[0]
        H, I = slab.hidden, slab.inter

        # gate, up: BF16 @ INT8.T → FP32. C is [m_e, inter].
        c_gate = np.empty((m_e, I), dtype=np.float32)
        c_up   = np.empty((m_e, I), dtype=np.float32)
        _C.gemm_bf16_int8_packed(self.rt, x_bf16_bits, slab.gate_packed[e],
                                  c_gate, I, H, 1.0, 0.0)
        _C.gemm_bf16_int8_packed(self.rt, x_bf16_bits, slab.up_packed[e],
                                  c_up,   I, H, 1.0, 0.0)

        # SwiGLU: silu(gate) * up. FP32 throughout.
        act = silu_fp32(c_gate) * c_up      # [m_e, inter] fp32

        # Down: need BF16 input → re-quantize.  Convert FP32 act to BF16 bits.
        act_bf16_bits = fp32_to_bf16_bits(act)
        c_down = np.empty((m_e, H), dtype=np.float32)
        _C.gemm_bf16_int8_packed(self.rt, act_bf16_bits, slab.down_packed[e],
                                  c_down, H, I, 1.0, 0.0)
        return c_down

    def _ensure_gpu_expert(self, e: int) -> dict:
        """Lazy-mirror the expert's INT8 weights to VRAM. Cached."""
        if e in self.slab.gpu_cache:
            return self.slab.gpu_cache[e]
        dev = f"cuda:{self.cuda_device}"
        entry = {
            "gate":   self.slab.gate_int8[e].to(dev, non_blocking=True),
            "gate_s": self.slab.gate_scales[e].to(dev, non_blocking=True),
            "up":     self.slab.up_int8[e].to(dev, non_blocking=True),
            "up_s":   self.slab.up_scales[e].to(dev, non_blocking=True),
            "down":   self.slab.down_int8[e].to(dev, non_blocking=True),
            "down_s": self.slab.down_scales[e].to(dev, non_blocking=True),
        }
        self.slab.gpu_cache[e] = entry
        return entry

    @staticmethod
    def _int_mm_padded(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        """torch._int_mm requires M > 16. Pad with zero rows if needed and
        slice the result. This is only hit when m_cpu is overridden low
        (e.g. =0 for dispatch-invariance tests); the production path keeps
        m_e > m_cpu >= 16, so the pad is dead code at default settings."""
        m, k = a.shape
        if m > 16:
            return torch._int_mm(a, b_t)
        pad = 17 - m
        a_pad = torch.nn.functional.pad(a, (0, 0, 0, pad))  # zero rows
        out = torch._int_mm(a_pad, b_t)
        return out[:m]

    def _gpu_expert_forward(
        self,
        e: int,
        x_bf16: torch.Tensor,   # [m_e, hidden] BF16 on CUDA
    ) -> torch.Tensor:           # [m_e, hidden] BF16 on CUDA
        """One expert's gate-up-SwiGLU-down on GPU via _int_mm."""
        w = self._ensure_gpu_expert(e)
        # A-side per-token quant.
        a_int8, sA = quantize_per_token_int8_gpu(x_bf16)
        # gate: A_int8 @ gate.T → INT32, then scale.
        c_gate_int = self._int_mm_padded(a_int8, w["gate"].t())     # [m_e, inter]
        c_up_int   = self._int_mm_padded(a_int8, w["up"].t())       # [m_e, inter]
        c_gate = c_gate_int.float() * sA[:, None] * w["gate_s"][None, :]
        c_up   = c_up_int.float()   * sA[:, None] * w["up_s"][None, :]
        # SwiGLU
        act = torch.nn.functional.silu(c_gate) * c_up
        # Down GEMM: re-quantize act (bf16 cast first to mirror CPU path).
        act_bf16 = act.to(torch.bfloat16)
        a2_int8, sA2 = quantize_per_token_int8_gpu(act_bf16)
        c_down_int = self._int_mm_padded(a2_int8, w["down"].t())    # [m_e, hidden]
        c_down = c_down_int.float() * sA2[:, None] * w["down_s"][None, :]
        return c_down.to(torch.bfloat16)

    # -----------------------------------------------------------
    # main forward (dispatch + scatter/gather + weighted reduce)
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

        # Move routing to CPU for dispatch decisions.
        expert_ids_cpu = expert_ids.to("cpu").to(torch.int64)
        # Per-expert token lists. token_slot[(t, j)] = (expert, original-row, slot-id).
        per_expert: list[list[int]] = [[] for _ in range(self.slab.num_experts)]
        per_expert_slot: list[list[int]] = [[] for _ in range(self.slab.num_experts)]
        # Each (t, j) routes one (token, expert) edge.
        ei = expert_ids_cpu.numpy()
        for t in range(T):
            for j in range(self.top_k):
                e = int(ei[t, j])
                per_expert[e].append(t)
                per_expert_slot[e].append(j)

        # Bucket experts by m_e against the threshold.
        cpu_experts, gpu_experts = [], []
        for e, rows in enumerate(per_expert):
            if not rows:
                continue
            (cpu_experts if len(rows) <= self.m_cpu else gpu_experts).append(e)

        # Output accumulator on the input's device (we'll write FP32 then cast).
        out_fp32 = torch.zeros((T, H), dtype=torch.float32, device=in_device)

        # ---- CPU bucket ----
        if cpu_experts:
            # Stage inputs on CPU.
            x_cpu_bf16 = x_bf16.to("cpu", dtype=torch.bfloat16).contiguous()
            x_bits_all = torch_bf16_to_np_bits(x_cpu_bf16)             # [T, H] uint16
            route_w_cpu = route_w.to("cpu", dtype=torch.float32).numpy()  # [T, top_k]
            cpu_results: list[tuple[int, list[int], list[int], np.ndarray]] = []
            for e in cpu_experts:
                rows = per_expert[e]
                slots = per_expert_slot[e]
                gathered = np.ascontiguousarray(x_bits_all[rows])     # [m_e, H]
                y = self._cpu_expert_forward(e, gathered)              # [m_e, H] fp32
                cpu_results.append((e, rows, slots, y))
            # Scatter + weighted-reduce on CPU then push to device.
            cpu_out = np.zeros((T, H), dtype=np.float32)
            for e, rows, slots, y in cpu_results:
                w = route_w_cpu[rows, slots][:, None]                   # [m_e, 1]
                np.add.at(cpu_out, rows, y * w)
            out_fp32 += torch.from_numpy(cpu_out).to(in_device, non_blocking=True)

        # ---- GPU bucket ----
        if gpu_experts:
            x_gpu_bf16 = x_bf16.to(in_device, dtype=torch.bfloat16).contiguous()
            for e in gpu_experts:
                rows = per_expert[e]
                slots = per_expert_slot[e]
                idx = torch.tensor(rows, device=in_device, dtype=torch.long)
                gathered = x_gpu_bf16.index_select(0, idx)             # [m_e, H] bf16
                y_bf16 = self._gpu_expert_forward(e, gathered)         # [m_e, H] bf16
                # Weight + scatter-add.
                w = route_w.to(in_device, dtype=torch.float32)[
                    torch.tensor(rows, device=in_device, dtype=torch.long),
                    torch.tensor(slots, device=in_device, dtype=torch.long),
                ]                                                       # [m_e]
                y_w = y_bf16.float() * w[:, None]
                out_fp32.index_add_(0, idx, y_w)

        return out_fp32.to(torch.bfloat16)
