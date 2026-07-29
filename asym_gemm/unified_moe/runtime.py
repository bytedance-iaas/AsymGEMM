"""Unified INT8 MoE layer (asym_gemm.unified_moe.Layer) — v3.

Per-expert dispatch, two modes:
- static (default): routed token count vs threshold —
    m_e <= m_cpu  → CPU AMX INT8 path (cpu_gemm, stride-aware row-major B)
    m_e >  m_cpu  → SM90 INT8 grouped-MoE WGMMA kernel (asym_gemm)
  plus the ASYMGEMM_CPU_PREFILL_FRACTION large-batch heuristic below.
- adaptive (``adaptive=True`` or ``set_adaptive()``): a runtime cost model
  partitions the streamed experts to balance the two buckets' predicted
  finish times (they execute concurrently); VRAM-cached experts always
  stay on the GPU. See ``dispatch_model.py`` and ``adaptive_dispatch.md``.
  ``layer.calibrate()`` seeds the model.

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
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import asym_gemm
from .. import _cpu_C as _C
from .dispatch_model import DispatchModel

try:  # fused kernels for the VRAM-cached paths (optional dependency)
    from . import _triton as _tk
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing or broken
    _tk = None
    _HAS_TRITON = False

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
# where the two buckets run concurrently: the GPU's streamed-expert cost is
# ~per expert (weight transport), the CPU cost scales ~per row, so the
# CPU takes the smallest-count streamed experts. 0 disables the split
# (GPU-only prefill, CPU idle). Override via ASYMGEMM_CPU_PREFILL_FRACTION.
#
# Default from a sweep with copy-engine staging on 2x8457C (64 threads,
# NUMA-TP) + H200, Qwen3-235B, cache=24 hot, chunked 4096 + mixed:
#   single 3500-tok TTFT:  0.0 -> 2179, 0.1 -> 2281, 0.2 -> 2512,
#                          0.3 -> 2762 ms
#   32x3500 conc16 r.25:   E2E 0.0 -> 15.4, 0.1 -> 10.1, 0.2 -> 9.5,
#                          0.3 -> 17.9 s
#   32x10K  conc16 r.1:    E2E 0.0 -> 9.0, 0.1 -> 7.3, 0.2 -> 9.4,
#                          0.3 -> 22.4 s
# 0.1 is best or within noise everywhere. Larger fractions starve decode
# tokens (mixed chunks queue behind CPU prefill rows); before staging the
# balance sat near 0.3 because the GPU streamed path was 2.3x slower.
_CPU_PREFILL_FRACTION: Optional[float] = None


def _cpu_prefill_fraction() -> float:
    global _CPU_PREFILL_FRACTION
    if _CPU_PREFILL_FRACTION is None:
        _CPU_PREFILL_FRACTION = float(
            os.getenv("ASYMGEMM_CPU_PREFILL_FRACTION", "0.1")
        )
    return _CPU_PREFILL_FRACTION


# ---------------------------------------------------------------------------
# NUMA tensor-parallelism (ASYMGEMM_NUMA_TP=1)
#
# Expert weights split into two pinned host halves — experts [0, G/2) on
# node 0's memory, [G/2, G) on node 1's — and the shared CPU runtime runs two
# worker pools, each affinity-bound to its node (see _cpu_C.Runtime(threads,
# node_a, node_b)). Every MoE call then reads expert weights at BOTH sockets'
# local bandwidth; a single pool bound to one node gets one socket's, and an
# interleaved slab pays ~50% remote (UPI) accesses.
# ---------------------------------------------------------------------------

def _numa_tp_enabled() -> bool:
    return os.getenv("ASYMGEMM_NUMA_TP", "0") == "1"


# ---------------------------------------------------------------------------
# Copy-engine staging for streamed (non-VRAM-cached) expert weights
#
# The SM90 asym kernel's in-kernel TMA reads of pinned host memory top out at
# ~21.5 GB/s on PCIe Gen5 (single-slot B smem: the fetch of tile k+1 waits for
# tile k to be consumed), while a plain copy-engine H2D on the same buffers
# reaches ~50 GB/s. Staging swaps the transport: the streamed partition's
# expert weights are copied into a double-buffered HBM ring on a side stream
# (overlapping the cached-partition GEMMs and the CPU bucket), and the grouped
# GEMMs read HBM (~320 GB/s) after a stream-ordered event wait. Same bytes
# over the link, ~2.3x the bandwidth (measured end-to-end: 28.8 -> 13.5 ms for
# a 654 MB layer partition).
# ---------------------------------------------------------------------------

def _stage_streamed() -> bool:
    """ASYMGEMM_STAGE_STREAMED=0 disables copy-engine staging of streamed
    expert weights (falls back to in-kernel TMA reads of pinned host)."""
    return os.getenv("ASYMGEMM_STAGE_STREAMED", "1") == "1"


class _StageRing:
    """Double-buffered HBM staging slots for one slab geometry.

    Two slots so layer L+1's copies can start (on the side stream) while
    layer L's staged GEMMs are still reading the other slot; ``free_ev[i]``
    is recorded on the compute stream after the last GEMM that reads slot i,
    and the side stream waits on it before overwriting the slot.
    """

    def __init__(self, slab: "ExpertSlab", cache_n: int, device: str):
        n = slab.num_experts - cache_n if cache_n > 0 else slab.num_experts
        opts = dict(dtype=torch.int8, device=device)
        sf = dict(dtype=torch.float32, device=device)
        # The buffers keep their FULL [n, ...] shape every call (a partition
        # only fills and schedules the first P groups): num_groups is a JIT
        # compile-time constant, so a varying first dim would mint a new
        # kernel variant per partition size — an unbounded set, each costing
        # seconds of NVRTC mid-serving. Fixed shapes = one variant per
        # projection, compiled once at warmup.
        self.slots = [
            {
                "gate": torch.empty(n, slab.inter, slab.hidden, **opts),
                "up":   torch.empty(n, slab.inter, slab.hidden, **opts),
                "down": torch.empty(n, slab.hidden, slab.inter, **opts),
                "gate_sfb": torch.empty(n, slab.inter, slab.kb_hidden, **sf),
                "up_sfb":   torch.empty(n, slab.inter, slab.kb_hidden, **sf),
                "down_sfb": torch.empty(n, slab.hidden, slab.kb_inter, **sf),
            }
            for _ in range(2)
        ]
        self.stream = torch.cuda.Stream(device=device)
        self.free_ev = [torch.cuda.Event(), torch.cuda.Event()]
        for ev in self.free_ev:
            ev.record()
        self._i = 0

    def next_slot(self) -> int:
        i = self._i
        self._i ^= 1
        return i


_STAGE_RINGS: dict = {}
_STAGE_DISABLED = False    # set on OOM: fall back to direct TMA for the session


def _get_stage_ring(slab: "ExpertSlab", cache_n: int, device: str):
    """Shared ring per (device, geometry) — every layer of a model reuses it."""
    global _STAGE_DISABLED
    if _STAGE_DISABLED or not _stage_streamed():
        return None
    key = (device, slab.num_experts, cache_n, slab.inter, slab.hidden)
    ring = _STAGE_RINGS.get(key)
    if ring is None:
        try:
            ring = _StageRing(slab, cache_n, device)
        except torch.cuda.OutOfMemoryError:
            import logging
            logging.getLogger(__name__).warning(
                "AsymGEMM: not enough HBM for the streamed-weight staging "
                "ring; falling back to in-kernel TMA reads of pinned host.")
            _STAGE_DISABLED = True
            return None
        _STAGE_RINGS[key] = ring
    return ring


def _num_numa_nodes() -> int:
    import glob as _glob
    return len(_glob.glob("/sys/devices/system/node/node[0-9]*"))


_libc = None


_pin_keepalive: list = []      # mmap objects backing node-bound pinned tensors
_cudart_handle = None


def _libc_handle():
    global _libc
    import ctypes
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    return _libc


def _cudart():
    global _cudart_handle
    if _cudart_handle is not None:
        return _cudart_handle
    import ctypes
    import glob as _glob
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart_handle = ctypes.CDLL(name)
            return _cudart_handle
        except OSError:
            continue
    for so in _glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib",
                                      "libcudart*")):
        _cudart_handle = ctypes.CDLL(so)
        return _cudart_handle
    raise RuntimeError("could not locate libcudart for cudaHostRegister")


def _pin_on_node(t: torch.Tensor, node: int) -> torch.Tensor:
    """Copy `t` into pinned host memory whose pages live on NUMA `node`.

    torch's pin_memory() goes through the caching host allocator, which
    recycles previously-created pinned segments — their pages keep whatever
    node they were first faulted on, so a mempolicy at allocation time is
    ignored (verified: a long-lived server ends up with every slab half on
    one node). Instead: anonymous mmap -> mbind(MPOL_BIND, node) -> copy in
    (faults the pages on `node`) -> cudaHostRegister (pin + UVA mapping for
    the GPU's PCIe/TMA reads). Falls back to plain pin_memory() on any
    failure.
    """
    import ctypes
    import mmap as _mmap

    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    try:
        mm = _mmap.mmap(-1, nbytes)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        mask = ctypes.c_ulong(1 << node)
        rc = _libc_handle().syscall(
            237,                                   # mbind (x86_64)
            ctypes.c_void_p(addr), ctypes.c_size_t(nbytes),
            2,                                     # MPOL_BIND
            ctypes.byref(mask), 65, 0,
        )
        if rc != 0:
            raise OSError("mbind failed")
        dst = torch.frombuffer(mm, dtype=t.dtype, count=t.numel()).view(t.shape)
        dst.copy_(t)                               # faults pages on `node`
        cr = _cudart()
        err = cr.cudaHostRegister(ctypes.c_void_p(addr),
                                  ctypes.c_size_t(nbytes), 0)
        if err != 0:
            raise OSError(f"cudaHostRegister returned {err}")
        _pin_keepalive.append(mm)
        return dst
    except Exception as e:  # pragma: no cover - fallback keeps things working
        import logging
        logging.getLogger(__name__).warning(
            "AsymGEMM NUMA-TP: node-bound pinned alloc failed (%s); falling "
            "back to pin_memory() with unknown placement.", e)
        return t.pin_memory()


# ---------------------------------------------------------------------------
# Expert routing statistics (for hot-expert cache selection)
#
# ASYMGEMM_RECORD_EXPERT_STATS=<path.pt>: eager forwards accumulate per-layer
# expert counts; dumped at exit, keyed by layer creation order. Feed the file
# back via ASYMGEMM_CACHE_HOT_EXPERTS=<path.pt> to cache each layer's top-N
# most-routed experts instead of experts 0..N-1.
# ---------------------------------------------------------------------------

_ALL_LAYERS: list = []
_stats_atexit_registered = False
_stats_step = 0


def _record_stats_path():
    return os.getenv("ASYMGEMM_RECORD_EXPERT_STATS", "")


def _dump_expert_stats():
    path = _record_stats_path()
    if not path:
        return
    out = {}
    for idx, layer in enumerate(_ALL_LAYERS):
        c = getattr(layer, "_stat_counts", None)
        if c is not None:
            out[idx] = c.cpu()
    if out:
        torch.save(out, path)


def _maybe_register_stats(layer) -> None:
    global _stats_atexit_registered
    _ALL_LAYERS.append(layer)
    if _record_stats_path() and not _stats_atexit_registered:
        import atexit
        atexit.register(_dump_expert_stats)
        _stats_atexit_registered = True


# ---------------------------------------------------------------------------
# Live per-expert routing counts (decode phase).
#
# ASYMGEMM_CACHE_TRACK_ROUTING > 0 allocates each Layer's _live_counts and
# starts accumulating. Nothing consumes the window yet; this is the signal a
# residency policy needs, kept separate from the policy itself.
# ---------------------------------------------------------------------------


def _cache_track_routing() -> bool:
    return int(os.getenv("ASYMGEMM_CACHE_TRACK_ROUTING", "0")) > 0


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

def torch_bf16_to_np_bits(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16 and not t.is_cuda
    return t.contiguous().view(torch.uint16).numpy(force=True)


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

    # NUMA-TP: when split, the *_int8/*_scales tensors above hold experts
    # [0, g_split) (pinned on node 0) and the *_b tensors below hold experts
    # [g_split, G) (pinned on node 1, local index g - g_split). Unsplit:
    # g_split == num_experts and the *_b tensors are None.
    g_split: int = 0
    gate_int8_b: Optional[torch.Tensor] = None
    gate_scales_b: Optional[torch.Tensor] = None
    up_int8_b: Optional[torch.Tensor] = None
    up_scales_b: Optional[torch.Tensor] = None
    down_int8_b: Optional[torch.Tensor] = None
    down_scales_b: Optional[torch.Tensor] = None

    def expert_rows(self, name: str, ids) -> torch.Tensor:
        """Gather experts `ids` (iterable of global ids) for tensor `name`
        ('gate_int8', 'up_scales', ...) across the split halves."""
        a = getattr(self, name)
        b = getattr(self, name + "_b")
        if b is None:
            return a[list(ids)]
        rows = [a[i] if i < self.g_split else b[i - self.g_split] for i in ids]
        return torch.stack(rows)

    def expert_row(self, name: str, i: int) -> torch.Tensor:
        """One expert's slice of `name` as a **view** into pinned host memory
        (split-aware). Unlike expert_rows this never copies or stacks, so the
        result is a valid source for a true async H2D DMA — see
        Layer.update_gpu_cache, where staging through a torch.cat'd (and
        therefore pageable) temporary cost ~10x the bandwidth."""
        b = getattr(self, name + "_b")
        a = getattr(self, name)
        if b is None or i < self.g_split:
            return a[i]
        return b[i - self.g_split]


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
        adaptive: bool = False,
        runtime: Optional["_C.Runtime"] = None,
        dispatch_model: Optional[DispatchModel] = None,
    ):
        self.slab = slab
        self.top_k = top_k
        self.cuda_device = cuda_device
        # An injected `runtime` (e.g. a single pool shared by every MoE layer
        # in a serving framework) takes precedence; `cpu_threads` is only used
        # for the build-your-own default and is ignored when `runtime` is given.
        self.rt = runtime if runtime is not None else _C.Runtime(cpu_threads)
        self.m_cpu = m_cpu
        # Adaptive dispatch: cost-model makespan partition of the streamed
        # experts instead of the static m_cpu threshold / cpu_frac heuristic
        # (VRAM-cached experts always stay on the GPU). Observations are
        # recorded in both modes so a calibrated/warmed model carries over
        # when adaptive is enabled. An injected `dispatch_model` (one
        # instance shared by every same-shape MoE layer, pooling their
        # observations) takes precedence over the shape-aware default.
        # Cross-shape warm start: build it via
        # DispatchModel.from_rates(other.rates(), hidden=…, inter=…).
        self.adaptive = adaptive
        if dispatch_model is not None:
            if dispatch_model.hidden is not None:
                assert (dispatch_model.hidden, dispatch_model.inter) == \
                    (slab.hidden, slab.inter), \
                    "shared dispatch_model shape mismatch: " \
                    f"{(dispatch_model.hidden, dispatch_model.inter)} vs " \
                    f"{(slab.hidden, slab.inter)}"
            self.dispatch = dispatch_model
        else:
            self.dispatch = DispatchModel(block_m=BLOCK_M,
                                          hidden=slab.hidden,
                                          inter=slab.inter)
        self._gpu_bucket_calls = 0   # first GPU call is JIT warm-up; not recorded
        self.cache_n = 0
        _maybe_register_stats(self)
        self._build_gpu_cache()
        # Live per-expert routing-count window for ASYMGEMM_CACHE_TRACK_ROUTING.
        # Only meaningful once a VRAM cache exists.
        # Read the env once, at build time: _cached_gpu_decode consults this on
        # every decode step (and inside CUDA-graph capture), so an os.getenv
        # there would be wasted work on the hot path.
        self._refresh_enabled = (
            self.cache_n > 0 and _cache_track_routing()
        )
        # float32, not int: a consumer will want to decay this window rather
        # than zero it, and mul_(0.9) on an integer tensor truncates to zero.
        self._live_counts = (
            torch.zeros(slab.num_experts, dtype=torch.float32,
                        device=f"cuda:{self.cuda_device}")
            if self._refresh_enabled and torch.cuda.is_available() else None
        )
        # Refreshes applied so far — drives a cold-start budget once a policy
        # exists; harmless and unread until then.
        self._refresh_n = 0

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
        G = s.num_experts
        n = min(n, G)
        dev = f"cuda:{self.cuda_device}"
        ids = self._pick_cache_ids(n, G)
        idx_t = torch.tensor(ids, dtype=torch.int64, device=dev)
        # gate and up are fused along the output dim ([n, 2*inter, hidden],
        # gate rows first): per-channel scales make the concat exact, and one
        # grouped call replaces two — one less launch and one less pass over
        # the quantized activations per layer.
        self.cache_gateup_int8 = torch.cat(
            (s.expert_rows("gate_int8", ids), s.expert_rows("up_int8", ids)),
            dim=1,
        ).to(dev)
        self.cache_down_int8 = s.expert_rows("down_int8", ids).to(dev)
        # SFBs are already device-resident; build contiguous copies the kernel
        # can index 0..n-1 like the cached INT8 slabs.
        self.cache_gateup_sfb = torch.cat(
            (s.gate_sfb.index_select(0, idx_t),
             s.up_sfb.index_select(0, idx_t)), dim=1,
        ).contiguous()
        self.cache_down_sfb = s.down_sfb.index_select(0, idx_t).contiguous()
        cached_slot = torch.full((G,), -1, dtype=torch.int64, device=dev)
        cached_slot[idx_t] = torch.arange(n, dtype=torch.int64, device=dev)
        self.cached_slot = cached_slot
        self.cache_n = n
        # Host-side companions for the prefill partition split.
        self._cached_mask_np = np.zeros(G, dtype=bool)
        self._cached_mask_np[ids] = True
        self._slot_np = np.full(G, -1, dtype=np.int64)
        self._slot_np[ids] = np.arange(n, dtype=np.int64)
        # Constant expert list (+ sentinel) for the fully-cached any-T path
        # (only used when n == G, where slot i == expert i by construction).
        self._experts_all = torch.cat((
            torch.arange(G, dtype=torch.int32, device=dev),
            torch.tensor([-1], dtype=torch.int32, device=dev),
        ))

    def _pick_cache_ids(self, n: int, G: int) -> list:
        """Expert ids to cache, ascending (slot = rank among cached ids).

        With ASYMGEMM_CACHE_HOT_EXPERTS=<stats.pt> (a file produced by a
        profiling run under ASYMGEMM_RECORD_EXPERT_STATS), pick this layer's
        top-n most-routed experts; otherwise — or for a full cache, where
        selection is moot and the any-T path assumes slot i == expert i —
        experts 0..n-1.
        """
        path = os.getenv("ASYMGEMM_CACHE_HOT_EXPERTS", "")
        if path and n < G:
            try:
                stats = torch.load(path)
                counts = stats.get(len(_ALL_LAYERS) - 1)
                if counts is not None and counts.numel() == G:
                    top = torch.argsort(counts, descending=True)[:n]
                    return sorted(int(i) for i in top)
            except Exception as e:  # pragma: no cover - fall back to first-n
                import logging
                logging.getLogger(__name__).warning(
                    "AsymGEMM: failed to load expert stats from %s (%s); "
                    "caching experts 0..%d", path, e, n - 1)
        return list(range(n))

    def update_gpu_cache(self, new_expert_ids) -> bool:
        """Replace the VRAM cache's residents with `new_expert_ids` (a target
        set — order does not matter, slot assignment for kept experts is
        preserved).

        Diffs against the currently-cached ids (self._slot_np) and rewrites
        ONLY the slots whose resident actually changes, via in-place
        index_copy_/index_fill_ into the SAME tensor objects _build_gpu_cache
        allocated — never reassigns them. This matters for CUDA-graph replay
        safety: capturable.py reads cached_slot's live contents fresh on
        every replay (it is looked up as an ordinary tensor inside the
        captured region, not baked in as a capture-time constant), so
        swapping in a new tensor object would leave a captured graph
        pointing at stale/freed memory, while in-place content mutation is
        picked up correctly.

        Caller is responsible for cadence gating and for not calling this
        during CUDA graph capture (see refresh_gpu_caches). Returns True iff
        anything was actually copied.
        """
        G = self.slab.num_experts
        if self.cache_n <= 0 or self.cache_n >= G:
            return False  # no cache, or already fully cached: nothing to do
        new_set = set(int(i) for i in new_expert_ids)
        cur_ids = np.nonzero(self._slot_np >= 0)[0].tolist()
        cur_set = set(cur_ids)
        added = sorted(new_set - cur_set)
        if not added:
            return False  # identical residency (or new_set already a subset)
        evicted = sorted(cur_set - new_set)
        # |added| may exceed the number of evictable slots if new_set has
        # fewer distinct entries than cache_n; only evict as many as we have
        # replacements for, keeping the rest of the cache warm rather than
        # blanking slots we can't refill this round.
        evicted = evicted[: len(added)]
        added = added[: len(evicted)]
        if not added:
            return False

        s = self.slab
        dev = self.cached_slot.device
        slots = [int(self._slot_np[e]) for e in evicted]
        slots_t = torch.tensor(slots, dtype=torch.int64, device=dev)
        added_t = torch.tensor(added, dtype=torch.int64, device=dev)
        evicted_t = torch.tensor(evicted, dtype=torch.int64, device=dev)
        I = s.inter

        # INT8 weights: copy each expert's pinned slice STRAIGHT into its
        # destination slot. The obvious vectorized form —
        #   torch.cat((expert_rows(gate), expert_rows(up)), 1).to(dev)
        # — gathers, concatenates into a fresh *pageable* host tensor, and so
        # the H2D stages through a bounce buffer: measured 3.2 GB/s vs 36.8
        # GB/s for these per-expert DMAs off the already-pinned slab (11x, and
        # ~1039ms -> ~92ms for a full 48-layer refresh at Qwen3-30B-A3B
        # shapes). The Python loop is over at most cache_n experts and is
        # nowhere near the cost of the bytes it moves.
        for e, sl in zip(added, slots):
            self.cache_gateup_int8[sl, :I].copy_(
                s.expert_row("gate_int8", e), non_blocking=True)
            self.cache_gateup_int8[sl, I:].copy_(
                s.expert_row("up_int8", e), non_blocking=True)
            self.cache_down_int8[sl].copy_(
                s.expert_row("down_int8", e), non_blocking=True)

        # SFBs are already device-resident, so these are D2D gathers — the
        # vectorized form is right here.
        gu_sfb = torch.cat(
            (s.gate_sfb.index_select(0, added_t),
             s.up_sfb.index_select(0, added_t)), dim=1,
        ).contiguous()
        self.cache_gateup_sfb.index_copy_(0, slots_t, gu_sfb)
        self.cache_down_sfb.index_copy_(
            0, slots_t, s.down_sfb.index_select(0, added_t).contiguous()
        )

        self.cached_slot.index_fill_(0, evicted_t, -1)
        self.cached_slot.index_copy_(0, added_t, slots_t)

        self._cached_mask_np[evicted] = False
        self._slot_np[evicted] = -1
        self._cached_mask_np[added] = True
        self._slot_np[added] = np.asarray(slots, dtype=np.int64)
        return True

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

        # Decode-time routing signal. This MUST live here rather than in
        # Layer.forward(): forward() is skipped entirely under CUDA-graph
        # replay, so a counter kept there sees prefill only and any consumer
        # would end up choosing decode residency from prefill statistics.
        # index_add_ over fixed-shape tensors is capturable, so this
        # accumulates on every replay. Weighted by (route_w > 0) so padded /
        # masked slots (expert_ids is clamped >= 0, which would otherwise
        # inflate expert 0) contribute nothing.
        if self._refresh_enabled and self._live_counts is not None:
            _flat = expert_ids.reshape(-1)
            self._live_counts.index_add_(
                0, _flat,
                (route_w.reshape(-1) > 0).to(self._live_counts.dtype),
            )

        if _HAS_TRITON and TK <= 2048:
            # One fused kernel builds dest/offsets/experts (vs ~18 tensor ops).
            dest, offsets, experts_t = _tk.decode_layout(
                expert_ids.reshape(-1), self.cached_slot, Nc, S, BLOCK_M
            )
            valid = dest < M                        # parking rows sit past M
            dest_safe = dest.clamp_max(M - 1)
        else:
            slot = self.cached_slot[expert_ids.reshape(-1)]    # [TK]
            valid = slot >= 0
            sc = slot.clamp_min(0)

            # rank of each item within its expert; per-slot counts.
            onehot = torch.zeros(TK, Nc, dtype=torch.int32, device=dev)
            onehot.scatter_(1, sc.unsqueeze(1), valid.to(torch.int32).unsqueeze(1))
            cum = onehot.cumsum(0)                              # int64 (promoted)
            rank = cum.gather(1, sc.unsqueeze(1)).squeeze(1) - 1
            counts = cum[-1]                                    # [Nc] int64

            # Pack active experts into segments [seg*256, seg*256+count).
            active = counts > 0
            seg_of = active.to(torch.int64).cumsum(0) - 1       # [Nc]
            dump = torch.full_like(seg_of, S)                   # park inactive
            seg_idx = torch.where(active, seg_of, dump)
            seg_counts = torch.zeros(S + 1, dtype=torch.int64, device=dev)
            seg_counts.scatter_(0, seg_idx, counts)
            experts_t = torch.zeros(S + 1, dtype=torch.int32, device=dev)
            experts_t.scatter_(
                0, seg_idx, torch.arange(Nc, dtype=torch.int32, device=dev)
            )
            experts_t.narrow(0, S, 1).fill_(-1)                 # sentinel
            starts = (
                torch.arange(S, dtype=torch.int32, device=dev) * BLOCK_M
            )
            offsets = torch.empty(2 * S, dtype=torch.int32, device=dev)
            offsets[0::2] = starts
            offsets[1::2] = starts + seg_counts[:S].to(torch.int32)

            # Item destinations; invalid items park past M.
            arange_tk = torch.arange(TK, dtype=torch.int64, device=dev)
            dest_valid = seg_of.index_select(0, sc) * BLOCK_M + rank
            dest = torch.where(valid, dest_valid, M + arange_tk)
            dest_safe = torch.where(valid, dest_valid, torch.zeros_like(dest_valid))

        # Gather + quantize only the TK routed rows, scatter into the layout
        # (quantizing the full padded buffer would read ~30x the bytes).
        tok = torch.div(
            torch.arange(TK, dtype=torch.int64, device=dev), K,
            rounding_mode="floor",
        )
        x_items = x_bf16.index_select(0, tok)                   # [TK, H]
        q_items, s_items = (
            _tk.quant_rows(x_items) if _HAS_TRITON
            else quantize_per_token_int8_gpu(x_items)
        )
        a_int8 = torch.zeros(M + TK, H, dtype=torch.int8, device=dev)
        a_int8.index_copy_(0, dest, q_items)
        sfa = torch.zeros(M + TK, kb_h, dtype=torch.float32, device=dev)
        sfa.index_copy_(0, dest, s_items.unsqueeze(1).expand(TK, kb_h).contiguous())

        d_gu = torch.empty(M, 2 * I, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8[:M], sfa[:M]),
            (self.cache_gateup_int8, self.cache_gateup_sfb),
            d_gu, offsets, experts_t, S + 1, recipe=(1, 1, GRAN_K),
        )

        # Gather the routed rows FIRST, then activate: SwiGLU runs on TK rows
        # instead of the whole padded buffer. Rows of inactive segments are
        # uninitialized — gather via dest_safe so invalid items read a real
        # row (their weight is 0) and land in parking on the scatter back.
        gu_sel = d_gu.index_select(0, dest_safe)                # [TK, 2I]
        act_sel = torch.nn.functional.silu(gu_sel[:, :I]) * gu_sel[:, I:]
        # Down: re-quantize (BF16 round-trip matches the CPU path).
        a2_items, s2_items = (
            _tk.quant_rows(act_sel.to(torch.bfloat16)) if _HAS_TRITON
            else quantize_per_token_int8_gpu(act_sel.to(torch.bfloat16))
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

    def _cached_gpu_forward_any(
        self,
        x_bf16: torch.Tensor,       # [T, H] bf16 device
        expert_ids: torch.Tensor,   # [T, K] int64 device (clamped >= 0)
        route_w: torch.Tensor,      # [T, K] fp32 device (masked slots = 0)
    ) -> torch.Tensor:              # [T, H] bf16
        """Fully-cached eager forward for ANY batch size, with zero host syncs.

        Used when every expert is VRAM-cached: the whole layout (per-expert
        counts, BLOCK_M-padded offsets, item destinations) is computed on the
        device, so consecutive layers pipeline without the per-layer
        counts-D2H stall of the bucketed forward. Buffers are sized to the
        worst case T*K + G*BLOCK_M rows; the kernel only touches the ranges
        named by the device-computed offsets. (Unlike _cached_gpu_decode's
        fixed 256-row segments, per-expert row counts here are unbounded, so
        offsets derive from a padded cumsum.)
        """
        slab = self.slab
        dev = x_bf16.device
        G = slab.num_experts
        H, I = slab.hidden, slab.inter
        kb_h, kb_i = slab.kb_hidden, slab.kb_inter
        T, K = expert_ids.shape
        TK = T * K

        flat_e = expert_ids.reshape(-1)
        counts = torch.bincount(flat_e, minlength=G)            # [G] device
        padded = ((counts + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        pad_start = padded.cumsum(0) - padded
        seg_start = counts.cumsum(0) - counts
        order = torch.argsort(flat_e, stable=True)
        rows_sorted = torch.div(order, K, rounding_mode="floor")
        e_sel = flat_e[order]
        pos = torch.arange(TK, dtype=torch.int64, device=dev)
        dest = pad_start[e_sel] + (pos - seg_start[e_sel])      # [TK]

        M_alloc = TK + G * BLOCK_M                              # >= padded.sum()
        offsets = torch.empty(2 * G, dtype=torch.int32, device=dev)
        offsets[0::2] = pad_start.to(torch.int32)
        offsets[1::2] = (pad_start + counts).to(torch.int32)

        x_items = x_bf16.index_select(0, rows_sorted)           # [TK, H]
        q_items, s_items = (
            _tk.quant_rows(x_items) if _HAS_TRITON
            else quantize_per_token_int8_gpu(x_items)
        )
        a_int8 = torch.zeros(M_alloc, H, dtype=torch.int8, device=dev)
        a_int8.index_copy_(0, dest, q_items)
        sfa = torch.zeros(M_alloc, kb_h, dtype=torch.float32, device=dev)
        sfa.index_copy_(0, dest, s_items.unsqueeze(1).expand(TK, kb_h).contiguous())

        d_gu = torch.empty(M_alloc, 2 * I, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a_int8, sfa), (self.cache_gateup_int8, self.cache_gateup_sfb),
            d_gu, offsets, self._experts_all, G + 1, recipe=(1, 1, GRAN_K),
        )

        # Gather the routed rows first, then activate — SwiGLU runs on TK rows
        # instead of the whole padded buffer (~2x fewer bytes at prefill).
        gu_sel = d_gu.index_select(0, dest)                     # [TK, 2I]
        act_sel = torch.nn.functional.silu(gu_sel[:, :I]) * gu_sel[:, I:]
        a2_items, s2_items = (
            _tk.quant_rows(act_sel.to(torch.bfloat16)) if _HAS_TRITON
            else quantize_per_token_int8_gpu(act_sel.to(torch.bfloat16))
        )
        a2_int8 = torch.zeros(M_alloc, I, dtype=torch.int8, device=dev)
        a2_int8.index_copy_(0, dest, a2_items)
        sfa2 = torch.zeros(M_alloc, kb_i, dtype=torch.float32, device=dev)
        sfa2.index_copy_(0, dest, s2_items.unsqueeze(1).expand(TK, kb_i).contiguous())

        d_down = torch.empty(M_alloc, H, dtype=torch.float32, device=dev)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a2_int8, sfa2), (self.cache_down_int8, self.cache_down_sfb),
            d_down, offsets, self._experts_all, G + 1, recipe=(1, 1, GRAN_K),
        )

        w = route_w.reshape(-1).index_select(0, order)
        y_items = d_down.index_select(0, dest) * w[:, None]     # [TK, H] fp32
        out = torch.zeros(T, H, dtype=torch.float32, device=dev)
        out.index_add_(0, rows_sorted, y_items)
        return out.to(torch.bfloat16)

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
        adaptive: bool = False,
        runtime: Optional["_C.Runtime"] = None,
        dispatch_model: Optional[DispatchModel] = None,
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
            cuda_device=cuda_device, m_cpu=m_cpu, adaptive=adaptive,
            runtime=runtime, dispatch_model=dispatch_model,
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
        adaptive: bool = False,
        runtime: Optional["_C.Runtime"] = None,
        dispatch_model: Optional[DispatchModel] = None,
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
            cuda_device=cuda_device, m_cpu=m_cpu, adaptive=adaptive,
            runtime=runtime, dispatch_model=dispatch_model,
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
        adaptive: bool = False,
        runtime: Optional["_C.Runtime"] = None,
        dispatch_model: Optional[DispatchModel] = None,
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

        # NUMA-TP: repin the slab as two node-local halves (experts [0, G/2)
        # on node 0, the rest on node 1). Pinned pages can't migrate, so this
        # must happen at (re)pin time; the sources may be pinned or plain CPU
        # tensors — contiguous().pin_memory() under a bound mempolicy copies
        # either way.
        if _numa_tp_enabled() and _num_numa_nodes() >= 2 and G >= 2:
            g_split = G // 2
            halves = {}
            for name, t in (("gate_int8", gate_int8), ("gate_scales", gate_s),
                            ("up_int8", up_int8), ("up_scales", up_s),
                            ("down_int8", down_int8), ("down_scales", down_s)):
                halves[name] = _pin_on_node(t[:g_split], 0)
                halves[name + "_b"] = _pin_on_node(t[g_split:], 1)
            slab = ExpertSlab(
                num_experts=G, hidden=K_hidden, inter=N_inter,
                kb_hidden=kb_h, kb_inter=kb_i,
                gate_int8=halves["gate_int8"], gate_scales=halves["gate_scales"],
                up_int8=halves["up_int8"],     up_scales=halves["up_scales"],
                down_int8=halves["down_int8"], down_scales=halves["down_scales"],
                gate_sfb=gate_sfb, up_sfb=up_sfb, down_sfb=down_sfb,
                g_split=g_split,
                gate_int8_b=halves["gate_int8_b"],
                gate_scales_b=halves["gate_scales_b"],
                up_int8_b=halves["up_int8_b"],
                up_scales_b=halves["up_scales_b"],
                down_int8_b=halves["down_int8_b"],
                down_scales_b=halves["down_scales_b"],
            )
        else:
            slab = ExpertSlab(
                num_experts=G, hidden=K_hidden, inter=N_inter,
                kb_hidden=kb_h, kb_inter=kb_i,
                gate_int8=gate_int8, gate_scales=gate_s,
                up_int8=up_int8,     up_scales=up_s,
                down_int8=down_int8, down_scales=down_s,
                gate_sfb=gate_sfb, up_sfb=up_sfb, down_sfb=down_sfb,
                g_split=G,
            )
        return cls(slab, top_k=top_k, cpu_threads=cpu_threads,
                   cuda_device=cuda_device, m_cpu=m_cpu, adaptive=adaptive,
                   runtime=runtime, dispatch_model=dispatch_model)

    def set_m_cpu(self, m_cpu: int) -> None:
        """Force the static threshold (also disables adaptive dispatch)."""
        self.m_cpu = int(m_cpu)
        self.adaptive = False

    def set_adaptive(self, adaptive: bool = True) -> None:
        self.adaptive = bool(adaptive)

    def calibrate(self, t_points: tuple[int, ...] = (1, 4, 16, 64),
                  repeats: int = 2, seed: int = 0) -> dict:
        """Seed the dispatch cost model with forced all-CPU / all-GPU runs.

        Runs synthetic forwards through the *real* bucket code paths (so
        Python/gather/quant overheads are included in the observations).
        Optional — priors plus online refit converge without it, but this
        removes the warm-up period. Returns the model snapshot.
        """
        global _CPU_PREFILL_FRACTION
        prev_adaptive, prev_m_cpu = self.adaptive, self.m_cpu
        prev_frac = _CPU_PREFILL_FRACTION
        self.adaptive = False
        # Zero the prefill fraction so the m_cpu extremes really force the
        # buckets (the cpu_frac heuristic would otherwise siphon rows to the
        # CPU during the all-GPU runs).
        _CPU_PREFILL_FRACTION = 0.0
        G, H = self.slab.num_experts, self.slab.hidden
        rng = torch.Generator().manual_seed(seed)
        try:
            for T in t_points:
                x = torch.randn(T, H, generator=rng, dtype=torch.float32)
                x = x.to(torch.bfloat16)
                ids = torch.empty(T, self.top_k, dtype=torch.int64)
                for j in range(self.top_k):
                    ids[:, j] = (torch.arange(T) + j) % G
                w = torch.full((T, self.top_k), 1.0 / self.top_k)

                # One unrecorded warm-up forward per shape and backend:
                # first-touch effects (per-shape kernel JIT, worker-pool /
                # allocator warm-up) inflate that run 5-10x, and a single
                # outlier in rings this small wrecks the ridge fit.
                self.m_cpu = 10**9                  # force all-CPU
                self.dispatch.paused = True
                self.forward(x, ids, w)
                self.dispatch.paused = False
                for _ in range(repeats):
                    self.forward(x, ids, w)
                if torch.cuda.is_available():
                    dev = f"cuda:{self.cuda_device}"
                    x_d, ids_d, w_d = x.to(dev), ids.to(dev), w.to(dev)
                    self.m_cpu = 0                  # force all-GPU
                    self.dispatch.paused = True
                    self.forward(x_d, ids_d, w_d)
                    self.dispatch.paused = False
                    for _ in range(repeats):
                        self.forward(x_d, ids_d, w_d)
            self.dispatch.harvest(wait=True)
        finally:
            self.adaptive, self.m_cpu = prev_adaptive, prev_m_cpu
            self.dispatch.paused = False
            _CPU_PREFILL_FRACTION = prev_frac
        return self.dispatch.snapshot()

    # -----------------------------------------------------------
    # GPU bucket — grouped INT8 over pinned weights, one launch / projection
    # -----------------------------------------------------------

    @staticmethod
    def _build_layout(
        part_experts: np.ndarray,   # global expert ids in this partition (sorted)
        kernel_ids: np.ndarray,     # ids the kernel uses to index its B tensor
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
        alignment. ``kernel_ids`` decouples routing ids from weight-tensor
        ids (cache slots, or half-local ids under NUMA-TP). Returns the
        (M_grouped, idx_to_orig, slot_to_orig, offsets, experts, list_size)
        tuple consumed by _gpu_grouped_forward.
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
        experts_np = np.concatenate((kernel_ids, [-1])).astype(np.int32)

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

    def _stage_src_views(self):
        """Per-expert pinned-host source views for staging copies, resolved
        across the NUMA-TP slab halves once and cached (the per-call loop then
        only issues ``copy_`` on prebuilt views)."""
        views = getattr(self, "_stage_srcs", None)
        if views is None:
            s = self.slab
            gs = s.g_split

            def per_expert(name):
                a = getattr(s, name)
                b = getattr(s, name + "_b")
                if b is None:
                    return [a[i] for i in range(s.num_experts)]
                return [a[i] if i < gs else b[i - gs]
                        for i in range(s.num_experts)]

            views = (per_expert("gate_int8"), per_expert("up_int8"),
                     per_expert("down_int8"))
            self._stage_srcs = views
        return views

    def _stage_copy(self, part: np.ndarray, ring: "_StageRing"):
        """Issue async copy-engine H2D copies of partition ``part``'s expert
        weights into the next ring slot, on the ring's side stream (overlaps
        the cached-partition GEMMs and the CPU bucket). Returns the context
        consumed by ``_gpu_grouped_forward(kind='staged')``."""
        dev = f"cuda:{self.cuda_device}"
        gate_src, up_src, down_src = self._stage_src_views()
        slot_i = ring.next_slot()
        slot = ring.slots[slot_i]
        ids_t = torch.from_numpy(part.astype(np.int64)).to(dev)
        with torch.cuda.stream(ring.stream):
            # Don't clobber a slot the compute stream may still be reading.
            torch.cuda.current_stream().wait_event(ring.free_ev[slot_i])
            for k, g in enumerate(part.tolist()):
                slot["gate"][k].copy_(gate_src[g], non_blocking=True)
                slot["up"][k].copy_(up_src[g], non_blocking=True)
                slot["down"][k].copy_(down_src[g], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(ring.stream)
        return ring, slot_i, ready, ids_t

    def _gpu_grouped_forward(
        self,
        x_gpu: torch.Tensor,        # [T, H] bf16 device
        layout,                     # contiguous layout built in forward()
        route_w_gpu: torch.Tensor,  # [T, top_k] fp32 device
        kind: str = "slab_a",       # 'cached' | 'staged' | 'slab_a' | 'slab_b'
        staged=None,                # (_StageRing, slot_i, ready_ev, ids_t)
    ):
        """Run one GPU-bucket partition's grouped GEMMs.

        ``layout`` is (M_grouped, idx_to_orig, slot_to_orig, offsets, experts,
        list_size): the AsymGEMM contiguous layout, built vectorized on the
        GPU in forward(), with the layout's expert ids already remapped to
        this partition's weight tensor ('cached' -> cache slots with the
        fused gate+up slab from HBM; 'slab_a'/'slab_b' -> the pinned-host
        halves read over PCIe). Returns (orig_rows_for_valid, y_weighted_fp32)
        where orig_rows_for_valid is a [n_valid] long tensor of source rows in
        the original [T, H] activations, and y_weighted_fp32 is [n_valid, H]
        fp32 ready for ``out_fp32.index_add_(0, orig_rows, y_w)``.
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

        if kind == "cached":
            # Fused gate+up: one grouped call over the [Nc, 2I, H] cache.
            d_gu = torch.empty(M_grouped, 2 * I, device=dev, dtype=torch.float32)
            asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
                (a_int8, sfa_h), (self.cache_gateup_int8, self.cache_gateup_sfb),
                d_gu, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
            )
            act = torch.nn.functional.silu(d_gu[:, :I]) * d_gu[:, I:]
            down_b, down_sfb = self.cache_down_int8, self.cache_down_sfb
        else:
            gs = slab.g_split
            if kind == "staged":
                ring, slot_i, ready_ev, ids_t = staged
                P = int(ids_t.numel())
                slot = ring.slots[slot_i]
                # Gather this partition's SFB rows into the slot's fixed-size
                # SFB buffers (compute stream: stream order both protects the
                # slot from the layer still reading it and sequences these
                # before the GEMMs). Independent of the weight copies, which
                # only the GEMMs must wait for.
                slot["gate_sfb"][:P].copy_(slab.gate_sfb.index_select(0, ids_t))
                slot["up_sfb"][:P].copy_(slab.up_sfb.index_select(0, ids_t))
                slot["down_sfb"][:P].copy_(slab.down_sfb.index_select(0, ids_t))
                torch.cuda.current_stream().wait_event(ready_ev)
                # Full-shape views — see _StageRing on why never sliced by P.
                gate_b, up_b, dn_b = slot["gate"], slot["up"], slot["down"]
                gate_sfb = slot["gate_sfb"]
                up_sfb = slot["up_sfb"]
                down_sfb = slot["down_sfb"]
            elif kind == "slab_b":
                gate_b, up_b, dn_b = (slab.gate_int8_b, slab.up_int8_b,
                                      slab.down_int8_b)
                gate_sfb = slab.gate_sfb[gs:]
                up_sfb = slab.up_sfb[gs:]
                down_sfb = slab.down_sfb[gs:]
            else:
                gate_b, up_b, dn_b = (slab.gate_int8, slab.up_int8,
                                      slab.down_int8)
                # Under NUMA-TP half A holds only [0, gs); the SFBs stay full.
                gate_sfb = slab.gate_sfb[:gs]
                up_sfb = slab.up_sfb[:gs]
                down_sfb = slab.down_sfb[:gs]

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
            down_b, down_sfb = dn_b, down_sfb

        # Down: re-quantize act (BF16 round-trip to match the CPU path).
        act_bf16 = act.to(torch.bfloat16)
        a2_int8, sA2 = quantize_per_token_int8_gpu(act_bf16)
        sfa_i = sA2.unsqueeze(1).expand(M_grouped, kb_i).contiguous()

        d_down = torch.empty(M_grouped, H, device=dev, dtype=torch.float32)
        asym_gemm.m_grouped_int8_asym_gemm_nt_contiguous(
            (a2_int8, sfa_i), (down_b, down_sfb),
            d_down, offsets, experts, list_size, recipe=(1, 1, GRAN_K),
        )

        if kind == "staged":
            # The down GEMM above is the last reader of this ring slot; let
            # the side stream reuse it once the compute stream gets here.
            staged[0].free_ev[staged[1]].record()

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

        if self.cache_n == G and torch.cuda.is_available():
            # Every expert is VRAM-cached: take the sync-free all-device path.
            eids = expert_ids
            if eids.dtype != torch.int64:
                eids = eids.to(torch.int64)
            return self._cached_gpu_forward_any(x_bf16, eids, route_w)

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

        if _record_stats_path():
            if getattr(self, "_stat_counts", None) is None:
                self._stat_counts = torch.zeros_like(counts)
            self._stat_counts += counts
            # Periodic dump via the first layer (atexit never runs when the
            # scheduler dies to SIGTERM).
            if self is _ALL_LAYERS[0]:
                global _stats_step
                _stats_step += 1
                if _stats_step % 20 == 0:
                    _dump_expert_stats()

        # Live routing-count signal, eager branch. Gated on the feature being
        # enabled so that with ASYMGEMM_CACHE_TRACK_ROUTING unset this path
        # costs *nothing* — not even the one small device add. counts is
        # already computed either way.
        #
        # Under sglang this is the EAGER branch only: asym_gemm_unified.py
        # routes anything captured to capturable_decode_forward, which counts
        # in _cached_gpu_decode instead. The two sites are mutually exclusive,
        # so nothing is double-counted. In a graph-enabled server that makes
        # this line effectively the *prefill* counter -- see the caveat below.
        #
        # CAVEAT: prefill and decode both land in the same _live_counts, and
        # prefill dominates it. At 512-in/256-out a request contributes
        # 512*top_k routing events here against 256*top_k from decode, so
        # residency would be chosen ~2:1 by prefill routing. That is defensible
        # if the goal is total token coverage (cached experts serve prefill
        # too), but it is NOT what "pick the hot experts for decode" means, and
        # decode is the throughput-limited phase. If decode-only residency is
        # wanted, drop this accumulation (or weight it down) and let
        # _cached_gpu_decode be the sole signal.
        if self._refresh_enabled and self._live_counts is not None:
            self._live_counts += counts

        counts_np = counts.cpu().numpy()                      # one small D2H
        active = np.nonzero(counts_np)[0]
        cpu_frac = _cpu_prefill_fraction()
        # Only streamed (non-VRAM-cached) experts are candidates for the CPU
        # prefill split: the CPU's job there is to relieve the PCIe weight
        # stream, and cached experts read HBM. With a full cache the CPU
        # share drops to zero automatically.
        if self.cache_n > 0:
            streamed = active[~self._cached_mask_np[active]]
            cached_active = active[self._cached_mask_np[active]]
        else:
            streamed = active
            cached_active = active[:0]
        self.dispatch.harvest()          # fold in completed GPU timings
        if self.adaptive:
            # Cost-model makespan partition of the streamed experts (see
            # dispatch_model.py / adaptive_dispatch.md): both buckets run
            # concurrently, so the solver balances their predicted finish
            # times instead of applying a fixed threshold or row fraction.
            # Cached experts always stay on the GPU.
            gpu_ok = torch.cuda.is_available() and x_bf16.is_cuda
            if not gpu_ok:
                # GPU unusable (CPU-resident input): everything — cached
                # experts included — runs on the CPU bucket, which reads
                # the same pinned weight bytes.
                cpu_experts = active
                gpu_experts = active[:0]
            else:
                m_list = counts_np[streamed].tolist()
                cpu_sel, gpu_sel = self.dispatch.partition(
                    m_list, gpu_available=True)
                cpu_experts = np.sort(
                    streamed[np.asarray(cpu_sel, dtype=np.int64)])
                gpu_experts = np.sort(np.concatenate((
                    streamed[np.asarray(gpu_sel, dtype=np.int64)],
                    cached_active,
                )))
        elif (
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
                (by_count[n_cpu:], cached_active)
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
        t_gather = 0.0
        if len(cpu_experts):
            t0 = time.perf_counter()
            cpu_e_t = torch.from_numpy(cpu_experts).to(in_device)
            is_cpu_item = torch.isin(sorted_e, cpu_e_t)
            rows_cpu = rows_sorted[is_cpu_item]
            slots_cpu = slots_sorted[is_cpu_item]
            x_cat_t = (
                x_bf16.index_select(0, rows_cpu).to("cpu").contiguous()
            )
            x_cat = torch_bf16_to_np_bits(x_cat_t)
            t_gather = time.perf_counter() - t0

        # ---- GPU bucket (grouped INT8): enqueue first, asynchronously; the
        # AMX bucket below then runs on the host while the GPU works through
        # these kernels. When a VRAM expert cache exists, the bucket splits
        # into a cached partition (kernels read HBM) and a streamed partition
        # (weights copy-engine-staged into an HBM ring, or read from pinned
        # host over PCIe when staging is off/OOM).
        gpu_parts = []
        if len(gpu_experts):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU bucket non-empty but no CUDA device available.")

            x_gpu_bf16 = x_bf16.to(in_device, dtype=torch.bfloat16).contiguous()
            route_w_gpu = route_w.to(in_device, dtype=torch.float32)
            # Partition: cached experts (HBM, fused gate+up), then the
            # streamed remainder by slab half (each half is a separate pinned
            # tensor in NUMA-TP mode, with half-local kernel expert ids).
            partitions = []
            rest = gpu_experts
            if self.cache_n > 0:
                cm = self._cached_mask_np[gpu_experts]
                partitions.append((gpu_experts[cm], "cached"))
                rest = gpu_experts[~cm]
            gs = self.slab.g_split
            staged_ctx = None
            ring = (
                _get_stage_ring(self.slab, self.cache_n,
                                f"cuda:{self.cuda_device}")
                if len(rest) else None
            )
            if ring is not None:
                # Copy-engine staging: issue the async H2D copies NOW (side
                # stream) so they overlap the cached partition's GEMMs and
                # the CPU bucket; the staged GEMMs event-wait on them.
                partitions.append((rest, "staged"))
                staged_ctx = self._stage_copy(rest, ring)
            elif self.slab.gate_int8_b is not None:
                partitions.append((rest[rest < gs], "slab_a"))
                partitions.append((rest[rest >= gs], "slab_b"))
            else:
                partitions.append((rest, "slab_a"))
            # Time the streamed partitions with a CUDA event pair for the
            # dispatch cost model. The bracket opens just before the first
            # streamed (non-cached) partition's work is enqueued, so the
            # cached partition's HBM GEMMs — a different cost regime the
            # model must not learn from — stay outside it. Stream order
            # makes elapsed(start→end) cover exactly the streamed work,
            # including any event-waits on the staging ring's copies (that
            # wait IS the PCIe transfer cost the model should see).
            ev_start = ev_end = None
            for part, kind in partitions:
                if not len(part):
                    continue
                if kind != "cached" and ev_start is None:
                    ev_start = torch.cuda.Event(enable_timing=True)
                    ev_start.record()
                if kind == "cached":
                    kernel_ids = self._slot_np[part]
                elif kind == "staged":
                    kernel_ids = np.arange(len(part), dtype=np.int64)
                elif kind == "slab_b":
                    kernel_ids = part - gs
                else:
                    kernel_ids = part
                layout = self._build_layout(
                    part, kernel_ids, counts_np, seg_start_np, sorted_e,
                    rows_sorted, slots_sorted, G, in_device,
                )
                rows_p, y_p = self._gpu_grouped_forward(
                    x_gpu_bf16, layout, route_w_gpu, kind=kind,
                    staged=staged_ctx if kind == "staged" else None,
                )
                if rows_p is not None:
                    gpu_parts.append((rows_p, y_p))
            if ev_start is not None:
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()
                self._gpu_bucket_calls += 1
                if self._gpu_bucket_calls > 1:   # first call is JIT warm-up
                    streamed_gpu = np.concatenate(
                        [p for p, kind in partitions
                         if kind != "cached" and len(p)])
                    total_padded = int(sum(
                        self.dispatch.padded(int(c))
                        for c in counts_np[streamed_gpu]))
                    self.dispatch.record_gpu_events(
                        ev_start, ev_end, len(streamed_gpu), total_padded)

        # ---- CPU bucket, stage 2: the blocking AMX call. The GIL and the
        # CUDA stream are both free here, so this host work overlaps the GPU
        # bucket's grouped GEMMs enqueued above.
        if x_cat is not None:
            t1 = time.perf_counter()
            slab = self.slab
            m_offsets = np.zeros(len(cpu_experts) + 1, dtype=np.int64)
            np.cumsum(counts_np[cpu_experts], out=m_offsets[1:])
            out_cat = np.empty((int(m_offsets[-1]), H), dtype=np.float32)
            if slab.gate_int8_b is not None:
                half_b = (
                    slab.gate_int8_b.numpy(), slab.gate_scales_b.numpy(),
                    slab.up_int8_b.numpy(), slab.up_scales_b.numpy(),
                    slab.down_int8_b.numpy(), slab.down_scales_b.numpy(),
                )
            else:
                half_b = (
                    np.empty(0, np.int8), np.empty(0, np.float32),
                    np.empty(0, np.int8), np.empty(0, np.float32),
                    np.empty(0, np.int8), np.empty(0, np.float32),
                )
            _C.moe_expert_forward_batch(
                self.rt, x_cat, m_offsets, cpu_experts.astype(np.int64),
                slab.gate_int8.numpy(), slab.gate_scales.numpy(),
                slab.up_int8.numpy(),   slab.up_scales.numpy(),
                slab.down_int8.numpy(), slab.down_scales.numpy(),
                *half_b,
                out_cat, slab.inter, slab.hidden,
            )
            self.dispatch.record_cpu(
                len(cpu_experts), int(m_offsets[-1]),
                t_gather + (time.perf_counter() - t1))
            y_cpu = torch.from_numpy(out_cat).to(in_device, non_blocking=True)
            w = route_w[rows_cpu, slots_cpu].to(torch.float32)
            out_fp32.index_add_(0, rows_cpu, y_cpu * w[:, None])

        for rows_p, y_p in gpu_parts:
            out_fp32.index_add_(0, rows_p, y_p)

        return out_fp32.to(torch.bfloat16)
