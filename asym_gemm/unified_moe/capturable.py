"""CUDA-graph-capturable decode path for the unified MoE layer.

The CPU MoE bucket is normally imperative host code (blocking ``.to("cpu")`` +
synchronous AMX call), which a CUDA graph can neither replay nor capture. This
module expresses the same work as a **stream node chain** instead:

    D2H copies (x / expert_ids / route_w -> pinned buffers)
    -> cudaLaunchHostFunc(moe_decode_host)   [the CPU AMX MoE, a host node]
    -> H2D copy (pinned out -> device out)

All three are stream-ordered operations, so when sglang captures the decode
graph this whole chain records into it; on replay the CUDA runtime re-invokes
the host node (CPU recomputes) — no ``--disable-cuda-graph`` needed. The
mechanism is validated standalone in tests/phase0_cudagraph_decode.py.

Scope: any batch size T (all-CPU). The host node groups the (token, slot)
routing into a per-expert concatenated layout and runs the same per-expert
batched CPU kernel the eager path uses, so larger batches engage more pool
threads. It does NOT offload large experts to the GPU, so capture should be
limited to a moderate batch (``--cuda-graph-max-bs``); larger decode batches,
where the eager path's GPU offload wins, are left to eager. The caller supplies
``expert_ids`` already clamped >= 0 and ``route_w`` already masked (invalid
slots = 0); routed_scaling is applied by the caller after the H2D copy.
"""
from __future__ import annotations

import ctypes
import glob
import os

import torch

from .. import _cpu_C as _C

# The host-node path now supports any batch size (all-CPU). The set of batch
# sizes actually pre-allocated is whatever init_capturable_decode is given
# (the CUDA-graph capture batch sizes); see asym_gemm_unified glue.

_cudart = None
_fn_ptr = None


def _cudart_lib():
    global _cudart
    if _cudart is not None:
        return _cudart
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _cudart is None:  # fall back to the libcudart torch already loaded
        for so in glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*")):
            _cudart = ctypes.CDLL(so)
            break
    if _cudart is None:
        raise RuntimeError("could not locate libcudart for cudaLaunchHostFunc")
    _cudart.cudaLaunchHostFunc.restype = ctypes.c_int
    _cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return _cudart


def _launch_host_func(stream_ptr: int, fn_ptr: int, args_ptr: int) -> None:
    """Enqueue the host-node callback on `stream_ptr` (recorded during capture)."""
    err = _cudart_lib().cudaLaunchHostFunc(
        ctypes.c_void_p(stream_ptr), ctypes.c_void_p(fn_ptr), ctypes.c_void_p(args_ptr)
    )
    if err != 0:
        raise RuntimeError(f"cudaLaunchHostFunc returned {err}")


def _host_fn_ptr() -> int:
    global _fn_ptr
    if _fn_ptr is None:
        _fn_ptr = _C.decode_host_fn_ptr()
    return _fn_ptr


class _DecodeBuffers:
    """Fixed-address IO buffers + a bound host-node args struct, per (layer, T).

    Buffers are per-layer (a few hundred KB total at T=1) rather than shared, so
    there is no cross-layer aliasing to reason about: each layer's host node
    reads/writes only its own pinned buffers, and the graph's stream ordering
    sequences the layers. The args struct binds these buffers' raw pointers plus
    this layer's INT8 weight slab; it must outlive the captured graph.
    """

    def __init__(self, layer, T: int):
        slab = layer.slab
        H, I, G, K = slab.hidden, slab.inter, slab.num_experts, layer.top_k
        dev = f"cuda:{layer.cuda_device}"
        self.x_cpu = torch.zeros(T, H, dtype=torch.bfloat16, pin_memory=True)
        self.eid_cpu = torch.zeros(T, K, dtype=torch.int64, pin_memory=True)
        self.rw_cpu = torch.zeros(T, K, dtype=torch.float32, pin_memory=True)
        self.out_cpu = torch.zeros(T, H, dtype=torch.bfloat16, pin_memory=True)
        self.out_gpu = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        self.args = _C.make_decode_args(
            layer.rt,
            self.x_cpu.data_ptr(), self.eid_cpu.data_ptr(),
            self.rw_cpu.data_ptr(), self.out_cpu.data_ptr(),
            slab.gate_int8.data_ptr(), slab.gate_scales.data_ptr(),
            slab.up_int8.data_ptr(), slab.up_scales.data_ptr(),
            slab.down_int8.data_ptr(), slab.down_scales.data_ptr(),
            T, K, G, H, I,
        )
        self.T = T

    def __del__(self):
        try:
            _C.free_decode_args(self.args)
        except Exception:
            pass


def init_capturable_decode(layer, batch_sizes) -> None:
    """Pre-allocate fixed buffers for the supported batch sizes.

    MUST run outside CUDA graph capture (it allocates). Call once at load, after
    the layer's INT8 slab is built.
    """
    # Warm the libcudart handle + host-fn pointer now, so the first call inside
    # CUDA graph capture issues only the cudaLaunchHostFunc stream op (no dlopen
    # / ctypes setup during capture).
    _cudart_lib()
    _host_fn_ptr()
    cap = getattr(layer, "_capturable", None) or {}
    for T in batch_sizes:
        t = int(T)
        if t >= 1 and t not in cap:
            cap[t] = _DecodeBuffers(layer, t)
    layer._capturable = cap

    # With a VRAM expert cache, the capturable path launches the grouped INT8
    # kernel with min(T*K, cache_n)+1 groups. Run each distinct shape once now,
    # eagerly, so the JIT ensure-compile (which synchronizes the stream) never
    # fires inside graph capture. The compile cache is process-global — warm
    # only once, on the first layer.
    if getattr(layer, "cache_n", 0) > 0 and _warm_cached_shapes(layer):
        dev = f"cuda:{layer.cuda_device}"
        H, K = layer.slab.hidden, layer.top_k
        for t in sorted({int(b) for b in batch_sizes}):
            x = torch.zeros(t, H, dtype=torch.bfloat16, device=dev)
            eids = torch.zeros(t, K, dtype=torch.int64, device=dev)
            rw = torch.ones(t, K, dtype=torch.float32, device=dev)
            layer._cached_gpu_decode(x, eids, rw)
        torch.cuda.synchronize()


_warmed_cache_keys = set()


def _warm_cached_shapes(layer) -> bool:
    key = (layer.cache_n, layer.slab.hidden, layer.slab.inter, layer.top_k)
    if key in _warmed_cache_keys:
        return False
    _warmed_cache_keys.add(key)
    return True


def capturable_decode_supported(layer, T: int) -> bool:
    return int(T) in getattr(layer, "_capturable", {})


def capturable_decode_forward(layer, x_bf16, expert_ids, route_w):
    """Issue the capturable decode MoE; return the layer output.

    ``expert_ids`` [T,K] must be clamped >= 0; ``route_w`` [T,K] must already be
    masked (invalid slots = 0) and scaled.

    Without a VRAM expert cache this is the pure [D2H -> host node -> H2D]
    chain over fixed pinned buffers. With a cache (layer.cache_n > 0) the two
    engines run concurrently inside the graph: the CPU chain is forked onto a
    side stream (cached items' route weights zeroed so the host node skips
    them) while the main stream computes the cached experts with the grouped
    INT8 kernel over HBM-resident weights; an event join then sums the two
    partial outputs. All of it is stream-ordered, so capture/replay works
    exactly as before.
    """
    T = x_bf16.shape[0]
    Nc = getattr(layer, "cache_n", 0)
    if Nc >= layer.slab.num_experts and Nc > 0:
        # Every expert is cached: pure-GPU decode, no host node at all.
        return layer._cached_gpu_decode(x_bf16, expert_ids, route_w)

    buf = layer._capturable[int(T)]
    if Nc == 0:
        # copy_ casts dtype in-kernel (no Python temp); sources are decode
        # intermediates with stable addresses inside the capture mempool.
        buf.x_cpu.copy_(x_bf16, non_blocking=True)
        buf.eid_cpu.copy_(expert_ids, non_blocking=True)
        buf.rw_cpu.copy_(route_w, non_blocking=True)
        stream = torch.cuda.current_stream(x_bf16.device).cuda_stream
        _launch_host_func(stream, _host_fn_ptr(), buf.args)
        buf.out_gpu.copy_(buf.out_cpu, non_blocking=True)
        return buf.out_gpu

    # ---- hybrid: CPU chain on a side stream, cached experts on the main ----
    slot = layer.cached_slot[expert_ids]                  # [T, K]
    on_gpu = slot >= 0
    rw_cpu = torch.where(on_gpu, torch.zeros_like(route_w), route_w)

    main = torch.cuda.current_stream(x_bf16.device)
    side = _side_stream(x_bf16.device)
    ev_fork = torch.cuda.Event()
    ev_join = torch.cuda.Event()
    ev_fork.record(main)
    side.wait_event(ev_fork)
    with torch.cuda.stream(side):
        buf.x_cpu.copy_(x_bf16, non_blocking=True)
        buf.eid_cpu.copy_(expert_ids, non_blocking=True)
        buf.rw_cpu.copy_(rw_cpu, non_blocking=True)
        _launch_host_func(side.cuda_stream, _host_fn_ptr(), buf.args)
        buf.out_gpu.copy_(buf.out_cpu, non_blocking=True)
        ev_join.record(side)
    if not torch.cuda.is_current_stream_capturing():
        # Eager: keep the temp alive for the side stream's async copy.
        rw_cpu.record_stream(side)

    # Runs on the main stream, concurrent with the host node above.
    y_gpu = layer._cached_gpu_decode(x_bf16, expert_ids, route_w)

    main.wait_event(ev_join)
    return y_gpu + buf.out_gpu


_side = None


def _side_stream(device):
    global _side
    if _side is None:
        _side = torch.cuda.Stream(device=device)
    return _side
