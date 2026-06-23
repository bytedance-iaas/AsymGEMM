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

Scope (this phase): batch size in ``SUPPORTED_BATCH`` and the all-CPU regime.
At T=1 every routed expert holds exactly one token (<= m_cpu), so the GPU
bucket is always empty and the host node does the whole layer. The caller
supplies ``expert_ids`` already clamped >= 0 and ``route_w`` already masked
(invalid slots = 0) and scaled (routed_scaling folded in), so the host node's
route-weighted reduce produces the final output directly.
"""
from __future__ import annotations

import ctypes
import glob
import os

import torch

from .. import _cpu_C as _C

# Batch sizes for which the capturable host-node path is available. Larger
# batches (where the m_cpu split genuinely activates) need the static-layout +
# routing-weight masking design and are out of scope here; they fall back to
# the BF16 path under capture.
SUPPORTED_BATCH = (1,)

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
    cap = {}
    for T in batch_sizes:
        if T in SUPPORTED_BATCH:
            cap[int(T)] = _DecodeBuffers(layer, int(T))
    layer._capturable = cap


def capturable_decode_supported(layer, T: int) -> bool:
    return int(T) in getattr(layer, "_capturable", {})


def capturable_decode_forward(layer, x_bf16, expert_ids, route_w):
    """Issue the capturable [D2H -> host node -> H2D] chain; return out_gpu.

    ``expert_ids`` [T,K] must be clamped >= 0; ``route_w`` [T,K] must already be
    masked (invalid slots = 0) and scaled. The returned tensor is a fixed,
    per-layer device buffer — safe to wire into the captured graph.
    """
    T = x_bf16.shape[0]
    buf = layer._capturable[int(T)]
    # copy_ casts dtype in-kernel (no Python temp); sources are decode
    # intermediates with stable addresses inside the capture mempool.
    buf.x_cpu.copy_(x_bf16, non_blocking=True)
    buf.eid_cpu.copy_(expert_ids, non_blocking=True)
    buf.rw_cpu.copy_(route_w, non_blocking=True)
    stream = torch.cuda.current_stream(x_bf16.device).cuda_stream
    _launch_host_func(stream, _host_fn_ptr(), buf.args)
    buf.out_gpu.copy_(buf.out_cpu, non_blocking=True)
    return buf.out_gpu
