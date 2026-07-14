"""Phase D4 (agent/impls/fix_throughput.md C3): async variants of the unsloth-GC
boundary-root offload and the recompute-side save_on_cpu hooks.

The stock LF path saves the per-layer root via ``hidden.to("cpu", non_blocking=True)``
— an UNPINNED destination, so the backward ``.to("cuda")`` fetch is a pageable H2D
(~1 GB at pageable bandwidth, host-blocking) once per layer: the measured ~147 ms
periodic backward gaps. ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` packs
pinned but its unpack is a blocking ``.to(device)`` per saved tensor.

Everything here is gated by ``ASYM_SAVED_TENSOR_ASYNC_UNPACK`` (default off) and
mirrors the attention-wrapper pattern: pinned buffers, D2H ready events, H2D restage
on the shared side stream, compute stream waits on an event — the host never blocks.
"""

from __future__ import annotations

import torch

from .attention_activation_offload import _h2d_restage_stream
from .decoder_activation_offload import _async_unpack_enabled


def async_gc_offload_enabled() -> bool:
    return _async_unpack_enabled()


def pack_root_to_pinned_cpu(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
    """D2H the GC boundary root into pinned CPU on the current stream; returns
    (pinned_cpu, ready_event). Caller keeps both until the backward fetch."""
    pinned = torch.empty(
        tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
    )
    with torch.no_grad():
        pinned.copy_(tensor.detach(), non_blocking=True)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(tensor.device))
    return pinned, ready


def fetch_root_async(
    pinned: torch.Tensor,
    ready: torch.cuda.Event | None,
    device: torch.device,
) -> torch.Tensor:
    """H2D the boundary root on the side stream; compute waits on an event."""
    staged = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    compute_stream = torch.cuda.current_stream(device)
    side = _h2d_restage_stream(device)
    if ready is not None:
        side.wait_event(ready)
    side.wait_stream(compute_stream)  # staged alloc ordering
    with torch.no_grad(), torch.cuda.stream(side):
        staged.copy_(pinned, non_blocking=True)
    done = torch.cuda.Event()
    done.record(side)
    compute_stream.wait_event(done)
    staged.record_stream(side)
    staged._asym_restage_keepalive = pinned  # type: ignore[attr-defined]
    return staged


class _AsyncCpuSave:
    __slots__ = ("pinned", "device", "ready")

    def __init__(self, pinned: torch.Tensor, device: torch.device, ready: torch.cuda.Event) -> None:
        self.pinned = pinned
        self.device = device
        self.ready = ready


class async_save_on_cpu(torch.autograd.graph.saved_tensors_hooks):
    """Drop-in for ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` whose
    unpack restages on the side stream instead of a blocking ``.to(device)``."""

    def __init__(self) -> None:
        def pack(tensor: torch.Tensor):
            if not tensor.is_cuda:
                return tensor
            pinned = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
            with torch.no_grad():
                pinned.copy_(tensor.detach(), non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(tensor.device))
            return _AsyncCpuSave(pinned, tensor.device, ready)

        def unpack(packed):
            if not isinstance(packed, _AsyncCpuSave):
                return packed
            return fetch_root_async(packed.pinned, packed.ready, packed.device)

        super().__init__(pack, unpack)
