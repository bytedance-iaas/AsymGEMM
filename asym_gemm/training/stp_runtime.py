"""STPRuntime — single-process dual-GPU runtime for GB200 sTP (gb200_tp.md Stage I1).

One process owns BOTH GPUs of a same-superchip pair: per-device streams, P2P exchange
primitives, and JIT prewarm. The base-weight streaming GEMMs use ZERO copy engines
(in-kernel TMA on the compute stream); copy engines serve only D2H act offload, H2D
restage, and the NVLink P2P exchanges — so P2P gets its own stream per device.

STREAM DISCIPLINE (binding for every primitive here): at entry the consuming stream
waits an event recorded on each operand's PRODUCING stream (ambient current_stream
unless documented otherwise); at exit the device's ambient stream waits the result
event, so ordinary consumers (cat, residual add, autograd) are ordered after the
exchange without knowing our streams. Violations are silent data races.

CUDA facts this file relies on (verified against vendored ATen Copy.cu):
  - a cross-device copy enqueues on the SOURCE device's CURRENT stream;
  - a cpu->cuda H2D copy enqueues on the DEST device's CURRENT stream;
  - peer access is enabled lazily by the first cross-device COPY (not by allocation);
  - kernel launches ride the CURRENT DEVICE's current stream, so every per-device
    launch must run under torch.cuda.device(dev).
"""
from __future__ import annotations

import os
from typing import Sequence

import torch

__all__ = ["STPRuntime", "stp_enabled", "get_runtime"]

_RUNTIME: "STPRuntime | None" = None


def stp_enabled() -> bool:
    return os.environ.get("ASYM_STP") == "1"


def get_runtime() -> "STPRuntime":
    global _RUNTIME
    if _RUNTIME is None:
        if not stp_enabled():
            raise RuntimeError("ASYM_STP != 1; STPRuntime is disabled")
        _RUNTIME = STPRuntime()
    return _RUNTIME


def _record(stream: torch.cuda.Stream) -> torch.cuda.Event:
    event = torch.cuda.Event()
    event.record(stream)
    return event


class STPRuntime:
    """Singleton pair-runtime. Device ids are PROCESS-relative (CUDA_VISIBLE_DEVICES
    already narrowed the pool to the pair, so this is (0, 1) in-process)."""

    def __init__(self, dev_ids: Sequence[int] = (0, 1)) -> None:
        tp_size = int(os.environ.get("ASYM_STP_TP_SIZE", "2"))
        if tp_size != 2:
            raise RuntimeError(f"ASYM_STP_TP_SIZE must be 2, got {tp_size}")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"ASYM_STP=1 needs 2 visible CUDA devices, got {torch.cuda.device_count()}"
            )
        self.dev_ids = tuple(int(i) for i in dev_ids)
        self.d = [torch.device("cuda", i) for i in self.dev_ids]
        self.primary = self.d[0]
        self.weight_mode = os.environ.get("ASYM_STP_WEIGHT_MODE", "stream")
        self.shared_arena = os.environ.get("ASYM_STP_SHARED_ARENA", "1") == "1"
        self.coord = os.environ.get("ASYM_STP_COORD", "1") == "1"

        for a, b in ((0, 1), (1, 0)):
            if not torch.cuda.can_device_access_peer(self.dev_ids[a], self.dev_ids[b]):
                raise RuntimeError(
                    f"peer access unavailable between cuda:{self.dev_ids[a]} and cuda:{self.dev_ids[b]}"
                )
        # ENABLE peer access NOW via a COPY (an alloc does NOT enable it — ATen Copy.cu).
        torch.zeros(1, device=self.d[0]).to(self.d[1])
        torch.zeros(1, device=self.d[1]).to(self.d[0])

        self.compute = [torch.cuda.Stream(device=d) for d in self.d]  # weight-TMA GEMMs
        self.d2h = [torch.cuda.Stream(device=d) for d in self.d]      # act offload (C2C)
        self.h2d = [torch.cuda.Stream(device=d) for d in self.d]      # restage/gather (C2C)
        self.p2p = [torch.cuda.Stream(device=d) for d in self.d]      # NVLink exchanges

        self.stp_dev1_modules_wrapped = 0  # setup counter consumed by the I4 wrap audit
        self._jit_prewarm()

    # ---- JIT prewarm (I1 blocker): one tiny asym GEMM per device so the first real
    # launch never races and the runtime-API (FIX A) path materializes per-device state.
    def _jit_prewarm(self) -> None:
        from asym_gemm.training.frozen_linear import _asym_bf16_nt

        for dev in self.d:
            with torch.cuda.device(dev):
                a = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
                b_cpu = torch.randn(64, 64, dtype=torch.bfloat16).pin_memory()
                out = _asym_bf16_nt(a, b_cpu)
                torch.cuda.synchronize(dev)
                if out.shape != (8, 64):
                    raise RuntimeError(f"stp prewarm GEMM wrong shape on {dev}: {out.shape}")

    # ---- raw exchange primitives (NOT autograd-safe; the boundary Functions land in I4) ----

    def allreduce2(self, y0: torch.Tensor, y1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """In-place sum-exchange: y0 += copy(y1), y1 += copy(y0), full duplex on the two
        p2p streams. The local adds wait BOTH copy-done events (each y_i is READ by its
        outgoing copy and WRITTEN by its add — write-after-read hazard)."""
        d0, d1 = y0.device, y1.device
        e0 = _record(torch.cuda.current_stream(d0))  # producer of y0
        e1 = _record(torch.cuda.current_stream(d1))  # producer of y1

        self.p2p[1].wait_event(e1)                   # src=dev1 will read y1
        with torch.cuda.stream(self.p2p[1]):
            t0 = torch.empty_like(y0)
            t0.copy_(y1, non_blocking=True)          # dev1->dev0, rides p2p[1] (source side)
            y1.record_stream(self.p2p[1])
        self.p2p[0].wait_event(e0)                   # src=dev0 will read y0
        with torch.cuda.stream(self.p2p[0]):
            t1 = torch.empty_like(y1)
            t1.copy_(y0, non_blocking=True)          # dev0->dev1, rides p2p[0]
            y0.record_stream(self.p2p[0])

        c0 = _record(self.p2p[1])                    # t0 ready on dev0
        c1 = _record(self.p2p[0])                    # t1 ready on dev1
        for ev in (c0, c1):
            self.compute[0].wait_event(ev)           # BOTH adds wait BOTH copy-dones
            self.compute[1].wait_event(ev)
        with torch.cuda.device(d0), torch.cuda.stream(self.compute[0]):
            y0.add_(t0)
            t0.record_stream(self.compute[0])
        with torch.cuda.device(d1), torch.cuda.stream(self.compute[1]):
            y1.add_(t1)
            t1.record_stream(self.compute[1])

        torch.cuda.current_stream(d0).wait_event(_record(self.compute[0]))
        torch.cuda.current_stream(d1).wait_event(_record(self.compute[1]))
        return y0, y1

    def _p2p_copy(
        self,
        src: torch.Tensor,
        dst_dev: torch.device,
        src_slot: int,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """One P2P copy src -> dst_dev on the SOURCE device's p2p stream.

        producer_event: event recorded on the stream that PRODUCED src (e.g. compute[i]
        inside its with-block). Defaults to src device's ambient stream. Passing the real
        producer avoids routing entry ordering through ambient, which would serialize
        otherwise-independent transfers (probe lesson: composed bcast01+to0 halved duplex BW).
        """
        e_src = producer_event or _record(torch.cuda.current_stream(src.device))
        stream = self.p2p[src_slot]
        stream.wait_event(e_src)
        with torch.cuda.stream(stream):
            out = torch.empty(src.shape, dtype=src.dtype, device=dst_dev)
            out.copy_(src, non_blocking=True)
            src.record_stream(stream)
        torch.cuda.current_stream(dst_dev).wait_event(_record(stream))
        return out

    def to0(self, y1: torch.Tensor, producer_event: torch.cuda.Event | None = None) -> torch.Tensor:
        """dev1 -> dev0 P2P copy on p2p[1] (source side)."""
        return self._p2p_copy(y1, self.d[0], 1, producer_event)

    def bcast01(self, x0: torch.Tensor, producer_event: torch.cuda.Event | None = None) -> torch.Tensor:
        """dev0 -> dev1 P2P copy on p2p[0] (source side)."""
        return self._p2p_copy(x0, self.d[1], 0, producer_event)

    def to0_sum(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        producer_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """y0 += to0(y1) with the add on dev0's ambient stream (Phase-A return path)."""
        t0 = self.to0(y1, producer_event)  # exit contract: dev0 ambient already waits the copy
        y0.add_(t0)                        # ambient add; ordered after both producers
        return y0

    def allreduce2_out(
        self, y0: torch.Tensor, y1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OUT-OF-PLACE sum-exchange: returns fresh (y0+y1', y1+y0') without mutating the
        inputs — required inside autograd Function backwards (IN-PLACE RULE: never mutate
        incoming grad_outputs) and forwards (version counter on Fn inputs)."""
        d0, d1 = y0.device, y1.device
        e0 = _record(torch.cuda.current_stream(d0))
        e1 = _record(torch.cuda.current_stream(d1))
        self.p2p[1].wait_event(e1)
        with torch.cuda.stream(self.p2p[1]):
            t0 = torch.empty_like(y0)
            t0.copy_(y1, non_blocking=True)
            y1.record_stream(self.p2p[1])
        self.p2p[0].wait_event(e0)
        with torch.cuda.stream(self.p2p[0]):
            t1 = torch.empty_like(y1)
            t1.copy_(y0, non_blocking=True)
            y0.record_stream(self.p2p[0])
        c0 = _record(self.p2p[1])
        c1 = _record(self.p2p[0])
        # each add reads its LOCAL y_i (producer-ordered via e_i) + its incoming temp
        self.compute[0].wait_event(e0)
        self.compute[0].wait_event(c0)
        self.compute[1].wait_event(e1)
        self.compute[1].wait_event(c1)
        with torch.cuda.device(d0), torch.cuda.stream(self.compute[0]):
            out0 = y0 + t0
            t0.record_stream(self.compute[0])
            y0.record_stream(self.compute[0])
        with torch.cuda.device(d1), torch.cuda.stream(self.compute[1]):
            out1 = y1 + t1
            t1.record_stream(self.compute[1])
            y1.record_stream(self.compute[1])
        torch.cuda.current_stream(d0).wait_event(_record(self.compute[0]))
        torch.cuda.current_stream(d1).wait_event(_record(self.compute[1]))
        return out0, out1

    def bcast01_from_host(self, host: torch.Tensor) -> torch.Tensor:
        """Pinned host -> dev1 H2D on h2d[1] (I5 residual restage: both lanes pull the ONE
        pinned copy concurrently; dev0's pull uses its own offload manager)."""
        if not host.is_pinned():
            raise RuntimeError("bcast01_from_host requires a pinned host tensor")
        with torch.cuda.device(self.d[1]), torch.cuda.stream(self.h2d[1]):
            out = torch.empty(host.shape, dtype=host.dtype, device=self.d[1])
            out.copy_(host, non_blocking=True)
        torch.cuda.current_stream(self.d[1]).wait_event(_record(self.h2d[1]))
        return out
