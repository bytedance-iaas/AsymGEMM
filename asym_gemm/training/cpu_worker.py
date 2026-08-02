"""Persistent CPU worker thread for op-level CPU/GPU overlap (cpu_compute.md Stage 2).

Design (researched + settled, see cpu_compute.md §2.4):
- ATen CPU ops and our pybind kernels (``py::gil_scoped_release``) release the GIL, so one
  persistent ``threading.Thread`` gives true parallelism with the main thread's CUDA
  launches.
- Grad mode / current device are THREAD-LOCAL: the worker sets both explicitly.
- The worker performs HOST waits (``event.synchronize()``) on producing D2H events before
  touching pinned buffers; consumers wait on a ``threading.Event`` (host) or on a CUDA
  event the worker records after enqueuing H2D on a side stream.
- Buffer alloc/release stays on the MAIN thread (the pinned pool is not thread-safe).
- Lazy singleton: created on first submit, never at import (LF dataloader forks).
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
from typing import Callable, Optional

import torch

__all__ = ["CpuTask", "enabled", "get_worker", "shutdown", "submit", "wait", "job_ms_snapshot"]

# K-11: per-tag wall-ms of worker jobs (worker-side time is invisible to NVTX/nsys).
_JOB_MS: dict = {}
_JOB_N: dict = {}
_JOB_LOCK = None  # created lazily with the first worker (threading.Lock)


def _record_job_ms(tag: str, ms: float) -> None:
    global _JOB_LOCK
    if _JOB_LOCK is None:
        import threading as _t

        _JOB_LOCK = _t.Lock()
    with _JOB_LOCK:
        _JOB_MS[tag] = _JOB_MS.get(tag, 0.0) + ms
        _JOB_N[tag] = _JOB_N.get(tag, 0) + 1


def job_ms_snapshot() -> dict:
    if _JOB_LOCK is None:
        return {}
    with _JOB_LOCK:
        return {t: {"ms": round(v, 3), "calls": _JOB_N.get(t, 0)} for t, v in _JOB_MS.items()}


def _timed(fn, tag):
    if tag is None:
        return fn
    import time as _time

    def _wrapped():
        t0 = _time.perf_counter()
        try:
            return fn()
        finally:
            _record_job_ms(tag, (_time.perf_counter() - t0) * 1000.0)

    return _wrapped

_STOP = object()


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() not in {"", "0", "false", "no", "off"}


def enabled() -> bool:
    if _env_on("ASYMM_CPU_WORKER"):
        return True
    from . import placement_policy

    return placement_policy.enabled() and placement_policy.cpu_worker()


class CpuTask:
    __slots__ = ("fn", "done", "exc", "result")

    def __init__(self, fn: Callable[[], object]):
        self.fn = fn
        self.done = threading.Event()
        self.exc: Optional[BaseException] = None
        self.result: object = None


class CpuWorker:
    def __init__(self, device_index: int):
        self._device_index = device_index
        self._q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="asym-cpu-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_index)  # thread-local
        with torch.no_grad():  # thread-local
            while True:
                item = self._q.get()
                if item is _STOP:
                    return
                task: CpuTask = item  # type: ignore[assignment]
                try:
                    task.result = task.fn()
                except BaseException as exc:  # propagate to the waiter, keep the worker alive
                    task.exc = exc
                finally:
                    task.done.set()

    def submit(self, fn: Callable[[], object]) -> CpuTask:
        task = CpuTask(fn)
        self._q.put(task)
        return task

    def stop(self, timeout: float = 5.0) -> None:
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)


_WORKER: Optional[CpuWorker] = None
_BG_WORKER: Optional[CpuWorker] = None
_LOCK = threading.RLock()  # RLock: get_bg_worker() calls get_worker() under the lock
_OWNER_PID: Optional[int] = None


def get_worker() -> CpuWorker:
    """Lazy per-process singleton (re-created after fork)."""
    global _WORKER, _OWNER_PID
    pid = os.getpid()
    with _LOCK:
        if _WORKER is None or _OWNER_PID != pid:
            device = torch.cuda.current_device() if torch.cuda.is_available() else 0
            _WORKER = CpuWorker(device)
            _OWNER_PID = pid
            atexit.register(shutdown)
        return _WORKER


def bg_enabled() -> bool:
    """K-5 (cpu_compute.md): second worker for fire-and-forget deposits so they never
    priority-invert layer-critical jobs (silu/act) on the primary FIFO."""
    if _env_on("ASYM_CPU_WORKER_BG"):
        return True
    from . import placement_policy

    return placement_policy.enabled() and placement_policy.cpu_worker_bg()


def get_bg_worker() -> CpuWorker:
    global _BG_WORKER
    pid = os.getpid()
    with _LOCK:
        if _BG_WORKER is None or _OWNER_PID != pid:
            get_worker()  # ensures _OWNER_PID current
            device = torch.cuda.current_device() if torch.cuda.is_available() else 0
            _BG_WORKER = CpuWorker(device)
        return _BG_WORKER


def submit(fn: Callable[[], object], tag: str | None = None) -> CpuTask:
    return get_worker().submit(_timed(fn, tag))


def submit_deposit(fn: Callable[[], object], tag: str | None = None) -> CpuTask:
    """Deposits go to the background worker when enabled (else the primary)."""
    if bg_enabled():
        return get_bg_worker().submit(_timed(fn, tag))
    return get_worker().submit(_timed(fn, tag))


def wait(task: Optional[CpuTask]):
    if task is None:
        return None
    task.done.wait()
    if task.exc is not None:
        raise task.exc
    return task.result


def shutdown() -> None:
    global _WORKER
    with _LOCK:
        if _WORKER is not None and _OWNER_PID == os.getpid():
            _WORKER.stop()
            _WORKER = None
