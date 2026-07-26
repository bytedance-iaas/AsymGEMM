"""Exact-size pinned host memory (capacity fix, 2026-07-25).

Why this exists: torch's CachingHostAllocator rounds EVERY pinned allocation up to
the next power of two (measured on this venv: 37→64, 259→512, 1531→2048 MiB —
+41% on a mixed batch), and `.to("cpu", non_blocking=True)` destinations are
allocated through it, so weight homes, unsloth-GC boundary roots and transients
all pay the tax. On the capacity metric (peak unevictable = pinned Shmem + anon)
this inflated asym's steady host slope to ~3.5 GiB/B vs ~2.6 logical (see
agent/impls/model_capacity.md §8c). SuperOffload pins exact via DeepSpeed's own
cudaHostAlloc and pays no such tax.

Fix: allocate exact-size pageable tensors and page-lock them in place with
``cudaHostRegister`` (same mechanism as shared_fabric's collective seal). RSS is
then exactly the logical bytes; ``is_pinned()`` is True; H2D/D2H stay within
~10% of allocator-pinned bandwidth on GB200 (measured 173-187 vs 191 GB/s).

Gates (all default OFF so existing behavior is bit-identical):
  ASYM_EXACT_PINNED=1          master gate for weight/optimizer homes
  ASYM_EXACT_PINNED_ROOTS=1    unsloth-GC boundary roots ride the RootPool
  ASYM_EXACT_PINNED_MIN_MB     below this, fall back to allocator pinning (default 8)
  ASYM_MEM_ATTRIB_LOG=path     3s attribution logger (smaps + registered bytes)

Validated 2026-07-25 (model_capacity.md §8c): 97.891B/128k anchor 699.3 →
303.5 GiB (−57%); crowns 128k 267.7B✓/279.2✗, 64k 340.4B✓/355.7✗.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import torch

_LOCK = threading.Lock()
_REGISTERED_BYTES = 0
_REGISTERED_COUNT = 0
_REGISTER_SECONDS = 0.0
_FAILED: list[str] = []


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def exact_pinned_enabled() -> bool:
    return _env_flag("ASYM_EXACT_PINNED")


def exact_roots_enabled() -> bool:
    return _env_flag("ASYM_EXACT_PINNED_ROOTS")


def _min_bytes() -> int:
    try:
        mb = float(os.environ.get("ASYM_EXACT_PINNED_MIN_MB", "8") or 8)
    except ValueError:
        mb = 8.0
    return int(mb * (1 << 20))


def register_stats() -> dict[str, Any]:
    return {
        "registered_bytes": _REGISTERED_BYTES,
        "registered_count": _REGISTERED_COUNT,
        "register_seconds": round(_REGISTER_SECONDS, 3),
        "failures": len(_FAILED),
    }


def register_inplace(tensor: torch.Tensor) -> str | None:
    """Page-lock ``tensor``'s storage in place via cudaHostRegister.

    Returns None on success, an error string on failure (caller falls back to
    ``pin_memory()``). The storage is intentionally never unregistered: every
    call site is a process-lifetime home (weights, optimizer state, root pool
    slots), so unregister-before-free ordering never arises.
    """
    global _REGISTERED_BYTES, _REGISTERED_COUNT, _REGISTER_SECONDS
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    if tensor.device.type != "cpu":
        return "not_cpu"
    if tensor.is_pinned():
        return None
    storage = tensor.untyped_storage()
    nbytes = storage.nbytes()
    if nbytes < _min_bytes():
        return "below_min_size"
    try:
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        cudart = torch.cuda.cudart()
        t0 = time.perf_counter()
        rc = cudart.cudaHostRegister(storage.data_ptr(), nbytes, 0)
        dt = time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover - driver hiccup
        return f"register_exception:{exc}"
    if int(rc) != 0:
        with _LOCK:
            _FAILED.append(str(rc))
        return f"register_rc:{rc}"
    with _LOCK:
        _REGISTERED_BYTES += nbytes
        _REGISTERED_COUNT += 1
        _REGISTER_SECONDS += dt
    return None


def alloc_exact_pinned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Exact-size pinned tensor: pageable alloc + in-place register. Falls back
    to allocator pinning on register failure."""
    t = torch.empty(shape, dtype=dtype)
    err = register_inplace(t)
    if err is not None and err != "below_min_size":
        t = torch.empty(shape, dtype=dtype, pin_memory=True)
    elif err == "below_min_size":
        t = t.pin_memory()
    return t


class RootPool:
    """Event-guarded free-list of exact-pinned buffers for unsloth-GC boundary
    roots. All roots of a run share one (shape, dtype), so the pool converges to
    exactly max-live-roots × exact_root_bytes — no pow2 tax, no register churn.
    """

    def __init__(self) -> None:
        self._free: dict[tuple, list[tuple[torch.Tensor, torch.cuda.Event | None]]] = {}
        self._lock = threading.Lock()
        self.allocated_bytes = 0
        self.slots = 0

    def _key(self, shape: torch.Size, dtype: torch.dtype) -> tuple:
        return (tuple(shape), dtype)

    def acquire(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        key = self._key(shape, dtype)
        with self._lock:
            bucket = self._free.get(key)
            if bucket:
                for i, (buf, ev) in enumerate(bucket):
                    if ev is None or ev.query():
                        bucket.pop(i)
                        return buf
        buf = alloc_exact_pinned(tuple(shape), dtype)
        with self._lock:
            self.allocated_bytes += buf.numel() * buf.element_size()
            self.slots += 1
        return buf

    def release(self, buf: torch.Tensor, ready: "torch.cuda.Event | None") -> None:
        key = self._key(buf.shape, buf.dtype)
        with self._lock:
            self._free.setdefault(key, []).append((buf, ready))

    def pack(self, hidden: torch.Tensor) -> torch.Tensor:
        """D2H ``hidden`` into an exact-pinned pool buffer on the current stream
        (drop-in for ``hidden.to("cpu", non_blocking=True)``)."""
        buf = self.acquire(hidden.shape, hidden.dtype)
        with torch.no_grad():
            buf.copy_(hidden.detach(), non_blocking=True)
        return buf


_ROOT_POOL: RootPool | None = None


def root_pool() -> RootPool:
    global _ROOT_POOL
    if _ROOT_POOL is None:
        _ROOT_POOL = RootPool()
    return _ROOT_POOL


# ── attribution logger (diagnostics; ASYM_MEM_ATTRIB_LOG=path) ────────────────

def _smaps_rollup() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].rstrip(":") in ("Rss", "Anonymous", "Shared_Dirty"):
                    out[parts[0].rstrip(":")] = int(parts[1]) / (1 << 20)  # GiB
    except OSError:
        pass
    return out


def _node_unevict_gib() -> float:
    total = 0
    for node in (0, 1):
        try:
            with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
                for line in f:
                    if "AnonPages:" in line or "Shmem:" in line:
                        total += int(line.split()[-2])
        except OSError:
            return -1.0
    return total / (1 << 20)


def _attrib_loop(path: str) -> None:
    while True:
        pool = _ROOT_POOL
        row = {
            "t": round(time.time(), 1),
            "unevict_gib": round(_node_unevict_gib(), 2),
            "registered_gib": round(_REGISTERED_BYTES / (1 << 30), 2),
            "root_pool_gib": round((pool.allocated_bytes if pool else 0) / (1 << 30), 2),
            "root_slots": pool.slots if pool else 0,
        }
        row.update({k.lower(): round(v, 2) for k, v in _smaps_rollup().items()})
        try:
            with open(path, "a") as f:
                f.write(repr(row) + "\n")
        except OSError:
            pass
        time.sleep(3.0)


def maybe_start_attrib_logger() -> None:
    path = os.environ.get("ASYM_MEM_ATTRIB_LOG", "").strip()
    if not path:
        return
    t = threading.Thread(target=_attrib_loop, args=(path,), daemon=True, name="asym-mem-attrib")
    t.start()


maybe_start_attrib_logger()
