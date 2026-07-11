"""Activation spill governor — the actnvme policy engine (Stage 3).

Tracks every CPU activation handle for pressure; spills SEALED handles oldest-first to NVMe
when live CPU activation bytes cross ``ASYM_NVME_ACT_CPU_BUDGET_BYTES`` (FIFO spill / LIFO
consume = exact Belady); fetches them back on the consuming (backward) side. Sync (v1) by
default; async (Stage 5) flips ``ASYM_NVME_SYNC=0``.

Enabled only when the store has role ``activation`` (rule 7: :func:`get_act_spill_governor`
returns ``None`` otherwise). All spill/fetch machinery is inert for handles never passed to
``seal()`` — those stay resident (pressure-only). ``activation_offload`` imports this module
only at its bottom, and this module imports ``activation_offload`` lazily inside the two
methods that need the pool, so there is no import-time cycle.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from .nvme_store import alloc_padded_pinned, get_nvme_store, io_ready


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _meminfo_available() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 64 * 1024 ** 3


def _budget_from_env() -> int:
    """ASYM_NVME_ACT_CPU_BUDGET_BYTES: int | 'auto' (0.85x MemAvailable at init) | 0 (eager spill-all).
    'auto' is only meaningful under numactl membind (HC2) — always set it explicitly for gates."""
    raw = os.environ.get("ASYM_NVME_ACT_CPU_BUDGET_BYTES", "auto").strip().lower()
    if raw in ("", "auto"):
        return int(0.85 * _meminfo_available())
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _record_event():
    if torch.cuda.is_available():
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        return ev
    return None


TRACKED, QUEUED, SUBMITTED, DURABLE, CLAIMED, FETCHED = range(6)
_SENTINEL = torch.empty(0, dtype=torch.uint8)   # data_ptr()==0 => stock release paths no-op


@dataclass
class _Rec:
    handle: Any
    manager: Any
    nbytes: int                # CACHED (handle.nbytes is live over .tensor)
    substrate: str             # "boundary" | "fg" (stats only; NO policy)
    order: int
    state: int = TRACKED
    seal_event: Any = None
    ref: Any = None
    prefetch_buf: Any = None


@dataclass
class _GovStats:
    spilled_bytes: dict = field(default_factory=dict)
    spilled_count: dict = field(default_factory=dict)
    fetched_bytes: dict = field(default_factory=dict)
    fetched_count: dict = field(default_factory=dict)
    live_peak: int = 0
    wasted_writes: int = 0

    def note_live_peak(self, live: int) -> None:
        self.live_peak = max(self.live_peak, live)

    def note_spill(self, substrate: str, nbytes: int) -> None:
        self.spilled_bytes[substrate] = self.spilled_bytes.get(substrate, 0) + nbytes
        self.spilled_count[substrate] = self.spilled_count.get(substrate, 0) + 1

    def note_fetch(self, substrate: str, nbytes: int) -> None:
        self.fetched_bytes[substrate] = self.fetched_bytes.get(substrate, 0) + nbytes
        self.fetched_count[substrate] = self.fetched_count.get(substrate, 0) + 1

    def as_dict(self) -> dict:
        return {
            "spilled_bytes": dict(self.spilled_bytes),
            "spilled_count": dict(self.spilled_count),
            "fetched_bytes": dict(self.fetched_bytes),
            "fetched_count": dict(self.fetched_count),
            "wasted_writes": self.wasted_writes,
        }


class ActSpillGovernor:
    def __init__(self, store):
        self._store = store
        self._lock = threading.RLock()
        self._by_id: dict = {}                 # id(handle) -> _Rec (handle alive until release)
        self._queue: "deque" = deque()         # QUEUED recs in creation order
        self._spilled: list = []               # SUBMITTED/DURABLE recs in spill order
        self._durable_pending: "deque" = deque()  # async: writes the writer signalled durable, awaiting main finish
        self._order = itertools.count()
        self.live_cpu_bytes = 0
        self.hi = _budget_from_env()
        self.lo = max(0, self.hi - _env_int("ASYM_NVME_ACT_LOW_SLACK_BYTES", 16 << 30))   # hysteresis
        self.prefetch_bytes = _env_int("ASYM_NVME_ACT_PREFETCH_BYTES", 0)                 # 0 until Stage 5
        assert self.prefetch_bytes < max(1, self.hi - self.lo), "prefetch must fit in the hi-lo band"
        self.stats = _GovStats()

    @property
    def align(self) -> int:
        return self._store.align

    def alloc_cpu(self, shape, dtype):
        """Aligned, padded pinned buffer (io_ready True) — used by activation_offload._alloc_cpu (C8)."""
        return alloc_padded_pinned(tuple(shape), dtype, align=self._store.align)

    # ---------- producer side ----------
    def on_offload(self, manager, handle) -> None:
        """Called ONLY at the two _mark_cpu_live sites (empty_cpu, adopt_cpu) — NOT offload() (C5).
        Pressure accounting for EVERY handle; nothing spillable until sealed."""
        substrate = "boundary" if handle.tag == "gc.boundary" else "fg"
        with self._lock:
            rec = _Rec(handle, manager, handle.nbytes, substrate, order=next(self._order))
            self._by_id[id(handle)] = rec
            self.live_cpu_bytes += rec.nbytes
            self.stats.note_live_peak(self.live_cpu_bytes)

    def on_seal(self, manager, handles, event=None) -> None:
        """ONE call at END of a Function.forward (Substrate B) or per boundary offload (A). The event
        orders after (a) the D2H fill and (b) every already-enqueued forward consumer (same stream).
        Handles failing io_ready or < min_swappable_bytes stay TRACKED (pressure-only)."""
        self._drain_durable()                           # async: reclaim completed spills first
        ev = event or _record_event()
        with self._lock:
            for h in handles:
                rec = self._by_id.get(id(h))
                if (rec is not None and rec.state == TRACKED
                        and rec.nbytes >= self._store.cfg.min_swappable_bytes
                        and io_ready(h.tensor, self._store.align)):
                    rec.seal_event = ev
                    rec.state = QUEUED
                    self._queue.append(rec)
        self._maybe_spill()

    def _maybe_spill(self) -> None:
        """Oldest-first until live <= lo. SYNC: inline blocking write on the caller (main) thread. The
        oldest sealed handles' events are long complete => synchronize() is ~free."""
        while True:
            with self._lock:
                if self.live_cpu_bytes <= self.hi or not self._queue:
                    return
                rec = self._queue.popleft()
                if rec.state != QUEUED:                 # CLAIMED/released while queued (lazy dequeue)
                    continue
                rec.state = SUBMITTED
            if self._store.cfg.sync:
                if rec.seal_event is not None:
                    rec.seal_event.synchronize()        # D2H + fwd consumers done => bytes stable
                rec.ref = self._store.spill_sync("activation", rec.handle.tensor)
                self._finish_spill(rec)                 # main thread
            else:                                       # async (Stage 5): writer syncs the seal event off main
                rec.ref = self._store.spill_async(
                    "activation", rec.handle.tensor,
                    ready_event=rec.seal_event,         # host-side sync ON THE WRITER (main never blocks)
                    on_done=self._make_on_durable(rec))
            with self._lock:
                self._spilled.append(rec)
                if self.live_cpu_bytes <= self.lo:
                    return

    def _finish_spill(self, rec) -> None:               # sync body / writer callback (Stage 5)
        from .activation_offload import _return_cpu
        with self._lock:
            if rec.state == CLAIMED:                    # async-only: consumed while in flight
                self._store.blob_done(rec.ref)
                self.stats.wasted_writes += 1
                return
            rec.state = DURABLE
            self.live_cpu_bytes -= rec.nbytes
            self.stats.note_spill(rec.substrate, rec.nbytes)
        rec.manager._pop_active(rec.handle)             # drop accounting + stale D2H event off old ptr
        _return_cpu(rec.handle.tensor, pin_memory=True)  # pool reuse safe: seal event already synced
        object.__setattr__(rec.handle, "tensor", _SENTINEL)  # stray compute read => loud 0-numel error

    def _make_on_durable(self, rec):                    # Stage 5: writer-thread callback (enqueue only)
        def _cb(_ref):
            with self._lock:
                self._durable_pending.append(rec)       # main thread runs _finish_spill via _drain_durable
        return _cb

    def _drain_durable(self) -> None:
        """MAIN THREAD: finish writes the writer signalled durable (async only). Keeps ALL manager/pool
        mutation on the main thread, so no pool lock is needed."""
        if self._store.cfg.sync:
            return
        while True:
            with self._lock:
                if not self._durable_pending:
                    return
                rec = self._durable_pending.popleft()
            self._finish_spill(rec)

    # ---------- consumer side (MAIN THREAD) ----------
    def ensure_local(self, handle) -> None:
        """First line of wait_cpu_ready() (stage*() already call it) + the one direct-read site
        (attn bwd, Stage 4). O(1) dict miss for untracked handles."""
        from .activation_offload import _alloc_cpu
        self._drain_durable()                           # async: reclaim completed spills first
        rec = self._by_id.get(id(handle))
        if rec is None:
            return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, SUBMITTED):   # CPU-valid (writer only READS the buffer)
                rec.state = CLAIMED
                return
            if rec.state in (CLAIMED, FETCHED):
                return
            assert rec.state == DURABLE
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)
        self._store.fetch_into(rec.ref, bounce)
        self._store.blob_done(rec.ref)
        object.__setattr__(handle, "tensor", bounce)
        rec.manager._mark_cpu_live(handle)              # re-enter accounting under the new data_ptr
        with self._lock:
            rec.state = FETCHED
            self.live_cpu_bytes += rec.nbytes
            self.stats.note_fetch(rec.substrate, rec.nbytes)
        self._maybe_spill()                             # fetch may push live over hi -> spill oldest (Belady)

    def on_release(self, handle) -> None:
        """FIRST line of release_cpu() — MUST precede its internal wait_cpu_ready, else a
        spilled-never-consumed handle is fetched just to be freed. Covers every terminal state."""
        rec = self._by_id.pop(id(handle), None)
        if rec is None:
            return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, CLAIMED, FETCHED):
                self.live_cpu_bytes -= rec.nbytes       # DURABLE bytes already decremented at spill
            if rec.state == SUBMITTED:
                rec.state = CLAIMED                      # async: on_durable will drop the blob
            elif rec.state == DURABLE:
                self._store.blob_done(rec.ref)           # spilled, never fetched
            elif rec.state == QUEUED:
                rec.state = CLAIMED                      # lazy-dequeue marker

    def summary(self) -> dict:
        self._drain_durable()                           # async: finish any completed spills before reporting
        with self._lock:
            open_recs = len(self._by_id)
            live = self.live_cpu_bytes
        return self.stats.as_dict() | {
            "live_peak_bytes": self.stats.live_peak,
            "live_cpu_bytes": live,
            "hi": self.hi,
            "lo": self.lo,
            "open_recs": open_recs,
        }


_GOV = None
_GOV_INIT = False


def get_act_spill_governor():
    global _GOV, _GOV_INIT
    if _GOV_INIT:
        return _GOV
    store = get_nvme_store()
    _GOV = ActSpillGovernor(store) if (store is not None and store.has_role("activation")) else None
    _GOV_INIT = True
    return _GOV


def _reset_governor_singleton_for_tests():
    global _GOV, _GOV_INIT
    _GOV = None
    _GOV_INIT = False
