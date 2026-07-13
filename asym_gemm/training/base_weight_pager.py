"""Base-weight NVMe pager — the ``asym_cpuadamwds_panvme`` engine (Stage 7).

A faithful re-implementation of DeepSpeed ZeRO-Infinity's parameter-NVMe path
(``AsyncPartitionedParameterSwapper`` + ``PartitionedParameterCoordinator``), specialized to
**frozen** :class:`~asym_gemm.training.host_weight.HostWeight` blobs (see
``agent/impls/nvme_offload_impl.md`` Stage 7 / §7.0 for the verified lineage table):

* write-once at :meth:`BaseWeightPager.register` (frozen ⇒ no re-``swap_out`` after steps, no
  write-after-read hazard — strictly simpler than DeepSpeed's trainable-param swapper);
* ``{ABSENT, INFLIGHT, RESIDENT}`` status machine ≡ DeepSpeed ``PartitionedParamStatus``;
* step-1 touch trace → freeze → cursor advance ≡ the coordinator's RECORD→COMPLETE machine;
  any beyond-window mismatch degrades to miss-driven sync fetches (≡ ``_invalidate_trace``) —
  **correctness never depends on prefetch**;
* byte-budgeted read-ahead over the frozen trace, default depth = 2× the largest blob ≡ the
  coordinator's ``__prefetch_nvme_param_partitions`` 2×-in-flight double-buffer;
* Belady eviction (farthest next-use on the frozen trace) ≡ ``max_reuse_distance``, exact here;
* evicted buffers are quarantined behind a CUDA event before reuse (design-contract rule 6 —
  ``_asym_bf16_nt`` returns while still streaming the pinned weight).

MAIN THREAD ONLY (same single-owner rule as the store's read handle). Off = byte-identical:
without a store with the ``base_weight`` role nothing constructs this class, and
``HostWeight._pager is None`` keeps every property on today's exact path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .nvme_store import NVMeStore, _env_bool, _env_int, _pad, alloc_padded_pinned

logger = logging.getLogger(__name__)

ABSENT, INFLIGHT, RESIDENT = range(3)


@dataclass
class _Entry:
    key: str
    hw: Any                                       # HostWeight (weak coupling: only ._tensor/._pager set here)
    ref: Any                                      # BlobRef
    shape: tuple
    dtype: torch.dtype
    padded_nbytes: int                            # == ref.length (whole-storage IO length)
    buf: Any = None                               # exact-shape pinned view over padded storage
    state: int = ABSENT
    positions: list = field(default_factory=list)  # trace indices, set at freeze
    next_gap: dict = field(default_factory=dict)  # position -> cyclic gap to next occurrence (freeze)
    next_abs: int = -1                            # absolute touch-count of next use; <= _abs ⇒ stale
    last_touch_seq: int = -1                      # pre-freeze LRU fallback


class _Quarantined:
    __slots__ = ("buf", "event", "cls", "padded_nbytes")

    def __init__(self, buf, event, cls, padded_nbytes):
        self.buf = buf
        self.event = event                        # None ⇒ immediately reusable
        self.cls = cls                            # (dtype, shape) free-list key
        self.padded_nbytes = padded_nbytes

    def ready(self) -> bool:
        return self.event is None or bool(self.event.query())


def _empty_pinned_host_cache() -> bool:
    """Release torch's CachingHostAllocator cache so freed pinned homes actually leave RSS.
    Best-effort across torch versions (2.12 has ``torch._C._host_emptyCache``)."""
    for fn_name in ("_host_emptyCache", "_accelerator_emptyHostCache"):
        fn = getattr(torch._C, fn_name, None)
        if fn is not None:
            try:
                fn()
                return True
            except Exception:  # pragma: no cover - defensive
                continue
    return False


class BaseWeightPager:
    """Trace-prefetching pinned cache over the NVMe store's ``base_weight`` role."""

    def __init__(
        self,
        store: NVMeStore,
        *,
        cache_bytes: int | None = None,
        prefetch_bytes: int | None = None,
        jitter_window: int | None = None,
    ) -> None:
        self._store = store
        self.cache_bytes = int(
            cache_bytes if cache_bytes is not None else _env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 16 << 30)
        )
        # env<=0 -> auto at finalize(): 2x largest blob (DeepSpeed 2x-in-flight double-buffer).
        # An explicit ctor value (including 0 = prefetch OFF) is honored verbatim.
        if prefetch_bytes is None:
            self.prefetch_bytes = _env_int("ASYM_NVME_BASE_WEIGHT_PREFETCH_BYTES", 0)
            self._auto_prefetch = self.prefetch_bytes <= 0
        else:
            self.prefetch_bytes = int(prefetch_bytes)
            self._auto_prefetch = False
        self.jitter_window = int(
            jitter_window if jitter_window is not None else _env_int("ASYM_NVME_BASE_WEIGHT_JITTER_WINDOW", 8)
        )
        # Transient over-cap slack while quarantined buffers ripen (DeepSpeed buffer_count-slack
        # analog). Too tight ⇒ ceiling event-syncs on the hot path (78 s / 5 steps measured at
        # 4.5 GiB during bring-up); the floor of 2x-largest is enforced at finalize().
        self.overshoot_bytes = _env_int("ASYM_NVME_BASE_WEIGHT_OVERSHOOT_BYTES", 8 << 30)
        self._debug = _env_bool("ASYM_NVME_DEBUG", False)
        self._main_ident = threading.get_ident()

        self._entries: dict[str, _Entry] = {}
        self._by_ref_id: dict[int, _Entry] = {}
        self._free: dict[tuple, list] = {}         # (dtype, shape) -> [padded pinned bufs]
        self._quarantine: list[_Quarantined] = []  # event-gated reuse (rule 6)

        # trace machine (DeepSpeed coordinator RECORD -> COMPLETE -> INVALID). Freeze is anchored
        # on OPTIMIZER-STEP boundaries via mark_step() — the reset_step() analog — because under
        # GC recompute each weight is touched 3x per step (fwd, bwd-recompute, bwd-inner) and any
        # same-key-recurrence heuristic freezes mid-backward (observed at bring-up). mark_step()
        # is wired from run_lf_profiled_train.py's optimizer-step wrapper; if it never fires the
        # pager simply stays in RECORD = miss-driven sync fetches (correct, unprefetched).
        self._trace: list[str] = []
        self._recording = False
        self._marks = 0
        self._frozen = False
        self._disabled = False
        self._period = 0
        self._cursor = -1
        self._abs = 0                                      # frozen-trace touch clock (for next_abs)
        self._last_key: str | None = None
        self._touch_seq = 0

        # accounting
        self.registered_bytes = 0
        self.largest_padded_nbytes = 0
        self._held_bytes = 0                       # entry bufs (RESIDENT/INFLIGHT) + free + quarantine
        self._inflight_prefetch_bytes = 0
        self.misses = 0
        self.misses_after_freeze = 0
        self.evictions = 0
        self.prefetch_submits = 0
        self.over_budget_allocs = 0
        self.staging_copies = 0
        self.quarantine_block_ms = 0.0

        global _PAGER
        _PAGER = self                              # process-wide accessor for the step-boundary hook

    # ------------------------------------------------------------------ util
    def _assert_main(self):
        if self._debug:
            assert threading.get_ident() == self._main_ident, "BaseWeightPager is main-thread only"

    def _make_event(self):
        """Post-launch CUDA event guarding buffer reuse; None when CUDA is absent (tests)."""
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            ev = torch.cuda.Event()
            ev.record()
            return ev
        return None

    # ------------------------------------------------------------ registration
    def register(self, key: str, hw: Any) -> bool:
        """Write ``hw``'s frozen home to NVMe once and free the CPU home. Returns True if paged.

        Small (< ``min_swappable_bytes``), already-paged (alias), or homeless weights are left
        untouched — they simply stay resident, exactly like DeepSpeed's ``swappable_tensor``
        gate (`partitioned_param_swapper.py:136-142`).
        """
        self._assert_main()
        assert not self._frozen, "register() after the trace froze — registration is startup-only"
        if getattr(hw, "_pager", None) is not None:      # alias: same HostWeight seen twice
            return False
        t = hw._tensor
        if t is None or not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            return False
        logical = t.numel() * t.element_size()
        if logical < self._store.cfg.min_swappable_bytes:
            return False
        if key in self._entries:
            raise ValueError(f"duplicate pager key {key!r}")

        padded = _pad(logical, self._store.align)
        storage_nbytes = t.untyped_storage().nbytes()
        if (
            t.is_contiguous()
            and t.storage_offset() == 0
            and storage_nbytes == padded
            and t.is_pinned()
        ):
            # Fast path: the pinned home IS a valid IO buffer — write it directly, zero staging.
            ref = self._store.spill_sync("base_weight", t)
        else:
            # Staging path (odd shapes / unpinned): bounce through a padded pinned copy.
            stage = alloc_padded_pinned(tuple(t.shape), t.dtype, align=self._store.align)
            stage.copy_(t)
            ref = self._store.spill_sync("base_weight", stage)
            self.staging_copies += 1
            del stage                                    # returns to torch's host cache; finalize() empties it

        entry = _Entry(
            key=key, hw=hw, ref=ref, shape=tuple(t.shape), dtype=t.dtype, padded_nbytes=ref.length
        )
        self._entries[key] = entry
        self._by_ref_id[id(ref)] = entry
        hw._pager = self
        hw._pager_key = key
        hw._tensor = None                                # the GB-scale home is freed NOW
        self.registered_bytes += logical
        self.largest_padded_nbytes = max(self.largest_padded_nbytes, entry.padded_nbytes)
        return True

    def finalize(self) -> None:
        """Call once after the registration walk: pin defaults, sanity-check the cache size, and
        release the freed pinned homes from torch's host-allocator cache (the actual RSS drop)."""
        if self._entries:
            if self._auto_prefetch:
                self.prefetch_bytes = 2 * self.largest_padded_nbytes   # DS 2x-in-flight analog
            need = 2 * self.largest_padded_nbytes
            if self.cache_bytes < need:
                raise RuntimeError(
                    f"ASYM_NVME_BASE_WEIGHT_CACHE_BYTES={self.cache_bytes} must hold >= 2x the "
                    f"largest blob ({need} B) for double-buffering"
                )
        released = _empty_pinned_host_cache()
        logger.info(
            "base_weight_pager: registered %d blobs / %.2f GiB -> NVMe (largest %.2f GiB, "
            "cache %.2f GiB, prefetch %.2f GiB, staging_copies=%d, host_cache_released=%s)",
            len(self._entries), self.registered_bytes / (1 << 30),
            self.largest_padded_nbytes / (1 << 30), self.cache_bytes / (1 << 30),
            self.prefetch_bytes / (1 << 30), self.staging_copies, released,
        )

    # ----------------------------------------------------------------- trace
    def mark_step(self) -> None:
        """Optimizer-step boundary (DeepSpeed ``reset_step`` analog). Step 1 warms the cache
        miss-driven; step 2 is RECORDed; the trace freezes at the end of step 2 — one full
        steady-state period (incl. recompute re-touches and grad-accum repeats) per cycle."""
        self._assert_main()
        self._marks += 1
        self._sweep_quarantine()
        if self._frozen or self._disabled:
            return
        if self._marks == 1:
            self._recording = True
            self._trace = []
            return
        if self._recording:
            self._period = len(self._trace)
            if self._period == 0:
                self._disabled = True                      # no touches recorded — nothing to drive
                return
            for pos, key in enumerate(self._trace):
                self._entries[key].positions.append(pos)
            # Successor gaps make eviction next-use O(1): full per-eviction rescans of every
            # entry's position list measured 0.93 ms x ~6.9k evictions/step = 6.4 s/step of
            # main-thread time on q3-32b s20000 b8.
            for e in self._entries.values():
                ps = e.positions                           # ascending (appended in pos order)
                for i, p in enumerate(ps):
                    nxt = ps[i + 1] if i + 1 < len(ps) else ps[0] + self._period
                    e.next_gap[p] = nxt - p
                e.next_abs = -1                            # heal lazily on first query
            self._abs = 0
            self._frozen = True
            self._recording = False
            self._cursor = self._period - 1                # last recorded touch = end of the period
            logger.info(
                "base_weight_pager: trace frozen at step boundary %d, period=%d touches over %d blobs",
                self._marks, self._period, len(self._entries),
            )

    def _record_or_advance(self, e: _Entry) -> None:
        if self._disabled:
            return
        if not self._frozen:
            if self._recording:
                self._trace.append(e.key)                  # RECORD (step 2 only)
            return
        # COMPLETE: cursor-advance with a jitter window; beyond-window mismatch -> INVALID
        # (miss-driven sync fallback; counted, loud in summary()).
        for d in range(1, min(self.jitter_window, self._period) + 1):
            m = (self._cursor + d) % self._period
            if self._trace[m] == e.key:
                self._cursor = m
                self._abs += d                             # monotone touch clock (never wraps)
                e.next_abs = self._abs + e.next_gap[m]
                return
        self._disabled = True
        logger.warning(
            "base_weight_pager: trace mismatch at cursor=%d for key=%s (window=%d) — "
            "prefetch disabled, falling back to miss-driven sync fetches",
            self._cursor, e.key, self.jitter_window,
        )

    def _next_use(self, e: _Entry) -> int:
        """Cyclic distance (in touches) from the cursor to ``e``'s next use on the frozen trace.
        O(1) via the successor-gap clock kept by ``_record_or_advance``; a jitter-window skip
        leaves ``next_abs`` in the past — heal with one exact position scan and re-anchor."""
        if not e.positions:
            return self._period + 1                       # never on trace: farthest possible
        if e.next_abs > self._abs:
            return e.next_abs - self._abs
        best = self._period
        for p in e.positions:
            d = (p - self._cursor) % self._period
            if d == 0:
                d = self._period
            if d < best:
                best = d
        e.next_abs = self._abs + best
        return best

    # --------------------------------------------------------------- buffers
    def _sweep_quarantine(self) -> None:
        still = []
        for q in self._quarantine:
            if q.ready():
                self._free.setdefault(q.cls, []).append(q.buf)
            else:
                still.append(q)
        self._quarantine = still
        # Trim: transient over-cap surplus (from overshoot allocs) must not become steady-state
        # residency — drop free buffers until we are back under the cap.
        while self._held_bytes > self.cache_bytes:
            fcls = next((c for c in self._free if self._free[c]), None)
            if fcls is None:
                break
            self._drop_free_buffer(fcls)

    def _drop_free_buffer(self, cls: tuple) -> None:
        """Drop one free-listed buffer of class ``cls`` (shape mismatch under pressure)."""
        buf = self._free[cls].pop()
        self._held_bytes -= buf.untyped_storage().nbytes()
        del buf

    def _pick_victim(self, protect: set, farther_than: int | None):
        best, best_d = None, float("-inf")        # pre-freeze LRU scores are negative seqs
        for e in self._entries.values():
            if e.state != RESIDENT or e.key in protect:
                continue
            d = self._next_use(e) if self._frozen and not self._disabled else -e.last_touch_seq
            if farther_than is not None and d <= farther_than:
                continue
            if d > best_d:
                best, best_d = e, d
        return best

    def _take_buffer(self, e: _Entry, *, protect: set, farther_than: int | None = None):
        """Return a padded pinned buffer for ``e``: free-list -> grow-within-budget -> ONE Belady
        eviction (evictee quarantined behind its CUDA event, rule 6) -> transient over-cap alloc
        while quarantined buffers ripen. The hot path NEVER blocks on a CUDA event (measured
        33 s/step at bring-up: eviction events record at the CPU-side stream tail while the GPU
        runs seconds behind); the over-cap surplus is reclaimed by the mismatched-free drop as
        events complete — the DeepSpeed buffer_count-slack analog. Hard event-sync only at the
        +overshoot ceiling. With ``farther_than`` set (prefetch), returns None instead of
        evicting nearer-use entries, over-allocating, or blocking."""
        cls = (e.dtype, e.shape)
        evicted_this_call = False
        for _attempt in range(2 * len(self._entries) + 4):
            self._sweep_quarantine()
            free = self._free.get(cls)
            if free:
                return free.pop()
            if self._held_bytes + e.padded_nbytes <= self.cache_bytes:
                buf = alloc_padded_pinned(e.shape, e.dtype, align=self._store.align)
                self._held_bytes += e.padded_nbytes
                return buf
            # Over budget: drop mismatched-shape free buffers first (cheapest reclaim) ...
            dropped = False
            for fcls in list(self._free):
                if fcls != cls and self._free[fcls]:
                    self._drop_free_buffer(fcls)
                    dropped = True
                    break
            if dropped:
                continue
            # ... then at most ONE Belady eviction per call (its buffer usually ripens by the
            # next miss and circulates through the same-shape free list without any new alloc) ...
            if not evicted_this_call:
                victim = self._pick_victim(protect, farther_than)
                if victim is not None:
                    self._quarantine.append(
                        _Quarantined(victim.buf, self._make_event(),
                                     (victim.dtype, victim.shape), victim.padded_nbytes)
                    )
                    victim.buf = None
                    victim.state = ABSENT
                    self.evictions += 1
                    evicted_this_call = True
                    continue                               # next pass: ripe -> free-list hit
            if farther_than is not None:
                return None                                # prefetch backs off; never blocks/forces
            overshoot = max(2 * self.largest_padded_nbytes, self.overshoot_bytes)
            if self._held_bytes + e.padded_nbytes <= self.cache_bytes + overshoot:
                self.over_budget_allocs += 1               # transient; reclaimed on next sweeps
                buf = alloc_padded_pinned(e.shape, e.dtype, align=self._store.align)
                self._held_bytes += e.padded_nbytes
                return buf
            if self._quarantine:                           # ceiling: last-resort bounded sync
                oldest = self._quarantine[0]
                if oldest.event is not None:
                    t0 = time.perf_counter()
                    oldest.event.synchronize()
                    self.quarantine_block_ms += (time.perf_counter() - t0) * 1e3
                    oldest.event = None
                continue
            self.over_budget_allocs += 1                   # nothing evictable at all: forced
            buf = alloc_padded_pinned(e.shape, e.dtype, align=self._store.align)
            self._held_bytes += e.padded_nbytes
            return buf
        raise RuntimeError("base_weight_pager: buffer acquisition did not converge")  # pragma: no cover

    # ----------------------------------------------------------------- fetch
    def _settle(self, arrived_ref_ids: set) -> None:
        for rid in arrived_ref_ids:
            entry = self._by_ref_id.get(rid)
            if entry is not None and entry.state == INFLIGHT:
                entry.state = RESIDENT
                self._inflight_prefetch_bytes = max(
                    0, self._inflight_prefetch_bytes - entry.padded_nbytes
                )

    def _issue_prefetches(self, protect: set) -> None:
        if not self._frozen or self._disabled or self.prefetch_bytes <= 0:
            return
        inflight = self._inflight_prefetch_bytes
        # Scan cap: prefetching >256 touches ahead is pointless and a full-period scan per touch
        # would waste host time when the whole model fits the cache (everything RESIDENT).
        for d in range(1, min(self._period, 256)):
            if inflight >= self.prefetch_bytes:
                break
            e = self._entries[self._trace[(self._cursor + d) % self._period]]
            if e.state != ABSENT:
                continue
            buf = self._take_buffer(e, protect=protect, farther_than=d)
            if buf is None:
                break                                       # cache too tight to prefetch past here
            e.buf = buf
            e.state = INFLIGHT
            self._store.submit_pread(e.ref, buf)
            inflight += e.padded_nbytes
            self._inflight_prefetch_bytes += e.padded_nbytes
            self.prefetch_submits += 1

    def touch(self, key: str) -> torch.Tensor:
        """The single consumer entry point (HostWeight ``.weight``/``.tensor`` when paged).
        Returns a RESIDENT pinned view — never an in-flight or stale buffer."""
        self._assert_main()
        e = self._entries[key]
        if key != self._last_key:                          # dedupe multi-reads within one Function
            self._last_key = key
            self._touch_seq += 1
            e.last_touch_seq = self._touch_seq
            self._record_or_advance(e)

        if e.state == INFLIGHT:
            self._settle(self._store.drain_reads())        # drains ALL pending -> includes e
            if e.state != RESIDENT:  # pragma: no cover - store invariant
                raise RuntimeError(f"prefetched read for {key} did not arrive")
        elif e.state != RESIDENT:
            buf = self._take_buffer(e, protect={key})
            arrived = self._store.fetch_into(e.ref, buf)   # drains in-flight prefetches too
            e.buf = buf
            e.state = RESIDENT
            self._settle(arrived)                          # no-op for e (already RESIDENT)
            self.misses += 1
            if self._frozen and not self._disabled:
                self.misses_after_freeze += 1

        self._issue_prefetches(protect={key})
        return e.buf

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "step_marks": self._marks,
            "registered": len(self._entries),
            "registered_bytes": self.registered_bytes,
            "largest_padded_nbytes": self.largest_padded_nbytes,
            "cache_bytes": self.cache_bytes,
            "prefetch_bytes": self.prefetch_bytes,
            "overshoot_bytes": self.overshoot_bytes,
            "jitter_window": self.jitter_window,
            "resident_bytes": self._held_bytes,
            "trace_frozen": self._frozen,
            "trace_disabled": self._disabled,
            "trace_period": self._period,
            "misses": self.misses,
            "misses_after_freeze": self.misses_after_freeze,
            "evictions": self.evictions,
            "prefetch_submits": self.prefetch_submits,
            "over_budget_allocs": self.over_budget_allocs,
            "staging_copies": self.staging_copies,
            "quarantine_block_ms": self.quarantine_block_ms,
        }


_PAGER: "BaseWeightPager | None" = None


def get_base_weight_pager() -> "BaseWeightPager | None":
    """Most-recently-constructed pager (one per process under panvme); None when disabled."""
    return _PAGER


__all__ = ["BaseWeightPager", "get_base_weight_pager", "ABSENT", "INFLIGHT", "RESIDENT"]
