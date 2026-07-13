"""AsymGEMM NVMe store — opt-in activation / base-weight spill substrate over DeepSpeed AIO.

Enabled ONLY when ``ASYM_NVME_ROLES`` is set (Design Contract rule 7: *off = byte-identical* —
without the env var this module imports nothing from deepspeed, allocates nothing, starts no
thread, and :func:`get_nvme_store` returns ``None``). Single-process only (``WORLD_SIZE==1``).

See ``agent/impls/nvme_offload_impl.md`` Stage 1 for the design. Hard AIO facts baked in from
re-verification:

* **C2** — ``async_pwrite``/``async_pread`` return ``-1`` on failure *without scheduling*; a
  following ``wait()`` then aborts the process (asserts are LIVE, ``-O0``). So we check
  ``rc == 0`` before every ``wait()`` and never ``wait()`` on an idle handle.
* **C3** — ``get_alignment() = intra_op_parallelism * 512`` (= 2048 @ intra=4). Padding to it
  auto-satisfies the ``len % intra_op == 0`` divisibility check too. Read ``align`` from the
  handle, never hardcode.
* **C8** — ``alloc_padded_pinned`` uses ``empty(padded_numel, pin)[:numel].view(shape)`` (a
  multi-dim ``set_`` onto an untyped storage is error-prone); ``is_pinned`` is storage-backed so
  the view stays pinned and its whole storage is align-padded.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _die(msg: str):
    raise RuntimeError(msg)


def _pad(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _numel(shape) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return p


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str                                    # ASYM_NVME_PATH (required when roles set)
    roles: frozenset                             # subset of {"base_weight","activation"}
    sync: bool = True                            # ASYM_NVME_SYNC (v1 default 1; Stage 5 flips to 0)
    aio_block_size: int = 1 << 20                # aio_handle ctor arg 1
    aio_queue_depth: int = 16                    # ctor arg 2 (AIO default 128; we pin 16)
    aio_single_submit: bool = False              # ctor arg 3
    aio_overlap_events: bool = True              # ctor arg 4
    aio_intra_op_parallelism: int = 4            # ctor arg 5 -> get_alignment() = 4*512 = 2048
    min_swappable_bytes: int = 1 << 20           # ASYM_NVME_MIN_SWAP_BYTES
    activation_arena_bytes: int = 1 << 41        # 2 TiB logical cap; sparse, allocated on demand
    max_inflight_spill_bytes: int = 8 << 30      # writer backpressure (async only)


def _config_from_env() -> "NVMeStoreConfig | None":
    roles = frozenset(r.strip() for r in os.environ.get("ASYM_NVME_ROLES", "").split(",") if r.strip())
    if not roles:
        return None
    bad = roles - {"base_weight", "activation"}
    if bad:
        raise ValueError(f"unknown ASYM_NVME_ROLES: {sorted(bad)}")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _die("asym NVMe store is single-process only (deferred)")
    path = os.environ.get("ASYM_NVME_PATH") or _die("ASYM_NVME_PATH required with ASYM_NVME_ROLES")
    return NVMeStoreConfig(
        path=path,
        roles=roles,
        sync=_env_bool("ASYM_NVME_SYNC", True),
        aio_block_size=_env_int("ASYM_NVME_AIO_BLOCK_SIZE", 1 << 20),
        aio_queue_depth=_env_int("ASYM_NVME_AIO_QUEUE_DEPTH", 16),
        aio_intra_op_parallelism=_env_int("ASYM_NVME_AIO_INTRA_OP", 4),
        min_swappable_bytes=_env_int("ASYM_NVME_MIN_SWAP_BYTES", 1 << 20),
        activation_arena_bytes=_env_int("ASYM_NVME_ACTIVATION_ARENA_BYTES", 1 << 41),
        max_inflight_spill_bytes=_env_int("ASYM_NVME_MAX_INFLIGHT_SPILL_BYTES", 8 << 30),
    )


# ---------------------------------------------------------------------------
# tensor <-> flat-byte helpers
# ---------------------------------------------------------------------------
def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of ``t``'s WHOLE storage (padded length) — one tensor == one IO op, never
    fragmented. Uses ``set_`` over the untyped storage (the standard flat-byte-view idiom)."""
    storage = t.untyped_storage()
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(storage, 0, (storage.nbytes(),))       # 1-D: contiguous stride (1,) implied
    return out


def alloc_padded_pinned(shape, dtype, *, align: int) -> torch.Tensor:
    """Pinned CPU tensor whose WHOLE storage is ``align``-padded; exact contiguous ``[shape]``
    view returned (C8). On pin failure fall back to unpinned (``io_ready`` then excludes it —
    never crash the run)."""
    elem = torch.empty(0, dtype=dtype).element_size()
    numel = _numel(shape)
    padded_numel = _pad(numel * elem, align) // elem   # align % elem == 0 for bf16/fp32 (align=2048)
    try:
        buf = torch.empty(padded_numel, dtype=dtype, pin_memory=True)
    except RuntimeError:
        buf = torch.empty(padded_numel, dtype=dtype)
    return buf[:numel].view(shape)                     # contiguous; storage == padded buf


def io_ready(t: torch.Tensor, align: int) -> bool:
    """Spill-eligibility: pinned + whole-storage length aligned. Foreign ``adopt_cpu`` tensors
    that fail this simply stay resident (counted for pressure, never spilled)."""
    return t.is_pinned() and (t.untyped_storage().nbytes() % align == 0)


# ---------------------------------------------------------------------------
# records + stats
# ---------------------------------------------------------------------------
@dataclass
class BlobRef:
    role: str
    file: str                                    # per-blob file (base_weight) | arena file (activation)
    offset: int                                  # aligned; 0 for per-blob files
    length: int                                  # padded bytes on disk == source storage nbytes
    logical_nbytes: int                          # actual tensor bytes (<= length)
    durable: threading.Event = field(default_factory=threading.Event)


@dataclass
class NVMeStoreStats:
    bytes_written: dict = field(default_factory=dict)
    bytes_read: dict = field(default_factory=dict)
    write_ops: dict = field(default_factory=dict)
    read_ops: dict = field(default_factory=dict)
    spill_wait_ms: float = 0.0                    # sync-mode inline write time (the v1 stall)
    fetch_wait_ms: float = 0.0
    spill_backpressure_ms: float = 0.0
    inflight_peak_bytes: int = 0
    arena_peak_bytes: dict = field(default_factory=dict)
    wasted_writes: int = 0

    def as_dict(self) -> dict:
        """Flat, prefixed so it survives the profile-row merge unchanged."""
        return {
            "asym_nvme_bytes_written": dict(self.bytes_written),
            "asym_nvme_bytes_read": dict(self.bytes_read),
            "asym_nvme_write_ops": dict(self.write_ops),
            "asym_nvme_read_ops": dict(self.read_ops),
            "asym_nvme_spill_wait_ms": self.spill_wait_ms,
            "asym_nvme_fetch_wait_ms": self.fetch_wait_ms,
            "asym_nvme_spill_backpressure_ms": self.spill_backpressure_ms,
            "asym_nvme_inflight_peak_bytes": self.inflight_peak_bytes,
            "asym_nvme_arena_peak_bytes": dict(self.arena_peak_bytes),
            "asym_nvme_wasted_writes": self.wasted_writes,
        }


# ---------------------------------------------------------------------------
# writer thread (async mode; Stage 5 exercises it) — SOLE owner of write handle in async
# ---------------------------------------------------------------------------
class _WriterThread(threading.Thread):
    _STOP = object()

    def __init__(self, write_h, cfg: NVMeStoreConfig, stats: NVMeStoreStats):
        super().__init__(daemon=True)
        self._h = write_h
        self._cfg = cfg
        self._stats = stats
        self._q: "queue.Queue" = queue.Queue()
        self._inflight = 0
        self._cv = threading.Condition()

    def submit(self, ready_event, buf, ref: BlobRef, on_done):
        with self._cv:
            t0 = time.perf_counter()
            while self._inflight + ref.length > self._cfg.max_inflight_spill_bytes and self._inflight > 0:
                self._cv.wait()                                    # backpressure
            self._stats.spill_backpressure_ms += (time.perf_counter() - t0) * 1e3
            self._inflight += ref.length
            self._stats.inflight_peak_bytes = max(self._stats.inflight_peak_bytes, self._inflight)
        self._q.put((ready_event, buf, ref, on_done))

    def run(self):
        while True:
            item = self._q.get()
            if item is self._STOP:
                return
            ready_event, buf, ref, on_done = item
            if ready_event is not None:
                ready_event.synchronize()                          # host-side wait — allowed off main
            rc = self._h.async_pwrite(buf, ref.file, ref.offset)
            if rc != 0:                                            # C2: check rc BEFORE wait()
                raise RuntimeError(f"writer async_pwrite rc={rc} file={ref.file} off={ref.offset}")
            n = self._h.wait()
            assert n == 1
            self._stats.bytes_written[ref.role] = self._stats.bytes_written.get(ref.role, 0) + ref.length
            self._stats.write_ops[ref.role] = self._stats.write_ops.get(ref.role, 0) + 1
            ref.durable.set()
            with self._cv:
                self._inflight -= ref.length
                self._cv.notify_all()
            on_done(ref)                                           # governor callback (CUDA-free)

    def stop(self):
        self._q.put(self._STOP)
        self.join()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class NVMeStore:
    def __init__(self, cfg: NVMeStoreConfig):
        from deepspeed.ops.op_builder import AsyncIOBuilder        # imported ONLY when enabled (rule 7)

        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(                                 # noqa: E731 - tiny local factory
            cfg.aio_block_size, cfg.aio_queue_depth, cfg.aio_single_submit,
            cfg.aio_overlap_events, cfg.aio_intra_op_parallelism)
        self.cfg = cfg
        self._read_h = mk()                                        # MAIN THREAD ONLY
        self._write_h = mk()                                       # main (sync) or writer thread (async)
        self.align = int(self._read_h.get_alignment())            # = intra_op*512 = 2048 (C3)
        self.stats = NVMeStoreStats()
        self._debug = _env_bool("ASYM_NVME_DEBUG", False)
        self._main_ident = threading.get_ident()
        os.makedirs(os.path.join(cfg.path, "base_weight"), exist_ok=True)
        self._blob_seq = 0
        self._arena_path = os.path.join(cfg.path, f"activation.{os.getpid()}.arena")
        self._arena_cursor = 0
        self._arena_live = 0
        self._pending_reads: dict = {}                             # id(ref) -> BlobRef (MAIN THREAD ONLY)
        self._writer = None if cfg.sync else _WriterThread(self._write_h, cfg, self.stats)
        if self._writer:
            self._writer.start()

    def has_role(self, role: str) -> bool:
        return role in self.cfg.roles

    def _assert_main(self):
        if self._debug:
            assert threading.get_ident() == self._main_ident, "read/main-only API off main thread"

    # --- filenames ---
    def _blob_file(self) -> str:
        self._blob_seq += 1
        return os.path.join(self.cfg.path, "base_weight", f"w{self._blob_seq:06d}.bin")

    # --- activation arena: bump allocator, reset-when-empty (blob lifetime = one microbatch fwd->bwd) ---
    def _arena_alloc(self, nbytes: int):
        length = _pad(nbytes, self.align)
        off = self._arena_cursor
        self._arena_cursor += length
        self._arena_live += 1
        self.stats.arena_peak_bytes["activation"] = max(
            self.stats.arena_peak_bytes.get("activation", 0), self._arena_cursor)
        if self._arena_cursor > self.cfg.activation_arena_bytes:
            raise RuntimeError("activation arena full — raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_path, off

    def blob_done(self, ref: BlobRef):                            # after final fetch OR dropped blob
        if ref.role == "activation":
            self._arena_live -= 1
            if self._arena_live == 0:
                self._arena_cursor = 0

    def _count_write(self, ref: BlobRef, nbytes: int):
        self.stats.bytes_written[ref.role] = self.stats.bytes_written.get(ref.role, 0) + nbytes
        self.stats.write_ops[ref.role] = self.stats.write_ops.get(ref.role, 0) + 1

    def _count_read(self, ref: BlobRef):
        self.stats.bytes_read[ref.role] = self.stats.bytes_read.get(ref.role, 0) + ref.length
        self.stats.read_ops[ref.role] = self.stats.read_ops.get(ref.role, 0) + 1

    # --- write paths ---
    def spill_sync(self, role: str, tensor: torch.Tensor) -> BlobRef:
        """MAIN THREAD. Caller has already synchronized the seal/ready event. Blocking write."""
        buf = _flat_u8(tensor)
        assert buf.numel() % self.align == 0                      # upstream guard: never feed C++ a
        file, off = ((self._blob_file(), 0) if role == "base_weight"           # misaligned length (would abort)
                     else self._arena_alloc(buf.numel()))
        ref = BlobRef(role, file, off, buf.numel(), tensor.numel() * tensor.element_size())
        t0 = time.perf_counter()
        rc = self._write_h.async_pwrite(buf, ref.file, ref.offset)   # C2: check rc BEFORE wait()
        if rc != 0:
            raise RuntimeError(f"async_pwrite rc={rc} (alignment/open) file={ref.file} off={ref.offset}")
        n = self._write_h.wait()
        assert n == 1
        self.stats.spill_wait_ms += (time.perf_counter() - t0) * 1e3
        self._count_write(ref, buf.numel())
        ref.durable.set()
        return ref

    def spill_async(self, role: str, tensor: torch.Tensor, *, ready_event, on_done) -> BlobRef:
        """ANY THREAD -> writer (Stage 5). ``ready_event`` synchronized ON THE WRITER before pwrite;
        ``on_done(ref)`` runs in writer context — NEVER touch CUDA there. ref alloc identical to sync."""
        buf = _flat_u8(tensor)
        assert buf.numel() % self.align == 0
        file, off = ((self._blob_file(), 0) if role == "base_weight"
                     else self._arena_alloc(buf.numel()))
        ref = BlobRef(role, file, off, buf.numel(), tensor.numel() * tensor.element_size())
        self._writer.submit(ready_event, buf, ref, on_done)          # writer sets ref.durable + counts
        return ref

    # --- read path (MAIN THREAD ONLY) ---
    def submit_pread(self, ref: BlobRef, dst_padded_pinned: torch.Tensor) -> None:
        self._assert_main()
        if not ref.durable.is_set():
            ref.durable.wait()                                       # write in flight (async) -> bounded rare block
        dst = _flat_u8(dst_padded_pinned)
        assert dst.numel() == ref.length                            # C2: exact padded length
        rc = self._read_h.async_pread(dst, ref.file, ref.offset)
        if rc != 0:
            raise RuntimeError(f"async_pread rc={rc} file={ref.file} off={ref.offset} len={ref.length}")
        self._pending_reads[id(ref)] = ref

    def drain_reads(self) -> set:
        """Blocks until ALL pending reads complete (``wait()`` drains the whole handle)."""
        if not self._pending_reads:
            return set()                                            # C2: never wait() idle
        n = self._read_h.wait()
        assert n == len(self._pending_reads)
        for r in self._pending_reads.values():
            self._count_read(r)
        done = set(self._pending_reads)
        self._pending_reads.clear()
        return done

    def fetch_into(self, ref: BlobRef, dst_padded_pinned: torch.Tensor) -> set:
        t0 = time.perf_counter()
        self.submit_pread(ref, dst_padded_pinned)
        done = self.drain_reads()                                   # drains any in-flight prefetches too (Stage 5)
        self.stats.fetch_wait_ms += (time.perf_counter() - t0) * 1e3
        return done

    def shutdown(self):
        # Best-effort: drain in-flight reads so the C++ aio_handle never destructs with pending
        # ops (its _stop_threads asserts 0 == _num_pending_ops and would SIGABRT the process,
        # masking whatever error caused the teardown).
        try:
            self.drain_reads()
        except Exception:
            pass
        if self._writer:
            self._writer.stop()


# ---------------------------------------------------------------------------
# lazy env singleton
# ---------------------------------------------------------------------------
_STORE: "NVMeStore | None" = None
_STORE_INIT = False


def get_nvme_store() -> "NVMeStore | None":
    """Lazy env singleton. Without ``ASYM_NVME_ROLES``: returns ``None``, imports nothing,
    allocates nothing (rule 7)."""
    global _STORE, _STORE_INIT
    if _STORE_INIT:
        return _STORE
    cfg = _config_from_env()
    _STORE = NVMeStore(cfg) if cfg is not None else None
    _STORE_INIT = True
    if _STORE is not None:
        import atexit

        atexit.register(_STORE.shutdown)
    return _STORE


def _reset_store_singleton_for_tests():
    """Test-only: drop the cached singleton (and stop its writer) so a fresh env is picked up."""
    global _STORE, _STORE_INIT
    if _STORE is not None:
        try:
            _STORE.shutdown()
        except Exception:
            pass
    _STORE = None
    _STORE_INIT = False
