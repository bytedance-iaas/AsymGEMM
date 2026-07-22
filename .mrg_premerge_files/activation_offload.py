from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import torch


_CPU_BUFFER_POOL: dict[tuple[torch.dtype, tuple[int, ...], bool], list[torch.Tensor]] = {}
_CPU_BUFFER_POOL_EVICTIONS = 0
_POOL_PIN_FALLBACK_CALLS = 0  # requested-pin allocs that fell back to pageable (process-wide)
_CPU_BUFFER_POOL_MAX_CACHED_BYTES = 0
_DEFAULT_CPU_POOL_MAX_BYTES = 32 * 1024**3

# actnvme governor (Stage 3): None unless ASYM_NVME_ROLES contains "activation" (rule 7).
# Bound at module bottom by get_act_spill_governor(); every hook below is inert while it is None.
_GOV = None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _cpu_pool_max_bytes() -> int:
    raw = os.environ.get("ASYM_EXPACT_CPU_POOL_MAX_BYTES")
    if raw is None or raw == "":
        return _DEFAULT_CPU_POOL_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_CPU_POOL_MAX_BYTES


_DEFAULT_FG_ELEMENTWISE_CHUNK_MB = 1024  # probe-swept 2026-07-12: 1024 minimizes
# reserved (fragmentation gap 21→5 GiB @120k); 512 and 2048 are both worse.


def fg_elementwise_chunk_bytes() -> int:
    """Staging-buffer byte budget for row-chunked fg elementwise phases (silu/mul,
    silu backward, LoRA-B delta adds). 0 disables chunking (legacy full-width
    staging). Chunking trades nothing but launch count: same copied bytes, same
    FLOPs, but only chunk-sized HBM transients instead of full-width ones."""
    raw = os.environ.get(
        "ASYMM_FG_ELEMENTWISE_CHUNK_MB",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_FG_ELEMENTWISE_CHUNK_MB"),
    )
    if raw is None or str(raw).strip() == "":
        mb = _DEFAULT_FG_ELEMENTWISE_CHUNK_MB
    else:
        try:
            mb = int(str(raw).strip())
        except ValueError:
            mb = _DEFAULT_FG_ELEMENTWISE_CHUNK_MB
    return max(0, mb) * 1024 * 1024


def fg_chunk_rows(total_rows: int, row_width: int, element_size: int = 2) -> int:
    """Rows per chunk for fg elementwise chunking; 0 means do not chunk."""
    budget = fg_elementwise_chunk_bytes()
    if budget <= 0 or total_rows <= 0 or row_width <= 0:
        return 0
    rows = budget // max(1, row_width * element_size)
    if rows >= total_rows:
        return 0
    return max(8192, int(rows))


def _cpu_pool_cached_bytes() -> int:
    return sum(_tensor_nbytes(tensor) for tensors in _CPU_BUFFER_POOL.values() for tensor in tensors)


def _cpu_pool_shape_key(key: tuple[torch.dtype, tuple[int, ...], bool]) -> str:
    dtype, shape, pinned = key
    return f"{dtype}:{tuple(int(dim) for dim in shape)}:pinned={int(pinned)}"


def _trim_cpu_pool(max_cached_bytes: int) -> None:
    global _CPU_BUFFER_POOL_EVICTIONS
    while _cpu_pool_cached_bytes() > max_cached_bytes and _CPU_BUFFER_POOL:
        key = next(iter(_CPU_BUFFER_POOL))
        tensors = _CPU_BUFFER_POOL[key]
        if tensors:
            tensors.pop()
            _CPU_BUFFER_POOL_EVICTIONS += 1
        if not tensors:
            _CPU_BUFFER_POOL.pop(key, None)


def activation_offload_cpu_pool_stats() -> dict[str, Any]:
    cached_by_shape = {
        _cpu_pool_shape_key(key): sum(_tensor_nbytes(tensor) for tensor in tensors)
        for key, tensors in _CPU_BUFFER_POOL.items()
        if tensors
    }
    return {
        "cpu_pool_cached_bytes": _cpu_pool_cached_bytes(),
        "cpu_pool_cached_bytes_by_shape": cached_by_shape,
        "cpu_pool_num_cached_tensors": sum(len(tensors) for tensors in _CPU_BUFFER_POOL.values()),
        "cpu_pool_evictions": _CPU_BUFFER_POOL_EVICTIONS,
        # process-wide, same value on every row: max across rows, never sum
        "cpu_pool_pin_fallback_calls": _POOL_PIN_FALLBACK_CALLS,
        "cpu_pool_max_cached_bytes": _CPU_BUFFER_POOL_MAX_CACHED_BYTES,
        "cpu_pool_limit_bytes": _cpu_pool_max_bytes(),
    }


def clear_activation_offload_cpu_pool() -> None:
    global _CPU_BUFFER_POOL_EVICTIONS, _CPU_BUFFER_POOL_MAX_CACHED_BYTES
    _CPU_BUFFER_POOL.clear()
    _CPU_BUFFER_POOL_EVICTIONS = 0
    _CPU_BUFFER_POOL_MAX_CACHED_BYTES = 0


_POOL_ROW_BUCKET = 65536  # dim-0 bucket for >=2D buffers: variable routed-row counts (they
# differ per layer/branch/step under EP sharding) must collapse onto few reusable keys, or
# every allocation is a fresh cudaHostAlloc (measured: ~7 GB/layer of pinned allocs = 100s+
# of backward). Buffers are allocated at the bucketed size; callers receive an exact-shape
# dim-0 narrow (contiguous); _return_cpu recovers the base via `_asym_pool_base`.


def _alloc_cpu(shape: tuple[int, ...], dtype: torch.dtype, *, pin_memory: bool) -> torch.Tensor:
    pinned = bool(pin_memory and torch.cuda.is_available())
    shape = tuple(int(dim) for dim in shape)
    bucket_rows = 0
    if _GOV is None and len(shape) >= 2 and shape[0] > 0:
        granule = _POOL_ROW_BUCKET if shape[0] > _POOL_ROW_BUCKET else max(8192, 1)
        bucket_rows = ((shape[0] + granule - 1) // granule) * granule
    alloc_shape = (bucket_rows,) + shape[1:] if bucket_rows else shape
    key = (dtype, alloc_shape, pinned)

    def _narrow(base: torch.Tensor) -> torch.Tensor:
        if not bucket_rows or base.shape[0] == shape[0]:
            return base
        view = base.narrow(0, 0, shape[0])
        view._asym_pool_base = base  # type: ignore[attr-defined]
        return view

    pool = _CPU_BUFFER_POOL.get(key)
    if pool:
        tensor = pool.pop()
        if not pool:
            _CPU_BUFFER_POOL.pop(key, None)
        return _narrow(tensor)
    if _GOV is not None and pinned:
        # C8: aligned, padded pinned storage so io_ready() is True (spill-eligible); exact-shape view.
        return _GOV.alloc_cpu(key[1], dtype)
    try:
        return _narrow(torch.empty(alloc_shape, device="cpu", dtype=dtype, pin_memory=pinned))
    except RuntimeError:
        # Pinned alloc failed (host pinned-pool exhaustion): the pageable buffer keeps
        # the step alive but silently downgrades every later touch (blocking copies,
        # no ready events). Count it — same read rule as the saved-tensor modules'
        # pin_fallback counters: process-wide, max-across-rows, never sum.
        global _POOL_PIN_FALLBACK_CALLS
        if pinned:
            _POOL_PIN_FALLBACK_CALLS += 1
        return _narrow(torch.empty(alloc_shape, device="cpu", dtype=dtype))


def _return_cpu(tensor: torch.Tensor, *, pin_memory: bool) -> None:
    global _CPU_BUFFER_POOL_EVICTIONS, _CPU_BUFFER_POOL_MAX_CACHED_BYTES
    base = getattr(tensor, "_asym_pool_base", None)
    if base is not None:
        tensor = base  # return the full bucketed buffer, keyed by its own shape
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        return
    pinned = bool(pin_memory and torch.cuda.is_available() and tensor.is_pinned())
    key = (tensor.dtype, tuple(int(dim) for dim in tensor.shape), pinned)
    returned = tensor.detach()
    nbytes = _tensor_nbytes(returned)
    max_cached_bytes = _cpu_pool_max_bytes()
    if nbytes > max_cached_bytes:
        _CPU_BUFFER_POOL_EVICTIONS += 1
        return
    _trim_cpu_pool(max_cached_bytes - nbytes)
    _CPU_BUFFER_POOL.setdefault(key, []).append(returned)
    _CPU_BUFFER_POOL_MAX_CACHED_BYTES = max(_CPU_BUFFER_POOL_MAX_CACHED_BYTES, _cpu_pool_cached_bytes())


@dataclass(frozen=True)
class CPUActivationHandle:
    tag: str
    tensor: torch.Tensor
    original_device: torch.device
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.tensor)


@dataclass
class ActivationOffloadStats:
    offloaded_bytes: int = 0
    cpu_owned_bytes: int = 0
    cpu_peak_bytes_live: int = 0
    staged_bytes: int = 0
    max_stage_bytes_live: int = 0
    num_offloads: int = 0
    num_cpu_allocs: int = 0
    num_stages: int = 0
    offload_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    cpu_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    cpu_peak_by_tag: dict[str, int] = field(default_factory=dict)
    stage_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    stage_peak_by_tag: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        pool_stats = activation_offload_cpu_pool_stats()
        return {
            "offloaded_bytes": self.offloaded_bytes,
            "cpu_owned_bytes": self.cpu_owned_bytes,
            "cpu_live_bytes": self.cpu_owned_bytes,
            "cpu_peak_bytes_live": self.cpu_peak_bytes_live,
            "staged_bytes": self.staged_bytes,
            "max_stage_bytes_live": self.max_stage_bytes_live,
            "num_offloads": self.num_offloads,
            "num_cpu_allocs": self.num_cpu_allocs,
            "num_stages": self.num_stages,
            "offload_bytes_by_tag": dict(self.offload_bytes_by_tag),
            "cpu_bytes_by_tag": dict(self.cpu_bytes_by_tag),
            "cpu_peak_by_tag": dict(self.cpu_peak_by_tag),
            "stage_bytes_by_tag": dict(self.stage_bytes_by_tag),
            "stage_peak_by_tag": dict(self.stage_peak_by_tag),
            **pool_stats,
        }


class ActivationOffloadManager:
    """Pinned CPU activation owner plus reusable HBM staging buffers."""

    def __init__(self, *, pin_memory: bool = True) -> None:
        self.pin_memory = bool(pin_memory)
        self.stats = ActivationOffloadStats()
        self._stage_cache: dict[tuple[str, torch.dtype, tuple[int, ...], str], torch.Tensor] = {}
        self._stage_keys_by_ptr: dict[int, tuple[str, torch.dtype, tuple[int, ...], str]] = {}
        self._active_cpu_bytes: dict[int, tuple[int, str]] = {}
        self._active_stage_bytes: dict[int, tuple[int, str]] = {}
        self._pending_cpu_ready_events: dict[int, torch.cuda.Event] = {}

    def empty_cpu(
        self,
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
        original_device: torch.device,
        tag: str,
    ) -> CPUActivationHandle:
        cpu = _alloc_cpu(tuple(int(dim) for dim in shape), dtype, pin_memory=self.pin_memory)
        self.stats.num_cpu_allocs += 1
        handle = CPUActivationHandle(
            tag=tag,
            tensor=cpu,
            original_device=torch.device(original_device),
            original_dtype=dtype,
            original_shape=tuple(int(dim) for dim in shape),
        )
        self._mark_cpu_live(handle)
        if _GOV is not None:
            _GOV.on_offload(self, handle)
        return handle

    def offload(self, tensor: torch.Tensor, tag: str) -> CPUActivationHandle:
        if tensor.device.type == "cpu":
            return self.adopt_cpu(tensor, tag, original_device=tensor.device)
        handle = self.empty_cpu(tuple(tensor.shape), tensor.dtype, tensor.device, tag)
        non_blocking = bool(handle.tensor.is_pinned())
        with torch.no_grad():
            handle.tensor.copy_(tensor.detach(), non_blocking=non_blocking)
        if non_blocking and tensor.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
            self._pending_cpu_ready_events[int(handle.tensor.data_ptr())] = event
        nbytes = handle.nbytes
        self.stats.num_offloads += 1
        self.stats.offloaded_bytes += nbytes
        self.stats.offload_bytes_by_tag[tag] = self.stats.offload_bytes_by_tag.get(tag, 0) + nbytes
        return handle

    def adopt_cpu(
        self,
        tensor: torch.Tensor,
        tag: str,
        *,
        original_device: torch.device | None = None,
    ) -> CPUActivationHandle:
        if tensor.device.type != "cpu":
            raise ValueError(f"adopt_cpu expects a CPU tensor for {tag}, got {tensor.device}")
        original = torch.device("cpu" if original_device is None else original_device)
        source = tensor.detach()
        if not source.is_contiguous() or (self.pin_memory and torch.cuda.is_available() and not source.is_pinned()):
            handle = self.empty_cpu(tuple(source.shape), source.dtype, original, tag)
            with torch.no_grad():
                handle.tensor.copy_(source, non_blocking=False)
            return handle
        handle = CPUActivationHandle(
            tag=tag,
            tensor=source,
            original_device=original,
            original_dtype=source.dtype,
            original_shape=tuple(int(dim) for dim in source.shape),
        )
        self._mark_cpu_live(handle)
        if _GOV is not None:
            _GOV.on_offload(self, handle)
        return handle

    def wait_cpu_ready(self, handle: CPUActivationHandle | None) -> None:
        if handle is None:
            return
        if _GOV is not None:
            _GOV.ensure_local(handle)          # un-spill if DURABLE, before reading the D2H event
        event = self._pending_cpu_ready_events.pop(int(handle.tensor.data_ptr()), None)
        if event is not None:
            if torch.cuda.is_available():
                # never block the host (Megatron rule): downstream copies/kernels on the
                # current stream order themselves behind the producing D2H via this wait.
                torch.cuda.current_stream().wait_event(event)
            else:
                event.synchronize()

    def wait_cpu_ready_host(self, handle: CPUActivationHandle | None) -> None:
        """Host-blocking counterpart of wait_cpu_ready for HOST readers of
        handle.tensor (host memcpy pads, cpu silu, index_select rebuilds).
        wait_cpu_ready only orders the CURRENT CUDA STREAM behind the producing
        D2H — it never orders the host, so a host read straight after it can
        observe pre-copy bytes (proven in vitro 2026-07-16; fix_merged.md V1/V2).
        This is a true data dependency, not an added stall: the caller is about
        to read exactly the bytes the D2H produces. Synchronizes the pending D2H
        event when present; if the event was already consumed by an earlier
        device-side waiter (wait_cpu_ready pops it), falls back to synchronizing
        the original device's current stream — the D2H was enqueued there, so
        the wait is bounded by already-queued work, never by future work."""
        if handle is None:
            return
        if _GOV is not None:
            _GOV.ensure_local(handle)
        # get, NOT pop: leave the (now-complete) event in the map so a SECOND host
        # wait on the same handle re-synchronizes it instantly instead of falling
        # into the full-stream-drain fallback (dense hits x_cpu twice per layer:
        # gate then up). The entry is overwritten by the next fill of this buffer
        # (offload/record_cpu_ready key by data_ptr) and popped by device-side
        # wait_cpu_ready / release_cpu, so no staleness or leak.
        event = self._pending_cpu_ready_events.get(int(handle.tensor.data_ptr()))
        if event is not None:
            event.synchronize()
        elif torch.cuda.is_available() and handle.original_device.type == "cuda":
            torch.cuda.current_stream(handle.original_device).synchronize()

    def stage(self, handle: CPUActivationHandle, *, tag: str | None = None) -> torch.Tensor:
        self.wait_cpu_ready(handle)
        stage_tag = handle.tag if tag is None else tag
        shape = tuple(int(dim) for dim in handle.tensor.shape)
        key = (str(handle.original_device), handle.tensor.dtype, shape, stage_tag)
        stage = self._stage_cache.get(key)
        if stage is None:
            stage = torch.empty(shape, device=handle.original_device, dtype=handle.tensor.dtype)
            self._stage_cache[key] = stage
            self._stage_keys_by_ptr[int(stage.data_ptr())] = key
        if handle.tensor.is_pinned() and handle.original_device.type == "cuda":
            # H2D restage on a side stream. NB: the device-side ordering here is
            # exactly that of a plain non_blocking copy_ on the compute stream (the
            # pre-existing else-branch pattern below): wait_stream orders the copy
            # after all prior compute, wait_event orders all later compute after the
            # copy, and nothing is enqueued between — so this construction is pure
            # structural overhead over that equivalent, which the D4 A/B corroborates
            # (+0.08 s, null; fix_throughput.md). Real overlap would need BOTH the
            # D2H ready-event re-anchored on the side stream (side.wait_event(ready),
            # as the saved-tensor unpack siblings do — dropping wait_stream alone
            # would let the H2D read handle.tensor before its producing D2H finished)
            # AND a fresh per-call destination (the _stage_cache reuses buffers whose
            # previous consumer may still be in flight on the compute stream).
            from .attention_activation_offload import _h2d_restage_stream

            compute_stream = torch.cuda.current_stream(handle.original_device)
            side = _h2d_restage_stream(handle.original_device)
            side.wait_stream(compute_stream)
            with torch.no_grad(), torch.cuda.stream(side):
                stage.copy_(handle.tensor, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
            compute_stream.wait_event(done)
            stage.record_stream(side)
        else:
            with torch.no_grad():
                stage.copy_(handle.tensor, non_blocking=handle.tensor.is_pinned())
        self._mark_stage_live(stage, stage_tag)
        return stage

    def record_cpu_ready(self, handle: CPUActivationHandle) -> None:
        """Seal a handle whose tensor was filled by direct row writes rather than by
        offload(): register its cpu-ready event and mirror offload()'s byte accounting
        (direct row writes must call this once after the last chunk).

        The accounting matters as much as the event: empty_cpu() only counts
        num_cpu_allocs, so without this the row-blocked paths would move their bytes
        to CPU invisibly and report offloaded_bytes ~0 for the tags they dominate.
        Called once per handle, so the counters stay in step with the copies."""
        nbytes = handle.nbytes
        self.stats.num_offloads += 1
        self.stats.offloaded_bytes += nbytes
        self.stats.offload_bytes_by_tag[handle.tag] = self.stats.offload_bytes_by_tag.get(handle.tag, 0) + nbytes
        if torch.cuda.is_available() and handle.tensor.is_pinned():
            event = torch.cuda.Event()
            # Record on the ORIGINAL device's stream (matching offload(), which uses
            # current_stream(tensor.device)) — the row-write D2H copies were enqueued
            # there, and on a multi-device process the ambient current_stream() could
            # belong to a different device, ordering nothing.
            event.record(torch.cuda.current_stream(handle.original_device))
            self._pending_cpu_ready_events[int(handle.tensor.data_ptr())] = event

    def stage_rows(self, handle: CPUActivationHandle, start: int, end: int, *, tag: str | None = None) -> torch.Tensor:
        self.wait_cpu_ready(handle)
        row_start = int(start)
        row_end = int(end)
        if row_start < 0 or row_end < row_start or row_end > int(handle.tensor.shape[0]):
            raise ValueError(f"invalid row range [{row_start}, {row_end}) for {handle.tag} with shape {tuple(handle.tensor.shape)}")
        stage_tag = handle.tag if tag is None else tag
        shape = (row_end - row_start, *tuple(int(dim) for dim in handle.tensor.shape[1:]))
        key = (str(handle.original_device), handle.tensor.dtype, shape, stage_tag)
        stage = self._stage_cache.get(key)
        if stage is None:
            stage = torch.empty(shape, device=handle.original_device, dtype=handle.tensor.dtype)
            self._stage_cache[key] = stage
            self._stage_keys_by_ptr[int(stage.data_ptr())] = key
        source = handle.tensor.narrow(0, row_start, row_end - row_start)
        with torch.no_grad():
            stage.copy_(source, non_blocking=handle.tensor.is_pinned())
        self._mark_stage_live(stage, stage_tag)
        return stage

    def stage_concat_columns(
        self,
        left: CPUActivationHandle,
        right: CPUActivationHandle,
        *,
        tag: str,
    ) -> torch.Tensor:
        self.wait_cpu_ready(left)
        self.wait_cpu_ready(right)
        if left.tensor.dim() != 2 or right.tensor.dim() != 2:
            raise ValueError("stage_concat_columns expects 2D CPU handles")
        if int(left.tensor.shape[0]) != int(right.tensor.shape[0]):
            raise ValueError(f"row mismatch for {tag}: {tuple(left.tensor.shape)} vs {tuple(right.tensor.shape)}")
        if left.tensor.dtype != right.tensor.dtype:
            raise ValueError(f"dtype mismatch for {tag}: {left.tensor.dtype} vs {right.tensor.dtype}")
        device = left.original_device
        shape = (int(left.tensor.shape[0]), int(left.tensor.shape[1]) + int(right.tensor.shape[1]))
        key = (str(device), left.tensor.dtype, shape, tag)
        stage = self._stage_cache.get(key)
        if stage is None:
            stage = torch.empty(shape, device=device, dtype=left.tensor.dtype)
            self._stage_cache[key] = stage
            self._stage_keys_by_ptr[int(stage.data_ptr())] = key
        with torch.no_grad():
            stage[:, : int(left.tensor.shape[1])].copy_(left.tensor, non_blocking=left.tensor.is_pinned())
            stage[:, int(left.tensor.shape[1]) :].copy_(right.tensor, non_blocking=right.tensor.is_pinned())
        self._mark_stage_live(stage, tag)
        return stage

    def release_stage(self, tensor: torch.Tensor | None, *, drop_cache: bool = False) -> None:
        if tensor is None:
            return
        ptr = int(tensor.data_ptr())
        entry = self._active_stage_bytes.pop(ptr, None)
        if entry is None:
            return
        nbytes, _tag = entry
        self.stats.staged_bytes = max(0, self.stats.staged_bytes - nbytes)
        if drop_cache:
            key = self._stage_keys_by_ptr.pop(ptr, None)
            if key is not None:
                self._stage_cache.pop(key, None)

    def release_cpu(self, handle: CPUActivationHandle | None) -> int:
        if handle is None:
            return 0
        if _GOV is not None:
            _GOV.on_release(handle)            # BEFORE wait_cpu_ready: never fetch a spilled-then-freed handle
        self.wait_cpu_ready(handle)
        entry = self._active_cpu_bytes.pop(int(handle.tensor.data_ptr()), None)
        if entry is None:
            return 0
        nbytes, tag = entry
        self.stats.cpu_owned_bytes = max(0, self.stats.cpu_owned_bytes - nbytes)
        self.stats.cpu_bytes_by_tag[tag] = max(0, self.stats.cpu_bytes_by_tag.get(tag, 0) - nbytes)
        _return_cpu(handle.tensor, pin_memory=self.pin_memory)
        return nbytes

    def seal(self, *handles: "CPUActivationHandle | None") -> None:
        """Mark handles spill-eligible (governor); one call at the end of a Function.forward. Inert
        without the governor. Handles failing io_ready/min-size stay resident (pressure-only)."""
        if _GOV is not None:
            _GOV.on_seal(self, [h for h in handles if h is not None])

    def _pop_active(self, handle: CPUActivationHandle) -> None:
        """Governor-only: drop cpu-owned accounting for the handle's CURRENT data_ptr WITHOUT pool
        return, and drop the stale D2H event for that ptr (mirrors release_cpu's stat decrements)."""
        ptr = int(handle.tensor.data_ptr())
        self._pending_cpu_ready_events.pop(ptr, None)
        entry = self._active_cpu_bytes.pop(ptr, None)
        if entry is None:
            return
        nbytes, tag = entry
        self.stats.cpu_owned_bytes = max(0, self.stats.cpu_owned_bytes - nbytes)
        self.stats.cpu_bytes_by_tag[tag] = max(0, self.stats.cpu_bytes_by_tag.get(tag, 0) - nbytes)

    def snapshot(self) -> dict[str, Any]:
        return self.stats.as_dict()

    def _mark_cpu_live(self, handle: CPUActivationHandle) -> None:
        ptr = int(handle.tensor.data_ptr())
        nbytes = handle.nbytes
        previous = self._active_cpu_bytes.get(ptr)
        if previous is not None:
            self.stats.cpu_owned_bytes = max(0, self.stats.cpu_owned_bytes - previous[0])
            self.stats.cpu_bytes_by_tag[previous[1]] = max(0, self.stats.cpu_bytes_by_tag.get(previous[1], 0) - previous[0])
        self._active_cpu_bytes[ptr] = (nbytes, handle.tag)
        self.stats.cpu_owned_bytes += nbytes
        self.stats.cpu_bytes_by_tag[handle.tag] = self.stats.cpu_bytes_by_tag.get(handle.tag, 0) + nbytes
        self.stats.cpu_peak_bytes_live = max(self.stats.cpu_peak_bytes_live, self.stats.cpu_owned_bytes)
        self.stats.cpu_peak_by_tag[handle.tag] = max(
            self.stats.cpu_peak_by_tag.get(handle.tag, 0),
            self.stats.cpu_bytes_by_tag[handle.tag],
        )

    def _mark_stage_live(self, tensor: torch.Tensor, tag: str) -> None:
        ptr = int(tensor.data_ptr())
        existing = self._active_stage_bytes.get(ptr)
        if existing is not None:
            self.stats.staged_bytes = max(0, self.stats.staged_bytes - existing[0])
        nbytes = _tensor_nbytes(tensor)
        self._active_stage_bytes[ptr] = (nbytes, tag)
        self.stats.num_stages += 1
        self.stats.staged_bytes += nbytes
        self.stats.max_stage_bytes_live = max(self.stats.max_stage_bytes_live, self.stats.staged_bytes)
        self.stats.stage_bytes_by_tag[tag] = self.stats.stage_bytes_by_tag.get(tag, 0) + nbytes
        self.stats.stage_peak_by_tag[tag] = max(self.stats.stage_peak_by_tag.get(tag, 0), nbytes)


# Bind the actnvme governor AFTER all pool functions + the manager are defined, so act_spill_governor
# (which lazy-imports _alloc_cpu/_return_cpu) resolves without an import cycle. None unless
# ASYM_NVME_ROLES contains "activation" (rule 7): no deepspeed import, no store, no hooks.
from .act_spill_governor import get_act_spill_governor  # noqa: E402

_GOV = get_act_spill_governor()
