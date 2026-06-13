from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


_CPU_BUFFER_POOL: dict[tuple[torch.dtype, tuple[int, ...], bool], list[torch.Tensor]] = {}


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _alloc_cpu(shape: tuple[int, ...], dtype: torch.dtype, *, pin_memory: bool) -> torch.Tensor:
    pinned = bool(pin_memory and torch.cuda.is_available())
    key = (dtype, tuple(int(dim) for dim in shape), pinned)
    pool = _CPU_BUFFER_POOL.get(key)
    if pool:
        return pool.pop()
    try:
        return torch.empty(key[1], device="cpu", dtype=dtype, pin_memory=pinned)
    except RuntimeError:
        return torch.empty(key[1], device="cpu", dtype=dtype)


def _return_cpu(tensor: torch.Tensor, *, pin_memory: bool) -> None:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        return
    pinned = bool(pin_memory and torch.cuda.is_available() and tensor.is_pinned())
    key = (tensor.dtype, tuple(int(dim) for dim in tensor.shape), pinned)
    _CPU_BUFFER_POOL.setdefault(key, []).append(tensor.detach())


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
    staged_bytes: int = 0
    max_stage_bytes_live: int = 0
    num_offloads: int = 0
    num_cpu_allocs: int = 0
    num_stages: int = 0
    offload_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    cpu_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    stage_bytes_by_tag: dict[str, int] = field(default_factory=dict)
    stage_peak_by_tag: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "offloaded_bytes": self.offloaded_bytes,
            "cpu_owned_bytes": self.cpu_owned_bytes,
            "staged_bytes": self.staged_bytes,
            "max_stage_bytes_live": self.max_stage_bytes_live,
            "num_offloads": self.num_offloads,
            "num_cpu_allocs": self.num_cpu_allocs,
            "num_stages": self.num_stages,
            "offload_bytes_by_tag": dict(self.offload_bytes_by_tag),
            "cpu_bytes_by_tag": dict(self.cpu_bytes_by_tag),
            "stage_bytes_by_tag": dict(self.stage_bytes_by_tag),
            "stage_peak_by_tag": dict(self.stage_peak_by_tag),
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
        return handle

    def offload(self, tensor: torch.Tensor, tag: str) -> CPUActivationHandle:
        if tensor.device.type == "cpu":
            return self.adopt_cpu(tensor, tag, original_device=tensor.device)
        handle = self.empty_cpu(tuple(tensor.shape), tensor.dtype, tensor.device, tag)
        with torch.no_grad():
            handle.tensor.copy_(tensor.detach(), non_blocking=handle.tensor.is_pinned())
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
        return handle

    def stage(self, handle: CPUActivationHandle, *, tag: str | None = None) -> torch.Tensor:
        stage_tag = handle.tag if tag is None else tag
        shape = tuple(int(dim) for dim in handle.tensor.shape)
        key = (str(handle.original_device), handle.tensor.dtype, shape, stage_tag)
        stage = self._stage_cache.get(key)
        if stage is None:
            stage = torch.empty(shape, device=handle.original_device, dtype=handle.tensor.dtype)
            self._stage_cache[key] = stage
            self._stage_keys_by_ptr[int(stage.data_ptr())] = key
        with torch.no_grad():
            stage.copy_(handle.tensor, non_blocking=handle.tensor.is_pinned())
        self._mark_stage_live(stage, stage_tag)
        return stage

    def stage_concat_columns(
        self,
        left: CPUActivationHandle,
        right: CPUActivationHandle,
        *,
        tag: str,
    ) -> torch.Tensor:
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

    def release_cpu(self, handle: CPUActivationHandle | None) -> None:
        if handle is None:
            return
        entry = self._active_cpu_bytes.pop(int(handle.tensor.data_ptr()), None)
        if entry is None:
            return
        nbytes, tag = entry
        self.stats.cpu_owned_bytes = max(0, self.stats.cpu_owned_bytes - nbytes)
        self.stats.cpu_bytes_by_tag[tag] = max(0, self.stats.cpu_bytes_by_tag.get(tag, 0) - nbytes)
        _return_cpu(handle.tensor, pin_memory=self.pin_memory)

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
