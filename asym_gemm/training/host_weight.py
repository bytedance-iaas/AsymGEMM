"""CPU-resident frozen weight handles for AsymGEMM training wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Any

import torch


@dataclass(frozen=True)
class HostWeightMetadata:
    """Best-effort placement metadata for a CPU-resident weight tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    device: str
    is_pinned: bool
    data_ptr: int
    numa_node: int | None
    cpu_socket: int | None
    can_map_host_memory: bool | None
    pointer_attributes: dict[str, Any]

    @property
    def pinned(self) -> bool:
        return self.is_pinned


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return _tensor_nbytes(tensor)


_profile_recorder: Any | None = None


def set_profile_recorder(recorder: Any | None) -> Any | None:
    global _profile_recorder
    previous = _profile_recorder
    _profile_recorder = recorder
    return previous


def _record_profile_event(event_name: str, **payload: Any) -> None:
    if _profile_recorder is None:
        return
    _profile_recorder(event_name, payload)


def _cuda_can_map_host_memory(device: int = 0) -> bool | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return None
    value = getattr(props, "can_map_host_memory", None)
    return None if value is None else bool(value)


def _mapping_start_for_address(address: int) -> int | None:
    try:
        with open("/proc/self/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                range_text = line.split(maxsplit=1)[0]
                start_text, end_text = range_text.split("-", maxsplit=1)
                start = int(start_text, 16)
                end = int(end_text, 16)
                if start <= address < end:
                    return start
    except OSError:
        return None
    return None


def _numa_node_for_address(address: int) -> int | None:
    mapping_start = _mapping_start_for_address(address)
    if mapping_start is None:
        return None

    prefix = f"{mapping_start:x} "
    try:
        with open("/proc/self/numa_maps", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(prefix):
                    continue
                node_pages: list[tuple[int, int]] = []
                for token in line.split():
                    match = re.fullmatch(r"N(\d+)=(\d+)", token)
                    if match:
                        node_pages.append((int(match.group(1)), int(match.group(2))))
                if not node_pages:
                    return None
                node_pages.sort(key=lambda item: item[1], reverse=True)
                return node_pages[0][0]
    except OSError:
        return None
    return None


def _first_cpu_from_cpulist(cpulist: str) -> int | None:
    first = cpulist.strip().split(",", maxsplit=1)[0]
    if not first:
        return None
    first = first.split("-", maxsplit=1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def _cpu_socket_for_numa_node(numa_node: int | None) -> int | None:
    if numa_node is None:
        return None
    cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
    try:
        with open(cpulist_path, "r", encoding="utf-8") as handle:
            cpu = _first_cpu_from_cpulist(handle.read())
    except OSError:
        return None
    if cpu is None:
        return None

    socket_path = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
    try:
        with open(socket_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _make_metadata(tensor: torch.Tensor, pin_error: str | None = None) -> HostWeightMetadata:
    data_ptr = tensor.data_ptr()
    numa_node = _numa_node_for_address(data_ptr)
    pointer_attributes: dict[str, Any] = {
        "kind": "cpu",
        "is_pinned": bool(tensor.is_pinned()),
    }
    if pin_error is not None:
        pointer_attributes["pin_memory_error"] = pin_error

    return HostWeightMetadata(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        nbytes=_tensor_nbytes(tensor),
        device=str(tensor.device),
        is_pinned=bool(tensor.is_pinned()),
        data_ptr=data_ptr,
        numa_node=numa_node,
        cpu_socket=_cpu_socket_for_numa_node(numa_node),
        can_map_host_memory=_cuda_can_map_host_memory(),
        pointer_attributes=pointer_attributes,
    )


def _extract_requested_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.device | None:
    if "device" in kwargs and kwargs["device"] is not None:
        return torch.device(kwargs["device"])
    for arg in args:
        if isinstance(arg, torch.device):
            return arg
        if isinstance(arg, str):
            try:
                return torch.device(arg)
            except (TypeError, RuntimeError):
                continue
    return None


class HostWeight:
    """Frozen 2D weight stored in CPU memory.

    The object intentionally is not a ``torch.nn.Parameter`` or module buffer, so
    ``nn.Module.to("cuda")`` cannot migrate it into HBM by accident.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        pin_memory: bool = True,
        clone: bool = True,
        require_2d: bool = True,
        name: str | None = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"HostWeight expects a torch.Tensor, got {type(tensor)!r}")
        if require_2d and tensor.dim() != 2:
            raise ValueError(f"HostWeight expects a 2D tensor, got shape {tuple(tensor.shape)}")

        init_start = time.perf_counter()
        source_device = str(tensor.device)
        source_nbytes = _tensor_nbytes(tensor)

        copy_start = time.perf_counter()
        cpu_tensor = tensor.detach()
        if cpu_tensor.device.type != "cpu":
            cpu_tensor = cpu_tensor.to(device="cpu", non_blocking=False)
        copy_seconds = time.perf_counter() - copy_start

        clone_start = time.perf_counter()
        if clone:
            cpu_tensor = cpu_tensor.clone(memory_format=torch.contiguous_format)
        elif not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        clone_seconds = time.perf_counter() - clone_start
        cpu_tensor.requires_grad_(False)

        pin_error: str | None = None
        pin_seconds = 0.0
        if pin_memory and torch.cuda.is_available() and not cpu_tensor.is_pinned():
            pin_start = time.perf_counter()
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError as exc:
                pin_error = str(exc)
            pin_seconds = time.perf_counter() - pin_start

        self._tensor = cpu_tensor
        self._transpose: torch.Tensor | None = None
        self._name = name
        self._pin_transpose = pin_memory
        self._metadata = _make_metadata(cpu_tensor, pin_error=pin_error)
        _record_profile_event(
            "host_weight_init",
            name=name,
            source_device=source_device,
            source_nbytes=source_nbytes,
            nbytes=self.nbytes,
            pinned=bool(cpu_tensor.is_pinned()),
            cpu_copy_seconds=copy_seconds,
            clone_seconds=clone_seconds,
            pin_seconds=pin_seconds,
            total_seconds=time.perf_counter() - init_start,
            pin_error=pin_error,
        )

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        pin_memory: bool = True,
        clone: bool = True,
        name: str | None = None,
    ) -> "HostWeight":
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return cls(tensor, pin_memory=pin_memory, clone=clone, name=name)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def metadata(self) -> HostWeightMetadata:
        return self._metadata

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def nbytes(self) -> int:
        return self._metadata.nbytes

    @property
    def is_pinned(self) -> bool:
        return self._metadata.is_pinned

    @property
    def out_features(self) -> int:
        return int(self._tensor.shape[0])

    @property
    def in_features(self) -> int:
        return int(self._tensor.shape[1])

    @property
    def grad(self) -> None:
        return self._tensor.grad

    @property
    def weight(self) -> torch.Tensor:
        return self._tensor

    @property
    def weight_t(self) -> torch.Tensor:
        return self.transposed_tensor()

    @property
    def weight_nbytes(self) -> int:
        return self.nbytes

    @property
    def pinned_cpu_bytes(self) -> int:
        total = self.nbytes if self.is_pinned else 0
        if self._transpose is not None and self._transpose.is_pinned():
            total += tensor_nbytes(self._transpose)
        return total

    def grouped_nt_tensor(self) -> torch.Tensor:
        return self._tensor.unsqueeze(0)

    def transposed_tensor(self) -> torch.Tensor:
        if self._transpose is None:
            total_start = time.perf_counter()
            transpose_start = time.perf_counter()
            transpose = self._tensor.t().contiguous()
            transpose_seconds = time.perf_counter() - transpose_start
            pin_seconds = 0.0
            pin_error: str | None = None
            if self._pin_transpose and torch.cuda.is_available() and not transpose.is_pinned():
                pin_start = time.perf_counter()
                try:
                    transpose = transpose.pin_memory()
                except RuntimeError as exc:
                    pin_error = str(exc)
                pin_seconds = time.perf_counter() - pin_start
            transpose.requires_grad_(False)
            self._transpose = transpose
            _record_profile_event(
                "host_weight_transpose",
                name=self._name,
                nbytes=tensor_nbytes(transpose),
                pinned=bool(transpose.is_pinned()),
                transpose_seconds=transpose_seconds,
                pin_seconds=pin_seconds,
                total_seconds=time.perf_counter() - total_start,
                pin_error=pin_error,
            )
        return self._transpose

    def transposed_grouped_nt_tensor(self) -> torch.Tensor:
        return self.transposed_tensor().unsqueeze(0)

    def to(self, *args: Any, **kwargs: Any) -> "HostWeight":
        requested_device = _extract_requested_device(args, kwargs)
        if requested_device is not None and requested_device.type == "cuda":
            raise RuntimeError("HostWeight is CPU-resident and cannot be moved to CUDA")
        if not args and not kwargs:
            return self
        if requested_device is None or requested_device.type == "cpu":
            dtype = kwargs.get("dtype", None)
            if dtype is None:
                for arg in args:
                    if isinstance(arg, torch.dtype):
                        dtype = arg
                        break
            if dtype is None or dtype == self.dtype:
                return self
        raise RuntimeError("Construct a new HostWeight to change dtype or memory placement")

    def cuda(self, *args: Any, **kwargs: Any) -> "HostWeight":
        raise RuntimeError("HostWeight is CPU-resident and cannot be moved to CUDA")

    def pin_memory(self) -> "HostWeight":
        if self.is_pinned:
            return self
        return HostWeight(self._tensor, pin_memory=True, clone=True, name=self._name)

    def __repr__(self) -> str:
        name = "" if self._name is None else f"name={self._name!r}, "
        return (
            f"HostWeight({name}shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"pinned={self.is_pinned}, nbytes={self.nbytes})"
        )


__all__ = ["HostWeight", "HostWeightMetadata", "set_profile_recorder", "tensor_nbytes"]
