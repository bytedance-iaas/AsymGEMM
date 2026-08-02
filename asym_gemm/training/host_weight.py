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


from .exact_pinned import exact_pinned_enabled, register_inplace


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
        exclusively_owned = False
        if cpu_tensor.device.type != "cpu":
            cpu_tensor = cpu_tensor.to(device="cpu", non_blocking=False)
            exclusively_owned = True
        copy_seconds = time.perf_counter() - copy_start

        clone_start = time.perf_counter()
        if clone:
            cpu_tensor = cpu_tensor.clone(memory_format=torch.contiguous_format)
            exclusively_owned = True
        elif not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
            exclusively_owned = True
        clone_seconds = time.perf_counter() - clone_start
        cpu_tensor.requires_grad_(False)

        pin_error: str | None = None
        pin_seconds = 0.0
        self._fabric_bank = False
        if pin_memory and torch.cuda.is_available() and not cpu_tensor.is_pinned():
            from .shared_fabric import fabric_enabled, get_fabric

            fabric_view = None
            if fabric_enabled():
                # asym_ep2 shared fabric (fix_gb200_ep.md S1 DELTA 2): the bank lives ONCE
                # in /dev/shm for ALL ranks; pinning happens collectively at seal() via one
                # cudaHostRegister of the whole used range. is_pinned() stays False until
                # seal — no GEMM may stream a fabric bank pre-seal (seal runs pre-train).
                # Post-seal latecomers get None back and take the private pin path below.
                pin_start = time.perf_counter()
                fabric_view = get_fabric().get_or_create(name or "bank", cpu_tensor)
                pin_seconds = time.perf_counter() - pin_start
            if fabric_view is not None:
                cpu_tensor = fabric_view
                self._fabric_bank = True
            else:
                pin_start = time.perf_counter()
                exact_err: str | None = "disabled"
                if exact_pinned_enabled() and exclusively_owned:
                    # capacity fix 2026-07-25: page-lock the exact-size clone in
                    # place instead of pin_memory()'s pow2-bucketed copy (see
                    # exact_pinned.py). Failure falls through to the stock path.
                    # 39-merge guard (mrg38on incident, 2026-08-01): ONLY when this
                    # HostWeight exclusively owns the buffer (we cloned/copied it).
                    # clone=False aliases of loader-owned params (qwen3 bank path)
                    # died mid-lifecycle when registered in place — copies stayed
                    # valid at registration (491/491 probes OK) but the aliased
                    # storage became uncopyable (sync AND async invalid-argument)
                    # by first forward. pin_memory() copy = stock, release-safe.
                    exact_err = register_inplace(cpu_tensor)
                if exact_err is not None:
                    try:
                        cpu_tensor = cpu_tensor.pin_memory()
                    except RuntimeError as exc:
                        pin_error = str(exc)
                pin_seconds = time.perf_counter() - pin_start

        self._tensor = cpu_tensor
        self._name = name
        self._metadata = _make_metadata(cpu_tensor, pin_error=pin_error)
        # NVMe pager hooks (asym_cpuadamwds_panvme, agent/impls/nvme_offload_impl.md Stage 7).
        # None ⇒ every property below takes today's exact path — off is byte-identical.
        self._pager: Any | None = None
        self._pager_key: str | None = None
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
    def is_paged(self) -> bool:
        """True when the home lives on NVMe behind the base-weight pager. Observers (profilers,
        reporters) MUST branch on this and use metadata — reading ``.tensor``/``.weight`` on a
        paged HostWeight fetches from NVMe (measured: per-step full-model profiler walks
        re-streamed ~64 GiB each, dominating the panvme step-time regression)."""
        return self._pager is not None

    @property
    def tensor(self) -> torch.Tensor:
        if self._pager is not None:                       # COMPUTE read: fetch/ensure resident
            return self._pager.touch(self._pager_key)
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
        if self._pager is not None:                       # reporting/predicate: NEVER fetches
            return torch.Size(self._metadata.shape)
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        if self._pager is not None:
            return self._metadata.dtype
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        if self._pager is not None:
            return torch.device(self._metadata.device)    # metadata stores device as str
        return self._tensor.device

    @property
    def nbytes(self) -> int:
        return self._metadata.nbytes

    @property
    def is_pinned(self) -> bool:
        return self._metadata.is_pinned

    @property
    def out_features(self) -> int:
        if self._pager is not None:
            return int(self._metadata.shape[0])
        return int(self._tensor.shape[0])

    @property
    def in_features(self) -> int:
        if self._pager is not None:
            return int(self._metadata.shape[1])
        return int(self._tensor.shape[1])

    @property
    def grad(self) -> None:
        if self._pager is not None:                       # frozen: no grad is ever set
            return None
        return self._tensor.grad

    @property
    def weight(self) -> torch.Tensor:
        if self._pager is not None:                       # COMPUTE read: fetch/ensure resident
            return self._pager.touch(self._pager_key)
        return self._tensor

    @property
    def weight_nbytes(self) -> int:
        return self.nbytes

    @property
    def pinned_cpu_bytes(self) -> int:
        if self._pager is not None:
            return 0                                      # home freed; pager cache reports separately
        return self.nbytes if self.is_pinned else 0

    def grouped_nt_tensor(self) -> torch.Tensor:
        if self._pager is not None:                       # compute path: fetch is correct
            return self._pager.touch(self._pager_key).unsqueeze(0)
        return self._tensor.unsqueeze(0)

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
        if self._pager is not None:                       # pager buffers are pinned already
            return self
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
