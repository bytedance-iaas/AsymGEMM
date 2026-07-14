from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import types
from typing import Any

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from .attention_activation_offload import _h2d_restage_stream
from .decoder_activation_offload import _async_unpack_enabled


_DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES = 1 * 1024**2
_DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_SAVED_TENSOR_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _linear_attention_saved_tensor_min_bytes() -> int:
    raw = os.environ.get("ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_MIN_BYTES")
    if raw is None or raw == "":
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES


def _linear_attention_saved_tensor_dtypes() -> frozenset[torch.dtype]:
    raw = os.environ.get("ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_DTYPES")
    if raw is None or raw.strip() == "":
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES
    allowed: set[torch.dtype] = set()
    for token in raw.replace(";", ",").split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key in {"all", "*"}:
            return _DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES
        dtype = _SAVED_TENSOR_DTYPE_ALIASES.get(key)
        if dtype is not None:
            allowed.add(dtype)
    return frozenset(allowed)


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _empty_strided_cpu_like(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    shape = tuple(int(dim) for dim in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    try:
        return torch.empty_strided(
            shape,
            stride,
            device="cpu",
            dtype=tensor.dtype,
            pin_memory=bool(pin_memory and torch.cuda.is_available()),
        )
    except RuntimeError:
        return torch.empty_strided(shape, stride, device="cpu", dtype=tensor.dtype)


@dataclass
class _SavedTensorOffloadHandle:
    tensor: torch.Tensor
    original_device: torch.device
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]
    original_stride: tuple[int, ...]
    nbytes: int
    tag: str
    ready_event: torch.cuda.Event | None = None
    released: bool = False


class LinearAttentionSavedTensorOffloadWrapper:
    """Forward wrapper that offloads large Qwen3.5 linear-attention saved tensors to CPU."""

    def __init__(
        self,
        module: nn.Module,
        *,
        pin_memory: bool = True,
        min_bytes: int | None = None,
        require_grad: bool | None = None,
        allowed_dtypes: set[torch.dtype] | frozenset[torch.dtype] | None = None,
    ) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.pin_memory = bool(pin_memory)
        self.min_bytes = _linear_attention_saved_tensor_min_bytes() if min_bytes is None else max(0, int(min_bytes))
        self.allowed_dtypes = (
            _linear_attention_saved_tensor_dtypes()
            if allowed_dtypes is None
            else frozenset(dtype for dtype in allowed_dtypes if isinstance(dtype, torch.dtype))
        )
        self.require_grad = (
            _env_bool("ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD", False)
            if require_grad is None
            else bool(require_grad)
        )
        self.calls = 0
        self.offload_calls = 0
        self.unpack_calls = 0
        self.skipped_tensors = 0
        self.skipped_bytes = 0
        self.offloaded_bytes = 0
        self.cpu_owned_bytes = 0
        self.cpu_peak_bytes_live = 0
        self.staged_bytes = 0
        self.max_stage_bytes_live = 0
        self.offload_bytes_by_tag: dict[str, int] = {}
        self.cpu_bytes_by_tag: dict[str, int] = {}
        self.cpu_peak_by_tag: dict[str, int] = {}
        self.stage_bytes_by_tag: dict[str, int] = {}
        self.stage_peak_by_tag: dict[str, int] = {}
        self.dtype_counts: dict[str, int] = {}
        self.shape_counts: dict[str, int] = {}
        self._sync_module_stats()

    def install(self) -> None:
        setattr(self.module, "_asym_linear_attention_saved_tensor_offload_wrapper", self)
        self.module.forward = types.MethodType(_linear_attention_saved_tensor_offload_forward, self.module)  # type: ignore[method-assign]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        with saved_tensors_hooks(self._pack, self._unpack):
            return self.original_forward(*args, **kwargs)

    def _should_offload(self, tensor: torch.Tensor) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type != "cuda":
            return False
        if tensor.dtype not in self.allowed_dtypes:
            return False
        nbytes = int(tensor.numel() * tensor.element_size())
        if self.require_grad and not tensor.requires_grad:
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        if nbytes < self.min_bytes:
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        # Skip ONLY real parameters (leaf+grad weights). qwen3.5's fla delta-net
        # emits leaf+requires_grad *activations* (param=False) — those must offload.
        if isinstance(tensor, torch.nn.Parameter):
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        return True

    def _tag_for(self, tensor: torch.Tensor) -> str:
        dtype_name = str(tensor.dtype).replace("torch.", "")
        shape = "x".join(str(int(dim)) for dim in tensor.shape) or "scalar"
        return f"linear_attention.saved.{dtype_name}.{shape}"

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _SavedTensorOffloadHandle:
        if not self._should_offload(tensor):
            return tensor
        cpu = _empty_strided_cpu_like(tensor, pin_memory=self.pin_memory)
        non_blocking = bool(cpu.is_pinned())
        with torch.no_grad():
            cpu.copy_(tensor.detach(), non_blocking=non_blocking)
        ready_event = None
        if non_blocking:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(tensor.device))
        nbytes = _tensor_storage_nbytes(cpu)
        tag = self._tag_for(tensor)
        self.offload_calls += 1
        self.offloaded_bytes += nbytes
        self.cpu_owned_bytes += nbytes
        self.cpu_peak_bytes_live = max(self.cpu_peak_bytes_live, self.cpu_owned_bytes)
        self.offload_bytes_by_tag[tag] = self.offload_bytes_by_tag.get(tag, 0) + nbytes
        self.cpu_bytes_by_tag[tag] = self.cpu_bytes_by_tag.get(tag, 0) + nbytes
        self.cpu_peak_by_tag[tag] = max(self.cpu_peak_by_tag.get(tag, 0), self.cpu_bytes_by_tag[tag])
        dtype_key = str(tensor.dtype).replace("torch.", "")
        self.dtype_counts[dtype_key] = self.dtype_counts.get(dtype_key, 0) + 1
        shape_key = f"{dtype_key}:{tuple(int(dim) for dim in tensor.shape)}"
        self.shape_counts[shape_key] = self.shape_counts.get(shape_key, 0) + 1
        self._sync_module_stats()
        return _SavedTensorOffloadHandle(
            tensor=cpu,
            original_device=torch.device(tensor.device),
            original_dtype=tensor.dtype,
            original_shape=tuple(int(dim) for dim in tensor.shape),
            original_stride=tuple(int(value) for value in tensor.stride()),
            nbytes=nbytes,
            tag=tag,
            ready_event=ready_event,
        )

    def _unpack(self, packed: torch.Tensor | _SavedTensorOffloadHandle) -> torch.Tensor:
        if not isinstance(packed, _SavedTensorOffloadHandle):
            return packed
        staged = torch.empty_strided(
            packed.original_shape,
            packed.original_stride,
            device=packed.original_device,
            dtype=packed.original_dtype,
        )
        if _async_unpack_enabled() and packed.tensor.is_pinned():
            compute_stream = torch.cuda.current_stream(packed.original_device)
            side = _h2d_restage_stream(packed.original_device)
            if packed.ready_event is not None:
                side.wait_event(packed.ready_event)
            side.wait_stream(compute_stream)  # staged alloc ordering
            with torch.no_grad(), torch.cuda.stream(side):
                staged.copy_(packed.tensor, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
            compute_stream.wait_event(done)
            staged.record_stream(side)
            # keep the cpu buffer alive until the staged tensor dies (async copy source)
            staged._asym_restage_keepalive = packed.tensor  # type: ignore[attr-defined]
        else:
            if packed.ready_event is not None:
                packed.ready_event.synchronize()
            with torch.no_grad():
                staged.copy_(packed.tensor, non_blocking=False)
        self.unpack_calls += 1
        self.staged_bytes += packed.nbytes
        self.max_stage_bytes_live = max(self.max_stage_bytes_live, self.staged_bytes)
        self.stage_bytes_by_tag[packed.tag] = self.stage_bytes_by_tag.get(packed.tag, 0) + packed.nbytes
        self.stage_peak_by_tag[packed.tag] = max(self.stage_peak_by_tag.get(packed.tag, 0), packed.nbytes)
        self.staged_bytes = max(0, self.staged_bytes - packed.nbytes)
        if not packed.released:
            packed.released = True
            self.cpu_owned_bytes = max(0, self.cpu_owned_bytes - packed.nbytes)
            self.cpu_bytes_by_tag[packed.tag] = max(0, self.cpu_bytes_by_tag.get(packed.tag, 0) - packed.nbytes)
        self._sync_module_stats()
        return staged

    def snapshot(self) -> dict[str, Any]:
        return {
            "linear_attention_saved_tensor_offload": True,
            "calls": self.calls,
            "min_bytes": self.min_bytes,
            "require_grad": self.require_grad,
            "allowed_dtypes": [str(dtype).replace("torch.", "") for dtype in sorted(self.allowed_dtypes, key=str)],
            "offloaded_bytes": self.offloaded_bytes,
            "cpu_owned_bytes": self.cpu_owned_bytes,
            "cpu_live_bytes": self.cpu_owned_bytes,
            "cpu_peak_bytes_live": self.cpu_peak_bytes_live,
            "staged_bytes": self.staged_bytes,
            "max_stage_bytes_live": self.max_stage_bytes_live,
            "num_offloads": self.offload_calls,
            "num_cpu_allocs": self.offload_calls,
            "num_stages": self.unpack_calls,
            "skipped_tensors": self.skipped_tensors,
            "skipped_bytes": self.skipped_bytes,
            "offload_bytes_by_tag": dict(self.offload_bytes_by_tag),
            "cpu_bytes_by_tag": dict(self.cpu_bytes_by_tag),
            "cpu_peak_by_tag": dict(self.cpu_peak_by_tag),
            "stage_bytes_by_tag": dict(self.stage_bytes_by_tag),
            "stage_peak_by_tag": dict(self.stage_peak_by_tag),
            "dtype_counts": dict(self.dtype_counts),
            "shape_counts": dict(self.shape_counts),
            "pre_final_cleanup_cpu_owned_bytes": self.cpu_owned_bytes,
            "final_cleanup_released_bytes": 0,
        }

    def _sync_module_stats(self) -> None:
        setattr(self.module, "_last_activation_offload_stats", self.snapshot())


def _linear_attention_saved_tensor_offload_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_linear_attention_saved_tensor_offload_wrapper", None)
    if not isinstance(wrapper, LinearAttentionSavedTensorOffloadWrapper):
        raise RuntimeError("linear-attention saved-tensor offload wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_linear_attention_saved_tensor_offload(
    module: nn.Module,
    *,
    min_bytes: int | None = None,
    require_grad: bool | None = None,
    allowed_dtypes: set[torch.dtype] | frozenset[torch.dtype] | None = None,
) -> LinearAttentionSavedTensorOffloadWrapper:
    existing = getattr(module, "_asym_linear_attention_saved_tensor_offload_wrapper", None)
    if isinstance(existing, LinearAttentionSavedTensorOffloadWrapper):
        return existing
    wrapper = LinearAttentionSavedTensorOffloadWrapper(
        module,
        min_bytes=min_bytes,
        require_grad=require_grad,
        allowed_dtypes=allowed_dtypes,
    )
    wrapper.install()
    return wrapper


def is_linear_attention_saved_tensor_offload_wrapper(module: nn.Module) -> bool:
    return isinstance(
        getattr(module, "_asym_linear_attention_saved_tensor_offload_wrapper", None),
        LinearAttentionSavedTensorOffloadWrapper,
    )


def linear_attention_saved_tensor_offload_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, module in model.named_modules() if name and is_linear_attention_saved_tensor_offload_wrapper(module)
    )


__all__ = [
    "LinearAttentionSavedTensorOffloadWrapper",
    "install_linear_attention_saved_tensor_offload",
    "is_linear_attention_saved_tensor_offload_wrapper",
    "linear_attention_saved_tensor_offload_module_names",
]
