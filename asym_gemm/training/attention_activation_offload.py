from __future__ import annotations

from dataclasses import dataclass
import os
import types
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, _check_backend, asym_bf16_cpu_right_matmul
from .host_weight import HostWeight
from .lora import _reset_lora_weights, grouped_expert_lora_cpu_left, normalize_lora_dtype


_SINGLE_GROUP_METADATA_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_QKV_SHARE_ROLES = frozenset({"q_proj", "k_proj", "v_proj"})
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


def _align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _flatten_last_dim(x: torch.Tensor, in_features: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.shape[-1] != int(in_features):
        raise ValueError(f"expected input last dim {int(in_features)}, got {x.shape[-1]}")
    input_shape = tuple(int(dim) for dim in x.shape)
    return x.reshape(-1, int(in_features)).contiguous(), input_shape


def _restore_last_dim(x: torch.Tensor, input_shape: tuple[int, ...], out_features: int) -> torch.Tensor:
    return x.reshape(*input_shape[:-1], int(out_features))


def _empty_cpu_like_rows(source: torch.Tensor, rows: int) -> torch.Tensor:
    shape = (int(rows), int(source.shape[1]))
    try:
        return torch.empty(shape, device="cpu", dtype=source.dtype, pin_memory=bool(source.is_pinned()))
    except RuntimeError:
        return torch.empty(shape, device="cpu", dtype=source.dtype)


def _pad_cpu_rows_to(source: torch.Tensor, rows: int) -> torch.Tensor:
    rows = int(rows)
    if source.dim() != 2:
        raise ValueError(f"CPU row padding expects a 2D tensor, got {tuple(source.shape)}")
    if source.device.type != "cpu":
        raise ValueError(f"CPU row padding expects a CPU tensor, got {source.device}")
    if int(source.shape[0]) == rows:
        return source.contiguous()
    if int(source.shape[0]) > rows:
        raise ValueError(f"cannot pad {int(source.shape[0])} rows down to {rows}")
    padded = _empty_cpu_like_rows(source, rows)
    with torch.no_grad():
        padded[: int(source.shape[0])].copy_(source, non_blocking=False)
        if rows > int(source.shape[0]):
            padded[int(source.shape[0]) :].zero_()
    return padded.contiguous()


def _pad_hbm_rows_to(source: torch.Tensor, rows: int) -> torch.Tensor:
    rows = int(rows)
    if source.dim() != 2:
        raise ValueError(f"HBM row padding expects a 2D tensor, got {tuple(source.shape)}")
    if int(source.shape[0]) == rows:
        return source.contiguous()
    if int(source.shape[0]) > rows:
        raise ValueError(f"cannot pad {int(source.shape[0])} rows down to {rows}")
    padded = torch.zeros((rows, int(source.shape[1])), device=source.device, dtype=source.dtype)
    if int(source.shape[0]) > 0:
        padded[: int(source.shape[0])].copy_(source)
    return padded.contiguous()


def _single_group_offsets_experts(device: torch.device | str, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    rows = int(m)
    if rows < 0:
        raise ValueError(f"single-group metadata row count must be non-negative, got {rows}")
    key = (str(device), rows)
    cached = _SINGLE_GROUP_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("attention activation offload metadata must be initialized before CUDA graph capture")
    offsets = torch.tensor([0, rows], device=device, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=device, dtype=torch.int32)
    _SINGLE_GROUP_METADATA_CACHE[key] = (offsets, experts)
    return offsets, experts


def _record_attn_hbm_gemm(stats: AsymExecutionStats | None, tag: str) -> None:
    if stats is None or not tag:
        return
    stats.attn_act_hbm_gemm_calls_by_tag[tag] = stats.attn_act_hbm_gemm_calls_by_tag.get(tag, 0) + 1


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _attention_saved_tensor_min_bytes() -> int:
    raw = os.environ.get("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_MIN_BYTES")
    if raw is None or raw == "":
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES


def _attention_saved_tensor_dtypes() -> frozenset[torch.dtype]:
    raw = os.environ.get("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_DTYPES")
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


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


class AttentionSavedTensorOffloadWrapper:
    """Forward wrapper that offloads large attention saved tensors to CPU."""

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
        self.min_bytes = _attention_saved_tensor_min_bytes() if min_bytes is None else max(0, int(min_bytes))
        self.allowed_dtypes = (
            _attention_saved_tensor_dtypes()
            if allowed_dtypes is None
            else frozenset(dtype for dtype in allowed_dtypes if isinstance(dtype, torch.dtype))
        )
        self.require_grad = (
            _env_bool("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD", True)
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
        setattr(self.module, "_asym_attention_saved_tensor_offload_wrapper", self)
        self.module.forward = types.MethodType(_attention_saved_tensor_offload_forward, self.module)  # type: ignore[method-assign]

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
        if self.require_grad and not tensor.requires_grad:
            self.skipped_tensors += 1
            self.skipped_bytes += int(tensor.numel() * tensor.element_size())
            return False
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes < self.min_bytes:
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        if tensor.is_leaf and tensor.requires_grad:
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        return True

    def _tag_for(self, tensor: torch.Tensor) -> str:
        dtype_name = str(tensor.dtype).replace("torch.", "")
        shape = "x".join(str(int(dim)) for dim in tensor.shape) or "scalar"
        return f"saved.{dtype_name}.{shape}"

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
        if packed.ready_event is not None:
            packed.ready_event.synchronize()
        staged = torch.empty_strided(
            packed.original_shape,
            packed.original_stride,
            device=packed.original_device,
            dtype=packed.original_dtype,
        )
        with torch.no_grad():
            staged.copy_(packed.tensor, non_blocking=False)
        self.unpack_calls += 1
        self.staged_bytes += packed.nbytes
        self.max_stage_bytes_live = max(self.max_stage_bytes_live, self.staged_bytes)
        self.stage_bytes_by_tag[packed.tag] = self.stage_bytes_by_tag.get(packed.tag, 0) + packed.nbytes
        self.stage_peak_by_tag[packed.tag] = max(self.stage_peak_by_tag.get(packed.tag, 0), packed.nbytes)
        self.staged_bytes = max(0, self.staged_bytes - packed.nbytes)
        self.cpu_owned_bytes = max(0, self.cpu_owned_bytes - packed.nbytes)
        self.cpu_bytes_by_tag[packed.tag] = max(0, self.cpu_bytes_by_tag.get(packed.tag, 0) - packed.nbytes)
        self._sync_module_stats()
        return staged

    def snapshot(self) -> dict[str, Any]:
        return {
            "attention_saved_tensor_offload": True,
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


def _attention_saved_tensor_offload_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None)
    if not isinstance(wrapper, AttentionSavedTensorOffloadWrapper):
        raise RuntimeError("attention saved-tensor offload wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_attention_saved_tensor_offload(
    module: nn.Module,
    *,
    min_bytes: int | None = None,
    require_grad: bool | None = None,
    allowed_dtypes: set[torch.dtype] | frozenset[torch.dtype] | None = None,
) -> AttentionSavedTensorOffloadWrapper:
    existing = getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None)
    if isinstance(existing, AttentionSavedTensorOffloadWrapper):
        return existing
    wrapper = AttentionSavedTensorOffloadWrapper(
        module,
        min_bytes=min_bytes,
        require_grad=require_grad,
        allowed_dtypes=allowed_dtypes,
    )
    wrapper.install()
    return wrapper


def is_attention_saved_tensor_offload_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None), AttentionSavedTensorOffloadWrapper)


def attention_saved_tensor_offload_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, module in model.named_modules() if name and is_attention_saved_tensor_offload_wrapper(module)
    )


def _source_key(tensor: torch.Tensor) -> tuple[str, int, int, tuple[int, ...], tuple[int, ...], str]:
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = id(tensor)
    return (
        str(tensor.device),
        storage_ptr,
        int(tensor.storage_offset()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
    )


class _SharedActivationSource:
    def __init__(self, context: "AttentionActivationOffloadContext", handle: CPUActivationHandle) -> None:
        self._context = context
        self.handle = handle
        self.refcount = 0
        self.released = False

    def retain(self) -> "_SharedActivationSource":
        if self.released:
            raise RuntimeError(f"cannot retain released shared attention source {self.handle.tag}")
        self.refcount += 1
        return self

    def release(self) -> None:
        if self.released:
            return
        self.refcount -= 1
        if self.refcount > 0:
            return
        if self.refcount < 0:
            self.refcount = 0
            raise RuntimeError(f"shared attention source {self.handle.tag} was released too many times")
        self.released = True
        self._context._release_source(self)


class AttentionActivationOffloadContext:
    """Per-attention-parent q/k/v CPU source sharing state."""

    def __init__(self, *, pin_memory: bool = True) -> None:
        self.manager = ActivationOffloadManager(pin_memory=pin_memory)
        self.source_share_hits = 0
        self.source_share_misses = 0
        self.source_share_duplicate_bytes_avoided = 0
        self.source_share_retained_bytes = 0
        self.source_share_released_bytes = 0
        self._cache: dict[tuple[str, int, int, tuple[int, ...], tuple[int, ...], str], _SharedActivationSource] = {}
        self._seen_roles: set[str] = set()

    def acquire_source(self, source: torch.Tensor, flat_source: torch.Tensor, role: str) -> _SharedActivationSource:
        role = str(role)
        if role not in _QKV_SHARE_ROLES:
            handle = self.manager.offload(flat_source, f"{role}.U")
            self.source_share_misses += 1
            return _SharedActivationSource(self, handle).retain()

        key = _source_key(source)
        cached = self._cache.get(key)
        if cached is not None and not cached.released:
            self.source_share_hits += 1
            self.source_share_duplicate_bytes_avoided += cached.handle.nbytes
            shared = cached.retain()
        else:
            handle = self.manager.offload(flat_source, f"{role}.U")
            shared = _SharedActivationSource(self, handle).retain()
            self._cache[key] = shared
            self.source_share_misses += 1
            self.source_share_retained_bytes += handle.nbytes

        self._seen_roles.add(role)
        if role == "v_proj" or _QKV_SHARE_ROLES <= self._seen_roles:
            self._cache.clear()
            self._seen_roles.clear()
        return shared

    def _release_source(self, source: _SharedActivationSource) -> None:
        self.source_share_released_bytes += self.manager.release_cpu(source.handle)

    def snapshot(self) -> dict[str, Any]:
        data = self.manager.snapshot()
        data.update(
            {
                "source_share_hits": self.source_share_hits,
                "source_share_misses": self.source_share_misses,
                "source_share_duplicate_bytes_avoided": self.source_share_duplicate_bytes_avoided,
                "source_share_retained_bytes": self.source_share_retained_bytes,
                "source_share_released_bytes": self.source_share_released_bytes,
                "source_share_cache_entries": len(self._cache),
                "source_share_live_handles": sum(1 for source in self._cache.values() if not source.released),
            }
        )
        return data


def _update_snapshot(
    snapshot: dict[str, Any] | None,
    local_manager: ActivationOffloadManager,
    source_context: AttentionActivationOffloadContext | None,
) -> None:
    if snapshot is None:
        return
    snapshot.clear()
    snapshot.update(local_manager.snapshot())
    if source_context is not None:
        snapshot["source_context"] = source_context.snapshot()


def _dense_lora_a_cpu_left(
    u_drop_cpu: torch.Tensor,
    a: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
    backend: str = "asym",
) -> torch.Tensor:
    """Compute dense LoRA-A as one logical CPU-left grouped projection."""

    _check_backend(backend)
    if u_drop_cpu.dim() != 2 or a.dim() != 2:
        raise ValueError(f"dense LoRA-A expects U=[M,in] and A=[r,in], got {tuple(u_drop_cpu.shape)} and {tuple(a.shape)}")
    if u_drop_cpu.dtype != torch.bfloat16 or a.dtype != torch.bfloat16:
        raise ValueError("dense LoRA-A CPU-left path expects BF16 operands")
    if u_drop_cpu.device.type != "cpu":
        raise ValueError(f"dense LoRA-A expects a CPU source activation, got {u_drop_cpu.device}")
    if not u_drop_cpu.is_contiguous() or not a.is_contiguous():
        raise ValueError("dense LoRA-A expects contiguous operands")
    if int(u_drop_cpu.shape[1]) != int(a.shape[1]):
        raise ValueError(f"dense LoRA-A shape mismatch: {tuple(u_drop_cpu.shape)} vs {tuple(a.shape)}")
    if backend == "asym" and a.device.type != "cuda":
        raise ValueError(f"dense LoRA-A AsymGEMM path expects CUDA LoRA-A weights, got {a.device}")

    if int(u_drop_cpu.shape[0]) == 0:
        return torch.empty((0, int(a.shape[0])), device=a.device, dtype=a.dtype)

    if backend == "torch":
        u_stage = u_drop_cpu.to(device=a.device, dtype=a.dtype, non_blocking=u_drop_cpu.is_pinned())
        out = u_stage @ a.t()
        _record_attn_hbm_gemm(stats, tag)
    else:
        offsets, experts = _single_group_offsets_experts(a.device, int(u_drop_cpu.shape[0]))
        out = grouped_expert_lora_cpu_left(
            u_drop_cpu,
            a.unsqueeze(0).contiguous(),
            offsets,
            experts,
            output_dtype=a.dtype,
            stats=stats,
        )
    if stats is not None:
        stats.attn_act_lora_a_forward_calls += 1
    return out


class _AsymActivationOffloadLoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        base_layer: AsymFrozenLinear,
        scaling: float,
        lora_dropout_p: float,
        training: bool,
        lora_dtype: torch.dtype,
        projection_role: str,
        stats: AsymExecutionStats | None,
        backend: str,
        snapshot: dict[str, Any] | None,
        attention_context: AttentionActivationOffloadContext | None,
    ) -> torch.Tensor:
        _check_backend(backend)
        if base_layer.precision != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 base weights")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("attention activation offload expects BF16 LoRA weights")
        if float(lora_dropout_p) != 0.0:
            raise NotImplementedError("attention activation offload dropout is not implemented yet")
        if x.dim() < 1:
            raise ValueError("attention activation offload input must have at least one dimension")

        flat, input_shape = _flatten_last_dim(x, base_layer.in_features)
        flat_lora = flat.to(dtype=lora_dtype).contiguous()
        if flat_lora.dtype != torch.bfloat16:
            raise ValueError("attention activation offload currently requires BF16 activation math")

        base = asym_bf16_cpu_right_matmul(
            flat_lora,
            base_layer.host_weight.weight,
            backend=base_layer.backend,
            stats=stats,
            phase="forward",
            tag=f"{projection_role}.base_forward",
            compiled_dims=base_layer.compiled_dims,
            output_dtype=base_layer.bf16_output_dtype,
        )
        if base_layer.bias_cpu is not None:
            base = base + base_layer.bias_cpu.to(device=base.device, dtype=base.dtype, non_blocking=base_layer.bias_cpu.is_pinned())

        manager = ActivationOffloadManager(pin_memory=True)
        shared_source = None
        if attention_context is None:
            u_handle = manager.offload(flat_lora, f"{projection_role}.U")
        else:
            shared_source = attention_context.acquire_source(x, flat_lora, projection_role)
            u_handle = shared_source.handle
        s = _dense_lora_a_cpu_left(
            u_handle.tensor,
            a.contiguous(),
            stats=stats,
            tag=f"{projection_role}.lora_a_forward",
            backend=backend,
        )
        _record_attn_hbm_gemm(stats, f"{projection_role}.lora_b_forward")
        delta = s @ b.t()
        out = base + (delta * float(scaling)).to(dtype=base.dtype)
        s_handle = manager.offload(s.contiguous(), f"{projection_role}.S")
        _update_snapshot(snapshot, manager, attention_context)

        ctx.save_for_backward(a, b)
        ctx.manager = manager
        ctx.u_handle = u_handle
        ctx.shared_source = shared_source
        ctx.s_handle = s_handle
        ctx.base_layer = base_layer
        ctx.input_shape = input_shape
        ctx.input_dtype = x.dtype
        ctx.scaling = float(scaling)
        ctx.projection_role = projection_role
        ctx.stats = stats
        ctx.backend = backend
        ctx.lora_dropout_p = float(lora_dropout_p)
        ctx.snapshot = snapshot
        ctx.attention_context = attention_context
        return _restore_last_dim(out, input_shape, base_layer.out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None, None, None, None, None]:
        if ctx.lora_dropout_p != 0.0:
            raise NotImplementedError("attention activation offload dropout is not implemented yet")

        a, b = ctx.saved_tensors
        manager: ActivationOffloadManager = ctx.manager
        u_handle: CPUActivationHandle = ctx.u_handle
        s_handle: CPUActivationHandle = ctx.s_handle
        base_layer: AsymFrozenLinear = ctx.base_layer
        stats: AsymExecutionStats | None = ctx.stats
        role = str(ctx.projection_role)

        grad_x = grad_a = grad_b = None
        s_stage = None
        try:
            d_y = grad_output.reshape(-1, base_layer.out_features).to(dtype=torch.bfloat16).contiguous()
            needs_grad_x = bool(ctx.needs_input_grad[0])
            needs_grad_a = bool(ctx.needs_input_grad[1])
            needs_grad_b = bool(ctx.needs_input_grad[2])
            needs_low_rank = needs_grad_x or needs_grad_a or needs_grad_b

            d_s = None
            if needs_low_rank:
                _record_attn_hbm_gemm(stats, f"{role}.dS")
                d_s = (d_y @ b).to(dtype=torch.bfloat16) * float(ctx.scaling)

            if needs_grad_x:
                d_u = asym_bf16_cpu_right_matmul(
                    d_y,
                    base_layer.host_weight.weight,
                    transpose_b=True,
                    backend=base_layer.backend,
                    stats=stats,
                    phase="attn_act_base_dx",
                    tag=f"{role}.base_dx",
                    compiled_dims=base_layer.compiled_dims,
                    output_dtype=torch.bfloat16,
                )
                if d_s is not None:
                    _record_attn_hbm_gemm(stats, f"{role}.lora_input_grad")
                    d_u = d_u + (d_s @ a).to(dtype=d_u.dtype)
                grad_x = d_u.to(dtype=ctx.input_dtype).reshape(ctx.input_shape)

            if needs_grad_a:
                if d_s is None:
                    raise RuntimeError("internal error: dS was not computed for dA")
                m_grad = _align_up(int(d_s.shape[0]), 64)
                if m_grad == 0:
                    grad_a = torch.zeros_like(a)
                else:
                    u_source = _pad_cpu_rows_to(u_handle.tensor, m_grad)
                    d_s_rows = _pad_hbm_rows_to(d_s, m_grad)
                    d_s_t = d_s_rows.t().contiguous()
                    grad_a = asym_bf16_cpu_right_matmul(
                        d_s_t,
                        u_source,
                        transpose_b=True,
                        backend=ctx.backend,
                        stats=stats,
                        phase="attn_act_dA",
                        tag=f"{role}.dA",
                        compiled_dims=base_layer.compiled_dims,
                        output_dtype=a.dtype,
                    ).to(dtype=a.dtype)

            if needs_grad_b:
                if stats is not None:
                    stats.attn_act_stage_low_rank_calls += 1
                s_stage = manager.stage(s_handle, tag=f"{role}.S_stage")
                _record_attn_hbm_gemm(stats, f"{role}.dB")
                grad_b = ((d_y.t().contiguous() @ s_stage).to(dtype=b.dtype) * float(ctx.scaling)).to(dtype=b.dtype)
        finally:
            manager.release_stage(s_stage)
            manager.release_cpu(s_handle)
            if ctx.shared_source is None:
                manager.release_cpu(u_handle)
            else:
                ctx.shared_source.release()
            _update_snapshot(ctx.snapshot, manager, ctx.attention_context)

        return grad_x, grad_a, grad_b, None, None, None, None, None, None, None, None, None, None


class AsymActivationOffloadLoRALinear(nn.Module):
    """Dense attention LoRA linear that offloads forward activations to CPU."""

    def __init__(
        self,
        source: HostWeight | torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
        projection_role: str = "attention",
        attention_context: AttentionActivationOffloadContext | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        if str(precision).lower() != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 precision")
        _check_backend(backend)
        if isinstance(source, HostWeight):
            host_weight = source
        elif isinstance(source, torch.Tensor):
            host_weight = HostWeight.from_tensor(source, dtype=source.dtype, pin_memory=True)
        else:
            raise TypeError(f"source must be a HostWeight or torch.Tensor, got {type(source)!r}")
        self._init_from_host_weight(
            host_weight,
            bias=bias,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            lora_generator=lora_generator,
            lora_dtype=lora_dtype,
            precision=precision,
            adapter_name=adapter_name,
            init_lora_weights=init_lora_weights,
            lora_dropout=lora_dropout,
            projection_role=projection_role,
            attention_context=attention_context,
        )

    @classmethod
    def from_host_weight(
        cls,
        host_weight: HostWeight,
        *,
        bias: torch.Tensor | None = None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
        projection_role: str = "attention",
        attention_context: AttentionActivationOffloadContext | None = None,
    ) -> "AsymActivationOffloadLoRALinear":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._init_from_host_weight(
            host_weight,
            bias=bias,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            lora_generator=lora_generator,
            lora_dtype=lora_dtype,
            precision=precision,
            adapter_name=adapter_name,
            init_lora_weights=init_lora_weights,
            lora_dropout=lora_dropout,
            projection_role=projection_role,
            attention_context=attention_context,
        )
        return obj

    def _init_from_host_weight(
        self,
        host_weight: HostWeight,
        *,
        bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None,
        device: torch.device | None,
        lora_generator: torch.Generator | None,
        lora_dtype: torch.dtype | str | None,
        precision: str,
        adapter_name: str,
        init_lora_weights: Literal["asym", "peft"],
        lora_dropout: float,
        projection_role: str,
        attention_context: AttentionActivationOffloadContext | None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        if str(precision).lower() != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 precision")
        _check_backend(backend)
        resolved_device = torch.device("cpu" if device is None else device)
        resolved_lora_dtype = normalize_lora_dtype(lora_dtype)
        if resolved_lora_dtype != torch.bfloat16:
            raise ValueError("attention activation offload currently requires BF16 LoRA weights")
        self.base_layer = AsymFrozenLinear.from_host_weight(
            host_weight,
            bias=bias,
            backend=backend,
            stats=stats,
            precision="bf16",
            bf16_output_dtype=torch.bfloat16,
        )
        self.lora_A = nn.ModuleDict(
            {
                adapter_name: nn.Linear(
                    host_weight.in_features,
                    rank,
                    bias=False,
                    device=resolved_device,
                    dtype=resolved_lora_dtype,
                )
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                adapter_name: nn.Linear(
                    rank,
                    host_weight.out_features,
                    bias=False,
                    device=resolved_device,
                    dtype=resolved_lora_dtype,
                )
            }
        )
        self.active_adapter = adapter_name
        self.lora_dtype = resolved_lora_dtype
        self.scaling = float(alpha) / float(rank)
        self.precision = "bf16"
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        self.projection_role = str(projection_role)
        self.attention_context = attention_context
        self._last_activation_offload_stats: dict[str, Any] = {}
        self._reset_lora(adapter_name, lora_generator, init_lora_weights=init_lora_weights)

    @property
    def base(self) -> AsymFrozenLinear:
        return self.base_layer

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.base_layer.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_hbm_saved_bytes

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return 0

    def _reset_lora(
        self,
        adapter_name: str,
        generator: torch.Generator | None,
        *,
        init_lora_weights: Literal["asym", "peft"],
    ) -> None:
        with torch.no_grad():
            _reset_lora_weights(
                self.lora_A[adapter_name].weight,
                self.lora_B[adapter_name].weight,
                init_lora_weights=init_lora_weights,
                generator=generator,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _AsymActivationOffloadLoRALinearFunction.apply(
            x,
            self.lora_A[self.active_adapter].weight,
            self.lora_B[self.active_adapter].weight,
            self.base_layer,
            self.scaling,
            self.lora_dropout_p,
            self.training,
            self.lora_dtype,
            self.projection_role,
            self.base_layer.stats,
            self.base_layer.backend,
            self._last_activation_offload_stats,
            self.attention_context,
        )


__all__ = [
    "AsymActivationOffloadLoRALinear",
    "AttentionActivationOffloadContext",
    "AttentionSavedTensorOffloadWrapper",
    "attention_saved_tensor_offload_module_names",
    "install_attention_saved_tensor_offload",
    "is_attention_saved_tensor_offload_wrapper",
    "_dense_lora_a_cpu_left",
    "_single_group_offsets_experts",
]
