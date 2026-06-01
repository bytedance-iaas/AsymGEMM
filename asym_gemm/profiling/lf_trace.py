from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
import re
import time
import warnings
from typing import Any, Callable, Iterator

import torch
from torch import nn

from asym_gemm.training.profile_ranges import current_profile_range, prof_range, scoped_name


_PATCH_ATTR = "_asym_lf_profile_wrapped"
_HOOK_ATTR = "_asym_lf_profile_hooks_installed"


def _csv_set(value: str) -> set[str]:
    return {part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()}


def _parse_bool(value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _semantic_module_name(name: str) -> str | None:
    layer = _layer_index(name)
    if layer is None:
        return None
    if name.endswith(".self_attn"):
        return f"layers.{layer}.self_attn"
    if name.endswith(".mlp.experts") or name.endswith(".experts"):
        return f"layers.{layer}.mlp.experts"
    if name.endswith(".mlp"):
        return f"layers.{layer}.mlp"
    return None


def _filter_token(semantic_name: str) -> str:
    if semantic_name.endswith(".self_attn"):
        return "attention"
    if semantic_name.endswith(".mlp.experts"):
        return "experts"
    if semantic_name.endswith(".mlp"):
        return "mlp"
    return semantic_name.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class LFTraceConfig:
    level: str = "stage"
    layers: str = "all"
    module_filter: str = "attention,mlp,experts,lora,optimizer"
    memory_attribution: bool = False
    sync: bool = False

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "LFTraceConfig":
        return cls(
            level=env.get("ASYM_GEMM_LF_PROFILE_LEVEL", "stage").strip().lower(),
            layers=env.get("ASYM_GEMM_LF_PROFILE_LAYERS", "all").strip().lower(),
            module_filter=env.get(
                "ASYM_GEMM_LF_PROFILE_MODULE_FILTER",
                "attention,mlp,experts,lora,optimizer",
            ),
            memory_attribution=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION", "0")),
            sync=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_SYNC", "0")),
        )

    def __post_init__(self) -> None:
        if self.level not in {"stage", "module", "op", "deep"}:
            raise ValueError(f"PROFILE_LEVEL must be stage, module, op, or deep; got {self.level!r}")

    @property
    def module_ranges_enabled(self) -> bool:
        return self.level in {"module", "op", "deep"}

    @property
    def backward_module_ranges_enabled(self) -> bool:
        return self.level == "deep" or "backward" in self.module_filter_set

    @property
    def op_ranges_enabled(self) -> bool:
        return self.level in {"op", "deep"}

    @property
    def module_filter_set(self) -> set[str]:
        values = _csv_set(self.module_filter)
        return values or {"attention", "mlp", "experts", "lora", "optimizer"}

    def layer_selected(self, index: int, max_index: int | None) -> bool:
        spec = self.layers
        if spec in {"", "all", "*"}:
            return True
        if spec in {"first,last", "first_last"}:
            return index == 0 or (max_index is not None and index == max_index)
        if spec.startswith("every"):
            try:
                interval = int(spec[len("every") :])
            except ValueError:
                return False
            return interval > 0 and index % interval == 0
        selected: set[int] = set()
        for part in spec.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if part == "last" and max_index is not None:
                selected.add(max_index)
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                try:
                    start, end = int(left), int(right)
                except ValueError:
                    continue
                selected.update(range(start, end + 1))
                continue
            try:
                selected.add(int(part))
            except ValueError:
                continue
        return index in selected


@dataclass
class _HookRange:
    name: str
    start_time: float


class SavedTensorTracker:
    def __init__(self) -> None:
        self.rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        self.total_reference_bytes = 0

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        owner = current_profile_range() or "unattributed"
        num_bytes = int(tensor.numel() * tensor.element_size())
        device = str(tensor.device)
        key = (int(tensor.untyped_storage().data_ptr()), num_bytes, str(tensor.dtype), device)
        self.total_reference_bytes += num_bytes
        row = self.rows.setdefault(
            key,
            {
                "owner": owner,
                "dtype": str(tensor.dtype),
                "device": device,
                "shape": list(tensor.shape),
                "bytes": num_bytes,
                "references": 0,
            },
        )
        row["references"] += 1
        if row.get("owner") == "unattributed" and owner != "unattributed":
            row["owner"] = owner
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def summary(self) -> dict[str, Any]:
        rows = list(self.rows.values())
        owner_bytes: dict[str, int] = {}
        for row in rows:
            owner = str(row.get("owner") or "unattributed")
            owner_bytes[owner] = owner_bytes.get(owner, 0) + int(row.get("bytes") or 0)
        owner_rows = [
            {"owner": owner, "bytes": value}
            for owner, value in sorted(owner_bytes.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "enabled": True,
            "unique_tensors": len(rows),
            "total_unique_bytes": sum(int(row.get("bytes") or 0) for row in rows),
            "total_reference_bytes": self.total_reference_bytes,
            "rows": sorted(rows, key=lambda row: int(row.get("bytes") or 0), reverse=True),
            "by_owner": owner_rows,
        }


@dataclass
class LFTraceHandle:
    config: LFTraceConfig
    recorder: Any | None = None
    originals: list[tuple[Any, str, Any]] = field(default_factory=list)
    module_hooks: list[Any] = field(default_factory=list)
    patched_optimizers: set[tuple[int, str]] = field(default_factory=set)
    patched_schedulers: set[tuple[int, str]] = field(default_factory=set)
    saved_tensor_tracker: SavedTensorTracker | None = None
    model: nn.Module | None = None

    def restore(self) -> None:
        for hook in reversed(self.module_hooks):
            try:
                hook.remove()
            except Exception:
                pass
        self.module_hooks.clear()
        for owner, attr, original in reversed(self.originals):
            setattr(owner, attr, original)
        self.originals.clear()

    def memory_summary(self, model: nn.Module | None = None) -> dict[str, Any]:
        summary = _model_memory_summary(model) if model is not None else {"enabled": False, "rows": []}
        if self.saved_tensor_tracker is not None:
            summary["saved_tensors"] = self.saved_tensor_tracker.summary()
        else:
            summary["saved_tensors"] = {"enabled": False, "rows": []}
        return summary

    @contextmanager
    def saved_tensor_context(self) -> Iterator[None]:
        if not self.config.memory_attribution:
            yield
            return
        self.saved_tensor_tracker = SavedTensorTracker()
        with torch.autograd.graph.saved_tensors_hooks(
            self.saved_tensor_tracker.pack,
            self.saved_tensor_tracker.unpack,
        ):
            yield


def _record_original(handle: LFTraceHandle, owner: Any, attr: str) -> Any:
    original = getattr(owner, attr)
    handle.originals.append((owner, attr, original))
    return original


def _stage(handle: LFTraceHandle, name: str):
    if handle.recorder is not None:
        return handle.recorder.stage(name, sync=handle.config.sync)
    return prof_range(name)


@contextmanager
def _range(handle: LFTraceHandle, name: str) -> Iterator[None]:
    with _stage(handle, name):
        yield


def _patch_method(handle: LFTraceHandle, cls: type[Any], attr: str, wrapper_factory: Callable[[Any], Any]) -> None:
    original = getattr(cls, attr, None)
    if original is None or getattr(original, _PATCH_ATTR, False):
        return
    wrapped = wrapper_factory(_record_original(handle, cls, attr))
    setattr(wrapped, _PATCH_ATTR, True)
    setattr(cls, attr, wrapped)


def _wrap_callable_once(owner: Any, attr: str, name: str, handle: LFTraceHandle, seen: set[tuple[int, str]]) -> None:
    if owner is None or not hasattr(owner, attr):
        return
    original = getattr(owner, attr)
    if not callable(original) or getattr(original, _PATCH_ATTR, False):
        return
    key = (id(owner), attr)
    if key in seen:
        return
    seen.add(key)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _range(handle, name):
            return original(*args, **kwargs)

    setattr(wrapped, _PATCH_ATTR, True)
    if getattr(original, "_wrapped_by_lr_sched", False):
        setattr(wrapped, "_wrapped_by_lr_sched", True)
    try:
        setattr(owner, attr, wrapped)
    except Exception:
        seen.discard(key)


def _patch_optimizer_objects(handle: LFTraceHandle, trainer: Any) -> None:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        _wrap_callable_once(optimizer, "step", "lf.optimizer.step", handle, handle.patched_optimizers)
        _wrap_callable_once(optimizer, "zero_grad", "lf.optimizer.zero_grad", handle, handle.patched_optimizers)
    scheduler = getattr(trainer, "lr_scheduler", None)
    if scheduler is not None:
        _wrap_callable_once(scheduler, "step", "lf.scheduler.step", handle, handle.patched_schedulers)


def _patch_training_phases(handle: LFTraceHandle) -> None:
    from accelerate import Accelerator
    from transformers import Trainer

    def training_step_factory(original):
        def training_step_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            _install_module_hooks_once(handle, getattr(self, "model", None))
            _patch_optimizer_objects(handle, self)
            with _range(handle, "lf.step.total"):
                return original(self, *args, **kwargs)

        return training_step_with_profile

    def prepare_inputs_factory(original):
        def prepare_inputs_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _range(handle, "lf.inputs.prepare"):
                return original(self, *args, **kwargs)

        return prepare_inputs_with_profile

    def compute_loss_factory(original):
        def compute_loss_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "_asym_lf_in_compute_loss_profile", False):
                return original(self, *args, **kwargs)
            setattr(self, "_asym_lf_in_compute_loss_profile", True)
            with _range(handle, "step.forward"):
                try:
                    with prof_range("lf.forward_loss"):
                        return original(self, *args, **kwargs)
                finally:
                    setattr(self, "_asym_lf_in_compute_loss_profile", False)

        return compute_loss_with_profile

    def train_factory(original):
        def train_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            _install_module_hooks_once(handle, getattr(self, "model", None))
            _patch_optimizer_objects(handle, self)
            with _range(handle, "lf.train.total"):
                return original(self, *args, **kwargs)

        return train_with_profile

    def maybe_log_factory(original):
        def maybe_log_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _range(handle, "lf.log_save_eval"):
                return original(self, *args, **kwargs)

        return maybe_log_with_profile

    def create_optimizer_factory(original):
        def create_optimizer_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            _patch_optimizer_objects(handle, self)
            return result

        return create_optimizer_with_profile

    def create_scheduler_factory(original):
        def create_scheduler_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            _patch_optimizer_objects(handle, self)
            return result

        return create_scheduler_with_profile

    def get_train_dataloader_factory(original):
        def get_train_dataloader_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            dataloader = original(self, *args, **kwargs)
            return _ProfiledDataLoader(dataloader, handle)

        return get_train_dataloader_with_profile

    def backward_factory(original):
        def backward_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _range(handle, "step.backward"):
                with prof_range("lf.backward"):
                    return original(self, *args, **kwargs)

        return backward_with_profile

    def clip_grad_factory(original):
        def clip_grad_with_profile(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _range(handle, "lf.grad_clip"):
                return original(self, *args, **kwargs)

        return clip_grad_with_profile

    _patch_method(handle, Trainer, "training_step", training_step_factory)
    _patch_method(handle, Trainer, "_prepare_inputs", prepare_inputs_factory)
    _patch_method(handle, Trainer, "compute_loss", compute_loss_factory)
    _patch_method(handle, Trainer, "train", train_factory)
    _patch_method(handle, Trainer, "_maybe_log_save_evaluate", maybe_log_factory)
    _patch_method(handle, Trainer, "create_optimizer", create_optimizer_factory)
    _patch_method(handle, Trainer, "create_scheduler", create_scheduler_factory)
    _patch_method(handle, Trainer, "get_train_dataloader", get_train_dataloader_factory)
    _patch_method(handle, Accelerator, "backward", backward_factory)
    _patch_method(handle, Accelerator, "clip_grad_norm_", clip_grad_factory)

    try:
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

        _patch_method(handle, CustomSeq2SeqTrainer, "compute_loss", compute_loss_factory)
        _patch_method(handle, CustomSeq2SeqTrainer, "create_optimizer", create_optimizer_factory)
        _patch_method(handle, CustomSeq2SeqTrainer, "create_scheduler", create_scheduler_factory)
    except Exception:
        pass


class _ProfiledDataLoader:
    def __init__(self, dataloader: Any, handle: LFTraceHandle) -> None:
        self._dataloader = dataloader
        self._handle = handle

    def __iter__(self):
        return _ProfiledIterator(iter(self._dataloader), self._handle)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataloader, name)


class _ProfiledIterator:
    def __init__(self, iterator: Any, handle: LFTraceHandle) -> None:
        self._iterator = iterator
        self._handle = handle

    def __iter__(self):
        return self

    def __next__(self):
        with _range(self._handle, "lf.data.next"):
            return next(self._iterator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)


def _range_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _range_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _install_module_hooks_once(handle: LFTraceHandle, model: nn.Module | None) -> None:
    if model is not None:
        handle.model = model
    if model is None or not handle.config.module_ranges_enabled or getattr(model, _HOOK_ATTR, False):
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Full backward hook is firing when gradients are computed with respect to module outputs.*",
        category=UserWarning,
    )
    modules = list(model.named_modules())
    layer_indices = [_layer_index(name) for name, _module in modules]
    max_layer = max((idx for idx in layer_indices if idx is not None), default=None)
    filters = handle.config.module_filter_set

    for module_name, module in modules:
        semantic_name = _semantic_module_name(module_name)
        if semantic_name is None:
            continue
        layer = _layer_index(semantic_name)
        if layer is None or not handle.config.layer_selected(layer, max_layer):
            continue
        if _filter_token(semantic_name) not in filters:
            continue

        forward_name = scoped_name("forward", semantic_name)
        backward_name = scoped_name("backward", semantic_name)

        def forward_pre(_module: nn.Module, _args: tuple[Any, ...], *, _name: str = forward_name) -> None:
            _range_push(_name)

        def forward_post(_module: nn.Module, _args: tuple[Any, ...], _output: Any, *, _name: str = forward_name) -> None:
            _range_pop()

        def backward_pre(_module: nn.Module, _grad_output: tuple[Any, ...], *, _name: str = backward_name) -> None:
            _range_push(_name)

        def backward_post(
            _module: nn.Module,
            _grad_input: tuple[Any, ...],
            _grad_output: tuple[Any, ...],
            *,
            _name: str = backward_name,
        ) -> None:
            _range_pop()

        handle.module_hooks.append(module.register_forward_pre_hook(forward_pre))
        handle.module_hooks.append(module.register_forward_hook(forward_post))
        if handle.config.backward_module_ranges_enabled:
            try:
                handle.module_hooks.append(module.register_full_backward_pre_hook(backward_pre))
                handle.module_hooks.append(module.register_full_backward_hook(backward_post))
            except AttributeError:
                pass

    setattr(model, _HOOK_ATTR, True)


def _component_from_name(name: str, param: nn.Parameter) -> str:
    lower = name.lower()
    if "lora" in lower:
        return "lora"
    if "optimizer" in lower:
        return "optimizer"
    if "embed" in lower:
        return "embedding"
    if "self_attn" in lower or any(part in lower for part in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if "expert" in lower or "mlp" in lower:
        return "mlp_or_experts"
    return "other_model"


def _model_memory_summary(model: nn.Module | None) -> dict[str, Any]:
    if model is None:
        return {"enabled": False, "rows": []}
    rows_by_category: dict[tuple[str, str, str], int] = {}
    total = 0
    for name, param in model.named_parameters():
        bytes_value = int(param.numel() * param.element_size())
        total += bytes_value
        component = _component_from_name(name, param)
        category = "trainable_param" if param.requires_grad else "frozen_param"
        device = "cpu" if param.device.type == "cpu" else "gpu"
        key = (category, component, device)
        rows_by_category[key] = rows_by_category.get(key, 0) + bytes_value
        if param.grad is not None:
            grad_key = ("gradient", component, "cpu" if param.grad.device.type == "cpu" else "gpu")
            rows_by_category[grad_key] = rows_by_category.get(grad_key, 0) + int(param.grad.numel() * param.grad.element_size())
    rows = [
        {"category": category, "component": component, "device": device, "bytes": bytes_value}
        for (category, component, device), bytes_value in sorted(rows_by_category.items(), key=lambda item: item[1], reverse=True)
    ]
    host_bytes = 0
    pinned_bytes = 0
    for module in model.modules():
        for attr in ("cpu_resident_base_bytes", "gpu_resident_base_bytes", "weight_hbm_saved_bytes"):
            value = getattr(module, attr, 0)
            if isinstance(value, int):
                host_bytes += value if attr != "gpu_resident_base_bytes" else 0
        for attr in ("host_weight", "weight_host"):
            host = getattr(module, attr, None)
            tensor = getattr(host, "tensor", None)
            if isinstance(tensor, torch.Tensor):
                bytes_value = int(tensor.numel() * tensor.element_size())
                host_bytes += bytes_value
                if tensor.is_pinned():
                    pinned_bytes += bytes_value
    if host_bytes:
        rows.append({"category": "host_weight", "component": "routed_experts", "device": "cpu", "bytes": host_bytes})
    if pinned_bytes:
        rows.append({"category": "pinned_host_weight", "component": "routed_experts", "device": "cpu", "bytes": pinned_bytes})
    return {"enabled": True, "total_parameter_bytes": total, "rows": rows}


def install_lf_trace(config: LFTraceConfig, recorder: Any | None = None) -> LFTraceHandle:
    handle = LFTraceHandle(config=config, recorder=recorder)
    _patch_training_phases(handle)
    return handle


def uninstall_lf_trace(handle: LFTraceHandle) -> None:
    handle.restore()


__all__ = [
    "LFTraceConfig",
    "LFTraceHandle",
    "install_lf_trace",
    "uninstall_lf_trace",
]
