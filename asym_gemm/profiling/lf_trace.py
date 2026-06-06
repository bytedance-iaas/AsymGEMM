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
_MEMORY_HOOK_ATTR = "_asym_lf_memory_breakdown_hooks_installed"


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
    memory_breakdown: bool = False
    memory_breakdown_interval: int = 1
    memory_breakdown_steps: str = ""
    memory_breakdown_modules: str = "attention,mlp,experts,lora,embedding,loss"
    memory_breakdown_output: str = "memory_breakdown"
    sync: bool = False
    nsys_capture_range: bool = False
    warmup_steps: int = 0
    total_steps: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "LFTraceConfig":
        def env_int(name: str) -> int:
            try:
                return max(int(env.get(name, "0")), 0)
            except ValueError:
                return 0

        return cls(
            level=env.get("ASYM_GEMM_LF_PROFILE_LEVEL", "stage").strip().lower(),
            layers=env.get("ASYM_GEMM_LF_PROFILE_LAYERS", "all").strip().lower(),
            module_filter=env.get(
                "ASYM_GEMM_LF_PROFILE_MODULE_FILTER",
                "attention,mlp,experts,lora,optimizer",
            ),
            memory_attribution=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION", "0")),
            memory_breakdown=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN", "0")),
            memory_breakdown_interval=max(env_int("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_INTERVAL"), 1),
            memory_breakdown_steps=env.get("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_STEPS", "").strip(),
            memory_breakdown_modules=env.get(
                "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_MODULES",
                "attention,mlp,experts,lora,embedding,loss",
            ),
            memory_breakdown_output=env.get(
                "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_OUTPUT",
                "memory_breakdown",
            ).strip()
            or "memory_breakdown",
            sync=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_SYNC", "0")),
            nsys_capture_range=_parse_bool(env.get("ASYM_GEMM_LF_NSYS_CAPTURE_RANGE", "0")),
            warmup_steps=env_int("ASYM_GEMM_LF_CONFIG_WARMUP_STEPS"),
            total_steps=env_int("ASYM_GEMM_LF_CONFIG_TOTAL_STEPS"),
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
    memory_breakdown_profiler: "LFMemoryBreakdownProfiler | None" = None
    model: nn.Module | None = None

    def restore(self) -> None:
        if self.memory_breakdown_profiler is not None:
            self.memory_breakdown_profiler.restore()
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

    def memory_breakdown_report(self) -> dict[str, Any]:
        if self.memory_breakdown_profiler is None:
            return {"enabled": False, "rows": [], "summary": {}}
        return self.memory_breakdown_profiler.report()

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


def _parse_step_set(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start, end = int(left), int(right)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            result.update(range(max(start, 1), end + 1))
            continue
        try:
            result.add(max(int(part), 1))
        except ValueError:
            continue
    return result


def _device_space(device: torch.device | str) -> str:
    device_type = torch.device(device).type if not isinstance(device, str) else device.split(":", 1)[0]
    return "GPU HBM" if device_type == "cuda" else "CPU host"


def _component_from_param_name(name: str) -> str:
    lower = name.lower()
    if "lora" in lower:
        if "self_attn" in lower or any(part in lower for part in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return "lora_attention"
        if "expert" in lower:
            return "lora_experts"
        if "mlp" in lower or any(part in lower for part in ("gate_proj", "up_proj", "down_proj")):
            return "lora_mlp"
        return "lora"
    if "embed" in lower:
        return "embedding"
    if "lm_head" in lower:
        return "lm_head"
    if "self_attn" in lower or any(part in lower for part in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if "expert" in lower:
        return "routed_experts"
    if "mlp" in lower or any(part in lower for part in ("gate_proj", "up_proj", "down_proj")):
        return "mlp_dense"
    if "router" in lower:
        return "router"
    if "norm" in lower:
        return "norms"
    return "other"


def _component_from_module_name(name: str) -> str | None:
    lower = name.lower()
    if not lower:
        return None
    if lower.endswith(".self_attn") or lower.endswith(".attention"):
        return "attention"
    if lower.endswith(".mlp.experts") or lower.endswith(".experts"):
        return "routed_experts"
    if lower.endswith(".mlp"):
        return "mlp_dense"
    if lower.endswith(".embed_tokens") or lower.endswith(".embed_in") or lower.endswith(".wte"):
        return "embedding"
    if lower.endswith(".lm_head"):
        return "lm_head"
    if lower.endswith(".norm") or lower.endswith(".input_layernorm") or lower.endswith(".post_attention_layernorm"):
        return "norms"
    if "lora" in lower:
        return "lora"
    return None


def _component_filter_token(component: str) -> str:
    if component == "routed_experts":
        return "experts"
    if component == "mlp_dense":
        return "mlp"
    if component in {"lora_attention", "lora_mlp", "lora_experts"}:
        return "lora"
    if component in {"lm_head", "loss_logits"}:
        return "loss"
    return component


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


class LFMemoryBreakdownProfiler:
    def __init__(self, config: LFTraceConfig) -> None:
        self.config = config
        self.enabled = bool(config.memory_breakdown)
        self.interval = max(int(config.memory_breakdown_interval), 1)
        self.step_filter = _parse_step_set(config.memory_breakdown_steps)
        self.module_filter = _csv_set(config.memory_breakdown_modules) or {
            "attention",
            "mlp",
            "experts",
            "lora",
            "embedding",
            "loss",
        }
        self.rows: list[dict[str, Any]] = []
        self._step = 0
        self._active = False
        self._current_activation_bytes: dict[str, int] = {}
        self._current_peak_allocated = 0
        self._current_peak_reserved = 0
        self._model: nn.Module | None = None
        self._optimizer: Any | None = None
        self._hooks: list[Any] = []

    def restore(self) -> None:
        for hook in reversed(self._hooks):
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()

    def set_optimizer(self, optimizer: Any | None) -> None:
        if optimizer is not None:
            self._optimizer = optimizer

    def should_record_step(self, step: int) -> bool:
        if not self.enabled:
            return False
        if step <= 0:
            return False
        if self.step_filter:
            return step in self.step_filter
        return (step - 1) % self.interval == 0

    def ensure_hooks(self, model: nn.Module | None) -> None:
        if not self.enabled or model is None or getattr(model, _MEMORY_HOOK_ATTR, False):
            return
        self._model = model
        modules = list(model.named_modules())
        module_names = {name for name, _module in modules}
        for module_name, module in modules:
            component = _component_from_module_name(module_name)
            if component is None:
                continue
            if _component_filter_token(component) not in self.module_filter:
                continue
            if component == "mlp_dense" and (
                f"{module_name}.experts" in module_names or f"{module_name}.mlp.experts" in module_names
            ):
                # MoE MLP wrappers include the experts; skip to avoid double counting.
                continue

            def forward_pre(_module: nn.Module, _args: tuple[Any, ...], *, _component: str = component) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                setattr(_module, "_asym_lf_memory_alloc_pre", int(torch.cuda.memory_allocated()))

            def forward_post(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _output: Any,
                *,
                _component: str = component,
            ) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                before = int(getattr(_module, "_asym_lf_memory_alloc_pre", torch.cuda.memory_allocated()))
                after = int(torch.cuda.memory_allocated())
                self._current_peak_allocated = max(self._current_peak_allocated, after, int(torch.cuda.max_memory_allocated()))
                self._current_peak_reserved = max(self._current_peak_reserved, int(torch.cuda.memory_reserved()))
                delta = after - before
                if delta > 0:
                    self._current_activation_bytes[_component] = self._current_activation_bytes.get(_component, 0) + int(delta)

            self._hooks.append(module.register_forward_pre_hook(forward_pre))
            self._hooks.append(module.register_forward_hook(forward_post))
        setattr(model, _MEMORY_HOOK_ATTR, True)

    def step_begin(self, model: nn.Module | None = None, optimizer: Any | None = None) -> None:
        if model is not None:
            self._model = model
            self.ensure_hooks(model)
        self.set_optimizer(optimizer)
        self._step += 1
        self._active = self.should_record_step(self._step)
        self._current_activation_bytes = {}
        allocated, reserved, peak_allocated, peak_reserved = self._snapshot_values()
        self._current_peak_allocated = max(allocated, peak_allocated)
        self._current_peak_reserved = max(reserved, peak_reserved)
        if self._active:
            self.record_phase("step_begin", model=model, optimizer=optimizer)

    def record_phase(self, phase: str, model: nn.Module | None = None, optimizer: Any | None = None) -> None:
        if not self.enabled or not self._active:
            return
        if model is not None:
            self._model = model
            self.ensure_hooks(model)
        self.set_optimizer(optimizer)
        allocated, reserved, peak_allocated, peak_reserved = self._snapshot_values()
        self._current_peak_allocated = max(self._current_peak_allocated, allocated, peak_allocated)
        self._current_peak_reserved = max(self._current_peak_reserved, reserved, peak_reserved)
        persistent = self._collect_persistent_bytes(self._model, self._optimizer)
        activation = dict(sorted(self._current_activation_bytes.items()))
        row = {
            "schema_version": 1,
            "rank": _distributed_rank(),
            "world_size": _distributed_world_size(),
            "step": self._step,
            "raw_step": self._step,
            "is_warmup": self._step <= max(int(self.config.warmup_steps), 0),
            "phase": phase,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_since_step_begin": self._current_peak_allocated,
            "peak_reserved_since_step_begin": self._current_peak_reserved,
            "persistent_bytes": persistent,
            "activation_bytes": activation,
            "closure_bytes": self._closure_bytes(
                self._current_peak_allocated,
                allocated,
                reserved,
                persistent,
                activation,
            ),
            "methods": {
                "persistent_bytes": "exact tensor size",
                "activation_bytes": "measured forward allocated delta",
                "framework_temp_workspace": "inferred residual to peak",
                "allocator_reserved_unallocated": "reserved - allocated snapshot",
            },
        }
        self.rows.append(row)

    def _snapshot_values(self) -> tuple[int, int, int, int]:
        if not torch.cuda.is_available():
            return 0, 0, 0, 0
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        try:
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        except RuntimeError:
            peak_allocated = allocated
            peak_reserved = reserved
        return allocated, reserved, peak_allocated, peak_reserved

    def _collect_persistent_bytes(self, model: nn.Module | None, optimizer: Any | None) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}

        def add(component: str, kind: str, tensor: torch.Tensor) -> None:
            space = _device_space(tensor.device)
            key = kind if space == "GPU HBM" else f"{kind}_cpu"
            bucket = result.setdefault(component, {})
            bucket[key] = bucket.get(key, 0) + _tensor_bytes(tensor)
            if space != "GPU HBM" and tensor.is_pinned():
                bucket[f"{kind}_cpu_pinned"] = bucket.get(f"{kind}_cpu_pinned", 0) + _tensor_bytes(tensor)

        param_names: dict[int, str] = {}
        if model is not None:
            for name, param in model.named_parameters():
                param_names[id(param)] = name
                add(_component_from_param_name(name), "weight" if param.requires_grad else "frozen_weight", param)
                if param.grad is not None:
                    add(_component_from_param_name(name), "grad", param.grad)
            for name, buffer in model.named_buffers():
                if isinstance(buffer, torch.Tensor):
                    add(_component_from_param_name(name), "buffer", buffer)
            for module in model.modules():
                for attr in ("host_weight", "weight_host"):
                    host = getattr(module, attr, None)
                    tensor = getattr(host, "tensor", None)
                    if isinstance(tensor, torch.Tensor):
                        add("routed_experts", "host_weight", tensor)

        if optimizer is not None:
            try:
                state_items = list(optimizer.state.items())
            except Exception:
                state_items = []
            for param, state in state_items:
                component = _component_from_param_name(param_names.get(id(param), "optimizer"))
                if not isinstance(state, dict):
                    continue
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        add(component, "optimizer_state", value)
                    elif isinstance(value, dict):
                        for nested in value.values():
                            if isinstance(nested, torch.Tensor):
                                add(component, "optimizer_state", nested)
        return result

    @staticmethod
    def _persistent_gpu_bytes(persistent: dict[str, dict[str, int]]) -> int:
        total = 0
        for kinds in persistent.values():
            for kind, value in kinds.items():
                if not kind.endswith("_cpu") and not kind.endswith("_cpu_pinned"):
                    total += int(value)
        return total

    def _closure_bytes(
        self,
        peak_allocated: int,
        allocated: int,
        reserved: int,
        persistent: dict[str, dict[str, int]],
        activation: dict[str, int],
    ) -> dict[str, int]:
        persistent_gpu = self._persistent_gpu_bytes(persistent)
        activation_gpu = sum(max(int(value), 0) for value in activation.values())
        known_allocated = persistent_gpu + activation_gpu
        return {
            "framework_temp_workspace": max(0, int(peak_allocated) - known_allocated),
            "allocator_reserved_unallocated": max(0, int(reserved) - int(allocated)),
        }

    def report(self) -> dict[str, Any]:
        summary = build_memory_breakdown_summary(self.rows)
        return {"enabled": self.enabled, "rows": self.rows, "summary": summary}


def _distributed_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _distributed_world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def _flatten_memory_breakdown_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    peak = int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)
    persistent = row.get("persistent_bytes", {})
    activation = row.get("activation_bytes", {})
    closure = row.get("closure_bytes", {})
    rows: list[dict[str, Any]] = []

    def add(memory_space: str, group: str, component: str, kind: str, value: int, method: str, accuracy: str) -> None:
        if int(value) <= 0:
            return
        rows.append(
            {
                "memory_space": memory_space,
                "group": group,
                "component": component,
                "kind": kind,
                "bytes": int(value),
                "method": method,
                "accuracy": accuracy,
            }
        )

    if isinstance(persistent, dict):
        for component, kinds in persistent.items():
            if not isinstance(kinds, dict):
                continue
            for kind, value in kinds.items():
                value_int = int(value or 0)
                if value_int <= 0:
                    continue
                if str(kind).endswith("_cpu") or str(kind).endswith("_cpu_pinned"):
                    add("CPU host", "host", str(component), str(kind), value_int, "exact tensor size", "exact")
                elif kind in {"weight", "frozen_weight", "buffer"}:
                    add("GPU HBM", "weights", str(component), str(kind), value_int, "exact tensor size", "exact")
                elif kind == "grad":
                    add("GPU HBM", "gradients", str(component), str(kind), value_int, "exact tensor size", "exact")
                elif kind == "optimizer_state":
                    add("GPU HBM", "optimizer", str(component), str(kind), value_int, "exact tensor size", "exact")
                else:
                    add("GPU HBM", "persistent", str(component), str(kind), value_int, "exact tensor size", "exact")

    allocated_known = sum(int(item["bytes"]) for item in rows if item.get("memory_space") == "GPU HBM")
    activation_items = [
        (str(component), int(value or 0))
        for component, value in (activation.items() if isinstance(activation, dict) else [])
        if int(value or 0) > 0
    ]
    activation_total = sum(value for _component, value in activation_items)
    activation_scale = 1.0
    if activation_total > 0:
        available = max(0, peak - allocated_known)
        if activation_total > available:
            activation_scale = float(available) / float(activation_total) if activation_total else 0.0
    for component, value in activation_items:
        scaled = int(round(value * activation_scale))
        method = "measured forward allocated delta"
        if activation_scale < 0.999:
            method = "measured forward allocated delta, scaled to peak closure"
        add("GPU HBM", "activations", component, "activation", scaled, method, "approximate")

    allocated_known = sum(int(item["bytes"]) for item in rows if item.get("memory_space") == "GPU HBM")
    framework_value = max(0, peak - allocated_known)
    if isinstance(closure, dict):
        framework_value = max(framework_value, int(closure.get("framework_temp_workspace") or 0))
        framework_value = min(framework_value, max(0, peak - allocated_known))
        reserved_unallocated = int(closure.get("allocator_reserved_unallocated") or 0)
    else:
        reserved_unallocated = 0
    add(
        "GPU HBM",
        "temp_workspace",
        "framework_temp_workspace",
        "temp_workspace",
        framework_value,
        "inferred residual to peak",
        "approximate",
    )
    add(
        "GPU reserved",
        "allocator",
        "allocator_reserved_unallocated",
        "reserved_unallocated",
        reserved_unallocated,
        "reserved - allocated snapshot",
        "exact allocator snapshot",
    )
    return rows, peak


def build_memory_breakdown_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"enabled": False, "breakdown_rows": [], "closure_ok": False, "closure_error_bytes": 0}
    candidates = [row for row in rows if not row.get("is_warmup")]
    if not candidates:
        candidates = rows
    selected = max(candidates, key=lambda row: int(row.get("peak_allocated_since_step_begin") or 0))
    breakdown_rows, peak = _flatten_memory_breakdown_row(selected)
    gpu_sum = sum(int(row.get("bytes") or 0) for row in breakdown_rows if row.get("memory_space") == "GPU HBM")
    closure_error = int(peak) - int(gpu_sum)
    return {
        "enabled": True,
        "schema_version": 1,
        "peak_hbm_bytes": int(peak),
        "selected_phase": selected.get("phase", ""),
        "selected_step": selected.get("step", 0),
        "rank_count": _distributed_world_size(),
        "breakdown_rows": breakdown_rows,
        "closure_ok": abs(closure_error) <= max(1024 * 1024, int(peak * 0.01)),
        "closure_error_bytes": closure_error,
        "reserved_bytes": int(selected.get("reserved_bytes") or 0),
        "allocated_bytes": int(selected.get("allocated_bytes") or 0),
        "notes": [
            "GPU HBM rows close to torch.cuda.max_memory_allocated-style allocated peak.",
            "GPU reserved allocator rows are reported separately from allocated HBM closure.",
        ],
    }
def _record_original(handle: LFTraceHandle, owner: Any, attr: str) -> Any:
    original = getattr(owner, attr)
    handle.originals.append((owner, attr, original))
    return original


def _stage(handle: LFTraceHandle, name: str):
    if handle.recorder is not None:
        return handle.recorder.stage(name, sync=handle.config.sync)
    return prof_range(name)


def _cuda_profiler_start_once(handle: LFTraceHandle) -> None:
    if getattr(handle, "_asym_lf_cuda_profiler_started", False):
        return
    setattr(handle, "_asym_lf_cuda_profiler_started", True)
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:
        warnings.warn(f"failed to start CUDA profiler capture: {exc}", RuntimeWarning)


def _cuda_profiler_stop_once(handle: LFTraceHandle) -> None:
    if getattr(handle, "_asym_lf_cuda_profiler_stopped", False):
        return
    setattr(handle, "_asym_lf_cuda_profiler_stopped", True)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as exc:
        warnings.warn(f"failed to stop CUDA profiler capture: {exc}", RuntimeWarning)


def _maybe_start_nsys_capture(handle: LFTraceHandle) -> None:
    if not handle.config.nsys_capture_range:
        return
    raw_step = int(getattr(handle, "_asym_lf_forward_raw_step", 0)) + 1
    setattr(handle, "_asym_lf_forward_raw_step", raw_step)
    start_step = max(int(handle.config.warmup_steps), 0) + 1
    if raw_step == start_step:
        _cuda_profiler_start_once(handle)


def _maybe_stop_nsys_capture(handle: LFTraceHandle) -> None:
    if not handle.config.nsys_capture_range:
        return
    raw_step = int(getattr(handle, "_asym_lf_backward_raw_step", 0)) + 1
    setattr(handle, "_asym_lf_backward_raw_step", raw_step)
    total_steps = int(handle.config.total_steps)
    if total_steps > 0 and raw_step >= total_steps:
        _cuda_profiler_stop_once(handle)


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
        if handle.memory_breakdown_profiler is not None and name == "lf.optimizer.step":
            handle.memory_breakdown_profiler.record_phase("before_optimizer_step", model=handle.model, optimizer=owner)
        with _range(handle, name):
            try:
                return original(*args, **kwargs)
            finally:
                if handle.memory_breakdown_profiler is not None and name == "lf.optimizer.step":
                    handle.memory_breakdown_profiler.record_phase("after_optimizer_step", model=handle.model, optimizer=owner)

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
        if handle.memory_breakdown_profiler is not None:
            handle.memory_breakdown_profiler.set_optimizer(optimizer)
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
            if handle.memory_breakdown_profiler is not None:
                handle.memory_breakdown_profiler.step_begin(
                    model=getattr(self, "model", None),
                    optimizer=getattr(self, "optimizer", None),
                )
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
            _maybe_start_nsys_capture(handle)
            with _range(handle, "step.forward"):
                try:
                    with prof_range("lf.forward_loss"):
                        result = original(self, *args, **kwargs)
                    if handle.memory_breakdown_profiler is not None:
                        handle.memory_breakdown_profiler.record_phase(
                            "after_forward",
                            model=getattr(self, "model", None),
                            optimizer=getattr(self, "optimizer", None),
                        )
                    return result
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
                    try:
                        result = original(self, *args, **kwargs)
                        if handle.memory_breakdown_profiler is not None:
                            handle.memory_breakdown_profiler.record_phase("after_backward", model=handle.model, optimizer=None)
                        return result
                    finally:
                        _maybe_stop_nsys_capture(handle)

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
    if handle.memory_breakdown_profiler is not None:
        handle.memory_breakdown_profiler.ensure_hooks(model)
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
    if config.memory_breakdown:
        handle.memory_breakdown_profiler = LFMemoryBreakdownProfiler(config)
    _patch_training_phases(handle)
    return handle


def uninstall_lf_trace(handle: LFTraceHandle) -> None:
    handle.restore()


__all__ = [
    "LFTraceConfig",
    "LFTraceHandle",
    "LFMemoryBreakdownProfiler",
    "build_memory_breakdown_summary",
    "install_lf_trace",
    "uninstall_lf_trace",
]
