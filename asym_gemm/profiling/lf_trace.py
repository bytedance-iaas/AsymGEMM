from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import re
import warnings
import weakref
from collections import Counter
from typing import Any, Callable, Iterator

import torch
from torch import nn

from asym_gemm.training.profile_ranges import current_profile_range, prof_range, scoped_name


_PATCH_ATTR = "_asym_lf_profile_wrapped"
_HOOK_ATTR = "_asym_lf_profile_hooks_installed"
_MEMORY_HOOK_ATTR = "_asym_lf_memory_breakdown_hooks_installed"

# Phase-boundary allocator release (agent/impls/fix_reserved.md). Under
# expandable segments, peak RESERVED is the union of each stream-pool's own
# high-water across phases; releasing free pages at phase boundaries makes each
# phase re-grow from ~allocated instead. Default OFF; enable per run with e.g.
# ASYM_EMPTY_CACHE_PHASES=forward_end,backward_end. Applied here (the source
# profiler wraps every backend identically) so scoreboards stay apples-to-apples.
_EMPTY_CACHE_PHASES = frozenset(
    token.strip()
    for token in os.environ.get("ASYM_EMPTY_CACHE_PHASES", "").split(",")
    if token.strip()
)


def _maybe_release_cached_blocks(phase: str) -> None:
    if phase not in _EMPTY_CACHE_PHASES:
        return
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass
# Stock MoE classes whose forward mutates a child's output in place — Llama4TextMoe does `out.add_(...)`
# on its shared_expert output. A full backward hook wraps a module's output in a custom autograd
# Function, and autograd forbids in-place edits of that wrapped output ("Output 0 of
# BackwardHookFunctionBackward is a view and is being modified inplace") -> crash in backward. So only
# when such a stock module is present (non-asym Llama4) do we skip BACKWARD hooks on the affected child
# subtree (shared_expert, including its inner down_proj Linear). The asym wrapper AsymLlama4Moe returns a
# fresh tensor (`out + routed`) and is safe -> the stock class is absent and its hooks are kept.
_INPLACE_MOE_CLASSES = {"Llama4TextMoe"}
_INPLACE_MOE_CHILD_MARKERS = ("shared_expert",)


def _has_inplace_moe(named_modules) -> bool:
    return any(type(module).__name__ in _INPLACE_MOE_CLASSES for _name, module in named_modules)


def _skip_backward_hook(module_name: str, has_inplace_moe: bool) -> bool:
    return has_inplace_moe and any(marker in module_name for marker in _INPLACE_MOE_CHILD_MARKERS)
_OTHER_SAVED_ACTIVATION_COMPONENT = "other_saved_activations"


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
    if name.endswith(".linear_attn"):
        return f"layers.{layer}.linear_attn"
    if name.endswith(".mlp.gate") or name.endswith(".feed_forward.router"):
        return f"layers.{layer}.router"
    if name.endswith(".mlp.experts") or name.endswith(".experts"):
        return f"layers.{layer}.mlp.experts"
    if name.endswith(".feed_forward.shared_expert"):
        return f"layers.{layer}.mlp"
    if name.endswith(".mlp"):
        return f"layers.{layer}.mlp"
    return None


def _filter_token(semantic_name: str) -> str:
    if semantic_name.endswith(".self_attn"):
        return "attention"
    if semantic_name.endswith(".linear_attn"):
        return "linear_attention"
    if semantic_name.endswith(".mlp.experts"):
        return "experts"
    if semantic_name.endswith(".mlp"):
        return "mlp"
    return semantic_name.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class LFTraceConfig:
    level: str = "stage"
    layers: str = "all"
    module_filter: str = "attention,linear_attention,router,mlp,experts,lora,optimizer"
    memory_attribution: bool = False
    memory_breakdown: bool = False
    memory_breakdown_interval: int = 1
    memory_breakdown_steps: str = ""
    memory_breakdown_modules: str = "attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,embed_tokens,norms,loss"
    memory_breakdown_output: str = "memory_breakdown"
    live_activation_details: bool = False
    live_activation_topk: int = 100
    external_memory_diagnostics: bool = False
    sync: bool = False
    nsys_capture_range: bool = False
    warmup_steps: int = 0
    total_steps: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "LFTraceConfig":
        def env_int(name: str, default: int = 0) -> int:
            try:
                return max(int(env.get(name, str(default))), 0)
            except ValueError:
                return max(int(default), 0)

        return cls(
            level=env.get("ASYM_GEMM_LF_PROFILE_LEVEL", "stage").strip().lower(),
            layers=env.get("ASYM_GEMM_LF_PROFILE_LAYERS", "all").strip().lower(),
            module_filter=env.get(
                "ASYM_GEMM_LF_PROFILE_MODULE_FILTER",
                "attention,linear_attention,router,mlp,experts,lora,optimizer",
            ),
            memory_attribution=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_MEMORY_ATTRIBUTION", "0")),
            memory_breakdown=_parse_bool(env.get("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN", "0")),
            memory_breakdown_interval=max(env_int("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_INTERVAL"), 1),
            memory_breakdown_steps=env.get("ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_STEPS", "").strip(),
            memory_breakdown_modules=env.get(
                "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_MODULES",
                "attention,linear_attention,router,mlp,experts,shared_experts,lora,embedding,embed_tokens,norms,loss",
            ),
            memory_breakdown_output=env.get(
                "ASYM_GEMM_LF_PROFILE_MEMORY_BREAKDOWN_OUTPUT",
                "memory_breakdown",
            ).strip()
            or "memory_breakdown",
            live_activation_details=_parse_bool(
                env.get("ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_DETAILS", "0")
            ),
            live_activation_topk=max(env_int("ASYM_GEMM_LF_PROFILE_LIVE_ACTIVATION_TOPK", 100), 1),
            external_memory_diagnostics=_parse_bool(
                env.get("ASYM_GEMM_LF_PROFILE_EXTERNAL_MEMORY", "0")
            ),
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
        # Enable backward NVTX ranges whenever forward module ranges are on (adds backward hooks).
        return self.level in {"module", "op", "deep"} or "backward" in self.module_filter_set

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
    prepared_optimizer: Any | None = None
    optimizer: Any | None = None

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
        if not self.config.memory_attribution and self.memory_breakdown_profiler is None:
            yield
            return
        if self.config.memory_attribution:
            self.saved_tensor_tracker = SavedTensorTracker()

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            result = tensor
            if self.saved_tensor_tracker is not None:
                result = self.saved_tensor_tracker.pack(result)
            if self.memory_breakdown_profiler is not None:
                result = self.memory_breakdown_profiler.saved_tensor_pack(result)
            return result

        def unpack(tensor: torch.Tensor) -> torch.Tensor:
            if self.memory_breakdown_profiler is not None:
                self.memory_breakdown_profiler.saved_tensor_unpack(tensor)
            if self.saved_tensor_tracker is not None:
                return self.saved_tensor_tracker.unpack(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
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


def _is_router_name(lower: str) -> bool:
    return (
        "router" in lower
        or re.search(r"(?:^|\.)mlp\.gate(?:\.|$)", lower) is not None
        or re.search(r"(?:^|\.)feed_forward\.router(?:\.|$)", lower) is not None
    )


def _is_attention_name(lower: str) -> bool:
    return "self_attn" in lower or any(part in lower for part in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _is_linear_attention_name(lower: str) -> bool:
    if "linear_attn" in lower or "gateddeltanet" in lower:
        return True
    gdn_leaves = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    return any(part in lower for part in gdn_leaves)


_VISION_PATH_MARKERS = (
    ".visual.",
    ".vision.",
    ".vision_model.",
    ".vision_tower.",
    ".multi_modal_projector.",
)


def _is_vision_name(lower: str) -> bool:
    path = f".{str(lower).lower().strip('.')}."
    return any(marker in path for marker in _VISION_PATH_MARKERS)


def _is_shared_expert_name(lower: str) -> bool:
    return "shared_expert" in lower or "shared_experts" in lower


def _is_routed_expert_name(lower: str) -> bool:
    return re.search(r"(?:^|\.)experts(?:\.|$)", lower) is not None or "routed_expert" in lower


def _is_dense_mlp_name(lower: str) -> bool:
    return (
        re.search(r"(?:^|\.)mlp(?:\.|$)", lower) is not None
        or "dense_mlp" in lower
        or "mlp_dense" in lower
        or any(part in lower for part in ("gate_proj", "up_proj", "down_proj"))
    )


def _component_from_param_name(name: str) -> str:
    try:
        from asym_gemm.integrations.lf import classify_lf_component

        component = classify_lf_component(name)
        if component != "other":
            return component
    except Exception:
        pass

    lower = name.lower()
    if _is_vision_name(lower):
        return "vision"
    if _is_linear_attention_name(lower):
        return "linear_attention"
    if "lora" in lower:
        if _is_linear_attention_name(lower):
            return "linear_attention"
        if _is_shared_expert_name(lower):
            return "shared_experts"
        if _is_attention_name(lower):
            return "attention"
        if _is_router_name(lower):
            return "router"
        if _is_routed_expert_name(lower):
            return "routed_experts"
        if _is_dense_mlp_name(lower):
            return "mlp_dense"
        return "lora"
    if "embed" in lower:
        return "embed_tokens"
    if "lm_head" in lower:
        return "lm_head"
    if _is_attention_name(lower):
        return "attention"
    if _is_router_name(lower):
        return "router"
    if _is_shared_expert_name(lower):
        return "shared_experts"
    if _is_routed_expert_name(lower):
        return "routed_experts"
    if _is_dense_mlp_name(lower):
        return "mlp_dense"
    if "norm" in lower:
        return "norms"
    return "other"


def _component_from_module_name(name: str) -> str | None:
    try:
        from asym_gemm.integrations.lf import classify_lf_component

        component = classify_lf_component(name)
        if component != "other":
            return component
    except Exception:
        pass

    lower = name.lower()
    if not lower:
        return None
    if _is_vision_name(lower):
        return "vision"
    if _is_router_name(lower):
        return "router"
    if lower.endswith(".self_attn") or lower.endswith(".attention") or lower.endswith(".self_attention"):
        return "attention"
    if lower.endswith(".linear_attn") or _is_linear_attention_name(lower):
        return "linear_attention"
    if _is_shared_expert_name(lower):
        return "shared_experts"
    if _is_routed_expert_name(lower):
        return "routed_experts"
    if lower.endswith(".mlp") or lower.endswith(".dense_mlp") or lower.endswith(".mlp_dense"):
        return "mlp_dense"
    if lower.endswith(".embed_tokens") or lower.endswith(".embed_in") or lower.endswith(".wte"):
        return "embed_tokens"
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
    if component == "linear_attention":
        return "linear_attention"
    if component == "shared_experts":
        return "shared_experts"
    if component == "mlp_dense":
        return "mlp"
    if component in {"embed_tokens", "embedding"}:
        return "embedding"
    if component in {"lora_attention", "lora_mlp", "lora_experts"}:
        return "lora"
    if component in {"lm_head", "loss_logits"}:
        return "loss"
    return component


def _component_from_range_name(name: str | None) -> str | None:
    lower = str(name or "").lower()
    if not lower:
        return None
    if _is_vision_name(lower):
        return "vision"
    if "forward_loss" in lower or "compute_loss" in lower or "cross_entropy" in lower or "loss" in lower:
        return "loss"
    if "lm_head" in lower:
        return "lm_head"
    if "embed" in lower:
        return "embed_tokens"
    if "norm" in lower:
        return "norms"
    if _is_attention_name(lower):
        return "attention"
    if _is_linear_attention_name(lower):
        return "linear_attention"
    if _is_router_name(lower) or ".router" in lower:
        return "router"
    if _is_shared_expert_name(lower):
        return "shared_experts"
    if _is_routed_expert_name(lower) or "expert_policy" in lower or ".expert" in lower:
        return "routed_experts"
    if _is_dense_mlp_name(lower):
        return "mlp_dense"
    if "lora" in lower:
        return "lora"
    return None


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[str, int, int, int, str]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        str(tensor.dtype),
    )


def _saved_tensor_storage_key(tensor: torch.Tensor) -> tuple[str, int, int, str] | None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return None
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()), str(tensor.dtype))


_PHASE_PRIORITY = {
    "after_backward": 60,
    "before_optimizer_step": 50,
    "after_forward": 40,
    "after_optimizer_step": 30,
    "step_begin": 0,
}


def _memory_breakdown_selection_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0),
        _PHASE_PRIORITY.get(str(row.get("phase") or ""), 10),
        int(row.get("allocated_bytes") or 0),
        int(row.get("step") or 0),
    )


def _saved_activation_component(component: Any) -> str:
    component_str = str(component or "").strip()
    if component_str in {"", "unknown_saved_activation"}:
        return _OTHER_SAVED_ACTIVATION_COMPONENT
    return component_str


def _saved_activation_mapping(row: dict[str, Any]) -> dict[str, int]:
    saved_activation = row.get("saved_activation_bytes_at_peak", {})
    if not isinstance(saved_activation, dict) or not saved_activation:
        saved_activation = row.get("saved_activation_bytes", {})
    if not isinstance(saved_activation, dict):
        return {}
    result: dict[str, int] = {}
    for component, value in saved_activation.items():
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        result[component_name] = result.get(component_name, 0) + value_int
    return result


def _saved_activation_total(row: dict[str, Any]) -> int:
    return sum(_saved_activation_mapping(row).values())


def _activation_mapping(row: dict[str, Any]) -> dict[str, int]:
    activation = row.get("live_activation_bytes_at_peak", {})
    if not isinstance(activation, dict) or not activation:
        activation = row.get("live_activation_bytes", {})
    if not isinstance(activation, dict):
        return {}
    result: dict[str, int] = {}
    for component, value in activation.items():
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        result[component_name] = result.get(component_name, 0) + value_int
    return result


def _activation_total(row: dict[str, Any]) -> int:
    return sum(_activation_mapping(row).values())


def _peak_growth_mapping(row: dict[str, Any]) -> dict[str, int]:
    peak_growth = row.get("peak_growth_bytes_at_peak", {})
    if not isinstance(peak_growth, dict) or not peak_growth:
        peak_growth = row.get("peak_growth_bytes", {})
    if not isinstance(peak_growth, dict):
        return {}
    result: dict[str, int] = {}
    for component, value in peak_growth.items():
        value_int = int(value or 0)
        if value_int <= 0:
            continue
        component_name = _saved_activation_component(component)
        result[component_name] = result.get(component_name, 0) + value_int
    return result


def _persistent_gpu_bytes_from_row(row: dict[str, Any]) -> int:
    persistent = row.get("persistent_bytes", {})
    if not isinstance(persistent, dict):
        return 0
    total = 0
    for kinds in persistent.values():
        if not isinstance(kinds, dict):
            continue
        for kind, value in kinds.items():
            if not str(kind).endswith("_cpu") and not str(kind).endswith("_cpu_pinned"):
                total += int(value or 0)
    return total


def _row_peak_allocated(row: dict[str, Any]) -> int:
    return int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)


def _row_attributed_gpu_bytes(row: dict[str, Any]) -> int:
    persistent_gpu = _persistent_gpu_bytes_from_row(row)
    saved_activation_gpu = _saved_activation_total(row)
    activation_gpu = _activation_total(row)
    known_allocated = persistent_gpu + saved_activation_gpu + activation_gpu
    peak_allocated = _row_peak_allocated(row)
    workspace_residual = max(0, peak_allocated - known_allocated)
    closure = row.get("closure_bytes", {})
    if isinstance(closure, dict):
        workspace_residual = min(workspace_residual, max(0, int(closure.get("unattributed_allocated_peak") or 0)))
    if not _peak_growth_mapping(row):
        workspace_residual = 0
    return known_allocated + workspace_residual


def _row_attribution_fraction(row: dict[str, Any]) -> float:
    peak = _row_peak_allocated(row)
    if peak <= 0:
        return 0.0
    return min(float(_row_attributed_gpu_bytes(row)), float(peak)) / float(peak)


def _select_memory_breakdown_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(rows, key=_memory_breakdown_selection_key)
    selected_peak = _row_peak_allocated(selected)
    if selected_peak <= 0 or _row_attribution_fraction(selected) >= 0.25:
        return selected

    same_step = [row for row in rows if row.get("step") == selected.get("step")]
    alternatives = [
        row
        for row in same_step
        if _row_peak_allocated(row) >= int(selected_peak * 0.5)
        and _row_attributed_gpu_bytes(row) > _row_attributed_gpu_bytes(selected)
    ]
    if not alternatives:
        return selected
    return max(
        alternatives,
        key=lambda row: (
            _row_attribution_fraction(row),
            _row_attributed_gpu_bytes(row),
            _row_peak_allocated(row),
            _PHASE_PRIORITY.get(str(row.get("phase") or ""), 10),
        ),
    )


def _actual_peak_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=_memory_breakdown_selection_key)


def _external_device_memory_used_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        value = torch.cuda.device_memory_used(torch.cuda.current_device())
    except Exception:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _ActivationRecord:
    component: str
    module_name: str
    dtype: str
    shape: list[int]
    bytes: int
    refs: list[weakref.ReferenceType[torch.Tensor]] = field(default_factory=list)


class LFMemoryBreakdownProfiler:
    def __init__(self, config: LFTraceConfig) -> None:
        self.config = config
        self.enabled = bool(config.memory_breakdown)
        self.interval = max(int(config.memory_breakdown_interval), 1)
        self.step_filter = _parse_step_set(config.memory_breakdown_steps)
        self.module_filter = _csv_set(config.memory_breakdown_modules) or {
            "attention",
            "router",
            "mlp",
            "experts",
            "shared_experts",
            "lora",
            "embedding",
            "embed_tokens",
            "loss",
        }
        if "embedding" in self.module_filter:
            self.module_filter.add("embed_tokens")
        if "embed_tokens" in self.module_filter:
            self.module_filter.add("embedding")
        self.rows: list[dict[str, Any]] = []
        self._step = 0
        self._active = False
        self._component_stack: list[str] = []
        self._saved_refcounts: Counter[tuple[str, int, int, str]] = Counter()
        self._saved_components: dict[tuple[str, int, int, str], str] = {}
        self._saved_bytes: dict[tuple[str, int, int, str], int] = {}
        self._persistent_storage_keys: set[tuple[str, int, int, str]] = set()
        self._storage_components: dict[tuple[str, int, int, str], str] = {}
        self._activation_refs: dict[tuple[str, int, int, str], list[weakref.ReferenceType[torch.Tensor]]] = {}
        self._activation_components: dict[tuple[str, int, int, str], str] = {}
        self._activation_bytes: dict[tuple[str, int, int, str], int] = {}
        self._activation_records: dict[tuple[str, int, int, str], _ActivationRecord] = {}
        self._peak_growth_bytes: dict[str, int] = {}
        self._peak_growth_bytes_at_peak: dict[str, int] = {}
        self._saved_activation_bytes_at_peak: dict[str, int] = {}
        self._activation_bytes_at_peak: dict[str, int] = {}
        self._activation_detail_rows_at_peak: list[dict[str, Any]] = []
        self._saved_activation_peak_allocated = 0
        self._activation_peak_allocated = 0
        self._current_peak_allocated = 0
        self._current_peak_reserved = 0
        self._external_memory_at_peak: dict[str, int] = {}
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
        if self._model is not None and getattr(self._model, _MEMORY_HOOK_ATTR, False):
            try:
                setattr(self._model, _MEMORY_HOOK_ATTR, False)
            except Exception:
                pass

    def set_optimizer(self, optimizer: Any | None) -> None:
        if optimizer is not None:
            self._optimizer = _unwrap_optimizer_for_introspection(optimizer)

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
        has_inplace_moe = _has_inplace_moe(modules)
        for module_name, module in modules:
            component = _component_from_module_name(module_name)
            if component is None:
                continue
            if component == "mlp_dense" and (
                f"{module_name}.experts" in module_names or f"{module_name}.mlp.experts" in module_names
            ):
                # MoE parent modules own routing/combination tensors outside the child expert/router modules.
                component = "routed_experts"
            if _component_filter_token(component) not in self.module_filter:
                continue
            module_label = _semantic_module_name(module_name) or module_name

            def forward_pre(_module: nn.Module, _args: tuple[Any, ...], *, _component: str = component) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                self._component_stack.append(_component)

            def forward_post(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _output: Any,
                *,
                _component: str = component,
                _module_name: str = module_label,
            ) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                after = self._record_component_peak(_component)
                self._remember_tensor_components(_output, _component, _module_name)
                self._capture_saved_activation_peak(after)
                if self._component_stack:
                    self._component_stack.pop()

            self._hooks.append(module.register_forward_pre_hook(forward_pre))
            self._hooks.append(module.register_forward_hook(forward_post))

            # Mirror the forward hooks on backward so backward peak growth attributes to its
            # component; defensive so a hook error can never break the backward pass.
            def backward_pre(_module: nn.Module, _grad_output: Any, *, _component: str = component) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                self._component_stack.append(_component)

            def backward_post(_module: nn.Module, _grad_input: Any, _grad_output: Any, *, _component: str = component) -> None:
                if not self._active or not torch.cuda.is_available():
                    return
                try:
                    self._capture_saved_activation_peak(self._record_component_peak(_component))
                except Exception:
                    pass
                finally:
                    if self._component_stack:
                        self._component_stack.pop()

            if not _skip_backward_hook(module_name, has_inplace_moe):
                try:
                    self._hooks.append(module.register_full_backward_pre_hook(backward_pre))
                    self._hooks.append(module.register_full_backward_hook(backward_post))
                except Exception:
                    pass
        setattr(model, _MEMORY_HOOK_ATTR, True)

    def _record_component_peak(self, component: str) -> int:
        after = int(torch.cuda.memory_allocated())
        try:
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        except RuntimeError:
            peak_allocated = after
            peak_reserved = int(torch.cuda.memory_reserved())
        self._update_peak_values(after, int(torch.cuda.memory_reserved()), peak_allocated, peak_reserved, component=component)
        return after

    def step_begin(self, model: nn.Module | None = None, optimizer: Any | None = None) -> None:
        if model is not None:
            self._model = model
            self.ensure_hooks(model)
        self.set_optimizer(optimizer)
        self._step += 1
        self._active = self.should_record_step(self._step)
        self._component_stack = []
        self._saved_refcounts.clear()
        self._saved_components.clear()
        self._saved_bytes.clear()
        self._persistent_storage_keys.clear()
        self._storage_components.clear()
        self._activation_refs.clear()
        self._activation_components.clear()
        self._activation_bytes.clear()
        self._activation_records.clear()
        self._peak_growth_bytes.clear()
        self._peak_growth_bytes_at_peak.clear()
        self._saved_activation_bytes_at_peak = {}
        self._activation_bytes_at_peak = {}
        self._activation_detail_rows_at_peak = []
        self._saved_activation_peak_allocated = 0
        self._activation_peak_allocated = 0
        self._external_memory_at_peak = {}
        if self._active and torch.cuda.is_available():
            if self.config.sync:
                torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
        allocated, reserved, peak_allocated, peak_reserved = self._snapshot_values()
        self._current_peak_allocated = max(allocated, peak_allocated)
        self._current_peak_reserved = max(reserved, peak_reserved)
        self._capture_saved_activation_peak(allocated)
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
        self._update_peak_values(allocated, reserved, peak_allocated, peak_reserved)
        self._capture_saved_activation_peak(allocated)
        persistent = self._collect_persistent_bytes(self._model, self._optimizer)
        saved_activation = dict(sorted(self._live_saved_activation_bytes().items()))
        saved_activation_at_peak = dict(sorted(self._saved_activation_bytes_at_peak.items()))
        activation = dict(sorted(self._live_activation_bytes().items()))
        activation_at_peak = dict(sorted(self._activation_bytes_at_peak.items()))
        activation_detail_rows = self._live_activation_detail_rows()
        activation_detail_rows_at_peak = list(self._activation_detail_rows_at_peak)
        external_memory = dict(self._external_memory_at_peak) or self._external_memory_snapshot(self._current_peak_reserved)
        row = {
            "schema_version": 2,
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
            # Phase-local peak (max since the last stage reset); exact for after_forward/after_backward.
            "peak_allocated_within_phase": int(peak_allocated),
            "peak_reserved_within_phase": int(peak_reserved),
            "persistent_bytes": persistent,
            "saved_activation_bytes": saved_activation,
            "saved_activation_bytes_at_peak": saved_activation_at_peak,
            "live_activation_bytes": activation,
            "live_activation_bytes_at_peak": activation_at_peak,
            "live_activation_detail_rows": activation_detail_rows,
            "live_activation_detail_rows_at_peak": activation_detail_rows_at_peak,
            "saved_activation_peak_allocated_bytes": self._saved_activation_peak_allocated,
            "activation_peak_allocated_bytes": self._activation_peak_allocated,
            "peak_growth_bytes": dict(sorted(self._peak_growth_bytes.items())),
            "peak_growth_bytes_at_peak": dict(sorted(self._peak_growth_bytes_at_peak.items())),
            "external_memory": external_memory,
            "closure_bytes": self._closure_bytes(
                self._current_peak_allocated,
                self._current_peak_reserved,
                persistent,
                saved_activation_at_peak,
                activation_at_peak,
                external_memory,
            ),
            "methods": {
                "persistent_bytes": "exact tensor size",
                "saved_activation_bytes": "autograd saved tensor live unique storage",
                "saved_activation_bytes_at_peak": "autograd saved tensor live unique storage at best observed live allocation point",
                "live_activation_bytes": "live module output tensor unique storage",
                "live_activation_bytes_at_peak": "live module output tensor unique storage at best observed live allocation point, excluding persistent and saved-for-backward storages",
                "peak_growth_bytes": "component responsible for raising the step CUDA allocated peak",
                "peak_growth_bytes_at_peak": "component peak-growth attribution at best observed live allocation point",
                "unattributed_allocated_peak": "residual to peak allocated",
                "allocator_reserved_unallocated": "peak reserved - peak allocated",
                "external_cuda_or_driver": "device used - PyTorch peak reserved",
            },
        }
        self.rows.append(row)

    def _live_saved_activation_bytes(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for key, count in self._saved_refcounts.items():
            if count <= 0:
                continue
            component = self._saved_components.get(key, _OTHER_SAVED_ACTIVATION_COMPONENT)
            result[component] = result.get(component, 0) + int(self._saved_bytes.get(key, key[2]))
        return result

    def _remember_live_activation(self, tensor: torch.Tensor, component: str, module_name: str = "") -> None:
        key = _saved_tensor_storage_key(tensor)
        if key is None:
            return
        self._storage_components[key] = component
        try:
            tensor_ref = weakref.ref(tensor)
        except TypeError:
            return
        refs = self._activation_refs.setdefault(key, [])
        refs.append(tensor_ref)
        self._activation_components.setdefault(key, component)
        self._activation_bytes[key] = key[2]
        if self.config.live_activation_details:
            record = self._activation_records.get(key)
            if record is None:
                record = _ActivationRecord(
                    component=component,
                    module_name=str(module_name or ""),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    shape=[int(dim) for dim in tensor.shape],
                    bytes=int(key[2]),
                )
                self._activation_records[key] = record
            elif not record.module_name and module_name:
                record.module_name = str(module_name)
            record.refs.append(tensor_ref)

    def _live_activation_bytes(self) -> dict[str, int]:
        result: dict[str, int] = {}
        dead_keys: list[tuple[str, int, int, str]] = []
        for key, refs in self._activation_refs.items():
            live_refs = [ref for ref in refs if ref() is not None]
            if not live_refs:
                dead_keys.append(key)
                continue
            self._activation_refs[key] = live_refs
            if key in self._persistent_storage_keys:
                continue
            if self._saved_refcounts.get(key, 0) > 0:
                continue
            component = self._activation_components.get(key) or self._storage_components.get(
                key, _OTHER_SAVED_ACTIVATION_COMPONENT
            )
            result[component] = result.get(component, 0) + int(self._activation_bytes.get(key, key[2]))
        for key in dead_keys:
            self._activation_refs.pop(key, None)
            self._activation_components.pop(key, None)
            self._activation_bytes.pop(key, None)
            self._activation_records.pop(key, None)
        return result

    def _live_activation_detail_rows(self) -> list[dict[str, Any]]:
        if not self.config.live_activation_details:
            return []
        rows: list[dict[str, Any]] = []
        dead_keys: list[tuple[str, int, int, str]] = []
        for key, record in self._activation_records.items():
            live_refs = [ref for ref in record.refs if ref() is not None]
            if not live_refs:
                dead_keys.append(key)
                continue
            record.refs = live_refs
            if key in self._persistent_storage_keys:
                continue
            if self._saved_refcounts.get(key, 0) > 0:
                continue
            rows.append(
                {
                    "component": record.component,
                    "module_name": record.module_name,
                    "dtype": record.dtype,
                    "shape": list(record.shape),
                    "bytes": int(record.bytes),
                    "ref_count": len(live_refs),
                    "device": key[0],
                    "storage_bytes": int(key[2]),
                }
            )
        for key in dead_keys:
            self._activation_records.pop(key, None)
        rows.sort(key=lambda row: int(row.get("bytes") or 0), reverse=True)
        return rows[: max(int(self.config.live_activation_topk), 1)]

    def _remember_tensor_components(self, value: Any, component: str, module_name: str = "") -> None:
        if isinstance(value, torch.Tensor):
            self._remember_live_activation(value, component, module_name)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._remember_tensor_components(item, component, module_name)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._remember_tensor_components(item, component, module_name)

    def _capture_saved_activation_peak(self, allocated: int | None = None) -> None:
        if not self._active:
            return
        if allocated is None:
            allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        live_saved_activation = self._live_saved_activation_bytes()
        live_activation = self._live_activation_bytes()
        live_saved_bytes = sum(max(int(value), 0) for value in live_saved_activation.values())
        live_activation_bytes = sum(max(int(value), 0) for value in live_activation.values())
        saved_peak_bytes = sum(max(int(value), 0) for value in self._saved_activation_bytes_at_peak.values())
        activation_peak_bytes = sum(max(int(value), 0) for value in self._activation_bytes_at_peak.values())
        if int(allocated) > int(self._saved_activation_peak_allocated) or (
            int(allocated) == int(self._saved_activation_peak_allocated)
            and (live_saved_bytes + live_activation_bytes) > (saved_peak_bytes + activation_peak_bytes)
        ):
            self._saved_activation_peak_allocated = int(allocated)
            self._activation_peak_allocated = int(allocated)
            self._saved_activation_bytes_at_peak = live_saved_activation
            self._activation_bytes_at_peak = live_activation
            self._activation_detail_rows_at_peak = self._live_activation_detail_rows()
            self._peak_growth_bytes_at_peak = dict(self._peak_growth_bytes)
            self._external_memory_at_peak = self._external_memory_snapshot(self._current_peak_reserved)

    def _external_memory_snapshot(self, peak_reserved: int | None = None) -> dict[str, int]:
        if not self.config.external_memory_diagnostics:
            return {}
        device_used = _external_device_memory_used_bytes()
        if device_used is None:
            return {}
        reserved = int(peak_reserved if peak_reserved is not None else torch.cuda.memory_reserved())
        return {
            "device_memory_used_bytes": int(device_used),
            "external_cuda_or_driver_bytes": max(0, int(device_used) - int(reserved)),
        }

    def _update_peak_values(
        self,
        allocated: int,
        reserved: int,
        peak_allocated: int,
        peak_reserved: int,
        *,
        component: str | None = None,
    ) -> None:
        observed_peak_allocated = max(int(allocated), int(peak_allocated))
        observed_peak_reserved = max(int(reserved), int(peak_reserved))
        if observed_peak_allocated > self._current_peak_allocated and self._current_peak_allocated > 0:
            owner = (
                component
                or (self._component_stack[-1] if self._component_stack else None)
                or _component_from_range_name(current_profile_range())
                or "source_runtime"
            )
            delta = observed_peak_allocated - int(self._current_peak_allocated)
            self._peak_growth_bytes[owner] = self._peak_growth_bytes.get(owner, 0) + int(delta)
        self._current_peak_allocated = max(self._current_peak_allocated, observed_peak_allocated)
        self._current_peak_reserved = max(self._current_peak_reserved, observed_peak_reserved)

    def _refresh_peak_snapshot(self) -> None:
        allocated, reserved, peak_allocated, peak_reserved = self._snapshot_values()
        component = self._component_stack[-1] if self._component_stack else None
        self._update_peak_values(allocated, reserved, peak_allocated, peak_reserved, component=component)
        self._capture_saved_activation_peak(allocated)

    def saved_tensor_pack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._active:
            return tensor
        key = _saved_tensor_storage_key(tensor)
        if key is None:
            return tensor
        if key in self._persistent_storage_keys:
            return tensor
        component = self._component_stack[-1] if self._component_stack else (
            self._storage_components.get(key)
            or _component_from_range_name(current_profile_range())
            or _OTHER_SAVED_ACTIVATION_COMPONENT
        )
        self._saved_refcounts[key] += 1
        self._saved_components.setdefault(key, component)
        self._saved_bytes[key] = key[2]
        self._refresh_peak_snapshot()
        return tensor

    def saved_tensor_unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._active:
            return tensor
        key = _saved_tensor_storage_key(tensor)
        if key is None:
            return tensor
        self._refresh_peak_snapshot()
        if key in self._saved_refcounts:
            self._saved_refcounts[key] -= 1
            if self._saved_refcounts[key] <= 0:
                self._saved_refcounts.pop(key, None)
                self._saved_components.pop(key, None)
                self._saved_bytes.pop(key, None)
        return tensor

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

        seen_storages: set[tuple[str, int, int, str]] = set()
        persistent_storage_keys: set[tuple[str, int, int, str]] = set()
        cpu_master_storage_keys: set[tuple[str, int, int, str]] = set()

        def storage_key_for(tensor: torch.Tensor) -> tuple[str, int, int, str]:
            try:
                storage = tensor.untyped_storage()
                storage_bytes = int(storage.nbytes())
                return (str(tensor.device), int(storage.data_ptr()), storage_bytes, str(tensor.dtype))
            except Exception:
                storage_bytes = _tensor_bytes(tensor)
                return (str(tensor.device), id(tensor), storage_bytes, str(tensor.dtype))

        def add(component: str, kind: str, tensor: torch.Tensor) -> tuple[str, int, int, str]:
            storage_key = storage_key_for(tensor)
            storage_bytes = int(storage_key[2])
            persistent_storage_keys.add(storage_key)
            if storage_key in seen_storages:
                return storage_key
            seen_storages.add(storage_key)
            space = _device_space(tensor.device)
            key = kind if space == "GPU HBM" else f"{kind}_cpu"
            bucket = result.setdefault(component, {})
            bucket[key] = bucket.get(key, 0) + storage_bytes
            if space != "GPU HBM" and tensor.is_pinned():
                bucket[f"{kind}_cpu_pinned"] = bucket.get(f"{kind}_cpu_pinned", 0) + storage_bytes
            return storage_key

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
            for module_name, module in model.named_modules():
                component = _component_from_module_name(module_name) or _component_from_param_name(module_name)
                for attr in ("host_weight", "weight_host"):
                    host = getattr(module, attr, None)
                    if getattr(host, "is_paged", False):
                        continue  # NVMe-paged: no persistent host bytes; .tensor would fetch from NVMe
                    tensor = getattr(host, "tensor", None)
                    if isinstance(tensor, torch.Tensor):
                        add(component, "host_weight", tensor)

        if optimizer is not None:
            cpu_param_names: dict[int, str] = {}
            name_map = getattr(optimizer, "asym_cpu_param_name_map", None)
            if callable(name_map):
                try:
                    cpu_param_names = {int(key): str(value) for key, value in name_map().items()}
                except Exception:
                    cpu_param_names = {}
            master_params = getattr(optimizer, "asym_cpu_master_params", None)
            if callable(master_params):
                try:
                    for cpu_param in master_params():
                        if isinstance(cpu_param, torch.Tensor):
                            name = cpu_param_names.get(id(cpu_param), "optimizer")
                            component = _component_from_param_name(name)
                            cpu_master_storage_keys.add(add(component, "cpu_master_weight", cpu_param))
                except Exception:
                    pass
            grad_offload_buffer = getattr(optimizer, "asym_cpu_adamw_grad_offload_buffer", None)
            if callable(grad_offload_buffer):
                try:
                    grad_buffer = grad_offload_buffer()
                except Exception:
                    grad_buffer = None
                if isinstance(grad_buffer, torch.Tensor):
                    add("optimizer", "offloaded_grad", grad_buffer)
            try:
                state_items = list(optimizer.state.items())
            except Exception:
                state_items = []
            for param, state in state_items:
                component = _component_from_param_name(
                    param_names.get(id(param), cpu_param_names.get(id(param), "optimizer"))
                )
                if not isinstance(state, dict):
                    continue
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        if storage_key_for(value) in cpu_master_storage_keys:
                            continue
                        add(component, "optimizer_state", value)
                    elif isinstance(value, dict):
                        for nested in value.values():
                            if isinstance(nested, torch.Tensor):
                                if storage_key_for(nested) in cpu_master_storage_keys:
                                    continue
                                add(component, "optimizer_state", nested)
        self._persistent_storage_keys = persistent_storage_keys
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
        peak_reserved: int,
        persistent: dict[str, dict[str, int]],
        saved_activation: dict[str, int],
        activation: dict[str, int],
        external_memory: dict[str, int] | None = None,
    ) -> dict[str, int]:
        persistent_gpu = self._persistent_gpu_bytes(persistent)
        saved_activation_gpu = sum(max(int(value), 0) for value in saved_activation.values())
        activation_gpu = sum(max(int(value), 0) for value in activation.values())
        known_allocated = persistent_gpu + saved_activation_gpu + activation_gpu
        return {
            "unattributed_allocated_peak": max(0, int(peak_allocated) - known_allocated),
            "allocator_reserved_unallocated": max(0, int(peak_reserved) - int(peak_allocated)),
            "external_cuda_or_driver": int((external_memory or {}).get("external_cuda_or_driver_bytes") or 0),
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


def _flatten_memory_breakdown_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    peak_allocated = int(row.get("peak_allocated_since_step_begin") or row.get("allocated_bytes") or 0)
    peak_reserved = int(row.get("peak_reserved_since_step_begin") or row.get("reserved_bytes") or 0)
    peak_reserved = max(peak_reserved, peak_allocated)
    persistent = row.get("persistent_bytes", {})
    saved_activation = _saved_activation_mapping(row)
    activation = _activation_mapping(row)
    peak_growth = _peak_growth_mapping(row)
    closure = row.get("closure_bytes", {})
    external_memory = row.get("external_memory", {})
    rows: list[dict[str, Any]] = []

    def add(
        memory_space: str,
        group: str,
        component: str,
        kind: str,
        value: int,
        method: str,
        accuracy: str,
        *,
        keep_zero: bool = False,
    ) -> None:
        if int(value) < 0 or (int(value) == 0 and not keep_zero):
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
                    space = "CPU pinned subset" if str(kind).endswith("_cpu_pinned") else "CPU host"
                    add(space, "host", str(component), str(kind), value_int, "unique tensor storage size", "exact")
                elif kind in {"weight", "frozen_weight", "buffer"}:
                    add("GPU HBM", "weights", str(component), str(kind), value_int, "unique tensor storage size", "exact")
                elif kind == "grad":
                    add("GPU HBM", "gradients", str(component), str(kind), value_int, "unique tensor storage size", "exact")
                elif kind == "optimizer_state":
                    add("GPU HBM", "optimizer", str(component), str(kind), value_int, "unique tensor storage size", "exact")
                else:
                    add("GPU HBM", "persistent", str(component), str(kind), value_int, "unique tensor storage size", "exact")

    saved_activation_items = [
        (str(component), int(value or 0))
        for component, value in (saved_activation.items() if isinstance(saved_activation, dict) else [])
        if int(value or 0) > 0
    ]
    for component, value in saved_activation_items:
        add(
            "GPU HBM",
            "saved_activations",
            component,
            "saved_activation",
            value,
            "autograd saved tensor live unique storage at best observed live allocation point",
            "exact live storage",
        )

    activation_items = [
        (str(component), int(value or 0))
        for component, value in (activation.items() if isinstance(activation, dict) else [])
        if int(value or 0) > 0
    ]
    for component, value in activation_items:
        add(
            "GPU HBM",
            "saved_activations",
            component,
            "live_activation",
            value,
            "live module output tensor unique storage at best observed live allocation point",
            "exact live storage",
        )

    known_before_workspace = sum(int(item["bytes"]) for item in rows if item.get("memory_space") == "GPU HBM")
    workspace_residual = max(0, peak_allocated - known_before_workspace)
    if isinstance(closure, dict):
        workspace_residual = min(workspace_residual, max(0, int(closure.get("unattributed_allocated_peak") or 0)))
    total_peak_growth = sum(max(int(value), 0) for value in peak_growth.values())
    if workspace_residual > 0 and total_peak_growth > 0:
        remaining = int(workspace_residual)
        growth_items = sorted(peak_growth.items(), key=lambda item: int(item[1]), reverse=True)
        for index, (component, growth_bytes) in enumerate(growth_items):
            if remaining <= 0:
                break
            if index == len(growth_items) - 1:
                value = remaining
            else:
                value = min(remaining, int(round(workspace_residual * (int(growth_bytes) / total_peak_growth))))
            if value <= 0:
                continue
            add(
                "GPU HBM",
                "temporary_workspace",
                component,
                "inferred_peak_workspace",
                value,
                "inferred from the component that raised the step CUDA allocated peak",
                "inferred peak residual",
            )
            remaining -= value

    allocated_known = sum(int(item["bytes"]) for item in rows if item.get("memory_space") == "GPU HBM")
    unattributed_value = max(0, peak_allocated - allocated_known)
    reserved_unallocated = max(0, peak_reserved - peak_allocated)
    add(
        "GPU HBM",
        "unattributed_allocated_peak",
        "unattributed_allocated_peak",
        "allocated_residual",
        unattributed_value,
        "inferred residual to peak",
        "residual",
    )
    add(
        "GPU reserved",
        "allocator",
        "allocator_reserved_unallocated",
        "reserved_unallocated",
        reserved_unallocated,
        "peak reserved - peak allocated",
        "exact allocator snapshot",
        keep_zero=True,
    )
    external_value = 0
    if isinstance(external_memory, dict):
        external_value = int(external_memory.get("external_cuda_or_driver_bytes") or 0)
    if isinstance(closure, dict):
        external_value = max(external_value, int(closure.get("external_cuda_or_driver") or 0))
    add(
        "External CUDA",
        "external",
        "external_cuda_or_driver",
        "process_or_driver_gap",
        external_value,
        "device used - PyTorch peak reserved",
        "diagnostic",
    )
    return rows, peak_allocated, peak_reserved


def build_memory_breakdown_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "enabled": False,
            "breakdown_rows": [],
            "schema_version": 2,
            "allocated_closure_ok": False,
            "reserved_closure_ok": False,
            "allocated_closure_error_bytes": 0,
            "reserved_closure_error_bytes": 0,
        }
    candidates = [row for row in rows if not row.get("is_warmup")]
    if not candidates:
        candidates = rows
    actual_peak = _actual_peak_row(candidates)
    selected = _select_memory_breakdown_row(candidates)
    breakdown_rows, peak_allocated, peak_reserved = _flatten_memory_breakdown_row(selected)
    actual_breakdown_rows, actual_peak_allocated, actual_peak_reserved = _flatten_memory_breakdown_row(actual_peak)
    live_activation_detail_rows = selected.get("live_activation_detail_rows_at_peak")
    if not isinstance(live_activation_detail_rows, list) or not live_activation_detail_rows:
        live_activation_detail_rows = selected.get("live_activation_detail_rows", [])
    if not isinstance(live_activation_detail_rows, list):
        live_activation_detail_rows = []
    actual_live_activation_detail_rows = actual_peak.get("live_activation_detail_rows_at_peak")
    if not isinstance(actual_live_activation_detail_rows, list) or not actual_live_activation_detail_rows:
        actual_live_activation_detail_rows = actual_peak.get("live_activation_detail_rows", [])
    if not isinstance(actual_live_activation_detail_rows, list):
        actual_live_activation_detail_rows = []
    allocated_sum = sum(int(row.get("bytes") or 0) for row in breakdown_rows if row.get("memory_space") == "GPU HBM")
    reserved_gap_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("component") == "allocator_reserved_unallocated"
    )
    actual_allocated_sum = sum(
        int(row.get("bytes") or 0) for row in actual_breakdown_rows if row.get("memory_space") == "GPU HBM"
    )
    actual_reserved_gap_sum = sum(
        int(row.get("bytes") or 0)
        for row in actual_breakdown_rows
        if row.get("component") == "allocator_reserved_unallocated"
    )
    reserved_stack_sum = int(allocated_sum) + int(reserved_gap_sum)
    allocated_closure_error = int(peak_allocated) - int(allocated_sum)
    reserved_closure_error = int(peak_reserved) - int(reserved_stack_sum)
    saved_activation_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("memory_space") == "GPU HBM"
        and row.get("group") == "saved_activations"
        and row.get("kind") == "saved_activation"
    )
    live_activation_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("memory_space") == "GPU HBM"
        and row.get("group") == "saved_activations"
        and row.get("kind") == "live_activation"
    )
    activation_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("memory_space") == "GPU HBM" and row.get("group") == "saved_activations"
    )
    workspace_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("memory_space") == "GPU HBM" and row.get("group") == "temporary_workspace"
    )
    unattributed_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("memory_space") == "GPU HBM" and row.get("group") == "unattributed_allocated_peak"
    )
    external_sum = sum(
        int(row.get("bytes") or 0)
        for row in breakdown_rows
        if row.get("component") == "external_cuda_or_driver"
    )
    return {
        "enabled": True,
        "schema_version": 2,
        "selected_metric": "peak_allocated_hbm_bytes",
        "peak_allocated_hbm_bytes": int(peak_allocated),
        "peak_reserved_hbm_bytes": int(peak_reserved),
        "reserved_unallocated_bytes": max(0, int(peak_reserved) - int(peak_allocated)),
        "external_cuda_or_driver_bytes": int(external_sum),
        "allocated_stack_sum_bytes": int(allocated_sum),
        "reserved_stack_sum_bytes": int(reserved_stack_sum),
        "saved_activation_hbm_bytes_at_peak": int(saved_activation_sum),
        "live_activation_hbm_bytes_at_peak": int(live_activation_sum),
        "activation_hbm_bytes_at_peak": int(activation_sum),
        "temporary_workspace_hbm_bytes_at_peak": int(workspace_sum),
        "unattributed_allocated_peak_bytes": int(unattributed_sum),
        "selected_phase": selected.get("phase", ""),
        "selected_step": selected.get("step", 0),
        "actual_peak_phase": actual_peak.get("phase", ""),
        "actual_peak_step": actual_peak.get("step", 0),
        "actual_peak_allocated_hbm_bytes": int(actual_peak_allocated),
        "actual_peak_reserved_hbm_bytes": int(actual_peak_reserved),
        "actual_peak_reserved_unallocated_bytes": max(0, int(actual_peak_reserved) - int(actual_peak_allocated)),
        "actual_peak_allocated_stack_sum_bytes": int(actual_allocated_sum),
        "actual_peak_reserved_stack_sum_bytes": int(actual_allocated_sum + actual_reserved_gap_sum),
        "actual_peak_allocated_closure_error_bytes": int(actual_peak_allocated) - int(actual_allocated_sum),
        "actual_peak_reserved_closure_error_bytes": int(actual_peak_reserved)
        - int(actual_allocated_sum + actual_reserved_gap_sum),
        "rank_count": _distributed_world_size(),
        "breakdown_rows": breakdown_rows,
        "actual_peak_breakdown_rows": actual_breakdown_rows,
        "live_activation_detail_rows_at_peak": live_activation_detail_rows,
        "actual_peak_live_activation_detail_rows": actual_live_activation_detail_rows,
        "allocated_closure_ok": abs(allocated_closure_error)
        <= max(1024 * 1024, int(peak_allocated * 0.01)),
        "reserved_closure_ok": abs(reserved_closure_error) <= max(1024 * 1024, int(peak_reserved * 0.01)),
        "allocated_closure_error_bytes": allocated_closure_error,
        "reserved_closure_error_bytes": reserved_closure_error,
        "reserved_bytes": int(selected.get("reserved_bytes") or 0),
        "allocated_bytes": int(selected.get("allocated_bytes") or 0),
        "notes": [
            "Actual peak fields report the non-warmup phase with the highest torch CUDA allocated/reserved peak.",
            "Selected breakdown rows use the actual peak phase unless that phase is mostly residual and a same-step phase has substantially better attribution.",
            "Weights, gradients, optimizer states, buffers, and host tensors are unique tensor-storage accounting.",
            "Activation rows include autograd saved tensors and live module output tensors at the best observed live allocation point.",
            "Temporary workspace rows are inferred from the component that raised the step CUDA allocated peak and are capped by the residual after exact live rows.",
            "Unattributed allocated peak is the remaining residual from known GPU HBM rows to peak allocated HBM.",
            "GPU HBM rows plus allocator reserved-unallocated close to torch.cuda.max_memory_reserved-style reserved peak.",
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
        is_optimizer_step = name == "lf.optimizer.step"
        if is_optimizer_step and getattr(handle, "_asym_lf_in_optimizer_step_range", False):
            return original(*args, **kwargs)
        previous_optimizer_step_range = bool(getattr(handle, "_asym_lf_in_optimizer_step_range", False))
        if is_optimizer_step:
            setattr(handle, "_asym_lf_in_optimizer_step_range", True)
        try:
            if handle.memory_breakdown_profiler is not None and is_optimizer_step:
                handle.memory_breakdown_profiler.record_phase("before_optimizer_step", model=handle.model, optimizer=owner)
            with _range(handle, name):
                try:
                    return original(*args, **kwargs)
                finally:
                    if handle.memory_breakdown_profiler is not None and is_optimizer_step:
                        handle.memory_breakdown_profiler.record_phase("after_optimizer_step", model=handle.model, optimizer=owner)
        finally:
            if is_optimizer_step:
                setattr(handle, "_asym_lf_in_optimizer_step_range", previous_optimizer_step_range)

    setattr(wrapped, _PATCH_ATTR, True)
    if getattr(original, "_wrapped_by_lr_sched", False):
        setattr(wrapped, "_wrapped_by_lr_sched", True)
    try:
        setattr(owner, attr, wrapped)
    except Exception:
        seen.discard(key)


def _unwrap_optimizer_for_introspection(optimizer: Any | None) -> Any | None:
    if optimizer is None:
        return None
    if callable(getattr(optimizer, "asym_cpu_adamw_summary", None)) or callable(
        getattr(optimizer, "asym_cpu_master_params", None)
    ):
        return optimizer
    seen = {id(optimizer)}
    current = optimizer
    for _ in range(4):
        next_optimizer = None
        for attr in ("optimizer", "inner_optimizer", "base_optimizer"):
            candidate = getattr(current, attr, None)
            if candidate is not None and id(candidate) not in seen:
                next_optimizer = candidate
                break
        if next_optimizer is None:
            return optimizer
        if callable(getattr(next_optimizer, "asym_cpu_adamw_summary", None)) or callable(
            getattr(next_optimizer, "asym_cpu_master_params", None)
        ):
            return next_optimizer
        seen.add(id(next_optimizer))
        current = next_optimizer
    return optimizer


def _patch_optimizer_objects(handle: LFTraceHandle, trainer: Any) -> None:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        handle.prepared_optimizer = optimizer
        handle.optimizer = _unwrap_optimizer_for_introspection(optimizer)
        if handle.memory_breakdown_profiler is not None:
            handle.memory_breakdown_profiler.set_optimizer(handle.optimizer)
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
                        # Opt-in: dump a full CUDA allocator snapshot at the forward->backward
                        # boundary so every live byte (incl. saved-activations, offload-staging,
                        # workspace -- not just live module outputs) can be attributed by
                        # allocation frame. Requires ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT=1 so
                        # blocks carry python stacks. Fires only on the measured step.
                        _af_prof = handle.memory_breakdown_profiler
                        _af_path = os.environ.get("ASYM_GEMM_LF_PROFILE_AFTER_FORWARD_SNAPSHOT_PATH", "").strip()
                        if _af_path and torch.cuda.is_available() and getattr(_af_prof, "_active", False):
                            try:
                                import pickle as _af_pickle

                                if getattr(getattr(_af_prof, "config", None), "sync", False):
                                    torch.cuda.synchronize()
                                _af_step = int(getattr(_af_prof, "_step", 0) or 0)
                                _af_out = _af_path.replace("{step}", str(_af_step)) if "{step}" in _af_path else _af_path
                                _af_dir = os.path.dirname(_af_out)
                                if _af_dir:
                                    os.makedirs(_af_dir, exist_ok=True)
                                with open(_af_out, "wb") as _af_fh:
                                    _af_pickle.dump(
                                        torch.cuda.memory._snapshot(),  # type: ignore[attr-defined]
                                        _af_fh,
                                        protocol=_af_pickle.HIGHEST_PROTOCOL,
                                    )
                            except Exception:
                                pass
                    _maybe_release_cached_blocks("forward_end")
                    return result
                except BaseException:
                    if handle.memory_breakdown_profiler is not None:
                        try:
                            handle.memory_breakdown_profiler.record_phase(
                                "forward_exception",
                                model=getattr(self, "model", None),
                                optimizer=getattr(self, "optimizer", None),
                            )
                        except Exception:
                            pass
                    raise
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
                        _maybe_release_cached_blocks("backward_end")
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
    has_inplace_moe = _has_inplace_moe(modules)
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
        if handle.config.backward_module_ranges_enabled and not _skip_backward_hook(module_name, has_inplace_moe):
            try:
                handle.module_hooks.append(module.register_full_backward_pre_hook(backward_pre))
                handle.module_hooks.append(module.register_full_backward_hook(backward_post))
            except Exception:
                pass

    setattr(model, _HOOK_ATTR, True)


def _component_from_name(name: str, param: nn.Parameter) -> str:
    component = _component_from_param_name(name)
    return "other_model" if component == "other" else component


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
    host_bytes_by_component: dict[str, int] = {}
    pinned_bytes_by_component: dict[str, int] = {}
    seen_host_keys: set[tuple[str, int, int, int, str]] = set()
    for module_name, module in model.named_modules():
        component = _component_from_module_name(module_name) or _component_from_param_name(module_name)
        for attr in ("host_weight", "weight_host"):
            host = getattr(module, attr, None)
            if getattr(host, "is_paged", False):
                continue  # NVMe-paged: pager cache reports residency; .tensor would fetch from NVMe
            tensor = getattr(host, "tensor", None)
            if isinstance(tensor, torch.Tensor):
                storage_key = _tensor_storage_key(tensor)
                if storage_key in seen_host_keys:
                    continue
                seen_host_keys.add(storage_key)
                bytes_value = _tensor_bytes(tensor)
                host_bytes_by_component[component] = host_bytes_by_component.get(component, 0) + bytes_value
                if tensor.is_pinned():
                    pinned_bytes_by_component[component] = pinned_bytes_by_component.get(component, 0) + bytes_value
    for component, bytes_value in sorted(host_bytes_by_component.items()):
        # Pinned bytes are the page-locked SUBSET of this component's host bytes, not additional
        # memory. Emit them as a subset field on the single host_weight row rather than a separate
        # additive "pinned_host_weight" row, so summing rows never double-counts offloaded weights
        # (e.g. routed_experts, whose GPU param is a 0-byte placeholder and whose real ~54 GiB lives
        # only on the CPU host).
        rows.append(
            {
                "category": "host_weight",
                "component": component,
                "device": "cpu",
                "bytes": bytes_value,
                "pinned_bytes": pinned_bytes_by_component.get(component, 0),
            }
        )
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
