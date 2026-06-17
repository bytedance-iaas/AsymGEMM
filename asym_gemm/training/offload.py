from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .host_weight import HostWeight, tensor_nbytes


ComponentClassifier = Callable[[str, nn.Module | None], str]


@dataclass(frozen=True)
class OffloadResidencyRow:
    name: str
    component: str
    kind: str
    device: str
    bytes: int
    requires_grad: bool
    selected_for_cpu: bool
    storage_key: tuple[Any, ...] | None


def storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
    )


def _default_classify_component(name: str, module: nn.Module | None = None) -> str:
    lower = name.lower()
    leaf = lower.rsplit(".", 1)[-1]
    if ".mlp.shared_expert" in lower or ".shared_expert." in lower or ".shared_experts." in lower:
        return "shared_experts"
    if lower == "shared_expert_gate" or lower.endswith(".shared_expert_gate") or ".shared_expert_gate." in lower:
        return "shared_experts"
    if lower == "experts" or ".mlp.experts." in lower or lower.endswith(".experts") or ".feed_forward.experts." in lower:
        return "routed_experts"
    if (
        lower == "router"
        or lower.endswith(".mlp.gate")
        or ".mlp.gate." in lower
        or lower.endswith(".router")
        or ".router." in lower
        or ".feed_forward.router." in lower
    ):
        return "router"
    if leaf == "weight" and "." in lower:
        parent_leaf = lower.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    else:
        parent_leaf = leaf
    attention_leaves = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if parent_leaf in attention_leaves or any(
        lower == target or lower.endswith(f".{target}") or f".{target}." in lower for target in attention_leaves
    ):
        return "attention"
    if (
        ".self_attn." in lower or ".self_attention." in lower or ".attention." in lower
    ) and parent_leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj"}:
        return "attention"
    if (
        lower in {"embed_tokens", "embed_in", "wte"}
        or lower.endswith(".embed_tokens")
        or ".embed_tokens." in lower
        or lower.endswith(".embed_in")
        or ".embed_in." in lower
        or lower.endswith(".wte")
        or ".wte." in lower
    ):
        return "embed_tokens"
    if (
        lower in {"lm_head", "output", "output_layer"}
        or lower.endswith(".lm_head")
        or ".lm_head." in lower
        or lower.endswith(".output")
        or ".output." in lower
        or lower.endswith(".output_layer")
        or ".output_layer." in lower
    ):
        return "lm_head"
    if (
        "norm" in leaf
        or "layernorm" in leaf
        or "rms_norm" in leaf
        or "q_norm" in leaf
        or "k_norm" in leaf
        or ".norm." in lower
        or ".input_layernorm." in lower
        or ".post_attention_layernorm." in lower
        or ".q_norm." in lower
        or ".k_norm." in lower
    ):
        return "norms"
    if parent_leaf in {"gate_proj", "up_proj", "down_proj"} and (".mlp." in lower or ".feed_forward." in lower):
        return "mlp_dense"
    return "other"


def _selection_component_selected(selection: Any, component: str, leaf: str = "", name: str = "") -> bool:
    if component == "routed_experts":
        return bool(getattr(selection, "routed_experts", False))
    if component == "router":
        return bool(getattr(selection, "router", False))
    if component == "shared_experts":
        return bool(getattr(selection, "shared_experts", False))
    if component == "attention":
        targets = getattr(selection, "attention_targets", frozenset())
        lower = name.lower()
        return bool(targets) and (
            not leaf
            or leaf in targets
            or any(lower == target or lower.endswith(f".{target}") or f".{target}." in lower for target in targets)
        )
    if component == "embed_tokens":
        return bool(getattr(selection, "embed_tokens", False))
    if component == "lm_head":
        return bool(getattr(selection, "lm_head", False))
    if component == "norms":
        return bool(getattr(selection, "norms", False))
    if component == "mlp_dense":
        return bool(getattr(selection, "mlp_dense", False))
    return False


def _is_lora_name(name: str) -> bool:
    return "lora_" in name or ".lora_A." in name or ".lora_B." in name


def adopt_host_weight(
    name: str,
    tensor: torch.Tensor,
    component: str,
    *,
    require_2d: bool = True,
    pin_memory_policy: str = "none",
    strict: bool = True,
) -> HostWeight:
    if tensor.device.type != "cpu":
        if strict:
            raise RuntimeError(f"{name} selected for CPU offload but source tensor is on {tensor.device}")
        tensor = tensor.to("cpu", non_blocking=False)
    if pin_memory_policy not in {"none", "auto", "require"}:
        raise ValueError("pin_memory_policy must be one of: none, auto, require")
    source_key = storage_key(tensor)
    pin_memory = pin_memory_policy in {"auto", "require"}
    host_weight = HostWeight(
        tensor.detach(),
        pin_memory=pin_memory,
        clone=False,
        require_2d=require_2d,
        name=name or component,
    )
    if strict and pin_memory_policy == "none" and storage_key(host_weight.weight) != source_key:
        raise RuntimeError(f"{name} selected for CPU offload but HostWeight adoption would require a CPU copy")
    if (
        strict
        and pin_memory_policy == "auto"
        and not host_weight.weight.is_pinned()
        and storage_key(host_weight.weight) != source_key
    ):
        raise RuntimeError(f"{name} selected for CPU offload but HostWeight adoption would require a CPU copy")
    if strict and pin_memory_policy == "require" and not host_weight.weight.is_pinned():
        raise RuntimeError(f"{name} selected for AsymGEMM CPU offload but host weight could not be pinned")
    if strict and pin_memory_policy == "auto" and torch.cuda.is_available() and not host_weight.weight.is_pinned():
        raise RuntimeError(f"{name} selected for AsymGEMM CPU offload but host weight could not be pinned")
    return host_weight


def _adopt_cpu_tensor_no_copy(name: str, tensor: torch.Tensor, *, strict: bool = True) -> torch.Tensor:
    adopted = tensor.detach()
    if strict and adopted.device.type != "cpu":
        raise RuntimeError(f"{name} selected for CPU offload but source tensor is on {adopted.device}")
    if adopted.device.type != "cpu":
        adopted = adopted.to("cpu", non_blocking=False)
    if strict and not adopted.is_contiguous():
        raise RuntimeError(f"{name} selected for CPU offload but is non-contiguous and would require a CPU copy")
    if not adopted.is_contiguous():
        adopted = adopted.contiguous()
    adopted.requires_grad_(False)
    return adopted


def _host_weight_from_module(module: nn.Module) -> HostWeight | None:
    host = getattr(module, "host_weight", None)
    if isinstance(host, HostWeight):
        return host
    base = getattr(module, "base_layer", None)
    host = getattr(base, "host_weight", None)
    if isinstance(host, HostWeight):
        return host
    return None


def collect_lf_offload_residency(
    model: nn.Module,
    selection: Any,
    *,
    classify_component: ComponentClassifier | None = None,
) -> list[OffloadResidencyRow]:
    classifier = classify_component or _default_classify_component
    module_by_name = dict(model.named_modules())
    rows: list[OffloadResidencyRow] = []

    for name, param in model.named_parameters():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = module_by_name.get(parent_name)
        component = classifier(name, module)
        leaf = name.rsplit(".", 1)[-1]
        rows.append(
            OffloadResidencyRow(
                name=name,
                component=component,
                kind="parameter",
                device=param.device.type,
                bytes=tensor_nbytes(param),
                requires_grad=bool(param.requires_grad),
                selected_for_cpu=_selection_component_selected(selection, component, leaf, name),
                storage_key=storage_key(param),
            )
        )

    for name, buffer in model.named_buffers():
        if not isinstance(buffer, torch.Tensor):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = module_by_name.get(parent_name)
        component = classifier(name, module)
        leaf = name.rsplit(".", 1)[-1]
        rows.append(
            OffloadResidencyRow(
                name=name,
                component=component,
                kind="buffer",
                device=buffer.device.type,
                bytes=tensor_nbytes(buffer),
                requires_grad=False,
                selected_for_cpu=_selection_component_selected(selection, component, leaf, name),
                storage_key=storage_key(buffer),
            )
        )

    seen_host_keys: set[tuple[Any, ...]] = set()
    for module_name, module in model.named_modules():
        host = _host_weight_from_module(module)
        if host is None:
            continue
        tensor = host.weight
        key = storage_key(tensor)
        kind = "host_weight_alias" if key in seen_host_keys else "host_weight"
        seen_host_keys.add(key)
        component = classifier(module_name, module)
        leaf = module_name.rsplit(".", 1)[-1]
        rows.append(
            OffloadResidencyRow(
                name=f"{module_name}.host_weight" if module_name else "host_weight",
                component=component,
                kind=kind,
                device=tensor.device.type,
                bytes=tensor_nbytes(tensor),
                requires_grad=False,
                selected_for_cpu=_selection_component_selected(selection, component, leaf, module_name),
                storage_key=key,
            )
        )

    return rows


def _row_is_frozen_base(row: OffloadResidencyRow) -> bool:
    return not row.requires_grad and not _is_lora_name(row.name)


def validate_lf_offload_residency(
    model: nn.Module,
    selection: Any,
    *,
    strict: bool = True,
    classify_component: ComponentClassifier | None = None,
) -> list[OffloadResidencyRow]:
    rows = collect_lf_offload_residency(model, selection, classify_component=classify_component)
    if not strict:
        return rows

    selected_cuda = [
        row.name
        for row in rows
        if row.selected_for_cpu and row.device == "cuda" and row.kind in {"parameter", "buffer"} and _row_is_frozen_base(row)
    ]
    if selected_cuda:
        raise RuntimeError(f"selected frozen base weights remain CUDA-resident: {selected_cuda[:20]}")

    non_lora_trainable = [row.name for row in rows if row.kind == "parameter" and row.requires_grad and not _is_lora_name(row.name)]
    if non_lora_trainable:
        raise RuntimeError(f"non-LoRA parameters remain trainable: {non_lora_trainable[:20]}")

    return rows


class AsymFrozenEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, *, host_weight: HostWeight | None = None, pin_memory: bool = False) -> None:
        super().__init__()
        self._output_device: torch.device | None = None
        if getattr(source, "max_norm", None) is not None:
            raise ValueError("AsymFrozenEmbedding does not support mutable max_norm embeddings")
        self.host_weight = host_weight or adopt_host_weight(
            "embed_tokens",
            source.weight,
            "embed_tokens",
            require_2d=False,
            pin_memory_policy="none" if not pin_memory else "none",
        )
        self.padding_idx = source.padding_idx
        self.scale_grad_by_freq = bool(source.scale_grad_by_freq)
        self.sparse = bool(source.sparse)
        if self.scale_grad_by_freq or self.sparse:
            raise ValueError("AsymFrozenEmbedding supports scale_grad_by_freq=False and sparse=False only")

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        super()._apply(fn, recurse=recurse)
        probe_device = self._output_device or self.host_weight.weight.device
        probe = torch.empty(0, dtype=self.host_weight.weight.dtype, device=probe_device)
        moved = fn(probe)
        if isinstance(moved, torch.Tensor):
            self._output_device = moved.device
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_device = self._output_device or input_ids.device
        input_ids_cpu = input_ids.to("cpu", non_blocking=False)
        out_cpu = F.embedding(input_ids_cpu, self.host_weight.weight, self.padding_idx)
        if target_device.type == "cpu":
            return out_cpu
        return out_cpu.to(device=target_device, non_blocking=True)


class AsymFrozenRMSNorm(nn.Module):
    def __init__(self, source: nn.Module, *, host_weight: HostWeight | None = None, pin_memory: bool = False) -> None:
        super().__init__()
        weight = getattr(source, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("AsymFrozenRMSNorm requires a weight tensor")
        source_class = source.__class__.__name__
        self.host_weight = host_weight or adopt_host_weight(
            "rms_norm",
            weight,
            "norms",
            require_2d=False,
            pin_memory_policy="none" if not pin_memory else "none",
        )
        self.eps = float(getattr(source, "variance_epsilon", getattr(source, "eps", 1e-6)))
        self.shifted_weight = source_class == "Qwen3_5MoeRMSNorm"
        self.gated = source_class == "Qwen3_5MoeRMSNormGated"

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        weight = self.host_weight.weight.to(device=x.device, non_blocking=True)
        if self.gated or gate is not None:
            if gate is None:
                raise TypeError("AsymFrozenRMSNorm gated mode requires gate")
            input_dtype = x.dtype
            out = x.to(torch.float32)
            variance = out.pow(2).mean(-1, keepdim=True)
            out = out * torch.rsqrt(variance + self.eps)
            out = weight * out.to(input_dtype)
            out = out * F.silu(gate.to(torch.float32))
            return out.to(dtype=input_dtype)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        out = x.float() * torch.rsqrt(variance + self.eps)
        scale = 1.0 + weight.float() if self.shifted_weight else weight.float()
        return (out * scale).to(dtype=x.dtype)


class AsymFrozenLayerNorm(nn.Module):
    def __init__(
        self,
        source: nn.LayerNorm,
        *,
        host_weight: HostWeight | None = None,
        pin_memory: bool = False,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.LayerNorm):
            raise TypeError("AsymFrozenLayerNorm requires nn.LayerNorm")
        self.normalized_shape = source.normalized_shape
        self.eps = float(source.eps)
        self.host_weight = host_weight or adopt_host_weight(
            "layer_norm",
            source.weight,
            "norms",
            require_2d=False,
            pin_memory_policy="none" if not pin_memory else "none",
        )
        self.bias_cpu = None if source.bias is None else _adopt_cpu_tensor_no_copy("layer_norm.bias", source.bias, strict=strict)

    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.bias_cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.host_weight.weight.to(device=x.device, non_blocking=True)
        bias = None if self.bias_cpu is None else self.bias_cpu.to(device=x.device, non_blocking=True)
        return F.layer_norm(x.float(), self.normalized_shape, weight.float(), None if bias is None else bias.float(), self.eps).to(
            dtype=x.dtype
        )


__all__ = [
    "AsymFrozenEmbedding",
    "AsymFrozenLayerNorm",
    "AsymFrozenRMSNorm",
    "OffloadResidencyRow",
    "adopt_host_weight",
    "collect_lf_offload_residency",
    "storage_key",
    "validate_lf_offload_residency",
]
