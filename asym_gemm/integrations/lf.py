from __future__ import annotations

import atexit
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear
from asym_gemm.training.lora import (
    AsymLoRALinear,
    TorchLoRALinear,
    freeze_non_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    lora_parameters,
)
from asym_gemm.training.moe import parse_expert_recompute_policy_spec
from asym_gemm.training.offload import (
    AsymFrozenEmbedding,
    AsymFrozenLayerNorm,
    AsymFrozenRMSNorm,
    OffloadResidencyRow,
    adopt_host_weight,
    collect_lf_offload_residency,
    storage_key,
    validate_lf_offload_residency,
)
from asym_gemm.training.qwen3_moe import (
    AsymQwen3Experts,
    AsymQwen3MoeBlock,
    AsymQwen3Router,
    is_qwen3_experts,
    is_qwen3_moe_block,
    wrap_qwen3_experts,
    wrap_qwen3_moe_block,
)
from asym_gemm.training.qwen35_moe import (
    AsymQwen35MoeBlock,
    is_qwen35_moe_block,
    wrap_qwen35_moe_block,
)
from asym_gemm.training.llama4_moe import AsymLlama4Moe, AsymLlama4Router, is_llama4_moe, wrap_llama4_moe


ASYM_LF_ADAPTER_FORMAT = "asym_gemm_lf_v1"
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset(
    {"routed_experts", "router", "shared_experts", "attention", "embed_tokens", "lm_head", "norms"}
)
_ALL_LF_OFFLOAD_COMPONENTS = frozenset(
    {
        "routed_experts",
        "router",
        "shared_experts",
        "attention",
        "embed_tokens",
        "lm_head",
        "norms",
    }
)
_ATTENTION_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


@dataclass(frozen=True)
class LFOffloadSelection:
    raw: str
    routed_experts: bool = False
    router: bool = False
    shared_experts: bool = False
    attention_targets: frozenset[str] = frozenset()
    embed_tokens: bool = False
    lm_head: bool = False
    norms: bool = False

    @property
    def any_cpu_offload(self) -> bool:
        return (
            self.routed_experts
            or self.router
            or self.shared_experts
            or bool(self.attention_targets)
            or self.embed_tokens
            or self.lm_head
            or self.norms
        )

    @property
    def implemented_components(self) -> frozenset[str]:
        components: set[str] = set()
        if self.routed_experts:
            components.add("routed_experts")
        if self.router:
            components.add("router")
        if self.shared_experts:
            components.add("shared_experts")
        if self.attention_targets:
            components.add("attention")
        if self.embed_tokens:
            components.add("embed_tokens")
        if self.lm_head:
            components.add("lm_head")
        if self.norms:
            components.add("norms")
        return frozenset(components)


@dataclass(frozen=True)
class LFAsymTargetPlan:
    wrap_experts: bool
    expert_module_names: tuple[str, ...] = ()
    asym_dense_lora_names: tuple[str, ...] = ()
    asym_dense_frozen_names: tuple[str, ...] = ()
    peft_dense_target_suffixes: tuple[str, ...] = ()
    router_names: tuple[str, ...] = ()
    shared_expert_names: tuple[str, ...] = ()
    embedding_names: tuple[str, ...] = ()
    lm_head_names: tuple[str, ...] = ()
    norm_names: tuple[str, ...] = ()
    unsupported_selected_names: tuple[tuple[str, str], ...] = ()


@dataclass
class LFAsymReport:
    packed_experts_wrapped: int = 0
    qwen3_moes_wrapped: int = 0
    qwen35_moes_wrapped: int = 0
    llama4_moes_wrapped: int = 0
    dense_lora_wrapped: int = 0
    trainable_lora_params: int = 0
    cpu_resident_base_bytes: int = 0
    gpu_resident_base_bytes: int = 0
    cpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    gpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    selected_gpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    offload_modules: str = "routed_experts"
    expert_recompute_policy: str = "none"
    router_mode: str = "whole"
    router_no_grad: bool = False
    skipped: list[str] = field(default_factory=list)
    stats: AsymExecutionStats | None = field(default=None, repr=False)

    @property
    def qwen3_experts_wrapped(self) -> int:
        return self.packed_experts_wrapped

    def to_log_string(self) -> str:
        skipped = "; ".join(self.skipped) if self.skipped else "none"
        cpu_by_component = ",".join(
            f"{key}:{value}" for key, value in sorted(self.cpu_resident_base_bytes_by_component.items())
        ) or "none"
        gpu_by_component = ",".join(
            f"{key}:{value}" for key, value in sorted(self.gpu_resident_base_bytes_by_component.items())
        ) or "none"
        selected_gpu_by_component = ",".join(
            f"{key}:{value}" for key, value in sorted(self.selected_gpu_resident_base_bytes_by_component.items())
        ) or "none"
        return (
            "AsymGEMM LoRA-SFT setup: "
            f"offload_modules={self.offload_modules}, "
            f"packed_experts_wrapped={self.packed_experts_wrapped}, "
            f"llama4_moes_wrapped={self.llama4_moes_wrapped}, "
            f"dense_lora_wrapped={self.dense_lora_wrapped}, "
            f"trainable_lora_params={self.trainable_lora_params}, "
            f"cpu_resident_base_bytes={self.cpu_resident_base_bytes}, "
            f"gpu_resident_base_bytes={self.gpu_resident_base_bytes}, "
            f"cpu_resident_base_bytes_by_component={cpu_by_component}, "
            f"gpu_resident_base_bytes_by_component={gpu_by_component}, "
            f"selected_gpu_resident_base_bytes_by_component={selected_gpu_by_component}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"router_mode={self.router_mode}, "
            f"router_no_grad={self.router_no_grad}, "
            f"qwen3_moes_wrapped={self.qwen3_moes_wrapped}, "
            f"qwen35_moes_wrapped={self.qwen35_moes_wrapped}, "
            f"skipped={skipped}"
        )

    def runtime_log_string(self) -> str:
        if self.stats is None:
            return "AsymGEMM LoRA-SFT runtime: stats=unavailable"
        fallbacks = (
            ";".join(f"{reason}:{count}" for reason, count in sorted(self.stats.fallback_reasons.items()))
            if self.stats.fallback_reasons
            else "none"
        )
        return (
            "AsymGEMM LoRA-SFT runtime: "
            f"asym_forward_calls={self.stats.asym_forward_calls}, "
            f"asym_dx_calls={self.stats.asym_dx_calls}, "
            f"torch_forward_calls={self.stats.torch_forward_calls}, "
            f"torch_dx_calls={self.stats.torch_dx_calls}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"router_mode={self.router_mode}, "
            f"router_no_grad={self.router_no_grad}, "
            f"reference_fallback_count={self.stats.reference_fallback_count}, "
            f"fallback_reasons={fallbacks}"
        )


def _env_true(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "y", "on"}


def _register_runtime_report(report: LFAsymReport) -> None:
    if not _env_true(os.environ.get("ASYM_GEMM_LF_LOG_RUNTIME_STATS")):
        return

    def _emit() -> None:
        print(report.runtime_log_string(), flush=True)

    atexit.register(_emit)


def _as_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [part.strip() for part in values.split(",") if part.strip()]
    return [str(value) for value in values]


def _selector_tokens(selector: Sequence[str] | str | None) -> list[str]:
    if selector is None:
        selector = "routed_experts"
    values: list[str] = []
    if isinstance(selector, str):
        raw_values = [selector]
    else:
        raw_values = [str(value) for value in selector]
    for value in raw_values:
        values.extend(part.strip().lower().replace("-", "_") for part in value.split(","))
    return [value for value in values if value]


def parse_lf_offload_modules(selector: Sequence[str] | str | None) -> LFOffloadSelection:
    tokens = _selector_tokens(selector)
    raw = ",".join(tokens) if tokens else "none"
    if not tokens:
        tokens = ["none"]
    if "none" in tokens and len(set(tokens)) > 1:
        raise ValueError("`asym_offload_modules=none` cannot be combined with other offload modules.")

    aliases = {
        "routed": "routed_experts",
        "experts": "routed_experts",
        "gate": "router",
        "expert_router": "router",
        "shared": "shared_experts",
        "shared_expert": "shared_experts",
        "attn": "attention",
        "embedding": "embed_tokens",
        "embeddings": "embed_tokens",
        "token_embeddings": "embed_tokens",
        "output_embedding": "lm_head",
        "output_embeddings": "lm_head",
        "norm": "norms",
        "layernorm": "norms",
        "rmsnorm": "norms",
        "whole_model": "all",
        "model": "all",
    }
    known = _ALL_LF_OFFLOAD_COMPONENTS | {"all", "none"} | _ATTENTION_TARGETS | set(aliases)
    expanded: set[str] = set()
    attention_targets: set[str] = set()
    for token in tokens:
        token = aliases.get(token, token)
        if token not in known:
            valid = ", ".join(sorted(known))
            raise ValueError(f"unknown `asym_offload_modules` token {token!r}; valid tokens: {valid}")
        if token == "none":
            continue
        if token == "all":
            expanded.update(SUPPORTED_LF_OFFLOAD_COMPONENTS)
            continue
        if token == "attention":
            expanded.add("attention")
            attention_targets.update(_ATTENTION_TARGETS)
            continue
        if token in _ATTENTION_TARGETS:
            expanded.add("attention")
            attention_targets.add(token)
            continue
        expanded.add(token)

    not_implemented = sorted(component for component in expanded if component not in SUPPORTED_LF_OFFLOAD_COMPONENTS)
    if not_implemented:
        implemented = ", ".join(sorted(SUPPORTED_LF_OFFLOAD_COMPONENTS))
        raise ValueError(
            "`asym_offload_modules` token(s) are not supported by this build: "
            f"{', '.join(not_implemented)}. Implemented tokens: {implemented}"
        )

    if "attention" in expanded and not attention_targets:
        attention_targets.update(_ATTENTION_TARGETS)
    if "attention" not in expanded:
        attention_targets.clear()
    return LFOffloadSelection(
        raw=raw,
        routed_experts="routed_experts" in expanded,
        router="router" in expanded,
        shared_experts="shared_experts" in expanded,
        attention_targets=frozenset(attention_targets),
        embed_tokens="embed_tokens" in expanded,
        lm_head="lm_head" in expanded,
        norms="norms" in expanded,
    )


def classify_lf_component(name: str, module: nn.Module | None = None) -> str:
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
    if ".mlp." in lower and parent_leaf in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp_dense"
    return "other"


def component_is_selected(component: str, leaf: str, selection: LFOffloadSelection) -> bool:
    if component == "routed_experts":
        return selection.routed_experts
    if component == "router":
        return selection.router
    if component == "shared_experts":
        return selection.shared_experts
    if component == "attention":
        return bool(selection.attention_targets) and leaf in selection.attention_targets
    if component == "embed_tokens":
        return selection.embed_tokens
    if component == "lm_head":
        return selection.lm_head
    if component == "norms":
        return selection.norms
    return False


def _is_stateless_module(module: nn.Module) -> bool:
    return not any(module.parameters(recurse=False)) and not any(module.buffers(recurse=False))


def _is_all_target(raw_lora_target: Sequence[str] | str | None) -> bool:
    return any(target == "all" for target in _as_list(raw_lora_target))


def _targets_experts(raw_lora_target: Sequence[str] | str | None) -> bool:
    targets = set(_as_list(raw_lora_target))
    return bool(targets.intersection({"all", "experts", "gate", "up", "down", "gate_up_proj", "down_proj"}))


def _matches_target(name: str, module: nn.Module, targets: Sequence[str] | str | None) -> bool:
    target_list = _as_list(targets)
    if not target_list:
        return False
    child = name.rsplit(".", 1)[-1]
    return any(name == target or name.endswith(f".{target}") or child == target for target in target_list)


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _layer_profile_prefix_from_module_name(name: str, suffix: str) -> str:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return f"layers.{parts[index + 1]}.{suffix}"
    return f"layers.unknown.{suffix}"


def _qwen3_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "mlp.experts")


def _gemma4_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "experts")


def _llama4_profile_prefix_from_module_name(name: str) -> str:
    return _layer_profile_prefix_from_module_name(name, "feed_forward")


def _packed_expert_family(module: nn.Module) -> str:
    class_name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    config = getattr(module, "config", None)
    config_type = str(getattr(config, "model_type", "")).lower()
    if "gemma4" in class_name or "gemma4" in module_name or config_type == "gemma4":
        return "gemma4"
    if "qwen3_5" in class_name or "qwen3_5" in module_name or config_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return "qwen3_5"
    if "qwen3" in class_name or "qwen3" in module_name or config_type in {"qwen3_moe", "qwen3_vl_moe"}:
        return "qwen3"
    return "packed"


def _output_router_logits_enabled(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if bool(getattr(config, "output_router_logits", False)):
        return True
    text_config = getattr(config, "text_config", None)
    if bool(getattr(text_config, "output_router_logits", False)):
        return True
    return False


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters(recurse=False):
        return torch.device(param.device), param.dtype
    for buffer in module.buffers(recurse=False):
        return torch.device(buffer.device), buffer.dtype
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device()), torch.bfloat16
    return torch.device("cpu"), torch.bfloat16


def _wrap_lf_linear_leaf(
    name: str,
    module: nn.Linear,
    *,
    component: str,
    is_lora_target: bool,
    selected_cpu_offload: bool,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    stats: AsymExecutionStats,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    strict: bool,
) -> nn.Module:
    device, dtype = _module_device_dtype(module)
    if selected_cpu_offload and backend == "asym":
        if strict and module.weight.device.type != "cpu":
            raise RuntimeError(f"{name} selected for CPU offload but source tensor is on {module.weight.device}")
        if strict and module.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} selected for BF16 CPU offload but source dtype is {module.weight.dtype}")
        host_weight = adopt_host_weight(
            f"{name}.weight",
            module.weight,
            component,
            require_2d=True,
            pin_memory_policy="auto",
            strict=strict,
        )
        bias = None if module.bias is None else module.bias.detach()
        if is_lora_target:
            return AsymLoRALinear.from_host_weight(
                host_weight,
                bias=bias,
                rank=lora_rank,
                alpha=lora_alpha,
                backend=backend,
                stats=stats,
                device=device,
                lora_dtype=torch.bfloat16,
                precision=precision,
                init_lora_weights="peft",
                lora_dropout=lora_dropout,
            )
        return AsymFrozenLinear.from_host_weight(
            host_weight,
            bias=bias,
            backend=backend,
            stats=stats,
            precision=precision,
        )

    if not is_lora_target:
        raise ValueError(f"{name} has no LoRA target and is not selected for CPU offload")
    return TorchLoRALinear(
        module,
        rank=lora_rank,
        alpha=lora_alpha,
        device=device,
        dtype=dtype,
        lora_dtype=torch.bfloat16,
        init_lora_weights="peft",
        lora_dropout=lora_dropout,
    )


def _is_under(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _is_router_module_name(name: str) -> bool:
    parts = name.split(".")
    return "router" in parts or name.endswith(".mlp.gate") or ".mlp.gate." in name


def _wrap_lf_router_module(
    name: str,
    module: nn.Module,
    *,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    stats: AsymExecutionStats,
    strict: bool,
) -> nn.Module | None:
    if backend != "asym":
        return None
    lower = name.lower()
    if ".feed_forward.router" in lower or lower.endswith(".router") or lower == "router":
        return AsymLlama4Router(module, backend=backend, precision=precision, stats=stats, strict=strict)
    if lower.endswith(".mlp.gate") or ".mlp.gate." in lower:
        return AsymQwen3Router(module, backend=backend, precision=precision, stats=stats, strict=strict)
    return None


def _reject_tied_lm_head_offload(model: nn.Module, selection: LFOffloadSelection, *, strict: bool) -> None:
    if not strict or not (selection.lm_head or selection.embed_tokens):
        return
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return
    input_embeddings = get_input()
    output_embeddings = get_output()
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if isinstance(input_weight, torch.Tensor) and isinstance(output_weight, torch.Tensor):
        if storage_key(input_weight) == storage_key(output_weight):
            raise ValueError("tied embed/lm_head weights are not supported by this offload stage")


def _validate_trainable_params(model: nn.Module) -> None:
    bad = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name and ".lora_A." not in name and ".lora_B." not in name]
    if bad:
        raise RuntimeError(f"AsymGEMM setup left non-LoRA trainable params: {bad[:20]}")
    router_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (
            ".mlp.gate." in name
            or name.endswith(".mlp.gate.weight")
            or ".router." in name
            or name.endswith(".router.weight")
            or ".feed_forward.router." in name
            or name.endswith(".feed_forward.router.weight")
        )
    ]
    if router_trainable:
        raise RuntimeError(f"AsymGEMM setup must not train router params: {router_trainable[:20]}")


def _aggregate_residency_rows(
    rows: Sequence[OffloadResidencyRow],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    cpu: dict[str, int] = {}
    gpu: dict[str, int] = {}
    selected_gpu: dict[str, int] = {}
    seen_host: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.kind in {"host_weight", "host_weight_alias"}:
            if row.storage_key is not None and row.storage_key in seen_host:
                continue
            if row.storage_key is not None:
                seen_host.add(row.storage_key)
            cpu[row.component] = cpu.get(row.component, 0) + int(row.bytes)
            continue
        if row.kind not in {"parameter", "buffer"}:
            continue
        if row.requires_grad or "lora_" in row.name or ".lora_A." in row.name or ".lora_B." in row.name:
            continue
        if row.device == "cuda":
            gpu[row.component] = gpu.get(row.component, 0) + int(row.bytes)
            if row.selected_for_cpu:
                selected_gpu[row.component] = selected_gpu.get(row.component, 0) + int(row.bytes)
    return cpu, gpu, selected_gpu


def build_lf_asym_report(
    report: LFAsymReport,
    rows: Sequence[OffloadResidencyRow],
    *,
    selection: LFOffloadSelection,
) -> LFAsymReport:
    cpu_by_component, gpu_by_component, selected_gpu_by_component = _aggregate_residency_rows(rows)
    report.cpu_resident_base_bytes_by_component = cpu_by_component
    report.gpu_resident_base_bytes_by_component = gpu_by_component
    report.selected_gpu_resident_base_bytes_by_component = selected_gpu_by_component
    if cpu_by_component:
        report.cpu_resident_base_bytes = sum(cpu_by_component.values())
    if gpu_by_component:
        report.gpu_resident_base_bytes = sum(gpu_by_component.values())
    report.offload_modules = selection.raw
    return report


def count_lora_wrapped_modules(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and not isinstance(module, (AsymQwen3Experts, AsymQwen3MoeBlock, AsymQwen35MoeBlock))
    )


def apply_lf_asym_lora(
    model: nn.Module,
    *,
    raw_lora_target: Sequence[str] | str,
    dense_target_modules: Sequence[str] | str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    backend: Literal["asym", "torch"],
    precision: Literal["bf16"],
    offload_modules: Sequence[str] | str,
    expert_recompute_policy: str = "none",
    router_mode: Literal["hf", "whole"] = "whole",
    wrap_dense: bool = True,
    preexisting_dense_lora_wrapped: int = 0,
    strict: bool = True,
) -> tuple[nn.Module, LFAsymReport]:
    if backend not in {"asym", "torch"}:
        raise ValueError("backend must be 'asym' or 'torch'")
    if precision != "bf16":
        raise ValueError("AsymGEMM LLaMA-Factory integration supports bf16 only")
    selection = parse_lf_offload_modules(offload_modules)
    _reject_tied_lm_head_offload(model, selection, strict=strict)
    if router_mode not in {"hf", "whole"}:
        raise ValueError("router_mode must be 'hf' or 'whole'")
    if router_mode == "whole" and _output_router_logits_enabled(model):
        raise ValueError("asym_router_mode=whole requires output_router_logits=False.")

    recompute_config = parse_expert_recompute_policy_spec(expert_recompute_policy)
    report = LFAsymReport(
        offload_modules=selection.raw,
        expert_recompute_policy=recompute_config.label,
        router_mode=router_mode,
        router_no_grad=router_mode == "whole",
    )
    report.dense_lora_wrapped = int(preexisting_dense_lora_wrapped)
    stats = AsymExecutionStats()
    report.stats = stats
    wrap_experts = _targets_experts(raw_lora_target)
    offload_experts = backend == "asym" and selection.routed_experts
    offload_router = backend == "asym" and selection.router

    expert_replacements: list[tuple[str, nn.Module, nn.Module, str]] = []
    if wrap_experts:
        if router_mode == "whole":
            for name, module in list(model.named_modules()):
                if is_qwen35_moe_block(module):
                    wrapped = wrap_qwen35_moe_block(
                        module,
                        backend=backend,
                        precision=precision,
                        offload=offload_experts,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_dtype=torch.bfloat16,
                        expert_recompute_policy=recompute_config.label,
                        router_mode="whole",
                        offload_router=offload_router,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                    )
                    wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")
                    wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                    expert_replacements.append((name, module, wrapped, f"{name}.experts"))
                elif is_qwen3_moe_block(module):
                    wrapped = wrap_qwen3_moe_block(
                        module,
                        backend=backend,
                        precision=precision,
                        offload=offload_experts,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_dtype=torch.bfloat16,
                        expert_recompute_policy=recompute_config.label,
                        router_mode="whole",
                        offload_router=offload_router,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                    )
                    wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")
                    wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                    expert_replacements.append((name, module, wrapped, f"{name}.experts"))
                elif is_llama4_moe(module):
                    wrapped = wrap_llama4_moe(
                        module,
                        backend=backend,
                        precision=precision,
                        offload=offload_experts,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_dtype=torch.bfloat16,
                        expert_recompute_policy=recompute_config.label,
                        router_mode="whole",
                        offload_router=offload_router,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                    )
                    wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                    wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                    expert_replacements.append((name, module, wrapped, f"{name}.experts"))
        else:
            for name, module in list(model.named_modules()):
                if is_qwen3_experts(module):
                    family = _packed_expert_family(module)
                    wrapped = wrap_qwen3_experts(
                        module,
                        backend=backend,
                        precision=precision,
                        offload=offload_experts,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_dtype=torch.bfloat16,
                        expert_recompute_policy=recompute_config.label,
                        stats=stats,
                        strict=strict,
                    )
                    wrapped.asym_expert_family = family
                    if family == "gemma4":
                        wrapped.profile_prefix = _gemma4_profile_prefix_from_module_name(name)
                    else:
                        wrapped.profile_prefix = _qwen3_profile_prefix_from_module_name(name)
                    expert_replacements.append((name, module, wrapped, name))
                elif is_llama4_moe(module):
                    wrapped = wrap_llama4_moe(
                        module,
                        backend=backend,
                        precision=precision,
                        offload=offload_experts,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        lora_dtype=torch.bfloat16,
                        expert_recompute_policy=recompute_config.label,
                        router_mode="hf",
                        offload_router=offload_router,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                    )
                    wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                    wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                    expert_replacements.append((name, module, wrapped, f"{name}.experts"))

        if not expert_replacements and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.")

        for name, _old_module, new_module, _skip_prefix in expert_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)
            if isinstance(new_module, AsymQwen35MoeBlock):
                report.qwen35_moes_wrapped += 1
            elif isinstance(new_module, AsymLlama4Moe):
                report.llama4_moes_wrapped += 1
            elif isinstance(new_module, AsymQwen3MoeBlock):
                report.qwen3_moes_wrapped += 1
            else:
                report.packed_experts_wrapped += 1
            report.cpu_resident_base_bytes += new_module.cpu_resident_base_bytes
            report.gpu_resident_base_bytes += new_module.gpu_resident_base_bytes

    expert_prefixes = [skip_prefix for _name, _old_module, _new_module, skip_prefix in expert_replacements]
    if offload_router:
        router_replacements: list[tuple[str, nn.Module, nn.Module]] = []
        for name, module in list(model.named_modules()):
            if not name or _is_under(name, expert_prefixes) or isinstance(module, (AsymQwen3Router, AsymLlama4Router)):
                continue
            lower_name = name.lower()
            if not (lower_name == "router" or lower_name.endswith(".mlp.gate") or lower_name.endswith(".feed_forward.router")):
                continue
            if classify_lf_component(name, module) != "router":
                continue
            wrapped_router = _wrap_lf_router_module(
                name,
                module,
                backend=backend,
                precision=precision,
                stats=stats,
                strict=strict,
            )
            if wrapped_router is None:
                if strict:
                    raise RuntimeError(f"{name} selected for router CPU offload but no supported router wrapper exists")
                report.skipped.append(f"{name}:unsupported_router")
                continue
            router_replacements.append((name, module, wrapped_router))

        for name, _old_module, new_module in router_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)

    if backend == "asym" and selection.embed_tokens:
        embedding_replacements: list[tuple[str, nn.Module, nn.Module]] = []
        for name, module in list(model.named_modules()):
            if not name or isinstance(module, AsymFrozenEmbedding):
                continue
            if classify_lf_component(name, module) != "embed_tokens":
                continue
            if not isinstance(module, nn.Embedding):
                if strict:
                    raise RuntimeError(f"{name} selected for embed_tokens CPU offload but is {type(module).__name__}")
                report.skipped.append(f"{name}:unsupported_embedding:{type(module).__name__}")
                continue
            embedding_replacements.append((name, module, AsymFrozenEmbedding(module)))

        for name, _old_module, new_module in embedding_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)

    if backend == "asym" and selection.norms:
        norm_replacements: list[tuple[str, nn.Module, nn.Module]] = []
        for name, module in list(model.named_modules()):
            if not name or isinstance(module, (AsymFrozenLayerNorm, AsymFrozenRMSNorm)):
                continue
            if classify_lf_component(name, module) != "norms":
                continue
            if isinstance(module, nn.LayerNorm):
                norm_replacements.append((name, module, AsymFrozenLayerNorm(module, strict=strict)))
                continue
            weight = getattr(module, "weight", None)
            class_name = type(module).__name__.lower()
            if isinstance(weight, torch.Tensor) and (
                "rmsnorm" in class_name or "rms_norm" in class_name or hasattr(module, "variance_epsilon")
            ):
                norm_replacements.append((name, module, AsymFrozenRMSNorm(module)))
                continue
            if _is_stateless_module(module):
                continue
            if strict:
                raise RuntimeError(f"{name} selected for norms CPU offload but is {type(module).__name__}")
            report.skipped.append(f"{name}:unsupported_norm:{type(module).__name__}")

        for name, _old_module, new_module in norm_replacements:
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)

    dense_replacements: list[tuple[str, nn.Module, nn.Module, bool]] = []
    if wrap_dense:
        for name, module in list(model.named_modules()):
            if not name or _is_under(name, expert_prefixes):
                continue
            if _is_router_module_name(name):
                report.skipped.append(f"{name}:router")
                continue
            if ".lora_A." in name or ".lora_B." in name or isinstance(module, (TorchLoRALinear, AsymLoRALinear)):
                continue
            component = classify_lf_component(name, module)
            leaf = name.rsplit(".", 1)[-1]
            selected_cpu_offload = backend == "asym" and component_is_selected(component, leaf, selection)
            is_lora_target = _matches_target(name, module, dense_target_modules)
            if not is_lora_target and not selected_cpu_offload:
                continue
            if not isinstance(module, nn.Linear):
                if selected_cpu_offload and not is_lora_target and _is_stateless_module(module):
                    continue
                report.skipped.append(f"{name}:not_nn_linear:{type(module).__name__}")
                continue
            dense_replacements.append(
                (
                    name,
                    module,
                    _wrap_lf_linear_leaf(
                        name,
                        module,
                        component=component,
                        is_lora_target=is_lora_target,
                        selected_cpu_offload=selected_cpu_offload,
                        backend=backend,
                        precision=precision,
                        stats=stats,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        strict=strict,
                    ),
                    is_lora_target,
                )
            )

    if wrap_dense and _is_all_target(raw_lora_target) and not dense_replacements and report.dense_lora_wrapped == 0 and strict:
        raise ValueError("AsymGEMM requested dense LoRA target=all but found no dense nn.Linear modules.")

    for name, _old_module, new_module, is_lora_target in dense_replacements:
        parent, child_name = _parent_and_child(model, name)
        _replace_child(parent, child_name, new_module)
        if is_lora_target:
            report.dense_lora_wrapped += 1
        report.gpu_resident_base_bytes += int(getattr(new_module, "gpu_resident_base_weight_bytes", 0))

    freeze_non_lora_params(model)
    _validate_trainable_params(model)
    residency_selection = selection if backend == "asym" else LFOffloadSelection(raw=selection.raw)
    rows = validate_lf_offload_residency(model, residency_selection, strict=strict, classify_component=classify_lf_component)
    build_lf_asym_report(report, rows, selection=selection)
    setattr(model, "_asym_offload_modules", selection.raw)
    report.trainable_lora_params = sum(param.numel() for param in lora_parameters(model))
    if report.trainable_lora_params == 0 and strict:
        raise ValueError("AsymGEMM setup produced zero trainable LoRA parameters.")
    _register_runtime_report(report)
    return model, report


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _infer_adapter_config(model: nn.Module, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "inference_mode": False,
        "asym_gemm": True,
        "asym_adapter_format": ASYM_LF_ADAPTER_FORMAT,
    }
    for module in model.modules():
        if isinstance(module, AsymQwen35MoeBlock):
            config.update(
                {
                    "asym_expert_format": "qwen3_5_owned_moe",
                    "asym_expert_family": "qwen3_5",
                    "asym_router_mode": "whole",
                    "r": module.experts.lora_rank,
                    "lora_alpha": module.experts.lora_alpha,
                    "lora_dropout": module.experts.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.experts.expert_recompute_config.label,
                }
            )
            break
        if isinstance(module, AsymQwen3MoeBlock):
            config.update(
                {
                    "asym_expert_format": "qwen3_owned_moe",
                    "asym_expert_family": "qwen3",
                    "asym_router_mode": "whole",
                    "r": module.experts.lora_rank,
                    "lora_alpha": module.experts.lora_alpha,
                    "lora_dropout": module.experts.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.experts.expert_recompute_config.label,
                }
            )
            break
        if isinstance(module, AsymQwen3Experts):
            family = getattr(module, "asym_expert_family", "qwen3")
            config.update(
                {
                    "asym_expert_format": "packed_gate_up_down",
                    "asym_expert_family": family,
                    "asym_router_mode": "hf",
                    "r": module.lora_rank,
                    "lora_alpha": module.lora_alpha,
                    "lora_dropout": module.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.expert_recompute_config.label,
                }
            )
            break
        if isinstance(module, AsymLlama4Moe):
            config.update(
                {
                    "asym_expert_format": "llama4_packed_moe",
                    "asym_expert_family": "llama4",
                    "asym_router_mode": module.router_mode,
                    "r": module.experts.lora_rank,
                    "lora_alpha": module.experts.lora_alpha,
                    "lora_dropout": module.experts.lora_dropout_p,
                    "asym_backend": module.backend,
                    "asym_precision": module.precision,
                    "asym_offload_modules": "routed_experts" if module.offload else "none",
                    "asym_expert_recompute_policy": module.experts.expert_recompute_config.label,
                }
            )
            break
    raw_selector = getattr(model, "_asym_offload_modules", None)
    if isinstance(raw_selector, str) and raw_selector:
        config["asym_offload_modules"] = raw_selector
    if metadata:
        config.update(_jsonable(dict(metadata)))
    return config


def get_asym_lora_state_dict(
    model: nn.Module,
    *,
    adapter_name: str = "default",
) -> OrderedDict[str, torch.Tensor]:
    state = get_lora_state_dict(model, adapter_name=adapter_name)
    return OrderedDict((name, tensor.detach().to(device="cpu").contiguous()) for name, tensor in state.items())


def save_asym_peft_adapter(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
    metadata: Mapping[str, Any] | None = None,
    safe_serialization: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = get_asym_lora_state_dict(model, adapter_name=adapter_name)
    if not state:
        raise ValueError("AsymGEMM adapter save found no LoRA parameters")

    config = _infer_adapter_config(model, metadata)
    (output_path / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("save_asym_peft_adapter requires safetensors when safe_serialization=True") from exc
        save_file(state, str(output_path / "adapter_model.safetensors"))
    else:
        torch.save(state, output_path / "adapter_model.bin")


def load_asym_peft_adapter(
    model: nn.Module,
    adapter_dir: str | os.PathLike[str],
    *,
    adapter_name: str = "default",
    strict: bool = True,
) -> None:
    adapter_path = Path(adapter_dir)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("load_asym_peft_adapter requires safetensors for adapter_model.safetensors") from exc
        state = load_file(str(safetensors_path))
    else:
        state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
    load_lora_state_dict(model, state, adapter_name=adapter_name, strict=strict)


__all__ = [
    "ASYM_LF_ADAPTER_FORMAT",
    "LFAsymReport",
    "LFAsymTargetPlan",
    "LFOffloadSelection",
    "apply_lf_asym_lora",
    "build_lf_asym_report",
    "classify_lf_component",
    "component_is_selected",
    "count_lora_wrapped_modules",
    "get_asym_lora_state_dict",
    "load_asym_peft_adapter",
    "parse_lf_offload_modules",
    "save_asym_peft_adapter",
]
