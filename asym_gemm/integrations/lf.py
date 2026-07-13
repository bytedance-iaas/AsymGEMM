from __future__ import annotations

import atexit
import ctypes
import gc
import json
import os
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from asym_gemm.training.attention_activation_offload import (
    AsymActivationOffloadLoRALinear,
    AttentionActivationOffloadContext,
    attention_saved_tensor_offload_module_names,
    install_attention_saved_tensor_offload,
)
from asym_gemm.training.sdpa_recompute import install_sdpa_recompute
from asym_gemm.training.attention_checkpoint import (
    attention_checkpoint_module_names,
    install_attention_checkpoint,
)
from asym_gemm.training.decoder_activation_offload import (
    decoder_saved_tensor_offload_module_names,
    install_decoder_saved_tensor_offload,
)
from asym_gemm.training.linear_attention_activation_offload import (
    install_linear_attention_saved_tensor_offload,
    linear_attention_saved_tensor_offload_module_names,
)
from asym_gemm.training.decoder_checkpoint import (
    decoder_checkpoint_module_names,
    install_decoder_checkpoint,
)
# DISABLED — token-chunked MLP superseded by the recomp-off design (agent/impls/finegrained_offload.md); kept as source.
# from asym_gemm.training.chunked_mlp import install_chunked_mlp_on_dense_mlps
from asym_gemm.training.decoder_layer_glue_gc import (
    decoder_layer_glue_gc_module_names,
    install_decoder_layer_glue_gc,
)
from asym_gemm.training.frozen_linear import AsymExecutionStats, AsymFrozenLinear
from asym_gemm.training.host_weight import tensor_nbytes
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
from asym_gemm.training.qwen35_shared_mlp import AsymQwen35SharedMLP, is_qwen35_shared_mlp_leaf
from asym_gemm.training.llama4_moe import AsymLlama4Moe, AsymLlama4Router, is_llama4_moe, wrap_llama4_moe
from asym_gemm.training.llama4_shared_mlp import AsymLlama4SharedMLP, is_llama4_shared_mlp_leaf


ASYM_LF_ADAPTER_FORMAT = "asym_gemm_lf_v1"
SUPPORTED_LF_OFFLOAD_COMPONENTS = frozenset(
    {
        "routed_experts",
        "router",
        "shared_experts",
        "attention",
        "linear_attention",
        "embed_tokens",
        "lm_head",
        "norms",
        "mlp_dense",
    }
)
_ALL_LF_OFFLOAD_COMPONENTS = frozenset(
    {
        "routed_experts",
        "router",
        "shared_experts",
        "attention",
        "linear_attention",
        "embed_tokens",
        "lm_head",
        "norms",
        "mlp_dense",
    }
)
_MALLOC_TRIM_FUNC: Any | None = None
_MALLOC_TRIM_LOOKED_UP = False


def _malloc_trim_func() -> Any | None:
    global _MALLOC_TRIM_FUNC, _MALLOC_TRIM_LOOKED_UP
    if _MALLOC_TRIM_LOOKED_UP:
        return _MALLOC_TRIM_FUNC
    _MALLOC_TRIM_LOOKED_UP = True
    try:
        trim = getattr(ctypes.CDLL(None), "malloc_trim")
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except Exception:
        trim = None
    _MALLOC_TRIM_FUNC = trim
    return _MALLOC_TRIM_FUNC


def _release_replaced_module_memory() -> None:
    gc.collect()
    if os.environ.get("ASYM_GEMM_DISABLE_MALLOC_TRIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    trim = _malloc_trim_func()
    if trim is None:
        return
    try:
        trim(0)
    except Exception:
        return
_ATTENTION_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_LINEAR_ATTENTION_LEAVES = frozenset({"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"})
_VISION_PATH_MARKERS = (
    ".visual.",
    ".vision.",
    ".vision_model.",
    ".vision_tower.",
)
_MULTIMODAL_PROJECTOR_PATH_MARKERS = (
    ".multi_modal_projector.",
    ".visual.merger.",
    ".model.visual.merger.",
)
_QWEN35_LINEAR_ATTN_RUNTIME_MARKERS = (
    ".linear_attn.conv1d.",
    ".linear_attn.dt_bias.",
    ".linear_attn.a_log.",
    ".linear_attn.norm.",
)


def _path_for_match(name: str) -> str:
    return f".{str(name).lower().strip('.')}."


def _is_vision_path(name: str) -> bool:
    lower = _path_for_match(name)
    return any(marker in lower for marker in _VISION_PATH_MARKERS)


def _is_multimodal_projector_path(name: str) -> bool:
    lower = _path_for_match(name)
    return any(marker in lower for marker in _MULTIMODAL_PROJECTOR_PATH_MARKERS)


def _is_vision_or_multimodal_path(name: str) -> bool:
    return _is_vision_path(name) or _is_multimodal_projector_path(name)


def _is_qwen35_linear_attn_runtime_state(name: str) -> bool:
    lower = _path_for_match(name)
    return any(marker in lower for marker in _QWEN35_LINEAR_ATTN_RUNTIME_MARKERS)


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_" in name or ".lora_A." in name or ".lora_B." in name


@dataclass(frozen=True)
class LFOffloadSelection:
    raw: str
    routed_experts: bool = False
    router: bool = False
    shared_experts: bool = False
    attention_targets: frozenset[str] = frozenset()
    linear_attention: bool = False
    embed_tokens: bool = False
    lm_head: bool = False
    norms: bool = False
    mlp_dense: bool = False

    @property
    def any_cpu_offload(self) -> bool:
        return (
            self.routed_experts
            or self.router
            or self.shared_experts
            or bool(self.attention_targets)
            or self.linear_attention
            or self.embed_tokens
            or self.lm_head
            or self.norms
            or self.mlp_dense
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
        if self.linear_attention:
            components.add("linear_attention")
        if self.embed_tokens:
            components.add("embed_tokens")
        if self.lm_head:
            components.add("lm_head")
        if self.norms:
            components.add("norms")
        if self.mlp_dense:
            components.add("mlp_dense")
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
    dense_mlp_act_offload_enabled: bool = False
    dense_mlp_act_offload_wrapped: int = 0
    dense_mlp_finegrained_offload_enabled: bool = False
    dense_mlp_finegrained_offload_wrapped: int = 0
    qwen3_moe_finegrained_offload_enabled: bool = False
    qwen3_moe_finegrained_offload_wrapped: int = 0
    trainable_lora_params: int = 0
    cpu_resident_base_bytes: int = 0
    gpu_resident_base_bytes: int = 0
    cpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    gpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    selected_gpu_resident_base_bytes_by_component: dict[str, int] = field(default_factory=dict)
    unselected_frozen_cuda_residue_bytes_by_component: dict[str, int] = field(default_factory=dict)
    offload_modules: str = "routed_experts"
    expert_recompute_policy: str = "none"
    router_mode: str = "whole"
    router_no_grad: bool = False
    attention_gc_enabled: bool = False
    attention_gc_wrapped: int = 0
    attention_gc_modules: tuple[str, ...] = ()
    attention_gc_skipped: tuple[str, ...] = ()
    layer_gc_enabled: bool = False
    layer_gc_wrapped: int = 0
    layer_gc_modules: tuple[str, ...] = ()
    layer_gc_skipped: tuple[str, ...] = ()
    attention_act_offload_enabled: bool = False
    attention_act_offload_wrapped: int = 0
    attention_act_offload_modules: tuple[str, ...] = ()
    attention_act_offload_skipped: tuple[str, ...] = ()
    layer_act_offload_enabled: bool = False
    layer_act_offload_wrapped: int = 0
    layer_act_offload_modules: tuple[str, ...] = ()
    layer_act_offload_skipped: tuple[str, ...] = ()
    layer_glue_gc_enabled: bool = False
    layer_glue_gc_wrapped: int = 0
    layer_glue_gc_modules: tuple[str, ...] = ()
    layer_glue_gc_skipped: tuple[str, ...] = ()
    attention_saved_tensor_offload_wrapped: int = 0
    attention_saved_tensor_offload_modules: tuple[str, ...] = ()
    attention_saved_tensor_offload_skipped: tuple[str, ...] = ()
    linear_attention_saved_tensor_offload_wrapped: int = 0
    linear_attention_saved_tensor_offload_modules: tuple[str, ...] = ()
    linear_attention_saved_tensor_offload_skipped: tuple[str, ...] = ()
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
        unselected_residue_by_component = ",".join(
            f"{key}:{value}" for key, value in sorted(self.unselected_frozen_cuda_residue_bytes_by_component.items())
        ) or "none"
        return (
            "AsymGEMM LoRA-SFT setup: "
            f"offload_modules={self.offload_modules}, "
            f"packed_experts_wrapped={self.packed_experts_wrapped}, "
            f"llama4_moes_wrapped={self.llama4_moes_wrapped}, "
            f"dense_lora_wrapped={self.dense_lora_wrapped}, "
            f"dense_mlp_act_offload_wrapped={self.dense_mlp_act_offload_wrapped}, "
            f"dense_mlp_finegrained_offload_wrapped={self.dense_mlp_finegrained_offload_wrapped}, "
            f"qwen3_moe_finegrained_offload_wrapped={self.qwen3_moe_finegrained_offload_wrapped}, "
            f"trainable_lora_params={self.trainable_lora_params}, "
            f"cpu_resident_base_bytes={self.cpu_resident_base_bytes}, "
            f"gpu_resident_base_bytes={self.gpu_resident_base_bytes}, "
            f"cpu_resident_base_bytes_by_component={cpu_by_component}, "
            f"gpu_resident_base_bytes_by_component={gpu_by_component}, "
            f"selected_gpu_resident_base_bytes_by_component={selected_gpu_by_component}, "
            f"unselected_frozen_cuda_residue_bytes_by_component={unselected_residue_by_component}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"attention_gc_enabled={self.attention_gc_enabled}, "
            f"attention_gc_wrapped={self.attention_gc_wrapped}, "
            f"layer_gc_enabled={self.layer_gc_enabled}, "
            f"layer_gc_wrapped={self.layer_gc_wrapped}, "
            f"attention_act_offload_enabled={self.attention_act_offload_enabled}, "
            f"attention_act_offload_wrapped={self.attention_act_offload_wrapped}, "
            f"layer_act_offload_enabled={self.layer_act_offload_enabled}, "
            f"layer_act_offload_wrapped={self.layer_act_offload_wrapped}, "
            f"layer_glue_gc_enabled={self.layer_glue_gc_enabled}, "
            f"layer_glue_gc_wrapped={self.layer_glue_gc_wrapped}, "
            f"dense_mlp_finegrained_offload_enabled={self.dense_mlp_finegrained_offload_enabled}, "
            f"dense_mlp_finegrained_offload_wrapped={self.dense_mlp_finegrained_offload_wrapped}, "
            f"qwen3_moe_finegrained_offload_enabled={self.qwen3_moe_finegrained_offload_enabled}, "
            f"qwen3_moe_finegrained_offload_wrapped={self.qwen3_moe_finegrained_offload_wrapped}, "
            f"attention_saved_tensor_offload_wrapped={self.attention_saved_tensor_offload_wrapped}, "
            f"linear_attention_saved_tensor_offload_wrapped={self.linear_attention_saved_tensor_offload_wrapped}, "
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
            f"forward_calls_total={self.stats.forward_calls_total}, "
            f"backward_calls_total={self.stats.backward_calls_total}, "
            f"calls_total={self.stats.calls_total}, "
            f"torch_forward_calls={self.stats.torch_forward_calls}, "
            f"torch_dx_calls={self.stats.torch_dx_calls}, "
            f"expert_recompute_policy={self.expert_recompute_policy}, "
            f"attention_gc_enabled={self.attention_gc_enabled}, "
            f"attention_gc_wrapped={self.attention_gc_wrapped}, "
            f"layer_gc_enabled={self.layer_gc_enabled}, "
            f"layer_gc_wrapped={self.layer_gc_wrapped}, "
            f"attention_act_offload_enabled={self.attention_act_offload_enabled}, "
            f"attention_act_offload_wrapped={self.attention_act_offload_wrapped}, "
            f"layer_act_offload_enabled={self.layer_act_offload_enabled}, "
            f"layer_act_offload_wrapped={self.layer_act_offload_wrapped}, "
            f"layer_glue_gc_enabled={self.layer_glue_gc_enabled}, "
            f"layer_glue_gc_wrapped={self.layer_glue_gc_wrapped}, "
            f"dense_mlp_finegrained_offload_enabled={self.dense_mlp_finegrained_offload_enabled}, "
            f"dense_mlp_finegrained_offload_wrapped={self.dense_mlp_finegrained_offload_wrapped}, "
            f"qwen3_moe_finegrained_offload_enabled={self.qwen3_moe_finegrained_offload_enabled}, "
            f"qwen3_moe_finegrained_offload_wrapped={self.qwen3_moe_finegrained_offload_wrapped}, "
            f"attention_saved_tensor_offload_wrapped={self.attention_saved_tensor_offload_wrapped}, "
            f"linear_attention_saved_tensor_offload_wrapped={self.linear_attention_saved_tensor_offload_wrapped}, "
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
        "linear_attn": "linear_attention",
        "gdn": "linear_attention",
        "gated_deltanet": "linear_attention",
        "embedding": "embed_tokens",
        "embeddings": "embed_tokens",
        "token_embeddings": "embed_tokens",
        "output_embedding": "lm_head",
        "output_embeddings": "lm_head",
        "norm": "norms",
        "layernorm": "norms",
        "rmsnorm": "norms",
        "dense_mlp": "mlp_dense",
        "mlp": "mlp_dense",
        "dense": "mlp_dense",
        "dense_ffn": "mlp_dense",
        "feed_forward": "mlp_dense",
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
        linear_attention="linear_attention" in expanded,
        embed_tokens="embed_tokens" in expanded,
        lm_head="lm_head" in expanded,
        norms="norms" in expanded,
        mlp_dense="mlp_dense" in expanded,
    )


def classify_lf_component(name: str, module: nn.Module | None = None) -> str:
    lower = name.lower()
    leaf = lower.rsplit(".", 1)[-1]
    module_name = type(module).__name__.lower() if module is not None else ""
    if _is_vision_or_multimodal_path(lower):
        return "vision"
    if (
        lower == "linear_attn"
        or lower.endswith(".linear_attn")
        or ".linear_attn." in lower
        or "gateddeltanet" in lower
        or "gateddeltanet" in module_name
    ):
        if leaf == "weight" and "." in lower:
            parent_leaf = lower.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        else:
            parent_leaf = leaf
        if parent_leaf in _LINEAR_ATTENTION_LEAVES or ".linear_attn." in lower or lower.endswith(".linear_attn"):
            return "linear_attention"
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


def component_is_selected(component: str, leaf: str, selection: LFOffloadSelection) -> bool:
    if component == "routed_experts":
        return selection.routed_experts
    if component == "router":
        return selection.router
    if component == "shared_experts":
        return selection.shared_experts
    if component == "attention":
        return bool(selection.attention_targets) and leaf in selection.attention_targets
    if component == "linear_attention":
        return selection.linear_attention and leaf in _LINEAR_ATTENTION_LEAVES
    if component == "embed_tokens":
        return selection.embed_tokens
    if component == "lm_head":
        return selection.lm_head
    if component == "norms":
        return selection.norms
    if component == "mlp_dense":
        return selection.mlp_dense
    return False


def _move_tensor_data_in_place(tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device == device:
        return
    tensor.data = tensor.data.to(device=device, non_blocking=False)
    if tensor.grad is not None:
        tensor.grad.data = tensor.grad.data.to(device=device, non_blocking=False)


def audit_lf_frozen_cuda_residue(
    model: nn.Module,
    *,
    offload_modules: Sequence[str] | str | None,
    strict: bool = True,
    max_allowed_unselected_cuda_bytes: int = 8 * 1024 * 1024,
    allowed_components: set[str] | frozenset[str] = frozenset({"linear_attention"}),
) -> dict[str, int]:
    selection = parse_lf_offload_modules(offload_modules)
    rows = collect_lf_offload_residency(model, selection, classify_component=classify_lf_component)
    residue: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.kind not in {"parameter", "buffer"}:
            continue
        if row.device != "cuda" or row.requires_grad or _is_lora_parameter_name(row.name):
            continue
        if row.selected_for_cpu:
            continue
        residue[row.component] = residue.get(row.component, 0) + int(row.bytes)
        if len(examples[row.component]) < 8:
            examples[row.component].append(row.name)

    bad = {
        component: bytes_value
        for component, bytes_value in residue.items()
        if component not in allowed_components and int(bytes_value) > int(max_allowed_unselected_cuda_bytes)
    }
    if strict and bad:
        bad_examples = {component: examples[component] for component in bad}
        raise RuntimeError(
            "unselected frozen CUDA residue remains under Asym strict mode: "
            f"bytes_by_component={bad}, examples={bad_examples}"
        )
    return dict(residue)


def summarize_lf_tensor_devices(
    model: nn.Module,
    *,
    offload_modules: Sequence[str] | str | None = None,
    max_examples_per_component: int = 4,
) -> dict[str, Any]:
    selection = parse_lf_offload_modules(offload_modules or getattr(model, "_asym_offload_modules", None))
    rows = collect_lf_offload_residency(model, selection, classify_component=classify_lf_component)
    bytes_by_device_component: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    selected_cuda_bytes_by_component: dict[str, int] = defaultdict(int)
    frozen_cuda_bytes_by_component: dict[str, int] = defaultdict(int)
    cuda_examples_by_component: dict[str, list[str]] = defaultdict(list)
    seen_host_keys: set[tuple[Any, ...]] = set()

    for row in rows:
        if row.kind == "host_weight_alias":
            continue
        if row.kind == "host_weight":
            if row.storage_key is not None and row.storage_key in seen_host_keys:
                continue
            if row.storage_key is not None:
                seen_host_keys.add(row.storage_key)

        bytes_by_device_component[row.device][row.component] += int(row.bytes)
        if row.device == "cuda":
            if row.selected_for_cpu:
                selected_cuda_bytes_by_component[row.component] += int(row.bytes)
            if not row.requires_grad and not _is_lora_parameter_name(row.name):
                frozen_cuda_bytes_by_component[row.component] += int(row.bytes)
            examples = cuda_examples_by_component[row.component]
            if len(examples) < max(0, int(max_examples_per_component)):
                examples.append(row.name)

    def _sorted_int_dict(values: Mapping[str, int]) -> dict[str, int]:
        return {key: int(values[key]) for key in sorted(values)}

    return {
        "bytes_by_device_component": {
            device: _sorted_int_dict(values)
            for device, values in sorted(bytes_by_device_component.items())
        },
        "cuda_bytes_by_component": _sorted_int_dict(bytes_by_device_component.get("cuda", {})),
        "cpu_bytes_by_component": _sorted_int_dict(bytes_by_device_component.get("cpu", {})),
        "selected_cuda_bytes_by_component": _sorted_int_dict(selected_cuda_bytes_by_component),
        "frozen_cuda_bytes_by_component": _sorted_int_dict(frozen_cuda_bytes_by_component),
        "cuda_examples_by_component": {
            component: tuple(names) for component, names in sorted(cuda_examples_by_component.items())
        },
    }


def _module_tensor_nbytes(module: nn.Module) -> int:
    total = 0
    seen: set[tuple[Any, ...]] = set()
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if not isinstance(tensor, torch.Tensor):
            continue
        key = storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        total += tensor_nbytes(tensor)
    return total


def _looks_like_qwen35_vision_tower(module: nn.Module) -> bool:
    if all(hasattr(module, attr) for attr in ("patch_embed", "blocks", "merger")):
        return True
    class_name = type(module).__name__.lower()
    return "visionmodel" in class_name or "vision_model" in class_name


def _looks_like_multimodal_vision_tower(module: nn.Module) -> bool:
    if isinstance(module, _CpuStagedFrozenModule):
        return False
    if bool(getattr(module, "input_modalities", ())):
        return True
    if all(hasattr(module, attr) for attr in ("patch_embed", "blocks", "merger")):
        return True
    if all(hasattr(module, attr) for attr in ("patch_embedding", "model", "vision_adapter")):
        return True
    class_name = type(module).__name__.lower().replace("_", "")
    return "visionmodel" in class_name or "visiontower" in class_name


def _first_cuda_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (tuple, list, set)):
        values = value
    else:
        return None
    for item in values:
        device = _first_cuda_tensor_device(item)
        if device is not None:
            return device
    return None


def _contains_requires_grad_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (tuple, list, set)):
        values = value
    else:
        return False
    return any(_contains_requires_grad_tensor(item) for item in values)


class _CpuStagedFrozenModule(nn.Module):
    """CPU-home wrapper for frozen multimodal modules that run with CUDA inputs."""

    def __init__(self, module: nn.Module, *, source_name: str, target_device: torch.device | str, component: str) -> None:
        super().__init__()
        self.module = module
        self.source_name = str(source_name)
        self.target_device = torch.device(target_device)
        self.component = str(component)
        self.cpu_resident_base_bytes = int(_module_tensor_nbytes(module))
        self.stage_calls = 0
        self.release_calls = 0

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as original:
            modules = self.__dict__.get("_modules", {})
            module = modules.get("module") if isinstance(modules, dict) else None
            if module is not None:
                try:
                    return getattr(module, name)
                except AttributeError:
                    pass
            raise original

    def _execution_device(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.device:
        input_device = _first_cuda_tensor_device((args, kwargs))
        if input_device is not None:
            return input_device
        return self.target_device

    @torch.no_grad()
    def _move_inner(self, device: torch.device | str) -> None:
        self.module.to(device=torch.device(device), non_blocking=False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        execution_device = self._execution_device(args, kwargs)
        self._move_inner(execution_device)
        self.stage_calls += 1
        try:
            output = self.module(*args, **kwargs)
            if _contains_requires_grad_tensor(output):
                raise RuntimeError(
                    f"{self.source_name} is installed as a frozen CPU-staged multimodal module, "
                    "but its forward output requires grad. Keep this module CUDA-resident or disable "
                    "vision/projector freezing when training through this path."
                )
            return output
        finally:
            self._move_inner("cpu")
            self.release_calls += 1


def install_lf_frozen_multimodal_cpu_staging(
    model: nn.Module,
    device: torch.device | str,
    *,
    stage_frozen_vision: bool = True,
    stage_frozen_projector: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    target_device = torch.device(device)
    if not stage_frozen_vision and not stage_frozen_projector:
        return {"staged_modules": tuple(), "skipped": tuple(), "staged_bytes": 0}

    candidates: list[tuple[str, str]] = []
    for name, module in list(model.named_modules()):
        if not name or isinstance(module, _CpuStagedFrozenModule):
            continue
        if stage_frozen_projector and _is_multimodal_projector_path(name):
            candidates.append((name, "projector"))
            continue
        if (
            stage_frozen_vision
            and _is_vision_path(name)
            and not _is_multimodal_projector_path(name)
            and _looks_like_multimodal_vision_tower(module)
        ):
            candidates.append((name, "vision"))

    selected: list[tuple[str, str]] = []
    for name, component in sorted(candidates, key=lambda item: (item[0].count("."), len(item[0]), item[0])):
        if any(name == parent or name.startswith(f"{parent}.") for parent, _ in selected):
            continue
        selected.append((name, component))

    staged: list[dict[str, Any]] = []
    skipped: list[str] = []
    staged_bytes = 0
    for name, component in selected:
        module = model.get_submodule(name)
        if isinstance(module, _CpuStagedFrozenModule):
            continue
        trainable = [param_name for param_name, param in module.named_parameters(recurse=True) if param.requires_grad]
        if trainable:
            message = f"{name}:trainable_params:{trainable[:8]}"
            if strict:
                raise RuntimeError(
                    f"refusing to install CPU staging for non-frozen multimodal module {name}; "
                    f"trainable params include {trainable[:8]}"
                )
            skipped.append(message)
            continue
        nbytes = _module_tensor_nbytes(module)
        parent, child_name = _parent_and_child(model, name)
        _replace_child(
            parent,
            child_name,
            _CpuStagedFrozenModule(module, source_name=name, target_device=target_device, component=component),
        )
        staged.append({"name": name, "component": component, "bytes": int(nbytes), "type": type(module).__name__})
        staged_bytes += int(nbytes)

    return {"staged_modules": tuple(staged), "skipped": tuple(skipped), "staged_bytes": int(staged_bytes)}


class _DroppedFrozenVisionTower(nn.Module):
    def __init__(self, source: nn.Module, *, source_name: str) -> None:
        super().__init__()
        self.source_name = str(source_name)
        self.spatial_merge_size = int(getattr(source, "spatial_merge_size", 1))
        self.dtype = getattr(source, "dtype", torch.bfloat16)
        config = getattr(source, "config", None)
        if config is not None:
            self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Qwen3.5 vision tower was removed by ASYM_GEMM_LF_DROP_FROZEN_VISION=1; "
            "this text-only profiling run cannot accept image/video inputs."
        )


def drop_lf_frozen_vision_tower_for_text_only_profile(
    model: nn.Module,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    candidates: list[str] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if not _is_vision_path(name) or _is_multimodal_projector_path(name):
            continue
        if _looks_like_qwen35_vision_tower(module):
            candidates.append(name)

    selected: list[str] = []
    for name in sorted(candidates, key=lambda item: (item.count("."), len(item), item)):
        if any(name == parent or name.startswith(f"{parent}.") for parent in selected):
            continue
        selected.append(name)

    if strict and not selected:
        raise RuntimeError("ASYM_GEMM_LF_DROP_FROZEN_VISION=1 but no frozen Qwen3.5 vision tower was found")

    summaries: list[dict[str, Any]] = []
    removed_bytes = 0
    for name in selected:
        module = model.get_submodule(name)
        trainable = [param_name for param_name, param in module.named_parameters(recurse=True) if param.requires_grad]
        if trainable and strict:
            raise RuntimeError(
                f"refusing to drop trainable vision tower {name}; trainable params include {trainable[:8]}"
            )
        nbytes = _module_tensor_nbytes(module)
        parent, child_name = _parent_and_child(model, name)
        _replace_child(parent, child_name, _DroppedFrozenVisionTower(module, source_name=name))
        removed_bytes += nbytes
        summaries.append({"name": name, "removed_bytes": int(nbytes), "type": type(module).__name__})
        del module

    _release_replaced_module_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"dropped_vision_towers": tuple(summaries), "removed_bytes": int(removed_bytes)}


@torch.no_grad()
def move_lf_asym_cpu_first_model_to_device(
    model: nn.Module,
    device: torch.device | str,
    *,
    offload_modules: Sequence[str] | str | None,
    strict: bool = True,
    keep_frozen_vision_on_cpu: bool = True,
    keep_frozen_projector_on_cpu: bool = True,
) -> dict[str, Any]:
    target_device = torch.device(device)
    selection = parse_lf_offload_modules(offload_modules)
    moved_bytes_by_reason: dict[str, int] = defaultdict(int)
    kept_cpu_bytes_by_component: dict[str, int] = defaultdict(int)
    staging_summary = install_lf_frozen_multimodal_cpu_staging(
        model,
        target_device,
        stage_frozen_vision=keep_frozen_vision_on_cpu,
        stage_frozen_projector=keep_frozen_projector_on_cpu,
        strict=strict,
    )
    setattr(model, "_asym_multimodal_cpu_staging", staging_summary)

    def should_keep_cpu(name: str, tensor: torch.Tensor) -> bool:
        if bool(getattr(tensor, "requires_grad", False)):
            return False
        if keep_frozen_vision_on_cpu and _is_vision_path(name):
            return True
        if keep_frozen_projector_on_cpu and _is_multimodal_projector_path(name):
            return True
        return False

    def should_force_cuda(name: str, tensor: torch.Tensor) -> bool:
        if bool(getattr(tensor, "requires_grad", False)):
            return True
        if _is_lora_parameter_name(name):
            return True
        if _is_qwen35_linear_attn_runtime_state(name):
            return True
        return False

    for name, param in model.named_parameters(recurse=True):
        if should_keep_cpu(name, param):
            kept_cpu_bytes_by_component["vision_or_projector"] += tensor_nbytes(param)
            continue
        if param.device.type == "cuda":
            # Already deliberately placed (LoRA params are created on-device at wrap time;
            # sTP branch1 params live on cuda:1). Re-moving would flatten multi-device
            # placement onto target_device. No-op for |1 (everything CUDA is target already).
            moved_bytes_by_reason["already_on_device"] += 0
            continue
        if should_force_cuda(name, param):
            _move_tensor_data_in_place(param, target_device)
            moved_bytes_by_reason["trainable_or_text_runtime"] += tensor_nbytes(param)
            continue

        component = classify_lf_component(name)
        leaf = name.rsplit(".", 1)[-1]
        selected = component_is_selected(component, leaf, selection)
        if selected and strict:
            raise RuntimeError(
                f"{name} is selected for Asym CPU offload but remains a raw parameter before device move"
            )

        _move_tensor_data_in_place(param, target_device)
        moved_bytes_by_reason[f"unselected_{component}"] += tensor_nbytes(param)

    for name, buffer in model.named_buffers(recurse=True):
        if not isinstance(buffer, torch.Tensor):
            continue
        if should_keep_cpu(name, buffer):
            kept_cpu_bytes_by_component["vision_or_projector"] += tensor_nbytes(buffer)
            continue
        if _is_qwen35_linear_attn_runtime_state(name):
            buffer.data = buffer.data.to(device=target_device, non_blocking=False)
            moved_bytes_by_reason["linear_attention_runtime_buffer"] += tensor_nbytes(buffer)
            continue
        if buffer.device != target_device:
            buffer.data = buffer.data.to(device=target_device, non_blocking=False)
            moved_bytes_by_reason[f"buffer_{classify_lf_component(name)}"] += tensor_nbytes(buffer)

    residue = audit_lf_frozen_cuda_residue(
        model,
        offload_modules=selection.raw,
        strict=strict,
        max_allowed_unselected_cuda_bytes=8 * 1024 * 1024,
        allowed_components=frozenset({"linear_attention"}),
    )
    return {
        "moved_bytes_by_reason": dict(moved_bytes_by_reason),
        "kept_cpu_bytes_by_component": dict(kept_cpu_bytes_by_component),
        "multimodal_cpu_staging": staging_summary,
        "unselected_frozen_cuda_residue_bytes_by_component": dict(residue),
    }


def _is_stateless_module(module: nn.Module) -> bool:
    has_param = next(module.parameters(recurse=False), None) is not None
    has_buffer = next(module.buffers(recurse=False), None) is not None
    return not has_param and not has_buffer


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


def _direct_bf16_linear_shape_reason(module: nn.Linear, *, require_backward: bool = True) -> str | None:
    weight = module.weight
    if weight.dim() != 2:
        return "requires_2d_weight"
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    if out_features <= 0 or in_features <= 0:
        return "requires_positive_nk"
    if out_features % 8 != 0 or in_features % 8 != 0:
        return "requires_8_aligned_nk"
    if require_backward and out_features % 64 != 0:
        return "dx_transpose_b_requires_64_aligned_out_features"
    return None


def _wrap_lf_linear_leaf(
    name: str,
    module: nn.Linear,
    *,
    component: str,
    is_lora_target: bool,
    selected_cpu_offload: bool,
    attention_act_offload_enabled: bool,
    attention_context: AttentionActivationOffloadContext | None,
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
        effective_backend: Literal["asym", "torch"] = backend
        if _direct_bf16_linear_shape_reason(module, require_backward=True) is not None:
            effective_backend = "torch"
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
            if attention_act_offload_enabled and component == "attention" and _is_text_attention_projection_name(name):
                if effective_backend != "asym":
                    if strict:
                        raise RuntimeError(
                            f"{name} cannot use attention activation offload because its shape falls back to torch"
                        )
                elif float(lora_dropout) != 0.0:
                    raise NotImplementedError("attention activation offload currently requires lora_dropout=0.0")
                else:
                    return AsymActivationOffloadLoRALinear.from_host_weight(
                        host_weight,
                        bias=bias,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        backend=effective_backend,
                        stats=stats,
                        device=device,
                        lora_dtype=torch.bfloat16,
                        precision=precision,
                        init_lora_weights="peft",
                        lora_dropout=lora_dropout,
                        projection_role=name.rsplit(".", 1)[-1],
                        attention_context=attention_context,
                    )
            return AsymLoRALinear.from_host_weight(
                host_weight,
                bias=bias,
                rank=lora_rank,
                alpha=lora_alpha,
                backend=effective_backend,
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
            backend=effective_backend,
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


_ATTENTION_GC_EXCLUDED_PATH_MARKERS = (
    ".vision_model.",
    ".vision_tower.",
    ".multi_modal_projector.",
    ".visual.",
    ".vision.",
)


def _attention_gc_enabled_for_policy(policy_label: str) -> bool:
    return str(policy_label).strip() == "gc-attn-exp"


def _layer_gc_enabled_for_policy(policy_label: str) -> bool:
    return str(policy_label).strip() == "gc-layer"


def _attention_act_offload_enabled() -> bool:
    return _env_true(os.environ.get("ASYMM_ATTN_ACT_OFFLOAD")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_OFFLOAD")
    )


def _recomp_off_full_fg_enabled() -> bool:
    return str(os.environ.get("ASYM_GEMM_LF_CONFIG_RECOMP_OFF_STAGE") or "").strip().lower() == "full-fg"


def _qwen3_moe_finegrained_offload_enabled() -> bool:
    return _env_true(os.environ.get("ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD")
    )


def _layer_act_offload_enabled() -> bool:
    return _env_true(os.environ.get("ASYMM_LAYER_ACT_OFFLOAD")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD")
    )


def _install_base_weight_pager(model: nn.Module) -> None:
    """asym_cpuadamwds_panvme (agent/impls/nvme_offload_impl.md Stage 7): page eligible frozen
    base weights to NVMe behind a trace-prefetching pinned cache. No-op (zero imports beyond the
    env check, zero allocations) unless the NVMe store carries the ``base_weight`` role."""
    from ..training.nvme_store import get_nvme_store

    store = get_nvme_store()
    if store is None or not store.has_role("base_weight"):
        return
    from ..training.base_weight_pager import BaseWeightPager
    from ..training.frozen_linear import AsymFrozenLinear, AsymGroupedFrozenLinear
    from ..training.host_weight import HostWeight

    if _qwen3_moe_finegrained_offload_enabled():
        # Force the lazy fused gate_up -> gate/up split (qwen3_moe.py:2509) BEFORE any home is
        # freed — the split slices the fused parent's resident ``_tensor``.
        for module in model.modules():
            ensure = getattr(module, "_ensure_qwen3_moe_finegrained_bases", None)
            if callable(ensure):
                try:
                    ensure()
                except NotImplementedError:
                    pass

    pager = BaseWeightPager(store)
    registered = 0
    for name, module in model.named_modules():
        # Eligible: bf16 AsymFrozenLinear (attn + dense MLP) and AsymGroupedFrozenLinear
        # (experts/shared/router). EXCLUDED by policy: AsymFrozenEmbedding (CPU-side F.embedding
        # per microbatch), norms (tiny, unpinned), and precision != bf16 (the quantized cache
        # builds from ``.weight``, frozen_linear.py:356-380).
        if not isinstance(module, (AsymFrozenLinear, AsymGroupedFrozenLinear)):
            continue
        if str(getattr(module, "precision", "bf16")) != "bf16":
            continue
        host_weight = getattr(module, "host_weight", None)
        if not isinstance(host_weight, HostWeight):
            continue
        if pager.register(f"{name}.host_weight", host_weight):
            registered += 1
    pager.finalize()
    setattr(model, "_asym_base_weight_pager", pager)
    print(
        f"asym panvme: paged {registered} base-weight blobs "
        f"({pager.registered_bytes / (1 << 30):.2f} GiB) to NVMe at {store.cfg.path}",
        flush=True,
    )


def _layer_glue_gc_enabled() -> bool:
    return _env_true(os.environ.get("ASYMM_LAYER_GC")) or _env_true(
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_GC")
    )


def _has_attention_excluded_path_marker(name: str) -> bool:
    lower = f".{name.lower()}."
    return any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS)


def _is_text_attention_projection_name(name: str) -> bool:
    if not name:
        return False
    if _has_attention_excluded_path_marker(name):
        return False
    return name.rsplit(".", 1)[-1] in _ATTENTION_TARGETS


def _attention_parent_name(name: str) -> str | None:
    if not _is_text_attention_projection_name(name) or "." not in name:
        return None
    return name.rsplit(".", 1)[0]


def _build_attention_activation_contexts(
    model: nn.Module,
    *,
    selection: LFOffloadSelection,
    dense_target_modules: Sequence[str] | str,
    backend: Literal["asym", "torch"],
    attention_act_enabled: bool,
) -> dict[str, AttentionActivationOffloadContext]:
    if not attention_act_enabled or backend != "asym":
        return {}

    roles_by_parent: dict[str, set[str]] = {}
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _is_text_attention_projection_name(name):
            continue
        component = classify_lf_component(name, module)
        leaf = name.rsplit(".", 1)[-1]
        if (
            component != "attention"
            or leaf not in {"q_proj", "k_proj", "v_proj"}
            or not component_is_selected(component, leaf, selection)
            or not _matches_target(name, module, dense_target_modules)
            or _direct_bf16_linear_shape_reason(module, require_backward=True) is not None
        ):
            continue
        parent = _attention_parent_name(name)
        if parent is not None:
            roles_by_parent.setdefault(parent, set()).add(leaf)

    return {
        parent: AttentionActivationOffloadContext()
        for parent, roles in roles_by_parent.items()
        if {"q_proj", "k_proj", "v_proj"} <= roles
    }


def _is_text_attention_module_name(name: str, module: nn.Module) -> bool:
    if not name:
        return False
    if _has_attention_excluded_path_marker(name):
        return False
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf not in {"self_attn", "self_attention", "attention", "attn"} and not leaf.endswith("attention"):
        return False
    children = dict(module.named_children())
    return all(target in children for target in _ATTENTION_TARGETS)


def _is_qwen3_decoder_layer_module_name(name: str, module: nn.Module) -> bool:
    if not name:
        return False
    if _has_attention_excluded_path_marker(name):
        return False
    children = dict(module.named_children())
    child_names = set(children)
    class_name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    config = getattr(module, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()

    # Qwen3.5-MoE hybrid decoder layer. Most layers use `linear_attn`
    # (Gated DeltaNet), while full-attention layers use `self_attn`.
    qwen35_required = {"mlp", "input_layernorm", "post_attention_layernorm"}
    qwen35_has_mixer = "linear_attn" in child_names or "self_attn" in child_names
    if qwen35_required <= child_names and qwen35_has_mixer:
        qwen35_class = "qwen3_5" in class_name or "qwen35" in class_name.replace("_", "")
        qwen35_module = "qwen3_5_moe" in module_name or "qwen35" in module_name.replace("_", "")
        qwen35_config = model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}
        qwen35_mlp = hasattr(children["mlp"], "_is_asym_qwen35_moe_block") or is_qwen35_moe_block(children["mlp"])
        if qwen35_class or qwen35_module or qwen35_config or qwen35_mlp:
            return True

    # Qwen3 / Qwen3-MoE decoder layer (token mixer `self_attn`, FFN child `mlp`).
    qwen3_required = {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
    if qwen3_required <= child_names:
        if "qwen3" in class_name or "qwen3" in module_name or model_type in {"qwen3_moe", "qwen3_vl_moe"}:
            return True
        if hasattr(children["mlp"], "_is_asym_qwen3_moe_block") or is_qwen3_moe_block(children["mlp"]):
            return True
    # Generic DENSE decoder layer (Qwen2 / Qwen2.5 / Llama-3.x / Mistral / etc.): standard
    # {self_attn, mlp, input_layernorm, post_attention_layernorm} with a dense FFN whose `mlp`
    # child exposes gate/up/down projections (in any wrapped form) or is an already-installed
    # AsymDenseMLP. MoE decoder layers (mlp = sparse block, no gate/up/down children) are
    # excluded, so this never reclassifies a routed-expert layer.
    if qwen3_required <= child_names:
        mlp_child = children["mlp"]
        if getattr(mlp_child, "_is_asym_dense_mlp", False):
            return True
        if {"gate_proj", "up_proj", "down_proj"} <= set(dict(mlp_child.named_children())):
            return True
    # Llama 4 decoder layer (token mixer `self_attn`, FFN child `feed_forward`; dense or MoE).
    llama4_required = {"self_attn", "feed_forward", "input_layernorm", "post_attention_layernorm"}
    if llama4_required <= child_names:
        if "llama4" in class_name or "llama4" in module_name or model_type in {"llama4", "llama4_text"}:
            return True
    return False


def _wrap_attention_checkpoint_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if _is_text_attention_module_name(name, module):
            install_attention_checkpoint(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {"self_attn", "self_attention", "attention", "attn"} and any(
            marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS
        ):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and not wrapped:
        raise RuntimeError("gc-attn-exp requested but no supported text attention modules were found")
    return tuple(wrapped), tuple(skipped)


def _wrap_qwen3_decoder_checkpoint_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if _is_qwen3_decoder_layer_module_name(name, module):
            install_decoder_checkpoint(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {"decoder_layer", "layer"} and any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and not wrapped:
        raise RuntimeError("gc-layer requested but no supported Qwen3 decoder layers were found")
    return tuple(wrapped), tuple(skipped)


def _wrap_decoder_layer_glue_gc_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if _is_qwen3_decoder_layer_module_name(name, module):
            install_decoder_layer_glue_gc(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {"decoder_layer", "layer"} and any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and not wrapped:
        raise RuntimeError("ASYMM_LAYER_GC requested but no supported decoder layers were found")
    return tuple(wrapped), tuple(skipped)


def _wrap_attention_saved_tensor_offload_modules(
    model: nn.Module,
    *,
    parent_names: Sequence[str],
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested = {name for name in parent_names if name}
    if not requested:
        return (), ()

    wrapped: list[str] = []
    skipped: list[str] = []
    modules = dict(model.named_modules())
    for name in sorted(requested):
        module = modules.get(name)
        if module is None:
            skipped.append(f"{name}:missing_attention_parent")
            continue
        if not _is_text_attention_module_name(name, module):
            skipped.append(f"{name}:unsupported_attention_parent:{type(module).__name__}")
            continue
        install_attention_saved_tensor_offload(module)
        wrapped.append(name)

    # SDPA-only recompute pairs with attn_act; self-gated on ASYMM_ATTN_SDPA_RECOMPUTE, text-scoped.
    install_sdpa_recompute(model)

    if strict and not wrapped:
        raise RuntimeError("attention activation offload requested but no supported text attention parents were found")
    return tuple(wrapped), tuple(skipped)


def _wrap_qwen3_decoder_saved_tensor_offload_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if _is_qwen3_decoder_layer_module_name(name, module):
            install_decoder_saved_tensor_offload(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in {"decoder_layer", "layer"} and any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and not wrapped:
        raise RuntimeError("Qwen3 decoder layer activation offload requested but no supported decoder layers were found")
    return tuple(wrapped), tuple(skipped)


def _is_qwen35_linear_attention_module_name(name: str, module: nn.Module) -> bool:
    if not name:
        return False
    if _has_attention_excluded_path_marker(name):
        return False
    if not name.lower().endswith(".linear_attn"):
        return False
    child_names = set(dict(module.named_children()))
    return _LINEAR_ATTENTION_LEAVES <= child_names


def _wrap_linear_attention_saved_tensor_offload_modules(
    model: nn.Module,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    wrapped: list[str] = []
    skipped: list[str] = []
    saw_linear_attn_name = False
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if name.lower().endswith(".linear_attn"):
            saw_linear_attn_name = True
        if _is_qwen35_linear_attention_module_name(name, module):
            install_linear_attention_saved_tensor_offload(module)
            wrapped.append(name)
            continue
        lower = f".{name.lower()}."
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf == "linear_attn" and any(marker in lower for marker in _ATTENTION_GC_EXCLUDED_PATH_MARKERS):
            skipped.append(f"{name}:vision_or_multimodal")
    if strict and saw_linear_attn_name and not wrapped:
        raise RuntimeError("Qwen3.5 linear-attention activation offload requested but no supported linear_attn modules were found")
    return tuple(wrapped), tuple(skipped)


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
    attention_act_enabled = _attention_act_offload_enabled()
    recomp_off_full_fg_enabled = _recomp_off_full_fg_enabled()
    layer_act_enabled = _layer_act_offload_enabled()
    layer_glue_gc_enabled = _layer_glue_gc_enabled()
    qwen3_moe_finegrained_enabled = _qwen3_moe_finegrained_offload_enabled()
    report.attention_act_offload_enabled = bool(attention_act_enabled)
    report.layer_act_offload_enabled = bool(layer_act_enabled)
    report.layer_glue_gc_enabled = bool(layer_glue_gc_enabled)
    report.qwen3_moe_finegrained_offload_enabled = bool(qwen3_moe_finegrained_enabled)
    wrap_experts = _targets_experts(raw_lora_target)
    offload_experts = backend == "asym" and selection.routed_experts
    offload_router = backend == "asym" and selection.router

    if layer_act_enabled and recompute_config.label != "none":
        raise RuntimeError("Qwen3 decoder layer activation offload requires expert policy none")
    if layer_act_enabled and backend != "asym":
        raise RuntimeError("Qwen3 decoder layer activation offload requires backend='asym'")
    if layer_glue_gc_enabled and recompute_config.label != "none":
        raise RuntimeError("ASYMM_LAYER_GC requires expert policy none")
    if layer_glue_gc_enabled and backend != "asym":
        raise RuntimeError("ASYMM_LAYER_GC requires backend='asym'")
    if layer_glue_gc_enabled and layer_act_enabled:
        raise RuntimeError("ASYMM_LAYER_GC and Qwen3 decoder layer activation offload are mutually exclusive")

    expert_prefixes: list[str] = []

    def _llama4_shared_expert_is_lora_target(name: str, module: nn.Module) -> bool:
        if not wrap_dense or not selection.shared_experts:
            return False
        shared = getattr(module, "shared_expert", None)
        if not isinstance(shared, nn.Module):
            return False
        leaf_supported: list[bool] = []
        leaf_targeted: list[bool] = []
        leaf_preexisting_lora: list[bool] = []
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            child = getattr(shared, leaf, None)
            leaf_supported.append(isinstance(child, nn.Module) and is_llama4_shared_mlp_leaf(child))
            leaf_targeted.append(
                _is_all_target(raw_lora_target)
                or (
                    isinstance(child, nn.Module)
                    and _matches_target(f"{name}.shared_expert.{leaf}", child, dense_target_modules)
                )
            )
            leaf_preexisting_lora.append(
                isinstance(child, nn.Module)
                and hasattr(child, "lora_A")
                and hasattr(child, "lora_B")
            )
        if not all(leaf_supported):
            return False
        return all(leaf_targeted) or all(leaf_preexisting_lora)

    def _qwen35_shared_expert_is_lora_target(name: str, module: nn.Module) -> bool:
        if not wrap_dense or not selection.shared_experts:
            return False
        shared = getattr(module, "shared_expert", None)
        if not isinstance(shared, nn.Module):
            return False
        leaf_supported: list[bool] = []
        leaf_targeted: list[bool] = []
        leaf_preexisting_lora: list[bool] = []
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            child = getattr(shared, leaf, None)
            leaf_supported.append(isinstance(child, nn.Module) and is_qwen35_shared_mlp_leaf(child))
            leaf_targeted.append(
                _is_all_target(raw_lora_target)
                or (
                    isinstance(child, nn.Module)
                    and _matches_target(f"{name}.shared_expert.{leaf}", child, dense_target_modules)
                )
            )
            leaf_preexisting_lora.append(
                isinstance(child, nn.Module)
                and hasattr(child, "lora_A")
                and hasattr(child, "lora_B")
            )
        if not all(leaf_supported):
            return False
        return all(leaf_targeted) or all(leaf_preexisting_lora)

    def _install_expert_replacement(name: str, new_module: nn.Module, skip_prefix: str) -> None:
        parent, child_name = _parent_and_child(model, name)
        _replace_child(parent, child_name, new_module)
        expert_prefixes.append(skip_prefix)
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

    if wrap_experts:
        expert_candidates: list[tuple[str, str]] = []
        if router_mode == "whole":
            for name, module in model.named_modules():
                if is_qwen35_moe_block(module):
                    expert_candidates.append((name, "qwen35_whole"))
                elif is_qwen3_moe_block(module):
                    expert_candidates.append((name, "qwen3_whole"))
                elif is_llama4_moe(module):
                    expert_candidates.append((name, "llama4_whole"))
        else:
            for name, module in model.named_modules():
                if is_qwen3_experts(module):
                    expert_candidates.append((name, "qwen3_experts"))
                elif is_llama4_moe(module):
                    expert_candidates.append((name, "llama4_hf"))

        for name, kind in expert_candidates:
            if not name or _is_under(name, expert_prefixes):
                continue
            try:
                module = model.get_submodule(name)
            except AttributeError:
                if strict:
                    raise RuntimeError(f"{name} disappeared while installing routed expert wrappers")
                report.skipped.append(f"{name}:missing_during_expert_wrap")
                continue

            if kind == "qwen35_whole":
                if not is_qwen35_moe_block(module):
                    if strict:
                        raise RuntimeError(f"{name} no longer looks like a Qwen3.5 MoE block")
                    report.skipped.append(f"{name}:stale_qwen35_moe_block")
                    continue
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
                        wrap_shared_expert=_qwen35_shared_expert_is_lora_target(name, module),
                        offload_shared_expert=backend == "asym" and selection.shared_experts,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                )
                wrapped.profile_prefix = _layer_profile_prefix_from_module_name(name, "mlp")
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                if qwen3_moe_finegrained_enabled and offload_experts:
                    wrapped.experts._qwen3_moe_finegrained_enabled = True
                    report.qwen3_moe_finegrained_offload_wrapped += 1
                if isinstance(wrapped.shared_expert, AsymQwen35SharedMLP):
                    wrapped.shared_expert.profile_prefix = f"{wrapped.profile_prefix}.shared_expert"
                    report.dense_lora_wrapped += max(0, 3 - int(wrapped.shared_expert.preexisting_lora_leaf_count))
                _install_expert_replacement(name, wrapped, f"{name}.experts")
            elif kind == "qwen3_whole":
                if not is_qwen3_moe_block(module):
                    if strict:
                        raise RuntimeError(f"{name} no longer looks like a Qwen3 MoE block")
                    report.skipped.append(f"{name}:stale_qwen3_moe_block")
                    continue
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
                if qwen3_moe_finegrained_enabled and offload_experts:
                    wrapped.experts._qwen3_moe_finegrained_enabled = True
                    report.qwen3_moe_finegrained_offload_wrapped += 1
                _install_expert_replacement(name, wrapped, f"{name}.experts")
            elif kind in {"llama4_whole", "llama4_hf"}:
                if not is_llama4_moe(module):
                    if strict:
                        raise RuntimeError(f"{name} no longer looks like a Llama4 MoE block")
                    report.skipped.append(f"{name}:stale_llama4_moe")
                    continue
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
                        router_mode="whole" if kind == "llama4_whole" else "hf",
                        offload_router=offload_router,
                        wrap_shared_expert=_llama4_shared_expert_is_lora_target(name, module),
                        offload_shared_expert=backend == "asym" and selection.shared_experts,
                        router_debug_grad=False,
                        stats=stats,
                        strict=strict,
                )
                wrapped.profile_prefix = _llama4_profile_prefix_from_module_name(name)
                wrapped.experts.profile_prefix = f"{wrapped.profile_prefix}.experts"
                if isinstance(wrapped.shared_expert, AsymLlama4SharedMLP):
                    wrapped.shared_expert.profile_prefix = f"{wrapped.profile_prefix}.shared_expert"
                    report.dense_lora_wrapped += max(0, 3 - int(wrapped.shared_expert.preexisting_lora_leaf_count))
                _install_expert_replacement(name, wrapped, f"{name}.experts")
            elif kind == "qwen3_experts":
                if not is_qwen3_experts(module):
                    if strict:
                        raise RuntimeError(f"{name} no longer looks like a Qwen3 packed expert module")
                    report.skipped.append(f"{name}:stale_qwen3_experts")
                    continue
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
                if family != "gemma4" and qwen3_moe_finegrained_enabled and offload_experts:
                    wrapped._qwen3_moe_finegrained_enabled = True
                    report.qwen3_moe_finegrained_offload_wrapped += 1
                _install_expert_replacement(name, wrapped, name)
            else:
                raise AssertionError(f"unknown expert candidate kind: {kind}")

            del module
            if backend == "asym" and offload_experts:
                _release_replaced_module_memory()

        # Dense models (e.g. Qwen/Qwen3-32B) have no packed-expert/MoE blocks, but
        # `--lora_target all` still sets wrap_experts. When the MoE detectors find ZERO
        # candidates the model is genuinely dense: record it and skip expert wrapping.
        # MoE models (Qwen3-MoE / Qwen3.5 / Llama4) ALWAYS produce candidates, so they
        # fall through to the original strict guard below and are unaffected.
        dense_no_experts = not expert_candidates
        if dense_no_experts:
            report.skipped.append("routed_experts:dense_model_no_experts")
        elif not expert_prefixes and strict:
            raise ValueError("AsymGEMM requested routed expert LoRA but found no supported packed expert/MoE modules.")

    dense_mlp_finegrained_enabled = (
        backend == "asym"
        and not expert_prefixes
        and (
            _env_true(os.environ.get("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD"))
            or _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD"))
        )
    )
    report.dense_mlp_finegrained_offload_enabled = bool(dense_mlp_finegrained_enabled)
    if dense_mlp_finegrained_enabled:
        from asym_gemm.training.dense_mlp import is_dense_mlp_module
        from asym_gemm.training.dense_mlp_finegrained import build_finegrained_dense_mlp

        dense_mlp_names = [
            name
            for name, module in model.named_modules()
            if name
            and not _is_under(name, expert_prefixes)
            and not _is_router_module_name(name)
            and not _is_vision_or_multimodal_path(name)
            and is_dense_mlp_module(module)
        ]
        for name in dense_mlp_names:
            module = model.get_submodule(name)
            wrapped = build_finegrained_dense_mlp(
                module,
                backend=backend,
                precision=precision,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                stats=stats,
                strict=strict,
                profile_prefix=_qwen3_profile_prefix_from_module_name(name),
            )
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, wrapped)
            expert_prefixes.append(name)
            report.dense_mlp_finegrained_offload_wrapped += 1
            report.cpu_resident_base_bytes += int(getattr(wrapped, "cpu_resident_base_bytes", 0))
            report.gpu_resident_base_bytes += int(getattr(wrapped, "gpu_resident_base_bytes", 0))
            del module
            _release_replaced_module_memory()

    # Dense-MLP surgical activation offload. On a DENSE model the MLP is the analog of routed
    # experts, so `ASYMM_EXPERT_ACT_OFFLOAD` should offload its `silu(gate)*up` intermediate to
    # CPU and run the down-proj LoRA backward on CPU via AsymGEMM (reusing the expert engine as a
    # single expert). Composes with attn_act and layer_gc (layer_gc then catches only the
    # residual/glue). Gated on dense (no expert prefixes) so MoE models never enter here.
    # OPT-IN ONLY. The surgical dense-MLP offload is numerically correct but runs the FULL dense
    # MLP forward/backward on CPU (all tokens x the large intermediate, every layer) — practical
    # for sparse MoE experts (top-k tokens) but it stalls a dense model at large seq/batch. The
    # standard `ASYMM_EXPERT_ACT_OFFLOAD` is therefore a no-op for the dense MLP; `ASYMM_LAYER_GC`
    # offloads the MLP activation (HBM win) while keeping the backward on GPU (fast). Set
    # ASYMM_DENSE_MLP_SURGICAL_OFFLOAD=1 only to opt into the CPU-side path for small workloads.
    dense_mlp_act_enabled = (
        backend == "asym"
        and not expert_prefixes
        and not dense_mlp_finegrained_enabled
        and (
            _env_true(os.environ.get("ASYMM_DENSE_MLP_SURGICAL_OFFLOAD"))
            or _env_true(os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_DENSE_MLP_SURGICAL_OFFLOAD"))
        )
    )
    report.dense_mlp_act_offload_enabled = bool(dense_mlp_act_enabled)
    if dense_mlp_act_enabled:
        from asym_gemm.training.dense_mlp import AsymDenseMLP, build_dense_mlp_expert_engine, is_dense_mlp_module
        from asym_gemm.training.exp_act_offload_lora import require_expert_activation_offload_kernels

        require_expert_activation_offload_kernels(scope="full")
        dense_mlp_names = [
            name
            for name, module in model.named_modules()
            if name
            and not _is_under(name, expert_prefixes)
            and not _is_router_module_name(name)
            and not _is_vision_or_multimodal_path(name)
            and is_dense_mlp_module(module)
        ]
        for name in dense_mlp_names:
            module = model.get_submodule(name)
            engine = build_dense_mlp_expert_engine(
                module,
                backend=backend,
                precision=precision,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                stats=stats,
                strict=strict,
            )
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, AsymDenseMLP(engine))
            expert_prefixes.append(name)
            report.dense_mlp_act_offload_wrapped += 1
            report.cpu_resident_base_bytes += int(getattr(engine, "cpu_resident_base_bytes", 0))
            report.gpu_resident_base_bytes += int(getattr(engine, "gpu_resident_base_bytes", 0))
            del module
            _release_replaced_module_memory()

    if _attention_gc_enabled_for_policy(recompute_config.label):
        wrapped_attention, skipped_attention = _wrap_attention_checkpoint_modules(model, strict=strict)
        report.attention_gc_enabled = True
        report.attention_gc_wrapped = len(wrapped_attention)
        report.attention_gc_modules = wrapped_attention
        report.attention_gc_skipped = skipped_attention
        report.skipped.extend(skipped_attention)
        setattr(model, "_asym_attention_gc_enabled", True)
        setattr(model, "_asym_attention_gc_modules", wrapped_attention)
    else:
        setattr(model, "_asym_attention_gc_enabled", False)
        setattr(model, "_asym_attention_gc_modules", tuple())

    if _layer_gc_enabled_for_policy(recompute_config.label):
        wrapped_layers, skipped_layers = _wrap_qwen3_decoder_checkpoint_modules(model, strict=strict)
        report.layer_gc_enabled = True
        report.layer_gc_wrapped = len(wrapped_layers)
        report.layer_gc_modules = wrapped_layers
        report.layer_gc_skipped = skipped_layers
        report.skipped.extend(skipped_layers)
        setattr(model, "_asym_layer_gc_enabled", True)
        setattr(model, "_asym_layer_gc_modules", wrapped_layers)
    else:
        setattr(model, "_asym_layer_gc_enabled", False)
        setattr(model, "_asym_layer_gc_modules", tuple())

    if offload_router:
        router_names: list[str] = []
        for name, module in model.named_modules():
            if not name or _is_under(name, expert_prefixes) or isinstance(module, (AsymQwen3Router, AsymLlama4Router)):
                continue
            lower_name = name.lower()
            if not (lower_name == "router" or lower_name.endswith(".mlp.gate") or lower_name.endswith(".feed_forward.router")):
                continue
            if classify_lf_component(name, module) != "router":
                continue
            router_names.append(name)

        for name in router_names:
            module = model.get_submodule(name)
            if isinstance(module, (AsymQwen3Router, AsymLlama4Router)):
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
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, wrapped_router)
            del module
            _release_replaced_module_memory()

    if backend == "asym" and selection.embed_tokens:
        embedding_names: list[str] = []
        for name, module in model.named_modules():
            if not name or isinstance(module, AsymFrozenEmbedding):
                continue
            if classify_lf_component(name, module) != "embed_tokens":
                continue
            if not isinstance(module, nn.Embedding):
                if strict:
                    raise RuntimeError(f"{name} selected for embed_tokens CPU offload but is {type(module).__name__}")
                report.skipped.append(f"{name}:unsupported_embedding:{type(module).__name__}")
                continue
            embedding_names.append(name)

        for name in embedding_names:
            module = model.get_submodule(name)
            if isinstance(module, AsymFrozenEmbedding):
                continue
            if not isinstance(module, nn.Embedding):
                if strict:
                    raise RuntimeError(f"{name} selected for embed_tokens CPU offload but is {type(module).__name__}")
                report.skipped.append(f"{name}:unsupported_embedding:{type(module).__name__}")
                continue
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, AsymFrozenEmbedding(module))
            del module
            _release_replaced_module_memory()

    if backend == "asym" and selection.norms:
        norm_names: list[str] = []
        for name, module in model.named_modules():
            if not name or isinstance(module, (AsymFrozenLayerNorm, AsymFrozenRMSNorm)):
                continue
            if classify_lf_component(name, module) != "norms":
                continue
            if isinstance(module, nn.LayerNorm):
                norm_names.append(name)
                continue
            weight = getattr(module, "weight", None)
            class_name = type(module).__name__.lower()
            if isinstance(weight, torch.Tensor) and (
                "rmsnorm" in class_name or "rms_norm" in class_name or hasattr(module, "variance_epsilon")
            ):
                norm_names.append(name)
                continue
            if _is_stateless_module(module):
                continue
            if strict:
                raise RuntimeError(f"{name} selected for norms CPU offload but is {type(module).__name__}")
            report.skipped.append(f"{name}:unsupported_norm:{type(module).__name__}")

        for name in norm_names:
            module = model.get_submodule(name)
            if isinstance(module, (AsymFrozenLayerNorm, AsymFrozenRMSNorm)):
                continue
            if isinstance(module, nn.LayerNorm):
                new_module = AsymFrozenLayerNorm(module, strict=strict)
            else:
                weight = getattr(module, "weight", None)
                class_name = type(module).__name__.lower()
                if isinstance(weight, torch.Tensor) and (
                    "rmsnorm" in class_name or "rms_norm" in class_name or hasattr(module, "variance_epsilon")
                ):
                    new_module = AsymFrozenRMSNorm(module)
                else:
                    if strict:
                        raise RuntimeError(f"{name} selected for norms CPU offload but is {type(module).__name__}")
                    report.skipped.append(f"{name}:unsupported_norm:{type(module).__name__}")
                    continue
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)
            del module
            _release_replaced_module_memory()

    dense_replacement_names: list[str] = []
    attention_act_modules: list[str] = []
    attention_act_skipped: list[str] = []
    attention_saved_modules: tuple[str, ...] = ()
    attention_saved_skipped: tuple[str, ...] = ()
    attention_contexts = _build_attention_activation_contexts(
        model,
        selection=selection,
        dense_target_modules=dense_target_modules,
        backend=backend,
        attention_act_enabled=attention_act_enabled,
    )
    if wrap_dense:
        for name, module in model.named_modules():
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
            dense_replacement_names.append(name)

        for name in dense_replacement_names:
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                report.skipped.append(f"{name}:not_nn_linear:{type(module).__name__}")
                continue
            component = classify_lf_component(name, module)
            leaf = name.rsplit(".", 1)[-1]
            selected_cpu_offload = backend == "asym" and component_is_selected(component, leaf, selection)
            is_lora_target = _matches_target(name, module, dense_target_modules)
            shape_fallback_reason = (
                _direct_bf16_linear_shape_reason(module, require_backward=True)
                if selected_cpu_offload and backend == "asym"
                else None
            )
            if (
                attention_act_enabled
                and component == "attention"
                and is_lora_target
                and selected_cpu_offload
                and _has_attention_excluded_path_marker(name)
            ):
                attention_act_skipped.append(f"{name}:vision_or_multimodal")
            attention_context = (
                attention_contexts.get(_attention_parent_name(name) or "")
                if leaf in {"q_proj", "k_proj", "v_proj"}
                else None
            )
            new_module = _wrap_lf_linear_leaf(
                name,
                module,
                component=component,
                is_lora_target=is_lora_target,
                selected_cpu_offload=selected_cpu_offload,
                attention_act_offload_enabled=attention_act_enabled,
                attention_context=attention_context,
                backend=backend,
                precision=precision,
                stats=stats,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                strict=strict,
            )
            if shape_fallback_reason is not None:
                report.skipped.append(f"{name}:torch_cpu_fetched:{shape_fallback_reason}")
            parent, child_name = _parent_and_child(model, name)
            _replace_child(parent, child_name, new_module)
            if is_lora_target:
                report.dense_lora_wrapped += 1
            if isinstance(new_module, AsymActivationOffloadLoRALinear):
                attention_act_modules.append(name)
            report.gpu_resident_base_bytes += int(getattr(new_module, "gpu_resident_base_weight_bytes", 0))
            del module
            if selected_cpu_offload:
                _release_replaced_module_memory()

    if wrap_dense and _is_all_target(raw_lora_target) and not dense_replacement_names and report.dense_lora_wrapped == 0 and strict:
        raise ValueError("AsymGEMM requested dense LoRA target=all but found no dense nn.Linear modules.")
    if attention_act_enabled and attention_contexts:
        attention_saved_modules, attention_saved_skipped = _wrap_attention_saved_tensor_offload_modules(
            model,
            parent_names=tuple(attention_contexts),
            strict=strict,
        )
    report.attention_act_offload_wrapped = len(attention_act_modules)
    report.attention_act_offload_modules = tuple(attention_act_modules)
    report.attention_act_offload_skipped = tuple(attention_act_skipped)
    if attention_act_skipped:
        report.skipped.extend(attention_act_skipped)
    report.attention_saved_tensor_offload_wrapped = len(attention_saved_modules)

    if os.environ.get("ASYM_STP") == "1" and os.environ.get("ASYM_STP_PHASE_A") != "1":
        # gb200_tp.md I4: convert every wrapped dense decoder layer into a two-branch
        # StpDecoderLayer over the sharded arena. Runs after the standard wrap (so the
        # full-weight HostWeights + LoRA params exist to slice from) and before
        # freeze/validate (branch params must be visible to both).
        from asym_gemm.training.stp_runtime import get_runtime as _stp_get_runtime
        from asym_gemm.training.stp_wrap import build_stp_full_tp

        _stp_get_runtime()
        stp_full_counts = build_stp_full_tp(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            backend=backend,
            stats=stats,
            strict=strict,
        )
        setattr(model, "_asym_stp_full_counts", stp_full_counts)
        # branch1 saved-tensor offload is installed inside build_stp_full_tp (the name-based
        # helper rejects the self_attn_stp1 leaf).
        _release_replaced_module_memory()
    report.attention_saved_tensor_offload_modules = tuple(attention_saved_modules)
    report.attention_saved_tensor_offload_skipped = tuple(attention_saved_skipped)
    if attention_saved_skipped:
        report.skipped.extend(attention_saved_skipped)
    setattr(model, "_asym_attention_act_offload_enabled", bool(attention_act_enabled))
    setattr(model, "_asym_attention_act_offload_modules", tuple(attention_act_modules))
    setattr(model, "_asym_attention_act_offload_skipped", tuple(attention_act_skipped))
    setattr(model, "_asym_attention_saved_tensor_offload_modules", tuple(attention_saved_modules))
    setattr(model, "_asym_attention_saved_tensor_offload_skipped", tuple(attention_saved_skipped))

    layer_saved_modules: tuple[str, ...] = ()
    layer_saved_skipped: tuple[str, ...] = ()
    linear_attention_saved_modules: tuple[str, ...] = ()
    linear_attention_saved_skipped: tuple[str, ...] = ()
    if layer_act_enabled:
        layer_saved_modules, layer_saved_skipped = _wrap_qwen3_decoder_saved_tensor_offload_modules(
            model,
            strict=strict,
        )
    linear_attention_full_fg_enabled = backend == "asym" and recomp_off_full_fg_enabled and attention_act_enabled
    if (layer_act_enabled or layer_glue_gc_enabled or linear_attention_full_fg_enabled) and selection.linear_attention:
        linear_attention_saved_modules, linear_attention_saved_skipped = _wrap_linear_attention_saved_tensor_offload_modules(
            model,
            strict=strict,
        )
    report.layer_act_offload_wrapped = len(layer_saved_modules)
    report.layer_act_offload_modules = tuple(layer_saved_modules)
    report.layer_act_offload_skipped = tuple(layer_saved_skipped)
    if layer_saved_skipped:
        report.skipped.extend(layer_saved_skipped)
    report.linear_attention_saved_tensor_offload_wrapped = len(linear_attention_saved_modules)
    report.linear_attention_saved_tensor_offload_modules = tuple(linear_attention_saved_modules)
    report.linear_attention_saved_tensor_offload_skipped = tuple(linear_attention_saved_skipped)
    if linear_attention_saved_skipped:
        report.skipped.extend(linear_attention_saved_skipped)
    setattr(model, "_asym_layer_act_offload_enabled", bool(layer_act_enabled))
    setattr(model, "_asym_layer_act_offload_modules", tuple(layer_saved_modules))
    setattr(model, "_asym_layer_act_offload_skipped", tuple(layer_saved_skipped))
    setattr(model, "_asym_linear_attention_saved_tensor_offload_modules", tuple(linear_attention_saved_modules))
    setattr(model, "_asym_linear_attention_saved_tensor_offload_skipped", tuple(linear_attention_saved_skipped))

    layer_glue_modules: tuple[str, ...] = ()
    layer_glue_skipped: tuple[str, ...] = ()
    if layer_glue_gc_enabled:
        layer_glue_modules, layer_glue_skipped = _wrap_decoder_layer_glue_gc_modules(
            model,
            strict=strict,
        )
    report.layer_glue_gc_wrapped = len(layer_glue_modules)
    report.layer_glue_gc_modules = tuple(layer_glue_modules)
    report.layer_glue_gc_skipped = tuple(layer_glue_skipped)
    if layer_glue_skipped:
        report.skipped.extend(layer_glue_skipped)
    setattr(model, "_asym_layer_glue_gc_enabled", bool(layer_glue_gc_enabled))
    setattr(model, "_asym_layer_glue_gc_modules", tuple(layer_glue_modules))
    setattr(model, "_asym_layer_glue_gc_skipped", tuple(layer_glue_skipped))

    # DISABLED (superseded by the fine-grained recomp-off design, agent/impls/finegrained_offload.md). The
    # token-chunked dense MLP (chunked_mlp.py / ASYMM_MLP_RECOMPUTE_CHUNK) is kept as source but NOT wired:
    # the recomp-off design reduces the [M,I] peak via CPU-elementwise placement, not token tiling. To re-enable,
    # uncomment the two lines below.
    # chunked_mlp_modules = install_chunked_mlp_on_dense_mlps(model)
    # setattr(model, "_asym_chunked_mlp_modules", tuple(chunked_mlp_modules))
    setattr(model, "_asym_chunked_mlp_modules", ())

    freeze_non_lora_params(model)
    _validate_trainable_params(model)
    residency_selection = selection if backend == "asym" else LFOffloadSelection(raw=selection.raw)
    rows = validate_lf_offload_residency(model, residency_selection, strict=strict, classify_component=classify_lf_component)
    build_lf_asym_report(report, rows, selection=selection)
    setattr(model, "_asym_offload_modules", selection.raw)
    setattr(model, "_asym_execution_stats", stats)
    report.trainable_lora_params = sum(param.numel() for param in lora_parameters(model))
    if report.trainable_lora_params == 0 and strict:
        raise ValueError("AsymGEMM setup produced zero trainable LoRA parameters.")
    _register_runtime_report(report)
    if os.environ.get("ASYM_STP") == "1" and os.environ.get("ASYM_STP_PHASE_A") != "1":
        # gb200_tp.md I4 (placement, deferred past residency validation): move all
        # still-CPU params/buffers to dev0, keep branch1 on dev1, and advertise model
        # parallelism so the HF Trainer skips its blanket cuda:0 move.
        from asym_gemm.training.stp_runtime import get_runtime as _stp_rt
        from asym_gemm.training.stp_wrap import finalize_stp_placement

        finalize_stp_placement(model, _stp_rt().d[0])
    if os.environ.get("ASYM_STP") == "1" and os.environ.get("ASYM_STP_PHASE_A") == "1":
        # Phase-A fallback/ablation path (gb200_tp.md I3): model stays on dev0; ONLY the
        # base GEMMs split via the _stp routing in asym_bf16_cpu_right_matmul.
        # gb200_tp.md I1/I2: bring up the pair runtime (peer enable + JIT prewarm on BOTH
        # devices) and repack every wrapped frozen weight into contiguous per-device shards
        # BEFORE the first forward. Runs after all shape-consuming setup; repack transient
        # stays under one layer (row carriers swap in place, col shards are views).
        from asym_gemm.training.stp_layout import repack_model_for_stp
        from asym_gemm.training.stp_runtime import get_runtime

        get_runtime()
        stp_named_host_weights = []
        seen_host_weight_ids: set[int] = set()
        for stp_mod_name, stp_module in model.named_modules():
            candidate_hw = getattr(stp_module, "host_weight", None)
            if candidate_hw is None or id(candidate_hw) in seen_host_weight_ids:
                continue
            seen_host_weight_ids.add(id(candidate_hw))
            stp_named_host_weights.append((stp_mod_name, candidate_hw))
        stp_repack_counts = repack_model_for_stp(stp_named_host_weights)
        setattr(model, "_asym_stp_repack_counts", stp_repack_counts)
    # Drop any orphaned pre-conversion modules/reference cycles before the first trainer step.
    _release_replaced_module_memory()
    # asym_cpuadamwds_panvme (Stage 7): page frozen base weights to NVMe. No-op without the
    # base_weight NVMe role; runs LAST so every HostWeight (incl. lazy moefg splits) exists.
    _install_base_weight_pager(model)
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
    attention_gc_modules = tuple(getattr(model, "_asym_attention_gc_modules", ())) or attention_checkpoint_module_names(model)
    if bool(getattr(model, "_asym_attention_gc_enabled", False)) or attention_gc_modules:
        config["asym_attention_gc_enabled"] = bool(getattr(model, "_asym_attention_gc_enabled", False))
        config["asym_attention_gc_modules"] = list(attention_gc_modules)
    layer_gc_modules = tuple(getattr(model, "_asym_layer_gc_modules", ())) or decoder_checkpoint_module_names(model)
    if bool(getattr(model, "_asym_layer_gc_enabled", False)) or layer_gc_modules:
        config["asym_layer_gc_enabled"] = bool(getattr(model, "_asym_layer_gc_enabled", False))
        config["asym_layer_gc_modules"] = list(layer_gc_modules)
    layer_glue_gc_modules = tuple(getattr(model, "_asym_layer_glue_gc_modules", ())) or decoder_layer_glue_gc_module_names(model)
    config["asymm_layer_gc"] = bool(getattr(model, "_asym_layer_glue_gc_enabled", False))
    if bool(getattr(model, "_asym_layer_glue_gc_enabled", False)) or layer_glue_gc_modules:
        config["asym_layer_glue_gc_enabled"] = bool(getattr(model, "_asym_layer_glue_gc_enabled", False))
        config["asym_layer_glue_gc_modules"] = list(layer_glue_gc_modules)
        config["asym_layer_glue_gc_skipped"] = list(getattr(model, "_asym_layer_glue_gc_skipped", ()))
    attention_act_modules = tuple(getattr(model, "_asym_attention_act_offload_modules", ()))
    if bool(getattr(model, "_asym_attention_act_offload_enabled", False)) or attention_act_modules:
        attention_saved_modules = tuple(
            getattr(model, "_asym_attention_saved_tensor_offload_modules", ())
        ) or attention_saved_tensor_offload_module_names(model)
        config["asym_attention_act_offload_enabled"] = bool(
            getattr(model, "_asym_attention_act_offload_enabled", False)
        )
        config["asym_attention_act_offload_modules"] = list(attention_act_modules)
        config["asym_attention_act_offload_skipped"] = list(
            getattr(model, "_asym_attention_act_offload_skipped", ())
        )
        config["asym_attention_saved_tensor_offload_modules"] = list(attention_saved_modules)
        config["asym_attention_saved_tensor_offload_skipped"] = list(
            getattr(model, "_asym_attention_saved_tensor_offload_skipped", ())
        )
    layer_act_modules = tuple(getattr(model, "_asym_layer_act_offload_modules", ()))
    if bool(getattr(model, "_asym_layer_act_offload_enabled", False)) or layer_act_modules:
        layer_saved_modules = tuple(
            getattr(model, "_asym_layer_act_offload_modules", ())
        ) or decoder_saved_tensor_offload_module_names(model)
        config["asym_layer_act_offload_enabled"] = bool(getattr(model, "_asym_layer_act_offload_enabled", False))
        config["asym_layer_act_offload_modules"] = list(layer_act_modules)
        config["asym_layer_act_offload_skipped"] = list(getattr(model, "_asym_layer_act_offload_skipped", ()))
        config["asym_decoder_saved_tensor_offload_modules"] = list(layer_saved_modules)
    linear_attention_saved_modules = tuple(
        getattr(model, "_asym_linear_attention_saved_tensor_offload_modules", ())
    ) or linear_attention_saved_tensor_offload_module_names(model)
    if linear_attention_saved_modules:
        config["asym_linear_attention_saved_tensor_offload_modules"] = list(linear_attention_saved_modules)
        config["asym_linear_attention_saved_tensor_offload_skipped"] = list(
            getattr(model, "_asym_linear_attention_saved_tensor_offload_skipped", ())
        )
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
    "install_lf_frozen_multimodal_cpu_staging",
    "load_asym_peft_adapter",
    "audit_lf_frozen_cuda_residue",
    "drop_lf_frozen_vision_tower_for_text_only_profile",
    "move_lf_asym_cpu_first_model_to_device",
    "parse_lf_offload_modules",
    "save_asym_peft_adapter",
    "summarize_lf_tensor_devices",
]
