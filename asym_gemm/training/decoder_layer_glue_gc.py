from __future__ import annotations

import inspect
import types
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint

from .decoder_activation_offload import DecoderSavedTensorOffloadWrapper


_FORWARD_RESERVED_KWARGS = frozenset(
    {
        "hidden_states",
        "attention_mask",
        "position_ids",
        "past_key_value",
        "past_key_values",
        "cache_position",
        "cache_params",
        "use_cache",
        "position_embeddings",
    }
)


def _has_var_keyword(signature: inspect.Signature) -> bool:
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _call_with_supported_kwargs(module: nn.Module, kwargs: Mapping[str, Any]) -> Any:
    wrapper = getattr(module, "_asym_linear_attention_saved_tensor_offload_wrapper", None)
    forward = getattr(wrapper, "original_forward", None)
    if forward is None:
        wrapper = getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None)
        forward = getattr(wrapper, "original_forward", None)
    if forward is None:
        forward = getattr(module, "forward")
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return module(**dict(kwargs))
    if _has_var_keyword(signature):
        return module(**dict(kwargs))
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return module(**filtered)


def _flatten_bound_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(arguments)
    extra = values.pop("kwargs", None)
    if isinstance(extra, Mapping):
        values.update(extra)
    return values


def _layer_family(layer: nn.Module) -> str:
    class_name = type(layer).__name__.lower()
    module_name = type(layer).__module__.lower()
    config = getattr(layer, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if (
        "qwen3_5" in class_name
        or "qwen3_5" in module_name
        or "qwen35" in class_name.replace("_", "")
        or "qwen35" in module_name.replace("_", "")
        or model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text"}
    ):
        return "qwen35"
    if "llama4" in class_name or "llama4" in module_name or model_type in {"llama4", "llama4_text"}:
        return "llama4"
    return "qwen3"


def _position_ids_for_qwen35_full_attention(position_ids: Any) -> Any:
    if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3:
        return position_ids[None, 0]
    return position_ids


def _extra_attention_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key not in _FORWARD_RESERVED_KWARGS}


def _call_self_attention(layer: nn.Module, normed: torch.Tensor, values: Mapping[str, Any]) -> Any:
    family = _layer_family(layer)
    attention = getattr(layer, "self_attn")
    extra = _extra_attention_kwargs(values)
    if family == "llama4":
        call_kwargs = {
            "hidden_states": normed,
            "position_embeddings": values.get("position_embeddings"),
            "attention_mask": values.get("attention_mask"),
            "past_key_values": values.get("past_key_values", values.get("past_key_value")),
            "use_cache": bool(values.get("use_cache", False)),
            "cache_position": values.get("cache_position"),
            **extra,
        }
    else:
        position_ids = values.get("position_ids")
        if family == "qwen35":
            position_ids = _position_ids_for_qwen35_full_attention(position_ids)
        call_kwargs = {
            "hidden_states": normed,
            "attention_mask": values.get("attention_mask"),
            "position_ids": position_ids,
            "past_key_values": values.get("past_key_values", values.get("past_key_value")),
            "use_cache": bool(values.get("use_cache", False)),
            "cache_position": values.get("cache_position"),
            "position_embeddings": values.get("position_embeddings"),
            **extra,
        }
    return _call_with_supported_kwargs(attention, call_kwargs)


def _call_qwen35_linear_attention(layer: nn.Module, normed: torch.Tensor, values: Mapping[str, Any]) -> Any:
    call_kwargs = {
        "hidden_states": normed,
        "cache_params": values.get("past_key_values", values.get("past_key_value")),
        "cache_position": values.get("cache_position"),
        "attention_mask": values.get("attention_mask"),
        "position_ids": values.get("position_ids"),
    }
    return _call_with_supported_kwargs(getattr(layer, "linear_attn"), call_kwargs)


class DecoderLayerGlueGCWrapper:
    """Checkpoint only decoder-layer glue work while leaving core modules intact."""

    def __init__(
        self,
        module: nn.Module,
        *,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.forward_signature = inspect.signature(self.original_forward)
        self.use_reentrant = bool(use_reentrant)
        self.preserve_rng_state = bool(preserve_rng_state)
        self.saved_tensor_offload = DecoderSavedTensorOffloadWrapper(module)
        self.calls = 0
        self.checkpoint_norm_calls = 0
        self.skipped_cache_calls = 0
        self.fallback_calls = 0
        self.saved_tensor_offload_calls = 0

    def install(self) -> None:
        setattr(self.module, "_asym_decoder_layer_glue_gc_wrapper", self)
        self.module.forward = types.MethodType(_decoder_layer_glue_gc_forward, self.module)  # type: ignore[method-assign]

    def _bind_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            bound = self.forward_signature.bind_partial(*args, **dict(kwargs))
        except TypeError:
            return None
        values = _flatten_bound_arguments(bound.arguments)
        if "hidden_states" not in values and args:
            values["hidden_states"] = args[0]
        return values

    def _checkpoint_norm(self, norm: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.requires_grad:
            return norm(hidden_states)

        def body(tensor: torch.Tensor) -> torch.Tensor:
            return norm(tensor)

        self.checkpoint_norm_calls += 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True.*")
            return checkpoint(
                body,
                hidden_states,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng_state,
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if bool(kwargs.get("use_cache", False)):
            self.skipped_cache_calls += 1
            return self.original_forward(*args, **kwargs)
        values = self._bind_inputs(args, kwargs)
        if values is None or not isinstance(values.get("hidden_states"), torch.Tensor):
            self.fallback_calls += 1
            return self.original_forward(*args, **kwargs)
        self.saved_tensor_offload.calls += 1
        self.saved_tensor_offload_calls += 1
        self.saved_tensor_offload._sync_module_stats()
        with saved_tensors_hooks(self.saved_tensor_offload._pack, self.saved_tensor_offload._unpack):
            return self._manual_forward(values)

    def _manual_forward(self, values: Mapping[str, Any]) -> torch.Tensor:
        layer = self.module
        hidden_states = values["hidden_states"]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a tensor")

        residual = hidden_states
        normed = self._checkpoint_norm(layer.input_layernorm, hidden_states)

        if getattr(layer, "layer_type", None) == "linear_attention" and hasattr(layer, "linear_attn"):
            mixer_out = _call_qwen35_linear_attention(layer, normed, values)
        else:
            mixer_out = _call_self_attention(layer, normed, values)
        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]
        hidden_states = residual + mixer_out

        residual = hidden_states
        normed = self._checkpoint_norm(layer.post_attention_layernorm, hidden_states)
        if hasattr(layer, "feed_forward"):
            mlp_out = layer.feed_forward(normed)
        else:
            mlp_out = layer.mlp(normed)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        if hasattr(layer, "feed_forward"):
            mlp_out = mlp_out.view(residual.shape)
        return residual + mlp_out


def _decoder_layer_glue_gc_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None)
    if not isinstance(wrapper, DecoderLayerGlueGCWrapper):
        raise RuntimeError("decoder layer glue GC wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_decoder_layer_glue_gc(module: nn.Module) -> DecoderLayerGlueGCWrapper:
    existing = getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None)
    if isinstance(existing, DecoderLayerGlueGCWrapper):
        return existing
    wrapper = DecoderLayerGlueGCWrapper(module)
    wrapper.install()
    return wrapper


def is_decoder_layer_glue_gc_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_decoder_layer_glue_gc_wrapper", None), DecoderLayerGlueGCWrapper)


def decoder_layer_glue_gc_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, module in model.named_modules() if name and is_decoder_layer_glue_gc_wrapper(module))


__all__ = [
    "DecoderLayerGlueGCWrapper",
    "decoder_layer_glue_gc_module_names",
    "install_decoder_layer_glue_gc",
    "is_decoder_layer_glue_gc_wrapper",
]
