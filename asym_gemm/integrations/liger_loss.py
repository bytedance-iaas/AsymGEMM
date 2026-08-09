from __future__ import annotations

from types import MethodType
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
from transformers.utils import can_return_tuple

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast
from liger_kernel.transformers.model.output_classes import LigerQwen3_5MoeCausalLMOutputWithPast

try:
    from liger_kernel.transformers.model.output_classes import LigerQwen3_5CausalLMOutputWithPast
except ImportError:  # pragma: no cover - compatibility with older local Liger checkouts
    LigerQwen3_5CausalLMOutputWithPast = None


# ---------------------------------------------------------------------------
# Model-shape resolution: PEFT wrappers, conditional-generation, nested LMs.
# ---------------------------------------------------------------------------
def _root_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base = get_base_model()
        except Exception:
            base = None
        if isinstance(base, nn.Module):
            return base
    return model


def _candidate_language_models(model: nn.Module) -> list[nn.Module]:
    root = _root_model(model)
    candidates = [model, root]

    for candidate in list(candidates):
        language_model = getattr(candidate, "language_model", None)
        if isinstance(language_model, nn.Module):
            candidates.append(language_model)

        inner = getattr(candidate, "model", None)
        if isinstance(inner, nn.Module):
            candidates.append(inner)
            inner_language_model = getattr(inner, "language_model", None)
            if isinstance(inner_language_model, nn.Module):
                candidates.append(inner_language_model)

    deduped = []
    seen = set()
    for candidate in candidates:
        ident = id(candidate)
        if isinstance(candidate, nn.Module) and ident not in seen:
            deduped.append(candidate)
            seen.add(ident)
    return deduped


def _base_causal_lm_model(model: nn.Module) -> nn.Module:
    for candidate in _candidate_language_models(model):
        if hasattr(candidate, "lm_head") and hasattr(candidate, "model"):
            return candidate
    return model


def _is_llama4_conditional_generation(model: nn.Module) -> bool:
    root = _root_model(model)
    return (
        getattr(getattr(root, "config", None), "model_type", None) == "llama4"
        and isinstance(getattr(root, "language_model", None), nn.Module)
        and hasattr(root.language_model, "lm_head")
        and hasattr(root.language_model, "model")
    )


# ---------------------------------------------------------------------------
# lm_head weight resolution + validation (shared across model types).
# ---------------------------------------------------------------------------
def _resolve_liger_lm_head_weight(
    lm_head: nn.Module, hidden_states: torch.Tensor, *, allow_bias: bool = False
) -> torch.Tensor:
    resolver = getattr(lm_head, "asym_liger_lm_head_weight", None)
    if callable(resolver):
        if allow_bias:
            return resolver(device=hidden_states.device, dtype=hidden_states.dtype, allow_bias=True)
        return resolver(device=hidden_states.device, dtype=hidden_states.dtype)

    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise TypeError(f"lm_head has no weight for Liger fused CE: {type(lm_head).__name__}")
    return weight


def _lm_head_weight_source(lm_head: nn.Module) -> str:
    if callable(getattr(lm_head, "asym_liger_lm_head_weight", None)):
        return "asym_host_staged"
    if getattr(lm_head, "weight", None) is not None:
        return "normal_parameter"
    return "unavailable"


def _validate_liger_lm_head(lm_head: nn.Module | None, *, model_label: str, strict: bool, allow_bias: bool = False):
    if lm_head is None:
        if strict:
            raise RuntimeError(f"{model_label} model has no lm_head.")
        return None
    if not isinstance(lm_head, nn.Module):
        raise RuntimeError(f"{model_label} lm_head is not a torch module: {type(lm_head).__name__}.")

    weight_source = _lm_head_weight_source(lm_head)
    if weight_source == "unavailable":
        if strict:
            raise RuntimeError("lm_head is not compatible with Liger fused CE weight resolution.")
        return None

    if getattr(lm_head, "bias", None) is not None or getattr(lm_head, "bias_cpu", None) is not None:
        if not allow_bias:
            raise RuntimeError("Liger loss bridge currently requires a bias-free lm_head.")

    if any(param.requires_grad for param in lm_head.parameters(recurse=True)):
        raise RuntimeError("Liger loss bridge supports frozen lm_head only.")

    return lm_head, weight_source


# ---------------------------------------------------------------------------
# Bridge metadata for source profiling.
# ---------------------------------------------------------------------------
def _mark_liger_bridge_installed(target_model, lm_head, weight_source, model_type, bridge_kind):
    target_model._asym_liger_lm_head_bridge_enabled = True
    target_model._asym_liger_lm_head_weight_source = weight_source
    target_model._asym_liger_lm_head_type = type(lm_head).__name__
    target_model._asym_liger_lm_head_staged_bytes = int(getattr(lm_head, "cpu_resident_base_weight_bytes", 0) or 0)
    target_model._asym_liger_model_type = model_type
    target_model._asym_liger_bridge_kind = bridge_kind


def _bridge_metadata_target(model: nn.Module) -> nn.Module:
    # Conditional Llama4 is marked on the top-level wrapper, while causal-LM
    # bridges are marked on the causal LM. Check both before falling back.
    candidates = [model, _root_model(model), *_candidate_language_models(model)]
    seen = set()
    for candidate in candidates:
        ident = id(candidate)
        if not isinstance(candidate, nn.Module) or ident in seen:
            continue
        seen.add(ident)
        if getattr(candidate, "_asym_liger_lm_head_bridge_enabled", False):
            return candidate
    return _base_causal_lm_model(model)


def asym_liger_lm_head_bridge_metadata(model: nn.Module) -> dict[str, Any]:
    target_model = _bridge_metadata_target(model)
    enabled = bool(getattr(target_model, "_asym_liger_lm_head_bridge_enabled", False))
    lm_head = getattr(target_model, "lm_head", None) or getattr(
        getattr(target_model, "language_model", None), "lm_head", None
    )
    if not enabled:
        return {
            "enabled": False,
            "weight_source": "disabled",
            "lm_head_type": type(lm_head).__name__ if lm_head is not None else None,
            "model_type": None,
            "bridge_kind": None,
        }
    return {
        "enabled": True,
        "weight_source": str(getattr(target_model, "_asym_liger_lm_head_weight_source", "unknown")),
        "staged_bytes": int(getattr(target_model, "_asym_liger_lm_head_staged_bytes", 0) or 0),
        "lm_head_type": str(getattr(target_model, "_asym_liger_lm_head_type", "unknown")),
        "model_type": str(getattr(target_model, "_asym_liger_model_type", "unknown")),
        "bridge_kind": str(getattr(target_model, "_asym_liger_bridge_kind", "unknown")),
    }


# ---------------------------------------------------------------------------
# Qwen3-MoE bridge (unchanged behavior).
# ---------------------------------------------------------------------------
def asym_qwen3_moe_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs: Any,
) -> LigerMoeCausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **kwargs,
            )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((aux_loss,) + output) if aux_loss is not None else output
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def install_asym_liger_qwen3_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_moe":
        if strict:
            raise ValueError("Asym Liger loss bridge only supports qwen3_moe.")
        return False

    validated = _validate_liger_lm_head(getattr(target_model, "lm_head", None), model_label="Qwen3-MoE", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated

    target_model.forward = MethodType(asym_qwen3_moe_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, "qwen3_moe", "causal_lm")
    return True


# ---------------------------------------------------------------------------
# Llama4 shift-label helpers (conditional bridge).
# ---------------------------------------------------------------------------
def _select_sequence_positions(tensor: torch.Tensor, slice_indices: slice | torch.Tensor) -> torch.Tensor:
    if isinstance(slice_indices, slice):
        return tensor[:, slice_indices].contiguous()
    if isinstance(slice_indices, torch.Tensor):
        return tensor.index_select(1, slice_indices.to(device=tensor.device, dtype=torch.long)).contiguous()
    return tensor[:, slice_indices].contiguous()


def _make_liger_shift_labels(labels, attention_mask, *, slice_indices: slice | torch.Tensor, ignore_index: int = -100):
    if labels is None:
        return None

    shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    if attention_mask is not None:
        active = torch.nn.functional.pad(attention_mask[..., 1:], (0, 1), value=0).to(dtype=torch.bool)
        shifted = shifted.masked_fill(~active.to(device=shifted.device), ignore_index)

    return _select_sequence_positions(shifted, slice_indices)


def _coerce_existing_shift_labels(
    shift_labels: torch.Tensor,
    *,
    slice_indices: slice | torch.Tensor,
    full_seq_len: int,
    kept_seq_len: int,
) -> torch.Tensor:
    if shift_labels.shape[1] == full_seq_len:
        return _select_sequence_positions(shift_labels, slice_indices)
    if shift_labels.shape[1] == kept_seq_len:
        return shift_labels.contiguous()
    raise ValueError(
        f"shift_labels length {shift_labels.shape[1]} does not match full seq {full_seq_len} or kept seq {kept_seq_len}"
    )


# ---------------------------------------------------------------------------
# Llama4 causal-LM bridge (llama4_text and nested causal LMs).
# Verbatim copy of Liger's llama4 lce_forward; the ONLY change is the
# lm_head_weight resolution so an Asym-staged / CPU-resident lm_head works.
# ---------------------------------------------------------------------------
def asym_llama4_causal_lm_lce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[tuple, LigerCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if self.training and (labels is not None or shift_labels is not None):
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)  # <-- only change vs Liger template
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:  # inference: materialize logits
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


# ---------------------------------------------------------------------------
# Llama4 conditional-generation bridge (real model_type="llama4").
# Mirrors Llama4ForConditionalGeneration.forward through embed/image merging,
# but computes the fused loss off language_model.model(...) hidden states so the
# top-level [batch, seq, vocab] logits are never materialized in training.
# ---------------------------------------------------------------------------
def asym_llama4_conditional_lce_forward(
    self,
    input_ids=None,
    pixel_values=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    vision_feature_select_strategy=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    logits_to_keep=0,
    **kwargs,
):
    from transformers.models.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time.")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        ).last_hidden_state
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected = self.multi_modal_projector(vision_flat).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=projected)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected)

    causal_lm = self.language_model
    outputs = causal_lm.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    if shift_labels is None:
        shift_labels = _make_liger_shift_labels(labels, attention_mask, slice_indices=slice_indices)
    else:
        shift_labels = _coerce_existing_shift_labels(
            shift_labels,
            slice_indices=slice_indices,
            full_seq_len=hidden_states.shape[1],
            kept_seq_len=kept_hidden_states.shape[1],
        )

    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None
    skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(causal_lm.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = causal_lm.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = causal_lm.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


# ---------------------------------------------------------------------------
# Installers.
# ---------------------------------------------------------------------------
def install_asym_liger_llama4_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)

    if _is_llama4_conditional_generation(root):
        causal_lm = root.language_model
        validated = _validate_liger_lm_head(getattr(causal_lm, "lm_head", None), model_label="Llama4", strict=strict)
        if validated is None:
            return False
        lm_head, weight_source = validated
        root.forward = MethodType(asym_llama4_conditional_lce_forward, root)
        _mark_liger_bridge_installed(root, lm_head, weight_source, "llama4", "conditional_generation")
        return True

    target = _base_causal_lm_model(root)
    model_type = getattr(getattr(target, "config", None), "model_type", None)
    if model_type not in {"llama4_text", "llama4"}:
        if strict:
            raise ValueError("Llama4 Liger loss bridge only supports llama4_text or llama4.")
        return False

    validated = _validate_liger_lm_head(getattr(target, "lm_head", None), model_label="Llama4", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated
    target.forward = MethodType(asym_llama4_causal_lm_lce_forward, target)
    _mark_liger_bridge_installed(target, lm_head, weight_source, model_type, "causal_lm")
    return True


# ---------------------------------------------------------------------------
# Qwen3.5-MoE conditional-generation bridge (real model_type="qwen3_5_moe").
# Qwen3.5-35B-A3B loads as Qwen3_5MoeForConditionalGeneration. Liger ships a
# working fused conditional forward (unlike llama4), so this is a near-verbatim
# copy of liger qwen3_5_moe.lce_forward_conditional_generation; the only behavior
# change is the lm_head_weight resolution so an Asym-staged lm_head works.
# ---------------------------------------------------------------------------
def _is_qwen3_5_moe_conditional_generation(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    return (
        getattr(config, "model_type", None) == "qwen3_5_moe"
        and getattr(config, "text_config", None) is not None
        and hasattr(model, "lm_head")
        and hasattr(model, "model")
    )


@can_return_tuple
def asym_qwen3_5_moe_conditional_lce_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    skip_logits=None,
    **kwargs,
):
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    loss = None
    logits = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits and labels is None and shift_labels is None:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)  # <-- only change vs Liger
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func as _qwen35_lb_loss

        aux_loss = _qwen35_lb_loss(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        if loss is not None and aux_loss is not None:
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    return LigerQwen3_5MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def install_asym_liger_qwen3_5_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)

    if _is_qwen3_5_moe_conditional_generation(root):
        validated = _validate_liger_lm_head(getattr(root, "lm_head", None), model_label="Qwen3.5-MoE", strict=strict)
        if validated is None:
            return False
        lm_head, weight_source = validated
        root.forward = MethodType(asym_qwen3_5_moe_conditional_lce_forward, root)
        _mark_liger_bridge_installed(root, lm_head, weight_source, "qwen3_5_moe", "conditional_generation")
        return True

    target = _base_causal_lm_model(root)
    model_type = getattr(getattr(target, "config", None), "model_type", None)
    if model_type not in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        if strict:
            raise ValueError("Qwen3.5 Liger loss bridge only supports qwen3_5_moe / qwen3_5_moe_text.")
        return False
    validated = _validate_liger_lm_head(getattr(target, "lm_head", None), model_label="Qwen3.5-MoE", strict=strict)
    if validated is None:
        return False
    lm_head, weight_source = validated
    target.forward = MethodType(asym_qwen3_moe_lce_forward, target)  # text-only fallback; == qwen3_moe shape
    _mark_liger_bridge_installed(target, lm_head, weight_source, model_type, "causal_lm")
    return True


# ---------------------------------------------------------------------------
# Qwen3.5 dense conditional bridge (real model_type="qwen3_5").
# Mirrors Liger's qwen3_5.lce_forward_for_multimodal; the only behavior change
# is lm_head_weight resolution so an Asym-staged / CPU-resident lm_head works.
# ---------------------------------------------------------------------------
def _is_qwen3_5_conditional_generation(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    return (
        getattr(config, "model_type", None) == "qwen3_5"
        and getattr(config, "text_config", None) is not None
        and hasattr(model, "lm_head")
        and hasattr(model, "model")
    )


def asym_qwen3_5_conditional_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs: Any,
):
    return_dict = kwargs.pop("return_dict", None)
    if return_dict is None:
        return_dict = self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        mm_token_type_ids=mm_token_type_ids,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.text_config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    if LigerQwen3_5CausalLMOutputWithPast is None:
        raise RuntimeError("LigerQwen3_5CausalLMOutputWithPast is unavailable in this Liger checkout.")
    return LigerQwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def install_asym_liger_qwen3_5_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)

    if _is_qwen3_5_conditional_generation(root):
        validated = _validate_liger_lm_head(getattr(root, "lm_head", None), model_label="Qwen3.5", strict=strict)
        if validated is None:
            return False
        lm_head, weight_source = validated
        root.forward = MethodType(asym_qwen3_5_conditional_lce_forward, root)
        _mark_liger_bridge_installed(root, lm_head, weight_source, "qwen3_5", "conditional_generation")
        return True

    target = _base_causal_lm_model(root)
    model_type = getattr(getattr(target, "config", None), "model_type", None)
    if model_type not in {"qwen3_5_text"}:
        if strict:
            raise ValueError("Qwen3.5 Liger loss bridge only supports qwen3_5 / qwen3_5_text.")
        return False
    return install_asym_liger_dense_loss_bridge(target, strict=strict)


# ---------------------------------------------------------------------------
# Dense causal-LM bridge (qwen2 / llama / qwen3 dense, router-free).
# Qwen2.5, Llama-3.1/3.3, Qwen3-dense, and Qwen3.5 text share one `...ForCausalLM.forward`
# shape, so a single router-free forward + installer covers all three.
# ---------------------------------------------------------------------------
_ASYM_LIGER_DENSE_MODEL_TYPES = {"qwen2", "llama", "qwen3", "qwen3_5_text"}

# model_integration.md families (2026-07-27): vanilla `*ForCausalLM` MoE archs
# whose loss path is the standard hidden→lm_head→CE. Their vocabs make full
# logits the peak-HBM driver (phi T3 128k·b3: loss saved 45.9 GiB + CE
# workspace 33.7 GiB of a 158.6 peak), so the fused-LCE bridge is the memory
# fix, not an optimization. Aux/router loss is NOT computed here — tf-5.6
# drops router-logit threading for these types and the wave-1 parity A/Bs
# confirmed reference and asym losses agree without it.
_ASYM_LIGER_GENERIC_MOE_MODEL_TYPES = {
    "phimoe",
    "mixtral",
    "hunyuan_v1_moe",
    "glm4_moe",
    "glm4_moe_lite",
    "gpt_oss",
    # Jamba2-Mini (model_integration.md #7) is deliberately ABSENT: the
    # instance bridge's FLCE hits a Triton IMA in liger_cross_entropy_kernel
    # under every asym tier on Jamba (jgate_t3*/jab_t1f, 2026-08-09), while
    # the CLASS-level vendored applier (apply_liger_kernel_to_jamba — the
    # DS-safe mechanism, LF resolver) runs the same fused math cleanly on
    # BOTH asym and baseline backends (T3 20.5 GiB vs uns-off 24.7 @32k·b1,
    # loss parity). Jamba therefore rides the class patch only.
}


def _resolve_liger_lm_head_bias(lm_head: nn.Module, reference: torch.Tensor) -> torch.Tensor | None:
    # Phi-3.5-MoE is the one family with a real lm_head bias; an Asym-staged
    # lm_head keeps it host-side as bias_cpu. Tiny (vocab-length), fetched per
    # call. liger's functional supports bias natively.
    bias = getattr(lm_head, "bias", None)
    if bias is None:
        bias = getattr(lm_head, "bias_cpu", None)
    if bias is None:
        return None
    with torch.no_grad():
        return bias.detach().to(device=reference.device, dtype=reference.dtype, non_blocking=True)


def asym_dense_causal_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs: Any,
) -> LigerCausalLMOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states)
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def asym_generic_moe_causal_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs: Any,
) -> LigerCausalLMOutputWithPast:
    # Generic-MoE twin of asym_dense_causal_lce_forward: duck-typed hidden-state
    # access (Moe*OutputWithPast vs BaseModelOutputWithPast), lm_head bias
    # threaded into the fused CE (phimoe), router/aux outputs not requested.
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        lm_head_weight = _resolve_liger_lm_head_weight(self.lm_head, kept_hidden_states, allow_bias=True)
        lm_head_bias = _resolve_liger_lm_head_bias(self.lm_head, kept_hidden_states)
        if lm_head_bias is not None:
            kwargs["bias"] = lm_head_bias
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=lm_head_weight,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = output + (predicted_tokens,) if predicted_tokens is not None else output
        return output

    return LigerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def install_asym_liger_generic_moe_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in _ASYM_LIGER_GENERIC_MOE_MODEL_TYPES:
        if strict:
            raise ValueError(
                f"Asym generic-MoE Liger loss bridge does not support model_type={model_type!r}."
            )
        return False

    validated = _validate_liger_lm_head(
        getattr(target_model, "lm_head", None),
        model_label=str(model_type),
        strict=strict,
        allow_bias=True,
    )
    if validated is None:
        return False
    lm_head, weight_source = validated

    target_model.forward = MethodType(asym_generic_moe_causal_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, str(model_type), "causal_lm")
    return True


def install_asym_liger_dense_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    target_model = _base_causal_lm_model(model)
    config = getattr(target_model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in _ASYM_LIGER_DENSE_MODEL_TYPES:
        if strict:
            raise ValueError(
                f"Asym Liger dense loss bridge only supports {sorted(_ASYM_LIGER_DENSE_MODEL_TYPES)}."
            )
        return False

    validated = _validate_liger_lm_head(
        getattr(target_model, "lm_head", None), model_label=f"{model_type} (dense)", strict=strict
    )
    if validated is None:
        return False
    lm_head, weight_source = validated

    target_model.forward = MethodType(asym_dense_causal_lce_forward, target_model)
    _mark_liger_bridge_installed(target_model, lm_head, weight_source, model_type, "causal_lm")
    return True


def install_asym_liger_loss_bridge(model: nn.Module, *, strict: bool = True) -> bool:
    root = _root_model(model)
    root_type = getattr(getattr(root, "config", None), "model_type", None)
    causal = _base_causal_lm_model(root)
    causal_type = getattr(getattr(causal, "config", None), "model_type", None)

    if root_type == "llama4" or causal_type in {"llama4", "llama4_text"}:
        return install_asym_liger_llama4_loss_bridge(model, strict=strict)
    if root_type == "qwen3_5" or causal_type in {"qwen3_5", "qwen3_5_text"}:
        return install_asym_liger_qwen3_5_loss_bridge(model, strict=strict)
    if root_type == "qwen3_5_moe" or causal_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return install_asym_liger_qwen3_5_moe_loss_bridge(model, strict=strict)
    if causal_type == "qwen3_moe":
        return install_asym_liger_qwen3_moe_loss_bridge(model, strict=strict)
    if causal_type in _ASYM_LIGER_DENSE_MODEL_TYPES:
        return install_asym_liger_dense_loss_bridge(model, strict=strict)
    if causal_type in _ASYM_LIGER_GENERIC_MOE_MODEL_TYPES:
        return install_asym_liger_generic_moe_loss_bridge(model, strict=strict)
    return False


__all__ = [
    "asym_liger_lm_head_bridge_metadata",
    "asym_qwen3_moe_lce_forward",
    "asym_qwen3_5_conditional_lce_forward",
    "asym_qwen3_5_moe_conditional_lce_forward",
    "asym_llama4_causal_lm_lce_forward",
    "asym_llama4_conditional_lce_forward",
    "asym_dense_causal_lce_forward",
    "asym_generic_moe_causal_lce_forward",
    "install_asym_liger_qwen3_moe_loss_bridge",
    "install_asym_liger_qwen3_5_loss_bridge",
    "install_asym_liger_qwen3_5_moe_loss_bridge",
    "install_asym_liger_llama4_loss_bridge",
    "install_asym_liger_dense_loss_bridge",
    "install_asym_liger_generic_moe_loss_bridge",
    "install_asym_liger_loss_bridge",
]
