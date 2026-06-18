from __future__ import annotations

from types import MethodType
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast


def _base_causal_lm_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
        except Exception:
            base_model = None
        if isinstance(base_model, nn.Module) and hasattr(base_model, "lm_head"):
            return base_model
    return model


def _resolve_liger_lm_head_weight(lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    resolver = getattr(lm_head, "asym_liger_lm_head_weight", None)
    if callable(resolver):
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

    lm_head = getattr(target_model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Qwen3-MoE model has no lm_head.")
    if not isinstance(lm_head, nn.Module):
        raise RuntimeError(f"Qwen3-MoE lm_head is not a torch module: {type(lm_head).__name__}.")

    weight_source = _lm_head_weight_source(lm_head)
    if weight_source == "unavailable":
        if strict:
            raise RuntimeError("lm_head is not compatible with Liger fused CE weight resolution.")
        return False
    if getattr(lm_head, "bias", None) is not None or getattr(lm_head, "bias_cpu", None) is not None:
        raise RuntimeError("Asym Liger loss bridge currently requires a bias-free lm_head.")

    if any(param.requires_grad for param in lm_head.parameters(recurse=True)):
        raise RuntimeError("Asym Liger loss bridge supports frozen lm_head only.")

    target_model.forward = MethodType(asym_qwen3_moe_lce_forward, target_model)
    target_model._asym_liger_lm_head_bridge_enabled = True
    target_model._asym_liger_lm_head_weight_source = weight_source
    target_model._asym_liger_lm_head_type = type(lm_head).__name__
    target_model._asym_liger_lm_head_staged_bytes = int(getattr(lm_head, "cpu_resident_base_weight_bytes", 0) or 0)
    return True


def asym_liger_lm_head_bridge_metadata(model: nn.Module) -> dict[str, Any]:
    target_model = _base_causal_lm_model(model)
    enabled = bool(getattr(target_model, "_asym_liger_lm_head_bridge_enabled", False))
    lm_head = getattr(target_model, "lm_head", None)
    if not enabled:
        return {
            "enabled": False,
            "weight_source": "disabled",
            "lm_head_type": type(lm_head).__name__ if lm_head is not None else None,
        }
    return {
        "enabled": True,
        "weight_source": str(getattr(target_model, "_asym_liger_lm_head_weight_source", "unknown")),
        "staged_bytes": int(getattr(target_model, "_asym_liger_lm_head_staged_bytes", 0) or 0),
        "lm_head_type": str(getattr(target_model, "_asym_liger_lm_head_type", type(lm_head).__name__)),
    }


__all__ = [
    "asym_liger_lm_head_bridge_metadata",
    "asym_qwen3_moe_lce_forward",
    "install_asym_liger_qwen3_moe_loss_bridge",
]
