# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer, _LoraParameterProxy
from peft.tuners.tuners_utils import check_adapters_to_merge

from ...extras import logging


if TYPE_CHECKING:
    from peft import LoraConfig, PeftConfig


logger = logging.get_logger(__name__)

QWEN_EXPERT_LORA_PEFT_TARGET_PARAMETERS = "peft-target-parameters"
QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS = "split-target-parameters"
QWEN_EXPERT_LORA_OFF = "off"
QWEN_EXPERT_LORA_AUTO = "auto"
QWEN_EXPERT_LORA_MODES = {
    QWEN_EXPERT_LORA_PEFT_TARGET_PARAMETERS,
    QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS,
    QWEN_EXPERT_LORA_OFF,
}


def _tensor_logical_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in tensor.shape)
    if tensor.numel() == 0 and hasattr(tensor, "ds_shape"):
        try:
            ds_shape = tuple(int(dim) for dim in tensor.ds_shape)
        except (TypeError, ValueError):
            ds_shape = ()
        if ds_shape:
            return ds_shape
    return shape


def _fused_experts_layout(module: nn.Module) -> tuple[bool, int, int, int] | None:
    """Detect a fused-expert MoE module and return ``(transposed, experts, hidden, intermediate)``.

    Standard layout (Qwen3 / Qwen3.5): ``gate_up_proj [E, 2*intermediate, hidden]``,
    ``down_proj [E, hidden, intermediate]`` (nn.Linear ``out, in`` convention).
    Transposed layout (Llama 4): ``gate_up_proj [E, hidden, 2*intermediate]``,
    ``down_proj [E, intermediate, hidden]`` (bmm ``in, out`` convention).
    Returns ``None`` when the module is not a recognized fused-expert module.
    """
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        return None
    gate_up_shape = _tensor_logical_shape(gate_up)
    down_shape = _tensor_logical_shape(down)
    if (
        len(gate_up_shape) != 3
        or len(down_shape) != 3
        or gate_up_shape[0] != down_shape[0]
        or not hasattr(module, "act_fn")
        or not hasattr(module, "num_experts")
    ):
        return None
    experts = int(gate_up_shape[0])
    # Standard (Qwen): gate_up [E, 2I, H], down [E, H, I]
    if gate_up_shape[1] == down_shape[2] * 2 and gate_up_shape[2] == down_shape[1]:
        return (False, experts, int(down_shape[1]), int(down_shape[2]))
    # Transposed (Llama 4): gate_up [E, H, 2I], down [E, I, H]
    if gate_up_shape[2] == down_shape[1] * 2 and gate_up_shape[1] == down_shape[2]:
        return (True, experts, int(down_shape[2]), int(down_shape[1]))
    return None


def _is_qwen_fused_experts_module(module: nn.Module) -> bool:
    return _fused_experts_layout(module) is not None


def _qwen_expert_dims(module: nn.Module) -> tuple[int, int, int]:
    layout = _fused_experts_layout(module)
    if layout is None:
        raise ValueError("module is not a recognized fused-expert MoE module")
    _transposed, experts, hidden, intermediate = layout
    return experts, hidden, intermediate


def _patch_peft_param_wrapper_zero3_shape() -> None:
    from peft.tuners.lora.layer import ParamWrapper

    if getattr(ParamWrapper.__init__, "_qwen_zero3_shape_patch", False):
        return

    original_init = ParamWrapper.__init__

    def patched_init(
        self,
        base_layer,
        adapter_name: str,
        parameter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        shape_base_layer = base_layer.get_base_layer() if hasattr(base_layer, "get_base_layer") else base_layer
        param = getattr(shape_base_layer, parameter_name, None)
        shape = _tensor_logical_shape(param) if isinstance(param, torch.Tensor) else ()
        if not isinstance(param, torch.Tensor) or param.numel() != 0 or len(shape) not in (2, 3):
            original_init(
                self,
                base_layer,
                adapter_name,
                parameter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=fan_in_fan_out,
                is_target_conv_1d_layer=is_target_conv_1d_layer,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                lora_bias=lora_bias,
                **kwargs,
            )
            return

        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.parameter_name = parameter_name
        if len(shape) == 3:
            self.num_experts, self.in_features, self.out_features = shape
        else:
            self.num_experts, self.in_features, self.out_features = 1, shape[1], shape[0]

        if lora_dropout:
            raise ValueError(f"lora.{self.__class__.__name__} does not work with lora_dropout != 0.")
        if fan_in_fan_out:
            raise ValueError(f"lora.{self.__class__.__name__} does not work with fan_in_fan_out.")
        if lora_bias:
            raise ValueError(f"lora.{self.__class__.__name__} does not work with lora_bias=True.")
        if use_dora:
            raise ValueError(f"lora.{self.__class__.__name__} does not work with use_dora=True.")
        if is_target_conv_1d_layer:
            raise ValueError(f"lora.{self.__class__.__name__} does not work with is_target_conv_1d_layer=True.")

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    patched_init._qwen_zero3_shape_patch = True  # type: ignore[attr-defined]
    ParamWrapper.__init__ = patched_init



def _normalize_targets(targets: Any) -> set[str]:
    if targets is None:
        return set()
    if isinstance(targets, str):
        return {targets.lower()}
    try:
        return {str(target).lower() for target in targets}
    except TypeError:
        return {str(targets).lower()}


def _target_selection_reaches_qwen_experts(
    raw_lora_target: Any,
    resolved_target_modules: Any,
    expert_module_name: str,
) -> bool:
    selectors = _normalize_targets(raw_lora_target) | _normalize_targets(resolved_target_modules)
    if not selectors:
        return False

    name = expert_module_name.lower()
    suffix = name.rsplit(".", 1)[-1]

    if selectors & {"all", "all-linear", "all_linear"}:
        return True
    if "experts" in selectors:
        return True
    if name in selectors or suffix in selectors:
        return True
    if any(name.endswith(f".{target}") for target in selectors):
        return True
    return any(
        target.endswith(".mlp.experts")
        or target == "mlp.experts"
        or target.endswith(".feed_forward.experts")
        or target == "feed_forward.experts"
        for target in selectors
    )


def _selected_qwen_expert_module_names(
    model: nn.Module,
    raw_lora_target: Any,
    resolved_target_modules: Any,
) -> list[str]:
    selected: list[str] = []
    for name, module in model.named_modules():
        if not _is_qwen_fused_experts_module(module):
            continue
        if _target_selection_reaches_qwen_experts(raw_lora_target, resolved_target_modules, name):
            selected.append(name)
    return selected


class QwenSplitMoeExpertParamWrapper(nn.Module, LoraLayer):
    adapter_layer_names = (
        "lora_gate_A",
        "lora_gate_B",
        "lora_up_A",
        "lora_up_B",
        "lora_down_A",
        "lora_down_B",
    )
    parameter_names = ("gate_up_proj", "down_proj")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: Any | None = None,
        r: int = 0,
        lora_alpha: int | float = 1,
        lora_dropout: float | None = None,
        init_lora_weights: bool | str | None = None,
        use_rslora: bool | None = None,
        use_dora: bool | None = None,
        lora_bias: bool | None = None,
        fan_in_fan_out: bool | None = None,
        is_target_conv_1d_layer: bool = False,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            lora_dropout = getattr(config, "lora_dropout", 0.0) if lora_dropout is None else lora_dropout
            init_lora_weights = (
                getattr(config, "init_lora_weights", True)
                if init_lora_weights is None
                else init_lora_weights
            )
            use_rslora = getattr(config, "use_rslora", False) if use_rslora is None else use_rslora
            use_dora = getattr(config, "use_dora", False) if use_dora is None else use_dora
            lora_bias = getattr(config, "lora_bias", False) if lora_bias is None else lora_bias
            fan_in_fan_out = getattr(config, "fan_in_fan_out", False) if fan_in_fan_out is None else fan_in_fan_out
        else:
            lora_dropout = 0.0 if lora_dropout is None else lora_dropout
            init_lora_weights = True if init_lora_weights is None else init_lora_weights
            use_rslora = False if use_rslora is None else use_rslora
            use_dora = False if use_dora is None else use_dora
            lora_bias = False if lora_bias is None else lora_bias
            fan_in_fan_out = False if fan_in_fan_out is None else fan_in_fan_out

        if float(lora_dropout or 0.0) != 0.0:
            raise ValueError("QwenSplitMoeExpertParamWrapper requires lora_dropout=0.0.")
        if fan_in_fan_out:
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support fan_in_fan_out.")
        if lora_bias:
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support LoRA bias.")
        if use_dora:
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support DoRA.")
        if is_target_conv_1d_layer:
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support Conv1D target layers.")
        layout = _fused_experts_layout(base_layer)
        if layout is None:
            raise TypeError("QwenSplitMoeExpertParamWrapper expects a fused MoE experts module.")

        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        self._transposed, self.experts, self.hidden_size, self.intermediate_size = layout
        self._active_adapter = adapter_name
        self.fan_in_fan_out = False
        self.use_rslora: dict[str, bool] = {}

        self.lora_gate_A = nn.ParameterDict()
        self.lora_gate_B = nn.ParameterDict()
        self.lora_up_A = nn.ParameterDict()
        self.lora_up_B = nn.ParameterDict()
        self.lora_down_A = nn.ParameterDict()
        self.lora_down_B = nn.ParameterDict()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=float(lora_dropout or 0.0),
            init_lora_weights=init_lora_weights,
            use_rslora=bool(use_rslora),
            inference_mode=bool(getattr(config, "inference_mode", False)),
        )

    def _get_in_out_features(self, module: nn.Module) -> tuple[int, int] | tuple[None, None]:
        if _is_qwen_fused_experts_module(module):
            _experts, hidden, intermediate = _qwen_expert_dims(module)
            return hidden, 2 * intermediate
        return None, None

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int | float,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str | None = True,
        use_rslora: bool = False,
        inference_mode: bool = False,
        **_: Any,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}.")
        if float(lora_dropout or 0.0) != 0.0:
            raise ValueError("QwenSplitMoeExpertParamWrapper requires lora_dropout=0.0.")

        gate_up = getattr(self.get_base_layer(), "gate_up_proj")
        experts, hidden, intermediate = self.experts, self.hidden_size, self.intermediate_size
        dtype = gate_up.dtype
        device = gate_up.device

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = float(lora_alpha) / math.sqrt(r) if use_rslora else float(lora_alpha) / r
        self.use_rslora[adapter_name] = use_rslora
        self.lora_dropout[adapter_name] = nn.Identity()

        self.lora_gate_A[adapter_name] = nn.Parameter(torch.empty(experts, r, hidden, dtype=dtype, device=device))
        self.lora_gate_B[adapter_name] = nn.Parameter(torch.zeros(experts, intermediate, r, dtype=dtype, device=device))
        self.lora_up_A[adapter_name] = nn.Parameter(torch.empty(experts, r, hidden, dtype=dtype, device=device))
        self.lora_up_B[adapter_name] = nn.Parameter(torch.zeros(experts, intermediate, r, dtype=dtype, device=device))
        self.lora_down_A[adapter_name] = nn.Parameter(torch.empty(experts, r, intermediate, dtype=dtype, device=device))
        self.lora_down_B[adapter_name] = nn.Parameter(torch.zeros(experts, hidden, r, dtype=dtype, device=device))

        if init_lora_weights is True:
            for weight_a in (self.lora_gate_A, self.lora_up_A, self.lora_down_A):
                nn.init.kaiming_uniform_(weight_a[adapter_name].view(experts * r, -1), a=math.sqrt(5))
        elif init_lora_weights in (False, None):
            pass
        else:
            raise ValueError("QwenSplitMoeExpertParamWrapper supports only default LoRA initialization.")

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: torch.device | None = None) -> None:
        base_param = getattr(self.get_base_layer(), "gate_up_proj")
        device = base_param.device if device is None else device
        dtype = base_param.dtype
        meta = torch.device("meta")
        for layer_name in self.adapter_layer_names:
            param_dict = getattr(self, layer_name, None)
            if not isinstance(param_dict, nn.ParameterDict) or adapter_name not in param_dict:
                continue
            param = param_dict[adapter_name]
            if param.device == meta:
                continue
            param_dict[adapter_name] = nn.Parameter(
                param.detach().to(device=device, dtype=dtype),
                requires_grad=param.requires_grad,
            )

    @staticmethod
    def _expert_delta(weight_b: torch.Tensor, weight_a: torch.Tensor, scaling: float) -> torch.Tensor:
        return torch.einsum("eor,eri->eoi", weight_b, weight_a) * scaling

    def get_delta_weight(self, adapter_name: str, parameter_name: str) -> torch.Tensor:
        if adapter_name not in self.lora_gate_A:
            raise ValueError(f"Adapter {adapter_name!r} is not registered on {self.__class__.__name__}.")

        scaling = self.scaling[adapter_name]
        base = self.get_base_layer()
        if parameter_name == "gate_up_proj":
            delta_gate = self._expert_delta(self.lora_gate_B[adapter_name], self.lora_gate_A[adapter_name], scaling)
            delta_up = self._expert_delta(self.lora_up_B[adapter_name], self.lora_up_A[adapter_name], scaling)
            delta = torch.cat((delta_gate, delta_up), dim=1)
            param = getattr(base, parameter_name)
        elif parameter_name == "down_proj":
            delta = self._expert_delta(self.lora_down_B[adapter_name], self.lora_down_A[adapter_name], scaling)
            param = getattr(base, parameter_name)
        else:
            raise ValueError(f"Unknown Qwen expert parameter: {parameter_name}.")

        # Llama 4 stores experts transposed (gate_up [E, H, 2I], down [E, I, H]) relative to Qwen's
        # [E, out, in]. The delta above is built in [E, out, in], so flip the last two dims to match
        # the transposed base parameter before it is added by the parametrization / merge.
        if getattr(self, "_transposed", False):
            delta = delta.transpose(1, 2).contiguous()

        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return delta.to(device=param.device, dtype=param.dtype)
        return delta.to(device=param.device)

    @contextmanager
    def _activate_lora(self, active_adapters: list[str]):
        if not active_adapters or not any(adapter in self.lora_gate_A for adapter in active_adapters):
            yield
            return

        base = self.get_base_layer()
        registered: list[str] = []
        try:
            for parameter_name in self.parameter_names:
                delta_weight = None
                for active_adapter in active_adapters:
                    if active_adapter not in self.lora_gate_A:
                        continue
                    candidate = self.get_delta_weight(active_adapter, parameter_name)
                    delta_weight = candidate if delta_weight is None else delta_weight + candidate
                if delta_weight is None:
                    continue

                requires_grad_before = getattr(base, parameter_name).requires_grad
                nn.utils.parametrize.register_parametrization(
                    base,
                    parameter_name,
                    _LoraParameterProxy(delta_weight),
                )
                base.parametrizations[parameter_name].original.requires_grad_(requires_grad_before)
                registered.append(parameter_name)

            with nn.utils.parametrize.cached():
                yield
        finally:
            for parameter_name in reversed(registered):
                self._remove_parametrization(parameter_name)

    def _remove_parametrization(self, parameter_name: str) -> None:
        base = self.get_base_layer()
        if parameter_name not in base.parametrizations:
            raise ValueError(f"Missing parametrization for Qwen expert parameter {parameter_name}.")

        param_list = base.parametrizations[parameter_name]
        if len(param_list) == 1:
            nn.utils.parametrize.remove_parametrizations(base, parameter_name, leave_parametrized=False)
            return

        for index in reversed(range(len(param_list))):
            if isinstance(param_list[index], _LoraParameterProxy):
                del param_list[index]
                return
        warnings.warn(f"Could not find LoRA parametrization for Qwen expert parameter {parameter_name}.")

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.lora_gate_A:
                continue
            for parameter_name in self.parameter_names:
                param = getattr(base, parameter_name)
                delta = self.get_delta_weight(active_adapter, parameter_name).to(param.dtype)
                if safe_merge:
                    merged = param.data.clone() + delta
                    if not torch.isfinite(merged).all():
                        raise ValueError(f"NaNs detected while merging adapter {active_adapter}.")
                    param.data = merged
                else:
                    param.data += delta
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        base = self.get_base_layer()
        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_gate_A:
                continue
            for parameter_name in self.parameter_names:
                param = getattr(base, parameter_name)
                param.data -= self.get_delta_weight(active_adapter, parameter_name).to(param.dtype)

    def _check_forward_args(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("adapter_names", None):
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support mixed adapter batches.")
        super()._check_forward_args(x, *args, **kwargs)

    def unload_and_optionally_merge_module(
        self,
        merge: bool,
        safe_merge: bool,
        adapter_names: list[str] | None,
    ) -> nn.Module:
        base_layer = self.base_layer
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        return base_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            raise ValueError("QwenSplitMoeExpertParamWrapper does not support mixed batch inference.")
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        with self._activate_lora(self.active_adapters):
            return self.base_layer(x, *args, **kwargs)


def _add_qwen_moe_target_parameters(
    model: nn.Module,
    peft_config: "LoraConfig",
    raw_lora_target: Any,
    resolved_target_modules: Any,
) -> "LoraConfig":
    expert_names = _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules)
    if not expert_names:
        logger.info_rank0("Qwen MoE expert LoRA target-parameters mode selected 0 fused expert modules.")
        return peft_config

    target_parameters = set(getattr(peft_config, "target_parameters", None) or [])
    for name in expert_names:
        target_parameters.add(f"{name}.gate_up_proj")
        target_parameters.add(f"{name}.down_proj")
    peft_config.target_parameters = sorted(target_parameters)
    logger.info_rank0(
        f"Qwen MoE expert LoRA target-parameters mode selected {len(expert_names)} fused expert modules "
        f"and {len(peft_config.target_parameters)} PEFT target parameters."
    )
    return peft_config


def _add_qwen_moe_split_target_parameters(
    model: nn.Module,
    peft_config: "LoraConfig",
    raw_lora_target: Any,
    resolved_target_modules: Any,
) -> "LoraConfig":
    expert_names = _selected_qwen_expert_module_names(model, raw_lora_target, resolved_target_modules)
    if not expert_names:
        logger.info_rank0("Qwen MoE split expert LoRA selected 0 fused expert modules.")
        return peft_config

    custom_types: dict[type[nn.Module], type[nn.Module]] = {}
    expert_name_set = set(expert_names)
    for name, module in model.named_modules():
        if name in expert_name_set:
            custom_types[type(module)] = QwenSplitMoeExpertParamWrapper

    targets = set(getattr(peft_config, "target_modules", None) or [])
    targets.update(expert_names)
    peft_config.target_modules = sorted(targets)
    peft_config._register_custom_module(custom_types)
    logger.info_rank0(
        f"Qwen MoE split expert LoRA selected {len(expert_names)} fused expert modules."
    )
    return peft_config


def prepare_qwen_moe_expert_lora_config(
    model: nn.Module,
    peft_config: "LoraConfig",
    mode: str,
    raw_lora_target: Any,
    resolved_target_modules: Any,
) -> "LoraConfig":
    mode = (mode or QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS).strip().lower()
    if mode == QWEN_EXPERT_LORA_OFF:
        return peft_config
    if mode == QWEN_EXPERT_LORA_PEFT_TARGET_PARAMETERS:
        if float(getattr(peft_config, "lora_dropout", 0.0) or 0.0) != 0.0:
            raise ValueError("peft-target-parameters expert LoRA requires lora_dropout=0.0.")
        _patch_peft_param_wrapper_zero3_shape()
        return _add_qwen_moe_target_parameters(model, peft_config, raw_lora_target, resolved_target_modules)
    if mode == QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS:
        if float(getattr(peft_config, "lora_dropout", 0.0) or 0.0) != 0.0:
            raise ValueError("split-target-parameters expert LoRA requires lora_dropout=0.0.")
        return _add_qwen_moe_split_target_parameters(model, peft_config, raw_lora_target, resolved_target_modules)
    raise ValueError(f"Unknown Qwen MoE expert LoRA implementation: {mode}.")


def infer_qwen_expert_lora_impl(peft_config: "PeftConfig", requested_mode: str = QWEN_EXPERT_LORA_AUTO) -> str:
    requested_mode = (requested_mode or QWEN_EXPERT_LORA_AUTO).strip().lower()
    if requested_mode != QWEN_EXPERT_LORA_AUTO:
        return requested_mode

    target_parameters = set(getattr(peft_config, "target_parameters", None) or [])
    if any(
        target.endswith(".mlp.experts.gate_up_proj")
        or target.endswith(".mlp.experts.down_proj")
        or target in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        for target in target_parameters
    ):
        return QWEN_EXPERT_LORA_PEFT_TARGET_PARAMETERS

    targets = {str(target).lower() for target in (getattr(peft_config, "target_modules", None) or [])}
    if any(
        target.endswith(".mlp.experts")
        or target.endswith(".feed_forward.experts")
        or target in {"experts", "mlp.experts", "feed_forward.experts"}
        for target in targets
    ):
        return QWEN_EXPERT_LORA_SPLIT_TARGET_PARAMETERS
    return QWEN_EXPERT_LORA_OFF


def count_qwen_moe_expert_lora(model: nn.Module) -> dict[str, int]:
    modules = 0
    tensors = 0
    params = 0
    for module in model.modules():
        if not isinstance(module, QwenSplitMoeExpertParamWrapper):
            continue
        modules += 1
        for layer_name in module.adapter_layer_names:
            param_dict = getattr(module, layer_name)
            tensors += len(param_dict)
            params += sum(int(param.numel()) for param in param_dict.values())
    return {"modules": modules, "tensors": tensors, "parameters": params}


def count_fused_moe_lora(model: nn.Module) -> dict[str, int]:
    return count_qwen_moe_expert_lora(model)
