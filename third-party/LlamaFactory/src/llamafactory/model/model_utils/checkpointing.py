# Copyright 2025 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from collections.abc import Callable
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default

    try:
        return int(value)
    except ValueError:
        return default


_UNSLOTH_GC_OUTER_SAVE_COUNTER = 0
_GC_BACKWARD_LAYER_COUNTER = 0


def _maybe_release_cached_blocks_per_gc_layer() -> None:
    # Under expandable segments, peak RESERVED is the
    # union of per-stream pool high-waters across backward sub-phases (per-layer
    # chunk recompute / expert / attention transients). Releasing free pages every
    # N checkpointed-layer backwards lets pools re-grow from ~allocated instead.
    # Default OFF; backend-agnostic (every backend runs this GC path).
    every_n = _env_int("ASYM_EMPTY_CACHE_EVERY_GC_LAYERS", 0)
    if every_n <= 0:
        return
    global _GC_BACKWARD_LAYER_COUNTER
    _GC_BACKWARD_LAYER_COUNTER += 1
    if _GC_BACKWARD_LAYER_COUNTER % every_n == 0 and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _should_keep_unsloth_outer_hidden_on_hbm() -> bool:
    # Diagnostic only. Keep UNSLOTH_GC_OUTER_HBM_EVERY_N=0 for apples-to-apples
    # comparisons; nonzero values trade CPU checkpoint roots for extra HBM.
    every_n = _env_int("UNSLOTH_GC_OUTER_HBM_EVERY_N", 0)
    if every_n <= 0:
        return False

    global _UNSLOTH_GC_OUTER_SAVE_COUNTER
    keep_on_hbm = (_UNSLOTH_GC_OUTER_SAVE_COUNTER % every_n) == 0
    _UNSLOTH_GC_OUTER_SAVE_COUNTER += 1
    return keep_on_hbm


def get_unsloth_gradient_checkpointing_func() -> Callable:
    if _env_flag("ASYM_UNSLOTH_GC_NVME"):        # exported by run_lf when ASYM_NVME_ROLES has "activation"
        from asym_gemm.training.gc_boundary_offload import get_asym_unsloth_gc_func

        return get_asym_unsloth_gc_func()

    try:
        # Phase D4 (asym fix_throughput C3): pinned root + side-stream restage and an
        # async save_on_cpu. Gated by ASYM_SAVED_TENSOR_ASYNC_UNPACK (default off) —
        # the stock path below saves the root to PAGEABLE cpu memory, making the
        # per-layer backward fetch a host-blocking pageable H2D.
        from asym_gemm.training import gc_async_offload as _asym_gc_async
    except Exception:  # pragma: no cover - asym_gemm absent
        _asym_gc_async = None

    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""Saves VRAM by smartly offloading to RAM."""

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            async_gc = _asym_gc_async is not None and _asym_gc_async.async_gc_offload_enabled()
            ctx.asym_root_ready = None
            ctx.asym_root_device = None
            if hidden_states.is_cuda and _should_keep_unsloth_outer_hidden_on_hbm():
                saved_hidden_states = hidden_states
            elif async_gc and hidden_states.is_cuda:
                saved_hidden_states, ready = _asym_gc_async.pack_root_to_pinned_cpu(hidden_states)
                ctx.asym_root_ready = ready
                ctx.asym_root_device = hidden_states.device
            else:
                saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)

            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors
            async_gc = _asym_gc_async is not None and _asym_gc_async.async_gc_offload_enabled()
            if getattr(ctx, "asym_root_device", None) is not None and not hidden_states.is_cuda:
                hidden_states = _asym_gc_async.fetch_root_async(
                    hidden_states, ctx.asym_root_ready, ctx.asym_root_device
                ).detach()
            else:
                hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)
            with torch.enable_grad():
                if _env_flag("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU"):
                    if async_gc:
                        save_ctx = _asym_gc_async.async_save_on_cpu()
                    else:
                        save_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
                    with save_ctx:
                        outputs = ctx.forward_function(hidden_states, *ctx.args)
                else:
                    outputs = ctx.forward_function(hidden_states, *ctx.args)
                output = outputs[0] if isinstance(outputs, tuple) else outputs

            torch.autograd.backward(output, grad_output)
            _maybe_release_cached_blocks_per_gc_layer()
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""Only applies gradient checkpointing to trainable layers."""

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    break  # assume the first tensor is always the hidden states

        if has_grad:
            return gradient_checkpointing_func(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning_rank0_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Prepare the model before training.

    Include:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) add the upcasting of the lm_head in fp32.
    """
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if (
        os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        and int(os.environ.get("FSDP_VERSION", "1")) == 2
    ):
        model_args.use_reentrant_gc = False
        logger.warning_rank0("You are using fsdp2, `use_reentrant_gc` has been set to False.")

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}
            )
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info_rank0("Gradient checkpointing enabled.")

    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")
            output_layer.register_forward_hook(_fp32_forward_post_hook)
