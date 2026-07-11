# Copyright 2025 the LlamaFactory team.
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
from typing import TYPE_CHECKING
from typing import Callable

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

_LOSS_ONLY_SUPPORTED_MODEL_TYPES = {
    "qwen3_moe",
    "llama4_text",
    "llama4",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen2",
    "llama",
    "qwen3",
}


def _resolve_liger_apply_fn(model_type: str | None) -> Callable[..., None] | None:
    if model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

        return apply_liger_kernel_to_qwen3_moe

    if model_type in {"llama4_text", "llama4"}:
        from liger_kernel.transformers import apply_liger_kernel_to_llama4

        return apply_liger_kernel_to_llama4

    if model_type in {"qwen3_5", "qwen3_5_text"}:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        return apply_liger_kernel_to_qwen3_5

    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe

        return apply_liger_kernel_to_qwen3_5_moe

    if model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        return apply_liger_kernel_to_qwen2

    if model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama

        return apply_liger_kernel_to_llama

    if model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        return apply_liger_kernel_to_qwen3

    return None


def _build_liger_loss_only_kwargs(apply_fn: Callable[..., None]) -> dict[str, bool] | None:
    signature = inspect.signature(apply_fn)
    if "fused_linear_cross_entropy" not in signature.parameters:
        return None

    kwargs: dict[str, bool] = {}
    for name, param in signature.parameters.items():
        if name == "model":
            continue
        if name == "fused_linear_cross_entropy":
            kwargs[name] = True
        elif isinstance(param.default, bool):
            kwargs[name] = False

    if kwargs.get("fused_linear_cross_entropy") is not True:
        return None
    return kwargs


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    if model_type not in _LOSS_ONLY_SUPPORTED_MODEL_TYPES:
        logger.warning_rank0(f"Skipping Liger loss-only for unvalidated model_type={model_type}.")
        return

    if require_logits:
        logger.warning_rank0("Skipping Liger loss-only because logits are required.")
        return

    apply_fn = _resolve_liger_apply_fn(model_type)
    if apply_fn is None:
        logger.warning_rank0(f"Skipping Liger loss-only for model_type={model_type}; apply function is unavailable.")
        return

    kwargs = _build_liger_loss_only_kwargs(apply_fn)
    if kwargs is None:
        logger.warning_rank0(
            f"Skipping Liger loss-only for model_type={model_type}; fused linear CE is unavailable."
        )
        return

    apply_fn(**kwargs)
    logger.info_rank0("Liger loss-only kernel has been applied.")
