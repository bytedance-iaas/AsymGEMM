from __future__ import annotations

import types
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DecoderCheckpointWrapper:
    """Forward wrapper that checkpoints an existing decoder layer in place."""

    def __init__(
        self,
        module: nn.Module,
        *,
        use_reentrant: bool = True,
        preserve_rng_state: bool = True,
    ) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.use_reentrant = bool(use_reentrant)
        self.preserve_rng_state = bool(preserve_rng_state)
        self.calls = 0
        self.checkpoint_calls = 0
        self.skipped_cache_calls = 0

    def install(self) -> None:
        setattr(self.module, "_asym_decoder_checkpoint_wrapper", self)
        self.module.forward = types.MethodType(_decoder_checkpoint_forward, self.module)  # type: ignore[method-assign]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if bool(kwargs.get("use_cache", False)):
            self.skipped_cache_calls += 1
            return self.original_forward(*args, **kwargs)

        has_grad = any(param.requires_grad for param in self.module.parameters())
        if not has_grad:
            return self.original_forward(*args, **kwargs)
        for arg in args:
            if torch.is_tensor(arg) and torch.is_floating_point(arg):
                arg.requires_grad_(True)
                break

        if self.use_reentrant:
            body = partial(self.original_forward, **kwargs)
            checkpoint_kwargs: dict[str, Any] = {}
        else:
            def body(*inner_args: Any, **inner_kwargs: Any) -> Any:
                return self.original_forward(*inner_args, **inner_kwargs)

            checkpoint_kwargs = kwargs

        self.checkpoint_calls += 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True.*")
            return checkpoint(
                body,
                *args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng_state,
                **checkpoint_kwargs,
            )


def _decoder_checkpoint_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_decoder_checkpoint_wrapper", None)
    if not isinstance(wrapper, DecoderCheckpointWrapper):
        raise RuntimeError("decoder checkpoint wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_decoder_checkpoint(module: nn.Module) -> DecoderCheckpointWrapper:
    existing = getattr(module, "_asym_decoder_checkpoint_wrapper", None)
    if isinstance(existing, DecoderCheckpointWrapper):
        return existing
    wrapper = DecoderCheckpointWrapper(module)
    wrapper.install()
    return wrapper


def is_decoder_checkpoint_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_decoder_checkpoint_wrapper", None), DecoderCheckpointWrapper)


def decoder_checkpoint_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, module in model.named_modules() if name and is_decoder_checkpoint_wrapper(module))


__all__ = [
    "DecoderCheckpointWrapper",
    "decoder_checkpoint_module_names",
    "install_decoder_checkpoint",
    "is_decoder_checkpoint_wrapper",
]
