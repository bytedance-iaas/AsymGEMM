"""Token-chunked dense-MLP forward.

The dense SwiGLU MLP (`down(silu(gate(x)) * up(x))`) is token-parallel, so its forward can be evaluated in
row-chunks. Under whole-layer gradient checkpointing (HF/Unsloth), the layer is recomputed in backward; if the
MLP recompute materializes the full `[M, I]` working set at once it blows up the transient HBM peak (the dense MLP
intermediate is `[M, intermediate_size]`, e.g. 360000x25600 ~= 18 GiB per tensor). Wrapping each chunk in a
checkpoint caps the recompute working set at `[chunk, I]` while keeping the result numerically identical.

This is intentionally a thin wrapper around the existing module forward (Qwen3/Llama-style MLP, whose projections
may already be AsymGEMM/LoRA leaves) so it composes with weight offload and the asym base GEMMs unchanged. It only
chunks when training, grad is enabled, the input requires grad, and the row count exceeds the chunk size; otherwise
it is the original forward verbatim.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


class ChunkedMLPWrapper:
    """Wrap a dense MLP forward so it runs in token row-chunks, each checkpointed."""

    def __init__(self, module: nn.Module, *, chunk_tokens: int, preserve_rng_state: bool = True) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.chunk_tokens = int(chunk_tokens)
        self.preserve_rng_state = bool(preserve_rng_state)
        self.calls = 0
        self.chunked_calls = 0

    def install(self) -> None:
        setattr(self.module, "_asym_chunked_mlp_wrapper", self)
        self.module.forward = types.MethodType(_chunked_mlp_forward, self.module)  # type: ignore[method-assign]

    def run(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if args or kwargs:
            # Only the single-tensor MLP forward is chunked; anything else falls back unchanged.
            return self.original_forward(hidden_states, *args, **kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            return self.original_forward(hidden_states)

        out_shape = tuple(hidden_states.shape)
        flat = hidden_states.reshape(-1, out_shape[-1])
        rows = int(flat.shape[0])
        chunk = int(self.chunk_tokens)
        if chunk <= 0 or rows <= chunk:
            return self.original_forward(hidden_states)

        # Grad path (e.g. the recompute pass under whole-layer GC, where the input requires grad): checkpoint each
        # chunk so its [chunk, I] activations are recomputed in backward. No-grad path (the Unsloth/HF no_grad
        # forward sweep): just loop the chunks so the [M, I] forward transient is capped to [chunk, I] too.
        grad_mode = bool(self.module.training and torch.is_grad_enabled() and flat.requires_grad)

        def _fwd(h: torch.Tensor) -> torch.Tensor:
            out = self.original_forward(h)
            out = out[0] if isinstance(out, tuple) else out
            return out.reshape(-1, out.shape[-1])

        self.chunked_calls += 1
        if grad_mode:
            pieces = [
                checkpoint(
                    _fwd, flat[start : start + chunk], use_reentrant=False, preserve_rng_state=self.preserve_rng_state
                )
                for start in range(0, rows, chunk)
            ]
        else:
            pieces = [_fwd(flat[start : start + chunk]) for start in range(0, rows, chunk)]
        out = torch.cat(pieces, dim=0)
        return out.reshape(*out_shape[:-1], out.shape[-1])


def _chunked_mlp_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_chunked_mlp_wrapper", None)
    if not isinstance(wrapper, ChunkedMLPWrapper):
        raise RuntimeError("chunked MLP wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_chunked_mlp(module: nn.Module, *, chunk_tokens: int) -> ChunkedMLPWrapper:
    existing = getattr(module, "_asym_chunked_mlp_wrapper", None)
    if isinstance(existing, ChunkedMLPWrapper):
        return existing
    wrapper = ChunkedMLPWrapper(module, chunk_tokens=chunk_tokens)
    wrapper.install()
    return wrapper


def is_chunked_mlp_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_chunked_mlp_wrapper", None), ChunkedMLPWrapper)


def _is_dense_mlp_module(name: str, module: nn.Module) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf not in {"mlp", "feed_forward"}:
        return False
    if getattr(module, "_is_asym_dense_mlp", False):
        return True
    children = set(dict(module.named_children()))
    return {"gate_proj", "up_proj", "down_proj"} <= children


def install_chunked_mlp_on_dense_mlps(model: nn.Module, *, chunk_tokens: int | None = None) -> tuple[str, ...]:
    """Install the chunked MLP wrapper on every dense MLP module. No-op when chunk size <= 0."""
    tokens = _env_int("ASYMM_MLP_RECOMPUTE_CHUNK", 0) if chunk_tokens is None else int(chunk_tokens)
    if tokens <= 0:
        return ()
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if name and _is_dense_mlp_module(name, module):
            install_chunked_mlp(module, chunk_tokens=tokens)
            wrapped.append(name)
    if wrapped:
        print(f"[asym] chunked dense MLP installed on {len(wrapped)} modules (chunk={tokens} tokens)", flush=True)
    return tuple(wrapped)


__all__ = [
    "ChunkedMLPWrapper",
    "install_chunked_mlp",
    "install_chunked_mlp_on_dense_mlps",
    "is_chunked_mlp_wrapper",
]
