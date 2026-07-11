from __future__ import annotations

import os

import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_INTERFACE_NAME = "asym_sdpa_recompute"
_FALSEY = {"", "0", "false", "no", "off"}


def _enabled() -> bool:
    # accept the direct policy flag OR the harness-forwarded ASYM_GEMM_LF_CONFIG_* mirror
    for name in ("ASYMM_ATTN_SDPA_RECOMPUTE", "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_SDPA_RECOMPUTE"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip().lower() not in _FALSEY:
            return True
    return False


def _make_checkpointed(base_fn):
    def _checkpointed(module, query, key, value, attention_mask=None, **kwargs):
        # eval / no-grad: passthrough, zero overhead
        if not (module.training and torch.is_grad_enabled()):
            return base_fn(module, query, key, value, attention_mask, **kwargs)
        # checkpoint ONLY the SDPA math; q/k/v are the recompute inputs (offloaded by attn_act);
        # non-tensor args ride the closure. SDPA's attn_weights is None -> keep output a tensor.
        def _run(q, k, v):
            out, _ = base_fn(module, q, k, v, attention_mask, **kwargs)
            return out

        attn_output = checkpoint(_run, query, key, value, use_reentrant=False)
        return attn_output, None

    return _checkpointed


def install_sdpa_recompute(model) -> bool:
    """Idempotent, self-gated, text-scoped. Returns True iff the interface is now active."""
    if not _enabled():
        return False
    try:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        base_impl = getattr(text_cfg, "_attn_implementation", "sdpa") or "sdpa"
        if base_impl == _INTERFACE_NAME:  # already installed
            return True
        base_fn = ALL_ATTENTION_FUNCTIONS.get(base_impl) or ALL_ATTENTION_FUNCTIONS.get("sdpa")
        if base_fn is None:
            return False
        ALL_ATTENTION_FUNCTIONS.register(_INTERFACE_NAME, _make_checkpointed(base_fn))
        text_cfg._attn_implementation = _INTERFACE_NAME  # plain attr; get_interface reads it
        print(f"[asym] sdpa_recompute installed (base={base_impl}) on text attention", flush=True)
        return True
    except Exception as exc:  # never break the run
        print(f"[asym] sdpa_recompute install skipped: {exc!r}", flush=True)
        return False


__all__ = ["install_sdpa_recompute"]
