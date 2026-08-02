from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_INTERFACE_NAME = "asym_qchunked_attn"
_FALSEY = {"", "0", "false", "no", "off"}


def _chunk_rows() -> int:
    """0 = disabled; otherwise the q-chunk length in tokens (env-gated)."""
    for name in ("ASYMM_ATTN_QCHUNK_ROWS", "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS"):
        raw = os.environ.get(name)
        if raw is None or raw.strip().lower() in _FALSEY:
            continue
        rows = int(raw)
        if rows > 0:
            return rows
    return 0


_flex_attention = None
_create_block_mask = None
_BLOCK_MASK_CACHE: dict[tuple[int, int, int, str], Any] = {}


def _mode() -> str:
    for name in ("ASYMM_ATTN_QCHUNK_MODE", "ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_MODE"):
        raw = os.environ.get(name)
        if raw:
            return raw.strip().lower()
    return "ckpt"


def _load_flex() -> None:
    global _flex_attention, _create_block_mask
    if _flex_attention is not None:
        return
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # dynamic=True keeps stride-general kernels so plain (copy-free) mode can
    # feed strided q-chunk views without recompiles or guard failures.
    _flex_attention = torch.compile(flex_attention, dynamic=True)
    _create_block_mask = create_block_mask


def _offset_causal_mask(offset: int, q_len: int, kv_len: int, device: torch.device):
    key = (offset, q_len, kv_len, str(device))
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    def _mask_mod(b, h, q_idx, kv_idx):
        return (q_idx + offset) >= kv_idx

    mask = _create_block_mask(_mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)
    _BLOCK_MASK_CACHE[key] = mask
    return mask


def _qchunked_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
):
    # Memory lever, not a semantics change: only the plain causal training path
    # is chunked. Anything else (masks, dropout, decode) must fail loud rather
    # than silently diverge — loss parity is a campaign invariant.
    rows = _chunk_rows()
    seq = int(query.shape[2])
    if not (module.training and torch.is_grad_enabled()) or rows <= 0 or seq <= rows:
        return _BASE_FN(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    if attention_mask is not None:
        raise RuntimeError("asym qchunked attention supports only mask-free causal training (attention_mask given)")
    if dropout and dropout > 0.0:
        raise RuntimeError("asym qchunked attention does not support attention dropout")
    resolved_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    if not resolved_causal:
        raise RuntimeError("asym qchunked attention supports only causal attention")

    _load_flex()
    enable_gqa = bool(key.shape[1] != query.shape[1])
    kv_len = int(key.shape[2])
    # compiled flex guards on strides: chunks must be contiguous, and K/V must
    # be contiguous once up front (MLA builds K by concat -> may be strided).
    key = key.contiguous()
    value = value.contiguous()
    # Each chunk is CHECKPOINTED: autograd saves only (q_chunk, K, V) refs —
    # K/V are shared storages and q inputs are already attn-act offloaded —
    # instead of per-chunk flex internals (out/LSE x chunks x layers), which
    # both bloats the per-layer save-on-cpu pinned pools (the first-forward
    # host blowup, l47q24f COOM) and re-inflates backward liveness. Backward
    # recomputes one chunk's flex forward then frees it (fwd cost ~2x attn).
    # Fixed kernel config: flex's default autotune BENCHMARKS candidate
    # kernels at real tensor sizes; on GB200 coherent memory those benchmark
    # allocations spill HBM -> host pages (the ~7 GB/s "Cached" drain the
    # host sampler caught during compile) until the watchdog kills the node.
    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    }

    def _chunk_fn(q_chunk: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor, start: int, stop: int):
        block_mask = _offset_causal_mask(start, stop - start, kv_len, q_chunk.device)
        return _flex_attention(
            q_chunk,
            k_full,
            v_full,
            block_mask=block_mask,
            scale=scaling,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

    mode = _mode()
    outs = []
    for start in range(0, seq, rows):
        stop = min(start + rows, seq)
        if mode == "plain":
            # copy-free: strided q view + shared K/V storages; autograd saves
            # per-chunk flex internals (same total bytes as the SDPA saves).
            out_chunk = _chunk_fn(query[:, :, start:stop, :], key, value, start, stop)
        else:
            out_chunk = checkpoint(
                _chunk_fn,
                query[:, :, start:stop, :].contiguous(),
                key,
                value,
                start,
                stop,
                use_reentrant=False,
            )
        outs.append(out_chunk)
    attn_output = torch.cat(outs, dim=2)
    return attn_output.transpose(1, 2).contiguous(), None


_BASE_FN = None


def install_qchunked_attention(model) -> bool:
    """Idempotent, env-gated, text-scoped (same pattern as sdpa_recompute)."""
    global _BASE_FN
    if _chunk_rows() <= 0:
        return False
    try:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        base_impl = getattr(text_cfg, "_attn_implementation", "sdpa") or "sdpa"
        if base_impl == _INTERFACE_NAME:
            return True
        base_fn = ALL_ATTENTION_FUNCTIONS.get(base_impl) or ALL_ATTENTION_FUNCTIONS.get("sdpa")
        if base_fn is None:
            return False
        _BASE_FN = base_fn
        ALL_ATTENTION_FUNCTIONS.register(_INTERFACE_NAME, _qchunked_forward)
        text_cfg._attn_implementation = _INTERFACE_NAME
        print(f"[asym] qchunked attention installed (base={base_impl}, rows={_chunk_rows()})", flush=True)
        return True
    except Exception as exc:
        print(f"[asym] qchunked attention install skipped: {exc!r}", flush=True)
        return False
