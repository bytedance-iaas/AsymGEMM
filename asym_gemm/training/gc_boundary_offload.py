"""Substrate A — unsloth-GC cross-layer boundary offload (Stage 3): the capacity lever.

Drop-in replacement for LlamaFactory's ``UnslothGradientCheckpointing.apply``, used only when
``ASYM_UNSLOTH_GC_NVME`` is set (``run_lf_lora_sft.sh`` exports it when ``ASYM_NVME_ROLES``
contains ``activation``). Same math, same recompute, same ``save_on_cpu`` region, same
``UNSLOTH_GC_OUTER_HBM_EVERY_N`` diagnostic as the reference; the ONLY change is that the
boundary hidden-states ride a governor-tracked PINNED pool handle (spillable to NVMe under CPU
pressure) instead of a pageable ``save_for_backward`` tensor.

C9: the boundary copy is a detached CPU blob made inside ``Function.forward`` (no grad_fn), so
stashing it on ``ctx`` cannot form a reference cycle; ``save_for_backward`` is unusable because
its ``SavedVariable`` would pin the buffer and defeat the spill. A ``try/finally`` guarantees
``release_cpu`` even if the inner backward raises (no governor leak).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .activation_offload import ActivationOffloadManager


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


_BOUNDARY_MANAGER: "ActivationOffloadManager | None" = None   # ONE process-global manager owns boundaries
_OUTER_SAVE_COUNTER = 0

# R5 one-layer-ahead boundary prefetch: boundary tensors exist from FORWARD time
# (unlike the intra-layer saves born during backward-recompute), so at each layer's
# backward we enqueue the NEXT-to-be-consumed (= previous layer's) boundary H2D on
# the dedicated prefetch stream. LIFO registry mirrors consumption order; double-
# buffering emerges naturally (<= 1 staged boundary ahead). G-guarded.
_PENDING_BOUNDARY_HANDLES: list = []           # forward-order stack of live handles
_BOUNDARY_PREFETCH: dict = {}                  # id(handle) -> (staged, done_event)


def _boundary_prefetch_next() -> None:
    if not _PENDING_BOUNDARY_HANDLES:
        return
    from .activation_offload import (
        _h2d_prefetch_stream,
        prefetch_engaged_once,
        prefetch_free_ok,
        restage_prefetch_enabled,
    )

    if not restage_prefetch_enabled():
        return
    nxt = _PENDING_BOUNDARY_HANDLES[-1]
    if id(nxt) in _BOUNDARY_PREFETCH or not nxt.tensor.is_pinned():
        return
    if not prefetch_free_ok(nxt.nbytes):
        return
    prefetch_engaged_once("gc.boundary")
    device = torch.device("cuda")
    staged = torch.empty(tuple(nxt.tensor.shape), device=device, dtype=nxt.tensor.dtype)
    pref = _h2d_prefetch_stream(device)
    mgr = _get_boundary_manager()
    ready = mgr.take_cpu_ready_event(nxt)
    if ready is not None:
        pref.wait_event(ready)
    pref.wait_stream(torch.cuda.current_stream(device))
    with torch.no_grad(), torch.cuda.stream(pref):
        staged.copy_(nxt.tensor, non_blocking=True)
    done = torch.cuda.Event()
    done.record(pref)
    staged.record_stream(pref)
    _BOUNDARY_PREFETCH[id(nxt)] = (staged, done)


def _get_boundary_manager() -> ActivationOffloadManager:
    global _BOUNDARY_MANAGER
    if _BOUNDARY_MANAGER is None:
        _BOUNDARY_MANAGER = ActivationOffloadManager(pin_memory=True)
    return _BOUNDARY_MANAGER


def get_boundary_offload_stats() -> dict[str, Any]:
    return _BOUNDARY_MANAGER.snapshot() if _BOUNDARY_MANAGER is not None else {"enabled": False}


def _keep_on_hbm_diagnostic() -> bool:
    """Mirror of LF checkpointing.py:_should_keep_unsloth_outer_hidden_on_hbm (verbatim semantics)."""
    try:
        every_n = int(os.environ.get("UNSLOTH_GC_OUTER_HBM_EVERY_N", "0") or 0)
    except ValueError:
        every_n = 0
    if every_n <= 0:
        return False
    global _OUTER_SAVE_COUNTER
    keep = (_OUTER_SAVE_COUNTER % every_n) == 0
    _OUTER_SAVE_COUNTER += 1
    return keep


def get_asym_unsloth_gc_func():
    """Return the drop-in autograd Function.apply. Decorators mirror the reference verbatim
    (C9-torch: torch.cuda.amp.custom_fwd/_bwd still delegate correctly in torch 2.12)."""
    mgr = _get_boundary_manager()

    class AsymUnslothGCOffload(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, forward_function, hidden_states, *args):
            if hidden_states.is_cuda and not _keep_on_hbm_diagnostic():
                # K-1 FIX: offload FLATTENED 2-D [B*S, H] — the pool's dim-0 row
                # bucketing rounds a 3-D [B, S, H] boundary's dim0 (e.g. 8 -> 8192),
                # allocating a ~1000 GiB base buffer (measured; the reason this path
                # was never enabled outside NVMe).
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                handle = mgr.offload(flat, "gc.boundary")            # pinned pool + async D2H; governor tracks
                mgr.seal(handle)                                     # sole consumer is backward => seal now
                _PENDING_BOUNDARY_HANDLES.append(handle)             # R5: LIFO consumption registry
                ctx.asym_handle = handle                            # tensor rides the handle, NOT autograd (C9)
                ctx.asym_hidden_shape = tuple(int(d) for d in hidden_states.shape)
                ctx.hbm_hidden = None
            else:
                ctx.asym_handle = None
                ctx.hbm_hidden = hidden_states
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)
            ctx.forward_function = forward_function
            ctx.args = args
            ctx.save_for_backward()                                 # empty: detached CPU blob has no grad_fn (C9)
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, grad_output):
            h = ctx.asym_handle
            try:
                if h is not None:
                    # R5: this handle is the top of the LIFO registry — drop it and
                    # enqueue the NEXT one's H2D on the prefetch stream (one ahead).
                    try:
                        if _PENDING_BOUNDARY_HANDLES and _PENDING_BOUNDARY_HANDLES[-1] is h:
                            _PENDING_BOUNDARY_HANDLES.pop()
                        else:
                            _PENDING_BOUNDARY_HANDLES[:] = [
                                p for p in _PENDING_BOUNDARY_HANDLES if p is not h
                            ]
                    except Exception:
                        pass
                    pre = _BOUNDARY_PREFETCH.pop(id(h), None)
                    _boundary_prefetch_next()
                    if pre is not None:
                        staged, done = pre
                        torch.cuda.current_stream().wait_event(done)
                        hidden = staged.reshape(ctx.asym_hidden_shape)
                        ev = torch.cuda.Event()
                        ev.record()
                    else:
                        mgr.wait_cpu_ready(h)                       # ensure_local: un-spill if DURABLE
                        hidden = h.tensor.to("cuda", non_blocking=True).reshape(ctx.asym_hidden_shape)
                        ev = torch.cuda.Event()
                        ev.record()
                else:
                    hidden = ctx.hbm_hidden
                hidden = hidden.detach()
                hidden.requires_grad_(True)
                with torch.enable_grad():
                    if _env_flag("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU"):
                        # item 2 (fix_cpu_compute.md): lossless dedup of duplicate
                        # packs (q/k-norm fp32 double-saves); stock save_on_cpu
                        # when the dedup flag/policy is off.
                        from .save_dedup import save_on_cpu_maybe_dedup

                        with save_on_cpu_maybe_dedup(pin_memory=True):
                            outputs = ctx.forward_function(hidden, *ctx.args)
                    else:
                        outputs = ctx.forward_function(hidden, *ctx.args)
                    output = outputs[0] if isinstance(outputs, tuple) else outputs
                torch.autograd.backward(output, grad_output)
                if h is not None:
                    ev.synchronize()                               # H2D done => pinned buffer reusable (rule 6)
                return (None, hidden.grad) + (None,) * len(ctx.args)
            finally:
                if h is not None:
                    mgr.release_cpu(h)                             # LIFO release -> pool + governor (C9)
                ctx.asym_handle = None                             # drop the stash ref (C9)

    return AsymUnslothGCOffload.apply
