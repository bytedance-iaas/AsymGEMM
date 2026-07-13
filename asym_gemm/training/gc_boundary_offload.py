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
                handle = mgr.offload(hidden_states, "gc.boundary")   # pinned pool + async D2H; governor tracks
                mgr.seal(handle)                                     # sole consumer is backward => seal now
                ctx.asym_handle = handle                            # tensor rides the handle, NOT autograd (C9)
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
                    mgr.wait_cpu_ready(h)                           # ensure_local: un-spill if DURABLE
                    hidden = h.tensor.to("cuda", non_blocking=True)  # pinned H2D (faster than pageable)
                    ev = torch.cuda.Event()
                    ev.record()
                else:
                    hidden = ctx.hbm_hidden
                hidden = hidden.detach()
                hidden.requires_grad_(True)
                with torch.enable_grad():
                    if _env_flag("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU"):
                        with torch.autograd.graph.save_on_cpu(pin_memory=True):
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
