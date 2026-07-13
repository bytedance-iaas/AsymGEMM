"""Stage I4 boundary autograd Functions (gb200_tp.md I1 contract, placed in I4).

Every cross-device exchange in the training graph is ONE of these Functions — mixing
regimes computes wrong grads:

  AllReduce2Fn : fwd = exchange+sum on both devs ; bwd = identity per dev
  Bcast01Fn    : fwd = copy dev0->dev1           ; bwd = pass g0, DROP g1 (g1==g0 by
                 construction under this regime — summing would double-count: dev1's
                 share already arrived via the first region boundary's bwd exchange)
  Join01Fn     : fwd = (x0,x1)->x0               ; bwd = g -> (g, bcast01(g)) [replaces
                 SPMD's replicated loss seed; without it dev1's row-shard grads are zero]
  TPRegionFn   : fwd = identity on (h0,h1)       ; bwd = allreduce2_out of the two
                 ACCUMULATED LOCAL dX partials (Megatron's f-operator at region ENTRY)

IN-PLACE RULE: forwards never mutate Fn inputs; backwards never mutate incoming
grad_outputs -> all exchanges here are the OUT-OF-PLACE allreduce2_out / fresh-copy
primitives (+record_stream inside the runtime).
"""
from __future__ import annotations

import torch

from .stp_runtime import get_runtime

__all__ = ["AllReduce2Fn", "Bcast01Fn", "Join01Fn", "TPRegionFn"]


class AllReduce2Fn(torch.autograd.Function):
    """Row-region exit (after o_proj / down_proj): fwd sums the two partials on BOTH
    devices; bwd hands each branch its own grad unchanged (g-operator)."""

    @staticmethod
    def forward(ctx, y0: torch.Tensor, y1: torch.Tensor):
        return get_runtime().allreduce2_out(y0, y1)

    @staticmethod
    def backward(ctx, g0: torch.Tensor, g1: torch.Tensor):
        return g0, g1


class Bcast01Fn(torch.autograd.Function):
    """Residual duplication right after embedding: fwd copies x0 -> dev1; bwd passes g0
    and DROPS g1 (do NOT sum — latent 2x on embedding grads)."""

    @staticmethod
    def forward(ctx, x0: torch.Tensor):
        rt = get_runtime()
        return x0, rt.bcast01(x0)

    @staticmethod
    def backward(ctx, g0: torch.Tensor, g1: torch.Tensor):
        return g0


class Join01Fn(torch.autograd.Function):
    """Top of the stack, before final-norm/lm_head (dev0-only): fwd consumes the x1
    branch and passes x0; bwd hands the SAME grad to both branches."""

    @staticmethod
    def forward(ctx, x0: torch.Tensor, x1: torch.Tensor):
        return x0

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        rt = get_runtime()
        return g, rt.bcast01(g)


class TPRegionFn(torch.autograd.Function):
    """Region ENTRY (the norm output feeding qkv / gate_up): fwd identity on both branch
    tensors; bwd = ONE allreduce2 of the ACCUMULATED local dX partials (autograd already
    summed every col module's base+LoRA partial locally per branch)."""

    @staticmethod
    def forward(ctx, h0: torch.Tensor, h1: torch.Tensor):
        # return aliases; autograd treats outputs-as-inputs as identity edges
        return h0, h1

    @staticmethod
    def backward(ctx, g0: torch.Tensor, g1: torch.Tensor):
        return get_runtime().allreduce2_out(g0, g1)
