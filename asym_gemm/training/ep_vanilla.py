"""S5b/N2 VANILLA-EP rung (fix_gb200_ep.md): the real Megatron-shape baseline sEP
must beat, with DIFFERENT data per rank (the mandated shape).

Design (allgather dispatcher — the Megatron variant that moves LESS than row-a2a at
topk=8, where ~every token hits both expert halves):
  entry : differentiable all_gather(hidden) + plain all_gather(topk idx/weights —
          detached router outputs) => every rank sees the GLOBAL batch.
  compute: rank r's experts module is a slice_experts_for_ep branch over experts
          [r*E/W, (r+1)*E/W) — dim-0 bank views + per-rank sliced LoRA; the existing
          ep_expert_range metadata slice runs the UNMODIFIED pipeline on exactly the
          rows routed to owned experts => the device-local PARTIAL over global tokens.
  exit  : differentiable reduce_scatter of the partial => each rank's own tokens'
          completed outputs. Both collectives are UNCONDITIONAL per call (empty-route
          steps still communicate) so fwd/GC-recompute stay lockstep across ranks.

Gradient semantics (structural, exact vs sEP): token outputs are row-local, so
d(loss_j)/d(hidden_i) == 0 for j != i — the collective backward returns exactly
d(loss_i)/d(hidden_i), as sEP computes. Expert-LoRA grads accrue on the OWNER as
sum over BOTH shards' tokens = 2x the mean-allreduce value sEP steps with; the
post-backward hook therefore scales owned grads by 1/world and SKIPS them in the
shared-param allreduce (vanilla EP shards expert optimizer state by owner).

Transport note: run with ASYM_ARENA_SHM=0 — vanilla EP has no ownerless shared
arena; each rank privately pins and streams only its owned banks (same pinned-C2C
transport class as sEP, so the A/B isolates assignment policy + exchange cost).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn


def vanilla_enabled() -> bool:
    return (
        os.environ.get("ASYM_EP_VANILLA") == "1"
        and int(os.environ.get("WORLD_SIZE", "1") or 1) >= 2
    )


# ASYM_EP_VANILLA_TIMING>=1: host-BLOCK attribution (perf_counter) for the suspected
# serialization sites; >=2 adds CUDA-event GPU time of the wrapper collectives (nccl
# kernel duration INCLUDES peer-arrival wait => direct rendezvous quantification).
# pad_item_s is fed from frozen_linear's per-launch `.item()` (the mid-layer drain).
_TIMING = int(os.environ.get("ASYM_EP_VANILLA_TIMING", "0") or 0)
# fix_ep (2026-07-11): periodic HOST re-alignment — every N layer-invocations
# both ranks fully drain (synchronize). Receipt: TIMING=2's incidental 48-call
# sweep sync damped the residual inter-rank oscillation by 14 s/step; this is
# the deliberate, tunable version. 0 = off.
_ALIGN_EVERY = int(os.environ.get("ASYM_EP_VANILLA_ALIGN_EVERY", "0") or 0)
_ALIGN_COUNT = 0


def _maybe_align() -> None:
    global _ALIGN_COUNT
    if _ALIGN_EVERY <= 0:
        return
    _ALIGN_COUNT += 1
    if _ALIGN_COUNT % _ALIGN_EVERY == 0:
        torch.cuda.synchronize()
_ACC = {"ag_s": 0.0, "rs_s": 0.0, "slice_item_s": 0.0, "pad_item_s": 0.0, "calls": 0}
_EVS: list = []


def _timing_add(key: str, seconds: float) -> None:
    if _TIMING:
        _ACC[key] += seconds


def _timing_events(kind: str):
    """Returns (ev0, ev1) to record around a collective, or None."""
    import torch as _t

    if _TIMING < 2:
        return None
    ev0 = _t.cuda.Event(enable_timing=True)
    ev1 = _t.cuda.Event(enable_timing=True)
    _EVS.append((kind, ev0, ev1))
    return ev0, ev1


def _timing_tick() -> None:
    if not _TIMING:
        return
    _ACC["calls"] += 1
    if _ACC["calls"] % 48 == 0:
        import sys
        import torch.distributed as _d

        gpu = {"ag": 0.0, "rs": 0.0}
        if _EVS:
            _EVS[-1][2].synchronize()  # one sweep-boundary sync
            for kind, e0, e1 in _EVS:
                try:
                    gpu[kind] += e0.elapsed_time(e1) / 1e3
                except Exception:
                    pass
            _EVS.clear()
        r = _d.get_rank() if _d.is_initialized() else -1
        print(
            f"[ep_vanilla_timing] rank{r} calls={_ACC['calls']} "
            f"ag_host_s={_ACC['ag_s']:.2f} rs_host_s={_ACC['rs_s']:.2f} "
            f"slice_item_s={_ACC['slice_item_s']:.2f} pad_item_s={_ACC['pad_item_s']:.2f} "
            f"ag_gpu_s={gpu['ag']:.2f} rs_gpu_s={gpu['rs']:.2f}",
            file=sys.stderr, flush=True,
        )
        for k in ("ag_s", "rs_s", "slice_item_s", "pad_item_s"):
            _ACC[k] = 0.0


# ASYM_EP_VANILLA_FUSED=1: custom collective autograd Functions with PERSISTENT
# per-shape buffer rings (all shapes are static per run => steady-state allocs on the
# collective path -> 0) and fused-tensor NCCL ops (no per-rank list + cat). Targets
# the attempt-1..4 s20000 pathology (fresh multi-GB allocs per layer per direction).
_FUSED = os.environ.get("ASYM_EP_VANILLA_FUSED") == "1"
_RING: dict = {}

# fix_ep D3 (Megatron fused_a2a async pattern): run the fused collectives on a
# DEDICATED comm stream with event handoffs. The point is ENQUEUE decoupling:
# each rank's host reaches the collective at layer entry with an EMPTY comm
# stream, so the NCCL kernel starts as soon as both hosts arrive — instead of
# queueing behind the peer's main-stream compute backlog. Host syncs then drain
# only local compute; the inter-rank antiphase loses its feed. Receipt
# motivating this: TIMING=2's incidental 48-call sweep sync DAMPED the residual
# oscillation by 14 s/step (176.8 vs 190.7 at 32k) — the waits were
# inter-kernel idle, not NCCL kernel time.
_COMM_STREAM_ENABLED = (os.environ.get("ASYM_EP_VANILLA_COMM_STREAM", "1") or "1") == "1"
_COMM_STREAM = None


def _comm_stream():
    global _COMM_STREAM
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream()
    return _COMM_STREAM


def _run_collective(inputs: tuple, fn):
    """Run fn() on the comm stream: inputs ordered after main's writes; output
    handed back with a GPU-side event wait (the HOST never blocks here)."""
    if not _COMM_STREAM_ENABLED:
        return fn()
    main = torch.cuda.current_stream()
    comm = _comm_stream()
    comm.wait_stream(main)
    with torch.cuda.stream(comm):
        for t in inputs:
            if t.is_cuda:
                # keep the allocator from recycling main-stream tensors while
                # the comm stream still reads them (ring outputs are persistent
                # and need no protection)
                t.record_stream(comm)
        out = fn()
        ev = torch.cuda.Event()
        ev.record()
    main.wait_event(ev)
    return out


def _buf(tag: str, shape: tuple, ref: torch.Tensor) -> torch.Tensor:
    key = (tag, shape, ref.dtype)
    ring = _RING.get(key)
    if ring is None:
        ring = [torch.empty(shape, dtype=ref.dtype, device=ref.device),
                torch.empty(shape, dtype=ref.dtype, device=ref.device), 0]
        _RING[key] = ring
    ring[2] ^= 1
    return ring[ring[2]]


class _AllGatherFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        world = dist.get_world_size()
        out = _buf("ag_f", (world * x.shape[0],) + tuple(x.shape[1:]), x)
        xc = x.contiguous()
        _run_collective((xc,), lambda: dist.all_gather_into_tensor(out, xc))
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world = dist.get_world_size()
        gx = _buf("ag_b", (grad.shape[0] // world,) + tuple(grad.shape[1:]), grad)
        gc = grad.contiguous()
        _run_collective((gc,), lambda: dist.reduce_scatter_tensor(gx, gc))
        return gx


class _ReduceScatterFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial: torch.Tensor) -> torch.Tensor:
        world = dist.get_world_size()
        out = _buf("rs_f", (partial.shape[0] // world,) + tuple(partial.shape[1:]), partial)
        pc = partial.contiguous()
        _run_collective((pc,), lambda: dist.reduce_scatter_tensor(out, pc))
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world = dist.get_world_size()
        gp = _buf("rs_b", (world * grad.shape[0],) + tuple(grad.shape[1:]), grad)
        gc = grad.contiguous()
        _run_collective((gc,), lambda: dist.all_gather_into_tensor(gp, gc))
        return gp


def gather_moe_routing(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fix_ep D1: the two TINY plain allgathers only (detached router outputs).
    Every host read of the layer (slice bounds, pad prewarm) happens on these —
    draining only µs-class work — BEFORE the big hidden collective ever
    enqueues, so no CPU can block on the peer mid-layer."""
    world = dist.get_world_size()

    def _gather_plain(t: torch.Tensor) -> torch.Tensor:
        out = t.new_empty((world * t.shape[0],) + tuple(t.shape[1:]))
        dist.all_gather_into_tensor(out, t.contiguous())
        return out

    return _gather_plain(top_k_index), _gather_plain(top_k_weights)


def gather_moe_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """fix_ep D1: the DIFFERENTIABLE hidden allgather only (residual grads flow
    back to the home rank). Called AFTER the layer's last host read."""
    import time

    from torch.distributed.nn.functional import all_gather as all_gather_diff

    t0 = time.perf_counter()
    evs = _timing_events("ag")
    if evs is not None:
        evs[0].record()
    if _FUSED:
        hidden_g = _AllGatherFused.apply(hidden)
    else:
        hidden_g = torch.cat(all_gather_diff(hidden.contiguous()), dim=0)
    if evs is not None:
        evs[1].record()
    _timing_add("ag_s", time.perf_counter() - t0)
    return hidden_g


def gather_moe_inputs(
    hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LEGACY ORDER (pre-fix_ep D1; kept for the ASYM_EP_VANILLA_LEGACY_ORDER=1
    A/B receipt): tiny gathers then hidden AG, all BEFORE any metadata work —
    the mid-layer host reads then drain the big collective (the stagger)."""
    idx_g, w_g = gather_moe_routing(top_k_index, top_k_weights)
    hidden_g = gather_moe_hidden(hidden)
    return hidden_g, idx_g, w_g


def reduce_scatter_partial(partial: torch.Tensor, local_tokens: int) -> torch.Tensor:
    """Sum the per-rank partials over global tokens; return this rank's slice."""
    import time

    from torch.distributed.nn.functional import reduce_scatter as reduce_scatter_diff

    t0 = time.perf_counter()
    chunks = list(partial.split(local_tokens, dim=0))
    if len(chunks) != dist.get_world_size():
        raise RuntimeError(
            f"vanilla-EP partial rows {partial.shape[0]} != world*local {local_tokens}"
        )
    evs = _timing_events("rs")
    if evs is not None:
        evs[0].record()
    if _FUSED:
        out = _ReduceScatterFused.apply(partial)
    else:
        out = reduce_scatter_diff(torch.empty_like(chunks[0]), chunks)
    if evs is not None:
        evs[1].record()
    _timing_add("rs_s", time.perf_counter() - t0)
    _timing_tick()
    _maybe_align()
    return out


def install_vanilla_ep(model: nn.Module) -> dict:
    """Swap every qwen3 MoE block's experts for its owned-slice branch and flag the
    a2a wrapper. Must run BEFORE optimizer creation (branch LoRA params replace the
    full-E params in the module tree). Returns a receipt dict."""
    from .qwen3_moe import AsymQwen3Experts
    from .stp_moe import slice_experts_for_ep

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("ASYM_EP_VANILLA=1 requires torch.distributed initialized")
    rank, world = dist.get_rank(), dist.get_world_size()
    swapped, owned_numel, full_numel = 0, 0, 0
    lo = hi = -1
    for module in model.modules():
        # the WRAPPED block (is_qwen3_moe_block matches only raw HF blocks pre-wrap)
        if not getattr(module, "_is_asym_qwen3_moe_block", False):
            continue
        experts = getattr(module, "experts", None)
        if not isinstance(experts, AsymQwen3Experts):
            continue
        n_experts = int(experts.num_experts)
        if n_experts % world != 0:
            raise RuntimeError(f"E={n_experts} not divisible by world={world}")
        lo, hi = rank * n_experts // world, (rank + 1) * n_experts // world
        device = experts.gate_lora_A.device
        full_numel += sum(p.numel() for p in experts.parameters() if p.requires_grad)
        branch = slice_experts_for_ep(experts, lo, hi, device)
        branch.ep_vanilla_a2a = True
        for param in branch.parameters():
            if param.requires_grad:
                param._asym_ep_owner_scaled = True  # type: ignore[attr-defined]
                owned_numel += param.numel()
        module.experts = branch
        module._modules["experts"] = branch
        swapped += 1
    if swapped == 0:
        raise RuntimeError("ASYM_EP_VANILLA=1 but no qwen3 MoE blocks found to swap")
    return {
        "blocks": swapped,
        "rank": rank,
        "owned_range": [lo, hi],
        "owned_lora_numel": owned_numel,
        "full_lora_numel_before": full_numel,
    }
