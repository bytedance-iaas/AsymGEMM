"""Lossless dedup of duplicate ``save_on_cpu`` packs (fix_cpu_compute.md item 2).

The unsloth-GC recompute region runs under ``torch.autograd.graph.save_on_cpu``:
every autograd save inside one layer's recompute is copied D2H into freshly
allocated pinned memory. The q/k-norm (+rope-adjacent) fp32 inputs are saved
TWICE (multiple autograd nodes of the norm save the *same* tensor object) —
a ~400 GB/step copy class at 30B@32k, ~4x that at 128k, all duplicated.

This context manager is a drop-in for ``save_on_cpu(pin_memory=True)`` that
content-dedups those saves LOSSLESSLY:

- key = (tensor OBJECT identity, tensor._version). Same object + same version
  implies bit-identical content; object identity (via WeakTensorKeyDictionary)
  cannot alias a reused ``data_ptr`` because a live key keeps its storage
  alive, and a dead key drops its entry. NO anonymous data_ptr matching (that
  approach was assessed and REJECTED — silent wrong-gradient risk).
- pack: first save does exactly what torch 2.12's ``save_on_cpu`` does
  (pinned ``torch.empty`` + ``copy_(non_blocking=True)``); duplicate saves
  return the SAME shared pack (refcount++) and skip allocation + D2H.
- unpack: each consumer gets its OWN ``.to(device, non_blocking=True)`` copy —
  exactly ``save_on_cpu`` semantics (no shared GPU tensor, so a consumer
  mutating its unpacked tensor cannot corrupt a sibling; H2D is NOT deduped
  by design — lossless-by-construction beats the extra copy win).
- the dedup map lives only for the duration of the context (one layer's
  recompute); entries also die automatically with their source tensor.

Fp32 stays fp32 — NO recast (the bf16-recast variant changes backward
numerics and stays outside the lossless claim).

Gate: ``ASYMM_SAVE_ON_CPU_DEDUP=1`` (default OFF), or the placement policy
(placement.md P5 duplicate-copy-removal class) once item-2 gates pass.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from typing import Any

import torch
from torch.utils.weak import WeakTensorKeyDictionary

__all__ = ["enabled", "save_on_cpu_maybe_dedup", "DedupSaveOnCpu", "stats", "reset_stats"]

_TRUTHY = {"1", "true", "yes", "y", "on"}

_LOCK = threading.Lock()
_HITS = 0
_MISSES = 0
_BYTES_DEDUPED = 0
_ENGAGED_PRINTED = False


def enabled() -> bool:
    raw = os.environ.get(
        "ASYMM_SAVE_ON_CPU_DEDUP",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_SAVE_ON_CPU_DEDUP", ""),
    )
    if raw.strip().lower() in _TRUTHY:
        return True
    from . import placement_policy

    return placement_policy.enabled() and placement_policy.save_on_cpu_dedup()


def stats() -> dict:
    with _LOCK:
        return {
            "hits": _HITS,
            "misses": _MISSES,
            "bytes_deduped": _BYTES_DEDUPED,
        }


def reset_stats() -> None:
    global _HITS, _MISSES, _BYTES_DEDUPED
    with _LOCK:
        _HITS = 0
        _MISSES = 0
        _BYTES_DEDUPED = 0


class _SharedPack:
    """One CPU copy shared by every autograd node that saved the same tensor."""

    __slots__ = ("device", "cpu", "refs")

    def __init__(self, device: torch.device, cpu: torch.Tensor):
        self.device = device
        self.cpu = cpu
        self.refs = 1


def _record_hit(nbytes: int, shape: Any, dtype: Any) -> None:
    global _HITS, _BYTES_DEDUPED, _ENGAGED_PRINTED
    first = False
    with _LOCK:
        _HITS += 1
        _BYTES_DEDUPED += int(nbytes)
        if not _ENGAGED_PRINTED:
            _ENGAGED_PRINTED = True
            first = True
    if first:
        print(
            f"[asym-save-dedup] save_on_cpu dedup ENGAGED (first hit: shape={tuple(shape)}, "
            f"dtype={dtype})",
            file=sys.stderr,
            flush=True,
        )


def _record_miss() -> None:
    global _MISSES
    with _LOCK:
        _MISSES += 1


class DedupSaveOnCpu(torch.autograd.graph.saved_tensors_hooks):
    """``save_on_cpu(pin_memory=True)`` with same-object same-version pack dedup."""

    def __init__(self, pin_memory: bool = True, device_type: str = "cuda"):
        device_module = getattr(torch, device_type, torch.cuda)
        # tensor(weak) -> {version: _SharedPack}; scoped to this context instance
        seen: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self._seen = seen
        # attempt #2 (classified from production pack identities, 2026-07-16): the big
        # duplicate saves present as DIFFERENT Python wrappers on the SAME storage —
        # a weakref-ANCHORED alias map keyed by (storage ptr, offset, dtype, sizes,
        # strides) catches them; a hit is honoured only while the first wrapper is
        # alive (its storage cannot have been freed/reused) and both wrappers' version
        # counters still equal the packed version (view-lineage aliases share the
        # counter, so an in-place write between the saves is refused).
        alias_map: dict = {}
        self._alias = alias_map

        def _alias_key(t: torch.Tensor):
            try:
                return (
                    t.untyped_storage().data_ptr(),
                    t.storage_offset(),
                    t.dtype,
                    tuple(t.shape),
                    tuple(t.stride()),
                )
            except Exception:
                return None

        def pack_to_cpu(tensor: torch.Tensor):
            dedupable = (
                tensor.is_cuda
                and not tensor.is_sparse
                and tensor.layout == torch.strided
                and not tensor.is_conj()
                and not tensor.is_neg()
            )
            akey = None
            per = None
            if dedupable:
                per = seen.get(tensor)
                if per is not None:
                    shared = per.get(tensor._version)
                    if shared is not None:
                        shared.refs += 1
                        _record_hit(
                            tensor.numel() * tensor.element_size(), tensor.shape, tensor.dtype
                        )
                        return shared
                akey = _alias_key(tensor) if tensor.dtype == torch.float32 else None
                if akey is not None:
                    ent = alias_map.get(akey)
                    if ent is not None:
                        anchor, ver0, shared = ent
                        # 2b: STRONG anchor (production saved-wrappers are ephemeral;
                        # a weakref dies between the two saves). Pop on hit.
                        if anchor._version == ver0 and tensor._version == ver0:
                            shared.refs += 1
                            _record_hit(
                                tensor.numel() * tensor.element_size(), tensor.shape, tensor.dtype
                            )
                            alias_map.pop(akey, None)
                            return shared
                        alias_map.pop(akey, None)  # mutated -> refuse and drop
            # exact torch-2.12 save_on_cpu pack semantics (+ item-4 pinned ledger:
            # cap denial degrades to the unpinned save_on_cpu behaviour, never OOMs)
            is_pinnable = device_module.is_available() and not tensor.is_sparse
            nbytes = tensor.numel() * tensor.element_size()
            if is_pinnable:
                from . import pinned_ledger

                is_pinnable = pinned_ledger.try_reserve("save_on_cpu", nbytes)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=is_pinnable,
            )
            if is_pinnable:
                pinned_ledger.register_tensor(packed, "save_on_cpu")
            packed.copy_(tensor, non_blocking=is_pinnable)
            _record_miss()
            if dedupable:
                shared = _SharedPack(tensor.device, packed)
                if per is None:
                    per = {}
                    seen[tensor] = per
                per[tensor._version] = shared
                if akey is not None:
                    alias_map[akey] = (tensor, tensor._version, shared)  # STRONG (2b)
                    while len(alias_map) > 2:  # FIFO cap: measured duplicate lag is 2
                        alias_map.pop(next(iter(alias_map)), None)
                return shared
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            if isinstance(packed, _SharedPack):
                # independent H2D per consumer — exact save_on_cpu semantics
                return packed.cpu.to(packed.device, non_blocking=True)
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)

    def __exit__(self, *args):
        # bound the map's lifetime to the recompute region (packs are region-local;
        # the shared packs themselves live on inside the autograd nodes)
        self._seen.clear()
        self._alias.clear()
        return super().__exit__(*args)


def save_on_cpu_maybe_dedup(pin_memory: bool = True):
    """The production entry point: dedup context when enabled, else stock torch."""
    if enabled():
        return DedupSaveOnCpu(pin_memory=pin_memory)
    return torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)
