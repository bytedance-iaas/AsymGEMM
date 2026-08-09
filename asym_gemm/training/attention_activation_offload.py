from __future__ import annotations

from dataclasses import dataclass
import os
import types
import weakref
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from .activation_offload import ActivationOffloadManager, CPUActivationHandle
from .frozen_linear import AsymExecutionStats, AsymFrozenLinear, _check_backend, asym_bf16_cpu_right_matmul
from .host_weight import HostWeight
from .lora import _reset_lora_weights, grouped_expert_lora_cpu_left, normalize_lora_dtype


_SINGLE_GROUP_METADATA_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
_QKV_SHARE_ROLES = frozenset({"q_proj", "k_proj", "v_proj"})
# MLA attention (glm4_moe_lite): q_a_proj and kv_a_proj_with_mqa consume the
# SAME layer input — share one CPU copy exactly like the q/k/v trio. The
# b-projections consume distinct latents and stay independent (role.U).
_MLA_SHARE_ROLES = frozenset({"q_a_proj", "kv_a_proj_with_mqa"})
_ALL_SHARE_ROLES = _QKV_SHARE_ROLES | _MLA_SHARE_ROLES
# The role that completes each share group in module-forward order (v_proj is
# last of the trio; kv_a_proj_with_mqa is last of the MLA pair).
_SHARE_FLUSH_ROLES = frozenset({"v_proj", "kv_a_proj_with_mqa"})
_DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES = 1 * 1024**2
_DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_SAVED_TENSOR_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
}


def _align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _flatten_last_dim(x: torch.Tensor, in_features: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.shape[-1] != int(in_features):
        raise ValueError(f"expected input last dim {int(in_features)}, got {x.shape[-1]}")
    input_shape = tuple(int(dim) for dim in x.shape)
    return x.reshape(-1, int(in_features)).contiguous(), input_shape


def _restore_last_dim(x: torch.Tensor, input_shape: tuple[int, ...], out_features: int) -> torch.Tensor:
    return x.reshape(*input_shape[:-1], int(out_features))


# Times a REQUESTED pin fell back to pageable in this module (process-wide, across all
# wrapper instances — read as max-across-rows in artifacts, never summed). A pageable
# buffer silently downgrades every later touch of it to the host-blocking path.
_PIN_FALLBACK_CALLS = 0


def _empty_cpu_like_rows(source: torch.Tensor, rows: int) -> torch.Tensor:
    global _PIN_FALLBACK_CALLS
    shape = (int(rows), int(source.shape[1]))
    want_pin = bool(source.is_pinned())
    try:
        return torch.empty(shape, device="cpu", dtype=source.dtype, pin_memory=want_pin)
    except RuntimeError:
        if want_pin:
            _PIN_FALLBACK_CALLS += 1
        return torch.empty(shape, device="cpu", dtype=source.dtype)


def _pad_cpu_rows_to(source: torch.Tensor, rows: int) -> torch.Tensor:
    rows = int(rows)
    if source.dim() != 2:
        raise ValueError(f"CPU row padding expects a 2D tensor, got {tuple(source.shape)}")
    if source.device.type != "cpu":
        raise ValueError(f"CPU row padding expects a CPU tensor, got {source.device}")
    if int(source.shape[0]) == rows:
        return source.contiguous()
    if int(source.shape[0]) > rows:
        raise ValueError(f"cannot pad {int(source.shape[0])} rows down to {rows}")
    padded = _empty_cpu_like_rows(source, rows)
    with torch.no_grad():
        padded[: int(source.shape[0])].copy_(source, non_blocking=False)
        if rows > int(source.shape[0]):
            padded[int(source.shape[0]) :].zero_()
    return padded.contiguous()


def _pad_hbm_rows_to(source: torch.Tensor, rows: int) -> torch.Tensor:
    rows = int(rows)
    if source.dim() != 2:
        raise ValueError(f"HBM row padding expects a 2D tensor, got {tuple(source.shape)}")
    if int(source.shape[0]) == rows:
        return source.contiguous()
    if int(source.shape[0]) > rows:
        raise ValueError(f"cannot pad {int(source.shape[0])} rows down to {rows}")
    padded = torch.zeros((rows, int(source.shape[1])), device=source.device, dtype=source.dtype)
    if int(source.shape[0]) > 0:
        padded[: int(source.shape[0])].copy_(source)
    return padded.contiguous()


def _single_group_offsets_experts(device: torch.device | str, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    rows = int(m)
    if rows < 0:
        raise ValueError(f"single-group metadata row count must be non-negative, got {rows}")
    key = (str(device), rows)
    cached = _SINGLE_GROUP_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("attention activation offload metadata must be initialized before CUDA graph capture")
    offsets = torch.tensor([0, rows], device=device, dtype=torch.int32)
    experts = torch.tensor([0, -1], device=device, dtype=torch.int32)
    _SINGLE_GROUP_METADATA_CACHE[key] = (offsets, experts)
    return offsets, experts


def _attn_qkv_fwd_shared_enabled() -> bool:
    """qkv shared-stream LoRA-A forward (default ON, 2026-07-28): when q/k/v
    share one offloaded source, ONE CPU-left pass at n=3r computes all three
    S projections — the host activation is streamed once instead of three
    times. Bit-identical to the per-projection calls (same kernel, same
    reduce order per output column block). ASYMM_ATTN_QKV_FWD_SHARED=0 falls
    back to per-projection streams."""
    value = os.environ.get("ASYMM_ATTN_QKV_FWD_SHARED")
    if value is None or value == "":
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _attn_dual_da_enabled() -> bool:
    """N2 dual-dataflow at the attention site (default OFF): backward runs ONE
    X pass producing BOTH dA (stream-end stationary) and S (for dB), replacing
    the upstream-form dA matmul + the S offload/re-stage round trip. Forward
    then skips offloading S entirely (drops the pinned S buffer per module).
    Requires rank 64 + asym backend; other configs keep the legacy path."""
    return os.environ.get("ASYMM_ATTN_DUAL_DA", "").strip().lower() in {"1", "true", "on", "yes"}


def _attn_lora_a_grad_cpu_deposit_enabled() -> bool:
    """K-2 (cpu_compute.md): attention LoRA-A wgrad on the CPU worker (deposit design),
    mirroring the MoE Stage-3 path — removes the per-projection C2C U re-read.

    Policy-ON (fix_cpu_compute.md item 1): placement.md P3 rules — the per-call
    rows gate in _try_deposit_attn_lora_a_grad closes the manual 32k/128k split."""
    from . import placement_policy

    if placement_policy.enabled():
        return placement_policy.attn_wgrad_feature()
    return os.environ.get(
        "ASYMM_ATTN_LORA_A_GRAD_CPU",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_LORA_A_GRAD_CPU", "0"),
    ).strip().lower() in {"1", "true", "yes", "y", "on"}


# (task, manager, u_handle, shared_source): U buffers whose release is deferred until the
# worker's wgrad job has consumed them. Swept on the MAIN thread (pool is not thread-safe).
_ATTN_DEPOSIT_PENDING: list = []
_ATTN_PAIR_META: dict = {}
_ATTN_DEPOSIT_DIAG = False


_DEPOSIT_PENDING_BUDGET_BYTES = int(
    float(os.environ.get("ASYM_DEPOSIT_RETAIN_BUDGET_GB", "48")) * (1 << 30)
)
_DEPOSIT_RETAINED_BYTES = 0
_DEPOSIT_RETAINED_HIGH_WATER = 0  # P11: retained-bytes high-water (profile counter)


def deposit_retention_stats() -> dict:
    """P11 counters for the profile: P7 backpressure budget + high-water."""
    return {
        "budget_bytes": _DEPOSIT_PENDING_BUDGET_BYTES,
        "retained_bytes_high_water": _DEPOSIT_RETAINED_HIGH_WATER,
        "retained_bytes_now": _DEPOSIT_RETAINED_BYTES,
        "pending_now": len(_ATTN_DEPOSIT_PENDING),
    }


def _handle_bytes(h) -> int:
    return int(h.tensor.numel()) * int(h.tensor.element_size())


def _defer_deposit_release(task, manager, handle, shared) -> None:
    """BACKPRESSURE (2026-07-15, fixes the 32B 10 GiB/s retention OOM): before
    deferring a new source, block the PRODUCER on the oldest outstanding deposit
    until the global retained-bytes budget holds. Converts OOM into bounded
    slowdown; lossless; FIFO order preserved. Main-thread only (pool rule)."""
    global _DEPOSIT_RETAINED_BYTES, _DEPOSIT_RETAINED_HIGH_WATER
    from . import cpu_worker

    new_bytes = _handle_bytes(handle)
    while _ATTN_DEPOSIT_PENDING and _DEPOSIT_RETAINED_BYTES + new_bytes > _DEPOSIT_PENDING_BUDGET_BYTES:
        t0, m0, h0, s0 = _ATTN_DEPOSIT_PENDING.pop(0)
        cpu_worker.wait(t0)
        _DEPOSIT_RETAINED_BYTES -= _handle_bytes(h0)
        if s0 is None:
            m0.release_cpu(h0)
        else:
            s0.release()
    _ATTN_DEPOSIT_PENDING.append((task, manager, handle, shared))
    _DEPOSIT_RETAINED_BYTES += new_bytes
    if _DEPOSIT_RETAINED_BYTES > _DEPOSIT_RETAINED_HIGH_WATER:
        _DEPOSIT_RETAINED_HIGH_WATER = _DEPOSIT_RETAINED_BYTES


def _sweep_attn_deposit_releases(force: bool = False) -> None:
    if not _ATTN_DEPOSIT_PENDING:
        return
    from . import cpu_worker

    global _DEPOSIT_RETAINED_BYTES
    remaining = []
    retained = 0
    for entry in _ATTN_DEPOSIT_PENDING:
        task, manager, u_handle, shared = entry
        if force:
            cpu_worker.wait(task)
        if task.done.is_set():
            if shared is None:
                manager.release_cpu(u_handle)
            else:
                shared.release()
        else:
            retained += _handle_bytes(u_handle)
            remaining.append(entry)
    _ATTN_DEPOSIT_PENDING[:] = remaining
    _DEPOSIT_RETAINED_BYTES = retained


def _try_deposit_attn_lora_a_grad(a_param, d_s, u_handle, manager, shared_source, role):
    """Fire-and-forget CPU wgrad for one attention projection: dA = dS^T @ U_cpu written
    fp32 into the optimizer's grad buffer. Returns a dummy CUDA grad or None."""
    import asym_gemm as _ag
    from . import cpu_adam as _cpu_adam
    from . import cpu_worker
    from . import placement_policy
    from .qwen3_moe_finegrained import _DS_SLOTS, _DsSlots

    if placement_policy.enabled() and not placement_policy.attn_wgrad_deposit(
        int(d_s.shape[0])
    ):
        return None  # P3 rows gate: fall back to the GPU wgrad path

    kernel = getattr(_ag, "cpu_grouped_lora_a_grad_bf16", None)
    adam = _cpu_adam.get_active_adamw()
    if kernel is None or adam is None or not cpu_worker.enabled():
        return None
    u = u_handle.tensor
    if d_s.dtype != torch.bfloat16 or u.dtype != torch.bfloat16 or not u.is_contiguous():
        return None
    buf = adam.get_grad_deposit_buffer(a_param)
    if (
        buf is None
        or buf.dtype != torch.float32
        or tuple(buf.shape) != (int(d_s.shape[1]), int(u.shape[1]))
    ):
        return None
    m = int(d_s.shape[0])
    meta = _ATTN_PAIR_META.get(m)
    if meta is None:
        meta = (torch.tensor([0, m], dtype=torch.long), torch.zeros(1, dtype=torch.long))
        _ATTN_PAIR_META[m] = meta
    pairs, ge = meta
    slots = _DS_SLOTS.setdefault((tuple(d_s.shape), d_s.dtype, f"attn.{role}"), _DsSlots())
    slot_i, ds_pin = slots.acquire(d_s)
    ds_pin.copy_(d_s if d_s.is_contiguous() else d_s.contiguous(), non_blocking=True)
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream())
    from . import cpu_ops as _cpu_ops
    nt = _cpu_ops.wgrad_threads()
    out3d = buf.view(1, int(buf.shape[0]), int(buf.shape[1]))

    def _job(ev=ev, ds=ds_pin, x=u, out=out3d, p=pairs, g=ge, n=nt, k=kernel):
        ev.synchronize()  # same-stream FIFO: also guarantees U's earlier D2H completed
        k(ds, x, out, p, g, n)

    task = cpu_worker.submit_deposit(_job, tag="deposit.dA.attn")
    slots.tasks[slot_i] = task
    if not adam.register_grad_deposit(a_param, task):
        cpu_worker.wait(task)
        return None
    _defer_deposit_release(task, manager, u_handle, shared_source)
    global _ATTN_DEPOSIT_DIAG
    if not _ATTN_DEPOSIT_DIAG:
        _ATTN_DEPOSIT_DIAG = True
        import sys
        print("[asym-cpu-wgrad] K-2 attention deposit path ENGAGED", file=sys.stderr, flush=True)
    return torch.empty(a_param.shape, device=d_s.device, dtype=torch.bfloat16)


def _record_attn_hbm_gemm(stats: AsymExecutionStats | None, tag: str) -> None:
    if stats is None or not tag:
        return
    stats.attn_act_hbm_gemm_calls_by_tag[tag] = stats.attn_act_hbm_gemm_calls_by_tag.get(tag, 0) + 1


def _attn_keep_acts_hbm_enabled() -> bool:
    """S-mem fix (agent/impls/fix_asym.md §2.1): keep the attention projection source
    (U) and low-rank S in HBM across the GC-recompute forward->backward window instead
    of the per-layer CPU round trip. Under staged dispatch the backward dA GEMM
    re-stages U to GPU anyway (raw H2D — the nsys +66 us/tok bucket) right after the
    forward's D2H offload (+55 us/tok bucket); both copies vanish when the tensors stay
    resident. The wrapper runs inside the unsloth-GC recompute, so everything kept here
    is consumed by the SAME layer's backward (LIFO): peak cost ~= one layer's qkv+o
    sources (~8 GB @128k b3). Default off = pre-fix behavior."""
    value = os.environ.get("ASYMM_ATTN_ACT_KEEP_ACTS_HBM")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _alias_key(tensor: torch.Tensor):
    """Identity key for same-storage alias dedup (item 2 attempt #2): two wrappers
    with equal (storage ptr, offset, dtype, sizes, strides) read the same bytes."""
    try:
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
    except Exception:
        return None


def _attention_saved_tensor_min_bytes() -> int:
    raw = os.environ.get("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_MIN_BYTES")
    if raw is None or raw == "":
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_MIN_BYTES


def _attention_saved_tensor_dtypes() -> frozenset[torch.dtype]:
    raw = os.environ.get("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_DTYPES")
    if raw is None or raw.strip() == "":
        return _DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES
    allowed: set[torch.dtype] = set()
    for token in raw.replace(";", ",").split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key in {"all", "*"}:
            return _DEFAULT_SAVED_TENSOR_OFFLOAD_DTYPES
        dtype = _SAVED_TENSOR_DTYPE_ALIASES.get(key)
        if dtype is not None:
            allowed.add(dtype)
    return frozenset(allowed)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _in_backward_graph_task() -> bool:
    # >= 0 exactly while a backward graph task is executing on this thread,
    # i.e. during GC recompute forwards. See the same helper in
    # linear_attention_activation_offload.py for the full rationale.
    try:
        return torch._C._current_graph_task_id() != -1
    except AttributeError:
        return False


_H2D_RESTAGE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _attn_lora_chunk_enabled() -> bool:
    """Chunk the attention LoRA full-width delta/dx adds (probe flag, default off):
    fwd `out = base + s@B^T` and bwd `d_u += dS@A` each materialize a [rows, width]
    tensor (7.3 GiB for q_proj @120k b8). Chunking reuses fg_chunk_rows sizing."""
    raw = os.environ.get(
        "ASYMM_ATTN_ACT_LORA_CHUNK",
        os.environ.get("ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_ACT_LORA_CHUNK"),
    )
    if raw is None or raw == "":
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _add_matmul_rows_(dest: torch.Tensor, lhs: torch.Tensor, rhs_t: torch.Tensor, *, scale: float = 1.0) -> None:
    """dest += (lhs @ rhs_t) * scale, chunked over rows when the chunk budget allows.
    C2 diet (fix_asym S2/S0'): under ASYMM_FUSED_LORA_ADDMM=1 the chain fuses into one
    addmm_ epilogue (kills two full-width elementwise sweeps + the [M,out] temp)."""
    from .activation_offload import fg_chunk_rows

    rows = int(dest.shape[0])
    fused = (
        os.environ.get("ASYMM_FUSED_LORA_ADDMM", "").strip().lower() in {"1", "true", "yes", "on"}
        and dest.dim() == 2
        and lhs.dim() == 2
        and rhs_t.dim() == 2
        and dest.dtype == lhs.dtype == rhs_t.dtype
    )
    chunk = fg_chunk_rows(rows, int(dest.shape[1]), dest.element_size()) if _attn_lora_chunk_enabled() else 0
    if chunk <= 0:
        if fused:
            dest.addmm_(lhs, rhs_t, alpha=float(scale))
            return
        part = lhs @ rhs_t
        if float(scale) != 1.0:
            part = part * float(scale)
        dest.add_(part.to(dtype=dest.dtype))
        return
    for row_start in range(0, rows, chunk):
        row_end = min(rows, row_start + chunk)
        if fused:
            dest[row_start:row_end].addmm_(lhs[row_start:row_end], rhs_t, alpha=float(scale))
            continue
        part = lhs[row_start:row_end] @ rhs_t
        if float(scale) != 1.0:
            part = part * float(scale)
        dest[row_start:row_end].add_(part.to(dtype=dest.dtype))
        del part


def _h2d_restage_stream(device: torch.device) -> torch.cuda.Stream:
    """Per-device side stream for backward restage copies (Megatron-style: the host never
    blocks on a copy; the compute stream waits on an EVENT instead)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = _H2D_RESTAGE_STREAMS.get(idx)
    if stream is None:
        with torch.cuda.device(idx):
            stream = torch.cuda.Stream()
        _H2D_RESTAGE_STREAMS[idx] = stream
    return stream


def _empty_strided_cpu_like(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    global _PIN_FALLBACK_CALLS
    shape = tuple(int(dim) for dim in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    want_pin = bool(pin_memory and torch.cuda.is_available())
    pinned = want_pin
    if pinned:
        # item 4 (fix_cpu_compute.md): these per-save fresh pinned allocs are the
        # largest untracked page-locked class on dense models — book them under the
        # "saved" family; cap denial degrades to the unpinned (sync-copy) behaviour.
        from . import pinned_ledger

        nbytes = tensor.numel() * tensor.element_size()
        pinned = pinned_ledger.try_reserve("saved", int(nbytes))
        if not pinned:
            # Pageable fallback: _pack leaves ready_event=None, so _unpack takes the
            # host-blocking branch — count it (a pin was requested but denied) so
            # async-unpack A/Bs can distinguish "on" from "silently degraded".
            _PIN_FALLBACK_CALLS += 1
    try:
        out = torch.empty_strided(shape, stride, device="cpu", dtype=tensor.dtype, pin_memory=pinned)
    except RuntimeError:
        if pinned:
            from . import pinned_ledger

            pinned_ledger.release("saved", int(tensor.numel() * tensor.element_size()))
            _PIN_FALLBACK_CALLS += 1
        return torch.empty_strided(shape, stride, device="cpu", dtype=tensor.dtype)
    if pinned:
        pinned_ledger.register_tensor(out, "saved")
    return out


@dataclass
class _SavedTensorOffloadHandle:
    tensor: torch.Tensor
    original_device: torch.device
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]
    original_stride: tuple[int, ...]
    nbytes: int
    tag: str
    ready_event: torch.cuda.Event | None = None
    # R5 prefetch: fresh staged tensor + its copy-done event, issued at region exit
    # on the DEDICATED prefetch stream (per-tensor event); consumed once by _unpack.
    prefetch_staged: "torch.Tensor | None" = None
    prefetch_done: "torch.cuda.Event | None" = None


@dataclass
class _RopeRecipeSavedHandle:
    """P2: a saved rope output represented by its recompute recipe (no bytes)."""

    recipe: object


_ATTN_DEDUP_ENGAGED = False


_ALIAS_MISS_DIAG_SEEN: set = set()


def _alias_miss_diag_once(cls: str, reason: str) -> None:
    key = (cls, reason)
    if key in _ALIAS_MISS_DIAG_SEEN or len(_ALIAS_MISS_DIAG_SEEN) > 24:
        return
    _ALIAS_MISS_DIAG_SEEN.add(key)
    import sys

    print(f"[attn-alias-miss] {cls}: {reason}", file=sys.stderr, flush=True)


_ALIAS_MISS_DIAG_COUNTS: dict = {}


def _alias_miss_diag(wrapper, tensor: torch.Tensor, akey, alias_map: dict) -> None:
    """First THREE lookup-misses per (dtype, shape) class (the first is always the
    class's initial pack; the second is the informative duplicate near-miss): report
    the closest near-miss so the exact key component that differs is visible."""
    cls = f"{tensor.dtype}.{tuple(tensor.shape)}"
    n = _ALIAS_MISS_DIAG_COUNTS.get(cls, 0)
    if n >= 3 or len(_ALIAS_MISS_DIAG_COUNTS) > 24:
        return
    _ALIAS_MISS_DIAG_COUNTS[cls] = n + 1
    near = None
    for k in alias_map.keys():
        if k[0] == akey[0]:  # same storage ptr
            near = ("same-sptr", k)
            break
        if k[2] == akey[2] and k[3] == akey[3]:  # same dtype+shape
            near = near or ("same-shape-diff-storage", k)
    import sys

    if near is None:
        print(
            f"[attn-alias-miss] {cls}: no candidate in map (map={len(alias_map)} keys) — "
            f"first save of this class in window OR lag exceeded FIFO",
            file=sys.stderr,
            flush=True,
        )
    else:
        kind, k = near
        print(
            f"[attn-alias-miss] {cls}: near-miss {kind}: mine=(sptr={akey[0]},off={akey[1]},"
            f"strides={akey[4]}) cand=(sptr={k[0]},off={k[1]},strides={k[4]})",
            file=sys.stderr,
            flush=True,
        )


def _attn_dedup_engaged_once(tag: str) -> None:
    global _ATTN_DEDUP_ENGAGED
    if _ATTN_DEDUP_ENGAGED:
        return
    _ATTN_DEDUP_ENGAGED = True
    import sys

    print(
        f"[asym-save-dedup] attention saved-tensor dedup ENGAGED (first hit tag={tag})",
        file=sys.stderr,
        flush=True,
    )


# item-2 diagnosis mode (inert unless ASYM_ATTN_SAVED_TENSOR_DEDUP_DEBUG=1): log the
# identity signature of the first few packs per shape so production same-shape pairs
# can be classified (same object / same storage / different tensors).
_DEDUP_DEBUG = os.environ.get("ASYM_ATTN_SAVED_TENSOR_DEDUP_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on"
}
_DEDUP_DEBUG_COUNTS: dict = {}


def _dedup_debug_log(wrapper, tensor: torch.Tensor) -> None:
    key = (tuple(tensor.shape), str(tensor.dtype))
    n = _DEDUP_DEBUG_COUNTS.get(key, 0)
    if n >= 8:
        return
    _DEDUP_DEBUG_COUNTS[key] = n + 1
    import sys

    try:
        sptr = tensor.untyped_storage().data_ptr()
    except Exception:
        sptr = -1
    print(
        f"[attn-dedup-debug] pack shape={key[0]} dtype={key[1]} id={id(tensor)} "
        f"ver={tensor._version} dptr={tensor.data_ptr()} sptr={sptr} "
        f"off={tensor.storage_offset()} req={tensor.requires_grad} leaf={tensor.is_leaf} "
        f"wrapper={id(wrapper)}",
        file=sys.stderr,
        flush=True,
    )


class AttentionSavedTensorOffloadWrapper:
    """Forward wrapper that offloads large attention saved tensors to CPU."""

    def __init__(
        self,
        module: nn.Module,
        *,
        pin_memory: bool = True,
        min_bytes: int | None = None,
        require_grad: bool | None = None,
        allowed_dtypes: set[torch.dtype] | frozenset[torch.dtype] | None = None,
    ) -> None:
        self.module = module
        self.original_forward: Callable[..., Any] = module.forward
        self.pin_memory = bool(pin_memory)
        self.min_bytes = _attention_saved_tensor_min_bytes() if min_bytes is None else max(0, int(min_bytes))
        self.allowed_dtypes = (
            _attention_saved_tensor_dtypes()
            if allowed_dtypes is None
            else frozenset(dtype for dtype in allowed_dtypes if isinstance(dtype, torch.dtype))
        )
        self.require_grad = (
            _env_bool("ASYM_ATTN_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD", True)
            if require_grad is None
            else bool(require_grad)
        )
        self.skip_in_backward = _env_bool(
            "ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD",
            # default AUTO: when the LF unsloth-GC save_on_cpu recompute wrapper is
            # active, the in-backward offload here is a redundant duplicate round
            # trip (fix_qwen3.5.md §10b C3) — skip it; explicit env overrides.
            _env_bool("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU", False),
        )
        self.skipped_backward_calls = 0
        self.calls = 0
        self.offload_calls = 0
        self.unpack_calls = 0
        self.skipped_tensors = 0
        self.skipped_bytes = 0
        # item 2 (fix_cpu_compute.md): lossless dedup of duplicate saves (the fp32
        # q/k-norm class is saved TWICE per layer through this pack). Map lives only
        # for one run() (one forward/recompute region); key = (object, _version).
        # Attempt #2 (2026-07-16, classified from production pack identities): the
        # big fp32 pairs are SAME-STORAGE ALIASES with different Python wrappers, so
        # a secondary weakref-ANCHORED alias map keyed by
        # (storage_ptr, offset, dtype, sizes, strides) catches them. An alias hit is
        # honoured only while the FIRST wrapper is still alive (anchor: its storage
        # cannot have been freed/reused) and both wrappers' version counters still
        # equal the packed version (view-lineage aliases share the counter, so any
        # in-place write between the saves is refused).
        self.dedup_hits = 0
        self.dedup_bytes = 0
        self.recipe_packs = 0
        self.recipe_bytes_avoided = 0
        self.recipe_unpacks = 0
        self._dedup_seen = None
        self._dedup_alias = None
        self._region_handles: "list[_SavedTensorOffloadHandle] | None" = None
        self.offloaded_bytes = 0
        self.cpu_owned_bytes = 0
        self.cpu_peak_bytes_live = 0
        self.staged_bytes = 0
        self.max_stage_bytes_live = 0
        self.offload_bytes_by_tag: dict[str, int] = {}
        self.cpu_bytes_by_tag: dict[str, int] = {}
        self.cpu_peak_by_tag: dict[str, int] = {}
        self.stage_bytes_by_tag: dict[str, int] = {}
        self.stage_peak_by_tag: dict[str, int] = {}
        self.dtype_counts: dict[str, int] = {}
        self.shape_counts: dict[str, int] = {}
        self._sync_module_stats()

    def install(self) -> None:
        setattr(self.module, "_asym_attention_saved_tensor_offload_wrapper", self)
        self.module.forward = types.MethodType(_attention_saved_tensor_offload_forward, self.module)  # type: ignore[method-assign]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if self.skip_in_backward and _in_backward_graph_task():
            # Grad-enabled forwards inside an active backward graph task are GC
            # recomputes: offloading there is a same-step D2H+H2D round trip that
            # doubles saved-tensor residency instead of shrinking the peak.
            self.skipped_backward_calls += 1
            self._sync_module_stats()
            return self.original_forward(*args, **kwargs)
        from . import save_dedup as _save_dedup

        prev_seen = self._dedup_seen
        prev_alias = self._dedup_alias
        prev_region = self._region_handles
        if _save_dedup.enabled():
            from torch.utils.weak import WeakTensorKeyDictionary

            self._dedup_seen = WeakTensorKeyDictionary()
            self._dedup_alias = {}
        self._region_handles = []
        try:
            with saved_tensors_hooks(self._pack, self._unpack):
                return self.original_forward(*args, **kwargs)
        finally:
            self._prefetch_region(self._region_handles)
            self._dedup_seen = prev_seen
            self._dedup_alias = prev_alias
            self._region_handles = prev_region

    def _prefetch_region(self, handles: "list[_SavedTensorOffloadHandle] | None") -> None:
        """R5: at region exit (the earliest legal point — these saves are born during
        the layer's backward-recompute, so one-layer-ahead is impossible by
        construction), issue every packed tensor's H2D restage on the DEDICATED
        prefetch stream in REVERSE pack order (last packed ~= first consumed), each
        with its own done event. Lazily-issued critical copies keep the original
        side stream, so they never queue behind these. G-guarded."""
        if not handles:
            return
        from .activation_offload import (
            _h2d_prefetch_stream,
            prefetch_engaged_once,
            prefetch_free_ok,
            restage_prefetch_enabled,
        )

        if not restage_prefetch_enabled():
            return
        total = sum(h.nbytes for h in handles if h.prefetch_staged is None)
        if total <= 0 or not prefetch_free_ok(total):
            return
        prefetch_engaged_once("attn.saved_region")
        for packed in reversed(handles):
            if packed.prefetch_staged is not None or not packed.tensor.is_pinned():
                continue
            if packed.original_device.type != "cuda":
                continue
            staged = torch.empty_strided(
                packed.original_shape,
                packed.original_stride,
                device=packed.original_device,
                dtype=packed.original_dtype,
            )
            pref = _h2d_prefetch_stream(packed.original_device)
            if packed.ready_event is not None:
                pref.wait_event(packed.ready_event)
            pref.wait_stream(torch.cuda.current_stream(packed.original_device))
            with torch.no_grad(), torch.cuda.stream(pref):
                staged.copy_(packed.tensor, non_blocking=True)
            done = torch.cuda.Event(enable_timing=True)
            done.record(pref)
            staged.record_stream(pref)
            staged._asym_restage_keepalive = packed.tensor  # type: ignore[attr-defined]
            packed.prefetch_staged = staged
            packed.prefetch_done = done

    def _should_offload(self, tensor: torch.Tensor) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type != "cuda":
            return False
        if tensor.dtype not in self.allowed_dtypes:
            return False
        if self.require_grad and not tensor.requires_grad:
            self.skipped_tensors += 1
            self.skipped_bytes += int(tensor.numel() * tensor.element_size())
            return False
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes < self.min_bytes:
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        # Skip ONLY real parameters (leaf+grad weights). qwen3.5's fla delta-net
        # emits leaf+requires_grad *activations* (param=False) — those must offload.
        if isinstance(tensor, torch.nn.Parameter):
            self.skipped_tensors += 1
            self.skipped_bytes += nbytes
            return False
        return True

    def _tag_for(self, tensor: torch.Tensor) -> str:
        dtype_name = str(tensor.dtype).replace("torch.", "")
        shape = "x".join(str(int(dim)) for dim in tensor.shape) or "scalar"
        return f"saved.{dtype_name}.{shape}"

    def _pack(self, tensor: torch.Tensor) -> "torch.Tensor | _SavedTensorOffloadHandle | _RopeRecipeSavedHandle":
        # P2 (2026-07-21): a rope output carrying a recompute recipe is saved as the
        # RECIPE (no bytes copied, no pinned buffer) and rebuilt bit-identically at
        # unpack from the norm wrapper's already-offloaded bf16 input. Explicit
        # object-attribute identification on the exact tensor SDPA saves — never
        # anonymous packed-list matching.
        recipe = getattr(tensor, "_asym_rope_recipe", None)
        if recipe is not None and getattr(recipe, "shared", None) is not None:
            tensor._asym_rope_recipe = None  # consumed by this save
            self.recipe_packs += 1
            self.recipe_bytes_avoided += int(tensor.numel() * tensor.element_size())
            return _RopeRecipeSavedHandle(recipe=recipe)
        if not self._should_offload(tensor):
            return tensor
        # item 2: same-object same-version duplicate saves share ONE handle (skips the
        # duplicate pinned alloc + D2H). Unpack stays per-consumer (independent staged
        # copies), so sharing the handle is lossless by construction.
        if _DEDUP_DEBUG:
            _dedup_debug_log(self, tensor)
        seen = self._dedup_seen
        dedupable = (
            seen is not None
            and tensor.layout == torch.strided
            and not tensor.is_conj()
            and not tensor.is_neg()
        )
        per = None
        akey = None
        if dedupable:
            per = seen.get(tensor)
            if per is not None:
                shared = per.get(tensor._version)
                if shared is not None:
                    self.dedup_hits += 1
                    self.dedup_bytes += int(tensor.numel() * tensor.element_size())
                    _attn_dedup_engaged_once(shared.tag)
                    return shared
            # attempt #2b: same-storage alias lookup. The anchor must be a STRONG ref:
            # production saved-wrappers are EPHEMERAL (a fresh wrapper per save), so a
            # weakref anchor dies between the two saves (measured: zero alias hits).
            # Anchors live in a small FIFO (N=4) — the duplicate save arrives within
            # the same norm a couple of packs later while the tensor is alive anyway,
            # so the strong ref extends no lifetime in practice; entries pop on hit
            # and the whole map clears at region exit.
            # fp32-only: the measured alias-duplicate classes are fp32 (norm upcast
            # + variance); bf16 packs are never anchored, so the FIFO cannot extend
            # the life of the big SDPA/rope saves (the +9.6 GiB G regression seen
            # with dtype-blind anchoring @32k).
            akey = _alias_key(tensor) if tensor.dtype == torch.float32 else None
            alias_map = self._dedup_alias
            if akey is not None and alias_map is not None:
                ent = alias_map.get(akey)
                if ent is not None:
                    anchor, ver0, shared = ent
                    if anchor._version == ver0 and tensor._version == ver0:
                        self.dedup_hits += 1
                        self.dedup_bytes += int(tensor.numel() * tensor.element_size())
                        _attn_dedup_engaged_once(shared.tag + " [alias]")
                        alias_map.pop(akey, None)  # served; release the anchor
                        return shared
                    alias_map.pop(akey, None)  # mutated -> refuse and drop
                    _alias_miss_diag_once(self._tag_for(tensor), "version-mismatch")
                else:
                    # one-line-per-class diagnosis of WHY big fp32 packs do not alias-hit
                    # (evidence for item-2 attempt #2b/#2c; prints at most once per class)
                    _alias_miss_diag(self, tensor, akey, alias_map)
        cpu = _empty_strided_cpu_like(tensor, pin_memory=self.pin_memory)
        non_blocking = bool(cpu.is_pinned())
        with torch.no_grad():
            cpu.copy_(tensor.detach(), non_blocking=non_blocking)
        ready_event = None
        if non_blocking:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(tensor.device))
        nbytes = _tensor_storage_nbytes(cpu)
        tag = self._tag_for(tensor)
        self.offload_calls += 1
        self.offloaded_bytes += nbytes
        self.cpu_owned_bytes += nbytes
        self.cpu_peak_bytes_live = max(self.cpu_peak_bytes_live, self.cpu_owned_bytes)
        self.offload_bytes_by_tag[tag] = self.offload_bytes_by_tag.get(tag, 0) + nbytes
        self.cpu_bytes_by_tag[tag] = self.cpu_bytes_by_tag.get(tag, 0) + nbytes
        self.cpu_peak_by_tag[tag] = max(self.cpu_peak_by_tag.get(tag, 0), self.cpu_bytes_by_tag[tag])
        dtype_key = str(tensor.dtype).replace("torch.", "")
        self.dtype_counts[dtype_key] = self.dtype_counts.get(dtype_key, 0) + 1
        shape_key = f"{dtype_key}:{tuple(int(dim) for dim in tensor.shape)}"
        self.shape_counts[shape_key] = self.shape_counts.get(shape_key, 0) + 1
        self._sync_module_stats()
        handle = _SavedTensorOffloadHandle(
            tensor=cpu,
            original_device=torch.device(tensor.device),
            original_dtype=tensor.dtype,
            original_shape=tuple(int(dim) for dim in tensor.shape),
            original_stride=tuple(int(value) for value in tensor.stride()),
            nbytes=nbytes,
            tag=tag,
            ready_event=ready_event,
        )
        if self._region_handles is not None:
            self._region_handles.append(handle)
        if dedupable:
            if per is None:
                per = {}
                seen[tensor] = per
            per[tensor._version] = handle
            if akey is not None and self._dedup_alias is not None:
                amap = self._dedup_alias
                amap[akey] = (tensor, tensor._version, handle)  # STRONG anchor (2b)
                while len(amap) > 2:  # FIFO cap: measured duplicate lag is 2 packs
                    amap.pop(next(iter(amap)), None)
        return handle

    def _unpack(self, packed: "torch.Tensor | _SavedTensorOffloadHandle | _RopeRecipeSavedHandle") -> torch.Tensor:
        if isinstance(packed, _RopeRecipeSavedHandle):
            from .qknorm_recompute import recompute_rope_saved

            self.recipe_unpacks += 1
            return recompute_rope_saved(packed.recipe)
        if not isinstance(packed, _SavedTensorOffloadHandle):
            return packed
        if packed.prefetch_staged is not None:
            # R5: the restage was issued at region exit on the prefetch stream —
            # wait its OWN event (per-tensor) and account the residual exposure.
            staged = packed.prefetch_staged
            done = packed.prefetch_done
            packed.prefetch_staged = None
            packed.prefetch_done = None
            compute_stream = torch.cuda.current_stream(packed.original_device)
            from .activation_offload import restage_gap_commit, restage_gap_events

            gap_wait, _unused = restage_gap_events(packed.original_device)
            if gap_wait is not None:
                gap_wait.record(compute_stream)
            compute_stream.wait_event(done)
            if gap_wait is not None and done is not None:
                restage_gap_commit(gap_wait, done, packed.nbytes, f"prefetch.{packed.tag}")
            self.unpack_calls += 1
            self.staged_bytes += packed.nbytes
            self.max_stage_bytes_live = max(self.max_stage_bytes_live, self.staged_bytes)
            self.stage_bytes_by_tag[packed.tag] = self.stage_bytes_by_tag.get(packed.tag, 0) + packed.nbytes
            self.stage_peak_by_tag[packed.tag] = max(self.stage_peak_by_tag.get(packed.tag, 0), packed.nbytes)
            self.staged_bytes = max(0, self.staged_bytes - packed.nbytes)
            self.cpu_owned_bytes = max(0, self.cpu_owned_bytes - packed.nbytes)
            self.cpu_bytes_by_tag[packed.tag] = max(0, self.cpu_bytes_by_tag.get(packed.tag, 0) - packed.nbytes)
            self._sync_module_stats()
            return staged
        staged = torch.empty_strided(
            packed.original_shape,
            packed.original_stride,
            device=packed.original_device,
            dtype=packed.original_dtype,
        )
        compute_stream = torch.cuda.current_stream(packed.original_device)
        if packed.tensor.is_pinned():
            # Async restage on the side stream; compute waits on the EVENT, the host never
            # blocks (the old non_blocking=False sync serialized the entire backward:
            # measured ~97s of host-blocked copies at s20000 under sTP).
            side = _h2d_restage_stream(packed.original_device)
            if packed.ready_event is not None:
                side.wait_event(packed.ready_event)
            side.wait_stream(compute_stream)  # staged alloc ordering
            with torch.no_grad(), torch.cuda.stream(side):
                staged.copy_(packed.tensor, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
            from .activation_offload import restage_gap_commit, restage_gap_events

            gap_wait, gap_done = restage_gap_events(packed.original_device)
            if gap_wait is not None:
                gap_wait.record(compute_stream)  # R5: compute-stream arrival before the wait
            compute_stream.wait_event(done)
            if gap_done is not None:
                gap_done.record(side)
                restage_gap_commit(gap_wait, gap_done, packed.nbytes, f"unpack.{packed.tag}")
            staged.record_stream(side)
            # keep the cpu buffer alive until the staged tensor dies (async copy source)
            staged._asym_restage_keepalive = packed.tensor  # type: ignore[attr-defined]
        else:
            import time as _t

            if packed.ready_event is not None:
                packed.ready_event.synchronize()
            _t0 = _t.perf_counter()
            with torch.no_grad():
                staged.copy_(packed.tensor, non_blocking=False)
            from .activation_offload import restage_gap_host_ms

            restage_gap_host_ms(f"unpack.{packed.tag}", (_t.perf_counter() - _t0) * 1000.0, packed.nbytes)
        self.unpack_calls += 1
        self.staged_bytes += packed.nbytes
        self.max_stage_bytes_live = max(self.max_stage_bytes_live, self.staged_bytes)
        self.stage_bytes_by_tag[packed.tag] = self.stage_bytes_by_tag.get(packed.tag, 0) + packed.nbytes
        self.stage_peak_by_tag[packed.tag] = max(self.stage_peak_by_tag.get(packed.tag, 0), packed.nbytes)
        self.staged_bytes = max(0, self.staged_bytes - packed.nbytes)
        self.cpu_owned_bytes = max(0, self.cpu_owned_bytes - packed.nbytes)
        self.cpu_bytes_by_tag[packed.tag] = max(0, self.cpu_bytes_by_tag.get(packed.tag, 0) - packed.nbytes)
        self._sync_module_stats()
        return staged

    def snapshot(self) -> dict[str, Any]:
        return {
            "attention_saved_tensor_offload": True,
            "calls": self.calls,
            "min_bytes": self.min_bytes,
            "require_grad": self.require_grad,
            "skip_in_backward": self.skip_in_backward,
            "skipped_backward_calls": self.skipped_backward_calls,
            # module-global (same value on every row): max across rows, never sum
            "pin_fallback_calls_module_global": _PIN_FALLBACK_CALLS,
            "allowed_dtypes": [str(dtype).replace("torch.", "") for dtype in sorted(self.allowed_dtypes, key=str)],
            "offloaded_bytes": self.offloaded_bytes,
            "cpu_owned_bytes": self.cpu_owned_bytes,
            "cpu_live_bytes": self.cpu_owned_bytes,
            "cpu_peak_bytes_live": self.cpu_peak_bytes_live,
            "staged_bytes": self.staged_bytes,
            "max_stage_bytes_live": self.max_stage_bytes_live,
            "num_offloads": self.offload_calls,
            "num_cpu_allocs": self.offload_calls,
            "num_stages": self.unpack_calls,
            "skipped_tensors": self.skipped_tensors,
            "skipped_bytes": self.skipped_bytes,
            "dedup_hits": self.dedup_hits,
            "dedup_bytes": self.dedup_bytes,
            "recipe_packs": self.recipe_packs,
            "recipe_bytes_avoided": self.recipe_bytes_avoided,
            "recipe_unpacks": self.recipe_unpacks,
            "offload_bytes_by_tag": dict(self.offload_bytes_by_tag),
            "cpu_bytes_by_tag": dict(self.cpu_bytes_by_tag),
            "cpu_peak_by_tag": dict(self.cpu_peak_by_tag),
            "stage_bytes_by_tag": dict(self.stage_bytes_by_tag),
            "stage_peak_by_tag": dict(self.stage_peak_by_tag),
            "dtype_counts": dict(self.dtype_counts),
            "shape_counts": dict(self.shape_counts),
            "pre_final_cleanup_cpu_owned_bytes": self.cpu_owned_bytes,
            "final_cleanup_released_bytes": 0,
        }

    def _sync_module_stats(self) -> None:
        setattr(self.module, "_last_activation_offload_stats", self.snapshot())


def _attention_saved_tensor_offload_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
    wrapper = getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None)
    if not isinstance(wrapper, AttentionSavedTensorOffloadWrapper):
        raise RuntimeError("attention saved-tensor offload wrapper is missing from module")
    return wrapper.run(*args, **kwargs)


def install_attention_saved_tensor_offload(
    module: nn.Module,
    *,
    min_bytes: int | None = None,
    require_grad: bool | None = None,
    allowed_dtypes: set[torch.dtype] | frozenset[torch.dtype] | None = None,
) -> AttentionSavedTensorOffloadWrapper:
    existing = getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None)
    if isinstance(existing, AttentionSavedTensorOffloadWrapper):
        return existing
    wrapper = AttentionSavedTensorOffloadWrapper(
        module,
        min_bytes=min_bytes,
        require_grad=require_grad,
        allowed_dtypes=allowed_dtypes,
    )
    wrapper.install()
    return wrapper


def is_attention_saved_tensor_offload_wrapper(module: nn.Module) -> bool:
    return isinstance(getattr(module, "_asym_attention_saved_tensor_offload_wrapper", None), AttentionSavedTensorOffloadWrapper)


def attention_saved_tensor_offload_module_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, module in model.named_modules() if name and is_attention_saved_tensor_offload_wrapper(module)
    )


def _source_key(tensor: torch.Tensor) -> tuple[str, int, int, tuple[int, ...], tuple[int, ...], str]:
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = id(tensor)
    return (
        str(tensor.device),
        storage_ptr,
        int(tensor.storage_offset()),
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
    )


class _SharedActivationSource:
    def __init__(self, context: "AttentionActivationOffloadContext", handle: CPUActivationHandle) -> None:
        self._context = context
        self.handle = handle
        self.refcount = 0
        self.released = False

    def retain(self) -> "_SharedActivationSource":
        if self.released:
            raise RuntimeError(f"cannot retain released shared attention source {self.handle.tag}")
        self.refcount += 1
        return self

    def release(self) -> None:
        if self.released:
            return
        self.refcount -= 1
        if self.refcount > 0:
            return
        if self.refcount < 0:
            self.refcount = 0
            raise RuntimeError(f"shared attention source {self.handle.tag} was released too many times")
        self.released = True
        self._context._release_source(self)


class AttentionActivationOffloadContext:
    """Per-attention-parent q/k/v CPU source sharing state."""

    def __init__(self, *, pin_memory: bool = True) -> None:
        self.manager = ActivationOffloadManager(pin_memory=pin_memory)
        self.source_share_hits = 0
        self.source_share_misses = 0
        self.source_share_duplicate_bytes_avoided = 0
        self.source_share_retained_bytes = 0
        self.source_share_released_bytes = 0
        self._cache: dict[tuple[str, int, int, tuple[int, ...], tuple[int, ...], str], _SharedActivationSource] = {}
        self._seen_roles: set[str] = set()
        self._lora_modules: dict[str, Any] = {}

    def register_lora_module(self, role: str, module: Any) -> None:
        """Track the q/k/v modules so the first shared-source consumer can
        batch all three LoRA-A projections into one streamed pass."""
        self._lora_modules[str(role)] = module

    @staticmethod
    def _shared_diag(reason: str) -> None:
        if os.environ.get("ASYMM_ATTN_QKV_FWD_SHARED_DIAG", "") not in {"1", "true", "on"}:
            return
        seen = getattr(AttentionActivationOffloadContext, "_shared_diag_seen", None)
        if seen is None:
            seen = set()
            AttentionActivationOffloadContext._shared_diag_seen = seen
        if reason in seen:
            return
        seen.add(reason)
        import sys

        print(f"[qkv-shared-diag] declined: {reason}", file=sys.stderr, flush=True)

    def shared_lora_a_forward(
        self,
        source: _SharedActivationSource,
        role: str,
        a: torch.Tensor,
        *,
        stats: AsymExecutionStats | None,
        backend: str,
    ) -> torch.Tensor | None:
        """qkv shared-stream LoRA-A forward: serve `role`'s S from a single
        CPU-left pass over the shared host source at n=3r. Returns None when
        the batch cannot be formed (caller falls back per-projection)."""
        role = str(role)
        cache = getattr(source, "lora_a_results", None)
        if cache is not None:
            s = cache.pop(role, None)
            if s is None:
                self._shared_diag(f"cache-miss-role:{role}")
                return None
            if stats is not None:
                stats.attn_act_lora_a_forward_calls += 1
                stats.attn_act_lora_a_shared_hits += 1
            return s
        if role not in _QKV_SHARE_ROLES:
            self._shared_diag(f"role-not-shared:{role}")
            return None
        roles = ("q_proj", "k_proj", "v_proj")
        mods = self._lora_modules
        if any(r not in mods for r in roles):
            self._shared_diag(
                f"roles-unregistered:{sorted(set(roles) - set(mods))} (have {sorted(mods)})"
            )
            return None
        weights: list[torch.Tensor] = []
        for r in roles:
            m = mods[r]
            # Weight-offload coordinators stage the WHOLE layer group in one
            # H2D (gather_group is layer-scoped and idempotent), so by the
            # time the first consumer runs — after its own gather — sibling
            # lora_a params already point at the staged slab. When a group is
            # NOT staged, param.data points at the CPU home and the device
            # check below rejects the batch safely. The cat below copies the
            # values, so later releases cannot invalidate the batch.
            w = a if r == role else m.lora_a
            if (
                w is None
                or w.device.type != "cuda"
                or w.dtype != torch.bfloat16
                or w.dim() != 2
            ):
                self._shared_diag(f"weight-unusable:{r}")
                return None
            weights.append(w)
        r0 = int(weights[0].shape[0])
        if any(tuple(w.shape) != (r0, int(weights[0].shape[1])) for w in weights):
            self._shared_diag("weight-shape-mismatch")
            return None
        results = None
        if os.environ.get("ASYMM_ATTN_QKV_FWD_TRIPLE", "1").lower() not in {"0", "false", "off"}:
            # N1 in-kernel triple: three adapters through the fetch-once slot
            # in ONE kernel — no A-cat memcpy, no split copies; bit-identical.
            try:
                from .cpu_left import grouped_expert_lora_triple_cpu_left

                u = source.handle.tensor
                offs, exps = _single_group_offsets_experts(weights[0].device, int(u.shape[0]))
                s0, s1, s2 = grouped_expert_lora_triple_cpu_left(
                    u, *(w.detach().unsqueeze(0).contiguous() for w in weights),
                    offsets=offs, experts=exps, stats=stats)
                results = {rr: t for rr, t in zip(roles, (s0, s1, s2))}
                if stats is not None:
                    stats.attn_act_lora_a_forward_calls += 1
            except RuntimeError:
                results = None
        if results is None:
            a_cat = torch.cat([w.detach().contiguous() for w in weights], dim=0).contiguous()
            s_cat = _dense_lora_a_cpu_left(
                source.handle.tensor,
                a_cat,
                stats=stats,
                tag="qkv.lora_a_forward_shared",
                backend=backend,
            )
            parts = s_cat.split(r0, dim=-1)
            results = {rr: p.contiguous() for rr, p in zip(roles, parts)}
            del s_cat, parts, a_cat
        source.lora_a_results = results  # lifetime rides the shared source
        if stats is not None:
            stats.attn_act_lora_a_shared_batches += 1
        return results.pop(role)

    def acquire_source(self, source: torch.Tensor, flat_source: torch.Tensor, role: str) -> _SharedActivationSource:
        role = str(role)
        if role not in _ALL_SHARE_ROLES:
            handle = self.manager.offload(flat_source, f"{role}.U")
            self.source_share_misses += 1
            return _SharedActivationSource(self, handle).retain()

        key = _source_key(source)
        cached = self._cache.get(key)
        if cached is not None and not cached.released:
            self.source_share_hits += 1
            self.source_share_duplicate_bytes_avoided += cached.handle.nbytes
            shared = cached.retain()
        else:
            handle = self.manager.offload(flat_source, f"{role}.U")
            shared = _SharedActivationSource(self, handle).retain()
            self._cache[key] = shared
            self.source_share_misses += 1
            self.source_share_retained_bytes += handle.nbytes

        self._seen_roles.add(role)
        if (
            role in _SHARE_FLUSH_ROLES
            or _QKV_SHARE_ROLES <= self._seen_roles
            or _MLA_SHARE_ROLES <= self._seen_roles
        ):
            self._cache.clear()
            self._seen_roles.clear()
        return shared

    def _release_source(self, source: _SharedActivationSource) -> None:
        self.source_share_released_bytes += self.manager.release_cpu(source.handle)

    def snapshot(self) -> dict[str, Any]:
        data = self.manager.snapshot()
        data.update(
            {
                # module-global (same value on every row): max across rows, never sum
                "pin_fallback_calls_module_global": _PIN_FALLBACK_CALLS,
                "source_share_hits": self.source_share_hits,
                "source_share_misses": self.source_share_misses,
                "source_share_duplicate_bytes_avoided": self.source_share_duplicate_bytes_avoided,
                "source_share_retained_bytes": self.source_share_retained_bytes,
                "source_share_released_bytes": self.source_share_released_bytes,
                "source_share_cache_entries": len(self._cache),
                "source_share_live_handles": sum(1 for source in self._cache.values() if not source.released),
            }
        )
        return data


def _update_snapshot(
    snapshot: dict[str, Any] | None,
    local_manager: ActivationOffloadManager,
    source_context: AttentionActivationOffloadContext | None,
) -> None:
    if snapshot is None:
        return
    snapshot.clear()
    snapshot.update(local_manager.snapshot())
    if source_context is not None:
        snapshot["source_context"] = source_context.snapshot()


def _dense_lora_a_cpu_left(
    u_drop_cpu: torch.Tensor,
    a: torch.Tensor,
    *,
    stats: AsymExecutionStats | None,
    tag: str,
    backend: str = "asym",
) -> torch.Tensor:
    """Compute dense LoRA-A as one logical CPU-left grouped projection."""

    _check_backend(backend)
    # fix_glm_t3.md (2026-08-08): the asym branch below hands CUDA-resident
    # LoRA-A weights to the native CPU-left binding, which segfaults on EVERY
    # such call (48/48 isolated repros incl. qwen shapes) — it is reachable
    # only when the shared q/k/v LoRA-A source path does not engage (e.g.
    # Flash's MLA projection pair), which no qwen run ever hits. Env-gated
    # reroute to the existing torch staging math; the GLM driver branch sets
    # the env, qwen paths see no change.
    if backend == "asym" and os.environ.get(
        "ASYMM_ATTN_LORA_A_CPU_LEFT_TORCH_STAGE", ""
    ).strip() == "1":
        backend = "torch"
    if u_drop_cpu.dim() != 2 or a.dim() != 2:
        raise ValueError(f"dense LoRA-A expects U=[M,in] and A=[r,in], got {tuple(u_drop_cpu.shape)} and {tuple(a.shape)}")
    if u_drop_cpu.dtype != torch.bfloat16 or a.dtype != torch.bfloat16:
        raise ValueError("dense LoRA-A CPU-left path expects BF16 operands")
    if u_drop_cpu.device.type != "cpu":
        raise ValueError(f"dense LoRA-A expects a CPU source activation, got {u_drop_cpu.device}")
    if not u_drop_cpu.is_contiguous() or not a.is_contiguous():
        raise ValueError("dense LoRA-A expects contiguous operands")
    if int(u_drop_cpu.shape[1]) != int(a.shape[1]):
        raise ValueError(f"dense LoRA-A shape mismatch: {tuple(u_drop_cpu.shape)} vs {tuple(a.shape)}")
    if backend == "asym" and a.device.type != "cuda":
        raise ValueError(f"dense LoRA-A AsymGEMM path expects CUDA LoRA-A weights, got {a.device}")

    if int(u_drop_cpu.shape[0]) == 0:
        return torch.empty((0, int(a.shape[0])), device=a.device, dtype=a.dtype)

    if backend == "torch":
        u_stage = u_drop_cpu.to(device=a.device, dtype=a.dtype, non_blocking=u_drop_cpu.is_pinned())
        out = u_stage @ a.t()
        _record_attn_hbm_gemm(stats, tag)
    else:
        offsets, experts = _single_group_offsets_experts(a.device, int(u_drop_cpu.shape[0]))
        out = grouped_expert_lora_cpu_left(
            u_drop_cpu,
            a.unsqueeze(0).contiguous(),
            offsets,
            experts,
            output_dtype=a.dtype,
            stats=stats,
        )
    if stats is not None:
        stats.attn_act_lora_a_forward_calls += 1
    return out


class _AsymActivationOffloadLoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        base_layer: AsymFrozenLinear,
        scaling: float,
        lora_dropout_p: float,
        training: bool,
        lora_dtype: torch.dtype,
        projection_role: str,
        stats: AsymExecutionStats | None,
        backend: str,
        snapshot: dict[str, Any] | None,
        attention_context: AttentionActivationOffloadContext | None,
        module: "AsymActivationOffloadLoRALinear | None" = None,
    ) -> torch.Tensor:
        _check_backend(backend)
        weight_offload_module = module if module is not None and getattr(module, "_weight_offload", None) is not None else None
        if weight_offload_module is not None:
            weight_offload_module.gather_lora_weights()
            a = weight_offload_module.lora_a
            b = weight_offload_module.lora_b
        if base_layer.precision != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 base weights")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("attention activation offload expects BF16 LoRA weights")
        if float(lora_dropout_p) != 0.0:
            raise NotImplementedError("attention activation offload dropout is not implemented yet")
        if x.dim() < 1:
            raise ValueError("attention activation offload input must have at least one dimension")

        flat, input_shape = _flatten_last_dim(x, base_layer.in_features)
        flat_lora = flat.to(dtype=lora_dtype).contiguous()
        if flat_lora.dtype != torch.bfloat16:
            raise ValueError("attention activation offload currently requires BF16 activation math")

        base = asym_bf16_cpu_right_matmul(
            flat_lora,
            base_layer.host_weight.weight,
            backend=base_layer.backend,
            stats=stats,
            phase="forward",
            tag=f"{projection_role}.base_forward",
            compiled_dims=base_layer.compiled_dims,
            output_dtype=base_layer.bf16_output_dtype,
        )
        if base_layer.bias_cpu is not None:
            base = base + base_layer.bias_cpu.to(device=base.device, dtype=base.dtype, non_blocking=base_layer.bias_cpu.is_pinned())

        manager = ActivationOffloadManager(pin_memory=True)
        shared_source = None
        keep_acts_hbm = _attn_keep_acts_hbm_enabled()
        if keep_acts_hbm:
            # S-mem: no U offload, no wait — LoRA-A directly on the HBM-resident source.
            u_handle = None
            _record_attn_hbm_gemm(stats, f"{projection_role}.lora_a_forward")
            s = (flat_lora @ a.contiguous().t()).contiguous()
            if stats is not None:
                stats.attn_act_lora_a_forward_calls += 1
        else:
            if attention_context is None:
                u_handle = manager.offload(flat_lora, f"{projection_role}.U")
            else:
                shared_source = attention_context.acquire_source(x, flat_lora, projection_role)
                u_handle = shared_source.handle
            # _dense_lora_a_cpu_left host-pads u_handle.tensor whenever M % 128 != 0
            # (flagship 45k×8 → M=360000, not a multiple of 128) — a HOST read of a
            # buffer just filled by a non-blocking D2H. Wait on the manager that OWNS
            # the handle's ready event: the shared q/k/v source was offloaded by the
            # attention context's manager, not this call's fresh one.
            (attention_context.manager if shared_source is not None else manager).wait_cpu_ready_host(u_handle)
            s = None
            if (
                shared_source is not None
                and attention_context is not None
                and backend == "asym"
                and _attn_qkv_fwd_shared_enabled()
            ):
                s = attention_context.shared_lora_a_forward(
                    shared_source, projection_role, a, stats=stats, backend=backend
                )
            elif attention_context is not None:
                attention_context._shared_diag(
                    f"hook-skipped:{projection_role} shared={shared_source is not None} "
                    f"backend={backend} env={_attn_qkv_fwd_shared_enabled()}"
                )
            if s is None:
                s = _dense_lora_a_cpu_left(
                    u_handle.tensor,
                    a.contiguous(),
                    stats=stats,
                    tag=f"{projection_role}.lora_a_forward",
                    backend=backend,
                )
        _record_attn_hbm_gemm(stats, f"{projection_role}.lora_b_forward")
        out = base
        _add_matmul_rows_(out, s, b.t(), scale=float(scaling))
        dual_da = (
            not keep_acts_hbm
            and backend == "asym"
            and int(a.shape[0]) == 64
            and _attn_dual_da_enabled()
        )
        if keep_acts_hbm:
            s_handle = None
            ctx._ka_u = flat_lora
            ctx._ka_s = s
        elif dual_da:
            # N2: S is NOT offloaded — backward regenerates it from the same
            # X pass that computes dA (dual-dataflow kernel).
            s_handle = None
            ctx._ka_u = None
            ctx._ka_s = None
        else:
            s_handle = manager.offload(s.contiguous(), f"{projection_role}.S")
            ctx._ka_u = None
            ctx._ka_s = None
        ctx.dual_da = dual_da
        ctx.keep_acts_hbm = keep_acts_hbm
        _update_snapshot(snapshot, manager, attention_context)

        if weight_offload_module is None:
            ctx.save_for_backward(a, b)
        else:
            ctx.save_for_backward()
        ctx.weight_offload_module = weight_offload_module
        ctx.manager = manager
        ctx.u_handle = u_handle
        ctx.shared_source = shared_source
        ctx.s_handle = s_handle
        ctx.base_layer = base_layer
        ctx.input_shape = input_shape
        ctx.input_dtype = x.dtype
        ctx.scaling = float(scaling)
        ctx.projection_role = projection_role
        ctx.stats = stats
        ctx.backend = backend
        ctx.lora_dropout_p = float(lora_dropout_p)
        ctx.snapshot = snapshot
        ctx.attention_context = attention_context
        if weight_offload_module is not None and bool(getattr(weight_offload_module, "_weight_offload_release_after_forward", True)):
            weight_offload_module.release_lora_weights()
        return _restore_last_dim(out, input_shape, base_layer.out_features)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None, None, None, None, None, None]:
        if ctx.lora_dropout_p != 0.0:
            raise NotImplementedError("attention activation offload dropout is not implemented yet")

        if getattr(ctx, "weight_offload_module", None) is not None:
            module: AsymActivationOffloadLoRALinear = ctx.weight_offload_module
            module.gather_lora_weights()
            a = module.lora_a
            b = module.lora_b
        else:
            a, b = ctx.saved_tensors
        manager: ActivationOffloadManager = ctx.manager
        u_handle: CPUActivationHandle = ctx.u_handle
        s_handle: CPUActivationHandle = ctx.s_handle
        base_layer: AsymFrozenLinear = ctx.base_layer
        stats: AsymExecutionStats | None = ctx.stats
        role = str(ctx.projection_role)

        grad_x = grad_a = grad_b = None
        s_stage = None
        deposited_u = False
        try:
            d_y = grad_output.reshape(-1, base_layer.out_features).to(dtype=torch.bfloat16).contiguous()
            needs_grad_x = bool(ctx.needs_input_grad[0])
            needs_grad_a = bool(ctx.needs_input_grad[1])
            needs_grad_b = bool(ctx.needs_input_grad[2])
            needs_low_rank = needs_grad_x or needs_grad_a or needs_grad_b

            d_s = None
            if needs_low_rank:
                _record_attn_hbm_gemm(stats, f"{role}.dS")
                d_s = (d_y @ b).to(dtype=torch.bfloat16) * float(ctx.scaling)

            if needs_grad_x:
                d_u = asym_bf16_cpu_right_matmul(
                    d_y,
                    base_layer.host_weight.weight,
                    transpose_b=True,
                    backend=base_layer.backend,
                    stats=stats,
                    phase="attn_act_base_dx",
                    tag=f"{role}.base_dx",
                    compiled_dims=base_layer.compiled_dims,
                    output_dtype=torch.bfloat16,
                )
                if d_s is not None:
                    _record_attn_hbm_gemm(stats, f"{role}.lora_input_grad")
                    _add_matmul_rows_(d_u, d_s, a)
                grad_x = d_u.to(dtype=ctx.input_dtype).reshape(ctx.input_shape)

            if needs_grad_a:
                if d_s is None:
                    raise RuntimeError("internal error: dS was not computed for dA")
                if getattr(ctx, "keep_acts_hbm", False):
                    # S-mem: dA from the HBM-resident source — native GEMM, no padding,
                    # no stage-back (kills the raw H2D that mirrored the fwd D2H).
                    if int(d_s.shape[0]) == 0:
                        grad_a = torch.zeros_like(a)
                    else:
                        _record_attn_hbm_gemm(stats, f"{role}.dA")
                        grad_a = (d_s.t().contiguous() @ ctx._ka_u).to(dtype=a.dtype)
                elif getattr(ctx, "dual_da", False):
                    # N2 dual-dataflow: one X pass -> dA AND S. Replaces the
                    # upstream-form dA matmul + the S re-stage for dB.
                    import asym_gemm as _ag

                    (ctx.attention_context.manager if ctx.shared_source is not None else manager).wait_cpu_ready_host(u_handle)
                    m_rows = int(d_s.shape[0])
                    rank = int(a.shape[0])
                    offs, exps = _single_group_offsets_experts(d_s.device, m_rows)
                    dual_s = torch.zeros((m_rows, rank), device=d_s.device, dtype=torch.float32)
                    grad_a3 = torch.empty((1, rank, int(a.shape[1])), device=d_s.device, dtype=a.dtype)
                    _ag.sm100_grouped_lora_a_dual_bf16_cpu_right(
                        d_s.contiguous(),
                        u_handle.tensor,
                        a.detach().unsqueeze(0).contiguous(),
                        dual_s,
                        grad_a3,
                        offs,
                        exps,
                        2,
                    )
                    grad_a = grad_a3[0]
                    ctx._dual_s = dual_s
                else:
                    # KA off ⇒ U is an offloaded pinned handle; the CPU deposit may
                    # claim the wgrad (K-2), else the legacy padded CPU-right kernel.
                    if _attn_lora_a_grad_cpu_deposit_enabled():
                        _sweep_attn_deposit_releases()
                        grad_a = _try_deposit_attn_lora_a_grad(
                            a, d_s, u_handle, manager, ctx.shared_source, role
                        )
                        deposited_u = grad_a is not None
                    m_grad = _align_up(int(d_s.shape[0]), 64)
                    if grad_a is not None:
                        pass
                    elif m_grad == 0:
                        grad_a = torch.zeros_like(a)
                    else:
                        # _pad_cpu_rows_to is a HOST memcpy of u_handle.tensor; nothing
                        # upstream host-orders it after the D2H that filled the handle
                        # (fix_merged.md V1 — race proven in vitro 2026-07-16). Shared
                        # q/k/v sources were offloaded by the attention context's
                        # manager — that map holds the ready event, not ctx.manager's.
                        (ctx.attention_context.manager if ctx.shared_source is not None else manager).wait_cpu_ready_host(u_handle)
                        u_source = _pad_cpu_rows_to(u_handle.tensor, m_grad)
                        d_s_rows = _pad_hbm_rows_to(d_s, m_grad)
                        d_s_t = d_s_rows.t().contiguous()
                        grad_a = asym_bf16_cpu_right_matmul(
                            d_s_t,
                            u_source,
                            transpose_b=True,
                            backend=ctx.backend,
                            stats=stats,
                            phase="attn_act_dA",
                            tag=f"{role}.dA",
                            compiled_dims=base_layer.compiled_dims,
                            output_dtype=a.dtype,
                        ).to(dtype=a.dtype)

            if needs_grad_b:
                if getattr(ctx, "keep_acts_hbm", False):
                    _record_attn_hbm_gemm(stats, f"{role}.dB")
                    grad_b = ((d_y.t().contiguous() @ ctx._ka_s).to(dtype=b.dtype) * float(ctx.scaling)).to(dtype=b.dtype)
                elif getattr(ctx, "_dual_s", None) is not None:
                    _record_attn_hbm_gemm(stats, f"{role}.dB")
                    grad_b = ((d_y.t().contiguous() @ ctx._dual_s.to(torch.bfloat16)).to(dtype=b.dtype) * float(ctx.scaling)).to(dtype=b.dtype)
                    ctx._dual_s = None
                elif s_handle is None and getattr(ctx, "dual_da", False):
                    # grad_b-only edge under dual mode: S was never offloaded;
                    # regenerate it with one K1 pass.
                    (ctx.attention_context.manager if ctx.shared_source is not None else manager).wait_cpu_ready_host(u_handle)
                    s_re = _dense_lora_a_cpu_left(
                        u_handle.tensor, a.detach().contiguous(), stats=stats,
                        tag=f"{role}.S_regen", backend=ctx.backend)
                    _record_attn_hbm_gemm(stats, f"{role}.dB")
                    grad_b = ((d_y.t().contiguous() @ s_re).to(dtype=b.dtype) * float(ctx.scaling)).to(dtype=b.dtype)
                else:
                    if stats is not None:
                        stats.attn_act_stage_low_rank_calls += 1
                    s_stage = manager.stage(s_handle, tag=f"{role}.S_stage")
                    _record_attn_hbm_gemm(stats, f"{role}.dB")
                    grad_b = ((d_y.t().contiguous() @ s_stage).to(dtype=b.dtype) * float(ctx.scaling)).to(dtype=b.dtype)
        finally:
            if s_stage is not None:
                manager.release_stage(s_stage)
            if s_handle is not None:
                manager.release_cpu(s_handle)
            if deposited_u:
                # U/shared-source release deferred to the worker sweep (K-2) —
                # _try_deposit took ownership of both.
                pass
            elif ctx.shared_source is not None:
                ctx.shared_source.release()
            elif u_handle is not None:
                manager.release_cpu(u_handle)
            ctx._ka_u = None
            ctx._ka_s = None
            _update_snapshot(ctx.snapshot, manager, ctx.attention_context)

        return grad_x, grad_a, grad_b, None, None, None, None, None, None, None, None, None, None, None


class AsymActivationOffloadLoRALinear(nn.Module):
    """Dense attention LoRA linear that offloads forward activations to CPU."""

    def __init__(
        self,
        source: HostWeight | torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
        projection_role: str = "attention",
        attention_context: AttentionActivationOffloadContext | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        if str(precision).lower() != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 precision")
        _check_backend(backend)
        if isinstance(source, HostWeight):
            host_weight = source
        elif isinstance(source, torch.Tensor):
            host_weight = HostWeight.from_tensor(source, dtype=source.dtype, pin_memory=True)
        else:
            raise TypeError(f"source must be a HostWeight or torch.Tensor, got {type(source)!r}")
        self._init_from_host_weight(
            host_weight,
            bias=bias,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            lora_generator=lora_generator,
            lora_dtype=lora_dtype,
            precision=precision,
            adapter_name=adapter_name,
            init_lora_weights=init_lora_weights,
            lora_dropout=lora_dropout,
            projection_role=projection_role,
            attention_context=attention_context,
        )

    @classmethod
    def from_host_weight(
        cls,
        host_weight: HostWeight,
        *,
        bias: torch.Tensor | None = None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None = None,
        device: torch.device | None = None,
        lora_generator: torch.Generator | None = None,
        lora_dtype: torch.dtype | str | None = torch.bfloat16,
        precision: str = "bf16",
        adapter_name: str = "default",
        init_lora_weights: Literal["asym", "peft"] = "asym",
        lora_dropout: float = 0.0,
        projection_role: str = "attention",
        attention_context: AttentionActivationOffloadContext | None = None,
    ) -> "AsymActivationOffloadLoRALinear":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._init_from_host_weight(
            host_weight,
            bias=bias,
            rank=rank,
            alpha=alpha,
            backend=backend,
            stats=stats,
            device=device,
            lora_generator=lora_generator,
            lora_dtype=lora_dtype,
            precision=precision,
            adapter_name=adapter_name,
            init_lora_weights=init_lora_weights,
            lora_dropout=lora_dropout,
            projection_role=projection_role,
            attention_context=attention_context,
        )
        return obj

    def _init_from_host_weight(
        self,
        host_weight: HostWeight,
        *,
        bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        backend: Literal["asym", "torch"],
        stats: AsymExecutionStats | None,
        device: torch.device | None,
        lora_generator: torch.Generator | None,
        lora_dtype: torch.dtype | str | None,
        precision: str,
        adapter_name: str,
        init_lora_weights: Literal["asym", "peft"],
        lora_dropout: float,
        projection_role: str,
        attention_context: AttentionActivationOffloadContext | None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= float(lora_dropout) <= 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1], got {lora_dropout}")
        if str(precision).lower() != "bf16":
            raise NotImplementedError("attention activation offload currently supports only BF16 precision")
        _check_backend(backend)
        resolved_device = torch.device("cpu" if device is None else device)
        resolved_lora_dtype = normalize_lora_dtype(lora_dtype)
        if resolved_lora_dtype != torch.bfloat16:
            raise ValueError("attention activation offload currently requires BF16 LoRA weights")
        self.base_layer = AsymFrozenLinear.from_host_weight(
            host_weight,
            bias=bias,
            backend=backend,
            stats=stats,
            precision="bf16",
            bf16_output_dtype=torch.bfloat16,
        )
        self.lora_A = nn.ModuleDict(
            {
                adapter_name: nn.Linear(
                    host_weight.in_features,
                    rank,
                    bias=False,
                    device=resolved_device,
                    dtype=resolved_lora_dtype,
                )
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                adapter_name: nn.Linear(
                    rank,
                    host_weight.out_features,
                    bias=False,
                    device=resolved_device,
                    dtype=resolved_lora_dtype,
                )
            }
        )
        self.active_adapter = adapter_name
        self.lora_dtype = resolved_lora_dtype
        self.scaling = float(alpha) / float(rank)
        self.precision = "bf16"
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if float(lora_dropout) > 0.0 else nn.Identity()
        self.projection_role = str(projection_role)
        self.attention_context = attention_context
        if attention_context is not None and self.projection_role in _QKV_SHARE_ROLES:
            attention_context.register_lora_module(self.projection_role, self)
        self._last_activation_offload_stats: dict[str, Any] = {}
        self._weight_offload = None
        object.__setattr__(self, "_weight_offload_owner", self)
        self._weight_offload_release_after_forward = True
        self._reset_lora(adapter_name, lora_generator, init_lora_weights=init_lora_weights)

    @property
    def base(self) -> AsymFrozenLinear:
        return self.base_layer

    @property
    def lora_a(self) -> torch.nn.Parameter:
        return self.lora_A[self.active_adapter].weight

    @property
    def lora_b(self) -> torch.nn.Parameter:
        return self.lora_B[self.active_adapter].weight

    @property
    def pinned_cpu_bytes(self) -> int:
        return self.base_layer.pinned_cpu_bytes

    @property
    def cpu_resident_base_weight_bytes(self) -> int:
        return self.base_layer.weight_hbm_saved_bytes

    @property
    def gpu_resident_base_weight_bytes(self) -> int:
        return 0

    def _reset_lora(
        self,
        adapter_name: str,
        generator: torch.Generator | None,
        *,
        init_lora_weights: Literal["asym", "peft"],
    ) -> None:
        with torch.no_grad():
            _reset_lora_weights(
                self.lora_A[adapter_name].weight,
                self.lora_B[adapter_name].weight,
                init_lora_weights=init_lora_weights,
                generator=generator,
            )

    def gather_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.gather_group(getattr(self, "_weight_offload_owner", self))

    def release_lora_weights(self) -> None:
        coordinator = getattr(self, "_weight_offload", None)
        if coordinator is not None:
            coordinator.release_group(getattr(self, "_weight_offload_owner", self))

    def _plain_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_offload_enabled = getattr(self, "_weight_offload", None) is not None
        if weight_offload_enabled:
            self.gather_lora_weights()
        try:
            base = self.base_layer(x)
            lora_x = self.lora_dropout(x).to(dtype=self.lora_dtype)
            delta = self.lora_B[self.active_adapter](self.lora_A[self.active_adapter](lora_x))
            return base + (delta * float(self.scaling)).to(dtype=base.dtype)
        finally:
            if weight_offload_enabled and bool(getattr(self, "_weight_offload_release_after_forward", True)):
                self.release_lora_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and torch.is_grad_enabled()):
            return self._plain_forward(x)
        return _AsymActivationOffloadLoRALinearFunction.apply(
            x,
            self.lora_A[self.active_adapter].weight,
            self.lora_B[self.active_adapter].weight,
            self.base_layer,
            self.scaling,
            self.lora_dropout_p,
            self.training,
            self.lora_dtype,
            self.projection_role,
            self.base_layer.stats,
            self.base_layer.backend,
            self._last_activation_offload_stats,
            self.attention_context,
            self,
        )


__all__ = [
    "AsymActivationOffloadLoRALinear",
    "AttentionActivationOffloadContext",
    "AttentionSavedTensorOffloadWrapper",
    "attention_saved_tensor_offload_module_names",
    "install_attention_saved_tensor_offload",
    "is_attention_saved_tensor_offload_wrapper",
    "_dense_lora_a_cpu_left",
    "_single_group_offsets_experts",
]
