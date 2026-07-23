"""fix_cpu_compute.md item 6 — q/k-norm (+rope) recompute-instead-of-save.

The measured save class (smoke5/smoke6 leaves, `offload_bytes_by_tag`): every
q/k-norm call saves TWO distinct fp32 [B,S,H,D] upcasts through the attention
saved-tensor pack (`AsymFrozenRMSNorm.forward` calls ``x.float()`` twice: pow
saves one, mul saves the other — the smoke6 "same-shape-diff-storage" proof)
plus the fp32 rsqrt pair. That is ~400 GB/step @30B·32k×b8 and ~1.6 TB/step
@128k of D2H+H2D pinned traffic, proven NOT dedupable (item 2) — collectible
only by recompute.

Mechanism (this module): wrap the q/k-norm modules' forward in a custom
``torch.autograd.Function`` that
  * computes the forward output under ``no_grad`` with the module's ORIGINAL
    forward (so no fp32 intermediate is ever saved),
  * saves the **bf16 input once** through the POOLED pinned offload
    (``ActivationOffloadManager`` — pooled/bucketed pinned buffers, async D2H
    + ready event),
  * at backward: restages the bf16 input (side-stream H2D, event-ordered,
    fresh transient stage buffer — no persistent stage cache: G@128k runs at
    ~180/186 GiB), recomputes the forward **with the same original forward on
    the same device under ``enable_grad``**, and backprops through the local
    graph. Same bf16 input + same math on the same device ⇒ the recomputed
    intermediates and therefore the gradients are BIT-IDENTICAL to the
    unwrapped graph (unit gate: exact equality).

Saved-tensor packed lists are never touched (the item-3 failure mode): the
wrapper adds a normal module-boundary Function; everything else in autograd
is unchanged.

CPU worker path (`ASYMM_QKNORM_RECOMPUTE_CPU`): the shipped
``cpu_rmsnorm_bf16`` kernel is FORWARD-only (bf16 out, ≤1 ulp). The norm
BACKWARD needs the exact fp32 chain (upcast/variance/rsqrt intermediates) on
the GPU, which the forward-only kernel cannot provide, and in this graph the
norm's recomputed OUTPUT has no other backward consumer (rope saves nothing:
cos/sin are frozen so the rope muls save no rotated operands — verified in the
smoke leaves). The flag is therefore accepted but resolves to the GPU
recompute with a one-line notice; a CPU path needs an rmsnorm-BACKWARD kernel
first (recorded as the K-7 follow-up).

Rope-recompute variant (`ASYMM_ROPE_RECOMPUTE`): same Function pattern over
``apply_rotary_pos_emb`` (save the bf16 q/k inputs once, recompute the
rotation exactly at backward). Built + parity-gated for completeness, but
DEFAULT OFF and NOT part of the production stack: in this graph rope saves
nothing (frozen cos/sin ⇒ backward is linear with constant coefficients), so
the variant can only ADD saved bytes; it applies to graphs whose rope
operands ARE saved (e.g. trainable rotary scales).

Flags (default OFF, gate-arm pattern like ASYMM_SAVE_ON_CPU_DEDUP):
  ASYMM_QKNORM_RECOMPUTE=1      arm the norm wrapper (env wins; else policy
                                 rule P12.qknorm_recompute, False until gated)
  ASYMM_ROPE_RECOMPUTE=1        arm the rope wrapper (evidence-off variant)
  ASYMM_QKNORM_RECOMPUTE_CPU=1  accepted; resolves to GPU recompute (see above)
"""

from __future__ import annotations

import os
import sys
import threading
import types
from typing import Any, Optional

import torch
from torch import nn

from .activation_offload import ActivationOffloadManager, CPUActivationHandle

__all__ = [
    "enabled",
    "rope_enabled",
    "install_qknorm_recompute",
    "install_rope_recompute",
    "qknorm_recompute_stats",
    "reset_stats_for_tests",
]

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_on(*names: str) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip().lower() in _TRUTHY:
            return True
    return False


def enabled() -> bool:
    """Gate-arm pattern (save_dedup precedent): the explicit env flag arms the
    feature even under the policy; otherwise the policy rule decides (False
    until the item-6 gates pass)."""
    if _env_on("ASYMM_QKNORM_RECOMPUTE", "ASYM_GEMM_LF_CONFIG_ASYMM_QKNORM_RECOMPUTE"):
        return True
    try:
        from . import placement_policy

        return placement_policy.enabled() and placement_policy.qknorm_recompute()
    except Exception:
        return False


def rope_enabled(tokens: "int | None" = None) -> bool:
    """P2 per-call gate: env force-arms; else the policy decides (dense-class True;
    MoE by per-call tokens >= ASYM_POLICY_ROPE_MIN_TOKENS — C-binding regimes only)."""
    if _env_on("ASYMM_ROPE_RECOMPUTE", "ASYM_GEMM_LF_CONFIG_ASYMM_ROPE_RECOMPUTE"):
        return True
    try:
        from . import placement_policy

        return placement_policy.enabled() and placement_policy.rope_recompute(tokens)
    except Exception:
        return False


def _rope_possible() -> bool:
    """Cheap superset check used at norm-forward time (attaching the src attribute
    costs nothing; the real per-call decision happens in the rope wrapper)."""
    if _env_on("ASYMM_ROPE_RECOMPUTE", "ASYM_GEMM_LF_CONFIG_ASYMM_ROPE_RECOMPUTE"):
        return True
    try:
        from . import placement_policy

        return placement_policy.enabled()
    except Exception:
        return False


def _cpu_path_requested() -> bool:
    return _env_on(
        "ASYMM_QKNORM_RECOMPUTE_CPU", "ASYM_GEMM_LF_CONFIG_ASYMM_QKNORM_RECOMPUTE_CPU"
    )


# ---------------------------------------------------------------------------
# shared pooled pinned offload + counters (main-thread only: pool rule)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_MANAGER: Optional[ActivationOffloadManager] = None
_ENGAGED_PRINTED = False
_CPU_NOTICE_PRINTED = False
_ROPE_ENGAGED_PRINTED = False
_COUNTERS = {
    "norm_offloads": 0,
    "norm_offload_bytes": 0,
    "norm_recomputes": 0,
    "norm_recompute_bytes_restaged": 0,
    "rope_offloads": 0,
    "rope_offload_bytes": 0,
    "rope_recomputes": 0,
    "rope_recompute_bytes_rebuilt": 0,
}


def _manager() -> ActivationOffloadManager:
    global _MANAGER
    if _MANAGER is None:
        with _LOCK:
            if _MANAGER is None:
                _MANAGER = ActivationOffloadManager(pin_memory=True)
    return _MANAGER


def qknorm_recompute_stats() -> dict[str, Any]:
    """P11 counters for the profile's placement_policy block."""
    with _LOCK:
        out: dict[str, Any] = {
            "enabled": enabled(),
            "rope_enabled": rope_enabled(),
            "cpu_path_requested": _cpu_path_requested(),
            **dict(_COUNTERS),
        }
    if _MANAGER is not None:
        out["offload_bytes_by_tag"] = dict(_MANAGER.stats.offload_bytes_by_tag)
        out["cpu_peak_bytes_live"] = _MANAGER.stats.cpu_peak_bytes_live
    return out


def reset_stats_for_tests() -> None:
    global _MANAGER, _ENGAGED_PRINTED, _ROPE_ENGAGED_PRINTED, _CPU_NOTICE_PRINTED
    with _LOCK:
        _MANAGER = None
        _ENGAGED_PRINTED = False
        _ROPE_ENGAGED_PRINTED = False
        _CPU_NOTICE_PRINTED = False
        for key in _COUNTERS:
            _COUNTERS[key] = 0


def _count(key: str, value: int = 1) -> None:
    with _LOCK:
        _COUNTERS[key] += value


def _engaged_once(role: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    global _ENGAGED_PRINTED
    if _ENGAGED_PRINTED:
        return
    _ENGAGED_PRINTED = True
    print(
        f"[asym-qknorm-recompute] qk-norm recompute ENGAGED "
        f"(first={role}, shape={tuple(shape)}, dtype={str(dtype).replace('torch.', '')})",
        file=sys.stderr,
        flush=True,
    )


def _cpu_notice_once() -> None:
    global _CPU_NOTICE_PRINTED
    if _CPU_NOTICE_PRINTED:
        return
    _CPU_NOTICE_PRINTED = True
    print(
        "[asym-qknorm-recompute] ASYMM_QKNORM_RECOMPUTE_CPU requested but the shipped "
        "cpu_rmsnorm_bf16 kernel is forward-only (no backward/rstd) — the exact fp32 "
        "chain must be rebuilt on the GPU for bit-identical gradients; using the GPU "
        "recompute (a CPU path needs an rmsnorm-backward kernel first).",
        file=sys.stderr,
        flush=True,
    )


def _rope_engaged_once(shape: tuple[int, ...]) -> None:
    global _ROPE_ENGAGED_PRINTED
    if _ROPE_ENGAGED_PRINTED:
        return
    _ROPE_ENGAGED_PRINTED = True
    print(
        f"[asym-qknorm-recompute] rope recompute ENGAGED (q shape={tuple(shape)})",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# staging helper — fresh transient stage (the pack `_unpack` pattern): side
# stream H2D, compute stream waits on the EVENT, buffer comes from (and
# returns to) the CUDA caching allocator so the backward's transient G high
# water matches today's unpack behaviour instead of pinning a persistent
# per-shape stage cache.
# ---------------------------------------------------------------------------


def _stage_fresh(handle: CPUActivationHandle, manager: ActivationOffloadManager) -> torch.Tensor:
    device = handle.original_device
    staged = torch.empty(handle.original_shape, device=device, dtype=handle.original_dtype)
    if handle.tensor.is_pinned() and device.type == "cuda":
        from .attention_activation_offload import _h2d_restage_stream

        compute_stream = torch.cuda.current_stream(device)
        manager.wait_cpu_ready(handle)  # orders the side copy behind the producing D2H
        side = _h2d_restage_stream(device)
        side.wait_stream(compute_stream)  # staged alloc ordering
        with torch.no_grad(), torch.cuda.stream(side):
            staged.copy_(handle.tensor, non_blocking=True)
        done = torch.cuda.Event()
        done.record(side)
        from .activation_offload import restage_gap_commit, restage_gap_events

        gap_wait, gap_done = restage_gap_events(device)
        if gap_wait is not None:
            gap_wait.record(compute_stream)  # R5: compute-stream arrival before the wait
        compute_stream.wait_event(done)
        if gap_done is not None:
            gap_done.record(side)
            restage_gap_commit(gap_wait, gap_done, handle.nbytes, f"qknorm.{handle.tag}")
        # NB deliberately NO staged.record_stream(side): compute_stream.wait_event(done)
        # above already orders every downstream use AND the eventual free behind the
        # side-stream copy. record_stream(side) additionally tagged each freed restage
        # buffer with a side-stream dependency, which segregated those bytes from the
        # compute-stream pool — the caching allocator then served autograd transients
        # from fresh segments (+10.98 GiB reserved at 176.7 vs 165.7, S6 M7; allocated
        # peak byte-identical 109.54 in the A/B — pure pool growth, not a hold).
        # keep the pinned source alive until the staged tensor dies (async copy source)
        staged._asym_qknorm_keepalive = handle.tensor  # type: ignore[attr-defined]
    else:
        import time as _t

        manager.wait_cpu_ready(handle)
        _t0 = _t.perf_counter()
        with torch.no_grad():
            staged.copy_(handle.tensor, non_blocking=handle.tensor.is_pinned())
        from .activation_offload import restage_gap_host_ms

        restage_gap_host_ms(f"qknorm.{handle.tag}", (_t.perf_counter() - _t0) * 1000.0, handle.nbytes)
    return staged


def _autocast_state() -> tuple[bool, Optional[torch.dtype]]:
    try:
        if torch.is_autocast_enabled("cuda"):
            return True, torch.get_autocast_dtype("cuda")
    except Exception:
        pass
    return False, None


class _autocast_replay:
    """Reproduce the forward-time autocast state around the backward recompute
    (the norm math is not autocast-listed, but exactness must not depend on
    that detail — mirror what torch.utils.checkpoint does)."""

    def __init__(self, state: tuple[bool, Optional[torch.dtype]]) -> None:
        self._enabled, self._dtype = state
        self._cm = None

    def __enter__(self):
        if self._enabled and self._dtype is not None:
            self._cm = torch.autocast(device_type="cuda", enabled=True, dtype=self._dtype)
            self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False


# ---------------------------------------------------------------------------
# the norm Function
# ---------------------------------------------------------------------------


class _SharedXHandle:
    """Refcounted owner of the offloaded bf16 norm input (P2: the handle now has
    up to two consumer classes — the norm backward and any rope/SDPA-operand
    recompute recipes chained off the norm output). release() is idempotent-safe
    per consumer; the pinned buffer returns to the pool at refcount 0."""

    __slots__ = ("manager", "handle", "x_shape", "_refs")

    def __init__(self, manager: ActivationOffloadManager, handle: CPUActivationHandle, x_shape: tuple) -> None:
        self.manager = manager
        self.handle = handle
        self.x_shape = x_shape
        self._refs = 1

    def retain(self) -> "_SharedXHandle":
        self._refs += 1
        return self

    def release(self) -> None:
        self._refs -= 1
        if self._refs == 0:
            self.manager.release_cpu(self.handle)


class _QKNormRecomputeFunction(torch.autograd.Function):
    """Save the bf16 norm input once (pooled pinned offload); recompute the
    exact forward at backward and backprop through the local graph."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, module: nn.Module) -> torch.Tensor:
        orig = module._asym_qknorm_recompute_original_forward  # type: ignore[attr-defined]
        with torch.no_grad():
            out = orig(x)
        role = getattr(module, "_asym_qknorm_recompute_role", "norm")
        flat = x.detach()
        if not flat.is_contiguous():
            flat = flat.contiguous()
        rows = int(flat.shape[0] * flat.shape[1]) if flat.dim() >= 3 else int(flat.shape[0])
        flat2d = flat.reshape(rows, -1)
        manager = _manager()
        handle = manager.offload(flat2d, f"qknorm.{role}.x")
        _count("norm_offloads")
        _count("norm_offload_bytes", handle.nbytes)
        _engaged_once(role, tuple(x.shape), x.dtype)
        shared = _SharedXHandle(manager, handle, tuple(int(d) for d in x.shape))
        ctx.shared = shared
        ctx.module = module
        ctx.x_shape = shared.x_shape
        ctx.autocast_state = _autocast_state()
        ctx.weight_requires_grad = bool(weight is not None and weight.requires_grad)
        if _rope_possible():
            # P2: expose the recompute source on the OUTPUT object so the rope
            # wrapper can chain SDPA-operand recipes off it (explicit object-
            # attribute chaining — never anonymous matching).
            out._asym_qknorm_src = (shared, module)  # type: ignore[attr-defined]
        if _cpu_path_requested():
            _cpu_notice_once()
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        module: nn.Module = ctx.module
        orig = module._asym_qknorm_recompute_original_forward  # type: ignore[attr-defined]
        manager = _manager()
        shared: _SharedXHandle = ctx.shared
        ctx.shared = None
        handle = shared.handle
        grad_x = grad_w = None
        try:
            staged = _stage_fresh(handle, manager)
            _count("norm_recomputes")
            _count("norm_recompute_bytes_restaged", handle.nbytes)
            x_dev = staged.view(ctx.x_shape)
            weight = getattr(module, "weight", None)
            with torch.enable_grad(), _autocast_replay(ctx.autocast_state):
                x_leaf = x_dev.detach().requires_grad_(True)
                inputs = [x_leaf]
                # rare (norm weights are frozen in LoRA SFT): the module forward reads
                # its own weight, which is already an autograd leaf — the recomputed
                # local graph links to it directly, so its exact grad falls out of the
                # same torch.autograd.grad call (returned, NOT accumulated into .grad).
                if ctx.weight_requires_grad and isinstance(weight, torch.Tensor):
                    inputs.append(weight)
                out = orig(x_leaf)
                grads = torch.autograd.grad(out, inputs, grad_out)
            grad_x = grads[0]
            if len(grads) > 1:
                grad_w = grads[1]
        finally:
            shared.release()
        return grad_x, grad_w, None


def _qknorm_recompute_forward(module: nn.Module, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
    orig = module._asym_qknorm_recompute_original_forward  # type: ignore[attr-defined]
    if args or kwargs:  # gated / non-standard call signatures: out of scope, passthrough
        return orig(hidden_states, *args, **kwargs)
    if (
        not enabled()
        or not torch.is_grad_enabled()
        or not isinstance(hidden_states, torch.Tensor)
        or not hidden_states.requires_grad
        or hidden_states.device.type != "cuda"
        or hidden_states.dim() < 2
        or hidden_states.dtype not in (torch.bfloat16, torch.float16)
    ):
        return orig(hidden_states)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return orig(hidden_states)
    return _QKNormRecomputeFunction.apply(hidden_states, weight, module)


def install_qknorm_recompute(attention_module: nn.Module) -> int:
    """Wrap `attention_module.{q_norm,k_norm}.forward` (idempotent). The wrapper
    is a pure passthrough until `enabled()`; install is therefore always safe."""
    wrapped = 0
    for role in ("q_norm", "k_norm"):
        norm = getattr(attention_module, role, None)
        if not isinstance(norm, nn.Module):
            continue
        if getattr(norm, "_asym_qknorm_recompute_installed", False):
            wrapped += 1
            continue
        norm._asym_qknorm_recompute_original_forward = norm.forward  # type: ignore[attr-defined]
        norm._asym_qknorm_recompute_role = role  # type: ignore[attr-defined]
        norm.forward = types.MethodType(_qknorm_recompute_forward, norm)  # type: ignore[method-assign]
        norm._asym_qknorm_recompute_installed = True  # type: ignore[attr-defined]
        wrapped += 1
    return wrapped


# ---------------------------------------------------------------------------
# P2 rope/SDPA-operand recompute (2026-07-21, replaces the evidence-off input-
# offload variant): rope itself saves nothing (frozen cos/sin) and its backward
# is input-free, so the autograd graph is left UNTOUCHED. Instead, the wrapper
# attaches a RECOMPUTE RECIPE to the rope OUTPUTS (q_embed/k_embed — exactly the
# bf16 [B,H,S,D] tensors SDPA saves, ~0.46 TB/step of D2H @128k): the attention
# pack stores the recipe instead of copying bytes, and unpack rebuilds the
# tensor bit-identically on the GPU from the norm wrapper's already-offloaded
# bf16 input (same device + same op chain: norm fwd -> transpose -> q*cos +
# rotate_half(q)*sin). Explicit object-attribute chaining — never anonymous
# packed-list matching. Gradients are unchanged by construction.
# ---------------------------------------------------------------------------


class RopeRecipe:
    """Everything needed to rebuild one rope output bit-identically at unpack."""

    __slots__ = ("shared", "module", "cos", "sin", "unsqueeze_dim", "out_shape", "nbytes")

    def __init__(self, shared: "_SharedXHandle", module: nn.Module, cos: torch.Tensor,
                 sin: torch.Tensor, unsqueeze_dim: int, out_shape: tuple, nbytes: int) -> None:
        self.shared = shared.retain()
        self.module = module
        self.cos = cos
        self.sin = sin
        self.unsqueeze_dim = int(unsqueeze_dim)
        self.out_shape = out_shape
        self.nbytes = int(nbytes)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # verbatim transformers.models.qwen3_moe.modeling rotate_half (bitwise-matching ops)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def recompute_rope_saved(recipe: RopeRecipe) -> torch.Tensor:
    """Rebuild the saved rope output (SDPA operand) from the offloaded bf16 norm
    input: stage x -> exact norm forward -> transpose -> exact rope ops. Same
    device + same op chain as the original forward => bit-identical."""
    shared = recipe.shared
    manager = shared.manager
    try:
        staged = _stage_fresh(shared.handle, manager)
        _count("rope_recomputes")
        _count("rope_recompute_bytes_rebuilt", recipe.nbytes)
        x_dev = staged.view(shared.x_shape)
        orig = recipe.module._asym_qknorm_recompute_original_forward  # type: ignore[attr-defined]
        with torch.no_grad():
            normed = orig(x_dev)
            qk = normed.transpose(1, 2)
            cos = recipe.cos.unsqueeze(recipe.unsqueeze_dim)
            sin = recipe.sin.unsqueeze(recipe.unsqueeze_dim)
            out = (qk * cos) + (_rotate_half(qk) * sin)
    finally:
        shared.release()
        recipe.shared = None  # consumed once; a second unpack of the same recipe is a bug
    return out


def _attach_rope_recipes(q, k, cos, sin, unsqueeze_dim, q_embed, k_embed) -> bool:
    """Attach recompute recipes to the rope outputs when both inputs chain back to
    norm-recompute handles. Returns True iff recipes were attached."""
    attached = False
    for src_t, out_t in ((q, q_embed), (k, k_embed)):
        base = src_t if getattr(src_t, "_base", None) is None else src_t._base
        src = getattr(src_t, "_asym_qknorm_src", None) or getattr(base, "_asym_qknorm_src", None)
        if src is None:
            continue
        shared, module = src
        nbytes = out_t.numel() * out_t.element_size()
        out_t._asym_rope_recipe = RopeRecipe(  # type: ignore[attr-defined]
            shared, module, cos, sin, unsqueeze_dim, tuple(int(d) for d in out_t.shape), nbytes
        )
        _count("rope_offloads")  # counter reused: recipes attached
        attached = True
    if attached:
        _rope_engaged_once(tuple(q_embed.shape))
    return attached


def _make_rope_wrapper(orig_fn):
    def apply_rotary_pos_emb_recompute(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if position_ids is not None:
            return orig_fn(q, k, cos, sin, position_ids, unsqueeze_dim)
        out = orig_fn(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        if (
            torch.is_grad_enabled()
            and isinstance(q, torch.Tensor)
            and q.device.type == "cuda"
            and q.dim() == 4
            and rope_enabled(int(q.shape[0]) * int(q.shape[2]))  # tokens = B*S ([B,H,S,D])
        ):
            # P2: graph untouched — only attach recompute recipes to the outputs
            # (consumed by the attention pack in place of byte copies).
            try:
                _attach_rope_recipes(q, k, cos, sin, int(unsqueeze_dim), out[0], out[1])
            except Exception:
                pass
        return out

    apply_rotary_pos_emb_recompute._asym_rope_recompute = True  # type: ignore[attr-defined]
    apply_rotary_pos_emb_recompute._asym_rope_recompute_orig = orig_fn  # type: ignore[attr-defined]
    return apply_rotary_pos_emb_recompute


_ROPE_PATCHED_MODULES: set[str] = set()


def install_rope_recompute() -> int:
    """Monkeypatch `apply_rotary_pos_emb` in the qwen3(-moe) modeling modules
    (idempotent, self-gated per call — inert unless `rope_enabled()`)."""
    import importlib

    patched = 0
    for mod_name in (
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.qwen3.modeling_qwen3",
    ):
        if mod_name in _ROPE_PATCHED_MODULES:
            patched += 1
            continue
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = getattr(mod, "apply_rotary_pos_emb", None)
        if fn is None:
            continue
        if getattr(fn, "_asym_rope_recompute", False):
            _ROPE_PATCHED_MODULES.add(mod_name)
            patched += 1
            continue
        mod.apply_rotary_pos_emb = _make_rope_wrapper(fn)
        _ROPE_PATCHED_MODULES.add(mod_name)
        patched += 1
    return patched
