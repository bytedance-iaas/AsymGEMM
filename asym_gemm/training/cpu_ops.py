"""Dispatch helpers for the Grace fused CPU ops (agent/impls/cpu_compute.md Stage 1).

Gate: env ``ASYMM_CPU_FUSED_SILU`` (default OFF per cpu_compute.md rule 0.3-5) and the
``asym_gemm._C`` build actually shipping the kernels. Thread count: ``ASYM_CPU_OPS_THREADS``
(default 32 — Grace DRAM bandwidth saturates at ~32-36 cores; the kernels spin up their own
OpenMP team, never the global torch/OMP pool)."""

from __future__ import annotations

import os

import torch

_STATE: tuple | None = None


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() not in {"", "0", "false", "no", "off"}


def _kernels_enabled() -> bool:
    if _env_on("ASYMM_CPU_FUSED_SILU"):
        return True
    from . import placement_policy

    return placement_policy.enabled() and placement_policy.fused_silu_kernels_enabled()


def fused_silu_kernels() -> tuple | None:
    """Return ``(fused_fwd, fused_bwd, num_threads)`` or ``None`` when disabled."""
    global _STATE
    if _STATE is None:
        state: tuple = (None,)
        if _kernels_enabled():
            import asym_gemm as _ag

            fwd = getattr(_ag, "cpu_fused_silu_mul_bf16", None)
            bwd = getattr(_ag, "cpu_fused_silu_backward_bf16", None)
            if fwd is not None and bwd is not None:
                state = (fwd, bwd, int(os.environ.get("ASYM_CPU_OPS_THREADS", "32")))
        _STATE = state
    return None if _STATE[0] is None else _STATE


def fused_silu_applicable(*tensors: torch.Tensor) -> bool:
    return all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in tensors)


def wgrad_threads() -> int:
    """Kernel-campaign per-kernel thread override (2026-07-21 sweep): the grouped
    LoRA wgrad is COMPUTE-bound and scales past one socket — measured 1.96 TF/s
    @48T -> 2.63 @72T (socket-local) -> 2.85 @96T (threads spread, data home on
    socket 0 = the production layout). ``ASYM_CPU_OPS_THREADS_WGRAD`` overrides the
    global ``ASYM_CPU_OPS_THREADS`` for the wgrad deposit jobs only; unset = old
    behavior (default-off variant per the no-regression protocol)."""
    raw = os.environ.get("ASYM_CPU_OPS_THREADS_WGRAD")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    # DEFAULT-ON under the policy (2026-07-21): micro +45% (1.96->2.85 TF/s @96T,
    # threads spread / data home) and the e2e ride was NEUTRAL (85.50 vs 85.58 --
    # deposits are off-critical-path; the win is hidden-tail headroom). placement.md P14.
    try:
        from . import placement_policy

        if placement_policy.enabled():
            return 96
    except Exception:
        pass
    return int(os.environ.get("ASYM_CPU_OPS_THREADS", "32"))


_SPLIT_STATE: tuple | None = None


def split_silu_kernels() -> tuple | None:
    """Return ``(cpu_silu, cpu_mul_, num_threads)`` or ``None`` when disabled.

    Same gate as :func:`fused_silu_kernels`. The split form costs one extra sweep
    (5 vs 3) but lets Stage-2 overlap silu(gate) with the GPU's up-block while the
    worker's FIFO chains the ``*= up`` pass (cpu_compute.md Stage 2)."""
    global _SPLIT_STATE
    if _SPLIT_STATE is None:
        state: tuple = (None,)
        if _kernels_enabled():
            import asym_gemm as _ag

            silu = getattr(_ag, "cpu_silu_bf16", None)
            mul_ = getattr(_ag, "cpu_mul_bf16_", None)
            if silu is not None and mul_ is not None:
                state = (silu, mul_, int(os.environ.get("ASYM_CPU_OPS_THREADS", "32")))
        _SPLIT_STATE = state
    return None if _SPLIT_STATE[0] is None else _SPLIT_STATE
