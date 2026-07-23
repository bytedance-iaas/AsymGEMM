"""Runtime CPU/GPU placement policy — implements agent/impls/placement.md P1-P9 with
P11 decision tracing (fix_cpu_compute.md item 1).

One master switch: ``ASYM_PLACEMENT_POLICY=1``. When ON, this module is the authority
for every CPU<->GPU placement decision: the per-feature env flags (``ASYMM_QWEN3_MOE_FG_
CPU_ACT``, ``ASYMM_LORA_A_GRAD_CPU``, ``ASYMM_ATTN_LORA_A_GRAD_CPU``, ``ASYM_UNSLOTH_GC_
PINNED``, ``ASYMM_QWEN3_MOE_FG_STAGE_DEDUP``, ``ASYM_CPU_ADAMW_FUSED_WIDEN``,
``ASYMM_CPU_WORKER``, ``ASYMM_CPU_FUSED_SILU``, ...) are ignored where the policy rules,
and the production placement sets of placement.md P10 materialize automatically from the
workload (model class, routed rows, activation bytes, per-projection rows). When OFF
(default), nothing changes — the flags stay the manual mechanism.

Model class is registered by the modules that actually execute (``register_model_class``
from the MoE / dense fine-grained paths); "dense" is sticky-conservative (P8: the host-
RAM-bound regime kills all CPU-compute placements until the pinned cap exists).

Design constraints: dependency-free (os/sys/json/threading/atexit only — imported from
gate helpers all over training/, must never cycle), decisions O(1), tracing deduplicated
per (rule, decision) signature, sidecar JSON written next to the source profile for the
harness to pick up (P11).

Thresholds default to the measured values in placement.md and honour the legacy env
names as overrides so existing configs keep meaning what they meant.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from typing import Any, Optional

__all__ = [
    "enabled",
    "register_model_class",
    "model_class",
    "thresholds",
    # P1 family (MoE CPU act + its riders/enablers)
    "moe_cpu_act",
    "moe_cpu_act_feature",
    "moe_cpu_act_async",
    "moe_cpu_act_chunked",
    # P2 / P3 (wgrad deposits)
    "moe_wgrad_deposit",
    "attn_wgrad_feature",
    "attn_wgrad_deposit",
    # P4/P5/P6 (never-hurts class)
    "boundary_pinned",
    "stage_dedup",
    "fused_widen",
    "save_on_cpu_dedup",
    "qknorm_recompute",
    "rope_recompute",
    "restage_prefetch",
    "moe_direct_reuse",
    # P8 / P9
    "dense_cpu_compute",
    "moe_cpu_silu_bwd",
    "moe_lora_b_deposit",
    # infra enablers implied by the placements
    "cpu_worker",
    "cpu_worker_bg",
    "fused_silu_kernels_enabled",
    # tracing
    "trace",
    "stats",
    "reset_for_tests",
]

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _env_int(names: list[str], default: int) -> int:
    for n in names:
        raw = os.environ.get(n)
        if raw is not None and str(raw).strip():
            try:
                return int(str(raw).strip())
            except ValueError:
                continue
    return default


# RLock: stats()/trace() call enabled()/thresholds() while holding the lock
# (the cpu_worker BG self-deadlock lesson, cpu_compute.md 2026-07-14).
_STATE_LOCK = threading.RLock()
_ENABLED: Optional[bool] = None
_THRESHOLDS: Optional[dict] = None
_MODEL_CLASSES: set = set()
_TRACE: dict[str, dict[str, Any]] = {}
_TRACE_SEEN: set = set()
_ATEXIT_ARMED = False


def enabled() -> bool:
    """Master switch (memoized; tests use reset_for_tests())."""
    global _ENABLED
    if _ENABLED is None:
        with _STATE_LOCK:
            if _ENABLED is None:
                _ENABLED = _env_on("ASYM_PLACEMENT_POLICY") or _env_on(
                    "ASYM_GEMM_LF_CONFIG_ASYM_PLACEMENT_POLICY"
                )
    return _ENABLED


def register_model_class(cls: str) -> None:
    """Called by the fine-grained modules that actually execute ("moe" / "dense").
    Sticky; "dense" anywhere in the model puts the run in the P8 host-RAM regime."""
    c = str(cls).strip().lower()
    if c not in {"moe", "dense"} or c in _MODEL_CLASSES:
        return
    with _STATE_LOCK:
        if c in _MODEL_CLASSES:
            return
        _MODEL_CLASSES.add(c)
    if enabled():
        print(f"[placement-policy] model_class registered: {c} -> {model_class()}",
              file=sys.stderr, flush=True)
        _export()


def model_class() -> str:
    """"dense" > "moe" > "unknown" (dense-conservative: P8)."""
    if "dense" in _MODEL_CLASSES:
        return "dense"
    if "moe" in _MODEL_CLASSES:
        return "moe"
    return "unknown"


def thresholds() -> dict:
    global _THRESHOLDS
    if _THRESHOLDS is None:
        with _STATE_LOCK:
            if _THRESHOLDS is None:
                _THRESHOLDS = {
                    # P1: measured win @2.1M rows / 3.2 GB, loss @8.2M rows / 12.6 GB
                    "act_max_rows": _env_int(
                        ["ASYM_POLICY_ACT_MAX_ROWS", "ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS",
                         "ASYM_GEMM_LF_CONFIG_ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS"],
                        4_194_304,
                    ),
                    "act_max_bytes": _env_int(
                        ["ASYM_POLICY_ACT_MAX_BYTES", "ASYMM_CPU_ACT_MAX_BYTES",
                         "ASYM_GEMM_LF_CONFIG_ASYMM_CPU_ACT_MAX_BYTES"],
                        6_400_000_000,
                    ),
                    # P3: measured win @256k proj rows (32k*b8), loss @1.02M (128k*b8)
                    "attn_wgrad_max_rows": _env_int(
                        ["ASYM_POLICY_ATTN_WGRAD_MAX_ROWS",
                         "ASYM_GEMM_LF_CONFIG_ASYM_POLICY_ATTN_WGRAD_MAX_ROWS"],
                        262_144,
                    ),
                }
    return _THRESHOLDS


def _sidecar_path() -> Optional[str]:
    src = os.environ.get("ASYM_GEMM_LF_PROFILE_SOURCE_JSON")
    if not src:
        return None
    d = os.path.dirname(src)
    return os.path.join(d, "placement_policy.json") if d else None


def trace(rule: str, decision: bool, **inputs: Any) -> None:
    """P11: record a decision. First occurrence per (rule, decision) prints one
    train.log line; the full trace is exported as placement_policy.json."""
    global _ATEXIT_ARMED
    key = (rule, bool(decision))
    with _STATE_LOCK:
        ent = _TRACE.setdefault(rule, {"decisions": {}})
        dkey = str(bool(decision))
        slot = ent["decisions"].setdefault(dkey, {"count": 0, "example_inputs": dict(inputs)})
        slot["count"] += 1
        first = key not in _TRACE_SEEN
        if first:
            _TRACE_SEEN.add(key)
        if not _ATEXIT_ARMED:
            _ATEXIT_ARMED = True
            atexit.register(_export)
    if first:
        kv = " ".join(f"{k}={v}" for k, v in inputs.items())
        print(f"[placement-policy] {rule}={decision} {kv} "
              f"(model_class={model_class()} thresholds={thresholds()})",
              file=sys.stderr, flush=True)
        _export()


def _export() -> None:
    path = _sidecar_path()
    if not path:
        return
    try:
        payload = stats()
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, default=str)
    except OSError:
        pass


def stats() -> dict:
    with _STATE_LOCK:
        return {
            "enabled": enabled(),
            "model_class": model_class(),
            "thresholds": dict(thresholds()),
            "rules": json.loads(json.dumps(_TRACE, default=str)),
        }


# ---------------- decision rules (placement.md) ----------------
#
# Feature-level rules answer "may this placement engage at all for this run?"
# (consulted where the code used to read an env flag); per-call rules take the
# actual workload numbers (rows/bytes) and are the automatic gates.


def _dense_regime() -> bool:
    return model_class() == "dense"


def moe_cpu_act_feature() -> bool:
    """P1 flag-level: MoE CPU-act path may engage (per-call guard still applies).
    P8: dense-class models are excluded entirely."""
    if _dense_regime():
        trace("P8.dense_cpu_compute", False, feature="P1.moe_cpu_act")
        return False
    return True


def moe_cpu_act(rows: int, nbytes: int) -> bool:
    """P1 per-call: MoE SwiGLU activation on CPU (async) iff it fits the hiding
    window: rows <= 4.2M AND bytes <= ~6.4 GB (measured crossover)."""
    if _dense_regime():
        trace("P8.dense_cpu_compute", False, feature="P1.moe_cpu_act")
        return False
    t = thresholds()
    d = int(rows) <= t["act_max_rows"] and int(nbytes) <= t["act_max_bytes"]
    trace("P1.moe_cpu_act", d, rows=int(rows), nbytes=int(nbytes))
    return d


def moe_cpu_act_async() -> bool:
    """P1 rider: the async (worker-overlapped) arm is the winning arm — always
    preferred when P1 engages (sync was the ablation)."""
    return True


def moe_cpu_act_chunked() -> bool:
    """K-4 rides P1's gate (placement.md P1 note): chunked mul+stage ON wherever
    the CPU act runs (provably inert above the P1 thresholds by construction)."""
    return True


def moe_wgrad_deposit() -> bool:
    """P2: MoE LoRA-A weight-grad deposit — ON for MoE at all context lengths
    (subject to P7 backpressure at runtime). OFF on dense (P8)."""
    if _dense_regime():
        trace("P8.dense_cpu_compute", False, feature="P2.moe_wgrad_deposit")
        return False
    d = model_class() == "moe"
    trace("P2.moe_wgrad_deposit", d, model_class=model_class())
    return d


def attn_wgrad_feature() -> bool:
    """P3 flag-level: attention deposits may engage (per-call rows gate applies).
    Dense-class models are excluded (P8); unknown-class conservative OFF."""
    if _dense_regime():
        trace("P8.dense_cpu_compute", False, feature="P3.attn_wgrad_deposit")
        return False
    return model_class() == "moe"


def attn_wgrad_deposit(proj_rows: int) -> bool:
    """P3 per-call: attention LoRA-A weight-grad deposit iff per-projection rows
    fit the measured window (<= ~256k; 1.02M @128k measured +1.0%)."""
    if not attn_wgrad_feature():
        return False
    d = int(proj_rows) <= thresholds()["attn_wgrad_max_rows"]
    trace("P3.attn_wgrad_deposit", d, proj_rows=int(proj_rows))
    return d


def boundary_pinned() -> bool:
    """P4: pinned+async GC boundary offload — never-hurts, ON for all models."""
    trace("P4.boundary_pinned", True)
    return True


def stage_dedup() -> bool:
    """P5: duplicate staging-copy removal — never-hurts, ON (MoE path)."""
    trace("P5.stage_dedup", True)
    return True


def fused_widen() -> bool:
    """P6: fused bf16->fp32 widen + squared-norm in the optimizer drain — ON."""
    trace("P6.fused_widen", True)
    return True


def save_on_cpu_dedup() -> bool:
    """P5 class (duplicate-copy removal): lossless dedup of double save_on_cpu
    packs (fix_cpu_compute.md item 2). OFF until its gates pass; the explicit
    ``ASYMM_SAVE_ON_CPU_DEDUP`` env flag is the gate-arm mechanism."""
    trace("P5.save_on_cpu_dedup", False)
    return False


def qknorm_recompute() -> bool:
    """P12 (fix_cpu_compute.md item 6/R2): q/k-norm recompute-instead-of-save.
    Save the bf16 norm input ONCE (pooled pinned offload) and rebuild the exact
    fp32 chain at backward (bit-identical). This is a save-traffic transform, not
    a CPU-compute placement, so P8 does NOT gate it (retention-safe: one input
    offloaded then released within the same layer backward). OFF until its gates
    pass; ``ASYMM_QKNORM_RECOMPUTE`` is the gate-arm mechanism (env wins even
    under the policy, save_dedup precedent).
    **DEFAULT-ON (2026-07-21)** — all gates green: unit bit-parity 9/9, SMOKE,
    e2e 30B@128k -3.57% (C -63.7 GB), 30B@32k -2.95% (C -16), dense 32B -4.16%
    (C -31.8)."""
    trace("P12.qknorm_recompute", True)
    return True


def rope_recompute(tokens: Optional[int] = None) -> bool:
    """P12 rope/SDPA-operand recompute (P2, gated 2026-07-21): TIME-NEUTRAL at all
    three cells (+0.36%/+1.5%/+0.7%, within/at noise) with a HOST-MEMORY WIN
    everywhere (C −18 GB @128k, −9 dense, −4.6 @32k) — a MEMORY feature.
    **Adoption: ON where C is the binding resource** — dense-class models (any
    length) and MoE calls at ≥ ``ASYM_POLICY_ROPE_MIN_TOKENS`` tokens (default
    524288 ⇒ ON at 128k-class, OFF at 32k where C is not binding and the recipe
    rebuild sits at the small-class time edge). ``ASYMM_ROPE_RECOMPUTE`` force-arms.
    """
    if model_class() == "dense":
        trace("P12.rope_recompute", True, model_class="dense")
        return True
    if tokens is None:
        return False
    d = int(tokens) >= _env_int(
        ["ASYM_POLICY_ROPE_MIN_TOKENS", "ASYM_GEMM_LF_CONFIG_ASYM_POLICY_ROPE_MIN_TOKENS"],
        524_288,
    )
    trace("P12.rope_recompute", d, tokens=int(tokens))
    return d


def restage_prefetch() -> bool:
    """P13 (fix_cpu_compute.md R5): restage prefetch/reuse — dedicated second H2D
    stream + begin/commit per-tensor events + on-GPU dgrad keeps, all G-headroom-
    guarded (ASYM_PREFETCH_MIN_FREE_GB). OFF until its gates pass;
    ``ASYMM_ATTN_RESTAGE_PREFETCH`` is the gate-arm mechanism.
    **DEFAULT-ON (2026-07-21)** — gates green: units 34/34, SMOKE, e2e 30B@32k
    -2.9% (new best 85.57), dense 32B -11.9% (guard-tuned re-gate 373.4->329.0),
    30B@128k neutral-never-hurts (G-guard no-ops at G~180/186)."""
    trace("P13.restage_prefetch", True)
    return True


def moe_direct_reuse() -> bool:
    """P15 (byte-diet round): direct reuse of GPU-born gate/up/act in place of
    their offload->restage roundtrips (per-mechanism G-guard at runtime). OFF
    until its gates pass; ``ASYMM_MOE_FG_DIRECT_REUSE`` force-arms."""
    trace("P15.moe_direct_reuse", False)
    return False


def dense_cpu_compute() -> bool:
    """P8: dense-model CPU compute — OFF until the pinned-bytes cap exists."""
    trace("P8.dense_cpu_compute", False)
    return False


def moe_cpu_silu_bwd() -> bool:
    """P9 (rejected, measured +7.7%): SwiGLU-backward on CPU — windowless."""
    trace("P9.moe_cpu_silu_bwd", False)
    return False


def moe_lora_b_deposit() -> bool:
    """P9 (rejected): LoRA-B deposits require CPU-born dgrads (silu-bwd) — off."""
    trace("P9.moe_lora_b_deposit", False)
    return False


def cpu_worker() -> bool:
    """Infra implied by P1/P2/P3: the CPU worker thread."""
    return True


def cpu_worker_bg() -> bool:
    """Infra implied by P2/P3 (deposit rescheduling; rides the deposit arms)."""
    return True


def fused_silu_kernels_enabled() -> bool:
    """Infra implied by P1: the fused/split SVE SwiGLU kernels (bit-accurate)."""
    return True


def reset_for_tests() -> None:
    global _ENABLED, _THRESHOLDS
    with _STATE_LOCK:
        _ENABLED = None
        _THRESHOLDS = None
        _MODEL_CLASSES.clear()
        _TRACE.clear()
        _TRACE_SEEN.clear()
