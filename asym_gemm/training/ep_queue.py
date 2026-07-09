"""S2a/S2b queued+steal launch plumbing (fix_gb200_ep.md).

S2a (ASYM_EP_QUEUED=1): per-rank OWN lists + PRIVATE pinned counters (zero steal).
Counter blocks rotate through a pinned pool, host-reset right before their launch;
claim validation is ONE STEP DELAYED (head+tail divisible by n_segments per launch).

S2b: the ARMED e2e steal path is PARKED (fix_gb200_ep.md 2026-07-08 S2b E2E VERDICT:
host-ahead enqueue makes per-launch pairing cost more than the jitter it absorbs, and
the ep2 shape has no cross-rank imbalance to repair by construction). The steal kernel
+ API + sync trio live in _C, probe-proven (PR-4); they serve the ownership-lane
ablations (gb200_ep.md E4, paper phase).
"""
from __future__ import annotations

import os

import torch

__all__ = ["queued_enabled", "get_state", "validate_previous_step"]

_POOL_SIZE = 1024
_state: "_QueueState | None" = None
# process-constant: cached at import so the |1 hot path pays ZERO per-call env reads
_QUEUED = os.environ.get("ASYM_EP_QUEUED") == "1"


def queued_enabled() -> bool:
    return _QUEUED


class _QueueState:
    def __init__(self) -> None:
        self.pool = torch.zeros(_POOL_SIZE, 3, dtype=torch.int32).pin_memory()
        self.idx = 0
        self.side = int(os.environ.get("LOCAL_RANK", "0")) & 1
        self.pending: list[tuple[int, int]] = []       # this step's (slot, n_items)
        self.pending_prev: list[tuple[int, int]] = []  # last step's (validated now)
        self.overflow = False

    def next_block(self, n_items: int) -> torch.Tensor:
        slot = self.idx % _POOL_SIZE
        self.idx += 1
        if len(self.pending) >= _POOL_SIZE:
            self.overflow = True  # a step issued more launches than the pool holds
        blk = self.pool[slot]
        blk[0] = 0
        blk[1] = 0
        blk[2] = 0
        self.pending.append((slot, int(n_items)))
        return blk


def get_state() -> _QueueState:
    global _state
    if _state is None:
        _state = _QueueState()
    return _state


def validate_previous_step() -> dict:
    """Check last step's claims (kernels provably complete) and rotate the ledgers."""
    st = _state
    if st is None:
        return {"launches": 0, "claim_mismatches": 0}
    bad = 0
    for slot, n_segments in st.pending_prev:
        blk = st.pool[slot]
        taken = int(blk[1]) + int(blk[2])
        # the queue pops (segment, n_block) tuples: total items = n_segments * n_blk
        # where n_blk is the kernel's JIT tile choice (host-unknowable). Invariants a
        # private zero-steal queue must satisfy: every segment fully claimed =>
        # taken > 0, divisible by n_segments (uniform n_blk across a launch's segments).
        if taken <= 0 or taken % n_segments != 0:
            bad += 1
    report = {
        "launches": len(st.pending_prev),
        "claim_mismatches": bad,
        "overflow": st.overflow,
    }
    st.pending_prev = st.pending
    st.pending = []
    return report
