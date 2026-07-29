"""_plan_layer_residency policy tests (no hardware required).

The policy is pure host-side logic over a numpy count array, deliberately
separated from the mechanism (Layer.update_gpu_cache) so a predictor can
replace it later. That separation is what makes it testable without a GPU,
an AMX host, or a built kernel — this file imports only the planner.

Runs eight tests:
    1. test_returns_none_when_nothing_moves   None, not [] — refresh_gpu_caches branches on it
    2. test_swaps_when_challenger_clears_margin
    3. test_no_swap_within_margin             hotter, but not by enough
    4. test_margin_boundary_is_exclusive      exactly (1+margin)x does NOT swap
    5. test_zero_signal_never_evicts          quiet periods must not churn
    6. test_max_swap_caps_steady_state_churn
    7. test_cold_start_may_replace_whole_set  the first _COLD_REFRESHES are unbounded
    8. test_converges_then_settles            reaches the true top-n, then stops moving

Designed to run two ways (AsymGEMM convention):
    python tests/test_residency_policy.py
    pytest tests/test_residency_policy.py -s
"""
from __future__ import annotations

import numpy as np

from asym_gemm.unified_moe.runtime import _COLD_REFRESHES, _plan_layer_residency

MARGIN = 0.25
MAX_SWAP = 4


class _FakeLayer:
    """Minimal stand-in: the planner only reads cache_n, _slot_np, _refresh_n.

    Passing a real Layer would drag in pinned host allocation and a CUDA
    context for logic that never touches either.
    """

    def __init__(self, num_experts, resident_ids, refresh_n=_COLD_REFRESHES):
        self.cache_n = len(resident_ids)
        self._slot_np = np.full(num_experts, -1, dtype=np.int64)
        self._slot_np[list(resident_ids)] = np.arange(len(resident_ids))
        # Default to steady state: cold-start budget already spent, so tests
        # exercise the max_swap path unless they say otherwise.
        self._refresh_n = refresh_n

    def resident(self):
        return sorted(np.nonzero(self._slot_np >= 0)[0].tolist())

    def apply(self, new_ids):
        """Slot-agnostic stand-in for update_gpu_cache. The planner only cares
        whether _slot_np[e] >= 0, never which slot, so re-numbering is fine."""
        self._slot_np[:] = -1
        self._slot_np[list(new_ids)] = np.arange(len(new_ids))
        self._refresh_n += 1


def _counts(pairs, n):
    c = np.zeros(n, dtype=np.float32)
    for e, v in pairs.items():
        c[e] = v
    return c


def test_returns_none_when_nothing_moves():
    """refresh_gpu_caches does `if new_ids is None: continue`, so an empty
    list here would be applied as a residency change and cost a barrier."""
    layer = _FakeLayer(8, [0, 1])
    c = _counts({0: 100, 1: 200, 2: 10}, 8)
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert new_ids is None, new_ids
    assert swaps == 0


def test_swaps_when_challenger_clears_margin():
    layer = _FakeLayer(8, [0, 1])
    # coldest resident is 0 (100); 200 > 100 * 1.25
    c = _counts({0: 100, 1: 300, 2: 200}, 8)
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert swaps == 1
    assert new_ids == [1, 2], new_ids
    assert len(new_ids) == layer.cache_n


def test_no_swap_within_margin():
    """Hotter than the victim, but not by enough to pay the eviction."""
    layer = _FakeLayer(8, [0, 1])
    c = _counts({0: 100, 1: 300, 2: 120}, 8)      # 120 < 125
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert new_ids is None
    assert swaps == 0


def test_margin_boundary_is_exclusive():
    """Exactly (1+margin)x must NOT swap — the comparison is `<=`, so ties
    lose. Otherwise a perfectly balanced pair trades places every window."""
    layer = _FakeLayer(8, [0, 1])
    c = _counts({0: 100, 1: 300, 2: 125}, 8)      # exactly 100 * 1.25
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert new_ids is None
    assert swaps == 0


def test_zero_signal_never_evicts():
    """A layer that saw no traffic must not churn: with an all-zero window
    every challenger is 0, and 0 > 0 * 1.25 is false anyway, but the explicit
    `c[cand] <= 0` guard also covers a zero challenger against a zero victim."""
    layer = _FakeLayer(8, [0, 1])
    new_ids, swaps = _plan_layer_residency(layer, np.zeros(8, np.float32),
                                           MAX_SWAP, MARGIN)
    assert new_ids is None
    assert swaps == 0

    # ... and a zero challenger cannot displace a zero resident either.
    layer2 = _FakeLayer(8, [0, 1])
    c = _counts({0: 0, 1: 0, 2: 0, 3: 0}, 8)
    assert _plan_layer_residency(layer2, c, MAX_SWAP, MARGIN) == (None, 0)


def test_max_swap_caps_steady_state_churn():
    """Past cold start the budget is max_swap, however many hot challengers
    are queued — eviction costs real PCIe bytes."""
    layer = _FakeLayer(16, [0, 1, 2, 3, 4, 5, 6, 7])
    c = np.ones(16, dtype=np.float32)
    for e in range(8, 16):
        c[e] = 100.0                              # eight blatant challengers
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert swaps == MAX_SWAP, swaps
    assert len(new_ids) == layer.cache_n          # size is invariant
    assert len(set(new_ids) - set(range(8))) == MAX_SWAP


def test_cold_start_may_replace_whole_set():
    """The construction-time set is arbitrary (experts 0..n-1), so the first
    _COLD_REFRESHES calls are unbudgeted; creeping at max_swap would take
    n/max_swap refreshes to reach a set worth having."""
    # cache_n must exceed MAX_SWAP for the two budgets to differ at all.
    resident, hot = [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]
    c = _counts({**{e: 1 for e in resident},
                 **{e: 100 - i for i, e in enumerate(hot)}}, 12)
    assert len(resident) > MAX_SWAP

    layer = _FakeLayer(12, resident, refresh_n=0)
    new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
    assert swaps == len(resident) > MAX_SWAP, (swaps, MAX_SWAP)
    assert new_ids == hot, new_ids

    # Same counts, but past the cold-start window: clamped to max_swap.
    clamped = _FakeLayer(12, resident, refresh_n=_COLD_REFRESHES)
    _, swaps_clamped = _plan_layer_residency(clamped, c, MAX_SWAP, MARGIN)
    assert swaps_clamped == MAX_SWAP, swaps_clamped


def test_converges_then_settles():
    """On a stationary skewed window the policy must stop moving, at a set
    that is a margin fixed point. A policy that keeps churning at steady
    state pays PCIe forever for no coverage gain.

    Note it settles *near* the true top-n, not necessarily on it: the margin
    is what stops the last few near-tied swaps, and that is the intended
    trade. Asserting exact optimality here would be asserting margin == 0.
    """
    rng = np.random.default_rng(0)
    G, N = 64, 16
    c = (1.0 / np.arange(1, G + 1) ** 1.1).astype(np.float32) * 1000.0
    c = c[rng.permutation(G)]                     # hot experts scattered, not 0..N
    truth = sorted(int(i) for i in np.argsort(c)[::-1][:N])

    layer = _FakeLayer(G, list(range(N)), refresh_n=0)
    history = []
    for _ in range(40):
        new_ids, swaps = _plan_layer_residency(layer, c, MAX_SWAP, MARGIN)
        history.append(swaps)
        if new_ids is None:
            break
        layer.apply(new_ids)

    assert history[-1] == 0, f"never settled: {history}"
    assert len(layer.resident()) == N

    # It stopped for the documented reason: no outsider beats the coldest
    # resident by the margin. This is the fixed point the policy promises.
    res = layer.resident()
    coldest = min(c[e] for e in res)
    for e in range(G):
        if e not in res:
            assert c[e] <= coldest * (1.0 + MARGIN), (
                f"expert {e} ({c[e]:.1f}) should have displaced the coldest "
                f"resident ({coldest:.1f})")

    # Near-optimal in the thing that actually matters: token coverage.
    cov = c[res].sum() / c.sum()
    best = c[truth].sum() / c.sum()
    assert cov >= 0.99 * best, (cov, best)

    # And it got there quickly rather than creeping max_swap at a time.
    assert len(history) <= 10, history


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} passed")
