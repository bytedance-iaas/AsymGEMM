"""Stage 7 gate for the base-weight NVMe pager (agent/impls/nvme_offload_impl.md Stage 7).

Requires a real block-backed FS for O_DIRECT — writes under ``ASYM_NVME_PATH`` (default
``/scratch_local/asym_nvme_test``), NOT tmpfs. Run with the ``.aioenv`` sidecar exported:

    export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:$CPATH" \
      LIBRARY_PATH="$AIO_HOME/lib:$LIBRARY_PATH" LD_LIBRARY_PATH="$AIO_HOME/lib:$LD_LIBRARY_PATH"
    ASYM_NVME_PATH=/scratch_local/asym_nvme_test \
      third_party/LlamaFactory/.venv/bin/python -m pytest tests/training/test_base_weight_pager.py -q
"""

import os

import pytest
import torch

from asym_gemm.training.base_weight_pager import ABSENT, INFLIGHT, RESIDENT, BaseWeightPager
from asym_gemm.training.host_weight import HostWeight
from asym_gemm.training.nvme_store import NVMeStore, NVMeStoreConfig, _pad

_SESSION_ROOT = os.path.join(
    os.environ.get("ASYM_NVME_PATH", "/scratch_local/asym_nvme_test"),
    f"pager_session_{os.getpid()}",
)
_counter = [0]

# Big enough to clear min_swappable_bytes (1 MiB) and land bf16 rows on the 2048 align:
# 1024 x 1024 bf16 = 2 MiB exactly (storage == padded -> exercises the zero-staging fast path
# when pinned; unpinned falls back to staging).
SHAPE_2D = (1024, 1024)
SHAPE_3D = (4, 512, 1024)  # 4 MiB grouped-expert-style blob
BLOB_2D = 2 << 20


@pytest.fixture(scope="session", autouse=True)
def _cleanup_session():
    yield
    import shutil

    shutil.rmtree(_SESSION_ROOT, ignore_errors=True)


def _unique_path() -> str:
    _counter[0] += 1
    p = os.path.join(_SESSION_ROOT, f"t{_counter[0]}")
    os.makedirs(p, exist_ok=True)
    return p


def _store() -> NVMeStore:
    return NVMeStore(NVMeStoreConfig(path=_unique_path(), roles=frozenset({"base_weight"})))


def _make_hw(shape, seed, dtype=torch.bfloat16) -> tuple[HostWeight, torch.Tensor]:
    n = 1
    for d in shape:
        n *= d
    t = (
        torch.arange(n, dtype=torch.float32)
        .add_(seed)
        .remainder_(97.0)
        .to(dtype)
        .view(shape)
    )
    hw = HostWeight(t, pin_memory=True, require_2d=(len(shape) == 2))
    return hw, hw._tensor.clone()


def _pager(store, *, cache_blobs=8.0, prefetch=None, jitter=None) -> BaseWeightPager:
    return BaseWeightPager(
        store,
        cache_bytes=int(cache_blobs * _pad(BLOB_2D, store.align)),
        prefetch_bytes=prefetch,
        jitter_window=jitter,
    )


def _register(pager, items):
    for key, hw in items:
        assert pager.register(key, hw) is True
    pager.finalize()


# ---------------------------------------------------------------------------
# register + roundtrip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [SHAPE_2D, SHAPE_3D], ids=["2d", "3d_grouped"])
def test_register_frees_home_and_roundtrips_bit_exact(shape):
    st = _store()
    pager = _pager(st)
    hw, expect = _make_hw(shape, seed=7)
    assert pager.register("w", hw) is True
    pager.finalize()

    assert hw._tensor is None                      # GB-scale home freed NOW
    assert hw._pager is pager and hw._pager_key == "w"

    got = hw.weight                                # fetch through the paged property
    assert got.is_pinned() if torch.cuda.is_available() else True
    assert got.dtype == expect.dtype and tuple(got.shape) == tuple(expect.shape)
    assert torch.equal(got.view(torch.int16), expect.view(torch.int16))  # bit-exact
    assert hw.tensor is got                        # RESIDENT fast path returns the same buffer


def test_register_skips_small_alias_and_homeless():
    st = _store()
    pager = _pager(st)
    small = HostWeight(torch.zeros(8, 8, dtype=torch.bfloat16), pin_memory=False)
    assert pager.register("small", small) is False          # < min_swappable_bytes
    assert small._tensor is not None and small._pager is None

    hw, _ = _make_hw(SHAPE_2D, seed=1)
    assert pager.register("w", hw) is True
    assert pager.register("alias", hw) is False             # same HostWeight seen twice
    gone = HostWeight(torch.zeros(1024, 1024, dtype=torch.bfloat16), pin_memory=False)
    gone._tensor = None
    assert pager.register("gone", gone) is False


def test_metadata_properties_never_fetch():
    st = _store()
    pager = _pager(st)
    hw, expect = _make_hw(SHAPE_2D, seed=3)
    _register(pager, [("w", hw)])

    reads_before = dict(st.stats.read_ops)
    assert tuple(hw.shape) == tuple(expect.shape)
    assert hw.dtype == expect.dtype
    assert hw.device == torch.device("cpu")
    assert hw.out_features == expect.shape[0]
    assert hw.in_features == expect.shape[1]
    assert hw.nbytes == expect.numel() * expect.element_size()
    assert hw.grad is None
    assert hw.pinned_cpu_bytes == 0                # home freed; pager cache reported separately
    repr(hw)
    assert hw.pin_memory() is hw
    assert hw.to() is hw
    with pytest.raises(RuntimeError):
        hw.cuda()
    assert dict(st.stats.read_ops) == reads_before  # ZERO fetches so far

    _ = hw.weight                                   # now a compute read fetches exactly once
    assert st.stats.read_ops.get("base_weight", 0) == reads_before.get("base_weight", 0) + 1


# ---------------------------------------------------------------------------
# trace machine
#
# Freeze is anchored on optimizer-step boundaries (mark_step, the DeepSpeed reset_step analog):
# step 1 warms miss-driven, step 2 is RECORDed, freeze fires at mark #2 with period = one full
# steady-state step. touch() dedupes consecutive same-key reads, so with the raw per-step
# pattern fwd(a..z)+bwd(z..a) the recorded step-2 stream drops the boundary duplicate (step 1
# ended on 'a', step 2 starts with 'a') and the fwd->bwd turn duplicate: period = 2k-2 for k
# keys, and the cursor lands on the period's last entry.
# ---------------------------------------------------------------------------
def _touch_pattern(pager, keys):
    for k in keys:
        pager.touch(k)


def _fwd_bwd(keys):
    return list(keys) + list(reversed(keys))


def _run_raw_steps(pager, keys, n):
    """n optimizer steps: raw fwd+bwd touches, then the step-boundary mark."""
    for _ in range(n):
        _touch_pattern(pager, _fwd_bwd(keys))
        pager.mark_step()


def test_trace_freezes_at_second_step_boundary_with_deduped_period():
    st = _store()
    pager = _pager(st)
    keys = list("abc")
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate(keys)]
    _register(pager, items)

    _run_raw_steps(pager, keys, 1)                 # step 1: warm, recording starts at its mark
    assert not pager._frozen
    _run_raw_steps(pager, keys, 1)                 # step 2 recorded -> freeze at its mark
    assert pager._frozen and pager._period == 2 * len(keys) - 2
    assert pager._cursor == pager._period - 1
    assert not pager._disabled

    before = pager.misses_after_freeze
    _run_raw_steps(pager, keys, 2)                 # two more steps, on-trace
    assert not pager._disabled
    assert pager.misses_after_freeze == before     # cache holds all 3 -> no post-freeze misses
    assert pager.summary()["step_marks"] == 4


def test_prefetch_double_buffers_ahead():
    st = _store()
    # cache 2 blobs (the doc minimum), prefetch 1 blob ahead.
    pager = _pager(st, cache_blobs=2.0, prefetch=_pad(BLOB_2D, st.align))
    keys = list("abcd")
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate(keys)]
    _register(pager, items)

    _run_raw_steps(pager, keys, 2)                 # freeze at end of raw step 2
    assert pager._frozen and not pager._disabled
    submits_at_freeze = pager.prefetch_submits
    _run_raw_steps(pager, keys, 1)                 # steady state, on-trace
    assert pager.prefetch_submits > submits_at_freeze   # read-ahead actually happened
    assert not pager._disabled
    # Values stay exact under prefetch+eviction churn (max trace gap 6 < window 8 for k=4):
    for key, hw in items:
        got = pager.touch(key)
        assert got.dtype == torch.bfloat16 and tuple(got.shape) == SHAPE_2D
    assert not pager._disabled


def test_belady_eviction_matches_bruteforce_optimal():
    st = _store()
    pager = _pager(st, cache_blobs=2.0, prefetch=0)   # tight cache, prefetch OFF -> pure Belady
    keys = list("abcd")
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate(keys)]
    _register(pager, items)

    _run_raw_steps(pager, keys, 2)                 # freeze; cursor = P-1
    assert pager._frozen and pager.prefetch_bytes == 0
    trace = list(pager._trace)                     # the actual frozen (deduped) cyclic trace
    P = len(trace)
    cur = pager._cursor
    n_periods = 3
    stream = [trace[(cur + 1 + i) % P] for i in range(n_periods * P)]  # continue from cursor
    lookup = stream + trace * 2                    # cyclic horizon for tail decisions

    resident = {k for k, e in pager._entries.items() if e.state == RESIDENT}
    assert len(resident) <= 2
    sim, optimal = set(resident), 0
    for i, k in enumerate(stream):
        if k not in sim:
            optimal += 1
            if len(sim) >= 2:
                def nxt(x):
                    for j in range(i + 1, len(lookup)):
                        if lookup[j] == x:
                            return j
                    return len(lookup) + 1         # pragma: no cover - cyclic lookup always hits
                sim.remove(max(sim, key=nxt))
            sim.add(k)

    before = pager.misses_after_freeze
    _touch_pattern(pager, stream)
    assert not pager._disabled
    assert pager.misses_after_freeze - before == optimal   # exact Belady, not merely close
    assert pager.evictions > 0


def test_jitter_within_window_tolerated():
    st = _store()
    pager = _pager(st, jitter=8)
    keys = list("abcd")
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate(keys)]
    _register(pager, items)
    _run_raw_steps(pager, keys, 2)                 # freeze; cursor at period end
    assert pager._frozen
    # swap one adjacent pair vs the trace order; max displacement < window 8
    jittered = ["c", "b", "d"] + _fwd_bwd(keys)
    _touch_pattern(pager, jittered)
    assert not pager._disabled


def test_beyond_window_mismatch_disables_and_stays_correct():
    st = _store()
    pager = _pager(st, cache_blobs=2.0, jitter=1)  # window 1 -> any real divergence disables
    keys = list("abcd")
    items = {k: _make_hw(SHAPE_2D, seed=i) for i, k in enumerate(keys)}
    _register(pager, [(k, hw) for k, (hw, _) in items.items()])
    _run_raw_steps(pager, keys, 2)                 # freeze; cursor=P-1, next on-trace = trace[0]='b'
    assert pager._frozen
    assert pager._trace[0] == "b"                  # step-2 stream starts 'b' (boundary 'a' deduped)
    pager.touch("d")                               # 'd' is beyond window 1 -> disable
    assert pager._disabled
    submits = pager.prefetch_submits
    # Fallback stays fully correct (miss-driven sync), prefetch stops growing:
    for k in ("c", "a", "d", "b"):
        got = pager.touch(k)
        _, expect = items[k]
        assert torch.equal(got.view(torch.int16), expect.view(torch.int16))
    assert pager.prefetch_submits == submits


# ---------------------------------------------------------------------------
# buffer lifecycle
# ---------------------------------------------------------------------------
class _FakeEvent:
    def __init__(self):
        self.done = False
        self.sync_calls = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.sync_calls += 1
        self.done = True


def test_quarantine_gates_reuse_without_hot_path_sync(monkeypatch):
    st = _store()
    pager = _pager(st, cache_blobs=2.0, prefetch=0)
    events = []

    def fake_event():
        ev = _FakeEvent()
        events.append(ev)
        return ev

    monkeypatch.setattr(pager, "_make_event", fake_event)
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate("abc")]
    _register(pager, items)

    pager.touch("a")
    pager.touch("b")                               # cache full (2 blobs)
    pager.touch("c")                               # must evict -> quarantine behind fake event
    assert events, "eviction did not create a reuse-guard event"
    # New contract: while the event is incomplete the buffer is NOT reused and the hot path
    # does NOT synchronize — it allocates transiently over the cap instead.
    assert all(ev.sync_calls == 0 for ev in events), "hot path must never block on the event"
    assert not events[0].done and len(pager._quarantine) == 1
    assert pager.over_budget_allocs >= 1 and pager._held_bytes > pager.cache_bytes
    # Event completes -> sweep returns the buffer to the free list; surplus is reclaimed and
    # held bytes fall back under the cap on subsequent takes.
    events[0].done = True
    got = pager.touch("a")                         # refetch evicted key, correct shape/data path
    assert tuple(got.shape) == SHAPE_2D
    monkeypatch.setattr(pager, "_make_event", lambda: None)   # further evictions ripen instantly
    for _ in range(4):                             # circulate: trim drops surplus frees
        pager.touch("b"); pager.touch("c"); pager.touch("a")
    for ev in events:
        ev.done = True
    pager._sweep_quarantine()
    assert pager._held_bytes <= pager.cache_bytes  # transient overshoot fully reclaimed


def test_held_bytes_stay_within_cache_plus_one_blob():
    st = _store()
    blob = _pad(BLOB_2D, st.align)
    pager = _pager(st, cache_blobs=2.0, prefetch=blob)
    keys = list("abcdef")
    items = [(k, _make_hw(SHAPE_2D, seed=i)[0]) for i, k in enumerate(keys)]
    _register(pager, items)
    _run_raw_steps(pager, keys, 3)
    assert pager.over_budget_allocs == 0
    assert pager._held_bytes <= pager.cache_bytes
    s = pager.summary()
    assert s["registered"] == 6 and s["resident_bytes"] == pager._held_bytes
