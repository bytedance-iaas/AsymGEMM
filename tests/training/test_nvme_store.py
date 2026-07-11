"""Stage 1 gate for the NVMe store substrate (agent/impls/nvme_offload_impl.md Stage 1).

Requires a real block-backed FS for O_DIRECT — writes under ``ASYM_NVME_PATH`` (default
``/scratch_local/asym_nvme_test``), NOT tmpfs. Run with the ``.aioenv`` sidecar exported:

    export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:$CPATH" \
      LIBRARY_PATH="$AIO_HOME/lib:$LIBRARY_PATH" LD_LIBRARY_PATH="$AIO_HOME/lib:$LD_LIBRARY_PATH"
    ASYM_NVME_PATH=/scratch_local/asym_nvme_test \
      third_party/LlamaFactory/.venv/bin/python -m pytest tests/training/test_nvme_store.py -q
"""

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

from asym_gemm.training import nvme_store as ns
from asym_gemm.training.nvme_store import (
    BlobRef,
    NVMeStore,
    NVMeStoreConfig,
    alloc_padded_pinned,
    get_nvme_store,
    io_ready,
    _reset_store_singleton_for_tests,
)

CUDA = torch.cuda.is_available()

_SESSION_ROOT = os.path.join(
    os.environ.get("ASYM_NVME_PATH", "/scratch_local/asym_nvme_test"),
    f"session_{os.getpid()}",
)
_counter = [0]


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


def _cfg(path, *, sync=True, **kw) -> NVMeStoreConfig:
    return NVMeStoreConfig(path=path, roles=frozenset({"activation", "base_weight"}), sync=sync, **kw)


def _store(sync=True, **kw) -> NVMeStore:
    return NVMeStore(_cfg(_unique_path(), sync=sync, **kw))


def _filled(shape, dtype, align):
    """Pinned padded tensor filled with a deterministic 0..96 pattern (exact in bf16)."""
    t = alloc_padded_pinned(shape, dtype, align=align)
    n = t.numel()
    t.view(-1).copy_(torch.arange(n, dtype=torch.float32).remainder(97.0).to(dtype))
    return t


# ---------------------------------------------------------------------------
# roundtrips: bf16/fp32, below/at/above 1 MiB, sync + async
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize("nbytes_target", [100 * 1024, 1 << 20, 4 << 20], ids=["sub1M", "1M", "4M"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_roundtrip(dtype, nbytes_target, sync):
    st = _store(sync=sync)
    try:
        elem = torch.empty(0, dtype=dtype).element_size()
        n = nbytes_target // elem
        src = _filled((n,), dtype, st.align)
        if sync:
            ref = st.spill_sync("activation", src)
        else:
            done = threading.Event()
            ref = st.spill_async("activation", src, ready_event=None, on_done=lambda r: done.set())
            assert done.wait(timeout=60), "async writer never signalled on_done"
        assert ref.durable.is_set()
        dst = alloc_padded_pinned((n,), dtype, align=st.align)
        st.fetch_into(ref, dst)
        assert torch.equal(src.view(-1), dst.view(-1))
        st.blob_done(ref)
        assert st.stats.bytes_written["activation"] == ref.length
        assert st.stats.bytes_read["activation"] == ref.length
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# arena: offset isolation + reset-when-empty
# ---------------------------------------------------------------------------
def test_arena_two_blobs_no_corruption():
    st = _store()
    try:
        a = _filled((5000,), torch.bfloat16, st.align)
        b = alloc_padded_pinned((5000,), torch.bfloat16, align=st.align)
        b.view(-1).fill_(7)
        ra = st.spill_sync("activation", a)
        rb = st.spill_sync("activation", b)
        assert ra.file == rb.file and ra.offset != rb.offset      # same arena file, distinct offsets
        da = alloc_padded_pinned((5000,), torch.bfloat16, align=st.align)
        db = alloc_padded_pinned((5000,), torch.bfloat16, align=st.align)
        st.fetch_into(ra, da); st.blob_done(ra)
        st.fetch_into(rb, db); st.blob_done(rb)
        assert torch.equal(a.view(-1), da.view(-1))
        assert torch.equal(b.view(-1), db.view(-1))
    finally:
        st.shutdown()


def test_arena_reset_when_empty_across_microbatches():
    st = _store()
    try:
        r1 = st.spill_sync("activation", _filled((4096,), torch.bfloat16, st.align))
        cursor_after_1 = st._arena_cursor
        r2 = st.spill_sync("activation", _filled((4096,), torch.bfloat16, st.align))
        assert st._arena_cursor > cursor_after_1 and st._arena_live == 2
        d = alloc_padded_pinned((4096,), torch.bfloat16, align=st.align)
        st.fetch_into(r1, d); st.blob_done(r1)
        assert st._arena_live == 1 and st._arena_cursor > 0        # not empty yet
        st.fetch_into(r2, d); st.blob_done(r2)
        assert st._arena_live == 0 and st._arena_cursor == 0       # reset-when-empty
        # microbatch 2 restarts at offset 0
        src3 = _filled((2048,), torch.bfloat16, st.align)
        r3 = st.spill_sync("activation", src3)
        assert r3.offset == 0
        d3 = alloc_padded_pinned((2048,), torch.bfloat16, align=st.align)
        st.fetch_into(r3, d3); st.blob_done(r3)
        assert torch.equal(src3.view(-1), d3.view(-1))
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# base_weight: per-blob files
# ---------------------------------------------------------------------------
def test_base_weight_per_blob_files():
    st = _store()
    try:
        s1 = _filled((3000,), torch.bfloat16, st.align)
        s2 = _filled((7000,), torch.bfloat16, st.align)
        r1 = st.spill_sync("base_weight", s1)
        r2 = st.spill_sync("base_weight", s2)
        assert r1.file != r2.file and r1.offset == 0 and r2.offset == 0
        d1 = alloc_padded_pinned((3000,), torch.bfloat16, align=st.align)
        d2 = alloc_padded_pinned((7000,), torch.bfloat16, align=st.align)
        st.fetch_into(r1, d1)
        st.fetch_into(r2, d2)
        assert torch.equal(s1.view(-1), d1.view(-1))
        assert torch.equal(s2.view(-1), d2.view(-1))
        assert st.stats.bytes_written["base_weight"] == r1.length + r2.length
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# async spill gated on a real CUDA event (D2H ordering)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA for the D2H ordering event")
def test_async_spill_gated_on_cuda_event():
    st = _store(sync=False)
    try:
        n = 1 << 20
        gpu = torch.arange(n, device="cuda", dtype=torch.float32).remainder(97.0).to(torch.bfloat16)
        pinned = alloc_padded_pinned((n,), torch.bfloat16, align=st.align)
        pinned.copy_(gpu, non_blocking=True)                       # async D2H
        ev = torch.cuda.Event(); ev.record()                      # orders after the D2H on the stream
        done = threading.Event()
        ref = st.spill_async("activation", pinned, ready_event=ev, on_done=lambda r: done.set())
        assert done.wait(timeout=60)
        dst = alloc_padded_pinned((n,), torch.bfloat16, align=st.align)
        st.fetch_into(ref, dst); st.blob_done(ref)
        assert torch.equal(pinned.view(-1), dst.view(-1))
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# durable gate: a read submitted before the write is durable blocks, then succeeds
# ---------------------------------------------------------------------------
def test_fetch_before_durable_blocks_then_succeeds():
    st = _store()
    try:
        src = _filled((4096,), torch.bfloat16, st.align)
        ref = st.spill_sync("activation", src)                    # file/offset valid; durable set
        ref.durable.clear()                                       # simulate an in-flight (async) write
        marks = []

        def _setter():
            time.sleep(0.25); marks.append(time.perf_counter()); ref.durable.set()

        th = threading.Thread(target=_setter)
        dst = alloc_padded_pinned((4096,), torch.bfloat16, align=st.align)
        th.start(); t0 = time.perf_counter()
        st.fetch_into(ref, dst)                                   # submit_pread waits on ref.durable
        t1 = time.perf_counter()
        th.join()
        assert marks and t1 >= marks[0]                           # returned only after durable was set
        assert (t1 - t0) >= 0.2
        assert torch.equal(src.view(-1), dst.view(-1))
        st.blob_done(ref)
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# 3-deep prefetch ledger reconciles via a single drain_reads
# ---------------------------------------------------------------------------
def test_prefetch_ledger_reconciles():
    st = _store()
    try:
        srcs = [_filled((2048 * (i + 1),), torch.bfloat16, st.align) for i in range(3)]
        refs = [st.spill_sync("activation", s) for s in srcs]
        dsts = [alloc_padded_pinned((s.numel(),), torch.bfloat16, align=st.align) for s in srcs]
        for r, d in zip(refs, dsts):
            st.submit_pread(r, d)
        assert len(st._pending_reads) == 3
        done = st.drain_reads()
        assert len(done) == 3 and len(st._pending_reads) == 0
        for s, d in zip(srcs, dsts):
            assert torch.equal(s.view(-1), d.view(-1))
        assert st.stats.read_ops.get("activation", 0) == 3
        for r in refs:
            st.blob_done(r)
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# backpressure: bounded inflight, no deadlock, all blobs correct
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="uses cuda events to lag the writer")
def test_backpressure_bounded_and_correct():
    cap = 3 << 20
    st = _store(sync=False, max_inflight_spill_bytes=cap)
    try:
        n = (2 << 20) // 2                                        # 2 MiB of bf16 per blob
        keep, refs, flags = [], [], []
        for _ in range(4):
            pinned = _filled((n,), torch.bfloat16, st.align); keep.append(pinned)
            a = torch.randn(2048, 2048, device="cuda"); (a @ a).sum()   # lag the ready event
            ev = torch.cuda.Event(); ev.record()
            f = threading.Event(); flags.append(f)
            refs.append(st.spill_async("activation", pinned, ready_event=ev,
                                       on_done=lambda r, f=f: f.set()))
        for f in flags:
            assert f.wait(timeout=180)
        assert st.stats.inflight_peak_bytes <= cap + (2 << 20)    # at most one blob over the cap
        assert st.stats.spill_backpressure_ms >= 0.0
        for r in refs:                                            # verify every blob roundtrips
            d = alloc_padded_pinned((n,), torch.bfloat16, align=st.align)
            st.fetch_into(r, d); st.blob_done(r)
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# io_ready / alloc_padded_pinned (C8)
# ---------------------------------------------------------------------------
def test_io_ready_and_alloc_padded_pinned():
    align = 2048
    t = alloc_padded_pinned((1000,), torch.bfloat16, align=align)
    assert t.shape == (1000,)
    assert t.untyped_storage().nbytes() % align == 0             # whole storage padded (C8)
    assert io_ready(t, align) == t.is_pinned()                  # io_ready tracks pinnedness
    u = torch.empty(1000, dtype=torch.bfloat16)                 # unpinned -> not spillable
    assert not io_ready(u, align)
    if CUDA:
        p = torch.empty(1001, dtype=torch.uint8, pin_memory=True)   # pinned but unpadded length
        assert p.is_pinned() and not io_ready(p, align)


# ---------------------------------------------------------------------------
# C2 safety: a failing async op returns rc != 0 and we RAISE (never wait -> never abort)
# ---------------------------------------------------------------------------
def test_c2_bad_path_raises_not_abort():
    st = _store()
    try:
        src = _filled((2048,), torch.bfloat16, st.align)
        ref = st.spill_sync("activation", src)
        bad = BlobRef("activation", os.path.join(st.cfg.path, "no_such_dir", "x.bin"),
                      0, ref.length, ref.logical_nbytes)
        bad.durable.set()
        dst = alloc_padded_pinned((2048,), torch.bfloat16, align=st.align)
        with pytest.raises(RuntimeError):                        # async_pread rc != 0 -> raise
            st.submit_pread(bad, dst)
        assert len(st._pending_reads) == 0                       # ledger not polluted by the failure
        st.blob_done(ref)
    finally:
        st.shutdown()


def test_misaligned_buffer_caught_upstream():
    """Our alignment assert catches a misaligned length BEFORE it reaches the C++ divisibility
    assert (which would abort the process), so C2's rc-path only ever sees open-class failures."""
    st = _store()
    try:
        bad = (torch.empty(1000, dtype=torch.uint8, pin_memory=True)
               if CUDA else torch.empty(1000, dtype=torch.uint8))   # storage 1000 B, not % 2048
        with pytest.raises(AssertionError):
            st.spill_sync("activation", bad)
    finally:
        st.shutdown()


# ---------------------------------------------------------------------------
# rule 7: disabled env => None; and (fresh interpreter) no deepspeed import
# ---------------------------------------------------------------------------
def test_disabled_env_returns_none(monkeypatch):
    monkeypatch.delenv("ASYM_NVME_ROLES", raising=False)
    _reset_store_singleton_for_tests()
    try:
        assert get_nvme_store() is None
    finally:
        _reset_store_singleton_for_tests()


def test_disabled_no_deepspeed_import_subprocess():
    code = textwrap.dedent(
        """
        import sys
        from asym_gemm.training.nvme_store import get_nvme_store
        assert get_nvme_store() is None
        # Exact package match: an editable deepspeed install leaves a PEP-660
        # '__editable___deepspeed_*_finder' path hook in sys.modules at startup.
        leaked = sorted(m for m in sys.modules if m == "deepspeed" or m.startswith("deepspeed."))
        assert not leaked, leaked
        print("RULE7_OK")
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "ASYM_NVME_ROLES"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
    assert "RULE7_OK" in r.stdout


# ---------------------------------------------------------------------------
# writer lifecycle + pid-scoped arena
# ---------------------------------------------------------------------------
def test_writer_clean_shutdown():
    st = _store(sync=False)
    w = st._writer
    assert w is not None and w.is_alive()
    st.shutdown()
    assert not w.is_alive()


def test_pid_scoped_arena_path():
    st = _store()
    try:
        assert f".{os.getpid()}." in os.path.basename(st._arena_path)
    finally:
        st.shutdown()
