"""SM90 hybrid INT8 fused grouped GEMM tests (hybridGEMM.md Phase B).

One launch of `m_grouped_int8_hybrid_gemm_nt_contiguous` computes host-side
segments (asym K-outer pipeline, CTA ranks < s_host) and HBM-side segments
(deep M-outer pipeline, remaining ranks) into one fp32 output. Exit gate B:
parity on forced splits (all-host / all-HBM / mixed, split-invariant) against
the float reference and the two standalone kernels, plus a launch-collapse
bench at decode shapes.

Runs two ways (AsymGEMM convention):
    python tests/test_sm90_int8_hybrid.py     # scripts/test.sh path; skips via exit 0
    pytest tests/test_sm90_int8_hybrid.py -s  # skips via pytest marks
"""
import random
import sys

import torch
import pytest
import asym_gemm
from asym_gemm.testing import calc_diff, get_arch_major
from asym_gemm.utils.math import ceil_div, per_channel_cast_to_int8, per_token_cast_to_int8

from test_sm90_int8 import GRAN_K, DIFF_TOL, ref_2d

RECIPE = (1, GRAN_K, GRAN_K)


def _hybrid_fn():
    fn = getattr(asym_gemm, "m_grouped_int8_hybrid_gemm_nt_contiguous", None)
    if fn is None:
        pytest.skip("m_grouped_int8_hybrid_gemm_nt_contiguous not exported by this build")
    return fn


def _empty_side():
    """(offsets, experts, list_size) for a side with no segments."""
    return (torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.tensor([-1], dtype=torch.int32, device="cuda"), 1)


def _build_case(num_host, num_hbm, m_per_group, n, k, pinned_host=False):
    """Interleaved host/HBM segments over one contiguous M layout.

    Host experts are b[:num_host] (LOCAL ids 0..num_host-1), HBM experts are
    b[num_host:] (local ids re-based). Segments alternate sides in M so both
    sides' persistent striding and the disjoint-row output contract are
    exercised. Returns the per-side layouts plus the combined float reference.
    """
    total = num_host + num_hbm
    m = total * m_per_group
    a = torch.randn((m, k), device="cuda", dtype=torch.float32) / k ** 0.25
    b = torch.randn((total, n, k), device="cuda", dtype=torch.float32) / k ** 0.25

    a_q, sfa = per_token_cast_to_int8(a)
    b_q = torch.empty_like(b, dtype=torch.int8)
    sfb = torch.empty((total, n, ceil_div(k, GRAN_K)), device="cuda", dtype=torch.float32)
    for g in range(total):
        b_q[g], sfb[g] = per_channel_cast_to_int8(b[g])

    host_ids = list(range(num_host))
    hbm_ids = list(range(num_host, total))
    random.shuffle(host_ids)
    random.shuffle(hbm_ids)
    seg_plan = []
    while host_ids or hbm_ids:
        if host_ids:
            seg_plan.append(("host", host_ids.pop()))
        if hbm_ids:
            seg_plan.append(("hbm", hbm_ids.pop()))

    m_indices = torch.empty(m, dtype=torch.int64, device="cuda")
    off_host, exp_host, off_hbm, exp_hbm = [], [], [], []
    for i, (side, gid) in enumerate(seg_plan):
        s, e = i * m_per_group, (i + 1) * m_per_group
        m_indices[s:e] = gid
        if side == "host":
            off_host += [s, e]
            exp_host.append(gid)                 # local id == global (host experts lead)
        else:
            off_hbm += [s, e]
            exp_hbm.append(gid - num_host)       # local id into b_hbm

    def to_layout(off, exp):
        if not exp:
            return _empty_side()
        return (torch.tensor(off, dtype=torch.int32, device="cuda"),
                torch.tensor(exp + [-1], dtype=torch.int32, device="cuda"),
                len(exp) + 1)

    host_layout = to_layout(off_host, exp_host)
    hbm_layout = to_layout(off_hbm, exp_hbm)

    # An empty side still needs well-formed (1-group dummy) weight tensors for
    # the TMA descriptors; its segment list is empty so they are never read.
    host_sl = slice(0, num_host) if num_host else slice(0, 1)
    hbm_sl = slice(num_host, total) if num_hbm else slice(0, 1)
    b_host = b_q[host_sl].contiguous()
    sfb_host = sfb[host_sl].contiguous()
    if pinned_host:
        b_host = b_host.cpu().pin_memory()
    b_hbm = b_q[hbm_sl].contiguous()
    sfb_hbm = sfb[hbm_sl].contiguous()

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    for g in range(total):
        rows = (m_indices == g)
        if rows.any():
            ref[rows] = ref_2d(a_q[rows], sfa[rows], b_q[g], sfb[g], k)

    return (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
            host_layout, hbm_layout, ref)


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_hybrid_mixed_parity() -> None:
    """Mixed splits: parity vs the float reference, invariant to s_host."""
    fn = _hybrid_fn()
    random.seed(0)
    torch.manual_seed(0)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("Testing SM90 hybrid INT8 fused GEMM (mixed splits):")

    cases = [
        # (num_host, num_hbm, m_per_group, n, k, pinned)
        (2, 6, 128, 256, 512, False),
        (4, 4, 64, 2048, 1024, False),    # decode-like, many n-blocks
        (3, 5, 192, 384, 640, False),     # odd sizes, pipeline wrap
        (2, 6, 128, 256, 512, True),      # the real thing: pinned host weights
    ]
    for num_host, num_hbm, mpg, n, k, pinned in cases:
        (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
         (off_h, exp_h, ls_h), (off_d, exp_d, ls_d), ref) = _build_case(
            num_host, num_hbm, mpg, n, k, pinned_host=pinned)
        m = ref.shape[0]
        for s_host in (1, 8, num_sms // 2):
            d = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float32)
            fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
               off_h, exp_h, ls_h, off_d, exp_d, ls_d, s_host, recipe=RECIPE)
            diff = calc_diff(d, ref)
            tag = "pinned" if pinned else "cuda"
            print(f"   > (host={num_host}, hbm={num_hbm}, m={m}, n={n}, k={k}, "
                  f"{tag}, s_host={s_host}): diff={diff:.5f}")
            assert diff < DIFF_TOL, \
                f"host={num_host}, hbm={num_hbm}, n={n}, k={k}, s_host={s_host}: diff={diff:.5f}"
    print()


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_hybrid_degenerate_splits() -> None:
    """s_host extremes exercise exactly one branch; each must match its
    standalone parent kernel on the same inputs (hybridGEMM.md §9.8)."""
    fn = _hybrid_fn()
    asym = getattr(asym_gemm, "m_grouped_int8_asym_gemm_nt_contiguous", None)
    deep = getattr(asym_gemm, "m_grouped_int8_gemm_nt_contiguous", None)
    random.seed(1)
    torch.manual_seed(1)
    print("Testing SM90 hybrid INT8 degenerate splits vs standalone kernels:")

    # ---- all-host (empty HBM side): the asym branch alone.
    (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
     (off_h, exp_h, ls_h), _, ref) = _build_case(4, 0, 128, 256, 512)
    m, n = ref.shape
    d = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
       off_h, exp_h, ls_h, *_empty_side(), 0, recipe=RECIPE)
    diff = calc_diff(d, ref)
    print(f"   > all-host vs ref: diff={diff:.5f}")
    assert diff < DIFF_TOL
    if asym is not None:
        d_ref = torch.zeros_like(d)
        asym((a_q, sfa), (b_host, sfb_host), d_ref, off_h, exp_h, ls_h, recipe=RECIPE)
        dk = calc_diff(d, d_ref)
        print(f"   > all-host vs asym kernel: diff={dk:.7f}")
        assert dk < 1e-6

    # ---- all-HBM (empty host side): the deep branch alone.
    (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
     _, (off_d, exp_d, ls_d), ref) = _build_case(0, 4, 128, 256, 512)
    m, n = ref.shape
    d = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
       *_empty_side(), off_d, exp_d, ls_d, 0, recipe=RECIPE)
    diff = calc_diff(d, ref)
    print(f"   > all-hbm vs ref: diff={diff:.5f}")
    assert diff < DIFF_TOL
    if deep is not None:
        d_ref = torch.zeros_like(d)
        deep((a_q, sfa), (b_hbm, sfb_hbm), d_ref, off_d, exp_d, ls_d, recipe=RECIPE)
        dk = calc_diff(d, d_ref)
        print(f"   > all-hbm vs deep kernel: diff={dk:.7f}")
        assert dk < 1e-7
    print()


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_hybrid_stealing_parity() -> None:
    """Phase C2: with enable_steal the deep side pops an atomic ticket counter
    and asym CTAs fall through to it after draining the host list. Every tile
    must still be computed exactly once, for any split — including grossly
    oversized s_host where most of the HBM work is done by stealing CTAs."""
    fn = _hybrid_fn()
    random.seed(3)
    torch.manual_seed(3)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("Testing SM90 hybrid INT8 stealing (ticket-pop + asym->hbm fallthrough):")

    cases = [
        # (num_host, num_hbm, m_per_group, n, k, pinned)
        (2, 6, 128, 256, 512, False),
        (4, 4, 64, 2048, 1024, False),    # decode-like, many n-blocks
        (1, 7, 128, 384, 640, True),      # tiny host bucket, pinned: heavy stealing
    ]
    for num_host, num_hbm, mpg, n, k, pinned in cases:
        (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
         (off_h, exp_h, ls_h), (off_d, exp_d, ls_d), ref) = _build_case(
            num_host, num_hbm, mpg, n, k, pinned_host=pinned)
        m = ref.shape[0]
        # num_sms - 1 leaves ONE native hbm CTA: nearly all hbm tiles are
        # computed by fallthrough (stealing) CTAs.
        for s_host in (1, 8, num_sms // 2, num_sms - 1):
            d = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float32)
            fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
               off_h, exp_h, ls_h, off_d, exp_d, ls_d, s_host,
               enable_steal=True, recipe=RECIPE)
            diff = calc_diff(d, ref)
            tag = "pinned" if pinned else "cuda"
            print(f"   > (host={num_host}, hbm={num_hbm}, m={m}, n={n}, k={k}, "
                  f"{tag}, s_host={s_host}, steal): diff={diff:.5f}")
            assert diff < DIFF_TOL, \
                f"steal: host={num_host}, hbm={num_hbm}, s_host={s_host}: diff={diff:.5f}"
    print()


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_hybrid_stealing_reclaims_bad_split() -> None:
    """Phase C2's point: a mispredicted (oversized) s_host strands SMs on the
    drained asym side; stealing should claw the loss back toward the
    well-balanced split's time.

    Uses a PREFILL shape: the deep side must be compute-bound for the CTA
    count to matter — at decode it is HBM-bandwidth-bound and even a quarter
    of the machine saturates DRAM, so a bad split costs (and stealing
    recovers) nothing there."""
    fn = _hybrid_fn()
    random.seed(4)
    torch.manual_seed(4)
    num_host, num_hbm, mpg, n, k = 2, 30, 256, 2048, 1024
    (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
     (off_h, exp_h, ls_h), (off_d, exp_d, ls_d), ref) = _build_case(
        num_host, num_hbm, mpg, n, k, pinned_host=True)
    m = ref.shape[0]
    d = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    def bench(s_host, steal, iters=20):
        def run():
            fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
               off_h, exp_h, ls_h, off_d, exp_d, ls_d, s_host,
               enable_steal=steal, recipe=RECIPE)
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            run()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    bad = num_sms * 3 // 4          # grossly oversized asym side
    print(f"Stealing reclaim (host={num_host} pinned, hbm={num_hbm}, N={n}, K={k}, prefill m/seg={mpg}):")
    ms_good = bench(16, False)
    ms_bad = bench(bad, False)
    ms_bad_steal = bench(bad, True)
    print(f"   > s_host=16 no-steal {ms_good:7.3f} ms | s_host={bad} no-steal {ms_bad:7.3f} ms | "
          f"s_host={bad} steal {ms_bad_steal:7.3f} ms | reclaim {ms_bad / ms_bad_steal:4.2f}x")
    # Stealing must recover a meaningful share of the misprediction penalty
    # (not just noise) whenever the bad split actually cost something.
    if ms_bad > ms_good * 1.2:
        assert ms_bad_steal < ms_bad * 0.85, \
            f"stealing reclaimed too little: {ms_bad:.3f} -> {ms_bad_steal:.3f} ms (good {ms_good:.3f})"
    print()


@pytest.mark.skipif(not torch.cuda.is_available() or get_arch_major() != 9, reason="SM90 required")
def test_hybrid_bench_vs_two_launch() -> None:
    """Exit gate B decode claim: one hybrid launch vs the two-launch path
    (deep kernel on HBM segments + asym kernel on host segments)."""
    fn = _hybrid_fn()
    asym = getattr(asym_gemm, "m_grouped_int8_asym_gemm_nt_contiguous", None)
    deep = getattr(asym_gemm, "m_grouped_int8_gemm_nt_contiguous", None)
    if asym is None or deep is None:
        pytest.skip("standalone kernels unavailable for the two-launch baseline")
    random.seed(2)
    torch.manual_seed(2)

    num_host, num_hbm, n, k = 8, 24, 2048, 1024
    print(f"Hybrid vs two-launch (host={num_host} pinned, hbm={num_hbm}, N={n}, K={k}):")
    for mpg, tag in ((64, "decode m/seg=64 "), (256, "prefill m/seg=256")):
        (a_q, sfa, b_host, sfb_host, b_hbm, sfb_hbm,
         (off_h, exp_h, ls_h), (off_d, exp_d, ls_d), ref) = _build_case(
            num_host, num_hbm, mpg, n, k, pinned_host=True)
        m = ref.shape[0]
        d = torch.zeros((m, n), device="cuda", dtype=torch.float32)
        s_host = 16

        def run_hybrid():
            fn((a_q, sfa), (b_host, sfb_host), (b_hbm, sfb_hbm), d,
               off_h, exp_h, ls_h, off_d, exp_d, ls_d, s_host, recipe=RECIPE)

        def run_two_launch():
            deep((a_q, sfa), (b_hbm, sfb_hbm), d, off_d, exp_d, ls_d, recipe=RECIPE)
            asym((a_q, sfa), (b_host, sfb_host), d, off_h, exp_h, ls_h, recipe=RECIPE)

        def bench(run, iters=20):
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                run()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        run_hybrid()
        diff = calc_diff(d, ref)
        assert diff < DIFF_TOL, f"bench-shape parity broke: diff={diff:.5f}"
        ms_hybrid = bench(run_hybrid)
        ms_two = bench(run_two_launch)
        print(f"   > {tag}: hybrid {ms_hybrid:7.3f} ms | two-launch {ms_two:7.3f} ms | "
              f"{ms_two / ms_hybrid:4.2f}x")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available() or get_arch_major() != 9:
        print("Skip: SM90 GPU required")
        sys.exit(0)
    if getattr(asym_gemm, "m_grouped_int8_hybrid_gemm_nt_contiguous", None) is None:
        print("Skip: m_grouped_int8_hybrid_gemm_nt_contiguous not exported by this build")
        sys.exit(0)
    test_hybrid_mixed_parity()
    test_hybrid_degenerate_splits()
    test_hybrid_stealing_parity()
    test_hybrid_stealing_reclaims_bad_split()
    test_hybrid_bench_vs_two_launch()
