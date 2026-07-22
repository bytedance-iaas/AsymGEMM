"""Unit gates for R5 (restage prefetch/reuse, fix_cpu_compute.md).

Run in a clean window: .venv/bin/python -m pytest tests/test_restage_prefetch.py -x -q
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("ASYM_PLACEMENT_POLICY", "0")

from asym_gemm.training import activation_offload as ao

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _reset():
    os.environ["ASYMM_ATTN_RESTAGE_PREFETCH"] = "0"
    os.environ.pop("ASYM_PREFETCH_MIN_FREE_GB", None)
    yield
    os.environ["ASYMM_ATTN_RESTAGE_PREFETCH"] = "0"
    os.environ.pop("ASYM_PREFETCH_MIN_FREE_GB", None)


@requires_cuda
def test_stage_begin_commit_identity():
    """begin/commit must return exactly the bytes stage() returns (event-ordered)."""
    mgr = ao.ActivationOffloadManager(pin_memory=True)
    src = torch.randn(9000, 512, device="cuda", dtype=torch.bfloat16)
    h = mgr.offload(src, "r5.test")
    lazy = mgr.stage(h, tag="r5.lazy").clone()
    stage, done = mgr.stage_begin(h, tag="r5.pref")
    # interleave unrelated compute between begin and commit (the overlap window)
    _ = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
    out = mgr.stage_commit(stage, done, nbytes=h.nbytes, tag="r5.pref")
    torch.cuda.synchronize()
    assert torch.equal(out, src) and torch.equal(lazy, src)
    mgr.release_stage(out, drop_cache=True)
    mgr.release_cpu(h)


@requires_cuda
def test_fused_silu_bwd_order_bit_identity():
    """The single-gate-stage order must produce bitwise-identical dup/dgate vs the
    legacy 3-stage sequence (same kernels, same operand orders; the eliminated
    re-stage was a lossless roundtrip)."""
    g = torch.Generator(device="cpu").manual_seed(7)
    gate = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()
    up = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()
    dact0 = torch.randn(65536, 768, generator=g).float().cuda()

    # legacy: silu in-place on stage 1; up stage; RE-STAGED gate for silu_backward
    gs1 = gate.clone()
    da = dact0.clone()
    F.silu(gs1, inplace=True)
    gs1.mul_(da)
    dup_legacy = gs1.to(torch.bfloat16).contiguous()
    da.mul_(up)  # up stage
    gs2 = gate.clone()  # the re-stage (bitwise-identical content by losslessness)
    dgate_legacy = torch.ops.aten.silu_backward(da, gs2).to(torch.bfloat16).contiguous()

    # fused: ONE gate stage, out-of-place silu keeps it intact
    gs = gate.clone()
    da2 = dact0.clone()
    silu_g = F.silu(gs)
    silu_g.mul_(da2)
    dup_fused = silu_g.to(torch.bfloat16).contiguous()
    da2.mul_(up)
    dgate_fused = torch.ops.aten.silu_backward(da2, gs).to(torch.bfloat16).contiguous()

    assert torch.equal(dup_fused, dup_legacy)
    assert torch.equal(dgate_fused, dgate_legacy)


@requires_cuda
def test_prefetch_flag_and_guard():
    assert ao.restage_prefetch_enabled() is False
    os.environ["ASYMM_ATTN_RESTAGE_PREFETCH"] = "1"
    assert ao.restage_prefetch_enabled() is True
    # guard: demand more free HBM than the device has -> must refuse
    os.environ["ASYM_PREFETCH_MIN_FREE_GB"] = "100000"
    assert ao.prefetch_free_ok(1 << 20) is False
    os.environ["ASYM_PREFETCH_MIN_FREE_GB"] = "1"
    assert ao.prefetch_free_ok(1 << 20) is True
