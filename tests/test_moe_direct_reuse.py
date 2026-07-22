"""Unit gates for the 128k moe-backward byte-diet (ASYMM_MOE_FG_DIRECT_REUSE, P15).

Direct reuse skips a lossless offload->restage roundtrip, so it is bit-identical
by construction — asserted here anyway, per the round's gate list.

Run in a clean window: .venv/bin/python -m pytest tests/test_moe_direct_reuse.py -x -q
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("ASYM_PLACEMENT_POLICY", "0")

from asym_gemm.training import qwen3_moe_finegrained as fg
from asym_gemm.training.activation_offload import ActivationOffloadManager

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _reset():
    os.environ["ASYMM_MOE_FG_DIRECT_REUSE"] = "0"
    os.environ.pop("ASYM_PREFETCH_MIN_FREE_GB", None)
    for k in fg._DIRECT_REUSE_COUNTS:
        fg._DIRECT_REUSE_COUNTS[k] = 0
    yield
    os.environ["ASYMM_MOE_FG_DIRECT_REUSE"] = "0"
    os.environ.pop("ASYM_PREFETCH_MIN_FREE_GB", None)


@requires_cuda
def test_forward_direct_reuse_bit_identity():
    """silu/mul/to(bf16) on the ORIGINAL GPU tensors must equal the same ops on the
    offload->restage copies (the roundtrip is lossless), for gate, up, and act."""
    g = torch.Generator(device="cpu").manual_seed(11)
    gate = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()
    up = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()

    mgr = ActivationOffloadManager(pin_memory=True)
    gate_cpu = mgr.offload(gate, "dr.gate")
    up_cpu = mgr.offload(up, "dr.up")
    gate_stage = mgr.stage(gate_cpu, tag="dr.gate_s")
    up_stage = mgr.stage(up_cpu, tag="dr.up_s")
    torch.cuda.synchronize()
    assert torch.equal(gate_stage, gate) and torch.equal(up_stage, up)

    # roundtrip arm
    gs = gate_stage.clone()
    F.silu(gs, inplace=True)
    gs.mul_(up_stage)
    act_rt = gs.to(dtype=torch.bfloat16).contiguous()
    # direct arm (mechs 1+3): same ops on the originals
    gd = gate.clone()
    F.silu(gd, inplace=True)
    gd.mul_(up)
    act_direct = gd.to(dtype=torch.bfloat16).contiguous()
    assert torch.equal(act_direct, act_rt)

    # mech 2: down_base consuming act directly vs the restaged act
    act_cpu = mgr.offload(act_direct, "dr.act")
    act_stage = mgr.stage(act_cpu, tag="dr.act_s")
    torch.cuda.synchronize()
    assert torch.equal(act_stage, act_direct)
    for st in (gate_stage, up_stage, act_stage):
        mgr.release_stage(st, drop_cache=True)
    for h in (gate_cpu, up_cpu, act_cpu):
        mgr.release_cpu(h)


@requires_cuda
def test_silu_bwd_single_stage_bit_identity():
    """mech 4: ONE gate stage + out-of-place silu must produce bitwise-identical
    dup/dgate vs the legacy 3-stage sequence (same kernels, same operand orders)."""
    g = torch.Generator(device="cpu").manual_seed(12)
    gate = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()
    up = (torch.randn(65536, 768, generator=g) * 3).bfloat16().cuda()
    dact0 = torch.randn(65536, 768, generator=g).bfloat16().cuda()

    # legacy: silu in-place on stage 1; up stage; RE-STAGED gate for silu_backward
    gs1 = gate.clone()
    da = dact0.clone()
    F.silu(gs1, inplace=True)
    gs1.mul_(da)
    dup_legacy = gs1.to(torch.bfloat16).contiguous()
    da.mul_(up)
    gs2 = gate.clone()  # the re-stage (bitwise-identical by losslessness)
    dgate_legacy = torch.ops.aten.silu_backward(da, gs2).to(torch.bfloat16).contiguous()

    # mech 4: single stage, out-of-place silu keeps gate intact
    gs = gate.clone()
    da2 = dact0.clone()
    silu_g = F.silu(gs)
    silu_g.mul_(da2)
    dup_single = silu_g.to(torch.bfloat16).contiguous()
    da2.mul_(up)
    dgate_single = torch.ops.aten.silu_backward(da2, gs).to(torch.bfloat16).contiguous()

    assert torch.equal(dup_single, dup_legacy)
    assert torch.equal(dgate_single, dgate_legacy)


@requires_cuda
def test_flag_and_guard_counters():
    assert fg._fg_direct_reuse_enabled() is False
    assert fg._direct_reuse_ok("up_direct", 1 << 20) is False
    assert fg._DIRECT_REUSE_COUNTS["up_direct"] == 0

    os.environ["ASYMM_MOE_FG_DIRECT_REUSE"] = "1"
    os.environ["ASYM_PREFETCH_MIN_FREE_GB"] = "100000"  # guard must deny
    assert fg._direct_reuse_ok("up_direct", 1 << 20) is False
    assert fg._DIRECT_REUSE_COUNTS["guard_denied"] == 1

    os.environ["ASYM_PREFETCH_MIN_FREE_GB"] = "1"
    assert fg._direct_reuse_ok("act_direct", 1 << 20) is True
    assert fg._DIRECT_REUSE_COUNTS["act_direct"] == 1
    st = fg.direct_reuse_stats()
    assert st["enabled"] is True and st["guard_denied"] == 1
