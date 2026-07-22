"""Synthetic stress test for asym_gemm.training.pinned_ledger (fix_cpu_compute.md
item 4 acceptance: "synthetic stress test shows the cap holds").

Run: .venv/bin/python tests/test_pinned_ledger.py     (CUDA parts need a GPU)
"""

import gc
import os
import sys

import torch

from asym_gemm.training import pinned_ledger as pl


def _reset(**caps):
    for k in list(os.environ):
        if k.startswith("ASYM_PINNED_CAP_"):
            os.environ.pop(k)
    for k, v in caps.items():
        os.environ[k] = str(v)
    pl.reset_for_tests()


def _cuda() -> bool:
    return torch.cuda.is_available()


MB = 1 << 20


def test_reserve_release_and_caps():
    _reset(ASYM_PINNED_CAP_GB_STRESS="0.0625")  # 64 MiB family cap
    assert pl.try_reserve("stress", 32 * MB) is True
    assert pl.try_reserve("stress", 32 * MB) is True
    assert pl.try_reserve("stress", 1) is False  # cap holds exactly
    st = pl.stats()
    assert st["live_bytes"]["stress"] == 64 * MB
    assert st["high_water_bytes"]["stress"] == 64 * MB
    assert st["denials"]["stress"] == 1
    pl.release("stress", 32 * MB)
    assert pl.try_reserve("stress", 16 * MB) is True
    assert pl.stats()["live_bytes"]["stress"] == 48 * MB
    # other families unaffected
    assert pl.try_reserve("other", 500 * MB) is True
    _reset()


def test_total_cap():
    _reset(ASYM_PINNED_CAP_TOTAL_GB="0.0625")
    assert pl.try_reserve("a", 40 * MB) is True
    assert pl.try_reserve("b", 40 * MB) is False  # total cap holds across families
    st = pl.stats()
    assert st["total_live_bytes"] == 40 * MB and st["denials"]["b"] == 1
    _reset()


def test_alloc_cpu_family_attribution_and_fallback_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    from asym_gemm.training import activation_offload as ao

    ao.clear_activation_offload_cpu_pool()
    gc.collect()
    # 1-D shapes bypass the pool's dim-0 row bucketing -> exact-size allocations
    _reset(ASYM_PINNED_CAP_GB_MOE="0.003")  # ~3 MiB: fits one 2 MiB buffer, not two
    t = ao._alloc_cpu((MB,), torch.bfloat16, pin_memory=True, tag="moe.gate")  # 2 MiB
    t2 = ao._alloc_cpu((2 * MB,), torch.bfloat16, pin_memory=True, tag="moe.up")  # 4 MiB > cap
    st = pl.stats()
    assert t.is_pinned() and st["live_bytes"].get("moe", 0) == 2 * MB
    assert not t2.is_pinned(), "over-cap alloc must fall back to unpinned"
    assert st["denials"].get("moe", 0) >= 1
    # attribution by family from the tag
    t3 = ao._alloc_cpu((MB,), torch.bfloat16, pin_memory=True, tag="gc.boundary")
    assert t3.is_pinned() and pl.stats()["live_bytes"].get("gc", 0) == 2 * MB
    del t, t2, t3
    gc.collect()
    assert pl.stats()["live_bytes"].get("moe", 0) == 0, "finalizer must release on GC"
    assert pl.stats()["live_bytes"].get("gc", 0) == 0
    _reset()


def test_alloc_cpu_books_bucketed_bytes_cuda():
    # 2-D allocs are dim-0 bucketed (min 8192 rows) by the pool: the ledger must book
    # the BUCKETED bytes (what is actually page-locked), not the requested view size.
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    from asym_gemm.training import activation_offload as ao

    ao.clear_activation_offload_cpu_pool()
    gc.collect()
    _reset()
    t = ao._alloc_cpu((1024, 1024), torch.bfloat16, pin_memory=True, tag="moe.gate")
    st = pl.stats()
    assert st["live_bytes"].get("moe", 0) == 8192 * 1024 * 2, f"bucketed booking wrong: {st}"
    assert t.shape == (1024, 1024)
    del t
    gc.collect()
    assert pl.stats()["live_bytes"].get("moe", 0) == 0
    _reset()


def test_ds_slots_denial_still_functional_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    _reset(ASYM_PINNED_CAP_GB_DEPOSIT="0.001")  # ~1 MiB
    from asym_gemm.training.qwen3_moe_finegrained import _DsSlots

    slots = _DsSlots()
    like = torch.empty(2048, 1024, dtype=torch.bfloat16, device="cuda")  # 4 MiB
    _, buf = slots.acquire(like)
    assert not buf.is_pinned(), "cap denial must yield an unpinned slot"
    buf.copy_(like, non_blocking=True)  # degrades to sync; still correct
    torch.cuda.synchronize()
    assert torch.equal(buf, like.cpu())
    assert pl.stats()["denials"].get("deposit", 0) >= 1
    _reset()


def test_pin_if_requested_denial():
    _reset(ASYM_PINNED_CAP_GB_ADAM="0.001")
    from asym_gemm.training.cpu_adam import _pin_if_requested

    t = torch.empty(4 * MB, dtype=torch.uint8)
    out, reason = _pin_if_requested(t, pin_memory=True)
    if torch.cuda.is_available():
        assert out is t and reason is not None and "cap denial" in reason
        assert pl.stats()["denials"].get("adam", 0) == 1
    _reset()


def test_save_dedup_pack_respects_cap_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    _reset(ASYM_PINNED_CAP_GB_SAVE_ON_CPU="0.002")
    from asym_gemm.training.save_dedup import DedupSaveOnCpu

    ctxm = DedupSaveOnCpu(pin_memory=True)
    small = torch.randn(256, 1024, device="cuda")  # 1 MiB fp32
    big = torch.randn(1024, 1024, device="cuda")  # 4 MiB fp32 > cap
    p_small = ctxm.pack_hook(small)
    p_big = ctxm.pack_hook(big)
    torch.cuda.synchronize()
    assert p_small.cpu.is_pinned()
    assert not p_big.cpu.is_pinned(), "over-cap pack must fall back to unpinned"
    u = ctxm.unpack_hook(p_big)
    torch.cuda.synchronize()
    assert torch.equal(u, big), "unpinned fallback must stay bit-correct"
    _reset()


def test_stress_cap_holds_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    from asym_gemm.training import activation_offload as ao

    ao.clear_activation_offload_cpu_pool()
    gc.collect()
    _reset(ASYM_PINNED_CAP_TOTAL_GB="0.0625")  # 64 MiB global
    keep = []
    pinned_count = 0
    for i in range(50):
        # 1-D 8 MiB allocs (no dim-0 bucketing) -> exact cap arithmetic
        t = ao._alloc_cpu((4 * MB,), torch.bfloat16, pin_memory=True, tag=f"moe.s{i}")
        keep.append(t)
        pinned_count += int(t.is_pinned())
    st = pl.stats()
    assert st["total_live_bytes"] <= 64 * MB, f"cap breached: {st['total_live_bytes']}"
    assert st["total_high_water_bytes"] <= 64 * MB
    assert pinned_count == 8, f"expected exactly 8x8MiB pinned under 64MiB cap, got {pinned_count}"
    assert sum(st["denials"].values()) == 42
    keep.clear()
    gc.collect()
    assert pl.stats()["total_live_bytes"] == 0
    _reset()


def test_dense_cpu_left_fallback_unpinned_cuda():
    """Item-4 attempt #3: cap-denied (unpinned) act handles must route the dense
    cpu-left LoRA-A forward AND the cpu-right wgrad to the staged-GPU fallback with
    exact reference-matmul results (the pinned kernel path stays for pinned handles)."""
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    from types import SimpleNamespace

    import torch

    from asym_gemm.training.activation_offload import CPUActivationHandle
    from asym_gemm.training.dense_mlp_finegrained import AsymFinegrainedDenseMLP

    _reset()
    torch.manual_seed(11)
    m, k, r = 4096, 512, 64
    x_gpu = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    lora_a = torch.randn(r, k, device="cuda", dtype=torch.bfloat16)

    def handle(pin: bool) -> CPUActivationHandle:
        t = torch.empty(m, k, dtype=torch.bfloat16, pin_memory=pin)
        t.copy_(x_gpu)
        return CPUActivationHandle(
            tag="mlp.act", tensor=t, original_device=torch.device("cuda"),
            original_dtype=torch.bfloat16, original_shape=(m, k),
        )

    def plan(rows, device):
        return SimpleNamespace(
            offsets=torch.tensor([0, rows], device=device, dtype=torch.long),
            experts=torch.tensor([0, -1], device=device, dtype=torch.long),
        )

    layer = SimpleNamespace(backend="asym", stats=None, _one_expert_plan=plan)
    ref = x_gpu.matmul(lora_a.t())

    out_unpinned = AsymFinegrainedDenseMLP._cpu_left_lora_a(layer, handle(False), lora_a, tag="down")
    torch.cuda.synchronize()
    assert torch.equal(out_unpinned, ref), "unpinned fallback must equal the reference matmul"

    out_pinned = AsymFinegrainedDenseMLP._cpu_left_lora_a(layer, handle(True), lora_a, tag="down")
    torch.cuda.synchronize()
    diff = (out_pinned.float() - ref.float()).abs().max().item()
    assert diff <= 0.5, f"pinned kernel path drifted vs reference: {diff}"

    # wgrad consumer: dS^T @ X
    dS = torch.randn(m, r, device="cuda", dtype=torch.bfloat16)
    gref = dS.t().contiguous().matmul(x_gpu).to(torch.bfloat16)
    g_unpinned = AsymFinegrainedDenseMLP._cpu_right_lora_a_grad(
        layer, dS, handle(False), lora_a, tag="down", deposit_ctx=None
    )
    torch.cuda.synchronize()
    assert torch.equal(g_unpinned, gref), "unpinned wgrad fallback must equal reference"
    _reset()


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
