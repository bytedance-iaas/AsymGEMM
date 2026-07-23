"""Unit tests for asym_gemm.training.save_dedup (fix_cpu_compute.md item 2).

Gate requirements covered: pack/unpack identity, refcount, version guard,
bit-identical gradients vs stock save_on_cpu on the exact q/k-norm double-save
pattern (fp32 upcast saved by multiple autograd nodes).

Run: .venv/bin/python tests/test_save_dedup.py     (needs a GPU for the CUDA parts)
"""

import os
import sys

import torch

from asym_gemm.training import save_dedup
from asym_gemm.training.save_dedup import DedupSaveOnCpu, _SharedPack


def _cuda() -> bool:
    return torch.cuda.is_available()


def test_disabled_falls_back_to_stock():
    os.environ.pop("ASYMM_SAVE_ON_CPU_DEDUP", None)
    os.environ.pop("ASYM_PLACEMENT_POLICY", None)
    from asym_gemm.training import placement_policy

    placement_policy.reset_for_tests()
    ctx = save_dedup.save_on_cpu_maybe_dedup(pin_memory=True)
    assert isinstance(ctx, torch.autograd.graph.save_on_cpu)
    os.environ["ASYMM_SAVE_ON_CPU_DEDUP"] = "1"
    try:
        ctx = save_dedup.save_on_cpu_maybe_dedup(pin_memory=True)
        assert isinstance(ctx, DedupSaveOnCpu)
    finally:
        os.environ.pop("ASYMM_SAVE_ON_CPU_DEDUP", None)


def test_policy_rule_default_off():
    from asym_gemm.training import placement_policy

    os.environ.pop("ASYMM_SAVE_ON_CPU_DEDUP", None)
    os.environ["ASYM_PLACEMENT_POLICY"] = "1"
    try:
        placement_policy.reset_for_tests()
        # item-2 gates have not passed yet: policy leaves dedup OFF
        assert save_dedup.enabled() is False
    finally:
        os.environ.pop("ASYM_PLACEMENT_POLICY", None)
        placement_policy.reset_for_tests()


def test_pack_unpack_identity_refcount_and_version_guard_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    save_dedup.reset_stats()
    ctxm = DedupSaveOnCpu(pin_memory=True)
    t = torch.randn(257, 129, device="cuda", dtype=torch.float32)
    ref0 = t.clone()

    p1 = ctxm.pack_hook(t)
    p2 = ctxm.pack_hook(t)  # same object, same version -> shared
    assert isinstance(p1, _SharedPack) and p1 is p2 and p1.refs == 2
    assert save_dedup.stats()["hits"] == 1 and save_dedup.stats()["misses"] == 1
    assert save_dedup.stats()["bytes_deduped"] == t.numel() * t.element_size()

    t.add_(1.0)  # bump version -> must NOT dedup against the old content
    ref1 = t.clone()
    p3 = ctxm.pack_hook(t)
    assert isinstance(p3, _SharedPack) and p3 is not p1
    assert save_dedup.stats()["misses"] == 2

    u1 = ctxm.unpack_hook(p1)
    u2 = ctxm.unpack_hook(p2)
    u3 = ctxm.unpack_hook(p3)
    torch.cuda.synchronize()
    assert torch.equal(u1, ref0) and torch.equal(u2, ref0), "unpack identity broken"
    assert torch.equal(u3, ref1), "version guard broken"
    assert u1.data_ptr() != u2.data_ptr(), "unpacked consumers must be independent copies"
    assert u1.device == t.device


def test_weak_map_entry_dies_with_source_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    ctxm = DedupSaveOnCpu(pin_memory=True)
    t = torch.randn(64, 64, device="cuda")
    ctxm.pack_hook(t)
    assert len(ctxm._seen) == 1
    del t
    import gc

    gc.collect()
    # 2b: the alias FIFO holds a bounded STRONG anchor until eviction (4 later
    # packs) or region exit — so the weak entry survives del(t) by design...
    assert len(ctxm._seen) == 1 and len(ctxm._alias) == 1
    ctxm._alias.clear()  # region exit clears anchors (see __exit__)
    gc.collect()
    assert len(ctxm._seen) == 0, "weak map must die once the anchor is released"


def _rmsnorm_like(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # the exact production double-save pattern: the fp32 upcast xf is saved by
    # BOTH pow (for its backward) and mul (for rsqrt-chain backward)
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (w * xn).to(x.dtype)


def _region(x, w, lin_w):
    h = torch.nn.functional.linear(x, lin_w)
    n = _rmsnorm_like(h, w)
    return (n * n.sigmoid()).sum()


def test_bitwise_identical_grads_vs_stock_save_on_cpu_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    torch.manual_seed(1234)
    x = torch.randn(8, 64, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    lin_w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        loss_ref = _region(x, w, lin_w)
    g_ref = torch.autograd.grad(loss_ref, [w, lin_w])

    save_dedup.reset_stats()
    with DedupSaveOnCpu(pin_memory=True):
        loss = _region(x, w, lin_w)
    g = torch.autograd.grad(loss, [w, lin_w])

    torch.cuda.synchronize()
    assert torch.equal(loss, loss_ref), "forward must be bit-identical"
    for a, b in zip(g, g_ref):
        assert torch.equal(a, b), "gradients must be bit-identical (pure dedup)"
    st = save_dedup.stats()
    assert st["hits"] >= 1, f"expected dedup hits on the norm double-save, got {st}"
    print(f"  (dedup stats on the norm pattern: {st})")


def test_attention_wrapper_dedup_bitwise_and_refcount_cuda():
    """Item-2 completion: the attention saved-tensor offload pack dedups the fp32
    q/k-norm-class double-saves (same object+version -> ONE shared handle), grads
    bit-identical, version guard intact."""
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    import torch.nn as nn
    from asym_gemm.training.attention_activation_offload import (
        AttentionSavedTensorOffloadWrapper,
    )

    class NormLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(256, device="cuda", dtype=torch.float32))

        def forward(self, x):
            xf = x.to(torch.float32)  # saved by pow AND mul (the production pattern)
            var = xf.pow(2).mean(-1, keepdim=True)
            xn = xf * torch.rsqrt(var + 1e-6)
            return (self.w * xn).to(x.dtype)

    torch.manual_seed(7)
    x = torch.randn(64, 512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run_module(mod, dedup: bool):
        os.environ.pop("ASYMM_SAVE_ON_CPU_DEDUP", None)
        if dedup:
            os.environ["ASYMM_SAVE_ON_CPU_DEDUP"] = "1"
        try:
            w = AttentionSavedTensorOffloadWrapper(mod, min_bytes=1)
            mod.train()
            with torch.enable_grad():
                y = w.run(x)
                loss = y.float().square().sum()
            g = torch.autograd.grad(loss, [x, mod.w])
            torch.cuda.synchronize()
            return loss.detach(), [t.detach().clone() for t in g], w
        finally:
            os.environ.pop("ASYMM_SAVE_ON_CPU_DEDUP", None)

    mod = NormLike()
    loss_ref, g_ref, w_ref = run_module(mod, dedup=False)
    loss_on, g_on, w_on = run_module(mod, dedup=True)
    assert torch.equal(loss_ref, loss_on), "forward must be bit-identical"
    for a, b in zip(g_ref, g_on):
        assert torch.equal(a, b), "gradients must be bit-identical through the wrapper"
    assert w_ref.dedup_hits == 0
    assert w_on.dedup_hits >= 1, f"expected dedup hits on the double-save, got {w_on.dedup_hits}"
    assert w_on.offload_calls < w_ref.offload_calls, "dedup must reduce pack count"
    assert w_on.dedup_bytes > 0
    print(f"  (attention wrapper: packs {w_ref.offload_calls}->{w_on.offload_calls}, "
          f"hits={w_on.dedup_hits}, bytes={w_on.dedup_bytes})")


def test_alias_dedup_same_storage_different_wrapper_cuda():
    """Attempt #2: production presents the big duplicate saves as DIFFERENT wrappers
    on the SAME storage — the weakref-anchored alias key must dedup them, and an
    in-place write (shared view version counter) must refuse the hit."""
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    import torch.nn as nn
    from torch.utils.weak import WeakTensorKeyDictionary
    from asym_gemm.training.attention_activation_offload import (
        AttentionSavedTensorOffloadWrapper,
    )

    w = AttentionSavedTensorOffloadWrapper(nn.Identity(), min_bytes=1)
    w._dedup_seen = WeakTensorKeyDictionary()
    w._dedup_alias = {}
    base = torch.randn(512, 512, device="cuda")
    base.requires_grad_(True)
    t1 = base * 1.0  # non-leaf offloadable
    t2 = t1.view(t1.shape)  # DIFFERENT wrapper, same storage/layout, shared counter
    assert t2 is not t1
    ref = t1.detach().clone()
    # 2b: pack via EPHEMERAL wrappers (production autograd re-wraps per save; the
    # first wrapper is dead by the time the duplicate arrives)
    h1 = w._pack(t1.view(t1.shape))
    h2 = w._pack(t2)
    assert h1 is h2, "alias pack must return the shared handle"
    assert w.dedup_hits == 1
    u = w._unpack(h2)
    torch.cuda.synchronize()
    assert torch.equal(u, ref)
    # version-guard through the SHARED counter: in-place write refuses the alias hit
    t3 = t1.view(t1.shape)
    with torch.no_grad():
        t1.add_(1.0)
    h3 = w._pack(t3)
    assert h3 is not h1, "mutated alias must not dedup against the stale pack"
    u3 = w._unpack(h3)
    torch.cuda.synchronize()
    assert torch.equal(u3, t3.detach()), "post-mutation pack must carry new content"
    # outer DedupSaveOnCpu alias path
    from asym_gemm.training.save_dedup import DedupSaveOnCpu
    import asym_gemm.training.save_dedup as sd

    sd.reset_stats()
    ctxm = DedupSaveOnCpu(pin_memory=True)
    a1 = base.detach() * 2.0
    a2 = a1.view(a1.shape)
    p1 = ctxm.pack_hook(a1)
    p2 = ctxm.pack_hook(a2)
    assert p1 is p2 and sd.stats()["hits"] == 1
    w._dedup_seen = None
    w._dedup_alias = None


def test_attention_wrapper_version_guard_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    import torch.nn as nn
    from asym_gemm.training.attention_activation_offload import (
        AttentionSavedTensorOffloadWrapper,
    )
    from torch.utils.weak import WeakTensorKeyDictionary

    mod = nn.Identity()
    w = AttentionSavedTensorOffloadWrapper(mod, min_bytes=1)
    w._dedup_seen = WeakTensorKeyDictionary()
    t = torch.randn(1024, 1024, device="cuda")
    t.requires_grad_(True)
    tt = t * 1.0  # non-leaf, non-param, requires grad -> offloadable
    ref0 = tt.detach().clone()
    h1 = w._pack(tt)
    h2 = w._pack(tt)
    assert h1 is h2 and w.dedup_hits == 1
    with torch.no_grad():
        tt.add_(1.0)  # version bump
    ref1 = tt.detach().clone()
    h3 = w._pack(tt)
    assert h3 is not h1, "version guard must refuse stale content"
    u1 = w._unpack(h1)
    u3 = w._unpack(h3)
    torch.cuda.synchronize()
    assert torch.equal(u1, ref0) and torch.equal(u3, ref1)
    w._dedup_seen = None


def test_context_exit_clears_map_cuda():
    if not _cuda():
        print("  (skip: no CUDA)")
        return
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    ctxm = DedupSaveOnCpu(pin_memory=True)
    with ctxm:
        y = (x * x).sum()
    assert len(ctxm._seen) == 0, "map must be cleared at region exit"
    y.backward()  # unpack after exit must still work (packs live on the nodes)
    assert torch.allclose(x.grad, 2 * x)


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
