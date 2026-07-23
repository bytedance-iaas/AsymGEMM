"""Unit gates for fix_cpu_compute.md item 6 (R2) — qknorm_recompute.

Bit-parity law under test: same bf16 input + same math on the same device ⇒
the recomputed forward output and the gradients through the recomputed local
graph are EXACTLY equal (torch.equal) to the direct-autograd baseline.

Run (clean window only — needs the GPU):
  .venv/bin/python -m pytest tests/test_qknorm_recompute.py -x -q
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

os.environ.setdefault("ASYM_PLACEMENT_POLICY", "0")

from asym_gemm.training import qknorm_recompute as qkr

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, S, HQ, HK, D = 2, 512, 8, 2, 128


class FrozenLikeRMSNorm(nn.Module):
    """Replica of the production AsymFrozenRMSNorm compute: CPU-resident frozen
    weight staged per call, TWO independent x.float() upcasts (the measured
    double-save class: pow saves one, mul saves the other)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight_cpu = torch.randn(dim, dtype=torch.float32) * 0.02 + 1.0

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_cpu.to(device=x.device, non_blocking=True)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        out = x.float() * torch.rsqrt(variance + self.eps)
        return (out * weight.float()).to(dtype=x.dtype)


class StockRMSNorm(nn.Module):
    """transformers Qwen3MoeRMSNorm math (weight is a trainable Parameter)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _Attn(nn.Module):
    def __init__(self, q_norm: nn.Module, k_norm: nn.Module) -> None:
        super().__init__()
        self.q_norm = q_norm
        self.k_norm = k_norm


def _fresh_input(shape, seed=1234, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return x.to(device)


def _run_fwd_bwd(module, x, grad_out, want_wgrad=False):
    x = x.clone().requires_grad_(True)
    out = module(x)
    inputs = [x, module.weight] if want_wgrad else [x]
    grads = torch.autograd.grad(out, inputs, grad_out)
    return out.detach(), [g.detach() for g in grads]


@pytest.fixture(autouse=True)
def _reset():
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "0"
    os.environ["ASYMM_ROPE_RECOMPUTE"] = "0"
    qkr.reset_stats_for_tests()
    yield
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "0"
    os.environ["ASYMM_ROPE_RECOMPUTE"] = "0"
    qkr.reset_stats_for_tests()


@requires_cuda
@pytest.mark.parametrize("norm_cls,want_wgrad", [(FrozenLikeRMSNorm, False), (StockRMSNorm, True)])
def test_norm_recompute_bit_parity(norm_cls, want_wgrad):
    torch.cuda.init()
    norm = norm_cls(D)
    if isinstance(norm, StockRMSNorm):
        norm = norm.cuda()
        with torch.no_grad():
            norm.weight.add_(torch.randn_like(norm.weight) * 0.02)
    x = _fresh_input((B, S, HQ, D))
    grad_out = _fresh_input((B, S, HQ, D), seed=999)

    # baseline determinism guard: the parity claim rests on it
    out_a, grads_a = _run_fwd_bwd(norm, x, grad_out, want_wgrad)
    out_b, grads_b = _run_fwd_bwd(norm, x, grad_out, want_wgrad)
    assert torch.equal(out_a, out_b) and all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))

    attn = _Attn(norm, norm_cls(D))
    assert qkr.install_qknorm_recompute(attn) == 2
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "1"
    out_w, grads_w = _run_fwd_bwd(norm, x, grad_out, want_wgrad)

    assert torch.equal(out_w, out_a), "recomputed forward output must be bit-identical"
    assert torch.equal(grads_w[0], grads_a[0]), "grad_x must be bit-identical"
    if want_wgrad:
        assert torch.equal(grads_w[1], grads_a[1]), "grad_weight must be bit-identical"
    st = qkr.qknorm_recompute_stats()
    assert st["norm_offloads"] >= 1 and st["norm_recomputes"] >= 1


@requires_cuda
def test_norm_recompute_removes_fp32_saves_from_pack():
    """Inside a saved_tensors_hooks region (the attention pack stand-in): with the
    wrapper OFF the norm saves fp32 [B,S,H,D] upcasts; with it ON, none — only the
    module's own pooled bf16 offload happens (which bypasses the hooks)."""
    from torch.autograd.graph import saved_tensors_hooks

    norm = FrozenLikeRMSNorm(D)
    attn = _Attn(norm, FrozenLikeRMSNorm(D))
    qkr.install_qknorm_recompute(attn)
    x = _fresh_input((B, S, HQ, D))
    big_fp32 = []

    def _pack(t):
        if t.dtype == torch.float32 and tuple(t.shape) == (B, S, HQ, D):
            big_fp32.append(tuple(t.shape))
        return t

    def _unpack(t):
        return t

    for flag, expected_min in (("0", 2), ("1", 0)):
        os.environ["ASYMM_QKNORM_RECOMPUTE"] = flag
        big_fp32.clear()
        x_l = x.clone().requires_grad_(True)
        with saved_tensors_hooks(_pack, _unpack):
            out = norm(x_l)
        (g,) = torch.autograd.grad(out, [x_l], torch.ones_like(out))
        if expected_min:
            assert len(big_fp32) >= expected_min, f"expected the fp32 double-save, saw {big_fp32}"
        else:
            assert not big_fp32, f"wrapper ON must eliminate fp32 saves, saw {len(big_fp32)}"
    st = qkr.qknorm_recompute_stats()
    assert st["norm_offloads"] == 1  # exactly one bf16 input save, wrapper-ON pass only


@requires_cuda
def test_norm_recompute_pool_release_and_reuse():
    norm = FrozenLikeRMSNorm(D)
    attn = _Attn(norm, FrozenLikeRMSNorm(D))
    qkr.install_qknorm_recompute(attn)
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "1"
    x = _fresh_input((B, S, HQ, D))
    for _ in range(3):
        x_l = x.clone().requires_grad_(True)
        out = norm(x_l)
        torch.autograd.grad(out, [x_l], torch.ones_like(out))
    mgr = qkr._manager()
    assert mgr.stats.cpu_owned_bytes == 0, "every offloaded input must be released after backward"
    from asym_gemm.training.activation_offload import activation_offload_cpu_pool_stats

    pool = activation_offload_cpu_pool_stats()
    assert pool.get("cpu_pool_cached_bytes", 0) > 0 or mgr.stats.num_cpu_allocs >= 3


@requires_cuda
def test_norm_recompute_flag_off_is_passthrough():
    norm = FrozenLikeRMSNorm(D)
    attn = _Attn(norm, FrozenLikeRMSNorm(D))
    qkr.install_qknorm_recompute(attn)
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "0"
    x = _fresh_input((B, S, HQ, D)).requires_grad_(True)
    out = norm(x)
    assert "QKNormRecompute" not in type(out.grad_fn).__name__
    assert qkr.qknorm_recompute_stats()["norm_offloads"] == 0


def _rope_orig(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@requires_cuda
def test_rope_recipe_recompute_bit_parity():
    """P2: the rope wrapper leaves values and the autograd graph untouched, attaches
    recompute recipes to its outputs, and the recipe rebuild is BIT-IDENTICAL to the
    saved tensor; the shared x handle refcounts to zero after both consumers."""
    os.environ["ASYMM_QKNORM_RECOMPUTE"] = "1"
    os.environ["ASYMM_ROPE_RECOMPUTE"] = "1"
    q_norm = FrozenLikeRMSNorm(D)
    k_norm = FrozenLikeRMSNorm(D)
    attn = _Attn(q_norm, k_norm)
    qkr.install_qknorm_recompute(attn)
    wrapper = qkr._make_rope_wrapper(_rope_orig)

    xq = _fresh_input((B, S, HQ, D), seed=21).requires_grad_(True)
    xk = _fresh_input((B, S, HK, D), seed=22).requires_grad_(True)
    cos = _fresh_input((B, S, D), seed=23)
    sin = _fresh_input((B, S, D), seed=24)

    q = q_norm(xq).transpose(1, 2)
    k = k_norm(xk).transpose(1, 2)
    qe, ke = wrapper(q, k, cos, sin)

    # values untouched (graph unchanged)
    qe_ref, ke_ref = _rope_orig(q, k, cos, sin)
    assert torch.equal(qe, qe_ref) and torch.equal(ke, ke_ref)

    # recipes attached; rebuild is bitwise-identical to the tensor SDPA would save
    rq = getattr(qe, "_asym_rope_recipe", None)
    rk = getattr(ke, "_asym_rope_recipe", None)
    assert rq is not None and rk is not None
    rebuilt_q = qkr.recompute_rope_saved(rq)
    rebuilt_k = qkr.recompute_rope_saved(rk)
    assert torch.equal(rebuilt_q, qe.detach()), "recipe rebuild must be bit-identical"
    assert torch.equal(rebuilt_k, ke.detach())

    # gradients: graph untouched, backward drains the norm handles; pool returns to 0
    (qe.float().square().mean() + ke.float().square().mean()).backward()
    assert xq.grad is not None and xk.grad is not None
    mgr = qkr._manager()
    assert mgr.stats.cpu_owned_bytes == 0, "all shared x handles must be released"
    st = qkr.qknorm_recompute_stats()
    assert st["rope_offloads"] == 2 and st["rope_recomputes"] == 2


@requires_cuda
def test_rope_saves_nothing_large_today_evidence():
    """Documents the R2 scope evidence: with frozen cos/sin the original rope saves
    NO tensor of the q/k operand shapes (backward is linear with constant coeffs),
    so a rope-recompute can only ADD saved bytes in this graph (variant stays off)."""
    from torch.autograd.graph import saved_tensors_hooks

    q = _fresh_input((B, HQ, S, D), seed=7).requires_grad_(True)
    k = _fresh_input((B, HK, S, D), seed=8).requires_grad_(True)
    cos = _fresh_input((B, S, D), seed=9)
    sin = _fresh_input((B, S, D), seed=10)
    saved_shapes = []

    def _pack(t):
        saved_shapes.append(tuple(t.shape))
        return t

    with saved_tensors_hooks(_pack, lambda t: t):
        qe, ke = _rope_orig(q, k, cos, sin)
    torch.autograd.grad((qe, ke), [q, k], (torch.ones_like(qe), torch.ones_like(ke)))
    assert tuple(q.shape) not in saved_shapes and tuple(k.shape) not in saved_shapes, (
        f"premise violated: rope saved an operand-shaped tensor: {saved_shapes}"
    )


def _bf16_ulp_dist(a: torch.Tensor, b: torch.Tensor) -> int:
    # map the sign-magnitude bf16 bit patterns onto one monotonic integer line
    # (u < 0x8000: positives ascend; u >= 0x8000: negatives descend; +-0 coincide)
    ua = a.view(torch.int16).to(torch.int32) & 0xFFFF
    ub = b.view(torch.int16).to(torch.int32) & 0xFFFF
    ka = torch.where(ua < 0x8000, ua + 0x8000, 0x10000 - ua)
    kb = torch.where(ub < 0x8000, ub + 0x8000, 0x10000 - ub)
    return int((ka - kb).abs().max().item())


def test_cpu_rmsnorm_kernel_ulp_contract():
    """Re-verifies the shipped forward kernel's <=1 ulp contract against the exact
    fp32-upcast math, and records WHY it cannot serve the bit-exact backward (any
    nonzero ulp distance breaks bit-parity; the backward also needs the fp32 chain)."""
    import asym_gemm

    kernel = getattr(asym_gemm, "cpu_rmsnorm_bf16", None)
    if kernel is None:
        pytest.skip("cpu_rmsnorm_bf16 not exported in this build")
    g = torch.Generator().manual_seed(42)
    x = (torch.randn(4096, D, generator=g, dtype=torch.float32)).to(torch.bfloat16)
    w = torch.randn(D, generator=g, dtype=torch.float32) * 0.02 + 1.0
    out = torch.empty_like(x)
    kernel(x, w, out, 1e-6, 8)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)
    dist = _bf16_ulp_dist(out, ref)
    assert dist <= 1, f"kernel drifted beyond its <=1 ulp contract: {dist}"


@requires_cuda
def test_policy_gate_arm_pattern():
    """DEFAULT-ON since 2026-07-21 (all item-6 gates green): policy-ON alone enables
    R2; the env flag still force-arms when the policy is off; both-off stays off."""
    from asym_gemm.training import placement_policy

    if not hasattr(placement_policy, "qknorm_recompute"):
        pytest.skip("P12 rule not wired yet")
    placement_policy.reset_for_tests()
    try:
        os.environ["ASYM_PLACEMENT_POLICY"] = "1"
        os.environ["ASYMM_QKNORM_RECOMPUTE"] = "0"
        assert qkr.enabled() is True, "policy default-on (P12 gated 2026-07-21)"
        placement_policy.reset_for_tests()
        os.environ["ASYM_PLACEMENT_POLICY"] = "0"
        assert qkr.enabled() is False, "policy-off + flag-off stays off"
        os.environ["ASYMM_QKNORM_RECOMPUTE"] = "1"
        assert qkr.enabled() is True, "env flag force-arms without the policy"
    finally:
        os.environ["ASYM_PLACEMENT_POLICY"] = "0"
        os.environ["ASYMM_QKNORM_RECOMPUTE"] = "0"
        placement_policy.reset_for_tests()
