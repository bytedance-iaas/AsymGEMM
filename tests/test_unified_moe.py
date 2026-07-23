"""asym_gemm.unified_moe parity + pinned-residency tests (§9.1 of docs/unified_moe.md).

Runs seven tests:
    1. test_cpu_vs_fp32              CPU bucket vs FP32 reference (≤ 4% scale-rel, cos ≥ 0.999)
    2. test_gpu_vs_fp32              GPU bucket vs FP32 reference (≤ 4% scale-rel, cos ≥ 0.999)
    3. test_cpu_vs_gpu_single_expert The headline INT8 parity (CPU == GPU bit-exact typical)
    4. test_dispatch_invariance      m_cpu=0 vs m_cpu=inf (≤ 1.5%)
    5. test_mixed_bucket_parity      mid-range m_cpu vs all-CPU (≤ 1.5%)
    6. test_pinned_weight_pointer_identity  GPU reads pinned host weights, no VRAM mirror
    7. test_no_vram_weight_residency        VRAM delta after Layer construction ≈ SFB only

Designed to run two ways (AsymGEMM convention):
    python tests/test_unified_moe.py          # used by scripts/test.sh
    pytest tests/test_unified_moe.py -s       # works too

Internal skip when AMX or CUDA is not present — the file stays safe on
hosts that don't have both halves of the unified kernel.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import asym_gemm

# -- Hardware/build gating ---------------------------------------------------

def _module_skip(msg: str) -> None:
    """Skip the whole file in a way that works under both launch modes:
    pytest.skip(allow_module_level=True) when collected by pytest, plain
    exit 0 under `python <file>` (a sys.exit at import would be a pytest
    collection *error*, not a skip)."""
    print(f"[SKIP] {msg}")
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    sys.exit(0)


if asym_gemm.unified_moe is None:
    _module_skip("asym_gemm.unified_moe sub-package not available "
                 "(CPU extension _cpu_C did not build).")

_caps = asym_gemm.unified_moe._C.caps()
if not _caps.get("has_amx_int8", False):
    _module_skip(f"AMX-INT8 not present on this host (caps={_caps}). "
                 "The unified_moe CPU bucket requires AMX.")

if not torch.cuda.is_available():
    _module_skip("No CUDA device — GPU-bucket tests cannot run.")

from asym_gemm.unified_moe import Layer


# -- Reference computation ---------------------------------------------------

def fp32_reference_expert(
    x_bf16_cpu: torch.Tensor,        # [M, H] bf16
    gate_bf16: torch.Tensor,         # [I, H] bf16
    up_bf16:   torch.Tensor,         # [I, H] bf16
    down_bf16: torch.Tensor,         # [H, I] bf16
) -> torch.Tensor:                    # [M, H] fp32
    """BF16 → FP32 matmul → SiLU → matmul. Mirrors the unified runtime by
    casting `act` to BF16 between the two GEMMs (both backends re-quantize
    after that round-trip)."""
    x = x_bf16_cpu.float()
    gate = x @ gate_bf16.float().t()
    up   = x @ up_bf16.float().t()
    act  = torch.nn.functional.silu(gate) * up
    act_bf16 = act.to(torch.bfloat16).float()
    return act_bf16 @ down_bf16.float().t()


def fp32_reference_moe(
    x_bf16_cpu: torch.Tensor,
    expert_ids: torch.Tensor,
    route_w:    torch.Tensor,
    gate, up, down,
) -> torch.Tensor:
    T, H = x_bf16_cpu.shape
    out_fp32 = torch.zeros((T, H), dtype=torch.float32)
    G = gate.shape[0]
    ei = expert_ids.cpu().to(torch.int64).numpy()
    rw = route_w.cpu().float().numpy()
    for e in range(G):
        rows, slots = [], []
        for t in range(T):
            for j in range(expert_ids.shape[1]):
                if int(ei[t, j]) == e:
                    rows.append(t); slots.append(j)
        if not rows:
            continue
        gathered = x_bf16_cpu[rows]
        y = fp32_reference_expert(gathered, gate[e], up[e], down[e])
        w = torch.tensor([rw[t, j] for t, j in zip(rows, slots)],
                         dtype=torch.float32)[:, None]
        out_fp32.index_add_(0, torch.tensor(rows, dtype=torch.long), y * w)
    return out_fp32.to(torch.bfloat16)


def unique_topk_routing(T: int, G: int, top_k: int) -> torch.Tensor:
    """[T, top_k] with no duplicates per row — real-router style."""
    assert G >= top_k
    out = torch.empty(T, top_k, dtype=torch.int64)
    for t in range(T):
        out[t] = torch.randperm(G)[:top_k]
    return out


def scale_rel(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a, r = actual.float(), ref.float()
    return float(((a - r).abs().max() / r.abs().max().clamp(min=1e-6)).item())


def cos_sim(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a, r = actual.float().flatten(), ref.float().flatten()
    return float(torch.dot(a, r) / (a.norm() * r.norm()).clamp(min=1e-12))


# -- Tests -------------------------------------------------------------------

def test_cpu_vs_fp32():
    torch.manual_seed(0)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=10**9)
    T = 16
    x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    expert_ids = unique_topk_routing(T, G, top_k)
    route_w    = torch.softmax(torch.randn(T, top_k), dim=-1).float()
    y_cpu = layer.forward(x_cpu, expert_ids, route_w)
    y_ref = fp32_reference_moe(x_cpu, expert_ids, route_w, gate, up, down)
    r, c = scale_rel(y_cpu, y_ref), cos_sim(y_cpu, y_ref)
    print(f"  [test_cpu_vs_fp32]            scale_rel={r:.3e}  cos_sim={c:.6f}")
    assert r <= 4e-2, f"CPU vs FP32 scale_rel {r:.3e} > 4e-2"
    assert c >= 0.999, f"CPU vs FP32 cos_sim {c:.6f} < 0.999"


def test_gpu_vs_fp32():
    torch.manual_seed(1)
    G, H, I, top_k = 2, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=0)
    T = 64
    x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    expert_ids = unique_topk_routing(T, G, top_k)
    route_w    = torch.softmax(torch.randn(T, top_k), dim=-1).float()
    y_gpu = layer.forward(x_cpu.to("cuda"), expert_ids.cuda(), route_w.cuda())
    y_ref = fp32_reference_moe(x_cpu, expert_ids, route_w, gate, up, down)
    r, c = scale_rel(y_gpu.cpu(), y_ref), cos_sim(y_gpu.cpu(), y_ref)
    print(f"  [test_gpu_vs_fp32]            scale_rel={r:.3e}  cos_sim={c:.6f}")
    assert r <= 4e-2, f"GPU vs FP32 scale_rel {r:.3e} > 4e-2"
    assert c >= 0.999, f"GPU vs FP32 cos_sim {c:.6f} < 0.999"


def test_cpu_vs_gpu_single_expert():
    """Headline test: same expert, same inputs, both backends; agree on INT8."""
    torch.manual_seed(2)
    H, I, M = 256, 512, 32
    gate = torch.randn(I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(H, I, dtype=torch.bfloat16) * 0.05
    x    = torch.randn(M, H, dtype=torch.bfloat16)

    layer = Layer.from_bf16(
        gate.unsqueeze(0), up.unsqueeze(0), down.unsqueeze(0),
        top_k=1, cpu_threads=8, m_cpu=0,            # all GPU
    )
    expert_ids = torch.zeros(M, 1, dtype=torch.int64)
    route_w    = torch.ones(M, 1, dtype=torch.float32)

    y_gpu = layer.forward(x.to("cuda"), expert_ids.cuda(), route_w.cuda()).cpu()
    layer.set_m_cpu(10**9)                          # all CPU
    y_cpu = layer.forward(x, expert_ids, route_w)
    y_ref = fp32_reference_expert(x, gate, up, down).to(torch.bfloat16)

    r_cpu_ref = scale_rel(y_cpu, y_ref)
    r_gpu_ref = scale_rel(y_gpu, y_ref)
    r_cpu_gpu = scale_rel(y_cpu, y_gpu)
    print(f"  [test_cpu_vs_gpu_single_expert] cpu_vs_ref={r_cpu_ref:.3e} "
          f"gpu_vs_ref={r_gpu_ref:.3e}  cpu_vs_gpu={r_cpu_gpu:.3e}")
    assert r_cpu_ref <= 4e-2
    assert r_gpu_ref <= 4e-2
    # The two backends share INT8 → INT32 → FP32 math. Typical: bit-exact.
    assert r_cpu_gpu <= 1.5e-2, \
        f"cpu_vs_gpu scale_rel {r_cpu_gpu:.3e} > 1.5e-2"


def test_dispatch_invariance():
    """Force m_cpu=0 vs m_cpu=inf; end-to-end logits agree within INT8 noise."""
    torch.manual_seed(3)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=0)
    T = 64
    x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    expert_ids = unique_topk_routing(T, G, top_k)
    route_w    = torch.softmax(torch.randn(T, top_k), dim=-1).float()

    y_all_gpu = layer.forward(x_cpu.to("cuda"), expert_ids.cuda(), route_w.cuda()).cpu()
    layer.set_m_cpu(10**9)
    y_all_cpu = layer.forward(x_cpu, expert_ids, route_w)

    r = scale_rel(y_all_cpu, y_all_gpu)
    print(f"  [test_dispatch_invariance]     all_cpu vs all_gpu scale_rel={r:.3e}")
    assert r <= 1.5e-2, f"dispatch-invariance scale_rel {r:.3e} > 1.5e-2"


def test_mixed_bucket_parity():
    """Mid-range m_cpu so some experts → CPU, some → GPU; vs all-CPU baseline."""
    torch.manual_seed(4)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=10**9)
    T = 96
    x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    # Concentrate ~T/2 tokens on expert 0 (→ GPU bucket); spread the rest.
    expert_ids = torch.zeros(T, top_k, dtype=torch.int64)
    expert_ids[:T // 2, 0] = 0
    expert_ids[:T // 2, 1] = 1
    expert_ids[T // 2:, 0] = torch.arange(T // 2) % G
    expert_ids[T // 2:, 1] = (torch.arange(T // 2) + 2) % G
    route_w = torch.softmax(torch.randn(T, top_k), dim=-1).float()

    y_all_cpu = layer.forward(x_cpu, expert_ids, route_w)
    layer.set_m_cpu(20)
    y_mixed = layer.forward(x_cpu.to("cuda"), expert_ids.cuda(), route_w.cuda()).cpu()

    r = scale_rel(y_mixed, y_all_cpu)
    print(f"  [test_mixed_bucket_parity]     mixed vs all_cpu scale_rel={r:.3e}")
    assert r <= 1.5e-2, f"mixed-bucket scale_rel {r:.3e} > 1.5e-2"


def test_adaptive_dispatch_parity():
    """Adaptive (cost-model) dispatch must be numerically indistinguishable
    from static dispatch: calibrate, run several adaptive forwards, compare
    each against the all-CPU baseline within the mixed-bucket envelope."""
    torch.manual_seed(5)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8)

    snap = layer.calibrate(t_points=(1, 8, 32), repeats=2)
    assert snap["cpu_obs"] > 0, f"calibration recorded no CPU obs: {snap}"
    assert snap["gpu_obs"] > 0, f"calibration recorded no GPU obs: {snap}"
    assert snap["pending_gpu"] == 0

    layer.set_adaptive(True)
    for T in (4, 32, 96):
        x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
        expert_ids = unique_topk_routing(T, G, top_k)
        route_w    = torch.softmax(torch.randn(T, top_k), dim=-1).float()

        y_adapt = layer.forward(x_cpu.to("cuda"), expert_ids.cuda(),
                                route_w.cuda()).cpu()
        layer.set_m_cpu(10**9)                       # all-CPU baseline
        y_cpu = layer.forward(x_cpu, expert_ids, route_w)
        layer.set_adaptive(True)                     # restore for next T

        r = scale_rel(y_adapt, y_cpu)
        print(f"  [test_adaptive_dispatch_parity] T={T:3d} adaptive vs "
              f"all_cpu scale_rel={r:.3e}")
        assert r <= 1.5e-2, f"T={T}: adaptive scale_rel {r:.3e} > 1.5e-2"

    # CPU-resident input in adaptive mode must route everything to the CPU
    # bucket (the GPU kernels need device tensors) and still be correct.
    x_cpu = torch.randn(16, H, dtype=torch.bfloat16)
    expert_ids = unique_topk_routing(16, G, top_k)
    route_w    = torch.softmax(torch.randn(16, top_k), dim=-1).float()
    y_adapt_cpu = layer.forward(x_cpu, expert_ids, route_w)
    layer.set_m_cpu(10**9)
    y_ref = layer.forward(x_cpu, expert_ids, route_w)
    r = scale_rel(y_adapt_cpu, y_ref)
    assert r <= 1e-6, f"CPU-input adaptive path diverged: {r:.3e}"
    print(f"  [test_adaptive_dispatch_parity] model snapshot: "
          f"{layer.dispatch.snapshot()}")


def test_adaptive_mixed_split_parity():
    """Adaptive dispatch with a *guaranteed mixed* CPU+GPU split.

    The plain parity test lets the fitted model choose the split, and on a
    fast host it may legitimately choose all-CPU for every T — leaving the
    mixed adaptive path untested. Here we plant coefficients with a known
    crossover and clear the observation rings (refit needs >= 4 obs, and we
    run fewer forwards than that), so the partition is deterministic:
    skewed routing (two 1-token experts, two heavy experts) must split
    2 CPU / 2 GPU.
    """
    torch.manual_seed(6)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8)
    layer.set_m_cpu(0)  # force the warm-up through the GPU bucket
    layer.forward(      # absorb the GPU JIT warm-up call (its timing is skipped)
        torch.randn(4, H, dtype=torch.bfloat16, device="cuda"),
        unique_topk_routing(4, G, top_k).cuda(),
        torch.full((4, top_k), 1.0 / top_k, device="cuda"))

    dm = layer.dispatch
    dm.harvest(wait=True)
    dm.cpu.coef = np.array([100e-6, 30e-6, 8e-6])     # per-row cost dominates
    dm.gpu.coef = np.array([150e-6, 120e-6, 0.05e-6]) # per-expert cost dominates
    dm.cpu.obs.clear()
    dm.gpu.obs.clear()
    dm.cpu._dirty = dm.gpu._dirty = False

    # Routing histogram [1, 1, T-1, T-1]: rows 0/1 hit the tiny experts once,
    # every row hits both heavy experts 2 and 3.
    T = 96
    expert_ids = torch.empty(T, top_k, dtype=torch.int64)
    expert_ids[:, 0] = 2
    expert_ids[:, 1] = 3
    expert_ids[0, 0] = 0
    expert_ids[1, 0] = 1
    m_list = [1, 1, T - 1, T - 1]

    # The planted model must choose a genuinely mixed split for this batch,
    # and forward() must see the same decision (rings are frozen).
    cpu_sel, gpu_sel = dm.partition(m_list)
    assert cpu_sel == [0, 1] and gpu_sel == [2, 3], \
        f"expected 2cpu/2gpu split, got {cpu_sel} / {gpu_sel}"

    x_cpu   = torch.randn(T, H, dtype=torch.bfloat16)
    route_w = torch.softmax(torch.randn(T, top_k), dim=-1).float()

    layer.set_adaptive(True)
    y_mixed = layer.forward(x_cpu.to("cuda"), expert_ids.cuda(),
                            route_w.cuda()).cpu()

    # Both backends really ran: the forward recorded one CPU observation and
    # one GPU event pair (obs rings were empty before it).
    dm.harvest(wait=True)
    snap = dm.snapshot()
    assert snap["cpu_obs"] == 1, f"CPU bucket did not run: {snap}"
    assert snap["gpu_obs"] == 1, f"GPU bucket did not run: {snap}"

    layer.set_m_cpu(10**9)                            # all-CPU baseline
    y_cpu = layer.forward(x_cpu, expert_ids, route_w)
    r = scale_rel(y_mixed, y_cpu)
    print(f"  [test_adaptive_mixed_split_parity] split={cpu_sel}cpu/{gpu_sel}gpu "
          f"scale_rel={r:.3e}")
    assert r <= 1.5e-2, f"mixed adaptive scale_rel {r:.3e} > 1.5e-2"


# --------------------------------------------------------------------------- #
# Pinned-residency tests (added with the SM90 INT8 grouped backend switch).
#
# The unified Layer keeps every expert byte in pinned host memory: the GPU
# kernel reads weights via Hopper TMA over PCIe (no VRAM mirror), and the
# AMX-packed twins are cudaHostRegister'd. These tests pin (pun intended)
# the contract that this is what is actually happening.
# --------------------------------------------------------------------------- #

def test_pinned_weight_pointer_identity():
    """The GPU kernel reads expert weights directly from the pinned host
    tensor — no VRAM copy. We prove this by mutating the pinned tensor
    between two otherwise-identical forwards and observing the output
    changes accordingly. If the kernel were reading a VRAM mirror, the
    second forward would be unaffected by the host mutation."""
    torch.manual_seed(11)
    G, H, I, top_k = 2, 256, 512, 1
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=0)
    T = 64
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    expert_ids = torch.zeros(T, 1, dtype=torch.int64, device="cuda")
    route_w    = torch.ones(T, 1, dtype=torch.float32, device="cuda")

    y_before = layer.forward(x, expert_ids, route_w).clone()

    # Mutate the pinned host tensor in place. If the GPU is reading from
    # VRAM the second forward will be unchanged; if it reads from the
    # pinned tensor, the output flips sign of the perturbed channels.
    with torch.no_grad():
        layer.slab.gate_int8[0].neg_()        # ← in-place on pinned host

    y_after = layer.forward(x, expert_ids, route_w)
    diff = (y_before.float() - y_after.float()).abs().mean()
    print(f"  [test_pinned_weight_pointer_identity] mean abs diff after host "
          f"mutation = {diff:.3e}  (must be > 0 — proves no VRAM mirror)")
    assert diff > 1e-3, (
        f"GPU appears to be reading from a VRAM cache, not pinned host "
        f"(mean diff after host mutation = {diff:.3e})"
    )


def test_no_vram_weight_residency():
    """Weight bytes (G·N·K) must not appear in VRAM after Layer construction.
    The only device-resident weight-adjacent tensors are the SFB broadcasts
    (G·N·K_blocks·4 bytes — much smaller). We check that the post-construction
    VRAM delta is well under the row-major weight size."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()

    torch.manual_seed(12)
    G, H, I, top_k = 4, 256, 512, 2
    gate = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    up   = torch.randn(G, I, H, dtype=torch.bfloat16) * 0.07
    down = torch.randn(G, H, I, dtype=torch.bfloat16) * 0.05
    layer = Layer.from_bf16(gate, up, down, top_k=top_k, cpu_threads=8, m_cpu=0)

    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated()
    delta = after_alloc - base_alloc

    # Reference sizes
    weight_bytes_per_proj = G * I * H * 1       # int8
    weight_bytes_total = 3 * weight_bytes_per_proj
    # Compact per-channel scale allocations only (3 projections × G × N × 4
    # bytes) — the [G, N, Kb] SFB broadcasts are expanded per use, not resident.
    sfb_bytes = (layer.slab.gate_s_dev.numel() + layer.slab.up_s_dev.numel()
                 + layer.slab.down_s_dev.numel()) * 4

    print(f"  [test_no_vram_weight_residency] VRAM delta = {delta} bytes  "
          f"(SFB only = {sfb_bytes}, weight bytes if mirrored = {weight_bytes_total})")
    # The VRAM delta should be no more than ~1.2× the SFB size (slack for
    # allocator rounding) — far below even a single mirrored projection.
    assert delta <= int(1.2 * sfb_bytes), (
        f"VRAM grew by {delta} bytes after Layer construction (SFB accounts "
        f"for {sfb_bytes}) — something weight-sized may be resident in VRAM "
        f"(full mirror would be {weight_bytes_total})"
    )

    # Confirm the user-visible markers reflect the v2 contract.
    assert layer.weight_residency == "pinned_host"
    assert layer.gpu_backend == "asym_gemm_sm90_int8_1d1d"


# -- Self-run convention (AsymGEMM scripts/test.sh runs `python <file>`) ----

if __name__ == "__main__":
    print(f"asym_gemm.unified_moe parity tests on {torch.cuda.get_device_name(0)}, "
          f"CPU caps AMX-INT8={_caps['has_amx_int8']}")
    test_cpu_vs_fp32()
    test_gpu_vs_fp32()
    test_cpu_vs_gpu_single_expert()
    test_dispatch_invariance()
    test_mixed_bucket_parity()
    test_adaptive_dispatch_parity()
    test_adaptive_mixed_split_parity()
    test_pinned_weight_pointer_identity()
    test_no_vram_weight_residency()
    print("All unified_moe parity + pinned-residency tests passed.")
