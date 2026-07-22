"""Parity + microbench for csrc/cpu_ops fused SwiGLU kernels (cpu_compute.md Stage 1).

Reference = the exact ATen sequences these kernels replace:
  fwd: qwen3_moe.py:900-912   out.copy_(F.silu(gate).mul(up))
  bwd: qwen3_moe.py:915-933   grad_up = grad_act.mul(silu(gate));
                              grad_gate = aten.silu_backward(grad_act.mul(up), gate)

Run: .venv/bin/python -m pytest tests/test_cpu_ops.py -v
Bench: .venv/bin/python tests/test_cpu_ops.py bench
"""

import sys
import time

import torch
import torch.nn.functional as F

import asym_gemm

BF = torch.bfloat16


def _ref_fwd(gate, up):
    return F.silu(gate.float()).mul(up.float()).to(BF)


def _ref_bwd(gate, up, ga):
    g32, u32, a32 = gate.float(), up.float(), ga.float()
    dup = a32.mul(F.silu(g32)).to(BF)
    dgate = torch.ops.aten.silu_backward(a32.mul(u32), g32).to(BF)
    return dgate, dup


def _max_ulp_bf16(a, b):
    # distance in bf16 representable steps (monotonic int16 mapping)
    ai = a.view(torch.int16).int()
    bi = b.view(torch.int16).int()
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs().max().item()


def test_sve_compiled():
    assert asym_gemm.cpu_ops_sve_compiled(), "expected SVE-BF16 build on Grace"


def test_fwd_parity():
    torch.manual_seed(0)
    for shape in [(1,), (7,), (128, 768), (1000, 33), (4096, 768)]:
        gate = (torch.randn(*shape) * 4).to(BF)
        up = (torch.randn(*shape) * 4).to(BF)
        out = torch.empty_like(gate)
        asym_gemm.cpu_fused_silu_mul_bf16(gate, up, out, 8)
        ref = _ref_fwd(gate, up)
        assert _max_ulp_bf16(out, ref) <= 1, f"fwd mismatch at {shape}"


def test_bwd_parity():
    torch.manual_seed(1)
    for shape in [(1,), (63,), (128, 768), (999, 65), (4096, 768)]:
        gate = (torch.randn(*shape) * 4).to(BF)
        up = (torch.randn(*shape) * 4).to(BF)
        ga = (torch.randn(*shape) * 2).to(BF)
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(gate)
        asym_gemm.cpu_fused_silu_backward_bf16(gate, up, ga, dgate, dup, 8)
        rg, ru = _ref_bwd(gate, up, ga)
        assert _max_ulp_bf16(dup, ru) <= 1, f"dup mismatch at {shape}"
        assert _max_ulp_bf16(dgate, rg) <= 2, f"dgate mismatch at {shape}"


def test_extremes():
    # large |x| saturation, zeros, denormal-ish values
    vals = torch.tensor([-100.0, -20.0, -1e-3, 0.0, 1e-3, 20.0, 100.0]).to(BF)
    gate = vals.repeat(1024)
    up = torch.ones_like(gate)
    ga = torch.ones_like(gate)
    out = torch.empty_like(gate)
    asym_gemm.cpu_fused_silu_mul_bf16(gate, up, out, 4)
    assert _max_ulp_bf16(out, _ref_fwd(gate, up)) <= 1
    dgate = torch.empty_like(gate)
    dup = torch.empty_like(gate)
    asym_gemm.cpu_fused_silu_backward_bf16(gate, up, ga, dgate, dup, 4)
    rg, ru = _ref_bwd(gate, up, ga)
    assert _max_ulp_bf16(dup, ru) <= 1
    assert _max_ulp_bf16(dgate, rg) <= 2


def bench():
    M, I = 2_097_152, 768  # q3-30b-a3b @32k b8 shapes
    for pin in ((False, True) if torch.cuda.is_available() else (False,)):
        gate = (torch.randn(M, I) * 3).to(BF)
        up = (torch.randn(M, I) * 3).to(BF)
        ga = (torch.randn(M, I)).to(BF)
        if pin:
            gate, up, ga = gate.pin_memory(), up.pin_memory(), ga.pin_memory()
        out, dgate, dup = (torch.empty_like(gate) for _ in range(3))
        for nt in (8, 16, 32, 48, 64):
            # fused fwd
            asym_gemm.cpu_fused_silu_mul_bf16(gate, up, out, nt)  # warm
            t0 = time.perf_counter()
            for _ in range(3):
                asym_gemm.cpu_fused_silu_mul_bf16(gate, up, out, nt)
            fwd = (time.perf_counter() - t0) / 3
            # fused bwd
            asym_gemm.cpu_fused_silu_backward_bf16(gate, up, ga, dgate, dup, nt)
            t0 = time.perf_counter()
            for _ in range(3):
                asym_gemm.cpu_fused_silu_backward_bf16(gate, up, ga, dgate, dup, nt)
            bwd = (time.perf_counter() - t0) / 3
            gbf = 3 * M * I * 2 / fwd / 1e9
            gbb = 5 * M * I * 2 / bwd / 1e9
            print(f"pin={pin} nt={nt:2d}: fwd {fwd*1e3:7.1f} ms ({gbf:5.0f} GB/s) | "
                  f"bwd {bwd*1e3:7.1f} ms ({gbb:5.0f} GB/s)")
    # ATen reference timing (multi-pass, default threads)
    t0 = time.perf_counter(); _ref_fwd(gate, up); print(f"ATen fwd ref {time.perf_counter()-t0:.3f} s")
    t0 = time.perf_counter(); _ref_bwd(gate, up, ga); print(f"ATen bwd ref {time.perf_counter()-t0:.3f} s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench()
    else:
        test_sve_compiled(); test_fwd_parity(); test_bwd_parity(); test_extremes()
        print("all parity tests passed")


def test_wgrad_single_group_and_db_swap():
    # K-2/K-3: single-group (E=1) attention/dense shapes; K-5: dB via operand swap
    torch.manual_seed(3)
    for (m, r, k) in [(1000, 64, 2048), (777, 64, 5120), (4096, 768, 64)]:
        ds = (torch.randn(m, r) * 2).bfloat16()
        x = (torch.randn(m, k) * 2).bfloat16()
        out = torch.zeros(1, r, k, dtype=torch.float32)
        pairs = torch.tensor([0, m], dtype=torch.long)
        ge = torch.zeros(1, dtype=torch.long)
        asym_gemm.cpu_grouped_lora_a_grad_bf16(ds, x, out, pairs, ge, 8)
        ref = ds.float().t() @ x.float()
        d = (out[0] - ref).abs().max().item() / max(1.0, ref.abs().max().item())
        assert d < 2e-2, (m, r, k, d)


def test_wgrad_skewed_groups():
    torch.manual_seed(4)
    m, r, k, e = 5000, 64, 512, 8
    ds = (torch.randn(m, r)).bfloat16()
    x = (torch.randn(m, k)).bfloat16()
    bounds = [0, 10, 10, 4000, 4200, 4200, 4999, 5000]  # incl. empty groups
    pairs = torch.tensor(sum(([bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)), []), dtype=torch.long)
    ge = torch.arange(len(bounds) - 1, dtype=torch.long)
    out = torch.zeros(e, r, k, dtype=torch.float32)
    asym_gemm.cpu_grouped_lora_a_grad_bf16(ds, x, out, pairs, ge, 16)
    for g in range(len(bounds) - 1):
        m0, m1 = bounds[g], bounds[g + 1]
        ref = ds[m0:m1].float().t() @ x[m0:m1].float() if m1 > m0 else torch.zeros(r, k)
        d = (out[g] - ref).abs().max().item() / max(1.0, ref.abs().max().item())
        assert d < 2e-2, (g, d)
