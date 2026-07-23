"""K-11 module-level micro before/after pairs + K-7/K-8 micro backfill.
Run ONLY in a clean window (GPU idle): .venv/bin/python tests/bench_modules.py"""
import os, time, torch, asym_gemm

def t(fn, iters=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e3

def main():
    M, I, H, R = 2_097_152, 768, 2048, 64
    Ma = 256_000  # attention rows @32k b8
    g = (torch.randn(M, I)*3).bfloat16(); u = (torch.randn(M, I)*3).bfloat16()
    a = torch.randn(M, I).bfloat16(); o = torch.empty_like(g)
    dg, du = torch.empty_like(g), torch.empty_like(g)
    print("== module micro (ms, 48T unless noted) ==")
    # RS-1 module: ATen-sequence (without) vs fused kernel (with)
    def aten_fwd():
        import torch.nn.functional as F
        with torch.no_grad(): o.copy_(F.silu(g).mul(u))
    print(f"moe.act fwd    : without(ATen)={t(aten_fwd,2):8.1f}  with(fused,48T)={t(lambda: asym_gemm.cpu_fused_silu_mul_bf16(g,u,o,48)):8.1f}")
    def aten_bwd():
        import torch.nn.functional as F
        with torch.no_grad():
            s_ = F.silu(g); du.copy_(a.mul(s_)); dg.copy_(torch.ops.aten.silu_backward(a.mul(u), g))
    print(f"moe.silu bwd   : without(ATen)={t(aten_bwd,2):8.1f}  with(fused,48T)={t(lambda: asym_gemm.cpu_fused_silu_backward_bf16(g,u,a,dg,du,48)):8.1f}")
    # RS-2/K-2 module: wgrad kernels at production shapes
    for (m, r, k, tag) in [(M, R, H, "moe.dA gate/up"), (M, R, I, "moe.dA down"), (Ma, R, H, "attn.dA q/k/v"), (M, I, R, "moe.dB")]:
        ds = torch.randn(m, r).bfloat16(); x = torch.randn(m, k).bfloat16()
        out = torch.zeros(1, r, k, dtype=torch.float32)
        pr = torch.tensor([0, m], dtype=torch.long); ge = torch.zeros(1, dtype=torch.long)
        print(f"{tag:15s}: with(CPU,48T)={t(lambda: asym_gemm.cpu_grouped_lora_a_grad_bf16(ds,x,out,pr,ge,48),3):8.1f}  (GPU-arm from nsys pair)")
    # K-7 backfill: rmsnorm kernel
    x = (torch.randn(Ma, H)*3).bfloat16(); w = torch.randn(H).bfloat16().abs(); on = torch.empty_like(x)
    def torch_norm():
        xf = x.float(); on.copy_((w.float()*(xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+1e-6))).bfloat16())
    print(f"attn.qknorm    : without(torch)={t(torch_norm,2):8.1f}  with(kernel,48T)={t(lambda: asym_gemm.cpu_rmsnorm_bf16(x,w,on,1e-6,48)):8.1f}")
    # K-8 backfill: widen+sqsum vs copy_ + separate norm pass
    src = torch.randn(1<<28).bfloat16(); dst = torch.empty(1<<28, dtype=torch.float32)
    def two_pass():
        dst.copy_(src); float((dst.double()**2).sum())
    print(f"opt.widen+norm : without(2pass)={t(two_pass,2):8.1f}  with(fused,48T)={t(lambda: asym_gemm.cpu_widen_bf16_sqsum(src,dst,48)):8.1f}")

def gpu_arms():
    """GPU 'without' arms + module-sequence pairs — CLEAN WINDOW ONLY (sole GPU user)."""
    assert torch.cuda.is_available()
    import torch.nn.functional as F
    dev = "cuda"
    M, I, H, R = 2_097_152, 768, 2048, 64
    Ma = 256_000

    def gt(fn, iters=5):
        fn(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters, (time.perf_counter() - t0) / iters * 1e3

    print("== GPU arms / module sequences (gpu-ms, wall-ms) ==")
    # K-1 module: boundary roundtrip pageable vs pinned
    hbm = torch.randn(Ma, H, device=dev, dtype=torch.bfloat16)
    def pageable_rt():
        c = hbm.to("cpu", non_blocking=True)   # unpinned => host-sync
        _ = c.to(dev, non_blocking=True)
    pin = torch.empty(Ma, H, dtype=torch.bfloat16, pin_memory=True)
    ev = torch.cuda.Event()
    def pinned_rt():
        pin.copy_(hbm, non_blocking=True); ev.record()
        torch.cuda.current_stream().wait_event(ev)
        _ = pin.to(dev, non_blocking=True)
    g1, w1 = gt(pageable_rt, 3); g2, w2 = gt(pinned_rt, 5)
    print(f"K-1 gc.boundary rt [256k,2048]: pageable gpu={g1:7.1f} wall={w1:7.1f} | pinned gpu={g2:7.1f} wall={w2:7.1f}")

    # RS-1 module SEQUENCE: old GPU-act staging chain vs fused-CPU + single stage
    gate_c = (torch.randn(M, I)*3).bfloat16().pin_memory()
    up_c = (torch.randn(M, I)*3).bfloat16().pin_memory()
    act_c = torch.empty(M, I, dtype=torch.bfloat16, pin_memory=True)
    gate_s = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    up_s = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    def old_seq():
        gate_s.copy_(gate_c, non_blocking=True)
        F.silu(gate_s, inplace=True)
        up_s.copy_(up_c, non_blocking=True)
        gate_s.mul_(up_s)
        act_c.copy_(gate_s, non_blocking=True)
        torch.cuda.synchronize()
    def new_seq():
        asym_gemm.cpu_fused_silu_mul_bf16(gate_c, up_c, act_c, 48)
        gate_s.copy_(act_c, non_blocking=True)  # single stage for down blocks
        torch.cuda.synchronize()
    g3, w3 = gt(old_seq, 3); g4, w4 = gt(new_seq, 3)
    print(f"RS-1 moe.act sequence [2.1M,768]: old(GPU-act,3 copies) gpu={g3:7.1f} wall={w3:7.1f} | new(CPU-act,1 copy) gpu={g4:7.1f} wall={w4:7.1f}")

    # RS-2/K-2 'without' arms: GPU cpu-right dA kernel
    kern = getattr(asym_gemm, "sm100_grouped_lora_a_grad_bf16_cpu_right", None)
    if kern is not None:
        for (m, r, k, tag) in [(M, R, H, "moe.dA gate/up"), (M, R, I, "moe.dA down"), (Ma, R, H, "attn.dA q/k/v")]:
            ds = torch.randn(m, r, device=dev, dtype=torch.bfloat16)
            xc = torch.randn(m, k, dtype=torch.bfloat16).pin_memory()
            ga = torch.empty(1, r, k, device=dev, dtype=torch.bfloat16)
            offs = torch.tensor([0, m], device=dev, dtype=torch.int32)
            exps = torch.tensor([0, -1], device=dev, dtype=torch.int32)
            gk, wk = gt(lambda: kern(ds, xc, ga, offs, exps, 2), 3)
            print(f"{tag:15s} GPU-arm: gpu={gk:7.1f} wall={wk:7.1f}")
    # dB GPU arm (single-group equivalent): dgate^T @ S on GPU with S staged H2D
    dgate_c = torch.randn(M, I, dtype=torch.bfloat16).pin_memory()
    s_c = torch.randn(M, R, dtype=torch.bfloat16).pin_memory()
    dgate_g = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    s_g = torch.empty(M, R, device=dev, dtype=torch.bfloat16)
    def db_gpu():
        s_g.copy_(s_c, non_blocking=True)
        _ = dgate_g.t().contiguous() @ s_g   # dgate already HBM in the real path
        torch.cuda.synchronize()
    g5, w5 = gt(db_gpu, 3)
    print(f"{'moe.dB':15s} GPU-arm(+S stage): gpu={g5:7.1f} wall={w5:7.1f}")


def fair3():
    """FAIR THREE-ARM module comparison (user directive 2026-07-17). For every CPU
    component, production-shaped tensors and three arms:
      A: CPU compute — inputs CPU-resident, our kernel, host wall-time (48T).
      B: GPU compute reading the input REMOTELY over C2C — input pinned in CPU
         memory, the existing cpu-right kernel, cuda-event time. n/a where no
         remote-read kernel exists (elementwise/norm classes).
      C: copy-then-GPU — H2D copy of the input (event-timed separately) + pure-GPU
         kernel on device-resident input (event-timed). Reports copy, compute,
         sum, and the realistic overlapped total max(copy, compute) (copy hidden
         iff a concurrent stream covers it).
    CLEAN WINDOW ONLY (sole GPU user). Numbers feed the fair-3-arm table in
    agent/impls/cpu_compute.md — production arm choice is about critical-path
    position and link/copy-engine contention, NOT raw isolated speed."""
    import torch.nn.functional as F

    assert torch.cuda.is_available()
    dev = "cuda"
    M, I, H, R = 2_097_152, 768, 2048, 64  # 32k*b8 expanded routed rows / ffn / hidden / rank
    Ma = 256_000                            # attention rows @32k*b8
    NT = 48

    def cpu_ms(fn, iters=3):
        fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e3

    def gpu_ms(fn, iters=5):
        fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    def row(name, a, b, c_copy, c_comp, note=""):
        c_sum = None if c_copy is None else c_copy + c_comp
        c_ovl = None if c_copy is None else max(c_copy, c_comp)
        fmt = lambda v: "   n/a " if v is None else f"{v:7.1f}"
        print(
            f"{name:24s} A={fmt(a)}  B={fmt(b)}  "
            f"C: copy={fmt(c_copy)} comp={fmt(c_comp)} sum={fmt(c_sum)} ovl={fmt(c_ovl)}  {note}"
        )

    print("== FAIR 3-ARM (ms; A=CPU-48T wall, B=GPU-remote-read event, C=copy-then-GPU event) ==")

    # ---- adapter weight-gradient (dA) family: dS^T @ X -------------------------
    kern_b = getattr(asym_gemm, "sm100_grouped_lora_a_grad_bf16_cpu_right", None)
    for (m, k, tag) in [(Ma, H, "attn.dA [256k,64]x2048"), (M, H, "moe.dA g/u [2.1M,64]x2048"), (M, I, "moe.dA down [2.1M,64]x768")]:
        ds_c = torch.randn(m, R, dtype=torch.bfloat16).pin_memory()
        x_c = torch.randn(m, k, dtype=torch.bfloat16).pin_memory()
        out_c = torch.zeros(1, R, k, dtype=torch.float32)
        pr = torch.tensor([0, m], dtype=torch.long)
        ge = torch.zeros(1, dtype=torch.long)
        a_ms = cpu_ms(lambda: asym_gemm.cpu_grouped_lora_a_grad_bf16(ds_c, x_c, out_c, pr, ge, NT))
        ds_g = ds_c.to(dev)
        b_ms = None
        if kern_b is not None:
            ga = torch.empty(1, R, k, device=dev, dtype=torch.bfloat16)
            offs = torch.tensor([0, m], device=dev, dtype=torch.int32)
            exps = torch.tensor([0, -1], device=dev, dtype=torch.int32)
            b_ms = gpu_ms(lambda: kern_b(ds_g, x_c, ga, offs, exps, 2))
        x_g = torch.empty(m, k, device=dev, dtype=torch.bfloat16)
        copy_ms = gpu_ms(lambda: x_g.copy_(x_c, non_blocking=True))
        x_g.copy_(x_c, non_blocking=True)
        dst = ds_g.t().contiguous()
        comp_ms = gpu_ms(lambda: dst @ x_g)
        row(tag, a_ms, b_ms, copy_ms, comp_ms)
        del ds_c, x_c, out_c, ds_g, x_g
        torch.cuda.empty_cache()

    # ---- SwiGLU forward: act = silu(gate)*up ----------------------------------
    gate_c = ((torch.randn(M, I) * 3).bfloat16()).pin_memory()
    up_c = ((torch.randn(M, I) * 3).bfloat16()).pin_memory()
    act_c = torch.empty(M, I, dtype=torch.bfloat16, pin_memory=True)
    a_ms = cpu_ms(lambda: asym_gemm.cpu_fused_silu_mul_bf16(gate_c, up_c, act_c, NT))
    gate_g = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    up_g = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    copy_ms = gpu_ms(lambda: (gate_g.copy_(gate_c, non_blocking=True), up_g.copy_(up_c, non_blocking=True)))
    gate_g.copy_(gate_c, non_blocking=True)
    up_g.copy_(up_c, non_blocking=True)
    o_g = torch.empty_like(gate_g)
    def _gpu_fwd():
        torch.mul(F.silu(gate_g), up_g, out=o_g)
    comp_ms = gpu_ms(_gpu_fwd)
    row("moe.swiglu fwd [2.1M,768]", a_ms, None, copy_ms, comp_ms, "(B n/a: no remote-read elementwise kernel)")

    # ---- SwiGLU backward: dgate/dup from gate,up,dact -------------------------
    dact_c = torch.randn(M, I, dtype=torch.bfloat16).pin_memory()
    dg_c = torch.empty(M, I, dtype=torch.bfloat16, pin_memory=True)
    du_c = torch.empty(M, I, dtype=torch.bfloat16, pin_memory=True)
    a_ms = cpu_ms(lambda: asym_gemm.cpu_fused_silu_backward_bf16(gate_c, up_c, dact_c, dg_c, du_c, NT))
    dact_g = dact_c.to(dev)  # dact is GPU-born in production; only gate/up need H2D
    def _gpu_bwd():
        s_ = F.silu(gate_g)
        du = dact_g.mul(s_)
        dg = torch.ops.aten.silu_backward(dact_g.mul(up_g), gate_g)
        return du, dg
    comp_ms = gpu_ms(_gpu_bwd)
    row("moe.swiglu bwd [2.1M,768]", a_ms, None, copy_ms, comp_ms, "(B n/a; copy = gate+up H2D, dact GPU-born)")
    del gate_c, up_c, act_c, dact_c, dg_c, du_c, gate_g, up_g, o_g, dact_g
    torch.cuda.empty_cache()

    # ---- qk-norm: recompute vs the save-reload roundtrip it replaces ----------
    B_, S_, Hh, D_ = 8, 32000, 32, 128
    rows = B_ * S_ * Hh
    x_c = ((torch.randn(rows, D_) * 2).bfloat16()).pin_memory()
    w = torch.randn(D_).float().abs()
    on_c = torch.empty_like(x_c)
    a_ms = cpu_ms(lambda: asym_gemm.cpu_rmsnorm_bf16(x_c, w, on_c, 1e-6, NT))
    x_g = torch.empty(rows, D_, device=dev, dtype=torch.bfloat16)
    copy_ms = gpu_ms(lambda: x_g.copy_(x_c, non_blocking=True))  # bf16 input H2D (2.1 GB)
    x_g.copy_(x_c, non_blocking=True)
    w_g = w.to(dev)
    def _gpu_norm():
        v = x_g.float().pow(2).mean(-1, keepdim=True)
        return (x_g.float() * torch.rsqrt(v + 1e-6) * w_g).to(torch.bfloat16)
    comp_ms = gpu_ms(_gpu_norm, iters=3)
    row("attn.qknorm [8.2M,128]", a_ms, None, copy_ms, comp_ms, "(C = the R2 production path: bf16 H2D + exact GPU recompute)")
    # the replaced baseline: save+reload roundtrip of the TWO fp32 upcasts (copy-only)
    f32_c = torch.empty(rows, D_, dtype=torch.float32, pin_memory=True)
    f32_g = torch.randn(rows, D_, device=dev, dtype=torch.float32)
    d2h_ms = gpu_ms(lambda: f32_c.copy_(f32_g, non_blocking=True), iters=3)
    h2d_ms = gpu_ms(lambda: f32_g.copy_(f32_c, non_blocking=True), iters=3)
    print(
        f"{'  replaced save-reload':24s} 2x fp32 roundtrip copy-only: D2H={2*d2h_ms:7.1f} H2D={2*h2d_ms:7.1f} "
        f"sum={2*(d2h_ms+h2d_ms):7.1f}  (per q-norm per layer @32k*b8; k-norm = /8)"
    )
    del x_c, on_c, x_g, f32_c, f32_g
    torch.cuda.empty_cache()

    # ---- gc boundary copy (kept from the K-1 pair; fairness = copy-mode arms) --
    hbm = torch.randn(Ma, H, device=dev, dtype=torch.bfloat16)
    def pageable_rt():
        c = hbm.to("cpu", non_blocking=True)
        _ = c.to(dev, non_blocking=True)
    pin = torch.empty(Ma, H, dtype=torch.bfloat16, pin_memory=True)
    ev = torch.cuda.Event()
    def pinned_rt():
        pin.copy_(hbm, non_blocking=True)
        ev.record()
        torch.cuda.current_stream().wait_event(ev)
        _ = pin.to(dev, non_blocking=True)
    pg = cpu_ms(pageable_rt, iters=3)   # pageable => host-blocking: wall IS the cost
    pn = gpu_ms(pinned_rt)
    print(f"{'gc.boundary rt [256k,2048]':24s} pageable(wall)={pg:7.1f}  pinned(event)={pn:7.1f}  no-copy=0.0")


def final_table():
    """MODULE MICROBENCH — FINAL (user mandate 2026-07-20): one row per adoptable
    CPU-compute optimization, isolated clean-window measurements at production
    shapes for BOTH regimes (30B@32k×b8 and 30B@128k×b8), with per-STEP aggregates
    (per-invocation ms × invocations/step × 48 layers). CLEAN WINDOW ONLY."""
    import torch.nn.functional as F

    assert torch.cuda.is_available()
    dev = "cuda"
    NT = 48
    L = 48  # layers (30B)

    def cpu_ms(fn, iters=3):
        fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e3

    def gpu_ms(fn, iters=5):
        fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    R = 64
    rows = {}

    def swiglu_shapes(M):
        g = (torch.randn(M, 768) * 3).bfloat16().pin_memory()
        u = (torch.randn(M, 768) * 3).bfloat16().pin_memory()
        return g, u

    # ---- rows 1+2: SwiGLU fwd/bwd (ours vs PyTorch-CPU vs copy+GPU) ----------
    for label, M in (("32k", 2_097_152), ("128k", 8_388_608)):
        g_c, u_c = swiglu_shapes(M)
        o_c = torch.empty(M, 768, dtype=torch.bfloat16, pin_memory=True)
        d_c = torch.randn(M, 768, dtype=torch.bfloat16).pin_memory()
        dg_c = torch.empty_like(o_c)
        du_c = torch.empty_like(o_c)

        def aten_fwd():
            with torch.no_grad():
                o_c.copy_(F.silu(g_c).mul(u_c))

        def aten_bwd():
            with torch.no_grad():
                s_ = F.silu(g_c)
                du_c.copy_(d_c.mul(s_))
                dg_c.copy_(torch.ops.aten.silu_backward(d_c.mul(u_c), g_c))

        ours_f = cpu_ms(lambda: asym_gemm.cpu_fused_silu_mul_bf16(g_c, u_c, o_c, NT))
        aten_f = cpu_ms(aten_fwd, iters=2)
        ours_b = cpu_ms(lambda: asym_gemm.cpu_fused_silu_backward_bf16(g_c, u_c, d_c, dg_c, du_c, NT))
        aten_b = cpu_ms(aten_bwd, iters=2)
        g_g = torch.empty(M, 768, device=dev, dtype=torch.bfloat16)
        u_g = torch.empty(M, 768, device=dev, dtype=torch.bfloat16)
        cp = gpu_ms(lambda: (g_g.copy_(g_c, non_blocking=True), u_g.copy_(u_c, non_blocking=True)))
        o_g = torch.empty_like(g_g)
        cf = gpu_ms(lambda: torch.mul(F.silu(g_g), u_g, out=o_g))
        d_g = d_c.to(dev)
        cb = gpu_ms(lambda: (d_g.mul(F.silu(g_g)), torch.ops.aten.silu_backward(d_g.mul(u_g), g_g)))
        rows[f"swiglu_fwd_{label}"] = dict(ours=ours_f, aten=aten_f, copy=cp, gpu=cf)
        rows[f"swiglu_bwd_{label}"] = dict(ours=ours_b, aten=aten_b, copy=cp, gpu=cb)
        del g_c, u_c, o_c, d_c, dg_c, du_c, g_g, u_g, o_g, d_g
        torch.cuda.empty_cache()

    # ---- rows 3+4: adapter-grad kernel (CPU vs GPU-remote vs copy+GPU) -------
    kern_b = getattr(asym_gemm, "sm100_grouped_lora_a_grad_bf16_cpu_right", None)
    for name, m, k in (
        ("attn_dA_32k", 256_000, 2048),
        ("attn_dA_128k", 1_024_000, 2048),
        ("moe_dA_gu_32k", 2_097_152, 2048),
        ("moe_dA_gu_128k", 8_388_608, 2048),
        ("moe_dA_down_32k", 2_097_152, 768),
        ("moe_dA_down_128k", 8_388_608, 768),
    ):
        ds_c = torch.randn(m, R, dtype=torch.bfloat16).pin_memory()
        x_c = torch.randn(m, k, dtype=torch.bfloat16).pin_memory()
        out_c = torch.zeros(1, R, k, dtype=torch.float32)
        pr = torch.tensor([0, m], dtype=torch.long)
        ge = torch.zeros(1, dtype=torch.long)
        a = cpu_ms(lambda: asym_gemm.cpu_grouped_lora_a_grad_bf16(ds_c, x_c, out_c, pr, ge, NT))
        ds_g = ds_c.to(dev)
        b = None
        if kern_b is not None:
            ga = torch.empty(1, R, k, device=dev, dtype=torch.bfloat16)
            offs = torch.tensor([0, m], device=dev, dtype=torch.int32)
            exps = torch.tensor([0, -1], device=dev, dtype=torch.int32)
            b = gpu_ms(lambda: kern_b(ds_g, x_c, ga, offs, exps, 2), iters=3)
        x_g = torch.empty(m, k, device=dev, dtype=torch.bfloat16)
        cp = gpu_ms(lambda: x_g.copy_(x_c, non_blocking=True), iters=3)
        x_g.copy_(x_c, non_blocking=True)
        dst = ds_g.t().contiguous()
        cc = gpu_ms(lambda: dst @ x_g, iters=3)
        rows[name] = dict(cpu=a, remote=b, copy=cp, gpu=cc)
        del ds_c, x_c, out_c, ds_g, x_g, dst
        torch.cuda.empty_cache()

    # ---- row 5: norm(+rope) recompute vs save+reload roundtrip ---------------
    for label, S in (("32k", 32000), ("128k", 128000)):
        B_, Hh, D_ = 8, 32, 128
        rws = B_ * S * Hh
        x_c = ((torch.randn(rws, D_) * 2).bfloat16()).pin_memory()
        w = torch.randn(D_).float().abs()
        on_c = torch.empty_like(x_c)
        a = cpu_ms(lambda: asym_gemm.cpu_rmsnorm_bf16(x_c, w, on_c, 1e-6, NT))
        x_g = torch.empty(rws, D_, device=dev, dtype=torch.bfloat16)
        cp = gpu_ms(lambda: x_g.copy_(x_c, non_blocking=True), iters=3)
        x_g.copy_(x_c, non_blocking=True)
        w_g = w.to(dev)

        def gpu_norm():
            v = x_g.float().pow(2).mean(-1, keepdim=True)
            return (x_g.float() * torch.rsqrt(v + 1e-6) * w_g).to(torch.bfloat16)

        cn = gpu_ms(gpu_norm, iters=3)
        q4 = x_g.view(B_, S, Hh, D_).transpose(1, 2)
        cos = torch.randn(B_, S, D_, device=dev, dtype=torch.bfloat16)
        sin = torch.randn(B_, S, D_, device=dev, dtype=torch.bfloat16)

        def rot(x):
            x1 = x[..., : D_ // 2]
            x2 = x[..., D_ // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cr = gpu_ms(lambda: q4 * cos.unsqueeze(1) + rot(q4) * sin.unsqueeze(1), iters=3)
        f32_c = torch.empty(rws, D_, dtype=torch.float32, pin_memory=True)
        f32_g = torch.randn(rws, D_, device=dev, dtype=torch.float32)
        d2h = gpu_ms(lambda: f32_c.copy_(f32_g, non_blocking=True), iters=3)
        h2d = gpu_ms(lambda: f32_g.copy_(f32_c, non_blocking=True), iters=3)
        bf_c = torch.empty(rws, D_, dtype=torch.bfloat16, pin_memory=True)
        bf_g = torch.randn(rws, D_, device=dev, dtype=torch.bfloat16)
        bd2h = gpu_ms(lambda: bf_c.copy_(bf_g, non_blocking=True), iters=3)
        bh2d = gpu_ms(lambda: bf_g.copy_(bf_c, non_blocking=True), iters=3)
        rows[f"norm_{label}"] = dict(
            cpu=a, copy=cp, gpu=cn, rope=cr,
            fp32_rt=2 * (d2h + h2d), bf16_rt=bd2h + bh2d,
        )
        del x_c, on_c, x_g, q4, cos, sin, f32_c, f32_g, bf_c, bf_g
        torch.cuda.empty_cache()

    # ---- rows 6+7: boundary copy + restage waited vs prefetched --------------
    from asym_gemm.training import activation_offload as ao

    os.environ["ASYMM_ATTN_RESTAGE_PREFETCH"] = "1"
    for name, m, k in (("boundary_32k", 256_000, 2048), ("boundary_128k", 1_024_000, 2048),
                       ("restage_32k", 2_097_152, 768), ("restage_128k", 8_388_608, 768)):
        hbm = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

        def pageable_rt():
            c = hbm.to("cpu", non_blocking=True)
            _ = c.to(dev, non_blocking=True)

        pg = cpu_ms(pageable_rt, iters=2)
        mgr = ao.ActivationOffloadManager(pin_memory=True)
        h = mgr.offload(hbm.reshape(m, k), "bench.pref")
        torch.cuda.synchronize()
        # waited (lazy) arm: measure the H2D exposure via the counters
        ao.reset_restage_gap_for_tests()
        st = mgr.stage(h, tag="bench.wait")
        torch.cuda.synchronize()
        waited = ao.restage_gap_stats()["total_exposed_ms"]
        mgr.release_stage(st, drop_cache=True)
        # prefetched arm: begin -> overlap compute -> commit
        ao.reset_restage_gap_for_tests()
        big = torch.randn(6144, 6144, device=dev)
        stage, done = mgr.stage_begin(h, tag="bench.pref")
        for _ in range(8):
            big = big @ big * 1e-6  # ~the down-block window
        out = mgr.stage_commit(stage, done, nbytes=h.nbytes, tag="bench.pref")
        torch.cuda.synchronize()
        prefetched = ao.restage_gap_stats()["total_exposed_ms"]
        mgr.release_stage(out, drop_cache=True)
        mgr.release_cpu(h)
        rows[name] = dict(pageable=pg, waited=waited, prefetched=prefetched)
        del hbm, big
        torch.cuda.empty_cache()
    os.environ["ASYMM_ATTN_RESTAGE_PREFETCH"] = "0"

    # ---- row 8: optimizer drain widen+sqsum ----------------------------------
    src = torch.randn(1 << 28).bfloat16()
    dst = torch.empty(1 << 28, dtype=torch.float32)

    def two_pass():
        dst.copy_(src)
        float((dst.double() ** 2).sum())

    rows["opt_widen"] = dict(
        unfused=cpu_ms(two_pass, iters=2),
        fused=cpu_ms(lambda: asym_gemm.cpu_widen_bf16_sqsum(src, dst, NT)),
    )
    del src, dst

    # ---- row 9: dedup pack (dup pack cost vs shared-handle) ------------------
    from asym_gemm.training.attention_activation_offload import _empty_strided_cpu_like

    var32 = torch.randn(8, 32000, 32, 1, device=dev, dtype=torch.float32)

    def dup_pack():
        cpu = _empty_strided_cpu_like(var32, pin_memory=True)
        with torch.no_grad():
            cpu.copy_(var32.detach(), non_blocking=cpu.is_pinned())
        torch.cuda.synchronize()

    shared = {}

    def dedup_hit():
        shared.get(("k", 0))  # the dedup hit is one dict lookup + refcount

    rows["dedup_pack"] = dict(dup=gpu_ms(dup_pack, iters=5), shared=cpu_ms(dedup_hit, iters=100))

    # ---- emit ----------------------------------------------------------------
    print("== MODULE MICROBENCH — FINAL (ms per invocation at the stated shape) ==")
    for k in sorted(rows):
        print(f"{k:18s} " + "  ".join(f"{a}={v:9.2f}" if v is not None else f"{a}=      n/a" for a, v in rows[k].items()))
    import json as _json

    with open("/tmp/bench_final_table.json", "w") as f:
        _json.dump(rows, f, indent=1)
    print("saved /tmp/bench_final_table.json")


if __name__ == "__main__":
    import sys

    if "--fair3" in sys.argv:
        fair3()
    elif "--final" in sys.argv:
        final_table()
    else:
        main()
        if torch.cuda.is_available():
            gpu_arms()
        fair3()
