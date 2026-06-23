import torch, asym_gemm
from asym_gemm.training.frozen_linear import _group_metadata_tensors
DEV = "cuda"

def make(G, mpg, n, k):
    m = G * mpg
    offsets = torch.arange(0, m + 1, mpg, dtype=torch.long)
    experts = torch.arange(0, G + 1, dtype=torch.long)
    off, exp, ls = _group_metadata_tensors(offsets.to(DEV), experts.to(DEV), device=DEV)
    return dict(m=m, G=G, n=n, k=k,
                a_cpu=torch.randn(m, k, dtype=torch.bfloat16).pin_memory(),
                a_gpu=torch.empty(m, k, dtype=torch.bfloat16, device=DEV),
                b=torch.randn(G, n, k, dtype=torch.bfloat16, device=DEV) * 0.1,
                d=torch.empty(m, n, dtype=torch.bfloat16, device=DEV),
                off=off, exp=exp, ls=ls)

def cl(x): asym_gemm.sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(x["a_cpu"], x["b"], x["d"], x["off"], x["exp"], x["ls"], "nk")
def mm(x): asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(x["a_gpu"], x["b"], x["d"], x["off"], x["exp"], x["ls"], "mnk", False)
def cp(x): x["a_gpu"].copy_(x["a_cpu"], non_blocking=True)
def staged(x): cp(x); mm(x)

def bench(fn, x, warmup=8, reps=6, iters=30):
    for _ in range(warmup): fn(x)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters): fn(x)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best

print(f"GPU {torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()}")

# ---- C2C bandwidth vs transfer size ----
print("\n== C2C H2D bandwidth (pinned CPU -> HBM) vs size ==")
print(f"{'MB':>7} | {'ms':>8} | {'GB/s':>6}")
for mb in [8, 32, 128, 512, 1024, 2048]:
    nelem = mb * 1024 * 1024 // 2
    a = torch.empty(nelem, dtype=torch.bfloat16).pin_memory()
    g = torch.empty(nelem, dtype=torch.bfloat16, device=DEV)
    f = lambda _x=None: g.copy_(a, non_blocking=True)
    t = bench(f, None, warmup=5, reps=6, iters=10)
    print(f"{mb:>7} | {t:>7.3f}m | {mb/1024/(t/1e3):>5.0f}")
    del a, g; torch.cuda.empty_cache()

# ---- N sweep (fixed m,k): from LoRA-rank to large ----
G, mpg, k = 8, 256, 2048
print(f"\n== N sweep (G={G}, m={G*mpg}, k={k}) :  speedup = staged/cpu_left (>1 = cpu_left wins) ==")
print(f"{'n':>6} | {'cpu_left':>9} | {'copy':>7} | {'gpu_mm':>7} | {'staged':>7} | {'speedup':>7}")
for n in [16, 64, 128, 256, 512, 1024, 2048, 4096]:
    x = make(G, mpg, n, k)
    tcl, tcp, tmm, tst = bench(cl, x), bench(cp, x), bench(mm, x), bench(staged, x)
    print(f"{n:>6} | {tcl:>8.3f}m | {tcp:>6.3f}m | {tmm:>6.3f}m | {tst:>6.3f}m | {tst/tcl:>6.2f}x")

# ---- M sweep (fixed n,k): occupancy effect on cpu_left ----
n, k = 64, 2048   # LoRA-A-like small n
print(f"\n== M sweep (G={G}, n={n} [LoRA-rank], k={k}) :  more tokens -> more M-tiles -> cpu_left occupancy ==")
print(f"{'m':>7} | {'Mtiles':>6} | {'cpu_left':>9} | {'copy':>7} | {'gpu_mm':>7} | {'staged':>7} | {'speedup':>7}")
for mpg2 in [128, 512, 1024, 2048, 4096]:
    x = make(G, mpg2, n, k)
    tcl, tcp, tmm, tst = bench(cl, x), bench(cp, x), bench(mm, x), bench(staged, x)
    print(f"{x['m']:>7} | {x['m']//128:>6} | {tcl:>8.3f}m | {tcp:>6.3f}m | {tmm:>6.3f}m | {tst:>6.3f}m | {tst/tcl:>6.2f}x")
print("\nms per call (median-of-best). copy=CPU->HBM H2D. staged=copy+gpu_mm.")
