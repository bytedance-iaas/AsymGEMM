import torch, asym_gemm
from asym_gemm.training.frozen_linear import _group_metadata_tensors
DEV = "cuda"

def make(G, mpg, n, k):
    m = G * mpg
    offsets = torch.arange(0, m + 1, mpg, dtype=torch.long)
    experts = torch.arange(0, G + 1, dtype=torch.long)
    off, exp, ls = _group_metadata_tensors(offsets.to(DEV), experts.to(DEV), device=DEV)
    a = torch.randn(m, k, dtype=torch.bfloat16, device=DEV) * 0.1
    b = torch.randn(G, n, k, dtype=torch.bfloat16, device=DEV) * 0.1
    d = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
    return dict(m=m, G=G, n=n, k=k, a=a, b=b, d=d, off=off, exp=exp, ls=ls)

def asym(x):
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(x["a"], x["b"], x["d"], x["off"], x["exp"], x["ls"], "mnk", False)

def cublas(x):
    ag = x["a"].view(x["G"], x["m"] // x["G"], x["k"])
    return torch.bmm(ag, x["b"].transpose(1, 2))

def bench(fn, x, warmup=10, reps=6, iters=30):
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

print(f"GPU {torch.cuda.get_device_name()}")
print("compute-bound grouped D=A@B.T, G=8.  TFLOP/s and asym-vs-cuBLAS ratio:\n")
print(f"{'m':>6} {'n':>6} {'k':>6} | {'asym_ms':>8} {'asymTF':>7} | {'cublas_ms':>9} {'cubTF':>7} | {'asym/cublas':>11}")
for (mpg, n, k) in [(256, 2048, 2048), (256, 4096, 4096), (256, 8192, 4096), (512, 8192, 8192), (1024, 8192, 8192)]:
    x = make(8, mpg, n, k)
    flop = 2 * x["m"] * n * k
    ta = bench(asym, x)
    tc = bench(cublas, x)
    tfa = flop / (ta / 1e3) / 1e12
    tfc = flop / (tc / 1e3) / 1e12
    print(f"{x['m']:>6} {n:>6} {k:>6} | {ta:>7.3f}m {tfa:>6.0f}T | {tc:>8.3f}m {tfc:>6.0f}T | {ta/tc:>10.2f}x")
print("\nasymTF/cubTF<1 in last col means asym slower; gap ~ latency headroom vs cuBLAS.")
