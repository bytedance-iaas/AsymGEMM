"""Benchmark: grouped D = A @ B.T where A starts on (pinned) CPU.

M1  asym_cpu_left : fused CPU->SMEM AsymGEMM kernel (A pinned on CPU)
M2  staged        : copy A CPU->HBM, then GPU-resident grouped AsymGEMM
also: gpu_mm_only  : the GPU grouped GEMM assuming A already resident (lower bound)
      stage+torch  : copy + per-group cuBLAS loop (sanity baseline)
"""
import torch, asym_gemm
from asym_gemm.training.frozen_linear import _group_metadata_tensors

DEV = "cuda"

def make_inputs(G, mpg, n, k):
    m = G * mpg
    offsets = torch.arange(0, m + 1, mpg, dtype=torch.long)   # cumulative boundaries, len G+1
    experts = torch.arange(0, G + 1, dtype=torch.long)        # len G+1
    a_cpu = torch.randn(m, k, dtype=torch.bfloat16).pin_memory()
    a_gpu = torch.empty(m, k, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(G, n, k, dtype=torch.bfloat16, device=DEV) * 0.1
    d = torch.empty(m, n, dtype=torch.bfloat16, device=DEV)
    off_i32, exp_i32, ls = _group_metadata_tensors(offsets.to(DEV), experts.to(DEV), device=DEV)
    return dict(m=m, G=G, n=n, k=k, a_cpu=a_cpu, a_gpu=a_gpu, b=b, d=d,
                off=off_i32, exp=exp_i32, ls=ls, offsets=offsets)

def f_cpu_left(x):
    asym_gemm.sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous(
        x["a_cpu"], x["b"], x["d"], x["off"], x["exp"], x["ls"], "nk")

def f_gpu_mm(x):
    asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(
        x["a_gpu"], x["b"], x["d"], x["off"], x["exp"], x["ls"], "mnk", False)

def f_copy(x):
    x["a_gpu"].copy_(x["a_cpu"], non_blocking=True)

def f_staged(x):
    x["a_gpu"].copy_(x["a_cpu"], non_blocking=True)
    f_gpu_mm(x)

def bench(fn, x, warmup=8, reps=6, iters=20):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn(x)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best  # ms

def correctness(x):
    f_cpu_left(x); d_cl = x["d"].clone()
    f_copy(x); torch.cuda.synchronize(); f_gpu_mm(x); d_gpu = x["d"].clone()
    ref = torch.empty_like(d_gpu)
    ag = x["a_cpu"].to(DEV)
    for g in range(x["G"]):
        s, e = x["offsets"][g].item(), x["offsets"][g + 1].item()
        ref[s:e] = (ag[s:e].float() @ x["b"][g].float().t()).bfloat16()
    def rel(a):
        return (a.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
    return rel(d_cl), rel(d_gpu)

print(f"GPU: {torch.cuda.get_device_name()}  cap={torch.cuda.get_device_capability()}")
G, mpg, k = 8, 256, 2048   # 8 experts, 256 tokens each (m=2048), k=2048
print(f"\nFixed: G={G} experts, m={G*mpg} tokens, k={k}.  Sweep n (the reuse dim).\n")
hdr = f"{'n':>6} | {'cpu_left':>9} | {'copy':>8} {'GBps':>6} | {'gpu_mm':>8} | {'staged':>8} | {'speedup':>7} | {'rel_cl':>7}"
print(hdr); print("-" * len(hdr))

for n in [512, 1024, 2048, 4096, 8192]:
    x = make_inputs(G, mpg, n, k)
    rcl, rgpu = correctness(x)
    t_cl = bench(f_cpu_left, x)
    t_cp = bench(f_copy, x)
    t_mm = bench(f_gpu_mm, x)
    t_st = bench(f_staged, x)
    a_bytes = x["m"] * k * 2
    gbps = a_bytes / (t_cp / 1e3) / 1e9
    speed = t_st / t_cl
    print(f"{n:>6} | {t_cl:>8.3f}m | {t_cp:>7.3f}m {gbps:>5.0f} | {t_mm:>7.3f}m | {t_st:>7.3f}m | {speed:>6.2f}x | {rcl:>7.4f}")

print("\n(ms per call, median-of-best.  speedup = staged / cpu_left;  >1 means cpu_left faster.)")
print("copy GBps = effective CPU->HBM bandwidth for A over C2C.")
