import time, torch

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

def t(fn, *a, sync=True, reps=1, **k):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a, **k)
    if sync: torch.cuda.synchronize()
    return (time.perf_counter()-t0)/reps, out

R, H = 5_120_000, 2048          # expanded rows x hidden (per layer, s80000.b8 topk8)
I = 768
GB = 1024**3
res = {}

# 1) fresh pinned alloc of 21 GB (X-sized)
dt, x_pin = t(lambda: torch.empty((R, H), dtype=torch.bfloat16, pin_memory=True))
res["pinned_alloc_21GB_s"] = dt
nbytes = R*H*2
# 2) zero_ (padding path does this)
dt, _ = t(lambda: x_pin.zero_(), sync=False)
res["cpu_zero_21GB_s"] = dt
# 3) cpu->cpu copy 21GB (padding copies rows group by group; emulate w/ 128 chunks)
src = torch.empty((R, H), dtype=torch.bfloat16, pin_memory=True)
chunks = torch.chunk(torch.arange(R), 128)
def group_copy():
    for c in chunks:
        x_pin[c[0]:c[-1]+1].copy_(src[c[0]:c[-1]+1])
dt, _ = t(group_copy, sync=False)
res["cpu_groupcopy_21GB_s"] = dt
# 4) pinned free (drop ref)
t0=time.perf_counter(); del src; import gc; gc.collect(); res["pinned_free_21GB_s"]=time.perf_counter()-t0

# 5) D2H pinned vs pageable, 7.9GB (gate-sized)
g = torch.randn((R, I), dtype=torch.bfloat16, device=dev)
dst_pin = torch.empty((R, I), dtype=torch.bfloat16, pin_memory=True)
dt, _ = t(lambda: dst_pin.copy_(g, non_blocking=True)); res["d2h_pinned_7.9GB_s"]=dt
dst_pag = torch.empty((R, I), dtype=torch.bfloat16)
dt, _ = t(lambda: dst_pag.copy_(g, non_blocking=False)); res["d2h_pageable_7.9GB_s"]=dt
# 6) H2D pinned vs pageable
dt, _ = t(lambda: g.copy_(dst_pin, non_blocking=True)); res["h2d_pinned_7.9GB_s"]=dt
dt, _ = t(lambda: g.copy_(dst_pag, non_blocking=False)); res["h2d_pageable_7.9GB_s"]=dt

# 7) repeat pinned alloc/free cycle like pool-miss churn (7.9GB x3)
def churn():
    bufs=[torch.empty((R,I),dtype=torch.bfloat16,pin_memory=True) for _ in range(3)]
    for b in bufs: b.zero_()
    del bufs
dt,_ = t(churn, sync=False); res["churn_3x7.9GB_alloc_zero_free_s"]=dt

nb = {"21GB": nbytes/GB, "7.9GB": R*I*2/GB}
print(f"tensor sizes: X={nb['21GB']:.1f} GiB gate={nb['7.9GB']:.1f} GiB")
for k,v in res.items():
    sz = 21 if "21GB" in k else (3*7.9 if "3x7.9" in k else 7.9)
    print(f"{k:38s} {v:8.3f}s  ({sz/v:6.1f} GB/s)")
