import os, torch
os.environ.setdefault("ASYM_EXACT_PINNED", "1")
os.environ.setdefault("ASYM_EXACT_PINNED_ROOTS", "1")
def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
t = torch.zeros(1, 384000, 6144, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
r0 = rss_gb()
stock = t.to("cpu", non_blocking=True)
torch.cuda.synchronize()
r1 = rss_gb()
print(f"stock .to(cpu,nb) [1,384000,6144]bf16 logical=4.72GB  rss_delta={r1-r0:.2f} GB  pinned={stock.is_pinned()}")
del stock
import gc; gc.collect(); torch.cuda.synchronize()
r2 = rss_gb()
from asym_gemm.training import exact_pinned
print("roots_enabled:", exact_pinned.exact_roots_enabled())
pool = exact_pinned.root_pool()
buf = pool.pack(t)
torch.cuda.synchronize()
r3 = rss_gb()
print(f"exact RootPool.pack       rss_delta={r3-r2:.2f} GB  pinned={buf.is_pinned()}  shape={tuple(buf.shape)}")
print("register_stats:", exact_pinned.register_stats())
