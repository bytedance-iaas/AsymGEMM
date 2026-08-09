import os, torch
os.environ["ASYM_EXACT_PINNED_SAVED"] = "1"
def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
torch.cuda.init()
from asym_gemm.training import activation_offload as ao
from asym_gemm.training import exact_pinned
r0 = rss_gb()
t = ao._alloc_cpu((352000, 6144), torch.bfloat16, pin_memory=True, tag="moe.X")
r1 = rss_gb()
t.view(-1)[0] = 1.0
r2 = rss_gb()
print(f"alloc [352000,6144]bf16 logical=4.33GB rss_delta_alloc={r1-r0:.2f} rss_after_touch={r2-r0:.2f} pinned={t.is_pinned()}")
print("register_stats:", exact_pinned.register_stats())
base = getattr(t, "_asym_pool_base", t)
print("marker on base:", getattr(base, "_asym_exact_registered", False), "base shape:", tuple(base.shape))
ao._return_cpu(t, pin_memory=True)
t2 = ao._alloc_cpu((352000, 6144), torch.bfloat16, pin_memory=True, tag="moe.X")
print("reuse: same storage:", t2.untyped_storage().data_ptr() == base.untyped_storage().data_ptr(),
      "registered_count still:", exact_pinned.register_stats()["registered_count"])
ao._return_cpu(t2, pin_memory=True)
ao._trim_cpu_pool(0)
print("after trim(0): pool still holds registered:", sum(len(v) for v in ao._CPU_BUFFER_POOL.values()),
      "evictions:", ao._CPU_BUFFER_POOL_EVICTIONS)
r3 = rss_gb()
print(f"final rss_delta={r3-r0:.2f} GB (expect ~4.4, not 8)")
