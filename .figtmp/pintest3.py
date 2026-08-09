import os, torch
def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
t = torch.zeros(1, 384000, 6144, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
r0 = rss_gb()
c = t.to("cpu", non_blocking=True)
torch.cuda.synchronize()
r1 = rss_gb()
print(f"ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF','<unset>')}")
print(f".to(cpu,nb) [1,384000,6144]bf16 logical=4.72GB rss_delta={r1-r0:.2f} GB pinned={c.is_pinned()}")
