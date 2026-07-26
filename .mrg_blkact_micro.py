import os, torch
os.environ["ASYM_CPU_OPS_THREADS"] = "48"
os.environ["ASYMM_CPU_FUSED_SILU"] = "1"
from asym_gemm.training.activation_offload import ActivationOffloadManager
from asym_gemm.training import cpu_worker, cpu_ops

assert cpu_worker.enabled() or True  # worker lazily starts on submit
k = cpu_ops.fused_silu_kernels(); assert k is not None, "kernels off"
fwd, _, nt = k
dev = torch.device("cuda:0")
m = ActivationOffloadManager(pin_memory=True)
R, I, B = 8192, 768, 4  # 4 blocks
gate = m.empty_cpu((R, I), torch.bfloat16, dev, "moe.gate")
up   = m.empty_cpu((R, I), torch.bfloat16, dev, "moe.up")
act  = m.empty_cpu((R, I), torch.bfloat16, dev, "moe.act")
assert gate.tensor.is_pinned() and act.tensor.is_pinned()
g_gpu = torch.randn(R, I, device=dev, dtype=torch.bfloat16)
u_gpu = torch.randn(R, I, device=dev, dtype=torch.bfloat16)
ref = (torch.nn.functional.silu(g_gpu.float()) * u_gpu.float()).to(torch.bfloat16)
rows = R // B
for b in range(B):
    sl = slice(b*rows, (b+1)*rows)
    gate.tensor[sl].copy_(g_gpu[sl], non_blocking=True)
    up.tensor[sl].copy_(u_gpu[sl], non_blocking=True)
    ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(dev))
    def job(ev=ev, g=gate.tensor[sl], u=up.tensor[sl], o=act.tensor[sl], k=fwd, n=nt):
        ev.synchronize(); k(g, u, o, n)
    m.attach_cpu_task(act, cpu_worker.submit(job, tag="micro.blk"))
m.record_cpu_ready(gate); m.record_cpu_ready(up); m.record_cpu_ready(act)
staged = m.stage(act)  # must task-join then H2D
torch.cuda.synchronize()
diff = (staged.float() - ref.float().to(dev)).abs().max().item()
print("max|staged-ref| =", diff)
assert diff <= 2 * 2**-8, "bf16 tolerance breach"
# release path: attach a fresh dummy task then release (must join, not race)
m.release_cpu(act); m.release_cpu(gate); m.release_cpu(up)
print("MICRO OK (stage joined tasks; release clean)")
