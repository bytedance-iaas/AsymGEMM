import os, gc, torch
import torch.nn as nn
from asym_gemm.training.qwen3_moe import AsymQwen3Experts

def vmrss_gib():
    for line in open(f"/proc/{os.getpid()}/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / (1024 * 1024)  # kB -> GiB
    return -1.0

E, H, I = 128, 2048, 768  # real Qwen3-30B-A3B expert dims
def one_copy_gib():
    return (E * (2 * I) * H + E * H * I) * 2 / (1024 ** 3)

class FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = E; self.hidden_dim = H; self.intermediate_dim = I
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H, dtype=torch.bfloat16), requires_grad=False)
        self.down_proj    = nn.Parameter(torch.randn(E, H, I, dtype=torch.bfloat16), requires_grad=False)
        self.config = None
        self.act_fn = nn.SiLU()

def build(src):
    return AsymQwen3Experts(src, backend="asym", precision="bf16", offload=True,
                            lora_rank=16, lora_alpha=32.0, lora_dropout=0.0, strict=False)

assert torch.cuda.is_available(), "need CUDA for pinned HostWeights"

# ---- correctness on one layer: faithful independent copy + source freed ----
s0 = FakeExperts()
ref_gu = s0.gate_up_proj.detach().clone()
ref_dn = s0.down_proj.detach().clone()
e0 = build(s0)
hw_gu = e0.gate_up_base.host_weight.weight
hw_dn = e0.down_base.host_weight.weight
print("pinned        :", bool(hw_gu.is_pinned()), bool(hw_dn.is_pinned()))
print("faithful copy :", bool(torch.equal(hw_gu, ref_gu)), bool(torch.equal(hw_dn, ref_dn)))
print("source numel  :", s0.gate_up_proj.numel(), s0.down_proj.numel(), "(must be 0,0)")
assert hw_gu.is_pinned() and hw_dn.is_pinned()
assert torch.equal(hw_gu, ref_gu) and torch.equal(hw_dn, ref_dn)
assert s0.gate_up_proj.numel() == 0 and s0.down_proj.numel() == 0
del s0, e0, ref_gu, ref_dn, hw_gu, hw_dn
gc.collect()

# ---- memory: K layers grow RSS by ~1x (one copy), not 2x ----
K = 8
gc.collect(); base = vmrss_gib()
mods, srcs = [], []
for _ in range(K):
    s = FakeExperts(); srcs.append(s); mods.append(build(s))
gc.collect()
grow = vmrss_gib() - base
oc = K * one_copy_gib()
src_resident = sum(s.gate_up_proj.numel() + s.down_proj.numel() for s in srcs)
print(f"\nK={K} layers  one-copy={oc:.2f} GiB")
print(f"RSS grew    = {grow:.2f} GiB   (1x~{oc:.2f}, 2x~{2*oc:.2f})")
print(f"ratio grow/1x = {grow/oc:.2f}   (expect ~1.0; un-fixed bug ~2.0)")
print(f"sources resident numel (must be 0): {src_resident}")
assert src_resident == 0, "sources NOT freed"
assert grow < 1.5 * oc, f"RSS {grow:.2f} >= 1.5x one-copy -> duplicate not freed"
print("\nSTAGE 1 VALIDATION: PASS (faithful copy, sources freed, RSS ~1x)")
