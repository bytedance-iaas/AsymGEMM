#!/usr/bin/env python3
"""Numeric equivalence: fg path with cpu-left vs GPU LoRA-A recompute-forward."""
import os

os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD"] = "1"
os.environ["ASYMM_EXPERT_ACT_OFFLOAD"] = "false"
os.environ["ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD"] = "cpu"
os.environ["ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE"] = "0"
os.environ["ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU"] = "0"

import torch
import torch.nn.functional as F
from torch import nn

from asym_gemm.training.qwen3_moe import AsymQwen3Experts

TOKENS, TOP_K, E, H, I = 65536, 8, 128, 2048, 768


class Src(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = E, H, I
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H, dtype=torch.bfloat16) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(E, H, I, dtype=torch.bfloat16) * 0.02)


def run(model, x0, idx, w):
    x = x0.clone().requires_grad_(True)
    for p in model.parameters():
        p.grad = None
    out = model(x, idx, w)
    loss = out.float().square().mean()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.item()), x.grad.detach().clone(), grads, out.detach().clone()


torch.manual_seed(3)
model = AsymQwen3Experts(Src(), backend="asym", precision="bf16", offload=True,
                         lora_rank=64, lora_alpha=16.0, lora_dropout=0.0,
                         init_lora_weights="peft", strict=False).cuda().train()
model._qwen3_moe_finegrained_enabled = True
with torch.no_grad():
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.copy_(torch.randn_like(p) * 0.01)

g = torch.Generator(device="cpu"); g.manual_seed(5)
idx = torch.randint(0, E, (TOKENS, TOP_K), generator=g).cuda()
w = torch.rand(TOKENS, TOP_K, generator=g)
w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16).cuda()
x0 = torch.randn(TOKENS, H, device="cuda", dtype=torch.bfloat16)

loss_cpu, dx_cpu, g_cpu, out_cpu = run(model, x0, idx, w)
os.environ["ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU"] = "1"
loss_gpu, dx_gpu, g_gpu, out_gpu = run(model, x0, idx, w)

def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return d, d / max(s, 1e-12)

print(f"loss cpu={loss_cpu:.6f} gpu={loss_gpu:.6f} absdiff={abs(loss_cpu-loss_gpu):.3e}")
print("out  max_abs_diff=%.3e rel=%.3e" % rel(out_gpu, out_cpu))
print("dx   max_abs_diff=%.3e rel=%.3e" % rel(dx_gpu, dx_cpu))
worst = ("", 0.0, 0.0)
for n in g_cpu:
    d, r = rel(g_gpu[n], g_cpu[n])
    if r > worst[2]:
        worst = (n, d, r)
print(f"worst grad: {worst[0]} max_abs_diff={worst[1]:.3e} rel={worst[2]:.3e}")
stats = model.stats.as_dict()
print("gpu_lora_a_fwd_calls:", stats.get("qwen3_moe_finegrained_lora_a_forward_gpu_calls"))
print("cpu_left_calls total:", stats.get("cpu_left_lora_a_calls"))
