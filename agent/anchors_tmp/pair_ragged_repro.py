#!/usr/bin/env python3
"""Minimal repro: native pair cpu-left on ragged groups vs torch reference."""
import torch
import asym_gemm  # noqa: F401
from asym_gemm.training import cpu_left as cl

E, R, K = 16, 64, 2048
torch.manual_seed(0)
dev = torch.device("cuda")
counts = [0, 133, 4096, 7, 2048, 0, 999, 512, 64, 8192, 3, 1024, 0, 4441, 256, 777]
M = sum(counts)
offs = [0]
for c in counts:
    offs.append(offs[-1] + c)
offsets = torch.tensor(offs, device=dev, dtype=torch.int32)
experts = torch.tensor(list(range(E)) + [-1], device=dev, dtype=torch.int32)
x = torch.empty((M, K), dtype=torch.bfloat16, pin_memory=True)
x.normal_()
wg = torch.randn((E, R, K), device=dev, dtype=torch.bfloat16).contiguous()
wu = torch.randn((E, R, K), device=dev, dtype=torch.bfloat16).contiguous()
x_dev = x.to(dev)

ref_g = torch.empty((M, R), device=dev, dtype=torch.bfloat16)
ref_u = torch.empty_like(ref_g)
for g in range(E):
    s, e = offs[g], offs[g + 1]
    if e > s:
        ref_g[s:e] = (x_dev[s:e].float() @ wg[g].float().t()).to(torch.bfloat16)
        ref_u[s:e] = (x_dev[s:e].float() @ wu[g].float().t()).to(torch.bfloat16)

def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))

for trial in range(4):
    sg = cl.grouped_expert_lora_cpu_left(x, wg, offsets, experts)
    pg, pu = cl.grouped_expert_lora_pair_cpu_left(x, wg, wu, offsets, experts)
    torch.cuda.synchronize()
    print(f"trial {trial}: single {rel(sg, ref_g):.3e}  pair.g {rel(pg, ref_g):.3e}  pair.u {rel(pu, ref_u):.3e}")
    # per-group localization on failure
    if rel(pg, ref_g) > 2e-2:
        for g in range(E):
            s, e = offs[g], offs[g + 1]
            if e > s:
                d = rel(pg[s:e], ref_g[s:e])
                if d > 2e-2:
                    print(f"   BAD group {g}: rows {s}:{e} ({e-s}) rel={d:.3e}")
