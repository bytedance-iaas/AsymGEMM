#!/usr/bin/env python3
"""Correctness check for the fig12 reaim arm (ASYMM_LORA_KERNELS=reaim).

Grouped ragged case (E experts, ragged segments incl. non-8-aligned tails and
empty groups): ours vs reaim vs torch reference for K1 single/pair/triple fwd
and K2 single/pair grad. Run inside the container on a GPU.
"""
import os
import sys

import torch

import asym_gemm  # noqa: F401
from asym_gemm.training import cpu_left as cl
from asym_gemm.training import exp_act_offload_lora as ea

E, R, K = 16, 64, 2048
torch.manual_seed(0)
dev = torch.device("cuda")

# ragged segments: some empty, some non-8-aligned
counts = [0, 133, 4096, 7, 2048, 0, 999, 512, 64, 8192, 3, 1024, 0, 4441, 256, 777]
assert len(counts) == E
M = sum(counts)
offs = [0]
for c in counts:
    offs.append(offs[-1] + c)
offsets = torch.tensor(offs, device=dev, dtype=torch.int32)
experts = torch.tensor(list(range(E)) + [-1], device=dev, dtype=torch.int32)

x = torch.empty((M, K), dtype=torch.bfloat16, pin_memory=True)
x.normal_()
w = [torch.randn((E, R, K), device=dev, dtype=torch.bfloat16).contiguous() for _ in range(3)]
x_dev = x.to(dev)


def ref_fwd(weight):
    out = torch.empty((M, R), device=dev, dtype=torch.bfloat16)
    for g in range(E):
        s, e = offs[g], offs[g + 1]
        if e > s:
            out[s:e] = (x_dev[s:e].float() @ weight[g].float().t()).to(torch.bfloat16)
    return out


def rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).norm() / b.norm().clamp_min(1e-9)
    return float(d)


def run(mode):
    os.environ["ASYMM_LORA_KERNELS"] = mode
    r = {}
    # sync between grouped calls: the ragged-padding stage buffer is pooled and
    # back-to-back unsynced calls race on it (latent bug in the unused ragged
    # path, 2026-08-13 repro pair_ragged_repro.py) — this test checks math, not
    # overlap, so serialize.
    r["single"] = cl.grouped_expert_lora_cpu_left(x, w[0], offsets, experts)
    torch.cuda.synchronize()
    r["pair"] = cl.grouped_expert_lora_pair_cpu_left(x, w[0], w[1], offsets, experts)
    torch.cuda.synchronize()
    r["triple"] = cl.grouped_expert_lora_triple_cpu_left(x, w[0], w[1], w[2], offsets, experts)
    torch.cuda.synchronize()
    ds_g = torch.randn((M, R), device=dev, dtype=torch.bfloat16).contiguous()
    ds_u = torch.randn((M, R), device=dev, dtype=torch.bfloat16).contiguous()
    torch.manual_seed(1)  # deterministic dS across modes
    ds_g.normal_(); ds_u.normal_()
    r["grad"] = ea.grouped_lora_a_grad_cpu_right(ds_g, x, offsets, experts, num_experts=E, stats=None, tag="t")
    r["pgrad"] = ea.grouped_lora_a_pair_grad_cpu_right(ds_g, ds_u, x, offsets, experts, num_experts=E, stats=None)
    r["_ds"] = (ds_g, ds_u)
    torch.cuda.synchronize()
    return r


ours = run("")
reaim = run("reaim")
staged = run("staged")

fwd_ref = [ref_fwd(w[i]) for i in range(3)]
ok = True
for name, got_o, got_r, refv in [
    ("single", ours["single"], reaim["single"], fwd_ref[0]),
    ("pair.g", ours["pair"][0], reaim["pair"][0], fwd_ref[0]),
    ("pair.u", ours["pair"][1], reaim["pair"][1], fwd_ref[1]),
    ("tri.0", ours["triple"][0], reaim["triple"][0], fwd_ref[0]),
    ("tri.1", ours["triple"][1], reaim["triple"][1], fwd_ref[1]),
    ("tri.2", ours["triple"][2], reaim["triple"][2], fwd_ref[2]),
]:
    ro, rr = rel(got_o, refv), rel(got_r, refv)
    flag = "OK" if max(ro, rr) < 2e-2 else "FAIL"
    ok &= flag == "OK"
    print(f"fwd {name:7} ours-vs-ref {ro:.3e}  reaim-vs-ref {rr:.3e}  {flag}")

ds_g, ds_u = ours["_ds"]
gref_g = torch.zeros((E, R, K), device=dev, dtype=torch.float32)
gref_u = torch.zeros_like(gref_g)
for g in range(E):
    s, e = offs[g], offs[g + 1]
    if e > s:
        gref_g[g] = ds_g[s:e].t().float() @ x_dev[s:e].float()
        gref_u[g] = ds_u[s:e].t().float() @ x_dev[s:e].float()
for name, got_o, got_r, refv in [
    ("grad", ours["grad"], reaim["grad"], gref_g),
    ("pgrad.g", ours["pgrad"][0], reaim["pgrad"][0], gref_g),
    ("pgrad.u", ours["pgrad"][1], reaim["pgrad"][1], gref_u),
]:
    ro, rr = rel(got_o, refv), rel(got_r, refv)
    flag = "OK" if max(ro, rr) < 2e-2 else "FAIL"
    ok &= flag == "OK"
    print(f"bwd {name:7} ours-vs-ref {ro:.3e}  reaim-vs-ref {rr:.3e}  {flag}")

for name, got_s, refv in [
    ("single", staged["single"], fwd_ref[0]),
    ("pair.g", staged["pair"][0], fwd_ref[0]),
    ("tri.2", staged["triple"][2], fwd_ref[2]),
    ("grad", staged["grad"], gref_g),
    ("pgrad.u", staged["pgrad"][1], gref_u),
]:
    rs = rel(got_s, refv)
    flag = "OK" if rs < 2e-2 else "FAIL"
    ok &= flag == "OK"
    print(f"staged {name:7} vs-ref {rs:.3e}  {flag}")

# single-group (attention-shaped) case incl. non-aligned M
for m1 in (65536, 65533):
    x1 = torch.empty((m1, K), dtype=torch.bfloat16, pin_memory=True)
    x1.normal_()
    o1 = torch.tensor([0, m1], device=dev, dtype=torch.int32)
    e1 = torch.tensor([0, -1], device=dev, dtype=torch.int32)
    os.environ["ASYMM_LORA_KERNELS"] = ""
    a = cl.grouped_expert_lora_cpu_left(x1, w[0][:1].contiguous(), o1, e1)
    os.environ["ASYMM_LORA_KERNELS"] = "reaim"
    b = cl.grouped_expert_lora_cpu_left(x1, w[0][:1].contiguous(), o1, e1)
    torch.cuda.synchronize()
    d = rel(b, a)
    flag = "OK" if d < 2e-2 else "FAIL"
    ok &= flag == "OK"
    print(f"attn-shape M={m1}: reaim-vs-ours {d:.3e}  {flag}")

print("ALL OK" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
