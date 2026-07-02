#!/usr/bin/env python3
"""Isolated per-phase timing of one Qwen3-30B-A3B MoE layer through the
qwen3_moe_finegrained fwd+bwd path at production shape (s80000.b8, top_k=8).

This reproduces exactly what runs per-layer inside step.backward of the e2e
recomp-off-full-fg run (recompute-forward with offload + fg backward).
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD"] = "1"
os.environ["ASYMM_EXPERT_ACT_OFFLOAD"] = "false"
os.environ["ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD"] = "cpu"
os.environ["ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE"] = "0"
os.environ["DG_BF16_CPU_LEFT_COMPACT_GRID"] = "0"

import torch
import torch.nn.functional as F
from torch import nn

import asym_gemm
import asym_gemm.training.activation_offload as ao_mod
import asym_gemm.training.cpu_left as cpu_left_mod
import asym_gemm.training.qwen3_moe_finegrained as fg_mod
from asym_gemm.training.activation_offload import ActivationOffloadManager
from asym_gemm.training.qwen3_moe import AsymQwen3Experts

TOKENS = int(os.environ.get("BENCH_TOKENS", 640000))
TOP_K = 8
NUM_EXPERTS = 128
HIDDEN = 2048
INTER = 768
RANK, ALPHA = 64, 16.0
ITERS = int(os.environ.get("BENCH_ITERS", 2))
WARMUP = int(os.environ.get("BENCH_WARMUP", 1))


class Rec:
    def __init__(self):
        self.rows = {}
        self.enabled = False

    @contextmanager
    def record(self, name, enabled=None):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            row = self.rows.setdefault(name, [0, 0.0])
            row[0] += 1
            row[1] += dt

    def summary(self, iters):
        out = [
            {"name": k, "count": c, "total_ms": tm, "per_step_ms": tm / max(1, iters)}
            for k, (c, tm) in self.rows.items()
        ]
        out.sort(key=lambda r: -r["total_ms"])
        return out


REC = Rec()

# ---- patch prof_range in fg module ----
_orig_prof_range = fg_mod.prof_range
def timed_prof_range(name, *, enabled=None):
    return REC.record(f"range.{name}")
fg_mod.prof_range = timed_prof_range

# ---- patch transfer + alloc primitives ----
_orig_offload = ActivationOffloadManager.offload
_orig_stage = ActivationOffloadManager.stage
_orig_alloc = ao_mod._alloc_cpu
_orig_pad = cpu_left_mod._pad_cpu_left_grouped_input_for_asym

def timed_offload(self, tensor, tag):
    with REC.record(f"xfer.offload.{tag}"):
        return _orig_offload(self, tensor, tag)

def timed_stage(self, handle, *, tag=None):
    with REC.record(f"xfer.stage.{handle.tag if tag is None else tag}"):
        return _orig_stage(self, handle, tag=tag)

ALLOC_BYTES = [0, 0]  # count, bytes
def timed_alloc(shape, dtype, *, pin_memory):
    import math
    n = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    with REC.record("host.alloc_cpu"):
        out = _orig_alloc(shape, dtype, pin_memory=pin_memory)
    ALLOC_BYTES[0] += 1
    ALLOC_BYTES[1] += n
    return out

def timed_pad(x_cpu, offsets, experts, *, index_device, block_m=cpu_left_mod.CPU_LEFT_GROUPED_BLOCK_M):
    with REC.record("host.cpu_left_pad"):
        return _orig_pad(x_cpu, offsets, experts, index_device=index_device, block_m=block_m)

ActivationOffloadManager.offload = timed_offload
ActivationOffloadManager.stage = timed_stage
ao_mod._alloc_cpu = timed_alloc
cpu_left_mod._pad_cpu_left_grouped_input_for_asym = timed_pad

# ---- patch fg module-level compute helpers ----
_orig_base_fwd = fg_mod._base_forward
_orig_base_dx = fg_mod._base_dx
_orig_la_fwd = fg_mod._lora_a_forward_cpu
_orig_la_grad = fg_mod._lora_a_grad_cpu
_orig_lb_grad = fg_mod._lora_b_grad
_orig_lds = fg_mod._lora_ds
_orig_lb_fwd = fg_mod._lora_b_forward

def _wrap(fn, name):
    def inner(*a, **k):
        with REC.record(name):
            return fn(*a, **k)
    return inner

fg_mod._base_forward = lambda layer, base, x, o, e, *, part: _wrap(_orig_base_fwd, f"gemm.base_fwd.{part}")(layer, base, x, o, e, part=part)
fg_mod._base_dx = lambda layer, base, g, o, e, *, part, input_dtype: _wrap(_orig_base_dx, f"gemm.base_dx.{part}")(layer, base, g, o, e, part=part, input_dtype=input_dtype)
fg_mod._lora_a_forward_cpu = lambda layer, s, a, o, e, m, *, tag: _wrap(_orig_la_fwd, f"lora.a_fwd_cpu.{tag}")(layer, s, a, o, e, m, tag=tag)
fg_mod._lora_a_grad_cpu = lambda layer, d, s, a, o, e, *, tag: _wrap(_orig_la_grad, f"lora.a_grad_cpu.{tag}")(layer, d, s, a, o, e, tag=tag)
fg_mod._lora_b_grad = _wrap(_orig_lb_grad, "lora.b_grad")
fg_mod._lora_ds = _wrap(_orig_lds, "lora.ds")
fg_mod._lora_b_forward = _wrap(_orig_lb_fwd, "lora.b_fwd")


class FakeQwen3Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN
        self.intermediate_dim = INTER
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=torch.bfloat16) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16) * 0.02)


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    src = FakeQwen3Experts()
    model = AsymQwen3Experts(
        src, backend="asym", precision="bf16", offload=True,
        lora_rank=RANK, lora_alpha=ALPHA, lora_dropout=0.0, init_lora_weights="peft", strict=False,
    )
    model.cuda().train()
    model._qwen3_moe_finegrained_enabled = True

    gen = torch.Generator(device="cpu")
    gen.manual_seed(11)
    top_k_index = torch.randint(0, NUM_EXPERTS, (TOKENS, TOP_K), generator=gen).cuda()
    w = torch.rand(TOKENS, TOP_K, generator=gen)
    top_k_weights = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16).cuda()

    def step():
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for p in model.parameters():
            p.grad = None
        out = model(x, top_k_index, top_k_weights)
        loss = out.float().square().mean()
        with REC.record("autograd.backward_total"):
            loss.backward()
        return float(loss.detach().item())

    pool_env = os.environ.get("ASYM_EXPACT_CPU_POOL_MAX_BYTES", "<default 32GiB>")
    for i in range(WARMUP):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        print(f"warmup {i}: {time.perf_counter()-t0:.2f}s", flush=True)

    REC.enabled = True
    ALLOC_BYTES[0] = ALLOC_BYTES[1] = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(ITERS):
        step()
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    print(f"\n=== pool={pool_env} tokens={TOKENS} iters={ITERS} ===")
    print(f"fwd+bwd per iter: {total/ITERS:.2f}s  peakHBM={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
    print(f"pinned allocs (measured iters): n={ALLOC_BYTES[0]} bytes={ALLOC_BYTES[1]/2**30:.1f}GiB")
    print(f"{'phase':52s} {'count':>5s} {'per_iter_ms':>12s}")
    for r in REC.summary(ITERS):
        print(f"{r['name']:52s} {r['count']:5d} {r['per_step_ms']:12.1f}")
    stats = ao_mod.activation_offload_cpu_pool_stats()
    print("pool:", {k: (f"{v/2**30:.1f}GiB" if "bytes" in k else v) for k, v in stats.items() if "by_shape" not in k})


if __name__ == "__main__":
    main()
