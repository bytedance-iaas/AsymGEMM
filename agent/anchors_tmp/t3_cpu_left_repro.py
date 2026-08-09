#!/usr/bin/env python3
"""t3_cpu_left_repro.py — minimal repro of the true-T3 segfault
(fix_glm_t3.md): the attn-act offload dense LoRA-A path calls
grouped_expert_lora_cpu_left(u_cpu [M,in] bf16, a [1,r,in] cuda bf16,
single-group offsets). Production crash site: cpu_left.py:263 (native
binding). Sweep M x in_features x pinning x compact-grid to find the
trigger; faulthandler prints the frame on segfault, and the harness runs
each case in a SUBPROCESS so one crash doesn't stop the sweep.
"""
from __future__ import annotations

import faulthandler
import os
import subprocess
import sys

faulthandler.enable()

REPO = "/workspace/AsymGEMM-SFT-46/third_party/AsymGEMM"


def _engine_init():
    """Production preconditions: construct a tiny wrapped MoE block first so
    the native CPU worker pool / pin pools exist (a bare call segfaults on
    ANY shape without this — repro v1 finding)."""
    import torch
    from transformers import AutoConfig

    from asym_gemm.training.glm47_moe import AsymGlm47MoeBlock

    cfg = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash")
    cfg = getattr(cfg, "text_config", cfg)
    import importlib
    mod = importlib.import_module(f"transformers.models.{cfg.model_type}.modeling_{cfg.model_type}")
    moe_cls = getattr(mod, "Glm4MoeLiteMoE", None) or getattr(mod, "Glm4MoeMoE")
    torch.manual_seed(0)
    blk = moe_cls(cfg).to(dtype=torch.bfloat16)
    w = AsymGlm47MoeBlock(blk, backend="asym", precision="bf16", offload=True,
                          lora_rank=64, lora_alpha=16.0, lora_dropout=0.0)
    dev = torch.device("cuda:0")
    for name, child in w.named_children():
        if name != "experts":
            child.to(dev)
    for prm in w.experts.parameters():
        if prm.requires_grad:
            prm.data = prm.data.to(dev)
    x = torch.randn(1, 256, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    out = w(x)
    (out[0] if isinstance(out, tuple) else out).float().sum().backward()
    torch.cuda.synchronize()
    return w  # keep alive: pools must persist


def one_case(m: int, in_features: int, pinned: bool, compact: str) -> None:
    os.environ["DG_BF16_CPU_LEFT_COMPACT_GRID"] = compact
    os.environ["ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD"] = "1"
    os.environ["ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU"] = "1"
    sys.path.insert(0, REPO)
    import torch

    from asym_gemm.training.attention_activation_offload import _single_group_offsets_experts
    from asym_gemm.training.lora import grouped_expert_lora_cpu_left

    keeper = _engine_init()
    r = 64
    torch.manual_seed(0)
    a = torch.randn(1, r, in_features, device="cuda:0", dtype=torch.bfloat16).contiguous()
    u = torch.randn(m, in_features, dtype=torch.bfloat16)
    if pinned:
        u = u.pin_memory()
    u = u.contiguous()
    offsets, experts = _single_group_offsets_experts(a.device, m)
    out = grouped_expert_lora_cpu_left(u, a, offsets, experts, output_dtype=a.dtype)
    torch.cuda.synchronize()
    ref = (u.to("cuda:0", torch.float32) @ a[0].t().float()).to(torch.bfloat16)
    d = (out.float() - ref.float()).abs().max().item()
    print(f"OK m={m} in={in_features} pinned={pinned} compact={compact} maxdiff={d:.3e}", flush=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "case":
        m, in_features, pinned, compact = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
        one_case(m, in_features, pinned, compact)
        return 0

    cases = []
    # MLA widths (576/1536) vs qwen-class widths; pinned only (unpinned is
    # a guarded RuntimeError, not the bug)
    for in_features in (576, 1536, 2048, 4096):
        for m in (1024, 64000):
            for pinned in (True,):
                for compact in ("0",):
                    cases.append((m, in_features, pinned, compact))
    crashed = []
    for m, in_features, pinned, compact in cases:
        proc = subprocess.run(
            [sys.executable, __file__, "case", str(m), str(in_features), "1" if pinned else "0", compact],
            capture_output=True, text=True, timeout=300,
        )
        line = (proc.stdout or "").strip().splitlines()
        if proc.returncode == 0 and line:
            print(line[-1], flush=True)
        else:
            sig = -proc.returncode if proc.returncode < 0 else proc.returncode
            print(f"CRASH m={m} in={in_features} pinned={pinned} compact={compact} rc={proc.returncode}", flush=True)
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            for t in tail:
                print(f"    {t}", flush=True)
            crashed.append((m, in_features, pinned, compact))
    print("CRASHING CASES:", crashed if crashed else "none", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
