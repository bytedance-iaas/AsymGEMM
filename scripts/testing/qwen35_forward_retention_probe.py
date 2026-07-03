"""Retention probe: who holds per-layer delta-net tensors through a GC'd forward?

Builds a small hybrid Qwen3_5TextModel (mostly linear_attention layers), applies the
real asym wrap + the canonical LF unsloth gradient checkpointing, runs ONE training
forward (no backward), then reports which per-layer module outputs are still alive on
GPU and WHO references them (gc referrer chains). Mirrors the s80000 canonical OOM
pathology (layer-1 linear_attn tensors alive at layer 40) at toy scale.

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_forward_retention_probe.py
"""

from __future__ import annotations

import gc
import os
import sys
from types import MethodType

import torch

sys.path.insert(0, "/workspace/AsymGEMM-SFT/third_party/AsymGEMM")
sys.path.insert(0, "/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src")

os.environ.setdefault("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU", "true")

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTextModel,
)

from asym_gemm.integrations.lf import apply_lf_asym_lora, move_lf_asym_cpu_first_model_to_device

TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj"
N_LAYERS = 6


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--input-require-grads", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    if args.moe:
        cfg = Qwen3_5MoeTextConfig(
            hidden_size=512, num_hidden_layers=N_LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
            shared_expert_intermediate_size=128, decoder_sparse_step=1,
            vocab_size=1024, max_position_embeddings=4096,
            layer_types=["linear_attention"] * (N_LAYERS - 1) + ["full_attention"],
        )
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(3)
        model = Qwen3_5MoeTextModel(cfg).to(torch.bfloat16)
    else:
        cfg = Qwen3_5TextConfig(
            hidden_size=512, intermediate_size=1024, num_hidden_layers=N_LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=4,
            vocab_size=1024, max_position_embeddings=4096,
            layer_types=["linear_attention"] * (N_LAYERS - 1) + ["full_attention"],
        )
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(3)
        model = Qwen3_5TextModel(cfg).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    model, report = apply_lf_asym_lora(
        model, raw_lora_target="all", dense_target_modules=TARGETS,
        lora_rank=16, lora_alpha=16.0, lora_dropout=0.0, backend="asym", precision="bf16",
        offload_modules="all", expert_recompute_policy="none", router_mode="whole", strict=True,
    )
    move_lf_asym_cpu_first_model_to_device(model, dev, offload_modules="all")

    # canonical LF unsloth GC, exactly as the trainer applies it
    from llamafactory.model.model_utils.checkpointing import _gradient_checkpointing_enable
    from functools import partial
    model.gradient_checkpointing_enable = MethodType(
        partial(_gradient_checkpointing_enable, use_unsloth_gc=True), model
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()

    # track module outputs by weakref
    import weakref
    tracked: dict[str, object] = {}
    sizes: dict[str, tuple] = {}

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.is_cuda:
                tracked[name] = weakref.ref(t)
                sizes[name] = tuple(t.shape)
        return f

    for n, m in model.named_modules():
        if n.count(".") <= 2 and n:
            m.register_forward_hook(hook(n))

    ids = torch.randint(0, cfg.vocab_size, (2, args.tokens), device=dev)
    mask = None
    if os.environ.get("PROBE_ATTENTION_MASK", "0") == "1":
        mask = torch.ones(2, args.tokens, dtype=torch.long, device=dev)
        print("[passing 2D all-ones attention_mask — SDPA-mode collator behavior]")
    if args.input_require_grads and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    out = model(input_ids=ids, attention_mask=mask)
    hidden = out.last_hidden_state
    del out
    gc.collect()

    print(f"alloc after forward: {torch.cuda.memory_allocated()/2**20:.1f} MiB")
    alive = {n: r() for n, r in tracked.items() if r() is not None}
    print(f"module outputs still alive after forward: {len(alive)} / {len(tracked)}")
    for n in sorted(alive):
        t = alive[n]
        print(f"  ALIVE {n:44s} {sizes[n]} rg={t.requires_grad} grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")

    # referrer chains for the first alive early-layer tensor
    for n in sorted(alive):
        if ".linear_attn" in n or "layers.0" in n or "layers.1" in n:
            t = alive[n]
            print(f"\nreferrer chain for {n}:")
            refs = [r for r in gc.get_referrers(t) if r is not tracked and r is not alive]
            for r in refs[:6]:
                desc = type(r).__name__
                if isinstance(r, dict):
                    keys = [k for k, v in r.items() if v is t]
                    desc += f" keys={keys[:3]}"
                elif isinstance(r, (list, tuple)):
                    desc += f" len={len(r)}"
                print(f"  <- {desc}")
                for r2 in gc.get_referrers(r)[:4]:
                    print(f"     <- {type(r2).__name__}" + (f" (module={getattr(r2, '__class__', None)})" if hasattr(r2, "forward") else ""))
            break

    hidden.sum().backward()
    print("\nbackward OK; loss-path grad flow intact")


if __name__ == "__main__":
    main()
