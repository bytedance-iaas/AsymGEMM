"""Per-layer forward memory trace: stock vs canonical-patched qwen3.5-MoE forward.

4 decoder layers (3 linear_attention + 1 full_attention) at REAL 35B dims and long
seq, random weights, plain bf16 GPU modules (no asym wrap). Measures allocated and
peak GPU memory after each layer under no_grad — the unsloth-GC pass-1 condition.
Growth in `alloc_after` across layers = forward retention; flat = transient only.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_moe_forward_memtrace.py [--seq 80000] [--patched]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO_ROOT), "LlamaFactory", "src"))

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTextModel,
)


def build(seq: int) -> Qwen3_5MoeTextModel:
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=2048, num_hidden_layers=4,
        num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
        shared_expert_intermediate_size=512, decoder_sparse_step=1,
        vocab_size=8192, max_position_embeddings=131072,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(5)
    model = Qwen3_5MoeTextModel(cfg)
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patched", action="store_true", help="apply the canonical LF qwen3.5 forward patch")
    args = ap.parse_args()
    dev = torch.device("cuda:0")

    model = build(args.seq)
    if args.patched:
        from llamafactory.model.patcher import patch_qwen3_5_forward

        class _FakeRoot:
            pass

        root = _FakeRoot()
        root.config = model.config
        root.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        patch_qwen3_5_forward(root)  # patches the CLASS forwards
        print("[patched canonical forward]")
    else:
        print("[stock transformers forward]")

    model = model.to(dev)
    model.eval()

    marks: list[tuple[str, float, float]] = []

    def hook(name):
        def f(mod, inp, out):
            torch.cuda.synchronize()
            marks.append((name, torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30))
        return f

    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(hook(f"layer{i}({model.config.layer_types[i]})"))

    ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device=dev)
    pos = torch.arange(args.seq, device=dev).unsqueeze(0).expand(args.batch, -1)
    base = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(input_ids=ids)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    del out

    print(f"weights+ids baseline: {base:.2f} GiB | run peak: {peak:.2f} GiB")
    prev = base
    for name, alloc, reserved in marks:
        print(f"  after {name:28s} alloc={alloc:7.2f} GiB (delta {alloc-prev:+6.2f}) reserved={reserved:7.2f}")
        prev = alloc


if __name__ == "__main__":
    main()
