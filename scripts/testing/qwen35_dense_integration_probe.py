"""Integration probe: qwen3.5 dense text model, pre-wrap vs asym-wrapped forward.

Builds a 2-layer Qwen3_5TextModel (one linear_attention + one full_attention layer,
random weights, real 27B dims scaled by --small), snapshots a reference forward, then
applies the real `apply_lf_asym_lora` wrap (backend=asym, target all, offload all) and
compares the wrapped forward on the same input. LoRA-B is zero at init, so outputs
must match to bf16 kernel noise. Divergence => the wrap composition corrupts forward.

Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/testing/qwen35_dense_integration_probe.py [--small]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel

from asym_gemm.integrations.lf import apply_lf_asym_lora, move_lf_asym_cpu_first_model_to_device

TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj"


def build(small: bool) -> Qwen3_5TextModel:
    if small:
        cfg = Qwen3_5TextConfig(
            hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=4,
            vocab_size=1024, layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=4096,
        )
    else:
        cfg = Qwen3_5TextConfig(
            hidden_size=5120, intermediate_size=17408, num_hidden_layers=2,
            num_attention_heads=24, num_key_value_heads=4, head_dim=256,
            linear_num_key_heads=16, linear_num_value_heads=48,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
            vocab_size=4096, layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=8192,
        )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(3)
    model = Qwen3_5TextModel(cfg).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run(model, ids, dev) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=ids.to(dev))
    return out.last_hidden_state.detach().float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()
    dev = torch.device("cuda:0")

    model = build(args.small)
    ids = torch.randint(0, model.config.vocab_size, (2, args.tokens))

    ref_model = copy.deepcopy(model).to(dev)
    ref_model.eval()
    ref = run(ref_model, ids, dev)
    del ref_model
    torch.cuda.empty_cache()

    model_wrapped, report = apply_lf_asym_lora(
        model,
        raw_lora_target="all",
        dense_target_modules=TARGETS,
        lora_rank=64,
        lora_alpha=16.0,
        lora_dropout=0.0,
        backend="asym",
        precision="bf16",
        offload_modules="all",
        expert_recompute_policy="none",
        router_mode="whole",
        strict=True,
    )
    print("wrap report:", str(report)[:600].replace(", ", ",\n  "))
    move_lf_asym_cpu_first_model_to_device(model_wrapped, dev, offload_modules="all")
    model_wrapped.eval()
    got_eval = run(model_wrapped, ids, dev)
    model_wrapped.train()
    got_train = run(model_wrapped, ids, dev)

    for tag, got in (("eval", got_eval), ("train+nograd", got_train)):
        diff = (got - ref).abs()
        rel = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
        print(f"{tag:14s} rel_fro={rel:9.5f} max|d|={diff.max():9.5f} ref_absmax={ref.abs().max():8.4f} got_absmax={got.abs().max():8.4f} nan={int(torch.isnan(got).sum())}")
        print("verdict:", "MATCH" if rel < 0.03 else "DIVERGED")


if __name__ == "__main__":
    main()
