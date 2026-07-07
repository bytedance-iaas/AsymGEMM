#!/usr/bin/env python3
"""I4 mini-parity smoke (gb200_tp.md): tiny 2-layer Qwen3, |1 wrap vs full-TP stp build.

Builds ONE tiny base model; wraps a |1 reference copy and an sTP copy through the SAME
factories; transplants the |1 adapter init into the sTP branches (stp_load_adapter_init);
then compares (a) forward logits and (b) per-adapter grads (|1 full tensors vs sTP pieces
re-assembled per the param map, mirrors merged) after one backward.

Cheap enough to iterate on every wrap bug before burning harness cycles.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("ASYM_STP", "1")
os.environ.setdefault("ASYM_STP_TP_SIZE", "2")
os.environ.setdefault("ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD", "1")
os.environ.setdefault("ASYMM_ATTN_ACT_OFFLOAD", "true")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

RANK, ALPHA = 16, 32.0
TOL = 2e-2


def build_base():
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3",
        hidden_size=1024,
        intermediate_size=2560,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=1024,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    return model, cfg


def wrap_one(model, tag: str):
    """Minimal |1-style wrap: attention projections -> AsymActivationOffloadLoRALinear
    (shared U ctx per layer), mlp -> fine-grained dense MLP."""
    from asym_gemm.training.attention_activation_offload import (
        AsymActivationOffloadLoRALinear,
        AttentionActivationOffloadContext,
    )
    from asym_gemm.training.dense_mlp_finegrained import build_finegrained_dense_mlp
    from asym_gemm.training.offload import adopt_host_weight
    from asym_gemm.training.frozen_linear import AsymExecutionStats

    stats = AsymExecutionStats()
    for idx, layer in enumerate(model.model.layers):
        ctx = AttentionActivationOffloadContext()
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, role)
            hw = adopt_host_weight(f"{tag}.l{idx}.{role}", lin.weight.data.pin_memory(), "attention",
                                   pin_memory_policy="auto", strict=True)
            wrapped = AsymActivationOffloadLoRALinear.from_host_weight(
                hw, rank=RANK, alpha=ALPHA, backend="asym", stats=stats,
                device=torch.device("cuda", 0), precision="bf16", lora_dropout=0.0,
                projection_role=role, attention_context=(ctx if role != "o_proj" else None))
            setattr(layer.self_attn, role, wrapped)
        for pname in ("gate_proj", "up_proj", "down_proj"):
            getattr(layer.mlp, pname).weight.data = getattr(layer.mlp, pname).weight.data.pin_memory()
        with torch.cuda.device(0):
            layer.mlp = build_finegrained_dense_mlp(
                layer.mlp, backend="asym", precision="bf16", lora_rank=RANK, lora_alpha=ALPHA,
                lora_dropout=0.0, stats=stats, strict=True, profile_prefix=f"model.layers.{idx}.mlp")
    # move the rest to dev0 (|1 style)
    for module in model.modules():
        for k, p in list(module._parameters.items()):
            if p is not None and p.device.type == "cpu":
                p.data = p.data.to("cuda:0")
        for k, b in list(module._buffers.items()):
            if b is not None and b.device.type == "cpu":
                module._buffers[k] = b.to("cuda:0")
    return stats


def named_lora(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad and "lora" in n}


def main() -> int:
    from asym_gemm.training.stp_runtime import get_runtime
    from asym_gemm.training.stp_wrap import (
        build_stp_full_tp, finalize_stp_placement, stp_load_adapter_init, stp_post_backward_merge)

    get_runtime()
    base, cfg = build_base()
    state = {k: v.clone() for k, v in base.state_dict().items()}

    # ---- |1 reference ----
    ref = base
    wrap_one(ref, "ref")
    ref_named = named_lora(ref)
    init_path = os.path.join(tempfile.mkdtemp(prefix="stp_mini_"), "init.pt")
    torch.save({"params": {n: p.detach().float().cpu() for n, p in ref.named_parameters() if p.requires_grad}}, init_path)

    # ---- sTP copy (fresh base with the SAME frozen weights) ----
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(7)
    stp = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    stp.load_state_dict(state)
    stats = wrap_one(stp, "stp")
    counts = build_stp_full_tp(stp, lora_rank=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
                               backend="asym", stats=stats, strict=True)
    finalize_stp_placement(stp, torch.device("cuda", 0))
    loaded = stp_load_adapter_init(stp, init_path)
    print(f"[mini] build counts={counts} init loaded={loaded}")

    # perturb LoRA B away from zero so dX-through-LoRA paths are exercised (identically on
    # both sides: reload-with-noise trick — add noise to REF B, re-dump, re-load into stp)
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(11)
        for n, p in ref.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.add_(torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.device, p.dtype) * 0.02)
    torch.save({"params": {n: p.detach().float().cpu() for n, p in ref.named_parameters() if p.requires_grad}}, init_path)
    loaded = stp_load_adapter_init(stp, init_path)
    print(f"[mini] B-perturbed reload: {loaded}")

    # ---- ENVELOPE model (gb200_tp.md I4): |1 wrap + Phase-A split routing — same math as
    # ref, only the GEMM partial-sum order differs => its grad deltas vs ref ARE the bf16
    # reduction-order noise floor that full-TP must be judged against.
    torch.manual_seed(7)
    env_model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    env_model.load_state_dict(state)
    wrap_one(env_model, "env")
    from asym_gemm.training.stp_layout import repack_host_weight_for_stp
    for mod_name, mod in env_model.named_modules():
        hw = getattr(mod, "host_weight", None)
        if hw is not None:
            repack_host_weight_for_stp(hw, mod_name)
    # transplant the SAME adapter values (names match ref exactly)
    env_named_all = dict(env_model.named_parameters())
    ref_init = torch.load(init_path, weights_only=False)["params"]
    with torch.no_grad():
        for n, v in ref_init.items():
            if n in env_named_all:
                env_named_all[n].copy_(v.to(env_named_all[n].device, env_named_all[n].dtype))

    torch.manual_seed(123)
    input_ids = torch.randint(0, 1024, (2, 96), device="cuda:0")
    labels = input_ids.clone()

    def run(model):
        model.train()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        return out.logits.detach().float().cpu(), float(out.loss)

    logits_ref, loss_ref = run(ref)
    logits_env, loss_env = run(env_model)
    logits_stp, loss_stp = run(stp)
    stp_post_backward_merge(stp)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    env_named = named_lora(env_model)
    envelope = {}
    for n, p in ref_named.items():
        if p.grad is None or env_named.get(n) is None or env_named[n].grad is None:
            continue
        g_ref = p.grad.float().cpu()
        g_env = env_named[n].grad.float().cpu()
        envelope[n] = ((g_env - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-8)).item()
    env_max = max(envelope.values()) if envelope else 0.0
    print(f"[mini] Phase-A ENVELOPE: logits max-abs-diff={(logits_ref - logits_env).abs().max().item():.4e}, "
          f"grad rel-err max={env_max:.3e} (loss {loss_env:.5f})")

    lerr = (logits_ref - logits_stp).abs().max().item()
    print(f"[mini] loss ref={loss_ref:.5f} stp={loss_stp:.5f}; logits max-abs-diff={lerr:.4e}")

    # grad comparison: reconstruct |1-shaped grads from the sTP pieces
    stp_named = dict(stp.named_parameters())
    pmap = stp._asym_stp_param_map
    failures = []
    for logical, meta in pmap.items():
        kind = meta["kind"]
        b1 = logical.replace(".self_attn.", ".self_attn_stp1.").replace(".mlp.", ".mlp_stp1.")
        for piece in ("lora_A", "lora_B"):
            ref_p = ref_named.get(f"{logical}.{piece}.default.weight")
            if ref_p is None or ref_p.grad is None:
                continue
            g_ref = ref_p.grad.float().cpu()
            own = stp_named.get(f"{logical}.{piece}.default.weight")
            other = stp_named.get(f"{b1}.{piece}.default.weight")
            sliced = (kind == "col" and piece == "lora_B") or (kind == "row" and piece == "lora_A")
            if sliced:
                dim = 0 if kind == "col" else 1
                g_stp = torch.cat([own.grad.float().cpu(), other.grad.float().cpu()], dim=dim)
            else:
                g_stp = own.grad.float().cpu()  # mirrors already merged
            rel = ((g_stp - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-8)).item()
            bound = max(TOL, 2.0 * env_max)  # measured envelope wins over the static band
            marker = "OK" if rel <= bound else "FAIL"
            if rel > bound:
                failures.append(f"{logical}.{piece} rel={rel:.3e} bound={bound:.3e}")
            print(f"[mini] {logical}.{piece} [{kind}] rel={rel:.3e} {marker}")
    print(f"[mini] envelope max={env_max:.3e}; full-TP bound={max(TOL, 2.0 * env_max):.3e}")
    print(f"[mini] {'PASS' if not failures and lerr < 0.15 else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures and lerr < 0.15 else 1


if __name__ == "__main__":
    sys.exit(main())
