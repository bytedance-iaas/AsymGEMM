#!/usr/bin/env python
"""Vanilla LoRA-SFT reference step for gpt-oss-20b (NOT the AsymLoRA stack —
gpt-oss is not integrated there; this is the denominator for its DSEP e2e
composition, labeled as such).

Single GPU, bf16, LoRA (r=8) on attention projections only, experts frozen
(matches the campaign's frozen-expert semantics), causal-LM loss on curated
pack tokens, fwd+bwd+opt step timed with CUDA events. MAX 3 steps, first
discarded.

Usage: gptoss_ref_step.py --seq 16384 --batch 8 --pack-file <packs.json>
"""

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--pack-file", default=None,
                    help="curated pack json (docs rebuilt); default: random tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    repo = "openai/gpt-oss-20b"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager",  # gpt_oss sinks: no SDPA in this hf build
    )
    # gradient checkpointing REQUIRED: gpt_oss eager attention (sinks) retains
    # [B,H,S,S] score tensors per full-attn layer for backward — 12 layers x
    # 65 GiB at 8k b8 without GC. With GC the experts run fwd + recompute +
    # dgrad => compose with F=3 for this model (vs F=2 in the asym cells).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()

    if args.pack_file:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from route_skew_probe import build_packs_from_file
        res = build_packs_from_file(args.pack_file, tok, args.seq)
        plist = res[0] if isinstance(res, tuple) else res
        toks = [p.tokens for p in plist]
        while len(toks) < args.batch:
            toks = toks + toks
        ids = torch.tensor([t[: args.seq] for t in toks[: args.batch]], device="cuda")
    else:
        ids = torch.randint(100, 20000, (args.batch, args.seq), device="cuda")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    times = []
    for i in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"step {i}: {times[-1]:.2f}s loss {out.loss.item():.3f}", flush=True)
    mean = sum(times[1:]) / max(1, len(times) - 1)
    res = {"seq": args.seq, "batch": args.batch, "steps": times,
           "mean_step_s": round(mean, 3),
           "tok_s_per_gpu": round(args.seq * args.batch / mean, 1),
           "note": "vanilla transformers+PEFT reference (grad ckpt, LoRA attn, frozen experts), single GPU"}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
