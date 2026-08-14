#!/usr/bin/env python
"""Assemble the measured MoE-share-vs-length curve (Kevin 2026-08-14).

Per (model, seq): step time from the real 2-step b1 anchor cell's
step_samples.json (non-warmup mean), expert walls from the math-domain
placed-hist bench at that length's true launch size. Reports expert-GEMM
share of step (F=2) and the DSEP end-to-end gain vs the placed static split.
"""

import glob
import json
import os

REPO = "/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM"
PROF = f"{REPO}/profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
H = f"{REPO}/profiling_results/ep_skew_deep/fig13"
F = 2.0

MODELS = [("qwen3-30b", "q30b"), ("glm4.7-flash", "gflash"), ("qwen3.5-122b", "q122b")]
SEQS = [48000, 64000, 80000, 100000, 128000]
TAG128 = {"q30b": "f128q30b", "gflash": "f128gflash", "q122b": "f128q122b"}


def step_seconds(tag, seq):
    d = f"{PROF}/{tag}__b1_s{seq}_ga1_drop000"
    ss = glob.glob(f"{d}/**/step_samples.json", recursive=True)
    if not ss:
        return None
    j = json.load(open(ss[0]))
    ms = [s["step_milliseconds"] for s in j]
    ms = ms[1:] if len(ms) > 1 else ms  # first = warmup
    return sum(ms) / len(ms) / 1000.0


def walls(model, seq):
    p = (f"{H}/walls128_{model}_math_placed.json" if seq == 128000
         else f"{H}/wallsS{seq}_{model}_math_placed.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {m: sum(c[m]["wall_s"] for c in d["cases"]) for m in d["modes"]}


def main():
    print(f"{'model':<13} {'seq':>6} {'step_s':>8} {'tok/s':>6} {'Wdsep':>6} "
          f"{'Wstat':>6} | {'share%':>6} {'e2e_gain%':>9}")
    out = {}
    for model, short in MODELS:
        rows = []
        for seq in SEQS:
            tag = TAG128[short] if seq == 128000 else f"fs{seq // 1000}{short}"
            st = step_seconds(tag, seq)
            W = walls(model, seq)
            if st is None or W is None:
                print(f"{model:<13} {seq:>6}  (pending)")
                continue
            toks = 2 * seq
            share = F * W["plan"] / st
            step_static = st + F * (W["owned"] - W["plan"])
            gain = st and (step_static / st - 1)
            rows.append({"seq": seq, "step_s": round(st, 2),
                         "tok_s": round(toks / st, 1),
                         "W_dsep": round(W["plan"], 3),
                         "W_static": round(W["owned"], 3),
                         "share": round(share, 4),
                         "e2e_gain": round(gain, 4)})
            print(f"{model:<13} {seq:>6} {st:>8.2f} {toks / st:>6.0f} "
                  f"{W['plan']:>6.3f} {W['owned']:>6.3f} | {100 * share:>5.1f}% "
                  f"{100 * gain:>8.1f}%")
        out[model] = rows
    p = f"{H}/share_vs_length.json"
    json.dump(out, open(p, "w"), indent=1)
    print("->", p)


if __name__ == "__main__":
    main()
