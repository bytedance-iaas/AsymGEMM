#!/usr/bin/env python
"""Summarize route_skew probe results (surface_ep_skew.md screening).

Reads profiling_results/ep_skew/route_skew_<model>_<dataset>.json and prints,
per model, datasets ranked by median max-over-layers hot-GPU share, plus the
P95, Zipf z anchor, the domain-mean prediction for natural long packs
(§1.5a: mean histogram over docs -> hot share), and the P0 winner line.
"""

import argparse
import glob
import json
import os


def domain_mean_hot(dm):
    """dm: [L][E] mean normalized histogram -> max-over-layers hot share."""
    best = 0.5
    for layer in dm:
        e = len(layer)
        a = sum(layer[: e // 2])
        tot = sum(layer) or 1.0
        share = a / tot
        best = max(best, share, 1.0 - share)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = args.dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "profiling_results",
        "ep_skew",
    )
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "route_skew_*.json"))):
        if p.endswith("_docs.json"):
            continue
        try:
            j = json.load(open(p))
        except Exception as e:
            print(f"[warn] unreadable {os.path.basename(p)}: {e}")
            continue
        s, sp = j.get("summary", {}), j.get("spec", {})
        rows.append(
            {
                "model": sp.get("model_key"),
                "dataset": sp.get("dataset_key"),
                "med": s.get("median_max_hot_gpu_share"),
                "p95": s.get("p95_max_hot_gpu_share"),
                "z": s.get("median_zipf_z"),
                "top_e": s.get("median_top_expert_share"),
                "dm_hot": domain_mean_hot(j.get("docs", {}).get("domain_mean_hist", [])),
                "S": sp.get("num_samples"),
                "T": sp.get("seq_len"),
                "B": sp.get("batch_size"),
            }
        )
    if not rows:
        print("no results in", d)
        return
    models = sorted({r["model"] for r in rows})
    print(f"{'model':<14} {'dataset':<12} {'medHot':>7} {'p95Hot':>7} {'domHot':>7} {'z':>5} {'topE%':>6}  S xT")
    for m in models:
        sub = sorted((r for r in rows if r["model"] == m), key=lambda r: -(r["med"] or 0))
        for r in sub:
            print(
                f"{r['model']:<14} {r['dataset']:<12} {r['med']:>7.4f} {r['p95']:>7.4f} "
                f"{r['dm_hot']:>7.4f} {r['z']:>5.2f} {100 * r['top_e']:>6.2f}  {r['S']}x{r['T'] // 1000}k b{r['B']}"
            )
        win = sub[0]
        print(
            f"  -> {m} winner: {win['dataset']} (median max-hot {win['med']:.3f}, "
            f"P95 {win['p95']:.3f}, real-z anchor ~{win['z']:.2f})\n"
        )
    man = os.path.join(d, "manifest.json")
    if os.path.exists(man):
        mj = json.load(open(man))
        bad = {k: v for k, v in mj.items() if v.get("status") != "done"}
        if bad:
            print("non-done cells:")
            for k, v in sorted(bad.items()):
                print(f"  {k}: {v.get('status')} {v.get('error', '')[:120]}")


if __name__ == "__main__":
    main()
