"""Summarize the EP-skew screen: per-cell metrics + stage-1.5 long-pack
prediction (domain-mean histogram hot share) from the recorded JSONs.
CPU-only; reads profiling_results/ep_skew/route_skew_*.json."""

import glob
import json
import os
import sys

import math

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "profiling_results", "ep_skew")


def domain_mean_hot(dom_hist):
    """dom_hist: L x E normalized mean histogram. Returns max-over-layers
    hot-GPU share of the MEAN distribution (the natural long-pack prediction:
    a 1M pack averages many docs, so its launch histogram converges to this)."""
    best = 0.0
    for layer in dom_hist:
        e = len(layer)
        tot = sum(layer) or 1.0
        a = sum(layer[: e // 2]) / tot
        best = max(best, max(a, 1.0 - a))
    return best


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "route_skew_*_*.json"))):
        name = os.path.basename(p)
        if name.endswith("_docs.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"?? {name}: {e}", file=sys.stderr)
            continue
        spec, summ = d.get("spec", {}), d.get("summary", {})
        if not summ:
            continue
        dm = d.get("docs", {}).get("domain_mean_hist")
        rows.append({
            "model": spec.get("model_key"),
            "dataset": spec.get("dataset_key"),
            "n": spec.get("num_samples"),
            "med": summ.get("median_max_hot_gpu_share"),
            "p95": summ.get("p95_max_hot_gpu_share"),
            "z": summ.get("median_zipf_z"),
            "top": summ.get("median_top_expert_share"),
            "dm_hot": domain_mean_hot(dm) if dm else None,
        })
    rows.sort(key=lambda r: (r["model"], -(r["med"] or 0)))
    print(f"{'model':14s} {'dataset':12s} {'n':>4s} {'med_hot':>8s} {'p95':>7s} "
          f"{'z':>5s} {'top_exp':>8s} {'1M_pred':>8s}")
    cur = None
    for r in rows:
        if r["model"] != cur:
            cur = r["model"]
            print("-" * 72)
        print(f"{r['model']:14s} {r['dataset']:12s} {r['n']:4d} "
              f"{r['med']:.4f}  {r['p95']:.4f} {r['z']:.2f} "
              f"{(r['top'] or 0):.4f}  "
              f"{('%.4f' % r['dm_hot']) if r['dm_hot'] else '   n/a'}")


if __name__ == "__main__":
    main()
