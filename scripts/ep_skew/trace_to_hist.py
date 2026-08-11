#!/usr/bin/env python
"""Convert a route_skew probe JSON into ep_balance_bench --hist files.

For the replay samples (median + P95 by max-over-layers hot share) emit one
hist JSON whose "layers" dict holds every MoE layer of that sample:
  {"layers": {"s<sample>_L<layer>": {"counts": [...],
              "static_e2_device_share_max": <hot share>}}}
so the harness's worst/median layer picks operate within the replayed sample
(UltraEP record-and-replay protocol; surface_ep_skew.md stage 2).
"""

import argparse
import json
import os


def hot_share(counts):
    e = len(counts)
    a = sum(counts[: e // 2])
    tot = sum(counts) or 1
    return max(a / tot, 1.0 - a / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    j = json.load(open(args.probe_json))
    counts = j["samples"]["counts"]  # S,L,E
    mh = j["samples"]["max_hot_per_sample"]
    order = sorted(range(len(mh)), key=lambda s: mh[s])
    picks = {
        "median": order[len(order) // 2],
        "p95": order[max(0, round(0.95 * (len(order) - 1)))],
    }
    base = os.path.basename(args.probe_json).replace("route_skew_", "").replace(".json", "")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.probe_json))
    for tag, s in picks.items():
        layers = {}
        for li, c in enumerate(counts[s]):
            layers[f"s{s}_L{li}"] = {
                "counts": c,
                "static_e2_device_share_max": hot_share(c),
            }
        spec = {
            "source_probe_json": os.path.basename(args.probe_json),
            "sample": s,
            "sample_tag": tag,
            "sample_max_hot": mh[s],
            "model": j["spec"]["model_key"],
            "dataset": j["spec"]["dataset_key"],
        }
        p = os.path.join(out_dir, f"ep_hist_real_{base}_{tag}.json")
        with open(p, "w") as f:
            json.dump({"spec": spec, "layers": layers}, f)
        print(p, f"sample={s} max_hot={mh[s]:.4f} layers={len(layers)}")


if __name__ == "__main__":
    main()
