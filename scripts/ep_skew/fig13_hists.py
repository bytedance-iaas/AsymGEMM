#!/usr/bin/env python
"""Build ep_balance_bench --hist inputs for Figure 13 (fig:ablation-balancer).

Per (model, set): take the verified curated pack whose gemm-avg is closest to
the set mean (the packs are near-identical), and emit THREE hists over ALL MoE
layers — identical count multisets per layer, differing only in expert-id
labeling so the bench's `owned` mode (rank r owns experts [r*E/2,(r+1)*E/2))
realizes three different static placements:

  placed : the routing screen's calibrated per-layer placement (the shipped
           partition_<model>_<set>.json) relabeled to contiguous — "Static EP"
           on a domain-shifted workload (measured 0.66-0.86 hot share).
  oracle : hindsight-best BALANCED per-layer split, computed on the full
           6-pack run's counts (greedy heaviest->lighter half, |A|=E/2) —
           "Static EP (Oracle)", the ceiling of repartitioning schemes.
  contig : counts as-is (default contiguous split) — reference point.

`plan` mode (DSEP) is labeling-invariant (it re-derives the cut from the
union counts), so it runs once, on the placed hist.
"""

import argparse
import json
import os

import numpy as np


def greedy_balance(counts, cap):
    order = np.argsort(-counts)
    a, la, lb, na, nb = [], 0.0, 0.0, 0, 0
    b = []
    for e in order:
        if (la <= lb and na < cap) or nb >= cap:
            a.append(int(e)); la += counts[e]; na += 1
        else:
            b.append(int(e)); lb += counts[e]; nb += 1
    return sorted(a)


def hot(counts, A):
    s = counts[A].sum() / max(counts.sum(), 1)
    return max(s, 1 - s)


def relabel(counts_LE, parts):
    """Permute expert ids per layer so parts[l] -> [0, E/2)."""
    L, E = counts_LE.shape
    out = np.empty_like(counts_LE)
    for l in range(L):
        rest = [e for e in range(E) if e not in set(parts[l])]
        perm = list(parts[l]) + rest          # new_id -> old_id
        out[l] = counts_LE[l][perm]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placed-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", dest="set_", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    j = json.load(open(os.path.join(
        args.placed_dir, f"route_skew_{args.model}_placed_{args.set_}_1m.json")))
    C = np.asarray(j["samples"]["counts"], dtype=np.int64)  # 6,L,E
    S, L, E = C.shape
    part = json.load(open(os.path.join(
        args.placed_dir, f"partition_{args.model}_{args.set_}.json")))["layers"]

    la = j["summary"].get("layer_avg_hot_per_sample")
    if la:
        pick = int(np.argmin(np.abs(np.asarray(la) - float(np.mean(la)))))
    else:
        pick = 0
    Cp = C[pick]  # L,E — the representative pack (one real step's routing)
    run_tot = C.sum(0)  # hindsight over the whole 6-pack run

    oracle = [greedy_balance(run_tot[l].astype(np.float64), E // 2) for l in range(L)]

    variants = {
        "placed": relabel(Cp, part),
        "oracle": relabel(Cp, oracle),
        "contig": Cp,
    }
    meta = {"model": args.model, "set": args.set_, "pack": pick,
            "L": L, "E": E, "hot": {}}
    A = np.arange(E // 2)
    for name, cnt in variants.items():
        hots = [hot(cnt[l].astype(np.float64), A) for l in range(L)]
        meta["hot"][name] = {"mean": round(float(np.mean(hots)), 4),
                             "max": round(float(np.max(hots)), 4)}
        layers = {f"L{l:02d}": {"counts": cnt[l].tolist(),
                                "static_e2_device_share_max": hots[l]}
                  for l in range(L)}
        p = os.path.join(args.out_dir,
                         f"fig13_{args.model}_{args.set_}_{name}.json")
        json.dump({"spec": {**meta, "labeling": name}, "layers": layers},
                  open(p, "w"))
    meta_p = os.path.join(args.out_dir, f"fig13_{args.model}_{args.set_}_meta.json")
    json.dump(meta, open(meta_p, "w"), indent=1)
    print(f"{args.model}/{args.set_}: pack {pick}  gemm-avg placed={meta['hot']['placed']['mean']} "
          f"oracle={meta['hot']['oracle']['mean']} contig={meta['hot']['contig']['mean']} "
          f"(L={L}, E={E})")


if __name__ == "__main__":
    main()
