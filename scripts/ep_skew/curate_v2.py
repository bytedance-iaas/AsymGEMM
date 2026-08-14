"""Aligned-cluster curation v2: maximize LAYER-AVERAGED hot share of 1M packs
by (a) sign-vector power iteration to find the best-aligned doc cluster,
(b) alignment-weighted REPETITION to fill 1M tokens (repetition preserves the
routing distribution; the house data pipeline already concatenates/repeats).

Emits pack-file JSONs for route_skew_probe.py --pack-file (docs listed with
explicit repeats; verifier cycles if still short).

Usage:
  python3 curate_v2.py --model glm4.7-flash --deep-dir profiling_results/ep_skew_deep \
    --set mathmix=dapo,megamath,openscience --set codemix=codeforces,swebench \
    --set allmix=... --top-k 4 --pack-tokens 1000000 --packs 6 \
    --out-prefix profiling_results/ep_skew_deep/packs2
"""
import argparse
import gzip
import json
import os

import numpy as np


def load_pool(deep_dir, model, datasets, top_k):
    ids, sigs, toks = [], [], []
    for ds in datasets:
        p = os.path.join(deep_dir, f"route_skew_{model}_{ds}_docs.json.gz")
        if not os.path.exists(p):
            print(f"  !! missing {ds}")
            continue
        for did, LE in json.load(gzip.open(p, "rt")).items():
            A = np.asarray(LE, dtype=np.float64)
            t = A.sum(1)
            if t.min() <= 0 or t.max() < 400:
                continue
            ids.append((ds, str(did)))
            sigs.append(A[:, : A.shape[1] // 2].sum(1) / t - 0.5)
            toks.append(t[0] / top_k)
    return ids, np.array(sigs), np.array(toks)


def aligned(S, W, min_tokens, seeds=20, iters=12):
    best = None
    for seed in np.argsort(-np.abs(S).mean(1))[:seeds]:
        sig = np.sign(S[seed]); sig[sig == 0] = 1
        sel = None
        for _ in range(iters):
            order = np.argsort(-(S @ sig))
            k = int(np.searchsorted(np.cumsum(W[order]), min_tokens)) + 1
            sel = order[:k]
            acc = (S[sel] * W[sel, None]).sum(0)
            new = np.sign(acc); new[new == 0] = 1
            if (new == sig).all():
                break
            sig = new
        avg = np.abs((S[sel] * W[sel, None]).sum(0)).mean() / W[sel].sum()
        if best is None or avg > best[0]:
            best = (avg, sel, sig)
    return best


def weighted_fill(S, W, sel, sig, pack_tokens, max_doc_frac=0.2):
    """Fractional-optimal repetition under a per-doc token cap: assign repeat
    mass to docs in alignment order, each up to max_doc_frac of the pack, then
    everyone else once. (With direction fixed, the objective is linear-
    fractional in the weights, so mass concentrates on the best docs.)"""
    # S rows are per-layer SHARES (already per-token), so a doc's alignment
    # quality is just its mean signed agreement with sig — no length division.
    align = S[sel] @ sig
    order = sel[np.argsort(-align)]
    cap = max_doc_frac * pack_tokens
    remaining = pack_tokens - W[order].sum()  # keep one natural copy of each
    mass = W[order].astype(float).copy()
    if remaining > 0:
        for i in range(len(order)):
            extra = min(cap - mass[i], remaining)
            if extra > 0:
                mass[i] += extra
                remaining -= extra
            if remaining <= 0:
                break
    reps = mass / W[order]
    acc = (S[order] * mass[:, None]).sum(0)
    avg = np.abs(acc).mean() / mass.sum()
    per_layer = 0.5 + np.abs(acc) / mass.sum()
    return order, reps, avg, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deep-dir", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--pack-tokens", type=int, default=1000000)
    ap.add_argument("--packs", type=int, default=6)
    ap.add_argument("--min-unique", type=float, default=50e3)
    ap.add_argument("--out-prefix", default="packs2")
    ap.add_argument("--max-doc-frac", type=float, default=0.2)
    ap.add_argument("--exclude-packs", action="append", default=[],
                    help="pack-file JSON(s) whose docs are removed from the pool "
                    "(forces a DISTINCT cluster)")
    args = ap.parse_args()

    excluded = set()
    for xp in args.exclude_packs:
        for p in json.load(open(xp))["packs"]:
            for ds, did in p["docs"]:
                excluded.add((ds, str(did)))
    if excluded:
        print(f"excluding {len(excluded)} docs from prior sets")

    for spec in args.set:
        name, dss = spec.split("=", 1)
        datasets = [x.strip() for x in dss.split(",") if x.strip()]
        ids, S, W = load_pool(args.deep_dir, args.model, datasets, args.top_k)
        if excluded and len(ids):
            keep = [i for i, d in enumerate(ids) if d not in excluded]
            ids = [ids[i] for i in keep]; S = S[keep]; W = W[keep]
        if not len(ids):
            print(f"SET {name}: empty"); continue
        print(f"SET {name}: pool {len(ids)} docs {W.sum()/1e6:.1f}M tok")
        best_cfg = None
        for mt in [10e3, 20e3, 30e3, 50e3, 100e3]:
            r = aligned(S, W, mt)
            if r is None:
                continue
            avg0, sel, sig = r
            order, reps, avg, per_layer = weighted_fill(S, W, sel, sig, args.pack_tokens, args.max_doc_frac)
            print(f"  unique~{mt/1e3:.0f}k: cluster avg={0.5+avg0:.4f} "
                  f"-> weighted-fill avg={0.5+avg:.4f} min_l={per_layer.min():.4f}")
            if best_cfg is None or avg > best_cfg[0]:
                best_cfg = (avg, order, reps, per_layer, mt)
        avg, order, reps, per_layer, mt = best_cfg
        print(f"  BEST: avg_hot={0.5+avg:.4f} "
              f"{'>=0.65 TARGET MET' if 0.5+avg >= 0.65 else '<0.65 (ceiling)'} "
              f"(unique~{mt/1e3:.0f}k, docs={len(order)})")
        rng = np.random.default_rng(7)
        packs = []
        for pi in range(args.packs):
            perm = rng.permutation(len(order))
            docs = []
            for j in perm:
                n = int(round(reps[j]))
                docs.extend([list(ids[order[j]])] * max(1, n))
            packs.append({
                "docs": docs,
                "pred_layer_avg_hot": round(0.5 + float(avg), 4),
                "pred_layer_min_hot": round(float(per_layer.min()), 4),
                "pred_layer_max_hot": round(float(per_layer.max()), 4),
                "unique_docs": int(len(order)),
            })
        out = f"{args.out_prefix}_{args.model}_{name}.json"
        json.dump({"model": args.model, "set": name, "datasets": datasets,
                   "pack_tokens": args.pack_tokens, "algo": "aligned+weighted-repeat",
                   "packs": packs}, open(out, "w"))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
