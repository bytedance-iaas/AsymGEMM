"""Greedy curation of 1M-token packs that maximize LAYER-AVERAGED hot-GPU
share under the contiguous E/2 split (EP=2), from banked per-doc router
signatures (route_skew_<model>_<ds>_docs.json.gz).

Objective per pack P (docs d, token weights w_d, per-layer signed deviation
s_{d,l} = shareA - 0.5): maximize  mean_l | sum_d w_d s_{d,l} / sum_d w_d |.
Greedy: seed with the doc of max mean|s|, then repeatedly add the doc that
maximizes the objective, until the token budget; docs are disjoint across
packs of one set.

Usage:
  python3 curate_packs.py --model glm4.7-flash --deep-dir .../ep_skew_deep \
      --set mathmix=dapo,megamath,openscience --set codemix=codeforces,swebench \
      --set allmix=dapo,codeforces,swebench,openscience,megamath,sft_mix,longbench \
      --pack-tokens 1000000 --packs 6 --out packs_glm4.7-flash.json
Prints predicted per-pack layer-avg (and min/max layer) and writes the
pack-file JSON for route_skew_probe.py --pack-file.
"""

import argparse
import glob
import gzip
import json
import os

import numpy as np


def load_pool(deep_dir, model, datasets):
    ids, sigs, toks, srcs = [], [], [], []
    for ds in datasets:
        p = os.path.join(deep_dir, f"route_skew_{model}_{ds}_docs.json.gz")
        if not os.path.exists(p):
            print(f"  !! missing {p} — skipping {ds}")
            continue
        d = json.load(gzip.open(p, "rt"))
        for did, LE in d.items():
            A = np.asarray(LE, dtype=np.float64)  # L x E
            t = A.sum(1)
            if t.min() <= 0 or t.max() < 400:  # drop fragments (<~100 tokens)
                continue
            E = A.shape[1]
            s = A[:, : E // 2].sum(1) / t - 0.5
            ids.append(did)
            sigs.append(s)
            # every MoE layer sees all of the doc's tokens, so layer 0's count
            # sum = tokens * top_k; divide by top_k later (main).
            toks.append(float(t[0]))
            srcs.append(ds)
    return ids, np.array(sigs), np.array(toks), srcs


def greedy_pack(sig, tok, avail, budget):
    """Pick doc indices (from avail mask) maximizing mean_l |weighted s| until budget."""
    L = sig.shape[1]
    cand = np.where(avail)[0]
    base = np.abs(sig[cand]).mean(1)
    seed = cand[int(np.argmax(base))]
    chosen = [seed]
    avail[seed] = False
    acc = sig[seed] * tok[seed]
    wsum = tok[seed]
    while wsum < budget:
        cand = np.where(avail)[0]
        if len(cand) == 0:
            break
        # objective if added: mean_l |acc + s_c*w_c| / (wsum + w_c)
        num = np.abs(acc[None, :] + sig[cand] * tok[cand, None]).mean(1)
        obj = num / (wsum + tok[cand])
        j = cand[int(np.argmax(obj))]
        chosen.append(j)
        avail[j] = False
        acc = acc + sig[j] * tok[j]
        wsum += tok[j]
    layer = np.abs(acc) / max(wsum, 1e-9)
    return chosen, wsum, 0.5 + layer  # per-layer hot share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deep-dir", required=True)
    ap.add_argument("--set", action="append", required=True,
                    help="name=ds1,ds2,...")
    ap.add_argument("--pack-tokens", type=int, default=1000000)
    ap.add_argument("--packs", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=4, help="model top_k (tokens = counts/k)")
    ap.add_argument("--out-prefix", default="packs")
    args = ap.parse_args()

    for spec in args.set:
        name, dss = spec.split("=", 1)
        datasets = [x.strip() for x in dss.split(",") if x.strip()]
        ids, sigs, toks_raw, srcs = load_pool(args.deep_dir, args.model, datasets)
        if not ids:
            print(f"SET {name}: no docs — skipped")
            continue
        # tokens per doc = (sum of counts at layer0) / top_k; recompute cleanly:
        toks = toks_raw / args.top_k
        avail = np.ones(len(ids), dtype=bool)
        packs_out = []
        print(f"SET {name}: pool {len(ids)} docs, {toks.sum()/1e6:.1f}M tokens "
              f"from {datasets}")
        for pi in range(args.packs):
            chosen, wsum, layer_hot = greedy_pack(sigs, toks, avail, args.pack_tokens)
            packs_out.append({
                "docs": [ids[j] for j in chosen],
                "sources": sorted({srcs[j] for j in chosen}),
                "tokens_est": int(wsum),
                "pred_layer_avg_hot": round(float(layer_hot.mean()), 4),
                "pred_layer_min_hot": round(float(layer_hot.min()), 4),
                "pred_layer_max_hot": round(float(layer_hot.max()), 4),
            })
            print(f"  pack {pi}: docs={len(chosen)} tok~{wsum/1e6:.2f}M "
                  f"pred avg={layer_hot.mean():.4f} min={layer_hot.min():.4f} "
                  f"max={layer_hot.max():.4f}")
        avg = np.mean([p["pred_layer_avg_hot"] for p in packs_out])
        print(f"  SET {name} MEAN over {len(packs_out)} packs: {avg:.4f} "
              f"{'>= 0.65 TARGET MET' if avg >= 0.65 else '< 0.65'}")
        out = f"{args.out_prefix}_{args.model}_{name}.json"
        json.dump({"model": args.model, "set": name, "datasets": datasets,
                   "pack_tokens": args.pack_tokens, "packs": packs_out},
                  open(out, "w"), indent=1)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
