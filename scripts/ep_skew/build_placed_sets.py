#!/usr/bin/env python
"""Build (placement, 1M packs) pairs per curated set — the 65/35 constructor.

Analysis (SKEW_CAMPAIGN.md 08-12): doc mixing under the CONTIGUOUS split is
ceiling-capped at ~0.58 layer-avg hot, but a FIXED per-layer placement
calibrated on a domain's traces puts each layer's hot experts on one side and
holds 0.76-0.93 layer-avg on held-out samples. This script:

  1. splits the deep-trace doc pool deterministically (sha1 parity) into
     CALIBRATION docs and PACK docs (never mixed),
  2. derives the per-layer top-E/2 placement from calibration counts only,
  3. greedily fills N disjoint ~1M-token packs from PACK docs ranked by
     token-weighted mean placed share (snake draft keeps packs comparable),
  4. writes partition JSON (per-layer, probe-compatible) + pack-file JSON
     (route_skew_probe.py --pack-file schema) + predictions.

Real-forward verification then runs, e.g.:
  route_skew_probe.py --model qwen3-30b --pack-file <packs.json> \
      --partition <partition.json> --seq-len 1048576 --batch-size 1
"""

import argparse
import glob
import gzip
import hashlib
import json
import os

import numpy as np


def load_docs(deep_dir, model, datasets):
    ids, cnts, toks, srcs = [], [], [], []
    for ds in datasets:
        p = os.path.join(deep_dir, f"route_skew_{model}_{ds}_docs.json.gz")
        if not os.path.exists(p):
            print(f"  !! missing {p} — {ds} skipped")
            continue
        d = json.load(gzip.open(p, "rt"))
        for did, LE in d.items():
            A = np.asarray(LE, dtype=np.float32)
            t = A.sum(1)
            if t.min() <= 0:
                continue
            ids.append(did)
            cnts.append(A)
            toks.append(float(t[0]))
            srcs.append(ds)
    return ids, cnts, np.array(toks), srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deep-dir", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=ds1,ds2")
    ap.add_argument("--pack-tokens", type=int, default=1_000_000)
    ap.add_argument("--packs", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-doc-tokens", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for spec in args.set:
        name, dss = spec.split("=", 1)
        datasets = [x.strip() for x in dss.split(",") if x.strip()]
        ids, cnts, toks_raw, srcs = load_docs(args.deep_dir, args.model, datasets)
        if not ids:
            print(f"SET {name}: empty pool"); continue
        toks = toks_raw / args.top_k
        keep = toks >= args.min_doc_tokens
        L, E = cnts[0].shape
        cal_mask = np.array(
            [int(hashlib.sha1(f"{s}|{i}".encode()).hexdigest(), 16) % 2 == 0
             for s, i in zip(srcs, ids)]
        )
        cal_idx = np.where(keep & cal_mask)[0]
        pack_idx = np.where(keep & ~cal_mask)[0]

        cal_tot = np.zeros((L, E), dtype=np.float64)
        for j in cal_idx:
            cal_tot += cnts[j]
        place = np.argsort(-cal_tot, axis=1)[:, : E // 2]
        mask = np.zeros((L, E), dtype=bool)
        for l in range(L):
            mask[l, place[l]] = True

        # per pack-doc placed share (token-weighted mean over layers)
        shares = np.empty((len(pack_idx), L), dtype=np.float64)
        for r, j in enumerate(pack_idx):
            A = cnts[j]
            shares[r] = (A * mask).sum(1) / np.maximum(A.sum(1), 1)
        score = shares.mean(1)
        order = np.argsort(-score)

        # snake draft into N disjoint packs until token budget
        packs = [[] for _ in range(args.packs)]
        ptok = np.zeros(args.packs)
        direction, pi = 1, 0
        for r in order:
            j = pack_idx[r]
            if ptok.min() >= args.pack_tokens:
                break
            tries = 0
            while ptok[pi] >= args.pack_tokens and tries <= args.packs:
                pi += direction
                if pi in (-1, args.packs):
                    direction *= -1
                    pi += direction
                tries += 1
            packs[pi].append(j)
            ptok[pi] += toks[j]
            pi += direction
            if pi in (-1, args.packs):
                direction *= -1
                pi += direction

        part_path = os.path.join(args.out_dir, f"partition_{args.model}_{name}.json")
        json.dump(
            {"model": args.model, "set": name, "datasets": datasets,
             "type": "per-layer-top-half-from-calibration",
             "calibration_docs": int(len(cal_idx)),
             "layers": [sorted(int(e) for e in place[l]) for l in range(L)]},
            open(part_path, "w"))

        out_packs = []
        for pidx, doc_js in enumerate(packs):
            A = np.zeros((L, E), dtype=np.float64)
            for j in doc_js:
                A += cnts[j]
            share = (A * mask).sum(1) / np.maximum(A.sum(1), 1)
            out_packs.append({
                "docs": [[srcs[j], ids[j]] for j in doc_js],
                "tokens_est": int(ptok[pidx]),
                "pred_layer_avg_hot": round(float(share.mean()), 4),
                "pred_layer_min_hot": round(float(share.min()), 4),
                "pred_layer_max_hot": round(float(share.max()), 4),
            })
            print(f"  pack {pidx}: docs={len(doc_js)} tok~{ptok[pidx]/1e6:.2f}M "
                  f"pred avg={share.mean():.4f} min={share.min():.4f} max={share.max():.4f}")
        pk_path = os.path.join(args.out_dir, f"packs_{args.model}_{name}.json")
        json.dump({"model": args.model, "set": name, "datasets": datasets,
                   "partition": os.path.basename(part_path),
                   "pack_tokens": args.pack_tokens, "packs": out_packs},
                  open(pk_path, "w"), indent=1)
        avg = float(np.mean([p["pred_layer_avg_hot"] for p in out_packs]))
        print(f"SET {name}: pool cal={len(cal_idx)} pack={len(pack_idx)} docs; "
              f"MEAN pred layer-avg over {args.packs} packs = {avg:.4f} "
              f"({'TARGET MET' if avg >= 0.65 else 'below 0.65'}) -> {pk_path}")


if __name__ == "__main__":
    main()
