#!/usr/bin/env python3
"""Compare two adapter-grad dumps (G-D2.1 / I4 step-2 parity). rel-err floor 1e-8, bf16 band 1e-2."""
import argparse, sys, torch

p = argparse.ArgumentParser(); p.add_argument("ref"); p.add_argument("test"); p.add_argument("--tol", type=float, default=1e-2)
p.add_argument("--scale", type=float, default=1.0, help="multiply TEST grads (e.g. DP mean vs sum conventions)")
a = p.parse_args()
ref, test = torch.load(a.ref, weights_only=False), torch.load(a.test, weights_only=False)
rg, tg = ref["grads"], test["grads"]
missing = set(rg) ^ set(tg)
worst, worst_name, n_bad = 0.0, "", 0
for name in sorted(set(rg) & set(tg)):
    r, t = rg[name], tg[name] * a.scale
    err = ((t - r).abs().max() / r.abs().max().clamp_min(1e-8)).item()
    if err > worst: worst, worst_name = err, name
    if err > a.tol: n_bad += 1
print(f"compared {len(set(rg)&set(tg))} params; missing-on-one-side: {len(missing)}")
print(f"worst max-rel-err {worst:.3e} @ {worst_name}; over-tol({a.tol}): {n_bad}")
print("PASS" if (n_bad == 0 and not missing) else "FAIL")
sys.exit(0 if (n_bad == 0 and not missing) else 1)
