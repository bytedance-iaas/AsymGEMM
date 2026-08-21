#!/usr/bin/env python3
"""Harvest stdz campaign cells (tags s1*/s2*/s3*/s4*): eff tok/s per house
rule (w1+m2, non-warmup steps; GLOBAL tok/s for rank-2 tags s2*), winning
batch per tag = highest-tok/s TRAINED batch dir with steps.

Usage: stdz_harvest.py [TAGGLOB ...]   (default: s1 s2)
Run from repo root (host or container — reads profiling_results symlink).
"""
import csv, glob, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
pats = sys.argv[1:] or ["s1", "s2"]
best = {}
for pat in pats:
    for d in sorted(glob.glob(f"{B}/{pat}*")):
        base = os.path.basename(d)
        tag = base.split("_", 2)
        # tag format: sN<name>_<model>__b..; recover full tag prefix before _<dmodel>__
        m = re.match(r"(s\d[a-z0-9_]+?)_([a-z0-9_.\-]+)__b(\d+)_s(\d+)_ga(\d+)", base)
        if not m:
            continue
        tag, dmodel, b, s, ga = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        ranks = 2 if tag.startswith("s2") or tag.startswith("s4x2") else 1
        for ss in glob.glob(f"{d}/**/step_samples.csv", recursive=True):
            steps = []
            for row in csv.DictReader(open(ss)):
                warm = str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}
                ms = float(row.get("step_milliseconds") or 0)
                if ms > 0 and not warm:
                    steps.append(ms)
            if not steps:
                continue
            eff = ranks * (len(steps) * b * s * ga) / (sum(steps) / 1000.0)
            key = (tag, s)
            if key not in best or eff > best[key][2]:
                best[key] = (b, ranks, eff, len(steps))
for (tag, s), (b, ranks, eff, n) in sorted(best.items()):
    print(f"{tag:16s} s={s:<8d} b={b} r={ranks} steps={n} eff={eff:8.1f} tok/s")
