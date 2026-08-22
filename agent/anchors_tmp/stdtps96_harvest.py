#!/usr/bin/env python3
"""Harvest 96G-campaign cells (tags s96*): eff tok/s (GLOBAL for 2r =
ranks x per-rank), peak resv GiB + % of 95.6, total host RSS GB (summed
across rank dirs) + HOST>909G flag. Usage: stdtps96_harvest.py [TAGGLOB...]
Run from repo root."""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
BUDGET_GB = 909.4
pats = sys.argv[1:] or ["s96"]
out = {}
for pat in pats:
    for d in sorted(glob.glob(f"{B}/{pat}*")):
        base = os.path.basename(d)
        m = re.match(r"(s96[a-z0-9_]+?)_([a-z0-9_.\-]+)__b(\d+)_s(\d+)_ga(\d+)", base)
        if not m:
            continue
        tag, dmodel, b, s, ga = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        ranks = 2 if re.match(r"s96(air|fl|hy|q122|g20|q30|q35|mx)2", tag) else 1
        effs, resvs, rsss = [], [], []
        for rd in {os.path.dirname(ss) for ss in glob.glob(f"{d}/**/step_samples.csv", recursive=True)}:
            steps = []
            for row in csv.DictReader(open(os.path.join(rd, "step_samples.csv"))):
                warm = str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}
                ms = float(row.get("step_milliseconds") or 0)
                if ms > 0 and not warm:
                    steps.append(ms)
            if steps:
                effs.append((len(steps) * b * s * ga) / (sum(steps) / 1000.0))
            spf = os.path.join(rd, "source_profile.json")
            if not os.path.exists(spf):
                spf = os.path.join(rd, "source_profile.partial.json")
            if os.path.exists(spf):
                try:
                    J = json.load(open(spf))
                    mem = J.get("memory", {}) or {}
                    rv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                    if rv:
                        resvs.append(rv)
                    rss = mem.get("peak_rss_bytes", 0) / 1e9
                    if rss:
                        rsss.append(rss)
                except Exception:
                    pass
        if not effs:
            continue
        eff = ranks * sum(effs) / len(effs)
        resv = max(resvs) if resvs else 0.0
        rss_tot = sum(rsss) if rsss else 0.0
        key = (tag, s)
        if key not in out or eff > out[key][1]:
            out[key] = (b, eff, resv, rss_tot, ranks)
for (tag, s), (b, eff, resv, rss, ranks) in sorted(out.items()):
    flag = "  HOST>909G!" if rss > BUDGET_GB else ""
    print(f"{tag:16s} s={s:<8d} b{b} r{ranks} eff={eff:8.1f}  "
          f"resv={resv:5.1f}G ({resv/95.6*100:4.1f}%)  rss_tot={rss:6.1f}GB{flag}")
