#!/usr/bin/env python3
"""Harvest hunyuan campaign cells (hy1r_*, hy2r_*, hyx_*, hysmk*) -> table+JSON.
Effective tok/s per house rule (post-warmup, w1+m2), GLOBAL for 2-rank."""
import csv, glob, json, os, re

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
rows = []
for cfg in sorted(glob.glob(f"{B}/hy*-c17_hunyuan-a13b__*")):
    tag = os.path.basename(cfg).split("-c17_")[0]
    ranks = 2 if tag.startswith("hy2r") else 1
    for ss in glob.glob(f"{cfg}/**/step_samples.csv", recursive=True):
        rd = os.path.dirname(ss)
        m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
        if not m:
            continue
        b, s, ga = map(int, m.groups())
        meas = [float(r["step_milliseconds"]) for r in csv.DictReader(open(ss))
                if r.get("is_warmup", "").lower() not in ("true", "1") and float(r.get("step_milliseconds") or 0) > 0]
        if not meas:
            continue
        eff = ranks * len(meas) * b * s * ga / (sum(meas) / 1000.0)
        resv = rss = 0.0
        pj = os.path.join(rd, "profile.json")
        if os.path.exists(pj):
            try:
                mem = json.load(open(pj)).get("memory", {})
                resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                rss = mem.get("process", {}).get("rss_peak_bytes", 0) / GIB
            except Exception:
                pass
        sysname = os.path.basename(os.path.dirname(rd)).split("__")[0]
        token = os.path.basename(os.path.dirname(rd)).split("__")[2]
        rows.append(dict(tag=tag, ranks=ranks, system=sysname, token=token, seq=s,
                         batch=b, eff=round(eff), resv=round(resv, 1), rss=round(rss)))

rows.sort(key=lambda r: (r["ranks"], r["system"], r["token"], r["seq"], r["batch"]))
for r in rows:
    print(f"r{r['ranks']} {r['tag']:14s} {r['system'][:20]:20s} {r['token'][:26]:26s} "
          f"s={r['seq']:>6} b{r['batch']:<2} eff={r['eff']:>5} resv={r['resv']:>6} rss={r['rss']:>4}")
json.dump(rows, open("agent/anchors_tmp/hy_cells.json", "w"), indent=1)
print(f"[{len(rows)} cells] -> agent/anchors_tmp/hy_cells.json")
