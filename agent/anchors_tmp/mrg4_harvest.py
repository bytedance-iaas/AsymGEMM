#!/usr/bin/env python3
"""Harvest mrg4 regression cells -> eff tok/s (house rule: post-warmup w1+m2),
peak reserved HBM, RSS, and mean measured loss; compare to anchors."""
import csv, glob, json, os, re

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
ANCHORS = {"mrg4q30": 1336, "mrg4hy": 929, "mrg4gf": 563}
for cfg in sorted(glob.glob(f"{B}/mrg4*-c17_*")):
    tag = os.path.basename(cfg).split("-c17_")[0]
    for ss in glob.glob(f"{cfg}/**/step_samples.csv", recursive=True):
        rd = os.path.dirname(ss)
        m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
        if not m:
            continue
        b, s, ga = map(int, m.groups())
        rows = list(csv.DictReader(open(ss)))
        meas = [float(r["step_milliseconds"]) for r in rows
                if r.get("is_warmup", "").lower() not in ("true", "1")
                and float(r.get("step_milliseconds") or 0) > 0]
        if not meas:
            continue
        eff = len(meas) * b * s * ga / (sum(meas) / 1000.0)
        losses = [float(r["loss"]) for r in rows if r.get("loss") not in (None, "", "nan")]
        resv = rss = 0.0
        pj = os.path.join(rd, "profile.json")
        if os.path.exists(pj):
            try:
                mem = json.load(open(pj)).get("memory", {})
                resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                rss = mem.get("process", {}).get("rss_peak_bytes", 0) / GIB
            except Exception:
                pass
        anchor = ANCHORS.get(tag)
        delta = f"{(eff / anchor - 1) * 100:+.1f}%" if anchor else "n/a"
        print(f"{tag:10s} {os.path.basename(os.path.dirname(rd)):48s} s={s} b{b} "
              f"eff={eff:7.0f} anchor={anchor} delta={delta} resv={resv:6.1f}G rss={rss:4.0f}G "
              f"losses={[round(x, 4) for x in losses[:4]]}")
