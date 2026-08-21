#!/usr/bin/env python3
"""stdtps harvest: eff tok/s per house rule (post-warmup w1+m2, step_samples
.csv; GLOBAL = ranks x). Ranks come from the status-log START lines (never
tag heuristics — hy 2r harvest-bug lesson). Usage: stdtps_harvest.py [tag...]
(no args = all TRAINED cells in stdtps_status.log)."""
import csv, glob, json, os, re, sys

R = "/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM"
B = f"{R}/profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
ranks_of, seq_of = {}, {}
trained = []
for line in open(f"{R}/agent/anchors_tmp/stdtps_status.log"):
    m = re.match(r"START (\S+) (\S+) (\S+) s=(\d+) b=(\d+) r=(\d+)", line)
    if m:
        ranks_of[m.group(1)] = int(m.group(6)); seq_of[m.group(1)] = int(m.group(4))
    m = re.match(r"CELL (\S+) (\S+) s=(\d+) b=(\d+) -> TRAINED", line)
    if m:
        trained.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
want = set(sys.argv[1:])
seen = set()
for tag, systok, seq, b in trained:
    if want and tag not in want: continue
    if (tag, b) in seen: continue
    seen.add((tag, b))
    ranks = ranks_of.get(tag, 1)
    dirs = glob.glob(f"{B}/{tag}-c11_*__b{b}_s{seq}_ga1_drop000")
    if not dirs:
        print(f"{tag}: NO RUN DIR"); continue
    for ss in glob.glob(f"{dirs[0]}/**/step_samples.csv", recursive=True):
        rd = os.path.dirname(ss)
        meas = [float(r["step_milliseconds"]) for r in csv.DictReader(open(ss))
                if r.get("is_warmup", "").lower() not in ("true", "1")
                and float(r.get("step_milliseconds") or 0) > 0]
        if not meas: continue
        eff = ranks * len(meas) * b * seq / (sum(meas) / 1000.0)
        spread = (max(meas) - min(meas)) / max(meas) * 100 if len(meas) > 1 else 0
        resv = rss = 0.0
        pj = os.path.join(rd, "profile.json")
        if os.path.exists(pj):
            try:
                mem = json.load(open(pj)).get("memory", {})
                resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                rss = mem.get("process", {}).get("rss_peak_bytes", 0) / GIB
            except Exception: pass
        print(f"{tag:14s} r{ranks} {systok[:30]:30s} s={seq:>7} b{b} "
              f"eff={eff:7.0f} resv={resv:6.1f}G ({resv/184.9*100:4.1f}%) rss={rss:5.0f}G spread={spread:.1f}%")
