#!/usr/bin/env python3
"""parse_fill_cell.py — extract the record cell (lat s/it, TP tok/s, HBM GiB/%, RSS GB)
from a tp_probe_fill run dir. Cell conventions match fix_qwen3.5_tp.md:
  lat  = lat.md "trainer e2e measured step incl optimizer" (ms -> s)
  TP   = ranks * seq * b / lat (GLOBAL tok/s at rank 2; per-invocation parse is per-rank)
  HBM  = summary.md "Whole-process peak reserved HBM" (MiB -> GiB), % of 189471 MiB
  RSS  = summary.md "RSS bytes" (-> decimal GB; per-rank at rank 2)
Usage: parse_fill_cell.py <run_dir(with jobs.tsv)|leaf_dir> <ranks> <seq> <b>
Prints one TSV line: lat_s  tp_global  hbm_gib  hbm_pct  rss_gb  spread_pct
"""
import glob
import json
import os
import re
import sys

HBM_TOTAL_MIB = 189471.0


def find_one(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True),
                  key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def main():
    root, ranks, seq, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    lat_md = find_one(root, "lat.md")
    summary = find_one(root, "summary.md")
    steps = find_one(root, "step_samples.json")
    lat_ms = None
    if lat_md:
        for line in open(lat_md):
            m = re.search(r"trainer e2e measured step incl optimizer \|\s*([0-9.]+)", line)
            if not m:
                m = re.search(r"trainer e2e measured step[^|]*\|\s*([0-9.]+)", line)
            if m:
                lat_ms = float(m.group(1))
                break
    spread = ""
    if steps:
        rows = json.load(open(steps))
        rows = rows if isinstance(rows, list) else rows.get("steps", [])
        meas = [r for r in rows if str(r.get("is_warmup", "")).lower() in ("false", "0", "")]
        durs = [r.get("total_milliseconds") or r.get("step_milliseconds")
                or (r.get("forward_milliseconds", 0) + r.get("backward_milliseconds", 0))
                for r in meas]
        durs = [d for d in durs if d]
        if durs:
            if lat_ms is None:
                lat_ms = sum(durs) / len(durs)
            if len(durs) >= 2 and min(durs) > 0:
                spread = f"{(max(durs) - min(durs)) / min(durs) * 100:.1f}"
    hbm_mib = rss_bytes = None
    if summary:
        txt = open(summary).read()
        m = re.search(r"Whole-process peak reserved HBM: `([0-9.]+) MiB`", txt)
        if m:
            hbm_mib = float(m.group(1))
        m = re.search(r"RSS bytes \|\s*([0-9]+)", txt)
        if m:
            rss_bytes = int(m.group(1))
    lat_s = lat_ms / 1000.0 if lat_ms else float("nan")
    tp = ranks * seq * b / lat_s if lat_ms else float("nan")
    hbm_gib = hbm_mib / 1024.0 if hbm_mib else float("nan")
    hbm_pct = hbm_mib / HBM_TOTAL_MIB * 100.0 if hbm_mib else float("nan")
    rss_gb = rss_bytes / 1e9 if rss_bytes else float("nan")
    print(f"{lat_s:.1f}\t{tp:.0f}\t{hbm_gib:.1f}\t{hbm_pct:.0f}\t{rss_gb:.0f}\t{spread}")


if __name__ == "__main__":
    main()
