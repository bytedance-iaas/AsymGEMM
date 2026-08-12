#!/usr/bin/env python3
"""Harvest one gpt-oss campaign cell: global tok/s, peak resv HBM, peak RSS,
per-step losses, wrap counter, ep_sep stats. Usage: gptoss_harvest.py TAG [ranks]
Prints one TSV line: tag tok/s resv_gib rss_gb losses wrapped epsep"""
import csv, glob, json, os, re, sys

TAG = sys.argv[1]
RANKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BASE = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"

run_dirs = sorted(glob.glob(f"{BASE}/{TAG}-c17_gpt-oss-20b__b*_s*_ga1_drop000"))
if not run_dirs:
    print(f"{TAG}\tNO_RUN_DIR")
    sys.exit(0)
rd = run_dirs[-1]
m = re.search(r"__b(\d+)_s(\d+)_ga1", rd)
b, s = int(m.group(1)), int(m.group(2))
leaves = [p for p in glob.glob(f"{rd}/*/b*_s*_ga1") if os.path.isdir(p)]
if not leaves:
    print(f"{TAG}\tNO_LEAF")
    sys.exit(0)
leaf = max(leaves, key=os.path.getmtime)

tok = resv = rss = None
try:
    rows = list(csv.DictReader(open(os.path.join(leaf, "step_samples.csv"))))
    meas = [float(r["step_milliseconds"]) for r in rows
            if r.get("is_warmup", "").lower() not in ("true", "1") and float(r["step_milliseconds"]) > 0]
    if meas:
        tok = RANKS * len(meas) * b * s / (sum(meas) / 1000.0)
except Exception:
    pass
try:
    prof = json.load(open(os.path.join(leaf, "profile.json")))
    resv = prof["memory"]["peak_reserved_hbm_bytes"] / 2**30
    rss = prof["memory"]["process"]["rss_peak_bytes"] / 2**30
except Exception:
    pass

losses, wrapped, epsep = [], None, ""
tl = os.path.join(leaf, "train.log")
if os.path.exists(tl):
    txt = open(tl, errors="replace").read()
    losses = re.findall(r"\{'loss': '([\d.]+)'", txt)
    wm = re.findall(r"gptoss_moes_wrapped=(\d+)", txt)
    wrapped = wm[-1] if wm else None
    em = re.findall(r"\[ep_sep\][^\n]*exit stats: (\{[^}]*\})", txt)
    epsep = em[-1] if em else ""

print(f"{TAG}\tb{b}_s{s}\ttok/s={'' if tok is None else round(tok)}"
      f"\tresv={'' if resv is None else round(resv,1)}G"
      f"\trss={'' if rss is None else round(rss)}G"
      f"\tloss={','.join(losses[:4])}\twrapped={wrapped}\tepsep={epsep}")
