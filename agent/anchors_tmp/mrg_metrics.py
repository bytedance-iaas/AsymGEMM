#!/usr/bin/env python3
"""Merge-validation metrics extractor (merge_progress.md Phase 3).

Usage: mrg_metrics.py <run_glob_under_B> [tokens_per_step]
Prints steady tok/s (mean of measured steps), peak reserved HBM, peak RSS, losses —
the previous_validation_results.md protocol fields.
"""
import glob
import json
import statistics
import sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
pat = sys.argv[1]
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dirs = sorted(glob.glob(f"{B}/{pat}/*/b*_s*_ga*"))
if not dirs:
    sys.exit(f"no run dirs match {B}/{pat}")
d = dirs[-1]
rows = json.load(open(d + "/step_samples.json"))
rows = rows if isinstance(rows, list) else rows.get("steps", [])
meas = [r for r in rows if not r.get("is_warmup")]
ms = statistics.mean([r["step_milliseconds"] for r in meas]) if meas else float("nan")
peak = max(
    (r.get("backward_global_peak_reserved_after_bytes")
     or r.get("training_step_global_peak_reserved_after_bytes") or 0)
    for r in rows
)
rss = max(r.get("process_rss_peak_bytes", 0) for r in rows)
print("run_dir:", d)
print(f"measured_steps={len(meas)} mean_step_s={ms/1000:.1f}")
if tokens:
    print(f"steady_tok_s={tokens/(ms/1000):.0f}")
print(f"peak_reserved_gib={peak/2**30:.1f}")
print(f"peak_step_rss_gib={rss/2**30:.0f}")
try:
    prof = json.load(open(d + "/profile.json"))
    for k in sorted(prof):
        if "rss" in k.lower() and isinstance(prof[k], (int, float)):
            v = prof[k]
            print(f"profile.{k}={round(v/2**30,1) if v > 1e9 else v}")
except FileNotFoundError:
    print("profile.json: missing")
tl = glob.glob(d.rsplit("/", 1)[0] + "/*/train.log") + glob.glob(d + "/train.log")
if tl:
    import re
    txt = open(tl[-1], errors="ignore").read()
    losses = re.findall(r"'loss': ([0-9.]+)", txt)
    print("losses:", losses[:6])
