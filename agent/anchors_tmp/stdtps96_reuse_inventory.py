#!/usr/bin/env python3
"""Reuse inventory for the 96G campaign (standardize_tps_96gb.md rule 1):
scan ALL banked profiling run dirs, extract (model, seq, batch, ranks-ish,
backend/tier, peak resv GiB, eff tok/s), and list cells with resv <= 92 GiB
— verbatim-reusable under the 96G budget. Run from repo root.

Usage: stdtps96_reuse_inventory.py [model-substring ...]
"""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
want = sys.argv[1:] or ["glm", "hunyuan", "mixtral", "qwen3", "gpt-oss"]
rows = []
for d in sorted(glob.glob(f"{B}/*__b*_s*_ga*")):
    base = os.path.basename(d)
    if not any(w in base.lower() for w in want):
        continue
    m = re.search(r"__b(\d+)_s(\d+)_ga(\d+)", base)
    if not m:
        continue
    b, s, ga = map(int, m.groups())
    for rd in {os.path.dirname(ss) for ss in glob.glob(f"{d}/**/step_samples.csv", recursive=True)}:
        spf = os.path.join(rd, "source_profile.json")
        if not os.path.exists(spf):
            spf = os.path.join(rd, "source_profile.partial.json")
            if not os.path.exists(spf):
                continue
        try:
            J = json.load(open(spf))
        except Exception:
            continue
        mem = J.get("memory", {}) or {}
        resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
        if not resv:
            continue
        steps = []
        try:
            for row in csv.DictReader(open(os.path.join(rd, "step_samples.csv"))):
                warm = str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}
                ms = float(row.get("step_milliseconds") or 0)
                if ms > 0 and not warm:
                    steps.append(ms)
        except Exception:
            pass
        eff = (len(steps) * b * s * ga) / (sum(steps) / 1000.0) if steps else 0.0
        cfg = J.get("config", {}) or {}
        backend = cfg.get("backend") or J.get("backend") or "?"
        rows.append((base.split("__")[0], s, b, backend, round(resv, 1), round(eff)))
rows.sort(key=lambda r: (r[0], r[1], r[2]))
ok = [r for r in rows if r[4] <= 92.0]
print(f"total cells with resv: {len(rows)}; REUSABLE (resv<=92G): {len(ok)}")
for r in ok:
    print(f"  {r[0][:44]:44s} s={r[1]:<8d} b{r[2]:<3d} {str(r[3])[:28]:28s} resv={r[4]:>6.1f}G eff={r[5]}")
