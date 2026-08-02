#!/usr/bin/env python3
"""Harvest tp-figure campaign cells -> effective tok/s + memory, per house rules.

Effective tok/s = (measured_steps * b * s * ga) / sum(step_ms)/1000, post-warmup
only (mirrors plot_lf_throughput.py). Prints one line per run dir + a JSON dump.
"""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
rows = []
pats = sys.argv[1:] or ["mx*", "ph*", "at*", "dv*", "vl*", "smk*"]
seen = set()
for pat in pats:
    for d in sorted(glob.glob(f"{B}/{pat}")):
        tag = os.path.basename(d)
        if not ("mixtral" in tag or "phi3_5" in tag or "glm4_7" in tag or "glm4_5" in tag) or tag in seen:
            continue
        seen.add(tag)
        for ss in glob.glob(f"{d}/**/step_samples.csv", recursive=True):
            rd = os.path.dirname(ss)
            m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
            if not m:
                continue
            b, s, ga = map(int, m.groups())
            tot_ms, n = 0.0, 0
            for row in csv.DictReader(open(ss)):
                if str(row.get("is_warmup", "")).strip().lower() in {"true", "1", "yes"}:
                    continue
                ms = float(row.get("step_milliseconds") or 0)
                if ms > 0:
                    tot_ms += ms; n += 1
            if n == 0:
                continue
            eff = (n * b * s * ga) / (tot_ms / 1000.0)
            resv = rss = 0.0
            pj = os.path.join(rd, "profile.json")
            if os.path.exists(pj):
                try:
                    prof = json.load(open(pj))
                    mem = prof.get("memory", {})
                    resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                    rss = mem.get("process", {}).get("rss_peak_bytes", 0) / GIB
                except Exception:
                    pass
            cfg = os.path.basename(os.path.dirname(rd))
            parts = cfg.split("__")
            backend, tok = parts[0], parts[2]
            model = "mixtral" if "mixtral" in tag else "phi"
            rows.append(dict(tag=tag, model=model, backend=backend, token=tok,
                             seq=s, batch=b, eff_tps=round(eff), resv_gib=round(resv, 1),
                             rss_gb=round(rss)))
rows.sort(key=lambda r: (r["model"], r["token"], r["seq"], r["batch"]))
w = max((len(r["tag"]) for r in rows), default=20)
for r in rows:
    print(f"{r['tag']:{w}} {r['backend'][:16]:16} {r['token'][:30]:30} s={r['seq']:>7} b={r['batch']} "
          f"eff={r['eff_tps']:>6} resv={r['resv_gib']:>6} rss={r['rss_gb']:>4}")
json.dump(rows, open("agent/anchors_tmp/tpfig_cells.json", "w"), indent=1)
print(f"[{len(rows)} cells] -> agent/anchors_tmp/tpfig_cells.json")
