#!/usr/bin/env python3
"""Harvest one sepplan cell (or all): GLOBAL eff tok/s (2 ranks, post-warmup
w1+m2 house rule), winning batch, peak resv HBM, RSS; delta vs banked sdp2."""
import csv, glob, json, os, re, sys

B = "profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
GIB = 2**30
ANCHORS = {
    ("mixtral-8x22b", 32000): 4439, ("mixtral-8x22b", 64000): 3823,
    ("mixtral-8x22b", 128000): 2513, ("mixtral-8x22b", 192000): 1987,
    ("mixtral-8x22b", 256000): 1635, ("mixtral-8x22b", 288000): 1129,
    ("mixtral-8x22b", 304000): 1110,
    ("glm4.7-flash", 32000): 7400, ("glm4.7-flash", 64000): 4362,
    ("glm4.7-flash", 96000): 3086, ("glm4.7-flash", 128000): 2386,
    ("glm4.7-flash", 160000): 1934, ("glm4.7-flash", 192000): 1526,
    ("glm4.7-flash", 256000): 1152, ("glm4.7-flash", 320000): 984,
    ("glm4.7-flash", 416000): 721, ("glm4.7-flash", 512000): 617,
    ("glm4.7-flash", 576000): 548, ("glm4.7-flash", 640000): 493,
    ("glm4.7-flash", 704000): 448, ("glm4.7-flash", 768000): 405,
    ("glm4.7-flash", 832000): 371, ("glm4.7-flash", 896000): 340,
    ("glm4.7-flash", 960000): 313, ("glm4.7-flash", 1024000): 294,
    ("glm4.5-air", 16000): 6844, ("glm4.5-air", 32000): 5646,
    ("glm4.5-air", 48000): 4730, ("glm4.5-air", 64000): 4156,
    ("glm4.5-air", 96000): 3302, ("glm4.5-air", 128000): 2162,
    ("glm4.5-air", 160000): 1686, ("glm4.5-air", 192000): 1573,
    ("glm4.5-air", 256000): 1233, ("glm4.5-air", 320000): 989,
}
MODEL_BY_DIR = {"mixtral-8x22b": "mixtral-8x22b", "glm4_7-flash": "glm4.7-flash", "glm4_5-air": "glm4.5-air"}

want = sys.argv[1] if len(sys.argv) > 1 else "sp"
rows = []
for cfg in sorted(glob.glob(f"{B}/{want}*-c17_*")):
    base = os.path.basename(cfg)
    tag = base.split("-c17_")[0]
    dmodel = base.split("-c17_")[1].split("__")[0]
    model = MODEL_BY_DIR.get(dmodel, dmodel)
    for ss in glob.glob(f"{cfg}/**/step_samples.csv", recursive=True):
        rd = os.path.dirname(ss)
        m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(rd))
        if not m:
            continue
        b, s, ga = map(int, m.groups())
        meas = [float(r["step_milliseconds"]) for r in csv.DictReader(open(ss))
                if r.get("is_warmup", "").lower() not in ("true", "1")
                and float(r.get("step_milliseconds") or 0) > 0]
        if not meas:
            continue
        eff = 2 * len(meas) * b * s * ga / (sum(meas) / 1000.0)
        resv = rss = 0.0
        pj = os.path.join(rd, "profile.json")
        if os.path.exists(pj):
            try:
                mem = json.load(open(pj)).get("memory", {})
                resv = mem.get("peak_reserved_hbm_bytes", 0) / GIB
                rss = mem.get("process", {}).get("rss_peak_bytes", 0) / GIB
            except Exception:
                pass
        a = ANCHORS.get((model, s))
        d = f"{(eff/a-1)*100:+.1f}%" if a else "n/a"
        rows.append((tag, model, s, b, round(eff), a, d, round(resv, 1), round(rss)))
for r in sorted(rows, key=lambda x: (x[1], x[2])):
    print(f"{r[0]:10s} {r[1]:14s} s={r[2]:>7} b{r[3]:<3} sepplan={r[4]:>5} sdp2={r[5]} delta={r[6]:>7} resv={r[7]:>6}G rss={r[8]:>4}G")
