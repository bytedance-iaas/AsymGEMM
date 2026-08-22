import csv, glob, json, os, re, sys
model = sys.argv[1]
roots = ["history/sft", "history/sft38", "history/sft39", "history/sft46", "live"]
rows = []
for r in roots:
    base = f"{r}/profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16"
    if not os.path.isdir(base): continue
    for d in glob.glob(f"{base}/*_{model}__b*_s*_ga1_drop000"):
        tsv = os.path.join(d, "jobs.tsv")
        try:
            with open(tsv) as f:
                rd = list(csv.DictReader(f, delimiter="\t"))[0]
        except Exception: continue
        gpu = rd.get("gpu",""); ranks = 2 if "," in gpu else 1
        for ss in glob.glob(f"{d}/*/b*_s*_ga*/step_samples.csv"):
            jd = os.path.dirname(ss)
            m = re.match(r"b(\d+)_s(\d+)_ga(\d+)", os.path.basename(jd))
            if not m: continue
            b, s_, ga = map(int, m.groups())
            srows = list(csv.DictReader(open(ss)))
            meas = [float(x["step_milliseconds"]) for x in srows if x.get("is_warmup","").lower() not in ("true","1") and float(x.get("step_milliseconds") or 0)>0]
            if len(meas) < 2: continue
            pj = os.path.join(jd, "profile.json")
            resv = rss = 0
            try:
                mem = json.load(open(pj)).get("memory", {})
                resv = mem.get("peak_reserved_hbm_bytes",0)/2**30
                rss = mem.get("process",{}).get("rss_peak_bytes",0)/2**30
            except Exception: pass
            sysname = os.path.basename(os.path.dirname(jd)).split("__")[0]
            tok = os.path.basename(os.path.dirname(jd)).split("__")[2]
            eff = ranks*len(meas)*b*s_*ga/(sum(meas)/1000)
            rows.append((ranks, sysname, tok, s_, b, round(eff), round(resv,1), round(rss), os.path.basename(d).split("_"+model)[0], r.split("/")[-1]))
best = {}
for row in rows:
    key = (row[0], row[1], row[2], row[3])
    if key not in best or row[5] > best[key][5]:
        best[key] = row
for row in sorted(best.values()):
    ranks, sysname, tok, s_, b, eff, resv, rss, tag, tree = row
    mark = "REUSE<=92G" if 0 < resv <= 92 else ("no-resv" if resv == 0 else "REMEASURE")
    print(f"r{ranks} {sysname[:24]:24s} {tok[:26]:26s} s={s_//1000:>5}k b{b:<2} eff={eff:>6} resv={resv:>6.1f} rss={rss:>4} {mark:10s} [{tag} {tree}]")
