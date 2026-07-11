#!/usr/bin/env python3
"""Extract steady-state lat / C / G from a confirm run's artifacts, per agent/RULES.md.

lat = 4 measured (non-warmup) steps, drop 1st and last, average middle 2 (seconds).
C   = peak host RSS (GiB) = max(process_rss_peak_bytes).
G   = peak reserved HBM (GiB) = max(peak_reserved_hbm_bytes).

Finds the newest step_samples.csv under OUTPUT_ROOT whose path encodes _s<seq>_ (and
optionally an ohbm / backend substring), so it works for asym and superoffload alike.
"""
import argparse, csv, glob, os, re, sys

def find_leaf(root, seq, contains):
    seqtag = f"_s{seq}_"
    cands = []
    for csvp in glob.glob(os.path.join(root, "**", "step_samples.csv"), recursive=True):
        p = csvp
        if seqtag not in p:
            continue
        if contains and not all(c in p for c in contains):
            continue
        cands.append(csvp)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="profiling_both_ceiling")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--contains", nargs="*", default=[],
                    help="extra substrings the artifact path must contain (e.g. ohbm8 asym_cpuadamwds)")
    a = ap.parse_args()

    csvp = find_leaf(a.root, a.seq, a.contains)
    if not csvp:
        print(f"NO ARTIFACT found under {a.root} for _s{a.seq}_ contains={a.contains}")
        sys.exit(3)
    leaf = os.path.dirname(csvp)
    with open(csvp) as f:
        rows = list(csv.DictReader(f))
    gib = 1024 ** 3
    hbm = [float(r["peak_reserved_hbm_bytes"]) for r in rows if r.get("peak_reserved_hbm_bytes")]
    rss = [float(r["process_rss_peak_bytes"]) for r in rows if r.get("process_rss_peak_bytes")]
    G = round(max(hbm) / gib, 1) if hbm else None
    C = round(max(rss) / gib, 1) if rss else None
    meas = sorted((r for r in rows if r.get("is_warmup", "").strip() in ("False", "false", "0")),
                  key=lambda r: int(float(r["measured_step"])))
    lat = None
    step_s = [round(float(r["step_milliseconds"]) / 1000.0, 1) for r in meas]
    if len(meas) >= 3:
        mid = [float(r["step_milliseconds"]) / 1000.0 for r in meas[1:-1]]
        lat = round(sum(mid) / len(mid), 1)

    # summary.md cross-check (best effort)
    sm_hbm = sm_rss = None
    smp = os.path.join(leaf, "summary.md")
    if os.path.exists(smp):
        txt = open(smp).read()
        m = re.search(r"[Ww]hole-process peak reserved HBM[^0-9]*([0-9.]+)", txt)
        if m: sm_hbm = m.group(1)
        m = re.search(r"RSS peak MiB[^0-9]*([0-9.]+)", txt)
        if m: sm_rss = round(float(m.group(1)) / 1024, 1)

    print(f"artifact : {leaf}")
    print(f"measured step_s (all {len(step_s)}): {step_s}  -> drop 1st+last, avg middle")
    print(f"lat = {lat} s   (steady middle-{max(len(meas)-2,0)})")
    print(f"C   = {C} GiB   (peak host RSS; summary.md RSS={sm_rss} GiB)")
    print(f"G   = {G} GiB   (peak reserved HBM; summary.md HBM={sm_hbm})")
    print(f"SUMMARY: {a.seq} :: lat={lat}s C-{C} G-{G}")

if __name__ == "__main__":
    main()
