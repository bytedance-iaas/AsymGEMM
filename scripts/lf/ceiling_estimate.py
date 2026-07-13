#!/usr/bin/env python3
"""Print the ceiling/latency/throughput table for a list of ceiling-search configs.

Usage:
  python3 scripts/lf/ceiling_table.py <config-list>          # print table + write scripts/lf/ceiling_table.md
  python3 scripts/lf/ceiling_table.py <config-list> -v       # + fit diagnostics
  python3 scripts/lf/ceiling_table.py <config-list> --md PATH   # write the markdown elsewhere
  python3 scripts/lf/ceiling_table.py <config-list> --no-md     # stdout only
  python3 scripts/lf/ceiling_table.py <config-list> --state-dir PATH  # read results from a
      specific ceiling_search_state_<profiler>_<host> dir (default: the sole
      state dir holding a results.jsonl; errors if none or several)

<config-list> is either:
  - a text file with rows in the ceiling_search_{source,both}.sh CONFIGS format
      seq0 : ohbm0 : model|gpus ; backend|recompute|liger ; {seq}|batch|ga ; flags[ : extra-json]
    ('#' comments and blank lines ignored), or
  - ceiling_search_source.sh / ceiling_search_both.sh itself (the active rows
    of its CONFIGS array are parsed).

Row sort order: model, backend, config.
Cell format: maxB / sec-per-step (tok/s) at maxB.
  'a' suffix = directly measured steady-state anchor: the confirm run's
  steady step seconds recorded on the results row (warmup, first and last
  measured steps dropped -- see ceiling_search.py confirm_metrics). Other
  latencies come from the per-config anchor + the analytic attention split
      t_step(B, s) = t0 + c_g * T * (1 + k*s),   T = B*s,
      k = (2*L*h) / (2*P_active)   (attention-vs-GEMM FLOPs per seq-token)
  '.' = not derivable yet (no ceiling / no confirm metrics recorded).
  'x' = slot above max seq (memory at B=1) or model context.
"""
import json, re, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD = Path(__file__).resolve().parent / "ceiling_table.md"
STATE = None  # resolved in main(): --state-dir, else the sole ceiling_search_state_*/ with results

SLOTS = [8000, 12000, 16000, 32000, 48000, 64000, 128000]
# model shorthand -> (layers, hidden, active matmul params) for the attention split
ARCH = {
    "llama3.3-70b": (80, 8192, 70.6e9),
    "llama3_3-70b": (80, 8192, 70.6e9),
    "q3-32b": (64, 5120, 32.8e9),
    "q3-30b-a3b": (48, 2048, 3.34e9),
    "q2.5-32b": (64, 5120, 32.8e9),
    "q2.5-72b": (80, 8192, 72.7e9),
    "llama4-scout": (48, 5120, 17e9),
}
CTX = {  # model context caps for the 'x' rule (adjust when rope config changes)
    "llama3.3-70b": 131072,
}


def parse_config_rows(path: Path):
    """Yield dicts {model, backend, recomp_base, batch, name} from a row list or .sh."""
    text = path.read_text()
    if path.suffix == ".sh":
        m = re.search(r"CONFIGS=\((.*?)\n\)", text, re.S)
        if not m:
            sys.exit(f"error: no CONFIGS=( ... ) array found in {path}")
        lines = [l for l in m.group(1).splitlines()]
        rows = [re.match(r'\s*"(.+)"\s*(#.*)?$', l) for l in lines]
        rows = [r.group(1) for r in rows if r]
    else:
        rows = [l.strip() for l in text.splitlines()
                if l.strip() and not l.strip().startswith("#")]
    for row in rows:
        parts = row.split(" : ")
        if len(parts) < 3:
            sys.exit(f"error: bad config row (need 'seq0 : ohbm0 : template'): {row!r}")
        template = parts[2]
        f = template.split(" ; ")
        model = f[0].split("|")[0].strip()
        backend = f[1].split("|")[0].strip()
        recomp = f[1].split("|")[1].strip()
        recomp_base = re.sub(r"-ohbm\{ohbm\}$", "", recomp)
        batch = int(f[2].split("|")[1])
        name = f"{model}__{backend}__{recomp_base}".replace("/", "_")
        yield {"model": model, "backend": backend, "recomp": recomp_base,
               "batch": batch, "name": name}


def load_results():
    """name -> full results.jsonl row (dict) — last confirmed line wins.
    FAILED rows (null ceiling_seq, written by aborted searches) are skipped:
    they carry nothing the table can use and must not shadow real rows."""
    out = {}
    f = STATE / "results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ceiling_seq") is None:
                continue
            if r.get("confirmed") or r["name"] not in out:
                out[r["name"]] = r
    return out


def t0_from_lat(leaf):
    f = leaf / "lat.md"
    if f.exists():
        for line in f.read_text().splitlines():
            if "optimizer/update side" in line:
                try:
                    return float(line.split("|")[2].strip()) / 1000.0
                except (IndexError, ValueError):
                    pass
    return 10.0


def find_anchor(res):
    """(seq, ohbm, sec, t0) from the confirm metrics recorded on the results
    row: `confirm_steady_step_s` is the anchor step time and `artifact_dir`
    points at the confirm leaf (its lat.md supplies t0)."""
    if not res or not res.get("confirm_steady_step_s") or not res.get("artifact_dir"):
        return None
    return (res["ceiling_seq"], res["ohbm"], res["confirm_steady_step_s"],
            t0_from_lat(ROOT / res["artifact_dir"]))


def main():
    global STATE
    argv = sys.argv[1:]
    verbose = any(a in ("-v", "--verbose") for a in argv)
    md_path = None if "--no-md" in argv else DEFAULT_MD
    if "--md" in argv:
        md_path = Path(argv[argv.index("--md") + 1])
    if "--state-dir" in argv:
        STATE = Path(argv[argv.index("--state-dir") + 1])
    else:
        # auto-pick the sole ceiling_search_state_<profiler>_<host> with results
        cands = sorted((ROOT / "scripts" / "lf").glob("ceiling_search_state_*/results.jsonl"))
        if len(cands) == 1:
            STATE = cands[0].parent
        elif not cands:
            sys.exit("error: no ceiling_search_state_*/results.jsonl found; pass --state-dir")
        else:
            sys.exit("error: multiple state dirs have results.jsonl; pass --state-dir:\n  "
                     + "\n  ".join(str(c.parent) for c in cands))
    args = [a for i, a in enumerate(argv)
            if not a.startswith("-")
            and (i == 0 or argv[i - 1] not in ("--md", "--state-dir"))]
    if not args:
        sys.exit(__doc__.strip())
    cfgs = list(parse_config_rows(Path(args[0])))
    results = load_results()

    fit_notes = []
    rows, trows = [], []
    for c in sorted(cfgs, key=lambda c: (c["model"], c["backend"], c["recomp"])):
        res = results.get(c["name"])
        anchor = find_anchor(res)
        arch = ARCH.get(c["model"])
        fit = None
        if anchor and arch:
            aseq, aohbm, asec, t0 = anchor
            L, H, P = arch
            k = (2 * L * H) / (2 * P)
            cg = (asec - t0) / ((c["batch"] * aseq) * (1 + k * aseq))
            fit = (t0, cg, k)
            fit_notes.append(f"`{c['name']}`: anchor s={aseq:,} ohbm{aohbm} {asec:.1f}s/step, "
                             f"t0={t0:.1f}s, c_g={cg:.4e} s/tok, k={k:.3e} "
                             f"(attn@anchor={k*aseq/(1+k*aseq)*100:.0f}%)")
            if verbose:
                print(f"# fit {fit_notes[-1]}")
        config_label = c["recomp"] + (f"-ohbm{res['ohbm']}" if res else "")
        cells, tcells = [], []
        for s in SLOTS:
            if res:
                tmax = c["batch"] * res["ceiling_seq"]
                if s > min(tmax, CTX.get(c["model"], 10 ** 9)):
                    cells.append("x"); tcells.append("x")
                    continue
                maxb = tmax // s
                if fit:
                    t0, cg, k = fit
                    T = maxb * s
                    t = t0 + cg * T * (1 + k * s)
                    tag = "a" if (anchor and anchor[0] == s and maxb == c["batch"]) else ""
                    if tag:
                        t = anchor[2]
                    cells.append(f"{maxb} / {t:,.0f}s ({T / t:,.0f}){tag}")
                    tcells.append(f"{T / t:,.0f}{tag}")
                else:
                    cells.append(f"{maxb} / . (.)"); tcells.append(".")
            else:
                cells.append(". / . (.)"); tcells.append(".")
        note = "" if (res and res.get("confirmed")) else ("  [unconfirmed]" if res else "  [no ceiling yet]")
        rows.append([c["model"], c["backend"], config_label + note] + cells)
        trows.append([c["model"], c["backend"], config_label + note] + tcells)

    hdr = ["model", "backend", "config"] + [f"{s // 1000}k" for s in SLOTS]
    w = [max(len(r[i]) for r in rows + [hdr]) for i in range(len(hdr))]
    print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * w[i] for i in range(len(hdr))))
    for r in rows:
        print(" | ".join(cell.ljust(w[i]) for i, cell in enumerate(r)))
    print("\ncell = maxB / sec-per-step (tok/s) at maxB;  a = measured anchor;  "
          ". = pending confirm metrics;  x = above max seq or context")

    print("\nTHROUGHPUT (tok/s) at fixed sequence length (at maxB; ~batch-independent, t0 amortized)")
    tw = [max(len(r[i]) for r in trows + [hdr]) for i in range(len(hdr))]
    print(" | ".join(h.ljust(tw[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * tw[i] for i in range(len(hdr))))
    for r in trows:
        print(" | ".join(cell.ljust(tw[i]) for i, cell in enumerate(r)))

    if md_path:
        lines = [
            "# Ceiling Table",
            "",
            f"Generated by `scripts/lf/ceiling_table.py {args[0]}` on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.",
            "",
            "Cell = `maxB / sec-per-step (tok/s)` at maxB. `a` = measured steady-state anchor: "
            "the confirm run's steady step (warmup, first and last measured steps dropped), "
            "as recorded in results.jsonl. Other latencies = "
            "anchor + analytic attention split `t = t0 + c_g·T·(1 + k·s)`, `T = B·s`. "
            "`.` = pending confirm metrics. `x` = above max seq (B=1) or model context.",
            "",
            "| " + " | ".join(hdr) + " |",
            "|" + "|".join("---" for _ in hdr) + "|",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        lines += ["", "## Throughput (tok/s) at fixed sequence length", "",
                  "At maxB; ~batch-independent since t0 is negligible. `a` = measured anchor.", "",
                  "| " + " | ".join(hdr) + " |",
                  "|" + "|".join("---" for _ in hdr) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in trows]
        if fit_notes:
            lines += ["", "## Fit constants", ""] + [f"- {n}" for n in fit_notes]
        lines += ["", f"Sources: everything from `{STATE.name}/results.jsonl` "
                      "(ceilings + recorded confirm metrics; each row's `artifact_dir` "
                      "points at the confirm artifacts). Re-run the command above to "
                      "refresh after new runs land.", ""]
        md_path.write_text("\n".join(lines))
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
