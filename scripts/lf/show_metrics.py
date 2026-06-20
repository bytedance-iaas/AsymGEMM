#!/usr/bin/env python3
"""Show LF profiling timing + memory metrics as one table per model.

Same layout as show_status.py, but each row reports the measured numbers:
    Model | Workload | Backend | Config |
    forward (s) | backward (s) | optimizer (s) | step (s) |
    forward (GiB) | backward (GiB) | step (GiB)

Times are seconds (x.xxx); memory is GiB peak-allocated (x.xxx). Configs that did
not produce a profile (OOM / failed / not run) show "-".

Usage:
    show_metrics.py [PROFILING_DIR]      # default profiling_both
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import show_status as S  # noqa: E402  (shared parsing: config_label, _tok, REPO, SKIP_DIRS)

GIB = 1024 ** 3


def _load_samples(leaf: Path, sp: dict) -> list:
    ssf = leaf / "step_samples.json"
    if ssf.exists():
        try:
            d = json.loads(ssf.read_text())
            return d if isinstance(d, list) else d.get("rows", [])
        except Exception:
            pass
    ss = sp.get("step_samples")
    if isinstance(ss, dict):
        return ss.get("rows", [])
    return ss if isinstance(ss, list) else []


def read_metrics(leaf: Path) -> dict | None:
    sp_path = leaf / "source_profile.json"
    if not sp_path.exists():
        return None
    try:
        sp = json.loads(sp_path.read_text())
    except Exception:
        return None
    step_rows = {r.get("name"): r.get("milliseconds") for r in (sp.get("step") or {}).get("rows", [])}
    samples = _load_samples(leaf, sp)
    meas = [r for r in samples if not r.get("is_warmup")] or samples

    def mean(key):
        vals = [r[key] for r in meas if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    def mx(key):
        vals = [r[key] for r in meas if isinstance(r.get(key), (int, float))]
        return max(vals) if vals else None

    fwd_ms = (sp.get("forward") or {}).get("total_milliseconds")
    bwd_ms = (sp.get("backward") or {}).get("total_milliseconds")
    opt_ms = step_rows.get("lf.optimizer.step")
    step_ms = mean("step_milliseconds") or step_rows.get("lf.step.total")
    sec = lambda ms: None if ms is None else ms / 1000.0
    gib = lambda b: None if b is None else b / GIB
    return {
        "fwd_s": sec(fwd_ms), "bwd_s": sec(bwd_ms), "opt_s": sec(opt_ms), "step_s": sec(step_ms),
        "fwd_g": gib(mx("forward_peak_allocated_bytes")),
        "bwd_g": gib(mx("backward_peak_allocated_bytes")),
        "step_g": gib(mx("peak_allocated_hbm_bytes")),
    }


def collect_leaves(root: Path) -> dict:
    """Return {logical_key: leaf_path} for the run with the most max_steps (ties: newest)."""
    runs: dict[tuple, dict] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cr in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name not in S.SKIP_DIRS):
            model = cr.name.split("__gpus")[0]
            m = re.search(r"b(\d+)_s(\d+)_ga(\d+)", cr.name)
            if not model or not m:
                continue
            batch, seq, ga = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ws = re.search(r"_w\d+_s(\d+)", cr.name)
            steps = int(ws.group(1)) if ws else 0
            for rd in sorted(p for p in cr.iterdir() if p.is_dir()):
                toks = rd.name.split("__")
                if len(toks) < 3 or toks[1] not in ("nsys", "source"):
                    continue
                config = S.config_label(
                    S._tok(toks, "pol"),
                    S._tok(toks, "expact", "0") == "1",
                    S._tok(toks, "attnact", "0") == "1",
                    S._tok(toks, "layeract", "0") == "1",
                    S._tok(toks, "layergc", "0") == "1",
                )
                leaf = next((p for p in rd.iterdir() if p.is_dir() and p.name.startswith("b")), None)
                if leaf is None:
                    continue
                sp = leaf / "source_profile.json"
                has = sp.exists()
                anchor = sp if has else (leaf / "train.log")
                mtime = anchor.stat().st_mtime if anchor.exists() else 0.0
                lkey = (model, seq, batch, ga, f"{toks[0]} ({toks[2]})", config)
                run = runs.get((lkey, cr.name))
                if run is None:
                    runs[(lkey, cr.name)] = {"lkey": lkey, "steps": steps, "has": has, "mtime": mtime, "leaf": leaf}
                elif (has, mtime) > (run["has"], run["mtime"]):
                    run.update(has=has, mtime=mtime, leaf=leaf)
    chosen: dict[tuple, tuple] = {}
    out: dict[tuple, Path] = {}
    for run in runs.values():
        lk = run["lkey"]
        key = (run["steps"], run["mtime"])
        if lk not in chosen or key > chosen[lk]:
            chosen[lk] = key
            out[lk] = run["leaf"]
    return out


HEAD = ["Model", "Workload", "Backend", "Config",
        "forward (s)", "backward (s)", "optimizer (s)", "step (s)",
        "forward (GiB)", "backward (GiB)", "step (GiB)"]
NUM_KEYS = ["fwd_s", "bwd_s", "opt_s", "step_s", "fwd_g", "bwd_g", "step_g"]
CFG_RANK = {c: i for i, c in enumerate(S.CONFIG_ORDER)}
MARKER = {  # shown (in the first metric column) for configs with no metrics
    "OOM (GPU)": "🔴", "OOM (host RAM)": "🟠", "FAILED (non-OOM)": "⚠️",
    "RUNNING": "🔵", "INCOMPLETE": "·", "NOT RUN": "—",
}


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show LF profiling timing/memory metrics, one table per model.")
    ap.add_argument("root", nargs="?", default="profiling_both",
                    help="profiling output dir (default: profiling_both); relative to repo root or absolute")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = S.REPO / args.root
    if not root.is_dir():
        sys.exit(f"show_metrics: not a directory: {root}")

    leaves = collect_leaves(root)
    if not leaves:
        print(f"No runs found under {root}")
        return

    by_model: dict[str, list] = {}
    for (model, seq, batch, ga, be, config), leaf in leaves.items():
        metrics = read_metrics(leaf) if leaf is not None else None
        wl = f"s{seq}·b{batch}" + (f"·ga{ga}" if ga != 1 else "")
        if metrics is not None:
            nums = [fmt(metrics.get(k)) for k in NUM_KEYS]
        else:
            status = S.classify(leaf)[0] if leaf is not None else "NOT RUN"
            nums = [MARKER.get(status, "—")] + [""] * (len(NUM_KEYS) - 1)
        cells = [model, wl, be, config] + nums
        by_model.setdefault(model, []).append((seq, batch, be, config, cells))

    print(f"Profiling metrics: {root}")
    print("Legend: 🔴 OOM (GPU)   🟠 OOM (host RAM)   ⚠️ failed   🔵 running   — not run\n")
    for model in sorted(by_model):
        recs = sorted(by_model[model], key=lambda r: (r[0], r[1], r[2], CFG_RANK.get(r[3], 99), r[3]))
        data = [r[4] for r in recs]
        w = [max(len(HEAD[i]), max((len(d[i]) for d in data), default=0)) for i in range(len(HEAD))]
        just = lambda i, s: s.ljust(w[i]) if i < 4 else s.rjust(w[i])  # text left, numbers right
        print("  ".join(just(i, HEAD[i]) for i in range(len(HEAD))))
        print("  ".join("-" * w[i] for i in range(len(HEAD))))
        for d in data:
            print("  ".join(just(i, d[i]) for i in range(len(HEAD))))
        print()


if __name__ == "__main__":
    main()
