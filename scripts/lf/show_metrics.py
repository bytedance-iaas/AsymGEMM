#!/usr/bin/env python3
"""Show LF source-profile timing + memory metrics as one table per model.

Same layout as show_status.py, but each row reports the measured numbers:
    Model | Workload | Backend | Config |
    fwd_s | bwd_s | opt_s | step_s |          # seconds
    fwd_H | bwd_H | step_H |                   # GPU HBM peak GiB
    RAM                                        # host RSS peak GiB (whole step)

Timing columns are explicit per-step averages from source-profile raw step
samples when the needed raw field exists. Warmup rows are identified by
row["is_warmup"]. The first and final measured rows are dropped when at least
three measured rows remain. Older profiles without raw optimizer rows fall back
to the source-stage optimizer mean. Peak-memory columns still use max over
non-warmup rows.

The Config column carries a compact "[lg± sd±]" tag for liger-loss / sdpa-recompute
usage (+ on, - off). Numbers are printed as x.x (1 decimal) to keep the table narrow.
Configs that did not produce a profile (OOM / failed / not run) show a status marker.

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


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"1", "true", "yes", "y"}:
            return True
        if lower in {"0", "false", "no", "n"}:
            return False
    return None


def _load_step_samples(leaf: Path, sp: dict) -> list:
    ssf = leaf / "step_samples.json"
    if ssf.exists():
        try:
            d = json.loads(ssf.read_text())
            if isinstance(d, dict):
                return d.get("rows", [])
            if isinstance(d, list):
                return d
        except Exception:
            pass
    ss = sp.get("step_samples")
    if isinstance(ss, dict):
        return ss.get("rows", [])
    return ss if isinstance(ss, list) else []


def _measured_samples(samples: list) -> list:
    rows = [r for r in samples if isinstance(r, dict) and _as_bool(r.get("is_warmup")) is False]
    return rows or samples


def _timing_average_samples(samples: list) -> list:
    rows = _measured_samples(samples)
    return rows[1:-1] if len(rows) > 2 else rows


def read_metrics(leaf: Path) -> dict | None:
    sp_path = leaf / "source_profile.json"
    if not sp_path.exists():
        return None
    try:
        sp = json.loads(sp_path.read_text())
    except Exception:
        return None
    step_rows = {r.get("name"): r.get("milliseconds") for r in (sp.get("step") or {}).get("rows", [])}
    samples = _load_step_samples(leaf, sp)
    timing_samples = _timing_average_samples(samples)
    peak_samples = _measured_samples(samples)

    def mean(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    def mx(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return max(vals) if vals else None

    fwd_ms = mean(timing_samples, "forward_milliseconds") or (sp.get("forward") or {}).get("total_milliseconds")
    bwd_ms = mean(timing_samples, "backward_milliseconds") or (sp.get("backward") or {}).get("total_milliseconds")
    opt_ms = (
        mean(timing_samples, "optimizer_milliseconds")
        or mean(timing_samples, "heartbeat_optimizer_step_milliseconds")
        or mean(timing_samples, "optimizer_step_milliseconds")
        or step_rows.get("lf.optimizer.step")
    )
    step_ms = mean(timing_samples, "step_milliseconds") or step_rows.get("lf.step.total")
    sec = lambda ms: None if ms is None else ms / 1000.0
    gib = lambda b: None if b is None else b / GIB
    return {
        "fwd_s": sec(fwd_ms), "bwd_s": sec(bwd_ms), "opt_s": sec(opt_ms), "step_s": sec(step_ms),
        "fwd_g": gib(mx(peak_samples, "forward_peak_allocated_bytes")),
        "bwd_g": gib(mx(peak_samples, "backward_peak_allocated_bytes")),
        "step_g": gib(mx(peak_samples, "peak_allocated_hbm_bytes")),
        # Host RAM high-water mark for the whole step (process RSS). RSS is monotonic
        # across stages, so one step-level peak captures it; per-stage peaks would be
        # near-identical. Falls back to the per-stage backward peak for older profiles.
        "ram_g": gib(mx(peak_samples, "process_rss_peak_bytes")
                     or mx(peak_samples, "training_step_process_rss_peak_end_bytes")
                     or mx(peak_samples, "backward_process_rss_peak_end_bytes")),
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
                config = config.replace("layer-offload", "layerOF")  # compact label
                # Compact liger / sdpa-recompute usage tag (replaces config_label's long
                # "+SDPArecomp"): [lg± sd±], + on / - off.
                liger_on = S._tok(toks, "ligerloss", "0") == "1"
                sdpa_on = S._tok(toks, "sdparecomp", "0") == "1"
                config += f"  [lg{'+' if liger_on else '-'} sd{'+' if sdpa_on else '-'}]"
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
        "fwd_s", "bwd_s", "opt_s", "step_s",
        "fwd_H", "bwd_H", "step_H", "RAM"]
NUM_KEYS = ["fwd_s", "bwd_s", "opt_s", "step_s", "fwd_g", "bwd_g", "step_g", "ram_g"]
# Start from the shared semantic order, then slot the layer-offload family (renamed
# "layerOF") right after its exp+attn+layerGC sibling so the "none+exp+attn..." configs
# stay adjacent instead of the unranked layerOF falling to the end of the backend group.
_CFG_ORDER = list(S.CONFIG_ORDER)
_anchor = "none+exp+attn-offload+layerGC"
if _anchor in _CFG_ORDER:
    _CFG_ORDER.insert(_CFG_ORDER.index(_anchor) + 1, "none+exp+attn+layerOF")
CFG_RANK = {c: i for i, c in enumerate(_CFG_ORDER)}
MARKER = {  # shown (in the first metric column) for configs with no metrics
    "OOM (GPU)": "🔴", "OOM (host RAM)": "🟠", "FAILED (non-OOM)": "⚠️",
    "RUNNING": "🔵", "INCOMPLETE": "·", "NOT RUN": "—",
}


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


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
    print("Legend: 🔴 OOM (GPU)   🟠 OOM (host RAM)   ⚠️ failed   🔵 running   — not run")
    print("Cols: _s seconds, avg over non-warmup raw steps excluding first/final measured steps"
          " (opt_s falls back to source-stage mean for older profiles without raw optimizer rows)"
          " | _H GPU HBM peak GiB | RAM host RSS peak GiB (whole step)"
          " | Config [lg± sd±] = liger / sdpa-recompute on(+)/off(-)\n")
    for model in sorted(by_model):
        # Rank on the base config label (strip the trailing "  [lg± sd±]" usage tag).
        recs = sorted(by_model[model], key=lambda r: (r[0], r[1], r[2], CFG_RANK.get(r[3].split("  [")[0], 99), r[3]))
        data = [r[4] for r in recs]
        w = [max(len(HEAD[i]), max((len(d[i]) for d in data), default=0)) for i in range(len(HEAD))]
        just = lambda i, s: s.ljust(w[i]) if i < 4 else s.rjust(w[i])  # text left, numbers right
        group_width = lambda start, end: sum(w[start:end]) + 2 * max(0, end - start - 1)
        separator = [
            *("-" * w[i] for i in range(4)),
            "-" * group_width(4, 8),
            "-" * group_width(8, 11),
            "-" * w[11],
        ]
        print("  ".join(just(i, HEAD[i]) for i in range(len(HEAD))))
        print("  ".join(separator))
        prev_wl = None
        for d in data:
            if prev_wl is not None and d[1] != prev_wl:  # heavier rule between workloads
                print("  ".join("=" * w[i] for i in range(len(HEAD))))
            print("  ".join(just(i, d[i]) for i in range(len(HEAD))))
            prev_wl = d[1]
        print()


if __name__ == "__main__":
    main()
