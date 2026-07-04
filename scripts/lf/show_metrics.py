#!/usr/bin/env python3
"""Show LF source-profile timing + memory metrics as one table per model/LoRA config.

Each row reports:
    Workload | Backend | Config |
    fwd_s | bwd_s | opt_s | step_s |          # seconds
    fwd_H | bwd_H | step_H |                   # GPU HBM peak GiB
    RAM                                        # host RSS peak GiB (whole step)

Timing columns are explicit per-step averages from source_profile.json
step_samples.rows. Warmup rows are identified only by boolean row["is_warmup"].
The first and final measured rows are dropped when at least three measured rows
remain. step_s is always computed as fwd_s + bwd_s + opt_s from those same
averages. The script does not fall back to alternate fields; completed profiles
missing required raw fields are treated as malformed and abort the report.

The Config column carries the Qwen3 MoE route tag plus compact "[lg± sd±]" usage
for liger-loss / sdpa-recompute (+ on, - off). Numbers are printed as x.x
(1 decimal) to keep the table narrow. Configs that did not produce a profile
(OOM / failed / not run) show a status marker in the first timing column.

Usage:
    show_metrics.py [PROFILING_DIR]      # default profiling_both
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

GIB = 1024 ** 3
REPO = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"combined", "metrics", "memory_combined", "c2c_combined", "utilization_combined", "throughput_combined"}
OOM_GPU_MARKERS = ("CUDA out of memory", "OutOfMemoryError")
OOM_HOST_MARKERS = (
    "HOST_OOM_EVIDENCE=true",
    "Out of memory: Killed process",
    "Memory cgroup out of memory",
    "cgroup out of memory",
)
CONFIG_ORDER = [
    "none (no offload)", "none+expert-offload", "none+exp+attn-offload",
    "none+exp+attn-offload+layerGC", "GC-experts", "GC-attn+experts", "GC-layer",
]


class MetricError(RuntimeError):
    pass


def config_label(policy: str, expact: bool, attnact: bool, layeract: bool, layergc: bool, sdparecomp: bool = False) -> str:
    sdpa = "+SDPArecomp" if sdparecomp else ""
    gc_map = {"gc-exp": "GC-experts", "gc-attn-exp": "GC-attn+experts", "gc-layer": "GC-layer"}
    if policy in gc_map:
        return gc_map[policy] + sdpa
    if policy and policy != "none":
        return policy + sdpa
    offs = [n for n, on in (("exp", expact), ("attn", attnact), ("layer", layeract)) if on]
    if not offs:
        label = "none (no offload)"
    elif offs == ["exp"]:
        label = "none+expert-offload"
    else:
        label = "none+" + "+".join(offs) + "-offload"
    if layergc:
        label += "+layerGC" if offs else " (layerGC)"
    return label + sdpa


def _tok(tokens: list[str], prefix: str, default: str = "") -> str:
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix):]
    return default


def route_token(tokens: list[str], default: str = "route_missing") -> str:
    for token in tokens:
        if token.startswith("route") and not token.startswith("router"):
            return token
    return default


def _failure_codes(text: str) -> list[int]:
    codes: list[int] = []
    for pattern in (
        r"Training command failed with status\s+(-?\d+)",
        r"failed with status\s+(-?\d+)",
        r"exitcode\s*[:=]\s*(-?\d+)",
    ):
        for match in re.findall(pattern, text):
            try:
                codes.append(int(match))
            except ValueError:
                pass
    return codes


def _signal_status(code: int) -> str | None:
    sig = None
    if code >= 128:
        sig = code - 128
    elif code < 0:
        sig = -code
    if sig is None:
        return None
    try:
        name = signal.Signals(sig).name
    except ValueError:
        name = f"SIG{sig}"
    return f"{name}/{code}"


def classify(leaf: Path | None) -> tuple[str, float]:
    if leaf is None:
        return "NOT RUN", 0.0
    if (leaf / "profile.json").exists():
        return "OK", (leaf / "profile.json").stat().st_mtime
    log = leaf / "train.log"
    if not log.exists():
        return "NOT RUN", 0.0
    mtime = log.stat().st_mtime
    try:
        size = log.stat().st_size
        with open(log, "r", errors="ignore") as handle:
            if size > 2_000_000:
                handle.seek(size - 2_000_000)
            text = handle.read()
    except OSError:
        text = ""
    if any(marker in text for marker in OOM_GPU_MARKERS):
        return "OOM (GPU)", mtime
    if any(marker in text for marker in OOM_HOST_MARKERS):
        return "OOM (host RAM)", mtime
    for code in reversed(_failure_codes(text)):
        signal_status = _signal_status(code)
        if signal_status is not None:
            return signal_status, mtime
    if any(marker in text for marker in ("Training command failed", "Traceback", "ChildFailedError", "failed with status")):
        return "FAILED (non-OOM)", mtime
    if time.time() - mtime < 600:
        return "RUNNING", mtime
    return "INCOMPLETE", mtime


def _require(condition: bool, leaf: Path, message: str) -> None:
    if not condition:
        raise MetricError(f"{leaf}: {message}")


def _load_step_samples(leaf: Path, sp: dict) -> list:
    step_samples = sp.get("step_samples")
    _require(isinstance(step_samples, dict), leaf, "source_profile.json missing object step_samples")
    rows = step_samples.get("rows")
    _require(isinstance(rows, list), leaf, "source_profile.json missing list step_samples.rows")
    _require(bool(rows), leaf, "source_profile.json step_samples.rows is empty")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), leaf, f"step_samples.rows[{index}] is not an object")
        _require(isinstance(row.get("is_warmup"), bool), leaf, f"step_samples.rows[{index}].is_warmup is not boolean")
    return rows


def _measured_samples(samples: list, leaf: Path) -> list:
    rows = [r for r in samples if r["is_warmup"] is False]
    _require(bool(rows), leaf, "step_samples.rows has no non-warmup rows")
    return rows


def _timing_average_samples(samples: list, leaf: Path) -> list:
    rows = _measured_samples(samples, leaf)
    return rows[1:-1] if len(rows) > 2 else rows


def read_metrics(leaf: Path) -> dict | None:
    sp_path = leaf / "source_profile.json"
    if not sp_path.exists():
        return None
    try:
        sp = json.loads(sp_path.read_text())
    except Exception as exc:
        raise MetricError(f"{leaf}: failed to parse source_profile.json: {exc}") from exc
    _require(isinstance(sp, dict), leaf, "source_profile.json root is not an object")
    samples = _load_step_samples(leaf, sp)
    timing_samples = _timing_average_samples(samples, leaf)
    peak_samples = _measured_samples(samples, leaf)

    def values(rows, key):
        vals = []
        for index, row in enumerate(rows):
            value = row.get(key)
            _require(isinstance(value, (int, float)), leaf, f"required numeric field {key!r} missing from selected row {index}")
            vals.append(value)
        return vals

    def mean(rows, key):
        vals = values(rows, key)
        return sum(vals) / len(vals)

    def mx(rows, key):
        return max(values(rows, key))

    fwd_ms = mean(timing_samples, "forward_milliseconds")
    bwd_ms = mean(timing_samples, "backward_milliseconds")
    opt_ms = mean(timing_samples, "optimizer_milliseconds")
    step_ms = fwd_ms + bwd_ms + opt_ms
    sec = lambda ms: None if ms is None else ms / 1000.0
    gib = lambda b: None if b is None else b / GIB
    return {
        "fwd_s": sec(fwd_ms), "bwd_s": sec(bwd_ms), "opt_s": sec(opt_ms), "step_s": sec(step_ms),
        "fwd_g": gib(mx(peak_samples, "forward_peak_allocated_bytes")),
        "bwd_g": gib(mx(peak_samples, "backward_peak_allocated_bytes")),
        "step_g": gib(mx(peak_samples, "peak_allocated_hbm_bytes")),
        "ram_g": gib(mx(peak_samples, "process_rss_peak_bytes")),
    }


def _dropout_label_to_value(label: str) -> str:
    if re.fullmatch(r"drop[0-9]{3}", label):
        return f"0.{label[-2:]}"
    return label.removeprefix("drop")


def lora_label(config_root_name: str) -> str:
    match = re.search(r"(?:^|_)r(?P<rank>\d+)_a(?P<alpha>\d+)_(?P<drop>drop[0-9A-Za-z.]+)(?:_|$)", config_root_name)
    if not match:
        raise MetricError(f"{config_root_name}: cannot parse LoRA params from config-root name")
    return f"r{match.group('rank')}/a{match.group('alpha')}/d{_dropout_label_to_value(match.group('drop'))}"


def collect_leaves(root: Path) -> dict:
    """Return {logical_key: leaf_path} for the run with the most max_steps (ties: newest)."""
    runs: dict[tuple, dict] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cr in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            model = cr.name.split("__gpus")[0]
            m = re.search(r"b(\d+)_s(\d+)_ga(\d+)", cr.name)
            if not model or not m:
                continue
            lora = lora_label(cr.name)
            batch, seq, ga = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ws = re.search(r"_w\d+_s(\d+)", cr.name)
            steps = int(ws.group(1)) if ws else 0
            for rd in sorted(p for p in cr.iterdir() if p.is_dir()):
                toks = rd.name.split("__")
                if len(toks) < 3 or toks[1] not in ("nsys", "source"):
                    continue
                config = config_label(
                    _tok(toks, "pol"),
                    _tok(toks, "expact", "0") == "1",
                    _tok(toks, "attnact", "0") == "1",
                    _tok(toks, "layeract", "0") == "1",
                    _tok(toks, "layergc", "0") == "1",
                )
                config = config.replace("layer-offload", "layerOF")  # compact label
                # Compact liger / sdpa-recompute usage tag (replaces config_label's long
                # "+SDPArecomp"): [lg± sd±], + on / - off.
                liger_on = _tok(toks, "ligerloss", "0") == "1"
                sdpa_on = _tok(toks, "sdparecomp", "0") == "1"
                route = route_token(toks)
                config += f"  {route} [lg{'+' if liger_on else '-'} sd{'+' if sdpa_on else '-'}]"
                ohbm_n = _tok(toks, "ohbm", "0")
                if ohbm_n != "0":
                    config += f" ohbm{ohbm_n}"
                leaf = next((p for p in rd.iterdir() if p.is_dir() and p.name.startswith("b")), None)
                if leaf is None:
                    continue
                sp = leaf / "source_profile.json"
                has = sp.exists()
                anchor = sp if has else (leaf / "train.log")
                mtime = anchor.stat().st_mtime if anchor.exists() else 0.0
                lkey = (model, lora, seq, batch, ga, f"{toks[0]} ({toks[2]})", config)
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


HEAD = ["Workload", "Backend", "Config",
        "fwd_s", "bwd_s", "opt_s", "step_s",
        "fwd_H", "bwd_H", "step_H", "RAM"]
NUM_KEYS = ["fwd_s", "bwd_s", "opt_s", "step_s", "fwd_g", "bwd_g", "step_g", "ram_g"]
# Start from the shared semantic order, then slot the layer-offload family (renamed
# "layerOF") right after its exp+attn+layerGC sibling so the "none+exp+attn..." configs
# stay adjacent instead of the unranked layerOF falling to the end of the backend group.
_CFG_ORDER = list(CONFIG_ORDER)
_anchor = "none+exp+attn-offload+layerGC"
if _anchor in _CFG_ORDER:
    _CFG_ORDER.insert(_CFG_ORDER.index(_anchor) + 1, "none+exp+attn+layerOF")
CFG_RANK = {c: i for i, c in enumerate(_CFG_ORDER)}
MARKER = {  # shown (in the first metric column) for configs with no metrics
    "OOM (GPU)": "🔴", "OOM (host RAM)": "🟠", "FAILED (non-OOM)": "⚠️",
    "RUNNING": "🔵", "INCOMPLETE": "·", "NOT RUN": "—",
}


def base_config_label(config: str) -> str:
    return config.split("  route", 1)[0].split("  [", 1)[0]


def status_marker(status: str) -> str:
    if status.startswith("SIGTERM"):
        return "TERM"
    if status.startswith("SIGKILL"):
        return "KILL"
    if status.startswith("SIG"):
        return "SIG"
    return MARKER.get(status, "—")


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show LF profiling timing/memory metrics, one table per model.")
    ap.add_argument("root", nargs="?", default="profiling_both",
                    help="profiling output dir (default: profiling_both); relative to repo root or absolute")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / args.root
    if not root.is_dir():
        sys.exit(f"show_metrics: not a directory: {root}")

    leaves = collect_leaves(root)
    if not leaves:
        print(f"No runs found under {root}")
        return

    by_group: dict[tuple[str, str], list] = {}
    for (model, lora, seq, batch, ga, be, config), leaf in leaves.items():
        metrics = read_metrics(leaf) if leaf is not None else None
        wl = f"s{seq}·b{batch}" + (f"·ga{ga}" if ga != 1 else "")
        if metrics is not None:
            nums = [fmt(metrics.get(k)) for k in NUM_KEYS]
        else:
            status = classify(leaf)[0] if leaf is not None else "NOT RUN"
            nums = [status_marker(status)] + [""] * (len(NUM_KEYS) - 1)
        cells = [wl, be, config] + nums
        by_group.setdefault((model, lora), []).append((seq, batch, be, config, cells))

    print(f"Profiling metrics: {root}")
    print("Legend: 🔴 GPU OOM   🟠 explicit host OOM   TERM SIGTERM/143   KILL SIGKILL/137"
          "   ⚠️ failed   🔵 running   — not run")
    print("Cols: _s seconds, avg over non-warmup raw steps excluding first/final measured steps"
          " (strict fields: forward_milliseconds, backward_milliseconds, optimizer_milliseconds;"
          " step_s = fwd_s + bwd_s + opt_s)"
          " | _H GPU HBM peak GiB | RAM host RSS peak GiB (whole step)"
          " | Config routeXYZ_loraN_accTYPE [lg± sd±] = routed kernels / liger / sdpa-recompute\n")
    for model, lora in sorted(by_group):
        recs = sorted(by_group[(model, lora)], key=lambda r: (r[0], r[1], r[2], CFG_RANK.get(base_config_label(r[3]), 99), r[3]))
        data = [r[4] for r in recs]
        w = [max(len(HEAD[i]), max((len(d[i]) for d in data), default=0)) for i in range(len(HEAD))]
        just = lambda i, s: s.ljust(w[i]) if i < 3 else s.rjust(w[i])  # text left, numbers right
        group_width = lambda start, end: sum(w[start:end]) + 2 * max(0, end - start - 1)
        separator = [
            *("-" * w[i] for i in range(3)),
            "-" * group_width(3, 7),
            "-" * group_width(7, 10),
            "-" * w[10],
        ]
        workload_separator = [
            *("=" * w[i] for i in range(3)),
            "=" * group_width(3, 7),
            "=" * group_width(7, 10),
            "=" * w[10],
        ]
        print(f"Model: {model}    LoRA: {lora}")
        print("  ".join(just(i, HEAD[i]) for i in range(len(HEAD))))
        print("  ".join(separator))
        prev_wl = None
        for d in data:
            if prev_wl is not None and d[0] != prev_wl:  # heavier rule between workloads
                print("  ".join(workload_separator))
            print("  ".join(just(i, d[i]) for i in range(len(HEAD))))
            prev_wl = d[0]
        print()


if __name__ == "__main__":
    try:
        main()
    except MetricError as exc:
        sys.exit(f"show_metrics: {exc}")
